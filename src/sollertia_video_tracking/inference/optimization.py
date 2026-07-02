"""Provides device capability detection and the resolved optimization profile that tunes DeepLabCut video inference."""

import os
import sys
from typing import Literal
from dataclasses import dataclass

import torch
import psutil

Toggle = Literal["auto", "on", "off"]
"""The tri-state control for one optimization: use the capability-detected default, force it on, or force it off."""

AmpMode = Literal["auto", "off", "bf16", "fp16"]
"""The automatic-mixed-precision selection: capability-detected default, disabled, or a forced compute dtype."""

_AMPERE_CAPABILITY: tuple[int, int] = (8, 0)
"""The minimum CUDA compute capability (Ampere) that provides TF32 and native bfloat16 tensor-core acceleration."""

DEFAULT_RESERVED_CPU_THREADS: int = 2
"""The number of CPU cores held back from the automatic worker and thread budgets so other work stays responsive."""

DEFAULT_GPU_PROCESSES: int = 1
"""The default number of inference worker processes to run per CUDA device.

One process per device is the predictable default: each GPU processes one whole video at a time, which is correct for
every DeepLabCut backend and needs no per-video coordination. Because inference is bottlenecked by single-threaded
video decode, a lone process can leave the GPU underused; raising this runs several videos concurrently on one device
so one worker's decode overlaps another's forward pass. The useful factor is workload-dependent and is best found by
measurement rather than assumed."""

_DEFAULT_CPU_THREADS_PER_WORKER: int = 8
"""The default intra-op thread count for one CPU inference worker, sized to roughly one core complex (CCX/CCD).

Convolutional inference stops scaling past a modest thread count, so throughput on a many-core CPU comes from running
several bounded-thread worker processes rather than one process that owns every core."""


@dataclass(frozen=True, slots=True)
class InferenceProfile:
    """Captures the fully resolved set of hardware optimizations to apply to a video-inference run.

    Notes:
        Every field is a concrete decision: the tri-state request flags and hardware capabilities have already been
        reconciled by ``resolve_inference_profile``, so consumers apply these values directly without any further
        capability checks. Parallelism is expressed as independent worker processes that each pull whole videos from a
        shared queue: ``gpu_processes`` per CUDA device (raising it oversubscribes a device to fill decode-starved GPU
        gaps) or ``cpu_workers`` core-block-pinned processes on CPU. The profile holds cores back rather than saturating
        the machine.
    """

    device: str
    """The base torch device type inference runs on: ``"cuda"``, ``"cpu"``, or ``"mps"`` (per-worker ``cuda:N`` is
    derived from ``gpus`` inside the pipeline)."""
    gpus: tuple[int, ...]
    """The CUDA device indices in use, empty for CPU or MPS runs."""
    gpu_processes: int
    """The number of inference worker processes to run per CUDA device, or 0 when not running on CUDA."""
    cpu_workers: int
    """The number of CPU inference worker processes, or 0 when not running on CPU."""
    cpu_threads_per_worker: int | None
    """The intra-op thread count each CPU worker restores, or None when not running on CPU."""
    amp_dtype: torch.dtype | None
    """The autocast compute dtype for mixed precision, or None to run in full float32 precision."""
    tf32: bool
    """Whether TF32 acceleration is enabled for float32 matmuls and convolutions (CUDA only)."""
    cudnn_benchmark: bool
    """Whether the cuDNN convolution autotuner is enabled, which trades a warm-up for speed on fixed input sizes
    (CUDA only)."""
    channels_last: bool
    """Whether the model and its inputs use the channels-last memory format, which accelerates convolutions on
    tensor-core GPUs and oneDNN CPU backends."""
    torch_compile: bool
    """Whether the model is wrapped with ``torch.compile`` before inference."""
    pin_memory: bool
    """Whether host frames are staged in pinned memory for non-blocking host-to-device transfer (CUDA only)."""

    @property
    def use_amp(self) -> bool:
        """Returns whether mixed precision is enabled for this run."""
        return self.amp_dtype is not None

    @property
    def on_cuda(self) -> bool:
        """Returns whether inference runs on CUDA devices."""
        return self.device == "cuda"

    @property
    def amp_device_type(self) -> str:
        """Returns the device type string passed to ``torch.autocast`` for this run."""
        return "cuda" if self.device == "cuda" else self.device

    @property
    def total_workers(self) -> int:
        """Returns the total number of worker processes the run spawns across all devices."""
        if self.on_cuda:
            return len(self.gpus) * self.gpu_processes
        if self.device == "cpu":
            return self.cpu_workers
        return 1

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the active optimizations for logging.

        Returns:
            A compact description of the device, parallelism, precision, and enabled accelerators.
        """
        if self.on_cuda:
            where = f"CUDA {list(self.gpus)} x{self.gpu_processes}/gpu"
        elif self.device == "cpu":
            where = f"CPU {self.cpu_workers}x{self.cpu_threads_per_worker}t"
        else:
            where = self.device.upper()
        precision = "fp32" if self.amp_dtype is None else str(self.amp_dtype).removeprefix("torch.")
        extras = [
            name
            for name, enabled in (
                ("tf32", self.tf32),
                ("cudnn.benchmark", self.cudnn_benchmark),
                ("channels_last", self.channels_last),
                ("compile", self.torch_compile),
                ("pin", self.pin_memory),
            )
            if enabled
        ]
        suffix = f", {'+'.join(extras)}" if extras else ""
        return f"{where} | {precision} | workers={self.total_workers}{suffix}"


def get_cuda_device_count() -> int:
    """Returns the number of visible CUDA devices, or zero when CUDA is unavailable.

    Returns:
        The count of CUDA devices reported by the active PyTorch build.
    """
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def resolve_bfloat16_support(gpus: tuple[int, ...]) -> bool:
    """Determines whether every listed CUDA device natively accelerates bfloat16 (Ampere or newer).

    Args:
        gpus: The CUDA device indices to check.

    Returns:
        True when all listed devices report at least Ampere compute capability, False otherwise.
    """
    return len(gpus) > 0 and all(
        torch.cuda.get_device_capability(device=index) >= _AMPERE_CAPABILITY for index in gpus
    )


def resolve_tf32_support(gpus: tuple[int, ...]) -> bool:
    """Determines whether every listed CUDA device supports TF32 matmul and convolution acceleration (Ampere+).

    Args:
        gpus: The CUDA device indices to check.

    Returns:
        True when all listed devices report at least Ampere compute capability, False otherwise.
    """
    return len(gpus) > 0 and all(
        torch.cuda.get_device_capability(device=index) >= _AMPERE_CAPABILITY for index in gpus
    )


def resolve_inference_profile(
    *,
    device: str | None = None,
    gpus: tuple[int, ...] | None = None,
    amp: AmpMode = "auto",
    tf32: Toggle = "auto",
    cudnn_benchmark: Toggle = "auto",
    channels_last: Toggle = "auto",
    torch_compile: Toggle = "auto",
    gpu_processes: int = -1,
    cpu_workers: int = -1,
    cpu_threads_per_worker: int = -1,
    pin_memory: Toggle = "auto",
    fixed_input_size: bool = False,
) -> InferenceProfile:
    """Reconciles the requested inference optimization flags with the available hardware into a concrete profile.

    Every optimization is exposed as an explicit request so an operator who knows their silicon can override the
    automatic defaults. ``"auto"`` selects a capability-detected default suited to the chosen device. An explicit
    ``"on"``/``"off"`` (or a forced AMP dtype) is always honored, with a warning when it contradicts the detected
    hardware rather than a silent refusal. The device selection cascades ``cuda`` -> ``cpu`` when no CUDA device is
    visible so the same call works unchanged on a GPU server or a CPU-only server.

    Args:
        device: The requested device (``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``, or ``"cuda:N"``), or None to
            select automatically.
        gpus: The explicitly requested CUDA device indices, or None to use every visible device.
        amp: The requested mixed-precision mode; ``"auto"`` enables bfloat16 only where it is natively fast.
        tf32: The requested TF32 setting (CUDA only; a no-op on other devices).
        cudnn_benchmark: The requested cuDNN autotuner setting; only safe when input spatial sizes are fixed.
        channels_last: The requested channels-last memory-format setting.
        torch_compile: The requested ``torch.compile`` setting; disabled by default because of its warm-up cost, which
            may not amortize over short videos.
        gpu_processes: The number of worker processes per CUDA device, or -1 to use the default of one video per GPU.
        cpu_workers: The number of CPU worker processes, or -1 to choose automatically from the physical core count.
        cpu_threads_per_worker: The intra-op thread count per CPU worker, or -1 to choose automatically.
        pin_memory: The requested host-memory pinning setting (meaningful for CUDA only).
        fixed_input_size: Whether the inference transform produces a single fixed input resolution, which is required
            for the cuDNN autotuner to be beneficial rather than harmful.

    Returns:
        The resolved ``InferenceProfile`` describing exactly what to apply to the run.
    """
    base_device, resolved_gpus = _resolve_target_device(device=device, gpus=gpus)
    on_cuda = base_device == "cuda"

    amp_dtype = _resolve_amp(amp=amp, device=base_device, gpus=resolved_gpus)

    resolved_tf32 = _resolve_toggle(value=tf32, auto=resolve_tf32_support(resolved_gpus)) if on_cuda else False

    resolved_benchmark = _resolve_toggle(value=cudnn_benchmark, auto=False) if on_cuda else False
    if resolved_benchmark and not fixed_input_size:
        _warn(
            "cuDNN benchmark is enabled without a fixed input size. Videos of differing resolutions re-tune the "
            "autotuner per size, which can be slower than leaving it off."
        )

    # channels-last helps convolutions on tensor-core GPUs (and oneDNN on CPU) but is only turned on automatically on
    # CUDA, where the benefit is largest and most reliable; CPU users can still opt in explicitly.
    resolved_channels_last = _resolve_toggle(value=channels_last, auto=on_cuda)

    resolved_pin_memory = _resolve_toggle(value=pin_memory, auto=on_cuda) if on_cuda else False

    resolved_gpu_processes = _resolve_gpu_processes(gpu_processes=gpu_processes) if on_cuda else 0
    resolved_cpu_workers, resolved_cpu_threads = (
        _resolve_cpu_parallelism(cpu_workers=cpu_workers, cpu_threads_per_worker=cpu_threads_per_worker)
        if base_device == "cpu"
        else (0, None)
    )

    return InferenceProfile(
        device=base_device,
        gpus=resolved_gpus,
        gpu_processes=resolved_gpu_processes,
        cpu_workers=resolved_cpu_workers,
        cpu_threads_per_worker=resolved_cpu_threads,
        amp_dtype=amp_dtype,
        tf32=resolved_tf32,
        cudnn_benchmark=resolved_benchmark,
        channels_last=resolved_channels_last,
        torch_compile=_resolve_toggle(value=torch_compile, auto=False),
        pin_memory=resolved_pin_memory,
    )


def apply_runtime_optimizations(profile: InferenceProfile) -> None:
    """Applies the process-global optimization flags described by the profile inside an inference worker.

    This must run inside each worker process before inference begins. TF32 and the cuDNN autotuner are process-global
    CUDA backend flags; the CPU thread count restores intra-op parallelism for the worker (the package pins
    ``OMP_NUM_THREADS=1`` at import for the extraction workers, so CPU inference must raise it deliberately).

    Args:
        profile: The resolved inference profile whose global flags should be applied.
    """
    if profile.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = profile.tf32
        torch.backends.cudnn.allow_tf32 = profile.tf32
        if profile.tf32:
            torch.set_float32_matmul_precision("high")
        if profile.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    if profile.cpu_threads_per_worker is not None:
        torch.set_num_threads(profile.cpu_threads_per_worker)


def _warn(message: str) -> None:
    """Writes a non-fatal warning to the standard error stream.

    Args:
        message: The warning text to emit, without the ``WARNING:`` prefix or trailing newline.
    """
    sys.stderr.write(f"WARNING: {message}\n")
    sys.stderr.flush()


def _resolve_target_device(device: str | None, gpus: tuple[int, ...] | None) -> tuple[str, tuple[int, ...]]:
    """Reconciles the requested device and GPU indices with the available hardware.

    Args:
        device: The requested device (``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``, or ``"cuda:N"``), or None for
            automatic selection.
        gpus: The explicitly requested CUDA device indices, or None to select them automatically.

    Returns:
        A tuple of the resolved base device type and the tuple of CUDA indices to use.

    Raises:
        ValueError: When an explicitly requested CUDA index is not present on the machine.
    """
    request = (device or "auto").lower()
    available = get_cuda_device_count()

    if request == "cpu":
        return "cpu", ()
    if request == "mps":
        return "mps", ()

    if request.startswith("cuda") or request == "auto":
        if available == 0:
            if request != "auto":
                _warn(f"Requested device '{request}' but no CUDA device is available. Falling back to CPU.")
            return "cpu", ()
        if gpus is not None:
            for index in gpus:
                if index < 0 or index >= available:
                    message = (
                        f"Unable to select GPUs using the requested indices. Expected each index below the visible "
                        f"device count {available}, but got {index}."
                    )
                    raise ValueError(message)
            return "cuda", tuple(gpus)
        if ":" in request:
            index = int(request.split(":", 1)[1])
            if index < 0 or index >= available:
                message = (
                    f"Unable to select a GPU using device '{request}'. Expected an index below the visible device "
                    f"count {available}, but got {index}."
                )
                raise ValueError(message)
            return "cuda", (index,)
        return "cuda", tuple(range(available))

    message = (
        f"Unable to resolve the inference device. Expected 'auto', 'cpu', 'mps', 'cuda', or 'cuda:N', "
        f"but got '{request}'."
    )
    raise ValueError(message)


def _resolve_amp(amp: AmpMode, device: str, gpus: tuple[int, ...]) -> torch.dtype | None:
    """Reconciles the requested mixed-precision mode with the device and its capabilities.

    Inference has no backward pass, so float16 needs no gradient scaler; this returns only the autocast dtype.

    Args:
        amp: The requested mixed-precision mode.
        device: The resolved base device type.
        gpus: The resolved CUDA device indices.

    Returns:
        The autocast dtype to use, or None when mixed precision is disabled.
    """
    if amp == "off":
        return None
    if amp == "auto":
        # Enable bfloat16 only where it is natively fast so the automatic default stays close to stock float32. On CPU
        # the benefit is chip-dependent, so bfloat16 there is left as an explicit opt-in rather than an auto default.
        if device == "cuda" and resolve_bfloat16_support(gpus):
            return torch.bfloat16
        return None
    if amp == "bf16":
        if device == "mps":
            _warn("bfloat16 autocast is unreliable on MPS. Disabling mixed precision.")
            return None
        if device == "cuda" and not resolve_bfloat16_support(gpus):
            _warn(
                "bfloat16 was requested but the selected GPU lacks native bfloat16 support (pre-Ampere); it may run "
                "slowly. Consider '--amp fp16' instead."
            )
        return torch.bfloat16
    # The only remaining mode is float16, which is a CUDA-only inference precision.
    if device != "cuda":
        _warn(f"float16 autocast is only supported on CUDA, not '{device}'. Disabling mixed precision.")
        return None
    return torch.float16


def _resolve_gpu_processes(gpu_processes: int) -> int:
    """Resolves the number of worker processes to run per CUDA device.

    Args:
        gpu_processes: The requested per-device process count, or -1 to use the default of one video per GPU.

    Returns:
        The number of worker processes to run per CUDA device (at least one).
    """
    if gpu_processes >= 1:
        return gpu_processes
    return DEFAULT_GPU_PROCESSES


def _resolve_cpu_parallelism(cpu_workers: int, cpu_threads_per_worker: int) -> tuple[int, int]:
    """Resolves the CPU worker count and per-worker thread budget from the physical core topology.

    Throughput on a many-core CPU comes from several bounded-thread worker processes, each pinned to a disjoint core
    block, rather than one process that owns every core. This shares the usable physical cores across workers while
    holding ``DEFAULT_RESERVED_CPU_THREADS`` back for decode and other work.

    Args:
        cpu_workers: The requested worker count, or -1 to choose automatically.
        cpu_threads_per_worker: The requested per-worker intra-op thread count, or -1 to choose automatically.

    Returns:
        A tuple of the resolved worker count and per-worker thread count (each at least one).
    """
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    usable = max(1, physical - DEFAULT_RESERVED_CPU_THREADS)

    threads = cpu_threads_per_worker if cpu_threads_per_worker >= 1 else min(_DEFAULT_CPU_THREADS_PER_WORKER, usable)
    workers = cpu_workers if cpu_workers >= 1 else max(1, usable // threads)
    return workers, threads


def _resolve_toggle(value: Toggle, *, auto: bool) -> bool:
    """Resolves a tri-state toggle to a boolean, using the given capability-detected default for ``"auto"``.

    Args:
        value: The requested tri-state value.
        auto: The capability-detected default applied when the value is ``"auto"``.

    Returns:
        The resolved boolean decision.
    """
    if value == "on":
        return True
    if value == "off":
        return False
    return auto
