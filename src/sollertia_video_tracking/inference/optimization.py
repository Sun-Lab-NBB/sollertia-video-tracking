"""Provides device capability detection and the resolved optimization profile that tunes DeepLabCut video inference."""

import os
from dataclasses import dataclass

import torch
import psutil

from ..hardware import (
    DEFAULT_RESERVED_CPU_THREADS,
    Toggle,
    AmpMode,
    warn,
    resolve_toggle,
    precision_label,
    supports_ampere,
    resolve_amp_dtype,
    apply_backend_flags,
    resolve_target_device,
)

_DEFAULT_GPU_PROCESSES: int = 1
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
    """Determines whether TF32 acceleration is enabled for float32 matmuls and convolutions (CUDA only)."""
    cudnn_benchmark: bool
    """Determines whether the cuDNN convolution autotuner is enabled, which trades a warm-up for speed on fixed
    input sizes (CUDA only)."""
    channels_last: bool
    """Determines whether the model and its inputs use the channels-last memory format, which accelerates
    convolutions on tensor-core GPUs and oneDNN CPU backends."""
    torch_compile: bool
    """Determines whether the model is wrapped with ``torch.compile`` before inference."""
    pin_memory: bool
    """Determines whether host frames are staged in pinned memory for non-blocking host-to-device transfer
    (CUDA only)."""

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
        """Returns the device type string to pass to ``torch.autocast`` for this run."""
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
        precision = precision_label(self.amp_dtype)
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
    ``"on"``/``"off"`` toggle is always honored; a forced AMP dtype is honored wherever the device can run it and is
    otherwise disabled with a warning rather than a silent refusal (bfloat16 on MPS and float16 on any non-CUDA device
    fall back to float32). The device selection cascades ``cuda`` -> ``cpu`` when no CUDA device is visible so the same
    call works unchanged on a GPU server or a CPU-only server.

    Args:
        device: The requested device (``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``, or ``"cuda:N"``), or None to
            select automatically.
        gpus: The explicitly requested CUDA device indices, or None to use every visible device.
        amp: The requested mixed-precision mode; ``"auto"`` enables bfloat16 only where it is natively fast.
        tf32: The requested TF32 setting (CUDA only; a no-op on other devices).
        cudnn_benchmark: The requested cuDNN autotuner setting; its ``"auto"`` default follows ``fixed_input_size``,
            since the autotuner only pays off when the input spatial size is fixed.
        channels_last: The requested channels-last memory-format setting.
        torch_compile: The requested ``torch.compile`` setting; disabled by default because of its warm-up cost, which
            may not amortize over short videos.
        gpu_processes: The number of worker processes per CUDA device, or -1 to use the default of one video per GPU.
        cpu_workers: The number of CPU worker processes, or -1 to choose automatically from the physical core count.
        cpu_threads_per_worker: The intra-op thread count per CPU worker, or -1 to choose automatically.
        pin_memory: The requested host-memory pinning setting (meaningful for CUDA only).
        fixed_input_size: Determines whether every video feeds the network one fixed input resolution, normally
            supplied by ``detect_fixed_input_size`` rather than the operator. The cuDNN autotuner's ``"auto"`` default
            enables it only when this holds, since a varying input size makes the autotuner harmful rather than
            beneficial.

    Returns:
        The resolved ``InferenceProfile`` describing exactly what to apply to the run.
    """
    base_device, resolved_gpus = resolve_target_device(device=device, gpus=gpus, role="inference")
    on_cuda = base_device == "cuda"

    amp_dtype = resolve_amp_dtype(amp=amp, device=base_device, gpus=resolved_gpus)

    resolved_tf32 = resolve_toggle(value=tf32, auto=supports_ampere(resolved_gpus)) if on_cuda else False

    resolved_benchmark = resolve_toggle(value=cudnn_benchmark, auto=fixed_input_size) if on_cuda else False
    if resolved_benchmark and not fixed_input_size:
        warn(
            "cuDNN benchmark was forced on, but the run was not detected to use a single fixed input size. Videos of "
            "differing resolutions re-tune the autotuner per size, which can be slower than leaving it off."
        )

    # channels-last helps convolutions on tensor-core GPUs (and oneDNN on CPU) but is only turned on automatically on
    # CUDA, where the benefit is largest and most reliable; CPU users can still opt in explicitly.
    resolved_channels_last = resolve_toggle(value=channels_last, auto=on_cuda)

    resolved_pin_memory = resolve_toggle(value=pin_memory, auto=on_cuda) if on_cuda else False

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
        torch_compile=resolve_toggle(value=torch_compile, auto=False),
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
    apply_backend_flags(device=profile.device, tf32=profile.tf32, cudnn_benchmark=profile.cudnn_benchmark)
    if profile.cpu_threads_per_worker is not None:
        torch.set_num_threads(profile.cpu_threads_per_worker)


def _resolve_gpu_processes(gpu_processes: int) -> int:
    """Resolves the number of worker processes to run per CUDA device.

    Args:
        gpu_processes: The requested per-device process count, or -1 to use the default of one video per GPU.

    Returns:
        The number of worker processes to run per CUDA device (at least one).
    """
    if gpu_processes >= 1:
        return gpu_processes
    return _DEFAULT_GPU_PROCESSES


def _resolve_cpu_parallelism(cpu_workers: int, cpu_threads_per_worker: int) -> tuple[int, int]:
    """Resolves the CPU worker count and per-worker thread budget from the physical core topology.

    Throughput on a many-core CPU comes from several bounded-thread worker processes, each pinned to a disjoint core
    block, rather than one process that owns every core. This shares the usable physical cores across workers while
    holding ``DEFAULT_RESERVED_CPU_THREADS`` back for decode and other work. When the worker count is given but the
    thread budget is left automatic, the per-worker thread count is derived from the worker count so the workers'
    pinned core blocks stay disjoint instead of oversubscribing the machine.

    Args:
        cpu_workers: The requested worker count, or -1 to choose automatically.
        cpu_threads_per_worker: The requested per-worker intra-op thread count, or -1 to choose automatically.

    Returns:
        A tuple of the resolved worker count and per-worker thread count (each at least one).
    """
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    usable = max(1, physical - DEFAULT_RESERVED_CPU_THREADS)

    if cpu_threads_per_worker >= 1:
        threads = cpu_threads_per_worker
    elif cpu_workers >= 1:
        # An explicit worker count with an automatic thread budget splits the usable cores across those workers, so
        # each worker's thread count matches its pinned core block rather than every worker claiming a full block.
        threads = max(1, usable // cpu_workers)
    else:
        threads = min(_DEFAULT_CPU_THREADS_PER_WORKER, usable)
    workers = cpu_workers if cpu_workers >= 1 else max(1, usable // threads)
    return workers, threads
