"""Provides device capability detection and the resolved optimization profile that tunes DeepLabCut training."""

import os
import sys
from typing import Literal
from dataclasses import dataclass

import torch

Toggle = Literal["auto", "on", "off"]
"""The tri-state control for one optimization: use the capability-detected default, force it on, or force it off."""

AmpMode = Literal["auto", "off", "bf16", "fp16"]
"""The automatic-mixed-precision selection: capability-detected default, disabled, or a forced compute dtype."""

MultiGpuStrategy = Literal["ddp", "dp", "single"]
"""The resolved multi-GPU execution strategy: DistributedDataParallel, DataParallel, or a single device."""

_AMPERE_CAPABILITY: tuple[int, int] = (8, 0)
"""The minimum CUDA compute capability (Ampere) that provides TF32 and native bfloat16 tensor-core acceleration."""

_MIN_MULTI_GPU_COUNT: int = 2
"""The minimum number of selected GPUs required to run any multi-GPU strategy rather than a single device."""

DEFAULT_RESERVED_CPU_THREADS: int = 2
"""The number of CPU cores held back from the automatic dataloader-worker and CPU-thread budgets for other work."""

_MAX_AUTO_DATALOADER_WORKERS: int = 8
"""The upper bound on the automatically chosen number of dataloader workers per training process."""


@dataclass(frozen=True, slots=True)
class OptimizationProfile:
    """Captures the fully resolved set of hardware optimizations to apply to a single training run.

    Notes:
        Every field is a concrete decision: the tri-state request flags and hardware capabilities have already been
        reconciled by ``resolve_optimization_profile``, so consumers apply these values directly without any further
        capability checks. The profile is device-aware and holds cores back for other work rather than saturating the
        machine, favoring smoother operation.
    """

    device: str
    """The base torch device type training runs on: ``"cuda"``, ``"cpu"``, or ``"mps"`` (per-rank ``cuda:N`` is
    derived from ``gpus`` inside the runner)."""
    gpus: tuple[int, ...]
    """The CUDA device indices in use, empty for CPU or MPS runs."""
    multi_gpu_strategy: MultiGpuStrategy
    """The resolved multi-GPU execution strategy."""
    amp_dtype: torch.dtype | None
    """The autocast compute dtype for mixed precision, or None to train in full float32 precision."""
    use_gradient_scaler: bool
    """Whether a gradient scaler is required, which is the case only for float16 mixed precision."""
    tf32: bool
    """Whether TF32 acceleration is enabled for float32 matmuls and convolutions (CUDA only)."""
    cudnn_benchmark: bool
    """Whether the cuDNN convolution autotuner is enabled, which trades reproducibility for speed on fixed input
    sizes (CUDA only)."""
    torch_compile: bool
    """Whether the model is wrapped with ``torch.compile`` before training."""
    dataloader_workers: int
    """The number of worker processes each training process uses to load and augment data."""
    pin_memory: bool
    """Whether dataloaders pin host memory to speed up host-to-device transfers (meaningful for CUDA only)."""
    cpu_threads: int | None
    """The intra-op thread count to restore for CPU training, or None to leave the process default untouched."""

    @property
    def use_amp(self) -> bool:
        """Returns whether mixed precision is enabled for this run."""
        return self.amp_dtype is not None

    @property
    def use_ddp(self) -> bool:
        """Returns whether the run trains with DistributedDataParallel across multiple processes."""
        return self.multi_gpu_strategy == "ddp"

    @property
    def world_size(self) -> int:
        """Returns the number of training processes, which is the GPU count under DDP and one otherwise."""
        return len(self.gpus) if self.use_ddp else 1

    @property
    def amp_device_type(self) -> str:
        """Returns the device type string passed to ``torch.autocast`` for this run."""
        return "cuda" if self.device == "cuda" else self.device

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the active optimizations for logging.

        Returns:
            A compact description of the device, parallelism, precision, and dataloader settings.
        """
        where = f"CUDA {list(self.gpus)} ({self.multi_gpu_strategy})" if self.device == "cuda" else self.device.upper()
        precision = "fp32" if self.amp_dtype is None else str(self.amp_dtype).removeprefix("torch.")
        extras = [
            name
            for name, enabled in (
                ("tf32", self.tf32),
                ("cudnn.benchmark", self.cudnn_benchmark),
                ("compile", self.torch_compile),
            )
            if enabled
        ]
        suffix = f", {'+'.join(extras)}" if extras else ""
        return f"{where} | {precision} | workers={self.dataloader_workers} pin={self.pin_memory}{suffix}"


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


def resolve_optimization_profile(
    *,
    device: str | None = None,
    gpus: tuple[int, ...] | None = None,
    multi_gpu: Literal["auto", "ddp", "dp", "single"] = "auto",
    amp: AmpMode = "auto",
    tf32: Toggle = "auto",
    cudnn_benchmark: Toggle = "auto",
    torch_compile: Toggle = "auto",
    dataloader_workers: int = -1,
    pin_memory: Toggle = "auto",
    fixed_input_size: bool = False,
) -> OptimizationProfile:
    """Reconciles the requested optimization flags with the available hardware into a concrete profile.

    Every optimization is exposed as an explicit request so an operator who knows their silicon can override the
    automatic defaults. ``"auto"`` selects a capability-detected default suited to the chosen device. An explicit
    ``"on"``/``"off"`` (or a forced AMP dtype) is always honored, with a warning when it contradicts the detected
    hardware rather than a silent refusal.

    Args:
        device: The requested device (``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``, or ``"cuda:N"``), or None to
            select automatically.
        gpus: The explicitly requested CUDA device indices, or None to use every visible device.
        multi_gpu: The requested multi-GPU strategy, downgraded to a single device when fewer than two GPUs are used.
        amp: The requested mixed-precision mode.
        tf32: The requested TF32 setting (CUDA only; a no-op on other devices).
        cudnn_benchmark: The requested cuDNN autotuner setting; only safe when input spatial sizes are fixed.
        torch_compile: The requested ``torch.compile`` setting; disabled by default because of its warm-up cost.
        dataloader_workers: The number of dataloader workers per process, or -1 to choose automatically.
        pin_memory: The requested host-memory pinning setting (meaningful for CUDA only).
        fixed_input_size: Whether the training transform produces a single fixed input resolution, which is required
            for the cuDNN autotuner to be beneficial rather than harmful.

    Returns:
        The resolved ``OptimizationProfile`` describing exactly what to apply to the run.
    """
    base_device, resolved_gpus = _resolve_target_device(device=device, gpus=gpus)
    strategy = _resolve_multi_gpu(multi_gpu=multi_gpu, gpus=resolved_gpus)
    amp_dtype, use_gradient_scaler = _resolve_amp(amp=amp, device=base_device, gpus=resolved_gpus)

    on_cuda = base_device == "cuda"
    resolved_tf32 = _resolve_toggle(value=tf32, auto=resolve_tf32_support(resolved_gpus)) if on_cuda else False

    resolved_benchmark = _resolve_toggle(value=cudnn_benchmark, auto=False) if on_cuda else False
    if resolved_benchmark and not fixed_input_size:
        _warn(
            "cuDNN benchmark is enabled without a fixed input size. DeepLabCut's dynamic-resize augmentation can "
            "make this slower, and it disables deterministic training."
        )

    resolved_pin_memory = _resolve_toggle(value=pin_memory, auto=on_cuda) if on_cuda else False

    world_size = len(resolved_gpus) if strategy == "ddp" else 1
    if dataloader_workers >= 0:
        workers = dataloader_workers
    elif base_device == "cpu":
        # Default CPU dataloader workers to 0: on CPU the main process performs the training compute, so worker
        # processes mostly contend for the same cores. This is the base-config value; some shipped model configs
        # (e.g. RTMPose) use 4, which the operator can restore with an explicit worker count.
        workers = 0
    else:
        workers = _choose_dataloader_worker_count(world_size=world_size)

    # Restore intra-op threading for CPU training (the package pins OMP_NUM_THREADS=1 for the extraction workers),
    # deliberately holding back DEFAULT_RESERVED_CPU_THREADS cores so other work stays responsive rather than
    # saturating the machine.
    cpu_threads = max(1, (os.cpu_count() or 1) - DEFAULT_RESERVED_CPU_THREADS) if base_device == "cpu" else None

    return OptimizationProfile(
        device=base_device,
        gpus=resolved_gpus,
        multi_gpu_strategy=strategy,
        amp_dtype=amp_dtype,
        use_gradient_scaler=use_gradient_scaler,
        tf32=resolved_tf32,
        cudnn_benchmark=resolved_benchmark,
        torch_compile=_resolve_toggle(value=torch_compile, auto=False),
        dataloader_workers=workers,
        pin_memory=resolved_pin_memory,
        cpu_threads=cpu_threads,
    )


def apply_runtime_optimizations(profile: OptimizationProfile) -> None:
    """Applies the process-global optimization flags described by the profile.

    This must run inside each training process and, on the CUDA path, after DeepLabCut's ``fix_seeds`` has run,
    because ``fix_seeds`` sets ``cudnn.benchmark = False`` and ``cudnn.deterministic = True`` and would otherwise
    clobber the autotuner setting applied here. TF32 flags and thread counts are independent of seeding.

    Args:
        profile: The resolved optimization profile whose global flags should be applied.
    """
    if profile.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = profile.tf32
        torch.backends.cudnn.allow_tf32 = profile.tf32
        if profile.tf32:
            torch.set_float32_matmul_precision("high")
        if profile.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    if profile.cpu_threads is not None:
        torch.set_num_threads(profile.cpu_threads)


def _warn(message: str) -> None:
    """Writes a non-fatal warning to the standard error stream.

    Args:
        message: The warning text to emit, without the ``WARNING:`` prefix or trailing newline.
    """
    sys.stderr.write(f"WARNING: {message}\n")
    sys.stderr.flush()


def _choose_dataloader_worker_count(world_size: int) -> int:
    """Chooses a dataloader-worker count per process that shares the usable CPU cores across all training ranks.

    Args:
        world_size: The number of training processes that will each spawn their own dataloader workers.

    Returns:
        The number of dataloader workers each process should use.
    """
    usable = (os.cpu_count() or 1) - DEFAULT_RESERVED_CPU_THREADS
    per_rank = usable // max(1, world_size)
    return max(0, min(_MAX_AUTO_DATALOADER_WORKERS, per_rank))


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
        f"Unable to resolve the training device. Expected 'auto', 'cpu', 'mps', 'cuda', or 'cuda:N', "
        f"but got '{request}'."
    )
    raise ValueError(message)


def _resolve_multi_gpu(multi_gpu: Literal["auto", "ddp", "dp", "single"], gpus: tuple[int, ...]) -> MultiGpuStrategy:
    """Reconciles the requested multi-GPU strategy with the number of selected GPUs.

    Args:
        multi_gpu: The requested strategy (``"auto"``, ``"ddp"``, ``"dp"``, or ``"single"``).
        gpus: The resolved CUDA device indices.

    Returns:
        The strategy to use, downgraded to ``"single"`` with a warning when fewer than two GPUs are selected.
    """
    if len(gpus) < _MIN_MULTI_GPU_COUNT:
        if multi_gpu in ("ddp", "dp"):
            _warn(
                f"Requested '{multi_gpu}' multi-GPU training but only {len(gpus)} GPU is selected. Using a single "
                f"device."
            )
        return "single"
    if multi_gpu == "dp":
        return "dp"
    if multi_gpu == "single":
        return "single"
    return "ddp"


def _resolve_amp(amp: AmpMode, device: str, gpus: tuple[int, ...]) -> tuple[torch.dtype | None, bool]:
    """Reconciles the requested mixed-precision mode with the device and its capabilities.

    Args:
        amp: The requested mixed-precision mode.
        device: The resolved base device type.
        gpus: The resolved CUDA device indices.

    Returns:
        A tuple of the autocast dtype (or None when disabled) and whether a gradient scaler is required.
    """
    if amp == "off":
        return None, False
    if amp == "auto":
        # Enable bfloat16 only where it is natively fast so the automatic default stays close to stock float32.
        if device == "cuda" and resolve_bfloat16_support(gpus):
            return torch.bfloat16, False
        return None, False
    if amp == "bf16":
        if device == "mps":
            _warn("bfloat16 autocast is unreliable on MPS. Disabling mixed precision.")
            return None, False
        if device == "cuda" and not resolve_bfloat16_support(gpus):
            _warn(
                "bfloat16 was requested but the selected GPU lacks native bfloat16 support (pre-Ampere); it may run "
                "slowly. Consider '--amp fp16' instead."
            )
        return torch.bfloat16, False
    # The only remaining mode is float16, which is CUDA-only and needs a gradient scaler to avoid underflow.
    if device != "cuda":
        _warn(f"float16 autocast is only supported on CUDA, not '{device}'. Disabling mixed precision.")
        return None, False
    return torch.float16, True


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
