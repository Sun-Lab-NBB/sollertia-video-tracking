"""Provides device capability detection and the resolved optimization profile that tunes DeepLabCut training."""

import os
from enum import StrEnum
from dataclasses import dataclass
import importlib.util

import torch

from ..hardware import (
    DEFAULT_RESERVED_CPU_THREADS,
    Toggle,
    AmpMode,
    DeviceType,
    warn,
    resolve_toggle,
    precision_label,
    supports_ampere,
    resolve_amp_dtype,
    apply_backend_flags,
    resolve_target_device,
)

_MIN_MULTI_GPU_COUNT: int = 2
"""The minimum number of selected GPUs required to run any multi-GPU strategy rather than a single device."""

_MAX_AUTO_DATALOADER_WORKERS: int = 8
"""The upper bound on the automatically chosen number of dataloader workers per training process."""


class MultiGpuStrategy(StrEnum):
    """Defines the multi-GPU execution strategy: automatic, DDP, DataParallel, or a single device."""

    AUTO = "auto"
    """Selects DistributedDataParallel when two or more GPUs are chosen, otherwise a single device. Request-only."""
    DDP = "ddp"
    """Trains with DistributedDataParallel, one process per GPU."""
    DP = "dp"
    """Trains with DataParallel in one process, which is slower and cannot combine with mixed precision."""
    SINGLE = "single"
    """Trains on a single device. This is the resolved outcome of selecting one GPU, not a user-selectable request."""


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
    """Determines whether a gradient scaler is required, which is the case only for float16 mixed precision."""
    tf32: bool
    """Determines whether TF32 acceleration is enabled for float32 matmuls and convolutions (CUDA only)."""
    cudnn_benchmark: bool
    """Determines whether the cuDNN convolution autotuner is enabled, which trades reproducibility for speed on
    fixed input sizes (CUDA only)."""
    torch_compile: bool
    """Determines whether the model is wrapped with ``torch.compile`` before training."""
    dataloader_workers: int
    """The number of worker processes each training process uses to load and augment data."""
    pin_memory: bool
    """Determines whether dataloaders pin host memory to speed up host-to-device transfers (meaningful for CUDA
    only)."""
    cpu_threads: int | None
    """The intra-op thread count to restore for CPU training, or None to leave the process default untouched."""

    @property
    def use_amp(self) -> bool:
        """Returns whether mixed precision is enabled for this run."""
        return self.amp_dtype is not None

    @property
    def use_ddp(self) -> bool:
        """Returns whether the run trains with DistributedDataParallel across multiple processes."""
        return self.multi_gpu_strategy == MultiGpuStrategy.DDP

    @property
    def world_size(self) -> int:
        """Returns the number of training processes, which is the GPU count under DDP and one otherwise."""
        return len(self.gpus) if self.use_ddp else 1

    @property
    def amp_device_type(self) -> str:
        """Returns the ``torch.autocast`` device-type string for this run's base device."""
        return "cuda" if self.device == "cuda" else self.device

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the active optimizations for logging.

        Returns:
            A compact description of the device, parallelism, precision, and dataloader settings.
        """
        where = f"CUDA {list(self.gpus)} ({self.multi_gpu_strategy})" if self.device == "cuda" else self.device.upper()
        precision = precision_label(self.amp_dtype)
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


def resolve_optimization_profile(
    *,
    device: DeviceType | None = None,
    gpus: tuple[int, ...] | None = None,
    multi_gpu: MultiGpuStrategy = MultiGpuStrategy.AUTO,
    amp: AmpMode = AmpMode.AUTO,
    tf32: Toggle = Toggle.AUTO,
    cudnn_benchmark: Toggle = Toggle.AUTO,
    torch_compile: Toggle = Toggle.AUTO,
    dataloader_workers: int = -1,
    pin_memory: Toggle = Toggle.AUTO,
    fixed_input_size: bool = False,
) -> OptimizationProfile:
    """Reconciles the requested optimization flags with the available hardware into a concrete profile.

    Every optimization is exposed as an explicit request so an operator who knows their silicon can override the
    automatic defaults. ``"auto"`` selects a capability-detected default suited to the chosen device. An explicit
    request is honored where it applies to the chosen device. A forced AMP dtype the device cannot support (bfloat16 on
    MPS, float16 off CUDA) is disabled with a warning, while the CUDA-only toggles (``tf32``, ``cudnn_benchmark``,
    ``pin_memory``) are forced off on non-CUDA devices. Mixed precision is additionally disabled with a warning when
    the DataParallel (``"dp"``) strategy is selected, because autocast does not reach DataParallel's per-GPU replica
    threads. Use DDP to combine mixed precision with multi-GPU training.

    Args:
        device: The requested base device (``"auto"``, ``"cpu"``, ``"mps"``, or ``"cuda"``), or None to select
            automatically.
        gpus: The explicitly requested CUDA device indices, or None to train on GPU 0. List two or more indices to
            train across multiple GPUs.
        multi_gpu: The requested multi-GPU strategy applied when two or more GPUs are selected. Resolves to a single
            device when fewer than two GPUs are used.
        amp: The requested mixed-precision mode.
        tf32: The requested TF32 setting (CUDA only, a no-op on other devices).
        cudnn_benchmark: The requested cuDNN autotuner setting. Its ``"auto"`` default follows ``fixed_input_size``,
            since the autotuner only pays off, and only stays deterministic-safe, when the input spatial size is fixed.
        torch_compile: The requested ``torch.compile`` setting. The default leaves it off because its warm-up cost
            may not amortize, and a CUDA run with no importable Triton falls back to eager execution with a warning.
        dataloader_workers: The number of dataloader workers per process, or -1 to choose automatically.
        pin_memory: The requested host-memory pinning setting (meaningful for CUDA only).
        fixed_input_size: Determines whether the training transform produces one fixed input resolution, normally
            supplied by ``detect_fixed_input_size`` rather than the operator. The cuDNN autotuner's ``"auto"`` default
            enables it only when this holds, since a varying input size makes the autotuner harmful and
            non-deterministic.

    Returns:
        The resolved ``OptimizationProfile`` describing exactly what to apply to the run.
    """
    base_device, resolved_gpus = resolve_target_device(
        device=device, gpus=gpus, role="training", default_all_gpus=False
    )
    strategy = _resolve_multi_gpu(multi_gpu=multi_gpu, gpus=resolved_gpus)
    amp_dtype = resolve_amp_dtype(amp=amp, device=base_device, gpus=resolved_gpus)
    use_gradient_scaler = amp_dtype is torch.float16
    if strategy == MultiGpuStrategy.DP and amp_dtype is not None:
        # Mixed precision cannot take effect under DataParallel: autocast state is thread-local and does not reach the
        # per-GPU replica threads DataParallel spawns, so the forward runs in float32 regardless. Disable it here so
        # the resolved profile reports the precision that actually runs and drops the then-pointless gradient scaler.
        warn(
            "Mixed precision has no effect under DataParallel (--multi-gpu dp) because autocast does not reach its "
            "per-GPU replica threads; training would run in float32. Disabling mixed precision. Use DDP (the default "
            "when two or more GPUs are selected) to combine mixed precision with multi-GPU training."
        )
        amp_dtype, use_gradient_scaler = None, False

    on_cuda = base_device == "cuda"
    resolved_tf32 = resolve_toggle(value=tf32, auto=supports_ampere(resolved_gpus)) if on_cuda else False

    resolved_benchmark = resolve_toggle(value=cudnn_benchmark, auto=fixed_input_size) if on_cuda else False
    if resolved_benchmark and not fixed_input_size:
        warn(
            "cuDNN benchmark was forced on, but the shuffle's training transform was not detected to use a single "
            "fixed input size. DeepLabCut's dynamic-resize augmentation can make this slower, and it disables "
            "deterministic training."
        )

    resolved_pin_memory = resolve_toggle(value=pin_memory, auto=on_cuda) if on_cuda else False

    # The inductor backend reaches CUDA hardware only through Triton, which torch bundles on Linux and leaves to a
    # separate distribution on Windows. Resolving the request against what is importable keeps a missing Triton out of
    # the first forward pass, where it surfaces as a compile failure partway into a run rather than at startup.
    resolved_torch_compile = resolve_toggle(value=torch_compile, auto=False)
    if resolved_torch_compile and on_cuda and importlib.util.find_spec("triton") is None:
        warn(
            "Model compilation was requested, but the torch.compile inductor backend generates its CUDA kernels "
            "through Triton, which is not importable in this environment. Disabling compilation for this run. "
            "Install the Triton distribution published for this platform to enable it."
        )
        resolved_torch_compile = False

    world_size = len(resolved_gpus) if strategy == MultiGpuStrategy.DDP else 1
    if dataloader_workers >= 0:
        workers = dataloader_workers
    elif base_device == "cpu":
        # Default CPU dataloader workers to 0: on CPU the main process performs the training compute, so worker
        # processes mostly contend for the same cores. This is the base-config value. Some shipped model configs
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
        torch_compile=resolved_torch_compile,
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
    apply_backend_flags(device=profile.device, tf32=profile.tf32, cudnn_benchmark=profile.cudnn_benchmark)
    if profile.cpu_threads is not None:
        torch.set_num_threads(profile.cpu_threads)


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


def _resolve_multi_gpu(multi_gpu: MultiGpuStrategy, gpus: tuple[int, ...]) -> MultiGpuStrategy:
    """Reconciles the requested multi-GPU strategy with the number of selected GPUs.

    Args:
        multi_gpu: The requested strategy (``"auto"``, ``"ddp"``, or ``"dp"``). Single-device training is not a request
            here. It is the resolved outcome of selecting a single GPU.
        gpus: The resolved CUDA device indices.

    Returns:
        The strategy to use. Resolves to ``"single"`` when fewer than two GPUs are selected, warning only when ``"ddp"``
        or ``"dp"`` was explicitly requested against a single GPU.
    """
    if len(gpus) < _MIN_MULTI_GPU_COUNT:
        if multi_gpu in (MultiGpuStrategy.DDP, MultiGpuStrategy.DP):
            warn(
                f"Requested '{multi_gpu}' multi-GPU training but only {len(gpus)} GPU is selected. Using a single "
                f"device. Select two or more GPUs with the GPU-indices option to train across multiple GPUs."
            )
        return MultiGpuStrategy.SINGLE
    if multi_gpu == MultiGpuStrategy.DP:
        return MultiGpuStrategy.DP
    return MultiGpuStrategy.DDP
