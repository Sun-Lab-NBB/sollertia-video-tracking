from enum import StrEnum
from dataclasses import dataclass

import torch

from ..hardware import (
    DEFAULT_RESERVED_CPU_THREADS as DEFAULT_RESERVED_CPU_THREADS,
    Toggle as Toggle,
    AmpMode as AmpMode,
    DeviceType as DeviceType,
    warn as warn,
    resolve_toggle as resolve_toggle,
    precision_label as precision_label,
    supports_ampere as supports_ampere,
    resolve_amp_dtype as resolve_amp_dtype,
    apply_backend_flags as apply_backend_flags,
    resolve_target_device as resolve_target_device,
)

class MultiGpuStrategy(StrEnum):
    AUTO = "auto"
    DDP = "ddp"
    DP = "dp"
    SINGLE = "single"

_MIN_MULTI_GPU_COUNT: int
_MAX_AUTO_DATALOADER_WORKERS: int

@dataclass(frozen=True, slots=True)
class OptimizationProfile:
    device: str
    gpus: tuple[int, ...]
    multi_gpu_strategy: MultiGpuStrategy
    amp_dtype: torch.dtype | None
    use_gradient_scaler: bool
    tf32: bool
    cudnn_benchmark: bool
    torch_compile: bool
    dataloader_workers: int
    pin_memory: bool
    cpu_threads: int | None
    @property
    def use_amp(self) -> bool: ...
    @property
    def use_ddp(self) -> bool: ...
    @property
    def world_size(self) -> int: ...
    @property
    def amp_device_type(self) -> str: ...
    def describe(self) -> str: ...

def resolve_optimization_profile(
    *,
    device: DeviceType | None = None,
    gpus: tuple[int, ...] | None = None,
    multi_gpu: MultiGpuStrategy = ...,
    amp: AmpMode = ...,
    tf32: Toggle = ...,
    cudnn_benchmark: Toggle = ...,
    torch_compile: Toggle = ...,
    dataloader_workers: int = -1,
    pin_memory: Toggle = ...,
    fixed_input_size: bool = False,
) -> OptimizationProfile: ...
def apply_runtime_optimizations(profile: OptimizationProfile) -> None: ...
def _choose_dataloader_worker_count(world_size: int) -> int: ...
def _resolve_multi_gpu(multi_gpu: MultiGpuStrategy, gpus: tuple[int, ...]) -> MultiGpuStrategy: ...
