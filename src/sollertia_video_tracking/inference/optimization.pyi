from dataclasses import dataclass

import torch

from ..hardware import (
    DEFAULT_RESERVED_CPU_THREADS as DEFAULT_RESERVED_CPU_THREADS,
    Toggle as Toggle,
    AmpMode as AmpMode,
    DeviceType as DeviceType,
    warn as warn,
    toggle_label as toggle_label,
    resolve_toggle as resolve_toggle,
    precision_label as precision_label,
    supports_ampere as supports_ampere,
    resolve_amp_dtype as resolve_amp_dtype,
    apply_backend_flags as apply_backend_flags,
    resolve_target_device as resolve_target_device,
)

_DEFAULT_GPU_PROCESSES: int
_DEFAULT_CPU_THREADS_PER_WORKER: int

@dataclass(frozen=True, slots=True)
class InferenceProfile:
    device: str
    gpus: tuple[int, ...]
    gpu_processes: int
    chunks: int
    cpu_workers: int
    cpu_threads_per_worker: int | None
    amp_dtype: torch.dtype | None
    tf32: bool
    cudnn_benchmark: bool
    channels_last: bool
    torch_compile: bool
    @property
    def use_amp(self) -> bool: ...
    @property
    def on_cuda(self) -> bool: ...
    @property
    def amp_device_type(self) -> str: ...
    @property
    def total_workers(self) -> int: ...
    def describe(self) -> str: ...
    def report_rows(self) -> tuple[tuple[str, str], ...]: ...

def resolve_inference_profile(
    *,
    device: DeviceType | None = None,
    gpus: tuple[int, ...] | None = None,
    amp: AmpMode = ...,
    tf32: Toggle = ...,
    cudnn_benchmark: Toggle = ...,
    channels_last: Toggle = ...,
    torch_compile: Toggle = ...,
    gpu_processes: int = -1,
    chunks: int = 1,
    cpu_workers: int = -1,
    cpu_threads_per_worker: int = -1,
    fixed_input_size: bool = False,
) -> InferenceProfile: ...
def apply_runtime_optimizations(profile: InferenceProfile) -> None: ...
def _resolve_gpu_processes(gpu_processes: int) -> int: ...
def _resolve_cpu_parallelism(cpu_workers: int, cpu_threads_per_worker: int) -> tuple[int, int]: ...
