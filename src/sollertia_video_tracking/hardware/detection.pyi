from enum import StrEnum

import torch

DEFAULT_RESERVED_CPU_THREADS: int
_AMPERE_CAPABILITY: tuple[int, int]

class Toggle(StrEnum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"

class AmpMode(StrEnum):
    AUTO = "auto"
    OFF = "off"
    BF16 = "bf16"
    FP16 = "fp16"

class DeviceType(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"

def warn(message: str) -> None: ...
def supports_ampere(gpus: tuple[int, ...]) -> bool: ...
def resolve_toggle(value: Toggle, *, auto: bool) -> bool: ...
def toggle_label(*, enabled: bool) -> str: ...
def resolve_target_device(
    device: str | None, gpus: tuple[int, ...] | None, *, role: str, default_all_gpus: bool = True
) -> tuple[str, tuple[int, ...]]: ...
def resolve_amp_dtype(amp: AmpMode, device: str, gpus: tuple[int, ...]) -> torch.dtype | None: ...
def apply_backend_flags(*, device: str, tf32: bool, cudnn_benchmark: bool) -> None: ...
def precision_label(amp_dtype: torch.dtype | None) -> str: ...
def _cuda_device_count() -> int: ...
