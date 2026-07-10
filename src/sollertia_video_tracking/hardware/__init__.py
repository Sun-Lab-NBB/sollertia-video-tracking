"""Provides shared device, capability, and mixed-precision detection for the training and inference optimizers."""

from .detection import (
    DEFAULT_RESERVED_CPU_THREADS,
    Toggle,
    AmpMode,
    DeviceType,
    warn,
    resolve_toggle,
    precision_label,
    supports_ampere,
    cuda_device_count,
    resolve_amp_dtype,
    apply_backend_flags,
    resolve_target_device,
)

__all__ = [
    "DEFAULT_RESERVED_CPU_THREADS",
    "AmpMode",
    "DeviceType",
    "Toggle",
    "apply_backend_flags",
    "cuda_device_count",
    "precision_label",
    "resolve_amp_dtype",
    "resolve_target_device",
    "resolve_toggle",
    "supports_ampere",
    "warn",
]
