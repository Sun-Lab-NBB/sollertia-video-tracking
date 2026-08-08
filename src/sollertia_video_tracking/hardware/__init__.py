"""Provides the shared device and capability detection, and the CUDA-enabled PyTorch build installation."""

from .detection import (
    DEFAULT_RESERVED_CPU_THREADS,
    Toggle,
    AmpMode,
    DeviceType,
    warn,
    toggle_label,
    resolve_toggle,
    precision_label,
    supports_ampere,
    resolve_amp_dtype,
    apply_backend_flags,
    resolve_target_device,
)
from .torch_installation import (
    TorchInstaller,
    TorchInstallationStatus,
    TorchInstallationSummary,
    install_cuda_torch,
)

__all__ = [
    "DEFAULT_RESERVED_CPU_THREADS",
    "AmpMode",
    "DeviceType",
    "Toggle",
    "TorchInstallationStatus",
    "TorchInstallationSummary",
    "TorchInstaller",
    "apply_backend_flags",
    "install_cuda_torch",
    "precision_label",
    "resolve_amp_dtype",
    "resolve_target_device",
    "resolve_toggle",
    "supports_ampere",
    "toggle_label",
    "warn",
]
