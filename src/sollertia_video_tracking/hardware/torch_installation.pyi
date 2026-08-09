import re
from enum import StrEnum
from dataclasses import dataclass

from .detection import warn as warn

_WHEEL_INDEX_ROOT: str
_OLDEST_SUPPORTED_VARIANT: tuple[int, int]
_TORCH_PACKAGES: tuple[str, ...]
_DEFAULT_TORCH_REQUIREMENT: str
_NUMPY_REQUIREMENT: str
_SUPPORTED_TORCH_MAJOR: int
_VERIFICATION_SCRIPT: str
_VERIFICATION_LINE_COUNT: int
_NVIDIA_SMI_TIMEOUT: float
_WHEEL_INDEX_TIMEOUT: float
_VERIFICATION_TIMEOUT: float
_TORCH_VERSION_PATTERN: re.Pattern[str]
_CUDA_VERSION_PATTERN: re.Pattern[str]
_GPU_NAME_PATTERN: re.Pattern[str]
_DOTTED_CUDA_PATTERN: re.Pattern[str]
_TAGGED_CUDA_PATTERN: re.Pattern[str]
_INDEX_VARIANT_PATTERN: re.Pattern[str]

class TorchInstaller(StrEnum):
    AUTO = "auto"
    UV = "uv"
    PIP = "pip"

class TorchInstallationStatus(StrEnum):
    ENABLED = "enabled"
    UNAVAILABLE = "unavailable"
    PREVIEWED = "previewed"
    REPLACED = "replaced"

@dataclass(frozen=True, slots=True)
class TorchInstallationSummary:
    status: TorchInstallationStatus
    unavailable_reason: str
    gpu_names: tuple[str, ...]
    driver_cuda_version: str | None
    wheel_variant: str | None
    index_url: str | None
    torch_version: str | None
    torch_cuda_version: str | None
    cuda_available: bool
    replaced_packages: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]
    @property
    def gpu_count(self) -> int: ...
    def describe(self) -> str: ...

def install_cuda_torch(
    *,
    cuda_version: str | None = None,
    index_url: str | None = None,
    torch_version: str | None = None,
    installer: TorchInstaller = ...,
    execute: bool = False,
    force: bool = False,
) -> TorchInstallationSummary: ...
def _summarize_environment(
    status: TorchInstallationStatus,
    gpu_names: tuple[str, ...],
    driver_cuda: tuple[int, int] | None,
    *,
    reason: str = "",
    variant: str | None = None,
    index: str | None = None,
    packages: tuple[str, ...] = (),
    commands: tuple[tuple[str, ...], ...] = (),
) -> TorchInstallationSummary: ...
def _validate_torch_version(torch_version: str) -> None: ...
def _query_nvidia_driver() -> tuple[tuple[str, ...], tuple[int, int] | None]: ...
def _run_nvidia_smi(arguments: tuple[str, ...]) -> str | None: ...
def _extract_cuda_version(report: str) -> tuple[int, int] | None: ...
def _parse_cuda_version(requested: str) -> tuple[int, int]: ...
def _resolve_wheel_variant(cuda_version: tuple[int, int]) -> str | None: ...
def _read_published_variants() -> tuple[tuple[int, int], ...]: ...
def _format_cuda_version(cuda_version: tuple[int, int] | None) -> str | None: ...
def _is_installed(distribution_name: str) -> bool: ...
def _resolve_installer_prefixes(installer: TorchInstaller) -> tuple[tuple[str, ...], tuple[str, ...]]: ...
def _run_installer_command(command: tuple[str, ...]) -> None: ...
def _verify_installation() -> tuple[str | None, str | None, bool]: ...
