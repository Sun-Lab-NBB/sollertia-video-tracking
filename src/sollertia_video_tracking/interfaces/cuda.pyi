from ..hardware import (
    TorchInstaller as TorchInstaller,
    TorchInstallationStatus as TorchInstallationStatus,
    install_cuda_torch as install_cuda_torch,
)

_CONTEXT_SETTINGS: dict[str, int]

def cuda_command(
    cuda_version: str | None,
    torch_version: str | None,
    index_url: str | None,
    installer: str,
    *,
    force: bool,
    yes: bool,
) -> None: ...
