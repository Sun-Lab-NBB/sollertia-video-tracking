"""Provides the NVIDIA driver query and wheel-variant resolution that install a CUDA-enabled PyTorch build."""

import re
import sys
from enum import StrEnum
import shutil
from importlib import metadata
import subprocess
from dataclasses import dataclass

import torch

from .detection import warn

_WHEEL_INDEX_ROOT: str = "https://download.pytorch.org/whl"
"""The root URL of the PyTorch wheel index whose per-CUDA-version subdirectories serve the CUDA builds."""

_CUDA_WHEEL_VARIANTS: tuple[tuple[int, int], ...] = ((11, 8), (12, 1), (12, 4), (12, 6), (12, 8), (12, 9), (13, 0))
"""The CUDA versions PyTorch publishes a wheel variant for, in ascending order.

A driver runs every CUDA version at or below the one it reports, so the newest entry that does not exceed the reported
version is the variant to install. A variant missing from this tuple is reachable only through an explicit CUDA version
or wheel index, so this tuple gains an entry whenever PyTorch publishes a new variant.
"""

_TORCH_PACKAGES: tuple[str, ...] = ("torch", "torchvision", "torchaudio")
"""The torch-family distributions replaced together, since torchvision and torchaudio pin an exact torch version."""

_DEFAULT_TORCH_PACKAGES: tuple[str, ...] = ("torch", "torchvision")
"""The distributions installed when the environment holds none of the torch family, both of which DeepLabCut needs."""

_DEFAULT_TORCH_REQUIREMENT: str = "torch>=2,<3"
"""The torch requirement installed when no exact version is requested, matching the range this library supports."""

_NUMPY_REQUIREMENT: str = "numpy>=1.26,<2"
"""The numpy requirement pinned alongside torch, since DeepLabCut supports the 1.x series only."""

_SUPPORTED_TORCH_MAJOR: int = 2
"""The only torch major version this library supports, which an explicitly requested version must also carry."""

_VERIFICATION_SCRIPT: str = (
    "import torch; print(torch.__version__); print(torch.version.cuda or ''); print(int(torch.cuda.is_available()))"
)
"""Reports the newly installed build's version, CUDA version, and GPU reachability from a fresh interpreter."""

_VERIFICATION_LINE_COUNT: int = 3
"""The number of lines the verification script prints, one per reported field."""

_NVIDIA_SMI_TIMEOUT: float = 30.0
"""The seconds to wait for an nvidia-smi query before treating the driver as unreadable."""

_VERIFICATION_TIMEOUT: float = 300.0
"""The seconds to wait for the post-install verification, which pays the full torch import cost."""

_TORCH_VERSION_PATTERN: re.Pattern[str] = re.compile(r"^\d+(\.\d+)*([a-z]+\d+)?(\+[0-9a-z.]+)?$")
"""The shape an explicitly requested torch version must take, such as ``2.9.1`` or ``2.9.1+cu126``."""

_CUDA_VERSION_PATTERN: re.Pattern[str] = re.compile(r"CUDA(?: UMD)? Version\s*:\s*(\d+)\.(\d+)")
"""Matches the CUDA version in both the nvidia-smi query report and the header of its summary table."""

_GPU_NAME_PATTERN: re.Pattern[str] = re.compile(r"Product Name\s*:\s*(.+)")
"""Matches the name of each visible GPU in the nvidia-smi query report."""

_DOTTED_CUDA_PATTERN: re.Pattern[str] = re.compile(r"^(\d+)\.(\d+)$")
"""Matches a requested CUDA version written as a dotted major and minor version."""

_TAGGED_CUDA_PATTERN: re.Pattern[str] = re.compile(r"^cu(\d{2,})(\d)$")
"""Matches a requested CUDA version written as a wheel-variant tag, whose trailing digit is the minor version."""


class TorchInstaller(StrEnum):
    """Defines the package manager that carries out the torch replacement."""

    AUTO = "auto"
    """Uses uv when it is present on the system path, and pip otherwise."""
    UV = "uv"
    """Uses uv, failing when it is absent from the system path."""
    PIP = "pip"
    """Uses pip."""


class TorchInstallationStatus(StrEnum):
    """Defines the outcome of a request to install a CUDA-enabled PyTorch build."""

    ENABLED = "enabled"
    """The installed build already targets CUDA and reaches the local GPUs, so nothing was changed."""
    UNAVAILABLE = "unavailable"
    """No CUDA build applies to this machine, so nothing was changed."""
    PREVIEWED = "previewed"
    """The replacement was resolved and reported as a dry run, so nothing was changed."""
    REPLACED = "replaced"
    """The torch-family distributions were replaced with the resolved CUDA build."""


@dataclass(frozen=True, slots=True)
class TorchInstallationSummary:
    """Summarizes a CUDA-enabled PyTorch installation, whether previewed as a dry run or actually carried out.

    Notes:
        The torch fields describe the build present in the environment once the call returns, so they report the
        replacement on an executed run and the untouched original on every other outcome.
    """

    status: TorchInstallationStatus
    """The outcome of the request."""
    unavailable_reason: str
    """The reason no CUDA build applies to this machine, empty for every other outcome."""
    gpu_names: tuple[str, ...]
    """The names of the GPUs nvidia-smi reported, in the order it reported them."""
    driver_cuda_version: str | None
    """The CUDA version the local driver runs, or None when no driver answered or its report carried no version."""
    wheel_variant: str | None
    """The resolved wheel-variant tag, such as ``cu130``, or None when no variant applies."""
    index_url: str | None
    """The wheel index the build is installed from, or None when no variant applies."""
    torch_version: str | None
    """The version of the installed torch distribution, or None when the verification could not read it."""
    torch_cuda_version: str | None
    """The CUDA version the installed torch build targets, or None for a CPU-only build."""
    cuda_available: bool
    """Determines whether the installed torch build reaches a local GPU."""
    replaced_packages: tuple[str, ...]
    """The torch-family distributions the replacement covers, empty when nothing is replaced."""
    commands: tuple[tuple[str, ...], ...]
    """The uninstall and install commands the replacement runs, empty when nothing is replaced."""

    @property
    def gpu_count(self) -> int:
        """Returns the number of GPUs nvidia-smi reported."""
        return len(self.gpu_names)

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the installation outcome.

        Returns:
            A compact description of what the call resolved and what it changed.
        """
        if self.status == TorchInstallationStatus.UNAVAILABLE:
            return f"no CUDA build applies: {self.unavailable_reason}"
        if self.status == TorchInstallationStatus.ENABLED:
            return (
                f"torch {self.torch_version} already targets CUDA {self.torch_cuda_version} and reaches "
                f"{self.gpu_count} GPU(s). Pass --force to replace it anyway."
            )
        if self.status == TorchInstallationStatus.PREVIEWED:
            return (
                f"dry run: would replace {', '.join(self.replaced_packages)} with the {self.wheel_variant} build. "
                f"Re-run with --yes to apply."
            )
        if self.torch_version is None:
            return f"installed the {self.wheel_variant} build, but the verification could not read the result."
        reach = f"reaches {self.gpu_count} GPU(s)" if self.cuda_available else "reaches no GPU"
        return f"installed torch {self.torch_version}, which targets CUDA {self.torch_cuda_version} and {reach}."


def install_cuda_torch(
    *,
    cuda_version: str | None = None,
    index_url: str | None = None,
    torch_version: str | None = None,
    installer: TorchInstaller = TorchInstaller.AUTO,
    execute: bool = False,
    force: bool = False,
) -> TorchInstallationSummary:
    """Replaces the environment's torch-family distributions with the CUDA build the local NVIDIA driver runs.

    The wheel variant is the newest one PyTorch publishes at or below the CUDA version nvidia-smi reports, since a
    driver runs every CUDA version at or below the one it reports. The replacement targets the interpreter this
    library is installed into, and pins numpy to its 1.x series alongside torch because DeepLabCut supports no other
    series.

    Args:
        cuda_version: The CUDA version to resolve the wheel variant from, written as ``12.6`` or ``cu126``, or None to
            read it from nvidia-smi. A version PyTorch publishes no variant for resolves to the newest variant below
            it.
        index_url: The wheel index to install from, which bypasses the wheel-variant resolution, or None to derive the
            index from the resolved variant.
        torch_version: The exact torch version to install, or None to install the newest version the wheel index
            carries within the range this library supports. An older variant carries only older torch versions, so an
            exact version that variant never shipped fails the installation.
        installer: The package manager that carries out the replacement.
        execute: Determines whether to run the resolved commands, where a dry run reports them and changes nothing.
        force: Determines whether to replace a build that already targets CUDA and reaches the local GPUs.

    Returns:
        The summary describing the resolved build, the resolved commands, and what the call changed.

    Raises:
        ValueError: If the requested CUDA version is not a recognizable version, if the requested torch version is
            malformed, carries a non-integer segment before its first dot, or falls outside the supported major
            series, or if uv is requested while it is absent from the system path.
        RuntimeError: If an executed uninstall or install command fails.
    """
    # Validates the requested version before anything is inspected or resolved, so a version this library cannot run
    # is rejected even when the environment's current build would otherwise short-circuit the installation.
    if torch_version is not None:
        _validate_torch_version(torch_version)
    gpu_names, driver_cuda = _query_nvidia_driver()
    resolved_index = index_url
    variant: str | None = None

    if resolved_index is None:
        if sys.platform == "darwin":
            reason = (
                "PyTorch publishes no CUDA build for macOS. The standard build already carries the Metal (MPS) "
                "backend this library uses on Apple hardware."
            )
            return _summarize_environment(
                status=TorchInstallationStatus.UNAVAILABLE,
                gpu_names=gpu_names,
                driver_cuda=driver_cuda,
                reason=reason,
            )
        requested = _parse_cuda_version(cuda_version) if cuda_version is not None else driver_cuda
        if requested is None:
            reason = (
                "nvidia-smi reported no CUDA version, which means no NVIDIA driver is installed or none is visible "
                "from this environment. Install the driver, or name the CUDA version explicitly."
            )
            return _summarize_environment(
                status=TorchInstallationStatus.UNAVAILABLE,
                gpu_names=gpu_names,
                driver_cuda=driver_cuda,
                reason=reason,
            )
        variant = _resolve_wheel_variant(requested)
        if variant is None:
            oldest = _format_cuda_version(_CUDA_WHEEL_VARIANTS[0])
            reason = (
                f"CUDA {_format_cuda_version(requested)} predates CUDA {oldest}, the oldest version PyTorch still "
                f"publishes a wheel variant for. Upgrade the NVIDIA driver."
            )
            return _summarize_environment(
                status=TorchInstallationStatus.UNAVAILABLE,
                gpu_names=gpu_names,
                driver_cuda=driver_cuda,
                reason=reason,
            )
        resolved_index = f"{_WHEEL_INDEX_ROOT}/{variant}"
    else:
        variant = resolved_index.rstrip("/").rpartition("/")[2]

    if torch.version.cuda is not None and torch.cuda.is_available() and not force:
        return _summarize_environment(
            status=TorchInstallationStatus.ENABLED,
            gpu_names=gpu_names,
            driver_cuda=driver_cuda,
            variant=variant,
            index=resolved_index,
        )

    packages = tuple(name for name in _TORCH_PACKAGES if _is_installed(name)) or _DEFAULT_TORCH_PACKAGES
    torch_requirement = f"torch=={torch_version}" if torch_version is not None else _DEFAULT_TORCH_REQUIREMENT
    requirements = (torch_requirement, *(name for name in packages if name != "torch"), _NUMPY_REQUIREMENT)
    uninstall_prefix, install_prefix = _resolve_installer_prefixes(installer)
    commands = (
        (*uninstall_prefix, *packages),
        (*install_prefix, "--index-url", resolved_index, *requirements),
    )

    if not execute:
        return _summarize_environment(
            status=TorchInstallationStatus.PREVIEWED,
            gpu_names=gpu_names,
            driver_cuda=driver_cuda,
            variant=variant,
            index=resolved_index,
            packages=packages,
            commands=commands,
        )

    for command in commands:
        _run_installer_command(command)
    installed_version, installed_cuda, reachable = _verify_installation()
    return TorchInstallationSummary(
        status=TorchInstallationStatus.REPLACED,
        unavailable_reason="",
        gpu_names=gpu_names,
        driver_cuda_version=_format_cuda_version(driver_cuda),
        wheel_variant=variant,
        index_url=resolved_index,
        torch_version=installed_version,
        torch_cuda_version=installed_cuda,
        cuda_available=reachable,
        replaced_packages=packages,
        commands=commands,
    )


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
) -> TorchInstallationSummary:
    """Builds a summary whose torch fields describe the build loaded into this interpreter.

    Args:
        status: The outcome the summary reports.
        gpu_names: The names of the GPUs nvidia-smi reported.
        driver_cuda: The CUDA version the local driver runs, or None when no driver answered or its report carried no
            version.
        reason: The reason no CUDA build applies to this machine.
        variant: The resolved wheel-variant tag, or None when no variant applies.
        index: The wheel index the build would be installed from, or None when no variant applies.
        packages: The torch-family distributions the replacement covers.
        commands: The uninstall and install commands the replacement would run.

    Returns:
        The assembled summary.
    """
    return TorchInstallationSummary(
        status=status,
        unavailable_reason=reason,
        gpu_names=gpu_names,
        driver_cuda_version=_format_cuda_version(driver_cuda),
        wheel_variant=variant,
        index_url=index,
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        replaced_packages=packages,
        commands=commands,
    )


def _validate_torch_version(torch_version: str) -> None:
    """Rejects an explicitly requested torch version the pinned DeepLabCut release cannot run against.

    An explicit version bypasses the default requirement range, so a version outside the supported major series would
    otherwise uninstall the environment's torch and install a build DeepLabCut cannot import, reporting success.

    Args:
        torch_version: The exact torch version requested.

    Raises:
        ValueError: If the version is malformed, if the segment before its first dot is not a plain integer, or if it
            carries an unsupported major version.
    """
    if _TORCH_VERSION_PATTERN.match(torch_version) is None:
        message = (
            f"Unable to install the requested torch version. Expected a dotted version such as '2.9.1', but got "
            f"'{torch_version}'."
        )
        raise ValueError(message)
    major = int(torch_version.split(".", maxsplit=1)[0])
    if major != _SUPPORTED_TORCH_MAJOR:
        message = (
            f"Unable to install torch {torch_version}. This library supports the torch {_SUPPORTED_TORCH_MAJOR}.x "
            f"series only, because the pinned DeepLabCut release is built against it, but got major version {major}."
        )
        raise ValueError(message)


def _query_nvidia_driver() -> tuple[tuple[str, ...], tuple[int, int] | None]:
    """Reads the visible GPU names and the driver's CUDA version from nvidia-smi.

    Returns:
        A tuple of the reported GPU names and the CUDA version the driver runs, where the version is None when no
        driver answered or its report carried no version.
    """
    report = _run_nvidia_smi(("-q",))
    if report is None:
        return (), None
    names = tuple(match.group(1).strip() for match in _GPU_NAME_PATTERN.finditer(report))
    version = _extract_cuda_version(report)
    if version is None:
        # Driver generations that leave the CUDA version out of the query report still print it in the header of the
        # summary table, so the bare invocation is the fallback rather than an equivalent second query.
        summary = _run_nvidia_smi(())
        version = _extract_cuda_version(summary) if summary is not None else None
    return names, version


def _run_nvidia_smi(arguments: tuple[str, ...]) -> str | None:
    """Runs nvidia-smi with the given arguments and captures its standard output.

    Args:
        arguments: The arguments to pass to nvidia-smi.

    Returns:
        The captured standard output, or None when nvidia-smi is absent, fails, or does not answer in time.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - the executable is resolved from the path and takes fixed arguments.
            [executable, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=_NVIDIA_SMI_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _extract_cuda_version(report: str) -> tuple[int, int] | None:
    """Reads the CUDA version out of an nvidia-smi report.

    Args:
        report: The captured nvidia-smi output.

    Returns:
        The reported CUDA version as a major and minor pair, or None when the report carries no version.
    """
    match = _CUDA_VERSION_PATTERN.search(report)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_cuda_version(requested: str) -> tuple[int, int]:
    """Parses a caller-supplied CUDA version written as a dotted version or as a wheel-variant tag.

    Notes:
        A tag's trailing digit is read as the whole minor version, since every variant PyTorch publishes carries a
        single-digit minor.

    Args:
        requested: The requested CUDA version, for example ``12.6`` or ``cu126``.

    Returns:
        The requested CUDA version as a major and minor pair.

    Raises:
        ValueError: If the request is not a recognizable CUDA version.
    """
    normalized = requested.strip().lower()
    match = _DOTTED_CUDA_PATTERN.match(normalized) or _TAGGED_CUDA_PATTERN.match(normalized)
    if match is None:
        message = (
            f"Unable to resolve the requested CUDA version. Expected a dotted version such as '12.6' or a wheel "
            f"variant such as 'cu126', but got '{requested}'."
        )
        raise ValueError(message)
    return int(match.group(1)), int(match.group(2))


def _resolve_wheel_variant(cuda_version: tuple[int, int]) -> str | None:
    """Selects the newest published wheel variant the given CUDA version runs.

    Args:
        cuda_version: The CUDA version the driver runs, as a major and minor pair.

    Returns:
        The wheel-variant tag, such as ``cu130``, or None when the version predates every published variant.
    """
    candidates = [variant for variant in _CUDA_WHEEL_VARIANTS if variant <= cuda_version]
    if not candidates:
        return None
    major, minor = candidates[-1]
    return f"cu{major}{minor}"


def _format_cuda_version(cuda_version: tuple[int, int] | None) -> str | None:
    """Formats a CUDA version pair as a dotted version.

    Args:
        cuda_version: The CUDA version as a major and minor pair, or None.

    Returns:
        The dotted version, or None when no version was given.
    """
    if cuda_version is None:
        return None
    return f"{cuda_version[0]}.{cuda_version[1]}"


def _is_installed(distribution_name: str) -> bool:
    """Determines whether the named distribution is installed into this environment.

    Args:
        distribution_name: The name of the distribution to look up.

    Returns:
        True when the distribution's metadata resolves, False otherwise.
    """
    try:
        metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return False
    else:
        return True


def _resolve_installer_prefixes(installer: TorchInstaller) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Builds the uninstall and install command prefixes for the requested package manager.

    Both prefixes target this interpreter explicitly, so the replacement lands in the environment the library is
    installed into rather than in whichever environment happens to be active.

    Args:
        installer: The requested package manager.

    Returns:
        A tuple of the uninstall command prefix and the install command prefix.

    Raises:
        ValueError: If uv is requested while it is absent from the system path.
    """
    executable = None if installer == TorchInstaller.PIP else shutil.which("uv")
    if installer == TorchInstaller.UV and executable is None:
        message = (
            "Unable to install the CUDA-enabled PyTorch build. The 'uv' installer was requested, but no 'uv' "
            "executable is present on the system path."
        )
        raise ValueError(message)
    if executable is None:
        return (sys.executable, "-m", "pip", "uninstall", "--yes"), (sys.executable, "-m", "pip", "install")
    return (
        (executable, "pip", "uninstall", "--python", sys.executable),
        (executable, "pip", "install", "--python", sys.executable),
    )


def _run_installer_command(command: tuple[str, ...]) -> None:
    """Runs one installer command, streaming its output to this process's streams.

    Args:
        command: The command to run, as an argument vector.

    Raises:
        RuntimeError: If the command exits with a non-zero status.
    """
    sys.stderr.write(f"running: {' '.join(command)}\n")
    sys.stderr.flush()
    completed = subprocess.run(command, check=False)  # noqa: S603 - the vector is assembled from library constants.
    if completed.returncode != 0:
        message = (
            f"Unable to install the CUDA-enabled PyTorch build. The command '{' '.join(command)}' exited with code "
            f"{completed.returncode}."
        )
        raise RuntimeError(message)


def _verify_installation() -> tuple[str | None, str | None, bool]:
    """Reads the newly installed build's identity from a fresh interpreter.

    The torch module this process imported at startup keeps reporting the build the replacement removed, so the
    verification runs in a subprocess that imports the new one.

    Returns:
        A tuple of the installed torch version, the CUDA version it targets, and whether it reaches a local GPU. The
        first two are None when the verification could not read the result.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - the vector holds this interpreter and a library constant.
            [sys.executable, "-c", _VERIFICATION_SCRIPT],
            capture_output=True,
            text=True,
            check=False,
            timeout=_VERIFICATION_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        warn("The installed PyTorch build could not be verified, because the verification interpreter did not run.")
        return None, None, False
    lines = completed.stdout.splitlines()
    if completed.returncode != 0 or len(lines) < _VERIFICATION_LINE_COUNT:
        warn(f"The installed PyTorch build could not be verified: {completed.stderr.strip()}")
        return None, None, False
    return lines[0].strip(), lines[1].strip() or None, lines[2].strip() == "1"
