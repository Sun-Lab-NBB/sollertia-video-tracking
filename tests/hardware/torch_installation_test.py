"""Contains tests for the driver query, wheel-variant resolution, and torch replacement helpers."""

import sys
import shutil
from typing import Any, Self
from urllib import request
from importlib import metadata
import subprocess
from dataclasses import dataclass

import torch
import pytest

from sollertia_video_tracking.hardware import torch_installation
from sollertia_video_tracking.hardware.torch_installation import (
    TorchInstaller,
    TorchInstallationStatus,
    TorchInstallationSummary,
    install_cuda_torch,
)

_QUERY_REPORT: str = """
    Product Name                          : NVIDIA RTX A6000
    Driver Version                        : 610.43.03
    CUDA Version                          : 13.3 [Deprecated]
    CUDA UMD Version                      : 13.3
    Product Name                          : NVIDIA RTX A6000
"""
"""A trimmed nvidia-smi query report carrying two GPUs and both spellings of the CUDA version field."""

_HEADER_REPORT: str = "| NVIDIA-SMI 550.54.14    Driver Version: 550.54.14    CUDA Version: 12.4    |\n"
"""The header line older driver generations print in place of a query-report CUDA version."""

_PUBLISHED_VARIANTS: tuple[tuple[int, int], ...] = (
    (11, 8),
    (12, 1),
    (12, 4),
    (12, 6),
    (12, 8),
    (12, 9),
    (13, 0),
    (13, 2),
)
"""The wheel variants the faked index publishes, so resolution does not depend on the live index."""

_INDEX_LISTING: str = (
    '<a href="cpu/">cpu</a>\n<a href="cu92/">cu92</a>\n<a href="cu118/">cu118</a>\n'
    '<a href="cu130/">cu130</a>\n<a href="cu132/">cu132</a>\n<a href="rocm7.2/">rocm7.2</a>\n'
)
"""A trimmed wheel index listing carrying supported variants, an unsupported one, and non-CUDA directories."""


@dataclass
class _Completed:
    """Stands in for the completed-process object the subprocess helpers inspect."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class _Response:
    """Stands in for the wheel index response the variant reader opens as a context manager."""

    payload: bytes = b""

    def __enter__(self) -> Self:
        """Returns this response as the managed resource."""
        return self

    def __exit__(self, *_arguments: object) -> None:
        """Releases the managed resource, which holds nothing to close."""

    def read(self) -> bytes:
        """Returns the listing payload."""
        return self.payload


def _summary(**overrides: Any) -> TorchInstallationSummary:
    """Builds a summary whose fields are the given overrides layered over a replaced-build baseline."""
    fields = {
        "status": TorchInstallationStatus.REPLACED,
        "unavailable_reason": "",
        "gpu_names": ("NVIDIA RTX A6000",),
        "driver_cuda_version": "13.3",
        "wheel_variant": "cu130",
        "index_url": "https://download.pytorch.org/whl/cu130",
        "torch_version": "2.13.0+cu130",
        "torch_cuda_version": "13.0",
        "cuda_available": True,
        "replaced_packages": ("torch", "torchvision"),
        "commands": (("uninstall",), ("install",)),
    }
    fields.update(overrides)
    return TorchInstallationSummary(**fields)  # type: ignore[arg-type]


def _fake_torch(monkeypatch: pytest.MonkeyPatch, version: str, cuda: str | None, *, available: bool) -> None:
    """Fakes the torch build this interpreter loaded, so the resolver sees a deterministic environment."""
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)


def _fake_driver(monkeypatch: pytest.MonkeyPatch, gpus: tuple[str, ...], cuda: tuple[int, int] | None) -> None:
    """Fakes the nvidia-smi query so the installer sees a deterministic driver."""
    monkeypatch.setattr(torch_installation, "_query_nvidia_driver", lambda: (gpus, cuda))


def _fake_index(monkeypatch: pytest.MonkeyPatch, variants: tuple[tuple[int, int], ...] = _PUBLISHED_VARIANTS) -> None:
    """Fakes the wheel index read, so the resolver sees a deterministic set of published variants."""
    monkeypatch.setattr(torch_installation, "_read_published_variants", lambda: variants)


def _record_runs(monkeypatch: pytest.MonkeyPatch, verification: _Completed) -> list[tuple[str, ...]]:
    """Replaces the subprocess runner with a recorder that answers the verification probe with the given result."""
    calls: list[tuple[str, ...]] = []

    def _run(command: Any, **_kwargs: Any) -> _Completed:
        calls.append(tuple(command))
        return verification if "-c" in command else _Completed()

    monkeypatch.setattr(subprocess, "run", _run)
    return calls


# TorchInstallationSummary
def test_summary_gpu_count_reports_reported_device_count() -> None:
    """Verifies that the GPU count property counts the names the driver query reported."""
    assert _summary(gpu_names=()).gpu_count == 0
    assert _summary(gpu_names=("a", "b", "c")).gpu_count == 3


def test_summary_describes_an_unavailable_outcome() -> None:
    """Verifies that an unavailable outcome is described by its recorded reason."""
    summary = _summary(status=TorchInstallationStatus.UNAVAILABLE, unavailable_reason="no driver answered")
    assert summary.describe() == "no CUDA build applies: no driver answered"


def test_summary_describes_an_already_enabled_build() -> None:
    """Verifies that an already-enabled build is described with its CUDA version and the force hint."""
    description = _summary(status=TorchInstallationStatus.ENABLED).describe()
    assert "torch 2.13.0+cu130 already targets CUDA 13.0 and reaches 1 GPU(s)" in description
    assert "--force" in description


def test_summary_describes_a_previewed_replacement() -> None:
    """Verifies that a previewed replacement names the covered distributions and the re-run hint."""
    description = _summary(status=TorchInstallationStatus.PREVIEWED).describe()
    assert description.startswith("dry run: would replace torch, torchvision with the cu130 build.")
    assert "--yes" in description


def test_summary_describes_an_executed_replacement() -> None:
    """Verifies that an executed replacement reports the installed version, its CUDA version, and its reach."""
    assert _summary().describe() == ("installed torch 2.13.0+cu130, which targets CUDA 13.0 and reaches 1 GPU(s).")
    assert "reaches no GPU" in _summary(cuda_available=False).describe()


def test_summary_describes_an_unverified_replacement() -> None:
    """Verifies that a replacement whose verification failed reports the variant instead of a missing version."""
    description = _summary(torch_version=None).describe()
    assert description == "installed the cu130 build, but the verification could not read the result."


# install_cuda_torch
def test_install_reports_macos_as_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that macOS is reported as unavailable, since PyTorch publishes no CUDA build for it."""
    _fake_driver(monkeypatch=monkeypatch, gpus=(), cuda=None)
    _fake_torch(monkeypatch=monkeypatch, version="2.13.0", cuda=None, available=False)
    monkeypatch.setattr(sys, "platform", "darwin")

    summary = install_cuda_torch()

    assert summary.status == TorchInstallationStatus.UNAVAILABLE
    assert "macOS" in summary.unavailable_reason
    assert summary.commands == ()


def test_install_reports_a_missing_driver_as_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a machine whose driver reports no CUDA version is reported as unavailable."""
    _fake_driver(monkeypatch=monkeypatch, gpus=(), cuda=None)
    _fake_torch(monkeypatch=monkeypatch, version="2.13.0", cuda=None, available=False)
    monkeypatch.setattr(sys, "platform", "win32")

    summary = install_cuda_torch()

    assert summary.status == TorchInstallationStatus.UNAVAILABLE
    assert "nvidia-smi" in summary.unavailable_reason
    assert summary.driver_cuda_version is None


def test_install_reports_a_prehistoric_driver_as_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a driver older than every published wheel variant is reported as unavailable."""
    _fake_driver(monkeypatch=monkeypatch, gpus=("NVIDIA GTX 1080",), cuda=(10, 2))
    _fake_torch(monkeypatch=monkeypatch, version="2.13.0", cuda=None, available=False)
    _fake_index(monkeypatch=monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")

    summary = install_cuda_torch()

    assert summary.status == TorchInstallationStatus.UNAVAILABLE
    assert "CUDA 10.2 predates CUDA 11.8" in summary.unavailable_reason


def test_install_aborts_when_the_wheel_index_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an unreadable wheel index aborts the run instead of resolving against a stale variant list."""
    _fake_driver(monkeypatch=monkeypatch, gpus=("NVIDIA RTX A6000",), cuda=(13, 3))
    _fake_torch(monkeypatch=monkeypatch, version="2.13.0", cuda=None, available=False)
    monkeypatch.setattr(sys, "platform", "win32")

    def _raise(*_args: Any, **_kwargs: Any) -> _Response:
        message = "the index did not answer"
        raise OSError(message)

    monkeypatch.setattr(request, "urlopen", _raise)

    with pytest.raises(RuntimeError, match="must be readable to list the published variants"):
        install_cuda_torch()


def test_install_leaves_an_enabled_build_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a build already targeting CUDA and reaching the GPUs is left untouched."""
    _fake_driver(monkeypatch=monkeypatch, gpus=("NVIDIA RTX A6000",), cuda=(13, 3))
    _fake_torch(monkeypatch=monkeypatch, version="2.13.0+cu130", cuda="13.0", available=True)
    _fake_index(monkeypatch=monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")

    summary = install_cuda_torch()

    assert summary.status == TorchInstallationStatus.ENABLED
    assert summary.wheel_variant == "cu132"
    assert summary.commands == ()


def test_install_previews_the_resolved_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a dry run resolves the variant and the commands without running anything."""
    _fake_driver(monkeypatch=monkeypatch, gpus=("NVIDIA RTX A6000",), cuda=(12, 7))
    _fake_torch(monkeypatch=monkeypatch, version="2.13.0", cuda=None, available=False)
    _fake_index(monkeypatch=monkeypatch)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch_installation, "_is_installed", lambda _distribution_name: True)
    monkeypatch.setattr(shutil, "which", lambda _command: None)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("must not run a command"))

    summary = install_cuda_torch()

    assert summary.status == TorchInstallationStatus.PREVIEWED
    assert summary.wheel_variant == "cu126"
    assert summary.index_url == "https://download.pytorch.org/whl/cu126"
    assert summary.replaced_packages == ("torch", "torchvision")
    uninstall, install = summary.commands
    assert uninstall == (sys.executable, "-m", "pip", "uninstall", "--yes", "torch", "torchvision")
    assert install == (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--index-url",
        "https://download.pytorch.org/whl/cu126",
        "torch>=2,<3",
        "torchvision",
        "numpy>=1.26,<2",
    )


def test_install_honors_an_explicit_cuda_version_and_torch_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the explicit CUDA and torch versions override the driver query and the default requirement."""
    _fake_driver(monkeypatch=monkeypatch, gpus=("NVIDIA RTX A6000",), cuda=(13, 3))
    _fake_torch(monkeypatch=monkeypatch, version="2.13.0", cuda=None, available=False)
    _fake_index(monkeypatch=monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch_installation, "_is_installed", lambda _distribution_name: False)
    monkeypatch.setattr(shutil, "which", lambda _command: "/usr/bin/uv")

    summary = install_cuda_torch(cuda_version="cu121", torch_version="2.5.1")

    assert summary.wheel_variant == "cu121"
    assert summary.replaced_packages == ("torch", "torchvision")
    assert summary.commands[0] == (
        "/usr/bin/uv",
        "pip",
        "uninstall",
        "--python",
        sys.executable,
        "torch",
        "torchvision",
    )
    assert "torch==2.5.1" in summary.commands[1]


def test_install_honors_an_explicit_index_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an explicit wheel index bypasses the driver query and names the variant from its final segment."""
    _fake_driver(monkeypatch=monkeypatch, gpus=(), cuda=None)
    _fake_torch(monkeypatch=monkeypatch, version="2.13.0", cuda=None, available=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(torch_installation, "_is_installed", lambda _distribution_name: False)
    monkeypatch.setattr(shutil, "which", lambda _command: None)

    summary = install_cuda_torch(index_url="https://example.org/whl/cu128/")

    assert summary.status == TorchInstallationStatus.PREVIEWED
    assert summary.wheel_variant == "cu128"
    assert summary.index_url == "https://example.org/whl/cu128/"


def test_install_replaces_an_unreachable_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a CUDA build the driver cannot run is replaced without the force flag."""
    _fake_driver(monkeypatch=monkeypatch, gpus=("NVIDIA RTX A6000",), cuda=(12, 4))
    _fake_torch(monkeypatch=monkeypatch, version="2.13.0+cu130", cuda="13.0", available=False)
    _fake_index(monkeypatch=monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch_installation, "_is_installed", lambda _distribution_name: False)
    monkeypatch.setattr(shutil, "which", lambda _command: None)

    summary = install_cuda_torch()

    assert summary.status == TorchInstallationStatus.PREVIEWED
    assert summary.wheel_variant == "cu124"


def test_install_replaces_an_enabled_build_when_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the force flag replaces a build that already reaches the GPUs."""
    _fake_driver(monkeypatch=monkeypatch, gpus=("NVIDIA RTX A6000",), cuda=(13, 3))
    _fake_torch(monkeypatch=monkeypatch, version="2.13.0+cu130", cuda="13.0", available=True)
    _fake_index(monkeypatch=monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch_installation, "_is_installed", lambda _distribution_name: False)
    monkeypatch.setattr(shutil, "which", lambda _command: None)

    summary = install_cuda_torch(force=True)

    assert summary.status == TorchInstallationStatus.PREVIEWED


def test_install_runs_and_verifies_the_replacement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that an executed run issues both commands and reports the verified build."""
    _fake_driver(monkeypatch=monkeypatch, gpus=("NVIDIA RTX A6000",), cuda=(13, 3))
    _fake_torch(monkeypatch=monkeypatch, version="2.13.0", cuda=None, available=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch_installation, "_is_installed", lambda _distribution_name: False)
    monkeypatch.setattr(shutil, "which", lambda _command: None)
    _fake_index(monkeypatch=monkeypatch)
    calls = _record_runs(monkeypatch=monkeypatch, verification=_Completed(stdout="2.13.0+cu130\n13.0\n1\n"))

    summary = install_cuda_torch(execute=True)

    assert summary.status == TorchInstallationStatus.REPLACED
    assert summary.torch_version == "2.13.0+cu130"
    assert summary.torch_cuda_version == "13.0"
    assert summary.cuda_available is True
    assert len(calls) == 3
    assert "uninstall" in calls[0]
    assert "install" in calls[1]
    assert calls[2][:2] == (sys.executable, "-c")
    assert "running:" in capsys.readouterr().err


def test_install_raises_when_a_command_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a failing installer command aborts the run with the command and its exit code."""
    _fake_driver(monkeypatch=monkeypatch, gpus=("NVIDIA RTX A6000",), cuda=(13, 3))
    _fake_torch(monkeypatch=monkeypatch, version="2.13.0", cuda=None, available=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch_installation, "_is_installed", lambda _distribution_name: False)
    monkeypatch.setattr(shutil, "which", lambda _command: None)
    _fake_index(monkeypatch=monkeypatch)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Completed(returncode=2))

    with pytest.raises(RuntimeError, match="exited with code 2"):
        install_cuda_torch(execute=True)


# _query_nvidia_driver
def test_driver_query_reads_names_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the query report yields every GPU name and the reported CUDA version."""
    monkeypatch.setattr(torch_installation, "_run_nvidia_smi", lambda _arguments: _QUERY_REPORT)

    names, version = torch_installation._query_nvidia_driver()

    assert names == ("NVIDIA RTX A6000", "NVIDIA RTX A6000")
    assert version == (13, 3)


def test_driver_query_falls_back_to_the_summary_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a query report without a CUDA version falls back to the summary table header."""
    reports = {("-q",): "    Product Name    : NVIDIA RTX A6000\n", (): _HEADER_REPORT}
    monkeypatch.setattr(torch_installation, "_run_nvidia_smi", reports.get)

    names, version = torch_installation._query_nvidia_driver()

    assert names == ("NVIDIA RTX A6000",)
    assert version == (12, 4)


def test_driver_query_reports_nothing_when_both_invocations_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an unreadable summary table leaves the version unresolved."""
    reports: dict[tuple[str, ...], str | None] = {("-q",): "Product Name : NVIDIA RTX A6000\n", (): None}
    monkeypatch.setattr(torch_installation, "_run_nvidia_smi", reports.get)

    assert torch_installation._query_nvidia_driver() == (("NVIDIA RTX A6000",), None)


def test_driver_query_reports_nothing_without_a_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a machine with no readable nvidia-smi yields no names and no version."""
    monkeypatch.setattr(torch_installation, "_run_nvidia_smi", lambda _arguments: None)

    assert torch_installation._query_nvidia_driver() == ((), None)


# _run_nvidia_smi
def test_run_nvidia_smi_returns_captured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a successful invocation returns the captured standard output."""
    monkeypatch.setattr(shutil, "which", lambda _command: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Completed(stdout="report"))

    assert torch_installation._run_nvidia_smi(("-q",)) == "report"


def test_run_nvidia_smi_returns_none_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a machine without nvidia-smi on the path yields no output."""
    monkeypatch.setattr(shutil, "which", lambda _command: None)

    assert torch_installation._run_nvidia_smi(()) is None


def test_run_nvidia_smi_returns_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a non-zero exit status yields no output."""
    monkeypatch.setattr(shutil, "which", lambda _command: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Completed(returncode=9, stdout="partial"))

    assert torch_installation._run_nvidia_smi(()) is None


def test_run_nvidia_smi_returns_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an unresponsive driver query yields no output rather than propagating."""
    monkeypatch.setattr(shutil, "which", lambda _command: "/usr/bin/nvidia-smi")

    def _raise(*_args: Any, **_kwargs: Any) -> _Completed:
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", _raise)

    assert torch_installation._run_nvidia_smi(()) is None


# _extract_cuda_version
def test_extract_cuda_version_reads_both_spellings() -> None:
    """Verifies that the version is read from the query report field and from the summary table header."""
    assert torch_installation._extract_cuda_version(_QUERY_REPORT) == (13, 3)
    assert torch_installation._extract_cuda_version(_HEADER_REPORT) == (12, 4)
    assert torch_installation._extract_cuda_version("CUDA UMD Version : 13.3") == (13, 3)


def test_extract_cuda_version_reports_nothing_for_a_versionless_report() -> None:
    """Verifies that a report carrying no CUDA version yields nothing."""
    assert torch_installation._extract_cuda_version("Product Name : NVIDIA RTX A6000") is None


# _parse_cuda_version
@pytest.mark.parametrize(
    "requested,expected",
    [
        ("12.6", (12, 6)),
        ("cu126", (12, 6)),
        ("CU130", (13, 0)),
        (" 11.8 ", (11, 8)),
        ("cu118", (11, 8)),
    ],
)
def test_parse_cuda_version_accepts_dotted_and_tagged_forms(requested: str, expected: tuple[int, int]) -> None:
    """Verifies that both the dotted version and the wheel-variant tag parse to the same pair."""
    assert torch_installation._parse_cuda_version(requested) == expected


@pytest.mark.parametrize("requested", ["nonsense", "12", "12.6.1", "", "cu"])
def test_parse_cuda_version_rejects_unrecognized_requests(requested: str) -> None:
    """Verifies that an unrecognizable request is rejected with the offending value."""
    with pytest.raises(ValueError, match="Unable to resolve the requested CUDA version"):
        torch_installation._parse_cuda_version(requested)


# _resolve_wheel_variant
@pytest.mark.parametrize(
    "cuda_version,expected",
    [
        ((13, 3), "cu132"),
        ((13, 2), "cu132"),
        ((13, 0), "cu130"),
        ((12, 9), "cu129"),
        ((12, 7), "cu126"),
        ((12, 0), "cu118"),
        ((11, 8), "cu118"),
    ],
)
def test_resolve_wheel_variant_takes_the_newest_supported_variant(
    monkeypatch: pytest.MonkeyPatch, cuda_version: tuple[int, int], expected: str
) -> None:
    """Verifies that the newest variant at or below the reported version wins."""
    _fake_index(monkeypatch=monkeypatch)

    assert torch_installation._resolve_wheel_variant(cuda_version) == expected


def test_resolve_wheel_variant_takes_a_variant_the_fallback_omits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a variant published after this release resolves without appearing in the fallback tuple."""
    _fake_index(monkeypatch=monkeypatch, variants=((13, 0), (13, 2), (14, 1)))

    assert torch_installation._resolve_wheel_variant((14, 3)) == "cu141"


def test_resolve_wheel_variant_reports_nothing_below_the_oldest_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a version older than every published variant resolves to nothing."""
    _fake_index(monkeypatch=monkeypatch)

    assert torch_installation._resolve_wheel_variant((11, 7)) is None


# _read_published_variants
def test_read_published_variants_parses_the_index_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the listing yields its CUDA variants in ascending order, without the unsupported ones."""
    monkeypatch.setattr(request, "urlopen", lambda *args, **kwargs: _Response(payload=_INDEX_LISTING.encode()))

    assert torch_installation._read_published_variants() == ((11, 8), (13, 0), (13, 2))


def test_read_published_variants_rejects_a_variantless_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a listing carrying no supported variant is rejected rather than resolving to nothing."""
    monkeypatch.setattr(request, "urlopen", lambda *args, **kwargs: _Response(payload=b'<a href="cpu/">cpu</a>'))

    with pytest.raises(RuntimeError, match=r"must list a CUDA variant at or above CUDA 11\.8"):
        torch_installation._read_published_variants()


def test_read_published_variants_rejects_an_unreachable_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an unreachable index aborts the resolution and names the failure it hit."""

    def _raise(*_args: Any, **_kwargs: Any) -> _Response:
        message = "the index did not answer"
        raise OSError(message)

    monkeypatch.setattr(request, "urlopen", _raise)

    with pytest.raises(RuntimeError, match="the index did not answer"):
        torch_installation._read_published_variants()


# _format_cuda_version
def test_format_cuda_version_renders_a_dotted_version() -> None:
    """Verifies that a version pair renders as a dotted version and that a missing pair renders as nothing."""
    assert torch_installation._format_cuda_version((12, 6)) == "12.6"
    assert torch_installation._format_cuda_version(None) is None


# _is_installed
def test_is_installed_detects_present_and_absent_distributions() -> None:
    """Verifies that an installed distribution resolves and an absent one does not."""
    assert torch_installation._is_installed("torch") is True
    assert torch_installation._is_installed("a-distribution-that-is-not-installed") is False


def test_is_installed_reports_false_for_a_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the metadata lookup failure is translated into a negative answer."""

    def _raise(name: str) -> None:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", _raise)

    assert torch_installation._is_installed("torch") is False


# _resolve_installer_prefixes
def test_installer_prefixes_prefer_uv_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the automatic choice targets this interpreter through uv when uv is on the path."""
    monkeypatch.setattr(shutil, "which", lambda _command: "/usr/bin/uv")

    uninstall, install = torch_installation._resolve_installer_prefixes(TorchInstaller.AUTO)

    assert uninstall == ("/usr/bin/uv", "pip", "uninstall", "--python", sys.executable)
    assert install == ("/usr/bin/uv", "pip", "install", "--python", sys.executable)


def test_installer_prefixes_fall_back_to_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the automatic choice falls back to this interpreter's pip when uv is absent."""
    monkeypatch.setattr(shutil, "which", lambda _command: None)

    uninstall, install = torch_installation._resolve_installer_prefixes(TorchInstaller.AUTO)

    assert uninstall == (sys.executable, "-m", "pip", "uninstall", "--yes")
    assert install == (sys.executable, "-m", "pip", "install")


def test_installer_prefixes_honor_a_forced_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that requesting pip skips the uv lookup entirely."""
    monkeypatch.setattr(shutil, "which", lambda _command: pytest.fail("must not probe for uv"))

    uninstall, _ = torch_installation._resolve_installer_prefixes(TorchInstaller.PIP)

    assert uninstall[:3] == (sys.executable, "-m", "pip")


def test_installer_prefixes_reject_a_missing_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that requesting uv while it is absent from the path is rejected."""
    monkeypatch.setattr(shutil, "which", lambda _command: None)

    with pytest.raises(ValueError, match="no 'uv' executable"):
        torch_installation._resolve_installer_prefixes(TorchInstaller.UV)


# _run_installer_command
def test_run_installer_command_announces_and_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that the command is announced on the standard error stream before it runs."""
    calls = _record_runs(monkeypatch=monkeypatch, verification=_Completed())

    torch_installation._run_installer_command(("pip", "install", "torch"))

    assert calls == [("pip", "install", "torch")]
    assert capsys.readouterr().err == "running: pip install torch\n"


def test_run_installer_command_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a non-zero exit status is reported with the command and the code."""
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Completed(returncode=1))

    with pytest.raises(RuntimeError, match="'pip install torch' exited with code 1"):
        torch_installation._run_installer_command(("pip", "install", "torch"))


# _verify_installation
def test_verify_installation_reads_the_replaced_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the probe's three lines are read back as the version, the CUDA version, and the reach."""
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Completed(stdout="2.13.0+cu130\n13.0\n1\n"))

    assert torch_installation._verify_installation() == ("2.13.0+cu130", "13.0", True)


def test_verify_installation_reads_a_cpu_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an empty CUDA line reads back as a CPU-only build that reaches no GPU."""
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Completed(stdout="2.13.0\n\n0\n"))

    assert torch_installation._verify_installation() == ("2.13.0", None, False)


def test_verify_installation_warns_on_a_failed_probe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that a failing probe warns with its error output instead of reporting a build."""
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: _Completed(returncode=1, stderr="ImportError: no torch\n")
    )

    assert torch_installation._verify_installation() == (None, None, False)
    assert "ImportError: no torch" in capsys.readouterr().err


def test_verify_installation_warns_on_a_truncated_probe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that a probe printing fewer fields than expected is treated as unverified."""
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Completed(stdout="2.13.0\n"))

    assert torch_installation._verify_installation() == (None, None, False)
    assert "could not be verified" in capsys.readouterr().err


def test_verify_installation_warns_when_the_probe_cannot_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that a probe that never starts is reported as unverified rather than propagating."""

    def _raise(*_args: Any, **_kwargs: Any) -> _Completed:
        message = "interpreter is gone"
        raise OSError(message)

    monkeypatch.setattr(subprocess, "run", _raise)

    assert torch_installation._verify_installation() == (None, None, False)
    assert "did not run" in capsys.readouterr().err


def test_validate_torch_version_accepts_the_supported_series():
    """Verifies that a supported dotted version is accepted with a pre-release segment, a build tag, or neither."""
    torch_installation._validate_torch_version("2.9.1")
    torch_installation._validate_torch_version("2.9.1+cu126")
    torch_installation._validate_torch_version("2.10.0rc1")


def test_validate_torch_version_rejects_a_malformed_version():
    """Verifies that a value that is not a dotted version is rejected before anything is uninstalled."""
    with pytest.raises(ValueError, match="dotted version"):
        torch_installation._validate_torch_version("latest")


def test_validate_torch_version_rejects_an_unsupported_major_series():
    """Verifies that a version outside the supported major series is rejected before anything is uninstalled."""
    with pytest.raises(ValueError, match=r"2\.x series only"):
        torch_installation._validate_torch_version("1.13.1")
