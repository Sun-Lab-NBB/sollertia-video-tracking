"""Provides the ``slvt cuda`` command that installs the CUDA-enabled PyTorch build for the local NVIDIA driver."""

import click

from ..hardware import TorchInstaller, TorchInstallationStatus, install_cuda_torch

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Widens displayed Click help messages to 120 columns so option descriptions wrap consistently."""


@click.command("cuda", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-cv",
    "--cuda-version",
    default=None,
    help="The CUDA version to install the build for, written as '12.6' or 'cu126'. Omit to read it from nvidia-smi. A "
    "version PyTorch publishes no build for resolves to the newest build below it.",
)
@click.option(
    "-tv",
    "--torch-version",
    default=None,
    help="The exact torch version to install, for example '2.9.1'. Omit to install the newest version the wheel index "
    "carries within the 'torch>=2,<3' range this library supports.",
)
@click.option(
    "-iu",
    "--index-url",
    default=None,
    help="The wheel index to install from, which bypasses both the driver query and the CUDA version resolution. Omit "
    "to derive the index from the resolved CUDA version.",
)
@click.option(
    "-i",
    "--installer",
    type=click.Choice([installer.value for installer in TorchInstaller]),
    default=TorchInstaller.AUTO.value,
    show_default=True,
    help="The package manager that carries out the replacement. 'auto' uses uv where it is available and pip "
    "otherwise.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Determines whether to replace a build that already targets CUDA and reaches the local GPUs.",
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Actually replace the installed build. Without this flag the command only reports what it would run, "
    "changing nothing.",
)
def cuda_command(
    cuda_version: str | None,
    torch_version: str | None,
    index_url: str | None,
    installer: str,
    *,
    force: bool,
    yes: bool,
) -> None:
    """Installs the CUDA-enabled PyTorch build matching the local NVIDIA driver, after a dry-run preview.

    The torch distribution published for Windows carries no CUDA support, so a stock install there runs training and
    inference on the CPU. This reads the CUDA version the local driver runs, resolves the newest PyTorch build that
    version supports, and replaces the installed torch, torchvision, and torchaudio distributions with it. The
    replacement runs through uv where it is available and through pip otherwise, and it targets the environment this
    library is installed into rather than whichever environment happens to be active. The command reports what it
    would run and changes nothing until ``--yes`` is given. A build that already targets CUDA and reaches the local
    GPUs is left alone unless ``--force`` is given.
    """
    try:
        summary = install_cuda_torch(
            cuda_version=cuda_version,
            index_url=index_url,
            torch_version=torch_version,
            installer=TorchInstaller(installer),
            execute=yes,
            force=force,
        )
    except (ValueError, RuntimeError) as error:
        raise click.ClickException(message=str(error)) from error

    if summary.gpu_names:
        click.echo(message=f"detected: {', '.join(summary.gpu_names)} (driver CUDA {summary.driver_cuda_version})")
    if summary.torch_version is not None:
        build = "CPU-only build" if summary.torch_cuda_version is None else f"CUDA {summary.torch_cuda_version} build"
        reach = "reaches the GPUs" if summary.cuda_available else "reaches no GPU"
        click.echo(message=f"current:  torch {summary.torch_version} ({build}, {reach})")
    if summary.index_url is not None:
        click.echo(message=f"target:   {summary.wheel_variant}  {summary.index_url}")
    if summary.commands:
        click.echo(message="ran:" if yes else "would run:")
        for command in summary.commands:
            click.echo(message=f"  {' '.join(command)}")
    click.echo(message=summary.describe())
    if summary.status == TorchInstallationStatus.UNAVAILABLE:
        raise SystemExit(1)
