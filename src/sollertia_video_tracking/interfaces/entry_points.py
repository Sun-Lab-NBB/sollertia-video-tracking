"""Provides the single ``slvt`` console-script root command group for the sollertia-video-tracking library."""

import click

from .infer import infer_command
from .train import train_command
from .export import export_command
from .extract import extract_group
from .prepare import prepare_command
from .predict import predict_command

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.group("slvt", context_settings=_CONTEXT_SETTINGS)
def slvt_cli() -> None:
    """Designs and deploys DeepLabCut video-tracking pipelines for the Sollertia platform."""


slvt_cli.add_command(cmd=extract_group)
slvt_cli.add_command(cmd=prepare_command)
slvt_cli.add_command(cmd=train_command)
slvt_cli.add_command(cmd=infer_command)
slvt_cli.add_command(cmd=export_command)
slvt_cli.add_command(cmd=predict_command)
