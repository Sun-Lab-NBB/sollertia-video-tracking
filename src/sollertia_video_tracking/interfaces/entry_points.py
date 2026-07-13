"""Provides the single ``slvt`` console-script root command group for the sollertia-video-tracking library."""

import click

from .gui import gui_command
from .infer import infer_command
from .train import train_command
from .extract import extract_group
from .prepare import prepare_command

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Widens displayed Click help messages to 120 columns so option descriptions wrap consistently."""


@click.group("slvt", context_settings=_CONTEXT_SETTINGS)
def slvt_cli() -> None:
    """Designs and deploys DeepLabCut video-tracking pipelines for the Sollertia platform."""


slvt_cli.add_command(cmd=extract_group)
slvt_cli.add_command(cmd=gui_command)
slvt_cli.add_command(cmd=prepare_command)
slvt_cli.add_command(cmd=train_command)
slvt_cli.add_command(cmd=infer_command)
