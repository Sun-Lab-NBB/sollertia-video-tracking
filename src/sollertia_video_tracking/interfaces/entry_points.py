"""Provides the single ``slvt`` console-script root command group for the sollertia-video-tracking library."""

import click

from .extract import extract_frames_command

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.group("slvt", context_settings=_CONTEXT_SETTINGS)
def slvt_cli() -> None:
    """Designs and deploys DeepLabCut video-tracking pipelines for the Sollertia platform."""


slvt_cli.add_command(cmd=extract_frames_command)
