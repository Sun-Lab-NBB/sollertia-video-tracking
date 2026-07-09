"""Provides the ``slvt export`` command that serializes a trained model into a portable, self-contained asset."""

from pathlib import Path

import click

from ..deploy import ArchiveCompression, export_model

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.command("export", context_settings=_CONTEXT_SETTINGS)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-d",
    "--destination",
    required=True,
    type=click.Path(path_type=Path),
    help="Where to write the model asset. Give a file path ending in .slvtmodel, or a directory to write a "
    "descriptively named asset into.",
)
@click.option(
    "-s", "--shuffle", default=1, show_default=True, help="The shuffle index whose trained model is exported."
)
@click.option(
    "-si",
    "--snapshot-index",
    type=int,
    default=None,
    help="The trained pose snapshot to bundle. Omit to use the project's configured snapshot.",
)
@click.option(
    "-dsi",
    "--detector-snapshot-index",
    type=int,
    default=None,
    help="The detector snapshot to bundle, for top-down models. Omit to use the configured or best detector snapshot.",
)
@click.option(
    "-lt",
    "--likelihood-threshold",
    type=float,
    default=0.0,
    show_default=True,
    help="The default confidence below which keypoint positions are cleared when the asset is deployed. This bakes a "
    "sensible floor into the asset; it can still be overridden at prediction time.",
)
@click.option(
    "-c",
    "--compression",
    type=click.Choice([compression.value for compression in ArchiveCompression]),
    default=ArchiveCompression.NONE.value,
    show_default=True,
    help="How to compress the asset. 'none' keeps runtime extraction fast for repeated deployment; 'gzip' or 'xz' "
    "make a smaller file for slow transport links, at a higher extraction cost.",
)
@click.option(
    "-cr",
    "--crop/--no-crop",
    default=False,
    show_default=True,
    help="Bundle the project's cropping rectangle so deployment analyzes the cropped region. Off by default, so the "
    "asset analyzes full frames and predictions land in full-frame coordinates.",
)
@click.option(
    "-o",
    "--overwrite",
    is_flag=True,
    help="Overwrite an existing asset at the destination instead of refusing to replace it.",
)
def export_command(
    config: Path,
    destination: Path,
    shuffle: int,
    snapshot_index: int | None,
    detector_snapshot_index: int | None,
    likelihood_threshold: float,
    compression: str,
    *,
    crop: bool,
    overwrite: bool,
) -> None:
    """Serializes a trained DeepLabCut shuffle into a portable, condensed model asset that runs on another machine.

    CONFIG is the path to the DeepLabCut project's config.yaml. The asset bundles the project configuration, the
    shuffle's network configuration, and exactly one pose snapshot (plus one detector snapshot for a top-down model)
    into a single tar file, with a manifest recording the model's identity and per-file checksums. The asset depends on
    none of the original project's labeled data or training datasets, so it can be moved anywhere and run with
    ``slvt predict``.
    """
    try:
        summary = export_model(
            config=config,
            destination=destination,
            shuffle=shuffle,
            snapshot_index=snapshot_index,
            detector_snapshot_index=detector_snapshot_index,
            likelihood_threshold=likelihood_threshold,
            compression=ArchiveCompression(compression),
            crop=crop,
            overwrite=overwrite,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise click.ClickException(message=str(error)) from error

    click.echo(message=summary.describe())
