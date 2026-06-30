"""Provides the ``slvt extract-frames`` command that runs parallel DeepLabCut k-means frame extraction for a project.

Notes:
    The command is a thin wrapper around the frame-extraction pipeline in the helpers package: it parses the
    command-line options, delegates the work, and translates the returned summary into console output and a process
    exit status.
"""

from pathlib import Path

import click

from ..frame_extraction import DEFAULT_RESERVE_CORES, extract_frames_kmeans

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.command("extract-frames", context_settings=_CONTEXT_SETTINGS)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-s",
    "--step",
    default=500,
    show_default=True,
    help="The clustering stride passed to DeepLabCut as cluster_step; every Nth frame is sampled. Larger values "
    "issue far fewer of the expensive HEVC seeks and finish sooner on long-GOP video.",
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=-1,
    show_default=True,
    help="The number of videos to decode in parallel. Set to -1 to fill the usable CPU cores automatically.",
)
@click.option(
    "-c",
    "--cores-per-worker",
    "cores_per_worker",
    type=int,
    default=-1,
    show_default=True,
    help="The number of CPU cores pinned to each worker. Set to -1 to spread the usable cores evenly across workers. "
    "HEVC decoding keeps roughly four cores per video busy, so larger values mostly idle cores while smaller values "
    "throttle each decode.",
)
@click.option(
    "-r",
    "--reserve-cores",
    "reserve_cores",
    default=DEFAULT_RESERVE_CORES,
    show_default=True,
    help="The number of CPU cores left free for other tasks while extraction runs.",
)
@click.option(
    "-n",
    "--num-frames",
    "num_frames",
    type=int,
    default=-1,
    show_default=True,
    help="The number of frames to keep per video, overriding numframes2pick in config.yaml. Set to -1 to use the "
    "value already stored in the configuration file.",
)
@click.option(
    "--resize-width",
    "resize_width",
    default=30,
    show_default=True,
    help="The downsample width applied before clustering, passed to DeepLabCut as cluster_resizewidth.",
)
@click.option(
    "--color/--grayscale",
    "color",
    default=False,
    show_default=True,
    help="Determines whether to cluster on color channels instead of grayscale.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Determines whether to re-extract videos whose labeled-data directory already contains frames.",
)
@click.option(
    "--only",
    "path_filters",
    multiple=True,
    metavar="SUBSTR",
    help="The path substring that restricts the run to videos whose path contains it. Provide the option multiple "
    "times to match several substrings.",
)
@click.option(
    "--heartbeat",
    default=30.0,
    show_default=True,
    metavar="SECONDS",
    help="The minimum interval, in seconds, between progress lines when the output is not a TTY.",
)
@click.option(
    "--progress/--no-progress",
    "display_progress",
    default=True,
    show_default=True,
    help="Determines whether to display the aggregate progress bar during extraction.",
)
def extract_frames_command(
    config: Path,
    step: int,
    workers: int,
    cores_per_worker: int,
    reserve_cores: int,
    num_frames: int,
    resize_width: int,
    path_filters: tuple[str, ...],
    heartbeat: float,
    *,
    color: bool,
    overwrite: bool,
    display_progress: bool,
) -> None:
    """Selects DeepLabCut training frames from a project's videos by clustering them in parallel.

    CONFIG is the path to the DeepLabCut project's config.yaml. Each video is clustered in its own worker process
    pinned to a disjoint block of CPU cores, and videos whose labeled-data directory already contains frames are
    skipped unless ``--overwrite`` is given.
    """
    try:
        summary = extract_frames_kmeans(
            config_path=config,
            step=step,
            workers=workers,
            cores_per_worker=cores_per_worker,
            reserve_cores=reserve_cores,
            num_frames=num_frames,
            resize_width=resize_width,
            color=color,
            overwrite=overwrite,
            path_filters=path_filters,
            heartbeat=heartbeat,
            display_progress=display_progress,
        )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(str(error)) from error

    for video, traceback_text in summary.errors:
        click.echo(message=f"\n--- error in {video} ---\n{traceback_text}", err=True)
    click.echo(
        message=(
            f"done: {summary.extracted} extracted, {summary.skipped} skipped, {summary.failed} failed "
            f"(of {summary.total})"
        )
    )
    if not summary.successful:
        raise SystemExit(1)
