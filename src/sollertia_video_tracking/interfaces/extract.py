"""Provides the ``slvt extract-frames`` command that runs parallel DeepLabCut k-means frame extraction for a project."""

from pathlib import Path

import click

from ..frame_extraction import extract_frames_kmeans

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.command("extract-frames", context_settings=_CONTEXT_SETTINGS)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-cs",
    "--clustering-stride",
    default=500,
    show_default=True,
    help="How far apart, in frames, to sample the video when selecting training frames. Larger values scan fewer "
    "frames and finish faster; smaller values sample more densely but take longer.",
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=-1,
    show_default=True,
    help="How many videos to process at the same time. Set to -1 to use the available CPU cores automatically.",
)
@click.option(
    "-co",
    "--cores",
    type=int,
    default=-1,
    show_default=True,
    help="How many CPU cores to devote to each video being processed. Set to -1 to share the available cores evenly. "
    "Each video keeps about four cores busy while it decodes, so larger values mostly idle cores and smaller values "
    "slow each video worker down.",
)
@click.option(
    "-fpv",
    "--frames-per-video",
    type=int,
    default=10,
    show_default=True,
    help="How many frames to keep from each processed video. Set to -1 to use the project's configured amount.",
)
@click.option(
    "-tf",
    "--total-frames",
    type=int,
    default=200,
    show_default=True,
    help="The total number of frames you want across the whole project. When set, not-yet-processed videos are "
    "sampled at random until this many frames are reached, so coverage grows over repeated runs (each video "
    "contributes --frames-per-video frames). Set to -1 to process every selected video instead.",
)
@click.option(
    "-s",
    "--seed",
    type=int,
    default=None,
    help="A fixed seed for the random video sampling. Set it to make the selection repeatable; omit for a different "
    "draw each run.",
)
@click.option(
    "-bg",
    "--balance-groups",
    is_flag=True,
    help="Spread the sampled frames evenly across groups of related videos instead of drawing purely at random, so "
    "every group is represented. Groups are inferred from the parts of the file names the videos share. Only applies "
    "together with --total-frames.",
)
@click.option(
    "-gb",
    "--group-by",
    default=None,
    metavar="REGEX",
    help="A regular expression whose first capturing group names the group for each video, for naming schemes the "
    "automatic grouping does not recognize (for example '(Grp\\d+)' for names like D1_Grp2). Implies --balance-groups.",
)
@click.option(
    "-ai",
    "--always-include",
    multiple=True,
    metavar="SUBSTRING",
    help="A path substring naming a video that must always be included in the sample, chosen before the rest of the "
    "budget is filled. Provide the option several times to pin multiple videos. Only applies together with "
    "--total-frames.",
)
@click.option(
    "-crw",
    "--clustering-resize-width",
    default=50,
    show_default=True,
    help="The width, in pixels, that frames are shrunk to before they are compared for similarity. Smaller values are "
    "faster but compare more coarsely.",
)
@click.option(
    "-cl",
    "--color/--grayscale",
    default=False,
    show_default=True,
    help="Compare frames in color instead of grayscale when selecting them.",
)
@click.option(
    "-o",
    "--overwrite",
    is_flag=True,
    help="Re-process videos that already have extracted frames. WARNING: this deletes the existing extracted frames "
    "AND their labels in each affected folder, which the new frames would otherwise orphan. Mutually exclusive with "
    "--reset.",
)
@click.option(
    "-r",
    "--reset",
    is_flag=True,
    help="Discard ALL extracted frames and their labels across the selection and start over. WARNING: this "
    "permanently deletes the extracted frames and any labels in every selected folder. Mutually exclusive with "
    "--overwrite.",
)
@click.option(
    "-pf",
    "--path-filter",
    multiple=True,
    metavar="SUBSTRING",
    help="Restrict the run to videos whose path contains this substring. Provide the option several times to allow "
    "multiple substrings.",
)
@click.option(
    "-mpi",
    "--minimum-progress-interval",
    default=30.0,
    show_default=True,
    metavar="SECONDS",
    help="The shortest time, in seconds, between progress updates when the output is not a live terminal.",
)
@click.option(
    "-pg",
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Show the aggregate progress bar during extraction.",
)
def extract_frames_command(
    config: Path,
    clustering_stride: int,
    workers: int,
    cores: int,
    frames_per_video: int,
    total_frames: int,
    seed: int | None,
    group_by: str | None,
    always_include: tuple[str, ...],
    clustering_resize_width: int,
    path_filter: tuple[str, ...],
    minimum_progress_interval: float,
    *,
    balance_groups: bool,
    color: bool,
    overwrite: bool,
    reset: bool,
    progress: bool,
) -> None:
    """Selects training frames from a project's videos by clustering them in parallel.

    CONFIG is the path to the project's config.yaml. Each video is clustered in its own worker process pinned to a
    disjoint block of CPU cores, and videos that already contain extracted frames are skipped unless ``--overwrite``
    is given. Passing ``--total-frames`` instead samples a random subset of not-yet-processed videos sized to reach
    that project-wide frame budget, growing coverage across repeated runs.
    """
    try:
        summary = extract_frames_kmeans(
            config_path=config,
            clustering_stride=clustering_stride,
            worker_count=workers,
            cores_per_worker=cores,
            frames_per_video=frames_per_video,
            total_frame_budget=total_frames,
            random_seed=seed,
            balance_groups=balance_groups,
            group_by_pattern=group_by,
            always_include_videos=always_include,
            clustering_resize_width=clustering_resize_width,
            cluster_in_color=color,
            overwrite=overwrite,
            reset=reset,
            path_filters=path_filter,
            minimum_progress_interval=minimum_progress_interval,
            display_progress=progress,
        )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(message=str(error)) from error

    for video, traceback_text in summary.errors:
        click.echo(message=f"\n--- error in {video} ---\n{traceback_text}", err=True)
    click.echo(
        message=(
            f"done: {summary.extracted_video_count} extracted, {summary.skipped_video_count} skipped, "
            f"{summary.failed_video_count} failed (of {summary.total_video_count})"
        ),
    )
    if not summary.successful:
        raise SystemExit(1)
