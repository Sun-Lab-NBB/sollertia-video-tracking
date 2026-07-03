"""Provides the ``slvt extract-frames`` command that runs parallel DeepLabCut k-means frame extraction for a project."""

from pathlib import Path

import click

from ..frame_extraction import DEFAULT_RESERVED_CORE_COUNT, extract_frames_kmeans

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.command("extract-frames", context_settings=_CONTEXT_SETTINGS)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-s",
    "--clustering-stride",
    "clustering_stride",
    default=500,
    show_default=True,
    help="The clustering stride passed to DeepLabCut as cluster_step; every Nth frame is sampled. Larger values "
    "issue far fewer of the expensive video encoder seeks and finish sooner on long-GOP video.",
)
@click.option(
    "-w",
    "--workers",
    "worker_count",
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
    "Video decoding keeps roughly four cores per video busy, so larger values mostly idle cores while smaller values "
    "throttle each decode.",
)
@click.option(
    "-r",
    "--reserve-cores",
    "reserved_core_count",
    default=DEFAULT_RESERVED_CORE_COUNT,
    show_default=True,
    help="The number of CPU cores left free for other tasks while extraction runs.",
)
@click.option(
    "-n",
    "--frames-per-video",
    "frames_per_video",
    type=int,
    default=-1,
    show_default=True,
    help="The number of frames to keep per video, overriding numframes2pick in config.yaml. Set to -1 to use the "
    "value already stored in the configuration file.",
)
@click.option(
    "-t",
    "--total-frames",
    "total_frame_budget",
    type=int,
    default=-1,
    show_default=True,
    help="The total number of frames to hold across the project. When set, a random subset of not-yet-extracted "
    "videos is sampled to reach this budget (each contributing --frames-per-video frames), growing coverage on "
    "repeated runs. Set to -1 to extract every selected video instead.",
)
@click.option(
    "--seed",
    "random_seed",
    type=int,
    default=None,
    help="The seed for the random video subset draw. Omit for a different random subset each run, or set it to make "
    "the selection reproducible.",
)
@click.option(
    "--balance-groups",
    "balance_groups",
    is_flag=True,
    help="Balance the budgeted sample across groups instead of drawing uniformly, so every group is represented and "
    "coverage evens out across repeated runs. The group of each video is inferred from the non-date components its "
    "file name shares with the others. Only applies together with --total-frames.",
)
@click.option(
    "--group-by",
    "group_by_pattern",
    default=None,
    metavar="REGEX",
    help="A regular expression whose first capturing group names the group for each video's file name, overriding the "
    "built-in inference for naming schemes it does not cover (for example '(Grp\\d+)' for names like D1_Grp2). "
    "Implies --balance-groups.",
)
@click.option(
    "--always-include",
    "always_include_videos",
    multiple=True,
    metavar="SUBSTRING",
    help="A path substring naming a video to always include in the budgeted sample, selected before the balanced or "
    "uniform draw fills the rest. Provide the option multiple times to pin several videos. Only applies together with "
    "--total-frames.",
)
@click.option(
    "--clustering-resize-width",
    "clustering_resize_width",
    default=30,
    show_default=True,
    help="The downsample width applied before clustering, passed to DeepLabCut as cluster_resizewidth.",
)
@click.option(
    "--color/--grayscale",
    "cluster_in_color",
    default=False,
    show_default=True,
    help="Determines whether to cluster on color channels instead of grayscale.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Determines whether to re-extract videos whose labeled-data directory already contains frames. WARNING: "
    "this deletes the existing extracted frames AND their labels in each re-extracted directory, which the new "
    "frames would otherwise orphan. Mutually exclusive with --reset.",
)
@click.option(
    "--reset",
    is_flag=True,
    help="Determines whether to discard ALL extracted frames and their labels in the selection and re-extract from "
    "scratch. WARNING: this permanently deletes the extracted frames and any labels in every selected video folder. "
    "Mutually exclusive with --overwrite.",
)
@click.option(
    "--path-filter",
    "path_filters",
    multiple=True,
    metavar="SUBSTRING",
    help="The path substring that restricts the run to videos whose path contains it. Provide the option multiple "
    "times to match several substrings.",
)
@click.option(
    "--minimum-progress-interval",
    "minimum_progress_interval",
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
    clustering_stride: int,
    worker_count: int,
    cores_per_worker: int,
    reserved_core_count: int,
    frames_per_video: int,
    total_frame_budget: int,
    random_seed: int | None,
    group_by_pattern: str | None,
    always_include_videos: tuple[str, ...],
    clustering_resize_width: int,
    path_filters: tuple[str, ...],
    minimum_progress_interval: float,
    *,
    balance_groups: bool,
    cluster_in_color: bool,
    overwrite: bool,
    reset: bool,
    display_progress: bool,
) -> None:
    """Selects DeepLabCut training frames from a project's videos by clustering them in parallel.

    CONFIG is the path to the DeepLabCut project's config.yaml. Each video is clustered in its own worker process
    pinned to a disjoint block of CPU cores, and videos whose labeled-data directory already contains frames are
    skipped unless ``--overwrite`` is given. Passing ``--total-frames`` instead samples a random subset of
    not-yet-extracted videos sized to reach that project-wide frame budget, growing coverage across repeated runs.
    """
    try:
        summary = extract_frames_kmeans(
            config_path=config,
            clustering_stride=clustering_stride,
            worker_count=worker_count,
            cores_per_worker=cores_per_worker,
            reserved_core_count=reserved_core_count,
            frames_per_video=frames_per_video,
            total_frame_budget=total_frame_budget,
            random_seed=random_seed,
            balance_groups=balance_groups,
            group_by_pattern=group_by_pattern,
            always_include_videos=always_include_videos,
            clustering_resize_width=clustering_resize_width,
            cluster_in_color=cluster_in_color,
            overwrite=overwrite,
            reset=reset,
            path_filters=path_filters,
            minimum_progress_interval=minimum_progress_interval,
            display_progress=display_progress,
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
