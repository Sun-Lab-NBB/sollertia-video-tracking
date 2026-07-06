"""Provides the ``slvt extract`` command group with its ``frames`` and ``outliers`` frame-extraction subcommands."""

from pathlib import Path
from dataclasses import dataclass

import click

from ..frame_extraction import (
    TrackingMethod,
    OutlierAlgorithm,
    ExtractionAlgorithm,
    extract_frames_kmeans,
    extract_outlier_frames_parallel,
)

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@dataclass(frozen=True)
class SharedExtractionParameters:
    """Bundles the parameters shared by the ``frames`` and ``outliers`` subcommands, parsed on the ``extract`` group.

    The ``extract`` group callback builds one of these from its options and stores it on the Click context, and each
    subcommand reads it back through the ``pass_shared_parameters`` decorator.
    """

    config_path: Path | None
    """The path to the DeepLabCut project's config.yaml both subcommands operate on.

    ``--config-path`` cannot be marked required on the group without also blocking ``slvt extract SUBCOMMAND --help``,
    so it is validated per subcommand through `require_config_path` instead of at parse time, leaving it None
    here when it was omitted.
    """

    worker_count: int
    """How many videos to process at the same time; -1 uses the available CPU cores automatically."""

    cores_per_worker: int
    """How many CPU cores to devote to each processed video; -1 shares the available cores evenly."""

    frames_per_video: int
    """How many frames to keep from each processed video; -1 uses the project's configured amount."""

    clustering_stride: int
    """How far apart, in frames, to sample before clustering; 1 uses every frame. For ``frames`` this strides the whole
    video, and for ``outliers`` it strides the flagged candidate frames."""

    clustering_resize_width: int
    """The width, in pixels, frames are shrunk to before they are compared for similarity."""

    cluster_in_color: bool
    """Whether frames are compared in color instead of grayscale when selecting them."""

    total_frame_budget: int
    """The total frames to sample the project toward; -1 processes every selected video instead of sampling."""

    random_seed: int | None
    """The seed for the random video sampling; None draws a different subset each run."""

    balance_groups: bool
    """Whether budgeted sampling spreads frames across groups of related videos instead of drawing at random."""

    group_by_pattern: str | None
    """A regex whose first capturing group names each video's group, or None to infer it; implies balance_groups."""

    always_include_videos: tuple[str, ...]
    """The path substrings naming videos to always include in the budgeted sample."""

    path_filters: tuple[str, ...]
    """The path substrings restricting the run to matching videos; an empty tuple selects every candidate video."""

    minimum_progress_interval: float
    """The shortest time, in seconds, between progress updates when the output is not a live terminal."""

    display_progress: bool
    """Whether the aggregate progress bar is shown during extraction."""

    def require_config_path(self) -> Path:
        """Returns the config.yaml path, raising a Click usage error when ``--config-path`` was not supplied.

        The ``extract`` group leaves ``--config-path`` optional so that subcommand help stays reachable without it, so
        each subcommand calls this to enforce the option only when it actually runs.
        """
        if self.config_path is None:
            message = "Missing option '-cp' / '--config-path'."
            raise click.UsageError(message)
        return self.config_path


pass_shared_parameters = click.make_pass_decorator(SharedExtractionParameters)
"""Injects the ``extract`` group's ``SharedExtractionParameters`` as each subcommand's first argument."""


@click.group("extract", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-cp",
    "--config-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The path to the DeepLabCut project's config.yaml for which to extract new or outlier frames.",
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
    "-c",
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
    default=-1,
    show_default=True,
    help="How many frames to keep from each processed video. Set to -1 to use the project's configured amount.",
)
@click.option(
    "-cs",
    "--clustering-stride",
    type=int,
    default=1,
    show_default=True,
    metavar="N",
    help="How far apart, in frames, to sample before clustering. The default of 1 uses every frame, matching "
    "DeepLabCut. For 'frames' this strides the whole video; for 'outliers' it strides the flagged candidate frames. "
    "Raise it to cluster fewer frames and lower memory use, trading coverage for speed.",
)
@click.option(
    "-crw",
    "--clustering-resize-width",
    default=30,
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
    "-tf",
    "--total-frames",
    type=int,
    default=200,
    show_default=True,
    help="The total number of frames you want across the whole project. When set, videos are sampled until this many "
    "frames are reached, so coverage grows over repeated runs (each sampled video contributes --frames-per-video "
    "frames). 'frames' samples not-yet-processed videos; 'outliers' samples analyzed videos, preferring those with no "
    "frames, then those with no outlier frames. Set to -1 to process every selected video instead.",
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
@click.pass_context
def extract_group(
    ctx: click.Context,
    config_path: Path,
    workers: int,
    cores: int,
    frames_per_video: int,
    clustering_stride: int,
    clustering_resize_width: int,
    total_frames: int,
    seed: int | None,
    group_by: str | None,
    always_include: tuple[str, ...],
    path_filter: tuple[str, ...],
    minimum_progress_interval: float,
    *,
    color: bool,
    balance_groups: bool,
    progress: bool,
) -> None:
    """Selects frames from a project's videos, either to bootstrap a model or to refine a trained one.

    ``--config-path`` names the DeepLabCut project's config.yaml both subcommands operate on, alongside the
    parallelism, frame-selection, and frame-budget sampling options common to k-means and outlier extraction (both
    grow the project toward ``--total-frames``, balanced across groups with ``--balance-groups``). Use the ``frames``
    subcommand to bootstrap a project's training frames by clustering raw video, or the ``outliers`` subcommand to
    extract a trained model's likely-wrong frames for refinement. These shared options must be given before the
    subcommand name.
    """
    ctx.obj = SharedExtractionParameters(
        config_path=config_path,
        worker_count=workers,
        cores_per_worker=cores,
        frames_per_video=frames_per_video,
        clustering_stride=clustering_stride,
        clustering_resize_width=clustering_resize_width,
        cluster_in_color=color,
        total_frame_budget=total_frames,
        random_seed=seed,
        balance_groups=balance_groups,
        group_by_pattern=group_by,
        always_include_videos=always_include,
        path_filters=path_filter,
        minimum_progress_interval=minimum_progress_interval,
        display_progress=progress,
    )


@extract_group.command("frames", context_settings=_CONTEXT_SETTINGS)
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
    "permanently removes each selected video's entire labeled-data folder, including its frames and labels, leaving "
    "no empty folder behind. Mutually exclusive with --overwrite.",
)
@pass_shared_parameters
def frames_command(
    shared: SharedExtractionParameters,
    *,
    overwrite: bool,
    reset: bool,
) -> None:
    """Selects training frames from a project's videos by clustering them in parallel.

    Each video is clustered in its own worker process pinned to a disjoint block of CPU cores, and videos that already
    contain extracted frames are skipped unless ``--overwrite`` is given. Passing ``--total-frames`` instead samples a
    random subset of not-yet-processed videos sized to reach that project-wide frame budget, growing coverage across
    repeated runs.
    """
    try:
        summary = extract_frames_kmeans(
            config_path=shared.require_config_path(),
            clustering_stride=shared.clustering_stride,
            worker_count=shared.worker_count,
            cores_per_worker=shared.cores_per_worker,
            frames_per_video=shared.frames_per_video,
            total_frame_budget=shared.total_frame_budget,
            random_seed=shared.random_seed,
            balance_groups=shared.balance_groups,
            group_by_pattern=shared.group_by_pattern,
            always_include_videos=shared.always_include_videos,
            clustering_resize_width=shared.clustering_resize_width,
            cluster_in_color=shared.cluster_in_color,
            overwrite=overwrite,
            reset=reset,
            path_filters=shared.path_filters,
            minimum_progress_interval=shared.minimum_progress_interval,
            display_progress=shared.display_progress,
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


@extract_group.command("outliers", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-v",
    "--video",
    "videos",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    metavar="PATH",
    help="An analyzed video file, or a directory of videos, to refine on. Provide the option several times to "
    "process multiple videos or directories. Omit to refine every analyzed video the project's config.yaml registers.",
)
@click.option(
    "-oa",
    "--outlier-algorithm",
    type=click.Choice([algorithm.value for algorithm in OutlierAlgorithm]),
    default=OutlierAlgorithm.JUMP.value,
    show_default=True,
    help="How likely-wrong frames are flagged. 'jump' flags large frame-to-frame jumps (motion), 'uncertain' flags "
    "low-confidence frames, 'fitting' flags departures from a fitted motion trajectory, and 'list' takes an explicit "
    "frame list.",
)
@click.option(
    "-ea",
    "--extraction-algorithm",
    type=click.Choice([algorithm.value for algorithm in ExtractionAlgorithm]),
    default=ExtractionAlgorithm.KMEANS.value,
    show_default=True,
    help="How the frames to keep are chosen from the flagged candidates.",
)
@click.option(
    "-sh",
    "--shuffle",
    default=1,
    show_default=True,
    help="The shuffle index identifying which trained model produced the predictions.",
)
@click.option(
    "-tsi",
    "--training-set-index",
    default=0,
    show_default=True,
    help="The training-set fraction index identifying which trained model produced the predictions.",
)
@click.option(
    "-pdt",
    "--pixel-distance-threshold",
    default=20.0,
    show_default=True,
    help="How far, in pixels, a bodypart may move (for 'jump') or depart from its fitted trajectory (for 'fitting') "
    "before its frame is flagged.",
)
@click.option(
    "-mc",
    "--minimum-confidence",
    default=0.01,
    show_default=True,
    help="The confidence below which a prediction is treated as unreliable, used by the 'uncertain' and 'fitting' "
    "detectors.",
)
@click.option(
    "-cb",
    "--comparison-bodyparts",
    multiple=True,
    metavar="BODYPART",
    help="A bodypart the detectors consider. Provide the option several times to restrict to several; omit to "
    "consider every bodypart.",
)
@click.option(
    "-fi",
    "--frame-index",
    multiple=True,
    type=int,
    metavar="FRAME",
    help="An explicit frame index to extract when --outlier-algorithm list is used. Provide the option several "
    "times to extract multiple frames.",
)
@click.option(
    "-ad",
    "--autoregressive-degree",
    default=3,
    show_default=True,
    help="How many past frames the 'fitting' detector's motion model uses to predict the next position.",
)
@click.option(
    "-ma",
    "--moving-average-degree",
    default=1,
    show_default=True,
    help="How many past prediction errors the 'fitting' detector's motion model smooths over.",
)
@click.option(
    "-sl",
    "--significance-level",
    default=0.01,
    show_default=True,
    help="The significance level (p-value) for the 'fitting' detector's confidence interval; smaller values are "
    "stricter, flagging only clearer departures from the expected motion.",
)
@click.option(
    "-sv",
    "--save-labeled",
    is_flag=True,
    help="Also save a copy of each extracted frame with the model's predictions drawn on it.",
)
@click.option(
    "-cv",
    "--copy-videos",
    is_flag=True,
    help="Copy newly added videos into the project instead of linking to them in place.",
)
@click.option(
    "-pd",
    "--predictions-directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="The directory holding the analyzed predictions. Omit to look for predictions beside each video.",
)
@click.option(
    "-mp",
    "--model-prefix",
    default="",
    help="The model subdirectory prefix identifying which trained model produced the predictions.",
)
@click.option(
    "-tm",
    "--tracking-method",
    type=click.Choice([method.value for method in TrackingMethod]),
    default=None,
    help="The multi-animal tracker that produced the data. Omit to use the project's setting.",
)
@click.option(
    "-si",
    "--snapshot-index",
    type=int,
    default=None,
    help="The pose snapshot identifying which saved model produced the predictions. Omit to use the default.",
)
@click.option(
    "-dsi",
    "--detector-snapshot-index",
    type=int,
    default=None,
    help="The detector snapshot, for top-down models. Omit to use the default.",
)
@click.option(
    "-ve",
    "--video-extensions",
    multiple=True,
    metavar="EXTENSION",
    help="A file extension used to filter videos found inside a supplied directory. Provide the option several times.",
)
@click.option(
    "-fw",
    "--fit-workers",
    type=int,
    default=-1,
    show_default=True,
    help="How many videos to flag in parallel during 'fitting' detection, the most expensive step. Set to -1 to use "
    "every available core.",
)
@pass_shared_parameters
def outliers_command(
    shared: SharedExtractionParameters,
    videos: tuple[Path, ...],
    outlier_algorithm: str,
    extraction_algorithm: str,
    shuffle: int,
    training_set_index: int,
    pixel_distance_threshold: float,
    minimum_confidence: float,
    comparison_bodyparts: tuple[str, ...],
    frame_index: tuple[int, ...],
    autoregressive_degree: int,
    moving_average_degree: int,
    significance_level: float,
    predictions_directory: Path | None,
    model_prefix: str,
    tracking_method: str | None,
    snapshot_index: int | None,
    detector_snapshot_index: int | None,
    video_extensions: tuple[str, ...],
    fit_workers: int,
    *,
    save_labeled: bool,
    copy_videos: bool,
) -> None:
    """Extracts a trained model's likely-wrong frames from analyzed videos to refine the model.

    Refines on the videos given with ``--video``, or, when none are given, on every analyzed video the project's
    config.yaml registers. Each video must already have been analyzed, since the detectors read the model's predictions
    rather than re-running the model. The flagged outlier frames are clustered in parallel, one video per worker pinned
    to a disjoint block of CPU cores, and added to each video's labeled-data directory alongside the model's predictions
    as machine pre-labels. Outlier extraction is additive, so repeated passes grow the refinement set.

    By default ``--total-frames`` samples the analyzed videos toward a project-wide frame budget, preferring videos
    with no frames, then videos with only raw frames, then videos that already have outlier frames, and balancing
    across groups with ``--balance-groups``. Pass ``--total-frames -1`` to refine every analyzed video instead.
    """
    try:
        summary = extract_outlier_frames_parallel(
            config_path=shared.require_config_path(),
            videos=list(videos),
            shuffle_index=shuffle,
            training_set_index=training_set_index,
            outlier_algorithm=OutlierAlgorithm(outlier_algorithm),
            explicit_frame_indices=frame_index,
            comparison_bodyparts=comparison_bodyparts,
            pixel_distance_threshold=pixel_distance_threshold,
            minimum_confidence=minimum_confidence,
            autoregressive_degree=autoregressive_degree,
            moving_average_degree=moving_average_degree,
            significance_level=significance_level,
            extraction_algorithm=ExtractionAlgorithm(extraction_algorithm),
            candidate_step=shared.clustering_stride,
            frames_per_video=shared.frames_per_video,
            total_frame_budget=shared.total_frame_budget,
            random_seed=shared.random_seed,
            balance_groups=shared.balance_groups,
            group_by_pattern=shared.group_by_pattern,
            always_include_videos=shared.always_include_videos,
            path_filters=shared.path_filters,
            clustering_resize_width=shared.clustering_resize_width,
            cluster_in_color=shared.cluster_in_color,
            save_labeled_frames=save_labeled,
            copy_videos=copy_videos,
            predictions_directory=predictions_directory,
            model_prefix=model_prefix,
            tracking_method=TrackingMethod(tracking_method) if tracking_method is not None else None,
            pose_snapshot_index=snapshot_index,
            detector_snapshot_index=detector_snapshot_index,
            video_extensions=video_extensions,
            worker_count=shared.worker_count,
            cores_per_worker=shared.cores_per_worker,
            fitting_worker_count=fit_workers,
            minimum_progress_interval=shared.minimum_progress_interval,
            display_progress=shared.display_progress,
        )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(message=str(error)) from error

    for video in summary.unanalyzed_videos:
        click.echo(message=f"skipped (not analyzed): {video}", err=True)
    for video, traceback_text in summary.errors:
        click.echo(message=f"\n--- error in {video} ---\n{traceback_text}", err=True)
    click.echo(message=summary.describe())
    if not summary.successful:
        raise SystemExit(1)
