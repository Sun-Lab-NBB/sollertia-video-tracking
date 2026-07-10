"""Provides the ``slvt extract`` group and its ``frames``, ``outliers``, ``purge``, and ``pending`` subcommands."""

from pathlib import Path
from dataclasses import dataclass

import click

from ..frame_extraction import (
    TrackingMethod,
    OutlierAlgorithm,
    ExtractionAlgorithm,
    purge_labeled_data,
    extract_frames_kmeans,
    summarize_refinement_status,
    extract_outlier_frames_parallel,
)

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@dataclass(frozen=True, slots=True)
class _SharedExtractionParameters:
    """Bundles the options parsed on the ``extract`` group and shared across its ``frames``, ``outliers``, ``purge``,
    and ``pending`` subcommands.

    The ``extract`` group callback builds one of these from its options and stores it on the Click context, and each
    subcommand reads it back through the ``_pass_shared_parameters`` decorator. ``purge`` and ``pending`` use only
    ``config_path`` and ``videos``. The clustering and parallelism fields apply to ``frames`` and ``outliers``.
    """

    config_path: Path | None
    """The path to the DeepLabCut project's config.yaml every subcommand operates on.

    ``--config-path`` cannot be marked required on the group without also blocking ``slvt extract SUBCOMMAND --help``,
    so it is validated per subcommand through ``require_config_path`` instead of at parse time, leaving it None
    here when it was omitted.
    """

    worker_count: int
    """How many videos to process at the same time; -1 uses the available CPU cores automatically."""

    cores_per_worker: int
    """How many CPU cores to devote to each processed video; -1 gives each worker a saturating block when --workers is
    automatic, or splits the usable cores evenly across an explicit --workers count."""

    frames_per_video: int
    """How many frames to keep from each processed video; -1 uses the project's configured amount."""

    clustering_stride: int
    """How far apart, in frames, to sample before clustering; 1 uses every frame. For ``frames`` this strides the whole
    video, and for ``outliers`` it strides the flagged candidate frames."""

    clustering_resize_width: int
    """The width, in pixels, frames are shrunk to before they are compared for similarity."""

    cluster_in_color: bool
    """Determines whether frames are compared in color instead of grayscale when selecting them."""

    display_progress: bool
    """Determines whether the aggregate progress bar is shown during extraction."""

    videos: tuple[Path, ...]
    """The project video files the subcommand targets. How they are used depends on the subcommand. An empty tuple
    means the whole project wherever that is allowed."""

    overwrite: bool
    """Determines whether to re-roll the selected videos, clearing their removable frames first instead of topping
    them up."""

    reset: bool
    """Determines whether to re-roll across the whole project, clearing removable frames first instead of topping up."""

    def require_config_path(self) -> Path:
        """Returns the config.yaml path, raising a Click usage error when ``--config-path`` was not supplied.

        The ``extract`` group leaves ``--config-path`` optional so that subcommand help stays reachable without it, so
        each subcommand calls this to enforce the option only when it actually runs.
        """
        if self.config_path is None:
            message = "Missing option '-cfg' / '--config-path'."
            raise click.UsageError(message=message)
        return self.config_path


_pass_shared_parameters = click.make_pass_decorator(_SharedExtractionParameters)
"""Injects the ``extract`` group's ``_SharedExtractionParameters`` as each subcommand's first argument."""


@click.group("extract", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-cfg",
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
    help="How many videos to process at the same time. Set to -1 to launch about one worker per four usable cores, "
    "capped at the number of videos.",
)
@click.option(
    "-c",
    "--cores",
    type=int,
    default=-1,
    show_default=True,
    help="How many CPU cores to devote to each video being processed. Set to -1 to give each worker a saturating block "
    "(about four cores) when --workers is automatic, or split the usable cores evenly across an explicit --workers "
    "count. Each video keeps about four cores busy while it decodes, so larger values mostly idle cores and smaller "
    "values slow each video worker down.",
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
    help="How far apart, in frames, to sample before clustering. The default of 1 uses every frame. For 'frames' this "
    "strides the whole video; for 'outliers' it strides the flagged candidate frames. Raise it to cluster fewer frames "
    "trading coverage for processing speed.",
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
    "-co",
    "--color/--grayscale",
    default=False,
    show_default=True,
    help="Determines whether frames are compared in color instead of grayscale when selecting them.",
)
@click.option(
    "-p",
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Determines whether the aggregate progress bar is shown during extraction.",
)
@click.option(
    "-v",
    "--videos",
    multiple=True,
    type=click.Path(dir_okay=False, path_type=Path),
    metavar="PATH",
    help="A project video, registered in config.yaml, that the subcommand targets. For 'frames' it is included in "
    "the sample first, or is one of the only videos processed with --exclusive. For 'outliers' it is a video to "
    "refine. For 'purge' its labeled-data directory is removed, and for 'pending' its directory is inspected. Provide "
    "the option several times for several videos. Omit --videos to target the whole project: every video for 'frames', "
    "'purge', and 'pending', and every video the current iteration model has analyzed for 'outliers'.",
)
@click.option(
    "-o",
    "--overwrite",
    is_flag=True,
    help="Re-roll the videos this run processes instead of topping them up, clearing each one's removable frames "
    "first. For 'frames' it clears each selected video's unlabeled bootstrap frames, refused for any already in "
    "outlier refinement. For 'outliers' it clears the current refinement iteration's outlier frames for the videos "
    "being refined. Mutually exclusive with --reset.",
)
@click.option(
    "-r",
    "--reset",
    is_flag=True,
    help="Re-roll across the whole project instead of topping up, clearing the relevant frames first. For 'frames' "
    "this clears every not-yet-refined video's unlabeled bootstrap frames and leaves videos already in outlier "
    "refinement untouched. For 'outliers' it clears the current iteration's outlier frames for every video. Mutually "
    "exclusive with --overwrite.",
)
@click.pass_context
def extract_group(
    ctx: click.Context,
    config_path: Path | None,
    workers: int,
    cores: int,
    frames_per_video: int,
    clustering_stride: int,
    clustering_resize_width: int,
    videos: tuple[Path, ...],
    *,
    color: bool,
    progress: bool,
    overwrite: bool,
    reset: bool,
) -> None:
    """Selects or clears frames in a project's videos to bootstrap a model, refine a trained one, or start over.

    ``--config-path`` names the DeepLabCut project's config.yaml every subcommand operates on. Use ``frames`` to
    bootstrap a project's training frames by clustering raw video, ``outliers`` to extract a trained model's
    likely-wrong frames for refinement, or ``purge`` to delete a video's entire labeled-data directory for a clean
    start. Use ``pending`` to list the directories whose machine-labeled frames still await your refinement. The
    parallelism and clustering options apply to ``frames`` and ``outliers``, while ``--videos``, ``--overwrite``, and
    ``--reset`` are shared by the subcommands that accept them. All of these shared options must be given before the
    subcommand name.
    """
    if overwrite and reset:
        message = "The --overwrite and --reset options are mutually exclusive."
        raise click.UsageError(message=message)
    ctx.obj = _SharedExtractionParameters(
        config_path=config_path,
        worker_count=workers,
        cores_per_worker=cores,
        frames_per_video=frames_per_video,
        clustering_stride=clustering_stride,
        clustering_resize_width=clustering_resize_width,
        cluster_in_color=color,
        display_progress=progress,
        videos=videos,
        overwrite=overwrite,
        reset=reset,
    )


@extract_group.command("frames", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-tf",
    "--total-frames",
    type=int,
    default=200,
    show_default=True,
    help="The total number of extracted frames you want across the whole project. When set, videos are selected until "
    "this many frames are reached, preferring not-yet-extracted videos and falling back to below-ceiling ones; each "
    "is topped up to --frames-per-video frames (a fresh video by a full set, a partly-extracted one by its remainder). "
    "Set to -1 to top up every below-ceiling selected video instead.",
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
    "-gr",
    "--group-regex",
    default=None,
    metavar="REGEX",
    help="A regular expression used to derive group names for each video, for naming schemes the "
    "automatic grouping does not recognize (for example '(Grp\\d+)' for names like D1_Grp2). Implies --balance-groups.",
)
@click.option(
    "-e",
    "--exclusive",
    is_flag=True,
    help="Restrict the run to exactly the --videos files, topping each up to --frames-per-video frames and ignoring "
    "the --total-frames budget and group balancing. A video already at that count is skipped unless --overwrite "
    "re-rolls it. Requires --videos.",
)
@_pass_shared_parameters
def frames_command(
    shared: _SharedExtractionParameters,
    total_frames: int,
    group_regex: str | None,
    *,
    balance_groups: bool,
    exclusive: bool,
) -> None:
    """Selects initial training frames from a subset of the project's videos by clustering them in parallel.

    Each video is clustered in its own worker process pinned to a disjoint block of CPU cores. ``--frames-per-video``
    is a per-video ceiling: each selected video is topped up to it, so a not-yet-extracted video gains a full set and a
    partly-extracted one gains only the frames that reach the ceiling. Passing ``--total-frames`` selects just enough
    videos to reach that project-wide total, preferring not-yet-extracted videos and falling back to below-ceiling
    ones. Any videos named with ``--videos`` are included first. If even topping every eligible video to the ceiling
    cannot reach the total in one pass, the run reports the shortfall and stops, so raise ``--frames-per-video``, lower
    ``--total-frames``, or register more videos. Passing ``--exclusive`` with ``--videos`` instead restricts the run to
    exactly those files, topping each up to ``--frames-per-video`` and ignoring the budget and group balancing. Passing
    ``--overwrite`` clears the selected videos' unlabeled frames first so they are re-rolled from scratch (refused for
    any already in outlier refinement), and ``--reset`` does the same across every not-yet-refined project video; both
    keep already-labeled and outlier frames.
    """
    if exclusive and not shared.videos:
        message = "The --exclusive flag requires at least one --videos file to restrict the run to."
        raise click.UsageError(message=message)
    try:
        summary = extract_frames_kmeans(
            config_path=shared.require_config_path(),
            requested_videos=shared.videos,
            exclusive=exclusive,
            clustering_stride=shared.clustering_stride,
            worker_count=shared.worker_count,
            cores_per_worker=shared.cores_per_worker,
            frames_per_video=shared.frames_per_video,
            total_frame_budget=total_frames,
            balance_groups=balance_groups,
            group_by_pattern=group_regex,
            clustering_resize_width=shared.clustering_resize_width,
            cluster_in_color=shared.cluster_in_color,
            overwrite=shared.overwrite,
            reset=shared.reset,
            display_progress=shared.display_progress,
        )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(message=str(error)) from error

    for video, traceback_text in summary.errors:
        click.echo(message=f"\n--- error in {video} ---\n{traceback_text}", err=True)
    click.echo(
        message=(
            f"done: {summary.extracted_video_count} extracted, {summary.cleared_frame_count} frames cleared, "
            f"{summary.failed_video_count} failed (of {summary.total_video_count})"
        ),
    )
    if not summary.successful:
        raise SystemExit(1)


@extract_group.command("outliers", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-oa",
    "--outlier-algorithm",
    type=click.Choice([algorithm.value for algorithm in OutlierAlgorithm]),
    default=OutlierAlgorithm.UNCERTAIN.value,
    show_default=True,
    help="How likely-wrong frames are identified. 'jump' flags large frame-to-frame jumps (motion), 'uncertain' flags "
    "low-confidence frames, 'fitting' flags departures from a fitted motion trajectory, and 'list' takes an explicit "
    "frame list.",
)
@click.option(
    "-ea",
    "--extraction-algorithm",
    type=click.Choice([algorithm.value for algorithm in ExtractionAlgorithm]),
    default=ExtractionAlgorithm.KMEANS.value,
    show_default=True,
    help="How the frames to keep are chosen from the identified candidates.",
)
@click.option(
    "-s",
    "--shuffle",
    default=1,
    show_default=True,
    help="The shuffle index whose trained model produced the predictions. It is the only model-selection option; "
    "everything else is taken from the project configuration.",
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
    help="The confidence below which a prediction is treated as unreliable: 'uncertain' flags a frame holding one, and "
    "'fitting' excludes it from driving the fitted trajectory.",
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
    "-mad",
    "--moving-average-degree",
    default=1,
    show_default=True,
    help="How many past prediction errors the 'fitting' detector's motion model smooths over.",
)
@click.option(
    "-sl",
    "--save-labeled",
    is_flag=True,
    help="Determines whether to also save a copy of each extracted frame with the model's predictions drawn on it.",
)
@click.option(
    "-tm",
    "--tracking-method",
    type=click.Choice([method.value for method in TrackingMethod]),
    default=None,
    help="The multi-animal tracker that produced the data. Omit to use the project's setting.",
)
@click.option(
    "-fw",
    "--fit-workers",
    type=int,
    default=-1,
    show_default=True,
    help="How many SARIMAX keypoint-trajectory fits to run in parallel during 'fitting' detection, the most expensive "
    "step. Set to -1 to use every available core.",
)
@_pass_shared_parameters
def outliers_command(
    shared: _SharedExtractionParameters,
    outlier_algorithm: str,
    extraction_algorithm: str,
    shuffle: int,
    pixel_distance_threshold: float,
    minimum_confidence: float,
    comparison_bodyparts: tuple[str, ...],
    frame_index: tuple[int, ...],
    autoregressive_degree: int,
    moving_average_degree: int,
    tracking_method: str | None,
    fit_workers: int,
    *,
    save_labeled: bool,
) -> None:
    """Extracts a trained model's likely-wrong frames from the project's analyzed videos to refine the model.

    Refines on the project videos given with ``--videos``, extracting ``--frames-per-video`` outlier frames from each.
    Omit ``--videos`` to refine every registered video the current model has already analyzed. Each named video must be
    registered in the project's config.yaml and analyzed, since the detectors read the model's predictions rather than
    re-running the model. Requested paths that are not registered project videos are skipped with a warning. The flagged
    outlier frames are clustered in parallel, one video per worker pinned to a disjoint block of CPU cores, and added to
    each video's labeled-data directory alongside the model's predictions as machine pre-labels. Outlier extraction is
    additive by default, so repeated passes grow the refinement set. Pass ``--overwrite`` to replace the refined
    videos' outlier frames for the current iteration, or ``--reset`` to clear the whole iteration's outlier frames
    before re-extracting. Both preserve every already-labeled training frame.
    """
    try:
        summary = extract_outlier_frames_parallel(
            config_path=shared.require_config_path(),
            videos=list(shared.videos),
            shuffle_index=shuffle,
            outlier_algorithm=OutlierAlgorithm(outlier_algorithm),
            explicit_frame_indices=frame_index,
            comparison_bodyparts=comparison_bodyparts,
            pixel_distance_threshold=pixel_distance_threshold,
            minimum_confidence=minimum_confidence,
            autoregressive_degree=autoregressive_degree,
            moving_average_degree=moving_average_degree,
            extraction_algorithm=ExtractionAlgorithm(extraction_algorithm),
            candidate_step=shared.clustering_stride,
            frames_per_video=shared.frames_per_video,
            clustering_resize_width=shared.clustering_resize_width,
            cluster_in_color=shared.cluster_in_color,
            save_labeled_frames=save_labeled,
            tracking_method=TrackingMethod(tracking_method) if tracking_method is not None else None,
            worker_count=shared.worker_count,
            cores_per_worker=shared.cores_per_worker,
            fitting_worker_count=fit_workers,
            overwrite=shared.overwrite,
            reset=shared.reset,
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


@extract_group.command("purge", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Actually delete the directories. Without this flag the command only previews what it would remove, deleting "
    "nothing.",
)
@_pass_shared_parameters
def purge_command(
    shared: _SharedExtractionParameters,
    *,
    yes: bool,
) -> None:
    """Deletes targeted videos' entire labeled-data directories, including their labels, after a dry-run preview.

    This is the wholesale reset the frame and outlier re-extraction options deliberately avoid: where ``--overwrite``
    and ``--reset`` clear only unlabeled or single-iteration frames and always keep the human labels, ``purge`` removes
    each targeted ``labeled-data`` directory outright. It exists for the rare start-completely-over case, such as
    changing the project crop, that the label-preserving options cannot serve. Target specific videos with ``--videos``,
    or omit ``--videos`` to purge the whole project. The command previews what it would delete and removes nothing until
    ``--yes`` is given.
    """
    try:
        summary = purge_labeled_data(
            config_path=shared.require_config_path(),
            videos=tuple(shared.videos),
            execute=yes,
        )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(message=str(error)) from error

    for video in summary.unmatched_videos:
        click.echo(message=f"skipped (not a registered project video): {video}", err=True)
    for directory in summary.removed_directories:
        label_marker = " [has labels]" if directory in summary.labeled_directories else ""
        verb = "removed" if summary.executed else "would remove"
        click.echo(message=f"{verb}: {directory}{label_marker}")
    if summary.executed:
        click.echo(
            message=(
                f"purged {summary.removed_directory_count} directory(ies), {summary.frame_count} frame(s) "
                f"({summary.labeled_directory_count} had labels)"
            )
        )
    else:
        click.echo(
            message=(
                f"dry run: would purge {summary.removed_directory_count} directory(ies), {summary.frame_count} "
                f"frame(s) ({summary.labeled_directory_count} contain labels). Re-run with --yes to delete."
            )
        )


@extract_group.command("pending", context_settings=_CONTEXT_SETTINGS)
@_pass_shared_parameters
def pending_command(shared: _SharedExtractionParameters) -> None:
    """Lists the video directories that still hold machine-labeled frames you have not refined for the current
    iteration.

    After ``outliers`` extracts a trained model's likely-wrong frames, each is saved as a machine pre-label that you
    refine in the labeling GUI. This reports each video directory that still has unrefined machine frames, and how many,
    so you know which directories to open next. It only reads the project, changing nothing. Target specific videos with
    ``--videos``, or omit ``--videos`` to scan the whole project.
    """
    try:
        summary = summarize_refinement_status(
            config_path=shared.require_config_path(),
            videos=tuple(shared.videos),
        )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(message=str(error)) from error

    for video in summary.unmatched_videos:
        click.echo(message=f"skipped (not a registered project video): {video}", err=True)
    for directory, detail in summary.unreadable:
        click.echo(message=f"warning: could not read {directory}: {detail}", err=True)

    project_directory = summary.config_path.parent
    for status in sorted(summary.pending_directories, key=lambda status: status.directory):
        location = (
            status.directory.relative_to(project_directory)
            if status.directory.is_relative_to(project_directory)
            else status.directory
        )
        click.echo(message=f"{location}   {status.unrefined_frame_count} frame(s) to refine")
    click.echo(message=summary.describe())
    if not summary.successful:
        raise SystemExit(1)
