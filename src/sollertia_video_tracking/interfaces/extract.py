"""Provides the ``slvt extract`` command group with its ``frames``, ``outliers``, and ``purge`` subcommands."""

from pathlib import Path
from dataclasses import dataclass

import click

from ..frame_extraction import (
    TrackingMethod,
    OutlierAlgorithm,
    ExtractionAlgorithm,
    purge_labeled_data,
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
    *,
    color: bool,
    progress: bool,
) -> None:
    """Selects or clears frames in a project's videos to bootstrap a model, refine a trained one, or start over.

    ``--config-path`` names the DeepLabCut project's config.yaml every subcommand operates on; the parallelism and
    frame-selection options apply to the ``frames`` and ``outliers`` extraction subcommands. Use ``frames`` to
    bootstrap a project's training frames by clustering raw video, ``outliers`` to extract a trained model's
    likely-wrong frames for refinement, or ``purge`` to delete a video's entire labeled-data folder for a clean start.
    These shared options must be given before the subcommand name.
    """
    ctx.obj = SharedExtractionParameters(
        config_path=config_path,
        worker_count=workers,
        cores_per_worker=cores,
        frames_per_video=frames_per_video,
        clustering_stride=clustering_stride,
        clustering_resize_width=clustering_resize_width,
        cluster_in_color=color,
        display_progress=progress,
    )


@extract_group.command("frames", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-v",
    "--videos",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="PATH",
    help="A project video file that must be included in the sample, chosen before the rest of the --total-frames "
    "budget is filled from the project's other videos. Matched against the videos registered in config.yaml. Provide "
    "the option several times to include several; omit to sample the project at large.",
)
@click.option(
    "-tf",
    "--total-frames",
    type=int,
    default=200,
    show_default=True,
    help="The total number of extracted frames you want across the whole project. When set, not-yet-processed videos "
    "are sampled at random until this many frames are reached, so coverage grows over repeated runs (each sampled "
    "video contributes --frames-per-video frames). Set to -1 to process every selected video instead.",
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
    "-x",
    "--exclusive",
    is_flag=True,
    help="Restrict the run to exactly the --videos files, extracting --frames-per-video frames from each and "
    "ignoring the --total-frames budget and group balancing. Requires --videos.",
)
@click.option(
    "-o",
    "--overwrite",
    is_flag=True,
    help="Re-roll the diverse frame selection for the videos being extracted: clear their unlabeled frames first, then "
    "re-extract. Already-labeled frames and any outlier frames are kept. Mutually exclusive with --reset.",
)
@click.option(
    "-r",
    "--reset",
    is_flag=True,
    help="Re-roll the diverse frame selection across ALL of the project's videos: clear every video's unlabeled frames "
    "first, then re-extract. Already-labeled frames and any outlier frames are kept. To instead wipe a video's labels "
    "and start over, use the 'purge' subcommand. Mutually exclusive with --overwrite.",
)
@pass_shared_parameters
def frames_command(
    shared: SharedExtractionParameters,
    videos: tuple[Path, ...],
    total_frames: int,
    seed: int | None,
    group_by: str | None,
    *,
    balance_groups: bool,
    exclusive: bool,
    overwrite: bool,
    reset: bool,
) -> None:
    """Selects initial training frames from a subset of the project's videos by clustering them in parallel.

    Each video is clustered in its own worker process pinned to a disjoint block of CPU cores, and extraction is
    additive: a run adds freshly clustered frames to whatever a video already holds. Passing ``--overwrite`` re-rolls
    the selection for the videos being extracted, and ``--reset`` re-rolls it across every project video, by clearing
    their unlabeled frames first; both keep already-labeled and outlier frames. Passing ``--total-frames`` samples a
    random subset of not-yet-processed videos sized to reach that project-wide frame budget, growing coverage across
    repeated runs; any videos named with ``--videos`` are always included first, and the remaining budget is filled from
    the project's other videos. Passing ``--exclusive`` with ``--videos`` instead restricts the run to exactly those
    files, extracting ``--frames-per-video`` frames from each and ignoring the budget and group balancing.
    """
    if exclusive and not videos:
        message = "The --exclusive flag requires at least one --videos file to restrict the run to."
        raise click.UsageError(message)
    try:
        summary = extract_frames_kmeans(
            config_path=shared.require_config_path(),
            requested_videos=videos,
            exclusive=exclusive,
            clustering_stride=shared.clustering_stride,
            worker_count=shared.worker_count,
            cores_per_worker=shared.cores_per_worker,
            frames_per_video=shared.frames_per_video,
            total_frame_budget=total_frames,
            random_seed=seed,
            balance_groups=balance_groups,
            group_by_pattern=group_by,
            clustering_resize_width=shared.clustering_resize_width,
            cluster_in_color=shared.cluster_in_color,
            overwrite=overwrite,
            reset=reset,
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
    "-v",
    "--videos",
    multiple=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="PATH",
    help="A project video file, already registered in the project's config.yaml, from which to extract outlier frames "
    "for model refinement. Matched against the videos registered in the project; paths that are not registered are "
    "skipped. Provide the option several times to refine several videos.",
)
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
    "-sh",
    "--shuffle",
    default=1,
    show_default=True,
    help="The shuffle index identifying which trained model produced the predictions. As in the DeepLabCut GUI, this "
    "is the only model-selection option: the training fraction, model prefix, and snapshots are read from the "
    "project's config.yaml.",
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
    "-sv",
    "--save-labeled",
    is_flag=True,
    help="Also save a copy of each extracted frame with the model's predictions drawn on it.",
)
@click.option(
    "-o",
    "--overwrite",
    is_flag=True,
    help="Re-extract outlier frames for the selected --videos, first discarding this refinement iteration's "
    "already-extracted outlier frames for those videos (and their machine labels) so they are replaced rather than "
    "added to. Other videos' outlier frames, and every already-labeled training frame, are left intact. Mutually "
    "exclusive with --reset.",
)
@click.option(
    "-r",
    "--reset",
    is_flag=True,
    help="Discard this refinement iteration's extracted outlier frames across ALL of the project's videos (and their "
    "machine labels) before re-extracting the selected --videos, giving the whole iteration a clean slate. "
    "Already-labeled training frames are preserved. Mutually exclusive with --overwrite.",
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
    overwrite: bool,
    reset: bool,
) -> None:
    """Extracts a trained model's likely-wrong frames from the project's analyzed videos to refine the model.

    Refines on the project videos given with ``--videos``, extracting ``--frames-per-video`` outlier frames from each.
    Each video must already be registered in the project's config.yaml and analyzed, since the detectors read the
    model's predictions rather than re-running the model; requested paths that are not registered project videos are
    skipped with a warning. The flagged outlier frames are clustered in parallel, one video per worker pinned to a
    disjoint block of CPU cores, and added to each video's labeled-data directory alongside the model's predictions as
    machine pre-labels. Outlier extraction is additive by default, so repeated passes grow the refinement set; pass
    ``--overwrite`` to replace the selected videos' outlier frames for the current iteration instead, or ``--reset`` to
    clear the whole iteration's outlier frames before re-extracting. Both preserve every already-labeled training frame.
    """
    try:
        summary = extract_outlier_frames_parallel(
            config_path=shared.require_config_path(),
            videos=list(videos),
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
            overwrite=overwrite,
            reset=reset,
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
    "-v",
    "--videos",
    multiple=True,
    type=click.Path(dir_okay=False, path_type=Path),
    metavar="PATH",
    help="A project video whose entire labeled-data folder to delete, matched against the videos registered in "
    "config.yaml. Provide the option several times to purge several. Mutually exclusive with --all.",
)
@click.option(
    "-a",
    "--all",
    "all_videos",
    is_flag=True,
    help="Purge every video's labeled-data folder in the project. Mutually exclusive with --videos.",
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Actually delete the folders. Without this flag the command only previews what it would remove, deleting "
    "nothing.",
)
@pass_shared_parameters
def purge_command(
    shared: SharedExtractionParameters,
    videos: tuple[Path, ...],
    *,
    all_videos: bool,
    yes: bool,
) -> None:
    """Deletes targeted videos' entire labeled-data folders, including their labels, after a dry-run preview.

    This is the wholesale reset the frame and outlier re-extraction options deliberately avoid: where ``--overwrite``
    and ``--reset`` clear only unlabeled or single-iteration frames and always keep the human labels, ``purge`` removes
    each targeted ``labeled-data`` folder outright. It exists for the rare start-completely-over case, such as changing
    the project crop, that the label-preserving options cannot serve. Target specific videos with ``--videos`` or the
    whole project with ``--all``. The command previews what it would delete and removes nothing until ``--yes`` is
    given.
    """
    try:
        summary = purge_labeled_data(
            config_path=shared.require_config_path(),
            videos=tuple(videos),
            all_videos=all_videos,
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
                f"purged {summary.removed_directory_count} folder(s), {summary.frame_count} frame(s) "
                f"({summary.labeled_directory_count} had labels)"
            )
        )
    else:
        click.echo(
            message=(
                f"dry run: would purge {summary.removed_directory_count} folder(s), {summary.frame_count} frame(s) "
                f"({summary.labeled_directory_count} contain labels). Re-run with --yes to delete."
            )
        )
