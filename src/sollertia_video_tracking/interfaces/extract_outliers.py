"""Provides the ``slvt extract-outliers`` command that extracts a trained model's likely-wrong frames for refinement."""

from pathlib import Path

import click

from ..frame_extraction import (
    TrackingMethod,
    OutlierAlgorithm,
    ExtractionAlgorithm,
    extract_outlier_frames_parallel,
)

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.command("extract-outliers", context_settings=_CONTEXT_SETTINGS)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("videos", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
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
    "-s",
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
    "-fpv",
    "--frames-per-video",
    type=int,
    default=-1,
    show_default=True,
    help="How many frames to keep from each processed video. Set to -1 to use the project's configured amount.",
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
    "-fw",
    "--fit-workers",
    type=int,
    default=-1,
    show_default=True,
    help="How many videos to flag in parallel during 'fitting' detection, the most expensive step. Set to -1 to use "
    "every available core.",
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
def extract_outliers_command(
    config: Path,
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
    frames_per_video: int,
    clustering_resize_width: int,
    predictions_directory: Path | None,
    model_prefix: str,
    tracking_method: str | None,
    snapshot_index: int | None,
    detector_snapshot_index: int | None,
    video_extensions: tuple[str, ...],
    workers: int,
    cores: int,
    fit_workers: int,
    minimum_progress_interval: float,
    *,
    color: bool,
    save_labeled: bool,
    copy_videos: bool,
    progress: bool,
) -> None:
    """Extracts a trained model's likely-wrong frames from analyzed videos to refine the model.

    CONFIG is the path to the DeepLabCut project's config.yaml, and VIDEOS are the analyzed video files (or directories
    of videos) to refine on. Each video must already have been analyzed, since the detectors read the model's
    predictions rather than re-running the model. The flagged outlier frames are clustered in parallel, one video per
    worker pinned to a disjoint block of CPU cores, and added to each video's labeled-data directory alongside the
    model's predictions as machine pre-labels. Outlier extraction is additive, so repeated passes grow the refinement
    set.
    """
    try:
        summary = extract_outlier_frames_parallel(
            config_path=config,
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
            frames_per_video=frames_per_video,
            clustering_resize_width=clustering_resize_width,
            cluster_in_color=color,
            save_labeled_frames=save_labeled,
            copy_videos=copy_videos,
            predictions_directory=predictions_directory,
            model_prefix=model_prefix,
            tracking_method=TrackingMethod(tracking_method) if tracking_method is not None else None,
            pose_snapshot_index=snapshot_index,
            detector_snapshot_index=detector_snapshot_index,
            video_extensions=video_extensions,
            worker_count=workers,
            cores_per_worker=cores,
            fitting_worker_count=fit_workers,
            minimum_progress_interval=minimum_progress_interval,
            display_progress=progress,
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
