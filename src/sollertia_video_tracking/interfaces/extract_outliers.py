"""Provides the ``slvt extract-outliers`` command that extracts a trained model's likely-wrong frames for refinement."""

from pathlib import Path

import click

from ..frame_extraction import DEFAULT_RESERVED_CORE_COUNT, extract_outlier_frames_parallel

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.command("extract-outliers", context_settings=_CONTEXT_SETTINGS)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("videos", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "-a",
    "--outlier-algorithm",
    "outlier_algorithm",
    type=click.Choice(["jump", "uncertain", "fitting", "list"]),
    default="jump",
    show_default=True,
    help="The outlier-detection algorithm. 'jump' flags large inter-frame jumps, 'uncertain' flags low-confidence "
    "frames, 'fitting' flags deviations from a fitted SARIMAX trajectory, and 'list' takes an explicit frame list.",
)
@click.option(
    "-e",
    "--extraction-algorithm",
    "extraction_algorithm",
    type=click.Choice(["kmeans", "uniform"]),
    default="kmeans",
    show_default=True,
    help="The algorithm that selects the extracted frames from the flagged outlier candidates.",
)
@click.option(
    "-s",
    "--shuffle",
    "shuffle_index",
    default=1,
    show_default=True,
    help="The shuffle index whose trained model was used.",
)
@click.option(
    "--training-set-index",
    "training_set_index",
    default=0,
    show_default=True,
    help="The training-set fraction index.",
)
@click.option(
    "--pixel-distance-threshold",
    "pixel_distance_threshold",
    default=20.0,
    show_default=True,
    help="The pixel bound for the 'jump' and 'fitting' algorithms.",
)
@click.option(
    "--minimum-confidence",
    "minimum_confidence",
    default=0.01,
    show_default=True,
    help="The likelihood bound for the 'uncertain' algorithm and the 'fitting' model's missing-data mask.",
)
@click.option(
    "--comparison-bodyparts",
    "comparison_bodyparts",
    multiple=True,
    metavar="BODYPART",
    help="A bodypart the detectors consider. Provide the option multiple times to restrict to several; omit to "
    "consider every bodypart.",
)
@click.option(
    "--frame-index",
    "explicit_frame_indices",
    multiple=True,
    type=int,
    metavar="FRAME",
    help="An explicit frame index to extract when --outlier-algorithm list is used. Provide the option multiple times.",
)
@click.option(
    "--autoregressive-degree",
    "autoregressive_degree",
    default=3,
    show_default=True,
    help="The 'fitting' algorithm's SARIMAX autoregressive degree.",
)
@click.option(
    "--moving-average-degree",
    "moving_average_degree",
    default=1,
    show_default=True,
    help="The 'fitting' algorithm's SARIMAX moving-average degree.",
)
@click.option(
    "--significance-level",
    "significance_level",
    default=0.01,
    show_default=True,
    help="The significance level for the 'fitting' algorithm's confidence interval.",
)
@click.option(
    "-n",
    "--frames-per-video",
    "frames_per_video",
    type=int,
    default=-1,
    show_default=True,
    help="The number of frames to extract per video, overriding numframes2pick in config.yaml. Set to -1 to use the "
    "value already stored in the configuration file.",
)
@click.option(
    "--clustering-resize-width",
    "clustering_resize_width",
    default=30,
    show_default=True,
    help="The downsample width applied before clustering when selecting with kmeans.",
)
@click.option(
    "--color/--grayscale",
    "cluster_in_color",
    default=False,
    show_default=True,
    help="Determines whether kmeans selection clusters on color channels instead of grayscale.",
)
@click.option(
    "--save-labeled",
    "save_labeled_frames",
    is_flag=True,
    help="Also save each extracted frame with the model's predictions drawn on it.",
)
@click.option(
    "--copy-videos",
    "copy_videos",
    is_flag=True,
    help="Copy newly added videos into the project instead of symlinking them.",
)
@click.option(
    "-d",
    "--predictions-directory",
    "predictions_directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="The directory holding the analyzed predictions. Omit to look for predictions beside each video.",
)
@click.option(
    "--model-prefix",
    "model_prefix",
    default="",
    help="The model subdirectory prefix, matching the trained shuffle.",
)
@click.option(
    "--tracking-method",
    "tracking_method",
    default="",
    help="The multi-animal tracker ('box', 'skeleton', or 'ellipse') that produced the data. Omit to read it from "
    "config.yaml.",
)
@click.option(
    "--snapshot-index",
    "pose_snapshot_index",
    type=int,
    default=None,
    help="The pose snapshot index whose scorer named the prediction files. Omit to use the configured default.",
)
@click.option(
    "--detector-snapshot-index",
    "detector_snapshot_index",
    type=int,
    default=None,
    help="The detector snapshot index, for top-down models. Omit to use the configured default.",
)
@click.option(
    "--video-extensions",
    "video_extensions",
    multiple=True,
    metavar="EXTENSION",
    help="A file extension used to filter videos found inside a supplied directory. Provide the option multiple times.",
)
@click.option(
    "-w",
    "--workers",
    "worker_count",
    type=int,
    default=-1,
    show_default=True,
    help="The number of videos to extract in parallel. Set to -1 to fill the usable CPU cores automatically.",
)
@click.option(
    "-c",
    "--cores-per-worker",
    "cores_per_worker",
    type=int,
    default=-1,
    show_default=True,
    help="The number of CPU cores pinned to each extraction worker. Set to -1 to spread the usable cores evenly.",
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
    "--fit-workers",
    "fitting_worker_count",
    type=int,
    default=-1,
    show_default=True,
    help="The number of processes fitting SARIMAX models during 'fitting' detection. Set to -1 to use every usable "
    "core, which scales the expensive fitting path across the whole run.",
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
def extract_outliers_command(
    config: Path,
    videos: tuple[Path, ...],
    outlier_algorithm: str,
    extraction_algorithm: str,
    shuffle_index: int,
    training_set_index: int,
    pixel_distance_threshold: float,
    minimum_confidence: float,
    comparison_bodyparts: tuple[str, ...],
    explicit_frame_indices: tuple[int, ...],
    autoregressive_degree: int,
    moving_average_degree: int,
    significance_level: float,
    frames_per_video: int,
    clustering_resize_width: int,
    predictions_directory: Path | None,
    model_prefix: str,
    tracking_method: str,
    pose_snapshot_index: int | None,
    detector_snapshot_index: int | None,
    video_extensions: tuple[str, ...],
    worker_count: int,
    cores_per_worker: int,
    reserved_core_count: int,
    fitting_worker_count: int,
    minimum_progress_interval: float,
    *,
    cluster_in_color: bool,
    save_labeled_frames: bool,
    copy_videos: bool,
    display_progress: bool,
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
            shuffle_index=shuffle_index,
            training_set_index=training_set_index,
            outlier_algorithm=outlier_algorithm,
            explicit_frame_indices=explicit_frame_indices,
            comparison_bodyparts=comparison_bodyparts,
            pixel_distance_threshold=pixel_distance_threshold,
            minimum_confidence=minimum_confidence,
            autoregressive_degree=autoregressive_degree,
            moving_average_degree=moving_average_degree,
            significance_level=significance_level,
            extraction_algorithm=extraction_algorithm,
            frames_per_video=frames_per_video,
            clustering_resize_width=clustering_resize_width,
            cluster_in_color=cluster_in_color,
            save_labeled_frames=save_labeled_frames,
            copy_videos=copy_videos,
            predictions_directory=predictions_directory,
            model_prefix=model_prefix,
            tracking_method=tracking_method,
            pose_snapshot_index=pose_snapshot_index,
            detector_snapshot_index=detector_snapshot_index,
            video_extensions=video_extensions,
            worker_count=worker_count,
            cores_per_worker=cores_per_worker,
            reserved_core_count=reserved_core_count,
            fitting_worker_count=fitting_worker_count,
            minimum_progress_interval=minimum_progress_interval,
            display_progress=display_progress,
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
