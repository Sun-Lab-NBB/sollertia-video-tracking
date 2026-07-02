"""Provides the ``slvt extract-outliers`` command that extracts a trained model's likely-wrong frames for refinement."""

from pathlib import Path

import click

from ..frame_extraction import DEFAULT_RESERVE_CORES, extract_outlier_frames_parallel

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.command("extract-outliers", context_settings=_CONTEXT_SETTINGS)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("videos", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "-a",
    "--algorithm",
    "outlier_algorithm",
    type=click.Choice(["jump", "uncertain", "fitting", "list"]),
    default="jump",
    show_default=True,
    help="The outlier-detection algorithm. 'jump' flags large inter-frame jumps, 'uncertain' flags low-confidence "
    "frames, 'fitting' flags deviations from a fitted SARIMAX trajectory, and 'list' takes an explicit frame list.",
)
@click.option(
    "-e",
    "--extraction",
    "extraction_algorithm",
    type=click.Choice(["kmeans", "uniform"]),
    default="kmeans",
    show_default=True,
    help="The algorithm that selects the extracted frames from the flagged outlier candidates.",
)
@click.option("-s", "--shuffle", default=1, show_default=True, help="The shuffle index whose trained model was used.")
@click.option(
    "--training-set-index",
    "training_set_index",
    default=0,
    show_default=True,
    help="The training-set fraction index.",
)
@click.option(
    "--epsilon",
    default=20.0,
    show_default=True,
    help="The pixel bound for the 'jump' and 'fitting' algorithms.",
)
@click.option(
    "--p-bound",
    "p_bound",
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
    "--frames",
    "frames_to_use",
    multiple=True,
    type=int,
    metavar="FRAME",
    help="An explicit frame index to extract when --algorithm list is used. Provide the option multiple times.",
)
@click.option(
    "--ar-degree",
    "ar_degree",
    default=3,
    show_default=True,
    help="The 'fitting' algorithm's SARIMAX autoregressive degree.",
)
@click.option(
    "--ma-degree",
    "ma_degree",
    default=1,
    show_default=True,
    help="The 'fitting' algorithm's SARIMAX moving-average degree.",
)
@click.option(
    "--alpha",
    default=0.01,
    show_default=True,
    help="The significance level for the 'fitting' algorithm's confidence interval.",
)
@click.option(
    "-n",
    "--num-frames",
    "num_frames",
    type=int,
    default=-1,
    show_default=True,
    help="The number of frames to extract per video, overriding numframes2pick in config.yaml. Set to -1 to use the "
    "value already stored in the configuration file.",
)
@click.option(
    "--cluster-resize-width",
    "cluster_resize_width",
    default=30,
    show_default=True,
    help="The downsample width applied before clustering when selecting with kmeans.",
)
@click.option(
    "--color/--grayscale",
    "cluster_color",
    default=False,
    show_default=True,
    help="Determines whether kmeans selection clusters on color channels instead of grayscale.",
)
@click.option(
    "--save-labeled",
    "save_labeled",
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
    "--destfolder",
    "destfolder",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="The directory holding the analyzed predictions. Omit to look for predictions beside each video.",
)
@click.option("--modelprefix", default="", help="The model subdirectory prefix, matching the trained shuffle.")
@click.option(
    "--track-method",
    "track_method",
    default="",
    help="The multi-animal tracker ('box', 'skeleton', or 'ellipse') that produced the data. Omit to read it from "
    "config.yaml.",
)
@click.option(
    "--snapshot-index",
    "snapshot_index",
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
    "reserve_cores",
    default=DEFAULT_RESERVE_CORES,
    show_default=True,
    help="The number of CPU cores left free for other tasks while extraction runs.",
)
@click.option(
    "--fit-workers",
    "fit_workers",
    type=int,
    default=-1,
    show_default=True,
    help="The number of processes fitting SARIMAX models during 'fitting' detection. Set to -1 to use every usable "
    "core, which scales the expensive fitting path across the whole run.",
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
def extract_outliers_command(
    config: Path,
    videos: tuple[Path, ...],
    outlier_algorithm: str,
    extraction_algorithm: str,
    shuffle: int,
    training_set_index: int,
    epsilon: float,
    p_bound: float,
    comparison_bodyparts: tuple[str, ...],
    frames_to_use: tuple[int, ...],
    ar_degree: int,
    ma_degree: int,
    alpha: float,
    num_frames: int,
    cluster_resize_width: int,
    destfolder: Path | None,
    modelprefix: str,
    track_method: str,
    snapshot_index: int | None,
    detector_snapshot_index: int | None,
    video_extensions: tuple[str, ...],
    workers: int,
    cores_per_worker: int,
    reserve_cores: int,
    fit_workers: int,
    heartbeat: float,
    *,
    cluster_color: bool,
    save_labeled: bool,
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
            shuffle=shuffle,
            training_set_index=training_set_index,
            outlier_algorithm=outlier_algorithm,
            frames_to_use=frames_to_use,
            comparison_bodyparts=comparison_bodyparts,
            epsilon=epsilon,
            p_bound=p_bound,
            ar_degree=ar_degree,
            ma_degree=ma_degree,
            alpha=alpha,
            extraction_algorithm=extraction_algorithm,
            num_frames=num_frames,
            cluster_resize_width=cluster_resize_width,
            cluster_color=cluster_color,
            save_labeled=save_labeled,
            copy_videos=copy_videos,
            destfolder=destfolder,
            modelprefix=modelprefix,
            track_method=track_method,
            snapshot_index=snapshot_index,
            detector_snapshot_index=detector_snapshot_index,
            video_extensions=video_extensions,
            workers=workers,
            cores_per_worker=cores_per_worker,
            reserve_cores=reserve_cores,
            fit_workers=fit_workers,
            heartbeat=heartbeat,
            display_progress=display_progress,
        )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(message=str(error)) from error

    for video in summary.not_analyzed:
        click.echo(message=f"skipped (not analyzed): {video}", err=True)
    for video, traceback_text in summary.errors:
        click.echo(message=f"\n--- error in {video} ---\n{traceback_text}", err=True)
    click.echo(message=summary.describe())
    if not summary.successful:
        raise SystemExit(1)
