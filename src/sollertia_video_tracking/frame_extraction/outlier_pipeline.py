"""Provides the parallel outlier-frame extraction pipeline that refines a DeepLabCut model on likely-wrong frames."""

from __future__ import annotations

import os
import sys
from enum import StrEnum
import math
from typing import TYPE_CHECKING, Any
from pathlib import Path
import traceback
import contextlib
from dataclasses import dataclass
import multiprocessing

import cv2
import numpy as np
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions, frameselectiontools
from deeplabcut.utils.auxfun_videos import collect_video_paths
from deeplabcut.refine_training_dataset import outlier_frames as dlc_outlier_frames

from .progress import make_progress_reporter
from .utilities import (
    iter_pinned_extraction,
    normalize_project_config,
    ensure_unique_video_stems,
    prune_empty_labeled_data_directories,
)
from .frame_reading import make_fast_kmeans_selector
from .cpu_allocation import DEFAULT_RESERVED_CORE_COUNT, plan_core_allocation
from .outlier_detection import (
    KeypointSeries,
    OutlierAlgorithm,
    jump_outlier_indices,
    fit_keypoint_distance,
    fitting_keypoint_series,
    fitting_outlier_indices,
    uncertain_outlier_indices,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_CROP_FIELD_COUNT: int = 4
"""The number of comma-separated integers, ``x1,x2,y1,y2``, in a video's config.yaml crop specification."""


class ExtractionAlgorithm(StrEnum):
    """The supported algorithms for selecting which flagged candidate frames to extract."""

    KMEANS = "kmeans"
    """Clusters the flagged candidates and keeps one representative frame per cluster."""
    UNIFORM = "uniform"
    """Keeps flagged candidates spread uniformly across the flagged range."""


class TrackingMethod(StrEnum):
    """The supported multi-animal trackers that may have produced a video's predictions."""

    BOX = "box"
    """The bounding-box tracker."""
    SKELETON = "skeleton"
    """The skeleton tracker."""
    ELLIPSE = "ellipse"
    """The ellipse tracker."""


@dataclass(frozen=True, slots=True)
class OutlierExtractionSummary:
    """Summarizes the outcome of a parallel outlier-frame extraction run.

    Notes:
        The pipeline never aborts on a single bad video, so per-video problems are collected here rather than raised.
        Videos whose predictions are missing are listed in ``unanalyzed_videos`` (they must be analyzed first), and
        videos that raised during detection or extraction are listed in ``errors`` as ``(video_path, detail)`` pairs.
        Callers inspect ``successful`` to decide the process exit status.
    """

    config_path: Path
    """The path to the DeepLabCut project's config.yaml the run used."""
    outlier_algorithm: OutlierAlgorithm
    """The outlier-detection algorithm that flagged the candidate frames."""
    extraction_algorithm: ExtractionAlgorithm
    """The frame-selection algorithm that picked the extracted frames from the candidates."""
    total_video_count: int
    """The total number of videos considered in the run."""
    extracted_video_count: int
    """The number of videos from which outlier frames were extracted."""
    worker_count: int
    """The number of worker processes that ran concurrently during the extraction phase."""
    used_core_count: int
    """The number of distinct CPU cores the extraction workers were pinned across."""
    total_core_count: int
    """The total number of CPU cores available on the machine."""
    candidate_frame_count: int
    """The total number of putative outlier frames flagged across all videos that had candidates, i.e. the videos
    submitted for extraction."""
    extracted_frame_count: int
    """The total number of frames freshly written into the project's labeled-data tree across all videos."""
    unanalyzed_videos: tuple[str, ...] = ()
    """The videos skipped because no matching predictions were found; they must be analyzed before refinement."""
    errors: tuple[tuple[str, str], ...] = ()
    """The ``(video_path, detail)`` pairs for every video that raised during detection or extraction."""

    @property
    def failed_video_count(self) -> int:
        """Returns the number of videos that raised during detection or extraction."""
        return len(self.errors)

    @property
    def successful(self) -> bool:
        """Returns whether the run completed with every video analyzed and extracted without error."""
        return not self.errors and not self.unanalyzed_videos

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the extraction run for the CLI.

        Returns:
            A compact description of how many videos yielded outlier frames and how many frames were written.
        """
        tail = ""
        if self.unanalyzed_videos:
            tail += f", {len(self.unanalyzed_videos)} not analyzed"
        if self.errors:
            tail += f", {self.failed_video_count} failed"
        return (
            f"{self.outlier_algorithm} outliers: extracted {self.extracted_frame_count} frames from "
            f"{self.extracted_video_count}/{self.total_video_count} videos ({self.candidate_frame_count} candidates) "
            f"on {self.worker_count} workers{tail}"
        )


def extract_outlier_frames_parallel(
    config_path: Path,
    videos: list[str | Path],
    *,
    shuffle_index: int = 1,
    training_set_index: int = 0,
    outlier_algorithm: OutlierAlgorithm = OutlierAlgorithm.JUMP,
    explicit_frame_indices: tuple[int, ...] = (),
    comparison_bodyparts: tuple[str, ...] = (),
    pixel_distance_threshold: float = 20.0,
    minimum_confidence: float = 0.01,
    autoregressive_degree: int = 3,
    moving_average_degree: int = 1,
    significance_level: float = 0.01,
    extraction_algorithm: ExtractionAlgorithm = ExtractionAlgorithm.KMEANS,
    candidate_step: int = 1,
    frames_per_video: int = -1,
    clustering_resize_width: int = 30,
    cluster_in_color: bool = False,
    save_labeled_frames: bool = False,
    copy_videos: bool = False,
    predictions_directory: Path | None = None,
    model_prefix: str = "",
    tracking_method: TrackingMethod | None = None,
    pose_snapshot_index: int | None = None,
    detector_snapshot_index: int | None = None,
    video_extensions: tuple[str, ...] = (),
    worker_count: int = -1,
    cores_per_worker: int = -1,
    reserved_core_count: int = DEFAULT_RESERVED_CORE_COUNT,
    fitting_worker_count: int = -1,
    minimum_progress_interval: float = 30.0,
    display_progress: bool = True,
) -> OutlierExtractionSummary:
    """Flags and extracts a trained model's likely-wrong frames across many analyzed videos in parallel.

    Reads the predictions a trained model already wrote for each video (which must therefore be analyzed first),
    flags putative outlier frames with the chosen algorithm, and pulls a ``numframes2pick`` budget of them into each
    video's ``labeled-data`` directory for correction. The run has two phases: detection loads every video's
    predictions and computes its outlier candidates, fanning the ``fitting`` algorithm's per-keypoint SARIMAX fits out
    across a process pool that spans the whole run; extraction then decodes and selects the frames one video per pinned
    worker. Videos are registered in config.yaml once, single-threaded, before the extraction workers start, so the
    concurrent workers never race on the configuration file. A single bad video is recorded in the returned summary
    rather than aborting the run.

    Notes:
        The pipeline uses the spawn multiprocessing start method on every platform, so a programmatic caller must guard
        the call with ``if __name__ == "__main__":``. The installed console-script entry point is already guarded.
        Outlier extraction is additive: re-running a video appends further frames rather than replacing the existing
        ones, so coverage grows across repeated passes.

        Empty ``labeled-data`` folders left by videos that were registered but never extracted are removed after every
        run, so the labeling GUI shows only the videos that have frames.

    Args:
        config_path: The path to the DeepLabCut project's config.yaml.
        videos: The video files (or directories of videos) to refine on; every video must already be analyzed.
        shuffle_index: The shuffle index whose trained model wrote the predictions.
        training_set_index: The training-set fraction index.
        outlier_algorithm: The detection algorithm: ``"jump"``, ``"uncertain"``, ``"fitting"``, or ``"list"``.
        explicit_frame_indices: The explicit frame indices to extract when ``outlier_algorithm`` is ``"list"``.
        comparison_bodyparts: The bodyparts the detectors consider; an empty tuple considers every bodypart.
        pixel_distance_threshold: The pixel bound for the ``jump`` and ``fitting`` algorithms.
        minimum_confidence: The likelihood bound for the ``uncertain`` algorithm and the ``fitting`` model's
            missing-data mask.
        autoregressive_degree: The autoregressive degree of the ``fitting`` algorithm's SARIMAX model.
        moving_average_degree: The moving-average degree of the ``fitting`` algorithm's SARIMAX model.
        significance_level: The significance level for the ``fitting`` algorithm's confidence interval.
        extraction_algorithm: The frame-selection algorithm applied to the candidates: ``"kmeans"`` or ``"uniform"``.
        candidate_step: The stride at which the flagged candidates are sub-sampled before selection. A value above one
            keeps every ``candidate_step``-th flagged frame, trading coverage for a smaller decode and, when it thins
            the candidates enough, a switch to seeking that avoids decoding the whole frame range.
        frames_per_video: The number of frames to extract per video, overriding ``numframes2pick`` in config.yaml. Set
            to -1 to use the value already stored in the configuration file.
        clustering_resize_width: The downsample width applied before clustering when selecting with ``"kmeans"``.
        cluster_in_color: Determines whether to cluster on color channels instead of grayscale.
        save_labeled_frames: Determines whether to also save each extracted frame with the model's predictions drawn on
            it.
        copy_videos: Determines whether newly added videos are copied into the project rather than symlinked.
        predictions_directory: The directory holding the analyzed predictions, or None to look beside each video.
        model_prefix: The model subdirectory prefix, matching the trained shuffle.
        tracking_method: The multi-animal tracker that produced the predictions, or None to read it from the project
            configuration.
        pose_snapshot_index: The pose snapshot index whose scorer named the prediction files, or None for the default.
        detector_snapshot_index: The detector snapshot index, for top-down models, or None for the default.
        video_extensions: The file extensions used to filter videos found inside a supplied directory.
        worker_count: The number of videos to extract in parallel. Set to -1 to fill the usable cores automatically.
        cores_per_worker: The number of CPU cores pinned to each extraction worker. Set to -1 to spread them evenly.
        reserved_core_count: The number of CPU cores to leave free for other tasks.
        fitting_worker_count: The number of processes fitting SARIMAX models during ``fitting`` detection. Set to -1 to
            use every usable core.
        minimum_progress_interval: The minimum interval, in seconds, between progress lines when the output is not a
            TTY.
        display_progress: Determines whether to render the run header and the aggregate progress bar.

    Returns:
        An OutlierExtractionSummary describing how many videos yielded frames, how many frames were written, and which
        videos were unanalyzed or failed.

    Raises:
        FileNotFoundError: If ``config_path`` does not point to an existing file.
        ValueError: If ``outlier_algorithm`` or ``extraction_algorithm`` is unknown, if ``frames_per_video`` is set
            below one (other than the -1 sentinel), if ``candidate_step`` is below one, if ``outlier_algorithm`` is
            ``"list"`` without ``explicit_frame_indices``, if the comparison bodyparts resolve to none, if no videos
            match the requested selection, or if two selected videos share a file-name stem and would collide in the
            labeled-data tree.
    """
    config_path = config_path.resolve()
    if not config_path.is_file():
        message = f"Unable to extract outlier frames. The config path '{config_path}' does not point to a file."
        raise FileNotFoundError(message)
    try:
        outlier_algorithm = OutlierAlgorithm(outlier_algorithm)
    except ValueError:
        valid_algorithms = ", ".join(algorithm.value for algorithm in OutlierAlgorithm)
        message = (
            f"Unable to extract outlier frames. The outlier algorithm must be one of ({valid_algorithms}), but got "
            f"'{outlier_algorithm}'."
        )
        raise ValueError(message) from None
    try:
        extraction_algorithm = ExtractionAlgorithm(extraction_algorithm)
    except ValueError:
        valid_algorithms = ", ".join(algorithm.value for algorithm in ExtractionAlgorithm)
        message = (
            f"Unable to extract outlier frames. The extraction algorithm must be one of ({valid_algorithms}), but got "
            f"'{extraction_algorithm}'."
        )
        raise ValueError(message) from None
    if outlier_algorithm is OutlierAlgorithm.LIST and not explicit_frame_indices:
        message = "Unable to extract outlier frames. The 'list' algorithm requires an explicit list of frames to use."
        raise ValueError(message)
    if candidate_step < 1:
        message = (
            f"Unable to extract outlier frames. The candidate step must be at least one, but got {candidate_step}."
        )
        raise ValueError(message)

    normalize_project_config(
        config_path, frames_per_video=frames_per_video, error_context="Unable to extract outlier frames."
    )
    configuration = auxiliaryfunctions.read_config(str(config_path))

    resolved_comparison_bodyparts = auxiliaryfunctions.intersection_of_body_parts_and_ones_given_by_user(
        cfg=configuration, comparisonbodyparts=list(comparison_bodyparts) if comparison_bodyparts else "all"
    )
    if not resolved_comparison_bodyparts:
        message = "Unable to extract outlier frames. The requested comparison bodyparts matched none in the project."
        raise ValueError(message)
    resolved_tracking_method = auxfun_multianimal.get_track_method(configuration, track_method=tracking_method or "")
    scorer, _ = auxiliaryfunctions.get_scorer_name(
        configuration,
        shuffle_index,
        trainFraction=configuration["TrainingFraction"][training_set_index],
        modelprefix=model_prefix,
        snapshot_index=pose_snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
    )

    video_paths = collect_video_paths(
        [str(video) for video in videos], extensions=list(video_extensions) if video_extensions else None
    )
    if not video_paths:
        message = "Unable to extract outlier frames. No videos matched the requested selection."
        raise ValueError(message)
    # Two videos that share a stem would write into one labeled-data folder, racing in the parallel extraction pool.
    ensure_unique_video_stems(video_paths, error_context="Unable to extract outlier frames.")

    candidates, unanalyzed_videos, errors = _detect_all_videos(
        video_paths=video_paths,
        predictions_directory=predictions_directory,
        scorer=scorer,
        configuration=configuration,
        tracking_method=resolved_tracking_method,
        resolved_comparison_bodyparts=resolved_comparison_bodyparts,
        outlier_algorithm=outlier_algorithm,
        explicit_frame_indices=explicit_frame_indices,
        pixel_distance_threshold=pixel_distance_threshold,
        minimum_confidence=minimum_confidence,
        autoregressive_degree=autoregressive_degree,
        moving_average_degree=moving_average_degree,
        significance_level=significance_level,
        fitting_worker_count=fitting_worker_count,
        reserved_core_count=reserved_core_count,
        display_progress=display_progress,
    )

    # Sub-sampling the flagged candidates thins the pool the frames are selected from. When it thins a dense pool
    # enough that seeking beats streaming, it avoids decoding the whole frame range, at the cost of coverage.
    if candidate_step > 1:
        candidates = {video: indices[::candidate_step] for video, indices in candidates.items()}

    extraction_videos = [video for video in video_paths if candidates.get(video)]
    if not extraction_videos:
        summary = OutlierExtractionSummary(
            config_path=config_path,
            outlier_algorithm=outlier_algorithm,
            extraction_algorithm=extraction_algorithm,
            total_video_count=len(video_paths),
            extracted_video_count=0,
            worker_count=0,
            used_core_count=0,
            total_core_count=os.cpu_count() or 1,
            candidate_frame_count=0,
            extracted_frame_count=0,
            unanalyzed_videos=tuple(unanalyzed_videos),
            errors=tuple(errors),
        )
    else:
        # Register every extraction video in config.yaml once, single-threaded, before the workers start.
        # DeepLabCut's own frame writer adds each video to the project, which the concurrent workers would otherwise
        # race on; pre-adding here and neutralizing the per-worker add keeps the configuration file writes serialized.
        _register_videos(
            config_path=config_path,
            configuration=configuration,
            videos=extraction_videos,
            copy_videos=copy_videos,
        )
        summary = _extract_all_videos(
            config_path=config_path,
            videos=extraction_videos,
            candidates=candidates,
            predictions_directory=predictions_directory,
            scorer=scorer,
            tracking_method=resolved_tracking_method,
            outlier_algorithm=outlier_algorithm,
            extraction_algorithm=extraction_algorithm,
            clustering_resize_width=clustering_resize_width,
            cluster_in_color=cluster_in_color,
            save_labeled_frames=save_labeled_frames,
            copy_videos=copy_videos,
            worker_count=worker_count,
            cores_per_worker=cores_per_worker,
            reserved_core_count=reserved_core_count,
            minimum_progress_interval=minimum_progress_interval,
            display_progress=display_progress,
            total_video_count=len(video_paths),
            unanalyzed_videos=tuple(unanalyzed_videos),
            detection_errors=errors,
        )

    prune_empty_labeled_data_directories(config_path.parent, display_progress=display_progress)
    return summary


def _detect_all_videos(
    *,
    video_paths: list[str],
    predictions_directory: Path | None,
    scorer: str,
    configuration: dict[str, Any],
    tracking_method: str,
    resolved_comparison_bodyparts: list[str],
    outlier_algorithm: str,
    explicit_frame_indices: tuple[int, ...],
    pixel_distance_threshold: float,
    minimum_confidence: float,
    autoregressive_degree: int,
    moving_average_degree: int,
    significance_level: float,
    fitting_worker_count: int,
    reserved_core_count: int,
    display_progress: bool,
) -> tuple[dict[str, list[int]], list[str], list[tuple[str, str]]]:
    """Computes the outlier candidate frames for every video, parallelizing the SARIMAX fits when fitting.

    Args:
        video_paths: The resolved video paths to detect outliers in.
        predictions_directory: The directory holding the predictions, or None to look beside each video.
        scorer: The DeepLabCut scorer string naming each video's prediction files.
        configuration: The loaded project configuration.
        tracking_method: The resolved multi-animal tracker method.
        resolved_comparison_bodyparts: The comparison bodyparts the detectors consider.
        outlier_algorithm: The detection algorithm to apply.
        explicit_frame_indices: The explicit frames for the ``"list"`` algorithm.
        pixel_distance_threshold: The pixel bound for ``jump`` and ``fitting``.
        minimum_confidence: The likelihood bound for ``uncertain`` and the ``fitting`` missing-data mask.
        autoregressive_degree: The autoregressive degree for ``fitting``.
        moving_average_degree: The moving-average degree for ``fitting``.
        significance_level: The significance level for ``fitting``.
        fitting_worker_count: The number of SARIMAX fit processes, or -1 to use every usable core.
        reserved_core_count: The number of cores to leave free when sizing the fit pool.
        display_progress: Determines whether to report the detection progress line.

    Returns:
        A tuple of the sorted, de-duplicated candidate frames keyed by video, the unanalyzed video paths, and the
        ``(video, detail)`` detection failures.
    """
    candidates: dict[str, list[int]] = {}
    keypoint_series_by_video: dict[str, list[KeypointSeries]] = {}
    unanalyzed_videos: list[str] = []
    errors: list[tuple[str, str]] = []

    for video in video_paths:
        video_predictions_directory = predictions_directory if predictions_directory is not None else Path(video).parent
        # Detection compute is inside the try alongside the load, so a malformed prediction table (empty, missing a
        # likelihood level, or a column count the fitting reshape rejects) is recorded per-video rather than aborting
        # the whole run, upholding the summary's per-video error contract.
        try:
            predictions = _load_sliced_predictions(
                video=video,
                video_predictions_directory=video_predictions_directory,
                scorer=scorer,
                configuration=configuration,
                tracking_method=tracking_method,
            )
            comparison_predictions = predictions.loc[
                :, predictions.columns.get_level_values("bodyparts").isin(resolved_comparison_bodyparts)
            ]
            if outlier_algorithm == "list":
                candidates[video] = sorted({int(frame) for frame in explicit_frame_indices})
            elif outlier_algorithm == "uncertain":
                candidates[video] = uncertain_outlier_indices(comparison_predictions, minimum_confidence)
            elif outlier_algorithm == "jump":
                candidates[video] = jump_outlier_indices(comparison_predictions, pixel_distance_threshold)
            else:
                keypoint_series_by_video[video] = fitting_keypoint_series(comparison_predictions)
        except FileNotFoundError:
            unanalyzed_videos.append(video)
            continue
        except Exception:  # noqa: BLE001 -- a missing or malformed prediction table is recorded, never aborting the rest.
            errors.append((video, "detection error:\n" + traceback.format_exc()))
            continue

    if keypoint_series_by_video:
        candidates.update(
            _detect_fitting_outliers(
                keypoint_series_by_video=keypoint_series_by_video,
                frames_per_video_count=int(configuration["numframes2pick"]),
                pixel_distance_threshold=pixel_distance_threshold,
                minimum_confidence=minimum_confidence,
                autoregressive_degree=autoregressive_degree,
                moving_average_degree=moving_average_degree,
                significance_level=significance_level,
                fitting_worker_count=fitting_worker_count,
                reserved_core_count=reserved_core_count,
                display_progress=display_progress,
            )
        )

    for video, indices in candidates.items():
        candidates[video] = sorted({int(index) for index in indices})
    return candidates, unanalyzed_videos, errors


def _detect_fitting_outliers(
    *,
    keypoint_series_by_video: dict[str, list[KeypointSeries]],
    frames_per_video_count: int,
    pixel_distance_threshold: float,
    minimum_confidence: float,
    autoregressive_degree: int,
    moving_average_degree: int,
    significance_level: float,
    fitting_worker_count: int,
    reserved_core_count: int,
    display_progress: bool,
) -> dict[str, list[int]]:
    """Fits every video's per-keypoint SARIMAX models across one shared pool and reduces them to outlier frames.

    Flattening every video's keypoints into a single pool keeps all usable cores busy even when only a few videos are
    refined, which is the expensive path the fitting algorithm needs to scale on high-core machines.

    Args:
        keypoint_series_by_video: The per-keypoint ``(x, y, likelihood)`` trajectories for each video needing SARIMAX
            fits.
        frames_per_video_count: The project's ``numframes2pick``, used to size the fallback selection.
        pixel_distance_threshold: The averaged-deviation bound above which a frame is flagged.
        minimum_confidence: The likelihood below which a position is treated as missing while fitting.
        autoregressive_degree: The autoregressive degree of the SARIMAX model.
        moving_average_degree: The moving-average degree of the SARIMAX model.
        significance_level: The significance level for the fitted model's confidence interval.
        fitting_worker_count: The number of fit processes, or -1 to use every usable core.
        reserved_core_count: The number of cores to leave free when sizing the pool.
        display_progress: Determines whether to report the number of fits being run.

    Returns:
        The outlier candidate frames keyed by video.
    """
    tasks: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, float, int, int]] = []
    owner_videos: list[str] = []
    for video, keypoint_series in keypoint_series_by_video.items():
        for horizontal_positions, vertical_positions, confidences in keypoint_series:
            tasks.append(
                (
                    horizontal_positions,
                    vertical_positions,
                    confidences,
                    minimum_confidence,
                    significance_level,
                    autoregressive_degree,
                    moving_average_degree,
                )
            )
            owner_videos.append(video)

    total_core_count = os.cpu_count() or 1
    usable_core_count = max(1, total_core_count - max(0, reserved_core_count))
    resolved_fitting_worker_count = usable_core_count if fitting_worker_count < 1 else fitting_worker_count
    resolved_fitting_worker_count = max(1, min(resolved_fitting_worker_count, len(tasks)))
    if display_progress:
        sys.stderr.write(
            f"fitting {len(tasks)} keypoint trajectories across {len(keypoint_series_by_video)} video(s) on "
            f"{resolved_fitting_worker_count} processes\n"
        )
        sys.stderr.flush()

    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=resolved_fitting_worker_count) as pool:
        keypoint_deviations = pool.starmap(fit_keypoint_distance, tasks)

    deviations_by_video: dict[str, list[NDArray[np.float64]]] = {video: [] for video in keypoint_series_by_video}
    for video, deviation in zip(owner_videos, keypoint_deviations, strict=True):
        deviations_by_video[video].append(deviation)

    return {
        video: fitting_outlier_indices(
            video_deviations,
            frames_per_video_count=frames_per_video_count,
            pixel_distance_threshold=pixel_distance_threshold,
        )
        for video, video_deviations in deviations_by_video.items()
    }


def _register_videos(
    config_path: Path,
    configuration: dict[str, Any],
    videos: list[str],
    *,
    copy_videos: bool,
) -> None:
    """Adds any not-yet-registered extraction videos to config.yaml once, before the concurrent workers start.

    Args:
        config_path: The project's config.yaml path.
        configuration: The loaded project configuration, read for the already-registered videos.
        videos: The videos that will be extracted from.
        copy_videos: Determines whether newly added videos are copied into the project rather than symlinked.
    """
    registered = {Path(video).resolve() for video in configuration.get("video_sets", {})}
    for video in videos:
        if Path(video).resolve() in registered:
            continue
        with contextlib.suppress(Exception):
            dlc_outlier_frames.attempt_to_add_video(
                config=str(config_path), video=video, copy_videos=copy_videos, coords=None
            )


def _extract_all_videos(
    *,
    config_path: Path,
    videos: list[str],
    candidates: dict[str, list[int]],
    predictions_directory: Path | None,
    scorer: str,
    tracking_method: str,
    outlier_algorithm: OutlierAlgorithm,
    extraction_algorithm: ExtractionAlgorithm,
    clustering_resize_width: int,
    cluster_in_color: bool,
    save_labeled_frames: bool,
    copy_videos: bool,
    worker_count: int,
    cores_per_worker: int,
    reserved_core_count: int,
    minimum_progress_interval: float,
    display_progress: bool,
    total_video_count: int,
    unanalyzed_videos: tuple[str, ...],
    detection_errors: list[tuple[str, str]],
) -> OutlierExtractionSummary:
    """Decodes and writes the flagged frames one video per pinned worker, then assembles the run summary.

    Args:
        config_path: The resolved project config.yaml path.
        videos: The videos that have outlier candidates to extract.
        candidates: The outlier candidate frames keyed by video.
        predictions_directory: The directory holding the predictions, or None to look beside each video.
        scorer: The DeepLabCut scorer string naming each video's prediction files.
        tracking_method: The resolved multi-animal tracker method.
        outlier_algorithm: The detection algorithm that produced the candidates.
        extraction_algorithm: The frame-selection algorithm applied to the candidates.
        clustering_resize_width: The downsample width for k-means selection.
        cluster_in_color: Determines whether k-means selection clusters on color channels.
        save_labeled_frames: Determines whether to also save each frame with the model's predictions drawn on it.
        copy_videos: Determines whether newly added videos are copied rather than symlinked.
        worker_count: The requested extraction worker count, or -1 to resolve automatically.
        cores_per_worker: The requested cores per worker, or -1 to spread them evenly.
        reserved_core_count: The number of cores to leave free.
        minimum_progress_interval: The minimum interval between progress lines when the output is not a TTY.
        display_progress: Determines whether to render the run header and progress bar.
        total_video_count: The total number of videos considered, for the summary.
        unanalyzed_videos: The unanalyzed videos, for the summary.
        detection_errors: The detection-phase failures, extended with any extraction failures.

    Returns:
        The completed OutlierExtractionSummary.
    """
    total_core_count = os.cpu_count() or 1
    resolved_worker_count, core_sets = plan_core_allocation(
        video_count=len(videos),
        total_core_count=total_core_count,
        worker_count=worker_count,
        cores_per_worker=cores_per_worker,
        reserved_core_count=reserved_core_count,
    )
    used_core_count = len({core for core_set in core_sets for core in core_set})
    frame_totals = {index: max(1, len(candidates[video])) for index, video in enumerate(videos)}
    candidate_frame_count = sum(len(candidates[video]) for video in videos)

    if display_progress:
        _report_plan(
            video_count=len(videos),
            outlier_algorithm=outlier_algorithm,
            extraction_algorithm=extraction_algorithm,
            candidate_frame_count=candidate_frame_count,
            worker_count=resolved_worker_count,
            used_core_count=used_core_count,
            total_core_count=total_core_count,
            config_path=config_path,
        )

    def build_tasks(reporting_queue: Any | None) -> list[tuple[Any, ...]]:
        """Packs one work item per video, embedding the progress queue only when progress is displayed."""
        return [
            (
                video,
                index,
                candidates[video],
                config_path,
                predictions_directory,
                scorer,
                tracking_method,
                extraction_algorithm,
                clustering_resize_width,
                cluster_in_color,
                save_labeled_frames,
                copy_videos,
                reporting_queue,
            )
            for index, video in enumerate(videos)
        ]

    extracted_video_count = 0
    extracted_frame_count = 0
    errors = list(detection_errors)
    unanalyzed_video_list = list(unanalyzed_videos)
    for video, written, status in iter_pinned_extraction(
        videos=videos,
        make_tasks=build_tasks,
        worker=_extract_one_video,
        worker_count=resolved_worker_count,
        core_sets=core_sets,
        frame_totals=frame_totals,
        minimum_progress_interval=minimum_progress_interval,
        display_progress=display_progress,
    ):
        if status == "ok":
            extracted_video_count += 1
            extracted_frame_count += written
        elif status == "not_analyzed":
            unanalyzed_video_list.append(video)
        else:
            errors.append((video, status))

    return OutlierExtractionSummary(
        config_path=config_path,
        outlier_algorithm=outlier_algorithm,
        extraction_algorithm=extraction_algorithm,
        total_video_count=total_video_count,
        extracted_video_count=extracted_video_count,
        worker_count=resolved_worker_count,
        used_core_count=used_core_count,
        total_core_count=total_core_count,
        candidate_frame_count=candidate_frame_count,
        extracted_frame_count=extracted_frame_count,
        unanalyzed_videos=tuple(unanalyzed_video_list),
        errors=tuple(errors),
    )


def _report_plan(
    video_count: int,
    outlier_algorithm: str,
    extraction_algorithm: str,
    candidate_frame_count: int,
    worker_count: int,
    used_core_count: int,
    total_core_count: int,
    config_path: Path,
) -> None:
    """Writes the run header describing the resolved extraction plan to the standard error stream.

    Args:
        video_count: The number of videos with outlier frames to extract.
        outlier_algorithm: The detection algorithm that flagged the candidates.
        extraction_algorithm: The frame-selection algorithm applied to the candidates.
        candidate_frame_count: The total number of flagged candidate frames across the videos.
        worker_count: The resolved number of concurrent workers.
        used_core_count: The number of distinct cores the workers are pinned across.
        total_core_count: The total number of cores on the machine.
        config_path: The resolved path to the project's config.yaml.
    """
    free_core_count = total_core_count - used_core_count
    sys.stderr.write(
        f"outlier extraction | {video_count} videos | detect={outlier_algorithm} | select={extraction_algorithm} | "
        f"{candidate_frame_count:,} candidate frames\n"
    )
    sys.stderr.write(
        f"workers={worker_count} | {used_core_count}/{total_core_count} cores used ({free_core_count} free) | "
        f"config={config_path}\n"
    )
    sys.stderr.flush()


def _load_sliced_predictions(
    video: str,
    video_predictions_directory: Path,
    scorer: str,
    configuration: dict[str, Any],
    tracking_method: str,
) -> Any:
    """Loads a video's predictions, applies the crop offset, and slices them to the configured start/stop window.

    This mirrors DeepLabCut's own preparation in ``extract_outlier_frames`` so the detected frames and the machine
    pre-labels match the upstream tool. It is used in both the detection and extraction phases against the same inputs.

    Args:
        video: The analyzed video path.
        video_predictions_directory: The directory holding the video's prediction files.
        scorer: The DeepLabCut scorer string naming the prediction files.
        configuration: The loaded project configuration, read for the start/stop bounds and the video's crop margins.
        tracking_method: The resolved multi-animal tracker method.

    Returns:
        The prediction table, offset-corrected and sliced to the start/stop window.

    Raises:
        FileNotFoundError: If the video has no matching prediction or metadata files.
        ValueError: If the video's crop specification in config.yaml is not four comma-separated integers, propagated
            from ``_video_cropping_offset``.
    """
    video_stem = Path(video).stem
    predictions, _, _, _ = auxiliaryfunctions.load_analyzed_data(
        folder=str(video_predictions_directory), videoname=video_stem, scorer=scorer, track_method=tracking_method
    )
    metadata = auxiliaryfunctions.load_video_metadata(
        folder=str(video_predictions_directory), videoname=video_stem, scorer=scorer
    )
    frame_count = len(predictions)
    start_index = max(math.floor(frame_count * configuration["start"]), 0)
    stop_index = min(math.ceil(frame_count * configuration["stop"]), frame_count)
    window = np.arange(start_index, stop_index)

    output_crop_x, output_crop_y = _video_cropping_offset(configuration, video)
    if metadata.get("data", {}).get("cropping"):
        x1, _, y1, _ = metadata["data"]["cropping_parameters"]
        predictions.iloc[:, predictions.columns.get_level_values(level="coords") == "x"] += x1 - output_crop_x
        predictions.iloc[:, predictions.columns.get_level_values(level="coords") == "y"] += y1 - output_crop_y
    return predictions.iloc[window]


def _video_cropping_offset(configuration: dict[str, Any], video: str) -> tuple[int, int]:
    """Reads the top-left crop origin config.yaml records for a video, used to undo output cropping on the predictions.

    Args:
        configuration: The loaded project configuration holding the ``video_sets`` crop specifications.
        video: The video path whose crop origin is read.

    Returns:
        The ``(x1, y1)`` crop origin, or ``(0, 0)`` when the video is uncropped or unregistered.

    Raises:
        ValueError: If the video's crop specification is not four comma-separated integers.
    """
    crop = configuration.get("video_sets", {}).get(video, {}).get("crop")
    if crop is None:
        return 0, 0
    parts = [part.strip() for part in str(crop).split(",")]
    if len(parts) != _CROP_FIELD_COUNT:
        message = (
            f"Unable to read the crop for video '{video}'. Expected four comma-separated integers 'x1,x2,y1,y2', but "
            f"got '{crop}'."
        )
        raise ValueError(message)
    x1, _, y1, _ = (int(part) for part in parts)
    return x1, y1


def _extract_one_video(task: tuple[Any, ...]) -> tuple[str, int, str]:
    """Selects and writes the flagged frames for a single video, reusing DeepLabCut's own frame writer.

    DeepLabCut's console output is silenced and exceptions are captured, so one bad video cannot kill the worker pool.
    DeepLabCut's frame writer re-registers the video in config.yaml; that add is neutralized here because the pipeline
    already registered every video single-threaded, so the concurrent workers never write the configuration file.

    Args:
        task: The packed work item carrying the video path, the video index, the candidate frames, the config path,
            the prediction directory, the scorer, the tracker method, the selection settings, and the progress queue.

    Returns:
        A tuple of the video path, the number of frames freshly written, and a status string (``"ok"``,
        ``"not_analyzed"``, or an ``"error:"`` traceback).
    """
    (
        video,
        video_index,
        indices,
        config_path,
        predictions_directory,
        scorer,
        tracking_method,
        extraction_algorithm,
        clustering_resize_width,
        cluster_in_color,
        save_labeled_frames,
        copy_videos,
        progress_queue,
    ) = task
    try:
        cv2.setNumThreads(1)
        configuration = auxiliaryfunctions.read_config(str(config_path))
        video_predictions_directory = predictions_directory if predictions_directory is not None else Path(video).parent
        predictions = _load_sliced_predictions(
            video=video,
            video_predictions_directory=video_predictions_directory,
            scorer=scorer,
            configuration=configuration,
            tracking_method=tracking_method,
        )

        output_directory = Path(configuration["project_path"]) / "labeled-data" / Path(video).stem
        frame_count_before = _count_extracted_frames(output_directory)

        # Swap DeepLabCut's random-seek k-means reader for the streaming one, routing its per-candidate progress to the
        # parent's aggregate bar. The queue is None when progress is disabled, leaving a plain (stream-suppressed) bar.
        # DeepLabCut's per-worker config write is neutralized because the pipeline already registered every video.
        progress_reporter = (
            make_progress_reporter(
                progress_queue=progress_queue, video_index=video_index, frame_total=max(1, len(indices))
            )
            if progress_queue is not None
            else None
        )
        frameselectiontools.KmeansbasedFrameselectioncv2 = make_fast_kmeans_selector(progress=progress_reporter)
        dlc_outlier_frames.attempt_to_add_video = _skip_video_registration

        with (
            Path(os.devnull).open("w") as null_stream,
            contextlib.redirect_stdout(null_stream),
            contextlib.redirect_stderr(null_stream),
        ):
            dlc_outlier_frames.ExtractFramesbasedonPreselection(
                Index=indices,
                extractionalgorithm=extraction_algorithm,
                data=predictions,
                video=video,
                cfg=configuration,
                config=str(config_path),
                opencv=True,
                cluster_resizewidth=clustering_resize_width,
                cluster_color=cluster_in_color,
                savelabeled=save_labeled_frames,
                with_annotations=True,
                copy_videos=copy_videos,
            )
    except FileNotFoundError:
        return video, 0, "not_analyzed"
    except Exception:  # noqa: BLE001 -- one bad video must not kill the pool; the traceback is returned as status.
        return video, 0, "error:\n" + traceback.format_exc()
    else:
        frames_written = _count_extracted_frames(output_directory) - frame_count_before
        return video, max(0, frames_written), "ok"


def _count_extracted_frames(output_directory: Path) -> int:
    """Counts the extracted image frames in a labeled-data directory, ignoring the predicted-label overlays.

    Args:
        output_directory: The ``labeled-data/<video>`` directory whose extracted frames are counted.

    Returns:
        The number of ``img*.png`` frames that are not ``*labeled.png`` prediction overlays.
    """
    if not output_directory.exists():
        return 0
    return sum(1 for frame in output_directory.glob("img*.png") if not frame.stem.endswith("labeled"))


def _skip_video_registration(**_kwargs: Any) -> bool:
    """Neutralizes DeepLabCut's per-video config.yaml registration inside a worker; the pipeline registers up front.

    Returns:
        True, reporting a successful registration so DeepLabCut's frame writer proceeds unchanged.
    """
    return True
