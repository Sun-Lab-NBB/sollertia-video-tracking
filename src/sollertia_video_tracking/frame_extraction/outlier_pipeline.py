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
import pandas as pd
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions, frameselectiontools
from deeplabcut.refine_training_dataset import outlier_frames as dlc_outlier_frames

from .progress import make_progress_reporter
from .utilities import (
    extracted_frame_paths,
    frame_names_from_index,
    iter_pinned_extraction,
    normalize_project_config,
    select_registered_videos,
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
    """Defines the supported algorithms for selecting which flagged candidate frames to extract."""

    KMEANS = "kmeans"
    """Clusters the flagged candidates and keeps one representative frame per cluster."""
    UNIFORM = "uniform"
    """Keeps flagged candidates spread uniformly across the flagged range."""


class TrackingMethod(StrEnum):
    """Defines the supported multi-animal trackers that may have produced a video's predictions."""

    BOX = "box"
    """Identifies the bounding-box tracker."""
    SKELETON = "skeleton"
    """Identifies the skeleton tracker."""
    ELLIPSE = "ellipse"
    """Identifies the ellipse tracker."""


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
    """The number of candidate frames submitted for extraction across all videos that had candidates, after any
    ``candidate_step`` sub-sampling of the flagged frames."""
    extracted_frame_count: int
    """The total number of frames freshly written into the project's labeled-data tree across all videos."""
    unanalyzed_videos: tuple[str, ...] = ()
    """The videos skipped because no matching predictions were found. They must be analyzed before refinement."""
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
    extraction_algorithm: ExtractionAlgorithm = ExtractionAlgorithm.KMEANS,
    candidate_step: int = 1,
    frames_per_video: int = -1,
    clustering_resize_width: int = 30,
    cluster_in_color: bool = False,
    save_labeled_frames: bool = False,
    tracking_method: TrackingMethod | None = None,
    pose_snapshot_index: int | None = None,
    detector_snapshot_index: int | None = None,
    worker_count: int = -1,
    cores_per_worker: int = -1,
    reserved_core_count: int = DEFAULT_RESERVED_CORE_COUNT,
    fitting_worker_count: int = -1,
    overwrite: bool = False,
    reset: bool = False,
    display_progress: bool = True,
) -> OutlierExtractionSummary:
    """Flags and extracts a trained model's likely-wrong frames across many analyzed videos in parallel.

    Reads the predictions a trained model already wrote for each video (which must therefore be analyzed first),
    flags putative outlier frames with the chosen algorithm, and pulls a ``numframes2pick`` budget of them into each
    video's ``labeled-data`` directory for correction. The run has two phases. Detection loads every video's
    predictions and computes its outlier candidates, fanning the ``fitting`` algorithm's per-keypoint SARIMAX fits out
    across a process pool that spans the whole run. Extraction then decodes and selects the frames one video per pinned
    worker. Only videos already registered in the project's config.yaml are refined, matched by resolved path like the
    k-means extractor, so the workers only ever read the configuration file and never race on writing it. A single bad
    video is recorded in the returned summary rather than aborting the run.

    Notes:
        The pipeline uses the spawn multiprocessing start method on every platform, so a programmatic caller must guard
        the call with ``if __name__ == "__main__":``. The installed console-script entry point is already guarded.
        Outlier extraction is additive by default: re-running a video appends further frames rather than replacing the
        existing ones, so coverage grows across repeated passes. Setting ``overwrite`` first discards the current
        refinement iteration's already-extracted outlier frames for the selected videos so they are replaced instead of
        added to, and ``reset`` discards the iteration's outlier frames across every project video for a clean slate.
        Both clear only this iteration's freshly extracted outlier frames (those recorded in the iteration's machine
        labels) and preserve every frame already carried in the human ``CollectedData`` labels.

        Empty ``labeled-data`` directories left by videos that were registered but never extracted are removed after
        every run, so the labeling GUI shows only the videos that have frames.

    Args:
        config_path: The path to the DeepLabCut project's config.yaml.
        videos: The project video files to refine on. Each must already be registered in the project's config.yaml
            video_sets and analyzed. Requested paths that match no registered video are skipped with a warning. Leave
            empty to refine every registered video the current model has already analyzed.
        shuffle_index: The shuffle index whose trained model wrote the predictions.
        training_set_index: The training-set fraction index.
        outlier_algorithm: The detection algorithm: ``"jump"``, ``"uncertain"``, ``"fitting"``, or ``"list"``.
        explicit_frame_indices: The explicit frame indices to extract when ``outlier_algorithm`` is ``"list"``.
        comparison_bodyparts: The bodyparts the detectors consider. An empty tuple considers every bodypart.
        pixel_distance_threshold: The pixel bound for the ``jump`` and ``fitting`` algorithms.
        minimum_confidence: The likelihood bound for the ``uncertain`` algorithm and the ``fitting`` model's
            missing-data mask.
        autoregressive_degree: The autoregressive degree of the ``fitting`` algorithm's SARIMAX model.
        moving_average_degree: The moving-average degree of the ``fitting`` algorithm's SARIMAX model.
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
        tracking_method: The multi-animal tracker that produced the predictions, or None to read it from the project
            configuration.
        pose_snapshot_index: The pose snapshot index whose scorer named the prediction files, or None for the default.
        detector_snapshot_index: The detector snapshot index, for top-down models, or None for the default.
        worker_count: The number of videos to extract in parallel. Set to -1 to fill the usable cores automatically.
        cores_per_worker: The number of CPU cores pinned to each extraction worker. Set to -1 to give each worker a
            saturating core block when the worker count is automatic, or to split the usable cores evenly across an
            explicit worker count.
        reserved_core_count: The number of CPU cores to leave free for other tasks.
        fitting_worker_count: The number of processes fitting SARIMAX models during ``fitting`` detection. Set to -1 to
            use every usable core.
        overwrite: Determines whether to re-extract the videos this run refines, first discarding this refinement
            iteration's already-extracted outlier frames for those videos, along with their machine labels and any
            manual refinement of them, so the frames are replaced rather than added to. Other videos' outlier frames are
            left intact, and every frame already carried in the human ``CollectedData`` labels is preserved. Mutually
            exclusive with ``reset``.
        reset: Determines whether to discard this refinement iteration's extracted outlier frames across every project
            video, along with their machine labels and any manual refinement, before re-extracting the selected videos,
            giving the whole iteration a clean slate. Every frame already carried in the human ``CollectedData`` labels
            is preserved. Mutually exclusive with ``overwrite``.
        display_progress: Determines whether to render the run header and the aggregate progress bar.

    Returns:
        An OutlierExtractionSummary describing how many videos yielded frames, how many frames were written, and which
        videos were unanalyzed or failed.

    Raises:
        FileNotFoundError: If ``config_path`` does not point to an existing file.
        ValueError: Raised when the options conflict: ``overwrite`` and ``reset`` are both set. Raised when an argument
            is invalid: ``outlier_algorithm`` or ``extraction_algorithm`` is unknown, ``frames_per_video`` (other than
            the -1 sentinel) or ``candidate_step`` is below one, or ``outlier_algorithm`` is ``"list"`` without
            ``explicit_frame_indices``. Raised when the comparison bodyparts resolve to none. Raised when no videos can
            be refined: the project lists none in ``video_sets``, no requested video matches a registered one, or no
            videos are named and the current model has analyzed none. Raised when two selected videos share a file-name
            stem and would collide in the labeled-data tree. Raised when the ``fitting`` algorithm is selected but the
            project's ``numframes2pick`` is missing or not a positive integer and no ``frames_per_video`` override is
            supplied. Raised when an explicit ``worker_count`` or ``cores_per_worker``, or their product, needs more
            cores than remain usable after reserving ``reserved_core_count``. This check runs after detection completes,
            so a ``fitting`` run has already spent its SARIMAX fits.
    """
    config_path = config_path.resolve()
    if not config_path.is_file():
        message = f"Unable to extract outlier frames. The config path '{config_path}' does not point to a file."
        raise FileNotFoundError(message)
    if overwrite and reset:
        message = "Unable to extract outlier frames. The overwrite and reset options are mutually exclusive."
        raise ValueError(message)
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
        config_path=config_path, frames_per_video=frames_per_video, error_context="Unable to extract outlier frames."
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
        cfg=configuration,
        shuffle=shuffle_index,
        trainFraction=configuration["TrainingFraction"][training_set_index],
        modelprefix="",
        snapshot_index=pose_snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
    )

    # Outlier extraction refines the project's own registered videos, matched by resolved path exactly like the k-means
    # extractor selects them, rather than adding arbitrary files to the project. This keeps the refinement loop closed
    # over config.yaml's video_sets and leaves the configuration file read-only during the run.
    registered_videos = list(configuration.get("video_sets") or {})
    if not registered_videos:
        message = "Unable to extract outlier frames. The project's config.yaml does not list any videos in video_sets."
        raise ValueError(message)
    if videos:
        matched_videos, unmatched_videos = select_registered_videos(
            registered_videos=registered_videos, requested_videos=tuple(videos)
        )
        for video in unmatched_videos:
            sys.stderr.write(f"WARNING: {video} is not registered in the project's config.yaml and was skipped.\n")
        sys.stderr.flush()
        if not matched_videos:
            message = (
                "Unable to extract outlier frames. None of the requested videos matched a registered project video. "
                "Outlier extraction only refines videos already registered in the project's config.yaml."
            )
            raise ValueError(message)
        video_paths = list(matched_videos)
    else:
        # With no videos named, refines every registered video the current model has already analyzed, closing the
        # default refinement pass over exactly this model's own outputs. A registered video with no predictions is not
        # part of the model's processed set, so it is left out rather than reported as an unanalyzed failure.
        video_paths = _discover_analyzed_videos(
            registered_videos=registered_videos, scorer=scorer, tracking_method=resolved_tracking_method
        )
        if not video_paths:
            message = (
                "Unable to extract outlier frames. No registered project video has predictions from the current model "
                f"(scorer '{scorer}'). Analyze videos with this model first, or name specific videos to refine."
            )
            raise ValueError(message)
    # Two videos that share a stem would write into one labeled-data directory, racing in the parallel extraction pool.
    ensure_unique_video_stems(videos=video_paths, error_context="Unable to extract outlier frames.")

    # Clearing runs single-threaded here, before detection, so the concurrent extraction workers never race on reading
    # the machine-label bookkeeping and so DeepLabCut's frame writer, which skips indices whose image already exists,
    # actually re-writes the frames it re-flags this pass.
    if overwrite or reset:
        _clear_iteration_outliers(
            config_path=config_path, configuration=configuration, selected_videos=video_paths, reset=reset
        )

    candidates, unanalyzed_videos, errors = _detect_all_videos(
        video_paths=video_paths,
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
        # Every extraction video is already registered in config.yaml (they were matched against video_sets above), so
        # the workers only read the configuration file. The per-worker DeepLabCut video-add is neutralized inside
        # _extract_one_video to keep it that way.
        summary = _extract_all_videos(
            config_path=config_path,
            videos=extraction_videos,
            candidates=candidates,
            scorer=scorer,
            tracking_method=resolved_tracking_method,
            outlier_algorithm=outlier_algorithm,
            extraction_algorithm=extraction_algorithm,
            clustering_resize_width=clustering_resize_width,
            cluster_in_color=cluster_in_color,
            save_labeled_frames=save_labeled_frames,
            worker_count=worker_count,
            cores_per_worker=cores_per_worker,
            reserved_core_count=reserved_core_count,
            display_progress=display_progress,
            total_video_count=len(video_paths),
            unanalyzed_videos=tuple(unanalyzed_videos),
            detection_errors=errors,
        )

    prune_empty_labeled_data_directories(project_directory=config_path.parent, display_progress=display_progress)
    return summary


def _discover_analyzed_videos(*, registered_videos: list[str], scorer: str, tracking_method: str) -> list[str]:
    """Returns the registered videos the current model has already analyzed, in configuration order.

    A video counts as analyzed when DeepLabCut can locate a prediction file named by ``scorer`` in the video's own
    directory. This is a filename probe rather than a full prediction-table load, so resolving the default refinement
    set, every video the current model processed, stays cheap even for a project with many large videos.

    Args:
        registered_videos: The project's registered video paths, in configuration order.
        scorer: The DeepLabCut scorer string naming the current model's prediction files.
        tracking_method: The resolved multi-animal tracker method, matched against the prediction file suffix.

    Returns:
        The registered videos that have a matching prediction file, in configuration order.
    """
    analyzed_videos: list[str] = []
    for video in registered_videos:
        try:
            auxiliaryfunctions.find_analyzed_data(
                folder=str(Path(video).parent),
                videoname=Path(video).stem,
                scorer=scorer,
                track_method=tracking_method,
            )
        except FileNotFoundError:
            continue
        analyzed_videos.append(video)
    return analyzed_videos


def _detect_all_videos(
    *,
    video_paths: list[str],
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
    fitting_worker_count: int,
    reserved_core_count: int,
    display_progress: bool,
) -> tuple[dict[str, list[int]], list[str], list[tuple[str, str]]]:
    """Computes the outlier candidate frames for every video, parallelizing the SARIMAX fits when fitting.

    Args:
        video_paths: The resolved video paths to detect outliers in.
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
        video_predictions_directory = Path(video).parent
        # Detection compute is inside the try alongside the load. A malformed prediction table (empty, missing a
        # likelihood level, or a column count the fitting reshape rejects) is therefore recorded per-video rather than
        # aborting the whole run, upholding the summary's per-video error contract.
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
                candidates[video] = uncertain_outlier_indices(
                    predictions=comparison_predictions, minimum_confidence=minimum_confidence
                )
            elif outlier_algorithm == "jump":
                candidates[video] = jump_outlier_indices(
                    predictions=comparison_predictions, pixel_distance_threshold=pixel_distance_threshold
                )
            else:
                keypoint_series_by_video[video] = fitting_keypoint_series(comparison_predictions)
        except FileNotFoundError:
            unanalyzed_videos.append(video)
            continue
        except Exception:
            errors.append((video, "detection error:\n" + traceback.format_exc()))
            continue

    if keypoint_series_by_video:
        # The fitting reduction needs a valid numframes2pick to size its fallback selection. Guards it here (the k-means
        # budgeted path guards the identical value) so a config that is missing or nulls the key fails with a clean
        # ValueError the CLI reports, rather than an uncaught KeyError/TypeError escaping as a raw traceback.
        frames_per_video_count = configuration.get("numframes2pick")
        if not isinstance(frames_per_video_count, int) or frames_per_video_count < 1:
            message = (
                "Unable to extract outlier frames. The project's numframes2pick must be a positive integer, but got "
                f"{frames_per_video_count!r}. Pass frames_per_video to set it."
            )
            raise ValueError(message)
        candidates.update(
            _detect_fitting_outliers(
                keypoint_series_by_video=keypoint_series_by_video,
                frames_per_video_count=frames_per_video_count,
                pixel_distance_threshold=pixel_distance_threshold,
                minimum_confidence=minimum_confidence,
                autoregressive_degree=autoregressive_degree,
                moving_average_degree=moving_average_degree,
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
        fitting_worker_count: The number of fit processes, or -1 to use every usable core.
        reserved_core_count: The number of cores to leave free when sizing the pool.
        display_progress: Determines whether to report the number of fits being run.

    Returns:
        The outlier candidate frames keyed by video.
    """
    tasks: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, int, int]] = []
    owner_videos: list[str] = []
    for video, keypoint_series in keypoint_series_by_video.items():
        for horizontal_positions, vertical_positions, confidences in keypoint_series:
            tasks.append(
                (
                    horizontal_positions,
                    vertical_positions,
                    confidences,
                    minimum_confidence,
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
        keypoint_deviations = pool.starmap(func=fit_keypoint_distance, iterable=tasks)

    deviations_by_video: dict[str, list[NDArray[np.float64]]] = {video: [] for video in keypoint_series_by_video}
    for video, deviation in zip(owner_videos, keypoint_deviations, strict=True):
        deviations_by_video[video].append(deviation)

    return {
        video: fitting_outlier_indices(
            keypoint_deviations=video_deviations,
            frames_per_video_count=frames_per_video_count,
            pixel_distance_threshold=pixel_distance_threshold,
        )
        for video, video_deviations in deviations_by_video.items()
    }


def _extract_all_videos(
    *,
    config_path: Path,
    videos: list[str],
    candidates: dict[str, list[int]],
    scorer: str,
    tracking_method: str,
    outlier_algorithm: OutlierAlgorithm,
    extraction_algorithm: ExtractionAlgorithm,
    clustering_resize_width: int,
    cluster_in_color: bool,
    save_labeled_frames: bool,
    worker_count: int,
    cores_per_worker: int,
    reserved_core_count: int,
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
        scorer: The DeepLabCut scorer string naming each video's prediction files.
        tracking_method: The resolved multi-animal tracker method.
        outlier_algorithm: The detection algorithm that produced the candidates.
        extraction_algorithm: The frame-selection algorithm applied to the candidates.
        clustering_resize_width: The downsample width for k-means selection.
        cluster_in_color: Determines whether k-means selection clusters on color channels.
        save_labeled_frames: Determines whether to also save each frame with the model's predictions drawn on it.
        worker_count: The requested extraction worker count, or -1 to resolve automatically.
        cores_per_worker: The requested cores per worker, or -1 for a saturating block per worker when the worker count
            is automatic and an even split of the usable cores across an explicit one.
        reserved_core_count: The number of cores to leave free.
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
                scorer,
                tracking_method,
                extraction_algorithm,
                clustering_resize_width,
                cluster_in_color,
                save_labeled_frames,
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

    output_crop_x, output_crop_y = _video_cropping_offset(configuration=configuration, video=video)
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


def _clear_iteration_outliers(
    *, config_path: Path, configuration: dict[str, Any], selected_videos: list[str], reset: bool
) -> None:
    """Discards this refinement iteration's extracted outlier frames before re-extraction, preserving labeled frames.

    Outlier frames are written into the same ``labeled-data/<stem>`` directories as the human-labeled training frames,
    so clearing cannot simply delete the images. Only the frames this iteration extracted as outliers, recorded in the
    iteration's ``machinelabels-iter<N>`` bookkeeping, are removed, and any of those that already appear in the human
    ``CollectedData`` labels are kept. ``reset`` wipes every project video's outlier set for the iteration, whereas
    ``overwrite`` (``reset`` False) wipes only the videos this run re-extracts, leaving the rest intact.

    Args:
        config_path: The resolved project config.yaml path, whose parent holds the ``labeled-data`` tree.
        configuration: The loaded project configuration, read for the current ``iteration`` and human ``scorer``.
        selected_videos: The registered project videos this run re-extracts, cleared under ``overwrite``.
        reset: Determines whether to clear every project video's outlier set for the iteration rather than only the
            selected videos.
    """
    iteration = int(configuration.get("iteration", 0))
    scorer = str(configuration.get("scorer", ""))
    labeled_data_directory = config_path.parent / "labeled-data"
    if reset:
        # Reset clears every directory holding an outlier set for this iteration, not just the videos being
        # re-extracted, so the whole refinement iteration starts from a clean slate.
        machine_labels_name = f"machinelabels-iter{iteration}.h5"
        target_directories = (
            sorted(path.parent for path in labeled_data_directory.glob(f"*/{machine_labels_name}"))
            if labeled_data_directory.exists()
            else []
        )
        scope_label = "--reset"
    else:
        target_directories = [labeled_data_directory / Path(video).stem for video in selected_videos]
        scope_label = "--overwrite"

    removed_frame_count = 0
    cleared_directory_count = 0
    refined_directory_count = 0
    for directory in target_directories:
        try:
            removed_count, had_refined_labels = _clear_video_iteration_outliers(
                directory=directory, iteration=iteration, scorer=scorer
            )
        except Exception:
            sys.stderr.write(f"WARNING: {scope_label} could not clear the outlier frames in '{directory}'.\n")
            continue
        if removed_count or had_refined_labels:
            cleared_directory_count += 1
        removed_frame_count += removed_count
        refined_directory_count += 1 if had_refined_labels else 0

    sys.stderr.write(
        f"{scope_label} removed {removed_frame_count} outlier frame(s) from {cleared_directory_count} video "
        f"directory(ies) for iteration {iteration} before re-extraction.\n"
    )
    if refined_directory_count:
        sys.stderr.write(
            f"WARNING: {scope_label} discarded manual outlier refinements in {refined_directory_count} directory(ies); "
            f"those MachineLabelsRefine corrections must be redone.\n"
        )
    sys.stderr.flush()


def _clear_video_iteration_outliers(*, directory: Path, iteration: int, scorer: str) -> tuple[int, bool]:
    """Removes one video directory's outlier frames for a refinement iteration, keeping any already-labeled frames.

    The iteration's ``machinelabels-iter<N>.h5`` records exactly the frames extracted as outliers this iteration, keyed
    by their ``imgNNNN.png`` names. Those frames are deleted, except any that also appear in the human
    ``CollectedData`` labels, along with the iteration's machine-label bookkeeping and any manual refinement of it, so
    re-extraction rebuilds the set from scratch without orphaning either images or labels.

    Args:
        directory: The ``labeled-data/<stem>`` directory whose iteration outlier frames are cleared.
        iteration: The refinement iteration whose machine-label record names the frames to remove.
        scorer: The human scorer naming the ``CollectedData`` labels whose frames are preserved.

    Returns:
        A tuple of the number of frames removed and whether a manual ``MachineLabelsRefine`` file was discarded.
    """
    machine_labels_path = directory / f"machinelabels-iter{iteration}.h5"
    if not machine_labels_path.is_file():
        return 0, False

    outlier_frame_names = frame_names_from_index(pd.read_hdf(machine_labels_path, key="df_with_missing").index)
    labeled_frame_names: set[str] = set()
    collected_data_path = directory / f"CollectedData_{scorer}.h5"
    if scorer and collected_data_path.is_file():
        labeled_frame_names = frame_names_from_index(pd.read_hdf(collected_data_path, key="df_with_missing").index)

    removed_count = 0
    for frame_name in outlier_frame_names - labeled_frame_names:
        frame_path = directory / frame_name
        if frame_path.is_file():
            frame_path.unlink()
            removed_count += 1
        # Drops the prediction overlay saved beside the frame when --save-labeled was used, so it never orphans.
        (directory / f"{Path(frame_name).stem}labeled.png").unlink(missing_ok=True)

    machine_labels_path.unlink(missing_ok=True)
    (directory / "machinelabels.csv").unlink(missing_ok=True)
    # Any manual refinement of this iteration's outliers references the frames just cleared, so it is discarded too.
    refinement_files = sorted(directory.glob("MachineLabelsRefine.*"))
    for refinement_file in refinement_files:
        refinement_file.unlink()
    return removed_count, bool(refinement_files)


def _extract_one_video(task: tuple[Any, ...]) -> tuple[str, int, str]:
    """Selects and writes the flagged frames for a single video, reusing DeepLabCut's own frame writer.

    DeepLabCut's console output is silenced and exceptions are captured, so one bad video cannot kill the worker pool.
    DeepLabCut's frame writer re-registers the video in config.yaml. That add is neutralized here because every refined
    video is already registered in the project, so the concurrent workers never write the configuration file.

    Args:
        task: The packed work item carrying the video path, the video index, the candidate frames, the config path,
            the scorer, the tracker method, the selection settings, and the progress queue.

    Returns:
        A tuple of the video path, the number of frames freshly written, and a status string (``"ok"``,
        ``"not_analyzed"``, or an ``"error:"`` traceback).
    """
    (
        video,
        video_index,
        indices,
        config_path,
        scorer,
        tracking_method,
        extraction_algorithm,
        clustering_resize_width,
        cluster_in_color,
        save_labeled_frames,
        progress_queue,
    ) = task
    try:
        cv2.setNumThreads(1)
        configuration = auxiliaryfunctions.read_config(str(config_path))
        video_predictions_directory = Path(video).parent
        predictions = _load_sliced_predictions(
            video=video,
            video_predictions_directory=video_predictions_directory,
            scorer=scorer,
            configuration=configuration,
            tracking_method=tracking_method,
        )

        output_directory = Path(configuration["project_path"]) / "labeled-data" / Path(video).stem
        frame_count_before = _count_directory_frames(output_directory)

        # Swaps DeepLabCut's random-seek k-means reader for the streaming one, routing its per-candidate progress to the
        # parent's aggregate bar. The queue is None when progress is disabled, leaving a plain (stream-suppressed) bar.
        # DeepLabCut's per-worker config write is neutralized because every refined video is already registered.
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
            )
    except FileNotFoundError:
        return video, 0, "not_analyzed"
    except Exception:
        return video, 0, "error:\n" + traceback.format_exc()
    else:
        frames_written = _count_directory_frames(output_directory) - frame_count_before
        return video, max(0, frames_written), "ok"


def _count_directory_frames(output_directory: Path) -> int:
    """Counts the extracted image frames in a labeled-data directory, ignoring the predicted-label overlays.

    Args:
        output_directory: The ``labeled-data/<video>`` directory whose extracted frames are counted.

    Returns:
        The number of ``img*.png`` frames that are not ``*labeled.png`` prediction overlays.
    """
    return len(extracted_frame_paths(output_directory))


def _skip_video_registration(**_kwargs: Any) -> bool:
    """Neutralizes DeepLabCut's per-video config.yaml registration inside a worker. The videos are pre-registered.

    Returns:
        True, reporting a successful registration so DeepLabCut's frame writer proceeds unchanged.
    """
    return True
