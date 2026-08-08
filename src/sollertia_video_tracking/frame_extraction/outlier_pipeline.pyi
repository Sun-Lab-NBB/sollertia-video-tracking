from enum import StrEnum
from typing import Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray as NDArray

from .progress import make_progress_reporter as make_progress_reporter
from .utilities import (
    run_supervised_tasks as run_supervised_tasks,
    extracted_frame_paths as extracted_frame_paths,
    frame_names_from_index as frame_names_from_index,
    iter_pinned_extraction as iter_pinned_extraction,
    drop_collected_data_rows as drop_collected_data_rows,
    normalize_project_config as normalize_project_config,
    select_registered_videos as select_registered_videos,
    ensure_unique_video_stems as ensure_unique_video_stems,
    finite_labeled_frame_names as finite_labeled_frame_names,
    prune_empty_labeled_data_directories as prune_empty_labeled_data_directories,
)
from ..reporting import enable_native_crash_dumps as enable_native_crash_dumps
from .frame_reading import make_fast_kmeans_selector as make_fast_kmeans_selector
from .cpu_allocation import (
    DEFAULT_RESERVED_CORE_COUNT as DEFAULT_RESERVED_CORE_COUNT,
    plan_core_allocation as plan_core_allocation,
)
from .outlier_detection import (
    OutlierAlgorithm as OutlierAlgorithm,
    jump_outlier_indices as jump_outlier_indices,
    fit_keypoint_distance as fit_keypoint_distance,
    fitting_keypoint_count as fitting_keypoint_count,
    fitting_keypoint_series as fitting_keypoint_series,
    fitting_outlier_indices as fitting_outlier_indices,
    uncertain_outlier_indices as uncertain_outlier_indices,
)

_CROP_FIELD_COUNT: int
_FITTING_MEMORY_REMEDY: str

class ExtractionAlgorithm(StrEnum):
    KMEANS = "kmeans"
    UNIFORM = "uniform"

class TrackingMethod(StrEnum):
    BOX = "box"
    SKELETON = "skeleton"
    ELLIPSE = "ellipse"

@dataclass(frozen=True, slots=True)
class OutlierExtractionSummary:
    config_path: Path
    outlier_algorithm: OutlierAlgorithm
    extraction_algorithm: ExtractionAlgorithm
    total_video_count: int
    extracted_video_count: int
    worker_count: int
    used_core_count: int
    total_core_count: int
    candidate_frame_count: int
    extracted_frame_count: int
    unanalyzed_videos: tuple[str, ...] = ...
    errors: tuple[tuple[str, str], ...] = ...
    @property
    def failed_video_count(self) -> int: ...
    @property
    def successful(self) -> bool: ...
    def describe(self) -> str: ...

def extract_outlier_frames_parallel(
    config_path: Path,
    videos: list[str | Path],
    *,
    shuffle_index: int = 1,
    training_set_index: int = 0,
    outlier_algorithm: OutlierAlgorithm = ...,
    explicit_frame_indices: tuple[int, ...] = (),
    comparison_bodyparts: tuple[str, ...] = (),
    pixel_distance_threshold: float = 20.0,
    minimum_confidence: float = 0.01,
    autoregressive_degree: int = 3,
    moving_average_degree: int = 1,
    extraction_algorithm: ExtractionAlgorithm = ...,
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
    reserved_core_count: int = ...,
    fitting_worker_count: int = -1,
    overwrite: bool = False,
    reset: bool = False,
    display_progress: bool = True,
) -> OutlierExtractionSummary: ...
def _discover_analyzed_videos(*, registered_videos: list[str], scorer: str, tracking_method: str) -> list[str]: ...
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
) -> tuple[dict[str, list[int]], list[str], list[tuple[str, str]]]: ...
def _detect_fitting_outliers(
    *,
    fitting_keypoint_counts: dict[str, int],
    scorer: str,
    configuration: dict[str, Any],
    tracking_method: str,
    resolved_comparison_bodyparts: list[str],
    frames_per_video_count: int,
    pixel_distance_threshold: float,
    minimum_confidence: float,
    autoregressive_degree: int,
    moving_average_degree: int,
    fitting_worker_count: int,
    reserved_core_count: int,
    display_progress: bool,
) -> dict[str, list[int]]: ...
def _fit_one_keypoint_task(task: tuple[Any, ...]) -> NDArray[np.float64]: ...
def _fit_video_keypoint(
    video: str,
    keypoint_index: int,
    scorer: str,
    configuration: dict[str, Any],
    tracking_method: str,
    resolved_comparison_bodyparts: list[str],
    minimum_confidence: float,
    autoregressive_degree: int,
    moving_average_degree: int,
) -> NDArray[np.float64]: ...
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
) -> OutlierExtractionSummary: ...
def _report_plan(
    video_count: int,
    outlier_algorithm: str,
    extraction_algorithm: str,
    candidate_frame_count: int,
    worker_count: int,
    used_core_count: int,
    total_core_count: int,
    config_path: Path,
) -> None: ...
def _load_sliced_predictions(
    video: str, video_predictions_directory: Path, scorer: str, configuration: dict[str, Any], tracking_method: str
) -> Any: ...
def _video_cropping_offset(configuration: dict[str, Any], video: str) -> tuple[int, int]: ...
def _clear_iteration_outliers(
    *, config_path: Path, configuration: dict[str, Any], selected_videos: list[str], reset: bool
) -> None: ...
def _clear_video_iteration_outliers(*, directory: Path, iteration: int, scorer: str) -> tuple[int, bool]: ...
def _extract_one_video(task: tuple[Any, ...]) -> tuple[str, int, str]: ...
def _count_directory_frames(output_directory: Path) -> int: ...
def _skip_video_registration(**_kwargs: Any) -> bool: ...
