from typing import Any
from pathlib import Path
from dataclasses import dataclass

from .progress import make_progress_reporter as make_progress_reporter
from .utilities import (
    extracted_frame_paths as extracted_frame_paths,
    iter_pinned_extraction as iter_pinned_extraction,
    normalize_project_config as normalize_project_config,
    select_registered_videos as select_registered_videos,
    ensure_unique_video_stems as ensure_unique_video_stems,
    machine_label_frame_names as machine_label_frame_names,
    finite_labeled_frame_names as finite_labeled_frame_names,
    has_outlier_refinement_data as has_outlier_refinement_data,
    prune_empty_labeled_data_directories as prune_empty_labeled_data_directories,
)
from .frame_reading import make_fast_kmeans_selector as make_fast_kmeans_selector
from .cpu_allocation import (
    DEFAULT_RESERVED_CORE_COUNT as DEFAULT_RESERVED_CORE_COUNT,
    plan_core_allocation as plan_core_allocation,
)
from .video_grouping import group_videos as group_videos
from .video_sampling import (
    VideoSamplingPlan as VideoSamplingPlan,
    plan_video_sampling as plan_video_sampling,
)

_MAXIMUM_HEADER_READ_THREADS: int

@dataclass(frozen=True, slots=True)
class FrameExtractionSummary:
    extracted_video_count: int
    cleared_frame_count: int
    total_video_count: int
    worker_count: int
    used_core_count: int
    total_core_count: int
    clustering_frame_count: int
    existing_frame_count: int = ...
    target_frame_count: int = ...
    errors: tuple[tuple[str, str], ...] = ...
    @property
    def failed_video_count(self) -> int: ...
    @property
    def successful(self) -> bool: ...

def extract_frames_kmeans(
    config_path: Path,
    *,
    clustering_stride: int = 1,
    worker_count: int = -1,
    cores_per_worker: int = -1,
    reserved_core_count: int = ...,
    frames_per_video: int = -1,
    total_frame_budget: int = -1,
    balance_groups: bool = False,
    group_by_pattern: str | None = None,
    requested_videos: tuple[str | Path, ...] = (),
    exclusive: bool = False,
    clustering_resize_width: int = 30,
    cluster_in_color: bool = False,
    overwrite: bool = False,
    reset: bool = False,
    display_progress: bool = True,
) -> FrameExtractionSummary: ...
def _report_plan(
    video_count: int,
    configured_frames_per_video: int | str,
    clustering_stride: int,
    clustering_resize_width: int,
    *,
    cluster_in_color: bool,
    worker_count: int,
    used_core_count: int,
    total_core_count: int,
    clustering_frame_count: int,
    config_path: Path,
) -> None: ...
def _report_sampling_plan(plan: VideoSamplingPlan) -> None: ...
def _count_extracted_frames(videos: list[str], project_directory: Path) -> dict[str, int]: ...
def _clear_bare_frames(
    *, project_directory: Path, video_stems: list[str], scorer: str, scope_label: str
) -> tuple[int, set[str]]: ...
def _clear_bare_frames_in_directory(*, directory: Path, scorer: str) -> int: ...
def _drop_collected_data_rows(*, collected_data_path: Path, removed_frame_names: set[str]) -> None: ...
def _count_clustering_frames(
    videos: list[str], start_fraction: float, stop_fraction: float, clustering_stride: int
) -> dict[int, int]: ...
def _extract_one_video(task: tuple[Any, ...]) -> tuple[str, int, str]: ...
