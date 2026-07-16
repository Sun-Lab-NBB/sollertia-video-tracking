from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import (
    Callable as Callable,
    Iterator,
)

from .progress import AggregateBar as AggregateBar
from .cpu_allocation import pin_worker_to_cores as pin_worker_to_cores

@dataclass(frozen=True, slots=True)
class PurgeSummary:
    config_path: Path
    executed: bool
    removed_directories: tuple[Path, ...]
    labeled_directories: tuple[Path, ...]
    frame_count: int
    unmatched_videos: tuple[str, ...] = ...
    @property
    def removed_directory_count(self) -> int: ...
    @property
    def labeled_directory_count(self) -> int: ...

@dataclass(frozen=True, slots=True)
class RefinementDirectoryStatus:
    directory: Path
    unrefined_frame_count: int

@dataclass(frozen=True, slots=True)
class RefinementStatusSummary:
    config_path: Path
    iteration: int
    pending_directories: tuple[RefinementDirectoryStatus, ...]
    unmatched_videos: tuple[str, ...] = ...
    unreadable: tuple[tuple[Path, str], ...] = ...
    @property
    def pending_directory_count(self) -> int: ...
    @property
    def pending_frame_count(self) -> int: ...
    @property
    def successful(self) -> bool: ...
    def describe(self) -> str: ...

def normalize_project_config(config_path: Path, *, frames_per_video: int, error_context: str) -> Any: ...
def iter_pinned_extraction(
    *,
    videos: list[str],
    make_tasks: Callable[[Any | None], list[tuple[Any, ...]]],
    worker: Callable[[tuple[Any, ...]], tuple[str, int, str]],
    worker_count: int,
    core_sets: list[set[int]],
    frame_totals: dict[int, int],
    display_progress: bool,
) -> Iterator[tuple[str, int, str]]: ...
def prune_empty_labeled_data_directories(project_directory: Path, *, display_progress: bool = False) -> int: ...
def extracted_frame_paths(directory: Path) -> list[Path]: ...
def frame_names_from_index(frame_index: Any) -> set[str]: ...
def finite_labeled_frame_names(collected_data_path: Path) -> set[str]: ...
def machine_label_frame_names(directory: Path) -> set[str]: ...
def has_outlier_refinement_data(directory: Path) -> bool: ...
def select_registered_videos(
    registered_videos: list[str], requested_videos: tuple[str | Path, ...]
) -> tuple[list[str], list[str]]: ...
def ensure_unique_video_stems(videos: list[str], *, error_context: str) -> None: ...
def purge_labeled_data(
    config_path: Path, *, videos: tuple[str | Path, ...] = (), execute: bool = False
) -> PurgeSummary: ...
def summarize_refinement_status(
    config_path: Path, *, videos: tuple[str | Path, ...] = ()
) -> RefinementStatusSummary: ...
