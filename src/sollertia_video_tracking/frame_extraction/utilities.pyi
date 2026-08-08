from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import (
    Callable as Callable,
    Iterator,
)

from .progress import AggregateBar as AggregateBar
from ..hardware import warn as warn
from ..reporting import (
    WorkerExit as WorkerExit,
    ProcessSupervisor as ProcessSupervisor,
    PipelineFailedError as PipelineFailedError,
    PipelineInterruptedError as PipelineInterruptedError,
    describe_process_exit as describe_process_exit,
    enable_native_crash_dumps as enable_native_crash_dumps,
)
from .cpu_allocation import pin_process_to_cores as pin_process_to_cores

_RESULT_POLL_SECONDS: float
_EXTRACTION_MEMORY_REMEDY: str

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
def run_supervised_tasks(
    *,
    tasks: list[tuple[Any, ...]],
    worker: Callable[[tuple[Any, ...]], Any],
    worker_count: int,
    role: str,
    memory_remedy: str,
) -> list[Any]: ...
def prune_empty_labeled_data_directories(project_directory: Path, *, display_progress: bool = False) -> int: ...
def extracted_frame_paths(directory: Path) -> list[Path]: ...
def frame_names_from_index(frame_index: Any) -> set[str]: ...
def finite_labeled_frame_names(collected_data_path: Path) -> set[str]: ...
def machine_label_frame_names(directory: Path) -> set[str]: ...
def drop_collected_data_rows(*, collected_data_path: Path, removed_frame_names: set[str]) -> None: ...
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
def _drain_worker_tasks(
    core_set: set[int], task_queue: Any, results_queue: Any, worker: Callable[[tuple[Any, ...]], tuple[str, int, str]]
) -> None: ...
def _describe_lost_supervised_task(
    task: tuple[Any, ...], failure: str | None, exit_record: WorkerExit | None, role: str, memory_remedy: str
) -> str: ...
def _start_extraction_manager(context: Any) -> Any: ...
def _next_extraction_message(results_queue: Any) -> tuple[str, int, Any] | None: ...
def _exit_for_claim(exits: tuple[WorkerExit, ...], pid: int | None) -> WorkerExit | None: ...
def _lost_task_result(task: tuple[Any, ...], exit_record: WorkerExit | None) -> tuple[str, int, str]: ...
