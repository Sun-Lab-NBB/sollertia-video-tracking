from typing import Any
from pathlib import Path
import contextlib
from dataclasses import dataclass
from collections.abc import Iterator

import torch
from _typeshed import Incomplete
from torch.utils.data import DataLoader
from deeplabcut.pose_estimation_pytorch.data import DLCLoader
from deeplabcut.pose_estimation_pytorch.task import Task

from .monitor import (
    TrainingMonitor as TrainingMonitor,
    QueueTrainingLogger as QueueTrainingLogger,
)
from .runners import build_optimized_training_runner as build_optimized_training_runner
from ..hardware import warn as warn
from ..reporting import (
    PipelineFailedError as PipelineFailedError,
    PipelineInterruptedError as PipelineInterruptedError,
    read_file_tail as read_file_tail,
    is_interrupt_signal as is_interrupt_signal,
    describe_process_exit as describe_process_exit,
    enable_native_crash_dumps as enable_native_crash_dumps,
    write_optimization_report as write_optimization_report,
)
from .evaluation import (
    EvaluationSummary as EvaluationSummary,
    evaluate_trained_model as evaluate_trained_model,
    resolve_evaluation_batch_size as resolve_evaluation_batch_size,
)
from .optimization import (
    MultiGpuStrategy as MultiGpuStrategy,
    OptimizationProfile as OptimizationProfile,
    apply_runtime_optimizations as apply_runtime_optimizations,
)

_logger: Incomplete
_TRAINING_LOG_TAIL_LINES: int
_TRAINING_MEMORY_REMEDY: str

class TrainingFailedError(PipelineFailedError): ...
class TrainingInterruptedError(PipelineInterruptedError): ...

@dataclass(frozen=True, slots=True)
class TrainingSummary:
    config: Path
    shuffle: int
    model_folder: Path
    tasks_trained: tuple[str, ...]
    device: str
    strategy: str
    world_size: int
    precision: str
    epochs: int
    evaluation: EvaluationSummary | None = ...
    evaluation_error: str | None = ...
    def describe(self) -> str: ...

@dataclass(frozen=True, slots=True)
class _TrainingLaunch:
    config: Path
    shuffle: int
    training_set_index: int
    profile: OptimizationProfile
    snapshot_path: str | Path | None
    detector_path: str | Path | None
    load_head_weights: bool
    maximum_snapshots_to_keep: int | None
    progress_queue: Any
    port: int
    world_size: int

def train_model(
    config: str | Path,
    profile: OptimizationProfile,
    *,
    shuffle: int = 1,
    training_set_index: int = 0,
    epochs: int | None = None,
    batch_size: int | None = None,
    save_epochs: int | None = None,
    display_iterations: int | None = None,
    snapshot_path: str | Path | None = None,
    detector_path: str | Path | None = None,
    detector_batch_size: int | None = None,
    detector_epochs: int | None = None,
    detector_save_epochs: int | None = None,
    maximum_snapshots_to_keep: int | None = None,
    load_head_weights: bool = True,
    evaluate: bool = True,
    evaluation_batch_size: int = 1,
    evaluation_confidence_cutoff: float | None = None,
    display_progress: bool = True,
) -> TrainingSummary: ...
def detect_fixed_input_size(config: str | Path, *, shuffle: int = 1, training_set_index: int = 0) -> bool: ...
def _augmentation_is_fixed_size(train_augmentation: dict[str, Any]) -> bool: ...
def _has_fixed_dimensions(block: dict[str, Any]) -> bool: ...
def _is_positive_dimension(value: Any) -> bool: ...
def _evaluate_after_training(
    config: Path,
    profile: OptimizationProfile,
    *,
    shuffle: int,
    training_set_index: int,
    batch_size: int,
    confidence_cutoff: float | None,
) -> EvaluationSummary: ...
def _find_free_port() -> int: ...
def _route_logging_to_file(model_folder: Path, *, quiet_console: bool) -> None: ...
def _start_progress_monitor() -> tuple[Any, Any, TrainingMonitor]: ...
def _stop_progress_monitor(monitor: TrainingMonitor | None, manager: Any) -> None: ...
@contextlib.contextmanager
def _redirect_worker_console(log_path: Path, *, active: bool) -> Iterator[None]: ...
def _is_operator_interrupt(error: BaseException) -> bool: ...
def _format_training_failure(
    error: BaseException, *, config: Path, shuffle: int, model_folder: Path, training_log: Path
) -> str: ...
def _describe_worker_failure(error: BaseException) -> str: ...
def _append_training_log(training_log: Path, text: str) -> None: ...
def _plan_training_tasks(loader: DLCLoader) -> tuple[str, ...]: ...
def _resolve_process_placement(
    profile: OptimizationProfile, rank: int, task: Task | None = None
) -> tuple[str, list[int] | None, bool, int]: ...
def _build_pose_or_detector_model(
    run_config: dict, task: Task, snapshot_path: str | Path | None
) -> torch.nn.Module: ...
def _build_dataloaders(
    loader: DLCLoader, run_config: dict, task: Task, *, ddp: bool, rank: int, world_size: int
) -> tuple[DataLoader, DataLoader]: ...
def _train_single_model(
    loader: DLCLoader,
    run_config: dict,
    task: Task,
    profile: OptimizationProfile,
    *,
    rank: int,
    world_size: int,
    snapshot_path: str | Path | None,
    load_head_weights: bool,
    maximum_snapshots_to_keep: int | None,
    progress_queue: Any,
) -> None: ...
def _run_training_worker(rank: int, launch: _TrainingLaunch) -> None: ...
