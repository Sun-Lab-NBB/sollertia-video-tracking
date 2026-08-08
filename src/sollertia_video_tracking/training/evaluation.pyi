from typing import Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from deeplabcut.pose_estimation_pytorch.data import DLCLoader, PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.task import Task

_logger: Incomplete
_SPLITS: tuple[str, ...]
_LABELED_DATA_DIR: str
_WORST_KEYPOINT_COUNT: int
_FEATHER_SCHEMA: dict[str, Any]

@dataclass(frozen=True, slots=True)
class _SplitMetrics:
    images: int
    rmse_px: float
    rmse_pcutoff_px: float
    map: float
    mar: float
    unmatched_images: int

@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    config: Path
    shuffle: int
    snapshot: str
    feather_path: Path
    provenance_path: Path
    pcutoff: float
    train: _SplitMetrics
    test: _SplitMetrics
    @property
    def generalization_gap_px(self) -> float: ...
    def describe(self) -> str: ...

def evaluate_trained_model(
    config: str | Path,
    *,
    shuffle: int = 1,
    training_set_index: int = 0,
    snapshot_index: int | str = "best",
    detector_snapshot_index: int = -1,
    confidence_cutoff: float | None = None,
    batch_size: int = 1,
    device: str | None = None,
    write_provenance: bool = True,
) -> EvaluationSummary: ...
def resolve_evaluation_batch_size(loader: DLCLoader, requested: int) -> int: ...
def _resolve_snapshot(loader: DLCLoader, index: int | str, task: Task, *, required: bool = True) -> Any: ...
def _realign_memory_replay_parameters(
    loader: DLCLoader, parameters: PoseDatasetParameters
) -> PoseDatasetParameters: ...
def _accumulate_split_rows(
    columns: dict[str, list[Any]],
    *,
    snapshot_name: str,
    split: str,
    image_paths: list[str],
    predictions: dict[str, dict[str, NDArray[np.float32]]],
    ground_truth: dict[str, NDArray[np.float32]],
    bodyparts: list[str],
    individuals: list[str],
    single_animal: bool,
    confidence_cutoff: float,
    prediction_key: str = "bodyparts",
) -> int: ...
def _surviving_individual_indices(ground_truth: NDArray[np.float32]) -> NDArray[np.intp]: ...
def _matched_individual(
    match: Any,
    prepared_ground_truth: NDArray[np.float32],
    surviving: NDArray[np.intp] | None,
    individuals: list[str],
    instance: int,
) -> str: ...
def _derive_relative_image_path(image: str) -> tuple[str, str]: ...
def _rank_worst_keypoints(metrics: dict[str, Any], bodyparts: list[str]) -> list[dict[str, Any]]: ...
def _write_provenance(
    provenance_path: Path,
    *,
    config: Path,
    loader: DLCLoader,
    parameters: PoseDatasetParameters,
    snapshot: Any,
    detector_snapshot: Any,
    device: str | None,
    batch_size: int,
    confidence_cutoff: float,
    single_animal: bool,
    split_metrics: dict[str, _SplitMetrics],
    feather_path: Path,
    worst_keypoints: list[dict[str, Any]],
) -> None: ...
