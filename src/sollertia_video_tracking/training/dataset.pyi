from enum import StrEnum
from typing import Any, TextIO
from pathlib import Path
import contextlib
from dataclasses import dataclass
from collections.abc import Iterator

from deeplabcut.core.weight_init import WeightInitialization as WeightInitialization

_TOP_DOWN_PREFIX: str
_SUPER_ANIMAL_DATASETS: tuple[str, ...]
_CONDITION_PREDICTION_SUFFIXES: tuple[str, ...]
_CONDITION_SNAPSHOT_SUFFIX: str
_UNANNOTATED_VIDEO_NOTICE: str

class WeightInitializationMethod(StrEnum):
    IMAGENET = "imagenet"
    TRANSFER = "transfer"
    FINE_TUNE = "fine-tune"

@dataclass(frozen=True, slots=True)
class TrainingDatasetSummary:
    config: Path
    shuffle: int
    net_type: str | None
    detector_type: str | None
    weight_init: str
    from_shuffle: int | None
    def describe(self) -> str: ...

def get_available_pose_models() -> tuple[str, ...]: ...
def get_available_object_detectors() -> tuple[str, ...]: ...
def get_available_super_animals() -> tuple[str, ...]: ...
def build_superanimal_weight_init(
    config: str | Path,
    super_animal: str,
    network_type: str,
    detector_type: str | None,
    *,
    fine_tune: bool = False,
    memory_replay: bool = False,
    customized_pose_checkpoint: str | Path | None = None,
    customized_detector_checkpoint: str | Path | None = None,
) -> WeightInitialization: ...
def build_conditional_top_down_conditions(conditions_path: str | Path) -> Path | tuple[int, str]: ...
def create_training_dataset(
    config: str | Path,
    *,
    shuffle: int = 1,
    network_type: str | None = None,
    detector_type: str | None = None,
    weight_initialization: WeightInitialization | None = None,
    conditional_top_down_conditions: Path | tuple[int, str] | None = None,
    from_shuffle: int | None = None,
    from_training_set_index: int = 0,
    overwrite: bool = False,
) -> TrainingDatasetSummary: ...

class _UnannotatedNoticeFilter:
    _target: TextIO
    _marker: str
    _pending: str
    def __init__(self, target: TextIO, marker: str) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def write(self, text: str) -> int: ...
    def flush(self) -> None: ...
    def drain(self) -> None: ...

@contextlib.contextmanager
def _suppress_unannotated_video_notices() -> Iterator[None]: ...
