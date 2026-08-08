from typing import Any
from pathlib import Path
from contextlib import AbstractContextManager, contextmanager
from collections.abc import Iterator

import torch
from torch import nn as nn
from _typeshed import Incomplete
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.runners.train import TrainingRunner, PoseTrainingRunner, DetectorTrainingRunner
from deeplabcut.pose_estimation_pytorch.runners.logger import BaseLogger as BaseLogger

_logger: Incomplete

def build_optimized_training_runner(
    runner_config: dict,
    model_folder: Path,
    task: Task,
    model: nn.Module,
    device: str,
    gpus: list[int] | None = None,
    snapshot_path: str | Path | None = None,
    logger: BaseLogger | None = None,
    *,
    load_head_weights: bool = True,
    amp_dtype: torch.dtype | None = None,
    use_gradient_scaler: bool = False,
    torch_compile: bool = False,
    ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
) -> TrainingRunner: ...

class _OptimizedTrainingRunnerMixin(TrainingRunner):
    model: Any
    _epoch_predictions: Any
    _epoch_ground_truth: Any
    _ddp_static_graph: bool
    _amp_dtype: Incomplete
    _gradient_scaler: Incomplete
    _torch_compile: Incomplete
    _ddp: Incomplete
    _rank: Incomplete
    _world_size: Incomplete
    _local_rank: Incomplete
    def __init__(
        self,
        *args: Any,
        amp_dtype: torch.dtype | None = None,
        use_gradient_scaler: bool = False,
        torch_compile: bool = False,
        ddp: bool = False,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs: Any,
    ) -> None: ...
    @property
    def _is_main(self) -> bool: ...
    def _unwrap(self) -> Any: ...
    def _build_autocast_context(self, *, enabled: bool = True) -> AbstractContextManager[None]: ...
    def _backward_and_step(self, loss: torch.Tensor) -> None: ...
    def _prepare_model_for_training(self) -> None: ...
    def state_dict(self) -> dict: ...
    current_epoch: Incomplete
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        epochs: int,
        display_iters: int,
    ) -> None: ...
    def _epoch(self, loader: torch.utils.data.DataLoader, mode: str = "train", display_iters: int = 500) -> float: ...

class _OptimizedPoseTrainingRunner(_OptimizedTrainingRunnerMixin, PoseTrainingRunner):
    def step(self, batch: dict[str, Any], mode: str = "train") -> dict[str, Any]: ...

class _OptimizedDetectorTrainingRunner(_OptimizedTrainingRunnerMixin, DetectorTrainingRunner):
    _ddp_static_graph: bool
    def step(self, batch: dict[str, Any], mode: str = "train") -> dict[str, Any]: ...

@contextmanager
def _overwriting_path_rename() -> Iterator[None]: ...
