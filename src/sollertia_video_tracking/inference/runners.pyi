from typing import Any
from contextlib import AbstractContextManager, contextmanager
from collections.abc import (
    Callable as Callable,
    Iterator,
)

import torch
from deeplabcut.pose_estimation_pytorch.runners.inference import (
    InferenceRunner as InferenceRunner,
    PoseInferenceRunner as PoseInferenceRunner,
    DetectorInferenceRunner,
)

from ..hardware import warn as warn
from .optimization import InferenceProfile as InferenceProfile

@contextmanager
def patch_dlc_runner_builders(profile: InferenceProfile) -> Iterator[None]: ...
def _optimize_inference_runner(runner: InferenceRunner, profile: InferenceProfile) -> InferenceRunner: ...
def _build_pose_predict(
    runner: PoseInferenceRunner,
    autocast_context: Callable[[], AbstractContextManager[None]],
    move_inputs: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[..., list[dict[str, dict[str, Any]]]]: ...
def _build_detector_predict(
    runner: DetectorInferenceRunner,
    autocast_context: Callable[[], AbstractContextManager[None]],
    move_inputs: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[..., list[dict[str, dict[str, Any]]]]: ...
