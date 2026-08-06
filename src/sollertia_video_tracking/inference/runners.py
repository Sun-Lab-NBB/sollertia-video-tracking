"""Provides wrappers optimizing DeepLabCut inference runners with mixed precision, channels-last, and torch.compile."""

from typing import Any
import warnings
from contextlib import AbstractContextManager, nullcontext, contextmanager
from collections.abc import Callable, Iterator

import torch
import deeplabcut.pose_estimation_pytorch.apis.utils as dlc_apis_utils
from deeplabcut.pose_estimation_pytorch.runners.inference import (
    InferenceRunner,
    CTDInferenceRunner,
    PoseInferenceRunner,
    DetectorInferenceRunner,
)

from ..hardware import warn
from .optimization import InferenceProfile


@contextmanager
def patch_dlc_runner_builders(profile: InferenceProfile) -> Iterator[None]:
    """Patches DeepLabCut's inference-runner builders so ``analyze_videos`` builds optimized runners for its duration.

    ``analyze_videos`` constructs its runners by calling ``get_pose_inference_runner`` and, for top-down models,
    ``get_detector_inference_runner`` on the DeepLabCut apis-utils module. This context manager temporarily replaces
    those two functions with versions that build the stock runner and then enhance it in place with the profile's
    optimizations, restoring the originals on exit. It affects only the calling process, so each worker patches its own
    interpreter.

    Notes:
        Wrapping the stock builders is worthwhile rather than reimplementing DeepLabCut's runner setup: DeepLabCut's own
        autocast is float16-only while these models train in bfloat16. DeepLabCut also calls
        ``torch.autocast(device_type=str(self.device))`` with a device string like ``"cuda:0"`` that is not a valid
        autocast device type. The stock autocast is disabled and replaced with one carrying the correct device type and
        dtype. Because the wrappers depend on the internal structure of DeepLabCut's inference runners, the DeepLabCut
        version is pinned exactly in ``pyproject.toml`` and must be re-verified against any new release.

    Args:
        profile: The resolved optimization profile applied to every runner built while the patch is active.

    Yields:
        None, for the duration of the patch.
    """
    original_pose = dlc_apis_utils.get_pose_inference_runner
    original_detector = dlc_apis_utils.get_detector_inference_runner

    # Reentrancy guard: only the outermost runner build is optimized. A conditional-top-down build recursively
    # constructs its own bottom-up conditioning runner through this same patched function; that nested runner must stay
    # stock so the conditional-top-down path runs entirely at stock precision, as documented. A worker analyzes one
    # video at a time on a single thread, so a plain flag is sufficient here.
    building = {"active": False}

    def wrap(builder: Callable[..., InferenceRunner]) -> Callable[..., InferenceRunner]:
        def build_and_optimize(*args: Any, **kwargs: Any) -> InferenceRunner:
            if building["active"]:
                return builder(*args, **kwargs)
            building["active"] = True
            try:
                runner = builder(*args, **kwargs)
            finally:
                building["active"] = False
            return _optimize_inference_runner(runner=runner, profile=profile)

        return build_and_optimize

    dlc_apis_utils.get_pose_inference_runner = wrap(original_pose)
    dlc_apis_utils.get_detector_inference_runner = wrap(original_detector)
    try:
        yield
    finally:
        dlc_apis_utils.get_pose_inference_runner = original_pose
        dlc_apis_utils.get_detector_inference_runner = original_detector


def _optimize_inference_runner(runner: InferenceRunner, profile: InferenceProfile) -> InferenceRunner:
    """Enhances a DeepLabCut inference runner in place with mixed precision, channels-last, and ``torch.compile``.

    The runner's forward pass is replaced with a version that wraps the model call in the profile's autocast context,
    with the correct device type and bfloat16/float16 dtype. That version moves each batch to the runner device on
    every forward pass, using a non-blocking transfer only when host-memory pinning is enabled, and additionally
    converts inputs to the channels-last memory format when channels-last is enabled. When enabled by the profile, the
    model is converted to channels-last and compiled before the swap.
    Conditional-top-down runners drive a stateful, multi-stage forward that this simple swap would not preserve, so
    they are left unmodified with a warning.

    Args:
        runner: The DeepLabCut inference runner to enhance.
        profile: The resolved optimization profile to apply.

    Returns:
        The same runner instance, enhanced in place.
    """
    if isinstance(runner, CTDInferenceRunner):
        warn(
            "Conditional-top-down inference is not accelerated by the optimized runner and will run at stock "
            "precision; its frame-to-frame tracking forward pass is left unmodified."
        )
        return runner

    # DeepLabCut's own autocast is disabled through the inference config passed to analyze_videos; the disable is
    # reasserted here so the stock forward path never double-applies autocast on top of the injected one even if a
    # runner was built differently.
    runner.inference_cfg.autocast.enabled = False

    if profile.channels_last:
        runner.model = runner.model.to(memory_format=torch.channels_last)
    if profile.torch_compile:
        # torch.compile can raise a range of backend errors; the wrapper falls back to eager execution when it does.
        try:
            runner.model = torch.compile(runner.model)
        except Exception as error:
            warnings.warn(f"torch.compile failed; falling back to eager execution. Error: {error}", stacklevel=2)

    device_type = "cuda" if str(runner.device).startswith("cuda") else str(runner.device)
    amp_dtype = profile.amp_dtype
    channels_last = profile.channels_last

    def autocast_context() -> AbstractContextManager[None]:
        """Returns the autocast context for the forward pass, or a null context when mixed precision is off."""
        if amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device_type, dtype=amp_dtype)

    def move_inputs(inputs: torch.Tensor) -> torch.Tensor:
        """Moves a batch to the runner device, adopting the channels-last format when enabled."""
        moved = inputs.to(runner.device)
        if channels_last:
            moved = moved.contiguous(memory_format=torch.channels_last)
        return moved

    if isinstance(runner, DetectorInferenceRunner):
        runner.predict = _build_detector_predict(
            runner=runner, autocast_context=autocast_context, move_inputs=move_inputs
        )
    else:
        runner.predict = _build_pose_predict(runner=runner, autocast_context=autocast_context, move_inputs=move_inputs)
    return runner


def _build_pose_predict(
    runner: PoseInferenceRunner,
    autocast_context: Callable[[], AbstractContextManager[None]],
    move_inputs: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[..., list[dict[str, dict[str, Any]]]]:
    """Builds the optimized pose ``predict`` replacement bound to the runner's model and dynamic cropper.

    Args:
        runner: The pose runner being enhanced.
        autocast_context: A no-argument callable returning the autocast context for the forward pass.
        move_inputs: A callable that moves a batch to the device with the configured memory format.

    Returns:
        A ``predict(inputs, **kwargs)`` callable mirroring ``PoseInferenceRunner.predict`` with the injected
        autocast applied.
    """

    def predict(inputs: torch.Tensor, **kwargs: Any) -> list[dict[str, dict[str, Any]]]:
        batch_size = len(inputs)
        if runner.dynamic is not None:
            inputs = runner.dynamic.crop(inputs)
        with autocast_context():
            outputs = runner.model(move_inputs(inputs), **kwargs)
            raw_predictions = runner.model.get_predictions(outputs)
        if runner.dynamic is not None:
            raw_predictions["bodypart"]["poses"] = runner.dynamic.update(raw_predictions["bodypart"]["poses"])
        # Copies each output tensor to host once, then indexes the resulting array per frame, rather than issuing a
        # separate device-to-host copy for every frame of every tensor.
        host_predictions = {
            head: {name: prediction.cpu().numpy() for name, prediction in head_outputs.items()}
            for head, head_outputs in raw_predictions.items()
        }
        return [
            {
                head: {name: array[index] for name, array in head_arrays.items()}
                for head, head_arrays in host_predictions.items()
            }
            for index in range(batch_size)
        ]

    return predict


def _build_detector_predict(
    runner: DetectorInferenceRunner,
    autocast_context: Callable[[], AbstractContextManager[None]],
    move_inputs: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[..., list[dict[str, dict[str, Any]]]]:
    """Builds the optimized detector ``predict`` replacement bound to the runner's model.

    Args:
        runner: The detector runner being enhanced.
        autocast_context: A no-argument callable returning the autocast context for the forward pass.
        move_inputs: A callable that moves a batch to the device with the configured memory format.

    Returns:
        A ``predict(inputs, **kwargs)`` callable mirroring ``DetectorInferenceRunner.predict`` with the injected
        autocast.
    """

    def predict(inputs: torch.Tensor, **kwargs: Any) -> list[dict[str, dict[str, Any]]]:
        with autocast_context():
            _, raw_predictions = runner.model(move_inputs(inputs))
        return [
            {
                "detection": {
                    "bboxes": item["boxes"].cpu().numpy().reshape(-1, 4),
                    "scores": item["scores"].cpu().numpy().reshape(-1),
                }
            }
            for item in raw_predictions
        ]

    return predict
