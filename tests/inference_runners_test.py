"""Contains tests for the DeepLabCut inference-runner optimization wrappers in ``inference/runners.py``.

The DeepLabCut runner classes are never fully constructed here: heavy runner objects are fabricated with
``object.__new__`` and given only the attributes the functions under test read, and the DLC apis-utils builder
functions and ``torch.compile`` are monkeypatched. Every optimization path runs on the CPU with tiny synthetic
tensors, so no GPU, network, or real model is required.
"""

from types import SimpleNamespace
from contextlib import nullcontext

import torch
import pytest
import deeplabcut.pose_estimation_pytorch.apis.utils as dlc_apis_utils
from deeplabcut.pose_estimation_pytorch.runners.inference import (
    CTDInferenceRunner,
    PoseInferenceRunner,
    DetectorInferenceRunner,
)

from sollertia_video_tracking.inference import runners
from sollertia_video_tracking.inference.optimization import InferenceProfile


def _make_profile(
    *,
    device="cpu",
    amp_dtype=None,
    channels_last=False,
    torch_compile=False,
):
    """Builds a fully specified InferenceProfile. Only the fields the runners module reads matter here."""
    return InferenceProfile(
        device=device,
        gpus=(),
        gpu_processes=0,
        chunks=1,
        cpu_workers=1,
        cpu_threads_per_worker=8,
        amp_dtype=amp_dtype,
        tf32=False,
        cudnn_benchmark=False,
        channels_last=channels_last,
        torch_compile=torch_compile,
    )


class _FakePoseModel:
    """Stands in for a DLC pose model, exposing the ``to``, ``__call__``, and ``get_predictions`` surface used."""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.to_calls = []
        self.last_inputs = None
        self.last_kwargs = None
        self.got_outputs = None

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self

    def __call__(self, inputs, **kwargs):
        self.last_inputs = inputs
        self.last_kwargs = kwargs
        return "raw-outputs"

    def get_predictions(self, outputs):
        self.got_outputs = outputs
        # Two heads, each a per-frame tensor whose first dimension is the batch. Values are unique so per-frame
        # indexing can be asserted precisely.
        return {
            "bodypart": {
                "poses": torch.arange(self.batch_size * 2, dtype=torch.float32).reshape(self.batch_size, 2),
                "unin": torch.arange(self.batch_size, dtype=torch.float32).reshape(self.batch_size, 1),
            },
            "unique": {
                "poses": torch.arange(self.batch_size * 3, dtype=torch.float32).reshape(self.batch_size, 3),
            },
        }


class _FakeDetectorModel:
    """Stands in for a DLC detector model, returning ``(losses, raw_predictions)`` per its forward contract."""

    def __init__(self, items):
        self.items = items
        self.to_calls = []
        self.last_inputs = None

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self

    def __call__(self, inputs):
        self.last_inputs = inputs
        return "losses", self.items


class _FakeDynamic:
    """Records the crop/update calls of a dynamic cropper.

    ``crop`` passes tensors through unchanged. ``update`` applies a distinct, observable offset so a test can prove the
    optimizer stores the update result back into the bodypart poses rather than discarding it.
    """

    # A recognizable, exactly representable offset that update() adds to every pose coordinate.
    UPDATE_OFFSET = 100.0

    def __init__(self):
        self.cropped = None
        self.updated = None

    def crop(self, inputs):
        self.cropped = inputs
        return inputs

    def update(self, poses):
        self.updated = poses
        return poses + self.UPDATE_OFFSET


class _FakeBatch:
    """Stands in for a batch, reporting a fixed length and yielding a real CPU tensor from ``.to``.

    It lets the pose predict body run to completion while a device string like ``"cuda:0"`` is configured, since a real
    ``tensor.to("cuda:0")`` would fail on a CPU-only box. It also records the device it was asked to move to, so a test
    can confirm ``move_inputs`` targeted the runner's actual device.
    """

    def __init__(self, *, length, moved):
        self._length = length
        self._moved = moved
        self.to_device = None
        self.to_non_blocking = None

    def __len__(self):
        return self._length

    def to(self, device, *, non_blocking=False):
        self.to_device = device
        self.to_non_blocking = non_blocking
        return self._moved


def _new_pose_runner(*, batch_size=3, device="cpu", dynamic=None):
    """Fabricates a PoseInferenceRunner with only the attributes the optimizer and pose-predict path read."""
    runner = object.__new__(PoseInferenceRunner)
    runner.inference_cfg = SimpleNamespace(autocast=SimpleNamespace(enabled=True))
    runner.model = _FakePoseModel(batch_size=batch_size)
    runner.device = device
    runner.dynamic = dynamic
    runner.predict = "STOCK"  # sentinel replaced by the optimizer when it runs
    return runner


def _new_detector_runner(*, items, device="cpu"):
    """Fabricates a DetectorInferenceRunner with only the attributes the optimizer and detector-predict path read."""
    runner = object.__new__(DetectorInferenceRunner)
    runner.inference_cfg = SimpleNamespace(autocast=SimpleNamespace(enabled=True))
    runner.model = _FakeDetectorModel(items=items)
    runner.device = device
    runner.predict = "STOCK"  # sentinel replaced by the optimizer when it runs
    return runner


def _new_ctd_runner():
    """Fabricates a CTDInferenceRunner. The optimizer returns early for these, so it needs no readable attributes."""
    runner = object.__new__(CTDInferenceRunner)
    runner.predict = "STOCK"  # sentinel that must remain untouched for a CTD runner
    return runner


def _detector_items():
    """Returns a two-detection raw prediction list matching the detector model's forward output shape."""
    return [
        {"boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]), "scores": torch.tensor([0.9])},
        {
            "boxes": torch.tensor([[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]),
            "scores": torch.tensor([0.8, 0.7]),
        },
    ]


# _optimize_inference_runner: dispatch and in-place enhancement


def test_optimize_ctd_runner_is_left_stock_and_warns(capsys):
    """Verifies that conditional-top-down runners are returned untouched with a stderr warning."""
    runner = _new_ctd_runner()
    result = runners._optimize_inference_runner(runner=runner, profile=_make_profile())
    assert result is runner
    assert runner.predict == "STOCK"  # predict was not swapped
    err = capsys.readouterr().err
    assert "Conditional-top-down" in err


def test_optimize_disables_stock_autocast_and_swaps_predict():
    """Verifies that a pose runner has its stock autocast disabled and its predict replaced with a callable."""
    runner = _new_pose_runner()
    result = runners._optimize_inference_runner(runner=runner, profile=_make_profile())
    assert result is runner
    assert runner.inference_cfg.autocast.enabled is False
    assert callable(runner.predict)
    assert runner.predict != "STOCK"


def test_optimize_detector_dispatch_selects_detector_predict():
    """Verifies that a detector runner is routed to the detector predict, whose output carries the "detection" head."""
    runner = _new_detector_runner(items=_detector_items())
    runners._optimize_inference_runner(runner=runner, profile=_make_profile())
    output = runner.predict(torch.zeros(2, 5))
    assert [set(frame) for frame in output] == [{"detection"}, {"detection"}]


def test_optimize_applies_channels_last_to_model():
    """Verifies that with channels_last enabled the model is converted via .to(memory_format=channels_last)."""
    runner = _new_pose_runner()
    runners._optimize_inference_runner(runner=runner, profile=_make_profile(channels_last=True))
    assert runner.model.to_calls == [((), {"memory_format": torch.channels_last})]


def test_optimize_torch_compile_success(monkeypatch):
    """Verifies that when torch.compile succeeds the runner's model is replaced by the returned compiled object."""
    # Returning a distinct sentinel (rather than the same model) makes the store-back observable: a version that called
    # compile but dropped the assignment would leave runner.model unchanged and fail here.
    compiled_sentinel = object()
    seen = {"model": None}

    def fake_compile(model):
        seen["model"] = model
        return compiled_sentinel

    monkeypatch.setattr(runners.torch, "compile", fake_compile)
    runner = _new_pose_runner()
    original_model = runner.model
    runners._optimize_inference_runner(runner=runner, profile=_make_profile(torch_compile=True))
    assert seen["model"] is original_model  # compile was handed the runner's original model
    assert runner.model is compiled_sentinel  # and the compiled result was stored back on the runner


def test_optimize_torch_compile_failure_falls_back_with_warning(monkeypatch):
    """Verifies that a torch.compile backend error is swallowed and downgraded to a warning, with eager fallback."""

    def boom(_model):
        error_message = "backend go boom"
        raise RuntimeError(error_message)

    monkeypatch.setattr(runners.torch, "compile", boom)
    runner = _new_pose_runner()
    original_model = runner.model
    with pytest.warns(UserWarning, match="torch.compile failed; falling back to eager"):
        runners._optimize_inference_runner(runner=runner, profile=_make_profile(torch_compile=True))
    assert runner.model is original_model  # model unchanged after the failed compile


def test_optimize_cuda_device_type_branch(monkeypatch):
    """Verifies that a "cuda:N"-prefixed device string collapses to the "cuda" autocast device type."""
    # torch.autocast rejects a bare "cuda:0". The stub captures the arguments torch.autocast receives and drives the
    # predict closure with a _FakeBatch, so no real GPU is needed. The assertion fails if the ternary took the else
    # branch (device_type would be the full "cuda:0" string) or forwarded the wrong dtype.
    recorded = {}

    def fake_autocast(*, device_type, dtype):
        recorded["device_type"] = device_type
        recorded["dtype"] = dtype
        return nullcontext()

    monkeypatch.setattr(runners.torch, "autocast", fake_autocast)
    runner = _new_pose_runner(batch_size=2, device="cuda:0")
    runners._optimize_inference_runner(runner=runner, profile=_make_profile(amp_dtype=torch.float16))

    batch = _FakeBatch(length=2, moved=torch.zeros(2, 5))
    output = runner.predict(batch)

    assert recorded["device_type"] == "cuda"  # cuda:0 -> "cuda", not the full device string
    assert recorded["dtype"] is torch.float16  # profile amp_dtype forwarded verbatim
    assert batch.to_device == "cuda:0"  # move_inputs targeted the runner's actual (full) device
    assert len(output) == 2  # the predict body ran end-to-end under the injected autocast context


def test_optimize_non_cuda_device_type_passthrough(monkeypatch):
    """Verifies that a non-"cuda" device string is forwarded to torch.autocast verbatim, never collapsed."""
    # The else branch of the device-type ternary (e.g. "mps"). This pins the branch that the cuda test above
    # deliberately does not exercise.
    recorded = {}

    def fake_autocast(*, device_type, dtype):
        recorded["device_type"] = device_type
        recorded["dtype"] = dtype
        return nullcontext()

    monkeypatch.setattr(runners.torch, "autocast", fake_autocast)
    runner = _new_pose_runner(batch_size=2, device="mps")
    runners._optimize_inference_runner(runner=runner, profile=_make_profile(amp_dtype=torch.bfloat16))

    batch = _FakeBatch(length=2, moved=torch.zeros(2, 5))
    runner.predict(batch)

    assert recorded["device_type"] == "mps"  # non-cuda device string passes through unchanged
    assert recorded["dtype"] is torch.bfloat16
    assert batch.to_device == "mps"


# pose predict path


def test_pose_predict_no_dynamic_no_amp():
    """Verifies that with amp off and no dynamic cropper, predict returns one dict per frame with indexed arrays."""
    runner = _new_pose_runner(batch_size=3, dynamic=None)
    runners._optimize_inference_runner(runner=runner, profile=_make_profile(amp_dtype=None))
    inputs = torch.zeros(3, 5)
    output = runner.predict(inputs)

    assert len(output) == 3
    for index, frame in enumerate(output):
        assert set(frame) == {"bodypart", "unique"}
        assert frame["bodypart"]["poses"].tolist() == [float(index * 2), float(index * 2 + 1)]
        assert frame["unique"]["poses"].tolist() == [float(index * 3), float(index * 3 + 1), float(index * 3 + 2)]
    # Model received the moved inputs and the stock autocast is disabled.
    assert runner.model.last_inputs is not None
    assert runner.inference_cfg.autocast.enabled is False


def test_pose_predict_forwards_kwargs_to_model():
    """Verifies that extra keyword arguments to predict are forwarded to the model forward call."""
    runner = _new_pose_runner(batch_size=2)
    runners._optimize_inference_runner(runner=runner, profile=_make_profile())
    runner.predict(torch.zeros(2, 4), extra="value")
    assert runner.model.last_kwargs == {"extra": "value"}


def test_pose_predict_with_dynamic_channels_last_and_amp():
    """Verifies that a dynamic cropper, channels_last, and amp run crop/update under a real CPU autocast context."""
    dynamic = _FakeDynamic()
    runner = _new_pose_runner(batch_size=2, dynamic=dynamic)
    profile = _make_profile(amp_dtype=torch.bfloat16, channels_last=True)
    runners._optimize_inference_runner(runner=runner, profile=profile)

    inputs = torch.zeros(2, 3, 4, 4)  # NCHW so the channels-last contiguous conversion is valid
    output = runner.predict(inputs)

    assert dynamic.cropped is inputs  # crop was called with the raw batch
    assert dynamic.updated is not None  # update was called with the bodypart poses
    assert len(output) == 2
    # The update RESULT (poses + UPDATE_OFFSET) must be stored back into the bodypart poses, not discarded. The fake
    # model emits arange-valued bodypart poses ([[0, 1], [2, 3]] for batch_size 2), so after the offset each frame's
    # poses shift by exactly UPDATE_OFFSET. Non-bodypart heads (unique) are untouched by update.
    offset = _FakeDynamic.UPDATE_OFFSET
    assert output[0]["bodypart"]["poses"].tolist() == [0.0 + offset, 1.0 + offset]
    assert output[1]["bodypart"]["poses"].tolist() == [2.0 + offset, 3.0 + offset]
    assert output[0]["unique"]["poses"].tolist() == [0.0, 1.0, 2.0]  # unique head not shifted by the dynamic update
    # The moved inputs carry the channels-last memory format.
    assert runner.model.last_inputs.is_contiguous(memory_format=torch.channels_last)


# detector predict path


def test_detector_predict_no_amp():
    """Verifies that detector predict reshapes boxes to (-1, 4) and scores to (-1) per detection into host arrays."""
    items = _detector_items()
    runner = _new_detector_runner(items=items)
    runners._optimize_inference_runner(runner=runner, profile=_make_profile(amp_dtype=None))
    output = runner.predict(torch.zeros(2, 5))

    assert len(output) == 2
    assert output[0]["detection"]["bboxes"].shape == (1, 4)
    assert output[0]["detection"]["scores"].tolist() == pytest.approx([0.9])
    assert output[1]["detection"]["bboxes"].shape == (2, 4)
    assert output[1]["detection"]["scores"].tolist() == pytest.approx([0.8, 0.7])


def test_detector_predict_returns_empty_for_a_batch_with_no_detections():
    """Verifies that a batch the detector returns nothing for yields no rows rather than concatenating empty tensors."""
    runner = _new_detector_runner(items=[])
    runners._optimize_inference_runner(runner=runner, profile=_make_profile(amp_dtype=None))
    assert runner.predict(torch.zeros(2, 5)) == []


def test_detector_predict_channels_last_and_amp():
    """Verifies that with channels_last and bfloat16 autocast the detector move path runs contiguous under autocast."""
    runner = _new_detector_runner(items=_detector_items())
    profile = _make_profile(amp_dtype=torch.bfloat16, channels_last=True)
    runners._optimize_inference_runner(runner=runner, profile=profile)
    inputs = torch.zeros(2, 3, 4, 4)
    output = runner.predict(inputs)
    assert len(output) == 2
    assert runner.model.last_inputs.is_contiguous(memory_format=torch.channels_last)


# patch_dlc_runner_builders: context manager wrapping / restoration / reentrancy


def test_patch_wraps_builders_and_restores(monkeypatch):
    """Verifies that inside the context the DLC builders are replaced by optimizing wrappers and restored on exit."""
    pose_calls = {"n": 0}
    detector_calls = {"n": 0}

    def fake_pose_builder(*args, **kwargs):
        pose_calls["n"] += 1
        return _new_pose_runner()

    def fake_detector_builder(*args, **kwargs):
        detector_calls["n"] += 1
        return _new_detector_runner(items=_detector_items())

    monkeypatch.setattr(dlc_apis_utils, "get_pose_inference_runner", fake_pose_builder)
    monkeypatch.setattr(dlc_apis_utils, "get_detector_inference_runner", fake_detector_builder)

    with runners.patch_dlc_runner_builders(_make_profile()):
        assert dlc_apis_utils.get_pose_inference_runner is not fake_pose_builder
        assert dlc_apis_utils.get_detector_inference_runner is not fake_detector_builder

        pose_runner = dlc_apis_utils.get_pose_inference_runner()
        detector_runner = dlc_apis_utils.get_detector_inference_runner()
        assert pose_calls["n"] == 1
        assert detector_calls["n"] == 1
        # The wrapped builders return optimized runners: predict is swapped and stock autocast disabled.
        assert callable(pose_runner.predict)
        assert pose_runner.predict != "STOCK"
        assert callable(detector_runner.predict)
        assert detector_runner.predict != "STOCK"
        assert pose_runner.inference_cfg.autocast.enabled is False

    # Originals restored on exit.
    assert dlc_apis_utils.get_pose_inference_runner is fake_pose_builder
    assert dlc_apis_utils.get_detector_inference_runner is fake_detector_builder


def test_patch_reentrancy_guard_leaves_nested_build_stock(monkeypatch):
    """Verifies that a runner recursively built through the patched builder stays stock, not optimized."""
    # Only the outermost build is optimized. The nested build hits the reentrancy guard and stays stock.
    nested = {}

    def outer_pose_builder(*args, **kwargs):
        # While this outer build is active, calling the patched detector builder hits the reentrancy guard.
        nested["runner"] = dlc_apis_utils.get_detector_inference_runner()
        return _new_pose_runner()

    def inner_detector_builder(*args, **kwargs):
        return _new_detector_runner(items=_detector_items())

    monkeypatch.setattr(dlc_apis_utils, "get_pose_inference_runner", outer_pose_builder)
    monkeypatch.setattr(dlc_apis_utils, "get_detector_inference_runner", inner_detector_builder)

    with runners.patch_dlc_runner_builders(_make_profile()):
        outer_runner = dlc_apis_utils.get_pose_inference_runner()

    # Outer runner was optimized. The nested runner built during the guard stayed stock.
    assert callable(outer_runner.predict)
    assert outer_runner.predict != "STOCK"
    assert nested["runner"].predict == "STOCK"
    assert nested["runner"].inference_cfg.autocast.enabled is True
