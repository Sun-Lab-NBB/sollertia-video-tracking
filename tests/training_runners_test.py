"""Tests for the mixed-precision / DistributedDataParallel training-runner wrappers in ``training/runners.py``.

The DeepLabCut runner classes are never fully constructed for the method-level tests: heavy runner objects are
fabricated with ``object.__new__`` and given only the attributes the method under test reads. Multi-GPU and
``torch.compile`` wrappers are monkeypatched with lightweight recorders, and every path runs on the CPU with tiny
synthetic tensors, so no GPU, network, or real model/training is required. The builder tests construct the real runner
subclasses end-to-end on the CPU with a tiny ``nn.Linear`` model to exercise ``build_optimized_training_runner`` and the
mixin ``__init__``.
"""

from types import SimpleNamespace
import contextlib

import numpy as np
import torch
from torch import nn
import pytest
from torch.nn.parallel import DataParallel, DistributedDataParallel
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.runners.logger import ImageLoggerMixin

from sollertia_video_tracking.training import runners

# ----------------------------------------------------------------------------------------------------------------------
# Shared fakes
# ----------------------------------------------------------------------------------------------------------------------


class FakeOptimizer:
    """Records ``zero_grad``/``step`` and exposes the ``param_groups`` the runner reads for the learning rate."""

    def __init__(self):
        self.zeroed = 0
        self.stepped = 0
        self.param_groups = [{"lr": 0.01}]

    def zero_grad(self):
        self.zeroed += 1

    def step(self):
        self.stepped += 1


class FakeScaler:
    """A stand-in gradient scaler that records the scale/step/update sequence and passes the loss through unchanged."""

    def __init__(self):
        self.calls = []

    def scale(self, loss):
        self.calls.append("scale")
        return loss

    def step(self, optimizer):
        self.stepped_optimizer = optimizer
        self.calls.append("step")

    def update(self):
        self.calls.append("update")


class FakePrepModel:
    """Minimal model stand-in for the ``_prepare_model_for_training`` and ``fit`` paths."""

    def __init__(self):
        self.to_device = None
        self.trained = 0
        self.evaled = 0

    def to(self, device):
        self.to_device = device
        return self

    def train(self):
        self.trained += 1

    def eval(self):
        self.evaled += 1

    def state_dict(self):
        return {"weight": 1}


class FakeImageLogger(ImageLoggerMixin):
    """Concrete ImageLoggerMixin so ``isinstance`` checks pass; records the calls the runner makes against it."""

    def __init__(self):
        self.selected = None
        self.logged = []
        self.image_steps = []

    def select_images_to_log(self, train, valid):
        self.selected = (train, valid)

    def log(self, metrics, step=None):
        self.logged.append((metrics, step))

    def log_images(self, inputs, outputs, targets, step):
        self.image_steps.append(step)
        self.last_logged = (inputs, outputs, targets)


class FakeLoader:
    """Iterable data-loader stand-in with an optional ``sampler`` exposing ``set_epoch`` for the DDP path."""

    def __init__(self, batches, sampler=None):
        self.batches = batches
        self.sampler = sampler

    def __iter__(self):
        return iter(self.batches)


def new_pose_runner():
    """Fabricates an ``_OptimizedPoseTrainingRunner`` shell without running the heavy DeepLabCut constructor."""
    return object.__new__(runners._OptimizedPoseTrainingRunner)


def new_detector_runner():
    """Fabricates an ``_OptimizedDetectorTrainingRunner`` shell without running the heavy DeepLabCut constructor."""
    return object.__new__(runners._OptimizedDetectorTrainingRunner)


# ----------------------------------------------------------------------------------------------------------------------
# build_optimized_training_runner: real end-to-end construction on the CPU
# ----------------------------------------------------------------------------------------------------------------------


def _base_runner_config():
    """Returns a runner configuration accepted by DeepLabCut's optimizer/scheduler/snapshot builders."""
    return {
        "optimizer": {"type": "SGD", "params": {"lr": 0.01}},
        "scheduler": {"type": "StepLR", "params": {"step_size": 10, "gamma": 0.1}},
        "snapshots": {"max_snapshots": 3, "save_epochs": 5, "save_optimizer_state": True},
        "eval_interval": 2,
    }


def test_build_pose_runner_defaults_prefix_and_gradient_scaler(tmp_path):
    # A BOTTOM_UP task builds a pose runner; with no snapshot_prefix it falls back to the task default, and
    # use_gradient_scaler=True constructs a real GradScaler (line 179), while load_head_weights threads through.
    runner = runners.build_optimized_training_runner(
        runner_config=_base_runner_config(),
        model_folder=tmp_path,
        task=Task.BOTTOM_UP,
        model=nn.Linear(3, 2),
        device="cpu",
        amp_dtype=torch.bfloat16,
        use_gradient_scaler=True,
        load_head_weights=False,
    )
    assert isinstance(runner, runners._OptimizedPoseTrainingRunner)
    assert runner._load_head_weights is False
    assert runner._gradient_scaler is not None
    assert runner._amp_dtype is torch.bfloat16
    assert runner.snapshot_manager.snapshot_prefix == "snapshot"  # task default
    assert runner.scheduler is not None


def test_build_detector_runner_custom_prefix_no_scaler(tmp_path):
    # A DETECT task builds a detector runner; an explicit snapshot_prefix wins, a None scheduler stays None, and
    # use_gradient_scaler defaults to False so no scaler is built.
    config = _base_runner_config()
    config["scheduler"] = None
    config["snapshot_prefix"] = "custom-prefix"
    runner = runners.build_optimized_training_runner(
        runner_config=config,
        model_folder=tmp_path,
        task=Task.DETECT,
        model=nn.Linear(2, 2),
        device="cpu",
    )
    assert isinstance(runner, runners._OptimizedDetectorTrainingRunner)
    assert runner._gradient_scaler is None
    assert runner._ddp_static_graph is False  # detector overrides the class attribute
    assert runner.snapshot_manager.snapshot_prefix == "custom-prefix"
    assert runner.scheduler is None


# ----------------------------------------------------------------------------------------------------------------------
# _is_main
# ----------------------------------------------------------------------------------------------------------------------


def test_is_main_without_ddp_is_always_main():
    runner = new_pose_runner()
    runner._ddp = False
    runner._rank = 7  # ignored when DDP is off
    assert runner._is_main is True


def test_is_main_with_ddp_rank_zero():
    runner = new_pose_runner()
    runner._ddp = True
    runner._rank = 0
    assert runner._is_main is True


def test_is_main_with_ddp_nonzero_rank_is_not_main():
    runner = new_pose_runner()
    runner._ddp = True
    runner._rank = 3
    assert runner._is_main is False


# ----------------------------------------------------------------------------------------------------------------------
# _unwrap
# ----------------------------------------------------------------------------------------------------------------------


def test_unwrap_plain_model_returns_itself():
    runner = new_pose_runner()
    model = nn.Linear(2, 2)
    runner.model = model
    assert runner._unwrap() is model


def test_unwrap_peels_torch_compile_orig_mod():
    # A compiled model exposes ``_orig_mod``; _unwrap peels it so snapshot keys stay clean.
    runner = new_pose_runner()
    runner.model = SimpleNamespace(_orig_mod="ORIGINAL")
    assert runner._unwrap() == "ORIGINAL"


def test_unwrap_data_parallel_then_orig_mod():
    # A DataParallel wrapper is peeled to ``.module``; a compiled inner model then has ``_orig_mod`` peeled too.
    runner = new_pose_runner()
    dp = object.__new__(DataParallel)
    dp.module = SimpleNamespace(_orig_mod="INNER")
    runner.model = dp
    assert runner._unwrap() == "INNER"


def test_unwrap_distributed_data_parallel_module():
    runner = new_pose_runner()
    ddp = object.__new__(DistributedDataParallel)
    # A plain sentinel (not an nn.Module) avoids torch's "assign module before __init__" guard; without an
    # ``_orig_mod`` attribute, _unwrap returns the module itself.
    inner = SimpleNamespace(name="inner-model")
    ddp.module = inner
    runner.model = ddp
    assert runner._unwrap() is inner


# ----------------------------------------------------------------------------------------------------------------------
# _build_autocast_context
# ----------------------------------------------------------------------------------------------------------------------


def test_autocast_context_float32_is_nullcontext():
    runner = new_pose_runner()
    runner._amp_dtype = None
    runner.device = "cpu"
    assert isinstance(runner._build_autocast_context(), contextlib.nullcontext)


def test_autocast_context_disabled_flag_forces_nullcontext():
    runner = new_pose_runner()
    runner._amp_dtype = torch.bfloat16
    runner.device = "cpu"
    assert isinstance(runner._build_autocast_context(enabled=False), contextlib.nullcontext)


def test_autocast_context_cpu_device_type_enters():
    runner = new_pose_runner()
    runner._amp_dtype = torch.bfloat16
    runner.device = "cpu"
    ctx = runner._build_autocast_context(enabled=True)
    assert isinstance(ctx, torch.autocast)
    # The context must carry the runner's device and compute dtype, not just be some autocast object.
    assert ctx.device == "cpu"
    assert ctx.fast_dtype == torch.bfloat16
    a = torch.ones(2, 2)
    b = torch.ones(2, 2)
    assert (a @ b).dtype == torch.float32  # baseline: no autocast in force
    with ctx:  # entering on a CPU box is valid; autocast must actually downcast the matmul it wraps
        assert (a @ b).dtype == torch.bfloat16


def test_autocast_context_cuda_device_type_resolved():
    # A "cuda"-prefixed device resolves the autocast device type to "cuda"; the context is only constructed, not
    # entered, so no real GPU is required.
    runner = new_pose_runner()
    runner._amp_dtype = torch.float16
    runner.device = "cuda:1"
    ctx = runner._build_autocast_context(enabled=True)
    assert isinstance(ctx, torch.autocast)
    assert ctx.device == "cuda"


# ----------------------------------------------------------------------------------------------------------------------
# _backward_and_step
# ----------------------------------------------------------------------------------------------------------------------


def test_backward_and_step_without_scaler():
    runner = new_pose_runner()
    runner._gradient_scaler = None
    optimizer = FakeOptimizer()
    runner.optimizer = optimizer
    param = torch.zeros(1, requires_grad=True)
    loss = (param * 3).sum()
    runner._backward_and_step(loss=loss)
    assert optimizer.stepped == 1
    assert param.grad.item() == pytest.approx(3.0)


def test_backward_and_step_with_scaler():
    runner = new_pose_runner()
    scaler = FakeScaler()
    runner._gradient_scaler = scaler
    optimizer = FakeOptimizer()
    runner.optimizer = optimizer
    param = torch.zeros(1, requires_grad=True)
    loss = (param * 5).sum()
    runner._backward_and_step(loss=loss)
    assert scaler.calls == ["scale", "step", "update"]
    assert optimizer.stepped == 0  # the scaler drives the optimizer step, not the runner directly
    assert param.grad.item() == pytest.approx(5.0)


# ----------------------------------------------------------------------------------------------------------------------
# _prepare_model_for_training
# ----------------------------------------------------------------------------------------------------------------------


def _prep_runner(cls, *, torch_compile=False, ddp=False, data_parallel=False, gpus=None, local_rank=0):
    """Fabricates a runner with just the flags ``_prepare_model_for_training`` reads."""
    runner = object.__new__(cls)
    runner.model = FakePrepModel()
    runner.device = "cpu"
    runner._torch_compile = torch_compile
    runner._ddp = ddp
    runner._local_rank = local_rank
    runner._data_parallel = data_parallel
    runner._gpus = gpus or []
    return runner


def test_prepare_moves_model_to_device_only():
    runner = _prep_runner(runners._OptimizedPoseTrainingRunner)
    model = runner.model
    runner._prepare_model_for_training()
    assert runner.model is model
    assert model.to_device == "cpu"


def test_prepare_applies_torch_compile(monkeypatch):
    runner = _prep_runner(runners._OptimizedPoseTrainingRunner, torch_compile=True)
    monkeypatch.setattr(runners.torch, "compile", lambda model: ("COMPILED", model))
    runner._prepare_model_for_training()
    assert runner.model[0] == "COMPILED"


def test_prepare_ddp_static_graph_for_pose(monkeypatch):
    captured = {}

    def fake_ddp(module, device_ids, output_device, **opts):
        captured["module"] = module
        captured["device_ids"] = device_ids
        captured["output_device"] = output_device
        captured["opts"] = opts
        return "DDP_WRAPPED"

    monkeypatch.setattr(runners, "DistributedDataParallel", fake_ddp)
    runner = _prep_runner(runners._OptimizedPoseTrainingRunner, ddp=True, local_rank=2)
    runner._prepare_model_for_training()
    assert runner.model == "DDP_WRAPPED"
    assert captured["device_ids"] == [2]
    assert captured["output_device"] == 2
    assert captured["opts"] == {"broadcast_buffers": False, "static_graph": True}


def test_prepare_ddp_find_unused_parameters_for_detector(monkeypatch):
    captured = {}

    def fake_ddp(module, device_ids, output_device, **opts):
        captured["module"] = module
        captured["device_ids"] = device_ids
        captured["output_device"] = output_device
        captured["opts"] = opts
        return "DDP_DETECTOR"

    monkeypatch.setattr(runners, "DistributedDataParallel", fake_ddp)
    # The detector subclass sets _ddp_static_graph = False, so the data-dependent-graph branch runs instead.
    runner = _prep_runner(runners._OptimizedDetectorTrainingRunner, ddp=True, local_rank=0)
    runner._prepare_model_for_training()
    assert runner.model == "DDP_DETECTOR"
    assert captured["opts"] == {"broadcast_buffers": False, "find_unused_parameters": True}


def test_prepare_data_parallel_branch(monkeypatch):
    class FakeDP:
        def __init__(self, module, device_ids):
            self.module = module
            self.device_ids = device_ids

        def cuda(self):
            return ("DP_CUDA", self.device_ids)

    monkeypatch.setattr(runners, "DataParallel", FakeDP)
    runner = _prep_runner(runners._OptimizedPoseTrainingRunner, data_parallel=True, gpus=[0, 1])
    runner._prepare_model_for_training()
    assert runner.model == ("DP_CUDA", [0, 1])


# ----------------------------------------------------------------------------------------------------------------------
# state_dict
# ----------------------------------------------------------------------------------------------------------------------


def test_state_dict_without_scheduler():
    runner = new_pose_runner()
    model = nn.Linear(2, 2)
    runner.model = model
    runner.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner.scheduler = None
    runner._metadata = {"epoch": 4}
    state = runner.state_dict()
    assert state["metadata"] == {"epoch": 4}
    assert set(state["model"].keys()) == set(model.state_dict().keys())
    assert "optimizer" in state
    assert "scheduler" not in state


def test_state_dict_with_scheduler():
    runner = new_pose_runner()
    model = nn.Linear(2, 2)
    runner.model = model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner.optimizer = optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
    runner.scheduler = scheduler
    runner._metadata = {"epoch": 1}
    state = runner.state_dict()
    # The scheduler branch must serialize the actual scheduler state, not merely add a truthy placeholder.
    assert state["scheduler"] == scheduler.state_dict()
    assert state["metadata"] == {"epoch": 1}
    assert set(state["model"].keys()) == set(model.state_dict().keys())


# ----------------------------------------------------------------------------------------------------------------------
# _epoch: invalid mode
# ----------------------------------------------------------------------------------------------------------------------


def test_epoch_invalid_mode_raises():
    runner = new_pose_runner()
    with pytest.raises(ValueError, match="must be 'train' or 'eval'"):
        runner._epoch(loader=[], mode="bogus")


# ----------------------------------------------------------------------------------------------------------------------
# fit (drives _epoch for both train and eval)
# ----------------------------------------------------------------------------------------------------------------------


def _fit_runner(cls, *, ddp=False, rank=0, starting_epoch=0, logger=None, print_valid_loss=True):
    """Fabricates a runner ready for ``fit``, shadowing the heavy hooks so no real model/snapshot work happens."""
    runner = object.__new__(cls)
    runner.model = FakePrepModel()
    runner.device = "cpu"
    runner.optimizer = FakeOptimizer()
    runner.scheduler = SimpleNamespace(step=lambda: steps.append("scheduler"))
    runner.eval_interval = 1
    runner.starting_epoch = starting_epoch
    runner.current_epoch = 0
    runner._ddp = ddp
    runner._rank = rank
    runner._print_valid_loss = print_valid_loss
    runner.logger = logger
    runner._metadata = {"epoch": 0, "metrics": {}, "losses": {}}
    runner.history = {"train_loss": [], "eval_loss": []}
    runner._epoch_predictions = {}
    runner._epoch_ground_truth = {}
    runner.csv_logger = SimpleNamespace(log=lambda metrics, step: csv_logs.append((metrics, step)))
    runner.snapshot_manager = SimpleNamespace(
        update=lambda epoch, state_dict, last: snapshots.append((epoch, state_dict, last)),
    )
    # Shadow the class methods that would otherwise pull in a real model / snapshot serialization.
    runner._prepare_model_for_training = lambda: prep.append(True)
    runner.state_dict = lambda: {"fake": 1}
    runner._compute_epoch_metrics = lambda: {"metrics/test.mAP": 55.5}

    def fake_step(batch, mode="train"):
        step_batches.append((batch, mode))
        return {"total_loss": np.float32(0.4), "aux": np.float32(0.1)}

    runner.step = fake_step
    return runner


# Module-level recorders reset by each fit test that reads them.
steps: list = []
csv_logs: list = []
snapshots: list = []
prep: list = []
step_batches: list = []


def test_fit_single_process_with_image_logger_and_metrics():
    # Non-DDP, main process, an ImageLoggerMixin logger, a scheduler, an evaluation epoch and metric printing.
    steps.clear()
    csv_logs.clear()
    snapshots.clear()
    prep.clear()
    step_batches.clear()

    logger = FakeImageLogger()
    runner = _fit_runner(runners._OptimizedPoseTrainingRunner, logger=logger)
    train_loader = FakeLoader([{"batch": 0}, {"batch": 1}])
    valid_loader = FakeLoader([{"batch": 2}])

    runner.fit(train_loader=train_loader, valid_loader=valid_loader, epochs=1, display_iters=1)

    assert prep == [True]  # _prepare_model_for_training ran
    assert logger.selected == (train_loader, valid_loader)  # ImageLoggerMixin image selection ran
    assert len(logger.logged) == 2  # _epoch logged once for the train pass and once for the eval pass
    assert len(csv_logs) == 2  # the csv logger mirrors those two epoch logs
    assert steps == ["scheduler"]  # scheduler stepped once
    assert snapshots == [(1, {"fake": 1}, True)]  # final epoch snapshot given the runner state and flagged last
    assert runner._metadata["metrics"] == {"metrics/test.mAP": 55.5}
    # _epoch computes the mean of the per-batch total_loss (0.4) over the 2 train batches and the 1 eval batch.
    assert runner.history["train_loss"] == [pytest.approx(0.4)]
    assert runner.history["eval_loss"] == [pytest.approx(0.4)]


def test_fit_ddp_resumes_and_barriers(monkeypatch):
    # DDP main process resuming from a snapshot: epoch budget extends, the sampler epoch is set, and the group
    # barrier fires at the end of the epoch.
    steps.clear()
    csv_logs.clear()
    snapshots.clear()
    prep.clear()
    step_batches.clear()
    barriers = []
    set_epochs = []

    monkeypatch.setattr(runners.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runners.dist, "barrier", lambda: barriers.append(True))

    runner = _fit_runner(runners._OptimizedPoseTrainingRunner, ddp=True, rank=0, starting_epoch=2, logger=None)
    sampler = SimpleNamespace(set_epoch=set_epochs.append)
    train_loader = FakeLoader([{"batch": 0}], sampler=sampler)
    valid_loader = FakeLoader([{"batch": 1}])

    runner.fit(train_loader=train_loader, valid_loader=valid_loader, epochs=1, display_iters=500)

    # starting_epoch=2 + epochs=1 -> total budget 3, so the single new epoch is epoch 3 (the last).
    assert set_epochs == [3]
    assert snapshots == [(3, {"fake": 1}, True)]
    assert barriers == [True]


# ----------------------------------------------------------------------------------------------------------------------
# _OptimizedPoseTrainingRunner.step
# ----------------------------------------------------------------------------------------------------------------------


class FakePoseModel:
    """Pose-model stand-in exposing the forward/get_target/get_loss/get_predictions surface the pose step uses."""

    def __init__(self, *, with_unique=True):
        self.weight = torch.zeros(1, requires_grad=True)
        self.forward_kwargs = None
        self.with_unique = with_unique

    def __call__(self, inputs, **kwargs):
        self.forward_inputs = inputs
        self.forward_kwargs = kwargs
        return "OUTPUTS"

    def get_target(self, outputs, labels):
        self.target_args = (outputs, labels)
        return "TARGET"

    def get_loss(self, outputs, targets):
        self.loss_args = (outputs, targets)
        return {"total_loss": (self.weight * 3).sum(), "aux": (self.weight * 2).sum()}

    def get_predictions(self, outputs):
        self.prediction_outputs = outputs
        predictions = {"bodypart": {"poses": torch.zeros(2, 4, 3)}}
        if self.with_unique:
            predictions["unique_bodypart"] = {"poses": torch.zeros(2, 1, 3)}
        return predictions


def _pose_runner_for_step(*, logger=None, epoch_predictions=None):
    runner = new_pose_runner()
    runner.model = FakePoseModel()
    runner.optimizer = FakeOptimizer()
    runner.device = "cpu"
    runner._amp_dtype = None
    runner._gradient_scaler = None
    runner.logger = logger
    runner.current_epoch = 3
    runner._epoch_predictions = {} if epoch_predictions is None else epoch_predictions
    runner._epoch_ground_truth = {}
    return runner


def test_pose_step_invalid_mode_raises():
    runner = new_pose_runner()
    with pytest.raises(ValueError, match="must be 'train' or 'eval'"):
        runner.step(batch={}, mode="bogus")


def test_pose_step_train_with_cond_keypoints_and_image_logger():
    logger = FakeImageLogger()
    runner = _pose_runner_for_step(logger=logger)
    batch = {
        "image": torch.zeros(2, 3, 4, 4),
        "context": {"cond_keypoints": torch.zeros(2, 4, 2)},
        "annotations": {"keypoints": torch.zeros(2, 4, 3)},
    }
    result = runner.step(batch=batch, mode="train")

    assert runner.optimizer.zeroed == 1
    assert runner.optimizer.stepped == 1
    assert runner.model.forward_kwargs == {"cond_kpts": batch["context"]["cond_keypoints"]}
    assert logger.image_steps == [3]  # log_images used the current epoch
    assert set(result) == {"total_loss", "aux"}
    assert isinstance(result["total_loss"], np.ndarray)
    assert result["total_loss"] == pytest.approx(0.0)  # weight starts at zero -> total_loss (weight*3) is zero
    # Only total_loss (weight*3) is back-propagated, not aux; a grad of exactly 3 proves the right tensor flowed back.
    assert runner.model.weight.grad.item() == pytest.approx(3.0)


def test_pose_step_eval_updates_predictions_with_center_and_unique():
    runner = _pose_runner_for_step(logger=None)
    batch = {
        "image": torch.zeros(2, 3, 4, 4),
        "context": {},  # no cond_keypoints -> the plain-forward branch
        "annotations": {
            "keypoints": torch.zeros(2, 4, 3),
            "with_center_keypoints": [True],  # drops the trailing center keypoint
            "keypoints_unique": torch.zeros(2, 1, 3),
        },
        "offsets": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "scales": torch.tensor([1.5, 2.0]),
    }
    result = runner.step(batch=batch, mode="eval")

    assert runner.optimizer.zeroed == 0  # no gradient step in eval
    assert set(runner._epoch_ground_truth) == {"bodyparts", "unique_bodyparts"}
    assert set(runner._epoch_predictions) == {"bodyparts", "unique_bodyparts"}
    assert set(result) == {"total_loss", "aux"}

    # The center-keypoint trim (ground_truth[..., :-1, :]) must actually drop the trailing keypoint: the stored
    # bodypart ground truth has 3 keypoints, not the 4 that were passed in, while the predictions keep all 4.
    gt_sample = runner._epoch_ground_truth["bodyparts"]["sample000000001"]
    pred_sample = runner._epoch_predictions["bodyparts"]["sample000000001"]
    assert gt_sample.shape == (3, 3)
    assert pred_sample.shape == (4, 3)
    # The per-sample offsets threaded from batch["offsets"] land on the stored (zero) ground truth: 0*scale + offset.
    assert gt_sample[0, :2].tolist() == [1.0, 2.0]  # offsets[0]
    # The unique-bodypart branch stored its own (untrimmed) 1-keypoint prediction.
    assert runner._epoch_predictions["unique_bodyparts"]["sample000000001"].shape == (1, 3)


# ----------------------------------------------------------------------------------------------------------------------
# _OptimizedDetectorTrainingRunner.step
# ----------------------------------------------------------------------------------------------------------------------


class FakeDetectorModel:
    """Detector-model stand-in returning ``(losses, predictions)`` and exposing train/eval/get_target."""

    def __init__(self):
        self.weight = torch.zeros(1, requires_grad=True)
        self.trained = 0
        self.evaled = 0
        self.forward_args = None

    def train(self):
        self.trained += 1

    def eval(self):
        self.evaled += 1

    def get_target(self, annotations):
        self.target_annotations = annotations
        # A per-image target with one populated tensor and one None value to drive both branches of the move loop.
        return [{"boxes": torch.zeros(1, 4), "labels": None}]

    def __call__(self, images, target):
        self.forward_args = (images, target)
        losses = {"loss_box": (self.weight * 3).sum(), "loss_cls": (self.weight * 2).sum()}
        predictions = [{"boxes": torch.tensor([[5.0, 5.0, 15.0, 15.0]]), "scores": torch.tensor([0.9])}]
        return losses, predictions


def _detector_runner_for_step():
    runner = new_detector_runner()
    runner.model = FakeDetectorModel()
    runner.optimizer = FakeOptimizer()
    runner.device = "cpu"
    runner._amp_dtype = None
    runner._gradient_scaler = None
    runner._epoch_predictions = {}
    runner._epoch_ground_truth = {}
    return runner


def test_detector_step_invalid_mode_raises():
    runner = new_detector_runner()
    with pytest.raises(ValueError, match="must be 'train' or 'eval'"):
        runner.step(batch={}, mode="bogus")


def test_detector_step_train_sums_losses_and_backpropagates():
    runner = _detector_runner_for_step()
    batch = {"image": torch.zeros(1, 3, 4, 4), "annotations": {"boxes": [torch.zeros(1, 4)]}}
    result = runner.step(batch=batch, mode="train")

    assert runner.model.trained == 1
    assert runner.optimizer.zeroed == 1
    assert runner.optimizer.stepped == 1
    assert set(result) == {"loss_box", "loss_cls", "total_loss"}
    assert isinstance(result["total_loss"], np.ndarray)
    assert result["total_loss"] == pytest.approx(0.0)  # weights start at zero
    # total_loss = sum(loss_box=weight*3, loss_cls=weight*2); a grad of exactly 5 proves both parts were summed
    # into the tensor that was back-propagated (not just one loss term).
    assert runner.model.weight.grad.item() == pytest.approx(5.0)


def test_detector_step_eval_updates_predictions():
    runner = _detector_runner_for_step()
    batch = {
        "image": torch.zeros(1, 3, 4, 4),
        "annotations": {"boxes": [torch.tensor([[10.0, 10.0, 20.0, 20.0], [0.0, 0.0, 0.0, 0.0]])]},
        "path": ["image-0"],
        "original_size": [torch.tensor([480, 640])],
        "offsets": [torch.tensor([0.0, 0.0])],
        "scales": [torch.tensor([1.0, 1.0])],
    }
    result = runner.step(batch=batch, mode="eval")

    assert runner.model.evaled == 1
    assert np.isnan(result["total_loss"])
    assert "image-0" in runner._epoch_ground_truth
    assert "image-0" in runner._epoch_predictions

    # The stored record must reflect the arguments the step threaded through (sizes, bboxes, predictions), not just
    # that some key was created. original_size=[480, 640] -> stored width/height; the zero-area second gt box is
    # dropped, leaving the single visible box; the predicted box converts to COCO xywh ([5,5,15,15] -> [5,5,10,10]).
    gt_record = runner._epoch_ground_truth["image-0"]
    pred_record = runner._epoch_predictions["image-0"]
    assert int(gt_record["width"]) == 640
    assert int(gt_record["height"]) == 480
    assert gt_record["bboxes"].shape == (1, 4)  # the [0,0,0,0] box was masked out
    assert gt_record["bboxes"][0].tolist() == [10.0, 10.0, 20.0, 20.0]
    assert pred_record["bboxes"][0].tolist() == [5.0, 5.0, 10.0, 10.0]
    assert pred_record["scores"].tolist() == [pytest.approx(0.9)]
