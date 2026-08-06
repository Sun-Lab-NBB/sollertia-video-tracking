"""Contains tests for the DeepLabCut training pipeline orchestration.

These tests drive the real orchestration logic in ``training.pipeline`` on a headless, GPU-free box by monkeypatching
only the module-level bindings that would otherwise touch a real DLC runtime, CUDA device, multiprocessing pool, or the
process's file descriptors. Every DLC handoff (``DLCLoader``, model builders, the training runner, evaluation) and every
process-management call (``mp.spawn``, ``dist.*``, ``torch.cuda.set_device``) is replaced with a lightweight recording
stub. The pipeline's own branching, config assembly, worker wiring, and reporting therefore run in-process without any
GPU, network, or long-running training.
"""

import os
import types
import socket
from typing import ClassVar
import logging
from pathlib import Path

import torch
import pytest
from deeplabcut.pose_estimation_pytorch.task import Task

from sollertia_video_tracking.training import pipeline
from sollertia_video_tracking.training.pipeline import (
    TrainingSummary,
    train_model,
    _TrainingLaunch,
    _find_free_port,
    _duplicate_stderr,
    _build_dataloaders,
    _train_single_model,
    _plan_training_tasks,
    _report_training_log,
    _run_training_worker,
    _has_fixed_dimensions,
    _is_positive_dimension,
    _route_logging_to_file,
    detect_fixed_input_size,
    _evaluate_after_training,
    _redirect_worker_console,
    _resolve_process_placement,
    _augmentation_is_fixed_size,
    _build_pose_or_detector_model,
)
from sollertia_video_tracking.training.optimization import MultiGpuStrategy, OptimizationProfile

# Shared helpers and fakes


def _profile(**overrides) -> OptimizationProfile:
    """Builds an OptimizationProfile from a CPU single-device baseline, overriding only the named fields."""
    defaults = {
        "device": "cpu",
        "gpus": (),
        "multi_gpu_strategy": MultiGpuStrategy.SINGLE,
        "amp_dtype": None,
        "use_gradient_scaler": False,
        "tf32": False,
        "cudnn_benchmark": False,
        "torch_compile": False,
        "dataloader_workers": 0,
        "pin_memory": False,
        "cpu_threads": None,
    }
    defaults.update(overrides)
    return OptimizationProfile(**defaults)


class FakeLoader:
    """A stand-in for DeepLabCut's DLCLoader exposing only the attributes the pipeline reads."""

    def __init__(self, model_cfg, pose_task, model_folder):
        self.model_cfg = model_cfg
        self.pose_task = pose_task
        self.model_folder = model_folder
        self.updates = []

    def update_model_cfg(self, updates):
        self.updates.append(updates)

    def create_dataset(self, transform, mode, task):
        return ("dataset", mode, task, transform)


class _FakeEval:
    """A stand-in evaluation summary that only needs to be describable."""

    def describe(self) -> str:
        return "eval-line"


def _make_launch(profile, **overrides) -> _TrainingLaunch:
    """Builds a _TrainingLaunch with sensible single-process defaults, overriding only the named fields."""
    defaults = {
        "config": Path("config.yaml"),
        "shuffle": 1,
        "training_set_index": 0,
        "profile": profile,
        "snapshot_path": None,
        "detector_path": None,
        "load_head_weights": True,
        "maximum_snapshots_to_keep": None,
        "progress_queue": None,
        "preserve_console": False,
        "port": 12345,
        "world_size": 1,
    }
    defaults.update(overrides)
    return _TrainingLaunch(**defaults)


# TrainingSummary.describe


def test_training_summary_describe_cuda_with_evaluation():
    """Verifies that a CUDA run reports the strategy/world-size location and appends the evaluation line."""
    summary = TrainingSummary(
        config=Path("c.yaml"),
        shuffle=3,
        model_folder=Path("/models/run"),
        tasks_trained=("detector", "pose"),
        device="cuda",
        strategy="ddp",
        world_size=2,
        precision="bfloat16",
        epochs=200,
        evaluation=_FakeEval(),
    )
    text = summary.describe()
    assert f"trained detector+pose (200 epochs) on cuda:ddpx2 in bfloat16 -> {Path('/models/run')}" in text
    assert text.endswith("\neval-line")


def test_training_summary_describe_cpu_no_tasks_no_evaluation():
    """Verifies that a CPU run with nothing trained reports 'nothing', omits the strategy suffix, and has one line."""
    summary = TrainingSummary(
        config=Path("c.yaml"),
        shuffle=1,
        model_folder=Path("/models/run"),
        tasks_trained=(),
        device="cpu",
        strategy="single",
        world_size=1,
        precision="fp32",
        epochs=0,
        evaluation=None,
    )
    text = summary.describe()
    assert text == f"trained nothing (0 epochs) on cpu in fp32 -> {Path('/models/run')}"
    assert "\n" not in text


# _is_positive_dimension / _has_fixed_dimensions / _augmentation_is_fixed_size


def test_is_positive_dimension_accepts_only_positive_ints():
    """Verifies that only positive integers are dimensions; zero, negatives, bools, floats, and strings are rejected."""
    bool_value = True  # A boolean is an int subclass but must still be rejected as a dimension.
    assert _is_positive_dimension(5) is True
    assert _is_positive_dimension(0) is False
    assert _is_positive_dimension(-3) is False
    assert _is_positive_dimension(bool_value) is False
    assert _is_positive_dimension(4.0) is False
    assert _is_positive_dimension("8") is False


def test_has_fixed_dimensions_requires_both_positive():
    """Verifies that a block has fixed dimensions only when both width and height are positive integers."""
    assert _has_fixed_dimensions({"width": 10, "height": 20}) is True
    assert _has_fixed_dimensions({"width": 0, "height": 20}) is False
    assert _has_fixed_dimensions({"width": 10}) is False


def test_augmentation_is_fixed_size_via_crop_sampling():
    """Verifies that a fixed crop-sampling block forces a single spatial size."""
    assert _augmentation_is_fixed_size({"crop_sampling": {"width": 64, "height": 64}}) is True


def test_augmentation_is_fixed_size_via_resize_without_keep_ratio():
    """Verifies that a fixed resize without aspect-ratio preservation forces a single spatial size."""
    assert _augmentation_is_fixed_size({"resize": {"width": 64, "height": 48}}) is True


def test_augmentation_is_fixed_size_resize_with_keep_ratio_is_not_fixed():
    """Verifies that a resize that preserves the aspect ratio does not force a single spatial size."""
    assert _augmentation_is_fixed_size({"resize": {"width": 64, "height": 48, "keep_ratio": True}}) is False


def test_augmentation_is_fixed_size_empty_is_not_fixed():
    """Verifies that a pipeline that neither crops nor resizes leaves the size free to vary."""
    assert _augmentation_is_fixed_size({}) is False


# detect_fixed_input_size


def _patch_loader(monkeypatch, loader):
    """Points the pipeline's DLCLoader binding at the given fake loader."""
    monkeypatch.setattr(pipeline, "DLCLoader", lambda **_kwargs: loader)


def test_detect_fixed_input_size_true_without_detector(monkeypatch):
    """Verifies that a bottom-up shuffle whose pose transform is fixed reports fixed size."""
    loader = FakeLoader(
        model_cfg={"data": {"train": {"crop_sampling": {"width": 100, "height": 100}}}},
        pose_task=Task.BOTTOM_UP,
        model_folder=Path("/m"),
    )
    _patch_loader(monkeypatch, loader)
    assert detect_fixed_input_size("cfg", shuffle=1, training_set_index=0) is True


def test_detect_fixed_input_size_false_when_detector_variable_size(monkeypatch):
    """Verifies that a trained detector with a non-fixed transform forces the whole run to report not-fixed."""
    loader = FakeLoader(
        model_cfg={
            "detector": {"train_settings": {"epochs": 2}, "data": {"train": {}}},
            "data": {"train": {"crop_sampling": {"width": 100, "height": 100}}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=Path("/m"),
    )
    _patch_loader(monkeypatch, loader)
    assert detect_fixed_input_size("cfg") is False


def test_detect_fixed_input_size_true_when_detector_and_pose_fixed(monkeypatch):
    """Verifies that a trained detector whose transform is fixed, with a fixed pose transform, reports fixed size."""
    loader = FakeLoader(
        model_cfg={
            "detector": {"train_settings": {"epochs": 2}, "data": {"train": {"resize": {"width": 50, "height": 50}}}},
            "data": {"train": {"crop_sampling": {"width": 100, "height": 100}}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=Path("/m"),
    )
    _patch_loader(monkeypatch, loader)
    assert detect_fixed_input_size("cfg") is True


def test_detect_fixed_input_size_untrained_detector_defers_to_pose(monkeypatch):
    """Verifies that a zero-epoch detector is not trained, so only the pose transform decides the result."""
    loader = FakeLoader(
        model_cfg={
            "detector": {"train_settings": {"epochs": 0}, "data": {"train": {}}},
            "data": {"train": {}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=Path("/m"),
    )
    _patch_loader(monkeypatch, loader)
    assert detect_fixed_input_size("cfg") is False


def test_detect_fixed_input_size_swallows_loader_errors(monkeypatch):
    """Verifies that any failure reading the configuration conservatively reports not-fixed."""

    def boom(**_kwargs):
        message = "cannot read config"
        raise RuntimeError(message)

    monkeypatch.setattr(pipeline, "DLCLoader", boom)
    assert detect_fixed_input_size("cfg") is False


# _plan_training_tasks


def test_plan_training_tasks_top_down_trains_detector_then_pose():
    """Verifies that a top-down shuffle with a trained detector plans the detector before the pose model."""
    loader = FakeLoader(
        model_cfg={"detector": {"train_settings": {"epochs": 3}}, "train_settings": {"epochs": 10}},
        pose_task=Task.TOP_DOWN,
        model_folder=Path("/m"),
    )
    assert _plan_training_tasks(loader) == ("detector", "pose")


def test_plan_training_tasks_top_down_untrained_detector_is_pose_only():
    """Verifies that a top-down shuffle whose detector has zero epochs plans only the pose model."""
    loader = FakeLoader(
        model_cfg={"detector": {"train_settings": {"epochs": 0}}, "train_settings": {"epochs": 10}},
        pose_task=Task.TOP_DOWN,
        model_folder=Path("/m"),
    )
    assert _plan_training_tasks(loader) == ("pose",)


def test_plan_training_tasks_bottom_up_is_pose_only():
    """Verifies that a bottom-up shuffle without a detector plans only the pose model."""
    loader = FakeLoader(
        model_cfg={"train_settings": {"epochs": 10}},
        pose_task=Task.BOTTOM_UP,
        model_folder=Path("/m"),
    )
    assert _plan_training_tasks(loader) == ("pose",)


def test_plan_training_tasks_detector_only_when_pose_epochs_zero():
    """Verifies that a top-down shuffle with a trained detector but zero pose epochs plans only the detector."""
    loader = FakeLoader(
        model_cfg={"detector": {"train_settings": {"epochs": 3}}, "train_settings": {"epochs": 0}},
        pose_task=Task.TOP_DOWN,
        model_folder=Path("/m"),
    )
    assert _plan_training_tasks(loader) == ("detector",)


# _resolve_process_placement


def test_resolve_process_placement_cpu():
    """Verifies that a CPU run uses the CPU device with no GPUs and no DDP."""
    assert _resolve_process_placement(_profile(device="cpu"), rank=0) == ("cpu", None, False, 0)


def test_resolve_process_placement_mps_detector_falls_back_to_cpu():
    """Verifies that the detector cannot train on MPS, so it falls back to the CPU."""
    assert _resolve_process_placement(_profile(device="mps"), rank=0, task=Task.DETECT) == ("cpu", None, False, 0)


def test_resolve_process_placement_mps_pose_stays_on_mps():
    """Verifies that a non-detector task keeps the MPS device."""
    assert _resolve_process_placement(_profile(device="mps"), rank=0, task=Task.BOTTOM_UP) == ("mps", None, False, 0)


def test_resolve_process_placement_cuda_ddp_uses_per_rank_gpu():
    """Verifies that under DDP the process's own GPU index drives the device, DDP flag, and local rank."""
    profile = _profile(device="cuda", gpus=(2, 3), multi_gpu_strategy=MultiGpuStrategy.DDP)
    assert _resolve_process_placement(profile, rank=1) == ("cuda", [3], True, 3)


def test_resolve_process_placement_cuda_dp_uses_all_gpus():
    """Verifies that under DataParallel one process holds every GPU and does not use DDP."""
    profile = _profile(device="cuda", gpus=(0, 1), multi_gpu_strategy=MultiGpuStrategy.DP)
    assert _resolve_process_placement(profile, rank=0) == ("cuda", [0, 1], False, 0)


def test_resolve_process_placement_cuda_single_uses_first_gpu():
    """Verifies that a single-GPU run uses the first configured GPU without DDP."""
    profile = _profile(device="cuda", gpus=(5,), multi_gpu_strategy=MultiGpuStrategy.SINGLE)
    assert _resolve_process_placement(profile, rank=0) == ("cuda", [5], False, 0)


# _build_pose_or_detector_model


class _FakeBuilder:
    """Records model-build calls and returns a sentinel instead of a real network."""

    def __init__(self):
        self.calls = []

    def build(self, model, **kwargs):
        self.calls.append((model, kwargs))
        return "built-model"


class _FakeWeightInit:
    """A stand-in for WeightInitialization exposing only from_dict."""

    @staticmethod
    def from_dict(config):
        return ("weight-init", config)


def test_build_model_detector_with_transfer_weights(monkeypatch):
    """Verifies that a weight-init config disables pretrained and routes a detector task through DETECTORS.build."""
    detectors = _FakeBuilder()
    pose = _FakeBuilder()
    monkeypatch.setattr(pipeline, "DETECTORS", detectors)
    monkeypatch.setattr(pipeline, "PoseModel", pose)
    monkeypatch.setattr(pipeline, "WeightInitialization", _FakeWeightInit)

    run_config = {"train_settings": {"weight_init": {"dataset": "superanimal"}}, "model": {"backbone": "hrnet"}}
    result = _build_pose_or_detector_model(run_config, Task.DETECT, snapshot_path=None)

    assert result == "built-model"
    assert detectors.calls == [
        ({"backbone": "hrnet"}, {"weight_init": ("weight-init", {"dataset": "superanimal"}), "pretrained": False})
    ]
    assert pose.calls == []


def test_build_model_pose_resumed_disables_pretrained_backbone(monkeypatch):
    """Verifies that resuming from a snapshot without a weight-init config disables the pretrained backbone."""
    pose = _FakeBuilder()
    monkeypatch.setattr(pipeline, "PoseModel", pose)
    monkeypatch.setattr(pipeline, "DETECTORS", _FakeBuilder())

    run_config = {"train_settings": {}, "model": {"backbone": "resnet"}}
    _build_pose_or_detector_model(run_config, Task.BOTTOM_UP, snapshot_path="snap.pt")

    assert pose.calls == [({"backbone": "resnet"}, {"weight_init": None, "pretrained_backbone": False})]


def test_build_model_pose_from_scratch_uses_pretrained_backbone(monkeypatch):
    """Verifies that a fresh pose model with neither weight-init nor a snapshot uses the pretrained backbone."""
    pose = _FakeBuilder()
    monkeypatch.setattr(pipeline, "PoseModel", pose)
    monkeypatch.setattr(pipeline, "DETECTORS", _FakeBuilder())

    run_config = {"train_settings": {}, "model": {"backbone": "resnet"}}
    _build_pose_or_detector_model(run_config, Task.BOTTOM_UP, snapshot_path=None)

    assert pose.calls == [({"backbone": "resnet"}, {"weight_init": None, "pretrained_backbone": True})]


# _build_dataloaders


def _patch_dataloader_factories(monkeypatch):
    """Replaces the DataLoader, DistributedSampler, transform, and collate builders with recorders."""
    dataloader_calls = []
    sampler_calls = []

    def fake_dataloader(**kwargs):
        dataloader_calls.append(kwargs)
        return ("dataloader", len(dataloader_calls))

    def fake_sampler(**kwargs):
        sampler_calls.append(kwargs)
        return "sampler"

    class FakeCollate:
        def __init__(self):
            self.calls = []

        def build(self, config):
            self.calls.append(config)
            return "collate-fn"

    collate = FakeCollate()
    monkeypatch.setattr(pipeline, "DataLoader", fake_dataloader)
    monkeypatch.setattr(pipeline, "DistributedSampler", fake_sampler)
    monkeypatch.setattr(pipeline, "COLLATE_FUNCTIONS", collate)
    monkeypatch.setattr(pipeline, "build_transforms", lambda config: ("transform", config))
    return dataloader_calls, sampler_calls, collate


def test_build_dataloaders_ddp_injects_distributed_sampler(monkeypatch):
    """Verifies that the DDP path builds a DistributedSampler, applies collate, and enables persistent workers."""
    dataloader_calls, sampler_calls, collate = _patch_dataloader_factories(monkeypatch)
    loader = FakeLoader(model_cfg={}, pose_task=Task.BOTTOM_UP, model_folder=Path("/m"))
    run_config = {
        "data": {"train": {"collate": {"type": "resize"}}, "inference": {}},
        "train_settings": {"batch_size": 4, "dataloader_workers": 2, "dataloader_pin_memory": True},
    }

    train_loader, valid_loader = _build_dataloaders(loader, run_config, Task.BOTTOM_UP, ddp=True, rank=1, world_size=2)

    assert sampler_calls == [
        {
            "dataset": ("dataset", "train", Task.BOTTOM_UP, ("transform", {"collate": {"type": "resize"}})),
            "num_replicas": 2,
            "rank": 1,
            "shuffle": True,
        }
    ]
    assert collate.calls == [{"type": "resize"}]
    # The training loader takes the sampler (shuffle off); the validation loader uses persistent workers when >0.
    assert dataloader_calls[0]["sampler"] == "sampler"
    assert dataloader_calls[0]["shuffle"] is False
    assert dataloader_calls[0]["collate_fn"] == "collate-fn"
    # The batch size, worker count, and pin-memory flag are threaded straight from the run configuration.
    assert dataloader_calls[0]["batch_size"] == 4
    assert dataloader_calls[0]["num_workers"] == 2
    assert dataloader_calls[0]["pin_memory"] is True
    # The validation loader always scores one frame at a time, never shuffles, and reuses the same worker count.
    assert dataloader_calls[1]["batch_size"] == 1
    assert dataloader_calls[1]["shuffle"] is False
    assert dataloader_calls[1]["num_workers"] == 2
    assert dataloader_calls[1]["pin_memory"] is True
    assert dataloader_calls[1]["persistent_workers"] is True
    assert (train_loader, valid_loader) == (("dataloader", 1), ("dataloader", 2))


def test_build_dataloaders_single_process_shuffles_without_collate(monkeypatch):
    """Verifies that without DDP the loader shuffles itself, no sampler is built, and a missing collate stays None."""
    dataloader_calls, sampler_calls, collate = _patch_dataloader_factories(monkeypatch)
    loader = FakeLoader(model_cfg={}, pose_task=Task.BOTTOM_UP, model_folder=Path("/m"))
    run_config = {
        "data": {"train": {}, "inference": {}},
        "train_settings": {"batch_size": 8, "dataloader_workers": 0, "dataloader_pin_memory": False},
    }

    _build_dataloaders(loader, run_config, Task.BOTTOM_UP, ddp=False, rank=0, world_size=1)

    assert sampler_calls == []
    assert collate.calls == []
    assert dataloader_calls[0]["shuffle"] is True
    assert dataloader_calls[0]["collate_fn"] is None
    # No DistributedSampler is passed at all on the single-process path.
    assert "sampler" not in dataloader_calls[0]
    # The batch size, worker count, and pin-memory flag are threaded straight from the run configuration.
    assert dataloader_calls[0]["batch_size"] == 8
    assert dataloader_calls[0]["num_workers"] == 0
    assert dataloader_calls[0]["pin_memory"] is False
    assert dataloader_calls[1]["batch_size"] == 1
    assert dataloader_calls[1]["persistent_workers"] is False


# _train_single_model


class _FakeModel:
    """A stand-in model recording the device it is moved to."""

    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class _FakeRunner:
    """A stand-in training runner recording its fit call."""

    def __init__(self, starting_epoch=0):
        self.starting_epoch = starting_epoch
        self.fit_kwargs = None

    def fit(self, **kwargs):
        self.fit_kwargs = kwargs


class _FakeQueueLogger:
    """A stand-in QueueTrainingLogger recording the config it is handed."""

    def __init__(self, progress_queue, task_name="pose"):
        self.progress_queue = progress_queue
        self.task_name = task_name
        self.logged_config = None

    def log_config(self, config):
        self.logged_config = config


def _patch_train_single_model_deps(monkeypatch, *, starting_epoch=0):
    """Replaces the model builder, logger, runner builder, and dataloader builder with recorders."""
    model = _FakeModel()
    runner = _FakeRunner(starting_epoch=starting_epoch)
    runner_calls = []
    logger_holder = {}

    monkeypatch.setattr(pipeline, "_build_pose_or_detector_model", lambda **_kwargs: model)

    def fake_logger(progress_queue, task_name="pose"):
        logger = _FakeQueueLogger(progress_queue, task_name=task_name)
        logger_holder["logger"] = logger
        return logger

    monkeypatch.setattr(pipeline, "QueueTrainingLogger", fake_logger)

    def fake_runner_builder(**kwargs):
        runner_calls.append(kwargs)
        return runner

    monkeypatch.setattr(pipeline, "build_optimized_training_runner", fake_runner_builder)
    monkeypatch.setattr(pipeline, "_build_dataloaders", lambda **_kwargs: ("train-loader", "valid-loader"))
    return model, runner, runner_calls, logger_holder


def test_train_single_model_rank0_builds_logger_and_fits(monkeypatch):
    """Verifies that rank 0 with a queue caps snapshots, resumes from the snapshot, logs the total budget, and fits."""
    model, runner, runner_calls, logger_holder = _patch_train_single_model_deps(monkeypatch, starting_epoch=2)
    loader = FakeLoader(model_cfg={}, pose_task=Task.BOTTOM_UP, model_folder=Path("/m"))
    run_config = {
        "runner": {"snapshots": {"max_snapshots": 1}},
        "train_settings": {"epochs": 10, "display_iters": 50},
        "resume_training_from": "resume.pt",
    }

    _train_single_model(
        loader,
        run_config,
        Task.BOTTOM_UP,
        _profile(device="cpu"),
        rank=0,
        world_size=1,
        snapshot_path=None,
        load_head_weights=True,
        maximum_snapshots_to_keep=7,
        progress_queue="queue",
    )

    assert run_config["runner"]["snapshots"]["max_snapshots"] == 7
    assert model.moved_to == "cpu"
    # snapshot_path fell back to the configured resume path.
    assert runner_calls[0]["snapshot_path"] == "resume.pt"
    assert logger_holder["logger"].task_name == "pose"
    # The logged budget is the starting epoch plus the configured epochs.
    assert logger_holder["logger"].logged_config["train_settings"]["epochs"] == 12
    assert runner.fit_kwargs == {
        "train_loader": "train-loader",
        "valid_loader": "valid-loader",
        "epochs": 10,
        "display_iters": 50,
    }


def test_train_single_model_detector_logger_uses_detector_label(monkeypatch):
    """Verifies that an explicit-snapshot detector skips the cap and resume fallback, labeling its logger 'detector'."""
    _model, _runner, runner_calls, logger_holder = _patch_train_single_model_deps(monkeypatch)
    loader = FakeLoader(model_cfg={}, pose_task=Task.TOP_DOWN, model_folder=Path("/m"))
    run_config = {
        "runner": {"snapshots": {"max_snapshots": 1}},
        "train_settings": {"epochs": 4, "display_iters": 20},
        "resume_training_from": "resume.pt",
    }

    _train_single_model(
        loader,
        run_config,
        Task.DETECT,
        _profile(device="cpu"),
        rank=0,
        world_size=1,
        snapshot_path="explicit.pt",
        load_head_weights=True,
        maximum_snapshots_to_keep=None,
        progress_queue="queue",
    )

    # max_snapshots untouched, and the explicit snapshot is not overridden by the resume fallback.
    assert run_config["runner"]["snapshots"]["max_snapshots"] == 1
    assert runner_calls[0]["snapshot_path"] == "explicit.pt"
    assert logger_holder["logger"].task_name == "detector"


def test_train_single_model_non_rank0_has_no_logger(monkeypatch):
    """Verifies that a non-rank-0 process attaches no logger and reports no epoch budget."""
    _model, runner, runner_calls, logger_holder = _patch_train_single_model_deps(monkeypatch)
    loader = FakeLoader(model_cfg={}, pose_task=Task.BOTTOM_UP, model_folder=Path("/m"))
    run_config = {"runner": {"snapshots": {"max_snapshots": 1}}, "train_settings": {"epochs": 4, "display_iters": 20}}

    _train_single_model(
        loader,
        run_config,
        Task.BOTTOM_UP,
        _profile(device="cpu"),
        rank=1,
        world_size=2,
        snapshot_path="explicit.pt",
        load_head_weights=False,
        maximum_snapshots_to_keep=None,
        progress_queue="queue",
    )

    assert "logger" not in logger_holder
    assert runner_calls[0]["logger"] is None
    assert runner.fit_kwargs["epochs"] == 4


# _run_training_worker


def _patch_worker_deps(monkeypatch, loader):
    """Replaces the worker's DLC handoffs (loader, seeding, optimizations, logging, per-model training) with stubs."""
    records = types.SimpleNamespace(seeds=[], optimizations=[], routes=[], train=[], destroy_file=[])
    monkeypatch.setattr(pipeline, "DLCLoader", lambda **_kwargs: loader)
    monkeypatch.setattr(pipeline, "fix_seeds", records.seeds.append)
    monkeypatch.setattr(pipeline, "apply_runtime_optimizations", records.optimizations.append)
    monkeypatch.setattr(
        pipeline,
        "_route_logging_to_file",
        lambda folder, *, quiet_console: records.routes.append((folder, quiet_console)),
    )
    monkeypatch.setattr(pipeline, "_train_single_model", lambda **kwargs: records.train.append(kwargs))
    monkeypatch.setattr(pipeline, "destroy_file_logging", lambda: records.destroy_file.append(True))
    return records


def test_run_training_worker_single_process_top_down(monkeypatch, tmp_path):
    """Verifies that the single-process top-down worker seeds, optimizes, routes logs, and trains detector then pose."""
    loader = FakeLoader(
        model_cfg={
            "device": "cpu",
            "train_settings": {"epochs": 5, "seed": 7, "weight_init": None},
            "detector": {"train_settings": {"epochs": 3}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=tmp_path,
    )
    records = _patch_worker_deps(monkeypatch, loader)
    launch = _make_launch(_profile(device="cpu"), progress_queue=None)

    _run_training_worker(rank=0, launch=launch)

    assert records.seeds == [7]
    assert records.optimizations == [launch.profile]
    assert records.routes == [(tmp_path, False)]
    assert records.destroy_file == [True]
    assert [call["task"] for call in records.train] == [Task.DETECT, Task.TOP_DOWN]
    # The detector config is a deep copy carrying the base device and the pose weight-init.
    detector_config = records.train[0]["run_config"]
    assert detector_config["device"] == "cpu"
    assert detector_config["train_settings"]["weight_init"] is None


def test_run_training_worker_ddp_initializes_and_tears_down_process_group(monkeypatch, tmp_path):
    """Verifies that a DDP worker inits the process group, sets its device, barriers models, and tears it down."""
    loader = FakeLoader(
        model_cfg={
            "device": "cuda",
            "train_settings": {"epochs": 5, "seed": 1, "weight_init": None},
            "detector": {"train_settings": {"epochs": 3}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=tmp_path,
    )
    records = _patch_worker_deps(monkeypatch, loader)

    init_calls, barrier_calls, destroy_group_calls, set_device_calls = [], [], [], []
    fake_dist = types.SimpleNamespace(
        init_process_group=lambda **kwargs: init_calls.append(kwargs),
        is_initialized=lambda: True,
        destroy_process_group=lambda: destroy_group_calls.append(True),
        barrier=lambda: barrier_calls.append(True),
    )
    monkeypatch.setattr(pipeline, "dist", fake_dist)
    monkeypatch.setattr(torch.cuda, "set_device", set_device_calls.append)

    profile = _profile(device="cuda", gpus=(0, 1), multi_gpu_strategy=MultiGpuStrategy.DDP)
    # progress_queue None keeps the console un-redirected so no descriptor-level redirection runs under DDP here.
    launch = _make_launch(profile, progress_queue=None, world_size=2, port=54321)

    _run_training_worker(rank=0, launch=launch)

    assert init_calls == [{"backend": "nccl", "rank": 0, "world_size": 2}]
    assert set_device_calls == [0]
    assert barrier_calls == [True]
    assert destroy_group_calls == [True]
    assert os.environ["MASTER_PORT"] == "54321"
    assert [call["task"] for call in records.train] == [Task.DETECT, Task.TOP_DOWN]


# train_model


class _FakeMonitor:
    """A stand-in TrainingMonitor recording its lifecycle."""

    instances: ClassVar[list] = []

    def __init__(self, progress_queue, stream=None):
        self.progress_queue = progress_queue
        self.stream = stream
        self.started = False
        self.stopped = False
        self.joined = None
        self.alive = False
        _FakeMonitor.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        self.joined = timeout

    def is_alive(self):
        return self.alive


class _FakeManager:
    """A stand-in multiprocessing manager returning a placeholder queue."""

    def __init__(self):
        self.shutdown_called = False

    def Queue(self):  # noqa: N802 - mirrors multiprocessing.Manager's Queue factory name.
        return "progress-queue"

    def shutdown(self):
        self.shutdown_called = True


class _FakeStream:
    """A stand-in preserved-stderr duplicate recording that it was closed."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _patch_train_model_deps(monkeypatch, loader, *, dup_stream, worker=None, evaluation=None):
    """Replaces train_model's process management, monitor, stderr duplicate, worker, and evaluation with recorders."""
    _FakeMonitor.instances = []
    records = types.SimpleNamespace(spawn=[], worker=[], evaluate=[], manager=_FakeManager())
    monkeypatch.setattr(pipeline, "DLCLoader", lambda **_kwargs: loader)
    monkeypatch.setattr(pipeline, "_find_free_port", lambda: 45000)
    monkeypatch.setattr(pipeline, "_duplicate_stderr", lambda: dup_stream)
    monkeypatch.setattr(pipeline, "TrainingMonitor", _FakeMonitor)

    fake_mp = types.SimpleNamespace(
        Manager=lambda: records.manager,
        spawn=lambda fn, args, nprocs, join: records.spawn.append((fn, args, nprocs, join)),
    )
    monkeypatch.setattr(pipeline, "mp", fake_mp)

    def default_worker(rank, launch):
        records.worker.append((rank, launch))

    monkeypatch.setattr(pipeline, "_run_training_worker", worker or default_worker)

    def fake_evaluate(**kwargs):
        records.evaluate.append(kwargs)
        return evaluation

    monkeypatch.setattr(pipeline, "_evaluate_after_training", fake_evaluate)
    return records


def test_train_model_single_process_with_monitor_and_evaluation(monkeypatch, tmp_path):
    """Verifies that a single-process run starts and stops the monitor, runs the rank-0 worker, and evaluates pose."""
    loader = FakeLoader(
        model_cfg={"train_settings": {"epochs": 10}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    dup_stream = _FakeStream()
    evaluation = _FakeEval()
    records = _patch_train_model_deps(monkeypatch, loader, dup_stream=dup_stream, evaluation=evaluation)

    summary = train_model(str(tmp_path / "config.yaml"), _profile(device="cpu"), shuffle=2, epochs=None)

    # The worker ran once in-process; DDP spawn was not used.
    assert len(records.worker) == 1
    assert records.worker[0][0] == 0
    assert records.spawn == []
    # The monitor started and was cleanly stopped, the manager shut down, and the preserved stream was closed.
    monitor = _FakeMonitor.instances[0]
    assert monitor.started
    assert monitor.stopped
    assert records.manager.shutdown_called
    assert dup_stream.closed
    # The pose model was evaluated and folded into the summary.
    assert len(records.evaluate) == 1
    assert summary.evaluation is evaluation
    assert summary.tasks_trained == ("pose",)
    assert summary.device == "cpu"
    assert summary.precision == "fp32"
    assert summary.epochs == 10
    assert summary.world_size == 1
    # The launch carried a preserved console because a stderr duplicate was held.
    assert records.worker[0][1].preserve_console is True
    assert records.worker[0][1].port == 45000


def test_train_model_retains_monitor_resources_when_renderer_outlives_join(monkeypatch, tmp_path):
    """Verifies that a renderer still running after the join keeps its queue manager and preserved stream open."""
    loader = FakeLoader(
        model_cfg={"train_settings": {"epochs": 10}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    dup_stream = _FakeStream()

    def mark_monitor_alive(**_kwargs):
        _FakeMonitor.instances[0].alive = True

    records = _patch_train_model_deps(monkeypatch, loader, dup_stream=dup_stream, worker=mark_monitor_alive)

    train_model(str(tmp_path / "config.yaml"), _profile(device="cpu"), shuffle=2, epochs=None)

    monitor = _FakeMonitor.instances[0]
    assert monitor.stopped
    assert not records.manager.shutdown_called
    assert not dup_stream.closed


def test_train_model_without_progress_skips_monitor(monkeypatch, tmp_path):
    """Verifies that disabling the progress display skips the monitor, manager, and stderr duplicate entirely."""
    loader = FakeLoader(
        model_cfg={"train_settings": {"epochs": 3}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    records = _patch_train_model_deps(monkeypatch, loader, dup_stream=None)

    summary = train_model(tmp_path / "config.yaml", _profile(device="cpu"), display_progress=False, evaluate=False)

    assert _FakeMonitor.instances == []
    assert records.manager.shutdown_called is False
    assert records.evaluate == []
    assert records.worker[0][1].progress_queue is None
    assert records.worker[0][1].preserve_console is False
    assert summary.evaluation is None


def test_train_model_ddp_spawns_workers(monkeypatch, tmp_path):
    """Verifies that a DDP profile spawns one worker per GPU and reports mixed-precision and strategy in the summary."""
    loader = FakeLoader(
        model_cfg={"train_settings": {"epochs": 100}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    dup_stream = _FakeStream()
    evaluation = _FakeEval()
    profile = _profile(device="cuda", gpus=(0, 1), multi_gpu_strategy=MultiGpuStrategy.DDP, amp_dtype=torch.bfloat16)
    records = _patch_train_model_deps(monkeypatch, loader, dup_stream=dup_stream, evaluation=evaluation)

    summary = train_model(tmp_path / "config.yaml", profile)

    assert len(records.spawn) == 1
    spawn_fn, spawn_args, nprocs, join = records.spawn[0]
    assert spawn_fn is pipeline._run_training_worker
    assert nprocs == 2
    assert join is True
    # The single positional argument bundled into the spawn call is the launch.
    assert isinstance(spawn_args[0], _TrainingLaunch)
    assert records.worker == []
    assert summary.strategy == "ddp"
    assert summary.world_size == 2
    assert summary.device == "cuda"
    assert summary.precision == "bfloat16"


def test_train_model_applies_detector_and_all_overrides(monkeypatch, tmp_path):
    """Verifies that every override is threaded into the config, including the detector-specific keys for top-down."""
    loader = FakeLoader(
        model_cfg={
            "train_settings": {"epochs": 5},
            "detector": {"train_settings": {"epochs": 3}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=tmp_path,
    )
    _patch_train_model_deps(monkeypatch, loader, dup_stream=None)

    summary = train_model(
        tmp_path / "config.yaml",
        _profile(device="cpu", dataloader_workers=2, pin_memory=False),
        batch_size=16,
        epochs=20,
        save_epochs=5,
        display_iterations=100,
        detector_batch_size=8,
        detector_epochs=7,
        detector_save_epochs=2,
        display_progress=False,
        evaluate=False,
    )

    updates = loader.updates[0]
    # The base device and dataloader settings from the profile are always threaded into the update, unconditionally.
    assert updates["device"] == "cpu"
    assert updates["train_settings.dataloader_workers"] == 2
    assert updates["train_settings.dataloader_pin_memory"] is False
    assert updates["train_settings.batch_size"] == 16
    assert updates["train_settings.epochs"] == 20
    assert updates["runner.snapshots.save_epochs"] == 5
    assert updates["train_settings.display_iters"] == 100
    # A top-down run mirrors the device and dataloader settings onto the detector, plus its own overrides.
    assert updates["detector.device"] == "cpu"
    assert updates["detector.train_settings.dataloader_workers"] == 2
    assert updates["detector.train_settings.dataloader_pin_memory"] is False
    assert updates["detector.train_settings.batch_size"] == 8
    assert updates["detector.train_settings.epochs"] == 7
    assert updates["detector.runner.snapshots.save_epochs"] == 2
    assert updates["detector.train_settings.display_iters"] == 100
    assert summary.tasks_trained == ("detector", "pose")


def test_train_model_skips_evaluation_without_pose(monkeypatch, tmp_path):
    """Verifies that a detector-only run skips the post-training pose evaluation even when evaluation is requested."""
    loader = FakeLoader(
        model_cfg={
            "train_settings": {"epochs": 0},
            "detector": {"train_settings": {"epochs": 3}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=tmp_path,
    )
    records = _patch_train_model_deps(monkeypatch, loader, dup_stream=None)

    summary = train_model(tmp_path / "config.yaml", _profile(device="cpu"), display_progress=False, evaluate=True)

    assert records.evaluate == []
    assert summary.tasks_trained == ("detector",)
    assert summary.evaluation is None


def test_train_model_rejects_memory_replay(monkeypatch, tmp_path):
    """Verifies that a SuperAnimal memory-replay shuffle is rejected before any training begins."""
    loader = FakeLoader(
        model_cfg={"train_settings": {"epochs": 10, "weight_init": {"memory_replay": True}}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    monkeypatch.setattr(pipeline, "DLCLoader", lambda **_kwargs: loader)
    monkeypatch.setattr(
        pipeline,
        "WeightInitialization",
        types.SimpleNamespace(from_dict=lambda *_: types.SimpleNamespace(memory_replay=True)),
    )

    with pytest.raises(ValueError, match="memory-replay"):
        train_model(tmp_path / "config.yaml", _profile(device="cpu"))


def test_train_model_reports_log_on_failure(monkeypatch, tmp_path):
    """Verifies that when the worker fails, the monitor is still cleaned up and the operator is pointed to the log."""
    loader = FakeLoader(
        model_cfg={"train_settings": {"epochs": 10}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    dup_stream = _FakeStream()

    def failing_worker(*_args, **_kwargs):
        message = "worker crashed"
        raise RuntimeError(message)

    records = _patch_train_model_deps(monkeypatch, loader, dup_stream=dup_stream, worker=failing_worker)
    reported = []
    monkeypatch.setattr(pipeline, "_report_training_log", reported.append)
    # The training log must already exist for the failure notice to be emitted.
    (tmp_path / "train.txt").write_text("worker output")

    with pytest.raises(RuntimeError, match="worker crashed"):
        train_model(tmp_path / "config.yaml", _profile(device="cpu"))

    assert reported == [tmp_path / "train.txt"]
    # The finally block still tore down the monitor and manager despite the failure.
    assert _FakeMonitor.instances[0].stopped
    assert records.manager.shutdown_called
    assert dup_stream.closed


# _evaluate_after_training


def test_evaluate_after_training_cuda_success(monkeypatch):
    """Verifies that on CUDA the evaluation targets the first GPU and empties the cache first."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    captured = {}

    def fake_evaluate(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return "evaluation-summary"

    monkeypatch.setattr(pipeline, "evaluate_trained_model", fake_evaluate)

    result = _evaluate_after_training(
        config=Path("config.yaml"),
        profile=_profile(device="cuda", gpus=(3,)),
        shuffle=2,
        training_set_index=1,
        batch_size=4,
        confidence_cutoff=0.5,
    )

    assert result == "evaluation-summary"
    assert emptied == [True]
    assert captured["device"] == "cuda:3"
    assert captured["shuffle"] == 2
    assert captured["confidence_cutoff"] == 0.5


def test_evaluate_after_training_cpu_skips_cache_empty(monkeypatch):
    """Verifies that on a CPU run the device is passed through and the CUDA cache is not touched."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    captured = {}

    def fake_evaluate(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return "cpu-summary"

    monkeypatch.setattr(pipeline, "evaluate_trained_model", fake_evaluate)

    result = _evaluate_after_training(
        config=Path("config.yaml"),
        profile=_profile(device="cpu"),
        shuffle=1,
        training_set_index=0,
        batch_size=1,
        confidence_cutoff=None,
    )

    assert result == "cpu-summary"
    assert captured["device"] == "cpu"


def test_evaluate_after_training_swallows_failure(monkeypatch, caplog):
    """Verifies that a failing evaluation is logged and swallowed so the completed training run is not lost."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def boom(*_args, **_kwargs):
        message = "evaluation blew up"
        raise RuntimeError(message)

    monkeypatch.setattr(pipeline, "evaluate_trained_model", boom)

    with caplog.at_level(logging.WARNING):
        result = _evaluate_after_training(
            config=Path("config.yaml"),
            profile=_profile(device="cpu"),
            shuffle=1,
            training_set_index=0,
            batch_size=1,
            confidence_cutoff=None,
        )

    assert result is None
    assert any("Post-training evaluation failed" in record.message for record in caplog.records)


# _find_free_port / _route_logging_to_file / _duplicate_stderr / _redirect_worker_console / _report_training_log


def test_find_free_port_returns_positive_port():
    """Verifies that a free port is reserved, released, and returned as a genuinely bindable positive integer."""
    port = _find_free_port()
    assert isinstance(port, int)
    assert 0 < port <= 65535
    # The probe socket must have been closed before returning, so the port is free to bind again here. If the
    # function held the socket open, this bind would raise OSError: Address already in use.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rebind:
        rebind.bind(("", port))


def test_route_logging_to_file_quiet_removes_stream_handler(monkeypatch, tmp_path):
    """Verifies that quieting the console detaches the plain stream handler while retaining the file handler."""
    recorded = []
    monkeypatch.setattr(pipeline, "setup_file_logging", recorded.append)
    root = logging.getLogger()
    saved = root.handlers[:]
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(tmp_path / "existing.log")
    try:
        root.handlers = [stream_handler, file_handler]
        _route_logging_to_file(tmp_path, quiet_console=True)
        assert stream_handler not in root.handlers
        assert file_handler in root.handlers
    finally:
        root.handlers = saved
        file_handler.close()
    assert recorded == [tmp_path / "train.txt"]


def test_route_logging_to_file_keeps_console_when_not_quiet(monkeypatch, tmp_path):
    """Verifies that without quieting, the stream handler is left in place after the log file is configured."""
    recorded = []
    monkeypatch.setattr(pipeline, "setup_file_logging", recorded.append)
    root = logging.getLogger()
    saved = root.handlers[:]
    stream_handler = logging.StreamHandler()
    try:
        root.handlers = [stream_handler]
        _route_logging_to_file(tmp_path, quiet_console=False)
        assert stream_handler in root.handlers
    finally:
        root.handlers = saved
    assert recorded == [tmp_path / "train.txt"]


def test_duplicate_stderr_returns_stream(monkeypatch, tmp_path):
    """Verifies that a distinct writable duplicate of stderr's descriptor is returned, and writes reach its file."""
    backing_path = tmp_path / "stderr.txt"
    backing = backing_path.open("w")
    monkeypatch.setattr(pipeline.sys, "stderr", backing)
    try:
        stream = _duplicate_stderr()
        assert stream is not None
        # The duplicate is a distinct descriptor, not a handle onto stderr's own file object.
        assert stream.fileno() != backing.fileno()
        # Because os.dup shares the underlying open file, output written through the duplicate lands in the same
        # file the original stderr points at.
        stream.write("duplicated-write\n")
        stream.close()
    finally:
        backing.close()
    assert "duplicated-write" in backing_path.read_text()


def test_duplicate_stderr_returns_none_without_descriptor(monkeypatch):
    """Verifies that when stderr has no usable descriptor, None is returned instead of raising."""

    class NoDescriptor:
        def fileno(self):
            message = "no descriptor under capture"
            raise ValueError(message)

    monkeypatch.setattr(pipeline.sys, "stderr", NoDescriptor())
    assert _duplicate_stderr() is None


def test_redirect_worker_console_inactive_is_noop(tmp_path):
    """Verifies that when inactive, the context manager runs its body without touching descriptors."""
    ran = False
    with _redirect_worker_console(tmp_path / "unused.log", active=False):
        ran = True
    assert ran


def test_redirect_worker_console_active_captures_descriptor_output(tmp_path):
    """Verifies that when active, descriptor-level stdout and stderr are appended to the log and restored on exit."""
    log_path = tmp_path / "worker.log"
    with _redirect_worker_console(log_path, active=True):
        os.write(1, b"redirected-stdout\n")
        os.write(2, b"redirected-stderr\n")
    content = log_path.read_text()
    assert "redirected-stdout" in content
    assert "redirected-stderr" in content


def test_report_training_log_writes_notice(capsys):
    """Verifies that the failure notice names the training log and explains what it captured."""
    _report_training_log(Path("/models/run/train.txt"))
    err = capsys.readouterr().err
    assert "Training did not complete" in err
    assert str(Path("/models/run/train.txt")) in err
