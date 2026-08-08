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
import signal
import socket
from typing import ClassVar
import logging
from pathlib import Path
import contextlib

import torch
import pytest
from torch.multiprocessing import ProcessExitedException, ProcessRaisedException
from deeplabcut.pose_estimation_pytorch.task import Task

from sollertia_video_tracking.training import pipeline
from sollertia_video_tracking.training.pipeline import (
    TrainingSummary,
    TrainingFailedError,
    TrainingInterruptedError,
    train_model,
    _TrainingLaunch,
    _find_free_port,
    _build_dataloaders,
    _train_single_model,
    _append_training_log,
    _plan_training_tasks,
    _run_training_worker,
    _has_fixed_dimensions,
    _is_operator_interrupt,
    _is_positive_dimension,
    _route_logging_to_file,
    _stop_progress_monitor,
    detect_fixed_input_size,
    _describe_worker_failure,
    _evaluate_after_training,
    _format_training_failure,
    _redirect_worker_console,
    _resolve_process_placement,
    _augmentation_is_fixed_size,
    _build_pose_or_detector_model,
)
from sollertia_video_tracking.training.optimization import MultiGpuStrategy, OptimizationProfile

# Shared helpers and fakes


class _FakeLoader:
    """Stands in for DeepLabCut's DLCLoader, exposing only the attributes the pipeline reads."""

    def __init__(self, model_cfg, pose_task, model_folder, image_sizes=((512, 512),)):
        self.model_cfg = model_cfg
        self.pose_task = pose_task
        self.model_folder = model_folder
        self.updates = []
        self.image_sizes = image_sizes

    def update_model_cfg(self, updates):
        self.updates.append(updates)

    def create_dataset(self, transform, mode, task):
        return ("dataset", mode, task, transform)

    def load_data(self, _split):
        return {"images": [{"height": height, "width": width} for height, width in self.image_sizes]}


def _exited(signal_name, exit_code):
    """Builds the exception torch raises when a spawned worker dies by signal or by a non-zero exit code."""
    return ProcessExitedException(
        f"process 0 terminated with {signal_name or exit_code}",
        error_index=0,
        error_pid=4242,
        exit_code=exit_code,
        signal_name=signal_name,
    )


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
    """Verifies that only positive integers are dimensions, rejecting zero, negatives, booleans, floats, and strings."""
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


def test_detect_fixed_input_size_true_without_detector(monkeypatch):
    """Verifies that a bottom-up shuffle whose pose transform is fixed reports fixed size."""
    loader = _FakeLoader(
        model_cfg={"data": {"train": {"crop_sampling": {"width": 100, "height": 100}}}},
        pose_task=Task.BOTTOM_UP,
        model_folder=Path("/m"),
    )
    _patch_loader(monkeypatch=monkeypatch, loader=loader)
    assert detect_fixed_input_size("cfg", shuffle=1, training_set_index=0) is True


def test_detect_fixed_input_size_false_when_detector_variable_size(monkeypatch):
    """Verifies that a trained detector with a non-fixed transform forces the whole run to report not-fixed."""
    loader = _FakeLoader(
        model_cfg={
            "detector": {"train_settings": {"epochs": 2}, "data": {"train": {}}},
            "data": {"train": {"crop_sampling": {"width": 100, "height": 100}}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=Path("/m"),
    )
    _patch_loader(monkeypatch=monkeypatch, loader=loader)
    assert detect_fixed_input_size("cfg") is False


def test_detect_fixed_input_size_true_when_detector_and_pose_fixed(monkeypatch):
    """Verifies that a trained detector whose transform is fixed, with a fixed pose transform, reports fixed size."""
    loader = _FakeLoader(
        model_cfg={
            "detector": {"train_settings": {"epochs": 2}, "data": {"train": {"resize": {"width": 50, "height": 50}}}},
            "data": {"train": {"crop_sampling": {"width": 100, "height": 100}}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=Path("/m"),
    )
    _patch_loader(monkeypatch=monkeypatch, loader=loader)
    assert detect_fixed_input_size("cfg") is True


def test_detect_fixed_input_size_untrained_detector_defers_to_pose(monkeypatch):
    """Verifies that a zero-epoch detector is not trained, so only the pose transform decides the result."""
    loader = _FakeLoader(
        model_cfg={
            "detector": {"train_settings": {"epochs": 0}, "data": {"train": {}}},
            "data": {"train": {}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=Path("/m"),
    )
    _patch_loader(monkeypatch=monkeypatch, loader=loader)
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
    loader = _FakeLoader(
        model_cfg={"detector": {"train_settings": {"epochs": 3}}, "train_settings": {"epochs": 10}},
        pose_task=Task.TOP_DOWN,
        model_folder=Path("/m"),
    )
    assert _plan_training_tasks(loader) == ("detector", "pose")


def test_plan_training_tasks_top_down_untrained_detector_is_pose_only():
    """Verifies that a top-down shuffle whose detector has zero epochs plans only the pose model."""
    loader = _FakeLoader(
        model_cfg={"detector": {"train_settings": {"epochs": 0}}, "train_settings": {"epochs": 10}},
        pose_task=Task.TOP_DOWN,
        model_folder=Path("/m"),
    )
    assert _plan_training_tasks(loader) == ("pose",)


def test_plan_training_tasks_bottom_up_is_pose_only():
    """Verifies that a bottom-up shuffle without a detector plans only the pose model."""
    loader = _FakeLoader(
        model_cfg={"train_settings": {"epochs": 10}},
        pose_task=Task.BOTTOM_UP,
        model_folder=Path("/m"),
    )
    assert _plan_training_tasks(loader) == ("pose",)


def test_plan_training_tasks_detector_only_when_pose_epochs_zero():
    """Verifies that a top-down shuffle with a trained detector but zero pose epochs plans only the detector."""
    loader = _FakeLoader(
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


def test_build_model_detector_with_transfer_weights(monkeypatch):
    """Verifies that a weight-init config disables pretrained and routes a detector task through DETECTORS.build."""
    detectors = _FakeBuilder()
    pose = _FakeBuilder()
    monkeypatch.setattr(pipeline, "DETECTORS", detectors)
    monkeypatch.setattr(pipeline, "PoseModel", pose)
    monkeypatch.setattr(pipeline, "WeightInitialization", _FakeWeightInit)

    run_config = {"train_settings": {"weight_init": {"dataset": "superanimal"}}, "model": {"backbone": "hrnet"}}
    result = _build_pose_or_detector_model(run_config=run_config, task=Task.DETECT, snapshot_path=None)

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
    _build_pose_or_detector_model(run_config=run_config, task=Task.BOTTOM_UP, snapshot_path="snap.pt")

    assert pose.calls == [({"backbone": "resnet"}, {"weight_init": None, "pretrained_backbone": False})]


def test_build_model_pose_from_scratch_uses_pretrained_backbone(monkeypatch):
    """Verifies that a fresh pose model with neither weight-init nor a snapshot uses the pretrained backbone."""
    pose = _FakeBuilder()
    monkeypatch.setattr(pipeline, "PoseModel", pose)
    monkeypatch.setattr(pipeline, "DETECTORS", _FakeBuilder())

    run_config = {"train_settings": {}, "model": {"backbone": "resnet"}}
    _build_pose_or_detector_model(run_config=run_config, task=Task.BOTTOM_UP, snapshot_path=None)

    assert pose.calls == [({"backbone": "resnet"}, {"weight_init": None, "pretrained_backbone": True})]


# _build_dataloaders


def test_build_dataloaders_ddp_injects_distributed_sampler(monkeypatch):
    """Verifies that the DDP path builds a DistributedSampler, applies collate, and enables persistent workers."""
    dataloader_calls, sampler_calls, collate = _patch_dataloader_factories(monkeypatch)
    loader = _FakeLoader(model_cfg={}, pose_task=Task.BOTTOM_UP, model_folder=Path("/m"))
    run_config = {
        "data": {"train": {"collate": {"type": "resize"}}, "inference": {}},
        "train_settings": {"batch_size": 4, "dataloader_workers": 2, "dataloader_pin_memory": True},
    }

    train_loader, valid_loader = _build_dataloaders(
        loader=loader,
        run_config=run_config,
        task=Task.BOTTOM_UP,
        ddp=True,
        rank=1,
        world_size=2,
    )

    assert sampler_calls == [
        {
            "dataset": ("dataset", "train", Task.BOTTOM_UP, ("transform", {"collate": {"type": "resize"}})),
            "num_replicas": 2,
            "rank": 1,
            "shuffle": True,
        }
    ]
    assert collate.calls == [{"type": "resize"}]
    # The training loader takes the sampler (shuffle off) and uses persistent workers when the worker count is nonzero.
    assert dataloader_calls[0]["sampler"] == "sampler"
    assert dataloader_calls[0]["shuffle"] is False
    assert dataloader_calls[0]["collate_fn"] == "collate-fn"
    # The batch size, worker count, and pin-memory flag are threaded straight from the run configuration.
    assert dataloader_calls[0]["batch_size"] == 4
    assert dataloader_calls[0]["num_workers"] == 2
    assert dataloader_calls[0]["pin_memory"] is True
    assert dataloader_calls[0]["persistent_workers"] is True
    # Uniformly sized labeled frames let validation batch, capped at the training batch size. It never shuffles and
    # loads in the training process.
    assert dataloader_calls[1]["batch_size"] == 4
    assert dataloader_calls[1]["shuffle"] is False
    assert "num_workers" not in dataloader_calls[1]
    assert "persistent_workers" not in dataloader_calls[1]
    assert dataloader_calls[1]["pin_memory"] is True
    assert (train_loader, valid_loader) == (("dataloader", 1), ("dataloader", 2))


def test_build_dataloaders_drops_validation_to_one_frame_on_mixed_resolutions(monkeypatch):
    """Verifies that labeled frames spanning several resolutions force validation back to one frame per batch."""
    dataloader_calls, _sampler_calls, _collate = _patch_dataloader_factories(monkeypatch)
    loader = _FakeLoader(
        model_cfg={},
        pose_task=Task.BOTTOM_UP,
        model_folder=Path("/m"),
        image_sizes=((512, 512), (640, 480)),
    )
    run_config = {
        "data": {"train": {}, "inference": {}},
        "train_settings": {"batch_size": 8, "dataloader_workers": 0, "dataloader_pin_memory": False},
    }

    _build_dataloaders(loader=loader, run_config=run_config, task=Task.BOTTOM_UP, ddp=False, rank=0, world_size=1)

    # Default collation cannot stack frames of differing size, so the training batch size is not carried over.
    assert dataloader_calls[0]["batch_size"] == 8
    assert dataloader_calls[1]["batch_size"] == 1


def test_build_dataloaders_single_process_shuffles_without_collate(monkeypatch):
    """Verifies that without DDP the loader shuffles itself, no sampler is built, and a missing collate stays None."""
    dataloader_calls, sampler_calls, collate = _patch_dataloader_factories(monkeypatch)
    loader = _FakeLoader(model_cfg={}, pose_task=Task.BOTTOM_UP, model_folder=Path("/m"))
    run_config = {
        "data": {"train": {}, "inference": {}},
        "train_settings": {"batch_size": 8, "dataloader_workers": 0, "dataloader_pin_memory": False},
    }

    _build_dataloaders(loader=loader, run_config=run_config, task=Task.BOTTOM_UP, ddp=False, rank=0, world_size=1)

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
    assert dataloader_calls[1]["batch_size"] == 8
    assert "num_workers" not in dataloader_calls[1]


# _train_single_model


def test_train_single_model_rank0_builds_logger_and_fits(monkeypatch):
    """Verifies that rank 0 with a queue caps snapshots, resumes from the snapshot, logs the total budget, and fits."""
    model, runner, runner_calls, logger_holder = _patch_train_single_model_deps(monkeypatch, starting_epoch=2)
    loader = _FakeLoader(model_cfg={}, pose_task=Task.BOTTOM_UP, model_folder=Path("/m"))
    run_config = {
        "runner": {"snapshots": {"max_snapshots": 1}},
        "train_settings": {"epochs": 10, "display_iters": 50},
        "resume_training_from": "resume.pt",
    }

    _train_single_model(
        loader=loader,
        run_config=run_config,
        task=Task.BOTTOM_UP,
        profile=_profile(device="cpu"),
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
    loader = _FakeLoader(model_cfg={}, pose_task=Task.TOP_DOWN, model_folder=Path("/m"))
    run_config = {
        "runner": {"snapshots": {"max_snapshots": 1}},
        "train_settings": {"epochs": 4, "display_iters": 20},
        "resume_training_from": "resume.pt",
    }

    _train_single_model(
        loader=loader,
        run_config=run_config,
        task=Task.DETECT,
        profile=_profile(device="cpu"),
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
    loader = _FakeLoader(model_cfg={}, pose_task=Task.BOTTOM_UP, model_folder=Path("/m"))
    run_config = {"runner": {"snapshots": {"max_snapshots": 1}}, "train_settings": {"epochs": 4, "display_iters": 20}}

    _train_single_model(
        loader=loader,
        run_config=run_config,
        task=Task.BOTTOM_UP,
        profile=_profile(device="cpu"),
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


def test_run_training_worker_single_process_top_down(monkeypatch, tmp_path):
    """Verifies that the single-process top-down worker seeds, optimizes, routes logs, and trains detector then pose."""
    loader = _FakeLoader(
        model_cfg={
            "device": "cpu",
            "train_settings": {"epochs": 5, "seed": 7, "weight_init": None},
            "detector": {"train_settings": {"epochs": 3}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=tmp_path,
    )
    records = _patch_worker_deps(monkeypatch=monkeypatch, loader=loader)
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
    loader = _FakeLoader(
        model_cfg={
            "device": "cuda",
            "train_settings": {"epochs": 5, "seed": 1, "weight_init": None},
            "detector": {"train_settings": {"epochs": 3}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=tmp_path,
    )
    records = _patch_worker_deps(monkeypatch=monkeypatch, loader=loader)

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


@pytest.mark.parametrize(
    ("progress_queue", "expected_active"),
    [("queue", True), (None, False)],
)
def test_run_training_worker_redirects_its_console_whenever_the_monitor_owns_the_terminal(
    monkeypatch, tmp_path, progress_queue, expected_active
):
    """Verifies that a worker redirects its own console exactly when progress reporting is on."""
    loader = _FakeLoader(
        model_cfg={"device": "cpu", "train_settings": {"epochs": 5, "seed": 7, "weight_init": None}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    _patch_worker_deps(monkeypatch=monkeypatch, loader=loader)
    redirects = []
    monkeypatch.setattr(
        pipeline,
        "_redirect_worker_console",
        lambda log_path, *, active: redirects.append((log_path, active)) or contextlib.nullcontext(),
    )

    _run_training_worker(rank=0, launch=_make_launch(_profile(device="cpu"), progress_queue=progress_queue))

    assert redirects == [(tmp_path / "train.txt", expected_active)]


def test_run_training_worker_reports_a_log_teardown_error_without_masking_the_run(monkeypatch, tmp_path):
    """Verifies that a log-handler teardown error is warned about rather than replacing the worker's own outcome."""
    loader = _FakeLoader(
        model_cfg={"device": "cpu", "train_settings": {"epochs": 5, "seed": 7, "weight_init": None}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    records = _patch_worker_deps(monkeypatch=monkeypatch, loader=loader)

    def failing_destroy():
        message = "handlers already gone"
        raise RuntimeError(message)

    monkeypatch.setattr(pipeline, "destroy_file_logging", failing_destroy)
    warnings = []
    monkeypatch.setattr(pipeline, "warn", warnings.append)

    _run_training_worker(rank=0, launch=_make_launch(_profile(device="cpu"), progress_queue=None))

    assert any("handlers already gone" in warning for warning in warnings)
    assert len(records.train) == 1


def test_run_training_worker_reports_a_process_group_teardown_error(monkeypatch, tmp_path):
    """Verifies that a distributed teardown error is warned about rather than replacing the worker's own outcome."""
    loader = _FakeLoader(
        model_cfg={"device": "cuda", "train_settings": {"epochs": 5, "seed": 1, "weight_init": None}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    _patch_worker_deps(monkeypatch=monkeypatch, loader=loader)

    def failing_destroy_group():
        message = "NCCL communicator is already gone"
        raise RuntimeError(message)

    monkeypatch.setattr(
        pipeline,
        "dist",
        types.SimpleNamespace(
            init_process_group=lambda **_kwargs: None,
            is_initialized=lambda: True,
            destroy_process_group=failing_destroy_group,
            barrier=lambda: None,
        ),
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda _index: None)
    warnings = []
    monkeypatch.setattr(pipeline, "warn", warnings.append)
    profile = _profile(device="cuda", gpus=(0, 1), multi_gpu_strategy=MultiGpuStrategy.DDP)

    _run_training_worker(rank=0, launch=_make_launch(profile, progress_queue=None, world_size=2))

    assert any("NCCL communicator is already gone" in warning for warning in warnings)


def test_run_training_worker_enables_native_crash_dumps_before_it_redirects(monkeypatch, tmp_path):
    """Verifies that the fault handler is installed first, so a native crash dumps into the training log."""
    loader = _FakeLoader(
        model_cfg={"device": "cpu", "train_settings": {"epochs": 5, "seed": 7, "weight_init": None}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    _patch_worker_deps(monkeypatch=monkeypatch, loader=loader)
    order = []
    monkeypatch.setattr(pipeline, "enable_native_crash_dumps", lambda: order.append("dumps"))
    monkeypatch.setattr(
        pipeline,
        "_redirect_worker_console",
        lambda log_path, *, active: order.append("redirect") or contextlib.nullcontext(),  # noqa: ARG005
    )

    _run_training_worker(rank=0, launch=_make_launch(_profile(device="cpu"), progress_queue="queue"))

    assert order == ["dumps", "redirect"]


# train_model


def test_train_model_single_process_with_monitor_and_evaluation(monkeypatch, tmp_path):
    """Verifies that a single-process run spawns one worker, starts and stops the monitor, and evaluates pose."""
    loader = _FakeLoader(
        model_cfg={"train_settings": {"epochs": 10}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    evaluation = _FakeEval()
    records = _patch_train_model_deps(monkeypatch=monkeypatch, loader=loader, evaluation=evaluation)

    summary = train_model(config=str(tmp_path / "config.yaml"), profile=_profile(device="cpu"), shuffle=2, epochs=None)

    # A single-process run still goes through spawn, so the reporting process never becomes the redirected one.
    assert len(records.spawn) == 1
    assert records.spawn[0][2] == 1
    assert len(records.worker) == 1
    assert records.worker[0][0] == 0
    # The monitor started and was cleanly stopped, and the manager shut down.
    monitor = _FakeMonitor.instances[0]
    assert monitor.started
    assert monitor.stopped
    assert records.manager.shutdown_called
    # The pose model was evaluated and folded into the summary.
    assert len(records.evaluate) == 1
    assert summary.evaluation is evaluation
    assert summary.evaluation_error is None
    assert summary.tasks_trained == ("pose",)
    assert summary.device == "cpu"
    assert summary.precision == "fp32"
    assert summary.epochs == 10
    assert summary.world_size == 1
    assert records.worker[0][1].port == 45000


def test_train_model_retains_monitor_resources_when_renderer_outlives_join(monkeypatch, tmp_path):
    """Verifies that a renderer still running after the join keeps its queue manager open and closes the bar line."""
    loader = _FakeLoader(
        model_cfg={"train_settings": {"epochs": 10}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )

    def mark_monitor_alive(*_args, **_kwargs):
        _FakeMonitor.instances[0].alive = True

    records = _patch_train_model_deps(monkeypatch=monkeypatch, loader=loader, worker=mark_monitor_alive)
    written = []
    monkeypatch.setattr(pipeline.sys, "stderr", types.SimpleNamespace(write=written.append, flush=lambda: None))

    train_model(config=str(tmp_path / "config.yaml"), profile=_profile(device="cpu"), shuffle=2, epochs=None)

    monitor = _FakeMonitor.instances[0]
    assert monitor.stopped
    assert not records.manager.shutdown_called
    # The renderer never drew its closing newline, so the teardown supplies one.
    assert written == ["\n"]


def test_train_model_without_progress_skips_monitor(monkeypatch, tmp_path):
    """Verifies that disabling the progress display skips the monitor and the manager entirely."""
    loader = _FakeLoader(
        model_cfg={"train_settings": {"epochs": 3}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    records = _patch_train_model_deps(monkeypatch=monkeypatch, loader=loader)

    summary = train_model(
        config=tmp_path / "config.yaml",
        profile=_profile(device="cpu"),
        display_progress=False,
        evaluate=False,
    )

    assert _FakeMonitor.instances == []
    assert records.manager.shutdown_called is False
    assert records.evaluate == []
    assert records.worker[0][1].progress_queue is None
    assert summary.evaluation is None


def test_train_model_ddp_spawns_workers(monkeypatch, tmp_path):
    """Verifies that a DDP profile spawns one worker per GPU and reports mixed-precision and strategy in the summary."""
    loader = _FakeLoader(
        model_cfg={"train_settings": {"epochs": 100}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    evaluation = _FakeEval()
    profile = _profile(device="cuda", gpus=(0, 1), multi_gpu_strategy=MultiGpuStrategy.DDP, amp_dtype=torch.bfloat16)
    records = _patch_train_model_deps(monkeypatch=monkeypatch, loader=loader, evaluation=evaluation)

    summary = train_model(config=tmp_path / "config.yaml", profile=profile)

    assert len(records.spawn) == 1
    spawn_fn, spawn_args, nprocs, join = records.spawn[0]
    assert spawn_fn is pipeline._run_training_worker
    assert nprocs == 2
    assert join is True
    # The single positional argument bundled into the spawn call is the launch.
    assert isinstance(spawn_args[0], _TrainingLaunch)
    assert [rank for rank, _launch in records.worker] == [0, 1]
    assert summary.strategy == "ddp"
    assert summary.world_size == 2
    assert summary.device == "cuda"
    assert summary.precision == "bfloat16"


def test_train_model_applies_detector_and_all_overrides(monkeypatch, tmp_path):
    """Verifies that every override is threaded into the config, including the detector-specific keys for top-down."""
    loader = _FakeLoader(
        model_cfg={
            "train_settings": {"epochs": 5},
            "detector": {"train_settings": {"epochs": 3}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=tmp_path,
    )
    _patch_train_model_deps(monkeypatch=monkeypatch, loader=loader)

    summary = train_model(
        config=tmp_path / "config.yaml",
        profile=_profile(device="cpu", dataloader_workers=2, pin_memory=False),
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
    loader = _FakeLoader(
        model_cfg={
            "train_settings": {"epochs": 0},
            "detector": {"train_settings": {"epochs": 3}},
        },
        pose_task=Task.TOP_DOWN,
        model_folder=tmp_path,
    )
    records = _patch_train_model_deps(monkeypatch=monkeypatch, loader=loader)

    summary = train_model(
        config=tmp_path / "config.yaml",
        profile=_profile(device="cpu"),
        display_progress=False,
        evaluate=True,
    )

    assert records.evaluate == []
    assert summary.tasks_trained == ("detector",)
    assert summary.evaluation is None


def test_train_model_rejects_memory_replay(monkeypatch, tmp_path):
    """Verifies that a SuperAnimal memory-replay shuffle is rejected before any training begins."""
    loader = _FakeLoader(
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
        train_model(config=tmp_path / "config.yaml", profile=_profile(device="cpu"))


def test_train_model_raises_a_report_naming_the_log_on_failure(monkeypatch, tmp_path):
    """Verifies that a worker failure raises a report quoting the training log, with the monitor still torn down."""
    loader = _FakeLoader(
        model_cfg={"train_settings": {"epochs": 10}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )

    killed = _exited("SIGKILL", -signal.SIGKILL)

    def failing_worker(*_args, **_kwargs):
        raise killed

    records = _patch_train_model_deps(monkeypatch=monkeypatch, loader=loader, worker=failing_worker)
    (tmp_path / "train.txt").write_text("epoch 3 of 10\nout of memory\n")

    with pytest.raises(TrainingFailedError) as failure:
        train_model(config=tmp_path / "config.yaml", profile=_profile(device="cpu"), shuffle=7)

    report = str(failure.value)
    assert "SIGKILL" in report
    assert "out-of-memory killer" in report
    assert str(tmp_path / "train.txt") in report
    assert "shuffle:      7" in report
    # The log tail is quoted, so the cause travels with the report rather than only with the pointer.
    assert "out of memory" in report
    # The finally block still tore down the monitor and manager despite the failure.
    assert _FakeMonitor.instances[0].stopped
    assert records.manager.shutdown_called


def test_train_model_raises_an_interruption_rather_than_a_failure(monkeypatch, tmp_path):
    """Verifies that an operator interrupt is reported as an interruption, not as a crash."""
    loader = _FakeLoader(
        model_cfg={"train_settings": {"epochs": 10}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )

    def interrupted_worker(*_args, **_kwargs):
        raise KeyboardInterrupt

    records = _patch_train_model_deps(monkeypatch=monkeypatch, loader=loader, worker=interrupted_worker)

    with pytest.raises(TrainingInterruptedError, match="interrupted"):
        train_model(config=tmp_path / "config.yaml", profile=_profile(device="cpu"))

    assert records.manager.shutdown_called


def test_train_model_rejects_a_run_that_plans_no_models(monkeypatch, tmp_path):
    """Verifies that zero pose and detector epochs are rejected before any process is created."""
    loader = _FakeLoader(
        model_cfg={"train_settings": {"epochs": 0}, "detector": {"train_settings": {"epochs": 0}}},
        pose_task=Task.TOP_DOWN,
        model_folder=tmp_path,
    )
    records = _patch_train_model_deps(monkeypatch=monkeypatch, loader=loader)

    with pytest.raises(ValueError, match="no model"):
        train_model(config=tmp_path / "config.yaml", profile=_profile(device="cpu"), display_progress=False)

    assert records.spawn == []


def test_train_model_records_an_evaluation_failure_without_losing_the_run(monkeypatch, tmp_path):
    """Verifies that a failed evaluation is recorded in the summary and the log, while training still succeeds."""
    loader = _FakeLoader(
        model_cfg={"train_settings": {"epochs": 10}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    _patch_train_model_deps(
        monkeypatch=monkeypatch,
        loader=loader,
        evaluation_error=RuntimeError("CUDA out of memory"),
    )

    summary = train_model(config=tmp_path / "config.yaml", profile=_profile(device="cpu"), display_progress=False)

    assert summary.tasks_trained == ("pose",)
    assert summary.evaluation is None
    assert summary.evaluation_error == "RuntimeError: CUDA out of memory"
    assert "evaluation FAILED (RuntimeError: CUDA out of memory)" in summary.describe()
    # The traceback outlives the terminal, because DeepLabCut's teardown has already stripped every log handler.
    log_text = (tmp_path / "train.txt").read_text()
    assert "Post-training evaluation failed." in log_text
    assert "CUDA out of memory" in log_text


def test_train_model_start_up_failure_of_the_monitor_is_reported(monkeypatch, tmp_path):
    """Verifies that a manager that fails to start raises a training failure instead of a bare EOFError."""
    loader = _FakeLoader(
        model_cfg={"train_settings": {"epochs": 10}},
        pose_task=Task.BOTTOM_UP,
        model_folder=tmp_path,
    )
    records = _patch_train_model_deps(monkeypatch=monkeypatch, loader=loader)

    def dead_manager():
        raise EOFError

    monkeypatch.setattr(pipeline.mp, "Manager", dead_manager)

    with pytest.raises(TrainingFailedError, match="progress monitor"):
        train_model(config=tmp_path / "config.yaml", profile=_profile(device="cpu"))

    assert records.spawn == []


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
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
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
    assert emptied == []


def test_evaluate_after_training_propagates_failure(monkeypatch):
    """Verifies that a failing evaluation propagates to its caller, which owns the record-rather-than-fail contract."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def boom(*_args, **_kwargs):
        message = "evaluation blew up"
        raise RuntimeError(message)

    monkeypatch.setattr(pipeline, "evaluate_trained_model", boom)

    with pytest.raises(RuntimeError, match="evaluation blew up"):
        _evaluate_after_training(
            config=Path("config.yaml"),
            profile=_profile(device="cpu"),
            shuffle=1,
            training_set_index=0,
            batch_size=1,
            confidence_cutoff=None,
        )


# Failure classification and reporting


def test_is_operator_interrupt_classification():
    """Verifies that only a keyboard interrupt and the deliberate-stop signals count as an operator interrupt."""
    assert _is_operator_interrupt(KeyboardInterrupt()) is True
    assert _is_operator_interrupt(_exited("SIGINT", -signal.SIGINT)) is True
    assert _is_operator_interrupt(_exited("SIGTERM", -signal.SIGTERM)) is True
    assert _is_operator_interrupt(_exited("SIGKILL", -signal.SIGKILL)) is False
    assert _is_operator_interrupt(_exited("SIGSEGV", -signal.SIGSEGV)) is False
    assert _is_operator_interrupt(RuntimeError("boom")) is False


def test_describe_worker_failure_classifies_each_end_of_a_worker():
    """Verifies that each failure class is named exactly, with a shell status on the signal deaths."""
    killed = _describe_worker_failure(_exited("SIGKILL", -signal.SIGKILL))
    assert "SIGKILL" in killed
    assert "shell status 137" in killed
    assert "out-of-memory killer" in killed

    crashed = _describe_worker_failure(_exited("SIGSEGV", -signal.SIGSEGV))
    assert "shell status 139" in crashed
    assert "native backend" in crashed

    exited = _describe_worker_failure(_exited(None, 3))
    assert "exited with status 3" in exited

    raised = _describe_worker_failure(ProcessRaisedException("traceback text", 0, 4242))
    assert "raised an exception" in raised

    other = _describe_worker_failure(RuntimeError("pickling failed"))
    assert "RuntimeError: pickling failed" in other


def test_describe_worker_failure_names_an_unclassified_signal():
    """Verifies that a signal outside the interrupt and native-crash sets is still named with its shell status."""
    described = _describe_worker_failure(_exited("SIGHUP", -signal.SIGHUP))
    assert "SIGHUP" in described
    assert f"shell status {128 + signal.SIGHUP}" in described


def test_format_training_failure_quotes_the_log_tail(tmp_path):
    """Verifies that a signal death report carries the project context and the tail of the training log."""
    training_log = tmp_path / "train.txt"
    training_log.write_text("".join(f"line {index}\n" for index in range(100)))

    report = _format_training_failure(
        _exited("SIGSEGV", -signal.SIGSEGV),
        config=tmp_path / "config.yaml",
        shuffle=2,
        model_folder=tmp_path,
        training_log=training_log,
    )

    assert "line 99" in report
    assert "line 60" in report
    # Only the trailing window is quoted, so the report stays readable.
    assert "line 59" not in report
    assert str(tmp_path / "config.yaml") in report


def test_format_training_failure_embeds_the_child_traceback_instead_of_the_tail(tmp_path):
    """Verifies that a worker that raised has its own traceback quoted rather than the log it already wrote."""
    training_log = tmp_path / "train.txt"
    training_log.write_text("redirected worker output\n")

    report = _format_training_failure(
        ProcessRaisedException("\n\n-- Process 0 terminated with the following error:\nBackendCompilerFailed\n", 0, 7),
        config=tmp_path / "config.yaml",
        shuffle=1,
        model_folder=tmp_path,
        training_log=training_log,
    )

    assert "BackendCompilerFailed" in report
    assert "redirected worker output" not in report


def test_format_training_failure_states_that_the_log_is_empty(tmp_path):
    """Verifies that a worker that died before logging anything is reported as such rather than with empty markers."""
    report = _format_training_failure(
        _exited("SIGKILL", -signal.SIGKILL),
        config=tmp_path / "config.yaml",
        shuffle=1,
        model_folder=tmp_path,
        training_log=tmp_path / "train.txt",
    )

    assert "holds no output" in report
    assert "last 40 lines" not in report


def test_append_training_log_preserves_existing_content(tmp_path):
    """Verifies that the appended block follows the existing log rather than replacing it."""
    training_log = tmp_path / "train.txt"
    training_log.write_text("worker output\n")

    _append_training_log(training_log=training_log, text="Post-training evaluation failed.")

    text = training_log.read_text()
    assert text.startswith("worker output\n")
    assert "Post-training evaluation failed." in text


def test_append_training_log_survives_an_unwritable_path(tmp_path):
    """Verifies that a log that cannot be written does not turn a failure report into a second failure."""
    _append_training_log(training_log=tmp_path / "missing-directory" / "train.txt", text="anything")


def test_start_progress_monitor_releases_a_manager_it_created_before_failing(monkeypatch):
    """Verifies that a manager created before the queue fails is shut down rather than left running."""
    manager = _FakeManager()

    def failing_queue():
        message = "queue server is gone"
        raise EOFError(message)

    manager.Queue = failing_queue
    monkeypatch.setattr(pipeline, "mp", types.SimpleNamespace(Manager=lambda: manager))

    with pytest.raises(TrainingFailedError, match="progress monitor"):
        pipeline._start_progress_monitor()

    assert manager.shutdown_called


def test_stop_progress_monitor_reports_a_manager_that_will_not_shut_down(monkeypatch):
    """Verifies that a manager refusing to shut down is warned about rather than allowed to mask the real failure."""

    class ExplodingManager(_FakeManager):
        def shutdown(self):
            message = "manager is unreachable"
            raise EOFError(message)

    warnings = []
    monkeypatch.setattr(pipeline, "warn", warnings.append)

    _stop_progress_monitor(monitor=None, manager=ExplodingManager())

    assert any("manager is unreachable" in warning for warning in warnings)


def test_stop_progress_monitor_reports_teardown_errors_without_raising(monkeypatch):
    """Verifies that a teardown error is warned about rather than allowed to replace the in-flight failure."""

    class ExplodingMonitor(_FakeMonitor):
        def stop(self):
            message = "queue is gone"
            raise EOFError(message)

    monitor = ExplodingMonitor(progress_queue=None)
    manager = _FakeManager()
    warnings = []
    monkeypatch.setattr(pipeline, "warn", warnings.append)

    _stop_progress_monitor(monitor=monitor, manager=manager)

    assert manager.shutdown_called
    assert any("queue is gone" in warning for warning in warnings)


# _find_free_port / _route_logging_to_file / _redirect_worker_console


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


def test_redirect_worker_console_records_traceback_in_log(tmp_path):
    """Verifies that a failure inside the redirected body is written to the log before descriptors are restored."""
    log_path = tmp_path / "worker.log"
    with pytest.raises(RuntimeError, match="worker exploded"):
        with _redirect_worker_console(log_path, active=True):
            message = "worker exploded"
            raise RuntimeError(message)
    content = log_path.read_text()
    assert "RuntimeError" in content
    assert "worker exploded" in content
    assert "Traceback" in content


# Private helpers and fakes


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


class _FakeEval:
    """Stands in for an evaluation summary that only needs to be describable."""

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
        "port": 12345,
        "world_size": 1,
    }
    defaults.update(overrides)
    return _TrainingLaunch(**defaults)


def _patch_loader(monkeypatch, loader):
    """Points the pipeline's DLCLoader binding at the given fake loader."""
    monkeypatch.setattr(pipeline, "DLCLoader", lambda **_kwargs: loader)


class _FakeBuilder:
    """Records model-build calls and returns a sentinel instead of a real network."""

    def __init__(self):
        self.calls = []

    def build(self, model, **kwargs):
        self.calls.append((model, kwargs))
        return "built-model"


class _FakeWeightInit:
    """Stands in for WeightInitialization, exposing only from_dict."""

    @staticmethod
    def from_dict(config):
        return ("weight-init", config)


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


class _FakeModel:
    """Records the device the model is moved to."""

    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class _FakeRunner:
    """Stands in for a training runner, recording its fit call."""

    def __init__(self, starting_epoch=0):
        self.starting_epoch = starting_epoch
        self.fit_kwargs = None

    def fit(self, **kwargs):
        self.fit_kwargs = kwargs


class _FakeQueueLogger:
    """Stands in for QueueTrainingLogger, recording the config it is handed."""

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


class _FakeMonitor:
    """Stands in for TrainingMonitor, recording its lifecycle."""

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
    """Stands in for a multiprocessing manager, returning a placeholder queue."""

    def __init__(self):
        self.shutdown_called = False

    def Queue(self):  # noqa: N802 - mirrors multiprocessing.Manager's Queue factory name.
        return "progress-queue"

    def shutdown(self):
        self.shutdown_called = True


def _patch_train_model_deps(monkeypatch, loader, *, worker=None, evaluation=None, evaluation_error=None):
    """Replaces train_model's process management, monitor, worker, and evaluation with recorders."""
    _FakeMonitor.instances = []
    records = types.SimpleNamespace(spawn=[], worker=[], evaluate=[], manager=_FakeManager())
    monkeypatch.setattr(pipeline, "DLCLoader", lambda **_kwargs: loader)
    monkeypatch.setattr(pipeline, "_find_free_port", lambda: 45000)
    monkeypatch.setattr(pipeline, "TrainingMonitor", _FakeMonitor)

    def default_worker(rank, launch):
        records.worker.append((rank, launch))

    monkeypatch.setattr(pipeline, "_run_training_worker", worker or default_worker)

    # Stands in for torch's spawn, which the pipeline uses for every strategy. It runs each rank in-process so a
    # test may inject a worker failure, while still recording the spawn call itself.
    def fake_spawn(fn, args, nprocs, join):
        records.spawn.append((fn, args, nprocs, join))
        for rank in range(nprocs):
            fn(rank, *args)

    fake_mp = types.SimpleNamespace(Manager=lambda: records.manager, spawn=fake_spawn)
    monkeypatch.setattr(pipeline, "mp", fake_mp)

    def fake_evaluate(**kwargs):
        records.evaluate.append(kwargs)
        if evaluation_error is not None:
            raise evaluation_error
        return evaluation

    monkeypatch.setattr(pipeline, "_evaluate_after_training", fake_evaluate)
    return records
