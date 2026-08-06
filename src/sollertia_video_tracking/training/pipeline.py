"""Provides the training pipeline that runs DeepLabCut model training with mixed precision, DDP, and a clean monitor."""

import os
import sys
import copy
import socket
from typing import Any, TextIO
import logging
from pathlib import Path
import traceback
import contextlib
from dataclasses import dataclass
from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch.data import DLCLoader, build_transforms
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.utils import fix_seeds
from deeplabcut.pose_estimation_pytorch.models import DETECTORS, PoseModel
from deeplabcut.pose_estimation_pytorch.data.collate import COLLATE_FUNCTIONS
from deeplabcut.pose_estimation_pytorch.runners.logger import setup_file_logging, destroy_file_logging

from .monitor import TrainingMonitor, QueueTrainingLogger
from .runners import build_optimized_training_runner
from .evaluation import EvaluationSummary, evaluate_trained_model
from .optimization import MultiGpuStrategy, OptimizationProfile, apply_runtime_optimizations

_logger = logging.getLogger(__name__)
"""The module logger; its records propagate to DeepLabCut's root training-log handlers (``train.txt``)."""


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Captures the outcome of a completed training run for reporting to the caller.

    Notes:
        The summary is constructed once, after training returns, from the resolved run configuration and optimization
        profile. It reports what was trained and how, not per-epoch metrics, which are streamed to the monitor and
        written to the model directory's per-model learning-statistics CSV (``learning_stats.csv`` for the pose model,
        ``learning_stats_detector.csv`` for the detector).
    """

    config: Path
    """The path of the DeepLabCut project configuration file training ran for."""
    shuffle: int
    """The shuffle index that was trained."""
    model_folder: Path
    """The directory containing the trained snapshots and training statistics."""
    tasks_trained: tuple[str, ...]
    """The models that were trained, in order (e.g. ``("detector", "pose")`` for a top-down model)."""
    device: str
    """The base device type training ran on (``"cuda"``, ``"cpu"``, or ``"mps"``)."""
    strategy: str
    """The multi-GPU execution strategy used (``"ddp"``, ``"dp"``, or ``"single"``)."""
    world_size: int
    """The number of training processes used."""
    precision: str
    """The compute precision used (``"bfloat16"``, ``"float16"``, or ``"fp32"``)."""
    epochs: int
    """The number of epochs the pose model was trained for."""

    evaluation: EvaluationSummary | None = None
    """The post-training evaluation summary, or None when evaluation was skipped or failed."""

    def describe(self) -> str:
        """Builds a human-readable summary of the training run, and the evaluation when one ran, for the CLI.

        Returns:
            A compact description of what was trained and the hardware configuration used, with the evaluation
            summary appended on a second line when a post-training evaluation ran.
        """
        trained = "+".join(self.tasks_trained) if self.tasks_trained else "nothing"
        where = f"{self.device}:{self.strategy}x{self.world_size}" if self.device == "cuda" else self.device
        summary = f"trained {trained} ({self.epochs} epochs) on {where} in {self.precision} -> {self.model_folder}"
        if self.evaluation is not None:
            summary = f"{summary}\n{self.evaluation.describe()}"
        return summary


@dataclass(frozen=True, slots=True)
class _TrainingLaunch:
    """Bundles the picklable per-run parameters shared by every training worker process."""

    config: Path
    """The path of the DeepLabCut project configuration file to train for."""
    shuffle: int
    """The shuffle index to train."""
    training_set_index: int
    """The training-set fraction index the shuffle was created with."""
    profile: OptimizationProfile
    """The resolved optimization profile describing the device, precision, and parallelism to use."""
    snapshot_path: str | Path | None
    """The pose snapshot to resume from, or None to start from scratch."""
    detector_path: str | Path | None
    """The detector snapshot to resume from, or None to start from scratch."""
    load_head_weights: bool
    """Determines whether to load head weights when resuming a pose model from a snapshot."""
    maximum_snapshots_to_keep: int | None
    """The maximum number of snapshots to retain, or None to use the configured value."""
    progress_queue: Any
    """The shared monitor queue workers report progress to, or None when progress reporting is disabled."""
    preserve_console: bool
    """Determines whether the parent holds a preserved stderr duplicate, letting the single-process worker redirect
    its console."""
    port: int
    """The free TCP port reserved for the DistributedDataParallel rendezvous."""
    world_size: int
    """The number of training processes to launch."""


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
) -> TrainingSummary:
    """Trains a DeepLabCut shuffle with the resolved hardware optimizations and a clean progress monitor.

    The training-dataset options (model architecture, split) are fixed when the shuffle is created; this function
    fits the already-created shuffle. It applies the requested overrides and the optimization profile to the
    configuration once, then launches training as either a DistributedDataParallel process group or a single process.
    The single-process path covers one GPU, the CPU, MPS, and DataParallel across multiple GPUs. For top-down
    shuffles the detector is trained before the pose model.

    Args:
        config: The path of the DeepLabCut project configuration file.
        profile: The resolved optimization profile describing the device, precision, and parallelism to use.
        shuffle: The shuffle index to train.
        training_set_index: The training-set fraction index.
        epochs: The maximum number of pose-model epochs, or None to use the configured value.
        batch_size: The pose-model batch size, or None to use the configured value.
        save_epochs: The number of epochs between pose-model snapshots, or None to use the configured value.
        display_iterations: The number of iterations between intra-epoch loss logs, or None for the configured value.
        snapshot_path: The pose snapshot to resume from, if any.
        detector_path: The detector snapshot to resume from, if any.
        detector_batch_size: The detector batch size (top-down only), or None to use the configured value.
        detector_epochs: The maximum number of detector epochs (top-down only); zero skips detector training.
        detector_save_epochs: The epochs between detector snapshots (top-down only), or None for the configured value.
        maximum_snapshots_to_keep: The maximum number of snapshots to retain, or None to use the configured value.
        load_head_weights: Determines whether to load head weights when resuming a pose model from a snapshot.
        evaluate: Determines whether to score the trained snapshot against the labeled frames as a final step and
            write the evaluation feather and provenance sidecar.
        evaluation_batch_size: The number of frames scored per forward pass during the post-training evaluation.
        evaluation_confidence_cutoff: The confidence cutoff for the evaluation's cutoff-filtered metrics, or None to
            fall back to the project configuration's ``pcutoff`` (0.6 when unset).
        display_progress: Determines whether to render the live progress monitor and route DeepLabCut's logs off the
            console.

    Returns:
        A summary of what was trained and the hardware configuration used.

    Raises:
        ValueError: When the shuffle requests SuperAnimal memory-replay fine-tuning, which this trainer does not
            support.
    """
    config = Path(config)
    loader = DLCLoader(config=config, shuffle=shuffle, trainset_index=training_set_index, modelprefix="")

    weight_init_config = loader.model_cfg["train_settings"].get("weight_init")
    if weight_init_config and WeightInitialization.from_dict(weight_init_config).memory_replay:
        message = (
            "Unable to train the shuffle using the optimized trainer. SuperAnimal memory-replay fine-tuning is not "
            "supported; use deeplabcut.train_network for that workflow."
        )
        raise ValueError(message)

    config_updates: dict[str, Any] = {
        "device": profile.device,
        "train_settings.dataloader_workers": profile.dataloader_workers,
        "train_settings.dataloader_pin_memory": profile.pin_memory,
    }
    if batch_size is not None:
        config_updates["train_settings.batch_size"] = batch_size
    if epochs is not None:
        config_updates["train_settings.epochs"] = epochs
    if save_epochs is not None:
        config_updates["runner.snapshots.save_epochs"] = save_epochs
    if display_iterations is not None:
        config_updates["train_settings.display_iters"] = display_iterations
    if loader.model_cfg.get("detector") is not None:
        config_updates["detector.device"] = profile.device
        config_updates["detector.train_settings.dataloader_workers"] = profile.dataloader_workers
        config_updates["detector.train_settings.dataloader_pin_memory"] = profile.pin_memory
        if detector_batch_size is not None:
            config_updates["detector.train_settings.batch_size"] = detector_batch_size
        if detector_epochs is not None:
            config_updates["detector.train_settings.epochs"] = detector_epochs
        if detector_save_epochs is not None:
            config_updates["detector.runner.snapshots.save_epochs"] = detector_save_epochs
        if display_iterations is not None:
            config_updates["detector.train_settings.display_iters"] = display_iterations
    loader.update_model_cfg(config_updates)

    tasks_trained = _plan_training_tasks(loader)
    model_folder = loader.model_folder
    pose_epochs = loader.model_cfg["train_settings"]["epochs"]

    progress_queue = None
    monitor = None
    manager = None
    monitor_stream = None
    if display_progress:
        manager = mp.Manager()
        progress_queue = manager.Queue()
        # Render the monitor to a preserved duplicate of stderr so it keeps reaching the terminal even when the
        # single-process training path later redirects this process's stdout and stderr to the training log.
        monitor_stream = _duplicate_stderr()
        monitor = TrainingMonitor(progress_queue=progress_queue, stream=monitor_stream)
        monitor.start()

    world_size = profile.world_size
    # Worker chatter is diverted into DeepLabCut's train.txt log while the monitor owns the console. The log is always
    # retained, and the operator is pointed to it when a run fails.
    training_log = model_folder / "train.txt" if display_progress else None
    launch = _TrainingLaunch(
        config=config,
        shuffle=shuffle,
        training_set_index=training_set_index,
        profile=profile,
        snapshot_path=snapshot_path,
        detector_path=detector_path,
        load_head_weights=load_head_weights,
        maximum_snapshots_to_keep=maximum_snapshots_to_keep,
        progress_queue=progress_queue,
        preserve_console=monitor_stream is not None,
        port=_find_free_port(),
        world_size=world_size,
    )
    succeeded = False
    try:
        if profile.use_ddp:
            mp.spawn(_run_training_worker, args=(launch,), nprocs=world_size, join=True)
        else:
            _run_training_worker(rank=0, launch=launch)
        succeeded = True
    finally:
        if monitor is not None:
            monitor.stop()
            monitor.join(timeout=3)
        # The manager owns the queue the monitor reads and the duplicate is the stream it writes, so both are released
        # only once the renderer has actually exited. A renderer still running past the join keeps them, because
        # leaking a descriptor until the process exits costs less than a live writer reaching a closed handle.
        monitor_released = monitor is None or not monitor.is_alive()
        if manager is not None and monitor_released:
            manager.shutdown()
        if monitor_stream is not None and monitor_released:
            with contextlib.suppress(Exception):
                monitor_stream.close()
        # The monitor has released the terminal, so on failure the operator can be pointed to the training log that
        # captured the worker output.
        if not succeeded and training_log is not None and training_log.exists():
            _report_training_log(training_log)

    evaluation = None
    if evaluate and "pose" in tasks_trained:
        evaluation = _evaluate_after_training(
            config=config,
            profile=profile,
            shuffle=shuffle,
            training_set_index=training_set_index,
            batch_size=evaluation_batch_size,
            confidence_cutoff=evaluation_confidence_cutoff,
        )

    precision = str(profile.amp_dtype).removeprefix("torch.") if profile.amp_dtype is not None else "fp32"
    return TrainingSummary(
        config=config,
        shuffle=shuffle,
        model_folder=model_folder,
        tasks_trained=tasks_trained,
        device=profile.device,
        strategy=profile.multi_gpu_strategy,
        world_size=world_size,
        precision=precision,
        epochs=pose_epochs,
        evaluation=evaluation,
    )


def detect_fixed_input_size(
    config: str | Path,
    *,
    shuffle: int = 1,
    training_set_index: int = 0,
) -> bool:
    """Determines whether a shuffle's training transform feeds the network a single fixed input resolution.

    The cuDNN autotuner only pays off, and only preserves deterministic training safely, when the convolution input
    shapes stay constant across steps, so this reports whether that precondition holds instead of asking the operator
    to assert it. A run is fixed-size when its pose data pipeline crops or resizes every image to one resolution and no
    variable-size detector is trained alongside it. Any inability to read the shuffle's configuration is treated
    conservatively as not fixed, since a wrong assertion of fixed size makes the autotuner harmful.

    Args:
        config: The path of the DeepLabCut project configuration file.
        shuffle: The shuffle index that will be trained.
        training_set_index: The training-set fraction index the shuffle was created with.

    Returns:
        True when the training transform's spatial input size is constant across the whole run, False otherwise.
    """
    try:
        loader = DLCLoader(
            config=Path(config),
            shuffle=shuffle,
            trainset_index=training_set_index,
            modelprefix="",
        )
        model_cfg = loader.model_cfg
        detector = model_cfg.get("detector")
        trains_detector = (
            loader.pose_task == Task.TOP_DOWN and detector is not None and detector["train_settings"]["epochs"] > 0
        )
        # A trained object detector consumes variable-size full frames, so the whole run's shared cuDNN autotuner
        # setting must stay off unless the detector's own transform is fixed-size too.
        if trains_detector and not _augmentation_is_fixed_size(detector.get("data", {}).get("train", {})):
            return False
        return _augmentation_is_fixed_size(model_cfg["data"]["train"])
    except Exception:  # noqa: BLE001 - detection is best-effort; any failure conservatively reports not-fixed.
        return False


def _augmentation_is_fixed_size(train_augmentation: dict[str, Any]) -> bool:
    """Returns whether a DeepLabCut training augmentation pipeline emits one constant spatial size.

    Fixed-size training comes from either sampling every image to a set crop (``crop_sampling``) or resizing every
    image to a set resolution (``resize`` without aspect-ratio preservation). A pipeline that does neither leaves the
    per-image size free to vary, which the padding step then rounds to differing sizes.

    Args:
        train_augmentation: The ``data.train`` augmentation block from the resolved model configuration.

    Returns:
        True when the augmentation forces a single spatial size, False otherwise.
    """
    crop = train_augmentation.get("crop_sampling")
    if isinstance(crop, dict) and _has_fixed_dimensions(crop):
        return True
    resize = train_augmentation.get("resize")
    return isinstance(resize, dict) and _has_fixed_dimensions(resize) and not resize.get("keep_ratio", False)


def _has_fixed_dimensions(block: dict[str, Any]) -> bool:
    """Returns whether an augmentation block sets a positive integer width and height.

    Args:
        block: The ``crop_sampling`` or ``resize`` augmentation block from the resolved model configuration.

    Returns:
        True when both the width and height are positive integers, False otherwise.
    """
    return _is_positive_dimension(block.get("width")) and _is_positive_dimension(block.get("height"))


def _is_positive_dimension(value: Any) -> bool:
    """Returns whether a configuration value is a positive integer image dimension.

    Args:
        value: The candidate width or height value from the augmentation configuration.

    Returns:
        True when the value is an integer greater than zero, False otherwise (booleans are rejected).
    """
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _evaluate_after_training(
    config: Path,
    profile: OptimizationProfile,
    *,
    shuffle: int,
    training_set_index: int,
    batch_size: int,
    confidence_cutoff: float | None,
) -> EvaluationSummary | None:
    """Scores the freshly trained snapshot on one device, never failing a completed training run.

    Evaluation runs in the main process after the training workers have exited, on the first configured GPU (or the
    base non-CUDA device, the CPU or MPS), so it never re-scores redundantly across the DistributedDataParallel ranks.
    Any failure is logged and swallowed, since a completed training run must not be lost to an evaluation error.

    Args:
        config: The path of the DeepLabCut project configuration file.
        profile: The resolved optimization profile, used only to choose the evaluation device.
        shuffle: The shuffle index that was trained.
        training_set_index: The training-set fraction index.
        batch_size: The number of frames scored per forward pass.
        confidence_cutoff: The confidence cutoff for the cutoff-filtered metrics, or None for the default.

    Returns:
        The evaluation summary, or None when evaluation failed.
    """
    device = f"cuda:{profile.gpus[0]}" if profile.device == "cuda" else profile.device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        return evaluate_trained_model(
            config,
            shuffle=shuffle,
            training_set_index=training_set_index,
            batch_size=batch_size,
            confidence_cutoff=confidence_cutoff,
            device=device,
        )
    except Exception:  # noqa: BLE001 - evaluation is best-effort; a completed training run must not be lost.
        _logger.warning("Post-training evaluation failed and was skipped.", exc_info=True)
        return None


def _find_free_port() -> int:
    """Reserves and returns a free TCP port for the DistributedDataParallel rendezvous.

    Returns:
        A port number the operating system reported as free.
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind(("", 0))
        return int(probe.getsockname()[1])


def _route_logging_to_file(model_folder: Path, *, quiet_console: bool) -> None:
    """Sends DeepLabCut's training logs to the model directory's ``train.txt`` and optionally off the console.

    Args:
        model_folder: The directory in which the ``train.txt`` log is written.
        quiet_console: Determines whether to detach the console handler so the progress monitor owns the terminal.
    """
    setup_file_logging(model_folder / "train.txt")
    if not quiet_console:
        return
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)


def _duplicate_stderr() -> TextIO | None:
    """Returns a writable stream on a duplicate of the standard error descriptor, or None when it has none.

    The duplicate refers to the same terminal as the original stderr but is a distinct descriptor, so it survives a
    later ``os.dup2`` redirection of descriptor 2. The progress monitor renders through it to keep the console while
    the single-process training path routes descriptors 1 and 2 to the training log.

    Returns:
        A stream wrapping a duplicate of the stderr descriptor, or None when stderr exposes no descriptor, such as
        under output capture.
    """
    try:
        descriptor = os.dup(sys.stderr.fileno())
    except (OSError, ValueError):
        return None
    return os.fdopen(descriptor, "w")


@contextlib.contextmanager
def _redirect_worker_console(log_path: Path, *, active: bool) -> Iterator[None]:
    """Routes this process's stdout and stderr into the training log at the descriptor level while active.

    A descriptor-level redirection, rather than reassigning ``sys.stdout`` and ``sys.stderr``, is required to capture
    the output the progress monitor must not compete with. That output is DeepLabCut's ``print`` calls, the Hugging
    Face download bar, and the C++ ``c10d`` and NCCL messages that write straight to descriptor 2. The original
    descriptors are restored on exit, so a re-raised worker traceback still reaches the console.

    Args:
        log_path: The training-log file the diverted output is appended to.
        active: Determines whether to redirect; when False the context does nothing, leaving raw output on the console.

    Yields:
        None, for the duration of the redirection.
    """
    if not active:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    log_descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(log_descriptor, 1)
        os.dup2(log_descriptor, 2)
        yield
    except BaseException:
        # Record the traceback while the descriptors still reach the log. Restoring them below sends the propagating
        # exception to the console alone, which leaves no record once the terminal is closed or its scrollback rolls,
        # and a worker that dies inside a spawned process writes only a bare object dump here to explain the run.
        os.write(log_descriptor, traceback.format_exc().encode("utf-8", errors="replace"))
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(log_descriptor)


def _report_training_log(training_log: Path) -> None:
    """Writes a console notice pointing the operator to the training log after a failed run.

    Args:
        training_log: The training-log file that captured the worker's diverted stdout and stderr.
    """
    sys.stderr.write(
        f"\nTraining did not complete. The worker output (DeepLabCut, Hugging Face, and distributed-backend messages) "
        f"was captured in the training log at {training_log}. Review it for the underlying cause.\n"
    )
    sys.stderr.flush()


def _plan_training_tasks(loader: DLCLoader) -> tuple[str, ...]:
    """Determines which models the run will train, in order, from the loaded configuration.

    Args:
        loader: The loader holding the resolved model configuration.

    Returns:
        The ordered names of the models to train (``"detector"`` before ``"pose"`` for trained top-down models).
    """
    tasks = []
    detector = loader.model_cfg.get("detector")
    if loader.pose_task == Task.TOP_DOWN and detector is not None and detector["train_settings"]["epochs"] > 0:
        tasks.append("detector")
    if loader.model_cfg["train_settings"]["epochs"] > 0:
        tasks.append("pose")
    return tuple(tasks)


def _resolve_process_placement(
    profile: OptimizationProfile, rank: int, task: Task | None = None
) -> tuple[str, list[int] | None, bool, int]:
    """Resolves the device, GPU indices, DDP flag, and local rank the runner uses for one training process.

    Args:
        profile: The resolved optimization profile.
        rank: The global rank of this process.
        task: The task the process is about to train, used to apply task-specific device fallbacks, or None when the
            placement is only needed for the DDP flag and local rank rather than a task's runner device.

    Returns:
        A tuple of the runner device string, the GPU index list (or None), whether DDP is used, and the local rank.
    """
    if profile.device != "cuda":
        # DeepLabCut cannot train object detectors on MPS (the torchvision NMS and ROI-align ops are unimplemented
        # there), so fall the detector back to the CPU exactly as deeplabcut.train does, while the pose model may
        # still train on MPS.
        if profile.device == "mps" and task == Task.DETECT:
            return "cpu", None, False, 0
        return profile.device, None, False, 0
    if profile.multi_gpu_strategy == MultiGpuStrategy.DDP:
        gpu_index = profile.gpus[rank]
        return "cuda", [gpu_index], True, gpu_index
    if profile.multi_gpu_strategy == MultiGpuStrategy.DP:
        return "cuda", list(profile.gpus), False, 0
    return "cuda", [profile.gpus[0]], False, 0


def _build_pose_or_detector_model(run_config: dict, task: Task, snapshot_path: str | Path | None) -> torch.nn.Module:
    """Builds the pose or detector model, honoring transfer-learning weights and pretrained-backbone rules.

    Args:
        run_config: The model and run configuration for the model to build.
        task: The task the model performs.
        snapshot_path: The snapshot training resumes from, if any, which disables pretrained initialization.

    Returns:
        The constructed DeepLabCut model.
    """
    weight_init = None
    pretrained = True
    weight_init_config = run_config["train_settings"].get("weight_init")
    if weight_init_config:
        weight_init = WeightInitialization.from_dict(weight_init_config)
        pretrained = False
    elif snapshot_path is not None:
        pretrained = False

    if task == Task.DETECT:
        return DETECTORS.build(run_config["model"], weight_init=weight_init, pretrained=pretrained)
    return PoseModel.build(run_config["model"], weight_init=weight_init, pretrained_backbone=pretrained)


def _build_dataloaders(
    loader: DLCLoader,
    run_config: dict,
    task: Task,
    *,
    ddp: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Builds the training and validation dataloaders, injecting a DistributedSampler under DDP.

    Args:
        loader: The loader that creates the datasets.
        run_config: The run configuration providing the batch size, worker count, and collate function.
        task: The task the datasets are built for.
        ddp: Determines whether the training dataloader must shard data across processes with a DistributedSampler.
        rank: The global rank of this process, used by the DistributedSampler.
        world_size: The number of processes, used by the DistributedSampler.

    Returns:
        The training and validation dataloaders.
    """
    transform = build_transforms(run_config["data"]["train"])
    inference_transform = build_transforms(run_config["data"]["inference"])
    train_dataset = loader.create_dataset(transform=transform, mode="train", task=task)
    valid_dataset = loader.create_dataset(transform=inference_transform, mode="test", task=task)

    collate_function = None
    collate_config = run_config["data"]["train"].get("collate")
    if collate_config:
        collate_function = COLLATE_FUNCTIONS.build(collate_config)

    batch_size = run_config["train_settings"]["batch_size"]
    worker_count = run_config["train_settings"]["dataloader_workers"]
    pin_memory = run_config["train_settings"]["dataloader_pin_memory"]

    # Retains the worker processes across epochs. Windows and macOS spawn each worker rather than forking it, so a
    # worker rebuilt every epoch re-imports this package and the DeepLabCut backend it loads, starving the GPU for
    # several seconds per worker per epoch.
    if ddp:
        sampler: DistributedSampler = DistributedSampler(
            dataset=train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_function,
            num_workers=worker_count,
            pin_memory=pin_memory,
            persistent_workers=worker_count > 0,
        )
    else:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_function,
            num_workers=worker_count,
            pin_memory=pin_memory,
            persistent_workers=worker_count > 0,
        )
    # Validation draws single-image batches over the held-out split, so worker processes add a second pool whose spawn
    # cost is not repaid by the little decoding they do. Loading it in the training process keeps that pool off the
    # platforms that spawn rather than fork, where each worker pays a full interpreter start.
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_dataloader, valid_dataloader


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
) -> None:
    """Builds and fits one model (pose or detector) on this process with the configured optimizations.

    Args:
        loader: The loader holding the datasets and model directory.
        run_config: The run configuration for the model being trained.
        task: The task the model performs.
        profile: The resolved optimization profile.
        rank: The global rank of this process.
        world_size: The number of training processes.
        snapshot_path: The snapshot to resume this model from, if any.
        load_head_weights: Determines whether to load head weights when resuming a pose model.
        maximum_snapshots_to_keep: The maximum number of snapshots to retain, or None to use the configured value.
        progress_queue: The shared monitor queue, or None when progress reporting is disabled.
    """
    device, gpus, ddp, local_rank = _resolve_process_placement(profile=profile, rank=rank, task=task)
    if maximum_snapshots_to_keep is not None:
        run_config["runner"]["snapshots"]["max_snapshots"] = maximum_snapshots_to_keep

    # Falls back to the configuration's resume snapshot when none was passed explicitly, matching DeepLabCut's train().
    if snapshot_path is None:
        snapshot_path = run_config.get("resume_training_from")

    model = _build_pose_or_detector_model(run_config=run_config, task=task, snapshot_path=snapshot_path)
    # Moves the model to its device before the optimizer is built and the snapshot loads, so a resumed optimizer
    # state lands on the parameters' device (DeepLabCut moves the model before building its runner).
    model.to(f"cuda:{gpus[0]}" if gpus else device)

    logger = None
    if rank == 0 and progress_queue is not None:
        logger = QueueTrainingLogger(progress_queue, task_name=("detector" if task == Task.DETECT else "pose"))

    runner = build_optimized_training_runner(
        runner_config=run_config["runner"],
        model_folder=loader.model_folder,
        task=task,
        model=model,
        device=device,
        gpus=gpus,
        snapshot_path=snapshot_path,
        logger=logger,
        load_head_weights=load_head_weights,
        amp_dtype=profile.amp_dtype,
        use_gradient_scaler=profile.use_gradient_scaler,
        torch_compile=profile.torch_compile,
        ddp=ddp,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    if logger is not None:
        # Reports the true total (starting epoch plus configured epochs) to the monitor, since the runner trains to
        # that total when resuming.
        total_epochs = runner.starting_epoch + run_config["train_settings"]["epochs"]
        logger.log_config({**run_config, "train_settings": {**run_config["train_settings"], "epochs": total_epochs}})

    train_dataloader, valid_dataloader = _build_dataloaders(
        loader=loader, run_config=run_config, task=task, ddp=ddp, rank=rank, world_size=world_size
    )
    runner.fit(
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        epochs=run_config["train_settings"]["epochs"],
        display_iters=run_config["train_settings"]["display_iters"],
    )


def _run_training_worker(rank: int, launch: _TrainingLaunch) -> None:
    """Runs training for one process, initializing DDP when required and training the detector then the pose model.

    Args:
        rank: The global rank of this process (0 for the single-process path).
        launch: The bundle of picklable per-run parameters shared by every worker process.
    """
    profile = launch.profile
    progress_queue = launch.progress_queue
    world_size = launch.world_size
    snapshot_path = launch.snapshot_path
    detector_path = launch.detector_path
    load_head_weights = launch.load_head_weights
    maximum_snapshots_to_keep = launch.maximum_snapshots_to_keep

    _device, _gpus, ddp, local_rank = _resolve_process_placement(profile=profile, rank=rank)
    # The loader is built before the process group so the training-log path is known in time to divert this worker's
    # console output around distributed initialization, where the c10d and NCCL C++ layers write their first messages.
    loader = DLCLoader(
        config=launch.config,
        shuffle=launch.shuffle,
        trainset_index=launch.training_set_index,
        modelprefix="",
    )
    # A spawned DDP worker is its own process, so redirecting its descriptors never touches the monitor in the parent.
    # The single-process path shares this process with the monitor, so it may only redirect once the monitor holds a
    # preserved stderr duplicate to render through.
    quiet_console = progress_queue is not None
    redirect_console = quiet_console and (ddp or launch.preserve_console)
    try:
        with _redirect_worker_console(loader.model_folder / "train.txt", active=redirect_console):
            if ddp:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["MASTER_PORT"] = str(launch.port)
                dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
                torch.cuda.set_device(local_rank)

            fix_seeds(loader.model_cfg["train_settings"]["seed"])
            apply_runtime_optimizations(profile)

            if rank == 0:
                _route_logging_to_file(loader.model_folder, quiet_console=quiet_console)
                _logger.info("Optimized training: %s", profile.describe())

            detector = loader.model_cfg.get("detector")
            if loader.pose_task == Task.TOP_DOWN and detector is not None and detector["train_settings"]["epochs"] > 0:
                detector_config = copy.deepcopy(detector)
                detector_config["device"] = loader.model_cfg["device"]
                detector_config["train_settings"]["weight_init"] = loader.model_cfg["train_settings"].get("weight_init")
                _train_single_model(
                    loader=loader,
                    run_config=detector_config,
                    task=Task.DETECT,
                    profile=profile,
                    rank=rank,
                    world_size=world_size,
                    snapshot_path=detector_path,
                    load_head_weights=load_head_weights,
                    maximum_snapshots_to_keep=maximum_snapshots_to_keep,
                    progress_queue=progress_queue,
                )
                if ddp:
                    dist.barrier()

            if loader.model_cfg["train_settings"]["epochs"] > 0:
                _train_single_model(
                    loader=loader,
                    run_config=loader.model_cfg,
                    task=loader.pose_task,
                    profile=profile,
                    rank=rank,
                    world_size=world_size,
                    snapshot_path=snapshot_path,
                    load_head_weights=load_head_weights,
                    maximum_snapshots_to_keep=maximum_snapshots_to_keep,
                    progress_queue=progress_queue,
                )
    finally:
        if rank == 0:
            destroy_file_logging()
        if ddp and dist.is_initialized():
            dist.destroy_process_group()
