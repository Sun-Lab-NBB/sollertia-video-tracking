"""Provides the training pipeline that runs DeepLabCut model training with mixed precision, DDP, and a clean monitor."""

import os
import sys
import copy
import time
import socket
from typing import Any
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
from torch.multiprocessing import ProcessExitedException, ProcessRaisedException
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch.data import DLCLoader, build_transforms
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.utils import fix_seeds
from deeplabcut.pose_estimation_pytorch.models import DETECTORS, PoseModel
from deeplabcut.pose_estimation_pytorch.data.collate import COLLATE_FUNCTIONS
from deeplabcut.pose_estimation_pytorch.runners.logger import setup_file_logging, destroy_file_logging

from .monitor import TrainingMonitor, QueueTrainingLogger
from .runners import build_optimized_training_runner
from ..hardware import warn
from ..reporting import (
    PipelineFailedError,
    PipelineInterruptedError,
    read_file_tail,
    is_interrupt_signal,
    describe_process_exit,
    enable_native_crash_dumps,
)
from .evaluation import EvaluationSummary, evaluate_trained_model, resolve_evaluation_batch_size
from .optimization import MultiGpuStrategy, OptimizationProfile, apply_runtime_optimizations

_logger = logging.getLogger(__name__)
"""The module logger. Its records propagate to DeepLabCut's root training-log handlers (``train.txt``)."""

_TRAINING_LOG_TAIL_LINES: int = 40
"""The number of trailing training-log lines quoted in a failure report. It is long enough to carry a native crash
message and the epoch it interrupted, and short enough to keep the report readable."""

_TRAINING_MEMORY_REMEDY: str = (
    "Lower the batch size, lower the dataloader worker count, or free host memory, then re-run."
)
"""The advice offered when the out-of-memory killer ends a training worker."""


class TrainingFailedError(PipelineFailedError):
    """Indicates that a training run did not complete, carrying the whole operator-facing report as its message."""


class TrainingInterruptedError(PipelineInterruptedError):
    """Indicates that a training run was stopped by the operator or by a termination signal rather than by a fault."""


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
    evaluation_error: str | None = None
    """The type and message of the post-training evaluation failure, or None when evaluation ran or was skipped."""

    def describe(self) -> str:
        """Builds a human-readable summary of the training run, and the evaluation outcome, for the CLI.

        Returns:
            A compact description of what was trained and the hardware configuration used, with the evaluation
            summary appended on a second line when a post-training evaluation ran, or the evaluation failure when
            one was attempted and failed.
        """
        trained = "+".join(self.tasks_trained) if self.tasks_trained else "nothing"
        where = f"{self.device}:{self.strategy}x{self.world_size}" if self.device == "cuda" else self.device
        summary = f"trained {trained} ({self.epochs} epochs) on {where} in {self.precision} -> {self.model_folder}"
        if self.evaluation is not None:
            summary = f"{summary}\n{self.evaluation.describe()}"
        elif self.evaluation_error is not None:
            summary = (
                f"{summary}\nevaluation FAILED ({self.evaluation_error}). The trained snapshots are intact, see "
                f"{self.model_folder / 'train.txt'}"
            )
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

    The training-dataset options (model architecture, split) are fixed when the shuffle is created. This function fits
    the already-created shuffle. It applies the requested overrides and the optimization profile to the configuration
    once, then launches training as either a DistributedDataParallel process group or a single process. The
    single-process path covers one GPU, the CPU, MPS, and DataParallel across multiple GPUs. For top-down shuffles the
    detector is trained before the pose model.

    Notes:
        Training always runs in spawned worker processes, so this process keeps its own standard output and error.
        That leaves it able to report a worker that dies without unwinding, such as one killed by the out-of-memory
        killer or by a crash inside a native backend. The workers divert their console into the training log while the
        monitor owns the terminal, and a failure report quotes the tail of that log.

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
        detector_epochs: The maximum number of detector epochs (top-down only). Zero skips detector training.
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
            support, or when the resolved epoch budgets leave no model to train.
        TrainingFailedError: When the progress monitor cannot be started, or when a training worker fails or dies,
            carrying the operator-facing failure report.
        TrainingInterruptedError: When the operator stops the run before it completes.
    """
    config = Path(config)
    # Installs the fault handler in this process before any worker exists, so a native crash on this side of the
    # launch leaves a readable dump on the terminal instead of ending the run without a word.
    enable_native_crash_dumps()
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
    if not tasks_trained:
        detector = loader.model_cfg.get("detector")
        detector_epochs_planned = detector["train_settings"]["epochs"] if detector is not None else 0
        message = (
            f"Unable to train shuffle {shuffle} of the project at {config}. The resolved epoch budgets leave no model "
            f"to train, because the pose budget is {pose_epochs} and the detector budget is {detector_epochs_planned}. "
            f"Request a positive epoch count, or a positive detector epoch count to train a top-down shuffle's "
            f"detector alone."
        )
        raise ValueError(message)

    progress_queue: Any = None
    monitor: TrainingMonitor | None = None
    manager: Any = None
    if display_progress:
        manager, progress_queue, monitor = _start_progress_monitor()

    world_size = profile.world_size
    # Worker chatter is diverted into DeepLabCut's train.txt log while the monitor owns the console. The log is always
    # retained, and its tail is quoted back to the operator when a run fails.
    training_log = model_folder / "train.txt"
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
        port=_find_free_port(),
        world_size=world_size,
    )
    try:
        # Every strategy launches through spawn, including the single-process ones, so the process that has to report
        # a failure is never the process that gave its descriptors to the training log. A worker killed by a signal
        # runs no Python at all, and only a surviving parent is able to turn that death into a report.
        mp.spawn(_run_training_worker, args=(launch,), nprocs=world_size, join=True)
    except (Exception, KeyboardInterrupt) as error:
        if _is_operator_interrupt(error):
            message = (
                f"Training was interrupted before it completed. Snapshots already written to {model_folder} are "
                f"intact, so the run can be resumed from the most recent one."
            )
            raise TrainingInterruptedError(message) from error
        message = _format_training_failure(
            error,
            config=config,
            shuffle=shuffle,
            model_folder=model_folder,
            training_log=training_log,
        )
        raise TrainingFailedError(message) from error
    finally:
        _stop_progress_monitor(monitor=monitor, manager=manager)

    evaluation = None
    evaluation_error = None
    if evaluate and "pose" in tasks_trained:
        try:
            evaluation = _evaluate_after_training(
                config=config,
                profile=profile,
                shuffle=shuffle,
                training_set_index=training_set_index,
                batch_size=evaluation_batch_size,
                confidence_cutoff=evaluation_confidence_cutoff,
            )
        except Exception as error:
            # A completed training run is never lost to an evaluation error, so the failure is recorded rather than
            # raised. DeepLabCut's teardown has already stripped every root log handler, so the traceback is appended
            # to the training log directly to leave a record that outlives the terminal.
            evaluation_error = f"{type(error).__name__}: {error}"
            _append_training_log(
                training_log=training_log,
                text=f"Post-training evaluation failed.\n{traceback.format_exc()}",
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
        evaluation_error=evaluation_error,
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
    except Exception:
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
) -> EvaluationSummary:
    """Scores the freshly trained snapshot on one device.

    Evaluation runs in the main process after the training workers have exited, on the first configured GPU (or the
    base non-CUDA device, the CPU or MPS), so it never re-scores redundantly across the DistributedDataParallel ranks.

    Args:
        config: The path of the DeepLabCut project configuration file.
        profile: The resolved optimization profile, used only to choose the evaluation device.
        shuffle: The shuffle index that was trained.
        training_set_index: The training-set fraction index.
        batch_size: The number of frames scored per forward pass.
        confidence_cutoff: The confidence cutoff for the cutoff-filtered metrics, or None for the default.

    Returns:
        The evaluation summary.

    Raises:
        Exception: Whatever the underlying evaluation raised, propagated unchanged.
    """
    device = f"cuda:{profile.gpus[0]}" if profile.device == "cuda" else profile.device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return evaluate_trained_model(
        config=config,
        shuffle=shuffle,
        training_set_index=training_set_index,
        batch_size=batch_size,
        confidence_cutoff=confidence_cutoff,
        device=device,
    )


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


def _start_progress_monitor() -> tuple[Any, Any, TrainingMonitor]:
    """Creates the shared progress queue and starts the monitor thread that renders the training progress bar.

    Returns:
        A tuple of the manager owning the queue, the queue itself, and the started monitor.

    Raises:
        TrainingFailedError: When the manager process or the monitor thread cannot be started.
    """
    manager = None
    try:
        manager = mp.Manager()
        progress_queue = manager.Queue()
        monitor = TrainingMonitor(progress_queue=progress_queue)
        monitor.start()
    except Exception as error:
        if manager is not None:
            with contextlib.suppress(Exception):
                manager.shutdown()
        # The manager runs a spawned child that re-imports this package, so its failure is reported here rather than
        # left to reach the CLI as the bare EOFError that a dead manager raises.
        message = (
            f"Unable to start the training progress monitor, so training was not launched. Its manager process "
            f"failed with {type(error).__name__}: {error}. Re-run without the progress monitor to train anyway."
        )
        raise TrainingFailedError(message) from error
    return manager, progress_queue, monitor


def _stop_progress_monitor(monitor: TrainingMonitor | None, manager: Any) -> None:
    """Tears the progress monitor down without ever masking the failure that reached the enclosing block.

    Args:
        monitor: The started monitor, or None when progress reporting is disabled.
        manager: The manager owning the progress queue, or None when progress reporting is disabled.
    """
    renderer_running = False
    if monitor is not None:
        try:
            monitor.stop()
            monitor.join(timeout=3)
        except Exception as error:
            warn(f"The training progress monitor did not stop cleanly ({type(error).__name__}: {error}).")
        renderer_running = monitor.is_alive()
    # The manager owns the queue the monitor reads, so it is released only once the renderer has actually exited. A
    # renderer still running past the join keeps it, because leaking it until the process exits costs less than a live
    # reader reaching a torn-down queue.
    if manager is not None and not renderer_running:
        try:
            manager.shutdown()
        except Exception as error:
            warn(f"The training progress monitor's queue manager did not shut down ({type(error).__name__}: {error}).")
    if renderer_running:
        # The renderer never drew its closing newline, so anything printed next would otherwise start mid-bar.
        sys.stderr.write("\n")
        sys.stderr.flush()


@contextlib.contextmanager
def _redirect_worker_console(log_path: Path, *, active: bool) -> Iterator[None]:
    """Routes this process's stdout and stderr into the training log at the descriptor level while active.

    A descriptor-level redirection, rather than reassigning ``sys.stdout`` and ``sys.stderr``, is required to capture
    the output the progress monitor must not compete with. That output is DeepLabCut's ``print`` calls, the Hugging
    Face download bar, and the C++ ``c10d`` and NCCL messages that write straight to descriptor 2. The original
    descriptors are restored on exit, so a re-raised worker traceback still reaches the console.

    Args:
        log_path: The training-log file the diverted output is appended to.
        active: Determines whether to redirect. When False the context does nothing, leaving raw output on the console.

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
        # exception to the console alone, which leaves no record once the terminal is closed or its scrollback rolls. A
        # worker that dies inside a spawned process writes only a bare object dump here to explain the run.
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


def _is_operator_interrupt(error: BaseException) -> bool:
    """Determines whether a failed training launch was a deliberate stop rather than a fault.

    Args:
        error: The exception that ended the training launch.

    Returns:
        True when the run was stopped by the operator or by a termination signal, False otherwise.
    """
    if isinstance(error, KeyboardInterrupt):
        return True
    return isinstance(error, ProcessExitedException) and is_interrupt_signal(error.signal_name)


def _format_training_failure(
    error: BaseException,
    *,
    config: Path,
    shuffle: int,
    model_folder: Path,
    training_log: Path,
) -> str:
    """Builds the operator-facing report for a training run that did not complete.

    Args:
        error: The exception that ended the training launch.
        config: The path of the DeepLabCut project configuration file training ran for.
        shuffle: The shuffle index that was being trained.
        model_folder: The directory holding the shuffle's snapshots and training statistics.
        training_log: The training-log file that captured the workers' diverted output.

    Returns:
        The report naming how the run ended, where its artifacts are, and the evidence for the cause.
    """
    report = [
        f"slvt train failed: {_describe_worker_failure(error)}",
        f"  project:      {config}",
        f"  shuffle:      {shuffle}",
        f"  model folder: {model_folder}",
        f"  training log: {training_log}",
    ]
    # A worker that raised carries its own traceback, and the console redirection has already written that same text
    # into the training log, so quoting the traceback alone keeps the report from printing the cause twice.
    if isinstance(error, ProcessRaisedException):
        report.append(error.msg.rstrip("\n"))
        return "\n".join(report)

    tail = read_file_tail(training_log, lines=_TRAINING_LOG_TAIL_LINES)
    if not tail:
        report.append(f"The training log at {training_log} holds no output, so the worker died before it logged any.")
        return "\n".join(report)
    report.append(f"--- last {_TRAINING_LOG_TAIL_LINES} lines of {training_log.name} ---")
    report.append(tail)
    report.append(f"--- end of {training_log.name} ---")
    return "\n".join(report)


def _describe_worker_failure(error: BaseException) -> str:
    """Builds the sentences naming how a training worker ended and what the operator should do about it.

    Args:
        error: The exception that ended the training launch.

    Returns:
        The reason the run ended, followed by the remediation for the failure classes that have one.
    """
    if isinstance(error, ProcessRaisedException):
        return "the training worker raised an exception, reproduced below."
    if not isinstance(error, ProcessExitedException):
        return f"the training launch failed with {type(error).__name__}: {error}."
    return describe_process_exit(
        error.exit_code,
        pid=error.error_pid,
        role="training worker",
        memory_remedy=_TRAINING_MEMORY_REMEDY,
    )


def _append_training_log(training_log: Path, text: str) -> None:
    """Appends a timestamped block to the training log, treating a failed write as non-fatal.

    Args:
        training_log: The training-log file the block is appended to.
        text: The block to append.
    """
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with contextlib.suppress(OSError), training_log.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{stamp}] {text}\n")


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

    # Retains the worker processes across epochs. The package forces the spawn start method on every platform, so a
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
    # Validation batches only as far as the training batch size. Its forward pass runs under no-grad, so a batch the
    # training step already holds cannot exhaust the device, which keeps the larger batch from turning a run that fits
    # into one that runs out of memory mid-epoch. The size drops back to one whenever the labeled frames span several
    # resolutions, which default collation cannot stack.
    valid_batch_size = resolve_evaluation_batch_size(loader=loader, requested=batch_size)
    # Validation decodes little, so worker processes add a second pool whose spawn cost is not repaid. Loading it in
    # the training process keeps that pool off a run whose every worker pays a full interpreter start, which the
    # package-wide spawn start method makes true on every platform.
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=valid_batch_size,
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
        logger = QueueTrainingLogger(
            progress_queue=progress_queue,
            task_name=("detector" if task == Task.DETECT else "pose"),
        )

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
    # Installs the fault handler before the console is redirected, so a native crash writes its dump into the training
    # log that the parent quotes back to the operator.
    enable_native_crash_dumps()
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
    # Every worker is a spawned process, so redirecting its descriptors never touches the monitor rendering in the
    # parent. The redirection is needed only while the monitor owns the terminal.
    quiet_console = progress_queue is not None
    redirect_console = quiet_console
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
        # Teardown runs while a training failure may already be propagating toward the parent's error queue, so an
        # error raised here is reported rather than allowed to replace the failure the operator needs to see.
        if rank == 0:
            try:
                destroy_file_logging()
            except Exception as error:
                warn(f"The training log handlers did not detach cleanly ({type(error).__name__}: {error}).")
        if ddp and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as error:
                warn(f"The distributed process group did not tear down cleanly ({type(error).__name__}: {error}).")
