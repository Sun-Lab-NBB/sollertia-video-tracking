"""Provides mixed-precision, DistributedDataParallel-capable subclasses of the DeepLabCut PyTorch training runners.

These runners override DeepLabCut's internal ``PoseTrainingRunner``/``DetectorTrainingRunner`` classes to add
automatic mixed precision, ``torch.compile``, and single-node multi-process DistributedDataParallel while reusing
every DeepLabCut building block (model, optimizer, scheduler, snapshot manager, and metric computation). Because the
overrides depend on those internal classes, the DeepLabCut version is pinned exactly in ``pyproject.toml``.
"""

from typing import Any
import logging
from contextlib import AbstractContextManager, nullcontext
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.runners import schedulers
from deeplabcut.pose_estimation_pytorch.runners.train import (
    TrainingRunner,
    PoseTrainingRunner,
    DetectorTrainingRunner,
    build_optimizer,
)
from deeplabcut.pose_estimation_pytorch.runners.logger import BaseLogger, ImageLoggerMixin
from deeplabcut.pose_estimation_pytorch.runners.snapshots import TorchSnapshotManager

_logger = logging.getLogger(__name__)
"""The module logger; its records propagate to DeepLabCut's root training-log handlers (``train.txt``)."""


class _OptimizedTrainingRunnerMixin(TrainingRunner):
    """Adds mixed precision, ``torch.compile``, and DistributedDataParallel to a DeepLabCut training runner.

    The mixin is placed before a concrete DeepLabCut runner in the method resolution order so its ``fit``, ``_epoch``,
    and ``state_dict`` overrides take precedence while ``super().__init__`` still builds the stock runner. Each
    concrete subclass overrides only ``step`` to wrap its forward pass and loss in autocast. It derives from the
    (untyped) ``TrainingRunner`` base so the shared runner attributes and helpers it relies on resolve without stubs;
    it is never instantiated directly, so its own ``step`` remains abstract.

    Notes:
        Under DistributedDataParallel every process passes a single GPU index, so the base-class ``_data_parallel``
        flag (``len(gpus) > 1``) is False and the stock ``.module`` unwrap guards do not fire. The mixin therefore
        resolves the underlying model itself in ``_unwrap``, which also peels a ``torch.compile`` wrapper so
        snapshots stay free of ``module.``/``_orig_mod.`` key prefixes and load unchanged in DeepLabCut.

    Attributes:
        _amp_dtype: The autocast compute dtype, or None when training in full float32 precision.
        _gradient_scaler: The gradient scaler used for float16 precision, or None for bfloat16/float32.
        _torch_compile: Whether the model is wrapped with ``torch.compile`` before training.
        _ddp: Whether this process trains as part of a DistributedDataParallel group.
        _rank: The global rank of this process within the DistributedDataParallel group.
        _world_size: The number of processes in the DistributedDataParallel group.
        _local_rank: The local GPU index this process trains on.
    """

    # Attributes reassigned within this mixin; DeepLabCut owns and initializes them but ships no stubs, so Any.
    model: Any
    _epoch_predictions: Any
    _epoch_ground_truth: Any

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
    ) -> None:
        """Stores the optimization state and delegates the rest of construction to the wrapped DeepLabCut runner.

        Args:
            args: Positional arguments forwarded to the wrapped DeepLabCut runner.
            amp_dtype: The autocast compute dtype, or None to disable mixed precision.
            use_gradient_scaler: Whether to use a gradient scaler, which is required only for float16 precision.
            torch_compile: Whether to wrap the model with ``torch.compile`` before training.
            ddp: Whether this process participates in a DistributedDataParallel group.
            rank: The global rank of this process.
            world_size: The number of processes in the group.
            local_rank: The local GPU index this process trains on.
            kwargs: Keyword arguments forwarded to the wrapped DeepLabCut runner.
        """
        self._amp_dtype = amp_dtype
        self._gradient_scaler = torch.amp.GradScaler(device="cuda") if use_gradient_scaler else None
        self._torch_compile = torch_compile
        self._ddp = ddp
        self._rank = rank
        self._world_size = world_size
        self._local_rank = local_rank
        super().__init__(*args, **kwargs)

    @property
    def _is_main(self) -> bool:
        """Returns whether this process is responsible for evaluation, snapshots, and progress reporting."""
        return not self._ddp or self._rank == 0

    def _unwrap(self) -> Any:
        """Resolves the underlying DeepLabCut model, peeling DataParallel, DDP, and ``torch.compile`` wrappers.

        Returns:
            The original DeepLabCut model exposing ``get_target``/``get_loss``/``get_predictions`` and clean
            ``state_dict`` keys. Typed ``Any`` because DeepLabCut's model classes ship no type stubs.
        """
        model = self.model
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module
        return getattr(model, "_orig_mod", model)

    def _build_autocast_context(self) -> AbstractContextManager[None]:
        """Builds the autocast context for the forward pass and loss, or a no-op context when precision is float32.

        Returns:
            A context manager that enables mixed precision on the runner's device when configured.
        """
        if self._amp_dtype is None:
            return nullcontext()
        device_type = "cuda" if str(self.device).startswith("cuda") else str(self.device)
        return torch.autocast(device_type=device_type, dtype=self._amp_dtype)

    def _backward_and_step(self, loss: torch.Tensor) -> None:
        """Runs the backward pass and optimizer step in full precision, scaling gradients when using float16.

        Args:
            loss: The total loss tensor to back-propagate.
        """
        if self._gradient_scaler is not None:
            self._gradient_scaler.scale(loss).backward()
            self._gradient_scaler.step(self.optimizer)
            self._gradient_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def _prepare_model_for_training(self) -> None:
        """Moves the model to its device and applies ``torch.compile`` and the multi-GPU wrapper for this run."""
        self.model.to(self.device)
        if self._torch_compile:
            self.model = torch.compile(model=self.model)
        if self._ddp:
            # broadcast_buffers=False avoids a per-forward buffer-sync NCCL collective. With it enabled, a rank-0-only
            # validation forward would issue a buffer broadcast that the other ranks (parked at the end-of-epoch
            # barrier) never join, deadlocking the group. Gradients are still all-reduced each step so weights stay in
            # sync; only BatchNorm running statistics remain per-rank, which is the standard multi-GPU trade-off.
            self.model = DistributedDataParallel(
                module=self.model,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
                broadcast_buffers=False,
            )
        elif getattr(self, "_data_parallel", False):
            self.model = DataParallel(module=self.model, device_ids=self._gpus).cuda()

    def state_dict(self) -> dict:
        """Returns the runner state with model weights taken from the unwrapped model so snapshots load unchanged.

        Returns:
            The runner state dictionary containing metadata, model, optimizer, and optional scheduler state.
        """
        state = {
            "metadata": self._metadata,
            "model": self._unwrap().state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        epochs: int,
        display_iters: int,
    ) -> None:
        """Trains the model, reshuffling per epoch under DDP and confining evaluation, snapshots, and logging to rank 0.

        Args:
            train_loader: The training data loader, using a ``DistributedSampler`` under DDP.
            valid_loader: The validation data loader, consumed only on the main process.
            epochs: The number of training epochs.
            display_iters: The number of iterations between each intra-epoch loss log.
        """
        self._prepare_model_for_training()

        if self._is_main and isinstance(self.logger, ImageLoggerMixin):
            self.logger.select_images_to_log(train_loader, valid_loader)

        # Extend the epoch budget when resuming so the count reflects the total, not the extra, epochs.
        if self.starting_epoch > 0:
            epochs = self.starting_epoch + epochs

        for epoch in range(self.starting_epoch + 1, epochs + 1):
            self.current_epoch = epoch
            self._metadata["epoch"] = epoch
            if self._ddp and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_loss = self._epoch(loader=train_loader, mode="train", display_iters=display_iters)
            if self.scheduler is not None:
                self.scheduler.step()

            learning_rate = self.optimizer.param_groups[0]["lr"]
            message = f"Epoch {epoch}/{epochs} (lr={learning_rate}), train loss {float(train_loss):.5f}"
            if epoch % self.eval_interval == 0 and self._is_main:
                with torch.no_grad():
                    _logger.info("Training for epoch %d done, starting evaluation", epoch)
                    valid_loss = self._epoch(loader=valid_loader, mode="eval", display_iters=display_iters)
                    if self._print_valid_loss:
                        message += f", valid loss {float(valid_loss):.5f}"
            message += self._gpu_usage_str()

            if self._is_main:
                self.snapshot_manager.update(epoch, self.state_dict(), last=(epoch == epochs))
                _logger.info("%s", message)
                epoch_metrics = self._metadata.get("metrics")
                if epoch % self.eval_interval == 0 and epoch_metrics:
                    _logger.info("Model performance:")
                    line_length = max(len(name) for name in epoch_metrics) + 2
                    for name, score in epoch_metrics.items():
                        _logger.info("  %s%6.2f", (name + ":").ljust(line_length), score)

            if self._ddp and dist.is_initialized():
                dist.barrier()

    def _epoch(
        self,
        loader: torch.utils.data.DataLoader,
        mode: str = "train",
        display_iters: int = 500,
    ) -> float:
        """Runs one training or evaluation epoch, restricting the intra-epoch and per-epoch logging to the main process.

        Args:
            loader: The data loader iterated over for this epoch.
            mode: Either ``"train"`` or ``"eval"``.
            display_iters: The number of iterations between each intra-epoch loss log.

        Raises:
            ValueError: When the mode is neither ``"train"`` nor ``"eval"``.

        Returns:
            The mean loss over the epoch, computed from this process's shard of the data under DDP.
        """
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()
        else:
            message = f"Unable to run the epoch using mode '{mode}'. Expected 'train' or 'eval', but got '{mode}'."
            raise ValueError(message)

        epoch_loss = []
        loss_metrics = defaultdict(list)
        for iteration, batch in enumerate(loader):
            losses_dict = self.step(batch=batch, mode=mode)
            if "total_loss" in losses_dict:
                epoch_loss.append(losses_dict["total_loss"])
                if (iteration + 1) % display_iters == 0 and mode != "eval" and self._is_main:
                    _logger.info(
                        "Number of iterations: %d, loss: %.5f, lr: %s",
                        iteration + 1,
                        losses_dict["total_loss"],
                        self.optimizer.param_groups[0]["lr"],
                    )

            for key in losses_dict:
                loss_metrics[key].append(losses_dict[key])

        performance_metrics = None
        if mode == "eval":
            performance_metrics = self._compute_epoch_metrics()
            self._metadata["metrics"] = performance_metrics
            self._epoch_predictions = {}
            self._epoch_ground_truth = {}

        mean_loss = float(np.mean(epoch_loss).item()) if epoch_loss else 0.0
        self.history[f"{mode}_loss"].append(mean_loss)

        metrics_to_log = {}
        if performance_metrics:
            for name, score in performance_metrics.items():
                metrics_to_log[name] = score if isinstance(score, (int, float)) else 0.0

        for key in loss_metrics:
            name = f"{mode}.{key}"
            value = float("nan")
            if np.sum(~np.isnan(loss_metrics[key])) > 0:
                value = np.nanmean(loss_metrics[key]).item()
            self._metadata["losses"][name] = value
            metrics_to_log[f"losses/{name}"] = value

        if self._is_main:
            self.csv_logger.log(metrics_to_log, step=self.current_epoch)
            if self.logger:
                self.logger.log(metrics_to_log, step=self.current_epoch)

        return mean_loss


class _OptimizedPoseTrainingRunner(_OptimizedTrainingRunnerMixin, PoseTrainingRunner):
    """Trains pose estimation models with mixed precision and DistributedDataParallel."""

    def step(self, batch: dict[str, Any], mode: str = "train") -> dict[str, Any]:
        """Runs a single pose training or evaluation step with the forward pass and loss under autocast.

        Args:
            batch: The batch of images, annotations, and context for the step.
            mode: Either ``"train"`` or ``"eval"``.

        Raises:
            ValueError: When the mode is neither ``"train"`` nor ``"eval"``.

        Returns:
            The per-loss values for the step as detached NumPy arrays.
        """
        if mode not in ("train", "eval"):
            message = f"Unable to run the pose step using mode '{mode}'. Expected 'train' or 'eval', but got '{mode}'."
            raise ValueError(message)

        if mode == "train":
            self.optimizer.zero_grad()

        inputs = batch["image"].to(self.device).float()
        underlying_model = self._unwrap()
        with self._build_autocast_context():
            if "cond_keypoints" in batch["context"]:
                outputs = self.model(inputs, cond_kpts=batch["context"]["cond_keypoints"])
            else:
                outputs = self.model(inputs)
            target = underlying_model.get_target(outputs, batch["annotations"])
            losses_dict = underlying_model.get_loss(outputs, target)
        if mode == "train":
            self._backward_and_step(loss=losses_dict["total_loss"])

        if isinstance(self.logger, ImageLoggerMixin):
            self.logger.log_images(batch, outputs, target, step=self.current_epoch)

        if mode == "eval":
            predictions = {
                name: {key: value.detach().cpu().numpy() for key, value in prediction.items()}
                for name, prediction in underlying_model.get_predictions(outputs).items()
            }

            ground_truth = batch["annotations"]["keypoints"]
            if batch["annotations"]["with_center_keypoints"][0]:
                ground_truth = ground_truth[..., :-1, :]

            self._update_epoch_predictions(
                name="bodyparts",
                gt_keypoints=ground_truth,
                pred_keypoints=predictions["bodypart"]["poses"],
                offsets=batch["offsets"],
                scales=batch["scales"],
            )
            if "unique_bodypart" in predictions:
                self._update_epoch_predictions(
                    name="unique_bodyparts",
                    gt_keypoints=batch["annotations"]["keypoints_unique"],
                    pred_keypoints=predictions["unique_bodypart"]["poses"],
                    offsets=batch["offsets"],
                    scales=batch["scales"],
                )

        return {key: value.detach().cpu().numpy() for key, value in losses_dict.items()}


class _OptimizedDetectorTrainingRunner(_OptimizedTrainingRunnerMixin, DetectorTrainingRunner):
    """Trains object detection models with mixed precision and DistributedDataParallel."""

    def step(self, batch: dict[str, Any], mode: str = "train") -> dict[str, Any]:
        """Runs a single detector training or evaluation step with the forward pass and loss under autocast.

        Args:
            batch: The batch of images and annotations for the step.
            mode: Either ``"train"`` or ``"eval"``.

        Raises:
            ValueError: When the mode is neither ``"train"`` nor ``"eval"``.

        Returns:
            The per-loss values for the step, as detached NumPy arrays during training.
        """
        if mode not in ("train", "eval"):
            message = (
                f"Unable to run the detector step using mode '{mode}'. Expected 'train' or 'eval', but got '{mode}'."
            )
            raise ValueError(message)

        if mode == "train":
            self.optimizer.zero_grad()
            self.model.train()
        else:
            self.model.eval()

        images = batch["image"].to(self.device)
        underlying_model = self._unwrap()
        target = underlying_model.get_target(batch["annotations"])
        # Move each per-image target tensor onto the training device so the detector forward can consume it.
        for item in target:
            for key in item:
                if item[key] is not None:
                    item[key] = item[key].to(self.device)

        with self._build_autocast_context():
            losses, predictions = self.model(images, target)

        # Losses are only returned during training, not evaluation.
        if mode == "train":
            losses["total_loss"] = sum(loss_part for loss_part in losses.values())
            self._backward_and_step(loss=losses["total_loss"])
            losses = {key: value.detach().cpu().numpy() for key, value in losses.items()}
        elif mode == "eval":
            losses["total_loss"] = float("nan")
            self._update_epoch_predictions(
                paths=batch["path"],
                sizes=batch["original_size"],
                bboxes=batch["annotations"]["boxes"],
                predictions=predictions,
                offsets=batch["offsets"],
                scales=batch["scales"],
            )

        return losses


def build_optimized_training_runner(
    runner_config: dict,
    model_folder: Any,
    task: Task,
    model: nn.Module,
    device: str,
    gpus: list[int] | None = None,
    snapshot_path: Any = None,
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
) -> TrainingRunner:
    """Builds an optimized training runner, mirroring DeepLabCut's ``build_training_runner`` with added optimizations.

    This reuses DeepLabCut's optimizer, scheduler, and snapshot-manager builders unchanged and only substitutes the
    optimized runner subclasses and threads the mixed-precision and DistributedDataParallel settings through.

    Args:
        runner_config: The ``runner`` section of the pose or detector configuration.
        model_folder: The folder in which snapshots and training statistics are written.
        task: The task the runner performs, selecting the pose or detector runner subclass.
        model: The model to train.
        device: The device to train on for this process, such as a CUDA index or the CPU.
        gpus: The GPU indices for DataParallel, or a single-element list per process under DDP.
        snapshot_path: The snapshot to resume training from, if any.
        logger: The metrics logger to attach, if any.
        load_head_weights: Whether to load head weights when resuming a pose model from a snapshot.
        amp_dtype: The autocast compute dtype, or None to disable mixed precision.
        use_gradient_scaler: Whether a gradient scaler is required (float16 only).
        torch_compile: Whether to wrap the model with ``torch.compile``.
        ddp: Whether this process trains in a DistributedDataParallel group.
        rank: The global rank of this process.
        world_size: The number of processes in the DDP group.
        local_rank: The local GPU index this process trains on.

    Returns:
        The constructed optimized training runner for the requested task.
    """
    optimizer = build_optimizer(model=model, optimizer_config=runner_config["optimizer"])
    scheduler = schedulers.build_scheduler(runner_config.get("scheduler"), optimizer)

    snapshot_prefix = runner_config.get("snapshot_prefix")
    if not snapshot_prefix:
        snapshot_prefix = task.snapshot_prefix

    kwargs = {
        "model": model,
        "optimizer": optimizer,
        "snapshot_manager": TorchSnapshotManager(
            snapshot_prefix=snapshot_prefix,
            model_folder=model_folder,
            key_metric=runner_config.get("key_metric"),
            key_metric_asc=runner_config.get("key_metric_asc"),
            max_snapshots=runner_config["snapshots"]["max_snapshots"],
            save_epochs=runner_config["snapshots"]["save_epochs"],
            save_optimizer_state=runner_config["snapshots"]["save_optimizer_state"],
        ),
        "device": device,
        "gpus": gpus,
        "eval_interval": runner_config.get("eval_interval"),
        "snapshot_path": snapshot_path,
        "scheduler": scheduler,
        "load_scheduler_state_dict": runner_config.get("load_scheduler_state_dict", True),
        "logger": logger,
        "load_weights_only": runner_config.get("load_weights_only"),
        "amp_dtype": amp_dtype,
        "use_gradient_scaler": use_gradient_scaler,
        "torch_compile": torch_compile,
        "ddp": ddp,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
    }
    if task == Task.DETECT:
        return _OptimizedDetectorTrainingRunner(**kwargs)

    kwargs["load_head_weights"] = load_head_weights
    return _OptimizedPoseTrainingRunner(**kwargs)
