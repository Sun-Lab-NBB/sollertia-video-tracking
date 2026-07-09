"""Provides the clean training progress monitor that replaces DeepLabCut's per-iteration terminal logging stream."""

import math
from typing import Any, TextIO
import contextlib

from deeplabcut.pose_estimation_pytorch.runners.logger import BaseLogger

from ..reporting import LiveBar, format_duration

_MAX_RENDERED_METRICS: int = 3
"""The maximum number of evaluation metrics shown on the progress line to keep it compact."""

_RENDERED_METRIC_PREFIXES: tuple[str, ...] = ("mAP", "mAR", "rmse")
"""The evaluation-metric name prefixes shown on the progress line, in priority order."""


class QueueTrainingLogger(BaseLogger):
    """Streams per-epoch training metrics to a monitor thread over a shared queue.

    Notes:
        This is a DeepLabCut ``BaseLogger`` built directly on the rank-0 training process and attached to the runner.
        It receives one call per training phase per epoch (train every epoch, evaluation on evaluation epochs) and
        forwards each as a small JSON-serializable message. A dropped message only skips a redraw, so a full or
        broken queue never disrupts training.

    Attributes:
        _progress_queue: The shared queue the monitor thread consumes progress messages from.
        _task_name: The training task label forwarded to the monitor.
    """

    def __init__(self, progress_queue: Any, task_name: str = "pose") -> None:
        """Initializes the logger over the shared progress queue.

        Args:
            progress_queue: The shared queue the monitor thread consumes progress messages from.
            task_name: The training task label forwarded to the monitor (e.g. ``"pose"`` or ``"detector"``).
        """
        self._progress_queue = progress_queue
        self._task_name = task_name

    def log_config(self, config: dict | None = None) -> None:
        """Forwards the epoch budget so the monitor can render an accurate progress bar.

        Args:
            config: The resolved training configuration for the run.
        """
        if config is None:
            return
        train_settings = config.get("train_settings", {})
        self._put(
            {
                "kind": "config",
                "epochs": train_settings.get("epochs"),
                "task": self._task_name,
            }
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Forwards one phase of epoch metrics to the monitor, tagged so it can be merged by epoch.

        Args:
            metrics: The metric values for this phase, keyed as ``losses/<phase>.<name>`` and ``metrics/test.<name>``.
            step: The epoch the metrics belong to.
        """
        phase = "evaluation" if any(key.startswith(("losses/eval.", "metrics/")) for key in metrics) else "training"
        self._put({"kind": "metrics", "epoch": step, "phase": phase, "metrics": dict(metrics)})

    def save(self) -> None:
        """Does nothing, because metrics are streamed as they arrive and teardown is signaled by the pipeline."""
        return

    def _put(self, message: dict[str, Any]) -> None:
        """Places a message on the progress queue, suppressing any error so training is never disrupted.

        Args:
            message: The progress message to forward to the monitor thread.
        """
        with contextlib.suppress(Exception):
            self._progress_queue.put_nowait(message)


class TrainingMonitor(LiveBar):
    """Renders a single live training progress bar from the metrics streamed by the rank-0 training process.

    Notes:
        The renderer consumes ``{"kind": "config", ...}`` and ``{"kind": "metrics", ...}`` messages, on top of the
        warm-up, spinner, interval, and ``stop``-sentinel handling inherited from ``LiveBar``. The warm-up line shows
        while no epoch budget has been reported yet, which covers the workers still preparing: initializing the process
        group, downloading pretrained weights, building the model, and any ``torch.compile`` warm-up. Because
        evaluation epochs deliver a train message and an evaluation message at the same epoch, the latest of each is
        retained and merged onto one line.

    Attributes:
        _total_epochs: The total number of epochs the run trains for, once reported by the configuration message.
        _task: The training task label reported by the configuration message.
        _current_epoch: The most recent epoch reported by the training process.
        _training_loss: The most recent training loss, or None before the first epoch completes.
        _validation_loss: The most recent validation loss, or None before the first evaluation epoch.
        _metrics: The most recent evaluation metrics keyed by their full metric name.
    """

    def __init__(self, progress_queue: Any, stream: TextIO | None = None) -> None:
        """Initializes the monitor thread over the shared progress queue.

        Args:
            progress_queue: The shared queue the training process streams progress messages to.
            stream: The output stream to render to, defaulting to the standard error stream.
        """
        super().__init__(
            progress_queue=progress_queue,
            preparing_label="preparing model...",
            stream=stream,
        )
        self._total_epochs = 0
        self._task = "pose"
        self._current_epoch = 0
        self._training_loss: float | None = None
        self._validation_loss: float | None = None
        self._metrics: dict[str, Any] = {}

    def __repr__(self) -> str:
        """Returns a string representation of the TrainingMonitor instance."""
        return f"TrainingMonitor(task={self._task}, epoch={self._current_epoch}/{self._total_epochs})"

    def _ingest(self, message: Any) -> bool:
        """Merges one ``config`` or ``metrics`` message into the retained training state.

        Args:
            message: A ``{"kind": "config", ...}`` or ``{"kind": "metrics", ...}`` message.

        Returns:
            True for a ``config`` message so the transition off the warm-up line is drawn immediately, False otherwise.
        """
        kind = message.get("kind")
        if kind == "config":
            self._total_epochs = message.get("epochs") or 0
            self._task = message.get("task") or "pose"
            return True
        if kind == "metrics":
            self._ingest_metrics(message)
        return False

    def _is_preparing(self) -> bool:
        """Returns whether no epoch budget has been reported yet, so the model is still preparing."""
        return self._total_epochs <= 0

    def _ingest_metrics(self, message: dict[str, Any]) -> None:
        """Merges one phase of epoch metrics into the retained state used for rendering.

        Args:
            message: A metrics message tagged with its epoch and training phase.
        """
        epoch = message.get("epoch")
        if epoch is not None:
            self._current_epoch = epoch
        metrics = message.get("metrics", {})
        if message.get("phase") == "evaluation":
            self._validation_loss = metrics.get("losses/eval.total_loss")
            self._metrics = {key: value for key, value in metrics.items() if key.startswith("metrics/")}
        else:
            self._training_loss = metrics.get("losses/train.total_loss")

    def _format_metrics(self) -> str:
        """Builds a compact string of the highest-priority evaluation metrics for the progress line.

        Returns:
            A space-separated string of up to ``_MAX_RENDERED_METRICS`` metric name-value pairs, or an empty string.
        """
        matched = []
        for key, value in self._metrics.items():
            if not key.startswith("metrics/test.") or not isinstance(value, (int, float)):
                continue
            short = key.removeprefix("metrics/test.")
            for priority, prefix in enumerate(_RENDERED_METRIC_PREFIXES):
                if short.startswith(prefix):
                    matched.append((priority, f"{short} {value:.1f}"))
                    break
        matched.sort(key=lambda item: item[0])
        return " ".join(part for _, part in matched[:_MAX_RENDERED_METRICS])

    def _compose_active(self, elapsed: float) -> str:
        """Builds the active line body from the retained epoch, losses, and evaluation metrics.

        Args:
            elapsed: The seconds elapsed since the renderer was constructed.

        Returns:
            The composed active line body.
        """
        bar, percent = self._bar(self._current_epoch / self._total_epochs)
        segments = [f"epoch {self._current_epoch}/{self._total_epochs}"]
        if self._training_loss is not None:
            segments.append(f"training {self._training_loss:.5f}")
        if self._validation_loss is not None and not math.isnan(self._validation_loss):
            segments.append(f"validation {self._validation_loss:.5f}")
        metric_text = self._format_metrics()
        if metric_text:
            segments.append(metric_text)
        eta = self._eta(done=self._current_epoch, total=self._total_epochs, elapsed=elapsed)
        return f"[{bar}] {percent:5.1f}% | {' | '.join(segments)} | {format_duration(elapsed)} | ETA {eta}"
