"""Provides the clean training progress monitor that replaces DeepLabCut's per-iteration terminal logging stream."""

import sys
import math
import time
from queue import Empty
from typing import Any, TextIO
from threading import Thread
import contextlib

from deeplabcut.pose_estimation_pytorch.runners.logger import BaseLogger

_PROGRESS_BAR_WIDTH: int = 30
"""The width, in characters, of the rendered training progress bar."""

_MAX_RENDERED_METRICS: int = 3
"""The maximum number of evaluation metrics shown on the progress line to keep it compact."""

_RENDERED_METRIC_PREFIXES: tuple[str, ...] = ("mAP", "mAR", "rmse")
"""The evaluation-metric name prefixes shown on the progress line, in priority order."""


class QueueTrainingLogger(BaseLogger):
    """Streams per-epoch training metrics to a monitor process over a shared queue.

    Notes:
        This is a DeepLabCut ``BaseLogger`` built directly on the rank-0 training process and attached to the runner.
        It receives one call per training phase per epoch (train every epoch, evaluation on evaluation epochs) and
        forwards each as a small JSON-serializable message. A dropped message only skips a redraw, so a full or
        broken queue never disrupts training.
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


class TrainingMonitor(Thread):
    """Renders a single live training progress bar from the metrics streamed by the rank-0 training process.

    Notes:
        The renderer consumes ``{"kind": "config", ...}`` and ``{"kind": "metrics", ...}`` messages plus a terminal
        ``{"kind": "stop"}`` sentinel. On a TTY the bar updates in place with carriage returns; when the output is
        redirected, it prints at most one line every ``heartbeat`` seconds. Because evaluation epochs deliver a train
        message and an evaluation message at the same epoch, the latest of each is retained and merged onto one line.

    Attributes:
        _progress_queue: The shared queue the training process streams progress messages to.
        _heartbeat: The minimum interval, in seconds, between rendered lines when the output is not a TTY.
        _width: The width, in characters, of the rendered bar.
        _stream: The output stream the bar renders to.
        _is_tty: True when the output stream is an interactive terminal.
        _total_epochs: The total number of epochs the run trains for, once reported by the configuration message.
        _task: The training task label reported by the configuration message.
        _current_epoch: The most recent epoch reported by the training process.
        _training_loss: The most recent training loss, or None before the first epoch completes.
        _validation_loss: The most recent validation loss, or None before the first evaluation epoch.
        _metrics: The most recent evaluation metrics keyed by their full metric name.
        _start_time: The monotonic timestamp captured when the renderer was constructed.
        _last_render_time: The monotonic timestamp of the most recent render.
    """

    def __init__(
        self,
        progress_queue: Any,
        heartbeat: float,
        stream: TextIO | None = None,
        width: int = _PROGRESS_BAR_WIDTH,
    ) -> None:
        """Initializes the monitor thread over the shared progress queue.

        Args:
            progress_queue: The shared queue the training process streams progress messages to.
            heartbeat: The minimum interval, in seconds, between rendered lines when the output is not a TTY.
            stream: The output stream to render to, defaulting to the standard error stream.
            width: The width, in characters, of the rendered bar.
        """
        super().__init__(daemon=True)
        self._progress_queue = progress_queue
        self._heartbeat = heartbeat
        self._width = width
        self._stream = stream if stream is not None else sys.stderr
        self._is_tty = self._stream.isatty()
        self._total_epochs = 0
        self._task = "pose"
        self._current_epoch = 0
        self._training_loss: float | None = None
        self._validation_loss: float | None = None
        self._metrics: dict[str, Any] = {}
        self._start_time = time.monotonic()
        self._last_render_time = 0.0

    def __repr__(self) -> str:
        """Returns a string representation of the TrainingMonitor instance."""
        return f"TrainingMonitor(task={self._task}, epoch={self._current_epoch}/{self._total_epochs})"

    def run(self) -> None:
        """Consumes queue messages and re-renders the bar until a ``{"kind": "stop"}`` sentinel arrives."""
        while True:
            try:
                message = self._progress_queue.get(timeout=0.2 if self._is_tty else 1.0)
            except Empty:
                self._render()
                continue
            kind = message.get("kind")
            if kind == "config":
                self._total_epochs = message.get("epochs") or 0
                self._task = message.get("task") or "pose"
                self._render(force=True)
            elif kind == "metrics":
                self._ingest_metrics(message)
                self._render()
            elif kind == "stop":
                break
        self._render(force=True)
        if self._is_tty:
            self._stream.write("\n")
            self._stream.flush()

    def stop(self) -> None:
        """Signals the renderer to draw a final frame and exit."""
        self._progress_queue.put({"kind": "stop"})

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

    def _render(self, *, force: bool = False) -> None:
        """Draws the bar, honoring the per-mode minimum render interval unless ``force`` is set.

        Args:
            force: Determines whether to render immediately, bypassing the minimum interval between renders.
        """
        now = time.monotonic()
        interval = 0.2 if self._is_tty else max(1.0, self._heartbeat)
        if not force and (now - self._last_render_time) < interval:
            return
        self._last_render_time = now

        elapsed = now - self._start_time

        if self._total_epochs <= 0:
            # No epoch budget has been reported yet, so the workers are still preparing: initializing the process
            # group, downloading pretrained weights, building the model, and any torch.compile warm-up. That can run
            # for minutes on a first run, so show elapsed time rather than a static empty bar that looks stalled.
            message = f"[{'-' * self._width}] preparing model... | {_format_duration(elapsed)} elapsed"
            if self._is_tty:
                self._stream.write("\r" + message + "\033[K")
            else:
                self._stream.write(message + "\n")
            self._stream.flush()
            return

        fraction = min(1.0, self._current_epoch / self._total_epochs)
        percent = 100.0 * fraction
        filled = int(self._width * fraction)
        bar = "#" * filled + "-" * (self._width - filled)
        rate = self._current_epoch / elapsed if elapsed > 0 else 0.0
        remaining = (self._total_epochs - self._current_epoch) / rate if rate > 0 and fraction < 1.0 else 0.0

        segments = [f"epoch {self._current_epoch}/{self._total_epochs}"]
        if self._training_loss is not None:
            segments.append(f"training {self._training_loss:.5f}")
        if self._validation_loss is not None and not math.isnan(self._validation_loss):
            segments.append(f"validation {self._validation_loss:.5f}")
        metric_text = self._format_metrics()
        if metric_text:
            segments.append(metric_text)

        message = (
            f"[{bar}] {percent:5.1f}% | {' | '.join(segments)} | "
            f"{_format_duration(elapsed)} | ETA {_format_duration(remaining)}"
        )
        if self._is_tty:
            self._stream.write("\r" + message + "\033[K")
        else:
            self._stream.write(message + "\n")
        self._stream.flush()


def _format_duration(seconds: float) -> str:
    """Formats a duration as ``MM:SS``, or as ``H:MM:SS`` when it spans an hour or more.

    Args:
        seconds: The duration to format, in seconds.

    Returns:
        The formatted duration string.
    """
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{whole_seconds:02d}" if hours else f"{minutes:02d}:{whole_seconds:02d}"
