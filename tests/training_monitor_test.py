"""Tests for the training progress monitor and its rank-0 queue logger (training/monitor.py).

These drive the logger and renderer directly with tiny in-memory messages; no queue thread, DLC runtime, or GPU is
started. The renderer's compose/ingest helpers are exercised in isolation so every line is covered deterministically.
"""

import io
import math

from sollertia_video_tracking.training.monitor import (
    _MAX_RENDERED_METRICS,
    TrainingMonitor,
    QueueTrainingLogger,
)


class _RecordingQueue:
    """A minimal queue stand-in that records every message put on it via put_nowait."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def put_nowait(self, message: object) -> None:
        self.messages.append(message)


class _RaisingQueue:
    """A queue stand-in whose put_nowait always raises, to drive the suppressed-error path."""

    def __init__(self) -> None:
        self.calls = 0

    def put_nowait(self, *_: object) -> None:
        self.calls += 1
        raise RuntimeError


def _make_monitor() -> TrainingMonitor:
    """Builds a TrainingMonitor over a recording queue and a non-tty StringIO stream (no thread started)."""
    return TrainingMonitor(progress_queue=_RecordingQueue(), stream=io.StringIO())


# QueueTrainingLogger


def test_logger_init_stores_queue_and_task() -> None:
    """The logger retains the shared queue and the task label supplied at construction."""
    queue = _RecordingQueue()
    logger = QueueTrainingLogger(queue, task_name="detector")
    assert logger._progress_queue is queue
    assert logger._task_name == "detector"


def test_logger_default_task_name_is_pose() -> None:
    """The task label defaults to ``pose`` when not supplied."""
    logger = QueueTrainingLogger(_RecordingQueue())
    assert logger._task_name == "pose"


def test_log_config_none_puts_nothing() -> None:
    """A None config short-circuits without forwarding any message."""
    queue = _RecordingQueue()
    logger = QueueTrainingLogger(queue)
    logger.log_config(None)
    assert queue.messages == []


def test_log_config_forwards_epoch_budget() -> None:
    """A config with train_settings forwards the epoch budget and task under the ``config`` kind."""
    queue = _RecordingQueue()
    logger = QueueTrainingLogger(queue, task_name="detector")
    logger.log_config({"train_settings": {"epochs": 120}})
    assert queue.messages == [{"kind": "config", "epochs": 120, "task": "detector"}]


def test_log_config_missing_train_settings_forwards_none_epochs() -> None:
    """A config without train_settings still forwards a config message with a None epoch budget."""
    queue = _RecordingQueue()
    logger = QueueTrainingLogger(queue)
    logger.log_config({})
    assert queue.messages == [{"kind": "config", "epochs": None, "task": "pose"}]


def test_log_tags_evaluation_phase_from_eval_loss_key() -> None:
    """A metrics dict carrying an eval-loss key is tagged as the evaluation phase."""
    queue = _RecordingQueue()
    logger = QueueTrainingLogger(queue)
    metrics = {"losses/eval.total_loss": 0.4}
    logger.log(metrics, step=7)
    assert queue.messages == [{"kind": "metrics", "epoch": 7, "phase": "evaluation", "metrics": metrics}]


def test_log_tags_evaluation_phase_from_metrics_key() -> None:
    """A metrics dict carrying a ``metrics/`` key is tagged as the evaluation phase."""
    queue = _RecordingQueue()
    logger = QueueTrainingLogger(queue)
    logger.log({"metrics/test.mAP": 0.9}, step=3)
    assert queue.messages[0]["phase"] == "evaluation"


def test_log_tags_training_phase_when_no_eval_keys() -> None:
    """A metrics dict with only train losses is tagged as the training phase and copies the metrics dict."""
    queue = _RecordingQueue()
    logger = QueueTrainingLogger(queue)
    original = {"losses/train.total_loss": 1.5}
    logger.log(original, step=2)
    message = queue.messages[0]
    assert message["phase"] == "training"
    # The forwarded metrics are a shallow copy, not the caller's dict.
    assert message["metrics"] == original
    assert message["metrics"] is not original


def test_log_allows_none_step() -> None:
    """A None step is forwarded verbatim as the epoch tag."""
    queue = _RecordingQueue()
    logger = QueueTrainingLogger(queue)
    logger.log({"losses/train.total_loss": 1.0})
    assert queue.messages[0]["epoch"] is None


def test_save_is_a_noop() -> None:
    """save returns None and forwards nothing, since metrics stream as they arrive."""
    queue = _RecordingQueue()
    logger = QueueTrainingLogger(queue)
    assert logger.save() is None
    assert queue.messages == []


def test_put_forwards_message_on_healthy_queue() -> None:
    """_put places the message on a healthy queue via put_nowait."""
    queue = _RecordingQueue()
    logger = QueueTrainingLogger(queue)
    logger._put({"kind": "custom"})
    assert queue.messages == [{"kind": "custom"}]


def test_put_suppresses_queue_errors() -> None:
    """A queue error inside _put is suppressed after put_nowait is attempted, so training is never disrupted."""
    queue = _RaisingQueue()
    logger = QueueTrainingLogger(queue)
    # Must not raise even though put_nowait raises...
    logger._put({"kind": "custom"})
    # ...and the failure was swallowed from the real put_nowait attempt, not from _put skipping the queue entirely.
    assert queue.calls == 1


# TrainingMonitor construction and ingest


def test_monitor_init_defaults() -> None:
    """The monitor starts with an unknown epoch budget and empty retained metric state."""
    monitor = _make_monitor()
    assert monitor._total_epochs == 0
    assert monitor._task == "pose"
    assert monitor._current_epoch == 0
    assert monitor._training_loss is None
    assert monitor._validation_loss is None
    assert monitor._metrics == {}
    assert monitor._preparing_label == "preparing model..."


def test_ingest_config_sets_budget_and_forces_redraw() -> None:
    """A config message records the epoch budget and task and returns True to force an immediate redraw."""
    monitor = _make_monitor()
    forced = monitor._ingest({"kind": "config", "epochs": 200, "task": "detector"})
    assert forced is True
    assert monitor._total_epochs == 200
    assert monitor._task == "detector"


def test_ingest_config_falls_back_to_defaults() -> None:
    """A config with a falsy epoch budget and missing task falls back to zero epochs and the ``pose`` task."""
    monitor = _make_monitor()
    forced = monitor._ingest({"kind": "config", "epochs": 0})
    assert forced is True
    assert monitor._total_epochs == 0
    assert monitor._task == "pose"


def test_ingest_metrics_returns_false_and_merges() -> None:
    """A metrics message merges into retained state and returns False (no forced redraw)."""
    monitor = _make_monitor()
    forced = monitor._ingest(
        {"kind": "metrics", "epoch": 5, "phase": "training", "metrics": {"losses/train.total_loss": 0.8}},
    )
    assert forced is False
    assert monitor._current_epoch == 5
    assert monitor._training_loss == 0.8


def test_ingest_unknown_kind_returns_false() -> None:
    """An unrecognized message kind leaves state untouched and returns False."""
    monitor = _make_monitor()
    forced = monitor._ingest({"kind": "heartbeat"})
    assert forced is False
    assert monitor._current_epoch == 0


def test_is_preparing_reflects_epoch_budget() -> None:
    """The monitor reports preparing until an epoch budget has been reported."""
    monitor = _make_monitor()
    assert monitor._is_preparing() is True
    monitor._total_epochs = 50
    assert monitor._is_preparing() is False


# TrainingMonitor metrics ingestion


def test_ingest_metrics_evaluation_sets_validation_and_filters_metrics() -> None:
    """An evaluation phase records the validation loss and keeps only ``metrics/`` keys in retained metrics."""
    monitor = _make_monitor()
    monitor._ingest_metrics(
        {
            "epoch": 9,
            "phase": "evaluation",
            "metrics": {
                "losses/eval.total_loss": 0.25,
                "metrics/test.mAP": 0.9,
                "not_a_metric": 123,
            },
        },
    )
    assert monitor._current_epoch == 9
    assert monitor._validation_loss == 0.25
    assert monitor._metrics == {"metrics/test.mAP": 0.9}


def test_ingest_metrics_training_sets_training_loss_only() -> None:
    """A training phase records the training loss and leaves validation loss and metrics untouched."""
    monitor = _make_monitor()
    monitor._ingest_metrics(
        {"epoch": 4, "phase": "training", "metrics": {"losses/train.total_loss": 1.1}},
    )
    assert monitor._training_loss == 1.1
    assert monitor._validation_loss is None
    assert monitor._metrics == {}


def test_ingest_metrics_none_epoch_keeps_current_epoch() -> None:
    """A metrics message without an epoch tag leaves the current epoch unchanged."""
    monitor = _make_monitor()
    monitor._current_epoch = 12
    monitor._ingest_metrics({"phase": "training", "metrics": {"losses/train.total_loss": 0.5}})
    assert monitor._current_epoch == 12


def test_ingest_metrics_defaults_empty_metrics() -> None:
    """A metrics message without a metrics payload defaults to an empty dict, running the branch without error."""
    monitor = _make_monitor()
    monitor._training_loss = 9.9  # A stale value the training branch must overwrite from the empty-dict default.
    monitor._ingest_metrics({"epoch": 1, "phase": "training"})
    # The method ran to completion: the epoch advanced and the loss was read from the empty default (-> None),
    # which also proves the missing "metrics" key defaulted to {} rather than raising on a None payload.
    assert monitor._current_epoch == 1
    assert monitor._training_loss is None


# TrainingMonitor metric formatting


def test_format_metrics_empty_when_no_metrics() -> None:
    """With no retained metrics the formatted string is empty."""
    monitor = _make_monitor()
    assert monitor._format_metrics() == ""


def test_format_metrics_orders_by_priority_and_skips_non_matching() -> None:
    """Metrics render in prefix-priority order, skipping non-test keys, non-numeric values, and unknown prefixes."""
    monitor = _make_monitor()
    monitor._metrics = {
        "metrics/test.mAR": 0.44,  # priority 1, inserted before the priority-0 entry to prove sorting.
        "metrics/test.mAP": 0.91,  # priority 0
        "metrics/test.rmse": 3.14,  # priority 2
        "metrics/test.precision": 0.7,  # matches metrics/test. but no known prefix -> not rendered.
        "metrics/other.mAP": 0.5,  # not a metrics/test. key -> skipped.
        "metrics/test.mAP_string": "high",  # non-numeric value -> skipped.
    }
    # Priority order mAP (0), mAR (1), rmse (2); values formatted to one decimal place.
    assert monitor._format_metrics() == "mAP 0.9 mAR 0.4 rmse 3.1"


def test_format_metrics_truncates_to_max_rendered() -> None:
    """No more than the configured maximum number of metric pairs are rendered."""
    monitor = _make_monitor()
    # Four keys all sharing the highest-priority ``mAP`` prefix; only _MAX_RENDERED_METRICS survive.
    monitor._metrics = {
        "metrics/test.mAP": 0.1,
        "metrics/test.mAP_50": 0.2,
        "metrics/test.mAP_75": 0.3,
        "metrics/test.mAP_large": 0.4,
    }
    rendered = monitor._format_metrics()
    assert rendered.count("mAP") == _MAX_RENDERED_METRICS


# TrainingMonitor active line composition


def test_compose_active_full_line() -> None:
    """A fully populated state composes epoch, both losses, metrics, elapsed, and an ETA onto one line."""
    monitor = _make_monitor()
    monitor._total_epochs = 100
    monitor._current_epoch = 50
    monitor._training_loss = 0.12345
    monitor._validation_loss = 0.54321
    monitor._metrics = {"metrics/test.mAP": 0.9}
    line = monitor._compose_active(elapsed=60.0)
    assert "epoch 50/100" in line
    assert "training 0.12345" in line
    assert "validation 0.54321" in line
    assert "mAP 0.9" in line
    assert " 50.0% " in line
    assert "ETA" in line


def test_compose_active_minimal_line_omits_optional_segments() -> None:
    """With no losses and no metrics only the epoch segment (plus elapsed and ETA) is rendered."""
    monitor = _make_monitor()
    monitor._total_epochs = 10
    monitor._current_epoch = 1
    line = monitor._compose_active(elapsed=5.0)
    assert "epoch 1/10" in line
    assert "training" not in line
    assert "validation" not in line


def test_compose_active_skips_nan_validation_loss() -> None:
    """A NaN validation loss is treated as absent and omitted from the composed line."""
    monitor = _make_monitor()
    monitor._total_epochs = 10
    monitor._current_epoch = 2
    monitor._validation_loss = math.nan
    line = monitor._compose_active(elapsed=5.0)
    assert "validation" not in line


def test_compose_active_completion_shows_zero_eta() -> None:
    """At completion the bar reads full and the ETA collapses to a zero duration."""
    monitor = _make_monitor()
    monitor._total_epochs = 10
    monitor._current_epoch = 10
    line = monitor._compose_active(elapsed=120.0)
    assert "100.0%" in line
    assert "ETA 00:00" in line


def test_repr_reports_task_and_epoch_progress() -> None:
    """The repr summarizes the task and epoch progress (exercised for completeness)."""
    monitor = _make_monitor()
    monitor._total_epochs = 100
    monitor._current_epoch = 3
    assert repr(monitor) == "TrainingMonitor(task=pose, epoch=3/100)"
