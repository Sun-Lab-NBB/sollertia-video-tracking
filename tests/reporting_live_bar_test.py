"""Tests for the shared live progress-bar base that the training, inference, and frame-extraction bars build on."""

import sys
import time
from queue import Empty, Queue

import pytest

from sollertia_video_tracking.reporting.live_bar import (
    _STOP_SENTINEL,
    _SPINNER_FRAMES,
    _NON_TTY_RENDER_INTERVAL,
    LiveBar,
    format_duration,
)


class _FakeStream:
    """A minimal text stream that records writes and reports a configurable interactive-terminal status."""

    def __init__(self, *, is_tty: bool) -> None:
        self._is_tty = is_tty
        self.chunks: list[str] = []
        self.flush_count = 0

    def isatty(self) -> bool:
        return self._is_tty

    def write(self, text: str) -> None:
        self.chunks.append(text)

    def flush(self) -> None:
        self.flush_count += 1

    @property
    def text(self) -> str:
        return "".join(self.chunks)


class _ScriptedQueue:
    """A fake progress queue whose ``get`` replays a scripted sequence, raising ``Empty`` where the script says so."""

    def __init__(self, script: list) -> None:
        # Each item is either a real message, the stop sentinel, or the ``Empty`` class to raise a timeout.
        self._script = list(script)
        self.put_items: list = []

    def get(self, **_kwargs: object) -> object:
        item = self._script.pop(0)
        if item is Empty:
            raise Empty
        return item

    def put(self, item: object) -> None:
        self.put_items.append(item)


class _FakeBar(LiveBar):
    """A concrete ``LiveBar`` that supplies the three subclass hooks so the shared scaffolding can be exercised."""

    def __init__(self, *args, preparing: bool = False, force_value: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._preparing = preparing
        self._force_value = force_value
        self.ingested: list = []

    def _ingest(self, message) -> bool:
        self.ingested.append(message)
        return self._force_value

    def _is_preparing(self) -> bool:
        return self._preparing

    def _compose_active(self, elapsed: float) -> str:
        return f"active {format_duration(elapsed)}"


# --- format_duration -------------------------------------------------------------------------------------------------


def test_format_duration_minutes_and_seconds() -> None:
    """Under an hour, the duration is rendered as zero-padded MM:SS."""
    assert format_duration(65) == "01:05"


def test_format_duration_spans_an_hour() -> None:
    """At or past an hour, the hours component is prepended as H:MM:SS."""
    assert format_duration(3661) == "1:01:01"


def test_format_duration_clamps_negative_to_zero() -> None:
    """A negative duration is clamped to zero so the bar never shows a negative clock."""
    assert format_duration(-42) == "00:00"


# --- construction ----------------------------------------------------------------------------------------------------


def test_init_defaults_to_stderr_stream() -> None:
    """With no explicit stream, the renderer writes to the standard error stream."""
    bar = _FakeBar(Queue())
    assert bar._stream is sys.stderr
    assert bar.daemon is True
    assert bar._spinner_index == 0
    assert bar._last_render_time == 0.0


def test_init_uses_explicit_stream_and_reads_tty_status() -> None:
    """An explicit stream is retained and its interactive-terminal status is captured at construction."""
    stream = _FakeStream(is_tty=True)
    bar = _FakeBar(Queue(), stream=stream, width=12, preparing_label="warming...")
    assert bar._stream is stream
    assert bar._is_tty is True
    assert bar._width == 12
    assert bar._preparing_label == "warming..."


def test_repr_reports_width_and_tty_status() -> None:
    """The repr surfaces the configured bar width and the captured interactive-terminal status."""
    bar = _FakeBar(Queue(), stream=_FakeStream(is_tty=True), width=17)
    assert repr(bar) == "LiveBar(width=17, is_tty=True)"


def test_base_class_hooks_are_unimplemented() -> None:
    """The base class leaves the three subclass hooks unimplemented, so a subclass must supply real behavior."""
    base = LiveBar(Queue(), stream=_FakeStream(is_tty=False))
    with pytest.raises(NotImplementedError):
        base._ingest("msg")
    with pytest.raises(NotImplementedError):
        base._is_preparing()
    with pytest.raises(NotImplementedError):
        base._compose_active(1.0)


# --- _bar ------------------------------------------------------------------------------------------------------------


def test_bar_partial_fill() -> None:
    """A mid-range fraction fills the proportional number of glyphs and reports the matching percent."""
    bar = _FakeBar(Queue(), stream=_FakeStream(is_tty=False), width=10)
    glyphs, percent = bar._bar(0.5)
    assert glyphs == "#####-----"
    assert percent == 50.0


def test_bar_clamps_above_one() -> None:
    """A fraction above one is clamped so the bar never overflows its width."""
    bar = _FakeBar(Queue(), stream=_FakeStream(is_tty=False), width=8)
    glyphs, percent = bar._bar(1.5)
    assert glyphs == "########"
    assert percent == 100.0


def test_bar_clamps_below_zero() -> None:
    """A negative fraction is clamped to an empty bar at zero percent."""
    bar = _FakeBar(Queue(), stream=_FakeStream(is_tty=False), width=6)
    glyphs, percent = bar._bar(-0.5)
    assert glyphs == "------"
    assert percent == 0.0


# --- _eta ------------------------------------------------------------------------------------------------------------


def test_eta_completed_shows_zero() -> None:
    """Once the work is done, the ETA collapses to a plain zero clock."""
    bar = _FakeBar(Queue(), stream=_FakeStream(is_tty=False))
    assert bar._eta(done=10, total=10, elapsed=5.0) == "00:00"


def test_eta_projects_remaining_from_rate() -> None:
    """With measurable throughput, the ETA projects the remaining time from the observed rate."""
    bar = _FakeBar(Queue(), stream=_FakeStream(is_tty=False))
    # rate = 5 / 5 = 1 unit/sec; remaining = (10 - 5) / 1 = 5 seconds.
    assert bar._eta(done=5, total=10, elapsed=5.0) == "00:05"


def test_eta_no_throughput_shows_placeholder() -> None:
    """Before any work is completed, the ETA stays a placeholder rather than a false zero."""
    bar = _FakeBar(Queue(), stream=_FakeStream(is_tty=False))
    assert bar._eta(done=0, total=10, elapsed=5.0) == "--:--"


def test_eta_zero_elapsed_shows_placeholder() -> None:
    """With no elapsed time yet, the rate is unknown and the ETA is a placeholder (the elapsed<=0 branch)."""
    bar = _FakeBar(Queue(), stream=_FakeStream(is_tty=False))
    assert bar._eta(done=5, total=10, elapsed=0.0) == "--:--"


# --- _compose_preparing ----------------------------------------------------------------------------------------------


def test_compose_preparing_default_body() -> None:
    """The default warm-up body shows an empty bar, the label, and the elapsed clock."""
    bar = _FakeBar(Queue(), stream=_FakeStream(is_tty=False), width=5, preparing_label="preparing...")
    assert bar._compose_preparing(72.0) == "[-----] preparing... | 01:12 elapsed"


# --- _render ---------------------------------------------------------------------------------------------------------


def test_render_forced_tty_redraws_in_place() -> None:
    """A forced render on a terminal rewrites the line in place with a carriage return and clear-to-end code."""
    stream = _FakeStream(is_tty=True)
    bar = _FakeBar(Queue(), stream=stream, preparing=False)
    bar._render(force=True)
    written = stream.text
    assert written.startswith("\r")
    assert written.endswith("\033[K")
    # The first drawn line uses the first spinner glyph, and the spinner index advances.
    assert _SPINNER_FRAMES[0] in written
    assert "active" in written
    assert bar._spinner_index == 1
    assert stream.flush_count == 1


def test_render_non_tty_appends_new_line() -> None:
    """A forced render to a non-terminal appends a fresh line rather than redrawing in place."""
    stream = _FakeStream(is_tty=False)
    bar = _FakeBar(Queue(), stream=stream, preparing=False)
    bar._render(force=True)
    written = stream.text
    assert written.endswith("\n")
    assert "\r" not in written
    assert written.startswith(f"{_SPINNER_FRAMES[0]} active")


def test_render_uses_preparing_body_before_work_starts() -> None:
    """While still warming up, the render draws the preparing body instead of the active one."""
    stream = _FakeStream(is_tty=False)
    bar = _FakeBar(Queue(), stream=stream, preparing=True, preparing_label="preparing...")
    bar._render(force=True)
    assert "preparing..." in stream.text
    assert "elapsed" in stream.text


def test_render_respects_minimum_interval() -> None:
    """An unforced render inside the minimum interval is suppressed and nothing is written."""
    stream = _FakeStream(is_tty=False)
    bar = _FakeBar(Queue(), stream=stream, preparing=False)
    # A render that just happened means a follow-up unforced render is within the interval and must be skipped.
    bar._last_render_time = time.monotonic()
    bar._render(force=False)
    assert stream.chunks == []
    assert bar._spinner_index == 0


def test_render_spinner_cycles_across_draws() -> None:
    """Successive drawn lines cycle through the spinner frames to signal liveness while the counter is static."""
    stream = _FakeStream(is_tty=False)
    bar = _FakeBar(Queue(), stream=stream, preparing=False)
    for _ in range(len(_SPINNER_FRAMES) + 1):
        bar._render(force=True)
    drawn_spinners = [chunk.split(" ", 1)[0] for chunk in stream.chunks]
    # The first len(frames) draws walk the whole cycle, then it wraps back to the first frame.
    assert drawn_spinners[: len(_SPINNER_FRAMES)] == list(_SPINNER_FRAMES)
    assert drawn_spinners[len(_SPINNER_FRAMES)] == _SPINNER_FRAMES[0]


# --- stop ------------------------------------------------------------------------------------------------------------


def test_stop_places_sentinel_on_queue() -> None:
    """Calling stop enqueues the shared terminal sentinel for the run loop to recognize."""
    queue: Queue = Queue()
    bar = _FakeBar(queue, stream=_FakeStream(is_tty=False))
    bar.stop()
    assert queue.get_nowait() == _STOP_SENTINEL


# --- run -------------------------------------------------------------------------------------------------------------


def test_run_non_tty_handles_empty_message_and_stop() -> None:
    """The non-terminal run loop renders on an idle timeout, ingests real messages, and exits on the sentinel."""
    stream = _FakeStream(is_tty=False)
    # First an idle timeout (renders), then one real message, then the terminal sentinel.
    scripted = _ScriptedQueue([Empty, "msg-1", _STOP_SENTINEL])
    bar = _FakeBar(scripted, stream=stream, preparing=False, force_value=False)
    # In production the very first render always fires because the monotonic clock is far past the initial
    # ``_last_render_time`` of 0.0. Pin that here so the idle-timeout render is deterministic and does not silently
    # depend on the host's uptime being at least one non-tty render interval.
    bar._last_render_time = -1000.0
    bar.run()
    # The real message was ingested exactly once.
    assert bar.ingested == ["msg-1"]
    # The idle-timeout render and the forced final render both wrote a non-terminal line; the in-loop unforced render
    # after ingest fell inside the long non-tty interval and was suppressed, so exactly two lines were drawn.
    assert stream.text.count("\n") == 2
    # A non-terminal stream is never sent the trailing terminal newline sequence.
    assert "\r" not in stream.text


def test_run_tty_draws_final_frame_and_trailing_newline() -> None:
    """The terminal run loop ingests messages, draws a forced final frame, and closes with a trailing newline."""
    stream = _FakeStream(is_tty=True)
    scripted = _ScriptedQueue(["msg-a", _STOP_SENTINEL])
    bar = _FakeBar(scripted, stream=stream, preparing=False, force_value=True)
    bar.run()
    assert bar.ingested == ["msg-a"]
    # A terminal run ends by writing a bare newline after the final in-place frame.
    assert stream.chunks[-1] == "\n"
    # Exactly two in-place frames are drawn: the one triggered by ingesting the message and the forced final frame.
    # Requiring both (rather than merely "some \\r chunk exists") is what actually pins the named final-frame render;
    # a single frame would satisfy a weaker check even if the final forced render were dropped.
    render_frames = [chunk for chunk in stream.chunks if chunk.startswith("\r")]
    assert len(render_frames) == 2
    assert all(chunk.endswith("\033[K") for chunk in render_frames)


def test_non_tty_render_interval_constant_is_slow_cadence() -> None:
    """The non-terminal render cadence is far slower than the terminal one, keeping redirected logs greppable."""
    # Guards the constant the non-tty suppression above relies on staying well above the in-loop poll cadence.
    assert _NON_TTY_RENDER_INTERVAL == 30.0
