"""Provides the shared live progress-bar base the training, inference, and frame-extraction progress bars build on."""

import sys
import time
from queue import Empty
from typing import Any, TextIO
from threading import Thread

_PROGRESS_BAR_WIDTH: int = 30
"""The default width, in characters, of the rendered progress bar."""

_SPINNER_FRAMES: str = "|/-\\"
"""The characters cycled through on each drawn line to show the bar is still alive while its counter is static. The
static stretches include model warm-up before the first unit of work is reported and a long decode before the first
frame stride."""

_STOP_SENTINEL: tuple[str] = ("__live_bar_stop__",)
"""The canonical terminal message ``stop`` places on the queue. It is distinct from every real progress message, so the
shared run loop recognizes termination regardless of each subclass's own message protocol."""

_TTY_RENDER_INTERVAL: float = 0.2
"""The minimum interval, in seconds, between rendered lines on an interactive terminal, where the bar redraws in place
at a fast live cadence."""

_NON_TTY_RENDER_INTERVAL: float = 30.0
"""The minimum interval, in seconds, between rendered lines when the output is not an interactive terminal. A forced
redraw, such as a completion, bypasses it. Redirected output cannot redraw a line in place, so each render appends a
whole new line. This steady cadence keeps such logs greppable without flooding them."""

_NON_TTY_POLL_INTERVAL: float = 1.0
"""The interval, in seconds, the non-interactive run loop waits for a message before waking to re-check whether a
render is due. It stays well below the render interval so a due render waits at most one poll."""


def format_duration(seconds: float) -> str:
    """Formats a duration as ``MM:SS``, or as ``H:MM:SS`` when it spans an hour or more.

    Args:
        seconds: The duration to format, in seconds.

    Returns:
        The duration rendered as ``MM:SS``, or ``H:MM:SS`` past an hour.
    """
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{whole_seconds:02d}" if hours else f"{minutes:02d}:{whole_seconds:02d}"


class LiveBar(Thread):
    """Renders a single in-place live progress bar from messages streamed over a shared queue.

    Notes:
        This base owns the rendering scaffolding every progress bar in the library shares: interactive-terminal
        detection, the message-consuming run loop, and the per-mode minimum render interval. It also owns the liveness
        spinner advanced on every drawn line, the elapsed clock, the warm-up ``preparing`` line shown before the first
        unit of work is reported, and the in-place carriage-return versus one-line-per-update rendering. It breaks the
        run loop on the shared ``stop`` sentinel, so a subclass never handles termination itself.

        Subclasses supply only their own behavior: ``_ingest`` merges one message and reports whether to force an
        immediate redraw, ``_is_preparing`` reports whether work has started, and ``_compose_active`` builds the line
        body once it has. The shared ``_bar`` and ``_eta`` helpers keep the bar glyphs and the honest ETA consistent
        across every subclass, and ``_compose_preparing`` may be overridden to add warm-up context such as a count.

    Attributes:
        _progress_queue: The shared queue the producer streams progress messages to.
        _preparing_label: The text shown during warm-up, before the first unit of work is reported.
        _width: The width, in characters, of the rendered bar.
        _stream: The output stream the bar renders to.
        _is_tty: Determines whether the output stream is an interactive terminal.
        _start_time: The monotonic timestamp captured when the renderer was constructed.
        _last_render_time: The monotonic timestamp of the most recent render.
        _last_progress_time: The monotonic timestamp of the most recent real progress message.
        _spinner_index: The number of lines drawn so far, used to advance the liveness spinner on each drawn line.
    """

    def __init__(
        self,
        progress_queue: Any,
        *,
        preparing_label: str = "preparing...",
        stream: TextIO | None = None,
        width: int = _PROGRESS_BAR_WIDTH,
    ) -> None:
        """Initializes the renderer thread.

        Args:
            progress_queue: The shared queue the producer streams progress messages to.
            preparing_label: The text shown during warm-up, before the first unit of work is reported.
            stream: The output stream to render to, defaulting to the standard error stream.
            width: The width, in characters, of the rendered bar.
        """
        super().__init__(daemon=True)
        self._progress_queue = progress_queue
        self._preparing_label = preparing_label
        self._width = width
        self._stream = stream if stream is not None else sys.stderr
        self._is_tty = self._stream.isatty()
        self._start_time = time.monotonic()
        self._last_render_time = 0.0
        self._last_progress_time = self._start_time
        self._spinner_index = 0

    def __repr__(self) -> str:
        """Returns a string representation of the LiveBar instance."""
        return f"LiveBar(width={self._width}, is_tty={self._is_tty})"

    def seconds_since_progress(self) -> float:
        """Returns the seconds elapsed since the producers last reported real progress.

        A rendered line proves only that the renderer is alive, so the elapsed silence measured here is what
        distinguishes a run whose workers are still working from one whose workers have wedged.
        """
        return time.monotonic() - self._last_progress_time

    def run(self) -> None:
        """Consumes queue messages and re-renders the bar until the shared ``stop`` sentinel arrives."""
        poll_timeout = _TTY_RENDER_INTERVAL if self._is_tty else _NON_TTY_POLL_INTERVAL
        try:
            while True:
                try:
                    message = self._progress_queue.get(timeout=poll_timeout)
                except Empty:
                    self._render()
                    continue
                if message == _STOP_SENTINEL:
                    break
                # Only a real message counts as progress. The stop sentinel marks the end of the run rather than work
                # having been done, so it must not refresh the staleness the supervisor watches.
                self._last_progress_time = time.monotonic()
                force = self._ingest(message)
                self._render(force=force)
            self._render(force=True)
            if self._is_tty:
                self._stream.write("\n")
                self._stream.flush()
        except (OSError, EOFError, ValueError):
            # The queue's manager, or the stream this bar renders through, was torn down while this thread was still
            # running. Ending the thread here keeps that teardown race from reaching the interpreter as an unraisable
            # exception, which reports as a bare object dump that hides whatever actually ended the run.
            return

    def stop(self) -> None:
        """Signals the renderer to draw a final frame and exit."""
        self._progress_queue.put(_STOP_SENTINEL)

    def _ingest(self, message: Any) -> bool:
        """Merges one progress message into the renderer's retained state.

        Args:
            message: A subclass-specific progress message pulled from the shared queue.

        Returns:
            True to force an immediate redraw of the resulting state (for example a completion or a phase change),
            bypassing the minimum render interval.
        """
        raise NotImplementedError

    def _is_preparing(self) -> bool:
        """Returns whether the run is still warming up, before the first unit of work has been reported."""
        raise NotImplementedError

    def _compose_active(self, elapsed: float) -> str:
        """Builds the line body, everything after the spinner, shown once work has started.

        Args:
            elapsed: The seconds elapsed since the renderer was constructed.

        Returns:
            The composed line body, typically ``_bar(...)`` followed by the subclass's own segments and an ETA.
        """
        raise NotImplementedError

    def _compose_preparing(self, elapsed: float) -> str:
        """Builds the warm-up line body shown before the first unit of work is reported.

        Subclasses may override to add context such as a video count. The default shows only the label and elapsed.

        Args:
            elapsed: The seconds elapsed since the renderer was constructed.

        Returns:
            The composed warm-up line body.
        """
        return f"[{'-' * self._width}] {self._preparing_label} | {format_duration(elapsed)} elapsed"

    def _bar(self, fraction: float) -> tuple[str, float]:
        """Builds the filled-and-empty bar string and its percent for a completion fraction.

        Args:
            fraction: The completion fraction, clamped into ``[0, 1]``.

        Returns:
            A tuple of the rendered bar string and its percent value.
        """
        fraction = min(1.0, max(0.0, fraction))
        filled = int(self._width * fraction)
        return "#" * filled + "-" * (self._width - filled), 100.0 * fraction

    def _eta(self, done: float, total: float, elapsed: float) -> str:
        """Formats an honest estimated time remaining from the work done so far.

        Shows a placeholder until there is measurable throughput, so warm-up never reads as a finished ``00:00``. Shows
        a plain zero at completion. Shows the projected remaining time otherwise.

        Args:
            done: The units of work completed so far.
            total: The total units of work.
            elapsed: The seconds elapsed since the renderer was constructed.

        Returns:
            The formatted ETA, or ``--:--`` while the rate is still unknown.
        """
        if total > 0 and done >= total:
            return format_duration(0)
        rate = done / elapsed if elapsed > 0 else 0.0
        if rate > 0 and total > 0:
            return format_duration((total - done) / rate)
        return "--:--"

    def _render(self, *, force: bool = False) -> None:
        """Draws the bar, honoring the per-mode minimum render interval unless ``force`` is set.

        Args:
            force: Determines whether to render immediately, bypassing the minimum interval between renders.
        """
        now = time.monotonic()
        interval = _TTY_RENDER_INTERVAL if self._is_tty else _NON_TTY_RENDER_INTERVAL
        if not force and (now - self._last_render_time) < interval:
            return
        self._last_render_time = now

        # Advances the spinner on every drawn line so the bar stays visibly alive through the count-static stretches of
        # a run: warm-up before the first unit of work is reported, and the decode/cluster gaps between updates.
        spinner = _SPINNER_FRAMES[self._spinner_index % len(_SPINNER_FRAMES)]
        self._spinner_index += 1
        elapsed = now - self._start_time
        body = self._compose_preparing(elapsed) if self._is_preparing() else self._compose_active(elapsed)
        message = f"{spinner} {body}"
        if self._is_tty:
            self._stream.write("\r" + message + "\033[K")
        else:
            self._stream.write(message + "\n")
        self._stream.flush()
