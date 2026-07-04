"""Provides the aggregate progress bar and the DeepLabCut tqdm shim used to report frame-extraction progress."""

import sys
import time
from queue import Empty
from typing import Any, TextIO
from threading import Thread
import contextlib
from collections.abc import Callable, Iterable

_PROGRESS_BAR_WIDTH: int = 30
"""The width, in characters, of the rendered aggregate progress bar."""

_MAX_PROGRESS_UPDATES_PER_VIDEO: int = 100
"""The approximate number of progress messages each worker targets per video; it sets the per-video update stride
(``frame_total // this value``), so the real count can exceed it for videos with fewer than 200 frames."""


def make_progress_reporter(progress_queue: Any, video_index: int, frame_total: int) -> Callable[..., Iterable[Any]]:
    """Builds a drop-in replacement for ``tqdm`` that streams frame counts from a worker to the parent process.

    DeepLabCut wraps its frame-reading loop in ``tqdm(enumerate(index))``. The returned callable wraps that same
    iterable, forwards every item unchanged, and reports progress (throttled to a bounded number of messages per
    video) so the parent can render a single aggregate bar.

    Args:
        progress_queue: The shared queue used to forward progress messages to the parent renderer.
        video_index: The index identifying which video this reporter tracks.
        frame_total: The total number of frames this video contributes to the aggregate bar.

    Returns:
        A callable that mirrors the tqdm interface and emits throttled progress messages while iterating.
    """
    frames_per_update = max(1, frame_total // _MAX_PROGRESS_UPDATES_PER_VIDEO)

    def reporter(iterable: Iterable[Any], *_args: Any, **_kwargs: Any) -> Iterable[Any]:
        """Iterates over the wrapped iterable, forwarding each item and emitting throttled progress messages."""
        for count, item in enumerate(iterable, start=1):
            if count % frames_per_update == 0 or count == frame_total:
                # A dropped progress update only skips a bar refresh, so a full queue is never fatal.
                with contextlib.suppress(Exception):
                    progress_queue.put_nowait(("progress", video_index, count))
            yield item

    return reporter


class AggregateBar(Thread):
    """Renders a single progress bar that tracks the total frames read across all extraction workers.

    Notes:
        The renderer consumes ``("progress", video_index, count)`` and ``("done", video_index)`` messages from the
        shared queue, plus a terminal ``("stop",)`` sentinel. On a TTY the bar updates in place with carriage
        returns; when the output is redirected, it prints at most one line every ``minimum_progress_interval``
        seconds (clamped to at least 1.0 second).

    Attributes:
        _progress_queue: The shared queue the workers stream progress and completion messages to.
        _total_video_count: The total number of videos in the extraction run.
        _frame_totals: The mapping of video index to the number of frames that video contributes to the bar.
        _grand_frame_total: The sum of all per-video frame totals, clamped to at least one.
        _minimum_progress_interval: The minimum interval, in seconds, between rendered lines when the output
            is not a TTY; the effective interval is clamped to at least 1.0 second.
        _width: The width, in characters, of the rendered bar.
        _stream: The output stream the bar renders to.
        _is_tty: True when the output stream is an interactive terminal.
        _frames: The mapping of video index to the most recent frame count reported for that video.
        _videos_done: The number of videos that have finished extraction.
        _start_time: The monotonic timestamp captured when the renderer was constructed.
        _last_render_time: The monotonic timestamp of the most recent render.
    """

    def __init__(
        self,
        progress_queue: Any,
        total_video_count: int,
        frame_totals: dict[int, int],
        minimum_progress_interval: float,
        width: int = _PROGRESS_BAR_WIDTH,
        stream: TextIO | None = None,
    ) -> None:
        """Initializes the renderer thread over the given per-video frame totals.

        Args:
            progress_queue: The shared queue the workers stream progress and completion messages to.
            total_video_count: The total number of videos in the extraction run.
            frame_totals: The mapping of video index to the number of frames that video contributes to the bar.
            minimum_progress_interval: The minimum interval, in seconds, between rendered lines when the output
                is not a TTY; the effective interval is clamped to at least 1.0 second.
            width: The width, in characters, of the rendered bar.
            stream: The output stream to render to, defaulting to the standard error stream.
        """
        super().__init__(daemon=True)
        self._progress_queue = progress_queue
        self._total_video_count = total_video_count
        self._frame_totals = frame_totals
        self._grand_frame_total = max(1, sum(frame_totals.values()))
        self._minimum_progress_interval = minimum_progress_interval
        self._width = width
        self._stream = stream if stream is not None else sys.stderr
        self._is_tty = self._stream.isatty()
        self._frames: dict[int, int] = {}
        self._videos_done = 0
        self._start_time = time.monotonic()
        self._last_render_time = 0.0

    def __repr__(self) -> str:
        """Returns a string representation of the AggregateBar instance."""
        return (
            f"AggregateBar(total_video_count={self._total_video_count}, grand_frame_total={self._grand_frame_total}, "
            f"videos_done={self._videos_done})"
        )

    def run(self) -> None:
        """Consumes queue messages and re-renders the bar until a ``("stop",)`` sentinel arrives."""
        while True:
            try:
                message = self._progress_queue.get(timeout=0.2 if self._is_tty else 1.0)
            except Empty:
                self._render()
                continue
            kind = message[0]
            if kind == "progress":
                _, video_index, count = message
                self._frames[video_index] = count
                self._render()
            elif kind == "done":
                _, video_index = message
                self._frames[video_index] = self._frame_totals.get(video_index, 0)
                self._videos_done += 1
                self._render(force=True)
            elif kind == "stop":
                break
        self._render(force=True)
        if self._is_tty:
            self._stream.write("\n")
            self._stream.flush()

    def stop(self) -> None:
        """Signals the renderer to draw a final frame and exit."""
        self._progress_queue.put(("stop",))

    def _render(self, *, force: bool = False) -> None:
        """Draws the bar, honoring the per-mode minimum render interval unless ``force`` is set.

        Args:
            force: Determines whether to render immediately, bypassing the minimum interval between renders.
        """
        now = time.monotonic()
        interval = 0.2 if self._is_tty else max(1.0, self._minimum_progress_interval)
        if not force and (now - self._last_render_time) < interval:
            return
        self._last_render_time = now

        frames_read = min(sum(self._frames.values()), self._grand_frame_total)
        elapsed = now - self._start_time
        rate = frames_read / elapsed if elapsed > 0 else 0.0
        estimated_seconds_remaining = (
            (self._grand_frame_total - frames_read) / rate
            if rate > 0 and frames_read < self._grand_frame_total
            else 0.0
        )
        percent = 100.0 * frames_read / self._grand_frame_total
        filled = int(self._width * frames_read / self._grand_frame_total)
        bar = "#" * filled + "-" * (self._width - filled)
        message = (
            f"[{bar}] {percent:5.1f}% | {self._videos_done}/{self._total_video_count} videos | "
            f"{frames_read:,}/{self._grand_frame_total:,} frames | {_format_duration(elapsed)} | "
            f"ETA {_format_duration(estimated_seconds_remaining)}"
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
