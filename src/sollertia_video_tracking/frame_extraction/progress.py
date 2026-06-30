"""Provides the aggregate progress bar and the DeepLabCut tqdm shim used to report frame-extraction progress."""

import sys
import time
from queue import Empty
from typing import Any
from threading import Thread
import contextlib
from collections.abc import Callable, Iterable

_PROGRESS_BAR_WIDTH: int = 30
"""The width, in characters, of the rendered aggregate progress bar."""

_MAX_PROGRESS_UPDATES_PER_VIDEO: int = 100
"""The upper bound on progress messages each worker emits, used to throttle queue traffic."""


def make_progress_reporter(progress_queue: Any, video_index: int, total: int) -> Callable[..., Iterable[Any]]:
    """Builds a drop-in replacement for ``tqdm`` that streams frame counts from a worker to the parent process.

    DeepLabCut wraps its frame-reading loop in ``tqdm(enumerate(index))``. The returned callable wraps that same
    iterable, forwards every item unchanged, and reports progress (throttled to a bounded number of messages per
    video) so the parent can render a single aggregate bar.

    Args:
        progress_queue: The shared queue used to forward progress messages to the parent renderer.
        video_index: The index identifying which video this reporter tracks.
        total: The total number of frames this video contributes to the aggregate bar.

    Returns:
        A callable that mirrors the tqdm interface and emits throttled progress messages while iterating.
    """
    every = max(1, total // _MAX_PROGRESS_UPDATES_PER_VIDEO)

    def reporter(iterable: Iterable[Any], *_args: Any, **_kwargs: Any) -> Iterable[Any]:
        """Iterates over the wrapped iterable, forwarding each item and emitting throttled progress messages."""
        for count, item in enumerate(iterable, start=1):
            if count % every == 0 or count == total:
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
        returns; when the output is redirected, it prints at most one line every ``heartbeat`` seconds.
    """

    def __init__(
        self,
        progress_queue: Any,
        total_videos: int,
        totals: dict[int, int],
        heartbeat: float,
        width: int = _PROGRESS_BAR_WIDTH,
        stream: Any = None,
    ) -> None:
        """Initializes the renderer thread over the given per-video frame totals.

        Args:
            progress_queue: The shared queue the workers stream progress and completion messages to.
            total_videos: The total number of videos in the extraction run.
            totals: The mapping of video index to the number of frames that video contributes to the bar.
            heartbeat: The minimum interval, in seconds, between rendered lines when the output is not a TTY.
            width: The width, in characters, of the rendered bar.
            stream: The output stream to render to, defaulting to the standard error stream.
        """
        super().__init__(daemon=True)
        self._progress_queue = progress_queue
        self._total_videos = total_videos
        self._totals = totals
        self._grand_total = max(1, sum(totals.values()))
        self._heartbeat = heartbeat
        self._width = width
        self._stream = stream if stream is not None else sys.stderr
        self._is_tty = self._stream.isatty()
        self._frames: dict[int, int] = {}
        self._videos_done = 0
        self._start_time = time.monotonic()
        self._last_render_time = 0.0

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
                self._frames[video_index] = self._totals.get(video_index, 0)
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
            force: Whether to render immediately, bypassing the minimum interval between renders.
        """
        now = time.monotonic()
        interval = 0.2 if self._is_tty else max(1.0, self._heartbeat)
        if not force and (now - self._last_render_time) < interval:
            return
        self._last_render_time = now

        frames_read = min(sum(self._frames.values()), self._grand_total)
        elapsed = now - self._start_time
        rate = frames_read / elapsed if elapsed > 0 else 0.0
        eta = (self._grand_total - frames_read) / rate if rate > 0 and frames_read < self._grand_total else 0.0
        percent = 100.0 * frames_read / self._grand_total
        filled = int(self._width * frames_read / self._grand_total)
        bar = "#" * filled + "-" * (self._width - filled)
        message = (
            f"[{bar}] {percent:5.1f}% | {self._videos_done}/{self._total_videos} videos | "
            f"{frames_read:,}/{self._grand_total:,} frames | {_format_duration(elapsed)} | "
            f"ETA {_format_duration(eta)}"
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
