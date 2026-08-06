"""Provides the aggregate progress bar and the DeepLabCut tqdm shim used to report frame-extraction progress."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import contextlib

from ..reporting import LiveBar, format_duration

if TYPE_CHECKING:
    from typing import TextIO
    from collections.abc import Callable, Iterable

_MAXIMUM_PROGRESS_UPDATES_PER_VIDEO: int = 100
"""The approximate number of progress messages each worker targets per video. It sets the base per-video update stride
of ``frame_total // this value``, which ``_MAXIMUM_FRAMES_PER_UPDATE`` then caps. The real message count can therefore
exceed this target both for very small videos and for very large ones whose stride is capped."""

_MAXIMUM_FRAMES_PER_UPDATE: int = 250
"""The largest per-video frame stride between progress messages. It caps the ``frame_total // updates`` stride so a
video with a very large frame count still reports at a steady, visible cadence. Without the cap a huge
outlier-candidate pool decoded off a multi-gigabyte file would update only once every one percent of a long decode."""


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
    frames_per_update = max(1, min(frame_total // _MAXIMUM_PROGRESS_UPDATES_PER_VIDEO, _MAXIMUM_FRAMES_PER_UPDATE))

    def reporter(iterable: Iterable[Any], *_args: Any, **_kwargs: Any) -> Iterable[Any]:
        """Iterates over the wrapped iterable, forwarding each item and emitting throttled progress messages."""
        # Announces the video as in flight before its first frame is decoded. A large candidate pool can spend minutes
        # in random-access reads before the first stride is crossed, and this initial zero lets the aggregate bar count
        # the video as actively decoding immediately instead of leaving it invisible until the first stride.
        with contextlib.suppress(Exception):
            progress_queue.put_nowait(("progress", video_index, 0))
        for count, item in enumerate(iterable, start=1):
            if count % frames_per_update == 0 or count == frame_total:
                # A dropped progress update only skips a bar refresh, so a full queue is never fatal.
                with contextlib.suppress(Exception):
                    progress_queue.put_nowait(("progress", video_index, count))
            yield item

    return reporter


class AggregateBar(LiveBar):
    """Renders a single progress bar that tracks the total frames read across all extraction or inference workers.

    Notes:
        The renderer consumes ``("progress", key, count)`` and ``("done", key)`` messages from the shared queue, where
        ``key`` identifies a work unit (a whole video, or one frame-range chunk of a video). The warm-up, spinner,
        interval, and ``stop``-sentinel handling is inherited from ``LiveBar``. Each reporter announces its work unit
        with a ``count`` of zero before its first frame is decoded, so the bar leaves the warm-up line for the active
        bar as soon as any worker begins. It counts every announced but unfinished video as actively decoding.

    Attributes:
        _total_video_count: The total number of videos in the run.
        _frame_totals: The mapping of work-unit key to the number of frames that unit contributes to the bar.
        _grand_frame_total: The sum of all per-unit frame totals, clamped to at least one.
        _frames: The mapping of work-unit key to the most recent frame count reported for that unit.
        _key_video: The mapping of work-unit key to the video index it belongs to, so chunk completions roll up.
        _video_remaining: The mapping of video index to the number of its work units that have not yet finished.
        _videos_done: The number of videos that have finished.
    """

    def __init__(
        self,
        progress_queue: Any,
        total_video_count: int,
        frame_totals: dict[int, int],
        preparing_label: str = "preparing...",
        stream: TextIO | None = None,
        *,
        key_video: dict[int, int] | None = None,
    ) -> None:
        """Initializes the renderer thread over the given per-work-unit frame totals.

        Args:
            progress_queue: The shared queue the workers stream progress and completion messages to.
            total_video_count: The total number of videos in the run.
            frame_totals: The mapping of work-unit key to the number of frames that unit contributes to the bar.
            preparing_label: The warm-up text shown before the first worker begins decoding.
            stream: The output stream to render to, defaulting to the standard error stream.
            key_video: The mapping of work-unit key to the video index it belongs to, or None when each key is its own
                video. Chunked inference passes one entry per frame-range chunk so a video is marked done only once all
                its chunks finish.
        """
        super().__init__(
            progress_queue=progress_queue,
            preparing_label=preparing_label,
            stream=stream,
        )
        self._total_video_count = total_video_count
        self._frame_totals = frame_totals
        self._grand_frame_total = max(1, sum(frame_totals.values()))
        self._frames: dict[int, int] = {}
        # Each frame-total key is one unit of work: a whole video, or one contiguous chunk of a video when inference
        # splits videos into parallel frame ranges. The identity fallback makes every key its own video, preserving the
        # one-producer-per-video behavior used by frame extraction and unchunked inference.
        self._key_video: dict[int, int] = key_video if key_video is not None else {key: key for key in frame_totals}
        self._video_remaining: dict[int, int] = {}
        for video in self._key_video.values():
            self._video_remaining[video] = self._video_remaining.get(video, 0) + 1
        self._videos_done = 0

    def __repr__(self) -> str:
        """Returns a string representation of the AggregateBar instance."""
        return (
            f"AggregateBar(total_video_count={self._total_video_count}, grand_frame_total={self._grand_frame_total}, "
            f"videos_done={self._videos_done})"
        )

    def _ingest(self, message: Any) -> bool:
        """Merges one ``progress`` or ``done`` message into the retained per-work-unit frame counts.

        Args:
            message: A ``("progress", key, count)`` or ``("done", key)`` message, where ``key`` identifies a work unit
                (a whole video, or one frame-range chunk of a video).

        Returns:
            True for a ``done`` message so the completion is drawn immediately, False otherwise.
        """
        kind = message[0]
        if kind == "progress":
            _, key, count = message
            self._frames[key] = count
            return False
        if kind == "done":
            _, key = message
            self._frames[key] = self._frame_totals.get(key, 0)
            video = self._key_video.get(key, key)
            self._video_remaining[video] = self._video_remaining.get(video, 1) - 1
            if self._video_remaining[video] == 0:
                self._videos_done += 1
            return True
        return False

    def _is_preparing(self) -> bool:
        """Returns whether no worker has begun decoding yet, so the warm-up line is still shown."""
        return not self._frames and self._videos_done == 0

    def _compose_preparing(self, elapsed: float) -> str:
        """Builds the warm-up line, adding the video count so the wait shows how much work is queued.

        Args:
            elapsed: The seconds elapsed since the renderer was constructed.

        Returns:
            The composed warm-up line body.
        """
        return (
            f"[{'-' * self._width}] {self._preparing_label} | {self._videos_done}/{self._total_video_count} videos | "
            f"{format_duration(seconds=elapsed)} elapsed"
        )

    def _compose_active(self, elapsed: float) -> str:
        """Builds the active line body from the frames read across all videos.

        Args:
            elapsed: The seconds elapsed since the renderer was constructed.

        Returns:
            The composed active line body.
        """
        frames_read = min(sum(self._frames.values()), self._grand_frame_total)
        bar, percent = self._bar(fraction=frames_read / self._grand_frame_total)
        # Videos that have announced their decode but are not yet done are actively working, so surfacing the count
        # tells the operator work is in flight even while the aggregate frame count holds steady. Distinct videos are
        # counted rather than chunk keys, so a chunked video reports as one decoding video, not several.
        active = max(0, len({self._key_video.get(key, key) for key in self._frames}) - self._videos_done)
        eta = self._eta(done=frames_read, total=self._grand_frame_total, elapsed=elapsed)
        return (
            f"[{bar}] {percent:5.1f}% | {self._videos_done}/{self._total_video_count} videos "
            f"({active} decoding) | {frames_read:,}/{self._grand_frame_total:,} frames | "
            f"{format_duration(seconds=elapsed)} | ETA {eta}"
        )
