from typing import Any, TextIO
from collections.abc import (
    Callable as Callable,
    Iterable,
)

from _typeshed import Incomplete

from ..reporting import (
    LiveBar as LiveBar,
    format_duration as format_duration,
)

_MAX_PROGRESS_UPDATES_PER_VIDEO: int
_MAX_FRAMES_PER_UPDATE: int

def make_progress_reporter(progress_queue: Any, video_index: int, frame_total: int) -> Callable[..., Iterable[Any]]: ...

class AggregateBar(LiveBar):
    _total_video_count: Incomplete
    _frame_totals: Incomplete
    _grand_frame_total: Incomplete
    _frames: dict[int, int]
    _key_video: dict[int, int]
    _video_remaining: dict[int, int]
    _videos_done: int
    def __init__(
        self,
        progress_queue: Any,
        total_video_count: int,
        frame_totals: dict[int, int],
        preparing_label: str = "preparing...",
        stream: TextIO | None = None,
        *,
        key_video: dict[int, int] | None = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def _ingest(self, message: Any) -> bool: ...
    def _is_preparing(self) -> bool: ...
    def _compose_preparing(self, elapsed: float) -> str: ...
    def _compose_active(self, elapsed: float) -> str: ...
