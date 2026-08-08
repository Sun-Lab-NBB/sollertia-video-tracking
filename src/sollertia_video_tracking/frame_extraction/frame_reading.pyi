from typing import Any
from collections.abc import (
    Callable as Callable,
    Iterable,
)

import numpy as np
from numpy.typing import NDArray as NDArray

_MINIMUM_STREAMABLE_CANDIDATE_COUNT: int
_STREAMING_MAX_MEAN_GAP: int

def make_fast_kmeans_selector(
    *, progress: Callable[..., Iterable[Any]] | None = None, frame_count: int | None = None
) -> Callable[..., list[int]]: ...
def _select_kmeans_frames(
    *,
    video_reader: Any,
    cluster_count: int,
    window_start: float,
    window_stop: float,
    frame_indices: Any,
    sampling_step: int,
    resize_width: int,
    batch_size: int,
    maximum_iterations: int,
    cluster_in_color: bool,
    progress: Callable[..., Iterable[Any]],
) -> list[int]: ...
def _resolve_candidate_indices(
    *, frame_indices: Any, frame_count: int, window_start: float, window_stop: float, sampling_step: int
) -> NDArray[np.int64]: ...
def _should_stream(candidate_indices: NDArray[np.int64]) -> bool: ...
def _read_thumbnails_streaming(
    *,
    video_reader: Any,
    candidate_indices: NDArray[np.int64],
    downsample_ratio: float,
    cluster_in_color: bool,
    thumbnails: NDArray[np.float64],
    progress: Callable[..., Iterable[Any]],
) -> None: ...
def _read_thumbnails_seeking(
    *,
    video_reader: Any,
    candidate_indices: NDArray[np.int64],
    downsample_ratio: float,
    cluster_in_color: bool,
    thumbnails: NDArray[np.float64],
    progress: Callable[..., Iterable[Any]],
) -> None: ...
def _downsample_frame(
    *, frame: NDArray[np.uint8], downsample_ratio: float, cluster_in_color: bool
) -> NDArray[np.float64]: ...
def _cluster_and_pick(
    *,
    thumbnails: NDArray[np.float64],
    candidate_indices: NDArray[np.int64],
    cluster_count: int,
    batch_size: int,
    maximum_iterations: int,
) -> list[int]: ...
