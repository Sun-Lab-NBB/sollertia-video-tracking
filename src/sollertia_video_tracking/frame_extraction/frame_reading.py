"""Provides a fast, decode-aware replacement for DeepLabCut's k-means candidate-frame reader."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
from tqdm import tqdm
import numpy as np
from skimage.util import img_as_ubyte
from sklearn.cluster import MiniBatchKMeans

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from numpy.typing import NDArray

_MINIMUM_STREAMABLE_CANDIDATE_COUNT: int = 2
"""The fewest candidates for which a mean inter-candidate gap is defined, below which the reader always seeks."""

_STREAMING_MAX_MEAN_GAP: int = 200
"""The largest mean candidate gap, in frames, at which streaming still beats seeking.

Measured as the crossover on long-GOP (~250) HEVC video. It affects only speed, never which frames are selected,
because the seek and stream paths read and cluster the identical candidate frames."""


def make_fast_kmeans_selector(*, progress: Callable[..., Iterable[Any]] | None = None) -> Callable[..., list[int]]:
    """Builds a drop-in replacement for DeepLabCut's ``KmeansbasedFrameselectioncv2`` that avoids per-candidate seeks.

    DeepLabCut's reader seeks to every flagged candidate, which rewinds to a keyframe on long-GOP video and collapses
    throughput on the dense candidate sets the ``uncertain`` detector produces. The returned reader instead streams the
    video once when the candidates are dense, decoding past non-candidates and downsampling only the flagged frames,
    and falls back to seeking when they are sparse. Its clustering mirrors DeepLabCut, so it selects the identical
    frames and only runs faster.

    Notes:
        A worker installs the returned callable onto ``frameselectiontools.KmeansbasedFrameselectioncv2`` before it
        invokes DeepLabCut's frame extraction, so the call site stays identical to upstream. The reader depends on that
        call convention and on DeepLabCut's ``VideoReader``, so it must be re-verified against any new pinned release.

    Args:
        progress: A ``tqdm``-compatible callable that wraps the per-candidate iteration to report progress, or None to
            fall back to a plain ``tqdm`` bar, matching DeepLabCut's own default.

    Returns:
        A callable mirroring ``KmeansbasedFrameselectioncv2`` that returns the frame indices to extract.
    """

    def select(
        video_reader: Any,
        cluster_count: int,
        window_start: float,
        window_stop: float,
        frame_indices: Any = None,
        **deeplabcut_options: Any,
    ) -> list[int]:
        """Adapts DeepLabCut's ``KmeansbasedFrameselectioncv2`` call convention to the module's clustering entry point.

        DeepLabCut passes the video reader, cluster count, window bounds, and candidate indices positionally, then its
        ``step``, ``resizewidth``, ``batchsize``, ``max_iter``, and ``color`` settings by keyword. Absorbing those
        keywords keeps the installed call site identical to upstream while letting every internal function use full,
        descriptive names.
        """
        return _select_kmeans_frames(
            video_reader=video_reader,
            cluster_count=cluster_count,
            window_start=window_start,
            window_stop=window_stop,
            frame_indices=frame_indices,
            sampling_step=deeplabcut_options.get("step", 1),
            resize_width=deeplabcut_options.get("resizewidth", 30),
            batch_size=deeplabcut_options.get("batchsize", 100),
            maximum_iterations=deeplabcut_options.get("max_iter", 50),
            cluster_in_color=deeplabcut_options.get("color", False),
            progress=progress if progress is not None else tqdm,
        )

    return select


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
) -> list[int]:
    """Downsamples the candidate frames, clusters them, and returns one representative frame index per cluster.

    Mirrors DeepLabCut's ``KmeansbasedFrameselectioncv2`` apart from the frame reading. When the candidates are dense,
    it streams the video once and downsamples only the flagged frames; when they are sparse, it seeks to each. With
    fewer candidates than the cluster count, it returns them all without decoding, as upstream does.

    Args:
        video_reader: DeepLabCut's ``VideoReader`` for the source video, already cropped to the analysis bounding box.
        cluster_count: The number of frames to select, used as the k-means cluster count.
        window_start: The fractional start of the analysis window, in [0, 1].
        window_stop: The fractional stop of the analysis window, in [0, 1].
        frame_indices: The candidate frame indices to select from, or None to sample the window at ``sampling_step``.
        sampling_step: The stride used to build the candidate indices when ``frame_indices`` is None.
        resize_width: The width, in pixels, each frame is downsampled to before clustering.
        batch_size: The mini-batch size for k-means.
        maximum_iterations: The maximum number of k-means iterations.
        cluster_in_color: Determines whether frames are clustered on their color channels rather than grayscale.
        progress: The ``tqdm``-compatible iteration wrapper used to report progress.

    Returns:
        The selected frame indices, one representative per non-empty cluster, or all candidate indices unchanged when
        there are fewer candidates than ``cluster_count``.

    Raises:
        ValueError: If ``resize_width`` would upsample the frames, matching DeepLabCut's own guard.
    """
    frame_count = len(video_reader)
    crop_width, crop_height = video_reader.dimensions
    downsample_ratio = resize_width / crop_width
    if downsample_ratio > 1:
        message = (
            f"Unable to select frames by clustering. The clustering resize width must not upsample the frames, but "
            f"{resize_width} exceeds the frame width {crop_width}."
        )
        raise ValueError(message)

    candidate_indices = _resolve_candidate_indices(
        frame_indices=frame_indices,
        frame_count=frame_count,
        window_start=window_start,
        window_stop=window_stop,
        sampling_step=sampling_step,
    )
    if len(candidate_indices) < cluster_count:
        return [int(frame_index) for frame_index in candidate_indices]

    scaled_height = int(np.round(crop_height * downsample_ratio))
    scaled_width = int(np.round(crop_width * downsample_ratio))
    column_count = scaled_width * 3 if cluster_in_color else scaled_width
    thumbnails = np.empty((len(candidate_indices), scaled_height, column_count), dtype=np.float64)

    should_stream = _should_stream(candidate_indices=candidate_indices)
    read_thumbnails = _read_thumbnails_streaming if should_stream else _read_thumbnails_seeking
    read_thumbnails(
        video_reader=video_reader,
        candidate_indices=candidate_indices,
        downsample_ratio=downsample_ratio,
        cluster_in_color=cluster_in_color,
        thumbnails=thumbnails,
        progress=progress,
    )

    return _cluster_and_pick(
        thumbnails=thumbnails,
        candidate_indices=candidate_indices,
        cluster_count=cluster_count,
        batch_size=batch_size,
        maximum_iterations=maximum_iterations,
    )


def _resolve_candidate_indices(
    *, frame_indices: Any, frame_count: int, window_start: float, window_stop: float, sampling_step: int
) -> NDArray[np.int64]:
    """Builds the sorted candidate frame indices, cropping them to the analysis window as DeepLabCut does.

    Args:
        frame_indices: The caller-supplied candidate indices, or None to sample the window at ``sampling_step``.
        frame_count: The total number of frames in the video.
        window_start: The fractional start of the analysis window, in [0, 1].
        window_stop: The fractional stop of the analysis window, in [0, 1].
        sampling_step: The stride used to build the candidate indices when ``frame_indices`` is None.

    Returns:
        The candidate frame indices confined to the analysis window.
    """
    start_index = int(np.floor(frame_count * window_start))
    stop_index = int(np.ceil(frame_count * window_stop))
    if frame_indices is None:
        return np.arange(start_index, stop_index, sampling_step, dtype=np.int64)
    candidate_indices = np.asarray(frame_indices, dtype=np.int64)
    # Mirrors DeepLabCut's strict-inequality window crop so the flagged frames match the upstream tool exactly.
    return candidate_indices[(candidate_indices > start_index) & (candidate_indices < stop_index)]


def _should_stream(candidate_indices: NDArray[np.int64]) -> bool:
    """Determines whether streaming the video reads fewer frames than seeking to each candidate.

    Streaming decodes every frame between the first and last candidate once, while seeking re-decodes roughly one
    keyframe interval per candidate. Streaming therefore wins when the mean candidate gap falls below the measured
    crossover.

    Args:
        candidate_indices: The sorted candidate frame indices.

    Returns:
        True when the candidates are dense enough to favor a single sequential pass.
    """
    if len(candidate_indices) < _MINIMUM_STREAMABLE_CANDIDATE_COUNT:
        return False
    span = int(candidate_indices[-1] - candidate_indices[0])
    mean_gap = span / (len(candidate_indices) - 1)
    return mean_gap <= _STREAMING_MAX_MEAN_GAP


def _read_thumbnails_streaming(
    *,
    video_reader: Any,
    candidate_indices: NDArray[np.int64],
    downsample_ratio: float,
    cluster_in_color: bool,
    thumbnails: NDArray[np.float64],
    progress: Callable[..., Iterable[Any]],
) -> None:
    """Fills the thumbnail buffer by streaming the video once and downsampling only the candidate frames.

    Seeks once to the first candidate, then advances frame by frame, grabbing past non-candidates without decoding
    them to an array and decoding only the flagged frames. This avoids the per-candidate keyframe rewind that makes
    seeking slow on dense candidates.

    Args:
        video_reader: DeepLabCut's ``VideoReader``, already cropped to the analysis bounding box.
        candidate_indices: The sorted candidate frame indices to downsample.
        downsample_ratio: The downsampling ratio applied to each candidate frame.
        cluster_in_color: Determines whether the thumbnail keeps the color channels rather than a grayscale mean.
        thumbnails: The pre-allocated buffer that receives one downsampled frame per candidate.
        progress: The ``tqdm``-compatible iteration wrapper used to report progress.
    """
    video_reader.set_to_frame(int(candidate_indices[0]))
    next_frame_index = int(candidate_indices[0])
    for row, candidate in progress(enumerate(candidate_indices)):
        candidate_index = int(candidate)
        while next_frame_index < candidate_index:
            video_reader.video.grab()
            next_frame_index += 1
        frame = video_reader.read_frame(crop=True)
        next_frame_index += 1
        if frame is not None:
            thumbnails[row] = _downsample_frame(
                frame=frame, downsample_ratio=downsample_ratio, cluster_in_color=cluster_in_color
            )


def _read_thumbnails_seeking(
    *,
    video_reader: Any,
    candidate_indices: NDArray[np.int64],
    downsample_ratio: float,
    cluster_in_color: bool,
    thumbnails: NDArray[np.float64],
    progress: Callable[..., Iterable[Any]],
) -> None:
    """Fills the thumbnail buffer by seeking to each candidate, matching DeepLabCut for sparse candidate sets.

    When the candidates are spread far apart, seeking to each one decodes fewer frames than streaming the whole span,
    so this mirrors DeepLabCut's original per-candidate seek.

    Args:
        video_reader: DeepLabCut's ``VideoReader``, already cropped to the analysis bounding box.
        candidate_indices: The sorted candidate frame indices to downsample.
        downsample_ratio: The downsampling ratio applied to each candidate frame.
        cluster_in_color: Determines whether the thumbnail keeps the color channels rather than a grayscale mean.
        thumbnails: The pre-allocated buffer that receives one downsampled frame per candidate.
        progress: The ``tqdm``-compatible iteration wrapper used to report progress.
    """
    for row, candidate in progress(enumerate(candidate_indices)):
        video_reader.set_to_frame(int(candidate))
        frame = video_reader.read_frame(crop=True)
        if frame is not None:
            thumbnails[row] = _downsample_frame(
                frame=frame, downsample_ratio=downsample_ratio, cluster_in_color=cluster_in_color
            )


def _downsample_frame(
    *, frame: NDArray[np.uint8], downsample_ratio: float, cluster_in_color: bool
) -> NDArray[np.float64]:
    """Downsamples one decoded frame to the clustering thumbnail, matching DeepLabCut's transform exactly.

    Args:
        frame: The decoded, cropped frame in RGB channel order, as DeepLabCut's ``read_frame`` returns it.
        downsample_ratio: The downsampling ratio applied to the frame.
        cluster_in_color: Determines whether the thumbnail keeps the color channels rather than a grayscale mean.

    Returns:
        The downsampled frame as a two-dimensional array, color channels stacked horizontally when in color.
    """
    image = img_as_ubyte(
        cv2.resize(frame, dsize=None, fx=downsample_ratio, fy=downsample_ratio, interpolation=cv2.INTER_NEAREST)
    )
    if cluster_in_color:
        return np.hstack([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
    return np.mean(image, axis=2)


def _cluster_and_pick(
    *,
    thumbnails: NDArray[np.float64],
    candidate_indices: NDArray[np.int64],
    cluster_count: int,
    batch_size: int,
    maximum_iterations: int,
) -> list[int]:
    """Clusters the downsampled frames and picks one random representative frame index per non-empty cluster.

    Mirrors DeepLabCut's mini-batch k-means selection, including its mean-centering, its per-cluster random pick, and
    its shrinking of the batch size when there are fewer candidates than the configured batch.

    Args:
        thumbnails: The downsampled frames, one row per candidate.
        candidate_indices: The candidate frame indices aligned with the thumbnail rows.
        cluster_count: The number of clusters, which is the number of frames to select.
        batch_size: The configured mini-batch size, shrunk when it exceeds the candidate count.
        maximum_iterations: The maximum number of k-means iterations.

    Returns:
        The selected frame indices, one representative per non-empty cluster.
    """
    candidate_count = len(candidate_indices)
    effective_batch_size = batch_size if batch_size <= candidate_count else candidate_count // 2
    # Center in place: thumbnails is a local buffer that is never read after this, so subtracting the column mean
    # into it avoids allocating a second full-size copy. The reshape is a view. Bit-identical to DeepLabCut, which
    # centers the same way (frameselectiontools.py) before discarding its own buffer.
    thumbnails -= thumbnails.mean(axis=0)
    centered = thumbnails.reshape(candidate_count, -1)

    kmeans = MiniBatchKMeans(
        n_clusters=cluster_count, tol=1e-3, batch_size=effective_batch_size, max_iter=maximum_iterations
    )
    kmeans.fit(centered)

    generator = np.random.default_rng()
    cluster_members = (np.flatnonzero(kmeans.labels_ == cluster_id) for cluster_id in range(cluster_count))
    return [
        int(candidate_indices[members[generator.integers(len(members))]]) for members in cluster_members if members.size
    ]
