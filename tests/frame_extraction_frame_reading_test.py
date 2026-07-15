"""Tests for the fast, decode-aware k-means candidate-frame reader that replaces DeepLabCut's own."""

import numpy as np
import pytest

from sollertia_video_tracking.frame_extraction import frame_reading
from sollertia_video_tracking.frame_extraction.frame_reading import (
    _should_stream,
    _cluster_and_pick,
    _downsample_frame,
    _select_kmeans_frames,
    _read_thumbnails_seeking,
    make_fast_kmeans_selector,
    _read_thumbnails_streaming,
    _resolve_candidate_indices,
)


class FakeVideoReader:
    """Minimal stand-in for DeepLabCut's ``VideoReader`` driven from an in-memory frame table.

    A single cursor backs both grabbing (skip past a frame) and reading (decode and advance), so the streaming and
    seeking readers exercise the identical positioning logic they would against the real reader. ``video`` points back
    at the instance so ``video_reader.video.grab()`` resolves to this object's ``grab``.
    """

    def __init__(self, *, frame_count, dimensions, frames):
        self._frame_count = frame_count
        self._dimensions = dimensions
        self._frames = frames  # maps a frame index to its uint8 array, or to None to model a failed decode
        self.cursor = 0
        self.grab_calls = 0
        self.set_to_frame_calls = []
        self.video = self

    def __len__(self):
        return self._frame_count

    @property
    def dimensions(self):
        return self._dimensions

    def set_to_frame(self, index):
        self.set_to_frame_calls.append(index)
        self.cursor = index

    def grab(self):
        self.grab_calls += 1
        self.cursor += 1

    def read_frame(self, **_read_options):
        # The real reader is called as read_frame(crop=True); the crop flag is irrelevant to this in-memory table.
        frame = self._frames.get(self.cursor)
        self.cursor += 1
        return frame


def _solid_frame(value, *, height=40, width=60):
    """Builds a uniform RGB frame whose grayscale thumbnail collapses to a single, predictable value."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def _make_fake_kmeans(labels, recorder):
    """Returns a MiniBatchKMeans replacement that records its constructor arguments and yields fixed cluster labels."""

    class _FakeKMeans:
        def __init__(self, *, n_clusters, tol, batch_size, max_iter):
            recorder["n_clusters"] = n_clusters
            recorder["tol"] = tol
            recorder["batch_size"] = batch_size
            recorder["max_iter"] = max_iter
            self.labels_ = None

        def fit(self, data):
            recorder["fitted_shape"] = data.shape
            recorder["fitted_data"] = np.array(data, copy=True)
            self.labels_ = np.asarray(labels)
            return self

    return _FakeKMeans


# ----------------------------------------------------------------------------------------------------------------------
# make_fast_kmeans_selector / the installed select closure
# ----------------------------------------------------------------------------------------------------------------------
def test_selector_uses_defaults_and_plain_tqdm(monkeypatch):
    """With no overrides the closure keeps DeepLabCut's cluster count and falls back to a plain tqdm bar."""
    captured = {}

    def fake_select(**kwargs):
        captured.update(kwargs)
        return [1, 2, 3]

    monkeypatch.setattr(frame_reading, "_select_kmeans_frames", fake_select)

    selector = make_fast_kmeans_selector()
    reader = object()
    # DeepLabCut passes reader, cluster count, and window bounds positionally; no keyword options here.
    result = selector(reader, 5, 0.0, 1.0)

    assert result == [1, 2, 3]
    assert captured["video_reader"] is reader
    assert captured["cluster_count"] == 5
    assert captured["window_start"] == 0.0
    assert captured["window_stop"] == 1.0
    assert captured["frame_indices"] is None
    assert captured["sampling_step"] == 1
    assert captured["resize_width"] == 30
    assert captured["batch_size"] == 100
    assert captured["maximum_iterations"] == 50
    assert captured["cluster_in_color"] is False
    assert captured["progress"] is frame_reading.tqdm


def test_selector_overrides_frame_count_and_forwards_options(monkeypatch):
    """An explicit frame count overrides the cluster count and every DeepLabCut keyword option is forwarded."""
    captured = {}

    def fake_select(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(frame_reading, "_select_kmeans_frames", fake_select)

    def my_progress(iterable):
        return iterable

    selector = make_fast_kmeans_selector(progress=my_progress, frame_count=9)
    reader = object()
    selector(
        reader,
        5,
        0.1,
        0.9,
        [1, 2, 3],
        step=2,
        resizewidth=15,
        batchsize=10,
        max_iter=7,
        color=True,
    )

    assert captured["cluster_count"] == 9  # frame_count overrides the DeepLabCut-derived cluster count
    assert captured["window_start"] == 0.1
    assert captured["window_stop"] == 0.9
    assert captured["frame_indices"] == [1, 2, 3]
    assert captured["sampling_step"] == 2
    assert captured["resize_width"] == 15
    assert captured["batch_size"] == 10
    assert captured["maximum_iterations"] == 7
    assert captured["cluster_in_color"] is True
    assert captured["progress"] is my_progress


# ----------------------------------------------------------------------------------------------------------------------
# _resolve_candidate_indices
# ----------------------------------------------------------------------------------------------------------------------
def test_resolve_candidate_indices_samples_full_window_when_none():
    """With no candidate list the whole window is sampled at the requested stride."""
    indices = _resolve_candidate_indices(
        frame_indices=None, frame_count=100, window_start=0.0, window_stop=1.0, sampling_step=2
    )
    assert indices.dtype == np.int64
    assert np.array_equal(indices, np.arange(0, 100, 2, dtype=np.int64))


def test_resolve_candidate_indices_uses_floor_and_ceil_window_bounds():
    """Fractional window bounds floor the start and ceil the stop before sampling."""
    indices = _resolve_candidate_indices(
        frame_indices=None, frame_count=100, window_start=0.1, window_stop=0.5, sampling_step=1
    )
    # The floored start is 10 and the ceiled stop is 50.
    assert np.array_equal(indices, np.arange(10, 50, dtype=np.int64))


def test_resolve_candidate_indices_crops_supplied_list_with_strict_inequalities():
    """Supplied candidate indices are cropped to the open window interval, matching DeepLabCut exactly."""
    indices = _resolve_candidate_indices(
        frame_indices=[0, 5, 50, 99, 100, 150],
        frame_count=100,
        window_start=0.0,
        window_stop=1.0,
        sampling_step=1,
    )
    # start_index == 0 and stop_index == 100, both excluded by the strict comparisons, along with the out-of-range 150.
    assert indices.dtype == np.int64
    assert np.array_equal(indices, np.array([5, 50, 99], dtype=np.int64))


def test_resolve_candidate_indices_crops_supplied_list_inside_subwindow():
    """A sub-window crops a supplied candidate list to the floored/ceiled open interval."""
    indices = _resolve_candidate_indices(
        frame_indices=[5, 10, 11, 49, 50, 60],
        frame_count=100,
        window_start=0.1,
        window_stop=0.5,
        sampling_step=1,
    )
    # Keeps only indices strictly inside (10, 50).
    assert np.array_equal(indices, np.array([11, 49], dtype=np.int64))


# ----------------------------------------------------------------------------------------------------------------------
# _should_stream
# ----------------------------------------------------------------------------------------------------------------------
def test_should_stream_false_when_too_few_candidates():
    """Fewer than two candidates leaves no gap to measure, so the reader always seeks."""
    assert _should_stream(candidate_indices=np.array([], dtype=np.int64)) is False
    assert _should_stream(candidate_indices=np.array([5], dtype=np.int64)) is False


def test_should_stream_true_for_dense_candidates():
    """A small mean gap favors a single sequential streaming pass."""
    assert _should_stream(candidate_indices=np.array([0, 1, 2], dtype=np.int64)) is True
    # Boundary: a mean gap exactly at the crossover still streams.
    assert _should_stream(candidate_indices=np.array([0, 200], dtype=np.int64)) is True


def test_should_stream_false_for_sparse_candidates():
    """A mean gap above the crossover favors seeking to each candidate."""
    assert _should_stream(candidate_indices=np.array([0, 201], dtype=np.int64)) is False


# ----------------------------------------------------------------------------------------------------------------------
# _downsample_frame
# ----------------------------------------------------------------------------------------------------------------------
def test_downsample_frame_grayscale_averages_channels():
    """The grayscale thumbnail downsamples the frame and averages its channels."""
    thumbnail = _downsample_frame(frame=_solid_frame(80), downsample_ratio=0.5, cluster_in_color=False)
    assert thumbnail.shape == (20, 30)
    assert np.allclose(thumbnail, 80.0)


def test_downsample_frame_color_stacks_channels_horizontally():
    """The color thumbnail downsamples the frame and stacks its channels side by side."""
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 20
    frame[:, :, 2] = 30
    thumbnail = _downsample_frame(frame=frame, downsample_ratio=0.5, cluster_in_color=True)
    assert thumbnail.shape == (20, 90)
    assert np.all(thumbnail[:, 0:30] == 10)
    assert np.all(thumbnail[:, 30:60] == 20)
    assert np.all(thumbnail[:, 60:90] == 30)


# ----------------------------------------------------------------------------------------------------------------------
# _read_thumbnails_streaming
# ----------------------------------------------------------------------------------------------------------------------
def test_read_thumbnails_streaming_grabs_gaps_and_skips_failed_decode():
    """Streaming grabs past non-candidate gaps, decodes candidates, and leaves failed-decode rows untouched."""
    frames = {2: _solid_frame(30), 4: None, 5: _solid_frame(200)}
    reader = FakeVideoReader(frame_count=100, dimensions=(60, 40), frames=frames)
    thumbnails = np.full((3, 20, 30), -1.0)

    _read_thumbnails_streaming(
        video_reader=reader,
        candidate_indices=np.array([2, 4, 5], dtype=np.int64),
        downsample_ratio=0.5,
        cluster_in_color=False,
        thumbnails=thumbnails,
        progress=lambda iterable: iterable,
    )

    # Seeks once to the first candidate, then advances by a single grab across the 3 -> 4 gap only.
    assert reader.set_to_frame_calls == [2]
    assert reader.grab_calls == 1
    assert np.all(thumbnails[0] == 30.0)
    assert np.all(thumbnails[2] == 200.0)
    # The None frame at index 4 leaves its row at the pre-filled sentinel.
    assert np.all(thumbnails[1] == -1.0)


# ----------------------------------------------------------------------------------------------------------------------
# _read_thumbnails_seeking
# ----------------------------------------------------------------------------------------------------------------------
def test_read_thumbnails_seeking_seeks_each_candidate_and_skips_failed_decode():
    """Seeking positions to each candidate independently and leaves failed-decode rows untouched."""
    frames = {10: _solid_frame(70), 20: None}
    reader = FakeVideoReader(frame_count=100, dimensions=(60, 40), frames=frames)
    thumbnails = np.full((2, 20, 30), -1.0)

    _read_thumbnails_seeking(
        video_reader=reader,
        candidate_indices=np.array([10, 20], dtype=np.int64),
        downsample_ratio=0.5,
        cluster_in_color=False,
        thumbnails=thumbnails,
        progress=lambda iterable: iterable,
    )

    assert reader.set_to_frame_calls == [10, 20]
    assert reader.grab_calls == 0
    assert np.all(thumbnails[0] == 70.0)
    assert np.all(thumbnails[1] == -1.0)


# ----------------------------------------------------------------------------------------------------------------------
# _cluster_and_pick
# ----------------------------------------------------------------------------------------------------------------------
def test_cluster_and_pick_keeps_configured_batch_and_returns_one_per_singleton_cluster(monkeypatch):
    """When the batch fits the candidates it is kept, and single-member clusters pick their sole member."""
    recorder = {}
    labels = np.array([0, 1, 2, 3, 4])
    monkeypatch.setattr(frame_reading, "MiniBatchKMeans", _make_fake_kmeans(labels, recorder))

    original_thumbnails = np.arange(5 * 2 * 3, dtype=np.float64).reshape(5, 2, 3)
    thumbnails = original_thumbnails.copy()  # the reader centers this buffer in place, so keep an untouched original
    candidate_indices = np.array([10, 20, 30, 40, 50], dtype=np.int64)

    result = _cluster_and_pick(
        thumbnails=thumbnails,
        candidate_indices=candidate_indices,
        cluster_count=5,
        batch_size=2,
        maximum_iterations=7,
    )

    # batch_size 2 <= 5 candidates, so it is used unchanged.
    assert recorder["batch_size"] == 2
    assert recorder["n_clusters"] == 5
    assert recorder["max_iter"] == 7
    assert recorder["fitted_shape"] == (5, 6)  # each thumbnail flattened to a single row
    # The data handed to k-means must be the column-mean-centered, per-row-flattened thumbnails, exactly as DeepLabCut
    # centers before clustering. This pins the in-place centering + reshape, which is otherwise invisible to the fake.
    expected_centered = (original_thumbnails - original_thumbnails.mean(axis=0)).reshape(5, 6)
    assert np.allclose(recorder["fitted_data"], expected_centered)
    assert np.allclose(recorder["fitted_data"].mean(axis=0), 0.0)  # each column is genuinely mean-centered
    # Every cluster has exactly one member, so the pick is forced and returns the candidates in order.
    assert result == [10, 20, 30, 40, 50]


def test_cluster_and_pick_shrinks_oversized_batch_and_drops_empty_cluster(monkeypatch):
    """An oversized batch shrinks to half the candidate count and empty clusters yield no representative."""
    recorder = {}
    # Cluster id 1 receives no members and must be dropped from the result.
    labels = np.array([0, 0, 2, 2, 2])
    monkeypatch.setattr(frame_reading, "MiniBatchKMeans", _make_fake_kmeans(labels, recorder))

    thumbnails = np.arange(5 * 2 * 3, dtype=np.float64).reshape(5, 2, 3)
    candidate_indices = np.array([10, 20, 30, 40, 50], dtype=np.int64)

    result = _cluster_and_pick(
        thumbnails=thumbnails,
        candidate_indices=candidate_indices,
        cluster_count=3,
        batch_size=100,
        maximum_iterations=5,
    )

    # batch_size 100 > 5 candidates, so it shrinks to 5 // 2 == 2.
    assert recorder["batch_size"] == 2
    assert len(result) == 2  # the empty cluster contributes nothing
    assert result[0] in (10, 20)
    assert result[1] in (30, 40, 50)
    assert all(isinstance(index, int) for index in result)


# ----------------------------------------------------------------------------------------------------------------------
# End-to-end tests for the frame-selection entry point _select_kmeans_frames
# ----------------------------------------------------------------------------------------------------------------------
def test_select_kmeans_frames_raises_when_resize_would_upsample():
    """A resize width wider than the frame would upsample, which DeepLabCut and this reader both reject."""
    reader = FakeVideoReader(frame_count=100, dimensions=(20, 40), frames={})
    with pytest.raises(ValueError, match="must not upsample"):
        _select_kmeans_frames(
            video_reader=reader,
            cluster_count=2,
            window_start=0.0,
            window_stop=1.0,
            frame_indices=None,
            sampling_step=1,
            resize_width=30,
            batch_size=100,
            maximum_iterations=10,
            cluster_in_color=False,
            progress=lambda iterable: iterable,
        )


def test_select_kmeans_frames_returns_all_candidates_below_cluster_count():
    """With fewer candidates than clusters the reader returns them all without decoding any frame."""
    reader = FakeVideoReader(frame_count=100, dimensions=(60, 40), frames={})
    result = _select_kmeans_frames(
        video_reader=reader,
        cluster_count=10,
        window_start=0.0,
        window_stop=1.0,
        frame_indices=np.array([5, 6, 7], dtype=np.int64),
        sampling_step=1,
        resize_width=30,
        batch_size=100,
        maximum_iterations=10,
        cluster_in_color=False,
        progress=lambda iterable: iterable,
    )
    assert result == [5, 6, 7]
    assert all(isinstance(index, int) for index in result)
    # No reading occurred because the candidates were returned directly.
    assert reader.set_to_frame_calls == []


def test_select_kmeans_frames_streams_dense_candidates():
    """Dense candidates take the streaming path and cluster down to at most the requested count."""
    rng = np.random.default_rng(0)
    frames = {index: rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8) for index in range(1, 8)}
    reader = FakeVideoReader(frame_count=100, dimensions=(60, 40), frames=frames)

    result = _select_kmeans_frames(
        video_reader=reader,
        cluster_count=3,
        window_start=0.0,
        window_stop=1.0,
        frame_indices=np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        sampling_step=1,
        resize_width=30,
        batch_size=100,
        maximum_iterations=10,
        cluster_in_color=False,
        progress=lambda iterable: iterable,
    )

    # A single seek marks the streaming path; the consecutive candidates never trigger a seek per frame.
    assert reader.set_to_frame_calls == [1]
    assert isinstance(result, list)
    assert 1 <= len(result) <= 3
    assert all(isinstance(index, int) for index in result)
    assert all(index in {1, 2, 3, 4, 5, 6, 7} for index in result)


def test_select_kmeans_frames_seeks_sparse_candidates():
    """Sparse candidates take the seeking path, seeking once per candidate."""
    rng = np.random.default_rng(1)
    frames = {index: rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8) for index in (10, 900)}
    reader = FakeVideoReader(frame_count=1000, dimensions=(60, 40), frames=frames)

    result = _select_kmeans_frames(
        video_reader=reader,
        cluster_count=2,
        window_start=0.0,
        window_stop=1.0,
        frame_indices=np.array([10, 900], dtype=np.int64),
        sampling_step=1,
        resize_width=30,
        batch_size=100,
        maximum_iterations=10,
        cluster_in_color=False,
        progress=lambda iterable: iterable,
    )

    # One seek per candidate marks the seeking path.
    assert reader.set_to_frame_calls == [10, 900]
    assert isinstance(result, list)
    assert 1 <= len(result) <= 2
    assert all(index in {10, 900} for index in result)
