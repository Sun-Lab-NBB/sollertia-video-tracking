"""Contains tests for the multi-device DeepLabCut inference pipeline orchestration.

These tests drive the whole pipeline module without a GPU, network, or real DeepLabCut runtime. Heavy handoffs
(``analyze_videos``, the multiprocessing manager/context, the aggregate progress bar, OpenCV decode) are replaced with
lightweight in-process fakes, and the worker/collector helpers are exercised directly so every branch runs on a
headless CPU box.
"""

import sys
import queue
from pathlib import Path
import contextlib
from collections import deque

import cv2
import numpy as np
import torch
import pytest

from sollertia_video_tracking.inference import pipeline
from sollertia_video_tracking.inference.optimization import InferenceProfile


def make_profile(**overrides):
    """Builds an InferenceProfile with sensible CPU defaults, overriding only the fields a test cares about."""
    defaults = {
        "device": "cpu",
        "gpus": (),
        "gpu_processes": 0,
        "chunks": 1,
        "cpu_workers": 1,
        "cpu_threads_per_worker": 8,
        "amp_dtype": None,
        "tf32": False,
        "cudnn_benchmark": False,
        "channels_last": False,
        "torch_compile": False,
    }
    defaults.update(overrides)
    return InferenceProfile(**defaults)


class _FakeCapture:
    """A minimal stand-in for cv2.VideoCapture that reports fixed header values without opening a file."""

    def __init__(self, *, opened=True, width=100, height=80, frames=50):
        self._opened = opened
        self._width = width
        self._height = height
        self._frames = frames
        self.released = False
        # cv2.VideoCapture exposes a camelCase ``isOpened``; bind it as an attribute so the fake matches the real
        # interface the pipeline calls without a snake_case method-name rename.
        self.isOpened = self._is_opened

    def _is_opened(self):
        return self._opened

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: self._width,
            cv2.CAP_PROP_FRAME_HEIGHT: self._height,
            cv2.CAP_PROP_FRAME_COUNT: self._frames,
        }.get(prop, 0)

    def release(self):
        self.released = True


def _install_capture(monkeypatch, capture_for):
    """Replaces cv2.VideoCapture with a factory that maps a path string to a preconfigured fake capture."""
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture_for(str(path)))


class _FakeQueue:
    """A single-process queue backed by a deque; ``get`` raises queue.Empty instead of blocking when drained."""

    def __init__(self):
        self._items = deque()

    def put(self, item):
        self._items.append(item)

    def get(self, **_kwargs):
        if not self._items:
            raise queue.Empty
        return self._items.popleft()


class _FakeManager:
    """Hands out fresh in-process queues and records whether it was shut down."""

    def __init__(self):
        self.shutdown_called = False

    def Queue(self):  # noqa: N802 - mirrors multiprocessing.Manager.Queue
        return _FakeQueue()

    def shutdown(self):
        self.shutdown_called = True


class _FakeProcess:
    """Runs the simulated worker synchronously on start so no real process is spawned."""

    def __init__(self, target, args, worker_fn):
        self.target = target
        self.args = args
        self._worker_fn = worker_fn

    def start(self):
        self._worker_fn(*self.args)

    def join(self, timeout=None):
        pass


class _FakeContext:
    """A spawn-context stand-in whose Process drives the provided simulated worker."""

    def __init__(self, worker_fn):
        self._worker_fn = worker_fn

    def Process(self, target, args):  # noqa: N802 - mirrors multiprocessing context Process
        return _FakeProcess(target, args, self._worker_fn)


class _FakeMp:
    """Replaces the torch.multiprocessing module reference used by run_inference."""

    def __init__(self, manager, worker_fn):
        self._manager = manager
        self._worker_fn = worker_fn

    def Manager(self):  # noqa: N802 - mirrors multiprocessing.Manager
        return self._manager

    def get_context(self, method):
        assert method == "spawn"
        return _FakeContext(self._worker_fn)


def _install_fake_mp(monkeypatch, worker_fn):
    """Swaps the pipeline's multiprocessing reference for an in-process fake and returns its manager."""
    manager = _FakeManager()
    monkeypatch.setattr(pipeline, "mp", _FakeMp(manager, worker_fn))
    return manager


def _make_launch(**overrides):
    """Builds an _InferenceLaunch with fake queues, overriding only the fields a test needs."""
    defaults = {
        "config": Path("/proj/config.yaml"),
        "shuffle": 1,
        "snapshot_index": None,
        "detector_snapshot_index": None,
        "profile": make_profile(),
        "batch_size": None,
        "detector_batch_size": None,
        "display_progress": False,
        "video_queue": _FakeQueue(),
        "progress_queue": _FakeQueue(),
        "results_queue": _FakeQueue(),
    }
    defaults.update(overrides)
    return pipeline._InferenceLaunch(**defaults)


# InferenceSummary.describe
def test_summary_describe_no_destinations_no_failures():
    """Verifies that describe reports each video's own directory and omits failures when the run had none."""
    summary = pipeline.InferenceSummary(
        config=Path("c"),
        video_count=2,
        destinations=None,
        device="cpu",
        workers=1,
        precision="fp32",
        outputs=(Path("a.h5"), Path("b.h5")),
        failures=(),
    )
    text = summary.describe()
    assert "each video's directory" in text
    assert "analyzed 2/2 videos" in text
    assert "cpu x1" in text
    assert "failed" not in text


def test_summary_describe_single_destination_with_failures():
    """Verifies that describe names the single destination directory and reports the failure count."""
    summary = pipeline.InferenceSummary(
        config=Path("c"),
        video_count=3,
        destinations=(Path("/out"),),
        device="cuda",
        workers=4,
        precision="bfloat16",
        outputs=(Path("a.h5"),),
        failures=(("v2.mp4", "boom"),),
    )
    text = summary.describe()
    assert str(Path("/out")) in text
    assert "1 failed" in text
    assert "cuda x4" in text


def test_summary_describe_multiple_destinations():
    """Verifies that describe summarizes multiple destinations as a per-video-directory count."""
    summary = pipeline.InferenceSummary(
        config=Path("c"),
        video_count=2,
        destinations=(Path("/a"), Path("/b")),
        device="cpu",
        workers=2,
        precision="fp32",
        outputs=(),
        failures=(),
    )
    assert "2 per-video directories" in summary.describe()


# resolve_project_videos
def test_resolve_project_videos_filters_existing(monkeypatch, tmp_path):
    """Verifies that resolve_project_videos keeps only the registered videos that still exist on disk."""
    existing = tmp_path / "exists.mp4"
    existing.write_bytes(b"")
    missing = tmp_path / "gone.mp4"
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"video_sets": {str(existing): {}, str(missing): {}}})
    assert pipeline.resolve_project_videos(str(tmp_path / "cfg.yaml")) == [existing]


def test_resolve_project_videos_no_video_sets(monkeypatch):
    """Verifies that resolve_project_videos returns an empty list when the configuration registers no video sets."""
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {})
    assert pipeline.resolve_project_videos("cfg") == []


# detect_fixed_input_size
def test_detect_fixed_empty_videos():
    """Verifies that detect_fixed_input_size reports not-fixed for an empty video list."""
    assert pipeline.detect_fixed_input_size(config="x", videos=[]) is False


def test_detect_fixed_crop_override_uniform():
    """Verifies that detect_fixed_input_size reports fixed when the per-video crop overrides share one size."""
    # Both rectangles reduce to a 10x20 region, so the run feeds one fixed input size.
    assert (
        pipeline.detect_fixed_input_size(config="x", videos=["a", "b"], crop_override=[(0, 10, 0, 20), (5, 15, 5, 25)])
        is True
    )


def test_detect_fixed_crop_override_mixed():
    """Verifies that detect_fixed_input_size reports not-fixed when the per-video crop overrides differ in size."""
    assert pipeline.detect_fixed_input_size("x", ["a", "b"], crop_override=[(0, 10, 0, 20), (0, 30, 0, 20)]) is False


def test_detect_fixed_from_config_uniform(monkeypatch, tmp_path):
    """Verifies that detect_fixed_input_size reports fixed when all videos share one native resolution."""
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"cropping": False})
    _install_capture(monkeypatch, lambda _p: _FakeCapture(width=100, height=80))
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    assert pipeline.detect_fixed_input_size(config=str(tmp_path / "cfg.yaml"), videos=videos) is True


def test_detect_fixed_from_config_mixed(monkeypatch, tmp_path):
    """Verifies that detect_fixed_input_size reports not-fixed when the videos differ in native resolution."""
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"cropping": False})
    widths = {str(tmp_path / "a.mp4"): 100, str(tmp_path / "b.mp4"): 200}
    _install_capture(monkeypatch, lambda p: _FakeCapture(width=widths[p], height=80))
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    assert pipeline.detect_fixed_input_size(str(tmp_path / "cfg.yaml"), videos) is False


def test_detect_fixed_from_config_unknown_size(monkeypatch, tmp_path):
    """Verifies that detect_fixed_input_size reports not-fixed when a video's dimensions cannot be resolved."""
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"cropping": False})
    opened = {str(tmp_path / "a.mp4"): True, str(tmp_path / "b.mp4"): False}
    _install_capture(monkeypatch, lambda p: _FakeCapture(opened=opened[p], width=100, height=80))
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    # One video's dimensions cannot be resolved (None), so the run is conservatively reported as not fixed.
    assert pipeline.detect_fixed_input_size(str(tmp_path / "cfg.yaml"), videos) is False


def test_detect_fixed_read_config_raises(monkeypatch):
    """Verifies that detect_fixed_input_size reports not-fixed when reading the configuration raises."""

    def boom(_config):
        message = "nope"
        raise RuntimeError(message)

    monkeypatch.setattr(pipeline, "read_config", boom)
    assert pipeline.detect_fixed_input_size("cfg", ["a"]) is False


# _resolve_input_size / _probe_frame_size / _probe_frame_count
def test_resolve_input_size_cropping_with_crop(tmp_path):
    """Verifies that _resolve_input_size returns the crop rectangle's size when cropping is configured."""
    video = tmp_path / "v.mp4"
    cfg = {"cropping": True, "video_sets": {str(video): {"crop": "0,10,0,20"}}}
    assert pipeline._resolve_input_size(project_config=cfg, video=video) == (10, 20)


def test_resolve_input_size_cropping_unresolved(tmp_path):
    """Verifies that _resolve_input_size returns None when the configured crop cannot be resolved."""
    cfg = {"cropping": True, "video_sets": {}}
    assert pipeline._resolve_input_size(project_config=cfg, video=tmp_path / "v.mp4") is None


def test_resolve_input_size_no_cropping(monkeypatch, tmp_path):
    """Verifies that _resolve_input_size returns the video's native resolution when cropping is disabled."""
    _install_capture(monkeypatch, lambda _p: _FakeCapture(width=320, height=240))
    assert pipeline._resolve_input_size(project_config={"cropping": False}, video=tmp_path / "v.mp4") == (320, 240)


def test_probe_frame_size_ok(monkeypatch, tmp_path):
    """Verifies that _probe_frame_size returns the frame dimensions reported by the container header."""
    _install_capture(monkeypatch, lambda _p: _FakeCapture(width=640, height=480))
    assert pipeline._probe_frame_size(tmp_path / "v.mp4") == (640, 480)


def test_probe_frame_size_not_opened(monkeypatch, tmp_path):
    """Verifies that _probe_frame_size returns None when the video cannot be opened."""
    _install_capture(monkeypatch, lambda _p: _FakeCapture(opened=False))
    assert pipeline._probe_frame_size(tmp_path / "v.mp4") is None


def test_probe_frame_size_nonpositive_dimension(monkeypatch, tmp_path):
    """Verifies that _probe_frame_size returns None when the container reports a non-positive dimension."""
    _install_capture(monkeypatch, lambda _p: _FakeCapture(width=0, height=480))
    assert pipeline._probe_frame_size(tmp_path / "v.mp4") is None


def test_probe_frame_count_ok(monkeypatch, tmp_path):
    """Verifies that _probe_frame_count returns the frame count reported by the container header."""
    _install_capture(monkeypatch, lambda _p: _FakeCapture(frames=7))
    assert pipeline._probe_frame_count(tmp_path / "v.mp4") == 7


def test_probe_frame_count_not_opened_clamped(monkeypatch, tmp_path):
    """Verifies that _probe_frame_count clamps a closed capture's frame count to at least one."""
    _install_capture(monkeypatch, lambda _p: _FakeCapture(opened=False, frames=99))
    # A closed capture reports zero frames, clamped to at least one for the progress bar.
    assert pipeline._probe_frame_count(tmp_path / "v.mp4") == 1


# _parse_crop / _resolve_video_cropping
def test_parse_crop_none():
    """Verifies that _parse_crop returns None for a missing crop specification."""
    assert pipeline._parse_crop(None) is None


def test_parse_crop_valid():
    """Verifies that _parse_crop parses a four-integer crop specification into a list of integers."""
    assert pipeline._parse_crop("1, 2, 3, 4") == [1, 2, 3, 4]


def test_parse_crop_wrong_field_count():
    """Verifies that _parse_crop returns None when the crop specification has the wrong field count."""
    assert pipeline._parse_crop("1,2,3") is None


def test_parse_crop_non_integer():
    """Verifies that _parse_crop returns None when the crop specification contains non-integer fields."""
    assert pipeline._parse_crop("a,b,c,d") is None


def test_resolve_video_cropping_disabled():
    """Verifies that _resolve_video_cropping returns None when the project is not configured to crop."""
    assert pipeline._resolve_video_cropping(project_config={"cropping": False}, video="/x.mp4") is None


def test_resolve_video_cropping_registered_crop(tmp_path):
    """Verifies that _resolve_video_cropping returns a registered video's own crop rectangle."""
    video = tmp_path / "v.mp4"
    cfg = {"cropping": True, "video_sets": {str(video): {"crop": "1,2,3,4"}}}
    assert pipeline._resolve_video_cropping(project_config=cfg, video=str(video)) == [1, 2, 3, 4]


def test_resolve_video_cropping_registered_no_crop_falls_back_to_corners(tmp_path):
    """Verifies that _resolve_video_cropping uses the project-wide rectangle for a registered video with no crop."""
    video = tmp_path / "v.mp4"
    # The video is registered but carries no parseable crop, so the project-wide rectangle is used.
    cfg = {"cropping": True, "video_sets": {str(video): {"other": 1}}, "x1": 0, "x2": 100, "y1": 5, "y2": 80}
    assert pipeline._resolve_video_cropping(project_config=cfg, video=str(video)) == [0, 100, 5, 80]


def test_resolve_video_cropping_non_dict_metadata_skipped(tmp_path):
    """Verifies that _resolve_video_cropping skips a non-dict video_sets entry and uses the project-wide corners."""
    video = tmp_path / "v.mp4"
    # A non-dict video_sets entry is skipped, and the project-wide corners are used instead.
    cfg = {"cropping": True, "video_sets": {str(video): ["not", "a", "dict"]}, "x1": 1, "x2": 2, "y1": 3, "y2": 4}
    assert pipeline._resolve_video_cropping(project_config=cfg, video=str(video)) == [1, 2, 3, 4]


def test_resolve_video_cropping_missing_corner_returns_none():
    """Verifies that _resolve_video_cropping returns None when a project-wide corner is missing."""
    cfg = {"cropping": True, "video_sets": {}, "x1": 0, "x2": 100, "y1": 0}  # y2 absent
    assert pipeline._resolve_video_cropping(project_config=cfg, video="/x.mp4") is None


# _describe_precision
def test_describe_precision_fp32():
    """Verifies that _describe_precision labels a profile with no autocast dtype as fp32."""
    assert pipeline._describe_precision(make_profile(amp_dtype=None)) == "fp32"


def test_describe_precision_bfloat16():
    """Verifies that _describe_precision labels a bfloat16 autocast profile as bfloat16."""
    assert pipeline._describe_precision(make_profile(amp_dtype=torch.bfloat16)) == "bfloat16"


# _build_slots / _usable_cpu_cores
def test_build_slots_cuda_round_robin():
    """Verifies that _build_slots round-robins the CUDA worker slots across the available GPUs."""
    profile = make_profile(device="cuda", gpus=(0, 1), gpu_processes=2)
    slots = pipeline._build_slots(profile=profile, video_count=10)
    assert [slot.device for slot in slots] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    assert all(slot.cores is None for slot in slots)


def test_build_slots_cuda_truncated_to_video_count():
    """Verifies that _build_slots truncates the CUDA slot list to the video count."""
    profile = make_profile(device="cuda", gpus=(0, 1), gpu_processes=2)
    slots = pipeline._build_slots(profile=profile, video_count=1)
    assert len(slots) == 1
    assert slots[0].device == "cuda:0"


def test_build_slots_cuda_no_gpus_raises():
    """Verifies that _build_slots raises when CUDA is selected but no GPU indices are resolved."""
    profile = make_profile(device="cuda", gpus=(), gpu_processes=1)
    with pytest.raises(ValueError, match="no GPU indices"):
        pipeline._build_slots(profile=profile, video_count=3)


def test_build_slots_mps_single_slot():
    """Verifies that _build_slots builds a single unpinned slot for the MPS device."""
    profile = make_profile(device="mps")
    slots = pipeline._build_slots(profile=profile, video_count=5)
    assert len(slots) == 1
    assert slots[0].device == "mps"
    assert slots[0].cores is None


def test_build_slots_cpu(monkeypatch):
    """Verifies that _build_slots pins the sole CPU worker to its planned two-core block."""
    # Pin the topology so a single worker with a definite two-core block is planned deterministically. With 16 cores
    # (logical and physical), one worker, and two threads/worker, _usable_cpu_cores is 2, so 14 cores are reserved and
    # exactly two usable cores (0 and 1) are pinned to the sole worker.
    monkeypatch.setattr(pipeline.psutil, "cpu_count", lambda **_kwargs: 16)
    profile = make_profile(device="cpu", cpu_workers=1, cpu_threads_per_worker=2)
    slots = pipeline._build_slots(profile=profile, video_count=1)
    assert len(slots) == 1
    assert slots[0].device == "cpu"
    # The exact pinned block, not merely a non-empty tuple, so a wrong core plan is caught.
    assert slots[0].cores == (0, 1)


def test_usable_cpu_cores_with_threads(monkeypatch):
    """Verifies that _usable_cpu_cores bounds the worker-thread product by the physical core count."""
    monkeypatch.setattr(pipeline.psutil, "cpu_count", lambda **_kwargs: 16)
    # min(16 physical, 2 workers * 4 threads) = 8.
    assert pipeline._usable_cpu_cores(make_profile(cpu_workers=2, cpu_threads_per_worker=4)) == 8


def test_usable_cpu_cores_threads_none_and_zero_workers(monkeypatch):
    """Verifies that _usable_cpu_cores clamps None threads and zero workers to a product of one."""
    monkeypatch.setattr(pipeline.psutil, "cpu_count", lambda **_kwargs: 16)
    # None threads -> 1, zero workers -> max(1, 0) = 1, so the product is 1.
    assert pipeline._usable_cpu_cores(make_profile(cpu_workers=0, cpu_threads_per_worker=None)) == 1


# _collect_results
def test_collect_results_success_and_failures():
    """Verifies that _collect_results separates successful outputs from explicit and empty-file failures."""
    results_queue = _FakeQueue()
    results_queue.put((0, "/out/a.h5", None))  # success
    results_queue.put((1, None, "RuntimeError: boom"))  # explicit error
    results_queue.put((2, None, None))  # reported but produced no file
    video_paths = [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")]
    outputs, failures = pipeline._collect_results(results_queue=results_queue, video_paths=video_paths)
    assert outputs == {0: Path("/out/a.h5")}
    assert ("b.mp4", "RuntimeError: boom") in failures
    assert ("c.mp4", "no prediction file was produced") in failures
    assert len(failures) == 2


def test_collect_results_worker_died_missing_result():
    """Verifies that _collect_results reports a never-arriving result as a worker-exited failure."""
    results_queue = _FakeQueue()
    results_queue.put((0, "/out/a.h5", None))
    video_paths = [Path("a.mp4"), Path("b.mp4")]
    # The second result never arrives, so the drain loop times out and the missing video is reported as a failure.
    outputs, failures = pipeline._collect_results(results_queue=results_queue, video_paths=video_paths)
    assert outputs == {0: Path("/out/a.h5")}
    assert failures == [("b.mp4", "the worker process exited before reporting a result")]


# _suppress_stdout
def test_suppress_stdout_inactive_passes_through(capsys):
    """Verifies that _suppress_stdout passes standard output through when inactive."""
    with pipeline._suppress_stdout(active=False):
        print("visible")
    assert "visible" in capsys.readouterr().out


def test_suppress_stdout_active_redirects(capsys):
    """Verifies that _suppress_stdout redirects standard output away from the console when active."""
    with pipeline._suppress_stdout(active=True):
        print("hidden")
    assert "hidden" not in capsys.readouterr().out


# _resolve_output
def test_resolve_output_exact(tmp_path):
    """Verifies that _resolve_output returns the exact per-frame prediction file when it exists."""
    (tmp_path / "clipScorer.h5").write_bytes(b"")
    out = pipeline._resolve_output(video=str(tmp_path / "clip.mp4"), scorer="Scorer", destination=tmp_path)
    assert out == tmp_path / "clipScorer.h5"


def test_resolve_output_glob_suffixed(tmp_path):
    """Verifies that _resolve_output picks the last tracker-suffixed prediction file when no plain file exists."""
    # No plain per-frame file exists; a tracker-suffixed file is picked as the last lexicographic match.
    (tmp_path / "clipScorer_bx.h5").write_bytes(b"")
    (tmp_path / "clipScorer_el.h5").write_bytes(b"")
    out = pipeline._resolve_output(video=str(tmp_path / "clip.mp4"), scorer="Scorer", destination=tmp_path)
    assert out == tmp_path / "clipScorer_el.h5"


def test_resolve_output_none(tmp_path):
    """Verifies that _resolve_output returns None when no prediction file was written."""
    out = pipeline._resolve_output(video=str(tmp_path / "clip.mp4"), scorer="Scorer", destination=tmp_path)
    assert out is None


# _run_inference_worker
def test_run_inference_worker_drains_queue_and_pins_cores(monkeypatch):
    """Verifies that _run_inference_worker pins its cores and drains every work item from the queue.

    macOS exposes no CPU-affinity API, so its workers drain the queue but run unpinned.
    """
    affinity_calls = []

    class _FakePsutilProcess:
        def cpu_affinity(self, cores):
            affinity_calls.append(cores)

    monkeypatch.setattr(pipeline.psutil, "Process", _FakePsutilProcess)
    applied = []
    monkeypatch.setattr(pipeline, "apply_runtime_optimizations", applied.append)
    monkeypatch.setattr(pipeline, "patch_dlc_runner_builders", lambda _profile: contextlib.nullcontext())
    analyzed = []
    monkeypatch.setattr(pipeline, "_analyze_one_video", lambda item, **_kwargs: analyzed.append(item))

    video_queue = _FakeQueue()
    video_queue.put((0, "/v0.mp4", 10, None, None))
    video_queue.put((1, "/v1.mp4", 10, None, None))
    video_queue.put(None)
    launch = _make_launch(video_queue=video_queue, profile=make_profile())
    slot = pipeline._Slot(device="cpu", cores=(0, 1))

    pipeline._run_inference_worker(slot, launch)

    assert affinity_calls == ([] if sys.platform == "darwin" else [[0, 1]])
    assert len(applied) == 1
    assert analyzed == [(0, "/v0.mp4", 10, None, None), (1, "/v1.mp4", 10, None, None)]


def test_run_inference_worker_no_cores_skips_affinity(monkeypatch):
    """Verifies that _run_inference_worker skips CPU affinity when its slot pins no cores."""
    affinity_calls = []

    class _FakePsutilProcess:
        def cpu_affinity(self, cores):
            affinity_calls.append(cores)

    monkeypatch.setattr(pipeline.psutil, "Process", _FakePsutilProcess)
    monkeypatch.setattr(pipeline, "apply_runtime_optimizations", lambda _profile: None)
    monkeypatch.setattr(pipeline, "patch_dlc_runner_builders", lambda _profile: contextlib.nullcontext())
    monkeypatch.setattr(pipeline, "_analyze_one_video", lambda **_kwargs: None)

    video_queue = _FakeQueue()
    video_queue.put(None)
    launch = _make_launch(video_queue=video_queue)
    slot = pipeline._Slot(device="mps", cores=None)

    pipeline._run_inference_worker(slot, launch)

    assert affinity_calls == []


def test_run_inference_worker_tolerates_affinity_failure(monkeypatch):
    """Verifies that _run_inference_worker tolerates a failing cpu_affinity and still drains its work."""

    class _FakePsutilProcess:
        def cpu_affinity(self, _cores):
            message = "affinity unavailable"
            raise OSError(message)

    monkeypatch.setattr(pipeline.psutil, "Process", _FakePsutilProcess)
    applied = []
    monkeypatch.setattr(pipeline, "apply_runtime_optimizations", applied.append)
    monkeypatch.setattr(pipeline, "patch_dlc_runner_builders", lambda _profile: contextlib.nullcontext())
    analyzed = []
    monkeypatch.setattr(pipeline, "_analyze_one_video", lambda item, **_kwargs: analyzed.append(item))

    video_queue = _FakeQueue()
    video_queue.put((0, "/v0.mp4", 10, None, None))
    video_queue.put(None)
    launch = _make_launch(video_queue=video_queue)
    slot = pipeline._Slot(device="cpu", cores=(0,))

    pipeline._run_inference_worker(slot, launch)

    # A failing cpu_affinity is suppressed, so the worker still applies runtime optimizations and drains the real
    # work item rather than aborting when pinning fails. Asserting the drained item catches a swallowed break.
    assert len(applied) == 1
    assert analyzed == [(0, "/v0.mp4", 10, None, None)]


# _analyze_one_video
def test_analyze_one_video_success_with_progress(monkeypatch, tmp_path):
    """Verifies that _analyze_one_video reports the resolved output and publishes a completion marker on success."""
    destination = tmp_path / "out"
    destination.mkdir()
    scorer = "DLCscorer"
    video = tmp_path / "clip.mp4"
    (destination / f"clip{scorer}.h5").write_bytes(b"")

    captured = {}

    def fake_analyze(**kwargs):
        captured.update(kwargs)
        return scorer

    monkeypatch.setattr(pipeline.dlc_videos, "analyze_videos", fake_analyze)
    monkeypatch.setattr(pipeline.dlc_videos, "tqdm", object(), raising=False)

    results_queue = _FakeQueue()
    progress_queue = _FakeQueue()
    launch = _make_launch(
        display_progress=True,
        results_queue=results_queue,
        progress_queue=progress_queue,
        config=tmp_path / "config.yaml",
    )
    slot = pipeline._Slot(device="cpu", cores=None)
    item = (3, str(video), 100, [0, 10, 0, 20], str(destination))

    pipeline._analyze_one_video(slot=slot, launch=launch, item=item)

    index, path, error = results_queue._items.popleft()
    assert index == 3
    assert error is None
    assert Path(path) == destination / f"clip{scorer}.h5"
    # The completion marker is published to the progress queue.
    assert progress_queue._items[-1] == ("done", 3)
    # analyze_videos received the resolved crop and the always-overwrite/acceleration-disabled settings.
    assert captured["cropping"] == [0, 10, 0, 20]
    assert captured["overwrite"] is True
    assert captured["destfolder"] == str(destination)
    assert captured["inference_cfg"] == pipeline._STOCK_ACCELERATION_DISABLED
    assert captured["shuffle"] == 1
    assert captured["device"] == "cpu"


def test_analyze_one_video_failure_without_progress(monkeypatch, tmp_path):
    """Verifies that _analyze_one_video reports an analysis failure as an error and emits no completion marker."""
    video = tmp_path / "sub" / "clip.mp4"
    video.parent.mkdir(parents=True)

    def boom(**kwargs):
        message = "bad"
        raise RuntimeError(message)

    monkeypatch.setattr(pipeline.dlc_videos, "analyze_videos", boom)
    monkeypatch.setattr(pipeline.dlc_videos, "tqdm", object(), raising=False)

    results_queue = _FakeQueue()
    progress_queue = _FakeQueue()
    launch = _make_launch(display_progress=False, results_queue=results_queue, progress_queue=progress_queue)
    slot = pipeline._Slot(device="cpu", cores=None)
    # A None destination writes beside the video, and the analysis failure is reported as an error.
    item = (0, str(video), 50, None, None)

    pipeline._analyze_one_video(slot=slot, launch=launch, item=item)

    index, path, error = results_queue._items.popleft()
    assert index == 0
    assert path is None
    assert error == "RuntimeError: bad"
    # No completion marker is emitted when progress display is off.
    assert len(progress_queue._items) == 0


def test_analyze_one_video_success_without_output_file(monkeypatch, tmp_path):
    """Verifies that _analyze_one_video reports a None output when analysis produces no prediction file."""
    destination = tmp_path / "out"
    destination.mkdir()
    monkeypatch.setattr(pipeline.dlc_videos, "analyze_videos", lambda **kwargs: "Scorer")
    monkeypatch.setattr(pipeline.dlc_videos, "tqdm", object(), raising=False)

    results_queue = _FakeQueue()
    launch = _make_launch(display_progress=False, results_queue=results_queue)
    slot = pipeline._Slot(device="cpu", cores=None)
    item = (2, str(tmp_path / "v.mp4"), 10, None, str(destination))

    pipeline._analyze_one_video(slot=slot, launch=launch, item=item)

    index, path, error = results_queue._items.popleft()
    assert index == 2
    assert path is None  # no file was produced, so the output resolves to None
    assert error is None


# run_inference input validation
def test_run_inference_empty_videos_raises():
    """Verifies that run_inference raises when given an empty video list."""
    with pytest.raises(ValueError, match="at least one video"):
        pipeline.run_inference(config="cfg", videos=[], profile=make_profile())


def test_run_inference_crop_override_length_mismatch_raises():
    """Verifies that run_inference raises when the crop override length does not match the video count."""
    with pytest.raises(ValueError, match="one crop rectangle per video"):
        pipeline.run_inference(config="cfg", videos=["a", "b"], profile=make_profile(), crop_override=[(0, 10, 0, 20)])


def test_run_inference_destination_override_length_mismatch_raises():
    """Verifies that run_inference raises when the destination override length does not match the video count."""
    with pytest.raises(ValueError, match="one output directory per video"):
        pipeline.run_inference(config="cfg", videos=["a", "b"], profile=make_profile(), destination_override=["d1"])


# run_inference orchestration
def test_run_inference_full_success_with_overrides(monkeypatch, tmp_path):
    """Verifies that run_inference forwards the crop and destination overrides and summarizes a fully successful run."""
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    destination_a = tmp_path / "outa"
    destination_b = tmp_path / "outb"
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"cropping": False})
    _install_capture(monkeypatch, lambda _p: _FakeCapture(frames=30))
    monkeypatch.setattr(
        pipeline,
        "_build_slots",
        lambda **_kwargs: [pipeline._Slot(device="cuda:0", cores=None), pipeline._Slot(device="cuda:1", cores=None)],
    )

    bars = []

    class _FakeBar:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = self.stopped = self.joined = False
            bars.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def join(self, **_kwargs):
            self.joined = True

    monkeypatch.setattr(pipeline, "AggregateBar", _FakeBar)

    def worker(_slot, launch):
        while True:
            item = launch.video_queue.get()
            if item is None:
                break
            index, _video, _total, crop, destination = item
            assert crop == [0, 10, 0, 20]  # supplied crop override is forwarded verbatim
            launch.results_queue.put((index, f"{destination}/pred_{index}.h5", None))

    manager = _install_fake_mp(monkeypatch, worker)

    profile = make_profile(device="cuda", gpus=(0, 1), gpu_processes=1, amp_dtype=torch.bfloat16)
    summary = pipeline.run_inference(
        config=str(tmp_path / "cfg.yaml"),
        videos=videos,
        profile=profile,
        destination_override=[destination_a, destination_b],
        crop_override=[(0, 10, 0, 20), (0, 10, 0, 20)],
        display_progress=True,
    )

    assert summary.video_count == 2
    assert summary.device == "cuda"
    assert summary.workers == 2
    assert summary.precision == "bfloat16"
    assert summary.destinations == (destination_a, destination_b)
    assert len(summary.outputs) == 2
    assert summary.failures == ()
    assert destination_a.is_dir()
    assert destination_b.is_dir()
    assert bars
    assert bars[0].started
    assert bars[0].stopped
    assert bars[0].joined
    assert manager.shutdown_called is True


def test_run_inference_partial_failure_no_overrides(monkeypatch, tmp_path):
    """Verifies that run_inference resolves crops from the project configuration and reports a partial failure."""
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    project_config = {"cropping": True, "video_sets": {}, "x1": 0, "x2": 100, "y1": 0, "y2": 80}
    monkeypatch.setattr(pipeline, "read_config", lambda _c: project_config)
    _install_capture(monkeypatch, lambda _p: _FakeCapture(frames=10))
    monkeypatch.setattr(
        pipeline,
        "_build_slots",
        lambda **_kwargs: [pipeline._Slot(device="cpu", cores=(0, 1))],
    )

    def worker(_slot, launch):
        while True:
            item = launch.video_queue.get()
            if item is None:
                break
            index, video, _total, crop, destination = item
            assert crop == [0, 100, 0, 80]  # resolved from the project's project-wide crop rectangle
            assert destination is None  # no destination override means write beside the video
            if index == 0:
                launch.results_queue.put((index, f"{video}.h5", None))
            else:
                launch.results_queue.put((index, None, "RuntimeError: kaboom"))

    manager = _install_fake_mp(monkeypatch, worker)

    profile = make_profile(device="cpu", amp_dtype=None)
    summary = pipeline.run_inference(
        config=str(tmp_path / "cfg.yaml"),
        videos=videos,
        profile=profile,
        display_progress=False,
    )

    assert summary.destinations is None
    assert summary.device == "cpu"
    assert summary.workers == 1
    assert summary.precision == "fp32"
    assert len(summary.outputs) == 1
    assert len(summary.failures) == 1
    assert summary.failures[0][0] == "b.mp4"
    assert manager.shutdown_called is True


# _partition_frame_ranges
def test_partition_single_chunk_covers_whole_video():
    """Verifies that a single chunk yields one range spanning the whole video."""
    assert pipeline._partition_frame_ranges(total_frames=100, chunks=1) == [(0, 100)]


def test_partition_chunks_below_one_collapse_to_whole():
    """Verifies that a zero chunk count collapses to one whole-video range rather than producing no ranges."""
    assert pipeline._partition_frame_ranges(total_frames=100, chunks=0) == [(0, 100)]


def test_partition_even_split_is_exact():
    """Verifies that an evenly divisible total splits into equal contiguous ranges."""
    assert pipeline._partition_frame_ranges(total_frames=100, chunks=4) == [(0, 25), (25, 50), (50, 75), (75, 100)]


def test_partition_remainder_goes_to_earliest_chunks():
    """Verifies that an uneven split gives the extra frames to the earliest chunks."""
    ranges = pipeline._partition_frame_ranges(total_frames=10, chunks=4)
    assert ranges == [(0, 3), (3, 6), (6, 8), (8, 10)]
    assert [end - start for start, end in ranges] == [3, 3, 2, 2]


def test_partition_is_contiguous_gapless_and_exact():
    """Verifies that the ranges are contiguous, gapless, non-empty, balanced, and exactly cover the whole video."""
    total, chunks = 253, 7
    ranges = pipeline._partition_frame_ranges(total_frames=total, chunks=chunks)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == total
    assert all(start < end for start, end in ranges)
    assert all(ranges[index][1] == ranges[index + 1][0] for index in range(len(ranges) - 1))
    assert sum(end - start for start, end in ranges) == total
    assert max(end - start for start, end in ranges) - min(end - start for start, end in ranges) <= 1


def test_partition_more_chunks_than_frames_caps_at_frame_count():
    """Verifies that requesting more chunks than frames yields one single-frame range per frame, never an empty one."""
    assert pipeline._partition_frame_ranges(total_frames=3, chunks=5) == [(0, 1), (1, 2), (2, 3)]


def test_partition_single_frame_video():
    """Verifies that a one-frame video yields a single unit range regardless of the requested chunk count."""
    assert pipeline._partition_frame_ranges(total_frames=1, chunks=4) == [(0, 1)]


# _BoundedVideoIterator
def _write_identifiable_clip(path, *, frames, width=64, height=48):
    """Writes a short clip whose every frame is a distinct solid gray level, for seek-accuracy checks.

    Each frame is a solid level derived from its index, so a frame read by seeking can be matched against the same
    frame read sequentially. The all-intra Motion JPEG codec is used so every frame is independently seekable, and the
    test skips when the environment ships no usable encoder for it.

    Args:
        path: The path to write the clip to, using an AVI container the Motion JPEG codec supports.
        frames: The number of frames to write, kept small so the per-frame level stays within one byte.
        width: The frame width in pixels.
        height: The frame height in pixels.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
    if not writer.isOpened():
        writer.release()
        pytest.skip("no Motion JPEG encoder is available to build the seek-accuracy fixture")
    for index in range(frames):
        writer.write(np.full((height, width, 3), index * 5 % 256, dtype=np.uint8))
    writer.release()


def _sequential_frame_levels(clip) -> list[int]:
    """Reads every frame of a clip sequentially, returning each frame's sampled level as the seek ground truth."""
    capture = cv2.VideoCapture(str(clip))
    levels: list[int] = []
    while True:
        read, frame = capture.read()
        if not read:
            break
        levels.append(int(frame[0, 0, 0]))
    capture.release()
    return levels


def test_bounded_iterator_reads_only_its_range(tmp_path):
    """Verifies that a bounded iterator seeks to its start and emits exactly the frames a sequential read gives."""
    clip = tmp_path / "clip.avi"
    _write_identifiable_clip(clip, frames=50)
    reference = _sequential_frame_levels(clip)
    iterator = pipeline._BoundedVideoIterator(str(clip), frame_start=20, frame_end=35)
    levels = [int(frame[0, 0, 0]) for frame in iterator]
    assert levels == reference[20:35]
    assert iterator.get_n_frames() == 15


def test_bounded_iterator_partition_reassembles_whole_video(tmp_path):
    """Verifies that reading every partition range by seek reproduces the whole-video frame sequence exactly."""
    clip = tmp_path / "clip.avi"
    total = 50
    _write_identifiable_clip(clip, frames=total)
    reference = _sequential_frame_levels(clip)
    ranges = pipeline._partition_frame_ranges(total_frames=total, chunks=5)
    stitched = [
        int(frame[0, 0, 0])
        for start, end in ranges
        for frame in pipeline._BoundedVideoIterator(str(clip), frame_start=start, frame_end=end)
    ]
    assert stitched == reference[:total]


# _collect_chunk_results
def _chunk_item(task_id, video_index, chunk_index, video, frame_start, frame_end):
    """Builds a _ChunkItem for the collector tests with the crop and destination left unset."""
    return pipeline._ChunkItem(
        task_id=task_id,
        video_index=video_index,
        chunk_index=chunk_index,
        video=video,
        frame_start=frame_start,
        frame_end=frame_end,
        crop=None,
        destination=None,
    )


def test_collect_chunk_results_stitches_chunks_in_frame_order(monkeypatch):
    """Verifies that per-chunk predictions are concatenated in ascending chunk order and written once per video."""
    work_items = [
        _chunk_item(0, 0, 0, "a.mp4", 0, 2),
        _chunk_item(1, 0, 1, "a.mp4", 2, 4),
        _chunk_item(2, 1, 0, "b.mp4", 0, 3),
    ]
    results_queue = _FakeQueue()
    # Report out of arrival order to prove the collector orders by chunk index, not by which worker finished first.
    results_queue.put((1, 0, 1, ["a2", "a3"], None))
    results_queue.put((2, 1, 0, ["b0", "b1", "b2"], None))
    results_queue.put((0, 0, 0, ["a0", "a1"], None))

    stitched: dict[str, list] = {}

    def fake_stitch(*, video, predictions, **_kwargs):
        stitched[video] = list(predictions)
        return Path(video).with_suffix(".h5")

    monkeypatch.setattr(pipeline, "_stitch_and_write", fake_stitch)
    outputs, failures = pipeline._collect_chunk_results(
        results_queue=results_queue,
        video_paths=[Path("a.mp4"), Path("b.mp4")],
        work_items=work_items,
        plan=None,
    )
    assert failures == []
    assert stitched["a.mp4"] == ["a0", "a1", "a2", "a3"]
    assert stitched["b.mp4"] == ["b0", "b1", "b2"]
    assert outputs == {0: Path("a.h5"), 1: Path("b.h5")}


def test_collect_chunk_results_reports_failed_chunk(monkeypatch):
    """Verifies that a video with any errored chunk is failed and never stitched, while other videos still write."""
    work_items = [
        _chunk_item(0, 0, 0, "a.mp4", 0, 2),
        _chunk_item(1, 0, 1, "a.mp4", 2, 4),
        _chunk_item(2, 1, 0, "b.mp4", 0, 2),
    ]
    results_queue = _FakeQueue()
    results_queue.put((0, 0, 0, ["a0", "a1"], None))
    results_queue.put((1, 0, 1, None, "RuntimeError: boom"))
    results_queue.put((2, 1, 0, ["b0", "b1"], None))

    monkeypatch.setattr(pipeline, "_stitch_and_write", lambda **kwargs: Path(kwargs["video"]).with_suffix(".h5"))
    outputs, failures = pipeline._collect_chunk_results(
        results_queue=results_queue,
        video_paths=[Path("a.mp4"), Path("b.mp4")],
        work_items=work_items,
        plan=None,
    )
    assert outputs == {1: Path("b.h5")}
    assert failures == [("a.mp4", "RuntimeError: boom")]
