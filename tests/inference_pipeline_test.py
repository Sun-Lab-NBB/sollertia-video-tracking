"""Tests for the multi-device DeepLabCut inference pipeline orchestration.

These tests drive the whole pipeline module without a GPU, network, or real DeepLabCut runtime. Heavy handoffs
(``analyze_videos``, the multiprocessing manager/context, the aggregate progress bar, OpenCV decode) are replaced with
lightweight in-process fakes, and the worker/collector helpers are exercised directly so every branch runs on a
headless CPU box.
"""

import queue
from pathlib import Path
import contextlib
from collections import deque

import cv2
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
        "cpu_workers": 1,
        "cpu_threads_per_worker": 8,
        "amp_dtype": None,
        "tf32": False,
        "cudnn_benchmark": False,
        "channels_last": False,
        "torch_compile": False,
        "pin_memory": False,
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


# --------------------------------------------------------------------------------------------------------------------
# InferenceSummary.describe
# --------------------------------------------------------------------------------------------------------------------


def test_summary_describe_no_destinations_no_failures():
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
    assert "/out" in text
    assert "1 failed" in text
    assert "cuda x4" in text


def test_summary_describe_multiple_destinations():
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


# --------------------------------------------------------------------------------------------------------------------
# resolve_project_videos
# --------------------------------------------------------------------------------------------------------------------


def test_resolve_project_videos_filters_existing(monkeypatch, tmp_path):
    existing = tmp_path / "exists.mp4"
    existing.write_bytes(b"")
    missing = tmp_path / "gone.mp4"
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"video_sets": {str(existing): {}, str(missing): {}}})
    assert pipeline.resolve_project_videos(str(tmp_path / "cfg.yaml")) == [existing]


def test_resolve_project_videos_no_video_sets(monkeypatch):
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {})
    assert pipeline.resolve_project_videos("cfg") == []


# --------------------------------------------------------------------------------------------------------------------
# detect_fixed_input_size
# --------------------------------------------------------------------------------------------------------------------


def test_detect_fixed_empty_videos():
    assert pipeline.detect_fixed_input_size(config="x", videos=[]) is False


def test_detect_fixed_crop_override_uniform():
    # Both rectangles reduce to a 10x20 region, so the run feeds one fixed input size.
    assert pipeline.detect_fixed_input_size("x", ["a", "b"], crop_override=[(0, 10, 0, 20), (5, 15, 5, 25)]) is True


def test_detect_fixed_crop_override_mixed():
    assert pipeline.detect_fixed_input_size("x", ["a", "b"], crop_override=[(0, 10, 0, 20), (0, 30, 0, 20)]) is False


def test_detect_fixed_from_config_uniform(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"cropping": False})
    _install_capture(monkeypatch, lambda _p: _FakeCapture(width=100, height=80))
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    assert pipeline.detect_fixed_input_size(str(tmp_path / "cfg.yaml"), videos) is True


def test_detect_fixed_from_config_mixed(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"cropping": False})
    widths = {str(tmp_path / "a.mp4"): 100, str(tmp_path / "b.mp4"): 200}
    _install_capture(monkeypatch, lambda p: _FakeCapture(width=widths[p], height=80))
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    assert pipeline.detect_fixed_input_size(str(tmp_path / "cfg.yaml"), videos) is False


def test_detect_fixed_from_config_unknown_size(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"cropping": False})
    opened = {str(tmp_path / "a.mp4"): True, str(tmp_path / "b.mp4"): False}
    _install_capture(monkeypatch, lambda p: _FakeCapture(opened=opened[p], width=100, height=80))
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    # One video's dimensions cannot be resolved (None), so the run is conservatively reported as not fixed.
    assert pipeline.detect_fixed_input_size(str(tmp_path / "cfg.yaml"), videos) is False


def test_detect_fixed_read_config_raises(monkeypatch):
    def boom(_config):
        message = "nope"
        raise RuntimeError(message)

    monkeypatch.setattr(pipeline, "read_config", boom)
    assert pipeline.detect_fixed_input_size("cfg", ["a"]) is False


# --------------------------------------------------------------------------------------------------------------------
# _resolve_input_size / _probe_frame_size / _probe_frame_count
# --------------------------------------------------------------------------------------------------------------------


def test_resolve_input_size_cropping_with_crop(tmp_path):
    video = tmp_path / "v.mp4"
    cfg = {"cropping": True, "video_sets": {str(video): {"crop": "0,10,0,20"}}}
    assert pipeline._resolve_input_size(project_config=cfg, video=video) == (10, 20)


def test_resolve_input_size_cropping_unresolved(tmp_path):
    cfg = {"cropping": True, "video_sets": {}}
    assert pipeline._resolve_input_size(project_config=cfg, video=tmp_path / "v.mp4") is None


def test_resolve_input_size_no_cropping(monkeypatch, tmp_path):
    _install_capture(monkeypatch, lambda _p: _FakeCapture(width=320, height=240))
    assert pipeline._resolve_input_size(project_config={"cropping": False}, video=tmp_path / "v.mp4") == (320, 240)


def test_probe_frame_size_ok(monkeypatch, tmp_path):
    _install_capture(monkeypatch, lambda _p: _FakeCapture(width=640, height=480))
    assert pipeline._probe_frame_size(tmp_path / "v.mp4") == (640, 480)


def test_probe_frame_size_not_opened(monkeypatch, tmp_path):
    _install_capture(monkeypatch, lambda _p: _FakeCapture(opened=False))
    assert pipeline._probe_frame_size(tmp_path / "v.mp4") is None


def test_probe_frame_size_nonpositive_dimension(monkeypatch, tmp_path):
    _install_capture(monkeypatch, lambda _p: _FakeCapture(width=0, height=480))
    assert pipeline._probe_frame_size(tmp_path / "v.mp4") is None


def test_probe_frame_count_ok(monkeypatch, tmp_path):
    _install_capture(monkeypatch, lambda _p: _FakeCapture(frames=7))
    assert pipeline._probe_frame_count(tmp_path / "v.mp4") == 7


def test_probe_frame_count_not_opened_clamped(monkeypatch, tmp_path):
    _install_capture(monkeypatch, lambda _p: _FakeCapture(opened=False, frames=99))
    # A closed capture reports zero frames, clamped to at least one for the progress bar.
    assert pipeline._probe_frame_count(tmp_path / "v.mp4") == 1


# --------------------------------------------------------------------------------------------------------------------
# _parse_crop / _resolve_video_cropping
# --------------------------------------------------------------------------------------------------------------------


def test_parse_crop_none():
    assert pipeline._parse_crop(None) is None


def test_parse_crop_valid():
    assert pipeline._parse_crop("1, 2, 3, 4") == [1, 2, 3, 4]


def test_parse_crop_wrong_field_count():
    assert pipeline._parse_crop("1,2,3") is None


def test_parse_crop_non_integer():
    assert pipeline._parse_crop("a,b,c,d") is None


def test_resolve_video_cropping_disabled():
    assert pipeline._resolve_video_cropping(project_config={"cropping": False}, video="/x.mp4") is None


def test_resolve_video_cropping_registered_crop(tmp_path):
    video = tmp_path / "v.mp4"
    cfg = {"cropping": True, "video_sets": {str(video): {"crop": "1,2,3,4"}}}
    assert pipeline._resolve_video_cropping(project_config=cfg, video=str(video)) == [1, 2, 3, 4]


def test_resolve_video_cropping_registered_no_crop_falls_back_to_corners(tmp_path):
    video = tmp_path / "v.mp4"
    # The video is registered but carries no parseable crop, so the project-wide rectangle is used.
    cfg = {"cropping": True, "video_sets": {str(video): {"other": 1}}, "x1": 0, "x2": 100, "y1": 5, "y2": 80}
    assert pipeline._resolve_video_cropping(project_config=cfg, video=str(video)) == [0, 100, 5, 80]


def test_resolve_video_cropping_non_dict_metadata_skipped(tmp_path):
    video = tmp_path / "v.mp4"
    # A non-dict video_sets entry is skipped, and the project-wide corners are used instead.
    cfg = {"cropping": True, "video_sets": {str(video): ["not", "a", "dict"]}, "x1": 1, "x2": 2, "y1": 3, "y2": 4}
    assert pipeline._resolve_video_cropping(project_config=cfg, video=str(video)) == [1, 2, 3, 4]


def test_resolve_video_cropping_missing_corner_returns_none():
    cfg = {"cropping": True, "video_sets": {}, "x1": 0, "x2": 100, "y1": 0}  # y2 absent
    assert pipeline._resolve_video_cropping(project_config=cfg, video="/x.mp4") is None


# --------------------------------------------------------------------------------------------------------------------
# _describe_precision
# --------------------------------------------------------------------------------------------------------------------


def test_describe_precision_fp32():
    assert pipeline._describe_precision(make_profile(amp_dtype=None)) == "fp32"


def test_describe_precision_bfloat16():
    assert pipeline._describe_precision(make_profile(amp_dtype=torch.bfloat16)) == "bfloat16"


# --------------------------------------------------------------------------------------------------------------------
# _build_slots / _usable_cpu_cores
# --------------------------------------------------------------------------------------------------------------------


def test_build_slots_cuda_round_robin():
    profile = make_profile(device="cuda", gpus=(0, 1), gpu_processes=2)
    slots = pipeline._build_slots(profile=profile, video_count=10)
    assert [slot.device for slot in slots] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    assert all(slot.cores is None for slot in slots)


def test_build_slots_cuda_truncated_to_video_count():
    profile = make_profile(device="cuda", gpus=(0, 1), gpu_processes=2)
    slots = pipeline._build_slots(profile=profile, video_count=1)
    assert len(slots) == 1
    assert slots[0].device == "cuda:0"


def test_build_slots_cuda_no_gpus_raises():
    profile = make_profile(device="cuda", gpus=(), gpu_processes=1)
    with pytest.raises(ValueError, match="no GPU indices"):
        pipeline._build_slots(profile=profile, video_count=3)


def test_build_slots_mps_single_slot():
    profile = make_profile(device="mps")
    slots = pipeline._build_slots(profile=profile, video_count=5)
    assert len(slots) == 1
    assert slots[0].device == "mps"
    assert slots[0].cores is None


def test_build_slots_cpu(monkeypatch):
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
    monkeypatch.setattr(pipeline.psutil, "cpu_count", lambda **_kwargs: 16)
    # min(16 physical, 2 workers * 4 threads) = 8.
    assert pipeline._usable_cpu_cores(make_profile(cpu_workers=2, cpu_threads_per_worker=4)) == 8


def test_usable_cpu_cores_threads_none_and_zero_workers(monkeypatch):
    monkeypatch.setattr(pipeline.psutil, "cpu_count", lambda **_kwargs: 16)
    # None threads -> 1, zero workers -> max(1, 0) = 1, so the product is 1.
    assert pipeline._usable_cpu_cores(make_profile(cpu_workers=0, cpu_threads_per_worker=None)) == 1


# --------------------------------------------------------------------------------------------------------------------
# _collect_results
# --------------------------------------------------------------------------------------------------------------------


def test_collect_results_success_and_failures():
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
    results_queue = _FakeQueue()
    results_queue.put((0, "/out/a.h5", None))
    video_paths = [Path("a.mp4"), Path("b.mp4")]
    # The second result never arrives, so the drain loop times out and the missing video is reported as a failure.
    outputs, failures = pipeline._collect_results(results_queue=results_queue, video_paths=video_paths)
    assert outputs == {0: Path("/out/a.h5")}
    assert failures == [("b.mp4", "the worker process exited before reporting a result")]


# --------------------------------------------------------------------------------------------------------------------
# _suppress_stdout
# --------------------------------------------------------------------------------------------------------------------


def test_suppress_stdout_inactive_passes_through(capsys):
    with pipeline._suppress_stdout(active=False):
        print("visible")
    assert "visible" in capsys.readouterr().out


def test_suppress_stdout_active_redirects(capsys):
    with pipeline._suppress_stdout(active=True):
        print("hidden")
    assert "hidden" not in capsys.readouterr().out


# --------------------------------------------------------------------------------------------------------------------
# _resolve_output
# --------------------------------------------------------------------------------------------------------------------


def test_resolve_output_exact(tmp_path):
    (tmp_path / "clipScorer.h5").write_bytes(b"")
    out = pipeline._resolve_output(video=str(tmp_path / "clip.mp4"), scorer="Scorer", destination=tmp_path)
    assert out == tmp_path / "clipScorer.h5"


def test_resolve_output_glob_suffixed(tmp_path):
    # No plain per-frame file exists; a tracker-suffixed file is picked as the last lexicographic match.
    (tmp_path / "clipScorer_bx.h5").write_bytes(b"")
    (tmp_path / "clipScorer_el.h5").write_bytes(b"")
    out = pipeline._resolve_output(video=str(tmp_path / "clip.mp4"), scorer="Scorer", destination=tmp_path)
    assert out == tmp_path / "clipScorer_el.h5"


def test_resolve_output_none(tmp_path):
    out = pipeline._resolve_output(video=str(tmp_path / "clip.mp4"), scorer="Scorer", destination=tmp_path)
    assert out is None


# --------------------------------------------------------------------------------------------------------------------
# _run_inference_worker
# --------------------------------------------------------------------------------------------------------------------


def test_run_inference_worker_drains_queue_and_pins_cores(monkeypatch):
    affinity_calls = []

    class _FakeProc:
        def cpu_affinity(self, cores):
            affinity_calls.append(cores)

    monkeypatch.setattr(pipeline.psutil, "Process", _FakeProc)
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

    assert affinity_calls == [[0, 1]]
    assert len(applied) == 1
    assert analyzed == [(0, "/v0.mp4", 10, None, None), (1, "/v1.mp4", 10, None, None)]


def test_run_inference_worker_no_cores_skips_affinity(monkeypatch):
    affinity_calls = []

    class _FakeProc:
        def cpu_affinity(self, cores):
            affinity_calls.append(cores)

    monkeypatch.setattr(pipeline.psutil, "Process", _FakeProc)
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
    class _FakeProc:
        def cpu_affinity(self, _cores):
            message = "affinity unavailable"
            raise OSError(message)

    monkeypatch.setattr(pipeline.psutil, "Process", _FakeProc)
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


# --------------------------------------------------------------------------------------------------------------------
# _analyze_one_video
# --------------------------------------------------------------------------------------------------------------------


def test_analyze_one_video_success_with_progress(monkeypatch, tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    scorer = "DLCscorer"
    video = tmp_path / "clip.mp4"
    (dest / f"clip{scorer}.h5").write_bytes(b"")

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
    item = (3, str(video), 100, [0, 10, 0, 20], str(dest))

    pipeline._analyze_one_video(slot=slot, launch=launch, item=item)

    index, path, error = results_queue._items.popleft()
    assert index == 3
    assert error is None
    assert Path(path) == dest / f"clip{scorer}.h5"
    # The completion marker is published to the progress queue.
    assert progress_queue._items[-1] == ("done", 3)
    # analyze_videos received the resolved crop and the always-overwrite/acceleration-disabled settings.
    assert captured["cropping"] == [0, 10, 0, 20]
    assert captured["overwrite"] is True
    assert captured["destfolder"] == str(dest)
    assert captured["inference_cfg"] == pipeline._STOCK_ACCELERATION_DISABLED
    assert captured["shuffle"] == 1
    assert captured["device"] == "cpu"


def test_analyze_one_video_failure_without_progress(monkeypatch, tmp_path):
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
    dest = tmp_path / "out"
    dest.mkdir()
    monkeypatch.setattr(pipeline.dlc_videos, "analyze_videos", lambda **kwargs: "Scorer")
    monkeypatch.setattr(pipeline.dlc_videos, "tqdm", object(), raising=False)

    results_queue = _FakeQueue()
    launch = _make_launch(display_progress=False, results_queue=results_queue)
    slot = pipeline._Slot(device="cpu", cores=None)
    item = (2, str(tmp_path / "v.mp4"), 10, None, str(dest))

    pipeline._analyze_one_video(slot=slot, launch=launch, item=item)

    index, path, error = results_queue._items.popleft()
    assert index == 2
    assert path is None  # no file was produced, so the output resolves to None
    assert error is None


# --------------------------------------------------------------------------------------------------------------------
# run_inference input validation
# --------------------------------------------------------------------------------------------------------------------


def test_run_inference_empty_videos_raises():
    with pytest.raises(ValueError, match="at least one video"):
        pipeline.run_inference(config="cfg", videos=[], profile=make_profile())


def test_run_inference_crop_override_length_mismatch_raises():
    with pytest.raises(ValueError, match="one crop rectangle per video"):
        pipeline.run_inference(config="cfg", videos=["a", "b"], profile=make_profile(), crop_override=[(0, 10, 0, 20)])


def test_run_inference_destination_override_length_mismatch_raises():
    with pytest.raises(ValueError, match="one output directory per video"):
        pipeline.run_inference(config="cfg", videos=["a", "b"], profile=make_profile(), destination_override=["d1"])


# --------------------------------------------------------------------------------------------------------------------
# run_inference orchestration
# --------------------------------------------------------------------------------------------------------------------


def test_run_inference_full_success_with_overrides(monkeypatch, tmp_path):
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    dest_a = tmp_path / "outa"
    dest_b = tmp_path / "outb"
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"cropping": False})
    _install_capture(monkeypatch, lambda _p: _FakeCapture(frames=30))
    monkeypatch.setattr(
        pipeline,
        "_build_slots",
        lambda **_kwargs: [pipeline._Slot("cuda:0", None), pipeline._Slot("cuda:1", None)],
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
        destination_override=[dest_a, dest_b],
        crop_override=[(0, 10, 0, 20), (0, 10, 0, 20)],
        display_progress=True,
    )

    assert summary.video_count == 2
    assert summary.device == "cuda"
    assert summary.workers == 2
    assert summary.precision == "bfloat16"
    assert summary.destinations == (dest_a, dest_b)
    assert len(summary.outputs) == 2
    assert summary.failures == ()
    assert dest_a.is_dir()
    assert dest_b.is_dir()
    assert bars
    assert bars[0].started
    assert bars[0].stopped
    assert bars[0].joined
    assert manager.shutdown_called is True


def test_run_inference_partial_failure_no_overrides(monkeypatch, tmp_path):
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    project_config = {"cropping": True, "video_sets": {}, "x1": 0, "x2": 100, "y1": 0, "y2": 80}
    monkeypatch.setattr(pipeline, "read_config", lambda _c: project_config)
    _install_capture(monkeypatch, lambda _p: _FakeCapture(frames=10))
    monkeypatch.setattr(
        pipeline,
        "_build_slots",
        lambda **_kwargs: [pipeline._Slot("cpu", (0, 1))],
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
