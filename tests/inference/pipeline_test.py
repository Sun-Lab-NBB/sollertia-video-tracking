"""Contains tests for the multi-device DeepLabCut inference pipeline orchestration.

These tests drive the whole pipeline module without a GPU, network, or real DeepLabCut runtime. Heavy handoffs
(``analyze_videos``, the multiprocessing manager/context, the aggregate progress bar, OpenCV decode) are replaced with
lightweight in-process fakes, and the worker/collector helpers are exercised directly so every branch runs on a
headless CPU box. The three bounded-iterator tests are the exception: they write and read a short Motion JPEG clip
through real OpenCV, and skip when the environment ships no encoder for it.
"""

import sys
import queue
from types import SimpleNamespace
import pickle
import signal
from pathlib import Path
import contextlib
from collections import deque

import cv2
import numpy as np
import torch
import pytest

from sollertia_video_tracking.inference import pipeline
from sollertia_video_tracking.reporting import WorkerExit
from sollertia_video_tracking.inference.optimization import InferenceProfile


def _make_profile(**overrides):
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


# InferenceSummary.successful
def test_inference_summary_reports_partial_success():
    """Verifies that a run with any failed video reports itself as unsuccessful so the CLI can exit non-zero."""
    common = {
        "config": Path("cfg.yaml"),
        "video_count": 2,
        "destinations": None,
        "device": "cuda",
        "workers": 2,
        "precision": "bfloat16",
    }
    complete = pipeline.InferenceSummary(outputs=(Path("a.h5"), Path("b.h5")), failures=(), **common)
    partial = pipeline.InferenceSummary(outputs=(Path("a.h5"),), failures=(("b.mp4", "boom"),), **common)

    assert complete.successful is True
    assert partial.successful is False


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


# discover_directory_videos
def test_discover_directory_videos_sorted_and_filtered(tmp_path):
    """Verifies that discover_directory_videos returns the directory's videos sorted, dropping non-video files."""
    for name in ("b.mp4", "a.avi", "notes.txt", "c.MOV"):
        (tmp_path / name).write_bytes(b"")
    assert pipeline.discover_directory_videos(tmp_path) == [tmp_path / "a.avi", tmp_path / "b.mp4", tmp_path / "c.MOV"]


def test_discover_directory_videos_skips_deeplabcut_companions(tmp_path):
    """Verifies that discover_directory_videos drops the labeled and full companion videos DeepLabCut writes."""
    (tmp_path / "session.mp4").write_bytes(b"")
    (tmp_path / "session_labeled.mp4").write_bytes(b"")
    (tmp_path / "session_full.mp4").write_bytes(b"")
    assert pipeline.discover_directory_videos(tmp_path) == [tmp_path / "session.mp4"]


def test_discover_directory_videos_is_not_recursive(tmp_path):
    """Verifies that discover_directory_videos ignores videos stored in subdirectories."""
    (tmp_path / "top.mp4").write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "deep.mp4").write_bytes(b"")
    assert pipeline.discover_directory_videos(tmp_path) == [tmp_path / "top.mp4"]


def test_discover_directory_videos_empty_directory(tmp_path):
    """Verifies that discover_directory_videos returns an empty list for a directory holding no videos."""
    assert pipeline.discover_directory_videos(tmp_path) == []


# ensure_unique_prediction_targets
def test_ensure_unique_prediction_targets_accepts_distinct_stems(tmp_path):
    """Verifies that ensure_unique_prediction_targets accepts videos whose stems differ."""
    pipeline.ensure_unique_prediction_targets(videos=[tmp_path / "a.mp4", tmp_path / "b.mp4"], destinations=None)


def test_ensure_unique_prediction_targets_rejects_shared_stem_in_one_directory(tmp_path):
    """Verifies that ensure_unique_prediction_targets rejects same-stem videos that write beside each other."""
    with pytest.raises(ValueError, match="share the file-name stem"):
        pipeline.ensure_unique_prediction_targets(
            videos=[tmp_path / "session.mp4", tmp_path / "session.avi"], destinations=None
        )


def test_ensure_unique_prediction_targets_accepts_shared_stem_in_separate_directories(tmp_path):
    """Verifies that ensure_unique_prediction_targets accepts same-stem videos stored in separate directories."""
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    pipeline.ensure_unique_prediction_targets(videos=[first / "session.mp4", second / "session.mp4"], destinations=None)


def test_ensure_unique_prediction_targets_rejects_shared_stem_funneled_into_one_output(tmp_path):
    """Verifies that ensure_unique_prediction_targets rejects same-stem videos sharing one --output directory."""
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    with pytest.raises(ValueError, match="share the file-name stem"):
        pipeline.ensure_unique_prediction_targets(
            videos=[first / "session.mp4", second / "session.mp4"], destinations=[tmp_path, tmp_path]
        )


def test_ensure_unique_prediction_targets_accepts_shared_stem_with_separate_outputs(tmp_path):
    """Verifies that ensure_unique_prediction_targets accepts same-stem videos given separate --output directories."""
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    pipeline.ensure_unique_prediction_targets(
        videos=[tmp_path / "session.mp4", tmp_path / "session.avi"], destinations=[first, second]
    )


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
    assert (
        pipeline.detect_fixed_input_size(config="x", videos=["a", "b"], crop_override=[(0, 10, 0, 20), (0, 30, 0, 20)])
        is False
    )


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
    assert pipeline.detect_fixed_input_size(config=str(tmp_path / "cfg.yaml"), videos=videos) is False


def test_detect_fixed_from_config_unknown_size(monkeypatch, tmp_path):
    """Verifies that detect_fixed_input_size reports not-fixed when a video's dimensions cannot be resolved."""
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"cropping": False})
    opened = {str(tmp_path / "a.mp4"): True, str(tmp_path / "b.mp4"): False}
    _install_capture(monkeypatch, lambda p: _FakeCapture(opened=opened[p], width=100, height=80))
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    # One video's dimensions cannot be resolved (None), so the run is conservatively reported as not fixed.
    assert pipeline.detect_fixed_input_size(config=str(tmp_path / "cfg.yaml"), videos=videos) is False


def test_detect_fixed_read_config_raises(monkeypatch):
    """Verifies that detect_fixed_input_size reports not-fixed when reading the configuration raises."""

    def boom(_config):
        message = "nope"
        raise RuntimeError(message)

    monkeypatch.setattr(pipeline, "read_config", boom)
    assert pipeline.detect_fixed_input_size(config="cfg", videos=["a"]) is False


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


def test_probe_frame_count_not_opened_returns_none(monkeypatch, tmp_path):
    """Verifies that _probe_frame_count reports an unreadable header as None rather than as a small frame count."""
    _install_capture(monkeypatch, lambda _p: _FakeCapture(opened=False, frames=99))
    # None keeps the failure distinguishable from a genuine one-frame video, which a chunked run would otherwise split
    # into a single range whose prediction count matches, hiding the failure behind a one-row prediction file.
    assert pipeline._probe_frame_count(tmp_path / "v.mp4") is None


def test_probe_frame_count_zero_frames_returns_none(monkeypatch, tmp_path):
    """Verifies that an opened capture reporting no frames is also treated as unreadable."""
    _install_capture(monkeypatch, lambda _p: _FakeCapture(frames=0))
    assert pipeline._probe_frame_count(tmp_path / "v.mp4") is None


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
    assert pipeline._describe_precision(_make_profile(amp_dtype=None)) == "fp32"


def test_describe_precision_bfloat16():
    """Verifies that _describe_precision labels a bfloat16 autocast profile as bfloat16."""
    assert pipeline._describe_precision(_make_profile(amp_dtype=torch.bfloat16)) == "bfloat16"


# _build_slots / _usable_cpu_cores
def test_build_slots_cuda_round_robin():
    """Verifies that _build_slots round-robins the CUDA worker slots across the available GPUs."""
    profile = _make_profile(device="cuda", gpus=(0, 1), gpu_processes=2)
    slots = pipeline._build_slots(profile=profile, video_count=10)
    assert [slot.device for slot in slots] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    assert all(slot.cores is None for slot in slots)


def test_build_slots_cuda_truncated_to_video_count():
    """Verifies that _build_slots truncates the CUDA slot list to the video count."""
    profile = _make_profile(device="cuda", gpus=(0, 1), gpu_processes=2)
    slots = pipeline._build_slots(profile=profile, video_count=1)
    assert len(slots) == 1
    assert slots[0].device == "cuda:0"


def test_build_slots_cuda_no_gpus_raises():
    """Verifies that _build_slots raises when CUDA is selected but no GPU indices are resolved."""
    profile = _make_profile(device="cuda", gpus=(), gpu_processes=1)
    with pytest.raises(ValueError, match="no GPU indices"):
        pipeline._build_slots(profile=profile, video_count=3)


def test_build_slots_mps_single_slot():
    """Verifies that _build_slots builds a single unpinned slot for the MPS device."""
    profile = _make_profile(device="mps")
    slots = pipeline._build_slots(profile=profile, video_count=5)
    assert len(slots) == 1
    assert slots[0].device == "mps"
    assert slots[0].cores is None


def test_build_slots_cpu(monkeypatch):
    """Verifies that _build_slots pins the sole CPU worker to its planned two-core block."""
    # Pins the topology so a single worker with a definite two-core block is planned deterministically. With 16 cores
    # (logical and physical), one worker, and two threads/worker, _usable_cpu_cores is 2, so 14 cores are reserved and
    # exactly two usable cores (0 and 1) are pinned to the sole worker.
    monkeypatch.setattr(pipeline.psutil, "cpu_count", lambda **_kwargs: 16)
    profile = _make_profile(device="cpu", cpu_workers=1, cpu_threads_per_worker=2)
    slots = pipeline._build_slots(profile=profile, video_count=1)
    assert len(slots) == 1
    assert slots[0].device == "cpu"
    # The exact pinned block, not merely a non-empty tuple, so a wrong core plan is caught.
    assert slots[0].cores == (0, 1)


def test_usable_cpu_cores_with_threads(monkeypatch):
    """Verifies that _usable_cpu_cores bounds the worker-thread product by the physical core count."""
    monkeypatch.setattr(pipeline.psutil, "cpu_count", lambda **_kwargs: 16)
    # min(16 physical, 2 workers * 4 threads) = 8.
    assert pipeline._usable_cpu_cores(_make_profile(cpu_workers=2, cpu_threads_per_worker=4)) == 8


def test_usable_cpu_cores_threads_none_and_zero_workers(monkeypatch):
    """Verifies that _usable_cpu_cores clamps None threads and zero workers to a product of one."""
    monkeypatch.setattr(pipeline.psutil, "cpu_count", lambda **_kwargs: 16)
    # None threads -> 1, zero workers -> max(1, 0) = 1, so the product is 1.
    assert pipeline._usable_cpu_cores(_make_profile(cpu_workers=0, cpu_threads_per_worker=None)) == 1


# _collect_results
def test_collect_results_success_and_failures():
    """Verifies that _collect_results separates successful outputs from explicit and empty-file failures."""
    results_queue = _FakeQueue()
    results_queue.put(("result", 0, ("/out/a.h5", None)))  # success
    results_queue.put(("result", 1, (None, "RuntimeError: boom")))  # explicit error
    results_queue.put(("result", 2, (None, None)))  # reported but produced no file
    video_paths = [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")]
    outputs, failures = pipeline._collect_results(results_queue=results_queue, video_paths=video_paths)
    assert outputs == {0: Path("/out/a.h5")}
    assert ("b.mp4", "RuntimeError: boom") in failures
    assert ("c.mp4", "no prediction file was produced") in failures
    assert len(failures) == 2


def test_collect_results_unclaimed_video_is_reported_as_never_started():
    """Verifies that a video no worker ever claimed is reported as never analyzed rather than as a crash."""
    results_queue = _FakeQueue()
    results_queue.put(("result", 0, ("/out/a.h5", None)))
    video_paths = [Path("a.mp4"), Path("b.mp4")]
    outputs, failures = pipeline._collect_results(results_queue=results_queue, video_paths=video_paths)
    assert outputs == {0: Path("/out/a.h5")}
    assert failures == [("b.mp4", "no worker started this video, so it was never analyzed.")]


def test_collect_results_classifies_the_death_of_the_worker_that_claimed_the_video():
    """Verifies that a claimed but unreported video is explained by its own worker's exit status."""
    results_queue = _FakeQueue()
    results_queue.put(("claim", 0, 4242))
    video_paths = [Path("a.mp4")]
    exits = (
        WorkerExit(name="cuda:0", pid=4242, exit_code=-signal.SIGKILL, signal_name="SIGKILL"),
        WorkerExit(name="cuda:1", pid=4243, exit_code=0, signal_name=None),
    )
    outputs, failures = pipeline._collect_results(results_queue=results_queue, video_paths=video_paths, exits=exits)
    assert outputs == {}
    assert len(failures) == 1
    video, detail = failures[0]
    assert video == "a.mp4"
    assert "SIGKILL" in detail
    assert "shell status 137" in detail
    assert "out-of-memory killer" in detail


def test_collect_results_interrupted_run_explains_the_gap_without_blaming_a_crash():
    """Verifies that an interrupted run reports its unfinished videos as interrupted rather than as failures."""
    results_queue = _FakeQueue()
    results_queue.put(("claim", 0, 4242))
    outputs, failures = pipeline._collect_results(
        results_queue=results_queue, video_paths=[Path("a.mp4")], interrupted=True
    )
    assert outputs == {}
    assert failures == [
        ("a.mp4", "the run was interrupted while this video was being analyzed, so its predictions were not written.")
    ]


def test_collect_results_reports_an_unstarted_video_after_an_interrupt():
    """Verifies that an interrupted run distinguishes a video nothing started from one it stopped mid-analysis."""
    _outputs, failures = pipeline._collect_results(
        results_queue=_FakeQueue(), video_paths=[Path("a.mp4")], interrupted=True
    )
    assert failures == [("a.mp4", "the run ended before any worker started this video.")]


def test_collect_results_falls_back_when_the_claiming_worker_is_unknown():
    """Verifies that a claim whose worker has no exit record still produces an explanatory failure."""
    results_queue = _FakeQueue()
    results_queue.put(("claim", 0, 9999))
    _outputs, failures = pipeline._collect_results(results_queue=results_queue, video_paths=[Path("a.mp4")])
    assert failures == [("a.mp4", "the worker process ended before reporting a result for this video.")]


# _start_inference_manager
def test_start_inference_manager_reports_a_manager_that_will_not_start(monkeypatch):
    """Verifies that a manager that cannot start raises a named failure rather than a bare EOFError."""

    def dead_manager():
        raise EOFError

    monkeypatch.setattr(pipeline.mp, "Manager", dead_manager)

    with pytest.raises(pipeline.PipelineFailedError, match="inference worker queues"):
        pipeline._start_inference_manager()


# _warn_inference_stalled
def test_warn_inference_stalled_names_the_elapsed_silence(monkeypatch):
    """Verifies that a stalled run is called out with its elapsed silence and stated as not terminated."""
    warnings: list[str] = []
    monkeypatch.setattr(pipeline, "warn", warnings.append)

    pipeline._warn_inference_stalled(1500.0)

    assert len(warnings) == 1
    assert "25:00" in warnings[0]
    assert "Nothing was terminated." in warnings[0]


# _format_inference_interruption
def test_format_inference_interruption_names_the_completed_predictions(tmp_path):
    """Verifies that an interrupted run's report names what was written and how to resume."""
    report = pipeline._format_inference_interruption(
        config=tmp_path / "cfg.yaml",
        shuffle=3,
        video_paths=[Path("a.mp4"), Path("b.mp4")],
        outputs={0: tmp_path / "a.h5"},
    )

    assert "2 submitted, 1 complete" in report
    assert "shuffle: 3" in report
    assert str(tmp_path / "a.h5") in report
    assert "Re-run naming only the videos" in report


def test_format_inference_interruption_states_when_nothing_was_written(tmp_path):
    """Verifies that an interrupted run that wrote nothing says so rather than listing an empty set."""
    report = pipeline._format_inference_interruption(
        config=tmp_path / "cfg.yaml", shuffle=1, video_paths=[Path("a.mp4")], outputs={}
    )

    assert "No prediction file was written" in report


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
    # No plain per-frame file exists, so a tracker-suffixed file is picked as the last lexicographic match.
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
    launch = _make_launch(video_queue=video_queue, profile=_make_profile())
    slot = pipeline._Slot(device="cpu", cores=(0, 1))

    pipeline._run_inference_worker(slot=slot, launch=launch)

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

    pipeline._run_inference_worker(slot=slot, launch=launch)

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

    pipeline._run_inference_worker(slot=slot, launch=launch)

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

    _kind, index, (path, error) = results_queue._items.popleft()
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

    _kind, index, (path, error) = results_queue._items.popleft()
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

    _kind, index, (path, error) = results_queue._items.popleft()
    assert index == 2
    assert path is None  # no file was produced, so the output resolves to None
    assert error is None


# run_inference input validation
def test_run_inference_empty_videos_raises():
    """Verifies that run_inference raises when given an empty video list."""
    with pytest.raises(ValueError, match="at least one video"):
        pipeline.run_inference(config="cfg", videos=[], profile=_make_profile())


def test_run_inference_crop_override_length_mismatch_raises():
    """Verifies that run_inference raises when the crop override length does not match the video count."""
    with pytest.raises(ValueError, match="one crop rectangle per video"):
        pipeline.run_inference(config="cfg", videos=["a", "b"], profile=_make_profile(), crop_override=[(0, 10, 0, 20)])


def test_run_inference_destination_override_length_mismatch_raises():
    """Verifies that run_inference raises when the destination override length does not match the video count."""
    with pytest.raises(ValueError, match="one output directory per video"):
        pipeline.run_inference(config="cfg", videos=["a", "b"], profile=_make_profile(), destination_override=["d1"])


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

    bars = _install_fake_bar(monkeypatch)

    def worker(_slot, launch):
        while True:
            item = launch.video_queue.get()
            if item is None:
                break
            index, _video, _total, crop, destination = item
            assert crop == [0, 10, 0, 20]  # supplied crop override is forwarded verbatim
            launch.results_queue.put(("result", index, (f"{destination}/pred_{index}.h5", None)))

    manager = _install_fake_mp(monkeypatch, worker)

    profile = _make_profile(device="cuda", gpus=(0, 1), gpu_processes=1, amp_dtype=torch.bfloat16)
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
                launch.results_queue.put(("result", index, (f"{video}.h5", None)))
            else:
                launch.results_queue.put(("result", index, (None, "RuntimeError: kaboom")))

    manager = _install_fake_mp(monkeypatch, worker)

    profile = _make_profile(device="cpu", amp_dtype=None)
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


def test_run_inference_interrupt_reports_the_completed_predictions(monkeypatch, tmp_path):
    """Verifies that an interrupted run raises an interruption naming what was written rather than losing the report."""
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for video in videos:
        video.write_bytes(b"")
    monkeypatch.setattr(pipeline, "_probe_frame_count", lambda _video: 10)
    monkeypatch.setattr(pipeline, "read_config", lambda _path: {})
    monkeypatch.setattr(pipeline, "_resolve_video_cropping", lambda **_kwargs: None)

    def worker(_slot, launch):
        # The first video completes, then the operator stops the run.
        launch.results_queue.put(("result", 0, (str(tmp_path / "a.h5"), None)))
        raise KeyboardInterrupt

    _install_fake_mp(monkeypatch, worker)

    with pytest.raises(pipeline.PipelineInterruptedError) as interruption:
        pipeline.run_inference(
            config=tmp_path / "cfg.yaml",
            videos=list(videos),
            profile=_make_profile(device="cpu", cpu_workers=1),
            display_progress=False,
        )

    report = str(interruption.value)
    assert "2 submitted, 1 complete" in report
    assert str(tmp_path / "a.h5") in report


def test_run_inference_emits_the_optimization_report(monkeypatch, tmp_path, capsys):
    """Verifies that the whole-video path writes the resolved-optimization report, so deleting the call fails here."""
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"cropping": False})
    _install_capture(monkeypatch, lambda _p: _FakeCapture(frames=30))
    monkeypatch.setattr(pipeline, "_build_slots", lambda **_kwargs: [pipeline._Slot(device="cpu", cores=(0,))])

    def worker(_slot, launch):
        while True:
            item = launch.video_queue.get()
            if item is None:
                break
            launch.results_queue.put(("result", item[0], (f"{tmp_path}/pred.h5", None)))

    _install_fake_mp(monkeypatch, worker)
    pipeline.run_inference(
        config=str(tmp_path / "cfg.yaml"),
        videos=[tmp_path / "a.mp4"],
        profile=_make_profile(),
        display_progress=False,
    )
    report = capsys.readouterr().err
    assert "-- inference optimizations " in report
    assert "cudnn.benchmark" in report
    assert "workers" in report


# _run_inference_chunked
def test_run_inference_chunked_stitches_each_video_from_its_own_chunks(monkeypatch, tmp_path):
    """Verifies that a chunked run splits every video, forwards the overrides, and writes one file per video."""
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    destination = tmp_path / "out"
    monkeypatch.setattr(pipeline, "_build_analysis_plan", lambda **_kwargs: _make_plan())
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"cropping": False})
    monkeypatch.setattr(pipeline, "_probe_frame_count", lambda _video: 4)
    monkeypatch.setattr(pipeline, "_build_slots", lambda **_kwargs: [pipeline._Slot(device="cuda:0", cores=None)])
    bars = _install_fake_bar(monkeypatch)

    dispatched = []

    def worker(_slot, launch):
        while True:
            item = launch.video_queue.get()
            if item is None:
                break
            dispatched.append(item)
            frames = [f"{Path(item.video).stem}{index}" for index in range(item.frame_start, item.frame_end)]
            launch.results_queue.put((item.task_id, item.video_index, item.chunk_index, frames, None))

    manager = _install_fake_mp(monkeypatch, worker)

    stitched = {}

    def fake_stitch(*, video, predictions, destination, **_kwargs):
        stitched[Path(video).name] = list(predictions)
        return Path(destination) / f"{Path(video).stem}.h5"

    monkeypatch.setattr(pipeline, "_stitch_and_write", fake_stitch)

    summary = pipeline.run_inference(
        config=tmp_path / "cfg.yaml",
        videos=list(videos),
        profile=_make_profile(device="cuda", gpus=(0,), gpu_processes=1, chunks=2, amp_dtype=torch.bfloat16),
        destination_override=[destination, destination],
        crop_override=[(0, 10, 0, 20), (0, 10, 0, 20)],
        display_progress=True,
    )

    # Each video is split into its own contiguous ranges, and every chunk carries the supplied crop and destination.
    assert [(item.video_index, item.chunk_index, item.frame_start, item.frame_end) for item in dispatched] == [
        (0, 0, 0, 2),
        (0, 1, 2, 4),
        (1, 0, 0, 2),
        (1, 1, 2, 4),
    ]
    assert all(item.crop == [0, 10, 0, 20] for item in dispatched)
    assert all(item.destination == str(destination) for item in dispatched)
    assert stitched == {"a.mp4": ["a0", "a1", "a2", "a3"], "b.mp4": ["b0", "b1", "b2", "b3"]}
    assert summary.outputs == (destination / "a.h5", destination / "b.h5")
    assert summary.failures == ()
    assert summary.destinations == (destination,)
    assert summary.precision == "bfloat16"
    assert summary.workers == 1
    # The bar tracks one key per chunk and rolls each key up to the video it belongs to.
    assert bars[0].kwargs["frame_totals"] == {0: 2, 1: 2, 2: 2, 3: 2}
    assert bars[0].kwargs["key_video"] == {0: 0, 1: 0, 2: 1, 3: 1}
    assert bars[0].kwargs["total_video_count"] == 2
    assert bars[0].started
    assert bars[0].stopped
    assert bars[0].joined
    assert manager.shutdown_called is True


def test_run_inference_chunked_rejects_a_multi_animal_project(monkeypatch, tmp_path):
    """Verifies that a chunked run refuses a multi-animal project, whose output the stitch path cannot assemble."""
    monkeypatch.setattr(pipeline, "_build_analysis_plan", lambda **_kwargs: _make_plan(multi_animal=True))

    with pytest.raises(ValueError, match="--chunks 1"):
        pipeline.run_inference(
            config=tmp_path / "cfg.yaml",
            videos=[tmp_path / "a.mp4"],
            profile=_make_profile(chunks=2),
            display_progress=False,
        )


def test_run_inference_chunked_rejects_a_model_that_is_not_bottom_up(monkeypatch, tmp_path):
    """Verifies that a chunked run refuses a top-down model, whose per-frame predictions do not stitch by frame."""
    monkeypatch.setattr(
        pipeline, "_build_analysis_plan", lambda **_kwargs: _make_plan(pose_task=pipeline.Task.TOP_DOWN)
    )

    with pytest.raises(ValueError, match="--chunks 1"):
        pipeline.run_inference(
            config=tmp_path / "cfg.yaml",
            videos=[tmp_path / "a.mp4"],
            profile=_make_profile(chunks=2),
            display_progress=False,
        )


def test_run_inference_chunked_fails_a_video_with_an_unreadable_frame_count(monkeypatch, tmp_path):
    """Verifies that a video whose header yields no frame count is failed rather than split into one short range."""
    videos = [tmp_path / "a.mp4", tmp_path / "bad.mp4"]
    counts = {str(videos[0]): 3, str(videos[1]): None}
    project_config = {"cropping": True, "video_sets": {}, "x1": 0, "x2": 100, "y1": 0, "y2": 80}
    monkeypatch.setattr(pipeline, "_build_analysis_plan", lambda **_kwargs: _make_plan())
    monkeypatch.setattr(pipeline, "read_config", lambda _c: project_config)
    monkeypatch.setattr(pipeline, "_probe_frame_count", lambda video: counts[str(video)])
    monkeypatch.setattr(pipeline, "_build_slots", lambda **_kwargs: [pipeline._Slot(device="cpu", cores=(0,))])

    def worker(_slot, launch):
        while True:
            item = launch.video_queue.get()
            if item is None:
                break
            assert item.crop == [0, 100, 0, 80]  # resolved from the project's project-wide crop rectangle
            assert item.destination is None  # no destination override means write beside the video
            frames = ["p"] * (item.frame_end - item.frame_start)
            launch.results_queue.put((item.task_id, item.video_index, item.chunk_index, frames, None))

    _install_fake_mp(monkeypatch, worker)
    monkeypatch.setattr(pipeline, "_stitch_and_write", lambda **kwargs: Path(kwargs["video"]).with_suffix(".h5"))

    summary = pipeline.run_inference(
        config=tmp_path / "cfg.yaml",
        videos=list(videos),
        profile=_make_profile(chunks=2),
        display_progress=False,
    )

    # The readable video is still analyzed, and the unreadable one is reported once rather than split.
    assert summary.outputs == (tmp_path / "a.h5",)
    assert summary.failures == (("bad.mp4", "the frame count could not be read from the container header"),)


def test_run_inference_chunked_interrupt_states_that_nothing_was_written(monkeypatch, tmp_path):
    """Verifies that an interrupted chunked run reports that no partially analyzed video reached the disk."""
    monkeypatch.setattr(pipeline, "_build_analysis_plan", lambda **_kwargs: _make_plan())
    monkeypatch.setattr(pipeline, "read_config", lambda _c: {"cropping": False})
    monkeypatch.setattr(pipeline, "_probe_frame_count", lambda _video: 8)
    monkeypatch.setattr(pipeline, "_build_slots", lambda **_kwargs: [pipeline._Slot(device="cpu", cores=(0,))])

    def worker(_slot, _launch):
        raise KeyboardInterrupt

    _install_fake_mp(monkeypatch, worker)

    with pytest.raises(pipeline.PipelineInterruptedError) as interruption:
        pipeline.run_inference(
            config=tmp_path / "cfg.yaml",
            videos=[tmp_path / "a.mp4"],
            profile=_make_profile(chunks=2),
            display_progress=False,
        )

    report = str(interruption.value)
    assert "1 submitted, 0 complete" in report
    assert "no partially analyzed video was written" in report


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


# _build_analysis_plan
def test_build_analysis_plan_resolves_the_scorer_and_the_project_batch_size(monkeypatch, tmp_path):
    """Verifies that the plan names its output files from the resolved snapshot and takes the project's batch size."""
    _install_fake_dlc_loader(monkeypatch)
    monkeypatch.setattr(pipeline, "read_plainconfig", lambda path: {"read_from": str(path)})

    plan = pipeline._build_analysis_plan(
        config=tmp_path / "cfg.yaml", shuffle=2, snapshot_index=None, detector_snapshot_index=None, batch_size=None
    )

    assert plan.scorer == "DLC_snapshot-050"
    assert plan.train_fraction == 0.95
    assert plan.batch_size == 8  # resolved from the project configuration, which no explicit batch size overrode
    assert plan.multi_animal is False
    assert plan.pose_task is pipeline.Task.BOTTOM_UP
    # The pose configuration is read from the shuffle's test directory, beside the model folder.
    assert plan.pose_config == {"read_from": str(_FAKE_MODEL_FOLDER.parent / "test" / "pose_cfg.yaml")}


def test_build_analysis_plan_prefers_an_explicit_batch_size(monkeypatch, tmp_path):
    """Verifies that an explicitly requested batch size is recorded in the metadata instead of the project's own."""
    _install_fake_dlc_loader(monkeypatch)
    monkeypatch.setattr(pipeline, "read_plainconfig", lambda _path: {})

    plan = pipeline._build_analysis_plan(
        config=tmp_path / "cfg.yaml", shuffle=1, snapshot_index=-1, detector_snapshot_index=None, batch_size=16
    )

    assert plan.batch_size == 16


# _run_chunk_worker
def test_run_chunk_worker_builds_one_runner_and_drains_every_chunk(monkeypatch):
    """Verifies that a chunk worker pins its cores, builds its runner once, and analyzes every chunk it pulls.

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
    built = []

    def fake_build_runner(**_kwargs):
        built.append("runner")
        return f"runner {len(built)}"

    monkeypatch.setattr(pipeline, "_build_pose_runner", fake_build_runner)
    analyzed = []
    monkeypatch.setattr(
        pipeline, "_analyze_one_chunk", lambda runner, item, **_kwargs: analyzed.append((runner, item.task_id))
    )

    video_queue = _FakeQueue()
    video_queue.put(_chunk_item(task_id=0, video_index=0, chunk_index=0, video="v.mp4", frame_start=0, frame_end=5))
    video_queue.put(_chunk_item(task_id=1, video_index=0, chunk_index=1, video="v.mp4", frame_start=5, frame_end=10))
    video_queue.put(None)
    launch = _make_launch(video_queue=video_queue)

    pipeline._run_chunk_worker(slot=pipeline._Slot(device="cpu", cores=(2, 3)), launch=launch)

    assert affinity_calls == ([] if sys.platform == "darwin" else [[2, 3]])
    assert len(applied) == 1
    # The runner is built once and reused, since rebuilding it per chunk would pay the model load on every range.
    assert analyzed == [("runner 1", 0), ("runner 1", 1)]


# _build_pose_runner
def test_build_pose_runner_builds_on_its_own_slot_device(monkeypatch):
    """Verifies that the runner is built on the worker's device through the patched apis-utils builder."""
    _install_fake_dlc_loader(monkeypatch)
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return "the runner"

    monkeypatch.setattr(pipeline.dlc_apis_utils, "get_pose_inference_runner", fake_builder)

    runner = pipeline._build_pose_runner(
        slot=pipeline._Slot(device="cuda:1", cores=None), launch=_make_launch(shuffle=2, batch_size=16)
    )

    assert runner == "the runner"
    assert captured["device"] == "cuda:1"
    assert captured["batch_size"] == 16
    assert captured["max_individuals"] == 1
    assert captured["snapshot_path"] == _FAKE_SNAPSHOT_PATH
    assert captured["inference_cfg"] == pipeline._STOCK_ACCELERATION_DISABLED


def test_build_pose_runner_falls_back_to_the_project_batch_size(monkeypatch):
    """Verifies that a worker given no explicit batch size builds its runner with the project's configured one."""
    _install_fake_dlc_loader(monkeypatch)
    captured = {}
    monkeypatch.setattr(pipeline.dlc_apis_utils, "get_pose_inference_runner", lambda **kwargs: captured.update(kwargs))

    pipeline._build_pose_runner(slot=pipeline._Slot(device="cpu", cores=None), launch=_make_launch(batch_size=None))

    assert captured["batch_size"] == 8


# _BoundedVideoIterator
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


def test_bounded_iterator_stops_when_the_video_ends_before_its_range(tmp_path):
    """Verifies that a range reaching past the end of the video stops at the last frame the decoder returns."""
    clip = tmp_path / "clip.avi"
    _write_identifiable_clip(clip, frames=10)
    reference = _sequential_frame_levels(clip)
    iterator = pipeline._BoundedVideoIterator(str(clip), frame_start=5, frame_end=30)

    levels = [int(frame[0, 0, 0]) for frame in iterator]

    assert levels == reference[5:]
    # The declared count stays the range length, so the shortfall surfaces as the frame-count mismatch a chunk reports.
    assert iterator.get_n_frames() == 25


# _analyze_one_chunk
def test_analyze_one_chunk_reports_its_predictions_and_marks_the_chunk_done(monkeypatch):
    """Verifies that a completed chunk reports its predictions under its own task id and publishes a done marker."""
    built = {}

    def fake_iterator(video, *, frame_start, frame_end, cropping):
        built.update(video=video, frame_start=frame_start, frame_end=frame_end, cropping=cropping)
        return "iterator"

    def fake_inference(*, video, pose_runner):
        built.update(analyzed=video, runner=pose_runner)
        return ["p0", "p1", "p2"]

    monkeypatch.setattr(pipeline, "_BoundedVideoIterator", fake_iterator)
    monkeypatch.setattr(pipeline, "video_inference", fake_inference)
    monkeypatch.setattr(pipeline.dlc_videos, "tqdm", object(), raising=False)

    launch = _make_launch(display_progress=True)
    item = pipeline._ChunkItem(
        task_id=7,
        video_index=1,
        chunk_index=2,
        video="v.mp4",
        frame_start=4,
        frame_end=7,
        crop=[0, 10, 0, 20],
        destination=None,
    )

    pipeline._analyze_one_chunk(runner="runner", launch=launch, item=item)

    # The bounded iterator receives the chunk's own range and crop, and the worker's runner analyzes it.
    assert built == {
        "video": "v.mp4",
        "frame_start": 4,
        "frame_end": 7,
        "cropping": [0, 10, 0, 20],
        "analyzed": "iterator",
        "runner": "runner",
    }
    assert launch.results_queue.get() == (7, 1, 2, ["p0", "p1", "p2"], None)
    assert launch.progress_queue._items[-1] == ("done", 7)


def test_analyze_one_chunk_fails_a_short_read_rather_than_misaligning_the_stitch(monkeypatch):
    """Verifies that a chunk decoding fewer frames than its range is failed instead of shifting every later frame."""
    monkeypatch.setattr(pipeline, "_BoundedVideoIterator", lambda *_args, **_kwargs: "iterator")
    monkeypatch.setattr(pipeline, "video_inference", lambda **_kwargs: ["p0", "p1"])

    launch = _make_launch()
    item = _chunk_item(task_id=4, video_index=0, chunk_index=1, video="v.mp4", frame_start=0, frame_end=5)

    pipeline._analyze_one_chunk(runner="runner", launch=launch, item=item)

    task_id, video_index, chunk_index, predictions, error = launch.results_queue.get()
    assert (task_id, video_index, chunk_index, predictions) == (4, 0, 1, None)
    assert "covers 5 frames but decoded 2" in error


def test_analyze_one_chunk_resets_runner_queue_on_failure(monkeypatch):
    """Verifies that a chunk failing inside inference discards the runner's already-preprocessed batches.

    A worker reuses one runner across chunks, and DeepLabCut's asynchronous inference loop never drains the queue those
    batches sit in, so leaving them would let the next chunk consume them as its own and report a matching frame count.
    """
    runner = SimpleNamespace(
        inference_cfg=SimpleNamespace(multithreading=SimpleNamespace(enabled=True, queue_length=4))
    )
    runner._input_queue = queue.Queue(maxsize=4)
    runner._input_queue.put("a batch preprocessed for this chunk")
    poisoned_queue = runner._input_queue

    monkeypatch.setattr(pipeline, "_BoundedVideoIterator", lambda *_args, **_kwargs: object())

    def _fail(**_kwargs):
        message = "CUDA out of memory"
        raise RuntimeError(message)

    monkeypatch.setattr(pipeline, "video_inference", _fail)
    launch = _make_launch()
    item = pipeline._ChunkItem(
        task_id=3,
        video_index=0,
        chunk_index=1,
        video="v.mp4",
        frame_start=0,
        frame_end=8,
        crop=None,
        destination=None,
    )

    pipeline._analyze_one_chunk(runner=runner, launch=launch, item=item)

    assert runner._input_queue is not poisoned_queue
    assert runner._input_queue.empty()
    task_id, video_index, chunk_index, predictions, error = launch.results_queue.get()
    assert (task_id, video_index, chunk_index, predictions) == (3, 0, 1, None)
    assert "CUDA out of memory" in error


# _collect_chunk_results
def test_collect_chunk_results_stitches_chunks_in_frame_order(monkeypatch):
    """Verifies that per-chunk predictions are concatenated in ascending chunk order and written once per video."""
    work_items = [
        _chunk_item(task_id=0, video_index=0, chunk_index=0, video="a.mp4", frame_start=0, frame_end=2),
        _chunk_item(task_id=1, video_index=0, chunk_index=1, video="a.mp4", frame_start=2, frame_end=4),
        _chunk_item(task_id=2, video_index=1, chunk_index=0, video="b.mp4", frame_start=0, frame_end=3),
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
        _chunk_item(task_id=0, video_index=0, chunk_index=0, video="a.mp4", frame_start=0, frame_end=2),
        _chunk_item(task_id=1, video_index=0, chunk_index=1, video="a.mp4", frame_start=2, frame_end=4),
        _chunk_item(task_id=2, video_index=1, chunk_index=0, video="b.mp4", frame_start=0, frame_end=2),
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


def test_collect_chunk_results_fails_a_video_whose_chunk_never_reported(monkeypatch):
    """Verifies that a video is failed, and never stitched, when a worker died before reporting one of its chunks."""
    work_items = [
        _chunk_item(task_id=0, video_index=0, chunk_index=0, video="a.mp4", frame_start=0, frame_end=2),
        _chunk_item(task_id=1, video_index=0, chunk_index=1, video="a.mp4", frame_start=2, frame_end=4),
    ]
    results_queue = _FakeQueue()
    results_queue.put((0, 0, 0, ["a0", "a1"], None))

    def fail_if_called(**_kwargs):
        message = "a video missing a chunk must never be stitched"
        raise AssertionError(message)

    monkeypatch.setattr(pipeline, "_stitch_and_write", fail_if_called)
    outputs, failures = pipeline._collect_chunk_results(
        results_queue=results_queue, video_paths=[Path("a.mp4")], work_items=work_items, plan=None
    )

    # Draining stops once the queue runs dry, so the second chunk is missing rather than reported with an error.
    assert outputs == {}
    assert failures == [("a.mp4", "a chunk worker exited before reporting a result")]


def test_collect_chunk_results_reports_a_failed_stitch(monkeypatch):
    """Verifies that a video whose stitched write raises is reported as failed rather than ending the collection."""
    work_items = [
        _chunk_item(task_id=0, video_index=0, chunk_index=0, video="a.mp4", frame_start=0, frame_end=2),
        _chunk_item(task_id=1, video_index=1, chunk_index=0, video="b.mp4", frame_start=0, frame_end=2),
    ]
    results_queue = _FakeQueue()
    results_queue.put((0, 0, 0, ["a0", "a1"], None))
    results_queue.put((1, 1, 0, ["b0", "b1"], None))

    def stitch(*, video, **_kwargs):
        if Path(video).name == "a.mp4":
            message = "no space left on device"
            raise OSError(message)
        return Path(video).with_suffix(".h5")

    monkeypatch.setattr(pipeline, "_stitch_and_write", stitch)
    outputs, failures = pipeline._collect_chunk_results(
        results_queue=results_queue, video_paths=[Path("a.mp4"), Path("b.mp4")], work_items=work_items, plan=None
    )

    # The failed write is attributed to its own video, and the remaining video is still written.
    assert outputs == {1: Path("b.h5")}
    assert failures == [("a.mp4", "OSError: no space left on device")]


# _stitch_and_write
def test_stitch_and_write_writes_the_prediction_files_beside_the_video(monkeypatch, tmp_path):
    """Verifies that a stitched video writes its metadata, full pickle, and prediction table beside the video."""
    _install_fake_output_writers(monkeypatch)
    written = {}
    monkeypatch.setattr(pipeline, "create_df_from_prediction", lambda **kwargs: written.update(kwargs))

    output = pipeline._stitch_and_write(
        plan=_make_plan(),
        video=str(tmp_path / "clip.mp4"),
        destination=None,
        crop=[0, 10, 0, 20],
        predictions=["p0", "p1"],
    )

    assert output == tmp_path / "clipDLCscorer.h5"
    with (tmp_path / "clipDLCscorer_meta.pickle").open("rb") as handle:
        # The metadata records the crop the video was analyzed at, which the whole-video path also stores.
        assert pickle.load(handle) == {"cropping": [0, 10, 0, 20]}  # noqa: S301 - reads back the file just written.
    with (tmp_path / "clipDLCscorer_full.pickle").open("rb") as handle:
        assert pickle.load(handle) == {"frames": 2}  # noqa: S301 - reads back the file just written.
    assert written["dlc_scorer"] == "DLCscorer"
    assert written["output_path"] == tmp_path
    assert written["output_prefix"] == "clipDLCscorer"
    assert written["multi_animal"] is False
    assert written["save_as_csv"] is False


def test_stitch_and_write_creates_the_requested_output_directory(monkeypatch, tmp_path):
    """Verifies that a stitched video writes into the requested output directory, creating it when it does not exist."""
    _install_fake_output_writers(monkeypatch)
    monkeypatch.setattr(pipeline, "create_df_from_prediction", lambda **_kwargs: None)
    destination = tmp_path / "predictions"

    output = pipeline._stitch_and_write(
        plan=_make_plan(),
        video=str(tmp_path / "clip.mp4"),
        destination=str(destination),
        crop=None,
        predictions=["p0"],
    )

    assert output == destination / "clipDLCscorer.h5"
    assert (destination / "clipDLCscorer_meta.pickle").is_file()
    assert (destination / "clipDLCscorer_full.pickle").is_file()


# _report_inference_optimizations
def test_optimization_report_states_the_spawned_worker_count_and_the_ceiling_it_missed(capsys):
    """Verifies that a run with less work than its worker ceiling reports the spawned count and the configured one."""
    profile = _make_profile(device="cuda", gpus=(0, 1), gpu_processes=2, chunks=4, cpu_workers=0)
    pipeline._report_inference_optimizations(profile=profile, worker_count=4)
    assert "4 of 16 configured" in capsys.readouterr().err


def test_optimization_report_states_a_bare_count_when_every_worker_spawns(capsys):
    """Verifies that a run reaching its ceiling reports the count alone, with no redundant configured figure."""
    profile = _make_profile(cpu_workers=2, chunks=1)
    pipeline._report_inference_optimizations(profile=profile, worker_count=2)
    report = capsys.readouterr().err
    assert "workers" in report
    assert "configured" not in report


def test_optimization_report_appends_the_worker_row_after_the_resolved_state(capsys):
    """Verifies that the run-scoped worker count is reported after the optimizations the profile resolved."""
    profile = _make_profile(cpu_workers=1, chunks=1)
    pipeline._report_inference_optimizations(profile=profile, worker_count=1)
    body = [line for line in capsys.readouterr().err.split("\n") if line and not line.startswith("-")]
    assert body[0].split()[0] == "device"
    assert body[-1].split()[0] == "workers"


# Shared fakes and helpers
_FAKE_MODEL_FOLDER = Path("/proj/dlc-models-pytorch/iteration-0/task/train")
"""The model folder the fake DeepLabCut loader reports, whose sibling test directory holds the pose configuration."""

_FAKE_SNAPSHOT_PATH = Path("/proj/dlc-models-pytorch/iteration-0/task/train/snapshot-050.pt")
"""The snapshot path the fake snapshot resolution returns, which names the scorer and loads the model weights."""


class _FakeCapture:
    """Stands in for cv2.VideoCapture, reporting fixed header values without opening a file."""

    def __init__(self, *, opened=True, width=100, height=80, frames=50):
        self._opened = opened
        self._width = width
        self._height = height
        self._frames = frames
        self.released = False
        # cv2.VideoCapture exposes a camelCase ``isOpened``. Binds it as an attribute so the fake matches the real
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
    """Provides a single-process queue backed by a deque. ``get`` raises queue.Empty when the deque is drained."""

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
        self.pid = None
        self.exitcode = None
        self.terminated = False

    def start(self):
        self.pid = 4242
        self._worker_fn(*self.args)
        self.exitcode = 0

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return self.pid is not None and self.exitcode is None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


class _FakeContext:
    """Provides a spawn-context stand-in whose Process drives the supplied simulated worker."""

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


class _FakeBar:
    """Stands in for the aggregate progress bar, recording its construction arguments and its lifecycle calls."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = self.stopped = self.joined = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self, **_kwargs):
        self.joined = True

    def seconds_since_progress(self):
        return 0.0


def _install_fake_bar(monkeypatch):
    """Replaces the aggregate progress bar with an in-process fake and returns the list every built bar lands in."""
    bars = []

    def build(**kwargs):
        bar = _FakeBar(**kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(pipeline, "AggregateBar", build)
    return bars


class _FakeLoader:
    """Stands in for the DeepLabCut project loader, exposing the configuration the chunked path reads from it."""

    def __init__(self, config, shuffle):
        self.config = config
        self.shuffle = shuffle
        self.project_cfg = {"TrainingFraction": [0.95], "batch_size": 8, "multianimalproject": False}
        self.model_cfg = {"metadata": {"individuals": ["single"]}}
        self.model_folder = _FAKE_MODEL_FOLDER
        self.pose_task = pipeline.Task.BOTTOM_UP

    def scorer(self, snapshot, **_kwargs):
        return f"DLC_{Path(snapshot.path).stem}"


def _install_fake_dlc_loader(monkeypatch):
    """Replaces the DeepLabCut loader and the snapshot resolution the chunked path drives with in-process fakes."""
    monkeypatch.setattr(pipeline, "DLCLoader", _FakeLoader)
    monkeypatch.setattr(pipeline.dlc_apis_utils, "parse_snapshot_index_for_analysis", lambda **_kwargs: (-1, None))
    monkeypatch.setattr(
        pipeline.dlc_apis_utils,
        "get_model_snapshots",
        lambda **_kwargs: [SimpleNamespace(path=_FAKE_SNAPSHOT_PATH)],
    )


def _install_fake_output_writers(monkeypatch):
    """Replaces the DeepLabCut metadata and full-pickle builders with fakes that record what the stitch passed them."""
    monkeypatch.setattr(pipeline, "VideoIterator", lambda path, cropping: SimpleNamespace(path=path, crop=cropping))
    monkeypatch.setattr(pipeline, "_generate_metadata", lambda **kwargs: {"cropping": kwargs["cropping"]})
    monkeypatch.setattr(pipeline, "_generate_output_data", lambda **kwargs: {"frames": len(kwargs["predictions"])})


def _make_plan(**overrides):
    """Builds an _AnalysisPlan for the chunked-path tests, overriding only the fields a test cares about."""
    defaults = {
        "scorer": "DLCscorer",
        "project_config": {"multianimalproject": False},
        "model_config": {"metadata": {"individuals": ["single"]}},
        "pose_config": {"all_joints_names": ["snout"]},
        "train_fraction": 0.95,
        "batch_size": 1,
        "multi_animal": False,
        "pose_task": pipeline.Task.BOTTOM_UP,
    }
    defaults.update(overrides)
    return pipeline._AnalysisPlan(**defaults)


def _make_launch(**overrides):
    """Builds an _InferenceLaunch with fake queues, overriding only the fields a test needs."""
    defaults = {
        "config": Path("/proj/config.yaml"),
        "shuffle": 1,
        "snapshot_index": None,
        "detector_snapshot_index": None,
        "profile": _make_profile(),
        "batch_size": None,
        "detector_batch_size": None,
        "display_progress": False,
        "video_queue": _FakeQueue(),
        "progress_queue": _FakeQueue(),
        "results_queue": _FakeQueue(),
    }
    defaults.update(overrides)
    return pipeline._InferenceLaunch(**defaults)


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


def test_collect_results_reports_a_stopped_worker_as_an_interruption():
    """Verifies that a worker a termination signal ended is reported as stopped rather than as a crash."""
    results_queue = _FakeQueue()
    results_queue.put(("claim", 0, 4242))
    exits = (WorkerExit(name="cuda:0", pid=4242, exit_code=-signal.SIGTERM, signal_name="SIGTERM"),)

    _outputs, failures = pipeline._collect_results(
        results_queue=results_queue, video_paths=[Path("a.mp4")], exits=exits
    )

    assert failures == [
        ("a.mp4", "the worker was stopped while analyzing this video, so its predictions were not written.")
    ]


def test_collect_results_reports_a_clean_exit_that_never_reported():
    """Verifies that a worker that exited cleanly without a result is not blamed on a crash."""
    results_queue = _FakeQueue()
    results_queue.put(("claim", 0, 4242))
    exits = (WorkerExit(name="cuda:0", pid=4242, exit_code=0, signal_name=None),)

    _outputs, failures = pipeline._collect_results(
        results_queue=results_queue, video_paths=[Path("a.mp4")], exits=exits
    )

    assert failures == [("a.mp4", "the worker exited without reporting a result for this video.")]
