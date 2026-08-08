"""Contains tests for the parallel outlier-frame extraction pipeline that refines a DeepLabCut model on
likely-wrong frames.

The DeepLabCut runtime boundary (``read_config``, ``load_analyzed_data``, and the frame writer) is stubbed so the
whole pipeline runs headless, deterministically, and without a GPU, network, or real decode. Worker and orchestration
bodies are driven in-process, with ``run_supervised_tasks`` and ``iter_pinned_extraction`` replaced by stand-ins that
apply their worker synchronously.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sollertia_video_tracking.frame_extraction import outlier_pipeline
from sollertia_video_tracking.frame_extraction.outlier_pipeline import (
    TrackingMethod,
    ExtractionAlgorithm,
    OutlierExtractionSummary,
)
from sollertia_video_tracking.frame_extraction.outlier_detection import OutlierAlgorithm


# Enums and the summary dataclass.
def test_extraction_and_tracking_enums_expose_expected_values() -> None:
    """Verifies that the string enums carry the wire values the CLI and DeepLabCut exchange."""
    assert ExtractionAlgorithm.KMEANS == "kmeans"
    assert ExtractionAlgorithm.UNIFORM == "uniform"
    assert {member.value for member in TrackingMethod} == {"box", "skeleton", "ellipse"}


def test_summary_properties_and_describe_clean_run() -> None:
    """Verifies that a clean run reports success, zero failures, and a describe line without a failure tail."""
    summary = OutlierExtractionSummary(
        config_path=Path("/p/config.yaml"),
        outlier_algorithm=OutlierAlgorithm.JUMP,
        extraction_algorithm=ExtractionAlgorithm.KMEANS,
        total_video_count=2,
        extracted_video_count=2,
        worker_count=3,
        used_core_count=6,
        total_core_count=8,
        candidate_frame_count=40,
        extracted_frame_count=10,
    )
    assert summary.failed_video_count == 0
    assert summary.successful is True
    description = summary.describe()
    assert "extracted 10 frames from 2/2 videos" in description
    assert "not analyzed" not in description
    assert "failed" not in description


def test_summary_describe_includes_unanalyzed_and_failed_tail() -> None:
    """Verifies that unanalyzed and failed videos surface in the describe tail and flip ``successful`` off."""
    summary = OutlierExtractionSummary(
        config_path=Path("/p/config.yaml"),
        outlier_algorithm=OutlierAlgorithm.UNCERTAIN,
        extraction_algorithm=ExtractionAlgorithm.UNIFORM,
        total_video_count=3,
        extracted_video_count=1,
        worker_count=2,
        used_core_count=4,
        total_core_count=8,
        candidate_frame_count=5,
        extracted_frame_count=2,
        unanalyzed_videos=("/v/una.mp4",),
        errors=(("/v/bad.mp4", "detection error"),),
    )
    assert summary.failed_video_count == 1
    assert summary.successful is False
    description = summary.describe()
    assert "1 not analyzed" in description
    assert "1 failed" in description


# _video_cropping_offset
def test_video_cropping_offset_uncropped_returns_origin() -> None:
    """Verifies that a video with no crop specification yields the zero origin."""
    assert outlier_pipeline._video_cropping_offset(configuration={}, video="/v/v1.mp4") == (0, 0)


def test_video_cropping_offset_reads_top_left_origin() -> None:
    """Verifies that a valid ``x1,x2,y1,y2`` crop yields the ``(x1, y1)`` top-left origin."""
    configuration = {"video_sets": {"/v/v1.mp4": {"crop": "1,2,3,4"}}}
    assert outlier_pipeline._video_cropping_offset(configuration=configuration, video="/v/v1.mp4") == (1, 3)


def test_video_cropping_offset_rejects_malformed_crop() -> None:
    """Verifies that a crop that is not four comma-separated integers raises a descriptive ValueError."""
    configuration = {"video_sets": {"/v/v1.mp4": {"crop": "1,2,3"}}}
    with pytest.raises(ValueError, match="four comma-separated integers"):
        outlier_pipeline._video_cropping_offset(configuration=configuration, video="/v/v1.mp4")


# _load_sliced_predictions
def test_load_sliced_predictions_applies_crop_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a cropped video's predictions are shifted by the metadata origin minus the output-crop origin."""
    video = "/v/v1.mp4"
    predictions = _predictions(frame_count=4)
    metadata = {"data": {"cropping": True, "cropping_parameters": (10, 200, 20, 200)}}
    monkeypatch.setattr(
        outlier_pipeline.auxiliaryfunctions, "load_analyzed_data", lambda **_k: (predictions, None, None, None)
    )
    monkeypatch.setattr(outlier_pipeline.auxiliaryfunctions, "load_video_metadata", lambda **_k: metadata)
    configuration = {"start": 0.0, "stop": 1.0, "video_sets": {video: {"crop": "5,100,5,100"}}}

    result = outlier_pipeline._load_sliced_predictions(
        video=video,
        video_predictions_directory=Path("/v"),
        scorer="S",
        configuration=configuration,
        tracking_method="",
    )

    # x offset = x1(10) - output_crop_x(5) = 5. y offset = y1(20) - output_crop_y(5) = 15. Likelihood untouched.
    assert (result.xs("x", level="coords", axis=1).to_numpy() == 5).all()
    assert (result.xs("y", level="coords", axis=1).to_numpy() == 15).all()
    assert (result.xs("likelihood", level="coords", axis=1).to_numpy() == 0).all()
    assert len(result) == 4


def test_load_sliced_predictions_slices_window_without_cropping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an uncropped video is only sliced to the configured start/stop window and its positions are
    unchanged.
    """
    video = "/v/v1.mp4"
    predictions = _predictions(frame_count=4)
    metadata = {"data": {"cropping": False}}
    monkeypatch.setattr(
        outlier_pipeline.auxiliaryfunctions, "load_analyzed_data", lambda **_k: (predictions, None, None, None)
    )
    monkeypatch.setattr(outlier_pipeline.auxiliaryfunctions, "load_video_metadata", lambda **_k: metadata)
    configuration = {"start": 0.25, "stop": 0.75, "video_sets": {video: {}}}

    result = outlier_pipeline._load_sliced_predictions(
        video=video,
        video_predictions_directory=Path("/v"),
        scorer="S",
        configuration=configuration,
        tracking_method="",
    )

    # floor(4 * 0.25) = 1 through ceil(4 * 0.75) = 3 -> rows [1, 2].
    assert list(result.index) == [1, 2]
    assert (result.to_numpy() == 0).all()


# _discover_analyzed_videos
def test_discover_analyzed_videos_keeps_only_videos_with_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that videos whose prediction probe raises FileNotFoundError are dropped, the rest kept in
    configuration order.
    """

    def fake_find(*, videoname, **_kwargs):
        if videoname == "v2":
            raise FileNotFoundError

    monkeypatch.setattr(outlier_pipeline.auxiliaryfunctions, "find_analyzed_data", fake_find)
    registered = ["/a/v1.mp4", "/b/v2.mp4", "/c/v3.mp4"]

    result = outlier_pipeline._discover_analyzed_videos(registered_videos=registered, scorer="S", tracking_method="")

    assert result == ["/a/v1.mp4", "/c/v3.mp4"]


# _detect_all_videos
def test_detect_all_videos_list_uses_explicit_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the ``list`` algorithm returns the sorted, de-duplicated explicit frame indices."""
    candidates, unanalyzed, errors = _detect(
        monkeypatch=monkeypatch,
        load_stub=lambda **_k: _predictions(),
        algorithm=OutlierAlgorithm.LIST,
        explicit_frame_indices=(3, 1, 2, 1),
    )
    assert candidates == {"v1": [1, 2, 3]}
    assert unanalyzed == []
    assert errors == []


def test_detect_all_videos_uncertain_flags_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the ``uncertain`` algorithm flags a frame whose likelihood falls below the confidence bound."""
    predictions = _predictions(frame_count=3)
    predictions.iloc[:, predictions.columns.get_level_values("coords") == "likelihood"] = 0.9
    # Drop frame 1 below the 0.6 confidence bound.
    predictions.loc[1, ("S", "nose", "likelihood")] = 0.1

    candidates, unanalyzed, errors = _detect(
        monkeypatch=monkeypatch, load_stub=lambda **_k: predictions, algorithm=OutlierAlgorithm.UNCERTAIN
    )
    assert candidates == {"v1": [1]}
    assert unanalyzed == []
    assert errors == []


def test_detect_all_videos_jump_flags_large_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the ``jump`` algorithm flags the frames straddling an over-threshold position jump."""
    predictions = _predictions(frame_count=4)
    predictions.loc[2, ("S", "nose", "x")] = 100.0  # a jump into and back out of frame 2.

    candidates, _unanalyzed, _errors = _detect(
        monkeypatch=monkeypatch,
        load_stub=lambda **_k: predictions,
        algorithm=OutlierAlgorithm.JUMP,
        pixel_distance_threshold=5.0,
    )
    assert candidates == {"v1": [2, 3]}


def test_detect_all_videos_fitting_reduces_via_the_supervised_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the ``fitting`` algorithm defers to the supervised trajectory-fit reduction after validating
    numframes2pick.
    """
    monkeypatch.setattr(outlier_pipeline, "_detect_fitting_outliers", lambda **_k: {"v1": [2]})

    candidates, unanalyzed, errors = _detect(
        monkeypatch=monkeypatch,
        load_stub=lambda **_k: _predictions(frame_count=5),
        algorithm=OutlierAlgorithm.FITTING,
        configuration={"numframes2pick": 5},
    )
    assert candidates == {"v1": [2]}
    assert unanalyzed == []
    assert errors == []


def test_detect_all_videos_fitting_rejects_bad_numframes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a missing or non-positive ``numframes2pick`` fails the fitting path with a clean ValueError."""
    with pytest.raises(ValueError, match="numframes2pick must be a positive integer"):
        _detect(
            monkeypatch=monkeypatch,
            load_stub=lambda **_k: _predictions(frame_count=5),
            algorithm=OutlierAlgorithm.FITTING,
            configuration={"numframes2pick": None},
        )


def test_detect_all_videos_records_missing_and_failed_per_video(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a missing prediction file is recorded as unanalyzed and a malformed table as a per-video error,
    not raised.
    """
    good = _predictions(frame_count=3)
    good.iloc[:, good.columns.get_level_values("coords") == "likelihood"] = 0.9

    def load_stub(*, video, **_k):
        if video == "missing":
            raise FileNotFoundError
        if video == "bad":
            message = "boom"
            raise ValueError(message)
        return good

    candidates, unanalyzed, errors = _detect(
        monkeypatch=monkeypatch,
        load_stub=load_stub,
        algorithm=OutlierAlgorithm.UNCERTAIN,
        video_paths=["ok", "missing", "bad"],
    )
    assert candidates == {"ok": []}
    assert unanalyzed == ["missing"]
    assert len(errors) == 1
    assert errors[0][0] == "bad"
    assert "detection error" in errors[0][1]


# _detect_fitting_outliers
def test_detect_fitting_outliers_auto_workers_reports_progress(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that automatic worker sizing fills the usable cores (clamped to the task count) and reduces to outlier
    frames.
    """
    recorded_worker_counts: list[int] = []
    # Pin the visible core count so the auto worker-sizing arithmetic is deterministic and can be asserted exactly.
    monkeypatch.setattr(outlier_pipeline.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        outlier_pipeline,
        "run_supervised_tasks",
        _recording_supervised_runner(recorded_worker_counts),
    )

    result = outlier_pipeline._detect_fitting_outliers(
        fitting_keypoint_counts=_fit_keypoint_counts(),
        scorer="DLC",
        configuration={},
        tracking_method="",
        resolved_comparison_bodyparts=["bp0"],
        frames_per_video_count=0,
        pixel_distance_threshold=20.0,
        minimum_confidence=0.5,
        autoregressive_degree=3,
        moving_average_degree=1,
        fitting_worker_count=-1,
        reserved_core_count=2,
        display_progress=True,
    )

    assert result == {"vA": [2], "vB": [2]}
    # fitting_worker_count=-1 -> auto: usable = 8 cores - 2 reserved = 6, clamped down to the 3 keypoint tasks.
    assert recorded_worker_counts == [3]
    assert "fitting 3 keypoint trajectories across 2 video(s) on 3 processes" in capsys.readouterr().err


def test_detect_fitting_outliers_explicit_workers_no_progress(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that an explicit fitting worker count is honored verbatim and the progress line is suppressed when
    disabled.
    """
    recorded_worker_counts: list[int] = []
    # With 8 - 2 = 6 usable cores the auto path would size the fan-out at 3 (the task count). An explicit 2 must win.
    monkeypatch.setattr(outlier_pipeline.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        outlier_pipeline,
        "run_supervised_tasks",
        _recording_supervised_runner(recorded_worker_counts),
    )

    result = outlier_pipeline._detect_fitting_outliers(
        fitting_keypoint_counts=_fit_keypoint_counts(),
        scorer="DLC",
        configuration={},
        tracking_method="",
        resolved_comparison_bodyparts=["bp0"],
        frames_per_video_count=0,
        pixel_distance_threshold=20.0,
        minimum_confidence=0.5,
        autoregressive_degree=3,
        moving_average_degree=1,
        fitting_worker_count=2,
        reserved_core_count=2,
        display_progress=False,
    )

    assert result == {"vA": [2], "vB": [2]}
    # The explicit count (2) is used as-is rather than filling the usable cores.
    assert recorded_worker_counts == [2]
    # display_progress=False suppresses the fitting progress line entirely.
    assert "fitting" not in capsys.readouterr().err


# _fit_one_keypoint_task
def test_fit_one_keypoint_task_unpacks_its_work_item(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the packed fitting work item is applied to the fit function as its arguments."""
    received: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        outlier_pipeline,
        "_fit_video_keypoint",
        lambda *arguments: received.append(arguments) or np.array([1.0]),
    )

    result = outlier_pipeline._fit_one_keypoint_task(("vA", 0, "DLC"))

    assert received == [("vA", 0, "DLC")]
    assert result.tolist() == [1.0]


# _fit_video_keypoint
def test_fit_video_keypoint_reloads_its_own_slice_and_forwards_the_fit_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that a fit worker reloads the video, restricts it to the comparison bodyparts, and fits one keypoint."""
    columns = pd.MultiIndex.from_product(
        [["DLC"], ["bp0", "bp1", "dropped"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    data = np.zeros((4, 9), dtype=np.float64)
    data[:, 3] = [10.0, 11.0, 12.0, 13.0]  # bp1 x
    data[:, 4] = [20.0, 21.0, 22.0, 23.0]  # bp1 y
    data[:, 5] = [0.1, 0.2, 0.3, 0.4]  # bp1 likelihood
    load_calls: list[dict[str, object]] = []
    fit_calls: list[dict[str, object]] = []

    def _fake_load(**kwargs):
        load_calls.append(kwargs)
        return pd.DataFrame(data=data, columns=columns)

    def _fake_fit(**kwargs):
        fit_calls.append(kwargs)
        return np.array([0.0, 1.0, 2.0, 3.0])

    monkeypatch.setattr(outlier_pipeline, "_load_sliced_predictions", _fake_load)
    monkeypatch.setattr(outlier_pipeline, "fit_keypoint_distance", _fake_fit)

    deviation = outlier_pipeline._fit_video_keypoint(
        "/videos/v1.mp4",
        1,
        "DLC",
        {"start": 0, "stop": 1},
        "",
        ["bp0", "bp1"],
        0.5,
        3,
        1,
    )

    np.testing.assert_array_equal(deviation, [0.0, 1.0, 2.0, 3.0])
    # The worker loads the predictions itself rather than receiving trajectories from the parent.
    assert load_calls[0]["video"] == "/videos/v1.mp4"
    assert load_calls[0]["video_predictions_directory"] == Path("/videos")
    # The "dropped" bodypart is filtered out, so keypoint index 1 selects bp1 rather than the third column triple.
    np.testing.assert_array_equal(fit_calls[0]["horizontal_positions"], [10.0, 11.0, 12.0, 13.0])
    np.testing.assert_array_equal(fit_calls[0]["vertical_positions"], [20.0, 21.0, 22.0, 23.0])
    np.testing.assert_array_equal(fit_calls[0]["confidences"], [0.1, 0.2, 0.3, 0.4])
    assert fit_calls[0]["minimum_confidence"] == 0.5
    assert fit_calls[0]["autoregressive_degree"] == 3
    assert fit_calls[0]["moving_average_degree"] == 1


# _extract_all_videos (and _report_plan through it)
def test_extract_all_videos_aggregates_results_and_reports_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that the three worker outcomes are aggregated into the summary and the plan header is written when
    displaying.
    """
    results = [("v1", 7, "ok"), ("v2", 0, "not_analyzed"), ("v3", 0, "error:\ntb")]
    monkeypatch.setattr(outlier_pipeline, "iter_pinned_extraction", _fake_iter_factory(results))

    summary = outlier_pipeline._extract_all_videos(
        config_path=tmp_path / "config.yaml",
        videos=["v1", "v2", "v3"],
        candidates={"v1": [0, 1, 2], "v2": [3], "v3": [4, 5]},
        scorer="S",
        tracking_method="",
        outlier_algorithm=OutlierAlgorithm.JUMP,
        extraction_algorithm=ExtractionAlgorithm.KMEANS,
        clustering_resize_width=30,
        cluster_in_color=False,
        save_labeled_frames=False,
        worker_count=-1,
        cores_per_worker=-1,
        reserved_core_count=2,
        display_progress=True,
        total_video_count=4,
        unanalyzed_videos=("pre_un",),
        detection_errors=[("pre_err", "boom")],
    )

    assert summary.extracted_video_count == 1
    assert summary.extracted_frame_count == 7
    assert summary.candidate_frame_count == 6
    assert summary.total_video_count == 4
    assert summary.unanalyzed_videos == ("pre_un", "v2")
    assert [video for video, _detail in summary.errors] == ["pre_err", "v3"]
    stderr = capsys.readouterr().err
    # The plan header renders the resolved algorithms and the real candidate total, not just the video count.
    assert "outlier extraction | 3 videos | detect=jump | select=kmeans | 6 candidate frames" in stderr
    assert "workers=" in stderr


def test_extract_all_videos_suppresses_plan_when_progress_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies that with progress disabled the plan header is skipped while the results still aggregate."""
    monkeypatch.setattr(outlier_pipeline, "iter_pinned_extraction", _fake_iter_factory([("v1", 3, "ok")]))

    summary = outlier_pipeline._extract_all_videos(
        config_path=tmp_path / "config.yaml",
        videos=["v1"],
        candidates={"v1": [0, 1]},
        scorer="S",
        tracking_method="",
        outlier_algorithm=OutlierAlgorithm.JUMP,
        extraction_algorithm=ExtractionAlgorithm.KMEANS,
        clustering_resize_width=30,
        cluster_in_color=False,
        save_labeled_frames=False,
        worker_count=-1,
        cores_per_worker=-1,
        reserved_core_count=2,
        display_progress=False,
        total_video_count=1,
        unanalyzed_videos=(),
        detection_errors=[],
    )

    assert summary.extracted_video_count == 1
    assert summary.extracted_frame_count == 3
    assert summary.successful is True


# _count_directory_frames and _skip_video_registration
def test_count_directory_frames_ignores_overlays(tmp_path: Path) -> None:
    """Verifies that only ``img*.png`` frames are counted. ``*labeled.png`` overlays and missing directories are
    ignored.
    """
    assert outlier_pipeline._count_directory_frames(tmp_path / "missing") == 0
    (tmp_path / "img0001.png").write_bytes(b"")
    (tmp_path / "img0002.png").write_bytes(b"")
    (tmp_path / "img0003labeled.png").write_bytes(b"")
    assert outlier_pipeline._count_directory_frames(tmp_path) == 2


def test_skip_video_registration_reports_success() -> None:
    """Verifies that the registration neutralizer always reports a successful add so the frame writer proceeds."""
    assert outlier_pipeline._skip_video_registration(video="anything", cfg={}) is True


# _extract_one_video
def test_extract_one_video_writes_frames_and_reports_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies that a successful extraction writes frames into the labeled-data directory and reports the freshly
    written count.
    """
    _install_extract_stubs(monkeypatch)
    configuration = {"project_path": str(tmp_path)}
    monkeypatch.setattr(outlier_pipeline.auxiliaryfunctions, "read_config", lambda _p: configuration)
    monkeypatch.setattr(outlier_pipeline, "_load_sliced_predictions", lambda **_k: _predictions())

    def fake_extract(**kwargs):
        output = Path(kwargs["cfg"]["project_path"]) / "labeled-data" / Path(kwargs["video"]).stem
        output.mkdir(parents=True, exist_ok=True)
        for index in kwargs["Index"]:
            (output / f"img{index:04d}.png").write_bytes(b"")

    monkeypatch.setattr(outlier_pipeline.dlc_outlier_frames, "ExtractFramesbasedonPreselection", fake_extract)

    task = (
        "/v/clip.mp4",
        0,
        [1, 2, 3],
        tmp_path / "config.yaml",
        "S",
        "",
        ExtractionAlgorithm.KMEANS,
        30,
        False,
        False,
        object(),  # a non-None progress queue drives the reporter branch.
    )
    video, written, status = outlier_pipeline._extract_one_video(task)

    assert (video, written, status) == ("/v/clip.mp4", 3, "ok")
    assert outlier_pipeline._count_directory_frames(tmp_path / "labeled-data" / "clip") == 3


def test_extract_one_video_without_progress_queue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies that a None progress queue leaves the reporter unset while the extraction still succeeds."""
    _install_extract_stubs(monkeypatch)
    configuration = {"project_path": str(tmp_path)}
    monkeypatch.setattr(outlier_pipeline.auxiliaryfunctions, "read_config", lambda _p: configuration)
    monkeypatch.setattr(outlier_pipeline, "_load_sliced_predictions", lambda **_k: _predictions())
    monkeypatch.setattr(outlier_pipeline.dlc_outlier_frames, "ExtractFramesbasedonPreselection", lambda **_k: None)

    task = (
        "/v/clip.mp4",
        0,
        [1],
        tmp_path / "config.yaml",
        "S",
        "",
        ExtractionAlgorithm.KMEANS,
        30,
        False,
        False,
        None,
    )
    video, written, status = outlier_pipeline._extract_one_video(task)

    # No frames were written by the no-op stub, so the count delta clamps to zero.
    assert (video, written, status) == ("/v/clip.mp4", 0, "ok")


def test_extract_one_video_missing_predictions_is_not_analyzed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies that a missing prediction file surfaces as the ``not_analyzed`` status rather than raising."""
    _install_extract_stubs(monkeypatch)
    monkeypatch.setattr(outlier_pipeline.auxiliaryfunctions, "read_config", lambda _p: {"project_path": str(tmp_path)})

    def raise_missing(**_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(outlier_pipeline, "_load_sliced_predictions", raise_missing)

    task = (
        "/v/clip.mp4",
        0,
        [1],
        tmp_path / "config.yaml",
        "S",
        "",
        ExtractionAlgorithm.KMEANS,
        30,
        False,
        False,
        None,
    )
    assert outlier_pipeline._extract_one_video(task) == ("/v/clip.mp4", 0, "not_analyzed")


def test_extract_one_video_captures_extraction_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies that an error raised by the frame writer is captured as an ``error:`` traceback status, not
    propagated.
    """
    _install_extract_stubs(monkeypatch)
    monkeypatch.setattr(outlier_pipeline.auxiliaryfunctions, "read_config", lambda _p: {"project_path": str(tmp_path)})
    monkeypatch.setattr(outlier_pipeline, "_load_sliced_predictions", lambda **_k: _predictions())

    def raise_boom(**_kwargs):
        message = "decode blew up"
        raise RuntimeError(message)

    monkeypatch.setattr(outlier_pipeline.dlc_outlier_frames, "ExtractFramesbasedonPreselection", raise_boom)

    task = (
        "/v/clip.mp4",
        0,
        [1],
        tmp_path / "config.yaml",
        "S",
        "",
        ExtractionAlgorithm.KMEANS,
        30,
        False,
        False,
        None,
    )
    video, written, status = outlier_pipeline._extract_one_video(task)

    assert video == "/v/clip.mp4"
    assert written == 0
    assert status.startswith("error:\n")
    assert "decode blew up" in status


# _clear_video_iteration_outliers
def test_clear_video_iteration_outliers_no_machine_labels_is_noop(tmp_path: Path) -> None:
    """Verifies that a directory without a machine-label table for the iteration removes nothing."""
    assert outlier_pipeline._clear_video_iteration_outliers(directory=tmp_path, iteration=0, scorer="human") == (
        0,
        False,
    )


def test_clear_video_iteration_outliers_keeps_labeled_and_drops_refinement(tmp_path: Path) -> None:
    """Verifies that only unlabeled outlier frames are removed. Overlays, bookkeeping, and refinements go too."""
    directory = tmp_path / "labeled-data" / "video"
    directory.mkdir(parents=True)
    _write_labels(path=directory / "machinelabels-iter0.h5", names=["img0001.png", "img0002.png"])
    _write_labels(
        path=directory / "CollectedData_human.h5", names=["img0002.png"]
    )  # img0002 is a real human label -> kept.
    (directory / "img0001.png").write_bytes(b"")
    (directory / "img0002.png").write_bytes(b"")
    (directory / "img0001labeled.png").write_bytes(b"")
    (directory / "machinelabels.csv").write_text("stale")
    (directory / "MachineLabelsRefine.h5").write_bytes(b"stale")

    removed, had_refined = outlier_pipeline._clear_video_iteration_outliers(
        directory=directory, iteration=0, scorer="human"
    )

    assert (removed, had_refined) == (1, True)
    assert not (directory / "img0001.png").exists()
    assert not (directory / "img0001labeled.png").exists()
    assert (directory / "img0002.png").exists()  # preserved human label.
    assert not (directory / "machinelabels-iter0.h5").exists()
    assert not (directory / "machinelabels.csv").exists()
    assert not (directory / "MachineLabelsRefine.h5").exists()


def test_clear_video_iteration_outliers_clears_placeholder_rows_and_keeps_annotated(tmp_path: Path) -> None:
    """Verifies that an opened-but-unannotated outlier frame is cleared with its placeholder row, while an annotated
    frame survives."""
    directory = tmp_path / "labeled-data" / "video"
    directory.mkdir(parents=True)
    _write_labels(path=directory / "machinelabels-iter0.h5", names=["img0001.png", "img0002.png"])
    # The labeling GUI reindexes CollectedData over every image in the directory on each save, so both outlier frames
    # gained a row even though only img0002 was actually annotated.
    _write_labels(
        path=directory / "CollectedData_human.h5",
        names=["img0001.png", "img0002.png"],
        finite_names={"img0002.png"},
    )
    (directory / "img0001.png").write_bytes(b"")
    (directory / "img0002.png").write_bytes(b"")

    removed, had_refined = outlier_pipeline._clear_video_iteration_outliers(
        directory=directory, iteration=0, scorer="human"
    )

    assert (removed, had_refined) == (1, False)
    assert not (directory / "img0001.png").exists()  # A placeholder row is not an annotation.
    assert (directory / "img0002.png").exists()
    # The placeholder row for the cleared frame is dropped, so no label references a deleted image.
    remaining = pd.read_hdf(path_or_buf=directory / "CollectedData_human.h5", key="df_with_missing")
    assert [entry[-1] for entry in remaining.index] == ["img0002.png"]


def test_clear_video_iteration_outliers_without_collected_data_removes_all(tmp_path: Path) -> None:
    """Verifies that with no human CollectedData table, every outlier frame is removed and no refinement is reported."""
    directory = tmp_path / "labeled-data" / "video"
    directory.mkdir(parents=True)
    _write_labels(path=directory / "machinelabels-iter0.h5", names=["img0001.png", "img0002.png"])
    (directory / "img0001.png").write_bytes(b"")
    (directory / "img0002.png").write_bytes(b"")

    removed, had_refined = outlier_pipeline._clear_video_iteration_outliers(
        directory=directory, iteration=0, scorer="human"
    )

    assert (removed, had_refined) == (2, False)
    assert not (directory / "img0001.png").exists()
    assert not (directory / "img0002.png").exists()


# _clear_iteration_outliers
def test_clear_iteration_outliers_reset_scans_all_and_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies that reset clears every iteration directory on disk, warns on unreadable ones and on discarded
    refinements.
    """
    config_path = tmp_path / "config.yaml"
    _make_iteration_dir(
        root=tmp_path, name="v1", frame="img0001.png", refinement=True
    )  # a discarded refinement -> warning.
    _make_iteration_dir(root=tmp_path, name="v2", frame="img0002.png")
    # A corrupt machine-label table matches the reset glob and raises when read -> the per-directory warning branch.
    bad_directory = tmp_path / "labeled-data" / "vbad"
    bad_directory.mkdir(parents=True)
    (bad_directory / "machinelabels-iter0.h5").write_bytes(b"not a real hdf file")

    outlier_pipeline._clear_iteration_outliers(
        config_path=config_path,
        configuration={"iteration": 0, "scorer": "human"},
        selected_videos=[],
        reset=True,
    )

    stderr = capsys.readouterr().err
    assert "--reset removed 2 outlier frame(s) from 2 video directory(ies)" in stderr
    assert "could not clear the outlier frames" in stderr
    assert "discarded manual outlier refinements in 1 directory(ies)" in stderr


def test_clear_iteration_outliers_reset_without_labeled_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that reset on a project with no labeled-data tree clears nothing and reports a zero count."""
    outlier_pipeline._clear_iteration_outliers(
        config_path=tmp_path / "config.yaml",
        configuration={"iteration": 0, "scorer": "human"},
        selected_videos=[],
        reset=True,
    )
    assert "--reset removed 0 outlier frame(s) from 0 video directory(ies)" in capsys.readouterr().err


def test_clear_iteration_outliers_overwrite_targets_selected_videos(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that overwrite clears only the selected videos' directories, skipping any that hold no iteration
    outliers.
    """
    config_path = tmp_path / "config.yaml"
    _make_iteration_dir(root=tmp_path, name="v1", frame="img0001.png")

    outlier_pipeline._clear_iteration_outliers(
        config_path=config_path,
        configuration={"iteration": 0, "scorer": "human"},
        # "v1" has an iteration table. "vempty" has no directory at all and is silently skipped.
        selected_videos=["/data/v1.mp4", "/data/vempty.mp4"],
        reset=False,
    )

    stderr = capsys.readouterr().err
    assert "--overwrite removed 1 outlier frame(s) from 1 video directory(ies)" in stderr
    assert "discarded manual outlier refinements" not in stderr


# Orchestrator: the extract_outlier_frames_parallel entry point.
def test_orchestrator_rejects_missing_config(tmp_path: Path) -> None:
    """Verifies that a config path that is not a file raises FileNotFoundError before any work begins."""
    with pytest.raises(FileNotFoundError, match="does not point to a file"):
        outlier_pipeline.extract_outlier_frames_parallel(config_path=tmp_path / "nope.yaml", videos=[])


def test_orchestrator_rejects_overwrite_and_reset(tmp_path: Path) -> None:
    """Verifies that the overwrite and reset options are mutually exclusive."""
    config_path = _write_config(tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        outlier_pipeline.extract_outlier_frames_parallel(config_path=config_path, videos=[], overwrite=True, reset=True)


def test_orchestrator_rejects_unknown_outlier_algorithm(tmp_path: Path) -> None:
    """Verifies that an unknown outlier algorithm raises listing the valid choices."""
    config_path = _write_config(tmp_path)
    with pytest.raises(ValueError, match="outlier algorithm must be one of"):
        outlier_pipeline.extract_outlier_frames_parallel(
            config_path=config_path,
            videos=[],
            outlier_algorithm="bogus",  # type: ignore[arg-type]
        )


def test_orchestrator_rejects_unknown_extraction_algorithm(tmp_path: Path) -> None:
    """Verifies that an unknown extraction algorithm raises listing the valid choices."""
    config_path = _write_config(tmp_path)
    with pytest.raises(ValueError, match="extraction algorithm must be one of"):
        outlier_pipeline.extract_outlier_frames_parallel(
            config_path=config_path,
            videos=[],
            extraction_algorithm="bogus",  # type: ignore[arg-type]
        )


def test_orchestrator_list_requires_explicit_indices(tmp_path: Path) -> None:
    """Verifies that the list algorithm requires an explicit list of frame indices."""
    config_path = _write_config(tmp_path)
    with pytest.raises(ValueError, match="requires an explicit list of frames"):
        outlier_pipeline.extract_outlier_frames_parallel(
            config_path=config_path, videos=[], outlier_algorithm=OutlierAlgorithm.LIST
        )


def test_orchestrator_rejects_non_positive_candidate_step(tmp_path: Path) -> None:
    """Verifies that a candidate step below one is rejected."""
    config_path = _write_config(tmp_path)
    with pytest.raises(ValueError, match="candidate step must be at least one"):
        outlier_pipeline.extract_outlier_frames_parallel(config_path=config_path, videos=[], candidate_step=0)


def test_orchestrator_rejects_empty_comparison_bodyparts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that comparison bodyparts that resolve to none are rejected."""
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(
        outlier_pipeline.auxiliaryfunctions, "read_config", lambda _p: {"video_sets": {"/v/v1.mp4": {}}}
    )
    monkeypatch.setattr(
        outlier_pipeline.auxiliaryfunctions, "intersection_of_body_parts_and_ones_given_by_user", lambda **_k: []
    )
    with pytest.raises(ValueError, match="comparison bodyparts matched none"):
        outlier_pipeline.extract_outlier_frames_parallel(config_path=config_path, videos=[])


def test_orchestrator_rejects_project_without_registered_videos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a project whose config lists no videos in video_sets is rejected."""
    config_path = _write_config(tmp_path)
    _patch_dlc_boundary(monkeypatch=monkeypatch, configuration={"video_sets": {}, "TrainingFraction": [0.95]})
    with pytest.raises(ValueError, match="does not list any videos in video_sets"):
        outlier_pipeline.extract_outlier_frames_parallel(config_path=config_path, videos=[])


def test_orchestrator_warns_and_raises_when_no_requested_video_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that requested videos that match no registered video are warned about, then the run raises."""
    config_path = _write_config(tmp_path)
    _patch_dlc_boundary(monkeypatch=monkeypatch, configuration=_base_config("/registered/v1.mp4"))
    with pytest.raises(ValueError, match="None of the requested videos matched"):
        outlier_pipeline.extract_outlier_frames_parallel(config_path=config_path, videos=["/other/nope.mp4"])
    assert "is not registered in the project's config.yaml and was skipped" in capsys.readouterr().err


def test_orchestrator_no_videos_and_none_analyzed_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that with no named videos and no analyzed videos, the run raises pointing at the analyze step."""
    config_path = _write_config(tmp_path)
    _patch_dlc_boundary(monkeypatch=monkeypatch, configuration=_base_config("/registered/v1.mp4"))
    monkeypatch.setattr(outlier_pipeline, "_discover_analyzed_videos", lambda **_k: [])
    with pytest.raises(ValueError, match="No registered project video has predictions"):
        outlier_pipeline.extract_outlier_frames_parallel(config_path=config_path, videos=[])


def test_orchestrator_matched_videos_run_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a requested, registered video flows through detection into the extraction phase."""
    video_path = tmp_path / "videos" / "v1.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"")
    config_path = _write_config(tmp_path)
    _patch_dlc_boundary(monkeypatch=monkeypatch, configuration=_base_config(str(video_path)))
    monkeypatch.setattr(outlier_pipeline, "_detect_all_videos", lambda **_k: ({str(video_path): [1, 2, 3]}, [], []))
    sentinel = object()
    captured: dict = {}

    def fake_extract(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(outlier_pipeline, "_extract_all_videos", fake_extract)

    result = outlier_pipeline.extract_outlier_frames_parallel(
        config_path=config_path, videos=[str(video_path)], display_progress=False
    )

    assert result is sentinel
    assert captured["videos"] == [str(video_path)]
    assert captured["candidates"] == {str(video_path): [1, 2, 3]}


def test_orchestrator_discovers_videos_when_none_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that with no named videos the discovered analyzed set drives the default refinement pass."""
    video_path = "/registered/v1.mp4"
    config_path = _write_config(tmp_path)
    _patch_dlc_boundary(monkeypatch=monkeypatch, configuration=_base_config(video_path))
    monkeypatch.setattr(outlier_pipeline, "_discover_analyzed_videos", lambda **_k: [video_path])
    monkeypatch.setattr(outlier_pipeline, "_detect_all_videos", lambda **_k: ({video_path: [4, 5]}, [], []))
    sentinel = object()
    monkeypatch.setattr(outlier_pipeline, "_extract_all_videos", lambda **_k: sentinel)

    assert (
        outlier_pipeline.extract_outlier_frames_parallel(config_path=config_path, videos=[], display_progress=False)
        is sentinel
    )


def test_orchestrator_empty_candidates_returns_early_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that when detection flags no candidates, an early zero-worker summary carries the unanalyzed and
    failed videos.
    """
    video_path = "/registered/v1.mp4"
    config_path = _write_config(tmp_path)
    _patch_dlc_boundary(monkeypatch=monkeypatch, configuration=_base_config(video_path))
    monkeypatch.setattr(outlier_pipeline, "_discover_analyzed_videos", lambda **_k: [video_path])
    monkeypatch.setattr(
        outlier_pipeline, "_detect_all_videos", lambda **_k: ({video_path: []}, ["/una.mp4"], [("/bad.mp4", "detail")])
    )
    extract_called = False

    def fail_if_called(**_kwargs):
        nonlocal extract_called
        extract_called = True

    monkeypatch.setattr(outlier_pipeline, "_extract_all_videos", fail_if_called)

    summary = outlier_pipeline.extract_outlier_frames_parallel(
        config_path=config_path, videos=[], display_progress=False
    )

    assert extract_called is False
    assert summary.worker_count == 0
    assert summary.extracted_video_count == 0
    assert summary.total_video_count == 1
    assert summary.unanalyzed_videos == ("/una.mp4",)
    assert summary.errors == (("/bad.mp4", "detail"),)
    assert summary.total_core_count == (os.cpu_count() or 1)


def test_orchestrator_candidate_step_subsamples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a candidate step above one thins the flagged candidates before extraction."""
    video_path = "/registered/v1.mp4"
    config_path = _write_config(tmp_path)
    _patch_dlc_boundary(monkeypatch=monkeypatch, configuration=_base_config(video_path))
    monkeypatch.setattr(outlier_pipeline, "_discover_analyzed_videos", lambda **_k: [video_path])
    monkeypatch.setattr(outlier_pipeline, "_detect_all_videos", lambda **_k: ({video_path: [0, 1, 2, 3, 4, 5]}, [], []))
    captured: dict = {}

    def fake_extract(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(outlier_pipeline, "_extract_all_videos", fake_extract)

    outlier_pipeline.extract_outlier_frames_parallel(
        config_path=config_path, videos=[], candidate_step=2, display_progress=False
    )

    assert captured["candidates"] == {video_path: [0, 2, 4]}


def test_orchestrator_overwrite_clears_iteration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the overwrite option clears this iteration's outliers for the selected videos before detection."""
    video_path = tmp_path / "videos" / "v1.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"")
    config_path = _write_config(tmp_path)
    _patch_dlc_boundary(monkeypatch=monkeypatch, configuration=_base_config(str(video_path)))
    clear_calls: list[dict] = []
    monkeypatch.setattr(outlier_pipeline, "_clear_iteration_outliers", lambda **kwargs: clear_calls.append(kwargs))
    # No candidates keeps the run in the early-summary path so no extraction pool is needed.
    monkeypatch.setattr(outlier_pipeline, "_detect_all_videos", lambda **_k: ({str(video_path): []}, [], []))

    outlier_pipeline.extract_outlier_frames_parallel(
        config_path=config_path, videos=[str(video_path)], overwrite=True, display_progress=False
    )

    assert len(clear_calls) == 1
    assert clear_calls[0]["reset"] is False
    assert clear_calls[0]["selected_videos"] == [str(video_path)]


def test_orchestrator_reset_clears_iteration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the reset option clears the iteration across every project video before detection."""
    video_path = "/registered/v1.mp4"
    config_path = _write_config(tmp_path)
    _patch_dlc_boundary(monkeypatch=monkeypatch, configuration=_base_config(video_path))
    monkeypatch.setattr(outlier_pipeline, "_discover_analyzed_videos", lambda **_k: [video_path])
    clear_calls: list[dict] = []
    monkeypatch.setattr(outlier_pipeline, "_clear_iteration_outliers", lambda **kwargs: clear_calls.append(kwargs))
    monkeypatch.setattr(outlier_pipeline, "_detect_all_videos", lambda **_k: ({video_path: []}, [], []))

    outlier_pipeline.extract_outlier_frames_parallel(
        config_path=config_path, videos=[], reset=True, display_progress=False
    )

    assert len(clear_calls) == 1
    assert clear_calls[0]["reset"] is True
    # Reset still passes the discovered video set through. The reset flag, not the list, widens the scope in the helper.
    assert clear_calls[0]["selected_videos"] == [video_path]


# Builders shared across tests.
def _predictions(frame_count: int = 4, bodyparts: tuple[str, ...] = ("nose",), scorer: str = "S") -> pd.DataFrame:
    """Builds a prediction table with a ``scorer / bodyparts / coords`` column MultiIndex filled with zeros."""
    columns = pd.MultiIndex.from_tuples(
        [(scorer, bodypart, coord) for bodypart in bodyparts for coord in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )
    return pd.DataFrame(np.zeros((frame_count, len(columns)), dtype=np.float64), columns=columns)


def _write_config(tmp_path: Path) -> Path:
    """Writes a minimal valid config.yaml so ``normalize_project_config`` (real) can read and normalize it."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_path: /old/project/path\n")
    return config_path


def _write_labels(path: Path, names: list[str], *, finite_names: set[str] | None = None) -> None:
    """Writes a DeepLabCut-style label HDF table indexed by ``(labeled-data, video, image)`` rows.

    Frames listed in ``finite_names`` receive a finite coordinate, and every other frame receives an all-NaN
    placeholder row, mirroring what the labeling GUI writes for opened-but-untouched frames.
    """
    if finite_names is None:
        finite_names = set(names)
    index = pd.MultiIndex.from_tuples([("labeled-data", "video", name) for name in names])
    coordinates = [0.0 if name in finite_names else np.nan for name in names]
    pd.DataFrame({"x": coordinates}, index=index).to_hdf(path, key="df_with_missing")


def _detect(monkeypatch: pytest.MonkeyPatch, load_stub, algorithm, **overrides):
    """Drives ``_detect_all_videos`` with a stubbed slice loader and sensible defaults."""
    monkeypatch.setattr(outlier_pipeline, "_load_sliced_predictions", load_stub)
    kwargs = {
        "video_paths": overrides.pop("video_paths", ["v1"]),
        "scorer": "S",
        "configuration": overrides.pop("configuration", {"numframes2pick": 5}),
        "tracking_method": "",
        "resolved_comparison_bodyparts": ["nose"],
        "outlier_algorithm": algorithm,
        "explicit_frame_indices": overrides.pop("explicit_frame_indices", ()),
        "pixel_distance_threshold": overrides.pop("pixel_distance_threshold", 20.0),
        "minimum_confidence": overrides.pop("minimum_confidence", 0.6),
        "autoregressive_degree": 3,
        "moving_average_degree": 1,
        "fitting_worker_count": -1,
        "reserved_core_count": 2,
        "display_progress": False,
    }
    kwargs.update(overrides)
    return outlier_pipeline._detect_all_videos(**kwargs)


def _fit_keypoint_counts() -> dict[str, int]:
    """Builds two videos' keypoint counts for the fitting fan-out, three fit tasks in total."""
    return {"vA": 2, "vB": 1}


def _fake_iter_factory(results):
    """Builds a fake ``iter_pinned_extraction`` that exercises ``make_tasks`` and yields canned results."""

    def fake_iter(*, videos, make_tasks, **_kwargs):
        # Exercise the build_tasks closure that packs the per-video work items.
        tasks = make_tasks(None)
        assert len(tasks) == len(videos)
        assert all(len(task) == 11 for task in tasks)
        yield from results

    return fake_iter


def _install_extract_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralizes the two DeepLabCut module attributes the worker reassigns so they restore after the test."""
    monkeypatch.setattr(outlier_pipeline.frameselectiontools, "KmeansbasedFrameselectioncv2", object(), raising=False)
    monkeypatch.setattr(outlier_pipeline.dlc_outlier_frames, "attempt_to_add_video", object(), raising=False)


def _make_iteration_dir(root: Path, name: str, frame: str, *, refinement: bool = False) -> Path:
    """Fabricates one labeled-data video directory with a single-frame machine-label table for iteration 0."""
    directory = root / "labeled-data" / name
    directory.mkdir(parents=True)
    _write_labels(path=directory / "machinelabels-iter0.h5", names=[frame])
    (directory / frame).write_bytes(b"")
    if refinement:
        (directory / "MachineLabelsRefine.h5").write_bytes(b"stale")
    return directory


def _patch_dlc_boundary(monkeypatch: pytest.MonkeyPatch, configuration: dict) -> None:
    """Stubs the DeepLabCut config/scorer boundary the orchestrator calls before any per-video work."""
    monkeypatch.setattr(outlier_pipeline.auxiliaryfunctions, "read_config", lambda _p: configuration)
    monkeypatch.setattr(
        outlier_pipeline.auxiliaryfunctions,
        "intersection_of_body_parts_and_ones_given_by_user",
        lambda **_k: ["nose"],
    )
    monkeypatch.setattr(outlier_pipeline.auxfun_multianimal, "get_track_method", lambda *_a, **_k: "")
    monkeypatch.setattr(outlier_pipeline.auxiliaryfunctions, "get_scorer_name", lambda **_k: ("scorerX", ""))


def _base_config(video: str) -> dict:
    """Builds the configuration dict ``read_config`` returns during the orchestrator tests."""
    return {
        "video_sets": {video: {}},
        "start": 0.0,
        "stop": 1.0,
        "TrainingFraction": [0.95],
        "numframes2pick": 5,
        "iteration": 0,
        "scorer": "human",
    }


def _recording_supervised_runner(recorded_worker_counts: list[int]):
    """Builds a stand-in for the supervised task runner that records its fan-out and returns one deviation per task."""

    def run(*, tasks, worker, worker_count, role, memory_remedy):
        recorded_worker_counts.append(worker_count)
        return [np.array([0.0, 0.0, 100.0, 0.0, 0.0]) for _ in tasks]

    return run
