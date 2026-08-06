"""Contains tests for the parallel k-means frame-extraction pipeline that prepares a DeepLabCut project's training
frames."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import deeplabcut
from ruamel.yaml import YAML
import deeplabcut.utils.frameselectiontools as frame_selection_tools

from sollertia_video_tracking.frame_extraction import extraction_pipeline
from sollertia_video_tracking.frame_extraction.video_sampling import VideoSamplingPlan


# FrameExtractionSummary dataclass
def test_summary_properties_no_errors() -> None:
    """Verifies that a summary with no errors reports zero failures and success."""
    summary = extraction_pipeline.FrameExtractionSummary(
        extracted_video_count=2,
        cleared_frame_count=0,
        total_video_count=2,
        worker_count=1,
        used_core_count=1,
        total_core_count=8,
        clustering_frame_count=10,
    )
    assert summary.failed_video_count == 0
    assert summary.successful is True


def test_summary_properties_with_errors() -> None:
    """Verifies that a summary carrying errors reports the failure count and unsuccessful status."""
    summary = extraction_pipeline.FrameExtractionSummary(
        extracted_video_count=1,
        cleared_frame_count=3,
        total_video_count=2,
        worker_count=1,
        used_core_count=1,
        total_core_count=8,
        clustering_frame_count=10,
        errors=(("/videos/bad.mp4", "empty"),),
    )
    assert summary.failed_video_count == 1
    assert summary.successful is False


# extract_frames_kmeans: option-validation errors (all raised before any config read)
def test_overwrite_and_reset_are_mutually_exclusive(tmp_path: Path) -> None:
    """Verifies that setting overwrite and reset together is rejected."""
    config_path = tmp_path / "config.yaml"
    with pytest.raises(ValueError, match="mutually exclusive"):
        extraction_pipeline.extract_frames_kmeans(config_path=config_path, overwrite=True, reset=True)


def test_exclusive_requires_requested_videos(tmp_path: Path) -> None:
    """Verifies that exclusive extraction without any requested video is rejected."""
    config_path = tmp_path / "config.yaml"
    with pytest.raises(ValueError, match="requires at least one requested video"):
        extraction_pipeline.extract_frames_kmeans(config_path=config_path, exclusive=True)


def test_exclusive_and_reset_are_contradictory(tmp_path: Path) -> None:
    """Verifies that exclusive and reset together are rejected as contradictory."""
    config_path = tmp_path / "config.yaml"
    with pytest.raises(ValueError, match="contradictory"):
        extraction_pipeline.extract_frames_kmeans(
            config_path=config_path, exclusive=True, reset=True, requested_videos=("/videos/a.mp4",)
        )


def test_total_frame_budget_below_one_rejected(tmp_path: Path) -> None:
    """Verifies that a non-sentinel total frame budget below one is rejected."""
    config_path = tmp_path / "config.yaml"
    with pytest.raises(ValueError, match="total frame budget must be at least one"):
        extraction_pipeline.extract_frames_kmeans(config_path=config_path, total_frame_budget=0)


def test_clustering_stride_below_one_rejected(tmp_path: Path) -> None:
    """Verifies that a clustering stride below one is rejected."""
    config_path = tmp_path / "config.yaml"
    with pytest.raises(ValueError, match="clustering stride must be at least one"):
        extraction_pipeline.extract_frames_kmeans(config_path=config_path, clustering_stride=0)


# extract_frames_kmeans: config-shape errors
def test_missing_video_sets_rejected(tmp_path: Path) -> None:
    """Verifies that a config that defines no video_sets is rejected."""
    project = tmp_path / "project"
    config_path = _write_config(project_directory=project, video_paths=[], include_video_sets=False)
    with pytest.raises(ValueError, match="does not define any video_sets"):
        extraction_pipeline.extract_frames_kmeans(config_path)


def test_empty_video_sets_rejected(tmp_path: Path) -> None:
    """Verifies that a config whose video_sets is present but empty is rejected."""
    project = tmp_path / "project"
    config_path = _write_config(project_directory=project, video_paths=[])
    with pytest.raises(ValueError, match="does not list any videos"):
        extraction_pipeline.extract_frames_kmeans(config_path)


def test_numframes2pick_must_be_positive_integer(tmp_path: Path) -> None:
    """Verifies that a non-integer numframes2pick is rejected."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick="lots")
    with pytest.raises(ValueError, match="numframes2pick must be a positive integer"):
        extraction_pipeline.extract_frames_kmeans(config_path)


# extract_frames_kmeans: full runs through the (faked) worker pool
def test_unbudgeted_run_extracts_all_below_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an unbudgeted run tops up every below-ceiling video and reports the plan when progress is
    displayed."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5, cropping=True)
    _patch_capture(monkeypatch=monkeypatch, frame_count=100)
    monkeypatch.setattr(extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory())

    summary = extraction_pipeline.extract_frames_kmeans(config_path=config_path, display_progress=True)

    assert summary.extracted_video_count == 2
    assert summary.total_video_count == 2
    assert summary.failed_video_count == 0
    assert summary.successful is True
    assert summary.existing_frame_count == 0
    assert summary.target_frame_count == -1
    assert summary.clustering_frame_count == 200  # 100 frames per video, stride 1.
    assert summary.worker_count >= 1


def test_unbudgeted_run_collects_worker_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a worker error is recorded in the summary rather than aborting the run, and progress can be
    disabled."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5)
    _patch_capture(monkeypatch=monkeypatch, frame_count=10)
    monkeypatch.setattr(
        extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory(statuses=["ok", "error:\nboom"])
    )

    summary = extraction_pipeline.extract_frames_kmeans(config_path=config_path, display_progress=False)

    assert summary.extracted_video_count == 1
    assert summary.failed_video_count == 1
    assert summary.successful is False
    assert summary.errors[0][1].startswith("error:")


def test_nothing_to_extract_when_all_videos_at_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that when every candidate already holds the ceiling of frames the run extracts nothing."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5)
    for video in videos:
        _touch_frames(directory=project / "labeled-data" / Path(video).stem, count=5)
    _patch_capture(monkeypatch=monkeypatch, frame_count=10)
    monkeypatch.setattr(extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory())

    summary = extraction_pipeline.extract_frames_kmeans(config_path)

    assert summary.extracted_video_count == 0
    assert summary.total_video_count == 0
    assert summary.existing_frame_count == 0
    assert summary.target_frame_count == -1


def test_reset_clears_and_reextracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that reset clears every candidate video's pre-existing bootstrap frames before re-extraction."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5)
    # Seed two bare (unlabeled) bootstrap frames in each candidate directory. Reset must clear all four before
    # selection.
    for video in videos:
        _touch_frames(directory=project / "labeled-data" / Path(video).stem, count=2)
    _patch_capture(monkeypatch=monkeypatch, frame_count=10)
    monkeypatch.setattr(extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory())

    summary = extraction_pipeline.extract_frames_kmeans(config_path=config_path, reset=True, display_progress=False)

    assert summary.extracted_video_count == 2
    assert summary.cleared_frame_count == 4  # Two bare frames per video were cleared before selection.
    # The seeded frames are physically removed from disk (the faked worker writes nothing back), proving reset cleared.
    for video in videos:
        assert not list((project / "labeled-data" / Path(video).stem).glob("img*.png"))


def test_overwrite_clears_selected_videos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that overwrite clears the selected videos' pre-existing bootstrap frames after selection, then
    re-extracts them."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5)
    # Seed three bare (unlabeled) bootstrap frames in each selected directory. Overwrite must clear all six.
    for video in videos:
        _touch_frames(directory=project / "labeled-data" / Path(video).stem, count=3)
    _patch_capture(monkeypatch=monkeypatch, frame_count=10)
    monkeypatch.setattr(extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory())

    summary = extraction_pipeline.extract_frames_kmeans(config_path=config_path, overwrite=True, display_progress=False)

    assert summary.extracted_video_count == 2
    assert summary.cleared_frame_count == 6  # Three bare frames per selected video were cleared before re-extraction.
    # The seeded frames are physically removed from disk (the faked worker writes nothing back), proving the clear ran.
    for video in videos:
        assert not list((project / "labeled-data" / Path(video).stem).glob("img*.png"))


def test_budgetless_options_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Verifies that budget-only options passed without a budget or exclusive emit a warning."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5)
    _patch_capture(monkeypatch=monkeypatch, frame_count=10)
    monkeypatch.setattr(extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory())

    extraction_pipeline.extract_frames_kmeans(config_path=config_path, balance_groups=True, display_progress=False)

    captured = capsys.readouterr()
    assert "only apply when sampling toward a frame budget" in captured.err


def test_exclusive_run_tops_up_requested_videos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Verifies that exclusive extraction restricts the run to the requested videos and ignores group balancing."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5)
    _patch_capture(monkeypatch=monkeypatch, frame_count=10)
    monkeypatch.setattr(extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory())

    summary = extraction_pipeline.extract_frames_kmeans(
        config_path=config_path,
        exclusive=True,
        requested_videos=(videos[0],),
        balance_groups=True,
        display_progress=True,
    )

    assert summary.total_video_count == 1
    assert summary.extracted_video_count == 1
    captured = capsys.readouterr()
    assert "ignored with --exclusive" in captured.err


def test_exclusive_run_with_no_match_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Verifies that an exclusive run whose requested videos match nothing is rejected, warning about the
    unmatched request."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5)
    _patch_capture(monkeypatch=monkeypatch, frame_count=10)
    monkeypatch.setattr(extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory())

    with pytest.raises(ValueError, match="None of the requested videos matched"):
        extraction_pipeline.extract_frames_kmeans(
            config_path=config_path, exclusive=True, requested_videos=("/nowhere/missing.mp4",), display_progress=False
        )
    captured = capsys.readouterr()
    assert "is not registered in the project's config.yaml" in captured.err


def test_budgeted_run_selects_toward_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a budgeted run selects just enough videos to grow toward the frame budget."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4", "c.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5)
    _patch_capture(monkeypatch=monkeypatch, frame_count=10)
    monkeypatch.setattr(extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory())

    summary = extraction_pipeline.extract_frames_kmeans(
        config_path=config_path, total_frame_budget=5, display_progress=False
    )

    assert summary.target_frame_count == 5
    assert summary.existing_frame_count == 0
    assert summary.extracted_video_count == 1  # One 5-frame video meets a 5-frame budget.


def test_budgeted_run_with_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a budgeted run with group balancing computes groups and reports the per-group breakdown."""
    project = tmp_path / "project"
    videos = _make_videos(
        project_directory=project, names=["M01_2024-01-15.mp4", "M02_2024-01-16.mp4", "M03_2024-01-17.mp4"]
    )
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5)
    _patch_capture(monkeypatch=monkeypatch, frame_count=10)
    monkeypatch.setattr(extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory())

    summary = extraction_pipeline.extract_frames_kmeans(
        config_path=config_path, total_frame_budget=5, balance_groups=True, display_progress=False
    )

    assert summary.target_frame_count == 5
    assert summary.extracted_video_count == 1


def test_budgeted_run_already_met_extracts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Verifies that a budget already met by existing frames extracts nothing and warns."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5)
    for video in videos:
        _touch_frames(directory=project / "labeled-data" / Path(video).stem, count=5)
    _patch_capture(monkeypatch=monkeypatch, frame_count=10)
    monkeypatch.setattr(extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory())

    summary = extraction_pipeline.extract_frames_kmeans(
        config_path=config_path, total_frame_budget=8, display_progress=False
    )

    assert summary.extracted_video_count == 0
    assert summary.existing_frame_count == 10
    assert summary.target_frame_count == 8
    captured = capsys.readouterr()
    assert "already holds" in captured.err


def test_budgeted_run_unreachable_target_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a budget unreachable even by topping every eligible video is rejected."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5)
    _patch_capture(monkeypatch=monkeypatch, frame_count=10)
    monkeypatch.setattr(extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory())

    with pytest.raises(ValueError, match="Unable to reach the requested total"):
        extraction_pipeline.extract_frames_kmeans(
            config_path=config_path, total_frame_budget=100, display_progress=False
        )


def test_requested_refined_video_with_overwrite_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a requested video already in outlier refinement cannot be re-extracted under overwrite."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5)
    refined_directory = project / "labeled-data" / Path(videos[0]).stem
    refined_directory.mkdir(parents=True, exist_ok=True)
    (refined_directory / "machinelabels-iter0.h5").write_bytes(b"x")
    _patch_capture(monkeypatch=monkeypatch, frame_count=10)
    monkeypatch.setattr(extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory())

    with pytest.raises(ValueError, match="already in outlier refinement"):
        extraction_pipeline.extract_frames_kmeans(
            config_path=config_path, overwrite=True, requested_videos=(videos[0],)
        )


def test_requested_refined_video_without_overwrite_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Verifies that a requested refined video is skipped with a warning in budgeted mode, leaving the rest
    to sample."""
    project = tmp_path / "project"
    videos = _make_videos(project_directory=project, names=["a.mp4", "b.mp4", "c.mp4"])
    config_path = _write_config(project_directory=project, video_paths=videos, numframes2pick=5)
    refined_directory = project / "labeled-data" / Path(videos[0]).stem
    refined_directory.mkdir(parents=True, exist_ok=True)
    (refined_directory / "MachineLabelsRefine.h5").write_bytes(b"x")
    _patch_capture(monkeypatch=monkeypatch, frame_count=10)
    monkeypatch.setattr(extraction_pipeline, "iter_pinned_extraction", _fake_iter_factory())

    summary = extraction_pipeline.extract_frames_kmeans(
        config_path=config_path, total_frame_budget=5, requested_videos=(videos[0],), display_progress=False
    )

    assert summary.extracted_video_count == 1
    captured = capsys.readouterr()
    assert "is already in outlier refinement and was skipped" in captured.err


# _report_plan / _report_sampling_plan
def test_report_plan_writes_header(capsys: pytest.CaptureFixture) -> None:
    """Verifies that the run header names the plan and core usage."""
    extraction_pipeline._report_plan(
        video_count=2,
        configured_frames_per_video=5,
        clustering_stride=1,
        clustering_resize_width=30,
        cluster_in_color=False,
        worker_count=2,
        used_core_count=8,
        total_core_count=10,
        clustering_frame_count=1234,
        config_path=Path("/project/config.yaml"),
    )
    captured = capsys.readouterr()
    assert "k-means extraction | 2 videos" in captured.err
    assert "workers=2 | 8/10 cores used (2 free)" in captured.err


def test_report_sampling_plan_budget_met(capsys: pytest.CaptureFixture) -> None:
    """Verifies that the budget-already-met plan warns and reports nothing to extract."""
    plan = VideoSamplingPlan(
        selected_videos=(),
        existing_frame_count=100,
        target_frame_count=80,
        projected_frame_count=100,
        budget_already_met=True,
        target_unreachable=False,
    )
    extraction_pipeline._report_sampling_plan(plan=plan)
    captured = capsys.readouterr()
    assert "already holds 100 frames" in captured.err


def test_report_sampling_plan_without_groups(capsys: pytest.CaptureFixture) -> None:
    """Verifies that an ungrouped sampling plan reports the existing-to-projected frame growth."""
    plan = VideoSamplingPlan(
        selected_videos=("a", "b"),
        existing_frame_count=10,
        target_frame_count=30,
        projected_frame_count=30,
        budget_already_met=False,
        target_unreachable=False,
    )
    extraction_pipeline._report_sampling_plan(plan=plan)
    captured = capsys.readouterr()
    assert "sampling 2 video(s)" in captured.err
    assert "10 existing -> 30 projected" in captured.err


def test_report_sampling_plan_with_groups_and_overshoot(capsys: pytest.CaptureFixture) -> None:
    """Verifies that a grouped plan reports the distribution, flags starved groups, and warns on pinned overshoot."""
    per_group = (
        ("g1", 0, 2, 10, 3),  # Sampled: two videos added.
        ("g2", 5, 0, 5, 1),  # Starved: below-ceiling videos available but none added.
        ("g3", 0, 0, 0, 0),  # Fully done: nothing available, so not starved.
    )
    plan = VideoSamplingPlan(
        selected_videos=("a", "b"),
        existing_frame_count=5,
        target_frame_count=30,
        projected_frame_count=25,
        budget_already_met=False,
        target_unreachable=False,
        per_group=per_group,
        always_included_overshoot=True,
    )
    extraction_pipeline._report_sampling_plan(plan=plan)
    captured = capsys.readouterr()
    assert "group balancing: 1/3 groups sampled" in captured.err
    assert "g1+2" in captured.err
    assert "1 group(s) had below-ceiling videos but received none" in captured.err
    assert "videos named with --videos alone exceeded the frame budget" in captured.err


# _count_extracted_frames / _count_clustering_frames
def test_count_extracted_frames_excludes_overlays(tmp_path: Path) -> None:
    """Verifies that extracted-frame counts exclude prediction overlays and cover absent directories."""
    project = tmp_path / "project"
    labeled = project / "labeled-data"
    _touch_frames(directory=labeled / "a", count=3)
    (labeled / "a" / "img0000labeled.png").write_bytes(b"")  # Overlay, must not be counted.
    counts = extraction_pipeline._count_extracted_frames(
        videos=["/videos/a.mp4", "/videos/b.mp4"], project_directory=project
    )
    assert counts == {"/videos/a.mp4": 3, "/videos/b.mp4": 0}


def test_count_clustering_frames_uses_bounds_and_stride(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the clustering-frame count mirrors DeepLabCut's start/stop/stride sampling per video."""
    _patch_capture(monkeypatch=monkeypatch, frame_count=100)
    totals = extraction_pipeline._count_clustering_frames(
        videos=["/videos/a.mp4", "/videos/b.mp4"], start_fraction=0.0, stop_fraction=1.0, clustering_stride=3
    )
    # range(0, 100, 3) -> 34 frames sampled for each video.
    assert totals == {0: 34, 1: 34}


def test_count_clustering_frames_clamps_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a video whose bounds yield no strided frames still contributes at least one frame."""
    _patch_capture(monkeypatch=monkeypatch, frame_count=0)
    totals = extraction_pipeline._count_clustering_frames(
        videos=["/videos/a.mp4"], start_fraction=0.0, stop_fraction=1.0, clustering_stride=1
    )
    assert totals == {0: 1}


# _clear_bare_frames / _clear_bare_frames_in_directory
def test_clear_bare_frames_in_directory_preserves_labeled_and_machine_frames(tmp_path: Path) -> None:
    """Verifies that clearing removes only unlabeled bootstrap frames, keeping human-labeled and
    machine-labeled ones."""
    directory = tmp_path / "labeled-data" / "vid"
    _touch_frames(directory=directory, count=4)  # img0000..img0003
    (directory / "img0002labeled.png").write_bytes(b"")  # Overlay for a bare frame, dropped defensively.
    _write_collected_data(
        path=directory / "CollectedData_tester.h5",
        rows=[("img0000.png", True), ("img0002.png", False)],  # img0000 finite (kept), and img0002 placeholder row.
    )
    _write_machine_labels(path=directory / "machinelabels-iter0.h5", image_names=["img0001.png"])  # img0001 kept.

    removed = extraction_pipeline._clear_bare_frames_in_directory(directory=directory, scorer="tester")

    assert removed == 2  # img0002 and img0003 were bare.
    assert (directory / "img0000.png").exists()
    assert (directory / "img0001.png").exists()
    assert not (directory / "img0002.png").exists()
    assert not (directory / "img0003.png").exists()
    assert not (directory / "img0002labeled.png").exists()
    # The placeholder row for img0002 was dropped, leaving only the finite img0000 row.
    remaining = pd.read_hdf(path_or_buf=directory / "CollectedData_tester.h5", key="df_with_missing")
    assert [entry[-1] for entry in remaining.index] == ["img0000.png"]


def test_clear_bare_frames_in_directory_empty_directory(tmp_path: Path) -> None:
    """Verifies that a directory with no extracted frames clears nothing."""
    directory = tmp_path / "labeled-data" / "vid"
    directory.mkdir(parents=True)
    assert extraction_pipeline._clear_bare_frames_in_directory(directory=directory, scorer="tester") == 0


def test_clear_bare_frames_reports_and_survives_unreadable(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Verifies that the clearing wrapper aggregates removals across directories and warns on an unreadable one."""
    project = tmp_path / "project"
    labeled = project / "labeled-data"
    # A clean directory with two bare frames and no labels: both are removed.
    _touch_frames(directory=labeled / "clean", count=2)
    # A directory whose label table cannot be read: warned about, left untouched.
    _touch_frames(directory=labeled / "broken", count=1)
    (labeled / "broken" / "CollectedData_tester.h5").write_bytes(b"not a real hdf file")

    removed_count, cleared_stems = extraction_pipeline._clear_bare_frames(
        project_directory=project, video_stems=["clean", "broken"], scorer="tester", scope_label="--reset"
    )

    assert removed_count == 2
    assert cleared_stems == {"clean"}
    captured = capsys.readouterr()
    assert "could not clear the unlabeled frames" in captured.err
    assert "--reset cleared 2 unlabeled bootstrap frame(s)" in captured.err


# _extract_one_video (the per-video worker)
def test_extract_one_video_success_without_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a successful extraction with no progress queue reports the freshly written frame count."""
    project = tmp_path / "project"
    config_path = _write_config(project_directory=project, video_paths=[], include_video_sets=False)
    video_path = str(project / "videos" / "a.mp4")
    extract_calls, selector_calls, reporter_calls = _install_worker_stubs(monkeypatch=monkeypatch, write_count=2)

    result = extraction_pipeline._extract_one_video(
        _worker_task(config_path=config_path, video_path=video_path, progress_queue=None, crop_frames=True)
    )

    assert result == (video_path, 2, "ok")
    assert selector_calls == [{"progress": None, "frame_count": 7}]  # No queue -> no reporter.
    assert reporter_calls == []
    call = extract_calls[0]
    assert call["config"] == str(config_path)
    assert call["mode"] == "automatic"
    assert call["algo"] == "kmeans"
    assert call["crop"] is True
    assert call["userfeedback"] is False
    assert call["cluster_step"] == 3
    assert call["cluster_resizewidth"] == 30
    assert call["cluster_color"] is False
    assert call["videos_list"] == [video_path]


def test_extract_one_video_success_with_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a progress queue routes the k-means selector through the streaming progress reporter."""
    project = tmp_path / "project"
    config_path = _write_config(project_directory=project, video_paths=[], include_video_sets=False)
    video_path = str(project / "videos" / "a.mp4")
    _extract_calls, selector_calls, reporter_calls = _install_worker_stubs(monkeypatch=monkeypatch, write_count=1)

    result = extraction_pipeline._extract_one_video(
        _worker_task(config_path=config_path, video_path=video_path, progress_queue=object())
    )

    assert result == (video_path, 1, "ok")
    assert reporter_calls == [(0, 50)]
    assert selector_calls[0]["progress"] == "REPORTER"


def test_extract_one_video_empty_when_no_frames_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an extraction that writes no frames is reported as empty."""
    project = tmp_path / "project"
    config_path = _write_config(project_directory=project, video_paths=[], include_video_sets=False)
    video_path = str(project / "videos" / "a.mp4")
    _install_worker_stubs(monkeypatch=monkeypatch, write_count=0)

    result = extraction_pipeline._extract_one_video(
        _worker_task(config_path=config_path, video_path=video_path, progress_queue=None)
    )

    assert result == (video_path, 0, "empty")


def test_extract_one_video_captures_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a failing extraction returns an error status carrying the traceback rather than raising."""
    project = tmp_path / "project"
    config_path = _write_config(project_directory=project, video_paths=[], include_video_sets=False)
    video_path = str(project / "videos" / "a.mp4")
    _install_worker_stubs(monkeypatch=monkeypatch, write_count=0, raise_exc=RuntimeError("decode failed"))

    video, written, status = extraction_pipeline._extract_one_video(
        _worker_task(config_path=config_path, video_path=video_path, progress_queue=None)
    )

    assert video == video_path
    assert written == 0
    assert status.startswith("error:")
    assert "decode failed" in status


# Helpers
def _write_config(
    project_directory: Path,
    video_paths: list[str],
    *,
    numframes2pick: object = 5,
    scorer: str = "tester",
    cropping: bool = False,
    start: float = 0.0,
    stop: float = 1.0,
    include_video_sets: bool = True,
) -> Path:
    """Writes a minimal DeepLabCut ``config.yaml`` and creates the ``labeled-data`` tree, returning the config path."""
    project_directory.mkdir(parents=True, exist_ok=True)
    (project_directory / "labeled-data").mkdir(exist_ok=True)
    configuration: dict[str, object] = {
        "project_path": str(project_directory),
        "scorer": scorer,
        "start": start,
        "stop": stop,
        "cropping": cropping,
        "iteration": 0,
    }
    if numframes2pick is not None:
        configuration["numframes2pick"] = numframes2pick
    if include_video_sets:
        configuration["video_sets"] = {video: {"crop": "0, 640, 0, 480"} for video in video_paths}
    config_path = project_directory / "config.yaml"
    with config_path.open("w") as config_file:
        YAML().dump(data=configuration, stream=config_file)
    return config_path


def _make_videos(project_directory: Path, names: list[str]) -> list[str]:
    """Creates empty video files under the project and returns their absolute path strings."""
    video_directory = project_directory / "videos"
    video_directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for name in names:
        path = video_directory / name
        path.write_bytes(b"")
        paths.append(str(path))
    return paths


def _touch_frames(directory: Path, count: int, *, start: int = 0) -> None:
    """Creates ``imgNNNN.png`` placeholder frames in a labeled-data directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(start, start + count):
        (directory / f"img{index:04d}.png").write_bytes(b"")


def _write_collected_data(
    path: Path, rows: list[tuple[str, bool]], *, stem: str = "vid", scorer: str = "tester"
) -> None:
    """Writes a ``CollectedData`` label table where each row is ``(image_name, is_finite)``."""
    index = pd.MultiIndex.from_tuples([("labeled-data", stem, image) for image, _finite in rows])
    columns = pd.MultiIndex.from_tuples([(scorer, "bodypart", "x"), (scorer, "bodypart", "y")])
    data = [[1.0, 2.0] if finite else [np.nan, np.nan] for _image, finite in rows]
    frame = pd.DataFrame(data=data, index=index, columns=columns)
    frame.to_hdf(path_or_buf=path, key="df_with_missing", mode="w")


def _write_machine_labels(path: Path, image_names: list[str], *, stem: str = "vid") -> None:
    """Writes a machine-label table referencing the given image names."""
    index = pd.MultiIndex.from_tuples([("labeled-data", stem, image) for image in image_names])
    frame = pd.DataFrame(data={"scorer": list(range(len(image_names)))}, index=index)
    frame.to_hdf(path_or_buf=path, key="df_with_missing", mode="w")


def _patch_capture(monkeypatch: pytest.MonkeyPatch, frame_count: int) -> None:
    """Replaces ``cv2.VideoCapture`` with a fake yielding a fixed frame count so no real decode occurs."""

    class _FakeCapture:
        def __init__(self, path: str) -> None:
            self._path = path

        def get(self, _prop: int) -> float:
            return float(frame_count)

        def release(self) -> None:
            pass

    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)


def _fake_iter_factory(statuses: list[str] | None = None):
    """Builds a synchronous stand-in for ``iter_pinned_extraction`` that never spawns a process pool."""

    def _fake_iter(*, make_tasks, **_ignored):
        # The remaining keyword arguments (videos, worker, worker_count, core_sets, frame_totals, display_progress)
        # mirror the real iter_pinned_extraction signature but are unused by this synchronous stand-in.
        # Building the tasks exercises the pipeline's inner build_tasks closure without running any worker.
        tasks = make_tasks(None)
        for index, task in enumerate(tasks):
            video = task[0]
            pick_count = task[7]
            status = statuses[index] if statuses is not None else "ok"
            yield video, pick_count, status

    return _fake_iter


def _install_worker_stubs(
    monkeypatch: pytest.MonkeyPatch, *, write_count: int, raise_exc: Exception | None = None
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    """Installs stubs for the DeepLabCut extract call, the k-means selector, and the progress reporter."""
    extract_calls: list[dict] = []
    selector_calls: list[dict] = []
    reporter_calls: list[tuple[int, int]] = []

    def _stub_extract_frames(**kwargs: object) -> None:
        extract_calls.append(kwargs)
        if raise_exc is not None:
            raise raise_exc
        output_directory = (
            Path(str(kwargs["config"])).parent / "labeled-data" / Path(str(kwargs["videos_list"][0])).stem
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        for index in range(write_count):
            (output_directory / f"img{index:04d}.png").write_bytes(b"")

    def _stub_selector(*, progress: object = None, frame_count: object = None) -> str:
        selector_calls.append({"progress": progress, "frame_count": frame_count})
        return "SELECTOR"

    def _stub_reporter(*, video_index: int, frame_total: int, **_ignored: object) -> str:
        # The progress_queue keyword the worker passes is absorbed by **_ignored. Only the identifying counts matter.
        reporter_calls.append((video_index, frame_total))
        return "REPORTER"

    monkeypatch.setattr(deeplabcut, "extract_frames", _stub_extract_frames)
    monkeypatch.setattr(extraction_pipeline, "make_fast_kmeans_selector", _stub_selector)
    monkeypatch.setattr(extraction_pipeline, "make_progress_reporter", _stub_reporter)
    # Track the module attribute the worker overwrites so monkeypatch restores it after the test.
    monkeypatch.setattr(frame_selection_tools, "KmeansbasedFrameselectioncv2", object(), raising=False)
    return extract_calls, selector_calls, reporter_calls


def _worker_task(config_path: Path, video_path: str, *, progress_queue: object, crop_frames: bool = False):
    """Packs a worker task tuple in the pipeline's positional order."""
    return (
        video_path,
        config_path,
        3,  # clustering_stride
        30,  # clustering_resize_width
        False,  # cluster_in_color
        0,  # video_index
        50,  # frame_total
        7,  # pick_count
        progress_queue,
        crop_frames,
    )
