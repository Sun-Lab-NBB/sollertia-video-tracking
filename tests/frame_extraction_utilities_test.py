"""Contains tests for the shared frame-extraction utilities the k-means and outlier pipelines both build on."""

import queue
from typing import ClassVar
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ruamel.yaml import YAML

from sollertia_video_tracking.frame_extraction import utilities
from sollertia_video_tracking.frame_extraction.utilities import (
    PurgeSummary,
    RefinementStatusSummary,
    RefinementDirectoryStatus,
    purge_labeled_data,
    extracted_frame_paths,
    frame_names_from_index,
    iter_pinned_extraction,
    drop_collected_data_rows,
    normalize_project_config,
    select_registered_videos,
    ensure_unique_video_stems,
    machine_label_frame_names,
    finite_labeled_frame_names,
    has_outlier_refinement_data,
    summarize_refinement_status,
    prune_empty_labeled_data_directories,
)


# Dataclasses and their properties
def test_purge_summary_count_properties():
    """Verifies that PurgeSummary reports the removed and human-labeled directory counts from its stored tuples."""
    summary = PurgeSummary(
        config_path=Path("cfg.yaml"),
        executed=False,
        removed_directories=(Path("a"), Path("b")),
        labeled_directories=(Path("a"),),
        frame_count=3,
    )
    assert summary.removed_directory_count == 2
    assert summary.labeled_directory_count == 1
    assert summary.unmatched_videos == ()


def test_refinement_status_summary_properties_and_describe():
    """Verifies that RefinementStatusSummary aggregates directory and frame counts, success, and its description."""
    pending = (
        RefinementDirectoryStatus(directory=Path("d1"), unrefined_frame_count=2),
        RefinementDirectoryStatus(directory=Path("d2"), unrefined_frame_count=3),
    )
    summary = RefinementStatusSummary(config_path=Path("cfg.yaml"), iteration=4, pending_directories=pending)
    assert summary.pending_directory_count == 2
    assert summary.pending_frame_count == 5
    assert summary.successful is True

    description = summary.describe()
    assert "2 directory(ies)" in description
    assert "5 frame(s)" in description
    assert "iteration 4" in description


def test_refinement_status_summary_unsuccessful_when_unreadable():
    """Verifies that a summary carrying an unreadable directory reports itself as not fully successful."""
    summary = RefinementStatusSummary(
        config_path=Path("cfg.yaml"),
        iteration=0,
        pending_directories=(),
        unreadable=((Path("bad"), "boom"),),
    )
    assert summary.successful is False
    assert summary.pending_frame_count == 0


# normalize_project_config
def test_normalize_project_config_rewrites_project_path(tmp_path):
    """Verifies that a project_path that differs from the config's own directory is normalized and persisted."""
    config_path = tmp_path / "config.yaml"
    _write_config(path=config_path, data={"project_path": "/wrong/place", "scorer": "S"})

    result = normalize_project_config(config_path=config_path, frames_per_video=-1, error_context="Ctx.")

    assert result["project_path"] == str(tmp_path)
    reloaded = YAML().load(config_path.read_text())
    assert reloaded["project_path"] == str(tmp_path)
    # The -1 sentinel leaves the frame count untouched.
    assert "numframes2pick" not in reloaded


def test_normalize_project_config_sets_frame_count(tmp_path):
    """Verifies that a positive frames_per_video is written as numframes2pick beside a correct project_path."""
    config_path = tmp_path / "config.yaml"
    _write_config(path=config_path, data={"project_path": str(tmp_path), "scorer": "S"})

    result = normalize_project_config(config_path=config_path, frames_per_video=5, error_context="Ctx.")

    assert result["numframes2pick"] == 5
    reloaded = YAML().load(config_path.read_text())
    assert reloaded["numframes2pick"] == 5


def test_normalize_project_config_no_change_skips_rewrite(tmp_path):
    """Verifies that when nothing needs changing, the config file is left byte-for-byte untouched."""
    config_path = tmp_path / "config.yaml"
    _write_config(path=config_path, data={"project_path": str(tmp_path), "scorer": "S"})
    before = config_path.read_text()

    result = normalize_project_config(config_path=config_path, frames_per_video=-1, error_context="Ctx.")

    assert result["project_path"] == str(tmp_path)
    assert config_path.read_text() == before


def test_normalize_project_config_rejects_bad_frame_count(tmp_path):
    """Verifies that a frame count below one that is not the -1 sentinel raises a ValueError naming the operation."""
    config_path = tmp_path / "config.yaml"
    _write_config(path=config_path, data={"project_path": str(tmp_path), "scorer": "S"})

    with pytest.raises(ValueError, match="at least one"):
        normalize_project_config(config_path=config_path, frames_per_video=0, error_context="Ctx.")


# iter_pinned_extraction (multiprocessing driven synchronously via fakes)
def test_iter_pinned_extraction_with_progress(monkeypatch):
    """Verifies that with progress on, the loop manages the bar, streams done messages, and shuts the manager down."""
    fake_context = _patch_pinned_extraction(monkeypatch)

    videos = ["/v/a.mp4", "/v/b.mp4"]
    captured = {}

    def make_tasks(progress_queue):
        captured["progress_queue"] = progress_queue
        return [("/v/a.mp4",), ("/v/b.mp4",)]

    def worker(task):
        return (task[0], 7, "written")

    results = list(
        iter_pinned_extraction(
            videos=videos,
            make_tasks=make_tasks,
            worker=worker,
            worker_count=2,
            core_sets=[{0, 1}, {2, 3}],
            frame_totals={0: 10, 1: 10},
            display_progress=True,
        ),
    )

    assert results == [("/v/a.mp4", 7, "written"), ("/v/b.mp4", 7, "written")]
    # The progress queue is handed to make_tasks when progress is displayed.
    assert captured["progress_queue"] is not None

    bar = _FakeBar.instances[-1]
    assert bar.started is True
    assert bar.stopped is True
    assert bar.joined is True
    assert bar.total_video_count == 2

    # The pool is pinned with the core-set queue and the real pinning initializer.
    pool = fake_context.pools[-1]
    assert pool.processes == 2
    assert pool.initializer is utilities.pin_worker_to_cores

    # A completion message is enqueued for each finished video.
    progress_queue = captured["progress_queue"]
    drained = []
    while not progress_queue.empty():
        drained.append(progress_queue.get_nowait())
    assert ("done", 0) in drained
    assert ("done", 1) in drained

    # The core-set queue was primed with one set per worker.
    core_set_queue = fake_context.manager.queues[1]
    remaining = []
    while not core_set_queue.empty():
        remaining.append(core_set_queue.get_nowait())
    assert remaining == [{0, 1}, {2, 3}]

    assert fake_context.manager.shutdown_called is True


def test_iter_pinned_extraction_without_progress(monkeypatch):
    """Verifies that with progress off, the bar never starts, make_tasks gets None, and the manager still shuts down."""
    fake_context = _patch_pinned_extraction(monkeypatch)

    captured = {}

    def make_tasks(progress_queue):
        captured["progress_queue"] = progress_queue
        return [("/v/only.mp4",)]

    def worker(task):
        return (task[0], 0, "empty")

    results = list(
        iter_pinned_extraction(
            videos=["/v/only.mp4"],
            make_tasks=make_tasks,
            worker=worker,
            worker_count=1,
            core_sets=[{0}],
            frame_totals={0: 5},
            display_progress=False,
        ),
    )

    assert results == [("/v/only.mp4", 0, "empty")]
    # No queue is streamed to when progress is off.
    assert captured["progress_queue"] is None

    bar = _FakeBar.instances[-1]
    assert bar.started is False
    assert bar.stopped is False
    assert bar.joined is False

    assert fake_context.manager.shutdown_called is True


def test_iter_pinned_extraction_tears_down_when_consumption_stops_early(monkeypatch):
    """Verifies that abandoning the generator early still tears down the bar and manager via the finally block.

    The loop documents that the bar and manager are always torn down even when the caller stops consuming results (or
    raises) mid-stream. Consuming a single result and then closing the still-suspended generator throws GeneratorExit
    at the yield, so the teardown must run from the finally rather than only on normal completion.
    """
    fake_context = _patch_pinned_extraction(monkeypatch)

    def make_tasks(_progress_queue):
        return [("/v/a.mp4",), ("/v/b.mp4",)]

    def worker(task):
        return (task[0], 1, "written")

    generator = iter_pinned_extraction(
        videos=["/v/a.mp4", "/v/b.mp4"],
        make_tasks=make_tasks,
        worker=worker,
        worker_count=2,
        core_sets=[{0}, {1}],
        frame_totals={0: 1, 1: 1},
        display_progress=True,
    )

    # Consume exactly one result: the bar is now started and the generator is suspended at its yield.
    assert next(generator) == ("/v/a.mp4", 1, "written")
    bar = _FakeBar.instances[-1]
    assert bar.started is True
    # Teardown has not run yet while the generator is merely suspended.
    assert bar.stopped is False
    assert fake_context.manager.shutdown_called is False

    # Abandon consumption early. GeneratorExit is thrown at the yield, so teardown must run from the finally block.
    generator.close()

    assert bar.stopped is True
    assert bar.joined is True
    assert bar.join_timeout == 3
    assert fake_context.manager.shutdown_called is True


# prune_empty_labeled_data_directories
def test_prune_no_labeled_data_directory(tmp_path):
    """Verifies that a project without a labeled-data tree prunes nothing and reports zero."""
    assert prune_empty_labeled_data_directories(project_directory=tmp_path) == 0


def test_prune_removes_only_empty_real_directories(tmp_path, capsys):
    """Verifies that only empty real directories are pruned. Populated dirs, symlinks, and stray files are kept."""
    labeled = tmp_path / "labeled-data"
    labeled.mkdir()

    empty = labeled / "empty1"
    empty.mkdir()

    full = labeled / "full1"
    full.mkdir()
    (full / "img0001.png").touch()

    (labeled / "afile.txt").touch()  # not a directory -> skipped

    link_target = tmp_path / "link_target"
    link_target.mkdir()
    (labeled / "linkdir").symlink_to(link_target, target_is_directory=True)  # symlink -> skipped

    removed = prune_empty_labeled_data_directories(project_directory=tmp_path, display_progress=True)

    assert removed == 1
    assert not empty.exists()
    assert full.exists()
    assert (labeled / "linkdir").exists()
    assert (labeled / "afile.txt").exists()
    assert "pruned 1 empty labeled-data directory(ies)" in capsys.readouterr().err


# extracted_frame_paths
def test_extracted_frame_paths_missing_directory(tmp_path):
    """Verifies that a directory that does not exist yields an empty list."""
    assert extracted_frame_paths(directory=tmp_path / "nope") == []


def test_extracted_frame_paths_excludes_labeled_overlays(tmp_path):
    """Verifies that prediction overlays and non-image files are excluded from the sorted extracted-frame listing."""
    directory = tmp_path / "vid"
    directory.mkdir()
    (directory / "img0001.png").touch()
    (directory / "img0010.png").touch()
    (directory / "img0002labeled.png").touch()  # overlay -> excluded
    (directory / "notes.txt").touch()  # non-image -> excluded

    assert extracted_frame_paths(directory=directory) == [directory / "img0001.png", directory / "img0010.png"]


# frame_names_from_index
def test_frame_names_from_index_handles_tuple_and_flat_entries():
    """Verifies that both the MultiIndex tuple form and the flat-path-string form yield the trailing image name."""
    names = frame_names_from_index(
        frame_index=[
            ("labeled-data", "vid", "img0001.png"),
            "labeled-data/vid/img0002.png",
        ],
    )
    assert names == {"img0001.png", "img0002.png"}


# finite_labeled_frame_names
def test_finite_labeled_frame_names_missing_file(tmp_path):
    """Verifies that a missing CollectedData file yields an empty set."""
    assert finite_labeled_frame_names(collected_data_path=tmp_path / "nope.h5") == set()


def test_finite_labeled_frame_names_filters_all_nan_rows(tmp_path):
    """Verifies that only frames carrying at least one finite coordinate count as human-labeled."""
    path = tmp_path / "CollectedData_S.h5"
    _write_label_table(path=path, frame_names=["img0001.png", "img0002.png"], finite_names={"img0001.png"})

    assert finite_labeled_frame_names(collected_data_path=path) == {"img0001.png"}


# machine_label_frame_names
def test_machine_label_frame_names_unions_all_tables(tmp_path):
    """Verifies that every machinelabels iteration table and the refine table contribute their frame names."""
    directory = tmp_path / "vid"
    directory.mkdir()
    _write_label_table(path=directory / "machinelabels-iter0.h5", frame_names=["img0001.png"])
    _write_label_table(path=directory / "machinelabels-iter1.h5", frame_names=["img0002.png"])
    _write_label_table(path=directory / "MachineLabelsRefine.h5", frame_names=["img0003.png"])

    assert machine_label_frame_names(directory=directory) == {"img0001.png", "img0002.png", "img0003.png"}


def test_machine_label_frame_names_empty_directory(tmp_path):
    """Verifies that a directory with no machine-label tables yields an empty set."""
    directory = tmp_path / "empty"
    directory.mkdir()
    assert machine_label_frame_names(directory=directory) == set()


# drop_collected_data_rows
def test_drop_collected_data_rows_missing_file_is_noop(tmp_path):
    """Verifies that dropping rows from an absent label table does nothing."""
    drop_collected_data_rows(collected_data_path=tmp_path / "absent.h5", removed_frame_names={"img0000.png"})
    assert not (tmp_path / "absent.h5").exists()


def test_drop_collected_data_rows_keeps_all_when_none_removed(tmp_path):
    """Verifies that when no row references a removed frame the table is left untouched."""
    path = tmp_path / "CollectedData_tester.h5"
    _write_label_table(path=path, frame_names=["img0000.png", "img0001.png"])
    drop_collected_data_rows(collected_data_path=path, removed_frame_names={"img9999.png"})
    assert len(pd.read_hdf(path_or_buf=path, key="df_with_missing")) == 2


def test_drop_collected_data_rows_deletes_emptied_table(tmp_path):
    """Verifies that dropping every remaining row deletes both the h5 and its csv sibling."""
    path = tmp_path / "CollectedData_tester.h5"
    csv_path = path.with_suffix(".csv")
    _write_label_table(path=path, frame_names=["img0000.png", "img0001.png"], finite_names={"img0000.png"})
    csv_path.write_text("placeholder csv\n")
    drop_collected_data_rows(collected_data_path=path, removed_frame_names={"img0000.png", "img0001.png"})
    assert not path.exists()
    assert not csv_path.exists()


def test_drop_collected_data_rows_flat_index_partial_removal(tmp_path):
    """Verifies that a flat-string index row is parsed by file name, and the surviving rows are rewritten to h5 and
    csv."""
    path = tmp_path / "CollectedData_tester.h5"
    frame = pd.DataFrame(
        data={"x": [1.0, 2.0]},
        index=pd.Index(["labeled-data/vid/img0000.png", "labeled-data/vid/img0001.png"]),
    )
    frame.to_hdf(path_or_buf=path, key="df_with_missing", mode="w")
    drop_collected_data_rows(collected_data_path=path, removed_frame_names={"img0000.png"})
    remaining = pd.read_hdf(path_or_buf=path, key="df_with_missing")
    assert [Path(str(entry)).name for entry in remaining.index] == ["img0001.png"]
    assert path.with_suffix(".csv").is_file()


# has_outlier_refinement_data
def test_has_outlier_refinement_data(tmp_path):
    """Verifies that a machinelabels iteration table or a refine table marks a directory as holding refinement data."""
    with_iteration = tmp_path / "with_iter"
    with_iteration.mkdir()
    (with_iteration / "machinelabels-iter3.h5").touch()
    assert has_outlier_refinement_data(directory=with_iteration) is True

    with_refine = tmp_path / "with_refine"
    with_refine.mkdir()
    (with_refine / "MachineLabelsRefine.h5").touch()
    assert has_outlier_refinement_data(directory=with_refine) is True

    empty = tmp_path / "empty"
    empty.mkdir()
    assert has_outlier_refinement_data(directory=empty) is False


# select_registered_videos
def test_select_registered_videos_matches_and_reports_unmatched(tmp_path):
    """Verifies that requests matching a registered video by resolved path return in order. The rest are unmatched."""
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    registered = [str(video_a), str(video_b)]
    bogus = "/nonexistent/z.mp4"

    matched, unmatched = select_registered_videos(
        registered_videos=registered, requested_videos=(str(video_a), bogus, str(video_a))
    )

    assert matched == [str(video_a)]
    assert unmatched == [bogus]


# ensure_unique_video_stems
def test_ensure_unique_video_stems_allows_distinct_and_duplicate_paths():
    """Verifies that distinct stems and an exact duplicate path do not collide."""
    # The repeated identical path must not be treated as a collision.
    ensure_unique_video_stems(videos=["/x/a.mp4", "/x/a.mp4", "/y/b.mp4"], error_context="Ctx.")


def test_ensure_unique_video_stems_raises_on_collision():
    """Verifies that two different paths sharing a file-name stem raise a ValueError."""
    with pytest.raises(ValueError, match="share the file-name stem"):
        ensure_unique_video_stems(videos=["/x/vid.mp4", "/y/vid.mp4"], error_context="Ctx.")


# purge_labeled_data
def test_purge_missing_config_raises(tmp_path):
    """Verifies that a config path that is not a file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="does not point to a file"):
        purge_labeled_data(config_path=tmp_path / "missing.yaml")


def test_purge_whole_project_dry_run(tmp_path):
    """Verifies that a dry run over the whole project reports every real per-video directory and deletes nothing."""
    config_path, _, _, labeled = _build_purge_project(tmp_path=tmp_path)

    summary = purge_labeled_data(config_path=config_path)

    assert summary.executed is False
    assert summary.removed_directories == (labeled / "vidA", labeled / "vidB")
    assert summary.removed_directory_count == 2
    assert summary.labeled_directories == (labeled / "vidA",)
    assert summary.labeled_directory_count == 1
    assert summary.frame_count == 3
    assert summary.unmatched_videos == ()
    assert summary.config_path == config_path.resolve()
    # Nothing removed on a dry run.
    assert (labeled / "vidA").exists()
    assert (labeled / "vidB").exists()


def test_purge_whole_project_execute(tmp_path):
    """Verifies that an executed purge removes the per-video directories but leaves the excluded entries alone."""
    config_path, _, _, labeled = _build_purge_project(tmp_path=tmp_path)

    summary = purge_labeled_data(config_path=config_path, execute=True)

    assert summary.executed is True
    assert not (labeled / "vidA").exists()
    assert not (labeled / "vidB").exists()
    # Overlay, hidden, and symlinked entries are never targeted.
    assert (labeled / "vidA_labeled").exists()
    assert (labeled / ".hidden").exists()
    assert (labeled / "linkdir").exists()


def test_purge_selected_videos_with_unmatched(tmp_path):
    """Verifies that selecting specific videos purges only their directories and reports videos that matched nothing."""
    config_path, video_a, _, labeled = _build_purge_project(tmp_path=tmp_path, with_video_sets=True)
    bogus = tmp_path / "videos" / "nope.mp4"

    summary = purge_labeled_data(config_path=config_path, videos=(str(video_a), str(bogus)))

    assert summary.removed_directories == (labeled / "vidA",)
    assert summary.unmatched_videos == (str(bogus),)
    assert summary.executed is False


def test_purge_whole_project_without_labeled_data(tmp_path):
    """Verifies that a project whose labeled-data tree does not exist purges nothing."""
    config_path = tmp_path / "config.yaml"
    _write_config(path=config_path, data={"scorer": "S", "project_path": str(tmp_path)})

    summary = purge_labeled_data(config_path=config_path)

    assert summary.removed_directories == ()
    assert summary.frame_count == 0
    assert summary.labeled_directories == ()


# summarize_refinement_status
def test_summarize_missing_config_raises(tmp_path):
    """Verifies that a config path that is not a file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="does not point to a file"):
        summarize_refinement_status(config_path=tmp_path / "missing.yaml")


def test_summarize_whole_project(tmp_path):
    """Verifies that the whole-project scan lists directories with unrefined frames and records unreadable ones."""
    scorer = "S"
    config_path = tmp_path / "config.yaml"
    _write_config(path=config_path, data={"scorer": scorer, "iteration": 0, "project_path": str(tmp_path)})

    labeled = tmp_path / "labeled-data"
    labeled.mkdir()

    # A pending directory: two machine frames, only one refined by finite human labels.
    pending = labeled / "vidPending"
    pending.mkdir()
    _write_label_table(path=pending / "machinelabels-iter0.h5", frame_names=["img0001.png", "img0002.png"])
    _write_label_table(
        path=pending / f"CollectedData_{scorer}.h5",
        frame_names=["img0001.png", "img0002.png"],
        finite_names={"img0001.png"},
    )

    # A fully refined directory: its single machine frame carries a finite human and refine label.
    refined = labeled / "vidRefined"
    refined.mkdir()
    _write_label_table(path=refined / "machinelabels-iter0.h5", frame_names=["img0005.png"])
    _write_label_table(
        path=refined / f"CollectedData_{scorer}.h5", frame_names=["img0005.png"], finite_names={"img0005.png"}
    )
    _write_label_table(
        path=refined / "MachineLabelsRefine.h5", frame_names=["img0005.png"], finite_names={"img0005.png"}
    )

    # A directory with no current-iteration machine table is skipped entirely.
    (labeled / "vidNoMachine").mkdir()

    # A directory whose machine table cannot be read is recorded as unreadable, not fatal.
    unreadable = labeled / "vidUnreadable"
    unreadable.mkdir()
    (unreadable / "machinelabels-iter0.h5").write_text("not an hdf file")

    summary = summarize_refinement_status(config_path=config_path)

    assert summary.iteration == 0
    assert summary.pending_directory_count == 1
    assert summary.pending_frame_count == 1
    assert summary.pending_directories[0].directory == pending
    assert summary.pending_directories[0].unrefined_frame_count == 1
    assert summary.successful is False
    assert len(summary.unreadable) == 1
    assert summary.unreadable[0][0] == unreadable
    assert "1 directory(ies)" in summary.describe()


def test_summarize_selected_videos(tmp_path):
    """Verifies that selecting specific videos inspects only their directories and reports unmatched requests."""
    scorer = "S"
    videos_directory = tmp_path / "videos"
    videos_directory.mkdir()
    video_path = videos_directory / "vidPending.mp4"
    video_path.touch()

    config_path = tmp_path / "config.yaml"
    _write_config(
        path=config_path,
        data={
            "scorer": scorer,
            "iteration": 0,
            "project_path": str(tmp_path),
            "video_sets": {str(video_path): {"crop": "0, 10, 0, 10"}},
        },
    )

    labeled = tmp_path / "labeled-data"
    labeled.mkdir()
    pending = labeled / "vidPending"
    pending.mkdir()
    # No human labels at all -> the machine frame is entirely unrefined.
    _write_label_table(path=pending / "machinelabels-iter0.h5", frame_names=["img0001.png"])

    bogus = videos_directory / "nope.mp4"
    summary = summarize_refinement_status(config_path=config_path, videos=(str(video_path), str(bogus)))

    assert summary.pending_directory_count == 1
    assert summary.pending_frame_count == 1
    assert summary.unmatched_videos == (str(bogus),)
    assert summary.successful is True


def test_summarize_without_labeled_data(tmp_path):
    """Verifies that a project whose labeled-data tree does not exist reports no pending directories."""
    config_path = tmp_path / "config.yaml"
    _write_config(path=config_path, data={"scorer": "S", "iteration": 2, "project_path": str(tmp_path)})

    summary = summarize_refinement_status(config_path=config_path)

    assert summary.pending_directories == ()
    assert summary.iteration == 2
    assert summary.successful is True


# Helpers
def _write_config(path, data) -> None:
    """Writes a minimal DeepLabCut config.yaml holding the given mapping."""
    with path.open("w") as config_file:
        YAML().dump(data, config_file)


def _write_label_table(path, frame_names, *, finite_names=None, video="vid") -> None:
    """Writes a DeepLabCut-style label table keyed by ``df_with_missing`` with per-frame finite or all-NaN rows.

    Frames listed in ``finite_names`` receive finite coordinates; every other frame receives an all-NaN placeholder
    row, mirroring what the labeling GUI writes for opened-but-untouched frames.
    """
    if finite_names is None:
        finite_names = set(frame_names)
    index = pd.MultiIndex.from_tuples([("labeled-data", video, name) for name in frame_names])
    columns = pd.MultiIndex.from_tuples([("scorer", "bodypart", "x"), ("scorer", "bodypart", "y")])
    rows = [[1.0, 2.0] if name in finite_names else [np.nan, np.nan] for name in frame_names]
    pd.DataFrame(rows, index=index, columns=columns).to_hdf(path, key="df_with_missing")


def _build_purge_project(tmp_path, *, scorer="S", with_video_sets=False):
    """Builds a project tree with two per-video labeled-data directories plus entries the scans must skip."""
    videos_directory = tmp_path / "videos"
    videos_directory.mkdir()
    video_a = videos_directory / "vidA.mp4"
    video_b = videos_directory / "vidB.mp4"
    video_a.touch()
    video_b.touch()

    data = {"scorer": scorer, "iteration": 0, "project_path": str(tmp_path)}
    if with_video_sets:
        data["video_sets"] = {
            str(video_a): {"crop": "0, 100, 0, 100"},
            str(video_b): {"crop": "0, 100, 0, 100"},
        }
    config_path = tmp_path / "config.yaml"
    _write_config(path=config_path, data=data)

    labeled = tmp_path / "labeled-data"
    labeled.mkdir()

    directory_a = labeled / "vidA"
    directory_a.mkdir()
    (directory_a / "img0001.png").touch()
    (directory_a / "img0002.png").touch()
    _write_label_table(path=directory_a / f"CollectedData_{scorer}.h5", frame_names=["img0001.png", "img0002.png"])

    directory_b = labeled / "vidB"
    directory_b.mkdir()
    (directory_b / "img0001.png").touch()

    # Entries the whole-project scan must ignore: a rendered overlay dir, a hidden dir, a stray file, and a symlink.
    (labeled / "vidA_labeled").mkdir()
    (labeled / ".hidden").mkdir()
    (labeled / "notes.txt").touch()
    real_target = tmp_path / "real_target"
    real_target.mkdir()
    (labeled / "linkdir").symlink_to(real_target, target_is_directory=True)

    return config_path, video_a, video_b, labeled


class _FakeManager:
    """Stands in for a multiprocessing manager, handing out plain thread-safe queues and recording its shutdown."""

    def __init__(self):
        self.shutdown_called = False
        self.queues = []

    def Queue(self):  # noqa: N802 - mirrors multiprocessing.Manager's capitalized factory name.
        created = queue.Queue()
        self.queues.append(created)
        return created

    def shutdown(self):
        self.shutdown_called = True


class _FakePool:
    """Stands in for a process pool, running the worker synchronously in-process and preserving task order."""

    def __init__(self, *, processes, initializer, initargs):
        self.processes = processes
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def imap_unordered(self, *, func, iterable):
        for task in iterable:
            yield func(task)


class _FakeContext:
    """Stands in for a spawn context, vending the fake manager and pool."""

    def __init__(self):
        self.manager = _FakeManager()
        self.pools = []

    def Manager(self):  # noqa: N802 - mirrors multiprocessing context's capitalized factory name.
        return self.manager

    def Pool(self, *, processes, initializer, initargs):  # noqa: N802 - mirrors the capitalized factory name.
        pool = _FakePool(processes=processes, initializer=initializer, initargs=initargs)
        self.pools.append(pool)
        return pool


class _FakeMultiprocessing:
    """Stands in for the module's multiprocessing reference, exposing only the get_context the code calls."""

    def __init__(self, context):
        self._context = context

    def get_context(self, method):
        assert method == "spawn"
        return self._context


class _FakeBar:
    """Records its lifecycle in place of AggregateBar, so the pinned-extraction loop's bar handling is observable."""

    instances: ClassVar[list["_FakeBar"]] = []

    def __init__(self, *, progress_queue, total_video_count, frame_totals):
        self.progress_queue = progress_queue
        self.total_video_count = total_video_count
        self.frame_totals = frame_totals
        self.started = False
        self.stopped = False
        self.joined = False
        self.join_timeout = None
        self._alive = False
        _FakeBar.instances.append(self)

    def start(self):
        self.started = True
        self._alive = True

    def is_alive(self):
        return self._alive

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        self.joined = True
        self.join_timeout = timeout
        self._alive = False


def _patch_pinned_extraction(monkeypatch):
    """Swaps the module's multiprocessing and AggregateBar references for synchronous fakes and returns the context."""
    _FakeBar.instances.clear()
    fake_context = _FakeContext()
    monkeypatch.setattr(utilities, "multiprocessing", _FakeMultiprocessing(fake_context))
    monkeypatch.setattr(utilities, "AggregateBar", _FakeBar)
    return fake_context
