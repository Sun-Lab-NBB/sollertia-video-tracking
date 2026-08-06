"""Provides the shared assets the k-means and outlier frame-extraction pipelines both build on."""

from __future__ import annotations

import sys
import shutil
from typing import TYPE_CHECKING, Any, cast
from pathlib import Path
from dataclasses import dataclass
import multiprocessing

import pandas as pd
from ruamel.yaml import YAML

from .progress import AggregateBar
from .cpu_allocation import pin_worker_to_cores

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass(frozen=True, slots=True)
class PurgeSummary:
    """Summarizes a wholesale labeled-data purge, whether previewed as a dry run or actually executed.

    Notes:
        A purge is deliberately destructive: unlike the frame and outlier re-extraction options, which preserve every
        human-labeled and machine-labeled frame, it removes each targeted video's entire ``labeled-data`` directory,
        including its labels. The ``executed`` flag distinguishes a dry-run preview from a completed deletion, and
        ``labeled_directories`` names the directories that held human labels so callers can warn before deleting them.
    """

    config_path: Path
    """The path to the DeepLabCut project's config.yaml the purge targeted."""
    executed: bool
    """Indicates whether the directories were actually deleted (True) or only previewed as a dry run (False)."""
    removed_directories: tuple[Path, ...]
    """The ``labeled-data/<stem>`` directories that were removed, or that a dry run would remove."""
    labeled_directories: tuple[Path, ...]
    """The targeted directories that held a human ``CollectedData`` label file, whose labels the purge discards."""
    frame_count: int
    """The total number of extracted frames across the targeted directories, reported to convey the purge's scope."""
    unmatched_videos: tuple[str, ...] = ()
    """The requested videos that matched no registered project video and were skipped."""

    @property
    def removed_directory_count(self) -> int:
        """Returns the number of labeled-data directories removed, or that a dry run would remove."""
        return len(self.removed_directories)

    @property
    def labeled_directory_count(self) -> int:
        """Returns the number of targeted directories that held human labels."""
        return len(self.labeled_directories)


@dataclass(frozen=True, slots=True)
class RefinementDirectoryStatus:
    """Summarizes one video directory's machine-labeled frames still awaiting refinement for the current iteration."""

    directory: Path
    """The ``labeled-data/<stem>`` directory that still holds unrefined machine frames."""
    unrefined_frame_count: int
    """The number of the directory's current-iteration machine frames the human has not refined yet."""


@dataclass(frozen=True, slots=True)
class RefinementStatusSummary:
    """Summarizes which of a project's video directories still hold machine-labeled frames awaiting refinement.

    Notes:
        Only directories that still hold unrefined machine frames for the project's current refinement iteration are
        recorded in ``pending_directories``; directories that are already fully refined, purely human-labeled, or still
        bootstrapping are left out because they need no attention this round. This is what ``extract pending`` lists.
    """

    config_path: Path
    """The path to the DeepLabCut project's config.yaml the status was computed from."""
    iteration: int
    """The project's current refinement iteration, whose machine-label tables were inspected."""
    pending_directories: tuple[RefinementDirectoryStatus, ...]
    """The directories that still hold current-iteration machine frames the human has not refined."""
    unmatched_videos: tuple[str, ...] = ()
    """The requested videos that matched no registered project video and were skipped."""
    unreadable: tuple[tuple[Path, str], ...] = ()
    """The ``(directory, detail)`` pairs for directories whose label tables could not be read and were skipped."""

    @property
    def pending_directory_count(self) -> int:
        """Returns the number of directories that still need refinement."""
        return len(self.pending_directories)

    @property
    def pending_frame_count(self) -> int:
        """Returns the total number of unrefined machine frames across all pending directories."""
        return sum(directory.unrefined_frame_count for directory in self.pending_directories)

    @property
    def successful(self) -> bool:
        """Returns whether every scanned directory's label tables were read without error."""
        return not self.unreadable

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the pending refinement work for the CLI.

        Returns:
            A compact description of how many directories and frames still need refinement.
        """
        return (
            f"{self.pending_directory_count} directory(ies) need refining "
            f"({self.pending_frame_count} frame(s)) at iteration {self.iteration}"
        )


def normalize_project_config(config_path: Path, *, frames_per_video: int, error_context: str) -> Any:
    """Persists a normalized project_path and optional numframes2pick before any worker reads the configuration.

    DeepLabCut's read_config rewrites config.yaml whenever project_path differs from the config's own directory, which
    races against the concurrent workers' reads. The path is normalized here, single-threaded, and any per-video frame
    override is written alongside it, so every later read is a pure read.

    Args:
        config_path: The resolved path to the project's config.yaml.
        frames_per_video: The per-video frame count to persist as numframes2pick, or -1 to leave it untouched.
        error_context: The leading sentence of the ValueError raised on an invalid frame count, naming the operation.

    Returns:
        The loaded project configuration, with the normalized project_path and any frame-count override applied.

    Raises:
        ValueError: If ``frames_per_video`` is set below one and is not the -1 sentinel.
    """
    yaml = YAML()
    configuration = yaml.load(config_path.read_text())
    config_changed = False
    project_directory = str(config_path.parent)
    if configuration.get("project_path") != project_directory:
        configuration["project_path"] = project_directory
        config_changed = True
    if frames_per_video != -1:
        if frames_per_video < 1:
            message = f"{error_context} The frame count per video must be at least one, but got {frames_per_video}."
            raise ValueError(message)
        configuration["numframes2pick"] = int(frames_per_video)
        config_changed = True
    if config_changed:
        with config_path.open("w") as config_file:
            yaml.dump(data=configuration, stream=config_file)
    return configuration


def iter_pinned_extraction(
    *,
    videos: list[str],
    make_tasks: Callable[[Any | None], list[tuple[Any, ...]]],
    worker: Callable[[tuple[Any, ...]], tuple[str, int, str]],
    worker_count: int,
    core_sets: list[set[int]],
    frame_totals: dict[int, int],
    display_progress: bool,
) -> Iterator[tuple[str, int, str]]:
    """Runs the workers over a pinned process pool, yielding each ``(video_path, frames_written, status)`` result.

    Owns the spawn context, the manager-backed progress and core-set queues, and the aggregate progress bar, so both
    pipelines share one worker-pool lifecycle. Each worker claims its assigned core block for CPU-affinity pinning
    (disjoint in the default allocation; explicit worker/core counts may overlap within the usable band). The progress
    queue is handed to ``make_tasks`` only when progress is displayed (None otherwise), so the workers never
    stream to a queue nobody drains. The bar and manager are always torn down, even if the caller's consumption of
    the results raises.

    Args:
        videos: The ordered video paths, used to map each result back to its progress-bar slot.
        make_tasks: A callable that builds the worker task tuples, receiving the progress queue when progress is
            displayed or None when it is not.
        worker: The per-video worker callable applied to each task, returning ``(video_path, frames_written, status)``.
        worker_count: The resolved number of concurrent workers.
        core_sets: The per-worker core-id sets the workers are pinned across.
        frame_totals: The mapping of video index to that video's frame contribution, driving the aggregate bar.
        display_progress: Determines whether the aggregate progress bar is rendered and progress is streamed to it.

    Yields:
        Each worker's ``(video_path, frames_written, status)`` result, in completion order.
    """
    context = multiprocessing.get_context("spawn")
    manager = context.Manager()
    progress_queue = manager.Queue()
    core_set_queue = manager.Queue()
    for core_set in core_sets:
        core_set_queue.put(core_set)

    video_indices = {video: index for index, video in enumerate(videos)}
    tasks = make_tasks(progress_queue if display_progress else None)
    bar = AggregateBar(
        progress_queue=progress_queue,
        total_video_count=len(videos),
        frame_totals=frame_totals,
    )
    try:
        with context.Pool(processes=worker_count, initializer=pin_worker_to_cores, initargs=(core_set_queue,)) as pool:
            if display_progress:
                bar.start()
            for video, written, status in pool.imap_unordered(func=worker, iterable=tasks):
                yield video, written, status
                if display_progress:
                    progress_queue.put(("done", video_indices[video]))
    finally:
        # The bar thread is only started once the pool is created, so joining it when progress is off or when pool
        # construction raised before the start would hit an unstarted thread and mask the real error with a
        # RuntimeError. Stop and join it only while it is actually running.
        if display_progress and bar.is_alive():
            bar.stop()
            bar.join(timeout=3)
        manager.shutdown()


def prune_empty_labeled_data_directories(project_directory: Path, *, display_progress: bool = False) -> int:
    """Removes empty per-video directories from the project's labeled-data tree so unlabeled videos do not clutter it.

    A project may register many videos upfront while only a subset has frames extracted, and DeepLabCut leaves an
    empty labeled-data directory for every registered video that was not sampled. Deleting the directories that hold no
    extracted frames keeps the labeling GUI focused on the videos that actually have frames to label.

    Args:
        project_directory: The DeepLabCut project directory that holds the labeled-data tree.
        display_progress: Determines whether the number of pruned directories is reported to the standard error stream.

    Returns:
        The number of empty labeled-data directories removed.
    """
    labeled_data_directory = project_directory / "labeled-data"
    if not labeled_data_directory.exists():
        return 0
    removed_count = 0
    for directory in sorted(labeled_data_directory.iterdir()):
        # A symlink is never pruned: is_dir() and iterdir() follow it to its target, but rmdir() would act on the
        # link itself and raise NotADirectoryError, so only real, empty directories are removed.
        if directory.is_symlink() or not directory.is_dir():
            continue
        if not any(directory.iterdir()):
            directory.rmdir()
            removed_count += 1
    if display_progress and removed_count:
        sys.stderr.write(f"pruned {removed_count} empty labeled-data directory(ies)\n")
        sys.stderr.flush()
    return removed_count


def extracted_frame_paths(directory: Path) -> list[Path]:
    """Lists a labeled-data directory's extracted frames, excluding any predicted-label overlays.

    The outlier-refinement pipeline may leave ``imgNNNNlabeled.png`` prediction overlays beside the extracted
    ``imgNNNN.png`` frames when it saves labeled frames. Those overlays are not training frames, so counting them
    would inflate the k-means budgeted-sampling accounting and the per-video totals; they are filtered out here.

    Args:
        directory: The ``labeled-data/<video>`` directory whose extracted frames are listed.

    Returns:
        The sorted paths of the ``img*.png`` frames that are not ``*labeled.png`` prediction overlays, or an empty
        list when the directory does not exist.
    """
    if not directory.exists():
        return []
    return sorted(frame for frame in directory.glob("img*.png") if not frame.stem.endswith("labeled"))


def frame_names_from_index(frame_index: Any) -> set[str]:
    """Extracts the ``imgNNNN.png`` file names from a labeled-data table's row index.

    DeepLabCut indexes its label tables by a ``("labeled-data", video, image)`` row MultiIndex, though older tables may
    store a flat path string; the trailing image name is taken from either form.

    Args:
        frame_index: The pandas row index of a CollectedData or machine-label table.

    Returns:
        The set of frame image file names the index references.
    """
    names: set[str] = set()
    for entry in frame_index:
        names.add(str(entry[-1]) if isinstance(entry, tuple) else Path(str(entry)).name)
    return names


def finite_labeled_frame_names(collected_data_path: Path) -> set[str]:
    """Returns the frames a video's human labels annotate with at least one finite coordinate.

    The DeepLabCut labeling GUI reindexes ``CollectedData`` to every image in the directory on each save, writing
    all-NaN rows for frames that were opened but never annotated. Those placeholder rows are not real labels, so a frame
    counts as human-labeled here only when it carries a finite coordinate, letting callers treat the all-NaN
    placeholders as still-unlabeled.

    Args:
        collected_data_path: The ``CollectedData_<scorer>.h5`` file holding one video's human labels.

    Returns:
        The set of ``imgNNNN.png`` frame names with at least one finite coordinate, or an empty set when the file does
        not exist.

    Raises:
        Exception: Propagates any error reading the label table, so callers can fail safe rather than under-protect.
    """
    if not collected_data_path.is_file():
        return set()
    labels = cast("pd.DataFrame", pd.read_hdf(collected_data_path, key="df_with_missing"))
    annotated_labels = labels[labels.notna().any(axis=1)]
    return frame_names_from_index(annotated_labels.index)


def machine_label_frame_names(directory: Path) -> set[str]:
    """Returns every frame a video directory's machine-label or refinement tables reference.

    Outlier extraction records its machine pre-labels in ``machinelabels-iter<N>.h5`` (one table per refinement
    iteration) and the labeling GUI writes human refinements of them to ``MachineLabelsRefine.h5``. Both name frames
    that belong to the outlier-refinement workflow rather than the k-means bootstrap set, so callers protect them when
    clearing bootstrap frames.

    Args:
        directory: The ``labeled-data/<stem>`` directory whose machine-label tables are scanned.

    Returns:
        The set of ``imgNNNN.png`` frame names referenced by any machine-label or refinement table in the directory.

    Raises:
        Exception: Propagates any error reading a table, so callers can fail safe rather than under-protect.
    """
    names: set[str] = set()
    table_paths = [*sorted(directory.glob("machinelabels-iter*.h5")), *sorted(directory.glob("MachineLabelsRefine.h5"))]
    for table_path in table_paths:
        names |= frame_names_from_index(pd.read_hdf(table_path, key="df_with_missing").index)
    return names


def has_outlier_refinement_data(directory: Path) -> bool:
    """Reports whether a video directory already holds outlier-refinement data from a prior ``extract outliers`` pass.

    A video enters the outlier-refinement workflow once its directory holds a ``machinelabels-iter<N>.h5`` machine-label
    record or a ``MachineLabelsRefine.h5`` manual-refinement table. Frame extraction is the bootstrap step that runs
    before refinement, so it treats such a video as off-limits. This is a cheap file-presence probe rather than a table
    read, so it never fails on a malformed table and cannot be fooled by a table that references no frames.

    Args:
        directory: The ``labeled-data/<stem>`` directory to probe for outlier-refinement tables.

    Returns:
        True when the directory holds at least one outlier-refinement table, False otherwise.
    """
    return any(directory.glob("machinelabels-iter*.h5")) or (directory / "MachineLabelsRefine.h5").is_file()


def select_registered_videos(
    registered_videos: list[str], requested_videos: tuple[str | Path, ...]
) -> tuple[list[str], list[str]]:
    """Resolves the caller's requested video files to the project's registered videos, in registered order.

    Each requested path and each registered video path is resolved to its absolute form before matching, so a request
    selects its video regardless of how the path was spelled (relative, symlinked, or otherwise un-normalized). The
    frames pipeline uses the matches either as always-included pins over the full project or, under exclusive
    extraction, as the entire set to extract.

    Args:
        registered_videos: The project's registered video paths, in configuration order.
        requested_videos: The specific project video files the caller named.

    Returns:
        A tuple of the matched registered video paths, deduplicated and in registered order, and the deduplicated
        requested paths, unresolved, that matched no registered video.
    """
    resolved_registered = {video: Path(video).resolve() for video in registered_videos}
    matched_videos: set[str] = set()
    unmatched_requests: list[str] = []
    for request in dict.fromkeys(str(video) for video in requested_videos):
        resolved_request = Path(request).resolve()
        matches = [video for video, resolved in resolved_registered.items() if resolved == resolved_request]
        if matches:
            matched_videos.update(matches)
        else:
            unmatched_requests.append(request)
    ordered_matches = [video for video in registered_videos if video in matched_videos]
    return ordered_matches, unmatched_requests


def ensure_unique_video_stems(videos: list[str], *, error_context: str) -> None:
    """Raises when two selected videos share a file-name stem and would collide in the labeled-data tree.

    DeepLabCut stores every video's extracted frames under ``labeled-data/<stem>/``, keyed by the video's file-name
    stem alone. Two videos with the same stem but different directories therefore map to one shared directory: run in
    parallel they race on identically named frame files, and a budgeted-sampling pass counts one sibling's frames as
    the other's. The collision cannot be resolved safely here, so it is reported up front rather than silently
    corrupting the dataset.

    Args:
        videos: The selected video paths to check for stem collisions.
        error_context: The leading sentence of the raised ValueError, naming the operation.

    Raises:
        ValueError: If two videos with different paths share a file-name stem.
    """
    stems: dict[str, str] = {}
    for video in videos:
        stem = Path(video).stem
        existing = stems.get(stem)
        if existing is not None and existing != video:
            message = (
                f"{error_context} Two videos share the file-name stem '{stem}' and would collide in the "
                f"labeled-data tree: '{existing}' and '{video}'. Rename one so their stems differ."
            )
            raise ValueError(message)
        stems[stem] = video


def purge_labeled_data(
    config_path: Path, *, videos: tuple[str | Path, ...] = (), execute: bool = False
) -> PurgeSummary:
    """Removes targeted videos' entire ``labeled-data`` directories, previewing the deletion unless ``execute`` is set.

    This is the wholesale counterpart to the frame and outlier re-extraction options. Where those clear only a video's
    unlabeled bootstrap frames or a single iteration's outlier frames and always keep the human labels, a purge deletes
    each targeted directory outright, labels included. It exists for the rare start-completely-over case, such as
    changing the project crop, that the label-preserving options cannot serve. It defaults to a dry run so the caller
    sees the scope before committing.

    Args:
        config_path: The path to the DeepLabCut project's config.yaml, whose parent holds the ``labeled-data`` tree.
        videos: The specific project video files whose directories to purge, matched to the project's registered videos
            by resolved path. Leave empty to purge every video directory in the project's ``labeled-data`` tree.
        execute: Determines whether to actually delete the directories. When False, the directories are only reported
            for a dry-run preview and nothing is removed.

    Returns:
        A PurgeSummary naming the directories removed (or that a dry run would remove), which of them held human labels,
        the total frame count, and any requested videos that matched no registered project video.

    Raises:
        FileNotFoundError: If ``config_path`` does not point to an existing file.
    """
    config_path = config_path.resolve()
    if not config_path.is_file():
        message = f"Unable to purge labeled data. The config path '{config_path}' does not point to a file."
        raise FileNotFoundError(message)

    configuration = YAML().load(config_path.read_text())
    scorer = str(configuration.get("scorer", ""))
    labeled_data_directory = config_path.parent / "labeled-data"

    unmatched_videos: tuple[str, ...] = ()
    if not videos:
        # Iterate the directories actually present on disk so the purge covers every video's labeled data, including any
        # directories for videos no longer registered in config.yaml. The '_labeled' directories hold DeepLabCut's
        # rendered label overlays rather than a video's frames, and dot-prefixed entries are temporary, so both are left
        # alone.
        target_directories = (
            sorted(
                path
                for path in labeled_data_directory.iterdir()
                if path.is_dir()
                and not path.is_symlink()
                and not path.name.startswith(".")
                and not path.name.endswith("_labeled")
            )
            if labeled_data_directory.exists()
            else []
        )
    else:
        registered_videos = list(configuration.get("video_sets") or {})
        matched_videos, unmatched = select_registered_videos(
            registered_videos=registered_videos, requested_videos=tuple(videos)
        )
        unmatched_videos = tuple(unmatched)
        target_directories = [labeled_data_directory / Path(video).stem for video in matched_videos]

    existing_directories = [directory for directory in target_directories if directory.exists()]
    labeled_directories = tuple(
        directory for directory in existing_directories if (directory / f"CollectedData_{scorer}.h5").is_file()
    )
    frame_count = sum(len(extracted_frame_paths(directory)) for directory in existing_directories)

    if execute:
        for directory in existing_directories:
            shutil.rmtree(directory)

    return PurgeSummary(
        config_path=config_path,
        executed=execute,
        removed_directories=tuple(existing_directories),
        labeled_directories=labeled_directories,
        frame_count=frame_count,
        unmatched_videos=unmatched_videos,
    )


def summarize_refinement_status(config_path: Path, *, videos: tuple[str | Path, ...] = ()) -> RefinementStatusSummary:
    """Reports which video directories still hold machine-labeled outlier frames the human has not refined.

    Outlier extraction writes a trained model's likely-wrong frames into each video's ``machinelabels-iter<N>.h5`` table
    for the project's current iteration, and the human refines them in the labeling GUI, which saves the corrected
    coordinates into the directory's human ``CollectedData`` labels. A machine frame therefore counts as refined once it
    carries a finite human coordinate; an all-NaN placeholder row the GUI writes for an opened-but-untouched frame does
    not count. This scans the project's labeled-data tree and reports, per directory, how many of the current
    iteration's machine frames remain unrefined, so the human knows which directories to open next. It only reads the
    project, mutating nothing.

    Args:
        config_path: The path to the DeepLabCut project's config.yaml, whose parent holds the ``labeled-data`` tree.
        videos: The specific project video files to inspect, matched to the project's registered videos by resolved
            path. Leave empty to inspect every directory in the project's ``labeled-data`` tree.

    Returns:
        A RefinementStatusSummary listing the directories that still hold unrefined current-iteration machine frames,
        any requested videos that matched no registered project video, and any directories whose label tables could not
        be read.

    Raises:
        FileNotFoundError: If ``config_path`` does not point to an existing file.
    """
    config_path = config_path.resolve()
    if not config_path.is_file():
        message = f"Unable to summarize refinement status. The config path '{config_path}' does not point to a file."
        raise FileNotFoundError(message)

    configuration = YAML().load(config_path.read_text())
    scorer = str(configuration.get("scorer", ""))
    iteration = int(configuration.get("iteration", 0))
    labeled_data_directory = config_path.parent / "labeled-data"

    unmatched_videos: tuple[str, ...] = ()
    if not videos:
        # Scan the directories present on disk, mirroring purge_labeled_data: skip rendered '_labeled' overlays,
        # temporary dot-prefixed entries, and symlinks so only real per-video directories are inspected.
        target_directories = (
            sorted(
                path
                for path in labeled_data_directory.iterdir()
                if path.is_dir()
                and not path.is_symlink()
                and not path.name.startswith(".")
                and not path.name.endswith("_labeled")
            )
            if labeled_data_directory.exists()
            else []
        )
    else:
        registered_videos = list(configuration.get("video_sets") or {})
        matched_videos, unmatched = select_registered_videos(
            registered_videos=registered_videos, requested_videos=tuple(videos)
        )
        unmatched_videos = tuple(unmatched)
        target_directories = [labeled_data_directory / Path(video).stem for video in matched_videos]

    machine_labels_name = f"machinelabels-iter{iteration}.h5"
    pending_directories: list[RefinementDirectoryStatus] = []
    unreadable: list[tuple[Path, str]] = []
    for directory in target_directories:
        machine_labels_path = directory / machine_labels_name
        if not machine_labels_path.is_file():
            # A directory without a current-iteration machine-label table has nothing to refine this round; skip it.
            continue
        try:
            machine_frame_names = frame_names_from_index(pd.read_hdf(machine_labels_path, key="df_with_missing").index)
            # A machine frame is refined only once it carries a finite human coordinate: the labeling GUI writes
            # all-NaN placeholder rows for opened-but-untouched frames, so index presence alone is not enough. The
            # MachineLabelsRefine table covers the legacy refine flow and is absent under the napari labeler.
            refined_frame_names = finite_labeled_frame_names(directory / f"CollectedData_{scorer}.h5")
            refined_frame_names |= finite_labeled_frame_names(directory / "MachineLabelsRefine.h5")
        except Exception as error:
            unreadable.append((directory, str(error)))
            continue
        unrefined_frame_names = machine_frame_names - refined_frame_names
        if unrefined_frame_names:
            pending_directories.append(
                RefinementDirectoryStatus(directory=directory, unrefined_frame_count=len(unrefined_frame_names))
            )

    return RefinementStatusSummary(
        config_path=config_path,
        iteration=iteration,
        pending_directories=tuple(pending_directories),
        unmatched_videos=unmatched_videos,
        unreadable=tuple(unreadable),
    )
