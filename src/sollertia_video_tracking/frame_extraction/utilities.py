"""Provides the shared assets the k-means and outlier frame-extraction pipelines both build on."""

import sys
from typing import Any
from pathlib import Path
from collections.abc import Callable, Iterator
import multiprocessing

from ruamel.yaml import YAML

from .progress import AggregateBar
from .cpu_allocation import pin_worker_to_cores


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
    minimum_progress_interval: float,
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
        minimum_progress_interval: The minimum interval, in seconds, between progress lines when output is not a TTY.
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
        minimum_progress_interval=minimum_progress_interval,
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
        if display_progress:
            bar.stop()
            bar.join(timeout=3)
        manager.shutdown()


def prune_empty_labeled_data_directories(project_directory: Path, *, display_progress: bool = False) -> int:
    """Removes empty per-video folders from the project's labeled-data tree so unlabeled videos do not clutter it.

    A project may register many videos upfront while only a subset has frames extracted, and DeepLabCut leaves an
    empty labeled-data folder for every registered video that was not sampled. Deleting the folders that hold no
    extracted frames keeps the labeling GUI focused on the videos that actually have frames to label.

    Args:
        project_directory: The DeepLabCut project directory that holds the labeled-data tree.
        display_progress: Determines whether the number of pruned folders is reported to the standard error stream.

    Returns:
        The number of empty labeled-data folders removed.
    """
    labeled_data_directory = project_directory / "labeled-data"
    if not labeled_data_directory.exists():
        return 0
    removed_count = 0
    for directory in sorted(labeled_data_directory.iterdir()):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
            removed_count += 1
    if display_progress and removed_count:
        sys.stderr.write(f"pruned {removed_count} empty labeled-data folder(s)\n")
        sys.stderr.flush()
    return removed_count
