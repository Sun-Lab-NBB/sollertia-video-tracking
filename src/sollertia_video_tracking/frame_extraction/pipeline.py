"""Provides the parallel k-means frame-extraction pipeline that prepares a DeepLabCut project's training frames."""

import os
import sys
import math
from typing import Any
from pathlib import Path
import traceback
import contextlib
from dataclasses import dataclass
import multiprocessing

import cv2
import psutil
import deeplabcut
from ruamel.yaml import YAML
import deeplabcut.utils.frameselectiontools as frame_selection_tools

from .progress import AggregateBar, make_progress_reporter
from .cpu_allocation import DEFAULT_RESERVE_CORES, plan_core_allocation
from .video_sampling import VideoSamplingPlan, plan_video_sampling

_REEXTRACTION_ARTIFACT_PATTERNS: tuple[str, ...] = (
    "img*.png",
    "CollectedData_*.h5",
    "CollectedData_*.csv",
    "MachineLabelsRefine.*",
)
"""The labeled-data file patterns removed before a video is re-extracted, covering both frames and their labels."""


@dataclass(frozen=True, slots=True)
class FrameExtractionSummary:
    """Summarizes the outcome of a parallel k-means frame-extraction run.

    Notes:
        The pipeline never aborts on a single bad video, so failures are collected here as ``(video_path,
        traceback)`` pairs rather than raised. Callers inspect ``failed`` (or ``errors``) to decide the process exit
        status.
    """

    extracted: int
    """The number of videos for which frames were freshly extracted."""
    skipped: int
    """The number of videos skipped because their labeled-data directory already contained frames."""
    total: int
    """The total number of videos considered in the run."""
    workers: int
    """The number of worker processes that ran concurrently."""
    cores_used: int
    """The number of distinct CPU cores the workers were pinned across."""
    core_count: int
    """The total number of CPU cores available on the machine."""
    frames_to_cluster: int
    """The total number of frames that were read and clustered across all videos."""
    existing_frames: int = 0
    """The number of frames already extracted across the selection before this run, reported in sampling mode."""
    target_frames: int = -1
    """The requested total-frame budget when sampling videos, or -1 when budgeted sampling was disabled."""
    errors: tuple[tuple[str, str], ...] = ()
    """The ``(video_path, traceback)`` pairs for every video that failed to extract."""

    @property
    def failed(self) -> int:
        """Returns the number of videos that failed to extract."""
        return len(self.errors)

    @property
    def successful(self) -> bool:
        """Returns whether the run completed with no failed videos."""
        return not self.errors


def extract_frames_kmeans(
    config_path: Path,
    *,
    step: int = 500,
    workers: int = -1,
    cores_per_worker: int = -1,
    reserve_cores: int = DEFAULT_RESERVE_CORES,
    num_frames: int = -1,
    total_frames: int = -1,
    seed: int | None = None,
    resize_width: int = 30,
    color: bool = False,
    overwrite: bool = False,
    reset: bool = False,
    path_filters: tuple[str, ...] = (),
    heartbeat: float = 30.0,
    display_progress: bool = True,
) -> FrameExtractionSummary:
    """Runs DeepLabCut k-means frame extraction across a project's videos in parallel and reports the outcome.

    Reads the run parameters from the project's config.yaml, plans the CPU-core allocation, and clusters every
    selected video in its own pinned worker process. Re-runs are resumable: videos whose ``labeled-data/<stem>/``
    directory already contains frames are skipped unless ``overwrite`` is set. A single bad video is recorded in the
    returned summary rather than aborting the run.

    When ``total_frames`` is set, the run samples a random subset of not-yet-extracted videos sized to reach that
    project-wide frame budget, so the full project can be added once and extraction grows coverage across repeated
    passes without manual selection. When the existing frames already meet the budget, the run extracts nothing and
    warns; raising ``total_frames`` or using ``reset`` grows the set further.

    Notes:
        The pipeline uses the spawn multiprocessing start method on every platform for reproducible behavior, so a
        programmatic caller must guard the call with ``if __name__ == "__main__":``. The installed console-script
        entry point is already guarded. CPU-affinity pinning is applied on Linux and Windows; macOS exposes no
        affinity API, so its workers run unpinned but still in parallel.

        Because k-means selects different frame indices on each run, re-extraction writes different image filenames
        and orphans the existing labels, so ``overwrite`` and ``reset`` both delete a video's ``CollectedData`` labels
        alongside its frames. The ``seed`` controls only the random choice of videos, not the frames within a video.

    Args:
        config_path: The path to the DeepLabCut project's config.yaml.
        step: The clustering stride passed to DeepLabCut as ``cluster_step``; every ``step``-th frame is sampled.
        workers: The number of videos to decode in parallel. Set to -1 to fill the usable cores automatically.
        cores_per_worker: The number of CPU cores pinned to each worker. Set to -1 to spread the usable cores evenly.
        reserve_cores: The number of CPU cores to leave free for other tasks.
        num_frames: The number of frames to keep per video, overriding ``numframes2pick`` in config.yaml. Set to -1
            to use the value already stored in the configuration file.
        total_frames: The total number of frames the project should hold, reached by randomly sampling not-yet-
            extracted videos. Set to -1 to extract every selected video instead of sampling toward a budget.
        seed: The seed for the random video sampling. Set to None to draw a different subset each run, or to an
            integer to make the selection reproducible.
        resize_width: The downsample width applied before clustering, passed to DeepLabCut as ``cluster_resizewidth``.
        color: Whether to cluster on color channels instead of grayscale.
        overwrite: Whether to re-extract videos whose labeled-data directory already contains frames. Re-extraction
            deletes the existing ``img*.png`` frames in each affected directory and the matching ``CollectedData``
            labels, which the new frames would otherwise orphan. Mutually exclusive with ``reset``.
        reset: Whether to discard all extracted frames and their labels across the selection and re-extract from
            scratch. This permanently deletes the extracted frames and any labels in every selected video folder.
            Mutually exclusive with ``overwrite``.
        path_filters: The substrings used to restrict the run to videos whose path contains any of them. An empty
            tuple selects every video in the project.
        heartbeat: The minimum interval, in seconds, between progress lines when the output is not a TTY.
        display_progress: Whether to render the run header and the aggregate progress bar to the standard error
            stream.

    Returns:
        A FrameExtractionSummary describing how many videos were extracted, skipped, and failed, alongside the
        resolved core-allocation plan and, in sampling mode, the existing and target frame counts.

    Raises:
        ValueError: If ``num_frames`` or ``total_frames`` is set below one (other than the -1 sentinel), if
            ``overwrite`` and ``reset`` are both set, if budgeted sampling is requested without a valid
            ``numframes2pick``, or if no videos in config.yaml match ``path_filters``.
    """
    config_path = config_path.resolve()
    if overwrite and reset:
        message = "Unable to extract frames. The overwrite and reset options are mutually exclusive."
        raise ValueError(message)
    if total_frames != -1 and total_frames < 1:
        message = f"Unable to extract frames. The total frame budget must be at least one, but got {total_frames}."
        raise ValueError(message)
    yaml = YAML()
    configuration = yaml.load(config_path.read_text())
    start = float(configuration.get("start", 0))
    stop = float(configuration.get("stop", 1))

    # DeepLabCut's read_config rewrites config.yaml whenever project_path differs from the config's own directory.
    # With many workers reading concurrently, that write races against sibling reads and intermittently returns an
    # empty config, so the project_path is normalized here, single-threaded, and persisted before any worker starts.
    config_changed = False
    project_directory = str(config_path.parent)
    if configuration.get("project_path") != project_directory:
        configuration["project_path"] = project_directory
        config_changed = True

    # Like the DeepLabCut GUI, a num_frames override is written straight into config.yaml. The -1 sentinel leaves the
    # value already stored in the configuration file untouched.
    if num_frames != -1:
        if num_frames < 1:
            message = f"Unable to extract frames. The frame count per video must be at least one, but got {num_frames}."
            raise ValueError(message)
        configuration["numframes2pick"] = int(num_frames)
        config_changed = True

    if config_changed:
        with config_path.open("w") as config_file:
            yaml.dump(configuration, config_file)
    frames_per_video = configuration.get("numframes2pick", "?")

    videos = list(configuration["video_sets"])
    if path_filters:
        videos = [video for video in videos if any(token in video for token in path_filters)]
    if not videos:
        message = "Unable to extract frames. No videos in the project's config.yaml matched the requested selection."
        raise ValueError(message)

    project_directory_path = config_path.parent
    if reset:
        cleared = sum(
            1 for video in videos if any((project_directory_path / "labeled-data" / Path(video).stem).glob("img*.png"))
        )
        sys.stderr.write(
            f"WARNING: --reset is deleting all extracted frames and labels from {cleared} video folder(s).\n"
        )
        sys.stderr.flush()
        for video in videos:
            _clear_extracted_data(output_directory=project_directory_path / "labeled-data" / Path(video).stem)

    existing_frames = 0
    target_frames = -1
    if total_frames != -1:
        frames_per_video_count = configuration.get("numframes2pick")
        if not isinstance(frames_per_video_count, int) or frames_per_video_count < 1:
            message = (
                "Unable to sample videos for a frame budget. The project's numframes2pick must be a positive "
                f"integer, but got {frames_per_video_count!r}. Pass num_frames to set it."
            )
            raise ValueError(message)
        plan = plan_video_sampling(
            videos=videos,
            extracted_frame_counts=_count_extracted_frames(videos=videos, project_directory=project_directory_path),
            frames_per_video=frames_per_video_count,
            total_frames=total_frames,
            seed=seed,
        )
        _report_sampling_plan(plan=plan)
        if not plan.selected:
            return FrameExtractionSummary(
                extracted=0,
                skipped=0,
                total=0,
                workers=0,
                cores_used=0,
                core_count=os.cpu_count() or 1,
                frames_to_cluster=0,
                existing_frames=plan.existing_frames,
                target_frames=plan.target_frames,
            )
        videos = list(plan.selected)
        existing_frames = plan.existing_frames
        target_frames = plan.target_frames

    core_count = os.cpu_count() or 1
    workers, core_sets = plan_core_allocation(
        video_count=len(videos),
        core_count=core_count,
        workers=workers,
        cores_per_worker=cores_per_worker,
        reserve_cores=reserve_cores,
    )
    cores_used = len({core for core_set in core_sets for core in core_set})

    totals = _count_clustering_frames(videos=videos, start=start, stop=stop, step=step)
    frames_to_cluster = sum(totals.values())

    # The spawn start method is used on every platform for reproducible, fork-safety-independent behavior. Each worker
    # claims one core block from the slot queue for CPU-affinity pinning; a manager queue is used (rather than a
    # fork-inherited shared counter) so the core blocks survive being pickled to the spawned worker processes.
    context = multiprocessing.get_context("spawn")
    manager = context.Manager()
    progress_queue = manager.Queue()
    slot_queue = manager.Queue()
    for core_set in core_sets:
        slot_queue.put(core_set)
    config_path_string = str(config_path)
    tasks = [
        (
            video,
            config_path_string,
            step,
            resize_width,
            color,
            overwrite,
            video_index,
            totals[video_index],
            progress_queue,
        )
        for video_index, video in enumerate(videos)
    ]
    video_indices = {video: index for index, video in enumerate(videos)}

    if display_progress:
        _report_plan(
            videos=len(videos),
            frames_per_video=frames_per_video,
            step=step,
            resize_width=resize_width,
            color=color,
            workers=workers,
            cores_used=cores_used,
            core_count=core_count,
            frames_to_cluster=frames_to_cluster,
            config_path=config_path,
        )

    bar = AggregateBar(progress_queue=progress_queue, total_videos=len(tasks), totals=totals, heartbeat=heartbeat)

    extracted_count = skipped_count = 0
    errors: list[tuple[str, str]] = []
    # Brings up the worker pool, then starts the renderer thread.
    with context.Pool(processes=workers, initializer=_pin_worker_to_cores, initargs=(slot_queue,)) as pool:
        if display_progress:
            bar.start()
        for video, _written, status in pool.imap_unordered(_extract_one_video, tasks):
            if status == "ok":
                extracted_count += 1
            elif status == "skipped":
                skipped_count += 1
            else:
                errors.append((video, status))
            progress_queue.put(("done", video_indices[video]))

    if display_progress:
        bar.stop()
        bar.join(timeout=3)

    return FrameExtractionSummary(
        extracted=extracted_count,
        skipped=skipped_count,
        total=len(tasks),
        workers=workers,
        cores_used=cores_used,
        core_count=core_count,
        frames_to_cluster=frames_to_cluster,
        existing_frames=existing_frames,
        target_frames=target_frames,
        errors=tuple(errors),
    )


def _report_plan(
    videos: int,
    frames_per_video: Any,
    step: int,
    resize_width: int,
    *,
    color: bool,
    workers: int,
    cores_used: int,
    core_count: int,
    frames_to_cluster: int,
    config_path: Path,
) -> None:
    """Writes the two-line run header describing the resolved extraction plan to the standard error stream.

    Args:
        videos: The number of videos selected for the run.
        frames_per_video: The resolved ``numframes2pick`` value from config.yaml.
        step: The clustering stride.
        resize_width: The downsample width applied before clustering.
        color: Whether clustering runs on color channels.
        workers: The resolved number of concurrent workers.
        cores_used: The number of distinct cores the workers are pinned across.
        core_count: The total number of cores on the machine.
        frames_to_cluster: The total number of frames to read and cluster.
        config_path: The resolved path to the project's config.yaml.
    """
    free_cores = core_count - cores_used
    sys.stderr.write(
        f"k-means extraction | {videos} videos | numframes2pick={frames_per_video} | "
        f"cluster_step={step} | resize_width={resize_width} | color={color}\n"
    )
    sys.stderr.write(
        f"workers={workers} | {cores_used}/{core_count} cores used ({free_cores} free) | "
        f"{frames_to_cluster:,} frames to cluster | config={config_path}\n"
    )
    sys.stderr.flush()


def _report_sampling_plan(plan: VideoSamplingPlan) -> None:
    """Writes the budgeted-sampling outcome, including any budget warnings, to the standard error stream.

    Args:
        plan: The sampling plan whose selection and budget flags are reported.
    """
    if plan.no_growth:
        sys.stderr.write(
            f"WARNING: the project already holds {plan.existing_frames:,} frames, which meets the requested total "
            f"of {plan.target_frames:,}. Nothing will be extracted. Raise the total frame budget to grow the set, "
            f"or use reset to start over.\n"
        )
    elif not plan.selected:
        sys.stderr.write(
            "WARNING: every selected video already has extracted frames, so none remain to sample. Use overwrite "
            "or reset to re-extract from existing videos.\n"
        )
    elif plan.target_unreachable:
        sys.stderr.write(
            f"WARNING: too few un-extracted videos remain to reach {plan.target_frames:,} frames. Sampling all "
            f"{len(plan.selected)} remaining video(s) for a projected {plan.projected_frames:,} frames.\n"
        )
    else:
        sys.stderr.write(
            f"sampling {len(plan.selected)} video(s) | {plan.existing_frames:,} existing -> "
            f"{plan.projected_frames:,} projected frames (target {plan.target_frames:,})\n"
        )
    sys.stderr.flush()


def _count_extracted_frames(videos: list[str], project_directory: Path) -> dict[str, int]:
    """Counts the frames already extracted for each video, keyed by the video's path.

    Args:
        videos: The candidate video paths to inspect.
        project_directory: The DeepLabCut project directory that holds the ``labeled-data`` tree.

    Returns:
        A mapping of video path to the number of ``img*.png`` frames currently in its labeled-data directory.
    """
    counts: dict[str, int] = {}
    for video in videos:
        directory = project_directory / "labeled-data" / Path(video).stem
        counts[video] = len(list(directory.glob("img*.png"))) if directory.exists() else 0
    return counts


def _clear_extracted_data(output_directory: Path) -> None:
    """Deletes a video's extracted frames and the labels they would orphan from its labeled-data directory.

    Re-extraction selects different frame indices and therefore writes different image filenames, so the existing
    ``CollectedData`` label rows would reference deleted images. The label and refinement files are removed alongside
    the frames to avoid leaving a dataset that mixes valid, dangling, and unlabeled entries.

    Args:
        output_directory: The ``labeled-data/<video>`` directory whose extracted frames and labels are removed.
    """
    for pattern in _REEXTRACTION_ARTIFACT_PATTERNS:
        for target in output_directory.glob(pattern):
            target.unlink()


def _count_clustering_frames(videos: list[str], start: float, stop: float, step: int) -> dict[int, int]:
    """Counts the frames DeepLabCut will read and cluster for each video, keyed by the video's index.

    The per-video total mirrors DeepLabCut's own sampling: the frames between the configured start and stop bounds
    are visited with the given stride.

    Args:
        videos: The ordered list of video paths to inspect.
        start: The fractional start position within each video, in the range [0, 1].
        stop: The fractional stop position within each video, in the range [0, 1].
        step: The sampling stride; every ``step``-th frame is clustered.

    Returns:
        A mapping of video index to the number of frames that video contributes to the aggregate total.
    """
    cv2.setNumThreads(1)
    totals: dict[int, int] = {}
    for video_index, video in enumerate(videos):
        frame_count = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))
        start_index, end_index = math.floor(frame_count * start), math.ceil(frame_count * stop)
        totals[video_index] = max(1, len(range(start_index, end_index, step)))
    return totals


def _extract_one_video(task: tuple[Any, ...]) -> tuple[str, int, str]:
    """Runs DeepLabCut k-means extraction for a single video and reports the outcome.

    DeepLabCut's console output is silenced and exceptions are captured, so one bad video cannot kill the worker
    pool. The native math-library thread pools are pinned to a single thread (configured in the package __init__
    before the heavy backends import), so each worker stays within its assigned CPU budget.

    Args:
        task: The packed work item carrying the video path, the config path, the clustering parameters, the resume
            flag, the video index, the per-video frame total, and the shared progress queue.

    Returns:
        A tuple of the video path, the number of frames written, and a status string (``"ok"``, ``"skipped"``,
        ``"empty"``, or an ``"error:"`` traceback).
    """
    video_path, config_path, step, resize_width, color, overwrite, video_index, total, progress_queue = task
    try:
        cv2.setNumThreads(1)

        stem = Path(video_path).stem
        output_directory = Path(config_path).parent / "labeled-data" / stem

        existing_frames = sorted(output_directory.glob("img*.png")) if output_directory.exists() else []
        if existing_frames and not overwrite:
            return video_path, len(existing_frames), "skipped"
        # On overwrite, drop the stale frames and the labels they would orphan before re-extracting.
        _clear_extracted_data(output_directory=output_directory)

        # Redirects DeepLabCut's internal tqdm progress to the parent's aggregate bar via the shared queue.
        frame_selection_tools.tqdm = make_progress_reporter(
            progress_queue=progress_queue, video_index=video_index, total=total
        )

        with (
            Path(os.devnull).open("w") as null_stream,
            contextlib.redirect_stdout(null_stream),
            contextlib.redirect_stderr(null_stream),
        ):
            deeplabcut.extract_frames(
                config_path,
                mode="automatic",
                algo="kmeans",
                crop=True,  # Applies the per-video crop stored in config.yaml.
                userfeedback=False,  # Runs non-interactively.
                cluster_step=step,
                cluster_resizewidth=resize_width,
                cluster_color=color,
                videos_list=[video_path],  # Restricts DeepLabCut to this one video.
            )
    except Exception:  # noqa: BLE001 -- one bad video must not kill the pool; the traceback is returned as status.
        return video_path, 0, "error:\n" + traceback.format_exc()
    else:
        written = len(list(output_directory.glob("img*.png")))
        return video_path, written, "ok" if written else "empty"


def _pin_worker_to_cores(slot_queue: Any) -> None:
    """Pins the calling pool worker to the next free core block, called once per worker at start.

    Each worker claims one core set from the shared queue and binds its CPU affinity to it, so the worker and every
    thread its decoder spawns stay within a disjoint block of cores. CPU affinity is supported on Linux and Windows;
    macOS exposes no affinity API, so its workers run unpinned. The binding is best-effort, so a missing slot or an
    unsupported platform degrades to an unpinned worker rather than aborting the extraction.

    Args:
        slot_queue: The shared queue holding one core-id set per worker, produced by the extraction pipeline.
    """
    with contextlib.suppress(Exception):
        core_set = slot_queue.get_nowait()
        psutil.Process().cpu_affinity(list(core_set))
