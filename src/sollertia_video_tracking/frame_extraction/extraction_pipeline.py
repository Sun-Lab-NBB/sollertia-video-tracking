"""Provides the parallel k-means frame-extraction pipeline that prepares a DeepLabCut project's training frames."""

import os
import sys
import math
import shutil
from typing import Any
from pathlib import Path
import traceback
import contextlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import cv2
import deeplabcut
import deeplabcut.utils.frameselectiontools as frame_selection_tools

from .progress import make_progress_reporter
from .utilities import (
    extracted_frame_paths,
    iter_pinned_extraction,
    resolve_video_overrides,
    normalize_project_config,
    ensure_unique_video_stems,
    prune_empty_labeled_data_directories,
)
from .frame_reading import make_fast_kmeans_selector
from .cpu_allocation import DEFAULT_RESERVED_CORE_COUNT, plan_core_allocation
from .video_grouping import group_videos
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
        The pipeline never aborts on a single bad video, so failures are collected here as ``(video_path, detail)``
        pairs rather than raised, where the detail is an ``error:``-prefixed traceback or the marker ``"empty"`` for a
        video that produced no frames. Callers inspect ``failed_video_count`` (or ``errors``) to decide the process
        exit status.
    """

    extracted_video_count: int
    """The number of videos for which frames were freshly extracted."""
    skipped_video_count: int
    """The number of videos skipped because their labeled-data directory already contained frames."""
    total_video_count: int
    """The total number of videos considered in the run."""
    worker_count: int
    """The number of worker processes that ran concurrently."""
    used_core_count: int
    """The number of distinct CPU cores the workers were pinned across."""
    total_core_count: int
    """The total number of CPU cores available on the machine."""
    clustering_frame_count: int
    """The total number of frames scheduled to be read and clustered across all selected videos, estimated from the
    video headers before extraction; this includes videos that are later skipped on a resumable re-run."""
    existing_frame_count: int = 0
    """The number of frames already extracted across the selection before this run, reported in sampling mode."""
    target_frame_count: int = -1
    """The requested total-frame budget when sampling videos, or -1 when budgeted sampling was disabled."""
    errors: tuple[tuple[str, str], ...] = ()
    """The ``(video_path, detail)`` pairs for every video that failed to extract or produced no frames, where the
    detail is an ``error:``-prefixed traceback or the marker ``"empty"``.
    """

    @property
    def failed_video_count(self) -> int:
        """Returns the number of videos that failed to extract or produced no frames."""
        return len(self.errors)

    @property
    def successful(self) -> bool:
        """Returns whether the run completed with no failed videos."""
        return not self.errors


def extract_frames_kmeans(
    config_path: Path,
    *,
    clustering_stride: int = 1,
    worker_count: int = -1,
    cores_per_worker: int = -1,
    reserved_core_count: int = DEFAULT_RESERVED_CORE_COUNT,
    frames_per_video: int = -1,
    total_frame_budget: int = -1,
    random_seed: int | None = None,
    balance_groups: bool = False,
    group_by_pattern: str | None = None,
    always_include_videos: tuple[str, ...] = (),
    clustering_resize_width: int = 30,
    cluster_in_color: bool = False,
    overwrite: bool = False,
    reset: bool = False,
    path_filters: tuple[str, ...] = (),
    minimum_progress_interval: float = 30.0,
    display_progress: bool = True,
) -> FrameExtractionSummary:
    """Runs DeepLabCut k-means frame extraction across a project's videos in parallel and reports the outcome.

    Reads the run parameters from the project's config.yaml, plans the CPU-core allocation, and clusters every
    selected video in its own pinned worker process. Re-runs are resumable: videos whose ``labeled-data/<stem>/``
    directory already contains frames are skipped unless ``overwrite`` is set. A single bad video is recorded in the
    returned summary rather than aborting the run.

    When ``total_frame_budget`` is set, the run samples a random subset of not-yet-extracted videos sized to reach that
    project-wide frame budget. This means that the full project's video set can be added once and extraction grows
    coverage across repeated passes without manual selection. When the existing frames already meet the budget, the
    run extracts nothing and warns; raising ``total_frame_budget`` or using ``reset`` grows the set further.

    Notes:
        The pipeline uses the spawn multiprocessing start method on every platform for reproducible behavior, so a
        programmatic caller must guard the call with ``if __name__ == "__main__":``. The installed console-script
        entry point is already guarded. CPU-affinity pinning is applied on Linux and Windows; macOS exposes no
        affinity API, so its workers run unpinned but still in parallel.

        Because k-means selects different frame indices on each run, re-extraction writes different image filenames
        and orphans the existing labels, so ``overwrite`` and ``reset`` both delete a video's ``CollectedData`` labels
        alongside its frames. The ``random_seed`` controls only the random choice of videos, not the frames within a
        video.

        Empty ``labeled-data`` folders left by videos that were registered but never extracted are removed after every
        run, so the labeling GUI shows only the videos that have frames.

    Args:
        config_path: The path to the DeepLabCut project's config.yaml.
        clustering_stride: The clustering stride passed to DeepLabCut as ``cluster_step``; every Nth frame is sampled,
            where N is this stride.
        worker_count: The number of videos to decode in parallel. Set to -1 to fill the usable cores automatically.
        cores_per_worker: The number of CPU cores pinned to each worker. Set to -1 to spread the usable cores evenly.
        reserved_core_count: The number of CPU cores to leave free for other tasks.
        frames_per_video: The number of frames to keep per video, overriding ``numframes2pick`` in config.yaml. Set to
            -1 to use the value already stored in the configuration file.
        total_frame_budget: The total number of frames the project should hold, reached by randomly sampling not-yet-
            extracted videos. Set to -1 to extract every selected video instead of sampling toward a budget.
        random_seed: The seed for the random video sampling. Set to None to draw a different subset each run, or to an
            integer to make the selection reproducible.
        balance_groups: Determines whether the budgeted sampling is balanced across groups rather than drawn
            uniformly, so every group is represented and coverage evens out across repeated passes. The group of each
            video is inferred from its file name, with videos that share their non-date name components grouped
            together. Only affects the run when ``total_frame_budget`` is set.
        group_by_pattern: A regular expression whose first capturing group names the group for each video's file-name
            stem, overriding the built-in inference for naming schemes it does not cover. Setting it implies
            ``balance_groups``.
        always_include_videos: The path substrings naming videos to always include in the budgeted sample, selected
            before the balanced or uniform draw fills the remaining budget. Only affects the run when
            ``total_frame_budget`` is set.
        clustering_resize_width: The downsample width applied before clustering, passed to DeepLabCut as
            ``cluster_resizewidth``.
        cluster_in_color: Determines whether to cluster on color channels instead of grayscale.
        overwrite: Determines whether to re-extract videos whose labeled-data directory already contains frames.
            Re-extraction deletes the existing ``img*.png`` frames in each affected directory and the matching
            ``CollectedData`` labels, which the new frames would otherwise orphan. Mutually exclusive with ``reset``.
        reset: Determines whether to discard every selected video's extracted frames and re-extract from scratch. This
            permanently deletes each selected video's entire ``labeled-data`` folder, including its extracted frames
            and labels, leaving no empty folder behind. Mutually exclusive with ``overwrite``.
        path_filters: The substrings used to restrict the run to videos whose path contains any of them. An empty
            tuple selects every video in the project.
        minimum_progress_interval: The minimum interval, in seconds, between progress lines when the output is not a
            TTY.
        display_progress: Determines whether to render the run header and the aggregate progress bar to the standard
            error stream.

    Returns:
        A FrameExtractionSummary describing how many videos were extracted, skipped, and failed, alongside the
        resolved core-allocation plan and, in sampling mode, the existing and target frame counts.

    Raises:
        FileNotFoundError: If ``config_path`` does not point to an existing file.
        ValueError: If ``overwrite`` and ``reset`` are both set, if ``clustering_stride`` is below one, or if
            ``frames_per_video`` or ``total_frame_budget`` is set below one (other than the -1 sentinel). Also raised
            when budgeted sampling is requested without a valid ``numframes2pick``, when config.yaml defines no
            ``video_sets``, when no videos in config.yaml match ``path_filters``, or when two selected videos share a
            file-name stem and would collide in the labeled-data tree.
    """
    config_path = config_path.resolve()
    if overwrite and reset:
        message = "Unable to extract frames. The overwrite and reset options are mutually exclusive."
        raise ValueError(message)
    if total_frame_budget != -1 and total_frame_budget < 1:
        message = (
            f"Unable to extract frames. The total frame budget must be at least one, but got {total_frame_budget}."
        )
        raise ValueError(message)
    if clustering_stride < 1:
        message = f"Unable to extract frames. The clustering stride must be at least one, but got {clustering_stride}."
        raise ValueError(message)
    # DeepLabCut's read_config rewrites config.yaml whenever project_path differs from the config's own directory. With
    # many workers reading concurrently, that write races against sibling reads and intermittently returns an empty
    # config. To avoid that, the project_path (and, like the DeepLabCut GUI, any frames_per_video override) is
    # normalized here, single-threaded, and persisted before any worker starts. The -1 sentinel leaves numframes2pick
    # untouched.
    configuration = normalize_project_config(
        config_path, frames_per_video=frames_per_video, error_context="Unable to extract frames."
    )
    start_fraction = float(configuration.get("start", 0))
    stop_fraction = float(configuration.get("stop", 1))
    configured_frames_per_video = configuration.get("numframes2pick", "?")

    if "video_sets" not in configuration:
        message = "Unable to extract frames. The project's config.yaml does not define any video_sets."
        raise ValueError(message)
    videos = list(configuration["video_sets"])
    if path_filters:
        videos = [video for video in videos if any(token in video for token in path_filters)]
    if not videos:
        message = "Unable to extract frames. No videos in the project's config.yaml matched the requested selection."
        raise ValueError(message)
    # Two videos that share a stem would map to one labeled-data folder, so their frame counts and writes collide;
    # this is checked before sampling, whose per-video accounting reads those same stem-keyed folders.
    ensure_unique_video_stems(videos, error_context="Unable to extract frames.")

    project_directory_path = config_path.parent
    if reset:
        reset_directories = [project_directory_path / "labeled-data" / Path(video).stem for video in videos]
        cleared = sum(1 for directory in reset_directories if directory.exists())
        sys.stderr.write(
            f"WARNING: --reset is removing {cleared} labeled-data video folder(s), including all their extracted "
            f"frames and labels.\n"
        )
        sys.stderr.flush()
        for directory in reset_directories:
            _remove_labeled_data_directory(directory=directory)

    if total_frame_budget == -1 and (balance_groups or group_by_pattern is not None or always_include_videos):
        sys.stderr.write(
            "WARNING: --balance-groups, --group-by, and --always-include only apply when sampling toward a frame "
            "budget. Pass --total-frames to enable budgeted sampling, otherwise every selected video is extracted.\n"
        )
        sys.stderr.flush()

    existing_frame_count = 0
    target_frame_count = -1
    if total_frame_budget != -1:
        frames_per_video_count = configuration.get("numframes2pick")
        if not isinstance(frames_per_video_count, int) or frames_per_video_count < 1:
            message = (
                "Unable to sample videos for a frame budget. The project's numframes2pick must be a positive "
                f"integer, but got {frames_per_video_count!r}. Pass frames_per_video to set it."
            )
            raise ValueError(message)
        groups = (
            group_videos(videos, group_by_pattern=group_by_pattern)
            if (balance_groups or group_by_pattern is not None)
            else None
        )
        pinned_videos, unmatched = resolve_video_overrides(always_include_videos=always_include_videos, videos=videos)
        for token in unmatched:
            sys.stderr.write(f"WARNING: the --always-include value {token!r} matched no video in the selection.\n")
        plan = plan_video_sampling(
            videos=videos,
            extracted_frame_counts=_count_extracted_frames(videos=videos, project_directory=project_directory_path),
            frames_per_video_count=frames_per_video_count,
            total_frame_budget=total_frame_budget,
            random_seed=random_seed,
            groups=groups,
            pinned_videos=pinned_videos,
        )
        _report_sampling_plan(plan=plan)
        if not plan.selected_videos:
            prune_empty_labeled_data_directories(project_directory_path, display_progress=display_progress)
            return FrameExtractionSummary(
                extracted_video_count=0,
                skipped_video_count=0,
                total_video_count=0,
                worker_count=0,
                used_core_count=0,
                total_core_count=os.cpu_count() or 1,
                clustering_frame_count=0,
                existing_frame_count=plan.existing_frame_count,
                target_frame_count=plan.target_frame_count,
            )
        videos = list(plan.selected_videos)
        existing_frame_count = plan.existing_frame_count
        target_frame_count = plan.target_frame_count

    total_core_count = os.cpu_count() or 1
    worker_count, core_sets = plan_core_allocation(
        video_count=len(videos),
        total_core_count=total_core_count,
        worker_count=worker_count,
        cores_per_worker=cores_per_worker,
        reserved_core_count=reserved_core_count,
    )
    used_core_count = len({core for core_set in core_sets for core in core_set})

    frame_totals = _count_clustering_frames(
        videos=videos, start_fraction=start_fraction, stop_fraction=stop_fraction, clustering_stride=clustering_stride
    )
    clustering_frame_count = sum(frame_totals.values())

    if display_progress:
        _report_plan(
            video_count=len(videos),
            configured_frames_per_video=configured_frames_per_video,
            clustering_stride=clustering_stride,
            clustering_resize_width=clustering_resize_width,
            cluster_in_color=cluster_in_color,
            worker_count=worker_count,
            used_core_count=used_core_count,
            total_core_count=total_core_count,
            clustering_frame_count=clustering_frame_count,
            config_path=config_path,
        )

    # Each worker decodes one video pinned to a disjoint core block, streaming progress to the shared aggregate bar.
    def build_tasks(reporting_queue: Any | None) -> list[tuple[Any, ...]]:
        """Packs one work item per selected video, embedding the progress queue only when progress is displayed."""
        return [
            (
                video,
                config_path,
                clustering_stride,
                clustering_resize_width,
                cluster_in_color,
                overwrite,
                video_index,
                frame_totals[video_index],
                reporting_queue,
            )
            for video_index, video in enumerate(videos)
        ]

    extracted_count = skipped_count = 0
    errors: list[tuple[str, str]] = []
    for video, _written, status in iter_pinned_extraction(
        videos=videos,
        make_tasks=build_tasks,
        worker=_extract_one_video,
        worker_count=worker_count,
        core_sets=core_sets,
        frame_totals=frame_totals,
        minimum_progress_interval=minimum_progress_interval,
        display_progress=display_progress,
    ):
        if status == "ok":
            extracted_count += 1
        elif status == "skipped":
            skipped_count += 1
        else:
            errors.append((video, status))

    prune_empty_labeled_data_directories(project_directory_path, display_progress=display_progress)
    return FrameExtractionSummary(
        extracted_video_count=extracted_count,
        skipped_video_count=skipped_count,
        total_video_count=len(videos),
        worker_count=worker_count,
        used_core_count=used_core_count,
        total_core_count=total_core_count,
        clustering_frame_count=clustering_frame_count,
        existing_frame_count=existing_frame_count,
        target_frame_count=target_frame_count,
        errors=tuple(errors),
    )


def _report_plan(
    video_count: int,
    configured_frames_per_video: int | str,
    clustering_stride: int,
    clustering_resize_width: int,
    *,
    cluster_in_color: bool,
    worker_count: int,
    used_core_count: int,
    total_core_count: int,
    clustering_frame_count: int,
    config_path: Path,
) -> None:
    """Writes the two-line run header describing the resolved extraction plan to the standard error stream.

    Args:
        video_count: The number of videos selected for the run.
        configured_frames_per_video: The resolved ``numframes2pick`` value from config.yaml.
        clustering_stride: The stride, in frames, between the frames sampled for clustering.
        clustering_resize_width: The downsample width applied before clustering.
        cluster_in_color: Determines whether clustering runs on color channels.
        worker_count: The resolved number of concurrent workers.
        used_core_count: The number of distinct cores the workers are pinned across.
        total_core_count: The total number of cores on the machine.
        clustering_frame_count: The total number of frames to read and cluster.
        config_path: The resolved path to the project's config.yaml.
    """
    free_core_count = total_core_count - used_core_count
    sys.stderr.write(
        f"k-means extraction | {video_count} videos | numframes2pick={configured_frames_per_video} | "
        f"cluster_step={clustering_stride} | resize_width={clustering_resize_width} | color={cluster_in_color}\n"
    )
    sys.stderr.write(
        f"workers={worker_count} | {used_core_count}/{total_core_count} cores used ({free_core_count} free) | "
        f"{clustering_frame_count:,} frames to cluster | config={config_path}\n"
    )
    sys.stderr.flush()


def _report_sampling_plan(plan: VideoSamplingPlan) -> None:
    """Writes the budgeted-sampling outcome, including any budget warnings, to the standard error stream.

    Args:
        plan: The sampling plan whose selection and budget flags are reported.
    """
    if plan.budget_already_met:
        sys.stderr.write(
            f"WARNING: the project already holds {plan.existing_frame_count:,} frames, which meets the requested total "
            f"of {plan.target_frame_count:,}. Nothing will be extracted. Raise the total frame budget to grow the set, "
            f"or use reset to start over.\n"
        )
    elif not plan.selected_videos:
        sys.stderr.write(
            "WARNING: every selected video already has extracted frames, so none remain to sample. Use overwrite "
            "or reset to re-extract from existing videos.\n"
        )
    elif plan.target_unreachable:
        sys.stderr.write(
            f"WARNING: too few un-extracted videos remain to reach {plan.target_frame_count:,} frames. Sampling all "
            f"{len(plan.selected_videos)} remaining video(s) for a projected {plan.projected_frame_count:,} frames.\n"
        )
    else:
        sys.stderr.write(
            f"sampling {len(plan.selected_videos)} video(s) | {plan.existing_frame_count:,} existing -> "
            f"{plan.projected_frame_count:,} projected frames (target {plan.target_frame_count:,})\n"
        )

    if plan.per_group:
        sampled = sum(1 for (_group, _existing, added, _projected, _available) in plan.per_group if added > 0)
        group_count = len(plan.per_group)
        distribution = ", ".join(
            f"{group}+{added}"
            for (group, _existing, added, _projected, _available) in sorted(
                plan.per_group, key=lambda item: (-item[2], item[0])
            )
            if added > 0
        )
        sys.stderr.write(f"group balancing: {sampled}/{group_count} groups sampled this pass | {distribution}\n")
        # A group with no un-extracted videos left is fully done, not starved for budget, so only the groups that still
        # had eligible videos but received none are flagged as needing a larger budget.
        starved = sum(
            1 for (_group, _existing, added, _projected, available) in plan.per_group if added == 0 and available > 0
        )
        if starved:
            sys.stderr.write(
                f"WARNING: {starved} group(s) had un-extracted videos but received none this pass because the budget "
                f"is too small. Raise the total frame budget to include them.\n"
            )
    if plan.always_included_overshoot:
        sys.stderr.write(
            "WARNING: the always-included videos alone exceeded the frame budget, so the projected total overshoots "
            "the target by the surplus always-included videos.\n"
        )
    sys.stderr.flush()


def _count_extracted_frames(videos: list[str], project_directory: Path) -> dict[str, int]:
    """Counts the frames already extracted for each video, keyed by the video's path.

    Args:
        videos: The candidate video paths to inspect.
        project_directory: The DeepLabCut project directory that holds the ``labeled-data`` tree.

    Returns:
        A mapping of video path to the number of extracted ``img*.png`` frames currently in its labeled-data
        directory, excluding any ``*labeled.png`` prediction overlays.
    """
    counts: dict[str, int] = {}
    for video in videos:
        directory = project_directory / "labeled-data" / Path(video).stem
        counts[video] = len(extracted_frame_paths(directory))
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


def _remove_labeled_data_directory(directory: Path) -> None:
    """Deletes a video's entire labeled-data directory, including its extracted frames, labels, and the folder itself.

    The reset workflow starts a video's extraction over from scratch, so the whole folder is removed rather than only
    its known artifacts, leaving no empty directory behind for the labeler to sift through.

    Args:
        directory: The ``labeled-data/<video>`` directory to remove.
    """
    if directory.exists():
        shutil.rmtree(directory)


def _count_clustering_frames(
    videos: list[str], start_fraction: float, stop_fraction: float, clustering_stride: int
) -> dict[int, int]:
    """Counts the frames DeepLabCut will read and cluster for each video, keyed by the video's index.

    The per-video total mirrors DeepLabCut's own sampling: the frames between the configured start and stop bounds
    are visited with the given stride.

    Args:
        videos: The ordered list of video paths to inspect.
        start_fraction: The fractional start position within each video, in the range [0, 1].
        stop_fraction: The fractional stop position within each video, in the range [0, 1].
        clustering_stride: The sampling stride; every ``clustering_stride``-th frame is clustered.

    Returns:
        A mapping of video index to the number of frames that video contributes to the aggregate total.
    """
    cv2.setNumThreads(1)

    def _frame_count(video: str) -> int:
        """Reads one video's frame count from its container header, releasing the capture regardless of outcome."""
        capture = cv2.VideoCapture(video)
        try:
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()

    # Opening a container to read its header is I/O-bound and releases the GIL, so the header reads overlap across a
    # small thread pool rather than serializing one video at a time. executor.map preserves the input order.
    frame_totals: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=min(len(videos), 8)) as executor:
        for video_index, frame_count in enumerate(executor.map(_frame_count, videos)):
            start_index, end_index = math.floor(frame_count * start_fraction), math.ceil(frame_count * stop_fraction)
            frame_totals[video_index] = max(1, len(range(start_index, end_index, clustering_stride)))
    return frame_totals


def _extract_one_video(task: tuple[Any, ...]) -> tuple[str, int, str]:
    """Runs DeepLabCut k-means extraction for a single video and reports the outcome.

    DeepLabCut's console output is silenced and exceptions are captured, so one bad video cannot kill the worker
    pool. The native math-library thread pools are pinned to a single thread (configured in the package __init__
    before the heavy backends import), so each worker stays within its assigned CPU budget.

    Args:
        task: The packed work item carrying the video path, the config path, the clustering parameters, the resume
            flag, the video index, the per-video frame total, and the progress queue (or None when progress is off).

    Returns:
        A tuple of the video path, the number of frames written, and a status string (``"ok"``, ``"skipped"``,
        ``"empty"``, or an ``"error:"`` traceback).
    """
    (
        video_path,
        config_path,
        clustering_stride,
        clustering_resize_width,
        cluster_in_color,
        overwrite,
        video_index,
        frame_total,
        progress_queue,
    ) = task
    try:
        cv2.setNumThreads(1)

        stem = Path(video_path).stem
        output_directory = config_path.parent / "labeled-data" / stem

        existing_frame_paths = extracted_frame_paths(output_directory)
        if existing_frame_paths and not overwrite:
            return video_path, len(existing_frame_paths), "skipped"
        # On overwrite, drop the stale frames and the labels they would orphan before re-extracting.
        _clear_extracted_data(output_directory=output_directory)

        # Swaps DeepLabCut's random-seek k-means reader for the decode-aware one, routing its per-candidate progress to
        # the parent's aggregate bar. The reader streams the video when the clustering stride is dense enough to favor
        # it, and keeps seeking otherwise. The queue is None when progress is disabled, leaving a plain bar.
        progress_reporter = (
            make_progress_reporter(progress_queue=progress_queue, video_index=video_index, frame_total=frame_total)
            if progress_queue is not None
            else None
        )
        frame_selection_tools.KmeansbasedFrameselectioncv2 = make_fast_kmeans_selector(progress=progress_reporter)

        with (
            Path(os.devnull).open("w") as null_stream,
            contextlib.redirect_stdout(null_stream),
            contextlib.redirect_stderr(null_stream),
        ):
            deeplabcut.extract_frames(
                str(config_path),
                mode="automatic",
                algo="kmeans",
                # Applies the per-video crop stored in config.yaml.
                crop=True,
                # Runs non-interactively.
                userfeedback=False,
                cluster_step=clustering_stride,
                cluster_resizewidth=clustering_resize_width,
                cluster_color=cluster_in_color,
                # Restricts DeepLabCut to this one video.
                videos_list=[video_path],
            )
    except Exception:  # noqa: BLE001 -- one bad video must not kill the pool; the traceback is returned as status.
        return video_path, 0, "error:\n" + traceback.format_exc()
    else:
        written = len(extracted_frame_paths(output_directory))
        return video_path, written, "ok" if written else "empty"
