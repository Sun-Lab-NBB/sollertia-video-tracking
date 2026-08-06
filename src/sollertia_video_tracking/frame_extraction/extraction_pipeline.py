"""Provides the parallel k-means frame-extraction pipeline that prepares a DeepLabCut project's training frames."""

import os
import sys
import math
from typing import Any
from pathlib import Path
import traceback
import contextlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import cv2
import pandas as pd
import deeplabcut
import deeplabcut.utils.frameselectiontools as frame_selection_tools

from .progress import make_progress_reporter
from .utilities import (
    extracted_frame_paths,
    iter_pinned_extraction,
    normalize_project_config,
    select_registered_videos,
    ensure_unique_video_stems,
    machine_label_frame_names,
    finite_labeled_frame_names,
    has_outlier_refinement_data,
    prune_empty_labeled_data_directories,
)
from .frame_reading import make_fast_kmeans_selector
from .cpu_allocation import DEFAULT_RESERVED_CORE_COUNT, plan_core_allocation
from .video_grouping import group_videos
from .video_sampling import VideoSamplingPlan, plan_video_sampling

_MAXIMUM_HEADER_READ_THREADS: int = 8
"""The largest thread pool used to read video container headers concurrently while sizing the aggregate bar."""


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
    cleared_frame_count: int
    """The number of unlabeled bootstrap frames removed across the selection by ``overwrite`` or ``reset`` before
    re-extraction. Zero when neither option was set."""
    total_video_count: int
    """The total number of videos considered in the run."""
    worker_count: int
    """The number of worker processes that ran concurrently."""
    used_core_count: int
    """The number of distinct CPU cores the workers were pinned across."""
    total_core_count: int
    """The total number of CPU cores available on the machine."""
    clustering_frame_count: int
    """The total number of frames scheduled to be read and clustered across the videos selected for extraction this
    pass, estimated from the video headers before extraction. Videos already at the per-video ceiling, and videos
    already in outlier refinement, are excluded from this count."""
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
    balance_groups: bool = False,
    group_by_pattern: str | None = None,
    requested_videos: tuple[str | Path, ...] = (),
    exclusive: bool = False,
    clustering_resize_width: int = 30,
    cluster_in_color: bool = False,
    overwrite: bool = False,
    reset: bool = False,
    display_progress: bool = True,
) -> FrameExtractionSummary:
    """Runs DeepLabCut k-means frame extraction across a project's videos in parallel and reports the outcome.

    Reads the run parameters from the project's config.yaml, plans the CPU-core allocation, and clusters every selected
    video in its own pinned worker process. ``numframes2pick`` is a per-video ceiling: each selected video is topped up
    to it. A not-yet-extracted video gains a full set, while a partly-extracted one gains only the frames that reach the
    ceiling, and a video already at the ceiling is skipped. ``overwrite`` and ``reset`` instead clear a video's
    unlabeled bootstrap frames first so it is re-rolled from scratch, always preserving every human-labeled and
    outlier-extracted frame. Frame extraction is the bootstrap step that precedes outlier refinement, so a video already
    in refinement is off-limits: it is dropped from the candidate pool, and an explicit refined ``requested_videos``
    target is refused under ``overwrite``. A single bad video is recorded in the returned summary rather than aborting
    the run.

    When ``total_frame_budget`` is set, the run selects just enough videos to reach that project-wide frame total,
    preferring not-yet-extracted videos and falling back to below-ceiling ones, so coverage grows before existing videos
    are deepened. The full project's video set can be added once and repeated passes grow it toward the budget without
    manual selection. When the existing frames already meet the budget, the run extracts nothing and warns. If even
    topping every eligible video to the ceiling cannot reach the budget in one pass, the run reports the shortfall and
    raises rather than extracting a partial set.

    Notes:
        The pipeline uses the spawn multiprocessing start method on every platform for reproducible behavior, so a
        programmatic caller must guard the call with ``if __name__ == "__main__":``. The installed console-script entry
        point is already guarded. CPU-affinity pinning is applied on Linux and Windows. macOS exposes no affinity API,
        so its workers run unpinned but still in parallel.

        ``overwrite`` and ``reset`` remove only a video's bare bootstrap frames: the extracted images that carry no
        human label and belong to no outlier-refinement iteration. Frames the human has annotated (a finite
        ``CollectedData`` coordinate) and machine-labeled outlier frames are always kept, so re-rolling the diverse
        bootstrap set never disturbs labeling or outlier work. ``overwrite`` clears the videos this run selects to
        process (refusing any already in outlier refinement), while ``reset`` clears every non-refined project video, so
        the run re-rolls its whole selection or the whole project respectively. The video subset is drawn fresh each
        run, and k-means picks the frames within a video, so a re-rolled selection differs each run. To reproduce a
        specific selection, name the videos explicitly with ``requested_videos``. To instead discard a video's labels
        and start its ``labeled-data`` directory over from scratch, use ``purge_labeled_data``.

        Empty ``labeled-data`` directories left by videos that were registered but never extracted are removed after
        every run, so the labeling GUI shows only the videos that have frames.

    Args:
        config_path: The path to the DeepLabCut project's config.yaml.
        clustering_stride: The clustering stride passed to DeepLabCut as ``cluster_step``. Every Nth frame is sampled,
            where N is this stride.
        worker_count: The number of videos to decode in parallel. Set to -1 to fill the usable cores automatically.
        cores_per_worker: The number of CPU cores pinned to each worker. Set to -1 to give each worker a saturating
            core block when the worker count is automatic, or to split the usable cores evenly across an explicit
            worker count.
        reserved_core_count: The number of CPU cores to leave free for other tasks.
        frames_per_video: The per-video frame ceiling, overriding ``numframes2pick`` in config.yaml. Each selected
            video is topped up to this many frames. Set to -1 to use the value already stored in the configuration file.
        total_frame_budget: The total number of frames the project should hold, reached by topping up videos toward
            their per-video ceiling, preferring not-yet-extracted videos over below-ceiling ones. Set to -1 to top up
            every below-ceiling selected video instead of sampling toward a budget.
        balance_groups: Determines whether the budgeted sampling is balanced across groups rather than drawn
            uniformly, so every group is represented and coverage evens out across repeated passes. The group of each
            video is inferred from its file name, with videos that share their non-date name components grouped
            together. Only affects the run when ``total_frame_budget`` is set.
        group_by_pattern: A regular expression whose first capturing group names the group for each video's file-name
            stem, overriding the built-in inference for naming schemes it does not cover. Setting it implies
            ``balance_groups``.
        requested_videos: The specific project video files this run targets, matched against the project's registered
            videos by resolved path. In budgeted mode they are the always-included pins, selected before the remaining
            budget is filled from the project's other videos. With ``exclusive`` they are the whole set to top up.
            Ignored when neither a budget, ``exclusive``, nor ``overwrite`` applies.
        exclusive: Determines whether to restrict the run to exactly the ``requested_videos``, topping each up to
            ``frames_per_video`` frames and bypassing the total-frame budget and group balancing. Requires
            ``requested_videos`` to be non-empty. A requested video already at the ceiling is skipped unless
            ``overwrite`` clears it first.
        clustering_resize_width: The downsample width applied before clustering, passed to DeepLabCut as
            ``cluster_resizewidth``.
        cluster_in_color: Determines whether to cluster on color channels instead of grayscale.
        overwrite: Determines whether to clear the selected videos' unlabeled bootstrap frames before re-extracting, so
            the run re-rolls the diverse selection for every video it processes rather than topping it up. Any
            ``requested_videos`` already in outlier refinement are refused. Human labels and outlier-extracted frames
            are preserved. Mutually exclusive with ``reset``.
        reset: Determines whether to clear every registered project video of its unlabeled bootstrap frames before
            re-extracting, re-rolling the diverse selection project-wide. Videos already in outlier refinement are left
            untouched. Human labels and outlier-extracted frames are preserved. Mutually exclusive with ``overwrite``.
        display_progress: Determines whether to render the run header and the aggregate progress bar to the standard
            error stream.

    Returns:
        A FrameExtractionSummary describing how many videos were extracted and failed and how many bootstrap frames
        were cleared, alongside the resolved core-allocation plan and, in sampling mode, the existing and target frame
        counts.

    Raises:
        FileNotFoundError: If ``config_path`` does not point to an existing file.
        ValueError: Raised when the options conflict: ``overwrite`` and ``reset`` are both set, ``exclusive`` and
            ``reset`` are both set, or ``exclusive`` is set without any ``requested_videos``. Raised when ``overwrite``
            targets a video already in outlier refinement. Raised when a value is out of range: ``clustering_stride`` is
            below one, or ``frames_per_video`` or ``total_frame_budget`` (other than the -1 sentinel) is below one.
            Raised when the project's ``numframes2pick`` is not a positive integer, when config.yaml defines no
            ``video_sets``, or when an exclusive run's requested videos match no eligible registered project video.
            Raised when a budgeted run cannot reach ``total_frame_budget`` in one pass even after topping every eligible
            video to its ceiling. Raised when two selected videos share a file-name stem and would collide in the
            labeled-data tree. Raised when an explicit ``worker_count`` or ``cores_per_worker``, or their product, needs
            more cores than remain usable after reserving ``reserved_core_count``.
    """
    config_path = config_path.resolve()
    if overwrite and reset:
        message = "Unable to extract frames. The overwrite and reset options are mutually exclusive."
        raise ValueError(message)
    if exclusive and not requested_videos:
        message = "Unable to extract frames. The exclusive option requires at least one requested video."
        raise ValueError(message)
    if exclusive and reset:
        message = (
            "Unable to extract frames. The exclusive and reset options are contradictory: reset re-rolls the whole "
            "project, while exclusive restricts extraction to the requested videos. Use overwrite to re-extract only "
            "the requested videos."
        )
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
        config_path=config_path, frames_per_video=frames_per_video, error_context="Unable to extract frames."
    )
    start_fraction = float(configuration.get("start", 0))
    stop_fraction = float(configuration.get("stop", 1))
    configured_frames_per_video = configuration.get("numframes2pick", "?")
    scorer = str(configuration.get("scorer", ""))

    if "video_sets" not in configuration:
        message = "Unable to extract frames. The project's config.yaml does not define any video_sets."
        raise ValueError(message)
    videos = list(configuration["video_sets"])
    if not videos:
        message = "Unable to extract frames. The project's config.yaml does not list any videos in video_sets."
        raise ValueError(message)
    project_directory_path = config_path.parent
    labeled_data_directory = project_directory_path / "labeled-data"

    # The requested videos are matched against the registered project videos whenever the match is used. The match
    # serves as always-included pins over the full project pool in budgeted mode, as the whole set with exclusive, or to
    # drive the unregistered warning and the in-refinement refusal under overwrite. Overwrite does not let --videos
    # steer the selection, as an unbudgeted non-exclusive run always tops up every below-ceiling video (warned below),
    # so matching outside these three cases would only emit a misleading per-video warning.
    requested_matched: list[str] = []
    if requested_videos and (exclusive or overwrite or total_frame_budget != -1):
        matched_videos, unmatched_videos = select_registered_videos(
            registered_videos=videos, requested_videos=tuple(requested_videos)
        )
        for video in unmatched_videos:
            sys.stderr.write(f"WARNING: {video} is not registered in the project's config.yaml and was skipped.\n")
        sys.stderr.flush()
        requested_matched = list(matched_videos)

    # Frame extraction is the pre-refinement bootstrap step, so it never touches a video already in outlier refinement.
    # Refined videos are dropped from the candidate pool, and an explicit refined --videos target is refused under
    # overwrite and skipped with a warning otherwise.
    def _is_in_refinement(video: str) -> bool:
        """Reports whether a candidate video's labeled-data directory already holds outlier-refinement tables."""
        return has_outlier_refinement_data(labeled_data_directory / Path(video).stem)

    refined_requested = [video for video in requested_matched if _is_in_refinement(video)]
    if refined_requested and overwrite:
        listed_videos = ", ".join(refined_requested)
        message = (
            "Unable to extract frames. These --videos are already in outlier refinement and cannot be re-extracted "
            f"with --overwrite: {listed_videos}. They belong to the refinement workflow ('extract outliers')."
        )
        raise ValueError(message)
    for video in refined_requested:
        sys.stderr.write(f"WARNING: {video} is already in outlier refinement and was skipped.\n")
    sys.stderr.flush()
    requested_matched = [video for video in requested_matched if not _is_in_refinement(video)]

    if exclusive:
        videos = list(requested_matched)
        if not videos:
            message = (
                "Unable to extract frames. None of the requested videos matched an eligible registered project video, "
                "so the exclusive run has nothing to extract."
            )
            raise ValueError(message)
    else:
        videos = [video for video in videos if not _is_in_refinement(video)]

    # Exclusive tops up the requested videos directly, so they are not pins. In every other mode they are the pins the
    # budgeted draw always includes first.
    pinned_videos: tuple[str, ...] = () if exclusive else tuple(requested_matched)
    # Two videos that share a stem would map to one labeled-data directory, so their frame counts and writes collide.
    # This is checked before sampling, whose per-video accounting reads those same stem-keyed directories.
    ensure_unique_video_stems(videos=videos, error_context="Unable to extract frames.")

    # numframes2pick is the per-video ceiling every selected video is topped up to. It is resolved once here and used
    # both to size the budgeted selection and to derive each worker's per-video pick count.
    frames_per_video_count = configuration.get("numframes2pick")
    if not isinstance(frames_per_video_count, int) or frames_per_video_count < 1:
        message = (
            "Unable to extract frames. The project's numframes2pick must be a positive integer, but got "
            f"{frames_per_video_count!r}. Pass frames_per_video to set it."
        )
        raise ValueError(message)

    cleared_frame_count = 0
    if reset:
        # Reset re-rolls the whole bootstrap: it clears every candidate video's unlabeled frames before selection, so
        # the cleared frames no longer count toward the per-video ceiling and each video is topped back up from its
        # surviving human labels. Videos already in outlier refinement were excluded from the candidate pool above.
        removed_frame_count, _ = _clear_bare_frames(
            project_directory=project_directory_path,
            video_stems=[Path(video).stem for video in videos],
            scorer=scorer,
            scope_label="--reset",
        )
        cleared_frame_count += removed_frame_count

    def _nothing_to_extract(existing_frames: int, target_frames: int) -> FrameExtractionSummary:
        """Prunes the empty labeled-data directories and returns a do-nothing summary carrying the frames cleared so
        far.
        """
        prune_empty_labeled_data_directories(
            project_directory=project_directory_path, display_progress=display_progress
        )
        return FrameExtractionSummary(
            extracted_video_count=0,
            cleared_frame_count=cleared_frame_count,
            total_video_count=0,
            worker_count=0,
            used_core_count=0,
            total_core_count=os.cpu_count() or 1,
            clustering_frame_count=0,
            existing_frame_count=existing_frames,
            target_frame_count=target_frames,
        )

    if exclusive and (balance_groups or group_by_pattern is not None):
        sys.stderr.write(
            "WARNING: --balance-groups and --group-regex are ignored with --exclusive, which tops up each requested "
            "video to --frames-per-video directly.\n"
        )
        sys.stderr.flush()
    budgetless_options_ignored = (
        total_frame_budget == -1
        and not exclusive
        and (balance_groups or group_by_pattern is not None or requested_videos)
    )
    if budgetless_options_ignored:
        sys.stderr.write(
            "WARNING: --balance-groups, --group-regex, and --videos only apply when sampling toward a frame budget. "
            "Pass --total-frames to enable budgeted sampling (or --exclusive to extract only the requested videos), "
            "otherwise every below-ceiling project video is topped up.\n"
        )
        sys.stderr.flush()

    existing_frame_count = 0
    target_frame_count = -1
    extracted_counts = _count_extracted_frames(videos=videos, project_directory=project_directory_path)
    if total_frame_budget != -1 and not exclusive:
        groups = (
            group_videos(videos=videos, group_by_pattern=group_by_pattern)
            if (balance_groups or group_by_pattern is not None)
            else None
        )
        plan = plan_video_sampling(
            videos=videos,
            extracted_frame_counts=extracted_counts,
            frames_per_video_count=frames_per_video_count,
            total_frame_budget=total_frame_budget,
            groups=groups,
            pinned_videos=pinned_videos,
        )
        if plan.target_unreachable:
            message = (
                f"Unable to reach the requested total of {total_frame_budget:,} frames in one pass. Topping every "
                f"eligible video up to {frames_per_video_count} frames yields at most {plan.projected_frame_count:,} "
                "frames. Lower --total-frames, raise --frames-per-video, or register more videos."
            )
            raise ValueError(message)
        _report_sampling_plan(plan=plan)
        if not plan.selected_videos:
            return _nothing_to_extract(existing_frames=plan.existing_frame_count, target_frames=plan.target_frame_count)
        selected_videos = list(plan.selected_videos)
        existing_frame_count = plan.existing_frame_count
        target_frame_count = plan.target_frame_count
    elif exclusive:
        selected_videos = list(videos)
    else:
        # Unbudgeted: tops up every below-ceiling candidate video to its per-video ceiling.
        selected_videos = [video for video in videos if extracted_counts.get(video, 0) < frames_per_video_count]

    # Overwrite re-rolls the whole selected set: each selected video's unlabeled frames are cleared here, after the
    # selection, so re-extraction re-clusters from the surviving human labels rather than adding to them. Reset already
    # cleared the candidate pool above, and the two options are mutually exclusive.
    if overwrite and selected_videos:
        removed_frame_count, _ = _clear_bare_frames(
            project_directory=project_directory_path,
            video_stems=[Path(video).stem for video in selected_videos],
            scorer=scorer,
            scope_label="--overwrite",
        )
        cleared_frame_count += removed_frame_count

    # Each selected video is topped up to the per-video ceiling: it picks the frames that reach the ceiling from
    # whatever it holds after any clearing (its surviving human labels, or nothing). A video already at the ceiling
    # picks nothing and is dropped, so a re-run never overshoots the ceiling.
    post_clear_counts = _count_extracted_frames(videos=selected_videos, project_directory=project_directory_path)
    pick_counts = {video: max(0, frames_per_video_count - post_clear_counts.get(video, 0)) for video in selected_videos}
    videos = [video for video in selected_videos if pick_counts[video] > 0]
    if not videos:
        return _nothing_to_extract(existing_frames=existing_frame_count, target_frames=target_frame_count)

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

    # Crops the extracted frames to each video's configured region only when the project is set to crop, so extraction
    # honors the same project-wide cropping toggle inference and outlier extraction do.
    crop_frames = bool(configuration.get("cropping", False))

    # Decodes one video per worker, pinned to a disjoint core block, streaming progress to the shared aggregate bar.
    def build_tasks(reporting_queue: Any | None) -> list[tuple[Any, ...]]:
        """Packs one work item per selected video, embedding the progress queue only when progress is displayed."""
        return [
            (
                video,
                config_path,
                clustering_stride,
                clustering_resize_width,
                cluster_in_color,
                video_index,
                frame_totals[video_index],
                pick_counts[video],
                reporting_queue,
                crop_frames,
            )
            for video_index, video in enumerate(videos)
        ]

    extracted_count = 0
    errors: list[tuple[str, str]] = []
    for video, _written, status in iter_pinned_extraction(
        videos=videos,
        make_tasks=build_tasks,
        worker=_extract_one_video,
        worker_count=worker_count,
        core_sets=core_sets,
        frame_totals=frame_totals,
        display_progress=display_progress,
    ):
        if status == "ok":
            extracted_count += 1
        else:
            errors.append((video, status))

    prune_empty_labeled_data_directories(project_directory=project_directory_path, display_progress=display_progress)
    return FrameExtractionSummary(
        extracted_video_count=extracted_count,
        cleared_frame_count=cleared_frame_count,
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
        # A group with no below-ceiling videos left is fully done, not starved for budget, so only the groups that still
        # had eligible videos but received none are flagged as needing a larger budget.
        starved = sum(
            1 for (_group, _existing, added, _projected, available) in plan.per_group if added == 0 and available > 0
        )
        if starved:
            sys.stderr.write(
                f"WARNING: {starved} group(s) had below-ceiling videos but received none this pass because the budget "
                f"is too small. Raise the total frame budget to include them.\n"
            )
    if plan.always_included_overshoot:
        sys.stderr.write(
            "WARNING: the videos named with --videos alone exceeded the frame budget, so the projected total "
            "overshoots the target by the surplus pinned videos.\n"
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
    return {
        video: len(extracted_frame_paths(project_directory / "labeled-data" / Path(video).stem)) for video in videos
    }


def _clear_bare_frames(
    *, project_directory: Path, video_stems: list[str], scorer: str, scope_label: str
) -> tuple[int, set[str]]:
    """Clears each named video's unlabeled bootstrap frames before re-extraction, reporting what was removed.

    Runs single-threaded before the workers start, so re-reading and rewriting the per-video label tables never races
    against the concurrent extraction. A video directory that cannot be read is warned about and left untouched rather
    than aborting the run or risking an under-protective deletion.

    Args:
        project_directory: The DeepLabCut project directory that holds the ``labeled-data`` tree.
        video_stems: The file-name stems of the videos whose bare frames are cleared.
        scorer: The human scorer naming the ``CollectedData`` labels whose finite-labeled frames are preserved.
        scope_label: The option name (``--overwrite`` or ``--reset``) reported in the clearing summary and any warning.

    Returns:
        A tuple of the total number of bare frames removed across the named videos and the set of video stems that
        had at least one bare frame removed.
    """
    labeled_data_directory = project_directory / "labeled-data"
    removed_frame_count = 0
    cleared_stems: set[str] = set()
    for stem in video_stems:
        directory = labeled_data_directory / stem
        try:
            removed_count = _clear_bare_frames_in_directory(directory=directory, scorer=scorer)
        except Exception:
            sys.stderr.write(f"WARNING: {scope_label} could not clear the unlabeled frames in '{directory}'.\n")
            continue
        if removed_count:
            cleared_stems.add(stem)
        removed_frame_count += removed_count

    sys.stderr.write(
        f"{scope_label} cleared {removed_frame_count} unlabeled bootstrap frame(s) from {len(cleared_stems)} "
        f"video directory(ies) before re-extraction.\n"
    )
    sys.stderr.flush()
    return removed_frame_count, cleared_stems


def _clear_bare_frames_in_directory(*, directory: Path, scorer: str) -> int:
    """Removes a video's extracted frames that carry no label of any kind, keeping labeled and outlier frames.

    A bare frame is an extracted ``imgNNNN.png`` that neither carries a finite human ``CollectedData`` coordinate nor
    belongs to any outlier-refinement machine-label table. Its image and any prediction overlay are deleted, and its
    all-NaN placeholder row, if the labeling GUI created one, is dropped from ``CollectedData`` so no label dangles.

    Args:
        directory: The ``labeled-data/<stem>`` directory to clear of unlabeled bootstrap frames.
        scorer: The human scorer naming the ``CollectedData`` labels whose finite-labeled frames are preserved.

    Returns:
        The number of bare frames removed.
    """
    disk_frame_names = {frame_path.name for frame_path in extracted_frame_paths(directory)}
    if not disk_frame_names:
        return 0
    collected_data_path = directory / f"CollectedData_{scorer}.h5"
    protected_frame_names = finite_labeled_frame_names(collected_data_path) | machine_label_frame_names(directory)
    bare_frame_names = disk_frame_names - protected_frame_names
    for frame_name in bare_frame_names:
        (directory / frame_name).unlink(missing_ok=True)
        # A bare frame never has an overlay, but --save-labeled overlays from any earlier run are dropped defensively.
        (directory / f"{Path(frame_name).stem}labeled.png").unlink(missing_ok=True)
    if bare_frame_names:
        _drop_collected_data_rows(collected_data_path=collected_data_path, removed_frame_names=bare_frame_names)
    return len(bare_frame_names)


def _drop_collected_data_rows(*, collected_data_path: Path, removed_frame_names: set[str]) -> None:
    """Drops any placeholder label rows for cleared bare frames from a video's ``CollectedData`` tables.

    The labeling GUI reindexes ``CollectedData`` to every image in the directory, so a cleared bare frame may leave
    behind an all-NaN row referencing the deleted image. Those rows are removed here so no label dangles. When that
    empties the table, its ``.h5`` and ``.csv`` files are deleted outright. Only bare (unlabeled) frames are ever passed
    in, so no finite human label is dropped.

    Args:
        collected_data_path: The ``CollectedData_<scorer>.h5`` file to prune, alongside its ``.csv`` sibling.
        removed_frame_names: The frame image names that were cleared and must not remain in the label tables.
    """
    if not collected_data_path.is_file():
        return
    labels = pd.read_hdf(collected_data_path, key="df_with_missing")
    row_frame_names = [entry[-1] if isinstance(entry, tuple) else Path(str(entry)).name for entry in labels.index]
    keep_row_mask = [frame_name not in removed_frame_names for frame_name in row_frame_names]
    if all(keep_row_mask):
        return
    remaining_labels = labels[keep_row_mask]
    collected_data_csv_path = collected_data_path.with_suffix(".csv")
    if remaining_labels.empty:
        collected_data_path.unlink(missing_ok=True)
        collected_data_csv_path.unlink(missing_ok=True)
        return
    remaining_labels.to_hdf(collected_data_path, key="df_with_missing", mode="w")
    remaining_labels.to_csv(collected_data_csv_path)


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
        clustering_stride: The sampling stride. Every ``clustering_stride``-th frame is clustered.

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
    with ThreadPoolExecutor(max_workers=min(len(videos), _MAXIMUM_HEADER_READ_THREADS)) as executor:
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
        task: The packed work item carrying the video path, the config path, the clustering parameters, the video
            index, the per-video frame total, the per-video pick count, the progress queue (or None when progress is
            off), and the crop flag.

    Returns:
        A tuple of the video path, the number of frames freshly written, and a status string (``"ok"``, ``"empty"``,
        or an ``"error:"`` traceback).
    """
    (
        video_path,
        config_path,
        clustering_stride,
        clustering_resize_width,
        cluster_in_color,
        video_index,
        frame_total,
        pick_count,
        progress_queue,
        crop_frames,
    ) = task
    try:
        cv2.setNumThreads(1)

        stem = Path(video_path).stem
        output_directory = config_path.parent / "labeled-data" / stem
        # Extraction is additive: the pipeline clears any unlabeled frames up front when re-rolling, so the worker
        # always clusters and appends. The pre-existing frame count is captured to report only the freshly written ones.
        frame_count_before = len(extracted_frame_paths(output_directory))

        # Swaps DeepLabCut's random-seek k-means reader for the decode-aware one, routing its per-candidate progress to
        # the parent's aggregate bar. The reader streams the video when the clustering stride is dense enough to favor
        # it, and keeps seeking otherwise. The queue is None when progress is disabled, leaving a plain bar.
        progress_reporter = (
            make_progress_reporter(progress_queue=progress_queue, video_index=video_index, frame_total=frame_total)
            if progress_queue is not None
            else None
        )
        frame_selection_tools.KmeansbasedFrameselectioncv2 = make_fast_kmeans_selector(
            progress=progress_reporter, frame_count=pick_count
        )

        with (
            Path(os.devnull).open("w") as null_stream,
            contextlib.redirect_stdout(null_stream),
            contextlib.redirect_stderr(null_stream),
        ):
            deeplabcut.extract_frames(
                config=str(config_path),
                mode="automatic",
                algo="kmeans",
                # Applies the per-video crop stored in config.yaml only when the project is configured to crop.
                crop=crop_frames,
                # Runs non-interactively.
                userfeedback=False,
                cluster_step=clustering_stride,
                cluster_resizewidth=clustering_resize_width,
                cluster_color=cluster_in_color,
                # Restricts DeepLabCut to this one video.
                videos_list=[video_path],
            )
    except Exception:
        return video_path, 0, "error:\n" + traceback.format_exc()
    else:
        written = len(extracted_frame_paths(output_directory)) - frame_count_before
        return video_path, max(0, written), "ok" if written > 0 else "empty"
