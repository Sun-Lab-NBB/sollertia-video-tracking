"""Provides the parallel outlier-frame extraction pipeline that refines a DeepLabCut model on its likely-wrong frames.

This is the model-refinement counterpart of the k-means frame-extraction pipeline in this subpackage. Instead of
clustering raw video, it reads the predictions a trained model already wrote for each analyzed video, flags the frames
the model most likely got wrong, and pulls a budget of those frames into the project's ``labeled-data`` tree for
correction. It runs in two phases. The detection phase loads every video's predictions and computes its outlier
candidate frames; for the ``fitting`` algorithm the per-keypoint SARIMAX fits of every video are fanned out across one
process pool spanning the whole run, so the expensive path scales with the core budget regardless of how few videos are
processed. The extraction phase then decodes and selects frames one video per pinned worker, exactly like the k-means
pipeline, reusing DeepLabCut's own frame-writing so the machine pre-labels and project registration stay faithful to the
upstream tool.

Because it depends on DeepLabCut's internal outlier-detection and frame-extraction functions, the DeepLabCut version is
pinned exactly in ``pyproject.toml`` and must be re-verified against any new release.
"""

import os
import sys
from typing import Any
from pathlib import Path
import traceback
import contextlib
from dataclasses import dataclass
import multiprocessing

import cv2
import numpy as np
from ruamel.yaml import YAML
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions, frameselectiontools
from deeplabcut.utils.auxfun_videos import collect_video_paths
from deeplabcut.refine_training_dataset import outlier_frames as dlc_outlier_frames

from .progress import AggregateBar, make_progress_reporter
from .cpu_allocation import DEFAULT_RESERVE_CORES, pin_worker_to_cores, plan_core_allocation
from .outlier_detection import (
    OUTLIER_ALGORITHMS,
    jump_outlier_indices,
    fit_keypoint_distance,
    fitting_keypoint_series,
    fitting_outlier_indices,
    uncertain_outlier_indices,
)

_EXTRACTION_ALGORITHMS: tuple[str, ...] = ("kmeans", "uniform")
"""The frame-selection algorithms that pick the extracted frames from the flagged outlier candidates."""

_CROP_FIELD_COUNT: int = 4
"""The number of comma-separated integers, ``x1,x2,y1,y2``, in a video's config.yaml crop specification."""


@dataclass(frozen=True, slots=True)
class OutlierExtractionSummary:
    """Summarizes the outcome of a parallel outlier-frame extraction run.

    Notes:
        The pipeline never aborts on a single bad video, so per-video problems are collected here rather than raised.
        Videos whose predictions are missing are listed in ``not_analyzed`` (they must be analyzed first), and videos
        that raised during detection or extraction are listed in ``errors`` as ``(video_path, detail)`` pairs. Callers
        inspect ``successful`` to decide the process exit status.
    """

    config: Path
    """The path to the DeepLabCut project's config.yaml the run used."""
    algorithm: str
    """The outlier-detection algorithm that flagged the candidate frames."""
    extraction_algorithm: str
    """The frame-selection algorithm that picked the extracted frames from the candidates."""
    total: int
    """The total number of videos considered in the run."""
    extracted: int
    """The number of videos from which outlier frames were extracted."""
    workers: int
    """The number of worker processes that ran concurrently during the extraction phase."""
    cores_used: int
    """The number of distinct CPU cores the extraction workers were pinned across."""
    core_count: int
    """The total number of CPU cores available on the machine."""
    candidate_frames: int
    """The total number of putative outlier frames flagged across the extracted videos."""
    frames_extracted: int
    """The total number of frames freshly written into the project's labeled-data tree across all videos."""
    not_analyzed: tuple[str, ...] = ()
    """The videos skipped because no matching predictions were found; they must be analyzed before refinement."""
    errors: tuple[tuple[str, str], ...] = ()
    """The ``(video_path, detail)`` pairs for every video that raised during detection or extraction."""

    @property
    def failed(self) -> int:
        """Returns the number of videos that raised during detection or extraction."""
        return len(self.errors)

    @property
    def successful(self) -> bool:
        """Returns whether the run completed with every video analyzed and extracted without error."""
        return not self.errors and not self.not_analyzed

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the extraction run for the CLI.

        Returns:
            A compact description of how many videos yielded outlier frames and how many frames were written.
        """
        tail = ""
        if self.not_analyzed:
            tail += f", {len(self.not_analyzed)} not analyzed"
        if self.errors:
            tail += f", {self.failed} failed"
        return (
            f"{self.algorithm} outliers: extracted {self.frames_extracted} frames from {self.extracted}/{self.total} "
            f"videos ({self.candidate_frames} candidates) on {self.workers} workers{tail}"
        )


def extract_outlier_frames_parallel(
    config_path: Path,
    videos: list[str | Path],
    *,
    shuffle: int = 1,
    training_set_index: int = 0,
    outlier_algorithm: str = "jump",
    frames_to_use: tuple[int, ...] = (),
    comparison_bodyparts: tuple[str, ...] = (),
    epsilon: float = 20.0,
    p_bound: float = 0.01,
    ar_degree: int = 3,
    ma_degree: int = 1,
    alpha: float = 0.01,
    extraction_algorithm: str = "kmeans",
    num_frames: int = -1,
    cluster_resize_width: int = 30,
    cluster_color: bool = False,
    save_labeled: bool = False,
    copy_videos: bool = False,
    destfolder: Path | None = None,
    modelprefix: str = "",
    track_method: str = "",
    snapshot_index: int | None = None,
    detector_snapshot_index: int | None = None,
    video_extensions: tuple[str, ...] = (),
    workers: int = -1,
    cores_per_worker: int = -1,
    reserve_cores: int = DEFAULT_RESERVE_CORES,
    fit_workers: int = -1,
    heartbeat: float = 30.0,
    display_progress: bool = True,
) -> OutlierExtractionSummary:
    """Flags and extracts a trained model's likely-wrong frames across many analyzed videos in parallel.

    Reads the predictions a trained model already wrote for each video (which must therefore be analyzed first),
    flags putative outlier frames with the chosen algorithm, and pulls a ``numframes2pick`` budget of them into each
    video's ``labeled-data`` directory for correction. The run has two phases: detection loads every video's
    predictions and computes its outlier candidates, fanning the ``fitting`` algorithm's per-keypoint SARIMAX fits out
    across a process pool that spans the whole run; extraction then decodes and selects the frames one video per pinned
    worker. Videos are registered in config.yaml once, single-threaded, before the extraction workers start, so the
    concurrent workers never race on the configuration file. A single bad video is recorded in the returned summary
    rather than aborting the run.

    Notes:
        The pipeline uses the spawn multiprocessing start method on every platform, so a programmatic caller must guard
        the call with ``if __name__ == "__main__":``. The installed console-script entry point is already guarded.
        Outlier extraction is additive: re-running a video appends further frames rather than replacing the existing
        ones, so coverage grows across repeated passes.

    Args:
        config_path: The path to the DeepLabCut project's config.yaml.
        videos: The video files (or directories of videos) to refine on; every video must already be analyzed.
        shuffle: The shuffle index whose trained model wrote the predictions.
        training_set_index: The training-set fraction index.
        outlier_algorithm: The detection algorithm: ``"jump"``, ``"uncertain"``, ``"fitting"``, or ``"list"``.
        frames_to_use: The explicit frame indices to extract when ``outlier_algorithm`` is ``"list"``.
        comparison_bodyparts: The bodyparts the detectors consider; an empty tuple considers every bodypart.
        epsilon: The pixel bound for the ``jump`` and ``fitting`` algorithms.
        p_bound: The likelihood bound for the ``uncertain`` algorithm and the ``fitting`` model's missing-data mask.
        ar_degree: The autoregressive degree of the ``fitting`` algorithm's SARIMAX model.
        ma_degree: The moving-average degree of the ``fitting`` algorithm's SARIMAX model.
        alpha: The significance level for the ``fitting`` algorithm's confidence interval.
        extraction_algorithm: The frame-selection algorithm applied to the candidates: ``"kmeans"`` or ``"uniform"``.
        num_frames: The number of frames to extract per video, overriding ``numframes2pick`` in config.yaml. Set to -1
            to use the value already stored in the configuration file.
        cluster_resize_width: The downsample width applied before clustering when selecting with ``"kmeans"``.
        cluster_color: Determines whether to cluster on color channels instead of grayscale.
        save_labeled: Determines whether to also save each extracted frame with the model's predictions drawn on it.
        copy_videos: Determines whether newly added videos are copied into the project rather than symlinked.
        destfolder: The directory holding the analyzed predictions, or None to look beside each video.
        modelprefix: The model subdirectory prefix, matching the trained shuffle.
        track_method: The multi-animal tracker used to generate the data (``"box"``, ``"skeleton"``, or ``"ellipse"``),
            or an empty string to read it from config.yaml.
        snapshot_index: The pose snapshot index whose scorer named the prediction files, or None for the default.
        detector_snapshot_index: The detector snapshot index, for top-down models, or None for the default.
        video_extensions: The file extensions used to filter videos found inside a supplied directory.
        workers: The number of videos to extract in parallel. Set to -1 to fill the usable cores automatically.
        cores_per_worker: The number of CPU cores pinned to each extraction worker. Set to -1 to spread them evenly.
        reserve_cores: The number of CPU cores to leave free for other tasks.
        fit_workers: The number of processes fitting SARIMAX models during ``fitting`` detection. Set to -1 to use
            every usable core.
        heartbeat: The minimum interval, in seconds, between progress lines when the output is not a TTY.
        display_progress: Determines whether to render the run header and the aggregate progress bar.

    Returns:
        An OutlierExtractionSummary describing how many videos yielded frames, how many frames were written, and which
        videos were unanalyzed or failed.

    Raises:
        FileNotFoundError: If ``config_path`` does not point to an existing file.
        ValueError: If ``outlier_algorithm`` or ``extraction_algorithm`` is unknown, if ``num_frames`` is set below one
            (other than the -1 sentinel), if ``outlier_algorithm`` is ``"list"`` without ``frames_to_use``, if the
            comparison bodyparts resolve to none, or if no videos match the requested selection.
    """
    config_path = config_path.resolve()
    if not config_path.is_file():
        message = f"Unable to extract outlier frames. The config path '{config_path}' does not point to a file."
        raise FileNotFoundError(message)
    if outlier_algorithm not in OUTLIER_ALGORITHMS:
        message = (
            f"Unable to extract outlier frames. The outlier algorithm must be one of {OUTLIER_ALGORITHMS}, but got "
            f"'{outlier_algorithm}'."
        )
        raise ValueError(message)
    if extraction_algorithm not in _EXTRACTION_ALGORITHMS:
        message = (
            f"Unable to extract outlier frames. The extraction algorithm must be one of {_EXTRACTION_ALGORITHMS}, but "
            f"got '{extraction_algorithm}'."
        )
        raise ValueError(message)
    if outlier_algorithm == "list" and not frames_to_use:
        message = "Unable to extract outlier frames. The 'list' algorithm requires an explicit list of frames to use."
        raise ValueError(message)

    _normalize_project_config(config_path, num_frames=num_frames)
    configuration = auxiliaryfunctions.read_config(str(config_path))

    bodyparts = auxiliaryfunctions.intersection_of_body_parts_and_ones_given_by_user(
        configuration, list(comparison_bodyparts) if comparison_bodyparts else "all"
    )
    if not bodyparts:
        message = "Unable to extract outlier frames. The requested comparison bodyparts matched none in the project."
        raise ValueError(message)
    resolved_track_method = auxfun_multianimal.get_track_method(configuration, track_method=track_method)
    scorer, _ = auxiliaryfunctions.get_scorer_name(
        configuration,
        shuffle,
        trainFraction=configuration["TrainingFraction"][training_set_index],
        modelprefix=modelprefix,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
    )

    video_paths = collect_video_paths(
        [str(video) for video in videos], extensions=list(video_extensions) if video_extensions else None
    )
    if not video_paths:
        message = "Unable to extract outlier frames. No videos matched the requested selection."
        raise ValueError(message)

    destination = str(destfolder) if destfolder is not None else None
    candidates, not_analyzed, errors = _detect_all_videos(
        video_paths=video_paths,
        destination=destination,
        scorer=scorer,
        configuration=configuration,
        track_method=resolved_track_method,
        bodyparts=bodyparts,
        outlier_algorithm=outlier_algorithm,
        frames_to_use=frames_to_use,
        epsilon=epsilon,
        p_bound=p_bound,
        ar_degree=ar_degree,
        ma_degree=ma_degree,
        alpha=alpha,
        fit_workers=fit_workers,
        reserve_cores=reserve_cores,
        display_progress=display_progress,
    )

    extraction_videos = [video for video in video_paths if candidates.get(video)]
    if not extraction_videos:
        return OutlierExtractionSummary(
            config=config_path,
            algorithm=outlier_algorithm,
            extraction_algorithm=extraction_algorithm,
            total=len(video_paths),
            extracted=0,
            workers=0,
            cores_used=0,
            core_count=os.cpu_count() or 1,
            candidate_frames=0,
            frames_extracted=0,
            not_analyzed=tuple(not_analyzed),
            errors=tuple(errors),
        )

    # Register every extraction video in config.yaml once, single-threaded, before the workers start. DeepLabCut's own
    # frame writer adds each video to the project, which the concurrent workers would otherwise race on; pre-adding
    # here and neutralizing the per-worker add keeps the configuration file writes serialized.
    _register_videos(
        config_path=str(config_path), configuration=configuration, videos=extraction_videos, copy_videos=copy_videos
    )

    return _extract_all_videos(
        config_path=config_path,
        videos=extraction_videos,
        candidates=candidates,
        destination=destination,
        scorer=scorer,
        track_method=resolved_track_method,
        outlier_algorithm=outlier_algorithm,
        extraction_algorithm=extraction_algorithm,
        cluster_resize_width=cluster_resize_width,
        cluster_color=cluster_color,
        save_labeled=save_labeled,
        copy_videos=copy_videos,
        workers=workers,
        cores_per_worker=cores_per_worker,
        reserve_cores=reserve_cores,
        heartbeat=heartbeat,
        display_progress=display_progress,
        total_videos=len(video_paths),
        not_analyzed=tuple(not_analyzed),
        detection_errors=errors,
    )


def _normalize_project_config(config_path: Path, *, num_frames: int) -> None:
    """Persists a normalized project_path and optional numframes2pick before any worker reads the configuration.

    DeepLabCut's read_config rewrites config.yaml whenever project_path differs from the config's own directory, which
    races against the concurrent workers' reads. The path is normalized here, single-threaded, and any per-video frame
    override is written alongside it, so every later read is a pure read.

    Args:
        config_path: The resolved path to the project's config.yaml.
        num_frames: The per-video frame count to persist as numframes2pick, or -1 to leave it untouched.

    Raises:
        ValueError: If ``num_frames`` is set below one and is not the -1 sentinel.
    """
    yaml = YAML()
    configuration = yaml.load(config_path.read_text())
    config_changed = False
    project_directory = str(config_path.parent)
    if configuration.get("project_path") != project_directory:
        configuration["project_path"] = project_directory
        config_changed = True
    if num_frames != -1:
        if num_frames < 1:
            message = (
                f"Unable to extract outlier frames. The frame count per video must be at least one, but got "
                f"{num_frames}."
            )
            raise ValueError(message)
        configuration["numframes2pick"] = int(num_frames)
        config_changed = True
    if config_changed:
        with config_path.open("w") as config_file:
            yaml.dump(configuration, config_file)


def _detect_all_videos(
    *,
    video_paths: list[str],
    destination: str | None,
    scorer: str,
    configuration: dict[str, Any],
    track_method: str,
    bodyparts: list[str],
    outlier_algorithm: str,
    frames_to_use: tuple[int, ...],
    epsilon: float,
    p_bound: float,
    ar_degree: int,
    ma_degree: int,
    alpha: float,
    fit_workers: int,
    reserve_cores: int,
    display_progress: bool,
) -> tuple[dict[str, list[int]], list[str], list[tuple[str, str]]]:
    """Computes the outlier candidate frames for every video, parallelizing the SARIMAX fits when fitting.

    Args:
        video_paths: The resolved video paths to detect outliers in.
        destination: The directory holding the predictions, or None to look beside each video.
        scorer: The DeepLabCut scorer string naming each video's prediction files.
        configuration: The loaded project configuration.
        track_method: The resolved multi-animal tracker method.
        bodyparts: The comparison bodyparts the detectors consider.
        outlier_algorithm: The detection algorithm to apply.
        frames_to_use: The explicit frames for the ``"list"`` algorithm.
        epsilon: The pixel bound for ``jump`` and ``fitting``.
        p_bound: The likelihood bound for ``uncertain`` and the ``fitting`` missing-data mask.
        ar_degree: The autoregressive degree for ``fitting``.
        ma_degree: The moving-average degree for ``fitting``.
        alpha: The significance level for ``fitting``.
        fit_workers: The number of SARIMAX fit processes, or -1 to use every usable core.
        reserve_cores: The number of cores to leave free when sizing the fit pool.
        display_progress: Whether to report the detection progress line.

    Returns:
        A tuple of the sorted, de-duplicated candidate frames keyed by video, the unanalyzed video paths, and the
        ``(video, detail)`` detection failures.
    """
    candidates: dict[str, list[int]] = {}
    fit_series: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    not_analyzed: list[str] = []
    errors: list[tuple[str, str]] = []

    for video in video_paths:
        videofolder = destination if destination is not None else str(Path(video).parents[0])
        try:
            predictions = _load_sliced_predictions(
                video=video,
                videofolder=videofolder,
                scorer=scorer,
                configuration=configuration,
                track_method=track_method,
            )
        except FileNotFoundError:
            not_analyzed.append(video)
            continue
        except Exception:  # noqa: BLE001 -- one unreadable prediction file must not abort detection for the rest.
            errors.append((video, "detection error:\n" + traceback.format_exc()))
            continue

        comparison = predictions.loc[:, predictions.columns.get_level_values("bodyparts").isin(bodyparts)]
        if outlier_algorithm == "list":
            candidates[video] = sorted({int(frame) for frame in frames_to_use})
        elif outlier_algorithm == "uncertain":
            candidates[video] = uncertain_outlier_indices(comparison, p_bound)
        elif outlier_algorithm == "jump":
            candidates[video] = jump_outlier_indices(comparison, epsilon)
        else:
            fit_series[video] = fitting_keypoint_series(comparison)

    if fit_series:
        candidates.update(
            _detect_fitting_outliers(
                fit_series=fit_series,
                num_frames_to_pick=int(configuration["numframes2pick"]),
                epsilon=epsilon,
                p_bound=p_bound,
                ar_degree=ar_degree,
                ma_degree=ma_degree,
                alpha=alpha,
                fit_workers=fit_workers,
                reserve_cores=reserve_cores,
                display_progress=display_progress,
            )
        )

    for video, indices in candidates.items():
        candidates[video] = sorted({int(index) for index in indices})
    return candidates, not_analyzed, errors


def _detect_fitting_outliers(
    *,
    fit_series: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    num_frames_to_pick: int,
    epsilon: float,
    p_bound: float,
    ar_degree: int,
    ma_degree: int,
    alpha: float,
    fit_workers: int,
    reserve_cores: int,
    display_progress: bool,
) -> dict[str, list[int]]:
    """Fits every video's per-keypoint SARIMAX models across one shared pool and reduces them to outlier frames.

    Flattening every video's keypoints into a single pool keeps all usable cores busy even when only a few videos are
    refined, which is the expensive path the fitting algorithm needs to scale on high-core machines.

    Args:
        fit_series: The per-keypoint ``(x, y, likelihood)`` trajectories for each video needing SARIMAX fits.
        num_frames_to_pick: The project's ``numframes2pick``, used to size the fallback selection.
        epsilon: The averaged-deviation bound above which a frame is flagged.
        p_bound: The likelihood below which a position is treated as missing while fitting.
        ar_degree: The autoregressive degree of the SARIMAX model.
        ma_degree: The moving-average degree of the SARIMAX model.
        alpha: The significance level for the fitted model's confidence interval.
        fit_workers: The number of fit processes, or -1 to use every usable core.
        reserve_cores: The number of cores to leave free when sizing the pool.
        display_progress: Whether to report the number of fits being run.

    Returns:
        The outlier candidate frames keyed by video.
    """
    tasks: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, int]] = []
    owners: list[str] = []
    for video, series in fit_series.items():
        for x, y, likelihood in series:
            tasks.append((x, y, likelihood, p_bound, alpha, ar_degree, ma_degree))
            owners.append(video)

    core_count = os.cpu_count() or 1
    usable = max(1, core_count - max(0, reserve_cores))
    pool_size = usable if fit_workers < 1 else fit_workers
    pool_size = max(1, min(pool_size, len(tasks)))
    if display_progress:
        sys.stderr.write(
            f"fitting {len(tasks)} keypoint trajectories across {len(fit_series)} video(s) on {pool_size} processes\n"
        )
        sys.stderr.flush()

    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=pool_size) as pool:
        distances = pool.starmap(fit_keypoint_distance, tasks)

    per_video: dict[str, list[np.ndarray]] = {video: [] for video in fit_series}
    for video, distance in zip(owners, distances, strict=True):
        per_video[video].append(distance)

    return {
        video: fitting_outlier_indices(video_distances, num_frames_to_pick=num_frames_to_pick, epsilon=epsilon)
        for video, video_distances in per_video.items()
    }


def _register_videos(
    config_path: str,
    configuration: dict[str, Any],
    videos: list[str],
    *,
    copy_videos: bool,
) -> None:
    """Adds any not-yet-registered extraction videos to config.yaml once, before the concurrent workers start.

    Args:
        config_path: The project's config.yaml path.
        configuration: The loaded project configuration, read for the already-registered videos.
        videos: The videos that will be extracted from.
        copy_videos: Whether newly added videos are copied into the project rather than symlinked.
    """
    registered = {str(Path(video).resolve()) for video in configuration.get("video_sets", {})}
    for video in videos:
        if str(Path(video).resolve()) in registered:
            continue
        with contextlib.suppress(Exception):
            dlc_outlier_frames.attempt_to_add_video(
                config=config_path, video=video, copy_videos=copy_videos, coords=None
            )


def _extract_all_videos(
    *,
    config_path: Path,
    videos: list[str],
    candidates: dict[str, list[int]],
    destination: str | None,
    scorer: str,
    track_method: str,
    outlier_algorithm: str,
    extraction_algorithm: str,
    cluster_resize_width: int,
    cluster_color: bool,
    save_labeled: bool,
    copy_videos: bool,
    workers: int,
    cores_per_worker: int,
    reserve_cores: int,
    heartbeat: float,
    display_progress: bool,
    total_videos: int,
    not_analyzed: tuple[str, ...],
    detection_errors: list[tuple[str, str]],
) -> OutlierExtractionSummary:
    """Decodes and writes the flagged frames one video per pinned worker, then assembles the run summary.

    Args:
        config_path: The resolved project config.yaml path.
        videos: The videos that have outlier candidates to extract.
        candidates: The outlier candidate frames keyed by video.
        destination: The directory holding the predictions, or None to look beside each video.
        scorer: The DeepLabCut scorer string naming each video's prediction files.
        track_method: The resolved multi-animal tracker method.
        outlier_algorithm: The detection algorithm that produced the candidates.
        extraction_algorithm: The frame-selection algorithm applied to the candidates.
        cluster_resize_width: The downsample width for k-means selection.
        cluster_color: Whether k-means selection clusters on color channels.
        save_labeled: Whether to also save each frame with the model's predictions drawn on it.
        copy_videos: Whether newly added videos are copied rather than symlinked.
        workers: The requested extraction worker count, or -1 to resolve automatically.
        cores_per_worker: The requested cores per worker, or -1 to spread them evenly.
        reserve_cores: The number of cores to leave free.
        heartbeat: The minimum interval between progress lines when the output is not a TTY.
        display_progress: Whether to render the run header and progress bar.
        total_videos: The total number of videos considered, for the summary.
        not_analyzed: The unanalyzed videos, for the summary.
        detection_errors: The detection-phase failures, extended with any extraction failures.

    Returns:
        The completed OutlierExtractionSummary.
    """
    core_count = os.cpu_count() or 1
    resolved_workers, core_sets = plan_core_allocation(
        video_count=len(videos),
        core_count=core_count,
        workers=workers,
        cores_per_worker=cores_per_worker,
        reserve_cores=reserve_cores,
    )
    cores_used = len({core for core_set in core_sets for core in core_set})
    totals = {index: max(1, len(candidates[video])) for index, video in enumerate(videos)}
    candidate_frames = sum(len(candidates[video]) for video in videos)

    context = multiprocessing.get_context("spawn")
    manager = context.Manager()
    progress_queue = manager.Queue()
    slot_queue = manager.Queue()
    for core_set in core_sets:
        slot_queue.put(core_set)

    video_indices = {video: index for index, video in enumerate(videos)}
    tasks = [
        (
            video,
            index,
            candidates[video],
            str(config_path),
            destination,
            scorer,
            track_method,
            extraction_algorithm,
            cluster_resize_width,
            cluster_color,
            save_labeled,
            copy_videos,
            progress_queue if display_progress else None,
        )
        for video, index in video_indices.items()
    ]

    if display_progress:
        _report_plan(
            video_count=len(videos),
            outlier_algorithm=outlier_algorithm,
            extraction_algorithm=extraction_algorithm,
            candidate_frames=candidate_frames,
            workers=resolved_workers,
            cores_used=cores_used,
            core_count=core_count,
            config_path=config_path,
        )
    bar = AggregateBar(progress_queue=progress_queue, total_videos=len(tasks), totals=totals, heartbeat=heartbeat)

    extracted_count = 0
    frames_extracted = 0
    errors = list(detection_errors)
    not_analyzed_list = list(not_analyzed)
    try:
        with context.Pool(processes=resolved_workers, initializer=pin_worker_to_cores, initargs=(slot_queue,)) as pool:
            if display_progress:
                bar.start()
            for video, written, status in pool.imap_unordered(_extract_one_video, tasks):
                if status == "ok":
                    extracted_count += 1
                    frames_extracted += written
                elif status == "not_analyzed":
                    not_analyzed_list.append(video)
                else:
                    errors.append((video, status))
                if display_progress:
                    progress_queue.put(("done", video_indices[video]))
    finally:
        if display_progress:
            bar.stop()
            bar.join(timeout=3)
        manager.shutdown()

    return OutlierExtractionSummary(
        config=config_path,
        algorithm=outlier_algorithm,
        extraction_algorithm=extraction_algorithm,
        total=total_videos,
        extracted=extracted_count,
        workers=resolved_workers,
        cores_used=cores_used,
        core_count=core_count,
        candidate_frames=candidate_frames,
        frames_extracted=frames_extracted,
        not_analyzed=tuple(not_analyzed_list),
        errors=tuple(errors),
    )


def _report_plan(
    video_count: int,
    outlier_algorithm: str,
    extraction_algorithm: str,
    candidate_frames: int,
    workers: int,
    cores_used: int,
    core_count: int,
    config_path: Path,
) -> None:
    """Writes the run header describing the resolved extraction plan to the standard error stream.

    Args:
        video_count: The number of videos with outlier frames to extract.
        outlier_algorithm: The detection algorithm that flagged the candidates.
        extraction_algorithm: The frame-selection algorithm applied to the candidates.
        candidate_frames: The total number of flagged candidate frames across the videos.
        workers: The resolved number of concurrent workers.
        cores_used: The number of distinct cores the workers are pinned across.
        core_count: The total number of cores on the machine.
        config_path: The resolved path to the project's config.yaml.
    """
    free_cores = core_count - cores_used
    sys.stderr.write(
        f"outlier extraction | {video_count} videos | detect={outlier_algorithm} | select={extraction_algorithm} | "
        f"{candidate_frames:,} candidate frames\n"
    )
    sys.stderr.write(
        f"workers={workers} | {cores_used}/{core_count} cores used ({free_cores} free) | config={config_path}\n"
    )
    sys.stderr.flush()


def _load_sliced_predictions(
    video: str,
    videofolder: str,
    scorer: str,
    configuration: dict[str, Any],
    track_method: str,
) -> Any:
    """Loads a video's predictions, applies the crop offset, and slices them to the configured start/stop window.

    This mirrors DeepLabCut's own preparation in ``extract_outlier_frames`` so the detected frames and the machine
    pre-labels match the upstream tool. It is used in both the detection and extraction phases against the same inputs.

    Args:
        video: The analyzed video path.
        videofolder: The directory holding the video's prediction files.
        scorer: The DeepLabCut scorer string naming the prediction files.
        configuration: The loaded project configuration, read for the start/stop bounds and the video's crop margins.
        track_method: The resolved multi-animal tracker method.

    Returns:
        The prediction table, offset-corrected and sliced to the start/stop window.

    Raises:
        FileNotFoundError: If the video has no matching prediction or metadata files.
    """
    vname = Path(video).stem
    predictions, _, _, _ = auxiliaryfunctions.load_analyzed_data(videofolder, vname, scorer, track_method=track_method)
    metadata = auxiliaryfunctions.load_video_metadata(videofolder, vname, scorer)
    frame_count = len(predictions)
    start_index = max(int(np.floor(frame_count * configuration["start"])), 0)
    stop_index = min(int(np.ceil(frame_count * configuration["stop"])), frame_count)
    window = np.arange(stop_index - start_index) + start_index

    out_x1, out_y1 = _video_cropping_offset(configuration, video)
    if metadata.get("data", {}).get("cropping"):
        x1, _, y1, _ = metadata["data"]["cropping_parameters"]
        predictions.iloc[:, predictions.columns.get_level_values(level="coords") == "x"] += x1 - out_x1
        predictions.iloc[:, predictions.columns.get_level_values(level="coords") == "y"] += y1 - out_y1
    return predictions.iloc[window]


def _video_cropping_offset(configuration: dict[str, Any], video: str) -> tuple[int, int]:
    """Reads the top-left crop origin config.yaml records for a video, used to undo output cropping on the predictions.

    Args:
        configuration: The loaded project configuration holding the ``video_sets`` crop specifications.
        video: The video path whose crop origin is read.

    Returns:
        The ``(x1, y1)`` crop origin, or ``(0, 0)`` when the video is uncropped or unregistered.

    Raises:
        ValueError: If the video's crop specification is not four comma-separated integers.
    """
    crop = configuration.get("video_sets", {}).get(str(video), {}).get("crop")
    if crop is None:
        return 0, 0
    parts = [part.strip() for part in str(crop).split(",")]
    if len(parts) != _CROP_FIELD_COUNT:
        message = (
            f"Unable to read the crop for video '{video}'. Expected four comma-separated integers 'x1,x2,y1,y2', but "
            f"got '{crop}'."
        )
        raise ValueError(message)
    x1, _, y1, _ = (int(part) for part in parts)
    return x1, y1


def _extract_one_video(task: tuple[Any, ...]) -> tuple[str, int, str]:
    """Selects and writes the flagged frames for a single video, reusing DeepLabCut's own frame writer.

    DeepLabCut's console output is silenced and exceptions are captured, so one bad video cannot kill the worker pool.
    DeepLabCut's frame writer re-registers the video in config.yaml; that add is neutralized here because the pipeline
    already registered every video single-threaded, so the concurrent workers never write the configuration file.

    Args:
        task: The packed work item carrying the video path, the video index, the candidate frames, the config path,
            the prediction directory, the scorer, the tracker method, the selection settings, and the progress queue.

    Returns:
        A tuple of the video path, the number of frames freshly written, and a status string (``"ok"``,
        ``"not_analyzed"``, or an ``"error:"`` traceback).
    """
    (
        video,
        video_index,
        indices,
        config_path,
        destination,
        scorer,
        track_method,
        extraction_algorithm,
        cluster_resize_width,
        cluster_color,
        save_labeled,
        copy_videos,
        progress_queue,
    ) = task
    try:
        cv2.setNumThreads(1)
        configuration = auxiliaryfunctions.read_config(config_path)
        videofolder = destination if destination is not None else str(Path(video).parents[0])
        predictions = _load_sliced_predictions(
            video=video,
            videofolder=videofolder,
            scorer=scorer,
            configuration=configuration,
            track_method=track_method,
        )

        output_directory = Path(configuration["project_path"]) / "labeled-data" / Path(video).stem
        before = _count_extracted_frames(output_directory)

        # Route DeepLabCut's frame-reading progress to the parent's aggregate bar and stop its per-worker config write.
        # The queue is None when progress is disabled, leaving DeepLabCut's own (stream-suppressed) tqdm in place.
        if progress_queue is not None:
            frameselectiontools.tqdm = make_progress_reporter(
                progress_queue=progress_queue, video_index=video_index, total=max(1, len(indices))
            )
        dlc_outlier_frames.attempt_to_add_video = _skip_video_registration

        with (
            Path(os.devnull).open("w") as null_stream,
            contextlib.redirect_stdout(null_stream),
            contextlib.redirect_stderr(null_stream),
        ):
            dlc_outlier_frames.ExtractFramesbasedonPreselection(
                indices,
                extraction_algorithm,
                predictions,
                video,
                configuration,
                config_path,
                opencv=True,
                cluster_resizewidth=cluster_resize_width,
                cluster_color=cluster_color,
                savelabeled=save_labeled,
                with_annotations=True,
                copy_videos=copy_videos,
            )
    except FileNotFoundError:
        return video, 0, "not_analyzed"
    except Exception:  # noqa: BLE001 -- one bad video must not kill the pool; the traceback is returned as status.
        return video, 0, "error:\n" + traceback.format_exc()
    else:
        written = _count_extracted_frames(output_directory) - before
        return video, max(0, written), "ok"


def _count_extracted_frames(output_directory: Path) -> int:
    """Counts the extracted image frames in a labeled-data directory, ignoring the predicted-label overlays.

    Args:
        output_directory: The ``labeled-data/<video>`` directory whose extracted frames are counted.

    Returns:
        The number of ``img*.png`` frames that are not ``*labeled.png`` prediction overlays.
    """
    if not output_directory.exists():
        return 0
    return sum(1 for frame in output_directory.glob("img*.png") if not frame.stem.endswith("labeled"))


def _skip_video_registration(**_kwargs: Any) -> bool:
    """Neutralizes DeepLabCut's per-video config.yaml registration inside a worker; the pipeline registers up front.

    Returns:
        True, reporting a successful registration so DeepLabCut's frame writer proceeds unchanged.
    """
    return True
