"""Provides the multi-device inference pipeline that runs DeepLabCut over many videos across worker slots."""

import os
import sys
import pickle
from typing import Any
from pathlib import Path
import contextlib
from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Iterator, Sequence

import cv2
import numpy as np
import psutil
from numpy.typing import NDArray
import torch.multiprocessing as mp
from deeplabcut.utils.auxiliaryfunctions import read_config, read_plainconfig
from deeplabcut.pose_estimation_pytorch.apis import videos as dlc_videos
from deeplabcut.pose_estimation_pytorch.data import DLCLoader
from deeplabcut.pose_estimation_pytorch.task import Task
import deeplabcut.pose_estimation_pytorch.apis.utils as dlc_apis_utils
from deeplabcut.pose_estimation_pytorch.apis.videos import (
    VideoIterator,
    video_inference,
    _generate_metadata,
    _generate_output_data,
    create_df_from_prediction,
)

from .runners import patch_dlc_runner_builders
from .optimization import InferenceProfile, apply_runtime_optimizations
from ..frame_extraction import AggregateBar, plan_core_allocation, make_progress_reporter

_STOCK_ACCELERATION_DISABLED: dict[str, dict[str, bool]] = {
    "autocast": {"enabled": False},
    "compile": {"enabled": False},
}
"""The inference-config overrides passed to analyze_videos: the runner wrappers own autocast and compilation, so
DeepLabCut's own autocast and compile are disabled while its async decode pipeline is left at its default."""

_CROP_FIELD_COUNT: int = 4
"""The number of comma-separated integers, ``x1,x2,y1,y2``, in a video's config.yaml crop specification."""

_RESULT_POLL_TIMEOUT_SECONDS: float = 5.0
"""The per-result wait when draining the worker results queue. A worker that misses this window is treated as dead."""

_BAR_JOIN_TIMEOUT_SECONDS: int = 3
"""The grace period to let the aggregate progress-bar thread finish its final render before the run returns."""


@dataclass(frozen=True, slots=True)
class InferenceSummary:
    """Captures the outcome of a completed multi-video inference run for reporting to the caller.

    Notes:
        The summary is built after every worker has finished. ``outputs`` holds the produced DeepLabCut ``.h5``
        prediction files in video order, and ``failures`` pairs each failed video with its error message so a partial
        run is reported honestly rather than silently.
    """

    config: Path
    """The path of the DeepLabCut project configuration file inference ran for."""
    video_count: int
    """The number of videos submitted for inference."""
    destinations: tuple[Path, ...] | None
    """The distinct directories the prediction files were written to, or None when each video's predictions were
    written beside the video itself."""
    device: str
    """The base device type inference ran on (``"cuda"``, ``"cpu"``, or ``"mps"``)."""
    workers: int
    """The number of worker processes used."""
    precision: str
    """The compute precision used (``"bfloat16"``, ``"float16"``, or ``"fp32"``)."""
    outputs: tuple[Path, ...]
    """The produced DeepLabCut ``.h5`` prediction files, one per successfully processed video."""
    failures: tuple[tuple[str, str], ...]
    """The videos that failed, each paired with its error message."""

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the inference run for the CLI.

        Returns:
            A compact description of how many videos were processed, on what hardware, and where results were written.
        """
        ok = len(self.outputs)
        where = f"{self.device} x{self.workers}"
        tail = f", {len(self.failures)} failed" if self.failures else ""
        if self.destinations is None:
            written_to: Path | str = "each video's directory"
        elif len(self.destinations) == 1:
            written_to = self.destinations[0]
        else:
            written_to = f"{len(self.destinations)} per-video directories"
        return f"analyzed {ok}/{self.video_count} videos on {where} in {self.precision} -> h5 in {written_to}{tail}"


@dataclass(frozen=True, slots=True)
class _Slot:
    """Describes one worker's placement: a device and, for CPU workers, the physical cores it is pinned to."""

    device: str
    """The device string this worker runs on, for example ``"cuda:0"`` or ``"cpu"``."""
    cores: tuple[int, ...] | None
    """The CPU core ids this worker is pinned to, or None for GPU and MPS workers."""


@dataclass(frozen=True, slots=True)
class _InferenceLaunch:
    """Bundles the picklable per-run parameters shared by every inference worker process."""

    config: Path
    """The path of the DeepLabCut project configuration file inference runs for."""
    shuffle: int
    """The shuffle index whose trained model is used."""
    snapshot_index: int | None
    """The pose snapshot index to use, or None for the configured default."""
    detector_snapshot_index: int | None
    """The detector snapshot index to use, or None for the configured default."""
    profile: InferenceProfile
    """The resolved optimization profile describing the device, precision, and parallelism to use."""
    batch_size: int | None
    """The pose-model inference batch size, or None to use the configured value."""
    detector_batch_size: int | None
    """The detector inference batch size, or None to use the configured value."""
    display_progress: bool
    """Determines whether the live aggregate progress bar is rendered."""
    video_queue: Any
    """The shared queue each worker pulls per-video work items from."""
    progress_queue: Any
    """The shared queue workers publish per-video progress updates to."""
    results_queue: Any
    """The shared queue workers report per-video results to."""


@dataclass(frozen=True, slots=True)
class _ChunkItem:
    """Describes one contiguous frame range of a video analyzed as an independent chunk-worker task."""

    task_id: int
    """The globally unique identifier of this chunk, used to key its progress and gather its result."""
    video_index: int
    """The index of the video this chunk belongs to, used to group chunks back into one prediction file."""
    chunk_index: int
    """The position of this chunk within its video, used to order chunk predictions by ascending frame."""
    video: str
    """The path of the video this chunk reads its frame range from."""
    frame_start: int
    """The inclusive index of the first frame this chunk analyzes."""
    frame_end: int
    """The exclusive index one past the last frame this chunk analyzes."""
    crop: list[int] | None
    """The ``[x1, x2, y1, y2]`` region every chunk of this video analyzes, or None to analyze the full frame."""
    destination: str | None
    """The directory this video's stitched prediction file is written to, or None to write beside the video."""


@dataclass(frozen=True, slots=True)
class _AnalysisPlan:
    """Holds the project configuration a chunked run resolves once, in the parent, to stitch prediction files."""

    scorer: str
    """The DeepLabCut scorer string that names each video's output files."""
    project_cfg: dict[str, Any]
    """The DeepLabCut project configuration read from the project's config.yaml."""
    model_cfg: dict[str, Any]
    """The trained model's pytorch configuration."""
    pose_cfg: dict[str, Any]
    """The pose configuration read from the shuffle's test directory, used to assemble the full-pickle output."""
    train_fraction: float
    """The training-set fraction the analyzed shuffle was trained with, recorded in the prediction metadata."""
    batch_size: int
    """The pose-model batch size recorded in the prediction metadata."""
    multi_animal: bool
    """Determines whether the project is multi-animal, which the single-file chunk-stitch path does not support."""
    pose_task: Any
    """The DeepLabCut pose task the shuffle uses, which the chunk-stitch path supports only when it is bottom-up."""


def resolve_project_videos(config: str | Path) -> list[Path]:
    """Returns the existing video files registered in the project configuration's ``video_sets``.

    Reads the paths the DeepLabCut project configuration registers and keeps the ones that still exist on disk, so
    inference can analyze the whole project without re-listing every video on the command line.

    Args:
        config: The path of the DeepLabCut project configuration file.

    Returns:
        The registered video paths that currently exist on disk, in the order the configuration lists them.
    """
    project = read_config(str(Path(config)))
    registered = project.get("video_sets") or {}
    videos = [Path(entry) for entry in registered]
    return [video for video in videos if video.exists()]


def detect_fixed_input_size(
    config: str | Path,
    videos: list[str | Path],
    crop_override: Sequence[tuple[int, int, int, int]] | None = None,
) -> bool:
    """Determines whether every video would feed the pose network a single fixed input resolution.

    The cuDNN autotuner only pays off when the convolution input shapes stay constant across the run, so this reports
    whether that precondition holds instead of asking the operator to assert it. When per-video crop rectangles are
    provided (the ``--crop`` override), each video is reduced to its rectangle, so the run is fixed-size exactly when
    all rectangles share one size. Otherwise, when the project is configured to crop, every analyzed video is reduced
    to its configured crop rectangle, so the run is fixed-size exactly when all videos share one crop size. When the
    project is not configured to crop, the network sees each video's native resolution, so the run is fixed-size
    exactly when all videos share one resolution. A single video is therefore always fixed-size. Any inability to read
    the configuration or a video's dimensions is treated conservatively as not fixed, since a wrong assertion of fixed
    size makes the autotuner harmful.

    Args:
        config: The path of the DeepLabCut project configuration file.
        videos: The video files the run will analyze.
        crop_override: The per-video crop rectangles that override the project configuration, parallel to ``videos``,
            or None to derive each video's input size from the project configuration.

    Returns:
        True when the network's spatial input size is provably constant across the whole run, False otherwise.
    """
    video_paths = [Path(video) for video in videos]
    if not video_paths:
        return False
    if crop_override is not None:
        sizes = {(x2 - x1, y2 - y1) for x1, x2, y1, y2 in crop_override}
        return len(sizes) == 1
    try:
        project_config = read_config(str(Path(config)))
        sizes_or_none = {_resolve_input_size(project_config=project_config, video=video) for video in video_paths}
    except Exception:
        return False
    return None not in sizes_or_none and len(sizes_or_none) == 1


def run_inference(
    config: str | Path,
    videos: list[str | Path],
    profile: InferenceProfile,
    *,
    destination_override: Sequence[str | Path] | None = None,
    shuffle: int = 1,
    snapshot_index: int | None = None,
    detector_snapshot_index: int | None = None,
    batch_size: int | None = None,
    detector_batch_size: int | None = None,
    crop_override: Sequence[tuple[int, int, int, int]] | None = None,
    display_progress: bool = True,
) -> InferenceSummary:
    """Runs DeepLabCut inference over many videos, distributing whole videos across GPU or CPU worker slots.

    Each worker pulls whole videos from a shared queue and analyzes them with DeepLabCut, so the work is balanced
    across slots without splitting any video. On CUDA a slot is a device (``gpu_processes`` of them per device); on CPU
    a slot is a disjoint, thread-bounded block of physical cores. Every worker's forward pass is wrapped with the
    profile's mixed precision and channels-last format, and each video's predictions are written as DeepLabCut's native
    ``.h5`` prediction file, beside the video or into its chosen output directory.

    Args:
        config: The path of the DeepLabCut project configuration file.
        videos: The video files to analyze.
        profile: The resolved optimization profile describing the device, precision, and parallelism to use.
        destination_override: The per-video output directories prediction files are written to, parallel to ``videos``,
            or None to write each video's predictions beside the video itself, matching DeepLabCut's own default and the
            location the outlier-extraction step reads. Pass one directory per video, so a single chosen directory can
            collect every video's predictions or each video's predictions can be bundled with its own directory.
        shuffle: The shuffle index whose trained model is used.
        snapshot_index: The pose snapshot index to use, or None for the configured default.
        detector_snapshot_index: The detector snapshot index to use, or None for the configured default.
        batch_size: The pose-model inference batch size, or None to use the configured value.
        detector_batch_size: The detector inference batch size, or None to use the configured value.
        crop_override: The per-video crop rectangles to analyze, parallel to ``videos``, each an ``(x1, x2, y1, y2)``
            tuple, or None to resolve each video's crop from the project configuration. Overrides the project's
            cropping so de-novo videos that are not registered in the configuration can be analyzed at a caller-chosen
            crop.
        display_progress: Determines whether to render the live aggregate progress bar.

    Returns:
        A summary of what was analyzed and the hardware configuration used.

    Raises:
        ValueError: Raised when no videos are provided, or when ``crop_override`` or ``destination_override`` is
            provided but its length does not match the number of videos. Raised when the profile selects CUDA but
            resolves no GPU indices to build worker slots from. Raised when an explicit CPU worker/thread configuration
            cannot be pinned to disjoint core blocks. Raised when ``profile.chunks`` exceeds one and the project is
            multi-animal or the model is not bottom-up.
    """
    config = Path(config)
    video_paths = [Path(video) for video in videos]
    if not video_paths:
        message = "Unable to run inference. Expected at least one video, but got an empty video list."
        raise ValueError(message)
    if crop_override is not None and len(crop_override) != len(video_paths):
        message = (
            f"Unable to run inference. Expected one crop rectangle per video, but got {len(crop_override)} crop "
            f"rectangles for {len(video_paths)} videos."
        )
        raise ValueError(message)
    if destination_override is not None and len(destination_override) != len(video_paths):
        message = (
            f"Unable to run inference. Expected one output directory per video, but got {len(destination_override)} "
            f"output directories for {len(video_paths)} videos."
        )
        raise ValueError(message)
    destinations = (
        tuple(dict.fromkeys(Path(directory) for directory in destination_override))
        if destination_override is not None
        else None
    )
    if destinations is not None:
        for directory in destinations:
            directory.mkdir(parents=True, exist_ok=True)

    # A chunked run splits every video into parallel frame ranges and stitches the predictions in the parent. The
    # default single-chunk run keeps the whole-video DeepLabCut path below unchanged.
    if profile.chunks > 1:
        return _run_inference_chunked(
            config=config,
            video_paths=video_paths,
            profile=profile,
            destinations=destinations,
            crop_override=crop_override,
            destination_override=destination_override,
            shuffle=shuffle,
            snapshot_index=snapshot_index,
            detector_snapshot_index=detector_snapshot_index,
            batch_size=batch_size,
            detector_batch_size=detector_batch_size,
            display_progress=display_progress,
        )

    totals = {index: _probe_frame_count(video) for index, video in enumerate(video_paths)}
    slots = _build_slots(profile=profile, video_count=len(video_paths))

    # Resolves each video's crop once, in the parent, so every worker analyzes the same region the frames were
    # extracted from. A caller-supplied override takes precedence over the project configuration, letting de-novo
    # videos be analyzed at a chosen crop; otherwise the crop is resolved from the project's cropping configuration.
    project_config = read_config(str(config))

    manager = mp.Manager()
    video_queue = manager.Queue()
    progress_queue = manager.Queue()
    results_queue = manager.Queue()
    for index, video in enumerate(video_paths):
        if crop_override is not None:
            crop: list[int] | None = list(crop_override[index])
        else:
            crop = _resolve_video_cropping(project_config=project_config, video=str(video))
        video_destination = str(destination_override[index]) if destination_override is not None else None
        video_queue.put((index, str(video), totals[index], crop, video_destination))
    for _ in slots:
        video_queue.put(None)

    bar = None
    if display_progress:
        bar = AggregateBar(
            progress_queue=progress_queue,
            total_video_count=len(video_paths),
            frame_totals=totals,
            preparing_label="compiling model...",
        )
        bar.start()

    launch = _InferenceLaunch(
        config=config,
        shuffle=shuffle,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
        profile=profile,
        batch_size=batch_size,
        detector_batch_size=detector_batch_size,
        display_progress=display_progress,
        video_queue=video_queue,
        progress_queue=progress_queue,
        results_queue=results_queue,
    )

    context = mp.get_context("spawn")
    processes = [context.Process(target=_run_inference_worker, args=(slot, launch)) for slot in slots]
    outputs: dict[int, Path] = {}
    failures: list[tuple[str, str]] = []
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        outputs, failures = _collect_results(results_queue=results_queue, video_paths=video_paths)
    finally:
        if bar is not None:
            bar.stop()
            bar.join(timeout=_BAR_JOIN_TIMEOUT_SECONDS)
        manager.shutdown()

    precision = _describe_precision(profile)
    return InferenceSummary(
        config=config,
        video_count=len(video_paths),
        destinations=destinations,
        device=profile.device,
        workers=len(slots),
        precision=precision,
        outputs=tuple(outputs[index] for index in sorted(outputs)),
        failures=tuple(failures),
    )


def _resolve_input_size(project_config: dict[str, Any], video: Path) -> tuple[int, int] | None:
    """Resolves the spatial size the pose network receives for a video, before padding, or None when unknown.

    Args:
        project_config: The loaded DeepLabCut project configuration.
        video: The path of the video about to be analyzed.

    Returns:
        The ``(width, height)`` the network sees, taken from the project crop rectangle when cropping is configured or
        the video's native resolution otherwise, or None when the crop or the video's dimensions cannot be resolved.
    """
    if project_config.get("cropping", False):
        crop = _resolve_video_cropping(project_config=project_config, video=str(video))
        if crop is None:
            return None
        x1, x2, y1, y2 = crop
        return x2 - x1, y2 - y1
    return _probe_frame_size(video)


def _probe_frame_size(video: Path) -> tuple[int, int] | None:
    """Reads a video's frame dimensions from its container header.

    Args:
        video: The path of the video whose frame dimensions are read.

    Returns:
        The ``(width, height)`` reported by the container, or None when the video cannot be opened or reports a
        non-positive dimension.
    """
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        return None
    return width, height


def _parse_crop(crop: str | None) -> list[int] | None:
    """Parses a ``"x1, x2, y1, y2"`` crop specification into four integers, or None when it is absent or malformed.

    Args:
        crop: The crop specification stored in the project configuration, a comma-separated string of four integers.

    Returns:
        The ``[x1, x2, y1, y2]`` crop as integers, or None when the value is missing or not four integers.
    """
    if crop is None:
        return None
    parts = [part.strip() for part in str(crop).split(",")]
    if len(parts) != _CROP_FIELD_COUNT:
        return None
    try:
        return [int(part) for part in parts]
    except ValueError:
        return None


def _resolve_video_cropping(project_config: dict[str, Any], video: str) -> list[int] | None:
    """Resolves the crop rectangle a video must be analyzed with, honoring the project's cropping configuration.

    When the project is configured to crop, inference must analyze the same region a video's frames are extracted
    from, so predictions land in the cropped coordinate space the model was trained on. The video's own registered
    crop is used when present, falling back to the project-wide rectangle for a video that is not yet registered.
    When the project is not configured to crop, the full frame is analyzed.

    Args:
        project_config: The loaded DeepLabCut project configuration.
        video: The path of the video about to be analyzed.

    Returns:
        The ``[x1, x2, y1, y2]`` crop rectangle to analyze, or None to analyze the full frame.
    """
    if not project_config.get("cropping", False):
        return None
    target = Path(video).resolve()
    for registered, metadata in (project_config.get("video_sets") or {}).items():
        if not isinstance(metadata, dict) or Path(registered).resolve() != target:
            continue
        crop = _parse_crop(metadata.get("crop"))
        if crop is not None:
            return crop
    corners: list[int] = []
    for key in ("x1", "x2", "y1", "y2"):
        corner = project_config.get(key)
        if corner is None:
            return None
        corners.append(int(corner))
    return corners


def _describe_precision(profile: InferenceProfile) -> str:
    """Returns the human-readable precision label for the profile.

    Args:
        profile: The resolved optimization profile.

    Returns:
        The precision label (``"bfloat16"``, ``"float16"``, or ``"fp32"``).
    """
    return str(profile.amp_dtype).removeprefix("torch.") if profile.amp_dtype is not None else "fp32"


def _build_slots(profile: InferenceProfile, video_count: int, *, chunks: int = 1) -> list[_Slot]:
    """Builds the worker slots for the run from the profile and the number of work units.

    Args:
        profile: The resolved optimization profile.
        video_count: The number of work units to process, used to avoid spawning more workers than there is work.
        chunks: The per-video frame-range piece count, which multiplies the per-device worker concurrency. One leaves
            the concurrency unchanged, so the whole-video path builds one slot per configured process.

    Returns:
        The list of worker slots to spawn.
    """
    if profile.on_cuda:
        # Round-robins the device order (cuda:0, cuda:1, cuda:0, ...) rather than grouping by device
        # (cuda:0, cuda:0, cuda:1, ...). When there are fewer work units than slots, the list is truncated below, so
        # this ordering spreads the surviving workers across every GPU before oversubscribing any single one.
        slots = [
            _Slot(device=f"cuda:{index}", cores=None)
            for _ in range(profile.gpu_processes * chunks)
            for index in profile.gpus
        ]
    elif profile.device == "cpu":
        core_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        # An explicit CPU worker count is multiplied by the chunk factor so chunking raises CPU concurrency too, while
        # the automatic -1 request is left untouched for the allocator to size from the core budget.
        worker_count = profile.cpu_workers * chunks if profile.cpu_workers >= 1 else profile.cpu_workers
        _workers, core_sets = plan_core_allocation(
            video_count=video_count,
            total_core_count=core_count,
            worker_count=worker_count,
            cores_per_worker=profile.cpu_threads_per_worker or -1,
            reserved_core_count=core_count - _usable_cpu_cores(profile),
        )
        slots = [_Slot(device="cpu", cores=tuple(sorted(cores))) for cores in core_sets]
    else:
        slots = [_Slot(device=profile.device, cores=None) for _ in range(chunks)]

    if not slots:
        message = (
            "Unable to build inference worker slots. The profile selected the CUDA device but no GPU indices were "
            "resolved. Pass a device or an explicit GPU list that selects at least one device."
        )
        raise ValueError(message)

    limit = max(1, min(len(slots), video_count))
    return slots[:limit]


def _partition_frame_ranges(total_frames: int, chunks: int) -> list[tuple[int, int]]:
    """Splits ``[0, total_frames)`` into up to ``chunks`` contiguous, balanced half-open frame ranges.

    The frames are divided as evenly as possible, with the earliest ranges taking the one-frame remainder so every
    frame is covered exactly once. The range count is clamped to the frame count, so a video split into more chunks
    than it has frames yields one single-frame range per frame rather than any empty range.

    Args:
        total_frames: The number of frames in the video, at least one.
        chunks: The requested number of frame-range pieces.

    Returns:
        The list of contiguous ``(start, end)`` ranges covering ``[0, total_frames)`` in ascending frame order.
    """
    pieces = max(1, min(chunks, total_frames))
    base, remainder = divmod(total_frames, pieces)
    ranges: list[tuple[int, int]] = []
    start = 0
    for piece in range(pieces):
        length = base + (1 if piece < remainder else 0)
        ranges.append((start, start + length))
        start += length
    return ranges


def _usable_cpu_cores(profile: InferenceProfile) -> int:
    """Returns the number of physical cores the CPU workers collectively occupy, for the core-block reservation.

    Args:
        profile: The resolved optimization profile.

    Returns:
        The product of the CPU worker count and per-worker thread count, bounded by the physical core count.
    """
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    threads = profile.cpu_threads_per_worker or 1
    return min(physical, max(1, profile.cpu_workers) * threads)


def _probe_frame_count(video: Path) -> int:
    """Reads a video's frame count from its container header for the aggregate progress bar.

    Args:
        video: The path of the video whose frame count is read.

    Returns:
        The reported frame count, clamped to at least one so the progress bar always has a positive total.
    """
    capture = cv2.VideoCapture(str(video))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    finally:
        capture.release()
    return max(1, frames)


def _collect_results(
    results_queue: Any,
    video_paths: list[Path],
) -> tuple[dict[int, Path], list[tuple[str, str]]]:
    """Drains the results queue into successful outputs and failures, keyed and ordered by video index.

    Args:
        results_queue: The shared queue workers report per-video results to.
        video_paths: The submitted videos, used to name failures and bound the number of results awaited.

    Returns:
        A tuple of the successful outputs keyed by video index and the list of failures as (video-name, error) pairs.
    """
    outputs: dict[int, Path] = {}
    failures: list[tuple[str, str]] = []
    reported: set[int] = set()
    for _ in video_paths:
        try:
            index, path, error = results_queue.get(timeout=_RESULT_POLL_TIMEOUT_SECONDS)
        except Exception:
            break
        reported.add(index)
        if error is None and path is not None:
            outputs[index] = Path(path)
        else:
            failures.append((video_paths[index].name, error or "no prediction file was produced"))

    for index, video in enumerate(video_paths):
        if index not in reported:
            failures.append((video.name, "the worker process exited before reporting a result"))
    return outputs, failures


@contextlib.contextmanager
def _suppress_stdout(*, active: bool) -> Iterator[None]:
    """Redirects standard output to the null device while active, keeping DeepLabCut worker chatter off the console.

    Args:
        active: Determines whether to suppress standard output; when False the context does nothing.

    Yields:
        None, for the duration of the suppression.
    """
    if not active:
        yield
        return
    with Path(os.devnull).open("w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
        yield


def _run_inference_worker(slot: _Slot, launch: _InferenceLaunch) -> None:
    """Runs inference for one worker slot, analyzing whole videos pulled from the shared queue until it is drained.

    Args:
        slot: The device and optional CPU-core placement for this worker.
        launch: The bundle of picklable per-run parameters shared by every worker process.
    """
    if slot.cores is not None and sys.platform != "darwin":
        with contextlib.suppress(Exception):
            psutil.Process().cpu_affinity(list(slot.cores))
    apply_runtime_optimizations(launch.profile)

    with patch_dlc_runner_builders(launch.profile):
        while True:
            item = launch.video_queue.get()
            if item is None:
                break
            _analyze_one_video(slot=slot, launch=launch, item=item)


def _analyze_one_video(
    slot: _Slot, launch: _InferenceLaunch, item: tuple[int, str, int, list[int] | None, str | None]
) -> None:
    """Analyzes a single video from the queue and reports its prediction file and progress.

    Args:
        slot: The device and optional CPU-core placement for this worker.
        launch: The bundle of picklable per-run parameters.
        item: The (video index, video path, frame total, crop rectangle, output directory) work item pulled from the
            queue. The crop rectangle is the ``[x1, x2, y1, y2]`` region to analyze, or None to analyze the full frame.
            The output directory is where this video's predictions are written, or None to write beside the video.
    """
    index, video, total, cropping, destination = item
    output_directory = Path(destination) if destination is not None else Path(video).parent
    original_tqdm = dlc_videos.tqdm
    if launch.display_progress:
        dlc_videos.tqdm = make_progress_reporter(
            progress_queue=launch.progress_queue, video_index=index, frame_total=total
        )
    try:
        with _suppress_stdout(active=launch.display_progress):
            scorer = dlc_videos.analyze_videos(
                config=str(launch.config),
                videos=[video],
                shuffle=launch.shuffle,
                device=slot.device,
                destfolder=str(output_directory),
                # Analyzes the same region the frames were extracted from; None analyzes the full frame.
                cropping=cropping,
                snapshot_index=launch.snapshot_index,
                detector_snapshot_index=launch.detector_snapshot_index,
                batch_size=launch.batch_size,
                detector_batch_size=launch.detector_batch_size,
                # Always (re)analyze the submitted video. DeepLabCut otherwise skips any video whose companion
                # ``_full.pickle`` already exists, which would silently skip re-runs when a batch is resubmitted.
                overwrite=True,
                inference_cfg=_STOCK_ACCELERATION_DISABLED,
            )
        output = _resolve_output(video=video, scorer=scorer, destination=output_directory)
        launch.results_queue.put((index, str(output) if output is not None else None, None))
    except Exception as error:
        launch.results_queue.put((index, None, f"{type(error).__name__}: {error}"))
    finally:
        dlc_videos.tqdm = original_tqdm
        if launch.display_progress:
            with contextlib.suppress(Exception):
                launch.progress_queue.put(("done", index))


def _resolve_output(video: str, scorer: str, destination: Path) -> Path | None:
    """Locates the DeepLabCut native ``.h5`` prediction file written for one analyzed video.

    Args:
        video: The analyzed video path.
        scorer: The DeepLabCut scorer string returned by ``analyze_videos``, which names the output file.
        destination: The directory this video's predictions were written to, which is the video's own directory when
            the run configured no destination.

    Returns:
        The produced ``.h5`` file, or None when no prediction file was written.
    """
    stem = Path(video).stem
    exact = destination / f"{stem}{scorer}.h5"
    if exact.exists():
        return exact
    # Multi-animal auto-tracking writes a tracker-suffixed file instead of the plain per-frame table.
    matches = sorted(destination.glob(f"{stem}{scorer}*.h5"))
    return matches[-1] if matches else None


def _run_inference_chunked(
    config: Path,
    video_paths: list[Path],
    profile: InferenceProfile,
    *,
    destinations: tuple[Path, ...] | None,
    crop_override: Sequence[tuple[int, int, int, int]] | None,
    destination_override: Sequence[str | Path] | None,
    shuffle: int,
    snapshot_index: int | None,
    detector_snapshot_index: int | None,
    batch_size: int | None,
    detector_batch_size: int | None,
    display_progress: bool,
) -> InferenceSummary:
    """Runs inference by splitting each video into parallel frame-range chunks and stitching predictions in the parent.

    Every video is divided into ``profile.chunks`` contiguous frame ranges, each analyzed by its own worker so several
    ranges of one video run concurrently on a device. Each worker returns its raw per-frame predictions rather than
    writing a file. Once every chunk of a video reports, the parent concatenates the chunks in frame order and writes
    the single prediction file the whole-video path would have written, preserving the beside-the-video or ``--output``
    destination.

    Args:
        config: The path of the DeepLabCut project configuration file.
        video_paths: The videos to analyze.
        profile: The resolved optimization profile, whose ``chunks`` field sets the per-video split count.
        destinations: The deduplicated output directories, recorded on the returned summary.
        crop_override: The per-video crop rectangles, or None to resolve each crop from the project configuration.
        destination_override: The per-video output directories, or None to write beside each video.
        shuffle: The shuffle index whose trained model is used.
        snapshot_index: The pose snapshot index to use, or None for the configured default.
        detector_snapshot_index: The detector snapshot index to use, or None for the configured default.
        batch_size: The pose-model batch size, or None to use the configured value.
        detector_batch_size: The detector batch size, or None to use the configured value.
        display_progress: Determines whether to render the live aggregate progress bar.

    Returns:
        A summary of what was analyzed and the hardware configuration used.

    Raises:
        ValueError: When the project is multi-animal or the model is not bottom-up, which the single-file chunk-stitch
            path does not support.
    """
    plan = _build_analysis_plan(
        config=config,
        shuffle=shuffle,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
        batch_size=batch_size,
    )
    if plan.multi_animal or plan.pose_task != Task.BOTTOM_UP:
        message = (
            "Unable to run chunked inference on this project. Chunking splits a video into frame ranges and stitches "
            "per-frame predictions, which supports only single-animal bottom-up models, not multi-animal or top-down "
            "models. Rerun with --chunks 1."
        )
        raise ValueError(message)

    project_config = read_config(str(config))
    totals = {index: _probe_frame_count(video) for index, video in enumerate(video_paths)}

    work_items: list[_ChunkItem] = []
    chunk_frame_totals: dict[int, int] = {}
    key_video: dict[int, int] = {}
    task_id = 0
    for index, video in enumerate(video_paths):
        if crop_override is not None:
            crop: list[int] | None = list(crop_override[index])
        else:
            crop = _resolve_video_cropping(project_config=project_config, video=str(video))
        destination = str(destination_override[index]) if destination_override is not None else None
        ranges = _partition_frame_ranges(total_frames=totals[index], chunks=profile.chunks)
        for chunk_index, (start, end) in enumerate(ranges):
            work_items.append(
                _ChunkItem(
                    task_id=task_id,
                    video_index=index,
                    chunk_index=chunk_index,
                    video=str(video),
                    frame_start=start,
                    frame_end=end,
                    crop=crop,
                    destination=destination,
                )
            )
            chunk_frame_totals[task_id] = end - start
            key_video[task_id] = index
            task_id += 1

    slots = _build_slots(profile=profile, video_count=len(work_items), chunks=profile.chunks)

    manager = mp.Manager()
    video_queue = manager.Queue()
    progress_queue = manager.Queue()
    results_queue = manager.Queue()
    for item in work_items:
        video_queue.put(item)
    for _ in slots:
        video_queue.put(None)

    bar = None
    if display_progress:
        bar = AggregateBar(
            progress_queue=progress_queue,
            total_video_count=len(video_paths),
            frame_totals=chunk_frame_totals,
            preparing_label="compiling model...",
            key_video=key_video,
        )
        bar.start()

    launch = _InferenceLaunch(
        config=config,
        shuffle=shuffle,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
        profile=profile,
        batch_size=batch_size,
        detector_batch_size=detector_batch_size,
        display_progress=display_progress,
        video_queue=video_queue,
        progress_queue=progress_queue,
        results_queue=results_queue,
    )

    context = mp.get_context("spawn")
    processes = [context.Process(target=_run_chunk_worker, args=(slot, launch)) for slot in slots]
    outputs: dict[int, Path] = {}
    failures: list[tuple[str, str]] = []
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        outputs, failures = _collect_chunk_results(
            results_queue=results_queue, video_paths=video_paths, work_items=work_items, plan=plan
        )
    finally:
        if bar is not None:
            bar.stop()
            bar.join(timeout=_BAR_JOIN_TIMEOUT_SECONDS)
        manager.shutdown()

    return InferenceSummary(
        config=config,
        video_count=len(video_paths),
        destinations=destinations,
        device=profile.device,
        workers=len(slots),
        precision=_describe_precision(profile),
        outputs=tuple(outputs[index] for index in sorted(outputs)),
        failures=tuple(failures),
    )


def _build_analysis_plan(
    config: Path,
    shuffle: int,
    snapshot_index: int | None,
    detector_snapshot_index: int | None,
    batch_size: int | None,
) -> _AnalysisPlan:
    """Resolves, in the parent process, the project and model configuration a chunked run needs to stitch outputs.

    This mirrors the setup ``analyze_videos`` performs before its per-video loop, resolving the scorer string that
    names the output files and the configurations that assemble the prediction pickle and metadata. It touches no GPU,
    so it also fails fast on a missing snapshot or malformed project before any worker is spawned.

    Args:
        config: The path of the DeepLabCut project configuration file.
        shuffle: The shuffle index whose trained model is analyzed.
        snapshot_index: The requested pose snapshot index, or None for the project default.
        detector_snapshot_index: The requested detector snapshot index, or None for the project default.
        batch_size: The requested pose-model batch size, or None to use the project default.

    Returns:
        The resolved plan carrying the scorer string, configurations, and metadata inputs.
    """
    loader = DLCLoader(str(config), shuffle=shuffle)
    train_fraction = loader.project_cfg["TrainingFraction"][0]
    pose_cfg = read_plainconfig(loader.model_folder.parent / "test" / "pose_cfg.yaml")
    resolved_snapshot_index, _detector_index = dlc_apis_utils.parse_snapshot_index_for_analysis(
        cfg=loader.project_cfg,
        model_cfg=loader.model_cfg,
        snapshot_index=snapshot_index,
        detector_snapshot_index=detector_snapshot_index,
    )
    snapshot = dlc_apis_utils.get_model_snapshots(
        index=resolved_snapshot_index, model_folder=loader.model_folder, task=loader.pose_task
    )[0]
    resolved_batch_size = batch_size if batch_size is not None else loader.project_cfg.get("batch_size", 1)
    return _AnalysisPlan(
        scorer=loader.scorer(snapshot=snapshot, detector_snapshot=None),
        project_cfg=loader.project_cfg,
        model_cfg=loader.model_cfg,
        pose_cfg=pose_cfg,
        train_fraction=train_fraction,
        batch_size=resolved_batch_size,
        multi_animal=loader.project_cfg["multianimalproject"],
        pose_task=loader.pose_task,
    )


def _run_chunk_worker(slot: _Slot, launch: _InferenceLaunch) -> None:
    """Runs one worker slot for a chunked run, analyzing frame-range chunks pulled from the queue until it is drained.

    The worker builds its pose runner once, inside the runner-builder patch so it inherits the profile's precision and
    memory-format optimizations, then reuses it for every chunk it pulls. A chunk carries a frame range rather than a
    whole video, and the worker returns the chunk's raw predictions for the parent to stitch.

    Args:
        slot: The device and optional CPU-core placement for this worker.
        launch: The bundle of picklable per-run parameters shared by every worker process.
    """
    if slot.cores is not None and sys.platform != "darwin":
        with contextlib.suppress(Exception):
            psutil.Process().cpu_affinity(list(slot.cores))
    apply_runtime_optimizations(launch.profile)

    with patch_dlc_runner_builders(launch.profile):
        runner = _build_pose_runner(slot=slot, launch=launch)
        while True:
            item = launch.video_queue.get()
            if item is None:
                break
            _analyze_one_chunk(runner=runner, launch=launch, item=item)


def _build_pose_runner(slot: _Slot, launch: _InferenceLaunch) -> Any:
    """Builds the DeepLabCut pose-inference runner one chunk worker reuses for every chunk it analyzes.

    It resolves the model configuration and snapshot the same way ``analyze_videos`` does, then builds the runner
    through the patched ``get_pose_inference_runner`` so the runner-builder patch applies the profile's optimizations.
    The runner is crop-independent, so one instance serves every chunk of every video the worker handles.

    Args:
        slot: The device placement for this worker, whose device the runner is built on.
        launch: The bundle of picklable per-run parameters carrying the config, shuffle, snapshot, and batch size.

    Returns:
        The DeepLabCut pose-inference runner to analyze this worker's chunks with.
    """
    loader = DLCLoader(str(launch.config), shuffle=launch.shuffle)
    model_cfg = loader.model_cfg
    resolved_snapshot_index, _detector_index = dlc_apis_utils.parse_snapshot_index_for_analysis(
        cfg=loader.project_cfg,
        model_cfg=model_cfg,
        snapshot_index=launch.snapshot_index,
        detector_snapshot_index=launch.detector_snapshot_index,
    )
    snapshot = dlc_apis_utils.get_model_snapshots(
        index=resolved_snapshot_index, model_folder=loader.model_folder, task=loader.pose_task
    )[0]
    batch_size = launch.batch_size if launch.batch_size is not None else loader.project_cfg.get("batch_size", 1)
    individuals = model_cfg["metadata"]["individuals"]
    # Calls the builder through the apis-utils module so the active runner-builder patch, which replaces the module
    # attribute, wraps the runner with the profile's precision and memory-format optimizations.
    return dlc_apis_utils.get_pose_inference_runner(
        model_config=model_cfg,
        snapshot_path=snapshot.path,
        device=slot.device,
        max_individuals=len(individuals),
        batch_size=batch_size,
        inference_cfg=_STOCK_ACCELERATION_DISABLED,
    )


class _BoundedVideoIterator(VideoIterator):
    """Iterates exactly the frames ``[frame_start, frame_end)`` of a video for a single chunk worker.

    It seeks to the chunk's first frame on each pass and stops after emitting the chunk's frame count, so a worker
    decodes only its own range instead of the whole video. Seeking relies on frame-accurate ``CAP_PROP_POS_FRAMES``
    positioning, which the acquisition videos support.
    """

    def __init__(self, video_path: str, *, frame_start: int, frame_end: int, cropping: list[int] | None = None) -> None:
        """Opens the video and records the half-open frame range this iterator emits.

        Args:
            video_path: The path of the video to read frames from.
            frame_start: The inclusive index of the first frame to emit.
            frame_end: The exclusive index one past the last frame to emit.
            cropping: The ``[x1, x2, y1, y2]`` region to crop each frame to, or None to emit the full frame.
        """
        super().__init__(video_path, cropping=cropping)
        self._frame_start = frame_start
        self._frame_end = frame_end
        self._emitted = 0

    def __iter__(self) -> "_BoundedVideoIterator":
        """Seeks to the chunk's first frame and resets the emitted-frame counter for a fresh pass."""
        self.set_to_frame(self._frame_start)
        self._index = 0
        self._emitted = 0
        return self

    def __next__(self) -> NDArray[np.uint8]:
        """Returns the next frame in the chunk range, stopping once the range is exhausted or the video ends."""
        if self._emitted >= self._frame_end - self._frame_start:
            raise StopIteration
        frame = self.read_frame(crop=self._crop)
        if frame is None:
            raise StopIteration
        self._emitted += 1
        self._index += 1
        return frame.copy()

    def get_n_frames(self, *, robust: bool = False) -> int:  # noqa: ARG002 - a chunk's count is its range length.
        """Returns the chunk's frame count, keeping the whole-video frame-count mismatch warning silent.

        Args:
            robust: Ignored, since a chunk's frame count is exactly its range length regardless of a robust recount.

        Returns:
            The number of frames in this chunk's range.
        """
        return self._frame_end - self._frame_start


def _analyze_one_chunk(runner: Any, launch: _InferenceLaunch, item: _ChunkItem) -> None:
    """Analyzes one frame-range chunk and reports its raw predictions and progress to the parent.

    Args:
        runner: The pose-inference runner this worker reuses across chunks.
        launch: The bundle of picklable per-run parameters.
        item: The chunk work item describing the video, its frame range, and its crop.
    """
    original_tqdm = dlc_videos.tqdm
    if launch.display_progress:
        dlc_videos.tqdm = make_progress_reporter(
            progress_queue=launch.progress_queue,
            video_index=item.task_id,
            frame_total=item.frame_end - item.frame_start,
        )
    try:
        with _suppress_stdout(active=launch.display_progress):
            iterator = _BoundedVideoIterator(
                item.video, frame_start=item.frame_start, frame_end=item.frame_end, cropping=item.crop
            )
            predictions = video_inference(video=iterator, pose_runner=runner)
        expected_frames = item.frame_end - item.frame_start
        if len(predictions) != expected_frames:
            # A short read leaves a non-final chunk's predictions misaligned once concatenated, so this reports the
            # chunk as failed rather than silently shifting every later frame of the video into the wrong row.
            error_message = (
                f"chunk {item.chunk_index} covers {expected_frames} frames but decoded {len(predictions)}, so its "
                f"predictions cannot be stitched in frame order"
            )
            launch.results_queue.put((item.task_id, item.video_index, item.chunk_index, None, error_message))
        else:
            launch.results_queue.put((item.task_id, item.video_index, item.chunk_index, predictions, None))
    except Exception as error:
        launch.results_queue.put(
            (item.task_id, item.video_index, item.chunk_index, None, f"{type(error).__name__}: {error}")
        )
    finally:
        dlc_videos.tqdm = original_tqdm
        if launch.display_progress:
            with contextlib.suppress(Exception):
                launch.progress_queue.put(("done", item.task_id))


def _collect_chunk_results(
    results_queue: Any,
    video_paths: list[Path],
    work_items: list[_ChunkItem],
    plan: _AnalysisPlan,
) -> tuple[dict[int, Path], list[tuple[str, str]]]:
    """Gathers per-chunk predictions, then stitches each video's chunks into one prediction file in the parent.

    A video succeeds only when every one of its chunks reports without error. Its chunk predictions are concatenated in
    ascending frame order and written as the single prediction file the whole-video path would have produced.

    Args:
        results_queue: The shared queue workers report per-chunk predictions to.
        video_paths: The submitted videos, used to name failures and locate each video's output.
        work_items: The dispatched chunk work items, used to group results by video and order them by frame.
        plan: The resolved project configuration used to assemble each video's output files.

    Returns:
        A tuple of the written outputs keyed by video index and the list of failures as (video-name, error) pairs.
    """
    gathered: dict[int, tuple[Any, str | None]] = {}
    for _ in work_items:
        try:
            task_id, _video_index, _chunk_index, predictions, error = results_queue.get(
                timeout=_RESULT_POLL_TIMEOUT_SECONDS
            )
        except Exception:
            break
        gathered[task_id] = (predictions, error)

    items_by_video: dict[int, list[_ChunkItem]] = defaultdict(list)
    for item in work_items:
        items_by_video[item.video_index].append(item)

    outputs: dict[int, Path] = {}
    failures: list[tuple[str, str]] = []
    for video_index, items in items_by_video.items():
        name = video_paths[video_index].name
        present = [gathered.get(item.task_id) for item in items]
        reported = [result for result in present if result is not None]
        if len(reported) != len(items):
            failures.append((name, "a chunk worker exited before reporting a result"))
            continue
        error = next((chunk_error for _predictions, chunk_error in reported if chunk_error is not None), None)
        if error is not None:
            failures.append((name, error))
            continue
        predictions = []
        for chunk_item in sorted(items, key=lambda queued: queued.chunk_index):
            chunk_predictions, _chunk_error = gathered[chunk_item.task_id]
            predictions.extend(chunk_predictions)
        try:
            outputs[video_index] = _stitch_and_write(
                plan=plan,
                video=str(video_paths[video_index]),
                destination=items[0].destination,
                crop=items[0].crop,
                predictions=predictions,
            )
        except Exception as stitch_error:
            failures.append((name, f"{type(stitch_error).__name__}: {stitch_error}"))
    return outputs, failures


def _stitch_and_write(
    plan: _AnalysisPlan,
    video: str,
    destination: str | None,
    crop: list[int] | None,
    predictions: list[Any],
) -> Path:
    """Writes one video's stitched predictions as the prediction files the whole-video path would have produced.

    It reproduces the single-animal output of ``analyze_videos``, writing the ``_meta.pickle``, the ``_full.pickle``,
    and the native ``.h5`` prediction table from the concatenated per-chunk predictions, into the video's directory or
    the configured output directory.

    Args:
        plan: The resolved project configuration carrying the scorer string and model configurations.
        video: The path of the analyzed video, whose stem names the output files.
        destination: The directory to write the prediction files to, or None to write beside the video.
        crop: The ``[x1, x2, y1, y2]`` region the video was analyzed at, or None for the full frame.
        predictions: The video's per-frame predictions concatenated in frame order.

    Returns:
        The path of the written ``.h5`` prediction file.
    """
    output_directory = Path(destination) if destination is not None else Path(video).parent
    output_directory.mkdir(parents=True, exist_ok=True)
    output_prefix = Path(video).stem + plan.scorer

    metadata_video = VideoIterator(video, cropping=crop)
    metadata = _generate_metadata(
        cfg=plan.project_cfg,
        pytorch_config=plan.model_cfg,
        dlc_scorer=plan.scorer,
        train_fraction=plan.train_fraction,
        batch_size=plan.batch_size,
        cropping=crop,
        runtime=(0.0, 0.0),
        video=metadata_video,
    )
    with (output_directory / f"{output_prefix}_meta.pickle").open("wb") as handle:
        pickle.dump(metadata, handle, pickle.HIGHEST_PROTOCOL)

    output_data = _generate_output_data(pose_config=plan.pose_cfg, predictions=predictions)
    with (output_directory / f"{output_prefix}_full.pickle").open("wb") as handle:
        pickle.dump(output_data, handle, pickle.HIGHEST_PROTOCOL)

    create_df_from_prediction(
        predictions=predictions,
        dlc_scorer=plan.scorer,
        multi_animal=plan.multi_animal,
        model_cfg=plan.model_cfg,
        output_path=output_directory,
        output_prefix=output_prefix,
        save_as_csv=False,
    )
    return output_directory / f"{output_prefix}.h5"
