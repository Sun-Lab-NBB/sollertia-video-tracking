"""Provides the multi-device inference pipeline that runs DeepLabCut over many videos across worker slots."""

import os
from typing import Any
from pathlib import Path
import contextlib
from dataclasses import dataclass
from collections.abc import Iterator, Sequence

import cv2
import psutil
import torch.multiprocessing as mp
from deeplabcut.utils.auxiliaryfunctions import read_config
from deeplabcut.pose_estimation_pytorch.apis import videos as dlc_videos

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
    except Exception:  # noqa: BLE001 - detection is best-effort; any failure conservatively reports not-fixed.
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
            cannot be pinned to disjoint core blocks.
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


def _build_slots(profile: InferenceProfile, video_count: int) -> list[_Slot]:
    """Builds the worker slots for the run from the profile and the number of videos.

    Args:
        profile: The resolved optimization profile.
        video_count: The number of videos to process, used to avoid spawning more workers than videos.

    Returns:
        The list of worker slots to spawn.
    """
    if profile.on_cuda:
        # Round-robins the device order (cuda:0, cuda:1, cuda:0, ...) rather than grouping by device
        # (cuda:0, cuda:0, cuda:1, ...). When there are fewer videos than slots, the list is truncated below, so this
        # ordering spreads the surviving workers across every GPU before oversubscribing any single one.
        slots = [
            _Slot(device=f"cuda:{index}", cores=None) for _ in range(profile.gpu_processes) for index in profile.gpus
        ]
    elif profile.device == "cpu":
        core_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        _workers, core_sets = plan_core_allocation(
            video_count=video_count,
            total_core_count=core_count,
            worker_count=profile.cpu_workers,
            cores_per_worker=profile.cpu_threads_per_worker or -1,
            reserved_core_count=core_count - _usable_cpu_cores(profile),
        )
        slots = [_Slot(device="cpu", cores=tuple(sorted(cores))) for cores in core_sets]
    else:
        slots = [_Slot(device=profile.device, cores=None)]

    if not slots:
        message = (
            "Unable to build inference worker slots. The profile selected the CUDA device but no GPU indices were "
            "resolved. Pass a device or an explicit GPU list that selects at least one device."
        )
        raise ValueError(message)

    limit = max(1, min(len(slots), video_count))
    return slots[:limit]


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
        except Exception:  # noqa: BLE001 - a missing result means a worker died; report the remaining videos below.
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
    if slot.cores is not None:
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
    except Exception as error:  # noqa: BLE001 - report the per-video failure and keep draining the queue.
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
