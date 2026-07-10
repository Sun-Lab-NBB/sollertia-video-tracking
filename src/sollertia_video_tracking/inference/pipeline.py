"""Provides the multi-device inference pipeline that runs DeepLabCut over many videos across worker slots."""

import os
from typing import Any
from pathlib import Path
import contextlib
from dataclasses import dataclass
from collections.abc import Iterator

import cv2
import psutil
import torch.multiprocessing as mp
from deeplabcut.utils.auxiliaryfunctions import read_config
from deeplabcut.pose_estimation_pytorch.apis import videos as dlc_videos

from .runners import patch_dlc_runner_builders
from .conversion import convert_predictions_to_feather
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
        The summary is built after every worker has finished. ``outputs`` holds the produced files in video order (a
        feather per video when conversion is enabled, otherwise the DeepLabCut ``.h5``), and ``failures`` pairs each
        failed video with its error message so a partial run is reported honestly rather than silently.
    """

    config: Path
    """The path of the DeepLabCut project configuration file inference ran for."""
    video_count: int
    """The number of videos submitted for inference."""
    destination: Path | None
    """The directory the prediction files were written to, or None when each video's predictions were written beside
    the video itself."""
    device: str
    """The base device type inference ran on (``"cuda"``, ``"cpu"``, or ``"mps"``)."""
    workers: int
    """The number of worker processes used."""
    precision: str
    """The compute precision used (``"bfloat16"``, ``"float16"``, or ``"fp32"``)."""
    converted: bool
    """Determines whether the predictions were converted in-flight to polars feather files."""
    outputs: tuple[Path, ...]
    """The produced output files, one per successfully processed video."""
    failures: tuple[tuple[str, str], ...]
    """The videos that failed, each paired with its error message."""

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the inference run for the CLI.

        Returns:
            A compact description of how many videos were processed, on what hardware, and where results were written.
        """
        ok = len(self.outputs)
        where = f"{self.device} x{self.workers}"
        output_format = "feather" if self.converted else "h5"
        tail = f", {len(self.failures)} failed" if self.failures else ""
        written_to = self.destination if self.destination is not None else "each video's directory"
        return (
            f"analyzed {ok}/{self.video_count} videos on {where} in {self.precision} -> "
            f"{output_format} in {written_to}{tail}"
        )


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
    destination: Path | None
    """The directory prediction files are written to, or None to write beside each video."""
    profile: InferenceProfile
    """The resolved optimization profile describing the device, precision, and parallelism to use."""
    batch_size: int | None
    """The pose-model inference batch size, or None to use the configured value."""
    detector_batch_size: int | None
    """The detector inference batch size, or None to use the configured value."""
    to_polars: bool
    """Determines whether each video's predictions are converted in-flight to a polars feather file."""
    likelihood_threshold: float
    """The likelihood below which keypoint positions are masked to NaN during conversion."""
    save_as_csv: bool
    """Determines whether DeepLabCut also writes a CSV alongside each prediction file."""
    keep_dlc_outputs: bool
    """Determines whether DeepLabCut's own prediction artifacts are kept after conversion."""
    write_provenance: bool
    """Determines whether the in-flight conversion writes its own provenance sidecar beside each feather."""
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


def detect_fixed_input_size(config: str | Path, videos: list[str | Path]) -> bool:
    """Determines whether every video would feed the pose network a single fixed input resolution.

    The cuDNN autotuner only pays off when the convolution input shapes stay constant across the run, so this reports
    whether that precondition holds instead of asking the operator to assert it. When the project is configured to
    crop, every analyzed video is reduced to its crop rectangle, so the run is fixed-size exactly when all videos
    share one crop size. Otherwise the network sees each video's native resolution, so the run is fixed-size exactly
    when all videos share one resolution. A single video is therefore always fixed-size. Any inability to read the
    configuration or a video's dimensions is treated conservatively as not fixed, since a wrong assertion of fixed
    size makes the autotuner harmful.

    Args:
        config: The path of the DeepLabCut project configuration file.
        videos: The video files the run will analyze.

    Returns:
        True when the network's spatial input size is provably constant across the whole run, False otherwise.
    """
    video_paths = [Path(video) for video in videos]
    if not video_paths:
        return False
    try:
        project_config = read_config(str(Path(config)))
        sizes = {_resolve_input_size(project_config=project_config, video=video) for video in video_paths}
    except Exception:  # noqa: BLE001 - detection is best-effort; any failure conservatively reports not-fixed.
        return False
    return None not in sizes and len(sizes) == 1


def run_inference(
    config: str | Path,
    videos: list[str | Path],
    profile: InferenceProfile,
    *,
    destination: str | Path | None = None,
    shuffle: int = 1,
    snapshot_index: int | None = None,
    detector_snapshot_index: int | None = None,
    batch_size: int | None = None,
    detector_batch_size: int | None = None,
    to_polars: bool = False,
    likelihood_threshold: float = 0.0,
    save_as_csv: bool = False,
    keep_dlc_outputs: bool = True,
    output_feathers: list[str | Path] | None = None,
    write_conversion_provenance: bool = True,
    display_progress: bool = True,
) -> InferenceSummary:
    """Runs DeepLabCut inference over many videos, distributing whole videos across GPU or CPU worker slots.

    Each worker pulls whole videos from a shared queue and analyzes them with DeepLabCut, so the work is balanced
    across slots without splitting any video. On CUDA a slot is a device (``gpu_processes`` of them per device); on CPU
    a slot is a disjoint, thread-bounded block of physical cores. Every worker's forward pass is wrapped with the
    profile's mixed precision and channels-last format, and each video's predictions are optionally converted in-flight
    to a wide polars feather.

    Args:
        config: The path of the DeepLabCut project configuration file.
        videos: The video files to analyze.
        profile: The resolved optimization profile describing the device, precision, and parallelism to use.
        destination: The directory prediction files are written to, or None to write each video's predictions beside
            the video itself, matching DeepLabCut's own default and the location the outlier-extraction step reads.
        shuffle: The shuffle index whose trained model is used.
        snapshot_index: The pose snapshot index to use, or None for the configured default.
        detector_snapshot_index: The detector snapshot index to use, or None for the configured default.
        batch_size: The pose-model inference batch size, or None to use the configured value.
        detector_batch_size: The detector inference batch size, or None to use the configured value.
        to_polars: Determines whether to convert each video's predictions in-flight to a polars feather file. Off by
            default so the pipeline behaves like DeepLabCut's own analyze and leaves the project's native prediction
            files intact for the rest of the model-refinement loop; the feather is deferred to the deployment path.
        likelihood_threshold: The likelihood below which keypoint positions are masked to NaN during conversion.
        save_as_csv: Determines whether DeepLabCut also writes a CSV alongside each prediction file.
        keep_dlc_outputs: Determines whether to keep DeepLabCut's own prediction artifacts (the HDF5 table, the
            full/meta/assemblies pickles, and any tracker files) after conversion. Kept by default so all DeepLabCut
            data survives; only relevant when ``to_polars`` is set, since conversion is what would otherwise remove
            them.
        output_feathers: The per-video feather paths to write, parallel to ``videos``, or None to name each video's
            feather from its stem inside ``destination``. Only valid together with ``to_polars``.
        write_conversion_provenance: Determines whether the in-flight conversion writes its own provenance sidecar
            beside each feather. Kept on by default; a deployment caller disables it to write a richer sidecar itself.
        display_progress: Determines whether to render the live aggregate progress bar.

    Returns:
        A summary of what was analyzed and the hardware configuration used.

    Raises:
        ValueError: Raised when no videos are provided. Raised when per-video output paths are provided while
            ``to_polars`` is disabled, or when their count does not match the videos. Raised when the profile selects
            CUDA but resolves no GPU indices to build worker slots from. Raised when an explicit CPU worker/thread
            configuration cannot be pinned to disjoint core blocks.
    """
    config = Path(config)
    destination = Path(destination) if destination is not None else None
    video_paths = [Path(video) for video in videos]
    if not video_paths:
        message = "Unable to run inference. Expected at least one video, but got an empty video list."
        raise ValueError(message)
    if output_feathers is not None:
        if not to_polars:
            message = (
                "Unable to run inference. Per-video output paths were provided, but polars conversion is disabled; "
                "enable to_polars to write feather files at the requested paths."
            )
            raise ValueError(message)
        if len(output_feathers) != len(video_paths):
            message = (
                f"Unable to run inference. Expected one output path per video, but got {len(output_feathers)} output "
                f"paths for {len(video_paths)} videos."
            )
            raise ValueError(message)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    totals = {index: _probe_frame_count(video) for index, video in enumerate(video_paths)}
    slots = _build_slots(profile=profile, video_count=len(video_paths))

    # Resolves each video's crop once, in the parent, so every worker analyzes the same region the frames were
    # extracted from when the project is configured to crop, keeping predictions in the model's coordinate space.
    project_config = read_config(str(config))

    manager = mp.Manager()
    video_queue = manager.Queue()
    progress_queue = manager.Queue()
    results_queue = manager.Queue()
    for index, video in enumerate(video_paths):
        crop = _resolve_video_cropping(project_config=project_config, video=str(video))
        feather = str(Path(output_feathers[index])) if output_feathers is not None else None
        video_queue.put((index, str(video), totals[index], crop, feather))
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
        destination=destination,
        profile=profile,
        batch_size=batch_size,
        detector_batch_size=detector_batch_size,
        to_polars=to_polars,
        likelihood_threshold=likelihood_threshold,
        save_as_csv=save_as_csv,
        keep_dlc_outputs=keep_dlc_outputs,
        write_provenance=write_conversion_provenance,
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
        destination=destination,
        device=profile.device,
        workers=len(slots),
        precision=precision,
        converted=to_polars,
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
        video: The path to the video file.

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


def _parse_crop(crop: Any) -> list[int] | None:
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
    for registered, meta in (project_config.get("video_sets") or {}).items():
        if not isinstance(meta, dict) or Path(registered).resolve() != target:
            continue
        crop = _parse_crop(meta.get("crop"))
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
        video: The path to the video file.

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
    """Analyzes a single video from the queue, converts its output, and reports the result and progress.

    Args:
        slot: The device and optional CPU-core placement for this worker.
        launch: The bundle of picklable per-run parameters.
        item: The (video index, video path, frame total, crop rectangle, output feather path) work item pulled from the
            queue. The crop rectangle is the ``[x1, x2, y1, y2]`` region to analyze, or None to analyze the full frame.
            The output feather path is the explicit destination for this video's feather, or None to name it from the
            video stem.
    """
    index, video, total, cropping, output_feather = item
    output_directory = launch.destination if launch.destination is not None else Path(video).parent
    original_tqdm = dlc_videos.tqdm
    if launch.display_progress:
        dlc_videos.tqdm = make_progress_reporter(launch.progress_queue, index, total)
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
                save_as_csv=launch.save_as_csv,
                # The submitted video is always (re)analyzed. DeepLabCut otherwise skips any video whose companion
                # ``_full.pickle`` already exists, which would silently skip re-runs (and, since the converted HDF5 is
                # deleted, report every already-completed video as a failure) when a batch is resubmitted.
                overwrite=True,
                inference_cfg=_STOCK_ACCELERATION_DISABLED,
            )
        output = _resolve_output(
            launch=launch, video=video, scorer=scorer, destination=output_directory, feather_override=output_feather
        )
        launch.results_queue.put((index, str(output) if output is not None else None, None))
    except Exception as error:  # noqa: BLE001 - report the per-video failure and keep draining the queue.
        launch.results_queue.put((index, None, f"{type(error).__name__}: {error}"))
    finally:
        dlc_videos.tqdm = original_tqdm
        if launch.display_progress:
            with contextlib.suppress(Exception):
                launch.progress_queue.put(("done", index))


def _cleanup_dlc_artifacts(destination: Path, stem: str, scorer: str, *, keep_csv: bool) -> None:
    """Removes DeepLabCut's own prediction files for one video, leaving only the converted feather and its sidecar.

    DeepLabCut writes several files per video that all share the ``{stem}{scorer}`` name prefix: the HDF5 table, the
    full/meta/assemblies pickles, and any tracker-suffixed files. The feather and its provenance sidecar are named from
    the video stem alone, so matching on the prefix removes exactly the DeepLabCut artifacts and nothing else.

    Args:
        destination: The directory the prediction files were written to.
        stem: The analyzed video's file stem.
        scorer: The DeepLabCut scorer string that prefixes every DeepLabCut output file for this video.
        keep_csv: Determines whether to keep a ``.csv`` export the caller explicitly requested.
    """
    prefix = f"{stem}{scorer}"
    for artifact in destination.iterdir():
        if not artifact.is_file() or not artifact.name.startswith(prefix):
            continue
        if keep_csv and artifact.suffix == ".csv":
            continue
        artifact.unlink(missing_ok=True)


def _resolve_output(
    launch: _InferenceLaunch, video: str, scorer: str, destination: Path, feather_override: str | None = None
) -> Path | None:
    """Locates the prediction HDF5 DeepLabCut wrote and optionally converts it to a polars feather.

    Args:
        launch: The bundle of per-run parameters, providing the conversion settings.
        video: The analyzed video path.
        scorer: The DeepLabCut scorer string returned by ``analyze_videos``, which names the output file.
        destination: The directory this video's predictions were written to, which is the video's own directory when the
            run configured no destination.
        feather_override: The explicit feather path to write for this video, or None to name it from the video stem
            inside ``destination``.

    Returns:
        The produced feather (when converting) or HDF5 file, or None when no prediction file was written.
    """
    stem = Path(video).stem
    exact = destination / f"{stem}{scorer}.h5"
    if exact.exists():
        h5_path: Path | None = exact
    else:
        # Multi-animal auto-tracking writes a tracker-suffixed file instead of the plain per-frame table.
        matches = sorted(destination.glob(f"{stem}{scorer}*.h5"))
        h5_path = matches[-1] if matches else None

    if h5_path is None:
        return None
    if not launch.to_polars:
        return h5_path

    feather_path = Path(feather_override) if feather_override is not None else destination / f"{stem}_pose.feather"
    convert_predictions_to_feather(
        h5_path=h5_path,
        feather_path=feather_path,
        likelihood_threshold=launch.likelihood_threshold,
        write_provenance=launch.write_provenance,
    )
    # Deployment leaves only the feather and its provenance sidecar: once conversion succeeds, DeepLabCut's own
    # prediction artifacts for this video are removed. Conversion runs just above, so a failed conversion raises before
    # this point and leaves those artifacts in place as a fallback.
    if not launch.keep_dlc_outputs:
        _cleanup_dlc_artifacts(destination=destination, stem=stem, scorer=scorer, keep_csv=launch.save_as_csv)
    return feather_path
