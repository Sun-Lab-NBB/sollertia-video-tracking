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
from deeplabcut.pose_estimation_pytorch.apis import videos as dlc_videos

from .runners import patch_dlc_runner_builders
from .conversion import convert_predictions_to_feather
from .optimization import InferenceProfile, apply_runtime_optimizations
from ..frame_extraction.progress import AggregateBar, make_progress_reporter
from ..frame_extraction.cpu_allocation import plan_core_allocation

# The inference-config overrides passed to analyze_videos: our runner wrappers own autocast and compilation, so
# DeepLabCut's own autocast and compile are disabled while its async decode pipeline is left at its default.
_STOCK_ACCELERATION_DISABLED: dict[str, dict[str, bool]] = {
    "autocast": {"enabled": False},
    "compile": {"enabled": False},
}


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
    destination: Path
    """The directory the prediction files were written to."""
    device: str
    """The base device type inference ran on (``"cuda"``, ``"cpu"``, or ``"mps"``)."""
    workers: int
    """The number of worker processes used."""
    precision: str
    """The compute precision used (``"bfloat16"``, ``"float16"``, or ``"fp32"``)."""
    converted: bool
    """Whether the predictions were converted in-flight to polars feather files."""
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
        fmt = "feather" if self.converted else "h5"
        tail = f", {len(self.failures)} failed" if self.failures else ""
        return (
            f"analyzed {ok}/{self.video_count} videos on {where} in {self.precision} "
            f"-> {fmt} in {self.destination}{tail}"
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
    shuffle: int
    snapshot_index: int | None
    detector_snapshot_index: int | None
    destination: Path
    profile: InferenceProfile
    batch_size: int | None
    detector_batch_size: int | None
    to_polars: bool
    likelihood_threshold: float
    save_as_csv: bool
    keep_dlc_outputs: bool
    display_progress: bool
    video_queue: Any
    progress_queue: Any
    results_queue: Any


def run_inference(
    config: str | Path,
    videos: list[str | Path],
    destination: str | Path,
    profile: InferenceProfile,
    *,
    shuffle: int = 1,
    snapshot_index: int | None = None,
    detector_snapshot_index: int | None = None,
    batch_size: int | None = None,
    detector_batch_size: int | None = None,
    to_polars: bool = True,
    likelihood_threshold: float = 0.0,
    save_as_csv: bool = False,
    keep_dlc_outputs: bool = False,
    heartbeat: float = 30.0,
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
        destination: The directory prediction files are written to.
        profile: The resolved optimization profile describing the device, precision, and parallelism to use.
        shuffle: The shuffle index whose trained model is used.
        snapshot_index: The pose snapshot index to use, or None for the configured default.
        detector_snapshot_index: The detector snapshot index to use, or None for the configured default.
        batch_size: The pose-model inference batch size, or None to use the configured value.
        detector_batch_size: The detector inference batch size, or None to use the configured value.
        to_polars: Whether to convert each video's predictions in-flight to a polars feather file.
        likelihood_threshold: The likelihood below which keypoint positions are masked to NaN during conversion.
        save_as_csv: Whether DeepLabCut also writes a CSV alongside each prediction file.
        keep_dlc_outputs: Whether to keep DeepLabCut's own prediction artifacts (the HDF5 table, the full/meta/
            assemblies pickles, and any tracker files) after conversion. By default deployment removes them once the
            feather is written, leaving only the feather and its provenance sidecar in the destination.
        heartbeat: The minimum interval, in seconds, between progress lines when the output is not a TTY.
        display_progress: Whether to render the live aggregate progress bar.

    Returns:
        A summary of what was analyzed and the hardware configuration used.

    Raises:
        ValueError: When no videos are provided.
    """
    config = Path(config)
    destination = Path(destination)
    video_paths = [Path(video) for video in videos]
    if not video_paths:
        message = "Unable to run inference. Expected at least one video, but got an empty video list."
        raise ValueError(message)
    destination.mkdir(parents=True, exist_ok=True)

    totals = {index: _probe_frame_count(video) for index, video in enumerate(video_paths)}
    slots = _build_slots(profile, video_count=len(video_paths))

    manager = mp.Manager()
    video_queue = manager.Queue()
    progress_queue = manager.Queue()
    results_queue = manager.Queue()
    for index, video in enumerate(video_paths):
        video_queue.put((index, str(video), totals[index]))
    for _ in slots:
        video_queue.put(None)

    bar = None
    if display_progress:
        bar = AggregateBar(
            progress_queue=progress_queue, total_videos=len(video_paths), totals=totals, heartbeat=heartbeat
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
        outputs, failures = _collect_results(results_queue, video_paths)
    finally:
        if bar is not None:
            bar.stop()
            bar.join(timeout=3)
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
        # Round-robin the device order (cuda:0, cuda:1, cuda:0, ...) rather than grouping by device
        # (cuda:0, cuda:0, cuda:1, ...). When there are fewer videos than slots, the list is truncated below, so this
        # ordering spreads the surviving workers across every GPU before oversubscribing any single one.
        slots = [
            _Slot(device=f"cuda:{index}", cores=None)
            for _ in range(profile.gpu_processes)
            for index in profile.gpus
        ]
    elif profile.device == "cpu":
        core_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        _workers, core_sets = plan_core_allocation(
            video_count=video_count,
            core_count=core_count,
            workers=profile.cpu_workers,
            cores_per_worker=profile.cpu_threads_per_worker or -1,
            reserve_cores=core_count - _usable_cpu_cores(profile),
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
            index, path, error = results_queue.get(timeout=5.0)
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
        active: Whether to suppress standard output; when False the context does nothing.

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
            _analyze_one_video(slot, launch, item)


def _analyze_one_video(slot: _Slot, launch: _InferenceLaunch, item: tuple[int, str, int]) -> None:
    """Analyzes a single video from the queue, converts its output, and reports the result and progress.

    Args:
        slot: The device and optional CPU-core placement for this worker.
        launch: The bundle of picklable per-run parameters.
        item: The (video index, video path, frame total) work item pulled from the queue.
    """
    index, video, total = item
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
                destfolder=str(launch.destination),
                snapshot_index=launch.snapshot_index,
                detector_snapshot_index=launch.detector_snapshot_index,
                batch_size=launch.batch_size,
                detector_batch_size=launch.detector_batch_size,
                save_as_csv=launch.save_as_csv,
                # Always (re)analyze the submitted video. DeepLabCut otherwise skips any video whose companion
                # ``_full.pickle`` already exists, which would silently skip re-runs (and, since the converted HDF5 is
                # deleted, report every already-completed video as a failure) when a batch is resubmitted.
                overwrite=True,
                inference_cfg=_STOCK_ACCELERATION_DISABLED,
            )
        output = _resolve_output(launch, video, scorer)
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
        keep_csv: Whether to keep a ``.csv`` export the caller explicitly requested.
    """
    prefix = f"{stem}{scorer}"
    for artifact in destination.iterdir():
        if not artifact.is_file() or not artifact.name.startswith(prefix):
            continue
        if keep_csv and artifact.suffix == ".csv":
            continue
        artifact.unlink(missing_ok=True)


def _resolve_output(launch: _InferenceLaunch, video: str, scorer: str) -> Path | None:
    """Locates the prediction HDF5 DeepLabCut wrote and optionally converts it to a polars feather.

    Args:
        launch: The bundle of per-run parameters, providing the destination and conversion settings.
        video: The analyzed video path.
        scorer: The DeepLabCut scorer string returned by ``analyze_videos``, which names the output file.

    Returns:
        The produced feather (when converting) or HDF5 file, or None when no prediction file was written.
    """
    stem = Path(video).stem
    exact = launch.destination / f"{stem}{scorer}.h5"
    if exact.exists():
        h5_path: Path | None = exact
    else:
        # Multi-animal auto-tracking writes a tracker-suffixed file instead of the plain per-frame table.
        matches = sorted(launch.destination.glob(f"{stem}{scorer}*.h5"))
        h5_path = matches[-1] if matches else None

    if h5_path is None:
        return None
    if not launch.to_polars:
        return h5_path

    feather_path = launch.destination / f"{stem}_pose.feather"
    convert_predictions_to_feather(h5_path, feather_path, likelihood_threshold=launch.likelihood_threshold)
    # Deployment leaves only the feather and its provenance sidecar: once conversion succeeds, remove DeepLabCut's own
    # prediction artifacts for this video. Conversion runs just above, so a failed conversion raises before this point
    # and leaves those artifacts in place as a fallback.
    if not launch.keep_dlc_outputs:
        _cleanup_dlc_artifacts(launch.destination, stem, scorer, keep_csv=launch.save_as_csv)
    return feather_path
