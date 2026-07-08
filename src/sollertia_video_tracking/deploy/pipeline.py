"""Runs a portable model asset over videos and routes each prediction to a caller-chosen feather path."""

from pathlib import Path
import tempfile
import contextlib
from dataclasses import dataclass
from collections.abc import Sequence

from ruamel.yaml import YAML

from .asset import ModelAsset, load_model_asset
from ..inference import InferenceProfile, InferenceSummary, run_inference


@dataclass(frozen=True, slots=True)
class PredictionJob:
    """Pairs one input video with the exact feather path its predictions are written to."""

    video: Path
    """The input video to analyze."""
    output: Path
    """The exact feather path this video's predictions are written to; its parent is created if missing."""


@dataclass(frozen=True, slots=True)
class JobResult:
    """Captures the outcome of one prediction job for reporting to the caller."""

    video: Path
    """The input video that was analyzed."""
    output: Path
    """The feather path the predictions were written to."""
    succeeded: bool
    """Determines whether the prediction feather was produced at the requested path."""
    error: str | None
    """The failure message when the job did not produce a feather, or None on success."""


@dataclass(frozen=True, slots=True)
class PredictionSummary:
    """Captures the outcome of a completed deployment run over one or more videos for reporting to the caller.

    Notes:
        The summary is built after every job has been attempted. ``results`` holds one entry per submitted job in
        submission order, so a caller can reconcile each video with its feather even when some jobs failed.
    """

    asset_path: Path
    """The model asset the predictions were produced with."""
    job_count: int
    """The number of prediction jobs submitted."""
    device: str
    """The base device type inference ran on (``"cuda"``, ``"cpu"``, or ``"mps"``)."""
    workers: int
    """The number of worker processes used per batch."""
    precision: str
    """The compute precision used (``"bfloat16"``, ``"float16"``, or ``"fp32"``)."""
    results: tuple[JobResult, ...]
    """The per-job outcomes, in submission order."""

    @property
    def successful(self) -> bool:
        """Returns whether every submitted job produced its prediction feather."""
        return all(result.succeeded for result in self.results)

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the deployment run for the CLI.

        Returns:
            A compact description of how many videos were predicted, on what hardware, and how many failed.
        """
        produced = sum(1 for result in self.results if result.succeeded)
        tail = f", {self.job_count - produced} failed" if produced < self.job_count else ""
        return (
            f"predicted {produced}/{self.job_count} videos on {self.device} x{self.workers} in {self.precision}{tail}"
        )


def run_predictions(
    asset: str | Path,
    jobs: Sequence[PredictionJob | tuple[str | Path, str | Path]],
    profile: InferenceProfile,
    *,
    likelihood_threshold: float | None = None,
    batch_size: int | None = None,
    detector_batch_size: int | None = None,
    scratch_directory: str | Path | None = None,
    display_progress: bool = True,
) -> PredictionSummary:
    """Runs a portable model asset over videos and writes each video's predictions to its requested feather path.

    The asset is extracted into a temporary project shell, DeepLabCut inference runs against that shell with the same
    hardware optimizations the profile describes, and each video's predictions are converted in-flight to a wide polars
    feather at the caller's chosen path. Every intermediary, including the extracted shell and DeepLabCut's own
    prediction artifacts, is removed before this returns, whether the run succeeds or fails, so each caller path holds
    only its prediction feather and nothing else.

    Args:
        asset: The path of the model asset to run.
        jobs: The prediction jobs to run, each pairing an input video with the feather path its predictions are written
            to, given as a ``PredictionJob`` or a ``(video, output)`` tuple.
        profile: The resolved optimization profile describing the device, precision, and parallelism to use.
        likelihood_threshold: The likelihood below which keypoint positions are masked to NaN, or None to use the
            default baked into the asset at export time.
        batch_size: The pose-model inference batch size, or None to use the value the model was configured with.
        detector_batch_size: The detector inference batch size for a top-down model, or None to use the configured
            value.
        scratch_directory: An existing directory the temporary extraction and inference files are placed under, or None
            to use the system temporary directory; point it at fast node-local storage on a cluster.
        display_progress: Determines whether to render the live aggregate progress bar during analysis.

    Returns:
        A summary describing which videos were predicted and the hardware configuration used.

    Raises:
        ValueError: When no jobs are provided or two jobs request the same output path.
    """
    asset = Path(asset)
    normalized_jobs = _normalize_jobs(jobs)
    scratch_root = Path(scratch_directory) if scratch_directory is not None else None

    device = profile.device
    workers = 0
    precision = ""
    results: list[JobResult] = []
    with contextlib.ExitStack() as stack:
        shell = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="slvt_model_", dir=scratch_root)))
        model = load_model_asset(asset, shell)
        _rewrite_project_path(model.config_path)
        threshold = model.manifest.likelihood_threshold if likelihood_threshold is None else likelihood_threshold

        for batch in _partition_by_unique_stem(normalized_jobs):
            pen = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="slvt_predict_", dir=scratch_root)))
            summary = _analyze_batch(
                model=model,
                batch=batch,
                profile=profile,
                scratch=pen,
                threshold=threshold,
                batch_size=batch_size,
                detector_batch_size=detector_batch_size,
                display_progress=display_progress,
            )
            workers = summary.workers
            precision = summary.precision
            results.extend(_finalize_job(job=job, summary=summary) for job in batch)

    return PredictionSummary(
        asset_path=asset,
        job_count=len(results),
        device=device,
        workers=workers,
        precision=precision,
        results=tuple(results),
    )


def _analyze_batch(
    model: ModelAsset,
    batch: list[PredictionJob],
    profile: InferenceProfile,
    scratch: Path,
    threshold: float,
    batch_size: int | None,
    detector_batch_size: int | None,
    *,
    display_progress: bool,
) -> InferenceSummary:
    """Runs inference for one stem-unique batch of jobs against the extracted model shell.

    Args:
        model: The extracted, verified model asset to run.
        batch: The jobs to analyze together, guaranteed to have distinct video stems.
        profile: The resolved optimization profile to apply.
        scratch: A temporary directory DeepLabCut's prediction artifacts are written to and cleaned from.
        threshold: The likelihood below which keypoint positions are masked to NaN during conversion.
        batch_size: The pose-model inference batch size, or None to use the configured value.
        detector_batch_size: The detector inference batch size, or None to use the configured value.
        display_progress: Determines whether to render the live aggregate progress bar.

    Returns:
        The inference summary DeepLabCut produced for the batch.
    """
    return run_inference(
        config=model.config_path,
        videos=[job.video for job in batch],
        profile=profile,
        destination=scratch,
        shuffle=model.manifest.shuffle,
        snapshot_index=model.snapshot_index,
        detector_snapshot_index=model.detector_snapshot_index,
        batch_size=batch_size,
        detector_batch_size=detector_batch_size,
        to_polars=True,
        keep_dlc_outputs=False,
        likelihood_threshold=threshold,
        output_feathers=[job.output for job in batch],
        write_conversion_provenance=False,
        display_progress=display_progress,
    )


def _finalize_job(job: PredictionJob, summary: InferenceSummary) -> JobResult:
    """Records one job's outcome from whether its prediction feather was produced at the requested path.

    Args:
        job: The prediction job to finalize.
        summary: The inference summary for the batch the job ran in, used to recover a failure message.

    Returns:
        The recorded outcome for the job.
    """
    if not job.output.exists():
        return JobResult(video=job.video, output=job.output, succeeded=False, error=_failure_message(summary, job))
    return JobResult(video=job.video, output=job.output, succeeded=True, error=None)


def _normalize_jobs(jobs: Sequence[PredictionJob | tuple[str | Path, str | Path]]) -> list[PredictionJob]:
    """Coerces the submitted jobs to ``PredictionJob`` objects and validates that they are runnable.

    Args:
        jobs: The prediction jobs to normalize, each a ``PredictionJob`` or a ``(video, output)`` tuple.

    Returns:
        The normalized jobs with path-typed fields.

    Raises:
        ValueError: When no jobs are provided or two jobs request the same output path.
    """
    normalized: list[PredictionJob] = []
    for job in jobs:
        if isinstance(job, PredictionJob):
            normalized.append(PredictionJob(video=Path(job.video), output=Path(job.output)))
        else:
            video, output = job
            normalized.append(PredictionJob(video=Path(video), output=Path(output)))

    if not normalized:
        message = "Unable to run predictions. Expected at least one (video, output) job, but got none."
        raise ValueError(message)

    outputs = [job.output.resolve() for job in normalized]
    if len(set(outputs)) != len(outputs):
        message = "Unable to run predictions. Two or more jobs request the same output path, which would overwrite one."
        raise ValueError(message)
    return normalized


def _partition_by_unique_stem(jobs: list[PredictionJob]) -> list[list[PredictionJob]]:
    """Partitions jobs into batches within which no two videos share a file stem.

    DeepLabCut names its per-video prediction artifacts from the video stem and writes them into one shared destination,
    so two videos with the same stem in one batch would collide and cross-delete during cleanup. Grouping by unique stem
    keeps the common case (distinct file names) as a single batch that fully uses the hardware, while duplicate stems
    fall into separate batches.

    Args:
        jobs: The normalized jobs to partition.

    Returns:
        The batches of jobs, each with distinct video stems.
    """
    batches: list[list[PredictionJob]] = []
    for job in jobs:
        stem = job.video.stem
        placed = False
        for batch in batches:
            if all(existing.video.stem != stem for existing in batch):
                batch.append(job)
                placed = True
                break
        if not placed:
            batches.append([job])
    return batches


def _failure_message(summary: InferenceSummary, job: PredictionJob) -> str:
    """Recovers the failure message for a job whose feather was not produced.

    Args:
        summary: The inference summary for the batch, holding per-video failure messages.
        job: The job whose failure message to recover.

    Returns:
        The matching failure message, or a generic message when the summary reported none.
    """
    for video_name, error in summary.failures:
        if video_name == job.video.name:
            return error
    return "no prediction feather was produced"


def _rewrite_project_path(config_path: Path) -> None:
    """Rewrites the extracted configuration's project path to the writable shell it was unpacked into.

    Multi-animal tracking reads the project path from the configuration, so it must point at the extracted shell rather
    than the original training machine's path baked in at export.

    Args:
        config_path: The path of the extracted project configuration to rewrite in place.
    """
    yaml = YAML()
    yaml.default_flow_style = False
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.load(stream)
    config["project_path"] = str(config_path.parent)
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)
