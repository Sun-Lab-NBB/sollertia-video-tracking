from pathlib import Path
from dataclasses import dataclass

import click
from _typeshed import Incomplete

from ..hardware import warn as warn
from ..frame_extraction import (
    TrackingMethod as TrackingMethod,
    OutlierAlgorithm as OutlierAlgorithm,
    ExtractionAlgorithm as ExtractionAlgorithm,
    PipelineFailedError as PipelineFailedError,
    PipelineInterruptedError as PipelineInterruptedError,
    purge_labeled_data as purge_labeled_data,
    extract_frames_kmeans as extract_frames_kmeans,
    summarize_refinement_status as summarize_refinement_status,
    extract_outlier_frames_parallel as extract_outlier_frames_parallel,
)

_CONTEXT_SETTINGS: dict[str, int]

@dataclass(frozen=True, slots=True)
class _SharedExtractionParameters:
    config_path: Path | None
    worker_count: int
    cores_per_worker: int
    frames_per_video: int
    clustering_stride: int
    clustering_resize_width: int
    cluster_in_color: bool
    display_progress: bool
    videos: tuple[Path, ...]
    overwrite: bool
    reset: bool
    def require_config_path(self) -> Path: ...

_pass_shared_parameters: Incomplete

@click.pass_context
def extract_group(
    context: click.Context,
    config_path: Path | None,
    workers: int,
    cores: int,
    frames_per_video: int,
    clustering_stride: int,
    clustering_resize_width: int,
    videos: tuple[Path, ...],
    *,
    color: bool,
    progress: bool,
    overwrite: bool,
    reset: bool,
) -> None: ...
@_pass_shared_parameters
def frames_command(
    shared: _SharedExtractionParameters,
    total_frames: int,
    group_regex: str | None,
    *,
    balance_groups: bool,
    exclusive: bool,
) -> None: ...
@_pass_shared_parameters
def outliers_command(
    shared: _SharedExtractionParameters,
    outlier_algorithm: str,
    extraction_algorithm: str,
    shuffle: int,
    pixel_distance_threshold: float,
    minimum_confidence: float,
    comparison_bodyparts: tuple[str, ...],
    frame_index: tuple[int, ...],
    autoregressive_degree: int,
    moving_average_degree: int,
    tracking_method: str | None,
    fit_workers: int,
    *,
    save_labeled: bool,
) -> None: ...
@_pass_shared_parameters
def purge_command(shared: _SharedExtractionParameters, *, yes: bool) -> None: ...
@_pass_shared_parameters
def pending_command(shared: _SharedExtractionParameters) -> None: ...
