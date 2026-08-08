from pathlib import Path

from ..hardware import warn as warn
from ..inference import (
    Toggle as Toggle,
    AmpMode as AmpMode,
    DeviceType as DeviceType,
    PipelineFailedError as PipelineFailedError,
    PipelineInterruptedError as PipelineInterruptedError,
    run_inference as run_inference,
    resolve_project_videos as resolve_project_videos,
    detect_fixed_input_size as detect_fixed_input_size,
    discover_directory_videos as discover_directory_videos,
    resolve_inference_profile as resolve_inference_profile,
    ensure_unique_prediction_targets as ensure_unique_prediction_targets,
)

_CONTEXT_SETTINGS: dict[str, int]
_CROP_FIELD_COUNT: int

def infer_command(
    config_path: Path,
    videos: tuple[Path, ...],
    videos_directory: Path | None,
    output: tuple[Path, ...],
    shuffle: int,
    snapshot_index: int | None,
    detector_snapshot_index: int | None,
    batch_size: int | None,
    detector_batch_size: int | None,
    crop: tuple[str, ...],
    device: str,
    gpus: str | None,
    gpu_processes: int,
    chunks: int,
    cpu_workers: int,
    cpu_threads_per_worker: int,
    amp: str,
    tf32: str,
    cudnn_benchmark: str,
    channels_last: str,
    compile_model: str,
    *,
    progress: bool,
) -> None: ...
def _parse_crop_option(value: str) -> tuple[int, int, int, int]: ...
def _resolve_crop_override(
    crop: tuple[str, ...], video_count: int, selection_source: str | None
) -> list[tuple[int, int, int, int]] | None: ...
def _resolve_output_override(
    output: tuple[Path, ...], video_count: int, selection_source: str | None
) -> list[Path] | None: ...
