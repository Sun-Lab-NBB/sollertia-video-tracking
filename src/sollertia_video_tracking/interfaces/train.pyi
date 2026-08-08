from pathlib import Path

from ..hardware import warn as warn
from ..training import (
    Toggle as Toggle,
    AmpMode as AmpMode,
    DeviceType as DeviceType,
    MultiGpuStrategy as MultiGpuStrategy,
    TrainingFailedError as TrainingFailedError,
    TrainingInterruptedError as TrainingInterruptedError,
    train_model as train_model,
    detect_fixed_input_size as detect_fixed_input_size,
    resolve_optimization_profile as resolve_optimization_profile,
)

_CONTEXT_SETTINGS: dict[str, int]

def train_command(
    config_path: Path,
    shuffle: int,
    epochs: int | None,
    batch_size: int | None,
    save_epochs: int | None,
    display_iterations: int | None,
    maximum_snapshots: int | None,
    snapshot_path: Path | None,
    detector_path: Path | None,
    detector_batch_size: int | None,
    detector_epochs: int | None,
    detector_save_epochs: int | None,
    device: str,
    gpus: str | None,
    multi_gpu: str,
    amp: str,
    tf32: str,
    cudnn_benchmark: str,
    compile_model: str,
    dataloader_workers: int,
    pin_memory: str,
    evaluation_batch_size: int,
    evaluation_confidence_cutoff: float | None,
    *,
    load_head_weights: bool,
    evaluate: bool,
    progress: bool,
) -> None: ...
