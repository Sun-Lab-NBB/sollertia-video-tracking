from typing import Any
from pathlib import Path
import contextlib
from dataclasses import dataclass
from collections.abc import Iterator, Sequence

from .runners import patch_dlc_runner_builders as patch_dlc_runner_builders
from .optimization import (
    InferenceProfile as InferenceProfile,
    apply_runtime_optimizations as apply_runtime_optimizations,
)
from ..frame_extraction import (
    AggregateBar as AggregateBar,
    plan_core_allocation as plan_core_allocation,
    make_progress_reporter as make_progress_reporter,
)

_STOCK_ACCELERATION_DISABLED: dict[str, dict[str, bool]]
_CROP_FIELD_COUNT: int
_RESULT_POLL_TIMEOUT_SECONDS: float
_BAR_JOIN_TIMEOUT_SECONDS: int

@dataclass(frozen=True, slots=True)
class InferenceSummary:
    config: Path
    video_count: int
    destinations: tuple[Path, ...] | None
    device: str
    workers: int
    precision: str
    outputs: tuple[Path, ...]
    failures: tuple[tuple[str, str], ...]
    def describe(self) -> str: ...

@dataclass(frozen=True, slots=True)
class _Slot:
    device: str
    cores: tuple[int, ...] | None

@dataclass(frozen=True, slots=True)
class _InferenceLaunch:
    config: Path
    shuffle: int
    snapshot_index: int | None
    detector_snapshot_index: int | None
    profile: InferenceProfile
    batch_size: int | None
    detector_batch_size: int | None
    display_progress: bool
    video_queue: Any
    progress_queue: Any
    results_queue: Any

def resolve_project_videos(config: str | Path) -> list[Path]: ...
def detect_fixed_input_size(
    config: str | Path, videos: list[str | Path], crop_override: Sequence[tuple[int, int, int, int]] | None = None
) -> bool: ...
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
) -> InferenceSummary: ...
def _resolve_input_size(project_config: dict[str, Any], video: Path) -> tuple[int, int] | None: ...
def _probe_frame_size(video: Path) -> tuple[int, int] | None: ...
def _parse_crop(crop: str | None) -> list[int] | None: ...
def _resolve_video_cropping(project_config: dict[str, Any], video: str) -> list[int] | None: ...
def _describe_precision(profile: InferenceProfile) -> str: ...
def _build_slots(profile: InferenceProfile, video_count: int) -> list[_Slot]: ...
def _usable_cpu_cores(profile: InferenceProfile) -> int: ...
def _probe_frame_count(video: Path) -> int: ...
def _collect_results(results_queue: Any, video_paths: list[Path]) -> tuple[dict[int, Path], list[tuple[str, str]]]: ...
@contextlib.contextmanager
def _suppress_stdout(*, active: bool) -> Iterator[None]: ...
def _run_inference_worker(slot: _Slot, launch: _InferenceLaunch) -> None: ...
def _analyze_one_video(
    slot: _Slot, launch: _InferenceLaunch, item: tuple[int, str, int, list[int] | None, str | None]
) -> None: ...
def _resolve_output(video: str, scorer: str, destination: Path) -> Path | None: ...
