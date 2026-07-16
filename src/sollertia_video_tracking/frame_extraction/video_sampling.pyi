from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class VideoSamplingPlan:
    selected_videos: tuple[str, ...]
    existing_frame_count: int
    target_frame_count: int
    projected_frame_count: int
    budget_already_met: bool
    target_unreachable: bool
    per_group: tuple[tuple[str, int, int, int, int], ...] = ...
    always_included_overshoot: bool = ...

def plan_video_sampling(
    videos: list[str],
    extracted_frame_counts: dict[str, int],
    frames_per_video_count: int,
    total_frame_budget: int,
    *,
    groups: dict[str, list[str]] | None = None,
    pinned_videos: tuple[str, ...] = (),
) -> VideoSamplingPlan: ...
def _select_uniform(
    videos: list[str],
    extracted_frame_counts: dict[str, int],
    capacity_of: dict[str, int],
    remaining_frame_count: int,
    pinned_videos: list[str],
) -> tuple[str, ...]: ...
def _select_balanced(
    groups: dict[str, list[str]],
    extracted_frame_counts: dict[str, int],
    capacity_of: dict[str, int],
    remaining_frame_count: int,
    pinned_videos: list[str],
) -> tuple[list[str], tuple[tuple[str, int, int, int, int], ...]]: ...
