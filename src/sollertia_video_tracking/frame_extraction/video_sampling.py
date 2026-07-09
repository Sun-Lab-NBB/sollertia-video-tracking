"""Provides the budgeted video-subset selection, uniform or balanced across groups, toward a total-frame target."""

import heapq
from random import Random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoSamplingPlan:
    """Describes which videos a budgeted extraction pass tops up and how the pass relates to the frame budget.

    Notes:
        Each video contributes only up to its per-video ceiling (``frames_per_video_count``), so a partly-extracted
        video is topped up to that ceiling rather than gaining another full batch. The pass prefers not-yet-extracted
        videos and falls back to below-ceiling extracted ones, so coverage grows before existing videos are deepened.
        When the videos are grouped, ``per_group`` reports each group's coverage before and after the pass.
    """

    selected_videos: tuple[str, ...]
    """The videos chosen this pass, preferring not-yet-extracted candidates over below-ceiling ones."""
    existing_frame_count: int
    """The number of frames already extracted across the candidate videos before this pass."""
    target_frame_count: int
    """The requested total number of frames the project should hold after extraction."""
    projected_frame_count: int
    """The number of frames the project is expected to hold once the selected videos are topped up to the ceiling."""
    budget_already_met: bool
    """Indicates whether the existing frames already meet the target, so the pass would extract nothing."""
    target_unreachable: bool
    """Indicates whether even topping every eligible video up to its per-video ceiling would fall short of the target,
    so the caller must report the shortfall rather than extract."""
    per_group: tuple[tuple[str, int, int, int, int], ...] = ()
    """The per-group breakdown when grouping is used, as ``(group, existing_frame_count, added_video_count,
    projected_frame_count, available_video_count)`` tuples in canonical group order, or empty when the selection was
    not grouped or the budget was already met. ``available_video_count`` is how many below-ceiling videos the group had
    before this pass."""
    always_included_overshoot: bool = False
    """Indicates whether the pinned videos alone contribute more than the remaining budget, so the projected total
    overshoots the target by the surplus pinned videos."""


def plan_video_sampling(
    videos: list[str],
    extracted_frame_counts: dict[str, int],
    frames_per_video_count: int,
    total_frame_budget: int,
    *,
    groups: dict[str, list[str]] | None = None,
    pinned_videos: tuple[str, ...] = (),
) -> VideoSamplingPlan:
    """Selects a subset of videos that grows the project toward a total-frame budget, topping each up to its ceiling.

    Sums the frames already extracted across the candidate videos and selects just enough additional videos to reach
    ``total_frame_budget``, where each selected video contributes only its remaining capacity toward the per-video
    ceiling ``frames_per_video_count``. A not-yet-extracted video contributes a full ceiling's worth, while a
    partly-extracted one contributes only the frames needed to top it up; videos already at the ceiling contribute
    nothing and are skipped. The selection prefers not-yet-extracted videos and falls back to below-ceiling extracted
    ones, so repeated passes broaden coverage before deepening existing videos.

    Without ``groups`` the additional videos are drawn uniformly at random within each preference tier. With ``groups``
    the additional videos are balanced across groups by repeatedly assigning the next video to the group with the fewest
    projected frames, still preferring each group's not-yet-extracted videos first. This equalizes cumulative per-group
    coverage over repeated passes. In both modes any ``pinned_videos`` are selected first and always included.

    Args:
        videos: The ordered candidate video paths the pass may sample from.
        extracted_frame_counts: The number of frames already extracted for each candidate video, keyed by path.
        frames_per_video_count: The per-video ceiling; each selected video is topped up to at most this many frames.
        total_frame_budget: The total number of frames the project should hold after extraction.
        groups: A mapping of group to that group's candidate videos, enabling balanced per-group selection. Set to None
            to draw uniformly across all candidates.
        pinned_videos: The videos to include first whenever the pass extracts frames, skipped entirely if the budget is
            already met. Duplicates and unknown entries are ignored. A pin already at the ceiling contributes no frames
            but is still returned, so the caller can re-roll it under overwrite.

    Returns:
        A VideoSamplingPlan naming the selected videos and reporting the existing, target, and projected frame counts
        alongside the budget-already-met, unreachable-target, per-group, and always-included-overshoot details.
    """
    existing_frame_count = sum(extracted_frame_counts.get(video, 0) for video in videos)
    remaining_frame_count = total_frame_budget - existing_frame_count

    if remaining_frame_count <= 0:
        return VideoSamplingPlan(
            selected_videos=(),
            existing_frame_count=existing_frame_count,
            target_frame_count=total_frame_budget,
            projected_frame_count=existing_frame_count,
            budget_already_met=True,
            target_unreachable=False,
        )

    # Each video can still contribute frames up to its per-video ceiling; a video already at the ceiling contributes 0.
    capacity_of = {video: max(0, frames_per_video_count - extracted_frame_counts.get(video, 0)) for video in videos}
    target_unreachable = sum(capacity_of.values()) < remaining_frame_count

    candidate_set = set(videos)
    eligible_pinned_videos = [video for video in dict.fromkeys(pinned_videos) if video in candidate_set]

    per_group: tuple[tuple[str, int, int, int, int], ...] = ()
    if groups is not None:
        selected_video_list, per_group = _select_balanced(
            groups=groups,
            extracted_frame_counts=extracted_frame_counts,
            capacity_of=capacity_of,
            remaining_frame_count=remaining_frame_count,
            pinned_videos=eligible_pinned_videos,
        )
        selected_videos = tuple(selected_video_list)
    else:
        selected_videos = _select_uniform(
            videos=videos,
            extracted_frame_counts=extracted_frame_counts,
            capacity_of=capacity_of,
            remaining_frame_count=remaining_frame_count,
            pinned_videos=eligible_pinned_videos,
        )

    projected_frame_count = existing_frame_count + sum(capacity_of[video] for video in selected_videos)
    always_included_overshoot = sum(capacity_of[video] for video in eligible_pinned_videos) > remaining_frame_count
    return VideoSamplingPlan(
        selected_videos=selected_videos,
        existing_frame_count=existing_frame_count,
        target_frame_count=total_frame_budget,
        projected_frame_count=projected_frame_count,
        budget_already_met=False,
        target_unreachable=target_unreachable,
        per_group=per_group,
        always_included_overshoot=always_included_overshoot,
    )


def _select_uniform(
    videos: list[str],
    extracted_frame_counts: dict[str, int],
    capacity_of: dict[str, int],
    remaining_frame_count: int,
    pinned_videos: list[str],
) -> tuple[str, ...]:
    """Selects the pins, then fills the budget from not-yet-extracted videos before below-ceiling extracted ones.

    Includes every pin first, then draws uniformly at random from the not-yet-extracted candidates and, once those run
    out, from the below-ceiling extracted candidates, accumulating each video's remaining capacity until the budget is
    reached.

    Args:
        videos: The ordered candidate video paths.
        extracted_frame_counts: The number of frames already extracted for each candidate video, keyed by path.
        capacity_of: The remaining capacity toward the per-video ceiling for each candidate, keyed by path.
        remaining_frame_count: The number of frames still needed to reach the budget after counting existing frames.
        pinned_videos: The eligible pinned videos to always include, already deduplicated.

    Returns:
        The selected videos, with the pinned ones first.
    """
    generator = Random()  # noqa: S311 -- video sampling is not security-sensitive.
    selected_videos = list(pinned_videos)
    seen = set(pinned_videos)
    accumulated_frame_count = sum(capacity_of[video] for video in pinned_videos)

    unextracted_videos = [video for video in videos if video not in seen and extracted_frame_counts.get(video, 0) == 0]
    below_ceiling_videos = [
        video
        for video in videos
        if video not in seen and extracted_frame_counts.get(video, 0) > 0 and capacity_of[video] > 0
    ]
    generator.shuffle(x=unextracted_videos)
    generator.shuffle(x=below_ceiling_videos)

    for tier in (unextracted_videos, below_ceiling_videos):
        for video in tier:
            if accumulated_frame_count >= remaining_frame_count:
                return tuple(selected_videos)
            selected_videos.append(video)
            accumulated_frame_count += capacity_of[video]
    return tuple(selected_videos)


def _select_balanced(
    groups: dict[str, list[str]],
    extracted_frame_counts: dict[str, int],
    capacity_of: dict[str, int],
    remaining_frame_count: int,
    pinned_videos: list[str],
) -> tuple[list[str], tuple[tuple[str, int, int, int, int], ...]]:
    """Balances the pass's videos across groups by always extending the group with the fewest projected frames.

    Seeds each group's projected frame count with the frames it already holds from prior passes, honors the pinned
    videos first, then repeatedly assigns the next video to the least-covered group that still has a below-ceiling
    video. It prefers that group's not-yet-extracted videos before its below-ceiling extracted ones. This equalizes
    cumulative per-group coverage. Each group's videos are shuffled randomly, so which of an equally-covered group's
    videos are sampled differs each run.

    Args:
        groups: A mapping of group to that group's candidate videos.
        extracted_frame_counts: The number of frames already extracted for each candidate video, keyed by path.
        capacity_of: The remaining capacity toward the per-video ceiling for each candidate, keyed by path.
        remaining_frame_count: The number of frames still needed to reach the budget after counting existing frames.
        pinned_videos: The eligible pinned videos to always include, already deduplicated.

    Returns:
        A tuple of the selected video paths (pinned first, then the balanced fill) and the per-group breakdown as
        ``(group, existing_frame_count, added_video_count, projected_frame_count, available_video_count)`` tuples in
        canonical group order. ``available_video_count`` is the group's below-ceiling video count before the pass.
    """
    generator = Random()  # noqa: S311 -- video sampling is not security-sensitive.
    group_keys = sorted(groups)

    group_of: dict[str, str] = {}
    existing_frames_by_group: dict[str, int] = {}
    available_videos_by_group: dict[str, list[str]] = {}
    available_video_counts_by_group: dict[str, int] = {}
    for group_key in group_keys:
        members = groups[group_key]
        for video in members:
            group_of[video] = group_key
        existing_frames_by_group[group_key] = sum(extracted_frame_counts.get(video, 0) for video in members)
        unextracted_videos = [video for video in members if extracted_frame_counts.get(video, 0) == 0]
        below_ceiling_videos = [
            video for video in members if extracted_frame_counts.get(video, 0) > 0 and capacity_of[video] > 0
        ]
        generator.shuffle(x=unextracted_videos)
        generator.shuffle(x=below_ceiling_videos)
        # pop() draws from the tail, so not-yet-extracted videos sit last and are used before the extracted ones.
        available_videos_by_group[group_key] = below_ceiling_videos + unextracted_videos
        available_video_counts_by_group[group_key] = len(available_videos_by_group[group_key])

    projected_frames_by_group = dict(existing_frames_by_group)
    selected_videos: list[str] = []
    seen: set[str] = set()
    accumulated_frame_count = 0

    for video in pinned_videos:
        if video in seen:
            continue
        selected_videos.append(video)
        seen.add(video)
        accumulated_frame_count += capacity_of[video]
        # An eligible pin is always included; it only participates in balancing when it maps to a known group.
        pin_group_key = group_of.get(video)
        if pin_group_key is not None:
            if video in available_videos_by_group[pin_group_key]:
                available_videos_by_group[pin_group_key].remove(video)
            projected_frames_by_group[pin_group_key] += capacity_of[video]

    heap = [
        (projected_frames_by_group[group_key], group_key)
        for group_key in group_keys
        if available_videos_by_group[group_key]
    ]
    heapq.heapify(heap)
    while accumulated_frame_count < remaining_frame_count and heap:
        _, group_key = heapq.heappop(heap)
        # The group's videos were shuffled when seeded, so popping the tail is a uniform-random draw at O(1) and
        # avoids the O(n) left shift of pop(0), while still favoring the not-yet-extracted videos placed at the tail.
        video = available_videos_by_group[group_key].pop()
        selected_videos.append(video)
        seen.add(video)
        accumulated_frame_count += capacity_of[video]
        projected_frames_by_group[group_key] += capacity_of[video]
        if available_videos_by_group[group_key]:
            heapq.heappush(heap, (projected_frames_by_group[group_key], group_key))

    per_group = tuple(
        (
            group_key,
            existing_frames_by_group[group_key],
            sum(1 for video in selected_videos if group_of.get(video) == group_key),
            projected_frames_by_group[group_key],
            available_video_counts_by_group[group_key],
        )
        for group_key in group_keys
    )
    return selected_videos, per_group
