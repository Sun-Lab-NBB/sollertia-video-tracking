"""Provides the budgeted video-subset selection, uniform or balanced across groups, toward a total-frame target."""

import math
import heapq
from random import Random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoSamplingPlan:
    """Describes which videos a budgeted extraction pass samples and how the pass relates to the frame budget.

    Notes:
        The selection only draws from videos that have no extracted frames yet, so repeated passes broaden dataset
        coverage by adding new videos rather than re-clustering ones that are already done. When the videos are grouped,
        ``per_group`` reports each group's coverage before and after the pass.
    """

    selected_videos: tuple[str, ...]
    """The videos chosen for extraction in this pass, drawn from the not-yet-extracted candidates."""
    existing_frame_count: int
    """The number of frames already extracted across the candidate videos before this pass."""
    target_frame_count: int
    """The requested total number of frames the project should hold after extraction."""
    projected_frame_count: int
    """The number of frames the project is expected to hold once the selected videos are extracted."""
    budget_already_met: bool
    """Indicates whether the existing frames already meet the target, so the pass would extract nothing."""
    target_unreachable: bool
    """Indicates whether too few un-extracted videos remain to reach the target, so the pass falls short of it."""
    per_group: tuple[tuple[str, int, int, int, int], ...] = ()
    """The per-group breakdown when grouping is used, as ``(group, existing_frame_count, added_video_count,
    projected_frame_count, available_video_count)`` tuples in canonical group order, or empty when the selection was
    not grouped or the budget was already met. ``available_video_count`` is how many un-extracted videos the group had
    before this pass."""
    always_included_overshoot: bool = False
    """Indicates whether the always-included videos alone exceeded the pass budget, so the projected total overshoots
    the target by the surplus always-included videos."""


def plan_video_sampling(
    videos: list[str],
    extracted_frame_counts: dict[str, int],
    frames_per_video_count: int,
    total_frame_budget: int,
    *,
    groups: dict[str, list[str]] | None = None,
    pinned_videos: tuple[str, ...] = (),
) -> VideoSamplingPlan:
    """Selects a subset of not-yet-extracted videos that grows the project toward a total-frame budget.

    Sums the frames already extracted across the candidate videos and selects just enough additional videos, each
    contributing ``frames_per_video_count`` frames, to reach ``total_frame_budget``. Only videos without extracted
    frames are eligible, so repeated passes broaden dataset coverage instead of re-clustering videos that are already
    done. The realized total rounds up to a whole video, so it may exceed the target by less than one video's worth of
    frames.

    Without ``groups`` the additional videos are drawn uniformly at random. With ``groups`` the additional videos are
    balanced across groups by repeatedly assigning the next video to the group with the fewest projected frames
    (counting frames from prior passes). This equalizes cumulative per-group coverage over repeated passes, though a
    single pass may skip a group when the video budget is smaller than the group count or the group is already
    well-covered or exhausted. In both modes any ``pinned_videos`` are selected first and always included in the draw.

    Args:
        videos: The ordered candidate video paths the pass may sample from.
        extracted_frame_counts: The number of frames already extracted for each candidate video, keyed by path.
        frames_per_video_count: The number of frames each newly sampled video contributes.
        total_frame_budget: The total number of frames the project should hold after extraction.
        groups: A mapping of group to that group's candidate videos, enabling balanced per-group selection. Set to None
            to draw uniformly across all candidates.
        pinned_videos: The videos to include first whenever the pass extracts frames — skipped entirely if the budget is
            already met — selected before the budgeted draw. Duplicates, unknown, and already-extracted entries are
            ignored, so a pin is re-extracted only when the caller has cleared its frames up front (with overwrite).

    Returns:
        A VideoSamplingPlan naming the selected videos and reporting the existing, target, and projected frame counts
        alongside the budget-already-met, unreachable-target, per-group, and always-included-overshoot details.
    """
    existing_frame_count = sum(extracted_frame_counts.get(video, 0) for video in videos)
    unextracted_videos = [video for video in videos if extracted_frame_counts.get(video, 0) == 0]
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

    needed_video_count = math.ceil(remaining_frame_count / frames_per_video_count)
    target_unreachable = needed_video_count > len(unextracted_videos)
    needed_video_count = min(needed_video_count, len(unextracted_videos))

    candidate_set = set(videos)
    eligible_pinned_videos = [
        video
        for video in dict.fromkeys(pinned_videos)
        if video in candidate_set and extracted_frame_counts.get(video, 0) == 0
    ]
    always_included_overshoot = len(eligible_pinned_videos) > needed_video_count

    per_group: tuple[tuple[str, int, int, int, int], ...] = ()
    if groups is not None:
        selected_video_list, per_group = _select_balanced(
            groups=groups,
            extracted_frame_counts=extracted_frame_counts,
            unextracted_videos=unextracted_videos,
            frames_per_video_count=frames_per_video_count,
            needed_video_count=needed_video_count,
            pinned_videos=eligible_pinned_videos,
        )
        selected_videos = tuple(selected_video_list)
    elif eligible_pinned_videos:
        selected_videos = _select_uniform_with_pinned_videos(
            unextracted_videos=unextracted_videos,
            needed_video_count=needed_video_count,
            pinned_videos=eligible_pinned_videos,
        )
    else:
        generator = Random()  # noqa: S311 -- video sampling is not security-sensitive.
        selected_videos = tuple(generator.sample(population=unextracted_videos, k=needed_video_count))

    projected_frame_count = existing_frame_count + len(selected_videos) * frames_per_video_count
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


def _select_balanced(
    groups: dict[str, list[str]],
    extracted_frame_counts: dict[str, int],
    unextracted_videos: list[str],
    frames_per_video_count: int,
    needed_video_count: int,
    pinned_videos: list[str],
) -> tuple[list[str], tuple[tuple[str, int, int, int, int], ...]]:
    """Balances the pass's videos across groups by always extending the group with the fewest projected frames.

    Seeds each group's projected frame count with the frames it already holds from prior passes, honors the pinned
    videos first, then repeatedly assigns the next video to the least-covered group that still has an un-extracted
    video. This ensures that the pass equalizes cumulative per-group coverage. Each group's videos are shuffled
    randomly, so which of an equally-covered group's videos are sampled differs each run.

    Args:
        groups: A mapping of group to that group's candidate videos.
        extracted_frame_counts: The number of frames already extracted for each candidate video, keyed by path.
        unextracted_videos: The not-yet-extracted candidate videos, used to filter each group's eligible videos.
        frames_per_video_count: The number of frames each newly sampled video contributes.
        needed_video_count: The total number of videos to select this pass, including any pinned videos.
        pinned_videos: The eligible pinned videos to always include, already deduplicated.

    Returns:
        A tuple of the selected video paths (pinned first, then the balanced fill) and the per-group breakdown as
        ``(group, existing_frame_count, added_video_count, projected_frame_count, available_video_count)`` tuples in
        canonical group order. ``available_video_count`` is how many un-extracted videos the group had before this pass.
    """
    generator = Random()  # noqa: S311 -- video sampling is not security-sensitive.
    unextracted_video_set = set(unextracted_videos)
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
        eligible_videos = sorted(video for video in members if video in unextracted_video_set)
        generator.shuffle(x=eligible_videos)
        available_videos_by_group[group_key] = eligible_videos
        available_video_counts_by_group[group_key] = len(eligible_videos)

    projected_frames_by_group = dict(existing_frames_by_group)
    selected_videos: list[str] = []
    seen: set[str] = set()

    for video in pinned_videos:
        if video in seen:
            continue
        selected_videos.append(video)
        seen.add(video)
        # An eligible pin is always included; it only participates in balancing when it maps to a known group.
        pin_group_key = group_of.get(video)
        if pin_group_key is not None:
            if video in available_videos_by_group[pin_group_key]:
                available_videos_by_group[pin_group_key].remove(video)
            projected_frames_by_group[pin_group_key] += frames_per_video_count

    remaining_video_budget = max(0, needed_video_count - len(selected_videos))
    heap = [
        (projected_frames_by_group[group_key], group_key)
        for group_key in group_keys
        if available_videos_by_group[group_key]
    ]
    heapq.heapify(heap)
    while remaining_video_budget and heap:
        _, group_key = heapq.heappop(heap)
        # The group's videos were shuffled when seeded, so popping the tail is a uniform-random draw at O(1) and
        # avoids the O(n) left shift of pop(0).
        video = available_videos_by_group[group_key].pop()
        selected_videos.append(video)
        seen.add(video)
        projected_frames_by_group[group_key] += frames_per_video_count
        remaining_video_budget -= 1
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


def _select_uniform_with_pinned_videos(
    unextracted_videos: list[str], needed_video_count: int, pinned_videos: list[str]
) -> tuple[str, ...]:
    """Selects the pinned videos and fills the remaining budget with a uniform random draw over the rest.

    Args:
        unextracted_videos: The not-yet-extracted candidate videos.
        needed_video_count: The number of videos the budget calls for this pass.
        pinned_videos: The eligible pinned videos to always include, already deduplicated.

    Returns:
        The selected videos, with the pinned ones first.
    """
    generator = Random()  # noqa: S311 -- video sampling is not security-sensitive.
    selected_videos = list(pinned_videos)
    pinned_video_set = set(pinned_videos)
    fill_candidate_videos = [video for video in unextracted_videos if video not in pinned_video_set]
    fill_video_count = min(max(0, needed_video_count - len(selected_videos)), len(fill_candidate_videos))
    selected_videos.extend(generator.sample(population=fill_candidate_videos, k=fill_video_count))
    return tuple(selected_videos)
