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

    selected: tuple[str, ...]
    """The videos chosen for extraction in this pass, drawn from the not-yet-extracted candidates."""
    existing_frames: int
    """The number of frames already extracted across the candidate videos before this pass."""
    target_frames: int
    """The requested total number of frames the project should hold after extraction."""
    projected_frames: int
    """The number of frames the project is expected to hold once the selected videos are extracted."""
    no_growth: bool
    """Indicates whether the existing frames already meet the target, so the pass would extract nothing."""
    target_unreachable: bool
    """Indicates whether too few un-extracted videos remain to reach the target, so the pass falls short of it."""
    per_group: tuple[tuple[str, int, int, int, int], ...] = ()
    """The per-group breakdown when grouping is used, as ``(group, existing_frames, added_videos, projected_frames,
    available_videos)`` tuples in canonical group order, or empty when the selection was not grouped.
    ``available_videos`` is how many un-extracted videos the group had before this pass."""
    pinned_overshoot: bool = False
    """Indicates whether the explicitly pinned videos alone exceeded the pass budget, so the projected total overshoots
    the target by the surplus pinned videos."""


def plan_video_sampling(
    videos: list[str],
    extracted_frame_counts: dict[str, int],
    frames_per_video: int,
    total_frames: int,
    seed: int | None,
    *,
    groups: dict[str, list[str]] | None = None,
    pinned: tuple[str, ...] = (),
) -> VideoSamplingPlan:
    """Selects a subset of not-yet-extracted videos that grows the project toward a total-frame budget.

    Sums the frames already extracted across the candidate videos and selects just enough additional videos, each
    contributing ``frames_per_video`` frames, to reach ``total_frames``. Only videos without extracted frames are
    eligible, so repeated passes broaden dataset coverage instead of re-clustering videos that are already done. The
    realized total rounds up to a whole video, so it may exceed the target by less than one video's worth of frames.

    Without ``groups`` the additional videos are drawn uniformly at random. With ``groups`` the additional videos are
    balanced across groups by repeatedly assigning the next video to the group with the fewest projected frames
    (counting frames from prior passes). This ensures that every group is represented and coverage evens out over
    repeated passes. In both modes any ``pinned`` videos are selected first and always included.

    Args:
        videos: The ordered candidate video paths the pass may sample from.
        extracted_frame_counts: The number of frames already extracted for each candidate video, keyed by path.
        frames_per_video: The number of frames each newly sampled video contributes.
        total_frames: The total number of frames the project should hold after extraction.
        seed: The seed for the random draw, or None to draw nondeterministically. With grouping, the seed also fixes
            the random choice of which of a group's videos are sampled.
        groups: A mapping of group to that group's candidate videos, enabling balanced per-group selection. Set to None
            to draw uniformly across all candidates.
        pinned: The videos to always include, selected before the budgeted draw. Duplicates and already-extracted or
            unknown entries are ignored.

    Returns:
        A VideoSamplingPlan naming the selected videos and reporting the existing, target, and projected frame counts
        alongside the no-growth, unreachable-target, per-group, and pinned-overshoot details.
    """
    existing_frames = sum(extracted_frame_counts.get(video, 0) for video in videos)
    un_extracted = [video for video in videos if extracted_frame_counts.get(video, 0) == 0]
    remaining = total_frames - existing_frames

    if remaining <= 0:
        return VideoSamplingPlan(
            selected=(),
            existing_frames=existing_frames,
            target_frames=total_frames,
            projected_frames=existing_frames,
            no_growth=True,
            target_unreachable=False,
        )

    needed = math.ceil(remaining / frames_per_video)
    target_unreachable = needed > len(un_extracted)
    needed = min(needed, len(un_extracted))

    candidate_set = set(videos)
    pinned_eligible = [
        video for video in dict.fromkeys(pinned) if video in candidate_set and extracted_frame_counts.get(video, 0) == 0
    ]
    pinned_overshoot = len(pinned_eligible) > needed

    per_group: tuple[tuple[str, int, int, int, int], ...] = ()
    if groups is not None:
        selected_list, per_group = _select_balanced(
            groups=groups,
            extracted_frame_counts=extracted_frame_counts,
            un_extracted=un_extracted,
            frames_per_video=frames_per_video,
            needed=needed,
            pinned=pinned_eligible,
            seed=seed,
        )
        selected = tuple(selected_list)
    elif pinned_eligible:
        selected = _select_uniform_with_pins(
            un_extracted=un_extracted, needed=needed, pinned=pinned_eligible, seed=seed
        )
    else:
        # A single uniform draw over the not-yet-extracted candidates.
        generator = Random(seed)  # noqa: S311 -- video sampling is not security-sensitive.
        selected = tuple(generator.sample(un_extracted, needed))

    projected_frames = existing_frames + len(selected) * frames_per_video
    return VideoSamplingPlan(
        selected=selected,
        existing_frames=existing_frames,
        target_frames=total_frames,
        projected_frames=projected_frames,
        no_growth=False,
        target_unreachable=target_unreachable,
        per_group=per_group,
        pinned_overshoot=pinned_overshoot,
    )


def _select_uniform_with_pins(
    un_extracted: list[str], needed: int, pinned: list[str], seed: int | None
) -> tuple[str, ...]:
    """Selects the pinned videos and fills the remaining budget with a uniform random draw over the rest.

    Args:
        un_extracted: The not-yet-extracted candidate videos.
        needed: The number of videos the budget calls for this pass.
        pinned: The eligible pinned videos to always include, already deduplicated.
        seed: The seed for the random fill draw, or None for a nondeterministic draw.

    Returns:
        The selected videos, the pinned ones first, in a tuple.
    """
    generator = Random(seed)  # noqa: S311 -- video sampling is not security-sensitive.
    selected = list(pinned)
    pinned_set = set(pinned)
    fill_pool = [video for video in un_extracted if video not in pinned_set]
    fill_count = min(max(0, needed - len(selected)), len(fill_pool))
    selected.extend(generator.sample(fill_pool, fill_count))
    return tuple(selected)


def _select_balanced(
    groups: dict[str, list[str]],
    extracted_frame_counts: dict[str, int],
    un_extracted: list[str],
    frames_per_video: int,
    needed: int,
    pinned: list[str],
    seed: int | None,
) -> tuple[list[str], tuple[tuple[str, int, int, int, int], ...]]:
    """Balances the pass's videos across groups by always extending the group with the fewest projected frames.

    Seeds each group's projected frame count with the frames it already holds from prior passes, honors the pinned
    videos first, then repeatedly assigns the next video to the least-covered group that still has an un-extracted
    video. This ensures that the pass equalizes cumulative per-group coverage. Determinism is fixed by a canonical
    group order, a seeded shuffle of each group's videos, and a group-name tiebreak, so a fixed seed reproduces
    the selection.

    Args:
        groups: A mapping of group to that group's candidate videos.
        extracted_frame_counts: The number of frames already extracted for each candidate video, keyed by path.
        un_extracted: The not-yet-extracted candidate videos, used to filter each group's eligible videos.
        frames_per_video: The number of frames each newly sampled video contributes.
        needed: The total number of videos to select this pass, including any pinned videos.
        pinned: The eligible pinned videos to always include, already deduplicated.
        seed: The seed for the per-group video shuffle, or None for a nondeterministic shuffle.

    Returns:
        A tuple of the selected video paths (pinned first, then the balanced fill) and the per-group breakdown as
        ``(group, existing_frames, added_videos, projected_frames, available_videos)`` tuples in canonical group order,
        where ``available_videos`` is how many un-extracted videos the group had before this pass.
    """
    generator = Random(seed)  # noqa: S311 -- video sampling is not security-sensitive.
    un_extracted_set = set(un_extracted)
    group_keys = sorted(groups)

    group_of: dict[str, str] = {}
    existing: dict[str, int] = {}
    available: dict[str, list[str]] = {}
    available_at_start: dict[str, int] = {}
    for group_key in group_keys:
        members = groups[group_key]
        for video in members:
            group_of[video] = group_key
        existing[group_key] = sum(extracted_frame_counts.get(video, 0) for video in members)
        eligible = sorted(video for video in members if video in un_extracted_set)
        generator.shuffle(eligible)
        available[group_key] = eligible
        available_at_start[group_key] = len(eligible)

    projected = dict(existing)
    selected: list[str] = []
    seen: set[str] = set()

    for video in pinned:
        if video in seen:
            continue
        selected.append(video)
        seen.add(video)
        # An eligible pin is always included; it only participates in balancing when it maps to a known group.
        pin_group = group_of.get(video)
        if pin_group is not None:
            if video in available[pin_group]:
                available[pin_group].remove(video)
            projected[pin_group] += frames_per_video

    budget = max(0, needed - len(selected))
    heap = [(projected[group_key], group_key) for group_key in group_keys if available[group_key]]
    heapq.heapify(heap)
    while budget > 0 and heap:
        _, group_key = heapq.heappop(heap)
        video = available[group_key].pop(0)
        selected.append(video)
        seen.add(video)
        projected[group_key] += frames_per_video
        budget -= 1
        if available[group_key]:
            heapq.heappush(heap, (projected[group_key], group_key))

    per_group = tuple(
        (
            group_key,
            existing[group_key],
            sum(1 for video in selected if group_of.get(video) == group_key),
            projected[group_key],
            available_at_start[group_key],
        )
        for group_key in group_keys
    )
    return selected, per_group
