"""Provides the budgeted video-subset selection, uniform or balanced across groups, toward a total-frame target."""

import math
import heapq
from random import Random
from dataclasses import dataclass

OUTLIER_SELECTION_TIERS: tuple[str, ...] = ("no-frames", "raw-only", "has-outlier")
"""The outlier-sampling priority tiers, highest priority first: videos with no frames, then videos with only raw
frames, then videos that already carry outlier frames. A budgeted pass fills its frame budget from the earliest tier
that still has videos before descending to the next."""


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
    not grouped. ``available_video_count`` is how many un-extracted videos the group had before this pass."""
    always_included_overshoot: bool = False
    """Indicates whether the always-included videos alone exceeded the pass budget, so the projected total overshoots
    the target by the surplus always-included videos."""


def plan_video_sampling(
    videos: list[str],
    extracted_frame_counts: dict[str, int],
    frames_per_video_count: int,
    total_frame_budget: int,
    random_seed: int | None,
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
    (counting frames from prior passes). This ensures that every group is represented and coverage evens out over
    repeated passes. In both modes any ``pinned_videos`` are selected first and always included in the draw.

    Args:
        videos: The ordered candidate video paths the pass may sample from.
        extracted_frame_counts: The number of frames already extracted for each candidate video, keyed by path.
        frames_per_video_count: The number of frames each newly sampled video contributes.
        total_frame_budget: The total number of frames the project should hold after extraction.
        random_seed: The seed for the random draw, or None to draw nondeterministically. With grouping, the seed also
            fixes the random choice of which of a group's videos are sampled.
        groups: A mapping of group to that group's candidate videos, enabling balanced per-group selection. Set to None
            to draw uniformly across all candidates.
        pinned_videos: The videos to include whenever the pass extracts frames — they are skipped entirely if the
            budget is already met — selected before the budgeted draw. Duplicates and already-extracted or unknown
            entries are ignored.

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
            random_seed=random_seed,
        )
        selected_videos = tuple(selected_video_list)
    elif eligible_pinned_videos:
        selected_videos = _select_uniform_with_pins(
            unextracted_videos=unextracted_videos,
            needed_video_count=needed_video_count,
            pinned_videos=eligible_pinned_videos,
            random_seed=random_seed,
        )
    else:
        generator = Random(random_seed)  # noqa: S311 -- video sampling is not security-sensitive.
        selected_videos = tuple(generator.sample(unextracted_videos, needed_video_count))

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


def _select_uniform_with_pins(
    unextracted_videos: list[str], needed_video_count: int, pinned_videos: list[str], random_seed: int | None
) -> tuple[str, ...]:
    """Selects the pinned videos and fills the remaining budget with a uniform random draw over the rest.

    Args:
        unextracted_videos: The not-yet-extracted candidate videos.
        needed_video_count: The number of videos the budget calls for this pass.
        pinned_videos: The eligible pinned videos to always include, already deduplicated.
        random_seed: The seed for the random fill draw, or None for a nondeterministic draw.

    Returns:
        The selected videos, the pinned ones first, in a tuple.
    """
    generator = Random(random_seed)  # noqa: S311 -- video sampling is not security-sensitive.
    selected_videos = list(pinned_videos)
    pinned_video_set = set(pinned_videos)
    fill_candidate_videos = [video for video in unextracted_videos if video not in pinned_video_set]
    fill_video_count = min(max(0, needed_video_count - len(selected_videos)), len(fill_candidate_videos))
    selected_videos.extend(generator.sample(fill_candidate_videos, fill_video_count))
    return tuple(selected_videos)


def _select_balanced(
    groups: dict[str, list[str]],
    extracted_frame_counts: dict[str, int],
    unextracted_videos: list[str],
    frames_per_video_count: int,
    needed_video_count: int,
    pinned_videos: list[str],
    random_seed: int | None,
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
        unextracted_videos: The not-yet-extracted candidate videos, used to filter each group's eligible videos.
        frames_per_video_count: The number of frames each newly sampled video contributes.
        needed_video_count: The total number of videos to select this pass, including any pinned videos.
        pinned_videos: The eligible pinned videos to always include, already deduplicated.
        random_seed: The seed for the per-group video shuffle, or None for a nondeterministic shuffle.

    Returns:
        A tuple of the selected video paths (pinned first, then the balanced fill) and the per-group breakdown as
        ``(group, existing_frame_count, added_video_count, projected_frame_count, available_video_count)`` tuples in
        canonical group order, where ``available_video_count`` is how many un-extracted videos the group had before
        this pass.
    """
    generator = Random(random_seed)  # noqa: S311 -- video sampling is not security-sensitive.
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
        generator.shuffle(eligible_videos)
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
    while remaining_video_budget > 0 and heap:
        _, group_key = heapq.heappop(heap)
        video = available_videos_by_group[group_key].pop(0)
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


@dataclass(frozen=True, slots=True)
class TieredVideoSamplingPlan:
    """Describes which videos a budgeted outlier-extraction pass samples across the frame-existence priority tiers.

    Notes:
        Unlike the raw-frame plan, which only ever draws from videos with no frames, the tiered plan may re-select a
        video that already holds frames so that outlier extraction adds more. It fills the frame budget from the
        highest-priority tier that still has videos before descending: videos with no frames, then videos with only raw
        frames, then videos that already carry outlier frames. Each newly selected video is projected to contribute
        ``frames_per_video_count`` further frames, whichever tier it came from.
    """

    selected_videos: tuple[str, ...]
    """The videos chosen for extraction in this pass, in selection order (pinned videos first)."""
    existing_frame_count: int
    """The number of frames already extracted across the candidate videos before this pass."""
    target_frame_count: int
    """The requested total number of frames the project should hold after extraction."""
    projected_frame_count: int
    """The number of frames the project is expected to hold once the selected videos are extracted."""
    budget_already_met: bool
    """Indicates whether the existing frames already meet the target, so the pass would extract nothing."""
    target_unreachable: bool
    """Indicates whether too few candidate videos remain to reach the target in a single pass, so the pass falls
    short of it; repeated passes keep adding frames to the lower tiers."""
    per_tier: tuple[tuple[str, int, int], ...] = ()
    """The per-tier breakdown as ``(tier_name, available_video_count, selected_video_count)`` tuples in priority
    order, where ``tier_name`` is one of ``OUTLIER_SELECTION_TIERS``."""
    per_group: tuple[tuple[str, int, int, int, int], ...] = ()
    """The per-group breakdown when grouping is used, as ``(group, existing_frame_count, added_video_count,
    projected_frame_count, available_video_count)`` tuples in canonical group order, or empty when the selection was
    not grouped. ``available_video_count`` is how many candidate videos the group held before this pass."""
    always_included_overshoot: bool = False
    """Indicates whether the always-included videos alone exceeded the pass budget, so the projected total overshoots
    the target by the surplus always-included videos."""


def plan_tiered_video_sampling(
    videos: list[str],
    extracted_frame_counts: dict[str, int],
    outlier_extracted_videos: set[str],
    frames_per_video_count: int,
    total_frame_budget: int,
    random_seed: int | None,
    *,
    groups: dict[str, list[str]] | None = None,
    pinned_videos: tuple[str, ...] = (),
) -> TieredVideoSamplingPlan:
    """Selects a subset of candidate videos toward a total-frame budget, prioritizing videos with the fewest frames.

    Sums the frames already extracted across the candidate videos and selects just enough additional videos, each
    projected to contribute ``frames_per_video_count`` frames, to reach ``total_frame_budget``. The candidates are
    partitioned into three priority tiers by what their labeled-data folders already hold: videos with no frames, then
    videos with only raw frames, then videos that already carry outlier frames. The pass fills its budget from the
    highest-priority tier that still has videos before descending to the next, so it adds frames to fresh videos first,
    then rounds out videos that have raw but no outlier frames, and only re-visits already-refined videos once every
    video has outlier frames.

    Without ``groups`` the videos within each tier are drawn uniformly at random. With ``groups`` they are balanced
    across groups by repeatedly assigning the next video to the group with the fewest projected frames, with the
    balancing state carried across tiers so cumulative per-group coverage evens out. In both modes any ``pinned_videos``
    are selected first, regardless of tier.

    Args:
        videos: The ordered candidate video paths the pass may sample from; every entry must be extractable (for
            outliers, already analyzed).
        extracted_frame_counts: The number of frames already extracted for each candidate video, keyed by path.
        outlier_extracted_videos: The candidate videos whose labeled-data folder already holds outlier frames.
        frames_per_video_count: The number of frames each newly sampled video is projected to contribute.
        total_frame_budget: The total number of frames the project should hold after extraction.
        random_seed: The seed for the random draw, or None to draw nondeterministically. With grouping, the seed also
            fixes which of a tier's videos within a group are sampled.
        groups: A mapping of group to that group's candidate videos, enabling balanced per-group selection. Set to None
            to draw uniformly within each tier.
        pinned_videos: The videos to include whenever the pass extracts frames, selected before the tiered draw.
            Duplicates and unknown entries are ignored; unlike the raw-frame plan, a pin that already has frames is
            still honored so outlier extraction adds more to it.

    Returns:
        A TieredVideoSamplingPlan naming the selected videos and reporting the existing, target, and projected frame
        counts alongside the budget-already-met, unreachable-target, per-tier, per-group, and overshoot details.
    """
    outlier_set = set(outlier_extracted_videos)
    tier_videos = (
        [video for video in videos if extracted_frame_counts.get(video, 0) == 0],
        [video for video in videos if extracted_frame_counts.get(video, 0) > 0 and video not in outlier_set],
        [video for video in videos if video in outlier_set],
    )

    existing_frame_count = sum(extracted_frame_counts.get(video, 0) for video in videos)
    remaining_frame_count = total_frame_budget - existing_frame_count

    if remaining_frame_count <= 0:
        return TieredVideoSamplingPlan(
            selected_videos=(),
            existing_frame_count=existing_frame_count,
            target_frame_count=total_frame_budget,
            projected_frame_count=existing_frame_count,
            budget_already_met=True,
            target_unreachable=False,
            per_tier=tuple(
                (name, len(tier), 0) for name, tier in zip(OUTLIER_SELECTION_TIERS, tier_videos, strict=True)
            ),
        )

    needed_video_count = math.ceil(remaining_frame_count / frames_per_video_count)
    candidate_count = len(videos)
    target_unreachable = needed_video_count > candidate_count
    needed_video_count = min(needed_video_count, candidate_count)

    candidate_set = set(videos)
    eligible_pinned_videos = [video for video in dict.fromkeys(pinned_videos) if video in candidate_set]
    always_included_overshoot = len(eligible_pinned_videos) > needed_video_count

    per_group: tuple[tuple[str, int, int, int, int], ...] = ()
    if groups is not None:
        selected_video_list, per_group, tier_selected_counts = _select_tiered_balanced(
            tier_videos=tier_videos,
            groups=groups,
            extracted_frame_counts=extracted_frame_counts,
            frames_per_video_count=frames_per_video_count,
            needed_video_count=needed_video_count,
            pinned_videos=eligible_pinned_videos,
            random_seed=random_seed,
        )
        selected_videos = tuple(selected_video_list)
    else:
        selected_videos, tier_selected_counts = _select_tiered_uniform(
            tier_videos=tier_videos,
            needed_video_count=needed_video_count,
            pinned_videos=eligible_pinned_videos,
            random_seed=random_seed,
        )

    projected_frame_count = existing_frame_count + len(selected_videos) * frames_per_video_count
    per_tier = tuple(
        (name, len(tier), selected_count)
        for name, tier, selected_count in zip(OUTLIER_SELECTION_TIERS, tier_videos, tier_selected_counts, strict=True)
    )
    return TieredVideoSamplingPlan(
        selected_videos=selected_videos,
        existing_frame_count=existing_frame_count,
        target_frame_count=total_frame_budget,
        projected_frame_count=projected_frame_count,
        budget_already_met=False,
        target_unreachable=target_unreachable,
        per_tier=per_tier,
        per_group=per_group,
        always_included_overshoot=always_included_overshoot,
    )


def _select_tiered_uniform(
    tier_videos: tuple[list[str], ...],
    needed_video_count: int,
    pinned_videos: list[str],
    random_seed: int | None,
) -> tuple[tuple[str, ...], list[int]]:
    """Fills the budget from the priority tiers in order, drawing uniformly at random within each tier.

    Args:
        tier_videos: The candidate videos partitioned into priority tiers, highest priority first.
        needed_video_count: The number of videos the budget calls for this pass, including any pinned videos.
        pinned_videos: The eligible pinned videos to always include, already deduplicated.
        random_seed: The seed for the random draws, or None for a nondeterministic draw.

    Returns:
        A tuple of the selected video paths (pinned first, then tier-by-tier) and the per-tier selected counts.
    """
    generator = Random(random_seed)  # noqa: S311 -- video sampling is not security-sensitive.
    selected_videos = list(dict.fromkeys(pinned_videos))
    seen = set(selected_videos)
    remaining_video_budget = max(0, needed_video_count - len(selected_videos))

    tier_selected_counts: list[int] = []
    for tier in tier_videos:
        selected_this_tier = 0
        if remaining_video_budget > 0:
            tier_pool = [video for video in tier if video not in seen]
            generator.shuffle(tier_pool)
            selected_this_tier = min(remaining_video_budget, len(tier_pool))
            chosen_videos = tier_pool[:selected_this_tier]
            selected_videos.extend(chosen_videos)
            seen.update(chosen_videos)
            remaining_video_budget -= selected_this_tier
        tier_selected_counts.append(selected_this_tier)
    return tuple(selected_videos), tier_selected_counts


def _select_tiered_balanced(
    tier_videos: tuple[list[str], ...],
    groups: dict[str, list[str]],
    extracted_frame_counts: dict[str, int],
    frames_per_video_count: int,
    needed_video_count: int,
    pinned_videos: list[str],
    random_seed: int | None,
) -> tuple[list[str], tuple[tuple[str, int, int, int, int], ...], list[int]]:
    """Fills the budget tier by tier, balancing each tier's draw across groups by least projected coverage.

    Seeds each group's projected frame count with the frames it already holds, honors the pinned videos first, then
    walks the priority tiers in order. Within a tier it repeatedly assigns the next video to the least-covered group
    that still has an un-selected video in that tier, carrying the projected-frame state across tiers so cumulative
    per-group coverage equalizes. Determinism is fixed by a canonical group order, a seeded shuffle of each group's
    videos, and a group-name tiebreak, so a fixed seed reproduces the selection.

    Args:
        tier_videos: The candidate videos partitioned into priority tiers, highest priority first.
        groups: A mapping of group to that group's candidate videos.
        extracted_frame_counts: The number of frames already extracted for each candidate video, keyed by path.
        frames_per_video_count: The number of frames each newly sampled video is projected to contribute.
        needed_video_count: The total number of videos to select this pass, including any pinned videos.
        pinned_videos: The eligible pinned videos to always include, already deduplicated.
        random_seed: The seed for the per-group video shuffle, or None for a nondeterministic shuffle.

    Returns:
        A tuple of the selected video paths (pinned first, then the balanced per-tier fill), the per-group breakdown as
        ``(group, existing_frame_count, added_video_count, projected_frame_count, available_video_count)`` tuples in
        canonical group order, and the per-tier selected counts.
    """
    generator = Random(random_seed)  # noqa: S311 -- video sampling is not security-sensitive.
    group_keys = sorted(groups)

    group_of: dict[str, str] = {}
    existing_frames_by_group: dict[str, int] = {}
    available_video_counts_by_group: dict[str, int] = dict.fromkeys(group_keys, 0)
    for group_key in group_keys:
        members = groups[group_key]
        for video in members:
            group_of[video] = group_key
        existing_frames_by_group[group_key] = sum(extracted_frame_counts.get(video, 0) for video in members)
    for tier in tier_videos:
        for video in tier:
            video_group_key = group_of.get(video)
            if video_group_key is not None:
                available_video_counts_by_group[video_group_key] += 1

    projected_frames_by_group = dict(existing_frames_by_group)
    selected_videos: list[str] = []
    seen: set[str] = set()

    for video in pinned_videos:
        if video in seen:
            continue
        selected_videos.append(video)
        seen.add(video)
        pin_group_key = group_of.get(video)
        if pin_group_key is not None:
            projected_frames_by_group[pin_group_key] += frames_per_video_count

    remaining_video_budget = max(0, needed_video_count - len(selected_videos))
    tier_selected_counts: list[int] = []
    for tier in tier_videos:
        selected_this_tier = 0
        if remaining_video_budget > 0:
            available_videos_by_group: dict[str, list[str]] = {group_key: [] for group_key in group_keys}
            for video in tier:
                if video in seen:
                    continue
                video_group_key = group_of.get(video)
                if video_group_key is not None:
                    available_videos_by_group[video_group_key].append(video)
            for group_key in group_keys:
                generator.shuffle(available_videos_by_group[group_key])
            heap = [
                (projected_frames_by_group[group_key], group_key)
                for group_key in group_keys
                if available_videos_by_group[group_key]
            ]
            heapq.heapify(heap)
            while remaining_video_budget > 0 and heap:
                _, group_key = heapq.heappop(heap)
                video = available_videos_by_group[group_key].pop(0)
                selected_videos.append(video)
                seen.add(video)
                projected_frames_by_group[group_key] += frames_per_video_count
                remaining_video_budget -= 1
                selected_this_tier += 1
                if available_videos_by_group[group_key]:
                    heapq.heappush(heap, (projected_frames_by_group[group_key], group_key))
        tier_selected_counts.append(selected_this_tier)

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
    return selected_videos, per_group, tier_selected_counts
