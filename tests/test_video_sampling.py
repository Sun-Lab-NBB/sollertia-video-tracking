"""Tests the budgeted video-subset selection, focusing on the tiered outlier sampler's priority and balancing."""

from sollertia_video_tracking.frame_extraction.video_sampling import (
    OUTLIER_SELECTION_TIERS,
    plan_tiered_video_sampling,
)

_FRAMES_PER_VIDEO = 10
"""The per-video frame contribution used across the sampling tests, standing in for ``numframes2pick``."""


def _tier_selected_counts(plan: object) -> dict[str, int]:
    """Maps each tier name to the number of videos the plan selected from it."""
    return {name: selected for name, _available, selected in plan.per_tier}  # type: ignore[attr-defined]


def _six_video_project() -> tuple[list[str], dict[str, int], set[str]]:
    """Builds a six-video project spanning all three tiers: two with no frames, two raw-only, two with outlier."""
    videos = ["v1", "v2", "v3", "v4", "v5", "v6"]
    frame_counts = {"v1": 0, "v2": 0, "v3": 10, "v4": 10, "v5": 10, "v6": 10}
    outlier_extracted = {"v5", "v6"}
    return videos, frame_counts, outlier_extracted


def test_draws_from_highest_priority_tier_first() -> None:
    """A budget that needs only a couple of videos draws exclusively from the no-frames tier."""
    videos, frame_counts, outlier_extracted = _six_video_project()
    plan = plan_tiered_video_sampling(
        videos, frame_counts, outlier_extracted, _FRAMES_PER_VIDEO, total_frame_budget=60, random_seed=1
    )
    assert set(plan.selected_videos) <= {"v1", "v2"}
    assert len(plan.selected_videos) == 2
    assert _tier_selected_counts(plan) == {"no-frames": 2, "raw-only": 0, "has-outlier": 0}


def test_spills_into_the_next_tier_when_a_tier_runs_out() -> None:
    """A larger budget exhausts the no-frames tier, then descends into the raw-only tier."""
    videos, frame_counts, outlier_extracted = _six_video_project()
    plan = plan_tiered_video_sampling(
        videos, frame_counts, outlier_extracted, _FRAMES_PER_VIDEO, total_frame_budget=80, random_seed=1
    )
    assert set(plan.selected_videos) == {"v1", "v2", "v3", "v4"}
    assert _tier_selected_counts(plan) == {"no-frames": 2, "raw-only": 2, "has-outlier": 0}


def test_all_videos_with_outlier_frames_are_re_sampled_from_the_last_tier() -> None:
    """When every video already has outlier frames, additional frames come from the has-outlier tier."""
    videos = ["v1", "v2", "v3"]
    frame_counts = dict.fromkeys(videos, 10)
    plan = plan_tiered_video_sampling(
        videos, frame_counts, set(videos), _FRAMES_PER_VIDEO, total_frame_budget=50, random_seed=1
    )
    assert len(plan.selected_videos) == 2
    assert _tier_selected_counts(plan) == {"no-frames": 0, "raw-only": 0, "has-outlier": 2}


def test_budget_already_met_selects_nothing() -> None:
    """A budget already covered by the existing frames selects no videos and flags the budget as met."""
    videos, frame_counts, outlier_extracted = _six_video_project()
    plan = plan_tiered_video_sampling(
        videos, frame_counts, outlier_extracted, _FRAMES_PER_VIDEO, total_frame_budget=30, random_seed=1
    )
    assert plan.budget_already_met
    assert plan.selected_videos == ()


def test_unreachable_budget_takes_every_candidate_once() -> None:
    """A budget larger than the candidates can supply in one pass selects every video once and flags it unreachable."""
    videos, frame_counts, outlier_extracted = _six_video_project()
    plan = plan_tiered_video_sampling(
        videos, frame_counts, outlier_extracted, _FRAMES_PER_VIDEO, total_frame_budget=1000, random_seed=1
    )
    assert plan.target_unreachable
    assert len(plan.selected_videos) == len(videos)
    assert _tier_selected_counts(plan) == {"no-frames": 2, "raw-only": 2, "has-outlier": 2}


def test_balances_the_draw_across_groups() -> None:
    """With grouping, a two-video budget spreads one video into each of the two groups rather than both into one."""
    videos = ["a1", "a2", "a3", "b1"]
    frame_counts = dict.fromkeys(videos, 0)
    groups = {"A": ["a1", "a2", "a3"], "B": ["b1"]}
    plan = plan_tiered_video_sampling(
        videos, frame_counts, set(), _FRAMES_PER_VIDEO, total_frame_budget=20, random_seed=1, groups=groups
    )
    added_by_group = {group: added for (group, _existing, added, _projected, _available) in plan.per_group}
    assert added_by_group == {"A": 1, "B": 1}


def test_pinned_videos_are_included_regardless_of_tier() -> None:
    """An always-included video is selected even when it sits in the lowest-priority tier."""
    videos, frame_counts, outlier_extracted = _six_video_project()
    plan = plan_tiered_video_sampling(
        videos,
        frame_counts,
        outlier_extracted,
        _FRAMES_PER_VIDEO,
        total_frame_budget=60,
        random_seed=1,
        pinned_videos=("v5",),
    )
    assert "v5" in plan.selected_videos


def test_a_fixed_seed_reproduces_the_selection() -> None:
    """A fixed seed yields an identical selection across runs, in both uniform and balanced modes."""
    videos, frame_counts, outlier_extracted = _six_video_project()
    groups = {"g1": ["v1", "v3", "v5"], "g2": ["v2", "v4", "v6"]}
    first_uniform = plan_tiered_video_sampling(
        videos, frame_counts, outlier_extracted, _FRAMES_PER_VIDEO, total_frame_budget=1000, random_seed=7
    )
    second_uniform = plan_tiered_video_sampling(
        videos, frame_counts, outlier_extracted, _FRAMES_PER_VIDEO, total_frame_budget=1000, random_seed=7
    )
    first_balanced = plan_tiered_video_sampling(
        videos, frame_counts, outlier_extracted, _FRAMES_PER_VIDEO, total_frame_budget=80, random_seed=7, groups=groups
    )
    second_balanced = plan_tiered_video_sampling(
        videos, frame_counts, outlier_extracted, _FRAMES_PER_VIDEO, total_frame_budget=80, random_seed=7, groups=groups
    )
    assert first_uniform.selected_videos == second_uniform.selected_videos
    assert first_balanced.selected_videos == second_balanced.selected_videos


def test_tier_names_match_the_public_priority_order() -> None:
    """The plan reports its tiers in the documented priority order."""
    videos, frame_counts, outlier_extracted = _six_video_project()
    plan = plan_tiered_video_sampling(
        videos, frame_counts, outlier_extracted, _FRAMES_PER_VIDEO, total_frame_budget=60, random_seed=1
    )
    assert tuple(name for name, _available, _selected in plan.per_tier) == OUTLIER_SELECTION_TIERS
