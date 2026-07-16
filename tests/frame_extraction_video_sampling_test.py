"""Contains tests for the budgeted video-subset sampling planner used to top a DLC project up toward a frame budget."""

from random import Random
from dataclasses import FrozenInstanceError

import pytest

from sollertia_video_tracking.frame_extraction import video_sampling
from sollertia_video_tracking.frame_extraction.video_sampling import (
    VideoSamplingPlan,
    _select_balanced,
    plan_video_sampling,
)


class _NoShuffleRandom:
    """A drop-in Random replacement whose shuffle is a no-op, making tier/group ordering deterministic in tests."""

    def shuffle(self, x):  # noqa: ARG002 -- the parameter must stay named "x" to accept Random.shuffle's keyword call.
        # Leave the list in its incoming order so selection order follows the input order exactly.
        return None


@pytest.fixture(autouse=True)
def _deterministic_random(monkeypatch):
    """Replaces the module's Random with a no-op-shuffle stub so every test observes deterministic selection order."""
    monkeypatch.setattr(video_sampling, "Random", _NoShuffleRandom)


def test_plan_defaults_and_frozen():
    """Verifies VideoSamplingPlan defaults per_group/overshoot and rejects mutation as a frozen, slotted dataclass."""
    plan = VideoSamplingPlan(
        selected_videos=("a",),
        existing_frame_count=1,
        target_frame_count=2,
        projected_frame_count=3,
        budget_already_met=False,
        target_unreachable=True,
    )
    assert plan.per_group == ()
    assert plan.always_included_overshoot is False
    assert plan.selected_videos == ("a",)
    assert plan.target_unreachable is True
    with pytest.raises(FrozenInstanceError):
        plan.existing_frame_count = 5  # frozen dataclass forbids attribute assignment.


def test_budget_already_met_when_existing_exceeds_target():
    """Verifies that when existing frames exceed the target the pass extracts nothing and flags budget_already_met."""
    plan = plan_video_sampling(["a", "b"], {"a": 10, "b": 10}, frames_per_video_count=10, total_frame_budget=5)
    assert plan.budget_already_met is True
    assert plan.selected_videos == ()
    assert plan.existing_frame_count == 20
    assert plan.projected_frame_count == 20
    assert plan.target_frame_count == 5
    assert plan.target_unreachable is False
    assert plan.per_group == ()
    assert plan.always_included_overshoot is False


def test_budget_already_met_at_exact_target():
    """Verifies that the remaining <= 0 guard also triggers when existing frames exactly equal the target (boundary)."""
    plan = plan_video_sampling(["a"], {"a": 10}, frames_per_video_count=10, total_frame_budget=10)
    assert plan.budget_already_met is True
    assert plan.selected_videos == ()
    assert plan.projected_frame_count == 10


def test_uniform_early_return_in_unextracted_tier():
    """Verifies that uniform selection fills from not-yet-extracted videos and stops once capacity meets the budget."""
    plan = plan_video_sampling(["a", "b", "c", "d"], {}, frames_per_video_count=10, total_frame_budget=25)
    # a, b, c contribute 30 >= 25; d is never reached.
    assert plan.selected_videos == ("a", "b", "c")
    assert plan.existing_frame_count == 0
    assert plan.target_frame_count == 25
    assert plan.projected_frame_count == 30
    assert plan.budget_already_met is False
    assert plan.target_unreachable is False
    assert plan.per_group == ()
    assert plan.always_included_overshoot is False


def test_uniform_spills_into_below_ceiling_tier_and_completes_loop():
    """Verifies that uniform selection exhausts the not-yet-extracted tier before topping up a below-ceiling video."""
    plan = plan_video_sampling(["a", "b"], {"a": 5}, frames_per_video_count=10, total_frame_budget=20)
    # unextracted [b] (cap 10) fills first, then below-ceiling [a] (cap 5) tops the budget up; loop runs to completion.
    assert plan.selected_videos == ("b", "a")
    assert plan.existing_frame_count == 5
    assert plan.projected_frame_count == 20
    assert plan.target_unreachable is False


def test_uniform_target_unreachable_selects_everything():
    """Verifies that when capacity is short of the budget the plan flags target_unreachable and selects all it can."""
    plan = plan_video_sampling(["a"], {}, frames_per_video_count=10, total_frame_budget=25)
    assert plan.target_unreachable is True
    assert plan.selected_videos == ("a",)
    assert plan.projected_frame_count == 10
    assert plan.budget_already_met is False


def test_uniform_pins_overshoot_budget():
    """Verifies that a pinned video is always included even when its capacity alone overshoots the remaining budget."""
    plan = plan_video_sampling(["a", "b"], {}, frames_per_video_count=10, total_frame_budget=5, pinned_videos=("a",))
    # Pin a (cap 10) alone exceeds the remaining 5, so the fill loop returns immediately and only a is selected.
    assert plan.selected_videos == ("a",)
    assert plan.always_included_overshoot is True
    assert plan.projected_frame_count == 10


def test_pins_are_deduplicated_and_unknown_pins_ignored():
    """Verifies that duplicate pins collapse to one entry and pins outside the candidate set are dropped."""
    plan = plan_video_sampling(
        ["a", "b", "c"], {}, frames_per_video_count=10, total_frame_budget=100, pinned_videos=("a", "a", "z")
    )
    # "z" is not a candidate and the duplicate "a" is collapsed; the unreachable budget then pulls in b and c too.
    assert plan.selected_videos == ("a", "b", "c")
    assert plan.target_unreachable is True
    assert plan.always_included_overshoot is False


def test_balanced_alternates_between_groups():
    """Verifies that grouped selection assigns each next video to the least-covered group, alternating equal groups."""
    plan = plan_video_sampling(
        ["a", "b", "c", "d"],
        {},
        frames_per_video_count=10,
        total_frame_budget=20,
        groups={"g1": ["a", "b"], "g2": ["c", "d"]},
    )
    assert plan.selected_videos == ("b", "d")
    assert plan.per_group == (("g1", 0, 1, 10, 2), ("g2", 0, 1, 10, 2))
    assert plan.projected_frame_count == 20
    assert plan.budget_already_met is False


def test_balanced_single_group_unreachable_exits_when_heap_empties():
    """Verifies that a grouped pass whose only group runs out of videos exits on an empty heap and flags unreachable."""
    plan = plan_video_sampling(["a"], {}, frames_per_video_count=10, total_frame_budget=100, groups={"g1": ["a"]})
    assert plan.target_unreachable is True
    assert plan.selected_videos == ("a",)
    assert plan.per_group == (("g1", 0, 1, 10, 1),)
    assert plan.projected_frame_count == 10


def test_balanced_group_exhaustion_does_not_requeue():
    """Verifies that a group is not re-queued once its last video is taken and the pass ends when the budget is met."""
    plan = plan_video_sampling(
        ["a", "b", "c"], {}, frames_per_video_count=10, total_frame_budget=30, groups={"g1": ["a"], "g2": ["b", "c"]}
    )
    # g1 (1 video) empties and is not requeued; g2 is drained tail-first (c then b).
    assert plan.selected_videos == ("a", "c", "b")
    assert plan.per_group == (("g1", 0, 1, 10, 1), ("g2", 0, 2, 20, 2))
    assert plan.projected_frame_count == 30


def test_balanced_seeds_existing_frames_and_uses_below_ceiling_tier():
    """Verifies that grouped seeding counts prior frames per group and drains below-ceiling before unextracted."""
    plan = plan_video_sampling(
        ["a", "b"], {"a": 5}, frames_per_video_count=10, total_frame_budget=20, groups={"g1": ["a", "b"]}
    )
    # Group g1 holds a (below-ceiling, cap 5) and b (not-yet-extracted, cap 10); the tail-first pop drains b then a.
    assert plan.selected_videos == ("b", "a")
    assert plan.per_group == (("g1", 5, 2, 20, 2),)
    assert plan.projected_frame_count == 20


def test_balanced_pins_across_group_membership_states():
    """Verifies that pins are always included whether grouped and below ceiling, grouped at ceiling, or ungrouped."""
    plan = plan_video_sampling(
        ["a", "b", "x"],
        {"b": 10},
        frames_per_video_count=10,
        total_frame_budget=100,
        groups={"g1": ["a", "b"]},
        pinned_videos=("a", "b", "x"),
    )
    # a: grouped and available -> removed and counted; b: grouped but at ceiling (not available); x: ungrouped.
    assert plan.selected_videos == ("a", "b", "x")
    assert plan.per_group == (("g1", 10, 2, 20, 1),)
    assert plan.projected_frame_count == 30
    assert plan.target_unreachable is True
    assert plan.always_included_overshoot is False


def test_select_balanced_skips_duplicate_pins_directly():
    """Verifies that _select_balanced with a repeated pin exercises the in-loop dedup guard the planner prevents."""
    selected, per_group = _select_balanced(
        groups={"g1": ["a", "b"]},
        extracted_frame_counts={},
        capacity_of={"a": 10, "b": 10},
        remaining_frame_count=5,
        pinned_videos=["a", "a"],
    )
    # The second "a" is already seen and is skipped; only one "a" is returned.
    assert selected == ["a"]
    assert per_group == (("g1", 0, 1, 10, 2),)


def test_uniform_selection_valid_with_real_random_shuffle(monkeypatch):
    """Verifies that with real shuffling Random restored, uniform selection yields a valid, correctly sized subset."""
    monkeypatch.setattr(video_sampling, "Random", Random)
    plan = plan_video_sampling(["a", "b", "c", "d", "e"], {}, frames_per_video_count=10, total_frame_budget=30)
    # Regardless of shuffle order, exactly three distinct candidates (3 * 10 == 30) are chosen to meet the budget.
    assert len(plan.selected_videos) == 3
    assert len(set(plan.selected_videos)) == 3
    assert set(plan.selected_videos).issubset({"a", "b", "c", "d", "e"})
    assert plan.projected_frame_count == 30
    assert plan.target_unreachable is False
