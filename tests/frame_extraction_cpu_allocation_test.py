"""Contains tests for the CPU-core allocation planner used to pin parallel frame-extraction workers."""

import queue

import pytest

from sollertia_video_tracking.frame_extraction.cpu_allocation import (
    pin_worker_to_cores,
    plan_core_allocation,
)


def test_automatic_worker_and_core_counts_saturating_blocks() -> None:
    """Verifies that automatic worker/core counts give each worker a saturating block, capped at the video count."""
    # 32 usable cores (34 - 2 reserved) -> 8 full four-core blocks, but only 3 videos.
    worker_count, core_sets = plan_core_allocation(
        video_count=3, total_core_count=34, worker_count=-1, cores_per_worker=-1, reserved_core_count=2
    )
    assert worker_count == 3
    assert [len(core_set) for core_set in core_sets] == [4, 4, 4]
    _assert_disjoint_within_usable(core_sets, usable_core_count=32)


def test_automatic_counts_fill_budget_with_leftover_block() -> None:
    """Verifies that when videos exceed full blocks and cores divide unevenly, a smaller leftover block is added."""
    # 30 usable cores -> 7 full four-core blocks (28 cores) + a 2-core leftover block for the 8th worker.
    worker_count, core_sets = plan_core_allocation(
        video_count=20, total_core_count=32, worker_count=-1, cores_per_worker=-1, reserved_core_count=2
    )
    assert worker_count == 8
    assert [len(core_set) for core_set in core_sets] == [4, 4, 4, 4, 4, 4, 4, 2]
    _assert_disjoint_within_usable(core_sets, usable_core_count=30)


def test_automatic_counts_no_leftover_when_budget_divides_evenly() -> None:
    """Verifies that an evenly divisible budget with surplus videos yields only full blocks and no leftover worker."""
    # 32 usable cores -> exactly 8 four-core blocks, no remainder.
    worker_count, core_sets = plan_core_allocation(
        video_count=50, total_core_count=34, worker_count=-1, cores_per_worker=-1, reserved_core_count=2
    )
    assert worker_count == 8
    assert [len(core_set) for core_set in core_sets] == [4] * 8


def test_automatic_counts_tiny_machine_single_worker() -> None:
    """Verifies that a machine too small for one saturating block collapses to a single worker on all usable cores."""
    worker_count, core_sets = plan_core_allocation(
        video_count=5, total_core_count=4, worker_count=-1, cores_per_worker=-1, reserved_core_count=2
    )
    assert worker_count == 1
    assert core_sets == [{0, 1}]


def test_explicit_worker_count_spreads_cores_evenly() -> None:
    """Verifies that an explicit worker count with automatic cores spreads the usable cores as evenly as possible."""
    # 30 usable cores across 4 workers -> 8, 8, 7, 7.
    worker_count, core_sets = plan_core_allocation(
        video_count=10, total_core_count=32, worker_count=4, cores_per_worker=-1, reserved_core_count=2
    )
    assert worker_count == 4
    assert sorted((len(core_set) for core_set in core_sets), reverse=True) == [8, 8, 7, 7]
    _assert_disjoint_within_usable(core_sets, usable_core_count=30)


def test_explicit_worker_count_capped_at_video_count() -> None:
    """Verifies that a request for more workers than videos runs one worker per video."""
    worker_count, _ = plan_core_allocation(
        video_count=2, total_core_count=32, worker_count=8, cores_per_worker=-1, reserved_core_count=2
    )
    assert worker_count == 2


def test_explicit_cores_per_worker_automatic_worker_count() -> None:
    """Verifies that explicit cores-per-worker sizes the worker count from usable cores, capped at the video count."""
    # 30 usable cores / 5 cores each -> 6 workers, but only 4 videos.
    worker_count, core_sets = plan_core_allocation(
        video_count=4, total_core_count=32, worker_count=-1, cores_per_worker=5, reserved_core_count=2
    )
    assert worker_count == 4
    assert [len(core_set) for core_set in core_sets] == [5, 5, 5, 5]


def test_explicit_both_counts() -> None:
    """Verifies that explicit worker and core counts are honored exactly when they fit the usable cores."""
    worker_count, core_sets = plan_core_allocation(
        video_count=10, total_core_count=32, worker_count=3, cores_per_worker=6, reserved_core_count=2
    )
    assert worker_count == 3
    assert [len(core_set) for core_set in core_sets] == [6, 6, 6]


def test_negative_reserved_core_count_treated_as_zero() -> None:
    """Verifies that a negative reserved-core request is clamped so the whole machine is usable."""
    worker_count, core_sets = plan_core_allocation(
        video_count=1, total_core_count=8, worker_count=1, cores_per_worker=-1, reserved_core_count=-5
    )
    assert worker_count == 1
    # All eight cores are usable because the negative reservation is clamped to zero.
    assert core_sets == [set(range(8))]


def test_oversubscription_raises_value_error() -> None:
    """Verifies that an explicit configuration that cannot fit usable cores raises rather than overlapping blocks."""
    with pytest.raises(ValueError, match="disjoint core blocks"):
        plan_core_allocation(
            video_count=10, total_core_count=8, worker_count=4, cores_per_worker=4, reserved_core_count=2
        )


def test_pin_worker_binds_to_claimed_core_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the worker pins its CPU affinity to the core set it claims from the shared queue."""
    core_queue: queue.Queue[set[int]] = queue.Queue()
    core_queue.put({1, 2, 3})

    recorded: list[list[int]] = []

    class _FakeProcess:
        def cpu_affinity(self, cores: list[int]) -> None:
            recorded.append(cores)

    monkeypatch.setattr("sollertia_video_tracking.frame_extraction.cpu_allocation.psutil.Process", _FakeProcess)
    pin_worker_to_cores(core_queue)

    assert recorded == [[1, 2, 3]]


def test_pin_worker_empty_queue_is_silent() -> None:
    """Verifies that an empty queue (more workers than slots) degrades to an unpinned worker without raising."""
    empty_queue: queue.Queue[set[int]] = queue.Queue()
    # Must not raise even though there is no core set to claim.
    pin_worker_to_cores(empty_queue)


def _assert_disjoint_within_usable(core_sets: list[set[int]], usable_core_count: int) -> None:
    """Verifies the planned core blocks are disjoint and stay within the usable core band."""
    seen: set[int] = set()
    for core_set in core_sets:
        assert core_set.isdisjoint(seen)
        assert all(0 <= core < usable_core_count for core in core_set)
        seen |= core_set
