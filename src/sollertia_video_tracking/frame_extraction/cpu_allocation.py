"""Provides the CPU-core allocation logic that distributes parallel frame-extraction workers across the machine."""

from typing import Any
import contextlib

import psutil

DEFAULT_RESERVED_CORE_COUNT: int = 4
"""The number of CPU cores left free by default for other work while frame extraction is running."""

_SATURATING_CORES_PER_WORKER: int = 4
"""The number of cores one video's decode keeps busy at high concurrency (measured at ~3.9 across 24 workers)."""


def plan_core_allocation(
    video_count: int,
    total_core_count: int,
    worker_count: int,
    cores_per_worker: int,
    reserved_core_count: int,
) -> tuple[int, list[set[int]]]:
    """Determines how many videos run at once and which cores each worker is pinned to.

    Partitions the usable cores (total minus reserved) into one disjoint block per worker, so the running workers
    occupy the usable cores without oversubscribing them. When the worker count is left automatic, it is capped so
    each worker receives at least a saturating core budget rather than spreading the cores thin across more, throttled
    workers; any remaining videos run in later waves. Explicit worker or core counts are honored as given and may
    overlap only within the usable band, leaving the reserved cores free regardless.

    Notes:
        DeepLabCut reads a single video's frames in one serial Python loop that cannot be sped up, but each frame's
        HEVC / H264 decode is itself multithreaded and keeps several cores busy. Throughput therefore comes from
        decoding many videos at once rather than from accelerating any single video.

    Args:
        video_count: The number of videos that will be processed.
        total_core_count: The total number of CPU cores available on the machine.
        worker_count: The requested number of concurrent workers, or -1 to resolve the count automatically.
        cores_per_worker: The requested number of cores per worker, or -1 to spread the usable cores evenly.
        reserved_core_count: The number of cores to leave free for other tasks.

    Returns:
        A tuple of the resolved worker count and a list of core-id sets, one per worker.
    """
    usable_core_count = max(1, total_core_count - max(0, reserved_core_count))

    if cores_per_worker < 1:
        if worker_count < 1:
            # Prefers saturated workers: runs only as many as can each hold a saturating block, leaving any remaining
            # videos for subsequent waves.
            worker_count = min(video_count, max(1, usable_core_count // _SATURATING_CORES_PER_WORKER))
        worker_count = max(1, min(worker_count, video_count))
        base_cores_per_worker, remainder = divmod(usable_core_count, worker_count)
        per_worker_core_counts = [
            max(1, base_cores_per_worker + (1 if worker < remainder else 0)) for worker in range(worker_count)
        ]
    else:
        if worker_count < 1:
            worker_count = max(1, usable_core_count // cores_per_worker)
        worker_count = max(1, min(worker_count, video_count))
        per_worker_core_counts = [max(1, cores_per_worker)] * worker_count

    core_sets: list[set[int]] = []
    core_cursor = 0
    for worker_core_count in per_worker_core_counts:
        core_sets.append({(core_cursor + offset) % usable_core_count for offset in range(worker_core_count)})
        core_cursor += worker_core_count
    return worker_count, core_sets


def pin_worker_to_cores(core_set_queue: Any) -> None:
    """Pins the calling pool worker to the next free core block, called once per worker at start.

    Each worker claims one core set from the shared queue and binds its CPU affinity to it, so the worker and every
    thread its decoder spawns stay within a disjoint block of cores. CPU affinity is supported on Linux and Windows;
    macOS exposes no affinity API, so its workers run unpinned. The binding is best-effort, so a missing slot or an
    unsupported platform degrades to an unpinned worker rather than aborting the run.

    Args:
        core_set_queue: The shared queue holding one core-id set per worker, produced by an extraction pipeline.
    """
    with contextlib.suppress(Exception):
        core_set = core_set_queue.get_nowait()
        psutil.Process().cpu_affinity(list(core_set))
