"""Provides the CPU-core allocation logic that distributes parallel frame-extraction workers across the machine."""

import sys
import contextlib

import psutil

DEFAULT_RESERVED_CORE_COUNT: int = 2
"""The number of CPU cores left free by default for other work while frame extraction is running."""

_SATURATING_CORES_PER_WORKER: int = 4
"""The number of cores one video's decode keeps busy at high concurrency, measured to be close to four."""


def plan_core_allocation(
    video_count: int,
    total_core_count: int,
    worker_count: int,
    cores_per_worker: int,
    reserved_core_count: int,
) -> tuple[int, list[set[int]]]:
    """Determines how many videos run at once and which cores each worker is pinned to.

    Partitions the usable cores (total minus reserved) into one disjoint block per worker, so the running workers
    occupy the usable cores without oversubscribing them. When both the worker count and cores-per-worker are left
    automatic, each worker is given a roughly saturating core block and only as many workers run as the selected
    videos need. The full core budget is therefore deployed only when there are enough videos to fill it, and any
    remaining videos run in later waves. When more videos than full blocks remain and the budget does not divide
    evenly, one extra worker runs on the leftover cores as a smaller block so the whole budget is used. An explicit
    cores-per-worker instead gives each worker exactly that many cores, sizing an automatic worker count from the
    usable cores, while an explicit worker count splits the usable cores evenly across those workers. In both cases
    the worker count is capped at the number of videos, so a request for more workers than videos runs one worker
    per video. The pinned blocks stay within the usable band, leaving the reserved cores free regardless. An explicit
    request that cannot fit (more workers than usable cores, or a worker count times cores-per-worker that exceeds
    them) raises ``ValueError`` instead of overlapping the blocks and thrashing.

    Notes:
        DeepLabCut reads a single video's frames in one serial Python loop that cannot be sped up, but each frame's
        HEVC / H264 decode is itself multithreaded and keeps several cores busy. Throughput therefore comes from
        decoding many videos at once rather than from accelerating any single video.

    Args:
        video_count: The number of videos that will be processed.
        total_core_count: The total number of CPU cores available on the machine.
        worker_count: The requested number of concurrent workers, or -1 to resolve the count automatically.
        cores_per_worker: The requested number of cores per worker. Set to -1 to give each worker a saturating core
            block when the worker count is automatic, or to split the usable cores evenly across an explicit worker
            count.
        reserved_core_count: The number of cores to leave free for other tasks.

    Returns:
        A tuple of the resolved worker count and a list of core-id sets, one per worker.

    Raises:
        ValueError: If an explicit worker count or cores-per-worker (or their combination) needs more cores than
            are usable, which would force two or more workers to share cores. Lower the request so the workers'
            core blocks fit within the usable cores.
    """
    usable_core_count = max(1, total_core_count - max(0, reserved_core_count))

    if cores_per_worker < 1:
        if worker_count < 1:
            # Automatic worker and core counts: a saturating block per worker, one worker per video up to the budget.
            full_worker_count = usable_core_count // _SATURATING_CORES_PER_WORKER
            if full_worker_count == 0:
                worker_count = 1
                per_worker_core_counts = [usable_core_count]
            elif video_count <= full_worker_count:
                worker_count = max(1, video_count)
                per_worker_core_counts = [_SATURATING_CORES_PER_WORKER] * worker_count
            else:
                worker_count = full_worker_count
                per_worker_core_counts = [_SATURATING_CORES_PER_WORKER] * full_worker_count
                leftover_core_count = usable_core_count - full_worker_count * _SATURATING_CORES_PER_WORKER
                if leftover_core_count > 0 and video_count > full_worker_count:
                    worker_count += 1
                    per_worker_core_counts.append(leftover_core_count)
        else:
            # Explicit worker count with automatic cores: spreads the usable cores evenly across the workers.
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

    requested_core_count = sum(per_worker_core_counts)
    if worker_count > 1 and requested_core_count > usable_core_count:
        message = (
            f"Unable to pin {worker_count} frame-extraction workers to disjoint core blocks. The requested "
            f"configuration needs {requested_core_count} cores, but only {usable_core_count} of the machine's "
            f"{total_core_count} cores are usable after reserving {max(0, reserved_core_count)}. Lower the worker "
            f"count or the cores-per-worker so their total fits the usable cores, or reserve fewer cores."
        )
        raise ValueError(message)

    core_sets: list[set[int]] = []
    core_cursor = 0
    for worker_core_count in per_worker_core_counts:
        core_sets.append({(core_cursor + offset) % usable_core_count for offset in range(worker_core_count)})
        core_cursor += worker_core_count
    return worker_count, core_sets


def pin_process_to_cores(core_set: set[int]) -> None:
    """Pins the calling worker process to its assigned core block, called once per worker at start.

    Each worker binds its CPU affinity to the block it was constructed with, so the worker and every thread its
    decoder spawns stay within a disjoint set of cores. CPU affinity is supported on Linux and Windows. macOS exposes
    no affinity API, so its workers run unpinned. The binding is best-effort, so an unsupported platform degrades to an
    unpinned worker rather than aborting the run.

    Args:
        core_set: The core ids this worker is confined to. An empty set leaves the worker unpinned.
    """
    if not core_set:
        return
    with contextlib.suppress(Exception):
        if sys.platform != "darwin":
            psutil.Process().cpu_affinity(list(core_set))
