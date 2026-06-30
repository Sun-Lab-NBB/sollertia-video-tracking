"""Provides the CPU-core allocation logic that distributes parallel frame-extraction workers across the machine."""

DEFAULT_RESERVE_CORES: int = 4
"""The number of CPU cores left free by default for other work while frame extraction is running."""

_SATURATING_CORES_PER_WORKER: int = 4
"""The number of cores one video's decode keeps busy at high concurrency (measured at ~3.9 across 24 workers)."""


def plan_core_allocation(
    video_count: int,
    core_count: int,
    workers: int,
    cores_per_worker: int,
    reserve_cores: int,
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
        core_count: The total number of CPU cores available on the machine.
        workers: The requested number of concurrent workers, or -1 to resolve the count automatically.
        cores_per_worker: The requested number of cores per worker, or -1 to spread the usable cores evenly.
        reserve_cores: The number of cores to leave free for other tasks.

    Returns:
        A tuple of the resolved worker count and a list of core-id sets, one per worker.
    """
    usable = max(1, core_count - max(0, reserve_cores))

    if cores_per_worker < 1:
        if workers < 1:
            # Prefers saturated workers: runs only as many as can each hold a saturating block, leaving any remaining
            # videos for subsequent waves.
            workers = min(video_count, max(1, usable // _SATURATING_CORES_PER_WORKER))
        workers = max(1, min(workers, video_count))
        base, remainder = divmod(usable, workers)
        counts = [max(1, base + (1 if worker < remainder else 0)) for worker in range(workers)]
    else:
        if workers < 1:
            workers = max(1, usable // cores_per_worker)
        workers = max(1, min(workers, video_count))
        counts = [max(1, cores_per_worker)] * workers

    core_sets: list[set[int]] = []
    cursor = 0
    for count in counts:
        core_sets.append({(cursor + offset) % usable for offset in range(count)})
        cursor += count
    return workers, core_sets
