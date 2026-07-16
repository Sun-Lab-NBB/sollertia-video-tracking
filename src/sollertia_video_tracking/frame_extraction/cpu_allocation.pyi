from typing import Any

DEFAULT_RESERVED_CORE_COUNT: int
_SATURATING_CORES_PER_WORKER: int

def plan_core_allocation(
    video_count: int, total_core_count: int, worker_count: int, cores_per_worker: int, reserved_core_count: int
) -> tuple[int, list[set[int]]]: ...
def pin_worker_to_cores(core_set_queue: Any) -> None: ...
