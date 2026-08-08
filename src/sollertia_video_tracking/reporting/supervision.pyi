from types import TracebackType
from typing import Any, Self
from dataclasses import dataclass
from collections.abc import (
    Callable as Callable,
    Sequence,
)

from _typeshed import Incomplete

from .failures import (
    is_interrupt_signal as is_interrupt_signal,
    signal_name_for_exit_code as signal_name_for_exit_code,
)

_TERMINATION_GRACE_SECONDS: float
_SUPERVISION_POLL_SECONDS: float
_REAP_POLL_SECONDS: float
_EXIT_COLLECTION_TIMEOUT_SECONDS: float

@dataclass(frozen=True, slots=True)
class WorkerExit:
    name: str
    pid: int | None
    exit_code: int
    signal_name: str | None
    @property
    def crashed(self) -> bool: ...
    @property
    def interrupted(self) -> bool: ...

class ProcessSupervisor:
    _processes: Incomplete
    _names: Incomplete
    _started: bool
    def __init__(self, processes: Sequence[Any], names: Sequence[str]) -> None: ...
    def __repr__(self) -> str: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def start(self) -> None: ...
    def any_alive(self) -> bool: ...
    def collect_exits(self, *, timeout: float = ...) -> tuple[WorkerExit, ...]: ...
    def supervise(
        self,
        *,
        stall_probe: Callable[[], float] | None = None,
        stall_timeout: float | None = None,
        on_stall: Callable[[float], None] | None = None,
    ) -> tuple[WorkerExit, ...]: ...
    def terminate_all(self, *, grace: float = ...) -> tuple[int, ...]: ...
    def _wait_for_exit(self, *, deadline: float, poll: float) -> None: ...

def _build_worker_exit(process: Any, name: str) -> WorkerExit: ...
