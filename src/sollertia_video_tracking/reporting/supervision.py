"""Provides the timed-poll supervisor that starts, waits on, and tears down the parallel pipelines' worker processes."""

import time
from types import TracebackType
from typing import Any, Self
from dataclasses import dataclass
from collections.abc import Callable, Sequence

from .failures import is_interrupt_signal, signal_name_for_exit_code

_TERMINATION_GRACE_SECONDS: float = 5.0
"""How long a worker is given to end on the termination request before it is killed outright."""

_SUPERVISION_POLL_SECONDS: float = 0.5
"""How often the supervisor wakes while waiting for its workers.

The wait is a poll rather than an untimed join so a pending SIGINT is delivered to the main thread promptly. An
untimed join parks the interpreter inside a lock acquire that stays deaf to Ctrl-C once heavyweight extension modules
are loaded, which is what leaves a wedged run unkillable from the terminal."""

_REAP_POLL_SECONDS: float = 0.02
"""How often the supervisor wakes while reaping workers that are already on their way out.

Reaping runs at the end of a run, when every worker has finished its work and is exiting, so waiting at the
steady-state cadence would add that cadence to the run's wall clock for no benefit. It is also the window a killed
worker is given to be reaped before ``terminate_all`` reports it as a survivor."""

_EXIT_COLLECTION_TIMEOUT_SECONDS: float = 5.0
"""How long the supervisor waits for finished workers to exit before reading their status anyway."""


@dataclass(frozen=True, slots=True)
class WorkerExit:
    """Captures how one supervised worker process ended."""

    name: str
    """The label the supervisor was given for this worker, used to attribute its death to a unit of work."""
    pid: int | None
    """The worker's process identifier, or None when it never started."""
    exit_code: int
    """The worker's exit code, which is the negated signal number when a signal ended it."""
    signal_name: str | None
    """The name of the signal that ended the worker, or None when it exited on its own or its signal has no name."""

    @property
    def crashed(self) -> bool:
        """Returns whether the worker ended in a way that leaves its assigned work unfinished."""
        return self.exit_code != 0

    @property
    def interrupted(self) -> bool:
        """Returns whether the worker was ended by a deliberate stop rather than by a fault."""
        return is_interrupt_signal(self.signal_name)


class ProcessSupervisor:
    """Starts, waits on, and always tears down a group of worker processes, reporting how each one ended.

    Notes:
        Waiting is a timed poll rather than an untimed join, which keeps Ctrl-C deliverable and gives the optional
        stall probe somewhere to run. Every worker's exit code is read once it has ended, so a worker killed by a
        signal is reported rather than folded into the same outcome as a worker that finished. The stall probe only
        warns, because a legitimately slow decode must never be terminated for looking idle.

    Args:
        processes: The already-constructed, not-yet-started worker processes.
        names: The label for each process, positionally matched, used to attribute a death to a unit of work.

    Attributes:
        _processes: The supervised worker processes.
        _names: The label carried for each supervised process.
        _started: Determines whether the workers have been started.
    """

    def __init__(self, processes: Sequence[Any], names: Sequence[str]) -> None:
        self._processes = list(processes)
        self._names = list(names)
        self._started = False

    def __repr__(self) -> str:
        """Returns a string representation of the ProcessSupervisor instance."""
        return f"ProcessSupervisor(processes={len(self._processes)}, started={self._started})"

    def __enter__(self) -> Self:
        """Starts every supervised worker and returns the supervisor."""
        self.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Terminates every worker still running, so no failure path leaves an orphan behind."""
        self.terminate_all()

    def start(self) -> None:
        """Starts every supervised worker process."""
        for process in self._processes:
            process.start()
        self._started = True

    def any_alive(self) -> bool:
        """Returns whether at least one supervised worker is still running."""
        return any(process.is_alive() for process in self._processes)

    def collect_exits(self, *, timeout: float = _EXIT_COLLECTION_TIMEOUT_SECONDS) -> tuple[WorkerExit, ...]:
        """Waits briefly for every worker to end and returns how each one did.

        The wait shares one deadline across the whole group rather than granting each worker its own, so reaping a
        finished run costs the same whether it ran one worker or twenty.

        Args:
            timeout: How long, in seconds, to wait for the group to exit before reading their status regardless.

        Returns:
            One record per supervised worker, describing how it ended.
        """
        self._wait_for_exit(deadline=time.monotonic() + timeout, poll=_REAP_POLL_SECONDS)
        return tuple(
            _build_worker_exit(process=process, name=name)
            for process, name in zip(self._processes, self._names, strict=True)
        )

    def supervise(
        self,
        *,
        stall_probe: Callable[[], float] | None = None,
        stall_timeout: float | None = None,
        on_stall: Callable[[float], None] | None = None,
    ) -> tuple[WorkerExit, ...]:
        """Waits for every worker to end, reporting a stalled run along the way, and returns how each one ended.

        Args:
            stall_probe: A callable returning the seconds elapsed since the run last made observable progress, or None
                to skip stall reporting.
            stall_timeout: The silence, in seconds, past which the run is reported as stalled, or None to skip stall
                reporting.
            on_stall: The callable invoked with the elapsed silence each time the stall threshold is newly crossed, or
                None to leave a crossed threshold unreported.

        Returns:
            One record per supervised worker, describing how it ended.
        """
        pending = list(zip(self._processes, self._names, strict=True))
        exits: list[WorkerExit] = []
        stall_reported = False
        while pending:
            still_running = []
            for process, name in pending:
                # Reaped without blocking, so one cycle costs the poll interval once rather than once per worker.
                process.join(timeout=0)
                if process.is_alive():
                    still_running.append((process, name))
                    continue
                exits.append(_build_worker_exit(process=process, name=name))
            pending = still_running
            if not pending:
                break
            if stall_probe is not None and stall_timeout is not None:
                silence = stall_probe()
                if silence < stall_timeout:
                    stall_reported = False
                elif not stall_reported:
                    stall_reported = True
                    if on_stall is not None:
                        on_stall(silence)
            time.sleep(_SUPERVISION_POLL_SECONDS)
        return tuple(exits)

    def terminate_all(self, *, grace: float = _TERMINATION_GRACE_SECONDS) -> tuple[int, ...]:
        """Ends every worker still running, escalating from a termination request to an outright kill.

        Args:
            grace: How long the workers are given to end on the termination request before they are killed.

        Returns:
            The process identifiers of the workers that survived even the kill, which is empty in every ordinary run.
        """
        if not self._started:
            return ()
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        self._wait_for_exit(deadline=time.monotonic() + grace, poll=_REAP_POLL_SECONDS)
        survivors = []
        for process in self._processes:
            if not process.is_alive():
                continue
            process.kill()
            process.join(timeout=_REAP_POLL_SECONDS)
            if process.is_alive() and process.pid is not None:
                survivors.append(process.pid)
        return tuple(survivors)

    def _wait_for_exit(self, *, deadline: float, poll: float) -> None:
        """Waits for every worker to exit, giving up at the deadline.

        Args:
            deadline: The monotonic timestamp past which the wait gives up.
            poll: How long, in seconds, to wait between liveness checks.
        """
        while self.any_alive() and time.monotonic() < deadline:
            for process in self._processes:
                process.join(timeout=0)
            if self.any_alive():
                time.sleep(poll)


def _build_worker_exit(process: Any, name: str) -> WorkerExit:
    """Reads an ended worker's exit status into a record naming how it ended.

    Args:
        process: The worker process, which has already ended.
        name: The label carried for this worker.

    Returns:
        The record describing how the worker ended.
    """
    # A process the supervisor never observed running reports no exit code, which is treated as a clean exit so a
    # worker that finished before its first poll is not misreported as a crash.
    exit_code = 0 if process.exitcode is None else int(process.exitcode)
    return WorkerExit(
        name=name,
        pid=process.pid,
        exit_code=exit_code,
        signal_name=signal_name_for_exit_code(exit_code),
    )
