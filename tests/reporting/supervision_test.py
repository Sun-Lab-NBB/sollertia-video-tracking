"""Contains tests for the worker-process supervisor the parallel pipelines wait on instead of joining untimed.

These drive the real supervisor against lightweight stand-in processes, so the start, poll, classify, and terminate
behavior is exercised without spawning a real interpreter.
"""

import signal

import pytest

from sollertia_video_tracking.reporting import (
    WorkerExit,
    ProcessSupervisor,
    supervision as supervision_module,
)


class _StubProcess:
    """Stands in for a spawned worker, ending after a configured number of polls with a configured exit code."""

    def __init__(self, *, exit_code=0, polls_until_exit=1, ignores_termination=False, never_exits=False):
        self.pid = None
        self.exitcode = None
        self._exit_code = exit_code
        self._polls_remaining = polls_until_exit
        self._ignores_termination = ignores_termination
        self._never_exits = never_exits
        self.started = False
        self.terminated = False
        self.killed = False

    def start(self):
        self.started = True
        self.pid = 5000 + id(self) % 1000

    def join(self, timeout=None):
        self.join_timeout = timeout
        # A worker that never exits on its own keeps its liveness regardless of how often it is reaped, so the
        # termination path is exercised rather than short-circuited by the stub's own poll clock.
        if self._never_exits:
            return
        if self._polls_remaining > 0:
            self._polls_remaining -= 1
        if self._polls_remaining == 0 and self.exitcode is None:
            self.exitcode = self._exit_code

    def is_alive(self):
        return self.started and self.exitcode is None

    def terminate(self):
        self.terminated = True
        if not self._ignores_termination:
            self.exitcode = -signal.SIGTERM

    def kill(self):
        self.killed = True
        self.exitcode = -signal.SIGKILL


# WorkerExit


def test_worker_exit_classifies_its_own_outcome():
    """Verifies that the exit record reports a clean exit, a crash, and a deliberate stop distinctly."""
    clean = WorkerExit(name="a", pid=1, exit_code=0, signal_name=None)
    crashed = WorkerExit(name="b", pid=2, exit_code=-signal.SIGKILL, signal_name="SIGKILL")
    stopped = WorkerExit(name="c", pid=3, exit_code=-signal.SIGTERM, signal_name="SIGTERM")

    assert clean.crashed is False
    assert clean.interrupted is False
    assert crashed.crashed is True
    assert crashed.interrupted is False
    assert stopped.crashed is True
    assert stopped.interrupted is True


# ProcessSupervisor


def test_supervise_returns_every_worker_exit_status():
    """Verifies that waiting returns one classified record per worker rather than discarding the exit codes."""
    processes = [
        _StubProcess(exit_code=0),
        _StubProcess(exit_code=-signal.SIGKILL, polls_until_exit=2),
        _StubProcess(exit_code=3),
    ]
    supervisor = ProcessSupervisor(processes=processes, names=["a", "b", "c"])
    supervisor.start()

    exits = supervisor.supervise()

    assert {record.name for record in exits} == {"a", "b", "c"}
    by_name = {record.name: record for record in exits}
    assert by_name["a"].exit_code == 0
    assert by_name["b"].signal_name == "SIGKILL"
    assert by_name["b"].crashed is True
    assert by_name["c"].exit_code == 3
    assert by_name["c"].signal_name is None


def test_supervise_returns_rather_than_blocking_when_a_worker_dies_abruptly():
    """Verifies that a worker that dies without reporting ends the wait, which is the regression guard for the hang."""
    processes = [_StubProcess(exit_code=-signal.SIGSEGV, polls_until_exit=1)]
    supervisor = ProcessSupervisor(processes=processes, names=["only"])
    supervisor.start()

    exits = supervisor.supervise()

    assert len(exits) == 1
    assert exits[0].signal_name == "SIGSEGV"


def test_supervise_warns_once_per_stall_and_never_terminates():
    """Verifies that a run reporting no progress is called out once, and that nothing is killed for looking idle."""
    process = _StubProcess(exit_code=0, polls_until_exit=4)
    supervisor = ProcessSupervisor(processes=[process], names=["only"])
    supervisor.start()
    warnings: list[float] = []

    supervisor.supervise(stall_probe=lambda: 999.0, stall_timeout=10.0, on_stall=warnings.append)

    assert warnings == [999.0]
    assert process.terminated is False
    assert process.killed is False


def test_supervise_reports_a_stall_again_once_progress_resumes_and_lapses():
    """Verifies that a run that recovers and stalls again is called out for the second stall too."""
    process = _StubProcess(exit_code=0, polls_until_exit=4)
    supervisor = ProcessSupervisor(processes=[process], names=["only"])
    supervisor.start()
    silences = iter([999.0, 0.0, 999.0, 999.0])
    warnings: list[float] = []

    supervisor.supervise(stall_probe=lambda: next(silences), stall_timeout=10.0, on_stall=warnings.append)

    assert warnings == [999.0, 999.0]


def test_terminate_all_escalates_to_a_kill(monkeypatch):
    """Verifies that a worker ignoring termination is killed outright and then reports no survivor."""
    monkeypatch.setattr(supervision_module.time, "sleep", lambda _seconds: None)
    stubborn = _StubProcess(ignores_termination=True, never_exits=True)
    supervisor = ProcessSupervisor(processes=[stubborn], names=["only"])
    supervisor.start()

    survivors = supervisor.terminate_all(grace=0.01)

    assert stubborn.terminated is True
    assert stubborn.killed is True
    assert survivors == ()


def test_terminate_all_is_a_no_op_before_the_workers_start():
    """Verifies that tearing down a supervisor that never started touches nothing."""
    process = _StubProcess()
    supervisor = ProcessSupervisor(processes=[process], names=["only"])

    assert supervisor.terminate_all() == ()
    assert process.terminated is False


def test_context_manager_starts_and_always_terminates(monkeypatch):
    """Verifies that the context manager starts every worker and tears them down even when its body raises."""
    monkeypatch.setattr(supervision_module.time, "sleep", lambda _seconds: None)
    process = _StubProcess(never_exits=True)

    def fail_inside_the_body():
        message = "body failed"
        raise RuntimeError(message)

    with (
        pytest.raises(RuntimeError, match="body failed"),
        ProcessSupervisor(processes=[process], names=["only"]) as supervisor,
    ):
        assert process.started is True
        assert supervisor.any_alive() is True
        fail_inside_the_body()

    assert process.terminated is True


def test_collect_exits_treats_a_never_observed_exit_code_as_clean():
    """Verifies that a worker reporting no exit code is recorded as a clean exit rather than as a crash."""
    process = _StubProcess(never_exits=True)
    supervisor = ProcessSupervisor(processes=[process], names=["only"])
    supervisor.start()

    exits = supervisor.collect_exits(timeout=0.01)

    assert exits[0].exit_code == 0
    assert exits[0].crashed is False


def test_terminate_all_names_a_worker_that_survives_even_a_kill(monkeypatch):
    """Verifies that a worker surviving termination and the kill is reported rather than silently abandoned."""
    monkeypatch.setattr(supervision_module.time, "sleep", lambda _seconds: None)

    class _UnkillableProcess(_StubProcess):
        def kill(self):
            self.killed = True

    unkillable = _UnkillableProcess(ignores_termination=True, never_exits=True)
    supervisor = ProcessSupervisor(processes=[unkillable], names=["only"])
    supervisor.start()

    survivors = supervisor.terminate_all(grace=0.01)

    assert survivors == (unkillable.pid,)
