"""Contains tests for the shared failure types and process-exit classification every pipeline reports through."""

import signal
import faulthandler

from sollertia_video_tracking.reporting import (
    PipelineFailedError,
    PipelineInterruptedError,
    failures as failures_module,
    read_file_tail,
    is_interrupt_signal,
    describe_process_exit,
    enable_native_crash_dumps,
)
from sollertia_video_tracking.training.pipeline import TrainingFailedError, TrainingInterruptedError
from sollertia_video_tracking.reporting.failures import signal_name_for_exit_code

_MEMORY_REMEDY = "Lower the batch size, then re-run."


# Error types


def test_training_errors_carry_the_shared_bases():
    """Verifies that the training error types are recognized by the shared funnel every command catches."""
    assert issubclass(TrainingFailedError, PipelineFailedError)
    assert issubclass(TrainingInterruptedError, PipelineInterruptedError)
    assert issubclass(PipelineFailedError, RuntimeError)
    assert issubclass(PipelineInterruptedError, RuntimeError)


# signal_name_for_exit_code / is_interrupt_signal


def test_signal_name_for_exit_code_maps_only_signal_deaths():
    """Verifies that a negated signal number resolves to its name while an ordinary exit code resolves to None."""
    assert signal_name_for_exit_code(-signal.SIGKILL) == "SIGKILL"
    assert signal_name_for_exit_code(-signal.SIGSEGV) == "SIGSEGV"
    assert signal_name_for_exit_code(0) is None
    assert signal_name_for_exit_code(3) is None
    # A negative code with no matching signal number resolves to None rather than raising.
    assert signal_name_for_exit_code(-999) is None


def test_is_interrupt_signal_covers_only_deliberate_stops():
    """Verifies that only the termination signals count as a deliberate stop."""
    assert is_interrupt_signal("SIGINT") is True
    assert is_interrupt_signal("SIGTERM") is True
    assert is_interrupt_signal("SIGKILL") is False
    assert is_interrupt_signal("SIGSEGV") is False
    assert is_interrupt_signal(None) is False


# describe_process_exit


def test_describe_process_exit_reports_the_out_of_memory_killer_with_its_remedy():
    """Verifies that a SIGKILL death names the out-of-memory killer, the shell status, and the caller's remedy."""
    described = describe_process_exit(-signal.SIGKILL, pid=4242, role="inference worker", memory_remedy=_MEMORY_REMEDY)
    assert "inference worker (PID 4242)" in described
    assert "SIGKILL" in described
    assert f"shell status {128 + signal.SIGKILL}" in described
    assert "out-of-memory killer" in described
    assert _MEMORY_REMEDY in described


def test_describe_process_exit_reports_a_native_crash():
    """Verifies that a native-crash signal is named as a backend crash rather than as an out-of-memory kill."""
    described = describe_process_exit(
        -signal.SIGSEGV, pid=7, role="frame-extraction worker", memory_remedy=_MEMORY_REMEDY
    )
    assert "SIGSEGV" in described
    assert f"shell status {128 + signal.SIGSEGV}" in described
    assert "native backend" in described
    assert _MEMORY_REMEDY not in described


def test_describe_process_exit_reports_a_plain_status_and_an_unclassified_signal():
    """Verifies that an ordinary non-zero exit and a signal outside both sets are still named exactly."""
    plain = describe_process_exit(3, pid=9, role="training worker", memory_remedy=_MEMORY_REMEDY)
    assert "exited with status 3" in plain

    hung_up = describe_process_exit(-signal.SIGHUP, pid=9, role="training worker", memory_remedy=_MEMORY_REMEDY)
    assert "SIGHUP" in hung_up
    assert f"shell status {128 + signal.SIGHUP}" in hung_up


def test_describe_process_exit_omits_an_unknown_process_identifier():
    """Verifies that a worker whose process identifier is unknown is described without a dangling 'PID None'."""
    described = describe_process_exit(1, pid=None, role="inference worker", memory_remedy=_MEMORY_REMEDY)
    assert "PID" not in described
    assert "the inference worker" in described


# read_file_tail


def test_read_file_tail_returns_only_the_trailing_window(tmp_path):
    """Verifies that a file longer than the window contributes only its trailing lines."""
    log = tmp_path / "run.log"
    log.write_text("".join(f"line {index}\n" for index in range(100)))

    tail = read_file_tail(log, lines=10)

    assert tail.splitlines() == [f"line {index}" for index in range(90, 100)]


def test_read_file_tail_returns_everything_shorter_than_the_window(tmp_path):
    """Verifies that a file shorter than the window is returned whole, without its trailing newline."""
    log = tmp_path / "run.log"
    log.write_text("only line\n")

    assert read_file_tail(log, lines=40) == "only line"


def test_read_file_tail_treats_an_unreadable_file_as_empty(tmp_path):
    """Verifies that a missing file reads as empty rather than raising while a failure is being reported."""
    assert read_file_tail(tmp_path / "absent.log", lines=40) == ""
    assert read_file_tail(tmp_path, lines=40) == ""


# enable_native_crash_dumps


def test_enable_native_crash_dumps_installs_the_fault_handler():
    """Verifies that the fault handler is installed so a native crash leaves a dump."""
    # pytest installs the fault handler for the whole session, so it is taken down here to observe the installation
    # rather than the state pytest left behind. The helper puts it back, which is why only the never-installed case
    # needs restoring.
    was_enabled = faulthandler.is_enabled()
    faulthandler.disable()
    try:
        enable_native_crash_dumps()
        assert faulthandler.is_enabled()
    finally:
        if not was_enabled:
            faulthandler.disable()


def test_enable_native_crash_dumps_survives_a_descriptorless_stream(monkeypatch):
    """Verifies that a stream exposing no descriptor does not turn crash reporting into a crash of its own."""

    def boom():
        message = "no descriptor under capture"
        raise ValueError(message)

    monkeypatch.setattr(failures_module.faulthandler, "enable", boom)
    enable_native_crash_dumps()
