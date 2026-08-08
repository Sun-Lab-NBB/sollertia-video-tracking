"""Provides the failure types and the process-exit classification every pipeline reports its runs through."""

import signal
from pathlib import Path
import contextlib
from collections import deque
import faulthandler

_INTERRUPT_SIGNALS: frozenset[str] = frozenset({"SIGINT", "SIGTERM"})
"""The signal names that end a process on a deliberate stop rather than a fault."""

_NATIVE_CRASH_SIGNALS: frozenset[str] = frozenset({"SIGSEGV", "SIGABRT", "SIGBUS", "SIGFPE", "SIGILL"})
"""The signal names that end a process on a crash inside a native backend such as CUDA, cuDNN, or OpenCV."""

_SHELL_SIGNAL_STATUS_BASE: int = 128
"""The offset a shell adds to a signal number when reporting the exit status of a process that a signal ended."""


class PipelineFailedError(RuntimeError):
    """Indicates that a pipeline run could not be completed, carrying the operator-facing report as its message."""


class PipelineInterruptedError(RuntimeError):
    """Indicates that a pipeline run was stopped by the operator or by a termination signal rather than by a fault."""


def enable_native_crash_dumps() -> None:
    """Installs the interpreter's fault handler so a native crash writes a traceback to the standard error stream.

    A crash inside CUDA, cuDNN, or OpenCV ends the process without unwinding Python, so no exception handler runs and
    the dump is the only record of where it happened. The handler retains the standard-error descriptor number rather
    than the stream object. A worker that swaps ``sys.stderr`` at the Python level must install the handler first to
    keep its dump on the terminal. A worker that reassigns descriptor 2 with ``os.dup2`` sends its dump to the new
    target whatever the install order.
    """
    with contextlib.suppress(Exception):
        faulthandler.enable()


def signal_name_for_exit_code(exit_code: int) -> str | None:
    """Resolves the name of the signal that ended a process from its reported exit code.

    Args:
        exit_code: The process exit code, which is the negated signal number when a signal ended the process.

    Returns:
        The signal name, or None when the process exited on its own or the signal number has no name.
    """
    if exit_code >= 0:
        return None
    try:
        return signal.Signals(-exit_code).name
    except ValueError:
        return None


def is_interrupt_signal(signal_name: str | None) -> bool:
    """Determines whether a signal name marks a deliberate stop rather than a fault.

    Args:
        signal_name: The name of the signal that ended a process, or None when no signal did.

    Returns:
        True when the signal marks a deliberate stop, False otherwise.
    """
    return signal_name in _INTERRUPT_SIGNALS


def describe_process_exit(exit_code: int, *, pid: int | None, role: str, memory_remedy: str) -> str:
    """Builds the sentences naming how a worker process ended and what the operator should do about it.

    Args:
        exit_code: The worker's exit code, which is the negated signal number when a signal ended it.
        pid: The worker's process identifier, or None when it is unknown.
        role: The noun naming the worker in the report, such as ``"inference worker"``.
        memory_remedy: The command-specific advice offered when the out-of-memory killer ended the worker.

    Returns:
        The reason the worker ended, followed by the remedy for the failure classes that have one.
    """
    worker = f"the {role} (PID {pid})" if pid is not None else f"the {role}"
    signal_name = signal_name_for_exit_code(exit_code)
    # A process a signal ended reports the negated signal number as its exit code, which the shell in turn reports as
    # 128 plus that number.
    status = _SHELL_SIGNAL_STATUS_BASE - exit_code if exit_code < 0 else exit_code
    if signal_name is None:
        return f"{worker} exited with status {status} without reporting a result."
    if signal_name == "SIGKILL":
        return (
            f"{worker} was killed by SIGKILL (shell status {status}), which during a long run is almost always the "
            f"operating system's out-of-memory killer. {memory_remedy}"
        )
    if signal_name in _NATIVE_CRASH_SIGNALS:
        return (
            f"{worker} was killed by {signal_name} (shell status {status}), which is a crash inside a native backend "
            f"such as CUDA, cuDNN, or OpenCV. Its fault-handler dump, when one was written, precedes this report."
        )
    return f"{worker} was killed by {signal_name} (shell status {status})."


def read_file_tail(path: Path, *, lines: int) -> str:
    """Reads the trailing lines of a text file without holding the whole file in memory.

    Args:
        path: The file to read.
        lines: The maximum number of trailing lines to return.

    Returns:
        The trailing lines joined into one block, or an empty string when the file is absent, empty, or unreadable.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as text_file:
            tail = deque(text_file, maxlen=lines)
    except OSError:
        return ""
    return "".join(tail).rstrip("\n")
