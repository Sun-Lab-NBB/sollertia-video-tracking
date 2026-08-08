"""Provides the shared live progress-bar, worker-supervision, and failure-reporting assets every pipeline builds on."""

from .failures import (
    PipelineFailedError,
    PipelineInterruptedError,
    read_file_tail,
    is_interrupt_signal,
    describe_process_exit,
    enable_native_crash_dumps,
)
from .live_bar import LiveBar, format_duration
from .supervision import WorkerExit, ProcessSupervisor

__all__ = [
    "LiveBar",
    "PipelineFailedError",
    "PipelineInterruptedError",
    "ProcessSupervisor",
    "WorkerExit",
    "describe_process_exit",
    "enable_native_crash_dumps",
    "format_duration",
    "is_interrupt_signal",
    "read_file_tail",
]
