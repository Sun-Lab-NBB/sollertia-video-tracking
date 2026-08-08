"""Provides the shared live progress-bar, worker-supervision, failure-reporting, and optimization-reporting assets."""

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
from .optimization_report import write_optimization_report

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
    "write_optimization_report",
]
