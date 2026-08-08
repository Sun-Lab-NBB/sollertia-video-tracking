"""Contains tests for the resolved-optimization report the training and inference pipelines write before they start."""

import sys

from sollertia_video_tracking.reporting.optimization_report import (
    _TITLE_OVERHEAD,
    _MINIMUM_RULE_TAIL,
    write_optimization_report,
    _format_optimization_report,
)


def test_format_aligns_the_value_column_to_the_widest_label() -> None:
    """Verifies that the label column is padded to its widest entry so the values line up down the report."""
    rows = (("device", "cuda [0]"), ("cudnn.benchmark", "off"))
    lines = _format_optimization_report(title="inference optimizations", rows=rows).split("\n")
    assert lines[1] == "  device           cuda [0]"
    assert lines[2] == "  cudnn.benchmark  off"


def test_format_sizes_the_rules_to_the_widest_rendered_row() -> None:
    """Verifies that the rules match each other and take the width of the widest row when it outruns the title."""
    rows = (("precision", "bfloat16, chosen for the tensor cores this machine reports"),)
    lines = _format_optimization_report(title="opts", rows=rows).split("\n")
    assert lines[0].startswith("-- opts ")
    assert set(lines[-1]) == {"-"}
    assert len(lines[0]) == len(lines[-1]) == len(lines[1])


def test_format_keeps_a_rule_tail_when_the_title_outruns_the_body() -> None:
    """Verifies that a title wider than every row still ends in a rule rather than in the title text alone."""
    title = "a very long report title that outruns every row in the body"
    lines = _format_optimization_report(title=title, rows=(("tf32", "on"),)).split("\n")
    assert len(lines[0]) == len(title) + _TITLE_OVERHEAD + _MINIMUM_RULE_TAIL
    assert lines[0].endswith("-" * _MINIMUM_RULE_TAIL)


def test_format_does_not_leave_trailing_padding_on_a_row() -> None:
    """Verifies that the label padding is stripped, so a row carrying no value ends without trailing whitespace."""
    body = _format_optimization_report(title="opts", rows=(("tf32", ""), ("workers", "4"))).split("\n")[1:-1]
    assert body[0] == "  tf32"
    assert all(line == line.rstrip() for line in body)


def test_write_renders_the_report_on_the_progress_bars_stderr_stream(capsys) -> None:
    """Verifies that the report joins the progress bar on standard error, newline-terminated and off standard out."""
    rows = (("device", "cuda [0]"),)
    write_optimization_report(title="inference optimizations", rows=rows)
    captured = capsys.readouterr()
    assert captured.err == f"{_format_optimization_report(title='inference optimizations', rows=rows)}\n"
    assert captured.out == ""


def test_write_emits_nothing_when_there_are_no_rows(capsys) -> None:
    """Verifies that a run with no resolved optimizations writes no report rather than an empty titled block."""
    write_optimization_report(title="opts", rows=())
    assert capsys.readouterr().err == ""


def test_write_flushes_so_the_report_precedes_the_progress_bar(monkeypatch) -> None:
    """Verifies that the writer flushes, since an unflushed report would surface after the bar it must precede."""
    flushed: list[bool] = []
    stream = _RecordingStream()
    monkeypatch.setattr(stream, "flush", lambda: flushed.append(True))
    monkeypatch.setattr(sys, "stderr", stream)
    write_optimization_report(title="opts", rows=(("tf32", "on"),))
    assert flushed == [True]


class _RecordingStream:
    """Stands in for the standard error stream, accepting writes so the flush can be observed."""

    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, text: str) -> None:
        """Records one written chunk."""
        self.written.append(text)

    def flush(self) -> None:
        """Accepts the flush the writer issues after the report."""
