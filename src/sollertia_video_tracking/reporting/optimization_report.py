"""Provides the resolved-optimization report the training and inference pipelines write before they start work."""

import sys
from collections.abc import Sequence

_ROW_INDENT: str = "  "
"""The indentation applied to every rendered row, setting the report body in from the rules that enclose it."""

_COLUMN_GAP: str = "  "
"""The separator placed between the label and value columns of a rendered row."""

_TITLE_OVERHEAD: int = 4
"""The number of characters the header's leading dashes and surrounding spaces add to the title itself."""

_MINIMUM_RULE_TAIL: int = 3
"""The smallest number of dashes kept after the title, so a short report's header still reads as a rule."""


def write_optimization_report(title: str, rows: Sequence[tuple[str, str]]) -> None:
    """Writes the resolved-optimization report for a run to the standard error stream.

    Args:
        title: The report's title, naming the run the rows belong to.
        rows: The resolved optimizations as ``(label, value)`` pairs, in display order. An empty sequence writes
            nothing, since a run with no resolved optimizations has nothing to report.
    """
    if not rows:
        return
    sys.stderr.write(f"{_format_optimization_report(title=title, rows=rows)}\n")
    sys.stderr.flush()


def _format_optimization_report(title: str, rows: Sequence[tuple[str, str]]) -> str:
    """Renders the rows as a titled block of label and value columns aligned to their widest entry.

    Args:
        title: The report's title, rendered into the header rule.
        rows: The resolved optimizations as ``(label, value)`` pairs, in display order. At least one is required,
            since the column width is measured from the rows.

    Returns:
        The rendered report as a multi-line string, without a trailing newline.
    """
    label_width = max(len(label) for label, _value in rows)
    body = [f"{_ROW_INDENT}{label:<{label_width}}{_COLUMN_GAP}{value}".rstrip() for label, value in rows]
    widest_row = max(len(line) for line in body)
    width = max(widest_row, len(title) + _TITLE_OVERHEAD + _MINIMUM_RULE_TAIL)
    return "\n".join([f"-- {title} ".ljust(width, "-"), *body, "-" * width])
