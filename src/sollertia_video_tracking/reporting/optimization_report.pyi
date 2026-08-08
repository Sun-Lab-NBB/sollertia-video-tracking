from collections.abc import Sequence

_ROW_INDENT: str
_COLUMN_GAP: str
_TITLE_OVERHEAD: int
_MINIMUM_RULE_TAIL: int

def write_optimization_report(title: str, rows: Sequence[tuple[str, str]]) -> None: ...
def _format_optimization_report(title: str, rows: Sequence[tuple[str, str]]) -> str: ...
