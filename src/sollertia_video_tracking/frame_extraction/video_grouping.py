"""Provides the algorithm for clustering related videos by their shared, non-date file-name components."""

import re
from pathlib import Path

# A date is only recognized as a full span, never as a lone four-digit run, so a name component that happens to contain
# a year-like number (a token numbered 2019, a "Cohort2020" prefix) is not mistaken for a date and dropped. Two spans
# are recognized: a compact YYYYMMDD run at the start of an all-digit component (any trailing time digits are ignored),
# or a year component immediately followed by a month component and a day component across delimiters.
_YEAR = r"(?:19|20)\d{2}"
_MONTH = r"(?:0[1-9]|1[0-2])"
_DAY = r"(?:0[1-9]|[12]\d|3[01])"
_COMPACT_DATE = re.compile(rf"^{_YEAR}{_MONTH}{_DAY}")
_YEAR_PIECE = re.compile(rf"^{_YEAR}$")
_MONTH_PIECE = re.compile(rf"^{_MONTH}$")
_DAY_PIECE = re.compile(rf"^{_DAY}$")


def _infer_group_key(stem: str) -> str | None:
    """Infers a grouping key from a video's file-name stem from its non-date components.

    Splits the stem into delimiter-bounded components (leaving fused alphanumeric tokens such as ``M01`` or ``MF_11``
    intact), locates the first ISO-style date span, and joins the components on the identity side of that date with an
    underscore. The working assumption is that the non-date components a set of file names share denote a common
    category (e.g. a subject), so files with the same key belong together while the date distinguishes
    recordings within a category. When the date leads the name, the components after it are used instead, so both
    ``M01_2024-01-15`` and ``2024-01-15_M01`` infer ``M01``. Only ISO-style dates are recognized; a name with no such
    date returns None so the caller can fall back rather than fold a session or counter token into the key.

    Notes:
        The date is matched only as a full span (a compact ``YYYYMMDD`` run or a year-month-day triple), never as a
        bare four-digit number, so a component that contains a year-like value is preserved rather than treated as a
        date boundary. Detection is deliberately limited to ISO-style dates; two-digit years, day-first or month-first
        orders, and non-date naming schemes are left to the pattern override in ``group_videos``.

    Args:
        stem: The video file name without its directory or extension, for example ``MF_11_2025_07_21``.

    Returns:
        The inferred grouping key, or None when the stem contains no recognizable ISO-style date span.
    """
    # The ISO 8601 date-time separator ``T`` is a letter, so a timestamp like ``20240115T093000`` would otherwise fuse
    # into one non-numeric component and hide the date. Splitting only a ``T`` that sits between digits exposes the date
    # without disturbing a name that merely contains a ``T`` (for example ``MT_3`` or ``T3``).
    normalized = re.sub(r"(?<=\d)T(?=\d)", "_", stem)
    pieces = [piece for piece in re.split(r"[^A-Za-z0-9]+", normalized) if piece]
    if not pieces:
        return None

    span = _find_date_span(pieces)
    if span is None:
        return None

    start, end = span
    identity_pieces = pieces[:start] or pieces[end:]
    if not identity_pieces:
        return None
    return "_".join(identity_pieces)


def _find_date_span(pieces: list[str]) -> tuple[int, int] | None:
    """Locates the first ISO-style date span in a list of delimiter-bounded name components.

    Args:
        pieces: The delimiter-bounded components of a file-name stem, in order.

    Returns:
        The ``(start, end)`` half-open index range of the first recognized date span, or None when none is present. A
        compact ``YYYYMMDD`` component spans one index; a year-month-day triple spans three.
    """
    for index, piece in enumerate(pieces):
        # An all-digit component whose leading eight digits form a valid YYYYMMDD is a compact date; the regex already
        # requires those eight digits, and any trailing time digits are ignored.
        if piece.isdigit() and _COMPACT_DATE.match(piece):
            return index, index + 1
        if (
            _YEAR_PIECE.match(piece)
            and index + 2 < len(pieces)
            and _MONTH_PIECE.match(pieces[index + 1])
            and _DAY_PIECE.match(pieces[index + 2])
        ):
            return index, index + 3
    return None


def group_videos(videos: list[str], key_pattern: str | None = None) -> dict[str, list[str]]:
    """Groups video paths that share the same non-date file-name components, preserving first-seen order per group.

    Each video is keyed either by a caller-supplied regular expression or, by default, by the structural inference in
    ``_infer_group_key``. A video the inference cannot key (no recognizable date) becomes its own group, so an unusual
    file name never collapses the rest of the project. Supplying ``key_pattern`` overrides the inference entirely for
    conventions it does not cover (two-digit years, day-first dates, session counters, and similar).

    Args:
        videos: The candidate video paths to group.
        key_pattern: A regular expression whose first capturing group (or whole match, if it has no groups) names the
            group for each video's stem. Set to None to infer the key structurally from the file-name date span.

    Returns:
        A mapping of grouping key to the list of that group's video paths, in first-seen order.

    Raises:
        ValueError: If ``key_pattern`` is not a valid regular expression.
    """
    compiled = None
    if key_pattern is not None:
        try:
            compiled = re.compile(key_pattern)
        except re.error as error:
            message = f"Unable to group videos. The group-by pattern is not a valid regular expression: {error}."
            raise ValueError(message) from error

    groups: dict[str, list[str]] = {}
    for video in videos:
        stem = Path(video).stem
        if compiled is not None:
            match = compiled.search(stem)
            if match is None:
                key = stem
            elif match.groups() and match.group(1) is not None:
                # A non-participating optional group (for example "M(\\d+)?" against "Mouse") yields a None group, so
                # fall through to the whole match rather than keying the video under None.
                key = match.group(1)
            else:
                key = match.group(0) or stem
        else:
            key = _infer_group_key(stem) or stem
        groups.setdefault(key, []).append(video)
    return groups
