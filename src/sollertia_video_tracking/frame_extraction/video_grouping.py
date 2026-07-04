"""Provides the algorithm for clustering related videos by their shared, non-date file-name components."""

import re
from pathlib import Path

# A date is recognized only when it is anchored by an unambiguous signal, never as a lone numeric run, so a name
# component that happens to contain a year-like number (a token numbered 2019, a "Cohort2020" prefix) or a run of
# counters (a session_trial_rep triple) is not mistaken for a date and dropped. Two anchors are trusted: a full
# four-digit year ``(19|20)YY``, and an alphabetic month name (``Jan``, ``September``). Every supported span carries one
# of these anchors, so purely two-digit numeric dates are left to the pattern override rather than guessed at. Grouping
# only needs to locate and remove the date, not read it, so day-first and month-first orders are both accepted wherever
# they cannot be told apart: the same components are stripped either way.
_YEAR = r"(?:19|20)\d{2}"
_MONTH = r"(?:0[1-9]|1[0-2])"
_DAY = r"(?:0[1-9]|[12]\d|3[01])"
_DAY_LOOSE = r"(?:0?[1-9]|[12]\d|3[01])"
_MONTH_NAMES = (
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
)

# Single all-digit component spanning one index: a compact ``YYYYMMDD`` run at the start (any trailing time digits are
# ignored), or an eight-digit ``DDMMYYYY``/``MMDDYYYY`` run whose trailing four digits are a year.
_COMPACT_YEAR_FIRST = re.compile(rf"^{_YEAR}{_MONTH}{_DAY}")
_COMPACT_YEAR_LAST = re.compile(rf"^(?:{_MONTH}{_DAY}|{_DAY}{_MONTH}){_YEAR}$")

# A single fused alphanumeric component carrying a month name and a full year, with an optional day, in any of the
# component orders that keep the name and the year adjacent to the month (``15Jan2024``, ``Jan2024``, ``2024Jan15``).
_FUSED_NAMED_DATE = re.compile(
    rf"^(?:{_DAY_LOOSE}(?:{_MONTH_NAMES}){_YEAR}"
    rf"|(?:{_MONTH_NAMES}){_DAY_LOOSE}?{_YEAR}"
    rf"|{_YEAR}(?:{_MONTH_NAMES}){_DAY_LOOSE}?)$",
    re.IGNORECASE,
)

# Whole-component matchers used to classify the pieces of a delimiter-separated date.
_YEAR_PIECE = re.compile(rf"^{_YEAR}$")
_MONTH_PIECE = re.compile(rf"^{_MONTH}$")
_DAY_PIECE = re.compile(rf"^{_DAY}$")
_DAY_LOOSE_PIECE = re.compile(rf"^{_DAY_LOOSE}$")
_TWO_DIGIT_PIECE = re.compile(r"^\d\d$")
_MONTH_NAME_PIECE = re.compile(rf"^(?:{_MONTH_NAMES})$", re.IGNORECASE)


def _infer_group_key(stem: str) -> str | None:
    """Infers a grouping key from a video's file-name stem from its non-date components.

    Splits the stem into delimiter-bounded components (leaving fused alphanumeric tokens such as ``M01`` or ``MF11``
    intact), locates the first date span, and joins the components on the identity side of that date with an underscore.
    The working assumption is that the non-date components a set of file names share denote a common category (e.g. a
    subject), so files with the same key belong together while the date distinguishes recordings within a category.
    When the date leads the name, the components after it are used instead, so both ``M01_2024-01-15`` and
    ``2024-01-15_M01`` infer ``M01``. A name with no recognizable date returns None so the caller can fall back rather
    than fold a session or counter token into the key.

    Notes:
        A span is recognized only when it is anchored by a full four-digit year or an alphabetic month name, never by a
        bare numeric run, so a component that contains a year-like value or a run of counters is preserved rather than
        treated as a date boundary. Supported spans cover compact ``YYYYMMDD`` and ``DDMMYYYY``/``MMDDYYYY`` runs,
        year-first and year-last numeric triples, and named-month dates both fused (``15Jan2024``) and delimited
        (``2024_January_15``). Because grouping only removes the date, day-first and month-first orders are both
        accepted. Two-digit-year numeric dates and other schemes are left to the pattern override in ``group_videos``.

    Args:
        stem: The video file name without its directory or extension, for example ``MF_11_2025_07_21``.

    Returns:
        The inferred grouping key, or None when the stem contains no recognizable date span, or when the recognized
        date span leaves no non-date components on either side.
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
    """Locates the first date span in a list of delimiter-bounded name components.

    Scans the components left to right and returns the first span that carries a trusted anchor: a four-digit year or a
    month name. At each position a three-component span (a numeric or named year-month-day triple) is preferred over a
    two-component span (a month name beside a year), which is preferred over a one-component span (a compact all-digit
    date or a fused named-month date), so the longest date rooted at that position wins.

    Args:
        pieces: The delimiter-bounded components of a file-name stem, in order.

    Returns:
        The ``(start, end)`` half-open index range of the first recognized date span, or None when none is present.
    """
    count = len(pieces)
    for index in range(count):
        piece = pieces[index]
        if index + 2 < count:
            trio = (piece, pieces[index + 1], pieces[index + 2])
            if _is_numeric_date_triple(trio) or _is_named_date_triple(trio):
                return index, index + 3
        if index + 1 < count and _is_named_month_year_pair(piece, pieces[index + 1]):
            return index, index + 2
        if _is_compact_date_piece(piece) or _is_fused_named_date_piece(piece):
            return index, index + 1
    return None


def _valid_month_day_pair(first: str, second: str) -> bool:
    """Reports whether two two-digit components form a month and a day in either order.

    Args:
        first: The first candidate component.
        second: The second candidate component.

    Returns:
        True when one component is a valid month and the other a valid day, regardless of which is which.
    """
    return bool(
        (_MONTH_PIECE.match(first) and _DAY_PIECE.match(second))
        or (_MONTH_PIECE.match(second) and _DAY_PIECE.match(first))
    )


def _is_numeric_date_triple(trio: tuple[str, str, str]) -> bool:
    """Reports whether three all-digit components form a year-anchored numeric date.

    A triple qualifies when its leading or trailing component is a four-digit year and the remaining two form a
    month-and-day pair in either order, covering both ``YYYY MM DD`` and ``DD MM YYYY`` conventions.

    Args:
        trio: Three consecutive name components, in order.

    Returns:
        True when the triple is a year-anchored numeric date span.
    """
    first, middle, last = trio
    if not (first.isdigit() and middle.isdigit() and last.isdigit()):
        return False
    if _YEAR_PIECE.match(first) and _valid_month_day_pair(middle, last):
        return True
    return bool(_YEAR_PIECE.match(last) and _valid_month_day_pair(first, middle))


def _is_named_date_triple(trio: tuple[str, str, str]) -> bool:
    """Reports whether three components form a named-month date of a month name, a day, and a year.

    Exactly one component must be a month name; the other two must be all-digit, one a day and the other a year (a full
    four-digit year or a two-digit year, trusted because the month name already anchors the span). The month name and
    year may sit in any position, so ``2024 January 15``, ``15 Jan 2024``, and ``Jan 15 24`` all qualify.

    Args:
        trio: Three consecutive name components, in order.

    Returns:
        True when the triple is a named-month date span.
    """
    named = [piece for piece in trio if _MONTH_NAME_PIECE.match(piece)]
    if len(named) != 1:
        return False
    others = [piece for piece in trio if not _MONTH_NAME_PIECE.match(piece)]
    first, second = others
    if not (first.isdigit() and second.isdigit()):
        return False
    return (_is_day_piece(first) and _is_year_piece(second)) or (_is_day_piece(second) and _is_year_piece(first))


def _is_day_piece(piece: str) -> bool:
    """Reports whether a component is a one- or two-digit day of month (1-31)."""
    return bool(_DAY_LOOSE_PIECE.match(piece))


def _is_year_piece(piece: str) -> bool:
    """Reports whether a component is a full four-digit year or a two-digit year."""
    return bool(_YEAR_PIECE.match(piece) or _TWO_DIGIT_PIECE.match(piece))


def _is_named_month_year_pair(first: str, second: str) -> bool:
    """Reports whether two components form a month name beside a full four-digit year, in either order.

    Only a full four-digit year is trusted here, because no day is present to reinforce the span, so a bare ``Jan_23``
    is not treated as a date.

    Args:
        first: The first candidate component.
        second: The second candidate component.

    Returns:
        True when one component is a month name and the other a four-digit year.
    """
    return bool(
        (_MONTH_NAME_PIECE.match(first) and _YEAR_PIECE.match(second))
        or (_MONTH_NAME_PIECE.match(second) and _YEAR_PIECE.match(first))
    )


def _is_compact_date_piece(piece: str) -> bool:
    """Reports whether a single all-digit component is a compact year-anchored date.

    A leading ``YYYYMMDD`` run (with any trailing time digits ignored) or an eight-digit ``DDMMYYYY``/``MMDDYYYY`` run
    whose trailing four digits are a year both qualify.

    Args:
        piece: The name component to test.

    Returns:
        True when the component is a compact year-anchored date.
    """
    if not piece.isdigit():
        return False
    return bool(_COMPACT_YEAR_FIRST.match(piece) or _COMPACT_YEAR_LAST.match(piece))


def _is_fused_named_date_piece(piece: str) -> bool:
    """Reports whether a single fused alphanumeric component is a named-month date carrying a full year.

    Args:
        piece: The name component to test.

    Returns:
        True when the component fuses a month name with a four-digit year (and an optional day), as in ``15Jan2024`` or
        ``2024Jan15``.
    """
    return bool(_FUSED_NAMED_DATE.match(piece))


def group_videos(videos: list[str], group_by_pattern: str | None = None) -> dict[str, list[str]]:
    """Groups video paths that share the same non-date file-name components, preserving first-seen order per group.

    Each video is keyed either by a caller-supplied regular expression or, by default, by the structural inference in
    ``_infer_group_key``. A video the inference cannot key (no recognizable date) becomes its own group, so an unusual
    file name never collapses the rest of the project. Supplying ``group_by_pattern`` overrides the inference entirely
    for conventions it does not cover (two-digit-year numeric dates, session counters, and similar).

    Args:
        videos: The candidate video paths to group.
        group_by_pattern: A regular expression whose first capturing group names the group for each video's stem, or
            the whole match when the pattern has no capturing groups or its first group did not participate in the
            match. A stem the pattern fails to match is keyed by its full stem, forming its own group. Set to None to
            infer the key structurally from the file-name date span.

    Returns:
        A mapping of grouping key to the list of that group's video paths, in first-seen order.

    Raises:
        ValueError: If ``group_by_pattern`` is not a valid regular expression.
    """
    compiled = None
    if group_by_pattern is not None:
        try:
            compiled = re.compile(group_by_pattern)
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
