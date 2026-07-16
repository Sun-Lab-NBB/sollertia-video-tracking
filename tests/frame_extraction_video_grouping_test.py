"""Contains tests for the video-grouping algorithm that clusters videos by their shared, non-date name components."""

import pytest

from sollertia_video_tracking.frame_extraction.video_grouping import (
    group_videos,
    _is_day_piece,
    _is_year_piece,
    _find_date_span,
    _infer_group_key,
    _is_month_day_pair,
    _is_named_date_triple,
    _is_compact_date_piece,
    _is_numeric_date_triple,
    _is_named_month_year_pair,
    _is_fused_named_date_piece,
)


# group_videos: default structural inference
def test_group_videos_infers_key_and_preserves_first_seen_order() -> None:
    """Verifies that default inference keys videos by non-date components, preserving first-seen order per group."""
    videos = [
        "M01_2024-01-15.mp4",
        "M02_2024-01-15.mp4",
        "M01_2024-02-20.mp4",
    ]
    groups = group_videos(videos)
    assert groups == {
        "M01": ["M01_2024-01-15.mp4", "M01_2024-02-20.mp4"],
        "M02": ["M02_2024-01-15.mp4"],
    }
    # First-seen ordering: "M01" is created before "M02", and its members stay in encounter order.
    assert list(groups.keys()) == ["M01", "M02"]


def test_group_videos_groups_leading_and_trailing_dates_together() -> None:
    """Verifies that a leading-date name and a trailing-date name that share an identity land in the same group."""
    videos = ["M01_2024-01-15.mp4", "2024-02-20_M01.mp4"]
    groups = group_videos(videos)
    assert groups == {"M01": ["M01_2024-01-15.mp4", "2024-02-20_M01.mp4"]}


def test_group_videos_paths_with_directories_use_stem_only() -> None:
    """Verifies that the grouping key is derived from the file stem, ignoring directory and extension."""
    groups = group_videos(["/data/vids/M01_2024-01-15.mp4"])
    assert groups == {"M01": ["/data/vids/M01_2024-01-15.mp4"]}


def test_group_videos_undatable_name_becomes_its_own_group() -> None:
    """Verifies that a name with no recognizable date falls back to its full stem as its own group key."""
    groups = group_videos(["randomfile.mp4", "M01_2024-01-15.mp4"])
    assert groups == {
        "randomfile": ["randomfile.mp4"],
        "M01": ["M01_2024-01-15.mp4"],
    }


# group_videos: caller-supplied pattern override
def test_group_videos_pattern_with_capturing_group() -> None:
    """Verifies that a pattern whose first capturing group participates keys the video by that group's text."""
    groups = group_videos(["M01_run1.mp4", "M01_run2.mp4", "M02_run1.mp4"], group_by_pattern=r"(M\d+)")
    assert groups == {
        "M01": ["M01_run1.mp4", "M01_run2.mp4"],
        "M02": ["M02_run1.mp4"],
    }


def test_group_videos_pattern_without_capturing_group_uses_whole_match() -> None:
    """Verifies that a pattern with no capturing groups keys the video by the whole matched text (group 0)."""
    groups = group_videos(["M01_run1.mp4", "M02_run1.mp4"], group_by_pattern=r"M\d+")
    assert groups == {"M01": ["M01_run1.mp4"], "M02": ["M02_run1.mp4"]}


def test_group_videos_pattern_non_participating_optional_group_falls_through_to_whole_match() -> None:
    """Verifies that a non-participating optional group yields None, so the key uses the whole match (group 0)."""
    # "M(\d+)?" matches "M" in "Mouse" with group 1 not participating -> key falls back to group 0 == "M".
    groups = group_videos(["Mouse.mp4"], group_by_pattern=r"M(\d+)?")
    assert groups == {"M": ["Mouse.mp4"]}


def test_group_videos_pattern_no_match_keys_by_full_stem() -> None:
    """Verifies that a stem the pattern fails to match is keyed by its full stem, forming its own group."""
    groups = group_videos(["control.mp4"], group_by_pattern=r"(M\d+)")
    assert groups == {"control": ["control.mp4"]}


def test_group_videos_pattern_empty_whole_match_falls_back_to_stem() -> None:
    """Verifies that when the pattern matches an empty span (group 0 is empty), the key falls back to the full stem."""
    # "x*" matches the empty string at position 0 of "abc"; group 0 is "" -> "" or stem -> "abc".
    groups = group_videos(["abc.mp4"], group_by_pattern=r"x*")
    assert groups == {"abc": ["abc.mp4"]}


def test_group_videos_invalid_pattern_raises_value_error() -> None:
    """Verifies that an invalid regular expression is reported as a ValueError chained from the underlying re.error."""
    with pytest.raises(ValueError, match="not a valid regular expression"):
        group_videos(["a.mp4"], group_by_pattern="(")


# _infer_group_key
def test_infer_group_key_trailing_date_identity_leads() -> None:
    """Verifies that when the date trails the name, the identity components before it form the key."""
    assert _infer_group_key("M01_2024-01-15") == "M01"


def test_infer_group_key_leading_date_identity_trails() -> None:
    """Verifies that when the date leads the name, the identity components after it form the key."""
    assert _infer_group_key("2024-01-15_M01") == "M01"


def test_infer_group_key_leading_date_drops_numeric_time_components() -> None:
    """Verifies that numeric time components between a leading date and the identity are dropped before keying."""
    assert _infer_group_key("2024-01-15_09-30-00_M01") == "M01"


def test_infer_group_key_normalizes_iso_t_separator() -> None:
    """Verifies that an ISO 8601 date-time 'T' between digits is normalized, exposing and stripping the fused date."""
    # A bare fused date-plus-time is fully consumed and leaves no identity -> None (documented contract).
    assert _infer_group_key("20240115T093000") is None
    # These two cases are load-bearing for the normalization. Without splitting the digit-adjacent 'T', the whole
    # "20240115T093000" run stays one non-numeric component (T is a letter), the date is never recognized, and the key
    # would collapse to None (falling back to the full stem in group_videos). The normalization exposes the compact
    # date so the surrounding identity survives.
    assert _infer_group_key("M01_20240115T093000") == "M01"
    assert _infer_group_key("2024-01-15T09-30-00_M01") == "M01"


def test_group_videos_iso_t_time_groups_with_delimited_date() -> None:
    """Verifies that an ISO 'T'-separated timestamp is normalized so its video groups with a plainly-dated sibling."""
    videos = ["/d/2024-01-15T09-30-00_M01.mp4", "M01_2024-02-20.mp4"]
    assert group_videos(videos) == {"M01": ["/d/2024-01-15T09-30-00_M01.mp4", "M01_2024-02-20.mp4"]}


def test_infer_group_key_multi_component_identity_is_joined() -> None:
    """Verifies that a multi-component identity preceding the date is joined with underscores."""
    assert _infer_group_key("MF_11_2025_07_21") == "MF_11"


def test_infer_group_key_empty_stem_returns_none() -> None:
    """Verifies that a stem with no alphanumeric components yields no pieces and returns None."""
    assert _infer_group_key("___") is None
    assert _infer_group_key("") is None


def test_infer_group_key_no_date_returns_none() -> None:
    """Verifies that a stem with no recognizable date span returns None so the caller can fall back."""
    assert _infer_group_key("Mouse_Session_1") is None


def test_infer_group_key_date_only_returns_none() -> None:
    """Verifies that a stem that is only a date leaves no identity components and returns None."""
    assert _infer_group_key("2024-01-15") is None


# _find_date_span
def test_find_date_span_numeric_triple_year_first() -> None:
    """Verifies that a year-first numeric triple is found as a three-component span."""
    assert _find_date_span(["M01", "2024", "01", "15"]) == (1, 4)


def test_find_date_span_numeric_triple_year_last() -> None:
    """Verifies that a year-last numeric triple is found as a three-component span."""
    assert _find_date_span(["M01", "15", "01", "2024"]) == (1, 4)


def test_find_date_span_named_triple() -> None:
    """Verifies that a named-month triple is found once the numeric-triple test fails at the same position."""
    assert _find_date_span(["M01", "15", "January", "2024"]) == (1, 4)


def test_find_date_span_named_month_year_pair() -> None:
    """Verifies that a month name beside a four-digit year is found as a two-component span."""
    assert _find_date_span(["M01", "January", "2024"]) == (1, 3)


def test_find_date_span_compact_piece() -> None:
    """Verifies that a single compact all-digit date is found as a one-component span."""
    assert _find_date_span(["M01", "15012024"]) == (1, 2)


def test_find_date_span_fused_named_piece() -> None:
    """Verifies that a single fused named-month date is found as a one-component span (compact test fails first)."""
    assert _find_date_span(["M01", "15Jan2024"]) == (1, 2)


def test_find_date_span_none_when_no_date() -> None:
    """Verifies that no span is returned when no component (or adjacency) forms a trusted date."""
    assert _find_date_span(["Mouse", "Session", "1"]) is None


# _is_month_day_pair
def test_is_month_day_pair_month_then_day() -> None:
    """Verifies that a month followed by a day is a valid month-day pair."""
    assert _is_month_day_pair("01", "15") is True


def test_is_month_day_pair_day_then_month() -> None:
    """Verifies that a day followed by a month is a valid month-day pair (reversed order)."""
    assert _is_month_day_pair("15", "01") is True


def test_is_month_day_pair_invalid() -> None:
    """Verifies that two components that cannot both be a month/day are not a pair."""
    assert _is_month_day_pair("13", "40") is False


# _is_numeric_date_triple
def test_is_numeric_date_triple_non_digit_rejected() -> None:
    """Verifies that a triple containing a non-digit component is not a numeric date."""
    assert _is_numeric_date_triple(("M01", "01", "15")) is False


def test_is_numeric_date_triple_year_first() -> None:
    """Verifies that a YYYY MM DD triple is a numeric date."""
    assert _is_numeric_date_triple(("2024", "01", "15")) is True


def test_is_numeric_date_triple_year_last() -> None:
    """Verifies that a DD MM YYYY triple is a numeric date."""
    assert _is_numeric_date_triple(("15", "01", "2024")) is True


def test_is_numeric_date_triple_no_year_rejected() -> None:
    """Verifies that an all-digit triple with no four-digit year at either end is not a numeric date."""
    assert _is_numeric_date_triple(("12", "01", "15")) is False


# _is_named_date_triple
def test_is_named_date_triple_no_month_name_rejected() -> None:
    """Verifies that a triple with no month name is not a named-month date."""
    assert _is_named_date_triple(("2024", "01", "15")) is False


def test_is_named_date_triple_two_month_names_rejected() -> None:
    """Verifies that a triple with two month names is not a named-month date."""
    assert _is_named_date_triple(("Jan", "Feb", "2024")) is False


def test_is_named_date_triple_non_digit_others_rejected() -> None:
    """Verifies that a named triple whose non-name components are not both digits is rejected."""
    assert _is_named_date_triple(("Jan", "M0", "2024")) is False


def test_is_named_date_triple_day_then_year() -> None:
    """Verifies that a month name with a day-then-year pairing is a named-month date."""
    assert _is_named_date_triple(("Jan", "15", "2024")) is True


def test_is_named_date_triple_year_then_day() -> None:
    """Verifies that a month name with a year-then-day pairing is a named-month date (reversed order)."""
    assert _is_named_date_triple(("Jan", "2024", "15")) is True


def test_is_named_date_triple_two_digit_year_trusted() -> None:
    """Verifies that the month name anchors the span, so a two-digit year is trusted alongside a day."""
    assert _is_named_date_triple(("Jan", "15", "24")) is True


def test_is_named_date_triple_neither_day_nor_year_rejected() -> None:
    """Verifies that digit components that are neither a valid day nor a plausible year are rejected."""
    assert _is_named_date_triple(("Jan", "40", "99")) is False


# _is_day_piece / _is_year_piece
def test_is_day_piece() -> None:
    """Verifies that a one- or two-digit day of month is recognized; an out-of-range value is not."""
    assert _is_day_piece("5") is True
    assert _is_day_piece("31") is True
    assert _is_day_piece("32") is False


def test_is_year_piece() -> None:
    """Verifies that a four-digit year and a bare two-digit year are recognized; a three-digit value is not."""
    assert _is_year_piece("2024") is True
    assert _is_year_piece("24") is True
    assert _is_year_piece("204") is False


# _is_named_month_year_pair
def test_is_named_month_year_pair_name_then_year() -> None:
    """Verifies that a month name followed by a four-digit year is a valid pair."""
    assert _is_named_month_year_pair("January", "2024") is True


def test_is_named_month_year_pair_year_then_name() -> None:
    """Verifies that a four-digit year followed by a month name is a valid pair (reversed order)."""
    assert _is_named_month_year_pair("2024", "January") is True


def test_is_named_month_year_pair_invalid() -> None:
    """Verifies that a component that is neither a month name nor a four-digit year is not part of a pair."""
    assert _is_named_month_year_pair("M01", "2024") is False


def test_is_named_month_year_pair_two_digit_year_rejected() -> None:
    """Verifies that only a full four-digit year is trusted beside a lone month name; a two-digit year is not a pair."""
    # With no day to reinforce the span, "Jan_23" must not read as a date (unlike the month-anchored named triple,
    # where a two-digit year is trusted). Guards against the pair mistakenly using the looser year predicate.
    assert _is_named_month_year_pair("Jan", "23") is False
    assert _is_named_month_year_pair("23", "Jan") is False


# _is_compact_date_piece
def test_is_compact_date_piece_non_digit_rejected() -> None:
    """Verifies that a component with non-digit characters is not a compact date."""
    assert _is_compact_date_piece("M01") is False


def test_is_compact_date_piece_year_first_ignores_trailing_time() -> None:
    """Verifies that a leading YYYYMMDD run qualifies, with any trailing time digits ignored."""
    assert _is_compact_date_piece("20240115") is True
    assert _is_compact_date_piece("20240115093000") is True


def test_is_compact_date_piece_year_last() -> None:
    """Verifies that an eight-digit DDMMYYYY run whose trailing four digits are a year qualifies."""
    assert _is_compact_date_piece("15012024") is True


def test_is_compact_date_piece_digit_but_not_date_rejected() -> None:
    """Verifies that an all-digit run that is neither year-first nor year-last is not a compact date."""
    assert _is_compact_date_piece("12345678") is False


# _is_fused_named_date_piece
def test_is_fused_named_date_piece_variants() -> None:
    """Verifies that fused named-month dates are recognized in day-first, name-first, and year-first forms."""
    assert _is_fused_named_date_piece("15Jan2024") is True
    assert _is_fused_named_date_piece("Jan2024") is True
    assert _is_fused_named_date_piece("2024Jan15") is True


def test_is_fused_named_date_piece_invalid() -> None:
    """Verifies that a component with no fused month-name/year date is rejected."""
    assert _is_fused_named_date_piece("M01") is False
