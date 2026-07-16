"""Contains tests for the outlier-frame detection algorithms that flag candidate frames from DeepLabCut predictions."""

import numpy as np
import pandas as pd
import pytest

from sollertia_video_tracking.frame_extraction import outlier_detection as od
from sollertia_video_tracking.frame_extraction.outlier_detection import (
    OutlierAlgorithm,
    jump_outlier_indices,
    fit_keypoint_distance,
    fitting_keypoint_series,
    fitting_outlier_indices,
    uncertain_outlier_indices,
)


def _make_predictions(
    per_bodypart: dict[str, tuple[list[float], list[float], list[float]]],
    index: list[int] | None = None,
    scorer: str = "DLC",
) -> pd.DataFrame:
    """Builds a single-animal DeepLabCut prediction table from per-bodypart (x, y, likelihood) triples.

    The columns carry the ``scorer / bodyparts / coords`` MultiIndex the detection functions read, ordered so that each
    keypoint's ``x, y, likelihood`` channels are contiguous, matching the reshape in fitting_keypoint_series.
    """
    bodyparts = list(per_bodypart.keys())
    frame_count = len(next(iter(per_bodypart.values()))[0])
    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    data = np.zeros((frame_count, len(bodyparts) * 3), dtype=np.float64)
    for position, bodypart in enumerate(bodyparts):
        horizontal, vertical, likelihood = per_bodypart[bodypart]
        data[:, position * 3 + 0] = horizontal
        data[:, position * 3 + 1] = vertical
        data[:, position * 3 + 2] = likelihood
    return pd.DataFrame(data, columns=columns, index=index)


# OutlierAlgorithm enum
def test_outlier_algorithm_members_are_lowercase_strings() -> None:
    """Verifies that every OutlierAlgorithm member is a StrEnum whose value equals its lowercase name."""
    assert OutlierAlgorithm.JUMP == "jump"
    assert OutlierAlgorithm.UNCERTAIN == "uncertain"
    assert OutlierAlgorithm.FITTING == "fitting"
    assert OutlierAlgorithm.LIST == "list"
    # StrEnum members compare and behave as plain strings.
    assert OutlierAlgorithm("fitting") is OutlierAlgorithm.FITTING
    assert isinstance(OutlierAlgorithm.LIST, str)
    assert {member.value for member in OutlierAlgorithm} == {"jump", "uncertain", "fitting", "list"}


# uncertain_outlier_indices
def test_uncertain_flags_frames_with_any_low_confidence_keypoint() -> None:
    """Verifies that a frame with any below-confidence bodypart is flagged, returning its own index label."""
    predictions = _make_predictions(
        {
            # Frame 1 has a low bp0 likelihood; frame 3 has a low bp1 likelihood; frames 0 and 2 are confident.
            "bp0": ([0, 0, 0, 0], [0, 0, 0, 0], [0.9, 0.3, 0.9, 0.9]),
            "bp1": ([1, 1, 1, 1], [1, 1, 1, 1], [0.9, 0.9, 0.9, 0.4]),
        },
        index=[10, 11, 12, 13],
    )
    # The returned values are the table's own index labels (offset here), not positional row numbers.
    assert uncertain_outlier_indices(predictions, minimum_confidence=0.5) == [11, 13]


def test_uncertain_returns_empty_when_all_confident() -> None:
    """Verifies that no frame is flagged when every keypoint clears the confidence bound."""
    predictions = _make_predictions(
        {"bp0": ([0, 0], [0, 0], [0.95, 0.99]), "bp1": ([1, 1], [1, 1], [0.9, 0.9])},
    )
    assert uncertain_outlier_indices(predictions, minimum_confidence=0.5) == []


# jump_outlier_indices
def test_jump_flags_over_threshold_displacement_and_never_first_frame() -> None:
    """Verifies that a bodypart moving past the threshold flags its frame; the NaN-displacement first frame is not."""
    predictions = _make_predictions(
        {
            # bp0 jumps 10px between frame 1 and 2 (>5px threshold); frame 0 carries a NaN diff and is never flagged.
            "bp0": ([0, 0, 10, 10], [0, 0, 0, 0], [0.5, 0.1, 0.9, 0.2]),
            "bp1": ([5, 5, 5, 5], [5, 5, 5, 5], [0.5, 0.5, 0.5, 0.5]),
        },
    )
    assert jump_outlier_indices(predictions, pixel_distance_threshold=5.0) == [2]


def test_jump_ignores_likelihood_channel() -> None:
    """Verifies that likelihood swings are dropped before the displacement test, so they cannot flag a frame alone."""
    predictions = _make_predictions(
        {
            # Positions are perfectly still; only likelihood changes wildly, and it must be ignored.
            "bp0": ([0, 0, 0], [0, 0, 0], [0.1, 0.9, 0.1]),
            "bp1": ([2, 2, 2], [2, 2, 2], [0.9, 0.1, 0.9]),
        },
    )
    assert jump_outlier_indices(predictions, pixel_distance_threshold=1.0) == []


def test_jump_sums_displacement_across_individuals_sharing_a_bodypart() -> None:
    """Verifies that columns sharing a bodypart name are summed together, matching DeepLabCut's axis=1 groupby."""
    # Two individuals each contribute an "ear" keypoint; the groupby sums their squared displacements per frame.
    columns = pd.MultiIndex.from_product(
        [["DLC"], ["mouse_a", "mouse_b"], ["ear"], ["x", "y", "likelihood"]],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    data = np.zeros((3, 6), dtype=np.float64)
    # mouse_a.ear moves 3px in x on frame 2, mouse_b.ear moves 4px in x on frame 2: summed squared step = 9 + 16 = 25.
    data[:, 0] = [0.0, 0.0, 3.0]  # mouse_a ear x
    data[:, 3] = [0.0, 0.0, 4.0]  # mouse_b ear x
    predictions = pd.DataFrame(data, columns=columns)
    # sqrt(25) == 5, so a 4px threshold flags frame 2 while a 6px threshold does not.
    assert jump_outlier_indices(predictions, pixel_distance_threshold=4.0) == [2]
    assert jump_outlier_indices(predictions, pixel_distance_threshold=6.0) == []


# fitting_keypoint_series
def test_fitting_keypoint_series_splits_into_contiguous_channel_arrays() -> None:
    """Verifies that each keypoint becomes one (x, y, likelihood) tuple of contiguous arrays, in column order."""
    predictions = _make_predictions(
        {
            "bp0": ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.7, 0.8, 0.9]),
            "bp1": ([10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [0.1, 0.2, 0.3]),
        },
    )
    series = fitting_keypoint_series(predictions)
    assert len(series) == 2

    horizontal_0, vertical_0, confidence_0 = series[0]
    np.testing.assert_array_equal(horizontal_0, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(vertical_0, [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(confidence_0, [0.7, 0.8, 0.9])

    horizontal_1, vertical_1, confidence_1 = series[1]
    np.testing.assert_array_equal(horizontal_1, [10.0, 11.0, 12.0])
    np.testing.assert_array_equal(vertical_1, [13.0, 14.0, 15.0])
    np.testing.assert_array_equal(confidence_1, [0.1, 0.2, 0.3])

    # The arrays are made contiguous for the downstream SARIMAX fit.
    for channel_array in (horizontal_0, vertical_0, confidence_0):
        assert channel_array.flags["C_CONTIGUOUS"]
        assert channel_array.dtype == np.float64


# fit_keypoint_distance
def test_fit_keypoint_distance_returns_euclidean_deviation_from_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the per-frame deviation is the Euclidean distance between observed and fitted positions."""
    recorded_calls: list[dict[str, object]] = []

    def _fake_fit(x, p, pcutoff, alpha, ARdegree, MAdegree):  # noqa: N803 -- mirrors FitSARIMAXModel's signature.
        recorded_calls.append(
            {"x": x, "p": p, "pcutoff": pcutoff, "alpha": alpha, "ARdegree": ARdegree, "MAdegree": MAdegree},
        )
        # A fitted trajectory shifted 1px below the observation on both axes -> per-frame deviation sqrt(1 + 1).
        return x - 1.0, None

    monkeypatch.setattr(od, "FitSARIMAXModel", _fake_fit)

    horizontal = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    vertical = np.array([5.0, 6.0, 7.0], dtype=np.float64)
    # Distinct, non-uniform confidences so an assertion on the forwarded `p` cannot pass by coincidence.
    confidences = np.array([0.9, 0.8, 0.7], dtype=np.float64)

    deviation = fit_keypoint_distance(
        horizontal_positions=horizontal,
        vertical_positions=vertical,
        confidences=confidences,
        minimum_confidence=0.6,
        autoregressive_degree=2,
        moving_average_degree=1,
    )
    np.testing.assert_allclose(deviation, np.full(3, np.sqrt(2.0)))

    # The horizontal axis is fitted first and the vertical axis second, each on its own positions but sharing the
    # keypoint's confidences, the caller's pcutoff and degrees, and the module's fixed discarded alpha.
    assert len(recorded_calls) == 2
    np.testing.assert_array_equal(recorded_calls[0]["x"], horizontal)
    np.testing.assert_array_equal(recorded_calls[1]["x"], vertical)
    for call in recorded_calls:
        np.testing.assert_array_equal(call["p"], confidences)
        assert call["pcutoff"] == 0.6
        assert call["alpha"] == od._DISCARDED_INTERVAL_ALPHA
        assert call["ARdegree"] == 2
        assert call["MAdegree"] == 1


def test_fit_keypoint_distance_returns_all_nan_when_fit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a keypoint whose SARIMAX fit raises yields an all-NaN deviation instead of aborting the pool."""

    def _raising_fit(**_kwargs):
        # Any exception from the fit must be swallowed into an all-NaN deviation.
        raise RuntimeError

    monkeypatch.setattr(od, "FitSARIMAXModel", _raising_fit)

    horizontal = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    vertical = np.array([5.0, 6.0, 7.0], dtype=np.float64)
    confidences = np.array([0.9, 0.9, 0.9], dtype=np.float64)

    deviation = fit_keypoint_distance(
        horizontal_positions=horizontal,
        vertical_positions=vertical,
        confidences=confidences,
        minimum_confidence=0.6,
        autoregressive_degree=1,
        moving_average_degree=1,
    )
    assert deviation.shape == (3,)
    assert deviation.dtype == np.float64
    assert np.isnan(deviation).all()


# fitting_outlier_indices
def test_fitting_outlier_indices_threshold_branch() -> None:
    """Verifies that frames whose keypoint-averaged deviation exceeds the threshold are flagged by positional index."""
    # A single keypoint's deviations; only frames 1 and 3 exceed the 5px threshold.
    deviations = [np.array([1.0, 10.0, 2.0, 20.0])]
    # numframes2pick=2 -> fallback needs len(mean)=4 > 4, which is false, so the plain threshold result is returned.
    assert fitting_outlier_indices(deviations, frames_per_video_count=2, pixel_distance_threshold=5.0) == [1, 3]


def test_fitting_outlier_indices_returns_all_qualifiers_without_truncation() -> None:
    """Verifies that when enough frames clear the threshold, every qualifier is returned and truncation is skipped."""
    # All five frames clear the 1px threshold. With numframes2pick=1 the fallback cap would be 2, but the first
    # condition (len(candidates) < 2) is False, so the fallback is skipped and all five qualifiers are returned.
    deviations = [np.array([2.0, 3.0, 4.0, 5.0, 6.0])]
    flagged = fitting_outlier_indices(deviations, frames_per_video_count=1, pixel_distance_threshold=1.0)
    assert flagged == [0, 1, 2, 3, 4]


def test_fitting_outlier_indices_fallback_takes_most_deviant() -> None:
    """Verifies that when too few frames clear the threshold, most-deviant frames are taken (DeepLabCut's fallback)."""
    deviations = [np.array([1.0, 100.0, 2.0, 50.0, 3.0])]
    # Threshold is unreachably high so zero frames qualify (0 < 2) while len(mean)=5 > 2 triggers the fallback,
    # which takes the top numframes2pick*2 == 2 most deviant frames: indices 1 (100) and 3 (50).
    assert fitting_outlier_indices(deviations, frames_per_video_count=1, pixel_distance_threshold=1000.0) == [1, 3]


def test_fitting_outlier_indices_all_skipped_keypoints_frame_is_nan_and_unflagged() -> None:
    """Verifies that a frame whose keypoints were all skipped averages an empty slice to NaN and stays unflagged."""
    # Frames 0 and 2 are NaN across both keypoints (nanmean over an empty slice); only frame 1 has real deviations.
    deviations = [np.array([np.nan, 5.0, np.nan]), np.array([np.nan, 7.0, np.nan])]
    # len(mean)=3 is not > numframes2pick*2 == 4, so the threshold branch is used and NaN frames never pass > 1.
    assert fitting_outlier_indices(deviations, frames_per_video_count=2, pixel_distance_threshold=1.0) == [1]
