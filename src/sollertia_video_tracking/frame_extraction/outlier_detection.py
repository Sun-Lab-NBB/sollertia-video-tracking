"""Provides the outlier-frame detection algorithms that flag candidate frames from a video's DeepLabCut predictions."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
import warnings

import numpy as np
from deeplabcut.refine_training_dataset.outlier_frames import FitSARIMAXModel

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

type KeypointSeries = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
"""One keypoint's per-frame horizontal positions, vertical positions, and confidences, as three float arrays."""

_DISCARDED_INTERVAL_ALPHA: float = 0.01
"""The significance level passed to DeepLabCut's ``FitSARIMAXModel``. Only the fitted mean trajectory is used to measure
each frame's deviation; the confidence interval, the sole output this value influences, is discarded. It is a fixed
constant rather than a parameter because it cannot change which frames are flagged, matching DeepLabCut's own
``fitting`` selection, which thresholds the mean deviation and ignores the interval."""


class OutlierAlgorithm(StrEnum):
    """Defines the supported frame-selection modes.

    ``jump``, ``uncertain``, and ``fitting`` flag likely-wrong frames from a trained model's predictions; ``list``
    extracts an explicit, caller-supplied index list instead.
    """

    JUMP = "jump"
    """Flags frames in which a bodypart moves further than the pixel threshold from the previous frame."""
    UNCERTAIN = "uncertain"
    """Flags frames holding any bodypart predicted below the confidence threshold."""
    FITTING = "fitting"
    """Flags frames that depart from a fitted per-keypoint motion trajectory."""
    LIST = "list"
    """Extracts an explicit, caller-supplied list of frame indices instead of detecting outliers."""


def uncertain_outlier_indices(predictions: pd.DataFrame, minimum_confidence: float) -> list[int]:
    """Flags frames in which any comparison bodypart was predicted below the confidence bound.

    Args:
        predictions: The prediction table for one video, restricted to the comparison bodyparts and sliced to the
            configured start/stop window, with a ``scorer / [individuals] / bodyparts / coords`` column MultiIndex.
        minimum_confidence: The likelihood below which a keypoint flags its frame as a putative outlier.

    Returns:
        The frame indices, in the prediction table's own index, that hold at least one low-confidence keypoint.
    """
    likelihoods = predictions.xs(key="likelihood", level="coords", axis=1)
    # pandas-stubs types the DataFrame.xs(axis=1) cross-section as DataFrame | Series, so the boolean reduction is
    # seen as a Series that rejects axis=1; the runtime object is always a DataFrame.
    return predictions.index[(likelihoods < minimum_confidence).any(axis=1)].tolist()  # type: ignore[arg-type]


def jump_outlier_indices(predictions: pd.DataFrame, pixel_distance_threshold: float) -> list[int]:
    """Flags frames in which any comparison bodypart jumped further than the threshold from the previous frame.

    Notes:
        The per-bodypart squared displacement is summed across the ``x`` and ``y`` channels (and, for multi-animal
        projects, across the individuals sharing a bodypart name), matching DeepLabCut's own grouping. The first frame
        carries a NaN displacement and is never flagged.

    Args:
        predictions: The prediction table for one video, restricted to the comparison bodyparts and sliced to the
            configured start/stop window, with a ``scorer / [individuals] / bodyparts / coords`` column MultiIndex.
        pixel_distance_threshold: The Euclidean distance, in pixels, a bodypart may move between consecutive frames
            before its frame is flagged.

    Returns:
        The frame indices, in the prediction table's own index, that hold at least one over-threshold jump.
    """
    with warnings.catch_warnings():
        # Mirrors DeepLabCut's own "jump" branch in extract_outlier_frames
        # (deeplabcut/refine_training_dataset/outlier_frames.py), which uses the same axis=1 groupby; matching it keeps
        # flagged frames identical to upstream. pandas 2.x deprecates axis=1 groupby but still evaluates it, and DLC
        # 3.0 itself calls it, so the env stays on pandas 2.x regardless. Follow DLC if it migrates off axis=1 groupby.
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        squared_step = predictions.diff(axis=0) ** 2
        squared_step = squared_step.drop(labels="likelihood", axis=1, level="coords")
        per_bodypart = squared_step.groupby(level="bodyparts", axis=1).sum()  # type: ignore[call-overload]
    return predictions.index[(per_bodypart > pixel_distance_threshold**2).any(axis=1)].tolist()


def fitting_keypoint_series(predictions: pd.DataFrame) -> list[KeypointSeries]:
    """Splits a prediction table into per-keypoint position-and-confidence trajectories for SARIMAX fitting.

    Notes:
        The columns are reshaped into ``(frames, keypoints, 3)`` exactly as DeepLabCut does before fitting, so every
        ``(individual, bodypart)`` keypoint becomes one work item regardless of whether the project is single- or
        multi-animal.

    Args:
        predictions: The prediction table for one video, restricted to the comparison bodyparts and sliced to the
            configured start/stop window.

    Returns:
        One ``(horizontal_positions, vertical_positions, confidences)`` tuple of per-frame arrays for each keypoint,
        in column order.
    """
    channels = predictions.to_numpy(dtype=np.float64).reshape((predictions.shape[0], -1, 3))
    horizontal_positions, vertical_positions, confidences = channels.T
    return [
        (
            np.ascontiguousarray(horizontal_positions[keypoint]),
            np.ascontiguousarray(vertical_positions[keypoint]),
            np.ascontiguousarray(confidences[keypoint]),
        )
        for keypoint in range(horizontal_positions.shape[0])
    ]


def fit_keypoint_distance(
    horizontal_positions: NDArray[np.float64],
    vertical_positions: NDArray[np.float64],
    confidences: NDArray[np.float64],
    minimum_confidence: float,
    autoregressive_degree: int,
    moving_average_degree: int,
) -> NDArray[np.float64]:
    """Fits a SARIMAX model to one keypoint's trajectory and returns its per-frame deviation from the fit.

    Notes:
        Confident positions drive the fit while low-confidence positions are treated as missing, following
        DeepLabCut's ``FitSARIMAXModel``. When too few positions are confident, or the fit itself raises, the returned
        deviation is all NaN, so the keypoint contributes nothing to the frame's averaged deviation and one bad
        trajectory cannot abort the shared fitting pool.

    Args:
        horizontal_positions: The keypoint's per-frame horizontal positions.
        vertical_positions: The keypoint's per-frame vertical positions.
        confidences: The keypoint's per-frame prediction confidences.
        minimum_confidence: The likelihood below which a position is treated as missing while fitting.
        autoregressive_degree: The number of lagged observations the SARIMAX fit regresses each point on.
        moving_average_degree: The number of lagged residual terms the SARIMAX fit includes.

    Returns:
        The per-frame Euclidean distance between the observed position and the model's fitted position.
    """
    with warnings.catch_warnings():
        # SARIMAX routinely reports non-convergence on noisy keypoint trajectories; the deviation is still usable, and
        # these fits run in pool workers that do not redirect their streams, so the warnings are silenced at the source.
        warnings.simplefilter("ignore")
        try:
            mean_horizontal, _ = FitSARIMAXModel(
                x=horizontal_positions,
                p=confidences,
                pcutoff=minimum_confidence,
                alpha=_DISCARDED_INTERVAL_ALPHA,
                ARdegree=autoregressive_degree,
                MAdegree=moving_average_degree,
            )
            mean_vertical, _ = FitSARIMAXModel(
                x=vertical_positions,
                p=confidences,
                pcutoff=minimum_confidence,
                alpha=_DISCARDED_INTERVAL_ALPHA,
                ARdegree=autoregressive_degree,
                MAdegree=moving_average_degree,
            )
        except Exception:
            return np.full(horizontal_positions.shape, fill_value=np.nan, dtype=np.float64)
    return np.sqrt((horizontal_positions - mean_horizontal) ** 2 + (vertical_positions - mean_vertical) ** 2)


def fitting_outlier_indices(
    keypoint_deviations: list[NDArray[np.float64]], frames_per_video_count: int, pixel_distance_threshold: float
) -> list[int]:
    """Selects outlier frames from per-keypoint SARIMAX deviations, averaging across keypoints as DeepLabCut does.

    Notes:
        The per-frame deviation is the keypoint-averaged distance to the fitted trajectory, ignoring keypoints whose
        fit was skipped. Frames deviating by more than ``pixel_distance_threshold`` are flagged; when too few qualify,
        the most deviant frames are taken instead, matching DeepLabCut's fallback. The returned indices are positional
        within the supplied window, as in DeepLabCut.

    Args:
        keypoint_deviations: One per-frame deviation array for each keypoint, produced by ``fit_keypoint_distance``.
        frames_per_video_count: The project's ``numframes2pick``, used to size the fallback selection.
        pixel_distance_threshold: The averaged deviation, in pixels, above which a frame is flagged.

    Returns:
        The positional indices of the flagged frames within the supplied window.
    """
    stacked_deviations = np.vstack(keypoint_deviations)
    with warnings.catch_warnings():
        # A frame whose keypoints were all skipped averages over an empty slice; NaN is the intended, unflagged result.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_deviation = np.nanmean(stacked_deviations, axis=0)
    candidate_indices = np.flatnonzero(mean_deviation > pixel_distance_threshold)
    if len(candidate_indices) < frames_per_video_count * 2 and len(mean_deviation) > frames_per_video_count * 2:
        candidate_indices = np.argsort(mean_deviation)[::-1][: frames_per_video_count * 2]
    return candidate_indices.tolist()
