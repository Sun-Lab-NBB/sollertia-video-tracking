"""Provides the outlier-frame detection algorithms that flag candidate frames from a video's DeepLabCut predictions."""

import warnings

import numpy as np
import pandas as pd
from deeplabcut.refine_training_dataset.outlier_frames import FitSARIMAXModel

OUTLIER_ALGORITHMS: tuple[str, ...] = ("jump", "uncertain", "fitting", "list")
"""The supported outlier-detection algorithm names, mirroring DeepLabCut's non-interactive selection methods."""


def uncertain_outlier_indices(predictions: pd.DataFrame, p_bound: float) -> list[int]:
    """Flags frames in which any comparison bodypart was predicted below the confidence bound.

    Args:
        predictions: The prediction table for one video, restricted to the comparison bodyparts and sliced to the
            configured start/stop window, with a ``scorer / [individuals] / bodyparts / coords`` column MultiIndex.
        p_bound: The likelihood below which a keypoint flags its frame as a putative outlier.

    Returns:
        The frame indices, in the prediction table's own index, that hold at least one low-confidence keypoint.
    """
    likelihoods = predictions.xs("likelihood", level="coords", axis=1)
    # pandas-stubs types the DataFrame.xs(axis=1) cross-section as DataFrame | Series, so the boolean reduction is
    # seen as a Series that rejects axis=1; the runtime object is always a DataFrame.
    return predictions.index[(likelihoods < p_bound).any(axis=1)].tolist()  # type: ignore[arg-type]


def jump_outlier_indices(predictions: pd.DataFrame, epsilon: float) -> list[int]:
    """Flags frames in which any comparison bodypart jumped further than ``epsilon`` from the previous frame.

    Notes:
        The per-bodypart squared displacement is summed across the ``x`` and ``y`` channels (and, for multi-animal
        projects, across the individuals sharing a bodypart name), matching DeepLabCut's own grouping. The first frame
        carries a NaN displacement and is never flagged.

    Args:
        predictions: The prediction table for one video, restricted to the comparison bodyparts and sliced to the
            configured start/stop window, with a ``scorer / [individuals] / bodyparts / coords`` column MultiIndex.
        epsilon: The Euclidean distance, in pixels, a bodypart may move between consecutive frames before its frame is
            flagged.

    Returns:
        The frame indices, in the prediction table's own index, that hold at least one over-threshold jump.
    """
    with warnings.catch_warnings():
        # This intentionally mirrors DeepLabCut's own "jump" branch in extract_outlier_frames
        # (deeplabcut/refine_training_dataset/outlier_frames.py), which uses the same axis=1 groupby; matching it keeps
        # flagged frames identical to upstream. pandas 2.x deprecates axis=1 groupby but still evaluates it, and DLC
        # 3.0 itself calls it, so the env stays on pandas 2.x regardless. Follow DLC if it migrates off axis=1 groupby.
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        squared_step = predictions.diff(axis=0) ** 2
        squared_step = squared_step.drop("likelihood", axis=1, level="coords")
        per_bodypart = squared_step.groupby(level="bodyparts", axis=1).sum()  # type: ignore[call-overload]
    return predictions.index[(per_bodypart > epsilon**2).any(axis=1)].tolist()


def fitting_keypoint_series(predictions: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Splits a prediction table into per-keypoint ``(x, y, likelihood)`` trajectories for SARIMAX fitting.

    Notes:
        The columns are reshaped into ``(frames, keypoints, 3)`` exactly as DeepLabCut does before fitting, so every
        ``(individual, bodypart)`` keypoint becomes one work item regardless of whether the project is single- or
        multi-animal.

    Args:
        predictions: The prediction table for one video, restricted to the comparison bodyparts and sliced to the
            configured start/stop window.

    Returns:
        One ``(x, y, likelihood)`` tuple of per-frame arrays for each keypoint, in column order.
    """
    channels = predictions.to_numpy().reshape((predictions.shape[0], -1, 3))
    x, y, likelihood = channels.T
    return [
        (
            np.ascontiguousarray(x[keypoint]),
            np.ascontiguousarray(y[keypoint]),
            np.ascontiguousarray(likelihood[keypoint]),
        )
        for keypoint in range(x.shape[0])
    ]


def fit_keypoint_distance(
    x: np.ndarray,
    y: np.ndarray,
    likelihood: np.ndarray,
    p_bound: float,
    alpha: float,
    ar_degree: int,
    ma_degree: int,
) -> np.ndarray:
    """Fits a SARIMAX model to one keypoint's trajectory and returns its per-frame deviation from the fit.

    Notes:
        Confident positions drive the fit while low-confidence positions are treated as missing, following
        DeepLabCut's ``FitSARIMAXModel``. When too few positions are confident, or the fit itself raises, the returned
        deviation is all NaN, so the keypoint contributes nothing to the frame's averaged deviation and one bad
        trajectory cannot abort the shared fitting pool.

    Args:
        x: The keypoint's per-frame horizontal positions.
        y: The keypoint's per-frame vertical positions.
        likelihood: The keypoint's per-frame prediction confidences.
        p_bound: The likelihood below which a position is treated as missing while fitting.
        alpha: The significance level for the fitted model's confidence interval.
        ar_degree: The autoregressive degree of the SARIMAX model.
        ma_degree: The moving-average degree of the SARIMAX model.

    Returns:
        The per-frame Euclidean distance between the observed position and the model's fitted position.
    """
    with warnings.catch_warnings():
        # SARIMAX routinely reports non-convergence on noisy keypoint trajectories; the deviation is still usable, and
        # these fits run in pool workers that do not redirect their streams, so the warnings are silenced at the source.
        warnings.simplefilter("ignore")
        try:
            mean_x, _ = FitSARIMAXModel(x, likelihood, p_bound, alpha, ar_degree, ma_degree)
            mean_y, _ = FitSARIMAXModel(y, likelihood, p_bound, alpha, ar_degree, ma_degree)
        except Exception:  # noqa: BLE001 -- a keypoint whose fit raises yields NaN, like one with too few points.
            return np.full(x.shape, np.nan, dtype=float)
    return np.sqrt((x - mean_x) ** 2 + (y - mean_y) ** 2)


def fitting_outlier_indices(distances: list[np.ndarray], num_frames_to_pick: int, epsilon: float) -> list[int]:
    """Selects outlier frames from per-keypoint SARIMAX deviations, averaging across keypoints as DeepLabCut does.

    Notes:
        The per-frame deviation is the keypoint-averaged distance to the fitted trajectory, ignoring keypoints whose
        fit was skipped. Frames deviating by more than ``epsilon`` are flagged; when too few qualify, the most deviant
        frames are taken instead, matching DeepLabCut's fallback. The returned indices are positional within the
        supplied window, as in DeepLabCut.

    Args:
        distances: One per-frame deviation array for each keypoint, produced by ``fit_keypoint_distance``.
        num_frames_to_pick: The project's ``numframes2pick``, used to size the fallback selection.
        epsilon: The averaged deviation, in pixels, above which a frame is flagged.

    Returns:
        The positional indices of the flagged frames within the supplied window.
    """
    stacked = np.vstack(distances)
    with warnings.catch_warnings():
        # A frame whose keypoints were all skipped averages over an empty slice; NaN is the intended, unflagged result.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        deviation = np.nanmean(stacked, axis=0)
    candidates = np.flatnonzero(deviation > epsilon)
    if len(candidates) < num_frames_to_pick * 2 and len(deviation) > num_frames_to_pick * 2:
        candidates = np.argsort(deviation)[::-1][: num_frames_to_pick * 2]
    return candidates.tolist()
