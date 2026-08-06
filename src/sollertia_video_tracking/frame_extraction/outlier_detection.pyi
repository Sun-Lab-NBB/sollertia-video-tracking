from enum import StrEnum

import numpy as np
import pandas as pd
from numpy.typing import NDArray

type _KeypointSeries = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
_DISCARDED_INTERVAL_ALPHA: float

class OutlierAlgorithm(StrEnum):
    JUMP = "jump"
    UNCERTAIN = "uncertain"
    FITTING = "fitting"
    LIST = "list"

def uncertain_outlier_indices(predictions: pd.DataFrame, minimum_confidence: float) -> list[int]: ...
def jump_outlier_indices(predictions: pd.DataFrame, pixel_distance_threshold: float) -> list[int]: ...
def fitting_keypoint_count(predictions: pd.DataFrame) -> int: ...
def fitting_keypoint_series(predictions: pd.DataFrame, keypoint_index: int) -> _KeypointSeries: ...
def fit_keypoint_distance(
    horizontal_positions: NDArray[np.float64],
    vertical_positions: NDArray[np.float64],
    confidences: NDArray[np.float64],
    minimum_confidence: float,
    autoregressive_degree: int,
    moving_average_degree: int,
) -> NDArray[np.float64]: ...
def fitting_outlier_indices(
    keypoint_deviations: list[NDArray[np.float64]], frames_per_video_count: int, pixel_distance_threshold: float
) -> list[int]: ...
