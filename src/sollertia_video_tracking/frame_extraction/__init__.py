"""Provides the parallel DeepLabCut k-means and model-outlier frame-extraction pipelines."""

from .cpu_allocation import DEFAULT_RESERVED_CORE_COUNT
from .video_grouping import group_videos
from .outlier_pipeline import (
    TrackingMethod,
    ExtractionAlgorithm,
    OutlierExtractionSummary,
    extract_outlier_frames_parallel,
)
from .outlier_detection import OutlierAlgorithm
from .extraction_pipeline import FrameExtractionSummary, extract_frames_kmeans

__all__ = [
    "DEFAULT_RESERVED_CORE_COUNT",
    "ExtractionAlgorithm",
    "FrameExtractionSummary",
    "OutlierAlgorithm",
    "OutlierExtractionSummary",
    "TrackingMethod",
    "extract_frames_kmeans",
    "extract_outlier_frames_parallel",
    "group_videos",
]
