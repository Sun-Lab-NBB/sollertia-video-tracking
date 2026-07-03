"""Provides the parallel DeepLabCut frame-extraction pipelines, covering both k-means and model-outlier selection.

The k-means pipeline clusters raw video to bootstrap a project's training frames; the outlier pipeline reads a trained
model's predictions and extracts the frames it most likely got wrong to refine the model. Both decode one video per
worker pinned to a disjoint block of CPU cores and share the same CPU-allocation and progress-reporting logic.
"""

from .cpu_allocation import DEFAULT_RESERVED_CORE_COUNT
from .video_grouping import group_videos
from .outlier_pipeline import OutlierExtractionSummary, extract_outlier_frames_parallel
from .extraction_pipeline import FrameExtractionSummary, extract_frames_kmeans

__all__ = [
    "DEFAULT_RESERVED_CORE_COUNT",
    "FrameExtractionSummary",
    "OutlierExtractionSummary",
    "extract_frames_kmeans",
    "extract_outlier_frames_parallel",
    "group_videos",
]
