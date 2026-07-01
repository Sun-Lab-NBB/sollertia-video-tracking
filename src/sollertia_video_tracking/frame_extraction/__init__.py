"""Provides the parallel DeepLabCut k-means frame-extraction pipeline and its CPU-allocation logic."""

from .pipeline import FrameExtractionSummary, extract_frames_kmeans
from .cpu_allocation import DEFAULT_RESERVE_CORES

__all__ = [
    "DEFAULT_RESERVE_CORES",
    "FrameExtractionSummary",
    "extract_frames_kmeans",
]
