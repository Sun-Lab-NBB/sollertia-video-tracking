"""Provides the parallel DeepLabCut k-means and model-outlier frame-extraction pipelines."""

from .progress import AggregateBar, make_progress_reporter
from .utilities import (
    PurgeSummary,
    RefinementStatusSummary,
    RefinementDirectoryStatus,
    purge_labeled_data,
    summarize_refinement_status,
)
from .cpu_allocation import plan_core_allocation
from .outlier_pipeline import (
    TrackingMethod,
    ExtractionAlgorithm,
    OutlierExtractionSummary,
    extract_outlier_frames_parallel,
)
from .outlier_detection import OutlierAlgorithm
from .extraction_pipeline import FrameExtractionSummary, extract_frames_kmeans

__all__ = [
    "AggregateBar",
    "ExtractionAlgorithm",
    "FrameExtractionSummary",
    "OutlierAlgorithm",
    "OutlierExtractionSummary",
    "PurgeSummary",
    "RefinementDirectoryStatus",
    "RefinementStatusSummary",
    "TrackingMethod",
    "extract_frames_kmeans",
    "extract_outlier_frames_parallel",
    "make_progress_reporter",
    "plan_core_allocation",
    "purge_labeled_data",
    "summarize_refinement_status",
]
