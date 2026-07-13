"""Provides assets for designing and deploying DeepLabCut video tracking pipelines within the Sollertia platform.

See the `documentation <https://sollertia-video-tracking-api-docs.netlify.app/>`_ for the description of available
assets. See the `source code repository <https://github.com/Sun-Lab-NBB/sollertia-video-tracking>`_ for more details.

Authors: Ivan Kondratyev (Inkaros)
"""

import os

# Pins the native math-library thread pools to one thread per worker and quiets OpenCV's logging. These environment
# variables are read when NumPy, OpenCV, and DeepLabCut initialize their native backends, so they must be set before
# the library's own imports (below) pull those backends in. The spawned extraction workers inherit this environment.
for _thread_limit_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_limit_variable, "1")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

# Hard-pins matplotlib to its non-interactive backend before DeepLabCut's outlier-extraction module imports pyplot.
# This library only ever writes frames to disk and targets headless compute servers, so the Agg backend is forced
# regardless of any inherited MPLBACKEND, keeping the spawned workers display-independent.
os.environ["MPLBACKEND"] = "Agg"

from .training import (  # noqa: E402 - after thread-limit setup
    TrainingSummary,
    OptimizationProfile,
    TrainingDatasetSummary,
    train_model,
    create_training_dataset,
    get_available_augmenters,
    get_available_pose_models,
    get_available_super_animals,
    resolve_optimization_profile,
    build_superanimal_weight_init,
    get_available_object_detectors,
    build_conditional_top_down_conditions,
)
from .inference import (  # noqa: E402 - after thread-limit setup
    InferenceProfile,
    InferenceSummary,
    run_inference,
    resolve_inference_profile,
)
from .frame_extraction import (  # noqa: E402 - after thread-limit setup
    PurgeSummary,
    TrackingMethod,
    OutlierAlgorithm,
    ExtractionAlgorithm,
    FrameExtractionSummary,
    RefinementStatusSummary,
    OutlierExtractionSummary,
    RefinementDirectoryStatus,
    purge_labeled_data,
    extract_frames_kmeans,
    summarize_refinement_status,
    extract_outlier_frames_parallel,
)

__all__ = [
    "ExtractionAlgorithm",
    "FrameExtractionSummary",
    "InferenceProfile",
    "InferenceSummary",
    "OptimizationProfile",
    "OutlierAlgorithm",
    "OutlierExtractionSummary",
    "PurgeSummary",
    "RefinementDirectoryStatus",
    "RefinementStatusSummary",
    "TrackingMethod",
    "TrainingDatasetSummary",
    "TrainingSummary",
    "build_conditional_top_down_conditions",
    "build_superanimal_weight_init",
    "create_training_dataset",
    "extract_frames_kmeans",
    "extract_outlier_frames_parallel",
    "get_available_augmenters",
    "get_available_object_detectors",
    "get_available_pose_models",
    "get_available_super_animals",
    "purge_labeled_data",
    "resolve_inference_profile",
    "resolve_optimization_profile",
    "run_inference",
    "summarize_refinement_status",
    "train_model",
]
