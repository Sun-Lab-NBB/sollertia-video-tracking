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
    ConversionSummary,
    run_inference,
    resolve_inference_profile,
    convert_predictions_to_feather,
)
from .frame_extraction import FrameExtractionSummary, extract_frames_kmeans  # noqa: E402 - after thread-limit setup

__all__ = [
    "ConversionSummary",
    "FrameExtractionSummary",
    "InferenceProfile",
    "InferenceSummary",
    "OptimizationProfile",
    "TrainingDatasetSummary",
    "TrainingSummary",
    "build_conditional_top_down_conditions",
    "build_superanimal_weight_init",
    "convert_predictions_to_feather",
    "create_training_dataset",
    "extract_frames_kmeans",
    "get_available_augmenters",
    "get_available_object_detectors",
    "get_available_pose_models",
    "get_available_super_animals",
    "resolve_inference_profile",
    "resolve_optimization_profile",
    "run_inference",
    "train_model",
]
