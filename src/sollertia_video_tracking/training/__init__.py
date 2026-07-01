"""Provides hardware-optimized DeepLabCut model training, including device profiling and mixed-precision DDP runners."""

from .dataset import (
    TrainingDatasetSummary,
    create_training_dataset,
    get_available_augmenters,
    get_available_pose_models,
    get_available_super_animals,
    build_superanimal_weight_init,
    get_available_object_detectors,
    build_conditional_top_down_conditions,
)
from .pipeline import TrainingSummary, train_model
from .optimization import (
    Toggle,
    AmpMode,
    MultiGpuStrategy,
    OptimizationProfile,
    apply_runtime_optimizations,
    resolve_optimization_profile,
)

__all__ = [
    "AmpMode",
    "MultiGpuStrategy",
    "OptimizationProfile",
    "Toggle",
    "TrainingDatasetSummary",
    "TrainingSummary",
    "apply_runtime_optimizations",
    "build_conditional_top_down_conditions",
    "build_superanimal_weight_init",
    "create_training_dataset",
    "get_available_augmenters",
    "get_available_object_detectors",
    "get_available_pose_models",
    "get_available_super_animals",
    "resolve_optimization_profile",
    "train_model",
]
