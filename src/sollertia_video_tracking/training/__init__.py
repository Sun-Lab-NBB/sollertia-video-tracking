"""Provides hardware-optimized DeepLabCut model training, including device profiling and mixed-precision DDP runners."""

from .dataset import (
    TrainingDatasetSummary,
    WeightInitializationMethod,
    create_training_dataset,
    get_available_augmenters,
    get_available_pose_models,
    get_available_super_animals,
    build_superanimal_weight_init,
    get_available_object_detectors,
    build_conditional_top_down_conditions,
)
from .pipeline import TrainingSummary, train_model, detect_fixed_input_size
from ..hardware import Toggle, AmpMode
from .evaluation import SplitMetrics, EvaluationSummary, evaluate_trained_model
from .optimization import (
    MultiGpuStrategy,
    OptimizationProfile,
    apply_runtime_optimizations,
    resolve_optimization_profile,
)

__all__ = [
    "AmpMode",
    "EvaluationSummary",
    "MultiGpuStrategy",
    "OptimizationProfile",
    "SplitMetrics",
    "Toggle",
    "TrainingDatasetSummary",
    "TrainingSummary",
    "WeightInitializationMethod",
    "apply_runtime_optimizations",
    "build_conditional_top_down_conditions",
    "build_superanimal_weight_init",
    "create_training_dataset",
    "detect_fixed_input_size",
    "evaluate_trained_model",
    "get_available_augmenters",
    "get_available_object_detectors",
    "get_available_pose_models",
    "get_available_super_animals",
    "resolve_optimization_profile",
    "train_model",
]
