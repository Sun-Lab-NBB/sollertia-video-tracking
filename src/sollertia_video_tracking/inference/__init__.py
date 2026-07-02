"""Provides optimized multi-device DeepLabCut video inference over one or many videos."""

from .pipeline import InferenceSummary, run_inference
from .conversion import ConversionSummary, convert_predictions_to_feather
from .optimization import (
    Toggle,
    AmpMode,
    InferenceProfile,
    resolve_inference_profile,
    apply_runtime_optimizations,
)

__all__ = [
    "AmpMode",
    "ConversionSummary",
    "InferenceProfile",
    "InferenceSummary",
    "Toggle",
    "apply_runtime_optimizations",
    "convert_predictions_to_feather",
    "resolve_inference_profile",
    "run_inference",
]
