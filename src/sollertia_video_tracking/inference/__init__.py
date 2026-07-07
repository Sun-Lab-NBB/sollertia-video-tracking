"""Provides optimized multi-device DeepLabCut video inference over one or many videos."""

from .pipeline import InferenceSummary, run_inference, resolve_project_videos, detect_fixed_input_size
from ..hardware import Toggle, AmpMode
from .conversion import ConversionSummary, convert_predictions_to_feather
from .optimization import (
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
    "detect_fixed_input_size",
    "resolve_inference_profile",
    "resolve_project_videos",
    "run_inference",
]
