"""Provides optimized multi-device DeepLabCut video inference over one or many videos."""

from .pipeline import InferenceSummary, run_inference, resolve_project_videos, detect_fixed_input_size
from ..hardware import Toggle, AmpMode, DeviceType
from ..reporting import PipelineFailedError, PipelineInterruptedError
from .optimization import InferenceProfile, resolve_inference_profile

__all__ = [
    "AmpMode",
    "DeviceType",
    "InferenceProfile",
    "InferenceSummary",
    "PipelineFailedError",
    "PipelineInterruptedError",
    "Toggle",
    "detect_fixed_input_size",
    "resolve_inference_profile",
    "resolve_project_videos",
    "run_inference",
]
