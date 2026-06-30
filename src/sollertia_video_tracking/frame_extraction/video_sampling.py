"""Provides the budgeted random video-subset selection that grows extraction coverage toward a total-frame target."""

import math
from random import Random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoSamplingPlan:
    """Describes which videos a budgeted extraction pass samples and how the pass relates to the frame budget.

    Notes:
        The selection only draws from videos that have no extracted frames yet, so repeated passes broaden dataset
        coverage by adding new videos rather than re-clustering ones that are already done.
    """

    selected: tuple[str, ...]
    """The videos chosen for extraction in this pass, drawn from the not-yet-extracted candidates."""
    existing_frames: int
    """The number of frames already extracted across the candidate videos before this pass."""
    target_frames: int
    """The requested total number of frames the project should hold after extraction."""
    projected_frames: int
    """The number of frames the project is expected to hold once the selected videos are extracted."""
    no_growth: bool
    """Whether the existing frames already meet the target, so the pass would extract nothing."""
    target_unreachable: bool
    """Whether too few un-extracted videos remain to reach the target, so the pass falls short of it."""


def plan_video_sampling(
    videos: list[str],
    extracted_frame_counts: dict[str, int],
    frames_per_video: int,
    total_frames: int,
    seed: int | None,
) -> VideoSamplingPlan:
    """Selects a random subset of not-yet-extracted videos that grows the project toward a total-frame budget.

    Sums the frames already extracted across the candidate videos and samples just enough additional videos, each
    contributing ``frames_per_video`` frames, to reach ``total_frames``. Only videos without extracted frames are
    eligible, so repeated passes broaden dataset coverage instead of re-clustering videos that are already done. The
    realized total rounds up to a whole video, so it may exceed the target by less than one video's worth of frames.

    Args:
        videos: The ordered candidate video paths the pass may sample from.
        extracted_frame_counts: The number of frames already extracted for each candidate video, keyed by path.
        frames_per_video: The number of frames each newly sampled video contributes.
        total_frames: The total number of frames the project should hold after extraction.
        seed: The seed for the random subset draw, or None to draw nondeterministically.

    Returns:
        A VideoSamplingPlan naming the selected videos and reporting the existing, target, and projected frame counts
        alongside the no-growth and unreachable-target flags.
    """
    existing_frames = sum(extracted_frame_counts.get(video, 0) for video in videos)
    un_extracted = [video for video in videos if extracted_frame_counts.get(video, 0) == 0]
    remaining = total_frames - existing_frames

    if remaining <= 0:
        return VideoSamplingPlan(
            selected=(),
            existing_frames=existing_frames,
            target_frames=total_frames,
            projected_frames=existing_frames,
            no_growth=True,
            target_unreachable=False,
        )

    needed = math.ceil(remaining / frames_per_video)
    target_unreachable = needed > len(un_extracted)
    needed = min(needed, len(un_extracted))

    generator = Random(seed)  # noqa: S311 -- video sampling is not security-sensitive.
    selected = tuple(generator.sample(un_extracted, needed))
    projected_frames = existing_frames + needed * frames_per_video
    return VideoSamplingPlan(
        selected=selected,
        existing_frames=existing_frames,
        target_frames=total_frames,
        projected_frames=projected_frames,
        no_growth=False,
        target_unreachable=target_unreachable,
    )
