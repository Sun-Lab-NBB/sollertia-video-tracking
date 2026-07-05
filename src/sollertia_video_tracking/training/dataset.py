"""Provides a wrapper over DeepLabCut training-dataset creation at parity with the GUI's create-dataset tab."""

import re
import sys
from typing import Any, TextIO
from pathlib import Path
import contextlib
from dataclasses import dataclass
from collections.abc import Iterator

import deeplabcut.compat as dlc_compat
from deeplabcut.modelzoo import build_weight_init
from deeplabcut.core.engine import Engine
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.generate_training_dataset import (
    create_training_dataset as dlc_create_training_dataset,
    get_existing_shuffle_indices,
    create_training_dataset_from_existing_split as dlc_create_training_dataset_from_existing_split,
)
from deeplabcut.pose_estimation_pytorch.config.utils import available_models, available_detectors

_TOP_DOWN_PREFIX: str = "top_down_"
"""The net_type prefix that marks a top-down pose model, stripped to recover the SuperAnimal model name."""

_SUPER_ANIMAL_DATASETS: tuple[str, ...] = (
    "superanimal_bird",
    "superanimal_quadruped",
    "superanimal_topviewmouse",
)
"""The SuperAnimal datasets that support transfer learning and fine-tuning, matching the DeepLabCut GUI's options."""

_CONDITION_PREDICTION_SUFFIXES: tuple[str, ...] = (".h5", ".json")
"""The conditional-top-down conditioning-file suffixes passed through unchanged as a predictions path."""

_CONDITION_SNAPSHOT_SUFFIX: str = ".pt"
"""The conditional-top-down conditioning-file suffix that identifies a snapshot, converted to a (shuffle, name) pair."""

_UNANNOTATED_VIDEO_NOTICE: str = "not found (perhaps not annotated)"
"""The substring DeepLabCut prints once per registered video that lacks an annotation file, filtered from the
training-dataset creation output because projects routinely register many more videos than any one shuffle labels."""


@dataclass(frozen=True, slots=True)
class TrainingDatasetSummary:
    """Captures the outcome of creating a training-dataset shuffle for reporting to the caller."""

    config: Path
    """The path of the DeepLabCut project configuration file the shuffle was created for."""
    shuffle: int
    """The shuffle index that was created."""
    net_type: str | None
    """The pose-model architecture the shuffle was created for, or None when the project default was used."""
    detector_type: str | None
    """The object detector the shuffle was created for, or None for bottom-up and conditional top-down models."""
    augmenter_type: str | None
    """The data-augmentation pipeline the shuffle was created with, or None when the engine default was used."""
    weight_init: str
    """A description of the weight initialization used (``imagenet`` or ``<super_animal> (transfer|fine-tune)``)."""
    from_shuffle: int | None
    """The shuffle whose train/test split was reused, or None when a fresh split was drawn."""

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the created shuffle for the CLI.

        Returns:
            A compact description of the model, weights, and split source of the created shuffle.
        """
        model = self.net_type or "project default"
        detector = f" + {self.detector_type}" if self.detector_type else ""
        split = f" (split from shuffle {self.from_shuffle})" if self.from_shuffle is not None else ""
        return f"created shuffle {self.shuffle} for {model}{detector} [{self.weight_init}]{split} from {self.config}"


def get_available_pose_models() -> tuple[str, ...]:
    """Returns the sorted catalog of pose-model architectures the PyTorch engine supports.

    Returns:
        Every ``net_type`` value the installed DeepLabCut PyTorch engine can train, sorted alphabetically.
    """
    return tuple(sorted(available_models()))


def get_available_object_detectors() -> tuple[str, ...]:
    """Returns the sorted catalog of object detectors the PyTorch engine supports for top-down models.

    Returns:
        Every ``detector_type`` value the installed DeepLabCut PyTorch engine can train, sorted alphabetically.
    """
    return tuple(sorted(available_detectors()))


def get_available_augmenters() -> tuple[str, ...]:
    """Returns the data-augmentation pipelines the PyTorch engine supports.

    Returns:
        Every ``augmenter_type`` value valid for the PyTorch engine (currently only ``albumentations``).
    """
    return tuple(dlc_compat.get_available_aug_methods(Engine.PYTORCH))


def get_available_super_animals() -> tuple[str, ...]:
    """Returns the SuperAnimal datasets available for transfer learning and fine-tuning.

    Returns:
        Every SuperAnimal dataset name that can initialize a project's weights, matching the GUI's options.
    """
    return _SUPER_ANIMAL_DATASETS


def build_superanimal_weight_init(
    config: str | Path,
    super_animal: str,
    net_type: str,
    detector_type: str | None,
    *,
    fine_tune: bool = False,
    memory_replay: bool = False,
    customized_pose_checkpoint: str | Path | None = None,
    customized_detector_checkpoint: str | Path | None = None,
) -> WeightInitialization:
    """Builds a SuperAnimal weight initialization for transfer learning or fine-tuning, mirroring the GUI selector.

    Args:
        config: The path of the DeepLabCut project configuration file.
        super_animal: The SuperAnimal dataset to initialize from, one of ``get_available_super_animals``.
        net_type: The project's pose-model architecture; a ``top_down_`` prefix is stripped to name the SuperAnimal
            pose model.
        detector_type: The project's detector architecture for top-down models, or None.
        fine_tune: Whether to fine-tune, loading the SuperAnimal decoder head (requires a conversion table), rather
            than transfer learning with a fresh head.
        memory_replay: Whether to enable memory replay, which is only valid when fine-tuning.
        customized_pose_checkpoint: A custom SuperAnimal pose checkpoint to use instead of the downloaded one.
        customized_detector_checkpoint: A custom SuperAnimal detector checkpoint to use instead of the downloaded one.

    Returns:
        The weight initialization to pass to ``create_training_dataset``.
    """
    model_name = net_type.removeprefix(_TOP_DOWN_PREFIX)
    return build_weight_init(
        cfg=str(config),
        super_animal=super_animal,
        model_name=model_name,
        detector_name=detector_type,
        with_decoder=fine_tune,
        memory_replay=memory_replay,
        customized_pose_checkpoint=customized_pose_checkpoint,
        customized_detector_checkpoint=customized_detector_checkpoint,
    )


def build_conditional_top_down_conditions(conditions_path: str | Path) -> Path | tuple[int, str]:
    """Converts a conditioning-file path into the format ``create_training_dataset`` expects for CTD models.

    Args:
        conditions_path: The path to a predictions file (``.h5``/``.json``) or a conditioning snapshot (``.pt``).

    Returns:
        The predictions path unchanged, or a ``(shuffle, snapshot_name)`` pair parsed from the snapshot path.

    Raises:
        ValueError: When the file type is unsupported or a shuffle index cannot be parsed from a snapshot path.
    """
    path = Path(conditions_path)
    suffix = path.suffix.lower()
    if suffix in _CONDITION_PREDICTION_SUFFIXES:
        return path
    if suffix == _CONDITION_SNAPSHOT_SUFFIX:
        match = re.search(r"shuffle(\d+)", str(path))
        if match is None:
            message = (
                f"Unable to build conditional top-down conditions using the snapshot path. Expected the path to "
                f"contain a 'shuffleN' segment, but got '{path}'."
            )
            raise ValueError(message)
        return int(match.group(1)), path.name
    message = (
        f"Unable to build conditional top-down conditions using '{path}'. Expected a .h5 or .json predictions file, "
        f"or a .pt snapshot, but got a '{path.suffix}' file."
    )
    raise ValueError(message)


def create_training_dataset(
    config: str | Path,
    *,
    shuffle: int = 1,
    net_type: str | None = None,
    detector_type: str | None = None,
    augmenter_type: str | None = None,
    weight_init: WeightInitialization | None = None,
    ctd_conditions: Any = None,
    from_shuffle: int | None = None,
    from_training_set_index: int = 0,
    overwrite: bool = False,
) -> TrainingDatasetSummary:
    """Creates a training-dataset shuffle for a project, at parity with the DeepLabCut GUI's create-dataset tab.

    Wraps DeepLabCut's training-dataset creation for the PyTorch engine, which bakes the model architecture, weight
    initialization, and train/test split into the shuffle (training is run afterward with ``slvt train``).
    Multi-animal projects are handled automatically. When ``from_shuffle`` is given, the new shuffle reuses that
    shuffle's train/test split instead of drawing a fresh one.

    Args:
        config: The path of the DeepLabCut project configuration file.
        shuffle: The shuffle index to create.
        net_type: The pose-model architecture, one of ``get_available_pose_models``. None uses the project default. A
            ``top_down_`` prefix creates a top-down model that also trains a detector.
        detector_type: The object detector for top-down models, one of ``get_available_object_detectors``.
        augmenter_type: The augmentation pipeline, one of ``get_available_augmenters``. None uses the engine default.
        weight_init: A SuperAnimal weight initialization from ``build_superanimal_weight_init``, or None for ImageNet
            transfer learning.
        ctd_conditions: The conditioning source for conditional top-down models, from
            ``build_conditional_top_down_conditions``, or None.
        from_shuffle: The existing shuffle whose train/test split to reuse, or None to draw a fresh split.
        from_training_set_index: The training-set fraction index of ``from_shuffle`` when reusing a split.
        overwrite: Whether to overwrite the shuffle if the index already exists.

    Returns:
        A summary of the created shuffle.

    Raises:
        ValueError: When ``net_type``, ``detector_type``, or ``augmenter_type`` is not in the installed engine's
            catalog.
    """
    if net_type is not None and net_type not in available_models():
        message = (
            f"Unable to create the training dataset using the requested net_type. Expected one of "
            f"{', '.join(get_available_pose_models())}, but got '{net_type}'."
        )
        raise ValueError(message)
    if detector_type is not None and detector_type not in available_detectors():
        message = (
            f"Unable to create the training dataset using the requested detector_type. Expected one of "
            f"{', '.join(get_available_object_detectors())}, but got '{detector_type}'."
        )
        raise ValueError(message)
    if augmenter_type is not None and augmenter_type not in get_available_augmenters():
        message = (
            f"Unable to create the training dataset using the requested augmenter_type. Expected one of "
            f"{', '.join(get_available_augmenters())}, but got '{augmenter_type}'."
        )
        raise ValueError(message)

    user_feedback = not overwrite
    with _suppress_unannotated_video_notices():
        if from_shuffle is not None:
            dlc_create_training_dataset_from_existing_split(
                config=str(config),
                from_shuffle=from_shuffle,
                from_trainsetindex=from_training_set_index,
                shuffles=[shuffle],
                net_type=net_type,
                detector_type=detector_type,
                augmenter_type=augmenter_type,
                userfeedback=user_feedback,
                weight_init=weight_init,
                ctd_conditions=ctd_conditions,
            )
        else:
            dlc_create_training_dataset(
                config=str(config),
                Shuffles=[shuffle],
                net_type=net_type,
                detector_type=detector_type,
                augmenter_type=augmenter_type,
                userfeedback=user_feedback,
                weight_init=weight_init,
                ctd_conditions=ctd_conditions,
            )

    # DeepLabCut silently returns without creating the shuffle when it finds no labeled data to build it from (for
    # example, every labeled frame was annotated by a scorer other than the one named in the project configuration, or
    # no frames are annotated yet). Confirm the shuffle now exists so a success summary is never reported for a no-op.
    if shuffle not in get_existing_shuffle_indices(str(config)):
        message = (
            f"Unable to create the training dataset for shuffle {shuffle}. DeepLabCut created no shuffle, which "
            f"usually means it found no labeled data to build it from (for example, every labeled frame was annotated "
            f"by a scorer other than the one named in the project configuration, or no frames are annotated yet). "
            f"Review the DeepLabCut output above for details."
        )
        raise ValueError(message)

    if weight_init is None:
        weight_init_description = "imagenet"
    else:
        mode = "fine-tune" if weight_init.with_decoder else "transfer"
        weight_init_description = f"{weight_init.dataset} ({mode})"

    return TrainingDatasetSummary(
        config=Path(config),
        shuffle=shuffle,
        net_type=net_type,
        detector_type=detector_type,
        augmenter_type=augmenter_type,
        weight_init=weight_init_description,
        from_shuffle=from_shuffle,
    )


class _UnannotatedNoticeFilter:
    """Forwards written text to a target stream, dropping whole lines that contain a marker substring.

    DeepLabCut's training-set collation prints one notice per registered video that has no annotation file. Projects
    commonly register many more videos than any single shuffle labels, so these notices flood the console without
    conveying anything actionable. This wrapper buffers writes until a line break, then forwards each completed line to
    the target stream unless that line contains the marker.
    """

    def __init__(self, target: TextIO, marker: str) -> None:
        """Initializes the filter with the stream to forward to and the marker that selects lines to drop.

        Args:
            target: The stream that surviving lines are written to.
            marker: The substring whose presence in a completed line drops that line.
        """
        self._target = target
        self._marker = marker
        self._pending = ""

    def write(self, text: str) -> int:
        """Buffers the text and forwards every newly completed line that does not contain the marker.

        Args:
            text: The text written to the redirected stream.

        Returns:
            The number of characters accepted, honoring the standard stream write contract.
        """
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            if self._marker not in line:
                self._target.write(f"{line}\n")
        return len(text)

    def flush(self) -> None:
        """Flushes the target stream, leaving any incomplete trailing line buffered until it is completed."""
        self._target.flush()

    def drain(self) -> None:
        """Forwards any buffered trailing text that never received a line break, unless it contains the marker."""
        if self._pending and self._marker not in self._pending:
            self._target.write(self._pending)
        self._pending = ""

    def __getattr__(self, name: str) -> Any:
        """Delegates stream attributes not defined on the filter to the underlying target stream.

        Args:
            name: The attribute requested on the filter.

        Returns:
            The corresponding attribute of the target stream.
        """
        return getattr(self._target, name)


@contextlib.contextmanager
def _suppress_unannotated_video_notices() -> Iterator[None]:
    """Filters DeepLabCut's per-video "not annotated" notices from standard output within the context.

    DeepLabCut prints one notice for every registered video that lacks an annotation file while collating the training
    set. These notices are expected whenever a project registers more videos than are labeled and are not actionable,
    so they are dropped while every other line reaches the console unchanged.

    Yields:
        None, for the duration of the filtering.
    """
    stream_filter = _UnannotatedNoticeFilter(target=sys.stdout, marker=_UNANNOTATED_VIDEO_NOTICE)
    try:
        with contextlib.redirect_stdout(stream_filter):
            yield
    finally:
        stream_filter.drain()
