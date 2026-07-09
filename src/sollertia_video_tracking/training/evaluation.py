"""Provides reproducible evaluation of a trained DeepLabCut shuffle as a clean, self-describing polars feather."""

from typing import Any
import logging
from pathlib import Path
from datetime import UTC, datetime
from dataclasses import dataclass

import numpy as np
import polars as pl
import deeplabcut
from ruamel.yaml import YAML
from numpy.typing import NDArray
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.core.metrics.api import prepare_evaluation_data
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch.data import DLCLoader, PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.core.metrics.distance_metrics import match_predictions_for_rmse
from deeplabcut.pose_estimation_pytorch.apis.utils import get_model_snapshots, get_inference_runners
from deeplabcut.pose_estimation_pytorch.apis.evaluation import evaluate

_logger = logging.getLogger(__name__)
"""The module logger; its records propagate to the root handlers configured for the training run."""

_SPLITS: tuple[str, ...] = ("train", "test")
"""The labeled-data partitions scored during evaluation, in report order."""

_LABELED_DATA_DIR: str = "labeled-data"
"""The project subdirectory anchoring a labeled frame's project-relative path."""

_WORST_KEYPOINT_COUNT: int = 5
"""The number of highest-error bodyparts reported in the provenance summary."""

_FEATHER_SCHEMA: dict[str, Any] = {
    "snapshot": pl.Utf8,
    "split": pl.Utf8,
    "video": pl.Utf8,
    "image": pl.Utf8,
    "individual": pl.Utf8,
    "bodypart": pl.Utf8,
    "pred_x": pl.Float32,
    "pred_y": pl.Float32,
    "pred_likelihood": pl.Float32,
    "gt_x": pl.Float32,
    "gt_y": pl.Float32,
    "error_px": pl.Float32,
    "above_pcutoff": pl.Boolean,
    "matched": pl.Boolean,
    "oks": pl.Float32,
}
"""The column names and polars dtypes of the tidy evaluation feather, in file order.

Notes:
    Each row is one predicted keypoint. ``individual`` is the ground-truth individual name for single-animal projects
    and for matched multi-animal predictions; an unmatched multi-animal prediction (a false positive, ``matched`` is
    False) is labeled ``instance_<order>`` by its per-image match order and carries no stable identity across images.
    ``error_px`` is the Euclidean pixel distance to the matched label, and is NaN when the keypoint is unmatched or its
    label is occluded. ``oks`` is NaN for an unmatched prediction and otherwise holds the match's OKS; for matched
    single-animal and unique-bodypart rows, which DeepLabCut scores by direct correspondence rather than OKS, that
    value is the matcher's default rather than a computed OKS.
"""


@dataclass(frozen=True, slots=True)
class SplitMetrics:
    """Captures DeepLabCut's canonical error metrics for one labeled-data partition.

    Notes:
        The values are taken verbatim from DeepLabCut's scorer so they agree with ``deeplabcut.evaluate_network``.
        RMSE values are in pixels; ``map`` and ``mar`` are on the DeepLabCut 0-100 scale. ``rmse_pcutoff`` counts only
        keypoints predicted above the confidence cutoff.
    """

    images: int
    """The number of labeled images scored in this partition."""
    rmse_px: float
    """The mean per-keypoint Euclidean error over every valid keypoint, in pixels."""
    rmse_pcutoff_px: float
    """The mean per-keypoint error over keypoints predicted above the confidence cutoff, in pixels."""
    map: float
    """The OKS-based mean average precision on the DeepLabCut 0-100 scale, computed for single- and multi-animal."""
    mar: float
    """The OKS-based mean average recall on the DeepLabCut 0-100 scale, computed for single- and multi-animal."""
    unmatched_images: int
    """The number of images in this partition for which no prediction matched a ground-truth individual."""


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Captures the outcome of evaluating one trained snapshot for reporting to the caller.

    Notes:
        The summary is built after the feather is written. It reports what was evaluated and the headline test-set
        accuracy; the per-frame, per-keypoint detail lives in the feather and the full metric set lives in the
        provenance sidecar.
    """

    config: Path
    """The DeepLabCut project configuration file the evaluated shuffle belongs to."""
    shuffle: int
    """The shuffle index that was evaluated."""
    snapshot: str
    """The name of the pose snapshot that was scored (for example ``snapshot-190``)."""
    feather_path: Path
    """The tidy evaluation feather that was written."""
    provenance_path: Path
    """The path of the YAML provenance-and-summary sidecar; it is written beside the feather only when
    ``write_provenance`` is True, otherwise this path names a file that was not created."""
    pcutoff: float
    """The confidence cutoff applied when computing the cutoff-filtered metrics and the ``above_pcutoff`` column."""
    train: SplitMetrics
    """The canonical metrics for the training frames."""
    test: SplitMetrics
    """The canonical metrics for the held-out test frames."""

    @property
    def generalization_gap_px(self) -> float:
        """Returns the test-minus-train RMSE gap in pixels; a large positive gap indicates overfitting."""
        return self.test.rmse_px - self.train.rmse_px

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the evaluation for the CLI.

        Returns:
            A compact description of the scored snapshot and its train and test error.
        """
        return (
            f"evaluated {self.snapshot}: test RMSE {self.test.rmse_px:.2f}px "
            f"(train {self.train.rmse_px:.2f}px, gap {self.generalization_gap_px:+.2f}px) -> {self.feather_path.name}"
        )


def evaluate_trained_model(
    config: str | Path,
    *,
    shuffle: int = 1,
    training_set_index: int = 0,
    model_prefix: str = "",
    snapshot_index: int | str = "best",
    detector_snapshot_index: int = -1,
    confidence_cutoff: float | None = None,
    batch_size: int = 16,
    device: str | None = None,
    write_provenance: bool = True,
) -> EvaluationSummary:
    """Scores a trained snapshot against the labeled frames and writes a tidy evaluation feather and provenance.

    The snapshot's predictions on the train and test frames are compared to the human labels using DeepLabCut's own
    scorer and matching, so the headline metrics agree with ``deeplabcut.evaluate_network``. The per-frame,
    per-keypoint comparison is written as a polars feather and the metric summary and run provenance as a YAML sidecar,
    both into the shuffle's evaluation-results directory.

    Notes:
        Evaluation is a single float32 forward pass over the small labeled train and test set on one device. Mixed
        precision is deliberately not used, because the canonical error metrics are defined at float32 and the labeled
        set is too small for throughput to matter.

    Args:
        config: The path of the DeepLabCut project configuration file.
        shuffle: The shuffle index to evaluate.
        training_set_index: The training-set fraction index.
        model_prefix: The model subdirectory prefix, matching the trained shuffle.
        snapshot_index: The snapshot to score: ``"best"`` (falling back to the last snapshot when no best snapshot was
            saved), an integer index, or ``-1`` for the last snapshot.
        detector_snapshot_index: The detector snapshot index for top-down models; ignored for bottom-up models and
            when no detector was trained, in which case ground-truth bounding boxes are used.
        confidence_cutoff: The confidence cutoff for the cutoff-filtered metrics and the ``above_pcutoff`` column, or
            None to fall back to the project configuration's ``pcutoff`` (0.6 when unset), matching
            ``deeplabcut.evaluate_network``.
        batch_size: The number of frames scored per forward pass; larger batches use the device more fully. It is
            reduced to one automatically when the labeled frames span more than one resolution, which DeepLabCut cannot
            stack into a single batch.
        device: The device to evaluate on (for example ``"cuda:0"`` or ``"cpu"``), or None to resolve it from the
            model configuration.
        write_provenance: Whether to write the ``<snapshot>_evaluation.yaml`` sidecar beside the feather.

    Returns:
        A summary describing what was evaluated and the resulting train and test accuracy.

    Raises:
        ValueError: When the requested snapshot cannot be resolved.
        OSError: When the provenance sidecar cannot be written; the feather written earlier is removed first so a
            feather is never left without its provenance sidecar.
    """
    config = Path(config)
    loader = DLCLoader(config=config, shuffle=shuffle, trainset_index=training_set_index, modelprefix=model_prefix)
    parameters = loader.get_dataset_parameters()
    single_animal = parameters.max_num_animals == 1
    cutoff = float(loader.project_cfg.get("pcutoff", 0.6)) if confidence_cutoff is None else float(confidence_cutoff)
    batch_size = _resolve_evaluation_batch_size(loader, batch_size)

    pose_snapshot = _resolve_snapshot(loader, snapshot_index, loader.pose_task)
    detector_snapshot = None
    if loader.pose_task == Task.TOP_DOWN and loader.model_cfg.get("detector") is not None:
        detector_snapshot = _resolve_snapshot(loader, detector_snapshot_index, Task.DETECT, required=False)

    pose_runner, detector_runner = get_inference_runners(
        model_config=loader.model_cfg,
        snapshot_path=pose_snapshot.path,
        max_individuals=parameters.max_num_animals,
        num_bodyparts=parameters.num_joints,
        num_unique_bodyparts=parameters.num_unique_bpts,
        with_identity=loader.model_cfg["metadata"]["with_identity"],
        batch_size=batch_size,
        detector_path=(detector_snapshot.path if detector_snapshot is not None else None),
        detector_batch_size=batch_size,
        device=device,
    )

    # The inference runners above are built for the network's own output space. A memory-replay model still predicts
    # the full SuperAnimal bodypart set. But evaluate() down-converts predictions (and the loader returns ground truth)
    # in the smaller project bodypart space. So the parameters used for scoring and row accumulation must be realigned
    # to the project bodyparts to keep the per-keypoint loop in range.
    parameters = _realign_memory_replay_parameters(loader, parameters)

    snapshot_name = pose_snapshot.path.stem
    columns: dict[str, list[Any]] = {name: [] for name in _FEATHER_SCHEMA}
    split_metrics: dict[str, SplitMetrics] = {}
    raw_metrics: dict[str, dict[str, Any]] = {}
    for split in _SPLITS:
        metrics, predictions = evaluate(
            pose_runner=pose_runner,
            loader=loader,
            mode=split,
            detector_runner=detector_runner,
            parameters=parameters,
            per_keypoint_evaluation=True,
            pcutoff=cutoff,
        )
        raw_metrics[split] = metrics
        image_paths = loader.image_filenames(split)
        ground_truth = loader.ground_truth_keypoints(split)
        unmatched = _accumulate_split_rows(
            columns,
            snapshot_name=snapshot_name,
            split=split,
            image_paths=image_paths,
            predictions=predictions,
            ground_truth=ground_truth,
            bodyparts=parameters.bodyparts,
            individuals=parameters.individuals,
            single_animal=single_animal,
            confidence_cutoff=cutoff,
        )
        if parameters.num_unique_bpts > 0:
            _accumulate_split_rows(
                columns,
                snapshot_name=snapshot_name,
                split=split,
                image_paths=image_paths,
                predictions=predictions,
                ground_truth=loader.ground_truth_keypoints(split, unique_bodypart=True),
                bodyparts=parameters.unique_bpts,
                individuals=["unique"],
                single_animal=True,
                confidence_cutoff=cutoff,
                prediction_key="unique_bodyparts",
            )
        split_metrics[split] = SplitMetrics(
            images=len(ground_truth),
            rmse_px=float(metrics["rmse"]),
            rmse_pcutoff_px=float(metrics["rmse_pcutoff"]),
            map=float(metrics["mAP"]),
            mar=float(metrics["mAR"]),
            unmatched_images=unmatched,
        )

    evaluation_directory = Path(loader.evaluation_folder)
    evaluation_directory.mkdir(parents=True, exist_ok=True)
    feather_path = evaluation_directory / f"{snapshot_name}_evaluation.feather"
    frame = pl.DataFrame(columns, schema=_FEATHER_SCHEMA)
    frame.write_ipc(file=feather_path, compression="uncompressed")

    provenance_path = evaluation_directory / f"{snapshot_name}_evaluation.yaml"
    if write_provenance:
        try:
            _write_provenance(
                provenance_path,
                config=config,
                loader=loader,
                parameters=parameters,
                snapshot=pose_snapshot,
                detector_snapshot=detector_snapshot,
                device=device,
                batch_size=batch_size,
                confidence_cutoff=cutoff,
                single_animal=single_animal,
                split_metrics=split_metrics,
                feather_path=feather_path,
                worst_keypoints=_rank_worst_keypoints(raw_metrics["test"], parameters.bodyparts),
            )
        except OSError:
            # Keep the two-file convention consistent: never leave a feather without its provenance sidecar.
            feather_path.unlink(missing_ok=True)
            raise

    return EvaluationSummary(
        config=config,
        shuffle=shuffle,
        snapshot=snapshot_name,
        feather_path=feather_path,
        provenance_path=provenance_path,
        pcutoff=cutoff,
        train=split_metrics["train"],
        test=split_metrics["test"],
    )


def _resolve_evaluation_batch_size(loader: DLCLoader, requested: int) -> int:
    """Reduces the evaluation batch size to one unless every labeled frame shares a single native resolution.

    DeepLabCut's inference runner stacks each forward-pass batch into one tensor, and its evaluation transforms do not
    resize every frame to a single common size. The HRNet and DEKR backbones pad each frame up to a multiple of 32, and
    a detector consumes frames at their native resolution. Labeled frames extracted at more than one native resolution
    therefore cannot share a batch, so this returns one whenever the frames differ in size and the requested size only
    when they are uniform. Comparing native resolutions is a safe over-approximation: frames of one native resolution
    always stack to one size, while distinct resolutions can produce distinct sizes. A top-down pose stage that crops
    each detection to a fixed size would in fact batch regardless, so treating its frames as unbatchable only costs
    speed and does not affect correctness. The frame dimensions are read from the loader's already-parsed annotations
    (populated from image headers), so no pixels are decoded.

    Args:
        loader: The loader for the evaluated shuffle, holding the labeled train and test frames.
        requested: The batch size requested for the forward pass.

    Returns:
        The requested batch size when every labeled frame shares one native resolution, or one otherwise.
    """
    if requested <= 1:
        return 1
    resolutions: set[tuple[int, int]] = set()
    for split in _SPLITS:
        for image in loader.load_data(split)["images"]:
            height, width = image.get("height"), image.get("width")
            if height is None or width is None:
                return 1
            resolutions.add((int(height), int(width)))
            if len(resolutions) > 1:
                _logger.info(
                    "The labeled frames span multiple resolutions, which DeepLabCut cannot stack into one batch; "
                    "scoring one frame at a time (batch size 1)."
                )
                return 1
    return requested


def _resolve_snapshot(loader: DLCLoader, index: int | str, task: Task, *, required: bool = True) -> Any:
    """Resolves a snapshot to score, falling back to the last snapshot when a best snapshot is requested but absent.

    Args:
        loader: The loader holding the model directory the snapshots live in.
        index: The requested snapshot index (``"best"``, an integer, or ``-1``).
        task: The task whose snapshots to search (pose or detector).
        required: Whether to raise when no snapshot is found; when False, returns None instead.

    Returns:
        The resolved DeepLabCut snapshot, or None when none is found and ``required`` is False.

    Raises:
        ValueError: When no snapshot can be resolved and ``required`` is True.
    """
    try:
        snapshots = get_model_snapshots(index=index, model_folder=loader.model_folder, task=task)
    except (ValueError, IndexError):
        if index == "best":
            try:
                snapshots = get_model_snapshots(index=-1, model_folder=loader.model_folder, task=task)
            except (ValueError, IndexError):
                snapshots = []
        else:
            snapshots = []
    if not snapshots:
        if required:
            message = f"Unable to resolve a '{task.value}' snapshot with index {index!r} in '{loader.model_folder}'."
            raise ValueError(message)
        return None
    return snapshots[0]


def _realign_memory_replay_parameters(loader: DLCLoader, parameters: PoseDatasetParameters) -> PoseDatasetParameters:
    """Realigns evaluation parameters to the project bodyparts for SuperAnimal memory-replay models.

    A memory-replay model keeps predicting the full SuperAnimal bodypart set, so ``get_dataset_parameters`` reports
    that full set, but DeepLabCut's ``evaluate`` down-converts predictions to the project's bodypart subset (and the
    loader returns ground truth in that subset). This rebuilds the parameters in the project bodypart space so the
    per-keypoint accumulation aligns with the scored predictions, mirroring DeepLabCut's ``evaluate_snapshot``. For all
    other models the parameters are returned unchanged.

    Args:
        loader: The loader for the evaluated shuffle.
        parameters: The dataset parameters reported for the model.

    Returns:
        The parameters to use for scoring, realigned to the project bodyparts when the shuffle used memory replay.
    """
    weight_init_config = loader.model_cfg["train_settings"].get("weight_init")
    if not weight_init_config:
        return parameters
    weight_init = WeightInitialization.from_dict(weight_init_config)
    if not weight_init.memory_replay:
        return parameters
    bodyparts = weight_init.bodyparts
    if bodyparts is None:
        bodyparts = auxiliaryfunctions.get_bodyparts(loader.project_cfg)
    return PoseDatasetParameters(
        bodyparts=bodyparts,
        unique_bpts=parameters.unique_bpts,
        individuals=parameters.individuals,
    )


def _accumulate_split_rows(
    columns: dict[str, list[Any]],
    *,
    snapshot_name: str,
    split: str,
    image_paths: list[str],
    predictions: dict[str, dict[str, NDArray[np.float32]]],
    ground_truth: dict[str, NDArray[np.float32]],
    bodyparts: list[str],
    individuals: list[str],
    single_animal: bool,
    confidence_cutoff: float,
    prediction_key: str = "bodyparts",
) -> int:
    """Appends one row per predicted keypoint for a partition, matching predictions to ground truth as DeepLabCut does.

    Each image's ground truth and predictions are prepared and matched with DeepLabCut's own routines, so the pixel
    errors written here equal the ones behind the canonical RMSE. Missing or occluded keypoints and unmatched
    predictions carry NaN errors. The same helper appends the main and the unique bodyparts, selected by
    ``prediction_key``.

    Args:
        columns: The mutable column accumulator to append rows to.
        snapshot_name: The name of the scored snapshot, written to the ``snapshot`` column.
        split: The partition name (``"train"`` or ``"test"``).
        image_paths: The labeled image paths for this partition, in a stable order.
        predictions: DeepLabCut's per-image predictions, keyed by image path.
        ground_truth: The per-image ground-truth keypoints, keyed by image path.
        bodyparts: The ordered bodypart names to write rows for.
        individuals: The ordered individual names; the first labels rows for single-animal projects.
        single_animal: Whether these keypoints are matched as a single individual.
        confidence_cutoff: The confidence cutoff used to fill the ``above_pcutoff`` column.
        prediction_key: The per-image prediction head to read (``"bodyparts"`` or ``"unique_bodyparts"``).

    Returns:
        The number of images in this partition for which no prediction matched a ground-truth individual.
    """
    unmatched_images = 0
    for image in image_paths:
        if image not in ground_truth or image not in predictions or prediction_key not in predictions[image]:
            continue
        gt = ground_truth[image]
        pred = predictions[image][prediction_key]
        prepared = prepare_evaluation_data(ground_truth={image: gt}, predictions={image: pred})
        prepared_gt = prepared[0][0]
        matches = match_predictions_for_rmse(data=prepared, single_animal=single_animal, oks_bbox_margin=0.0)
        surviving = None if single_animal else _surviving_individual_indices(gt)
        video, relative_image = _derive_relative_image_path(image)

        matched_any = False
        for instance, match in enumerate(matches):
            errors = match.pixel_errors()
            pose = match.pose
            matched = match.gt is not None
            matched_any = matched_any or matched
            if single_animal:
                individual = individuals[0]
            else:
                individual = _matched_individual(match, prepared_gt, surviving, individuals, instance)
            for keypoint, bodypart in enumerate(bodyparts):
                likelihood = float(pose[keypoint, 2])
                columns["snapshot"].append(snapshot_name)
                columns["split"].append(split)
                columns["video"].append(video)
                columns["image"].append(relative_image)
                columns["individual"].append(individual)
                columns["bodypart"].append(bodypart)
                columns["pred_x"].append(float(pose[keypoint, 0]))
                columns["pred_y"].append(float(pose[keypoint, 1]))
                columns["pred_likelihood"].append(likelihood)
                columns["gt_x"].append(float(match.gt[keypoint, 0]) if matched else float("nan"))
                columns["gt_y"].append(float(match.gt[keypoint, 1]) if matched else float("nan"))
                columns["error_px"].append(float(errors[keypoint]))
                columns["above_pcutoff"].append(likelihood >= confidence_cutoff)
                columns["matched"].append(matched)
                # OKS is only meaningful for a matched prediction; an unmatched prediction keeps the match object's
                # default 0.0, so store NaN instead to honor the "populated only for matched predictions" contract.
                columns["oks"].append(float(match.oks) if matched else float("nan"))
        if not matched_any:
            unmatched_images += 1

    return unmatched_images


def _surviving_individual_indices(ground_truth: NDArray[np.float32]) -> NDArray[np.intp]:
    """Returns the original individual indices that survive DeepLabCut's ground-truth preparation, in order.

    ``prepare_evaluation_data`` sets keypoints with visibility at or below zero to NaN and then drops individuals with
    no valid keypoint, so the prepared ground-truth rows no longer align with the original individual order. This
    replicates that filtering to map a prepared row back to its original individual, and hence to its name.

    Args:
        ground_truth: The per-image ground-truth array of shape (num_individuals, num_bodyparts, 3).

    Returns:
        The original indices of the individuals that survive preparation, in prepared-row order.
    """
    visible = ground_truth.astype(float)
    visible[visible[..., 2] <= 0] = np.nan
    kept = np.any(np.all(~np.isnan(visible), axis=-1), axis=-1)
    return np.nonzero(kept)[0]


def _matched_individual(
    match: Any,
    prepared_ground_truth: NDArray[np.float32],
    surviving: NDArray[np.intp] | None,
    individuals: list[str],
    instance: int,
) -> str:
    """Names the ground-truth individual a multi-animal prediction was matched to, or labels it by match order.

    The matcher stores the matched ground-truth pose but not its index, so the prepared row is located by value and
    mapped back to its original individual name. Unmatched predictions, which have no ground-truth identity, are
    labeled by their per-image match order instead.

    Args:
        match: The prediction-to-ground-truth match for one predicted individual.
        prepared_ground_truth: The prepared ground-truth array for the image, whose rows ``match.gt`` is drawn from.
        surviving: The original individual indices for the prepared rows, from ``_surviving_individual_indices``.
        individuals: The ordered individual names.
        instance: The per-image match order index, used for unmatched predictions.

    Returns:
        The matched individual's name, or ``instance_<order>`` when the prediction matched no ground-truth individual.
    """
    if match.gt is None or surviving is None:
        return f"instance_{instance}"
    for row in range(len(prepared_ground_truth)):
        if np.array_equal(a1=prepared_ground_truth[row], a2=match.gt, equal_nan=True):
            original = int(surviving[row])
            if 0 <= original < len(individuals):
                return individuals[original]
            break
    return f"instance_{instance}"


def _derive_relative_image_path(image: str) -> tuple[str, str]:
    """Reduces an absolute labeled-frame path to its video name and project-relative path.

    Args:
        image: The absolute path of a labeled image.

    Returns:
        A tuple of the containing video (directory) name and the project-relative image path, anchored at the
        ``labeled-data`` component when present and falling back to the parent directory and file name otherwise.
    """
    path = Path(image)
    parts = path.parts
    if _LABELED_DATA_DIR in parts:
        relative = parts[parts.index(_LABELED_DATA_DIR) :]
        _anchor, *tail = relative
        video = tail[0] if tail else ""
        return video, Path(*relative).as_posix()
    return path.parent.name, path.name


def _rank_worst_keypoints(metrics: dict[str, Any], bodyparts: list[str]) -> list[dict[str, Any]]:
    """Ranks the highest-error bodyparts from DeepLabCut's per-keypoint RMSE metrics.

    Args:
        metrics: A metrics dict from ``evaluate`` with ``per_keypoint_evaluation`` enabled, holding
            ``rmse_keypoint_<index>`` entries.
        bodyparts: The ordered bodypart names the ``rmse_keypoint_<index>`` entries index into.

    Returns:
        The highest-error bodyparts, each as a ``{"bodypart", "rmse_px"}`` mapping, sorted from worst to best and
        truncated to the reported count. Empty when no per-keypoint metrics are present.
    """
    ranked: list[dict[str, Any]] = []
    for index, bodypart in enumerate(bodyparts):
        value = metrics.get(f"rmse_keypoint_{index}")
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        ranked.append({"bodypart": bodypart, "rmse_px": round(float(value), 3)})
    ranked.sort(key=lambda entry: entry["rmse_px"], reverse=True)
    return ranked[:_WORST_KEYPOINT_COUNT]


def _write_provenance(
    provenance_path: Path,
    *,
    config: Path,
    loader: DLCLoader,
    parameters: Any,
    snapshot: Any,
    detector_snapshot: Any,
    device: str | None,
    batch_size: int,
    confidence_cutoff: float,
    single_animal: bool,
    split_metrics: dict[str, SplitMetrics],
    feather_path: Path,
    worst_keypoints: list[dict[str, Any]],
) -> None:
    """Writes the YAML provenance-and-summary sidecar describing the run and its headline metrics.

    Args:
        provenance_path: The path of the sidecar to write.
        config: The project configuration file.
        loader: The loader for the evaluated shuffle.
        parameters: The dataset parameters holding the bodypart and individual names.
        snapshot: The scored pose snapshot.
        detector_snapshot: The scored detector snapshot, or None.
        device: The evaluation device, or None when resolved from the configuration.
        batch_size: The forward-pass batch size used.
        confidence_cutoff: The confidence cutoff applied.
        single_animal: Whether the project tracks a single individual.
        split_metrics: The canonical metrics for each partition.
        feather_path: The feather written beside this sidecar.
        worst_keypoints: The highest-error bodyparts on the test frames, worst first.
    """
    record: dict[str, Any] = {
        "source_config": str(config),
        "shuffle": int(loader.shuffle),
        "train_fraction": float(loader.train_fraction),
        "engine": "pytorch",
        "snapshot": snapshot.path.stem,
        "snapshot_path": str(snapshot.path),
        "detector_snapshot": (detector_snapshot.path.stem if detector_snapshot is not None else None),
        "device": device,
        "batch_size": int(batch_size),
        "pcutoff": float(confidence_cutoff),
        "single_animal": bool(single_animal),
        "bodyparts": list(parameters.bodyparts),
        "individuals": list(parameters.individuals),
        "metrics": {
            split: {
                "images": metrics.images,
                "rmse_px": round(metrics.rmse_px, 3),
                "rmse_pcutoff_px": round(metrics.rmse_pcutoff_px, 3),
                "mAP": round(metrics.map, 2),
                "mAR": round(metrics.mar, 2),
                "unmatched_images": metrics.unmatched_images,
            }
            for split, metrics in split_metrics.items()
        },
        "generalization_gap_px": round(split_metrics["test"].rmse_px - split_metrics["train"].rmse_px, 3),
        "worst_keypoints": worst_keypoints,
        "feather": feather_path.name,
        "deeplabcut_version": deeplabcut.__version__,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    yaml = YAML()
    yaml.default_flow_style = False
    with provenance_path.open("w", encoding="utf-8") as stream:
        yaml.dump(record, stream)
