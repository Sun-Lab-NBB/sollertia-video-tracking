"""Contains tests for the trained-shuffle evaluation feather writer.

The module hands off to DeepLabCut for loading, inference-runner construction, snapshot resolution, scoring, and
prediction/ground-truth matching. Every such handoff is monkeypatched at the exact module binding
(``evaluation.DLCLoader``, ``evaluation.evaluate``, ``evaluation.get_inference_runners``,
``evaluation.get_model_snapshots``) so no real DLC runtime, GPU, network, or project on disk is needed. The
per-keypoint accumulation, path derivation, individual matching, and provenance writing are driven with real numpy
arrays through DeepLabCut's genuine matching routines wherever they run headless, and with controlled fakes only where
greedy multi-animal matching would be brittle to construct.
"""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML
from deeplabcut.core.metrics.matching import PotentialMatch
from deeplabcut.pose_estimation_pytorch.data import PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.task import Task

from sollertia_video_tracking.training import evaluation
from sollertia_video_tracking.training.evaluation import (
    EvaluationSummary,
    _SplitMetrics,
    _resolve_snapshot,
    _matched_individual,
    _rank_worst_keypoints,
    _accumulate_split_rows,
    evaluate_trained_model,
    _derive_relative_image_path,
    _surviving_individual_indices,
    resolve_evaluation_batch_size,
    _realign_memory_replay_parameters,
)


# _SplitMetrics and EvaluationSummary dataclasses
def test_split_metrics_fields() -> None:
    """Verifies that the metrics dataclass stores each canonical value verbatim."""
    metrics = _SplitMetrics(images=5, rmse_px=2.0, rmse_pcutoff_px=1.5, map=90.0, mar=88.0, unmatched_images=1)
    assert metrics.images == 5
    assert metrics.rmse_px == 2.0
    assert metrics.rmse_pcutoff_px == 1.5
    assert metrics.map == 90.0
    assert metrics.mar == 88.0
    assert metrics.unmatched_images == 1


def test_evaluation_summary_gap_and_describe() -> None:
    """Verifies that the generalization gap is test-minus-train RMSE and the one-line description names both errors."""
    train = _make_split_metrics(2.0)
    test = _make_split_metrics(5.0)
    summary = EvaluationSummary(
        config=Path("/proj/config.yaml"),
        shuffle=1,
        snapshot="snapshot-190",
        feather_path=Path("/proj/eval/snapshot-190_evaluation.feather"),
        provenance_path=Path("/proj/eval/snapshot-190_evaluation.yaml"),
        pcutoff=0.6,
        train=train,
        test=test,
    )
    assert summary.generalization_gap_px == pytest.approx(3.0)
    text = summary.describe()
    assert "evaluated snapshot-190" in text
    assert "test RMSE 5.00px" in text
    assert "train 2.00px" in text
    assert "gap +3.00px" in text
    assert "snapshot-190_evaluation.feather" in text


# _derive_relative_image_path
def test_derive_relative_image_path_with_labeled_data() -> None:
    """Verifies that a path under ``labeled-data`` yields the video directory and the project-relative posix path."""
    video, relative = _derive_relative_image_path("/proj/labeled-data/vidX/img001.png")
    assert video == "vidX"
    assert relative == "labeled-data/vidX/img001.png"


def test_derive_relative_image_path_labeled_data_is_leaf() -> None:
    """Verifies that when ``labeled-data`` is the last component there is no video tail, so the video name is empty."""
    video, relative = _derive_relative_image_path("/a/labeled-data")
    assert video == ""
    assert relative == "labeled-data"


def test_derive_relative_image_path_without_labeled_data() -> None:
    """Verifies that a path with no ``labeled-data`` component falls back to the parent directory and file name."""
    video, relative = _derive_relative_image_path("/some/dir/frame.png")
    assert video == "dir"
    assert relative == "frame.png"


# _surviving_individual_indices
def test_surviving_individual_indices_drops_fully_occluded() -> None:
    """Verifies that individuals with no visible keypoint are dropped and those with at least one survive in order."""
    ground_truth = np.array(
        [
            [[1.0, 1.0, 2.0], [2.0, 2.0, 2.0]],  # fully visible -> kept
            [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]],  # fully occluded -> dropped
            [[1.0, 1.0, 0.0], [2.0, 2.0, 2.0]],  # one visible -> kept
        ],
        dtype=np.float32,
    )
    surviving = _surviving_individual_indices(ground_truth)
    assert surviving.tolist() == [0, 2]


# _matched_individual
def test_matched_individual_names_the_matched_row() -> None:
    """Verifies that a matched prediction whose ground truth equals a prepared row is named by its individual."""
    prepared = _prepared_ground_truth()
    name = _matched_individual(
        match=SimpleNamespace(gt=prepared[1]),
        prepared_ground_truth=prepared,
        surviving=np.array([0, 1]),
        individuals=["mouse_a", "mouse_b"],
        instance=0,
    )
    assert name == "mouse_b"


def test_matched_individual_unmatched_uses_order() -> None:
    """Verifies that a prediction with no ground truth is labeled by its per-image match order."""
    prepared = _prepared_ground_truth()
    name = _matched_individual(
        match=SimpleNamespace(gt=None),
        prepared_ground_truth=prepared,
        surviving=np.array([0, 1]),
        individuals=["mouse_a", "mouse_b"],
        instance=2,
    )
    assert name == "instance_2"


def test_matched_individual_no_surviving_uses_order() -> None:
    """Verifies that without a surviving-index map the prediction cannot be named, so it falls back to match order."""
    prepared = _prepared_ground_truth()
    name = _matched_individual(
        match=SimpleNamespace(gt=prepared[0]),
        prepared_ground_truth=prepared,
        surviving=None,
        individuals=["mouse_a"],
        instance=3,
    )
    assert name == "instance_3"


def test_matched_individual_out_of_range_index_uses_order() -> None:
    """Verifies that an out-of-range matched original index makes the prediction fall back to match order."""
    prepared = _prepared_ground_truth()
    name = _matched_individual(
        match=SimpleNamespace(gt=prepared[0]),
        prepared_ground_truth=prepared,
        surviving=np.array([5]),  # points past the single-name list
        individuals=["mouse_a"],
        instance=4,
    )
    assert name == "instance_4"


def test_matched_individual_no_matching_row_uses_order() -> None:
    """Verifies that when the matched ground truth equals no prepared row, the prediction falls back to match order."""
    prepared = _prepared_ground_truth()
    name = _matched_individual(
        match=SimpleNamespace(gt=np.array([[99.0, 99.0, 2.0], [98.0, 98.0, 2.0]], dtype=np.float32)),
        prepared_ground_truth=prepared,
        surviving=np.array([0, 1]),
        individuals=["mouse_a", "mouse_b"],
        instance=7,
    )
    assert name == "instance_7"


# _rank_worst_keypoints
def test_rank_worst_keypoints_sorts_and_truncates() -> None:
    """Verifies that ranking skips missing and NaN entries, sorts worst-first, and truncates to the reported five."""
    metrics = {
        "rmse_keypoint_0": 1.0,
        "rmse_keypoint_1": 9.0,
        "rmse_keypoint_2": float("nan"),  # NaN -> skipped
        "rmse_keypoint_3": 5.0,
        "rmse_keypoint_4": 7.0,
        "rmse_keypoint_5": 3.0,
        "rmse_keypoint_6": 8.0,
        # index 7 has no metric entry -> skipped
    }
    bodyparts = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
    ranked = _rank_worst_keypoints(metrics=metrics, bodyparts=bodyparts)
    assert [entry["bodypart"] for entry in ranked] == ["b1", "b6", "b4", "b3", "b5"]
    assert ranked[0]["rmse_px"] == 9.0
    assert len(ranked) == 5  # six valid keypoints truncated to five


def test_rank_worst_keypoints_empty_when_no_metrics() -> None:
    """Verifies that with no per-keypoint metrics the ranking is empty."""
    assert _rank_worst_keypoints(metrics={}, bodyparts=["only"]) == []


# resolve_evaluation_batch_size
def test_resolve_batch_size_one_is_kept() -> None:
    """Verifies that a requested size of one short-circuits without inspecting any frame."""
    loader = _loader_with_images({"train": [], "test": []})
    assert resolve_evaluation_batch_size(loader=loader, requested=1) == 1


def test_resolve_batch_size_uniform_resolution_keeps_request() -> None:
    """Verifies that when every frame shares one native resolution the requested batch size is kept."""
    uniform = {"train": [{"height": 100, "width": 200}], "test": [{"height": 100, "width": 200}]}
    assert resolve_evaluation_batch_size(loader=_loader_with_images(uniform), requested=8) == 8


def test_resolve_batch_size_multiple_resolutions_falls_to_one(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that frames spanning more than one resolution cannot be stacked, so the batch size drops to one."""
    multi = {"train": [{"height": 100, "width": 200}], "test": [{"height": 50, "width": 60}]}
    with caplog.at_level("INFO"):
        assert resolve_evaluation_batch_size(loader=_loader_with_images(multi), requested=8) == 1
    assert "multiple resolutions" in caplog.text


def test_resolve_batch_size_missing_dimension_falls_to_one() -> None:
    """Verifies that a frame whose header lacks a dimension is treated as unbatchable, so batch size drops to one."""
    missing = {"train": [{"height": None, "width": 200}], "test": []}
    assert resolve_evaluation_batch_size(loader=_loader_with_images(missing), requested=8) == 1


# _resolve_snapshot
def test_resolve_snapshot_returns_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the first snapshot returned by DeepLabCut is the resolved one."""
    loader = SimpleNamespace(model_folder="/m")
    monkeypatch.setattr(
        evaluation,
        "get_model_snapshots",
        lambda **kwargs: [SimpleNamespace(path=Path("/m/snapshot-100.pt")), "ignored"],
    )
    resolved = _resolve_snapshot(loader, index=0, task=Task.BOTTOM_UP)
    assert resolved.path.stem == "snapshot-100"


def test_resolve_snapshot_best_falls_back_to_last(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a missing best snapshot falls back to the last snapshot."""
    loader = SimpleNamespace(model_folder="/m")

    def fake(**kwargs):
        if kwargs["index"] == "best":
            message = "no best snapshot saved"
            raise ValueError(message)
        return [SimpleNamespace(path=Path("/m/snapshot-last.pt"))]

    monkeypatch.setattr(evaluation, "get_model_snapshots", fake)
    assert _resolve_snapshot(loader, index="best", task=Task.BOTTOM_UP).path.stem == "snapshot-last"


def test_resolve_snapshot_best_all_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that when both the best and last lookups fail and a snapshot is required, a ValueError is raised."""
    loader = SimpleNamespace(model_folder="/m")
    monkeypatch.setattr(evaluation, "get_model_snapshots", _raising_stub(IndexError("no snapshots")))
    with pytest.raises(ValueError, match="Unable to resolve"):
        _resolve_snapshot(loader, index="best", task=Task.DETECT)


def test_resolve_snapshot_best_all_missing_optional_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an absent optional best snapshot resolves to None rather than raising."""
    loader = SimpleNamespace(model_folder="/m")
    monkeypatch.setattr(evaluation, "get_model_snapshots", _raising_stub(ValueError("none")))
    assert _resolve_snapshot(loader, index="best", task=Task.DETECT, required=False) is None


def test_resolve_snapshot_int_index_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a missing integer-indexed snapshot raises without attempting the best-snapshot fallback."""
    loader = SimpleNamespace(model_folder="/m")
    monkeypatch.setattr(evaluation, "get_model_snapshots", _raising_stub(ValueError("bad index")))
    with pytest.raises(ValueError, match="Unable to resolve"):
        _resolve_snapshot(loader, index=5, task=Task.BOTTOM_UP)


def test_resolve_snapshot_int_index_missing_optional_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a missing optional integer-indexed snapshot resolves to None."""
    loader = SimpleNamespace(model_folder="/m")
    monkeypatch.setattr(evaluation, "get_model_snapshots", _raising_stub(IndexError("none")))
    assert _resolve_snapshot(loader, index=3, task=Task.BOTTOM_UP, required=False) is None


# _realign_memory_replay_parameters
def test_realign_no_weight_init_returns_unchanged() -> None:
    """Verifies that without a weight-initialization config the parameters are returned unchanged."""
    params = _params(bodyparts=["a", "b", "c"], unique=["u"], individuals=["single"])
    assert _realign_memory_replay_parameters(loader=_loader_with_weight_init(None), parameters=params) is params


def test_realign_non_memory_replay_returns_unchanged() -> None:
    """Verifies that a weight-init config that is not memory replay leaves the parameters unchanged."""
    params = _params(bodyparts=["a", "b", "c"], unique=["u"], individuals=["single"])
    weight_init = {"snapshot_path": "x", "with_decoder": False, "memory_replay": False}
    assert _realign_memory_replay_parameters(loader=_loader_with_weight_init(weight_init), parameters=params) is params


def test_realign_memory_replay_uses_weight_init_bodyparts() -> None:
    """Verifies that a memory-replay config with explicit bodyparts rebuilds the parameters in that bodypart space."""
    params = _params(bodyparts=["full0", "full1", "full2"], unique=["u"], individuals=["single"])
    weight_init = {
        "snapshot_path": "x",
        "with_decoder": True,
        "memory_replay": True,
        "bodyparts": ["proj_a", "proj_b"],
        "conversion_array": [0, 1],
    }
    realigned = _realign_memory_replay_parameters(loader=_loader_with_weight_init(weight_init), parameters=params)
    assert realigned.bodyparts == ["proj_a", "proj_b"]
    assert realigned.unique_bpts == ["u"]  # unique and individuals carried through
    assert realigned.individuals == ["single"]


def test_realign_memory_replay_without_bodyparts_reads_project(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a memory-replay config with no bodyparts falls back to the project configuration's bodyparts."""
    params = _params(bodyparts=["full0", "full1", "full2"], unique=["u"], individuals=["single"])
    weight_init = {"snapshot_path": "x", "with_decoder": True, "memory_replay": True, "conversion_array": [0, 1, 2]}
    monkeypatch.setattr(evaluation.auxiliaryfunctions, "get_bodyparts", lambda _cfg: ["proj0", "proj1"])
    realigned = _realign_memory_replay_parameters(
        loader=_loader_with_weight_init(weight_init=weight_init, project_cfg={"k": 1}),
        parameters=params,
    )
    assert realigned.bodyparts == ["proj0", "proj1"]


# _accumulate_split_rows
def test_accumulate_single_animal_matched_and_skips_missing() -> None:
    """Verifies that single-animal rows are matched with real DeepLabCut routines and absent images are skipped."""
    image = "/proj/labeled-data/vid/img0.png"
    ground_truth = {image: np.array([[[10.0, 10.0, 2.0], [20.0, 20.0, 2.0]]], dtype=np.float32)}
    predictions = {image: {"bodyparts": np.array([[[11.0, 10.0, 0.9], [19.0, 21.0, 0.8]]], dtype=np.float32)}}
    columns = _empty_columns()
    unmatched = _accumulate_split_rows(
        columns,
        snapshot_name="snapshot-100",
        split="train",
        image_paths=[image, "/proj/labeled-data/vid/absent.png"],  # second path is in neither dict -> skipped
        predictions=predictions,
        ground_truth=ground_truth,
        bodyparts=["nose", "tail"],
        individuals=["single"],
        single_animal=True,
        confidence_cutoff=0.85,
    )
    assert unmatched == 0
    assert len(columns["snapshot"]) == 2  # one image, two bodyparts
    assert columns["individual"] == ["single", "single"]
    assert columns["matched"] == [True, True]
    assert columns["above_pcutoff"] == [True, False]  # likelihoods 0.9 and 0.8 against a 0.85 cutoff
    assert columns["error_px"][0] == pytest.approx(1.0)
    assert all(np.isnan(oks) for oks in columns["oks"])  # single-animal RMSE match carries no OKS
    assert columns["video"] == ["vid", "vid"]
    assert columns["image"] == ["labeled-data/vid/img0.png", "labeled-data/vid/img0.png"]


def test_accumulate_single_animal_occluded_is_unmatched() -> None:
    """Verifies that a frame whose only individual is fully occluded yields no match and is counted as unmatched."""
    image = "/some/dir/frame.png"
    ground_truth = {image: np.array([[[10.0, 10.0, 0.0], [20.0, 20.0, 0.0]]], dtype=np.float32)}
    predictions = {image: {"bodyparts": np.array([[[11.0, 10.0, 0.9], [19.0, 21.0, 0.8]]], dtype=np.float32)}}
    columns = _empty_columns()
    unmatched = _accumulate_split_rows(
        columns,
        snapshot_name="s",
        split="test",
        image_paths=[image],
        predictions=predictions,
        ground_truth=ground_truth,
        bodyparts=["nose", "tail"],
        individuals=["single"],
        single_animal=True,
        confidence_cutoff=0.5,
    )
    assert unmatched == 1
    assert columns["snapshot"] == []  # no rows appended for an unmatched image


def test_accumulate_multi_animal_matched_and_unmatched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that multi-animal rows name a matched individual and label an unmatched prediction by match order."""
    image = "/proj/labeled-data/vid/img0.png"
    ground_truth = np.array(
        [[[10.0, 10.0, 2.0], [20.0, 20.0, 2.0]], [[30.0, 30.0, 2.0], [40.0, 40.0, 2.0]]], dtype=np.float32
    )
    predictions = np.array(
        [[[11.0, 10.0, 0.9], [19.0, 21.0, 0.8]], [[31.0, 30.0, 0.5], [41.0, 40.0, 0.4]]], dtype=np.float32
    )
    # Controls preparation and matching so the greedy multi-animal matcher needs no real assembly construction.
    # prepare_evaluation_data is called per image as {image: array}, so the stub returns that image's ground truth as
    # the prepared array _matched_individual locates match.gt within.
    monkeypatch.setattr(evaluation, "prepare_evaluation_data", lambda **kwargs: [(kwargs["ground_truth"][image], None)])
    matched = PotentialMatch(pose=predictions[0], score=0.85, gt=ground_truth[0], oks=0.8)
    unmatched = PotentialMatch(pose=predictions[1], score=0.45, gt=None, oks=0.0)
    monkeypatch.setattr(evaluation, "match_predictions_for_rmse", lambda **kwargs: [matched, unmatched])

    columns = _empty_columns()
    unmatched_count = _accumulate_split_rows(
        columns,
        snapshot_name="s",
        split="train",
        image_paths=[image],
        predictions={image: {"bodyparts": predictions}},
        ground_truth={image: ground_truth},
        bodyparts=["nose", "tail"],
        individuals=["mouse_a", "mouse_b"],
        single_animal=False,
        confidence_cutoff=0.5,
    )
    assert unmatched_count == 0  # the image had at least one matched individual
    assert columns["individual"] == ["mouse_a", "mouse_a", "instance_1", "instance_1"]
    assert columns["matched"] == [True, True, False, False]
    assert columns["oks"][0] == pytest.approx(0.8)
    assert np.isnan(columns["oks"][2])  # unmatched predictions carry NaN OKS
    assert columns["gt_x"][0] == pytest.approx(10.0)
    assert np.isnan(columns["gt_x"][2])  # unmatched predictions carry NaN ground truth
    assert columns["pred_x"][0] == pytest.approx(11.0)  # the matched prediction's own coordinate is written
    assert columns["error_px"][0] == pytest.approx(1.0)  # matched nose: (11, 10) is 1px from its (10, 10) label
    assert np.isnan(columns["error_px"][2])  # an unmatched prediction has no pixel error


def test_accumulate_skips_image_missing_prediction_key() -> None:
    """Verifies that an image lacking the requested prediction head is skipped."""
    image = "/proj/labeled-data/vid/img0.png"
    columns = _empty_columns()
    unmatched = _accumulate_split_rows(
        columns,
        snapshot_name="s",
        split="train",
        image_paths=[image],
        predictions={image: {"bodyparts": np.zeros((1, 2, 3), dtype=np.float32)}},
        ground_truth={image: np.zeros((1, 2, 3), dtype=np.float32)},
        bodyparts=["nose", "tail"],
        individuals=["unique"],
        single_animal=True,
        confidence_cutoff=0.5,
        prediction_key="unique_bodyparts",  # not present in the prediction dict -> skipped
    )
    assert unmatched == 0
    assert columns["snapshot"] == []


# evaluate_trained_model orchestration
def test_evaluate_trained_model_bottom_up_single_animal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies that a bottom-up single-animal run writes the feather and provenance sidecar and reports both errors."""
    image_train = "/proj/labeled-data/vid/train0.png"
    image_test = "/proj/labeled-data/vid/test0.png"
    gt = np.array([[[10.0, 10.0, 2.0], [20.0, 20.0, 2.0]]], dtype=np.float32)
    prediction = np.array([[[11.0, 10.0, 0.9], [19.0, 21.0, 0.8]]], dtype=np.float32)
    params = _params(bodyparts=["nose", "tail"], unique=[], individuals=["single"])
    loader = _FakeLoader(
        project_cfg={"pcutoff": 0.55},
        pose_task=Task.BOTTOM_UP,
        model_cfg={"metadata": {"with_identity": False}, "train_settings": {}, "detector": None},
        model_folder=str(tmp_path / "model"),
        evaluation_folder=str(tmp_path / "eval"),
        params=params,
        images={"train": [image_train], "test": [image_test]},
        ground_truth={"train": {image_train: gt}, "test": {image_test: gt}},
    )

    def evaluate_function(**kwargs):
        image = image_train if kwargs["mode"] == "train" else image_test
        rmse = 3.0 if kwargs["mode"] == "train" else 5.0
        metrics = {
            "rmse": rmse,
            "rmse_pcutoff": rmse - 0.5,
            "mAP": 80.0,
            "mAR": 70.0,
            "rmse_keypoint_0": 4.0,
            "rmse_keypoint_1": 6.0,
        }
        return metrics, {image: {"bodyparts": prediction}}

    _install_common_stubs(monkeypatch=monkeypatch, loader=loader, evaluate_function=evaluate_function)

    summary = evaluate_trained_model(tmp_path / "config.yaml")

    assert summary.snapshot == "snapshot-100"
    assert summary.pcutoff == pytest.approx(0.55)  # cutoff resolved from the project configuration
    assert summary.train.rmse_px == pytest.approx(3.0)
    assert summary.test.rmse_px == pytest.approx(5.0)
    assert summary.generalization_gap_px == pytest.approx(2.0)
    assert summary.feather_path.exists()
    assert summary.provenance_path.exists()

    frame = pl.read_ipc(summary.feather_path)
    assert set(frame.columns) == set(evaluation._FEATHER_SCHEMA)
    assert frame.height == 4  # two splits x one image x two bodyparts
    # Every row is the single individual and matched, and the nose prediction sits 1px from its label.
    assert frame["individual"].unique().to_list() == ["single"]
    assert frame["matched"].to_list() == [True, True, True, True]
    assert frame["snapshot"].unique().to_list() == ["snapshot-100"]
    assert frame["split"].to_list() == ["train", "train", "test", "test"]
    nose = frame.filter(pl.col("bodypart") == "nose")
    assert nose["pred_x"].to_list() == pytest.approx([11.0, 11.0])
    assert nose["gt_x"].to_list() == pytest.approx([10.0, 10.0])
    assert nose["error_px"].to_list() == pytest.approx([1.0, 1.0])
    # Likelihoods 0.9 (nose) and 0.8 (tail) both clear the 0.55 cutoff resolved from the project config.
    assert frame["above_pcutoff"].to_list() == [True, True, True, True]
    assert all(np.isnan(oks) for oks in frame["oks"].to_list())  # single-animal RMSE matches carry no OKS

    record = YAML().load(summary.provenance_path.read_text(encoding="utf-8"))
    assert record["engine"] == "pytorch"
    assert record["snapshot"] == "snapshot-100"
    assert record["shuffle"] == 1
    assert record["train_fraction"] == pytest.approx(0.95)
    assert record["single_animal"] is True
    assert record["detector_snapshot"] is None  # a bottom-up run has no detector
    assert record["pcutoff"] == pytest.approx(0.55)
    assert list(record["bodyparts"]) == ["nose", "tail"]
    assert list(record["individuals"]) == ["single"]
    assert record["feather"] == "snapshot-100_evaluation.feather"
    assert record["deeplabcut_version"] == evaluation.deeplabcut.__version__
    assert record["metrics"]["train"]["rmse_px"] == pytest.approx(3.0)
    assert record["metrics"]["test"]["rmse_px"] == pytest.approx(5.0)
    assert record["metrics"]["train"]["images"] == 1
    assert record["metrics"]["test"]["unmatched_images"] == 0
    assert record["generalization_gap_px"] == pytest.approx(2.0)
    # worst_keypoints ranks the test per-keypoint RMSE, tail (6.0) ahead of nose (4.0).
    assert [entry["bodypart"] for entry in record["worst_keypoints"]] == ["tail", "nose"]
    assert record["worst_keypoints"][0]["rmse_px"] == pytest.approx(6.0)


def test_evaluate_trained_model_top_down_detector_and_unique(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies that a top-down run with a detector, unique bodyparts, and an explicit cutoff scores every head."""
    image_train = "/proj/labeled-data/vid/train0.png"
    image_test = "/proj/labeled-data/vid/test0.png"
    gt = np.array([[[10.0, 10.0, 2.0], [20.0, 20.0, 2.0]]], dtype=np.float32)
    prediction = np.array([[[11.0, 10.0, 0.9], [19.0, 21.0, 0.8]]], dtype=np.float32)
    unique_gt = np.array([[[5.0, 5.0, 2.0]]], dtype=np.float32)
    unique_prediction = np.array([[[6.0, 5.0, 0.7]]], dtype=np.float32)
    params = _params(bodyparts=["nose", "tail"], unique=["led"], individuals=["single"])
    loader = _FakeLoader(
        project_cfg={"pcutoff": 0.6},
        pose_task=Task.TOP_DOWN,
        model_cfg={"metadata": {"with_identity": False}, "train_settings": {}, "detector": {"type": "ssd"}},
        model_folder=str(tmp_path / "model"),
        evaluation_folder=str(tmp_path / "eval"),
        params=params,
        images={"train": [image_train], "test": [image_test]},
        ground_truth={"train": {image_train: gt}, "test": {image_test: gt}},
        unique_ground_truth={"train": {image_train: unique_gt}, "test": {image_test: unique_gt}},
    )

    def evaluate_function(**kwargs):
        image = image_train if kwargs["mode"] == "train" else image_test
        metrics = {"rmse": 3.0, "rmse_pcutoff": 2.5, "mAP": 80.0, "mAR": 70.0, "rmse_keypoint_0": 4.0}
        predictions = {image: {"bodyparts": prediction, "unique_bodyparts": unique_prediction}}
        return metrics, predictions

    _install_common_stubs(monkeypatch=monkeypatch, loader=loader, evaluate_function=evaluate_function)

    summary = evaluate_trained_model(
        str(tmp_path / "config.yaml"),
        confidence_cutoff=0.4,
        write_provenance=True,
    )

    assert summary.pcutoff == pytest.approx(0.4)  # explicit cutoff overrides the project value
    frame = pl.read_ipc(summary.feather_path)
    # Two splits, one image each, two main bodyparts + one unique bodypart -> six rows.
    assert frame.height == 6
    led = frame.filter(pl.col("bodypart") == "led")
    assert led.height == 2  # one unique-head row per split
    assert led["individual"].unique().to_list() == ["unique"]  # unique bodyparts are labeled "unique"
    assert led["pred_x"].to_list() == pytest.approx([6.0, 6.0])
    assert led["gt_x"].to_list() == pytest.approx([5.0, 5.0])
    assert led["error_px"].to_list() == pytest.approx([1.0, 1.0])  # (6, 5) is 1px from the (5, 5) label

    record = YAML().load(summary.provenance_path.read_text(encoding="utf-8"))
    assert record["detector_snapshot"] == "snapshot-detector-050"  # the detector snapshot is recorded
    assert list(record["bodyparts"]) == ["nose", "tail"]  # the unique head is not part of the pose bodyparts
    assert list(record["individuals"]) == ["single"]
    assert record["pcutoff"] == pytest.approx(0.4)
    # Only rmse_keypoint_0 (nose) is present in the test metrics, so it is the lone ranked keypoint.
    assert [entry["bodypart"] for entry in record["worst_keypoints"]] == ["nose"]


def test_evaluate_trained_model_provenance_failure_removes_feather(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verifies that when the provenance sidecar cannot be written, the feather is removed and OSError propagates."""
    image_train = "/proj/labeled-data/vid/train0.png"
    image_test = "/proj/labeled-data/vid/test0.png"
    gt = np.array([[[10.0, 10.0, 2.0], [20.0, 20.0, 2.0]]], dtype=np.float32)
    prediction = np.array([[[11.0, 10.0, 0.9], [19.0, 21.0, 0.8]]], dtype=np.float32)
    params = _params(bodyparts=["nose", "tail"], unique=[], individuals=["single"])
    loader = _FakeLoader(
        project_cfg={"pcutoff": 0.6},
        pose_task=Task.BOTTOM_UP,
        model_cfg={"metadata": {"with_identity": False}, "train_settings": {}, "detector": None},
        model_folder=str(tmp_path / "model"),
        evaluation_folder=str(tmp_path / "eval"),
        params=params,
        images={"train": [image_train], "test": [image_test]},
        ground_truth={"train": {image_train: gt}, "test": {image_test: gt}},
    )

    def evaluate_function(**kwargs):
        image = image_train if kwargs["mode"] == "train" else image_test
        metrics = {"rmse": 3.0, "rmse_pcutoff": 2.5, "mAP": 80.0, "mAR": 70.0, "rmse_keypoint_0": 4.0}
        return metrics, {image: {"bodyparts": prediction}}

    _install_common_stubs(monkeypatch=monkeypatch, loader=loader, evaluate_function=evaluate_function)

    monkeypatch.setattr(evaluation, "_write_provenance", _raising_stub(OSError("disk full")))

    feather_path = tmp_path / "eval" / "snapshot-100_evaluation.feather"
    with pytest.raises(OSError, match="disk full"):
        evaluate_trained_model(tmp_path / "config.yaml")
    assert not feather_path.exists()  # the feather written earlier is removed so no sidecar-less feather survives


def test_evaluate_trained_model_reduces_batch_for_multi_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verifies that a requested batch larger than one is reduced when labeled frames span multiple resolutions."""
    image_train = "/proj/labeled-data/vid/train0.png"
    image_test = "/proj/labeled-data/vid/test0.png"
    gt = np.array([[[10.0, 10.0, 2.0], [20.0, 20.0, 2.0]]], dtype=np.float32)
    prediction = np.array([[[11.0, 10.0, 0.9], [19.0, 21.0, 0.8]]], dtype=np.float32)
    params = _params(bodyparts=["nose", "tail"], unique=[], individuals=["single"])
    loader = _FakeLoader(
        project_cfg={"pcutoff": 0.6},
        pose_task=Task.BOTTOM_UP,
        model_cfg={"metadata": {"with_identity": False}, "train_settings": {}, "detector": None},
        model_folder=str(tmp_path / "model"),
        evaluation_folder=str(tmp_path / "eval"),
        params=params,
        images={"train": [image_train], "test": [image_test]},
        ground_truth={"train": {image_train: gt}, "test": {image_test: gt}},
        data={
            "train": {"images": [{"height": 100, "width": 200}]},
            "test": {"images": [{"height": 50, "width": 60}]},
        },
    )

    recorded_batch = {}

    def evaluate_function(**kwargs):
        image = image_train if kwargs["mode"] == "train" else image_test
        metrics = {"rmse": 3.0, "rmse_pcutoff": 2.5, "mAP": 80.0, "mAR": 70.0, "rmse_keypoint_0": 4.0}
        return metrics, {image: {"bodyparts": prediction}}

    def capture_runners(**kwargs):
        recorded_batch["batch_size"] = kwargs["batch_size"]
        return ("pose_runner", "detector_runner")

    _install_common_stubs(monkeypatch=monkeypatch, loader=loader, evaluate_function=evaluate_function)
    monkeypatch.setattr(evaluation, "get_inference_runners", capture_runners)

    summary = evaluate_trained_model(tmp_path / "config.yaml", batch_size=8, write_provenance=False)

    assert recorded_batch["batch_size"] == 1  # multi-resolution frames force a batch size of one
    assert summary.feather_path.exists()
    assert not summary.provenance_path.exists()  # provenance was not requested


def _raising_stub(exception: Exception):
    """Builds a stub that raises the given exception, mimicking a DeepLabCut handoff that fails."""

    def _stub(*args, **kwargs):
        raise exception

    return _stub


def _make_split_metrics(rmse_px: float) -> _SplitMetrics:
    return _SplitMetrics(
        images=3,
        rmse_px=rmse_px,
        rmse_pcutoff_px=rmse_px - 0.5,
        map=80.0,
        mar=70.0,
        unmatched_images=0,
    )


def _prepared_ground_truth() -> np.ndarray:
    return np.array(
        [[[10.0, 10.0, 2.0], [20.0, 20.0, 2.0]], [[30.0, 30.0, 2.0], [40.0, 40.0, 2.0]]],
        dtype=np.float32,
    )


def _loader_with_images(images_by_split: dict[str, list[dict[str, int | None]]]) -> SimpleNamespace:
    return SimpleNamespace(load_data=lambda split: {"images": images_by_split[split]})


def _params(bodyparts: list[str], unique: list[str], individuals: list[str]) -> PoseDatasetParameters:
    return PoseDatasetParameters(bodyparts=bodyparts, unique_bpts=unique, individuals=individuals)


def _loader_with_weight_init(weight_init: dict | None, project_cfg: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(model_cfg={"train_settings": {"weight_init": weight_init}}, project_cfg=project_cfg or {})


def _empty_columns() -> dict[str, list]:
    return {name: [] for name in evaluation._FEATHER_SCHEMA}


class _FakeLoader:
    """Stands in for DeepLabCut's DLCLoader, exposing only the attributes the evaluator reads."""

    def __init__(
        self,
        *,
        project_cfg,
        pose_task,
        model_cfg,
        model_folder,
        evaluation_folder,
        params,
        images,
        ground_truth,
        unique_ground_truth=None,
        data=None,
        shuffle=1,
        train_fraction=0.95,
    ):
        self.project_cfg = project_cfg
        self.pose_task = pose_task
        self.model_cfg = model_cfg
        self.model_folder = model_folder
        self.evaluation_folder = evaluation_folder
        self.shuffle = shuffle
        self.train_fraction = train_fraction
        self._params = params
        self._images = images
        self._ground_truth = ground_truth
        self._unique_ground_truth = unique_ground_truth or {}
        self._data = data or {}

    def get_dataset_parameters(self):
        return self._params

    def image_filenames(self, split):
        return self._images[split]

    def ground_truth_keypoints(self, split, *, unique_bodypart=False):
        return self._unique_ground_truth[split] if unique_bodypart else self._ground_truth[split]

    def load_data(self, split):
        return self._data[split]


def _install_common_stubs(monkeypatch: pytest.MonkeyPatch, loader: _FakeLoader, evaluate_function) -> None:
    """Redirects every DeepLabCut handoff the evaluator makes to in-process fakes."""
    monkeypatch.setattr(evaluation, "DLCLoader", lambda **kwargs: loader)
    monkeypatch.setattr(evaluation, "get_inference_runners", lambda **kwargs: ("pose_runner", "detector_runner"))
    monkeypatch.setattr(evaluation, "evaluate", evaluate_function)

    def fake_snapshots(**kwargs):
        folder = Path(kwargs["model_folder"])
        if kwargs["task"] == Task.DETECT:
            return [SimpleNamespace(path=folder / "snapshot-detector-050.pt")]
        return [SimpleNamespace(path=folder / "snapshot-100.pt")]

    monkeypatch.setattr(evaluation, "get_model_snapshots", fake_snapshots)
