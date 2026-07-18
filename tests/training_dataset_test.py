"""Contains tests for the DeepLabCut training-dataset creation wrapper.

Every DeepLabCut handoff (``available_models``/``available_detectors``, ``build_weight_init``, the two
``create_training_dataset`` entry points, ``get_existing_shuffle_indices``, and the augmentation catalog through
``dlc_compat``) is monkeypatched at the exact module binding so no real DLC runtime, GPU, or project on disk is needed.
"""

import io
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from sollertia_video_tracking.training import dataset
from sollertia_video_tracking.training.dataset import (
    _SUPER_ANIMAL_DATASETS,
    _UNANNOTATED_VIDEO_NOTICE,
    TrainingDatasetSummary,
    WeightInitializationMethod,
    create_training_dataset,
    _UnannotatedNoticeFilter,
    get_available_pose_models,
    get_available_super_animals,
    build_superanimal_weight_init,
    get_available_object_detectors,
    _suppress_unannotated_video_notices,
    build_conditional_top_down_conditions,
)


# WeightInitializationMethod enum
def test_weight_initialization_method_values() -> None:
    """Verifies that each enum member carries the exact string the DeepLabCut GUI uses and behaves as a string."""
    assert WeightInitializationMethod.IMAGENET == "imagenet"
    assert WeightInitializationMethod.TRANSFER == "transfer"
    assert WeightInitializationMethod.FINE_TUNE == "fine-tune"
    # StrEnum members are usable as plain strings.
    assert isinstance(WeightInitializationMethod.IMAGENET, str)
    assert f"{WeightInitializationMethod.FINE_TUNE}" == "fine-tune"


# TrainingDatasetSummary.describe
def test_summary_describe_full_detail() -> None:
    """Verifies that with every optional field set, the description names the model, detector, weights, split, and
    config.
    """
    summary = TrainingDatasetSummary(
        config=Path("/proj/config.yaml"),
        shuffle=3,
        net_type="resnet50",
        detector_type="ssdlite",
        weight_init="superanimal_quadruped (transfer)",
        from_shuffle=2,
    )
    text = summary.describe()
    assert "created shuffle 3 for resnet50 + ssdlite" in text
    assert "[superanimal_quadruped (transfer)]" in text
    assert "(split from shuffle 2)" in text
    assert str(Path("/proj/config.yaml")) in text


def test_summary_describe_defaults() -> None:
    """Verifies that with no net_type, detector, or split source, the description falls back to the project-default
    wording.
    """
    summary = TrainingDatasetSummary(
        config=Path("config.yaml"),
        shuffle=1,
        net_type=None,
        detector_type=None,
        weight_init="imagenet",
        from_shuffle=None,
    )
    text = summary.describe()
    assert text == "created shuffle 1 for project default [imagenet] from config.yaml"
    assert "+" not in text
    assert "split from" not in text


def test_summary_describe_from_shuffle_zero_still_shows_split() -> None:
    """Verifies that a ``from_shuffle`` of 0 is falsy but not None, so the split note must still appear."""
    summary = TrainingDatasetSummary(
        config=Path("c.yaml"),
        shuffle=5,
        net_type="x",
        detector_type=None,
        weight_init="imagenet",
        from_shuffle=0,
    )
    assert "(split from shuffle 0)" in summary.describe()


# Catalog accessors
def test_get_available_pose_models_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the pose-model catalog is returned sorted and as an immutable tuple."""
    monkeypatch.setattr(dataset, "available_models", lambda: ["zeb", "alpha", "mid"])
    assert get_available_pose_models() == ("alpha", "mid", "zeb")


def test_get_available_object_detectors_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the detector catalog is returned sorted and as an immutable tuple."""
    monkeypatch.setattr(dataset, "available_detectors", lambda: ["ssd", "fasterrcnn"])
    assert get_available_object_detectors() == ("fasterrcnn", "ssd")


def test_get_available_super_animals() -> None:
    """Verifies that the SuperAnimal list is the module constant, matching the GUI's fixed options."""
    assert get_available_super_animals() == _SUPER_ANIMAL_DATASETS
    assert get_available_super_animals() == (
        "superanimal_bird",
        "superanimal_quadruped",
        "superanimal_topviewmouse",
    )


# build_superanimal_weight_init
def test_build_superanimal_weight_init_strips_top_down_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a ``top_down_`` network prefix is stripped to recover the SuperAnimal pose-model name
    and all args forward.
    """
    recorded: dict[str, object] = {}
    sentinel = object()

    def fake_build(**kwargs: object) -> object:
        recorded.update(kwargs)
        return sentinel

    monkeypatch.setattr(dataset, "build_weight_init", fake_build)
    result = build_superanimal_weight_init(
        config=Path("/p/config.yaml"),
        super_animal="superanimal_quadruped",
        network_type="top_down_resnet50",
        detector_type="ssdlite",
        fine_tune=True,
        memory_replay=True,
        customized_pose_checkpoint="/p/pose.pt",
        customized_detector_checkpoint="/p/det.pt",
    )
    assert result is sentinel
    assert recorded["cfg"] == str(Path("/p/config.yaml"))
    assert recorded["model_name"] == "resnet50"  # top_down_ prefix removed
    assert recorded["super_animal"] == "superanimal_quadruped"
    assert recorded["detector_name"] == "ssdlite"
    assert recorded["with_decoder"] is True  # fine_tune maps to with_decoder
    assert recorded["memory_replay"] is True
    assert recorded["customized_pose_checkpoint"] == "/p/pose.pt"
    assert recorded["customized_detector_checkpoint"] == "/p/det.pt"


def test_build_superanimal_weight_init_without_prefix_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a network name without the prefix passes through untouched, and the optional args default
    correctly.
    """
    recorded: dict[str, object] = {}

    def fake_build(**kwargs: object) -> str:
        recorded.update(kwargs)
        return "wi"

    monkeypatch.setattr(dataset, "build_weight_init", fake_build)
    build_superanimal_weight_init(
        config="config.yaml",
        super_animal="superanimal_bird",
        network_type="resnet50",
        detector_type=None,
    )
    assert recorded["model_name"] == "resnet50"  # no prefix to strip
    assert recorded["detector_name"] is None
    assert recorded["with_decoder"] is False  # fine_tune default
    assert recorded["memory_replay"] is False  # memory_replay default
    assert recorded["customized_pose_checkpoint"] is None
    assert recorded["customized_detector_checkpoint"] is None


# build_conditional_top_down_conditions
def test_ctd_conditions_h5_returns_path_unchanged() -> None:
    """Verifies that an ``.h5`` predictions file is returned as a Path, unchanged."""
    result = build_conditional_top_down_conditions("/p/preds.h5")
    assert result == Path("/p/preds.h5")
    assert isinstance(result, Path)


def test_ctd_conditions_json_returns_path_unchanged() -> None:
    """Verifies that a ``.json`` predictions file is returned as a Path, unchanged."""
    result = build_conditional_top_down_conditions(Path("/p/preds.json"))
    assert result == Path("/p/preds.json")


def test_ctd_conditions_suffix_comparison_is_case_insensitive() -> None:
    """Verifies that an uppercase suffix still matches the predictions extensions (the suffix is lowercased before
    comparison).
    """
    assert build_conditional_top_down_conditions("/p/PREDS.H5") == Path("/p/PREDS.H5")


def test_ctd_conditions_snapshot_returns_shuffle_and_name() -> None:
    """Verifies that a ``.pt`` snapshot path yields a ``(shuffle, snapshot_name)`` pair parsed from its ``shuffleN``
    segment.
    """
    result = build_conditional_top_down_conditions("/p/dlc-models/iteration-0/shuffle3/snapshot-050.pt")
    assert result == (3, "snapshot-050.pt")


def test_ctd_conditions_snapshot_without_shuffle_raises() -> None:
    """Verifies that a snapshot path with no ``shuffleN`` segment cannot be parsed and raises."""
    with pytest.raises(ValueError, match="must contain a 'shuffleN' segment"):
        build_conditional_top_down_conditions("/p/snapshot-050.pt")


def test_ctd_conditions_unsupported_suffix_raises() -> None:
    """Verifies that a suffix that is neither a predictions file nor a snapshot raises."""
    with pytest.raises(ValueError, match=r"must be a \.h5 or \.json"):
        build_conditional_top_down_conditions("/p/weights.pth")


# create_training_dataset
def _patch_dlc(
    monkeypatch: pytest.MonkeyPatch,
    *,
    models: tuple[str, ...] = ("resnet50",),
    detectors: tuple[str, ...] = ("ssdlite",),
    existing: tuple[int, ...] = (1,),
) -> dict[str, list[dict[str, object]]]:
    """Patches every DLC boundary create_training_dataset touches and records the two create calls."""
    monkeypatch.setattr(dataset, "available_models", lambda: list(models))
    monkeypatch.setattr(dataset, "available_detectors", lambda: list(detectors))
    calls: dict[str, list[dict[str, object]]] = {"fresh": [], "existing_split": []}
    monkeypatch.setattr(dataset, "dlc_create_training_dataset", lambda **kwargs: calls["fresh"].append(kwargs))
    monkeypatch.setattr(
        dataset,
        "dlc_create_training_dataset_from_existing_split",
        lambda **kwargs: calls["existing_split"].append(kwargs),
    )
    monkeypatch.setattr(dataset, "get_existing_shuffle_indices", lambda _cfg: list(existing))
    return calls


def test_create_training_dataset_fresh_split_imagenet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a default run draws a fresh split, uses ImageNet weights, and returns a matching summary."""
    calls = _patch_dlc(monkeypatch, existing=(1,))
    summary = create_training_dataset(
        "/p/config.yaml",
        shuffle=1,
        network_type="resnet50",
        detector_type="ssdlite",
    )
    assert isinstance(summary, TrainingDatasetSummary)
    assert summary.config == Path("/p/config.yaml")
    assert summary.shuffle == 1
    assert summary.net_type == "resnet50"
    assert summary.detector_type == "ssdlite"
    assert summary.weight_init == "imagenet"
    assert summary.from_shuffle is None
    # The fresh-split entry point ran, the existing-split one did not.
    assert len(calls["fresh"]) == 1
    assert not calls["existing_split"]
    fresh = calls["fresh"][0]
    assert fresh["config"] == "/p/config.yaml"  # the project config reaches the DLC boundary as a string
    assert fresh["Shuffles"] == [1]
    assert fresh["net_type"] == "resnet50"
    assert fresh["detector_type"] == "ssdlite"
    assert fresh["userfeedback"] is True  # overwrite False -> user feedback enabled
    assert fresh["weight_init"] is None
    assert fresh["ctd_conditions"] is None


def test_create_training_dataset_from_shuffle_reuses_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that from_shuffle routes to the existing-split entry point, forwards every arg, and disables feedback
    on overwrite.
    """
    calls = _patch_dlc(monkeypatch, existing=(2,))
    weight_init = SimpleNamespace(with_decoder=False, dataset="superanimal_quadruped")
    ctd = Path("/p/preds.h5")
    summary = create_training_dataset(
        "/p/config.yaml",
        shuffle=2,
        network_type="resnet50",
        detector_type="ssdlite",
        weight_initialization=weight_init,
        conditional_top_down_conditions=ctd,
        from_shuffle=1,
        from_training_set_index=1,
        overwrite=True,
    )
    assert len(calls["existing_split"]) == 1
    assert not calls["fresh"]
    reuse = calls["existing_split"][0]
    assert reuse["config"] == "/p/config.yaml"
    assert reuse["from_shuffle"] == 1
    assert reuse["from_trainsetindex"] == 1
    assert reuse["shuffles"] == [2]
    # Every model/weight/condition argument must reach the existing-split entry point unchanged.
    assert reuse["net_type"] == "resnet50"
    assert reuse["detector_type"] == "ssdlite"
    assert reuse["weight_init"] is weight_init
    assert reuse["ctd_conditions"] is ctd
    assert reuse["userfeedback"] is False  # overwrite True -> user feedback disabled
    assert summary.from_shuffle == 1
    assert summary.detector_type == "ssdlite"
    assert summary.weight_init == "superanimal_quadruped (transfer)"


def test_create_training_dataset_transfer_weight_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a weight initialization without a decoder head is described as transfer learning and forwarded
    to DLC.
    """
    calls = _patch_dlc(monkeypatch, existing=(1,))
    weight_init = SimpleNamespace(with_decoder=False, dataset="superanimal_quadruped")
    summary = create_training_dataset("/p/config.yaml", weight_initialization=weight_init)
    assert summary.weight_init == "superanimal_quadruped (transfer)"
    # The exact weight-init object must be handed to DLC, not just reflected in the summary string.
    assert calls["fresh"][0]["weight_init"] is weight_init


def test_create_training_dataset_fine_tune_weight_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a weight initialization that loads the decoder head is described as fine-tuning and forwarded
    to DLC.
    """
    calls = _patch_dlc(monkeypatch, existing=(1,))
    weight_init = SimpleNamespace(with_decoder=True, dataset="superanimal_bird")
    summary = create_training_dataset("/p/config.yaml", weight_initialization=weight_init)
    assert summary.weight_init == "superanimal_bird (fine-tune)"
    assert calls["fresh"][0]["weight_init"] is weight_init


def test_create_training_dataset_fresh_forwards_ctd_conditions_and_coerces_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that the fresh-split path forwards conditional-top-down conditions verbatim and coerces a Path config
    to a string.
    """
    calls = _patch_dlc(monkeypatch, existing=(1,))
    ctd = (3, "snapshot-050.pt")
    create_training_dataset(
        Path("/p/config.yaml"),
        shuffle=1,
        network_type="resnet50",
        conditional_top_down_conditions=ctd,
    )
    fresh = calls["fresh"][0]
    assert fresh["config"] == str(Path("/p/config.yaml"))  # Path coerced to str at the DLC boundary
    assert fresh["ctd_conditions"] is ctd  # the conditioning source is forwarded unchanged, not dropped to None


def test_create_training_dataset_invalid_network_type_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an unknown network_type is rejected before any DLC call."""
    _patch_dlc(monkeypatch, models=("resnet50",))
    with pytest.raises(ValueError, match="network_type must be one of"):
        create_training_dataset("/p/config.yaml", network_type="bogus")


def test_create_training_dataset_invalid_detector_type_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an unknown detector_type is rejected before any DLC call."""
    _patch_dlc(monkeypatch, detectors=("ssdlite",))
    with pytest.raises(ValueError, match="detector_type must be"):
        create_training_dataset("/p/config.yaml", network_type="resnet50", detector_type="bogus")


def test_create_training_dataset_missing_shuffle_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that when DLC silently creates no shuffle, the wrapper detects the missing index and raises."""
    # The requested shuffle never appears in the existing-index listing.
    _patch_dlc(monkeypatch, existing=())
    with pytest.raises(ValueError, match="DeepLabCut created no shuffle"):
        create_training_dataset("/p/config.yaml", shuffle=1)


# _UnannotatedNoticeFilter
class _RecordingStream:
    """A minimal text stream that records writes and flush calls for filter assertions."""

    def __init__(self) -> None:
        self.written: list[str] = []
        self.flush_count = 0
        self.encoding = "utf-8"

    def write(self, text: str) -> int:
        self.written.append(text)
        return len(text)

    def flush(self) -> None:
        self.flush_count += 1


def test_filter_write_buffers_until_newline() -> None:
    """Verifies that text without a line break is buffered and only forwarded once the line completes."""
    target = _RecordingStream()
    stream_filter = _UnannotatedNoticeFilter(target=target, marker="DROP")
    accepted = stream_filter.write("partial")
    assert accepted == len("partial")  # honors the write contract
    assert target.written == []  # nothing forwarded yet
    stream_filter.write(" line\n")
    assert target.written == ["partial line\n"]


def test_filter_write_drops_marker_lines_forwards_others() -> None:
    """Verifies that completed lines containing the marker are dropped; other completed lines are forwarded verbatim."""
    target = _RecordingStream()
    stream_filter = _UnannotatedNoticeFilter(target=target, marker="DROP")
    stream_filter.write("keep me\nplease DROP this\nkeep two\n")
    assert target.written == ["keep me\n", "keep two\n"]


def test_filter_flush_delegates_to_target() -> None:
    """Verifies that flushing the filter flushes the underlying target stream."""
    target = _RecordingStream()
    stream_filter = _UnannotatedNoticeFilter(target=target, marker="DROP")
    stream_filter.flush()
    assert target.flush_count == 1


def test_filter_drain_forwards_incomplete_trailing_line() -> None:
    """Verifies that draining forwards a buffered trailing line that never got a newline, then clears the buffer."""
    target = _RecordingStream()
    stream_filter = _UnannotatedNoticeFilter(target=target, marker="DROP")
    stream_filter.write("trailing no newline")
    stream_filter.drain()
    assert target.written == ["trailing no newline"]
    # The buffer is cleared, so a second drain forwards nothing (empty-pending branch).
    stream_filter.drain()
    assert target.written == ["trailing no newline"]


def test_filter_drain_drops_marker_trailing_line() -> None:
    """Verifies that draining drops a buffered trailing line that contains the marker."""
    target = _RecordingStream()
    stream_filter = _UnannotatedNoticeFilter(target=target, marker="DROP")
    stream_filter.write("this has DROP in it")
    stream_filter.drain()
    assert target.written == []


def test_filter_getattr_delegates_to_target() -> None:
    """Verifies that attributes the filter does not define are read from the target stream."""
    target = _RecordingStream()
    stream_filter = _UnannotatedNoticeFilter(target=target, marker="DROP")
    assert stream_filter.encoding == "utf-8"  # resolved via __getattr__ on the target


# _suppress_unannotated_video_notices
def test_suppress_unannotated_notices_filters_marker_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that inside the context, marker lines are dropped, plain lines pass through, and trailing text is
    drained on exit.
    """
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    with _suppress_unannotated_video_notices():
        print("visible line")
        print(f"video path {_UNANNOTATED_VIDEO_NOTICE} extra")
        # A trailing partial line with no newline must be flushed by drain on context exit.
        sys.stdout.write("trailing partial")
    output = captured.getvalue()
    assert "visible line" in output
    assert _UNANNOTATED_VIDEO_NOTICE not in output  # the noticed line was filtered out
    assert "trailing partial" in output  # drained on exit
