"""Serializes trained DeepLabCut shuffles into portable model assets and loads them back for deployment."""

import copy
from enum import StrEnum
import shutil
from typing import Any
from pathlib import Path
from datetime import UTC, datetime
import tempfile
from dataclasses import dataclass

import deeplabcut
from ruamel.yaml import YAML
from deeplabcut.pose_estimation_pytorch.data import DLCLoader
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.apis.utils import get_model_snapshots

from .utilities import extract_tar, compute_sha256, read_tar_member, resolve_slvt_version, pack_directory_to_tar

_FORMAT_VERSION: int = 1
"""The model-asset layout version this module writes and can load; a differing major version is refused on load."""

_MANIFEST_FILENAME: str = "manifest.yaml"
"""The name of the manifest, stored as the first member of the archive so a reader can inspect it without unpacking."""

_MODEL_DIRECTORY: str = "model"
"""The archive-relative directory holding the extracted DeepLabCut project shell (config plus the shuffle subtree)."""

_CONFIG_RELPATH: str = f"{_MODEL_DIRECTORY}/config.yaml"
"""The archive-relative path of the project configuration inside the model shell."""

_ASSET_SUFFIX: str = ".slvtmodel"
"""The file extension identifying a Sollertia model asset; the archive underneath is a standard tar."""

_SNAPSHOT_INDEX: int = 0
"""The snapshot index the deployment inference uses, since exactly one pose (and one detector) snapshot is bundled."""


class ArchiveCompression(StrEnum):
    """Defines the compression applied to a model asset's tar archive."""

    NONE = "none"
    """Stores the archive uncompressed, the default; the payload is dominated by already-compressed weights, so this
    keeps runtime extraction fast without meaningfully growing the file."""
    GZIP = "gzip"
    """Compresses the archive with gzip, trading extraction speed for a smaller file on slow transport links."""
    XZ = "xz"
    """Compresses the archive with xz for the smallest file, at the highest extraction cost."""

    @property
    def tar_mode(self) -> str:
        """Returns the ``tarfile`` write mode that applies this compression."""
        return {ArchiveCompression.NONE: "w", ArchiveCompression.GZIP: "w:gz", ArchiveCompression.XZ: "w:xz"}[self]

    @property
    def file_suffix(self) -> str:
        """Returns the extension appended after the asset suffix, empty for an uncompressed archive."""
        return {ArchiveCompression.NONE: "", ArchiveCompression.GZIP: ".gz", ArchiveCompression.XZ: ".xz"}[self]


@dataclass(frozen=True, slots=True)
class PayloadEntry:
    """Records one file inside a model asset with its checksum, so a corrupted or truncated asset is caught on load."""

    path: str
    """The archive-relative path of the file, in POSIX form."""
    sha256: str
    """The SHA-256 hex digest of the file's contents at export time."""
    size_bytes: int
    """The size of the file in bytes at export time."""


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Describes a portable model asset's identity and contents, written at export and read back at deployment.

    Notes:
        The manifest is the first member of the archive, so a consumer can read the model's identity (task, bodyparts,
        snapshots) and verify a rig without unpacking the weights. The pose-identity fields are copied from the trained
        shuffle's configuration; ``source`` holds reproducibility-only provenance and ``payload`` holds the per-file
        checksums used to verify the extracted shell.
    """

    format_version: int
    """The asset layout version, checked against the loader's supported version."""
    slvt_version: str
    """The sollertia-video-tracking version that produced the asset."""
    dlc_version: str
    """The DeepLabCut version that produced the asset."""
    created: str
    """The UTC timestamp the asset was created, in ISO-8601 form."""
    multianimal: bool
    """Determines whether the model tracks more than one individual."""
    task: str
    """The pose task the model performs (bottom-up, top-down, or conditional top-down)."""
    net_type: str
    """The network architecture identifier the model was trained with."""
    scorer: str
    """The DeepLabCut scorer string the model's predictions are named with, kept for reference."""
    bodyparts: tuple[str, ...]
    """The ordered bodypart names the model predicts."""
    individuals: tuple[str, ...]
    """The ordered individual names, holding a single entry for a single-animal model."""
    unique_bodyparts: tuple[str, ...]
    """The ordered unique-bodypart names, empty when the model has none."""
    with_identity: bool
    """Determines whether the model predicts an identity head."""
    shuffle: int
    """The shuffle index the exported model belongs to."""
    training_fraction: float
    """The training fraction the exported shuffle was trained with."""
    iteration: int
    """The project iteration the exported shuffle belongs to."""
    config_relpath: str
    """The archive-relative path of the project configuration inside the model shell."""
    model_relpath: str
    """The project-relative path of the shuffle's model directory inside the model shell."""
    pose_snapshot: str
    """The file name of the bundled pose snapshot."""
    detector_snapshot: str | None
    """The file name of the bundled detector snapshot, or None for a model that needs no detector."""
    likelihood_threshold: float
    """The default likelihood below which keypoint positions are masked to NaN when the asset is deployed."""
    source: dict[str, Any]
    """The reproducibility-only provenance of the source project (original config path, project path, snapshot stem)."""
    payload: tuple[PayloadEntry, ...]
    """The per-file checksums of every file in the model shell, verified after extraction."""

    def to_mapping(self) -> dict[str, Any]:
        """Renders the manifest as a plain mapping of built-in types for YAML serialization.

        Returns:
            A mapping whose values are strings, numbers, booleans, lists, and nested mappings only.
        """
        return {
            "format_version": self.format_version,
            "slvt_version": self.slvt_version,
            "dlc_version": self.dlc_version,
            "created": self.created,
            "multianimal": self.multianimal,
            "task": self.task,
            "net_type": self.net_type,
            "scorer": self.scorer,
            "bodyparts": list(self.bodyparts),
            "individuals": list(self.individuals),
            "unique_bodyparts": list(self.unique_bodyparts),
            "with_identity": self.with_identity,
            "shuffle": self.shuffle,
            "training_fraction": self.training_fraction,
            "iteration": self.iteration,
            "config_relpath": self.config_relpath,
            "model_relpath": self.model_relpath,
            "pose_snapshot": self.pose_snapshot,
            "detector_snapshot": self.detector_snapshot,
            "likelihood_threshold": self.likelihood_threshold,
            "source": dict(self.source),
            "payload": [{"path": e.path, "sha256": e.sha256, "size_bytes": e.size_bytes} for e in self.payload],
        }

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "ModelManifest":
        """Builds a manifest from a mapping loaded from YAML, restoring the tuple and payload fields.

        Args:
            mapping: The manifest mapping read from an asset's manifest member.

        Returns:
            The reconstructed manifest.
        """
        payload = tuple(
            PayloadEntry(path=str(e["path"]), sha256=str(e["sha256"]), size_bytes=int(e["size_bytes"]))
            for e in mapping.get("payload", [])
        )
        return cls(
            format_version=int(mapping["format_version"]),
            slvt_version=str(mapping["slvt_version"]),
            dlc_version=str(mapping["dlc_version"]),
            created=str(mapping["created"]),
            multianimal=bool(mapping["multianimal"]),
            task=str(mapping["task"]),
            net_type=str(mapping["net_type"]),
            scorer=str(mapping["scorer"]),
            bodyparts=tuple(mapping.get("bodyparts", [])),
            individuals=tuple(mapping.get("individuals", [])),
            unique_bodyparts=tuple(mapping.get("unique_bodyparts", [])),
            with_identity=bool(mapping["with_identity"]),
            shuffle=int(mapping["shuffle"]),
            training_fraction=float(mapping["training_fraction"]),
            iteration=int(mapping["iteration"]),
            config_relpath=str(mapping["config_relpath"]),
            model_relpath=str(mapping["model_relpath"]),
            pose_snapshot=str(mapping["pose_snapshot"]),
            detector_snapshot=(None if mapping.get("detector_snapshot") is None else str(mapping["detector_snapshot"])),
            likelihood_threshold=float(mapping["likelihood_threshold"]),
            source=dict(mapping.get("source", {})),
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class ModelAsset:
    """Bundles the extracted, verified contents of a model asset for the deployment pipeline to run against."""

    root: Path
    """The directory the asset was extracted into."""
    config_path: Path
    """The path of the project configuration inside the extracted model shell."""
    snapshot_index: int
    """The pose snapshot index deployment inference should use."""
    detector_snapshot_index: int | None
    """The detector snapshot index for a top-down model, or None when the model needs no detector."""
    manifest: ModelManifest
    """The asset's manifest describing its identity and contents."""


@dataclass(frozen=True, slots=True)
class ExportSummary:
    """Captures the outcome of exporting a trained shuffle to a portable model asset for reporting to the caller."""

    asset_path: Path
    """The model asset that was written."""
    asset_sha256: str
    """The SHA-256 hex digest of the written asset."""
    size_bytes: int
    """The size of the written asset in bytes."""
    task: str
    """The pose task the exported model performs."""
    multianimal: bool
    """Determines whether the exported model tracks more than one individual."""
    pose_snapshot: str
    """The file name of the bundled pose snapshot."""
    detector_snapshot: str | None
    """The file name of the bundled detector snapshot, or None."""
    payload_file_count: int
    """The number of files bundled inside the asset's model shell."""

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the export for the CLI.

        Returns:
            A compact description of what was exported and the size of the resulting asset.
        """
        animals = "multi-animal" if self.multianimal else "single-animal"
        megabytes = self.size_bytes / (1024 * 1024)
        detector = f" + {self.detector_snapshot}" if self.detector_snapshot is not None else ""
        return (
            f"exported {animals} {self.task} model ({self.pose_snapshot}{detector}) -> "
            f"{self.asset_path.name} ({megabytes:.1f} MiB)"
        )


def export_model(
    config: str | Path,
    destination: str | Path,
    *,
    shuffle: int = 1,
    snapshot_index: int | None = None,
    detector_snapshot_index: int | None = None,
    training_set_index: int = 0,
    likelihood_threshold: float = 0.0,
    compression: ArchiveCompression = ArchiveCompression.NONE,
    crop: bool = False,
    overwrite: bool = False,
) -> ExportSummary:
    """Serializes a trained DeepLabCut shuffle into a portable, condensed, self-contained model asset.

    The asset is a tar archive holding a minimal DeepLabCut project shell: the project configuration, the shuffle's
    ``pytorch_config.yaml`` and ``test/pose_cfg.yaml``, and exactly one pose snapshot (plus one detector snapshot for
    a top-down model). The shell reproduces just enough of the project layout for the deployment pipeline to run
    DeepLabCut's own inference against it on another machine, with no dependence on the original project's labeled data
    or training datasets. A manifest recording the model's identity and per-file checksums is written as the archive's
    first member.

    Args:
        config: The path of the DeepLabCut project configuration file the trained shuffle belongs to.
        destination: The asset file to write, or a directory to write a name-derived asset into.
        shuffle: The shuffle index to export.
        snapshot_index: The pose snapshot index to bundle, or None to use the project's configured snapshot.
        detector_snapshot_index: The detector snapshot index to bundle for a top-down model, or None for the configured
            or best detector snapshot.
        training_set_index: The training-set fraction index the shuffle was created with.
        likelihood_threshold: The default likelihood below which keypoint positions are masked to NaN when the asset is
            deployed; overridable at deployment time.
        compression: The compression applied to the asset archive; uncompressed by default for fast runtime extraction.
        crop: Determines whether to bundle the project's cropping rectangle, so deployment analyzes the cropped region
            rather than the full frame.
        overwrite: Determines whether to overwrite an existing asset at the resolved path.

    Returns:
        A summary describing the written asset.

    Raises:
        FileExistsError: When the resolved asset path already exists and ``overwrite`` is not set.
        ValueError: When the requested pose snapshot cannot be resolved.
    """
    loader = DLCLoader(config=Path(config), shuffle=shuffle, trainset_index=training_set_index)
    pose_snapshot = _resolve_snapshot(loader=loader, index=snapshot_index, task=loader.pose_task)
    detector_snapshot = None
    if loader.pose_task == Task.TOP_DOWN and loader.model_cfg.get("detector") is not None:
        detector_snapshot = _resolve_snapshot(
            loader=loader, index=detector_snapshot_index, task=Task.DETECT, required=False
        )

    asset_path = _resolve_asset_path(
        destination=Path(destination),
        loader=loader,
        shuffle=shuffle,
        pose_snapshot=pose_snapshot,
        compression=compression,
    )
    if asset_path.exists() and not overwrite:
        message = (
            f"Unable to export the model. An asset already exists at '{asset_path}'; pass overwrite to replace it."
        )
        raise FileExistsError(message)

    with tempfile.TemporaryDirectory(prefix="slvt_export_") as stage:
        model_relpath = loader.model_folder.parent.relative_to(loader.project_root).as_posix()
        train_directory = Path(stage) / _MODEL_DIRECTORY / model_relpath / "train"
        test_directory = Path(stage) / _MODEL_DIRECTORY / model_relpath / "test"
        train_directory.mkdir(parents=True, exist_ok=True)
        test_directory.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src=loader.model_config_path, dst=train_directory / loader.model_config_path.name)
        shutil.copy2(src=pose_snapshot.path, dst=train_directory / pose_snapshot.path.name)
        if detector_snapshot is not None:
            shutil.copy2(src=detector_snapshot.path, dst=train_directory / detector_snapshot.path.name)
        shutil.copy2(src=loader.model_folder.parent / "test" / "pose_cfg.yaml", dst=test_directory / "pose_cfg.yaml")

        shell_config = _build_shell_config(loader=loader, keep_crop=crop)
        _dump_yaml(path=Path(stage) / _CONFIG_RELPATH, mapping=shell_config)

        payload = _checksum_payload(model_root=Path(stage) / _MODEL_DIRECTORY, stage=Path(stage))
        manifest = _build_manifest(
            loader=loader,
            shuffle=shuffle,
            pose_snapshot=pose_snapshot,
            detector_snapshot=detector_snapshot,
            model_relpath=model_relpath,
            likelihood_threshold=likelihood_threshold,
            source_config=Path(config),
            payload=payload,
        )
        _dump_yaml(path=Path(stage) / _MANIFEST_FILENAME, mapping=manifest.to_mapping())

        asset_path.parent.mkdir(parents=True, exist_ok=True)
        pack_directory_to_tar(source_root=Path(stage), tar_path=asset_path, mode=compression.tar_mode)

    return ExportSummary(
        asset_path=asset_path,
        asset_sha256=compute_sha256(asset_path),
        size_bytes=asset_path.stat().st_size,
        task=manifest.task,
        multianimal=manifest.multianimal,
        pose_snapshot=manifest.pose_snapshot,
        detector_snapshot=manifest.detector_snapshot,
        payload_file_count=len(payload),
    )


def load_model_asset(asset: str | Path, scratch_directory: str | Path) -> ModelAsset:
    """Extracts a model asset into a scratch directory and verifies its contents against its manifest.

    The manifest is read from the archive's first member and its layout version is checked before anything is unpacked.
    The archive is then extracted with the safe-extraction filter, and every bundled file's checksum is verified against
    the manifest, so a corrupt or tampered asset is refused before inference runs.

    Args:
        asset: The path of the model asset to load.
        scratch_directory: An existing, writable directory to extract the asset into; the caller owns its cleanup.

    Returns:
        The extracted, verified asset with the path of its project configuration and its snapshot indices.

    Raises:
        ValueError: When the asset's layout version is unsupported or a bundled file fails its checksum.
    """
    asset = Path(asset)
    scratch_directory = Path(scratch_directory)

    manifest = ModelManifest.from_mapping(_load_yaml_bytes(read_tar_member(asset, _MANIFEST_FILENAME)))
    if manifest.format_version != _FORMAT_VERSION:
        message = (
            f"Unable to load the model asset '{asset}'. It uses layout version {manifest.format_version}, but this "
            f"version of sollertia-video-tracking reads layout version {_FORMAT_VERSION}."
        )
        raise ValueError(message)

    extract_tar(asset, scratch_directory)
    for entry in manifest.payload:
        actual = compute_sha256(scratch_directory / entry.path)
        if actual != entry.sha256:
            message = (
                f"Unable to load the model asset '{asset}'. The bundled file '{entry.path}' failed its integrity "
                f"check, so the asset is corrupt or was modified after export."
            )
            raise ValueError(message)

    detector_index = _SNAPSHOT_INDEX if manifest.detector_snapshot is not None else None
    return ModelAsset(
        root=scratch_directory,
        config_path=scratch_directory / manifest.config_relpath,
        snapshot_index=_SNAPSHOT_INDEX,
        detector_snapshot_index=detector_index,
        manifest=manifest,
    )


def _resolve_snapshot(loader: DLCLoader, index: int | None, task: Task, *, required: bool = True) -> Any:
    """Resolves the single snapshot to bundle, falling back to the configured or best snapshot when none is requested.

    Args:
        loader: The loader holding the model directory the snapshots live in.
        index: The requested snapshot index, or None to use the project's configured snapshot.
        task: The task whose snapshots to search (pose or detector).
        required: Determines whether to raise when no snapshot is found; when False, returns None instead.

    Returns:
        The resolved DeepLabCut snapshot, or None when none is found and ``required`` is False.

    Raises:
        ValueError: When no snapshot can be resolved and ``required`` is True.
    """
    resolved_index: int | str
    if index is not None:
        resolved_index = index
    else:
        configured = loader.project_cfg.get("snapshotindex", "best")
        resolved_index = "best" if configured == "all" else configured
    try:
        snapshots = get_model_snapshots(index=resolved_index, model_folder=loader.model_folder, task=task)
    except (ValueError, IndexError):
        if resolved_index == "best":
            try:
                snapshots = get_model_snapshots(index=-1, model_folder=loader.model_folder, task=task)
            except (ValueError, IndexError):
                snapshots = []
        else:
            snapshots = []
    if not snapshots:
        if required:
            message = (
                f"Unable to export the model. No '{task.value}' snapshot with index {resolved_index!r} was found in "
                f"'{loader.model_folder}'."
            )
            raise ValueError(message)
        return None
    return snapshots[0]


def _resolve_asset_path(
    destination: Path, loader: DLCLoader, shuffle: int, pose_snapshot: Any, compression: ArchiveCompression
) -> Path:
    """Resolves the asset file to write, deriving a descriptive name when the destination is a directory.

    Args:
        destination: The asset file to write, or a directory to write a name-derived asset into.
        loader: The loader for the exported shuffle, providing the project name and date.
        shuffle: The exported shuffle index.
        pose_snapshot: The resolved pose snapshot, whose stem names the derived asset.
        compression: The compression applied, which appends the matching suffix to a derived name.

    Returns:
        The full path of the asset file to write, including the asset and compression suffixes.
    """
    is_asset_file = destination.suffix == _ASSET_SUFFIX or destination.suffixes[-2:-1] == [_ASSET_SUFFIX]
    if is_asset_file:
        return destination
    task = loader.project_cfg["Task"]
    date = loader.project_cfg["date"]
    name = f"{task}{date}-shuffle{shuffle}-{pose_snapshot.path.stem}{_ASSET_SUFFIX}{compression.file_suffix}"
    return destination / name


def _build_shell_config(loader: DLCLoader, *, keep_crop: bool) -> dict[str, Any]:
    """Builds the relocated project configuration for the model shell from the source project configuration.

    The source configuration is copied and the fields that would tie it to the original machine or ambiguously select a
    snapshot are rewritten: the project path is neutralized, the snapshot indices are pinned to the single bundled
    snapshot, the engine is fixed to PyTorch, the video registry is cleared, and the training fraction is reduced to the
    exported shuffle's own fraction so it always resolves at index zero. Cropping is disabled unless it is bundled.

    Args:
        loader: The loader holding the source project configuration.
        keep_crop: Determines whether to keep the project's cropping rectangle rather than disabling cropping.

    Returns:
        The rewritten project configuration to write into the model shell.
    """
    config = copy.deepcopy(loader.project_cfg)
    config["project_path"] = "."
    config["snapshotindex"] = _SNAPSHOT_INDEX
    config["detector_snapshotindex"] = _SNAPSHOT_INDEX
    config["engine"] = "pytorch"
    config["video_sets"] = {}
    config["TrainingFraction"] = [loader.train_fraction]
    config.pop("scorer", None)
    if not keep_crop:
        config["cropping"] = False
    return config


def _checksum_payload(model_root: Path, stage: Path) -> tuple[PayloadEntry, ...]:
    """Computes a checksum entry for every file in the model shell, keyed by its archive-relative path.

    Args:
        model_root: The staged model-shell directory whose files are checksummed.
        stage: The staging root the archive-relative paths are computed against.

    Returns:
        The per-file checksum entries, ordered by archive-relative path.
    """
    files = sorted((path for path in model_root.rglob("*") if path.is_file()), key=lambda path: path.as_posix())
    return tuple(
        PayloadEntry(
            path=path.relative_to(stage).as_posix(),
            sha256=compute_sha256(path),
            size_bytes=path.stat().st_size,
        )
        for path in files
    )


def _build_manifest(
    loader: DLCLoader,
    shuffle: int,
    pose_snapshot: Any,
    detector_snapshot: Any,
    model_relpath: str,
    likelihood_threshold: float,
    source_config: Path,
    payload: tuple[PayloadEntry, ...],
) -> ModelManifest:
    """Assembles the asset manifest from the loaded model configuration and the resolved snapshots.

    Args:
        loader: The loader holding the source project and model configuration.
        shuffle: The exported shuffle index.
        pose_snapshot: The resolved pose snapshot bundled in the asset.
        detector_snapshot: The resolved detector snapshot bundled in the asset, or None.
        model_relpath: The project-relative path of the shuffle's model directory inside the shell.
        likelihood_threshold: The default likelihood floor baked into the asset.
        source_config: The original project configuration path, kept as provenance.
        payload: The per-file checksums of the model shell.

    Returns:
        The assembled manifest.
    """
    metadata = loader.model_cfg["metadata"]
    return ModelManifest(
        format_version=_FORMAT_VERSION,
        slvt_version=resolve_slvt_version(),
        dlc_version=deeplabcut.__version__,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        multianimal=bool(loader.project_cfg.get("multianimalproject", False)),
        task=loader.pose_task.value,
        net_type=str(loader.model_cfg["net_type"]),
        scorer=loader.scorer(pose_snapshot, detector_snapshot),
        bodyparts=tuple(metadata["bodyparts"]),
        individuals=tuple(metadata["individuals"]),
        unique_bodyparts=tuple(metadata["unique_bodyparts"]),
        with_identity=bool(metadata["with_identity"]),
        shuffle=shuffle,
        training_fraction=float(loader.train_fraction),
        iteration=int(loader.project_cfg["iteration"]),
        config_relpath=_CONFIG_RELPATH,
        model_relpath=model_relpath,
        pose_snapshot=pose_snapshot.path.name,
        detector_snapshot=(detector_snapshot.path.name if detector_snapshot is not None else None),
        likelihood_threshold=float(likelihood_threshold),
        source={
            "config": str(source_config),
            "project_path": str(loader.project_root),
            "pose_snapshot_stem": pose_snapshot.path.stem,
        },
        payload=payload,
    )


def _dump_yaml(path: Path, mapping: dict[str, Any]) -> None:
    """Writes a mapping to a YAML file in block style.

    Args:
        path: The path of the YAML file to write.
        mapping: The mapping to serialize.
    """
    yaml = YAML()
    yaml.default_flow_style = False
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(data=mapping, stream=stream)


def _load_yaml_bytes(raw: bytes) -> dict[str, Any]:
    """Parses a YAML mapping from raw bytes.

    Args:
        raw: The raw YAML bytes to parse.

    Returns:
        The parsed mapping.
    """
    return dict(YAML(typ="safe").load(raw.decode("utf-8")))
