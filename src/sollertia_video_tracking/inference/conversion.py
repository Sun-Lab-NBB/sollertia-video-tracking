"""Provides conversion of DeepLabCut prediction HDF5 files into flat, wide polars feather files for the Sollertia stack.

DeepLabCut writes predictions as a pandas HDF5 file whose columns are a ``scorer / [individuals] / bodyparts / coords``
MultiIndex and whose row index is the frame number. The rest of the Sollertia stack runs on numpy 2 / Python 3.14 and
consumes uncompressed Apache Arrow "feather" files written by polars. This module renders that prediction table into
a wide, snake-cased feather: one row per frame (``frame``) and, per keypoint, ``[<individual>_]<bodypart>_x``, ``_y``,
and ``_likelihood`` columns. It is a direct, project-agnostic transcription of DeepLabCut's own output ("DeepLabCut, but
in polars"); domain-specific derived quantities are left to downstream consumers. A YAML provenance sidecar records the
source file and the conversion parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from ruamel.yaml import YAML

if TYPE_CHECKING:
    from numpy.typing import NDArray

_COORDINATES: tuple[str, ...] = ("x", "y", "likelihood")
"""The per-keypoint coordinate channels DeepLabCut stores, rendered as column suffixes in this fixed order."""

_POSITION_COORDINATES: tuple[str, ...] = ("x", "y")
"""The coordinate channels masked to NaN when a keypoint's likelihood is below the confidence threshold."""


@dataclass(frozen=True, slots=True)
class ConversionSummary:
    """Captures the outcome of converting one DeepLabCut prediction file to a feather file.

    Notes:
        The summary is built after the feather is written. ``keypoints`` lists the flattened per-keypoint prefixes in
        file order (for example ``"snout"`` for a single animal, or ``"mouse1_snout"`` for a tracked one), and
        ``frame_count`` is the number of frames (rows) written.
    """

    h5_path: Path
    """The DeepLabCut prediction file that was converted."""
    feather_path: Path
    """The feather file that was written."""
    provenance_path: Path | None
    """The provenance sidecar that was written, or None when provenance was not requested."""
    frame_count: int
    """The number of frames (rows) written."""
    keypoints: tuple[str, ...]
    """The flattened per-keypoint column prefixes, in file order."""
    likelihood_threshold: float
    """The likelihood threshold below which keypoint positions were masked to NaN."""

    def describe(self) -> str:
        """Builds a one-line human-readable summary of the conversion for the CLI.

        Returns:
            A compact description of what was converted and where it was written.
        """
        return f"converted {self.frame_count} frames x {len(self.keypoints)} keypoints -> {self.feather_path.name}"


def convert_predictions_to_feather(
    h5_path: str | Path,
    feather_path: str | Path,
    *,
    likelihood_threshold: float = 0.0,
    write_provenance: bool = True,
) -> ConversionSummary:
    """Converts a DeepLabCut prediction HDF5 file into a wide polars feather file with an optional provenance sidecar.

    Args:
        h5_path: The path to the DeepLabCut prediction ``.h5`` file to convert.
        feather_path: The path of the feather file to write.
        likelihood_threshold: The likelihood below which a keypoint's ``x`` and ``y`` are masked to NaN; 0.0 keeps
            every prediction.
        write_provenance: Determines whether to write a ``<feather-stem>_provenance.yaml`` sidecar next to the feather.

    Returns:
        A summary describing the converted file.

    Raises:
        ValueError: When the HDF5 file holds no prediction table.
    """
    h5_path = Path(h5_path)
    feather_path = Path(feather_path)

    predictions = _read_prediction_dataframe(h5_path)
    frame_count = len(predictions)
    keypoints, columns = _flatten_predictions(predictions, likelihood_threshold=likelihood_threshold)

    data: dict[str, pl.Series] = {
        "frame": pl.Series(name="frame", values=np.asarray(predictions.index, dtype=np.uint64)),
    }
    for name, values in columns.items():
        data[name] = pl.Series(name=name, values=values, dtype=pl.Float32)

    feather_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data).write_ipc(file=feather_path, compression="uncompressed")

    provenance_path: Path | None = None
    if write_provenance:
        provenance_path = feather_path.with_name(f"{feather_path.stem}_provenance.yaml")
        _write_provenance(
            provenance_path,
            h5_path=h5_path,
            feather_path=feather_path,
            frame_count=frame_count,
            keypoints=keypoints,
            likelihood_threshold=likelihood_threshold,
        )

    return ConversionSummary(
        h5_path=h5_path,
        feather_path=feather_path,
        provenance_path=provenance_path,
        frame_count=frame_count,
        keypoints=keypoints,
        likelihood_threshold=likelihood_threshold,
    )


def _read_prediction_dataframe(h5_path: Path) -> pd.DataFrame:
    """Reads the single prediction table from a DeepLabCut HDF5 file, whatever key it is stored under.

    DeepLabCut stores pose predictions under the ``df_with_missing`` key and tracked multi-animal predictions under
    ``tracks``; both are read the same way here by taking the file's only stored object.

    Args:
        h5_path: The path to the DeepLabCut prediction ``.h5`` file.

    Returns:
        The prediction table with its ``scorer / [individuals] / bodyparts / coords`` column MultiIndex.

    Raises:
        ValueError: When the file holds no stored table.
    """
    with pd.HDFStore(str(h5_path), mode="r") as store:
        keys = store.keys()
        if not keys:
            message = (
                f"Unable to convert the predictions in '{h5_path}'. Expected an HDF5 file holding a DeepLabCut "
                f"prediction table, but it stores no objects."
            )
            raise ValueError(message)
        # pandas-stubs types HDFStore.__getitem__ as DataFrame | Series; a DeepLabCut prediction table is always a
        # DataFrame.
        return store[keys[0]]  # type: ignore[return-value]


def _flatten_predictions(
    predictions: pd.DataFrame,
    *,
    likelihood_threshold: float,
) -> tuple[tuple[str, ...], dict[str, NDArray[np.float32]]]:
    """Flattens the prediction MultiIndex columns into snake-cased per-keypoint float columns.

    The leading ``scorer`` column level is dropped and the remaining levels above the coordinate are joined with
    underscores to form each keypoint prefix, so a single-animal ``(scorer, snout, x)`` becomes ``snout_x`` and a
    multi-animal ``(scorer, mouse1, snout, x)`` becomes ``mouse1_snout_x``. Positions below the likelihood threshold
    are masked to NaN while the likelihood channel is preserved.

    Args:
        predictions: The DeepLabCut prediction table.
        likelihood_threshold: The likelihood below which a keypoint's ``x`` and ``y`` are masked to NaN.

    Returns:
        A tuple of the ordered keypoint prefixes and a mapping of flat column name to float32 values.
    """
    groups: dict[str, dict[str, NDArray[np.float32]]] = {}
    keypoint_order: list[str] = []
    for column in predictions.columns:
        # Iterating a MultiIndex yields per-column tuples, but pandas-stubs types the elements as str; the unpack is
        # valid at runtime.
        *prefix, coordinate = column  # type: ignore[str-unpack]
        keypoint = "_".join(str(level) for level in prefix[1:])
        if keypoint not in groups:
            groups[keypoint] = {}
            keypoint_order.append(keypoint)
        groups[keypoint][coordinate] = predictions[column].to_numpy(dtype=np.float32)

    columns: dict[str, NDArray[np.float32]] = {}
    for keypoint in keypoint_order:
        channels = groups[keypoint]
        likelihood = channels.get("likelihood")
        mask = likelihood < likelihood_threshold if likelihood is not None and likelihood_threshold > 0.0 else None
        for coordinate in _COORDINATES:
            if coordinate not in channels:
                continue
            values = channels[coordinate]
            if mask is not None and coordinate in _POSITION_COORDINATES:
                values = values.copy()
                values[mask] = np.nan
            columns[f"{keypoint}_{coordinate}"] = values

    return tuple(keypoint_order), columns


def _write_provenance(
    provenance_path: Path,
    *,
    h5_path: Path,
    feather_path: Path,
    frame_count: int,
    keypoints: tuple[str, ...],
    likelihood_threshold: float,
) -> None:
    """Writes a YAML provenance sidecar recording the source file and conversion parameters.

    Args:
        provenance_path: The path of the provenance sidecar to write.
        h5_path: The source DeepLabCut prediction file.
        feather_path: The feather file that was written.
        frame_count: The number of frames written.
        keypoints: The flattened per-keypoint prefixes.
        likelihood_threshold: The likelihood threshold applied during conversion.
    """
    record: dict[str, str | int | float | list[str]] = {
        "source_h5": str(h5_path),
        "feather": str(feather_path.name),
        "frame_count": int(frame_count),
        "keypoints": list(keypoints),
        "coordinates": list(_COORDINATES),
        "likelihood_threshold": float(likelihood_threshold),
    }
    yaml = YAML()
    yaml.default_flow_style = False
    with provenance_path.open("w", encoding="utf-8") as stream:
        yaml.dump(data=record, stream=stream)
