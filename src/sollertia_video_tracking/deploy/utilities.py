"""Provides the shared deployment helpers for checksums, tar packaging, safe extraction, and version lookup."""

import hashlib
from pathlib import Path
import tarfile
from importlib.metadata import PackageNotFoundError, version

_HASH_CHUNK_BYTES: int = 1024 * 1024
"""The number of bytes read per iteration when streaming a file through the hash, bounding peak memory."""

_DISTRIBUTION_NAME: str = "sollertia-video-tracking"
"""The installed distribution name whose version is recorded in exported assets and prediction provenance."""


def resolve_slvt_version() -> str:
    """Returns the installed sollertia-video-tracking version, or a placeholder when it cannot be resolved.

    Returns:
        The distribution version string, or ``"unknown"`` when the package metadata is unavailable.
    """
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return "unknown"


def compute_sha256(path: str | Path) -> str:
    """Computes the SHA-256 hex digest of a file by streaming it through the hash in fixed-size chunks.

    Args:
        path: The path of the file to hash.

    Returns:
        The lowercase hexadecimal SHA-256 digest of the file's contents.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_directory_to_tar(source_root: str | Path, tar_path: str | Path, *, mode: str = "w") -> None:
    """Packs every file under a directory into a tar archive, writing the shallowest files first.

    Files are added in order of increasing path depth, so a file at the archive root is written before any file
    nested under a subdirectory. This lets a reader recover a root-level manifest by reading only the first member
    rather than scanning the whole archive.

    Args:
        source_root: The directory whose files are packed; each file is stored under its path relative to this root.
        tar_path: The path of the tar archive to write.
        mode: The ``tarfile`` write mode selecting the compression, one of ``"w"`` (uncompressed), ``"w:gz"``, or
            ``"w:xz"``.
    """
    source_root = Path(source_root)
    files = sorted(
        (path for path in source_root.rglob("*") if path.is_file()),
        key=lambda path: (len(path.relative_to(source_root).parts), path.relative_to(source_root).as_posix()),
    )
    # The mode is one of the write literals tarfile.open accepts, but it arrives as a plain string, which its
    # per-mode overloads cannot narrow; the value is controlled by the caller, so the call is safe.
    with tarfile.open(name=Path(tar_path), mode=mode) as archive:  # type: ignore[call-overload]
        for path in files:
            archive.add(name=path, arcname=path.relative_to(source_root).as_posix())


def extract_tar(tar_path: str | Path, destination: str | Path) -> None:
    """Extracts every member of a tar archive into a destination directory, rejecting unsafe members.

    Extraction uses the ``"data"`` filter, which refuses members whose paths escape the destination or that carry
    symlink, device, or absolute-path payloads, so a hostile archive cannot write outside the destination.

    Args:
        tar_path: The path of the tar archive to extract.
        destination: The directory members are extracted into.
    """
    with tarfile.open(name=Path(tar_path), mode="r:*") as archive:
        archive.extractall(path=Path(destination), filter="data")


def read_tar_member(tar_path: str | Path, member: str) -> bytes:
    """Reads a single member's bytes from a tar archive without extracting the rest of it.

    Args:
        tar_path: The path of the tar archive to read from.
        member: The archive-relative name of the member to read.

    Returns:
        The raw bytes of the requested member.

    Raises:
        KeyError: When the archive holds no member with the requested name.
    """
    with tarfile.open(name=Path(tar_path), mode="r:*") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            message = f"Unable to read '{member}' from '{tar_path}'. The archive holds no such file member."
            raise KeyError(message)
        with extracted:
            return extracted.read()
