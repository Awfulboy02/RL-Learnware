"""Safe artifact IO with canonical JSON and deterministic NPZ output."""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .hashing import canonical_json_bytes, sha256_file


class ArtifactExistsError(FileExistsError):
    """Raised when immutable artifact publication would overwrite a file."""


class DigestMismatchError(ValueError):
    """Raised when a persisted artifact does not match its expected digest."""


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - uncommon platform limitation
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    path: str | Path,
    payload: bytes,
    *,
    overwrite: bool = False,
) -> str:
    """Publish bytes durably, without racing immutable writers.

    Immutable publication uses a hard-link from a same-directory temporary
    file.  Creating that final directory entry is atomic and fails when the
    destination already exists; unlike a check followed by ``os.replace``, it
    cannot overwrite a winner from another process.  Explicit ``overwrite``
    retains replacement semantics for the few mutable/debug call sites that
    request it deliberately.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise ArtifactExistsError(f"refusing to overwrite artifact: {destination}")

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise ArtifactExistsError(
                    f"refusing to overwrite artifact: {destination}"
                ) from error
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(destination)


def atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    overwrite: bool = False,
) -> str:
    return atomic_write_bytes(
        path, canonical_json_bytes(value) + b"\n", overwrite=overwrite
    )


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Encode arrays as an NPZ whose bytes do not contain wall-clock metadata."""

    destination = io.BytesIO()
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for name in sorted(arrays):
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                raise ValueError(f"invalid NPZ member name: {name!r}")
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise TypeError(f"object array {name!r} cannot be persisted")
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue())
    return destination.getvalue()


def atomic_write_npz(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    overwrite: bool = False,
) -> str:
    return atomic_write_bytes(
        path, deterministic_npz_bytes(arrays), overwrite=overwrite
    )


def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def verify_file_digest(path: str | Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise DigestMismatchError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
