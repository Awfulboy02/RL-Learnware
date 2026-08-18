"""Canonical hashing helpers used by every persisted artifact."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import numpy as np


class CanonicalizationError(TypeError):
    """Raised when a value has no unambiguous canonical JSON encoding."""


def canonicalize(value: Any) -> Any:
    """Convert supported Python/NumPy values to canonical JSON primitives.

    Mappings are sorted at serialization time.  NaN and infinities are
    rejected because otherwise two parsers can disagree about the hash.
    """

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, np.ndarray):
        return canonicalize(value.tolist())
    if isinstance(value, np.generic):
        return canonicalize(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("non-finite floats are not canonical JSON")
        # JSON has a single numeric type; normalise negative zero.
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("canonical JSON mapping keys must be strings")
            result[key] = canonicalize(item)
        return result
    if isinstance(value, (tuple, list)):
        return [canonicalize(item) for item in value]
    raise CanonicalizationError(
        f"unsupported canonical JSON value: {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_stream(handle: BinaryIO, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while True:
        block = handle.read(chunk_size)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return sha256_stream(handle)


def sha256_ndarrays(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash named arrays including exact dtype, shape, order-independent names."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        if not isinstance(name, str):
            raise TypeError("array names must be strings")
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise TypeError(f"object array {name!r} cannot be hashed safely")
        contiguous = np.ascontiguousarray(array)
        header = canonical_json_bytes(
            {
                "name": name,
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
            }
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        payload = contiguous.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
