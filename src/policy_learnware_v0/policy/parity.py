"""Golden input/output parity checks for restored native policies."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bundle import PolicyBundleMetadata, validate_bundle
from .loader import FrozenPolicy


@dataclass(frozen=True)
class ParityReport:
    passed: bool
    raw_checked: bool
    raw_max_abs_error: float | None
    environment_max_abs_error: float
    atol: float
    rtol: float
    sample_count: int


def _restore_key(key_data: np.ndarray) -> Any:
    try:
        jax = importlib.import_module("jax")
        wrap = getattr(jax.random, "wrap_key_data", None)
        return wrap(key_data) if wrap is not None else key_data
    except ImportError:
        # Useful for CPU-only structural smoke tests with an injected fake
        # policy.  A real upstream policy will itself require JAX.
        return key_data


def verify_golden_parity(
    policy: FrozenPolicy,
    bundle: str | Path | PolicyBundleMetadata,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
    expected_fpo_commit: str | None = None,
    expected_runtime_digest: str | None = None,
) -> ParityReport:
    """Replay exported observations and key, comparing raw and env actions."""

    metadata = (
        bundle
        if isinstance(bundle, PolicyBundleMetadata)
        else validate_bundle(
            bundle,
            expected_fpo_commit=expected_fpo_commit,
            expected_runtime_digest=expected_runtime_digest,
        )
    )
    with np.load(metadata.bundle_dir / "golden_io.npz", allow_pickle=False) as archive:
        observation = np.asarray(archive["observation"])
        key = _restore_key(np.asarray(archive["prng_key_data"]))
        expected_raw = np.asarray(archive["raw_action"])
        expected_environment = np.asarray(archive["environment_action"])

    raw_checked = hasattr(policy, "act_raw")
    if raw_checked:
        actual_raw, _ = policy.act_raw(observation, key, deterministic=True)  # type: ignore[attr-defined]
        actual_raw_array = np.asarray(actual_raw)
        actual_environment = np.tanh(actual_raw_array)
        raw_error = float(np.max(np.abs(actual_raw_array - expected_raw), initial=0.0))
        raw_passed = bool(np.allclose(actual_raw_array, expected_raw, atol=atol, rtol=rtol))
    else:
        actual_environment, _ = policy.act(observation, key, deterministic=True)
        actual_environment = np.asarray(actual_environment)
        raw_error = None
        raw_passed = True
    environment_error = float(
        np.max(np.abs(np.asarray(actual_environment) - expected_environment), initial=0.0)
    )
    environment_passed = bool(
        np.allclose(actual_environment, expected_environment, atol=atol, rtol=rtol)
    )
    return ParityReport(
        passed=raw_passed and environment_passed,
        raw_checked=raw_checked,
        raw_max_abs_error=raw_error,
        environment_max_abs_error=environment_error,
        atol=float(atol),
        rtol=float(rtol),
        sample_count=int(observation.shape[0]),
    )
