"""Validation for immutable PPO/FPO policy-only bundles.

The exporter lives in the read-only reproduction repository.  This module is
deliberately a consumer: it verifies the manifest and the structural contract
before any untrusted array is handed to a policy runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


BUNDLE_SCHEMA = "policy-learnware.policy-bundle.v0"
REQUIRED_PAYLOADS = frozenset(
    {"actor.npz", "obs_stats.npz", "golden_io.npz", "policy_spec.json", "provenance.json"}
)
_OUTER_DIRECTORY = re.compile(r"^outer_(\d{6})$")


class BundleValidationError(ValueError):
    """A bundle is incomplete, inconsistent, or fails an integrity check."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise BundleValidationError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BundleValidationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise BundleValidationError(f"{label} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise BundleValidationError(f"{label} must be an integer") from error
    if value != result:
        raise BundleValidationError(f"{label} must be an integer")
    return result


def _expect_equal(label: str, *values: Any) -> Any:
    first = values[0]
    if any(item != first for item in values[1:]):
        raise BundleValidationError(f"inconsistent {label}: {values!r}")
    return first


def _validate_npz_structure(bundle_dir: Path, spec: Mapping[str, Any]) -> None:
    try:
        with np.load(bundle_dir / "actor.npz", allow_pickle=False) as actor:
            kernel_names = sorted(name for name in actor.files if name.endswith("_kernel"))
            bias_names = sorted(name for name in actor.files if name.endswith("_bias"))
            if not kernel_names or len(kernel_names) != len(bias_names):
                raise BundleValidationError("actor.npz has incomplete kernel/bias pairs")
            architecture: list[int] = []
            previous_output: int | None = None
            for index, (kernel_name, bias_name) in enumerate(zip(kernel_names, bias_names)):
                expected_kernel = f"layer_{index:02d}_kernel"
                expected_bias = f"layer_{index:02d}_bias"
                if (kernel_name, bias_name) != (expected_kernel, expected_bias):
                    raise BundleValidationError("actor layers are not contiguous from layer_00")
                kernel = np.asarray(actor[kernel_name])
                bias = np.asarray(actor[bias_name])
                if kernel.dtype.kind != "f" or bias.dtype.kind != "f":
                    raise BundleValidationError("actor parameters must be floating point")
                if not np.all(np.isfinite(kernel)) or not np.all(np.isfinite(bias)):
                    raise BundleValidationError("actor parameters contain non-finite values")
                if kernel.ndim != 2 or bias.ndim != 1 or kernel.shape[1] != bias.shape[0]:
                    raise BundleValidationError(f"invalid actor layer shapes at layer {index}")
                if previous_output is not None and kernel.shape[0] != previous_output:
                    raise BundleValidationError(f"actor layer {index} input width is inconsistent")
                if index == 0:
                    architecture.append(int(kernel.shape[0]))
                architecture.append(int(kernel.shape[1]))
                previous_output = int(kernel.shape[1])
            declared_architecture = spec.get("actor_layer_sizes")
            if declared_architecture is not None and list(declared_architecture) != architecture:
                raise BundleValidationError(
                    f"actor architecture {architecture} != policy_spec {declared_architecture}"
                )

        observation_dim = _strict_int(spec.get("observation_size"), "observation_size")
        action_dim = _strict_int(spec.get("action_size"), "action_size")
        if observation_dim <= 0 or action_dim <= 0:
            raise BundleValidationError("native observation/action dimensions must be positive")
        algorithm = str(spec.get("algorithm", ""))
        if algorithm == "ppo":
            expected_input, expected_output = observation_dim, 2 * action_dim
        elif algorithm == "fpo":
            inference = spec.get("inference")
            if not isinstance(inference, Mapping):
                raise BundleValidationError("FPO policy_spec.inference must be an object")
            timestep_dim = _strict_int(
                inference.get("timestep_embed_dim"), "inference.timestep_embed_dim"
            )
            expected_input = observation_dim + action_dim + timestep_dim
            expected_output = action_dim
        else:
            raise BundleValidationError(f"unsupported algorithm: {algorithm!r}")
        if architecture[0] != expected_input or architecture[-1] != expected_output:
            raise BundleValidationError(
                f"{algorithm.upper()} actor endpoints {architecture[0]}->{architecture[-1]} "
                f"do not match expected {expected_input}->{expected_output}"
            )
        algorithm = spec.get("algorithm")
        expected_output = 2 * action_dim if algorithm == "ppo" else action_dim
        if architecture[-1] != expected_output:
            raise BundleValidationError(
                f"actor output width {architecture[-1]} != expected {expected_output} for {algorithm}"
            )

        with np.load(bundle_dir / "obs_stats.npz", allow_pickle=False) as stats:
            missing_stats = {"count", "mean", "std", "var_sum"}.difference(stats.files)
            if missing_stats:
                raise BundleValidationError(f"obs_stats.npz misses {sorted(missing_stats)}")
            for name in ("mean", "std"):
                array = np.asarray(stats[name])
                if array.shape != (observation_dim,):
                    raise BundleValidationError(
                        f"obs_stats {name} shape {array.shape} != ({observation_dim},)"
                    )
            if np.asarray(stats["count"]).shape != ():
                raise BundleValidationError("obs_stats count must be scalar")
            if np.asarray(stats["var_sum"]).shape != (observation_dim,):
                raise BundleValidationError("obs_stats var_sum has the wrong shape")
            if not np.all(np.isfinite(stats["mean"])) or not np.all(np.isfinite(stats["std"])):
                raise BundleValidationError("obs_stats contains non-finite values")
            if np.any(np.asarray(stats["std"]) <= 0):
                raise BundleValidationError("obs_stats std must be positive")

        with np.load(bundle_dir / "golden_io.npz", allow_pickle=False) as golden:
            required = {"observation", "prng_key_data", "raw_action", "environment_action"}
            missing = required.difference(golden.files)
            if missing:
                raise BundleValidationError(f"golden_io.npz misses {sorted(missing)}")
            observation = np.asarray(golden["observation"])
            raw_action = np.asarray(golden["raw_action"])
            environment_action = np.asarray(golden["environment_action"])
            if observation.ndim != 2 or observation.shape[1] != observation_dim:
                raise BundleValidationError("golden observation has the wrong native shape")
            if observation.shape[0] != 8:
                raise BundleValidationError("golden_io must contain exactly 8 samples")
            if raw_action.shape != environment_action.shape:
                raise BundleValidationError("golden raw/environment action shapes differ")
            if raw_action.ndim != 2 or raw_action.shape != (observation.shape[0], action_dim):
                raise BundleValidationError("golden action has the wrong native shape")
            arrays = (observation, raw_action, environment_action)
            if any(not np.all(np.isfinite(array)) for array in arrays):
                raise BundleValidationError("golden_io contains non-finite values")
            key_data = np.asarray(golden["prng_key_data"])
            if key_data.shape != (2,) or key_data.dtype != np.dtype(np.uint32):
                raise BundleValidationError(
                    "golden prng_key_data must have shape (2,) and dtype uint32"
                )
            # Export uses JAX's float32 tanh while this structural check uses
            # NumPy.  Their correctly-rounded results can differ by a few ULPs.
            if not np.allclose(environment_action, np.tanh(raw_action), atol=1e-6, rtol=1e-6):
                raise BundleValidationError("golden environment action is not tanh(raw_action)")
    except (OSError, ValueError) as error:
        if isinstance(error, BundleValidationError):
            raise
        raise BundleValidationError(f"invalid NPZ payload in {bundle_dir}: {error}") from error


@dataclass(frozen=True)
class PolicyBundleMetadata:
    """Validated private metadata; never expose this object to the selector."""

    bundle_dir: Path
    bundle_digest: str
    task: str
    algorithm: str
    training_seed: int
    outer_iteration: int
    environment_steps: int
    observation_dim: int
    action_dim: int
    manifest: Mapping[str, Any] = field(repr=False, compare=False)
    policy_spec: Mapping[str, Any] = field(repr=False, compare=False)
    provenance: Mapping[str, Any] = field(repr=False, compare=False)


def validate_bundle(
    bundle_dir: str | Path,
    *,
    expected_task: str | None = None,
    expected_algorithm: str | None = None,
    expected_seed: int | None = None,
    expected_outer: int | None = None,
    expected_environment_steps: int | None = None,
    expected_fpo_commit: str | None = None,
    expected_runtime_digest: str | None = None,
    runtime_only: bool = False,
) -> PolicyBundleMetadata:
    """Validate one exported policy bundle.

    ``runtime_only`` keeps structural, checksum, shape, and finite-value checks
    while treating historical clean-tree/external-authority provenance as
    metadata.  It is the appropriate boundary for a real rollout.
    """

    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise BundleValidationError(f"bundle directory does not exist: {bundle_dir}")
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest = _load_object(manifest_path)
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise BundleValidationError(f"unsupported bundle schema: {manifest.get('schema')!r}")
    if manifest.get("complete") is not True:
        raise BundleValidationError("bundle manifest is not marked complete")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise BundleValidationError("bundle manifest files must be an object")
    missing = REQUIRED_PAYLOADS.difference(files)
    if missing:
        raise BundleValidationError(f"bundle manifest misses payloads {sorted(missing)}")
    for filename, record in files.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise BundleValidationError(f"unsafe manifest filename: {filename!r}")
        if not isinstance(record, dict):
            raise BundleValidationError(f"invalid manifest record for {filename}")
        payload_path = bundle_dir / filename
        if not payload_path.is_file():
            raise BundleValidationError(f"manifest payload is missing: {filename}")
        expected_bytes = _strict_int(record.get("bytes"), f"{filename}.bytes")
        if payload_path.stat().st_size != expected_bytes:
            raise BundleValidationError(f"size mismatch for {filename}")
        actual_sha = _sha256(payload_path)
        if actual_sha != record.get("sha256"):
            raise BundleValidationError(f"SHA-256 mismatch for {filename}")

    spec = _load_object(bundle_dir / "policy_spec.json")
    provenance = _load_object(bundle_dir / "provenance.json")
    if spec.get("schema") != BUNDLE_SCHEMA or provenance.get("schema") != BUNDLE_SCHEMA:
        raise BundleValidationError("payload schema does not match bundle schema")
    if spec.get("actor_weights_file") != "actor.npz":
        raise BundleValidationError("policy_spec actor_weights_file is not actor.npz")
    if spec.get("golden_parity_file") != "golden_io.npz":
        raise BundleValidationError("policy_spec golden_parity_file is not golden_io.npz")
    if spec.get("environment_action_transform") != "tanh(raw_action)":
        raise BundleValidationError("unsupported environment action transform")
    preprocessing = spec.get("observation_preprocessing")
    if (
        not isinstance(preprocessing, Mapping)
        or preprocessing.get("statistics_file") != "obs_stats.npz"
        or preprocessing.get("normalize") is not True
    ):
        raise BundleValidationError("policy_spec observation preprocessing is incomplete")

    task = str(_expect_equal("task", manifest.get("task"), spec.get("task"), provenance.get("task")))
    if not task or task == "None":
        raise BundleValidationError("bundle task must be a non-empty string")
    algorithm = str(
        _expect_equal(
            "algorithm",
            manifest.get("algorithm"),
            spec.get("algorithm"),
            provenance.get("algorithm"),
        )
    )
    if algorithm not in {"ppo", "fpo"}:
        raise BundleValidationError(f"unsupported algorithm: {algorithm!r}")
    fpo_commit = provenance.get("fpo_commit")
    expected_commit = provenance.get("expected_fpo_commit")
    if not runtime_only and (
        not isinstance(fpo_commit, str)
        or len(fpo_commit) not in {40, 64}
        or fpo_commit != expected_commit
        or provenance.get("fpo_commit_matches_expected") is not True
        or provenance.get("fpo_tracked_dirty") is not False
        or provenance.get("fpo_tracked_changes") != []
    ):
        raise BundleValidationError("bundle provenance does not bind a clean FPO commit")
    formal_eligible = provenance.get("formal_eligible")
    if formal_eligible is not None and not isinstance(formal_eligible, bool):
        raise BundleValidationError("bundle formal_eligible must be boolean when present")
    if not runtime_only and formal_eligible is True and (
        expected_fpo_commit is None or expected_runtime_digest is None
    ):
        raise BundleValidationError(
            "formal bundle validation requires external commit and runtime authority"
        )
    if not runtime_only and expected_fpo_commit is not None and re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", expected_fpo_commit
    ) is None:
        raise BundleValidationError("external FPO commit authority is malformed")
    if not runtime_only and expected_runtime_digest is not None and re.fullmatch(
        r"[0-9a-f]{64}", expected_runtime_digest
    ) is None:
        raise BundleValidationError("external runtime digest authority is malformed")
    if not runtime_only and expected_fpo_commit is not None and fpo_commit != expected_fpo_commit:
        raise BundleValidationError(
            "bundle FPO commit differs from external runtime authority"
        )
    if (
        not runtime_only
        and expected_runtime_digest is not None
        and provenance.get("runtime_digest") != expected_runtime_digest
    ):
        raise BundleValidationError(
            "bundle runtime digest differs from external runtime authority"
        )
    training_config = spec.get("training_config")
    if not isinstance(training_config, Mapping):
        raise BundleValidationError("policy_spec.training_config must be an object")
    if (
        _strict_int(
            training_config.get("episode_length"),
            "training_config.episode_length",
        )
        != 1000
    ):
        raise BundleValidationError("v0 policy training episode_length must be 1000")
    if training_config.get("normalize_observations") is not True:
        raise BundleValidationError("policy training must normalize observations")
    if algorithm == "fpo":
        inference = spec.get("inference")
        assert isinstance(inference, Mapping)
        for name in (
            "timestep_embed_dim",
            "flow_steps",
            "feather_std",
            "sde_sigma",
            "policy_mlp_output_scale",
        ):
            if inference.get(name) != training_config.get(name):
                raise BundleValidationError(
                    f"FPO inference.{name} differs from training_config.{name}"
                )
    seed = _strict_int(
        _expect_equal("training seed", manifest.get("seed"), provenance.get("training_seed")),
        "training_seed",
    )
    outer = _strict_int(
        _expect_equal(
            "outer iteration", manifest.get("outer_iteration"), provenance.get("outer_iteration")
        ),
        "outer_iteration",
    )
    environment_steps = _strict_int(
        _expect_equal(
            "environment steps",
            manifest.get("environment_steps"),
            provenance.get("environment_steps"),
        ),
        "environment_steps",
    )

    directory_match = _OUTER_DIRECTORY.fullmatch(bundle_dir.name)
    if directory_match and int(directory_match.group(1)) != outer:
        raise BundleValidationError("outer directory name disagrees with manifest")

    expected_values = {
        "task": (expected_task, task),
        "algorithm": (expected_algorithm, algorithm),
        "seed": (expected_seed, seed),
        "outer": (expected_outer, outer),
        "environment_steps": (expected_environment_steps, environment_steps),
    }
    for label, (expected, actual) in expected_values.items():
        if expected is not None and expected != actual:
            raise BundleValidationError(f"expected {label}={expected!r}, found {actual!r}")

    _validate_npz_structure(bundle_dir, spec)
    return PolicyBundleMetadata(
        bundle_dir=bundle_dir.resolve(),
        bundle_digest=_sha256(manifest_path),
        task=task,
        algorithm=algorithm,
        training_seed=seed,
        outer_iteration=outer,
        environment_steps=environment_steps,
        observation_dim=_strict_int(spec.get("observation_size"), "observation_size"),
        action_dim=_strict_int(spec.get("action_size"), "action_size"),
        manifest=manifest,
        policy_spec=spec,
        provenance=provenance,
    )
