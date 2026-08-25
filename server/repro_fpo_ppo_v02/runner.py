#!/usr/bin/env python3
"""Train/evaluate/export on one digest-bound native source-anchor environment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import socket
import stat
import subprocess
import sys
import time
import traceback
import uuid
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Package import for tests/``python -m``; fallback for direct script use.
    from .anchor_binding import (
        AnchorBindingError,
        AnchorManifest,
        BoundEnvironment,
        load_and_bind_anchor,
    )
    from .provenance import (
        AUDIT_SMOKE_EXECUTION_MODE,
        AUDIT_SMOKE_EXECUTION_PURPOSE,
        ContractError,
        EXECUTION_EVIDENCE_SCHEMA,
        EXECUTION_PURPOSES,
        FPO_SOURCE_ATTESTATION_KEYS,
        FORMAL_EXECUTION_PURPOSE,
        FORMAL_GPU_EXECUTION_MODE,
        NumericalIntegrityError,
        TRAINING_RECORD_SCHEMA,
        append_jsonl,
        assert_finite_array,
        assert_finite_mapping,
        atomic_write_bytes,
        atomic_write_json,
        json_ready,
        load_strict_json,
        runtime_contract_projection,
        sha256_json,
        utc_now,
        validate_attempt,
        validate_execution_evidence,
        validate_fpo_source_attestation,
        validate_policy_bundle,
        with_self_digest,
    )
    from .implementation import inspect_implementation_inventory
    from .vendor import inspect_vendor_directory, require_vendor_pythonpath_first
except ImportError:  # pragma: no cover - exercised by executable entry points
    from anchor_binding import (
        AnchorBindingError,
        AnchorManifest,
        BoundEnvironment,
        load_and_bind_anchor,
    )
    from provenance import (
        AUDIT_SMOKE_EXECUTION_MODE,
        AUDIT_SMOKE_EXECUTION_PURPOSE,
        ContractError,
        EXECUTION_EVIDENCE_SCHEMA,
        EXECUTION_PURPOSES,
        FPO_SOURCE_ATTESTATION_KEYS,
        FORMAL_EXECUTION_PURPOSE,
        FORMAL_GPU_EXECUTION_MODE,
        NumericalIntegrityError,
        TRAINING_RECORD_SCHEMA,
        append_jsonl,
        assert_finite_array,
        assert_finite_mapping,
        atomic_write_bytes,
        atomic_write_json,
        json_ready,
        load_strict_json,
        runtime_contract_projection,
        sha256_json,
        utc_now,
        validate_attempt,
        validate_execution_evidence,
        validate_fpo_source_attestation,
        validate_policy_bundle,
        with_self_digest,
    )
    from implementation import inspect_implementation_inventory
    from vendor import inspect_vendor_directory, require_vendor_pythonpath_first


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fpo-root", type=Path, required=True)
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        required=True,
        help="pinned legacy dependencies already prepended to PYTHONPATH",
    )
    parser.add_argument(
        "--legacy-policy-io",
        type=Path,
        required=True,
        help="exact legacy policy_io.py whose bytes are bound to this attempt",
    )
    parser.add_argument(
        "--execution-purpose",
        choices=tuple(sorted(EXECUTION_PURPOSES)),
        required=True,
    )
    parser.add_argument(
        "--allow-non-gpu",
        action="store_true",
        help="Dependency/smoke debugging only; formal plans must use GPU",
    )
    return parser


def _git(root: Path, *args: str) -> str | None:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _git_raw(root: Path, *args: str) -> bytes:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            timeout=60,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"cannot inspect upstream FPO checkout with git {args}") from error
    return result.stdout


def _git_path(value: bytes, *, where: str) -> str:
    try:
        path = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{where} is not UTF-8") from error
    pure = PurePosixPath(path)
    if (
        not path
        or pure.is_absolute()
        or ".." in pure.parts
        or "\n" in path
        or "\r" in path
    ):
        raise RuntimeError(f"unsafe {where}: {path!r}")
    return path


def _blob_object_id(data: bytes, *, object_format: str) -> str:
    if object_format not in {"sha1", "sha256"}:
        raise RuntimeError(f"unsupported Git object format: {object_format!r}")
    digest = hashlib.new(object_format)
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _snapshot_fpo_tree(fpo_root: Path, *, commit: str) -> dict[str, Any]:
    """Hash every tracked execution byte and expose Git status bypasses."""

    object_format = _git(fpo_root, "rev-parse", "--show-object-format")
    if object_format is None:
        raise RuntimeError("cannot resolve upstream FPO Git object format")
    replacement_refs = _git_raw(
        fpo_root, "for-each-ref", "--format=%(refname)", "refs/replace"
    ).strip()
    if replacement_refs:
        raise RuntimeError("upstream FPO checkout contains forbidden Git replace refs")
    tree = _git_raw(fpo_root, "ls-tree", "-rz", "--full-tree", commit)
    head_entries: list[dict[str, Any]] = []
    worktree_entries: list[dict[str, Any]] = []
    execution_entries: list[dict[str, Any]] = []
    content_changes: list[str] = []
    for raw in tree.split(b"\0"):
        if not raw:
            continue
        try:
            header, raw_path = raw.split(b"\t", 1)
            mode, kind, object_id = header.decode("ascii").split(" ")
        except (ValueError, UnicodeDecodeError) as error:
            raise RuntimeError("cannot parse upstream FPO HEAD tree") from error
        path = _git_path(raw_path, where="tracked FPO path")
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise RuntimeError(
                f"FPO source proof rejects non-regular tracked entry {path!r}: "
                f"mode={mode!r}, kind={kind!r}"
            )
        absolute = fpo_root.joinpath(*PurePosixPath(path).parts)
        actual_object: str | None = None
        actual_mode: str | None = None
        raw_sha256: str | None = None
        byte_count: int | None = None
        try:
            metadata = absolute.lstat()
            if absolute.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"tracked FPO entry is not a regular file: {path}")
            data = absolute.read_bytes()
            actual_mode = "100755" if metadata.st_mode & 0o111 else "100644"
            actual_object = _blob_object_id(data, object_format=object_format)
            raw_sha256 = hashlib.sha256(data).hexdigest()
            byte_count = len(data)
        except FileNotFoundError:
            pass
        head_entries.append({"mode": mode, "object": object_id, "path": path})
        worktree_entries.append(
            {"mode": actual_mode, "object": actual_object, "path": path}
        )
        execution_entries.append(
            {
                "bytes": byte_count,
                "mode": actual_mode,
                "path": path,
                "sha256": raw_sha256,
            }
        )
        if actual_mode != mode or actual_object != object_id:
            content_changes.append(path)
    if not head_entries:
        raise RuntimeError("upstream FPO HEAD tree has no regular files")

    untracked = sorted(
        _git_path(raw, where="untracked FPO path")
        for raw in _git_raw(
            # Include ignored files too: a stale/forged ``__pycache__`` can be
            # imported even when porcelain status reports a clean checkout.
            fpo_root,
            "ls-files",
            "--others",
            "-z",
        ).split(b"\0")
        if raw
    )
    index_flags: list[str] = []
    for raw in _git_raw(fpo_root, "ls-files", "-v", "-z").split(b"\0"):
        if not raw:
            continue
        if len(raw) < 3 or raw[1:2] != b" ":
            raise RuntimeError("cannot parse upstream FPO index flags")
        tag = chr(raw[0])
        path = _git_path(raw[2:], where="indexed FPO path")
        if tag == "S" or tag.islower():
            index_flags.append(f"{tag} {path}")

    status_records = [
        raw
        for raw in _git_raw(
            fpo_root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ).split(b"\0")
        if raw
    ]
    tracked_status = []
    for raw in status_records:
        if raw.startswith(b"?? "):
            continue
        try:
            tracked_status.append(raw.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise RuntimeError("upstream FPO status contains a non-UTF-8 path") from error
    tracked_changes = sorted(set(content_changes + tracked_status))
    return {
        "fpo_tracked_dirty": bool(tracked_changes),
        "fpo_tracked_changes": tracked_changes,
        "fpo_head_tree_digest": sha256_json(head_entries),
        "fpo_worktree_tree_digest": sha256_json(worktree_entries),
        "fpo_execution_tree_digest": sha256_json(execution_entries),
        "fpo_source_file_count": len(head_entries),
        "fpo_index_flags": sorted(index_flags),
        "fpo_untracked_paths": untracked,
    }


def _load_upstream(fpo_root: Path) -> tuple[Any, ...]:
    source_dir = fpo_root / "playground" / "src"
    if not (source_dir / "flow_policy" / "fpo.py").is_file():
        raise FileNotFoundError(f"not an FPO checkout: {fpo_root}")
    sys.path.insert(0, str(source_dir))
    import jax
    import jax_dataclasses as jdc
    from jax import numpy as jnp
    from mujoco_playground import dm_control_suite, registry
    from flow_policy import fpo, ppo, rollouts

    return jax, jdc, jnp, dm_control_suite, registry, fpo, ppo, rollouts


def _legacy_export_policy_bundle(policy_io_path: Path) -> Any:
    policy_io = policy_io_path.resolve()
    if policy_io.name != "policy_io.py" or not policy_io.is_file():
        raise FileNotFoundError(
            f"read-only legacy policy exporter is unavailable: {policy_io}"
        )
    spec = importlib.util.spec_from_file_location(
        "_policy_learnware_v02_legacy_policy_io", policy_io
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load exact legacy policy exporter: {policy_io}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    export_policy_bundle = getattr(module, "export_policy_bundle", None)
    if not callable(export_policy_bundle):
        raise ImportError("legacy policy_io.py has no callable export_policy_bundle")
    return export_policy_bundle


def _assert_clean_attempt_root(run_dir: Path, attempt_path: Path) -> None:
    if attempt_path.resolve() != (run_dir / "attempt_manifest.json").resolve():
        raise ContractError("attempt manifest must be the immutable file inside run-dir")
    forbidden = {
        "run_manifest.json",
        "status.json",
        "events.jsonl",
        "training_record.json",
        "traceback.txt",
        "checkpoints",
    }
    existing = forbidden & {path.name for path in run_dir.iterdir()}
    if existing:
        raise ContractError(
            f"runner refuses in-place resume/overwrite of an existing attempt: {sorted(existing)}"
        )


def _inspect_fpo_source(
    fpo_root: Path, *, expected_commit: str
) -> dict[str, Any]:
    """Derive a consumer-compatible attestation from the live FPO checkout."""

    commit = _git(fpo_root, "rev-parse", "HEAD")
    if commit is None:
        raise RuntimeError("cannot resolve upstream FPO commit")
    result = {
        "fpo_commit": commit,
        "expected_fpo_commit": expected_commit,
        "fpo_commit_matches_expected": commit == expected_commit,
        **_snapshot_fpo_tree(fpo_root, commit=commit),
    }
    if _git(fpo_root, "rev-parse", "HEAD") != commit:
        raise RuntimeError("upstream FPO HEAD changed during source attestation")
    return result


def _verify_source(fpo_root: Path, anchor: AnchorManifest) -> dict[str, Any]:
    source = _inspect_fpo_source(
        fpo_root, expected_commit=str(anchor.runtime["fpo_commit"])
    )
    try:
        validated = validate_fpo_source_attestation(
            source,
            expected_commit=str(anchor.runtime["fpo_commit"]),
            require_exact=True,
        )
    except ContractError as error:
        raise RuntimeError(
            f"upstream FPO source proof failed; refusing provenance drift: {error}"
        ) from error
    commit = str(source["fpo_commit"])
    actual_runtime = runtime_contract_projection(fpo_commit=commit)
    if actual_runtime != anchor.runtime:
        raise RuntimeError(
            f"native runtime contract mismatch: actual={actual_runtime}, frozen={anchor.runtime}"
        )
    if sha256_json(actual_runtime) != anchor.runtime_digest:
        raise RuntimeError("native runtime digest mismatch")
    return validated


def _assert_source_unchanged(
    fpo_root: Path,
    anchor: AnchorManifest,
    expected: Mapping[str, Any],
    *,
    where: str,
) -> None:
    observed = _verify_source(fpo_root, anchor)
    frozen = {key: expected.get(key) for key in FPO_SOURCE_ATTESTATION_KEYS}
    if observed != frozen:
        raise RuntimeError(f"upstream FPO source changed {where}")


def _runtime_provenance(
    *,
    args: argparse.Namespace,
    attempt: Mapping[str, Any],
    anchor: AnchorManifest,
    jax: Any,
    source: Mapping[str, Any],
    vendor: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    source = validate_fpo_source_attestation(
        source,
        expected_commit=str(anchor.runtime["fpo_commit"]),
        require_exact=True,
    )
    hardware = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    execution = with_self_digest(
        {
            "schema": EXECUTION_EVIDENCE_SCHEMA,
            "config_digest": attempt["config_digest"],
            "execution_purpose": attempt["execution_purpose"],
            "execution_mode": attempt["execution_mode"],
            "formal_eligible": attempt["formal_eligible"],
            "allow_non_gpu": bool(args.allow_non_gpu),
            "jax_backend": hardware["jax_backend"],
            "jax_devices": hardware["jax_devices"],
            "cuda_visible_devices": hardware["cuda_visible_devices"],
            "hardware_digest": sha256_json(hardware),
            "job_digest": attempt["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "attempt_root": str(args.run_dir.resolve()),
        },
        key="execution_evidence_digest",
    )
    validate_execution_evidence(
        execution,
        expected_job_digest=attempt["job_digest"],
        expected_attempt_digest=attempt["attempt_digest"],
        expected_hardware_digest=sha256_json(hardware),
        expected_config_digest=attempt["config_digest"],
        expected_execution_purpose=attempt["execution_purpose"],
        expected_attempt_root=args.run_dir,
        require_formal=bool(attempt["formal_eligible"]),
    )
    return {
        "runner_schema": "policy-learnware.v02-anchor-aware-runner.v0",
        "runner_file": str(Path(__file__).resolve()),
        "fpo_root": str(args.fpo_root),
        **dict(source),
        "runtime_contract": dict(anchor.runtime),
        "runtime_digest": anchor.runtime_digest,
        "vendor": dict(vendor),
        "implementation": dict(implementation),
        "legacy_policy_io_path": str(args.legacy_policy_io),
        "pythonpath_vendor_precedence_verified": True,
        "wandb_mode": os.environ.get("WANDB_MODE"),
        "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "host": hardware["host"],
        "pid": os.getpid(),
        "platform": platform.platform(),
        "python": sys.version,
        "cuda_visible_devices": hardware["cuda_visible_devices"],
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "jax_backend": hardware["jax_backend"],
        "jax_devices": hardware["jax_devices"],
        "hardware_contract": hardware,
        "hardware_digest": sha256_json(hardware),
        "execution_evidence": execution,
        "command": sys.argv,
        "started_at": utc_now(),
    }


def _build_agent_and_rollout(
    *,
    job: Mapping[str, Any],
    bound: BoundEnvironment,
    jax: Any,
    jdc: Any,
    fpo: Any,
    ppo: Any,
    rollouts: Any,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    protocol = job["training_protocol"]
    config_payload = dict(protocol["trainer_config"])
    if protocol["algorithm"] == "fpo":
        config = fpo.FpoConfig(**config_payload)
        agent_state = fpo.FpoState.init(
            prng=jax.random.key(job["seed"]),
            env=bound.env,
            config=config,
        )
    else:
        config = ppo.PpoConfig(**config_payload)
        agent_state = ppo.PpoState.init(
            prng=jax.random.key(job["seed"]),
            env=bound.env,
            config=config,
        )
    reconstructed = json_ready(jdc.asdict(config))
    if reconstructed != config_payload:
        raise ContractError(
            "trainer_config is not complete/exact after native config reconstruction"
        )
    rollout_state = rollouts.BatchedRolloutState.init(
        bound.env,
        prng=jax.random.key(job["seed"]),
        num_envs=config.num_envs,
    )
    if agent_state.env is not bound.env or rollout_state.env is not bound.env:
        raise AnchorBindingError("agent and rollout state did not retain the same bound env object")
    jax.device_get(agent_state.steps)
    return agent_state, rollout_state, config, reconstructed


def _assert_jax_tree_finite(jax: Any, value: Any, *, where: str) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves:
        raise NumericalIntegrityError(f"{where} has no numerical leaves")
    for index, leaf in enumerate(jax.device_get(leaves)):
        assert_finite_array(leaf, where=f"{where}[{index}]")


def _assert_policy_state_finite(jax: Any, state: Any) -> None:
    _assert_jax_tree_finite(jax, state.params.policy, where="actor")
    _assert_jax_tree_finite(
        jax,
        {
            "count": state.obs_stats.count,
            "mean": state.obs_stats.mean,
            "var_sum": state.obs_stats.var_sum,
            "std": state.obs_stats.std,
        },
        where="obs_stats",
    )


def _strict_scalar(value: Any, where: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise NumericalIntegrityError(f"{where} is non-finite")
    return number


def _reduce_train_metrics(jax: Any, jnp: Any, transitions: Any, metrics: Any) -> dict[str, Any]:
    _assert_jax_tree_finite(
        jax,
        {
            "obs": transitions.obs,
            "next_obs": transitions.next_obs,
            "action": transitions.action,
            "action_info": transitions.action_info,
            "reward": transitions.reward,
            "truncation": transitions.truncation,
            "discount": transitions.discount,
            "optimizer": metrics,
        },
        where="training_step",
    )
    termination_count = jnp.sum(transitions.discount == 0.0)
    reduced = {
        "mean_reward": jnp.mean(transitions.reward),
        "reward_std": jnp.std(transitions.reward),
        "termination_count": termination_count,
        **{f"optimizer/{key}": jnp.mean(value) for key, value in metrics.items()},
    }
    host = jax.device_get(reduced)
    result = {key: _strict_scalar(item, f"train_metrics.{key}") for key, item in host.items()}
    count = result["termination_count"]
    result["has_terminated_episodes"] = count > 0.0
    result["mean_steps_per_terminated_episode"] = (
        float(transitions.discount.size) / count if count > 0.0 else 0.0
    )
    assert_finite_mapping(result, where="train_metrics")
    return result


def _reduce_eval_metrics(jax: Any, outputs: Any) -> dict[str, Any]:
    _assert_jax_tree_finite(
        jax,
        {
            "scalar_metrics": outputs.scalar_metrics,
            "actions": outputs.actions,
            "action_timestep_mask": outputs.action_timestep_mask,
        },
        where="evaluation",
    )
    host = jax.device_get(outputs.scalar_metrics)
    result = {key: _strict_scalar(item, f"eval_metrics.{key}") for key, item in host.items()}
    assert_finite_mapping(result, where="eval_metrics")
    return result


def _obs_stats_summary(jax: Any, jnp: Any, state: Any) -> dict[str, Any]:
    host = jax.device_get(
        {
            "count": state.obs_stats.count,
            "mean_min": jnp.min(state.obs_stats.mean),
            "mean_max": jnp.max(state.obs_stats.mean),
            "std_min": jnp.min(state.obs_stats.std),
            "std_max": jnp.max(state.obs_stats.std),
        }
    )
    result = {key: _strict_scalar(item, f"obs_stats_summary.{key}") for key, item in host.items()}
    assert_finite_mapping(result, where="obs_stats_summary")
    return result


def _verify_golden(
    *, jax: Any, jnp: Any, state: Any, observations: Any, seed: int
) -> None:
    _assert_jax_tree_finite(jax, observations, where="golden.observation")
    raw_action, _ = state.sample_action(
        observations,
        jax.random.key(seed),
        deterministic=True,
    )
    _assert_jax_tree_finite(jax, raw_action, where="golden.raw_action")
    _assert_jax_tree_finite(jax, jnp.tanh(raw_action), where="golden.environment_action")


def _load_policy_verifiers() -> tuple[Any, ...]:
    """Import the versioned consumer-side loader from either deployment layout.

    The tracked copy lives under ``policy_learnware_v0/server`` while the server
    deployment is a sibling of ``policy_learnware_v0``.  Only those two exact
    source layouts are considered; an unrelated working directory is never
    scanned for a package with the same name.
    """

    here = Path(__file__).resolve()
    candidates = (
        here.parents[2] / "src",
        here.parents[1] / "policy_learnware_v0" / "src",
    )
    for source in candidates:
        if (source / "policy_learnware_v0" / "policy" / "loader.py").is_file():
            source_text = str(source)
            if source_text not in sys.path:
                sys.path.insert(0, source_text)
            break
    else:
        raise FileNotFoundError(
            "the versioned policy_learnware_v0 consumer package is unavailable"
        )

    from policy_learnware_v0.policy.bundle import validate_bundle
    from policy_learnware_v0.policy.evaluate import verify_compiled_policy_parity
    from policy_learnware_v0.policy.loader import load_policy
    from policy_learnware_v0.policy.parity import verify_golden_parity

    return (
        validate_bundle,
        load_policy,
        verify_golden_parity,
        verify_compiled_policy_parity,
    )


def _verify_reloaded_checkpoint(
    *,
    bundle_dir: Path,
    args: argparse.Namespace,
    job: Mapping[str, Any],
    anchor: AnchorManifest,
    bound: BoundEnvironment,
    jax: Any,
    jdc: Any,
    jnp: Any,
    fpo: Any,
    ppo: Any,
    integrity: Mapping[str, Any],
    environment_steps: int,
    outer: int,
) -> dict[str, Any]:
    """Reload exported bytes and prove golden/scalar/compiled equivalence."""

    (
        validate_bundle,
        load_policy,
        verify_golden_parity,
        verify_compiled_policy_parity,
    ) = _load_policy_verifiers()
    metadata = validate_bundle(
        bundle_dir,
        expected_task=anchor.task,
        expected_algorithm=job["training_protocol"]["algorithm"],
        expected_seed=job["seed"],
        expected_outer=outer,
        expected_environment_steps=environment_steps,
        expected_fpo_commit=str(anchor.runtime["fpo_commit"]),
        expected_runtime_digest=anchor.runtime_digest,
    )

    def restore_runtime(
        restored_metadata: Any,
        actor: Mapping[str, Any],
        obs_stats: Mapping[str, Any],
        _fpo_root: Path,
    ) -> Any:
        module = fpo if restored_metadata.algorithm == "fpo" else ppo
        config_name = "FpoConfig" if restored_metadata.algorithm == "fpo" else "PpoConfig"
        state_name = "FpoState" if restored_metadata.algorithm == "fpo" else "PpoState"
        config = getattr(module, config_name)(
            **dict(restored_metadata.policy_spec["training_config"])
        )
        state = getattr(module, state_name).init(
            prng=jax.random.key(0), env=bound.env, config=config
        )
        kernel_names = sorted(name for name in actor if name.endswith("_kernel"))
        if not kernel_names:
            raise ContractError("reloaded actor has no ordered kernel/bias layers")
        ordered_actor = tuple(
            (
                jnp.asarray(actor[name]),
                jnp.asarray(actor[name.replace("_kernel", "_bias")]),
            )
            for name in kernel_names
        )
        with jdc.copy_and_mutate(state) as restored:
            restored.params.policy = ordered_actor
            for name in ("count", "mean", "var_sum", "std"):
                if name in obs_stats:
                    setattr(restored.obs_stats, name, jnp.asarray(obs_stats[name]))
        if restored.env is not bound.env:
            raise AnchorBindingError("reloaded policy escaped the source-anchor environment")
        return restored

    policy = load_policy(
        metadata,
        fpo_root=args.fpo_root,
        runtime_factory=restore_runtime,
    )
    parity = job["training_protocol"]["parity"]
    atol = float(parity["atol"])
    rtol = float(parity["rtol"])
    golden = verify_golden_parity(policy, metadata, atol=atol, rtol=rtol)
    golden_payload = json_ready(asdict(golden))
    if golden_payload["passed"] is not True or golden_payload["raw_checked"] is not True:
        raise ContractError(f"reloaded golden parity failed: {golden_payload}")
    with np.load(metadata.bundle_dir / "golden_io.npz", allow_pickle=False) as archive:
        observations = np.asarray(archive["observation"])
        key_data = np.asarray(archive["prng_key_data"])
    compiled = verify_compiled_policy_parity(
        policy,
        observations,
        key_data,
        atol=atol,
        rtol=rtol,
        sample_count=int(parity["compiled_sample_count"]),
    )
    compiled_payload = json_ready(asdict(compiled))
    if compiled_payload["passed"] is not True or compiled_payload["next_keys_equal"] is not True:
        raise ContractError(f"reloaded compiled-policy parity failed: {compiled_payload}")

    finiteness = {
        "passed": True,
        "all_arrays_finite": True,
        "bundle_manifest_sha256": integrity["bundle_manifest_sha256"],
        "validated_file_digests": dict(integrity["files"]),
    }
    return {
        "bundle_digest": metadata.bundle_digest,
        "finiteness_audit": with_self_digest(finiteness, key="report_digest"),
        "golden_parity": with_self_digest(golden_payload, key="report_digest"),
        "compiled_parity": with_self_digest(compiled_payload, key="report_digest"),
    }


def _publish_checkpoint(
    *,
    args: argparse.Namespace,
    export_policy_bundle: Any,
    job: Mapping[str, Any],
    attempt: Mapping[str, Any],
    anchor: AnchorManifest,
    bound: BoundEnvironment,
    runtime: Mapping[str, Any],
    agent_state: Any,
    config_payload: Mapping[str, Any],
    outer: int,
    environment_steps: int,
    evaluation: Mapping[str, Any] | None,
    golden_observations: Any,
    golden_seed: int,
    jax: Any,
    jdc: Any,
    jnp: Any,
    fpo: Any,
    ppo: Any,
) -> dict[str, Any]:
    _assert_source_unchanged(
        args.fpo_root,
        anchor,
        runtime,
        where=f"before checkpoint outer={outer}",
    )
    bound.verify()
    if agent_state.env is not bound.env:
        raise AnchorBindingError("checkpoint actor is no longer bound to the source-anchor env")
    _assert_policy_state_finite(jax, agent_state)
    _verify_golden(
        jax=jax,
        jnp=jnp,
        state=agent_state,
        observations=golden_observations,
        seed=golden_seed,
    )
    checkpoint_parent = args.run_dir / "checkpoints"
    checkpoint_parent.mkdir(parents=True, exist_ok=True)
    final = checkpoint_parent / f"outer_{outer:06d}"
    if final.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {final}")
    temporary = checkpoint_parent / f".outer_{outer:06d}.validating-{uuid.uuid4().hex}"
    export_policy_bundle(
        temporary,
        agent_state=agent_state,
        algo=job["training_protocol"]["algorithm"],
        task=anchor.task,
        seed=job["seed"],
        outer_iteration=outer,
        environment_steps=environment_steps,
        nominal_timesteps=int(config_payload["num_timesteps"]),
        config=config_payload,
        provenance={
            **runtime,
            "run_manifest": "../../run_manifest.json",
            "config_digest": job["config_digest"],
            "execution_purpose": job["execution_purpose"],
            "job_digest": job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "anchor_manifest_digest": anchor.manifest_digest,
            "environment_instance_digest": anchor.environment_instance_digest,
            "operator_digest": anchor.operator_digest,
            "model_diff_digest": anchor.model_diff_digest,
            "actual_bound_model_digest": bound.verify().digest,
            "execution_mode": runtime["execution_evidence"]["execution_mode"],
            "formal_eligible": runtime["execution_evidence"]["formal_eligible"],
            "execution_evidence_digest": runtime["execution_evidence"][
                "execution_evidence_digest"
            ],
            "attempt_root": runtime["execution_evidence"]["attempt_root"],
            "obs_stats_summary": _obs_stats_summary(jax, jnp, agent_state),
        },
        golden_observations=golden_observations,
        golden_seed=golden_seed,
        evaluation=evaluation,
    )
    integrity = validate_policy_bundle(
        temporary,
        require_evaluation=bool(job["training_protocol"]["evaluation"]["enabled"]),
    )
    reload_audit = _verify_reloaded_checkpoint(
        bundle_dir=temporary,
        args=args,
        job=job,
        anchor=anchor,
        bound=bound,
        jax=jax,
        jdc=jdc,
        jnp=jnp,
        fpo=fpo,
        ppo=ppo,
        integrity=integrity,
        environment_steps=environment_steps,
        outer=outer,
    )
    _assert_source_unchanged(
        args.fpo_root,
        anchor,
        runtime,
        where=f"during checkpoint outer={outer}",
    )
    os.rename(temporary, final)
    return {
        "outer_iteration": outer,
        "environment_steps": environment_steps,
        "path": str(final),
        **integrity,
        "config_digest": job["config_digest"],
        "execution_purpose": job["execution_purpose"],
        "execution_mode": runtime["execution_evidence"]["execution_mode"],
        "formal_eligible": runtime["execution_evidence"]["formal_eligible"],
        "execution_evidence_digest": runtime["execution_evidence"][
            "execution_evidence_digest"
        ],
        **reload_audit,
    }


def run(args: argparse.Namespace) -> None:
    args.run_dir = args.run_dir.resolve()
    args.fpo_root = args.fpo_root.resolve()
    args.attempt_manifest = args.attempt_manifest.resolve()
    vendor = inspect_vendor_directory(args.vendor_dir)
    args.vendor_dir = Path(vendor["path"])
    args.legacy_policy_io = args.legacy_policy_io.resolve()
    implementation = inspect_implementation_inventory(
        runner_path=Path(__file__),
        legacy_policy_io_path=args.legacy_policy_io,
    )
    require_vendor_pythonpath_first(vendor)
    if os.environ.get("WANDB_MODE") != "disabled":
        raise ContractError("runner requires WANDB_MODE=disabled")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ContractError("runner requires PYTHONDONTWRITEBYTECODE=1")
    if not args.run_dir.is_dir():
        raise FileNotFoundError(f"run-dir must be an allocated attempt directory: {args.run_dir}")
    _assert_clean_attempt_root(args.run_dir, args.attempt_manifest)
    attempt = validate_attempt(load_strict_json(args.attempt_manifest))
    if attempt["implementation"] != implementation:
        raise ContractError(
            "runner implementation bytes differ from the immutable attempt"
        )
    if args.execution_purpose != attempt["execution_purpose"]:
        raise ContractError(
            "explicit runner execution purpose differs from the immutable attempt"
        )
    if args.allow_non_gpu is not (
        args.execution_purpose == AUDIT_SMOKE_EXECUTION_PURPOSE
    ):
        raise ContractError(
            "--allow-non-gpu must be supplied exactly for audit_smoke execution"
        )
    expected_execution_mode = (
        AUDIT_SMOKE_EXECUTION_MODE
        if args.allow_non_gpu
        else FORMAL_GPU_EXECUTION_MODE
    )
    expected_formal_eligible = (
        args.execution_purpose == FORMAL_EXECUTION_PURPOSE
        and not args.allow_non_gpu
    )
    if (
        attempt["execution_mode"] != expected_execution_mode
        or attempt["formal_eligible"] is not expected_formal_eligible
    ):
        raise ContractError(
            "runner flag disagrees with the execution mode frozen in the attempt"
        )
    job = attempt["job"]
    if (
        job["config_digest"] != attempt["config_digest"]
        or job["execution_purpose"] != args.execution_purpose
    ):
        raise ContractError("runner job config/purpose differs from its attempt")
    anchor_path = Path(job["anchor_manifest_path"])
    anchor = AnchorManifest.from_path(anchor_path)
    if anchor.manifest_digest != job["anchor_manifest_digest"]:
        raise ContractError("job's source anchor manifest digest drifted before execution")
    protocol = job["training_protocol"]
    events_path = args.run_dir / "events.jsonl"
    status_path = args.run_dir / "status.json"
    atomic_write_json(
        status_path,
        {
            "state": "initializing",
            "job_digest": job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "anchor_manifest_digest": anchor.manifest_digest,
            "updated_at": utc_now(),
        },
    )
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    source = _verify_source(args.fpo_root, anchor)
    (
        jax,
        jdc,
        jnp,
        dm_control_suite,
        registry,
        fpo,
        ppo,
        rollouts,
    ) = _load_upstream(args.fpo_root)
    _assert_source_unchanged(
        args.fpo_root, anchor, source, where="while importing the native runtime"
    )
    if anchor.task not in dm_control_suite.ALL_ENVS:
        raise ContractError(f"anchor task is not a registered DMC environment: {anchor.task}")
    if not args.allow_non_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(
            f"non-smoke training requires GPU; backend={jax.default_backend()!r}"
        )
    bound = load_and_bind_anchor(registry=registry, manifest=anchor)
    bound.verify()
    agent_state, rollout_state, config, config_payload = _build_agent_and_rollout(
        job=job,
        bound=bound,
        jax=jax,
        jdc=jdc,
        fpo=fpo,
        ppo=ppo,
        rollouts=rollouts,
    )
    _assert_policy_state_finite(jax, agent_state)
    per_outer = int(config.iterations_per_env * config.num_envs)
    if per_outer <= 0:
        raise ContractError("native trainer produced a non-positive outer-step geometry")
    maximum = int(protocol["max_outer_iterations"])
    nominal_floor = int(config.num_timesteps) // per_outer
    if maximum > nominal_floor:
        raise ContractError(
            f"max_outer_iterations exceeds explicit trainer budget floor: {maximum}>{nominal_floor}"
        )
    runtime = _runtime_provenance(
        args=args,
        attempt=attempt,
        anchor=anchor,
        jax=jax,
        source=source,
        vendor=vendor,
        implementation=implementation,
    )
    execution = runtime["execution_evidence"]
    run_manifest = with_self_digest(
        {
            "schema": "policy-learnware.v02-anchor-training-run.v0",
            "job": job,
            "job_digest": job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "config_digest": job["config_digest"],
            "execution_purpose": job["execution_purpose"],
            "anchor_manifest": anchor.to_dict(),
            "anchor_manifest_digest": anchor.manifest_digest,
            "environment_instance_digest": anchor.environment_instance_digest,
            "model_diff_digest": anchor.model_diff_digest,
            "binding_audit": bound.audit.to_dict(),
            "training_protocol_digest": protocol["protocol_digest"],
            "config": config_payload,
            "num_envs": int(config.num_envs),
            "iterations_per_env": int(config.iterations_per_env),
            "transitions_per_outer": per_outer,
            "planned_environment_steps": maximum * per_outer,
            "execution_mode": execution["execution_mode"],
            "formal_eligible": execution["formal_eligible"],
            "execution_evidence_digest": execution["execution_evidence_digest"],
            "runtime": runtime,
        },
        key="run_manifest_digest",
    )
    atomic_write_json(args.run_dir / "run_manifest.json", run_manifest, overwrite=False)
    append_jsonl(
        events_path,
        {
            "event": "run_started",
            "at": utc_now(),
            "job_digest": job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "config_digest": job["config_digest"],
            "execution_purpose": job["execution_purpose"],
            "environment_instance_digest": anchor.environment_instance_digest,
            "run_manifest_digest": run_manifest["run_manifest_digest"],
            "implementation_digest": implementation["implementation_digest"],
        },
    )
    atomic_write_json(
        status_path,
        {
            "state": "running",
            "job_digest": job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "anchor_manifest_digest": anchor.manifest_digest,
            "environment_instance_digest": anchor.environment_instance_digest,
            "last_completed_outer": 0,
            "environment_steps": 0,
            "updated_at": utc_now(),
        },
    )
    export_policy_bundle = _legacy_export_policy_bundle(args.legacy_policy_io)
    export_outers = set(protocol["export_outer_iterations"])
    eval_contract = protocol["evaluation"]
    checkpoints: list[dict[str, Any]] = []
    started_at = utc_now()
    started = time.monotonic()
    latest_metrics: dict[str, Any] | None = None
    for index in range(maximum):
        outer = index + 1
        step_started = time.monotonic()
        if rollout_state.env is not bound.env or agent_state.env is not bound.env:
            raise AnchorBindingError("training state escaped the bound source-anchor env")
        rollout_state, transitions = rollout_state.rollout(
            agent_state,
            episode_length=config.episode_length,
            iterations_per_env=config.iterations_per_env,
        )
        if int(transitions.reward.size) != per_outer:
            raise RuntimeError("native transition geometry drifted during training")
        agent_state, optimizer_metrics = agent_state.training_step(transitions)
        summary = _reduce_train_metrics(jax, jnp, transitions, optimizer_metrics)
        _assert_policy_state_finite(jax, agent_state)
        elapsed = time.monotonic() - step_started
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise NumericalIntegrityError("outer elapsed time is invalid")
        environment_steps = outer * per_outer
        summary.update(
            {
                "outer_iteration": outer,
                "environment_steps": environment_steps,
                "optimizer_minibatch_steps": int(jax.device_get(agent_state.steps)),
                "elapsed_seconds": elapsed,
                "environment_steps_per_second": per_outer / elapsed,
            }
        )
        assert_finite_mapping(summary, where="outer_summary")
        append_jsonl(events_path, {"event": "train_outer_completed", "at": utc_now(), **summary})
        latest_metrics = summary
        if outer in export_outers:
            bound.verify()
            evaluation: dict[str, Any] | None = None
            if eval_contract["enabled"]:
                if agent_state.env is not bound.env:
                    raise AnchorBindingError("evaluator actor is not bound to source-anchor env")
                outputs = rollouts.eval_policy(
                    agent_state,
                    prng=jax.random.fold_in(jax.random.key(eval_contract["base_seed"]), outer),
                    num_envs=eval_contract["num_envs"],
                    max_episode_length=config.episode_length,
                )
                evaluation = _reduce_eval_metrics(jax, outputs)
                append_jsonl(
                    events_path,
                    {
                        "event": "post_update_evaluation",
                        "at": utc_now(),
                        "outer_iteration": outer,
                        "environment_steps": environment_steps,
                        "environment_instance_digest": anchor.environment_instance_digest,
                        **evaluation,
                    },
                )
                del outputs
            golden_batch = int(protocol["parity"]["golden_sample_count"])
            if int(config.num_envs) < golden_batch:
                raise ContractError(
                    "trainer num_envs is smaller than the frozen golden parity sample count"
                )
            golden_observations = transitions.obs[0, :golden_batch]
            golden_seed = (1_000_003 + int(job["seed"]) * 10_007 + outer) % (2**31 - 1)
            checkpoint = _publish_checkpoint(
                args=args,
                export_policy_bundle=export_policy_bundle,
                job=job,
                attempt=attempt,
                anchor=anchor,
                bound=bound,
                runtime=runtime,
                agent_state=agent_state,
                config_payload=config_payload,
                outer=outer,
                environment_steps=environment_steps,
                evaluation=evaluation,
                golden_observations=golden_observations,
                golden_seed=golden_seed,
                jax=jax,
                jdc=jdc,
                jnp=jnp,
                fpo=fpo,
                ppo=ppo,
            )
            checkpoints.append(checkpoint)
            append_jsonl(
                events_path,
                {"event": "checkpoint_published", "at": utc_now(), **checkpoint},
            )
        atomic_write_json(
            status_path,
            {
                "state": "running",
                "job_digest": job["job_digest"],
                "attempt_digest": attempt["attempt_digest"],
                "anchor_manifest_digest": anchor.manifest_digest,
                "environment_instance_digest": anchor.environment_instance_digest,
                "last_completed_outer": outer,
                "environment_steps": environment_steps,
                "latest_metrics": summary,
                "exported_outer_iterations": [item["outer_iteration"] for item in checkpoints],
                "updated_at": utc_now(),
            },
        )
        print(
            f"[{protocol['algorithm']} anchor={anchor.anchor_id} seed={job['seed']}] "
            f"outer={outer}/{maximum} env_steps={environment_steps} "
            f"reward={summary['mean_reward']:.6f} seconds={elapsed:.2f}",
            flush=True,
        )
    expected_exports = list(protocol["export_outer_iterations"])
    observed_exports = [item["outer_iteration"] for item in checkpoints]
    if observed_exports != expected_exports:
        raise RuntimeError(
            f"checkpoint export set mismatch: observed={observed_exports}, expected={expected_exports}"
        )
    _assert_source_unchanged(
        args.fpo_root, anchor, runtime, where="before final success publication"
    )
    bound.verify()
    finished_at = utc_now()
    wall_seconds = time.monotonic() - started
    record = with_self_digest(
        {
            "schema": TRAINING_RECORD_SCHEMA,
            "state": "succeeded",
            "config_digest": job["config_digest"],
            "execution_purpose": job["execution_purpose"],
            "job_digest": job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "anchor_manifest_digest": anchor.manifest_digest,
            "environment_instance_digest": anchor.environment_instance_digest,
            "training_protocol_digest": protocol["protocol_digest"],
            "algorithm": protocol["algorithm"],
            "seed": job["seed"],
            "execution_mode": execution["execution_mode"],
            "formal_eligible": execution["formal_eligible"],
            "implementation": implementation,
            "execution_evidence_digest": execution["execution_evidence_digest"],
            "checkpoint_bundles": checkpoints,
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_seconds": wall_seconds,
        },
        key="record_digest",
    )
    atomic_write_json(args.run_dir / "training_record.json", record, overwrite=False)
    completed = {
        "state": "completed",
        "job_digest": job["job_digest"],
        "attempt_digest": attempt["attempt_digest"],
        "anchor_manifest_digest": anchor.manifest_digest,
        "environment_instance_digest": anchor.environment_instance_digest,
        "last_completed_outer": maximum,
        "environment_steps": maximum * per_outer,
        "latest_metrics": latest_metrics,
        "exported_outer_iterations": observed_exports,
        "training_record_digest": record["record_digest"],
        "wall_seconds": wall_seconds,
        "updated_at": finished_at,
    }
    append_jsonl(events_path, {"event": "run_completed", "at": finished_at, **completed})
    atomic_write_json(status_path, completed)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    try:
        run(args)
    except BaseException as error:
        trace = traceback.format_exc()
        try:
            args.run_dir.mkdir(parents=True, exist_ok=True)
            trace_path = args.run_dir / "traceback.txt"
            if not trace_path.exists():
                atomic_write_bytes(trace_path, trace.encode("utf-8"), overwrite=False)
            previous: dict[str, Any] = {}
            status_path = args.run_dir / "status.json"
            if status_path.is_file():
                try:
                    previous = load_strict_json(status_path)
                except ContractError:
                    previous = {}
            failed = {
                **previous,
                "state": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback_file": "traceback.txt",
                "updated_at": utc_now(),
            }
            atomic_write_json(status_path, failed)
            append_jsonl(
                args.run_dir / "events.jsonl",
                {"event": "run_failed", "at": utc_now(), **failed},
            )
        except BaseException:
            print("failed to persist runner exception metadata", file=sys.stderr)
            traceback.print_exc()
        print(trace, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
