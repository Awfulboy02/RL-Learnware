"""Dependency-light end-to-end smoke checks for the v0 control flow.

This module deliberately does not train an encoder or execute a real policy.
It exercises the immutable data path, the episode-balanced KME/RKME path, the
public/private pool boundary, nearest-spec retrieval, and selected-only
deployment with deterministic synthetic inputs.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .config import ProbeConfig
from .envs.base import SyntheticEnvAdapter
from .evaluation.deployment import deploy_selected
from .hashing import sha256_json
from .pool.learnware import (
    LearnwarePool,
    SelectorEntry,
    SelectorTaskSpec,
    load_public_pool,
    save_public_pool,
)
from .pool.registry import DeploymentRegistry, RegistryRecord
from .probe.collector import collect_probe_episodes
from .probe.seed_plan import SeedPlan
from .representation.canonicalizer import TransitionCanonicalizer
from .representation.normalization import fit_normalizer
from .reuse.selector import NearestSpecSelector
from .rkme.empirical import build_empirical_kme
from .rkme.gaussian import GaussianKernel
from .rkme.reducer import ReducerConfig, reduce_kme


_FORBIDDEN_SELECTOR_FIELDS = (
    "task_name",
    "algorithm",
    "training_seed",
    "outer_iteration",
    "checkpoint_evaluation",
    "championization_return",
    "target_return",
    "oracle_return",
)


@dataclass(frozen=True)
class SmokeResult:
    passed: bool
    checks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": self.checks}


class _FakePolicy:
    observation_dim = 3
    action_dim = 2

    def act(
        self, observation: Any, key: Any, *, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]:
        del observation, deterministic
        return np.zeros(self.action_dim, dtype=np.float32), key


def _synthetic_probe_and_pack() -> dict[str, Any]:
    """Exercise probe collection and the exact 109-column layout."""

    probe_config = ProbeConfig(
        type="clipped_gaussian", sigma=1.0, action_low=-1.0, action_high=1.0
    )
    seed_plan = SeedPlan(20260811)
    adapters = {
        "SyntheticFinger": SyntheticEnvAdapter(
            task="SyntheticFinger", observation_dim=9, action_dim=2, horizon=4
        ),
        "SyntheticWalker": SyntheticEnvAdapter(
            task="SyntheticWalker", observation_dim=24, action_dim=6, horizon=4
        ),
    }
    datasets = {
        task: collect_probe_episodes(
            adapter,
            "encoder_train",
            (0, 1),
            probe_config,
            seed_plan=seed_plan,
            task_index=index,
        )
        for index, (task, adapter) in enumerate(adapters.items())
    }
    repeated = collect_probe_episodes(
        adapters["SyntheticFinger"],
        "encoder_train",
        (0, 1),
        probe_config,
        seed_plan=seed_plan,
        task_index=0,
    )
    if repeated.digest != datasets["SyntheticFinger"].digest:
        raise AssertionError("fixed probe seeds did not reproduce the same dataset")

    schemas = {task: adapter.schema for task, adapter in adapters.items()}
    stats = fit_normalizer(
        datasets,
        schemas,
        max_observation_dim=24,
        role="source",
    )
    canonicalizer = TransitionCanonicalizer(stats=stats, max_action_dim=6)
    packed = {
        task: canonicalizer.pack(dataset, schemas[task])
        for task, dataset in datasets.items()
    }
    widths = {task: value.packed_dim for task, value in packed.items()}
    if set(widths.values()) != {109}:
        raise AssertionError(f"unexpected canonical widths: {widths}")
    return {
        "dataset_reproducible": True,
        "packed_widths": widths,
        "episode_counts": {
            task: dataset.episode_count for task, dataset in datasets.items()
        },
    }


def _semantic_dataset(
    center: np.ndarray,
    *,
    seed: int,
    episodes: int = 4,
    transitions_per_episode: int = 4,
) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    count = episodes * transitions_per_episode
    points = center[None, :] + rng.normal(0.0, 0.025, (count, center.size))
    offsets = np.arange(
        0, count + 1, transitions_per_episode, dtype=np.int64
    )
    return SimpleNamespace(points=points, episode_offsets=offsets)


def _synthetic_pool_and_queries() -> tuple[
    LearnwarePool,
    DeploymentRegistry,
    list[Any],
    dict[str, Any],
]:
    """Build six independent TaskSpecs and retrieve independent target draws."""

    protocol_id = sha256_json({"protocol": "policy-learnware-v0-logic-smoke"})
    kernel = GaussianKernel(0.75)
    latent_dim = 8
    entries: list[SelectorEntry] = []
    records: list[RegistryRecord] = []
    targets: list[Any] = []
    expected_ids: list[str] = []
    reduction_errors: list[float] = []

    for task_index in range(6):
        center = np.zeros(latent_dim, dtype=np.float64)
        center[task_index] = 2.5
        source = build_empirical_kme(
            _semantic_dataset(center, seed=100 + task_index),
            kernel,
            protocol_id=protocol_id,
            dataset_digest=sha256_json({"source": task_index}),
        )
        reduced = reduce_kme(
            source,
            ReducerConfig(
                support_budget=4,
                support_steps=8,
                learning_rate=5.0e-3,
                ridge=1.0e-6,
                kmeans_steps=8,
            ),
        )
        opaque_id = "lw-" + sha256_json({"entry": task_index})[:20]
        task_spec = SelectorTaskSpec.from_rkme(
            reduced,
            protocol_id=protocol_id,
            kernel_bandwidth=kernel.bandwidth,
        )
        reduction_errors.append(float(reduced.reduction_error))
        entries.append(SelectorEntry(opaque_id, protocol_id, task_spec))
        records.append(
            RegistryRecord(
                opaque_id=opaque_id,
                protocol_id=protocol_id,
                policy_bundle=Path(f"/immutable/smoke/{opaque_id}"),
                policy_bundle_digest=sha256_json({"bundle": task_index}),
                native_observation_dim=3,
                native_action_dim=2,
                source_task=f"private-source-{task_index}",
                provenance={"test_only": True},
            )
        )
        targets.append(
            build_empirical_kme(
                _semantic_dataset(center, seed=10_000 + task_index),
                kernel,
                protocol_id=protocol_id,
                dataset_digest=sha256_json({"target": task_index}),
            )
        )
        expected_ids.append(opaque_id)

    pool = LearnwarePool(
        pool_id="policy-learnware-v0-logic-smoke",
        protocol_id=protocol_id,
        kernel_bandwidth=kernel.bandwidth,
        entries=tuple(entries),
    )
    pool.validate_expected_size(6)
    registry = DeploymentRegistry(
        tuple(records),
        pool_id=pool.pool_id,
        pool_digest=sha256_json(pool.public_manifest()),
    )
    registry.validate_against(pool)

    with tempfile.TemporaryDirectory() as directory:
        artifact = save_public_pool(pool, Path(directory) / "public_pool")
        reloaded = load_public_pool(artifact)
        if reloaded.public_manifest() != pool.public_manifest():
            raise AssertionError("public pool changed after fresh-process-style reload")

    public_json = json.dumps(pool.public_manifest(), sort_keys=True)
    leaked = [field for field in _FORBIDDEN_SELECTOR_FIELDS if field in public_json]
    if leaked:
        raise AssertionError(f"selector-visible manifest leaked fields: {leaked}")

    selector = NearestSpecSelector(pool)
    selections = []
    for index, (target, expected_id) in enumerate(zip(targets, expected_ids, strict=True)):
        result = selector.select(
            target,
            target_dataset_digest=target.dataset_digest,
            probe_episode_count=target.episode_count,
            probe_steps=target.transition_count,
        )
        if result.selected_opaque_id != expected_id:
            raise AssertionError(
                f"synthetic exact-recurrent retrieval {index} selected "
                f"{result.selected_opaque_id}, expected {expected_id}"
            )
        selections.append(result)

    return pool, registry, selections, {
        "retrieval_correct": len(selections),
        "retrieval_total": len(targets),
        "public_manifest_has_private_fields": False,
        "pool_reload_verified": True,
        "max_reduction_error": max(reduction_errors),
    }


def _selected_only_deployment(
    registry: DeploymentRegistry, selection: Any
) -> dict[str, Any]:
    load_count = 0

    def loader(_record: RegistryRecord) -> _FakePolicy:
        nonlocal load_count
        load_count += 1
        return _FakePolicy()

    success = deploy_selected(
        selection,
        registry,
        SimpleNamespace(observation_dim=3, action_dim=2),
        policy_loader=loader,
        evaluator=lambda _policy: (1.0, 2.0),
    )
    if not success.deployable or load_count != 1:
        raise AssertionError("compatible deployment did not load exactly one policy")

    failure = deploy_selected(
        selection,
        registry,
        SimpleNamespace(observation_dim=24, action_dim=6),
        policy_loader=loader,
        evaluator=lambda _policy: (999.0,),
    )
    if failure.deployment_failure != "incompatible_native_schema":
        raise AssertionError("incompatible deployment did not fail closed")
    if load_count != 1:
        raise AssertionError("schema failure loaded a candidate policy")
    return {
        "compatible_loaded_policy_count": 1,
        "incompatible_loaded_policy_count": 0,
        "incompatible_status": failure.deployment_failure,
    }


def run_logic_smoke() -> SmokeResult:
    """Run a fast, deterministic, training-free end-to-end logic check."""

    checks: dict[str, Any] = {}
    checks["probe_and_canonicalizer"] = _synthetic_probe_and_pack()
    _, registry, selections, retrieval = _synthetic_pool_and_queries()
    checks["retrieval"] = retrieval
    checks["deployment"] = _selected_only_deployment(registry, selections[0])
    return SmokeResult(passed=True, checks=checks)
