"""Minimal staged runner for the v0.4a fixed-probe BI0 comparison.

The runner is intentionally strict and small.  It adapts the frozen v0.31
Raw-Delta RKME operator to a same-task five-policy endpoint, fits BPR/EBPR
models from source contexts only, and keeps target scoring physically separate
from oracle evaluation::

    prepare -> fit-source -> score-fp -> seal-rankings
            -> oracle-evaluate -> summarize

Existing v02/v03/v0.31 inputs are read-only.  Every output is an immutable file
under a new v0.4a run directory.  Missing lineage/logger/utility evidence is a
recorded NO_GO, never an invitation to synthesize a replacement.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import resource
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from policy_learnware_v0.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
    sha256_ndarrays,
)
from policy_learnware_v0.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_npz,
    read_json,
)
from policy_learnware_v0.policy.bundle import BundleValidationError, validate_bundle
from policy_learnware_v0.policy.evaluate import verify_compiled_policy_parity
from policy_learnware_v0.policy.loader import load_policy
from policy_learnware_v0.rkme.reducer import ReducedRKME
from policy_learnware_v0.v03.source_market import (
    SourceMarketError,
    V03SourcePolicyMarket,
)
from policy_learnware_v0.v03.canonicalization import (
    GlobalCanonicalizerSpec,
    NativeShapeRegistry,
    fit_global_normalizer,
)
from policy_learnware_v0.v04a.bpr import (
    BPRModelError,
    BPRGaussianModel,
    summarize_episode,
    summarize_probe,
)
from policy_learnware_v0.v04a.ebpr import EBPRError, EBPRFixedProbe, TransitionEpisode
from policy_learnware_v0.v04a.metrics import (
    aggregate_metrics,
    evaluate_ranking,
    hierarchical_bootstrap_intervals,
)
from policy_learnware_v0.v04a.protocol import (
    BUDGET_EPISODES,
    BudgetLedger,
    ProbeMembership,
    RankingSeal,
    RewardFreeProbe,
    canonical_tie_token,
    derive_probe_membership,
    seal_rankings,
    tie_break_key,
    verify_ranking_seal,
)
from server.repro_fpo_ppo_v03.development_baseline_runner import (
    SCHEMA as V03_BASELINE_SCHEMA,
    DevelopmentBaselineError,
    _context_rows,
    _distance,
    _empirical,
    _market,
)
from server.repro_fpo_ppo_v03.signal_bank_runner import (
    SignalBankRunnerError,
    _load_arrays as _load_signal_arrays,
    _load_index as _load_signal_index,
    _source_partition as _v031_source_partition,
    _subset as _signal_subset,
)
from server.repro_fpo_ppo_v02.provenance import (
    ContractError as V02ContractError,
    load_strict_json,
    validate_self_digest,
    validate_success_record,
)
from server.repro_fpo_ppo_v02.pool_acceptance import accept_policy_pool


SCHEMA = "policy-learnware.v04a-fixed-probe-run.v1"
PLAN_SHA256 = "d1860c1418fe807bf640e9cfb8a816b7f58e8797db76345af212796fa6d487c0"
RAW_METHOD = "RAW_DELTA_TASK5"
BPR_METHOD = "BPR_FP"
EBPR_METHOD = "EBPR_FP"
HYBRID_METHOD = "EBPR_FP_BPR_U"
PRIMARY_METHODS = (RAW_METHOD, BPR_METHOD, EBPR_METHOD)
ALL_FP_METHODS = (*PRIMARY_METHODS, HYBRID_METHOD)
CORE_ARRAYS = (
    "observation",
    "action",
    "reward",
    "next_observation",
    "terminated",
    "truncated",
    "episode_offsets",
)
LOGGER_FIELDS = (
    "v02_config_digest",
    "cp0_draft_hash",
    "frozen_probe_protocol_id",
    "probe_type",
    "probe_rng_backend",
    "probe_sigma",
    "seed_namespace",
    "episode_count",
    "steps_per_episode",
)

# The v02 exporter hard-gated its same-runtime golden replay at 1e-6.  The
# frozen v03 source-market later showed that those float32 action bytes are not
# portable across otherwise valid JAX/XLA backends.  The frozen v03 evidence
# reached 2.27e-2 before tanh and 5.39e-3 after tanh.  A later read-only replay
# of all 30 selected policies on the original m2 host's CPU reached 6.43e-2
# and 2.86e-2 respectively, while every deterministic/transform/key invariant
# passed and scalar/compiled error remained below 7.75e-7.  Keep the origin
# receipt and current compiled path strict; this rounded envelope applies only
# to stored-golden cross-backend diagnostics, which do not enter BI0 ranking.
ORIGIN_PARITY_ATOL = 1.0e-6
ORIGIN_PARITY_RTOL = 1.0e-6
COMPILED_PARITY_SAMPLE_COUNT = 2
CROSS_BACKEND_PARITY = {
    "version": "v04a-selected-market-f32-m2cpu-v2",
    "dtype": "float32",
    "scope": "stored-golden compatibility guard; values never enter ranking or identity",
    "raw_atol": 7.0e-2,
    "environment_atol": 3.0e-2,
    "rtol": 1.0e-5,
    "v03_evidence_file_count": 70,
    "v03_evidence_aggregate_sha256": (
        "7892455fae56637dbc44c0bdd969cfc7c7182ec67af5a4db3b71b1d961911089"
    ),
    "v03_observed_raw_max_abs_error": 2.269744873046875e-2,
    "v03_observed_environment_max_abs_error": 5.387336015701294e-3,
    "m2_cpu_evidence_sha256": (
        "50ac5e13b021a415ab251f51672fabb61a334e79ae25f94e95d63c35a8f9fc46"
    ),
    "m2_cpu_evidence_candidate_count": 30,
    "m2_cpu_evidence_commit": "de184808bc83fdeaba9ed81bdf548e364345402a",
    "observed_raw_max_abs_error": 6.4258873462677e-2,
    "observed_environment_max_abs_error": 2.8588712215423584e-2,
    "observed_compiled_max_abs_error": 7.748603820800781e-7,
}


class V04ARunnerError(RuntimeError):
    """The requested stage cannot preserve the v0.4a evidence contract."""


class GateFailure(V04ARunnerError):
    """A fail-closed asset or evidence gate."""

    def __init__(self, status: str, message: str, *, details: Any = None):
        super().__init__(message)
        self.status = status
        self.details = details


def _json(path: Path) -> Mapping[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise V04ARunnerError(f"expected one JSON object: {path}")
    return value


def _publish(path: Path, value: Any, *, resume: bool = False) -> str:
    """Publish canonical JSON once, or byte-verify an explicit resume."""

    expected = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if (
            not resume
            or path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != expected
        ):
            raise V04ARunnerError(f"immutable artifact already differs: {path}")
        return sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_json(path, value)


def _publish_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, resume: bool = False
) -> str:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    if path.exists():
        if (
            not resume
            or path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != payload
        ):
            raise V04ARunnerError(f"immutable artifact already differs: {path}")
        return sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_bytes(path, payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise V04ARunnerError(f"JSONL artifact is absent or unsafe: {path}")
    result: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise V04ARunnerError(f"non-object at {path}:{line_number}")
        result.append(dict(value))
    if not result:
        raise V04ARunnerError(f"empty JSONL artifact: {path}")
    return result


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise V04ARunnerError(f"configuration is absent or unsafe: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise V04ARunnerError("v0.4a configuration must be one mapping")
    config = dict(value)
    required = {
        "schema": "policy-learnware.v04a-config.v0",
        "version": "0.4a.0",
        "plan_sha256": PLAN_SHA256,
        "scope": "development",
        "candidate_scope": "TASK_5",
        "budgets": list(BUDGET_EPISODES),
        "transitions_per_episode": 64,
        "interaction_steps_per_episode": 1000,
        "deferred": [
            "P4_CONTROLS",
            "P4_FULL_TRANSITION_SENSITIVITY",
            "BPR_SEQ",
            "EBPR_SEQ",
            "confirmatory",
        ],
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise V04ARunnerError(f"configuration mismatch at {key!r}")
    split = config.get("source_split")
    if not isinstance(split, Mapping) or [
        split.get("train_episodes"),
        split.get("validation_episodes"),
        split.get("repeat_report_episodes"),
    ] != [19, 6, 7]:
        raise V04ARunnerError("source split must be the frozen 19/6/7 partition")
    utility_split = config.get("source_utility_split")
    if not isinstance(utility_split, Mapping) or [
        utility_split.get("train_episodes"),
        utility_split.get("validation_episodes"),
        utility_split.get("repeat_report_episodes"),
    ] != [30, 10, 10]:
        raise V04ARunnerError(
            "source utility split must be the frozen 30/10/10 seed partition"
        )
    utility_seeds = config.get("source_utility_seed_namespace")
    if not isinstance(utility_seeds, Mapping) or dict(utility_seeds) != {
        "reset_seed_start": 730000,
        "episode_count": 50,
        "policy_seed_offset": 1_000_003,
    }:
        raise V04ARunnerError("source utility seed namespace differs from v03 freeze")
    controls = config.get("controls")
    if (
        not isinstance(controls, Mapping)
        or controls.get("enable_hybrid") is not True
        or not isinstance(controls.get("epsilon_optimal"), (int, float))
        or not math.isfinite(float(controls["epsilon_optimal"]))
        or float(controls["epsilon_optimal"]) < 0.0
        or not isinstance(controls.get("posterior_confidence_threshold"), (int, float))
        or not 0.0 < float(controls["posterior_confidence_threshold"]) < 1.0
        or isinstance(controls.get("hierarchical_bootstrap_replicates"), bool)
        or not isinstance(controls.get("hierarchical_bootstrap_replicates"), int)
        or int(controls["hierarchical_bootstrap_replicates"]) < 100
        or isinstance(controls.get("hierarchical_bootstrap_seed"), bool)
        or not isinstance(controls.get("hierarchical_bootstrap_seed"), int)
        or int(controls["hierarchical_bootstrap_seed"]) < 0
    ):
        raise V04ARunnerError("v0.4a controls are incomplete or invalid")
    return config, sha256_json(config)


def _assert_new_output_root(output: Path, inputs: Iterable[Path]) -> None:
    resolved = output.expanduser().resolve()
    for raw in inputs:
        source = raw.expanduser().resolve()
        if (
            resolved == source
            or source in resolved.parents
            or resolved in source.parents
        ):
            raise V04ARunnerError(
                f"v0.4a output must be disjoint from frozen input {source}"
            )
    if resolved.exists():
        if resolved.is_symlink() or not resolved.is_dir():
            raise V04ARunnerError(f"unsafe output root: {resolved}")
        if any(resolved.iterdir()):
            raise V04ARunnerError(
                "prepare requires a new empty v0.4a directory; use later stages to resume"
            )
    else:
        resolved.mkdir(parents=True)


def _load_bank(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = Path(str(row["npz"]))
    if path.is_symlink() or not path.is_file():
        raise GateFailure("NO_GO_MISSING_ASSET", f"missing/unsafe bank: {path}")
    if (
        row.get("bank_npz_sha256") is not None
        and sha256_file(path) != row["bank_npz_sha256"]
    ):
        raise GateFailure("NO_GO_ASSET_DIGEST_MISMATCH", f"bank digest differs: {path}")
    with np.load(path, allow_pickle=False) as payload:
        missing = set(CORE_ARRAYS) - set(payload.files)
        if missing:
            raise GateFailure(
                "NO_GO_MISSING_ASSET",
                f"{row['context_id']}: bank lacks {sorted(missing)}",
            )
        arrays = {name: np.asarray(payload[name]) for name in CORE_ARRAYS}
    offsets = np.asarray(arrays["episode_offsets"], dtype=np.int64)
    if (
        offsets.shape != (33,)
        or offsets[0] != 0
        or not np.all(np.diff(offsets) == 1000)
        or offsets[-1] != 32_000
    ):
        raise GateFailure(
            "NO_GO_LOGGER_MISMATCH",
            f"{row['context_id']}: bank is not exactly 32x1000",
        )
    rows = int(offsets[-1])
    if (
        arrays["observation"].ndim != 2
        or arrays["next_observation"].shape != arrays["observation"].shape
        or arrays["action"].ndim != 2
        or arrays["action"].shape[0] != rows
        or arrays["observation"].shape[0] != rows
        or any(
            arrays[name].shape != (rows,)
            for name in CORE_ARRAYS[2:3] + CORE_ARRAYS[4:6]
        )
    ):
        raise GateFailure(
            "NO_GO_LOGGER_MISMATCH",
            f"{row['context_id']}: transition arrays are not aligned",
        )
    numeric = (arrays["observation"], arrays["action"], arrays["next_observation"])
    if any(not np.all(np.isfinite(value)) for value in numeric):
        raise GateFailure(
            "NO_GO_LOGGER_MISMATCH", f"{row['context_id']}: non-finite transition"
        )
    return arrays


def _reward_free_projection(
    row: Mapping[str, Any], membership: ProbeMembership
) -> RewardFreeProbe:
    """Read only (s,a,s',offsets) and freeze the complete 32x64 BI0 view."""

    path = Path(str(row["npz"]))
    if path.is_symlink() or not path.is_file():
        raise GateFailure("NO_GO_MISSING_ASSET", f"missing/unsafe bank: {path}")
    if (
        row.get("bank_npz_sha256") is not None
        and sha256_file(path) != row["bank_npz_sha256"]
    ):
        raise GateFailure(
            "NO_GO_ASSET_DIGEST_MISMATCH",
            f"bank changed between census and projection: {path}",
        )
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "observation",
            "action",
            "next_observation",
            "episode_offsets",
        }
        if not required.issubset(payload.files):
            raise GateFailure(
                "NO_GO_LOGGER_MISMATCH",
                f"{row['context_id']}: reward-free transition fields are incomplete",
            )
        observation = np.asarray(payload["observation"])
        action = np.asarray(payload["action"])
        next_observation = np.asarray(payload["next_observation"])
        offsets = np.asarray(payload["episode_offsets"], dtype=np.int64)
    if offsets.shape != (33,) or not np.all(np.diff(offsets) == 1000):
        raise GateFailure(
            "NO_GO_LOGGER_MISMATCH",
            f"{row['context_id']}: reward-free bank is not 32x1000",
        )
    return RewardFreeProbe.from_full_episodes(
        observation,
        action,
        next_observation,
        membership=membership,
        budget_episodes=32,
    )


def _publish_probe(path: Path, probe: RewardFreeProbe) -> str:
    return atomic_write_npz(
        path,
        {
            "observation": probe.observation,
            "action": probe.action,
            "next_observation": probe.next_observation,
            "episode_offsets": probe.episode_offsets,
            "probe_membership_digest": np.asarray(probe.probe_membership_digest),
        },
    )


def _projected_probe(
    run_dir: Path, row: Mapping[str, Any], budget: int
) -> RewardFreeProbe:
    """Load only the sanitized reward-free projection produced by prepare."""

    path = run_dir / str(row["reward_free_npz"])
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != row["reward_free_npz_sha256"]
    ):
        raise V04ARunnerError(
            f"reward-free scoring projection moved or changed: {row['context_id']}"
        )
    with np.load(path, allow_pickle=False) as payload:
        expected_arrays = {
            "observation",
            "action",
            "next_observation",
            "episode_offsets",
            "probe_membership_digest",
        }
        if set(payload.files) != expected_arrays:
            raise V04ARunnerError(
                f"reward-free projection exposes unexpected channels: {row['context_id']}"
            )
        observation = np.asarray(payload["observation"])
        action = np.asarray(payload["action"])
        next_observation = np.asarray(payload["next_observation"])
        offsets = np.asarray(payload["episode_offsets"], dtype=np.int64)
        membership_digest = str(payload["probe_membership_digest"])
    if membership_digest != row["probe_membership_digest"]:
        raise V04ARunnerError("reward-free projection membership digest differs")
    count = int(budget) * 64
    return RewardFreeProbe(
        observation=observation[:count],
        action=action[:count],
        next_observation=next_observation[:count],
        episode_offsets=offsets[: int(budget) + 1],
        probe_membership_digest=membership_digest,
    )


def _rebuild_v031_raw_adapter(
    context_index: Path, raw_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild the exact source-only v0.31 canonicalizer without target data."""

    _, signal_rows = _load_signal_index(context_index)
    source_rows = tuple(row for row in signal_rows if row["role"] == "source")
    train_native = []
    validation_native = []
    for row in source_rows:
        arrays = _load_signal_arrays(row)
        train_ep, validation_ep, _, _ = _v031_source_partition(row, arrays, 0.6, 0.2, 0)
        train_bank, _, _ = _signal_subset(
            row,
            arrays,
            train_ep,
            bank_id=f"{row['bank_id']}-train",
            role="source_representation_train",
        )
        validation_bank, _, _ = _signal_subset(
            row,
            arrays,
            validation_ep,
            bank_id=f"{row['bank_id']}-validation",
            role="source_representation_validation",
        )
        train_native.append(train_bank)
        validation_native.append(validation_bank)
    registry = NativeShapeRegistry.from_source_banks(
        (*train_native, *validation_native)
    )
    normalizer = fit_global_normalizer((*train_native, *validation_native), registry)
    canonicalizer = GlobalCanonicalizerSpec(registry, normalizer)
    if canonicalizer.canonicalizer_digest != raw_config.get("canonicalizer_digest"):
        raise GateFailure(
            "NO_GO_RAW_PARITY",
            "reconstructed source-only canonicalizer differs from frozen v0.31 config",
        )
    feature_width = registry.max_observation_dim + registry.max_action_dim
    if feature_width != int(raw_config.get("feature_width", -1)):
        raise GateFailure(
            "NO_GO_RAW_PARITY",
            "reconstructed Raw-Delta feature width differs from frozen v0.31 config",
        )
    return {
        "schema": "policy-learnware.v04a-raw-delta-adapter.v1",
        "identity": "V031_SOURCE_ONLY_CANONICALIZER_REPLAY",
        "canonicalizer_digest": canonicalizer.canonicalizer_digest,
        "normalizer_digest": normalizer.normalizer_digest,
        "registry_digest": registry.registry_digest,
        "v031_source_partition": {
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "split_seed": 0,
        },
        "max_observation_dim": registry.max_observation_dim,
        "max_action_dim": registry.max_action_dim,
        "observation_mean": normalizer.observation_mean.tolist(),
        "observation_std": normalizer.observation_std.tolist(),
        "action_mean": normalizer.action_mean.tolist(),
        "action_std": normalizer.action_std.tolist(),
        "tasks": {
            record.task_private_id: {
                "observation_dim": record.observation_dim,
                "action_dim": record.action_dim,
                "native_schema_digest": record.native_schema_digest,
            }
            for record in registry.records
        },
        "numeric_path": "normalize float64 -> canonical TransitionBank float32 -> V_DELTA_ONLY -> float64 RKME",
        "target_rows_read_during_fit": 0,
    }


def _logger_record(row: Mapping[str, Any]) -> dict[str, Any]:
    index_path = Path(str(row["npz"])).parent / "index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise GateFailure(
            "NO_GO_LOGGER_MISMATCH",
            f"{row['context_id']}: per-context logger index is absent",
        )
    record = _json(index_path)
    missing = [name for name in LOGGER_FIELDS if name not in record]
    if missing:
        raise GateFailure(
            "NO_GO_LOGGER_MISMATCH",
            f"{row['context_id']}: logger evidence lacks {missing}",
        )
    if record.get("context_id") != row["context_id"]:
        raise GateFailure(
            "NO_GO_LOGGER_MISMATCH",
            f"{row['context_id']}: logger index names another context",
        )
    return {name: record[name] for name in LOGGER_FIELDS}


def _task_layout(
    rows: Sequence[Mapping[str, Any]], market: V03SourcePolicyMarket
) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["role"] == "source":
            by_task[str(row["task_id"])].append(row)
    if len(by_task) != 6 or any(len(group) != 5 for group in by_task.values()):
        raise GateFailure(
            "NO_GO_TASK5_COVERAGE", "source contexts must be six tasks x five types"
        )
    result: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for task_id, group in sorted(by_task.items()):
        sources = sorted(group, key=lambda row: str(row["context_id"]))
        paired: dict[str, str] = {}
        for row in sources:
            anchor = str(row.get("source_anchor_id", ""))
            opaque = market.anchor_to_opaque_learnware_id.get(anchor)
            if opaque is None:
                raise GateFailure(
                    "NO_GO_TASK5_COVERAGE",
                    f"{row['context_id']}: source anchor is absent from policy market",
                )
            paired[str(row["context_id"])] = opaque
            seen.add(opaque)
        candidates = tuple(sorted(paired.values()))
        if len(candidates) != 5:
            raise GateFailure(
                "NO_GO_TASK5_COVERAGE", f"{task_id}: TASK_5 is not unique"
            )
        result[task_id] = {
            "source_rows": sources,
            "paired_policy_by_type": paired,
            "candidate_ids": candidates,
        }
    if seen != set(market.entries):
        raise GateFailure(
            "NO_GO_TASK5_COVERAGE",
            "TASK_5 blocks do not partition the 30-policy market",
        )
    return result


def _raw_root(root: Path) -> Path:
    direct = root / "config.json"
    nested = root / "views" / "delta_action" / "config.json"
    if direct.is_file() and root.name == "delta_action":
        return root
    if nested.is_file():
        return nested.parent
    raise GateFailure(
        "NO_GO_MISSING_ASSET", f"Raw-Delta config/source artifacts absent under {root}"
    )


def _origin_pool_acceptance(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Revalidate the immutable v02 84-direct + 6-promotion pool once."""

    handoff_root = path.parent
    experiment_root = handoff_root.parent
    promotion_path = handoff_root / "compiled_parity_promotions.json"
    plan_path = (
        experiment_root / "training_private" / "plans" / "server_training_plan.json"
    )
    runs_root = experiment_root / "training_private" / "server_runs"
    files = (path, promotion_path, plan_path)
    if (
        any(item.is_symlink() or not item.is_file() for item in files)
        or runs_root.is_symlink()
        or not runs_root.is_dir()
        or (path.stat().st_mode & 0o777) != 0o444
        or (promotion_path.stat().st_mode & 0o777) != 0o444
    ):
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            "frozen v02 policy-pool acceptance authority is absent or writable",
        )
    try:
        stored = load_strict_json(path)
        validate_self_digest(
            stored, key="report_digest", where="policy-pool acceptance"
        )
        promotion = load_strict_json(promotion_path)
        plan = load_strict_json(plan_path)
        recomputed = accept_policy_pool(
            server_plan=plan,
            runs_root=runs_root,
            promotion_manifest=promotion,
        )
    except (OSError, KeyError, V02ContractError, ValueError) as error:
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            f"cannot revalidate frozen v02 policy-pool acceptance: {error}",
        ) from error
    ignored = {"accepted_at", "report_digest"}
    stored_semantics = {
        key: value for key, value in stored.items() if key not in ignored
    }
    recomputed_semantics = {
        key: value for key, value in recomputed.items() if key not in ignored
    }
    if stored_semantics != recomputed_semantics:
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            "stored v02 policy-pool acceptance differs from canonical replay",
        )
    cells = stored.get("cells")
    if not isinstance(cells, Mapping):
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            "v02 policy-pool acceptance has no cell mapping",
        )
    return cells, {
        "status": "PASS",
        "scope": "frozen-v02-development-attestation",
        "policy_pool_acceptance_path": str(path),
        "policy_pool_acceptance_sha256": sha256_file(path),
        "policy_pool_acceptance_report_digest": stored["report_digest"],
        "compiled_parity_promotions_path": str(promotion_path),
        "compiled_parity_promotions_sha256": sha256_file(promotion_path),
        "compiled_parity_promotions_digest": promotion["manifest_digest"],
        "server_training_plan_path": str(plan_path),
        "server_training_plan_sha256": sha256_file(plan_path),
        "canonical_replay": "PASS",
        "direct_terminal_record_count": stored["direct_terminal_record_count"],
        "compiled_parity_fallback_promotion_count": stored[
            "compiled_parity_fallback_promotion_count"
        ],
    }


def _origin_parity_receipt(
    metadata: Any, accepted_cell: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify one bundle against its canonical v02 pool-acceptance cell."""

    attempt_root = metadata.bundle_dir.parents[1]
    job_id = metadata.bundle_dir.parents[2].name
    record_path = attempt_root / "training_record.json"
    status_path = attempt_root / "status.json"
    bundle_root = metadata.bundle_dir.resolve()
    if (
        accepted_cell.get("job_id") != job_id
        or accepted_cell.get("job_digest") != metadata.provenance.get("job_digest")
        or accepted_cell.get("attempt_digest")
        != metadata.provenance.get("attempt_digest")
        or Path(str(accepted_cell.get("bundle_path", ""))).resolve() != bundle_root
        or accepted_cell.get("bundle_digest") != metadata.bundle_digest
        or accepted_cell.get("outer_iteration") != metadata.outer_iteration
        or accepted_cell.get("environment_steps") != metadata.environment_steps
        or accepted_cell.get("seed") != metadata.training_seed
    ):
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            f"policy-pool acceptance cell differs for {metadata.bundle_dir}",
        )
    resolution = accepted_cell.get("resolution")
    if resolution == "compiled_parity_fallback_promotion":
        if (
            status_path.is_symlink()
            or not status_path.is_file()
            or record_path.is_symlink()
            or record_path.exists()
            or _json(status_path).get("state") != "failed"
            or not isinstance(accepted_cell.get("promotion_entry_digest"), str)
            or not isinstance(accepted_cell.get("failure_trace_digest"), str)
        ):
            raise GateFailure(
                "NO_GO_MARKET_OR_TASK5_ABI",
                f"canonical fallback promotion differs for {metadata.bundle_dir}",
            )
        return {
            "status": "PASS",
            "resolution": resolution,
            "training_record_sha256": None,
            "training_record_digest": None,
            "golden_report_digest": accepted_cell["golden_parity_digest"],
            "compiled_report_digest": accepted_cell["compiled_parity_digest"],
            "promotion_entry_digest": accepted_cell["promotion_entry_digest"],
            "failure_trace_digest": accepted_cell["failure_trace_digest"],
            "atol": ORIGIN_PARITY_ATOL,
            "rtol": ORIGIN_PARITY_RTOL,
        }
    if resolution != "direct_terminal_record" or any(
        path.is_symlink() or not path.is_file() for path in (record_path, status_path)
    ):
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            f"origin parity receipt is absent for {metadata.bundle_dir}",
        )
    try:
        record = validate_success_record(
            record_path,
            expected_job_digest=str(metadata.provenance["job_digest"]),
            expected_attempt_digest=str(metadata.provenance["attempt_digest"]),
            expected_anchor_manifest_digest=str(
                metadata.provenance["anchor_manifest_digest"]
            ),
            expected_environment_instance_digest=str(
                metadata.provenance["environment_instance_digest"]
            ),
            expected_config_digest=str(metadata.provenance["config_digest"]),
            expected_execution_purpose=str(metadata.provenance["execution_purpose"]),
        )
    except (KeyError, V02ContractError) as error:
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            f"origin training receipt is invalid for {metadata.bundle_dir}: {error}",
        ) from error
    record_digest = record["record_digest"]
    status = _json(status_path)
    expected_status = "completed" if record["state"] == "succeeded" else "recovered"
    if (
        status.get("state") != expected_status
        or status.get("training_record_digest") != record_digest
        or status.get("promoted_outer_iteration") != metadata.outer_iteration
        or status.get("promoted_environment_steps") != metadata.environment_steps
        or accepted_cell.get("training_record_digest") != record_digest
        or accepted_cell.get("terminal_record_state") != record["state"]
    ):
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            f"origin training status differs for {metadata.bundle_dir}",
        )
    checkpoints = record.get("checkpoint_bundles")
    matching = [
        row
        for row in checkpoints
        if isinstance(row, Mapping)
        and Path(str(row.get("path", ""))).resolve() == bundle_root
    ]
    if len(matching) != 1 or matching[0].get("bundle_digest") != metadata.bundle_digest:
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            f"origin checkpoint identity differs for {metadata.bundle_dir}",
        )
    checkpoint = matching[0]
    golden = checkpoint.get("golden_parity")
    compiled = checkpoint.get("compiled_parity")
    assert isinstance(golden, Mapping) and isinstance(compiled, Mapping)
    if (
        golden.get("passed") is not True
        or golden.get("raw_checked") is not True
        or golden.get("sample_count") != 8
        or float(golden.get("atol", float("nan"))) != ORIGIN_PARITY_ATOL
        or float(golden.get("rtol", float("nan"))) != ORIGIN_PARITY_RTOL
        or compiled.get("passed") is not True
        or compiled.get("next_keys_equal") is not True
        or float(compiled.get("atol", float("nan"))) != ORIGIN_PARITY_ATOL
        or float(compiled.get("rtol", float("nan"))) != ORIGIN_PARITY_RTOL
        or accepted_cell.get("golden_parity_digest") != golden.get("report_digest")
        or accepted_cell.get("compiled_parity_digest") != compiled.get("report_digest")
    ):
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            f"origin parity did not pass the frozen 1e-6 gate for {metadata.bundle_dir}",
        )
    return {
        "status": "PASS",
        "resolution": resolution,
        "training_record_sha256": sha256_file(record_path),
        "training_record_digest": record_digest,
        "golden_report_digest": golden["report_digest"],
        "compiled_report_digest": compiled["report_digest"],
        "atol": ORIGIN_PARITY_ATOL,
        "rtol": ORIGIN_PARITY_RTOL,
    }


def _restore_policy_key(key_data: np.ndarray) -> Any:
    import jax

    wrap = getattr(jax.random, "wrap_key_data", None)
    return wrap(key_data) if wrap is not None else key_data


def _policy_key_data(key: Any) -> np.ndarray:
    import jax

    return np.asarray(jax.device_get(jax.random.key_data(key)))


def _symmetric_relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    scale = np.maximum(
        np.maximum(
            np.abs(actual.astype(np.float64)), np.abs(expected.astype(np.float64))
        ),
        np.finfo(np.float32).eps,
    )
    return float(np.max(delta / scale, initial=0.0))


def _deployment_action_audit(policy: Any, metadata: Any) -> dict[str, Any]:
    """Audit current deterministic deployment semantics without bitwise host affinity."""

    with np.load(metadata.bundle_dir / "golden_io.npz", allow_pickle=False) as archive:
        observation = np.asarray(archive["observation"])
        key_data = np.asarray(archive["prng_key_data"])
        expected_raw = np.asarray(archive["raw_action"])
        expected_environment = np.asarray(archive["environment_action"])
    if any(
        array.dtype != np.dtype(np.float32)
        for array in (observation, expected_raw, expected_environment)
    ):
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            f"golden deployment arrays are not float32 for {metadata.bundle_dir}",
        )
    key = _restore_policy_key(key_data)
    raw_first, raw_next_first = policy.act_raw(observation, key, deterministic=True)
    raw_second, raw_next_second = policy.act_raw(observation, key, deterministic=True)
    environment, environment_next = policy.act(observation, key, deterministic=True)
    actual_raw = np.asarray(raw_first)
    repeated_raw = np.asarray(raw_second)
    actual_environment = np.asarray(environment)
    if (
        actual_raw.shape != expected_raw.shape
        or actual_environment.shape != expected_environment.shape
        or actual_raw.dtype != np.dtype(np.float32)
        or actual_environment.dtype != np.dtype(np.float32)
        or not np.all(np.isfinite(actual_raw))
        or not np.all(np.isfinite(actual_environment))
        or not np.array_equal(actual_raw, repeated_raw)
        or not np.array_equal(
            _policy_key_data(raw_next_first), _policy_key_data(raw_next_second)
        )
        or not np.array_equal(
            _policy_key_data(raw_next_first), _policy_key_data(environment_next)
        )
        or np.any(actual_environment < -1.0)
        or np.any(actual_environment > 1.0)
        or not np.allclose(
            actual_environment,
            np.tanh(actual_raw),
            atol=ORIGIN_PARITY_ATOL,
            rtol=ORIGIN_PARITY_RTOL,
        )
    ):
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            f"current deterministic deployment action invariants failed for {metadata.bundle_dir}",
        )
    compiled = verify_compiled_policy_parity(
        policy,
        observation,
        key_data,
        atol=ORIGIN_PARITY_ATOL,
        rtol=ORIGIN_PARITY_RTOL,
        sample_count=COMPILED_PARITY_SAMPLE_COUNT,
    )
    if not compiled.passed or not compiled.next_keys_equal:
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            f"current scalar/compiled deployment parity failed for {metadata.bundle_dir}",
            details={
                "max_abs_error": compiled.max_abs_error,
                "sample_count": compiled.sample_count,
            },
        )
    raw_delta = np.abs(actual_raw.astype(np.float64) - expected_raw.astype(np.float64))
    environment_delta = np.abs(
        actual_environment.astype(np.float64) - expected_environment.astype(np.float64)
    )
    raw_max_abs = float(np.max(raw_delta, initial=0.0))
    environment_max_abs = float(np.max(environment_delta, initial=0.0))
    cross_backend_compatible = bool(
        np.allclose(
            actual_raw,
            expected_raw,
            atol=float(CROSS_BACKEND_PARITY["raw_atol"]),
            rtol=float(CROSS_BACKEND_PARITY["rtol"]),
        )
        and np.allclose(
            actual_environment,
            expected_environment,
            atol=float(CROSS_BACKEND_PARITY["environment_atol"]),
            rtol=float(CROSS_BACKEND_PARITY["rtol"]),
        )
    )
    if not cross_backend_compatible:
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            f"cross-backend action drift exceeds the frozen compatibility envelope for {metadata.bundle_dir}",
            details={
                "raw_max_abs_error": raw_max_abs,
                "environment_max_abs_error": environment_max_abs,
                "compatibility": CROSS_BACKEND_PARITY,
            },
        )
    exact_golden = bool(
        np.allclose(
            actual_raw,
            expected_raw,
            atol=ORIGIN_PARITY_ATOL,
            rtol=ORIGIN_PARITY_RTOL,
        )
        and np.allclose(
            actual_environment,
            expected_environment,
            atol=ORIGIN_PARITY_ATOL,
            rtol=ORIGIN_PARITY_RTOL,
        )
    )
    try:
        import jax

        runtime = {
            "backend": jax.default_backend(),
            "device_kind": str(jax.devices()[0].device_kind),
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "default_matmul_precision": str(jax.config.jax_default_matmul_precision),
            "default_prng_impl": str(jax.config.jax_default_prng_impl),
        }
    except (AttributeError, ImportError, RuntimeError):
        runtime = {"backend": "unavailable"}
    return {
        "status": "PASS" if exact_golden else "WARNING_CROSS_BACKEND_COMPATIBLE",
        "current_backend_deterministic_repeat": True,
        "deployment_transform_matches": True,
        "compiled_parity": {
            "status": "PASS",
            "max_abs_error": compiled.max_abs_error,
            "next_keys_equal": compiled.next_keys_equal,
            "sample_count": compiled.sample_count,
            "atol": compiled.atol,
            "rtol": compiled.rtol,
        },
        "cross_backend_golden_diagnostic": {
            "exact_origin_tolerance_passed": exact_golden,
            "compatibility_envelope_passed": True,
            "raw_max_abs_error": raw_max_abs,
            "raw_max_relative_error": _symmetric_relative_error(
                actual_raw, expected_raw
            ),
            "environment_max_abs_error": environment_max_abs,
            "environment_max_relative_error": _symmetric_relative_error(
                actual_environment, expected_environment
            ),
            "compatibility": CROSS_BACKEND_PARITY,
        },
        "runtime": runtime,
    }


def inspect_assets(
    *,
    context_index: Path,
    public_policy_market: Path,
    deployment_private_registry: Path,
    origin_pool_acceptance: Path,
    raw_delta_root: Path,
    fpo_root: Path,
    split_seed: int,
) -> tuple[
    tuple[dict[str, Any], ...],
    V03SourcePolicyMarket,
    dict[str, dict[str, Any]],
    dict[str, ProbeMembership],
    dict[str, Any],
]:
    """Read-only census and hard BI0 logger/TASK_5/Raw gates."""

    raw_index = _json(context_index)
    if (
        raw_index.get("v02_config_digest") is None
        or raw_index.get("cp0_draft_hash") is None
    ):
        raise GateFailure(
            "NO_GO_LOGGER_MISMATCH", "merged context index lacks collector lineage"
        )
    rows = tuple(dict(row) for row in _context_rows(context_index))
    sources = [row for row in rows if row["role"] == "source"]
    targets = [row for row in rows if row["role"] != "source"]
    if len(sources) != 30 or len(targets) != 24:
        raise GateFailure(
            "NO_GO_CONTEXT_COVERAGE",
            f"development scope requires 30 source + 24 target, got {len(sources)} + {len(targets)}",
        )
    if any(row["role"] != "development_query" for row in targets):
        raise GateFailure(
            "NO_GO_CONTEXT_COVERAGE",
            "v0.4a development runner refuses confirmatory or extrapolation contexts",
        )
    market = _market(public_policy_market, deployment_private_registry)
    accepted_cells, origin_acceptance = _origin_pool_acceptance(origin_pool_acceptance)
    if fpo_root.is_symlink() or not fpo_root.is_dir():
        raise GateFailure(
            "NO_GO_MARKET_OR_TASK5_ABI",
            f"FPO runtime root is absent or unsafe: {fpo_root}",
        )
    layout = _task_layout(rows, market)
    target_counts = {
        task_id: sum(row["task_id"] == task_id for row in targets) for task_id in layout
    }
    if set(row["task_id"] for row in targets) != set(layout) or any(
        count != 4 for count in target_counts.values()
    ):
        raise GateFailure(
            "NO_GO_CONTEXT_COVERAGE",
            "development contexts must be exactly four per each of the six source tasks",
            details={"development_count_by_task": target_counts},
        )

    logger_values: list[dict[str, Any]] = []
    bank_digests: dict[str, str] = {}
    bank_dimensions: dict[str, tuple[int, int]] = {}
    memberships: dict[str, ProbeMembership] = {}
    for row in rows:
        arrays = _load_bank(row)
        logger_values.append(_logger_record(row))
        bank_digests[str(row["context_id"])] = sha256_file(Path(str(row["npz"])))
        bank_dimensions[str(row["context_id"])] = (
            int(arrays["observation"].shape[1]),
            int(arrays["action"].shape[1]),
        )
        memberships[str(row["context_id"])] = derive_probe_membership(
            str(row["context_id"]), split_seed
        )
    logger_signature = logger_values[0]
    if any(value != logger_signature for value in logger_values[1:]):
        raise GateFailure(
            "NO_GO_LOGGER_MISMATCH",
            "source and target banks do not share one probe/logger configuration",
        )
    if (
        logger_signature["v02_config_digest"] != raw_index["v02_config_digest"]
        or logger_signature["cp0_draft_hash"] != raw_index["cp0_draft_hash"]
        or logger_signature["episode_count"] != 32
        or logger_signature["steps_per_episode"] != 1000
    ):
        raise GateFailure(
            "NO_GO_LOGGER_MISMATCH", "collector lineage or 32x1000 protocol drifted"
        )

    bundle_audit: dict[str, Any] = {}
    abi_by_task: dict[str, set[str]] = defaultdict(set)
    for task_id, task in layout.items():
        for source in task["source_rows"]:
            type_id = str(source["context_id"])
            opaque = task["paired_policy_by_type"][type_id]
            private = market.deployment_private[opaque]
            accepted_cell = accepted_cells.get(private.candidate_id)
            if (
                not isinstance(accepted_cell, Mapping)
                or accepted_cell.get("source_anchor_id") != private.source_anchor_id
            ):
                raise GateFailure(
                    "NO_GO_MARKET_OR_TASK5_ABI",
                    f"policy market differs from accepted v02 pool: {opaque}",
                )
            bundle_path = Path(private.bundle_path).expanduser()
            if not bundle_path.is_absolute():
                bundle_path = deployment_private_registry.parent / bundle_path
            if bundle_path.is_symlink():
                raise GateFailure(
                    "NO_GO_MARKET_OR_TASK5_ABI",
                    f"policy bundle path is a symlink: {opaque}",
                )
            try:
                metadata = validate_bundle(
                    bundle_path,
                    expected_task=task_id,
                    expected_algorithm="fpo",
                    expected_seed=private.seed,
                    expected_outer=private.outer_iteration,
                    expected_environment_steps=private.environment_steps,
                    runtime_only=True,
                )
                origin_parity = _origin_parity_receipt(metadata, accepted_cell)
                policy = load_policy(
                    metadata,
                    fpo_root=fpo_root,
                    runtime_only=True,
                )
                deployment_audit = _deployment_action_audit(policy, metadata)
            except GateFailure:
                raise
            except (
                BundleValidationError,
                ImportError,
                OSError,
                RuntimeError,
                ValueError,
            ) as error:
                raise GateFailure(
                    "NO_GO_MARKET_OR_TASK5_ABI",
                    f"policy bundle validation/parity failed for {opaque}: {error}",
                ) from error
            if metadata.bundle_digest != private.bundle_digest:
                raise GateFailure(
                    "NO_GO_MARKET_OR_TASK5_ABI",
                    f"policy bundle digest differs from private registry: {opaque}",
                )
            if bank_dimensions[type_id] != (
                metadata.observation_dim,
                metadata.action_dim,
            ):
                raise GateFailure(
                    "NO_GO_MARKET_OR_TASK5_ABI",
                    f"source bank and policy native ABI dimensions differ: {opaque}",
                )
            abi_by_task[task_id].add(private.execution_abi.digest)
            bundle_audit[opaque] = {
                "bundle_manifest_digest": metadata.bundle_digest,
                "algorithm": metadata.algorithm,
                "observation_dim": metadata.observation_dim,
                "action_dim": metadata.action_dim,
                "execution_abi_digest": private.execution_abi.digest,
                "bundle_structure_and_normalizer_payload_verified": True,
                "origin_same_runtime_parity": origin_parity,
                "current_deployment_action_audit": deployment_audit,
                "attestation_receipt_digest": private.attestation_receipt_digest,
            }
        if len(abi_by_task[task_id]) != 1:
            raise GateFailure(
                "NO_GO_MARKET_OR_TASK5_ABI",
                f"{task_id}: five candidates do not share one execution ABI",
            )
        expected_dims = {
            bank_dimensions[str(source["context_id"])] for source in task["source_rows"]
        }
        target_dims = {
            bank_dimensions[str(row["context_id"])]
            for row in targets
            if row["task_id"] == task_id
        }
        if len(expected_dims) != 1 or target_dims != expected_dims:
            raise GateFailure(
                "NO_GO_MARKET_OR_TASK5_ABI",
                f"{task_id}: source/target native bank dimensions differ",
            )

    raw_view = _raw_root(raw_delta_root)
    raw_config = dict(_json(raw_view / "config.json"))
    required_raw = {
        "view_id": "V_DELTA_ONLY",
        "source_count": 30,
        "query_count": 24,
        "channels": "o'-o, a",
        "bandwidth_source_key": "opaque_learnware_id",
    }
    for key, expected in required_raw.items():
        if raw_config.get(key) != expected:
            raise GateFailure(
                "NO_GO_RAW_PARITY", f"frozen Raw-Delta config mismatch at {key}"
            )
    raw_run_config_path = raw_view.parents[1] / "run_config.json"
    if raw_run_config_path.is_symlink() or not raw_run_config_path.is_file():
        raise GateFailure(
            "NO_GO_RAW_PARITY",
            "frozen v0.31 run_config.json is absent beside views/delta_action",
        )
    raw_run_config = _json(raw_run_config_path)
    raw_bindings = {
        "context_index_sha256": sha256_file(context_index),
        "public_policy_market_sha256": sha256_file(public_policy_market),
        "deployment_private_registry_sha256": sha256_file(deployment_private_registry),
        "policy_market_id": market.policy_market_id,
    }
    for key, expected in raw_bindings.items():
        if raw_run_config.get(key) != expected:
            raise GateFailure(
                "NO_GO_RAW_PARITY",
                f"v0.31 run binding differs from current frozen input at {key}",
            )
    measurement = raw_run_config.get("measurement")
    if (
        not isinstance(measurement, Mapping)
        or measurement.get("episode_cap") != 16
        or measurement.get("transitions_per_episode_cap") != 64
        or measurement.get("full_collector_npz_preserved") is not True
    ):
        raise GateFailure(
            "NO_GO_RAW_PARITY",
            "v0.31 run does not bind the frozen 16-episode/64-transition measurement",
        )
    raw_sources = raw_view / "source"
    missing_raw = [
        opaque
        for opaque in market.entries
        if not (raw_sources / f"{opaque}.npz").is_file()
        or (raw_sources / f"{opaque}.npz").is_symlink()
    ]
    if missing_raw:
        raise GateFailure(
            "NO_GO_MISSING_ASSET",
            f"frozen Raw-Delta source RKME missing for {len(missing_raw)} policies",
            details={"missing_opaque_ids": missing_raw},
        )
    candidate_task = {
        candidate: task_id
        for task_id, task in layout.items()
        for candidate in task["candidate_ids"]
    }
    raw_source_digests: dict[str, str] = {}
    for opaque in sorted(market.entries):
        path = raw_sources / f"{opaque}.npz"
        try:
            reduced = ReducedRKME.load_npz(path)
        except (OSError, KeyError, ValueError) as error:
            raise GateFailure(
                "NO_GO_RAW_PARITY",
                f"frozen Raw-Delta source RKME is malformed: {opaque}",
            ) from error
        if (
            reduced.protocol_id != raw_config.get("protocol_id")
            or not np.isclose(
                reduced.bandwidth,
                float(raw_config.get("bandwidth", float("nan"))),
                rtol=0.0,
                atol=1.0e-12,
            )
            or reduced.supports.shape[1] != int(raw_config.get("feature_width", -1))
            or reduced.source_task != candidate_task[opaque]
        ):
            raise GateFailure(
                "NO_GO_RAW_PARITY",
                f"frozen Raw-Delta source RKME binding differs: {opaque}",
            )
        raw_source_digests[opaque] = sha256_file(path)
    census = {
        "schema": SCHEMA,
        "stage": "asset-census",
        "status": "PASS",
        "formal": False,
        "context_index_sha256": sha256_file(context_index),
        "source_context_count": len(sources),
        "development_context_count": len(targets),
        "policy_count": len(market.entries),
        "task_count": len(layout),
        "candidates_per_task": 5,
        "bank_shape": [32, 1000],
        "bank_digests": bank_digests,
        "logger_equivalence": {
            "status": "PASS",
            "evidence": "common collector lineage plus per-context logger indices",
            "signature": logger_signature,
        },
        "policy_bundle_and_abi": {
            "status": "PASS",
            "runtime_only_bundle_validation": True,
            "origin_pool_acceptance": origin_acceptance,
            "origin_same_runtime_golden_parity": "digest-bound PASS at 1e-6",
            "current_backend_deployment_determinism": "hard-gated",
            "cross_backend_float32_golden_replay": (
                "versioned compatibility diagnostic; never an asset identity gate"
            ),
            "cross_backend_parity_version": CROSS_BACKEND_PARITY,
            "candidate_count": len(bundle_audit),
            "task_abi_digests": {
                task: next(iter(values)) for task, values in sorted(abi_by_task.items())
            },
            "candidates": bundle_audit,
        },
        "raw_delta": {
            "config_sha256": sha256_file(raw_view / "config.json"),
            "run_config_sha256": sha256_file(raw_run_config_path),
            "protocol_id": raw_config["protocol_id"],
            "canonicalizer_digest": raw_config["canonicalizer_digest"],
            "bandwidth": raw_config["bandwidth"],
            "source_rkme_count": 30,
            "source_rkme_sha256": raw_source_digests,
        },
        "source_utility": "NOT_YET_BOUND; fit-source must pass 5x5 source-only gate",
        "sequential_per_episode_evidence": "NOT_REQUIRED_FOR_BI0; BI1_DEFERRED",
    }
    return rows, market, layout, memberships, census


def _method_cards(config_digest: str) -> dict[str, Any]:
    common = {
        "access_track": "BI0-FP-RF",
        "candidate_scope": "TASK_5",
        "target_signal": "candidate-independent fixed probe (s,a,s'); reward/done removed",
        "prior": "uniform over five source dynamics types",
        "target_reward_access": False,
        "candidate_rollout_access": False,
        "ranking_before_oracle": True,
        "tie_token": canonical_tie_token(config_digest),
        "acquisition_rule": "none; BI0 is a non-adaptive fixed probe",
        "code_source": "in-repository v0.4a clean-room adapter over frozen v03 primitives",
        "runtime_accounting": "method-specific adapter plus score with warm-cached source models; shared probe-file load reported separately",
    }
    return {
        "schema": SCHEMA,
        "cards": {
            RAW_METHOD: {
                **common,
                "identity": "NATIVE_OPERATOR / NEW_SCOPE",
                "prior": None,
                "tie_token": None,
                "tie_token_source": "per-candidate frozen v0.31 public-market tie_break_token",
                "code_source": "frozen in-repository v0.31 Raw-Delta RKME numeric operator",
                "source_privilege": "v0.31 source Reduced RKME and source-only canonicalizer/kernel",
                "likelihood": None,
                "decision": "minimum frozen Raw-Delta Gaussian-kernel MMD within TASK_5",
                "deviation": "only TASK_5 filtering and unified rows; numeric operator unchanged",
                "upstream": "v0.31 Raw-Delta RKME",
                "tie_rule": "frozen v0.31 public market tie token",
            },
            BPR_METHOD: {
                **common,
                "identity": "FIXED_PROBE_ADAPTER",
                "source_privilege": "source type labels, Gaussian summary model, 5x5 source utility",
                "likelihood": "source-validation-calibrated diagonal Gaussian episode summaries",
                "target_predictive_nll_unit": "per probe episode; summary dimension is task-dependent, so do not compare absolute values across tasks/methods",
                "utility": "posterior expected source normalized return",
                "decision": "argmax expected utility",
                "deviation": "candidate-generated BPR observation replaced by fixed probe",
                "upstream": "F02 Bayesian Policy Reuse",
            },
            EBPR_METHOD: {
                **common,
                "identity": "FIXED_PROBE_ADAPTER",
                "source_privilege": "source type labels, conditional forward models, paired policy mapping",
                "likelihood": "calibrated p(delta s | s,a), normalized per valid state dimension",
                "target_predictive_nll_unit": "per visible transition per valid native state dimension",
                "utility": None,
                "decision": "posterior MAP source type -> paired policy",
                "deviation": "candidate-generated transitions replaced by fixed probe; new-policy learning off",
                "upstream": "F12 Efficient Bayesian Policy Reuse",
            },
            HYBRID_METHOD: {
                **common,
                "identity": "INSPIRED_HYBRID",
                "source_privilege": "EBPR forward likelihood plus BPR 5x5 source utility",
                "likelihood": "same frozen EBPR likelihood",
                "target_predictive_nll_unit": "per visible transition per valid native state dimension",
                "utility": "posterior expected source normalized return",
                "decision": "argmax expected utility",
                "deviation": "not a native full algorithm from either paper",
                "upstream": "F12 likelihood + F02 utility",
            },
        },
        "deferred": {
            "P4_CONTROLS": "deferred after core-MVP scope freeze; requires a separately sealed control grid",
            "P4_FULL_TRANSITION_SENSITIVITY": "deferred after core-MVP scope freeze; requires independent 32x1000 source fits and artifacts",
            "BPR_SEQ": "P5 conditional supplement; not part of this MVP",
            "EBPR_SEQ": "P5 conditional supplement; not part of this MVP",
            "confirmatory": "P6 requires separately approved fresh sealed contexts",
        },
    }


def _available_input_metadata(
    *,
    context_index: Path,
    public_policy_market: Path,
    deployment_private_registry: Path,
    origin_pool_acceptance: Path,
    raw_delta_root: Path,
) -> dict[str, Any]:
    """Describe only verifiably present inputs for a failed local census."""

    paths = {
        "context_index": context_index,
        "public_policy_market": public_policy_market,
        "deployment_private_registry": deployment_private_registry,
        "origin_pool_acceptance": origin_pool_acceptance,
    }
    result: dict[str, Any] = {
        name: {
            "path": str(path.resolve()),
            "present_regular_file": path.is_file() and not path.is_symlink(),
            "sha256": (
                sha256_file(path) if path.is_file() and not path.is_symlink() else None
            ),
        }
        for name, path in paths.items()
    }
    result["raw_delta_root"] = {
        "path": str(raw_delta_root.resolve()),
        "present_directory": raw_delta_root.is_dir()
        and not raw_delta_root.is_symlink(),
    }
    try:
        view = _raw_root(raw_delta_root)
        config = _json(view / "config.json")
        result["raw_delta_root"].update(
            {
                "resolved_view_root": str(view.resolve()),
                "config_sha256": sha256_file(view / "config.json"),
                "view_id": config.get("view_id"),
                "source_count": config.get("source_count"),
                "query_count": config.get("query_count"),
                "protocol_id": config.get("protocol_id"),
                "source_rkme_count_present": len(
                    tuple((view / "source").glob("*.npz"))
                ),
            }
        )
    except (GateFailure, OSError, ValueError):
        result["raw_delta_root"]["resolved_view_root"] = None
    return result


def _record_prepare_failure(
    args: argparse.Namespace,
    config_digest: str,
    error: Exception,
    *,
    status: str | None = None,
) -> Mapping[str, Any]:
    required_files = (
        args.context_index,
        args.public_policy_market,
        args.deployment_private_registry,
        args.origin_pool_acceptance,
    )
    resolved_status = status
    if resolved_status is None:
        resolved_status = (
            "NO_GO_REQUIRED_ASSETS_ABSENT"
            if any(not path.is_file() for path in required_files)
            or not args.raw_delta_root.is_dir()
            else "NO_GO_ASSET_INVALID"
        )
    failed = {
        "schema": SCHEMA,
        "stage": "prepare",
        "status": resolved_status,
        "formal": False,
        "message": str(error),
        "details": getattr(error, "details", None),
        "config_digest": config_digest,
        "plan_sha256": PLAN_SHA256,
        "available_input_metadata": _available_input_metadata(
            context_index=args.context_index,
            public_policy_market=args.public_policy_market,
            deployment_private_registry=args.deployment_private_registry,
            origin_pool_acceptance=args.origin_pool_acceptance,
            raw_delta_root=args.raw_delta_root,
        ),
    }
    _publish(args.run_dir / "asset_census.json", failed)
    _publish(args.run_dir / "run.json", failed)
    return failed


def _materialize_sanitized_assets(
    *,
    run_dir: Path,
    context_index: Path,
    raw_delta_root: Path,
    rows: Sequence[Mapping[str, Any]],
    memberships: Mapping[str, ProbeMembership],
    raw_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Publish role-separated reward-free signals and the source-only Raw operator."""

    raw_view = _raw_root(raw_delta_root)
    raw_adapter = _rebuild_v031_raw_adapter(
        context_index, _json(raw_view / "config.json")
    )
    operator_relative = Path("raw_source_operator")
    config_source = raw_view / "config.json"
    atomic_write_bytes(
        run_dir / operator_relative / "config.json", config_source.read_bytes()
    )
    expected_sources = raw_binding.get("source_rkme_sha256")
    if not isinstance(expected_sources, Mapping):
        raise GateFailure(
            "NO_GO_RAW_PARITY", "Raw source digest binding is missing during projection"
        )
    for candidate, expected_digest in sorted(expected_sources.items()):
        source = raw_view / "source" / f"{candidate}.npz"
        if (
            source.is_symlink()
            or not source.is_file()
            or sha256_file(source) != expected_digest
        ):
            raise GateFailure(
                "NO_GO_RAW_PARITY",
                f"Raw source changed between census and source-only projection: {candidate}",
            )
        atomic_write_bytes(
            run_dir / operator_relative / "source" / source.name,
            source.read_bytes(),
        )

    sanitized_contexts: list[dict[str, Any]] = []
    for row in rows:
        context_id = str(row["context_id"])
        probe = _reward_free_projection(row, memberships[context_id])
        relative = (
            Path("source_fit_banks")
            if row["role"] == "source"
            else Path("target_scoring_banks")
        ) / f"{context_id}.npz"
        projection_sha = _publish_probe(run_dir / relative, probe)
        sanitized_contexts.append(
            {
                "context_id": context_id,
                "role": "source" if row["role"] == "source" else "development",
                "task_id": str(row["task_id"]),
                "reward_free_npz": str(relative),
                "reward_free_npz_sha256": projection_sha,
                "probe_membership_digest": probe.probe_membership_digest,
                "episode_count": 32,
                "visible_transitions_per_episode": 64,
            }
        )
    scoring_raw_binding = {
        **dict(raw_binding),
        "root": str(operator_relative),
        "root_scope": "run_relative_source_only_copy",
        "contains_query_artifacts": False,
    }
    return raw_adapter, sanitized_contexts, scoring_raw_binding


def prepare(args: argparse.Namespace) -> Mapping[str, Any]:
    config, config_digest = _load_config(args.config)
    inputs = (
        args.context_index,
        args.public_policy_market,
        args.deployment_private_registry,
        args.origin_pool_acceptance,
        args.raw_delta_root,
        args.fpo_root,
    )
    _assert_new_output_root(args.run_dir, inputs)
    try:
        rows, market, layout, memberships, census = inspect_assets(
            context_index=args.context_index,
            public_policy_market=args.public_policy_market,
            deployment_private_registry=args.deployment_private_registry,
            origin_pool_acceptance=args.origin_pool_acceptance,
            raw_delta_root=args.raw_delta_root,
            fpo_root=args.fpo_root,
            split_seed=int(config["split_seed"]),
        )
    except GateFailure as error:
        return _record_prepare_failure(args, config_digest, error, status=error.status)
    except (
        DevelopmentBaselineError,
        SourceMarketError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        return _record_prepare_failure(args, config_digest, error)

    try:
        (
            raw_adapter,
            sanitized_contexts,
            scoring_raw_binding,
        ) = _materialize_sanitized_assets(
            run_dir=args.run_dir,
            context_index=args.context_index,
            raw_delta_root=args.raw_delta_root,
            rows=rows,
            memberships=memberships,
            raw_binding=census["raw_delta"],
        )
    except GateFailure as error:
        return _record_prepare_failure(args, config_digest, error, status=error.status)
    except (SignalBankRunnerError, OSError, ValueError, KeyError) as error:
        return _record_prepare_failure(args, config_digest, error)
    fixed_probe_protocol_id = sha256_json(
        {
            "operator": "V04A_FIXED_PROBE_REWARD_FREE",
            "logger_signature": census["logger_equivalence"]["signature"],
            "membership_schema": next(iter(memberships.values())).schema,
            "episode_membership": "context-hash permutation; first/last plus 62 hash-ranked timesteps",
            "visible_transitions_per_episode": 64,
            "interaction_steps_per_episode": 1000,
            "source_split": [19, 6, 7],
            "channels": ["observation", "action", "next_observation"],
            "masked_channels": ["reward", "terminated", "truncated"],
        }
    )
    membership_payload = {
        "schema": SCHEMA,
        "split_seed": int(config["split_seed"]),
        "fixed_probe_protocol_id": fixed_probe_protocol_id,
        "contexts": {
            context_id: membership.to_dict()
            for context_id, membership in sorted(memberships.items())
        },
    }
    method_cards = _method_cards(config_digest)
    layout_payload = {
        "schema": SCHEMA,
        "tasks": {
            task: {
                "source_type_ids": [row["context_id"] for row in value["source_rows"]],
                "paired_policy_by_type": value["paired_policy_by_type"],
                "candidate_ids": list(value["candidate_ids"]),
            }
            for task, value in layout.items()
        },
        "source_only": True,
    }
    source_fit_manifest = {
        "schema": "policy-learnware.v04a-source-fit-manifest.v1",
        "contains_reward_or_done": False,
        "contains_target_contexts": False,
        "contexts": sorted(
            (row for row in sanitized_contexts if row["role"] == "source"),
            key=lambda item: item["context_id"],
        ),
        "tasks": {
            task_id: {
                "task_id": task_id,
                "source_type_ids": [
                    str(row["context_id"]) for row in task["source_rows"]
                ],
                "paired_policy_by_type": dict(task["paired_policy_by_type"]),
                "candidate_ids": list(task["candidate_ids"]),
                "candidate_bundle_digests": {
                    candidate: market.deployment_private[candidate].bundle_digest
                    for candidate in task["candidate_ids"]
                },
            }
            for task_id, task in sorted(layout.items())
        },
    }
    scoring_manifest = {
        "schema": "policy-learnware.v04a-sanitized-scoring-manifest.v1",
        "access_track": "BI0-FP-RF",
        "contains_reward_or_done": False,
        "contains_target_construction_metadata": False,
        "contexts": sorted(sanitized_contexts, key=lambda item: item["context_id"]),
        "tasks": {
            task_id: {
                "task_id": task_id,
                "source_type_ids": [
                    str(row["context_id"]) for row in task["source_rows"]
                ],
                "paired_policy_by_type": dict(task["paired_policy_by_type"]),
                "candidate_ids": list(task["candidate_ids"]),
                "candidate_bundle_digests": {
                    candidate: market.deployment_private[candidate].bundle_digest
                    for candidate in task["candidate_ids"]
                },
                "raw_tie_break_tokens": {
                    candidate: market.entries[candidate].tie_break_token
                    for candidate in task["candidate_ids"]
                },
            }
            for task_id, task in sorted(layout.items())
        },
    }
    run = {
        "schema": SCHEMA,
        "stage": "prepare",
        "status": "PREPARED",
        "formal": False,
        "scope": "30 source + 24 frozen development; never formal/confirmatory",
        "config": config,
        "config_digest": config_digest,
        "plan_sha256": PLAN_SHA256,
        "policy_market_id": market.policy_market_id,
        "raw_delta": scoring_raw_binding,
        "source_fit_roles": [19, 6, 7],
        "fixed_probe_protocol_id": fixed_probe_protocol_id,
        "probe_membership_payload_digest": sha256_json(membership_payload),
        "method_cards_payload_digest": sha256_json(method_cards),
        "source_task_layout_payload_digest": sha256_json(layout_payload),
        "source_fit_manifest_payload_digest": sha256_json(source_fit_manifest),
        "scoring_manifest_payload_digest": sha256_json(scoring_manifest),
        "raw_delta_adapter_payload_digest": sha256_json(raw_adapter),
        "target_oracle_bound": False,
    }
    _publish(args.run_dir / "asset_census.json", census)
    _publish(args.run_dir / "run.json", run)
    _publish(args.run_dir / "method_cards.json", method_cards)
    _publish(args.run_dir / "probe_membership.json", membership_payload)
    _publish(args.run_dir / "source_task_layout.json", layout_payload)
    _publish(args.run_dir / "source_fit_manifest.json", source_fit_manifest)
    _publish(args.run_dir / "scoring_manifest.json", scoring_manifest)
    _publish(args.run_dir / "raw_delta_adapter.json", raw_adapter)
    return run


def _prepared(run_dir: Path) -> tuple[Mapping[str, Any], dict[str, Any], str]:
    run = _json(run_dir / "run.json")
    if run.get("status") != "PREPARED" or run.get("stage") != "prepare":
        raise V04ARunnerError("run directory did not pass prepare")
    config = run.get("config")
    if not isinstance(config, Mapping):
        raise V04ARunnerError("run.json lacks frozen configuration")
    digest = sha256_json(config)
    if digest != run.get("config_digest"):
        raise V04ARunnerError("frozen configuration digest differs")
    return run, dict(config), digest


def _memberships(run_dir: Path) -> dict[str, ProbeMembership]:
    value = _json(run_dir / "probe_membership.json")
    run = _json(run_dir / "run.json")
    if sha256_json(value) != run.get("probe_membership_payload_digest"):
        raise V04ARunnerError("probe membership artifact differs from prepare binding")
    contexts = value.get("contexts")
    if not isinstance(contexts, Mapping):
        raise V04ARunnerError("probe membership artifact is malformed")
    return {
        str(context_id): ProbeMembership.from_dict(payload)
        for context_id, payload in contexts.items()
    }


def _scoring_manifest(run_dir: Path, run: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _json(run_dir / "scoring_manifest.json")
    if sha256_json(value) != run.get("scoring_manifest_payload_digest"):
        raise V04ARunnerError("sanitized scoring manifest differs from prepare binding")
    if (
        value.get("contains_reward_or_done") is not False
        or value.get("contains_target_construction_metadata") is not False
    ):
        raise V04ARunnerError("scoring manifest is not BI0-sanitized")
    return value


def _source_fit_layout(
    run_dir: Path, run: Mapping[str, Any]
) -> tuple[tuple[dict[str, Any], ...], dict[str, dict[str, Any]]]:
    """Validate the source-only fit closure without opening the scoring manifest."""

    value = _json(run_dir / "source_fit_manifest.json")
    if sha256_json(value) != run.get("source_fit_manifest_payload_digest"):
        raise V04ARunnerError("source-only fit manifest differs from prepare binding")
    if (
        set(value)
        != {
            "schema",
            "contains_reward_or_done",
            "contains_target_contexts",
            "contexts",
            "tasks",
        }
        or value.get("schema") != "policy-learnware.v04a-source-fit-manifest.v1"
    ):
        raise V04ARunnerError("source-only fit manifest has an unexpected capability")
    if (
        value.get("contains_reward_or_done") is not False
        or value.get("contains_target_contexts") is not False
    ):
        raise V04ARunnerError("source-only fit manifest exposes forbidden data")
    raw_contexts = value.get("contexts")
    raw_tasks = value.get("tasks")
    if not isinstance(raw_contexts, list) or not isinstance(raw_tasks, Mapping):
        raise V04ARunnerError("source-only fit manifest is malformed")
    context_fields = {
        "context_id",
        "role",
        "task_id",
        "reward_free_npz",
        "reward_free_npz_sha256",
        "probe_membership_digest",
        "episode_count",
        "visible_transitions_per_episode",
    }
    contexts: list[dict[str, Any]] = []
    observed_contexts: set[str] = set()
    for raw_row in raw_contexts:
        if not isinstance(raw_row, Mapping) or set(raw_row) != context_fields:
            raise V04ARunnerError("source-only fit context row is malformed")
        row = dict(raw_row)
        context_id = str(row["context_id"])
        relative = Path(str(row["reward_free_npz"]))
        if (
            not context_id
            or context_id in observed_contexts
            or row["role"] != "source"
            or row["episode_count"] != 32
            or row["visible_transitions_per_episode"] != 64
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.parent != Path("source_fit_banks")
        ):
            raise V04ARunnerError("source-only fit context protocol differs")
        observed_contexts.add(context_id)
        contexts.append(row)
    if len(contexts) != 30:
        raise V04ARunnerError(
            "source-only fit closure must contain exactly 30 contexts"
        )

    task_fields = {
        "task_id",
        "source_type_ids",
        "paired_policy_by_type",
        "candidate_ids",
        "candidate_bundle_digests",
    }
    layout: dict[str, dict[str, Any]] = {}
    seen_candidates: set[str] = set()
    if len(raw_tasks) != 6:
        raise V04ARunnerError("source-only TASK_5 layout must contain six tasks")
    for task_id, raw_task in raw_tasks.items():
        if not isinstance(raw_task, Mapping) or set(raw_task) != task_fields:
            raise V04ARunnerError(f"source-only task layout is malformed: {task_id}")
        task = dict(raw_task)
        source_ids = tuple(str(item) for item in task["source_type_ids"])
        candidates = tuple(str(item) for item in task["candidate_ids"])
        paired = task["paired_policy_by_type"]
        bundle_digests = task["candidate_bundle_digests"]
        if (
            task.get("task_id") != task_id
            or len(source_ids) != 5
            or len(set(source_ids)) != 5
            or len(candidates) != 5
            or len(set(candidates)) != 5
            or not isinstance(paired, Mapping)
            or set(paired) != set(source_ids)
            or set(paired.values()) != set(candidates)
            or not isinstance(bundle_digests, Mapping)
            or set(bundle_digests) != set(candidates)
            or not set(source_ids).issubset(observed_contexts)
        ):
            raise V04ARunnerError(f"source-only TASK_5 binding differs: {task_id}")
        seen_candidates.update(candidates)
        layout[str(task_id)] = task
    if (
        len(seen_candidates) != 30
        or {str(row["task_id"]) for row in contexts} != set(layout)
        or any(
            {str(row["context_id"]) for row in contexts if row["task_id"] == task_id}
            != set(task["source_type_ids"])
            for task_id, task in layout.items()
        )
    ):
        raise V04ARunnerError("source-only fit task/context coverage differs")
    return tuple(contexts), layout


def _sanitized_layout(
    run_dir: Path, run: Mapping[str, Any]
) -> tuple[tuple[dict[str, Any], ...], dict[str, dict[str, Any]]]:
    """Validate the complete score-visible closure without reopening private inputs."""

    value = _scoring_manifest(run_dir, run)
    allowed_top = {
        "schema",
        "access_track",
        "contains_reward_or_done",
        "contains_target_construction_metadata",
        "contexts",
        "tasks",
    }
    if set(value) != allowed_top or value.get("access_track") != "BI0-FP-RF":
        raise V04ARunnerError("sanitized scoring manifest has an unexpected capability")
    raw_contexts = value.get("contexts")
    raw_tasks = value.get("tasks")
    if not isinstance(raw_contexts, list) or not isinstance(raw_tasks, Mapping):
        raise V04ARunnerError("sanitized scoring manifest is malformed")
    context_fields = {
        "context_id",
        "role",
        "task_id",
        "reward_free_npz",
        "reward_free_npz_sha256",
        "probe_membership_digest",
        "episode_count",
        "visible_transitions_per_episode",
    }
    contexts = tuple(dict(row) for row in raw_contexts)
    if any(set(row) != context_fields for row in contexts):
        raise V04ARunnerError("score-visible context rows expose unexpected metadata")
    context_ids = [str(row["context_id"]) for row in contexts]
    if len(contexts) != 54 or len(set(context_ids)) != 54:
        raise V04ARunnerError("score-visible context set must contain 54 unique rows")
    if (
        sum(row["role"] == "source" for row in contexts) != 30
        or sum(row["role"] == "development" for row in contexts) != 24
    ):
        raise V04ARunnerError("score-visible roles must be 30 source + 24 development")
    if any(
        row["role"] not in {"source", "development"}
        or row["episode_count"] != 32
        or row["visible_transitions_per_episode"] != 64
        for row in contexts
    ):
        raise V04ARunnerError("score-visible context protocol differs")
    for row in contexts:
        relative = Path(str(row["reward_free_npz"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise V04ARunnerError(
                "reward-free projection path escapes the run directory"
            )

    task_fields = {
        "task_id",
        "source_type_ids",
        "paired_policy_by_type",
        "candidate_ids",
        "candidate_bundle_digests",
        "raw_tie_break_tokens",
    }
    layout: dict[str, dict[str, Any]] = {}
    seen_candidates: set[str] = set()
    if len(raw_tasks) != 6:
        raise V04ARunnerError("TASK_5 layout must contain six tasks")
    for task_id, raw_task in raw_tasks.items():
        if not isinstance(raw_task, Mapping) or set(raw_task) != task_fields:
            raise V04ARunnerError(f"score-visible task layout is malformed: {task_id}")
        task = dict(raw_task)
        if task.get("task_id") != task_id:
            raise V04ARunnerError(
                f"task identity differs in scoring manifest: {task_id}"
            )
        source_ids = tuple(str(value) for value in task["source_type_ids"])
        candidates = tuple(str(value) for value in task["candidate_ids"])
        paired = task["paired_policy_by_type"]
        bundle_digests = task["candidate_bundle_digests"]
        raw_ties = task["raw_tie_break_tokens"]
        if (
            len(source_ids) != 5
            or len(set(source_ids)) != 5
            or len(candidates) != 5
            or len(set(candidates)) != 5
            or not isinstance(paired, Mapping)
            or set(paired) != set(source_ids)
            or set(paired.values()) != set(candidates)
            or not isinstance(bundle_digests, Mapping)
            or set(bundle_digests) != set(candidates)
            or not isinstance(raw_ties, Mapping)
            or set(raw_ties) != set(candidates)
        ):
            raise V04ARunnerError(f"TASK_5 source/candidate binding differs: {task_id}")
        task_contexts = [row for row in contexts if row["task_id"] == task_id]
        if {
            row["context_id"] for row in task_contexts if row["role"] == "source"
        } != set(source_ids) or sum(
            row["role"] == "development" for row in task_contexts
        ) != 4:
            raise V04ARunnerError(
                f"source/development task coverage differs: {task_id}"
            )
        seen_candidates.update(candidates)
        layout[str(task_id)] = task
    if len(seen_candidates) != 30 or any(
        row["task_id"] not in layout for row in contexts
    ):
        raise V04ARunnerError("scoring manifest does not partition contexts/candidates")
    return contexts, layout


def _raw_adapter(run_dir: Path, run: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _json(run_dir / "raw_delta_adapter.json")
    if sha256_json(value) != run.get("raw_delta_adapter_payload_digest"):
        raise V04ARunnerError("Raw-Delta adapter differs from prepare binding")
    return value


def _transition_episodes(probe: RewardFreeProbe) -> tuple[TransitionEpisode, ...]:
    return tuple(
        TransitionEpisode(
            probe.observation[probe.episode_slice(index)],
            probe.action[probe.episode_slice(index)],
            probe.next_observation[probe.episode_slice(index)],
        )
        for index in range(probe.episode_count)
    )


def _evidence_root(root: Path) -> Path:
    direct = root
    nested = root / "oracle"
    if nested.is_dir() and any(nested.glob("*/*.json")):
        return nested.resolve()
    if direct.is_dir():
        for context_dir in direct.iterdir():
            if context_dir.is_dir() and len(list(context_dir.glob("*.json"))) >= 5:
                return direct.resolve()
    raise GateFailure(
        "NO_GO_MISSING_ASSET", f"no context/candidate evidence records under {root}"
    )


def _validated_evidence_namespace(
    root: Path,
    *,
    policy_market_id: str,
    expected_contexts: Mapping[str, tuple[str, str]],
    failure_status: str,
) -> tuple[Path, dict[str, Any]]:
    """Bind v03 oracle cells to their completed development-run manifests.

    The v03 cell format has no embedded manifest digest.  After the public
    rankings are sealed, its target-oracle namespace is therefore admitted
    only together with the sibling ``representations/build_config.json`` and
    ``summary.json``.  The pre-seal source-only fitter deliberately does not
    call this mixed-namespace helper.
    """

    evidence = _evidence_root(root)
    baseline_root = evidence.parent
    build_path = baseline_root / "representations" / "build_config.json"
    summary_path = baseline_root / "summary.json"
    if (
        evidence.name != "oracle"
        or baseline_root.is_symlink()
        or build_path.is_symlink()
        or summary_path.is_symlink()
        or not build_path.is_file()
        or not summary_path.is_file()
    ):
        raise GateFailure(
            failure_status,
            "evidence namespace lacks safe v03 build/summary provenance",
        )
    build = _json(build_path)
    summary = _json(summary_path)
    if (
        build.get("schema") != V03_BASELINE_SCHEMA
        or build.get("stage") != "representations"
        or build.get("context_count") != 54
        or build.get("source_count") != 30
        or build.get("policy_market_id") != policy_market_id
        or summary.get("schema") != V03_BASELINE_SCHEMA
        or summary.get("stage") != "summary"
        or summary.get("status") != "COMPLETE"
        or summary.get("formal") is not False
        or summary.get("context_count") != 54
        or summary.get("policy_market_id") != policy_market_id
        or summary.get("missing") != []
    ):
        raise GateFailure(
            failure_status,
            "v03 evidence build/summary provenance is incomplete or differently bound",
        )
    raw_rows = build.get("context_rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 54:
        raise GateFailure(failure_status, "v03 evidence context manifest is incomplete")
    observed: dict[str, tuple[str, str]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise GateFailure(
                failure_status, "v03 evidence context manifest contains a non-object"
            )
        context_id = str(raw_row.get("context_id", ""))
        task_id = str(raw_row.get("task_id", ""))
        raw_role = str(raw_row.get("role", ""))
        role = "source" if raw_role == "source" else "development"
        if (
            not context_id
            or not task_id
            or raw_role not in {"source", "development", "development_query"}
            or context_id in observed
        ):
            raise GateFailure(
                failure_status, "v03 evidence context manifest identity is malformed"
            )
        observed[context_id] = (role, task_id)
    if observed != dict(expected_contexts):
        raise GateFailure(
            failure_status,
            "v03 evidence context manifest differs from the prepared 30+24 scope",
        )
    return evidence, {
        "schema": V03_BASELINE_SCHEMA,
        "build_config_sha256": sha256_file(build_path),
        "summary_sha256": sha256_file(summary_path),
        "policy_market_id": policy_market_id,
        "context_count": 54,
        "source_count": 30,
        "formal": False,
    }


def _validated_return_record(
    record: Mapping[str, Any],
    *,
    context_id: str,
    task_id: str,
    candidate_id: str,
    bundle_digest: str,
    failure_status: str,
) -> tuple[float, tuple[float, ...], tuple[int, ...], tuple[int, ...]]:
    """Validate one immutable 50-episode utility/oracle evidence cell."""

    identity = {
        "schema": V03_BASELINE_SCHEMA,
        "stage": "PRIVATE_ORACLE",
        "context_id": context_id,
        "task_id": task_id,
        "opaque_learnware_id": candidate_id,
        "bundle_digest": bundle_digest,
        "status": "OK",
        "executed": True,
        "horizon": 1000,
    }
    for key, expected in identity.items():
        if record.get(key) != expected:
            raise GateFailure(
                failure_status,
                f"evidence identity/binding mismatch at {context_id}/{candidate_id}.{key}",
            )
    returns = record.get("episode_returns")
    if (
        not isinstance(returns, list)
        or len(returns) != 50
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in returns
        )
    ):
        raise GateFailure(
            failure_status,
            f"per-episode evidence is not an exact finite 50-vector: {context_id}/{candidate_id}",
        )
    values = np.asarray(returns, dtype=np.float64)
    mean = float(np.mean(values))
    try:
        reported_mean = float(record.get("mean_return", float("nan")))
        normalized = float(record.get("normalized_mean_return", float("nan")))
    except (TypeError, ValueError) as error:
        raise GateFailure(
            failure_status,
            f"mean evidence is malformed: {context_id}/{candidate_id}",
        ) from error
    if (
        not math.isfinite(reported_mean)
        or not math.isfinite(normalized)
        or not np.isclose(reported_mean, mean, rtol=0.0, atol=1.0e-10)
        or not np.isclose(normalized, mean / 1000.0, rtol=0.0, atol=1.0e-12)
    ):
        raise GateFailure(
            failure_status,
            f"mean/per-episode evidence disagree: {context_id}/{candidate_id}",
        )
    seed_vectors: dict[str, tuple[int, ...]] = {}
    for seed_field in ("reset_seeds", "policy_seeds"):
        seeds = record.get(seed_field)
        if (
            not isinstance(seeds, list)
            or len(seeds) != 50
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
            )
            or len(set(seeds)) != 50
        ):
            raise GateFailure(
                failure_status,
                f"{seed_field} is not an exact unique 50-vector: {context_id}/{candidate_id}",
            )
        seed_vectors[seed_field] = tuple(int(seed) for seed in seeds)
    if seed_vectors["policy_seeds"] != tuple(
        seed + 1_000_003 for seed in seed_vectors["reset_seeds"]
    ):
        raise GateFailure(
            failure_status,
            f"policy/reset seed namespace differs: {context_id}/{candidate_id}",
        )
    return (
        normalized,
        tuple(float(value) / 1000.0 for value in values),
        seed_vectors["reset_seeds"],
        seed_vectors["policy_seeds"],
    )


def _utility_matrix(
    root: Path,
    layout: Mapping[str, Any],
    *,
    train_episode_count: int = 30,
    expected_reset_seeds: Sequence[int] = tuple(range(730000, 730050)),
    policy_seed_offset: int = 1_000_003,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    evidence = _evidence_root(root)
    rows = layout["source_rows"]
    if any(row.get("role") != "source" for row in rows):
        raise GateFailure(
            "NO_GO_SOURCE_UTILITY_GAP",
            "utility builder accepts source-role contexts only",
        )
    candidates = layout["candidate_ids"]
    task_id = str(layout["task_id"])
    bundle_digests = layout["candidate_bundle_digests"]
    if (
        isinstance(train_episode_count, bool)
        or not isinstance(train_episode_count, int)
        or not 0 < train_episode_count < 50
    ):
        raise V04ARunnerError("source utility train split must be within 1..49")
    expected_resets = tuple(int(seed) for seed in expected_reset_seeds)
    if (
        len(expected_resets) != 50
        or len(set(expected_resets)) != 50
        or any(
            isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            for seed in expected_reset_seeds
        )
        or isinstance(policy_seed_offset, bool)
        or not isinstance(policy_seed_offset, int)
    ):
        raise V04ARunnerError("source utility seed binding must be an exact 50-vector")
    expected_policies = tuple(seed + policy_seed_offset for seed in expected_resets)
    utility: dict[str, dict[str, float]] = {}
    digests: dict[str, str] = {}
    for source in rows:
        type_id = str(source["context_id"])
        utility[type_id] = {}
        common_seed_bank: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        for candidate in candidates:
            path = evidence / type_id / f"{candidate}.json"
            if path.is_symlink() or not path.is_file():
                raise GateFailure(
                    "NO_GO_SOURCE_UTILITY_GAP",
                    f"source-only utility cell missing: {type_id}/{candidate}",
                )
            _, episode_returns, reset_seeds, policy_seeds = _validated_return_record(
                _json(path),
                context_id=type_id,
                task_id=task_id,
                candidate_id=candidate,
                bundle_digest=str(bundle_digests[candidate]),
                failure_status="NO_GO_SOURCE_UTILITY_GAP",
            )
            seed_bank = (reset_seeds, policy_seeds)
            if reset_seeds != expected_resets or policy_seeds != expected_policies:
                raise GateFailure(
                    "NO_GO_SOURCE_UTILITY_GAP",
                    f"source utility seed identity/order differs from freeze: {type_id}/{candidate}",
                )
            if common_seed_bank is None:
                common_seed_bank = seed_bank
            elif seed_bank != common_seed_bank:
                raise GateFailure(
                    "NO_GO_SOURCE_UTILITY_GAP",
                    f"five source utility candidates lack a common seed bank: {type_id}",
                )
            utility[type_id][candidate] = float(
                np.mean(episode_returns[:train_episode_count])
            )
            digests[f"{type_id}/{candidate}"] = sha256_file(path)
    return utility, digests


def _source_repeat_metrics(
    *,
    model: BPRGaussianModel,
    ebpr: EBPRFixedProbe,
    repeat_summaries: Mapping[str, np.ndarray],
    repeat_episodes: Mapping[str, Sequence[TransitionEpisode]],
    tie_identity_by_type: Mapping[str, str],
) -> dict[str, Any]:
    if set(tie_identity_by_type) != set(model.type_ids):
        raise V04ARunnerError("source repeat tie identities do not cover model types")
    bpr_hits = 0
    ebpr_hits = 0
    bpr_mrr: list[float] = []
    ebpr_mrr: list[float] = []
    bpr_entropy: list[float] = []
    ebpr_entropy: list[float] = []
    bpr_brier: list[float] = []
    ebpr_brier: list[float] = []
    bpr_order_delta = 0.0
    ebpr_order_delta = 0.0
    for type_id in model.type_ids:
        summaries = np.asarray(repeat_summaries[type_id], dtype=np.float64)
        episodes = tuple(repeat_episodes[type_id])
        if summaries.shape[0] != 7 or len(episodes) != 7:
            raise V04ARunnerError(
                f"source repeat report requires seven sealed episodes: {type_id}"
            )
        for index in range(7):
            bpr_posterior = model.posterior_dict(summaries[index : index + 1])
            bpr_order = sorted(
                bpr_posterior,
                key=lambda value: (
                    -bpr_posterior[value],
                    tie_break_key(model.config_digest, tie_identity_by_type[value]),
                ),
            )
            bpr_hits += int(bpr_order[0] == type_id)
            bpr_mrr.append(1.0 / (bpr_order.index(type_id) + 1))
            bpr_entropy.append(_entropy(bpr_posterior))
            bpr_brier.append(
                float(
                    sum(
                        (probability - float(candidate_type == type_id)) ** 2
                        for candidate_type, probability in bpr_posterior.items()
                    )
                )
            )

            ebpr_posterior = ebpr.posterior((episodes[index],))
            ebpr_order = sorted(
                ebpr_posterior,
                key=lambda value: (
                    -ebpr_posterior[value],
                    tie_break_key(model.config_digest, tie_identity_by_type[value]),
                ),
            )
            ebpr_hits += int(ebpr_order[0] == type_id)
            ebpr_mrr.append(1.0 / (ebpr_order.index(type_id) + 1))
            ebpr_entropy.append(_entropy(ebpr_posterior))
            ebpr_brier.append(
                float(
                    sum(
                        (probability - float(candidate_type == type_id)) ** 2
                        for candidate_type, probability in ebpr_posterior.items()
                    )
                )
            )

        combined_bpr = model.posterior_dict(summaries)
        reversed_bpr = model.posterior_dict(summaries[::-1])
        bpr_order_delta = max(
            bpr_order_delta,
            max(abs(combined_bpr[key] - reversed_bpr[key]) for key in combined_bpr),
        )
        combined_ebpr = ebpr.posterior(episodes)
        reversed_ebpr = ebpr.posterior(tuple(reversed(episodes)))
        ebpr_order_delta = max(
            ebpr_order_delta,
            max(abs(combined_ebpr[key] - reversed_ebpr[key]) for key in combined_ebpr),
        )
    count = len(bpr_mrr)
    return {
        "role": "source_repeat_report_only",
        "type_count": len(model.type_ids),
        "evaluation_unit": "one independently sealed repeat episode",
        "tie_identity": "paired opaque policy ID",
        "repeat_episode_count": count,
        "BPR_FP": {
            "type_hit_at_1": bpr_hits / count,
            "type_mrr": float(np.mean(bpr_mrr)),
            "validation_nll": model.validation_nll,
            "mean_posterior_entropy": float(np.mean(bpr_entropy)),
            "mean_multiclass_brier": float(np.mean(bpr_brier)),
            "probe_order_max_abs_posterior_delta": bpr_order_delta,
            "utility_matrix_condition_number": (
                float(np.linalg.cond(model.utility_matrix))
                if np.isfinite(np.linalg.cond(model.utility_matrix))
                else None
            ),
            "utility_matrix_rank": int(np.linalg.matrix_rank(model.utility_matrix)),
        },
        "EBPR_FP": {
            "type_hit_at_1": ebpr_hits / count,
            "type_mrr": float(np.mean(ebpr_mrr)),
            "validation_classification_nll": ebpr.calibration.get("classification_nll"),
            "validation_predictive_nll_per_transition_per_valid_dim": ebpr.calibration.get(
                "true_type_predictive_nll_per_transition_per_valid_dim"
            ),
            "mean_posterior_entropy": float(np.mean(ebpr_entropy)),
            "mean_multiclass_brier": float(np.mean(ebpr_brier)),
            "probe_order_max_abs_posterior_delta": ebpr_order_delta,
        },
    }


def fit_source(args: argparse.Namespace) -> Mapping[str, Any]:
    if any(
        (args.run_dir / name).exists()
        for name in (
            "rankings.jsonl",
            "rankings.seal.json",
            "oracle_binding.json",
            "metrics.jsonl",
        )
    ):
        raise V04ARunnerError(
            "fit-source cannot run after target scoring or oracle binding"
        )
    run, config, config_digest = _prepared(args.run_dir)
    rows, sanitized_tasks = _source_fit_layout(args.run_dir, run)
    source_by_id = {str(row["context_id"]): row for row in rows}
    layout: dict[str, dict[str, Any]] = {}
    for task_id, raw_task in sanitized_tasks.items():
        task = dict(raw_task)
        task["source_rows"] = [
            source_by_id[type_id] for type_id in task["source_type_ids"]
        ]
        layout[str(task_id)] = task
    utility_root = args.source_utility_root.resolve()
    raw_operator_root = (args.run_dir / str(run["raw_delta"]["root"])).resolve()
    if utility_root == raw_operator_root:
        raise V04ARunnerError("source utility and Raw artifact roots cannot alias")

    bpr_cfg = config["bpr"]
    ebpr_cfg = config["ebpr"]
    utility_split = config["source_utility_split"]
    utility_seed_namespace = config["source_utility_seed_namespace"]
    expected_utility_reset_seeds = tuple(
        range(
            int(utility_seed_namespace["reset_seed_start"]),
            int(utility_seed_namespace["reset_seed_start"])
            + int(utility_seed_namespace["episode_count"]),
        )
    )
    models: dict[str, Any] = {}
    utility_artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "source_only": True,
        "source_cell_projection": "150 validated source-context cells; mixed input root path withheld from scorer",
        "episode_role_split": {
            "source_train": {
                "episode_count_per_cell": int(utility_split["train_episodes"]),
                "usage": "fit source task-policy expected utility matrix",
            },
            "source_validation": {
                "episode_count_per_cell": int(utility_split["validation_episodes"]),
                "usage": "reserved; not used by BI0 utility fit",
            },
            "source_repeat_report": {
                "episode_count_per_cell": int(utility_split["repeat_report_episodes"]),
                "usage": "reserved sealed repeat evidence; not used by BI0 utility fit",
            },
            "seed_order": "the immutable per-cell common reset-seed vector order",
            "reset_seed_vector_sha256": sha256_json(list(expected_utility_reset_seeds)),
            "policy_seed_offset": int(utility_seed_namespace["policy_seed_offset"]),
        },
        "tasks": {},
    }
    repeat_reports: dict[str, Any] = {}
    source_evidence_digests: dict[str, str] = {}
    fitted: dict[
        str,
        tuple[
            BPRGaussianModel,
            EBPRFixedProbe,
            Mapping[str, np.ndarray],
            Mapping[str, Sequence[TransitionEpisode]],
        ],
    ] = {}
    try:
        # Do not read the mixed 54-context v03 build/summary manifests here:
        # they contain target construction metadata and oracle-derived
        # aggregates.  Source admission is instead an exact allowlist of the
        # 150 source-role cells, each schema/stage/bundle/seed/episode checked
        # below and then frozen by digest in the source-only projection.
        evidence_root = _evidence_root(utility_root)
        if evidence_root.name != "oracle":
            raise GateFailure(
                "NO_GO_SOURCE_UTILITY_GAP",
                "source utility evidence must come from an oracle-cell namespace",
            )
        utility_artifact["input_namespace_provenance"] = {
            "source_projection": "exact allowlisted source-role cells only",
            "expected_cell_schema": V03_BASELINE_SCHEMA,
            "expected_cell_stage": "PRIVATE_ORACLE",
            "expected_source_context_count": 30,
            "expected_cell_count": 150,
            "mixed_manifest_or_target_records_read": False,
            "input_path_withheld_from_scorer": True,
        }
        # Preflight every source-only utility/per-episode cell before any model
        # is fitted or checkpoint is published.
        for task_id, task in layout.items():
            utility, digests = _utility_matrix(
                evidence_root,
                task,
                train_episode_count=int(utility_split["train_episodes"]),
                expected_reset_seeds=expected_utility_reset_seeds,
                policy_seed_offset=int(utility_seed_namespace["policy_seed_offset"]),
            )
            source_evidence_digests.update(digests)
            utility_artifact["tasks"][task_id] = utility
        if len(source_evidence_digests) != 150:
            raise GateFailure(
                "NO_GO_SOURCE_UTILITY_GAP",
                "source utility projection must contain exactly 150 unique cells",
            )
        for task_id, task in layout.items():
            utility = utility_artifact["tasks"][task_id]
            train_summaries: dict[str, np.ndarray] = {}
            validation_summaries: dict[str, np.ndarray] = {}
            repeat_summaries: dict[str, np.ndarray] = {}
            train_episodes: dict[str, tuple[TransitionEpisode, ...]] = {}
            validation_episodes: dict[str, tuple[TransitionEpisode, ...]] = {}
            repeat_episodes: dict[str, tuple[TransitionEpisode, ...]] = {}
            for type_id in task["paired_policy_by_type"]:
                probe = _projected_probe(args.run_dir, source_by_id[type_id], 32)
                summaries = summarize_probe(probe)
                episodes = _transition_episodes(probe)
                train_summaries[type_id] = summaries[:19]
                validation_summaries[type_id] = summaries[19:25]
                repeat_summaries[type_id] = summaries[25:]
                train_episodes[type_id] = episodes[:19]
                validation_episodes[type_id] = episodes[19:25]
                repeat_episodes[type_id] = episodes[25:]

            bpr = BPRGaussianModel.fit(
                train_summaries,
                validation_summaries,
                utility,
                config_digest=config_digest,
                protocol_id=str(run["fixed_probe_protocol_id"]),
                candidate_ids=task["candidate_ids"],
                lambda_grid=bpr_cfg["shrinkage_grid"],
                variance_floor_grid=bpr_cfg["variance_floor_grid"],
                temperature_grid=bpr_cfg["temperature_grid"],
            )
            ebpr = EBPRFixedProbe.fit(
                train_episodes,
                validation_episodes,
                task["paired_policy_by_type"],
                hidden_dim=int(ebpr_cfg["hidden_dim"]),
                ridge=float(ebpr_cfg["ridge"]),
                feature_seed=int(ebpr_cfg["feature_seed"]),
                variance_floor_candidates=ebpr_cfg["variance_floor_grid"],
                temperature_candidates=ebpr_cfg["temperature_grid"],
                tie_token=canonical_tie_token(config_digest),
            )
            fitted[task_id] = (
                bpr,
                ebpr,
                repeat_summaries,
                repeat_episodes,
            )
    except GateFailure as error:
        status = {
            "schema": SCHEMA,
            "stage": "fit-source",
            "status": error.status,
            "message": str(error),
            "details": error.details,
            "source_only": True,
        }
        _publish(args.run_dir / "fit_source_status.json", status, resume=args.resume)
        return status
    except (
        BPRModelError,
        EBPRError,
        V04ARunnerError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        FloatingPointError,
    ) as error:
        status = {
            "schema": SCHEMA,
            "stage": "fit-source",
            "status": "NO_GO_SOURCE_MODEL_FIT",
            "message": str(error),
            "error_type": type(error).__name__,
            "source_only": True,
        }
        _publish(args.run_dir / "fit_source_status.json", status, resume=args.resume)
        return status

    # Only after all six tasks have fitted successfully are immutable
    # checkpoints published.
    for task_id, task in layout.items():
        bpr, ebpr, repeat_summaries, repeat_episodes = fitted[task_id]
        bpr_path = args.run_dir / "checkpoints" / f"{task_id}--bpr.json"
        ebpr_path = args.run_dir / "checkpoints" / f"{task_id}--ebpr.json"
        _publish(bpr_path, bpr.to_dict(), resume=args.resume)
        _publish(ebpr_path, ebpr.to_dict(), resume=args.resume)
        models[task_id] = {
            "BPR_FP": {
                "path": str(bpr_path.relative_to(args.run_dir)),
                "sha256": sha256_file(bpr_path),
                "model_digest": bpr.model_digest,
                "source_train_episodes_per_type": 19,
                "source_validation_episodes_per_type": 6,
            },
            "EBPR_FP": {
                "path": str(ebpr_path.relative_to(args.run_dir)),
                "sha256": sha256_file(ebpr_path),
                "model_digest": sha256_json(ebpr.to_dict()),
                "variance_floor": ebpr.variance_floor,
                "temperature": ebpr.posterior_temperature,
                "ridge": ebpr.ridge,
                "feature_seed": ebpr.feature_seed,
                "tie_token": ebpr.tie_token,
                "source_train_episodes_per_type": 19,
                "source_validation_episodes_per_type": 6,
            },
        }
        repeat_reports[task_id] = _source_repeat_metrics(
            model=bpr,
            ebpr=ebpr,
            repeat_summaries=repeat_summaries,
            repeat_episodes=repeat_episodes,
            tie_identity_by_type=task["paired_policy_by_type"],
        )

    utility_artifact["record_digests"] = source_evidence_digests
    utility_artifact["per_episode_evidence"] = "PASS_50_EPISODES_PER_CELL"
    source_utility_sha256 = _publish(
        args.run_dir / "source_utility.json", utility_artifact, resume=args.resume
    )
    manifest = {
        "schema": SCHEMA,
        "stage": "fit-source",
        "status": "COMPLETE",
        "source_only": True,
        "source_fit_manifest_sha256": sha256_file(
            args.run_dir / "source_fit_manifest.json"
        ),
        "config_digest": config_digest,
        "fixed_probe_protocol_id": run["fixed_probe_protocol_id"],
        "source_utility_sha256": source_utility_sha256,
        "models": models,
        "source_repeat_report": repeat_reports,
        "target_contexts_read": 0,
        "sequential_status": "DEFERRED_BY_MVP_SCOPE; source return evidence present",
    }
    _publish(
        args.run_dir / "source_observation_model_manifest.json",
        manifest,
        resume=args.resume,
    )
    _publish(args.run_dir / "fit_source_status.json", manifest, resume=args.resume)
    return manifest


def _load_source_models(
    run_dir: Path,
    task_id: str,
    manifest: Mapping[str, Any],
    *,
    task_layout: Mapping[str, Any],
    utility: Mapping[str, Mapping[str, float]],
) -> tuple[BPRGaussianModel, EBPRFixedProbe]:
    task = manifest["models"][task_id]
    bpr_relative = Path(str(task["BPR_FP"]["path"]))
    ebpr_relative = Path(str(task["EBPR_FP"]["path"]))
    if (
        bpr_relative.is_absolute()
        or ebpr_relative.is_absolute()
        or ".." in bpr_relative.parts
        or ".." in ebpr_relative.parts
    ):
        raise V04ARunnerError(f"source checkpoint escapes run directory: {task_id}")
    bpr_path = run_dir / bpr_relative
    ebpr_path = run_dir / ebpr_relative
    if bpr_path.is_symlink() or sha256_file(bpr_path) != task["BPR_FP"]["sha256"]:
        raise V04ARunnerError(f"BPR checkpoint changed for {task_id}")
    if ebpr_path.is_symlink() or sha256_file(ebpr_path) != task["EBPR_FP"]["sha256"]:
        raise V04ARunnerError(f"EBPR checkpoint changed for {task_id}")
    bpr = BPRGaussianModel.from_dict(_json(bpr_path))
    ebpr = EBPRFixedProbe.from_dict(_json(ebpr_path))
    source_types = tuple(sorted(str(value) for value in task_layout["source_type_ids"]))
    candidates = tuple(str(value) for value in task_layout["candidate_ids"])
    paired = {
        str(key): str(value)
        for key, value in task_layout["paired_policy_by_type"].items()
    }
    expected_utility = np.asarray(
        [
            [utility[type_id][candidate] for candidate in candidates]
            for type_id in source_types
        ],
        dtype=np.float64,
    )
    ebpr_record = task["EBPR_FP"]
    if (
        bpr.model_digest != task["BPR_FP"].get("model_digest")
        or bpr.config_digest != manifest.get("config_digest")
        or bpr.protocol_id != manifest.get("fixed_probe_protocol_id")
        or bpr.type_ids != source_types
        or bpr.candidate_ids != candidates
        or not np.array_equal(bpr.utility_matrix, expected_utility)
        or ebpr.type_ids != source_types
        or dict(ebpr.paired_policy_by_type) != paired
        or sha256_json(ebpr.to_dict()) != ebpr_record.get("model_digest")
        or ebpr.variance_floor != ebpr_record.get("variance_floor")
        or ebpr.posterior_temperature != ebpr_record.get("temperature")
        or ebpr.ridge != ebpr_record.get("ridge")
        or ebpr.feature_seed != ebpr_record.get("feature_seed")
        or ebpr.tie_token != ebpr_record.get("tie_token")
        or ebpr.tie_token != canonical_tie_token(str(manifest.get("config_digest")))
    ):
        raise V04ARunnerError(
            f"source checkpoint task/config binding differs: {task_id}"
        )
    return bpr, ebpr


def _verify_raw_binding(
    run: Mapping[str, Any], raw_view: Path, candidate_ids: Iterable[str]
) -> None:
    binding = run.get("raw_delta")
    if not isinstance(binding, Mapping):
        raise V04ARunnerError("asset census lacks a frozen Raw-Delta binding")
    config_path = raw_view / "config.json"
    if not config_path.is_file() or sha256_file(config_path) != binding.get(
        "config_sha256"
    ):
        raise V04ARunnerError("frozen Raw-Delta config moved or changed")
    expected = binding.get("source_rkme_sha256")
    candidates = set(candidate_ids)
    if not isinstance(expected, Mapping) or set(expected) != candidates:
        raise V04ARunnerError("Raw-Delta source digest binding is incomplete")
    for candidate, digest in expected.items():
        path = raw_view / "source" / f"{candidate}.npz"
        if not path.is_file() or sha256_file(path) != digest:
            raise V04ARunnerError(
                f"frozen Raw-Delta source moved or changed: {candidate}"
            )


def _canonical_raw_delta_points(
    probe: RewardFreeProbe,
    task_id: str,
    adapter: Mapping[str, Any],
) -> np.ndarray:
    tasks = adapter.get("tasks")
    if not isinstance(tasks, Mapping) or task_id not in tasks:
        raise GateFailure("NO_GO_RAW_PARITY", f"Raw adapter lacks task {task_id}")
    task = tasks[task_id]
    observation_dim = int(task["observation_dim"])
    action_dim = int(task["action_dim"])
    if (
        probe.observation.shape[1] != observation_dim
        or probe.action.shape[1] != action_dim
    ):
        raise GateFailure(
            "NO_GO_RAW_PARITY", f"Raw adapter native dimensions differ for {task_id}"
        )
    max_observation = int(adapter["max_observation_dim"])
    max_action = int(adapter["max_action_dim"])
    observation_mean = np.asarray(adapter["observation_mean"], dtype=np.float64)
    observation_std = np.asarray(adapter["observation_std"], dtype=np.float64)
    action_mean = np.asarray(adapter["action_mean"], dtype=np.float64)
    action_std = np.asarray(adapter["action_std"], dtype=np.float64)
    observation = np.zeros((probe.transition_count, max_observation), dtype=np.float64)
    next_observation = np.zeros_like(observation)
    action = np.zeros((probe.transition_count, max_action), dtype=np.float64)
    observation[:, :observation_dim] = (
        probe.observation - observation_mean[:observation_dim]
    ) / observation_std[:observation_dim]
    next_observation[:, :observation_dim] = (
        probe.next_observation - observation_mean[:observation_dim]
    ) / observation_std[:observation_dim]
    action[:, :action_dim] = (probe.action - action_mean[:action_dim]) / action_std[
        :action_dim
    ]
    # This float32 round-trip is part of the frozen v0.31 numeric path:
    # GlobalCanonicalizerSpec -> TransitionBank -> V_DELTA_ONLY.
    observation32 = observation.astype(np.float32)
    next_observation32 = next_observation.astype(np.float32)
    action32 = action.astype(np.float32)
    return np.concatenate(
        (next_observation32 - observation32, action32), axis=1
    ).astype(np.float64)


def raw_delta_task5_scores(
    *,
    probe: RewardFreeProbe,
    task_id: str,
    candidate_ids: Sequence[str],
    raw_view_root: Path,
    raw_adapter: Mapping[str, Any],
    source_models: Mapping[str, ReducedRKME] | None = None,
    block_size: int = 2048,
) -> dict[str, float]:
    """Use the exact frozen v0.31 EmpiricalKME/ReducedRKME/MMD operators."""

    config = _json(raw_view_root / "config.json")
    if raw_adapter.get("canonicalizer_digest") != config.get("canonicalizer_digest"):
        raise GateFailure(
            "NO_GO_RAW_PARITY", "Raw adapter canonicalizer binding differs"
        )
    points = _canonical_raw_delta_points(probe, task_id, raw_adapter)
    if points.shape[1] != int(config.get("feature_width", -1)):
        raise GateFailure("NO_GO_RAW_PARITY", "Raw query feature width differs")
    dataset_digest = sha256_ndarrays(
        {"points": points, "episode_offsets": probe.episode_offsets}
    )
    query = _empirical(
        points,
        probe.episode_offsets,
        bandwidth=float(config["bandwidth"]),
        protocol_id=str(config["protocol_id"]),
        dataset_digest=dataset_digest,
        task=task_id,
        backend="numpy",
        block_size=block_size,
    )
    scores: dict[str, float] = {}
    for candidate in candidate_ids:
        source = (
            ReducedRKME.load_npz(raw_view_root / "source" / f"{candidate}.npz")
            if source_models is None
            else source_models[candidate]
        )
        if (
            source.protocol_id != config["protocol_id"]
            or not np.isclose(
                source.bandwidth,
                float(config["bandwidth"]),
                rtol=0.0,
                atol=1.0e-12,
            )
            or source.source_task != task_id
            or source.supports.shape[1] != points.shape[1]
        ):
            raise GateFailure(
                "NO_GO_RAW_PARITY", f"Raw source artifact protocol drift: {candidate}"
            )
        scores[candidate] = -_distance(
            query, source, backend="numpy", block_size=block_size
        )
    return scores


def _ranked(
    scores: Mapping[str, float],
    *,
    config_digest: str,
    raw_ties: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    if len(scores) != 5 or any(
        not math.isfinite(float(value)) for value in scores.values()
    ):
        raise V04ARunnerError("ranking scores must be one finite TASK_5 vector")
    if raw_ties is None:
        tie = {key: tie_break_key(config_digest, key) for key in scores}
    else:
        if set(raw_ties) != set(scores):
            raise V04ARunnerError("Raw tie tokens do not cover TASK_5")
        tie = dict(raw_ties)
    order = sorted(scores, key=lambda key: (-float(scores[key]), tie[key]))
    return [
        {
            "rank": rank,
            "opaque_candidate_id": candidate,
            "score": float(scores[candidate]),
            "tie_break_token": tie[candidate],
        }
        for rank, candidate in enumerate(order, 1)
    ]


def _entropy(posterior: Mapping[str, float]) -> float:
    values = np.asarray(list(posterior.values()), dtype=np.float64)
    return float(-np.sum(values * np.log(np.maximum(values, np.finfo(float).tiny))))


def _score_fp_impl(args: argparse.Namespace) -> Mapping[str, Any]:
    if any(
        (args.run_dir / name).exists()
        for name in (
            "rankings.seal.json",
            "oracle_binding.json",
            "metrics.jsonl",
            "summary.json",
        )
    ):
        raise V04ARunnerError("score-fp cannot run after ranking seal or oracle access")
    run, config, config_digest = _prepared(args.run_dir)
    rows, layout = _sanitized_layout(args.run_dir, run)
    manifest = _json(args.run_dir / "source_observation_model_manifest.json")
    source_fit_manifest_path = args.run_dir / "source_fit_manifest.json"
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("source_only") is not True
        or manifest.get("source_fit_manifest_sha256")
        != sha256_file(source_fit_manifest_path)
        or sha256_json(_json(source_fit_manifest_path))
        != run.get("source_fit_manifest_payload_digest")
        or manifest.get("config_digest") != config_digest
        or manifest.get("fixed_probe_protocol_id") != run.get("fixed_probe_protocol_id")
        or set(manifest.get("models", {})) != set(layout)
        or manifest.get("target_contexts_read") != 0
    ):
        raise V04ARunnerError("source models are absent or not source-only")
    utility_artifact = _json(args.run_dir / "source_utility.json")
    if (
        sha256_file(args.run_dir / "source_utility.json")
        != manifest.get("source_utility_sha256")
        or utility_artifact.get("source_only") is not True
    ):
        raise V04ARunnerError("source utility artifact differs from model manifest")
    if set(utility_artifact.get("tasks", {})) != set(layout):
        raise V04ARunnerError("source utility artifact does not cover six tasks")
    method_cards = _json(args.run_dir / "method_cards.json")
    if sha256_json(method_cards) != run.get("method_cards_payload_digest"):
        raise V04ARunnerError("method cards differ from prepare binding")
    raw_binding = run.get("raw_delta")
    if (
        not isinstance(raw_binding, Mapping)
        or raw_binding.get("root_scope") != "run_relative_source_only_copy"
        or raw_binding.get("contains_query_artifacts") is not False
    ):
        raise V04ARunnerError("score stage lacks a source-only Raw operator")
    raw_relative = Path(str(raw_binding["root"]))
    if raw_relative.is_absolute() or ".." in raw_relative.parts:
        raise V04ARunnerError("Raw source operator escapes the run directory")
    raw_view = (args.run_dir / raw_relative).resolve()
    if not raw_view.is_dir():
        raise V04ARunnerError("source-only Raw-Delta operator is unavailable")
    raw_adapter = _raw_adapter(args.run_dir, run)
    all_candidates = {
        candidate for task in layout.values() for candidate in task["candidate_ids"]
    }
    _verify_raw_binding(run, raw_view, all_candidates)
    development = [row for row in rows if row["role"] == "development"]
    methods = list(PRIMARY_METHODS)
    if bool(config["controls"].get("enable_hybrid", False)):
        methods.append(HYBRID_METHOD)
    output_rankings: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    cache: dict[str, tuple[BPRGaussianModel, EBPRFixedProbe]] = {}
    raw_source_cache: dict[str, dict[str, ReducedRKME]] = {
        task_id: {
            candidate: ReducedRKME.load_npz(raw_view / "source" / f"{candidate}.npz")
            for candidate in task["candidate_ids"]
        }
        for task_id, task in layout.items()
    }
    try:
        for row in development:
            task_id = str(row["task_id"])
            task = layout[task_id]
            utility = utility_artifact["tasks"][task_id]
            if task_id not in cache:
                cache[task_id] = _load_source_models(
                    args.run_dir,
                    task_id,
                    manifest,
                    task_layout=task,
                    utility=utility,
                )
            bpr, ebpr = cache[task_id]
            candidates = tuple(task["candidate_ids"])
            for budget in BUDGET_EPISODES:
                probe_started = time.perf_counter()
                probe = _projected_probe(args.run_dir, row, budget)
                shared_probe_load_seconds = time.perf_counter() - probe_started
                ledger = BudgetLedger.for_budget(budget)
                started = time.perf_counter()
                raw_scores = raw_delta_task5_scores(
                    probe=probe,
                    task_id=task_id,
                    candidate_ids=candidates,
                    raw_view_root=raw_view,
                    raw_adapter=raw_adapter,
                    source_models=raw_source_cache[task_id],
                    block_size=args.block_size,
                )
                raw_seconds = time.perf_counter() - started
                bpr_started = time.perf_counter()
                summaries = summarize_probe(probe)
                bpr_posterior = bpr.posterior_dict(summaries)
                bpr_scores = bpr.utility_scores(summaries)
                bpr_nll = bpr.target_predictive_nll(summaries)
                bpr_seconds = time.perf_counter() - bpr_started
                ebpr_started = time.perf_counter()
                episodes = _transition_episodes(probe)
                ebpr_selection = ebpr.select_map(episodes)
                ebpr_seconds = time.perf_counter() - ebpr_started
                ebpr_scores = {
                    task["paired_policy_by_type"][type_id]: probability
                    for type_id, probability in ebpr_selection.posterior.items()
                }
                method_values: dict[
                    str,
                    tuple[
                        Mapping[str, float],
                        float,
                        Mapping[str, float] | None,
                        float | None,
                    ],
                ] = {
                    RAW_METHOD: (raw_scores, raw_seconds, None, None),
                    BPR_METHOD: (
                        bpr_scores,
                        bpr_seconds,
                        bpr_posterior,
                        bpr_nll,
                    ),
                    EBPR_METHOD: (
                        ebpr_scores,
                        ebpr_seconds,
                        ebpr_selection.posterior,
                        ebpr_selection.target_predictive_nll,
                    ),
                }
                if HYBRID_METHOD in methods:
                    hybrid_started = time.perf_counter()
                    hybrid_selection = ebpr.select_hybrid(episodes, utility)
                    if hybrid_selection.expected_utility is None:
                        raise V04ARunnerError(
                            "EBPR hybrid did not return expected utility scores"
                        )
                    method_values[HYBRID_METHOD] = (
                        hybrid_selection.expected_utility,
                        time.perf_counter() - hybrid_started,
                        hybrid_selection.posterior,
                        hybrid_selection.target_predictive_nll,
                    )
                raw_ties = dict(task["raw_tie_break_tokens"])
                for method_id in methods:
                    scores, seconds, posterior, predictive_nll = method_values[
                        method_id
                    ]
                    ranking = _ranked(
                        scores,
                        config_digest=config_digest,
                        raw_ties=raw_ties if method_id == RAW_METHOD else None,
                    )
                    ledger_payload = ledger.to_dict()
                    output_rankings.append(
                        {
                            "schema": SCHEMA,
                            "stage": "PUBLIC_RANKING_PRE_ORACLE",
                            "context_id": row["context_id"],
                            "context_role": "development",
                            "task_id": task_id,
                            "method_id": method_id,
                            "method_version": "0.4a.0",
                            "access_track": "BI0-FP-RF",
                            "candidate_scope": "TASK_5",
                            "faithfulness": method_cards["cards"][method_id][
                                "identity"
                            ],
                            "source_evidence_privileges": method_cards["cards"][
                                method_id
                            ]["source_privilege"],
                            "budget_episodes": budget,
                            "probe_membership_digest": probe.probe_membership_digest,
                            **{
                                key: value
                                for key, value in ledger_payload.items()
                                if key != "schema"
                            },
                            "budget_ledger": ledger_payload,
                            "selected_opaque_candidate_id": ranking[0][
                                "opaque_candidate_id"
                            ],
                            "score_semantics": (
                                "negative_mmd"
                                if method_id == RAW_METHOD
                                else "posterior_expected_source_utility"
                                if method_id in {BPR_METHOD, HYBRID_METHOD}
                                else "paired_source_type_posterior"
                            ),
                            "ranking": ranking,
                            "runtime_seconds": float(seconds),
                            "runtime_accounting": "method-specific adapter plus score; source models warm-cached; shared probe-file load reported separately",
                            "shared_probe_load_seconds": float(
                                shared_probe_load_seconds
                            ),
                            "peak_memory_ru_maxrss": int(
                                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            ),
                            "status": "OK",
                        }
                    )
                    if posterior is not None:
                        if predictive_nll is None:
                            raise V04ARunnerError(
                                "posterior trace lacks predictive NLL"
                            )
                        traces.append(
                            {
                                "schema": SCHEMA,
                                "context_id": row["context_id"],
                                "task_id": task_id,
                                "method_id": method_id,
                                "budget_episodes": budget,
                                "probe_membership_digest": probe.probe_membership_digest,
                                "posterior": posterior,
                                "posterior_entropy": _entropy(posterior),
                                "target_predictive_nll": float(predictive_nll),
                                "fit_on_target": False,
                            }
                        )
    except GateFailure as error:
        result = {
            "schema": SCHEMA,
            "stage": "score-fp",
            "status": error.status,
            "message": str(error),
            "oracle_access": False,
        }
        _publish(args.run_dir / "score_fp_status.json", result, resume=args.resume)
        return result
    output_rankings.sort(
        key=lambda row: (row["context_id"], row["budget_episodes"], row["method_id"])
    )
    traces.sort(
        key=lambda row: (row["context_id"], row["budget_episodes"], row["method_id"])
    )
    rankings_sha256 = _publish_jsonl(
        args.run_dir / "rankings.jsonl", output_rankings, resume=args.resume
    )
    traces_sha256 = _publish_jsonl(
        args.run_dir / "posterior_traces.jsonl", traces, resume=args.resume
    )
    result = {
        "schema": SCHEMA,
        "stage": "score-fp",
        "status": "COMPLETE",
        "oracle_access": False,
        "development_context_count": len(development),
        "method_count": len(methods),
        "budget_count": len(BUDGET_EPISODES),
        "ranking_record_count": len(output_rankings),
        "posterior_trace_count": len(traces),
        "rankings_sha256": rankings_sha256,
        "posterior_traces_sha256": traces_sha256,
        "score_visible_input_scope": "sanitized_reward_free_and_source_only",
        "runtime_accounting": "method-specific adapter plus score; source models warm-cached; shared probe-file load excluded and reported per row",
    }
    _publish(args.run_dir / "score_fp_status.json", result, resume=args.resume)
    return result


def score_fp(args: argparse.Namespace) -> Mapping[str, Any]:
    """Run target scoring and persist every fail-closed terminal status."""

    if any(
        (args.run_dir / name).exists()
        for name in (
            "rankings.seal.json",
            "oracle_binding.json",
            "metrics.jsonl",
            "summary.json",
        )
    ):
        raise V04ARunnerError("score-fp cannot run after ranking seal or oracle access")
    try:
        return _score_fp_impl(args)
    except GateFailure as error:
        result = {
            "schema": SCHEMA,
            "stage": "score-fp",
            "status": error.status,
            "message": str(error),
            "details": error.details,
            "error_type": type(error).__name__,
            "oracle_access": False,
        }
    except (
        BPRModelError,
        EBPRError,
        V04ARunnerError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        result = {
            "schema": SCHEMA,
            "stage": "score-fp",
            "status": "NO_GO_FP_SCORING",
            "message": str(error),
            "error_type": type(error).__name__,
            "oracle_access": False,
        }
    _publish(args.run_dir / "score_fp_status.json", result, resume=args.resume)
    return result


def _validate_ranking_records(
    run_dir: Path,
    run: Mapping[str, Any],
    config: Mapping[str, Any],
    rankings: Sequence[Mapping[str, Any]],
) -> None:
    """Require the exact sealed development grid and every BI0 accounting field."""

    contexts, layout = _sanitized_layout(run_dir, run)
    development = {
        str(row["context_id"]): row for row in contexts if row["role"] == "development"
    }
    methods = list(PRIMARY_METHODS)
    if bool(config["controls"].get("enable_hybrid", False)):
        methods.append(HYBRID_METHOD)
    expected_keys = {
        (context_id, budget, method)
        for context_id in development
        for budget in BUDGET_EPISODES
        for method in methods
    }
    if len(rankings) != len(expected_keys):
        raise V04ARunnerError(
            f"ranking grid requires {len(expected_keys)} records, got {len(rankings)}"
        )
    observed: set[tuple[str, int, str]] = set()
    for raw_row in rankings:
        row = dict(raw_row)
        try:
            context_id = str(row["context_id"])
            task_id = str(row["task_id"])
            method_id = str(row["method_id"])
            budget = int(row["budget_episodes"])
        except (KeyError, TypeError, ValueError) as error:
            raise V04ARunnerError("ranking record identity is malformed") from error
        key = (context_id, budget, method_id)
        if key not in expected_keys or key in observed:
            raise V04ARunnerError(
                f"ranking grid has an unexpected/duplicate record: {key}"
            )
        observed.add(key)
        context = development[context_id]
        if (
            row.get("schema") != SCHEMA
            or row.get("stage") != "PUBLIC_RANKING_PRE_ORACLE"
            or row.get("status") != "OK"
            or row.get("context_role") != "development"
            or row.get("task_id") != context["task_id"]
            or row.get("method_version") != "0.4a.0"
            or row.get("access_track") != "BI0-FP-RF"
            or row.get("candidate_scope") != "TASK_5"
            or row.get("probe_membership_digest") != context["probe_membership_digest"]
            or not isinstance(row.get("faithfulness"), str)
            or not row.get("faithfulness")
            or not isinstance(row.get("source_evidence_privileges"), str)
            or not row.get("source_evidence_privileges")
        ):
            raise V04ARunnerError(f"ranking record protocol differs: {key}")
        ledger = BudgetLedger.for_budget(budget)
        ledger_payload = ledger.to_dict()
        if row.get("budget_ledger") != ledger_payload or any(
            row.get(name) != value
            for name, value in ledger_payload.items()
            if name != "schema"
        ):
            raise V04ARunnerError(f"ranking budget ledger differs: {key}")
        ranking = row.get("ranking")
        candidates = tuple(layout[task_id]["candidate_ids"])
        if not isinstance(ranking, list) or len(ranking) != 5:
            raise V04ARunnerError(f"ranking is not TASK_5: {key}")
        ranked_candidates: list[str] = []
        scores: dict[str, float] = {}
        ties: dict[str, str] = {}
        for rank, item in enumerate(ranking, 1):
            if not isinstance(item, Mapping) or item.get("rank") != rank:
                raise V04ARunnerError(f"ranking positions are malformed: {key}")
            candidate = str(item.get("opaque_candidate_id"))
            try:
                score = float(item["score"])
            except (KeyError, TypeError, ValueError) as error:
                raise V04ARunnerError(f"ranking score is malformed: {key}") from error
            tie = item.get("tie_break_token")
            if not math.isfinite(score) or not isinstance(tie, str) or not tie:
                raise V04ARunnerError(f"ranking score/tie is malformed: {key}")
            ranked_candidates.append(candidate)
            scores[candidate] = score
            ties[candidate] = tie
        if len(scores) != 5 or set(scores) != set(candidates):
            raise V04ARunnerError(f"ranking candidate set differs: {key}")
        expected_ties = (
            dict(layout[task_id]["raw_tie_break_tokens"])
            if method_id == RAW_METHOD
            else {
                candidate: tie_break_key(str(run["config_digest"]), candidate)
                for candidate in candidates
            }
        )
        if ties != expected_ties:
            raise V04ARunnerError(f"ranking tie rule differs: {key}")
        expected_order = sorted(
            candidates,
            key=lambda candidate: (-scores[candidate], expected_ties[candidate]),
        )
        if (
            ranked_candidates != expected_order
            or row.get("selected_opaque_candidate_id") != expected_order[0]
        ):
            raise V04ARunnerError(f"ranking order/selection differs: {key}")
        runtime = row.get("runtime_seconds")
        memory = row.get("peak_memory_ru_maxrss")
        if (
            isinstance(runtime, bool)
            or not isinstance(runtime, (int, float))
            or not math.isfinite(float(runtime))
            or float(runtime) < 0.0
            or isinstance(memory, bool)
            or not isinstance(memory, int)
            or memory < 0
        ):
            raise V04ARunnerError(f"ranking runtime/memory differs: {key}")
    if observed != expected_keys:
        raise V04ARunnerError("ranking grid is incomplete")
    score_status_path = run_dir / "score_fp_status.json"
    if score_status_path.is_symlink() or not score_status_path.is_file():
        raise V04ARunnerError(
            "ranking seal requires an immutable score completion record"
        )
    status = _json(score_status_path)
    if (
        status.get("schema") != SCHEMA
        or status.get("stage") != "score-fp"
        or status.get("status") != "COMPLETE"
        or status.get("oracle_access") is not False
        or status.get("development_context_count") != 24
        or status.get("method_count") != len(methods)
        or status.get("budget_count") != len(BUDGET_EPISODES)
        or status.get("ranking_record_count") != len(rankings)
        or status.get("rankings_sha256") != sha256_file(run_dir / "rankings.jsonl")
    ):
        raise V04ARunnerError("ranking bytes differ from score completion record")


def seal_stage(args: argparse.Namespace) -> Mapping[str, Any]:
    run, config, _ = _prepared(args.run_dir)
    rankings = _read_jsonl(args.run_dir / "rankings.jsonl")
    _validate_ranking_records(args.run_dir, run, config, rankings)
    if (args.run_dir / "oracle_binding.json").exists():
        raise V04ARunnerError("oracle was already bound before this seal attempt")
    seal = seal_rankings(rankings)
    _publish(args.run_dir / "rankings.seal.json", seal.to_dict(), resume=args.resume)
    result = {
        "schema": SCHEMA,
        "stage": "seal-rankings",
        "status": "SEALED_PRE_ORACLE",
        "ranking_record_count": len(rankings),
        "rankings_digest": seal.rankings_digest,
    }
    _publish(args.run_dir / "seal_status.json", result, resume=args.resume)
    return result


def _oracle_returns(
    root: Path,
    context_id: str,
    task_id: str,
    candidates: Sequence[str],
    candidate_bundle_digests: Mapping[str, str],
) -> tuple[dict[str, float], dict[str, str], dict[str, tuple[float, ...]]]:
    evidence = _evidence_root(root)
    values: dict[str, float] = {}
    digests: dict[str, str] = {}
    episodes: dict[str, tuple[float, ...]] = {}
    common_seed_bank: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    for candidate in candidates:
        path = evidence / context_id / f"{candidate}.json"
        if path.is_symlink() or not path.is_file():
            raise GateFailure(
                "NO_GO_TARGET_ORACLE_GAP",
                f"target oracle missing: {context_id}/{candidate}",
            )
        value, episode_returns, reset_seeds, policy_seeds = _validated_return_record(
            _json(path),
            context_id=context_id,
            task_id=task_id,
            candidate_id=candidate,
            bundle_digest=str(candidate_bundle_digests[candidate]),
            failure_status="NO_GO_TARGET_ORACLE_GAP",
        )
        seed_bank = (reset_seeds, policy_seeds)
        if common_seed_bank is None:
            common_seed_bank = seed_bank
        elif seed_bank != common_seed_bank:
            raise GateFailure(
                "NO_GO_TARGET_ORACLE_GAP",
                f"five target oracle candidates lack a common seed bank: {context_id}",
            )
        values[candidate] = value
        episodes[candidate] = episode_returns
        digests[f"{context_id}/{candidate}"] = sha256_file(path)
    if len(values) != 5:
        raise GateFailure(
            "NO_GO_TARGET_ORACLE_GAP", f"{context_id}: oracle is not TASK_5"
        )
    return values, digests, episodes


def _posterior_trace_map(
    run_dir: Path,
    run: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    contexts, layout = _sanitized_layout(run_dir, run)
    development = {
        str(row["context_id"]): row for row in contexts if row["role"] == "development"
    }
    methods = [BPR_METHOD, EBPR_METHOD]
    if bool(config["controls"].get("enable_hybrid", False)):
        methods.append(HYBRID_METHOD)
    expected = {
        (context_id, budget, method)
        for context_id in development
        for budget in BUDGET_EPISODES
        for method in methods
    }
    traces = _read_jsonl(run_dir / "posterior_traces.jsonl")
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in traces:
        key = (
            str(row.get("context_id")),
            int(row.get("budget_episodes", -1)),
            str(row.get("method_id")),
        )
        context = development.get(key[0])
        if key not in expected or key in result or context is None:
            raise V04ARunnerError(f"posterior trace grid differs: {key}")
        posterior = row.get("posterior")
        type_ids = set(layout[str(context["task_id"])]["source_type_ids"])
        if not isinstance(posterior, Mapping) or set(posterior) != type_ids:
            raise V04ARunnerError(f"posterior type support differs: {key}")
        probabilities = np.asarray(list(posterior.values()), dtype=np.float64)
        entropy = float(row.get("posterior_entropy", float("nan")))
        predictive_nll = float(row.get("target_predictive_nll", float("nan")))
        if (
            row.get("schema") != SCHEMA
            or row.get("task_id") != context["task_id"]
            or row.get("probe_membership_digest") != context["probe_membership_digest"]
            or row.get("fit_on_target") is not False
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or not np.isclose(np.sum(probabilities), 1.0, rtol=0.0, atol=1.0e-10)
            or not math.isfinite(entropy)
            or not np.isclose(entropy, _entropy(posterior), rtol=0.0, atol=1.0e-12)
            or not math.isfinite(predictive_nll)
        ):
            raise V04ARunnerError(f"posterior trace payload differs: {key}")
        result[key] = row
    if set(result) != expected:
        raise V04ARunnerError(
            f"posterior trace grid requires {len(expected)} rows, got {len(result)}"
        )
    status = _json(run_dir / "score_fp_status.json")
    if status.get("posterior_trace_count") != len(result) or status.get(
        "posterior_traces_sha256"
    ) != sha256_file(run_dir / "posterior_traces.jsonl"):
        raise V04ARunnerError("posterior traces differ from score completion record")
    return result


def oracle_evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    run, config, _ = _prepared(args.run_dir)
    rankings = _read_jsonl(args.run_dir / "rankings.jsonl")
    seal = RankingSeal.from_dict(_json(args.run_dir / "rankings.seal.json"))
    verify_ranking_seal(seal, rankings)
    _validate_ranking_records(args.run_dir, run, config, rankings)
    contexts, layout = _sanitized_layout(args.run_dir, run)
    traces = _posterior_trace_map(args.run_dir, run, config)
    # The oracle root is deliberately resolved only after the ranking bytes and
    # seal are verified above.
    oracle_root = args.oracle_root.resolve()
    epsilon = float(config["controls"]["epsilon_optimal"])
    metrics: list[dict[str, Any]] = []
    binding_digests: dict[str, str] = {}
    cache: dict[tuple[str, tuple[str, ...]], dict[str, float]] = {}
    episode_cache: dict[str, dict[str, tuple[float, ...]]] = {}
    try:
        oracle_root, oracle_namespace_provenance = _validated_evidence_namespace(
            oracle_root,
            policy_market_id=str(run["policy_market_id"]),
            expected_contexts={
                str(row["context_id"]): (str(row["role"]), str(row["task_id"]))
                for row in contexts
            },
            failure_status="NO_GO_TARGET_ORACLE_GAP",
        )
        for row in rankings:
            candidate_ids = tuple(
                item["opaque_candidate_id"] for item in row["ranking"]
            )
            key = (str(row["context_id"]), tuple(sorted(candidate_ids)))
            if key not in cache:
                task = layout[str(row["task_id"])]
                oracle, digests, episode_returns = _oracle_returns(
                    oracle_root,
                    key[0],
                    str(row["task_id"]),
                    key[1],
                    task["candidate_bundle_digests"],
                )
                cache[key] = oracle
                episode_cache[key[0]] = episode_returns
                binding_digests.update(digests)
            scores = {
                item["opaque_candidate_id"]: float(item["score"])
                for item in row["ranking"]
            }
            metric = evaluate_ranking(
                context_id=str(row["context_id"]),
                task_id=str(row["task_id"]),
                method_id=str(row["method_id"]),
                budget_episodes=int(row["budget_episodes"]),
                ranked_candidate_ids=candidate_ids,
                scores=scores,
                oracle_returns=cache[key],
                epsilon=epsilon,
            )
            trace_key = (
                str(row["context_id"]),
                int(row["budget_episodes"]),
                str(row["method_id"]),
            )
            trace = None if row["method_id"] == RAW_METHOD else traces[trace_key]
            score_margin = float(scores[candidate_ids[0]] - scores[candidate_ids[1]])
            metrics.append(
                {
                    "schema": SCHEMA,
                    **metric,
                    "visible_transition_count": row["visible_transition_count"],
                    "interaction_cost_steps": row["interaction_cost_steps"],
                    "candidate_conditioned_steps": row["candidate_conditioned_steps"],
                    "reward_queries": row["reward_queries"],
                    "total_target_steps": row["total_target_steps"],
                    "runtime_seconds": row["runtime_seconds"],
                    "runtime_accounting": row.get("runtime_accounting"),
                    "shared_probe_load_seconds": row.get(
                        "shared_probe_load_seconds", 0.0
                    ),
                    "peak_memory_ru_maxrss": row["peak_memory_ru_maxrss"],
                    "posterior_entropy": (
                        None if trace is None else trace["posterior_entropy"]
                    ),
                    "posterior_max_probability": (
                        None
                        if trace is None
                        else float(max(trace["posterior"].values()))
                    ),
                    "target_predictive_nll": (
                        None if trace is None else trace["target_predictive_nll"]
                    ),
                    "selection_score_margin": score_margin,
                    "utility_linf_robust_radius": (
                        0.5 * score_margin
                        if row["method_id"] in {BPR_METHOD, HYBRID_METHOD}
                        else None
                    ),
                    "record_status": row["status"],
                    "rankings_digest": seal.rankings_digest,
                    "context_role": "development",
                    "formal": False,
                }
            )
        ranking_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(
            list
        )
        for ranking_row in rankings:
            ranking_groups[
                (
                    str(ranking_row["method_id"]),
                    int(ranking_row["budget_episodes"]),
                )
            ].append(ranking_row)
        bootstrap_base_seed = int(config["controls"]["hierarchical_bootstrap_seed"])
        bootstrap_replicates = int(
            config["controls"]["hierarchical_bootstrap_replicates"]
        )
        uncertainty_rows = []
        for (method_id, budget), group in sorted(ranking_groups.items()):
            group_seed = (
                bootstrap_base_seed
                + int(
                    sha256_json(
                        {
                            "domain": "v04a-hierarchical-bootstrap",
                            "method_id": method_id,
                            "budget_episodes": budget,
                            "rankings_digest": seal.rankings_digest,
                        }
                    )[:16],
                    16,
                )
            ) % (2**63 - 1)
            uncertainty_rows.append(
                hierarchical_bootstrap_intervals(
                    group,
                    episode_cache,
                    epsilon=epsilon,
                    seed=group_seed,
                    replicates=bootstrap_replicates,
                )
            )
    except GateFailure as error:
        result = {
            "schema": SCHEMA,
            "stage": "oracle-evaluate",
            "status": error.status,
            "message": str(error),
            "rankings_digest": seal.rankings_digest,
        }
        _publish(
            args.run_dir / "oracle_evaluate_status.json", result, resume=args.resume
        )
        return result
    except (
        V04ARunnerError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        FloatingPointError,
    ) as error:
        result = {
            "schema": SCHEMA,
            "stage": "oracle-evaluate",
            "status": "NO_GO_ORACLE_EVALUATION",
            "message": str(error),
            "error_type": type(error).__name__,
            "rankings_digest": seal.rankings_digest,
        }
        _publish(
            args.run_dir / "oracle_evaluate_status.json", result, resume=args.resume
        )
        return result
    uncertainty = {
        "schema": SCHEMA,
        "stage": "oracle-uncertainty-after-ranking-seal",
        "status": "COMPLETE",
        "rankings_digest": seal.rankings_digest,
        "oracle_episode_resampling": "paired common-reset index within context",
        "rows": uncertainty_rows,
    }
    uncertainty_sha256 = _publish(
        args.run_dir / "uncertainty.json", uncertainty, resume=args.resume
    )
    metrics.sort(
        key=lambda row: (row["context_id"], row["budget_episodes"], row["method_id"])
    )
    binding = {
        "schema": SCHEMA,
        "stage": "oracle-binding-after-ranking-seal",
        "status": "BOUND",
        "oracle_root": str(oracle_root),
        "rankings_digest": seal.rankings_digest,
        "record_digests": binding_digests,
        "input_namespace_provenance": oracle_namespace_provenance,
        "uncertainty_sha256": uncertainty_sha256,
        "development_only": True,
    }
    oracle_binding_sha256 = _publish(
        args.run_dir / "oracle_binding.json", binding, resume=args.resume
    )
    metrics_sha256 = _publish_jsonl(
        args.run_dir / "metrics.jsonl", metrics, resume=args.resume
    )
    result = {
        "schema": SCHEMA,
        "stage": "oracle-evaluate",
        "status": "COMPLETE",
        "metric_record_count": len(metrics),
        "rankings_digest": seal.rankings_digest,
        "oracle_binding_sha256": oracle_binding_sha256,
        "metrics_sha256": metrics_sha256,
        "uncertainty_sha256": uncertainty_sha256,
    }
    _publish(args.run_dir / "oracle_evaluate_status.json", result, resume=args.resume)
    return result


def summarize(args: argparse.Namespace) -> Mapping[str, Any]:
    run, config, _ = _prepared(args.run_dir)
    rankings = _read_jsonl(args.run_dir / "rankings.jsonl")
    seal = RankingSeal.from_dict(_json(args.run_dir / "rankings.seal.json"))
    verify_ranking_seal(seal, rankings)
    _validate_ranking_records(args.run_dir, run, config, rankings)
    oracle_status = _json(args.run_dir / "oracle_evaluate_status.json")
    if oracle_status.get("status") != "COMPLETE":
        raise V04ARunnerError("cannot summarize an incomplete oracle join")
    oracle_binding = _json(args.run_dir / "oracle_binding.json")
    uncertainty = _json(args.run_dir / "uncertainty.json")
    rows = _read_jsonl(args.run_dir / "metrics.jsonl")
    if (
        len(rows) != len(rankings)
        or oracle_status.get("metric_record_count") != len(rows)
        or oracle_status.get("rankings_digest") != seal.rankings_digest
        or oracle_status.get("metrics_sha256")
        != sha256_file(args.run_dir / "metrics.jsonl")
        or oracle_status.get("oracle_binding_sha256")
        != sha256_file(args.run_dir / "oracle_binding.json")
        or oracle_status.get("uncertainty_sha256")
        != sha256_file(args.run_dir / "uncertainty.json")
        or oracle_binding.get("uncertainty_sha256")
        != oracle_status.get("uncertainty_sha256")
        or uncertainty.get("status") != "COMPLETE"
        or uncertainty.get("stage") != "oracle-uncertainty-after-ranking-seal"
        or uncertainty.get("rankings_digest") != seal.rankings_digest
    ):
        raise V04ARunnerError("metrics do not cover the sealed ranking grid")
    raw_uncertainty_rows = uncertainty.get("rows")
    if not isinstance(raw_uncertainty_rows, list):
        raise V04ARunnerError("hierarchical uncertainty artifact is malformed")
    uncertainty_by_group: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in raw_uncertainty_rows:
        if not isinstance(row, Mapping):
            raise V04ARunnerError("hierarchical uncertainty row is malformed")
        key = (str(row.get("method_id")), int(row.get("budget_episodes", -1)))
        if key in uncertainty_by_group:
            raise V04ARunnerError("duplicate hierarchical uncertainty row")
        uncertainty_by_group[key] = row
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method_id"]), int(row["budget_episodes"]))].append(row)
    table = []
    for (method, budget), group in sorted(grouped.items()):
        if len(group) != 24:
            raise V04ARunnerError(
                f"metric group must cover 24 development contexts: {method}/{budget}"
            )
        aggregate = aggregate_metrics(group)
        uncertainty_row = uncertainty_by_group.get((method, budget))
        if uncertainty_row is None:
            raise V04ARunnerError(
                f"hierarchical uncertainty is missing: {method}/{budget}"
            )
        posterior_entropy = [
            float(row["posterior_entropy"])
            for row in group
            if row.get("posterior_entropy") is not None
        ]
        predictive_nll = [
            float(row["target_predictive_nll"])
            for row in group
            if row.get("target_predictive_nll") is not None
        ]
        posterior_max = [
            float(row["posterior_max_probability"])
            for row in group
            if row.get("posterior_max_probability") is not None
        ]
        utility_radius = [
            float(row["utility_linf_robust_radius"])
            for row in group
            if row.get("utility_linf_robust_radius") is not None
        ]
        runtimes = np.asarray(
            [float(row["runtime_seconds"]) for row in group], dtype=np.float64
        )
        shared_probe_load = np.asarray(
            [float(row.get("shared_probe_load_seconds", 0.0)) for row in group],
            dtype=np.float64,
        )
        table.append(
            {
                "method_id": method,
                "budget_episodes": budget,
                "visible_transition_count": budget * 64,
                "interaction_cost_steps": budget * 1000,
                **aggregate,
                "hierarchical_bootstrap": dict(uncertainty_row),
                "mean_posterior_entropy": (
                    None if not posterior_entropy else float(np.mean(posterior_entropy))
                ),
                "mean_posterior_max_probability": (
                    None if not posterior_max else float(np.mean(posterior_max))
                ),
                "mean_target_predictive_nll": (
                    None if not predictive_nll else float(np.mean(predictive_nll))
                ),
                "mean_utility_linf_robust_radius": (
                    None if not utility_radius else float(np.mean(utility_radius))
                ),
                "mean_runtime_seconds": float(np.mean(runtimes)),
                "p95_runtime_seconds": float(np.quantile(runtimes, 0.95)),
                "mean_shared_probe_load_seconds": float(np.mean(shared_probe_load)),
                "peak_memory_ru_maxrss": max(
                    int(row["peak_memory_ru_maxrss"]) for row in group
                ),
                "status_coverage": float(
                    np.mean([row.get("record_status") == "OK" for row in group])
                ),
                "failure_count": sum(row.get("record_status") != "OK" for row in group),
            }
        )
    if set(uncertainty_by_group) != set(grouped):
        raise V04ARunnerError("hierarchical uncertainty grid differs from metrics")
    confidence_threshold = float(config["controls"]["posterior_confidence_threshold"])
    confidence: dict[str, Any] = {}
    by_method_context: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        by_method_context[(str(row["method_id"]), str(row["context_id"]))].append(row)
    for method in (BPR_METHOD, EBPR_METHOD, HYBRID_METHOD):
        first_budgets: list[int] = []
        for (candidate_method, _), context_rows in by_method_context.items():
            if candidate_method != method:
                continue
            ordered = sorted(context_rows, key=lambda row: int(row["budget_episodes"]))
            for index, row in enumerate(ordered):
                probability = row.get("posterior_max_probability")
                selected = row["selected_candidate_id"]
                if (
                    probability is not None
                    and float(probability) >= confidence_threshold
                    and all(
                        later["selected_candidate_id"] == selected
                        for later in ordered[index:]
                    )
                ):
                    first_budgets.append(int(row["budget_episodes"]))
                    break
        confidence[method] = {
            "threshold": confidence_threshold,
            "definition": "first nested budget with max source-type posterior >= threshold and selection unchanged thereafter",
            "reached_context_count": len(first_budgets),
            "context_count": 24,
            "coverage": len(first_budgets) / 24.0,
            "mean_probe_episodes_when_reached": (
                None if not first_budgets else float(np.mean(first_budgets))
            ),
            "median_probe_episodes_when_reached": (
                None if not first_budgets else float(np.median(first_budgets))
            ),
        }
    method_cards = _json(args.run_dir / "method_cards.json")
    stage_status = {
        "fit_source": _json(args.run_dir / "fit_source_status.json").get("status"),
        "score_fp": _json(args.run_dir / "score_fp_status.json").get("status"),
        "seal_rankings": _json(args.run_dir / "seal_status.json").get("status"),
        "oracle_evaluate": oracle_status.get("status"),
    }
    summary = {
        "schema": SCHEMA,
        "stage": "summarize",
        "status": "COMPLETE_DEVELOPMENT",
        "formal": False,
        "scope": "24 frozen development contexts; not confirmatory",
        "primary_methods": list(PRIMARY_METHODS),
        "ablation_methods": [HYBRID_METHOD],
        "equal_weighting": "one row per development context within each method/budget; six tasks have four contexts each",
        "likelihood_scale_note": "target_predictive_nll is diagnostic within a method/task; BPR and EBPR units differ and are not a shared likelihood leaderboard",
        "rows": table,
        "time_to_confident_selection": confidence,
        "source_privilege_table": {
            method: {
                "access_track": method_cards["cards"][method]["access_track"],
                "source_evidence": method_cards["cards"][method]["source_privilege"],
                "faithfulness": method_cards["cards"][method]["identity"],
            }
            for method in ALL_FP_METHODS
        },
        "failure_coverage": {
            "stage_status": stage_status,
            "ranking_rows_ok": len(rows),
            "ranking_rows_failed": 0,
            "coverage": 1.0,
        },
        "source_sanity_manifest_sha256": sha256_file(
            args.run_dir / "source_observation_model_manifest.json"
        ),
        "deferred": list(config["deferred"]),
    }
    _publish(args.run_dir / "summary.json", summary, resume=args.resume)
    return summary


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", type=_path, required=True)
    prepare_parser.add_argument("--run-dir", type=_path, required=True)
    prepare_parser.add_argument("--context-index", type=_path, required=True)
    prepare_parser.add_argument("--public-policy-market", type=_path, required=True)
    prepare_parser.add_argument(
        "--deployment-private-registry", type=_path, required=True
    )
    prepare_parser.add_argument("--origin-pool-acceptance", type=_path, required=True)
    prepare_parser.add_argument("--raw-delta-root", type=_path, required=True)
    prepare_parser.add_argument("--fpo-root", type=_path, required=True)
    prepare_parser.set_defaults(handler=prepare)

    fit = subparsers.add_parser("fit-source")
    fit.add_argument("--run-dir", type=_path, required=True)
    fit.add_argument("--source-utility-root", type=_path, required=True)
    fit.add_argument("--resume", action="store_true")
    fit.set_defaults(handler=fit_source)

    score = subparsers.add_parser("score-fp")
    score.add_argument("--run-dir", type=_path, required=True)
    score.add_argument("--block-size", type=int, default=2048)
    score.add_argument("--resume", action="store_true")
    score.set_defaults(handler=score_fp)

    seal = subparsers.add_parser("seal-rankings")
    seal.add_argument("--run-dir", type=_path, required=True)
    seal.add_argument("--resume", action="store_true")
    seal.set_defaults(handler=seal_stage)

    evaluate = subparsers.add_parser("oracle-evaluate")
    evaluate.add_argument("--run-dir", type=_path, required=True)
    evaluate.add_argument("--oracle-root", type=_path, required=True)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.set_defaults(handler=oracle_evaluate)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--run-dir", type=_path, required=True)
    summary.add_argument("--resume", action="store_true")
    summary.set_defaults(handler=summarize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, "block_size", 1) <= 0:
        raise SystemExit("--block-size must be positive")
    result = args.handler(args)
    print(json.dumps(result, sort_keys=True))
    status = str(result.get("status", ""))
    return 2 if status.startswith("NO_GO") or status == "INCOMPLETE_ENGINEERING" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "GateFailure",
    "V04ARunnerError",
    "fit_source",
    "inspect_assets",
    "main",
    "oracle_evaluate",
    "prepare",
    "raw_delta_task5_scores",
    "score_fp",
    "seal_stage",
    "summarize",
]
