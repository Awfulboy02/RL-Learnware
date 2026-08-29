"""Thin orchestration for the six matched v0.5 P0 classifiers.

Collection, private label joins, and artifact publication live outside this
module.  The runner accepts only canonical reward-free episode banks, binds all
methods to one source-evidence summary, scores nested target prefixes, and uses
the existing v0.4a ranking seal before any evaluation code can run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import secrets
import sys
import time
from types import MappingProxyType
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import yaml

from policy_learnware_v0.artifacts import (
    ARTIFACTS_ROOT_ENV as _ARTIFACTS_ROOT_ENV,
    ArtifactLayoutError,
    resolve_artifacts_root,
)
from policy_learnware_v0.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_ndarrays,
)
from policy_learnware_v0.io import (
    atomic_write_json,
    atomic_write_npz,
    read_npz,
)
from policy_learnware_v0.rkme.empirical import EmpiricalKME, build_empirical_kme
from policy_learnware_v0.rkme.gaussian import GaussianKernel, calibrate_bandwidth
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.rkme.reducer import ReducedRKME, reduce_kme
from policy_learnware_v0.v03.canonicalization import (
    GlobalCanonicalizerSpec,
    GlobalNormalizer,
    NativeShapeRegistry,
    NativeTransitionBank,
    fit_global_normalizer,
)
from policy_learnware_v0.v03.source_market import (
    SELECTION_RULE,
    SourceChampionizationRecord,
    V03DeploymentPrivateEntry,
)
from policy_learnware_v0.v03.transition_views import (
    V_DELTA_ONLY,
    TransitionBank,
    apply_transition_view,
)
from policy_learnware_v0.v04a.protocol import (
    BudgetLedger,
    ProbeMembership,
    RankingSeal,
    derive_probe_membership,
    seal_rankings,
    tie_break_key,
    verify_ranking_seal,
)
from policy_learnware_v0.v05.classifiers import (
    EMPIRICAL_MMD_NN,
    KME_KRR,
    P0_METHOD_IDS,
    RAW_DELTA_RKME,
    RFF_KME_NN,
    SUMMARY_LOGREG,
    SWE_NN,
    EmpiricalMMDNN,
    EpisodeBank,
    KMEKRR,
    RFFKMENN,
    RawDeltaRKMENN,
    SWENN,
    SummaryLogReg,
    p0_method_cards,
)
from policy_learnware_v0.v05.labels import (
    CertificateResolver,
    CertifiedPolicyManifest,
    project_certificate_manifest,
)
from policy_learnware_v0.v05.metrics import (
    MARKET_30_CERT,
    TASK_5_CERT,
    PredictionRanking,
    TruthBinding,
    build_development_report,
    prediction_payload,
    require_prediction_cell_coverage,
)
from policy_learnware_v0.v05.specifications import RFFMap, SWEMap
from server.repro_fpo_ppo_v02.provenance import load_strict_json, utc_now
from server.repro_fpo_ppo_v05.blind_query_bank import (
    AuthorizedQueryViews,
    load_authorized_query,
    load_private_truth_binding,
    prepare_blinded_episode_bank,
    project_verified_source_banks,
)


Q0_COMMON_GAUSSIAN_OPEN_LOOP = "Q0_COMMON_GAUSSIAN_OPEN_LOOP"
P1_STATUS = "DEFERRED_NOT_IMPLEMENTED"
SCORER_SOURCE_FIT_SCHEMA = "policy-learnware.v05-scorer-source-fit.v1"
_OPAQUE_QUERY_ID = re.compile(r"^q-[0-9a-f]{20,64}$")
_DEVELOPMENT_BUDGETS = (1, 2, 4)
_EXPECTED_CONFIG_DIGEST = (
    "c6b808fb0ea1f8acce8406b6b21ba29908fac790c0c2bf118b31e1aa0b13b725"
)
_R4_RELATIVE = Path("v04a/runs/v04a-primary-dev-20260828-r4")
_V03_RELATIVE = Path("v03/runs/v03-main-20260827-r0")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_yaml_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise V05RunnerError(f"duplicate YAML key is forbidden: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_yaml_mapping
)


class V05RunnerError(ValueError):
    """A matched-source binding, target view, or score cell is invalid."""


def _exact_fields(value: Any, fields: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise V05RunnerError(
            f"{where} fields differ; missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return value


def load_development_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load the one preregistered development YAML without accepting drift."""

    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise V05RunnerError("development config is absent or unsafe")
    try:
        value = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise V05RunnerError("development config is not strict YAML") from error
    if not isinstance(value, Mapping):
        raise V05RunnerError("development config must be a YAML object")
    config = dict(value)
    try:
        digest = sha256_json(config)
    except (TypeError, ValueError) as error:
        raise V05RunnerError("development YAML is not canonical finite JSON") from error
    if digest != _EXPECTED_CONFIG_DIGEST:
        raise V05RunnerError("preregistered development config digest drifted")
    return config, digest


def _resolve_v05_frozen_roots(
    explicit_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve the two frozen inputs without relying on legacy path geometry."""

    try:
        raw_root = resolve_artifacts_root(explicit_root)
    except ArtifactLayoutError as error:
        raise V05RunnerError(str(error)) from error
    known = (
        raw_root,
        raw_root / "v04a",
        raw_root / "v04a" / "runs",
        raw_root / _R4_RELATIVE,
        raw_root / "v03",
        raw_root / "v03" / "runs",
        raw_root / _V03_RELATIVE,
    )
    if any(path.is_symlink() for path in known):
        raise V05RunnerError("frozen artifact roots cannot use symlinks")
    root = raw_root.resolve()
    r4_root = (root / _R4_RELATIVE).resolve()
    v03_root = (root / _V03_RELATIVE).resolve()
    if (
        not root.is_dir()
        or not r4_root.is_dir()
        or not v03_root.is_dir()
        or not r4_root.is_relative_to(root)
        or not v03_root.is_relative_to(root)
    ):
        raise V05RunnerError("canonical frozen artifact layout is absent or unsafe")
    return root, r4_root, v03_root


@dataclass(frozen=True)
class FrozenR4Assets:
    config: Mapping[str, Any]
    config_digest: str
    r4_root: Path
    v03_root: Path
    arrays_by_anchor: Mapping[str, Mapping[str, np.ndarray]]
    membership_by_anchor: Mapping[str, ProbeMembership]
    context_by_anchor: Mapping[str, str]
    task_by_anchor: Mapping[str, str]
    parent_asset_sha256: Mapping[str, str]
    parent_membership_digest: Mapping[str, str]
    native_schema_by_task: Mapping[str, str]
    certificate_manifest: CertifiedPolicyManifest
    probe_protocol_digest: str
    provenance: Mapping[str, Any]


def _frozen_bytes(path: Path, expected_sha: str, where: str) -> bytes:
    """Read, authenticate, and return one immutable frozen-file buffer."""

    expected = _digest(expected_sha, f"{where} SHA")
    if path.is_symlink() or not path.is_file():
        raise V05RunnerError(f"{where} frozen file is absent or unsafe")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise V05RunnerError(f"{where} frozen file cannot be read") from error
    if sha256_bytes(payload) != expected:
        raise V05RunnerError(f"{where} frozen file digest changed")
    return payload


def _frozen_json(
    root: Path, row: Mapping[str, Any], *, expected_sha256: str, where: str
) -> tuple[Path, dict[str, Any]]:
    relative = Path(str(row.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise V05RunnerError(f"{where} relative path is unsafe")
    path = _owned_frozen_path(root, relative, where)
    payload = _frozen_bytes(path, expected_sha256, where)

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V05RunnerError(f"{where} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise V05RunnerError(f"{where} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
        canonical_json_bytes(value)
    except V05RunnerError:
        raise
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise V05RunnerError(f"{where} is not strict finite JSON") from error
    if not isinstance(value, dict):
        raise V05RunnerError(f"{where} JSON must be a top-level object")
    return path, value


def _owned_frozen_path(root: Path, relative: Path, where: str) -> Path:
    """Resolve one owner-relative frozen path without following any symlink."""

    owner = root.resolve()
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise V05RunnerError(f"{where} path contains a symlink")
    resolved = current.resolve()
    if resolved == owner or not resolved.is_relative_to(owner):
        raise V05RunnerError(f"{where} path escapes its frozen owner root")
    return resolved


def _load_frozen_r4_assets(
    config: Mapping[str, Any],
    config_digest: str,
    artifacts_root: str | Path | None = None,
) -> FrozenR4Assets:
    """Strictly admit the 30 frozen r4 source banks and v03 label authority."""

    _, root, v03_root = _resolve_v05_frozen_roots(artifacts_root)
    frozen = config["frozen_assets"]
    r4_config = frozen["r4"]
    frozen_file_sha256 = {
        name: _digest(r4_config[name].get("file_sha256"), f"{name} file SHA")
        for name in (
            "source_fit_manifest",
            "probe_membership",
            "source_task_layout",
            "asset_census",
            "raw_delta_adapter",
        )
    }
    documents: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for name in (
        "source_fit_manifest",
        "probe_membership",
        "source_task_layout",
        "asset_census",
        "raw_delta_adapter",
    ):
        paths[name], documents[name] = _frozen_json(
            root,
            r4_config[name],
            expected_sha256=frozen_file_sha256[name],
            where=name,
        )
    v03_config = frozen["v03"]
    for name in ("deployment_private_registry", "championization"):
        frozen_file_sha256[name] = _digest(
            v03_config[name].get("file_sha256"), f"{name} file SHA"
        )
        paths[name], documents[name] = _frozen_json(
            v03_root,
            v03_config[name],
            expected_sha256=frozen_file_sha256[name],
            where=name,
        )

    source_fit = _exact_fields(
        documents["source_fit_manifest"],
        {
            "schema",
            "contains_reward_or_done",
            "contains_target_contexts",
            "contexts",
            "tasks",
        },
        "source_fit_manifest",
    )
    if (
        source_fit["schema"] != "policy-learnware.v04a-source-fit-manifest.v1"
        or source_fit["contains_reward_or_done"] is not False
        or source_fit["contains_target_contexts"] is not False
        or not isinstance(source_fit["contexts"], list)
        or len(source_fit["contexts"]) != 30
        or not isinstance(source_fit["tasks"], Mapping)
        or len(source_fit["tasks"]) != 6
    ):
        raise V05RunnerError("r4 source-fit closure differs")
    membership_doc = _exact_fields(
        documents["probe_membership"],
        {"schema", "split_seed", "fixed_probe_protocol_id", "contexts"},
        "probe_membership",
    )
    if not isinstance(membership_doc["contexts"], Mapping):
        raise V05RunnerError("probe membership contexts are malformed")
    split_seed = membership_doc["split_seed"]
    r4_probe_protocol_digest = _digest(
        membership_doc["fixed_probe_protocol_id"], "fixed_probe_protocol_id"
    )
    probe_protocol_digest = sha256_json(
        {
            "v05_probe": config["probe"],
            "r4_fixed_probe_protocol_id": r4_probe_protocol_digest,
            "probe_membership_file_sha256": frozen_file_sha256["probe_membership"],
        }
    )
    layout = _exact_fields(
        documents["source_task_layout"],
        {"schema", "tasks", "source_only"},
        "source_task_layout",
    )
    if layout["source_only"] is not True or not isinstance(layout["tasks"], Mapping):
        raise V05RunnerError("source task layout is not source-only")
    raw_adapter = documents["raw_delta_adapter"]
    required_adapter = {
        "schema",
        "identity",
        "canonicalizer_digest",
        "normalizer_digest",
        "registry_digest",
        "v031_source_partition",
        "max_observation_dim",
        "max_action_dim",
        "observation_mean",
        "observation_std",
        "action_mean",
        "action_std",
        "tasks",
        "numeric_path",
        "target_rows_read_during_fit",
    }
    _exact_fields(raw_adapter, required_adapter, "raw_delta_adapter")
    if (
        raw_adapter["schema"] != "policy-learnware.v04a-raw-delta-adapter.v1"
        or raw_adapter["target_rows_read_during_fit"] != 0
        or not isinstance(raw_adapter["tasks"], Mapping)
    ):
        raise V05RunnerError("historical Raw adapter metadata differs")

    registry_doc = _exact_fields(
        documents["deployment_private_registry"],
        {
            "schema",
            "policy_market_id",
            "intake_record_digest",
            "championization_digest",
            "entries",
            "anchor_to_opaque_learnware_id",
        },
        "deployment_private_registry",
    )
    if not isinstance(registry_doc["entries"], Mapping) or not isinstance(
        registry_doc["anchor_to_opaque_learnware_id"], Mapping
    ):
        raise V05RunnerError("deployment registry mappings are malformed")
    entries = {
        str(policy): V03DeploymentPrivateEntry.from_dict(value)
        for policy, value in registry_doc["entries"].items()
    }
    anchor_to_policy = {
        str(anchor): str(policy)
        for anchor, policy in registry_doc["anchor_to_opaque_learnware_id"].items()
    }
    if (
        len(entries) != 30
        or len(anchor_to_policy) != 30
        or set(anchor_to_policy.values()) != set(entries)
        or any(
            entries[policy].source_anchor_id != anchor
            for anchor, policy in anchor_to_policy.items()
        )
    ):
        raise V05RunnerError("deployment registry does not bind 30 anchors")
    championization = SourceChampionizationRecord.from_dict(
        documents["championization"]
    )
    if (
        championization.championization_digest
        != _digest(
            v03_config["championization"].get("championization_digest"),
            "championization_digest",
        )
        or registry_doc["championization_digest"]
        != championization.championization_digest
        or championization.competence_mode != "OBSERVE"
        or championization.attestation_plan_digest is not None
    ):
        raise V05RunnerError("frozen championization authority differs")

    task_fields = {
        "task_id",
        "source_type_ids",
        "paired_policy_by_type",
        "candidate_ids",
        "candidate_bundle_digests",
    }
    simple_task_fields = {"source_type_ids", "paired_policy_by_type", "candidate_ids"}
    context_to_task: dict[str, str] = {}
    context_to_policy: dict[str, str] = {}
    task_by_anchor: dict[str, str] = {}
    for task_id, raw_task in source_fit["tasks"].items():
        task = _exact_fields(raw_task, task_fields, f"source task {task_id}")
        simple = _exact_fields(
            layout["tasks"].get(task_id), simple_task_fields, f"layout task {task_id}"
        )
        source_ids = tuple(task["source_type_ids"])
        paired = dict(task["paired_policy_by_type"])
        candidates = tuple(task["candidate_ids"])
        if (
            task["task_id"] != task_id
            or len(source_ids) != 5
            or len(set(source_ids)) != 5
            or set(source_ids) != set(paired)
            or set(candidates) != set(paired.values())
            or dict(task["candidate_bundle_digests"])
            != {policy: entries[policy].bundle_digest for policy in candidates}
            or dict(simple)
            != {
                "source_type_ids": list(source_ids),
                "paired_policy_by_type": paired,
                "candidate_ids": list(candidates),
            }
        ):
            raise V05RunnerError(f"TASK_5 layout differs for {task_id}")
        for context_id, policy in paired.items():
            anchor = entries[policy].source_anchor_id
            if context_id in context_to_policy or anchor in task_by_anchor:
                raise V05RunnerError("source task layout contains duplicates")
            context_to_task[context_id] = str(task_id)
            context_to_policy[context_id] = policy
            task_by_anchor[anchor] = str(task_id)
    if set(layout["tasks"]) != set(source_fit["tasks"]):
        raise V05RunnerError("source task layouts cover different tasks")

    expected_npz_keys = tuple(r4_config["source_fit_bank_npz_keys"])
    if expected_npz_keys != (
        "observation",
        "action",
        "next_observation",
        "episode_offsets",
        "probe_membership_digest",
    ):
        raise V05RunnerError("r4 source bank key allowlist drifted")
    bank_dir = str(r4_config["source_fit_bank_directory"])
    physical_bank_root = root / bank_dir
    if (
        physical_bank_root.is_symlink()
        or not physical_bank_root.is_dir()
        or {item.name for item in physical_bank_root.iterdir()}
        != {Path(str(row["reward_free_npz"])).name for row in source_fit["contexts"]}
    ):
        raise V05RunnerError("source-fit bank directory closure differs")
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
    arrays_by_anchor: dict[str, Mapping[str, np.ndarray]] = {}
    memberships_by_anchor: dict[str, ProbeMembership] = {}
    context_by_anchor: dict[str, str] = {}
    asset_sha_by_anchor: dict[str, str] = {}
    membership_digest_by_anchor: dict[str, str] = {}
    observed_contexts: set[str] = set()
    for raw_row in source_fit["contexts"]:
        row = _exact_fields(raw_row, context_fields, "source-fit context")
        context_id = str(row["context_id"])
        relative = Path(str(row["reward_free_npz"]))
        if (
            context_id in observed_contexts
            or context_id not in context_to_policy
            or row["role"] != "source"
            or row["task_id"] != context_to_task[context_id]
            or row["episode_count"] != 32
            or row["visible_transitions_per_episode"] != 64
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.parent != Path(bank_dir)
        ):
            raise V05RunnerError("source-fit context protocol differs")
        observed_contexts.add(context_id)
        membership_value = membership_doc["contexts"].get(context_id)
        expected_membership = derive_probe_membership(context_id, split_seed)
        if (
            not isinstance(membership_value, Mapping)
            or dict(membership_value) != expected_membership.to_dict()
        ):
            raise V05RunnerError("probe membership does not replay from split seed")
        path = _owned_frozen_path(root, relative, "source bank")
        expected_sha = _digest(row["reward_free_npz_sha256"], "source bank SHA")
        bank_bytes = _frozen_bytes(path, expected_sha, "source bank")
        try:
            with np.load(io.BytesIO(bank_bytes), allow_pickle=False) as archive:
                if set(archive.files) != set(expected_npz_keys):
                    raise V05RunnerError("source bank NPZ fields differ")
                arrays = {
                    name: np.array(archive[name], copy=True) for name in archive.files
                }
        except V05RunnerError:
            raise
        except (OSError, ValueError, EOFError) as error:
            raise V05RunnerError("source bank NPZ cannot be parsed") from error
        membership_scalar = np.asarray(arrays.pop("probe_membership_digest"))
        observation = arrays["observation"]
        action = arrays["action"]
        next_observation = arrays["next_observation"]
        offsets = arrays["episode_offsets"]
        task_schema = raw_adapter["tasks"].get(row["task_id"])
        if not isinstance(task_schema, Mapping):
            raise V05RunnerError("Raw task ABI metadata is absent")
        _exact_fields(
            task_schema,
            {"observation_dim", "action_dim", "native_schema_digest"},
            "Raw task ABI",
        )
        if (
            membership_scalar.shape != ()
            or str(membership_scalar) != expected_membership.membership_digest
            or row["probe_membership_digest"] != expected_membership.membership_digest
            or observation.dtype != np.float64
            or action.dtype != np.float64
            or next_observation.dtype != np.float64
            or observation.shape != (2048, int(task_schema["observation_dim"]))
            or action.shape != (2048, int(task_schema["action_dim"]))
            or next_observation.shape != observation.shape
            or offsets.shape != (33,)
            or offsets.dtype.kind not in "iu"
            or not np.array_equal(offsets, np.arange(33, dtype=np.int64) * 64)
            or any(
                not np.all(np.isfinite(array))
                for array in (observation, action, next_observation)
            )
        ):
            raise V05RunnerError("source bank is not the frozen 32x64 float64 view")
        policy = context_to_policy[context_id]
        anchor = entries[policy].source_anchor_id
        arrays_by_anchor[anchor] = MappingProxyType(arrays)
        memberships_by_anchor[anchor] = expected_membership
        context_by_anchor[anchor] = context_id
        asset_sha_by_anchor[anchor] = expected_sha
        membership_digest_by_anchor[anchor] = expected_membership.membership_digest
    if observed_contexts != set(context_to_policy) or set(arrays_by_anchor) != set(
        anchor_to_policy
    ):
        raise V05RunnerError("r4 source bank coverage differs from the registry")

    census = documents["asset_census"]
    policy_audit = census.get("policy_bundle_and_abi")
    if (
        census.get("status") != "PASS"
        or census.get("source_context_count") != 30
        or census.get("policy_count") != 30
        or census.get("task_count") != 6
        or census.get("candidates_per_task") != 5
        or not isinstance(policy_audit, Mapping)
        or not isinstance(policy_audit.get("candidates"), Mapping)
        or set(policy_audit["candidates"]) != set(entries)
        or not isinstance(policy_audit.get("task_abi_digests"), Mapping)
    ):
        raise V05RunnerError("r4 asset census coverage differs")
    for policy, entry in entries.items():
        audit = policy_audit["candidates"][policy]
        task_id = task_by_anchor[entry.source_anchor_id]
        champion = championization.champions[entry.source_anchor_id]
        if (
            audit.get("bundle_manifest_digest") != entry.bundle_digest
            or audit.get("execution_abi_digest") != entry.execution_abi.digest
            or policy_audit["task_abi_digests"].get(task_id)
            != entry.execution_abi.digest
            or champion.candidate_id != entry.candidate_id
            or champion.bundle_digest != entry.bundle_digest
            or champion.champion_digest != entry.champion_digest
        ):
            raise V05RunnerError("registry/champion/census provenance differs")
    passed = sum(item.competence.passed for item in championization.champions.values())
    if passed != 23:
        raise V05RunnerError("champion competence pass/fail census differs")
    certificate_manifest = project_certificate_manifest(
        anchor_to_policy,
        task_by_anchor=task_by_anchor,
        policy_bundle_digest_by_policy={
            policy: entry.bundle_digest for policy, entry in entries.items()
        },
        championization_admission_digest_by_anchor={
            anchor: champion.champion_digest
            for anchor, champion in championization.champions.items()
        },
        execution_abi_digest_by_policy={
            policy: entry.execution_abi.digest for policy, entry in entries.items()
        },
        expected_anchor_ids=anchor_to_policy,
    )
    competence = tuple(item.competence for item in championization.champions.values())
    provenance = {
        "frozen_file_sha256": frozen_file_sha256,
        "policy_market_id": registry_doc["policy_market_id"],
        "r4_fixed_probe_protocol_id": r4_probe_protocol_digest,
        "championization_digest": championization.championization_digest,
        "certificate_scope": "FROZEN_V03_DEVELOPMENT_CHAMPIONIZATION_OBSERVE",
        "competence_mode": "OBSERVE",
        "selection_rule": SELECTION_RULE,
        "attestation_plan_digest": None,
        "formal_eligible": False,
        "passed_count": passed,
        "failed_count": 30 - passed,
        "competence_mean_range": [
            min(item.mean for item in competence),
            max(item.mean for item in competence),
        ],
        "competence_lcb_range": [
            min(item.lcb for item in competence),
            max(item.lcb for item in competence),
        ],
    }
    return FrozenR4Assets(
        config=MappingProxyType(dict(config)),
        config_digest=_digest(config_digest, "config_digest"),
        r4_root=root,
        v03_root=v03_root,
        arrays_by_anchor=MappingProxyType(arrays_by_anchor),
        membership_by_anchor=MappingProxyType(memberships_by_anchor),
        context_by_anchor=MappingProxyType(context_by_anchor),
        task_by_anchor=MappingProxyType(task_by_anchor),
        parent_asset_sha256=MappingProxyType(asset_sha_by_anchor),
        parent_membership_digest=MappingProxyType(membership_digest_by_anchor),
        native_schema_by_task=MappingProxyType(
            {
                task: _digest(value["native_schema_digest"], "native_schema_digest")
                for task, value in raw_adapter["tasks"].items()
            }
        ),
        certificate_manifest=certificate_manifest,
        probe_protocol_digest=probe_protocol_digest,
        provenance=MappingProxyType(provenance),
    )


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise V05RunnerError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise V05RunnerError(f"{where} must be a lowercase SHA-256 digest") from error
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _publish_or_match_json(
    path: Path, value: Mapping[str, Any], *, resume: bool
) -> str:
    if path.exists():
        if not resume or path.is_symlink() or not path.is_file():
            raise V05RunnerError(f"refusing to reuse artifact: {path}")
        existing = load_strict_json(path)
        if canonical_json_bytes(existing) != canonical_json_bytes(value):
            raise V05RunnerError(f"resume JSON digest changed: {path}")
        return sha256_file(path)
    return atomic_write_json(path, value)


def _publish_or_match_npz(
    path: Path, arrays: Mapping[str, np.ndarray], *, resume: bool
) -> str:
    expected = {name: np.asarray(value) for name, value in arrays.items()}
    if path.exists():
        if not resume or path.is_symlink() or not path.is_file():
            raise V05RunnerError(f"refusing to reuse artifact: {path}")
        observed = read_npz(path)
        if set(observed) != set(expected) or any(
            observed[name].dtype != expected[name].dtype
            or not np.array_equal(observed[name], expected[name])
            for name in expected
        ):
            raise V05RunnerError(f"resume NPZ content changed: {path}")
        return sha256_file(path)
    return atomic_write_npz(path, expected)


def _source_native_bank(
    assets: FrozenR4Assets,
    source_id: str,
    *,
    start_episode: int,
    stop_episode: int,
    data_role: str,
) -> NativeTransitionBank:
    arrays = assets.arrays_by_anchor[source_id]
    start = start_episode * 64
    stop = stop_episode * 64
    episode_count = stop_episode - start_episode
    rows = episode_count * 64
    context_id = assets.context_by_anchor[source_id]
    task_id = assets.task_by_anchor[source_id]
    membership = assets.membership_by_anchor[source_id]
    raw_digest = sha256_json(
        {
            "parent_asset_sha256": assets.parent_asset_sha256[source_id],
            "parent_membership_digest": assets.parent_membership_digest[source_id],
            "physical_episode_positions": list(range(start_episode, stop_episode)),
            "original_episode_ids": list(
                membership.episode_order[start_episode:stop_episode]
            ),
            "synthetic_reward": "SYNTHETIC_ZERO_NOT_READ",
            "synthetic_terminated": "ALL_FALSE",
            "synthetic_truncated": "EPISODE_BOUNDARY_ONLY",
        }
    )
    truncated = np.zeros(rows, dtype=np.bool_)
    truncated[63::64] = True
    return NativeTransitionBank(
        bank_id=f"v05-{context_id}-{start_episode}-{stop_episode}",
        task_private_id=task_id,
        data_role=data_role,
        native_schema_digest=assets.native_schema_by_task[task_id],
        raw_dataset_digest=raw_digest,
        observation=arrays["observation"][start:stop],
        action=arrays["action"][start:stop],
        reward=np.zeros(rows, dtype=np.float64),
        next_observation=arrays["next_observation"][start:stop],
        terminated=np.zeros(rows, dtype=np.bool_),
        truncated=truncated,
        episode_id=np.repeat(np.arange(episode_count, dtype=np.int64), 64),
        timestep=np.tile(np.arange(64, dtype=np.int64), episode_count),
    )


def _normalizer_arrays(normalizer: GlobalNormalizer) -> dict[str, np.ndarray]:
    return {
        "observation_mean": normalizer.observation_mean,
        "observation_std": normalizer.observation_std,
        "action_mean": normalizer.action_mean,
        "action_std": normalizer.action_std,
        "observation_task_count": normalizer.observation_task_count,
        "action_task_count": normalizer.action_task_count,
        "reward_mean": np.asarray(normalizer.reward_mean, dtype=np.float64),
        "reward_std": np.asarray(normalizer.reward_std, dtype=np.float64),
        "source_bank_digests": np.asarray(normalizer.source_bank_digests),
        "source_fit_roles": np.asarray(normalizer.source_fit_roles),
        "std_floor": np.asarray(normalizer.std_floor, dtype=np.float64),
    }


def _normalizer_from_arrays(
    arrays: Mapping[str, np.ndarray],
    registry: NativeShapeRegistry,
    expected_digest: str,
) -> GlobalNormalizer:
    expected = {
        "observation_mean",
        "observation_std",
        "action_mean",
        "action_std",
        "observation_task_count",
        "action_task_count",
        "reward_mean",
        "reward_std",
        "source_bank_digests",
        "source_fit_roles",
        "std_floor",
    }
    if set(arrays) != expected:
        raise V05RunnerError("persisted normalizer arrays differ")
    return GlobalNormalizer(
        registry_digest=str(registry.registry_digest),
        observation_mean=arrays["observation_mean"],
        observation_std=arrays["observation_std"],
        action_mean=arrays["action_mean"],
        action_std=arrays["action_std"],
        reward_mean=float(arrays["reward_mean"]),
        reward_std=float(arrays["reward_std"]),
        observation_task_count=arrays["observation_task_count"],
        action_task_count=arrays["action_task_count"],
        source_bank_digests=tuple(str(item) for item in arrays["source_bank_digests"]),
        source_fit_roles=tuple(str(item) for item in arrays["source_fit_roles"]),
        std_floor=float(arrays["std_floor"]),
        normalizer_digest=expected_digest,
    )


def _canonical_source_stage(
    assets: FrozenR4Assets, run_dir: Path, *, resume: bool
) -> tuple[dict[str, EpisodeBank], dict[str, Any]]:
    """Persist source-fit 19+6 banks; return full32 only to privileged memory."""

    root = run_dir / "source_fit" / "canonical"
    input_digest = sha256_json(
        {
            "config_digest": assets.config_digest,
            "probe_protocol_digest": assets.probe_protocol_digest,
            "parent_assets": dict(assets.parent_asset_sha256),
            "parent_memberships": dict(assets.parent_membership_digest),
            "source_roles": {"train": [0, 19], "validation": [19, 25]},
            "numeric_path": (
                "v03 normalize float64 -> canonical TransitionBank float32 -> "
                "V_DELTA_ONLY -> EpisodeBank float64"
            ),
        }
    )
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise V05RunnerError("canonical source root is unsafe")
    source_ids = tuple(sorted(assets.arrays_by_anchor))
    state_path = root / "normalizer_state.npz"
    manifest_path = root / "canonicalizer_manifest.json"
    complete_path = root / "complete.json"
    fit_paths = {
        source_id: root / "fit_banks" / f"{source_id}.npz" for source_id in source_ids
    }
    if any(
        path.is_symlink()
        for path in (
            root.parent,
            root,
            root / "fit_banks",
            state_path,
            manifest_path,
            complete_path,
            *fit_paths.values(),
        )
    ):
        raise V05RunnerError("canonical source state contains a symlink")
    if root.exists():
        allowed = {state_path, manifest_path, complete_path, *fit_paths.values()}
        observed = {
            path for path in root.rglob("*") if path.is_file() or path.is_symlink()
        }
        present_fit = {path for path in fit_paths.values() if path.exists()}
        if not observed.issubset(allowed) or (
            present_fit and present_fit != set(fit_paths.values())
        ):
            raise V05RunnerError("canonical source state is partial or contains extras")
    existing_complete: dict[str, Any] | None = None
    if complete_path.exists():
        if not resume or not complete_path.is_file():
            raise V05RunnerError("completed canonical source stage cannot be reused")
        existing_complete = load_strict_json(complete_path)
        unsigned_complete = {
            key: value
            for key, value in existing_complete.items()
            if key != "complete_digest"
        }
        expected_files = {state_path, manifest_path, complete_path, *fit_paths.values()}
        observed_files = {
            path for path in root.rglob("*") if path.is_file() or path.is_symlink()
        }
        files = existing_complete.get("files")
        if (
            set(existing_complete)
            != {
                "schema",
                "status",
                "input_digest",
                "canonicalizer_manifest_digest",
                "normalizer_digest",
                "canonicalizer_digest",
                "registry_digest",
                "source_count",
                "persisted_roles",
                "persisted_episode_count_per_anchor",
                "held_repeat_persisted",
                "files",
                "complete_digest",
            }
            or existing_complete.get("schema")
            != "policy-learnware.v05-canonical-source-complete.v1"
            or existing_complete.get("status") != "COMPLETE"
            or existing_complete.get("input_digest") != input_digest
            or existing_complete.get("source_count") != 30
            or existing_complete.get("persisted_roles")
            != ["source_train", "source_validation"]
            or existing_complete.get("persisted_episode_count_per_anchor") != 25
            or existing_complete.get("held_repeat_persisted") is not False
            or existing_complete.get("complete_digest")
            != sha256_json(unsigned_complete)
            or not isinstance(files, Mapping)
            or set(files) != set(source_ids)
            or observed_files != expected_files
        ):
            raise V05RunnerError("completed canonical source closure changed")
        for source_id, path in fit_paths.items():
            row = _exact_fields(
                files[source_id],
                {
                    "relative_path",
                    "file_sha256",
                    "fit_bank_digest",
                    "persisted_episode_count",
                },
                "canonical fit-bank receipt",
            )
            arrays = read_npz(path)
            if (
                set(arrays) != {"points", "episode_offsets"}
                or row["relative_path"] != path.relative_to(run_dir).as_posix()
                or row["file_sha256"] != sha256_file(path)
                or row["persisted_episode_count"] != 25
                or arrays["points"].shape != (25 * 64, 30)
                or arrays["episode_offsets"].shape != (26,)
                or not np.array_equal(
                    np.diff(arrays["episode_offsets"]), np.full(25, 64)
                )
                or EpisodeBank(arrays["points"], arrays["episode_offsets"]).bank_digest
                != row["fit_bank_digest"]
            ):
                raise V05RunnerError("completed canonical fit bank changed")
    train_native = {
        source_id: _source_native_bank(
            assets,
            source_id,
            start_episode=0,
            stop_episode=19,
            data_role="source_representation_train",
        )
        for source_id in sorted(assets.arrays_by_anchor)
    }
    validation_native = {
        source_id: _source_native_bank(
            assets,
            source_id,
            start_episode=19,
            stop_episode=25,
            data_role="source_representation_validation",
        )
        for source_id in sorted(assets.arrays_by_anchor)
    }
    registry = NativeShapeRegistry.from_source_banks(
        tuple(train_native.values()) + tuple(validation_native.values())
    )
    if registry.max_observation_dim + registry.max_action_dim != 30:
        raise V05RunnerError("r4 canonical V_DELTA_ONLY width must equal 30")
    if state_path.exists() and manifest_path.exists():
        if not resume:
            raise V05RunnerError("canonical source stage already exists")
        manifest = load_strict_json(manifest_path)
        _exact_fields(
            manifest,
            {
                "schema",
                "input_digest",
                "registry",
                "normalizer",
                "canonicalizer",
                "normalizer_state_sha256",
                "synthetic_channels",
                "numeric_path",
                "manifest_digest",
            },
            "canonicalizer manifest",
        )
        unsigned = {
            key: value for key, value in manifest.items() if key != "manifest_digest"
        }
        if (
            manifest["manifest_digest"] != sha256_json(unsigned)
            or manifest["input_digest"] != input_digest
            or manifest["registry"] != registry.to_dict()
            or manifest["normalizer_state_sha256"] != sha256_file(state_path)
        ):
            raise V05RunnerError("canonical source state changed during resume")
        normalizer = _normalizer_from_arrays(
            read_npz(state_path), registry, manifest["normalizer"]["normalizer_digest"]
        )
        canonicalizer = GlobalCanonicalizerSpec(registry, normalizer)
        if (
            normalizer.to_dict() != manifest["normalizer"]
            or canonicalizer.to_dict() != manifest["canonicalizer"]
        ):
            raise V05RunnerError("canonical source typed replay differs")
    else:
        if state_path.exists() != manifest_path.exists():
            raise V05RunnerError("canonical source state is partially published")
        normalizer = fit_global_normalizer(
            tuple(train_native.values()) + tuple(validation_native.values()),
            registry,
            std_floor=float(assets.config["measurement"]["normalizer_std_floor"]),
        )
        canonicalizer = GlobalCanonicalizerSpec(registry, normalizer)
        state_sha = _publish_or_match_npz(
            state_path, _normalizer_arrays(normalizer), resume=resume
        )
        unsigned = {
            "schema": "policy-learnware.v05-canonical-source-state.v1",
            "input_digest": input_digest,
            "registry": registry.to_dict(),
            "normalizer": normalizer.to_dict(),
            "canonicalizer": canonicalizer.to_dict(),
            "normalizer_state_sha256": state_sha,
            "synthetic_channels": {
                "reward": "SYNTHETIC_ZERO_NOT_READ",
                "terminated": "ALL_FALSE",
                "truncated": "EPISODE_BOUNDARY_ONLY",
            },
            "numeric_path": (
                "normalize float64 -> canonical TransitionBank float32 -> "
                "V_DELTA_ONLY -> float64 EpisodeBank"
            ),
        }
        manifest = {**unsigned, "manifest_digest": sha256_json(unsigned)}
        _publish_or_match_json(manifest_path, manifest, resume=resume)

    banks: dict[str, EpisodeBank] = {}
    files: dict[str, Any] = {}
    for source_id in source_ids:
        path = fit_paths[source_id]
        native = _source_native_bank(
            assets,
            source_id,
            start_episode=0,
            stop_episode=32,
            data_role="source_reference_spec",
        )
        receipt = canonicalizer.transform(native)
        transition_bank = TransitionBank.from_canonical_batch(receipt.batch)
        view = apply_transition_view(transition_bank, V_DELTA_ONLY)
        full_bank = EpisodeBank(
            np.asarray(view.feature_matrix, dtype=np.float64),
            transition_bank.episode_offsets,
        )
        fit_bank = full_bank.prefix(25)
        if path.exists():
            if not resume or path.is_symlink():
                raise V05RunnerError("canonical source bank cannot be reused")
            arrays = read_npz(path)
            if set(arrays) != {"points", "episode_offsets"}:
                raise V05RunnerError("canonical source bank arrays differ")
            persisted = EpisodeBank(arrays["points"], arrays["episode_offsets"])
            if persisted.bank_digest != fit_bank.bank_digest:
                raise V05RunnerError("persisted 19+6 canonical source bank changed")
        else:
            atomic_write_npz(
                path,
                {
                    "points": fit_bank.points,
                    "episode_offsets": fit_bank.episode_offsets,
                },
            )
        if (
            full_bank.episode_count != 32
            or full_bank.input_dim != 30
            or full_bank.points.shape != (2048, 30)
            or fit_bank.points.shape != (25 * 64, 30)
            or np.any(np.diff(full_bank.episode_offsets) != 64)
        ):
            raise V05RunnerError("canonical source bank shape differs")
        banks[source_id] = full_bank
        files[source_id] = {
            "relative_path": path.relative_to(run_dir).as_posix(),
            "file_sha256": sha256_file(path),
            "fit_bank_digest": fit_bank.bank_digest,
            "persisted_episode_count": 25,
        }
    expected_npz = {state_path} | {
        root / "fit_banks" / f"{source_id}.npz" for source_id in banks
    }
    if set(root.rglob("*.npz")) != expected_npz:
        raise V05RunnerError(
            "canonical source directory contains non-fit or held-repeat data"
        )
    complete_unsigned = {
        "schema": "policy-learnware.v05-canonical-source-complete.v1",
        "status": "COMPLETE",
        "input_digest": input_digest,
        "canonicalizer_manifest_digest": manifest["manifest_digest"],
        "normalizer_digest": normalizer.normalizer_digest,
        "canonicalizer_digest": canonicalizer.canonicalizer_digest,
        "registry_digest": registry.registry_digest,
        "source_count": len(banks),
        "persisted_roles": ["source_train", "source_validation"],
        "persisted_episode_count_per_anchor": 25,
        "held_repeat_persisted": False,
        "files": files,
    }
    complete = {
        **complete_unsigned,
        "complete_digest": sha256_json(complete_unsigned),
    }
    _publish_or_match_json(complete_path, complete, resume=resume)
    return banks, complete


def _scorer_source_fit_manifest(role_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project only train/validation provenance into the scorer-visible panel."""

    try:
        roles = {
            role: dict(role_manifest["roles"][role])
            for role in ("source_train", "source_validation")
        }
        sources = {
            source_id: {
                "roles": {
                    role: dict(row["roles"][role])
                    for role in ("source_train", "source_validation")
                }
            }
            for source_id, row in role_manifest["sources"].items()
        }
    except (AttributeError, KeyError, TypeError) as error:
        raise V05RunnerError("privileged source role manifest is malformed") from error
    unsigned = {
        "schema": SCORER_SOURCE_FIT_SCHEMA,
        "source_count": role_manifest.get("source_count"),
        "roles": roles,
        "sources": sources,
    }
    return {**unsigned, "source_fit_provenance_digest": sha256_json(unsigned)}


@dataclass(frozen=True)
class P0Panel:
    """The six fitted models plus their one shared source-evidence binding."""

    certificate_manifest: CertifiedPolicyManifest
    config_digest: str
    source_binding: Mapping[str, Any]
    source_role_manifest: Mapping[str, Any]
    bandwidth: float
    raw: RawDeltaRKMENN
    empirical_mmd: EmpiricalMMDNN
    summary_logreg: SummaryLogReg
    kme_krr: KMEKRR
    rff: RFFKMENN
    swe: SWENN

    def __post_init__(self) -> None:
        if not isinstance(self.certificate_manifest, CertifiedPolicyManifest):
            raise V05RunnerError("certificate_manifest has the wrong type")
        object.__setattr__(
            self, "config_digest", _digest(self.config_digest, "config_digest")
        )
        if not isinstance(self.source_binding, Mapping):
            raise V05RunnerError("source_binding must be a mapping")
        binding = dict(self.source_binding)
        digest_fields = (
            "probe_protocol_digest",
            "normalization_digest",
            "source_fit_provenance_digest",
            "source_train_membership_digest",
            "source_validation_membership_digest",
            "source_train_bank_digest",
            "source_validation_bank_digest",
        )
        if set(binding) != {*digest_fields, "episode_counts_per_anchor"}:
            raise V05RunnerError("scorer source binding fields differ")
        counts = binding["episode_counts_per_anchor"]
        if (
            not isinstance(counts, (list, tuple))
            or any(type(value) is not int for value in counts)
            or tuple(counts) != (19, 6)
        ):
            raise V05RunnerError("scorer source episode counts differ")
        binding["episode_counts_per_anchor"] = (19, 6)
        for field in digest_fields:
            binding[field] = _digest(binding.get(field), field)
        object.__setattr__(self, "source_binding", MappingProxyType(binding))
        if not isinstance(self.source_role_manifest, Mapping):
            raise V05RunnerError("source_role_manifest must be a mapping")
        role_manifest = dict(self.source_role_manifest)
        unsigned_role_manifest = {
            key: value
            for key, value in role_manifest.items()
            if key != "source_fit_provenance_digest"
        }
        if (
            set(role_manifest)
            != {
                "schema",
                "source_count",
                "roles",
                "sources",
                "source_fit_provenance_digest",
            }
            or role_manifest.get("schema") != SCORER_SOURCE_FIT_SCHEMA
            or set(role_manifest.get("roles", ()))
            != {"source_train", "source_validation"}
            or any("repeat" in str(key).lower() for key in role_manifest)
            or role_manifest.get("source_fit_provenance_digest")
            != sha256_json(unsigned_role_manifest)
        ):
            raise V05RunnerError("scorer source-fit manifest digest changed")
        try:
            expected_positions = {
                "source_train": list(range(0, 19)),
                "source_validation": list(range(19, 25)),
            }
            for role, positions in expected_positions.items():
                aggregate = role_manifest["roles"][role]
                if set(aggregate) != {
                    "episode_count",
                    "membership_digest",
                    "bank_digest",
                } or aggregate["episode_count"] != len(positions):
                    raise V05RunnerError("scorer source-fit aggregate fields differ")
                memberships = {}
                banks = {}
                for source_id, row in role_manifest["sources"].items():
                    role_row = row["roles"][role]
                    if set(role_row) != {
                        "episode_positions",
                        "membership_digest",
                        "bank_digest",
                    } or tuple(role_row["episode_positions"]) != tuple(positions):
                        raise V05RunnerError("scorer source-fit source fields differ")
                    memberships[source_id] = _digest(
                        role_row["membership_digest"], "role membership digest"
                    )
                    banks[source_id] = _digest(
                        role_row["bank_digest"], "role bank digest"
                    )
                if aggregate["membership_digest"] != sha256_json(
                    memberships
                ) or aggregate["bank_digest"] != sha256_json(banks):
                    raise V05RunnerError("scorer source-fit aggregate binding differs")
            manifest_binding = {
                "source_fit_provenance_digest": role_manifest[
                    "source_fit_provenance_digest"
                ],
                "source_train_membership_digest": role_manifest["roles"][
                    "source_train"
                ]["membership_digest"],
                "source_validation_membership_digest": role_manifest["roles"][
                    "source_validation"
                ]["membership_digest"],
                "source_train_bank_digest": role_manifest["roles"]["source_train"][
                    "bank_digest"
                ],
                "source_validation_bank_digest": role_manifest["roles"][
                    "source_validation"
                ]["bank_digest"],
            }
        except (KeyError, TypeError) as error:
            raise V05RunnerError("scorer source-fit manifest is malformed") from error
        if (
            any(binding[key] != value for key, value in manifest_binding.items())
            or role_manifest.get("source_count")
            != len(self.certificate_manifest.bindings)
            or set(role_manifest.get("sources", ())) != set(self.resolver.anchor_ids)
            or any(
                set(row) != {"roles"}
                or set(row["roles"]) != {"source_train", "source_validation"}
                for row in role_manifest.get("sources", {}).values()
            )
        ):
            raise V05RunnerError("scorer source-fit manifest binding differs")
        object.__setattr__(self, "source_role_manifest", _freeze_json(role_manifest))
        bandwidth = float(self.bandwidth)
        if not math.isfinite(bandwidth) or bandwidth <= 0.0:
            raise V05RunnerError("bandwidth must be finite and positive")
        object.__setattr__(self, "bandwidth", bandwidth)
        if (
            self.raw.bandwidth != bandwidth
            or self.empirical_mmd.bandwidth != bandwidth
            or self.kme_krr.bandwidth != bandwidth
            or self.rff.rff_map.bandwidth != bandwidth
        ):
            raise V05RunnerError("Raw, Empirical-MMD, KRR, and RFF bandwidths differ")
        normalizer = binding["normalization_digest"]
        if (
            self.rff.rff_map.normalization_digest != normalizer
            or self.swe.swe_map.normalization_digest != normalizer
        ):
            raise V05RunnerError("fixed-vector maps bind another normalizer")

    @property
    def resolver(self) -> CertificateResolver:
        return CertificateResolver(self.certificate_manifest)

    @property
    def source_model_manifest(self) -> dict[str, Any]:
        raw_arrays: dict[str, np.ndarray] = {}
        for source_id, source in self.raw.sources.items():
            raw_arrays.update(
                {
                    f"{source_id}.supports": source.supports,
                    f"{source_id}.beta": source.beta,
                    f"{source_id}.rkme_norm2": np.asarray(source.rkme_norm2),
                    f"{source_id}.empirical_norm2": np.asarray(source.empirical_norm2),
                    f"{source_id}.reduction_error": np.asarray(source.reduction_error),
                }
            )
        empirical_arrays: dict[str, np.ndarray] = {}
        for source_id, source in self.empirical_mmd.sources.items():
            empirical_arrays.update(
                {
                    f"{source_id}.points": source.points,
                    f"{source_id}.weights": source.weights,
                    f"{source_id}.episode_offsets": source.episode_offsets,
                    f"{source_id}.norm2": np.asarray(source.norm2),
                }
            )
        rff_arrays = dict(self.rff.prototypes)
        swe_arrays = dict(self.swe.prototypes)
        payload = {
            "p0_method_ids": list(P0_METHOD_IDS),
            "p1_status": P1_STATUS,
            "config_digest": self.config_digest,
            "certificate_manifest_digest": (
                self.certificate_manifest.certificate_manifest_digest
            ),
            "source_binding": dict(self.source_binding),
            "source_role_manifest": self.source_role_manifest,
            "bandwidth": self.bandwidth,
            "model_digests": {
                RAW_DELTA_RKME: sha256_ndarrays(raw_arrays),
                EMPIRICAL_MMD_NN: sha256_ndarrays(empirical_arrays),
                SUMMARY_LOGREG: self.summary_logreg.model_digest,
                KME_KRR: self.kme_krr.model_digest,
                RFF_KME_NN: sha256_json(
                    {
                        "map_digest": self.rff.rff_map.map_digest,
                        "prototypes_sha256": sha256_ndarrays(rff_arrays),
                    }
                ),
                SWE_NN: sha256_json(
                    {
                        "map_digest": self.swe.swe_map.map_digest,
                        "prototypes_sha256": sha256_ndarrays(swe_arrays),
                    }
                ),
            },
        }
        payload["source_model_manifest_digest"] = sha256_json(payload)
        return payload

    @property
    def method_cards(self) -> tuple[dict[str, Any], ...]:
        cards = []
        for card in p0_method_cards():
            supervised = card["method_id"] in {SUMMARY_LOGREG, KME_KRR}
            row = {
                **card,
                "source_binding": dict(self.source_binding),
                "config_digest": self.config_digest,
                "base_gaussian_bandwidth": self.bandwidth,
                "policy_resolution": (
                    "direct source-label class score, then endpoint policy mask"
                    if supervised
                    else "temperature-1 softmax over the complete anchor vector, "
                    "then additive many-to-one aggregation and endpoint mask"
                ),
            }
            if card["method_id"] == RFF_KME_NN:
                row["fixed_map"] = self.rff.rff_map.to_dict()
            elif card["method_id"] == SWE_NN:
                row["fixed_map"] = self.swe.swe_map.to_dict()
            elif card["method_id"] == SUMMARY_LOGREG:
                row["selected_l2"] = self.summary_logreg.selected_l2
            elif card["method_id"] == KME_KRR:
                row["selected_ridge"] = self.kme_krr.selected_ridge
            row["aggregation_digest"] = sha256_json(
                {
                    "method_id": row["method_id"],
                    "target_aggregation": row["target_aggregation"],
                    "score": row["score"],
                    "policy_resolution": row["policy_resolution"],
                }
            )
            cards.append(row)
        return tuple(cards)


def fit_p0_panel(
    verified_source_banks: Mapping[str, EpisodeBank],
    certificate_manifest: CertifiedPolicyManifest,
    *,
    config_digest: str,
    probe_protocol_digest: str,
    normalization_digest: str,
    source_parent_asset_sha256: Mapping[str, str],
    source_parent_membership_digest: Mapping[str, str],
    bandwidth: float,
    expected_source_count: int = 30,
    reducer_config: ReducerConfig = ReducerConfig(),
    rff_frequency_count: int = 512,
    rff_seed: int = 50_501,
    swe_direction_count: int = 64,
    swe_quantile_count: int = 64,
    swe_seed: int = 50_502,
    logreg_l2_grid: Sequence[float] = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0),
    logreg_max_iter: int = 3_000,
    logreg_tolerance: float = 1.0e-9,
    krr_ridge_grid: Sequence[float] = (1.0e-6, 1.0e-4, 1.0e-2, 1.0),
) -> P0Panel:
    """Project each verified 32-episode source bank, then fit every P0."""

    if (
        isinstance(expected_source_count, (bool, np.bool_))
        or not isinstance(expected_source_count, (int, np.integer))
        or int(expected_source_count) <= 0
    ):
        raise V05RunnerError("expected_source_count must be a positive integer")
    projection = project_verified_source_banks(
        verified_source_banks,
        parent_asset_sha256=source_parent_asset_sha256,
        parent_membership_digest=source_parent_membership_digest,
        expected_source_count=int(expected_source_count),
    )
    train = dict(projection.source_train)
    validation = dict(projection.source_validation)
    role_manifest = _scorer_source_fit_manifest(projection.manifest)
    resolver = CertificateResolver(certificate_manifest)
    if set(resolver.anchor_ids) != set(train):
        raise V05RunnerError("certificate/source anchor coverage differs")
    widths = {
        bank.input_dim for banks in (train, validation) for bank in banks.values()
    }
    if len(widths) != 1:
        raise V05RunnerError("source roles have different canonical point widths")
    labels = {
        binding.source_anchor_id: binding.opaque_certified_policy_id
        for binding in certificate_manifest.bindings
    }
    normalizer = _digest(normalization_digest, "normalization_digest")
    input_dim = widths.pop()
    rff_map = RFFMap(
        input_dim=input_dim,
        bandwidth=bandwidth,
        normalization_digest=normalizer,
        frequency_count=rff_frequency_count,
        public_seed=rff_seed,
    )
    swe_map = SWEMap(
        input_dim=input_dim,
        normalization_digest=normalizer,
        direction_count=swe_direction_count,
        quantile_count=swe_quantile_count,
        public_seed=swe_seed,
    )
    source_binding = {
        "probe_protocol_digest": _digest(
            probe_protocol_digest, "probe_protocol_digest"
        ),
        "normalization_digest": normalizer,
        "source_fit_provenance_digest": role_manifest["source_fit_provenance_digest"],
        "source_train_membership_digest": role_manifest["roles"]["source_train"][
            "membership_digest"
        ],
        "source_validation_membership_digest": role_manifest["roles"][
            "source_validation"
        ]["membership_digest"],
        "source_train_bank_digest": role_manifest["roles"]["source_train"][
            "bank_digest"
        ],
        "source_validation_bank_digest": role_manifest["roles"]["source_validation"][
            "bank_digest"
        ],
        "episode_counts_per_anchor": [19, 6],
    }
    return P0Panel(
        certificate_manifest=certificate_manifest,
        config_digest=config_digest,
        source_binding=source_binding,
        source_role_manifest=role_manifest,
        bandwidth=bandwidth,
        raw=RawDeltaRKMENN.fit(
            train,
            bandwidth=bandwidth,
            protocol_id=Q0_COMMON_GAUSSIAN_OPEN_LOOP,
            reducer_config=reducer_config,
        ),
        empirical_mmd=EmpiricalMMDNN.fit(
            train,
            bandwidth=bandwidth,
            protocol_id=Q0_COMMON_GAUSSIAN_OPEN_LOOP,
        ),
        summary_logreg=SummaryLogReg.fit(
            train,
            labels,
            validation,
            l2_grid=logreg_l2_grid,
            max_iter=logreg_max_iter,
            tolerance=logreg_tolerance,
        ),
        kme_krr=KMEKRR.fit(
            train,
            labels,
            validation,
            bandwidth=bandwidth,
            ridge_grid=krr_ridge_grid,
        ),
        rff=RFFKMENN.fit(train, rff_map=rff_map),
        swe=SWENN.fit(train, swe_map=swe_map),
    )


def _source_fit_stage(
    assets: FrozenR4Assets,
    full_banks: Mapping[str, EpisodeBank],
    canonical_receipt: Mapping[str, Any],
    run_dir: Path,
    *,
    resume: bool,
) -> tuple[P0Panel, Any]:
    """Recover or fit each concrete source model, publishing COMPLETE last."""

    root = run_dir / "source_fit" / "models"
    projection = project_verified_source_banks(
        full_banks,
        parent_asset_sha256=assets.parent_asset_sha256,
        parent_membership_digest=assets.parent_membership_digest,
        expected_source_count=30,
    )
    train = dict(projection.source_train)
    validation = dict(projection.source_validation)
    if any(bank.points.shape != (19 * 64, 30) for bank in train.values()) or any(
        bank.points.shape != (6 * 64, 30) for bank in validation.values()
    ):
        raise V05RunnerError("source train/validation shapes differ from 19/6x64x30")
    privileged_role_manifest = projection.manifest
    scorer_role_manifest = _scorer_source_fit_manifest(privileged_role_manifest)
    labels = {
        binding.source_anchor_id: binding.opaque_certified_policy_id
        for binding in assets.certificate_manifest.bindings
    }
    fit_input_digest = sha256_json(
        {
            "config_digest": assets.config_digest,
            "canonical_complete_digest": canonical_receipt["complete_digest"],
            "certificate_manifest_digest": (
                assets.certificate_manifest.certificate_manifest_digest
            ),
            "source_fit_provenance_digest": scorer_role_manifest[
                "source_fit_provenance_digest"
            ],
        }
    )
    source_ids = tuple(sorted(train))
    expected_model_paths = {
        **{
            f"empirical/{source_id}": root / "empirical" / f"{source_id}.npz"
            for source_id in source_ids
        },
        **{
            f"raw/{source_id}": root / "raw" / f"{source_id}.npz"
            for source_id in source_ids
        },
        SUMMARY_LOGREG: root / "summary_logreg.npz",
        KME_KRR: root / "kme_krr.npz",
        "rff_map": root / "rff_map.npz",
        "swe_map": root / "swe_map.npz",
        "fixed_vectors": root / "fixed_prototypes.npz",
    }
    if len(expected_model_paths) != 65:
        raise V05RunnerError("source checkpoint closure must contain 65 NPZ files")
    if any(
        path.is_symlink()
        for path in (
            root.parent,
            root,
            root / "empirical",
            root / "raw",
            *expected_model_paths.values(),
        )
    ):
        raise V05RunnerError("source checkpoint tree contains a symlink")
    progress_path = root / "progress.json"
    if progress_path.exists():
        if not resume or progress_path.is_symlink() or not progress_path.is_file():
            raise V05RunnerError("source model progress cannot be reused")
        progress = load_strict_json(progress_path)
        if (
            set(progress) != {"schema", "fit_input_digest", "models"}
            or progress.get("schema") != "policy-learnware.v05-source-fit-progress.v1"
            or progress.get("fit_input_digest") != fit_input_digest
            or not isinstance(progress.get("models"), Mapping)
        ):
            raise V05RunnerError("source model progress binding differs")
        model_records = dict(progress["models"])
    else:
        model_records = {}
    if not set(model_records).issubset(expected_model_paths):
        raise V05RunnerError("source model progress contains an unknown checkpoint")
    for key, path in expected_model_paths.items():
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise V05RunnerError(f"source checkpoint path is unsafe: {key}")
        if path.exists() != (key in model_records):
            raise V05RunnerError(f"source checkpoint/progress XOR mismatch: {key}")

    bandwidth_path = root / "bandwidth.json"
    manifest_path = root / "source_model_manifest.json"
    complete_path = root / "complete.json"
    existing_complete: dict[str, Any] | None = None
    if complete_path.exists():
        if not resume or complete_path.is_symlink() or not complete_path.is_file():
            raise V05RunnerError("completed source stage cannot be reused")
        existing_complete = load_strict_json(complete_path)
        unsigned_complete = {
            key: value
            for key, value in existing_complete.items()
            if key != "complete_digest"
        }
        expected_relative_paths = {
            path.relative_to(run_dir).as_posix()
            for path in expected_model_paths.values()
        }
        auxiliary = (bandwidth_path, progress_path, manifest_path)
        if (
            set(existing_complete)
            != {
                "schema",
                "status",
                "fit_input_digest",
                "source_model_manifest_digest",
                "source_model_manifest_sha256",
                "bandwidth_receipt_digest",
                "progress_sha256",
                "model_files",
                "complete_digest",
            }
            or existing_complete.get("schema")
            != "policy-learnware.v05-source-model-complete.v1"
            or existing_complete.get("status") != "COMPLETE"
            or existing_complete.get("fit_input_digest") != fit_input_digest
            or existing_complete.get("complete_digest")
            != sha256_json(unsigned_complete)
            or set(model_records) != set(expected_model_paths)
            or any(path.is_symlink() or not path.is_file() for path in auxiliary)
            or set(root.rglob("*.npz")) != set(expected_model_paths.values())
            or not isinstance(existing_complete.get("model_files"), Mapping)
            or set(existing_complete["model_files"]) != expected_relative_paths
            or any(
                existing_complete["model_files"][path.relative_to(run_dir).as_posix()]
                != sha256_file(path)
                for path in expected_model_paths.values()
            )
            or existing_complete.get("progress_sha256") != sha256_file(progress_path)
            or existing_complete.get("source_model_manifest_sha256")
            != sha256_file(manifest_path)
        ):
            raise V05RunnerError("completed source stage closure changed")

    bandwidth_config = assets.config["measurement"]["gaussian_bandwidth"]
    if bandwidth_path.exists():
        if not resume:
            raise V05RunnerError("source bandwidth stage already exists")
        bandwidth_row = load_strict_json(bandwidth_path)
        unsigned = {
            key: value
            for key, value in bandwidth_row.items()
            if key != "receipt_digest"
        }
        if (
            set(bandwidth_row)
            != {
                "schema",
                "fit_input_digest",
                "rule",
                "calibration_pairs",
                "public_seed",
                "bandwidth",
                "receipt_digest",
            }
            or bandwidth_row["receipt_digest"] != sha256_json(unsigned)
            or bandwidth_row["fit_input_digest"] != fit_input_digest
            or bandwidth_row["rule"] != bandwidth_config["rule"]
            or bandwidth_row["calibration_pairs"]
            != bandwidth_config["calibration_pairs"]
            or bandwidth_row["public_seed"] != bandwidth_config["public_seed"]
        ):
            raise V05RunnerError("persisted bandwidth provenance differs")
        bandwidth = float(bandwidth_row["bandwidth"])
    else:
        bandwidth = calibrate_bandwidth(
            train,
            calibration_pairs=int(bandwidth_config["calibration_pairs"]),
            seed=int(bandwidth_config["public_seed"]),
        )
        unsigned = {
            "schema": "policy-learnware.v05-source-bandwidth.v1",
            "fit_input_digest": fit_input_digest,
            "rule": bandwidth_config["rule"],
            "calibration_pairs": int(bandwidth_config["calibration_pairs"]),
            "public_seed": int(bandwidth_config["public_seed"]),
            "bandwidth": bandwidth,
        }
        bandwidth_row = {**unsigned, "receipt_digest": sha256_json(unsigned)}
        atomic_write_json(bandwidth_path, bandwidth_row)
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise V05RunnerError("source bandwidth is invalid")

    def record_model(
        key: str, path: Path, started: float, extra: Mapping[str, Any]
    ) -> None:
        row = {
            "relative_path": path.relative_to(run_dir).as_posix(),
            "file_sha256": sha256_file(path),
            "fit_input_digest": fit_input_digest,
            "bandwidth": bandwidth,
            "elapsed_seconds": max(0.0, time.monotonic() - started),
            "peak_rss_bytes": _rss_bytes(),
            "finite": True,
            **dict(extra),
        }
        previous = model_records.get(key)
        if previous is not None:
            stable_fields = set(row) - {"elapsed_seconds", "peak_rss_bytes"}
            if set(previous) != set(row) or any(
                previous.get(name) != row[name] for name in stable_fields
            ):
                raise V05RunnerError(f"source model progress changed: {key}")
            return
        if existing_complete is not None:
            raise V05RunnerError("completed source stage attempted to add a checkpoint")
        model_records[key] = row
        atomic_write_json(
            progress_path,
            {
                "schema": "policy-learnware.v05-source-fit-progress.v1",
                "fit_input_digest": fit_input_digest,
                "models": model_records,
            },
            overwrite=progress_path.exists(),
        )

    kernel = GaussianKernel(bandwidth)
    reducer_config = ReducerConfig(**dict(assets.config["raw_delta_rkme"]))
    train_bank_digest = scorer_role_manifest["roles"]["source_train"]["bank_digest"]
    validation_bank_digest = scorer_role_manifest["roles"]["source_validation"][
        "bank_digest"
    ]
    reducer_config_digest = sha256_json(dict(assets.config["raw_delta_rkme"]))
    summary_config_digest = sha256_json(
        {
            "train_bank_digest": train_bank_digest,
            "validation_bank_digest": validation_bank_digest,
            "config": dict(assets.config["summary_logreg"]),
        }
    )
    krr_config_digest = sha256_json(
        {
            "train_bank_digest": train_bank_digest,
            "validation_bank_digest": validation_bank_digest,
            "bandwidth": bandwidth,
            "config": dict(assets.config["kme_krr"]),
        }
    )
    empirical_sources: dict[str, EmpiricalKME] = {}
    raw_sources: dict[str, ReducedRKME] = {}
    for source_id in source_ids:
        empirical_path = expected_model_paths[f"empirical/{source_id}"]
        started = time.monotonic()
        if empirical_path.exists():
            if not resume:
                raise V05RunnerError("empirical checkpoint already exists")
            empirical = EmpiricalKME.load_npz(empirical_path)
        else:
            empirical = build_empirical_kme(
                train[source_id].points,
                kernel,
                episode_offsets=train[source_id].episode_offsets,
                protocol_id=Q0_COMMON_GAUSSIAN_OPEN_LOOP,
                dataset_digest=train[source_id].bank_digest,
                source_task=source_id,
            )
            empirical.save_npz(empirical_path)
        if (
            empirical.dataset_digest != train[source_id].bank_digest
            or empirical.protocol_id != Q0_COMMON_GAUSSIAN_OPEN_LOOP
            or empirical.source_task != source_id
            or empirical.bandwidth != bandwidth
            or empirical.points.shape != (19 * 64, 30)
        ):
            raise V05RunnerError("empirical source checkpoint binding differs")
        record_model(
            f"empirical/{source_id}",
            empirical_path,
            started,
            {
                "input_bank_digest": train[source_id].bank_digest,
                "norm2": empirical.norm2,
            },
        )
        empirical_sources[source_id] = empirical

        raw_path = expected_model_paths[f"raw/{source_id}"]
        started = time.monotonic()
        if raw_path.exists():
            if not resume:
                raise V05RunnerError("Raw checkpoint already exists")
            reduced = ReducedRKME.load_npz(raw_path)
        else:
            reduced = reduce_kme(empirical, reducer_config)
            reduced.save_npz(raw_path)
        if (
            reduced.source_dataset_digest != train[source_id].bank_digest
            or reduced.protocol_id != Q0_COMMON_GAUSSIAN_OPEN_LOOP
            or reduced.source_task != source_id
            or reduced.bandwidth != bandwidth
            or reduced.supports.shape[1] != 30
        ):
            raise V05RunnerError("Raw source checkpoint binding differs")
        record_model(
            f"raw/{source_id}",
            raw_path,
            started,
            {
                "input_bank_digest": train[source_id].bank_digest,
                "empirical_file_sha256": sha256_file(empirical_path),
                "reducer_config_digest": reducer_config_digest,
                "reduction_error": reduced.reduction_error,
            },
        )
        raw_sources[source_id] = reduced

    logreg_path = expected_model_paths[SUMMARY_LOGREG]
    started = time.monotonic()
    if logreg_path.exists():
        if not resume:
            raise V05RunnerError("Summary checkpoint already exists")
        logreg = SummaryLogReg.load_npz(logreg_path)
    else:
        cfg = assets.config["summary_logreg"]
        logreg = SummaryLogReg.fit(
            train,
            labels,
            validation,
            l2_grid=cfg["l2_grid"],
            max_iter=int(cfg["max_iter"]),
            tolerance=float(cfg["gradient_tolerance"]),
        )
        logreg.save_npz(logreg_path)
    record_model(
        SUMMARY_LOGREG,
        logreg_path,
        started,
        {
            "train_bank_digest": train_bank_digest,
            "validation_bank_digest": validation_bank_digest,
            "method_config_digest": summary_config_digest,
            "model_digest": logreg.model_digest,
            "selected_l2": logreg.selected_l2,
        },
    )
    krr_path = expected_model_paths[KME_KRR]
    started = time.monotonic()
    if krr_path.exists():
        if not resume:
            raise V05RunnerError("KRR checkpoint already exists")
        krr = KMEKRR.load_npz(krr_path)
    else:
        krr = KMEKRR.fit(
            train,
            labels,
            validation,
            bandwidth=bandwidth,
            ridge_grid=assets.config["kme_krr"]["ridge_grid"],
        )
        krr.save_npz(krr_path)
    if krr.training_bank.points.shape != (30 * 19 * 64, 30):
        raise V05RunnerError("KRR training matrix differs from 570 episodes")
    record_model(
        KME_KRR,
        krr_path,
        started,
        {
            "train_bank_digest": train_bank_digest,
            "validation_bank_digest": validation_bank_digest,
            "method_config_digest": krr_config_digest,
            "model_digest": krr.model_digest,
            "selected_ridge": krr.selected_ridge,
        },
    )

    rff_cfg = assets.config["rff_kme_nn"]
    swe_cfg = assets.config["swe_nn"]
    rff_map_path = expected_model_paths["rff_map"]
    swe_map_path = expected_model_paths["swe_map"]
    expected_rff_map = RFFMap(
        30,
        bandwidth,
        str(canonical_receipt["normalizer_digest"]),
        frequency_count=int(rff_cfg["frequency_count"]),
        public_seed=int(rff_cfg["public_seed"]),
    )
    expected_swe_map = SWEMap(
        30,
        str(canonical_receipt["normalizer_digest"]),
        direction_count=int(swe_cfg["direction_count"]),
        quantile_count=int(swe_cfg["quantile_count"]),
        public_seed=int(swe_cfg["public_seed"]),
    )
    started = time.monotonic()
    rff_map = (
        RFFMap.load_npz(rff_map_path) if rff_map_path.exists() else expected_rff_map
    )
    if rff_map.to_dict() != expected_rff_map.to_dict():
        raise V05RunnerError("RFF map checkpoint config differs")
    if not rff_map_path.exists():
        rff_map.save_npz(rff_map_path)
    record_model(
        "rff_map",
        rff_map_path,
        started,
        {
            "map_digest": rff_map.map_digest,
            "method_config_digest": sha256_json(dict(rff_cfg)),
            "normalizer_digest": canonical_receipt["normalizer_digest"],
        },
    )
    started = time.monotonic()
    swe_map = (
        SWEMap.load_npz(swe_map_path) if swe_map_path.exists() else expected_swe_map
    )
    if swe_map.to_dict() != expected_swe_map.to_dict():
        raise V05RunnerError("SWE map checkpoint config differs")
    if not swe_map_path.exists():
        swe_map.save_npz(swe_map_path)
    record_model(
        "swe_map",
        swe_map_path,
        started,
        {
            "map_digest": swe_map.map_digest,
            "method_config_digest": sha256_json(dict(swe_cfg)),
            "normalizer_digest": canonical_receipt["normalizer_digest"],
        },
    )
    prototypes_path = expected_model_paths["fixed_vectors"]
    started = time.monotonic()
    if prototypes_path.exists():
        if not resume:
            raise V05RunnerError("fixed prototype checkpoint already exists")
        arrays = read_npz(prototypes_path)
        if set(arrays) != {"source_ids", "rff", "swe"}:
            raise V05RunnerError("fixed prototype arrays differ")
        source_ids = tuple(str(item) for item in arrays["source_ids"])
        rff_prototypes = dict(zip(source_ids, arrays["rff"], strict=True))
        swe_prototypes = dict(zip(source_ids, arrays["swe"], strict=True))
    else:
        rff_model = RFFKMENN.fit(train, rff_map=rff_map)
        swe_model = SWENN.fit(train, swe_map=swe_map)
        source_ids = tuple(sorted(train))
        rff_prototypes = dict(rff_model.prototypes)
        swe_prototypes = dict(swe_model.prototypes)
        atomic_write_npz(
            prototypes_path,
            {
                "source_ids": np.asarray(source_ids),
                "rff": np.stack([rff_prototypes[item] for item in source_ids]),
                "swe": np.stack([swe_prototypes[item] for item in source_ids]),
            },
        )
    rff_matrix = np.stack([rff_prototypes[item] for item in source_ids])
    swe_matrix = np.stack([swe_prototypes[item] for item in source_ids])
    if (
        source_ids != tuple(sorted(train))
        or rff_matrix.shape != (30, int(rff_cfg["output_dimension"]))
        or swe_matrix.shape != (30, int(swe_cfg["output_dimension"]))
        or not np.all(np.isfinite(rff_matrix))
        or not np.all(np.isfinite(swe_matrix))
    ):
        raise V05RunnerError("fixed prototype checkpoint coverage differs")
    rff_model = RFFKMENN(rff_map, rff_prototypes)
    swe_model = SWENN(swe_map, swe_prototypes)
    record_model(
        "fixed_vectors",
        prototypes_path,
        started,
        {
            "source_train_bank_digest": train_bank_digest,
            "source_ids_digest": sha256_json(list(source_ids)),
            "rff_map_digest": rff_map.map_digest,
            "swe_map_digest": swe_map.map_digest,
            "rff_prototypes_digest": sha256_ndarrays(rff_prototypes),
            "swe_prototypes_digest": sha256_ndarrays(swe_prototypes),
        },
    )

    source_binding = {
        "probe_protocol_digest": assets.probe_protocol_digest,
        "normalization_digest": canonical_receipt["normalizer_digest"],
        "source_fit_provenance_digest": scorer_role_manifest[
            "source_fit_provenance_digest"
        ],
        "source_train_membership_digest": scorer_role_manifest["roles"]["source_train"][
            "membership_digest"
        ],
        "source_validation_membership_digest": scorer_role_manifest["roles"][
            "source_validation"
        ]["membership_digest"],
        "source_train_bank_digest": scorer_role_manifest["roles"]["source_train"][
            "bank_digest"
        ],
        "source_validation_bank_digest": scorer_role_manifest["roles"][
            "source_validation"
        ]["bank_digest"],
        "episode_counts_per_anchor": [19, 6],
    }
    panel = P0Panel(
        assets.certificate_manifest,
        assets.config_digest,
        source_binding,
        scorer_role_manifest,
        bandwidth,
        RawDeltaRKMENN(raw_sources, bandwidth, Q0_COMMON_GAUSSIAN_OPEN_LOOP),
        EmpiricalMMDNN(empirical_sources, bandwidth, Q0_COMMON_GAUSSIAN_OPEN_LOOP),
        logreg,
        krr,
        rff_model,
        swe_model,
    )
    model_manifest = panel.source_model_manifest
    manifest_sha = _publish_or_match_json(manifest_path, model_manifest, resume=resume)
    observed_npz = set(root.rglob("*.npz"))
    if set(model_records) != set(expected_model_paths) or observed_npz != set(
        expected_model_paths.values()
    ):
        raise V05RunnerError("source model checkpoint closure is not exactly 65 files")
    files = {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in sorted(expected_model_paths.values())
    }
    complete_unsigned = {
        "schema": "policy-learnware.v05-source-model-complete.v1",
        "status": "COMPLETE",
        "fit_input_digest": fit_input_digest,
        "source_model_manifest_digest": model_manifest["source_model_manifest_digest"],
        "source_model_manifest_sha256": manifest_sha,
        "bandwidth_receipt_digest": bandwidth_row["receipt_digest"],
        "progress_sha256": sha256_file(progress_path),
        "model_files": files,
    }
    complete = {
        **complete_unsigned,
        "complete_digest": sha256_json(complete_unsigned),
    }
    _publish_or_match_json(complete_path, complete, resume=resume)
    return panel, projection


def _rank(scores: Mapping[str, float], config_digest: str) -> tuple[str, ...]:
    if not scores or any(not math.isfinite(float(value)) for value in scores.values()):
        raise V05RunnerError("score vectors must be non-empty and finite")
    return tuple(
        sorted(
            scores,
            key=lambda item: (
                -float(scores[item]),
                tie_break_key(config_digest, item),
                item,
            ),
        )
    )


def _posterior(scores: Mapping[str, float]) -> dict[str, float]:
    identifiers = tuple(sorted(scores))
    values = np.asarray([scores[item] for item in identifiers], dtype=np.float64)
    values = np.exp(values - np.max(values))
    values /= np.sum(values)
    return dict(zip(identifiers, values.tolist(), strict=True))


def score_query(
    panel: P0Panel,
    query_views: AuthorizedQueryViews,
    *,
    budgets: Sequence[int] | None = None,
    expected_task_candidate_count: int = 5,
    timings: MutableMapping[str, float] | None = None,
) -> tuple[tuple[PredictionRanking, ...], tuple[dict[str, Any], ...]]:
    """Score only verified public views, then apply masks and break ties."""

    if not isinstance(panel, P0Panel) or not isinstance(
        query_views, AuthorizedQueryViews
    ):
        raise V05RunnerError("panel/query types are invalid")
    if timings is not None:
        if not isinstance(timings, MutableMapping) or not set(timings).issubset(
            P0_METHOD_IDS
        ):
            raise V05RunnerError("timings must be a mutable P0-method mapping")
        for method_id in P0_METHOD_IDS:
            timings.setdefault(method_id, 0.0)
    opaque_query_id = query_views.opaque_query_id
    if (
        not isinstance(opaque_query_id, str)
        or _OPAQUE_QUERY_ID.fullmatch(opaque_query_id) is None
    ):
        raise V05RunnerError("opaque_query_id must match q-[0-9a-f]{20,64}")
    manifest = query_views.manifest
    target_binding = {
        "probe_protocol_digest": _digest(
            manifest.get("probe_protocol_digest"), "probe_protocol_digest"
        ),
        "reward_free_bank_sha256": _digest(
            manifest.get("reward_free_bank_sha256"), "reward_free_bank_sha256"
        ),
        "target_membership_digest": _digest(
            manifest.get("target_membership_digest"), "target_membership_digest"
        ),
        "normalization_digest": _digest(
            manifest.get("normalization_digest"), "normalization_digest"
        ),
        "authorized_query_manifest_digest": _digest(
            manifest.get("manifest_digest"), "authorized_query_manifest_digest"
        ),
    }
    if (
        target_binding["probe_protocol_digest"]
        != panel.source_binding["probe_protocol_digest"]
        or target_binding["normalization_digest"]
        != panel.source_binding["normalization_digest"]
        or manifest.get("certificate_manifest_digest")
        != panel.certificate_manifest.certificate_manifest_digest
    ):
        raise V05RunnerError(
            "source/target common-probe, normalization, or certificate differs"
        )
    source_ids = tuple(panel.resolver.anchor_ids)
    if manifest.get("market_order_digest") != sha256_json(list(source_ids)):
        raise V05RunnerError("authorized candidate mask uses another market order")
    raw_mask = manifest.get("candidate_mask")
    if (
        not isinstance(raw_mask, list)
        or len(raw_mask) != len(source_ids)
        or any(type(item) is not bool for item in raw_mask)
    ):
        raise V05RunnerError("authorized candidate mask is malformed")
    candidates = tuple(
        source_id
        for source_id, allowed in zip(source_ids, raw_mask, strict=True)
        if allowed
    )
    if (
        isinstance(expected_task_candidate_count, (bool, np.bool_))
        or not isinstance(expected_task_candidate_count, (int, np.integer))
        or int(expected_task_candidate_count) <= 0
        or len(candidates) != int(expected_task_candidate_count)
    ):
        raise V05RunnerError("TASK candidate mask has the wrong size or coverage")
    candidate_records = [panel.resolver.record_for_anchor(item) for item in candidates]
    task_abi = {(item.task_id, item.execution_abi_digest) for item in candidate_records}
    if len(task_abi) != 1:
        raise V05RunnerError("TASK candidates must share one task and execution ABI")
    candidate_task, candidate_abi = next(iter(task_abi))
    complete_candidate_group = {
        item.source_anchor_id
        for item in panel.certificate_manifest.bindings
        if item.task_id == candidate_task and item.execution_abi_digest == candidate_abi
    }
    if set(candidates) != complete_candidate_group:
        raise V05RunnerError("TASK candidate mask is not the complete task/ABI group")
    if manifest.get("execution_abi_digest") != candidate_abi or manifest.get(
        "task_scope_digest"
    ) != sha256_json(
        {"task_id": candidate_task, "execution_abi_digest": candidate_abi}
    ):
        raise V05RunnerError("authorized task/ABI binding differs")
    labels = {
        item.source_anchor_id: item.opaque_certified_policy_id
        for item in panel.certificate_manifest.bindings
    }
    predictions: list[PredictionRanking] = []
    score_rows: list[dict[str, Any]] = []
    source_model_manifest_digest = panel.source_model_manifest[
        "source_model_manifest_digest"
    ]
    selected_budgets = (
        query_views.authorized_budgets if budgets is None else tuple(budgets)
    )
    if selected_budgets != query_views.authorized_budgets:
        raise V05RunnerError("runner budgets differ from the authorized query subset")
    for raw_budget in selected_budgets:
        if isinstance(raw_budget, (bool, np.bool_)) or not isinstance(
            raw_budget, (int, np.integer)
        ):
            raise V05RunnerError("budgets must contain only discrete integers")
        ledger = BudgetLedger.for_budget(int(raw_budget))
        raw_query = query_views.bank_for(RAW_DELTA_RKME, ledger.budget_episodes)
        empirical_query = query_views.bank_for(EMPIRICAL_MMD_NN, ledger.budget_episodes)
        krr_query = query_views.bank_for(KME_KRR, ledger.budget_episodes)
        canonical_budget_bank_digest = query_views.canonical_bank_digest_for_budget(
            ledger.budget_episodes
        )
        if {
            raw_query.bank_digest,
            empirical_query.bank_digest,
            krr_query.bank_digest,
        } != {canonical_budget_bank_digest}:
            raise V05RunnerError(
                "authorized method views differ from the canonical budget bank"
            )
        scorers = {
            RAW_DELTA_RKME: lambda: panel.raw.score(raw_query),
            EMPIRICAL_MMD_NN: lambda: panel.empirical_mmd.score(empirical_query),
            SUMMARY_LOGREG: lambda: panel.summary_logreg.score_summaries(
                query_views.summaries_for(ledger.budget_episodes)
            ),
            KME_KRR: lambda: panel.kme_krr.score(krr_query),
            RFF_KME_NN: lambda: panel.rff.score_specification(
                query_views.rff_specs[ledger.budget_episodes]
            ),
            SWE_NN: lambda: panel.swe.score_specification(
                query_views.swe_specs[ledger.budget_episodes]
            ),
        }
        full_scores = {}
        for method_id, scorer in scorers.items():
            started = time.monotonic()
            full_scores[method_id] = scorer()
            if timings is not None:
                timings[method_id] += max(0.0, time.monotonic() - started)
        for method_id in P0_METHOD_IDS:
            method_scores = full_scores[method_id]
            supervised = method_id in {SUMMARY_LOGREG, KME_KRR}
            anchor_scores = (
                {anchor: method_scores[policy] for anchor, policy in labels.items()}
                if supervised
                else method_scores
            )
            if set(anchor_scores) != set(source_ids):
                raise V05RunnerError(
                    "one method does not cover the common source market"
                )
            score_vector_digest = sha256_json(
                {
                    "method_id": method_id,
                    "budget_episodes": ledger.budget_episodes,
                    "opaque_query_id": opaque_query_id,
                    "source_order": list(source_ids),
                    "scores_before_mask": [
                        float(anchor_scores[source_id]) for source_id in source_ids
                    ],
                }
            )
            budget_ledger_digest = sha256_json(ledger.to_dict())
            for endpoint, mask in (
                (MARKET_30_CERT, source_ids),
                (TASK_5_CERT, candidates),
            ):
                masked_anchor_scores = {
                    anchor: anchor_scores[anchor] for anchor in mask
                }
                if supervised:
                    allowed_policies = {labels[anchor] for anchor in mask}
                    policy_scores = {
                        policy: method_scores[policy]
                        for policy in method_scores
                        if policy in allowed_policies
                    }
                else:
                    policy_scores = panel.resolver.aggregate_anchor_scores(
                        _posterior(anchor_scores), candidate_anchor_ids=mask
                    )
                ranked_anchors = _rank(masked_anchor_scores, panel.config_digest)
                ranked_policies = _rank(policy_scores, panel.config_digest)
                predictions.append(
                    PredictionRanking(
                        method_id=method_id,
                        endpoint=endpoint,
                        budget_episodes=ledger.budget_episodes,
                        opaque_query_id=opaque_query_id,
                        ranked_anchor_ids=ranked_anchors,
                        ranked_policy_ids=ranked_policies,
                        probe_protocol_digest=target_binding["probe_protocol_digest"],
                        reward_free_bank_sha256=target_binding[
                            "reward_free_bank_sha256"
                        ],
                        canonical_query_bank_digest=canonical_budget_bank_digest,
                        source_train_membership_digest=panel.source_binding[
                            "source_train_membership_digest"
                        ],
                        source_validation_membership_digest=panel.source_binding[
                            "source_validation_membership_digest"
                        ],
                        target_membership_digest=target_binding[
                            "target_membership_digest"
                        ],
                        normalization_digest=target_binding["normalization_digest"],
                        config_digest=panel.config_digest,
                        source_model_manifest_digest=source_model_manifest_digest,
                        authorized_query_manifest_digest=target_binding[
                            "authorized_query_manifest_digest"
                        ],
                        score_vector_digest=score_vector_digest,
                        budget_ledger_digest=budget_ledger_digest,
                    )
                )
                score_rows.append(
                    {
                        "method_id": method_id,
                        "endpoint": endpoint,
                        "budget_episodes": ledger.budget_episodes,
                        "opaque_query_id": opaque_query_id,
                        "scores_before_mask": [
                            float(anchor_scores[source_id]) for source_id in source_ids
                        ],
                        "source_binding": dict(panel.source_binding),
                        "target_binding": {
                            **target_binding,
                            "canonical_query_bank_digest": (
                                canonical_budget_bank_digest
                            ),
                        },
                        "ledger": ledger.to_dict(),
                        "score_vector_digest": score_vector_digest,
                        "budget_ledger_digest": budget_ledger_digest,
                    }
                )
    if timings is not None and any(
        not math.isfinite(float(timings[method_id])) or float(timings[method_id]) < 0.0
        for method_id in P0_METHOD_IDS
    ):
        raise V05RunnerError(
            "per-method scoring timings must be finite and nonnegative"
        )
    return tuple(predictions), tuple(score_rows)


def seal_prediction_rows(
    rankings: Iterable[PredictionRanking],
    *,
    expected_budgets: Sequence[int] | None = None,
) -> tuple[dict[str, Any], RankingSeal]:
    """The only runner exit toward evaluation: canonical truth-free bytes + seal."""

    if expected_budgets is None:
        raise V05RunnerError("seal requires the explicit authorized budget subset")
    rows = require_prediction_cell_coverage(
        rankings,
        expected_method_ids=P0_METHOD_IDS,
        expected_budgets=expected_budgets,
    )
    payload = prediction_payload(rows)
    return payload, seal_rankings(payload)


def _read_query_index(panel: P0Panel, path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise V05RunnerError("precommitted public query index is absent or unsafe")
    value = load_strict_json(path)
    fields = {
        "schema",
        "status",
        "query_count",
        "authorized_artifact_count",
        "budgets",
        "config_digest",
        "certificate_manifest_digest",
        "normalization_digest",
        "rff_map_digest",
        "swe_map_digest",
        "queries",
        "index_digest",
    }
    unsigned = {key: item for key, item in value.items() if key != "index_digest"}
    queries = value.get("queries")
    if (
        set(value) != fields
        or value.get("schema") != "policy-learnware.v05-public-query-index.v1"
        or value.get("status") != "COMPLETE"
        or value.get("query_count") != 30
        or value.get("authorized_artifact_count") != 180
        or tuple(value.get("budgets", ())) != _DEVELOPMENT_BUDGETS
        or value.get("config_digest") != panel.config_digest
        or value.get("certificate_manifest_digest")
        != panel.certificate_manifest.certificate_manifest_digest
        or value.get("normalization_digest")
        != panel.source_binding["normalization_digest"]
        or value.get("rff_map_digest") != panel.rff.rff_map.map_digest
        or value.get("swe_map_digest") != panel.swe.swe_map.map_digest
        or value.get("index_digest") != sha256_json(unsigned)
        or not isinstance(queries, Mapping)
        or len(queries) != 30
        or any(_OPAQUE_QUERY_ID.fullmatch(item) is None for item in queries)
        or any(not isinstance(item, str) for item in queries.values())
        or len(set(queries.values())) != 30
    ):
        raise V05RunnerError("precommitted public query index differs")
    for digest in queries.values():
        _digest(digest, "precommitted query manifest digest")
    return value


def _prepare_query_index(
    panel: P0Panel,
    projection: Any,
    assets: FrozenR4Assets,
    run_dir: Path,
    *,
    resume: bool,
) -> Path:
    """Privileged one-time projection of the seven held-repeat episodes."""

    run_root = Path(run_dir).expanduser()
    public_root = run_root / "public"
    private_root = run_root / "private"
    queries_root = public_root / "queries"
    binding_root = private_root / "query_bindings"
    index_path = public_root / "query_index.json"
    nonce_path = private_root / "blinding_nonce.json"
    progress_path = private_root / "query_prepare_progress.json"
    directories = (run_root, public_root, private_root, queries_root, binding_root)
    files = (nonce_path, progress_path, index_path)
    if (
        any(path.is_symlink() for path in (*directories, *files))
        or any(path.exists() and not path.is_dir() for path in directories)
        or any(path.exists() and not path.is_file() for path in files)
    ):
        raise V05RunnerError("query preparation path is absent or unsafe")
    resolved_run = run_root.resolve()
    resolved_public = public_root.resolve()
    resolved_private = private_root.resolve()
    if (
        resolved_public == resolved_run
        or not resolved_public.is_relative_to(resolved_run)
        or resolved_private == resolved_run
        or not resolved_private.is_relative_to(resolved_run)
        or resolved_public == resolved_private
        or resolved_public.is_relative_to(resolved_private)
        or resolved_private.is_relative_to(resolved_public)
        or not queries_root.resolve().is_relative_to(resolved_public)
        or not binding_root.resolve().is_relative_to(resolved_private)
    ):
        raise V05RunnerError("public/private query roots overlap or escape run_dir")
    if index_path.exists():
        if not resume:
            raise V05RunnerError("public query index already exists")
        _read_query_index(panel, index_path)
        return index_path
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if nonce_path.exists():
        if not resume:
            raise V05RunnerError("private blinding nonce already exists")
        nonce_row = load_strict_json(nonce_path)
        if (
            set(nonce_row) != {"schema", "config_digest", "nonce"}
            or nonce_row.get("config_digest") != panel.config_digest
        ):
            raise V05RunnerError("private blinding nonce binding differs")
        nonce = str(nonce_row["nonce"])
    else:
        nonce = secrets.token_hex(32)
        atomic_write_json(
            nonce_path,
            {
                "schema": "policy-learnware.v05-private-blinding-nonce.v1",
                "config_digest": panel.config_digest,
                "nonce": nonce,
            },
        )
        os.chmod(nonce_path, 0o600)
    progress = load_strict_json(progress_path) if progress_path.exists() else {}
    anchor_to_query = dict(progress.get("anchor_to_query", {}))
    if anchor_to_query and (
        progress.get("config_digest") != panel.config_digest
        or not set(anchor_to_query).issubset(projection.source_repeat)
    ):
        raise V05RunnerError("private query-prepare progress differs")
    manifest_by_query: dict[str, str] = {}
    for source_id in sorted(projection.source_repeat):
        if source_id in anchor_to_query:
            prepared_query_root = queries_root / anchor_to_query[source_id]
            views = load_authorized_query(
                prepared_query_root,
                rff_map=panel.rff.rff_map,
                swe_map=panel.swe.swe_map,
            )
        else:
            views = prepare_blinded_episode_bank(
                episode_bank=projection.source_repeat[source_id],
                parent_source_role_manifest=projection.manifest,
                expected_source_role_manifest_digest=projection.manifest[
                    "source_role_manifest_digest"
                ],
                parent_asset_sha256=assets.parent_asset_sha256[source_id],
                parent_membership_digest=assets.parent_membership_digest[source_id],
                probe_protocol_digest=assets.probe_protocol_digest,
                public_parent=queries_root,
                private_binding_root=binding_root,
                truth_source_anchor_id=source_id,
                certificate_manifest=panel.certificate_manifest,
                blinding_nonce=nonce,
                normalization_digest=panel.source_binding["normalization_digest"],
                rff_map=panel.rff.rff_map,
                swe_map=panel.swe.swe_map,
                authorized_budgets=_DEVELOPMENT_BUDGETS,
            )
            anchor_to_query[source_id] = views.opaque_query_id
            atomic_write_json(
                progress_path,
                {
                    "schema": "policy-learnware.v05-private-query-progress.v1",
                    "config_digest": panel.config_digest,
                    "anchor_to_query": anchor_to_query,
                },
                overwrite=progress_path.exists(),
            )
        manifest_by_query[views.opaque_query_id] = views.manifest["manifest_digest"]
    if len(manifest_by_query) != 30 or len(set(manifest_by_query.values())) != 30:
        raise V05RunnerError("prepared public query identity coverage differs")
    unsigned = {
        "schema": "policy-learnware.v05-public-query-index.v1",
        "status": "COMPLETE",
        "query_count": 30,
        "authorized_artifact_count": 180,
        "budgets": list(_DEVELOPMENT_BUDGETS),
        "config_digest": panel.config_digest,
        "certificate_manifest_digest": (
            panel.certificate_manifest.certificate_manifest_digest
        ),
        "normalization_digest": panel.source_binding["normalization_digest"],
        "rff_map_digest": panel.rff.rff_map.map_digest,
        "swe_map_digest": panel.swe.swe_map.map_digest,
        "queries": dict(sorted(manifest_by_query.items())),
    }
    atomic_write_json(index_path, {**unsigned, "index_digest": sha256_json(unsigned)})
    _read_query_index(panel, index_path)
    return index_path


def _decode_score_cell(
    value: Mapping[str, Any], query_id: str, method_id: str, manifest_digest: str
) -> tuple[tuple[PredictionRanking, ...], tuple[dict[str, Any], ...]]:
    unsigned = {key: item for key, item in value.items() if key != "cell_digest"}
    if (
        set(value)
        != {
            "schema",
            "opaque_query_id",
            "method_id",
            "authorized_query_manifest_digest",
            "predictions",
            "score_rows",
            "elapsed_seconds",
            "peak_rss_bytes",
            "cell_digest",
        }
        or value.get("schema") != "policy-learnware.v05-score-cell.v1"
        or value.get("opaque_query_id") != query_id
        or value.get("method_id") != method_id
        or value.get("authorized_query_manifest_digest") != manifest_digest
        or value.get("cell_digest") != sha256_json(unsigned)
        or not isinstance(value.get("predictions"), list)
        or not isinstance(value.get("score_rows"), list)
    ):
        raise V05RunnerError("persisted score cell differs")
    predictions = tuple(
        PredictionRanking.from_dict(item) for item in value["predictions"]
    )
    if (
        len(predictions) != 6
        or len(value["score_rows"]) != 6
        or any(
            item.opaque_query_id != query_id or item.method_id != method_id
            for item in predictions
        )
    ):
        raise V05RunnerError("persisted score cell coverage differs")
    return predictions, tuple(dict(item) for item in value["score_rows"])


def score_precommitted_queries(
    panel: P0Panel,
    query_index_path: str | Path,
    scoring_root: str | Path,
    *,
    resume: bool,
) -> tuple[dict[str, Any], tuple[PredictionRanking, ...], tuple[dict[str, Any], ...]]:
    """Production scorer: only strict public paths from the frozen q index enter."""

    index_path = Path(query_index_path)
    index = _read_query_index(panel, index_path)
    root = Path(scoring_root)
    if root.is_symlink():
        raise V05RunnerError("score-cell root cannot be a symlink")
    predictions: list[PredictionRanking] = []
    score_rows: list[dict[str, Any]] = []
    for query_id, manifest_digest in sorted(index["queries"].items()):
        views = load_authorized_query(
            index_path.parent / "queries" / query_id,
            rff_map=panel.rff.rff_map,
            swe_map=panel.swe.swe_map,
        )
        if (
            views.opaque_query_id != query_id
            or views.manifest["manifest_digest"] != manifest_digest
            or views.authorized_budgets != _DEVELOPMENT_BUDGETS
        ):
            raise V05RunnerError("authorized query differs from its precommitment")
        missing = [
            method
            for method in P0_METHOD_IDS
            if not (root / query_id / f"{method}.json").exists()
        ]
        fresh_predictions: tuple[PredictionRanking, ...] = ()
        fresh_scores: tuple[dict[str, Any], ...] = ()
        fresh_timings: dict[str, float] = {}
        if missing:
            fresh_predictions, fresh_scores = score_query(
                panel, views, timings=fresh_timings
            )
        for method_id in P0_METHOD_IDS:
            path = root / query_id / f"{method_id}.json"
            if path.exists():
                if not resume:
                    raise V05RunnerError("score cell already exists")
                cell = load_strict_json(path)
            else:
                method_predictions = [
                    item.to_dict()
                    for item in fresh_predictions
                    if item.method_id == method_id
                ]
                method_scores = [
                    item for item in fresh_scores if item["method_id"] == method_id
                ]
                unsigned = {
                    "schema": "policy-learnware.v05-score-cell.v1",
                    "opaque_query_id": query_id,
                    "method_id": method_id,
                    "authorized_query_manifest_digest": manifest_digest,
                    "predictions": method_predictions,
                    "score_rows": method_scores,
                    "elapsed_seconds": fresh_timings[method_id],
                    "peak_rss_bytes": _rss_bytes(),
                }
                cell = {**unsigned, "cell_digest": sha256_json(unsigned)}
                atomic_write_json(path, cell)
            cell_predictions, cell_scores = _decode_score_cell(
                cell, query_id, method_id, manifest_digest
            )
            predictions.extend(cell_predictions)
            score_rows.extend(cell_scores)
    expected_files = {
        root / query_id / f"{method_id}.json"
        for query_id in index["queries"]
        for method_id in P0_METHOD_IDS
    }
    observed_files = set(root.glob("*/*.json")) if root.exists() else set()
    if observed_files != expected_files:
        raise V05RunnerError(
            "score cell directory is not the canonical 180-cell closure"
        )
    return index, tuple(predictions), tuple(score_rows)


def _validate_development_batch(
    panel: P0Panel,
    index: Mapping[str, Any],
    predictions: Sequence[PredictionRanking],
    score_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[PredictionRanking, ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    rows = require_prediction_cell_coverage(
        predictions,
        expected_method_ids=P0_METHOD_IDS,
        expected_budgets=_DEVELOPMENT_BUDGETS,
    )
    if len(rows) != 1080 or {item.opaque_query_id for item in rows} != set(
        index["queries"]
    ):
        raise V05RunnerError("global prediction panel is not exactly 30x36")
    ranking_by_cell = {item.cell_key: item for item in rows}
    source_order = tuple(panel.resolver.anchor_ids)
    expected_source_binding_digest = sha256_json(dict(panel.source_binding))
    normalized_scores: list[dict[str, Any]] = []
    ledgers: dict[str, Any] = {}
    seen_score_cells: set[tuple[str, str, int, str]] = set()
    required = {
        "method_id",
        "endpoint",
        "budget_episodes",
        "opaque_query_id",
        "scores_before_mask",
        "source_binding",
        "target_binding",
        "ledger",
        "score_vector_digest",
        "budget_ledger_digest",
    }
    for raw in score_rows:
        row = dict(_exact_fields(raw, required, "score row"))
        key = (
            row["method_id"],
            row["endpoint"],
            row["budget_episodes"],
            row["opaque_query_id"],
        )
        if key in seen_score_cells:
            raise V05RunnerError("global score material contains a duplicate cell")
        seen_score_cells.add(key)
        ranking = ranking_by_cell.get(key)
        scores = row["scores_before_mask"]
        ledger = BudgetLedger.for_budget(row["budget_episodes"]).to_dict()
        target = row["target_binding"]
        expected_target_fields = {
            "probe_protocol_digest",
            "reward_free_bank_sha256",
            "target_membership_digest",
            "normalization_digest",
            "authorized_query_manifest_digest",
            "canonical_query_bank_digest",
        }
        score_digest = sha256_json(
            {
                "method_id": row["method_id"],
                "budget_episodes": row["budget_episodes"],
                "opaque_query_id": row["opaque_query_id"],
                "source_order": list(source_order),
                "scores_before_mask": scores,
            }
        )
        if (
            ranking is None
            or len(scores) != 30
            or any(not math.isfinite(float(value)) for value in scores)
            or not isinstance(row["source_binding"], Mapping)
            or sha256_json(row["source_binding"]) != expected_source_binding_digest
            or row["ledger"] != ledger
            or row["score_vector_digest"] != score_digest
            or ranking.score_vector_digest != score_digest
            or row["budget_ledger_digest"] != sha256_json(ledger)
            or ranking.budget_ledger_digest != sha256_json(ledger)
            or ranking.authorized_query_manifest_digest
            != index["queries"].get(row["opaque_query_id"])
            or not isinstance(target, Mapping)
            or set(target) != expected_target_fields
            or target["probe_protocol_digest"] != ranking.probe_protocol_digest
            or target["reward_free_bank_sha256"] != ranking.reward_free_bank_sha256
            or target["target_membership_digest"] != ranking.target_membership_digest
            or target["normalization_digest"] != ranking.normalization_digest
            or target["authorized_query_manifest_digest"]
            != ranking.authorized_query_manifest_digest
            or target["canonical_query_bank_digest"]
            != ranking.canonical_query_bank_digest
        ):
            raise V05RunnerError("score/ledger material differs from sealed ranking")
        normalized_scores.append(row)
        ledgers[f"{row['opaque_query_id']}/{row['budget_episodes']}"] = ledger
    if (
        len(normalized_scores) != 1080
        or seen_score_cells != set(ranking_by_cell)
        or len(ledgers) != 90
    ):
        raise V05RunnerError("global score/ledger coverage differs")
    return (
        tuple(sorted(rows, key=lambda item: item.cell_key)),
        tuple(
            sorted(
                normalized_scores,
                key=lambda item: (
                    item["method_id"],
                    item["endpoint"],
                    item["budget_episodes"],
                    item["opaque_query_id"],
                ),
            )
        ),
        dict(sorted(ledgers.items())),
    )


def _publish_global_seal(
    panel: P0Panel, query_index_path: Path, scoring_root: Path, *, resume: bool
) -> tuple[RankingSeal, dict[str, Any], dict[str, Any]]:
    index, predictions, score_rows = score_precommitted_queries(
        panel, query_index_path, scoring_root / "cells", resume=resume
    )
    predictions, score_rows, ledgers = _validate_development_batch(
        panel, index, predictions, score_rows
    )
    payload = prediction_payload(predictions)
    seal = seal_rankings(payload)
    cell_files = {
        path.relative_to(scoring_root).as_posix(): sha256_file(path)
        for path in sorted((scoring_root / "cells").glob("*/*.json"))
    }
    if len(cell_files) != 180:
        raise V05RunnerError("global seal requires the exact 180 score-cell files")
    unsigned = {
        "schema": "policy-learnware.v05-global-development-seal.v1",
        "status": "SEALED",
        "query_index_digest": index["index_digest"],
        "query_count": 30,
        "prediction_cell_count": 1080,
        "score_rows_digest": sha256_json(list(score_rows)),
        "budget_ledger_digest": sha256_json(ledgers),
        "cell_files": cell_files,
        "prediction_seal": seal.to_dict(),
    }
    receipt = {**unsigned, "global_seal_digest": sha256_json(unsigned)}
    _publish_or_match_json(scoring_root / "global_seal.json", receipt, resume=resume)
    return seal, receipt, index


def _evaluate_after_persisted_seal(
    panel: P0Panel,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    run_dir: Path,
    scoring_root: Path,
    *,
    source_input_bytes: int,
    resume: bool,
) -> dict[str, Any]:
    """Revalidate the unique public seal, then and only then open private truth."""

    index = _read_query_index(panel, run_dir / "public" / "query_index.json")
    seal_path = scoring_root / "global_seal.json"
    if scoring_root.is_symlink() or seal_path.is_symlink() or not seal_path.is_file():
        raise V05RunnerError("persisted global seal is absent or unsafe")
    receipt = load_strict_json(seal_path)
    fields = {
        "schema",
        "status",
        "query_index_digest",
        "query_count",
        "prediction_cell_count",
        "score_rows_digest",
        "budget_ledger_digest",
        "cell_files",
        "prediction_seal",
        "global_seal_digest",
    }
    unsigned = dict(_exact_fields(receipt, fields, "global seal receipt"))
    unsigned.pop("global_seal_digest")
    expected_cells = {
        f"cells/{query_id}/{method_id}.json"
        for query_id in index["queries"]
        for method_id in P0_METHOD_IDS
    }
    if (
        receipt["schema"] != "policy-learnware.v05-global-development-seal.v1"
        or receipt["status"] != "SEALED"
        or receipt["query_index_digest"] != index["index_digest"]
        or receipt["query_count"] != 30
        or receipt["prediction_cell_count"] != 1080
        or receipt["global_seal_digest"] != sha256_json(unsigned)
        or not isinstance(receipt["cell_files"], Mapping)
        or set(receipt["cell_files"]) != expected_cells
    ):
        raise V05RunnerError("persisted global seal closure differs")
    seal = RankingSeal.from_dict(receipt["prediction_seal"])
    verify_ranking_seal(seal, seal.rankings)
    predictions: list[PredictionRanking] = []
    score_rows: list[dict[str, Any]] = []
    method_runtime = {method_id: 0.0 for method_id in P0_METHOD_IDS}
    peak_rss = 0
    for relative in sorted(expected_cells):
        path = scoring_root / relative
        parts = Path(relative).parts
        query_id, method_file = parts[1], parts[2]
        method_id = method_file.removesuffix(".json")
        if (
            path.is_symlink()
            or not path.is_file()
            or receipt["cell_files"][relative] != sha256_file(path)
        ):
            raise V05RunnerError("sealed score-cell file changed")
        cell = load_strict_json(path)
        cell_predictions, cell_scores = _decode_score_cell(
            cell, query_id, method_id, index["queries"][query_id]
        )
        elapsed = float(cell["elapsed_seconds"])
        cell_rss = int(cell["peak_rss_bytes"])
        if not math.isfinite(elapsed) or elapsed < 0.0 or cell_rss < 0:
            raise V05RunnerError("score-cell runtime accounting differs")
        method_runtime[method_id] += elapsed
        peak_rss = max(peak_rss, cell_rss)
        predictions.extend(cell_predictions)
        score_rows.extend(cell_scores)
    predictions, score_rows, ledgers = _validate_development_batch(
        panel, index, predictions, score_rows
    )
    if (
        seal_rankings(prediction_payload(predictions)).rankings_digest
        != seal.rankings_digest
        or receipt["score_rows_digest"] != sha256_json(list(score_rows))
        or receipt["budget_ledger_digest"] != sha256_json(ledgers)
    ):
        raise V05RunnerError("persisted score material differs from the global seal")

    # Private capability materialization is deliberately below every public check.
    private_root = run_dir / "private"
    report_path = run_dir / "results" / "development_report.json"
    release_path = private_root / "truth_release.json"
    if report_path.exists() != release_path.exists():
        raise V05RunnerError("development report/truth release is partially published")
    if private_root.is_symlink():
        raise V05RunnerError("private truth root cannot be a symlink")
    nonce_row = load_strict_json(private_root / "blinding_nonce.json")
    if (
        set(nonce_row) != {"schema", "config_digest", "nonce"}
        or nonce_row["schema"] != "policy-learnware.v05-private-blinding-nonce.v1"
        or nonce_row["config_digest"] != panel.config_digest
    ):
        raise V05RunnerError("private blinding nonce binding differs")
    binding_root = private_root / "query_bindings"
    if binding_root.is_symlink():
        raise V05RunnerError("private truth binding root cannot be a symlink")
    expected_bindings = {
        binding_root / f"{query_id}.json" for query_id in index["queries"]
    }
    if set(binding_root.glob("*.json")) != expected_bindings:
        raise V05RunnerError("private truth binding closure differs")
    truths: list[TruthBinding] = []
    for query_id, manifest_digest in sorted(index["queries"].items()):
        truths.append(
            load_private_truth_binding(
                binding_root / f"{query_id}.json",
                opaque_query_id=query_id,
                authorized_query_manifest_digest=manifest_digest,
                prediction_seal=seal,
                certificate_manifest=panel.certificate_manifest,
                blinding_nonce=str(nonce_row["nonce"]),
            )
        )
    if (
        len(truths) != 30
        or {item.opaque_query_id for item in truths} != set(index["queries"])
        or {item.source_anchor_id for item in truths} != set(panel.resolver.anchor_ids)
    ):
        raise V05RunnerError("private truth release does not cover the 30 queries")
    release_unsigned = {
        "schema": "policy-learnware.v05-private-truth-release.v1",
        "status": "RELEASED_AFTER_SEAL",
        "global_seal_digest": receipt["global_seal_digest"],
        "prediction_seal_digest": seal.rankings_digest,
        "query_index_digest": index["index_digest"],
        "truth_count": 30,
        "truth_bindings": [item.to_dict() for item in sorted(truths)],
    }
    release = {
        **release_unsigned,
        "truth_release_digest": sha256_json(release_unsigned),
    }
    _publish_or_match_json(release_path, release, resume=resume)

    core = build_development_report(
        seal,
        truths,
        panel.certificate_manifest,
        P0_METHOD_IDS,
        _DEVELOPMENT_BUDGETS,
    )
    source_progress = load_strict_json(
        run_dir / "source_fit" / "models" / "progress.json"
    )
    source_elapsed = sum(
        float(row["elapsed_seconds"]) for row in source_progress["models"].values()
    )
    peak_rss = max(
        peak_rss,
        *(int(row["peak_rss_bytes"]) for row in source_progress["models"].values()),
    )
    file_bytes = lambda root: sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    report_unsigned = {
        "schema": "policy-learnware.v05-development-results.v1",
        "status": "COMPLETE",
        "config_digest": panel.config_digest,
        "query_index_digest": index["index_digest"],
        "global_seal_digest": receipt["global_seal_digest"],
        "prediction_seal_digest": seal.rankings_digest,
        "truth_release_digest": release["truth_release_digest"],
        "certificate": {
            **dict(provenance),
            "certificate_manifest_digest": (
                panel.certificate_manifest.certificate_manifest_digest
            ),
        },
        "privacy": dict(config["privacy"]),
        "method_cards": [
            {
                **card,
                "privacy_scope": config["privacy"]["method_scope"][card["method_id"]],
            }
            for card in panel.method_cards
        ],
        "accounting": {
            "budgets": [
                {
                    **BudgetLedger.for_budget(budget).to_dict(),
                    "actual_new_acquisition_steps": 0,
                    "evidence_scope": "REUSED_DEVELOPMENT_HELD_REPEAT",
                }
                for budget in _DEVELOPMENT_BUDGETS
            ],
            "budget_7_scope": "SANITY_ONLY_NOT_IN_AUC",
        },
        "resources": {
            "source_fit_elapsed_seconds_sum": source_elapsed,
            "score_elapsed_seconds_sum": sum(method_runtime.values()),
            "score_elapsed_seconds_by_method": method_runtime,
            "peak_rss_bytes": peak_rss,
            "frozen_source_input_array_bytes": source_input_bytes,
            "source_checkpoint_bytes": file_bytes(run_dir / "source_fit"),
            "public_query_artifact_bytes": file_bytes(run_dir / "public" / "queries"),
            "sealed_score_artifact_bytes": file_bytes(scoring_root),
            "source_train_rows": 30 * 19 * 64,
            "source_validation_rows": 30 * 6 * 64,
            "privileged_repeat_rows": 30 * 7 * 64,
            "visible_query_rows_by_budget": {
                str(budget): 30 * budget * 64 for budget in _DEVELOPMENT_BUDGETS
            },
        },
        "coverage": {
            "query_success": 30,
            "query_failure": 0,
            "source_model_checkpoint_success": 65,
            "score_cell_success": 180,
            "score_cell_failure": 0,
            "prediction_cell_success": 1080,
        },
        "metrics": core,
    }
    report = {**report_unsigned, "report_digest": sha256_json(report_unsigned)}
    _publish_or_match_json(report_path, report, resume=resume)
    return report


def run_development(
    config_path: str | Path,
    new_run_dir: str | Path,
    *,
    artifacts_root: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run or strictly resume the frozen r4 held-repeat development panel."""

    config, config_digest = load_development_config(config_path)
    assets = _load_frozen_r4_assets(config, config_digest, artifacts_root)
    requested_run_dir = Path(new_run_dir).expanduser()
    if not requested_run_dir.is_absolute():
        requested_run_dir = Path.cwd() / requested_run_dir
    lexical_run_dir = Path(os.path.abspath(requested_run_dir))
    if any(
        path.exists() and path.is_symlink()
        for path in (*reversed(lexical_run_dir.parents), lexical_run_dir)
    ):
        raise V05RunnerError("new run directory has a symlink ancestor")
    output_root = next(
        (parent for parent in lexical_run_dir.parents if parent.exists()), None
    )
    if output_root is None or not output_root.is_dir():
        raise V05RunnerError("new run directory has no valid output root")
    run_dir = lexical_run_dir.resolve()
    resolved_output_root = output_root.resolve()
    if run_dir == resolved_output_root or not run_dir.is_relative_to(
        resolved_output_root
    ):
        raise V05RunnerError("new run directory escapes its output root")
    if any(
        run_dir == frozen
        or run_dir.is_relative_to(frozen)
        or frozen.is_relative_to(run_dir)
        for frozen in (assets.r4_root, assets.v03_root)
    ):
        raise V05RunnerError("new run directory overlaps a frozen asset root")
    if resume:
        if not run_dir.is_dir():
            raise V05RunnerError("resume run directory is absent")
    else:
        if run_dir.exists():
            raise V05RunnerError("new run directory already exists")
        run_dir.mkdir(parents=True, mode=0o755)
    manifest_unsigned = {
        "schema": "policy-learnware.v05-development-run.v2",
        "config_digest": config_digest,
        "config_file_sha256": sha256_file(Path(config_path).expanduser()),
        "frozen_asset_layout": {
            "root_environment_variable": _ARTIFACTS_ROOT_ENV,
            "r4_relative_path": _R4_RELATIVE.as_posix(),
            "v03_relative_path": _V03_RELATIVE.as_posix(),
        },
        "frozen_provenance_digest": sha256_json(dict(assets.provenance)),
        "budgets": list(_DEVELOPMENT_BUDGETS),
        "source_count": 30,
        "method_ids": list(P0_METHOD_IDS),
    }
    run_manifest = {
        **manifest_unsigned,
        "run_manifest_digest": sha256_json(manifest_unsigned),
    }
    _publish_or_match_json(run_dir / "run_manifest.json", run_manifest, resume=resume)

    def stage_event(number: int, stage: str, artifact_digest: str) -> None:
        stable = {
            "schema": "policy-learnware.v05-development-event.v1",
            "stage": stage,
            "status": "COMPLETE",
            "run_manifest_digest": run_manifest["run_manifest_digest"],
            "artifact_digest": artifact_digest,
        }
        path = run_dir / "events" / f"{number:02d}-{stage}.json"
        fresh = not path.exists()
        if path.exists():
            existing = load_strict_json(path)
            if not resume or any(
                existing.get(key) != value for key, value in stable.items()
            ):
                raise V05RunnerError(f"development event changed: {stage}")
        else:
            atomic_write_json(path, {**stable, "recorded_at": utc_now()})
        if fresh:
            atomic_write_json(
                run_dir / "status.json",
                {**stable, "updated_at": utc_now()},
                overwrite=(run_dir / "status.json").exists(),
            )

    current_stage = "frozen-assets"
    try:
        stage_event(1, current_stage, manifest_unsigned["frozen_provenance_digest"])
        source_input_bytes = sum(
            array.nbytes
            for arrays in assets.arrays_by_anchor.values()
            for array in arrays.values()
        )
        current_stage = "canonical-source"
        full_banks, canonical = _canonical_source_stage(assets, run_dir, resume=resume)
        stage_event(2, current_stage, canonical["complete_digest"])
        current_stage = "source-models"
        panel, projection = _source_fit_stage(
            assets, full_banks, canonical, run_dir, resume=resume
        )
        stage_event(
            3,
            current_stage,
            panel.source_model_manifest["source_model_manifest_digest"],
        )
        current_stage = "opaque-query-prepare"
        query_index_path = _prepare_query_index(
            panel, projection, assets, run_dir, resume=resume
        )
        query_index = _read_query_index(panel, query_index_path)
        stage_event(4, current_stage, query_index["index_digest"])
        provenance = dict(assets.provenance)
        del projection, full_banks, assets
        current_stage = "global-score-seal"
        scoring_root = run_dir / "public" / "scoring"
        _, seal_receipt, _ = _publish_global_seal(
            panel, query_index_path, scoring_root, resume=resume
        )
        stage_event(5, current_stage, seal_receipt["global_seal_digest"])
        current_stage = "post-seal-evaluation"
        report = _evaluate_after_persisted_seal(
            panel,
            config,
            provenance,
            run_dir,
            scoring_root,
            source_input_bytes=source_input_bytes,
            resume=resume,
        )
        stage_event(6, current_stage, report["report_digest"])
        return report
    except Exception as error:
        atomic_write_json(
            run_dir / "status.json",
            {
                "schema": "policy-learnware.v05-development-status.v1",
                "stage": current_stage,
                "status": "FAILED",
                "run_manifest_digest": run_manifest["run_manifest_digest"],
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_at": utc_now(),
            },
            overwrite=(run_dir / "status.json").exists(),
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="%(prog)s 0.5.0")
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts-root")
    parser.add_argument("--new-run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run_development(
        args.config,
        args.new_run_dir,
        artifacts_root=args.artifacts_root,
        resume=bool(args.resume),
    )
    return 0


__all__ = [
    "P0Panel",
    "P1_STATUS",
    "Q0_COMMON_GAUSSIAN_OPEN_LOOP",
    "V05RunnerError",
    "fit_p0_panel",
    "run_development",
    "score_query",
    "seal_prediction_rows",
]


if __name__ == "__main__":
    raise SystemExit(main())
