"""Thin source-role and private-to-opaque projections for v0.5.

The privileged prepare side is the only code here that can open a private
collector directory or a truth binding.  The scorer-side loader accepts only a
new opaque directory containing method-specific, reward-free authorized views.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from policy_learnware_v0.hashing import sha256_file, sha256_json, sha256_ndarrays
from policy_learnware_v0.io import atomic_write_json, atomic_write_npz, read_json
from policy_learnware_v0.v04a.protocol import (
    BUDGET_EPISODES,
    RankingSeal,
    RewardFreeProbe,
    V04AProtocolError,
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
    EpisodeBank,
)
from policy_learnware_v0.v05.labels import (
    CertificateResolver,
    CertifiedPolicyManifest,
)
from policy_learnware_v0.v05.metrics import (
    MARKET_30_CERT,
    PREDICTION_PAYLOAD_SCHEMA,
    TASK_5_CERT,
    PredictionRanking,
    TruthBinding,
    V05MetricError,
    require_prediction_cell_coverage,
)
from policy_learnware_v0.v05.specifications import (
    RFFMap,
    RFFSpecification,
    SWEMap,
    SWESpecification,
)
from server.repro_fpo_ppo_v05.exact_repeat_collector import (
    load_published_collection,
)


SOURCE_TRAIN = "source_train"
SOURCE_VALIDATION = "source_validation"
SOURCE_REPEAT = "source_repeat_report"
SOURCE_ROLE_SLICES = MappingProxyType(
    {
        SOURCE_TRAIN: (0, 19),
        SOURCE_VALIDATION: (19, 25),
        SOURCE_REPEAT: (25, 32),
    }
)
SOURCE_ROLE_SCHEMA = "policy-learnware.v05-source-role-projection.v1"
PUBLIC_QUERY_SCHEMA = "policy-learnware.v05-authorized-query.v1"
PRIVATE_BINDING_SCHEMA = "policy-learnware.v05-private-query-binding.v1"
PUBLIC_MANIFEST_FILE = "query_manifest.json"
AUTHORIZED_VIEW_DIR = "authorized_views"
_FULL_BANK_METHODS = (RAW_DELTA_RKME, EMPIRICAL_MMD_NN, KME_KRR)
_FORBIDDEN_PUBLIC_KEYS = (
    "anchor",
    "path",
    "policy",
    "factor",
    "return",
    "expected",
    "label",
    "truth",
    "checkpoint",
)
_ACCESS_TIER = {
    RAW_DELTA_RKME: "STRUCTURED_SPEC",
    EMPIRICAL_MMD_NN: "FULL_SUPPORT_CONTROL",
    SUMMARY_LOGREG: "STRUCTURED_SPEC",
    KME_KRR: "JOINT_SOURCE_FIT",
    RFF_KME_NN: "FIXED_VECTOR_SPEC",
    SWE_NN: "FIXED_VECTOR_SPEC",
}
_PUBLIC_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "opaque_query_id",
        "certificate_manifest_digest",
        "probe_protocol_digest",
        "target_membership_digest",
        "normalization_digest",
        "reward_free_bank_sha256",
        "canonical_query_bank_digest",
        "canonical_budget_bank_digests",
        "market_order_digest",
        "task_scope_digest",
        "execution_abi_digest",
        "candidate_mask",
        "budget_episodes",
        "authorized_views",
        "manifest_digest",
    }
)
_VIEW_FIELDS = frozenset(
    {
        "access_tier",
        "artifact_sha256",
        "arrays_digest",
        "scorer_visible_raw_rows",
    }
)
_HELD_DEVELOPMENT_BUDGETS = (1, 2, 4)


class V05BlindError(ValueError):
    """A source split or public query projection is unsafe or inconsistent."""


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise V05BlindError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    value = _text(value, where)
    if len(value) != 64 or value != value.lower():
        raise V05BlindError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise V05BlindError(f"{where} must be a lowercase SHA-256 digest") from error
    return value


def _authorized_budget_tuple(
    values: Sequence[int], *, episode_count: int
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise V05BlindError("authorized_budgets must be an integer sequence")
    try:
        raw = tuple(values)
    except TypeError as error:
        raise V05BlindError("authorized_budgets must be an integer sequence") from error
    if not raw:
        raise V05BlindError("authorized_budgets must not be empty")
    budgets: list[int] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise V05BlindError("authorized_budgets must contain integers")
        budgets.append(int(value))
    result = tuple(budgets)
    if (
        result != tuple(sorted(set(result)))
        or not set(result).issubset(BUDGET_EPISODES)
        or result[-1] > episode_count
    ):
        raise V05BlindError(
            "authorized_budgets must be a nested frozen subset within the bank"
        )
    return result


def _episode_slice(bank: EpisodeBank, start: int, stop: int) -> EpisodeBank:
    first = int(bank.episode_offsets[start])
    last = int(bank.episode_offsets[stop])
    offsets = bank.episode_offsets[start : stop + 1] - first
    return EpisodeBank(bank.points[first:last], offsets)


def _bank_mapping(
    values: Mapping[str, EpisodeBank], where: str
) -> dict[str, EpisodeBank]:
    if not isinstance(values, Mapping) or not values:
        raise V05BlindError(f"{where} must be a non-empty bank mapping")
    result: dict[str, EpisodeBank] = {}
    for source_id in sorted(values):
        identifier = _text(source_id, f"{where} source ID")
        bank = values[source_id]
        if not isinstance(bank, EpisodeBank):
            raise V05BlindError(f"{where} values must be EpisodeBank objects")
        result[identifier] = bank
    return result


def _role_membership_digest_from_parent(
    source_id: str,
    parent_bank_digest: str,
    parent_asset_sha256: str,
    parent_membership_digest: str,
    role: str,
    start: int,
    stop: int,
) -> str:
    return sha256_json(
        {
            "source_id": source_id,
            "parent_asset_sha256": parent_asset_sha256,
            "parent_membership_digest": parent_membership_digest,
            "parent_bank_digest": parent_bank_digest,
            "role": role,
            "episode_positions": list(range(start, stop)),
        }
    )


def _role_membership_digest(
    source_id: str,
    full_bank: EpisodeBank,
    parent_asset_sha256: str,
    parent_membership_digest: str,
    role: str,
    start: int,
    stop: int,
) -> str:
    return _role_membership_digest_from_parent(
        source_id,
        full_bank.bank_digest,
        parent_asset_sha256,
        parent_membership_digest,
        role,
        start,
        stop,
    )


@dataclass(frozen=True)
class SourceRoleProjection:
    """Three disjoint views that can only be validated against one full bank."""

    full_banks: Mapping[str, EpisodeBank]
    parent_asset_sha256: Mapping[str, str]
    parent_membership_digest: Mapping[str, str]
    source_train: Mapping[str, EpisodeBank]
    source_validation: Mapping[str, EpisodeBank]
    source_repeat: Mapping[str, EpisodeBank]

    def __post_init__(self) -> None:
        full = _bank_mapping(self.full_banks, "full source")
        if (
            not isinstance(self.parent_asset_sha256, Mapping)
            or not isinstance(self.parent_membership_digest, Mapping)
            or set(self.parent_asset_sha256) != set(full)
            or set(self.parent_membership_digest) != set(full)
        ):
            raise V05BlindError("parent source evidence coverage differs")
        parent_assets = {
            source_id: _digest(
                self.parent_asset_sha256[source_id],
                f"{source_id} parent_asset_sha256",
            )
            for source_id in full
        }
        parent_memberships = {
            source_id: _digest(
                self.parent_membership_digest[source_id],
                f"{source_id} parent_membership_digest",
            )
            for source_id in full
        }
        roles = {
            SOURCE_TRAIN: _bank_mapping(self.source_train, SOURCE_TRAIN),
            SOURCE_VALIDATION: _bank_mapping(self.source_validation, SOURCE_VALIDATION),
            SOURCE_REPEAT: _bank_mapping(self.source_repeat, SOURCE_REPEAT),
        }
        if any(set(role_banks) != set(full) for role_banks in roles.values()):
            raise V05BlindError("source role coverage differs from the full banks")
        if any(
            bank.episode_count != 32 or np.any(np.diff(bank.episode_offsets) != 64)
            for bank in full.values()
        ):
            raise V05BlindError("each verified source bank must contain 32 episodes")
        positions = [
            set(range(start, stop)) for start, stop in SOURCE_ROLE_SLICES.values()
        ]
        if any(
            positions[i] & positions[j] for i in range(3) for j in range(i)
        ) or set.union(*positions) != set(range(32)):
            raise V05BlindError("source role positions overlap or are incomplete")
        for source_id, full_bank in full.items():
            for role, (start, stop) in SOURCE_ROLE_SLICES.items():
                expected = _episode_slice(full_bank, start, stop)
                if roles[role][source_id].bank_digest != expected.bank_digest:
                    raise V05BlindError(
                        f"{source_id} {role} is not the frozen slice of its full bank"
                    )
        object.__setattr__(self, "full_banks", MappingProxyType(full))
        object.__setattr__(self, "parent_asset_sha256", MappingProxyType(parent_assets))
        object.__setattr__(
            self,
            "parent_membership_digest",
            MappingProxyType(parent_memberships),
        )
        object.__setattr__(self, "source_train", MappingProxyType(roles[SOURCE_TRAIN]))
        object.__setattr__(
            self,
            "source_validation",
            MappingProxyType(roles[SOURCE_VALIDATION]),
        )
        object.__setattr__(
            self, "source_repeat", MappingProxyType(roles[SOURCE_REPEAT])
        )

    @property
    def manifest(self) -> dict[str, Any]:
        anchors: dict[str, Any] = {}
        role_memberships: dict[str, dict[str, str]] = {
            role: {} for role in SOURCE_ROLE_SLICES
        }
        role_banks: dict[str, dict[str, str]] = {
            role: {} for role in SOURCE_ROLE_SLICES
        }
        for source_id, full_bank in self.full_banks.items():
            role_rows = {}
            for role, (start, stop) in SOURCE_ROLE_SLICES.items():
                role_bank = getattr(
                    self, role if role != SOURCE_REPEAT else "source_repeat"
                )[source_id]
                membership = _role_membership_digest(
                    source_id,
                    full_bank,
                    self.parent_asset_sha256[source_id],
                    self.parent_membership_digest[source_id],
                    role,
                    start,
                    stop,
                )
                role_memberships[role][source_id] = membership
                role_banks[role][source_id] = role_bank.bank_digest
                role_rows[role] = {
                    "episode_positions": list(range(start, stop)),
                    "membership_digest": membership,
                    "bank_digest": role_bank.bank_digest,
                }
            anchors[source_id] = {
                "parent_asset_sha256": self.parent_asset_sha256[source_id],
                "parent_membership_digest": self.parent_membership_digest[source_id],
                "full_bank_digest": full_bank.bank_digest,
                "roles": role_rows,
            }
        payload = {
            "schema": SOURCE_ROLE_SCHEMA,
            "source_count": len(anchors),
            "full_bank_digest": sha256_json(
                {
                    source_id: bank.bank_digest
                    for source_id, bank in self.full_banks.items()
                }
            ),
            "parent_asset_binding_digest": sha256_json(
                {
                    source_id: {
                        "parent_asset_sha256": self.parent_asset_sha256[source_id],
                        "parent_membership_digest": self.parent_membership_digest[
                            source_id
                        ],
                        "full_bank_digest": self.full_banks[source_id].bank_digest,
                    }
                    for source_id in self.full_banks
                }
            ),
            "roles": {
                role: {
                    "episode_count": stop - start,
                    "membership_digest": sha256_json(role_memberships[role]),
                    "bank_digest": sha256_json(role_banks[role]),
                }
                for role, (start, stop) in SOURCE_ROLE_SLICES.items()
            },
            "sources": anchors,
        }
        payload["source_role_manifest_digest"] = sha256_json(payload)
        return payload


def project_verified_source_banks(
    full_banks: Mapping[str, EpisodeBank],
    *,
    parent_asset_sha256: Mapping[str, str],
    parent_membership_digest: Mapping[str, str],
    expected_source_count: int = 30,
) -> SourceRoleProjection:
    """Derive 19/6/7 from each one-and-only verified 32-episode bank."""

    full = _bank_mapping(full_banks, "full source")
    if (
        isinstance(expected_source_count, (bool, np.bool_))
        or not isinstance(expected_source_count, (int, np.integer))
        or int(expected_source_count) <= 0
        or len(full) != int(expected_source_count)
    ):
        raise V05BlindError("full source market has the wrong size")
    split = {
        role: {
            source_id: _episode_slice(bank, start, stop)
            for source_id, bank in full.items()
        }
        for role, (start, stop) in SOURCE_ROLE_SLICES.items()
    }
    return SourceRoleProjection(
        full_banks=full,
        parent_asset_sha256=parent_asset_sha256,
        parent_membership_digest=parent_membership_digest,
        source_train=split[SOURCE_TRAIN],
        source_validation=split[SOURCE_VALIDATION],
        source_repeat=split[SOURCE_REPEAT],
    )


def _summary_rows(bank: EpisodeBank) -> np.ndarray:
    return np.asarray(
        [
            np.concatenate(
                (
                    np.mean(bank.episode(index), axis=0),
                    np.std(bank.episode(index), axis=0),
                )
            )
            for index in range(bank.episode_count)
        ],
        dtype=np.float64,
    )


def _assert_disjoint(left: Path, right: Path) -> None:
    left = left.expanduser().resolve()
    right = right.expanduser().resolve()
    if left == right or left in right.parents or right in left.parents:
        raise V05BlindError(f"private and public roots overlap: {left} / {right}")


def _assert_public_safe(value: Any, private_values: Sequence[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = _text(key, "public manifest key").lower()
            if any(token in name for token in _FORBIDDEN_PUBLIC_KEYS):
                raise V05BlindError(f"public manifest contains forbidden field: {key}")
            _assert_public_safe(item, private_values)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_public_safe(item, private_values)
    elif isinstance(value, str) and value in set(private_values):
        raise V05BlindError("public manifest contains a private identity")


def _validate_public_manifest_shape(manifest: Mapping[str, Any]) -> None:
    if set(manifest) != _PUBLIC_MANIFEST_FIELDS:
        raise V05BlindError("authorized query manifest fields differ")
    raw_budgets = manifest.get("budget_episodes")
    if not isinstance(raw_budgets, list):
        raise V05BlindError("authorized query budgets differ")
    budgets = _authorized_budget_tuple(raw_budgets, episode_count=max(BUDGET_EPISODES))
    mask = manifest.get("candidate_mask")
    if (
        not isinstance(mask, list)
        or not mask
        or any(type(item) is not bool for item in mask)
    ):
        raise V05BlindError("authorized query candidate mask is malformed")
    views = manifest.get("authorized_views")
    if not isinstance(views, Mapping) or set(views) != set(P0_METHOD_IDS):
        raise V05BlindError("authorized method coverage differs")
    for method_id in P0_METHOD_IDS:
        row = views[method_id]
        expected_fields = _VIEW_FIELDS | (
            {"map_digest"} if method_id in {RFF_KME_NN, SWE_NN} else set()
        )
        visible_rows = (
            row.get("scorer_visible_raw_rows") if isinstance(row, Mapping) else None
        )
        visible_rows_valid = (
            isinstance(visible_rows, int)
            and not isinstance(visible_rows, bool)
            and (
                visible_rows > 0
                if method_id in _FULL_BANK_METHODS
                else visible_rows == 0
            )
        )
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_fields
            or row.get("access_tier") != _ACCESS_TIER[method_id]
            or not visible_rows_valid
        ):
            raise V05BlindError(f"{method_id} authorized view metadata differs")


@dataclass(frozen=True)
class AuthorizedQueryViews:
    """Scorer-only views loaded from one verified public opaque directory."""

    manifest: Mapping[str, Any]
    banks: Mapping[str, EpisodeBank]
    summary_rows: np.ndarray
    rff_specs: Mapping[int, RFFSpecification]
    swe_specs: Mapping[int, SWESpecification]

    def __post_init__(self) -> None:
        manifest = dict(self.manifest)
        if manifest.get("schema") != PUBLIC_QUERY_SCHEMA:
            raise V05BlindError("authorized query manifest has the wrong schema")
        unsigned = {
            key: value for key, value in manifest.items() if key != "manifest_digest"
        }
        if manifest.get("manifest_digest") != sha256_json(unsigned):
            raise V05BlindError("authorized query manifest digest changed")
        _assert_public_safe(manifest, ())
        _validate_public_manifest_shape(manifest)
        banks = _bank_mapping(self.banks, "authorized full-support")
        if set(banks) != set(_FULL_BANK_METHODS):
            raise V05BlindError("authorized full-support method coverage differs")
        episode_counts = {bank.episode_count for bank in banks.values()}
        if len(episode_counts) != 1:
            raise V05BlindError("authorized method bank episode counts differ")
        episode_count = next(iter(episode_counts))
        authorized_budgets = _authorized_budget_tuple(
            manifest["budget_episodes"], episode_count=episode_count
        )
        bank_digests = {bank.bank_digest for bank in banks.values()}
        if bank_digests != {manifest.get("canonical_query_bank_digest")}:
            raise V05BlindError("authorized method banks differ")
        raw_budget_digests = manifest.get("canonical_budget_bank_digests")
        if not isinstance(raw_budget_digests, Mapping) or set(raw_budget_digests) != {
            str(budget) for budget in authorized_budgets
        }:
            raise V05BlindError("canonical budget bank digest coverage differs")
        budget_digests = {
            str(budget): _digest(
                raw_budget_digests[str(budget)],
                f"canonical budget {budget} bank digest",
            )
            for budget in authorized_budgets
        }
        if any(
            bank.prefix(budget).bank_digest != budget_digests[str(budget)]
            for bank in banks.values()
            for budget in authorized_budgets
        ):
            raise V05BlindError("canonical budget bank digest changed")
        views = manifest["authorized_views"]
        if any(
            views[method_id]["scorer_visible_raw_rows"]
            != banks[method_id].points.shape[0]
            for method_id in _FULL_BANK_METHODS
        ):
            raise V05BlindError("authorized full-support row accounting differs")
        summaries = np.asarray(self.summary_rows, dtype=np.float64)
        if (
            summaries.ndim != 2
            or summaries.shape[0] != episode_count
            or not np.all(np.isfinite(summaries))
            or not np.array_equal(summaries, _summary_rows(banks[RAW_DELTA_RKME]))
        ):
            raise V05BlindError("authorized summary rows are malformed")
        rff = dict(self.rff_specs)
        swe = dict(self.swe_specs)
        if set(rff) != set(authorized_budgets) or set(swe) != set(authorized_budgets):
            raise V05BlindError("fixed-vector views do not cover authorized budgets")
        summaries = np.array(summaries, copy=True)
        summaries.setflags(write=False)
        object.__setattr__(self, "manifest", MappingProxyType(manifest))
        object.__setattr__(self, "banks", MappingProxyType(banks))
        object.__setattr__(self, "summary_rows", summaries)
        object.__setattr__(self, "rff_specs", MappingProxyType(rff))
        object.__setattr__(self, "swe_specs", MappingProxyType(swe))

    @property
    def opaque_query_id(self) -> str:
        return str(self.manifest["opaque_query_id"])

    @property
    def candidate_mask(self) -> tuple[bool, ...]:
        return tuple(bool(item) for item in self.manifest["candidate_mask"])

    @property
    def authorized_budgets(self) -> tuple[int, ...]:
        return tuple(int(item) for item in self.manifest["budget_episodes"])

    def bank_for(self, method_id: str, budget: int) -> EpisodeBank:
        if method_id not in self.banks:
            raise V05BlindError(f"{method_id} has no full-support authorized view")
        if budget not in self.authorized_budgets:
            raise V05BlindError("full-support budget is not authorized")
        return self.banks[method_id].prefix(budget)

    def summaries_for(self, budget: int) -> np.ndarray:
        if budget not in self.authorized_budgets:
            raise V05BlindError("summary budget is not authorized")
        return self.summary_rows[:budget]

    def canonical_bank_digest_for_budget(self, budget: int) -> str:
        if budget not in self.authorized_budgets:
            raise V05BlindError("canonical bank digest budget is not authorized")
        return str(self.manifest["canonical_budget_bank_digests"][str(budget)])


def _view_arrays(
    bank: EpisodeBank,
    rff_map: RFFMap,
    swe_map: SWEMap,
    authorized_budgets: tuple[int, ...],
) -> dict[str, dict[str, np.ndarray]]:
    summaries = _summary_rows(bank)
    budget_array = np.asarray(authorized_budgets, dtype=np.int64)
    rff_vectors = np.stack(
        [
            rff_map.embed(
                bank.prefix(budget).points, bank.prefix(budget).episode_offsets
            ).vector
            for budget in authorized_budgets
        ]
    )
    swe_vectors = np.stack(
        [
            swe_map.embed(
                bank.prefix(budget).points, bank.prefix(budget).episode_offsets
            ).vector
            for budget in authorized_budgets
        ]
    )
    full = {"points": bank.points, "episode_offsets": bank.episode_offsets}
    return {
        RAW_DELTA_RKME: full,
        EMPIRICAL_MMD_NN: full,
        SUMMARY_LOGREG: {"episode_summaries": summaries},
        KME_KRR: full,
        RFF_KME_NN: {"budget_episodes": budget_array, "vectors": rff_vectors},
        SWE_NN: {"budget_episodes": budget_array, "vectors": swe_vectors},
    }


def _publish_blinded_bank(
    *,
    bank: EpisodeBank,
    public_parent: str | Path,
    private_binding_root: str | Path,
    truth_source_anchor_id: str,
    certificate_manifest: CertifiedPolicyManifest,
    blinding_nonce: str,
    probe_protocol_digest: str,
    normalization_digest: str,
    rff_map: RFFMap,
    swe_map: SWEMap,
    authorized_budgets: Sequence[int],
    private_evidence: Mapping[str, Any],
    private_values: Sequence[str],
    expected_candidate_count: int = 5,
) -> AuthorizedQueryViews:
    """Publish one already-verified canonical bank through a single blind path."""

    if not isinstance(bank, EpisodeBank):
        raise V05BlindError("blind publication requires an EpisodeBank")
    nonce = _text(blinding_nonce, "blinding_nonce")
    if len(nonce) < 16:
        raise V05BlindError("blinding_nonce must contain at least 16 characters")
    protocol_digest = _digest(probe_protocol_digest, "probe_protocol_digest")
    normalizer = _digest(normalization_digest, "normalization_digest")
    if (
        rff_map.normalization_digest != normalizer
        or swe_map.normalization_digest != normalizer
        or bank.input_dim != rff_map.input_dim
        or bank.input_dim != swe_map.input_dim
    ):
        raise V05BlindError("canonical bank differs from the fixed maps")
    budgets = _authorized_budget_tuple(
        authorized_budgets, episode_count=bank.episode_count
    )
    authorized_bank = bank.prefix(budgets[-1])
    if not isinstance(private_evidence, Mapping):
        raise V05BlindError("private evidence must be a mapping")
    evidence = dict(private_evidence)
    evidence_digest = sha256_json(evidence)
    public_input = Path(public_parent).expanduser()
    binding_input = Path(private_binding_root).expanduser()
    if public_input.is_symlink() or binding_input.is_symlink():
        raise V05BlindError("private/public roots cannot be symlinks")
    public_root = public_input.resolve()
    binding_root = binding_input.resolve()
    _assert_disjoint(public_root, binding_root)

    resolver = CertificateResolver(certificate_manifest)
    truth = resolver.record_for_anchor(truth_source_anchor_id)
    market_order = tuple(resolver.anchor_ids)
    group = tuple(
        item.source_anchor_id
        for item in certificate_manifest.bindings
        if item.task_id == truth.task_id
        and item.execution_abi_digest == truth.execution_abi_digest
    )
    if (
        isinstance(expected_candidate_count, (bool, np.bool_))
        or not isinstance(expected_candidate_count, (int, np.integer))
        or len(group) != int(expected_candidate_count)
    ):
        raise V05BlindError("truth task/ABI group has the wrong candidate count")
    candidate_mask = tuple(source_id in set(group) for source_id in market_order)
    q_payload = {
        "domain": PUBLIC_QUERY_SCHEMA,
        "private_evidence_digest": evidence_digest,
        "private_canonical_bank_digest": bank.bank_digest,
        "authorized_budgets": list(budgets),
        "truth_source_anchor_id": truth.source_anchor_id,
        "certificate_manifest_digest": certificate_manifest.certificate_manifest_digest,
    }
    opaque_query_id = (
        "q-"
        + hmac.new(
            nonce.encode("utf-8"),
            sha256_json(q_payload).encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
    )
    if truth.source_anchor_id.lower() in str(public_root).lower():
        raise V05BlindError("public root contains the private anchor identity")
    destination = public_root / opaque_query_id
    if destination.exists():
        raise V05BlindError("refusing to overwrite an authorized query")
    public_root.mkdir(parents=True, exist_ok=True)
    if public_root.is_symlink():
        raise V05BlindError("public parent cannot be a symlink")

    arrays_by_method = _view_arrays(authorized_bank, rff_map, swe_map, budgets)
    stage = Path(tempfile.mkdtemp(prefix=f".{opaque_query_id}.", dir=public_root))
    try:
        views: dict[str, Any] = {}
        for method_id in P0_METHOD_IDS:
            arrays = arrays_by_method[method_id]
            artifact_sha = atomic_write_npz(
                stage / AUTHORIZED_VIEW_DIR / f"{method_id}.npz", arrays
            )
            visible_rows = (
                authorized_bank.points.shape[0]
                if method_id in _FULL_BANK_METHODS
                else 0
            )
            views[method_id] = {
                "access_tier": _ACCESS_TIER[method_id],
                "artifact_sha256": artifact_sha,
                "arrays_digest": sha256_ndarrays(arrays),
                "scorer_visible_raw_rows": int(visible_rows),
            }
            if method_id == RFF_KME_NN:
                views[method_id]["map_digest"] = rff_map.map_digest
            elif method_id == SWE_NN:
                views[method_id]["map_digest"] = swe_map.map_digest
        task_scope_digest = sha256_json(
            {
                "task_id": truth.task_id,
                "execution_abi_digest": truth.execution_abi_digest,
            }
        )
        unsigned = {
            "schema": PUBLIC_QUERY_SCHEMA,
            "opaque_query_id": opaque_query_id,
            "certificate_manifest_digest": (
                certificate_manifest.certificate_manifest_digest
            ),
            "probe_protocol_digest": protocol_digest,
            "target_membership_digest": sha256_json(
                {
                    "probe_protocol_digest": protocol_digest,
                    "private_evidence_digest": evidence_digest,
                    "canonical_bank_digest": authorized_bank.bank_digest,
                }
            ),
            "normalization_digest": normalizer,
            "reward_free_bank_sha256": views[EMPIRICAL_MMD_NN]["artifact_sha256"],
            "canonical_query_bank_digest": authorized_bank.bank_digest,
            "canonical_budget_bank_digests": {
                str(budget): authorized_bank.prefix(budget).bank_digest
                for budget in budgets
            },
            "market_order_digest": sha256_json(list(market_order)),
            "task_scope_digest": task_scope_digest,
            "execution_abi_digest": truth.execution_abi_digest,
            "candidate_mask": list(candidate_mask),
            "budget_episodes": list(budgets),
            "authorized_views": views,
        }
        public_manifest = {**unsigned, "manifest_digest": sha256_json(unsigned)}
        _assert_public_safe(
            public_manifest,
            tuple(private_values)
            + (
                truth.source_anchor_id,
                truth.task_id,
                truth.opaque_certified_policy_id,
            ),
        )
        atomic_write_json(stage / PUBLIC_MANIFEST_FILE, public_manifest)
        private_unsigned = {
            "schema": PRIVATE_BINDING_SCHEMA,
            "opaque_query_id": opaque_query_id,
            "source_anchor_id": truth.source_anchor_id,
            "task_id": truth.task_id,
            "opaque_certified_policy_id": truth.opaque_certified_policy_id,
            "public_manifest_digest": public_manifest["manifest_digest"],
            "query_identity_payload": q_payload,
            "private_evidence": evidence,
        }
        private_binding = {
            **private_unsigned,
            "binding_digest": sha256_json(private_unsigned),
        }
        binding_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(binding_root / f"{opaque_query_id}.json", private_binding)
        if destination.exists():
            raise V05BlindError("another writer published this authorized query")
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return load_authorized_query(destination, rff_map=rff_map, swe_map=swe_map)


def prepare_blinded_query(
    *,
    private_collection_root: str | Path,
    public_parent: str | Path,
    private_binding_root: str | Path,
    truth_source_anchor_id: str,
    certificate_manifest: CertifiedPolicyManifest,
    blinding_nonce: str,
    canonicalize_probe: Callable[[RewardFreeProbe], EpisodeBank],
    normalization_digest: str,
    rff_map: RFFMap,
    swe_map: SWEMap,
    authorized_budgets: Sequence[int] = BUDGET_EPISODES,
    expected_candidate_count: int = 5,
) -> AuthorizedQueryViews:
    """Prepare a future confirmatory view from one verified 32-episode bank."""

    if not callable(canonicalize_probe):
        raise V05BlindError("canonicalize_probe must be a privileged callable")
    private_input = Path(private_collection_root).expanduser()
    public_input = Path(public_parent).expanduser()
    binding_input = Path(private_binding_root).expanduser()
    if any(item.is_symlink() for item in (private_input, public_input, binding_input)):
        raise V05BlindError("private/public roots cannot be symlinks")
    private_root = private_input.resolve()
    public_root = public_input.resolve()
    binding_root = binding_input.resolve()
    if not private_root.is_dir():
        raise V05BlindError("private collection root is absent or unsafe")
    _assert_disjoint(private_root, public_root)
    _assert_disjoint(private_root, binding_root)
    collection = load_published_collection(private_root)
    bank = canonicalize_probe(collection.probe)
    if not isinstance(bank, EpisodeBank):
        raise V05BlindError("canonicalize_probe must return an EpisodeBank")
    if (
        bank.episode_count != 32
        or not np.array_equal(bank.episode_offsets, collection.probe.episode_offsets)
        or np.any(np.diff(bank.episode_offsets) != 64)
    ):
        raise V05BlindError("canonical query bank is not the verified 32x64 view")
    private_index = read_json(private_root / "index.json")
    if not isinstance(private_index, Mapping):
        raise V05BlindError("private collection index is malformed")
    return _publish_blinded_bank(
        bank=bank,
        public_parent=public_root,
        private_binding_root=binding_root,
        truth_source_anchor_id=truth_source_anchor_id,
        certificate_manifest=certificate_manifest,
        blinding_nonce=blinding_nonce,
        probe_protocol_digest=collection.probe_protocol_digest,
        normalization_digest=normalization_digest,
        rff_map=rff_map,
        swe_map=swe_map,
        authorized_budgets=authorized_budgets,
        private_evidence={
            "evidence_kind": "verified_private_collection",
            "private_collection_index_digest": _digest(
                private_index.get("index_digest"), "private collection index digest"
            ),
            "private_reward_free_bank_digest": collection.reward_free_bank_digest,
            "private_probe_membership_digest": (
                collection.membership.membership_digest
            ),
            "private_context_id": collection.context_id,
        },
        private_values=(
            collection.context_id,
            private_root.name,
            str(private_root),
        ),
        expected_candidate_count=expected_candidate_count,
    )


def prepare_blinded_episode_bank(
    *,
    episode_bank: EpisodeBank,
    parent_source_role_manifest: Mapping[str, Any],
    expected_source_role_manifest_digest: str,
    parent_asset_sha256: str,
    parent_membership_digest: str,
    probe_protocol_digest: str,
    public_parent: str | Path,
    private_binding_root: str | Path,
    truth_source_anchor_id: str,
    certificate_manifest: CertifiedPolicyManifest,
    blinding_nonce: str,
    normalization_digest: str,
    rff_map: RFFMap,
    swe_map: SWEMap,
    authorized_budgets: Sequence[int],
    expected_candidate_count: int = 5,
) -> AuthorizedQueryViews:
    """Prepare the strictly parent-bound seven-episode held-repeat view."""

    if not isinstance(episode_bank, EpisodeBank):
        raise V05BlindError("held-repeat query must be an EpisodeBank")
    if episode_bank.episode_count != 7 or np.any(
        np.diff(episode_bank.episode_offsets) != 64
    ):
        raise V05BlindError("held-repeat query must contain exactly 7x64 points")
    if not isinstance(parent_source_role_manifest, Mapping):
        raise V05BlindError("parent source role manifest must be a mapping")
    role_manifest = dict(parent_source_role_manifest)
    if set(role_manifest) != {
        "schema",
        "source_count",
        "full_bank_digest",
        "parent_asset_binding_digest",
        "roles",
        "sources",
        "source_role_manifest_digest",
    }:
        raise V05BlindError("parent source role manifest fields differ")
    unsigned_manifest = {
        key: value
        for key, value in role_manifest.items()
        if key != "source_role_manifest_digest"
    }
    role_manifest_digest = _digest(
        role_manifest.get("source_role_manifest_digest"),
        "source role manifest digest",
    )
    expected_role_manifest_digest = _digest(
        expected_source_role_manifest_digest,
        "expected source role manifest digest",
    )
    if (
        role_manifest.get("schema") != SOURCE_ROLE_SCHEMA
        or role_manifest_digest != sha256_json(unsigned_manifest)
        or role_manifest_digest != expected_role_manifest_digest
    ):
        raise V05BlindError("parent source role manifest digest changed")
    source_id = _text(truth_source_anchor_id, "truth_source_anchor_id")
    asset_digest = _digest(parent_asset_sha256, "parent_asset_sha256")
    membership_digest = _digest(parent_membership_digest, "parent_membership_digest")
    try:
        source_row = role_manifest["sources"][source_id]
        repeat_row = source_row["roles"][SOURCE_REPEAT]
        aggregate_repeat = role_manifest["roles"][SOURCE_REPEAT]
    except (KeyError, TypeError) as error:
        raise V05BlindError("held-repeat parent binding is absent") from error
    if not all(
        isinstance(item, Mapping) for item in (source_row, repeat_row, aggregate_repeat)
    ):
        raise V05BlindError("held-repeat parent binding is malformed")
    if (
        set(source_row)
        != {
            "parent_asset_sha256",
            "parent_membership_digest",
            "full_bank_digest",
            "roles",
        }
        or not isinstance(source_row["roles"], Mapping)
        or set(source_row["roles"]) != set(SOURCE_ROLE_SLICES)
        or set(repeat_row) != {"episode_positions", "membership_digest", "bank_digest"}
        or set(aggregate_repeat)
        != {"episode_count", "membership_digest", "bank_digest"}
    ):
        raise V05BlindError("held-repeat parent binding fields differ")
    full_bank_digest = _digest(
        source_row.get("full_bank_digest"), "parent full bank digest"
    )
    repeat_membership_digest = _digest(
        repeat_row.get("membership_digest"), "source repeat membership digest"
    )
    expected_repeat_membership_digest = _role_membership_digest_from_parent(
        source_id,
        full_bank_digest,
        asset_digest,
        membership_digest,
        SOURCE_REPEAT,
        25,
        32,
    )
    held_budgets = _authorized_budget_tuple(
        authorized_budgets, episode_count=episode_bank.episode_count
    )
    if held_budgets != (1, 2, 4):
        raise V05BlindError("held-repeat authorized budgets must equal (1, 2, 4)")
    if (
        source_row.get("parent_asset_sha256") != asset_digest
        or source_row.get("parent_membership_digest") != membership_digest
        or repeat_membership_digest != expected_repeat_membership_digest
        or repeat_row.get("bank_digest") != episode_bank.bank_digest
        or repeat_row.get("episode_positions") != list(range(25, 32))
        or aggregate_repeat.get("episode_count") != 7
    ):
        raise V05BlindError("held-repeat bank differs from its parent source role")
    return _publish_blinded_bank(
        bank=episode_bank,
        public_parent=public_parent,
        private_binding_root=private_binding_root,
        truth_source_anchor_id=source_id,
        certificate_manifest=certificate_manifest,
        blinding_nonce=blinding_nonce,
        probe_protocol_digest=probe_protocol_digest,
        normalization_digest=normalization_digest,
        rff_map=rff_map,
        swe_map=swe_map,
        authorized_budgets=held_budgets,
        private_evidence={
            "evidence_kind": "source_repeat_role",
            "source_role_manifest_digest": role_manifest_digest,
            "parent_asset_sha256": asset_digest,
            "parent_membership_digest": membership_digest,
            "source_repeat_membership_digest": repeat_membership_digest,
            "source_repeat_bank_digest": episode_bank.bank_digest,
        },
        private_values=(source_id,),
        expected_candidate_count=expected_candidate_count,
    )


def load_authorized_query(
    query_root: str | Path, *, rff_map: RFFMap, swe_map: SWEMap
) -> AuthorizedQueryViews:
    """Load only public artifacts; this API has no private-root capability."""

    raw_root = Path(query_root).expanduser()
    if raw_root.is_symlink():
        raise V05BlindError("authorized query root is absent or unsafe")
    root = raw_root.resolve()
    if not root.is_dir():
        raise V05BlindError("authorized query root is absent or unsafe")
    if {item.name for item in root.iterdir()} != {
        PUBLIC_MANIFEST_FILE,
        AUTHORIZED_VIEW_DIR,
    }:
        raise V05BlindError("authorized query root has unexpected entries")
    manifest_path = root / PUBLIC_MANIFEST_FILE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise V05BlindError("authorized query manifest is absent or unsafe")
    manifest_value = read_json(manifest_path)
    if not isinstance(manifest_value, Mapping):
        raise V05BlindError("authorized query manifest is malformed")
    manifest = dict(manifest_value)
    if manifest.get("opaque_query_id") != root.name:
        raise V05BlindError("opaque query directory and manifest differ")
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_digest"
    }
    if manifest.get("manifest_digest") != sha256_json(unsigned):
        raise V05BlindError("authorized query manifest digest changed")
    _assert_public_safe(manifest, ())
    _validate_public_manifest_shape(manifest)
    views = manifest.get("authorized_views")
    assert isinstance(views, Mapping)
    view_root = root / AUTHORIZED_VIEW_DIR
    if view_root.is_symlink() or not view_root.is_dir():
        raise V05BlindError("authorized view directory is absent or unsafe")
    files = {item.name for item in view_root.iterdir()}
    if files != {f"{method_id}.npz" for method_id in P0_METHOD_IDS}:
        raise V05BlindError("authorized view directory has unexpected files")
    loaded: dict[str, dict[str, np.ndarray]] = {}
    expected_arrays = {
        RAW_DELTA_RKME: {"points", "episode_offsets"},
        EMPIRICAL_MMD_NN: {"points", "episode_offsets"},
        SUMMARY_LOGREG: {"episode_summaries"},
        KME_KRR: {"points", "episode_offsets"},
        RFF_KME_NN: {"budget_episodes", "vectors"},
        SWE_NN: {"budget_episodes", "vectors"},
    }
    for method_id in P0_METHOD_IDS:
        path = view_root / f"{method_id}.npz"
        if path.is_symlink() or not path.is_file():
            raise V05BlindError("authorized view artifact is absent or unsafe")
        row = views[method_id]
        if not isinstance(row, Mapping) or sha256_file(path) != row.get(
            "artifact_sha256"
        ):
            raise V05BlindError("authorized view artifact digest changed")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_arrays[method_id]:
                raise V05BlindError("authorized view exposes unexpected arrays")
            loaded[method_id] = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
        if sha256_ndarrays(loaded[method_id]) != row.get("arrays_digest"):
            raise V05BlindError("authorized view array digest changed")
    banks = {
        method_id: EpisodeBank(arrays["points"], arrays["episode_offsets"])
        for method_id, arrays in loaded.items()
        if method_id in _FULL_BANK_METHODS
    }
    authorized_budgets = _authorized_budget_tuple(
        manifest["budget_episodes"],
        episode_count=banks[RAW_DELTA_RKME].episode_count,
    )
    budget_array = np.asarray(authorized_budgets, dtype=np.int64)
    rff_arrays = loaded[RFF_KME_NN]
    swe_arrays = loaded[SWE_NN]
    if (
        views[RFF_KME_NN].get("map_digest") != rff_map.map_digest
        or views[SWE_NN].get("map_digest") != swe_map.map_digest
        or not np.array_equal(rff_arrays["budget_episodes"], budget_array)
        or not np.array_equal(swe_arrays["budget_episodes"], budget_array)
        or rff_arrays["vectors"].shape != (len(authorized_budgets), rff_map.output_dim)
        or swe_arrays["vectors"].shape != (len(authorized_budgets), swe_map.output_dim)
    ):
        raise V05BlindError("fixed-vector budget rows differ")
    rff_specs = {
        budget: RFFSpecification(rff_arrays["vectors"][index], rff_map.map_digest)
        for index, budget in enumerate(authorized_budgets)
    }
    swe_specs = {
        budget: SWESpecification(swe_arrays["vectors"][index], swe_map.map_digest)
        for index, budget in enumerate(authorized_budgets)
    }
    if (
        manifest.get("normalization_digest")
        not in {
            rff_map.normalization_digest,
            swe_map.normalization_digest,
        }
        or rff_map.normalization_digest != swe_map.normalization_digest
    ):
        raise V05BlindError("authorized fixed maps bind another normalizer")
    return AuthorizedQueryViews(
        manifest=manifest,
        banks=banks,
        summary_rows=loaded[SUMMARY_LOGREG]["episode_summaries"],
        rff_specs=rff_specs,
        swe_specs=swe_specs,
    )


def load_private_truth_binding(
    binding_file: str | Path,
    *,
    opaque_query_id: str,
    authorized_query_manifest_digest: str,
    prediction_seal: RankingSeal,
    certificate_manifest: CertifiedPolicyManifest,
    blinding_nonce: str,
) -> TruthBinding:
    """Privileged post-seal loader; scorer code has no reason to call this."""

    opaque_query_id = _text(opaque_query_id, "opaque_query_id")
    if not opaque_query_id.startswith("q-") or len(opaque_query_id) < 22:
        raise V05BlindError("opaque_query_id is not an opaque query")
    try:
        int(opaque_query_id[2:], 16)
    except ValueError as error:
        raise V05BlindError("opaque_query_id is not an opaque query") from error
    expected_public_manifest_digest = _digest(
        authorized_query_manifest_digest,
        "authorized_query_manifest_digest",
    )
    if not isinstance(certificate_manifest, CertifiedPolicyManifest):
        raise V05BlindError("private truth loading requires a certificate manifest")
    if not isinstance(prediction_seal, RankingSeal):
        raise V05BlindError("private truth loading requires a ranking seal")
    sealed_budgets = _HELD_DEVELOPMENT_BUDGETS
    try:
        sealed_payload = prediction_seal.rankings
        verify_ranking_seal(prediction_seal, sealed_payload)
        if (
            not isinstance(sealed_payload, Mapping)
            or set(sealed_payload) != {"schema", "predictions"}
            or sealed_payload.get("schema") != PREDICTION_PAYLOAD_SCHEMA
            or not isinstance(sealed_payload.get("predictions"), list)
        ):
            raise V05BlindError("sealed prediction payload schema differs")
        rows = tuple(
            PredictionRanking.from_dict(item) for item in sealed_payload["predictions"]
        )
        query_rows = tuple(
            item for item in rows if item.opaque_query_id == opaque_query_id
        )
        require_prediction_cell_coverage(
            query_rows,
            expected_method_ids=P0_METHOD_IDS,
            expected_endpoints=(MARKET_30_CERT, TASK_5_CERT),
            expected_budgets=sealed_budgets,
        )
    except V05BlindError:
        raise
    except (
        KeyError,
        TypeError,
        V04AProtocolError,
        V05MetricError,
        ValueError,
    ) as error:
        raise V05BlindError(
            "sealed query rows lack complete P0 prediction coverage"
        ) from error
    sealed_manifest_digests = {
        item.authorized_query_manifest_digest for item in query_rows
    }
    if tuple(sorted({item.budget_episodes for item in query_rows})) != sealed_budgets:
        raise V05BlindError("sealed query rows do not cover development budgets")
    if sealed_manifest_digests != {expected_public_manifest_digest}:
        raise V05BlindError(
            "sealed query rows differ from the authorized public manifest"
        )

    # The private path capability is materialized only after every truth-free
    # seal, cell, query, budget, and public-manifest check has passed.
    path = Path(binding_file).expanduser()
    if path.name != f"{opaque_query_id}.json":
        raise V05BlindError("private truth binding filename differs")
    if path.is_symlink() or not path.is_file():
        raise V05BlindError("private truth binding is absent or unsafe")
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise V05BlindError("private truth binding is malformed")
    binding = dict(value)
    fields = {
        "schema",
        "opaque_query_id",
        "source_anchor_id",
        "task_id",
        "opaque_certified_policy_id",
        "public_manifest_digest",
        "query_identity_payload",
        "private_evidence",
        "binding_digest",
    }
    if set(binding) != fields or binding.get("schema") != PRIVATE_BINDING_SCHEMA:
        raise V05BlindError("private truth binding fields differ")
    unsigned = {key: item for key, item in binding.items() if key != "binding_digest"}
    if binding.get("binding_digest") != sha256_json(unsigned):
        raise V05BlindError("private truth binding digest changed")
    bound_query_id = _text(binding.get("opaque_query_id"), "opaque_query_id")
    if bound_query_id != opaque_query_id:
        raise V05BlindError("private truth binding filename differs")
    evidence = binding.get("private_evidence")
    q_payload = binding.get("query_identity_payload")
    if not isinstance(evidence, Mapping) or not isinstance(q_payload, Mapping):
        raise V05BlindError("private truth evidence is malformed")
    q_fields = {
        "domain",
        "private_evidence_digest",
        "private_canonical_bank_digest",
        "authorized_budgets",
        "truth_source_anchor_id",
        "certificate_manifest_digest",
    }
    if (
        set(q_payload) != q_fields
        or q_payload.get("domain") != PUBLIC_QUERY_SCHEMA
        or q_payload.get("truth_source_anchor_id") != binding["source_anchor_id"]
        or q_payload.get("certificate_manifest_digest")
        != certificate_manifest.certificate_manifest_digest
    ):
        raise V05BlindError("private query identity binding differs")
    if _digest(
        q_payload.get("private_evidence_digest"),
        "private_evidence_digest",
    ) != sha256_json(evidence):
        raise V05BlindError("private query evidence digest changed")
    _digest(
        q_payload.get("private_canonical_bank_digest"),
        "private_canonical_bank_digest",
    )
    nonce = _text(blinding_nonce, "blinding_nonce")
    if len(nonce) < 16:
        raise V05BlindError("blinding_nonce must contain at least 16 characters")
    expected_query_id = (
        "q-"
        + hmac.new(
            nonce.encode("utf-8"),
            sha256_json(q_payload).encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
    )
    if expected_query_id != opaque_query_id:
        raise V05BlindError("opaque query HMAC does not match private truth")
    private_public_manifest_digest = _digest(
        binding.get("public_manifest_digest"), "public_manifest_digest"
    )
    claimed_budgets = _authorized_budget_tuple(
        q_payload.get("authorized_budgets", ()),
        episode_count=max(BUDGET_EPISODES),
    )
    evidence_kind = evidence.get("evidence_kind")
    if evidence_kind == "source_repeat_role":
        if set(evidence) != {
            "evidence_kind",
            "source_role_manifest_digest",
            "parent_asset_sha256",
            "parent_membership_digest",
            "source_repeat_membership_digest",
            "source_repeat_bank_digest",
        }:
            raise V05BlindError("source-repeat private evidence fields differ")
        for field in set(evidence) - {"evidence_kind"}:
            _digest(evidence.get(field), field)
        if claimed_budgets != (1, 2, 4) or q_payload.get(
            "private_canonical_bank_digest"
        ) != evidence.get("source_repeat_bank_digest"):
            raise V05BlindError("source-repeat private evidence binding differs")
    elif evidence_kind == "verified_private_collection":
        if set(evidence) != {
            "evidence_kind",
            "private_collection_index_digest",
            "private_reward_free_bank_digest",
            "private_probe_membership_digest",
            "private_context_id",
        }:
            raise V05BlindError("private collection evidence fields differ")
        for field in (
            "private_collection_index_digest",
            "private_reward_free_bank_digest",
            "private_probe_membership_digest",
        ):
            _digest(evidence.get(field), field)
        _text(evidence.get("private_context_id"), "private_context_id")
    else:
        raise V05BlindError("private evidence kind is unsupported")
    if (
        sealed_budgets != claimed_budgets
        or private_public_manifest_digest != expected_public_manifest_digest
    ):
        raise V05BlindError("sealed query rows differ from the private binding")
    resolver = CertificateResolver(certificate_manifest)
    certificate = resolver.record_for_anchor(binding["source_anchor_id"])
    if (
        binding.get("task_id") != certificate.task_id
        or binding.get("opaque_certified_policy_id")
        != certificate.opaque_certified_policy_id
    ):
        raise V05BlindError("private truth differs from the certificate")
    return TruthBinding(
        opaque_query_id=opaque_query_id,
        source_anchor_id=certificate.source_anchor_id,
        task_id=certificate.task_id,
        opaque_certified_policy_id=certificate.opaque_certified_policy_id,
        authorized_query_manifest_digest=private_public_manifest_digest,
        prediction_seal_digest=prediction_seal.rankings_digest,
    )


__all__ = [
    "AUTHORIZED_VIEW_DIR",
    "PUBLIC_MANIFEST_FILE",
    "SOURCE_REPEAT",
    "SOURCE_ROLE_SLICES",
    "SOURCE_TRAIN",
    "SOURCE_VALIDATION",
    "AuthorizedQueryViews",
    "SourceRoleProjection",
    "V05BlindError",
    "load_authorized_query",
    "load_private_truth_binding",
    "prepare_blinded_episode_bank",
    "prepare_blinded_query",
    "project_verified_source_banks",
]
