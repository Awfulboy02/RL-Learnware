"""Thin orchestration for the six matched v0.5 P0 classifiers.

Collection, private label joins, and artifact publication live outside this
module.  The runner accepts only canonical reward-free episode banks, binds all
methods to one source-evidence summary, scores nested target prefixes, and uses
the existing v0.4a ranking seal before any evaluation code can run.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from policy_learnware_v0.hashing import sha256_json, sha256_ndarrays
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v04a.protocol import (
    BUDGET_EPISODES,
    BudgetLedger,
    RankingSeal,
    seal_rankings,
    tie_break_key,
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
)
from policy_learnware_v0.v05.metrics import (
    MARKET_30_CERT,
    TASK_5_CERT,
    PredictionRanking,
    prediction_payload,
    require_prediction_cell_coverage,
)
from policy_learnware_v0.v05.specifications import RFFMap, SWEMap
from server.repro_fpo_ppo_v05.blind_query_bank import (
    AuthorizedQueryViews,
    project_verified_source_banks,
)


Q0_COMMON_GAUSSIAN_OPEN_LOOP = "Q0_COMMON_GAUSSIAN_OPEN_LOOP"
P1_STATUS = "DEFERRED_NOT_IMPLEMENTED"
SCORER_SOURCE_FIT_SCHEMA = "policy-learnware.v05-scorer-source-fit.v1"
_OPAQUE_QUERY_ID = re.compile(r"^q-[0-9a-f]{20,64}$")


class V05RunnerError(ValueError):
    """A matched-source binding, target view, or score cell is invalid."""


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
            train, labels, validation, l2_grid=logreg_l2_grid
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
) -> tuple[tuple[PredictionRanking, ...], tuple[dict[str, Any], ...]]:
    """Score only verified public views, then apply masks and break ties."""

    if not isinstance(panel, P0Panel) or not isinstance(
        query_views, AuthorizedQueryViews
    ):
        raise V05RunnerError("panel/query types are invalid")
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
        full_scores = {
            RAW_DELTA_RKME: panel.raw.score(raw_query),
            EMPIRICAL_MMD_NN: panel.empirical_mmd.score(empirical_query),
            SUMMARY_LOGREG: panel.summary_logreg.score_summaries(
                query_views.summaries_for(ledger.budget_episodes)
            ),
            KME_KRR: panel.kme_krr.score(krr_query),
            RFF_KME_NN: panel.rff.score_specification(
                query_views.rff_specs[ledger.budget_episodes]
            ),
            SWE_NN: panel.swe.score_specification(
                query_views.swe_specs[ledger.budget_episodes]
            ),
        }
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
                    }
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


__all__ = [
    "P0Panel",
    "P1_STATUS",
    "Q0_COMMON_GAUSSIAN_OPEN_LOOP",
    "V05RunnerError",
    "fit_p0_panel",
    "score_query",
    "seal_prediction_rows",
]
