"""Minimal v0.31 Raw-RKME transition-view experiment.

This runner deliberately sits beside, rather than inside, the frozen v0.3
39-cell Signal Atlas.  It reuses the existing transition banks, anonymous
policy market and development policy-return oracle, then changes exactly one
thing: the raw transition measurement presented to the Gaussian RKME.

No encoder, policy rollout, target-return fitting, or confirmatory artifact is
loaded.  Every view uses the same production B3b protocol: one source-balanced
bandwidth, source Reduced RKME, query Empirical KME, and nearest MMD ranking.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from policy_learnware_v0.hashing import (
    sha256_file,
    sha256_json,
    sha256_ndarrays,
)
from policy_learnware_v0.io import atomic_write_bytes, atomic_write_json
from policy_learnware_v0.rkme.empirical import EmpiricalKME
from policy_learnware_v0.rkme.gaussian import calibrate_bandwidth
from policy_learnware_v0.rkme.reducer import ReducedRKME, ReducerConfig, reduce_kme
from policy_learnware_v0.v03.signal_controls import RewardFreeShuffledNextSpec
from policy_learnware_v0.v03.signal_metrics import SignalDistanceRow, SignalMetricRecord
from policy_learnware_v0.v03.transition_views import (
    V_DELTA_ONLY,
    V_FULL_LEGACY,
    V_NO_MASK,
    V_REWARD_FREE_TRANSITION,
    V_STATE_ACTION,
    V_STATE_ONLY,
    TransitionBank,
    apply_transition_view,
)
from server.repro_fpo_ppo_v03.development_baseline_runner import (
    _context_rows,
    _distance,
    _empirical,
    _json,
    _market,
    _publish,
    _rank_rows,
)
from server.repro_fpo_ppo_v03.signal_bank_runner import (
    _interpolation_metrics,
    prepare_signal_inputs,
)


SCHEMA = "policy-learnware.v031-raw-transition-controls.v0"
C_RF_SHUFFLED_NEXT = "C_RF_SHUFFLED_NEXT"
C_ACTION_VALUES_ONLY = "C_ACTION_VALUES_ONLY"

VIEWS = (
    V_FULL_LEGACY,
    V_NO_MASK,
    V_REWARD_FREE_TRANSITION,
    C_RF_SHUFFLED_NEXT,
    V_DELTA_ONLY,
    C_ACTION_VALUES_ONLY,
    V_STATE_ACTION,
    V_STATE_ONLY,
)

VIEW_SLUG = {
    V_FULL_LEGACY: "full",
    V_NO_MASK: "no_mask",
    V_REWARD_FREE_TRANSITION: "reward_free",
    C_RF_SHUFFLED_NEXT: "rf_shuffled_next",
    V_DELTA_ONLY: "delta_action",
    C_ACTION_VALUES_ONLY: "action_values_only",
    V_STATE_ACTION: "state_action",
    V_STATE_ONLY: "state_only",
}

VIEW_LABEL = {
    V_FULL_LEGACY: "FULL",
    V_NO_MASK: "No mask",
    V_REWARD_FREE_TRANSITION: "Reward-free transition",
    C_RF_SHUFFLED_NEXT: "Reward-free shuffled-next",
    V_DELTA_ONLY: "Delta + action",
    C_ACTION_VALUES_ONLY: "Action values only",
    V_STATE_ACTION: "State + action",
    V_STATE_ONLY: "State only",
}

VIEW_CHANNELS = {
    V_FULL_LEGACY: "o, masks, a, r, o'",
    V_NO_MASK: "o, a, r, o'",
    V_REWARD_FREE_TRANSITION: "o, a, o'",
    C_RF_SHUFFLED_NEXT: "o, a, perm(o')",
    V_DELTA_ONLY: "o'-o, a",
    C_ACTION_VALUES_ONLY: "a",
    V_STATE_ACTION: "o, a",
    V_STATE_ONLY: "o",
}

VIEW_ROLE = {
    V_FULL_LEGACY: "paper operator / all registered raw channels",
    V_NO_MASK: "mask control",
    V_REWARD_FREE_TRANSITION: "reward-channel control",
    C_RF_SHUFFLED_NEXT: "transition-pairing control",
    V_DELTA_ONLY: "dynamics coordinate",
    C_ACTION_VALUES_ONLY: "clean delta reference",
    V_STATE_ACTION: "next-state control",
    V_STATE_ONLY: "occupancy control",
}

PAIRED_CONTRASTS = (
    ("add masks", V_FULL_LEGACY, V_NO_MASK),
    ("add reward", V_NO_MASK, V_REWARD_FREE_TRANSITION),
    ("preserve next-state pairing", V_REWARD_FREE_TRANSITION, C_RF_SHUFFLED_NEXT),
    ("add next state", V_REWARD_FREE_TRANSITION, V_STATE_ACTION),
    ("add state delta", V_DELTA_ONLY, C_ACTION_VALUES_ONLY),
)


class V031RawTransitionError(RuntimeError):
    pass


def _publish_text(path: Path, text: str, *, resume: bool) -> str:
    data = text.encode("utf-8")
    if path.exists():
        if not resume or not path.is_file() or path.read_bytes() != data:
            raise V031RawTransitionError(f"existing artifact differs: {path}")
        return sha256_file(path)
    try:
        return atomic_write_bytes(path, data)
    except FileExistsError:
        if resume and path.is_file() and path.read_bytes() == data:
            return sha256_file(path)
        raise


def _progress(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_json(path, value, overwrite=True)


def _view_points(
    bank: TransitionBank,
    view_id: str,
    *,
    rf_seed: int,
) -> tuple[np.ndarray, str, Mapping[str, Any] | None]:
    if view_id == C_RF_SHUFFLED_NEXT:
        result = RewardFreeShuffledNextSpec(seed=rf_seed).apply(bank)
        audit = {
            **result.marginal_audit.__dict__,
            "passed": result.marginal_audit.passed,
            "audit_digest": result.marginal_audit.audit_digest,
            "dataset_digest": result.dataset_digest,
        }
        return (
            np.asarray(result.feature_matrix, dtype=np.float64),
            str(result.transform_digest),
            audit,
        )
    if view_id == C_ACTION_VALUES_ONLY:
        transform = sha256_json(
            {
                "control_id": C_ACTION_VALUES_ONLY,
                "channels": ["action"],
                "purpose": "strict_delta_reference",
            }
        )
        return np.asarray(bank.action, dtype=np.float64), transform, None
    result = apply_transition_view(bank, view_id)
    return np.asarray(result.feature_matrix, dtype=np.float64), result.spec.digest, None


def _dataset_digest(points: np.ndarray, offsets: np.ndarray) -> str:
    return sha256_ndarrays(
        {
            "points": np.asarray(points, dtype=np.float64),
            "episode_offsets": np.asarray(offsets, dtype=np.int64),
        }
    )


def _load_or_build_source(
    path: Path,
    *,
    points: np.ndarray,
    offsets: np.ndarray,
    bandwidth: float,
    protocol_id: str,
    dataset_digest: str,
    task_id: str,
    reducer: ReducerConfig,
    backend: str,
    block_size: int,
    resume: bool,
) -> ReducedRKME:
    if path.exists():
        if not resume:
            raise V031RawTransitionError(f"source asset exists; pass --resume: {path}")
        value = ReducedRKME.load_npz(path)
        if (
            value.protocol_id != protocol_id
            or value.source_dataset_digest != dataset_digest
            or not np.isclose(value.bandwidth, bandwidth, rtol=0.0, atol=1.0e-12)
        ):
            raise V031RawTransitionError(f"source resume binding differs: {path}")
        return value
    empirical = _empirical(
        points,
        offsets,
        bandwidth=bandwidth,
        protocol_id=protocol_id,
        dataset_digest=dataset_digest,
        task=task_id,
        backend=backend,
        block_size=block_size,
    )
    value = reduce_kme(empirical, reducer)
    value.save_npz(path)
    return value


def _load_or_build_query(
    path: Path,
    *,
    points: np.ndarray,
    offsets: np.ndarray,
    bandwidth: float,
    protocol_id: str,
    dataset_digest: str,
    task_id: str,
    backend: str,
    block_size: int,
    resume: bool,
) -> EmpiricalKME:
    if path.exists():
        if not resume:
            raise V031RawTransitionError(f"query asset exists; pass --resume: {path}")
        value = EmpiricalKME.load_npz(path)
        if (
            value.protocol_id != protocol_id
            or value.dataset_digest != dataset_digest
            or not np.isclose(value.bandwidth, bandwidth, rtol=0.0, atol=1.0e-12)
        ):
            raise V031RawTransitionError(f"query resume binding differs: {path}")
        return value
    value = _empirical(
        points,
        offsets,
        bandwidth=bandwidth,
        protocol_id=protocol_id,
        dataset_digest=dataset_digest,
        task=task_id,
        backend=backend,
        block_size=block_size,
    )
    value.save_npz(path)
    return value


def _signal_row(
    query_index: int,
    source_index: int,
    distance: float,
    inputs: Any,
) -> SignalDistanceRow:
    query_receipt = inputs.receipts[query_index]
    source_receipt = inputs.receipts[source_index]
    query = inputs.identities[query_index]
    source = inputs.identities[source_index]
    return SignalDistanceRow(
        query_bank_id=query.bank_id,
        source_bank_id=source.bank_id,
        query_receipt_digest=str(query_receipt.receipt_digest),
        source_receipt_digest=str(source_receipt.receipt_digest),
        query_raw_dataset_digest=query_receipt.raw_dataset_digest,
        source_raw_dataset_digest=source_receipt.raw_dataset_digest,
        query_task_id=query.task_private_id,
        source_task_id=source.task_private_id,
        query_context_id=query.context_id,
        source_context_id=source.context_id,
        query_embodiment_id=query.embodiment_id,
        source_embodiment_id=source.embodiment_id,
        query_abi_contract_id=query.abi_contract_id,
        source_abi_contract_id=source.abi_contract_id,
        query_goal_contract_id=query.goal_contract_id,
        source_goal_contract_id=source.goal_contract_id,
        query_dynamics_context_id=query.dynamics_context_id,
        source_dynamics_context_id=source.dynamics_context_id,
        query_equivalence_class_id=query.equivalence_class_id,
        source_equivalence_class_id=source.equivalence_class_id,
        distance=distance,
    )


def _metric_record(
    *,
    view_id: str,
    transform_digest: str,
    rows: Sequence[SignalDistanceRow],
    expected: Mapping[str, str],
    inputs: Any,
) -> SignalMetricRecord:
    return SignalMetricRecord(
        cell_id=f"V031_RAW::{view_id}",
        view_or_condition_id=view_id,
        representation_id="R0_RAW_IDENTITY",
        representation_coordinate_digest=sha256_json(
            {
                "view_id": view_id,
                "transform_digest": transform_digest,
                "canonicalizer": inputs.canonicalizer.canonicalizer_digest,
            }
        ),
        representation_seed=None,
        source_index_digest=sha256_json(
            [
                {
                    "bank_id": item.bank_id,
                    "receipt_digest": item.receipt_digest,
                }
                for item in inputs.identities[: inputs.source_count]
            ]
        ),
        query_manifest_digest=sha256_json(
            {
                "query_ids": sorted(expected),
                "expected": dict(sorted(expected.items())),
            }
        ),
        rows=tuple(rows),
        expected_source_by_query=expected,
    )


def _repeat_summary(record: SignalMetricRecord) -> Mapping[str, Any]:
    rows = record.rows
    details: list[dict[str, Any]] = []
    for query_id in sorted({row.query_bank_id for row in rows}):
        group = [row for row in rows if row.query_bank_id == query_id]
        direct = [
            row
            for row in group
            if row.query_context_id == row.source_context_id
            and row.query_task_id == row.source_task_id
            and row.query_dynamics_context_id == row.source_dynamics_context_id
        ]
        if len(direct) != 1:
            raise V031RawTransitionError(f"{query_id}: exact-repeat pair is not unique")
        exact = direct[0]
        between = [
            row.distance
            for row in group
            if row.source_task_id == exact.query_task_id
            and row.source_goal_contract_id == exact.query_goal_contract_id
            and row.source_embodiment_id == exact.query_embodiment_id
            and row.source_abi_contract_id == exact.query_abi_contract_id
            and row.source_dynamics_context_id != exact.query_dynamics_context_id
        ]
        between_mean = None if not between else float(np.mean(between))
        ratio = (
            None
            if between_mean is None or exact.distance == 0.0
            else between_mean / exact.distance
        )
        details.append(
            {
                "query_bank_id": query_id,
                "source_bank_id": exact.source_bank_id,
                "direct_repeat_mmd": exact.distance,
                "between_dynamics_mean_mmd": between_mean,
                "between_repeat_ratio": ratio,
                "ratio_kind": (
                    "NO_BETWEEN_DYNAMICS"
                    if between_mean is None
                    else "INFINITE_ZERO_NOISE_FLOOR"
                    if exact.distance == 0.0
                    else "FINITE"
                ),
            }
        )
    ratios = [float(row["between_repeat_ratio"]) for row in details if row["between_repeat_ratio"] is not None]
    return {
        "row_count": len(details),
        "finite_ratio_count": len(ratios),
        "mean_between_repeat_ratio": None if not ratios else float(np.mean(ratios)),
        "probability_ratio_gt_1": None if not ratios else float(np.mean(np.asarray(ratios) > 1.0)),
        "rows": details,
    }


def _oracle_utility(
    *,
    oracle_root: Path,
    ranking: Mapping[str, Any],
    context: Mapping[str, Any],
    expected_opaque: str | None,
) -> Mapping[str, Any]:
    values: dict[str, float] = {}
    for item in ranking["rows"]:
        opaque_id = str(item["opaque_learnware_id"])
        record = _json(oracle_root / str(context["context_id"]) / f"{opaque_id}.json")
        if record.get("status") == "OK":
            values[opaque_id] = float(record["normalized_mean_return"])
        elif record.get("status") != "ABI_FLOOR_NA":
            raise V031RawTransitionError(
                f"oracle is incomplete for {context['context_id']} / {opaque_id}"
            )
    if len(values) != 5:
        raise V031RawTransitionError(
            f"{context['context_id']}: expected five compatible oracle policies"
        )
    ranked = [str(item["opaque_learnware_id"]) for item in ranking["rows"]]
    selected = ranked[0]
    best_return = max(values.values())
    best_ids = sorted(
        key
        for key, value in values.items()
        if np.isclose(value, best_return, rtol=0.0, atol=1.0e-12)
    )
    best_id = best_ids[0]
    selected_return = values.get(selected, 0.0)
    return {
        "context_id": context["context_id"],
        "regime": "source" if context["role"] == "source" else "development",
        "task_id": context["task_id"],
        "selected_opaque_learnware_id": selected,
        "selected_return": selected_return,
        "oracle_best_id": best_id,
        "oracle_best_return": best_return,
        "normalized_regret": best_return - selected_return,
        "task_compatible": selected in values,
        "exact_anchor": None if expected_opaque is None else selected == expected_opaque,
        "top3_oracle_coverage": any(key in ranked[:3] for key in best_ids),
    }


def _aggregate_utility(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    def subset(regime: str | None) -> list[Mapping[str, Any]]:
        return [row for row in rows if regime is None or row["regime"] == regime]

    result: dict[str, Any] = {}
    for name, regime in (("source", "source"), ("development", "development"), ("all", None)):
        selected = subset(regime)
        result[name] = {
            "context_count": len(selected),
            "mean_selected_return": float(np.mean([row["selected_return"] for row in selected])),
            "mean_normalized_regret": float(np.mean([row["normalized_regret"] for row in selected])),
            "task_compatibility": float(np.mean([row["task_compatible"] for row in selected])),
            "top3_oracle_coverage": float(np.mean([row["top3_oracle_coverage"] for row in selected])),
            "exact_anchor_accuracy": (
                None
                if not any(row["exact_anchor"] is not None for row in selected)
                else float(np.mean([row["exact_anchor"] for row in selected if row["exact_anchor"] is not None]))
            ),
        }
    return result


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )


def _process_view(args: argparse.Namespace, inputs: Any, context_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    view_id = args.current_view
    slug = VIEW_SLUG[view_id]
    root = args.output_dir / "views" / slug
    source_count = inputs.source_count
    query_count = inputs.query_count
    repeat_start = source_count + query_count
    rf_seed = int(inputs.condition_plan.rf_shuffled_next_seed)

    observed_transforms: set[str] = set()
    marginal_audits: list[Mapping[str, Any]] = []
    point_cache: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
    for index, bank in enumerate(inputs.banks):
        points, transform, audit = _view_points(bank, view_id, rf_seed=rf_seed)
        offsets = np.asarray(bank.episode_offsets, dtype=np.int64)
        digest = _dataset_digest(points, offsets)
        point_cache[index] = (points, offsets, digest)
        observed_transforms.add(transform)
        if audit is not None:
            marginal_audits.append(
                {"bank_id": inputs.identities[index].bank_id, **dict(audit)}
            )
    if len(observed_transforms) != 1:
        raise V031RawTransitionError(f"{view_id}: transform identity drifted across banks")
    transform_digest = next(iter(observed_transforms))
    normalized_by_bank = {str(row["bank_id"]): row for row in context_rows}
    market = args.market
    source_opaque: dict[int, str] = {}
    for index in range(source_count):
        bank_id = inputs.identities[index].bank_id
        row = normalized_by_bank[bank_id]
        anchor = str(row["source_anchor_id"])
        source_opaque[index] = market.anchor_to_opaque_learnware_id[anchor]
    if set(source_opaque.values()) != set(market.entries):
        raise V031RawTransitionError("source views and public market differ")

    # Match the frozen B3b production sampler exactly.  The source-balanced
    # sampler sorts mapping keys before drawing task/episode/transition; B3b
    # used anonymous learnware IDs, so bank IDs would change the seeded draw.
    sources_for_bandwidth = {
        source_opaque[index]: SimpleNamespace(
            points=point_cache[index][0], episode_offsets=point_cache[index][1]
        )
        for index in range(source_count)
    }
    bandwidth = calibrate_bandwidth(
        sources_for_bandwidth,
        calibration_pairs=args.calibration_pairs,
        seed=args.bandwidth_seed,
    )
    protocol_id = sha256_json(
        {
            "operator": "V031_RAW_RKME",
            "view_id": view_id,
            "transform_digest": transform_digest,
            "canonicalizer": inputs.canonicalizer.canonicalizer_digest,
            "kernel": "gaussian",
            "weighting": "episode_balanced",
            "source_mode": "SOURCE_REDUCED",
            "query_mode": "QUERY_EMPIRICAL",
        }
    )
    view_config = {
        "schema": SCHEMA,
        "view_id": view_id,
        "view_slug": slug,
        "label": VIEW_LABEL[view_id],
        "channels": VIEW_CHANNELS[view_id],
        "scientific_role": VIEW_ROLE[view_id],
        "transform_digest": transform_digest,
        "canonicalizer_digest": inputs.canonicalizer.canonicalizer_digest,
        "protocol_id": protocol_id,
        "bandwidth": bandwidth,
        "bandwidth_source_key": "opaque_learnware_id",
        "feature_width": int(point_cache[0][0].shape[1]),
        "source_count": source_count,
        "query_count": query_count,
        "repeat_count": inputs.repeat_count,
        "rf_shuffled_next_seed": rf_seed if view_id == C_RF_SHUFFLED_NEXT else None,
    }
    _publish(root / "config.json", view_config, resume=args.resume)
    if marginal_audits:
        _publish(
            root / "marginal_audits.json",
            {
                "schema": SCHEMA,
                "view_id": view_id,
                "all_passed": all(bool(row["passed"]) for row in marginal_audits),
                "bank_count": len(marginal_audits),
                "rows": marginal_audits,
            },
            resume=args.resume,
        )

    reducer = ReducerConfig(
        support_budget=args.support_budget,
        support_steps=args.support_steps,
        kmeans_steps=args.kmeans_steps,
        optimizer_backend=args.backend,
        negative_tolerance=args.negative_tolerance,
    )
    reduced: dict[int, ReducedRKME] = {}
    for index in range(source_count):
        points, offsets, digest = point_cache[index]
        reduced[index] = _load_or_build_source(
            root / "source" / f"{source_opaque[index]}.npz",
            points=points,
            offsets=offsets,
            bandwidth=bandwidth,
            protocol_id=protocol_id,
            dataset_digest=digest,
            task_id=inputs.identities[index].task_private_id,
            reducer=reducer,
            backend=args.backend,
            block_size=args.block_size,
            resume=args.resume,
        )
        _progress(
            args.output_dir / "progress.json",
            {"schema": SCHEMA, "stage": "source_reduction", "view_id": view_id, "completed": index + 1, "total": source_count},
        )

    tie = {key: market.entries[key].tie_break_token for key in market.entries}
    dev_signal_rows: list[SignalDistanceRow] = []
    repeat_signal_rows: list[SignalDistanceRow] = []
    flat_distances: list[Mapping[str, Any]] = []
    utility_rows: list[Mapping[str, Any]] = []
    query_items: list[tuple[int, Mapping[str, Any], str, str | None]] = []
    for offset, row in enumerate(inputs.query_rows):
        query_items.append((source_count + offset, normalized_by_bank[str(row["bank_id"])], "development", None))
    for offset, row in enumerate(inputs.source_rows):
        context = normalized_by_bank[str(row["bank_id"])]
        expected = market.anchor_to_opaque_learnware_id[str(context["source_anchor_id"])]
        query_items.append((repeat_start + offset, context, "source", expected))

    for completed, (query_index, context, regime, expected_opaque) in enumerate(query_items, 1):
        points, offsets, digest = point_cache[query_index]
        query = _load_or_build_query(
            root / "query" / f"{context['context_id']}.npz",
            points=points,
            offsets=offsets,
            bandwidth=bandwidth,
            protocol_id=protocol_id,
            dataset_digest=digest,
            task_id=inputs.identities[query_index].task_private_id,
            backend=args.backend,
            block_size=args.block_size,
            resume=args.resume,
        )
        distances = {
            source_opaque[index]: _distance(
                query, reduced[index], backend=args.backend, block_size=args.block_size
            )
            for index in range(source_count)
        }
        ranking_rows = _rank_rows(
            {key: -value for key, value in distances.items()}, tie, distances
        )
        ranking = {
            "schema": SCHEMA,
            "stage": "PUBLIC_RANKING",
            "method": "RAW_RKME",
            "view_id": view_id,
            "context_id": context["context_id"],
            "context_role": regime,
            "query_bank_id": inputs.identities[query_index].bank_id,
            "selected_opaque_learnware_id": ranking_rows[0]["opaque_learnware_id"],
            "rows": ranking_rows,
        }
        _publish(
            root / "rankings" / f"{context['context_id']}.json",
            ranking,
            resume=args.resume,
        )
        utility_rows.append(
            _oracle_utility(
                oracle_root=args.oracle_root,
                ranking=ranking,
                context=context,
                expected_opaque=expected_opaque,
            )
        )
        for source_index in range(source_count):
            opaque_id = source_opaque[source_index]
            distance = distances[opaque_id]
            signal = _signal_row(query_index, source_index, distance, inputs)
            (dev_signal_rows if regime == "development" else repeat_signal_rows).append(signal)
            flat_distances.append(
                {
                    "view_id": view_id,
                    "regime": regime,
                    "context_id": context["context_id"],
                    "query_bank_id": inputs.identities[query_index].bank_id,
                    "source_bank_id": inputs.identities[source_index].bank_id,
                    "opaque_learnware_id": opaque_id,
                    "distance": distance,
                }
            )
        _progress(
            args.output_dir / "progress.json",
            {"schema": SCHEMA, "stage": "query_distance", "view_id": view_id, "completed": completed, "total": len(query_items)},
        )

    dev_expected = {
        str(row["bank_id"]): str(row["expected_source_bank_id"])
        for row in inputs.query_rows
    }
    repeat_expected = {
        f"{row['bank_id']}-repeat": str(row["bank_id"]) for row in inputs.source_rows
    }
    dev_record = _metric_record(
        view_id=view_id,
        transform_digest=transform_digest,
        rows=dev_signal_rows,
        expected=dev_expected,
        inputs=inputs,
    )
    repeat_record = _metric_record(
        view_id=view_id,
        transform_digest=transform_digest,
        rows=repeat_signal_rows,
        expected=repeat_expected,
        inputs=inputs,
    )
    interpolation_rows, interpolation_metrics = _interpolation_metrics(
        dev_record, inputs.source_rows, inputs.query_rows
    )
    repeat = _repeat_summary(repeat_record)
    metrics = {
        "schema": SCHEMA,
        "formal": False,
        "scope": "30 source exact recurrence + 24 frozen development; no confirmatory/extrapolation",
        "view": view_config,
        "semantic": {
            **dict(dev_record.metric_values),
            **dict(interpolation_metrics),
            "repeat_exact_source_top1": repeat_record.metric_values["exact_source_top1"],
            "repeat_exact_source_mrr": repeat_record.metric_values["exact_source_mrr"],
            "repeat_mean_between_repeat_ratio": repeat["mean_between_repeat_ratio"],
            "repeat_probability_ratio_gt_1": repeat["probability_ratio_gt_1"],
        },
        "policy_utility": _aggregate_utility(utility_rows),
        "utility_rows": utility_rows,
        "repeat_rows": repeat["rows"],
        "interpolation_rows": interpolation_rows,
    }
    _publish_text(root / "distances.jsonl", _jsonl(flat_distances), resume=args.resume)
    _publish(root / "metrics.json", metrics, resume=args.resume)
    return metrics


def _fmt(value: Any, digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _paired_rows(metrics: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = []
    for label, candidate, reference in PAIRED_CONTRASTS:
        left = metrics[candidate]
        right = metrics[reference]
        by_context_left = {row["context_id"]: row for row in left["utility_rows"]}
        by_context_right = {row["context_id"]: row for row in right["utility_rows"]}
        win = tie = loss = 0
        for context_id in sorted(by_context_left):
            delta = (
                float(by_context_left[context_id]["normalized_regret"])
                - float(by_context_right[context_id]["normalized_regret"])
            )
            if delta < -1.0e-12:
                win += 1
            elif delta > 1.0e-12:
                loss += 1
            else:
                tie += 1
        result.append(
            {
                "factor": label,
                "candidate": candidate,
                "reference": reference,
                "all_regret_benefit": (
                    right["policy_utility"]["all"]["mean_normalized_regret"]
                    - left["policy_utility"]["all"]["mean_normalized_regret"]
                ),
                "axis_bracket_benefit": (
                    left["semantic"].get("interpolation_axis_bracket_top2_rate", float("nan"))
                    - right["semantic"].get("interpolation_axis_bracket_top2_rate", float("nan"))
                ),
                "repeat_ratio_benefit": (
                    left["semantic"].get("repeat_mean_between_repeat_ratio", float("nan"))
                    - right["semantic"].get("repeat_mean_between_repeat_ratio", float("nan"))
                ),
                "wins": win,
                "ties": tie,
                "losses": loss,
            }
        )
    return result


def _build_table(args: argparse.Namespace) -> Mapping[str, Any] | None:
    paths = {view: args.output_dir / "views" / VIEW_SLUG[view] / "metrics.json" for view in VIEWS}
    if not all(path.is_file() for path in paths.values()):
        return None
    metrics = {view: _json(path) for view, path in paths.items()}
    rows = []
    for view in VIEWS:
        value = metrics[view]
        semantic = value["semantic"]
        utility = value["policy_utility"]
        task = semantic.get("task_top1")
        abi = semantic.get("abi_top1")
        task_abi = task if task == abi else None
        rows.append(
            {
                "view_id": view,
                "view": VIEW_LABEL[view],
                "channels": VIEW_CHANNELS[view],
                "isolates_or_role": VIEW_ROLE[view],
                "task_abi_top1": task_abi,
                "task_top1": task,
                "abi_top1": abi,
                "repeat_exact_source_top1": semantic.get("repeat_exact_source_top1"),
                "nearest_bracket": semantic.get("interpolation_nearest_bracket_rate"),
                "axis_bracket_at_2": semantic.get("interpolation_axis_bracket_top2_rate"),
                "rho_log_factor": semantic.get("interpolation_log_factor_spearman"),
                "repeat_separation": semantic.get("repeat_mean_between_repeat_ratio"),
                "p_ratio_gt_1": semantic.get("repeat_probability_ratio_gt_1"),
                "source_regret": utility["source"]["mean_normalized_regret"],
                "dev_regret": utility["development"]["mean_normalized_regret"],
                "all_regret": utility["all"]["mean_normalized_regret"],
                "task_compatible": utility["all"]["task_compatibility"],
                "exact_anchor": utility["source"]["exact_anchor_accuracy"],
                "top3_oracle": utility["all"]["top3_oracle_coverage"],
            }
        )
    contrasts = _paired_rows(metrics)
    summary = {
        "schema": SCHEMA,
        "formal": False,
        "table_id": "V031_RESET_TABLE_1",
        "fixed_operator": "R0 raw identity -> source Reduced RKME / query Empirical KME -> nearest Gaussian MMD",
        "rows": rows,
        "paired_contrasts": contrasts,
    }
    _publish(args.output_dir / "table1_reset.json", summary, resume=args.resume)

    fields = list(rows[0])
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _publish_text(args.output_dir / "table1_reset.csv", stream.getvalue(), resume=args.resume)

    header = (
        "| View | Channels | Role / isolated factor | Task/ABI Top-1 | Exact-source Top-1 | "
        "Axis bracket@2 | ρ(log-factor) | Repeat separation | Source regret | "
        "Dev regret | All regret | Task-compatible |\n"
    )
    separator = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = "".join(
        "| {view} | `{channels}` | {isolates_or_role} | {task_abi_top1} | {repeat_exact_source_top1} | "
        "{axis_bracket_at_2} | {rho_log_factor} | {repeat_separation} | "
        "{source_regret} | {dev_regret} | {all_regret} | {task_compatible} |\n".format(
            **{
                key: (
                    _fmt(value)
                    if key not in {"view_id", "view", "channels", "isolates_or_role"}
                    else value
                )
                for key, value in row.items()
            }
        )
        for row in rows
    )
    contrast_header = (
        "\n### Paired controls\n\n"
        "Each benefit is oriented so positive favors the candidate (the richer member of the pair).\n\n"
        "| Factor | Candidate | Reference | All-regret benefit | Axis-bracket benefit | Repeat-ratio benefit | W/T/L |\n"
        "|---|---|---|---:|---:|---:|---:|\n"
    )
    contrast_body = "".join(
        f"| {row['factor']} | `{row['candidate']}` | `{row['reference']}` | "
        f"{_fmt(row['all_regret_benefit'], 4)} | {_fmt(row['axis_bracket_benefit'], 4)} | "
        f"{_fmt(row['repeat_ratio_benefit'], 4)} | {row['wins']}/{row['ties']}/{row['losses']} |\n"
        for row in contrasts
    )
    note = (
        "\nAll rows fix the Raw-RKME operator and change only the transition measurement. "
        "`C_ACTION_VALUES_ONLY=(a)` is runner-local so the Delta comparison does not inherit "
        "the registered `V_ACTION_ONLY=(a, action_mask)` confound. Development only; no "
        "confirmatory or extrapolation oracle was read.\n"
    )
    _publish_text(
        args.output_dir / "table1_reset.md",
        "# v0.31 reset Table 1 — Raw-RKME transition controls\n\n" + header + separator + body + contrast_header + contrast_body + note,
        resume=args.resume,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal v0.31 Raw-RKME transition controls")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-index", type=Path, required=True)
    parser.add_argument("--public-policy-market", type=Path, required=True)
    parser.add_argument("--deployment-private-registry", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--support-budget", type=int, default=100)
    parser.add_argument("--support-steps", type=int, default=1000)
    parser.add_argument("--kmeans-steps", type=int, default=25)
    parser.add_argument("--negative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--calibration-pairs", type=int, default=10_000)
    parser.add_argument("--measurement-episodes", type=int, default=16)
    parser.add_argument("--measurement-transitions", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--bandwidth-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.context_index = args.context_index.expanduser().resolve()
    args.public_policy_market = args.public_policy_market.expanduser().resolve()
    args.deployment_private_registry = args.deployment_private_registry.expanduser().resolve()
    args.oracle_root = args.oracle_root.expanduser().resolve()
    args.market = _market(args.public_policy_market, args.deployment_private_registry)
    context_rows = _context_rows(args.context_index)
    forbidden = [
        str(row["context_id"])
        for row in context_rows
        if row["role"] not in {"source", "development_query"}
    ]
    if forbidden:
        raise V031RawTransitionError(
            "v0.31 development runner refuses non-development queries: "
            + ", ".join(forbidden)
        )
    inputs = prepare_signal_inputs(
        args.context_index,
        args.train_fraction,
        args.validation_fraction,
        args.split_seed,
        args.measurement_episodes,
        args.measurement_transitions,
    )
    run_config = {
        "schema": SCHEMA,
        "formal": False,
        "scope": "development only; no confirmatory/extrapolation",
        "context_index": str(args.context_index),
        "context_index_sha256": sha256_file(args.context_index),
        "public_policy_market": str(args.public_policy_market),
        "public_policy_market_sha256": sha256_file(args.public_policy_market),
        "deployment_private_registry_sha256": sha256_file(args.deployment_private_registry),
        "policy_market_id": args.market.policy_market_id,
        "oracle_root": str(args.oracle_root),
        "operator": "RAW_RKME",
        "views": list(VIEWS),
        "backend": args.backend,
        "block_size": args.block_size,
        "calibration_pairs": args.calibration_pairs,
        "bandwidth_seed": args.bandwidth_seed,
        "measurement": dict(inputs.measurement),
        "reducer": {
            "support_budget": args.support_budget,
            "support_steps": args.support_steps,
            "kmeans_steps": args.kmeans_steps,
            "negative_tolerance": args.negative_tolerance,
        },
    }
    _publish(args.output_dir / "run_config.json", run_config, resume=args.resume)
    completed = []
    for view_id in args.views:
        args.current_view = view_id
        _process_view(args, inputs, context_rows)
        completed.append(view_id)
    table = _build_table(args)
    _progress(
        args.output_dir / "progress.json",
        {
            "schema": SCHEMA,
            "stage": "complete" if table is not None else "partial",
            "completed_views": completed,
            "table_complete": table is not None,
        },
    )
    print(json.dumps({"completed_views": completed, "table_complete": table is not None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
