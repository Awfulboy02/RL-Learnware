from __future__ import annotations

from dataclasses import replace
import json
from types import MappingProxyType

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.preoracle_signal import (
    REGISTERED_SIGNAL_METRIC_ID,
    FormalSignalExtractionPlan,
    FormalSignalQueryBankJoin,
    PreOracleSignalError,
    PreOracleSignalOutcomePublication,
    build_preoracle_signal_outcome,
)
from policy_learnware_v0.v03.signal_prefix import FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
from tests.v03.test_signal_readout import _fixture


def _d(label: str) -> str:
    return sha256_json({"preoracle-signal-test": label})


def _extraction(fixture) -> FormalSignalExtractionPlan:
    extraction = fixture["extraction_plan"]
    assert isinstance(extraction, FormalSignalExtractionPlan)
    return extraction


def _outcome(fixture, run_id: str):
    return build_preoracle_signal_outcome(
        run_id=run_id,
        extraction_plan=_extraction(fixture),
        query_bank_alias_join=fixture["query_bank_alias_join"],
        readout_bundle=fixture["bundle"],
    )


def _changed_prefix_bundle(fixture, *, maximum: bool):
    bundle = fixture["bundle"]
    prefix_run = fixture["prefix_run"]
    points = list(prefix_run.points)
    point_index = -1 if maximum else 0
    point = points[point_index]
    metric = point.metric_record
    query_id = metric.rows[0].query_bank_id
    query_context = next(
        row.query_dynamics_context_id
        for row in metric.rows
        if row.query_bank_id == query_id
    )
    own_scope = query_context.rsplit("-", 2)[0]
    rows = list(metric.rows)
    changed = 0
    for index, row in enumerate(rows):
        if row.query_bank_id != query_id:
            continue
        if row.source_dynamics_context_id == f"{own_scope}-anchor-0":
            rows[index] = replace(row, distance=0.9)
            changed += 1
        elif row.source_dynamics_context_id == f"{own_scope}-anchor-1":
            rows[index] = replace(row, distance=0.1)
            changed += 1
    assert changed == 2
    forged_metric = replace(metric, rows=tuple(rows), metric_values=None)
    points[point_index] = replace(
        point, metric_record=forged_metric, point_digest=None
    )
    changed_run = replace(prefix_run, points=tuple(points), run_digest=None)
    return replace(
        bundle,
        prefix_runs={fixture["historical_key"]: changed_run},
        bundle_digest=None,
    )


def test_pure_builder_derives_exact_66_registered_rows_and_provenance() -> None:
    fixture = _fixture()
    bundle = fixture["bundle"]
    extraction = _extraction(fixture)
    outcome = _outcome(fixture, "v03-preoracle-signal-test")
    manifest = outcome.signal_outcome_manifest
    assert len(manifest.rows) == 66
    assert set(manifest.opaque_query_ids) == set(
        bundle.public_query_plan.opaque_query_ids
    )
    assert set(manifest.opaque_query_ids).isdisjoint(
        fixture["query_bank_id_by_opaque_query_id"].values()
    )
    assert {row.signal_metric_id for row in manifest.rows} == {
        REGISTERED_SIGNAL_METRIC_ID
    }
    assert all(
        tuple(row.prefix_signal_values) == FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
        and row.prefix_signal_values[64] == row.signal_value
        for row in manifest.rows
    )
    assert {row.task_id for row in manifest.rows} == {"task-0", "task-1"}
    assert {row.axis_id for row in manifest.rows} == {"axis-0", "axis-1"}
    assert manifest.freeze_manifest_digest == (
        bundle.atlas_run.formal_authorization.freeze_manifest.freeze_manifest_digest
    )
    assert extraction.plan_digest == (
        bundle.atlas_run.formal_authorization.freeze_manifest.preoracle_signal_outcome_plan_digest
    )
    assert manifest.signal_atlas_digest == bundle.atlas_run.run_digest
    assert manifest.public_query_plan_digest == bundle.public_query_plan.plan_digest
    assert manifest.query_alias_manifest_digest == (
        bundle.dynamics_public_query_join.query_alias_manifest_digest
    )
    assert outcome.to_dict()["oracle_data_accessed"] is False
    assert outcome.to_dict()["caller_supplied_numeric_values"] is False


def test_metric_and_work_key_are_not_caller_configurable() -> None:
    fixture = _fixture()
    extraction = _extraction(fixture)
    with pytest.raises(PreOracleSignalError, match="fixed as"):
        replace(
            extraction,
            signal_metric_id="caller-chosen-value",
            plan_digest=None,
        )
    with pytest.raises(PreOracleSignalError, match="both reviewed"):
        FormalSignalExtractionPlan.create(
            readout_plan=fixture["bundle"].plan,
            selected_work_key="not-a-reviewed-work-key",
            query_bank_alias_join=fixture["query_bank_alias_join"],
            selection_review_evidence_digest=_d(
                "external-preoracle-signal-selection-review"
            ),
            review_authority_receipt_digest=(
                fixture["bundle"].atlas_run.formal_authorization.freeze_manifest.review_authority_receipt_digest
            ),
        )


def test_forged_row_value_identity_and_evidence_fail_closed() -> None:
    fixture = _fixture()
    outcome = _outcome(fixture, "v03-preoracle-forgery-test")
    manifest = outcome.signal_outcome_manifest
    first = manifest.rows[0]
    forged_prefixes = dict(first.prefix_signal_values)
    forged_prefixes[64] = 0.0 if first.signal_value != 0.0 else 1.0
    for forged_row in (
        replace(
            first,
            signal_value=forged_prefixes[64],
            prefix_signal_values=forged_prefixes,
        ),
        replace(first, task_id="forged-task"),
        replace(first, signal_evidence_digest=_d("forged-evidence")),
    ):
        rows = list(manifest.rows)
        rows[0] = forged_row
        forged_manifest = replace(manifest, rows=tuple(rows), manifest_digest=None)
        with pytest.raises(PreOracleSignalError, match="pure bundle derivation"):
            replace(
                outcome,
                signal_outcome_manifest=forged_manifest,
                outcome_digest=None,
            )


def test_cross_bundle_and_cross_query_mapping_fail_closed() -> None:
    fixture = _fixture()
    extraction = _extraction(fixture)
    outcome = _outcome(fixture, "v03-preoracle-cross-test")
    changed_bundle = _changed_prefix_bundle(fixture, maximum=False)
    with pytest.raises(PreOracleSignalError, match="pure bundle derivation"):
        replace(outcome, readout_bundle=changed_bundle, outcome_digest=None)

    forged_mapping_plan = replace(
        extraction,
        query_alias_manifest_digest=_d("another-query-alias-mapping"),
        plan_digest=None,
    )
    with pytest.raises(PreOracleSignalError, match="externally frozen plan"):
        build_preoracle_signal_outcome(
            run_id="v03-preoracle-cross-test",
            extraction_plan=forged_mapping_plan,
            query_bank_alias_join=fixture["query_bank_alias_join"],
            readout_bundle=fixture["bundle"],
        )

    # Model persisted-object corruption: exchange two same-regime aliases while
    # retaining the old typed join digest.  The builder revalidates the nested
    # join rather than trusting the enclosing bundle's historical validation.
    join = fixture["query_join"]
    aliases = dict(join.dynamics_context_by_opaque_query_id)
    left, right = tuple(sorted(aliases))[:2]
    aliases[left], aliases[right] = aliases[right], aliases[left]
    object.__setattr__(
        join,
        "dynamics_context_by_opaque_query_id",
        MappingProxyType(aliases),
    )
    with pytest.raises(PreOracleSignalError, match="private query mapping"):
        replace(outcome, outcome_digest=None)


def test_maximum_prefix_must_equal_frozen_atlas_dynamics_readout() -> None:
    fixture = _fixture()
    changed_bundle = _changed_prefix_bundle(fixture, maximum=True)
    with pytest.raises(PreOracleSignalError, match="maximum-prefix"):
        build_preoracle_signal_outcome(
            run_id="v03-preoracle-max-prefix-test",
            extraction_plan=_extraction(fixture),
            query_bank_alias_join=fixture["query_bank_alias_join"],
            readout_bundle=changed_bundle,
        )


def test_private_bank_join_is_exact_bijective_and_publicly_withheld() -> None:
    fixture = _fixture()
    join = fixture["query_bank_alias_join"]
    assert isinstance(join, FormalSignalQueryBankJoin)
    opaque_ids = tuple(sorted(join.query_bank_id_by_opaque_query_id))
    bank_ids = tuple(join.query_bank_id_by_opaque_query_id.values())
    assert len(opaque_ids) == len(set(bank_ids)) == 66
    assert set(opaque_ids).isdisjoint(bank_ids)
    public_bytes = json.dumps(join.to_public_dict(), sort_keys=True)
    assert all(bank_id not in public_bytes for bank_id in bank_ids)
    assert "query_bank_id_by_opaque_query_id" not in public_bytes

    missing = dict(join.query_bank_id_by_opaque_query_id)
    missing.pop(opaque_ids[0])
    with pytest.raises(PreOracleSignalError, match="exactly 66"):
        replace(join, query_bank_id_by_opaque_query_id=missing, join_digest=None)

    duplicate = dict(join.query_bank_id_by_opaque_query_id)
    duplicate[opaque_ids[1]] = duplicate[opaque_ids[0]]
    with pytest.raises(PreOracleSignalError, match="one-to-one"):
        replace(join, query_bank_id_by_opaque_query_id=duplicate, join_digest=None)

    swapped = dict(join.query_bank_id_by_opaque_query_id)
    left = opaque_ids[0]
    right = next(
        query_id
        for query_id in opaque_ids[1:]
        if fixture["query_join"].dynamics_context_by_opaque_query_id[query_id]
        != fixture["query_join"].dynamics_context_by_opaque_query_id[left]
    )
    swapped[left], swapped[right] = swapped[right], swapped[left]
    swapped_join = replace(
        join,
        query_bank_id_by_opaque_query_id=swapped,
        join_digest=None,
    )
    with pytest.raises(PreOracleSignalError, match="dynamics identity"):
        swapped_join.validate_against_bundle(fixture["bundle"])


def test_extraction_plan_must_equal_external_freeze() -> None:
    fixture = _fixture()
    forged = replace(
        _extraction(fixture),
        selection_review_evidence_digest=_d("post-freeze-selection"),
        plan_digest=None,
    )
    with pytest.raises(PreOracleSignalError, match="externally frozen plan"):
        build_preoracle_signal_outcome(
            run_id="v03-preoracle-freeze-test",
            extraction_plan=forged,
            query_bank_alias_join=fixture["query_bank_alias_join"],
            readout_bundle=fixture["bundle"],
        )


def test_full_publication_round_trip_and_tamper_fail_closed() -> None:
    fixture = _fixture()
    outcome = _outcome(fixture, "v03-preoracle-publication-test")
    publication = PreOracleSignalOutcomePublication.from_outcome(outcome)
    assert len(publication.signal_outcome_manifest.rows) == 66
    assert publication.oracle_data_accessed is False
    assert publication.signal_extraction_plan_digest == outcome.extraction_plan.plan_digest
    assert publication.formal_signal_readout_bundle_digest == (
        outcome.readout_bundle.bundle_digest
    )
    assert (
        PreOracleSignalOutcomePublication.from_dict(publication.to_dict()).to_dict()
        == publication.to_dict()
    )

    with pytest.raises(PreOracleSignalError, match="cannot report oracle access"):
        replace(publication, oracle_data_accessed=True, publication_digest=None)
    extra = {**publication.to_dict(), "unexpected": "field"}
    with pytest.raises(PreOracleSignalError, match="fields differ"):
        PreOracleSignalOutcomePublication.from_dict(extra)
    unsigned = {**publication.to_dict(), "publication_digest": None}
    with pytest.raises(PreOracleSignalError, match="publication_digest"):
        PreOracleSignalOutcomePublication.from_dict(unsigned)
    forged_manifest = replace(
        publication.signal_outcome_manifest,
        run_id="forged-publication-run",
        manifest_digest=None,
    )
    with pytest.raises(PreOracleSignalError, match="provenance differs"):
        replace(
            publication,
            signal_outcome_manifest=forged_manifest,
            signal_outcome_manifest_digest=str(forged_manifest.manifest_digest),
            publication_digest=None,
        )
