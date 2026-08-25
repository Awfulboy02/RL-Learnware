from __future__ import annotations

import copy
from pathlib import Path

import pytest

from policy_learnware_v0.io import atomic_write_json
from policy_learnware_v0.v02.completion import (
    V02CompletionError,
    V02CompletionRecord,
    validate_gate_state_payload,
)
from policy_learnware_v0.v02.gates import (
    FORMAL_GATE_STATE_SCHEMA,
    GATE_ORDER,
    GATE_REQUIREMENTS,
    FormalGateEvidenceError,
    FormalV02GateState,
    GateCriterionEvidence,
    GateEvidenceManifest,
    build_canonical_evidence_ref,
    criterion_evidence_relative_path,
    evaluate_gate,
    evaluate_gate_state,
    evaluate_registered_gate_criterion,
    gate_evidence_manifest_relative_path,
    missing_registered_gate_evaluators,
    registered_gate_evaluator_descriptor,
    validate_formal_gate_state_payload,
)


def _digest(character: str) -> str:
    return character * 64


def _ready_state():
    return evaluate_gate_state(
        tuple(
            evaluate_gate(gate, {name: True for name in GATE_REQUIREMENTS[gate]})
            for gate in GATE_ORDER
        )
    )


def _formal_state(
    tmp_path: Path,
    *,
    passed_overrides: dict[tuple[str, str], bool] | None = None,
):
    experiment_id = "v02-ready-r0"
    config_digest = _digest("a")
    root = tmp_path / "artifacts" / experiment_id
    root.mkdir(parents=True)
    overrides = passed_overrides or {}
    refs = {}
    evidence_by_gate = {}
    source_paths: dict[tuple[str, str], Path] = {}
    for gate in GATE_ORDER:
        refs[gate] = {}
        evidence_by_gate[gate] = {}
        for criterion in GATE_REQUIREMENTS[gate]:
            source_path = (
                root / "analysis" / "gate_sources" / f"{gate}-{criterion}.json"
            )
            atomic_write_json(
                source_path,
                {
                    "schema": "policy-learnware.v02-test-gate-source.v0",
                    "config_digest": config_digest,
                    "source_check": criterion,
                },
            )
            source_paths[(gate, criterion)] = source_path
            source_ref = build_canonical_evidence_ref(
                source_path,
                experiment_root=root,
                expected_config_digest=config_digest,
            )
            evidence = GateCriterionEvidence(
                config_digest=config_digest,
                gate=gate,
                criterion=criterion,
                passed=overrides.get((gate, criterion), True),
                derivation_id=f"policy-learnware.v02.gate/{gate}/{criterion}/v0",
                evaluator_digest=_digest("e"),
                source_artifacts=(source_ref,),
            )
            evidence_path = root / criterion_evidence_relative_path(gate, criterion)
            atomic_write_json(evidence_path, evidence.to_dict())
            evidence_by_gate[gate][criterion] = evidence
            refs[gate][criterion] = build_canonical_evidence_ref(
                evidence_path,
                experiment_root=root,
                expected_config_digest=config_digest,
            )
    manifest = GateEvidenceManifest(
        experiment_id=experiment_id,
        config_digest=config_digest,
        criteria=refs,
    )
    manifest_path = root / gate_evidence_manifest_relative_path()
    atomic_write_json(manifest_path, manifest.to_dict())
    manifest_ref = build_canonical_evidence_ref(
        manifest_path,
        experiment_root=root,
        expected_config_digest=config_digest,
    )
    core = evaluate_gate_state(
        tuple(
            evaluate_gate(
                gate,
                {
                    name: overrides.get((gate, name), True)
                    for name in GATE_REQUIREMENTS[gate]
                },
            )
            for gate in GATE_ORDER
        )
    )
    state = FormalV02GateState(
        experiment_id=experiment_id,
        config_digest=config_digest,
        gate_evidence_manifest=manifest,
        gate_evidence_manifest_ref=manifest_ref,
        criterion_evidence=evidence_by_gate,
        core_state=core,
    )
    return state, root, source_paths


def test_gate_state_is_rebuilt_from_primitive_checks() -> None:
    state = _ready_state()
    rebuilt = validate_gate_state_payload(state.to_dict())
    assert rebuilt.status == "READY_FOR_V03_JOINT_CONFIRMATORY"

    forged = copy.deepcopy(state.to_dict())
    forged["decisions"][0]["criteria"][0]["observed"] = False
    with pytest.raises(V02CompletionError, match="not derived|differs"):
        validate_gate_state_payload(forged)


def test_formal_gate_state_binds_every_criterion_to_typed_evidence(
    tmp_path: Path,
) -> None:
    state, root, _ = _formal_state(tmp_path)
    payload = state.to_dict()

    assert payload["schema"] == FORMAL_GATE_STATE_SCHEMA
    assert payload["config_digest"] == _digest("a")
    assert payload["gate_evidence_manifest_digest"] == (
        state.gate_evidence_manifest.digest
    )
    assert state.status == "READY_FOR_V03_JOINT_CONFIRMATORY"
    assert all(
        "evidence_ref" in criterion
        for decision in payload["decisions"]
        for criterion in decision["criteria"]
    )
    assert not state.is_formally_authoritative
    with pytest.raises(FormalGateEvidenceError, match="no trusted content-derived"):
        validate_formal_gate_state_payload(
            payload,
            experiment_root=root,
            expected_experiment_id="v02-ready-r0",
            expected_config_digest=_digest("a"),
        )


def test_formal_validation_rejects_legacy_naked_boolean_state(tmp_path: Path) -> None:
    state = _ready_state()
    with pytest.raises(FormalGateEvidenceError, match="formal|fields differ"):
        validate_formal_gate_state_payload(
            state.to_dict(),
            experiment_root=tmp_path,
            expected_experiment_id="v02-ready-r0",
            expected_config_digest=_digest("a"),
        )


def test_summary_only_axis_evidence_cannot_receive_formal_evaluator_authority(
    tmp_path: Path,
) -> None:
    state, _, _ = _formal_state(tmp_path)
    original = state.criterion_evidence["G02-Engineering"]["all_registered_axis_audits"]
    with pytest.raises(FormalGateEvidenceError, match="no trusted content evaluator"):
        registered_gate_evaluator_descriptor(
            "G02-Engineering", "all_registered_axis_audits"
        )
    source = {
        "schema": "policy-learnware.v02-axis-audit-validation.v0",
        "passed": True,
        "config_digest": _digest("a"),
        "axis_registry_digest": _digest("b"),
        # This self-consistent 1/1 summary is the poison case that the former
        # evaluator could not distinguish from the reviewed 36-unit universe.
        "expected_work_units": 1,
        "validated_work_units": 1,
        "violations": [],
    }
    with pytest.raises(FormalGateEvidenceError, match="no trusted content-derived"):
        evaluate_registered_gate_criterion(
            original,
            (source,),
            expected_experiment_id="v02-ready-r0",
            expected_config_digest=_digest("a"),
        )
    assert len(missing_registered_gate_evaluators()) == sum(
        len(GATE_REQUIREMENTS[gate]) for gate in GATE_ORDER
    )


def test_formal_validation_rereads_and_rejects_changed_source_bytes(
    tmp_path: Path,
) -> None:
    state, root, source_paths = _formal_state(tmp_path)
    first = source_paths[(GATE_ORDER[0], GATE_REQUIREMENTS[GATE_ORDER[0]][0])]
    atomic_write_json(
        first,
        {
            "schema": "policy-learnware.v02-test-gate-source.v0",
            "config_digest": _digest("a"),
            "source_check": "changed-after-gate-evaluation",
        },
        overwrite=True,
    )

    with pytest.raises(FormalGateEvidenceError, match="bytes changed"):
        validate_formal_gate_state_payload(
            state.to_dict(),
            experiment_root=root,
            expected_experiment_id="v02-ready-r0",
            expected_config_digest=_digest("a"),
        )


def test_formal_validation_rejects_self_consistent_manifest_digest_forgery(
    tmp_path: Path,
) -> None:
    state, root, _ = _formal_state(tmp_path)
    forged = copy.deepcopy(state.to_dict())
    forged["gate_evidence_manifest_digest"] = _digest("f")

    with pytest.raises(FormalGateEvidenceError, match="manifest digest"):
        validate_formal_gate_state_payload(
            forged,
            experiment_root=root,
            expected_experiment_id="v02-ready-r0",
            expected_config_digest=_digest("a"),
        )


def test_completion_record_rejects_hand_constructed_formal_state(
    tmp_path: Path,
) -> None:
    formal_state, _, _ = _formal_state(tmp_path)
    with pytest.raises(V02CompletionError, match="trusted in-process"):
        V02CompletionRecord(
            experiment_id="v02-ready-r0",
            config_digest=_digest("a"),
            config_file_sha256=_digest("b"),
            gate_state=formal_state,
            gate_state_digest=formal_state.digest,
            gate_state_file_sha256=_digest("d"),
            recompute_audit_digest=_digest("e"),
            recompute_audit_file_sha256=_digest("f"),
            theory_status="PENDING",
            literature_novelty_audit_status="PENDING",
        )


def test_completion_record_requires_explicit_review_statuses() -> None:
    with pytest.raises(V02CompletionError, match="evidence-bound"):
        V02CompletionRecord(
            experiment_id="v02-ready-r0",
            config_digest=_digest("a"),
            config_file_sha256=_digest("b"),
            gate_state=_ready_state(),
            gate_state_digest=_digest("c"),
            gate_state_file_sha256=_digest("d"),
            recompute_audit_digest=_digest("e"),
            recompute_audit_file_sha256=_digest("f"),
            theory_status="PENDING",
            literature_novelty_audit_status="PENDING",
        )


def test_completion_record_checks_authority_before_review_statuses(
    tmp_path: Path,
) -> None:
    formal_state, _, _ = _formal_state(tmp_path)
    with pytest.raises(V02CompletionError, match="trusted in-process"):
        V02CompletionRecord(
            experiment_id="v02-ready-r0",
            config_digest=_digest("a"),
            config_file_sha256=_digest("b"),
            gate_state=formal_state,
            gate_state_digest=formal_state.digest,
            gate_state_file_sha256=_digest("d"),
            recompute_audit_digest=_digest("e"),
            recompute_audit_file_sha256=_digest("f"),
            theory_status="PENDING",
            literature_novelty_audit_status="review later",
        )
