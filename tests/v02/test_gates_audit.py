from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from policy_learnware_v0.io import atomic_write_json, atomic_write_npz
from policy_learnware_v0.v02.audit import (
    PublicArtifactRule,
    artifact_tree_digest,
    audit_evidence_contract,
    audit_oracle_independence,
    audit_public_artifacts,
    audit_public_market_entries,
)
from policy_learnware_v0.v02.gates import (
    GATE_ORDER,
    GATE_REQUIREMENTS,
    evaluate_gate,
    evaluate_gate_state,
)
from policy_learnware_v0.v02.selectors import EvidenceContract


def _passing_gate(name: str):
    return evaluate_gate(name, {check: True for check in GATE_REQUIREMENTS[name]})


def test_all_five_gates_advance_only_to_v03_ready() -> None:
    decisions = tuple(_passing_gate(name) for name in GATE_ORDER)
    state = evaluate_gate_state(decisions)

    assert state.status == "READY_FOR_V03_JOINT_CONFIRMATORY"
    assert state.ready_for_v03
    assert state.blocking_gate is None
    assert state.passed_gates == GATE_ORDER
    assert all(decision.to_dict()["fail_closed"] for decision in decisions)
    assert "COMPLETE_GO_LMIN_EMPIRICAL" not in json.dumps(state.to_dict())


def test_gate_missing_unknown_or_nonboolean_checks_fail_closed() -> None:
    gate = "G02-Scope"
    values = {check: True for check in GATE_REQUIREMENTS[gate]}
    values.pop(GATE_REQUIREMENTS[gate][0])
    values[GATE_REQUIREMENTS[gate][1]] = 1
    values["caller_uploaded_passed"] = True

    decision = evaluate_gate(gate, values)

    assert not decision.passed
    assert decision.outcome == "FAIL"
    assert GATE_REQUIREMENTS[gate][0] in decision.missing_checks
    assert decision.unexpected_checks == ("caller_uploaded_passed",)
    assert evaluate_gate_state((decision,)).status == "BLOCKED_ENGINEERING"


@pytest.mark.parametrize(
    ("failed_gate", "expected_status"),
    (
        ("G02-Scope", "BLOCKED_ENGINEERING"),
        ("G02-Engineering", "BLOCKED_ENGINEERING"),
        ("G02-Market", "COMPLETE_NO_GO_MARKET"),
        ("G02-Replace", "COMPLETE_NO_GO_CORRO_INCUMBENT"),
        ("G02-SplitFreeze", "BLOCKED_ENGINEERING"),
    ),
)
def test_state_machine_distinguishes_blockage_from_scientific_no_go(
    failed_gate: str, expected_status: str
) -> None:
    decisions = []
    for name in GATE_ORDER:
        checks = {check: True for check in GATE_REQUIREMENTS[name]}
        if name == failed_gate:
            checks[GATE_REQUIREMENTS[name][0]] = False
        decisions.append(evaluate_gate(name, checks))

    state = evaluate_gate_state(decisions)

    assert state.status == expected_status
    assert state.blocking_gate == failed_gate
    assert state.passed_gates == GATE_ORDER[: GATE_ORDER.index(failed_gate)]
    assert not state.ready_for_v03


def test_later_pass_cannot_skip_a_missing_prerequisite() -> None:
    decisions = tuple(_passing_gate(name) for name in GATE_ORDER[1:])
    state = evaluate_gate_state(decisions)

    assert state.status == "BLOCKED_ENGINEERING"
    assert state.blocking_gate == "G02-Scope"
    assert state.passed_gates == ()


def test_public_market_entry_allowlist_rejects_runtime_schema_and_tie_collision() -> None:
    first = "lw-" + "1" * 20
    second = "lw-" + "2" * 20
    safe = {
        first: {
            "opaque_learnware_id": first,
            "normalized_source_competence": 0.8,
            "tie_break_token": "token-a",
        },
        second: {
            "opaque_learnware_id": second,
            "normalized_source_competence": 0.7,
            "tie_break_token": "token-b",
        },
    }
    assert audit_public_market_entries(safe).passed

    leaked = {key: dict(value) for key, value in safe.items()}
    leaked[first]["runtime_contract"] = {"task_contract_digest": "a" * 64}
    leaked[second]["tie_break_token"] = "token-a"
    audit = audit_public_market_entries(leaked)

    assert not audit.passed
    reasons = {item.reason for item in audit.violations}
    assert "entry_field_allowlist_mismatch" in reasons
    assert "tie_break_token_collision" in reasons


def test_public_tree_exact_allowlist_and_forbidden_json_string_scans(tmp_path: Path) -> None:
    root = tmp_path / "public"
    atomic_write_json(
        root / "manifest.json",
        {"schema": "policy-learnware.v02-public-test.v0", "opaque_id": "lw-" + "a" * 20},
    )
    rules = (
        PublicArtifactRule(
            "manifest.json",
            "json",
            json_keys=frozenset({"schema", "opaque_id"}),
        ),
    )
    assert audit_public_artifacts(root, rules).passed

    atomic_write_json(root / "extra.json", {"schema": "unknown"})
    audit = audit_public_artifacts(root, rules)
    assert not audit.passed
    assert any(item.reason == "unregistered_artifact" for item in audit.violations)

    leaked_root = tmp_path / "leaked"
    atomic_write_json(
        leaked_root / "manifest.json",
        {
            "schema": "policy-learnware.v02-public-test.v0",
            "opaque_id": "../oracle/private.json",
            "runtime_contract": {"task": "WalkerWalk"},
        },
    )
    leaked_rule = PublicArtifactRule(
        "manifest.json",
        "json",
        json_keys=frozenset({"schema", "opaque_id", "runtime_contract"}),
    )
    leaked_audit = audit_public_artifacts(leaked_root, (leaked_rule,))
    leaked_reasons = {item.reason for item in leaked_audit.violations}
    assert {"forbidden_json_key", "forbidden_string_token", "path_like_public_string"} <= leaked_reasons


def test_public_tree_rejects_forbidden_npz_members_and_member_traversal(tmp_path: Path) -> None:
    root = tmp_path / "public"
    atomic_write_npz(
        root / "spec.npz",
        {
            "supports": np.zeros((1, 1), dtype=np.float64),
            "probe_seeds": np.asarray([1], dtype=np.int64),
        },
    )
    rule = PublicArtifactRule(
        "spec.npz",
        "npz",
        npz_members=frozenset({"supports", "probe_seeds"}),
    )
    audit = audit_public_artifacts(root, (rule,))
    assert not audit.passed
    assert any(item.reason == "forbidden_npz_member" for item in audit.violations)

    traversal_root = tmp_path / "traversal"
    traversal_root.mkdir()
    member = io.BytesIO()
    np.lib.format.write_array(
        member, np.asarray([1.0], dtype=np.float64), allow_pickle=False
    )
    with zipfile.ZipFile(traversal_root / "spec.npz", "w") as archive:
        archive.writestr("../factor.npy", member.getvalue())
    traversal_rule = PublicArtifactRule(
        "spec.npz", "npz", npz_members=frozenset({"../factor"})
    )
    traversal_audit = audit_public_artifacts(traversal_root, (traversal_rule,))
    assert not traversal_audit.passed
    assert any(item.reason == "npz_member_traversal" for item in traversal_audit.violations)


def test_public_tree_rejects_symlink_escape_and_traversing_rules(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (public / "escape.json").symlink_to(outside)
    rule = PublicArtifactRule(
        "escape.json", "json", json_keys=frozenset({"schema"})
    )

    audit = audit_public_artifacts(public, (rule,))

    assert not audit.passed
    assert any(item.reason == "symlink_escape" for item in audit.violations)
    with pytest.raises(ValueError, match="traversal-free"):
        PublicArtifactRule(
            "../escape.json", "json", json_keys=frozenset({"schema"})
        )


def _safe_evidence() -> EvidenceContract:
    return EvidenceContract(
        reads_source_raw_data=False,
        reads_development_policy_returns=False,
        reads_target_parameters=False,
        reads_target_transitions=True,
        reads_candidate_independent_probe_rewards=False,
        reads_candidate_target_rollouts=False,
        reads_candidate_policy_target_rewards=False,
        target_gradient_updates=0,
        reads_submit_side_profiles=False,
    )


def test_evidence_contract_requires_zero_target_policy_rollout_reward_and_update() -> None:
    safe = _safe_evidence()
    assert audit_evidence_contract(safe).passed

    for field, value, expected_check in (
        (
            "reads_target_task_reward_schema_identity",
            True,
            "target_task_reward_schema_identity_hidden",
        ),
        ("reads_candidate_target_rollouts", True, "candidate_target_rollouts_zero"),
        (
            "reads_candidate_policy_target_rewards",
            True,
            "candidate_policy_target_rewards_zero",
        ),
        ("target_gradient_updates", 1, "target_gradient_updates_zero"),
    ):
        payload = safe.to_dict()
        payload[field] = value
        audit = audit_evidence_contract(payload)
        assert not audit.passed
        assert audit.checks[expected_check] is False

    uploaded = {**safe.to_dict(), "passed": True}
    audit = audit_evidence_contract(uploaded)
    assert not audit.passed
    assert any(item.reason == "contract_field_allowlist_mismatch" for item in audit.violations)


def _public_replay(_market: Path, _measurement: Path, selector: Path) -> str:
    return artifact_tree_digest(selector)


def test_oracle_missing_and_poison_do_not_change_selection_digest(tmp_path: Path) -> None:
    market = tmp_path / "run" / "market_public"
    measurement = tmp_path / "run" / "measurement"
    selector = tmp_path / "run" / "selector_outputs"
    atomic_write_json(market / "market.json", {"schema": "market"})
    atomic_write_json(measurement / "query.json", {"schema": "query"})
    atomic_write_json(selector / "selection.json", {"schema": "selection", "selected": "lw-a"})

    audit = audit_oracle_independence(
        _public_replay,
        market_public_root=market,
        measurement_root=measurement,
        selector_outputs_root=selector,
    )

    assert audit.passed
    assert audit.to_dict()["oracle_root_passed_to_replay"] is False
    assert set(audit.scenarios) == {"missing", "poison"}


def test_oracle_poison_detects_hidden_sibling_lookup(tmp_path: Path) -> None:
    market = tmp_path / "run" / "market_public"
    measurement = tmp_path / "run" / "measurement"
    selector = tmp_path / "run" / "selector_outputs"
    atomic_write_json(market / "market.json", {"schema": "market"})
    atomic_write_json(measurement / "query.json", {"schema": "query"})
    atomic_write_json(selector / "selection.json", {"schema": "selection"})

    def cheating_replay(_market: Path, _measurement: Path, selection: Path) -> str:
        if (selection.parent / "confirmatory_oracle_private").exists():
            return "0" * 64
        return artifact_tree_digest(selection)

    audit = audit_oracle_independence(
        cheating_replay,
        market_public_root=market,
        measurement_root=measurement,
        selector_outputs_root=selector,
    )

    assert not audit.passed
    assert audit.scenarios["missing"]["passed"] is True
    assert audit.scenarios["poison"]["passed"] is False
