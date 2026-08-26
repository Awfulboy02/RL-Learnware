from __future__ import annotations

import json
from pathlib import Path

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.cli import CLI_VERSION, _parser, main
from policy_learnware_v0.v03.schemas import ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST
from policy_learnware_v0.v03.windowing import WindowingProtocol


def _config(path: Path) -> None:
    protocol = WindowingProtocol(4, 2, "mean", True)
    payload = {
        "schema": "policy-learnware.v03-foundation-config.v0",
        "development_id": "v03-cli-test",
        "stage": "foundation_development",
        "protocol_id": sha256_json({"protocol": "test"}),
        "task_private_ids": ["task-a", "task-b"],
        "artifact_root": str(path.parent / "artifacts"),
        "anonymous_public_allowlist": sorted(ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST),
        "window_protocol": {
            "window_length": 4,
            "stride": 2,
            "pooling": "mean",
            "pad_final_window": True,
            "protocol_id": protocol.window_protocol_digest,
        },
        "primary_freeze": {
            "query_mode": "QUERY_EMPIRICAL",
            "selector_mode": "distance_only",
            "pool_scope": "anonymous_global",
            "opaque_learnware_field": "opaque_learnware_id",
            "opaque_query_field": "opaque_query_id",
            "oracle_owner": "policy-learnware-paper1",
        },
        "review_decisions_digest": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_help_is_a_real_command_surface_and_does_not_run_acceptance(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "accept-numeric" in output
    assert "intake-v02-policy-pool" in output
    assert '"checks"' not in output
    assert _parser().parse_args(["accept-numeric"]).command == "accept-numeric"
    commands = set(_parser()._subparsers._group_actions[0].choices)
    assert commands == {
        "accept-numeric",
        "intake-v02-policy-pool",
        "validate-config",
    }


def test_validate_config_and_numeric_acceptance_emit_canonical_json(
    tmp_path: Path, capsys
) -> None:
    config = tmp_path / "v03.json"
    _config(config)
    assert main(["validate-config", str(config)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "VALID"
    assert validated["payload"]["formal_scope_ready"] is False
    assert validated["payload"]["review_digest_status"] == "NOT_DECLARED"
    gate = validated["payload"]["encoder_extension_gate"]
    assert gate == {
        "enabled": False,
        "migration_target": None,
        "optional_asset_requirements_active": False,
        "completion_eligible": False,
        "confirmatory_artifact_access": False,
        "formal_authority": False,
    }

    assert main(["accept-numeric"]) == 0
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["status"] == "ENGINEERING_PASS"
    assert accepted["payload"]["passed"] is True


def test_validate_config_does_not_self_sign_declared_review_digest(
    tmp_path: Path, capsys
) -> None:
    config = tmp_path / "v03-formal.json"
    _config(config)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["stage"] = "formal_freeze"
    payload["review_decisions_digest"] = sha256_json({"review": "declared-only"})
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["validate-config", str(config)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "VALID"
    assert validated["payload"]["review_digest_status"] == "DECLARED_UNVERIFIED"
    assert validated["payload"]["formal_scope_ready"] is False


def test_validate_config_reports_opted_in_v04_gate_without_granting_authority(
    tmp_path: Path, capsys
) -> None:
    config = tmp_path / "v03-v04-gate.json"
    _config(config)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["encoder_extension_gate"] = {
        "enabled": True,
        "migration_target": "v0.4",
        "authority_digest": sha256_json({"v04": "migration-decision"}),
    }
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["validate-config", str(config)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["payload"]["encoder_extension_gate"] == {
        "enabled": True,
        "migration_target": "v0.4",
        "optional_asset_requirements_active": False,
        "completion_eligible": False,
        "confirmatory_artifact_access": False,
        "formal_authority": False,
    }
    assert validated["payload"]["formal_scope_ready"] is False


def test_validate_config_rejects_enabled_extension_gate_at_formal_freeze(
    tmp_path: Path, capsys
) -> None:
    config = tmp_path / "v03-formal-enabled-gate.json"
    _config(config)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload.update(
        {
            "stage": "formal_freeze",
            "review_decisions_digest": sha256_json({"review": "declared-only"}),
            "encoder_extension_gate": {
                "enabled": True,
                "migration_target": "v0.4",
                "authority_digest": sha256_json({"v04": "migration-decision"}),
            },
        }
    )
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["validate-config", str(config)]) == 1
    blocked = json.loads(capsys.readouterr().err)
    assert blocked["status"] == "BLOCKED"
    assert "encoder_extension_gate.enabled=false" in blocked["payload"]["error"]


def test_validate_config_explicit_disabled_gate_is_asset_free(
    tmp_path: Path, capsys
) -> None:
    config = tmp_path / "v03-disabled-gate.json"
    _config(config)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["encoder_extension_gate"] = {"enabled": False}
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["validate-config", str(config)]) == 0
    validated = json.loads(capsys.readouterr().out)
    gate = validated["payload"]["encoder_extension_gate"]
    assert gate["enabled"] is False
    assert gate["optional_asset_requirements_active"] is False
    assert gate["confirmatory_artifact_access"] is False
    assert gate["formal_authority"] is False


def test_production_intake_parser_has_no_trust_anchor_override() -> None:
    options = {
        option
        for action in _parser()._subparsers._group_actions[0].choices[
            "intake-v02-policy-pool"
        ]._actions
        for option in action.option_strings
    }
    assert "--trust-anchor" not in options
    assert CLI_VERSION == "0.3.0"
