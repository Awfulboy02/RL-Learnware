from __future__ import annotations

from pathlib import Path

import pytest

import policy_learnware_v0.v01.cli as cli
import policy_learnware_v0.v01.recompute as recompute
from policy_learnware_v0.v01.recompute import (
    ExecutableEvidence,
    compute_gate_d_from_evidence,
    compute_oracle_poison_evidence,
    compute_taskspec_capability_evidence,
)


def test_real_taskspec_parser_and_source_have_least_privilege() -> None:
    source_root = Path(cli.__file__).resolve().parent
    evidence = compute_taskspec_capability_evidence(
        cli.build_parser(),
        measurement_module_paths=[source_root / "taskspec.py", source_root / "plans.py"],
        orchestration_path=source_root / "cli.py",
    )
    assert evidence.passed, evidence.details["errors"]
    assert {
        name
        for name in evidence.details["parser_actions"]
        if name.endswith("_root")
    } == {"base_artifacts_root", "measurement_root"}
    assert evidence.details["orchestration"]["writer_domains"] == ["measurement"]
    assert set(evidence.details["orchestration"]["audited_function_names"]) == {
        "_compute_taskspec",
        "_compute_taskspec_impl",
        "_resume_verified_taskspec_bundle",
    }
    functions = evidence.details["orchestration"]["functions"]
    assert "_compute_taskspec_impl" in functions["_compute_taskspec"][
        "called_local_functions"
    ]
    assert "_resume_verified_taskspec_bundle" in functions[
        "_compute_taskspec_impl"
    ]["called_local_functions"]


def test_source_dependency_audit_detects_oracle_import(tmp_path: Path) -> None:
    bad_module = tmp_path / "bad_taskspec.py"
    bad_module.write_text("from policy_learnware_v0.v01.oracle import OracleShard\n")
    handler = tmp_path / "handler.py"
    handler.write_text(
        "def _compute_taskspec(args):\n"
        "    writer = layout.writer('measurement')\n"
        "    return writer\n"
    )
    evidence = compute_taskspec_capability_evidence(
        cli.build_parser(),
        measurement_module_paths=[bad_module],
        orchestration_path=handler,
    )
    assert not evidence.passed
    assert "forbidden imports" in " ".join(evidence.details["errors"])


def test_capability_audit_checks_actual_impl_private_root_access(
    tmp_path: Path,
) -> None:
    handler = tmp_path / "handler.py"
    handler.write_text(
        "def _compute_taskspec(args):\n"
        "    return _compute_taskspec_impl(args)\n"
        "def _compute_taskspec_impl(args):\n"
        "    return args.oracle_root\n"
        "def _resume_verified_taskspec_bundle(layout):\n"
        "    return layout.writer('measurement')\n"
    )
    evidence = compute_taskspec_capability_evidence(
        cli.build_parser(),
        measurement_module_paths=[],
        orchestration_path=handler,
    )
    assert not evidence.passed
    assert "_compute_taskspec_impl" in " ".join(evidence.details["errors"])
    assert "oracle_root" in " ".join(evidence.details["errors"])


def test_oracle_poison_independence_is_actually_executed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    independent_calls = []
    measurement = tmp_path / "experiment" / "measurement"
    measurement.mkdir(parents=True)
    (measurement / "opaque-input.txt").write_text("measurement-only")
    baseline = {
        "output_artifact_sha256": {"taskspec_matrix_axes.json": "a" * 64},
        "output_digest": "b" * 64,
        "success_attempt_id": "v01xa-" + "1" * 24,
        "success_attempt_digest": "c" * 64,
        "success_attempt_sha256": "d" * 64,
        "execution_attempt_artifact_sha256": {"attempt.json": "d" * 64},
        "release_digest": "e" * 64,
    }
    monkeypatch.setattr(
        recompute,
        "_taskspec_release_snapshot",
        lambda _root, *, source_support_sizes: dict(baseline),
    )

    def independent(measurement: Path):
        assert (measurement / "opaque-input.txt").read_text() == "measurement-only"
        independent_calls.append(measurement)
        return {
            "resumed": True,
            "execution_attempt_id": baseline["success_attempt_id"],
            "execution_attempt_digest": baseline["success_attempt_digest"],
            "execution_attempt_sha256": baseline["success_attempt_sha256"],
            "taskspec_matrix_sha256": "a" * 64,
        }

    passed = compute_oracle_poison_evidence(
        independent,
        measurement_root=measurement,
        source_support_sizes=(1,),
    )
    assert passed.passed, passed.details
    assert len(independent_calls) == 2
    assert independent_calls[0] != independent_calls[1]
    assert passed.details["oracle_root_passed_to_runner"] is False

    def contaminated(staged_measurement: Path):
        result = independent(staged_measurement)
        if (staged_measurement.parent / "oracle_private").exists():
            result["taskspec_matrix_sha256"] = "f" * 64
        return result

    failed = compute_oracle_poison_evidence(
        contaminated,
        measurement_root=measurement,
        source_support_sizes=(1,),
    )
    assert not failed.passed
    assert "resume result differs" in " ".join(failed.details["errors"])


def _evidence(kind: str, passed: bool = True) -> ExecutableEvidence:
    return ExecutableEvidence(kind, passed, {"computed": True})


def test_gate_d_derives_all_seven_checks_and_fails_closed() -> None:
    values = {
        "taskspec_capability": _evidence("taskspec_capability_and_source"),
        "measurement_visibility": _evidence("measurement_visibility"),
        "protocol_binding": _evidence("protocol_digest_binding"),
        "smoke_formal_separation": _evidence("smoke_formal_separation"),
        "oracle_poison_independence": _evidence("oracle_poison_independence"),
    }
    report = compute_gate_d_from_evidence(**values)
    assert report["passed"]
    assert len(report["criteria"]) == 7
    assert report["caller_supplied_passed_attestations_consumed"] is False

    values["oracle_poison_independence"] = _evidence(
        "oracle_poison_independence", False
    )
    report = compute_gate_d_from_evidence(**values)
    assert not report["passed"]
    criterion = next(
        item
        for item in report["criteria"]
        if item["name"] == "oracle_poison_does_not_change_taskspec_digest"
    )
    assert criterion["passed"] is False

    with pytest.raises(TypeError):
        compute_gate_d_from_evidence(**{**values, "unexpected_passed": True})
