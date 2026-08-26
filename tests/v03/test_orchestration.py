from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from policy_learnware_v0.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from policy_learnware_v0.v03.orchestration import (
    DEVELOPMENT_MODE,
    FORMAL_MODE,
    PRODUCTION_STAGE_IDS,
    STAGE_SEMANTIC_REQUIREMENTS,
    FormalStageRequestTemplate,
    PipelineCompletionManifest,
    StageAdapterResult,
    StageDependencyBinding,
    StageExecutionContext,
    StageExecutionError,
    StageExecutionManifest,
    StageExecutionReceipt,
    StageInputBinding,
    StageOutputArtifact,
    StageSemanticOutput,
    execute_stage,
    execute_stage_from_files,
    formal_stage_output_payload_digest,
    verify_pipeline_completion,
)
from policy_learnware_v0.v03.artifacts import V03ArtifactLayout
from policy_learnware_v0.v03.baselines import REQUIRED_BASELINE_METHOD_IDS
from policy_learnware_v0.v03.claim_audit import FormalClaimAudit
from policy_learnware_v0.v03.cli import main
from policy_learnware_v0.v03.costs import (
    COST_COMPONENT_IDS,
    CostComponentRecord,
    V03CostLedger,
    frozen_cost_protocol_digest,
)
from policy_learnware_v0.v03.formal_gates import (
    FormalAttributionAdmission,
    FormalMarketAdmission,
    FormalProbeAdmission,
)
from policy_learnware_v0.v03.policy_outcomes import (
    SignalOutcomeManifest,
    SignalOutcomeRow,
)
from policy_learnware_v0.v03.preoracle_signal import (
    REGISTERED_SIGNAL_METRIC_ID,
    PreOracleSignalOutcomePublication,
)
from policy_learnware_v0.v03.preflight import (
    FORMAL_PRODUCTION_STAGE_IDS,
    HARD_TODO_IDS,
    HardTodoEvidence,
    IndependentRecomputeAttestation,
    OracleUnlockHandoff,
    PreExperimentFreezeManifest,
    PublicQueryPlan,
    PublicRankingBarrier,
    PublicRankingPublication,
    formal_baseline_input_plan_digest,
    formal_stage_adapter_binding_digest,
)
from policy_learnware_v0.v03.statistics import FormalStatisticsResult
from policy_learnware_v0.v03.signal_prefix import (
    FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS,
)


def _d(label: str) -> str:
    return sha256_json({"orchestration-test": label})


def _todo(todo_id: str) -> HardTodoEvidence:
    return HardTodoEvidence(
        todo_id=todo_id,
        contract_digest=_d(f"{todo_id}:contract"),
        implementation_digest=_d(f"{todo_id}:implementation"),
        unit_test_evidence_digest=_d(f"{todo_id}:unit"),
        synthetic_fixture_evidence_digest=_d(f"{todo_id}:fixture"),
        cpu_integration_evidence_digest=_d(f"{todo_id}:cpu"),
    )


def _adapter_identity(stage_id: str) -> tuple[str, str]:
    index = PRODUCTION_STAGE_IDS.index(stage_id)
    if stage_id == "collect-source-receipts":
        return _FixtureAdapter.adapter_id, _FixtureAdapter.adapter_contract_digest
    return f"adapter-{index}", _d(f"adapter:{stage_id}")


def _formal_adapter_bindings() -> dict[str, str]:
    return {
        stage_id: formal_stage_adapter_binding_digest(
            stage_id, *_adapter_identity(stage_id)
        )
        for stage_id in FORMAL_PRODUCTION_STAGE_IDS
    }


def _formal_request_templates(
    static_inputs_by_stage: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    static = static_inputs_by_stage or {}
    result = {}
    for stage_id in FORMAL_PRODUCTION_STAGE_IDS:
        adapter_id, adapter_contract_digest = _adapter_identity(stage_id)
        parameters_digest = (
            _d("source-parameters")
            if stage_id == "collect-source-receipts"
            else _d(f"parameters:{stage_id}")
        )
        template = FormalStageRequestTemplate(
            stage_id=stage_id,
            adapter_id=adapter_id,
            adapter_contract_digest=adapter_contract_digest,
            parameters_digest=parameters_digest,
            static_input_content_digests=static.get(stage_id, {}),
        )
        result[stage_id] = template.request_template_digest
    return result


def _freeze(
    *,
    formal: bool = False,
    public_query_plan_digest: str | None = None,
    baseline_plan_digest: str | None = None,
    formal_stage_request_template_digests: dict[str, str] | None = None,
) -> PreExperimentFreezeManifest:
    return PreExperimentFreezeManifest(
        freeze_id="orchestration-test-freeze",
        config_bytes_digest=_d("config"),
        implementation_tree_digest=_d("tree"),
        clean_commit_digest=_d("commit"),
        review_decisions_digest=_d("review"),
        review_authority_receipt_digest=_d("external-authority") if formal else None,
        review_authority_verified=formal,
        encoder_extension_gate_enabled=False,
        data_role_manifest_digest=_d("roles"),
        canonicalizer_registry_digest=_d("canonicalizer"),
        signal_matrix_digest=_d("matrix"),
        signal_contrast_plan_digest=_d("signal-contrast-plan"),
        signal_materiality_threshold_digest=_d("signal-materiality-thresholds"),
        formal_signal_readout_plan_digest=_d("formal-signal-readout-plan"),
        preoracle_signal_outcome_plan_digest=_d("preoracle-signal-outcome-plan"),
        signal_identity_registry_digest=_d("identity"),
        signal_execution_protocol_digest=_d("execution"),
        representation_plan_digest=_d("representations"),
        condition_plan_digest=_d("conditions"),
        formal_source_fit_schedule_digest=_d("source-fit-schedule"),
        formal_source_membership_digest=_d("source-membership"),
        signal_work_item_graph_digest=_d("work-items"),
        formal_signal_prefix_schedule_digest=_d("signal-prefix-schedule"),
        dynamics_axis_registry_digest=_d("dynamics-axis-registry"),
        public_query_plan_digest=(
            _d("queries")
            if public_query_plan_digest is None
            else public_query_plan_digest
        ),
        baseline_plan_digest=(
            _d("baselines") if baseline_plan_digest is None else baseline_plan_digest
        ),
        statistics_plan_digest=_d("statistics"),
        cost_protocol_digest=frozen_cost_protocol_digest(),
        source_reduced_query_empirical_protocol_digest=_d("kme"),
        formal_gate_plan_digests=(
            {
                "G03-Attribution": _d("formal-attribution-plan"),
                "G03-Probe": _d("formal-probe-plan"),
                "G03-Market": _d("formal-market-plan"),
            }
            if formal
            else {}
        ),
        formal_stage_request_template_digests=(
            (
                _formal_request_templates()
                if formal_stage_request_template_digests is None
                else formal_stage_request_template_digests
            )
            if formal
            else {}
        ),
        hard_todo_evidence=tuple(_todo(item) for item in HARD_TODO_IDS),
        formal_stage_adapter_binding_digests=(
            _formal_adapter_bindings() if formal else {}
        ),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _formal_readout_payload(
    freeze: PreExperimentFreezeManifest, *, attribution_evidence_digest: str
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "policy-learnware.v03-formal-signal-readout-bundle.v0",
        "readout_plan_digest": freeze.formal_signal_readout_plan_digest,
        "freeze_manifest_digest": freeze.freeze_manifest_digest,
        "formal_authorization_digest": _d("signal-atlas-authorization"),
        "atlas_run_digest": _d("signal-atlas-run"),
        "atlas_public_projection_digest": _d("signal-atlas-public"),
        "prefix_run_digests": {"work-0": _d("prefix-run")},
        "prefix_public_projection_digests": {
            "work-0": _d("prefix-public")
        },
        "dynamics_diagnostic_digests": {"work-0": _d("dynamics-run")},
        "dynamics_public_projection_digests": {
            "work-0": _d("dynamics-public")
        },
        "dynamics_public_query_join_digest": _d("dynamics-query-join"),
        "dynamics_query_join_public_projection_digest": _d(
            "dynamics-query-join-public"
        ),
        "contrast_gate_evaluation_digest": _d("contrast-gate"),
        "contrast_gate_public_projection_digest": _d("contrast-gate-public"),
        "pair_control_evidence_set_digest": _d("pair-control-evidence"),
        "attribution_gate_evidence_digest": attribution_evidence_digest,
    }
    return {**body, "bundle_digest": sha256_json(body)}


def _public_readout_payload(
    freeze: PreExperimentFreezeManifest, *, private_bundle_digest: str
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "policy-learnware.v03-public-signal-readout-bundle.v0",
        "readout_plan_digest": freeze.formal_signal_readout_plan_digest,
        "freeze_manifest_digest": freeze.freeze_manifest_digest,
        "formal_authorization_digest": _d("signal-atlas-authorization"),
        "atlas": {"summary_digest": _d("public-atlas-summary")},
        "prefix_readouts": {"summary_digest": _d("public-prefix-summary")},
        "dynamics_readouts": {
            "summary_digest": _d("public-dynamics-summary")
        },
        "dynamics_query_join": {"summary_digest": _d("public-query-join")},
        "contrast_gate": {"summary_digest": _d("public-contrast-gate")},
        "pair_control_evidence_set_digest": _d("pair-control-evidence"),
        "pair_control_evidence_count": 4,
        "attribution_gate_evidence_digest": _d("formal-attribution-evidence"),
        "private_bank_task_context_and_alias_rows_withheld": True,
        "private_bundle_digest": private_bundle_digest,
    }
    return {**body, "public_projection_digest": sha256_json(body)}


def _cost_component(component_id: str) -> CostComponentRecord:
    return CostComponentRecord(
        component_id=component_id,
        measurement_receipt_digest=_d(f"cost-receipt:{component_id}"),
        input_artifact_set_digest=_d(f"cost-input:{component_id}"),
        output_artifact_set_digest=_d(f"cost-output:{component_id}"),
        wall_seconds=1.0,
        gpu_seconds=0.0,
        peak_memory_bytes=1024,
        artifact_bytes=512,
        environment_steps=64 if component_id == "PROBE_COLLECTION" else 0,
        invocation_count=3 if component_id == "END_TO_END_WARM" else 1,
    )


class _FixtureAdapter:
    adapter_id = "fixture-source-adapter"
    adapter_contract_digest = _d("fixture-adapter-contract")

    def __init__(self, *, filename: str = "source-receipts.json") -> None:
        self.calls = 0
        self.filename = filename

    def execute(self, context: StageExecutionContext) -> StageAdapterResult:
        self.calls += 1
        evidence_path = context.layout.artifact(
            "source_market", f"evidence-{self.filename}"
        )
        evidence_sha = context.layout.writer("source_market").publish_json(
            evidence_path,
            {
                "schema": "fixture-source-receipt-evidence.v0",
                "verified_inputs": {
                    input_id: sha256_file(path)
                    for input_id, path in context.verified_inputs.items()
                },
            },
        )
        payload = StageSemanticOutput(
            stage_id=context.manifest.stage_id,
            semantic_id="source-receipt-set",
            run_id=context.manifest.run_id,
            freeze_manifest_digest=context.freeze_manifest.freeze_manifest_digest,
            evidence_digests={"source-receipt-evidence": evidence_sha},
            record_count=30,
        )
        path = context.layout.artifact("source_market", self.filename)
        digest = context.layout.writer("source_market").publish_json(
            path, payload.to_dict()
        )
        artifact = StageOutputArtifact(
            domain="source_market",
            path=context.layout.relative(path),
            sha256=digest,
            semantic_id="source-receipt-set",
            payload_schema=payload.schema,
        )
        evidence_artifact = StageOutputArtifact(
            domain="source_market",
            path=context.layout.relative(evidence_path),
            sha256=evidence_sha,
        )
        return StageAdapterResult(
            output_payload_digest=formal_stage_output_payload_digest(
                (evidence_artifact, artifact),
                {"source-receipt-set": str(payload.semantic_digest)},
            ),
            artifacts=(evidence_artifact, artifact),
            record_counts={"source-receipt-set": 30},
        )


def _first_manifest(
    tmp_path: Path,
    freeze: PreExperimentFreezeManifest,
    *,
    formal: bool = False,
) -> StageExecutionManifest:
    source = tmp_path / "intake.json"
    _write_json(source, {"schema": "fixture-intake.v0", "cells": 90})
    return StageExecutionManifest(
        stage_id="collect-source-receipts",
        execution_id="collect-source-receipts-001",
        execution_mode=FORMAL_MODE if formal else DEVELOPMENT_MODE,
        run_id="v03-fixture-run",
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        adapter_id=_FixtureAdapter.adapter_id,
        adapter_contract_digest=_FixtureAdapter.adapter_contract_digest,
        parameters_digest=_d("source-parameters"),
        inputs=(
            StageInputBinding(
                input_id="v02-pool-intake",
                path=str(source),
                sha256=sha256_file(source),
            ),
        ),
        dependencies=(),
    )


def test_typed_adapter_execution_and_resume_are_end_to_end_and_byte_exact(
    tmp_path: Path,
) -> None:
    freeze = _freeze()
    manifest = _first_manifest(tmp_path, freeze)
    adapter = _FixtureAdapter()

    receipt, receipt_path, receipt_sha, resumed = execute_stage(
        manifest,
        freeze,
        artifacts_root=tmp_path / "artifacts",
        adapter=adapter,
    )
    assert resumed is False
    assert adapter.calls == 1
    assert receipt.stage_id == "collect-source-receipts"
    assert receipt.oracle_accessed is False
    assert receipt.adapter_result.record_counts == {"source-receipt-set": 30}
    assert sha256_file(receipt_path) == receipt_sha

    resumed_receipt, resumed_path, resumed_sha, resumed = execute_stage(
        manifest,
        freeze,
        artifacts_root=tmp_path / "artifacts",
        adapter=adapter,
        resume=True,
    )
    assert resumed is True
    assert adapter.calls == 1
    assert resumed_receipt.receipt_digest == receipt.receipt_digest
    assert resumed_path == receipt_path
    assert resumed_sha == receipt_sha

    with pytest.raises(StageExecutionError, match="complete execution request"):
        execute_stage(
            replace(manifest, parameters_digest=_d("changed-parameters")),
            freeze,
            artifacts_root=tmp_path / "artifacts",
            adapter=adapter,
            resume=True,
        )
    with pytest.raises(StageExecutionError, match="identity or verified binding"):
        replace(receipt, adapter_id="forged-adapter")

    output = tmp_path / "artifacts" / manifest.run_id / "source_market" / "source-receipts.json"
    output.write_bytes(b"tampered\n")
    with pytest.raises(StageExecutionError, match="output digest mismatch"):
        execute_stage(
            manifest,
            freeze,
            artifacts_root=tmp_path / "artifacts",
            adapter=adapter,
            resume=True,
        )
    assert adapter.calls == 1


def test_file_driver_requires_injected_adapter_and_exact_command_identity(
    tmp_path: Path,
) -> None:
    freeze = _freeze()
    manifest = _first_manifest(tmp_path, freeze)
    freeze_path = tmp_path / "freeze.json"
    stage_path = tmp_path / "stage.json"
    _write_json(freeze_path, freeze.to_dict())
    _write_json(stage_path, manifest.to_dict())

    with pytest.raises(StageExecutionError, match="server-injected adapter"):
        execute_stage_from_files(
            expected_stage_id=manifest.stage_id,
            stage_manifest_path=stage_path,
            freeze_manifest_path=freeze_path,
            artifacts_root=tmp_path / "artifacts",
            adapters=None,
        )
    with pytest.raises(StageExecutionError, match="received manifest"):
        execute_stage_from_files(
            expected_stage_id="build-market",
            stage_manifest_path=stage_path,
            freeze_manifest_path=freeze_path,
            artifacts_root=tmp_path / "artifacts",
            adapters={_FixtureAdapter.adapter_id: _FixtureAdapter()},
        )

    adapter = _FixtureAdapter()
    receipt, _, _, resumed = execute_stage_from_files(
        expected_stage_id=manifest.stage_id,
        stage_manifest_path=stage_path,
        freeze_manifest_path=freeze_path,
        artifacts_root=tmp_path / "artifacts",
        adapters={adapter.adapter_id: adapter},
    )
    assert receipt.status == "COMPLETE"
    assert resumed is False
    assert adapter.calls == 1


def test_cli_real_stage_path_executes_injected_adapter_and_resumes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze()
    manifest = _first_manifest(tmp_path, freeze)
    freeze_path = tmp_path / "freeze.json"
    stage_path = tmp_path / "stage.json"
    _write_json(freeze_path, freeze.to_dict())
    _write_json(stage_path, manifest.to_dict())
    adapter = _FixtureAdapter()
    argv = [
        "collect-source-receipts",
        "--stage-manifest",
        str(stage_path),
        "--freeze-manifest",
        str(freeze_path),
        "--artifacts-root",
        str(tmp_path / "artifacts"),
    ]
    assert main(argv, stage_adapters={adapter.adapter_id: adapter}) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "STAGE_COMPLETE"
    assert first["payload"]["adapter_executed"] is True
    assert first["payload"]["oracle_accessed"] is False
    assert adapter.calls == 1

    assert main([*argv, "--resume"], stage_adapters={adapter.adapter_id: adapter}) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["status"] == "STAGE_RESUMED"
    assert resumed["payload"]["adapter_executed"] is False
    assert adapter.calls == 1


def test_console_cli_loads_only_an_explicit_manifest_bound_adapter_entrypoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = _freeze()
    manifest = _first_manifest(tmp_path, freeze)
    freeze_path = tmp_path / "freeze.json"
    stage_path = tmp_path / "stage.json"
    _write_json(freeze_path, freeze.to_dict())
    _write_json(stage_path, manifest.to_dict())
    plugin = ModuleType("v03_fixture_adapter_plugin")
    plugin.make_adapter = lambda: _FixtureAdapter()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    assert (
        main(
            [
                manifest.stage_id,
                "--stage-manifest",
                str(stage_path),
                "--freeze-manifest",
                str(freeze_path),
                "--artifacts-root",
                str(tmp_path / "artifacts"),
                "--adapter-entrypoint",
                f"{plugin.__name__}:make_adapter",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "STAGE_COMPLETE"

    bad_plugin = ModuleType("v03_unreviewed_adapter_plugin")

    class _DifferentAdapter(_FixtureAdapter):
        adapter_id = "different-adapter"

    bad_plugin.make_adapter = lambda: _DifferentAdapter()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, bad_plugin.__name__, bad_plugin)
    assert (
        main(
            [
                manifest.stage_id,
                "--stage-manifest",
                str(stage_path),
                "--freeze-manifest",
                str(freeze_path),
                "--artifacts-root",
                str(tmp_path / "other-artifacts"),
                "--adapter-entrypoint",
                f"{bad_plugin.__name__}:make_adapter",
            ]
        )
        == 1
    )
    blocked = json.loads(capsys.readouterr().err)
    assert "server-injected adapter registry" in blocked["payload"]["error"]

def test_formal_mode_requires_external_authority_and_joint_namespace(
    tmp_path: Path,
) -> None:
    engineering_freeze = _freeze()
    formal_manifest = _first_manifest(tmp_path, engineering_freeze, formal=True)
    with pytest.raises(StageExecutionError, match="external review authority"):
        execute_stage(
            formal_manifest,
            engineering_freeze,
            artifacts_root=tmp_path / "artifacts",
            adapter=_FixtureAdapter(),
        )

    formal_freeze = _freeze(
        formal=True,
        formal_stage_request_template_digests=_formal_request_templates(
            {
                "collect-source-receipts": {
                    formal_manifest.inputs[0].input_id: formal_manifest.inputs[0].sha256
                }
            }
        ),
    )
    formal_manifest = replace(
        formal_manifest,
        freeze_manifest_digest=formal_freeze.freeze_manifest_digest,
    )
    receipt, path, _, _ = execute_stage(
        formal_manifest,
        formal_freeze,
        artifacts_root=tmp_path / "formal-artifacts",
        adapter=_FixtureAdapter(),
    )
    assert receipt.execution_mode == FORMAL_MODE
    assert path.name == "collect-source-receipts-001.json"
    assert path.parts[-3:] == ("scope", "stage_executions", path.name)

    changed_parameters = replace(
        formal_manifest,
        execution_id="changed-formal-parameters",
        parameters_digest=_d("unreviewed-formal-parameters"),
    )
    untouched = _FixtureAdapter()
    with pytest.raises(StageExecutionError, match="reviewed request template"):
        execute_stage(
            changed_parameters,
            formal_freeze,
            artifacts_root=tmp_path / "changed-parameter-artifacts",
            adapter=untouched,
        )
    assert untouched.calls == 0

    extra_static = tmp_path / "extra-static.json"
    _write_json(extra_static, {"schema": "unreviewed-static-input.v0"})
    changed_static_inputs = replace(
        formal_manifest,
        execution_id="changed-formal-static-inputs",
        inputs=(
            *formal_manifest.inputs,
            StageInputBinding(
                input_id="unreviewed-static-input",
                path=str(extra_static),
                sha256=sha256_file(extra_static),
            ),
        ),
    )
    untouched = _FixtureAdapter()
    with pytest.raises(StageExecutionError, match="reviewed request template"):
        execute_stage(
            changed_static_inputs,
            formal_freeze,
            artifacts_root=tmp_path / "changed-static-artifacts",
            adapter=untouched,
        )
    assert untouched.calls == 0

    class _ArbitraryFileAdapter(_FixtureAdapter):
        def execute(self, context: StageExecutionContext) -> StageAdapterResult:
            self.calls += 1
            output = context.layout.artifact("source_market", "arbitrary.json")
            digest = context.layout.writer("source_market").publish_json(
                output, {"schema": "caller-owned-file.v0"}
            )
            return StageAdapterResult(
                output_payload_digest=_d("caller-owned-output"),
                artifacts=(
                    StageOutputArtifact(
                        domain="source_market",
                        path=context.layout.relative(output),
                        sha256=digest,
                    ),
                ),
                record_counts={"rows": 30},
            )

    with pytest.raises(StageExecutionError, match="semantic output coverage"):
        execute_stage(
            replace(formal_manifest, execution_id="arbitrary-formal-output"),
            formal_freeze,
            artifacts_root=tmp_path / "arbitrary-output-artifacts",
            adapter=_ArbitraryFileAdapter(),
        )

    class _UnboundSemanticAdapter(_FixtureAdapter):
        def execute(self, context: StageExecutionContext) -> StageAdapterResult:
            self.calls += 1
            evidence_path = context.layout.artifact(
                "source_market", "actual-underlying-evidence.json"
            )
            evidence_sha = context.layout.writer("source_market").publish_json(
                evidence_path, {"schema": "fixture-actual-evidence.v0"}
            )
            payload = StageSemanticOutput(
                stage_id=context.manifest.stage_id,
                semantic_id="source-receipt-set",
                run_id=context.manifest.run_id,
                freeze_manifest_digest=context.freeze_manifest.freeze_manifest_digest,
                evidence_digests={"forged-evidence": _d("not-an-output-byte")},
                record_count=30,
            )
            output = context.layout.artifact(
                "source_market", "unbound-source-receipts.json"
            )
            output_sha = context.layout.writer("source_market").publish_json(
                output, payload.to_dict()
            )
            evidence_artifact = StageOutputArtifact(
                domain="source_market",
                path=context.layout.relative(evidence_path),
                sha256=evidence_sha,
            )
            semantic_artifact = StageOutputArtifact(
                domain="source_market",
                path=context.layout.relative(output),
                sha256=output_sha,
                semantic_id="source-receipt-set",
                payload_schema=payload.schema,
            )
            return StageAdapterResult(
                output_payload_digest=formal_stage_output_payload_digest(
                    (evidence_artifact, semantic_artifact),
                    {"source-receipt-set": str(payload.semantic_digest)},
                ),
                artifacts=(evidence_artifact, semantic_artifact),
                record_counts={"source-receipt-set": 30},
            )

    with pytest.raises(StageExecutionError, match="lacks a bound underlying"):
        execute_stage(
            replace(formal_manifest, execution_id="unbound-formal-semantic"),
            formal_freeze,
            artifacts_root=tmp_path / "unbound-semantic-artifacts",
            adapter=_UnboundSemanticAdapter(),
        )

    class _OvercompleteSourceAdapter(_FixtureAdapter):
        def execute(self, context: StageExecutionContext) -> StageAdapterResult:
            self.calls += 1
            evidence_path = context.layout.artifact(
                "source_market", "overcomplete-source-evidence.json"
            )
            evidence_sha = context.layout.writer("source_market").publish_json(
                evidence_path, {"schema": "fixture-overcomplete-evidence.v0"}
            )
            payload = StageSemanticOutput(
                stage_id=context.manifest.stage_id,
                semantic_id="source-receipt-set",
                run_id=context.manifest.run_id,
                freeze_manifest_digest=context.freeze_manifest.freeze_manifest_digest,
                evidence_digests={"formal-evidence": evidence_sha},
                record_count=31,
            )
            output = context.layout.artifact(
                "source_market", "overcomplete-source.json"
            )
            digest = context.layout.writer("source_market").publish_json(
                output, payload.to_dict()
            )
            artifact = StageOutputArtifact(
                domain="source_market",
                path=context.layout.relative(output),
                sha256=digest,
                semantic_id="source-receipt-set",
                payload_schema=payload.schema,
            )
            evidence_artifact = StageOutputArtifact(
                domain="source_market",
                path=context.layout.relative(evidence_path),
                sha256=evidence_sha,
            )
            return StageAdapterResult(
                output_payload_digest=formal_stage_output_payload_digest(
                    (evidence_artifact, artifact),
                    {"source-receipt-set": str(payload.semantic_digest)},
                ),
                artifacts=(evidence_artifact, artifact),
                record_counts={"source-receipt-set": 31},
            )

    with pytest.raises(StageExecutionError, match="requires exactly 30"):
        execute_stage(
            replace(formal_manifest, execution_id="overcomplete-formal-output"),
            formal_freeze,
            artifacts_root=tmp_path / "overcomplete-output-artifacts",
            adapter=_OvercompleteSourceAdapter(),
        )

    class _UnreviewedAdapter(_FixtureAdapter):
        adapter_id = "unreviewed-adapter"
        adapter_contract_digest = _d("unreviewed-adapter-contract")

    unreviewed_manifest = replace(
        formal_manifest,
        execution_id="unreviewed-formal-stage",
        adapter_id=_UnreviewedAdapter.adapter_id,
        adapter_contract_digest=_UnreviewedAdapter.adapter_contract_digest,
    )
    with pytest.raises(StageExecutionError, match="not allowlisted"):
        execute_stage(
            unreviewed_manifest,
            formal_freeze,
            artifacts_root=tmp_path / "formal-artifacts",
            adapter=_UnreviewedAdapter(),
        )


def test_successor_requires_every_predecessor_output_as_exact_physical_input(
    tmp_path: Path,
) -> None:
    freeze = _freeze()
    first_manifest = _first_manifest(tmp_path, freeze)
    first_receipt, first_path, first_file_sha, _ = execute_stage(
        first_manifest,
        freeze,
        artifacts_root=tmp_path / "artifacts",
        adapter=_FixtureAdapter(),
    )
    dependency = StageDependencyBinding(
        stage_id=first_receipt.stage_id,
        receipt_path=str(first_path),
        receipt_file_sha256=first_file_sha,
        receipt_digest=first_receipt.receipt_digest,
    )
    unrelated = tmp_path / "unrelated.json"
    _write_json(unrelated, {"schema": "unrelated.v0"})
    successor = StageExecutionManifest(
        stage_id="build-market",
        execution_id="build-market-unlinked",
        execution_mode=DEVELOPMENT_MODE,
        run_id=first_manifest.run_id,
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        adapter_id=_FixtureAdapter.adapter_id,
        adapter_contract_digest=_FixtureAdapter.adapter_contract_digest,
        parameters_digest=_d("build-market-parameters"),
        inputs=(
            StageInputBinding(
                input_id="unrelated-input",
                path=str(unrelated),
                sha256=sha256_file(unrelated),
            ),
        ),
        dependencies=(dependency,),
    )
    with pytest.raises(StageExecutionError, match="every predecessor output"):
        execute_stage(
            successor,
            freeze,
            artifacts_root=tmp_path / "artifacts",
            adapter=_FixtureAdapter(filename="market.json"),
        )


def test_manifest_rejects_oracle_access_and_wrong_predecessor() -> None:
    freeze = _freeze()
    source = StageInputBinding(
        input_id="input",
        path="/does/not-need-to-exist-at-parse-time",
        sha256=_d("input"),
    )
    with pytest.raises(StageExecutionError, match="cannot request oracle"):
        StageExecutionManifest(
            stage_id="collect-source-receipts",
            execution_id="oracle-request",
            execution_mode=DEVELOPMENT_MODE,
            run_id="run",
            freeze_manifest_digest=freeze.freeze_manifest_digest,
            adapter_id="adapter",
            adapter_contract_digest=_d("adapter"),
            parameters_digest=_d("parameters"),
            inputs=(source,),
            dependencies=(),
            oracle_access_requested=True,
        )
    with pytest.raises(StageExecutionError, match="exact predecessor"):
        StageExecutionManifest(
            stage_id="build-market",
            execution_id="missing-predecessor",
            execution_mode=DEVELOPMENT_MODE,
            run_id="run",
            freeze_manifest_digest=freeze.freeze_manifest_digest,
            adapter_id="adapter",
            adapter_contract_digest=_d("adapter"),
            parameters_digest=_d("parameters"),
            inputs=(source,),
            dependencies=(),
        )


def test_completion_checker_requires_all_formal_stage_and_external_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    regimes = {
        f"v03q-{index:032x}": (
            "EXACT" if index < 30 else "INTERPOLATION" if index < 54 else "EXTRAPOLATION"
        )
        for index in range(66)
    }
    query_plan = PublicQueryPlan(
        regime_by_opaque_query_id=regimes,
        query_alias_manifest_digest=_d("completion-query-aliases"),
    )
    publications = tuple(
        PublicRankingPublication(
            method_id=method_id,
            opaque_query_id=query_id,
            ranking_digest=_d(f"ranking:{method_id}:{query_id}"),
            query_spec_digest=_d(f"query-spec:{query_id}"),
            probe_dataset_digest=_d(f"probe:{query_id}"),
            target_evidence_digest=_d(f"target:{query_id}"),
            cost_digest=_d(f"cost:{method_id}:{query_id}"),
            policy_market_id=_d("completion-market"),
            representation_index_digest=_d(f"representation:{method_id}"),
            selector_view_digest=_d(f"selector-view:{method_id}"),
            evidence_contract_digest=_d(f"evidence:{method_id}"),
            selector_artifact_digest=_d(f"selector:{method_id}"),
            development_freeze_digest=_d("development-freeze"),
            query_input_digest=_d(f"query-input:{method_id}:{query_id}"),
            query_mode="QUERY_EMPIRICAL",
            execution_mode=FORMAL_MODE,
            development_context_count=24,
        )
        for method_id in REQUIRED_BASELINE_METHOD_IDS
        for query_id in query_plan.opaque_query_ids
    )
    root_input = tmp_path / "formal-root-input.json"
    _write_json(root_input, {"schema": "fixture-formal-root-input.v0"})
    freeze = _freeze(
        formal=True,
        public_query_plan_digest=str(query_plan.plan_digest),
        baseline_plan_digest=formal_baseline_input_plan_digest(
            publications,
            expected_opaque_query_ids=query_plan.opaque_query_ids,
            query_alias_manifest_digest=query_plan.query_alias_manifest_digest,
        ),
        formal_stage_request_template_digests=_formal_request_templates(
            {
                "collect-source-receipts": {
                    "formal-root-input": sha256_file(root_input)
                }
            }
        ),
    )
    layout = V03ArtifactLayout.joint(tmp_path / "artifacts", "formal-completion-run")
    attribution_evidence_digest = _d("formal-attribution-evidence")
    private_readout = _formal_readout_payload(
        freeze, attribution_evidence_digest=attribution_evidence_digest
    )
    public_readout = _public_readout_payload(
        freeze, private_bundle_digest=str(private_readout["bundle_digest"])
    )
    bindings: list[StageDependencyBinding] = []
    previous_outputs: tuple[tuple[Path, str], ...] = ()
    for index, stage_id in enumerate(PRODUCTION_STAGE_IDS):
        artifacts: list[StageOutputArtifact] = []
        semantic_digests: dict[str, str] = {}
        semantic_counts: dict[str, int] = {}
        current_outputs: list[tuple[Path, str]] = []
        for requirement in STAGE_SEMANTIC_REQUIREMENTS[stage_id]:
            if requirement.semantic_id == "formal-signal-readout-bundle":
                payload = private_readout
            elif requirement.semantic_id == "public-signal-readout-bundle":
                payload = public_readout
            else:
                evidence_path = layout.artifact(
                    requirement.domain,
                    f"stage-{index}-{requirement.semantic_id}-evidence.json",
                )
                evidence_sha = layout.writer(requirement.domain).publish_json(
                    evidence_path,
                    {
                        "schema": "fixture-formal-stage-evidence.v0",
                        "stage_id": stage_id,
                        "semantic_id": requirement.semantic_id,
                    },
                )
                artifacts.append(
                    StageOutputArtifact(
                        domain=requirement.domain,
                        path=layout.relative(evidence_path),
                        sha256=evidence_sha,
                    )
                )
                current_outputs.append((evidence_path, evidence_sha))
                payload = StageSemanticOutput(
                    stage_id=stage_id,
                    semantic_id=requirement.semantic_id,
                    run_id=layout.run_id,
                    freeze_manifest_digest=freeze.freeze_manifest_digest,
                    evidence_digests={"formal-evidence": evidence_sha},
                    record_count=(
                        requirement.exact_record_count
                        if requirement.exact_record_count is not None
                        else requirement.minimum_record_count
                    ),
                ).to_dict()
            output_path = layout.artifact(
                requirement.domain, f"stage-{index}-{requirement.semantic_id}.json"
            )
            if requirement.domain == "signal_atlas":
                output_sha = layout._authorized_signal_atlas_writer().publish_json(
                    output_path, payload
                )
            else:
                output_sha = layout.writer(requirement.domain).publish_json(
                    output_path, payload
                )
            artifact = StageOutputArtifact(
                domain=requirement.domain,
                path=layout.relative(output_path),
                sha256=output_sha,
                semantic_id=requirement.semantic_id,
                payload_schema=requirement.payload_schema,
            )
            artifacts.append(artifact)
            current_outputs.append((output_path, output_sha))
            if requirement.semantic_id == "formal-signal-readout-bundle":
                semantic_digests[requirement.semantic_id] = str(
                    private_readout["bundle_digest"]
                )
                semantic_counts[requirement.semantic_id] = 39
            elif requirement.semantic_id == "public-signal-readout-bundle":
                semantic_digests[requirement.semantic_id] = str(
                    public_readout["public_projection_digest"]
                )
                semantic_counts[requirement.semantic_id] = 39
            else:
                semantic_digests[requirement.semantic_id] = str(
                    payload["semantic_digest"]
                )
                semantic_counts[requirement.semantic_id] = int(
                    payload["record_count"]
                )
        result = StageAdapterResult(
            output_payload_digest=formal_stage_output_payload_digest(
                artifacts, semantic_digests
            ),
            artifacts=tuple(artifacts),
            record_counts=semantic_counts,
        )
        if index == 0:
            stage_inputs = (
                StageInputBinding(
                    input_id="formal-root-input",
                    path=str(root_input),
                    sha256=sha256_file(root_input),
                ),
            )
            dependencies: tuple[StageDependencyBinding, ...] = ()
        else:
            stage_inputs = tuple(
                StageInputBinding(
                    input_id=f"predecessor-output-{index - 1}-{position}",
                    path=str(output_path),
                    sha256=output_sha,
                )
                for position, (output_path, output_sha) in enumerate(previous_outputs)
            )
            dependencies = (bindings[-1],)
        adapter_id, adapter_contract_digest = _adapter_identity(stage_id)
        stage_manifest = StageExecutionManifest(
            stage_id=stage_id,
            execution_id=f"formal-stage-{index}",
            execution_mode=FORMAL_MODE,
            run_id=layout.run_id,
            freeze_manifest_digest=freeze.freeze_manifest_digest,
            adapter_id=adapter_id,
            adapter_contract_digest=adapter_contract_digest,
            parameters_digest=(
                _d("source-parameters")
                if stage_id == "collect-source-receipts"
                else _d(f"parameters:{stage_id}")
            ),
            inputs=stage_inputs,
            dependencies=dependencies,
        )
        receipt = StageExecutionReceipt(
            stage_id=stage_id,
            execution_id=f"formal-stage-{index}",
            execution_mode=FORMAL_MODE,
            run_id=layout.run_id,
            freeze_manifest_digest=freeze.freeze_manifest_digest,
            manifest_digest=stage_manifest.manifest_digest,
            adapter_id=adapter_id,
            adapter_contract_digest=adapter_contract_digest,
            verified_input_set_digest=sha256_json(
                {item.input_id: item.binding_digest for item in stage_inputs}
            ),
            verified_dependency_set_digest=sha256_json(
                {item.stage_id: item.receipt_digest for item in dependencies}
            ),
            execution_manifest=stage_manifest,
            adapter_result=result,
        )
        receipt_path = layout.artifact(
            "scope", "stage_executions", f"formal-stage-{index}.json"
        )
        receipt_sha = layout.writer("scope").publish_json(receipt_path, receipt.to_dict())
        bindings.append(
            StageDependencyBinding(
                stage_id=stage_id,
                receipt_path=str(receipt_path),
                receipt_file_sha256=receipt_sha,
                receipt_digest=receipt.receipt_digest,
            )
        )
        previous_outputs = tuple(current_outputs)

    signal_rows = tuple(
        SignalOutcomeRow(
            opaque_query_id=query_id,
            task_id=f"task-{index % 2}",
            axis_id=f"axis-{(index // 2) % 2}",
            context_id=f"context-{index}",
            signal_metric_id=REGISTERED_SIGNAL_METRIC_ID,
            signal_value=float(index) / 100.0,
            prefix_signal_values={
                prefix: float(index) / 100.0
                for prefix in FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
            },
            signal_evidence_digest=_d(f"signal-evidence:{query_id}"),
        )
        for index, query_id in enumerate(query_plan.opaque_query_ids)
    )
    signal_manifest = SignalOutcomeManifest(
        run_id=layout.run_id,
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        public_query_plan_digest=str(query_plan.plan_digest),
        query_alias_manifest_digest=query_plan.query_alias_manifest_digest,
        signal_atlas_digest=str(private_readout["atlas_run_digest"]),
        signal_prefix_schedule_digest=freeze.formal_signal_prefix_schedule_digest,
        rows=signal_rows,
    )
    signal_publication = PreOracleSignalOutcomePublication(
        run_id=layout.run_id,
        signal_extraction_plan_digest=freeze.preoracle_signal_outcome_plan_digest,
        formal_query_bank_alias_join_digest=_d("formal-query-bank-alias-join"),
        formal_signal_readout_bundle_digest=str(private_readout["bundle_digest"]),
        preoracle_signal_outcome_digest=_d("preoracle-signal-outcome"),
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        public_query_plan_digest=str(query_plan.plan_digest),
        query_alias_manifest_digest=query_plan.query_alias_manifest_digest,
        signal_atlas_digest=str(private_readout["atlas_run_digest"]),
        signal_prefix_schedule_digest=freeze.formal_signal_prefix_schedule_digest,
        signal_outcome_manifest_digest=str(signal_manifest.manifest_digest),
        prefix_diagnostics_digests={
            prefix: _d(f"prefix-diagnostic:{prefix}")
            for prefix in FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
        },
        signal_outcome_manifest=signal_manifest,
    )
    barrier = PublicRankingBarrier(
        run_id=layout.run_id,
        freeze_manifest=freeze,
        query_plan=query_plan,
        expected_opaque_query_ids=query_plan.opaque_query_ids,
        expected_method_ids=REQUIRED_BASELINE_METHOD_IDS,
        publications=publications,
        query_alias_manifest_digest=query_plan.query_alias_manifest_digest,
        preoracle_signal_outcome_manifest_digest=str(signal_manifest.manifest_digest),
    )
    handoff = OracleUnlockHandoff(
        run_id=layout.run_id,
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        public_ranking_barrier_digest=barrier.barrier_digest,
    )
    statistics = FormalStatisticsResult(
        run_id=layout.run_id,
        statistics_plan_digest=freeze.statistics_plan_digest,
        statistics_input_digest=_d("completion-statistics-input"),
        preexperiment_freeze_manifest_digest=freeze.freeze_manifest_digest,
        public_ranking_barrier_digest=barrier.barrier_digest,
        oracle_unlock_handoff_digest=handoff.handoff_digest,
        oracle_evidence_manifest_digest=_d("completion-oracle-evidence"),
        contrast_results={"h-completion": {"status": "OBSERVED"}},
        family_results={"family-completion": {"status": "COMPUTED"}},
    )
    recompute = IndependentRecomputeAttestation(
        run_id=layout.run_id,
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        public_ranking_barrier_digest=barrier.barrier_digest,
        formal_statistics_result_digest=statistics.result_digest,
        raw_input_manifest_digest=statistics.statistics_input_digest,
        primary_artifact_root_digest=_d("completion-primary-root"),
        recompute_artifact_root_digest=_d("completion-recompute-root"),
        primary_result_digest=statistics.result_digest,
        recompute_result_digest=statistics.result_digest,
        primary_process_nonce_digest=_d("completion-primary-process"),
        recompute_process_nonce_digest=_d("completion-recompute-process"),
    )
    attribution = FormalAttributionAdmission(
        plan_digest=freeze.formal_gate_plan_digests["G03-Attribution"],
        evidence_digest=attribution_evidence_digest,
        authority_receipt_digest=_d("formal-attribution-authority"),
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        status="PASS",
        failure_reasons=(),
    )
    probe = FormalProbeAdmission(
        plan_digest=freeze.formal_gate_plan_digests["G03-Probe"],
        evidence_digest=_d("formal-probe-evidence"),
        authority_receipt_digest=_d("formal-probe-authority"),
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        status="PASS",
        failure_reasons=(),
        task_count=2,
        axis_count=2,
    )
    market = FormalMarketAdmission(
        plan_digest=freeze.formal_gate_plan_digests["G03-Market"],
        evidence_digest=_d("formal-market-evidence"),
        authority_receipt_digest=_d("formal-market-authority"),
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        status="ASSET_READY",
        failure_reasons=(),
        candidate_count=90,
        market_entry_count=30,
    )
    cost = V03CostLedger(
        run_id=layout.run_id,
        execution_scope="FORMAL",
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        cost_protocol_digest=freeze.cost_protocol_digest,
        prefix_cost_evidence_digest=_d("formal-prefix-cost-evidence"),
        components=tuple(_cost_component(item) for item in COST_COMPONENT_IDS),
    )
    claim = FormalClaimAudit(
        run_id=layout.run_id,
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        attribution_admission_digest=str(attribution.admission_digest),
        probe_admission_digest=str(probe.admission_digest),
        market_admission_digest=str(market.admission_digest),
        signal_readout_bundle_digest=str(private_readout["bundle_digest"]),
        cost_ledger_digest=str(cost.ledger_digest),
        preoracle_signal_outcome_digest=(
            signal_publication.preoracle_signal_outcome_digest
        ),
        pre_oracle_signal_manifest_digest=str(signal_manifest.manifest_digest),
        public_ranking_barrier_digest=barrier.barrier_digest,
        statistics_result_digest=statistics.result_digest,
        independent_recompute_attestation_digest=recompute.attestation_digest,
        completion_state="COMPLETE_GO_PAPER_I",
        allowed_claim_ids=("paper-i-primary",),
        review_authority_receipt_digest=str(
            freeze.review_authority_receipt_digest
        ),
    )
    external_records = {
        "formal-attribution-admission": attribution.to_dict(),
        "formal-claim-audit": claim.to_dict(),
        "formal-cost-ledger": cost.to_dict(),
        "formal-market-admission": market.to_dict(),
        "formal-probe-admission": probe.to_dict(),
        "formal-signal-readout-bundle": private_readout,
        "independent-recompute-attestation": recompute.to_dict(),
        "oracle-unlock-handoff": handoff.to_dict(),
        "pre-oracle-signal-manifest": signal_publication.to_dict(),
        "public-ranking-barrier": barrier.to_dict(),
        "statistics-result": statistics.to_dict(),
    }
    external = []
    for artifact_id, payload in external_records.items():
        path = tmp_path / "external" / f"{artifact_id}.json"
        _write_json(path, payload)
        external.append(
            StageInputBinding(
                input_id=artifact_id,
                path=str(path),
                sha256=sha256_file(path),
            )
        )
    completion_manifest = PipelineCompletionManifest(
        completion_id="formal-pipeline-completion",
        run_id=layout.run_id,
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        stage_receipts=tuple(bindings),
        external_artifacts=tuple(external),
    )
    # Plan §20.3: formal v0.3 completion remains eligible when no v0.4
    # extension directory or asset exists.  The extension gate is frozen off,
    # and none of the exact completion inputs may name an extension artifact.
    assert freeze.encoder_extension_gate_enabled is False
    assert not (layout.run_root / "optional_extensions").exists()
    assert not (layout.run_root / "v04_development").exists()
    assert all(
        "v04" not in item.input_id.lower()
        and "optional_extension" not in item.input_id.lower()
        for item in completion_manifest.external_artifacts
    )
    completion, path, file_sha, resumed = verify_pipeline_completion(
        completion_manifest,
        freeze,
        artifacts_root=tmp_path / "artifacts",
    )
    assert completion.status == "COMPLETE_PRECONDITIONS_VERIFIED"
    assert completion.oracle_read_by_v03_driver is False
    assert set(completion.stage_receipt_digests) == set(PRODUCTION_STAGE_IDS)
    assert sha256_file(path) == file_sha
    assert resumed is False
    assert not (layout.run_root / "optional_extensions").exists()
    assert not (layout.run_root / "v04_development").exists()

    resumed_receipt, _, resumed_sha, resumed = verify_pipeline_completion(
        completion_manifest,
        freeze,
        artifacts_root=tmp_path / "artifacts",
        resume=True,
    )
    assert resumed is True
    assert resumed_receipt.receipt_digest == completion.receipt_digest
    assert resumed_sha == file_sha

    completion_manifest_path = tmp_path / "completion-manifest.json"
    freeze_path = tmp_path / "formal-freeze.json"
    _write_json(completion_manifest_path, completion_manifest.to_dict())
    _write_json(freeze_path, freeze.to_dict())
    assert (
        main(
            [
                "complete",
                "--completion-manifest",
                str(completion_manifest_path),
                "--freeze-manifest",
                str(freeze_path),
                "--artifacts-root",
                str(tmp_path / "artifacts"),
                "--resume",
            ]
        )
        == 0
    )
    cli_result = json.loads(capsys.readouterr().out)
    assert cli_result["status"] == "COMPLETE_RESUMED"
    assert cli_result["payload"]["oracle_read_by_v03_driver"] is False

    mismatched_statistics = replace(
        statistics,
        public_ranking_barrier_digest=_d("different-barrier"),
    )
    mismatched_path = tmp_path / "external" / "mismatched-statistics.json"
    _write_json(mismatched_path, mismatched_statistics.to_dict())
    mismatched_external = tuple(
        StageInputBinding(
            input_id=item.input_id,
            path=(
                str(mismatched_path)
                if item.input_id == "statistics-result"
                else item.path
            ),
            sha256=(
                sha256_file(mismatched_path)
                if item.input_id == "statistics-result"
                else item.sha256
            ),
        )
        for item in completion_manifest.external_artifacts
    )
    with pytest.raises(StageExecutionError, match="exact ranking barrier"):
        verify_pipeline_completion(
            replace(completion_manifest, external_artifacts=mismatched_external),
            freeze,
            artifacts_root=tmp_path / "artifacts",
            resume=True,
        )

    mismatched_publication = replace(
        signal_publication,
        signal_extraction_plan_digest=_d("different-signal-extraction-plan"),
        publication_digest=None,
    )
    mismatched_publication_path = (
        tmp_path / "external" / "mismatched-preoracle-publication.json"
    )
    _write_json(mismatched_publication_path, mismatched_publication.to_dict())
    mismatched_external = tuple(
        StageInputBinding(
            input_id=item.input_id,
            path=(
                str(mismatched_publication_path)
                if item.input_id == "pre-oracle-signal-manifest"
                else item.path
            ),
            sha256=(
                sha256_file(mismatched_publication_path)
                if item.input_id == "pre-oracle-signal-manifest"
                else item.sha256
            ),
        )
        for item in completion_manifest.external_artifacts
    )
    with pytest.raises(StageExecutionError, match="pre-oracle signal publication"):
        verify_pipeline_completion(
            replace(completion_manifest, external_artifacts=mismatched_external),
            freeze,
            artifacts_root=tmp_path / "artifacts",
            resume=True,
        )

    mismatched_claim = replace(
        claim, cost_ledger_digest=_d("different-cost-ledger"), audit_digest=None
    )
    mismatched_claim_path = tmp_path / "external" / "mismatched-claim.json"
    _write_json(mismatched_claim_path, mismatched_claim.to_dict())
    mismatched_external = tuple(
        StageInputBinding(
            input_id=item.input_id,
            path=(str(mismatched_claim_path) if item.input_id == "formal-claim-audit" else item.path),
            sha256=(sha256_file(mismatched_claim_path) if item.input_id == "formal-claim-audit" else item.sha256),
        )
        for item in completion_manifest.external_artifacts
    )
    with pytest.raises(StageExecutionError, match="formal claim audit"):
        verify_pipeline_completion(
            replace(completion_manifest, external_artifacts=mismatched_external),
            freeze,
            artifacts_root=tmp_path / "artifacts",
            resume=True,
        )

    swapped_readout_body = dict(private_readout)
    swapped_readout_body.pop("bundle_digest")
    swapped_readout_body["atlas_run_digest"] = _d("swapped-atlas-run")
    swapped_readout = {
        **swapped_readout_body,
        "bundle_digest": sha256_json(swapped_readout_body),
    }
    swapped_readout_path = tmp_path / "external" / "swapped-readout.json"
    _write_json(swapped_readout_path, swapped_readout)
    swapped_external = tuple(
        StageInputBinding(
            input_id=item.input_id,
            path=(str(swapped_readout_path) if item.input_id == "formal-signal-readout-bundle" else item.path),
            sha256=(sha256_file(swapped_readout_path) if item.input_id == "formal-signal-readout-bundle" else item.sha256),
        )
        for item in completion_manifest.external_artifacts
    )
    with pytest.raises(StageExecutionError, match="build-signal-atlas stage receipt"):
        verify_pipeline_completion(
            replace(completion_manifest, external_artifacts=swapped_external),
            freeze,
            artifacts_root=tmp_path / "artifacts",
            resume=True,
        )
