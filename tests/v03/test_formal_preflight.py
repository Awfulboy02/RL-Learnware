from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path

import pytest

from policy_learnware_v0.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from policy_learnware_v0.v03.cli import main
from policy_learnware_v0.v03.costs import frozen_cost_protocol_digest
from policy_learnware_v0.v03.formal_preflight import (
    FORMAL_LAUNCH_REVIEW_AUTHORITY_SCHEMA,
    FormalLaunchPreflightReport,
    FormalPipelineLaunchManifest,
    FormalPreflightError,
    FormalStageLaunchManifest,
    formal_freeze_authorization_surface_digest,
    formal_pipeline_launch_surface_digest,
    verify_formal_launch_preflight,
    verify_formal_launch_preflight_from_files,
)
from policy_learnware_v0.v03.orchestration import (
    PRODUCTION_STAGE_IDS,
    REQUIRED_PREDECESSOR,
    FormalStageRequestTemplate,
    StageInputBinding,
)
from policy_learnware_v0.v03.preflight import (
    HARD_TODO_IDS,
    HardTodoEvidence,
    PreExperimentFreezeManifest,
    formal_stage_adapter_binding_digest,
)


def _d(label: str) -> str:
    return sha256_json({"formal-preflight-test": label})


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _todo(todo_id: str) -> HardTodoEvidence:
    return HardTodoEvidence(
        todo_id=todo_id,
        contract_digest=_d(f"{todo_id}:contract"),
        implementation_digest=_d(f"{todo_id}:implementation"),
        unit_test_evidence_digest=_d(f"{todo_id}:unit"),
        synthetic_fixture_evidence_digest=_d(f"{todo_id}:fixture"),
        cpu_integration_evidence_digest=_d(f"{todo_id}:cpu"),
    )


def _freeze(
    *,
    authority: bool,
    authority_receipt_digest: str,
    templates: dict[str, str],
    adapters: dict[str, str],
) -> PreExperimentFreezeManifest:
    return PreExperimentFreezeManifest(
        freeze_id="formal-preflight-test-freeze",
        config_bytes_digest=_d("config"),
        implementation_tree_digest=_d("tree"),
        clean_commit_digest=_d("commit"),
        review_decisions_digest=_d("review"),
        review_authority_receipt_digest=(
            authority_receipt_digest if authority else None
        ),
        review_authority_verified=authority,
        encoder_extension_gate_enabled=False,
        data_role_manifest_digest=_d("roles"),
        canonicalizer_registry_digest=_d("canonicalizer"),
        signal_matrix_digest=_d("matrix"),
        signal_contrast_plan_digest=_d("contrast"),
        signal_materiality_threshold_digest=_d("materiality"),
        formal_signal_readout_plan_digest=_d("readout"),
        preoracle_signal_outcome_plan_digest=_d("preoracle"),
        signal_identity_registry_digest=_d("identity"),
        signal_execution_protocol_digest=_d("execution"),
        representation_plan_digest=_d("representation"),
        condition_plan_digest=_d("condition"),
        formal_source_fit_schedule_digest=_d("source-fit"),
        formal_source_membership_digest=_d("source-membership"),
        signal_work_item_graph_digest=_d("work-graph"),
        formal_signal_prefix_schedule_digest=_d("prefix"),
        dynamics_axis_registry_digest=_d("dynamics"),
        public_query_plan_digest=_d("query"),
        baseline_plan_digest=_d("baseline"),
        statistics_plan_digest=_d("statistics"),
        cost_protocol_digest=frozen_cost_protocol_digest(),
        source_reduced_query_empirical_protocol_digest=_d("asymmetric-kme"),
        formal_gate_plan_digests=(
            {
                "G03-Attribution": _d("attribution-plan"),
                "G03-Probe": _d("probe-plan"),
                "G03-Market": _d("market-plan"),
            }
            if authority
            else {}
        ),
        formal_stage_request_template_digests=templates if authority else {},
        hard_todo_evidence=tuple(_todo(todo_id) for todo_id in HARD_TODO_IDS),
        formal_stage_adapter_binding_digests=adapters if authority else {},
    )


def _case(
    tmp_path: Path,
    *,
    authority: bool = True,
) -> tuple[
    FormalPipelineLaunchManifest,
    PreExperimentFreezeManifest,
    Path,
    Path,
    tuple[Path, ...],
]:
    stages = []
    template_digests: dict[str, str] = {}
    adapter_digests: dict[str, str] = {}
    static_paths = []
    authority_path = (tmp_path / "review" / "authority.json").resolve()
    adapter_module_root = (tmp_path / "adapter-module-root").resolve()
    for index, stage_id in enumerate(PRODUCTION_STAGE_IDS):
        static_path = (tmp_path / "static" / f"{index:02d}.json").resolve()
        _write_json(static_path, {"stage_id": stage_id, "index": index})
        static_paths.append(static_path)
        input_binding = StageInputBinding(
            input_id=f"static-{index:02d}",
            path=str(static_path),
            sha256=sha256_file(static_path),
        )
        adapter_id = f"formal-preflight-adapter-{index:02d}"
        adapter_contract_digest = _d(f"adapter-contract:{stage_id}")
        template = FormalStageRequestTemplate(
            stage_id=stage_id,
            adapter_id=adapter_id,
            adapter_contract_digest=adapter_contract_digest,
            parameters_digest=_d(f"parameters:{stage_id}"),
            static_input_content_digests={
                input_binding.input_id: input_binding.sha256
            },
        )
        template_path = (tmp_path / "templates" / f"{index:02d}.json").resolve()
        _write_json(template_path, template.to_dict())
        adapter_source_path = (
            adapter_module_root
            / "reviewed"
            / "adapters"
            / f"adapter_{index:02d}.py"
        ).resolve()
        adapter_source_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_source_path.write_text(
            f'"""Reviewed fixture adapter source for {stage_id}."""\n',
            encoding="utf-8",
        )
        template_digests[stage_id] = template.request_template_digest
        adapter_digests[stage_id] = formal_stage_adapter_binding_digest(
            stage_id,
            adapter_id,
            adapter_contract_digest,
        )
        stages.append(
            FormalStageLaunchManifest(
                stage_id=stage_id,
                execution_id=f"formal-preflight-execution-{index:02d}",
                request_template_path=str(template_path),
                request_template_file_sha256=sha256_file(template_path),
                adapter_source_path=str(adapter_source_path),
                adapter_source_file_sha256=sha256_file(adapter_source_path),
                adapter_entrypoint=f"reviewed.adapters.adapter_{index:02d}:create",
                static_inputs=(input_binding,),
                predecessor_stage_ids=tuple(REQUIRED_PREDECESSOR[stage_id]),
            )
        )
    draft_freeze = _freeze(
        authority=True,
        authority_receipt_digest=_d("draft-authority-receipt"),
        templates=template_digests,
        adapters=adapter_digests,
    )
    authority_record = {
        "schema": FORMAL_LAUNCH_REVIEW_AUTHORITY_SCHEMA,
        "decision": "AUTHORIZED",
        "authority_id": "formal-preflight-test-reviewer",
        "formal_freeze_authorization_surface_digest": (
            formal_freeze_authorization_surface_digest(draft_freeze)
        ),
        "formal_launch_surface_digest": formal_pipeline_launch_surface_digest(
            run_id="formal-preflight-run",
            adapter_module_root=adapter_module_root,
            stages=tuple(stages),
        ),
    }
    _write_json(authority_path, authority_record)
    freeze = _freeze(
        authority=authority,
        authority_receipt_digest=sha256_json(authority_record),
        templates=template_digests,
        adapters=adapter_digests,
    )
    launch = FormalPipelineLaunchManifest(
        run_id="formal-preflight-run",
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        review_authority_receipt_path=str(authority_path),
        review_authority_receipt_file_sha256=sha256_file(authority_path),
        adapter_module_root=str(adapter_module_root),
        stages=tuple(stages),
    )
    freeze_path = (tmp_path / "freeze.json").resolve()
    launch_path = (tmp_path / "launch.json").resolve()
    _write_json(freeze_path, freeze.to_dict())
    _write_json(launch_path, launch.to_dict())
    return launch, freeze, launch_path, freeze_path, tuple(static_paths)


def test_formal_preflight_checks_all_static_bytes_without_writes(tmp_path: Path) -> None:
    launch, freeze, launch_path, freeze_path, _ = _case(tmp_path)
    before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    report = verify_formal_launch_preflight(launch, freeze)
    from_files = verify_formal_launch_preflight_from_files(
        launch_manifest_path=launch_path,
        freeze_manifest_path=freeze_path,
    )
    after = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))

    assert report == from_files
    assert report.status == "FORMAL_STATIC_BINDINGS_READY"
    assert report.stage_count == 11
    assert report.checked_request_template_file_count == 11
    assert report.checked_adapter_source_file_count == 11
    assert report.checked_static_input_file_count == 11
    assert report.adapter_executed is False
    assert report.artifacts_written is False
    assert FormalLaunchPreflightReport.from_dict(report.to_dict()) == report
    assert after == before


def test_formal_preflight_cli_is_read_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, launch_path, freeze_path, _ = _case(tmp_path)

    def reject_adapter_import(name: str, package: str | None = None) -> None:
        raise AssertionError(f"formal preflight attempted an import: {name!r}")

    monkeypatch.setattr(importlib, "import_module", reject_adapter_import)
    assert (
        main(
            [
                "preflight-formal",
                "--manifest",
                str(launch_path),
                "--freeze-manifest",
                str(freeze_path),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "FORMAL_STATIC_BINDINGS_READY"
    assert output["payload"]["adapter_executed"] is False
    assert output["payload"]["artifacts_written"] is False


def test_formal_preflight_rejects_unverified_freeze(tmp_path: Path) -> None:
    launch, freeze, _, _, _ = _case(tmp_path, authority=False)
    with pytest.raises(FormalPreflightError, match="external.*authority"):
        verify_formal_launch_preflight(launch, freeze)


def test_formal_preflight_rejects_static_input_byte_drift(tmp_path: Path) -> None:
    launch, freeze, _, _, static_paths = _case(tmp_path)
    _write_json(static_paths[3], {"tampered": True})
    with pytest.raises(FormalPreflightError, match="static input byte digest mismatch"):
        verify_formal_launch_preflight(launch, freeze)


def test_formal_preflight_rejects_adapter_or_authority_byte_drift(
    tmp_path: Path,
) -> None:
    launch, freeze, _, _, _ = _case(tmp_path)
    adapter_path = Path(launch.stages[4].adapter_source_path)
    adapter_path.write_text("# changed adapter bytes\n", encoding="utf-8")
    with pytest.raises(FormalPreflightError, match="adapter source byte digest"):
        verify_formal_launch_preflight(launch, freeze)

    launch, freeze, _, _, _ = _case(tmp_path / "authority-case")
    authority_path = Path(launch.review_authority_receipt_path)
    _write_json(authority_path, {"decision": "changed"})
    with pytest.raises(FormalPreflightError, match="authority receipt byte drift"):
        verify_formal_launch_preflight(launch, freeze)


def test_external_authority_binds_adapter_source_launch_surface(
    tmp_path: Path,
) -> None:
    launch, freeze, _, _, _ = _case(tmp_path)
    source_path = Path(launch.stages[0].adapter_source_path)
    source_path.write_text("# attacker-controlled replacement\n", encoding="utf-8")
    rebound_stage = replace(
        launch.stages[0],
        adapter_source_file_sha256=sha256_file(source_path),
        manifest_digest=None,
    )
    rebound_launch = replace(
        launch,
        stages=(rebound_stage, *launch.stages[1:]),
        manifest_digest=None,
    )
    with pytest.raises(FormalPreflightError, match="binds another launch surface"):
        verify_formal_launch_preflight(rebound_launch, freeze)


@pytest.mark.parametrize(
    "surface_change",
    (
        {"config_bytes_digest": _d("attacker-config")},
        {"implementation_tree_digest": _d("attacker-implementation")},
        {
            "formal_gate_plan_digests": {
                "G03-Attribution": _d("attacker-attribution-plan"),
                "G03-Probe": _d("probe-plan"),
                "G03-Market": _d("market-plan"),
            }
        },
        {"public_query_plan_digest": _d("attacker-query-plan")},
        {"statistics_plan_digest": _d("attacker-statistics-plan")},
    ),
    ids=("config", "implementation", "gate", "query", "statistics"),
)
def test_external_authority_binds_freeze_authorization_surface(
    tmp_path: Path,
    surface_change: dict[str, object],
) -> None:
    launch, freeze, _, _, _ = _case(tmp_path)
    rebound_freeze = replace(freeze, **surface_change)
    rebound_launch = replace(
        launch,
        freeze_manifest_digest=rebound_freeze.freeze_manifest_digest,
        manifest_digest=None,
    )

    with pytest.raises(
        FormalPreflightError,
        match="binds another freeze authorization surface",
    ):
        verify_formal_launch_preflight(rebound_launch, rebound_freeze)


def test_formal_preflight_binds_entrypoint_module_to_reviewed_source(
    tmp_path: Path,
) -> None:
    launch, freeze, _, _, _ = _case(tmp_path)
    changed_stage = replace(
        launch.stages[3],
        adapter_entrypoint="reviewed.adapters.adapter_04:create",
        manifest_digest=None,
    )
    changed = replace(
        launch,
        stages=(*launch.stages[:3], changed_stage, *launch.stages[4:]),
        manifest_digest=None,
    )
    with pytest.raises(
        FormalPreflightError,
        match="binds another launch surface",
    ):
        verify_formal_launch_preflight(changed, freeze)


def test_formal_preflight_rejects_symlinked_entrypoint_tree(tmp_path: Path) -> None:
    launch, freeze, _, _, _ = _case(tmp_path)
    module_root = Path(launch.adapter_module_root)
    reviewed = module_root / "reviewed"
    reviewed.rename(module_root / "real-reviewed")
    reviewed.symlink_to(module_root / "real-reviewed", target_is_directory=True)

    with pytest.raises(FormalPreflightError, match="regular directories"):
        verify_formal_launch_preflight(launch, freeze)


def test_formal_preflight_rejects_template_or_topology_drift(tmp_path: Path) -> None:
    launch, freeze, _, _, _ = _case(tmp_path)
    first_template = Path(launch.stages[0].request_template_path)
    template_payload = json.loads(first_template.read_text(encoding="utf-8"))
    template_payload["parameters_digest"] = _d("tampered-parameters")
    _write_json(first_template, template_payload)
    updated_first = replace(
        launch.stages[0],
        request_template_file_sha256=sha256_file(first_template),
        manifest_digest=None,
    )
    changed = replace(
        launch,
        stages=(updated_first, *launch.stages[1:]),
        manifest_digest=None,
    )
    with pytest.raises(FormalPreflightError, match="binds another launch surface"):
        verify_formal_launch_preflight(changed, freeze)

    with pytest.raises(FormalPreflightError, match="exact predecessor topology"):
        replace(
            launch.stages[1],
            predecessor_stage_ids=(),
            manifest_digest=None,
        )


def test_persisted_launch_manifests_reject_null_digest(tmp_path: Path) -> None:
    launch, freeze, _, _, _ = _case(tmp_path)
    stage_payload = launch.stages[0].to_dict()
    stage_payload["manifest_digest"] = None
    with pytest.raises(FormalPreflightError, match="requires manifest_digest"):
        FormalStageLaunchManifest.from_dict(stage_payload)

    launch_payload = launch.to_dict()
    launch_payload["manifest_digest"] = None
    with pytest.raises(FormalPreflightError, match="requires manifest_digest"):
        FormalPipelineLaunchManifest.from_dict(launch_payload)

    report = verify_formal_launch_preflight(launch, freeze)
    report_payload = report.to_dict()
    report_payload["report_digest"] = None
    with pytest.raises(FormalPreflightError, match="requires report_digest"):
        FormalLaunchPreflightReport.from_dict(report_payload)


def test_later_stage_may_have_no_static_inputs_but_first_stage_may_not(
    tmp_path: Path,
) -> None:
    launch, _, _, _, _ = _case(tmp_path)
    last = launch.stages[-1]
    template_path = Path(last.request_template_path)
    template = FormalStageRequestTemplate.from_dict(
        json.loads(template_path.read_text(encoding="utf-8"))
    )
    empty_template = replace(template, static_input_content_digests={})
    _write_json(template_path, empty_template.to_dict())
    empty_last = replace(
        last,
        request_template_file_sha256=sha256_file(template_path),
        static_inputs=(),
        manifest_digest=None,
    )
    partly_static = replace(
        launch,
        stages=(*launch.stages[:-1], empty_last),
        manifest_digest=None,
    )
    assert partly_static.stages[-1].static_inputs == ()

    first_empty = replace(
        partly_static.stages[0],
        static_inputs=(),
        manifest_digest=None,
    )
    with pytest.raises(
        FormalPreflightError,
        match="first formal stage requires at least one static input",
    ):
        replace(
            partly_static,
            stages=(first_empty, *partly_static.stages[1:]),
            manifest_digest=None,
        )
