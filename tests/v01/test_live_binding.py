from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import policy_learnware_v0.v01.cli as cli
import policy_learnware_v0.v01.probe as v01_probe
import policy_learnware_v0.v01.variant_env as variant_env
from policy_learnware_v0.hashing import sha256_file
from policy_learnware_v0.probe.dataset import EpisodeDataset
from policy_learnware_v0.v01.live_binding import (
    LiveInstanceBinding,
    LiveInstanceBindingError,
    build_collection_binding_attestation,
    verify_collection_binding_attestation,
    verify_live_instance_binding,
)
from policy_learnware_v0.v01.recompute import verify_private_collection_bindings
from policy_learnware_v0.v01.artifacts import V01ArtifactLayout
from policy_learnware_v0.v01.seeds import V01SeedPlan
from policy_learnware_v0.v01.schemas import (
    EnvironmentInstanceRecord,
    MeasurementSchemaView,
    PrivateContextRecord,
    ShiftManifest,
    VariantDatasetManifest,
)


VARIANT_ID = "v01v-11111111111111111111"


def _finite_summary(*, passed: bool = True) -> dict[str, object]:
    return {
        "schema": "policy-learnware.v01-finite-termination-audit.v0",
        "episode_count": 4,
        "steps_per_episode": 3,
        "all_finite": passed,
        "no_early_termination": passed,
        "reward_minimum": -1.0,
        "reward_maximum": 2.0,
        "passed": passed,
        "reason": None if passed else "synthetic failure",
    }


def _record(
    *,
    finite_passed: bool = True,
    shifted_model_digest: str = "3" * 64,
    schema_digest: str = "a" * 64,
    shift_manifest_digest: str = "b" * 64,
) -> EnvironmentInstanceRecord:
    return EnvironmentInstanceRecord(
        variant_id=VARIANT_ID,
        env_schema_digest="1" * 64,
        measurement_schema_view_digest=schema_digest,
        shift_manifest_digest=shift_manifest_digest,
        base_model_digest="2" * 64,
        shifted_model_digest=shifted_model_digest,
        changed_leaf="sys.dof_damping",
        changed_index_count=2,
        before_leaf_digest="4" * 64,
        after_leaf_digest="5" * 64,
        operator_digest="6" * 64,
        runtime_versions={"python": "3.12.13"},
        finite_termination_audit_summary=_finite_summary(passed=finite_passed),
    )


def _dataset(
    *,
    reset_seeds: tuple[int, int] = (10, 11),
    probe_seeds: tuple[int, int] = (20, 21),
) -> EpisodeDataset:
    transitions = 6
    return EpisodeDataset(
        observation=np.arange(transitions * 2, dtype=np.float32).reshape(transitions, 2),
        action=np.zeros((transitions, 1), dtype=np.float32),
        reward=np.linspace(-1.0, 1.0, transitions, dtype=np.float32),
        next_observation=np.arange(
            1, transitions * 2 + 1, dtype=np.float32
        ).reshape(transitions, 2),
        terminated=np.zeros(transitions, dtype=np.bool_),
        truncated=np.asarray([False, False, True, False, False, True]),
        episode_offsets=np.asarray([0, 3, 6], dtype=np.int64),
        reset_seeds=np.asarray(reset_seeds, dtype=np.int64),
        probe_seeds=np.asarray(probe_seeds, dtype=np.int64),
    )


class _Adapter:
    def __init__(
        self,
        record: EnvironmentInstanceRecord,
        view: MeasurementSchemaView | None = None,
    ):
        self.record = record
        self.measurement_schema_view = view
        self.received_finite: dict[str, object] | None = None

    def create_instance_record(self, *, finite_termination_audit_summary):
        self.received_finite = dict(finite_termination_audit_summary)
        return self.record


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_live_instance_binding_is_exact_and_fails_on_model_or_finite_drift() -> None:
    record = _record()
    adapter = _Adapter(record)
    binding = verify_live_instance_binding(
        adapter,
        record.to_dict(),
        audited_instance_record_sha256="f" * 64,
    )
    assert binding.passed
    assert binding.verified_instance_digest == record.digest
    assert adapter.received_finite == _finite_summary()

    with pytest.raises(LiveInstanceBindingError, match="shifted_model_digest"):
        verify_live_instance_binding(
            _Adapter(_record(shifted_model_digest="7" * 64)),
            record.to_dict(),
            audited_instance_record_sha256="f" * 64,
        )
    with pytest.raises(LiveInstanceBindingError, match="unconditional pass"):
        failed = _record(finite_passed=False)
        verify_live_instance_binding(
            _Adapter(failed),
            failed.to_dict(),
            audited_instance_record_sha256="f" * 64,
        )


def test_private_attestation_is_rebuilt_not_trusted() -> None:
    record = _record()
    dataset = _dataset()
    binding = LiveInstanceBinding(record, record, "f" * 64)
    attestation = build_collection_binding_attestation(
        binding,
        dataset,
        bank=0,
        expected_episode_count=2,
        expected_horizon=3,
        run_manifest_sha256="e" * 64,
    )
    rebuilt = verify_collection_binding_attestation(
        attestation,
        audited_record=record,
        audited_instance_record_sha256="f" * 64,
        dataset=dataset,
        bank=0,
        expected_episode_count=2,
        expected_horizon=3,
        run_manifest_sha256="e" * 64,
    )
    assert rebuilt == attestation

    poisoned = dict(attestation)
    poisoned["passed"] = False
    with pytest.raises(LiveInstanceBindingError, match="executable evidence"):
        verify_collection_binding_attestation(
            poisoned,
            audited_record=record,
            audited_instance_record_sha256="f" * 64,
            dataset=dataset,
            bank=0,
            expected_episode_count=2,
            expected_horizon=3,
            run_manifest_sha256="e" * 64,
        )


def test_gate_d_private_join_consumes_every_attestation_without_public_digest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal"
    frozen = root / "frozen"
    private = root / "benchmark_private"
    measurement = root / "measurement"
    for path in (frozen, private, measurement):
        path.mkdir(parents=True)

    run_path = frozen / "run_manifest.json"
    _write_json(run_path, {"schema": "synthetic-run"})
    view = MeasurementSchemaView(
        observation_dim=2,
        action_dim=1,
        observation_dtype="float32",
        action_dtype="float32",
        action_low=np.asarray([-1.0], dtype=np.float32),
        action_high=np.asarray([1.0], dtype=np.float32),
        horizon=3,
        action_repeat=1,
        control_dt=0.02,
        flatten_fingerprint_without_task="synthetic-flat-v0",
    )
    _write_json(
        measurement / "schema_views" / f"{view.schema_view_id}.json",
        view.to_dict(),
    )
    contract = {
        "schema": "policy-learnware.v01-measurement-contract.v0",
        "measurement_protocol_id": "9" * 64,
        "base_protocol_id": "8" * 64,
        "probe_banks": 1,
        "episodes_per_bank": 2,
        "prefix_grid": [1, 2],
        "gate_prefix": 2,
        "pair_plan_digest": "7" * 64,
        "variant_ids": [VARIANT_ID],
        "schema_view_digests": {VARIANT_ID: view.digest},
        "visibility": "opaque_variant_only_no_context_policy_or_outcome",
    }
    contract_path = measurement / "measurement_contract.json"
    _write_json(contract_path, contract)

    context = PrivateContextRecord.new(
        task="SyntheticTask",
        shift_id="damping",
        factor=1.0,
        context_token=b"c" * 16,
        nonce_token=b"n" * 32,
    )
    shift = ShiftManifest.create(
        shift_id="damping",
        factor=1.0,
        registry_digest="6" * 64,
        base_protocol_id="8" * 64,
        task=context.task,
        private_context_id=context.private_context_id,
    )
    contexts = {
        "schema": "policy-learnware.v01-private-context-map.v0",
        "experiment_id": "formal",
        "entries": [
            {
                "context": context.to_dict(),
                "shift_manifest": shift.to_dict(),
                "shift_manifest_digest": shift.digest,
                "variant_id": VARIANT_ID,
            }
        ],
    }
    _write_json(private / "contexts.json", contexts)
    variant_dir = private / "variants" / context.task / VARIANT_ID
    _write_json(variant_dir / "shift_manifest.json", shift.to_dict())
    record = _record(
        schema_digest=view.digest,
        shift_manifest_digest=shift.digest,
    )
    instance_path = variant_dir / "instance.json"
    _write_json(instance_path, record.to_dict())

    dataset = _dataset()
    dataset_path = measurement / "datasets" / VARIANT_ID / "bank_000" / "dataset.npz"
    dataset.save_npz(dataset_path)
    public_manifest = VariantDatasetManifest(
        variant_id=VARIANT_ID,
        bank=0,
        episode_count=dataset.episode_count,
        transition_count=dataset.transition_count,
        reset_seeds=tuple(int(value) for value in dataset.reset_seeds),
        probe_seeds=tuple(int(value) for value in dataset.probe_seeds),
        dataset_digest=dataset.digest,
        base_protocol_id="8" * 64,
        measurement_contract_digest=sha256_file(contract_path),
        measurement_schema_view_digest=view.digest,
    )
    _write_json(dataset_path.with_name("manifest.json"), public_manifest.to_dict())
    assert "collection_attestation_digest" not in public_manifest.to_dict()

    binding = LiveInstanceBinding(record, record, sha256_file(instance_path))
    attestation = build_collection_binding_attestation(
        binding,
        dataset,
        bank=0,
        expected_episode_count=2,
        expected_horizon=3,
        run_manifest_sha256=sha256_file(run_path),
    )
    attestation_path = (
        private / "collection_attestations" / VARIANT_ID / "bank_000.json"
    )
    _write_json(attestation_path, attestation)

    evidence = verify_private_collection_bindings(
        frozen_root=frozen,
        benchmark_private_root=private,
        measurement_root=measurement,
    )
    assert evidence.passed, evidence.details["errors"]
    assert evidence.details["verified_unit_count"] == 1
    assert evidence.details["public_instance_digest_exposed"] is False

    poisoned = dict(attestation)
    poisoned["verified_instance_digest"] = "0" * 64
    _write_json(attestation_path, poisoned)
    failed = verify_private_collection_bindings(
        frozen_root=frozen,
        benchmark_private_root=private,
        measurement_root=measurement,
    )
    assert not failed.passed
    assert "executable evidence" in " ".join(failed.details["errors"])


def test_cli_verifies_fresh_instance_before_probe_and_oracle_execution() -> None:
    collect_source = inspect.getsource(cli._collect_probes)
    assert collect_source.index("expected_view =") < collect_source.index(
        "live_binding = verify_live_instance_binding"
    )
    assert collect_source.index(
        "live_binding = verify_live_instance_binding"
    ) < collect_source.index("dataset = collect_probe_batch")
    assert collect_source.index(
        "attestation = build_collection_binding_attestation"
    ) < collect_source.index("measurement.publish_npz")
    assert "collection_attestation_digest" not in collect_source

    oracle_source = inspect.getsource(cli._evaluate_oracle)
    assert oracle_source.index(
        "live_binding = verify_live_instance_binding"
    ) < oracle_source.index("if path.is_file() and args.resume")
    assert "instance_digest = live_binding.verified_instance_digest" in oracle_source


def test_collect_probes_resume_revalidates_complete_bundle_without_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = V01ArtifactLayout(tmp_path, "formal")
    project_seed = 123
    task = "SyntheticTask"
    view = MeasurementSchemaView(
        observation_dim=2,
        action_dim=1,
        observation_dtype="float32",
        action_dtype="float32",
        action_low=np.asarray([-1.0], dtype=np.float32),
        action_high=np.asarray([1.0], dtype=np.float32),
        horizon=3,
        action_repeat=1,
        control_dt=0.02,
        flatten_fingerprint_without_task="synthetic-flat-v0",
    )
    context = PrivateContextRecord.new(
        task=task,
        shift_id="damping",
        factor=1.0,
        context_token=b"c" * 16,
        nonce_token=b"n" * 32,
    )
    base_protocol_id = "8" * 64
    shift = ShiftManifest.create(
        shift_id="damping",
        factor=1.0,
        registry_digest="6" * 64,
        base_protocol_id=base_protocol_id,
        task=task,
        private_context_id=context.private_context_id,
    )
    contexts = {
        "schema": "policy-learnware.v01-private-context-map.v0",
        "experiment_id": "formal",
        "entries": [
            {
                "context": context.to_dict(),
                "shift_manifest": shift.to_dict(),
                "shift_manifest_digest": shift.digest,
                "variant_id": VARIANT_ID,
            }
        ],
    }
    _write_json(layout.run_manifest, {"schema": "synthetic-run"})
    _write_json(
        layout.base_protocol_ref,
        {"schema": "synthetic-base-ref", "protocol_id": base_protocol_id},
    )
    contract = {
        "schema": "policy-learnware.v01-measurement-contract.v0",
        "measurement_protocol_id": "9" * 64,
        "base_protocol_id": base_protocol_id,
        "probe_banks": 1,
        "episodes_per_bank": 2,
        "prefix_grid": [1, 2],
        "gate_prefix": 2,
        "pair_plan_digest": "7" * 64,
        "variant_ids": [VARIANT_ID],
        "schema_view_digests": {VARIANT_ID: view.digest},
        "visibility": "opaque_variant_only_no_context_policy_or_outcome",
    }
    _write_json(layout.measurement_contract, contract)
    _write_json(
        layout.measurement_run_ref,
        {"schema_view_digests": {VARIANT_ID: view.digest}},
    )
    record = _record(
        schema_digest=view.digest,
        shift_manifest_digest=shift.digest,
    )
    instance_path = layout.instance_record(task, VARIANT_ID)
    _write_json(instance_path, record.to_dict())

    seed_rows = [
        V01SeedPlan(project_seed).probe_episode(task, 0, index)
        for index in range(2)
    ]
    dataset = _dataset(
        reset_seeds=tuple(row.reset_seed for row in seed_rows),
        probe_seeds=tuple(row.probe_seed for row in seed_rows),
    )
    dataset_path = layout.dataset_npz(VARIANT_ID, 0)
    dataset.save_npz(dataset_path)
    sidecar = VariantDatasetManifest(
        variant_id=VARIANT_ID,
        bank=0,
        episode_count=dataset.episode_count,
        transition_count=dataset.transition_count,
        reset_seeds=tuple(int(value) for value in dataset.reset_seeds),
        probe_seeds=tuple(int(value) for value in dataset.probe_seeds),
        dataset_digest=dataset.digest,
        base_protocol_id=base_protocol_id,
        measurement_contract_digest=sha256_file(layout.measurement_contract),
        measurement_schema_view_digest=view.digest,
    )
    _write_json(layout.dataset_manifest(VARIANT_ID, 0), sidecar.to_dict())
    attestation_path = layout.collection_attestation(VARIANT_ID, 0)
    _write_json(
        attestation_path,
        build_collection_binding_attestation(
            LiveInstanceBinding(record, record, sha256_file(instance_path)),
            dataset,
            bank=0,
            expected_episode_count=2,
            expected_horizon=3,
            run_manifest_sha256=sha256_file(layout.run_manifest),
        ),
    )

    class _Factory:
        def __init__(self, *_args, **_kwargs):
            pass

        def create(self, **_kwargs):
            return _Adapter(record, view)

    rollout_calls = 0

    def forbidden_rollout(*_args, **_kwargs):
        nonlocal rollout_calls
        rollout_calls += 1
        raise AssertionError("resume must not recollect a complete probe bundle")

    monkeypatch.setattr(cli, "_full_layout", lambda _args: layout)
    monkeypatch.setattr(
        cli,
        "_frozen_state",
        lambda _layout: (
            {"project_seed": project_seed},
            contexts,
            {"registry": {}},
        ),
    )
    monkeypatch.setattr(cli, "_require_private_gate0", lambda _layout: {})
    monkeypatch.setattr(
        cli.ShiftRegistry, "from_dict", classmethod(lambda _cls, _value: object())
    )
    monkeypatch.setattr(variant_env, "VariantEnvFactory", _Factory)
    monkeypatch.setattr(v01_probe, "collect_probe_batch", forbidden_rollout)
    args = SimpleNamespace(
        artifacts_root=tmp_path,
        experiment_id="formal",
        shard_index=None,
        shard_count=None,
        resume=True,
    )
    result = cli._collect_probes(args)
    assert result == {"work_units": 1, "written": 0, "resumed": 1}
    assert rollout_calls == 0

    attestation_path.unlink()
    with pytest.raises(cli.V01CommandFailure, match="incomplete probe"):
        cli._collect_probes(args)
    assert rollout_calls == 0
