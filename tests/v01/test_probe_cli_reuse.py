from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import policy_learnware_v0.v01.cli as cli
import policy_learnware_v0.v01.probe as v01_probe
import policy_learnware_v0.v01.variant_env as variant_env
from policy_learnware_v0.probe.dataset import EpisodeDataset
from policy_learnware_v0.v01.artifacts import V01ArtifactLayout
from policy_learnware_v0.v01.schemas import (
    EnvironmentInstanceRecord,
    MeasurementSchemaView,
    PrivateContextRecord,
    ShiftManifest,
)
from policy_learnware_v0.v01.seeds import V01SeedPlan


VARIANT_ID = "v01v-22222222222222222222"
PROJECT_SEED = 417
TASK = "SyntheticTask"
BASE_PROTOCOL_ID = "8" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _finite_summary() -> dict[str, object]:
    return {
        "schema": "policy-learnware.v01-finite-termination-audit.v0",
        "episode_count": 2,
        "steps_per_episode": 3,
        "all_finite": True,
        "no_early_termination": True,
        "reward_minimum": -1.0,
        "reward_maximum": 1.0,
        "passed": True,
        "reason": None,
    }


def _dataset(
    *, reset_seeds: tuple[int, ...], probe_seeds: tuple[int, ...]
) -> EpisodeDataset:
    episode_count = len(reset_seeds)
    horizon = 3
    transition_count = episode_count * horizon
    truncated = np.zeros(transition_count, dtype=np.bool_)
    truncated[horizon - 1 :: horizon] = True
    observation = np.arange(transition_count * 2, dtype=np.float32).reshape(
        transition_count, 2
    )
    return EpisodeDataset(
        observation=observation,
        action=np.zeros((transition_count, 1), dtype=np.float32),
        reward=np.linspace(-1.0, 1.0, transition_count, dtype=np.float32),
        next_observation=observation + np.float32(0.25),
        terminated=np.zeros(transition_count, dtype=np.bool_),
        truncated=truncated,
        episode_offsets=np.arange(episode_count + 1, dtype=np.int64) * horizon,
        reset_seeds=np.asarray(reset_seeds, dtype=np.int64),
        probe_seeds=np.asarray(probe_seeds, dtype=np.int64),
    )


class _Adapter:
    def __init__(
        self,
        record: EnvironmentInstanceRecord,
        view: MeasurementSchemaView,
    ) -> None:
        self._record = record
        self.measurement_schema_view = view

    def create_instance_record(
        self, *, finite_termination_audit_summary: dict[str, object]
    ) -> EnvironmentInstanceRecord:
        assert finite_termination_audit_summary == _finite_summary()
        return self._record


class _Harness:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.layout = V01ArtifactLayout(root, "probe-reuse")
        self.view = MeasurementSchemaView(
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
            task=TASK,
            shift_id="damping",
            factor=1.0,
            context_token=b"c" * 16,
            nonce_token=b"n" * 32,
        )
        shift = ShiftManifest.create(
            shift_id="damping",
            factor=1.0,
            registry_digest="6" * 64,
            base_protocol_id=BASE_PROTOCOL_ID,
            task=TASK,
            private_context_id=context.private_context_id,
        )
        self.contexts = {
            "schema": "policy-learnware.v01-private-context-map.v0",
            "experiment_id": self.layout.experiment_id,
            "entries": [
                {
                    "context": context.to_dict(),
                    "shift_manifest": shift.to_dict(),
                    "shift_manifest_digest": shift.digest,
                    "variant_id": VARIANT_ID,
                }
            ],
        }
        self.record = EnvironmentInstanceRecord(
            variant_id=VARIANT_ID,
            env_schema_digest="1" * 64,
            measurement_schema_view_digest=self.view.digest,
            shift_manifest_digest=shift.digest,
            base_model_digest="2" * 64,
            shifted_model_digest="3" * 64,
            changed_leaf="sys.dof_damping",
            changed_index_count=2,
            before_leaf_digest="4" * 64,
            after_leaf_digest="5" * 64,
            operator_digest="7" * 64,
            runtime_versions={"python": "3.12.13"},
            finite_termination_audit_summary=_finite_summary(),
        )
        _write_json(self.layout.run_manifest, {"schema": "synthetic-run"})
        _write_json(
            self.layout.base_protocol_ref,
            {"schema": "synthetic-base-ref", "protocol_id": BASE_PROTOCOL_ID},
        )
        _write_json(
            self.layout.measurement_contract,
            {
                "schema": "policy-learnware.v01-measurement-contract.v0",
                "measurement_protocol_id": "9" * 64,
                "base_protocol_id": BASE_PROTOCOL_ID,
                "probe_banks": 2,
                "episodes_per_bank": 2,
                "prefix_grid": [1, 2],
                "gate_prefix": 2,
                "pair_plan_digest": "a" * 64,
                "variant_ids": [VARIANT_ID],
                "schema_view_digests": {VARIANT_ID: self.view.digest},
                "visibility": "opaque_variant_only_no_context_policy_or_outcome",
            },
        )
        _write_json(
            self.layout.measurement_run_ref,
            {"schema_view_digests": {VARIANT_ID: self.view.digest}},
        )
        _write_json(
            self.layout.instance_record(TASK, VARIANT_ID), self.record.to_dict()
        )

        self.factory_calls: list[dict[str, Any]] = []
        self.executor_calls: list[dict[str, Any]] = []
        self.collect_calls: list[dict[str, Any]] = []
        harness = self

        class _Factory:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def create(self, **kwargs: Any) -> _Adapter:
                harness.factory_calls.append(dict(kwargs))
                return _Adapter(harness.record, harness.view)

        class _Executor:
            def __init__(self, adapter: _Adapter, *, episode_count: int) -> None:
                self.adapter = adapter
                self.episode_count = episode_count
                harness.executor_calls.append(
                    {
                        "adapter": adapter,
                        "executor": self,
                        "episode_count": episode_count,
                    }
                )

        def collect_probe_batch(
            adapter: _Adapter,
            *,
            reset_seeds: list[int],
            probe_seeds: list[int],
            executor: _Executor,
            **_kwargs: Any,
        ) -> EpisodeDataset:
            harness.collect_calls.append(
                {
                    "adapter": adapter,
                    "executor": executor,
                    "reset_seeds": tuple(reset_seeds),
                    "probe_seeds": tuple(probe_seeds),
                }
            )
            return _dataset(
                reset_seeds=tuple(reset_seeds),
                probe_seeds=tuple(probe_seeds),
            )

        monkeypatch.setattr(cli, "_full_layout", lambda _args: self.layout)
        monkeypatch.setattr(
            cli,
            "_frozen_state",
            lambda _layout: (
                {"project_seed": PROJECT_SEED},
                self.contexts,
                {"registry": {}},
            ),
        )
        monkeypatch.setattr(cli, "_require_private_gate0", lambda _layout: {})
        monkeypatch.setattr(
            cli.ShiftRegistry,
            "from_dict",
            classmethod(lambda _cls, _value: object()),
        )
        monkeypatch.setattr(variant_env, "VariantEnvFactory", _Factory)
        monkeypatch.setattr(v01_probe, "ProbeBatchExecutor", _Executor)
        monkeypatch.setattr(v01_probe, "collect_probe_batch", collect_probe_batch)

    def clear_calls(self) -> None:
        self.factory_calls.clear()
        self.executor_calls.clear()
        self.collect_calls.clear()

    def run(self, *, resume: bool) -> dict[str, Any]:
        return cli._collect_probes(
            SimpleNamespace(
                artifacts_root=self.layout.artifacts_root,
                experiment_id=self.layout.experiment_id,
                shard_index=None,
                shard_count=None,
                resume=resume,
            )
        )


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    return _Harness(tmp_path, monkeypatch)


def _assert_one_unjitted_adapter(harness: _Harness) -> None:
    assert len(harness.factory_calls) == 1
    assert harness.factory_calls[0]["variant_id"] == VARIANT_ID
    assert harness.factory_calls[0]["jit"] is False


def test_collect_probes_reuses_one_adapter_and_executor_across_two_banks(
    harness: _Harness,
) -> None:
    assert harness.run(resume=False) == {
        "work_units": 2,
        "written": 2,
        "resumed": 0,
    }
    _assert_one_unjitted_adapter(harness)
    assert len(harness.executor_calls) == 1
    assert harness.executor_calls[0]["episode_count"] == 2
    assert len(harness.collect_calls) == 2
    assert harness.collect_calls[0]["adapter"] is harness.collect_calls[1]["adapter"]
    assert harness.collect_calls[0]["executor"] is harness.collect_calls[1]["executor"]
    assert harness.collect_calls[0]["executor"] is harness.executor_calls[0]["executor"]

    plan = V01SeedPlan(PROJECT_SEED)
    for bank, call in enumerate(harness.collect_calls):
        expected = [plan.probe_episode(TASK, bank, index) for index in range(2)]
        assert call["reset_seeds"] == tuple(item.reset_seed for item in expected)
        assert call["probe_seeds"] == tuple(item.probe_seed for item in expected)


def test_collect_probes_complete_resume_builds_no_executor_and_does_no_rollout(
    harness: _Harness,
) -> None:
    harness.run(resume=False)
    harness.clear_calls()

    assert harness.run(resume=True) == {
        "work_units": 2,
        "written": 0,
        "resumed": 2,
    }
    _assert_one_unjitted_adapter(harness)
    assert harness.executor_calls == []
    assert harness.collect_calls == []


def test_collect_probes_partial_resume_fails_closed_before_executor_or_rollout(
    harness: _Harness,
) -> None:
    harness.run(resume=False)
    harness.layout.collection_attestation(VARIANT_ID, 1).unlink()
    harness.clear_calls()

    with pytest.raises(cli.V01CommandFailure, match="incomplete probe"):
        harness.run(resume=True)
    _assert_one_unjitted_adapter(harness)
    assert harness.executor_calls == []
    assert harness.collect_calls == []
