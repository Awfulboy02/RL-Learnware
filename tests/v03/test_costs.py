from __future__ import annotations

from dataclasses import replace

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.costs import (
    COST_COMPONENT_IDS,
    CostComponentRecord,
    V03CostError,
    V03CostLedger,
    frozen_cost_protocol_digest,
)


def _d(label: str) -> str:
    return sha256_json({"cost-test": label})


def _component(component_id: str) -> CostComponentRecord:
    return CostComponentRecord(
        component_id=component_id,
        measurement_receipt_digest=_d(f"receipt:{component_id}"),
        input_artifact_set_digest=_d(f"input:{component_id}"),
        output_artifact_set_digest=_d(f"output:{component_id}"),
        wall_seconds=1.0,
        gpu_seconds=0.5 if component_id in {"REPRESENTATION_FIT", "ENCODE"} else 0.0,
        peak_memory_bytes=1024,
        artifact_bytes=256,
        environment_steps=64 if component_id == "PROBE_COLLECTION" else 0,
        invocation_count=3 if component_id == "END_TO_END_WARM" else 1,
        device_class="gpu" if component_id in {"REPRESENTATION_FIT", "ENCODE"} else "cpu",
    )


def _formal() -> V03CostLedger:
    return V03CostLedger(
        run_id="formal-cost-test",
        execution_scope="FORMAL",
        freeze_manifest_digest=_d("freeze"),
        cost_protocol_digest=frozen_cost_protocol_digest(),
        prefix_cost_evidence_digest=_d("prefix-costs"),
        components=tuple(_component(item) for item in COST_COMPONENT_IDS),
    )


def test_formal_cost_ledger_has_exact_14_4_coverage_and_round_trips() -> None:
    ledger = _formal()
    assert tuple(item.component_id for item in ledger.components) == COST_COMPONENT_IDS
    assert V03CostLedger.from_dict(ledger.to_dict()) == ledger
    public = ledger.to_public_dict()
    assert public["probe_environment_steps"] == 64
    assert public["peak_memory_bytes"] == 1024
    assert public["private_physical_paths_withheld"] is True
    assert "measurement_receipt_digest" not in public


def test_cost_ledger_rejects_missing_formal_component_and_fake_zero_probe() -> None:
    ledger = _formal()
    with pytest.raises(V03CostError, match="exact component coverage"):
        replace(ledger, components=ledger.components[:-1], ledger_digest=None)
    with pytest.raises(V03CostError, match="positive environment steps"):
        replace(_component("PROBE_COLLECTION"), environment_steps=0)
    with pytest.raises(V03CostError, match="only PROBE_COLLECTION"):
        replace(_component("DISTANCE"), environment_steps=1)
    with pytest.raises(V03CostError, match="warm latency"):
        replace(_component("END_TO_END_WARM"), invocation_count=1)


def test_development_cost_ledger_may_be_partial_but_not_empty_or_formal_claiming() -> None:
    partial = V03CostLedger(
        run_id="development-cost-test",
        execution_scope="DEVELOPMENT",
        freeze_manifest_digest=None,
        cost_protocol_digest=frozen_cost_protocol_digest(),
        prefix_cost_evidence_digest=_d("development-prefix"),
        components=(_component("PROBE_COLLECTION"),),
    )
    assert len(partial.components) == 1
    with pytest.raises(V03CostError, match="cannot claim a formal freeze"):
        replace(partial, freeze_manifest_digest=_d("fake-freeze"), ledger_digest=None)
    with pytest.raises(V03CostError, match="cannot be empty"):
        replace(partial, components=(), ledger_digest=None)
