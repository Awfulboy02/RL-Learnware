from __future__ import annotations

from pathlib import Path

import pytest

from repro_fpo_ppo_v02.formal_plan import validate_formal_training_projection
from repro_fpo_ppo_v02.provenance import (
    FORMAL_EXECUTION_PURPOSE,
    TRAINING_JOB_SCHEMA,
    ContractError,
    finalize_training_job,
    finalize_training_plan,
)
from repro_fpo_ppo_v02.tests.helpers import (
    make_formal_freeze_binding,
    make_protocol,
    make_shifted_anchor,
)


def _formal_case(tmp_path: Path) -> tuple[dict, dict]:
    anchors: list[tuple[Path, dict]] = []
    semantics: list[dict] = []
    for index in range(30):
        path = tmp_path / f"anchor-{index:02d}.json"
        anchor = make_shifted_anchor(path, marker=f"formal-{index:02d}")
        anchors.append((path, anchor))
        semantics.append(
            {
                "source_anchor_id": anchor["anchor_id"],
                "task": "SyntheticTask",
                "nominal": False,
                "factor": 2.0,
                "factor_id": f"factor-{index:02d}",
                "axis_id": "synthetic-damping",
                "operator_id": "synthetic-damping-x2",
                "axis_binding_digest": "c" * 64,
                "leaf_allowlist": ["_mjx_model.dof_damping"],
            }
        )
    protocol = make_protocol()
    binding = make_formal_freeze_binding(
        tmp_path,
        anchor_ids=[anchor["anchor_id"] for _path, anchor in anchors],
        anchor_semantics=semantics,
        seeds=[7, 8, 9],
        config_digest="0" * 64,
        algorithm="ppo",
        training_steps=128,
        checkpoint_rule="fixed_final",
    )
    jobs = []
    for path, anchor in sorted(anchors, key=lambda item: item[1]["anchor_id"]):
        for seed in (7, 8, 9):
            jobs.append(
                finalize_training_job(
                    {
                        "schema": TRAINING_JOB_SCHEMA,
                        "job_id": f"v02-formal-{anchor['anchor_id'][:16]}-{seed}",
                        "config_digest": "0" * 64,
                        "execution_purpose": FORMAL_EXECUTION_PURPOSE,
                        "formal_protocol_freeze_digest": binding["binding_digest"],
                        "anchor_manifest_path": str(path.resolve()),
                        "anchor_manifest_digest": anchor["manifest_digest"],
                        "training_protocol": protocol,
                        "training_protocol_digest": protocol["protocol_digest"],
                        "seed": seed,
                    }
                )
            )
    plan = finalize_training_plan(jobs, formal_protocol_freeze=binding)
    return plan, binding


def test_formal_projection_accepts_only_exact_30_by_3_semantic_grid(
    tmp_path: Path,
) -> None:
    plan, binding = _formal_case(tmp_path)
    validated = validate_formal_training_projection(plan, binding)
    assert validated["expected_job_count"] == 90


def test_formal_projection_rejects_missing_seed_unit(tmp_path: Path) -> None:
    plan, binding = _formal_case(tmp_path)
    shortened = finalize_training_plan(
        plan["jobs"][:-1], formal_protocol_freeze=binding
    )
    with pytest.raises(ContractError, match="grid differs"):
        validate_formal_training_projection(shortened, binding)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task", "WrongTask", "task differs"),
        ("factor", 1.5, "factor differs"),
        ("axis_id", "wrong-axis", "axis differs"),
        ("operator_id", "wrong-operator", "operator differs"),
        ("axis_binding_digest", "d" * 64, "axis binding differs"),
        ("leaf_allowlist", ["_mjx_model.body_mass"], "mutation leaves differ"),
    ],
)
def test_formal_projection_rejects_anchor_semantic_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    plan, binding = _formal_case(tmp_path)
    semantics = [dict(row) for row in binding["training_contract"]["source_anchors"]]
    semantics[0][field] = value
    drifted = make_formal_freeze_binding(
        tmp_path / "drifted",
        anchor_ids=list(binding["training_contract"]["source_anchor_ids"]),
        anchor_semantics=semantics,
        seeds=[7, 8, 9],
        config_digest="0" * 64,
        algorithm="ppo",
        training_steps=128,
        checkpoint_rule="fixed_final",
    )
    drifted_plan = finalize_training_plan(
        [
            finalize_training_job(
                {
                    **{key: value for key, value in job.items() if key != "job_digest"},
                    "formal_protocol_freeze_digest": drifted["binding_digest"],
                }
            )
            for job in plan["jobs"]
        ],
        formal_protocol_freeze=drifted,
    )
    with pytest.raises(ContractError, match=message):
        validate_formal_training_projection(drifted_plan, drifted)
