from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from policy_learnware_v0.io import ArtifactExistsError
from policy_learnware_v0.v02.artifacts import V02ArtifactLayout, V02ArtifactLayoutError


def test_capability_domains_are_physically_separate_and_path_safe(tmp_path: Path) -> None:
    layout = V02ArtifactLayout(tmp_path, "v02-smoke-r0")
    with pytest.raises(V02ArtifactLayoutError):
        V02ArtifactLayout(tmp_path, "../escape")
    with pytest.raises(V02ArtifactLayoutError):
        layout.target_dataset_artifact("../private", 0, "dataset.npz")
    assert layout.gate_artifact("v02_gate_state.json") == (
        layout.analysis_dir / "gates" / "v02_gate_state.json"
    )
    assert layout.recompute_audit == layout.analysis_dir / "recompute_audit.json"

    measurement = layout.writer("measurement")
    with pytest.raises(V02ArtifactLayoutError):
        measurement.publish_json(
            layout.variant_artifact("opaque-env", "contexts.json"), {"factor": 2.0}
        )
    with pytest.raises(V02ArtifactLayoutError, match="unknown artifact capability"):
        layout.writer("confirmatory_oracle_private")  # type: ignore[arg-type]
    assert not hasattr(layout, "oracle_shard")
    assert not hasattr(layout, "confirmatory_oracle_private_dir")


def test_symlink_escape_and_inward_symlink_are_rejected(tmp_path: Path) -> None:
    layout = V02ArtifactLayout(tmp_path, "v02-smoke-r0")
    layout.measurement_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = layout.measurement_dir / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    with pytest.raises(V02ArtifactLayoutError, match="symlink"):
        layout.writer("measurement").publish_json(escape / "bad.json", {})

    inward_target = layout.measurement_dir / "real"
    inward_target.mkdir()
    inward = layout.measurement_dir / "alias"
    inward.symlink_to(inward_target, target_is_directory=True)
    with pytest.raises(V02ArtifactLayoutError, match="symlink"):
        layout.writer("measurement").publish_json(inward / "bad.json", {})


def test_immutable_resume_is_byte_exact_for_json_npz_and_text(tmp_path: Path) -> None:
    layout = V02ArtifactLayout(tmp_path, "v02-smoke-r0")
    writer = layout.writer("measurement")
    json_path = layout.target_dataset_artifact("target-a", 0, "manifest.json")
    digest = writer.publish_json(json_path, {"opaque": "value"})
    assert digest == writer.publish_json(json_path, {"opaque": "value"}, resume=True)
    with pytest.raises(V02ArtifactLayoutError, match="resume content mismatch"):
        writer.publish_json(json_path, {"opaque": "changed"}, resume=True)
    with pytest.raises(ArtifactExistsError):
        writer.publish_json(json_path, {"opaque": "value"})

    npz_path = layout.target_dataset_artifact("target-a", 0, "dataset.npz")
    arrays = {"points": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    npz_digest = writer.publish_npz(npz_path, arrays)
    assert npz_digest == writer.publish_npz(npz_path, arrays, resume=True)
    with pytest.raises(V02ArtifactLayoutError, match="resume content mismatch"):
        writer.publish_npz(npz_path, {"points": np.asarray([[9.0]])}, resume=True)

    text_path = layout.measurement_dir / "query_costs" / "note.txt"
    text_digest = writer.publish_text(text_path, "cold=1\n")
    assert text_digest == writer.publish_text(text_path, "cold=1\n", resume=True)


def test_completion_writer_can_only_publish_completion_manifests(tmp_path: Path) -> None:
    layout = V02ArtifactLayout(tmp_path, "v02-smoke-r0")
    completion = layout.writer("completion")
    completion.publish_json(layout.preflight_completion_manifest, {"status": "BLOCKED"})
    with pytest.raises(V02ArtifactLayoutError):
        completion.publish_json(layout.run_lock, {"bad": True})


def test_all_plan_domains_have_distinct_roots(tmp_path: Path) -> None:
    layout = V02ArtifactLayout(tmp_path, "v02-smoke-r0")
    domains = (
        "frozen",
        "benchmark_private",
        "training_private",
        "market_public",
        "representation_indices",
        "deployment_private",
        "measurement",
        "selector_outputs",
        "analysis",
    )
    roots = [layout.domain_dir(domain) for domain in domains]  # type: ignore[arg-type]
    assert len(roots) == len(set(roots))
