from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from policy_learnware_v0.io import ArtifactExistsError
from policy_learnware_v0.v01.artifacts import V01ArtifactLayout, V01ArtifactLayoutError


def test_layout_is_path_safe_and_domains_are_physically_separate(tmp_path: Path) -> None:
    layout = V01ArtifactLayout(tmp_path, "experiment-r0")
    with pytest.raises(V01ArtifactLayoutError):
        V01ArtifactLayout(tmp_path, "../escape")
    with pytest.raises(V01ArtifactLayoutError):
        layout.dataset_npz("../private", 0)
    with pytest.raises(V01ArtifactLayoutError):
        layout.assert_managed(tmp_path / "outside.json")

    measurement = layout.writer("measurement")
    private_path = layout.contexts
    with pytest.raises(V01ArtifactLayoutError):
        measurement.publish_json(private_path, {"factor": 2.0})
    oracle = layout.writer("oracle_private")
    with pytest.raises(V01ArtifactLayoutError):
        oracle.publish_json(layout.measurement_run_ref, {"leak": True})


def test_immutable_publish_resume_and_tamper_detection(tmp_path: Path) -> None:
    layout = V01ArtifactLayout(tmp_path, "experiment-r0")
    writer = layout.writer("measurement")
    digest = writer.publish_json(layout.measurement_run_ref, {"run": "opaque"})
    assert digest == writer.publish_json(layout.measurement_run_ref, {"run": "opaque"}, resume=True)
    with pytest.raises(V01ArtifactLayoutError):
        writer.publish_json(layout.measurement_run_ref, {"run": "changed"}, resume=True)
    with pytest.raises(ArtifactExistsError):
        writer.publish_json(layout.measurement_run_ref, {"run": "opaque"})

    arrays = {"points": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    npz = layout.semantic_cache("v01v-" + "a" * 20, 0)
    first = writer.publish_npz(npz, arrays)
    assert first == writer.publish_npz(npz, arrays, resume=True)
    with pytest.raises(V01ArtifactLayoutError):
        writer.publish_npz(npz, {"points": np.asarray([[2.0]])}, resume=True)


def test_completion_capability_can_only_publish_root_completion_manifests(tmp_path: Path) -> None:
    layout = V01ArtifactLayout(tmp_path, "experiment-r0")
    completion = layout.writer("completion")
    completion.publish_json(layout.preflight_completion_manifest, {"status": "NO_GO_COMPUTE"})
    with pytest.raises(V01ArtifactLayoutError):
        completion.publish_json(layout.run_lock, {"bad": True})
    assert layout.relative(layout.preflight_completion_manifest) == "preflight_completion_manifest.json"
