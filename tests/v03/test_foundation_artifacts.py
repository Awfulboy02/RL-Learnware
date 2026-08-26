from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from policy_learnware_v0.io import ArtifactExistsError
from policy_learnware_v0.v03.artifacts import V03ArtifactError, V03ArtifactLayout


def test_development_capabilities_are_separate_and_immutable(tmp_path: Path) -> None:
    layout = V03ArtifactLayout.development(tmp_path, "dev-r0")
    writer = layout.writer("encoder_checkpoints")
    checkpoint = layout.encoder_checkpoint_artifact(
        "fold-a", "fake-e0", "checkpoint.bin"
    )
    digest = writer.publish_bytes(checkpoint, b"frozen-checkpoint")
    assert digest == writer.publish_bytes(
        checkpoint, b"frozen-checkpoint", resume=True
    )
    with pytest.raises(V03ArtifactError, match="resume content mismatch"):
        writer.publish_bytes(checkpoint, b"changed", resume=True)
    with pytest.raises(ArtifactExistsError):
        writer.publish_bytes(checkpoint, b"frozen-checkpoint")

    with pytest.raises(V03ArtifactError):
        writer.publish_json(
            layout.artifact("development_oracle_private", "oracle.json"), {}
        )
    with pytest.raises(V03ArtifactError):
        layout.artifact("encoder_checkpoints", "..", "escape.bin")
    with pytest.raises(V03ArtifactError, match="unknown development artifact"):
        layout.writer("bakeoff_tables")
    with pytest.raises(V03ArtifactError, match="unknown development artifact"):
        layout.writer("optional_extensions")


def test_joint_v03_has_no_oracle_writer_and_completion_is_single_path(
    tmp_path: Path,
) -> None:
    layout = V03ArtifactLayout.joint(tmp_path, "joint-r0")
    with pytest.raises(V03ArtifactError, match="no confirmatory-oracle"):
        layout.writer("confirmatory_oracle_private")
    completion = layout.writer("joint_completion")
    completion.publish_json(layout.completion_manifest, {"status": "BLOCKED"})
    with pytest.raises(V03ArtifactError):
        completion.publish_json(layout.run_root / "other.json", {})


def test_symlinks_and_digest_or_noncanonical_loads_fail_closed(tmp_path: Path) -> None:
    layout = V03ArtifactLayout.development(tmp_path, "dev-r0")
    destination = layout.artifact("attribution", "report.json")
    digest = layout.writer("attribution").publish_json(destination, {"value": 1})
    assert layout.reader("attribution").load_json(
        destination, expected_sha256=digest
    ) == {"value": 1}
    with pytest.raises(V03ArtifactError, match="digest mismatch"):
        layout.reader("attribution").load_json(
            destination, expected_sha256="0" * 64
        )

    arrays_path = layout.artifact("attribution", "rows.npz")
    arrays_digest = layout.writer("attribution").publish_npz(
        arrays_path, {"rows": np.asarray([[1.0, 2.0]])}
    )
    loaded = layout.reader("attribution").load_npz(
        arrays_path, expected_sha256=arrays_digest
    )
    assert not loaded["rows"].flags.writeable

    outside = tmp_path / "outside"
    outside.mkdir()
    layout.domain_dir("probe_discovery").mkdir(parents=True)
    escape = layout.domain_dir("probe_discovery") / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    with pytest.raises(V03ArtifactError, match="symlink"):
        layout.writer("probe_discovery").publish_json(escape / "bad.json", {})


def test_reader_rejects_valid_digest_for_noncanonical_json(tmp_path: Path) -> None:
    from policy_learnware_v0.hashing import sha256_file

    layout = V03ArtifactLayout.development(tmp_path, "dev-r0")
    path = layout.artifact("attribution", "manual.json")
    path.parent.mkdir(parents=True)
    path.write_text('{ "value": 1 }\n', encoding="utf-8")
    with pytest.raises(V03ArtifactError, match="not canonical"):
        layout.reader("attribution").load_json(
            path, expected_sha256=sha256_file(path)
        )
