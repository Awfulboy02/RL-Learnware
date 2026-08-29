from argparse import Namespace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from policy_learnware_v0.v03.artifact_paths import (
    ARTIFACTS_ROOT_ENV,
    RELOCATION_MANIFEST_SCHEMA,
    ArtifactPathError,
    V03ArtifactLayout,
    resolve_artifacts_root,
    resolve_recorded_path,
)
from policy_learnware_v0.v03 import artifact_paths
from server.repro_fpo_ppo_v03 import development_baseline_runner
from server.repro_fpo_ppo_v03 import v031_raw_transition_runner


def _inventory(root: Path) -> tuple[str, int, int]:
    rows: list[tuple[bytes, Path]] = []
    if root.is_file():
        rows.append((root.name.encode(), root))
    elif not root.is_symlink():
        rows.extend(
            (path.relative_to(root).as_posix().encode(), path)
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    rows.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    total_bytes = 0
    for relative, path in rows:
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        digest.update(b"  ")
        digest.update(relative)
        digest.update(b"\n")
        total_bytes += path.stat().st_size
    return digest.hexdigest(), len(rows), total_bytes


def _mapping(
    source: Path | str,
    target_path: Path,
    *,
    artifacts_root: Path,
    **updates: object,
) -> dict[str, object]:
    inventory = _inventory(target_path)
    value: dict[str, object] = {
        "kind": "prefix",
        "source": str(source),
        "target": target_path.relative_to(artifacts_root).as_posix(),
        "content_manifest_sha256": inventory[0],
        "file_count": inventory[1],
        "total_bytes": inventory[2],
        "role": "policy_pool",
        "access_class": "restricted",
        "status": "verified",
    }
    value.update(updates)
    return value


def _manifest(root: Path, rows: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "relocation_manifest.json"
    path.write_text(
        json.dumps({"schema": RELOCATION_MANIFEST_SCHEMA, "mappings": rows}),
        encoding="utf-8",
    )
    return path


def test_artifact_root_precedence(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    from_env = tmp_path / "environment"
    repository = tmp_path / "repo"

    assert resolve_artifacts_root(
        explicit,
        environ={ARTIFACTS_ROOT_ENV: str(from_env)},
        repository_root=repository,
    ) == explicit.resolve()
    assert resolve_artifacts_root(
        environ={ARTIFACTS_ROOT_ENV: str(from_env)}, repository_root=repository
    ) == from_env.resolve()
    assert resolve_artifacts_root(environ={}, repository_root=repository) == (
        tmp_path / "artifacts"
    ).resolve()


def test_v03_layout_matches_frozen_run_names(tmp_path: Path) -> None:
    layout = V03ArtifactLayout(tmp_path)

    assert layout.relocation_manifest == tmp_path / "relocation_manifest.json"
    assert layout.context_index == (
        tmp_path
        / "v03/runs/v03-signal-ranking-20260827-r1/probes/context_index.json"
    )
    assert layout.public_policy_market == (
        tmp_path
        / "v03/runs/v03-main-20260827-r0/source-market/public_policy_market.json"
    )
    assert layout.development_oracle == (
        tmp_path / "v03/runs/v03-signal-ranking-20260827-r1/baseline/oracle"
    )


def test_sha256sum_relative_v1_fixture(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "nested").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"alpha\n")
    (root / "nested/b.bin").write_bytes(b"\x00\x01")

    assert artifact_paths._target_inventory(str(root)) == (
        "c62be7b78f68dceef6e40162e5ae4dac100b901fca0722df293ecbd0fac5b7bd",
        2,
        8,
    )


def test_verified_relocation_preserves_immutable_recorded_path(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    relocated = root / "v02/exact90/policies/policy-7"
    relocated.mkdir(parents=True)
    manifest = _manifest(
        root,
        [
            _mapping(
                "/share/songyf/RL_Learnware/v02_formal_artifacts",
                root / "v02/exact90",
                artifacts_root=root,
                kind="directory",
            )
        ],
    )

    resolved = resolve_recorded_path(
        "/share/songyf/RL_Learnware/v02_formal_artifacts/policies/policy-7",
        relocation_manifest=manifest,
        artifacts_root=root,
    )
    assert resolved == relocated.resolve()


def test_manifest_mapping_wins_even_when_recorded_path_still_exists(tmp_path: Path) -> None:
    original_root = tmp_path / "legacy"
    original = original_root / "policy"
    original.mkdir(parents=True)
    root = tmp_path / "artifacts"
    relocated = root / "verified/policy"
    relocated.mkdir(parents=True)
    manifest = _manifest(
        root, [_mapping(original_root, root / "verified", artifacts_root=root)]
    )

    assert resolve_recorded_path(
        original, relocation_manifest=manifest, artifacts_root=root
    ) == relocated.resolve()


def test_one_manifest_is_portable_across_two_artifacts_roots(tmp_path: Path) -> None:
    source = Path("/legacy/v03-run")
    root_a = tmp_path / "mirror-a/artifacts"
    root_b = tmp_path / "mirror-b/artifacts"
    target_a = root_a / "v03/runs/run-r0"
    (target_a / "nested").mkdir(parents=True)
    (target_a / "nested/record.json").write_text("{}", encoding="utf-8")
    target_b = root_b / "v03/runs/run-r0"
    shutil.copytree(target_a, target_b)
    rows = [_mapping(source, target_a, artifacts_root=root_a)]
    manifest_a = _manifest(root_a, rows)
    manifest_b = _manifest(root_b, rows)

    assert manifest_a.read_bytes() == manifest_b.read_bytes()
    assert resolve_recorded_path(
        source / "nested/record.json",
        relocation_manifest=manifest_a,
        artifacts_root=root_a,
    ) == (target_a / "nested/record.json").resolve()
    assert resolve_recorded_path(
        source / "nested/record.json",
        relocation_manifest=manifest_b,
        artifacts_root=root_b,
    ) == (target_b / "nested/record.json").resolve()


def test_longest_verified_prefix_wins(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    root = tmp_path / "artifacts"
    broad = root / "broad/special/item"
    narrow = root / "narrow/item"
    broad.mkdir(parents=True)
    narrow.mkdir(parents=True)
    manifest = _manifest(
        root,
        [
            _mapping(source, root / "broad", artifacts_root=root),
            _mapping(
                source / "special",
                root / "narrow",
                artifacts_root=root,
                role="nested",
            ),
        ],
    )

    assert resolve_recorded_path(
        source / "special/item", relocation_manifest=manifest, artifacts_root=root
    ) == narrow.resolve()


def test_file_mapping_is_exact_not_a_prefix(tmp_path: Path) -> None:
    source = tmp_path / "legacy.json"
    root = tmp_path / "artifacts"
    target = root / "records/current.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    manifest = _manifest(
        root, [_mapping(source, target, artifacts_root=root, kind="file")]
    )

    assert resolve_recorded_path(
        source, relocation_manifest=manifest, artifacts_root=root
    ) == target.resolve()
    with pytest.raises(ArtifactPathError, match="no verified relocation"):
        resolve_recorded_path(
            Path(f"{source}/child"),
            relocation_manifest=manifest,
            artifacts_root=root,
        )


def test_existing_original_fallback_is_only_allowed_without_manifest(tmp_path: Path) -> None:
    original = tmp_path / "legacy-policy"
    original.mkdir()

    assert resolve_recorded_path(
        original, relocation_manifest=None, artifacts_root=tmp_path / "unused"
    ) == original.resolve()


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "pending"},
        {"target": "../escape"},
        {"target": "/absolute/escape"},
        {"target": "v03\\escape"},
        {"target": "v03//escape"},
        {"target": "v03/escape\n"},
        {"target": "."},
        {"content_manifest_sha256": "not-a-digest"},
        {"kind": "directory_prefix"},
        {"unexpected": "field"},
    ],
)
def test_relocation_rejects_unverified_or_invalid_rows(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    source = tmp_path / "legacy"
    (source / "item").mkdir(parents=True)
    root = tmp_path / "artifacts"
    (root / "target/item").mkdir(parents=True)
    manifest = _manifest(
        root,
        [_mapping(source, root / "target", artifacts_root=root, **updates)],
    )

    with pytest.raises(ArtifactPathError):
        resolve_recorded_path(
            source / "item", relocation_manifest=manifest, artifacts_root=root
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"content_manifest_sha256": "f" * 64},
        {"file_count": 999},
        {"total_bytes": 999},
    ],
)
def test_relocation_rejects_inventory_mismatch(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    source = tmp_path / "legacy"
    root = tmp_path / "artifacts"
    (root / "target/item").mkdir(parents=True)
    manifest = _manifest(
        root,
        [_mapping(source, root / "target", artifacts_root=root, **updates)],
    )

    with pytest.raises(ArtifactPathError, match="inventory differs"):
        resolve_recorded_path(
            source / "item", relocation_manifest=manifest, artifacts_root=root
        )


def test_relocation_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    root = tmp_path / "artifacts"
    root.mkdir()
    manifest = root / "relocation_manifest.json"
    manifest.write_text(
        '{"schema":"rl-learnware-relocation/v1",'
        '"schema":"rl-learnware-relocation/v1","mappings":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ArtifactPathError, match="duplicate JSON key"):
        resolve_recorded_path(
            source / "item", relocation_manifest=manifest, artifacts_root=root
        )


def test_relocation_rejects_duplicate_prefixes(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    root = tmp_path / "artifacts"
    (root / "one/item").mkdir(parents=True)
    (root / "two/item").mkdir(parents=True)
    manifest = _manifest(
        root,
        [
            _mapping(source, root / "one", artifacts_root=root),
            _mapping(
                source,
                root / "two",
                artifacts_root=root,
                role="duplicate",
            ),
        ],
    )

    with pytest.raises(ArtifactPathError, match="duplicate relocation source"):
        resolve_recorded_path(
            source / "item", relocation_manifest=manifest, artifacts_root=root
        )


def test_relocation_rejects_symlink_escape(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    (outside / "item").mkdir()
    manifest = _manifest(
        root, [_mapping(source, root / "linked", artifacts_root=root)]
    )

    with pytest.raises(ArtifactPathError, match="symlink"):
        resolve_recorded_path(
            source / "item", relocation_manifest=manifest, artifacts_root=root
        )


def test_relocation_rejects_non_authoritative_manifest(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    root = tmp_path / "artifacts"
    (root / "target/item").mkdir(parents=True)
    _manifest(root, [_mapping(source, root / "target", artifacts_root=root)])
    sidecar = tmp_path / "owner-sidecar.json"
    sidecar.write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactPathError, match="authoritative root manifest"):
        resolve_recorded_path(
            source / "item", relocation_manifest=sidecar, artifacts_root=root
        )


def test_relocation_rejects_missing_target_and_non_normal_source(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    root = tmp_path / "artifacts"
    missing = root / "missing"
    manifest = _manifest(
        root,
        [
            _mapping(
                f"{source}/../legacy",
                missing,
                artifacts_root=root,
                content_manifest_sha256="0" * 64,
            )
        ],
    )
    with pytest.raises(ArtifactPathError, match="source must be normalized"):
        resolve_recorded_path(
            source / "item", relocation_manifest=manifest, artifacts_root=root
        )

    manifest = _manifest(
        root,
        [
            _mapping(
                source,
                missing,
                artifacts_root=root,
                content_manifest_sha256="0" * 64,
            )
        ],
    )
    with pytest.raises(ArtifactPathError, match="not a directory"):
        resolve_recorded_path(
            source / "item", relocation_manifest=manifest, artifacts_root=root
        )


def test_runners_derive_inputs_from_common_root(tmp_path: Path) -> None:
    raw_args = Namespace(
        artifacts_root=tmp_path,
        context_index=None,
        public_policy_market=None,
        deployment_private_registry=None,
        oracle_root=None,
    )
    v031_raw_transition_runner._resolve_artifact_inputs(raw_args)
    layout = V03ArtifactLayout(tmp_path.resolve())
    assert raw_args.context_index == layout.context_index
    assert raw_args.oracle_root == layout.development_oracle

    layout.root.mkdir(parents=True, exist_ok=True)
    layout.relocation_manifest.write_text("{}", encoding="utf-8")
    baseline_args = Namespace(
        artifacts_root=tmp_path,
        relocation_manifest=None,
        context_index=None,
        public_policy_market=None,
        deployment_private_registry=None,
        r5_checkpoint_root=None,
    )
    development_baseline_runner._resolve_artifact_inputs(baseline_args)
    assert baseline_args.public_policy_market == layout.public_policy_market
    assert baseline_args.r5_checkpoint_root == layout.signal_fit_root
    assert baseline_args.relocation_manifest == layout.relocation_manifest
