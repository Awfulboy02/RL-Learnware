from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

import policy_learnware_v0.v02.artifacts as artifacts_module
from policy_learnware_v0.v02.artifacts import (
    ASSET_EXPECTATIONS,
    EXPECTED_VENDOR_TREE_DIGEST,
    RELOCATION_SCHEMA,
    RelocationResolver,
    V02AssetError,
    V02AssetLayout,
    V02_RUN_ID,
    attest_directory,
    capability_status,
    resolve_artifacts_root,
    validate_relocation_manifest,
)


_FIXTURE_SOURCES = {
    "exact90": "/legacy/exact90",
    "formal_inputs": "/legacy/formal",
    "legacy_v02": "/legacy/repro",
    "runtime_state": "/legacy/exact90/training_private/coordination",
}


def _bind_fixture_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    source_hashes = dict(artifacts_module._V02_SOURCE_SHA256)
    source_hashes.update(
        {
            asset_id: hashlib.sha256(source.encode("utf-8")).hexdigest()
            for asset_id, source in _FIXTURE_SOURCES.items()
        }
    )
    monkeypatch.setattr(artifacts_module, "_V02_SOURCE_SHA256", source_hashes)


def _bind_fixture_inventory(
    monkeypatch: pytest.MonkeyPatch,
    layout: V02AssetLayout,
    asset_id: str,
) -> None:
    observed = attest_directory(layout.asset(asset_id))
    expected = ASSET_EXPECTATIONS[asset_id]
    monkeypatch.setitem(
        ASSET_EXPECTATIONS,
        asset_id,
        replace(
            expected,
            tree_digest=observed.tree_digest,
            file_count=observed.file_count,
            total_bytes=observed.total_bytes,
        ),
    )


def _row(
    source: str,
    target: str,
    *,
    kind: str = "directory",
    status: str = "verified",
    role: str = "fixture",
    access_class: str = "immutable_read_only",
    digest: str = "0" * 64,
    file_count: int = 0,
    total_bytes: int = 0,
    completeness: str | None = None,
    known_missing: list[str] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "kind": kind,
        "source": source,
        "target": target,
        "content_manifest_sha256": digest,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "role": role,
        "access_class": access_class,
        "status": status,
    }
    if completeness is not None:
        row["completeness"] = completeness
    if known_missing is not None:
        row["known_missing"] = known_missing
    return row


def _asset_row(
    asset_id: str,
    source: str,
    *,
    status: str = "verified",
) -> dict[str, object]:
    expected = ASSET_EXPECTATIONS[asset_id]
    assert expected.target_relpath is not None
    roles = {
        "exact90": "v02-exact90-handoff-and-training-evidence",
        "formal_inputs": "v02-formal-inputs",
        "legacy_v02": "legacy-v02-policy-training-backup",
        "runtime_state": "v02-operational-runtime-state",
    }
    return _row(
        source,
        expected.target_relpath,
        kind="prefix",
        status=status,
        role=roles[asset_id],
        access_class="restricted",
        digest=expected.tree_digest or hashlib.sha256(asset_id.encode()).hexdigest(),
        file_count=expected.file_count if expected.file_count is not None else 3,
        total_bytes=expected.total_bytes if expected.total_bytes is not None else 17,
        completeness="incomplete" if asset_id == "legacy_v02" else "complete",
        known_missing=["_vendor"] if asset_id == "legacy_v02" else None,
    )


def _manifest(*, exact_status: str = "verified") -> dict[str, object]:
    return {
        "schema": RELOCATION_SCHEMA,
        "mappings": [
            _asset_row("exact90", "/legacy/exact90", status=exact_status),
            _asset_row("formal_inputs", "/legacy/formal"),
            _asset_row("legacy_v02", "/legacy/repro"),
            _asset_row(
                "runtime_state",
                "/legacy/exact90/training_private/coordination",
            ),
            _row(
                "/legacy/v03/query",
                "v03/query",
                kind="prefix",
                role="v03-query",
            ),
        ],
    }


def _git_checkout_with_sibling_manifest(
    parent: Path,
    *,
    manifest: object | None = None,
) -> Path:
    repository = parent / "policy_learnware_v0"
    repository.mkdir(parents=True)
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'policy-learnware-fixture'\nversion = '0'\n",
        encoding="utf-8",
    )
    subprocess.run(
        ("git", "init", "--quiet", str(repository)),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        ("git", "-C", str(repository), "add", "--", "pyproject.toml"),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "-m",
            "fixture",
        ),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if manifest is not None:
        artifacts = parent / "artifacts"
        artifacts.mkdir()
        (artifacts / "relocation_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return repository


def test_root_resolution_precedence_and_single_manifest_location(tmp_path: Path) -> None:
    fake_repo = tmp_path / "fake" / "policy_learnware_v0"
    fake_repo.mkdir(parents=True)
    (fake_repo / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    environment = {"RL_LEARNWARE_ARTIFACTS_ROOT": str(tmp_path / "from-env")}
    assert resolve_artifacts_root(
        tmp_path / "explicit", repository_root=fake_repo, environ=environment
    ) == (tmp_path / "explicit").resolve()
    assert resolve_artifacts_root(repository_root=fake_repo, environ=environment) == (
        tmp_path / "from-env"
    ).resolve()
    with pytest.raises(V02AssetError, match="Git checkout proof"):
        resolve_artifacts_root(repository_root=fake_repo, environ={})

    real_parent = tmp_path / "real-checkout"
    repository = _git_checkout_with_sibling_manifest(real_parent, manifest=_manifest())
    assert resolve_artifacts_root(repository_root=repository, environ={}) == (
        real_parent / "artifacts"
    ).resolve()

    layout = V02AssetLayout.resolve(tmp_path / "artifacts", environ={})
    assert layout.relocation_manifest == (
        tmp_path / "artifacts" / "relocation_manifest.json"
    ).resolve()
    assert layout.exact90 == (
        tmp_path / "artifacts" / "v02" / "exact90" / V02_RUN_ID
    ).resolve()
    assert layout.fpo == (
        tmp_path / "artifacts" / "shared" / "runtime" / "fpo-418c2554"
    ).resolve()

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(V02AssetError, match="contains a symlink"):
        resolve_artifacts_root(linked_parent / "artifacts", environ={})
    with pytest.raises(V02AssetError, match="contains a symlink"):
        resolve_artifacts_root(
            environ={"RL_LEARNWARE_ARTIFACTS_ROOT": str(linked_parent / "artifacts")}
        )


@pytest.mark.parametrize("value", ["", " ", "\t\n"])
def test_explicit_and_environment_empty_roots_fail_closed(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(V02AssetError, match="explicit artifacts root cannot be empty"):
        resolve_artifacts_root(value, repository_root=tmp_path, environ={})
    with pytest.raises(V02AssetError, match="cannot be empty or whitespace"):
        resolve_artifacts_root(
            repository_root=tmp_path,
            environ={"RL_LEARNWARE_ARTIFACTS_ROOT": value},
        )


def test_repository_fallback_requires_checkout_top_and_strict_sibling_manifest(
    tmp_path: Path,
) -> None:
    no_manifest = _git_checkout_with_sibling_manifest(tmp_path / "no-manifest")
    with pytest.raises(V02AssetError, match="strict root relocation manifest"):
        resolve_artifacts_root(repository_root=no_manifest, environ={})

    invalid = _git_checkout_with_sibling_manifest(
        tmp_path / "invalid-manifest", manifest={"schema": RELOCATION_SCHEMA}
    )
    with pytest.raises(V02AssetError, match="strict root relocation manifest"):
        resolve_artifacts_root(repository_root=invalid, environ={})

    valid = _git_checkout_with_sibling_manifest(
        tmp_path / "valid-manifest", manifest=_manifest()
    )
    (valid / "nested").mkdir()
    with pytest.raises(V02AssetError, match="checkout top-level"):
        resolve_artifacts_root(repository_root=valid / "nested", environ={})


def test_repository_fallback_scrubs_git_environment_for_every_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = _git_checkout_with_sibling_manifest(
        tmp_path / "real", manifest=_manifest()
    )
    fake = tmp_path / "fake" / "policy_learnware_v0"
    fake.mkdir(parents=True)
    (fake / "pyproject.toml").write_text(
        (real / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )

    hostile = {
        "GIT_DIR": str(real / ".git"),
        "GIT_WORK_TREE": str(fake),
        "GIT_COMMON_DIR": str(real / ".git"),
        "GIT_INDEX_FILE": str(real / ".git" / "index"),
        "GIT_OBJECT_DIRECTORY": str(real / ".git" / "objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(real / ".git" / "objects"),
        "GIT_CEILING_DIRECTORIES": str(tmp_path),
        "GIT_DISCOVERY_ACROSS_FILESYSTEM": "1",
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))

    original_run = subprocess.run
    observed: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def audited_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = tuple(str(item) for item in args[0])  # type: ignore[index]
        environment = dict(kwargs["env"])  # type: ignore[arg-type]
        observed.append((command, environment))
        return original_run(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(artifacts_module.subprocess, "run", audited_run)
    with pytest.raises(V02AssetError, match="Git checkout proof"):
        resolve_artifacts_root(repository_root=fake, environ={})
    assert resolve_artifacts_root(repository_root=real, environ={}) == (
        tmp_path / "real" / "artifacts"
    ).resolve()

    expected_queries = [
        ("rev-parse", "--show-toplevel"),
        ("rev-parse", "--verify", "HEAD^{commit}"),
        ("ls-files", "--error-unmatch", "--", "pyproject.toml"),
    ]
    queries = []
    for command, _ in observed[-3:]:
        assert command[0] == "/usr/bin/git"
        root_index = command.index("-C")
        queries.append(command[root_index + 2 :])
    assert queries == expected_queries
    assert len(observed) == 4
    for _, environment in observed:
        assert not set(hostile).intersection(environment)
        assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
        assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
        assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
        assert environment["GIT_OPTIONAL_LOCKS"] == "0"
        assert environment["LC_ALL"] == "C"


def test_default_loader_reads_only_root_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bind_fixture_sources(monkeypatch)
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "relocation_manifest.json").write_text(
        json.dumps(_manifest()), encoding="utf-8"
    )
    resolver = RelocationResolver.load(artifacts_root=root, environ={})
    assert resolver.layout.relocation_manifest == root / "relocation_manifest.json"

    (root / "v02").mkdir()
    (root / "v02" / "relocation_manifest.json").write_text(
        "not authoritative", encoding="utf-8"
    )
    assert RelocationResolver.load(artifacts_root=root, environ={}).manifest == (
        resolver.manifest
    )


def test_manifest_path_must_be_absolute_regular_and_non_symlink(tmp_path: Path) -> None:
    real = tmp_path / "relocation_manifest.json"
    real.write_text(json.dumps(_manifest()), encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(real)
    with pytest.raises(V02AssetError, match="symlink"):
        validate_relocation_manifest(alias)
    with pytest.raises(V02AssetError, match="must be absolute"):
        validate_relocation_manifest(Path("relocation_manifest.json"))


def test_directory_attestation_exact_sha256sum_relative_v1(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "nested").mkdir(parents=True)
    payloads = {
        "a.txt": b"a\n",
        "nested/z.txt": b"z\n",
        "nested/é.txt": b"e\n",
    }
    for relative, payload in payloads.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    lines = [
        (
            relative,
            f"{hashlib.sha256(payload).hexdigest()}  {relative}\n".encode("utf-8"),
        )
        for relative, payload in payloads.items()
    ]
    expected = hashlib.sha256(
        b"".join(line for _, line in sorted(lines, key=lambda item: item[0].encode("utf-8")))
    ).hexdigest()
    observed = attest_directory(root)
    assert observed.file_count == 3
    assert observed.total_bytes == 6
    assert observed.tree_digest == expected

    linked_parent = tmp_path / "linked-tree-parent"
    linked_parent.symlink_to(root.parent, target_is_directory=True)
    with pytest.raises(V02AssetError, match="contains a symlink"):
        attest_directory(linked_parent / root.name)

    (root / "alias").symlink_to(root / "a.txt")
    with pytest.raises(V02AssetError, match="symlink"):
        attest_directory(root)
    (root / "alias").unlink()

    os.mkfifo(root / "named-pipe")
    with pytest.raises(V02AssetError, match="special file"):
        attest_directory(root)
    (root / "named-pipe").unlink()


def test_manifest_has_exact_top_level_and_row_inventory() -> None:
    manifest = _manifest()
    assert set(validate_relocation_manifest(manifest)) == {"schema", "mappings"}

    extra_top = dict(manifest)
    extra_top["manifest_digest"] = "0" * 64
    with pytest.raises(V02AssetError, match="keys differ"):
        validate_relocation_manifest(extra_top)

    extra_row = _manifest()
    extra_row["mappings"][0]["resolver_policy"] = "legacy"  # type: ignore[index]
    with pytest.raises(V02AssetError, match="unknown=.*resolver_policy"):
        validate_relocation_manifest(extra_row)

    missing_row_key = _manifest()
    del missing_row_key["mappings"][0]["content_manifest_sha256"]  # type: ignore[index]
    with pytest.raises(V02AssetError, match="missing=.*content_manifest_sha256"):
        validate_relocation_manifest(missing_row_key)


@pytest.mark.parametrize(
    "target",
    ["", "/absolute", "./dot", "parent/../escape", "double//slash", "back\\slash", "nul\0x", "line\nx"],
)
def test_manifest_rejects_unsafe_nonportable_targets(target: str) -> None:
    manifest = _manifest()
    manifest["mappings"][0]["target"] = target  # type: ignore[index]
    with pytest.raises(V02AssetError, match="target"):
        validate_relocation_manifest(manifest)


@pytest.mark.parametrize(
    "source",
    ["relative", "/", "/dot/./x", "/parent/../x", "/double//x", "/back\\slash", "/nul\0x", "/line\nx"],
)
def test_manifest_rejects_unsafe_historical_sources(source: str) -> None:
    manifest = _manifest()
    manifest["mappings"][0]["source"] = source  # type: ignore[index]
    with pytest.raises(V02AssetError, match="source"):
        validate_relocation_manifest(manifest)


def test_manifest_kind_and_actual_inventory_fields_are_strict() -> None:
    bad_kind = _manifest()
    bad_kind["mappings"][0]["kind"] = "asset"  # type: ignore[index]
    with pytest.raises(V02AssetError, match="prefix, directory, or file"):
        validate_relocation_manifest(bad_kind)

    bad_digest = _manifest()
    bad_digest["mappings"][0]["content_manifest_sha256"] = "A" * 64  # type: ignore[index]
    with pytest.raises(V02AssetError, match="lowercase SHA-256"):
        validate_relocation_manifest(bad_digest)

    bad_count = _manifest()
    bad_count["mappings"][0]["file_count"] = None  # type: ignore[index]
    with pytest.raises(V02AssetError, match="actual file_count"):
        validate_relocation_manifest(bad_count)

    bad_file = _manifest()
    bad_file["mappings"].append(  # type: ignore[union-attr]
        _row(
            "/legacy/one.json",
            "v02/one.json",
            kind="file",
            file_count=2,
        )
    )
    with pytest.raises(V02AssetError, match="file_count=1"):
        validate_relocation_manifest(bad_file)


def test_resolver_verified_longest_prefix_and_canonical_equivalence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = V02AssetLayout.resolve(tmp_path / "artifacts", environ={})
    exact_file = layout.exact90 / "receipt.json"
    exact_file.parent.mkdir(parents=True)
    exact_file.write_text("{}\n", encoding="utf-8")
    runtime_file = layout.asset("runtime_state") / "waiters" / "g0544.json"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text("{}\n", encoding="utf-8")
    _bind_fixture_sources(monkeypatch)
    _bind_fixture_inventory(monkeypatch, layout, "exact90")
    _bind_fixture_inventory(monkeypatch, layout, "runtime_state")
    resolver = RelocationResolver(layout, validate_relocation_manifest(_manifest()))

    assert resolver.resolve("/legacy/exact90/receipt.json") == exact_file.resolve()
    assert resolver.resolve(exact_file) == exact_file.resolve()
    assert resolver.resolve(
        "/legacy/exact90/training_private/coordination/waiters/g0544.json"
    ) == runtime_file.resolve()

    with pytest.raises(V02AssetError, match="no allowlisted"):
        resolver.resolve("/unknown/path")
    with pytest.raises(V02AssetError, match="normalized absolute"):
        resolver.resolve("/legacy/exact90/../escape")


def test_unknown_file_and_unverified_v02_rows_never_activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = V02AssetLayout.resolve(tmp_path / "artifacts", environ={})
    _bind_fixture_sources(monkeypatch)
    target = layout.root / "v02" / "single.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    manifest = _manifest(exact_status="pending")
    manifest["mappings"].append(  # type: ignore[union-attr]
        _row(
            "/legacy/single.json",
            "v02/single.json",
            kind="file",
            role="single-file",
            file_count=1,
            total_bytes=3,
        )
    )
    resolver = RelocationResolver(layout, validate_relocation_manifest(manifest))
    with pytest.raises(V02AssetError, match="no allowlisted"):
        resolver.resolve("/legacy/single.json")
    with pytest.raises(V02AssetError, match="no allowlisted"):
        resolver.resolve(target)
    with pytest.raises(V02AssetError, match="no allowlisted"):
        resolver.resolve("/legacy/exact90/receipt.json", must_exist=False)


def test_resolver_rejects_symlink_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = V02AssetLayout.resolve(tmp_path / "artifacts", environ={})
    _bind_fixture_sources(monkeypatch)
    layout.exact90.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (layout.exact90 / "escape").symlink_to(outside, target_is_directory=True)
    resolver = RelocationResolver(layout, validate_relocation_manifest(_manifest()))
    with pytest.raises(V02AssetError, match="symlink"):
        resolver.resolve("/legacy/exact90/escape/file.json", must_exist=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [("content_manifest_sha256", "f" * 64), ("file_count", 999), ("total_bytes", 999)],
)
def test_fixed_v02_identity_rejects_wrong_inventory_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    layout = V02AssetLayout.resolve(tmp_path / "artifacts", environ={})
    layout.exact90.mkdir(parents=True)
    (layout.exact90 / "receipt.json").write_text("{}\n", encoding="utf-8")
    _bind_fixture_sources(monkeypatch)
    _bind_fixture_inventory(monkeypatch, layout, "exact90")
    manifest = _manifest()
    manifest["mappings"][0][field] = value  # type: ignore[index]
    with pytest.raises(V02AssetError, match="fixed v0.2 identity"):
        RelocationResolver(layout, validate_relocation_manifest(manifest))


def test_verified_unknown_overlap_with_historical_or_current_path_is_no_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = V02AssetLayout.resolve(tmp_path / "artifacts", environ={})
    _bind_fixture_sources(monkeypatch)
    for source in (
        "/legacy/exact90/unknown",
        str(layout.exact90 / "unknown"),
    ):
        manifest = _manifest()
        manifest["mappings"].append(  # type: ignore[union-attr]
            _row(source, "v03/unknown", kind="prefix", role="unknown-overlap")
        )
        with pytest.raises(V02AssetError, match="overlaps"):
            RelocationResolver(layout, validate_relocation_manifest(manifest))


def test_same_manifest_is_portable_across_two_artifacts_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left_layout = V02AssetLayout.resolve(tmp_path / "left", environ={})
    right_layout = V02AssetLayout.resolve(tmp_path / "right", environ={})
    for layout in (left_layout, right_layout):
        layout.exact90.mkdir(parents=True)
        (layout.exact90 / "anchor.txt").write_text("same\n", encoding="utf-8")
    _bind_fixture_sources(monkeypatch)
    _bind_fixture_inventory(monkeypatch, left_layout, "exact90")
    manifest = validate_relocation_manifest(_manifest())
    left = RelocationResolver(left_layout, manifest)
    right = RelocationResolver(right_layout, manifest)
    recorded = "/legacy/exact90/receipt.json"
    assert left.resolve(recorded, must_exist=False) == (
        tmp_path / "left" / "v02" / "exact90" / V02_RUN_ID / "receipt.json"
    ).resolve()
    assert right.resolve(recorded, must_exist=False) == (
        tmp_path / "right" / "v02" / "exact90" / V02_RUN_ID / "receipt.json"
    ).resolve()
    assert left.resolve(recorded, must_exist=False) != right.resolve(
        recorded, must_exist=False
    )


def test_capabilities_keep_missing_original_vendor_out_of_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = V02AssetLayout.resolve(tmp_path / "artifacts", environ={})
    layout.exact90.mkdir(parents=True)
    layout.formal_inputs.mkdir(parents=True)
    _bind_fixture_sources(monkeypatch)
    _bind_fixture_inventory(monkeypatch, layout, "exact90")
    _bind_fixture_inventory(monkeypatch, layout, "formal_inputs")
    status = capability_status(layout, _manifest())
    assert status["schema"] == "policy-learnware.v02-capability-status.v1"
    assert status["readiness_scope"] == "asset_and_provenance_only"
    assert status["runtime_dependency_check"] == "not_performed"
    assert status["handoff_verification"] == {
        "available": True,
        "provenance_class": "ORIGINAL_IMMUTABLE_EVIDENCE",
    }
    assert status["policy_inference"] == {
        "asset_provenance_ready": False,
        "runtime_dependency_check": "not_performed",
        "runtime_dependency_ready": None,
        "provenance_class_if_runtime_ready": "UNAVAILABLE",
    }
    assert status["training_replay"] == {
        "asset_provenance_ready": False,
        "runtime_dependency_check": "not_performed",
        "runtime_dependency_ready": None,
        "provenance_class_if_runtime_ready": "UNAVAILABLE",
        "blocker": "MISSING_ORIGINAL_VENDOR_RUNTIME",
    }


def test_capability_asset_readiness_never_calls_runtime_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from policy_learnware_v0.v02 import runtime

    layout = V02AssetLayout.resolve(tmp_path / "artifacts", environ={})
    for asset_id in ("exact90", "formal_inputs", "legacy_v02", "fpo"):
        layout.asset(asset_id).mkdir(parents=True)
    policy_io = layout.legacy_v02 / "policy_io.py"
    policy_io.write_text("# recovered fixture\n", encoding="utf-8")

    _bind_fixture_sources(monkeypatch)
    for asset_id in ("exact90", "formal_inputs", "legacy_v02"):
        _bind_fixture_inventory(monkeypatch, layout, asset_id)
    monkeypatch.setattr(
        artifacts_module,
        "EXPECTED_POLICY_IO_SHA256",
        hashlib.sha256(policy_io.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(runtime, "verify_fpo_checkout", lambda _path: {"passed": True})

    def forbidden_loader(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("capability_status must not import runtime dependencies")

    monkeypatch.setattr(runtime, "load_verified_fpo_upstream", forbidden_loader)
    status = capability_status(layout, _manifest())
    assert status["policy_inference"] == {
        "asset_provenance_ready": True,
        "runtime_dependency_check": "not_performed",
        "runtime_dependency_ready": None,
        "provenance_class_if_runtime_ready": "RECONSTRUCTED_RUNTIME",
    }
    assert "available" not in status["policy_inference"]
    assert "provenance_class" not in status["policy_inference"]


def test_capability_rejects_legacy_symlink_before_hashing_policy_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = V02AssetLayout.resolve(tmp_path / "artifacts", environ={})
    layout.legacy_v02.mkdir(parents=True)
    outside = tmp_path / "outside-policy-io.py"
    outside.write_text("# must never be read\n", encoding="utf-8")
    (layout.legacy_v02 / "policy_io.py").symlink_to(outside)
    _bind_fixture_sources(monkeypatch)

    hashed: list[Path] = []

    def forbidden_hash(path: str | Path) -> str:
        hashed.append(Path(path))
        raise AssertionError("untrusted policy_io was hashed before attestation")

    monkeypatch.setattr(artifacts_module, "sha256_file", forbidden_hash)
    with pytest.raises(V02AssetError, match="symlink"):
        capability_status(layout, _manifest())
    assert hashed == []
    resolver = RelocationResolver(layout, validate_relocation_manifest(_manifest()))
    with pytest.raises(V02AssetError, match="MISSING_ORIGINAL"):
        resolver.ensure_verified_asset("vendor_original", must_exist=False)
    assert EXPECTED_VENDOR_TREE_DIGEST == ASSET_EXPECTATIONS["vendor_original"].tree_digest
