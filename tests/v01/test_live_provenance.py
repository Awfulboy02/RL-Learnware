from __future__ import annotations

from copy import deepcopy

import pytest

from policy_learnware_v0.v01 import cli


COMMIT_A = "a" * 40
COMMIT_B = "b" * 40


def _source_maps() -> dict[str, dict[str, str]]:
    return {
        "measurement": {"probe_source": "1" * 64, "cli_source": "2" * 64},
        "oracle": {"oracle_source": "3" * 64, "cli_source": "2" * 64},
        "analysis": {"gates_source": "4" * 64, "cli_source": "2" * 64},
    }


def _patch_live(monkeypatch: pytest.MonkeyPatch, *, commit: str = COMMIT_A, clean: bool = True) -> None:
    monkeypatch.setattr(
        cli,
        "_git_state",
        lambda _root: {
            "commit": commit,
            "clean": clean,
            "porcelain_sha256": ("e3b0c44298fc1c149afbf4c8996fb924"
                                  "27ae41e4649b934ca495991b7852b855")
            if clean
            else "9" * 64,
        },
    )
    monkeypatch.setattr(cli, "_runtime_versions", lambda: {"python": "3.12.13"})
    maps = _source_maps()
    monkeypatch.setattr(
        cli,
        "_live_source_digests_for_domains",
        lambda domains: {domain: maps[domain] for domain in domains},
    )


def _verify(monkeypatch: pytest.MonkeyPatch, *, formal: bool, frozen_clean: bool = True) -> dict:
    _patch_live(monkeypatch)
    maps = _source_maps()
    return cli._validate_live_provenance(
        formal=formal,
        frozen_git={
            "commit": COMMIT_A,
            "clean": frozen_clean,
            "porcelain_sha256": ("e3b0c44298fc1c149afbf4c8996fb924"
                                  "27ae41e4649b934ca495991b7852b855")
            if frozen_clean
            else "8" * 64,
        },
        frozen_runtime_versions={"python": "3.12.13"},
        frozen_source_digests=maps,
        domains=("measurement", "oracle", "analysis"),
        protocol_component_digests={
            "measurement": maps["measurement"],
            "oracle": maps["oracle"],
        },
    )


def test_formal_live_provenance_requires_exact_clean_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _verify(monkeypatch, formal=True)
    assert result["git_policy"] == "exact_commit_and_clean"

    _patch_live(monkeypatch, commit=COMMIT_B)
    with pytest.raises(cli.V01CommandFailure, match="Git commit differs"):
        cli._validate_live_provenance(
            formal=True,
            frozen_git={
                "commit": COMMIT_A,
                "clean": True,
                "porcelain_sha256": ("e3b0c44298fc1c149afbf4c8996fb924"
                                      "27ae41e4649b934ca495991b7852b855"),
            },
            frozen_runtime_versions={"python": "3.12.13"},
            frozen_source_digests=_source_maps(),
            domains=("measurement",),
        )

    _patch_live(monkeypatch, clean=False)
    with pytest.raises(cli.V01CommandFailure, match="requires a clean"):
        cli._validate_live_provenance(
            formal=True,
            frozen_git={
                "commit": COMMIT_A,
                "clean": True,
                "porcelain_sha256": ("e3b0c44298fc1c149afbf4c8996fb924"
                                      "27ae41e4649b934ca495991b7852b855"),
            },
            frozen_runtime_versions={"python": "3.12.13"},
            frozen_source_digests=_source_maps(),
            domains=("measurement",),
        )

    with pytest.raises(cli.V01CommandFailure, match="was not frozen from a clean"):
        _verify(monkeypatch, formal=True, frozen_clean=False)


def test_smoke_allows_git_drift_but_not_runtime_or_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_live(monkeypatch, commit=COMMIT_B, clean=False)
    maps = _source_maps()
    result = cli._validate_live_provenance(
        formal=False,
        frozen_git={
            "commit": COMMIT_A,
            "clean": False,
            "porcelain_sha256": "8" * 64,
        },
        frozen_runtime_versions={"python": "3.12.13"},
        frozen_source_digests=maps,
        domains=("measurement",),
    )
    assert result["git_policy"] == "smoke_source_scoped"

    with pytest.raises(cli.V01CommandFailure, match="runtime differs"):
        cli._validate_live_provenance(
            formal=False,
            frozen_git={
                "commit": COMMIT_A,
                "clean": False,
                "porcelain_sha256": "8" * 64,
            },
            frozen_runtime_versions={"python": "3.11.0"},
            frozen_source_digests=maps,
            domains=("measurement",),
        )

    changed = deepcopy(maps)
    changed["measurement"]["probe_source"] = "f" * 64
    with pytest.raises(cli.V01CommandFailure, match="source digest mismatch"):
        cli._validate_live_provenance(
            formal=False,
            frozen_git={
                "commit": COMMIT_A,
                "clean": False,
                "porcelain_sha256": "8" * 64,
            },
            frozen_runtime_versions={"python": "3.12.13"},
            frozen_source_digests=changed,
            domains=("measurement",),
        )


def test_live_provenance_requires_protocol_and_manifest_component_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_live(monkeypatch)
    maps = _source_maps()
    protocol = deepcopy(maps["measurement"])
    protocol["probe_source"] = "e" * 64
    with pytest.raises(cli.V01CommandFailure, match="protocol component digests"):
        cli._validate_live_provenance(
            formal=False,
            frozen_git={
                "commit": COMMIT_A,
                "clean": False,
                "porcelain_sha256": "8" * 64,
            },
            frozen_runtime_versions={"python": "3.12.13"},
            frozen_source_digests=maps,
            domains=("measurement",),
            protocol_component_digests={"measurement": protocol},
        )
