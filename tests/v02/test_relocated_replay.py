from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from server.repro_fpo_ppo_v02 import replay
from server.repro_fpo_ppo_v02.provenance import ContractError, with_self_digest


class _Resolver:
    def __init__(self, root: Path) -> None:
        exact90 = root / "v02" / "exact90" / "v02-reacher-formal-2r-20260825-r2"
        self.layout = SimpleNamespace(
            exact90=exact90,
            formal_inputs=(
                root
                / "v02"
                / "formal_inputs"
                / "v02-reacher-formal-2r-20260825-r2"
            ),
            frozen_acceptance=(
                exact90
                / "policy_pool_handoff_a7d10c0"
                / "policy_pool_acceptance.json"
            ),
            promotions=(
                exact90
                / "policy_pool_handoff_a7d10c0"
                / "compiled_parity_promotions.json"
            ),
            server_plan=(
                exact90 / "training_private" / "plans" / "server_training_plan.json"
            ),
            runs_root=exact90 / "training_private" / "server_runs",
        )
        self._source = Path("/legacy/v02-exact90")
        self.seen: list[str] = []

    def ensure_verified_asset(
        self, kind: str, *, verify_bytes: bool = False
    ) -> Path:
        assert kind in {"exact90", "formal_inputs"}
        assert verify_bytes is True
        target = getattr(self.layout, kind)
        target.mkdir(parents=True, exist_ok=True)
        return target.resolve()

    def resolve(self, value: str | Path, *, must_exist: bool = True) -> Path:
        raw = str(value)
        self.seen.append(raw)
        if (
            not raw.startswith("/")
            or "\\" in raw
            or "\0" in raw
            or "\n" in raw
            or "\r" in raw
            or any(part in {"", ".", ".."} for part in raw.split("/")[1:])
        ):
            raise ValueError("unsafe recorded path")
        path = Path(raw)
        for prefix in (self._source, self.layout.exact90):
            try:
                suffix = path.relative_to(prefix)
            except ValueError:
                continue
            target = self.layout.exact90 / suffix
            if must_exist and not target.exists():
                raise ValueError("mapped target is missing")
            return target.resolve()
        raise ValueError("unknown relocation")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[_Resolver, dict[str, Any]]:
    resolver = _Resolver(tmp_path / "artifacts")
    layout = resolver.layout
    layout.runs_root.mkdir(parents=True)
    anchor = layout.exact90 / "anchors" / "anchor.json"
    anchor.parent.mkdir(parents=True)
    anchor.write_text("{}\n", encoding="utf-8")
    attempt = layout.runs_root / "jobs" / "job-000" / "attempt_001"
    attempt.mkdir(parents=True)
    bundle = attempt / "checkpoints" / "outer_000183"
    bundle.mkdir(parents=True)

    plan = {
        "jobs": [
            {
                "job_id": "job-000",
                "anchor_manifest_path": "/legacy/v02-exact90/anchors/anchor.json",
            }
        ]
    }
    promotion = {
        "entries": {
            "job-000": {
                "attempt_dir": (
                    "/legacy/v02-exact90/training_private/server_runs/"
                    "jobs/job-000/attempt_001"
                )
            }
        }
    }
    frozen = with_self_digest(
        {
            "schema": "policy-learnware.v02-policy-pool-acceptance.v0",
            "decision": "PASS",
            "accepted_at": "historical",
            "pool_digest": "a" * 64,
            "cells": {
                "job-000": {
                    "job_id": "job-000",
                    "bundle_path": (
                        "/legacy/v02-exact90/training_private/server_runs/"
                        "jobs/job-000/attempt_001/checkpoints/outer_000183"
                    ),
                    "bundle_digest": "b" * 64,
                }
            },
        },
        key="report_digest",
    )
    _write_json(layout.server_plan, plan)
    _write_json(layout.promotions, promotion)
    _write_json(layout.frozen_acceptance, frozen)

    monkeypatch.setattr(
        replay.RelocationResolver,
        "load",
        lambda **_kwargs: resolver,
    )
    return resolver, frozen


def test_replay_routes_anchor_attempt_and_bundle_through_one_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver, frozen = _fixture(tmp_path, monkeypatch)

    def accept(**kwargs: Any) -> dict[str, Any]:
        anchor_path = kwargs["server_plan"]["jobs"][0]["anchor_manifest_path"]
        attempt_path = kwargs["promotion_manifest"]["entries"]["job-000"][
            "attempt_dir"
        ]
        assert kwargs["path_resolver"](anchor_path).name == "anchor.json"
        assert kwargs["path_resolver"](attempt_path).name == "attempt_001"
        result = {
            key: value
            for key, value in frozen.items()
            if key not in {"accepted_at", "report_digest", "cells"}
        }
        result["accepted_at"] = "replay-time"
        result["cells"] = {
            "job-000": {
                **frozen["cells"]["job-000"],
                "bundle_path": str(
                    resolver.layout.runs_root
                    / "jobs"
                    / "job-000"
                    / "attempt_001"
                    / "checkpoints"
                    / "outer_000183"
                ),
            }
        }
        return with_self_digest(result, key="report_digest")

    monkeypatch.setattr(replay, "accept_policy_pool", accept)
    observed = replay.replay_relocated_policy_pool_acceptance(
        artifacts_root=resolver.layout.exact90.parents[2],
        acceptance_path=resolver.layout.frozen_acceptance,
    )
    assert observed == frozen
    assert any(path.endswith("anchors/anchor.json") for path in resolver.seen)
    assert any(path.endswith("attempt_001") for path in resolver.seen)
    assert sum(path.endswith("outer_000183") for path in resolver.seen) == 2


@pytest.mark.parametrize(
    "malicious",
    [
        "/unknown/root/bundle",
        "/legacy/v02-exact90/../escape",
        "/legacy/v02-exact90\\escape",
        "/legacy/v02-exact90/evil\0name",
        "/legacy/v02-exact90/evil\nname",
    ],
)
def test_replay_fails_closed_for_unknown_or_malformed_bundle_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malicious: str,
) -> None:
    resolver, frozen = _fixture(tmp_path, monkeypatch)
    poisoned = dict(frozen)
    poisoned["cells"] = {
        "job-000": {**frozen["cells"]["job-000"], "bundle_path": malicious}
    }
    poisoned = with_self_digest(
        {key: value for key, value in poisoned.items() if key != "report_digest"},
        key="report_digest",
    )
    _write_json(resolver.layout.frozen_acceptance, poisoned)

    def accept(**_kwargs: Any) -> dict[str, Any]:
        return dict(poisoned)

    monkeypatch.setattr(replay, "accept_policy_pool", accept)
    with pytest.raises(ContractError, match="verified relocation"):
        replay.replay_relocated_policy_pool_acceptance(
            artifacts_root=resolver.layout.exact90.parents[2]
        )


def test_replay_never_relaxes_non_path_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver, frozen = _fixture(tmp_path, monkeypatch)
    recomputed = dict(frozen)
    recomputed["pool_digest"] = "c" * 64
    recomputed = with_self_digest(
        {key: value for key, value in recomputed.items() if key != "report_digest"},
        key="report_digest",
    )
    monkeypatch.setattr(replay, "accept_policy_pool", lambda **_kwargs: recomputed)
    with pytest.raises(ContractError, match="differs at pool_digest"):
        replay.replay_relocated_policy_pool_acceptance(
            artifacts_root=resolver.layout.exact90.parents[2]
        )


def test_explicit_acceptance_cannot_select_an_alternate_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver, _frozen = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ContractError, match="outside the canonical exact90"):
        replay.replay_relocated_policy_pool_acceptance(
            artifacts_root=resolver.layout.exact90.parents[2],
            acceptance_path=tmp_path / "alternate" / "policy_pool_acceptance.json",
        )
