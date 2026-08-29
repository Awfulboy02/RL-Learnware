from argparse import Namespace
import json
from pathlib import Path

from policy_learnware_v0.v03.artifact_paths import (
    ARTIFACTS_ROOT_ENV,
    V03ArtifactLayout,
    resolve_artifacts_root,
    resolve_recorded_path,
)
from server.repro_fpo_ppo_v03 import development_baseline_runner
from server.repro_fpo_ppo_v03 import v031_raw_transition_runner


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


def test_relocation_manifest_preserves_immutable_recorded_path(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    relocated = root / "v02/exact90/policies/policy-7"
    relocated.mkdir(parents=True)
    manifest = tmp_path / "relocation.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "old_path": "/share/songyf/RL_Learnware/v02_formal_artifacts",
                        "new_path": "v02/exact90",
                        "content_digest": "0" * 64,
                        "role": "policy_pool",
                        "access_class": "restricted",
                        "verification_status": "verified",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_recorded_path(
        "/share/songyf/RL_Learnware/v02_formal_artifacts/policies/policy-7",
        relocation_manifest=manifest,
        artifacts_root=root,
    )
    assert resolved == relocated.resolve()


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
