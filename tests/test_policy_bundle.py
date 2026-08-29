from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from policy_learnware_v0.policy.bundle import BundleValidationError, validate_bundle
from policy_learnware_v0.policy import loader as loader_module
from policy_learnware_v0.policy.loader import load_policy
from policy_learnware_v0.policy.parity import verify_golden_parity


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(
    path: Path,
    *,
    algorithm: str = "fpo",
    formal_eligible: bool | None = None,
    commit: str = "a" * 40,
    runtime_digest: str | None = None,
) -> None:
    path.mkdir(parents=True)
    observation_dim, action_dim, hidden = 3, 2, 4
    if algorithm == "fpo":
        timestep_dim = 4
        input_dim, output_dim = observation_dim + action_dim + timestep_dim, action_dim
    else:
        timestep_dim = None
        input_dim, output_dim = observation_dim, 2 * action_dim
    np.savez_compressed(
        path / "actor.npz",
        layer_00_kernel=np.zeros((input_dim, hidden), dtype=np.float32),
        layer_00_bias=np.zeros(hidden, dtype=np.float32),
        layer_01_kernel=np.zeros((hidden, output_dim), dtype=np.float32),
        layer_01_bias=np.zeros(output_dim, dtype=np.float32),
    )
    np.savez_compressed(
        path / "obs_stats.npz",
        count=np.asarray(10.0),
        mean=np.zeros(observation_dim),
        var_sum=np.ones(observation_dim),
        std=np.ones(observation_dim),
    )
    raw = np.zeros((8, action_dim), dtype=np.float32)
    np.savez_compressed(
        path / "golden_io.npz",
        observation=np.zeros((8, observation_dim), dtype=np.float32),
        prng_key_data=np.asarray([1, 2], dtype=np.uint32),
        raw_action=raw,
        environment_action=np.tanh(raw),
    )
    common = {
        "schema": "policy-learnware.policy-bundle.v0",
        "algorithm": algorithm,
        "task": "ReacherEasy",
    }
    fpo_inference = {
        "timestep_embed_dim": timestep_dim,
        "flow_steps": 10,
        "feather_std": 0.0,
        "sde_sigma": 0.0,
        "policy_mlp_output_scale": 0.25,
    }
    spec = {
        **common,
        "observation_size": observation_dim,
        "action_size": action_dim,
        "actor_layer_sizes": [input_dim, hidden, output_dim],
        "actor_weights_file": "actor.npz",
        "golden_parity_file": "golden_io.npz",
        "environment_action_transform": "tanh(raw_action)",
        "observation_preprocessing": {
            "statistics_file": "obs_stats.npz",
            "normalize": True,
        },
        "training_config": {
            "episode_length": 1000,
            "normalize_observations": True,
            **(fpo_inference if timestep_dim is not None else {}),
        },
    }
    if timestep_dim is not None:
        spec["inference"] = fpo_inference
    _write_json(path / "policy_spec.json", spec)
    provenance = {
        **common,
        "training_seed": 2,
        "outer_iteration": 6,
        "environment_steps": 5_898_240,
        "fpo_commit": commit,
        "expected_fpo_commit": commit,
        "fpo_commit_matches_expected": True,
        "fpo_tracked_dirty": False,
        "fpo_tracked_changes": [],
    }
    if formal_eligible is not None:
        provenance["formal_eligible"] = formal_eligible
    if runtime_digest is not None:
        provenance["runtime_digest"] = runtime_digest
    _write_json(path / "provenance.json", provenance)
    payloads = (
        "actor.npz",
        "obs_stats.npz",
        "golden_io.npz",
        "policy_spec.json",
        "provenance.json",
    )
    _write_json(
        path / "bundle_manifest.json",
        {
            **common,
            "complete": True,
            "seed": 2,
            "outer_iteration": 6,
            "environment_steps": 5_898_240,
            "files": {
                name: {"bytes": (path / name).stat().st_size, "sha256": _sha(path / name)}
                for name in payloads
            },
        },
    )


@pytest.mark.parametrize("algorithm", ["ppo", "fpo"])
def test_bundle_validation_covers_both_frozen_policy_runtimes(
    tmp_path: Path, algorithm: str
) -> None:
    bundle = tmp_path / "outer_000006"
    _bundle(bundle, algorithm=algorithm)
    metadata = validate_bundle(bundle, expected_algorithm=algorithm, expected_outer=6)
    assert metadata.action_dim == 2
    assert metadata.observation_dim == 3
    with (bundle / "actor.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(BundleValidationError, match="size mismatch"):
        validate_bundle(bundle)


def test_external_runtime_authority_is_required_for_formal_bundle(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "outer_000006"
    _bundle(
        bundle,
        formal_eligible=True,
        commit="b" * 40,
        runtime_digest="c" * 64,
    )
    with pytest.raises(BundleValidationError, match="requires external commit"):
        validate_bundle(bundle)
    validate_bundle(
        bundle,
        expected_fpo_commit="b" * 40,
        expected_runtime_digest="c" * 64,
    )


def test_injected_runtime_loads_and_replays_golden_actions(tmp_path: Path) -> None:
    bundle = tmp_path / "outer_000006"
    _bundle(
        bundle,
        formal_eligible=True,
        commit="b" * 40,
        runtime_digest="c" * 64,
    )

    class Policy:
        observation_dim = 3
        action_dim = 2

        def act_raw(self, observation, key, *, deterministic=True):
            assert deterministic
            return np.zeros((np.asarray(observation).shape[0], 2)), key

        def act(self, observation, key, *, deterministic=True):
            raw, next_key = self.act_raw(
                observation, key, deterministic=deterministic
            )
            return np.tanh(raw), next_key

    def runtime_factory(metadata, actor, obs_stats, fpo_root):
        assert metadata.bundle_dir == bundle.resolve()
        assert sorted(actor) == [
            "layer_00_bias",
            "layer_00_kernel",
            "layer_01_bias",
            "layer_01_kernel",
        ]
        assert sorted(obs_stats) == ["count", "mean", "std", "var_sum"]
        assert fpo_root == tmp_path / "fpo"
        return Policy()

    policy = load_policy(
        bundle,
        fpo_root=tmp_path / "fpo",
        runtime_factory=runtime_factory,
        expected_fpo_commit="b" * 40,
        expected_runtime_digest="c" * 64,
    )
    report = verify_golden_parity(
        policy,
        bundle,
        expected_fpo_commit="b" * 40,
        expected_runtime_digest="c" * 64,
    )
    assert report.passed is True
    assert report.raw_checked is True
    assert report.sample_count == 8


@pytest.mark.parametrize("numeric", [0, 1])
def test_numeric_formal_flag_cannot_bypass_boolean_contract(
    tmp_path: Path, numeric: int
) -> None:
    bundle = tmp_path / f"case-{numeric}" / "outer_000006"
    _bundle(bundle)
    provenance_path = bundle / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["formal_eligible"] = numeric
    _write_json(provenance_path, provenance)
    manifest_path = bundle / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["provenance.json"] = {
        "bytes": provenance_path.stat().st_size,
        "sha256": _sha(provenance_path),
    }
    _write_json(manifest_path, manifest)
    with pytest.raises(BundleValidationError, match="formal_eligible must be boolean"):
        validate_bundle(bundle)


def test_default_factory_uses_public_reconstructed_runtime_bridge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt = MappingProxyType(
        {
            "runtime_status": "RECONSTRUCTED_RUNTIME",
            "training_replay_capable": False,
        }
    )
    state = SimpleNamespace(
        params=SimpleNamespace(policy=None),
        obs_stats=SimpleNamespace(count=None, mean=None, var_sum=None, std=None),
    )

    class Config:
        def __init__(self, **values):
            self.values = values

    class State:
        @classmethod
        def init(cls, *, prng, env, config):
            assert prng == ("key", 0)
            assert env == "loaded-env"
            assert isinstance(config, Config)
            return state

    @contextmanager
    def copy_and_mutate(value):
        assert value is state
        yield value

    upstream = SimpleNamespace(
        source_attestation={"fpo_commit": "a" * 40},
        runtime_receipt=receipt,
        jax=SimpleNamespace(random=SimpleNamespace(key=lambda seed: ("key", seed))),
        jax_numpy=np,
        jax_dataclasses=SimpleNamespace(copy_and_mutate=copy_and_mutate),
        registry=SimpleNamespace(
            get_default_config=lambda task: {"task": task},
            load=lambda task, config: "loaded-env",
        ),
        fpo=SimpleNamespace(FpoConfig=Config, FpoState=State),
        ppo=SimpleNamespace(PpoConfig=Config, PpoState=State),
    )
    calls: list[tuple[Path, bool]] = []

    class RuntimeVerificationError(RuntimeError):
        pass

    runtime_module = SimpleNamespace(
        RuntimeVerificationError=RuntimeVerificationError,
        load_verified_fpo_upstream=lambda root, allow_reconstructed: (
            calls.append((Path(root), allow_reconstructed)) or upstream
        ),
    )
    monkeypatch.setattr(
        loader_module.importlib,
        "import_module",
        lambda name: runtime_module
        if name == "policy_learnware_v0.v02.runtime"
        else (_ for _ in ()).throw(AssertionError(name)),
    )
    metadata = SimpleNamespace(
        algorithm="fpo",
        task="ReacherEasy",
        provenance={"fpo_commit": "a" * 40},
        policy_spec={"training_config": {"episode_length": 1000}},
    )
    actor = {
        "layer_00_kernel": np.zeros((2, 3)),
        "layer_00_bias": np.zeros(3),
    }
    stats = {
        "count": np.asarray(1.0),
        "mean": np.zeros(2),
        "var_sum": np.ones(2),
        "std": np.ones(2),
    }

    restored = loader_module._default_runtime_factory(
        metadata, actor, stats, tmp_path / "fpo"
    )

    assert calls == [(tmp_path / "fpo", True)]
    assert restored.state is state
    assert restored.runtime_receipt is receipt
    assert state.params.policy[0][0].shape == (2, 3)
    np.testing.assert_array_equal(state.obs_stats.mean, stats["mean"])
