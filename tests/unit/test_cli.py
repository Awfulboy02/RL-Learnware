from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from policy_learnware_v0.artifacts import (  # noqa: E402
    ArtifactLayout,
    ArtifactLayoutError,
)
from policy_learnware_v0.cli import (  # noqa: E402
    COMMANDS,
    ENVIRONMENT_MANIFEST_SCHEMA,
    CommandFailure,
    _championization_evaluator_contract,
    _championization_seed_vectors,
    _deployment_pair_plan,
    _exclusive_championization_lock,
    _parse_championization_worker_output,
    _resolve_championization_devices,
    _validate_championization_candidate_shard,
    _handle_evaluate_retrieval,
    _query_id,
    main,
)
from policy_learnware_v0.config import ProtocolDraft, load_protocol_draft  # noqa: E402
from policy_learnware_v0.envs.base import SyntheticEnvAdapter  # noqa: E402
from policy_learnware_v0.envs.inspect import (  # noqa: E402
    EnvironmentInspection,
    save_inspections,
)
from policy_learnware_v0.representation.normalization import (  # noqa: E402
    NormalizationStats,
)
from policy_learnware_v0.policy.parity import ParityReport  # noqa: E402
from policy_learnware_v0.probe.collector import (  # noqa: E402
    collect_probe_episodes as real_collect_probe_episodes,
)
from policy_learnware_v0.probe.dataset import EpisodeDataset  # noqa: E402
from policy_learnware_v0.probe.seed_plan import SeedPlan  # noqa: E402
from policy_learnware_v0.pool.learnware import (  # noqa: E402
    LearnwarePool,
    SelectorEntry,
    SelectorTaskSpec,
)
from policy_learnware_v0.reuse.selector import (  # noqa: E402
    NearestSpecSelector,
    target_source_cross_terms as real_target_source_cross_terms,
)
from policy_learnware_v0.evaluation.retrieval_accel import (  # noqa: E402
    nested_prefix_self_kernel_sums_jax as real_nested_prefix_self_kernel_sums_jax,
)
from policy_learnware_v0.rkme.empirical import (  # noqa: E402
    build_empirical_kme as real_build_empirical_kme,
)
from policy_learnware_v0.rkme.gaussian import GaussianKernel  # noqa: E402
from policy_learnware_v0.rkme.reducer import ReducedRKME  # noqa: E402


def _invoke(arguments: list[str]) -> tuple[int, dict, dict | None]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        returncode = main(arguments)
    output = json.loads(stdout.getvalue()) if stdout.getvalue().strip() else {}
    error = json.loads(stderr.getvalue()) if stderr.getvalue().strip() else None
    return returncode, output, error


def _write_synthetic_config(path: Path, *, reproduction_root: Path | None = None) -> None:
    payload = yaml.safe_load((PROJECT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    payload["environment"].update(
        {
            "backend": "synthetic.test-only",
            "tasks": ["TaskA", "TaskB"],
            "horizon": 3,
            "max_observation_dim": 3,
            "max_action_dim": 2,
        }
    )
    payload["encoder"]["batch_size"] = 4
    if reproduction_root is not None:
        payload["runtime"]["reproduction_root"] = str(reproduction_root)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _inspection(adapter: SyntheticEnvAdapter) -> EnvironmentInspection:
    state, observation = adapter.reset(0)
    action = np.zeros(adapter.schema.action_dim, dtype=np.float32)
    _, result = adapter.step(state, action)
    return EnvironmentInspection(
        schema=adapter.schema,
        registry_config={"test_only": True},
        reset_seed=0,
        reset_observation=observation,
        fixed_action=action,
        next_observation=result.observation,
        reward=result.reward,
        terminated=result.terminated,
        truncated=result.truncated,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retrieval_pool(pool_id: str) -> LearnwarePool:
    def spec(location: float) -> SelectorTaskSpec:
        reduced = ReducedRKME(
            supports=np.array([[location]], dtype=np.float64),
            beta=np.array([1.0], dtype=np.float64),
            bandwidth=1.0,
            rkme_norm2=1.0,
            empirical_norm2=1.0,
            reduction_error=0.0,
            protocol_id="protocol",
        )
        return SelectorTaskSpec.from_rkme(
            reduced, protocol_id="protocol", kernel_bandwidth=1.0
        )

    return LearnwarePool(
        pool_id=pool_id,
        protocol_id="protocol",
        kernel_bandwidth=1.0,
        entries=(
            SelectorEntry("lw-" + "a" * 20, "protocol", spec(0.0)),
            SelectorEntry("lw-" + "b" * 20, "protocol", spec(5.0)),
        ),
    )


def _target_query_dataset(
    config: ProtocolDraft, *, task_index: int, bank: int
) -> EpisodeDataset:
    location = float(task_index * 5)
    seeds = SeedPlan(config.project_seed).episodes(
        "target_query",
        task_index,
        range(config.episodes.target_query_max_per_task),
        bank_index=bank,
    )
    episode_count = config.episodes.target_query_max_per_task
    length = 2
    transitions = episode_count * length
    truncated = np.zeros(transitions, dtype=np.bool_)
    truncated[length - 1 :: length] = True
    return EpisodeDataset(
        observation=np.full((transitions, 1), location, dtype=np.float32),
        action=np.zeros((transitions, 1), dtype=np.float32),
        reward=np.zeros(transitions, dtype=np.float32),
        next_observation=np.full((transitions, 1), location, dtype=np.float32),
        terminated=np.zeros(transitions, dtype=np.bool_),
        truncated=truncated,
        episode_offsets=np.arange(
            0, transitions + 1, length, dtype=np.int64
        ),
        reset_seeds=np.asarray([item.reset_seed for item in seeds], dtype=np.int64),
        probe_seeds=np.asarray([item.probe_seed for item in seeds], dtype=np.int64),
    )


def _make_bundle(
    path: Path,
    *,
    task: str,
    algorithm: str,
    seed: int,
    observation_dim: int,
    action_dim: int,
) -> None:
    path.mkdir(parents=True)
    input_dim = observation_dim if algorithm == "ppo" else observation_dim + action_dim + 8
    output_dim = 2 * action_dim if algorithm == "ppo" else action_dim
    np.savez_compressed(
        path / "actor.npz",
        layer_00_kernel=np.zeros((input_dim, 4), dtype=np.float32),
        layer_00_bias=np.zeros(4, dtype=np.float32),
        layer_01_kernel=np.zeros((4, output_dim), dtype=np.float32),
        layer_01_bias=np.zeros(output_dim, dtype=np.float32),
    )
    np.savez_compressed(
        path / "obs_stats.npz",
        count=np.asarray(1.0, dtype=np.float32),
        mean=np.zeros(observation_dim, dtype=np.float32),
        std=np.ones(observation_dim, dtype=np.float32),
        var_sum=np.zeros(observation_dim, dtype=np.float32),
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
        "task": task,
        "algorithm": algorithm,
    }
    policy_spec = {
        **common,
        "observation_size": observation_dim,
        "action_size": action_dim,
        "actor_layer_sizes": [input_dim, 4, output_dim],
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
        },
    }
    if algorithm == "fpo":
        policy_spec["inference"] = {
            "timestep_embed_dim": 8,
            "flow_steps": 10,
            "feather_std": 0.0,
            "sde_sigma": 0.0,
            "policy_mlp_output_scale": 0.25,
        }
        policy_spec["training_config"].update(policy_spec["inference"])
    _write_json(path / "policy_spec.json", policy_spec)
    _write_json(
        path / "provenance.json",
        {
            **common,
            "training_seed": seed,
            "outer_iteration": 6,
            "environment_steps": 5_898_240,
            "fpo_commit": "a" * 40,
            "expected_fpo_commit": "a" * 40,
            "fpo_commit_matches_expected": True,
            "fpo_tracked_dirty": False,
            "fpo_tracked_changes": [],
        },
    )
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
            "seed": seed,
            "outer_iteration": 6,
            "environment_steps": 5_898_240,
            "files": {
                name: {
                    "bytes": (path / name).stat().st_size,
                    "sha256": _sha256(path / name),
                }
                for name in payloads
            },
        },
    )


class ArtifactLayoutTest(unittest.TestCase):
    def test_layout_rejects_escape_and_resume_requires_identical_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = ArtifactLayout(Path(directory), "pool")
            digest = layout.publish_json(layout.smoke_report, {"passed": True})
            self.assertEqual(len(digest), 64)
            self.assertEqual(
                layout.publish_json(
                    layout.smoke_report, {"passed": True}, resume=True
                ),
                digest,
            )
            with self.assertRaises(ArtifactLayoutError):
                layout.publish_json(
                    layout.smoke_report, {"passed": False}, resume=True
                )
            with self.assertRaises(ArtifactLayoutError):
                layout.publish_json(Path(directory).parent / "escape.json", {})

    def test_candidate_shard_paths_reject_unsafe_job_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = ArtifactLayout(Path(directory), "pool")
            self.assertEqual(
                layout.championization_candidate("job-1").name, "job-1.json"
            )
            with self.assertRaises(ArtifactLayoutError):
                layout.championization_candidate("../escape")


class CliTest(unittest.TestCase):
    def test_championization_devices_are_strict_and_unique(self) -> None:
        self.assertEqual(_resolve_championization_devices("0,2,7"), ("0", "2", "7"))
        with self.assertRaises(CommandFailure):
            _resolve_championization_devices("0,0")
        with self.assertRaises(CommandFailure):
            _resolve_championization_devices("0,gpu1")

    def test_championization_auto_respects_visible_device_scope(self) -> None:
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"},
            clear=False,
        ):
            self.assertEqual(
                _resolve_championization_devices("auto"), ("GPU-a", "GPU-b")
            )

    def test_championization_lock_rejects_a_second_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "championization.lock"
            with _exclusive_championization_lock(path):
                with self.assertRaises(CommandFailure):
                    with _exclusive_championization_lock(path):
                        pass

    def test_championization_worker_output_ignores_third_party_noise(self) -> None:
        payload = {
            "schema": "policy-learnware.cli-result.v0",
            "command": "championize",
            "status": "ok",
            "result": {"worker": True, "shard_index": 0, "shard_count": 1},
        }
        stdout = (
            "Failed to import optional backend\n"
            + json.dumps(payload, indent=2)
            + "\n"
        )
        self.assertEqual(_parse_championization_worker_output(stdout), payload)
        with self.assertRaises(CommandFailure):
            _parse_championization_worker_output(stdout + json.dumps(payload))
        with self.assertRaises(CommandFailure):
            _parse_championization_worker_output("warning only")

    def test_candidate_checkpoint_is_hash_bound_and_tamper_evident(self) -> None:
        config = load_protocol_draft(PROJECT / "configs" / "smoke.yaml")
        item = {
            "job_id": "full__FingerSpin__fpo__seed0",
            "bundle_digest": "a" * 64,
            "task": "FingerSpin",
            "algorithm": "fpo",
            "training_seed": 0,
        }
        reset_seeds, policy_seeds = _championization_seed_vectors(item, config)
        contract = _championization_evaluator_contract()
        payload = {
            "schema": "policy-learnware.championization-candidate.v0",
            "complete": True,
            "protocol_draft_hash": config.draft_hash,
            "inventory_sha256": "b" * 64,
            "verification_sha256": "c" * 64,
            "environment_manifest_sha256": "d" * 64,
            "candidate_index": 0,
            **item,
            "reset_seeds": list(reset_seeds),
            "policy_seeds": list(policy_seeds),
            "episode_returns": [1.0, 2.0],
            "parity": {
                "passed": True,
                "atol": config.policy.parity_atol,
                "rtol": config.policy.parity_rtol,
            },
            "compiled_parity": {
                "passed": True,
                "next_keys_equal": True,
                "atol": config.policy.parity_atol,
                "rtol": config.policy.parity_rtol,
            },
            "evaluator_contract": contract,
        }
        validated = _validate_championization_candidate_shard(
            payload,
            item=item,
            candidate_index=0,
            config=config,
            inventory_sha256="b" * 64,
            verification_sha256="c" * 64,
            environment_manifest_sha256="d" * 64,
            evaluator_contract=contract,
        )
        self.assertEqual(validated["episode_returns"], [1.0, 2.0])
        tampered = json.loads(json.dumps(payload))
        tampered["evaluator_contract"]["execution"] = "scalar"
        with self.assertRaises(ArtifactLayoutError):
            _validate_championization_candidate_shard(
                tampered,
                item=item,
                candidate_index=0,
                config=config,
                inventory_sha256="b" * 64,
                verification_sha256="c" * 64,
                environment_manifest_sha256="d" * 64,
                evaluator_contract=contract,
            )

    def test_all_commands_have_side_effect_free_dry_run_contracts(self) -> None:
        extras = {
            "collect-probe": ["--split", "encoder_train"],
            "inventory-policies": ["--outer", "6"],
            "championize": ["--outer", "6"],
        }
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            for command in COMMANDS:
                returncode, payload, error = _invoke(
                    [
                        command,
                        "--config",
                        str(PROJECT / "configs" / "smoke.yaml"),
                        "--artifacts-root",
                        str(artifacts),
                        "--dry-run",
                        *extras.get(command, []),
                    ]
                )
                self.assertEqual(returncode, 0, (command, error))
                self.assertIsNone(error)
                self.assertEqual(payload["command"], command)
                self.assertEqual(payload["status"], "dry_run")
                self.assertEqual(len(payload["protocol_draft_hash"]), 64)
                self.assertIn("inputs", payload)
                self.assertIn("outputs", payload)
                self.assertIn("seed_ranges", payload)
                self.assertFalse(payload["will_write"])
                self.assertFalse(payload["will_execute_gpu_work"])
            self.assertFalse(artifacts.exists(), "dry-run created an artifact directory")

    def test_missing_prerequisites_fail_closed_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            returncode, output, error = _invoke(
                [
                    "championize",
                    "--config",
                    str(PROJECT / "configs" / "smoke.yaml"),
                    "--artifacts-root",
                    str(artifacts),
                    "--outer",
                    "6",
                ]
            )
            self.assertEqual(returncode, 1)
            self.assertEqual(output, {})
            assert error is not None
            self.assertTrue(error["fail_closed"])
            self.assertEqual(error["error_type"], "CommandFailure")
            self.assertFalse(artifacts.exists())

    def test_smoke_is_atomic_immutable_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = [
                "smoke",
                "--config",
                str(PROJECT / "configs" / "smoke.yaml"),
                "--artifacts-root",
                str(Path(directory) / "artifacts"),
            ]
            with patch("policy_learnware_v0.cli.run_logic_smoke") as smoke:
                smoke.return_value.passed = True
                smoke.return_value.to_dict.return_value = {
                    "passed": True,
                    "checks": {"cli_test": True},
                }
                returncode, payload, error = _invoke(base)
            self.assertEqual(returncode, 0, error)
            report = Path(payload["result"]["report"])
            self.assertTrue(report.is_file())
            self.assertTrue(json.loads(report.read_text(encoding="utf-8"))["passed"])

            returncode, _, error = _invoke(base)
            self.assertEqual(returncode, 1)
            assert error is not None
            self.assertEqual(error["error_type"], "ArtifactExistsError")

            returncode, payload, error = _invoke([*base, "--resume"])
            self.assertEqual(returncode, 0, error)
            self.assertTrue(payload["result"]["resumed"])

    def test_collect_probe_then_fit_normalizer_uses_managed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "synthetic.yaml"
            _write_synthetic_config(config_path)
            config = load_protocol_draft(config_path)
            layout = ArtifactLayout(root / "artifacts", config.pool.pool_id)
            adapters = {
                "TaskA": SyntheticEnvAdapter(
                    task="TaskA", observation_dim=2, action_dim=1, horizon=3
                ),
                "TaskB": SyntheticEnvAdapter(
                    task="TaskB", observation_dim=3, action_dim=2, horizon=3
                ),
            }
            inspection_digests = save_inspections(
                {task: _inspection(adapter) for task, adapter in adapters.items()},
                layout.protocol_dir,
            )
            layout.publish_json(
                layout.environment_manifest,
                {
                    "schema": ENVIRONMENT_MANIFEST_SCHEMA,
                    "complete": True,
                    "protocol_draft_hash": config.draft_hash,
                    "pool_id": config.pool.pool_id,
                    "files": {
                        "env_schemas": {
                            "path": layout.relative(layout.env_schemas),
                            "sha256": inspection_digests["env_schemas"],
                        },
                        "env_golden_io": {
                            "path": layout.relative(layout.env_golden_io),
                            "sha256": inspection_digests["env_golden_io"],
                        },
                    },
                },
            )

            common = [
                "--config",
                str(config_path),
                "--artifacts-root",
                str(root / "artifacts"),
            ]
            with patch(
                "policy_learnware_v0.cli.make_env_adapter",
                side_effect=lambda task, _config, **_kwargs: adapters[task],
            ), patch(
                "policy_learnware_v0.cli.collect_probe_episodes",
                wraps=real_collect_probe_episodes,
            ) as collector:
                returncode, payload, error = _invoke(
                    ["collect-probe", *common, "--split", "encoder_train"]
                )
            self.assertEqual(returncode, 0, error)
            self.assertTrue(
                all(
                    isinstance(call.args[3], ProtocolDraft)
                    for call in collector.call_args_list
                )
            )
            self.assertEqual(len(payload["result"]["published"]), 2)
            for task in adapters:
                self.assertTrue(layout.dataset_npz("encoder_train", task).is_file())
                self.assertTrue(layout.dataset_manifest("encoder_train", task).is_file())

            with patch(
                "policy_learnware_v0.cli.make_env_adapter",
                side_effect=AssertionError("resume must not construct environments"),
            ):
                returncode, payload, error = _invoke(
                    [
                        "collect-probe",
                        *common,
                        "--split",
                        "encoder_train",
                        "--resume",
                    ]
                )
            self.assertEqual(returncode, 0, error)
            self.assertEqual(len(payload["result"]["resumed"]), 2)

            returncode, payload, error = _invoke(["fit-normalizer", *common])
            self.assertEqual(returncode, 0, error)
            self.assertTrue(layout.normalization.is_file())
            self.assertTrue(layout.normalization_manifest.is_file())
            stats = NormalizationStats.load_npz(layout.normalization)
            self.assertEqual(stats.source_task_count, 2)
            self.assertEqual(stats.max_observation_dim, 3)
            self.assertEqual(len(payload["result"]["normalization_sha256"]), 64)

    def test_inventory_and_bundle_verification_orchestration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reproduction = root / "reproduction"
            config_path = root / "synthetic.yaml"
            _write_synthetic_config(config_path, reproduction_root=reproduction)
            config = load_protocol_draft(config_path)
            adapters = {
                "TaskA": SyntheticEnvAdapter(
                    task="TaskA", observation_dim=2, action_dim=1, horizon=3
                ),
                "TaskB": SyntheticEnvAdapter(
                    task="TaskB", observation_dim=3, action_dim=2, horizon=3
                ),
            }

            jobs: list[dict] = []
            for seed in range(5):
                for task in config.environment.tasks:
                    for algorithm in ("ppo", "fpo"):
                        job_id = f"full__{task}__{algorithm}__seed{seed}"
                        job = {
                            "job_id": job_id,
                            "phase": "full",
                            "task": task,
                            "algorithm": algorithm,
                            "seed": seed,
                            "export_outers": [6],
                        }
                        jobs.append(job)
                        job_dir = reproduction / "runs" / "full" / job_id
                        attempt = job_dir / "attempt_01"
                        result = {"state": "succeeded", "attempt": 1, "returncode": 0}
                        _write_json(job_dir / "queue_result.json", result)
                        _write_json(attempt / "queue_result.json", result)
                        _write_json(attempt / "queue_job.json", {"job": job})
                        schema = adapters[task].schema
                        _make_bundle(
                            attempt / "checkpoints" / "outer_000006",
                            task=task,
                            algorithm=algorithm,
                            seed=seed,
                            observation_dim=schema.observation_dim,
                            action_dim=schema.action_dim,
                        )
            _write_json(reproduction / "jobs_manifest.json", {"full": jobs})

            artifacts_root = root / "artifacts"
            common = [
                "--config",
                str(config_path),
                "--artifacts-root",
                str(artifacts_root),
            ]
            returncode, payload, error = _invoke(
                ["inventory-policies", *common, "--outer", "6"]
            )
            self.assertEqual(returncode, 0, error)
            self.assertEqual(payload["result"]["item_count"], 20)

            layout = ArtifactLayout(artifacts_root, config.pool.pool_id)
            inspection_digests = save_inspections(
                {task: _inspection(adapter) for task, adapter in adapters.items()},
                layout.protocol_dir,
            )
            layout.publish_json(
                layout.environment_manifest,
                {
                    "schema": ENVIRONMENT_MANIFEST_SCHEMA,
                    "complete": True,
                    "protocol_draft_hash": config.draft_hash,
                    "pool_id": config.pool.pool_id,
                    "files": {
                        "env_schemas": {
                            "path": layout.relative(layout.env_schemas),
                            "sha256": inspection_digests["env_schemas"],
                        },
                        "env_golden_io": {
                            "path": layout.relative(layout.env_golden_io),
                            "sha256": inspection_digests["env_golden_io"],
                        },
                    },
                },
            )
            parity = ParityReport(
                passed=True,
                raw_checked=True,
                raw_max_abs_error=0.0,
                environment_max_abs_error=0.0,
                atol=1.0e-6,
                rtol=1.0e-6,
                sample_count=2,
            )
            with patch(
                "policy_learnware_v0.cli._verify_fpo_checkout",
                return_value={"root": "synthetic", "commit": config.runtime.fpo_commit},
            ), patch(
                "policy_learnware_v0.cli.load_policy", return_value=object()
            ), patch(
                "policy_learnware_v0.cli.verify_golden_parity", return_value=parity
            ):
                returncode, payload, error = _invoke(
                    ["verify-policy-bundles", *common]
                )
            self.assertEqual(returncode, 0, error)
            self.assertEqual(payload["result"]["verified_count"], 20)
            self.assertTrue(layout.bundle_verification.is_file())
            self.assertEqual(len(tuple(layout.parity_reports_dir.glob("*.json"))), 20)

    def test_retrieval_resume_reuses_verified_query_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "synthetic.yaml"
            _write_synthetic_config(config_path)
            config = load_protocol_draft(config_path)
            layout = ArtifactLayout(root / "artifacts", config.pool.pool_id)
            pool = _retrieval_pool(config.pool.pool_id)
            kernel = GaussianKernel(1.0)
            datasets = {
                (task, bank): _target_query_dataset(
                    config, task_index=task_index, bank=bank
                )
                for bank in range(config.episodes.target_query_banks)
                for task_index, task in enumerate(config.environment.tasks)
            }
            expected_by_task = {
                "TaskA": "lw-" + "a" * 20,
                "TaskB": "lw-" + "b" * 20,
            }
            pool_manifest = {
                "entries": {
                    task: {"opaque_id": opaque_id}
                    for task, opaque_id in expected_by_task.items()
                }
            }

            # Model an interruption after the first atomic selection publish but
            # before retrieval_metrics.json exists.
            first_dataset = datasets[("TaskA", 0)]
            first_prefix = first_dataset.prefix(1)
            first_empirical = real_build_empirical_kme(
                SimpleNamespace(
                    points=first_prefix.observation.astype(np.float64),
                    episode_offsets=first_prefix.episode_offsets,
                ),
                kernel,
                protocol_id="protocol",
                dataset_digest=first_prefix.digest,
            )
            first_selection = NearestSpecSelector(pool).select(first_empirical)
            first_path = layout.selection_result(_query_id("TaskA", 0, 1))
            layout.publish_json(first_path, first_selection.to_dict())
            first_digest = _sha256(first_path)
            layout.publish_json(layout.encoder_config, {})

            def load_dataset(npz_path: Path, _manifest_path: Path):
                task = Path(npz_path).stem
                bank = int(Path(npz_path).parent.name.removeprefix("bank_"))
                dataset = datasets[(task, bank)]
                return dataset, SimpleNamespace(
                    task=task,
                    split="target_query",
                    protocol_draft_hash=config.draft_hash,
                )

            class FakeCanonicalizer:
                def __init__(self, **_kwargs):
                    pass

                def pack(self, dataset: EpisodeDataset, _schema: object):
                    return SimpleNamespace(packed=dataset.observation)

            class FakeEncoder:
                def __init__(self, _checkpoint: object):
                    pass

                def encode(self, packed: np.ndarray) -> np.ndarray:
                    return np.asarray(packed, dtype=np.float64)

            common_patches = (
                patch(
                    "policy_learnware_v0.cli._load_frozen_protocol",
                    return_value=SimpleNamespace(protocol_id="protocol"),
                ),
                patch("policy_learnware_v0.cli.load_public_pool", return_value=pool),
                patch(
                    "policy_learnware_v0.cli._load_verified_pool_build_manifest",
                    return_value=pool_manifest,
                ),
                patch(
                    "policy_learnware_v0.cli.load_dataset_artifact",
                    side_effect=load_dataset,
                ),
                patch(
                    "policy_learnware_v0.cli._load_env_schemas",
                    return_value={task: object() for task in config.environment.tasks},
                ),
                patch(
                    "policy_learnware_v0.cli.NormalizationStats.load_npz",
                    return_value=object(),
                ),
                patch(
                    "policy_learnware_v0.cli.TransitionCanonicalizer",
                    FakeCanonicalizer,
                ),
                patch(
                    "policy_learnware_v0.cli.EncoderCheckpoint.load",
                    return_value=object(),
                ),
                patch("policy_learnware_v0.cli.TransitionSemanticEncoder", FakeEncoder),
                patch(
                    "policy_learnware_v0.cli.GaussianKernel.load_json",
                    return_value=kernel,
                ),
                patch(
                    "policy_learnware_v0.cli.nested_prefix_self_kernel_sums_jax",
                    wraps=real_nested_prefix_self_kernel_sums_jax,
                ),
                patch(
                    "policy_learnware_v0.cli.target_source_cross_terms",
                    wraps=real_target_source_cross_terms,
                ),
            )
            entered = [item.start() for item in common_patches]
            try:
                nested_mock = entered[-2]
                cross_mock = entered[-1]
                result = _handle_evaluate_retrieval(
                    SimpleNamespace(resume=True), config, layout
                )
                self.assertEqual(result["resumed_query_count"], 1)
                self.assertEqual(result["computed_query_count"], 7)
                self.assertEqual(nested_mock.call_count, 4)
                self.assertEqual(cross_mock.call_count, 7)
                self.assertEqual(_sha256(first_path), first_digest)
                self.assertTrue(layout.retrieval_metrics.is_file())

                completed = _handle_evaluate_retrieval(
                    SimpleNamespace(resume=True), config, layout
                )
                self.assertTrue(completed["resumed"])
                self.assertEqual(completed["resumed_query_count"], 8)
                self.assertEqual(completed["computed_query_count"], 0)
                self.assertEqual(nested_mock.call_count, 4)
                self.assertEqual(cross_mock.call_count, 7)
            finally:
                for item in reversed(common_patches):
                    item.stop()

    def test_console_script_is_declared(self) -> None:
        project = (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('[project.scripts]', project)
        self.assertIn('policy-learnware-v0 = "policy_learnware_v0.cli:main"', project)

    def test_deployment_pair_plan_is_unique_deterministic_and_query_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.yaml"
            _write_synthetic_config(config_path)
            config = load_protocol_draft(config_path)
            first_task, second_task = config.environment.tasks[:2]
            queries = [
                {
                    "target_task": second_task,
                    "selected_opaque_id": "lw-second",
                    "query_id": "q-2",
                    "selection_sha256": "2" * 64,
                },
                {
                    "target_task": first_task,
                    "selected_opaque_id": "lw-first",
                    "query_id": "q-1b",
                    "selection_sha256": "b" * 64,
                },
                {
                    "target_task": first_task,
                    "selected_opaque_id": "lw-first",
                    "query_id": "q-1a",
                    "selection_sha256": "a" * 64,
                },
            ]
            plan = _deployment_pair_plan(queries, config)
            self.assertEqual(len(plan), 2)
            self.assertEqual(plan[0]["target_task"], first_task)
            self.assertEqual(plan[1]["target_task"], second_task)
            self.assertEqual(
                [item["query_id"] for item in plan[0]["queries"]],
                ["q-1a", "q-1b"],
            )
            self.assertEqual(plan, _deployment_pair_plan(tuple(reversed(queries)), config))


if __name__ == "__main__":
    unittest.main()
