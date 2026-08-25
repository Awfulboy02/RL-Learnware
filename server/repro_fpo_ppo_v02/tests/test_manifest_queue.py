from __future__ import annotations

from pathlib import Path
import base64
import hashlib
import os
import sys
import tempfile
import unittest

from repro_fpo_ppo_v02.generate_manifest import build_training_plan
from repro_fpo_ppo_v02.implementation import inspect_implementation_inventory
from repro_fpo_ppo_v02.provenance import (
    ContractError,
    atomic_write_json,
    load_strict_json,
    with_self_digest,
)
from repro_fpo_ppo_v02.queue_master import main as queue_main
from repro_fpo_ppo_v02.tests.helpers import make_protocol, make_shifted_anchor
from repro_fpo_ppo_v02.vendor import inspect_vendor_directory


def _make_fake_vendor(path: Path) -> None:
    package = path / "wandb"
    distribution = path / "wandb-0.21.0.dist-info"
    package.mkdir(parents=True)
    distribution.mkdir()
    init_bytes = b'__version__ = "0.21.0"\n'
    (package / "__init__.py").write_bytes(init_bytes)
    (path / "dependency.py").write_text("PIN = 1\n", encoding="utf-8")
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: wandb\nVersion: 0.21.0\n",
        encoding="utf-8",
    )
    encoded = base64.urlsafe_b64encode(hashlib.sha256(init_bytes).digest()).rstrip(b"=")
    (distribution / "RECORD").write_text(
        f"wandb/__init__.py,sha256={encoded.decode('ascii')},{len(init_bytes)}\n",
        encoding="utf-8",
    )


def _write_fake_runner(path: Path, repository_root: Path) -> None:
    source = f'''#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, {str(repository_root)!r})

from repro_fpo_ppo_v02.anchor_binding import AnchorManifest
from repro_fpo_ppo_v02.provenance import (
    EXECUTION_EVIDENCE_SCHEMA,
    TRAINING_RECORD_SCHEMA,
    atomic_write_json,
    load_strict_json,
    sha256_json,
    with_self_digest,
)
from repro_fpo_ppo_v02.tests.helpers import make_bundle
from repro_fpo_ppo_v02.implementation import inspect_implementation_inventory
from repro_fpo_ppo_v02.vendor import inspect_vendor_directory, require_vendor_pythonpath_first

parser = argparse.ArgumentParser()
parser.add_argument("--attempt-manifest", type=Path, required=True)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--fpo-root", type=Path, required=True)
parser.add_argument("--vendor-dir", type=Path, required=True)
parser.add_argument("--legacy-policy-io", type=Path, required=True)
parser.add_argument("--execution-purpose", required=True)
parser.add_argument("--allow-non-gpu", action="store_true")
args = parser.parse_args()
vendor = inspect_vendor_directory(args.vendor_dir)
implementation = inspect_implementation_inventory(
    runner_path=Path(__file__), legacy_policy_io_path=args.legacy_policy_io
)
require_vendor_pythonpath_first(vendor)
attempt = load_strict_json(args.attempt_manifest)
if attempt["implementation"] != implementation:
    raise RuntimeError("implementation inventory drift")
job = attempt["job"]
anchor = AnchorManifest.from_path(job["anchor_manifest_path"])
hardware = {{
    "host": "synthetic-host",
    "platform": "synthetic-platform",
    "jax_backend": "cpu",
    "jax_devices": ["SyntheticCpuDevice(id=0)"],
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
}}
execution = with_self_digest(
    {{
        "schema": EXECUTION_EVIDENCE_SCHEMA,
        "config_digest": job["config_digest"],
        "execution_purpose": job["execution_purpose"],
        "execution_mode": attempt["execution_mode"],
        "formal_eligible": attempt["formal_eligible"],
        "allow_non_gpu": args.allow_non_gpu,
        "jax_backend": hardware["jax_backend"],
        "jax_devices": hardware["jax_devices"],
        "cuda_visible_devices": hardware["cuda_visible_devices"],
        "hardware_digest": sha256_json(hardware),
        "job_digest": job["job_digest"],
        "attempt_digest": attempt["attempt_digest"],
        "attempt_root": str(args.run_dir.resolve()),
    }},
    key="execution_evidence_digest",
)
source_proof = {{
    "fpo_commit": anchor.runtime["fpo_commit"],
    "expected_fpo_commit": anchor.runtime["fpo_commit"],
    "fpo_commit_matches_expected": True,
    "fpo_tracked_dirty": False,
    "fpo_tracked_changes": [],
    "fpo_head_tree_digest": "8" * 64,
    "fpo_worktree_tree_digest": "8" * 64,
    "fpo_execution_tree_digest": "9" * 64,
    "fpo_source_file_count": 12,
    "fpo_index_flags": [],
    "fpo_untracked_paths": [],
}}
changed_leaves = (
    []
    if anchor.operator is None
    else sorted(item.leaf for item in anchor.operator.mutations)
)
run_manifest = with_self_digest(
    {{
        "schema": "policy-learnware.v02-anchor-training-run.v0",
        "job": job,
        "job_digest": job["job_digest"],
        "attempt_digest": attempt["attempt_digest"],
        "config_digest": job["config_digest"],
        "execution_purpose": job["execution_purpose"],
        "anchor_manifest": anchor.to_dict(),
        "anchor_manifest_digest": anchor.manifest_digest,
        "environment_instance_digest": anchor.environment_instance_digest,
        "model_diff_digest": anchor.model_diff_digest,
        "binding_audit": {{
            "anchor_id": anchor.anchor_id,
            "environment_instance_digest": anchor.environment_instance_digest,
            "nominal_model_digest": anchor.expected_nominal_model_digest,
            "bound_model_digest": anchor.expected_bound_model_digest,
            "changed_leaves": changed_leaves,
            "model_diff_digest": anchor.model_diff_digest,
            "source_unchanged": True,
            "operator_digest": anchor.operator_digest,
            "manifest_digest": anchor.manifest_digest,
        }},
        "training_protocol_digest": job["training_protocol_digest"],
        "config": job["training_protocol"]["trainer_config"],
        "num_envs": 8,
        "iterations_per_env": 16,
        "transitions_per_outer": 128,
        "planned_environment_steps": 128,
        "execution_mode": execution["execution_mode"],
        "formal_eligible": execution["formal_eligible"],
        "execution_evidence_digest": execution["execution_evidence_digest"],
        "runtime": {{
            **source_proof,
            "hardware_digest": sha256_json(hardware),
            "command": sys.argv,
            "vendor": vendor,
            "implementation": implementation,
            "legacy_policy_io_path": str(args.legacy_policy_io.resolve()),
            "pythonpath_vendor_precedence_verified": True,
            "wandb_mode": os.environ.get("WANDB_MODE"),
            "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "execution_evidence": execution,
        }},
    }},
    key="run_manifest_digest",
)
atomic_write_json(args.run_dir / "run_manifest.json", run_manifest, overwrite=False)
bundle = make_bundle(
    args.run_dir / "checkpoints" / "outer_000001",
    algorithm=job["training_protocol"]["algorithm"],
    task=anchor.task,
    seed=job["seed"],
    config_digest=job["config_digest"],
    execution_purpose=job["execution_purpose"],
    execution_mode=execution["execution_mode"],
    formal_eligible=execution["formal_eligible"],
    execution_evidence_digest=execution["execution_evidence_digest"],
    attempt_root=execution["attempt_root"],
    provenance_extra={{
        "job_digest": job["job_digest"],
        "attempt_digest": attempt["attempt_digest"],
        "anchor_manifest_digest": anchor.manifest_digest,
        "environment_instance_digest": anchor.environment_instance_digest,
        "operator_digest": anchor.operator_digest,
        "actual_bound_model_digest": anchor.expected_bound_model_digest,
        "model_diff_digest": anchor.model_diff_digest,
        "runtime_digest": anchor.runtime_digest,
        **source_proof,
        "execution_mode": execution["execution_mode"],
        "formal_eligible": execution["formal_eligible"],
        "implementation": implementation,
        "execution_evidence_digest": execution["execution_evidence_digest"],
        "attempt_root": execution["attempt_root"],
    }},
)
record = with_self_digest(
    {{
        "schema": TRAINING_RECORD_SCHEMA,
        "state": "succeeded",
        "config_digest": job["config_digest"],
        "execution_purpose": job["execution_purpose"],
        "job_digest": job["job_digest"],
        "attempt_digest": attempt["attempt_digest"],
        "anchor_manifest_digest": anchor.manifest_digest,
        "environment_instance_digest": anchor.environment_instance_digest,
        "training_protocol_digest": job["training_protocol_digest"],
        "algorithm": job["training_protocol"]["algorithm"],
        "seed": job["seed"],
        "execution_mode": execution["execution_mode"],
        "formal_eligible": execution["formal_eligible"],
        "implementation": implementation,
        "execution_evidence_digest": execution["execution_evidence_digest"],
        "checkpoint_bundles": [bundle],
        "started_at": "synthetic-start",
        "finished_at": "synthetic-finish",
        "wall_seconds": 0.01,
    }},
    key="record_digest",
)
atomic_write_json(args.run_dir / "training_record.json", record, overwrite=False)
atomic_write_json(
    args.run_dir / "status.json",
    {{
        "state": "completed",
        "job_digest": job["job_digest"],
        "attempt_digest": attempt["attempt_digest"],
        "anchor_manifest_digest": anchor.manifest_digest,
        "environment_instance_digest": anchor.environment_instance_digest,
        "training_record_digest": record["record_digest"],
        "exported_outer_iterations": [1],
    }},
    overwrite=False,
)
(args.run_dir / "observed_gpu.txt").write_text(
    os.environ["CUDA_VISIBLE_DEVICES"] + "\\n", encoding="utf-8"
)
(args.run_dir / "observed_pythonpath.txt").write_text(
    os.environ["PYTHONPATH"].split(os.pathsep)[0] + "\\n", encoding="utf-8"
)
'''
    path.write_text(source, encoding="utf-8")


class ManifestQueueTests(unittest.TestCase):
    def test_plan_is_deterministic_and_embeds_reviewed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            anchor_path = Path(directory) / "anchor.json"
            anchor = make_shifted_anchor(anchor_path)
            protocol = make_protocol()

            first = build_training_plan(
                anchor_paths=[anchor_path], protocol=protocol, seeds=[0, 1],
                config_digest="0" * 64, execution_purpose="audit_smoke",
            )
            second = build_training_plan(
                anchor_paths=[anchor_path], protocol=protocol, seeds=[0, 1],
                config_digest="0" * 64, execution_purpose="audit_smoke",
            )

            self.assertEqual(first, second)
            self.assertEqual(first["expected_job_count"], 2)
            self.assertEqual([job["seed"] for job in first["jobs"]], [0, 1])
            self.assertTrue(
                all(
                    job["anchor_manifest_digest"] == anchor["manifest_digest"]
                    for job in first["jobs"]
                )
            )
            self.assertTrue(all(job["training_protocol"] == protocol for job in first["jobs"]))

    def test_queue_runs_one_job_per_gpu_and_resumes_only_valid_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository_root = Path(__file__).resolve().parents[2]
            anchor_path = root / "anchor.json"
            make_shifted_anchor(anchor_path)
            plan = build_training_plan(
                anchor_paths=[anchor_path],
                protocol=make_protocol(),
                seeds=[0, 1],
                config_digest="0" * 64,
                execution_purpose="audit_smoke",
            )
            plan_path = root / "plan.json"
            atomic_write_json(plan_path, plan, overwrite=False)
            runner = root / "fake_runner.py"
            _write_fake_runner(runner, repository_root)
            fpo_root = root / "fake_fpo"
            fpo_root.mkdir()
            vendor_dir = root / "fake_vendor"
            _make_fake_vendor(vendor_dir)
            legacy_policy_io = root / "policy_io.py"
            legacy_policy_io.write_text(
                "def export_policy_bundle(*args, **kwargs):\n    return None\n",
                encoding="utf-8",
            )
            runs_root = root / "runs"
            arguments = [
                "--plan",
                str(plan_path),
                "--execution-purpose",
                "audit_smoke",
                "--runner",
                str(runner),
                "--fpo-root",
                str(fpo_root),
                "--runs-root",
                str(runs_root),
                "--gpus",
                "0,1",
                "--python",
                sys.executable,
                "--vendor-dir",
                str(vendor_dir),
                "--legacy-policy-io",
                str(legacy_policy_io),
                "--max-attempts",
                "1",
                "--poll-seconds",
                "0.01",
                "--terminate-grace-seconds",
                "0.1",
                "--allow-non-gpu",
            ]

            self.assertEqual(queue_main(arguments), 0)
            attempts = sorted(runs_root.glob("jobs/*/attempt_001"))
            self.assertEqual(len(attempts), 2)
            self.assertEqual(
                sorted((path / "observed_gpu.txt").read_text().strip() for path in attempts),
                ["0", "1"],
            )
            self.assertEqual(
                {Path((path / "observed_pythonpath.txt").read_text().strip()) for path in attempts},
                {vendor_dir.resolve()},
            )
            self.assertFalse(list(runs_root.glob("jobs/*/.attempt_*.pending-*")))
            queue_status = load_strict_json(runs_root / "queue_status.json")
            self.assertEqual(queue_status["state"], "completed")
            self.assertEqual(queue_status["vendor"], inspect_vendor_directory(vendor_dir))
            expected_implementation = inspect_implementation_inventory(
                runner_path=runner,
                legacy_policy_io_path=legacy_policy_io,
            )
            self.assertEqual(queue_status["implementation"], expected_implementation)
            for attempt_dir in attempts:
                attempt = load_strict_json(attempt_dir / "attempt_manifest.json")
                record = load_strict_json(attempt_dir / "training_record.json")
                checkpoint = record["checkpoint_bundles"][0]
                provenance = load_strict_json(
                    Path(checkpoint["path"]) / "provenance.json"
                )
                self.assertEqual(attempt["execution_mode"], "audit_smoke")
                self.assertEqual(attempt["implementation"], expected_implementation)
                self.assertFalse(attempt["formal_eligible"])
                self.assertFalse(record["formal_eligible"])
                self.assertFalse(checkpoint["formal_eligible"])
                self.assertFalse(provenance["formal_eligible"])
                self.assertEqual(
                    record["execution_evidence_digest"],
                    checkpoint["execution_evidence_digest"],
                )
                self.assertEqual(
                    record["execution_evidence_digest"],
                    provenance["execution_evidence_digest"],
                )
                run_manifest = load_strict_json(attempt_dir / "run_manifest.json")
                queue_result = load_strict_json(attempt_dir / "queue_result.json")
                self.assertEqual(run_manifest["runtime"]["vendor"], queue_status["vendor"])
                self.assertEqual(queue_result["vendor"], queue_status["vendor"])
                self.assertEqual(
                    run_manifest["runtime"]["implementation"], expected_implementation
                )
                self.assertEqual(queue_result["implementation"], expected_implementation)
                self.assertEqual(record["implementation"], expected_implementation)

            # A second invocation verifies every record/result/bundle digest and skips
            # the immutable successful attempts rather than creating attempt_002.
            self.assertEqual(queue_main(arguments), 0)
            self.assertEqual(len(list(runs_root.glob("jobs/*/attempt_*"))), 2)

            # Re-hashing a forged clean-tree claim cannot turn an old attempt
            # into resumable success.
            run_manifest_path = attempts[0] / "run_manifest.json"
            original_run_manifest = run_manifest_path.read_bytes()
            forged_run = load_strict_json(run_manifest_path)
            forged_runtime = dict(forged_run["runtime"])
            forged_runtime["fpo_tracked_dirty"] = True
            forged_runtime["fpo_tracked_changes"] = ["M tracked.py"]
            forged_run["runtime"] = forged_runtime
            atomic_write_json(
                run_manifest_path,
                with_self_digest(
                    {
                        key: value
                        for key, value in forged_run.items()
                        if key != "run_manifest_digest"
                    },
                    key="run_manifest_digest",
                ),
            )
            self.assertEqual(queue_main(arguments), 1)
            self.assertEqual(len(list(runs_root.glob("jobs/*/attempt_*"))), 2)
            run_manifest_path.write_bytes(original_run_manifest)

            exporter_bytes = legacy_policy_io.read_bytes()
            legacy_policy_io.write_text("def export_policy_bundle():\n    return 'drift'\n")
            with self.assertRaisesRegex(
                ContractError, "implementation provenance differs"
            ):
                queue_main(arguments)
            legacy_policy_io.write_bytes(exporter_bytes)

            # A non-RECORD dependency byte is still part of the full tree digest,
            # so a changed vendor cannot reuse the previous successful evidence.
            dependency = vendor_dir / "dependency.py"
            dependency.write_text("PIN = 2\n", encoding="utf-8")
            self.assertEqual(queue_main(arguments), 1)
            self.assertEqual(len(list(runs_root.glob("jobs/*/attempt_*"))), 2)
            dependency.write_text("PIN = 1\n", encoding="utf-8")

            formal_arguments = list(arguments)
            formal_arguments.remove("--allow-non-gpu")
            formal_arguments[formal_arguments.index("audit_smoke")] = "v02_freeze_ready"
            with self.assertRaisesRegex(
                ContractError, "execution purpose differs from the immutable plan"
            ):
                queue_main(formal_arguments)

            # Tampered immutable success metadata is not accepted as resumable
            # success, even though the child process originally returned zero.
            result_path = attempts[0] / "queue_result.json"
            tampered_result = load_strict_json(result_path)
            tampered_result["result_digest"] = "0" * 64
            atomic_write_json(result_path, tampered_result)
            self.assertEqual(queue_main(arguments), 1)
            self.assertEqual(len(list(runs_root.glob("jobs/*/attempt_*"))), 2)


if __name__ == "__main__":
    unittest.main()
