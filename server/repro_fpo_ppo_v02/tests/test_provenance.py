from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from repro_fpo_ppo_v02.provenance import (
    ContractError,
    IMPLEMENTATION_FILE_LABELS,
    IMPLEMENTATION_PROVENANCE_SCHEMA,
    NumericalIntegrityError,
    TRAINING_RECORD_SCHEMA,
    atomic_write_json,
    assert_finite_mapping,
    load_strict_json,
    sha256_file,
    sha256_json,
    validate_success_record,
    validate_policy_bundle,
    with_self_digest,
)
from repro_fpo_ppo_v02.tests.helpers import make_bundle


class ProvenanceTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
                load_strict_json(duplicate)

            nonfinite = root / "nan.json"
            nonfinite.write_text('{"a":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "non-finite JSON constant"):
                load_strict_json(nonfinite)

    def test_bundle_validation_checks_all_numerical_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            finite = root / "finite"
            expected = make_bundle(finite)
            observed = validate_policy_bundle(finite, require_evaluation=False)
            self.assertEqual(
                observed["bundle_manifest_digest"], expected["bundle_manifest_digest"]
            )

            poisoned = root / "poisoned"
            make_bundle(poisoned, finite=False)
            with self.assertRaisesRegex(NumericalIntegrityError, "non-finite"):
                validate_policy_bundle(poisoned, require_evaluation=False)

    def test_metrics_are_fail_closed(self) -> None:
        for value in (None, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(NumericalIntegrityError):
                    assert_finite_mapping({"metric": value})

    def test_success_record_rejects_forged_reload_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = make_bundle(root / "bundle")
            implementation_files = {
                label: {"bytes": 1, "sha256": sha256_json({"label": label})}
                for label in sorted(IMPLEMENTATION_FILE_LABELS)
            }
            implementation_material = {
                "schema": IMPLEMENTATION_PROVENANCE_SCHEMA,
                "files": implementation_files,
            }
            implementation = {
                **implementation_material,
                "implementation_digest": sha256_json(implementation_material),
            }
            record = with_self_digest(
                {
                    "schema": TRAINING_RECORD_SCHEMA,
                    "state": "succeeded",
                    "config_digest": "0" * 64,
                    "execution_purpose": "v02_freeze_ready",
                    "job_digest": "a" * 64,
                    "attempt_digest": "b" * 64,
                    "anchor_manifest_digest": "c" * 64,
                    "environment_instance_digest": "d" * 64,
                    "training_protocol_digest": "e" * 64,
                    "algorithm": "ppo",
                    "seed": 0,
                    "execution_mode": "formal_gpu",
                    "formal_eligible": True,
                    "implementation": implementation,
                    "execution_evidence_digest": "f" * 64,
                    "checkpoint_bundles": [checkpoint],
                    "planned_outer_iterations": 1,
                    "completed_outer_iterations": 1,
                    "promoted_outer_iteration": 1,
                    "planned_environment_steps": 128,
                    "completed_environment_steps": 128,
                    "promoted_environment_steps": 128,
                    "terminal_failure": None,
                    "started_at": "synthetic-start",
                    "finished_at": "synthetic-finish",
                    "wall_seconds": 0.1,
                },
                key="record_digest",
            )
            path = root / "training_record.json"
            atomic_write_json(path, record, overwrite=False)
            validate_success_record(path, expected_job_digest="a" * 64)

            forged = load_strict_json(path)
            forged["checkpoint_bundles"][0]["golden_parity"]["passed"] = False
            forged = with_self_digest(
                {key: value for key, value in forged.items() if key != "record_digest"},
                key="record_digest",
            )
            atomic_write_json(path, forged)
            with self.assertRaisesRegex(ContractError, "golden_parity did not pass"):
                validate_success_record(path, expected_job_digest="a" * 64)

    def test_numerical_recovery_records_actual_prefix_budget_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = make_bundle(root / "bundle")
            implementation_files = {
                label: {"bytes": 1, "sha256": sha256_json({"label": label})}
                for label in sorted(IMPLEMENTATION_FILE_LABELS)
            }
            implementation_material = {
                "schema": IMPLEMENTATION_PROVENANCE_SCHEMA,
                "files": implementation_files,
            }
            implementation = {
                **implementation_material,
                "implementation_digest": sha256_json(implementation_material),
            }
            trace = root / "recovery_traceback.txt"
            trace.write_text("synthetic numerical failure\n", encoding="utf-8")
            record = with_self_digest(
                {
                    "schema": TRAINING_RECORD_SCHEMA,
                    "state": "recovered",
                    "config_digest": "0" * 64,
                    "execution_purpose": "v02_freeze_ready",
                    "job_digest": "a" * 64,
                    "attempt_digest": "b" * 64,
                    "anchor_manifest_digest": "c" * 64,
                    "environment_instance_digest": "d" * 64,
                    "training_protocol_digest": "e" * 64,
                    "algorithm": "fpo",
                    "seed": 0,
                    "execution_mode": "formal_gpu",
                    "formal_eligible": True,
                    "implementation": implementation,
                    "execution_evidence_digest": "f" * 64,
                    "checkpoint_bundles": [checkpoint],
                    "planned_outer_iterations": 3,
                    "completed_outer_iterations": 2,
                    "promoted_outer_iteration": 1,
                    "planned_environment_steps": 384,
                    "completed_environment_steps": 256,
                    "promoted_environment_steps": 128,
                    "terminal_failure": {
                        "type": "NumericalIntegrityError",
                        "message": "training_step contains non-finite values",
                        "traceback_file": trace.name,
                        "traceback_sha256": sha256_file(trace),
                    },
                    "started_at": "synthetic-start",
                    "finished_at": "synthetic-finish",
                    "wall_seconds": 0.1,
                },
                key="record_digest",
            )
            path = root / "training_record.json"
            atomic_write_json(path, record, overwrite=False)
            observed = validate_success_record(path, expected_job_digest="a" * 64)
            self.assertEqual(observed["state"], "recovered")
            self.assertEqual(observed["promoted_environment_steps"], 128)

            trace.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "trace digest mismatch"):
                validate_success_record(path, expected_job_digest="a" * 64)


if __name__ == "__main__":
    unittest.main()
