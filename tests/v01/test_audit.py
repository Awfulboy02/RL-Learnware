from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from policy_learnware_v0.io import atomic_write_npz
from policy_learnware_v0.v01.audit import (
    assert_no_oracle_dependencies,
    assert_measurement_schema_allowlist,
    scan_measurement_tree,
)


class AuditV01Test(unittest.TestCase):
    def test_recursive_forbidden_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ok.json").write_text(
                json.dumps({"nested": [{"candidate_id": "bad"}]}), encoding="utf-8"
            )
            atomic_write_npz(root / "cache.npz", {"points": np.zeros((2, 2))})
            violations = scan_measurement_tree(root, ["candidate_id", "return"])
            self.assertEqual(len(violations), 1)
            self.assertEqual(violations[0].forbidden_field, "candidate_id")

    def test_forbidden_string_payloads_are_scanned_without_substring_false_positives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload.json").write_text(
                json.dumps(
                    {
                        "schema": "policy-learnware.v01-taskspec-matrix.v0",
                        "opaque": "bundle-path/candidate_id",
                    }
                ),
                encoding="utf-8",
            )
            (root / "payload.csv").write_text(
                "opaque\nreturn\n",
                encoding="utf-8",
            )
            atomic_write_npz(
                root / "strings.npz",
                {"opaque": np.asarray(["source_task"], dtype="U16")},
            )
            violations = scan_measurement_tree(
                root, ["task", "bundle", "candidate_id", "return", "source_task"]
            )
            observed = {item.forbidden_field for item in violations}
            self.assertEqual(
                observed,
                {"bundle", "candidate_id", "return", "source_task"},
            )

    def test_taskspec_roots_reject_private_inputs(self) -> None:
        assert_no_oracle_dependencies(["/tmp/run/measurement", "/tmp/base"])
        with self.assertRaises(PermissionError):
            assert_no_oracle_dependencies(["/tmp/run/oracle_private"])

    def test_measurement_schema_allowlist_rejects_unknown_artifacts_and_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_ref.json").write_text(
                json.dumps(
                    {
                        "schema": "s",
                        "measurement_protocol_id": "p",
                        "measurement_run_id": "r",
                        "measurement_protocol_sha256": "d",
                        "base_protocol_ref": {},
                        "measurement_contract_digest": "d",
                        "pair_plan_digest": "d",
                        "schema_view_digests": {},
                        "formal": False,
                        "git": {},
                        "runtime_versions": {},
                        "measurement_component_digests": {},
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(assert_measurement_schema_allowlist(root)["passed"])
            (root / "unknown.json").write_text('{"passed":true}', encoding="utf-8")
            result = assert_measurement_schema_allowlist(root)
            self.assertFalse(result["passed"])
            self.assertEqual(
                result["violations"][0]["reason"], "unregistered_json_artifact"
            )


if __name__ == "__main__":
    unittest.main()
