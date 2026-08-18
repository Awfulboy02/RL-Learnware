from __future__ import annotations

import copy
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from policy_learnware_v0.config import ConfigError, ProtocolDraft, load_protocol_draft


class ProtocolConfigTest(unittest.TestCase):
    def test_bundled_configs_load_and_pin_open_definitions(self) -> None:
        main = load_protocol_draft(PROJECT / "configs" / "dmc6_outer006_v0.yaml")
        smoke = load_protocol_draft(PROJECT / "configs" / "smoke.yaml")

        self.assertEqual(len(main.environment.tasks), 6)
        self.assertEqual(main.effective_task_balanced_batch_size, 1020)
        self.assertEqual(main.episodes.separability_calibration_per_task, 16)
        self.assertEqual(main.episodes.target_query_banks, 10)
        self.assertEqual(smoke.episodes.target_query_banks, 2)
        self.assertEqual(main.normalization.fit_split, "encoder_train")
        self.assertEqual(main.encoder.checkpoint_metric, "validation_supcon_loss")
        self.assertEqual(main.encoder.checkpoint_tie_break, "earliest_step")
        self.assertEqual(main.reducer.objective, "weighted_kme_ridge")
        self.assertEqual(main.reducer.reconstruction_tolerance_metric, "rkhs_norm")
        self.assertEqual(len(main.draft_hash), 64)
        self.assertEqual(main.draft_hash, load_protocol_draft(
            PROJECT / "configs" / "dmc6_outer006_v0.yaml"
        ).draft_hash)

    def test_protocol_is_immutable_and_unknown_keys_fail_closed(self) -> None:
        protocol = load_protocol_draft(PROJECT / "configs" / "smoke.yaml")
        with self.assertRaises(FrozenInstanceError):
            protocol.project_seed = 7  # type: ignore[misc]

        payload = copy.deepcopy(protocol.to_dict())
        payload["unexpected"] = True
        with self.assertRaises(ConfigError):
            ProtocolDraft.from_dict(payload)

    def test_source_taskspec_is_distinct_from_calibration(self) -> None:
        protocol = load_protocol_draft(PROJECT / "configs" / "smoke.yaml")
        self.assertNotEqual(
            "separability_calibration", "source_taskspec"
        )
        self.assertGreater(protocol.episodes.source_taskspec_per_task, 0)
        self.assertGreater(protocol.episodes.separability_calibration_per_task, 0)

    def test_fixed_policy_inventory_and_runtime_guards_cannot_be_relaxed(self) -> None:
        protocol = load_protocol_draft(PROJECT / "configs" / "smoke.yaml")
        mutations = (
            ("pool", "candidates_per_task", 9),
            ("policy", "golden_sample_count", 7),
            ("policy", "golden_parity_on_load", False),
            ("policy", "require_runtime_commit_match", False),
            ("policy", "verify_module_origin", False),
        )
        for section, field, value in mutations:
            with self.subTest(section=section, field=field):
                payload = copy.deepcopy(protocol.to_dict())
                payload[section][field] = value
                with self.assertRaises(ConfigError):
                    ProtocolDraft.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
