from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from repro_fpo_ppo_v02.anchor_binding import (
    AnchorBindingError,
    AnchorManifest,
    derive_environment_instance_digest,
    derive_source_anchor_id,
    load_and_bind_anchor,
    snapshot_model,
    verify_bound_environment,
)
from repro_fpo_ppo_v02.provenance import ContractError, with_self_digest
from repro_fpo_ppo_v02.tests.helpers import FakeEnv, FakeRegistry, fake_model, make_shifted_anchor


class AnchorBindingTests(unittest.TestCase):
    def test_shift_is_applied_before_use_and_nominal_source_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = make_shifted_anchor(Path(directory) / "anchor.json")
            manifest = AnchorManifest.from_dict(raw)
            source = fake_model()
            before = snapshot_model(source)
            bound = load_and_bind_anchor(
                registry=FakeRegistry(source, raw["registry_config"]),
                manifest=manifest,
            )

            self.assertIsNot(bound.env._mjx_model, source)
            self.assertEqual(bound.verify().digest, raw["expected_bound_model_digest"])
            self.assertEqual(bound.audit.changed_leaves, ("_mjx_model.dof_damping",))
            self.assertEqual(snapshot_model(source).digest, before.digest)

    def test_poisoned_shifted_directory_with_nominal_env_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = AnchorManifest.from_dict(
                make_shifted_anchor(Path(directory) / "anchor.json")
            )
            with self.assertRaisesRegex(AnchorBindingError, "poisoned shifted run"):
                verify_bound_environment(FakeEnv(fake_model()), manifest)

    def test_manifest_unknown_key_and_digest_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = make_shifted_anchor(Path(directory) / "anchor.json")
            unknown = deepcopy(raw)
            unknown["unreviewed_default"] = 1
            with self.assertRaisesRegex(ContractError, "unknown"):
                AnchorManifest.from_dict(unknown)

            drifted = deepcopy(raw)
            drifted["factor"] = 3.0
            with self.assertRaises(ContractError):
                AnchorManifest.from_dict(drifted)

            model_diff_poison = {
                key: value
                for key, value in deepcopy(raw).items()
                if key != "manifest_digest"
            }
            model_diff_poison["model_diff_digest"] = "0" * 64
            model_diff_poison["environment_instance_digest"] = (
                derive_environment_instance_digest(model_diff_poison)
            )
            model_diff_poison["anchor_id"] = derive_source_anchor_id(
                environment_instance_digest=model_diff_poison[
                    "environment_instance_digest"
                ],
                axis_binding_digest=model_diff_poison["axis_binding_digest"],
            )
            model_diff_poison = with_self_digest(
                model_diff_poison, key="manifest_digest"
            )
            with self.assertRaisesRegex(ContractError, "model_diff_digest mismatch"):
                AnchorManifest.from_dict(model_diff_poison)

    def test_wrong_anchor_or_axis_binding_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = make_shifted_anchor(Path(directory) / "anchor.json")

            wrong_anchor = deepcopy(raw)
            wrong_anchor["anchor_id"] = "0" * 64
            with self.assertRaisesRegex(ContractError, "canonical source-anchor"):
                AnchorManifest.from_dict(wrong_anchor)

            wrong_binding = {
                key: value for key, value in deepcopy(raw).items() if key != "manifest_digest"
            }
            wrong_binding["axis_binding_digest"] = "d" * 64
            wrong_binding["environment_instance_digest"] = derive_environment_instance_digest(
                wrong_binding
            )
            # Keep the original anchor id: the binding tamper must be caught by
            # the canonical source-anchor relation, not merely manifest SHA.
            wrong_binding = with_self_digest(wrong_binding, key="manifest_digest")
            with self.assertRaisesRegex(ContractError, "canonical source-anchor"):
                AnchorManifest.from_dict(wrong_binding)


if __name__ == "__main__":
    unittest.main()
