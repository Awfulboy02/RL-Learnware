from __future__ import annotations

import copy
import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from policy_learnware_v0.config import ConfigError, ProtocolDraft, load_protocol_draft
from policy_learnware_v0.artifacts import ArtifactLayout
from policy_learnware_v0.cli import (
    CommandFailure,
    _require_gate_passed,
    _handle_diagnose_unreduced,
    _handle_reduce_task_specs,
    _validate_ranking_artifact,
    _validate_unreduced_diagnostics_artifact,
)
from policy_learnware_v0.hashing import sha256_file
from policy_learnware_v0.gates import (
    deployment_gate,
    deterministic_ranking,
    nonoverlapping_half_ranges,
    pairwise_order_agreement,
    ranking_gate,
    retrieval_gate,
    unreduced_gate,
    validate_gate_record,
)


class GateConfigTest(unittest.TestCase):
    def test_bundled_gate_thresholds_are_explicit_and_frozen(self) -> None:
        config = load_protocol_draft(PROJECT / "configs" / "dmc6_outer006_v0.yaml")
        self.assertEqual(config.gates.unreduced.minimum_between_within_ratio, 1.25)
        self.assertEqual(config.gates.unreduced.minimum_absolute_margin, 1.0e-6)
        self.assertEqual(config.gates.unreduced.minimum_split_retrieval_accuracy, 1.0)
        self.assertEqual(
            config.gates.reduced_unreduced_ranking.minimum_top1_agreement, 1.0
        )
        self.assertEqual(config.gates.retrieval.minimum_max_prefix_accuracy, 0.95)
        self.assertEqual(
            config.gates.deployment.minimum_correct_retrieval_deployability_rate,
            1.0,
        )

    def test_gate_config_rejects_unknown_or_out_of_range_values(self) -> None:
        config = load_protocol_draft(PROJECT / "configs" / "smoke.yaml")
        unknown = copy.deepcopy(config.to_dict())
        unknown["gates"]["retrieval"]["overall_accuracy"] = 0.5
        with self.assertRaises(ConfigError):
            ProtocolDraft.from_dict(unknown)

        invalid_probability = copy.deepcopy(config.to_dict())
        invalid_probability["gates"]["retrieval"][
            "minimum_max_prefix_accuracy"
        ] = 1.01
        with self.assertRaises(ConfigError):
            ProtocolDraft.from_dict(invalid_probability)

        invalid_ratio = copy.deepcopy(config.to_dict())
        invalid_ratio["gates"]["unreduced"][
            "minimum_between_within_ratio"
        ] = 0.99
        with self.assertRaises(ConfigError):
            ProtocolDraft.from_dict(invalid_ratio)


class GateDecisionTest(unittest.TestCase):
    def test_unreduced_gate_is_auditable_and_fails_each_threshold(self) -> None:
        passed = unreduced_gate(
            minimum_between_mmd=0.30,
            maximum_within_mmd=0.20,
            split_retrieval_accuracy=1.0,
            minimum_between_within_ratio=1.25,
            minimum_absolute_margin=1.0e-6,
            minimum_split_retrieval_accuracy=1.0,
        )
        self.assertTrue(passed.passed)
        self.assertTrue(
            validate_gate_record(
                passed.to_dict(), expected_name="unreduced_separability"
            ).passed
        )

        weak_ratio = unreduced_gate(
            minimum_between_mmd=0.21,
            maximum_within_mmd=0.20,
            split_retrieval_accuracy=1.0,
            minimum_between_within_ratio=1.25,
            minimum_absolute_margin=1.0e-6,
            minimum_split_retrieval_accuracy=1.0,
        )
        self.assertFalse(weak_ratio.passed)

        wrong_retrieval = unreduced_gate(
            minimum_between_mmd=0.30,
            maximum_within_mmd=0.20,
            split_retrieval_accuracy=0.99,
            minimum_between_within_ratio=1.25,
            minimum_absolute_margin=1.0e-6,
            minimum_split_retrieval_accuracy=1.0,
        )
        self.assertFalse(wrong_retrieval.passed)

        tampered = passed.to_dict()
        tampered["passed"] = False
        with self.assertRaises(ValueError):
            validate_gate_record(tampered, expected_name="unreduced_separability")

    def test_ranking_retrieval_and_conditional_deployment_gates(self) -> None:
        self.assertTrue(
            ranking_gate(top1_agreement=1.0, minimum_top1_agreement=1.0).passed
        )
        self.assertFalse(
            retrieval_gate(
                max_prefix_accuracy=0.949,
                minimum_max_prefix_accuracy=0.95,
            ).passed
        )
        self.assertTrue(
            deployment_gate(
                correct_retrieval_count=12,
                correct_retrieval_deployability_rate=1.0,
                minimum_correct_retrieval_deployability_rate=1.0,
            ).passed
        )
        self.assertFalse(
            deployment_gate(
                correct_retrieval_count=0,
                correct_retrieval_deployability_rate=None,
                minimum_correct_retrieval_deployability_rate=1.0,
            ).passed
        )

    def test_rank_diagnostics_and_strict_episode_split(self) -> None:
        left = deterministic_ranking({"a": 0.1, "b": 0.2, "c": 0.3})
        right = deterministic_ranking({"a": 0.1, "b": 0.3, "c": 0.2})
        self.assertEqual(left[0], right[0])
        self.assertAlmostEqual(pairwise_order_agreement(left, right), 2.0 / 3.0)

        source_range, query_range = nonoverlapping_half_ranges(5)
        source_ids = set(range(*source_range))
        query_ids = set(range(*query_range))
        self.assertEqual(source_range, (0, 2))
        self.assertEqual(query_range, (2, 5))
        self.assertFalse(source_ids.intersection(query_ids))
        self.assertEqual(source_ids.union(query_ids), set(range(5)))
        with self.assertRaises(ValueError):
            nonoverlapping_half_ranges(1)

    def test_failed_gate_artifact_is_retained_before_downstream_block(self) -> None:
        decision = retrieval_gate(
            max_prefix_accuracy=0.90,
            minimum_max_prefix_accuracy=0.95,
        )
        payload = {
            "gate": decision.to_dict(),
            "gate_passed": decision.passed,
        }
        with tempfile.TemporaryDirectory() as directory:
            layout = ArtifactLayout(Path(directory), "pool")
            layout.publish_json(layout.retrieval_metrics, payload)
            with self.assertRaises(CommandFailure):
                _require_gate_passed(
                    payload,
                    expected_name="exact_recurrent_retrieval",
                    artifact=layout.retrieval_metrics,
                )
            self.assertTrue(layout.retrieval_metrics.is_file())

    def test_unreduced_validator_recomputes_metrics_and_config_thresholds(self) -> None:
        config = load_protocol_draft(PROJECT / "configs" / "smoke.yaml")
        tasks = config.environment.tasks
        with tempfile.TemporaryDirectory() as directory:
            layout = ArtifactLayout(Path(directory), config.pool.pool_id)
            dataset_digests: dict[str, str] = {}
            for index, task in enumerate(tasks):
                dataset_digest = f"{index + 1:064x}"
                dataset_digests[task] = dataset_digest
                layout.publish_json(
                    layout.dataset_manifest("separability_calibration", task),
                    {"dataset_sha256": dataset_digest},
                )
            matrix = ["," + ",".join(tasks)]
            for task in tasks:
                matrix.append(
                    task
                    + ","
                    + ",".join("0" if candidate == task else "0.3" for candidate in tasks)
                )
            matrix_digest = layout.publish_text(
                layout.mmd_matrix, "\n".join(matrix) + "\n"
            )
            within = {task: 0.1 for task in tasks}
            distances = {
                task: {
                    candidate: 0.1 if candidate == task else 0.3
                    for candidate in tasks
                }
                for task in tasks
            }
            decision = unreduced_gate(
                minimum_between_mmd=0.3,
                maximum_within_mmd=0.1,
                split_retrieval_accuracy=1.0,
                minimum_between_within_ratio=1.25,
                minimum_absolute_margin=1.0e-6,
                minimum_split_retrieval_accuracy=1.0,
            )
            payload = {
                "schema": "policy-learnware.unreduced-diagnostics.v0",
                "complete": True,
                "protocol_draft_hash": config.draft_hash,
                "protocol_id": "protocol-test",
                "mmd_matrix_sha256": matrix_digest,
                "minimum_between_mmd": 0.3,
                "maximum_within_mmd": 0.1,
                "within_task_mmd": within,
                "split_retrieval_accuracy": 1.0,
                "split_retrieval": {task: task for task in tasks},
                "split_retrieval_distances": distances,
                "split_protocol": {
                    "source_role": "first_half",
                    "query_role": "second_half",
                    "candidate_sources_use_query_episodes": False,
                    "episode_ranges": {
                        task: {
                            "dataset_sha256": dataset_digests[task],
                            "source_episode_range_half_open": [0, 1],
                            "query_episode_range_half_open": [1, 2],
                            "overlap_episode_count": 0,
                        }
                        for task in tasks
                    },
                },
                "gate": decision.to_dict(),
                "gate_passed": True,
            }
            _validate_unreduced_diagnostics_artifact(
                payload,
                config,
                layout,
                expected_protocol_id="protocol-test",
            )

            roundoff = copy.deepcopy(payload)
            roundoff["split_retrieval_distances"][tasks[0]][tasks[0]] += 5.0e-14
            _validate_unreduced_diagnostics_artifact(
                roundoff,
                config,
                layout,
                expected_protocol_id="protocol-test",
            )

            inconsistent_self_distance = copy.deepcopy(payload)
            inconsistent_self_distance["split_retrieval_distances"][tasks[0]][
                tasks[0]
            ] += 1.0e-6
            with self.assertRaises(CommandFailure):
                _validate_unreduced_diagnostics_artifact(
                    inconsistent_self_distance,
                    config,
                    layout,
                    expected_protocol_id="protocol-test",
                )

            tampered = copy.deepcopy(payload)
            tampered["minimum_between_mmd"] = 9.0
            forged = unreduced_gate(
                minimum_between_mmd=9.0,
                maximum_within_mmd=0.1,
                split_retrieval_accuracy=1.0,
                minimum_between_within_ratio=1.25,
                minimum_absolute_margin=1.0e-6,
                minimum_split_retrieval_accuracy=1.0,
            )
            tampered["gate"] = forged.to_dict()
            with self.assertRaises(CommandFailure):
                _validate_unreduced_diagnostics_artifact(
                    tampered,
                    config,
                    layout,
                    expected_protocol_id="protocol-test",
                )

    def test_ranking_validator_recomputes_queries_and_gate(self) -> None:
        config = load_protocol_draft(PROJECT / "configs" / "smoke.yaml")
        tasks = config.environment.tasks
        with tempfile.TemporaryDirectory() as directory:
            layout = ArtifactLayout(Path(directory), config.pool.pool_id)
            source_hashes: dict[str, str] = {}
            query_hashes: dict[str, str] = {}
            task_hashes: dict[str, str] = {}
            queries: dict[str, dict] = {}
            for index, task in enumerate(tasks):
                dataset_digest = f"{index + 1:064x}"
                source_manifest = layout.dataset_manifest("source_taskspec", task)
                query_manifest = layout.dataset_manifest(
                    "separability_calibration", task
                )
                layout.publish_json(source_manifest, {"dataset_sha256": dataset_digest})
                layout.publish_json(query_manifest, {"dataset_sha256": dataset_digest})
                layout.publish_text(layout.task_rkme(task), f"rkme:{task}\n")
                source_hashes[task] = sha256_file(source_manifest)
                query_hashes[task] = sha256_file(query_manifest)
                task_hashes[task] = sha256_file(layout.task_rkme(task))
                distances = {
                    candidate: 0.1 + tasks.index(candidate) * 0.01
                    for candidate in tasks
                }
                ranking = deterministic_ranking(distances)
                queries[task] = {
                    "query_dataset_sha256": dataset_digest,
                    "unreduced_distances": distances,
                    "reduced_distances": distances,
                    "unreduced_ranking": list(ranking),
                    "reduced_ranking": list(ranking),
                    "unreduced_top1": ranking[0],
                    "reduced_top1": ranking[0],
                    "top1_agrees": True,
                    "pairwise_order_agreement": 1.0,
                }
            decision = ranking_gate(top1_agreement=1.0, minimum_top1_agreement=1.0)
            payload = {
                "schema": "policy-learnware.reduced-unreduced-ranking.v0",
                "complete": True,
                "protocol_draft_hash": config.draft_hash,
                "protocol_id": "protocol-test",
                "source_split": "source_taskspec",
                "query_split": "separability_calibration",
                "source_query_splits_are_distinct": True,
                "source_dataset_manifest_sha256": source_hashes,
                "query_dataset_manifest_sha256": query_hashes,
                "task_rkme_sha256": task_hashes,
                "top1_agreement": 1.0,
                "mean_pairwise_order_agreement": 1.0,
                "queries": queries,
                "gate": decision.to_dict(),
                "gate_passed": True,
            }
            _validate_ranking_artifact(
                payload,
                config,
                layout,
                expected_protocol_id="protocol-test",
            )

            tampered = copy.deepcopy(payload)
            tampered["gate"]["checks"][0]["observed"] = 0.5
            tampered["gate"]["checks"][0]["minimum"] = 0.5
            with self.assertRaises(CommandFailure):
                _validate_ranking_artifact(
                    tampered,
                    config,
                    layout,
                    expected_protocol_id="protocol-test",
                )

            tampered_query = copy.deepcopy(payload)
            query = tampered_query["queries"][tasks[0]]
            query["reduced_distances"][tasks[-1]] = 0.0
            with self.assertRaises(CommandFailure):
                _validate_ranking_artifact(
                    tampered_query,
                    config,
                    layout,
                    expected_protocol_id="protocol-test",
                )

    def test_gate_resume_paths_reload_runtime_bound_protocol(self) -> None:
        config = load_protocol_draft(PROJECT / "configs" / "smoke.yaml")
        args = argparse.Namespace(resume=True)
        with tempfile.TemporaryDirectory() as directory:
            layout = ArtifactLayout(Path(directory), config.pool.pool_id)
            matrix_digest = layout.publish_text(layout.mmd_matrix, "placeholder\n")
            layout.publish_json(
                layout.unreduced_diagnostics,
                {
                    "complete": True,
                    "protocol_draft_hash": config.draft_hash,
                    "mmd_matrix_sha256": matrix_digest,
                },
            )
            protocol = SimpleNamespace(protocol_id="protocol-test")
            with patch(
                "policy_learnware_v0.cli._load_frozen_protocol",
                return_value=protocol,
            ) as loader, patch(
                "policy_learnware_v0.cli._validate_unreduced_diagnostics_artifact"
            ) as validator:
                result = _handle_diagnose_unreduced(args, config, layout)
            self.assertTrue(result["resumed"])
            loader.assert_called_once_with(layout, config)
            validator.assert_called_once_with(
                ANY,
                config,
                layout,
                expected_protocol_id="protocol-test",
            )

        with tempfile.TemporaryDirectory() as directory:
            layout = ArtifactLayout(Path(directory), config.pool.pool_id)
            for task in config.environment.tasks:
                layout.publish_json(layout.empirical_summary(task), {"task": task})
                rkme_digest = layout.publish_text(layout.task_rkme(task), task)
                layout.publish_json(
                    layout.task_rkme_manifest(task),
                    {
                        "complete": True,
                        "protocol_draft_hash": config.draft_hash,
                        "protocol_id": "protocol-test",
                        "rkme_sha256": rkme_digest,
                    },
                )
            layout.publish_json(layout.reduced_unreduced_ranking, {})
            protocol = SimpleNamespace(protocol_id="protocol-test")
            with patch(
                "policy_learnware_v0.cli._load_frozen_protocol",
                return_value=protocol,
            ) as loader, patch(
                "policy_learnware_v0.cli._validate_ranking_artifact"
            ) as validator:
                result = _handle_reduce_task_specs(args, config, layout)
            self.assertTrue(result["resumed"])
            loader.assert_called_once_with(layout, config)
            validator.assert_called_once_with(
                {},
                config,
                layout,
                expected_protocol_id="protocol-test",
            )


if __name__ == "__main__":
    unittest.main()
