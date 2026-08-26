from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"

_IMPORT_GUARD = r"""
import importlib.abc
import sys

BLOCKED_MODULES = (
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "torch",
    "transformers",
    "brax",
    "mujoco",
    "mujoco_playground",
    "policy_learnware_v0.v03.fpo_source_backend",
    "policy_learnware_v0.v03.checkpoints",
    "policy_learnware_v0.v03.encoder_registry",
    "policy_learnware_v0.v03.anonymous_market",
    "policy_learnware_v0.v03.source_market",
)


class _BlockedOptionalDependencyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == blocked or fullname.startswith(blocked + ".")
            for blocked in BLOCKED_MODULES
        ):
            raise ModuleNotFoundError(
                "optional dependency blocked by v0.3 base-path audit: " + fullname
            )
        return None


sys.meta_path.insert(0, _BlockedOptionalDependencyFinder())


def assert_no_blocked_module_loaded():
    leaked = sorted(
        name
        for name in sys.modules
        if any(
            name == blocked or name.startswith(blocked + ".")
            for blocked in BLOCKED_MODULES
        )
    )
    assert not leaked, leaked
"""


def _run_in_clean_interpreter(source: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(_SOURCE_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_IMPORT_GUARD + source)],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, (
        f"isolated interpreter failed with code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return completed


def test_disabled_extension_raw_acceptance_has_no_optional_dependency_or_market_imports() -> None:
    completed = _run_in_clean_interpreter(
        r"""
from policy_learnware_v0.v03.acceptance import run_minimal_compute_acceptance
from policy_learnware_v0.v03.config import EncoderExtensionGateConfig

gate = EncoderExtensionGateConfig.disabled()
assert gate.to_dict() == {"enabled": False}
report = run_minimal_compute_acceptance()
assert report.passed
assert all(report.checks.values())
assert_no_blocked_module_loaded()
print("RAW_BASE_PATH_ISOLATED")
"""
    )
    assert completed.stdout.strip() == "RAW_BASE_PATH_ISOLATED"


def test_disabled_extension_legacy_callable_runs_all_14_views_without_optional_dependencies() -> None:
    completed = _run_in_clean_interpreter(
        r"""
import numpy as np

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.attribution import (
    ArchivedLegacyReference,
    AttributionMeasurement,
    CallableLegacyReplayAdapter,
    run_attribution_replay,
)
from policy_learnware_v0.v03.config import EncoderExtensionGateConfig
from policy_learnware_v0.v03.transition_views import (
    REGISTERED_VIEW_IDS,
    V_FULL_LEGACY,
    TransitionBank,
    apply_transition_view,
)


def digest(label):
    return sha256_json(
        {
            "schema": "policy-learnware.v03-extension-isolation-fixture.v0",
            "label": label,
        }
    )


bank = TransitionBank(
    observation=np.asarray(
        [[0.0, 0.2], [0.1, 0.3], [1.0, 1.2], [1.1, 1.3]],
        dtype=np.float32,
    ),
    action=np.asarray([[0.0], [0.1], [0.2], [0.3]], dtype=np.float32),
    reward=np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
    next_observation=np.asarray(
        [[0.1, 0.3], [0.2, 0.4], [1.1, 1.3], [1.2, 1.4]],
        dtype=np.float32,
    ),
    terminated=np.asarray([False, True, False, True]),
    truncated=np.asarray([False, False, False, False]),
    episode_offsets=np.asarray([0, 2, 4], dtype=np.int64),
    observation_mask=np.ones((4, 2), dtype=np.float32),
    action_mask=np.ones((4, 1), dtype=np.float32),
)


def replay(view, prefix_episode_counts):
    matrix = np.asarray(view.feature_matrix, dtype=np.float64)
    mean = float(np.mean(matrix))
    energy = float(np.mean(np.square(matrix)))
    return AttributionMeasurement(
        view_id=view.view_id,
        task_group="isolated-legacy-task",
        shared_schema_group="isolated-shared-schema",
        retrieval_metrics={"mean": mean},
        between_within_mmd_summaries={"energy": energy},
        prefix_curves={
            "mean": {
                int(prefix): mean for prefix in prefix_episode_counts
            }
        },
        failure_identifiability_notes=("isolated legacy callable smoke",),
    )


checkpoint_digest = digest("legacy-corro-checkpoint")
implementation_digest = digest("legacy-corro-implementation")
adapter = CallableLegacyReplayAdapter(
    encoder_checkpoint_digest=checkpoint_digest,
    implementation_digest=implementation_digest,
    replay_callable=replay,
)
full = replay(
    apply_transition_view(bank, V_FULL_LEGACY, shuffle_seed=7),
    (1, 2),
)
reference = ArchivedLegacyReference(
    archive_protocol_id="isolated-legacy-archive-v0",
    archive_manifest_digest=digest("archive-manifest"),
    archived_dataset_digest=str(bank.archived_dataset_digest),
    canonical_bank_digest=bank.canonical_bank_digest,
    encoder_checkpoint_digest=checkpoint_digest,
    encoder_implementation_digest=implementation_digest,
    reference_metrics=full.flat_metrics,
    absolute_tolerance=0.0,
    relative_tolerance=0.0,
)

gate = EncoderExtensionGateConfig.disabled()
assert gate.enabled is False
suite = run_attribution_replay(
    bank,
    adapter,
    reference,
    prefix_episode_counts=(1, 2),
    shuffle_seed=7,
)
assert len(REGISTERED_VIEW_IDS) == 14
assert {report.view_id for report in suite.reports} == set(REGISTERED_VIEW_IDS)
assert suite.gate_evidence.gate_status == "DEVELOPMENT_PASS"
assert_no_blocked_module_loaded()
print("LEGACY_14_VIEW_BASE_PATH_ISOLATED")
"""
    )
    assert completed.stdout.strip() == "LEGACY_14_VIEW_BASE_PATH_ISOLATED"
