from __future__ import annotations

from pathlib import Path
import sys

import pytest

from policy_learnware_v0.v02.schemas import ExecutionABIRecord
from policy_learnware_v0.v03.fpo_source_backend import (
    FpoJaxSourceEvaluatorBackend,
    FpoSourceBackendError,
)
from policy_learnware_v0.v03.source_evaluator import (
    BackendEpisodeResult,
    CanonicalSourceAnchor,
    SourceCandidateRequest,
)

sys.path.insert(0, str(Path(__file__).parent))
from p5_asset_fixtures import digest, exact90_handoff  # noqa: E402


def _abi() -> ExecutionABIRecord:
    return ExecutionABIRecord(
        protocol_family_id="continuous-vector-mdp-v02",
        observation_tensor_abi_digest=digest("observation-abi"),
        action_tensor_abi_digest=digest("action-abi"),
        action_transform_id="tanh",
        policy_runtime_id="legacy-ppo-fpo-v0",
        state_abi_id="stateless-v0",
    )


class RecordingDriver:
    def __init__(self, *, driver_digest: str = digest("runtime-driver")) -> None:
        self.runtime_driver_digest = driver_digest
        self.validate_calls: list[str] = []
        self.blocks: list[tuple[int, ...]] = []
        self.runtime_drift = False
        self.reverse = False

    def validate_candidate(self, request: SourceCandidateRequest) -> ExecutionABIRecord:
        self.validate_calls.append(request.request_digest)
        return _abi()

    def evaluate_seed_block(
        self,
        request: SourceCandidateRequest,
        *,
        reset_seeds: tuple[int, ...],
    ) -> tuple[BackendEpisodeResult, ...]:
        self.blocks.append(reset_seeds)
        runtime = (
            digest("runtime-drift")
            if self.runtime_drift
            else request.anchor.runtime_digest
        )
        rows = tuple(
            BackendEpisodeResult.succeeded(
                reset_seed=seed,
                runtime_digest=runtime,
                raw_return=float(seed),
                steps=1000,
                terminated=False,
                truncated=True,
            )
            for seed in reset_seeds
        )
        return tuple(reversed(rows)) if self.reverse else rows


def _request(
    tmp_path: Path,
    *,
    evaluator_implementation_digest: str,
) -> SourceCandidateRequest:
    root, handoff, _trust = exact90_handoff(tmp_path)
    acceptance = __import__("json").loads(
        (handoff / "policy_pool_acceptance.json").read_text(encoding="utf-8")
    )
    candidate_id, cell = next(iter(sorted(acceptance["cells"].items())))
    anchor = CanonicalSourceAnchor.from_path(
        root / "source_anchor_manifests" / f"{cell['source_anchor_id']}.json"
    )
    return SourceCandidateRequest(
        evaluator_implementation_digest=evaluator_implementation_digest,
        intake_cell_digest=digest("intake-cell"),
        candidate_id=candidate_id,
        source_anchor_id=cell["source_anchor_id"],
        attempt_number=cell["attempt_number"],
        attempt_digest=cell["attempt_digest"],
        bundle_path=cell["bundle_path"],
        bundle_digest=cell["bundle_digest"],
        outer_iteration=cell["outer_iteration"],
        environment_steps=cell["environment_steps"],
        source_environment_digest=anchor.environment_instance_digest,
        anchor=anchor,
    )


def _backend(driver: RecordingDriver) -> FpoJaxSourceEvaluatorBackend:
    return FpoJaxSourceEvaluatorBackend(
        runtime_driver=driver,
        selection_reset_seeds=(101, 102, 103),
        attestation_reset_seeds=(201, 202),
    )


def test_backend_batches_each_frozen_seed_block_once(tmp_path: Path) -> None:
    driver = RecordingDriver()
    backend = _backend(driver)
    request = _request(
        tmp_path,
        evaluator_implementation_digest=backend.evaluator_implementation_digest,
    )
    binding = backend.validate_candidate(request)

    assert backend.evaluate_episode(binding, reset_seed=101).raw_return == 101.0
    assert backend.evaluate_episode(binding, reset_seed=103).raw_return == 103.0
    assert driver.blocks == [(101, 102, 103)]

    assert backend.evaluate_episode(binding, reset_seed=202).raw_return == 202.0
    assert backend.evaluate_episode(binding, reset_seed=201).raw_return == 201.0
    assert driver.blocks == [(101, 102, 103), (201, 202)]


def test_backend_rejects_unfrozen_seed_and_driver_block_drift(tmp_path: Path) -> None:
    driver = RecordingDriver()
    backend = _backend(driver)
    request = _request(
        tmp_path,
        evaluator_implementation_digest=backend.evaluator_implementation_digest,
    )
    binding = backend.validate_candidate(request)
    with pytest.raises(FpoSourceBackendError, match="outside both frozen"):
        backend.evaluate_episode(binding, reset_seed=999)

    driver.runtime_drift = True
    with pytest.raises(FpoSourceBackendError, match="drifted block rows"):
        backend.evaluate_episode(binding, reset_seed=101)


def test_backend_rejects_reordered_rows_and_overlapping_seed_blocks(
    tmp_path: Path,
) -> None:
    driver = RecordingDriver()
    driver.reverse = True
    backend = _backend(driver)
    request = _request(
        tmp_path,
        evaluator_implementation_digest=backend.evaluator_implementation_digest,
    )
    binding = backend.validate_candidate(request)
    with pytest.raises(FpoSourceBackendError, match="reordered"):
        backend.evaluate_episode(binding, reset_seed=101)

    with pytest.raises(FpoSourceBackendError, match="overlap"):
        FpoJaxSourceEvaluatorBackend(
            runtime_driver=RecordingDriver(),
            selection_reset_seeds=(1, 2),
            attestation_reset_seeds=(2, 3),
        )


def test_backend_digest_binds_driver_and_request_identity(tmp_path: Path) -> None:
    left = _backend(RecordingDriver(driver_digest=digest("driver-left")))
    right = _backend(RecordingDriver(driver_digest=digest("driver-right")))
    assert left.evaluator_implementation_digest != right.evaluator_implementation_digest
    different_seeds = FpoJaxSourceEvaluatorBackend(
        runtime_driver=RecordingDriver(driver_digest=digest("driver-left")),
        selection_reset_seeds=(101, 102, 104),
        attestation_reset_seeds=(201, 202),
    )
    assert (
        left.evaluator_implementation_digest
        != different_seeds.evaluator_implementation_digest
    )

    request = _request(
        tmp_path,
        evaluator_implementation_digest=left.evaluator_implementation_digest,
    )
    with pytest.raises(FpoSourceBackendError, match="another evaluator"):
        right.validate_candidate(request)


def test_backend_rejects_uint32_policy_seed_overflow() -> None:
    with pytest.raises(FpoSourceBackendError, match="uint32-compatible"):
        FpoJaxSourceEvaluatorBackend(
            runtime_driver=RecordingDriver(),
            selection_reset_seeds=(2**32 - 1,),
            attestation_reset_seeds=(1,),
        )
