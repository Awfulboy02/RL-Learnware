from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.canonicalization import (
    CanonicalizationError,
    GlobalCanonicalizerSpec,
    NativeShapeRegistry,
    NativeTransitionBank,
    fit_global_normalizer,
    require_formal_cross_task_raw_receipts,
)


def _d(label: str) -> str:
    return sha256_json(
        {"schema": "policy-learnware.v03-canonicalization-test.v0", "label": label}
    )


def _bank(
    task: str,
    *,
    observation_dim: int,
    action_dim: int,
    role: str = "source_representation_train",
    shift: float = 0.0,
    bank_suffix: str = "a",
) -> NativeTransitionBank:
    rows = 4
    observation = (
        np.arange(rows * observation_dim, dtype=np.float64).reshape(
            rows, observation_dim
        )
        / 10.0
        + shift
    )
    action = (
        np.arange(rows * action_dim, dtype=np.float64).reshape(rows, action_dim)
        / 20.0
        - shift
    )
    return NativeTransitionBank(
        bank_id=f"bank-{task}-{bank_suffix}",
        task_private_id=task,
        data_role=role,  # type: ignore[arg-type]
        native_schema_digest=_d(f"schema:{task}"),
        raw_dataset_digest=_d(f"raw:{task}:{bank_suffix}:{role}"),
        observation=observation,
        action=action,
        reward=np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float64) + shift,
        next_observation=observation + 0.05,
        terminated=np.asarray([False, True, False, True]),
        truncated=np.asarray([False, False, False, False]),
        episode_id=np.asarray([0, 0, 1, 1], dtype=np.int64),
        timestep=np.asarray([0, 1, 0, 1], dtype=np.int64),
    )


def _protocol():
    walker = _bank("walker", observation_dim=2, action_dim=1)
    finger = _bank("finger", observation_dim=3, action_dim=2)
    registry = NativeShapeRegistry.from_source_banks(
        (walker, finger), max_observation_dim=4, max_action_dim=3
    )
    normalizer = fit_global_normalizer((walker, finger), registry)
    return walker, finger, registry, normalizer, GlobalCanonicalizerSpec(
        registry=registry, normalizer=normalizer
    )


def test_heterogeneous_native_widths_share_one_frozen_coordinate_system() -> None:
    walker, _, registry, normalizer, canonicalizer = _protocol()
    finger_query = _bank(
        "finger",
        observation_dim=3,
        action_dim=2,
        role="development_query",
        shift=2.0,
        bank_suffix="query",
    )
    walker_receipt = canonicalizer.transform(walker)
    finger_receipt = canonicalizer.transform(finger_query)

    assert walker_receipt.batch.observation.shape == (4, 4)
    assert finger_receipt.batch.observation.shape == (4, 4)
    assert walker_receipt.batch.action.shape == (4, 3)
    assert finger_receipt.batch.action.shape == (4, 3)
    np.testing.assert_array_equal(
        walker_receipt.batch.observation_mask, [True, True, False, False]
    )
    np.testing.assert_array_equal(
        finger_receipt.batch.observation_mask, [True, True, True, False]
    )
    np.testing.assert_array_equal(
        walker_receipt.batch.action_mask, [True, False, False]
    )
    assert np.all(walker_receipt.batch.observation[:, 2:] == 0.0)
    assert np.all(finger_receipt.batch.action[:, 2:] == 0.0)
    assert walker_receipt.canonicalizer_digest == finger_receipt.canonicalizer_digest
    assert walker_receipt.normalizer_digest == normalizer.normalizer_digest
    assert walker_receipt.native_shape_registry_digest == registry.registry_digest

    admitted = require_formal_cross_task_raw_receipts(
        (walker_receipt, finger_receipt)
    )
    assert admitted == (walker_receipt, finger_receipt)


def test_normalizer_fit_rejects_target_or_query_banks_and_query_is_transform_only() -> None:
    walker, finger, registry, normalizer, canonicalizer = _protocol()
    query = _bank(
        "finger",
        observation_dim=3,
        action_dim=2,
        role="confirmatory_query",
        shift=9.0,
        bank_suffix="confirm",
    )
    with pytest.raises(CanonicalizationError, match="source-only"):
        fit_global_normalizer((walker, query), registry)
    before = normalizer.normalizer_digest
    receipt = canonicalizer.transform(query)
    assert normalizer.normalizer_digest == before
    assert query.native_bank_digest not in normalizer.source_bank_digests
    assert receipt.data_role == "confirmatory_query"
    assert set(normalizer.source_bank_digests) == {
        walker.native_bank_digest,
        finger.native_bank_digest,
    }


def test_registry_receipt_tamper_and_native_width_bypass_fail_closed() -> None:
    walker, _, registry, _, canonicalizer = _protocol()
    receipt = canonicalizer.transform(walker)
    with pytest.raises(CanonicalizationError, match="registry_digest"):
        replace(registry, registry_digest=_d("tampered-registry"))
    with pytest.raises(CanonicalizationError, match="canonical_transition_digest"):
        replace(receipt, canonical_transition_digest=_d("tampered-transition"))
    with pytest.raises(CanonicalizationError, match="receipt_digest"):
        replace(receipt, receipt_digest=_d("tampered-receipt"))
    with pytest.raises(CanonicalizationError, match="typed"):
        require_formal_cross_task_raw_receipts(  # type: ignore[arg-type]
            (receipt, receipt.batch)
        )


def test_unregistered_or_shape_drifted_query_is_rejected() -> None:
    _, _, _, _, canonicalizer = _protocol()
    unknown = _bank("cheetah", observation_dim=2, action_dim=1)
    with pytest.raises(CanonicalizationError, match="absent"):
        canonicalizer.transform(unknown)
    drifted = NativeTransitionBank(
        **{
            **_bank(
                "walker",
                observation_dim=3,
                action_dim=1,
                role="development_query",
                bank_suffix="drift",
            ).__dict__,
            "native_schema_digest": _d("schema:walker"),
        }
    )
    with pytest.raises(CanonicalizationError, match="differs"):
        canonicalizer.transform(drifted)
