from __future__ import annotations

import numpy as np

from policy_learnware_v0.hashing import sha256_json, sha256_ndarrays
from policy_learnware_v0.rkme.empirical import build_empirical_kme, empirical_mmd2
from policy_learnware_v0.rkme.gaussian import GaussianKernel
from policy_learnware_v0.v03.baselines import REQUIRED_BASELINE_METHOD_IDS
from policy_learnware_v0.v03.canonicalization import (
    GlobalCanonicalizerSpec,
    NativeShapeRegistry,
    NativeTransitionBank,
    fit_global_normalizer,
)
from policy_learnware_v0.v03.policy_outcomes import (
    ExternalOracleEvidenceManifest,
    OracleEpisodeEvidence,
    OraclePolicyEvidence,
)
from policy_learnware_v0.v03.prelarge_acceptance import run_prelarge_acceptance
from policy_learnware_v0.v03.probe_audit import ProbeDistanceEvidence
from policy_learnware_v0.v03.representation_ladder import (
    R0_PADDED_RAW,
    RepresentationBatch,
    TrainedCallableArtifact,
    bind_r4_frozen_callable,
    fit_r0_identity,
    fit_r1_random_linear,
    fit_r2_pca_whitening,
    fit_r3_matched_random_mlp,
    fit_r5_corro_style,
    fit_r5l_supervised_linear,
)
from policy_learnware_v0.v03.signal_controls import RewardFreeShuffledNextSpec
from policy_learnware_v0.v03.signal_matrix import (
    CORE_INPUT_VIEW_IDS,
    C_RF_SHUFFLED_NEXT,
    V_TEMPORAL_SHUFFLE,
    build_signal_matrix_plan,
)
from policy_learnware_v0.v03.transition_views import (
    V_FULL_LEGACY,
    V_MASK_ONLY,
    V_NO_MASK,
    V_REWARD_FREE_TRANSITION,
    V_REWARD_ONLY,
    TransitionBank,
    apply_transition_view,
)

# These are the real synthetic-market fixture and public baseline runner.  They
# are imported as pytest helpers so this integration test exercises the same
# typed market/index/query contracts as the production baseline path.
from tests.v03.test_baselines import baseline_case, _rank  # noqa: F401,E402
from tests.v03.test_signal_artifacts import (  # noqa: E402
    test_fresh_process_resume_verifies_completed_bytes_and_continues as _exercise_fresh_process_resume,
)


def _d(label: str) -> str:
    return sha256_json(
        {"schema": "policy-learnware.v03-scientific-fixture.v0", "label": label}
    )


def _bank(
    *,
    observation: np.ndarray,
    next_observation: np.ndarray,
    reward: np.ndarray | None = None,
    observation_mask: np.ndarray | None = None,
) -> TransitionBank:
    rows = len(observation)
    return TransitionBank(
        observation=np.asarray(observation, dtype=np.float32),
        action=np.linspace(-0.3, 0.3, rows, dtype=np.float32)[:, None],
        reward=(
            np.zeros(rows, dtype=np.float32)
            if reward is None
            else np.asarray(reward, dtype=np.float32)
        ),
        next_observation=np.asarray(next_observation, dtype=np.float32),
        terminated=np.asarray([False] * (rows - 1) + [True]),
        truncated=np.zeros(rows, dtype=np.bool_),
        episode_offsets=np.asarray([0, rows], dtype=np.int64),
        observation_mask=observation_mask,
        action_mask=np.ones((rows, 1), dtype=np.float32),
    )


def _mmd(left: np.ndarray, right: np.ndarray, label: str) -> float:
    kernel = GaussianKernel(1.0)
    protocol = _d(f"mmd-protocol:{label}")
    left_kme = build_empirical_kme(
        np.asarray(left, dtype=np.float64),
        kernel,
        protocol_id=protocol,
        dataset_digest=_d(f"left:{label}"),
    )
    right_kme = build_empirical_kme(
        np.asarray(right, dtype=np.float64),
        kernel,
        protocol_id=protocol,
        dataset_digest=_d(f"right:{label}"),
    )
    return empirical_mmd2(left_kme, right_kme)


def _collapsed_trainer(values, labels, request) -> TrainedCallableArtifact:
    del values, labels
    return TrainedCallableArtifact(
        checkpoint_bytes=("collapsed:" + request.request_digest).encode("ascii"),
        parameter_digest=_d(f"collapsed-parameter:{request.request_digest}"),
        trainer_implementation_digest=_d("collapsed-trainer"),
        transform=lambda batch: np.zeros(
            (len(batch), request.output_dim), dtype=np.float64
        ),
    )


def _projection_trainer(values, labels, request) -> TrainedCallableArtifact:
    del labels
    rng = np.random.default_rng(request.seed + 991)
    matrix = rng.normal(size=(values.shape[1], request.output_dim)).astype(np.float64)
    return TrainedCallableArtifact(
        checkpoint_bytes=("projection:" + request.request_digest).encode("ascii"),
        parameter_digest=sha256_ndarrays({"matrix": matrix}),
        trainer_implementation_digest=_d("projection-trainer"),
        transform=lambda batch: np.asarray(batch, dtype=np.float64) @ matrix,
    )


def test_synthetic_scientific_fixture_recovers_each_known_mechanism() -> None:
    base_observation = np.asarray(
        [[-1.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    base_next = base_observation + np.asarray([0.2, 0.0], dtype=np.float32)

    # Schema only: numeric state/action/reward/next-state arrays are identical;
    # only the active observation mask differs.  The schema view detects it,
    # while the mask-free view is exactly invariant.
    full_schema = _bank(
        observation=base_observation,
        next_observation=base_next,
        observation_mask=np.ones_like(base_observation),
    )
    narrow_schema = _bank(
        observation=base_observation,
        next_observation=base_next,
        observation_mask=np.tile([1.0, 0.0], (4, 1)),
    )
    schema_distance = _mmd(
        apply_transition_view(full_schema, V_MASK_ONLY).feature_matrix,
        apply_transition_view(narrow_schema, V_MASK_ONLY).feature_matrix,
        "schema-only",
    )
    no_mask_distance = _mmd(
        apply_transition_view(full_schema, V_NO_MASK).feature_matrix,
        apply_transition_view(narrow_schema, V_NO_MASK).feature_matrix,
        "schema-no-mask",
    )
    assert schema_distance > 0.1
    assert no_mask_distance == 0.0

    # Reward only: transition dynamics are byte-identical.  Reward view detects
    # the goal signal and the reward-free transition view is exactly invariant.
    reward_a = _bank(
        observation=base_observation,
        next_observation=base_next,
        reward=np.asarray([0.0, 0.0, 0.0, 0.0]),
    )
    reward_b = _bank(
        observation=base_observation,
        next_observation=base_next,
        reward=np.asarray([-2.0, -1.0, 1.0, 2.0]),
    )
    assert _mmd(
        apply_transition_view(reward_a, V_REWARD_ONLY).feature_matrix,
        apply_transition_view(reward_b, V_REWARD_ONLY).feature_matrix,
        "reward-only",
    ) > 0.1
    assert _mmd(
        apply_transition_view(reward_a, V_REWARD_FREE_TRANSITION).feature_matrix,
        apply_transition_view(reward_b, V_REWARD_FREE_TRANSITION).feature_matrix,
        "reward-free",
    ) == 0.0

    # Same marginals, different coupling: only the o->o' pairing changes.  The
    # formal C_RF control proves exact marginals while the joint KME recovers a
    # non-zero transition-mechanism signal.
    coupled = _bank(observation=base_observation, next_observation=base_next)
    permuted = _bank(
        observation=base_observation,
        next_observation=base_next[[2, 0, 3, 1]],
    )
    coupled_rf = apply_transition_view(coupled, V_REWARD_FREE_TRANSITION)
    permuted_rf = apply_transition_view(permuted, V_REWARD_FREE_TRANSITION)
    assert np.array_equal(
        np.sort(coupled.next_observation, axis=0),
        np.sort(permuted.next_observation, axis=0),
    )
    assert _mmd(coupled_rf.feature_matrix, permuted_rf.feature_matrix, "coupling") > 0.01
    shuffled = RewardFreeShuffledNextSpec(seed=7).apply(coupled)
    assert shuffled.control_id == C_RF_SHUFFLED_NEXT
    assert shuffled.marginal_audit.passed
    assert _mmd(coupled_rf.feature_matrix, shuffled.feature_matrix, "c-rf") > 0.01

    # Exact repeat is the calibrated numerical null, not a made-up epsilon.
    assert _mmd(coupled_rf.feature_matrix, coupled_rf.feature_matrix, "repeat") == 0.0

    # Probe nuisance is represented by the production R0 typed evidence: style
    # separation is deliberately small relative to true dynamics separation.
    nuisance = ProbeDistanceEvidence(
        task_id="fixture-task",
        axis_id="fixture-mass",
        representation_id=R0_PADDED_RAW,
        representation_protocol_digest=_d("probe-r0-protocol"),
        semantic_bank_digests={
            "cp0": _d("probe-cp0"),
            "cp2": _d("probe-cp2"),
            "shifted": _d("probe-shifted"),
        },
        encoder_checkpoint_digest=_d("raw-identity"),
        distance_matrix_digest=_d("probe-distance-matrix"),
        independent_recompute_digest=_d("probe-independent-recompute"),
        same_environment_cross_probe_distances=(0.02, 0.03),
        different_dynamics_same_probe_distances=(0.40, 0.45),
        repeated_bank_noise_distances=(0.005, 0.006),
        probe_style_classifier_accuracy=0.51,
    )
    assert nuisance.invariance_ratio < 0.1
    assert nuisance.signal_to_noise_ratio > 50.0

    # Raw-present/MLP-collapse: the typed trainer path is allowed to produce a
    # bad checkpoint, but the scientific fixture must expose the collapse.
    raw_values = np.asarray(
        [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64
    )
    source = RepresentationBatch(raw_values, _d("collapse-source"), "SOURCE_FIT")
    fitted = fit_r5_corro_style(
        source,
        labels=np.asarray(["left", "left", "right", "right"]),
        trainer=_collapsed_trainer,
        objective_digest=_d("collapse-objective"),
        seed=3,
        output_dim=2,
        hidden_dims=(4, 4),
    )
    collapsed = fitted.transform(
        RepresentationBatch(raw_values, _d("collapse-query"), "QUERY_TRANSFORM")
    ).values
    assert _mmd(raw_values[:2], raw_values[2:], "raw-present") > 0.1
    assert _mmd(collapsed[:2], collapsed[2:], "mlp-collapse") == 0.0


def _native(
    bank_id: str,
    task_id: str,
    role: str,
    observation_dim: int,
    action_dim: int,
    center: float,
) -> NativeTransitionBank:
    observation = center + np.arange(8 * observation_dim, dtype=np.float64).reshape(
        8, observation_dim
    ) / 20.0
    action = center / 10.0 + np.arange(8 * action_dim, dtype=np.float64).reshape(
        8, action_dim
    ) / 30.0
    return NativeTransitionBank(
        bank_id=bank_id,
        task_private_id=task_id,
        data_role=role,
        native_schema_digest=_d(f"schema:{task_id}"),
        raw_dataset_digest=_d(f"raw:{bank_id}"),
        observation=observation,
        action=action,
        reward=np.linspace(-0.5, 0.5, 8) + center / 100.0,
        next_observation=observation + 0.1,
        terminated=np.asarray([False, False, False, True] * 2),
        truncated=np.zeros(8, dtype=np.bool_),
        episode_id=np.asarray([0] * 4 + [1] * 4),
        timestep=np.asarray([0, 1, 2, 3] * 2),
    )


def test_cpu_canonical_views_representation_staircase_and_kme_ranking() -> None:
    train_native = (
        _native("train-a", "task-a", "source_representation_train", 2, 1, 0.0),
        _native("train-b", "task-b", "source_representation_train", 3, 2, 4.0),
    )
    registry = NativeShapeRegistry.from_source_banks(train_native)
    canonicalizer = GlobalCanonicalizerSpec(
        registry, fit_global_normalizer(train_native, registry=registry)
    )
    source_receipts = tuple(canonicalizer.transform(bank) for bank in train_native)
    source_banks = tuple(
        TransitionBank.from_canonical_batch(receipt.batch) for receipt in source_receipts
    )

    # canonicalizer -> all 13 scientific input views -> paired Raw/R5.  Temporal
    # shuffle is retained as the structural N/A and therefore is not refit.
    plan = build_signal_matrix_plan()
    numeric_views = []
    for view_id in CORE_INPUT_VIEW_IDS:
        views = tuple(apply_transition_view(bank, view_id, shuffle_seed=13) for bank in source_banks)
        values = np.concatenate([view.feature_matrix for view in views], axis=0)
        batch = RepresentationBatch(values, _d(f"source:{view_id}"), "SOURCE_FIT")
        raw = fit_r0_identity(batch)
        if view_id == V_TEMPORAL_SHUFFLE:
            assert plan.cell(f"CORE_PAIRED::{view_id}::R0_PADDED_RAW").applicability == "STRUCTURAL_NA"
            assert plan.cell(
                f"CORE_PAIRED::{view_id}::R5_VIEW_SPECIFIC_CORRO_REFIT"
            ).applicability == "STRUCTURAL_NA"
            continue
        fitted = fit_r5_corro_style(
            batch,
            labels=np.asarray(["task-a"] * 8 + ["task-b"] * 8),
            trainer=_projection_trainer,
            objective_digest=_d(f"objective:{view_id}"),
            seed=2,
            output_dim=2,
            hidden_dims=(4, 4),
        )
        transformed = fitted.transform(
            RepresentationBatch(values, _d(f"query:{view_id}"), "QUERY_TRANSFORM")
        )
        assert raw.manifest.output_dim == values.shape[1]
        assert transformed.values.shape == (16, 2)
        numeric_views.append(view_id)
    assert len(CORE_INPUT_VIEW_IDS) == 13
    assert len(numeric_views) == 12

    # FULL/RF/C_RF -> R1/R2/R3/R5L staircase is numerically executable on CPU.
    full = tuple(apply_transition_view(bank, V_FULL_LEGACY) for bank in source_banks)
    rf = tuple(
        apply_transition_view(bank, V_REWARD_FREE_TRANSITION) for bank in source_banks
    )
    c_rf = tuple(RewardFreeShuffledNextSpec(seed=17).apply(bank) for bank in source_banks)
    staircase_outputs = {}
    for condition, rows in (
        (V_FULL_LEGACY, full),
        (V_REWARD_FREE_TRANSITION, rf),
        (C_RF_SHUFFLED_NEXT, c_rf),
    ):
        values = np.concatenate([row.feature_matrix for row in rows], axis=0)
        source = RepresentationBatch(values, _d(f"staircase:{condition}"), "SOURCE_FIT")
        representations = (
            fit_r1_random_linear(source, output_dim=2, seed=0),
            fit_r2_pca_whitening(source, output_dim=2),
            fit_r3_matched_random_mlp(
                source, output_dim=2, hidden_dims=(4, 4), seed=1
            ),
            fit_r5l_supervised_linear(
                source,
                labels=np.asarray(["task-a"] * 8 + ["task-b"] * 8),
                trainer=_projection_trainer,
                objective_digest=_d(f"linear-objective:{condition}"),
                seed=2,
                output_dim=2,
            ),
        )
        for representation in representations:
            output = representation.transform(
                RepresentationBatch(
                    values, _d(f"staircase-query:{condition}"), "QUERY_TRANSFORM"
                )
            )
            staircase_outputs[(condition, representation.manifest.representation_id)] = output
    assert len(staircase_outputs) == 12

    # Fake archived/frozen and refit MLP both produce actual empirical KMEs and
    # recover the nearest source by sorting computed MMD, not by fixture labels.
    full_values = [view.feature_matrix for view in full]
    combined = RepresentationBatch(
        np.concatenate(full_values, axis=0), _d("kme-source"), "SOURCE_FIT"
    )
    frozen = bind_r4_frozen_callable(
        combined,
        output_dim=2,
        checkpoint_digest=_d("fake-archived-checkpoint"),
        normalizer_digest=_d("fake-archived-normalizer"),
        implementation_digest=_d("fake-archived-implementation"),
        transform=lambda values: np.asarray(values[:, :2], dtype=np.float64),
    )
    refit = fit_r5_corro_style(
        combined,
        labels=np.asarray(["task-a"] * 8 + ["task-b"] * 8),
        trainer=_projection_trainer,
        objective_digest=_d("kme-refit-objective"),
        seed=5,
        output_dim=2,
        hidden_dims=(4, 4),
    )
    for representation in (frozen, refit):
        source_outputs = [
            representation.transform(
                RepresentationBatch(values, _d(f"kme-bank:{i}"), "QUERY_TRANSFORM")
            ).values
            for i, values in enumerate(full_values)
        ]
        query = source_outputs[0].copy()
        kernel = GaussianKernel(1.0)
        protocol = _d(f"kme:{representation.manifest.representation_id}")
        query_kme = build_empirical_kme(
            query, kernel, protocol_id=protocol, dataset_digest=_d("kme-query")
        )
        distances = []
        for index, values in enumerate(source_outputs):
            source_kme = build_empirical_kme(
                values,
                kernel,
                protocol_id=protocol,
                dataset_digest=_d(f"kme-ranked-source:{index}"),
            )
            distances.append(empirical_mmd2(query_kme, source_kme))
        assert sorted(range(len(distances)), key=distances.__getitem__)[0] == 0
        assert distances[0] == 0.0


def test_synthetic_market_baselines_feed_a_typed_oracle_rectangle(baseline_case) -> None:
    rankings = {
        method_id: _rank(baseline_case, method_id)
        for method_id in REQUIRED_BASELINE_METHOD_IDS
    }
    assert set(rankings) == set(REQUIRED_BASELINE_METHOD_IDS)
    assert all(len(ranking.rows) == 30 for ranking in rankings.values())

    query_id = baseline_case["raw_query"].opaque_query_id
    policy_ids = tuple(sorted(baseline_case["market"].entries))
    rows = []
    for index, policy_id in enumerate(policy_ids):
        episode = OracleEpisodeEvidence(
            episode_id="episode-0",
            episode_seed_digest=_d(f"oracle-seed:{policy_id}"),
            status="EXECUTED",
            return_value=float(index),
            evidence_digest=_d(f"oracle-evidence:{policy_id}"),
        )
        rows.append(
            OraclePolicyEvidence(
                opaque_query_id=query_id,
                opaque_policy_id=policy_id,
                target_execution_abi_digest=_d("target-abi"),
                policy_execution_abi_digest=_d(f"policy-abi:{policy_id}"),
                executable=True,
                policy_value=float(index),
                episodes=(episode,),
            )
        )
    oracle = ExternalOracleEvidenceManifest(
        scope="DEVELOPMENT",
        run_id="scientific-fixture",
        freeze_manifest_digest=_d("development-freeze"),
        public_ranking_barrier_digest=_d("development-barrier"),
        public_query_plan_digest=_d("development-query-plan"),
        query_alias_manifest_digest=_d("development-aliases"),
        signal_outcome_manifest_digest=_d("development-signal-outcomes"),
        policy_market_id=baseline_case["market"].policy_market_id,
        expected_opaque_query_ids=(query_id,),
        expected_opaque_policy_ids=policy_ids,
        episode_ids_by_query={query_id: ("episode-0",)},
        rows=tuple(rows),
    )
    oracle_by_policy = {row.opaque_policy_id: row.policy_value for row in oracle.rows}
    for ranking in rankings.values():
        assert ranking.opaque_query_id == query_id
        assert ranking.selected_opaque_learnware_id in oracle_by_policy
        assert oracle_by_policy[ranking.selected_opaque_learnware_id] is not None
    assert oracle.scope == "DEVELOPMENT"


def test_cpu_forced_interruption_and_fresh_process_resume(tmp_path) -> None:
    # Reuse the canonical typed atlas fixture rather than inventing a second
    # checkpoint format here.  It forces a mid-run exception, restarts with a
    # new runner object, verifies completed bytes, and rejects later tampering.
    _exercise_fresh_process_resume(tmp_path)


def test_optional_extension_absence_preserves_prelarge_completion_eligibility(
    tmp_path,
) -> None:
    # The acceptance path is run in a root where no v0.4/optional directory was
    # created.  Passing it cannot authorize formal execution, but it proves the
    # extension is neither imported nor an engineering-completion dependency.
    assert not (tmp_path / "optional_extensions").exists()
    report = run_prelarge_acceptance()
    assert report.passed
    assert report.v04_assets_required is False
    assert report.formal_run_authorized is False
    assert report.large_experiment_executed is False
    assert not (tmp_path / "optional_extensions").exists()
