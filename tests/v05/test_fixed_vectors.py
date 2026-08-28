from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from policy_learnware_v0.rkme.distance import empirical_to_reduced_distance
from policy_learnware_v0.rkme.empirical import build_empirical_kme
from policy_learnware_v0.rkme.gaussian import GaussianKernel
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v05.classifiers import (
    EmpiricalMMDNN,
    EpisodeBank,
    KMEKRR,
    RawDeltaRKMENN,
    SummaryLogReg,
)
from policy_learnware_v0.v05.specifications import (
    RFFMap,
    RFFSpecification,
    SWEMap,
    SWESpecification,
    V05SpecificationError,
    squared_vector_distance,
)


NORMALIZATION_DIGEST = "0123456789abcdef" * 4
OTHER_NORMALIZATION_DIGEST = "f" * 64
PROTOCOL_ID = "q0-common-gaussian-open-loop"


def _bank(*episodes: list[float]) -> EpisodeBank:
    arrays = [
        np.asarray(episode, dtype=np.float64).reshape(-1, 1) for episode in episodes
    ]
    return EpisodeBank(
        np.concatenate(arrays),
        np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.cumsum([len(array) for array in arrays], dtype=np.int64),
            )
        ),
    )


def _b2_query() -> EpisodeBank:
    """Shared counterexample: equal-episode and transition pooling differ."""

    return _bank([0.0], [4.0, 4.0, 4.0])


def _gaussian_gram(
    left: np.ndarray, right: np.ndarray, *, bandwidth: float
) -> np.ndarray:
    squared_distances = np.sum(
        (left[:, None, :] - right[None, :, :]) ** 2,
        axis=2,
    )
    return np.exp(-squared_distances / (2.0 * bandwidth**2))


def _replace_npz_entry(path, *, name: str, value: str) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload[name] = np.asarray(value)
    np.savez(path, **payload)


def test_rff_public_replay_dimension_and_point_norm() -> None:
    first = RFFMap(
        input_dim=2,
        bandwidth=0.8,
        normalization_digest=NORMALIZATION_DIGEST,
        frequency_count=128,
        public_seed=17,
    )
    second = RFFMap(
        input_dim=2,
        bandwidth=0.8,
        normalization_digest=NORMALIZATION_DIGEST,
        frequency_count=128,
        public_seed=17,
    )

    np.testing.assert_array_equal(first.frequencies, second.frequencies)
    assert first.map_digest == second.map_digest
    assert first.output_dim == 256

    features = first.transform_points(np.array([[0.0, 0.0], [1.5, -0.5], [-0.25, 2.0]]))
    assert features.shape == (3, 256)
    np.testing.assert_allclose(
        np.linalg.norm(features, axis=1),
        np.ones(3),
        rtol=0.0,
        atol=64.0 * np.finfo(np.float64).eps,
    )


def test_rff_npz_round_trip_for_map_and_specification(tmp_path) -> None:
    public_map = RFFMap(
        input_dim=2,
        bandwidth=1.25,
        normalization_digest=NORMALIZATION_DIGEST,
        frequency_count=64,
        public_seed=23,
    )
    points = np.array([[0.0, 1.0], [2.0, -1.0], [3.0, 0.5]])
    specification = public_map.embed(points, np.array([0, 1, 3]))

    map_path = tmp_path / "rff_map.npz"
    spec_path = tmp_path / "rff_spec.npz"
    public_map.save_npz(map_path)
    specification.save_npz(spec_path)

    loaded_map = RFFMap.load_npz(map_path)
    loaded_spec = RFFSpecification.load_npz(spec_path)
    assert loaded_map.map_digest == public_map.map_digest
    assert loaded_map.normalization_digest == NORMALIZATION_DIGEST
    np.testing.assert_array_equal(loaded_map.frequencies, public_map.frequencies)
    assert loaded_spec.specification_digest == specification.specification_digest
    np.testing.assert_array_equal(loaded_spec.vector, specification.vector)


def test_rff_uses_equal_episode_aggregation() -> None:
    public_map = RFFMap(
        input_dim=2,
        bandwidth=0.7,
        normalization_digest=NORMALIZATION_DIGEST,
        frequency_count=256,
        public_seed=31,
    )
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )
    offsets = np.array([0, 1, 4])
    features = public_map.transform_points(points)
    expected = 0.5 * features[0] + 0.5 * np.mean(features[1:], axis=0)
    transition_pooled = np.mean(features, axis=0)

    actual = public_map.embed(points, offsets).vector
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-15)
    assert not np.allclose(actual, transition_pooled, rtol=0.0, atol=1.0e-8)


def test_rff_squared_distance_approximates_biased_gaussian_mmd() -> None:
    bandwidth = 0.85
    left = np.array([[-1.2, 0.3], [-0.2, 0.9], [0.6, -0.7], [1.1, 0.4]])
    right = np.array([[-0.8, -0.4], [0.1, 0.2], [0.9, 1.0]])
    public_map = RFFMap(
        input_dim=2,
        bandwidth=bandwidth,
        normalization_digest=NORMALIZATION_DIGEST,
        frequency_count=32_768,
        public_seed=1_701,
    )

    left_vector = public_map.embed(left, np.array([0, len(left)])).vector
    right_vector = public_map.embed(right, np.array([0, len(right)])).vector
    approximate = squared_vector_distance(left_vector, right_vector)
    exact_biased = float(
        np.mean(_gaussian_gram(left, left, bandwidth=bandwidth))
        + np.mean(_gaussian_gram(right, right, bandwidth=bandwidth))
        - 2.0 * np.mean(_gaussian_gram(left, right, bandwidth=bandwidth))
    )

    assert approximate == pytest.approx(exact_biased, abs=3.0e-3)


def test_swe_midpoint_grid_linear_interpolation_and_constant_endpoints() -> None:
    public_map = SWEMap(
        input_dim=1,
        normalization_digest=NORMALIZATION_DIGEST,
        direction_count=1,
        quantile_count=4,
        public_seed=0,
    )

    np.testing.assert_array_equal(
        public_map.quantile_grid,
        np.array([0.125, 0.375, 0.625, 0.875]),
    )
    assert np.all((public_map.quantile_grid > 0.0) & (public_map.quantile_grid < 1.0))
    np.testing.assert_array_equal(public_map.directions, np.array([[1.0]]))

    # Equal atoms at 0 and 10 have interpolation knots at 0.25 and 0.75.
    # The outer public quantiles therefore exercise both constant endpoints.
    specification = public_map.embed(
        np.array([[0.0], [10.0]]),
        np.array([0, 2]),
    )
    np.testing.assert_allclose(
        specification.vector,
        np.array([0.0, 1.25, 3.75, 5.0]),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert specification.vector.shape == (public_map.output_dim,)


def test_swe_permutation_identity_and_nonnegative_distance() -> None:
    public_map = SWEMap(
        input_dim=2,
        normalization_digest=NORMALIZATION_DIGEST,
        direction_count=9,
        quantile_count=11,
        public_seed=41,
    )
    first_episode = np.array([[0.0, 1.0], [2.0, -1.0]])
    second_episode = np.array([[1.0, 3.0], [-2.0, 0.0], [4.0, 2.0]])
    points = np.concatenate((first_episode, second_episode), axis=0)
    permuted = np.concatenate(
        (second_episode[[2, 0, 1]], first_episode[[1, 0]]),
        axis=0,
    )

    original = public_map.embed(points, np.array([0, 2, 5]))
    reordered = public_map.embed(permuted, np.array([0, 3, 5]))
    np.testing.assert_allclose(original.vector, reordered.vector, rtol=0.0, atol=2e-15)
    assert squared_vector_distance(original.vector, original.vector) == 0.0
    assert squared_vector_distance(original.vector, reordered.vector) >= 0.0


def test_swe_repeated_atoms_with_unequal_episode_weights_are_permutation_invariant() -> None:
    public_map = SWEMap(
        input_dim=2,
        normalization_digest=NORMALIZATION_DIGEST,
        direction_count=5,
        quantile_count=7,
        public_seed=19,
    )
    # The origin appears once in the one-row episode (mass 1/2) and once in
    # the three-row episode (mass 1/6), creating equal projections with
    # different atom weights in every public direction.
    points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    reordered = np.array([[2.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

    original = public_map.embed(points, np.array([0, 1, 4]))
    permuted = public_map.embed(reordered, np.array([0, 3, 4]))
    np.testing.assert_allclose(original.vector, permuted.vector, rtol=0.0, atol=2e-15)


def test_swe_uses_equal_episode_mixture_before_quantiles() -> None:
    public_map = SWEMap(
        input_dim=1,
        normalization_digest=NORMALIZATION_DIGEST,
        direction_count=1,
        quantile_count=4,
        public_seed=0,
    )
    query = _b2_query()

    # The one-row and three-row episodes each contribute mass 1/2. Quantiles
    # are constructed after mixing them, not by transition pooling.
    actual = public_map.embed(query.points, query.episode_offsets).vector
    expected = np.array([0.0, 0.5, 1.5, 2.0])
    transition_pooled = np.array([0.0, 1.0, 2.0, 2.0])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)
    assert not np.allclose(actual, transition_pooled, rtol=0.0, atol=1.0e-8)


def test_swe_npz_round_trip_for_map_and_specification(tmp_path) -> None:
    public_map = SWEMap(
        input_dim=2,
        normalization_digest=NORMALIZATION_DIGEST,
        direction_count=7,
        quantile_count=5,
        public_seed=47,
    )
    specification = public_map.embed(
        np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 3.0]]),
        np.array([0, 1, 3]),
    )
    map_path = tmp_path / "swe_map.npz"
    spec_path = tmp_path / "swe_spec.npz"
    public_map.save_npz(map_path)
    specification.save_npz(spec_path)

    loaded_map = SWEMap.load_npz(map_path)
    loaded_spec = SWESpecification.load_npz(spec_path)
    assert loaded_map.map_digest == public_map.map_digest
    assert loaded_map.normalization_digest == NORMALIZATION_DIGEST
    np.testing.assert_array_equal(loaded_map.directions, public_map.directions)
    np.testing.assert_array_equal(loaded_map.quantile_grid, public_map.quantile_grid)
    assert loaded_spec.specification_digest == specification.specification_digest
    np.testing.assert_array_equal(loaded_spec.vector, specification.vector)


@pytest.mark.parametrize("invalid_digest", ["a" * 63, "A" * 64, "g" * 64])
def test_maps_reject_invalid_normalization_digest(invalid_digest: str) -> None:
    with pytest.raises(V05SpecificationError, match="lowercase SHA-256"):
        RFFMap(
            input_dim=2,
            bandwidth=1.0,
            normalization_digest=invalid_digest,
        )
    with pytest.raises(V05SpecificationError, match="lowercase SHA-256"):
        SWEMap(input_dim=2, normalization_digest=invalid_digest)


def test_normalization_identity_is_bound_into_map_digest() -> None:
    first_rff = RFFMap(
        input_dim=2,
        bandwidth=1.0,
        normalization_digest=NORMALIZATION_DIGEST,
        public_seed=3,
    )
    other_rff = RFFMap(
        input_dim=2,
        bandwidth=1.0,
        normalization_digest=OTHER_NORMALIZATION_DIGEST,
        public_seed=3,
    )
    first_swe = SWEMap(
        input_dim=2,
        normalization_digest=NORMALIZATION_DIGEST,
        public_seed=5,
    )
    other_swe = SWEMap(
        input_dim=2,
        normalization_digest=OTHER_NORMALIZATION_DIGEST,
        public_seed=5,
    )
    assert first_rff.map_digest != other_rff.map_digest
    assert first_swe.map_digest != other_swe.map_digest


def test_maps_reject_persisted_map_digest_mismatch(tmp_path) -> None:
    rff = RFFMap(
        input_dim=2,
        bandwidth=1.0,
        normalization_digest=NORMALIZATION_DIGEST,
        frequency_count=8,
        public_seed=3,
    )
    swe = SWEMap(
        input_dim=2,
        normalization_digest=NORMALIZATION_DIGEST,
        direction_count=4,
        quantile_count=4,
        public_seed=5,
    )
    rff_path = tmp_path / "rff_map_mismatch.npz"
    swe_path = tmp_path / "swe_map_mismatch.npz"
    rff.save_npz(rff_path)
    swe.save_npz(swe_path)
    _replace_npz_entry(rff_path, name="map_digest", value=OTHER_NORMALIZATION_DIGEST)
    _replace_npz_entry(swe_path, name="map_digest", value=OTHER_NORMALIZATION_DIGEST)

    with pytest.raises(V05SpecificationError, match="RFF map digest does not match"):
        RFFMap.load_npz(rff_path)
    with pytest.raises(V05SpecificationError, match="SWE map digest does not match"):
        SWEMap.load_npz(swe_path)


def test_public_arrays_must_replay_their_frozen_map() -> None:
    rff = RFFMap(
        input_dim=2,
        bandwidth=0.9,
        normalization_digest=NORMALIZATION_DIGEST,
        frequency_count=16,
        public_seed=7,
    )
    mismatched_frequencies = rff.frequencies.copy()
    mismatched_frequencies[0, 0] += 1.0e-12
    with pytest.raises(V05SpecificationError, match="do not replay"):
        RFFMap(
            input_dim=2,
            bandwidth=0.9,
            normalization_digest=NORMALIZATION_DIGEST,
            frequency_count=16,
            public_seed=7,
            frequencies=mismatched_frequencies,
        )

    swe = SWEMap(
        input_dim=2,
        normalization_digest=NORMALIZATION_DIGEST,
        direction_count=4,
        quantile_count=4,
        public_seed=11,
    )
    mismatched_directions = swe.directions.copy()
    mismatched_directions[0, 0] += 1.0e-12
    with pytest.raises(V05SpecificationError, match="do not replay"):
        SWEMap(
            input_dim=2,
            normalization_digest=NORMALIZATION_DIGEST,
            direction_count=4,
            quantile_count=4,
            public_seed=11,
            directions=mismatched_directions,
        )

    mismatched_grid = swe.quantile_grid.copy()
    mismatched_grid[0] = math.nextafter(mismatched_grid[0], 1.0)
    with pytest.raises(V05SpecificationError, match="frozen midpoint grid"):
        SWEMap(
            input_dim=2,
            normalization_digest=NORMALIZATION_DIGEST,
            direction_count=4,
            quantile_count=4,
            public_seed=11,
            quantile_grid=mismatched_grid,
        )


def test_mmd_and_raw_use_one_episode_balanced_b2_mixture() -> None:
    query = _b2_query()
    sources = {"far": _bank([9.0], [10.0, 10.5, 11.0]), "match": query}
    bandwidth = 2.0
    mmd = EmpiricalMMDNN.fit(sources, bandwidth=bandwidth, protocol_id=PROTOCOL_ID)
    np.testing.assert_allclose(
        mmd.sources["match"].weights,
        [0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
        rtol=0.0,
        atol=4.0 * np.finfo(np.float64).eps,
    )
    cross = math.exp(-(4.0**2) / (2.0 * bandwidth**2))
    assert mmd.sources["match"].norm2 == pytest.approx(0.5 + 0.5 * cross)
    mmd_scores = mmd.score(query)
    assert mmd_scores["match"] > mmd_scores["far"]
    assert mmd_scores["match"] != pytest.approx(
        np.mean(
            [mmd.score(_bank([0.0]))["match"], mmd.score(_bank([4.0] * 3))["match"]]
        )
    )

    raw = RawDeltaRKMENN.fit(
        sources,
        bandwidth=bandwidth,
        protocol_id=PROTOCOL_ID,
        reducer_config=ReducerConfig(
            support_budget=4,
            support_steps=0,
            kmeans_steps=0,
            ridge=0.0,
            pinv_rcond=1.0e-12,
        ),
    )
    empirical = build_empirical_kme(
        query.points,
        GaussianKernel(bandwidth),
        episode_offsets=query.episode_offsets,
        protocol_id=PROTOCOL_ID,
        dataset_digest=query.bank_digest,
    )
    expected = {
        source_id: -empirical_to_reduced_distance(empirical, representation).distance
        for source_id, representation in raw.sources.items()
    }
    assert raw.score(query) == pytest.approx(expected, abs=2.0e-15)
    assert raw.score(query)["match"] != pytest.approx(
        np.mean(
            [raw.score(_bank([0.0]))["match"], raw.score(_bank([4.0] * 3))["match"]]
        )
    )


def test_summary_logreg_is_source_only_and_means_b2_episode_logits(tmp_path) -> None:
    def pair(center: float) -> list[float]:
        return [center - 0.1, center + 0.1]

    train = {
        "mid": _bank(*[pair(value) for value in (-0.5, 0.0, 0.5)]),
        "neg": _bank(*[pair(value) for value in (-5.0, -4.5, -4.0)]),
        "pos": _bank(*[pair(value) for value in (4.0, 4.5, 5.0)]),
    }
    labels = {"mid": "class-mid", "neg": "class-neg", "pos": "class-pos"}
    validation = {
        "mid": _bank(pair(0.25)),
        "neg": _bank(pair(-4.25)),
        "pos": _bank(pair(4.25)),
    }
    parameters = inspect.signature(SummaryLogReg.fit).parameters
    assert tuple(parameters)[:3] == (
        "source_train",
        "source_labels",
        "source_validation",
    )
    assert all("target" not in name for name in parameters)
    grid = (0.0, 1.0e-3, 1.0e-1, 10.0)
    model = SummaryLogReg.fit(
        train, labels, validation, l2_grid=grid, max_iter=300, tolerance=1.0e-8
    )
    assert model.selected_l2 in grid
    for source_id, bank in train.items():
        scores = model.score(bank)
        assert max(scores, key=scores.get) == labels[source_id]

    query = _b2_query()
    episode_logits = model.episode_logits(query)
    scores = model.score(query)
    np.testing.assert_allclose(
        [scores[class_id] for class_id in model.class_ids],
        np.mean(episode_logits, axis=0),
    )
    merged = np.concatenate(
        (np.mean(query.points, axis=0), np.std(query.points, axis=0))
    )
    merged_logits = (
        (merged - model.feature_mean) / model.feature_scale
    ) @ model.weights
    merged_logits += model.intercept
    assert not np.allclose(np.mean(episode_logits, axis=0), merged_logits)

    path = tmp_path / "summary.npz"
    model.save_npz(path)
    loaded = SummaryLogReg.load_npz(path)
    assert loaded.model_digest == model.model_digest
    assert loaded.score(query) == pytest.approx(scores)


def test_krr_exact_expected_kernel_and_b2_episode_score_mean(tmp_path) -> None:
    train = {
        "a": _bank([-4.1, -3.9]),
        "b": _bank([-0.1, 0.1]),
        "c": _bank([3.9, 4.1]),
    }
    labels = {"a": "class-a", "b": "class-b", "c": "class-c"}
    validation = {"a": _bank([-4.0]), "b": _bank([0.0]), "c": _bank([4.0])}
    bandwidth = 1.5
    model = KMEKRR.fit(
        train,
        labels,
        validation,
        bandwidth=bandwidth,
        ridge_grid=(1.0e-4, 1.0e-2, 1.0),
    )
    kernel = GaussianKernel(bandwidth)
    episodes = [train[source_id].episode(0) for source_id in sorted(train)]
    gram = np.asarray(
        [[np.mean(kernel.gram(left, right)) for right in episodes] for left in episodes]
    )
    expected_alpha = np.linalg.solve(gram + model.selected_ridge * np.eye(3), np.eye(3))
    np.testing.assert_allclose(model.alpha, expected_alpha, rtol=1.0e-13, atol=1.0e-13)

    query = _b2_query()
    episode_scores = model.episode_scores(query)
    scores = model.score(query)
    np.testing.assert_allclose(
        [scores[class_id] for class_id in model.class_ids],
        np.mean(episode_scores, axis=0),
    )
    merged = EpisodeBank(query.points, np.array([0, len(query.points)]))
    assert not np.allclose(
        np.mean(episode_scores, axis=0), model.episode_scores(merged)[0]
    )

    path = tmp_path / "krr.npz"
    model.save_npz(path)
    loaded = KMEKRR.load_npz(path)
    assert loaded.model_digest == model.model_digest
    assert loaded.score(query) == pytest.approx(scores)
