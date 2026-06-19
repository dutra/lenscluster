from __future__ import annotations

import numpy as np
import pandas as pd

from lenscluster.family_partition import (
    build_pair_table,
    canonicalize_partition,
    create_anchor_labels,
    family_catalog_partition_loglike,
    feature_matrix,
    fit_logistic_map,
    generate_candidate_partitions,
    repair_candidate_partitions,
    run_family_partition_engine,
    same_family_matrix,
    score_partitions,
    weighted_pair_probability,
)
from lenscluster.source_plane_partition import (
    cached_lens_cluster_partitions_callback,
    cached_lens_proposal_matrix_callback,
    cached_source_plane_score_callback,
    lens_pair_affinity_from_cache,
    marginalized_source_plane_score_callback_from_ray_shooters,
    precompute_source_plane_cache,
    score_source_plane_partitions,
    source_plane_score_callback_from_ray_shooter,
)


def _catalog() -> pd.DataFrame:
    rows = [
        {
            "object_id": "a1",
            "family_id": "A",
            "ra": 10.00000,
            "dec": 0.00000,
            "zspec_best": 2.0,
            "zspec_best_confidence": "secure",
            "zspec_best_confidence_rank": 4,
            "mag_F606W": 24.0,
            "mag_F814W": 23.7,
            "mag_F160W": 23.2,
            "image_size_arcsec": 0.25,
            "image_ellipticity": 0.2,
            "family_reliability": 0.95,
            "object_source": "test",
        },
        {
            "object_id": "a2",
            "family_id": "A",
            "ra": 10.00025,
            "dec": 0.00000,
            "zspec_best": 2.01,
            "zspec_best_confidence": "secure",
            "zspec_best_confidence_rank": 4,
            "mag_F606W": 25.0,
            "mag_F814W": 24.7,
            "mag_F160W": 24.2,
            "image_size_arcsec": 0.28,
            "image_ellipticity": 0.22,
            "family_reliability": 0.90,
            "object_source": "test",
        },
        {
            "object_id": "b1",
            "family_id": "B",
            "ra": 10.01000,
            "dec": 0.00000,
            "zspec_best": 3.2,
            "zspec_best_confidence": "secure",
            "zspec_best_confidence_rank": 4,
            "mag_F606W": 23.2,
            "mag_F814W": 23.1,
            "mag_F160W": 23.0,
            "image_size_arcsec": 0.40,
            "image_ellipticity": 0.5,
            "family_reliability": 0.85,
            "object_source": "test",
        },
        {
            "object_id": "b2",
            "family_id": "B",
            "ra": 10.01025,
            "dec": 0.00000,
            "zspec_best": 3.19,
            "zspec_best_confidence": "secure",
            "zspec_best_confidence_rank": 4,
            "mag_F606W": 24.2,
            "mag_F814W": 24.1,
            "mag_F160W": 24.0,
            "image_size_arcsec": 0.43,
            "image_ellipticity": 0.48,
            "family_reliability": 0.88,
            "object_source": "test",
        },
    ]
    return pd.DataFrame(rows)


def test_partition_canonicalization_and_pair_probability_conservation() -> None:
    assignment = canonicalize_partition([7, 7, 2, 9, 2])

    np.testing.assert_array_equal(assignment, np.asarray([0, 0, 1, 2, 1]))
    np.testing.assert_array_equal(np.diag(same_family_matrix(assignment)), np.ones(5, dtype=bool))

    partitions = np.asarray([[0, 0, 1], [0, 1, 1]], dtype=int)
    weights = np.asarray([0.25, 0.75], dtype=float)
    pair_probability = weighted_pair_probability(partitions, weights)

    np.testing.assert_allclose(np.diag(pair_probability), np.ones(3))
    np.testing.assert_allclose(pair_probability[0, 1], 0.25)
    np.testing.assert_allclose(pair_probability[1, 2], 0.75)


def test_logistic_fit_recovers_sensible_feature_signs() -> None:
    x = np.asarray(
        [
            [1.0, 0.95, 0.90],
            [1.0, 0.90, 0.85],
            [1.0, 0.10, 0.20],
            [1.0, 0.20, 0.10],
        ],
        dtype=float,
    )
    y = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=float)

    fit = fit_logistic_map(
        x,
        y,
        sample_weight=np.ones(4),
        gaussian_prior_sigma=5.0,
        feature_names=["intercept", "z_similarity", "color_similarity"],
    )

    assert fit.success
    assert fit.coefficients[1] > 0.0
    assert fit.coefficients[2] > 0.0


def test_anchor_labels_dominate_incompatible_redshift_pairs() -> None:
    pairs = build_pair_table(_catalog())
    labels, weights = create_anchor_labels(pairs, incompatible_z_delta=0.5)
    incompatible = pairs["z_delta"].to_numpy(dtype=float) > 0.5

    assert np.any(incompatible)
    np.testing.assert_array_equal(labels[incompatible], np.zeros(int(np.sum(incompatible))))
    assert np.all(weights[incompatible] >= 12.0)


def test_candidate_generation_returns_valid_weighted_partitions() -> None:
    pairs = build_pair_table(_catalog())
    x, names = feature_matrix(pairs)
    weights = {name: 0.0 for name in names}
    weights.update({"intercept": -2.0, "z_similarity": 3.0, "color_similarity": 2.0, "radius_similarity": 0.5})
    pair_values = 1.0 / (1.0 + np.exp(-(x @ np.asarray([weights[name] for name in names], dtype=float))))
    matrix = np.eye(4)
    for value, row in zip(pair_values, pairs.itertuples(index=False)):
        matrix[int(row.left_index), int(row.right_index)] = value
        matrix[int(row.right_index), int(row.left_index)] = value

    partitions = generate_candidate_partitions(
        pairs,
        matrix,
        n_images=4,
        n_thresholds=8,
        n_noisy_partitions=8,
        rng=np.random.default_rng(5),
    )

    assert partitions.ndim == 2
    assert partitions.shape[1] == 4
    for assignment in partitions:
        np.testing.assert_array_equal(assignment, canonicalize_partition(assignment))


def test_candidate_generation_includes_correlation_clustering_partition() -> None:
    pairs = build_pair_table(_catalog())
    matrix = np.full((4, 4), 0.05, dtype=float)
    np.fill_diagonal(matrix, 1.0)
    matrix[0, 1] = matrix[1, 0] = 0.9
    matrix[2, 3] = matrix[3, 2] = 0.85

    partitions = generate_candidate_partitions(
        pairs,
        matrix,
        n_images=4,
        n_thresholds=4,
        n_noisy_partitions=0,
        rng=np.random.default_rng(11),
    )

    assert any(np.array_equal(canonicalize_partition(row), np.asarray([0, 0, 1, 1])) for row in partitions)


def test_em_engine_preserves_simple_two_family_solution() -> None:
    result = run_family_partition_engine(
        _catalog(),
        random_seed=123,
        n_iterations=3,
        n_thresholds=10,
        n_noisy_partitions=25,
        gaussian_prior_sigma=3.0,
    )

    assert np.isclose(float(np.sum(result.weights)), 1.0)
    assert result.pair_probability.shape == (4, 4)
    assert result.pair_probability[0, 1] > 0.95
    assert result.pair_probability[2, 3] > 0.95
    assert result.pair_probability[0, 2] < 0.05
    np.testing.assert_array_equal(result.map_assignment, np.asarray([0, 0, 1, 1]))


def test_partition_temperature_softens_weights() -> None:
    cold = run_family_partition_engine(
        _catalog(),
        random_seed=123,
        n_iterations=1,
        n_thresholds=10,
        n_noisy_partitions=25,
        gaussian_prior_sigma=3.0,
        partition_temperature=1.0,
    )
    warm = run_family_partition_engine(
        _catalog(),
        random_seed=123,
        n_iterations=1,
        n_thresholds=10,
        n_noisy_partitions=25,
        gaussian_prior_sigma=3.0,
        partition_temperature=3.0,
    )

    assert np.isclose(float(np.sum(warm.weights)), 1.0)
    assert warm.partitions.shape == cold.partitions.shape
    np.testing.assert_array_equal(warm.map_assignment, cold.map_assignment)
    cold_ess = 1.0 / float(np.sum(np.square(cold.weights)))
    warm_ess = 1.0 / float(np.sum(np.square(warm.weights)))
    assert warm_ess >= cold_ess


def test_block_normalized_score_reduces_large_family_pair_bonus() -> None:
    pair_table = pd.DataFrame(
        [
            {"left_index": left, "right_index": right}
            for left in range(4)
            for right in range(left + 1, 4)
        ]
    )
    pair_probability = np.full((4, 4), 0.8, dtype=float)
    np.fill_diagonal(pair_probability, 1.0)
    truth_like = np.asarray([0, 0, 1, 1], dtype=int)
    large_family = np.asarray([0, 0, 0, 1], dtype=int)
    partitions = np.asarray([truth_like, large_family], dtype=int)

    summed = score_partitions(partitions, pair_table, pair_probability, pair_score_mode="sum")
    normalized = score_partitions(partitions, pair_table, pair_probability, pair_score_mode="block_normalized")

    summed_large_family_advantage = float(summed[1] - summed[0])
    normalized_large_family_advantage = float(normalized[1] - normalized[0])

    assert normalized_large_family_advantage < summed_large_family_advantage


def test_family_catalog_loglike_prefers_consistent_families() -> None:
    partitions = np.asarray(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
        ],
        dtype=int,
    )

    scores = family_catalog_partition_loglike(
        partitions,
        _catalog(),
        redshift_weight=1.0,
        color_weight=1.0,
        morphology_weight=0.2,
        color_sigma=0.2,
    )

    assert scores[0] > scores[1]

    likelihood_ratio_scores = family_catalog_partition_loglike(
        partitions,
        _catalog(),
        redshift_weight=1.0,
        color_weight=1.0,
        morphology_weight=0.2,
        color_sigma=0.2,
        score_mode="likelihood_ratio",
    )

    assert likelihood_ratio_scores[0] > likelihood_ratio_scores[1]


def test_partition_repair_can_reattach_singleton() -> None:
    pair_table = pd.DataFrame(
        [
            {"left_index": left, "right_index": right}
            for left in range(4)
            for right in range(left + 1, 4)
        ]
    )
    pair_probability = np.full((4, 4), 0.05, dtype=float)
    np.fill_diagonal(pair_probability, 1.0)
    pair_probability[0, 1] = pair_probability[1, 0] = 0.95
    pair_probability[2, 3] = pair_probability[3, 2] = 0.95
    broken = np.asarray([[0, 0, 1, 2]], dtype=int)
    scores = score_partitions(broken, pair_table, pair_probability)

    repaired, _repair_scores = repair_candidate_partitions(
        broken,
        scores,
        pair_table,
        pair_probability,
        anchor_labels=np.full(len(pair_table), np.nan),
        anchor_weights=np.zeros(len(pair_table)),
        pair_score_mode="sum",
        top_k=1,
        max_rounds=1,
    )

    assert any(np.array_equal(canonicalize_partition(row), np.asarray([0, 0, 1, 1])) for row in repaired)


def test_source_plane_score_prefers_common_source_partition() -> None:
    partitions = np.asarray(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
        ],
        dtype=int,
    )
    beta_x = np.asarray(
        [
            [0.0, 0.02, 1.0, 1.02],
            [0.0, 0.02, 1.0, 1.02],
        ],
        dtype=float,
    )
    beta_y = np.zeros_like(beta_x)

    result = score_source_plane_partitions(
        partitions,
        beta_x,
        beta_y,
        sigma_arcsec=0.05,
        source_scatter_arcsec=0.02,
    )

    assert result.log_scores[0] > result.log_scores[1]
    assert not result.family_table.empty


def test_source_plane_score_callback_uses_candidate_family_redshift() -> None:
    images = pd.DataFrame(
        {
            "object_id": ["a1", "a2", "b1", "b2"],
            "x_obs": [0.0, 0.02, 1.0, 1.02],
            "y_obs": [0.0, 0.0, 0.0, 0.0],
            "catalog_z": [2.0, 2.0, 3.0, 3.0],
            "catalog_z_sigma": [0.1, 0.1, 0.1, 0.1],
        }
    )

    def ray_shooter(x: np.ndarray, y: np.ndarray, z_source: float) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(x, dtype=float) - (0.0 if z_source < 2.5 else 1.0), np.asarray(y, dtype=float)

    callback = source_plane_score_callback_from_ray_shooter(
        images,
        ray_shooter,
        position_sigma_arcsec=0.05,
        source_scatter_arcsec=0.02,
    )
    partitions = np.asarray([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=int)
    scores = callback(partitions, build_pair_table(images))

    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_marginalized_source_plane_callback_combines_ray_shooters() -> None:
    images = pd.DataFrame(
        {
            "object_id": ["a1", "a2", "b1", "b2"],
            "x_obs": [0.0, 0.02, 1.0, 1.02],
            "y_obs": [0.0, 0.0, 0.0, 0.0],
            "catalog_z": [2.0, 2.0, 3.0, 3.0],
            "catalog_z_sigma": [0.1, 0.1, 0.1, 0.1],
        }
    )

    def ray_shooter_a(x: np.ndarray, y: np.ndarray, z_source: float) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(x, dtype=float) - (0.0 if z_source < 2.5 else 1.0), np.asarray(y, dtype=float)

    def ray_shooter_b(x: np.ndarray, y: np.ndarray, z_source: float) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(x, dtype=float) - (0.01 if z_source < 2.5 else 1.01), np.asarray(y, dtype=float)

    callback = marginalized_source_plane_score_callback_from_ray_shooters(
        images,
        [ray_shooter_a, ray_shooter_b],
        position_sigma_arcsec=0.05,
        source_scatter_arcsec=0.02,
    )
    scores = callback(np.asarray([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=int), build_pair_table(images))

    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_cached_redshift_marginal_lens_affinity_guides_proposals() -> None:
    images = pd.DataFrame(
        {
            "object_id": ["a1", "a2", "b1", "b2"],
            "x_obs": [0.0, 0.02, 1.0, 1.02],
            "y_obs": [0.0, 0.0, 0.0, 0.0],
            "catalog_z": [2.0, 2.0, 3.0, 3.0],
            "catalog_z_sigma": [0.1, 0.1, 0.1, 0.1],
        }
    )

    def ray_shooter(x: np.ndarray, y: np.ndarray, z_source: float) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(x, dtype=float) - (0.0 if z_source < 2.5 else 1.0), np.asarray(y, dtype=float)

    cache = precompute_source_plane_cache(
        images,
        ray_shooter,
        z_grid=np.asarray([2.0, 3.0], dtype=float),
        position_sigma_arcsec=0.05,
    )
    affinity = lens_pair_affinity_from_cache(cache, source_scatter_arcsec=0.02)

    assert affinity[0, 1] > 0.9
    assert affinity[2, 3] > 0.9
    assert affinity[0, 2] < 0.1

    score_callback = cached_source_plane_score_callback(cache, source_scatter_arcsec=0.02)
    partitions = np.asarray([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=int)
    scores = score_callback(partitions, build_pair_table(images))
    assert scores[0] > scores[1]
    bf_score_callback = cached_source_plane_score_callback(
        cache,
        source_scatter_arcsec=0.02,
        score_mode="likelihood_ratio",
    )
    bf_scores = bf_score_callback(partitions, build_pair_table(images))
    assert bf_scores[0] > bf_scores[1]

    proposal_callback = cached_lens_proposal_matrix_callback(affinity, lens_weight=1.0)
    catalog = np.full((4, 4), 0.5, dtype=float)
    np.fill_diagonal(catalog, 1.0)
    matrices = proposal_callback(catalog, build_pair_table(images))
    assert len(matrices) == 2
    assert matrices[1][0, 1] > matrices[1][0, 2]


def test_lens_first_cluster_proposals_include_source_plane_partition() -> None:
    images = pd.DataFrame(
        {
            "object_id": ["a1", "a2", "b1", "b2"],
            "x_obs": [0.0, 0.02, 1.0, 1.02],
            "y_obs": [0.0, 0.0, 0.0, 0.0],
            "catalog_z": [2.0, 2.0, 3.0, 3.0],
            "catalog_z_sigma": [0.1, 0.1, 0.1, 0.1],
        }
    )

    def ray_shooter(x: np.ndarray, y: np.ndarray, z_source: float) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(x, dtype=float) - (0.0 if z_source < 2.5 else 1.0), np.asarray(y, dtype=float)

    cache = precompute_source_plane_cache(
        images,
        ray_shooter,
        z_grid=np.asarray([2.0, 3.0], dtype=float),
        position_sigma_arcsec=0.05,
    )
    callback = cached_lens_cluster_partitions_callback(
        cache,
        beta_radius_grid=(0.05,),
        min_cluster_size=2,
        min_redshift_weight=0.05,
        source_scatter_arcsec=0.02,
        beam_width=10,
    )
    catalog = np.full((4, 4), 0.1, dtype=float)
    catalog[0, 1] = catalog[1, 0] = 0.9
    catalog[2, 3] = catalog[3, 2] = 0.9
    np.fill_diagonal(catalog, 1.0)
    partitions = callback(catalog, build_pair_table(images))

    assert any(np.array_equal(canonicalize_partition(row), np.asarray([0, 0, 1, 1])) for row in partitions)


def test_missing_optional_columns_degrade_gracefully() -> None:
    images = pd.DataFrame(
        {
            "object_id": ["x1", "x2", "x3"],
            "ra": [10.0, 10.001, 10.02],
            "dec": [0.0, 0.0, 0.0],
        }
    )

    result = run_family_partition_engine(images, random_seed=7, n_iterations=2, n_thresholds=5, n_noisy_partitions=5)

    assert result.partitions.shape[1] == 3
    assert np.isclose(float(np.sum(result.weights)), 1.0)
    assert np.isfinite(result.pair_table["model_pair_probability"]).all()
