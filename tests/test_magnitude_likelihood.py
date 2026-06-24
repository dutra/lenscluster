import jax.numpy as jnp
import pytest

from lenscluster.cluster_solver import _family_magnitude_loglike


def test_family_magnitude_loglike_prefers_magnification_corrected_consistency():
    family_idx = jnp.array([0, 0, 0], dtype=jnp.int32)
    image_has_constraint = jnp.array([True, True, True])
    reliability = jnp.ones(3)
    jacobian_entries = (jnp.ones(3), jnp.zeros(3), jnp.zeros(3), jnp.ones(3))

    consistent = _family_magnitude_loglike(
        jnp.array([24.0, 24.05, 23.95]),
        None,
        reliability,
        image_has_constraint,
        family_idx,
        1,
        *jacobian_entries,
    )
    inconsistent = _family_magnitude_loglike(
        jnp.array([24.0, 25.5, 22.5]),
        None,
        reliability,
        image_has_constraint,
        family_idx,
        1,
        *jacobian_entries,
    )

    assert consistent > inconsistent


def test_family_magnitude_loglike_skips_single_image_families():
    family_idx = jnp.array([0], dtype=jnp.int32)
    image_has_constraint = jnp.array([False])
    reliability = jnp.ones(1)
    jacobian_entries = (jnp.ones(1), jnp.zeros(1), jnp.zeros(1), jnp.ones(1))

    value = _family_magnitude_loglike(
        jnp.array([24.0]),
        None,
        reliability,
        image_has_constraint,
        family_idx,
        1,
        *jacobian_entries,
    )

    assert float(value) == 0.0


def test_family_magnitude_loglike_treats_bands_independently():
    family_idx = jnp.array([0, 0, 0], dtype=jnp.int32)
    image_has_constraint = jnp.array([True, True, True])
    reliability = jnp.ones(3)
    jacobian_entries = (jnp.ones(3), jnp.zeros(3), jnp.zeros(3), jnp.ones(3))

    consistent_colors = _family_magnitude_loglike(
        jnp.array(
            [
                [24.0, 27.0],
                [24.05, 27.05],
                [23.95, 26.95],
            ]
        ),
        None,
        reliability,
        image_has_constraint,
        family_idx,
        1,
        *jacobian_entries,
    )
    inconsistent_colors = _family_magnitude_loglike(
        jnp.array(
            [
                [24.0, 27.0],
                [25.5, 27.05],
                [22.5, 26.95],
            ]
        ),
        None,
        reliability,
        image_has_constraint,
        family_idx,
        1,
        *jacobian_entries,
    )

    assert consistent_colors > inconsistent_colors


def test_family_magnitude_loglike_normalizes_repeated_bands():
    family_idx = jnp.array([0, 0, 0], dtype=jnp.int32)
    image_has_constraint = jnp.array([True, True, True])
    reliability = jnp.ones(3)
    jacobian_entries = (jnp.ones(3), jnp.zeros(3), jnp.zeros(3), jnp.ones(3))
    single_band = jnp.array([24.0, 25.5, 22.5])
    repeated_bands = jnp.repeat(single_band[:, None], 7, axis=1)

    single_value = _family_magnitude_loglike(
        single_band,
        None,
        reliability,
        image_has_constraint,
        family_idx,
        1,
        *jacobian_entries,
    )
    repeated_value = _family_magnitude_loglike(
        repeated_bands,
        None,
        reliability,
        image_has_constraint,
        family_idx,
        1,
        *jacobian_entries,
    )

    assert float(repeated_value) == pytest.approx(float(single_value))


def test_family_magnitude_loglike_uses_arc_gated_scatter():
    family_idx = jnp.array([0, 0, 0], dtype=jnp.int32)
    image_has_constraint = jnp.array([True, True, True])
    reliability = jnp.ones(3)
    magnitudes = jnp.array([24.0, 25.5, 22.5])
    jacobian_entries = (jnp.ones(3), jnp.zeros(3), jnp.zeros(3), jnp.ones(3))
    singular_min = jnp.array([0.01, 0.01, 0.01])

    base_only = _family_magnitude_loglike(
        magnitudes,
        None,
        reliability,
        image_has_constraint,
        family_idx,
        1,
        *jacobian_entries,
        magnitude_base_scatter=0.05,
        magnitude_arc_scatter=0.05,
        singular_min_precomputed=singular_min,
        singular_threshold=0.05,
        singular_softness=0.01,
    )
    arc_broadened = _family_magnitude_loglike(
        magnitudes,
        None,
        reliability,
        image_has_constraint,
        family_idx,
        1,
        *jacobian_entries,
        magnitude_base_scatter=0.05,
        magnitude_arc_scatter=1.0,
        singular_min_precomputed=singular_min,
        singular_threshold=0.05,
        singular_softness=0.01,
    )

    assert arc_broadened > base_only
