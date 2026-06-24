import jax.numpy as jnp
import numpy as np
import pytest
from astropy.wcs import WCS

from lenscluster.cluster_solver import _family_magnitude_loglike
from lenscluster.image_tools.truth_magnitudes import TruthMagnitudeConfig, _effective_abs_magnification


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


def test_family_magnitude_loglike_uses_arc_bias_to_repair_systematic_arc_offset():
    family_idx = jnp.array([0, 0, 0], dtype=jnp.int32)
    image_has_constraint = jnp.array([True, True, True])
    reliability = jnp.ones(3)
    magnitudes = jnp.array([24.0, 24.05, 24.50])
    jacobian_entries = (jnp.ones(3), jnp.zeros(3), jnp.zeros(3), jnp.ones(3))
    singular_min = jnp.array([1.0, 1.0, 0.01])

    no_bias = _family_magnitude_loglike(
        magnitudes,
        None,
        reliability,
        image_has_constraint,
        family_idx,
        1,
        *jacobian_entries,
        magnitude_base_scatter=0.05,
        magnitude_arc_scatter=0.10,
        magnitude_arc_bias=0.0,
        singular_min_precomputed=singular_min,
        singular_threshold=0.05,
        singular_softness=0.01,
    )
    repaired = _family_magnitude_loglike(
        magnitudes,
        None,
        reliability,
        image_has_constraint,
        family_idx,
        1,
        *jacobian_entries,
        magnitude_base_scatter=0.05,
        magnitude_arc_scatter=0.10,
        magnitude_arc_bias=0.50,
        singular_min_precomputed=singular_min,
        singular_threshold=0.05,
        singular_softness=0.01,
    )

    assert repaired > no_bias


def test_truth_magnitude_uses_capped_aperture_average_for_near_critical_pixel():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [3.0, 3.0]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]

    kappa = np.full((5, 5), 0.2)
    gamma_x = np.zeros((5, 5))
    gamma_y = np.zeros((5, 5))
    kappa[2, 2] = 0.999999

    config = TruthMagnitudeConfig(
        mu_floor=1.0e-3,
        mu_max=50.0,
        mu_aperture_radius_arcsec=1.0,
    )
    effective = _effective_abs_magnification(
        kappa,
        gamma_x,
        gamma_y,
        np.asarray([2.0]),
        np.asarray([2.0]),
        np.asarray([1.0]),
        wcs=wcs,
        config=config,
    )

    assert float(effective[0]) < 50.0
    assert float(effective[0]) > 1.0
