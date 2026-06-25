import importlib.util
from argparse import Namespace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS

from lenscluster.cluster_solver import _family_magnitude_loglike
from lenscluster.image_tools.photometry import read_lenstool_image_catalog
from lenscluster.image_tools.truth_magnitudes import (
    TruthMagnitudeConfig,
    _effective_abs_magnification,
    _lensing_efficiency,
)


def _load_truth_magnitude_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_ff_sims_truth_magnitudes.py"
    spec = importlib.util.spec_from_file_location("make_ff_sims_truth_magnitudes_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_truth_magnitude_lensing_efficiency_uses_astropy8_two_redshift_distance():
    cosmo = FlatLambdaCDM(H0=72.0, Om0=0.24, Tcmb0=2.725)
    z_lens = 0.507
    z_source = np.asarray([0.1, z_lens, 1.0, 9.0])

    efficiency = _lensing_efficiency(cosmo, z_lens, z_source)

    assert efficiency[0] == pytest.approx(0.0)
    assert efficiency[1] == pytest.approx(0.0)
    assert np.all(np.isfinite(efficiency[2:]))
    assert np.all(efficiency[2:] > 0.0)
    expected = (
        cosmo.angular_diameter_distance(z_lens, z_source[2:]).value
        / cosmo.angular_diameter_distance(z_source[2:]).value
    )
    np.testing.assert_allclose(efficiency[2:], expected)


def test_ff_sims_truth_magnitude_script_cosmology_matches_run_config():
    module = _load_truth_magnitude_script_module()

    assert module.CLUSTER_COSMOLOGY["ares"] == {"h0": 70.4, "om0": 0.272, "z_lens": 0.5}
    assert module.CLUSTER_COSMOLOGY["hera"] == {"h0": 72.0, "om0": 0.24, "z_lens": 0.507}
    assert module.CLUSTER_COSMOLOGY["hera"] != module.CLUSTER_COSMOLOGY["ares"]


def test_ff_sims_truth_magnitude_generation_rewrites_solver_catalog(tmp_path):
    module = _load_truth_magnitude_script_module()
    data_root = tmp_path / "ff_sims"
    cluster_dir = data_root / "ares"
    map_dir = data_root / "published" / "ares"
    cluster_dir.mkdir(parents=True)
    map_dir.mkdir(parents=True)
    catalog_path = cluster_dir / "ares_obs_arcs.cat"
    catalog_path.write_text(
        "\n".join(
            [
                "#REFERENCE 3",
                "1.a 0.00000000 0.00000000 0.3734 0.3734 90.0 1.00000000 25.000000",
                "1.b 0.50000000 0.00000000 0.3734 0.3734 90.0 1.00000000 25.000000",
                "2.a -0.50000000 0.00000000 0.3734 0.3734 90.0 2.00000000 25.000000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [3.0, 3.0]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    header = wcs.to_header()
    fits.writeto(map_dir / "kappa_z9_0.fits", np.full((5, 5), 0.2), header=header)
    fits.writeto(map_dir / "gammax_z9_0.fits", np.full((5, 5), 0.05), header=header)
    fits.writeto(map_dir / "gammay_z9_0.fits", np.full((5, 5), 0.02), header=header)

    module._process_cluster(
        "ares",
        data_root,
        Namespace(
            random_seed=12345,
            source_mag_f160w_mean=27.0,
            source_mag_f160w_sigma=0.2,
            color_scatter_sigma=0.0,
            magnitude_error=0.01,
            mu_floor=1.0e-3,
            mu_max=50.0,
            mu_aperture_radius_arcsec=0.0,
        ),
    )

    non_comment_lines = [
        line.split()
        for line in catalog_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert all(len(parts) == 10 for parts in non_comment_lines)
    _reference, catalog = read_lenstool_image_catalog(catalog_path)
    assert np.all(np.isfinite(catalog["catalog_mag"]))
    assert not np.allclose(catalog["catalog_mag"].to_numpy(dtype=float), 25.0)
    assert np.all(np.isfinite(catalog["catalog_mag_err"]))
    assert np.all(catalog["catalog_mag_err"] > 0.0)
    assert (cluster_dir / "ares_obs_arcs_truthmag_band_magnitudes.csv").is_file()
    assert not (cluster_dir / "ares_obs_arcs_truthmag.cat").exists()
