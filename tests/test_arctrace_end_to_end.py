import math

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from arctrace.config import ArcMeasureConfig
from arctrace.geometry import axial_difference
from arctrace.measure import measure_arc
from arctrace.mosaics import load_band_mosaic
from arctrace_synth import write_synthetic_arc_fits


@pytest.mark.slow
def test_measure_arc_recovers_truth_on_rotated_wcs(tmp_path) -> None:
    path = tmp_path / "synthetic_f814w.fits"
    truth = write_synthetic_arc_fits(
        path,
        rotation_deg=25.0,
        radius_arcsec=8.0,
        half_span_rad=0.6,
        width_arcsec=0.35,
        peak=2.0,
        noise_sigma=0.05,
        rng=np.random.default_rng(42),
    )
    mosaic = load_band_mosaic("F814W", path)
    seed = SkyCoord(ra=truth["anchor_ra_deg"] * u.deg, dec=truth["anchor_dec_deg"] * u.deg)
    cfg = ArcMeasureConfig(cutout_size_arcsec=22.0, n_bootstrap=120)
    measurement = measure_arc({"F814W": mosaic}, seed, cfg, label="1.a")

    assert measurement.success, measurement.failure_reason
    assert measurement.sigma_tangent_rad > 0.0
    assert measurement.sigma_curvature_arcsec_inv > 0.0

    dphi = axial_difference(measurement.tangent_angle_offset_rad, truth["tangent_angle_offset_rad"])
    assert abs(dphi) < 3.0 * measurement.sigma_tangent_rad

    dkappa = measurement.curvature_arcsec_inv - truth["curvature_arcsec_inv"]
    assert abs(dkappa) < 3.0 * measurement.sigma_curvature_arcsec_inv

    # Anchor should land on the painted arc.
    anchor = SkyCoord(ra=measurement.anchor_ra_deg * u.deg, dec=measurement.anchor_dec_deg * u.deg)
    separation = anchor.separation(seed).arcsec
    assert separation < 1.0

    # Curvature-center side: the painted center is the WCS reference point.
    reference = measurement.bands[0]
    assert reference.center_ra_deg is not None
    center = SkyCoord(ra=reference.center_ra_deg * u.deg, dec=reference.center_dec_deg * u.deg)
    center_truth = SkyCoord(ra=truth["ra0_deg"] * u.deg, dec=truth["dec0_deg"] * u.deg)
    assert center.separation(center_truth).arcsec < 1.5
    assert reference.radius_arcsec == pytest.approx(8.0, abs=1.0)
    assert reference.length_arcsec == pytest.approx(truth["length_arcsec"], rel=0.25)


@pytest.mark.slow
def test_forward_refinement_consistent_on_short_arc(tmp_path) -> None:
    path = tmp_path / "synthetic_short.fits"
    truth = write_synthetic_arc_fits(
        path,
        radius_arcsec=3.0,
        half_span_rad=0.35,  # length ~2.1 arcsec
        width_arcsec=0.3,
        peak=2.5,
        noise_sigma=0.05,
        rng=np.random.default_rng(7),
    )
    mosaic = load_band_mosaic("F814W", path)
    seed = SkyCoord(ra=truth["anchor_ra_deg"] * u.deg, dec=truth["anchor_dec_deg"] * u.deg)

    geo = measure_arc({"F814W": mosaic}, seed, ArcMeasureConfig(n_bootstrap=80), label="1.a")
    fwd = measure_arc(
        {"F814W": mosaic}, seed, ArcMeasureConfig(n_bootstrap=80, refine="forward"), label="1.a"
    )
    assert geo.success and fwd.success
    assert fwd.bands[0].refine_applied

    kappa_true = truth["curvature_arcsec_inv"]
    assert abs(fwd.curvature_arcsec_inv - kappa_true) < 3.0 * fwd.sigma_curvature_arcsec_inv
    assert abs(axial_difference(fwd.tangent_angle_offset_rad, truth["tangent_angle_offset_rad"])) < (
        3.0 * fwd.sigma_tangent_rad
    )
    # Refinement must not be wildly off the geometric result.
    assert abs(fwd.curvature_arcsec_inv - geo.curvature_arcsec_inv) < 0.2 * kappa_true + 3.0 * (
        geo.sigma_curvature_arcsec_inv + fwd.sigma_curvature_arcsec_inv
    )


@pytest.mark.slow
def test_cli_measure_end_to_end(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from arctrace.cli import main
    from lenscluster.lenstool_parser import _load_arc_constraints_catalog

    fits_path = tmp_path / "synthetic_f814w.fits"
    truth = write_synthetic_arc_fits(fits_path, rng=np.random.default_rng(1))
    regions_path = tmp_path / "seeds.reg"
    regions_path.write_text(
        "fk5\n"
        f"point({truth['anchor_ra_deg']:.7f},{truth['anchor_dec_deg']:.7f}) # text={{1.a}}\n"
        f"point({truth['ra0_deg'] + 8.0 / 3600.0:.7f},{truth['dec0_deg'] - 8.0 / 3600.0:.7f}) # text={{9.z}}\n"
    )
    output_dir = tmp_path / "out"
    exit_code = main(
        [
            "measure",
            "--image",
            f"F814W={fits_path}",
            "--regions",
            str(regions_path),
            "--output-dir",
            str(output_dir),
            "--n-bootstrap",
            "60",
        ]
    )
    assert exit_code == 0

    arcfile = output_dir / "arcfile.cat"
    assert arcfile.exists()
    df = _load_arc_constraints_catalog(arcfile, None)
    assert list(df["arc_id"]) == ["1.a"]
    assert list(df["z_arc"]) == [-1.0]
    assert float(df.iloc[0]["arc_curvature_arcsec_inv"]) == pytest.approx(
        truth["curvature_arcsec_inv"], rel=0.5
    )

    sidetable = output_dir / "arctrace_sidetable.csv"
    assert sidetable.exists()
    import pandas as pd

    side = pd.read_csv(sidetable)
    assert set(side["label"]) == {"1.a", "9.z"}
    assert bool(side.loc[side["label"] == "1.a", "success"].iloc[0])
    assert not bool(side.loc[side["label"] == "9.z", "success"].iloc[0])

    assert (output_dir / "qa" / "1_a.png").exists()
    assert (output_dir / "arctrace_run.json").exists()


@pytest.mark.slow
def test_method_floors_set_realistic_sigma(tmp_path) -> None:
    # A long, well-sampled arc yields a tiny bootstrap formal sigma; the
    # irreducible method floor must keep the reported sigma realistic so the
    # solver is not handed an over-tight constraint.
    path = tmp_path / "synthetic_long.fits"
    truth = write_synthetic_arc_fits(
        path,
        radius_arcsec=10.0,
        half_span_rad=0.9,
        width_arcsec=0.3,
        peak=3.0,
        noise_sigma=0.03,
        rng=np.random.default_rng(13),
    )
    mosaic = load_band_mosaic("F814W", path)
    seed = SkyCoord(ra=truth["anchor_ra_deg"] * u.deg, dec=truth["anchor_dec_deg"] * u.deg)
    cfg = ArcMeasureConfig(cutout_size_arcsec=26.0, n_bootstrap=120)
    measurement = measure_arc({"F814W": mosaic}, seed, cfg, label="1.a")
    assert measurement.success
    # Reported tangent sigma is at least the method floor; bootstrap-only would
    # be far smaller for ~100 ridge points.
    assert measurement.sigma_tangent_rad >= cfg.tangent_method_floor_rad
    reference = measurement.bands[0]
    assert reference.sigma_tangent_stat_rad < cfg.tangent_method_floor_rad
    # Curvature sigma respects the fractional floor.
    assert measurement.sigma_curvature_arcsec_inv >= (
        cfg.curvature_method_floor_frac * measurement.curvature_arcsec_inv * 0.99
    )


def test_blank_seed_returns_failure(tmp_path) -> None:
    path = tmp_path / "synthetic_blank.fits"
    truth = write_synthetic_arc_fits(path, rng=np.random.default_rng(3))
    mosaic = load_band_mosaic("F814W", path)
    # Seed well away from the arc but inside the mosaic.
    seed = SkyCoord(
        ra=(truth["ra0_deg"] + 8.0 / 3600.0) * u.deg,
        dec=(truth["dec0_deg"] - 8.0 / 3600.0) * u.deg,
    )
    measurement = measure_arc({"F814W": mosaic}, seed, ArcMeasureConfig(), label="9.z")
    assert not measurement.success
    assert measurement.failure_reason
