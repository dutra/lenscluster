import math

import numpy as np
import pytest

from arctrace.arcfile import measurements_to_dataframe, write_arcfile, write_sidetable
from arctrace.errors import ArcfileWriteError
from arctrace.measure import ArcMeasurement, BandArcMeasurement
from lenscluster.lenstool_parser import _load_arc_constraints_catalog


def _measurement(label: str, *, ra=39.9712345, dec=-1.5823456, angle=1.2345678,
                 kappa=0.025, sig_phi=0.045, sig_kappa=0.006, reliability=0.85,
                 success=True) -> ArcMeasurement:
    band = BandArcMeasurement(
        band="F814W",
        success=success,
        failure_reason=None if success else "synthetic failure",
        tangent_angle_offset_rad=angle,
        curvature_arcsec_inv=kappa,
        sigma_tangent_stat_rad=sig_phi,
        sigma_curvature_stat=sig_kappa,
        anchor_ra_deg=ra,
        anchor_dec_deg=dec,
        length_arcsec=3.0,
        width_arcsec=0.4,
        axis_ratio=0.13,
    )
    return ArcMeasurement(
        label=label,
        seed_ra_deg=ra,
        seed_dec_deg=dec,
        success=success,
        failure_reason=None if success else "synthetic failure",
        anchor_ra_deg=ra,
        anchor_dec_deg=dec,
        tangent_angle_offset_rad=angle,
        curvature_arcsec_inv=kappa,
        sigma_tangent_rad=sig_phi,
        sigma_curvature_arcsec_inv=sig_kappa,
        reliability=reliability,
        reference_band="F814W",
        bands=(band,),
    )


def test_round_trip_through_lenscluster_parser(tmp_path) -> None:
    rows = [
        _measurement("2.a"),
        _measurement("2.b", ra=39.96, dec=-1.59, angle=0.31, kappa=0.041, reliability=1.4),
    ]
    path = write_arcfile(rows, tmp_path / "arcs.cat")
    df = _load_arc_constraints_catalog(path, None)
    assert list(df["arc_id"]) == ["2.a", "2.b"]
    assert list(df["z_arc"]) == [-1.0, -1.0]
    first = df.iloc[0]
    assert float(first["arc_anchor_ra"]) == pytest.approx(39.9712345, abs=1e-7)
    assert float(first["arc_anchor_dec"]) == pytest.approx(-1.5823456, abs=1e-7)
    assert float(first["arc_tangent_angle_rad"]) == pytest.approx(1.2345678, abs=1e-7)
    assert float(first["arc_curvature_arcsec_inv"]) == pytest.approx(0.025, rel=1e-5)
    assert float(first["arc_sigma_tangent_angle_rad"]) == pytest.approx(0.045, rel=1e-5)
    assert float(first["arc_sigma_curvature_arcsec_inv"]) == pytest.approx(0.006, rel=1e-5)
    # Reliability written clipped to [0, 1]; the loader clips as well.
    assert float(df.iloc[1]["arc_reliability"]) == pytest.approx(1.0)


def test_failed_measurements_are_skipped(tmp_path) -> None:
    rows = [_measurement("2.a"), _measurement("9.z", success=False)]
    path = write_arcfile(rows, tmp_path / "arcs.cat")
    df = _load_arc_constraints_catalog(path, None)
    assert list(df["arc_id"]) == ["2.a"]


def test_duplicate_labels_raise(tmp_path) -> None:
    rows = [_measurement("2.a"), _measurement("2.a", ra=39.96)]
    with pytest.raises(ArcfileWriteError, match="Duplicate"):
        write_arcfile(rows, tmp_path / "arcs.cat")


def test_invalid_arc_id_raises(tmp_path) -> None:
    with pytest.raises(ArcfileWriteError, match="whitespace"):
        write_arcfile([_measurement("bad id")], tmp_path / "arcs.cat")


def test_nonfinite_values_raise(tmp_path) -> None:
    with pytest.raises(ArcfileWriteError, match="non-finite"):
        write_arcfile([_measurement("2.a", kappa=math.nan)], tmp_path / "arcs.cat")


def test_nonpositive_sigma_raises(tmp_path) -> None:
    with pytest.raises(ArcfileWriteError, match="strictly positive"):
        write_arcfile([_measurement("2.a", sig_phi=0.0)], tmp_path / "arcs.cat")


def test_no_overwrite_without_flag(tmp_path) -> None:
    path = write_arcfile([_measurement("2.a")], tmp_path / "arcs.cat")
    with pytest.raises(ArcfileWriteError, match="exists"):
        write_arcfile([_measurement("2.a")], path)
    write_arcfile([_measurement("2.a")], path, overwrite=True)


def test_negative_curvature_written_as_magnitude(tmp_path) -> None:
    path = write_arcfile([_measurement("2.a", kappa=-0.02)], tmp_path / "arcs.cat")
    df = _load_arc_constraints_catalog(path, None)
    assert float(df.iloc[0]["arc_curvature_arcsec_inv"]) == pytest.approx(0.02, rel=1e-6)


def test_legacy_header_without_z_arc_is_rejected(tmp_path) -> None:
    # Pre-z_arc arctrace format: 8 columns, header names them but omits z_arc.
    # The old loader padded a trailing reliability and silently mis-shifted every
    # column from the 4th on; it must now raise instead.
    path = tmp_path / "legacy.cat"
    path.write_text(
        "# arctrace v0.0.1\n"
        "#REFERENCE 0\n"
        "# image_label ra_deg dec_deg tangent_angle_rad curvature_arcsec_inv "
        "sigma_tangent_angle_rad sigma_curvature_arcsec_inv reliability\n"
        "1.a 0.0035403 -0.0062014 2.4288193 2.815062e-02 3.600544e-02 9.292779e-02 0.850\n"
    )
    with pytest.raises(ValueError, match="z_arc"):
        _load_arc_constraints_catalog(path, None)


def test_named_header_maps_by_name_with_reordered_and_optional_columns(tmp_path) -> None:
    # A named column header is authoritative: columns may be reordered and the
    # optional reliability column omitted (defaults to 1.0).
    path = tmp_path / "named.cat"
    path.write_text(
        "#REFERENCE 0\n"
        "# arc_id z_arc dec_deg ra_deg curvature_arcsec_inv tangent_angle_rad "
        "sigma_curvature_arcsec_inv sigma_tangent_angle_rad\n"
        "arcX -1 20.0 10.0 0.09 0.82 0.02 0.05\n"
    )
    df = _load_arc_constraints_catalog(path, None)
    row = df.iloc[0]
    assert str(row["arc_id"]) == "arcX"
    assert float(row["z_arc"]) == pytest.approx(-1.0)
    assert float(row["arc_anchor_ra"]) == pytest.approx(10.0)
    assert float(row["arc_anchor_dec"]) == pytest.approx(20.0)
    assert float(row["arc_tangent_angle_rad"]) == pytest.approx(0.82)
    assert float(row["arc_curvature_arcsec_inv"]) == pytest.approx(0.09)
    assert float(row["arc_sigma_tangent_angle_rad"]) == pytest.approx(0.05)
    assert float(row["arc_sigma_curvature_arcsec_inv"]) == pytest.approx(0.02)
    assert float(row["arc_reliability"]) == pytest.approx(1.0)


def test_sidetable_outputs(tmp_path) -> None:
    rows = [_measurement("2.a"), _measurement("9.z", success=False)]
    frame = measurements_to_dataframe(rows)
    assert len(frame) == 2
    assert "band_F814W_curvature_arcsec_inv" in frame.columns
    write_sidetable(rows, tmp_path / "side.csv", tmp_path / "side.json")
    assert (tmp_path / "side.csv").exists()
    payload = (tmp_path / "side.json").read_text()
    assert "2.a" in payload and "9.z" in payload
    assert np.isfinite(frame.iloc[0]["curvature_arcsec_inv"])
