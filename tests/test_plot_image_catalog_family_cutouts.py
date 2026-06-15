from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

import lenscluster.plotting as solver_plotting


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_image_catalog_family_cutouts.py"
spec = importlib.util.spec_from_file_location("plot_image_catalog_family_cutouts", SCRIPT_PATH)
assert spec is not None
plotter = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = plotter
spec.loader.exec_module(plotter)


def _wcs_header(ra: float = 10.0, dec: float = 0.0, crpix: tuple[float, float] = (50.0, 50.0)) -> fits.Header:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = list(crpix)
    wcs.wcs.cdelt = np.array([-1.0 / 3600.0, 1.0 / 3600.0])
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs.to_header()


def _write_band_image(
    root: Path,
    band: str,
    *,
    cluster_token: str = "abell370",
    crpix: tuple[float, float] = (50.0, 50.0),
) -> Path:
    path = root / cluster_token / f"hlsp_buffalo_hst_multi_{cluster_token}_{band.lower()}_v1.0_drz.fits"
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:100, :100]
    offsets = {"F435W": 5.0, "F606W": 15.0, "F814W": 30.0}
    data = offsets.get(band, 0.0) + 0.1 * xx + 0.2 * yy
    fits.PrimaryHDU(data.astype(np.float32), header=_wcs_header(crpix=crpix)).writeto(path, overwrite=True)
    return path


def _write_rgb_images(root: Path) -> None:
    for band in plotter.DEFAULT_BANDS:
        _write_band_image(root, band)


def _write_par(path: Path, *, ra0: float = 10.0, dec0: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "runmode",
                f"  reference 3 {ra0:.8f} {dec0:.8f}",
                "  end",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_run_writes_pdf_for_full_lenstool_image_catalog(tmp_path: Path) -> None:
    catalog_path = tmp_path / "obs_arcs.cat"
    catalog_path.write_text(
        "#REFERENCE 0\n"
        "1.1 10.000000 0.000000 0.1 0.1 0.0 2.0 25.0\n"
        "1.2 10.000500 0.000000 0.1 0.1 0.0 2.0 25.0\n"
        "2.1 10.001000 0.000000 0.1 0.1 0.0 3.0 25.0\n",
        encoding="utf-8",
    )
    _write_rgb_images(tmp_path / "images")

    output = plotter.run(
        "A307",
        catalog_path,
        image_dir=tmp_path / "images",
        output=tmp_path / "full_catalog_cutouts.pdf",
        cutout_size_arcsec=8.0,
        families_per_page=1,
    )

    assert output == tmp_path / "full_catalog_cutouts.pdf"
    assert output.exists()
    assert output.stat().st_size > 0


def test_run_writes_pdf_for_compact_image_catalog(tmp_path: Path) -> None:
    catalog_path = tmp_path / "sl-final.dat"
    catalog_path.write_text(
        "# ID RA DEC z cat\n"
        "1.1 10.000000 0.000000 2.5 Gold\n"
        "1.2 10.000500 0.000000 2.5 Gold\n",
        encoding="utf-8",
    )
    _write_rgb_images(tmp_path / "images")

    output = plotter.run(
        "a370",
        catalog_path,
        image_dir=tmp_path / "images",
        output=tmp_path / "compact_catalog_cutouts.pdf",
        cutout_size_arcsec=8.0,
        families_per_page=1,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_run_writes_reference3_catalog_with_par_reference(tmp_path: Path) -> None:
    catalog_path = tmp_path / "offset_obs_arcs.cat"
    catalog_path.write_text(
        "#REFERENCE 3\n"
        "1.1 0.000000 0.000000 0.1 0.1 0.0 2.0 25.0\n"
        "1.2 -1.000000 0.000000 0.1 0.1 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    par_path = _write_par(tmp_path / "model.par")
    _write_rgb_images(tmp_path / "images")

    output = plotter.run(
        "a370",
        catalog_path,
        image_dir=tmp_path / "images",
        output=tmp_path / "reference3_cutouts.pdf",
        cutout_size_arcsec=8.0,
        families_per_page=1,
        par_path=par_path,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_missing_rgb_images_fail_with_download_hint(tmp_path: Path) -> None:
    catalog_path = tmp_path / "obs_arcs.cat"
    catalog_path.write_text(
        "#REFERENCE 0\n"
        "1.1 10.000000 0.000000 0.1 0.1 0.0 2.0 25.0\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        plotter.run("a370", catalog_path, image_dir=tmp_path / "missing_images")

    message = str(exc_info.value)
    assert "Missing BUFFALO RGB FITS image(s)" in message
    assert "/home/dutra/.conda/envs/lenstronomy/bin/python download_catalogs.py --catalog buffalo-images" in message


def test_solver_cutout_overlay_positions_use_clamped_edge_window(tmp_path: Path) -> None:
    path = _write_band_image(tmp_path / "images", "F814W", crpix=(2.0, 2.0))
    image = plotter.load_rgb_metadata({"F814W": path}, bands=("F814W",))["F814W"]
    coord = SkyCoord(ra=10.0 * u.deg, dec=0.0 * u.deg, frame="icrs")

    pixel = solver_plotting._cutout_pixel_xy(image, coord, coord, cutout_size_arcsec=8.0)

    assert pixel == pytest.approx((1.0, 1.0), abs=1.0e-3)


def test_image_catalog_family_cutout_blocks_span_and_include_all_panels() -> None:
    n_images = 11
    family = SimpleNamespace(
        family_id="7",
        z_source=2.3,
        effective_z_source=2.0,
        image_labels=[f"7.{index}" for index in range(1, n_images + 1)],
        x_obs=np.arange(n_images, dtype=float),
        y_obs=np.zeros(n_images, dtype=float),
    )
    state = SimpleNamespace(reference=(3, 10.0, 0.0), family_data=[family])
    image_df = pd.DataFrame(
        {
            "family_id": ["7"] * n_images,
            "image_label": family.image_labels,
            "x_model_arcsec": [*(np.arange(n_images - 1, dtype=float) + 0.1), np.nan],
            "y_model_arcsec": [*(np.zeros(n_images - 1, dtype=float) + 0.1), np.nan],
            "image_recovery_status": ["recovered"] * (n_images - 1) + ["not_recovered"],
        }
    )
    extra_image_df = pd.DataFrame(
        {
            "family_id": ["7"],
            "extra_image_index": [1],
            "x_model_arcsec": [20.0],
            "y_model_arcsec": [0.0],
            "z_source": [2.3],
            "effective_z_source": [2.0],
        }
    )

    catalog_df = solver_plotting._image_catalog_cutout_rows(state, image_df)
    extra_df = solver_plotting._image_catalog_extra_cutout_rows(state, extra_image_df)
    blocks = solver_plotting._image_catalog_family_cutout_blocks(
        state,
        catalog_df,
        extra_df,
        detail_cols=3,
        default_cutout_size_arcsec=10.0,
    )

    assert len(blocks) == 1
    block = blocks[0]
    assert block["detail_row_count"] == 4
    assert block["overview_units"] == 3
    assert block["layout_rowspan"] == 7
    assert block["overview_rowspan"] == 7
    assert block["overview_cutout_size_arcsec"] >= 40.0
    assert len(block["detail_panels"]) == 12
    assert [panel["panel_kind"] for panel in block["detail_panels"][:11]] == ["observed"] * 11
    assert block["detail_panels"][0]["panel_status"] == "POINT_RECOVERED"
    assert block["detail_panels"][10]["panel_status"] == "MISSED"
    assert block["detail_panels"][11]["panel_kind"] == "extra"
    assert block["detail_panels"][11]["panel_status"] == "EXTRA"
    assert block["detail_panels"][11]["x_center_arcsec"] == pytest.approx(20.0)


def test_image_catalog_cutout_rows_merge_zero_padded_family_diagnostics() -> None:
    family = SimpleNamespace(
        family_id="0121",
        z_source=2.3,
        effective_z_source=2.0,
        image_labels=["0121.a", "0121.b"],
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, 1.0], dtype=float),
    )
    state = SimpleNamespace(reference=(3, 10.0, 0.0), family_data=[family])
    image_df = pd.DataFrame(
        {
            "family_id": [121, 121],
            "image_label": ["121.a", "0121.b"],
            "x_model_arcsec": [0.1, np.nan],
            "y_model_arcsec": [0.1, np.nan],
            "image_recovery_status": ["recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "arc_supported"],
            "arc_supported": [False, True],
            "arc_aware_image_residual_arcsec": [0.12, 0.18],
        }
    )

    catalog_df = solver_plotting._image_catalog_cutout_rows(state, image_df)

    assert catalog_df["family_id"].tolist() == ["0121", "0121"]
    assert catalog_df["image_label"].tolist() == ["0121.a", "0121.b"]
    assert catalog_df["image_recovery_status"].tolist() == ["recovered", "not_recovered"]
    assert catalog_df["arc_recovery_status"].tolist() == ["point_recovered", "arc_supported"]
    assert [
        solver_plotting._image_catalog_observed_panel_status(row)
        for _, row in catalog_df.iterrows()
    ] == ["POINT_RECOVERED", "ARC_RECOVERED"]


def test_image_catalog_family_cutout_compact_labels_and_legend_handles() -> None:
    observed = pd.DataFrame(
        {
            "image_recovery_status": ["recovered", "not_recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "arc_supported", "not_recovered"],
            "arc_aware_image_residual_arcsec": [0.1, 0.2, np.nan],
            "x_model_arcsec": [0.1, np.nan, np.nan],
            "y_model_arcsec": [0.0, np.nan, np.nan],
            "arc_support_anchor_x_arcsec": [9.0, 2.5, 7.0],
            "arc_support_anchor_y_arcsec": [9.5, -1.5, 7.5],
        }
    )
    block = {
        "family_id": "4",
        "z_source": 2.345,
        "observed": observed,
        "extras": pd.DataFrame({"x_model_arcsec": [1.0], "y_model_arcsec": [2.0]}),
    }
    overview_label = solver_plotting._format_image_catalog_overview_label(block)

    assert overview_label == (
        "Family 4  z=2.35\n"
        "Nobs=3  Npoint_recovered=1  Narc_recovered=1  Narc_supported=1  Nextra=1"
    )
    assert [solver_plotting._image_catalog_observed_panel_status(row) for _, row in observed.iterrows()] == [
        "POINT_RECOVERED",
        "ARC_RECOVERED",
        "MISSED",
    ]
    assert solver_plotting._image_catalog_status_color("POINT_RECOVERED") == "#4da3ff"
    assert solver_plotting._image_catalog_status_color("ARC_RECOVERED") == "#ffd54f"
    assert solver_plotting._image_catalog_status_color("arc_supported") == "#ffd54f"
    assert solver_plotting._image_catalog_status_color("MISSED") == "#ff4d5e"
    assert solver_plotting._image_catalog_status_color("not_recovered") == "#ff4d5e"
    assert solver_plotting._image_catalog_status_color("EXTRA") == "tab:purple"
    assert solver_plotting._image_catalog_status_display_text("arc_supported") == "arc recovered"
    assert solver_plotting._image_catalog_display_model_arcsec(observed.iloc[0]) == pytest.approx((0.1, 0.0))
    assert solver_plotting._image_catalog_display_model_arcsec(observed.iloc[1]) is None
    assert solver_plotting._image_catalog_display_model_arcsec(observed.iloc[2]) is None

    detail_label = solver_plotting._format_image_catalog_compact_detail_label(
        pd.Series(
            {
                "image_label": "4.1",
                "panel_status": "POINT_RECOVERED",
                "image_recovery_status": "recovered",
                "arc_recovery_status": "point_recovered",
                "arc_candidate_supported": True,
                "point_image_residual_arcsec": 0.054,
                "arc_candidate_image_residual_arcsec": 0.0456,
                "image_residual_arcsec": 0.054,
                "arc_aware_image_residual_arcsec": 0.054,
                "arc_curve_distance_arcsec": 0.0456,
                "arc_prior_probability": 0.75,
                "arc_noncritical_direction_residual_arcsec": 0.02,
                "arc_critical_direction_residual_arcsec": 4.0,
                "arc_s_min": 0.04,
                "arc_s_max": 1.1,
                "arc_detA": 0.044,
            }
        )
    )
    assert detail_label == (
        "4.1  point recovered (arc supported)\n"
        "point=recovered  r_point=0.054\n"
        "arc=supported  r_arc=0.046  p_arc=0.75\n"
        "d_curve=0.046  N=0.02  T=4\n"
        "s=0.04/1.1  detA=0.044"
    )
    arc_detail_label = solver_plotting._format_image_catalog_compact_detail_label(
        pd.Series(
            {
                "image_label": "4.2",
                "panel_status": "ARC_RECOVERED",
                "image_recovery_status": "not_recovered",
                "preferred_recovery_status": "arc_supported",
                "arc_recovery_status": "arc_supported",
                "arc_supported": True,
                "image_residual_arcsec": np.nan,
                "arc_curve_distance_arcsec": 0.041,
                "arc_prior_probability": 0.8,
                "arc_noncritical_direction_residual_arcsec": 0.02,
                "arc_critical_direction_residual_arcsec": 3.0,
                "arc_s_min": 0.01,
                "arc_s_max": 0.9,
                "arc_detA": 0.05,
            }
        )
    )
    assert arc_detail_label == (
        "4.2  arc recovered\n"
        "point=arc recovered  r_point=na\n"
        "arc=recovered  r_arc=0.041  p_arc=0.8\n"
        "d_curve=0.041  N=0.02  T=3\n"
        "s=0.01/0.9  detA=0.05"
    )

    extra_label = solver_plotting._format_image_catalog_extra_label(
        pd.Series(
            {
                "image_label": "4.extra2",
                "extra_image_index": 2,
                "x_model_arcsec": 1.234,
                "y_model_arcsec": -5.678,
            }
        )
    )
    assert extra_label == "4.extra2  extra\nmodel x=1.23 y=-5.68"

    handles = solver_plotting._image_catalog_legend_handles(include_critical_lines=True)
    assert [handle.get_label() for handle in handles] == [
        "point recovered image",
        "arc recovered image",
        "missed observed image",
        "matched model image",
        "extra model image",
        "arc-support curve",
        "observed-to-model residual",
        "tangential arc displacement",
        "linearized arc anchor",
        "tangential critical line",
        "radial critical line",
    ]
    handle_by_label = {handle.get_label(): handle for handle in handles}
    assert handle_by_label["arc-support curve"].get_color() == solver_plotting._image_catalog_status_color("ARC_RECOVERED")
    assert handle_by_label["arc-support curve"].get_linestyle() == "--"
    assert handle_by_label["tangential arc displacement"].get_color() == solver_plotting._image_catalog_status_color("ARC_RECOVERED")
    assert handle_by_label["tangential arc displacement"].get_linestyle() == "-"
    assert handle_by_label["tangential critical line"].get_color() == solver_plotting.IMAGE_CATALOG_TANGENTIAL_CRITICAL_COLOR
    assert handle_by_label["tangential critical line"].get_linestyle() == "-"
    assert handle_by_label["radial critical line"].get_color() == solver_plotting.IMAGE_CATALOG_RADIAL_CRITICAL_COLOR
    assert handle_by_label["radial critical line"].get_linestyle() == "--"


def test_image_catalog_axis_legend_uses_larger_font(monkeypatch: pytest.MonkeyPatch) -> None:
    legend_calls: list[dict[str, object]] = []
    original_legend = solver_plotting.plt.Axes.legend

    def recording_legend(self: object, *args: object, **kwargs: object) -> object:
        legend_calls.append(kwargs.copy())
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(solver_plotting.plt.Axes, "legend", recording_legend)
    fig, ax = plt.subplots()
    try:
        solver_plotting._add_image_catalog_axis_legend(ax)
    finally:
        plt.close(fig)

    assert legend_calls
    assert legend_calls[0]["fontsize"] == pytest.approx(12.0)


def test_image_catalog_arc_support_geometry_uses_closest_curve_point() -> None:
    row = pd.Series(
        {
            "image_recovery_status": "not_recovered",
            "arc_recovery_status": "arc_supported",
            "x_obs_arcsec": 1.5,
            "y_obs_arcsec": 1.0,
            "x_model_arcsec": np.nan,
            "y_model_arcsec": np.nan,
            "arc_support_anchor_x_arcsec": 2.0,
            "arc_support_anchor_y_arcsec": 2.0,
            "arc_support_curve_x_arcsec": "[0.0, 2.0, 2.0]",
            "arc_support_curve_y_arcsec": "[0.0, 0.0, 2.0]",
            "arc_curve_distance_arcsec": 0.5,
        }
    )

    geometry = solver_plotting._image_catalog_arc_support_geometry(row)

    assert geometry is not None
    np.testing.assert_allclose(geometry["closest_arcsec"], np.asarray([2.0, 1.0]), rtol=1.0e-12)
    assert geometry["residual_arcsec"] == pytest.approx(row["arc_curve_distance_arcsec"])
    np.testing.assert_allclose(geometry["tangential_curve_arcsec"][0], np.asarray([2.0, 1.0]), rtol=1.0e-12)
    np.testing.assert_allclose(geometry["tangential_curve_arcsec"][-1], np.asarray([2.0, 2.0]), rtol=1.0e-12)
    assert solver_plotting._image_catalog_display_model_arcsec(row) is None


def _arc_anchor_overlay_row(**overrides: object) -> pd.Series:
    data: dict[str, object] = {
        "image_recovery_status": "not_recovered",
        "arc_recovery_status": "arc_supported",
        "x_obs_arcsec": 1.5,
        "y_obs_arcsec": 1.0,
        "x_model_arcsec": np.nan,
        "y_model_arcsec": np.nan,
        "arc_aware_image_residual_arcsec": 0.08,
        "arc_support_anchor_x_arcsec": 2.0,
        "arc_support_anchor_y_arcsec": 2.0,
        "arc_support_curve_x_arcsec": "[0.0, 2.0, 2.0]",
        "arc_support_curve_y_arcsec": "[0.0, 0.0, 2.0]",
        "arc_curve_distance_arcsec": 0.5,
        "arc_supported": True,
    }
    data.update(overrides)
    return pd.Series(data)


def test_image_catalog_arc_anchor_overlays_skip_point_recovered_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _arc_anchor_overlay_row(
        image_recovery_status="recovered",
        preferred_recovery_status="point_recovered",
        arc_recovery_status="point_recovered",
        x_model_arcsec=1.6,
        y_model_arcsec=1.0,
        point_image_residual_arcsec=0.1,
        preferred_image_residual_arcsec=0.1,
        arc_supported=False,
    )

    def fail_if_drawn(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("arc-anchor overlay drawing should be gated off")

    monkeypatch.setattr(solver_plotting, "_draw_cutout_polyline_arcsec", fail_if_drawn)
    monkeypatch.setattr(solver_plotting, "_draw_cutout_segment", fail_if_drawn)
    monkeypatch.setattr(solver_plotting, "_draw_image_catalog_arc_anchor_marker", fail_if_drawn)

    assert not solver_plotting._image_catalog_draw_arc_anchor_overlays(row)
    assert not solver_plotting._draw_image_catalog_arc_support_curve(
        object(),
        object(),
        SkyCoord(ra=10.0 * u.deg, dec=0.0 * u.deg, frame="icrs"),
        row,
        (3, 10.0, 0.0),
        cutout_size_arcsec=8.0,
        rendered_shape=(100, 100),
    )
    assert not solver_plotting._draw_image_catalog_arc_supported_components(
        object(),
        object(),
        SkyCoord(ra=10.0 * u.deg, dec=0.0 * u.deg, frame="icrs"),
        row,
        (3, 10.0, 0.0),
        cutout_size_arcsec=8.0,
        rendered_shape=(100, 100),
    )


def test_image_catalog_arc_anchor_overlays_draw_for_point_recovered_arc_supported_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _arc_anchor_overlay_row(
        image_recovery_status="recovered",
        preferred_recovery_status="point_recovered",
        arc_recovery_status="point_recovered",
        x_model_arcsec=1.6,
        y_model_arcsec=1.0,
        point_image_residual_arcsec=0.1,
        arc_candidate_supported=True,
        arc_candidate_image_residual_arcsec=0.08,
        preferred_image_residual_arcsec=0.1,
        arc_supported=False,
        arc_aware_image_residual_arcsec=0.1,
    )
    polyline_calls: list[dict[str, object]] = []
    segment_calls: list[dict[str, object]] = []
    marker_calls: list[dict[str, object]] = []

    def fake_polyline(*_args: object, **kwargs: object) -> bool:
        polyline_calls.append(dict(kwargs))
        return True

    def fake_segment(*_args: object, **kwargs: object) -> tuple[float, float]:
        segment_calls.append(dict(kwargs))
        return (0.0, 0.0)

    def fake_marker(*_args: object, **kwargs: object) -> bool:
        marker_calls.append(dict(kwargs))
        return True

    monkeypatch.setattr(solver_plotting, "_draw_cutout_polyline_arcsec", fake_polyline)
    monkeypatch.setattr(solver_plotting, "_draw_cutout_segment", fake_segment)
    monkeypatch.setattr(solver_plotting, "_draw_image_catalog_arc_anchor_marker", fake_marker)

    center = SkyCoord(ra=10.0 * u.deg, dec=0.0 * u.deg, frame="icrs")

    assert solver_plotting._image_catalog_observed_panel_status(row) == "POINT_RECOVERED"
    assert solver_plotting._image_catalog_draw_arc_anchor_overlays(row)
    assert solver_plotting._draw_image_catalog_arc_support_curve(
        object(),
        object(),
        center,
        row,
        (3, 10.0, 0.0),
        cutout_size_arcsec=8.0,
        rendered_shape=(100, 100),
    )
    assert solver_plotting._draw_image_catalog_arc_supported_components(
        object(),
        object(),
        center,
        row,
        (3, 10.0, 0.0),
        cutout_size_arcsec=8.0,
        rendered_shape=(100, 100),
    )
    assert len(polyline_calls) == 2
    assert len(segment_calls) == 1
    assert len(marker_calls) == 1


def test_image_catalog_overview_geometry_gates_arc_anchor_bounds() -> None:
    point_row = _arc_anchor_overlay_row(
        image_recovery_status="recovered",
        arc_recovery_status="point_recovered",
        x_obs_arcsec=0.0,
        y_obs_arcsec=0.0,
        x_model_arcsec=0.1,
        y_model_arcsec=0.0,
        arc_support_anchor_x_arcsec=100.0,
        arc_support_anchor_y_arcsec=0.0,
        arc_supported=False,
    )
    point_center_x, point_center_y, point_size = solver_plotting._image_catalog_overview_geometry(
        pd.DataFrame([point_row.to_dict()]),
        pd.DataFrame(),
        10.0,
    )

    assert not solver_plotting._image_catalog_draw_arc_anchor_overlays(point_row)
    assert point_center_x == pytest.approx(0.05)
    assert point_center_y == pytest.approx(0.0)
    assert point_size == pytest.approx(40.0)

    arc_row = _arc_anchor_overlay_row(
        image_recovery_status="not_recovered",
        preferred_recovery_status="arc_supported",
        arc_recovery_status="arc_supported",
        x_obs_arcsec=0.0,
        y_obs_arcsec=0.0,
        x_model_arcsec=np.nan,
        y_model_arcsec=np.nan,
        arc_candidate_supported=True,
        arc_candidate_image_residual_arcsec=0.08,
        preferred_image_residual_arcsec=0.08,
        arc_aware_image_residual_arcsec=0.08,
        arc_support_anchor_x_arcsec=100.0,
        arc_support_anchor_y_arcsec=0.0,
        arc_supported=True,
    )
    arc_center_x, arc_center_y, arc_size = solver_plotting._image_catalog_overview_geometry(
        pd.DataFrame([arc_row.to_dict()]),
        pd.DataFrame(),
        10.0,
    )

    assert solver_plotting._image_catalog_draw_arc_anchor_overlays(arc_row)
    assert arc_center_x == pytest.approx(50.0)
    assert arc_center_y == pytest.approx(1.0)
    assert arc_size == pytest.approx(130.0)


def test_image_catalog_cluster_overview_includes_model_extra_and_arc_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_df = pd.DataFrame(
        {
            "image_label": ["1.1", "1.2"],
            "x_obs_arcsec": [0.0, 50.0],
            "y_obs_arcsec": [0.0, 10.0],
            "x_model_arcsec": [2.0, np.nan],
            "y_model_arcsec": [0.0, np.nan],
            "image_recovery_status": ["recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "arc_supported"],
            "arc_candidate_supported": [False, True],
            "arc_candidate_image_residual_arcsec": [np.nan, 0.25],
            "arc_support_anchor_x_arcsec": [np.nan, 100.0],
            "arc_support_anchor_y_arcsec": [np.nan, 0.0],
            "arc_support_curve_x_arcsec": ["[]", "[50.0,100.0,120.0]"],
            "arc_support_curve_y_arcsec": ["[]", "[10.0,0.0,0.0]"],
            "arc_supported": [False, True],
        }
    )
    extra_df = pd.DataFrame(
        {
            "x_model_arcsec": [-40.0],
            "y_model_arcsec": [-10.0],
        }
    )

    center_x, center_y, size = solver_plotting._image_catalog_cluster_overview_geometry(catalog_df, extra_df)

    assert center_x == pytest.approx(40.0)
    assert center_y == pytest.approx(0.0)
    assert size == pytest.approx(208.0)

    observed_statuses: list[str] = []
    model_marker_count = 0
    extra_marker_count = 0
    residual_segment_count = 0
    arc_curve_labels: list[str] = []
    arc_component_labels: list[str] = []

    def fake_rgb(*_args: object, **_kwargs: object) -> np.ndarray:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def fake_observed_marker(*_args: object, status: str, **_kwargs: object) -> None:
        observed_statuses.append(status)

    def fake_model_marker(*_args: object, **_kwargs: object) -> None:
        nonlocal model_marker_count
        model_marker_count += 1

    def fake_extra_marker(*_args: object, **_kwargs: object) -> None:
        nonlocal extra_marker_count
        extra_marker_count += 1

    def fake_segment(*_args: object, **_kwargs: object) -> tuple[float, float]:
        nonlocal residual_segment_count
        residual_segment_count += 1
        return (0.0, 0.0)

    def fake_arc_curve(_ax: object, _image: object, _center: object, row: pd.Series, *_args: object, **_kwargs: object) -> bool:
        arc_curve_labels.append(str(row.get("image_label", "")))
        return True

    def fake_arc_components(_ax: object, _image: object, _center: object, row: pd.Series, *_args: object, **_kwargs: object) -> bool:
        arc_component_labels.append(str(row.get("image_label", "")))
        return True

    monkeypatch.setattr(solver_plotting, "_image_catalog_draw_rgb_cutout", fake_rgb)
    monkeypatch.setattr(solver_plotting, "_draw_image_catalog_observed_marker", fake_observed_marker)
    monkeypatch.setattr(solver_plotting, "_draw_image_catalog_model_marker", fake_model_marker)
    monkeypatch.setattr(solver_plotting, "_draw_image_catalog_extra_marker", fake_extra_marker)
    monkeypatch.setattr(solver_plotting, "_draw_cutout_segment", fake_segment)
    monkeypatch.setattr(solver_plotting, "_draw_image_catalog_arc_support_curve", fake_arc_curve)
    monkeypatch.setattr(solver_plotting, "_draw_image_catalog_arc_supported_components", fake_arc_components)

    fig, ax = plt.subplots()
    try:
        solver_plotting._draw_image_catalog_cluster_overview_panel(
            ax,
            object(),
            {},
            ("F435W", "F606W", "F814W"),
            object(),
            object(),
            catalog_df,
            extra_df,
            (3, 10.0, 0.0),
        )
    finally:
        plt.close(fig)

    assert observed_statuses == ["POINT_RECOVERED", "ARC_RECOVERED"]
    assert model_marker_count == 1
    assert extra_marker_count == 1
    assert residual_segment_count == 1
    assert arc_curve_labels == ["1.2"]
    assert arc_component_labels == ["1.2"]


def test_lock_cutout_axis_to_image_disables_autoscale() -> None:
    fig, ax = plt.subplots()
    try:
        solver_plotting._lock_cutout_axis_to_image(ax, (200, 200, 3))

        assert ax.get_xlim() == pytest.approx((-0.5, 199.5))
        assert ax.get_ylim() == pytest.approx((-0.5, 199.5))
        assert not ax.get_autoscale_on()
    finally:
        plt.close(fig)


def test_locked_cutout_axis_limits_survive_out_of_bounds_artists() -> None:
    fig, ax = plt.subplots()
    try:
        solver_plotting._lock_cutout_axis_to_image(ax, (200, 200, 3))
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.plot([-100.0, 300.0], [-100.0, 300.0], clip_on=True)

        assert ax.get_xlim() == pytest.approx(xlim)
        assert ax.get_ylim() == pytest.approx(ylim)
    finally:
        plt.close(fig)


def test_main_returns_2_for_missing_catalog_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = plotter.main(["a370", str(tmp_path / "missing.cat")])

    captured = capsys.readouterr()
    assert result == 2
    assert "error: Missing image catalog" in captured.err


def test_stage_image_catalog_family_cutouts_include_solver_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_rgb_images(tmp_path / "images")
    run_dir = tmp_path / "stage4_critical_arc_mixture_image_plane"
    family = SimpleNamespace(
        family_id="1",
        z_source=2.3,
        effective_z_source=2.0,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0], dtype=float),
    )
    state = SimpleNamespace(
        run_name="a370_test",
        par_path="data/a370_niemiec/a_sl_normal.par",
        reference=(3, 10.0, 0.0),
        parameter_specs=[],
        family_data=[family],
    )
    image_df = pd.DataFrame(
        {
            "family_id": ["1", "1"],
            "image_label": ["1.1", "1.2"],
            "x_obs_arcsec": [0.0, 1.0],
            "y_obs_arcsec": [0.0, 0.0],
            "x_model_arcsec": [0.05, np.nan],
            "y_model_arcsec": [0.02, np.nan],
            "image_residual_arcsec": [0.054, np.nan],
            "arc_aware_image_residual_arcsec": [0.054, np.nan],
            "arc_noncritical_direction_residual_arcsec": [0.02, 0.03],
            "arc_critical_direction_residual_arcsec": [0.01, 4.0],
            "arc_critical_direction_x": [1.0, 1.0],
            "arc_critical_direction_y": [0.0, 0.0],
            "arc_noncritical_direction_x": [0.0, 0.0],
            "arc_noncritical_direction_y": [1.0, 1.0],
            "arc_s_min": [0.2, 0.04],
            "arc_s_max": [1.0, 1.1],
            "arc_detA": [0.2, 0.044],
            "arc_prior_probability": [0.4, 0.75],
            "arc_curve_distance_arcsec": [np.nan, 0.03],
            "arc_curve_arclength_arcsec": [np.nan, 3.8],
            "arc_curve_finite": [False, True],
            "arc_support_anchor_x_arcsec": [np.nan, 4.8],
            "arc_support_anchor_y_arcsec": [np.nan, 0.0],
            "arc_support_curve_x_arcsec": ["[]", "[-0.2,1.0,2.6,4.8]"],
            "arc_support_curve_y_arcsec": ["[]", "[0.0,0.02,0.04,0.0]"],
            "image_recovery_status": ["recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "not_recovered"],
            "arc_supported": [False, False],
            "arc_support_finite": [True, True],
        }
    )
    extra_image_df = pd.DataFrame(
        {
            "family_id": ["1"],
            "extra_image_index": [1],
            "image_recovery_status": ["extra"],
            "x_model_arcsec": [-1.0],
            "y_model_arcsec": [0.5],
            "z_source": [2.3],
            "effective_z_source": [2.0],
        }
    )
    z_calls: list[float] = []
    packed_z_calls: list[float] = []
    packed_state_calls: list[dict[str, float]] = []
    conversion_calls: list[np.ndarray] = []
    fake_model = object()

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, object] = {}

        def reported_physical_to_latent_parameter_vector(self, theta):
            theta_array = np.asarray(theta, dtype=float)
            conversion_calls.append(theta_array.copy())
            return theta_array + 1.0

        def _get_exact_model_solver(self, z_source):
            z_calls.append(float(z_source))
            return fake_model, None

        def _build_packed_lens_state(self, theta, z_source):
            theta_array = np.asarray(theta, dtype=float)
            packed_z_calls.append(float(z_source))
            return {"z_source": float(z_source), "theta0": float(theta_array[0])}

        def _packed_to_kwargs_lens(self, packed_state):
            payload = dict(packed_state)
            packed_state_calls.append(payload)
            return [payload]

    critical_calls: list[dict[str, object]] = []

    def recording_critical_curve_caustics(
        model: object,
        kwargs_lens: list[dict[str, float]],
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        *,
        include_tangential: bool,
        include_radial: bool,
    ) -> list[dict[str, np.ndarray]]:
        critical_calls.append(
            {
                "model": model,
                "kwargs_lens": [dict(item) for item in kwargs_lens],
                "center": (float(0.5 * (x_axis[0] + x_axis[-1])), float(0.5 * (y_axis[0] + y_axis[-1]))),
                "include_tangential": include_tangential,
                "include_radial": include_radial,
            }
        )
        return []

    monkeypatch.setattr(solver_plotting, "_critical_curve_caustics", recording_critical_curve_caustics)
    helpers = solver_plotting._load_image_catalog_cutout_helpers()
    original_build_rgb_display = helpers.build_rgb_display
    build_rgb_calls: list[dict[str, object]] = []

    def recording_build_rgb_display(*args: object, **kwargs: object) -> object:
        build_rgb_calls.append(dict(kwargs))
        return original_build_rgb_display(*args, **kwargs)

    monkeypatch.setattr(helpers, "build_rgb_display", recording_build_rgb_display)

    evaluator = FakeEvaluator()
    solver_plotting._plot_image_catalog_family_cutouts(
        run_dir,
        state,
        evaluator,
        np.asarray([4.0], dtype=float),
        image_df,
        extra_image_df,
        SimpleNamespace(
            image_catalog_family_cutout_image_dir=tmp_path / "images",
            image_catalog_family_cutout_image_scale="60mas",
            image_catalog_family_cutout_rgb_q=6.5,
            image_catalog_family_cutout_rgb_stretch=0.0165,
            image_catalog_family_cutout_rgb_minimum=0.0012,
            image_catalog_family_cutout_rgb_red_gain=0.68,
            image_catalog_family_cutout_rgb_green_gain=0.75,
            image_catalog_family_cutout_rgb_blue_gain=3.5,
        ),
    )

    output = run_dir / "image_catalog_family_cutouts.pdf"
    cluster_output = run_dir / "image_catalog_family_cluster.pdf"
    assert output.exists()
    assert output.stat().st_size > 0
    assert cluster_output.exists()
    assert cluster_output.stat().st_size > 0
    assert [call.tolist() for call in conversion_calls] == [[4.0]]
    assert z_calls == [2.3]
    assert packed_z_calls == [2.3]
    assert packed_state_calls == [{"z_source": 2.3, "theta0": 5.0}]
    assert len(critical_calls) == 4
    assert all(call["model"] is fake_model for call in critical_calls)
    assert all(call["kwargs_lens"] == [{"z_source": 2.3, "theta0": 5.0}] for call in critical_calls)
    assert all(call["include_tangential"] is True for call in critical_calls)
    assert all(call["include_radial"] is True for call in critical_calls)
    centers = [call["center"] for call in critical_calls]
    for expected_center in ((0.0, 0.25), (0.0, 0.0), (1.0, 0.0), (-1.0, 0.5)):
        assert any(np.allclose(center, expected_center) for center in centers)
    assert build_rgb_calls
    assert build_rgb_calls[0]["q"] == 6.5
    assert build_rgb_calls[0]["stretch"] == 0.0165
    assert build_rgb_calls[0]["minimum"] == 0.0012
    assert build_rgb_calls[0]["channel_gains"] == {"red": 0.68, "green": 0.75, "blue": 3.5}
