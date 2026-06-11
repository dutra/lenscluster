from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_extra_image_cutouts.py"
spec = importlib.util.spec_from_file_location("plot_extra_image_cutouts", SCRIPT_PATH)
assert spec is not None
plotter = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = plotter
spec.loader.exec_module(plotter)


def _wcs_header(ra: float = 10.0, dec: float = 0.0) -> fits.Header:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50.0, 50.0]
    wcs.wcs.cdelt = np.array([-1.0 / 3600.0, 1.0 / 3600.0])
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs.to_header()


def _write_band_image(root: Path, band: str, *, cluster_token: str = "abell370", ra: float = 10.0, dec: float = 0.0) -> Path:
    path = root / cluster_token / f"hlsp_buffalo_hst_multi_{cluster_token}_{band.lower()}_v1.0_drz.fits"
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:100, :100]
    offsets = {"F435W": 5.0, "F606W": 15.0, "F814W": 30.0}
    data = offsets.get(band, 0.0) + 0.1 * xx + 0.2 * yy
    fits.PrimaryHDU(data.astype(np.float32), header=_wcs_header(ra=ra, dec=dec)).writeto(path, overwrite=True)
    return path


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


def _write_extra_images_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "family_id": 14,
                "extra_image_index": 1,
                "image_recovery_status": "extra",
                "x_model_arcsec": 0.0,
                "y_model_arcsec": 0.0,
                "z_source": 3.1,
                "effective_z_source": 2.9,
            },
            {
                "family_id": 14,
                "extra_image_index": 2,
                "image_recovery_status": "extra",
                "x_model_arcsec": -1.0,
                "y_model_arcsec": 0.0,
                "z_source": 3.1,
                "effective_z_source": 2.9,
            },
        ]
    ).to_csv(path, index=False)
    return path


def _write_image_fit_quality_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "family_id": 14,
                "image_label": "14.1",
                "x_obs_arcsec": 0.0,
                "y_obs_arcsec": 0.0,
                "x_model_arcsec": 0.2,
                "y_model_arcsec": 0.0,
                "x_model_q50": 0.25,
                "y_model_q50": 0.05,
                "image_recovery_status": "recovered",
                "image_residual_arcsec": 0.2,
                "image_residual_q50": 0.26,
                "z_source": 3.1,
                "effective_z_source": 2.9,
            },
            {
                "family_id": 14,
                "image_label": "14.2",
                "x_obs_arcsec": -1.0,
                "y_obs_arcsec": 0.0,
                "x_model_arcsec": np.nan,
                "y_model_arcsec": np.nan,
                "x_model_q50": np.nan,
                "y_model_q50": np.nan,
                "image_recovery_status": "not_recovered",
                "image_residual_arcsec": np.nan,
                "image_residual_q50": np.nan,
                "z_source": 3.1,
                "effective_z_source": 2.9,
            },
        ]
    ).to_csv(path, index=False)
    return path


def _write_master_catalog(root: Path) -> Path:
    cluster_dir = root / "a370"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    path = cluster_dir / "a370_master_catalog.csv"
    pd.DataFrame(
        [
            {
                "cluster_key": "a370",
                "object_id": "near",
                "object_source": "pagul2024",
                "catalog_sources": "pagul2024",
                "ra": 10.0 + 0.1 / 3600.0,
                "dec": 0.0,
                "zspec_best": 3.1,
                "zspec_best_source": "spec",
                "zspec_best_confidence": "secure",
                "zphot_best": 3.0,
                "mag_F435W": 24.1,
                "mag_F606W": 23.8,
                "mag_F814W": 23.3,
            },
            {
                "cluster_key": "a370",
                "object_id": "far",
                "object_source": "pagul2024",
                "catalog_sources": "pagul2024",
                "ra": 10.1,
                "dec": 0.0,
                "zspec_best": 1.0,
                "zspec_best_source": "spec",
                "zspec_best_confidence": "secure",
                "zphot_best": 1.1,
                "mag_F435W": 25.1,
                "mag_F606W": 24.8,
                "mag_F814W": 24.3,
            },
        ]
    ).to_csv(path, index=False)
    return path


def test_offsets_to_radec_matches_lenstool_convention() -> None:
    ra, dec = plotter.offsets_to_radec(3.6, -7.2, ra0_deg=10.0, dec0_deg=30.0)

    assert ra == pytest.approx(10.0 - 3.6 / (3600.0 * math.cos(math.radians(30.0))))
    assert dec == pytest.approx(30.0 - 7.2 / 3600.0)


def test_match_extra_images_selects_nearest_master_within_half_arcsec() -> None:
    extra = pd.DataFrame(
        [
            {"family_id": "1", "extra_image_index": 1, "ra": 10.0, "dec": 0.0},
            {"family_id": "1", "extra_image_index": 2, "ra": 10.01, "dec": 0.0},
        ]
    )
    master = pd.DataFrame(
        [
            {"object_id": "best", "object_source": "test", "catalog_sources": "test", "ra": 10.0 + 0.2 / 3600.0, "dec": 0.0},
            {"object_id": "worse", "object_source": "test", "catalog_sources": "test", "ra": 10.0 + 0.4 / 3600.0, "dec": 0.0},
        ]
    )

    matches = plotter.match_extra_images_to_master(extra, master, radius_arcsec=0.5)

    assert bool(matches.loc[0, "matched"])
    assert matches.loc[0, "master_object_id"] == "best"
    assert matches.loc[0, "match_separation_arcsec"] == pytest.approx(0.2)
    assert not bool(matches.loc[1, "matched"])
    assert matches.loc[1, "master_object_id"] == ""


def test_load_image_fit_quality_converts_observed_and_model_offsets(tmp_path: Path) -> None:
    image_fit_csv = _write_image_fit_quality_csv(tmp_path / "image_fit_quality.csv")
    reference = plotter.ReferenceFrame(3, 10.0, 0.0)

    image_fit = plotter.load_image_fit_quality(image_fit_csv, reference)

    assert image_fit.loc[0, "observed_ra"] == pytest.approx(10.0)
    assert image_fit.loc[0, "observed_dec"] == pytest.approx(0.0)
    assert image_fit.loc[0, "x_model_cutout_arcsec"] == pytest.approx(0.25)
    assert image_fit.loc[0, "y_model_cutout_arcsec"] == pytest.approx(0.05)
    assert image_fit.loc[0, "model_ra"] == pytest.approx(10.0 - 0.25 / 3600.0)
    assert image_fit.loc[0, "model_dec"] == pytest.approx(0.05 / 3600.0)
    assert np.isnan(image_fit.loc[1, "model_ra"])


def _label_row(
    *,
    matched: bool = True,
    z_source: float = 3.0,
    zspec: float = np.nan,
    zphot: float = np.nan,
) -> pd.Series:
    return pd.Series(
        {
            "matched": matched,
            "z_source": z_source,
            "master_zspec_best": zspec,
            "master_zphot_best": zphot,
        }
    )


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (_label_row(z_source=3.0, zspec=3.1), plotter.LABEL_COLOR_SPECZ_MATCH),
        (_label_row(z_source=3.0, zspec=1.0, zphot=3.49), plotter.LABEL_COLOR_PHOTOZ_MATCH),
        (_label_row(z_source=3.0, zspec=3.05, zphot=3.1), plotter.LABEL_COLOR_SPECZ_MATCH),
        (_label_row(matched=False, z_source=3.0, zspec=3.0, zphot=3.0), plotter.LABEL_COLOR_DEFAULT),
        (_label_row(z_source=np.nan, zspec=3.0, zphot=3.0), plotter.LABEL_COLOR_DEFAULT),
        (_label_row(z_source=3.0, zspec=3.2, zphot=3.7), plotter.LABEL_COLOR_DEFAULT),
    ],
)
def test_panel_label_color_matches_master_redshift_thresholds(row: pd.Series, expected: str) -> None:
    assert plotter._panel_label_color(row) == expected


def test_status_label_colors_are_bright_and_all_caps() -> None:
    for status in (plotter.STATUS_OBSERVED, plotter.STATUS_RECOVERED, plotter.STATUS_MISSING, plotter.STATUS_EXTRA):
        assert status == status.upper()
        assert plotter._status_label_color(status).startswith("#")


def test_recovery_cutout_panels_order_recovered_missing_then_extra(tmp_path: Path) -> None:
    reference = plotter.ReferenceFrame(3, 10.0, 0.0)
    image_fit = plotter.load_image_fit_quality(_write_image_fit_quality_csv(tmp_path / "image_fit_quality.csv"), reference)
    extra_images = plotter.load_extra_images(_write_extra_images_csv(tmp_path / "extra_images.csv"), reference)
    extra_matches = plotter.match_extra_images_to_master(extra_images, pd.DataFrame(), radius_arcsec=0.5)

    panels = plotter.build_recovery_cutout_panels(image_fit, extra_matches)

    assert panels["panel_status"].tolist() == [
        plotter.STATUS_OBSERVED,
        plotter.STATUS_RECOVERED,
        plotter.STATUS_MISSING,
        plotter.STATUS_EXTRA,
        plotter.STATUS_EXTRA,
    ]
    assert panels.loc[0, "ra"] == pytest.approx(panels.loc[0, "observed_ra"])
    assert panels.loc[1, "ra"] == pytest.approx(panels.loc[1, "model_ra"])
    assert panels.loc[2, "ra"] == pytest.approx(panels.loc[2, "observed_ra"])
    assert panels.loc[0, "detail_label"].startswith("fam 14 image 14.1")
    assert panels.loc[3, "detail_label"].startswith("fam 14 extra 1")


def test_extra_redshift_cutout_panels_include_all_extras_and_mark_matches() -> None:
    extra_matches = pd.DataFrame(
        [
            _label_row(matched=True, z_source=3.0, zspec=1.0, zphot=3.3).to_dict()
            | {"family_id": "10", "extra_image_index": 2, "x_model_arcsec": 0.0, "y_model_arcsec": 0.0},
            _label_row(matched=True, z_source=3.0, zspec=3.05, zphot=4.0).to_dict()
            | {"family_id": "10", "extra_image_index": 1, "x_model_arcsec": 1.0, "y_model_arcsec": 0.0},
            _label_row(matched=True, z_source=3.0, zspec=1.0, zphot=4.0).to_dict()
            | {"family_id": "11", "extra_image_index": 1, "x_model_arcsec": 2.0, "y_model_arcsec": 0.0},
            _label_row(matched=False, z_source=3.0, zspec=3.0, zphot=3.0).to_dict()
            | {"family_id": "11", "extra_image_index": 2, "x_model_arcsec": 3.0, "y_model_arcsec": 0.0},
        ]
    )

    panels = plotter.build_extra_redshift_cutout_panels(extra_matches)

    assert panels["panel_status"].tolist() == [plotter.STATUS_EXTRA] * 4
    assert panels["extra_image_index"].tolist() == [1, 2, 1, 2]
    assert panels["detail_label_color"].tolist() == [
        plotter.LABEL_COLOR_SPECZ_MATCH,
        plotter.LABEL_COLOR_PHOTOZ_MATCH,
        plotter.LABEL_COLOR_DEFAULT,
        plotter.LABEL_COLOR_DEFAULT,
    ]
    assert panels["status_label_color"].tolist() == panels["detail_label_color"].tolist()


def _page_family_ids(pages: list[list[pd.DataFrame]]) -> list[list[str]]:
    return [
        [str(family["family_id"].astype(str).iloc[0]) for family in page]
        for page in pages
    ]


def test_family_page_groups_keep_each_family_on_one_row() -> None:
    panels = pd.DataFrame(
        [
            {"family_id": "2", "panel_index": 0, "panel_status": plotter.STATUS_MISSING},
            {"family_id": "1", "panel_index": 1, "panel_status": plotter.STATUS_RECOVERED},
            {"family_id": "1", "panel_index": 0, "panel_status": plotter.STATUS_OBSERVED},
            {"family_id": "3", "panel_index": 0, "panel_status": plotter.STATUS_EXTRA},
        ]
    )

    pages = plotter._family_page_groups(panels, images_per_page=3)

    assert _page_family_ids(pages) == [["1", "2"], ["3"]]
    assert pages[0][0]["panel_status"].tolist() == [plotter.STATUS_OBSERVED, plotter.STATUS_RECOVERED]
    assert len(pages[0][0]) == 2
    assert len(pages[0][1]) == 1


def test_family_page_groups_do_not_split_oversized_family() -> None:
    panels = pd.DataFrame(
        [
            {"family_id": "1", "panel_index": 0},
            {"family_id": "2", "panel_index": 0},
            {"family_id": "2", "panel_index": 1},
            {"family_id": "2", "panel_index": 2},
            {"family_id": "3", "panel_index": 0},
        ]
    )

    pages = plotter._family_page_groups(panels, images_per_page=2)

    assert _page_family_ids(pages) == [["1"], ["2"], ["3"]]
    assert len(pages[1][0]) == 3


def test_stage_dir_defaults_resolve_under_tables(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage4_linearized_image_plane"
    (stage_dir / "tables").mkdir(parents=True)
    args = plotter.parse_args([str(stage_dir)])

    assert args.stage_dir == stage_dir
    assert args.output is None
    assert args.extras_output is None
    assert args.matches_output is None
    assert plotter.image_fit_quality_csv_path(stage_dir) == stage_dir / "tables" / "image_fit_quality.csv"
    assert plotter.extra_images_csv_path(stage_dir) == stage_dir / "tables" / "image_recovery_extra_images.csv"
    assert plotter.default_output_path(stage_dir) == stage_dir / "tables" / "extra_image_cutouts.pdf"
    assert plotter.default_extras_output_path(stage_dir) == stage_dir / "tables" / "extra_image_redshift_marked_cutouts.pdf"
    assert plotter.default_matches_output_path(stage_dir) == stage_dir / "tables" / "extra_image_cutout_matches.csv"


def test_stage_dir_must_be_existing_directory(tmp_path: Path) -> None:
    csv_path = tmp_path / "stage4" / "tables" / "image_recovery_extra_images.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("family_id,extra_image_index,x_model_arcsec,y_model_arcsec\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stage_dir must be an existing directory"):
        plotter.extra_images_csv_path(csv_path)


def test_run_fails_fast_when_image_fit_quality_csv_is_missing(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage4"
    stage_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing image-fit quality CSV"):
        plotter.run(stage_dir)


def test_run_fails_fast_when_extra_images_csv_is_missing(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage4"
    _write_image_fit_quality_csv(stage_dir / "tables" / "image_fit_quality.csv")

    with pytest.raises(FileNotFoundError, match="Missing extra-images CSV"):
        plotter.run(stage_dir)


def test_run_writes_synthetic_pdf_and_match_csv(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage4"
    _write_image_fit_quality_csv(stage_dir / "tables" / "image_fit_quality.csv")
    _write_extra_images_csv(stage_dir / "tables" / "image_recovery_extra_images.csv")
    par_path = _write_par(tmp_path / "a370_niemiec" / "a_sl_normal.par")
    _write_master_catalog(tmp_path / "catalogs")
    for band in plotter.DEFAULT_BANDS:
        _write_band_image(tmp_path / "images", band)

    output, matches_output, extras_output = plotter.run(
        stage_dir,
        output=tmp_path / "cutouts.pdf",
        extras_output=tmp_path / "extras.pdf",
        matches_output=tmp_path / "matches.csv",
        cluster="a370",
        catalog_root=tmp_path / "catalogs",
        image_dir=tmp_path / "images",
        cutout_size_arcsec=8.0,
        images_per_page=2,
        par_path=par_path,
    )

    assert output == tmp_path / "cutouts.pdf"
    assert output.exists()
    assert output.stat().st_size > 0
    assert extras_output == tmp_path / "extras.pdf"
    assert extras_output.exists()
    assert extras_output.stat().st_size > 0
    assert matches_output == tmp_path / "matches.csv"
    matches = pd.read_csv(matches_output)
    assert len(matches) == 2
    assert bool(matches.loc[0, "matched"])
    assert matches.loc[0, "master_object_id"] == "near"
    assert not bool(matches.loc[1, "matched"])


def test_write_extra_image_cutout_pdf_reuses_shared_rgb_display(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panels = pd.DataFrame(
        [
            {
                "family_id": 14,
                "panel_status": plotter.STATUS_OBSERVED,
                "ra": 10.0,
                "dec": 0.0,
                "observed_ra": 10.0,
                "observed_dec": 0.0,
                "model_ra": 10.0,
                "model_dec": 0.0,
            },
            {
                "family_id": 14,
                "panel_status": plotter.STATUS_RECOVERED,
                "ra": 10.0 + 0.0001,
                "dec": 0.0,
                "observed_ra": 10.0 + 0.0001,
                "observed_dec": 0.0,
                "model_ra": 10.0 + 0.0001,
                "model_dec": 0.0,
            },
        ]
    )
    paths = {band: _write_band_image(tmp_path / "images", band) for band in plotter.DEFAULT_BANDS}
    band_images = plotter.load_rgb_metadata(paths)
    display = object()
    seen_displays: list[object | None] = []

    monkeypatch.setattr(plotter, "build_rgb_display", lambda band_images_arg, *, bands: display)

    def fake_make_rgb_cutout(
        cutouts_by_band: dict[str, np.ndarray],
        bands: tuple[str, ...] = plotter.DEFAULT_BANDS,
        *,
        rgb_display: object | None = None,
    ) -> np.ndarray:
        seen_displays.append(rgb_display)
        return np.zeros((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(plotter, "make_rgb_cutout", fake_make_rgb_cutout)

    n_pages = plotter.write_extra_image_cutout_pdf(
        panels,
        band_images,
        tmp_path / "extra_cutouts.pdf",
        cutout_size_arcsec=8.0,
    )

    assert n_pages == 1
    assert seen_displays
    assert all(seen_display is display for seen_display in seen_displays)
