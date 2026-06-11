from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_literature_family_cutouts.py"
spec = importlib.util.spec_from_file_location("plot_literature_family_cutouts", SCRIPT_PATH)
assert spec is not None
plotter = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = plotter
spec.loader.exec_module(plotter)


def _write_literature_catalog(
    root: Path,
    name: str,
    rows: list[str] | None = None,
    *,
    cluster: str = "a370",
    source_slug: str = "niemiec_buffalo",
) -> Path:
    path = root / cluster / source_slug / name
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows or [
        "1.1 10.000000 0.000000 0.1 0.1 0.0 1.0 25.0",
        "1.2 10.000500 0.000000 0.1 0.1 0.0 1.0 25.0",
        "2.1 10.001000 0.000000 0.1 0.1 0.0 2.0 25.0",
    ]
    path.write_text("#REFERENCE 0\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _write_final_literature_catalog(root: Path, rows: list[str] | None = None) -> Path:
    path = root / "a370" / "niemiec_buffalo" / "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-final.dat"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows or [
        "1.1 10.000000 0.000000 1.0 Gold",
        "1.2 10.000500 0.000000 1.0 Gold",
        "2.1 10.001000 0.000000 2.0 Silver",
    ]
    path.write_text("# ID RA DEC z cat\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


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
    ra: float = 10.0,
    dec: float = 0.0,
    crpix: tuple[float, float] = (50.0, 50.0),
) -> Path:
    path = root / cluster_token / f"hlsp_buffalo_hst_multi_{cluster_token}_{band.lower()}_v1.0_drz.fits"
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:100, :100]
    band_offsets = {"F435W": 5.0, "F606W": 15.0, "F814W": 30.0}
    data = band_offsets.get(band, 0.0) + 0.1 * xx + 0.2 * yy
    fits.PrimaryHDU(data.astype(np.float32), header=_wcs_header(ra=ra, dec=dec, crpix=crpix)).writeto(path, overwrite=True)
    return path


def _flux_for_ab_mag(magnitude: float) -> float:
    return float(10 ** (-(magnitude + 48.6) / 2.5))


def _write_pagul_catalog(pagul_root: Path) -> Path:
    pagul_root.mkdir(parents=True, exist_ok=True)
    path = pagul_root / "hlsp_buffalo_hst_ir-weighted_abell370_multi_v2.0_catalog.fits"
    table = Table(
        {
            "ID": np.array(["p1", "p2"]),
            "FIELD": np.array(["A370clu", "A370clu"]),
            "ALPHA_J2000_STACK": np.array([10.0 + 0.04 / 3600.0, 10.2]),
            "DELTA_J2000_STACK": np.array([0.0, 0.0]),
            "ZSPEC": np.array([2.0, 1.0]),
            "ZSPEC_Q": np.array([4.0, 2.0]),
            "ZPDF": np.array([2.1, 1.2]),
            "ZPDF_LOW": np.array([1.9, 1.0]),
            "ZPDF_HIGH": np.array([2.3, 1.4]),
            "CHI2_RED": np.array([1.1, 1.2]),
            "FLUX_F606W": np.array([_flux_for_ab_mag(23.4), _flux_for_ab_mag(24.4)]),
            "FLUX_F814W": np.array([_flux_for_ab_mag(23.0), _flux_for_ab_mag(24.0)]),
            "FLUX_F160W": np.array([_flux_for_ab_mag(22.0), _flux_for_ab_mag(23.0)]),
            "NB_USED": np.array([6.0, 5.0]),
        }
    )
    table.write(path, overwrite=True)
    return path


def _write_rms_pagul_catalog(pagul_root: Path) -> Path:
    pagul_root.mkdir(parents=True, exist_ok=True)
    path = pagul_root / "hlsp_buffalo_hst_ir-weighted_abell370_multi_v2.0_catalog.fits"
    magnitudes = {
        "p1": {"F435W": 24.0, "F814W": 23.0, "F160W": 22.0},
        "p2": {"F435W": 25.0, "F814W": 24.2, "F160W": 23.4},
        "p3": {"F435W": 26.0, "F814W": 25.0, "F160W": 24.0},
    }
    ids = np.array(list(magnitudes))
    table = Table(
        {
            "ID": ids,
            "FIELD": np.array(["A370clu", "A370clu", "A370clu"]),
            "ALPHA_J2000_STACK": np.array(
                [
                    10.0 + 0.04 / 3600.0,
                    10.000500 + 0.04 / 3600.0,
                    10.001000 + 0.04 / 3600.0,
                ]
            ),
            "DELTA_J2000_STACK": np.array([0.0, 0.0, 0.0]),
            "ZSPEC": np.array([2.0, 2.1, 2.2]),
            "ZSPEC_Q": np.array([4.0, 4.0, 4.0]),
            "ZPDF": np.array([2.1, 2.2, 2.3]),
            "ZPDF_LOW": np.array([1.9, 2.0, 2.1]),
            "ZPDF_HIGH": np.array([2.3, 2.4, 2.5]),
            "CHI2_RED": np.array([1.1, 1.2, 1.3]),
            "FLUX_F435W": np.array([_flux_for_ab_mag(magnitudes[object_id]["F435W"]) for object_id in ids]),
            "FLUX_F814W": np.array([_flux_for_ab_mag(magnitudes[object_id]["F814W"]) for object_id in ids]),
            "FLUX_F160W": np.array([_flux_for_ab_mag(magnitudes[object_id]["F160W"]) for object_id in ids]),
            "NB_USED": np.array([6.0, 6.0, 6.0]),
        }
    )
    table.write(path, overwrite=True)
    return path


def _write_hff_catalogs(hff_root: Path, *, cluster: str = "a370") -> None:
    cluster_dir = hff_root / cluster
    cluster_dir.mkdir(parents=True, exist_ok=True)
    (cluster_dir / f"{cluster}_candidate_image_families.csv").write_text(
        "\n".join(
            [
                "cluster_key,candidate_family_id,n_images,family_probability,max_separation_arcsec,max_separation_kpc,family_z_best,family_z_method,min_specz_confidence,median_sed_rms,min_pair_score,review_flags",
                f"{cluster},{cluster}_IF00001,2,0.82,3.0,15.0,2.0,strong_specz_median,4,0.21,0.77,incomplete_strong_specz",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cluster_dir / f"{cluster}_candidate_family_members.csv").write_text(
        "\n".join(
            [
                "cluster_key,candidate_family_id,object_id,ra,dec,membership_probability,raw_probability,zspec_best,zspec_best_confidence,zspec_best_native_quality,zphot_best,n_valid_bands,object_source,catalog_sources,image_preclean_selected,image_preclean_reject_reason,image_size_arcsec,image_ellipticity,image_photoz_quality_selected,image_photoz_reject_reason,image_zphot_family,image_rank,mean_pair_score_to_family",
                f"{cluster},{cluster}_IF00001,pagul2024:p1,10.0,0.0,0.61,0.61,2.0,secure,4,2.1,15,pagul2024,pagul2024,True,,0.2,0.1,True,,2.1,1,0.77",
                f"{cluster},{cluster}_IF00001,pagul2024:p2,10.0005,0.0,0.73,0.73,2.0,secure,4,2.1,15,pagul2024,pagul2024,True,,0.2,0.1,True,,2.1,2,0.77",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cluster_dir / f"{cluster}_candidate_family_pairs.csv").write_text(
        "\n".join(
            [
                "cluster_key,left_object_id,right_object_id,separation_arcsec,separation_kpc,pair_score,specz_score,photoz_score,zphot_delta,color_score,sed_rms,n_common_bands,hard_reject_reason,redshift_relation",
                f"{cluster},pagul2024:p1,pagul2024:p2,1.8,9.0,0.77,1.0,0.8,0.1,0.9,0.21,3,,specz_photoz",
                f"{cluster},pagul2024:p1,pagul2024:p3,3.6,18.0,0.42,0.0,0.2,1.2,0.1,1.8,3,color_rms_too_large,photoz_inconsistent",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_select_literature_catalog_picks_a370_niemiec_sl_final_by_default(tmp_path: Path) -> None:
    final_path = _write_final_literature_catalog(tmp_path)
    gold_path = _write_literature_catalog(tmp_path, "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat")

    catalog = plotter.select_literature_catalog(
        tmp_path,
        cluster="a370",
        source_slug="niemiec_buffalo",
        catalog_contains=plotter.DEFAULT_CATALOG_CONTAINS,
    )

    assert catalog.path == final_path
    assert catalog.path != gold_path
    assert catalog.cluster == "a370"
    assert catalog.source_slug == "niemiec_buffalo"
    assert catalog.catalog_kind == "image"
    assert catalog.data["family_id"].astype(str).tolist() == ["1", "1", "2"]
    assert catalog.data["catalog_quality"].tolist() == ["Gold", "Gold", "Silver"]


def test_select_bergamini_catalogs_finds_staged_image_catalogs(tmp_path: Path) -> None:
    a2744_path = _write_literature_catalog(
        tmp_path,
        "obs_arcs.cat",
        cluster="a2744",
        source_slug="bergamini23",
    )
    m0416_path = _write_literature_catalog(
        tmp_path,
        "obs_arcs.cat",
        cluster="m0416",
        source_slug="bergamini22",
    )
    _write_literature_catalog(
        tmp_path,
        "mul.cat",
        cluster="as1063",
        source_slug="beauchesne23",
    )

    catalogs = plotter.select_bergamini_catalogs(tmp_path)

    assert [(catalog.cluster, catalog.source_slug, catalog.path) for catalog in catalogs] == [
        ("a2744", "bergamini23", a2744_path),
        ("m0416", "bergamini22", m0416_path),
    ]


def test_find_rgb_band_paths_requires_all_three_bands(tmp_path: Path) -> None:
    expected = {band: _write_band_image(tmp_path, band) for band in plotter.DEFAULT_BANDS}

    actual = plotter.find_rgb_band_paths(tmp_path, cluster="a370", bands=plotter.DEFAULT_BANDS)

    assert actual == expected


def test_find_rgb_band_paths_prefers_requested_image_scale(tmp_path: Path) -> None:
    expected_30mas = {band: _write_band_image(tmp_path / "30mas", band) for band in plotter.DEFAULT_BANDS}
    expected_60mas = {band: _write_band_image(tmp_path / "60mas", band) for band in plotter.DEFAULT_BANDS}

    actual_30mas = plotter.find_rgb_band_paths(tmp_path, cluster="a370", bands=plotter.DEFAULT_BANDS, image_scale="30mas")
    actual_60mas = plotter.find_rgb_band_paths(tmp_path, cluster="a370", bands=plotter.DEFAULT_BANDS, image_scale="60mas")
    actual_auto = plotter.find_rgb_band_paths(tmp_path, cluster="a370", bands=plotter.DEFAULT_BANDS, image_scale="auto")

    assert actual_30mas == expected_30mas
    assert actual_60mas == expected_60mas
    assert actual_auto == expected_30mas


def test_find_rgb_band_paths_falls_back_when_requested_scale_missing(tmp_path: Path) -> None:
    expected = {band: _write_band_image(tmp_path, band) for band in plotter.DEFAULT_BANDS}

    actual = plotter.find_rgb_band_paths(tmp_path, cluster="a370", bands=plotter.DEFAULT_BANDS, image_scale="30mas")

    assert actual == expected


def test_find_rgb_band_paths_supports_ff_sims_simulation_names(tmp_path: Path) -> None:
    root = tmp_path / "fits"
    bands = ("F435W", "F606W", "F814W")
    expected = {}
    for band in bands:
        path = root / "ares" / f"simulation_hst_{band.lower()}.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(np.ones((20, 20), dtype=np.float32), header=_wcs_header(ra=0.0, dec=0.0)).writeto(path)
        expected[band] = path

    actual_from_root = plotter.find_rgb_band_paths(root, cluster="ares", bands=bands, image_scale="auto")
    actual_from_cluster_dir = plotter.find_rgb_band_paths(root / "ares", cluster="ares", bands=bands, image_scale="auto")

    assert actual_from_root == expected
    assert actual_from_cluster_dir == expected


def test_find_rgb_band_paths_raises_download_command_when_images_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        plotter.find_rgb_band_paths(tmp_path, cluster="a370", bands=plotter.DEFAULT_BANDS, image_scale="30mas")

    message = str(exc_info.value)
    assert "Missing BUFFALO RGB FITS image(s)" in message
    assert "/home/dutra/.conda/envs/lenstronomy/bin/python download_catalogs.py --catalog buffalo-images" in message
    assert "--image-scale 30mas" in message


def test_parse_args_exposes_image_scale() -> None:
    args = plotter.parse_args(["--image-scale", "30mas"])

    assert args.image_scale == "30mas"


def test_family_ids_order_by_best_quality_then_catalog_order() -> None:
    data = plotter.pd.DataFrame(
        {
            "family_id": ["1", "1", "2", "3", "4", "4", "5"],
            "catalog_quality": ["Gold", "Bronze", "Silver", "Platinum", "Bronze", "Silver", ""],
        }
    )

    assert plotter._family_ids_in_catalog_order(data) == ["3", "1", "2", "4", "5"]


def test_quality_helpers_map_known_and_unknown_styles() -> None:
    assert plotter._quality_text("Platinum") == "Platinum"
    assert plotter._quality_text("gold") == "Gold"
    assert plotter._quality_text("unknown") == ""
    assert plotter._quality_color("Gold") == plotter.QUALITY_COLORS["gold"]
    assert plotter._quality_color("Silver") == plotter.QUALITY_COLORS["silver"]
    assert plotter._quality_color("Bronze") == plotter.QUALITY_COLORS["bronze"]
    assert plotter._quality_color("Platinum") == plotter.QUALITY_COLORS["platinum"]
    assert plotter._quality_color("") == plotter.UNKNOWN_QUALITY_COLOR
    bbox = plotter._quality_text_bbox("Bronze")
    assert bbox["edgecolor"] == plotter.QUALITY_COLORS["bronze"]
    assert bbox["facecolor"] == "black"


def test_catalog_with_pagul_match_info_labels_matched_and_unmatched_rows(tmp_path: Path) -> None:
    _write_literature_catalog(
        tmp_path / "literature",
        "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat",
        rows=[
            "1.1 10.000000 0.000000 0.1 0.1 0.0 1.0 25.0",
            "1.2 10.001000 0.000000 0.1 0.1 0.0 1.0 25.0",
        ],
    )
    _write_pagul_catalog(tmp_path / "pagul")
    catalog = plotter.select_literature_catalog(tmp_path / "literature", catalog_contains="sl-gold")

    annotated = plotter._catalog_with_pagul_match_info(
        catalog,
        pagul_root=tmp_path / "pagul",
        match_radius_arcsec=0.5,
    )

    assert annotated.data["pagul_catalog_present"].tolist() == [True, True]
    assert annotated.data["pagul_matched"].tolist() == [True, False]
    assert annotated.data["pagul_object_id"].tolist() == ["p1", ""]
    matched_label = plotter._format_pagul_match(annotated.data.iloc[0])
    assert matched_label.startswith("Pagul: yes p1 (")
    assert "F814W-F160W=1" in matched_label
    assert "F606W-F814W=0.4" in matched_label
    assert "specz=2" in matched_label
    assert "photoz=2.1" in matched_label
    assert "crms=" not in matched_label
    assert all(len(line) <= 28 for line in matched_label.splitlines())
    assert plotter._format_pagul_match(annotated.data.iloc[1]) == "Pagul: no"
    assert np.isnan(annotated.data.loc[0, "pagul_color_rms"])
    assert np.isnan(annotated.data.loc[0, "pagul_family_color_rms"])
    assert "crms=na" in plotter._format_family_label("1", annotated.data, 5)


def test_catalog_with_pagul_match_info_labels_missing_catalog(tmp_path: Path) -> None:
    _write_literature_catalog(tmp_path / "literature", "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat")
    catalog = plotter.select_literature_catalog(tmp_path / "literature", catalog_contains="sl-gold")

    annotated = plotter._catalog_with_pagul_match_info(catalog, pagul_root=tmp_path / "missing-pagul")

    assert annotated.data["pagul_catalog_present"].tolist() == [False, False, False]
    assert annotated.data["pagul_matched"].tolist() == [False, False, False]
    assert plotter._format_pagul_match(annotated.data.iloc[0]) == "Pagul: no catalog"


def test_pagul_match_label_formats_missing_second_color_as_na(tmp_path: Path) -> None:
    _write_literature_catalog(
        tmp_path / "literature",
        "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat",
        rows=["1.1 10.000000 0.000000 0.1 0.1 0.0 1.0 25.0"],
    )
    _write_pagul_catalog(tmp_path / "pagul")
    catalog = plotter.select_literature_catalog(tmp_path / "literature", catalog_contains="sl-gold")
    annotated = plotter._catalog_with_pagul_match_info(catalog, pagul_root=tmp_path / "pagul")
    annotated.data.loc[0, plotter._pagul_second_color_column_name()] = np.nan

    label = plotter._format_pagul_match(annotated.data.iloc[0])

    assert "F814W-F160W=1" in label
    assert "F606W-F814W=na" in label
    assert "crms=" not in label


def test_hff_diagnostics_labels_selected_nonselected_and_pair_rejects(tmp_path: Path) -> None:
    _write_literature_catalog(
        tmp_path / "literature",
        "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat",
        rows=[
            "1.1 10.000000 0.000000 0.1 0.1 0.0 1.0 25.0",
            "1.2 10.000500 0.000000 0.1 0.1 0.0 1.0 25.0",
            "1.3 10.001000 0.000000 0.1 0.1 0.0 1.0 25.0",
        ],
    )
    _write_rms_pagul_catalog(tmp_path / "pagul")
    _write_hff_catalogs(tmp_path / "hff")
    catalog = plotter.select_literature_catalog(tmp_path / "literature", catalog_contains="sl-gold")
    annotated = plotter._catalog_with_pagul_match_info(catalog, pagul_root=tmp_path / "pagul")
    annotated = plotter._catalog_with_hff_diagnostics(annotated, hff_root=tmp_path / "hff")

    assert annotated.data.loc[0, "hff_object_id"] == "pagul2024:p1"
    assert annotated.data.loc[0, "hff_candidate_family_id"] == "a370_IF00001"
    selected_label = plotter._format_hff_diagnostics(annotated.data.iloc[0])
    assert "HFF: a370_IF00001 P=0.82 mem=0.61" in selected_label
    assert "minpair=0.77 hffcrms=0.21" in selected_label
    assert "bestpair=0.77" in selected_label
    assert "flags=incomplete_strong_specz" in selected_label

    nonselected_label = plotter._format_hff_diagnostics(annotated.data.iloc[2])
    assert "HFF: no family" in nonselected_label
    assert "bestpair=0.42" in nonselected_label
    assert "reject=color_rms_too_large" in nonselected_label


def test_hff_diagnostics_missing_catalog_is_nonfatal(tmp_path: Path) -> None:
    _write_literature_catalog(tmp_path / "literature", "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat")
    _write_pagul_catalog(tmp_path / "pagul")
    catalog = plotter.select_literature_catalog(tmp_path / "literature", catalog_contains="sl-gold")
    annotated = plotter._catalog_with_pagul_match_info(catalog, pagul_root=tmp_path / "pagul")
    annotated = plotter._catalog_with_hff_diagnostics(annotated, hff_root=tmp_path / "missing-hff")

    assert annotated.data["hff_catalog_present"].tolist() == [False, False, False]
    assert plotter._format_hff_diagnostics(annotated.data.iloc[0]) == "HFF: no catalog"


def test_pagul_color_rms_ignores_absolute_magnitude_offset(tmp_path: Path) -> None:
    _write_literature_catalog(
        tmp_path / "literature",
        "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat",
        rows=[
            "1.1 10.000000 0.000000 0.1 0.1 0.0 1.0 25.0",
            "1.2 10.000500 0.000000 0.1 0.1 0.0 1.0 25.0",
            "1.3 10.001000 0.000000 0.1 0.1 0.0 1.0 25.0",
        ],
    )
    _write_rms_pagul_catalog(tmp_path / "pagul")
    catalog = plotter.select_literature_catalog(tmp_path / "literature", catalog_contains="sl-gold")
    annotated = plotter._catalog_with_pagul_match_info(catalog, pagul_root=tmp_path / "pagul")

    expected_rms = float(np.sqrt((0.2**2 + 0.0 + 0.2**2) / 3.0))

    np.testing.assert_allclose(plotter._pair_color_rms(annotated.data.iloc[0], annotated.data.iloc[1]), expected_rms)
    np.testing.assert_allclose(plotter._pair_color_rms(annotated.data.iloc[0], annotated.data.iloc[2]), 0.0, atol=1.0e-12)


def test_pagul_color_rms_columns_use_image_and_family_medians(tmp_path: Path) -> None:
    _write_literature_catalog(
        tmp_path / "literature",
        "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat",
        rows=[
            "1.1 10.000000 0.000000 0.1 0.1 0.0 1.0 25.0",
            "1.2 10.000500 0.000000 0.1 0.1 0.0 1.0 25.0",
            "1.3 10.001000 0.000000 0.1 0.1 0.0 1.0 25.0",
        ],
    )
    _write_rms_pagul_catalog(tmp_path / "pagul")
    catalog = plotter.select_literature_catalog(tmp_path / "literature", catalog_contains="sl-gold")
    annotated = plotter._catalog_with_pagul_match_info(catalog, pagul_root=tmp_path / "pagul")
    expected_rms = float(np.sqrt((0.2**2 + 0.0 + 0.2**2) / 3.0))

    np.testing.assert_allclose(annotated.data["pagul_color_rms"], [expected_rms / 2.0, expected_rms, expected_rms / 2.0])
    np.testing.assert_allclose(annotated.data["pagul_family_color_rms"], [expected_rms, expected_rms, expected_rms])
    assert "crms=0.16" in plotter._format_family_label("1", annotated.data, 5)
    assert "crms=" not in plotter._format_pagul_match(annotated.data.iloc[0])


def test_write_family_cutout_pdf_renders_synthetic_multipage_rgb(tmp_path: Path) -> None:
    _write_literature_catalog(
        tmp_path / "literature",
        "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat",
        rows=[
            "1.1 10.000000 0.000000 0.1 0.1 0.0 1.0 25.0",
            "1.2 10.000500 0.000000 0.1 0.1 0.0 1.0 25.0",
            "2.1 10.001000 0.000000 0.1 0.1 0.0 2.0 25.0",
        ],
    )
    catalog = plotter.select_literature_catalog(tmp_path / "literature", catalog_contains="sl-gold")
    paths = {band: _write_band_image(tmp_path / "images", band) for band in plotter.DEFAULT_BANDS}
    band_images = plotter.load_rgb_metadata(paths)
    output = tmp_path / "cutouts.pdf"

    n_pages = plotter.write_family_cutout_pdf(
        catalog,
        band_images,
        output,
        cutout_size_arcsec=8.0,
        families_per_page=1,
    )

    assert n_pages == 3
    assert output.exists()
    assert output.stat().st_size > 0


def test_make_rgb_cutout_delegates_to_shared_natural_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    yy, xx = np.mgrid[:12, :12]
    cutouts = {
        "F435W": (0.3 * xx + 0.2 * yy + 5.0).astype(float),
        "F606W": (0.5 * xx + 0.1 * yy + 15.0).astype(float),
        "F814W": (0.4 * xx + 0.6 * yy + 30.0).astype(float),
    }
    expected_rgb = np.full((12, 12, 3), 0.37, dtype=float)
    calls: list[dict[str, Any]] = []
    display = object()

    def fake_make_natural_rgb(cutouts_arg: dict[str, np.ndarray], *, bands: tuple[str, ...], display: object | None) -> np.ndarray:
        calls.append({"cutouts": cutouts_arg, "bands": bands, "display": display})
        return expected_rgb

    monkeypatch.setattr(plotter, "make_natural_rgb", fake_make_natural_rgb)

    actual = plotter.make_rgb_cutout(cutouts, rgb_display=display)

    np.testing.assert_allclose(actual, expected_rgb)
    assert calls
    assert calls[0]["bands"] == plotter.DEFAULT_BANDS
    assert calls[0]["display"] is display
    assert calls[0]["cutouts"] is cutouts


def test_write_family_cutout_pdf_reuses_shared_rgb_display(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_path = _write_literature_catalog(
        tmp_path / "literature",
        "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat",
        rows=[
            "1.1 10.000000 0.000000 0.1 0.1 0.0 1.0 25.0",
            "1.2 10.000500 0.000000 0.1 0.1 0.0 1.0 25.0",
        ],
    )
    catalog = plotter.select_literature_catalog(tmp_path / "literature", catalog_contains="sl-gold")
    paths = {band: _write_band_image(tmp_path / "images", band) for band in plotter.DEFAULT_BANDS}
    band_images = plotter.load_rgb_metadata(paths)
    output = tmp_path / "cutouts.pdf"
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

    n_pages = plotter.write_family_cutout_pdf(catalog, band_images, output, cutout_size_arcsec=8.0)

    assert catalog_path.exists()
    assert n_pages == 2
    assert output.exists()
    assert seen_displays
    assert all(seen_display is display for seen_display in seen_displays)


def test_write_family_cutout_pdf_detail_grid_ignores_legacy_image_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_literature_catalog(
        tmp_path / "literature",
        "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat",
        rows=[
            "1.1 10.000000 0.000000 0.1 0.1 0.0 1.0 25.0",
            "1.2 10.000500 0.000000 0.1 0.1 0.0 1.0 25.0",
            "1.3 10.001000 0.000000 0.1 0.1 0.0 1.0 25.0",
            "1.4 10.001500 0.000000 0.1 0.1 0.0 1.0 25.0",
        ],
    )
    catalog = plotter.select_literature_catalog(tmp_path / "literature", catalog_contains="sl-gold")
    paths = {band: _write_band_image(tmp_path / "images", band) for band in plotter.DEFAULT_BANDS}
    band_images = plotter.load_rgb_metadata(paths)
    detail_labels: list[str] = []

    def fake_detail_panel(
        ax: object,
        band_images_arg: dict[str, object],
        bands: tuple[str, ...],
        rgb_display: object,
        image_row: plotter.pd.Series,
        *,
        cutout_size_arcsec: float,
    ) -> None:
        detail_labels.append(str(image_row.get("literature_id")))

    monkeypatch.setattr(plotter, "_draw_literature_detail_panel", fake_detail_panel)

    n_pages = plotter.write_family_cutout_pdf(
        catalog,
        band_images,
        tmp_path / "cutouts.pdf",
        cutout_size_arcsec=8.0,
        max_images_per_family=1,
    )

    assert n_pages == 2
    assert detail_labels == ["1.1", "1.2", "1.3", "1.4"]


def test_extract_band_cutout_clamps_edge_window_to_real_pixels(tmp_path: Path) -> None:
    path = _write_band_image(tmp_path / "images", "F814W", crpix=(2.0, 2.0))
    image = plotter.load_band_metadata("F814W", path)
    coord = plotter.SkyCoord(ra=10.0 * plotter.u.deg, dec=0.0 * plotter.u.deg, frame="icrs")

    cutout = plotter.extract_band_cutout(image, coord, cutout_size_arcsec=8.0)

    yy, xx = np.mgrid[:8, :8]
    expected = 30.0 + 0.1 * xx + 0.2 * yy
    assert cutout.shape == (8, 8)
    assert np.isfinite(cutout).all()
    np.testing.assert_allclose(cutout, expected.astype(np.float32), atol=1.0e-6)


def test_cutout_marker_uses_clamped_edge_window_origin(tmp_path: Path) -> None:
    path = _write_band_image(tmp_path / "images", "F814W", crpix=(2.0, 2.0))
    image = plotter.load_band_metadata("F814W", path)
    coord = plotter.SkyCoord(ra=10.0 * plotter.u.deg, dec=0.0 * plotter.u.deg, frame="icrs")

    x, y, radius = plotter._cutout_pixel_position(
        image,
        coord,
        coord,
        cutout_size_arcsec=8.0,
    )

    assert radius == pytest.approx(0.2)
    assert x == pytest.approx(1.0, abs=1.0e-3)
    assert y == pytest.approx(1.0, abs=1.0e-3)


def test_cutout_layout_uses_black_tight_page_style() -> None:
    fig, ax = plotter.plt.subplots(1, 1, figsize=plotter._figure_size(1, 1))
    try:
        plotter._style_cutout_figure(fig)
        plotter._style_cutout_axis(ax)

        assert fig.get_facecolor() == plotter.matplotlib.colors.to_rgba("black")
        assert ax.get_facecolor() == plotter.matplotlib.colors.to_rgba("black")
        assert fig.subplotpars.left == pytest.approx(0.005)
        assert fig.subplotpars.right == pytest.approx(0.995)
        assert fig.subplotpars.wspace == pytest.approx(0.015)
        assert fig.subplotpars.hspace == pytest.approx(0.015)
        assert plotter._figure_size(2, 5) == (16.0, 6.4)
    finally:
        plotter.plt.close(fig)


def test_write_family_cutout_pdf_saves_black_tight_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_literature_catalog(
        tmp_path / "literature",
        "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat",
        rows=["1.1 10.000000 0.000000 0.1 0.1 0.0 1.0 25.0"],
    )
    catalog = plotter.select_literature_catalog(tmp_path / "literature", catalog_contains="sl-gold")
    paths = {band: _write_band_image(tmp_path / "images", band) for band in plotter.DEFAULT_BANDS}
    band_images = plotter.load_rgb_metadata(paths)
    save_calls: list[dict[str, Any]] = []
    original_savefig = plotter.PdfPages.savefig

    def spy_savefig(self: object, fig: object, *args: object, **kwargs: object) -> object:
        save_calls.append(kwargs.copy())
        return original_savefig(self, fig, *args, **kwargs)

    monkeypatch.setattr(plotter.PdfPages, "savefig", spy_savefig)

    plotter.write_family_cutout_pdf(
        catalog,
        band_images,
        tmp_path / "cutouts.pdf",
        cutout_size_arcsec=8.0,
        families_per_page=1,
    )

    assert save_calls
    assert save_calls[0]["facecolor"] == plotter.matplotlib.colors.to_rgba("black")
    assert save_calls[0]["bbox_inches"] == "tight"
    assert save_calls[0]["pad_inches"] == 0.02
    assert save_calls[0]["dpi"] == plotter.CUTOUT_FIGURE_DPI


def test_run_renders_pdf_with_pagul_match_info_enabled(tmp_path: Path) -> None:
    _write_literature_catalog(
        tmp_path / "literature",
        "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat",
        rows=[
            "1.1 10.000000 0.000000 0.1 0.1 0.0 1.0 25.0",
            "1.2 10.001000 0.000000 0.1 0.1 0.0 1.0 25.0",
        ],
    )
    _write_pagul_catalog(tmp_path / "pagul")
    for band in plotter.DEFAULT_BANDS:
        _write_band_image(tmp_path / "images", band)
    output = tmp_path / "cutouts.pdf"

    result = plotter.run(
        literature_root=tmp_path / "literature",
        image_dir=tmp_path / "images",
        pagul_root=tmp_path / "pagul",
        output=output,
        catalog_contains="sl-gold",
        cutout_size_arcsec=8.0,
        families_per_page=1,
    )

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_pagul_marker_uses_matched_pagul_coordinate_and_radius(tmp_path: Path) -> None:
    path = _write_band_image(tmp_path / "images", "F814W")
    image = plotter.load_band_metadata("F814W", path)
    center = plotter.SkyCoord(ra=10.0 * plotter.u.deg, dec=0.0 * plotter.u.deg, frame="icrs")
    pagul = plotter.SkyCoord(ra=(10.0 + 0.04 / 3600.0) * plotter.u.deg, dec=0.0 * plotter.u.deg, frame="icrs")

    x, y, radius = plotter._cutout_pixel_position(
        image,
        center,
        pagul,
        cutout_size_arcsec=8.0,
    )

    assert radius == pytest.approx(0.2)
    assert x == pytest.approx(3.96, abs=1.0e-2)
    assert y == pytest.approx(4.0, abs=1.0e-2)


def test_literature_and_pagul_markers_and_legend_draw_expected_artists(tmp_path: Path) -> None:
    path = _write_band_image(tmp_path / "images", "F814W")
    image = plotter.load_band_metadata("F814W", path)
    center = plotter.SkyCoord(ra=10.0 * plotter.u.deg, dec=0.0 * plotter.u.deg, frame="icrs")
    image_row = plotter.pd.Series(
        {
            "pagul_matched": True,
            "pagul_ra": 10.0 + 0.04 / 3600.0,
            "pagul_dec": 0.0,
        }
    )
    fig, ax = plotter.plt.subplots(1, 1)
    try:
        plotter._draw_literature_marker(ax, image, center, cutout_size_arcsec=8.0, rendered_shape=(8, 8))
        plotter._draw_pagul_marker(ax, image, image_row, center, cutout_size_arcsec=8.0, rendered_shape=(8, 8))
        plotter._draw_marker_legend(ax)

        assert len(ax.patches) == 4
        literature_marker = ax.patches[0]
        pagul_marker = ax.patches[1]
        literature_legend = ax.patches[2]
        pagul_legend = ax.patches[3]
        assert literature_marker.center == pytest.approx((4.0, 4.0))
        assert pagul_marker.center == pytest.approx((3.96, 4.0), abs=1.0e-2)
        assert literature_marker.get_edgecolor() == plotter.matplotlib.colors.to_rgba(
            plotter.LITERATURE_MARKER_COLOR,
            plotter.LITERATURE_MARKER_ALPHA,
        )
        assert pagul_marker.get_edgecolor() == plotter.matplotlib.colors.to_rgba(
            plotter.PAGUL_MARKER_COLOR,
            plotter.PAGUL_MARKER_ALPHA,
        )
        assert pagul_marker.get_linestyle() == "--"
        assert pagul_marker.get_zorder() > literature_marker.get_zorder()
        labels = {text.get_text(): text for text in ax.texts if text.get_text() in {"lit", "Pagul"}}
        assert set(labels) == {"lit", "Pagul"}
        assert literature_legend.get_edgecolor() == plotter.matplotlib.colors.to_rgba(
            plotter.LITERATURE_MARKER_COLOR,
            plotter.LITERATURE_MARKER_ALPHA,
        )
        assert pagul_legend.get_edgecolor() == plotter.matplotlib.colors.to_rgba(
            plotter.PAGUL_MARKER_COLOR,
            plotter.PAGUL_MARKER_ALPHA,
        )
        assert pagul_legend.get_linestyle() == "--"
        assert pagul_legend.get_zorder() > literature_legend.get_zorder()
        assert literature_legend.center[1] == pytest.approx(labels["lit"].get_position()[1])
        assert pagul_legend.center[1] == pytest.approx(labels["Pagul"].get_position()[1])
    finally:
        plotter.plt.close(fig)


def test_write_family_cutout_pdf_omits_bottom_left_family_label(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_literature_catalog(
        tmp_path / "literature",
        "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-gold.dat",
        rows=["1.1 10.000000 0.000000 0.1 0.1 0.0 1.0 25.0"],
    )
    catalog = plotter.select_literature_catalog(tmp_path / "literature", catalog_contains="sl-gold")
    paths = {band: _write_band_image(tmp_path / "images", band) for band in plotter.DEFAULT_BANDS}
    band_images = plotter.load_rgb_metadata(paths)
    text_calls: list[dict[str, Any]] = []
    original_text = plotter.matplotlib.axes.Axes.text

    def spy_text(self: object, x: float, y: float, s: str, *args: object, **kwargs: object) -> object:
        text_calls.append({"x": x, "y": y, "s": s, **kwargs})
        return original_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(plotter.matplotlib.axes.Axes, "text", spy_text)

    plotter.write_family_cutout_pdf(
        catalog,
        band_images,
        tmp_path / "cutouts.pdf",
        cutout_size_arcsec=8.0,
        families_per_page=1,
    )

    assert any(str(call["s"]).startswith("Family ") for call in text_calls)
    assert not any(str(call["s"]).startswith("Family ") and call["x"] == 0.04 and call["y"] == 0.06 for call in text_calls)


def test_parse_args_keeps_a370_niemiec_pagul_defaults() -> None:
    args = plotter.parse_args([])

    assert args.cluster == "a370"
    assert args.source_slug == "niemiec_buffalo"
    assert args.catalog_contains == "sl-final"
    assert args.pagul_root == Path("data") / "Pagul2024"
    assert args.hff_catalog_root == Path("results") / "hff_master_catalogs"
    assert args.pagul_match_radius_arcsec == 0.5
    assert args.no_pagul_match_info is False
    assert args.no_hff_diagnostics is False


def test_run_bergamini_renders_one_pdf_per_catalog(tmp_path: Path) -> None:
    rows = [
        "1.1 10.000000 0.000000 0.1 0.1 0.0 1.0 25.0",
        "1.2 10.000500 0.000000 0.1 0.1 0.0 1.0 25.0",
        "2.1 10.001000 0.000000 0.1 0.1 0.0 2.0 25.0",
    ]
    _write_literature_catalog(
        tmp_path / "literature",
        "obs_arcs.cat",
        rows=rows,
        cluster="a2744",
        source_slug="bergamini23",
    )
    _write_literature_catalog(
        tmp_path / "literature",
        "obs_arcs.cat",
        rows=rows,
        cluster="m0416",
        source_slug="bergamini22",
    )
    for cluster_token in ("abell2744", "macs0416"):
        for band in plotter.DEFAULT_BANDS:
            _write_band_image(tmp_path / "images", band, cluster_token=cluster_token)
    output_dir = tmp_path / "plots"

    outputs = plotter.run_bergamini(
        literature_root=tmp_path / "literature",
        image_dir=tmp_path / "images",
        output=output_dir,
        cutout_size_arcsec=8.0,
        families_per_page=1,
    )

    assert outputs == [
        output_dir / "a2744_bergamini23_obs_arcs_family_cutouts.pdf",
        output_dir / "m0416_bergamini22_obs_arcs_family_cutouts.pdf",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
