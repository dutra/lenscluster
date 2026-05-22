from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_literature_family_diagnostics.py"
spec = importlib.util.spec_from_file_location("analyze_literature_family_diagnostics", SCRIPT_PATH)
assert spec is not None
analyzer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = analyzer
spec.loader.exec_module(analyzer)


def _flux_for_ab_mag(magnitude: float) -> float:
    return float(10 ** (-(magnitude + 48.6) / 2.5))


def _base_master_row(object_id: str, ra: float, *, zspec: float = 2.0, zphot: float = 2.1) -> dict[str, object]:
    row: dict[str, object] = {
        "object_id": object_id,
        "object_source": "pagul2024",
        "catalog_sources": "pagul2024",
        "ra": ra,
        "dec": 0.0,
        "zspec_best": zspec,
        "zspec_best_source": "pagul2024",
        "zspec_best_confidence": "secure",
        "zspec_best_confidence_rank": 3.0,
        "zspec_best_native_quality": 4.0,
        "zspec_conflict": False,
        "zphot_best": zphot,
        "pagul_zpdf_low": zphot - 0.2,
        "pagul_zpdf_high": zphot + 0.2,
    }
    for band in analyzer.MAG_BANDS:
        row[f"mag_{band}"] = np.nan
    return row


def _write_synthetic_pagul_catalog(pagul_root: Path) -> Path:
    pagul_root.mkdir(parents=True, exist_ok=True)
    path = pagul_root / "hlsp_buffalo_hst_ir-weighted_abell370_multi_v2.0_catalog.fits"
    table = Table(
        {
            "ID": np.array(["p1", "p2", "p3"]),
            "FIELD": np.array(["A370clu", "A370clu", "A370clu"]),
            "ALPHA_J2000_STACK": np.array([10.0 + 0.04 / 3600.0, 10.0 + 0.20 / 3600.0, 10.1]),
            "DELTA_J2000_STACK": np.array([0.0, 0.0, 0.0]),
            "ZSPEC": np.array([2.000, 2.003, 1.5]),
            "ZSPEC_Q": np.array([4.0, 4.0, 2.0]),
            "ZPDF": np.array([2.11, 2.16, 1.7]),
            "ZPDF_LOW": np.array([1.91, 1.96, 1.5]),
            "ZPDF_HIGH": np.array([2.31, 2.36, 1.9]),
            "CHI2_RED": np.array([1.1, 1.2, 1.3]),
            "FLUX_F160W": np.array([_flux_for_ab_mag(22.0), _flux_for_ab_mag(22.2), _flux_for_ab_mag(23.0)]),
            "NB_USED": np.array([6.0, 6.0, 5.0]),
        }
    )
    table.write(path, overwrite=True)
    return path


def test_load_pagul_catalog_uses_zpdf_for_photoz(tmp_path: Path) -> None:
    _write_synthetic_pagul_catalog(tmp_path)

    pagul = analyzer.load_pagul_catalog(tmp_path, "a370")

    assert len(pagul) == 3
    assert pagul["object_id"].tolist() == ["p1", "p2", "p3"]
    np.testing.assert_allclose(pagul.loc[0, "zphot"], 2.11)
    np.testing.assert_allclose(pagul.loc[0, "mag_f160w"], 22.0, atol=1.0e-8)


def test_literature_images_match_direct_pagul_catalog_with_zpdf_photoz(tmp_path: Path) -> None:
    _write_synthetic_pagul_catalog(tmp_path)
    pagul = analyzer.load_pagul_catalog(tmp_path, "a370")
    literature = pd.DataFrame(
        [
            {"literature_id": "1.1", "family_id": "1", "image_id": "1", "ra": 10.0 + 0.05 / 3600.0, "dec": 0.0},
            {"literature_id": "1.2", "family_id": "1", "image_id": "2", "ra": 10.0 + 0.24 / 3600.0, "dec": 0.0},
            {"literature_id": "2.1", "family_id": "2", "image_id": "1", "ra": 10.0 + 2.0 / 3600.0, "dec": 0.0},
        ]
    )

    matches = analyzer.match_literature_images_to_pagul(literature, pagul, radius_arcsec=0.5)

    assert matches["pagul_matched"].tolist() == [True, True, False]
    assert matches["pagul_object_id"].tolist() == ["p1", "p2", ""]
    np.testing.assert_allclose(matches.loc[0, "pagul_zphot"], 2.11)


def test_pagul_zphot_background_cut_summary_uses_zpdf() -> None:
    matches = pd.DataFrame(
        [
            {"cluster": "a370", "pagul_matched": True, "pagul_zphot": 0.40},
            {"cluster": "a370", "pagul_matched": True, "pagul_zphot": 0.46},
            {"cluster": "a370", "pagul_matched": True, "pagul_zphot": 0.60},
            {"cluster": "a370", "pagul_matched": True, "pagul_zphot": np.nan},
            {"cluster": "a370", "pagul_matched": False, "pagul_zphot": 0.70},
        ]
    )

    summary = analyzer.compute_pagul_zphot_cut_summary(matches)

    row = summary.loc[summary["cluster"].eq("a370")].iloc[0]
    np.testing.assert_allclose(row["z_lens"], analyzer.CLUSTER_Z_LENS["a370"])
    np.testing.assert_allclose(row["zphot_cut"], analyzer.CLUSTER_Z_LENS["a370"] + analyzer.BACKGROUND_Z_MARGIN)
    assert row["matched_with_zpdf"] == 3
    assert row["above_cut"] == 1
    assert row["below_or_equal_cut"] == 2
    np.testing.assert_allclose(row["fraction_above_cut"], 1.0 / 3.0)
    np.testing.assert_allclose(row["median_zpdf"], 0.46)

    all_row = summary.loc[summary["cluster"].eq("all")].iloc[0]
    assert all_row["matched_with_zpdf"] == 3
    assert all_row["above_cut"] == 1


def test_pagul_specz_coverage_counts_only_matched_positive_zspec() -> None:
    rows = []
    for index, (matched, zspec) in enumerate([(True, 2.0), (True, np.nan), (True, 0.0), (False, 4.0)]):
        rows.append(
            {
                "cluster": "a370",
                "source_slug": "src",
                "source_id": "a370/src/mul.cat",
                "source_path": "mul.cat",
                "literature_row_index": index,
                "pagul_matched": matched,
                "pagul_zspec": zspec,
            }
        )
    matches = pd.DataFrame(rows)

    summary = analyzer.compute_pagul_specz_coverage_summary(matches)

    row = summary.loc[summary["scope"].eq("source")].iloc[0]
    assert row["n_literature_images"] == 4
    assert row["n_pagul_matched_images"] == 3
    assert row["n_pagul_matched_with_zspec"] == 1
    assert row["n_pagul_matched_without_zspec"] == 2
    np.testing.assert_allclose(row["pagul_specz_fraction_among_matches"], 1.0 / 3.0)

    all_row = summary.loc[summary["scope"].eq("all")].iloc[0]
    assert all_row["n_pagul_matched_images"] == 3
    assert all_row["n_pagul_matched_with_zspec"] == 1


def test_literature_images_match_closest_master_within_half_arcsec_and_flag_duplicates() -> None:
    master = pd.DataFrame(
        [
            _base_master_row("m1", 10.0 + 0.05 / 3600.0),
            _base_master_row("m2", 10.0 + 0.25 / 3600.0),
        ]
    )
    literature = pd.DataFrame(
        [
            {"literature_id": "1.1", "family_id": "1", "image_id": "1", "ra": 10.0 + 0.04 / 3600.0, "dec": 0.0, "catalog_z": 2.0, "catalog_mag": 25.0},
            {"literature_id": "1.2", "family_id": "1", "image_id": "2", "ra": 10.0 + 0.20 / 3600.0, "dec": 0.0, "catalog_z": 2.0, "catalog_mag": 25.0},
            {"literature_id": "2.1", "family_id": "2", "image_id": "1", "ra": 10.0 + 0.06 / 3600.0, "dec": 0.0, "catalog_z": 2.0, "catalog_mag": 25.0},
            {"literature_id": "3.1", "family_id": "3", "image_id": "1", "ra": 10.0 + 2.00 / 3600.0, "dec": 0.0, "catalog_z": 2.0, "catalog_mag": 25.0},
        ]
    )

    matches = analyzer.match_literature_images_to_master(literature, master, radius_arcsec=0.5)

    assert matches["matched"].tolist() == [True, True, True, False]
    assert matches["master_object_id"].tolist() == ["m1", "m2", "m1", ""]
    np.testing.assert_allclose(matches.loc[0, "match_separation_arcsec"], 0.01, atol=1.0e-5)
    assert matches.loc[0, "duplicate_master_match"]
    assert matches.loc[2, "duplicate_master_match"]
    assert not matches.loc[1, "duplicate_master_match"]


def test_pair_metrics_capture_specz_photoz_pdf_and_constant_offset_colors() -> None:
    matches = pd.DataFrame(
        [
            {
                "cluster": "a370",
                "source_slug": "src",
                "source_id": "a370/src/mul.cat",
                "source_path": "mul.cat",
                "literature_family_id": "1",
                "literature_id": "1.1",
                "literature_ra": 10.0,
                "literature_dec": 0.0,
                "literature_z": 2.0,
                "matched": True,
                "match_separation_arcsec": 0.1,
                "duplicate_master_match": False,
                "master_object_id": "m1",
                "master_zspec_best": 2.000,
                "master_zspec_best_confidence_rank": 3.0,
                "master_zspec_best_confidence": "secure",
                "master_zspec_best_source": "pagul2024",
                "master_zphot_best": 2.10,
                "master_zpdf_low": 1.90,
                "master_zpdf_high": 2.30,
                "master_mag_F606W": 24.0,
                "master_mag_F814W": 23.0,
                "master_mag_F160W": 22.0,
            },
            {
                "cluster": "a370",
                "source_slug": "src",
                "source_id": "a370/src/mul.cat",
                "source_path": "mul.cat",
                "literature_family_id": "1",
                "literature_id": "1.2",
                "literature_ra": 10.0 + 0.1 / 3600.0,
                "literature_dec": 0.0,
                "literature_z": 2.0,
                "matched": True,
                "match_separation_arcsec": 0.2,
                "duplicate_master_match": False,
                "master_object_id": "m2",
                "master_zspec_best": 2.003,
                "master_zspec_best_confidence_rank": 2.0,
                "master_zspec_best_confidence": "probable",
                "master_zspec_best_source": "lagattuta22",
                "master_zphot_best": 2.15,
                "master_zpdf_low": 2.00,
                "master_zpdf_high": 2.40,
                "master_mag_F606W": 25.0,
                "master_mag_F814W": 24.0,
                "master_mag_F160W": 23.0,
            },
        ]
    )

    pairs = analyzer.compute_family_pair_metrics(matches)

    assert len(pairs) == 1
    row = pairs.iloc[0]
    np.testing.assert_allclose(row["master_zspec_delta"], 0.003)
    assert row["both_strong_specz"]
    assert row["strong_specz_consistent"]
    assert not row["strong_specz_conflict"]
    np.testing.assert_allclose(row["master_zphot_delta"], 0.05)
    assert row["zpdf_overlap"]
    assert row["n_common_bands"] == 3
    np.testing.assert_allclose(row["sed_rms"], 0.0)
    np.testing.assert_allclose(row["median_mag_offset"], -1.0)


def test_one_common_band_is_flagged_as_weak_color_evidence() -> None:
    matches = pd.DataFrame(
        [
            {
                "cluster": "a370",
                "source_slug": "src",
                "source_id": "a370/src/mul.cat",
                "source_path": "mul.cat",
                "literature_family_id": "1",
                "literature_id": "1.1",
                "literature_ra": 10.0,
                "literature_dec": 0.0,
                "literature_z": 2.0,
                "matched": True,
                "match_separation_arcsec": 0.1,
                "duplicate_master_match": False,
                "master_object_id": "m1",
                "master_zspec_best": 2.0,
                "master_zspec_best_confidence_rank": 3.0,
                "master_mag_F814W": 24.0,
            },
            {
                "cluster": "a370",
                "source_slug": "src",
                "source_id": "a370/src/mul.cat",
                "source_path": "mul.cat",
                "literature_family_id": "1",
                "literature_id": "1.2",
                "literature_ra": 10.0 + 0.1 / 3600.0,
                "literature_dec": 0.0,
                "literature_z": 2.0,
                "matched": True,
                "match_separation_arcsec": 0.2,
                "duplicate_master_match": False,
                "master_object_id": "m2",
                "master_zspec_best": 2.0,
                "master_zspec_best_confidence_rank": 3.0,
                "master_mag_F814W": 25.0,
            },
        ]
    )

    pairs = analyzer.compute_family_pair_metrics(matches)
    families = analyzer.compute_family_summary(matches, pairs)

    assert pairs.iloc[0]["n_common_bands"] == 1
    assert pairs.iloc[0]["weak_color_evidence"]
    assert "weak_color_evidence" in families.iloc[0]["review_flags"]


def _write_synthetic_inputs(catalog_root: Path, literature_root: Path, pagul_root: Path) -> None:
    _write_synthetic_pagul_catalog(pagul_root)
    master_dir = catalog_root / "a370"
    master_dir.mkdir(parents=True)
    row1 = _base_master_row("m1", 10.0 + 0.05 / 3600.0, zspec=2.000, zphot=2.10)
    row2 = _base_master_row("m2", 10.0 + 0.25 / 3600.0, zspec=2.003, zphot=2.15)
    for band, value in {"F606W": 24.0, "F814W": 23.0, "F160W": 22.0}.items():
        row1[f"mag_{band}"] = value
        row2[f"mag_{band}"] = value + 1.0
    pd.DataFrame([row1, row2]).to_csv(master_dir / "a370_master_catalog.csv", index=False)

    literature_dir = literature_root / "a370" / "example"
    literature_dir.mkdir(parents=True)
    image_path = literature_dir / "mul.cat"
    image_path.write_text(
        "#REFERENCE 0\n"
        f"1.1 {10.0 + 0.04 / 3600.0:.12f} 0.0 0.1 0.1 0.0 2.0 25.0\n"
        f"1.2 {10.0 + 0.20 / 3600.0:.12f} 0.0 0.1 0.1 0.0 2.0 25.0\n"
        f"2.1 {10.0 + 2.00 / 3600.0:.12f} 0.0 0.1 0.1 0.0 2.5 25.0\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "cluster": "a370",
                "source_slug": "example",
                "file_role": "image_catalog",
                "source_path": "",
                "copied_path": image_path,
                "size_bytes": 1,
                "copied_at": "",
                "status": "copied",
            },
            {
                "cluster": "m0717",
                "source_slug": "no_local_literature_catalog",
                "file_role": "missing_source",
                "source_path": "",
                "copied_path": "",
                "size_bytes": 0,
                "copied_at": "",
                "status": "missing_source",
            },
        ]
    ).to_csv(literature_root / "literature_copy_manifest.csv", index=False)


def test_cli_defaults_and_analysis_outputs(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalogs"
    literature_root = tmp_path / "literature"
    pagul_root = tmp_path / "pagul"
    output_dir = tmp_path / "diagnostics"
    _write_synthetic_inputs(catalog_root, literature_root, pagul_root)

    args = analyzer.build_arg_parser().parse_args([])
    assert args.catalog_root == Path("results") / "hff_master_catalogs"
    assert args.literature_root == Path("data") / "literature_lenstool_models"
    assert args.pagul_root == Path("data") / "Pagul2024"
    assert args.output_dir == Path("results") / "literature_family_diagnostics"
    assert args.match_radius_arcsec == 0.5

    assert analyzer.main(
        [
            "--catalog-root",
            str(catalog_root),
            "--literature-root",
            str(literature_root),
            "--pagul-root",
            str(pagul_root),
            "--output-dir",
            str(output_dir),
            "--clusters",
            "a370,m0717",
        ]
    ) == 0

    matches = pd.read_csv(output_dir / "literature_image_master_matches.csv")
    pairs = pd.read_csv(output_dir / "literature_family_pair_metrics.csv")
    families = pd.read_csv(output_dir / "literature_family_summary.csv")
    manifest = pd.read_csv(output_dir / "literature_family_diagnostics_manifest.csv")
    pagul_summary = pd.read_csv(output_dir / "pagul_crossmatch_summary.csv")
    pagul_zphot_cut_summary = pd.read_csv(output_dir / "pagul_zphot_background_cut_summary.csv")
    pagul_specz_coverage_summary = pd.read_csv(output_dir / "pagul_specz_coverage_summary.csv")

    assert matches["matched"].map(analyzer._bool_value).sum() == 2
    assert matches["pagul_matched"].map(analyzer._bool_value).sum() == 2
    np.testing.assert_allclose(matches.loc[0, "pagul_zphot"], 2.11)
    assert len(pairs) == 1
    assert families.loc[families["literature_family_id"].astype(str).eq("1"), "analysis_selected"].map(analyzer._bool_value).iloc[0]
    assert "no_literature_image_catalog" in set(manifest["status"])
    source_summary = pagul_summary.loc[pagul_summary["scope"].eq("source")].iloc[0]
    assert source_summary["n_literature_images"] == 3
    assert source_summary["n_pagul_matched_images"] == 2
    assert source_summary["n_matched_both"] == 2
    cut_summary = pagul_zphot_cut_summary.loc[pagul_zphot_cut_summary["cluster"].eq("a370")].iloc[0]
    assert cut_summary["matched_with_zpdf"] == 2
    assert cut_summary["above_cut"] == 2
    np.testing.assert_allclose(cut_summary["zphot_cut"], analyzer.CLUSTER_Z_LENS["a370"] + analyzer.BACKGROUND_Z_MARGIN)
    specz_summary = pagul_specz_coverage_summary.loc[pagul_specz_coverage_summary["scope"].eq("source")].iloc[0]
    assert specz_summary["n_literature_images"] == 3
    assert specz_summary["n_pagul_matched_images"] == 2
    assert specz_summary["n_pagul_matched_with_zspec"] == 2
    assert specz_summary["n_pagul_matched_without_zspec"] == 0
    np.testing.assert_allclose(specz_summary["pagul_specz_fraction_among_matches"], 1.0)
    assert (output_dir / "literature_sources.csv").exists()
    assert (output_dir / "literature_family_pair_metrics.csv").exists()
    assert (output_dir / "pagul_crossmatch_summary.csv").exists()
    assert (output_dir / "pagul_zphot_background_cut_summary.csv").exists()
    assert (output_dir / "pagul_specz_coverage_summary.csv").exists()
    assert (output_dir / "plots" / "pagul_crossmatch_summary.png").exists()
    assert (output_dir / "plots" / "pagul_crossmatch_summary.pdf").exists()
    assert (output_dir / "plots" / "pagul_specz_coverage.png").exists()
    assert (output_dir / "plots" / "pagul_specz_coverage.pdf").exists()
    assert (output_dir / "plots" / "zphot_delta_by_literature_source.png").exists()
    assert (output_dir / "plots" / "zphot_delta_by_literature_source.pdf").exists()
    assert (output_dir / "plots" / "pagul_matched_literature_zspec_vs_zphot.png").exists()
    assert (output_dir / "plots" / "pagul_matched_literature_zspec_vs_zphot.pdf").exists()
    assert (output_dir / "plots" / "pagul_zphot_background_cut.png").exists()
    assert (output_dir / "plots" / "pagul_zphot_background_cut.pdf").exists()
    assert any(output_dir.rglob("*.png"))
    assert any(output_dir.rglob("*.pdf"))
