from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_hff_master_catalog.py"
spec = importlib.util.spec_from_file_location("build_hff_master_catalog", SCRIPT_PATH)
assert spec is not None
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


def _cgs_flux_for_abmag(mag: float) -> float:
    return 10.0 ** (-(mag + 48.6) / 2.5)


MEMBER_BANDS = ["F435W", "F475W", "F606W", "F625W", "F814W", "F105W", "F125W", "F160W"]


class FakeProgress:
    def __init__(self) -> None:
        self.started: list[tuple[str, int | None]] = []
        self.advanced = 0
        self.finished = 0

    def start_step(self, label: str, total: int | None = None) -> None:
        self.started.append((label, total))

    def advance_step(self, n: int = 1) -> None:
        self.advanced += int(n)

    def finish_step(self) -> None:
        self.finished += 1


def _member_row(
    object_id: str,
    *,
    zspec: float = np.nan,
    rank: float = np.nan,
    confidence: str = "",
    zphot: float = np.nan,
    zphot_low: float = np.nan,
    zphot_high: float = np.nan,
    ra_offset_arcsec: float = 0.0,
    mags: dict[str, float] | None = None,
    missing_bands: set[str] | None = None,
) -> dict[str, object]:
    values = {
        "F435W": 22.0,
        "F475W": 21.5,
        "F606W": 21.0,
        "F625W": 20.5,
        "F814W": 20.0,
        "F105W": 19.6,
        "F125W": 19.4,
        "F160W": 19.2,
    }
    if mags:
        values.update(mags)
    row: dict[str, object] = {
        "cluster_key": "a370",
        "object_id": object_id,
        "object_source": "pagul2024",
        "catalog_sources": "pagul2024",
        "ra": 10.0 + ra_offset_arcsec / 3600.0,
        "dec": 0.0,
        "zspec_best": zspec,
        "zspec_best_confidence_rank": rank,
        "zspec_best_confidence": confidence,
        "zspec_best_native_quality": rank,
        "zspec_selection_note": "selected_by_normalized_confidence",
        "zphot_best": zphot,
        "pagul_zpdf": zphot,
        "pagul_zpdf_low": zphot_low,
        "pagul_zpdf_high": zphot_high,
        "pagul_nb_used": 6.0 if np.isfinite(zphot) else np.nan,
    }
    missing_bands = missing_bands or set()
    for band in MEMBER_BANDS:
        row[f"mag_{band}"] = np.nan if band in missing_bands else values[band]
    return row


def _image_candidate_row(object_id: str, **updates: object) -> dict[str, object]:
    updates = dict(updates)
    mags = {band: 24.0 for band in MEMBER_BANDS}
    mags.update(updates.pop("mags", {}) or {})
    row = _member_row(
        object_id,
        zspec=updates.pop("zspec", 2.0),
        rank=updates.pop("rank", 3),
        confidence=str(updates.pop("confidence", "secure")),
        zphot=updates.pop("zphot", np.nan),
        zphot_low=updates.pop("zphot_low", np.nan),
        zphot_high=updates.pop("zphot_high", np.nan),
        ra_offset_arcsec=updates.pop("ra_offset_arcsec", 0.0),
        mags=mags,
        missing_bands=updates.pop("missing_bands", None),
    )
    row.update(updates)
    return row


def test_flux_to_abmag_conversions() -> None:
    np.testing.assert_allclose(builder.flux_to_abmag_uJy(1.0), 23.9)

    flux = _cgs_flux_for_abmag(22.0)
    np.testing.assert_allclose(builder.flux_to_abmag_cgs_fnu(flux), 22.0)


def test_one_to_one_sky_match_keeps_nearest_pairs() -> None:
    left = pd.DataFrame(
        {
            "ra": [10.0, 10.001],
            "dec": [0.0, 0.0],
        }
    )
    right = pd.DataFrame(
        {
            "ra": [10.0 + 0.1 / 3600.0, 10.001 + 0.1 / 3600.0],
            "dec": [0.0, 0.0],
        }
    )

    matches = builder.one_to_one_sky_match(
        left,
        right,
        left_ra="ra",
        left_dec="dec",
        right_ra="ra",
        right_dec="dec",
        radius_arcsec=0.5,
    )

    assert len(matches) == 2
    assert set(matches["left_index"]) == {0, 1}
    assert set(matches["right_index"]) == {0, 1}
    np.testing.assert_allclose(matches["separation_arcsec"], [0.1, 0.1], rtol=0.0, atol=1.0e-6)


def test_one_to_one_sky_match_allows_shared_closest_right_object() -> None:
    left = pd.DataFrame(
        {
            "ra": [10.0, 10.0 + 0.2 / 3600.0],
            "dec": [0.0, 0.0],
        }
    )
    right = pd.DataFrame(
        {
            "ra": [10.0 + 0.1 / 3600.0],
            "dec": [0.0],
        }
    )

    matches = builder.one_to_one_sky_match(
        left,
        right,
        left_ra="ra",
        left_dec="dec",
        right_ra="ra",
        right_dec="dec",
        radius_arcsec=0.5,
    )

    assert len(matches) == 2
    assert set(matches["left_index"]) == {0, 1}
    assert matches["right_index"].tolist() == [0, 0]
    np.testing.assert_allclose(matches["separation_arcsec"], [0.1, 0.1], rtol=0.0, atol=1.0e-6)


def test_build_cluster_catalog_reports_subprogress() -> None:
    pagul = pd.DataFrame(
        {
            "ID": [101],
            "FIELD": ["A370clu"],
            "ALPHA_J2000_STACK": [10.0],
            "DELTA_J2000_STACK": [0.0],
            "FLUX_F160W": [_cgs_flux_for_abmag(22.0)],
        }
    )
    shipley = pd.DataFrame(
        {
            "Cl": ["A370-clu", "A370-clu"],
            "ID": [201, 202],
            "RAJ2000": [10.0 + 0.1 / 3600.0, 11.0],
            "DEJ2000": [0.0, 0.0],
            "FF160W": [10.0, 1.0],
        }
    )
    progress = FakeProgress()

    builder.build_cluster_catalog(
        spec=builder.CLUSTER_BY_KEY["a370"],
        pagul=pagul,
        shipley=shipley,
        match_radius_arcsec=0.5,
        progress=progress,
    )

    labels = [label for label, _total in progress.started]
    assert any("matching sky positions" in label for label in labels)
    assert any("building master rows" in label for label in labels)
    assert progress.advanced >= 3
    assert progress.finished >= 2


def test_build_cluster_catalog_appends_all_unmatched_shipley_and_tracks_sources() -> None:
    pagul = pd.DataFrame(
        {
            "ID": [101],
            "FIELD": ["A370clu"],
            "ALPHA_J2000_STACK": [10.0],
            "DELTA_J2000_STACK": [0.0],
            "FLUX_F160W": [_cgs_flux_for_abmag(22.0)],
            "ZSPEC": [0.400],
            "ZSPEC_Q": [4.0],
            "ZSPEC_REF": ["NED"],
            "CHI2_RED": [1.2],
            "ZPDF": [0.390],
            "ZPDF_LOW": [0.360],
            "ZPDF_HIGH": [0.420],
            "ZSECOND": [-999.0],
            "NB_USED": [12],
            "BITMASK": [123],
        }
    )
    shipley = pd.DataFrame(
        {
            "Cl": ["A370-clu", "A370-clu"],
            "ID": [201, 202],
            "RAJ2000": [10.0 + 0.1 / 3600.0, 11.0],
            "DEJ2000": [0.0, 0.0],
            "FF160W": [10.0, 1.0],
            "zspec": [0.410, 0.500],
            "r_zspec": ["muse", "none"],
            "Use": [1, 0],
            "S_G": [0, 1],
            "BandTot": ["F160W", "F814W"],
            "FRad": [3.0, 4.0],
            "FlF814W": [0, 3],
            "FlF160W": [0, 3],
        }
    )
    catalog, audit, manifest = builder.build_cluster_catalog(
        spec=builder.CLUSTER_BY_KEY["a370"],
        pagul=pagul,
        shipley=shipley,
        match_radius_arcsec=0.5,
    )

    assert len(catalog) == 2
    assert manifest["n_shipley_unmatched_appended"] == 1
    assert set(catalog["object_source"]) == {"pagul2024", "shipley2018_unmatched"}
    assert set(audit["match_type"]) == {"pagul2024_shipley2018"}
    assert manifest["n_lagattuta22_redshift_rows"] == 0
    assert manifest["n_lagattuta22_redshift_matched"] == 0

    matched = catalog.loc[catalog["object_source"] == "pagul2024"].iloc[0]
    assert matched["catalog_sources"] == "pagul2024|shipley2018"
    np.testing.assert_allclose(matched["mag_F160W"], 22.0)
    assert matched["zspec_best_source"] == "pagul2024"
    np.testing.assert_allclose(matched["zspec_best"], 0.400)
    assert matched["zspec_best_confidence"] == "secure"
    assert matched["zspec_best_native_quality"] == 4
    assert matched["zspec_selection_note"] == "candidate_conflict"
    assert bool(matched["zspec_conflict"])
    np.testing.assert_allclose(matched["zphot_best"], 0.390)
    assert matched["zphot_best_source"] == "pagul2024_zpdf"
    np.testing.assert_allclose(matched["pagul_zpdf"], 0.390)
    assert "lagattuta22_zspec" in catalog.columns
    assert math.isnan(float(matched["lagattuta22_zspec"]))
    assert "multiple_image_id" not in catalog.columns

    unmatched = catalog.loc[catalog["object_source"] == "shipley2018_unmatched"].iloc[0]
    assert unmatched["catalog_sources"] == "shipley2018"
    assert unmatched["shipley_use"] == 0
    np.testing.assert_allclose(unmatched["mag_F160W"], 23.9)


def test_choose_best_zspec_ignores_unrecognized_sources() -> None:
    result = builder.choose_best_zspec(
        [
            {"source": "pagul2024", "z": 0.20, "quality": 4},
            {"source": "external", "z": 0.30, "quality": 3},
            {"source": "shipley2018", "z": 0.21},
        ]
    )

    assert result["zspec_best"] == 0.20
    assert result["zspec_best_source"] == "pagul2024"
    assert result["zspec_best_confidence"] == "secure"
    assert result["zspec_best_native_quality"] == 4
    assert result["zspec_selection_note"] == "candidate_conflict"
    assert result["zspec_candidate_sources"] == "pagul2024|shipley2018"


def test_pagul_photoz_uses_zpdf_for_best_photoz() -> None:
    row = pd.Series(
        {
            "CHI2_RED": 2.0,
            "ZPDF": 0.390,
            "ZPDF_LOW": 0.360,
            "ZPDF_HIGH": 0.420,
            "ZSECOND": 0.7,
            "NB_USED": 12,
            "BITMASK": 123,
        }
    )

    result = builder._pagul_photoz(row)

    np.testing.assert_allclose(result["zphot_best"], 0.390)
    assert result["zphot_best_source"] == "pagul2024_zpdf"
    np.testing.assert_allclose(result["pagul_zpdf"], 0.390)


def test_choose_best_zspec_uses_shipley_only_as_fallback() -> None:
    low_pagul = builder.choose_best_zspec(
        [
            {"source": "pagul2024", "z": 0.20, "quality": 1},
            {"source": "shipley2018", "z": 0.21},
        ]
    )
    tentative_pagul = builder.choose_best_zspec(
        [
            {"source": "pagul2024", "z": 0.20, "quality": 2},
            {"source": "shipley2018", "z": 0.21},
        ]
    )
    only_shipley = builder.choose_best_zspec([{"source": "shipley2018", "z": 0.21}])

    assert low_pagul["zspec_best_source"] == "shipley2018"
    assert low_pagul["zspec_best_confidence"] == "fallback"
    assert low_pagul["zspec_selection_note"] == "candidate_conflict"
    assert tentative_pagul["zspec_best_source"] == "pagul2024"
    assert tentative_pagul["zspec_best_confidence"] == "tentative"
    assert only_shipley["zspec_best_source"] == "shipley2018"
    assert only_shipley["zspec_selection_note"] == "fallback_compiled_specz"


def test_choose_best_zspec_handles_lagattuta22_precedence() -> None:
    pagul_tie = builder.choose_best_zspec(
        [
            {"source": "pagul2024", "z": 1.20, "quality": 4},
            {"source": "lagattuta22", "z": 1.20, "quality": 3},
        ]
    )
    secure_lagattuta = builder.choose_best_zspec(
        [
            {"source": "ned", "z": 0.70, "quality": "SLS"},
            {"source": "simbad", "z": 0.70, "quality": "C"},
            {"source": "lagattuta22", "z": 0.70, "quality": 3},
        ]
    )
    probable_lagattuta = builder.choose_best_zspec([{"source": "lagattuta22", "z": 1.35, "quality": 2}])

    assert pagul_tie["zspec_best_source"] == "pagul2024"
    assert pagul_tie["zspec_best_confidence"] == "secure"
    assert secure_lagattuta["zspec_best_source"] == "lagattuta22"
    assert secure_lagattuta["zspec_best_confidence"] == "secure"
    assert probable_lagattuta["zspec_best_source"] == "lagattuta22"
    assert probable_lagattuta["zspec_best_confidence"] == "probable"
    np.testing.assert_allclose(probable_lagattuta["zspec_best_confidence_rank"], builder.SPECZ_CONFIDENCE_PROBABLE)


def test_external_redshift_loaders_standardize_ned_and_simbad(tmp_path: Path) -> None:
    cluster_dir = tmp_path / "a370"
    cluster_dir.mkdir()
    pd.DataFrame(
        {
            "Object Name": ["ned-spec", "ned-photo"],
            "RA": [10.0, 10.1],
            "DEC": [0.0, 0.1],
            "Type": ["G", "G"],
            "Redshift": [0.45, 1.2],
            "Redshift Flag": ["SLS", "PUN"],
            "References": [3, 1],
        }
    ).to_csv(cluster_dir / "ned_core_redshifts.csv", index=False)
    pd.DataFrame(
        {
            "main_id": ["sim-a"],
            "ra": [10.2],
            "dec": [0.2],
            "otype": ["G"],
            "rvz_redshift": [0.55],
            "rvz_qual": ["C"],
            "rvz_bibcode": ["2024A&A..."],
        }
    ).to_csv(cluster_dir / "simbad_core_redshifts.csv", index=False)

    external = builder.load_external_redshift_catalogs(tmp_path, builder.CLUSTER_BY_KEY["a370"])

    ned = external["ned"].set_index("external_id")
    simbad = external["simbad"].set_index("external_id")
    np.testing.assert_allclose(ned.loc["ned-spec", "zspec"], 0.45)
    assert math.isnan(float(ned.loc["ned-photo", "zspec"]))
    np.testing.assert_allclose(ned.loc["ned-photo", "redshift_raw"], 1.2)
    np.testing.assert_allclose(simbad.loc["sim-a", "zspec"], 0.55)
    assert simbad.loc["sim-a", "native_quality"] == "C"


def test_lagattuta22_loader_standardizes_pilotwings_fits(tmp_path: Path) -> None:
    path = tmp_path / "A370_PilotWINGS_data_catalog.fits"
    Table(
        {
            "iden": [1, 2],
            "idfrom": ["PRIOR", "MUSELET"],
            "Field": ["CORE", "P02"],
            "RA": [39.0, 39.1],
            "DEC": [-1.0, -1.1],
            "z": [0.375, 0.0],
            "zconf": [3, 1],
            "MUL": ["5.1", ""],
        }
    ).write(path)

    external = builder.load_external_redshift_catalogs(
        tmp_path,
        builder.CLUSTER_BY_KEY["a370"],
        lagattuta22_path=path,
    )
    lagattuta = external["lagattuta22"].set_index("external_id")

    np.testing.assert_allclose(lagattuta.loc["1:PRIOR:CORE", "zspec"], 0.375)
    assert math.isnan(float(lagattuta.loc["2:MUSELET:P02", "zspec"]))
    assert lagattuta.loc["1:PRIOR:CORE", "native_quality"] == 3
    assert lagattuta.loc["1:PRIOR:CORE", "reference"] == builder.LAGATTUTA22_REFERENCE
    assert lagattuta.loc["1:PRIOR:CORE", "id_source"] == "PRIOR"
    assert lagattuta.loc["1:PRIOR:CORE", "multiple_image_id"] == "5.1"

    other_cluster = builder.load_external_redshift_catalogs(
        tmp_path,
        builder.CLUSTER_BY_KEY["a2744"],
        lagattuta22_path=path,
    )
    assert other_cluster["lagattuta22"].empty


def test_build_cluster_catalog_matches_external_redshifts_and_uses_as_fallback() -> None:
    pagul = pd.DataFrame(
        {
            "ID": [101, 102, 103],
            "FIELD": ["A370clu", "A370clu", "A370clu"],
            "ALPHA_J2000_STACK": [10.0, 20.0, 30.0],
            "DELTA_J2000_STACK": [0.0, 0.0, 0.0],
            "FLUX_F160W": [_cgs_flux_for_abmag(22.0), _cgs_flux_for_abmag(22.0), _cgs_flux_for_abmag(22.0)],
            "ZSPEC": [np.nan, 0.40, np.nan],
            "ZSPEC_Q": [np.nan, 4.0, np.nan],
        }
    )
    shipley = pd.DataFrame(
        {
            "Cl": ["A370-clu"],
            "ID": [201],
            "RAJ2000": [40.0],
            "DEJ2000": [0.0],
            "FF160W": [1.0],
            "zspec": [np.nan],
        }
    )
    ned = pd.DataFrame(
        {
            "source": ["ned", "ned", "ned"],
            "field": ["core", "core", "core"],
            "external_id": ["ned-a", "ned-nearer", "ned-phot"],
            "ra": [10.0 + 0.1 / 3600.0, 20.0 + 0.05 / 3600.0, 30.0 + 0.1 / 3600.0],
            "dec": [0.0, 0.0, 0.0],
            "redshift_raw": [0.50, 0.90, 1.30],
            "zspec": [0.50, 0.90, np.nan],
            "native_quality": ["SLS", "SLS", "PUN"],
            "reference": ["1", "2", "3"],
            "object_type": ["G", "G", "G"],
        }
    )
    simbad = pd.DataFrame(
        {
            "source": ["simbad"],
            "field": ["core"],
            "external_id": ["sim-a"],
            "ra": [10.0 + 0.2 / 3600.0],
            "dec": [0.0],
            "redshift_raw": [0.51],
            "zspec": [0.51],
            "native_quality": ["C"],
            "reference": ["bib"],
            "object_type": ["G"],
        }
    )

    catalog, audit, manifest = builder.build_cluster_catalog(
        spec=builder.CLUSTER_BY_KEY["a370"],
        pagul=pagul,
        shipley=shipley,
        external_redshifts={"ned": ned, "simbad": simbad},
        match_radius_arcsec=0.5,
        redshift_match_radius_arcsec=0.5,
    )
    indexed = catalog.set_index("pagul_id")

    assert indexed.loc["101", "zspec_best_source"] == "ned"
    np.testing.assert_allclose(indexed.loc["101", "zspec_best"], 0.50)
    assert indexed.loc["101", "zspec_best_confidence"] == "probable"
    np.testing.assert_allclose(indexed.loc["101", "zspec_best_confidence_rank"], 2.0)
    assert indexed.loc["101", "zspec_selection_note"] == "candidate_conflict"
    assert indexed.loc["101", "catalog_sources"] == "pagul2024|ned|simbad"
    assert indexed.loc["102", "zspec_best_source"] == "pagul2024"
    assert bool(indexed.loc["102", "zspec_conflict"])
    assert indexed.loc["103", "zspec_best_source"] == ""
    np.testing.assert_allclose(indexed.loc["103", "ned_redshift_raw"], 1.30)
    assert math.isnan(float(indexed.loc["103", "ned_zspec"]))
    assert set(audit["match_type"]) >= {"master_ned_redshift", "master_simbad_redshift"}
    assert manifest["n_ned_redshift_rows"] == 3
    assert manifest["n_ned_redshift_matched"] == 3
    assert manifest["n_simbad_redshift_matched"] == 1
    assert manifest["n_ned_zspec_best"] == 1
    assert manifest["n_external_zspec_conflicts"] == 2


def test_build_cluster_catalog_matches_lagattuta22_redshifts() -> None:
    pagul = pd.DataFrame(
        {
            "ID": [201, 202],
            "FIELD": ["A370clu", "A370clu"],
            "ALPHA_J2000_STACK": [10.0, 20.0],
            "DELTA_J2000_STACK": [0.0, 0.0],
            "FLUX_F160W": [_cgs_flux_for_abmag(22.0), _cgs_flux_for_abmag(22.0)],
            "ZSPEC": [np.nan, 0.40],
            "ZSPEC_Q": [np.nan, 4.0],
        }
    )
    shipley = pd.DataFrame(
        {
            "Cl": [],
            "ID": [],
            "RAJ2000": [],
            "DEJ2000": [],
            "FF160W": [],
            "zspec": [],
        }
    )
    lagattuta = pd.DataFrame(
        {
            "source": ["lagattuta22", "lagattuta22"],
            "field": ["CORE", "P02"],
            "external_id": ["11:PRIOR:CORE", "12:MUSELET:P02"],
            "ra": [10.0 + 0.1 / 3600.0, 20.0 + 0.1 / 3600.0],
            "dec": [0.0, 0.0],
            "redshift_raw": [0.50, 0.42],
            "zspec": [0.50, 0.42],
            "native_quality": [2, 3],
            "reference": [builder.LAGATTUTA22_REFERENCE, builder.LAGATTUTA22_REFERENCE],
            "object_type": ["PRIOR", "MUSELET"],
            "id_source": ["PRIOR", "MUSELET"],
            "multiple_image_id": ["", "8.1"],
        }
    )

    catalog, audit, manifest = builder.build_cluster_catalog(
        spec=builder.CLUSTER_BY_KEY["a370"],
        pagul=pagul,
        shipley=shipley,
        external_redshifts={"lagattuta22": lagattuta},
        redshift_match_radius_arcsec=0.5,
    )
    indexed = catalog.set_index("pagul_id")

    assert indexed.loc["201", "catalog_sources"] == "pagul2024|lagattuta22"
    assert indexed.loc["201", "zspec_best_source"] == "lagattuta22"
    assert indexed.loc["201", "zspec_best_confidence"] == "probable"
    np.testing.assert_allclose(indexed.loc["201", "zspec_best"], 0.50)
    np.testing.assert_allclose(indexed.loc["201", "lagattuta22_zspec"], 0.50)
    assert indexed.loc["201", "lagattuta22_external_id"] == "11:PRIOR:CORE"
    assert indexed.loc["201", "lagattuta22_id_source"] == "PRIOR"

    assert indexed.loc["202", "zspec_best_source"] == "pagul2024"
    assert bool(indexed.loc["202", "zspec_conflict"])
    assert indexed.loc["202", "lagattuta22_multiple_image_id"] == "8.1"

    assert "master_lagattuta22_redshift" in set(audit["match_type"])
    assert manifest["n_lagattuta22_redshift_rows"] == 2
    assert manifest["n_lagattuta22_redshift_matched"] == 2
    assert manifest["n_lagattuta22_zspec_best"] == 1
    assert manifest["n_external_zspec_conflicts"] == 1
    assert manifest["external_redshift_sources"] == "lagattuta22"


def test_external_specz_without_conflict_is_probable() -> None:
    ned = builder.choose_best_zspec([{"source": "ned", "z": 0.70, "quality": "SLS"}])
    simbad = builder.choose_best_zspec([{"source": "simbad", "z": 0.71, "quality": "C"}])

    assert ned["zspec_best_source"] == "ned"
    assert ned["zspec_best_confidence"] == "probable"
    assert ned["zspec_best_confidence_rank"] == builder.SPECZ_CONFIDENCE_PROBABLE
    assert ned["zspec_selection_note"] == "external_probable_specz"
    assert simbad["zspec_best_source"] == "simbad"
    assert simbad["zspec_best_confidence"] == "probable"
    assert simbad["zspec_best_confidence_rank"] == builder.SPECZ_CONFIDENCE_PROBABLE
    assert simbad["zspec_selection_note"] == "external_probable_specz"


def test_external_specz_does_not_override_secure_curated_specz() -> None:
    result = builder.choose_best_zspec(
        [
            {"source": "pagul2024", "z": 0.40, "quality": 3},
            {"source": "ned", "z": 0.70, "quality": "SLS"},
            {"source": "simbad", "z": 0.71, "quality": "C"},
        ]
    )

    assert result["zspec_best_source"] == "pagul2024"
    assert result["zspec_best_confidence"] == "probable"
    assert result["zspec_candidate_sources"] == "pagul2024|ned|simbad"
    assert result["zspec_conflict"]


def test_locate_pagul_catalog_raises_for_missing_cluster(tmp_path: Path) -> None:
    (tmp_path / "hlsp_buffalo_hst_ir-weighted_abell370_multi_v2.0_catalog.fits").write_text("", encoding="utf-8")

    with pytest.raises(builder.MissingCatalogError):
        builder.locate_pagul_catalog(builder.CLUSTER_BY_KEY["a2744"], tmp_path)


def test_secure_specz_member_inside_velocity_window_gets_high_probability() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame([_member_row("secure", zspec=0.376, rank=3, confidence="secure")])

    scores, _red_sequence, manifest = builder.score_cluster_members(catalog, spec)

    row = scores.iloc[0]
    assert row["member_probability"] >= 0.95
    assert row["member_class"] == "secure_spec_member"
    assert bool(row["cluster_member_selected"])
    assert manifest["member_velocity_window_kms"] == 3000.0


def test_secure_specz_nonmember_is_rejected_even_if_photoz_is_cluster_like() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _member_row(
                "background",
                zspec=1.0,
                rank=3,
                confidence="secure",
                zphot=0.375,
                zphot_low=0.34,
                zphot_high=0.41,
            )
        ]
    )

    scores, _red_sequence, _manifest = builder.score_cluster_members(catalog, spec)

    row = scores.iloc[0]
    assert row["member_probability"] == 0.0
    assert row["member_class"] == "rejected_foreground_background"
    assert not bool(row["cluster_member_selected"])


def test_photoz_only_cluster_member_is_capped_below_spec_member() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _member_row("secure", zspec=0.3755, rank=3, confidence="secure"),
            _member_row("photo", zphot=0.375, zphot_low=0.34, zphot_high=0.41, ra_offset_arcsec=1.0),
        ]
    )

    scores, _red_sequence, _manifest = builder.score_cluster_members(catalog, spec)
    indexed = scores.set_index("object_id")

    assert indexed.loc["photo", "member_probability"] <= 0.60
    assert indexed.loc["photo", "member_probability"] < indexed.loc["secure", "member_probability"]
    assert indexed.loc["photo", "member_probability"] < builder.DEFAULT_MEMBER_PROBABILITY_THRESHOLD
    assert not bool(indexed.loc["photo", "cluster_member_selected"])


def test_member_photoz_scoring_ignores_zpdf_interval() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    row = pd.Series(_member_row("photo", zphot=2.0, zphot_low=0.34, zphot_high=0.41))

    score, note = builder.photoz_member_score(row, spec)

    assert note == "photoz_best_offset"
    assert score < 1.0e-20


def test_candidate_pair_scoring_accepts_consistent_photoz_only_rows() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _image_candidate_row("photo-a", zspec=np.nan, rank=np.nan, zphot=2.0, zphot_low=1.0, zphot_high=4.0, ra_offset_arcsec=0.0),
            _image_candidate_row("photo-b", zspec=np.nan, rank=np.nan, zphot=2.1, zphot_low=1.0, zphot_high=4.0, ra_offset_arcsec=10.0),
        ]
    )

    candidates = builder.prepare_candidates(catalog, spec)
    pairs = builder.score_candidate_pairs(
        candidates,
        spec,
        pair_score_threshold=0.0,
        family_pair_diagnostics="all",
    )

    assert set(candidates["object_id"]) == {"photo-a", "photo-b"}
    assert len(pairs) == 1
    assert pairs.iloc[0]["hard_reject_reason"] == ""
    assert pairs.iloc[0]["redshift_relation"] == "photoz_only"
    assert pairs.iloc[0]["photoz_score"] > 0.0


def test_image_preclean_rejects_bad_rows_before_candidate_selection() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    good_quality = {
        "shipley_use": 1,
        "shipley_flf814w": 0,
        "shipley_frad": 3.0,
        "shipley_aimg": 3.0,
        "shipley_bimg": 2.0,
        "magerr_F814W": 0.1,
    }
    rows = [
        _image_candidate_row("good", **good_quality),
        _image_candidate_row("missing-optional"),
        _image_candidate_row("too-bright", **good_quality, mags={"F814W": 23.5}),
        _image_candidate_row("too-faint-hff", **good_quality, mags={"F814W": 28.6}),
        _image_candidate_row("too-faint-outer", **good_quality, mags={"F814W": 27.1}, in_hff_footprint=False),
        _image_candidate_row("undersized", **{**good_quality, "shipley_frad": 1.0}),
        _image_candidate_row("bad-use", **{**good_quality, "shipley_use": 0}),
        _image_candidate_row("bad-flag", **{**good_quality, "shipley_flf814w": 3}),
        _image_candidate_row("bad-shape", **{**good_quality, "shipley_aimg": 2.0, "shipley_bimg": 2.0}),
        _image_candidate_row("bad-error", **{**good_quality, "magerr_F814W": 0.0}),
    ]
    catalog = pd.DataFrame(rows)

    precleaned = builder.apply_image_precuts(catalog)
    reasons = precleaned.set_index("object_id")["image_preclean_reject_reason"].to_dict()
    candidates = builder.prepare_candidates(catalog, spec)

    assert set(candidates["object_id"]) == {
        "good",
        "missing-optional",
        "too-bright",
        "too-faint-hff",
        "too-faint-outer",
        "bad-flag",
    }
    assert reasons["too-bright"] == ""
    assert reasons["too-faint-hff"] == "too_faint_f814w"
    assert reasons["too-faint-outer"] == "too_faint_f814w"
    assert reasons["undersized"] == "too_small"
    assert reasons["bad-use"] == "bad_shipley_use"
    assert reasons["bad-flag"] == "bad_f814w_flag"
    assert "rescued_bad_f814w_flag" in candidates.set_index("object_id").loc["bad-flag", "image_family_review_flags"]
    assert "rescued_too_faint_f814w" in candidates.set_index("object_id").loc["too-faint-hff", "image_family_review_flags"]
    assert reasons["bad-shape"] == "invalid_ellipticity"
    assert reasons["bad-error"] == "nonpositive_f814w_error"
    assert precleaned.set_index("object_id").loc["good", "image_size_arcsec"] >= 0.11
    assert 0.0 < precleaned.set_index("object_id").loc["good", "image_ellipticity"] < 1.0
    metrics = candidates.attrs["image_preclean_metrics"]
    assert metrics["n_image_preclean_rejected"] == 7
    assert metrics["n_image_preclean_size_available"] == 9
    assert metrics["n_image_preclean_shape_available"] == 9


def test_image_preclean_missing_optional_columns_passes_and_reports_manifest_notes() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _image_candidate_row("a", ra_offset_arcsec=0.0, zspec=2.0),
            _image_candidate_row("b", ra_offset_arcsec=10.0, zspec=2.0),
        ]
    )

    families, members, pairs, manifest = builder.build_cluster_image_families(
        catalog,
        spec,
        pair_score_threshold=0.0,
        two_image_score_threshold=0.0,
    )

    assert not families.empty
    assert set(members["object_id"]) == {"a", "b"}
    assert len(pairs) == 1
    assert manifest["n_image_preclean_rows"] == 2
    assert manifest["n_image_preclean_rejected"] == 0
    assert manifest["n_image_preclean_size_available"] == 0
    assert manifest["n_image_preclean_shape_available"] == 0
    assert manifest["n_image_preclean_shipley_use_available"] == 0
    assert manifest["n_image_preclean_f814w_error_available"] == 0


def test_parse_reference_family_catalog_handles_blank_redshifts(tmp_path: Path) -> None:
    path = tmp_path / "sl-final.dat"
    path.write_text(
        "\n".join(
            [
                "# ID RA DEC z_spec cat",
                '9.2 39.9694583 -1.5762853 "" Gold',
                "10.3 39.9690233 -1.5775039 1.5182 Bronze",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = builder.parse_reference_family_catalog(path)

    assert parsed["reference_image_id"].tolist() == ["9.2", "10.3"]
    assert parsed["reference_family_id"].tolist() == ["9", "10"]
    assert np.isnan(parsed.loc[0, "reference_z"])
    np.testing.assert_allclose(parsed.loc[1, "reference_z"], 1.5182)


def test_reference_family_diagnostics_counts_recovered_family(tmp_path: Path) -> None:
    reference_path = tmp_path / "sl-final.dat"
    reference_path.write_text(
        "\n".join(
            [
                "# ID RA DEC z_spec cat",
                "1.1 10.0000000 0.0000000 2.0 Gold",
                "1.2 10.0027778 0.0000000 2.0 Gold",
                "2.1 10.0100000 0.0000000 3.0 Silver",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    master = pd.DataFrame(
        [
            {"object_id": "m1", "ra": 10.0, "dec": 0.0, "mag_F435W": 25.0, "mag_F814W": 24.0, "zspec_best": 2.0},
            {"object_id": "m2", "ra": 10.0027778, "dec": 0.0, "mag_F435W": 25.2, "mag_F814W": 24.2, "zspec_best": 2.0},
        ]
    )
    members = pd.DataFrame(
        [
            {"candidate_family_id": "fam", "object_id": "m1", "ra": 10.0, "dec": 0.0, "membership_probability": 0.9},
            {"candidate_family_id": "fam", "object_id": "m2", "ra": 10.0027778, "dec": 0.0, "membership_probability": 0.8},
        ]
    )

    crossmatch, recovery, metrics = builder.build_reference_family_diagnostics(
        reference_path=reference_path,
        master=master,
        families=pd.DataFrame([{"candidate_family_id": "fam"}]),
        family_members=members,
        match_radius_arcsec=0.5,
    )

    assert len(crossmatch) == 3
    assert metrics["n_reference_images"] == 3
    assert metrics["n_reference_master_matches"] == 2
    assert metrics["n_reference_recoverable_families"] == 1
    assert metrics["n_reference_recovered_families"] == 1
    indexed = recovery.set_index("reference_family_id")
    assert bool(indexed.loc["1", "recoverable_family"])
    assert bool(indexed.loc["1", "recovered_family"])


def test_strong_lensing_rescue_keeps_bad_f814w_flag_candidate() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _image_candidate_row(
                "rescued",
                zspec=2.0,
                rank=3,
                shipley_flf814w=3,
                shipley_frad=3.0,
                shipley_aimg=3.0,
                shipley_bimg=2.0,
                magerr_F814W=0.1,
            ),
            _image_candidate_row(
                "bad-use",
                zspec=2.0,
                rank=3,
                shipley_use=0,
                shipley_frad=3.0,
                shipley_aimg=3.0,
                shipley_bimg=2.0,
                magerr_F814W=0.1,
            ),
        ]
    )

    candidates = builder.prepare_candidates(catalog, spec)
    indexed = candidates.set_index("object_id")

    assert candidates["object_id"].tolist() == ["rescued"]
    assert bool(indexed.loc["rescued", "image_family_rescue_selected"])
    assert "rescued_bad_f814w_flag" in indexed.loc["rescued", "image_family_review_flags"]
    assert candidates.attrs["image_family_selection_metrics"]["n_image_family_rescue_candidates"] == 1


def test_faint_strong_lensing_rescue_requires_strong_redshift_evidence() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _image_candidate_row(
                "strong-faint",
                zspec=2.0,
                rank=3,
                shipley_flf814w=3,
                mags={"F814W": 29.5},
            ),
            _image_candidate_row(
                "photo-faint",
                zspec=np.nan,
                rank=np.nan,
                zphot=2.0,
                zphot_low=1.8,
                zphot_high=2.2,
                shipley_flf814w=3,
                mags={"F814W": 29.5},
            ),
        ]
    )

    candidates = builder.prepare_candidates(catalog, spec)

    assert candidates["object_id"].tolist() == ["strong-faint"]
    assert "rescued_too_faint_f814w" in candidates.iloc[0]["image_family_review_flags"]


def test_red_sequence_fit_recovers_bright_no_spec_candidate_and_rejects_blue_row() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    seeds = [
        _member_row(f"seed-{index}", zspec=0.374 + index * 0.0003, rank=3, confidence="secure", ra_offset_arcsec=index)
        for index in range(5)
    ]
    red_candidate = _member_row(
        "red-no-spec",
        ra_offset_arcsec=10.0,
        mags={"F435W": 24.8, "F606W": 23.8, "F814W": 22.8, "F105W": 22.4, "F125W": 22.2, "F160W": 22.0},
    )
    blue_candidate = _member_row(
        "blue-no-spec",
        ra_offset_arcsec=12.0,
        mags={"F435W": 23.0, "F606W": 22.8, "F814W": 22.7, "F105W": 22.6, "F125W": 22.5, "F160W": 22.4},
    )
    catalog = pd.DataFrame([*seeds, red_candidate, blue_candidate])

    scores, red_sequence, _manifest = builder.score_cluster_members(catalog, spec)
    indexed = scores.set_index("object_id")

    assert not red_sequence.empty
    assert indexed.loc["red-no-spec", "member_probability"] >= 0.50
    assert bool(indexed.loc["red-no-spec", "cluster_member_selected"])
    assert indexed.loc["red-no-spec", "member_class"] == "red_sequence_member"
    assert indexed.loc["blue-no-spec", "member_probability"] < 0.50


def test_blue_secure_spec_member_is_retained_despite_red_sequence_mismatch() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    seeds = [
        _member_row(f"seed-{index}", zspec=0.374 + index * 0.0003, rank=3, confidence="secure", ra_offset_arcsec=index)
        for index in range(5)
    ]
    blue_spec = _member_row(
        "blue-spec",
        zspec=0.375,
        rank=3,
        confidence="secure",
        ra_offset_arcsec=20.0,
        mags={"F435W": 23.0, "F606W": 22.8, "F814W": 22.7, "F105W": 22.6, "F125W": 22.5, "F160W": 22.4},
    )
    catalog = pd.DataFrame([*seeds, blue_spec])

    scores, _red_sequence, _manifest = builder.score_cluster_members(catalog, spec)
    row = scores.set_index("object_id").loc["blue-spec"]

    assert row["member_probability"] >= 0.95
    assert row["member_class"] == "secure_spec_member"
    assert "blue_or_off_sequence_spec_member" in row["member_selection_note"]


def test_red_sequence_outlier_spec_seed_is_clipped_from_fit() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    seeds = [
        _member_row(
            f"seed-{index}",
            zspec=0.374 + index * 0.0002,
            rank=3,
            confidence="secure",
            ra_offset_arcsec=index,
            mags={
                "F435W": 22.0 + 0.30 * index,
                "F606W": 21.0 + 0.28 * index,
                "F814W": 20.0 + 0.25 * index,
                "F105W": 19.6 + 0.24 * index,
                "F125W": 19.4 + 0.23 * index,
                "F160W": 19.2 + 0.22 * index,
            },
        )
        for index in range(8)
    ]
    outlier = _member_row(
        "blue-spec-outlier",
        zspec=0.375,
        rank=3,
        confidence="secure",
        ra_offset_arcsec=20.0,
        mags={"F435W": 23.0, "F606W": 22.7, "F814W": 22.6, "F105W": 22.5, "F125W": 22.4, "F160W": 22.3},
    )
    red_no_spec = _member_row(
        "red-no-spec",
        ra_offset_arcsec=25.0,
        mags={"F435W": 24.24, "F606W": 23.16, "F814W": 21.93, "F105W": 21.45, "F125W": 21.18, "F160W": 20.9},
    )
    blue_no_spec = _member_row(
        "blue-no-spec",
        ra_offset_arcsec=28.0,
        mags={"F435W": 23.1, "F606W": 22.8, "F814W": 22.7, "F105W": 22.6, "F125W": 22.5, "F160W": 22.4},
    )
    catalog = pd.DataFrame([*seeds, outlier, red_no_spec, blue_no_spec])

    scores, red_sequence, _manifest = builder.score_cluster_members(catalog, spec)
    indexed = scores.set_index("object_id")

    assert (red_sequence["n_used"] < red_sequence["n_seed"]).any()
    assert red_sequence["scatter_mag"].max() <= 0.20
    assert indexed.loc["blue-spec-outlier", "member_probability"] >= 0.95
    assert not bool(indexed.loc["blue-no-spec", "cluster_member_selected"])
    assert bool(indexed.loc["red-no-spec", "cluster_member_selected"])


def test_one_plane_red_sequence_only_row_is_not_selected() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    seeds = [
        _member_row(
            f"sparse-{index}",
            zspec=0.375 + index * 0.0001,
            rank=3,
            confidence="secure",
            ra_offset_arcsec=index,
            missing_bands={"F435W", "F606W", "F105W", "F125W"},
        )
        for index in range(5)
    ]
    candidate = _member_row(
        "one-plane-no-spec",
        ra_offset_arcsec=12.0,
        missing_bands={"F435W", "F606W", "F105W", "F125W"},
    )
    catalog = pd.DataFrame([*seeds, candidate])

    scores, red_sequence, _manifest = builder.score_cluster_members(catalog, spec)
    row = scores.set_index("object_id").loc["one-plane-no-spec"]

    assert len(red_sequence) == 1
    assert row["red_sequence_n_planes"] == 1
    assert row["member_probability"] < builder.DEFAULT_MEMBER_PROBABILITY_THRESHOLD
    assert not bool(row["cluster_member_selected"])


def test_red_sequence_photoz_needs_two_valid_planes_to_select() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    seeds = [
        _member_row(f"seed-{index}", zspec=0.375 + index * 0.0001, rank=3, confidence="secure", ra_offset_arcsec=index)
        for index in range(5)
    ]
    good = _member_row(
        "red-photo-good",
        ra_offset_arcsec=10.0,
        zphot=0.375,
        zphot_low=0.34,
        zphot_high=0.41,
        mags={"F435W": 24.8, "F606W": 23.8, "F814W": 22.8, "F105W": 22.4, "F125W": 22.2, "F160W": 22.0},
    )
    sparse = _member_row(
        "red-photo-sparse",
        ra_offset_arcsec=12.0,
        zphot=0.375,
        zphot_low=0.34,
        zphot_high=0.41,
        missing_bands={"F435W", "F606W", "F105W", "F125W"},
    )
    catalog = pd.DataFrame([*seeds, good, sparse])

    scores, _red_sequence, _manifest = builder.score_cluster_members(catalog, spec)
    indexed = scores.set_index("object_id")

    assert bool(indexed.loc["red-photo-good", "cluster_member_selected"])
    assert indexed.loc["red-photo-good", "member_selection_evidence"] == "red_sequence_photoz"
    assert indexed.loc["red-photo-sparse", "red_sequence_n_planes"] == 1
    assert not bool(indexed.loc["red-photo-sparse", "cluster_member_selected"])


def test_bcg_special_members_are_excluded_from_default_potfile(tmp_path: Path) -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _member_row("bright-bcg", zspec=0.375, rank=3, confidence="secure", mags={"F814W": 19.0, "F160W": 18.0}),
            _member_row(
                "lensing-member",
                zspec=0.376,
                rank=3,
                confidence="secure",
                ra_offset_arcsec=20.0,
                mags={"F814W": 24.5, "F160W": 20.0},
            ),
        ]
    )
    args = SimpleNamespace(
        member_probability_threshold=builder.DEFAULT_MEMBER_PROBABILITY_THRESHOLD,
        lensing_member_probability_threshold=builder.DEFAULT_LENSING_MEMBER_PROBABILITY_THRESHOLD,
        lensing_bright_mag_f160w=builder.DEFAULT_LENSING_BRIGHT_MAG_F160W,
        member_faint_mag_f814w=builder.DEFAULT_MEMBER_FAINT_MAG_F814W,
        bcg_special_max=1,
    )

    scores, manifest = builder._write_member_outputs(
        output_dir=tmp_path,
        spec=spec,
        catalog=catalog,
        args=args,
        master_path=tmp_path / "a370" / "a370_master_catalog.csv",
    )
    potfile = Path(manifest["member_potfile_path"]).read_text(encoding="utf-8")

    assert bool(scores.set_index("object_id").loc["bright-bcg", "bcg_special_member_candidate"])
    assert "bright-bcg" not in potfile
    assert "lensing-member" in potfile
    assert "24.5000" in potfile
    assert "20.0000" not in potfile


def test_member_outputs_enforce_f814w_window_and_potfile_magnitudes(tmp_path: Path) -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    catalog = pd.DataFrame(
        [
            _member_row("bcg", zspec=0.375, rank=3, confidence="secure", mags={"F814W": 19.0, "F160W": 18.0}),
            _member_row(
                "member",
                zspec=0.376,
                rank=3,
                confidence="secure",
                ra_offset_arcsec=20.0,
                mags={"F814W": 24.5, "F160W": 20.0},
            ),
            _member_row(
                "too-faint",
                zspec=0.376,
                rank=3,
                confidence="secure",
                ra_offset_arcsec=40.0,
                mags={"F814W": 25.5, "F160W": 20.0},
            ),
        ]
    )
    args = SimpleNamespace(
        member_probability_threshold=builder.DEFAULT_MEMBER_PROBABILITY_THRESHOLD,
        lensing_member_probability_threshold=builder.DEFAULT_LENSING_MEMBER_PROBABILITY_THRESHOLD,
        lensing_bright_mag_f160w=builder.DEFAULT_LENSING_BRIGHT_MAG_F160W,
        member_faint_mag_f814w=25.0,
        bcg_special_max=1,
    )

    scores, manifest = builder._write_member_outputs(
        output_dir=tmp_path,
        spec=spec,
        catalog=catalog,
        args=args,
        master_path=tmp_path / "a370" / "a370_master_catalog.csv",
    )
    indexed = scores.set_index("object_id")
    selected = pd.read_csv(builder.cluster_member_catalog_paths(tmp_path, spec)[1])
    special = pd.read_csv(builder.cluster_member_catalog_paths(tmp_path, spec)[4])
    potfile = Path(manifest["member_potfile_path"]).read_text(encoding="utf-8")

    assert bool(indexed.loc["bcg", "cluster_member_selected"])
    assert bool(indexed.loc["member", "cluster_member_selected"])
    assert not bool(indexed.loc["too-faint", "cluster_member_selected"])
    assert not bool(indexed.loc["too-faint", "member_for_lensing"])
    assert indexed.loc["too-faint", "member_f814w_reject_reason"] == "too_faint_f814w"
    assert set(selected["object_id"]) == {"bcg", "member"}
    assert special["mag_F814W"].le(25.0).all()
    assert "too-faint" not in potfile
    assert "24.5000" in potfile
    assert "20.0000" not in potfile
    assert manifest["member_bcg_mag_f814w"] == 19.0
    assert manifest["member_faint_mag_f814w"] == 25.0
    assert manifest["n_member_f814w_window_rejected"] == 1


def test_score_cluster_members_reports_subprogress() -> None:
    catalog = pd.DataFrame(
        [
            _member_row("secure-a", zspec=0.375, rank=3, confidence="secure"),
            _member_row("secure-b", zspec=0.376, rank=2, confidence="probable"),
        ]
    )
    progress = FakeProgress()

    builder.score_cluster_members(catalog, builder.CLUSTER_BY_KEY["a370"], progress=progress)

    labels = [label for label, _total in progress.started]
    assert any("fitting member red sequence" in label for label in labels)
    assert any("scoring member rows" in label for label in labels)
    assert progress.advanced >= len(catalog)
    assert progress.finished >= 2


def test_score_candidate_pairs_reports_subprogress() -> None:
    rows: list[dict[str, object]] = []
    for idx, offset in enumerate([0.0, 2.0]):
        row: dict[str, object] = {
            "object_id": f"bg-{idx}",
            "ra": 10.0 + offset / 3600.0,
            "dec": 0.0,
            "zspec_best": 2.0,
            "zspec_best_confidence_rank": 3.0,
            "zspec_best_confidence": "secure",
            "zspec_best_native_quality": 3.0,
            "zphot_best": 2.0,
        }
        for column in builder.MAG_COLUMNS:
            row[column] = 24.0 + idx
        rows.append(row)
    progress = FakeProgress()

    pairs = builder.score_candidate_pairs(
        pd.DataFrame(rows),
        builder.CLUSTER_BY_KEY["a370"],
        min_common_bands=5,
        progress=progress,
    )

    labels = [label for label, _total in progress.started]
    assert len(pairs) == 1
    assert any("finding nearby candidate pairs" in label for label in labels)
    assert any("scoring candidate pairs" in label for label in labels)
    assert progress.advanced >= 1
    assert progress.finished >= 2


def test_jax_score_candidate_pairs_reports_progress_from_caller() -> None:
    rows = [
        _member_row(f"bg-{idx}", zspec=2.0, rank=3, confidence="secure", ra_offset_arcsec=float(offset))
        for idx, offset in enumerate([0.0, 2.0, 4.0])
    ]
    progress = FakeProgress()

    pairs = builder.score_candidate_pairs(
        pd.DataFrame(rows),
        builder.CLUSTER_BY_KEY["a370"],
        min_common_bands=5,
        family_pair_batch_size=2,
        progress=progress,
    )

    assert len(pairs) == 3
    assert progress.advanced >= 3
    assert progress.finished >= 2


def test_tqdm_catalog_progress_smoke_with_recording_console() -> None:
    console = builder.Console(record=True)

    with builder.make_progress(console) as progress:
        reporter = builder.CatalogProgress(progress, total_clusters=1, update_interval=2)
        reporter.set_cluster_phase("a370: cluster")
        reporter.start_step("a370: determinate step", total=3)
        reporter.advance_step()
        reporter.advance_step()
        reporter.advance_step()
        reporter.finish_step()
        reporter.start_step("a370: indeterminate step", total=None)
        reporter.advance_step()
        reporter.finish_step()
        reporter.advance_cluster()


def test_cli_subcommands_parse_stage_specific_options(tmp_path: Path) -> None:
    default_args = builder._parse_args(["all", "--clusters", "a370"])
    master_args = builder._parse_args(["master", "--output-dir", str(tmp_path), "--clusters", "a370"])
    members_args = builder._parse_args(["members", "--output-dir", str(tmp_path), "--clusters", "a370"])
    families_args = builder._parse_args(
        [
            "families",
            "--output-dir",
            str(tmp_path),
            "--clusters",
            "a370",
            "--family-pair-batch-size",
            "8",
            "--family-pair-diagnostics",
            "accepted",
            "--image-family-fov-kpc",
            "1000",
            "--family-color-rms-max",
            "0.3",
            "--family-photoz-delta-max",
            "0.8",
        ]
    )
    all_args = builder._parse_args(["all", "--output-dir", str(tmp_path), "--clusters", "a370"])
    all_plot_args = builder._parse_args(
        [
            "all",
            "--output-dir",
            str(tmp_path),
            "--clusters",
            "a370",
            "--plot-kind",
            "publication",
            "--image-background",
            "never",
            "--image-dir",
            str(tmp_path / "images"),
            "--image-scale",
            "30mas",
            "--family-cutout-size-arcsec",
            "8",
            "--family-cutout-circle-radius-arcsec",
            "0.8",
            "--family-cutout-families-per-page",
            "2",
            "--family-cutout-bands",
            "F475W",
            "F625W",
            "F160W",
        ]
    )
    plots_args = builder._parse_args(
        [
            "plots",
            "--output-dir",
            str(tmp_path),
            "--clusters",
            "a370",
            "--plot-kind",
            "diagnostic",
            "--image-background",
            "never",
        ]
    )

    assert default_args.output_dir == Path("results") / "hff_master_catalogs"
    assert master_args.command == "master"
    assert master_args.output_dir == tmp_path
    assert master_args.clusters == ["a370"]
    assert master_args.match_radius_arcsec == builder.DEFAULT_MATCH_RADIUS_ARCSEC
    assert master_args.redshift_dir == builder.DEFAULT_REDSHIFT_DIR
    assert master_args.redshift_match_radius_arcsec == builder.DEFAULT_REDSHIFT_MATCH_RADIUS_ARCSEC
    assert master_args.lagattuta22_path == builder.DEFAULT_LAGATTUTA22_PATH
    assert members_args.command == "members"
    assert members_args.member_probability_threshold == builder.DEFAULT_MEMBER_PROBABILITY_THRESHOLD
    assert members_args.member_faint_mag_f814w == builder.DEFAULT_MEMBER_FAINT_MAG_F814W
    assert families_args.command == "families"
    assert builder.DEFAULT_MAX_FAMILY_SPAN_KPC == 600.0
    assert builder.DEFAULT_IMAGE_FAMILY_FOV_KPC == 1000.0
    assert families_args.max_family_span_kpc == builder.DEFAULT_MAX_FAMILY_SPAN_KPC
    assert families_args.family_pair_batch_size == 8
    assert families_args.family_pair_diagnostics == "accepted"
    assert families_args.image_bright_mag_f814w == builder.DEFAULT_IMAGE_BRIGHT_MAG_F814W
    assert families_args.image_hff_faint_mag_f814w == builder.DEFAULT_IMAGE_HFF_FAINT_MAG_F814W
    assert families_args.image_outer_faint_mag_f814w == builder.DEFAULT_IMAGE_OUTER_FAINT_MAG_F814W
    assert families_args.image_min_size_arcsec == builder.DEFAULT_IMAGE_MIN_SIZE_ARCSEC
    assert families_args.image_size_pixel_scale_arcsec == builder.DEFAULT_IMAGE_SIZE_PIXEL_SCALE_ARCSEC
    assert families_args.image_photoz_min_mag_f160w == builder.DEFAULT_IMAGE_PHOTOZ_MIN_MAG_F160W
    assert families_args.image_photoz_max_mag_f160w == builder.DEFAULT_IMAGE_PHOTOZ_MAX_MAG_F160W
    assert families_args.image_photoz_min_nb_used == builder.DEFAULT_IMAGE_PHOTOZ_MIN_NB_USED
    assert families_args.image_photoz_max_dz_norm == builder.DEFAULT_IMAGE_PHOTOZ_MAX_DZ_NORM
    assert families_args.strong_lensing_rescue_faint_mag_f814w == builder.DEFAULT_STRONG_LENSING_RESCUE_FAINT_MAG_F814W
    assert families_args.strong_lensing_rescue_min_bands == builder.DEFAULT_STRONG_LENSING_RESCUE_MIN_BANDS
    assert families_args.reference_family_path is None
    assert families_args.reference_match_radius_arcsec == builder.DEFAULT_REFERENCE_MATCH_RADIUS_ARCSEC
    assert families_args.image_family_fov_kpc == 1000.0
    assert families_args.family_color_rms_max == 0.3
    assert families_args.family_photoz_delta_max == 0.8
    assert all_args.command == "all"
    assert all_args.min_common_bands == builder.DEFAULT_MIN_COMMON_BANDS
    assert all_args.family_pair_batch_size == builder.DEFAULT_FAMILY_PAIR_BATCH_SIZE
    assert all_args.match_radius_arcsec == builder.DEFAULT_MATCH_RADIUS_ARCSEC
    assert all_args.redshift_dir == builder.DEFAULT_REDSHIFT_DIR
    assert all_args.redshift_match_radius_arcsec == builder.DEFAULT_REDSHIFT_MATCH_RADIUS_ARCSEC
    assert all_args.lagattuta22_path == builder.DEFAULT_LAGATTUTA22_PATH
    assert all_args.lensing_bright_mag_f160w == builder.DEFAULT_LENSING_BRIGHT_MAG_F160W
    assert all_args.member_faint_mag_f814w == builder.DEFAULT_MEMBER_FAINT_MAG_F814W
    assert all_args.image_bright_mag_f814w == builder.DEFAULT_IMAGE_BRIGHT_MAG_F814W
    assert all_args.image_photoz_max_dz_norm == builder.DEFAULT_IMAGE_PHOTOZ_MAX_DZ_NORM
    assert all_args.strong_lensing_rescue_faint_mag_f814w == builder.DEFAULT_STRONG_LENSING_RESCUE_FAINT_MAG_F814W
    assert all_args.strong_lensing_rescue_min_bands == builder.DEFAULT_STRONG_LENSING_RESCUE_MIN_BANDS
    assert all_args.image_family_fov_kpc == builder.DEFAULT_IMAGE_FAMILY_FOV_KPC
    assert all_args.family_color_rms_max == builder.DEFAULT_FAMILY_COLOR_RMS_MAX
    assert all_args.family_photoz_delta_max == builder.DEFAULT_FAMILY_PHOTOZ_DELTA_MAX
    assert all_args.plot_kind == "all"
    assert all_args.image_background == "auto"
    assert all_args.image_dir == builder.DEFAULT_IMAGE_DIR
    assert all_args.image_scale == builder.DEFAULT_IMAGE_SCALE
    assert builder.DEFAULT_FAMILY_CUTOUT_SIZE_ARCSEC == 5.0
    assert all_args.family_cutout_size_arcsec == builder.DEFAULT_FAMILY_CUTOUT_SIZE_ARCSEC
    assert all_args.family_cutout_circle_radius_arcsec == builder.DEFAULT_FAMILY_CUTOUT_CIRCLE_RADIUS_ARCSEC
    assert all_args.family_cutout_families_per_page == builder.DEFAULT_FAMILY_CUTOUT_FAMILIES_PER_PAGE
    assert not hasattr(all_args, "family_cutout_max_images_per_family")
    assert not hasattr(all_args, "family_cutout_max_families")
    assert tuple(all_args.family_cutout_bands) == builder.DEFAULT_FAMILY_CUTOUT_BANDS
    assert all_plot_args.command == "all"
    assert all_plot_args.plot_kind == "publication"
    assert all_plot_args.image_background == "never"
    assert all_plot_args.image_dir == tmp_path / "images"
    assert all_plot_args.image_scale == "30mas"
    assert all_plot_args.family_cutout_size_arcsec == 8.0
    assert all_plot_args.family_cutout_circle_radius_arcsec == 0.8
    assert all_plot_args.family_cutout_families_per_page == 2
    assert not hasattr(all_plot_args, "family_cutout_max_images_per_family")
    assert not hasattr(all_plot_args, "family_cutout_max_families")
    assert all_plot_args.family_cutout_bands == ["F475W", "F625W", "F160W"]
    assert plots_args.command == "plots"
    assert plots_args.plot_kind == "diagnostic"
    assert plots_args.image_background == "never"
    assert plots_args.image_dir == builder.DEFAULT_IMAGE_DIR
    assert plots_args.image_scale == builder.DEFAULT_IMAGE_SCALE
    assert plots_args.family_cutout_circle_radius_arcsec == builder.DEFAULT_FAMILY_CUTOUT_CIRCLE_RADIUS_ARCSEC
    assert tuple(plots_args.family_cutout_bands) == builder.DEFAULT_FAMILY_CUTOUT_BANDS


def test_removed_family_backend_and_worker_flags_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        builder._parse_args(["families", "--output-dir", str(tmp_path), "--clusters", "a370", "--family-workers", "2"])
    with pytest.raises(SystemExit):
        builder._parse_args(
            ["families", "--output-dir", str(tmp_path), "--clusters", "a370", "--family-score-backend", "legacy"]
        )


def test_all_command_runs_catalogs_then_plots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_run_master_stage(
        args: object,
        _console: object,
        selected_specs: list[object],
        *,
        build_members: bool,
        build_families: bool,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        calls.append("catalogs")
        assert getattr(args, "command") == "all"
        assert getattr(args, "output_dir") == tmp_path
        assert getattr(args, "plot_kind") == "diagnostic"
        assert [spec.key for spec in selected_specs] == ["a370"]
        assert build_members is True
        assert build_families is True
        return [], [], []

    def fake_run_plots_stage(
        args: object,
        _console: object,
        selected_specs: list[object],
    ) -> list[dict[str, object]]:
        calls.append("plots")
        assert getattr(args, "command") == "all"
        assert getattr(args, "plot_kind") == "diagnostic"
        assert getattr(args, "image_background") == "never"
        assert [spec.key for spec in selected_specs] == ["a370"]
        return []

    monkeypatch.setattr(builder, "run_master_stage", fake_run_master_stage)
    monkeypatch.setattr(builder, "run_plots_stage", fake_run_plots_stage)

    builder.main(
        [
            "all",
            "--output-dir",
            str(tmp_path),
            "--clusters",
            "a370",
            "--plot-kind",
            "diagnostic",
            "--image-background",
            "never",
        ]
    )

    assert calls == ["catalogs", "plots"]


def test_per_cluster_output_paths_are_under_cluster_folder(tmp_path: Path) -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]

    assert builder.master_catalog_path(tmp_path, spec) == tmp_path / "a370" / "a370_master_catalog.csv"
    assert builder.match_audit_path(tmp_path, spec) == tmp_path / "a370" / "a370_match_audit.csv"
    score_path, selected_path, potfile_path, red_sequence_path, special_path = builder.cluster_member_catalog_paths(
        tmp_path,
        spec,
    )
    assert score_path == tmp_path / "a370" / "a370_cluster_member_scores.csv"
    assert selected_path == tmp_path / "a370" / "a370_cluster_members.csv"
    assert potfile_path == tmp_path / "a370" / "a370_cluster_members_potfile.cat"
    assert red_sequence_path == tmp_path / "a370" / "a370_cluster_member_red_sequence.csv"
    assert special_path == tmp_path / "a370" / "a370_bcg_special_member_candidates.csv"
    family_path, member_path, pair_path = builder.family_catalog_paths(tmp_path, spec)
    assert family_path == tmp_path / "a370" / "a370_candidate_image_families.csv"
    assert member_path == tmp_path / "a370" / "a370_candidate_family_members.csv"
    assert pair_path == tmp_path / "a370" / "a370_candidate_family_pairs.csv"
    diagnostic_dir, publication_dir = builder.plot_output_dirs(tmp_path, spec)
    assert diagnostic_dir == tmp_path / "a370" / "plots" / "diagnostic"
    assert publication_dir == tmp_path / "a370" / "plots" / "publication"
    assert builder.catalog_plot_manifest_path(tmp_path) == tmp_path / "hff_catalog_plot_manifest.csv"


def test_families_stage_fails_when_per_cluster_master_catalog_missing(tmp_path: Path) -> None:
    args = builder._parse_args(["families", "--output-dir", str(tmp_path), "--clusters", "a370"])

    with pytest.raises(builder.MissingCatalogError, match="Missing master catalog for a370"):
        builder.run_families_stage(args, builder.Console(record=True), [builder.CLUSTER_BY_KEY["a370"]])


def test_members_stage_fails_when_per_cluster_master_catalog_missing(tmp_path: Path) -> None:
    args = builder._parse_args(["members", "--output-dir", str(tmp_path), "--clusters", "a370"])

    with pytest.raises(builder.MissingCatalogError, match="Missing master catalog for a370"):
        builder.run_members_stage(args, builder.Console(record=True), [builder.CLUSTER_BY_KEY["a370"]])


def _write_plot_fixture(root: Path, spec: builder.ClusterSpec, *, optional: bool = True) -> None:
    cluster_dir = root / spec.key
    cluster_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _member_row(
            "member-a",
            zspec=spec.z_lens,
            rank=3,
            confidence="secure",
            zphot=spec.z_lens,
            zphot_low=spec.z_lens - 0.03,
            zphot_high=spec.z_lens + 0.03,
            ra_offset_arcsec=0.0,
        ),
        _member_row(
            "member-b",
            zspec=spec.z_lens + 0.0005,
            rank=2,
            confidence="probable",
            zphot=spec.z_lens,
            zphot_low=spec.z_lens - 0.03,
            zphot_high=spec.z_lens + 0.03,
            ra_offset_arcsec=5.0,
        ),
        _member_row(
            "background-a",
            zspec=2.0,
            rank=3,
            confidence="secure",
            zphot=2.0,
            zphot_low=1.8,
            zphot_high=2.2,
            ra_offset_arcsec=10.0,
            mags={"F160W": 24.0},
        ),
    ]
    master = pd.DataFrame(rows)
    master.to_csv(builder.master_catalog_path(root, spec), index=False)
    if not optional:
        return

    match_audit = pd.DataFrame(
        {
            "cluster_key": [spec.key],
            "match_type": ["pagul2024_shipley2018"],
            "left_index": [0],
            "right_index": [10],
            "separation_arcsec": [0.08],
        }
    )
    match_audit.to_csv(builder.match_audit_path(root, spec), index=False)

    scores = master.copy()
    scores["member_probability"] = [0.98, 0.93, 0.0]
    scores["member_class"] = ["secure_spec_member", "probable_spec_member", "rejected_foreground_background"]
    scores["member_selection_evidence"] = ["specz_velocity", "specz_velocity", ""]
    scores["cluster_member_selected"] = [True, True, False]
    scores["member_for_lensing"] = [True, True, False]
    scores["bcg_special_member_candidate"] = [True, False, False]
    scores["member_delta_v_kms"] = [0.0, 110.0, 355000.0]
    scores["member_specz_score"] = [0.98, 0.93, 0.0]
    scores["red_sequence_n_planes"] = [2, 2, 2]
    scores["red_sequence_n_consistent"] = [2, 2, 0]
    scores["red_sequence_score"] = [0.95, 0.90, 0.1]
    score_path, selected_path, _potfile_path, red_sequence_path, special_path = builder.cluster_member_catalog_paths(root, spec)
    scores.to_csv(score_path, index=False)
    scores.loc[scores["cluster_member_selected"]].to_csv(selected_path, index=False)
    scores.loc[scores["bcg_special_member_candidate"]].to_csv(special_path, index=False)
    pd.DataFrame(
        [
            {
                "cluster_key": spec.key,
                "blue_band": "F606W",
                "red_band": "F814W",
                "mag_band": "F814W",
                "color_name": "F606W-F814W",
                "slope": 0.0,
                "intercept": 1.0,
                "scatter_mag": 0.08,
                "n_seed": 2,
                "n_used": 2,
            },
            {
                "cluster_key": spec.key,
                "blue_band": "F814W",
                "red_band": "F160W",
                "mag_band": "F160W",
                "color_name": "F814W-F160W",
                "slope": 0.0,
                "intercept": 0.8,
                "scatter_mag": 0.08,
                "n_seed": 2,
                "n_used": 2,
            },
        ]
    ).to_csv(red_sequence_path, index=False)

    family_path, family_member_path, pair_path = builder.family_catalog_paths(root, spec)
    pd.DataFrame(
        [
            {
                "cluster_key": spec.key,
                "candidate_family_id": f"{spec.key}_family_001",
                "n_images": 2,
                "family_probability": 0.92,
                "max_separation_arcsec": 5.0,
                "max_separation_kpc": 26.0,
                "family_z_best": 2.0,
                "family_z_method": "specz",
                "min_specz_confidence": "secure",
                "median_sed_rms": 0.02,
                "min_pair_score": 0.92,
                "review_flags": "",
            }
        ]
    ).to_csv(family_path, index=False)
    pd.DataFrame(
        [
            {
                "cluster_key": spec.key,
                "candidate_family_id": f"{spec.key}_family_001",
                "object_id": "member-a",
                "ra": master.loc[0, "ra"],
                "dec": master.loc[0, "dec"],
                "membership_probability": 0.92,
                "raw_probability": 0.92,
                "zspec_best": 2.0,
                "zspec_best_confidence": "secure",
                "zspec_best_native_quality": 3,
                "zphot_best": 2.0,
                "n_valid_bands": 6,
                "object_source": "pagul2024",
                "catalog_sources": "pagul2024",
            },
            {
                "cluster_key": spec.key,
                "candidate_family_id": f"{spec.key}_family_001",
                "object_id": "member-b",
                "ra": master.loc[1, "ra"],
                "dec": master.loc[1, "dec"],
                "membership_probability": 0.92,
                "raw_probability": 0.92,
                "zspec_best": 2.0,
                "zspec_best_confidence": "secure",
                "zspec_best_native_quality": 3,
                "zphot_best": 2.0,
                "n_valid_bands": 6,
                "object_source": "pagul2024",
                "catalog_sources": "pagul2024",
            },
        ]
    ).to_csv(family_member_path, index=False)
    pd.DataFrame(
        [
            {
                "cluster_key": spec.key,
                "left_object_id": "member-a",
                "right_object_id": "member-b",
                "separation_arcsec": 5.0,
                "separation_kpc": 26.0,
                "pair_score": 0.92,
                "specz_score": 1.0,
                "photoz_score": 0.0,
                "color_score": 0.95,
                "sed_rms": 0.02,
                "n_common_bands": 6,
                "hard_reject_reason": "",
                "redshift_relation": "both_specz",
            },
            {
                "cluster_key": spec.key,
                "left_object_id": "member-a",
                "right_object_id": "background-a",
                "separation_arcsec": 10.0,
                "separation_kpc": 52.0,
                "pair_score": 0.0,
                "specz_score": 0.0,
                "photoz_score": 0.0,
                "color_score": 0.0,
                "sed_rms": 0.9,
                "n_common_bands": 6,
                "hard_reject_reason": "secure_or_probable_specz_conflict",
                "redshift_relation": "secure_or_probable_specz_conflict",
            },
        ]
    ).to_csv(pair_path, index=False)


def _write_test_wcs_fits(path: Path, *, center_ra: float, center_dec: float, value: float = 1.0, size: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [float(size) / 2.0, float(size) / 2.0]
    wcs.wcs.cdelt = np.array([-0.0000166667, 0.0000166667])
    wcs.wcs.crval = [center_ra, center_dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    yy, xx = np.mgrid[0:size, 0:size]
    data = value + 0.01 * xx + 0.02 * yy
    fits.PrimaryHDU(data.astype(np.float32), header=wcs.to_header()).writeto(path, overwrite=True)


def test_overlay_crop_extent_matches_catalog_offset_convention(tmp_path: Path) -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    path = tmp_path / "hlsp_buffalo_abell370_f814w_test.fits"
    _write_test_wcs_fits(path, center_ra=10.0, center_dec=0.0, value=1.0)

    crop = builder._load_fits_crop_for_overlay(path, center_ra=10.0, center_dec=0.0, spec=spec)
    pixel_x = 32.0
    pixel_y = 31.0
    ra, dec = crop.wcs.wcs_pix2world(pixel_x, pixel_y, 0)
    ra = float(np.asarray(ra))
    dec = float(np.asarray(dec))
    projected = builder._add_projected_offsets(
        pd.DataFrame([{"ra": ra, "dec": dec}]),
        spec,
        center_ra=10.0,
        center_dec=0.0,
    )

    x0 = crop.x_min - 0.5 * crop.stride
    x1 = crop.x_min + (crop.data.shape[1] - 0.5) * crop.stride
    y0 = crop.y_min - 0.5 * crop.stride
    y1 = crop.y_min + (crop.data.shape[0] - 0.5) * crop.stride
    plot_x = crop.extent[0] + ((pixel_x - x0) / (x1 - x0)) * (crop.extent[1] - crop.extent[0])
    plot_y = crop.extent[2] + ((pixel_y - y0) / (y1 - y0)) * (crop.extent[3] - crop.extent[2])

    np.testing.assert_allclose(plot_x, projected["x_kpc"].iloc[0], atol=1.0e-3)
    np.testing.assert_allclose(plot_y, projected["y_kpc"].iloc[0], atol=1.0e-3)


def test_image_scale_prefers_matching_background_and_cutout_paths(tmp_path: Path) -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for scale in ("60mas", "30mas"):
        scale_dir = image_dir / scale
        scale_dir.mkdir()
        for band in builder.DEFAULT_FAMILY_CUTOUT_BANDS:
            (scale_dir / f"hlsp_buffalo_abell370_{band.lower()}_test_drz.fits").write_text("", encoding="utf-8")

    background_paths = builder._find_background_band_paths(image_dir, spec, image_scale="30mas")
    cutout_paths = builder._find_family_cutout_band_paths(
        image_dir,
        spec,
        builder.DEFAULT_FAMILY_CUTOUT_BANDS,
        image_scale="30mas",
    )

    assert all("30mas" in str(path.relative_to(image_dir)).lower() for path in background_paths.values())
    assert all("30mas" in str(path.relative_to(image_dir)).lower() for path in cutout_paths.values())


def test_spatial_plot_selection_keeps_square_fov_and_ranks_member_probability() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    scale = builder.kpc_per_arcsec(spec.z_lens)
    rows = pd.DataFrame(
        [
            _member_row("low-in", ra_offset_arcsec=10.0, zspec=spec.z_lens, rank=3),
            _member_row("high-in", ra_offset_arcsec=20.0, zspec=spec.z_lens, rank=3),
            _member_row("outside", ra_offset_arcsec=(builder.PLOT_FOV_HALF_WIDTH_KPC / scale) + 10.0),
        ]
    )
    rows["member_probability"] = [0.2, 0.9, 1.0]

    selected, metadata = builder._select_spatial_plot_rows(
        rows,
        spec,
        center_ra=10.0,
        center_dec=0.0,
        max_rows=1,
        probability_column="member_probability",
        label="cluster members",
    )

    assert selected["object_id"].tolist() == ["high-in"]
    assert metadata["n_spatial_input_rows"] == 3
    assert metadata["n_spatial_fov_rows"] == 2
    assert metadata["n_spatial_plotted_rows"] == 1
    assert selected["x_kpc"].abs().max() <= builder.PLOT_FOV_HALF_WIDTH_KPC


def test_spatial_plot_selection_ranks_family_membership_probability() -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    rows = pd.DataFrame(
        [
            {"candidate_family_id": "fam", "object_id": "a", "ra": 10.0, "dec": 0.0, "membership_probability": 0.2},
            {"candidate_family_id": "fam", "object_id": "b", "ra": 10.0 + 1.0 / 3600.0, "dec": 0.0, "membership_probability": 0.8},
        ]
    )

    selected, _metadata = builder._select_spatial_plot_rows(
        rows,
        spec,
        center_ra=10.0,
        center_dec=0.0,
        max_rows=1,
        probability_column="membership_probability",
        label="candidate image-family members",
    )

    assert selected["object_id"].tolist() == ["b"]


def test_plot_center_falls_back_to_master_catalog_without_members() -> None:
    master = pd.DataFrame(
        [
            {"object_id": "a", "ra": 10.0, "dec": 0.0},
            {"object_id": "b", "ra": 12.0, "dec": 2.0},
        ]
    )

    center_ra, center_dec, note = builder._choose_plot_center(master, None)

    assert center_ra == 11.0
    assert center_dec == 1.0
    assert note == "median_ra_dec_of_master_catalog"


def test_family_cutout_member_crms_uses_in_family_accepted_pairs() -> None:
    selected_members = pd.DataFrame(
        [
            {"candidate_family_id": "fam-a", "object_id": "a1"},
            {"candidate_family_id": "fam-a", "object_id": "a2"},
            {"candidate_family_id": "fam-a", "object_id": "a3"},
            {"candidate_family_id": "fam-b", "object_id": "b1"},
        ]
    )
    pairs = pd.DataFrame(
        [
            {"left_object_id": "a1", "right_object_id": "a2", "sed_rms": 0.2, "hard_reject_reason": ""},
            {"left_object_id": "a1", "right_object_id": "a3", "sed_rms": 0.6, "hard_reject_reason": ""},
            {"left_object_id": "a2", "right_object_id": "a3", "sed_rms": 1.0, "hard_reject_reason": ""},
            {"left_object_id": "a1", "right_object_id": "b1", "sed_rms": 0.01, "hard_reject_reason": ""},
            {"left_object_id": "a2", "right_object_id": "b1", "sed_rms": 0.02, "hard_reject_reason": "color_rms_too_large"},
        ]
    )

    crms = builder._family_cutout_member_crms(selected_members, pairs)

    assert crms[("fam-a", "a1")] == pytest.approx(0.4)
    assert crms[("fam-a", "a2")] == pytest.approx(0.6)
    assert crms[("fam-a", "a3")] == pytest.approx(0.8)
    assert ("fam-b", "b1") not in crms


def test_plots_stage_fails_when_per_cluster_master_catalog_missing(tmp_path: Path) -> None:
    args = builder._parse_args(["plots", "--output-dir", str(tmp_path), "--clusters", "a370"])

    with pytest.raises(builder.MissingCatalogError, match="Missing master catalog for a370"):
        builder.run_plots_stage(args, builder.Console(record=True), [builder.CLUSTER_BY_KEY["a370"]])


def test_plots_stage_writes_synthetic_plots_and_manifest(tmp_path: Path) -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    _write_plot_fixture(tmp_path, spec, optional=True)
    args = builder._parse_args(
        ["plots", "--output-dir", str(tmp_path), "--clusters", "a370", "--image-background", "never"]
    )

    rows = builder.run_plots_stage(args, builder.Console(record=True), [spec])
    manifest = pd.read_csv(builder.catalog_plot_manifest_path(tmp_path))

    assert rows
    assert not manifest.empty
    assert (tmp_path / "a370" / "plots" / "diagnostic" / "a370_master_sky_footprint.png").exists()
    assert (tmp_path / "a370" / "plots" / "publication" / "a370_catalog_overview.png").exists()
    assert (tmp_path / "a370" / "plots" / "publication" / "a370_catalog_overview.pdf").exists()
    assert (tmp_path / "a370" / "plots" / "publication" / "a370_cluster_image_overlay.png").exists()
    assert (tmp_path / "a370" / "plots" / "publication" / "a370_cluster_image_overlay.pdf").exists()
    assert (tmp_path / "plots" / "publication" / "hff_all_cluster_catalog_summary.png").exists()
    assert "generated" in set(manifest["status"])
    assert set(manifest.loc[manifest["status"].eq("generated"), "plot_kind"]) >= {"diagnostic", "publication"}
    cutouts = manifest.loc[manifest["plot_name"].eq("family_cutouts")].iloc[0]
    assert cutouts["status"] == "skipped"
    assert "background disabled" in cutouts["reason"]
    assert cutouts["family_cutout_bands"] == "|".join(builder.DEFAULT_FAMILY_CUTOUT_BANDS)
    spatial = manifest.loc[manifest["plot_name"].eq("member_sky_map")].iloc[0]
    assert spatial["n_spatial_plotted_rows"] <= builder.PLOT_MEMBER_MAX_ROWS
    assert spatial["plot_fov_half_width_kpc"] == builder.PLOT_FOV_HALF_WIDTH_KPC


def test_plots_stage_auto_background_falls_back_and_records_skips(tmp_path: Path) -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    _write_plot_fixture(tmp_path, spec, optional=False)
    args = builder._parse_args(
        [
            "plots",
            "--output-dir",
            str(tmp_path),
            "--clusters",
            "a370",
            "--plot-kind",
            "all",
            "--image-dir",
            str(tmp_path / "missing_images"),
            "--image-background",
            "auto",
        ]
    )

    builder.run_plots_stage(args, builder.Console(record=True), [spec])
    manifest = pd.read_csv(builder.catalog_plot_manifest_path(tmp_path))

    overview = manifest.loc[manifest["plot_name"].eq("catalog_overview")].iloc[0]
    assert overview["status"] == "generated"
    assert not bool(overview["used_background"])
    assert "No usable FITS background" in overview["reason"]
    assert "skipped" in set(manifest["status"])


def test_plots_stage_caps_many_spatial_points_in_manifest(tmp_path: Path) -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    _write_plot_fixture(tmp_path, spec, optional=True)
    score_path, _selected_path, _potfile_path, _red_sequence_path, _special_path = builder.cluster_member_catalog_paths(
        tmp_path,
        spec,
    )
    scores = pd.read_csv(score_path)
    extra_rows = []
    for index in range(700):
        row = _member_row(
            f"extra-{index}",
            zspec=spec.z_lens,
            rank=3,
            confidence="secure",
            ra_offset_arcsec=0.01 * index,
        )
        row["member_probability"] = 0.1 + 0.0001 * index
        row["cluster_member_selected"] = True
        row["member_for_lensing"] = False
        row["bcg_special_member_candidate"] = False
        row["member_delta_v_kms"] = 0.0
        row["member_specz_score"] = 0.98
        row["red_sequence_n_planes"] = 2
        row["red_sequence_n_consistent"] = 2
        row["red_sequence_score"] = 0.9
        extra_rows.append(row)
    pd.concat([scores, pd.DataFrame(extra_rows)], ignore_index=True).to_csv(score_path, index=False)
    args = builder._parse_args(
        [
            "plots",
            "--output-dir",
            str(tmp_path),
            "--clusters",
            "a370",
            "--plot-kind",
            "diagnostic",
            "--image-background",
            "never",
        ]
    )

    builder.run_plots_stage(args, builder.Console(record=True), [spec])
    manifest = pd.read_csv(builder.catalog_plot_manifest_path(tmp_path))
    member_sky = manifest.loc[manifest["plot_name"].eq("member_sky_map")].iloc[0]

    assert member_sky["n_spatial_input_rows"] > builder.PLOT_MEMBER_MAX_ROWS
    assert member_sky["n_spatial_plotted_rows"] == builder.PLOT_MEMBER_MAX_ROWS
    assert "member_probability" in member_sky["spatial_selection_note"]


def test_cluster_image_overlay_uses_grayscale_and_rgb_fits(tmp_path: Path) -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    _write_plot_fixture(tmp_path, spec, optional=True)
    image_dir = tmp_path / "images"
    _write_test_wcs_fits(image_dir / "hlsp_buffalo_abell370_f814w_test.fits", center_ra=10.0, center_dec=0.0, value=3.0)
    grayscale_args = builder._parse_args(
        [
            "plots",
            "--output-dir",
            str(tmp_path),
            "--clusters",
            "a370",
            "--plot-kind",
            "publication",
            "--image-dir",
            str(image_dir),
            "--image-background",
            "auto",
        ]
    )

    builder.run_plots_stage(grayscale_args, builder.Console(record=True), [spec])
    manifest = pd.read_csv(builder.catalog_plot_manifest_path(tmp_path))
    overlay = manifest.loc[manifest["plot_name"].eq("cluster_image_overlay")].iloc[0]
    assert overlay["image_render_mode"] == "grayscale"
    assert bool(overlay["used_background"])
    cutouts = manifest.loc[manifest["plot_name"].eq("family_cutouts")].iloc[0]
    assert cutouts["status"] == "skipped"
    assert "Missing RGB FITS" in cutouts["reason"]

    for band_index, band in enumerate(builder.DEFAULT_FAMILY_CUTOUT_BANDS, start=1):
        _write_test_wcs_fits(
            image_dir / f"hlsp_buffalo_abell370_{band.lower()}_test.fits",
            center_ra=10.0,
            center_dec=0.0,
            value=float(band_index),
            size=300,
        )

    builder.run_plots_stage(grayscale_args, builder.Console(record=True), [spec])
    manifest = pd.read_csv(builder.catalog_plot_manifest_path(tmp_path))
    overlay = manifest.loc[manifest["plot_name"].eq("cluster_image_overlay")].iloc[0]
    assert overlay["image_render_mode"] == "rgb"
    background = builder._load_cluster_overlay_background(
        spec,
        image_dir,
        "required",
        center_ra=10.0,
        center_dec=0.0,
        image_scale=builder.DEFAULT_IMAGE_SCALE,
    )
    assert background["mode"] == "rgb"
    assert "extent" in background
    assert background["extent"][0] > background["extent"][1]
    assert background["extent"][2] > background["extent"][3]
    assert (tmp_path / "a370" / "plots" / "publication" / "a370_cluster_image_overlay.png").exists()
    cutouts = manifest.loc[manifest["plot_name"].eq("family_cutouts")].iloc[0]
    assert cutouts["status"] == "generated"
    assert cutouts["n_cutout_families"] == 1
    assert cutouts["n_cutout_images"] == 2
    assert cutouts["family_cutout_size_arcsec"] == builder.DEFAULT_FAMILY_CUTOUT_SIZE_ARCSEC
    assert cutouts["family_cutout_circle_radius_arcsec"] == builder.DEFAULT_FAMILY_CUTOUT_CIRCLE_RADIUS_ARCSEC
    assert bool(cutouts["family_cutout_color_rms_label"])
    assert cutouts["image_scale"] == builder.DEFAULT_IMAGE_SCALE
    assert cutouts["family_cutout_bands"] == "|".join(builder.DEFAULT_FAMILY_CUTOUT_BANDS)
    assert (tmp_path / "a370" / "plots" / "publication" / "a370_candidate_family_cutouts.pdf").exists()


def test_plots_stage_required_background_missing_fails(tmp_path: Path) -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    _write_plot_fixture(tmp_path, spec, optional=False)
    args = builder._parse_args(
        [
            "plots",
            "--output-dir",
            str(tmp_path),
            "--clusters",
            "a370",
            "--plot-kind",
            "publication",
            "--image-dir",
            str(tmp_path / "missing_images"),
            "--image-background",
            "required",
        ]
    )

    with pytest.raises(builder.MissingCatalogError, match="No usable FITS background"):
        builder.run_plots_stage(args, builder.Console(record=True), [spec])


def test_family_cutouts_required_rgb_missing_fails(tmp_path: Path) -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    _write_plot_fixture(tmp_path, spec, optional=True)
    image_dir = tmp_path / "images"
    _write_test_wcs_fits(image_dir / "hlsp_buffalo_abell370_f814w_test.fits", center_ra=10.0, center_dec=0.0, value=3.0)
    args = builder._parse_args(
        [
            "plots",
            "--output-dir",
            str(tmp_path),
            "--clusters",
            "a370",
            "--plot-kind",
            "publication",
            "--image-dir",
            str(image_dir),
            "--image-background",
            "required",
        ]
    )

    with pytest.raises(builder.MissingCatalogError, match="Could not load RGB FITS for candidate family cutouts"):
        builder.run_plots_stage(args, builder.Console(record=True), [spec])


def test_family_cutouts_render_all_finite_members(tmp_path: Path) -> None:
    spec = builder.CLUSTER_BY_KEY["a370"]
    _write_plot_fixture(tmp_path, spec, optional=True)
    family_path, family_member_path, _pair_path = builder.family_catalog_paths(tmp_path, spec)
    families = pd.DataFrame(
        [
            {
                "cluster_key": spec.key,
                "candidate_family_id": f"{spec.key}_family_{index:03d}",
                "n_images": 3,
                "family_probability": probability,
                "max_separation_arcsec": 4.0,
                "max_separation_kpc": 20.0,
                "family_z_best": 2.0,
                "family_z_method": "specz",
                "min_specz_confidence": "secure",
                "median_sed_rms": 0.03,
                "min_pair_score": probability,
                "review_flags": "",
            }
            for index, probability in [(1, 0.95), (2, 0.85), (3, 0.75)]
        ]
    )
    members = []
    for family_index in range(1, 4):
        for image_index in range(3):
            members.append(
                {
                    "cluster_key": spec.key,
                    "candidate_family_id": f"{spec.key}_family_{family_index:03d}",
                    "object_id": f"f{family_index}-image-{image_index}",
                    "ra": 10.0 + 0.0001 * (family_index + image_index),
                    "dec": 0.0001 * image_index,
                    "membership_probability": 0.9 - 0.1 * image_index,
                    "raw_probability": 0.9 - 0.1 * image_index,
                    "zspec_best": 2.0,
                    "zspec_best_confidence": "secure",
                    "zspec_best_native_quality": 3,
                    "zphot_best": 2.0,
                    "n_valid_bands": 6,
                    "object_source": "pagul2024",
                    "catalog_sources": "pagul2024",
                }
            )
    families.to_csv(family_path, index=False)
    pd.DataFrame(members).to_csv(family_member_path, index=False)
    image_dir = tmp_path / "images"
    for band_index, band in enumerate(builder.DEFAULT_FAMILY_CUTOUT_BANDS, start=1):
        _write_test_wcs_fits(
            image_dir / f"hlsp_buffalo_abell370_{band.lower()}_test.fits",
            center_ra=10.0,
            center_dec=0.0,
            value=float(band_index),
            size=300,
        )
    args = builder._parse_args(
        [
            "plots",
            "--output-dir",
            str(tmp_path),
            "--clusters",
            "a370",
            "--plot-kind",
            "publication",
            "--image-dir",
            str(image_dir),
            "--image-background",
            "auto",
            "--image-scale",
            "30mas",
            "--family-cutout-families-per-page",
            "1",
            "--family-cutout-size-arcsec",
            "6",
        ]
    )

    builder.run_plots_stage(args, builder.Console(record=True), [spec])
    manifest = pd.read_csv(builder.catalog_plot_manifest_path(tmp_path))
    cutouts = manifest.loc[manifest["plot_name"].eq("family_cutouts")].iloc[0]

    assert cutouts["status"] == "generated"
    assert cutouts["n_cutout_families"] == 3
    assert cutouts["n_cutout_images"] == 9
    assert cutouts["family_cutout_size_arcsec"] == 6.0
    assert cutouts["family_cutout_circle_radius_arcsec"] == builder.DEFAULT_FAMILY_CUTOUT_CIRCLE_RADIUS_ARCSEC
    assert bool(cutouts["family_cutout_color_rms_label"])
    assert cutouts["image_scale"] == "30mas"
    assert (tmp_path / "a370" / "plots" / "publication" / "a370_candidate_family_cutouts.pdf").exists()


def test_all_stage_writes_master_member_and_family_manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shipley_path = tmp_path / "Shipley2018.fit"
    shipley_path.write_text("", encoding="utf-8")
    pagul_path = tmp_path / "pagul.fits"
    args = builder._parse_args(
        [
            "all",
            "--output-dir",
            str(tmp_path / "out"),
            "--shipley-path",
            str(shipley_path),
            "--clusters",
            "a370",
        ]
    )

    monkeypatch.setattr(builder, "locate_pagul_catalog", lambda _spec, _pagul_dir: pagul_path)
    monkeypatch.setattr(builder, "_table_to_dataframe", lambda _path: pd.DataFrame())

    def fake_build_cluster_catalog(**_kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        catalog = pd.DataFrame(
            [
                _member_row("member", zspec=0.375, rank=3, confidence="secure"),
                _member_row(
                    "background-a",
                    zspec=2.0,
                    rank=3,
                    confidence="secure",
                    ra_offset_arcsec=100.0,
                    mags={"F160W": 24.0},
                ),
            ]
        )
        audit = pd.DataFrame()
        manifest = {
            "cluster_key": "a370",
            "pagul_flux_scale": "cgs_fnu",
            "match_radius_arcsec": 0.5,
            "n_pagul_rows": 1,
            "n_shipley_rows": 0,
            "n_shipley_matched": 0,
            "n_shipley_unmatched_appended": 0,
            "n_output_rows": 2,
        }
        return catalog, audit, manifest

    monkeypatch.setattr(builder, "build_cluster_catalog", fake_build_cluster_catalog)

    master_rows, member_rows, family_rows = builder.run_master_stage(
        args,
        builder.Console(record=True),
        [builder.CLUSTER_BY_KEY["a370"]],
        build_members=True,
        build_families=True,
    )

    assert len(master_rows) == 1
    assert len(member_rows) == 1
    assert len(family_rows) == 1
    assert (tmp_path / "out" / "hff_master_catalog_manifest.csv").exists()
    assert (tmp_path / "out" / "hff_cluster_member_manifest.csv").exists()
    assert (tmp_path / "out" / "hff_image_family_manifest.csv").exists()


def test_real_master_catalog_member_selection_is_conservative_when_data_available() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "hff_master_catalogs"
    catalog_paths: list[tuple[str, Path]] = []
    for key in builder.CLUSTER_BY_KEY:
        flat_path = root / f"{key}_master_catalog.csv"
        nested_path = root / key / f"{key}_master_catalog.csv"
        if nested_path.exists():
            catalog_paths.append((key, nested_path))
        elif flat_path.exists():
            catalog_paths.append((key, flat_path))
    if not catalog_paths:
        pytest.skip("Local HFF master catalogs are not available.")

    for key, path in catalog_paths:
        scores, _red_sequence, _manifest = builder.score_cluster_members(
            pd.read_csv(path, low_memory=False),
            builder.CLUSTER_BY_KEY[key],
        )
        selected = scores["cluster_member_selected"].map(builder._bool_value)
        zspec = pd.to_numeric(scores["zspec_best"], errors="coerce").map(builder.valid_redshift)
        no_spec = ~np.isfinite(zspec)
        n_selected = int(selected.sum())
        one_plane_no_spec = selected & no_spec & (pd.to_numeric(scores["red_sequence_n_planes"], errors="coerce").fillna(0) <= 1)
        photo_only = selected & no_spec & scores["member_selection_evidence"].fillna("").eq("")

        assert 100 <= n_selected <= 3000, f"{key} selected {n_selected} members from {path}"
        assert int(one_plane_no_spec.sum()) == 0, key
        assert int(photo_only.sum()) == 0, key
