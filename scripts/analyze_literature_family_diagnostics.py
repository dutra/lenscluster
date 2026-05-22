#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-literature-family-diagnostics")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import astropy.units as u

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_hff_to_literature import (  # noqa: E402
    DEFAULT_CLUSTERS,
    DEFAULT_LITERATURE_ROOT,
    LiteratureCatalog,
    discover_literature_catalogs,
)


DEFAULT_CATALOG_ROOT = Path("results") / "hff_master_catalogs"
DEFAULT_PAGUL_ROOT = Path("data") / "Pagul2024"
DEFAULT_OUTPUT_DIR = Path("results") / "literature_family_diagnostics"
DEFAULT_MATCH_RADIUS_ARCSEC = 0.5
DEFAULT_MIN_MATCHED_IMAGES = 2
BACKGROUND_Z_MARGIN = 0.1
ZSPEC_CONFLICT_TOL = 0.005
STRONG_SPEC_RANK = 2.0
COLOR_RMS_REFERENCE = 0.30
INVALID_REDSHIFT_SENTINELS = {-999.0, -99.0, -9.0, 99.0, 999.0, 1.0e9}
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
CLUSTER_Z_LENS = {
    "a2744": 0.308,
    "a370": 0.375,
    "as1063": 0.348,
    "m0416": 0.396,
    "m0717": 0.545,
    "m1149": 0.543,
}
PAGUL_SLUGS_BY_CLUSTER = {
    "a2744": ("abell2744", "a2744"),
    "a370": ("abell370", "a370"),
    "as1063": ("abells1063", "as1063", "rxcj2248"),
    "m0416": ("macs0416", "m0416"),
    "m0717": ("macs0717", "m0717"),
    "m1149": ("macs1149", "m1149"),
}
MAG_BANDS = (
    "F275W",
    "F336W",
    "F435W",
    "F475W",
    "F606W",
    "F625W",
    "F814W",
    "F105W",
    "F110W",
    "F125W",
    "F140W",
    "F160W",
    "Ks",
    "I1",
    "I2",
)

MATCH_COLUMNS = [
    "cluster",
    "source_slug",
    "source_id",
    "source_path",
    "literature_row_index",
    "literature_id",
    "literature_family_id",
    "literature_image_id",
    "literature_ra",
    "literature_dec",
    "literature_z",
    "literature_mag",
    "matched",
    "match_separation_arcsec",
    "duplicate_master_match",
    "pagul_matched",
    "pagul_match_separation_arcsec",
    "pagul_object_id",
    "pagul_ra",
    "pagul_dec",
    "pagul_zspec",
    "pagul_zspec_q",
    "pagul_zphot",
    "pagul_zpdf_low",
    "pagul_zpdf_high",
    "pagul_chi2_red",
    "pagul_mag_f160w",
    "pagul_nb_used",
    "master_row_index",
    "master_object_id",
    "master_ra",
    "master_dec",
    "master_object_source",
    "master_catalog_sources",
    "master_zspec_best",
    "master_zspec_best_source",
    "master_zspec_best_confidence",
    "master_zspec_best_confidence_rank",
    "master_zspec_best_native_quality",
    "master_zspec_conflict",
    "master_zphot_best",
    "master_zpdf_low",
    "master_zpdf_high",
    "master_n_valid_bands",
]
for band in MAG_BANDS:
    MATCH_COLUMNS.append(f"master_mag_{band}")

PAIR_COLUMNS = [
    "cluster",
    "source_slug",
    "source_id",
    "literature_family_id",
    "left_literature_id",
    "right_literature_id",
    "left_master_object_id",
    "right_master_object_id",
    "duplicate_master_pair",
    "left_match_separation_arcsec",
    "right_match_separation_arcsec",
    "max_match_separation_arcsec",
    "literature_pair_separation_arcsec",
    "literature_pair_separation_kpc",
    "literature_z_left",
    "literature_z_right",
    "literature_z_delta",
    "master_zspec_left",
    "master_zspec_right",
    "master_zspec_delta",
    "left_zspec_confidence_rank",
    "right_zspec_confidence_rank",
    "left_zspec_confidence",
    "right_zspec_confidence",
    "left_zspec_source",
    "right_zspec_source",
    "both_strong_specz",
    "strong_specz_consistent",
    "strong_specz_conflict",
    "master_zphot_left",
    "master_zphot_right",
    "master_zphot_delta",
    "zpdf_overlap",
    "zpdf_left_low",
    "zpdf_left_high",
    "zpdf_right_low",
    "zpdf_right_high",
    "sed_rms",
    "n_common_bands",
    "median_mag_offset",
    "weak_color_evidence",
]

FAMILY_COLUMNS = [
    "cluster",
    "source_slug",
    "source_id",
    "source_path",
    "literature_family_id",
    "n_literature_images",
    "n_matched_images",
    "match_fraction",
    "analysis_selected",
    "n_duplicate_master_matches",
    "has_duplicate_master_match",
    "n_pairs",
    "n_strong_specz_pairs",
    "n_strong_specz_conflicts",
    "strong_specz_consistent_fraction",
    "literature_z_median",
    "literature_z_max_delta",
    "master_zspec_median",
    "master_zspec_delta_median",
    "master_zspec_delta_max",
    "master_zphot_median",
    "master_zphot_delta_median",
    "master_zphot_delta_max",
    "sed_rms_median",
    "sed_rms_p90",
    "sed_rms_max",
    "n_common_bands_min",
    "n_common_bands_median",
    "max_match_separation_arcsec",
    "max_pair_separation_arcsec",
    "max_pair_separation_kpc",
    "review_flags",
]

MANIFEST_COLUMNS = [
    "cluster",
    "source_slug",
    "source_id",
    "status",
    "match_radius_arcsec",
    "min_matched_images",
    "n_literature_images",
    "n_literature_families",
    "n_matched_images",
    "n_duplicate_master_matches",
    "n_analyzed_families",
    "n_pair_metrics",
    "note",
]

PAGUL_MATCH_COLUMNS = [
    "pagul_matched",
    "pagul_match_separation_arcsec",
    "pagul_object_id",
    "pagul_ra",
    "pagul_dec",
    "pagul_zspec",
    "pagul_zspec_q",
    "pagul_zphot",
    "pagul_zpdf_low",
    "pagul_zpdf_high",
    "pagul_chi2_red",
    "pagul_mag_f160w",
    "pagul_nb_used",
]

PAGUL_CROSSMATCH_SUMMARY_COLUMNS = [
    "scope",
    "cluster",
    "source_slug",
    "source_id",
    "source_path",
    "n_literature_images",
    "n_pagul_matched_images",
    "pagul_match_fraction",
    "n_master_matched_images",
    "n_matched_both",
    "n_pagul_only",
    "n_master_only",
]

PAGUL_ZPHOT_CUT_SUMMARY_COLUMNS = [
    "cluster",
    "z_lens",
    "zphot_cut",
    "matched_with_zpdf",
    "above_cut",
    "below_or_equal_cut",
    "fraction_above_cut",
    "median_zpdf",
    "min_zpdf",
    "max_zpdf",
]

PAGUL_SPECZ_COVERAGE_SUMMARY_COLUMNS = [
    "scope",
    "cluster",
    "source_slug",
    "source_id",
    "source_path",
    "n_literature_images",
    "n_pagul_matched_images",
    "n_pagul_matched_with_zspec",
    "n_pagul_matched_without_zspec",
    "pagul_specz_fraction_among_matches",
]


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _numeric_array(values: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _valid_redshift_array(values: Any, *, max_z: float = 20.0) -> np.ndarray:
    array = _numeric_array(values)
    invalid = ~np.isfinite(array) | np.isin(array, list(INVALID_REDSHIFT_SENTINELS)) | (array <= 0.0) | (array >= max_z)
    result = array.copy()
    result[invalid] = np.nan
    return result


def _decode_string_series(values: Any) -> pd.Series:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="ignore").strip())
        else:
            decoded.append(str(value).strip())
    return pd.Series(decoded, dtype=object)


def _f160w_ab_magnitude(flux_values: Any) -> np.ndarray:
    flux = _numeric_array(flux_values)
    magnitude = np.full(flux.shape, np.nan, dtype=float)
    positive = np.isfinite(flux) & (flux > 0.0)
    magnitude[positive] = -2.5 * np.log10(flux[positive]) - 48.6
    return magnitude


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _finite_median(values: Iterable[Any]) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _finite_max(values: Iterable[Any]) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(np.max(array)) if array.size else float("nan")


def _finite_percentile(values: Iterable[Any], percentile: float) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, percentile)) if array.size else float("nan")


def kpc_per_arcsec(cluster: str) -> float:
    z_lens = CLUSTER_Z_LENS.get(cluster)
    if z_lens is None:
        return float("nan")
    return float(COSMO.kpc_proper_per_arcmin(z_lens).value / 60.0)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "source"


def _master_catalog_path(catalog_root: Path, cluster: str) -> Path:
    nested = Path(catalog_root) / cluster / f"{cluster}_master_catalog.csv"
    if nested.exists():
        return nested
    return Path(catalog_root) / f"{cluster}_master_catalog.csv"


def load_master_catalog(catalog_root: Path, cluster: str) -> pd.DataFrame:
    path = _master_catalog_path(catalog_root, cluster)
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path, low_memory=False)
    if "object_id" not in data.columns:
        data["object_id"] = data.index.map(lambda index: f"row:{index}")
    return data


def _empty_pagul_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cluster",
            "object_id",
            "field",
            "ra",
            "dec",
            "zspec",
            "zspec_q",
            "zphot",
            "zpdf_low",
            "zpdf_high",
            "chi2_red",
            "mag_f160w",
            "nb_used",
            "catalog_path",
        ]
    )


def _pagul_catalog_path(pagul_root: Path, cluster: str) -> Path | None:
    slugs = PAGUL_SLUGS_BY_CLUSTER.get(str(cluster), (str(cluster),))
    candidates = sorted(path for path in Path(pagul_root).glob("*catalog.fits") if path.is_file())
    for path in candidates:
        name = path.name.lower()
        if any(slug.lower() in name for slug in slugs):
            return path
    return None


def load_pagul_catalog(pagul_root: Path, cluster: str) -> pd.DataFrame:
    path = _pagul_catalog_path(Path(pagul_root), cluster)
    if path is None:
        return _empty_pagul_catalog()

    required_columns = [
        "ID",
        "FIELD",
        "ALPHA_J2000_STACK",
        "DELTA_J2000_STACK",
        "ZSPEC",
        "ZSPEC_Q",
        "ZPDF",
        "ZPDF_LOW",
        "ZPDF_HIGH",
        "CHI2_RED",
        "FLUX_F160W",
        "NB_USED",
    ]
    table = Table.read(path)
    missing_columns = [column for column in required_columns if column not in table.colnames]
    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {missing_columns}")

    return pd.DataFrame(
        {
            "cluster": str(cluster),
            "object_id": _decode_string_series(table["ID"]),
            "field": _decode_string_series(table["FIELD"]),
            "ra": _numeric_array(table["ALPHA_J2000_STACK"]),
            "dec": _numeric_array(table["DELTA_J2000_STACK"]),
            "zspec": _valid_redshift_array(table["ZSPEC"]),
            "zspec_q": _numeric_array(table["ZSPEC_Q"]),
            "zphot": _valid_redshift_array(table["ZPDF"]),
            "zpdf_low": _valid_redshift_array(table["ZPDF_LOW"]),
            "zpdf_high": _valid_redshift_array(table["ZPDF_HIGH"]),
            "chi2_red": _numeric_array(table["CHI2_RED"]),
            "mag_f160w": _f160w_ab_magnitude(table["FLUX_F160W"]),
            "nb_used": _numeric_array(table["NB_USED"]),
            "catalog_path": str(path),
        }
    )


def _copy_master_fields(master_row: pd.Series | None) -> dict[str, Any]:
    if master_row is None:
        row = {
            "master_row_index": np.nan,
            "master_object_id": "",
            "master_ra": np.nan,
            "master_dec": np.nan,
            "master_object_source": "",
            "master_catalog_sources": "",
            "master_zspec_best": np.nan,
            "master_zspec_best_source": "",
            "master_zspec_best_confidence": "",
            "master_zspec_best_confidence_rank": np.nan,
            "master_zspec_best_native_quality": np.nan,
            "master_zspec_conflict": False,
            "master_zphot_best": np.nan,
            "master_zpdf_low": np.nan,
            "master_zpdf_high": np.nan,
            "master_n_valid_bands": 0,
        }
        for band in MAG_BANDS:
            row[f"master_mag_{band}"] = np.nan
        return row

    row = {
        "master_row_index": int(master_row.name),
        "master_object_id": str(master_row.get("object_id", "")),
        "master_ra": _safe_float(master_row.get("ra")),
        "master_dec": _safe_float(master_row.get("dec")),
        "master_object_source": str(master_row.get("object_source", "")),
        "master_catalog_sources": str(master_row.get("catalog_sources", "")),
        "master_zspec_best": _safe_float(master_row.get("zspec_best")),
        "master_zspec_best_source": str(master_row.get("zspec_best_source", "")),
        "master_zspec_best_confidence": str(master_row.get("zspec_best_confidence", "")),
        "master_zspec_best_confidence_rank": _safe_float(master_row.get("zspec_best_confidence_rank")),
        "master_zspec_best_native_quality": _safe_float(master_row.get("zspec_best_native_quality")),
        "master_zspec_conflict": _bool_value(master_row.get("zspec_conflict", False)),
        "master_zphot_best": _safe_float(master_row.get("zphot_best")),
        "master_zpdf_low": _safe_float(master_row.get("pagul_zpdf_low")),
        "master_zpdf_high": _safe_float(master_row.get("pagul_zpdf_high")),
    }
    n_valid_bands = 0
    for band in MAG_BANDS:
        value = _safe_float(master_row.get(f"mag_{band}"))
        row[f"master_mag_{band}"] = value
        if np.isfinite(value):
            n_valid_bands += 1
    row["master_n_valid_bands"] = n_valid_bands
    return row


def _copy_pagul_fields(pagul_row: pd.Series | None, separation_arcsec: float | None = None) -> dict[str, Any]:
    if pagul_row is None:
        return {
            "pagul_matched": False,
            "pagul_match_separation_arcsec": np.nan,
            "pagul_object_id": "",
            "pagul_ra": np.nan,
            "pagul_dec": np.nan,
            "pagul_zspec": np.nan,
            "pagul_zspec_q": np.nan,
            "pagul_zphot": np.nan,
            "pagul_zpdf_low": np.nan,
            "pagul_zpdf_high": np.nan,
            "pagul_chi2_red": np.nan,
            "pagul_mag_f160w": np.nan,
            "pagul_nb_used": np.nan,
        }
    return {
        "pagul_matched": True,
        "pagul_match_separation_arcsec": np.nan if separation_arcsec is None else float(separation_arcsec),
        "pagul_object_id": str(pagul_row.get("object_id", "")),
        "pagul_ra": _safe_float(pagul_row.get("ra")),
        "pagul_dec": _safe_float(pagul_row.get("dec")),
        "pagul_zspec": _safe_float(pagul_row.get("zspec")),
        "pagul_zspec_q": _safe_float(pagul_row.get("zspec_q")),
        "pagul_zphot": _safe_float(pagul_row.get("zphot")),
        "pagul_zpdf_low": _safe_float(pagul_row.get("zpdf_low")),
        "pagul_zpdf_high": _safe_float(pagul_row.get("zpdf_high")),
        "pagul_chi2_red": _safe_float(pagul_row.get("chi2_red")),
        "pagul_mag_f160w": _safe_float(pagul_row.get("mag_f160w")),
        "pagul_nb_used": _safe_float(pagul_row.get("nb_used")),
    }


def match_literature_images_to_pagul(
    literature: pd.DataFrame,
    pagul: pd.DataFrame,
    *,
    radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    literature = literature.reset_index(drop=True)
    pagul = pagul.reset_index(drop=True)
    base_matches: dict[int, tuple[int, float]] = {}

    if not literature.empty and not pagul.empty:
        lit_ra = pd.to_numeric(literature.get("ra"), errors="coerce").to_numpy(dtype=float)
        lit_dec = pd.to_numeric(literature.get("dec"), errors="coerce").to_numpy(dtype=float)
        pagul_ra = pd.to_numeric(pagul.get("ra"), errors="coerce").to_numpy(dtype=float)
        pagul_dec = pd.to_numeric(pagul.get("dec"), errors="coerce").to_numpy(dtype=float)
        valid_lit = np.isfinite(lit_ra) & np.isfinite(lit_dec)
        valid_pagul = np.isfinite(pagul_ra) & np.isfinite(pagul_dec)
        if valid_lit.any() and valid_pagul.any():
            lit_indices = np.flatnonzero(valid_lit)
            pagul_indices = np.flatnonzero(valid_pagul)
            lit_coords = SkyCoord(ra=lit_ra[valid_lit] * u.deg, dec=lit_dec[valid_lit] * u.deg)
            pagul_coords = SkyCoord(ra=pagul_ra[valid_pagul] * u.deg, dec=pagul_dec[valid_pagul] * u.deg)
            nearest, separation, _ = lit_coords.match_to_catalog_sky(pagul_coords)
            for local_lit_index, local_pagul_index, sep in zip(lit_indices, nearest, separation.arcsec, strict=True):
                if float(sep) <= float(radius_arcsec):
                    base_matches[int(local_lit_index)] = (int(pagul_indices[int(local_pagul_index)]), float(sep))

    for index, _lit_row in literature.iterrows():
        match = base_matches.get(int(index))
        pagul_row = pagul.iloc[match[0]] if match is not None else None
        rows.append(
            {
                "literature_row_index": int(index),
                **_copy_pagul_fields(pagul_row, None if match is None else match[1]),
            }
        )

    return pd.DataFrame(rows, columns=["literature_row_index", *PAGUL_MATCH_COLUMNS])


def match_literature_images_to_master(
    literature: pd.DataFrame,
    master: pd.DataFrame,
    *,
    radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    literature = literature.reset_index(drop=True)
    master = master.reset_index(drop=True)
    base_matches: dict[int, tuple[int, float]] = {}

    if not literature.empty and not master.empty:
        lit_ra = pd.to_numeric(literature.get("ra"), errors="coerce").to_numpy(dtype=float)
        lit_dec = pd.to_numeric(literature.get("dec"), errors="coerce").to_numpy(dtype=float)
        master_ra = pd.to_numeric(master.get("ra"), errors="coerce").to_numpy(dtype=float)
        master_dec = pd.to_numeric(master.get("dec"), errors="coerce").to_numpy(dtype=float)
        valid_lit = np.isfinite(lit_ra) & np.isfinite(lit_dec)
        valid_master = np.isfinite(master_ra) & np.isfinite(master_dec)
        if valid_lit.any() and valid_master.any():
            lit_indices = np.flatnonzero(valid_lit)
            master_indices = np.flatnonzero(valid_master)
            lit_coords = SkyCoord(ra=lit_ra[valid_lit] * u.deg, dec=lit_dec[valid_lit] * u.deg)
            master_coords = SkyCoord(ra=master_ra[valid_master] * u.deg, dec=master_dec[valid_master] * u.deg)
            nearest, separation, _ = lit_coords.match_to_catalog_sky(master_coords)
            for local_lit_index, local_master_index, sep in zip(lit_indices, nearest, separation.arcsec, strict=True):
                if float(sep) <= float(radius_arcsec):
                    base_matches[int(local_lit_index)] = (int(master_indices[int(local_master_index)]), float(sep))

    for index, lit_row in literature.iterrows():
        match = base_matches.get(int(index))
        master_row = master.iloc[match[0]] if match is not None else None
        rows.append(
            {
                "literature_row_index": int(index),
                "literature_id": str(lit_row.get("literature_id", "")),
                "literature_family_id": str(lit_row.get("family_id", "")),
                "literature_image_id": str(lit_row.get("image_id", "")),
                "literature_ra": _safe_float(lit_row.get("ra")),
                "literature_dec": _safe_float(lit_row.get("dec")),
                "literature_z": _safe_float(lit_row.get("catalog_z")),
                "literature_mag": _safe_float(lit_row.get("catalog_mag")),
                "matched": match is not None,
                "match_separation_arcsec": np.nan if match is None else float(match[1]),
                "duplicate_master_match": False,
                **_copy_pagul_fields(None),
                **_copy_master_fields(master_row),
            }
        )

    matches = pd.DataFrame(rows, columns=[column for column in MATCH_COLUMNS if column not in {"cluster", "source_slug", "source_id", "source_path"}])
    matched_object_ids = matches.loc[matches["matched"].map(_bool_value), "master_object_id"].astype(str)
    duplicate_ids = set(matched_object_ids.loc[matched_object_ids.duplicated(keep=False)])
    matches["duplicate_master_match"] = matches["matched"].map(_bool_value) & matches["master_object_id"].astype(str).isin(duplicate_ids)
    return matches


def _color_difference(left: pd.Series, right: pd.Series) -> tuple[float, int, float]:
    left_values = np.asarray([_safe_float(left.get(f"master_mag_{band}")) for band in MAG_BANDS], dtype=float)
    right_values = np.asarray([_safe_float(right.get(f"master_mag_{band}")) for band in MAG_BANDS], dtype=float)
    common = np.isfinite(left_values) & np.isfinite(right_values)
    n_common = int(common.sum())
    if n_common == 0:
        return float("nan"), 0, float("nan")
    diff = left_values[common] - right_values[common]
    offset = float(np.median(diff))
    residuals = diff - offset
    return float(np.sqrt(np.mean(residuals**2))), n_common, offset


def _zpdf_overlap(left: pd.Series, right: pd.Series) -> bool | float:
    left_low = _safe_float(left.get("master_zpdf_low"))
    left_high = _safe_float(left.get("master_zpdf_high"))
    right_low = _safe_float(right.get("master_zpdf_low"))
    right_high = _safe_float(right.get("master_zpdf_high"))
    if not all(np.isfinite(value) for value in (left_low, left_high, right_low, right_high)):
        return np.nan
    return bool(max(left_low, right_low) <= min(left_high, right_high))


def _sky_separation(left: pd.Series, right: pd.Series, cluster: str) -> tuple[float, float]:
    left_ra = _safe_float(left.get("literature_ra"))
    left_dec = _safe_float(left.get("literature_dec"))
    right_ra = _safe_float(right.get("literature_ra"))
    right_dec = _safe_float(right.get("literature_dec"))
    if not all(np.isfinite(value) for value in (left_ra, left_dec, right_ra, right_dec)):
        return float("nan"), float("nan")
    left_coord = SkyCoord(ra=left_ra * u.deg, dec=left_dec * u.deg)
    right_coord = SkyCoord(ra=right_ra * u.deg, dec=right_dec * u.deg)
    sep_arcsec = float(left_coord.separation(right_coord).arcsec)
    scale = kpc_per_arcsec(cluster)
    return sep_arcsec, sep_arcsec * scale if np.isfinite(scale) else float("nan")


def compute_family_pair_metrics(
    matches: pd.DataFrame,
    *,
    min_matched_images: int = DEFAULT_MIN_MATCHED_IMAGES,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if matches.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    for (_cluster, _source_id, family_id), group in matches.groupby(["cluster", "source_id", "literature_family_id"], dropna=False):
        matched = group.loc[group["matched"].map(_bool_value)].reset_index(drop=True)
        if len(matched) < int(min_matched_images):
            continue
        for left_index, right_index in itertools.combinations(range(len(matched)), 2):
            left = matched.iloc[left_index]
            right = matched.iloc[right_index]
            sed_rms, n_common, mag_offset = _color_difference(left, right)
            literature_sep_arcsec, literature_sep_kpc = _sky_separation(left, right, str(_cluster))
            literature_z_left = _safe_float(left.get("literature_z"))
            literature_z_right = _safe_float(right.get("literature_z"))
            zspec_left = _safe_float(left.get("master_zspec_best"))
            zspec_right = _safe_float(right.get("master_zspec_best"))
            rank_left = _safe_float(left.get("master_zspec_best_confidence_rank"))
            rank_right = _safe_float(right.get("master_zspec_best_confidence_rank"))
            zphot_left = _safe_float(left.get("master_zphot_best"))
            zphot_right = _safe_float(right.get("master_zphot_best"))
            both_strong = (
                np.isfinite(zspec_left)
                and np.isfinite(zspec_right)
                and np.isfinite(rank_left)
                and np.isfinite(rank_right)
                and rank_left >= STRONG_SPEC_RANK
                and rank_right >= STRONG_SPEC_RANK
            )
            zspec_delta = abs(zspec_left - zspec_right) if np.isfinite(zspec_left) and np.isfinite(zspec_right) else float("nan")
            zphot_delta = abs(zphot_left - zphot_right) if np.isfinite(zphot_left) and np.isfinite(zphot_right) else float("nan")
            rows.append(
                {
                    "cluster": str(_cluster),
                    "source_slug": str(left.get("source_slug", "")),
                    "source_id": str(_source_id),
                    "literature_family_id": str(family_id),
                    "left_literature_id": str(left.get("literature_id", "")),
                    "right_literature_id": str(right.get("literature_id", "")),
                    "left_master_object_id": str(left.get("master_object_id", "")),
                    "right_master_object_id": str(right.get("master_object_id", "")),
                    "duplicate_master_pair": str(left.get("master_object_id", "")) == str(right.get("master_object_id", "")),
                    "left_match_separation_arcsec": _safe_float(left.get("match_separation_arcsec")),
                    "right_match_separation_arcsec": _safe_float(right.get("match_separation_arcsec")),
                    "max_match_separation_arcsec": max(
                        _safe_float(left.get("match_separation_arcsec")),
                        _safe_float(right.get("match_separation_arcsec")),
                    ),
                    "literature_pair_separation_arcsec": literature_sep_arcsec,
                    "literature_pair_separation_kpc": literature_sep_kpc,
                    "literature_z_left": literature_z_left,
                    "literature_z_right": literature_z_right,
                    "literature_z_delta": (
                        abs(literature_z_left - literature_z_right)
                        if np.isfinite(literature_z_left) and np.isfinite(literature_z_right)
                        else float("nan")
                    ),
                    "master_zspec_left": zspec_left,
                    "master_zspec_right": zspec_right,
                    "master_zspec_delta": zspec_delta,
                    "left_zspec_confidence_rank": rank_left,
                    "right_zspec_confidence_rank": rank_right,
                    "left_zspec_confidence": str(left.get("master_zspec_best_confidence", "")),
                    "right_zspec_confidence": str(right.get("master_zspec_best_confidence", "")),
                    "left_zspec_source": str(left.get("master_zspec_best_source", "")),
                    "right_zspec_source": str(right.get("master_zspec_best_source", "")),
                    "both_strong_specz": both_strong,
                    "strong_specz_consistent": bool(both_strong and np.isfinite(zspec_delta) and zspec_delta <= ZSPEC_CONFLICT_TOL),
                    "strong_specz_conflict": bool(both_strong and np.isfinite(zspec_delta) and zspec_delta > ZSPEC_CONFLICT_TOL),
                    "master_zphot_left": zphot_left,
                    "master_zphot_right": zphot_right,
                    "master_zphot_delta": zphot_delta,
                    "zpdf_overlap": _zpdf_overlap(left, right),
                    "zpdf_left_low": _safe_float(left.get("master_zpdf_low")),
                    "zpdf_left_high": _safe_float(left.get("master_zpdf_high")),
                    "zpdf_right_low": _safe_float(right.get("master_zpdf_low")),
                    "zpdf_right_high": _safe_float(right.get("master_zpdf_high")),
                    "sed_rms": sed_rms,
                    "n_common_bands": n_common,
                    "median_mag_offset": mag_offset,
                    "weak_color_evidence": n_common <= 1,
                }
            )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def _review_flags(family: pd.DataFrame, pairs: pd.DataFrame, min_matched_images: int) -> str:
    flags: list[str] = []
    if int(family["matched"].map(_bool_value).sum()) < int(min_matched_images):
        flags.append("insufficient_matched_images")
    if family["duplicate_master_match"].map(_bool_value).any():
        flags.append("duplicate_master_matches")
    matched_fraction = float(family["matched"].map(_bool_value).mean()) if len(family) else 0.0
    if matched_fraction < 0.67:
        flags.append("poor_match_fraction")
    if pairs.empty:
        flags.append("no_pair_metrics")
    else:
        if int(pairs["both_strong_specz"].map(_bool_value).sum()) == 0:
            flags.append("weak_redshift_coverage")
        if pairs["strong_specz_conflict"].map(_bool_value).any():
            flags.append("strong_specz_conflict")
        if _finite_max(pairs["sed_rms"]) > COLOR_RMS_REFERENCE:
            flags.append("large_color_scatter")
        if pairs["weak_color_evidence"].map(_bool_value).any():
            flags.append("weak_color_evidence")
    return "|".join(flags)


def compute_family_summary(
    matches: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    min_matched_images: int = DEFAULT_MIN_MATCHED_IMAGES,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if matches.empty:
        return pd.DataFrame(columns=FAMILY_COLUMNS)

    for (cluster, source_id, family_id), family in matches.groupby(["cluster", "source_id", "literature_family_id"], dropna=False):
        source_slug = str(family["source_slug"].iloc[0]) if "source_slug" in family.columns and len(family) else ""
        source_path = str(family["source_path"].iloc[0]) if "source_path" in family.columns and len(family) else ""
        family_pairs = pairs.loc[
            (pairs["cluster"].astype(str) == str(cluster))
            & (pairs["source_id"].astype(str) == str(source_id))
            & (pairs["literature_family_id"].astype(str) == str(family_id))
        ].copy()
        matched = family.loc[family["matched"].map(_bool_value)].copy()
        n_matched = int(len(matched))
        strong_pairs = family_pairs.loc[family_pairs["both_strong_specz"].map(_bool_value)] if not family_pairs.empty else pd.DataFrame()
        rows.append(
            {
                "cluster": str(cluster),
                "source_slug": source_slug,
                "source_id": str(source_id),
                "source_path": source_path,
                "literature_family_id": str(family_id),
                "n_literature_images": int(len(family)),
                "n_matched_images": n_matched,
                "match_fraction": float(n_matched / len(family)) if len(family) else np.nan,
                "analysis_selected": n_matched >= int(min_matched_images),
                "n_duplicate_master_matches": int(family["duplicate_master_match"].map(_bool_value).sum()),
                "has_duplicate_master_match": bool(family["duplicate_master_match"].map(_bool_value).any()),
                "n_pairs": int(len(family_pairs)),
                "n_strong_specz_pairs": int(len(strong_pairs)),
                "n_strong_specz_conflicts": int(family_pairs["strong_specz_conflict"].map(_bool_value).sum()) if not family_pairs.empty else 0,
                "strong_specz_consistent_fraction": (
                    float(strong_pairs["strong_specz_consistent"].map(_bool_value).sum() / len(strong_pairs))
                    if len(strong_pairs)
                    else np.nan
                ),
                "literature_z_median": _finite_median(family["literature_z"]),
                "literature_z_max_delta": _finite_max(family_pairs["literature_z_delta"]) if not family_pairs.empty else np.nan,
                "master_zspec_median": _finite_median(matched["master_zspec_best"]),
                "master_zspec_delta_median": _finite_median(family_pairs["master_zspec_delta"]) if not family_pairs.empty else np.nan,
                "master_zspec_delta_max": _finite_max(family_pairs["master_zspec_delta"]) if not family_pairs.empty else np.nan,
                "master_zphot_median": _finite_median(_column_or_nan(matched, "master_zphot_best")),
                "master_zphot_delta_median": _finite_median(family_pairs["master_zphot_delta"]) if not family_pairs.empty else np.nan,
                "master_zphot_delta_max": _finite_max(family_pairs["master_zphot_delta"]) if not family_pairs.empty else np.nan,
                "sed_rms_median": _finite_median(family_pairs["sed_rms"]) if not family_pairs.empty else np.nan,
                "sed_rms_p90": _finite_percentile(family_pairs["sed_rms"], 90.0) if not family_pairs.empty else np.nan,
                "sed_rms_max": _finite_max(family_pairs["sed_rms"]) if not family_pairs.empty else np.nan,
                "n_common_bands_min": np.nan if family_pairs.empty else _finite_min(family_pairs["n_common_bands"]),
                "n_common_bands_median": _finite_median(family_pairs["n_common_bands"]) if not family_pairs.empty else np.nan,
                "max_match_separation_arcsec": _finite_max(_column_or_nan(family, "match_separation_arcsec")),
                "max_pair_separation_arcsec": _finite_max(family_pairs["literature_pair_separation_arcsec"]) if not family_pairs.empty else np.nan,
                "max_pair_separation_kpc": _finite_max(family_pairs["literature_pair_separation_kpc"]) if not family_pairs.empty else np.nan,
                "review_flags": _review_flags(family, family_pairs, min_matched_images),
            }
        )
    return pd.DataFrame(rows, columns=FAMILY_COLUMNS)


def _finite_min(values: Iterable[Any]) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(np.min(array)) if array.size else float("nan")


def _column_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(np.nan, index=frame.index)


def _annotate_matches(source: LiteratureCatalog, matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches.insert(0, "source_path", str(source.path))
    matches.insert(0, "source_id", source.source_id)
    matches.insert(0, "source_slug", source.source_slug)
    matches.insert(0, "cluster", source.cluster)
    return matches.loc[:, MATCH_COLUMNS]


def analyze_literature_source(
    source: LiteratureCatalog,
    master: pd.DataFrame,
    pagul: pd.DataFrame,
    *,
    match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
    min_matched_images: int = DEFAULT_MIN_MATCHED_IMAGES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    master_matches = match_literature_images_to_master(source.data, master, radius_arcsec=match_radius_arcsec)
    pagul_matches = match_literature_images_to_pagul(source.data, pagul, radius_arcsec=match_radius_arcsec)
    for column in PAGUL_MATCH_COLUMNS:
        master_matches[column] = pagul_matches[column].to_numpy() if column in pagul_matches.columns else np.nan
    matches = _annotate_matches(
        source,
        master_matches,
    )
    pairs = compute_family_pair_metrics(matches, min_matched_images=min_matched_images)
    family_summary = compute_family_summary(matches, pairs, min_matched_images=min_matched_images)
    manifest = {
        "cluster": source.cluster,
        "source_slug": source.source_slug,
        "source_id": source.source_id,
        "status": "analyzed",
        "match_radius_arcsec": float(match_radius_arcsec),
        "min_matched_images": int(min_matched_images),
        "n_literature_images": int(len(source.data)),
        "n_literature_families": int(source.data["family_id"].nunique()) if "family_id" in source.data.columns else 0,
        "n_matched_images": int(matches["matched"].map(_bool_value).sum()),
        "n_duplicate_master_matches": int(matches["duplicate_master_match"].map(_bool_value).sum()),
        "n_analyzed_families": int(family_summary["analysis_selected"].map(_bool_value).sum()) if not family_summary.empty else 0,
        "n_pair_metrics": int(len(pairs)),
        "note": "",
    }
    return matches, pairs, family_summary, manifest


def _source_summary_rows(sources: pd.DataFrame, catalogs: list[LiteratureCatalog]) -> pd.DataFrame:
    image_catalogs = [catalog for catalog in catalogs if catalog.catalog_kind == "image"]
    image_keys = {(catalog.cluster, catalog.source_slug, str(catalog.path)) for catalog in image_catalogs}
    rows: list[dict[str, Any]] = []
    for source in image_catalogs:
        rows.append(
            {
                "cluster": source.cluster,
                "source_slug": source.source_slug,
                "source_id": source.source_id,
                "catalog_kind": source.catalog_kind,
                "path": str(source.path),
                "n_rows": int(len(source.data)),
                "n_families": int(source.data["family_id"].nunique()) if "family_id" in source.data.columns else 0,
                "status": source.status,
                "note": source.note,
            }
        )
    for row in sources.itertuples(index=False):
        if str(getattr(row, "status", "")) == "missing_source":
            rows.append(
                {
                    "cluster": str(row.cluster),
                    "source_slug": str(row.source_slug),
                    "source_id": f"{row.cluster}/{row.source_slug}",
                    "catalog_kind": "missing",
                    "path": "",
                    "n_rows": 0,
                    "n_families": 0,
                    "status": "missing_source",
                    "note": "",
                }
            )
        elif str(getattr(row, "catalog_kind", "")) == "image":
            path = str(getattr(row, "copied_path", ""))
            if (str(row.cluster), str(row.source_slug), path) in image_keys:
                continue
            rows.append(
                {
                    "cluster": str(row.cluster),
                    "source_slug": str(row.source_slug),
                    "source_id": str(getattr(row, "source_id", f"{row.cluster}/{row.source_slug}/{Path(path).name}")),
                    "catalog_kind": "image",
                    "path": path,
                    "n_rows": int(getattr(row, "n_rows", 0)),
                    "n_families": 0,
                    "status": str(getattr(row, "status", "")),
                    "note": str(getattr(row, "note", "")),
                }
            )
    return pd.DataFrame(rows)


def analyze_all(
    catalog_root: Path = DEFAULT_CATALOG_ROOT,
    literature_root: Path = DEFAULT_LITERATURE_ROOT,
    pagul_root: Path = DEFAULT_PAGUL_ROOT,
    *,
    clusters: Iterable[str] = DEFAULT_CLUSTERS,
    match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
    min_matched_images: int = DEFAULT_MIN_MATCHED_IMAGES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources_df, catalogs = discover_literature_catalogs(Path(literature_root))
    requested = tuple(clusters)
    image_catalogs = [catalog for catalog in catalogs if catalog.catalog_kind == "image" and catalog.cluster in requested]
    literature_sources = _source_summary_rows(sources_df, image_catalogs)
    literature_sources = literature_sources.loc[
        literature_sources["cluster"].astype(str).isin(requested)
    ].reset_index(drop=True)

    match_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    family_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []

    for cluster in requested:
        master = load_master_catalog(Path(catalog_root), cluster)
        pagul = load_pagul_catalog(Path(pagul_root), cluster)
        cluster_sources = [source for source in image_catalogs if source.cluster == cluster]
        if not cluster_sources:
            manifest_rows.append(
                {
                    "cluster": cluster,
                    "source_slug": "",
                    "source_id": "",
                    "status": "no_literature_image_catalog",
                    "match_radius_arcsec": float(match_radius_arcsec),
                    "min_matched_images": int(min_matched_images),
                    "n_literature_images": 0,
                    "n_literature_families": 0,
                    "n_matched_images": 0,
                    "n_duplicate_master_matches": 0,
                    "n_analyzed_families": 0,
                    "n_pair_metrics": 0,
                    "note": "",
                }
            )
            continue
        if master.empty:
            for source in cluster_sources:
                manifest_rows.append(
                    {
                        "cluster": cluster,
                        "source_slug": source.source_slug,
                        "source_id": source.source_id,
                        "status": "missing_master_catalog",
                        "match_radius_arcsec": float(match_radius_arcsec),
                        "min_matched_images": int(min_matched_images),
                        "n_literature_images": int(len(source.data)),
                        "n_literature_families": int(source.data["family_id"].nunique()) if "family_id" in source.data.columns else 0,
                        "n_matched_images": 0,
                        "n_duplicate_master_matches": 0,
                        "n_analyzed_families": 0,
                        "n_pair_metrics": 0,
                        "note": str(_master_catalog_path(Path(catalog_root), cluster)),
                    }
                )
            continue
        for source in cluster_sources:
            matches, pairs, families, manifest = analyze_literature_source(
                source,
                master,
                pagul,
                match_radius_arcsec=match_radius_arcsec,
                min_matched_images=min_matched_images,
            )
            match_frames.append(matches)
            pair_frames.append(pairs)
            family_frames.append(families)
            manifest_rows.append(manifest)

    image_matches = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame(columns=MATCH_COLUMNS)
    pair_metrics = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame(columns=PAIR_COLUMNS)
    family_summary = pd.concat(family_frames, ignore_index=True) if family_frames else pd.DataFrame(columns=FAMILY_COLUMNS)
    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    pagul_crossmatch_summary = compute_pagul_crossmatch_summary(image_matches)
    return literature_sources, image_matches, pair_metrics, family_summary, manifest, pagul_crossmatch_summary


def _pagul_summary_row(scope: str, label: tuple[str, str, str, str], group: pd.DataFrame) -> dict[str, Any]:
    cluster, source_slug, source_id, source_path = label
    pagul_matched = group["pagul_matched"].map(_bool_value) if "pagul_matched" in group.columns else pd.Series(False, index=group.index)
    master_matched = group["matched"].map(_bool_value) if "matched" in group.columns else pd.Series(False, index=group.index)
    n_literature = int(len(group))
    n_pagul = int(pagul_matched.sum())
    n_master = int(master_matched.sum())
    return {
        "scope": scope,
        "cluster": cluster,
        "source_slug": source_slug,
        "source_id": source_id,
        "source_path": source_path,
        "n_literature_images": n_literature,
        "n_pagul_matched_images": n_pagul,
        "pagul_match_fraction": float(n_pagul / n_literature) if n_literature else np.nan,
        "n_master_matched_images": n_master,
        "n_matched_both": int((pagul_matched & master_matched).sum()),
        "n_pagul_only": int((pagul_matched & ~master_matched).sum()),
        "n_master_only": int((master_matched & ~pagul_matched).sum()),
    }


def compute_pagul_crossmatch_summary(image_matches: pd.DataFrame) -> pd.DataFrame:
    if image_matches.empty:
        return pd.DataFrame(columns=PAGUL_CROSSMATCH_SUMMARY_COLUMNS)

    rows: list[dict[str, Any]] = []
    group_columns = ["cluster", "source_slug", "source_id", "source_path"]
    for label, group in image_matches.groupby(group_columns, dropna=False, sort=True):
        rows.append(_pagul_summary_row("source", tuple(str(item) for item in label), group))

    for cluster, group in image_matches.groupby("cluster", dropna=False, sort=True):
        rows.append(_pagul_summary_row("cluster", (str(cluster), "all_sources", f"{cluster}/all_sources", ""), group))

    rows.append(_pagul_summary_row("all", ("all", "all_sources", "all/all_sources", ""), image_matches))
    return pd.DataFrame(rows, columns=PAGUL_CROSSMATCH_SUMMARY_COLUMNS)


def _pagul_specz_coverage_row(scope: str, label: tuple[str, str, str, str], group: pd.DataFrame) -> dict[str, Any]:
    cluster, source_slug, source_id, source_path = label
    pagul_matched = group["pagul_matched"].map(_bool_value) if "pagul_matched" in group.columns else pd.Series(False, index=group.index)
    pagul_zspec = pd.to_numeric(group["pagul_zspec"], errors="coerce") if "pagul_zspec" in group.columns else pd.Series(np.nan, index=group.index)
    has_zspec = pagul_matched & np.isfinite(pagul_zspec) & (pagul_zspec > 0.0)
    n_pagul = int(pagul_matched.sum())
    n_with_zspec = int(has_zspec.sum())
    return {
        "scope": scope,
        "cluster": cluster,
        "source_slug": source_slug,
        "source_id": source_id,
        "source_path": source_path,
        "n_literature_images": int(len(group)),
        "n_pagul_matched_images": n_pagul,
        "n_pagul_matched_with_zspec": n_with_zspec,
        "n_pagul_matched_without_zspec": int(n_pagul - n_with_zspec),
        "pagul_specz_fraction_among_matches": float(n_with_zspec / n_pagul) if n_pagul else np.nan,
    }


def compute_pagul_specz_coverage_summary(image_matches: pd.DataFrame) -> pd.DataFrame:
    if image_matches.empty:
        return pd.DataFrame(columns=PAGUL_SPECZ_COVERAGE_SUMMARY_COLUMNS)

    rows: list[dict[str, Any]] = []
    group_columns = ["cluster", "source_slug", "source_id", "source_path"]
    for label, group in image_matches.groupby(group_columns, dropna=False, sort=True):
        rows.append(_pagul_specz_coverage_row("source", tuple(str(item) for item in label), group))

    for cluster, group in image_matches.groupby("cluster", dropna=False, sort=True):
        rows.append(_pagul_specz_coverage_row("cluster", (str(cluster), "all_sources", f"{cluster}/all_sources", ""), group))

    rows.append(_pagul_specz_coverage_row("all", ("all", "all_sources", "all/all_sources", ""), image_matches))
    return pd.DataFrame(rows, columns=PAGUL_SPECZ_COVERAGE_SUMMARY_COLUMNS)


def _pagul_zphot_cut_row(cluster: str, group: pd.DataFrame) -> dict[str, Any]:
    zphot = pd.to_numeric(group["pagul_zphot"], errors="coerce").to_numpy(dtype=float)
    z_lens = CLUSTER_Z_LENS.get(str(cluster))
    zphot_cut = float(z_lens + BACKGROUND_Z_MARGIN) if z_lens is not None else np.nan
    if z_lens is None:
        above_cut = np.nan
        below_or_equal_cut = np.nan
        fraction_above_cut = np.nan
    else:
        above_cut = int(np.sum(zphot > zphot_cut))
        below_or_equal_cut = int(zphot.size - above_cut)
        fraction_above_cut = float(above_cut / zphot.size) if zphot.size else np.nan

    return {
        "cluster": str(cluster),
        "z_lens": np.nan if z_lens is None else float(z_lens),
        "zphot_cut": zphot_cut,
        "matched_with_zpdf": int(zphot.size),
        "above_cut": above_cut,
        "below_or_equal_cut": below_or_equal_cut,
        "fraction_above_cut": fraction_above_cut,
        "median_zpdf": float(np.nanmedian(zphot)) if zphot.size else np.nan,
        "min_zpdf": float(np.nanmin(zphot)) if zphot.size else np.nan,
        "max_zpdf": float(np.nanmax(zphot)) if zphot.size else np.nan,
    }


def compute_pagul_zphot_cut_summary(image_matches: pd.DataFrame) -> pd.DataFrame:
    if image_matches.empty or "pagul_matched" not in image_matches.columns or "pagul_zphot" not in image_matches.columns:
        return pd.DataFrame(columns=PAGUL_ZPHOT_CUT_SUMMARY_COLUMNS)

    zphot = pd.to_numeric(image_matches["pagul_zphot"], errors="coerce")
    selected = image_matches.loc[image_matches["pagul_matched"].map(_bool_value) & np.isfinite(zphot) & (zphot > 0.0)].copy()
    if selected.empty:
        return pd.DataFrame(columns=PAGUL_ZPHOT_CUT_SUMMARY_COLUMNS)
    selected["pagul_zphot"] = pd.to_numeric(selected["pagul_zphot"], errors="coerce")

    rows = [_pagul_zphot_cut_row(str(cluster), group) for cluster, group in selected.groupby("cluster", dropna=False, sort=True)]

    all_zphot = selected["pagul_zphot"].to_numpy(dtype=float)
    row_cuts = selected["cluster"].astype(str).map(lambda cluster: CLUSTER_Z_LENS.get(cluster, np.nan) + BACKGROUND_Z_MARGIN)
    known_cut = np.isfinite(pd.to_numeric(row_cuts, errors="coerce"))
    if known_cut.any():
        above_cut = int((selected.loc[known_cut, "pagul_zphot"].to_numpy(dtype=float) > row_cuts.loc[known_cut].to_numpy(dtype=float)).sum())
        below_or_equal_cut = int(known_cut.sum() - above_cut)
        fraction_above_cut = float(above_cut / known_cut.sum())
    else:
        above_cut = np.nan
        below_or_equal_cut = np.nan
        fraction_above_cut = np.nan
    rows.append(
        {
            "cluster": "all",
            "z_lens": np.nan,
            "zphot_cut": np.nan,
            "matched_with_zpdf": int(all_zphot.size),
            "above_cut": above_cut,
            "below_or_equal_cut": below_or_equal_cut,
            "fraction_above_cut": fraction_above_cut,
            "median_zpdf": float(np.nanmedian(all_zphot)),
            "min_zpdf": float(np.nanmin(all_zphot)),
            "max_zpdf": float(np.nanmax(all_zphot)),
        }
    )
    return pd.DataFrame(rows, columns=PAGUL_ZPHOT_CUT_SUMMARY_COLUMNS)


def _save_figure(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def _hist_values(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def write_metric_plots(
    out_dir: Path,
    pair_metrics: pd.DataFrame,
    family_summary: pd.DataFrame,
    image_matches: pd.DataFrame,
    pagul_crossmatch_summary: pd.DataFrame,
    pagul_specz_coverage_summary: pd.DataFrame,
) -> None:
    plot_dir = Path(out_dir) / "plots"
    if not pair_metrics.empty:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        specs = [
            (axes[0, 0], "master_zspec_delta", "abs(delta zspec)", ZSPEC_CONFLICT_TOL),
            (axes[0, 1], "master_zphot_delta", "abs(delta zphot)", None),
            (axes[1, 0], "sed_rms", "color RMS [mag]", COLOR_RMS_REFERENCE),
            (axes[1, 1], "n_common_bands", "common bands", None),
        ]
        for ax, column, label, reference in specs:
            values = _hist_values(pair_metrics[column]) if column in pair_metrics.columns else np.asarray([])
            if values.size:
                ax.hist(values, bins=30, color="#4477aa", alpha=0.8)
            if reference is not None:
                ax.axvline(reference, color="#cc3311", linestyle="--", linewidth=1.5)
            ax.set_xlabel(label)
            ax.set_ylabel("Confirmed-family pairs")
        _save_figure(fig, plot_dir / "confirmed_family_pair_metric_histograms")

        x = _hist_values(pair_metrics["master_zspec_delta"])
        y = _hist_values(pair_metrics["sed_rms"])
        scatter = pair_metrics.loc[
            np.isfinite(pd.to_numeric(pair_metrics["master_zspec_delta"], errors="coerce"))
            & np.isfinite(pd.to_numeric(pair_metrics["sed_rms"], errors="coerce"))
        ]
        if not scatter.empty:
            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
            ax.scatter(scatter["master_zspec_delta"], scatter["sed_rms"], s=16, alpha=0.65, color="#228833")
            ax.axvline(ZSPEC_CONFLICT_TOL, color="#cc3311", linestyle="--", linewidth=1.2)
            ax.axhline(COLOR_RMS_REFERENCE, color="#cc3311", linestyle="--", linewidth=1.2)
            ax.set_xlabel("abs(delta zspec)")
            ax.set_ylabel("color RMS [mag]")
            _save_figure(fig, plot_dir / "color_rms_vs_specz_delta")

        scatter = pair_metrics.loc[
            np.isfinite(pd.to_numeric(pair_metrics["n_common_bands"], errors="coerce"))
            & np.isfinite(pd.to_numeric(pair_metrics["sed_rms"], errors="coerce"))
        ]
        if not scatter.empty:
            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
            ax.scatter(scatter["n_common_bands"], scatter["sed_rms"], s=16, alpha=0.65, color="#aa3377")
            ax.axhline(COLOR_RMS_REFERENCE, color="#cc3311", linestyle="--", linewidth=1.2)
            ax.set_xlabel("common bands")
            ax.set_ylabel("color RMS [mag]")
            _save_figure(fig, plot_dir / "color_rms_vs_common_bands")

        _write_source_boxplot(plot_dir / "color_rms_by_literature_source", pair_metrics, "sed_rms", "color RMS [mag]")
        _write_source_boxplot(plot_dir / "specz_delta_by_literature_source", pair_metrics, "master_zspec_delta", "abs(delta zspec)")
        _write_source_boxplot(plot_dir / "zphot_delta_by_literature_source", pair_metrics, "master_zphot_delta", "abs(delta zphot)")

    _write_sky_maps(plot_dir / "sky", image_matches)
    _write_family_summary_plot(plot_dir / "family_summary_counts", family_summary)
    _write_pagul_crossmatch_summary_plot(plot_dir / "pagul_crossmatch_summary", pagul_crossmatch_summary)
    _write_pagul_specz_coverage_plot(plot_dir / "pagul_specz_coverage", pagul_specz_coverage_summary)
    _write_pagul_zspec_zphot_plot(plot_dir / "pagul_matched_literature_zspec_vs_zphot", image_matches)
    _write_pagul_zphot_background_cut_plot(plot_dir / "pagul_zphot_background_cut", image_matches)


def _write_source_boxplot(path_base: Path, data: pd.DataFrame, column: str, ylabel: str) -> None:
    if data.empty or column not in data.columns:
        return
    working = data.copy()
    working["source_label"] = working["cluster"].astype(str) + "/" + working["source_slug"].astype(str)
    groups: list[np.ndarray] = []
    labels: list[str] = []
    for label, group in working.groupby("source_label", sort=True):
        values = _hist_values(group[column])
        if values.size:
            groups.append(values)
            labels.append(label)
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(labels)), 5), constrained_layout=True)
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    if column == "sed_rms":
        ax.axhline(COLOR_RMS_REFERENCE, color="#cc3311", linestyle="--", linewidth=1.2)
    elif column == "master_zspec_delta":
        ax.axhline(ZSPEC_CONFLICT_TOL, color="#cc3311", linestyle="--", linewidth=1.2)
    _save_figure(fig, path_base)


def _write_sky_maps(out_dir: Path, matches: pd.DataFrame) -> None:
    if matches.empty:
        return
    for (cluster, source_id), group in matches.groupby(["cluster", "source_id"], dropna=False):
        if group.empty:
            continue
        source_slug = str(group["source_slug"].iloc[0]) if "source_slug" in group else ""
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        unmatched = group.loc[~group["matched"].map(_bool_value)]
        matched = group.loc[group["matched"].map(_bool_value)]
        if not unmatched.empty:
            ax.scatter(unmatched["literature_ra"], unmatched["literature_dec"], s=14, color="#999999", alpha=0.45, label="unmatched literature")
        if not matched.empty:
            ax.scatter(matched["literature_ra"], matched["literature_dec"], s=18, color="#4477aa", alpha=0.75, label="matched literature")
            ax.scatter(matched["master_ra"], matched["master_dec"], s=16, facecolors="none", edgecolors="#cc3311", linewidths=0.8, label="master match")
        ax.invert_xaxis()
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.set_title(f"{cluster} {source_slug}")
        ax.legend(loc="best", fontsize="small")
        _save_figure(fig, out_dir / f"{cluster}_{_safe_filename(str(source_id))}_sky")


def _write_family_summary_plot(path_base: Path, family_summary: pd.DataFrame) -> None:
    if family_summary.empty:
        return
    working = family_summary.copy()
    working["source_label"] = working["cluster"].astype(str) + "/" + working["source_slug"].astype(str)
    grouped = working.groupby("source_label", sort=True).agg(
        n_families=("literature_family_id", "count"),
        n_analyzed=("analysis_selected", lambda values: int(pd.Series(values).map(_bool_value).sum())),
    )
    if grouped.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(grouped)), 5), constrained_layout=True)
    x = np.arange(len(grouped))
    ax.bar(x - 0.18, grouped["n_families"], width=0.36, label="literature families")
    ax.bar(x + 0.18, grouped["n_analyzed"], width=0.36, label=">=2 matched images")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=90, fontsize=7)
    ax.set_ylabel("Families")
    ax.legend(loc="best")
    _save_figure(fig, path_base)


def _write_pagul_crossmatch_summary_plot(path_base: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    working = summary.loc[summary["scope"].astype(str).eq("source")].copy()
    if working.empty:
        return
    working["source_label"] = working["cluster"].astype(str) + "/" + working["source_slug"].astype(str)
    x = np.arange(len(working))
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(working)), 5), constrained_layout=True)
    ax.bar(x - 0.27, working["n_literature_images"], width=0.18, label="literature images", color="#999999")
    ax.bar(x - 0.09, working["n_master_matched_images"], width=0.18, label="master matched", color="#4477aa")
    ax.bar(x + 0.09, working["n_pagul_matched_images"], width=0.18, label="Pagul matched", color="#228833")
    ax.bar(x + 0.27, working["n_matched_both"], width=0.18, label="matched both", color="#aa3377")
    ax.set_xticks(x)
    ax.set_xticklabels(working["source_label"], rotation=90, fontsize=7)
    ax.set_ylabel("Images")
    ax.legend(loc="upper left", fontsize="small")

    fraction_ax = ax.twinx()
    fraction = pd.to_numeric(working["pagul_match_fraction"], errors="coerce").to_numpy(dtype=float)
    fraction_ax.plot(x, fraction, color="#cc3311", marker="o", linewidth=1.2, markersize=3, label="Pagul match fraction")
    fraction_ax.set_ylim(0.0, 1.05)
    fraction_ax.set_ylabel("Pagul match fraction")
    fraction_ax.legend(loc="upper right", fontsize="small")
    _save_figure(fig, path_base)


def _write_pagul_specz_coverage_plot(path_base: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    working = summary.loc[summary["scope"].astype(str).eq("source")].copy()
    if working.empty:
        return
    working["source_label"] = working["cluster"].astype(str) + "/" + working["source_slug"].astype(str)
    x = np.arange(len(working))
    with_zspec = pd.to_numeric(working["n_pagul_matched_with_zspec"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    without_zspec = pd.to_numeric(working["n_pagul_matched_without_zspec"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(working)), 5), constrained_layout=True)
    ax.bar(x, with_zspec, width=0.55, label="Pagul matched with ZSPEC", color="#228833")
    ax.bar(x, without_zspec, width=0.55, bottom=with_zspec, label="Pagul matched without ZSPEC", color="#bbbbbb")
    ax.set_xticks(x)
    ax.set_xticklabels(working["source_label"], rotation=90, fontsize=7)
    ax.set_ylabel("Pagul-matched images")
    ax.legend(loc="upper left", fontsize="small")

    fraction_ax = ax.twinx()
    fraction = pd.to_numeric(working["pagul_specz_fraction_among_matches"], errors="coerce").to_numpy(dtype=float)
    fraction_ax.plot(x, fraction, color="#cc3311", marker="o", linewidth=1.2, markersize=3, label="ZSPEC fraction")
    fraction_ax.set_ylim(0.0, 1.05)
    fraction_ax.set_ylabel("ZSPEC fraction among Pagul matches")
    fraction_ax.legend(loc="upper right", fontsize="small")
    _save_figure(fig, path_base)


def _write_pagul_zphot_background_cut_plot(path_base: Path, matches: pd.DataFrame) -> None:
    if matches.empty or "pagul_matched" not in matches.columns or "pagul_zphot" not in matches.columns:
        return
    zphot = pd.to_numeric(matches["pagul_zphot"], errors="coerce")
    selected = matches.loc[matches["pagul_matched"].map(_bool_value) & np.isfinite(zphot) & (zphot > 0.0)].copy()
    if selected.empty:
        return
    selected["pagul_zphot"] = pd.to_numeric(selected["pagul_zphot"], errors="coerce")
    summary = compute_pagul_zphot_cut_summary(selected)

    clusters = sorted(selected["cluster"].astype(str).unique())
    values_all = selected["pagul_zphot"].to_numpy(dtype=float)
    x_max = max(1.0, float(np.nanmax(values_all)) * 1.05)
    bins = np.linspace(0.0, x_max, 32)
    fig, axes = plt.subplots(
        len(clusters),
        1,
        figsize=(7.5, max(3.2, 1.75 * len(clusters))),
        sharex=True,
        constrained_layout=True,
    )
    if len(clusters) == 1:
        axes = np.asarray([axes])

    for ax, cluster in zip(axes, clusters, strict=True):
        group = selected.loc[selected["cluster"].astype(str).eq(cluster)]
        values = group["pagul_zphot"].to_numpy(dtype=float)
        ax.hist(values, bins=bins, color="#4477aa", alpha=0.78)
        z_lens = CLUSTER_Z_LENS.get(cluster)
        if z_lens is not None:
            zphot_cut = z_lens + BACKGROUND_Z_MARGIN
            ax.axvline(z_lens, color="#666666", linestyle=":", linewidth=1.4, label=r"$z_{lens}$")
            ax.axvline(zphot_cut, color="#cc3311", linestyle="--", linewidth=1.5, label=r"$z_{lens}+0.1$")
        row = summary.loc[summary["cluster"].astype(str).eq(cluster)]
        if row.empty or not np.isfinite(pd.to_numeric(row["fraction_above_cut"], errors="coerce").iloc[0]):
            annotation = f"{cluster}\nN={len(values):,}"
        else:
            row = row.iloc[0]
            annotation = (
                f"{cluster}\n"
                f"N={int(row['matched_with_zpdf']):,}\n"
                f">{row['zphot_cut']:.3f}: {int(row['above_cut']):,} "
                f"({100.0 * float(row['fraction_above_cut']):.1f}%)"
            )
        ax.text(0.98, 0.92, annotation, transform=ax.transAxes, ha="right", va="top", fontsize=10)
        ax.set_ylabel("Images")
        ax.tick_params(direction="in", top=True, right=True)

    axes[-1].set_xlabel(r"$z_{phot}$ (Pagul ZPDF)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper left", fontsize="small")
    _save_figure(fig, path_base)


def _pagul_photoz_metrics(frame: pd.DataFrame) -> dict[str, float]:
    residual = pd.to_numeric(frame.get("dz_norm"), errors="coerce").to_numpy(dtype=float)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return {"bias": np.nan, "nmad": np.nan, "eta": np.nan}
    bias = float(np.nanmedian(residual))
    nmad = float(1.4826 * np.nanmedian(np.abs(residual - bias)))
    eta = float(np.mean(np.abs(residual) > 0.15))
    return {"bias": bias, "nmad": nmad, "eta": eta}


def _write_pagul_zspec_zphot_plot(path_base: Path, matches: pd.DataFrame) -> None:
    if matches.empty or "pagul_zspec" not in matches.columns or "pagul_zphot" not in matches.columns:
        return
    zspec = pd.to_numeric(matches["pagul_zspec"], errors="coerce")
    zphot = pd.to_numeric(matches["pagul_zphot"], errors="coerce")
    selected = matches.loc[matches["pagul_matched"].map(_bool_value) & np.isfinite(zspec) & np.isfinite(zphot) & (zspec > 0.0) & (zphot > 0.0)].copy()
    if selected.empty:
        return

    selected["pagul_zspec"] = pd.to_numeric(selected["pagul_zspec"], errors="coerce")
    selected["pagul_zphot"] = pd.to_numeric(selected["pagul_zphot"], errors="coerce")
    selected["dz"] = selected["pagul_zphot"] - selected["pagul_zspec"]
    selected["dz_norm"] = selected["dz"] / (1.0 + selected["pagul_zspec"])
    metrics = _pagul_photoz_metrics(selected)

    plot_values = np.concatenate([selected["pagul_zspec"].to_numpy(dtype=float), selected["pagul_zphot"].to_numpy(dtype=float)])
    plot_values = plot_values[np.isfinite(plot_values) & (plot_values > 0.0)]
    plot_x_min = 0.1
    plot_x_max = max(8.0, float(np.nanmax(plot_values)) * 1.05) if plot_values.size else 8.0
    plot_y_min = 0.1
    plot_y_max = plot_x_max
    line = np.geomspace(plot_x_min, plot_x_max, 512)
    upper = line + 0.15 * (1.0 + line)
    lower = line - 0.15 * (1.0 + line)
    lower_positive = lower > 0.0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.0},
    )
    point_color = "#1f77b4"
    axes[0].scatter(selected["pagul_zspec"], selected["pagul_zphot"], s=18, color=point_color, alpha=0.45)
    axes[0].plot(line, line, color="black", linewidth=1.5)
    axes[0].plot(line, upper, color="black", linestyle="--", linewidth=1.2)
    axes[0].plot(line[lower_positive], lower[lower_positive], color="black", linestyle="--", linewidth=1.2)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(plot_x_min, plot_x_max)
    axes[0].set_ylim(plot_y_min, plot_y_max)
    axes[0].set_ylabel(r"$z_{phot}$ (Pagul ZPDF)")
    axes[0].text(
        0.025,
        0.96,
        f"N={len(selected):,}\n"
        f"NMAD={metrics['nmad']:.3f}\n"
        f"$\\eta$={100.0 * metrics['eta']:.1f}%\n"
        f"bias={metrics['bias']:.3f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=14,
    )

    axes[1].scatter(selected["pagul_zspec"], selected["dz_norm"], s=18, color=point_color, alpha=0.45)
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.2)
    axes[1].axhline(0.15, color="black", linestyle="--", linewidth=1.2)
    axes[1].axhline(-0.15, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_xscale("log")
    axes[1].set_xlim(plot_x_min, plot_x_max)
    axes[1].set_ylim(-0.8, 0.8)
    axes[1].set_xlabel(r"$z_{spec}$ (Pagul ZSPEC)")
    axes[1].set_ylabel(r"$\Delta z / (1 + z_{spec})$")
    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True, which="both")
    _save_figure(fig, path_base)


def write_outputs(
    output_dir: Path,
    literature_sources: pd.DataFrame,
    image_matches: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    family_summary: pd.DataFrame,
    manifest: pd.DataFrame,
    pagul_crossmatch_summary: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pagul_zphot_cut_summary = compute_pagul_zphot_cut_summary(image_matches)
    pagul_specz_coverage_summary = compute_pagul_specz_coverage_summary(image_matches)
    literature_sources.to_csv(output_dir / "literature_sources.csv", index=False)
    image_matches.to_csv(output_dir / "literature_image_master_matches.csv", index=False)
    pair_metrics.to_csv(output_dir / "literature_family_pair_metrics.csv", index=False)
    family_summary.to_csv(output_dir / "literature_family_summary.csv", index=False)
    manifest.to_csv(output_dir / "literature_family_diagnostics_manifest.csv", index=False)
    pagul_crossmatch_summary.to_csv(output_dir / "pagul_crossmatch_summary.csv", index=False)
    pagul_zphot_cut_summary.to_csv(output_dir / "pagul_zphot_background_cut_summary.csv", index=False)
    pagul_specz_coverage_summary.to_csv(output_dir / "pagul_specz_coverage_summary.csv", index=False)
    write_metric_plots(
        output_dir,
        pair_metrics,
        family_summary,
        image_matches,
        pagul_crossmatch_summary,
        pagul_specz_coverage_summary,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze confirmed literature image-family redshift and color diagnostics.")
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--literature-root", type=Path, default=DEFAULT_LITERATURE_ROOT)
    parser.add_argument("--pagul-root", type=Path, default=DEFAULT_PAGUL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clusters", default=",".join(DEFAULT_CLUSTERS))
    parser.add_argument("--match-radius-arcsec", type=float, default=DEFAULT_MATCH_RADIUS_ARCSEC)
    parser.add_argument("--min-matched-images", type=int, default=DEFAULT_MIN_MATCHED_IMAGES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    clusters = tuple(item.strip() for item in str(args.clusters).split(",") if item.strip())
    literature_sources, image_matches, pair_metrics, family_summary, manifest, pagul_crossmatch_summary = analyze_all(
        args.catalog_root,
        args.literature_root,
        args.pagul_root,
        clusters=clusters,
        match_radius_arcsec=args.match_radius_arcsec,
        min_matched_images=args.min_matched_images,
    )
    write_outputs(
        args.output_dir,
        literature_sources,
        image_matches,
        pair_metrics,
        family_summary,
        manifest,
        pagul_crossmatch_summary,
    )
    print(f"Wrote literature family diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
