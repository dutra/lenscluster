from __future__ import annotations

import argparse
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


CATALOG_VERSION = "HFF_Pagul21"
DEFAULT_DATA_DIR = Path("data") / "HFF_Pagul21"
DEFAULT_OUTPUT_SUBDIR = "prepared"
DEFAULT_POS_SIGMA_ARCSEC = 0.5
DEFAULT_IMAGE_A = 0.3734
DEFAULT_IMAGE_B = 0.3734
DEFAULT_IMAGE_THETA = 90.0
INVALID_SENTINELS = {-999.0, -99.0, 99.0}
MAGNIFICATION_INVALID_SENTINEL = -999.0
SIMBAD_TAP_URL = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
DEFAULT_SIMBAD_MATCH_ARCSEC = 1.0
DEFAULT_SIMBAD_QUERY_MARGIN_ARCMIN = 2.0
DEFAULT_SIMBAD_TIMEOUT_SEC = 120.0
DEFAULT_SIMBAD_MIN_LENS_DELTA = 0.1
DEFAULT_SIMBAD_MAX_PHOTOZ_DELTA = 0.2
DEFAULT_FAMILY_SPECZ_TOL = 0.005
DEFAULT_FAMILY_COLOR_RMS_MAX = 0.25
DEFAULT_FAMILY_MIN_COMMON_COLORS = 3
DEFAULT_FAMILY_MIN_SIZE = 2
DEFAULT_FAMILY_MAX_SEPARATION_ARCSEC = 120.0
DEFAULT_FAMILY_RELIABILITY_FLOOR = 0.05
DEFAULT_FAMILY_RELIABILITY_CEIL = 0.98
DEFAULT_CANDIDATE_FAMILY_START = 900001
DEFAULT_MEMBER_RADIUS_ARCSEC = 120.0
DEFAULT_MEMBER_Z_TOL = 0.12
DEFAULT_MEMBER_MAX_COUNT = 300
DEFAULT_BCG_RADIUS_ARCSEC = 30.0
COLOR_MAG_COLUMNS = tuple(f"MAG_OBS{index}" for index in range(0, 9))
SIMBAD_SYSTEM_PATTERN = re.compile(r"\b(\d+)(?:\.\d+){1,4}\b")


@dataclass(frozen=True)
class HFFClusterSpec:
    key: str
    field: str
    cluster_name: str
    z_lens: float
    magnification_filename: str


HFF_CLUSTER_SPECS: tuple[HFFClusterSpec, ...] = (
    HFFClusterSpec("a2744", "A2744clu", "abell2744", 0.3080, "magnifications_a2744.cat"),
    HFFClusterSpec("a370", "A370clu", "abell370", 0.3750, "magnifications_a370.cat"),
    HFFClusterSpec("as1063", "AS1063clu", "abells1063", 0.3480, "magnifications_as1063.cat"),
    HFFClusterSpec("m0416", "M0416clu", "macs0416", 0.3960, "magnifications_m0416.cat"),
    HFFClusterSpec("m0717", "M0717clu", "macs0717", 0.5450, "magnifications_m0717.cat"),
    HFFClusterSpec("m1149", "M1149clu", "macs1149", 0.5430, "magnifications_m1149.cat"),
)
HFF_CLUSTER_BY_KEY = {spec.key: spec for spec in HFF_CLUSTER_SPECS}


def _first_header_columns(path: Path, prefix: str) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(prefix):
                return line[1:].strip().split()
    raise ValueError(f"Could not find header starting with {prefix!r} in {path}.")


def _finite_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(result):
        return float("nan")
    if result in INVALID_SENTINELS:
        return float("nan")
    return float(result)


def _safe_mag(row: pd.Series) -> float:
    preferred = [
        "MAG_OBS8",  # F160W
        "MAG_OBS7",  # F140W
        "MAG_OBS6",  # F125W
        "MAG_OBS5",  # F105W
        "MAG_OBS4",  # F814W
        "MAG_OBS3",  # F606W
    ]
    for column in preferred:
        if column in row:
            value = _finite_or_nan(row[column])
            if np.isfinite(value) and 0.0 < value < 60.0:
                return value
    return 25.0


def _radec_offset_arcsec(ra_deg: Any, dec_deg: Any, ra0_deg: float, dec0_deg: float) -> tuple[float, float]:
    x = (float(ra_deg) - float(ra0_deg)) * np.cos(np.deg2rad(float(dec0_deg))) * 3600.0
    y = (float(dec_deg) - float(dec0_deg)) * 3600.0
    return float(x), float(y)


def _valid_mag(value: Any) -> float:
    mag = _finite_or_nan(value)
    if np.isfinite(mag) and 0.0 < mag < 60.0:
        return mag
    return float("nan")


def read_lph_catalog(path: str | Path) -> pd.DataFrame:
    """Read a Pagul21 LePhare catalog."""

    catalog_path = Path(path)
    columns = _first_header_columns(catalog_path, "# IDENT")
    df = pd.read_csv(
        catalog_path,
        sep=r"\s+",
        comment="#",
        names=columns,
        na_values=["-999", "-999.0", "-999.000", "-999.00000", "-99", "-99.0", "N/A", "-nan", "nan"],
        keep_default_na=True,
    )
    for column in ("IDENT", "FIELD", "RA", "DEC", "Z_BEST", "Z_ML", "ZSPEC"):
        if column not in df.columns:
            raise ValueError(f"Required column {column!r} is missing from {catalog_path}.")
    return df


def _is_magnification_data_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    first = stripped.split(maxsplit=1)[0]
    try:
        int(first)
    except ValueError:
        return False
    return True


def read_magnification_catalog(path: str | Path) -> pd.DataFrame:
    """Read a Pagul21 HFF magnification catalog.

    The magnification files use inconsistent two-line headers. The data rows are
    stable: object id, RA, Dec, source redshift, cluster name, lens redshift,
    followed by per-model magnification values.
    """

    catalog_path = Path(path)
    rows: list[dict[str, Any]] = []
    max_mu_columns = 0
    raw_mu_values: list[list[float]] = []
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not _is_magnification_data_line(line):
                continue
            tokens = line.split()
            if len(tokens) < 7:
                continue
            mu_values = [float(token) for token in tokens[6:]]
            raw_mu_values.append(mu_values)
            max_mu_columns = max(max_mu_columns, len(mu_values))
            valid_mu = [value for value in mu_values if value != MAGNIFICATION_INVALID_SENTINEL and np.isfinite(value)]
            rows.append(
                {
                    "id": str(tokens[0]),
                    "ra": float(tokens[1]),
                    "dec": float(tokens[2]),
                    "z_source": float(tokens[3]),
                    "cluster": str(tokens[4]),
                    "z_lens": float(tokens[5]),
                    "best_mu": float(valid_mu[0]) if valid_mu else float("nan"),
                    "median_mu": float(np.nanmedian(valid_mu)) if valid_mu else float("nan"),
                    "n_valid_mu_models": int(len(valid_mu)),
                }
            )
    if not rows:
        raise ValueError(f"No magnification data rows found in {catalog_path}.")
    for row, values in zip(rows, raw_mu_values):
        for index in range(max_mu_columns):
            value = values[index] if index < len(values) else float("nan")
            row[f"mu_model_{index:03d}"] = float("nan") if value == MAGNIFICATION_INVALID_SENTINEL else float(value)
    return pd.DataFrame(rows)


def load_hff_pagul21(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, dict[str, pd.DataFrame]]:
    """Load cluster and magnification tables for all HFF clusters."""

    root = Path(data_dir)
    cluster_catalog = read_lph_catalog(root / "HFF-clu_v4.4_20200715_lph.out")
    cluster_catalog_noirac = read_lph_catalog(root / "HFF-clu_v4.4_20200715-noirac_lph.out")
    parallel_catalog = read_lph_catalog(root / "HFF-par_v4.4_20200715_lph.out")
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for spec in HFF_CLUSTER_SPECS:
        result[spec.key] = {
            "cluster_photometry": cluster_catalog[cluster_catalog["FIELD"].astype(str) == spec.field].copy(),
            "cluster_photometry_noirac": cluster_catalog_noirac[
                cluster_catalog_noirac["FIELD"].astype(str) == spec.field
            ].copy(),
            "parallel_photometry": parallel_catalog[parallel_catalog["FIELD"].astype(str) == spec.field].copy(),
            "magnifications": read_magnification_catalog(root / spec.magnification_filename),
        }
    return result


def _cluster_reference(photometry: pd.DataFrame, magnifications: pd.DataFrame) -> tuple[float, float]:
    source = magnifications if not magnifications.empty else photometry
    return (
        float(pd.to_numeric(source["ra" if "ra" in source.columns else "RA"], errors="coerce").median()),
        float(pd.to_numeric(source["dec" if "dec" in source.columns else "DEC"], errors="coerce").median()),
    )


def _prepare_pagul21_cluster_members(
    photometry: pd.DataFrame,
    *,
    reference_ra: float,
    reference_dec: float,
    z_lens: float,
    member_radius_arcsec: float = DEFAULT_MEMBER_RADIUS_ARCSEC,
    member_z_tol: float = DEFAULT_MEMBER_Z_TOL,
    member_max_count: int = DEFAULT_MEMBER_MAX_COUNT,
    bcg_radius_arcsec: float = DEFAULT_BCG_RADIUS_ARCSEC,
) -> tuple[pd.DataFrame, pd.Series | None]:
    members = photometry.copy()
    x_arcsec: list[float] = []
    y_arcsec: list[float] = []
    for row in members.itertuples(index=False):
        x, y = _radec_offset_arcsec(row.RA, row.DEC, reference_ra, reference_dec)
        x_arcsec.append(x)
        y_arcsec.append(y)
    members["x_arcsec"] = x_arcsec
    members["y_arcsec"] = y_arcsec
    members["r_arcsec"] = np.sqrt(np.square(members["x_arcsec"]) + np.square(members["y_arcsec"]))
    z_best = pd.to_numeric(members["Z_BEST"], errors="coerce")
    z_ml = pd.to_numeric(members["Z_ML"], errors="coerce")
    f160w = pd.to_numeric(members["MAG_OBS8"], errors="coerce")
    mass = pd.to_numeric(members["MASS_MED"], errors="coerce")
    z_member = (np.abs(z_best - z_lens) <= member_z_tol) | (np.abs(z_ml - z_lens) <= member_z_tol)
    finite_f160w = f160w.notna() & np.isfinite(f160w) & (f160w > 0.0) & (f160w < 35.0)
    radius_member = members["r_arcsec"] <= member_radius_arcsec
    selected = members.loc[z_member & finite_f160w & radius_member].copy()
    if selected.empty:
        selected = members.loc[finite_f160w & radius_member].copy()
    selected["member_selection"] = "pagul21_photoz_f160w_radius"
    selected["bcg_score"] = (
        selected["r_arcsec"] / max(bcg_radius_arcsec, 1.0)
        + 0.35 * (pd.to_numeric(selected["MAG_OBS8"], errors="coerce") - float(pd.to_numeric(selected["MAG_OBS8"], errors="coerce").min()))
        - 0.15 * (pd.to_numeric(selected["MASS_MED"], errors="coerce").fillna(pd.to_numeric(selected["MASS_MED"], errors="coerce").median()) - float(mass.median(skipna=True) if mass.notna().any() else 0.0))
    )
    central = selected[selected["r_arcsec"] <= bcg_radius_arcsec].copy()
    if central.empty:
        central = selected.copy()
    bcg = central.sort_values(["bcg_score", "MAG_OBS8", "r_arcsec"]).iloc[0] if not central.empty else None
    if bcg is not None:
        selected["is_bcg_candidate"] = selected["IDENT"].astype(str) == str(bcg["IDENT"])
    else:
        selected["is_bcg_candidate"] = False
    selected = selected.sort_values(["MAG_OBS8", "r_arcsec"]).head(max(0, int(member_max_count))).copy()
    return selected.reset_index(drop=True), bcg


def _write_member_potfile(path: Path, members: pd.DataFrame) -> None:
    lines = ["#REFERENCE 0"]
    for row in members.itertuples(index=False):
        if bool(getattr(row, "is_bcg_candidate", False)):
            continue
        mag = _finite_or_nan(getattr(row, "MAG_OBS8", np.nan))
        if not np.isfinite(mag):
            continue
        lines.append(
            f"{str(getattr(row, 'IDENT')):>10s} {float(getattr(row, 'RA')): .8f} {float(getattr(row, 'DEC')): .8f} "
            f"1.0000 1.0000 0.0 {mag:.4f} 1.0000"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _catalog_radius_deg(
    photometry: pd.DataFrame,
    magnifications: pd.DataFrame,
    reference_ra: float,
    reference_dec: float,
    margin_arcmin: float,
) -> float:
    coords: list[SkyCoord] = []
    if not photometry.empty:
        phot_ra = pd.to_numeric(photometry["RA"], errors="coerce")
        phot_dec = pd.to_numeric(photometry["DEC"], errors="coerce")
        mask = phot_ra.notna() & phot_dec.notna()
        if mask.any():
            coords.append(SkyCoord(phot_ra[mask].to_numpy() * u.deg, phot_dec[mask].to_numpy() * u.deg))
    if not magnifications.empty:
        mag_ra = pd.to_numeric(magnifications["ra"], errors="coerce")
        mag_dec = pd.to_numeric(magnifications["dec"], errors="coerce")
        mask = mag_ra.notna() & mag_dec.notna()
        if mask.any():
            coords.append(SkyCoord(mag_ra[mask].to_numpy() * u.deg, mag_dec[mask].to_numpy() * u.deg))
    if not coords:
        return margin_arcmin / 60.0
    center = SkyCoord(reference_ra * u.deg, reference_dec * u.deg)
    max_sep_deg = max(float(center.separation(coord).deg.max()) for coord in coords)
    return max_sep_deg + margin_arcmin / 60.0


def query_simbad_specz_sources(
    reference_ra: float,
    reference_dec: float,
    radius_deg: float,
    *,
    timeout_sec: float = DEFAULT_SIMBAD_TIMEOUT_SEC,
) -> pd.DataFrame:
    """Query SIMBAD TAP for objects with redshift in a circular HFF footprint."""

    query = f"""
SELECT main_id, ra, dec, otype, rvz_redshift, rvz_qual, rvz_bibcode
FROM basic
WHERE rvz_redshift IS NOT NULL
  AND CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {reference_ra:.10f}, {reference_dec:.10f}, {radius_deg:.10f})
  ) = 1
"""
    body = urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}).encode("utf-8")
    request = Request(
        SIMBAD_TAP_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "lenscluster/1.0"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_sec) as response:
        text = response.read().decode("utf-8")
    if "ERROR" in text[:500].upper():
        raise RuntimeError(f"SIMBAD TAP query failed:\n{text[:1000]}")
    if not text.strip():
        return pd.DataFrame(
            columns=["main_id", "ra", "dec", "otype", "rvz_redshift", "rvz_qual", "rvz_bibcode"]
        )
    return pd.read_csv(io.StringIO(text), comment="#")


def _add_simbad_specz_matches(
    table: pd.DataFrame,
    simbad_sources: pd.DataFrame,
    *,
    ra_col: str,
    dec_col: str,
    match_arcsec: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = table.copy()
    for column in (
        "SIMBAD_MAIN_ID",
        "SIMBAD_RA",
        "SIMBAD_DEC",
        "SIMBAD_OTYPE",
        "SIMBAD_ZSPEC",
        "SIMBAD_RVZ_QUAL",
        "SIMBAD_RVZ_BIBCODE",
        "SIMBAD_MATCH_SEP_ARCSEC",
        "ZSPEC_SOURCE",
    ):
        enriched[column] = np.nan
    enriched["SIMBAD_MAIN_ID"] = ""
    enriched["SIMBAD_OTYPE"] = ""
    enriched["SIMBAD_RVZ_QUAL"] = ""
    enriched["SIMBAD_RVZ_BIBCODE"] = ""
    enriched["ZSPEC_SOURCE"] = ""
    if table.empty or simbad_sources.empty:
        return enriched, pd.DataFrame()

    simbad = simbad_sources.copy()
    for column in ("ra", "dec", "rvz_redshift"):
        simbad[column] = pd.to_numeric(simbad[column], errors="coerce")
    simbad = simbad.dropna(subset=["ra", "dec", "rvz_redshift"])
    if simbad.empty:
        return enriched, pd.DataFrame()

    ra = pd.to_numeric(enriched[ra_col], errors="coerce")
    dec = pd.to_numeric(enriched[dec_col], errors="coerce")
    valid = ra.notna() & dec.notna()
    if not valid.any():
        return enriched, pd.DataFrame()

    target_coords = SkyCoord(ra[valid].to_numpy() * u.deg, dec[valid].to_numpy() * u.deg)
    simbad_coords = SkyCoord(simbad["ra"].to_numpy() * u.deg, simbad["dec"].to_numpy() * u.deg)
    matched_index, sep2d, _ = target_coords.match_to_catalog_sky(simbad_coords)
    accepted = sep2d <= match_arcsec * u.arcsec
    target_indices = enriched.index[valid].to_numpy()[accepted]
    source_rows = simbad.iloc[matched_index[accepted]].reset_index(drop=True)
    seps = sep2d[accepted].arcsec
    if len(target_indices) == 0:
        return enriched, pd.DataFrame()

    enriched.loc[target_indices, "SIMBAD_MAIN_ID"] = source_rows["main_id"].astype(str).to_numpy()
    enriched.loc[target_indices, "SIMBAD_RA"] = source_rows["ra"].to_numpy()
    enriched.loc[target_indices, "SIMBAD_DEC"] = source_rows["dec"].to_numpy()
    enriched.loc[target_indices, "SIMBAD_OTYPE"] = source_rows.get("otype", pd.Series([""] * len(source_rows))).astype(str).to_numpy()
    enriched.loc[target_indices, "SIMBAD_ZSPEC"] = source_rows["rvz_redshift"].to_numpy()
    enriched.loc[target_indices, "SIMBAD_RVZ_QUAL"] = source_rows.get("rvz_qual", pd.Series([""] * len(source_rows))).astype(str).to_numpy()
    enriched.loc[target_indices, "SIMBAD_RVZ_BIBCODE"] = source_rows.get("rvz_bibcode", pd.Series([""] * len(source_rows))).astype(str).to_numpy()
    enriched.loc[target_indices, "SIMBAD_MATCH_SEP_ARCSEC"] = seps
    enriched.loc[target_indices, "ZSPEC_SOURCE"] = "SIMBAD"

    matches = enriched.loc[target_indices].copy()
    matches.insert(0, "catalog_index", target_indices)
    return enriched, matches


def _simbad_z_usable(
    catalog_z: float,
    simbad_z: float,
    z_lens: float,
    *,
    min_lens_delta: float,
    max_photoz_delta: float,
) -> bool:
    if not np.isfinite(simbad_z):
        return False
    if simbad_z <= z_lens + min_lens_delta:
        return False
    if not np.isfinite(catalog_z):
        return True
    return abs(simbad_z - catalog_z) / (1.0 + catalog_z) <= max_photoz_delta


def _mark_usable_simbad_z(
    table: pd.DataFrame,
    *,
    catalog_z_col: str,
    z_lens: float,
    min_lens_delta: float,
    max_photoz_delta: float,
) -> pd.DataFrame:
    marked = table.copy()
    if "SIMBAD_ZSPEC" not in marked.columns or catalog_z_col not in marked.columns:
        marked["SIMBAD_ZSPEC_USABLE_FOR_LENSING"] = False
        return marked
    catalog_z = pd.to_numeric(marked[catalog_z_col], errors="coerce")
    simbad_z = pd.to_numeric(marked["SIMBAD_ZSPEC"], errors="coerce")
    marked["SIMBAD_ZSPEC_USABLE_FOR_LENSING"] = [
        _simbad_z_usable(
            float(cat_z) if np.isfinite(cat_z) else float("nan"),
            float(sim_z) if np.isfinite(sim_z) else float("nan"),
            z_lens,
            min_lens_delta=min_lens_delta,
            max_photoz_delta=max_photoz_delta,
        )
        for cat_z, sim_z in zip(catalog_z, simbad_z)
    ]
    return marked


def _photometry_color_vector(phot_row: Any | None) -> np.ndarray:
    if phot_row is None:
        return np.full(len(COLOR_MAG_COLUMNS) - 1, np.nan)
    values = [_valid_mag(getattr(phot_row, column, np.nan)) for column in COLOR_MAG_COLUMNS]
    colors: list[float] = []
    for left, right in zip(values[:-1], values[1:]):
        if np.isfinite(left) and np.isfinite(right):
            colors.append(left - right)
        else:
            colors.append(float("nan"))
    return np.asarray(colors, dtype=float)


def _color_rms_delta(left: np.ndarray, right: np.ndarray, min_common_colors: int) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < min_common_colors:
        return float("nan")
    return float(np.sqrt(np.mean((left[mask] - right[mask]) ** 2)))


def _family_reliability(
    *,
    delta_z: float,
    color_rms: float,
    centroid_radius_arcsec: float,
    specz_tol: float,
    color_rms_max: float,
    max_separation_arcsec: float,
    reliability_floor: float,
    reliability_ceil: float,
) -> float:
    z_term = delta_z / max(specz_tol, 1.0e-12)
    color_term = color_rms / max(color_rms_max, 1.0e-12)
    radius_term = centroid_radius_arcsec / max(max_separation_arcsec, 1.0e-12)
    score = z_term**2 + color_term**2 + radius_term**2
    q = reliability_floor + (reliability_ceil - reliability_floor) * np.exp(-0.5 * score)
    return float(np.clip(q, reliability_floor, reliability_ceil))


def _simbad_system_hint(main_id: Any) -> str:
    match = SIMBAD_SYSTEM_PATTERN.search(str(main_id))
    if match is None:
        return ""
    return match.group(1)


def _connected_components(nodes: list[int], edges: dict[int, set[int]]) -> list[list[int]]:
    remaining = set(nodes)
    components: list[list[int]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            node = stack.pop()
            for neighbor in edges.get(node, set()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(sorted(component))
    return components


def assign_candidate_families(
    magnifications: pd.DataFrame,
    photometry: pd.DataFrame,
    *,
    specz_tol: float = DEFAULT_FAMILY_SPECZ_TOL,
    color_rms_max: float = DEFAULT_FAMILY_COLOR_RMS_MAX,
    min_common_colors: int = DEFAULT_FAMILY_MIN_COMMON_COLORS,
    min_family_size: int = DEFAULT_FAMILY_MIN_SIZE,
    max_separation_arcsec: float = DEFAULT_FAMILY_MAX_SEPARATION_ARCSEC,
    reliability_floor: float = DEFAULT_FAMILY_RELIABILITY_FLOOR,
    reliability_ceil: float = DEFAULT_FAMILY_RELIABILITY_CEIL,
    family_start: int = DEFAULT_CANDIDATE_FAMILY_START,
    require_simbad_system_hint: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign heuristic candidate families from close SIMBAD spec-z and colors."""

    assigned = magnifications.copy()
    assigned["CANDIDATE_FAMILY_ID"] = ""
    assigned["CANDIDATE_FAMILY_IMAGE_ID"] = np.nan
    assigned["CANDIDATE_FAMILY_METHOD"] = ""
    assigned["CANDIDATE_FAMILY_Z"] = np.nan
    assigned["CANDIDATE_FAMILY_COLOR_RMS_MAX"] = np.nan
    assigned["CANDIDATE_FAMILY_DELTA_Z"] = np.nan
    assigned["CANDIDATE_FAMILY_COLOR_RMS"] = np.nan
    assigned["CANDIDATE_FAMILY_CENTROID_RADIUS_ARCSEC"] = np.nan
    assigned["CANDIDATE_FAMILY_RELIABILITY"] = 1.0

    if assigned.empty or "SIMBAD_ZSPEC_USABLE_FOR_LENSING" not in assigned.columns:
        return assigned, pd.DataFrame(), pd.DataFrame()

    phot_by_id = {str(row.IDENT): row for row in photometry.itertuples(index=False)}
    candidate_indices: list[int] = []
    z_values: dict[int, float] = {}
    colors: dict[int, np.ndarray] = {}
    coords: dict[int, SkyCoord] = {}
    system_hints: dict[int, str] = {}
    for index, row in assigned.iterrows():
        if not bool(row.get("SIMBAD_ZSPEC_USABLE_FOR_LENSING", False)):
            continue
        system_hint = _simbad_system_hint(row.get("SIMBAD_MAIN_ID", ""))
        if require_simbad_system_hint and not system_hint:
            continue
        z = _finite_or_nan(row.get("SIMBAD_ZSPEC"))
        if not np.isfinite(z):
            continue
        phot_row = phot_by_id.get(str(row["id"]))
        color_vector = _photometry_color_vector(phot_row)
        if int(np.isfinite(color_vector).sum()) < min_common_colors:
            continue
        candidate_indices.append(index)
        z_values[index] = z
        colors[index] = color_vector
        coords[index] = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
        system_hints[index] = system_hint

    edges: dict[int, set[int]] = {index: set() for index in candidate_indices}
    pair_rows: list[dict[str, Any]] = []
    for i, left_index in enumerate(candidate_indices):
        for right_index in candidate_indices[i + 1 :]:
            if require_simbad_system_hint and system_hints[left_index] != system_hints[right_index]:
                continue
            separation_arcsec = float(coords[left_index].separation(coords[right_index]).arcsec)
            if separation_arcsec > max_separation_arcsec:
                continue
            z_left = z_values[left_index]
            z_right = z_values[right_index]
            z_delta = abs(z_left - z_right)
            if z_delta > specz_tol:
                continue
            color_rms = _color_rms_delta(colors[left_index], colors[right_index], min_common_colors)
            if not np.isfinite(color_rms) or color_rms > color_rms_max:
                continue
            edges[left_index].add(right_index)
            edges[right_index].add(left_index)
            pair_rows.append(
                {
                    "left_catalog_index": int(left_index),
                    "right_catalog_index": int(right_index),
                    "left_id": str(assigned.at[left_index, "id"]),
                    "right_id": str(assigned.at[right_index, "id"]),
                    "z_left": z_left,
                    "z_right": z_right,
                    "delta_z": z_delta,
                    "color_rms": color_rms,
                    "separation_arcsec": separation_arcsec,
                    "simbad_system_hint": system_hints[left_index],
                }
            )

    family_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    next_family = family_start
    for component in _connected_components(candidate_indices, edges):
        if len(component) < min_family_size:
            continue
        if all(len(edges[index]) == 0 for index in component):
            continue
        component_span = max(
            float(coords[left].separation(coords[right]).arcsec)
            for i, left in enumerate(component)
            for right in component[i + 1 :]
        )
        if component_span > max_separation_arcsec:
            continue
        family_id = str(next_family)
        next_family += 1
        family_z = float(np.median([z_values[index] for index in component]))
        component_color_matrix = np.vstack([colors[index] for index in component])
        family_color_median = np.asarray(
            [
                float(np.median(column_values[np.isfinite(column_values)]))
                if np.any(np.isfinite(column_values))
                else float("nan")
                for column_values in component_color_matrix.T
            ],
            dtype=float,
        )
        centroid_ra = float(np.mean([float(assigned.at[index, "ra"]) for index in component]))
        centroid_dec = float(np.mean([float(assigned.at[index, "dec"]) for index in component]))
        centroid = SkyCoord(centroid_ra * u.deg, centroid_dec * u.deg)
        component_pair_rms = [
            row["color_rms"]
            for row in pair_rows
            if row["left_catalog_index"] in component and row["right_catalog_index"] in component
        ]
        max_color_rms = float(max(component_pair_rms)) if component_pair_rms else float("nan")
        member_reliabilities: list[float] = []
        member_metrics: dict[int, dict[str, float]] = {}
        for index in component:
            delta_z = abs(z_values[index] - family_z)
            color_rms = _color_rms_delta(colors[index], family_color_median, min_common_colors)
            centroid_radius = float(coords[index].separation(centroid).arcsec)
            reliability = _family_reliability(
                delta_z=delta_z,
                color_rms=color_rms,
                centroid_radius_arcsec=centroid_radius,
                specz_tol=specz_tol,
                color_rms_max=color_rms_max,
                max_separation_arcsec=max_separation_arcsec,
                reliability_floor=reliability_floor,
                reliability_ceil=reliability_ceil,
            )
            member_reliabilities.append(reliability)
            member_metrics[index] = {
                "delta_z": delta_z,
                "color_rms": color_rms,
                "centroid_radius_arcsec": centroid_radius,
                "reliability": reliability,
            }
        family_rows.append(
            {
                "candidate_family_id": family_id,
                "n_images": len(component),
                "median_simbad_zspec": family_z,
                "max_pair_color_rms": max_color_rms,
                "max_pair_separation_arcsec": component_span,
                "median_reliability": float(np.median(member_reliabilities)),
                "min_reliability": float(np.min(member_reliabilities)),
                "simbad_system_hint": system_hints[component[0]],
                "method": "simbad_specz_color_graph",
                "specz_tol": specz_tol,
                "color_rms_max": color_rms_max,
                "max_separation_arcsec": max_separation_arcsec,
                "reliability_floor": reliability_floor,
                "reliability_ceil": reliability_ceil,
                "min_common_colors": min_common_colors,
                "require_simbad_system_hint": require_simbad_system_hint,
            }
        )
        for image_number, index in enumerate(component, start=1):
            metrics = member_metrics[index]
            assigned.at[index, "CANDIDATE_FAMILY_ID"] = family_id
            assigned.at[index, "CANDIDATE_FAMILY_IMAGE_ID"] = image_number
            assigned.at[index, "CANDIDATE_FAMILY_METHOD"] = "simbad_specz_color_graph"
            assigned.at[index, "CANDIDATE_FAMILY_Z"] = family_z
            assigned.at[index, "CANDIDATE_FAMILY_COLOR_RMS_MAX"] = max_color_rms
            assigned.at[index, "CANDIDATE_FAMILY_DELTA_Z"] = metrics["delta_z"]
            assigned.at[index, "CANDIDATE_FAMILY_COLOR_RMS"] = metrics["color_rms"]
            assigned.at[index, "CANDIDATE_FAMILY_CENTROID_RADIUS_ARCSEC"] = metrics["centroid_radius_arcsec"]
            assigned.at[index, "CANDIDATE_FAMILY_RELIABILITY"] = metrics["reliability"]
            member_rows.append(
                {
                    "candidate_family_id": family_id,
                    "image_number": image_number,
                    "catalog_index": int(index),
                    "id": str(assigned.at[index, "id"]),
                    "ra": float(assigned.at[index, "ra"]),
                    "dec": float(assigned.at[index, "dec"]),
                    "catalog_z": float(assigned.at[index, "z_source"]),
                    "simbad_zspec": z_values[index],
                    "simbad_main_id": str(assigned.at[index, "SIMBAD_MAIN_ID"]),
                    "simbad_system_hint": system_hints[index],
                    "simbad_match_sep_arcsec": _finite_or_nan(assigned.at[index, "SIMBAD_MATCH_SEP_ARCSEC"]),
                    "delta_z_from_family_median": metrics["delta_z"],
                    "color_rms_from_family_median": metrics["color_rms"],
                    "centroid_radius_arcsec": metrics["centroid_radius_arcsec"],
                    "family_reliability": metrics["reliability"],
                }
            )

    return assigned, pd.DataFrame(family_rows), pd.DataFrame(member_rows)


def _obs_arc_redshift(
    row: Any,
    z_lens: float,
    use_simbad_specz: bool,
    *,
    min_lens_delta: float,
    max_photoz_delta: float,
) -> tuple[float, bool]:
    default_z = float(row.z_source)
    if not use_simbad_specz or not hasattr(row, "SIMBAD_ZSPEC"):
        return default_z, False
    simbad_z = _finite_or_nan(getattr(row, "SIMBAD_ZSPEC"))
    if _simbad_z_usable(
        default_z,
        simbad_z,
        z_lens,
        min_lens_delta=min_lens_delta,
        max_photoz_delta=max_photoz_delta,
    ):
        return simbad_z, True
    return default_z, False


def _write_obs_arcs(
    path: Path,
    magnifications: pd.DataFrame,
    photometry: pd.DataFrame,
    *,
    z_lens: float,
    use_simbad_specz: bool,
    simbad_min_lens_delta: float,
    simbad_max_photoz_delta: float,
    use_candidate_families: bool,
) -> int:
    phot_by_id = {str(row.IDENT): row for row in photometry.itertuples(index=False)}
    rows_to_write: list[tuple[tuple[int, str, int, str], str]] = []
    n_simbad_z = 0
    for row in magnifications.itertuples(index=False):
        phot_row = phot_by_id.get(str(row.id))
        mag = _safe_mag(pd.Series(phot_row._asdict())) if phot_row is not None else 25.0
        candidate_family_id = str(getattr(row, "CANDIDATE_FAMILY_ID", "") or "")
        candidate_image_id = _finite_or_nan(getattr(row, "CANDIDATE_FAMILY_IMAGE_ID", np.nan))
        candidate_family_z = _finite_or_nan(getattr(row, "CANDIDATE_FAMILY_Z", np.nan))
        reliability = _finite_or_nan(getattr(row, "CANDIDATE_FAMILY_RELIABILITY", 1.0))
        if not np.isfinite(reliability):
            reliability = 1.0
        reliability = float(np.clip(reliability, 0.0, 1.0))
        z_source, used_simbad = _obs_arc_redshift(
            row,
            z_lens,
            use_simbad_specz,
            min_lens_delta=simbad_min_lens_delta,
            max_photoz_delta=simbad_max_photoz_delta,
        )
        if use_candidate_families and candidate_family_id and np.isfinite(candidate_image_id):
            label = f"{candidate_family_id}.{int(candidate_image_id)}"
            if np.isfinite(candidate_family_z):
                z_source = candidate_family_z
            sort_key = (0, candidate_family_id, int(candidate_image_id), str(row.id))
        else:
            label = f"{row.id}.1"
            sort_key = (1, str(row.id), 1, str(row.id))
        n_simbad_z += int(used_simbad)
        rows_to_write.append(
            (
                sort_key,
                f"{label:>12s} {float(row.ra): .8f} {float(row.dec): .8f} "
                f"{DEFAULT_IMAGE_A:.4f} {DEFAULT_IMAGE_B:.4f} {DEFAULT_IMAGE_THETA:.1f} "
                f"{z_source:.8f} {mag:.4f} {reliability:.6f}",
            )
        )
    lines = ["#REFERENCE 0"] + [line for _sort_key, line in sorted(rows_to_write, key=lambda item: item[0])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_simbad_z


def _write_par(
    path: Path,
    spec: HFFClusterSpec,
    reference_ra: float,
    reference_dec: float,
    *,
    bcg: pd.Series | None = None,
    member_potfile_name: str = "cluster_members_potfile.cat",
    member_mag0: float | None = None,
) -> None:
    if bcg is not None:
        bcg_x, bcg_y = _radec_offset_arcsec(bcg["RA"], bcg["DEC"], reference_ra, reference_dec)
        bcg_mag = _finite_or_nan(bcg.get("MAG_OBS8", np.nan))
        bcg_id = str(bcg.get("IDENT", "bcg"))
    else:
        bcg_x, bcg_y, bcg_mag, bcg_id = 0.0, 0.0, float("nan"), "bcg"
    mag0 = float(member_mag0) if member_mag0 is not None and np.isfinite(member_mag0) else 20.0
    text = f"""# Auto-generated bootstrap lenscluster input for {spec.cluster_name}.
# Source: Zenodo 10.5281/zenodo.5338978, prepared from Pagul21 HFF catalogs.
# WARNING: the Pagul21 files do not provide multiple-image family membership.
# The generated obs_arcs.cat contains one image per catalog object, so this file
# is a data-ingestion/bootstrap product, not a science-ready strong-lensing model.

runmode
    reference 3 {reference_ra:.8f} {reference_dec:.8f}
    inverse   3 0.5 100
    restart   0 12345
    end

image
    multfile    1 obs_arcs.cat
    forme       0
    sigposArcsec 0.5
    end

cosmology
    H0 70.0
    omega 0.3
    lambda 0.7
    end

grille
    nlens 1000
    nlens_opt 2
    nombre 256
    end

potentiel 1 # bootstrap cluster-scale dPIE halo
    profil      81
    x_centre    0.0
    y_centre    0.0
    ellipticite 0.3
    angle_pos   0.0
    core_radius 20.0
    cut_radius  2000.0
    v_disp      800.0
    z_lens      {spec.z_lens:.8f}
    end

limit 1
    v_disp      1 300.0 1600.0 0.1
    angle_pos   1 -180.0 180.0 0.1
    ellipticity 1 0.0 0.8 0.05
    x_centre    1 -60.0 60.0 0.1
    y_centre    1 -60.0 60.0 0.1
    core_radius 1 0.1 80.0 0.1
    end

potentiel 2 # Pagul21 BCG candidate {bcg_id}, F160W={bcg_mag:.4f}
    profil      81
    x_centre    {bcg_x:.6f}
    y_centre    {bcg_y:.6f}
    ellipticite 0.0
    angle_pos   0.0
    core_radius 0.2
    cut_radius  50.0
    v_disp      250.0
    z_lens      {spec.z_lens:.8f}
    end

limit 2
    v_disp      1 100.0 500.0 0.1
    cut_radius  1 5.0 150.0 0.1
    core_radius 1 0.01 10.0 0.1
    end

potfile
    filein 3 {member_potfile_name}
    type 81
    mag0 {mag0:.4f}
    core 0.0001
    z_lens {spec.z_lens:.8f}
    sigma 3 248 28.
    cut   1 1 50
    vdslope 0 3.33333 0
    slope 0 3.33333 0
    end

fini
"""
    path.write_text(text, encoding="utf-8")


def prepare_cluster_inputs(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = None,
    cluster_keys: list[str] | None = None,
    *,
    query_simbad_specz: bool = False,
    simbad_match_arcsec: float = DEFAULT_SIMBAD_MATCH_ARCSEC,
    simbad_query_margin_arcmin: float = DEFAULT_SIMBAD_QUERY_MARGIN_ARCMIN,
    simbad_timeout_sec: float = DEFAULT_SIMBAD_TIMEOUT_SEC,
    simbad_min_lens_delta: float = DEFAULT_SIMBAD_MIN_LENS_DELTA,
    simbad_max_photoz_delta: float = DEFAULT_SIMBAD_MAX_PHOTOZ_DELTA,
    use_simbad_specz: bool = True,
    assign_families: bool = False,
    use_pagul21_members: bool = True,
    member_radius_arcsec: float = DEFAULT_MEMBER_RADIUS_ARCSEC,
    member_z_tol: float = DEFAULT_MEMBER_Z_TOL,
    member_max_count: int = DEFAULT_MEMBER_MAX_COUNT,
    bcg_radius_arcsec: float = DEFAULT_BCG_RADIUS_ARCSEC,
    family_specz_tol: float = DEFAULT_FAMILY_SPECZ_TOL,
    family_color_rms_max: float = DEFAULT_FAMILY_COLOR_RMS_MAX,
    family_min_common_colors: int = DEFAULT_FAMILY_MIN_COMMON_COLORS,
    family_min_size: int = DEFAULT_FAMILY_MIN_SIZE,
    family_max_separation_arcsec: float = DEFAULT_FAMILY_MAX_SEPARATION_ARCSEC,
    family_reliability_floor: float = DEFAULT_FAMILY_RELIABILITY_FLOOR,
    family_reliability_ceil: float = DEFAULT_FAMILY_RELIABILITY_CEIL,
    require_simbad_system_hint: bool = False,
) -> list[dict[str, Any]]:
    root = Path(data_dir)
    output_root = Path(output_dir) if output_dir is not None else root / DEFAULT_OUTPUT_SUBDIR
    loaded = load_hff_pagul21(root)
    keys = cluster_keys or [spec.key for spec in HFF_CLUSTER_SPECS]
    summaries: list[dict[str, Any]] = []
    for key in keys:
        if key not in HFF_CLUSTER_BY_KEY:
            raise ValueError(f"Unknown HFF cluster key {key!r}. Valid keys: {sorted(HFF_CLUSTER_BY_KEY)}")
        spec = HFF_CLUSTER_BY_KEY[key]
        tables = loaded[key]
        cluster_dir = output_root / key
        cluster_dir.mkdir(parents=True, exist_ok=True)
        photometry = tables["cluster_photometry"].copy()
        magnifications = tables["magnifications"].copy()
        reference_ra, reference_dec = _cluster_reference(photometry, magnifications)
        members = pd.DataFrame()
        bcg: pd.Series | None = None
        member_mag0 = float("nan")
        member_potfile_name = "cluster_members_potfile.cat"
        if use_pagul21_members:
            members, bcg = _prepare_pagul21_cluster_members(
                photometry,
                reference_ra=reference_ra,
                reference_dec=reference_dec,
                z_lens=spec.z_lens,
                member_radius_arcsec=member_radius_arcsec,
                member_z_tol=member_z_tol,
                member_max_count=member_max_count,
                bcg_radius_arcsec=bcg_radius_arcsec,
            )
            non_bcg_members = members[~members.get("is_bcg_candidate", False)].copy()
            member_mag0 = float(pd.to_numeric(non_bcg_members.get("MAG_OBS8"), errors="coerce").min()) if not non_bcg_members.empty else float("nan")
            members.to_csv(cluster_dir / "pagul21_cluster_members.csv", index=False)
            if bcg is not None:
                pd.DataFrame([bcg.to_dict()]).to_csv(cluster_dir / "pagul21_bcg_candidate.csv", index=False)
            _write_member_potfile(cluster_dir / member_potfile_name, members)
        simbad_summary: dict[str, int] = {
            "n_simbad_sources": 0,
            "n_simbad_photometry_matches": 0,
            "n_simbad_magnification_matches": 0,
        }
        family_summary: dict[str, int] = {
            "n_candidate_families": 0,
            "n_candidate_family_images": 0,
        }
        if query_simbad_specz:
            radius_deg = _catalog_radius_deg(
                photometry,
                magnifications,
                reference_ra,
                reference_dec,
                simbad_query_margin_arcmin,
            )
            simbad_sources = query_simbad_specz_sources(
                reference_ra,
                reference_dec,
                radius_deg,
                timeout_sec=simbad_timeout_sec,
            )
            simbad_sources.to_csv(cluster_dir / "simbad_sources.csv", index=False)
            photometry, phot_matches = _add_simbad_specz_matches(
                photometry,
                simbad_sources,
                ra_col="RA",
                dec_col="DEC",
                match_arcsec=simbad_match_arcsec,
            )
            magnifications, mag_matches = _add_simbad_specz_matches(
                magnifications,
                simbad_sources,
                ra_col="ra",
                dec_col="dec",
                match_arcsec=simbad_match_arcsec,
            )
            phot_matches.to_csv(cluster_dir / "simbad_matches_photometry.csv", index=False)
            mag_matches.to_csv(cluster_dir / "simbad_matches_magnifications.csv", index=False)
            photometry = _mark_usable_simbad_z(
                photometry,
                catalog_z_col="Z_BEST",
                z_lens=spec.z_lens,
                min_lens_delta=simbad_min_lens_delta,
                max_photoz_delta=simbad_max_photoz_delta,
            )
            magnifications = _mark_usable_simbad_z(
                magnifications,
                catalog_z_col="z_source",
                z_lens=spec.z_lens,
                min_lens_delta=simbad_min_lens_delta,
                max_photoz_delta=simbad_max_photoz_delta,
            )
            simbad_summary = {
                "n_simbad_sources": int(len(simbad_sources)),
                "n_simbad_photometry_matches": int(len(phot_matches)),
                "n_simbad_magnification_matches": int(len(mag_matches)),
            }
        if assign_families:
            magnifications, candidate_families, candidate_members = assign_candidate_families(
                magnifications,
                photometry,
                specz_tol=family_specz_tol,
                color_rms_max=family_color_rms_max,
                min_common_colors=family_min_common_colors,
                min_family_size=family_min_size,
                max_separation_arcsec=family_max_separation_arcsec,
                reliability_floor=family_reliability_floor,
                reliability_ceil=family_reliability_ceil,
                require_simbad_system_hint=require_simbad_system_hint,
            )
            candidate_families.to_csv(cluster_dir / "candidate_families.csv", index=False)
            candidate_members.to_csv(cluster_dir / "candidate_family_members.csv", index=False)
            family_summary = {
                "n_candidate_families": int(len(candidate_families)),
                "n_candidate_family_images": int(len(candidate_members)),
            }
        photometry.to_csv(cluster_dir / "photometry_cluster.csv", index=False)
        tables["cluster_photometry_noirac"].to_csv(cluster_dir / "photometry_cluster_noirac.csv", index=False)
        tables["parallel_photometry"].to_csv(cluster_dir / "photometry_parallel.csv", index=False)
        magnifications.to_csv(cluster_dir / "magnifications.csv", index=False)
        n_obs_arcs_simbad_z = _write_obs_arcs(
            cluster_dir / "obs_arcs.cat",
            magnifications,
            photometry,
            z_lens=spec.z_lens,
            use_simbad_specz=use_simbad_specz,
            simbad_min_lens_delta=simbad_min_lens_delta,
            simbad_max_photoz_delta=simbad_max_photoz_delta,
            use_candidate_families=assign_families,
        )
        _write_par(
            cluster_dir / f"{key}_bootstrap.par",
            spec,
            reference_ra,
            reference_dec,
            bcg=bcg,
            member_potfile_name=member_potfile_name,
            member_mag0=member_mag0,
        )
        summary = {
            "cluster_key": key,
            "cluster_name": spec.cluster_name,
            "field": spec.field,
            "z_lens": spec.z_lens,
            "reference_ra": reference_ra,
            "reference_dec": reference_dec,
            "n_cluster_photometry": int(len(photometry)),
            "n_parallel_photometry": int(len(tables["parallel_photometry"])),
            "n_magnification_objects": int(len(magnifications)),
            "n_pagul21_members": int(len(members)),
            "bcg_candidate_id": None if bcg is None else str(bcg.get("IDENT")),
            "bcg_candidate_f160w": None if bcg is None else _finite_or_nan(bcg.get("MAG_OBS8")),
            **simbad_summary,
            **family_summary,
            "n_obs_arcs_simbad_z": int(n_obs_arcs_simbad_z),
            "par_path": str(cluster_dir / f"{key}_bootstrap.par"),
            "obs_arcs_path": str(cluster_dir / "obs_arcs.cat"),
            "strong_lensing_ready": False,
            "note": "Pagul21 does not provide multiple-image family membership; obs_arcs has one image per catalog object.",
        }
        (cluster_dir / "README.txt").write_text(
            "\n".join(
                [
                    f"Cluster: {spec.cluster_name}",
                    f"Field: {spec.field}",
                    f"Lens redshift: {spec.z_lens:.4f}",
                    f"Reference: RA={reference_ra:.8f}, Dec={reference_dec:.8f}",
                    f"Photometric rows: {len(photometry)}",
                    f"Magnification rows: {len(magnifications)}",
                    f"Pagul21 member rows: {len(members)}",
                    f"BCG candidate id: {None if bcg is None else str(bcg.get('IDENT'))}",
                    f"SIMBAD sources queried: {simbad_summary['n_simbad_sources']}",
                    f"SIMBAD photometry matches: {simbad_summary['n_simbad_photometry_matches']}",
                    f"SIMBAD magnification matches: {simbad_summary['n_simbad_magnification_matches']}",
                    f"obs_arcs rows using SIMBAD zspec: {n_obs_arcs_simbad_z}",
                    f"Candidate families: {family_summary['n_candidate_families']}",
                    f"Candidate-family images: {family_summary['n_candidate_family_images']}",
                    "",
                    "WARNING:",
                    "The Pagul21 Zenodo files do not include multiple-image family membership.",
                    "The generated obs_arcs.cat has one image per catalog object and is intended",
                    "for parser/integration work only. Add real multiple-image family labels before",
                    "using this as a science strong-lensing fit.",
                    "",
                    f"Run parser/integration smoke test with:",
                    f"python -m lenscluster.cluster_solver --par-path {cluster_dir / f'{key}_bootstrap.par'} --fit-mode joint --fit-method svi --svi-steps 1 --samples 2 --skip-validation --skip-plots",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output_root / "hff_pagul21_manifest.csv", index=False)
    (output_root / "hff_pagul21_manifest.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read and prepare Pagul21 HFF catalogs for lenscluster.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--clusters", nargs="+", default=None, help=f"Cluster keys: {', '.join(HFF_CLUSTER_BY_KEY)}")
    parser.add_argument("--summary-only", action="store_true", help="Read catalogs and print summary without writing outputs.")
    parser.add_argument(
        "--query-simbad-specz",
        action="store_true",
        help="Query SIMBAD TAP around each cluster and add coordinate-matched spectroscopic redshifts.",
    )
    parser.add_argument("--simbad-match-arcsec", type=float, default=DEFAULT_SIMBAD_MATCH_ARCSEC)
    parser.add_argument("--simbad-query-margin-arcmin", type=float, default=DEFAULT_SIMBAD_QUERY_MARGIN_ARCMIN)
    parser.add_argument("--simbad-timeout-sec", type=float, default=DEFAULT_SIMBAD_TIMEOUT_SEC)
    parser.add_argument(
        "--simbad-min-lens-delta",
        type=float,
        default=DEFAULT_SIMBAD_MIN_LENS_DELTA,
        help="Require matched SIMBAD redshifts used in obs_arcs.cat to exceed z_lens by this amount.",
    )
    parser.add_argument(
        "--simbad-max-photoz-delta",
        type=float,
        default=DEFAULT_SIMBAD_MAX_PHOTOZ_DELTA,
        help="Require |z_simbad - z_catalog| / (1 + z_catalog) below this value before replacing arc redshifts.",
    )
    parser.add_argument(
        "--no-use-simbad-specz",
        action="store_true",
        help="Write SIMBAD matches but keep obs_arcs.cat redshifts from the Pagul21 magnification catalogs.",
    )
    parser.add_argument(
        "--no-pagul21-members",
        action="store_true",
        help="Do not write Pagul21 member potfiles or explicit BCG candidate components into bootstrap .par files.",
    )
    parser.add_argument("--member-radius-arcsec", type=float, default=DEFAULT_MEMBER_RADIUS_ARCSEC)
    parser.add_argument("--member-z-tol", type=float, default=DEFAULT_MEMBER_Z_TOL)
    parser.add_argument("--member-max-count", type=int, default=DEFAULT_MEMBER_MAX_COUNT)
    parser.add_argument("--bcg-radius-arcsec", type=float, default=DEFAULT_BCG_RADIUS_ARCSEC)
    parser.add_argument(
        "--assign-families",
        action="store_true",
        help="Assign heuristic candidate families using close usable SIMBAD spec-z and HST color agreement.",
    )
    parser.add_argument("--family-specz-tol", type=float, default=DEFAULT_FAMILY_SPECZ_TOL)
    parser.add_argument("--family-color-rms-max", type=float, default=DEFAULT_FAMILY_COLOR_RMS_MAX)
    parser.add_argument("--family-min-common-colors", type=int, default=DEFAULT_FAMILY_MIN_COMMON_COLORS)
    parser.add_argument("--family-min-size", type=int, default=DEFAULT_FAMILY_MIN_SIZE)
    parser.add_argument("--family-max-separation-arcsec", type=float, default=DEFAULT_FAMILY_MAX_SEPARATION_ARCSEC)
    parser.add_argument("--family-reliability-floor", type=float, default=DEFAULT_FAMILY_RELIABILITY_FLOOR)
    parser.add_argument("--family-reliability-ceil", type=float, default=DEFAULT_FAMILY_RELIABILITY_CEIL)
    parser.add_argument(
        "--family-require-name-hint",
        action="store_true",
        help="Also require a matching SIMBAD object-name system hint. Off by default; coordinates/spec-z/colors drive grouping.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    keys = args.clusters
    if args.summary_only:
        loaded = load_hff_pagul21(args.data_dir)
        for key in keys or [spec.key for spec in HFF_CLUSTER_SPECS]:
            spec = HFF_CLUSTER_BY_KEY[key]
            tables = loaded[key]
            print(
                f"{key:6s} {spec.cluster_name:12s} z_lens={spec.z_lens:.4f} "
                f"cluster_rows={len(tables['cluster_photometry'])} "
                f"parallel_rows={len(tables['parallel_photometry'])} "
                f"magnification_rows={len(tables['magnifications'])}"
            )
        return
    summaries = prepare_cluster_inputs(
        args.data_dir,
        args.output_dir,
        keys,
        query_simbad_specz=args.query_simbad_specz,
        simbad_match_arcsec=args.simbad_match_arcsec,
        simbad_query_margin_arcmin=args.simbad_query_margin_arcmin,
        simbad_timeout_sec=args.simbad_timeout_sec,
        simbad_min_lens_delta=args.simbad_min_lens_delta,
        simbad_max_photoz_delta=args.simbad_max_photoz_delta,
        use_simbad_specz=not args.no_use_simbad_specz,
        assign_families=args.assign_families,
        use_pagul21_members=not args.no_pagul21_members,
        member_radius_arcsec=args.member_radius_arcsec,
        member_z_tol=args.member_z_tol,
        member_max_count=args.member_max_count,
        bcg_radius_arcsec=args.bcg_radius_arcsec,
        family_specz_tol=args.family_specz_tol,
        family_color_rms_max=args.family_color_rms_max,
        family_min_common_colors=args.family_min_common_colors,
        family_min_size=args.family_min_size,
        family_max_separation_arcsec=args.family_max_separation_arcsec,
        family_reliability_floor=args.family_reliability_floor,
        family_reliability_ceil=args.family_reliability_ceil,
        require_simbad_system_hint=args.family_require_name_hint,
    )
    for summary in summaries:
        print(
            f"{summary['cluster_key']}: {summary['par_path']} "
            f"(SIMBAD mag matches={summary['n_simbad_magnification_matches']}, "
            f"obs_arcs zspec={summary['n_obs_arcs_simbad_z']}, "
            f"candidate families={summary['n_candidate_families']})"
        )


if __name__ == "__main__":
    main()
