#!/usr/bin/env python
from __future__ import annotations

import argparse
from functools import partial
import itertools
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hff-catalog-plots")
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
from rich.console import Console
from rich.table import Table as RichTable
from scipy.spatial import cKDTree
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_path in (REPO_ROOT / "src",):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from lenscluster.rgb import (  # noqa: E402
    CALIBRATED_RGB_MINIMUM_SKY_SIGMA,
    DEFAULT_HFF_RGB_BANDS,
    DEFAULT_HFF_RGB_CHANNEL_GAINS,
    DEFAULT_HFF_RGB_CHANNEL_WEIGHTS,
    DEFAULT_HFF_RGB_HIGHLIGHT_CEILING,
    DEFAULT_HFF_RGB_HIGHLIGHT_KNEE,
    DEFAULT_HFF_RGB_HIGHLIGHT_SOFTNESS,
    DEFAULT_HFF_RGB_MINIMUM,
    DEFAULT_HFF_RGB_Q,
    DEFAULT_HFF_RGB_REFERENCE_BAND,
    DEFAULT_HFF_RGB_STRETCH,
    DEFAULT_HFF_RGB_WARM_HIGHLIGHT_DESATURATION,
    CalibratedRGBDisplayConfig,
    RGBDisplayConfig,
    build_rgb_display_from_band_images,
    compute_fnu_band_fluxscales,
    make_natural_rgb,
)


PAGUL2024_SOURCE = "pagul2024"
SHIPLEY2018_SOURCE = "shipley2018"
SHIPLEY2018_UNMATCHED_SOURCE = "shipley2018_unmatched"
NED_SOURCE = "ned"
SIMBAD_SOURCE = "simbad"
LAGATTUTA22_SOURCE = "lagattuta22"

DEFAULT_MATCH_RADIUS_ARCSEC = 0.5
DEFAULT_REDSHIFT_DIR = Path("data") / "HFF_Redshifts"
DEFAULT_REDSHIFT_MATCH_RADIUS_ARCSEC = 0.5
DEFAULT_LAGATTUTA22_PATH = Path("data") / "Lagattuta22" / "A370_PilotWINGS_data_catalog.fits"
DEFAULT_OUTPUT_DIR = Path("results") / "hff_master_catalogs"
DEFAULT_IMAGE_DIR = Path("data") / "BUFFALO_Images"
DEFAULT_IMAGE_SCALE = "auto"
IMAGE_SCALE_CHOICES = ("auto", "30mas", "60mas")
DEFAULT_MAX_FAMILY_SPAN_KPC = 600.0
DEFAULT_MIN_PAIR_SEPARATION_ARCSEC = 1.0
DEFAULT_MIN_COMMON_BANDS = 8
DEFAULT_PAIR_SCORE_THRESHOLD = 0.70
DEFAULT_TWO_IMAGE_SCORE_THRESHOLD = 0.75
DEFAULT_FAMILY_PROBABILITY_THRESHOLD = 0.25
DEFAULT_MAX_FAMILIES_PER_OBJECT = 3
DEFAULT_FAMILY_PAIR_BATCH_SIZE = 262_144
DEFAULT_FAMILY_PAIR_DIAGNOSTICS = "scored"
DEFAULT_IMAGE_FAMILY_FOV_KPC = 1000.0
DEFAULT_FAMILY_COLOR_RMS_MAX = 1.25
DEFAULT_FAMILY_PHOTOZ_DELTA_MAX = 1.0
DEFAULT_CONFLICT_COLOR_FAMILY_RMS_MAX = 0.75
DEFAULT_PHOTOZ_ONLY_FAMILY_PROBABILITY_CAP = 0.72
FAMILY_INCOMPLETE_SPECZ_CAP = 0.93
FAMILY_COMPLETE_SPECZ_CAP = 0.98
FAMILY_GENERIC_LARGE_SED_CAP = 0.85
FAMILY_GENERIC_LOW_MIN_PAIR_CAP = 0.75
FAMILY_PHOTOZ_ONLY_LARGE_SED_CAP = 0.62
FAMILY_PHOTOZ_ONLY_LOW_MIN_PAIR_CAP = 0.55
FAMILY_ANCHORED_CAP = 0.85
FAMILY_ANCHORED_LARGE_SED_OR_SPAN_CAP = 0.75
FAMILY_ANCHORED_LOW_MIN_PAIR_CAP = 0.70
DEFAULT_FAMILY_GROWTH_SEED_BATCH_SIZE = 2048
DEFAULT_FAMILY_GROWTH_MAX_OBJECTS = 12_000
DEFAULT_FAMILY_GROWTH_MAX_BATCH_CELLS = DEFAULT_FAMILY_GROWTH_SEED_BATCH_SIZE * DEFAULT_FAMILY_GROWTH_MAX_OBJECTS
DEFAULT_IMAGE_BRIGHT_MAG_F814W = 23.5
DEFAULT_IMAGE_HFF_FAINT_MAG_F814W = 28.5
DEFAULT_IMAGE_OUTER_FAINT_MAG_F814W = 27.0
DEFAULT_IMAGE_MIN_SIZE_ARCSEC = 0.11
DEFAULT_IMAGE_SIZE_PIXEL_SCALE_ARCSEC = 0.06
DEFAULT_STRONG_LENSING_RESCUE_FAINT_MAG_F814W = 30.8
DEFAULT_STRONG_LENSING_RESCUE_MIN_BANDS = 6
DEFAULT_REFERENCE_MATCH_RADIUS_ARCSEC = 0.5
DEFAULT_REFERENCE_FAMILY_PATHS = {
    "a370": Path("data")
    / "literature_lenstool_models"
    / "a370"
    / "niemiec_buffalo"
    / "hlsp_buffalo_hst_multi_abell370_multi_v1.0_sl-final.dat",
}
DEFAULT_IMAGE_PHOTOZ_MIN_MAG_F160W = 16.0
DEFAULT_IMAGE_PHOTOZ_MAX_MAG_F160W = 26.0
DEFAULT_IMAGE_PHOTOZ_MIN_NB_USED = 5.0
DEFAULT_IMAGE_PHOTOZ_MAX_DZ_NORM = 0.15
DEFAULT_MEMBER_PROBABILITY_THRESHOLD = 0.50
DEFAULT_LENSING_MEMBER_PROBABILITY_THRESHOLD = 0.80
DEFAULT_LENSING_BRIGHT_MAG_F160W = 22.5
DEFAULT_MEMBER_FAINT_MAG_F814W = 25.0
DEFAULT_BCG_SPECIAL_MAX = 5
DEFAULT_BCG_SPECIAL_RADIUS_KPC = 250.0
DEFAULT_PROGRESS_UPDATE_INTERVAL = 250
PLOT_KEY_BANDS = ("F435W", "F606W", "F814W", "F105W", "F125W", "F160W")
PLOT_RGB_BANDS = ("F435W", "F606W", "F814W")
PLOT_FOV_HALF_WIDTH_KPC = 500.0
PLOT_MASTER_CONTEXT_MAX_ROWS = 1000
PLOT_MEMBER_MAX_ROWS = 500
PLOT_FAMILY_MEMBER_MAX_ROWS = 500
PLOT_IMAGE_MAX_PIXELS = 2500
DEFAULT_FAMILY_CUTOUT_SIZE_ARCSEC = 5.0
DEFAULT_FAMILY_CUTOUT_CIRCLE_RADIUS_ARCSEC = 0.6
DEFAULT_FAMILY_CUTOUT_FAMILIES_PER_PAGE = 6
DEFAULT_FAMILY_CUTOUT_BANDS = DEFAULT_HFF_RGB_BANDS
INVALID_SENTINELS = {-999.0, -99.0, 99.0, -1.0, 1.0e9}
ZSPEC_CONFLICT_TOL = 0.005
ZSPEC_EXCELLENT_TOL = 0.005
ZSPEC_HARD_CONFLICT_TOL = 0.01
ZSPEC_HARD_CONFLICT_NORM_TOL = 0.005
MEMBER_Z_TOL = 0.12
C_KMS = 299792.458
BACKGROUND_Z_MARGIN = 0.1
SECURE_PROBABLE_SPEC_RANK = 2.0
PHOTOZ_ONLY_SCORE_CAP = 0.72
SPEC_PHOTO_SCORE_CAP = 0.70
COLOR_RMS_REJECT = DEFAULT_FAMILY_COLOR_RMS_MAX
COLOR_RMS_SCALE = 0.25
FAMILY_COLOR_RMS_STRONG = 0.50
FAMILY_COLOR_RMS_ACCEPTABLE = 1.00
PHOTOZ_SIGMA_FLOOR = 0.15
PHOTOZ_SIGMA_SCALE = 0.08
SPECZ_CONFIDENCE_SECURE = 3.0
SPECZ_CONFIDENCE_PROBABLE = 2.0
SPECZ_CONFIDENCE_TENTATIVE = 1.0
SPECZ_CONFIDENCE_FALLBACK = 0.5
SPECZ_CONFIDENCE_LOW = 0.0
SPECZ_SOURCE_PRIORITY = {
    PAGUL2024_SOURCE: 3,
    LAGATTUTA22_SOURCE: 2,
    NED_SOURCE: 2,
    SIMBAD_SOURCE: 2,
    SHIPLEY2018_SOURCE: 1,
}
EXTERNAL_REDSHIFT_SOURCES = (NED_SOURCE, SIMBAD_SOURCE, LAGATTUTA22_SOURCE)
LAGATTUTA22_REFERENCE = "Lagattuta et al. 2022 Pilot-WINGS"
ARCSEC_PER_RADIAN = 206264.80624709636

PAIR_REJECT_NONE = 0
PAIR_REJECT_TOO_CLOSE = 1
PAIR_REJECT_TOO_FAR = 2
PAIR_REJECT_SPECZ_CONFLICT = 3
PAIR_REJECT_INSUFFICIENT_BANDS = 4
PAIR_REJECT_COLOR_RMS = 5
PAIR_REJECT_MISSING_QUALIFIED_PHOTOZ = 6
PAIR_REJECT_PHOTOZ_INCONSISTENT = 7
PAIR_REJECT_MISSING_STRONG_SPECZ = 8
PAIR_REJECT_PHOTOZ_DELTA = 9
PAIR_REJECT_REASON_BY_CODE = {
    PAIR_REJECT_NONE: "",
    PAIR_REJECT_TOO_CLOSE: "too_close_or_invalid_separation",
    PAIR_REJECT_TOO_FAR: "separation_exceeds_max_family_span",
    PAIR_REJECT_SPECZ_CONFLICT: "secure_or_probable_specz_conflict",
    PAIR_REJECT_INSUFFICIENT_BANDS: "insufficient_common_bands",
    PAIR_REJECT_COLOR_RMS: "color_rms_too_large",
    PAIR_REJECT_MISSING_QUALIFIED_PHOTOZ: "missing_qualified_photoz",
    PAIR_REJECT_PHOTOZ_INCONSISTENT: "photoz_inconsistent",
    PAIR_REJECT_MISSING_STRONG_SPECZ: "missing_strong_specz",
    PAIR_REJECT_PHOTOZ_DELTA: "photoz_delta_too_large",
}
PAIR_RELATION_BOTH_SPECZ = 1
PAIR_RELATION_SPECZ_PHOTOZ = 2
PAIR_RELATION_SINGLE_SPECZ_NO_PHOTOZ = 3
PAIR_RELATION_PHOTOZ_ONLY = 4
PAIR_RELATION_NO_REDSHIFT = 5
PAIR_RELATION_SPECZ_CONFLICT = 6
PAIR_RELATION_SECURE_SPECZ_PHOTOZ = 7
PAIR_RELATION_QUALIFIED_PHOTOZ = 8
PAIR_RELATION_MISSING_QUALIFIED_PHOTOZ = 9
PAIR_RELATION_PHOTOZ_INCONSISTENT = 10
PAIR_RELATION_MISSING_STRONG_SPECZ = 11
PAIR_RELATION_COLOR_GUIDED_REDSHIFT_CONFLICT = 12
PAIR_RELATION_BY_CODE = {
    0: "",
    PAIR_RELATION_BOTH_SPECZ: "both_specz",
    PAIR_RELATION_SPECZ_PHOTOZ: "specz_photoz",
    PAIR_RELATION_SINGLE_SPECZ_NO_PHOTOZ: "single_specz_no_photoz",
    PAIR_RELATION_PHOTOZ_ONLY: "photoz_only",
    PAIR_RELATION_NO_REDSHIFT: "no_redshift_pair",
    PAIR_RELATION_SPECZ_CONFLICT: "secure_or_probable_specz_conflict",
    PAIR_RELATION_SECURE_SPECZ_PHOTOZ: "secure_specz_photoz",
    PAIR_RELATION_QUALIFIED_PHOTOZ: "qualified_photoz",
    PAIR_RELATION_MISSING_QUALIFIED_PHOTOZ: "missing_qualified_photoz",
    PAIR_RELATION_PHOTOZ_INCONSISTENT: "photoz_inconsistent",
    PAIR_RELATION_MISSING_STRONG_SPECZ: "missing_strong_specz",
    PAIR_RELATION_COLOR_GUIDED_REDSHIFT_CONFLICT: "color_guided_redshift_conflict",
}


@dataclass(frozen=True)
class ClusterSpec:
    key: str
    pagul_field: str
    shipley_cluster: str
    z_lens: float
    pagul_slugs: tuple[str, ...]


@dataclass(frozen=True)
class BandSpec:
    name: str
    pagul_flux: str
    shipley_flux: str


@dataclass(frozen=True)
class PlotBandImage:
    band: str
    path: Path
    hdu_index: int
    shape: tuple[int, int]
    wcs: WCS
    pixel_scale_arcsec: float
    photflam: float | None = None
    photplam: float | None = None
    background: float = 0.0
    background_sigma: float = 0.0


@dataclass(frozen=True)
class OverlayCrop:
    data: np.ndarray
    extent: tuple[float, float, float, float]
    wcs: WCS
    x_min: int
    y_min: int
    stride: int
    center_ra: float
    center_dec: float
    z_lens: float


@dataclass
class PairScoreResult:
    pairs: pd.DataFrame
    accepted_arrays: dict[str, Any]
    metrics: dict[str, Any]


@dataclass
class FamilyGrowthResult:
    families: list[set[str]]
    family_masks: np.ndarray
    compact_object_ids: list[str]
    compact_candidate_indices: np.ndarray
    pair_score_matrix: np.ndarray
    sed_rms_matrix: np.ndarray
    separation_arcsec_matrix: np.ndarray
    separation_kpc_matrix: np.ndarray
    metrics: dict[str, Any]


CLUSTER_SPECS: tuple[ClusterSpec, ...] = (
    ClusterSpec("a2744", "A2744clu", "A2744-clu", 0.3080, ("abell2744", "a2744")),
    ClusterSpec(
        "a370",
        "A370clu",
        "A370-clu",
        0.3750,
        ("abell370", "a370"),
    ),
    ClusterSpec("as1063", "AS1063clu", "A1063-clu", 0.3480, ("abells1063", "as1063", "rxcj2248")),
    ClusterSpec("m0416", "M0416clu", "M0416-clu", 0.3960, ("macs0416", "m0416")),
    ClusterSpec("m0717", "M0717clu", "M0717-clu", 0.5450, ("macs0717", "m0717")),
    ClusterSpec("m1149", "M1149clu", "M1149-clu", 0.5430, ("macs1149", "m1149")),
)
CLUSTER_BY_KEY = {spec.key: spec for spec in CLUSTER_SPECS}
COSMOLOGY = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=2.725)

BANDS: tuple[BandSpec, ...] = (
    BandSpec("F275W", "FLUX_F275W", "FF275W"),
    BandSpec("F336W", "FLUX_F336W", "FF336W"),
    BandSpec("F435W", "FLUX_F435W", "FF435W"),
    BandSpec("F475W", "FLUX_F475W", "FF475W"),
    BandSpec("F606W", "FLUX_F606W", "FF606W"),
    BandSpec("F625W", "FLUX_F625W", "FF625W"),
    BandSpec("F814W", "FLUX_F814W", "FF814W"),
    BandSpec("F105W", "FLUX_F105W", "FF105W"),
    BandSpec("F110W", "FLUX_F110W", "FF110W"),
    BandSpec("F125W", "FLUX_F125W", "FF125W"),
    BandSpec("F140W", "FLUX_F140W", "FF140W"),
    BandSpec("F160W", "FLUX_F160W", "FF160W"),
    BandSpec("Ks", "FLUX_Ks", "FKs"),
    BandSpec("I1", "FLUX_I1", "Fch1"),
    BandSpec("I2", "FLUX_I2", "Fch2"),
)
IMAGE_MAG_BANDS: tuple[str, ...] = tuple(band.name for band in BANDS)
MAG_COLUMNS: tuple[str, ...] = tuple(f"mag_{band}" for band in IMAGE_MAG_BANDS)

FAMILY_COLUMNS = [
    "cluster_key",
    "candidate_family_id",
    "n_images",
    "family_probability",
    "max_separation_arcsec",
    "max_separation_kpc",
    "family_z_best",
    "family_z_method",
    "min_specz_confidence",
    "median_sed_rms",
    "min_pair_score",
    "review_flags",
]
MEMBER_COLUMNS = [
    "cluster_key",
    "candidate_family_id",
    "object_id",
    "ra",
    "dec",
    "membership_probability",
    "raw_probability",
    "zspec_best",
    "zspec_best_confidence",
    "zspec_best_native_quality",
    "zphot_best",
    "n_valid_bands",
    "object_source",
    "catalog_sources",
    "image_preclean_selected",
    "image_preclean_reject_reason",
    "image_size_arcsec",
    "image_ellipticity",
    "image_photoz_quality_selected",
    "image_photoz_reject_reason",
    "image_zphot_family",
]
PAIR_COLUMNS = [
    "cluster_key",
    "left_object_id",
    "right_object_id",
    "separation_arcsec",
    "separation_kpc",
    "pair_score",
    "specz_score",
    "photoz_score",
    "zphot_delta",
    "color_score",
    "sed_rms",
    "n_common_bands",
    "hard_reject_reason",
]

RED_SEQUENCE_PLANES: tuple[tuple[str, str, str], ...] = (
    ("F435W", "F606W", "F814W"),
    ("F606W", "F814W", "F814W"),
    ("F814W", "F105W", "F105W"),
    ("F814W", "F125W", "F125W"),
    ("F814W", "F160W", "F160W"),
)


class MissingCatalogError(FileNotFoundError):
    pass


def _decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return value


def _table_to_dataframe(path: Path) -> pd.DataFrame:
    table = Table.read(path)
    df = table.to_pandas()
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].map(_decode_value)
    return df


def _normal_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(_decode_value(value)).strip()


def _to_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(result):
        return float("nan")
    return result


def _valid_number(value: Any) -> float:
    result = _to_float(value)
    if not np.isfinite(result):
        return float("nan")
    if result in INVALID_SENTINELS:
        return float("nan")
    return result


def valid_redshift(value: Any, *, max_z: float = 20.0) -> float:
    result = _valid_number(value)
    if np.isfinite(result) and 0.0 < result < max_z:
        return result
    return float("nan")


def _valid_mag(value: Any) -> float:
    result = _valid_number(value)
    if np.isfinite(result) and 0.0 < result < 80.0:
        return result
    return float("nan")


def _first_finite(values: Iterable[float]) -> float:
    for value in values:
        if np.isfinite(value):
            return float(value)
    return float("nan")


def flux_to_abmag_uJy(flux_uJy: Any) -> float:
    flux = _to_float(flux_uJy)
    if not np.isfinite(flux) or flux <= 0.0:
        return float("nan")
    return float(23.9 - 2.5 * math.log10(flux))


def flux_to_abmag_cgs_fnu(flux_fnu: Any) -> float:
    flux = _to_float(flux_fnu)
    if not np.isfinite(flux) or flux <= 0.0:
        return float("nan")
    return float(-2.5 * math.log10(flux) - 48.6)


def infer_pagul_flux_scale(pagul: pd.DataFrame) -> str:
    positive_values: list[float] = []
    for band in BANDS:
        if band.pagul_flux not in pagul.columns:
            continue
        values = pd.to_numeric(pagul[band.pagul_flux], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            positive_values.extend(values[: min(values.size, 1000)].tolist())
    if not positive_values:
        return "unknown"
    median_flux = float(np.median(np.asarray(positive_values)))
    return "cgs_fnu" if median_flux < 1.0e-10 else "uJy"


def pagul_flux_to_abmag(flux: Any, scale: str) -> float:
    if scale == "cgs_fnu":
        return flux_to_abmag_cgs_fnu(flux)
    if scale == "uJy":
        return flux_to_abmag_uJy(flux)
    return float("nan")


def locate_pagul_catalog(spec: ClusterSpec, pagul_dir: Path) -> Path:
    candidates = sorted(path for path in pagul_dir.glob("*catalog.fits") if path.is_file())
    matches = [
        path
        for path in candidates
        if any(slug.lower() in path.name.lower() for slug in spec.pagul_slugs)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        expected = ", ".join(f"*{slug}*catalog.fits" for slug in spec.pagul_slugs)
        raise MissingCatalogError(f"Missing Pagul2024 catalog for {spec.key}; expected one of {expected} in {pagul_dir}.")
    match_list = ", ".join(str(path) for path in matches)
    raise MissingCatalogError(f"Multiple Pagul2024 catalog matches for {spec.key}: {match_list}")


def one_to_one_sky_match(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_ra: str,
    left_dec: str,
    right_ra: str,
    right_dec: str,
    radius_arcsec: float,
) -> pd.DataFrame:
    """Match each left row to its closest right row inside the search radius."""

    columns = ["left_index", "right_index", "separation_arcsec"]
    if left.empty or right.empty:
        return pd.DataFrame(columns=columns)

    left_ra_values = pd.to_numeric(left[left_ra], errors="coerce")
    left_dec_values = pd.to_numeric(left[left_dec], errors="coerce")
    right_ra_values = pd.to_numeric(right[right_ra], errors="coerce")
    right_dec_values = pd.to_numeric(right[right_dec], errors="coerce")
    left_valid = left_ra_values.notna() & left_dec_values.notna()
    right_valid = right_ra_values.notna() & right_dec_values.notna()
    if not left_valid.any() or not right_valid.any():
        return pd.DataFrame(columns=columns)

    left_indices = left.index[left_valid].to_numpy()
    right_indices = right.index[right_valid].to_numpy()
    left_coords = SkyCoord(left_ra_values[left_valid].to_numpy() * u.deg, left_dec_values[left_valid].to_numpy() * u.deg)
    right_coords = SkyCoord(
        right_ra_values[right_valid].to_numpy() * u.deg,
        right_dec_values[right_valid].to_numpy() * u.deg,
    )
    rows: list[dict[str, Any]] = []
    closest_right_positions, sep2d, _ = left_coords.match_to_catalog_sky(right_coords)
    for left_position, (right_position, sep) in enumerate(zip(closest_right_positions, sep2d.arcsec)):
        if float(sep) > radius_arcsec:
            continue
        rows.append(
            {
                "left_index": int(left_indices[left_position]),
                "right_index": int(right_indices[right_position]),
                "separation_arcsec": float(sep),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _filter_pagul_cluster(pagul: pd.DataFrame, spec: ClusterSpec) -> pd.DataFrame:
    if "FIELD" not in pagul.columns:
        return pagul.reset_index(drop=True)
    field = pagul["FIELD"].map(_normal_string)
    filtered = pagul.loc[field == spec.pagul_field].copy()
    if filtered.empty:
        return pagul.reset_index(drop=True)
    return filtered.reset_index(drop=True)


def _filter_shipley_cluster(shipley: pd.DataFrame, spec: ClusterSpec) -> pd.DataFrame:
    cl = shipley["Cl"].map(_normal_string)
    return shipley.loc[cl == spec.shipley_cluster].copy().reset_index(drop=True)


EXTERNAL_REDSHIFT_COLUMNS = [
    "source",
    "field",
    "external_id",
    "ra",
    "dec",
    "redshift_raw",
    "zspec",
    "native_quality",
    "reference",
    "object_type",
    "id_source",
    "multiple_image_id",
]


def _empty_external_redshift_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EXTERNAL_REDSHIFT_COLUMNS)


def _ned_is_spectroscopic_flag(value: Any) -> bool:
    text = _normal_string(value).upper()
    return text.startswith("S")


def _standardize_ned_redshifts(path: Path, *, field: str) -> pd.DataFrame:
    if not path.exists():
        return _empty_external_redshift_frame()
    raw = pd.read_csv(path, low_memory=False)
    if raw.empty:
        return _empty_external_redshift_frame()
    redshift_raw = pd.to_numeric(raw.get("Redshift", pd.Series(np.nan, index=raw.index)), errors="coerce")
    flags = raw.get("Redshift Flag", pd.Series("", index=raw.index)).map(_normal_string)
    zspec = redshift_raw.where(flags.map(_ned_is_spectroscopic_flag), np.nan)
    result = pd.DataFrame(
        {
            "source": NED_SOURCE,
            "field": field,
            "external_id": raw.get("Object Name", pd.Series("", index=raw.index)).map(_normal_string),
            "ra": pd.to_numeric(raw.get("RA", pd.Series(np.nan, index=raw.index)), errors="coerce"),
            "dec": pd.to_numeric(raw.get("DEC", pd.Series(np.nan, index=raw.index)), errors="coerce"),
            "redshift_raw": redshift_raw,
            "zspec": zspec,
            "native_quality": flags,
            "reference": raw.get("References", pd.Series("", index=raw.index)).map(_normal_string),
            "object_type": raw.get("Type", pd.Series("", index=raw.index)).map(_normal_string),
        }
    )
    return result.loc[result["ra"].notna() & result["dec"].notna()].reset_index(drop=True)


def _standardize_simbad_redshifts(path: Path, *, field: str) -> pd.DataFrame:
    if not path.exists():
        return _empty_external_redshift_frame()
    raw = pd.read_csv(path, low_memory=False)
    if raw.empty:
        return _empty_external_redshift_frame()
    redshift_raw = pd.to_numeric(raw.get("rvz_redshift", pd.Series(np.nan, index=raw.index)), errors="coerce")
    result = pd.DataFrame(
        {
            "source": SIMBAD_SOURCE,
            "field": field,
            "external_id": raw.get("main_id", pd.Series("", index=raw.index)).map(_normal_string),
            "ra": pd.to_numeric(raw.get("ra", pd.Series(np.nan, index=raw.index)), errors="coerce"),
            "dec": pd.to_numeric(raw.get("dec", pd.Series(np.nan, index=raw.index)), errors="coerce"),
            "redshift_raw": redshift_raw,
            "zspec": redshift_raw,
            "native_quality": raw.get("rvz_qual", pd.Series("", index=raw.index)).map(_normal_string),
            "reference": raw.get("rvz_bibcode", pd.Series("", index=raw.index)).map(_normal_string),
            "object_type": raw.get("otype", pd.Series("", index=raw.index)).map(_normal_string),
        }
    )
    return result.loc[result["ra"].notna() & result["dec"].notna()].reset_index(drop=True)


def _standardize_lagattuta22_redshifts(path: Path, spec: ClusterSpec) -> pd.DataFrame:
    if spec.key != "a370" or not path.exists():
        return _empty_external_redshift_frame()
    raw = _table_to_dataframe(path)
    if raw.empty:
        return _empty_external_redshift_frame()

    fields = raw.get("Field", pd.Series("", index=raw.index)).map(_normal_string)
    id_sources = raw.get("idfrom", pd.Series("", index=raw.index)).map(_normal_string)
    identifiers = raw.get("iden", pd.Series("", index=raw.index)).map(_normal_string)
    external_ids = [
        f"{identifier}:{id_source}:{field}"
        for identifier, id_source, field in zip(identifiers, id_sources, fields, strict=False)
    ]
    redshift_raw = pd.to_numeric(raw.get("z", pd.Series(np.nan, index=raw.index)), errors="coerce")
    result = pd.DataFrame(
        {
            "source": LAGATTUTA22_SOURCE,
            "field": fields,
            "external_id": external_ids,
            "ra": pd.to_numeric(raw.get("RA", pd.Series(np.nan, index=raw.index)), errors="coerce"),
            "dec": pd.to_numeric(raw.get("DEC", pd.Series(np.nan, index=raw.index)), errors="coerce"),
            "redshift_raw": redshift_raw,
            "zspec": redshift_raw.map(valid_redshift),
            "native_quality": pd.to_numeric(raw.get("zconf", pd.Series(np.nan, index=raw.index)), errors="coerce"),
            "reference": LAGATTUTA22_REFERENCE,
            "object_type": id_sources,
            "id_source": id_sources,
            "multiple_image_id": raw.get("MUL", pd.Series("", index=raw.index)).map(_normal_string),
        }
    )
    return result.loc[result["ra"].notna() & result["dec"].notna()].reset_index(drop=True)


def load_external_redshift_catalogs(
    redshift_dir: Path,
    spec: ClusterSpec,
    *,
    lagattuta22_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    cluster_dir = redshift_dir / spec.key
    frames: dict[str, list[pd.DataFrame]] = {source: [] for source in EXTERNAL_REDSHIFT_SOURCES}
    for field in ("core", "parallel"):
        frames[NED_SOURCE].append(_standardize_ned_redshifts(cluster_dir / f"ned_{field}_redshifts.csv", field=field))
        frames[SIMBAD_SOURCE].append(
            _standardize_simbad_redshifts(cluster_dir / f"simbad_{field}_redshifts.csv", field=field)
        )
    if lagattuta22_path is not None:
        frames[LAGATTUTA22_SOURCE].append(_standardize_lagattuta22_redshifts(lagattuta22_path, spec))
    return {
        source: pd.concat(parts, ignore_index=True) if parts else _empty_external_redshift_frame()
        for source, parts in frames.items()
    }


def _candidate_sources(values: Iterable[str]) -> str:
    return "|".join(value for value in values if value)


def _specz_confidence(source: str, native_quality: Any) -> tuple[float, str]:
    quality = _to_float(native_quality)
    if source == PAGUL2024_SOURCE:
        if np.isfinite(quality) and quality > 3.0:
            return SPECZ_CONFIDENCE_SECURE, "secure"
        if np.isfinite(quality) and quality == 3.0:
            return SPECZ_CONFIDENCE_PROBABLE, "probable"
        if np.isfinite(quality) and 2.0 <= quality <= 2.5:
            return SPECZ_CONFIDENCE_TENTATIVE, "tentative"
        return SPECZ_CONFIDENCE_LOW, "low_quality"
    if source == LAGATTUTA22_SOURCE:
        if np.isfinite(quality) and quality >= 3.0:
            return SPECZ_CONFIDENCE_SECURE, "secure"
        if np.isfinite(quality) and quality >= 2.0:
            return SPECZ_CONFIDENCE_PROBABLE, "probable"
        return SPECZ_CONFIDENCE_LOW, "low_quality"
    if source == SHIPLEY2018_SOURCE:
        return SPECZ_CONFIDENCE_FALLBACK, "fallback"
    if source in {NED_SOURCE, SIMBAD_SOURCE}:
        return SPECZ_CONFIDENCE_PROBABLE, "probable"
    return SPECZ_CONFIDENCE_LOW, "unknown"


def choose_best_zspec(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    allowed_sources = {PAGUL2024_SOURCE, SHIPLEY2018_SOURCE, NED_SOURCE, SIMBAD_SOURCE, LAGATTUTA22_SOURCE}
    valid = [
        candidate.copy()
        for candidate in candidates
        if str(candidate.get("source", "")) in allowed_sources and np.isfinite(valid_redshift(candidate.get("z")))
    ]
    if not valid:
        return {
            "zspec_best": float("nan"),
            "zspec_best_source": "",
            "zspec_best_quality": float("nan"),
            "zspec_best_confidence": "",
            "zspec_best_confidence_rank": float("nan"),
            "zspec_best_native_quality": float("nan"),
            "zspec_best_native_quality_label": "",
            "zspec_selection_note": "no_valid_specz",
            "zspec_conflict": False,
            "zspec_delta_max": float("nan"),
            "zspec_candidate_sources": "",
            "zspec_candidate_values": "",
        }

    for order, candidate in enumerate(valid):
        source = str(candidate.get("source", ""))
        native_quality = candidate.get("native_quality", candidate.get("quality"))
        confidence_rank, confidence = _specz_confidence(source, native_quality)
        priority = SPECZ_SOURCE_PRIORITY.get(source, _to_float(candidate.get("priority", 0)))
        candidate["z"] = valid_redshift(candidate["z"])
        candidate["native_quality_sort"] = _to_float(native_quality)
        candidate["native_quality_label"] = _normal_string(native_quality)
        candidate["confidence_rank"] = confidence_rank
        candidate["confidence"] = confidence
        candidate["priority_sort"] = priority
        candidate["order_sort"] = order
    best = sorted(valid, key=lambda item: (-item["confidence_rank"], -item["priority_sort"], item["order_sort"]))[0]
    redshifts = np.asarray([candidate["z"] for candidate in valid], dtype=float)
    delta_max = float(np.nanmax(redshifts) - np.nanmin(redshifts)) if redshifts.size > 1 else 0.0
    secure_redshifts = np.asarray(
        [candidate["z"] for candidate in valid if candidate["confidence_rank"] >= SPECZ_CONFIDENCE_SECURE],
        dtype=float,
    )
    secure_delta = (
        float(np.nanmax(secure_redshifts) - np.nanmin(secure_redshifts))
        if secure_redshifts.size > 1
        else 0.0
    )
    if secure_delta > ZSPEC_CONFLICT_TOL:
        selection_note = "secure_conflict_requires_review"
    elif delta_max > ZSPEC_CONFLICT_TOL:
        selection_note = "candidate_conflict"
    elif best["source"] in {NED_SOURCE, SIMBAD_SOURCE}:
        selection_note = "external_probable_specz"
    elif best["confidence_rank"] == SPECZ_CONFIDENCE_FALLBACK:
        selection_note = "fallback_compiled_specz"
    elif best["confidence_rank"] <= SPECZ_CONFIDENCE_LOW:
        selection_note = "only_low_confidence_specz"
    else:
        selection_note = "selected_by_normalized_confidence"
    native_quality = best["native_quality_sort"]
    return {
        "zspec_best": float(best["z"]),
        "zspec_best_source": str(best["source"]),
        "zspec_best_quality": float(best["confidence_rank"]),
        "zspec_best_confidence": str(best["confidence"]),
        "zspec_best_confidence_rank": float(best["confidence_rank"]),
        "zspec_best_native_quality": float(native_quality) if np.isfinite(native_quality) else float("nan"),
        "zspec_best_native_quality_label": str(best.get("native_quality_label", "")),
        "zspec_selection_note": selection_note,
        "zspec_conflict": bool(delta_max > ZSPEC_CONFLICT_TOL),
        "zspec_delta_max": delta_max,
        "zspec_candidate_sources": _candidate_sources(str(candidate["source"]) for candidate in valid),
        "zspec_candidate_values": "|".join(f"{float(candidate['z']):.6g}" for candidate in valid),
    }


def _pagul_photoz(row: pd.Series | None) -> dict[str, Any]:
    result = {
        "zphot_best": float("nan"),
        "zphot_best_source": "",
        "pagul_chi2_red": float("nan"),
        "pagul_zpdf": float("nan"),
        "pagul_zpdf_low": float("nan"),
        "pagul_zpdf_high": float("nan"),
        "pagul_zsecond": float("nan"),
        "pagul_nb_used": float("nan"),
        "pagul_bitmask": float("nan"),
    }
    if row is None:
        return result

    zpdf = valid_redshift(row.get("ZPDF"))
    result.update(
        {
            "pagul_chi2_red": _valid_number(row.get("CHI2_RED")),
            "pagul_zpdf": zpdf,
            "pagul_zpdf_low": valid_redshift(row.get("ZPDF_LOW")),
            "pagul_zpdf_high": valid_redshift(row.get("ZPDF_HIGH")),
            "pagul_zsecond": valid_redshift(row.get("ZSECOND")),
            "pagul_nb_used": _valid_number(row.get("NB_USED")),
            "pagul_bitmask": _valid_number(row.get("BITMASK")),
        }
    )
    if np.isfinite(zpdf):
        result["zphot_best"] = zpdf
        result["zphot_best_source"] = "pagul2024_zpdf"
    return result


def _redshift_columns(
    *,
    pagul_row: pd.Series | None,
    shipley_row: pd.Series | None,
    ned_row: pd.Series | None = None,
    simbad_row: pd.Series | None = None,
    lagattuta22_row: pd.Series | None = None,
    ned_match_sep_arcsec: float = float("nan"),
    simbad_match_sep_arcsec: float = float("nan"),
    lagattuta22_match_sep_arcsec: float = float("nan"),
) -> dict[str, Any]:
    pagul_zspec = valid_redshift(pagul_row.get("ZSPEC")) if pagul_row is not None else float("nan")
    pagul_zspec_q = _valid_number(pagul_row.get("ZSPEC_Q")) if pagul_row is not None else float("nan")
    shipley_zspec = valid_redshift(shipley_row.get("zspec")) if shipley_row is not None else float("nan")
    ned_redshift_raw = valid_redshift(ned_row.get("redshift_raw")) if ned_row is not None else float("nan")
    ned_zspec = valid_redshift(ned_row.get("zspec")) if ned_row is not None else float("nan")
    simbad_redshift_raw = valid_redshift(simbad_row.get("redshift_raw")) if simbad_row is not None else float("nan")
    simbad_zspec = valid_redshift(simbad_row.get("zspec")) if simbad_row is not None else float("nan")
    lagattuta22_zspec = valid_redshift(lagattuta22_row.get("zspec")) if lagattuta22_row is not None else float("nan")
    lagattuta22_zspec_q = (
        _valid_number(lagattuta22_row.get("native_quality")) if lagattuta22_row is not None else float("nan")
    )

    candidates = [
        {
            "source": PAGUL2024_SOURCE,
            "z": pagul_zspec,
            "quality": pagul_zspec_q,
            "priority": SPECZ_SOURCE_PRIORITY[PAGUL2024_SOURCE],
        },
        {
            "source": LAGATTUTA22_SOURCE,
            "z": lagattuta22_zspec,
            "quality": lagattuta22_zspec_q,
            "priority": SPECZ_SOURCE_PRIORITY[LAGATTUTA22_SOURCE],
        },
        {
            "source": SHIPLEY2018_SOURCE,
            "z": shipley_zspec,
            "quality": 0.5,
            "priority": 1,
        },
        {
            "source": NED_SOURCE,
            "z": ned_zspec,
            "quality": ned_row.get("native_quality") if ned_row is not None else "",
            "priority": SPECZ_SOURCE_PRIORITY[NED_SOURCE],
        },
        {
            "source": SIMBAD_SOURCE,
            "z": simbad_zspec,
            "quality": simbad_row.get("native_quality") if simbad_row is not None else "",
            "priority": SPECZ_SOURCE_PRIORITY[SIMBAD_SOURCE],
        },
    ]
    result = {
        "pagul_zspec": pagul_zspec,
        "pagul_zspec_q": pagul_zspec_q,
        "pagul_zspec_ref": _normal_string(pagul_row.get("ZSPEC_REF")) if pagul_row is not None else "",
        "shipley_zspec": shipley_zspec,
        "shipley_zspec_ref": _normal_string(shipley_row.get("r_zspec")) if shipley_row is not None else "",
        "ned_zspec": ned_zspec,
        "ned_redshift_raw": ned_redshift_raw,
        "ned_zspec_quality": _normal_string(ned_row.get("native_quality")) if ned_row is not None else "",
        "ned_zspec_ref": _normal_string(ned_row.get("reference")) if ned_row is not None else "",
        "ned_zspec_match_sep_arcsec": ned_match_sep_arcsec,
        "ned_object_name": _normal_string(ned_row.get("external_id")) if ned_row is not None else "",
        "ned_object_type": _normal_string(ned_row.get("object_type")) if ned_row is not None else "",
        "ned_redshift_field": _normal_string(ned_row.get("field")) if ned_row is not None else "",
        "simbad_zspec": simbad_zspec,
        "simbad_redshift_raw": simbad_redshift_raw,
        "simbad_zspec_quality": _normal_string(simbad_row.get("native_quality")) if simbad_row is not None else "",
        "simbad_zspec_ref": _normal_string(simbad_row.get("reference")) if simbad_row is not None else "",
        "simbad_zspec_match_sep_arcsec": simbad_match_sep_arcsec,
        "simbad_main_id": _normal_string(simbad_row.get("external_id")) if simbad_row is not None else "",
        "simbad_otype": _normal_string(simbad_row.get("object_type")) if simbad_row is not None else "",
        "simbad_redshift_field": _normal_string(simbad_row.get("field")) if simbad_row is not None else "",
        "lagattuta22_zspec": lagattuta22_zspec,
        "lagattuta22_zspec_quality": lagattuta22_zspec_q,
        "lagattuta22_zspec_ref": _normal_string(lagattuta22_row.get("reference")) if lagattuta22_row is not None else "",
        "lagattuta22_zspec_match_sep_arcsec": lagattuta22_match_sep_arcsec,
        "lagattuta22_external_id": _normal_string(lagattuta22_row.get("external_id")) if lagattuta22_row is not None else "",
        "lagattuta22_field": _normal_string(lagattuta22_row.get("field")) if lagattuta22_row is not None else "",
        "lagattuta22_id_source": _normal_string(lagattuta22_row.get("id_source")) if lagattuta22_row is not None else "",
        "lagattuta22_multiple_image_id": (
            _normal_string(lagattuta22_row.get("multiple_image_id")) if lagattuta22_row is not None else ""
        ),
    }
    result.update(choose_best_zspec(candidates))
    return result


def _shipley_metadata(shipley_row: pd.Series | None) -> dict[str, Any]:
    result = {
        "shipley_id": "",
        "shipley_use": float("nan"),
        "shipley_s_g": float("nan"),
        "shipley_bandtot": "",
        "shipley_frad": float("nan"),
        "shipley_rad": float("nan"),
        "shipley_aimg": float("nan"),
        "shipley_bimg": float("nan"),
        "shipley_theta": float("nan"),
        "shipley_flf814w": float("nan"),
        "shipley_flf160w": float("nan"),
    }
    if shipley_row is None:
        return result
    result.update(
        {
            "shipley_id": _normal_string(shipley_row.get("ID")),
            "shipley_use": _valid_number(shipley_row.get("Use")),
            "shipley_s_g": _valid_number(shipley_row.get("S_G")),
            "shipley_bandtot": _normal_string(shipley_row.get("BandTot")),
            "shipley_frad": _valid_number(shipley_row.get("FRad")),
            "shipley_rad": _valid_number(shipley_row.get("Rad")),
            "shipley_aimg": _valid_number(shipley_row.get("Aimg")),
            "shipley_bimg": _valid_number(shipley_row.get("Bimg")),
            "shipley_theta": _valid_number(shipley_row.get("theta")),
            "shipley_flf814w": _valid_number(shipley_row.get("FlF814W")),
            "shipley_flf160w": _valid_number(shipley_row.get("FlF160W")),
        }
    )
    return result


def _magnitude_columns(
    *,
    pagul_row: pd.Series | None,
    shipley_row: pd.Series | None,
    pagul_flux_scale: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for band in BANDS:
        pagul_error_column = f"FLUXERR_{band.name}"
        shipley_error_column = f"e_{band.shipley_flux}"
        pagul_mag = (
            _valid_mag(pagul_flux_to_abmag(pagul_row.get(band.pagul_flux), pagul_flux_scale))
            if pagul_row is not None and band.pagul_flux in pagul_row.index
            else float("nan")
        )
        shipley_mag = (
            _valid_mag(flux_to_abmag_uJy(shipley_row.get(band.shipley_flux)))
            if shipley_row is not None and band.shipley_flux in shipley_row.index
            else float("nan")
        )
        pagul_fluxerr = (
            _valid_number(pagul_row.get(pagul_error_column))
            if pagul_row is not None and pagul_error_column in pagul_row.index
            else float("nan")
        )
        shipley_fluxerr = (
            _valid_number(shipley_row.get(shipley_error_column))
            if shipley_row is not None and shipley_error_column in shipley_row.index
            else float("nan")
        )
        result[f"pagul_mag_{band.name}"] = pagul_mag
        result[f"shipley_mag_{band.name}"] = shipley_mag
        result[f"mag_{band.name}"] = _first_finite((pagul_mag, shipley_mag))
        result[f"pagul_fluxerr_{band.name}"] = pagul_fluxerr
        result[f"shipley_fluxerr_{band.name}"] = shipley_fluxerr
        result[f"fluxerr_{band.name}"] = _first_finite((pagul_fluxerr, shipley_fluxerr))
    return result


def _build_output_row(
    *,
    spec: ClusterSpec,
    pagul_row: pd.Series | None,
    shipley_row: pd.Series | None,
    ned_row: pd.Series | None = None,
    simbad_row: pd.Series | None = None,
    lagattuta22_row: pd.Series | None = None,
    object_source: str,
    pagul_flux_scale: str,
    shipley_match_sep_arcsec: float = float("nan"),
    ned_match_sep_arcsec: float = float("nan"),
    simbad_match_sep_arcsec: float = float("nan"),
    lagattuta22_match_sep_arcsec: float = float("nan"),
) -> dict[str, Any]:
    sources = []
    if pagul_row is not None:
        sources.append(PAGUL2024_SOURCE)
    if shipley_row is not None:
        sources.append(SHIPLEY2018_SOURCE)
    if ned_row is not None:
        sources.append(NED_SOURCE)
    if simbad_row is not None:
        sources.append(SIMBAD_SOURCE)
    if lagattuta22_row is not None:
        sources.append(LAGATTUTA22_SOURCE)

    if pagul_row is not None:
        object_id = f"pagul2024:{_normal_string(pagul_row.get('ID'))}"
        ra = _valid_number(pagul_row.get("ALPHA_J2000_STACK"))
        dec = _valid_number(pagul_row.get("DELTA_J2000_STACK"))
        pagul_id = _normal_string(pagul_row.get("ID"))
    elif shipley_row is not None:
        object_id = f"shipley2018:{_normal_string(shipley_row.get('ID'))}"
        ra = _valid_number(shipley_row.get("RAJ2000"))
        dec = _valid_number(shipley_row.get("DEJ2000"))
        pagul_id = ""
    else:
        raise ValueError("At least one of pagul_row or shipley_row is required.")

    row: dict[str, Any] = {
        "cluster_key": spec.key,
        "object_id": object_id,
        "object_source": object_source,
        "catalog_sources": _candidate_sources(sources),
        "ra": ra,
        "dec": dec,
        "pagul_id": pagul_id,
        "shipley_match_sep_arcsec": shipley_match_sep_arcsec,
    }
    row.update(_shipley_metadata(shipley_row))
    row.update(_magnitude_columns(pagul_row=pagul_row, shipley_row=shipley_row, pagul_flux_scale=pagul_flux_scale))
    row.update(
        _redshift_columns(
            pagul_row=pagul_row,
            shipley_row=shipley_row,
            ned_row=ned_row,
            simbad_row=simbad_row,
            lagattuta22_row=lagattuta22_row,
            ned_match_sep_arcsec=ned_match_sep_arcsec,
            simbad_match_sep_arcsec=simbad_match_sep_arcsec,
            lagattuta22_match_sep_arcsec=lagattuta22_match_sep_arcsec,
        )
    )
    row.update(_pagul_photoz(pagul_row))

    zspec_best = valid_redshift(row["zspec_best"])
    zphot_best = valid_redshift(row["zphot_best"])
    row["member_zspec_candidate"] = bool(np.isfinite(zspec_best) and abs(zspec_best - spec.z_lens) <= MEMBER_Z_TOL)
    row["member_photoz_candidate"] = bool(np.isfinite(zphot_best) and abs(zphot_best - spec.z_lens) <= MEMBER_Z_TOL)
    return row


def build_cluster_catalog(
    *,
    spec: ClusterSpec,
    pagul: pd.DataFrame,
    shipley: pd.DataFrame,
    external_redshifts: dict[str, pd.DataFrame] | None = None,
    match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
    redshift_match_radius_arcsec: float = DEFAULT_REDSHIFT_MATCH_RADIUS_ARCSEC,
    progress: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pagul_cluster = _filter_pagul_cluster(pagul, spec)
    shipley_cluster = _filter_shipley_cluster(shipley, spec)
    external_redshifts = external_redshifts or {}
    pagul_flux_scale = infer_pagul_flux_scale(pagul_cluster)

    if progress is not None:
        progress.start_step(f"{spec.key}: matching sky positions", total=1)
    shipley_matches = one_to_one_sky_match(
        pagul_cluster,
        shipley_cluster,
        left_ra="ALPHA_J2000_STACK",
        left_dec="DELTA_J2000_STACK",
        right_ra="RAJ2000",
        right_dec="DEJ2000",
        radius_arcsec=match_radius_arcsec,
    )
    if progress is not None:
        progress.advance_step()
    if progress is not None:
        progress.finish_step()

    shipley_by_pagul = {
        int(row.left_index): (int(row.right_index), float(row.separation_arcsec))
        for row in shipley_matches.itertuples(index=False)
    }
    matched_shipley_indices = {right_index for right_index, _ in shipley_by_pagul.values()}

    row_entries: list[dict[str, Any]] = []
    for pagul_index, pagul_row in pagul_cluster.iterrows():
        shipley_index, shipley_sep = shipley_by_pagul.get(int(pagul_index), (None, float("nan")))
        shipley_row = shipley_cluster.loc[shipley_index] if shipley_index is not None else None
        row_entries.append(
            {
                "pagul_row": pagul_row,
                "shipley_row": shipley_row,
                "object_source": PAGUL2024_SOURCE,
                "shipley_match_sep_arcsec": shipley_sep,
                "ra": _valid_number(pagul_row.get("ALPHA_J2000_STACK")),
                "dec": _valid_number(pagul_row.get("DELTA_J2000_STACK")),
            }
        )

    for shipley_index, shipley_row in shipley_cluster.iterrows():
        if int(shipley_index) in matched_shipley_indices:
            continue
        row_entries.append(
            {
                "pagul_row": None,
                "shipley_row": shipley_row,
                "object_source": SHIPLEY2018_UNMATCHED_SOURCE,
                "shipley_match_sep_arcsec": float("nan"),
                "ra": _valid_number(shipley_row.get("RAJ2000")),
                "dec": _valid_number(shipley_row.get("DEJ2000")),
            }
        )

    master_positions = pd.DataFrame(
        {"ra": [entry["ra"] for entry in row_entries], "dec": [entry["dec"] for entry in row_entries]}
    )
    external_match_maps: dict[str, dict[int, tuple[int, float]]] = {}
    external_match_frames: dict[str, pd.DataFrame] = {}
    if progress is not None:
        progress.start_step(f"{spec.key}: matching external redshifts", total=len(EXTERNAL_REDSHIFT_SOURCES))
    for source in EXTERNAL_REDSHIFT_SOURCES:
        redshift_frame = external_redshifts.get(source, _empty_external_redshift_frame())
        external_match_frames[source] = redshift_frame
        matches = one_to_one_sky_match(
            master_positions,
            redshift_frame,
            left_ra="ra",
            left_dec="dec",
            right_ra="ra",
            right_dec="dec",
            radius_arcsec=redshift_match_radius_arcsec,
        )
        external_match_maps[source] = {
            int(row.left_index): (int(row.right_index), float(row.separation_arcsec))
            for row in matches.itertuples(index=False)
        }
        external_match_frames[f"{source}_matches"] = matches
        if progress is not None:
            progress.advance_step()
    if progress is not None:
        progress.finish_step()

    output_rows: list[dict[str, Any]] = []
    expected_output_rows = len(row_entries)
    if progress is not None:
        progress.start_step(f"{spec.key}: building master rows", total=expected_output_rows)
    for entry_index, entry in enumerate(row_entries):
        ned_index, ned_sep = external_match_maps.get(NED_SOURCE, {}).get(entry_index, (None, float("nan")))
        simbad_index, simbad_sep = external_match_maps.get(SIMBAD_SOURCE, {}).get(entry_index, (None, float("nan")))
        lagattuta22_index, lagattuta22_sep = external_match_maps.get(LAGATTUTA22_SOURCE, {}).get(
            entry_index, (None, float("nan"))
        )
        ned_row = (
            external_match_frames[NED_SOURCE].loc[ned_index]
            if ned_index is not None and not external_match_frames[NED_SOURCE].empty
            else None
        )
        simbad_row = (
            external_match_frames[SIMBAD_SOURCE].loc[simbad_index]
            if simbad_index is not None and not external_match_frames[SIMBAD_SOURCE].empty
            else None
        )
        lagattuta22_row = (
            external_match_frames[LAGATTUTA22_SOURCE].loc[lagattuta22_index]
            if lagattuta22_index is not None and not external_match_frames[LAGATTUTA22_SOURCE].empty
            else None
        )
        output_rows.append(
            _build_output_row(
                spec=spec,
                pagul_row=entry["pagul_row"],
                shipley_row=entry["shipley_row"],
                ned_row=ned_row,
                simbad_row=simbad_row,
                lagattuta22_row=lagattuta22_row,
                object_source=entry["object_source"],
                pagul_flux_scale=pagul_flux_scale,
                shipley_match_sep_arcsec=entry["shipley_match_sep_arcsec"],
                ned_match_sep_arcsec=ned_sep,
                simbad_match_sep_arcsec=simbad_sep,
                lagattuta22_match_sep_arcsec=lagattuta22_sep,
            )
        )
        if progress is not None:
            progress.advance_step()
    if progress is not None:
        progress.finish_step()

    audit_rows: list[dict[str, Any]] = []
    for row in shipley_matches.itertuples(index=False):
        audit_rows.append(
            {
                "cluster_key": spec.key,
                "match_type": "pagul2024_shipley2018",
                "left_id": _normal_string(pagul_cluster.loc[int(row.left_index), "ID"]),
                "right_id": _normal_string(shipley_cluster.loc[int(row.right_index), "ID"]),
                "separation_arcsec": float(row.separation_arcsec),
            }
        )
    catalog = pd.DataFrame(output_rows)
    for source in EXTERNAL_REDSHIFT_SOURCES:
        redshift_frame = external_match_frames[source]
        matches = external_match_frames.get(f"{source}_matches", pd.DataFrame())
        for row in matches.itertuples(index=False):
            catalog_row = catalog.iloc[int(row.left_index)] if not catalog.empty else pd.Series(dtype=object)
            right = redshift_frame.loc[int(row.right_index)]
            audit_rows.append(
                {
                    "cluster_key": spec.key,
                    "match_type": f"master_{source}_redshift",
                    "left_id": _normal_string(catalog_row.get("object_id")),
                    "right_id": _normal_string(right.get("external_id")),
                    "separation_arcsec": float(row.separation_arcsec),
                }
            )

    external_sources = "|".join(source for source in EXTERNAL_REDSHIFT_SOURCES if not external_match_frames[source].empty)
    external_conflicts = 0
    if not catalog.empty and "zspec_candidate_sources" in catalog.columns:
        external_pattern = "|".join(EXTERNAL_REDSHIFT_SOURCES)
        has_external_candidate = catalog["zspec_candidate_sources"].fillna("").astype(str).str.contains(external_pattern)
        conflict = (
            catalog["zspec_conflict"].map(_bool_value)
            if "zspec_conflict" in catalog.columns
            else pd.Series(False, index=catalog.index)
        )
        external_conflicts = int((has_external_candidate & conflict).sum())

    manifest = {
        "cluster_key": spec.key,
        "pagul_flux_scale": pagul_flux_scale,
        "match_radius_arcsec": match_radius_arcsec,
        "redshift_match_radius_arcsec": redshift_match_radius_arcsec,
        "n_pagul_rows": int(len(pagul_cluster)),
        "n_shipley_rows": int(len(shipley_cluster)),
        "n_shipley_matched": int(len(shipley_matches)),
        "n_shipley_unmatched_appended": int(len(shipley_cluster) - len(matched_shipley_indices)),
        "n_ned_redshift_rows": int(len(external_match_frames[NED_SOURCE])),
        "n_simbad_redshift_rows": int(len(external_match_frames[SIMBAD_SOURCE])),
        "n_lagattuta22_redshift_rows": int(len(external_match_frames[LAGATTUTA22_SOURCE])),
        "n_ned_redshift_matched": int(len(external_match_frames.get("ned_matches", pd.DataFrame()))),
        "n_simbad_redshift_matched": int(len(external_match_frames.get("simbad_matches", pd.DataFrame()))),
        "n_lagattuta22_redshift_matched": int(
            len(external_match_frames.get("lagattuta22_matches", pd.DataFrame()))
        ),
        "n_ned_zspec_best": int((catalog.get("zspec_best_source", pd.Series(dtype=str)) == NED_SOURCE).sum()),
        "n_simbad_zspec_best": int((catalog.get("zspec_best_source", pd.Series(dtype=str)) == SIMBAD_SOURCE).sum()),
        "n_lagattuta22_zspec_best": int(
            (catalog.get("zspec_best_source", pd.Series(dtype=str)) == LAGATTUTA22_SOURCE).sum()
        ),
        "n_external_zspec_conflicts": external_conflicts,
        "external_redshift_sources": external_sources,
        "n_output_rows": int(len(output_rows)),
    }
    audit = pd.DataFrame(audit_rows)
    return catalog, audit, manifest


def _bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _object_id(row: pd.Series) -> str:
    value = row.get("object_id", "")
    if pd.isna(value) or str(value).strip() == "":
        return str(row.name)
    return str(value)


def kpc_per_arcsec(z_lens: float) -> float:
    return float(COSMOLOGY.kpc_proper_per_arcmin(float(z_lens)).to(u.kpc / u.arcsec).value)


def projected_radius_kpc(ra: pd.Series, dec: pd.Series, center_ra: float, center_dec: float, z_lens: float) -> pd.Series:
    ra_values = pd.to_numeric(ra, errors="coerce")
    dec_values = pd.to_numeric(dec, errors="coerce")
    result = pd.Series(np.nan, index=ra.index, dtype=float)
    valid = ra_values.notna() & dec_values.notna() & np.isfinite(ra_values) & np.isfinite(dec_values)
    if not valid.any() or not np.isfinite(center_ra) or not np.isfinite(center_dec):
        return result
    coords = SkyCoord(ra_values[valid].to_numpy(dtype=float) * u.deg, dec_values[valid].to_numpy(dtype=float) * u.deg)
    center = SkyCoord(float(center_ra) * u.deg, float(center_dec) * u.deg)
    result.loc[valid] = coords.separation(center).arcsec * kpc_per_arcsec(z_lens)
    return result


def max_span_arcsec(spec: ClusterSpec, max_family_span_kpc: float) -> float:
    return float(max_family_span_kpc) / kpc_per_arcsec(spec.z_lens)


def delta_v_kms(z: Any, z_lens: float) -> float:
    redshift = valid_redshift(z)
    if not np.isfinite(redshift):
        return float("nan")
    return float(C_KMS * (redshift - float(z_lens)) / (1.0 + float(z_lens)))


def _robust_sigma(values: np.ndarray, *, floor: float = 0.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(finite)) if finite.size > 1 else floor
    return float(max(sigma, floor))


def member_velocity_window(catalog: pd.DataFrame, spec: ClusterSpec) -> tuple[float, float, int]:
    if catalog.empty or "zspec_best" not in catalog.columns:
        return 1200.0, 3000.0, 0
    zspec = pd.to_numeric(catalog["zspec_best"], errors="coerce").map(valid_redshift)
    rank = (
        pd.to_numeric(catalog["zspec_best_confidence_rank"], errors="coerce")
        if "zspec_best_confidence_rank" in catalog.columns
        else pd.Series(np.nan, index=catalog.index)
    )
    velocities = zspec.map(lambda value: delta_v_kms(value, spec.z_lens))
    seed = velocities[np.isfinite(velocities) & (rank >= SECURE_PROBABLE_SPEC_RANK) & (np.abs(velocities) <= 9000.0)]
    if len(seed) >= 5:
        sigma = _robust_sigma(seed.to_numpy(dtype=float), floor=800.0)
    else:
        sigma = 1200.0
    sigma = float(np.clip(sigma, 800.0, 2400.0))
    return sigma, float(np.clip(2.5 * sigma, 3000.0, 6000.0)), int(len(seed))


def _fit_red_sequence_plane(
    catalog: pd.DataFrame,
    seed_mask: pd.Series,
    *,
    blue_band: str,
    red_band: str,
    mag_band: str,
) -> dict[str, Any] | None:
    blue_col = f"mag_{blue_band}"
    red_col = f"mag_{red_band}"
    mag_col = f"mag_{mag_band}"
    missing = [column for column in (blue_col, red_col, mag_col) if column not in catalog.columns]
    if missing:
        return None
    blue = pd.to_numeric(catalog[blue_col], errors="coerce").map(_valid_mag)
    red = pd.to_numeric(catalog[red_col], errors="coerce").map(_valid_mag)
    mag = pd.to_numeric(catalog[mag_col], errors="coerce").map(_valid_mag)
    valid = seed_mask & np.isfinite(blue) & np.isfinite(red) & np.isfinite(mag)
    if int(valid.sum()) < 3:
        return None

    x = mag.loc[valid].to_numpy(dtype=float)
    y = (blue - red).loc[valid].to_numpy(dtype=float)

    def robust_line(values_x: np.ndarray, values_y: np.ndarray) -> tuple[float, float]:
        n_unique_mag = len(np.unique(np.round(values_x, decimals=3)))
        if float(np.nanmax(values_x) - np.nanmin(values_x)) < 1.0e-6 or n_unique_mag < 3:
            fitted_slope = 0.0
        else:
            slopes: list[np.ndarray] = []
            for index in range(len(values_x) - 1):
                dx = values_x[index + 1 :] - values_x[index]
                valid_dx = np.abs(dx) > 1.0e-6
                if np.any(valid_dx):
                    slopes.append((values_y[index + 1 :][valid_dx] - values_y[index]) / dx[valid_dx])
            fitted_slope = float(np.median(np.concatenate(slopes))) if slopes else 0.0
        fitted_intercept = float(np.median(values_y - fitted_slope * values_x))
        return fitted_slope, fitted_intercept

    keep = np.ones_like(x, dtype=bool)
    slope = 0.0
    intercept = float(np.median(y))
    scatter = 0.08
    for _ in range(4):
        if int(keep.sum()) < 3:
            break
        slope, intercept = robust_line(x[keep], y[keep])
        residuals = y - (slope * x + intercept)
        center = float(np.median(residuals[keep]))
        scatter = max(1.4826 * float(np.median(np.abs(residuals[keep] - center))), 0.05)
        keep = np.abs(residuals - center) <= max(3.0 * scatter, 0.15)
    if int(keep.sum()) < 3:
        return None
    residuals = y[keep] - (slope * x[keep] + intercept)
    center = float(np.median(residuals))
    scatter = max(1.4826 * float(np.median(np.abs(residuals - center))), 0.05)
    return {
        "blue_band": blue_band,
        "red_band": red_band,
        "mag_band": mag_band,
        "color_name": f"{blue_band}-{red_band}",
        "slope": float(slope),
        "intercept": float(intercept),
        "scatter_mag": float(scatter),
        "n_seed": int(valid.sum()),
        "n_used": int(keep.sum()),
    }


def fit_cluster_red_sequence(catalog: pd.DataFrame, spec: ClusterSpec, velocity_window_kms: float) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame()
    zspec = (
        pd.to_numeric(catalog["zspec_best"], errors="coerce").map(valid_redshift)
        if "zspec_best" in catalog.columns
        else pd.Series(np.nan, index=catalog.index)
    )
    rank = (
        pd.to_numeric(catalog["zspec_best_confidence_rank"], errors="coerce")
        if "zspec_best_confidence_rank" in catalog.columns
        else pd.Series(np.nan, index=catalog.index)
    )
    velocities = zspec.map(lambda value: delta_v_kms(value, spec.z_lens))
    seed_mask = np.isfinite(velocities) & (np.abs(velocities) <= velocity_window_kms) & (rank >= SECURE_PROBABLE_SPEC_RANK)
    rows = []
    for blue_band, red_band, mag_band in RED_SEQUENCE_PLANES:
        fit = _fit_red_sequence_plane(catalog, seed_mask, blue_band=blue_band, red_band=red_band, mag_band=mag_band)
        if fit is not None:
            fit["cluster_key"] = spec.key
            rows.append(fit)
    return pd.DataFrame(rows)


def red_sequence_scores(catalog: pd.DataFrame, red_sequence: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=catalog.index)
    result["red_sequence_n_planes"] = 0
    result["red_sequence_n_consistent"] = 0
    result["red_sequence_score"] = 0.0
    result["red_sequence_min_abs_sigma"] = np.nan
    result["red_sequence_median_abs_sigma"] = np.nan
    if catalog.empty or red_sequence.empty:
        return result

    normalized_residuals: list[pd.Series] = []
    for fit in red_sequence.itertuples(index=False):
        blue_col = f"mag_{fit.blue_band}"
        red_col = f"mag_{fit.red_band}"
        mag_col = f"mag_{fit.mag_band}"
        if blue_col not in catalog.columns or red_col not in catalog.columns or mag_col not in catalog.columns:
            continue
        blue = pd.to_numeric(catalog[blue_col], errors="coerce").map(_valid_mag)
        red = pd.to_numeric(catalog[red_col], errors="coerce").map(_valid_mag)
        mag = pd.to_numeric(catalog[mag_col], errors="coerce").map(_valid_mag)
        color = blue - red
        expected = float(fit.slope) * mag + float(fit.intercept)
        scatter = max(_to_float(fit.scatter_mag), 0.05)
        normalized_residuals.append(((color - expected).abs() / scatter).rename(fit.color_name))
    if not normalized_residuals:
        return result

    residual_frame = pd.concat(normalized_residuals, axis=1)
    finite_counts = residual_frame.notna().sum(axis=1).astype(int)
    consistent_counts = (residual_frame <= 3.0).sum(axis=1).astype(int)
    score_frame = np.exp(-0.5 * np.square(residual_frame.clip(upper=8.0) / 2.0))
    score = score_frame.mean(axis=1).fillna(0.0)
    result["red_sequence_n_planes"] = finite_counts
    result["red_sequence_n_consistent"] = consistent_counts
    result["red_sequence_score"] = score.clip(lower=0.0, upper=1.0)
    result["red_sequence_min_abs_sigma"] = residual_frame.min(axis=1)
    result["red_sequence_median_abs_sigma"] = residual_frame.median(axis=1)
    return result


def photoz_member_score(row: pd.Series, spec: ClusterSpec) -> tuple[float, str]:
    zphot = valid_redshift(row.get("zphot_best"))
    if not np.isfinite(zphot):
        return 0.0, "no_photoz"
    sigma = max(0.05, 0.08 * (1.0 + spec.z_lens))
    return float(0.60 * math.exp(-0.5 * ((zphot - spec.z_lens) / sigma) ** 2)), "photoz_best_offset"


def _cluster_member_class(probability: float, reasons: list[str], *, selected: bool) -> str:
    if "secure_spec_member" in reasons:
        return "secure_spec_member"
    if "probable_spec_member" in reasons:
        return "probable_spec_member"
    if "tentative_spec_member" in reasons:
        return "tentative_spec_member"
    if "fallback_spec_member" in reasons:
        return "fallback_spec_member"
    if "rejected_secure_specz_nonmember" in reasons:
        return "rejected_foreground_background"
    if selected and "red_sequence_candidate" in reasons and "photoz_member_candidate" in reasons:
        return "red_sequence_photoz_member"
    if selected and "red_sequence_candidate" in reasons:
        return "red_sequence_member"
    if selected and "photoz_member_candidate" in reasons:
        return "photoz_member"
    if probability > 0.0:
        return "low_probability_member"
    return "non_member"


def score_cluster_members(
    catalog: pd.DataFrame,
    spec: ClusterSpec,
    *,
    member_probability_threshold: float = DEFAULT_MEMBER_PROBABILITY_THRESHOLD,
    lensing_member_probability_threshold: float = DEFAULT_LENSING_MEMBER_PROBABILITY_THRESHOLD,
    lensing_bright_mag_f160w: float = DEFAULT_LENSING_BRIGHT_MAG_F160W,
    member_faint_mag_f814w: float = DEFAULT_MEMBER_FAINT_MAG_F814W,
    bcg_special_max: int = DEFAULT_BCG_SPECIAL_MAX,
    progress: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    working = catalog.copy()
    if working.empty:
        red_sequence = pd.DataFrame()
        manifest = {
            "cluster_key": spec.key,
            "z_lens": spec.z_lens,
            "n_master_rows": 0,
            "n_cluster_members": 0,
            "n_lensing_members": 0,
            "n_bcg_special_candidates": 0,
            "member_velocity_sigma_kms": 1200.0,
            "member_velocity_window_kms": 3000.0,
            "n_velocity_seed_members": 0,
            "n_red_sequence_planes": 0,
            "member_probability_threshold": member_probability_threshold,
            "lensing_member_probability_threshold": lensing_member_probability_threshold,
            "lensing_bright_mag_f160w": lensing_bright_mag_f160w,
            "member_bcg_mag_f814w": float("nan"),
            "member_faint_mag_f814w": member_faint_mag_f814w,
            "n_member_f814w_window_rejected": 0,
            "n_member_f814w_window_selected": 0,
        }
        return working, red_sequence, manifest

    for column in ("ra", "dec", "zspec_best", "zphot_best", "zspec_best_confidence_rank", "mag_F814W", "mag_F160W"):
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")
    for column in ("zspec_selection_note", "zspec_best_confidence", "object_id", "object_source", "catalog_sources"):
        if column not in working.columns:
            working[column] = ""

    if progress is not None:
        progress.start_step(f"{spec.key}: fitting member red sequence", total=None)
    sigma_v, velocity_window, n_velocity_seed = member_velocity_window(working, spec)
    red_sequence = fit_cluster_red_sequence(working, spec, velocity_window)
    red_scores = red_sequence_scores(working, red_sequence)
    working = pd.concat([working, red_scores], axis=1)
    if progress is not None:
        progress.finish_step()

    zspec_values = _valid_redshift_array(working["zspec_best"])
    rank_values = pd.to_numeric(working["zspec_best_confidence_rank"], errors="coerce").to_numpy(dtype=float)
    delta_v_values = C_KMS * (zspec_values - spec.z_lens) / (1.0 + spec.z_lens)
    delta_v_values[~np.isfinite(zspec_values)] = np.nan
    working["member_delta_v_kms"] = delta_v_values
    working["member_velocity_window_kms"] = velocity_window

    if progress is not None:
        progress.start_step(f"{spec.key}: scoring member rows", total=len(working))
    n_rows = len(working)
    red_score_values = np.clip(pd.to_numeric(working["red_sequence_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
    n_planes_values = pd.to_numeric(working["red_sequence_n_planes"], errors="coerce").fillna(0).to_numpy(dtype=int)
    n_consistent_values = pd.to_numeric(working["red_sequence_n_consistent"], errors="coerce").fillna(0).to_numpy(dtype=int)
    required_planes = np.minimum(2, n_planes_values)
    red_sequence_candidate = (n_planes_values > 0) & (n_consistent_values >= np.maximum(1, required_planes))

    zphot_values = _valid_redshift_array(working["zphot_best"])
    sigma_photo = max(0.05, 0.08 * (1.0 + spec.z_lens))
    has_photo = np.isfinite(zphot_values)
    photo_scores = np.zeros(n_rows, dtype=float)
    photo_notes = np.full(n_rows, "no_photoz", dtype=object)
    photo_scores[has_photo] = 0.60 * np.exp(-0.5 * ((zphot_values[has_photo] - spec.z_lens) / sigma_photo) ** 2)
    photo_notes[has_photo] = "photoz_best_offset"

    has_spec = np.isfinite(delta_v_values) & np.isfinite(rank_values) & (rank_values > SPECZ_CONFIDENCE_LOW)
    inside_velocity = has_spec & (np.abs(delta_v_values) <= velocity_window)
    strong_spec = has_spec & (rank_values >= SECURE_PROBABLE_SPEC_RANK)
    selection_note_values = working["zspec_selection_note"].astype(str).to_numpy()
    conflict_note = np.char.find(selection_note_values.astype(str), "conflict") >= 0
    hard_reject = strong_spec & ~inside_velocity & ~conflict_note
    near_low_confidence = has_spec & ~inside_velocity & ~hard_reject & (np.abs(delta_v_values) <= 1.5 * velocity_window)

    spec_scores = np.zeros(n_rows, dtype=float)
    spec_reason = np.full(n_rows, "", dtype=object)
    secure_member = inside_velocity & (rank_values >= SPECZ_CONFIDENCE_SECURE)
    probable_member = inside_velocity & ~secure_member & (rank_values >= SPECZ_CONFIDENCE_PROBABLE)
    tentative_member = inside_velocity & ~secure_member & ~probable_member & (rank_values >= SPECZ_CONFIDENCE_TENTATIVE)
    fallback_member = (
        inside_velocity
        & ~secure_member
        & ~probable_member
        & ~tentative_member
        & (rank_values >= SPECZ_CONFIDENCE_FALLBACK)
    )
    spec_scores[secure_member] = 0.98
    spec_scores[probable_member] = 0.93
    spec_scores[tentative_member] = 0.72
    spec_scores[fallback_member] = 0.62
    spec_scores[near_low_confidence] = 0.20
    spec_reason[secure_member] = "secure_spec_member"
    spec_reason[probable_member] = "probable_spec_member"
    spec_reason[tentative_member] = "tentative_spec_member"
    spec_reason[fallback_member] = "fallback_spec_member"
    spec_reason[hard_reject] = "rejected_secure_specz_nonmember"
    spec_reason[near_low_confidence] = "specz_near_cluster_low_confidence"

    mag_f814w_values = _valid_mag_array(working["mag_F814W"])
    red_probability = np.zeros(n_rows, dtype=float)
    base_red_probability = np.minimum(0.68, 0.35 + 0.35 * red_score_values + 0.05 * np.minimum(n_consistent_values, 3))
    red_probability[red_sequence_candidate] = base_red_probability[red_sequence_candidate]
    low_photo_sparse_red = red_sequence_candidate & (photo_scores < 0.35) & (n_planes_values < 4)
    high_photo_one_plane = red_sequence_candidate & (photo_scores >= 0.35) & (n_planes_values < 2)
    red_probability[low_photo_sparse_red | high_photo_one_plane] = np.minimum(
        red_probability[low_photo_sparse_red | high_photo_one_plane],
        member_probability_threshold - 0.01,
    )
    no_spec_probability_branch = (~hard_reject) & (spec_scores <= 0.0)
    bright_red_reason = (
        no_spec_probability_branch
        & red_sequence_candidate
        & np.isfinite(mag_f814w_values)
        & (mag_f814w_values <= float(member_faint_mag_f814w))
        & (n_planes_values >= 4)
    )
    red_probability[bright_red_reason] = np.minimum(0.72, red_probability[bright_red_reason] + 0.07)

    probabilities = np.zeros(n_rows, dtype=float)
    high_spec_probability = (~hard_reject) & (spec_scores >= 0.90)
    mid_spec_probability = (~hard_reject) & (spec_scores > 0.0) & ~high_spec_probability
    probabilities[high_spec_probability] = spec_scores[high_spec_probability]
    mid_red_component = np.where(red_sequence_candidate, 0.70 * red_score_values, 0.0)
    probabilities[mid_spec_probability] = np.minimum(
        np.maximum.reduce(
            [
                spec_scores[mid_spec_probability],
                0.65 * photo_scores[mid_spec_probability],
                mid_red_component[mid_spec_probability],
            ]
        ),
        0.78,
    )
    no_spec_photo_red = no_spec_probability_branch & (photo_scores > 0.0) & (red_probability > 0.0)
    probabilities[no_spec_photo_red] = np.minimum(
        0.60,
        np.maximum.reduce(
            [
                photo_scores[no_spec_photo_red],
                red_probability[no_spec_photo_red],
                0.55 * photo_scores[no_spec_photo_red] + 0.55 * red_probability[no_spec_photo_red],
            ]
        ),
    )
    no_spec_photo_only = no_spec_probability_branch & (photo_scores > 0.0) & (red_probability <= 0.0)
    probabilities[no_spec_photo_only] = np.minimum(member_probability_threshold - 0.01, photo_scores[no_spec_photo_only])
    no_spec_red_only = no_spec_probability_branch & (photo_scores <= 0.0)
    probabilities[no_spec_red_only] = red_probability[no_spec_red_only]
    probabilities = np.clip(probabilities, 0.0, 1.0)

    reliable_spec_member = inside_velocity & has_spec & (rank_values >= SPECZ_CONFIDENCE_FALLBACK)
    non_spec_evidence = (~has_spec) | (rank_values < SPECZ_CONFIDENCE_FALLBACK)
    red_photo_member = (
        non_spec_evidence
        & red_sequence_candidate
        & (n_planes_values >= 2)
        & (n_consistent_values >= 2)
        & (red_score_values >= 0.55)
        & (photo_scores >= 0.45)
    )
    bright_red_member = (
        non_spec_evidence
        & red_sequence_candidate
        & (n_planes_values >= 4)
        & (n_consistent_values >= 3)
        & (red_score_values >= 0.75)
        & np.isfinite(mag_f814w_values)
        & (mag_f814w_values <= float(member_faint_mag_f814w))
    )
    selected_flags = (~hard_reject) & (reliable_spec_member | red_photo_member | bright_red_member)
    selection_evidence = np.full(n_rows, "", dtype=object)
    selection_evidence[bright_red_member] = "bright_red_sequence"
    selection_evidence[red_photo_member] = "red_sequence_photoz"
    selection_evidence[reliable_spec_member] = "specz_velocity"

    preliminary_selected_flags = selected_flags.copy()
    preliminary_bright_red_lensing = (
        np.isfinite(mag_f814w_values)
        & (mag_f814w_values <= float(member_faint_mag_f814w))
        & (probabilities >= 0.65)
        & (n_consistent_values >= 1)
    )
    preliminary_lensing_flags = (probabilities >= float(lensing_member_probability_threshold)) | preliminary_bright_red_lensing
    preliminary_member_pool = preliminary_selected_flags | preliminary_lensing_flags
    bcg_pool_mags = mag_f814w_values[preliminary_member_pool & np.isfinite(mag_f814w_values)]
    member_bcg_mag_f814w = float(np.nanmin(bcg_pool_mags)) if bcg_pool_mags.size else float("nan")
    member_f814w_window = np.isfinite(mag_f814w_values) & (mag_f814w_values <= float(member_faint_mag_f814w))
    if np.isfinite(member_bcg_mag_f814w):
        member_f814w_window &= mag_f814w_values >= member_bcg_mag_f814w

    member_f814w_reject_reason = np.full(n_rows, "", dtype=object)
    f814w_window_rejected = preliminary_member_pool & ~member_f814w_window
    missing_f814w = f814w_window_rejected & ~np.isfinite(mag_f814w_values)
    too_faint_f814w = f814w_window_rejected & np.isfinite(mag_f814w_values) & (mag_f814w_values > float(member_faint_mag_f814w))
    brighter_than_bcg = (
        f814w_window_rejected
        & np.isfinite(mag_f814w_values)
        & np.isfinite(member_bcg_mag_f814w)
        & (mag_f814w_values < member_bcg_mag_f814w)
    )
    member_f814w_reject_reason[missing_f814w] = "missing_or_invalid_f814w"
    member_f814w_reject_reason[too_faint_f814w] = "too_faint_f814w"
    member_f814w_reject_reason[brighter_than_bcg] = "brighter_than_bcg_f814w"
    selected_flags = selected_flags & member_f814w_window

    blue_or_off_sequence_spec = high_spec_probability & ~red_sequence_candidate & (n_planes_values > 0)
    notes: list[str] = []
    classes: list[str] = []
    for idx in range(n_rows):
        reasons: list[str] = []
        if red_sequence_candidate[idx]:
            reasons.append("red_sequence_candidate")
        if photo_scores[idx] >= 0.35:
            reasons.append("photoz_member_candidate")
        if photo_notes[idx]:
            reasons.append(str(photo_notes[idx]))
        if spec_reason[idx]:
            reasons.append(str(spec_reason[idx]))
        if blue_or_off_sequence_spec[idx]:
            reasons.append("blue_or_off_sequence_spec_member")
        if bright_red_reason[idx]:
            reasons.append("bright_red_sequence_candidate")
        if member_f814w_reject_reason[idx]:
            reasons.append(str(member_f814w_reject_reason[idx]))
        deduped = list(dict.fromkeys(reasons))
        notes.append("|".join(deduped))
        classes.append(_cluster_member_class(float(probabilities[idx]), deduped, selected=bool(selected_flags[idx])))
    if progress is not None:
        progress.advance_step(n_rows)
    if progress is not None:
        progress.finish_step()

    working["member_probability"] = probabilities.tolist()
    working["member_class"] = classes
    working["member_selection_note"] = notes
    working["member_specz_score"] = spec_scores.tolist()
    working["member_photoz_score"] = photo_scores.tolist()
    working["member_red_sequence_score"] = red_score_values.tolist()
    working["member_selection_evidence"] = selection_evidence.tolist()
    working["member_bcg_mag_f814w"] = member_bcg_mag_f814w
    working["member_faint_mag_f814w"] = float(member_faint_mag_f814w)
    working["member_f814w_window_selected"] = member_f814w_window.tolist()
    working["member_f814w_reject_reason"] = member_f814w_reject_reason.tolist()
    working["cluster_member_selected"] = selected_flags.tolist()

    selected_for_center = working.loc[(working["member_probability"] >= 0.80) & working["member_f814w_window_selected"].map(_bool_value)].copy()
    if selected_for_center.empty:
        selected_for_center = working.loc[working["cluster_member_selected"]].copy()
    center_ra = float(pd.to_numeric(selected_for_center["ra"], errors="coerce").median()) if not selected_for_center.empty else float("nan")
    center_dec = float(pd.to_numeric(selected_for_center["dec"], errors="coerce").median()) if not selected_for_center.empty else float("nan")
    working["clustercentric_radius_kpc"] = projected_radius_kpc(working["ra"], working["dec"], center_ra, center_dec, spec.z_lens)

    mag_f814w = working["mag_F814W"].map(_valid_mag)
    bright_red_lensing = (
        (mag_f814w <= float(member_faint_mag_f814w))
        & (working["member_probability"] >= 0.65)
        & (working["red_sequence_n_consistent"] >= 1)
    )
    working["member_for_lensing"] = working["member_f814w_window_selected"].map(_bool_value) & (
        (working["member_probability"] >= float(lensing_member_probability_threshold)) | bright_red_lensing
    )
    working["bcg_special_member_candidate"] = False
    special_pool = working.loc[
        working["member_for_lensing"]
        & np.isfinite(mag_f814w)
        & (
            (working["clustercentric_radius_kpc"] <= DEFAULT_BCG_SPECIAL_RADIUS_KPC)
            | (mag_f814w <= min(float(member_faint_mag_f814w), 21.5))
        )
    ].copy()
    if not special_pool.empty and bcg_special_max > 0:
        special_indices = special_pool.sort_values(
            ["mag_F814W", "clustercentric_radius_kpc", "member_probability"],
            ascending=[True, True, False],
        ).head(int(bcg_special_max)).index
        working.loc[special_indices, "bcg_special_member_candidate"] = True

    member_f814w_window_rejected = preliminary_member_pool & ~member_f814w_window
    member_f814w_window_kept = preliminary_member_pool & member_f814w_window
    manifest = {
        "cluster_key": spec.key,
        "z_lens": spec.z_lens,
        "n_master_rows": int(len(working)),
        "n_cluster_members": int(working["cluster_member_selected"].sum()),
        "n_lensing_members": int(working["member_for_lensing"].sum()),
        "n_bcg_special_candidates": int(working["bcg_special_member_candidate"].sum()),
        "member_velocity_sigma_kms": float(sigma_v),
        "member_velocity_window_kms": float(velocity_window),
        "n_velocity_seed_members": int(n_velocity_seed),
        "n_red_sequence_planes": int(len(red_sequence)),
        "member_probability_threshold": float(member_probability_threshold),
        "lensing_member_probability_threshold": float(lensing_member_probability_threshold),
        "lensing_bright_mag_f160w": float(lensing_bright_mag_f160w),
        "member_bcg_mag_f814w": float(member_bcg_mag_f814w),
        "member_faint_mag_f814w": float(member_faint_mag_f814w),
        "n_member_f814w_window_rejected": int(member_f814w_window_rejected.sum()),
        "n_member_f814w_window_selected": int(member_f814w_window_kept.sum()),
        "cluster_center_ra": center_ra,
        "cluster_center_dec": center_dec,
    }
    return working, red_sequence, manifest


def _confidence_label(row: pd.Series) -> str:
    value = row.get("zspec_best_confidence", "")
    if pd.isna(value):
        return ""
    return str(value)


def _numeric_catalog_column(catalog: pd.DataFrame, column: str) -> pd.Series:
    if column not in catalog.columns:
        return pd.Series(np.nan, index=catalog.index, dtype=float)
    return pd.to_numeric(catalog[column], errors="coerce")


def _image_hff_footprint_mask(catalog: pd.DataFrame) -> pd.Series:
    for column in ("image_hff_footprint", "in_hff_footprint", "hff_footprint", "inside_hff_footprint"):
        if column not in catalog.columns:
            continue

        def _footprint_value(value: Any) -> bool:
            if value is None or pd.isna(value):
                return True
            text = str(value).strip().lower()
            if text in {"", "nan", "none", "unknown"}:
                return True
            return text in {"1", "true", "t", "yes", "y", "hff", "core", "inside"}

        return catalog[column].map(_footprint_value).astype(bool)
    return pd.Series(True, index=catalog.index, dtype=bool)


def _image_size_arcsec(catalog: pd.DataFrame, pixel_scale_arcsec: float) -> pd.Series:
    size_pix = _numeric_catalog_column(catalog, "shipley_frad").copy()
    missing_size = ~np.isfinite(size_pix)
    if "shipley_rad" in catalog.columns:
        rad = _numeric_catalog_column(catalog, "shipley_rad")
        size_pix.loc[missing_size & np.isfinite(rad)] = rad.loc[missing_size & np.isfinite(rad)]
        missing_size = ~np.isfinite(size_pix)
    if {"shipley_aimg", "shipley_bimg"}.issubset(catalog.columns):
        aimg = _numeric_catalog_column(catalog, "shipley_aimg")
        bimg = _numeric_catalog_column(catalog, "shipley_bimg")
        valid_axes = missing_size & np.isfinite(aimg) & np.isfinite(bimg) & (aimg > 0.0) & (bimg > 0.0)
        size_pix.loc[valid_axes] = np.sqrt(aimg.loc[valid_axes] * bimg.loc[valid_axes])
    return size_pix * float(pixel_scale_arcsec)


def _image_ellipticity(catalog: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    aimg = _numeric_catalog_column(catalog, "shipley_aimg")
    bimg = _numeric_catalog_column(catalog, "shipley_bimg")
    shape_available = np.isfinite(aimg) & np.isfinite(bimg)
    ellipticity = np.full(len(catalog), np.nan, dtype=float)
    if len(catalog) == 0:
        return pd.Series(ellipticity, index=catalog.index), pd.Series(False, index=catalog.index, dtype=bool)

    a_values = aimg.to_numpy(dtype=float)
    b_values = bimg.to_numpy(dtype=float)
    major = np.maximum(a_values, b_values)
    minor = np.minimum(a_values, b_values)
    valid_axes = shape_available.to_numpy(dtype=bool) & (major > 0.0) & (minor > 0.0)
    denominator = np.square(major) + np.square(minor)
    valid_axes &= denominator > 0.0
    ellipticity[valid_axes] = np.sqrt(
        np.clip((np.square(major[valid_axes]) - np.square(minor[valid_axes])) / denominator[valid_axes], 0.0, 1.0)
    )
    return pd.Series(ellipticity, index=catalog.index), shape_available.astype(bool)


def _positive_f814w_error_mask(catalog: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    error_columns = [
        column
        for column in (
            "magerr_F814W",
            "mag_error_F814W",
            "fluxerr_F814W",
            "pagul_fluxerr_F814W",
            "shipley_fluxerr_F814W",
            "FLUXERR_F814W",
            "e_FF814W",
        )
        if column in catalog.columns
    ]
    if not error_columns:
        return pd.Series(True, index=catalog.index, dtype=bool), pd.Series(False, index=catalog.index, dtype=bool)

    errors = pd.DataFrame({column: pd.to_numeric(catalog[column], errors="coerce") for column in error_columns})
    error_available = errors.notna().any(axis=1)
    has_positive_error = (errors > 0.0).any(axis=1)
    return ((~error_available) | has_positive_error).astype(bool), error_available.astype(bool)


def apply_image_precuts(
    catalog: pd.DataFrame,
    *,
    image_bright_mag_f814w: float = DEFAULT_IMAGE_BRIGHT_MAG_F814W,
    image_hff_faint_mag_f814w: float = DEFAULT_IMAGE_HFF_FAINT_MAG_F814W,
    image_outer_faint_mag_f814w: float = DEFAULT_IMAGE_OUTER_FAINT_MAG_F814W,
    image_min_size_arcsec: float = DEFAULT_IMAGE_MIN_SIZE_ARCSEC,
    image_size_pixel_scale_arcsec: float = DEFAULT_IMAGE_SIZE_PIXEL_SCALE_ARCSEC,
) -> pd.DataFrame:
    working = catalog.copy()
    hff_footprint = _image_hff_footprint_mask(working)
    mag_f814w = _valid_mag_array(_numeric_catalog_column(working, "mag_F814W"))
    size_arcsec = _image_size_arcsec(working, image_size_pixel_scale_arcsec)
    ellipticity, shape_available = _image_ellipticity(working)
    positive_error, error_available = _positive_f814w_error_mask(working)

    shipley_use = _numeric_catalog_column(working, "shipley_use")
    shipley_use_available = np.isfinite(shipley_use)
    f814w_flag = _numeric_catalog_column(working, "shipley_flf814w")
    f814w_flag_available = np.isfinite(f814w_flag)

    selected = np.ones(len(working), dtype=bool)
    reason_lists: list[list[str]] = [[] for _ in range(len(working))]

    def _reject(mask: pd.Series | np.ndarray, reason: str) -> None:
        mask_array = np.asarray(mask, dtype=bool)
        selected[mask_array] = False
        for position in np.flatnonzero(mask_array):
            reason_lists[int(position)].append(reason)

    finite_f814w = np.isfinite(mag_f814w)
    faint_limit = np.where(hff_footprint.to_numpy(dtype=bool), float(image_hff_faint_mag_f814w), float(image_outer_faint_mag_f814w))
    _reject(~finite_f814w, "missing_or_invalid_f814w")
    _reject(finite_f814w & (mag_f814w > faint_limit), "too_faint_f814w")
    _reject(np.isfinite(size_arcsec) & (size_arcsec < float(image_min_size_arcsec)), "too_small")
    _reject(shipley_use_available & (shipley_use != 1.0), "bad_shipley_use")
    _reject(f814w_flag_available & (f814w_flag != 0.0), "bad_f814w_flag")
    _reject(shape_available & ~((ellipticity > 0.0) & (ellipticity < 1.0)), "invalid_ellipticity")
    _reject(~positive_error, "nonpositive_f814w_error")

    working["image_hff_footprint"] = hff_footprint.to_numpy(dtype=bool)
    working["image_size_arcsec"] = size_arcsec.to_numpy(dtype=float)
    working["image_ellipticity"] = ellipticity.to_numpy(dtype=float)
    working["image_preclean_selected"] = selected.tolist()
    working["image_preclean_reject_reason"] = ["|".join(reasons) for reasons in reason_lists]
    working.attrs["image_preclean_metrics"] = {
        "image_bright_mag_f814w": float(image_bright_mag_f814w),
        "image_hff_faint_mag_f814w": float(image_hff_faint_mag_f814w),
        "image_outer_faint_mag_f814w": float(image_outer_faint_mag_f814w),
        "image_min_size_arcsec": float(image_min_size_arcsec),
        "image_size_pixel_scale_arcsec": float(image_size_pixel_scale_arcsec),
        "n_image_preclean_rows": int(len(working)),
        "n_image_preclean_selected": int(selected.sum()),
        "n_image_preclean_rejected": int((~selected).sum()),
        "n_image_preclean_size_available": int(np.isfinite(size_arcsec).sum()),
        "n_image_preclean_shape_available": int(shape_available.sum()),
        "n_image_preclean_shipley_use_available": int(shipley_use_available.sum()),
        "n_image_preclean_f814w_flag_available": int(f814w_flag_available.sum()),
        "n_image_preclean_f814w_error_available": int(error_available.sum()),
        "n_image_preclean_reject_missing_or_invalid_f814w": int((~finite_f814w).sum()),
        "n_image_preclean_reject_too_bright_f814w": 0,
        "n_image_preclean_reject_too_faint_f814w": int((finite_f814w & (mag_f814w > faint_limit)).sum()),
        "n_image_preclean_reject_too_small": int((np.isfinite(size_arcsec) & (size_arcsec < float(image_min_size_arcsec))).sum()),
        "n_image_preclean_reject_bad_shipley_use": int((shipley_use_available & (shipley_use != 1.0)).sum()),
        "n_image_preclean_reject_bad_f814w_flag": int((f814w_flag_available & (f814w_flag != 0.0)).sum()),
        "n_image_preclean_reject_invalid_ellipticity": int(
            (shape_available & ~((ellipticity > 0.0) & (ellipticity < 1.0))).sum()
        ),
        "n_image_preclean_reject_nonpositive_f814w_error": int((~positive_error).sum()),
    }
    return working


def apply_image_photoz_quality(
    catalog: pd.DataFrame,
    *,
    image_photoz_min_mag_f160w: float = DEFAULT_IMAGE_PHOTOZ_MIN_MAG_F160W,
    image_photoz_max_mag_f160w: float = DEFAULT_IMAGE_PHOTOZ_MAX_MAG_F160W,
    image_photoz_min_nb_used: float = DEFAULT_IMAGE_PHOTOZ_MIN_NB_USED,
) -> pd.DataFrame:
    working = catalog.copy()
    zpdf = _valid_redshift_array(_numeric_catalog_column(working, "pagul_zpdf"))
    mag_f160w = _valid_mag_array(_numeric_catalog_column(working, "mag_F160W"))
    nb_used = _numeric_catalog_column(working, "pagul_nb_used")

    selected = (
        np.isfinite(zpdf)
        & np.isfinite(mag_f160w)
        & (mag_f160w > float(image_photoz_min_mag_f160w))
        & (mag_f160w < float(image_photoz_max_mag_f160w))
        & np.isfinite(nb_used)
        & (nb_used >= float(image_photoz_min_nb_used))
    )
    reason_lists: list[list[str]] = [[] for _ in range(len(working))]

    def _append_reason(mask: pd.Series | np.ndarray, reason: str) -> None:
        mask_array = np.asarray(mask, dtype=bool)
        for position in np.flatnonzero(mask_array):
            reason_lists[int(position)].append(reason)

    _append_reason(~np.isfinite(zpdf), "missing_or_invalid_zpdf")
    _append_reason(
        ~np.isfinite(mag_f160w)
        | (mag_f160w <= float(image_photoz_min_mag_f160w))
        | (mag_f160w >= float(image_photoz_max_mag_f160w)),
        "f160w_outside_figure9_range",
    )
    _append_reason(~np.isfinite(nb_used) | (nb_used < float(image_photoz_min_nb_used)), "low_nb_used")

    working["image_photoz_quality_selected"] = selected.tolist()
    working["image_photoz_reject_reason"] = ["|".join(reasons) if not selected[index] else "" for index, reasons in enumerate(reason_lists)]
    working["image_zphot_family"] = np.where(selected, zpdf, np.nan)
    working.attrs["image_photoz_quality_metrics"] = {
        "image_photoz_min_mag_f160w": float(image_photoz_min_mag_f160w),
        "image_photoz_max_mag_f160w": float(image_photoz_max_mag_f160w),
        "image_photoz_min_nb_used": float(image_photoz_min_nb_used),
        "n_image_photoz_quality_rows": int(len(working)),
        "n_image_photoz_quality_selected": int(selected.sum()),
        "n_image_photoz_quality_rejected": int((~selected).sum()),
        "n_image_photoz_reject_missing_or_invalid_zpdf": int((~np.isfinite(zpdf)).sum()),
        "n_image_photoz_reject_f160w_outside_figure9_range": int(
            (
                ~np.isfinite(mag_f160w)
                | (mag_f160w <= float(image_photoz_min_mag_f160w))
                | (mag_f160w >= float(image_photoz_max_mag_f160w))
            ).sum()
        ),
        "n_image_photoz_reject_low_nb_used": int((~np.isfinite(nb_used) | (nb_used < float(image_photoz_min_nb_used))).sum()),
    }
    return working


def prepare_candidates(
    catalog: pd.DataFrame,
    spec: ClusterSpec,
    *,
    min_common_bands: int = DEFAULT_MIN_COMMON_BANDS,
    strong_lensing_rescue_faint_mag_f814w: float = DEFAULT_STRONG_LENSING_RESCUE_FAINT_MAG_F814W,
    strong_lensing_rescue_min_bands: int = DEFAULT_STRONG_LENSING_RESCUE_MIN_BANDS,
    image_family_fov_kpc: float = DEFAULT_IMAGE_FAMILY_FOV_KPC,
    image_bright_mag_f814w: float = DEFAULT_IMAGE_BRIGHT_MAG_F814W,
    image_hff_faint_mag_f814w: float = DEFAULT_IMAGE_HFF_FAINT_MAG_F814W,
    image_outer_faint_mag_f814w: float = DEFAULT_IMAGE_OUTER_FAINT_MAG_F814W,
    image_min_size_arcsec: float = DEFAULT_IMAGE_MIN_SIZE_ARCSEC,
    image_size_pixel_scale_arcsec: float = DEFAULT_IMAGE_SIZE_PIXEL_SCALE_ARCSEC,
    image_photoz_min_mag_f160w: float = DEFAULT_IMAGE_PHOTOZ_MIN_MAG_F160W,
    image_photoz_max_mag_f160w: float = DEFAULT_IMAGE_PHOTOZ_MAX_MAG_F160W,
    image_photoz_min_nb_used: float = DEFAULT_IMAGE_PHOTOZ_MIN_NB_USED,
) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame()

    working = catalog.copy()
    for column in ("ra", "dec", "zspec_best", "zphot_best", "zspec_best_confidence_rank"):
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")
    for column in MAG_COLUMNS:
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")

    mag_values = working.loc[:, MAG_COLUMNS].apply(lambda column: column.map(_valid_mag))
    working["n_valid_bands"] = mag_values.notna().sum(axis=1).astype(int)
    working = apply_image_precuts(
        working,
        image_bright_mag_f814w=image_bright_mag_f814w,
        image_hff_faint_mag_f814w=image_hff_faint_mag_f814w,
        image_outer_faint_mag_f814w=image_outer_faint_mag_f814w,
        image_min_size_arcsec=image_min_size_arcsec,
        image_size_pixel_scale_arcsec=image_size_pixel_scale_arcsec,
    )
    image_preclean_metrics = working.attrs.get("image_preclean_metrics", {})
    working = apply_image_photoz_quality(
        working,
        image_photoz_min_mag_f160w=image_photoz_min_mag_f160w,
        image_photoz_max_mag_f160w=image_photoz_max_mag_f160w,
        image_photoz_min_nb_used=image_photoz_min_nb_used,
    )
    image_photoz_quality_metrics = working.attrs.get("image_photoz_quality_metrics", {})
    finite_coords = np.isfinite(working["ra"]) & np.isfinite(working["dec"])

    zspec = working["zspec_best"].map(valid_redshift)
    rank = pd.to_numeric(working["zspec_best_confidence_rank"], errors="coerce")
    usable_spec = pd.Series(_usable_family_spec_mask(zspec.to_numpy(dtype=float)), index=working.index)
    background_spec = zspec > spec.z_lens + BACKGROUND_Z_MARGIN
    image_zphot_family = _valid_redshift_array(_numeric_catalog_column(working, "image_zphot_family"))
    image_photoz_selected = (
        working["image_photoz_quality_selected"].map(_bool_value)
        if "image_photoz_quality_selected" in working.columns
        else pd.Series(False, index=working.index)
    )
    background_photoz = image_zphot_family > spec.z_lens + BACKGROUND_Z_MARGIN
    qualified_background_photoz = image_photoz_selected & background_photoz & ~usable_spec
    background_evidence = (usable_spec & background_spec) | qualified_background_photoz

    if "zspec_conflict" in working.columns:
        conflict_flag = working["zspec_conflict"].map(_bool_value)
    else:
        conflict_flag = pd.Series(False, index=working.index)
    if "zspec_selection_note" in working.columns:
        conflict_note = working["zspec_selection_note"].astype(str).str.lower().str.contains("conflict", na=False)
    else:
        conflict_note = pd.Series(False, index=working.index)
    specz_conflict_diagnostic = conflict_flag | conflict_note

    if "member_probability" in working.columns:
        if "cluster_member_selected" in working.columns:
            likely_member = working["cluster_member_selected"].map(_bool_value)
        else:
            member_probability = pd.to_numeric(working["member_probability"], errors="coerce").fillna(0.0)
            likely_member = member_probability >= DEFAULT_MEMBER_PROBABILITY_THRESHOLD
        if "member_for_lensing" in working.columns:
            likely_member = likely_member | working["member_for_lensing"].map(_bool_value)
    else:
        member_zspec = (
            working["member_zspec_candidate"].map(_bool_value)
            if "member_zspec_candidate" in working.columns
            else pd.Series(False, index=working.index)
        )
        member_photo = (
            working["member_photoz_candidate"].map(_bool_value)
            if "member_photoz_candidate" in working.columns
            else pd.Series(False, index=working.index)
        )
        likely_member = member_zspec | (member_photo & ~background_spec)

    center_ra, center_dec, center_note = _choose_plot_center(working, working)
    fov_radius_kpc = float(image_family_fov_kpc) / 2.0
    projected = _add_projected_offsets(working, spec, center_ra=center_ra, center_dec=center_dec)
    working["image_family_x_kpc"] = np.nan
    working["image_family_y_kpc"] = np.nan
    working["image_family_radius_kpc"] = np.nan
    if not projected.empty:
        working.loc[projected.index, "image_family_x_kpc"] = pd.to_numeric(projected["x_kpc"], errors="coerce")
        working.loc[projected.index, "image_family_y_kpc"] = pd.to_numeric(projected["y_kpc"], errors="coerce")
        working.loc[projected.index, "image_family_radius_kpc"] = pd.to_numeric(projected["radius_kpc"], errors="coerce")
    fov_mask = (
        np.isfinite(working["image_family_radius_kpc"])
        & (working["image_family_radius_kpc"] <= fov_radius_kpc)
    )
    preclean_selected = working["image_preclean_selected"].map(_bool_value)
    valid_band_count = working["n_valid_bands"] >= min_common_bands
    rescue_min_bands = max(1, min(int(strong_lensing_rescue_min_bands), int(min_common_bands)))
    rescue_band_count = working["n_valid_bands"] >= rescue_min_bands
    preclean_reasons = (
        working["image_preclean_reject_reason"].astype(str)
        if "image_preclean_reject_reason" in working.columns
        else pd.Series("", index=working.index)
    )
    reason_sets = preclean_reasons.map(lambda text: {item for item in str(text).split("|") if item})
    rescue_allowed_reasons = reason_sets.map(
        lambda reasons: bool(reasons)
        and reasons.issubset({"bad_f814w_flag", "too_faint_f814w"})
        and bool(reasons & {"bad_f814w_flag", "too_faint_f814w"})
    )
    finite_f814w = np.isfinite(mag_values["mag_F814W"].to_numpy(dtype=float)) if "mag_F814W" in mag_values else np.zeros(len(working), dtype=bool)
    mag_f814w = mag_values["mag_F814W"].to_numpy(dtype=float) if "mag_F814W" in mag_values else np.full(len(working), np.nan)
    strong_rescue_evidence = usable_spec & background_spec
    within_rescue_faint_limit = finite_f814w & (mag_f814w <= float(strong_lensing_rescue_faint_mag_f814w))
    rescue_selected = (
        ~preclean_selected
        & rescue_allowed_reasons
        & rescue_band_count
        & strong_rescue_evidence
        & within_rescue_faint_limit
    )
    family_candidate_flags: list[str] = []
    for row_index in working.index:
        flags: list[str] = []
        if bool(rescue_selected.loc[row_index]):
            flags.append("strong_lensing_rescue")
            reasons = sorted(reason_sets.loc[row_index])
            flags.extend(f"rescued_{reason}" for reason in reasons)
            if int(working.loc[row_index, "n_valid_bands"]) < int(min_common_bands):
                flags.append("rescued_low_band_count")
        if bool(specz_conflict_diagnostic.loc[row_index]):
            flags.append("zspec_conflict_candidate")
        family_candidate_flags.append("|".join(dict.fromkeys(flags)))
    working["image_family_rescue_selected"] = rescue_selected.to_numpy(dtype=bool)
    working["image_family_review_flags"] = family_candidate_flags
    candidate_preclean_selected = preclean_selected | rescue_selected
    base_selection = finite_coords & candidate_preclean_selected & ~likely_member & (valid_band_count | rescue_selected)
    mask = base_selection & background_evidence & fov_mask
    candidates = working.loc[mask].copy().reset_index().rename(columns={"index": "catalog_index"})
    candidates.attrs["image_preclean_metrics"] = image_preclean_metrics
    candidates.attrs["image_photoz_quality_metrics"] = image_photoz_quality_metrics
    candidates.attrs["image_family_selection_metrics"] = {
        "image_family_fov_kpc": float(image_family_fov_kpc),
        "image_family_fov_shape": "circle",
        "image_family_fov_radius_kpc": fov_radius_kpc,
        "image_family_fov_half_width_kpc": fov_radius_kpc,
        "image_family_center_ra": float(center_ra),
        "image_family_center_dec": float(center_dec),
        "image_family_center_note": center_note,
        "n_image_family_fov_input_rows": int(len(working)),
        "n_image_family_fov_rows": int((finite_coords & fov_mask).sum()),
        "n_rejected_missing_strong_specz": int((base_selection & ~usable_spec & ~qualified_background_photoz).sum()),
        "n_rejected_missing_family_redshift_evidence": int((base_selection & ~background_evidence).sum()),
        "n_image_family_strong_specz_candidates": int((mask & usable_spec & background_spec).sum()),
        "n_image_family_photoz_companion_candidates": int((mask & ~usable_spec & qualified_background_photoz).sum()),
        "n_image_family_rescue_candidates": int((mask & rescue_selected).sum()),
        "strong_lensing_rescue_faint_mag_f814w": float(strong_lensing_rescue_faint_mag_f814w),
        "strong_lensing_rescue_min_bands": int(strong_lensing_rescue_min_bands),
        "n_rejected_outside_family_fov": int((finite_coords & candidate_preclean_selected & ~likely_member & background_evidence & (valid_band_count | rescue_selected) & ~fov_mask).sum()),
        "n_rejected_zspec_conflict": 0,
        "n_zspec_conflict_candidates": int((mask & specz_conflict_diagnostic).sum()),
        "n_rejected_not_background_specz": int((base_selection & usable_spec & ~background_spec & ~qualified_background_photoz).sum()),
    }
    if "object_id" not in candidates.columns:
        candidates["object_id"] = candidates["catalog_index"].map(lambda value: f"row:{value}")
    if "object_source" not in candidates.columns:
        candidates["object_source"] = ""
    if "catalog_sources" not in candidates.columns:
        candidates["catalog_sources"] = ""
    return candidates


def _empty_pair_score_metrics(backend: str, n_spatial_pairs: int, elapsed: float) -> dict[str, Any]:
    return {
        "pair_score_backend": backend,
        "n_spatial_pairs": int(n_spatial_pairs),
        "n_prefiltered_pairs": 0,
        "n_scored_pairs": 0,
        "n_pruned_redshift": 0,
        "n_pruned_photoz_delta": 0,
        "n_pruned_score_upper_bound": 0,
        "n_pruned_common_bands": 0,
        "n_prefilter_full_score_pairs": 0,
        "n_prefilter_redshift_rejects": 0,
        "n_prefilter_photoz_delta_rejects": 0,
        "n_accepted_array_pairs": 0,
        "pair_score_seconds": float(elapsed),
    }


def _valid_redshift_array(values: pd.Series | np.ndarray, *, max_z: float = 20.0) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float) if isinstance(values, pd.Series) else np.asarray(values, dtype=float)
    invalid = ~np.isfinite(arr) | np.isin(arr, list(INVALID_SENTINELS)) | (arr <= 0.0) | (arr >= max_z)
    result = arr.copy()
    result[invalid] = np.nan
    return result


def _valid_mag_array(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float) if isinstance(values, pd.Series) else np.asarray(values, dtype=float)
    invalid = ~np.isfinite(arr) | np.isin(arr, list(INVALID_SENTINELS)) | (arr <= 0.0) | (arr >= 80.0)
    result = arr.copy()
    result[invalid] = np.nan
    return result


def _usable_family_spec_mask(zspec_values: np.ndarray) -> np.ndarray:
    return np.isfinite(zspec_values)


def _finite_min(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.min(finite)) if finite.size else float("nan")


def _candidate_pair_arrays(candidates: pd.DataFrame) -> dict[str, Any]:
    mags = np.column_stack(
        [
            _valid_mag_array(candidates[column]) if column in candidates.columns else np.full(len(candidates), np.nan)
            for column in MAG_COLUMNS
        ]
    )
    return {
        "object_ids": candidates["object_id"].astype(str).to_numpy()
        if "object_id" in candidates.columns
        else np.asarray([f"row:{idx}" for idx in range(len(candidates))], dtype=object),
        "ra": pd.to_numeric(candidates["ra"], errors="coerce").to_numpy(dtype=float),
        "dec": pd.to_numeric(candidates["dec"], errors="coerce").to_numpy(dtype=float),
        "zspec": _valid_redshift_array(candidates["zspec_best"])
        if "zspec_best" in candidates.columns
        else np.full(len(candidates), np.nan),
        "rank": pd.to_numeric(candidates["zspec_best_confidence_rank"], errors="coerce").to_numpy(dtype=float)
        if "zspec_best_confidence_rank" in candidates.columns
        else np.full(len(candidates), np.nan),
        "zphot": _valid_redshift_array(candidates["image_zphot_family"])
        if "image_zphot_family" in candidates.columns
        else np.full(len(candidates), np.nan),
        "zlow": _valid_redshift_array(candidates["pagul_zpdf_low"])
        if "pagul_zpdf_low" in candidates.columns
        else np.full(len(candidates), np.nan),
        "zhigh": _valid_redshift_array(candidates["pagul_zpdf_high"])
        if "pagul_zpdf_high" in candidates.columns
        else np.full(len(candidates), np.nan),
        "redshift_conflict": (
            candidates["image_family_review_flags"].astype(str).str.contains("zspec_conflict_candidate", na=False).to_numpy(dtype=bool)
            if "image_family_review_flags" in candidates.columns
            else np.zeros(len(candidates), dtype=bool)
        ),
        "mags": mags,
    }


def _empty_accepted_pair_arrays(object_ids: np.ndarray | None = None) -> dict[str, Any]:
    base_object_ids = np.asarray([], dtype=object) if object_ids is None else np.asarray(object_ids, dtype=object)
    return {
        "object_ids": base_object_ids,
        "left_idx": np.empty(0, dtype=np.int32),
        "right_idx": np.empty(0, dtype=np.int32),
        "pair_score": np.empty(0, dtype=np.float32),
        "specz_score": np.empty(0, dtype=np.float32),
        "photoz_score": np.empty(0, dtype=np.float32),
        "zphot_delta": np.empty(0, dtype=np.float32),
        "color_score": np.empty(0, dtype=np.float32),
        "sed_rms": np.empty(0, dtype=np.float32),
        "n_common_bands": np.empty(0, dtype=np.int32),
        "separation_arcsec": np.empty(0, dtype=np.float32),
        "separation_kpc": np.empty(0, dtype=np.float32),
    }


def _accepted_arrays_from_scored_pairs(
    arrays: dict[str, Any],
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    selected: np.ndarray,
    scores: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    if not np.any(selected):
        return _empty_accepted_pair_arrays(arrays.get("object_ids"))
    (
        separation_arcsec,
        separation_kpc,
        pair_score,
        specz_score,
        photoz_score,
        zphot_delta,
        color_score,
        sed_rms,
        n_common,
        _reject_code,
        _relation_code,
        _score_upper_bound,
    ) = scores
    return {
        "object_ids": np.asarray(arrays["object_ids"], dtype=object),
        "left_idx": np.asarray(left_idx[selected], dtype=np.int32),
        "right_idx": np.asarray(right_idx[selected], dtype=np.int32),
        "pair_score": np.asarray(pair_score[selected], dtype=np.float32),
        "specz_score": np.asarray(specz_score[selected], dtype=np.float32),
        "photoz_score": np.asarray(photoz_score[selected], dtype=np.float32),
        "zphot_delta": np.asarray(zphot_delta[selected], dtype=np.float32),
        "color_score": np.asarray(color_score[selected], dtype=np.float32),
        "sed_rms": np.asarray(sed_rms[selected], dtype=np.float32),
        "n_common_bands": np.asarray(n_common[selected], dtype=np.int32),
        "separation_arcsec": np.asarray(separation_arcsec[selected], dtype=np.float32),
        "separation_kpc": np.asarray(separation_kpc[selected], dtype=np.float32),
    }


def _concat_accepted_arrays(chunks: list[dict[str, Any]], object_ids: np.ndarray) -> dict[str, Any]:
    if not chunks:
        return _empty_accepted_pair_arrays(object_ids)
    result = {"object_ids": np.asarray(object_ids, dtype=object)}
    for key in (
        "left_idx",
        "right_idx",
        "pair_score",
        "specz_score",
        "photoz_score",
        "zphot_delta",
        "color_score",
        "sed_rms",
        "n_common_bands",
        "separation_arcsec",
        "separation_kpc",
    ):
        result[key] = np.concatenate([chunk[key] for chunk in chunks]) if chunks else _empty_accepted_pair_arrays(object_ids)[key]
    return result


def _accepted_arrays_from_pair_dataframe(candidates: pd.DataFrame, accepted: pd.DataFrame) -> dict[str, Any]:
    object_ids = (
        candidates["object_id"].astype(str).to_numpy()
        if "object_id" in candidates.columns
        else np.asarray([f"row:{idx}" for idx in range(len(candidates))], dtype=object)
    )
    if accepted.empty:
        return _empty_accepted_pair_arrays(object_ids)
    object_index = {object_id: idx for idx, object_id in enumerate(object_ids.astype(str))}
    left_idx = accepted["left_object_id"].astype(str).map(object_index).to_numpy(dtype=np.int32)
    right_idx = accepted["right_object_id"].astype(str).map(object_index).to_numpy(dtype=np.int32)
    return {
        "object_ids": object_ids,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "pair_score": pd.to_numeric(accepted["pair_score"], errors="coerce").to_numpy(dtype=np.float32),
        "specz_score": pd.to_numeric(accepted["specz_score"], errors="coerce").to_numpy(dtype=np.float32),
        "photoz_score": pd.to_numeric(accepted["photoz_score"], errors="coerce").to_numpy(dtype=np.float32),
        "zphot_delta": pd.to_numeric(
            accepted["zphot_delta"] if "zphot_delta" in accepted.columns else pd.Series(np.nan, index=accepted.index),
            errors="coerce",
        ).to_numpy(dtype=np.float32),
        "color_score": pd.to_numeric(accepted["color_score"], errors="coerce").to_numpy(dtype=np.float32),
        "sed_rms": pd.to_numeric(accepted["sed_rms"], errors="coerce").to_numpy(dtype=np.float32),
        "n_common_bands": pd.to_numeric(accepted["n_common_bands"], errors="coerce").fillna(0).to_numpy(dtype=np.int32),
        "separation_arcsec": pd.to_numeric(accepted["separation_arcsec"], errors="coerce").to_numpy(dtype=np.float32),
        "separation_kpc": pd.to_numeric(accepted["separation_kpc"], errors="coerce").to_numpy(dtype=np.float32),
    }


def _spatial_candidate_pairs(arrays: dict[str, Any], max_sep_arcsec: float) -> np.ndarray:
    ra = np.asarray(arrays["ra"], dtype=float)
    dec = np.asarray(arrays["dec"], dtype=float)
    valid = np.isfinite(ra) & np.isfinite(dec)
    if int(valid.sum()) < 2:
        return np.empty((0, 2), dtype=np.int64)
    valid_indices = np.flatnonzero(valid)
    ra_valid = ra[valid]
    dec_valid = dec[valid]
    ra0 = float(np.nanmedian(ra_valid))
    dec0 = float(np.nanmedian(dec_valid))
    cos_dec0 = math.cos(math.radians(dec0))
    xy = np.column_stack(((ra_valid - ra0) * cos_dec0 * 3600.0, (dec_valid - dec0) * 3600.0))
    pairs = cKDTree(xy).query_pairs(float(max_sep_arcsec), output_type="ndarray")
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    result = valid_indices[np.asarray(pairs, dtype=np.int64)]
    result.sort(axis=1)
    order = np.lexsort((result[:, 1], result[:, 0]))
    return result[order]


def _rank_to_quality_array(xp: Any, rank: Any) -> Any:
    return xp.where(
        rank >= 3.0,
        1.0,
        xp.where(rank >= 2.0, 0.85, xp.where(rank >= 1.0, 0.55, xp.where(rank >= 0.5, 0.35, 0.15))),
    )


def _photo_sigma_array(xp: Any, z: Any) -> Any:
    return xp.maximum(PHOTOZ_SIGMA_FLOOR, PHOTOZ_SIGMA_SCALE * (1.0 + xp.where(xp.isfinite(z), z, 0.0)))


def _photo_pair_score_array(xp: Any, z_left: Any, z_right: Any, low_left: Any, high_left: Any, low_right: Any, high_right: Any) -> Any:
    valid = xp.isfinite(z_left) & xp.isfinite(z_right)
    scale = xp.maximum(_photo_sigma_array(xp, z_left), _photo_sigma_array(xp, z_right))
    gaussian = PHOTOZ_ONLY_SCORE_CAP * xp.exp(-0.5 * ((z_left - z_right) / scale) ** 2)
    return xp.where(valid, gaussian, 0.0)


def _photo_consistent_with_spec_array(xp: Any, z_photo: Any, low: Any, high: Any, z_spec: Any) -> Any:
    valid = xp.isfinite(z_photo) & xp.isfinite(z_spec)
    scale = _photo_sigma_array(xp, z_spec)
    gaussian = SPEC_PHOTO_SCORE_CAP * xp.exp(-0.5 * ((z_photo - z_spec) / scale) ** 2)
    return xp.where(valid, gaussian, 0.0)


def _spec_agreement_score_array(xp: Any, spec_dz: Any) -> Any:
    transition = max(ZSPEC_HARD_CONFLICT_TOL - ZSPEC_EXCELLENT_TOL, 1.0e-6)
    excess = xp.clip((spec_dz - ZSPEC_EXCELLENT_TOL) / transition, 0.0, 1.0)
    return xp.where(spec_dz <= ZSPEC_EXCELLENT_TOL, 1.0, 1.0 - 0.30 * excess)


def _spec_photo_score_array(
    xp: Any,
    z_spec: Any,
    z_photo: Any,
    z_photo_low: Any,
    z_photo_high: Any,
    image_photoz_max_dz_norm: float,
) -> tuple[Any, Any]:
    valid = xp.isfinite(z_spec) & xp.isfinite(z_photo)
    has_interval = xp.isfinite(z_photo_low) & xp.isfinite(z_photo_high) & (z_photo_low <= z_photo_high)
    interval_consistent = has_interval & (z_photo_low <= z_spec) & (z_spec <= z_photo_high)
    dz_norm = xp.abs(z_photo - z_spec) / (1.0 + xp.maximum(z_spec, 0.0))
    normalized_consistent = dz_norm <= image_photoz_max_dz_norm
    scale = xp.maximum(image_photoz_max_dz_norm, 1.0e-6)
    gaussian = SPEC_PHOTO_SCORE_CAP * xp.exp(-0.5 * (dz_norm / scale) ** 2)
    interval_floor = SPEC_PHOTO_SCORE_CAP * 0.85
    score = xp.where(interval_consistent, xp.maximum(gaussian, interval_floor), gaussian)
    consistent = valid & (interval_consistent | normalized_consistent)
    return xp.where(consistent, score, 0.0), consistent


def _family_redshift_evidence_array(
    xp: Any,
    zspec_left: Any,
    zspec_right: Any,
    rank_left: Any,
    rank_right: Any,
    zphot_left: Any,
    zphot_right: Any,
    zlow_left: Any,
    zlow_right: Any,
    zhigh_left: Any,
    zhigh_right: Any,
    image_photoz_max_dz_norm: float,
    family_photoz_delta_max: float,
) -> tuple[Any, ...]:
    usable_spec_left = xp.isfinite(zspec_left)
    usable_spec_right = xp.isfinite(zspec_right)
    both_specz = usable_spec_left & usable_spec_right
    one_spec_left = usable_spec_left & ~usable_spec_right & xp.isfinite(zphot_right)
    one_spec_right = usable_spec_right & ~usable_spec_left & xp.isfinite(zphot_left)
    one_spec_one_photo = one_spec_left | one_spec_right
    spec_dz = xp.abs(zspec_left - zspec_right)
    zphot_delta = xp.abs(zphot_left - zphot_right)
    quality_left = _rank_to_quality_array(xp, rank_left)
    quality_right = _rank_to_quality_array(xp, rank_right)
    both_quality = xp.sqrt(quality_left * quality_right)
    spec_agreement = _spec_agreement_score_array(xp, spec_dz)
    both_specz_score = xp.clip(spec_agreement * both_quality, 0.0, 1.0)
    specz_score = xp.where(both_specz, spec_agreement, 0.0)
    left_spec_photo_score, left_spec_photo_consistent = _spec_photo_score_array(
        xp,
        zspec_left,
        zphot_right,
        zlow_right,
        zhigh_right,
        image_photoz_max_dz_norm,
    )
    right_spec_photo_score, right_spec_photo_consistent = _spec_photo_score_array(
        xp,
        zspec_right,
        zphot_left,
        zlow_left,
        zhigh_left,
        image_photoz_max_dz_norm,
    )
    spec_photo_score = xp.where(one_spec_left, left_spec_photo_score, xp.where(one_spec_right, right_spec_photo_score, 0.0))
    spec_photo_consistent = xp.where(
        one_spec_left,
        left_spec_photo_consistent,
        xp.where(one_spec_right, right_spec_photo_consistent, False),
    )
    spec_mid = 0.5 * (zspec_left + zspec_right)
    spec_dz_norm = spec_dz / xp.maximum(1.0 + spec_mid, 1.0e-6)
    specz_conflict = both_specz & (spec_dz > ZSPEC_HARD_CONFLICT_TOL) & (spec_dz_norm > ZSPEC_HARD_CONFLICT_NORM_TOL)
    both_have_photoz = xp.isfinite(zphot_left) & xp.isfinite(zphot_right)
    photoz_only_pair = ~usable_spec_left & ~usable_spec_right & both_have_photoz
    photoz_pair_score = _photo_pair_score_array(xp, zphot_left, zphot_right, zlow_left, zhigh_left, zlow_right, zhigh_right)
    photoz_delta_too_large = (one_spec_one_photo | photoz_only_pair) & both_have_photoz & ~(zphot_delta < family_photoz_delta_max)
    missing_specz_evidence = ~(both_specz | one_spec_one_photo | photoz_only_pair)
    photoz_inconsistent = one_spec_one_photo & ~spec_photo_consistent
    redshift_reject_code = xp.where(
        specz_conflict,
        PAIR_REJECT_SPECZ_CONFLICT,
        xp.where(
            photoz_delta_too_large,
            PAIR_REJECT_PHOTOZ_DELTA,
            xp.where(
                photoz_inconsistent,
                PAIR_REJECT_PHOTOZ_INCONSISTENT,
                xp.where(missing_specz_evidence, PAIR_REJECT_MISSING_STRONG_SPECZ, PAIR_REJECT_NONE),
            ),
        ),
    )
    specphoto_score = xp.where(
        both_specz,
        both_specz_score,
        xp.where(one_spec_one_photo, spec_photo_score, xp.where(photoz_only_pair, photoz_pair_score, 0.0)),
    )
    specphoto_score = xp.where(redshift_reject_code == PAIR_REJECT_NONE, specphoto_score, 0.0)
    specz_score = xp.where(redshift_reject_code == PAIR_REJECT_NONE, specz_score, 0.0)
    photoz_score = xp.where(
        (redshift_reject_code == PAIR_REJECT_NONE),
        xp.where(one_spec_one_photo, spec_photo_score, xp.where(photoz_only_pair, photoz_pair_score, 0.0)),
        0.0,
    )
    single_quality = xp.where(one_spec_left, xp.sqrt(quality_left * 0.70), xp.where(one_spec_right, xp.sqrt(quality_right * 0.70), 0.0))
    quality_score = xp.where(both_specz, both_quality, xp.where(one_spec_one_photo, single_quality, xp.where(photoz_only_pair, 0.70, 0.0)))
    relation_code = xp.where(both_specz, PAIR_RELATION_BOTH_SPECZ, PAIR_RELATION_MISSING_STRONG_SPECZ)
    relation_code = xp.where(one_spec_one_photo, PAIR_RELATION_SPECZ_PHOTOZ, relation_code)
    relation_code = xp.where(photoz_only_pair, PAIR_RELATION_PHOTOZ_ONLY, relation_code)
    relation_code = xp.where(photoz_inconsistent, PAIR_RELATION_PHOTOZ_INCONSISTENT, relation_code)
    relation_code = xp.where(specz_conflict, PAIR_RELATION_SPECZ_CONFLICT, relation_code)
    return specphoto_score, specz_score, photoz_score, quality_score, relation_code, redshift_reject_code, zphot_delta


def _score_pair_batch_array(
    xp: Any,
    left_idx: Any,
    right_idx: Any,
    valid_pair: Any,
    ra_deg: Any,
    dec_deg: Any,
    zspec: Any,
    rank: Any,
    zphot: Any,
    zlow: Any,
    zhigh: Any,
    redshift_conflict: Any,
    mags: Any,
    kpc_scale: float,
    max_family_span_kpc: float,
    min_pair_separation_arcsec: float,
    min_common_bands: int,
    image_photoz_max_dz_norm: float,
    family_color_rms_max: float,
    family_photoz_delta_max: float,
) -> tuple[Any, ...]:
    left_idx = left_idx.astype("int32")
    right_idx = right_idx.astype("int32")
    ra_left = ra_deg[left_idx]
    ra_right = ra_deg[right_idx]
    dec_left = dec_deg[left_idx]
    dec_right = dec_deg[right_idx]

    ra1 = ra_left * (math.pi / 180.0)
    ra2 = ra_right * (math.pi / 180.0)
    dec1 = dec_left * (math.pi / 180.0)
    dec2 = dec_right * (math.pi / 180.0)
    sin_ddec = xp.sin((dec2 - dec1) / 2.0)
    sin_dra = xp.sin((ra2 - ra1) / 2.0)
    hav = sin_ddec**2 + xp.cos(dec1) * xp.cos(dec2) * sin_dra**2
    sep_rad = 2.0 * xp.arcsin(xp.minimum(1.0, xp.sqrt(xp.maximum(hav, 0.0))))
    separation_arcsec = sep_rad * ARCSEC_PER_RADIAN
    separation_kpc = separation_arcsec * kpc_scale
    finite_sep = xp.isfinite(separation_arcsec) & valid_pair
    too_close = (~finite_sep) | (separation_arcsec < min_pair_separation_arcsec)
    too_far = separation_kpc > max_family_span_kpc

    zspec_left = zspec[left_idx]
    zspec_right = zspec[right_idx]
    rank_left = rank[left_idx]
    rank_right = rank[right_idx]
    zphot_left = zphot[left_idx]
    zphot_right = zphot[right_idx]
    zlow_left = zlow[left_idx]
    zlow_right = zlow[right_idx]
    zhigh_left = zhigh[left_idx]
    zhigh_right = zhigh[right_idx]
    conflict_pair = redshift_conflict[left_idx] | redshift_conflict[right_idx]
    specphoto_score, specz_score, photoz_score, quality_score, relation_code, redshift_reject_code, zphot_delta = (
        _family_redshift_evidence_array(
            xp,
            zspec_left,
            zspec_right,
            rank_left,
            rank_right,
            zphot_left,
            zphot_right,
            zlow_left,
            zlow_right,
            zhigh_left,
            zhigh_right,
            image_photoz_max_dz_norm,
            family_photoz_delta_max,
        )
    )

    mags_left = mags[left_idx]
    mags_right = mags[right_idx]
    common = xp.isfinite(mags_left) & xp.isfinite(mags_right)
    n_common = xp.sum(common, axis=1)
    diff = mags_left - mags_right
    filled = xp.where(common, diff, xp.inf)
    sorted_diff = xp.sort(filled, axis=1)
    safe_n_common = xp.maximum(n_common, 1)
    low_index = ((safe_n_common - 1) // 2).astype("int32")
    high_index = (safe_n_common // 2).astype("int32")
    row_index = xp.arange(left_idx.shape[0])
    median_low = sorted_diff[row_index, low_index]
    median_high = sorted_diff[row_index, high_index]
    offset = 0.5 * (median_low + median_high)
    residuals = xp.where(common, diff - offset[:, None], 0.0)
    sed_rms = xp.sqrt(xp.sum(residuals**2, axis=1) / safe_n_common)
    sed_rms = xp.where(n_common > 0, sed_rms, xp.nan)
    color_mid_width = max(FAMILY_COLOR_RMS_ACCEPTABLE - FAMILY_COLOR_RMS_STRONG, 1.0e-6)
    color_weak_width = xp.maximum(family_color_rms_max - FAMILY_COLOR_RMS_ACCEPTABLE, 1.0e-6)
    strong_color_score = 1.0 - 0.20 * xp.clip(sed_rms / FAMILY_COLOR_RMS_STRONG, 0.0, 1.0)
    mid_color_score = 0.80 - 0.30 * xp.clip((sed_rms - FAMILY_COLOR_RMS_STRONG) / color_mid_width, 0.0, 1.0)
    weak_color_score = 0.50 - 0.25 * xp.clip((sed_rms - FAMILY_COLOR_RMS_ACCEPTABLE) / color_weak_width, 0.0, 1.0)
    raw_color_score = xp.where(
        sed_rms <= FAMILY_COLOR_RMS_STRONG,
        strong_color_score,
        xp.where(sed_rms <= FAMILY_COLOR_RMS_ACCEPTABLE, mid_color_score, weak_color_score),
    )
    raw_color_score = xp.clip(raw_color_score, 0.0, 1.0)

    insufficient = n_common < min_common_bands
    color_reject = sed_rms >= family_color_rms_max
    reject_code = xp.where(too_close, PAIR_REJECT_TOO_CLOSE, PAIR_REJECT_NONE)
    reject_code = xp.where((reject_code == PAIR_REJECT_NONE) & too_far, PAIR_REJECT_TOO_FAR, reject_code)
    redshift_reject_for_color_fallback = (
        conflict_pair
        & (
            (redshift_reject_code == PAIR_REJECT_SPECZ_CONFLICT)
            | (redshift_reject_code == PAIR_REJECT_PHOTOZ_INCONSISTENT)
            | (redshift_reject_code == PAIR_REJECT_PHOTOZ_DELTA)
        )
    )
    reject_code = xp.where(
        (reject_code == PAIR_REJECT_NONE) & (redshift_reject_code != PAIR_REJECT_NONE) & ~redshift_reject_for_color_fallback,
        redshift_reject_code,
        reject_code,
    )
    reject_code = xp.where((reject_code == PAIR_REJECT_NONE) & insufficient, PAIR_REJECT_INSUFFICIENT_BANDS, reject_code)
    reject_code = xp.where((reject_code == PAIR_REJECT_NONE) & color_reject, PAIR_REJECT_COLOR_RMS, reject_code)
    reject_code = xp.where(valid_pair, reject_code, PAIR_REJECT_TOO_CLOSE)

    separation_score = xp.maximum(0.60, 1.0 - 0.40 * (separation_kpc / max_family_span_kpc))
    score_upper_bound = (
        xp.maximum(specphoto_score, 1.0e-6) ** 0.60
        * xp.maximum(separation_score, 1.0e-6) ** 0.07
        * xp.maximum(quality_score, 1.0e-6) ** 0.03
    )
    redshift_rejected = redshift_reject_code != PAIR_REJECT_NONE
    color_fallback = redshift_reject_for_color_fallback & ~too_close & ~too_far & ~insufficient & (sed_rms <= DEFAULT_CONFLICT_COLOR_FAMILY_RMS_MAX)
    color_score = xp.where(color_reject | insufficient | (redshift_rejected & ~color_fallback) | too_close | too_far, 0.0, raw_color_score)
    color_fallback_redshift_score = xp.clip(0.52 + 0.20 * (1.0 - sed_rms / DEFAULT_CONFLICT_COLOR_FAMILY_RMS_MAX), 0.35, 0.72)
    effective_specphoto_score = xp.where(color_fallback, color_fallback_redshift_score, specphoto_score)
    effective_specz_score = xp.where(color_fallback, color_fallback_redshift_score, specz_score)
    effective_photoz_score = xp.where(color_fallback, color_fallback_redshift_score, photoz_score)
    effective_quality_score = xp.where(color_fallback, xp.maximum(quality_score, 0.70), quality_score)
    relation_code = xp.where(color_fallback, PAIR_RELATION_COLOR_GUIDED_REDSHIFT_CONFLICT, relation_code)
    pair_score = (
        xp.maximum(effective_specphoto_score, 1.0e-6) ** 0.60
        * xp.maximum(color_score, 1.0e-6) ** 0.30
        * xp.maximum(separation_score, 1.0e-6) ** 0.07
        * xp.maximum(effective_quality_score, 1.0e-6) ** 0.03
    )
    reject_code = xp.where((reject_code == PAIR_REJECT_NONE) & redshift_reject_for_color_fallback & ~color_fallback, redshift_reject_code, reject_code)
    pair_score = xp.where(reject_code == PAIR_REJECT_NONE, xp.clip(pair_score, 0.0, 1.0), 0.0)
    n_common_out = xp.where((too_close | too_far | (redshift_rejected & ~color_fallback)), 0, n_common)
    sed_rms_out = xp.where((too_close | too_far | (redshift_rejected & ~color_fallback) | insufficient), xp.nan, sed_rms)
    return (
        separation_arcsec,
        separation_kpc,
        pair_score,
        effective_specz_score,
        effective_photoz_score,
        zphot_delta,
        color_score,
        sed_rms_out,
        n_common_out,
        reject_code,
        relation_code,
        score_upper_bound,
    )


def _prefilter_pair_batch_array(
    xp: Any,
    left_idx: Any,
    right_idx: Any,
    valid_pair: Any,
    ra_deg: Any,
    dec_deg: Any,
    zspec: Any,
    rank: Any,
    zphot: Any,
    zlow: Any,
    zhigh: Any,
    mags: Any,
    kpc_scale: float,
    max_family_span_kpc: float,
    min_pair_separation_arcsec: float,
    min_common_bands: int,
    image_photoz_max_dz_norm: float,
    family_photoz_delta_max: float,
) -> tuple[Any, ...]:
    left_idx = left_idx.astype("int32")
    right_idx = right_idx.astype("int32")
    ra_left = ra_deg[left_idx]
    ra_right = ra_deg[right_idx]
    dec_left = dec_deg[left_idx]
    dec_right = dec_deg[right_idx]

    ra1 = ra_left * (math.pi / 180.0)
    ra2 = ra_right * (math.pi / 180.0)
    dec1 = dec_left * (math.pi / 180.0)
    dec2 = dec_right * (math.pi / 180.0)
    sin_ddec = xp.sin((dec2 - dec1) / 2.0)
    sin_dra = xp.sin((ra2 - ra1) / 2.0)
    hav = sin_ddec**2 + xp.cos(dec1) * xp.cos(dec2) * sin_dra**2
    sep_rad = 2.0 * xp.arcsin(xp.minimum(1.0, xp.sqrt(xp.maximum(hav, 0.0))))
    separation_arcsec = sep_rad * ARCSEC_PER_RADIAN
    separation_kpc = separation_arcsec * kpc_scale
    finite_sep = xp.isfinite(separation_arcsec) & valid_pair
    too_close = (~finite_sep) | (separation_arcsec < min_pair_separation_arcsec)
    too_far = separation_kpc > max_family_span_kpc

    zspec_left = zspec[left_idx]
    zspec_right = zspec[right_idx]
    rank_left = rank[left_idx]
    rank_right = rank[right_idx]
    zphot_left = zphot[left_idx]
    zphot_right = zphot[right_idx]
    zlow_left = zlow[left_idx]
    zlow_right = zlow[right_idx]
    zhigh_left = zhigh[left_idx]
    zhigh_right = zhigh[right_idx]
    specphoto_score, specz_score, photoz_score, quality_score, relation_code, redshift_reject_code, zphot_delta = (
        _family_redshift_evidence_array(
            xp,
            zspec_left,
            zspec_right,
            rank_left,
            rank_right,
            zphot_left,
            zphot_right,
            zlow_left,
            zlow_right,
            zhigh_left,
            zhigh_right,
            image_photoz_max_dz_norm,
            family_photoz_delta_max,
        )
    )

    common = xp.isfinite(mags[left_idx]) & xp.isfinite(mags[right_idx])
    n_common = xp.sum(common, axis=1)
    insufficient = n_common < min_common_bands
    reject_code = xp.where(too_close, PAIR_REJECT_TOO_CLOSE, PAIR_REJECT_NONE)
    reject_code = xp.where((reject_code == PAIR_REJECT_NONE) & too_far, PAIR_REJECT_TOO_FAR, reject_code)
    reject_code = xp.where((reject_code == PAIR_REJECT_NONE) & (redshift_reject_code != PAIR_REJECT_NONE), redshift_reject_code, reject_code)
    reject_code = xp.where((reject_code == PAIR_REJECT_NONE) & insufficient, PAIR_REJECT_INSUFFICIENT_BANDS, reject_code)
    reject_code = xp.where(valid_pair, reject_code, PAIR_REJECT_TOO_CLOSE)

    separation_score = xp.maximum(0.60, 1.0 - 0.40 * (separation_kpc / max_family_span_kpc))
    score_upper_bound = (
        xp.maximum(specphoto_score, 1.0e-6) ** 0.60
        * xp.maximum(separation_score, 1.0e-6) ** 0.07
        * xp.maximum(quality_score, 1.0e-6) ** 0.03
    )
    n_common_out = xp.where((too_close | too_far | (redshift_reject_code != PAIR_REJECT_NONE)), 0, n_common)
    return (
        separation_arcsec,
        separation_kpc,
        specz_score,
        photoz_score,
        zphot_delta,
        n_common_out,
        reject_code,
        relation_code,
        score_upper_bound,
    )


@partial(jax.jit, static_argnames=("min_common_bands",))
def _prefilter_pair_batch_jax(
    left_idx: Any,
    right_idx: Any,
    valid_pair: Any,
    ra_deg: Any,
    dec_deg: Any,
    zspec: Any,
    rank: Any,
    zphot: Any,
    zlow: Any,
    zhigh: Any,
    mags: Any,
    kpc_scale: float,
    max_family_span_kpc: float,
    min_pair_separation_arcsec: float,
    min_common_bands: int,
    image_photoz_max_dz_norm: float,
    family_photoz_delta_max: float,
) -> tuple[Any, ...]:
    return _prefilter_pair_batch_array(
        jnp,
        left_idx,
        right_idx,
        valid_pair,
        ra_deg,
        dec_deg,
        zspec,
        rank,
        zphot,
        zlow,
        zhigh,
        mags,
        kpc_scale,
        max_family_span_kpc,
        min_pair_separation_arcsec,
        min_common_bands,
        image_photoz_max_dz_norm,
        family_photoz_delta_max,
    )


@partial(jax.jit, static_argnames=("min_common_bands",))
def _score_pair_batch_jax(
    left_idx: Any,
    right_idx: Any,
    valid_pair: Any,
    ra_deg: Any,
    dec_deg: Any,
    zspec: Any,
    rank: Any,
    zphot: Any,
    zlow: Any,
    zhigh: Any,
    redshift_conflict: Any,
    mags: Any,
    kpc_scale: float,
    max_family_span_kpc: float,
    min_pair_separation_arcsec: float,
    min_common_bands: int,
    image_photoz_max_dz_norm: float,
    family_color_rms_max: float,
    family_photoz_delta_max: float,
) -> tuple[Any, ...]:
    return _score_pair_batch_array(
        jnp,
        left_idx,
        right_idx,
        valid_pair,
        ra_deg,
        dec_deg,
        zspec,
        rank,
        zphot,
        zlow,
        zhigh,
        redshift_conflict,
        mags,
        kpc_scale,
        max_family_span_kpc,
        min_pair_separation_arcsec,
        min_common_bands,
        image_photoz_max_dz_norm,
        family_color_rms_max,
        family_photoz_delta_max,
    )


def _decode_pair_rows(
    spec: ClusterSpec,
    arrays: dict[str, Any],
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    selected: np.ndarray,
    scores: tuple[np.ndarray, ...],
) -> list[dict[str, Any]]:
    if not np.any(selected):
        return []
    (
        separation_arcsec,
        separation_kpc,
        pair_score,
        specz_score,
        photoz_score,
        zphot_delta,
        color_score,
        sed_rms,
        n_common,
        reject_code,
        relation_code,
        _score_upper_bound,
    ) = scores
    object_ids = arrays["object_ids"]
    rows: list[dict[str, Any]] = []
    for pos in np.flatnonzero(selected):
        reason_code = int(reject_code[pos])
        relation = int(relation_code[pos])
        redshift_reject = reason_code in {
            PAIR_REJECT_SPECZ_CONFLICT,
            PAIR_REJECT_MISSING_QUALIFIED_PHOTOZ,
            PAIR_REJECT_PHOTOZ_INCONSISTENT,
            PAIR_REJECT_MISSING_STRONG_SPECZ,
            PAIR_REJECT_PHOTOZ_DELTA,
        }
        rows.append(
            {
                "cluster_key": spec.key,
                "left_object_id": str(object_ids[int(left_idx[pos])]),
                "right_object_id": str(object_ids[int(right_idx[pos])]),
                "separation_arcsec": float(separation_arcsec[pos]),
                "separation_kpc": float(separation_kpc[pos]),
                "pair_score": float(pair_score[pos]),
                "specz_score": float(specz_score[pos]),
                "photoz_score": float(photoz_score[pos]),
                "zphot_delta": float(zphot_delta[pos]) if np.isfinite(zphot_delta[pos]) else float("nan"),
                "color_score": float(color_score[pos]),
                "sed_rms": float(sed_rms[pos]) if np.isfinite(sed_rms[pos]) else float("nan"),
                "n_common_bands": int(n_common[pos]),
                "hard_reject_reason": PAIR_REJECT_REASON_BY_CODE.get(reason_code, ""),
                "redshift_relation": PAIR_RELATION_BY_CODE.get(relation, ""),
            }
        )
    return rows


def _decode_prefilter_rows(
    spec: ClusterSpec,
    arrays: dict[str, Any],
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    selected: np.ndarray,
    prefilter: tuple[np.ndarray, ...],
) -> list[dict[str, Any]]:
    if not np.any(selected):
        return []
    (
        separation_arcsec,
        separation_kpc,
        specz_score,
        photoz_score,
        zphot_delta,
        n_common,
        reject_code,
        relation_code,
        _score_upper_bound,
    ) = prefilter
    object_ids = arrays["object_ids"]
    rows: list[dict[str, Any]] = []
    for pos in np.flatnonzero(selected):
        reason_code = int(reject_code[pos])
        relation = int(relation_code[pos])
        redshift_reject = reason_code in {
            PAIR_REJECT_SPECZ_CONFLICT,
            PAIR_REJECT_MISSING_QUALIFIED_PHOTOZ,
            PAIR_REJECT_PHOTOZ_INCONSISTENT,
            PAIR_REJECT_MISSING_STRONG_SPECZ,
            PAIR_REJECT_PHOTOZ_DELTA,
        }
        rows.append(
            {
                "cluster_key": spec.key,
                "left_object_id": str(object_ids[int(left_idx[pos])]),
                "right_object_id": str(object_ids[int(right_idx[pos])]),
                "separation_arcsec": float(separation_arcsec[pos]),
                "separation_kpc": float(separation_kpc[pos]),
                "pair_score": 0.0,
                "specz_score": 0.0 if redshift_reject else float(specz_score[pos]),
                "photoz_score": 0.0 if redshift_reject else float(photoz_score[pos]),
                "zphot_delta": float(zphot_delta[pos]) if np.isfinite(zphot_delta[pos]) else float("nan"),
                "color_score": 0.0,
                "sed_rms": float("nan"),
                "n_common_bands": int(n_common[pos]),
                "hard_reject_reason": PAIR_REJECT_REASON_BY_CODE.get(reason_code, ""),
                "redshift_relation": PAIR_RELATION_BY_CODE.get(relation, ""),
            }
        )
    return rows


def score_candidate_pairs(
    candidates: pd.DataFrame,
    spec: ClusterSpec,
    *,
    max_family_span_kpc: float = DEFAULT_MAX_FAMILY_SPAN_KPC,
    min_pair_separation_arcsec: float = DEFAULT_MIN_PAIR_SEPARATION_ARCSEC,
    min_common_bands: int = DEFAULT_MIN_COMMON_BANDS,
    pair_score_threshold: float = DEFAULT_PAIR_SCORE_THRESHOLD,
    family_pair_batch_size: int = DEFAULT_FAMILY_PAIR_BATCH_SIZE,
    family_pair_diagnostics: str = DEFAULT_FAMILY_PAIR_DIAGNOSTICS,
    image_photoz_max_dz_norm: float = DEFAULT_IMAGE_PHOTOZ_MAX_DZ_NORM,
    family_color_rms_max: float = DEFAULT_FAMILY_COLOR_RMS_MAX,
    family_photoz_delta_max: float = DEFAULT_FAMILY_PHOTOZ_DELTA_MAX,
    progress: Any | None = None,
) -> pd.DataFrame:
    diagnostics = str(family_pair_diagnostics).lower()
    if diagnostics not in {"scored", "accepted", "all"}:
        raise ValueError(f"Unsupported family pair diagnostics mode: {family_pair_diagnostics}")
    result = _score_candidate_pair_result_jax(
        candidates,
        spec,
        max_family_span_kpc=max_family_span_kpc,
        min_pair_separation_arcsec=min_pair_separation_arcsec,
        min_common_bands=min_common_bands,
        pair_score_threshold=pair_score_threshold,
        family_pair_batch_size=family_pair_batch_size,
        family_pair_diagnostics=diagnostics,
        image_photoz_max_dz_norm=image_photoz_max_dz_norm,
        family_color_rms_max=family_color_rms_max,
        family_photoz_delta_max=family_photoz_delta_max,
        progress=progress,
    )
    return result.pairs


def _score_candidate_pair_result_jax(
    candidates: pd.DataFrame,
    spec: ClusterSpec,
    *,
    max_family_span_kpc: float,
    min_pair_separation_arcsec: float,
    min_common_bands: int,
    pair_score_threshold: float,
    family_pair_batch_size: int,
    family_pair_diagnostics: str,
    image_photoz_max_dz_norm: float,
    family_color_rms_max: float,
    family_photoz_delta_max: float,
    progress: Any | None = None,
) -> PairScoreResult:
    start_time = time.perf_counter()
    if len(candidates) < 2:
        result = pd.DataFrame(columns=[*PAIR_COLUMNS, "redshift_relation"])
        metrics = _empty_pair_score_metrics("jax", 0, time.perf_counter() - start_time)
        result.attrs["pair_score_metrics"] = metrics
        result.attrs["accepted_pair_arrays"] = _empty_accepted_pair_arrays()
        return PairScoreResult(result, result.attrs["accepted_pair_arrays"], metrics)

    arrays = _candidate_pair_arrays(candidates)
    max_sep_arcsec = max_span_arcsec(spec, max_family_span_kpc)
    if progress is not None:
        progress.start_step(f"{spec.key}: finding nearby candidate pairs", total=None)
    pairs = _spatial_candidate_pairs(arrays, max_sep_arcsec)
    if progress is not None:
        progress.finish_step()
    n_spatial_pairs = int(len(pairs))
    if n_spatial_pairs == 0:
        result = pd.DataFrame(columns=[*PAIR_COLUMNS, "redshift_relation"])
        metrics = _empty_pair_score_metrics("jax", 0, time.perf_counter() - start_time)
        accepted_arrays = _empty_accepted_pair_arrays(arrays["object_ids"])
        result.attrs["pair_score_metrics"] = metrics
        result.attrs["accepted_pair_arrays"] = accepted_arrays
        return PairScoreResult(result, accepted_arrays, metrics)

    batch_size = max(1, int(family_pair_batch_size))
    rows: list[dict[str, Any]] = []
    accepted_chunks: list[dict[str, Any]] = []
    n_pruned_redshift = 0
    n_pruned_photoz_delta = 0
    n_pruned_score_upper_bound = 0
    n_pruned_common_bands = 0
    n_prefiltered_pairs = 0
    n_full_score_pairs = 0
    n_prefilter_redshift_rejects = 0
    n_prefilter_photoz_delta_rejects = 0
    kpc_scale = kpc_per_arcsec(spec.z_lens)
    diagnostics = family_pair_diagnostics

    device_arrays = {
        "ra": jnp.asarray(arrays["ra"], dtype=jnp.float64),
        "dec": jnp.asarray(arrays["dec"], dtype=jnp.float64),
        "zspec": jnp.asarray(arrays["zspec"], dtype=jnp.float64),
        "rank": jnp.asarray(arrays["rank"], dtype=jnp.float64),
        "zphot": jnp.asarray(arrays["zphot"], dtype=jnp.float64),
        "zlow": jnp.asarray(arrays["zlow"], dtype=jnp.float64),
        "zhigh": jnp.asarray(arrays["zhigh"], dtype=jnp.float64),
        "redshift_conflict": jnp.asarray(arrays["redshift_conflict"], dtype=jnp.bool_),
        "mags": jnp.asarray(arrays["mags"], dtype=jnp.float64),
    }

    if progress is not None:
        progress.start_step(f"{spec.key}: scoring candidate pairs", total=n_spatial_pairs)
    for start in range(0, n_spatial_pairs, batch_size):
        stop = min(start + batch_size, n_spatial_pairs)
        batch = pairs[start:stop]
        valid_count = int(len(batch))
        if valid_count < batch_size:
            padded = np.zeros((batch_size, 2), dtype=np.int64)
            padded[:valid_count] = batch
            batch = padded
        valid_pair = np.zeros(batch_size, dtype=bool)
        valid_pair[:valid_count] = True
        left_idx = batch[:, 0].astype(np.int32, copy=False)
        right_idx = batch[:, 1].astype(np.int32, copy=False)
        prefilter_tuple = _prefilter_pair_batch_jax(
            jnp.asarray(left_idx),
            jnp.asarray(right_idx),
            jnp.asarray(valid_pair),
            device_arrays["ra"],
            device_arrays["dec"],
            device_arrays["zspec"],
            device_arrays["rank"],
            device_arrays["zphot"],
            device_arrays["zlow"],
            device_arrays["zhigh"],
            device_arrays["mags"],
            float(kpc_scale),
            float(max_family_span_kpc),
            float(min_pair_separation_arcsec),
            int(min_common_bands),
            float(image_photoz_max_dz_norm),
            float(family_photoz_delta_max),
        )
        prefilter_tuple = tuple(np.asarray(jax.device_get(value)) for value in prefilter_tuple)

        pre_reject_code = prefilter_tuple[6]
        score_upper_bound = prefilter_tuple[8]
        valid = valid_pair
        redshift_reject = valid & np.isin(
            pre_reject_code,
            [
                PAIR_REJECT_SPECZ_CONFLICT,
                PAIR_REJECT_MISSING_QUALIFIED_PHOTOZ,
                PAIR_REJECT_PHOTOZ_INCONSISTENT,
                PAIR_REJECT_MISSING_STRONG_SPECZ,
                PAIR_REJECT_PHOTOZ_DELTA,
            ],
        )
        photoz_delta_reject = valid & (pre_reject_code == PAIR_REJECT_PHOTOZ_DELTA)
        common_reject = valid & (pre_reject_code == PAIR_REJECT_INSUFFICIENT_BANDS)
        score_prune = valid & (pre_reject_code == PAIR_REJECT_NONE) & (score_upper_bound < pair_score_threshold)
        conflict_pair = arrays["redshift_conflict"][left_idx] | arrays["redshift_conflict"][right_idx]
        color_fallback_reject = redshift_reject & conflict_pair & (
            (pre_reject_code == PAIR_REJECT_SPECZ_CONFLICT)
            | (pre_reject_code == PAIR_REJECT_PHOTOZ_INCONSISTENT)
            | (pre_reject_code == PAIR_REJECT_PHOTOZ_DELTA)
        )
        n_prefilter_redshift_rejects += int(redshift_reject.sum())
        n_prefilter_photoz_delta_rejects += int(photoz_delta_reject.sum())
        needs_full_score = (valid & (pre_reject_code == PAIR_REJECT_NONE) & (score_upper_bound >= pair_score_threshold)) | color_fallback_reject
        if diagnostics == "all":
            needs_full_score = color_fallback_reject | (
                valid
                & (
                    (pre_reject_code == PAIR_REJECT_NONE)
                    | (
                        (pre_reject_code != PAIR_REJECT_TOO_CLOSE)
                        & (pre_reject_code != PAIR_REJECT_TOO_FAR)
                        & (pre_reject_code != PAIR_REJECT_SPECZ_CONFLICT)
                        & (pre_reject_code != PAIR_REJECT_MISSING_QUALIFIED_PHOTOZ)
                        & (pre_reject_code != PAIR_REJECT_PHOTOZ_INCONSISTENT)
                        & (pre_reject_code != PAIR_REJECT_MISSING_STRONG_SPECZ)
                        & (pre_reject_code != PAIR_REJECT_PHOTOZ_DELTA)
                        & (pre_reject_code != PAIR_REJECT_INSUFFICIENT_BANDS)
                    )
                )
            )
        full_positions = np.flatnonzero(needs_full_score)
        n_full_score_pairs += int(len(full_positions))

        if diagnostics == "all":
            pre_selected = valid & ~needs_full_score
        elif diagnostics == "accepted":
            pre_selected = np.zeros_like(valid, dtype=bool)
        else:
            pre_selected = redshift_reject
        rows.extend(_decode_prefilter_rows(spec, arrays, left_idx, right_idx, pre_selected, prefilter_tuple))

        if len(full_positions) == 0:
            if diagnostics != "all":
                n_pruned_redshift += int(redshift_reject.sum()) if diagnostics == "accepted" else 0
                n_pruned_photoz_delta += int(photoz_delta_reject.sum()) if diagnostics == "accepted" else 0
                n_pruned_common_bands += int((common_reject & ~pre_selected).sum())
                n_pruned_score_upper_bound += int((score_prune & ~pre_selected).sum())
            n_prefiltered_pairs += int(pre_selected.sum())
            if progress is not None:
                progress.advance_step(valid_count)
            continue

        full_left_idx = np.zeros(batch_size, dtype=np.int32)
        full_right_idx = np.zeros(batch_size, dtype=np.int32)
        full_valid_pair = np.zeros(batch_size, dtype=bool)
        full_count = int(len(full_positions))
        full_left_idx[:full_count] = left_idx[full_positions]
        full_right_idx[:full_count] = right_idx[full_positions]
        full_valid_pair[:full_count] = True
        score_tuple = _score_pair_batch_jax(
            jnp.asarray(full_left_idx),
            jnp.asarray(full_right_idx),
            jnp.asarray(full_valid_pair),
            device_arrays["ra"],
            device_arrays["dec"],
            device_arrays["zspec"],
            device_arrays["rank"],
            device_arrays["zphot"],
            device_arrays["zlow"],
            device_arrays["zhigh"],
            device_arrays["redshift_conflict"],
            device_arrays["mags"],
            float(kpc_scale),
            float(max_family_span_kpc),
            float(min_pair_separation_arcsec),
            int(min_common_bands),
            float(image_photoz_max_dz_norm),
            float(family_color_rms_max),
            float(family_photoz_delta_max),
        )
        score_tuple = tuple(np.asarray(jax.device_get(value)) for value in score_tuple)

        reject_code = score_tuple[9]
        pair_score = score_tuple[2]
        if diagnostics == "all":
            selected = full_valid_pair
        elif diagnostics == "accepted":
            selected = full_valid_pair & (reject_code == PAIR_REJECT_NONE) & (pair_score >= pair_score_threshold)
        else:
            selected = full_valid_pair
        if diagnostics != "all":
            n_pruned_redshift += int(redshift_reject.sum()) if diagnostics == "accepted" else 0
            n_pruned_photoz_delta += int(photoz_delta_reject.sum()) if diagnostics == "accepted" else 0
            n_pruned_common_bands += int((common_reject & ~pre_selected).sum())
            n_pruned_score_upper_bound += int((score_prune & ~pre_selected).sum())
        accepted_selected = full_valid_pair & (reject_code == PAIR_REJECT_NONE) & (pair_score >= pair_score_threshold)
        n_prefiltered_pairs += int(pre_selected.sum()) + int(selected.sum())
        rows.extend(_decode_pair_rows(spec, arrays, full_left_idx, full_right_idx, selected, score_tuple))
        accepted_chunks.append(_accepted_arrays_from_scored_pairs(arrays, full_left_idx, full_right_idx, accepted_selected, score_tuple))
        if progress is not None:
            progress.advance_step(valid_count)
    if progress is not None:
        progress.finish_step()

    result = pd.DataFrame(rows, columns=[*PAIR_COLUMNS, "redshift_relation"])
    accepted_arrays = _concat_accepted_arrays(accepted_chunks, arrays["object_ids"])
    metrics = {
        "pair_score_backend": "jax",
        "n_spatial_pairs": n_spatial_pairs,
        "n_prefiltered_pairs": int(n_prefiltered_pairs),
        "n_scored_pairs": int(len(result)),
        "n_pruned_redshift": int(n_pruned_redshift),
        "n_pruned_photoz_delta": int(n_pruned_photoz_delta),
        "n_pruned_score_upper_bound": int(n_pruned_score_upper_bound),
        "n_pruned_common_bands": int(n_pruned_common_bands),
        "n_prefilter_full_score_pairs": int(n_full_score_pairs),
        "n_prefilter_redshift_rejects": int(n_prefilter_redshift_rejects),
        "n_prefilter_photoz_delta_rejects": int(n_prefilter_photoz_delta_rejects),
        "n_accepted_array_pairs": int(len(accepted_arrays["left_idx"])),
        "pair_score_seconds": float(time.perf_counter() - start_time),
    }
    result.attrs["pair_score_metrics"] = metrics
    result.attrs["accepted_pair_arrays"] = accepted_arrays
    return PairScoreResult(result, accepted_arrays, metrics)


def _pair_key(left_object_id: str, right_object_id: str) -> frozenset[str]:
    return frozenset((str(left_object_id), str(right_object_id)))


def _family_pair_rows(family_ids: Iterable[str], pair_lookup: dict[frozenset[str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(str(item) for item in family_ids), 2):
        pair = pair_lookup.get(_pair_key(left, right))
        if pair is not None:
            rows.append(pair)
    return rows


def _family_redshift_summary(family: set[str], by_object: dict[str, pd.Series]) -> tuple[float, str, float]:
    specz_values: list[float] = []
    specz_ranks: list[float] = []
    for object_id in family:
        row = by_object[object_id]
        zspec = valid_redshift(row.get("zspec_best"))
        rank = _to_float(row.get("zspec_best_confidence_rank"))
        if np.isfinite(zspec):
            specz_values.append(zspec)
            specz_ranks.append(rank)

    min_spec_rank = _finite_min(np.asarray(specz_ranks, dtype=float))
    if specz_values:
        return float(np.median(specz_values)), "specz_median", min_spec_rank
    return float("nan"), "no_redshift", min_spec_rank


def _family_probability(
    family: set[str],
    pair_rows: list[dict[str, Any]],
    by_object: dict[str, pd.Series],
    *,
    is_two_image: bool,
) -> tuple[float, str]:
    pair_scores = np.asarray([_to_float(row.get("pair_score")) for row in pair_rows], dtype=float)
    sed_rms = np.asarray([_to_float(row.get("sed_rms")) for row in pair_rows], dtype=float)
    min_pair = float(np.nanmin(pair_scores))
    median_pair = float(np.nanmedian(pair_scores))
    base = 0.55 * min_pair + 0.45 * median_pair

    specz_ranks: list[float] = []
    for object_id in family:
        row = by_object[object_id]
        zspec = valid_redshift(row.get("zspec_best"))
        rank = _to_float(row.get("zspec_best_confidence_rank"))
        if np.isfinite(zspec):
            specz_ranks.append(rank)

    flags: list[str] = []
    if is_two_image:
        flags.append("two_image_candidate")
    if not specz_ranks:
        cap = DEFAULT_PHOTOZ_ONLY_FAMILY_PROBABILITY_CAP
        flags.append("missing_specz")
    elif len(specz_ranks) < len(family):
        cap = FAMILY_INCOMPLETE_SPECZ_CAP
        flags.append("incomplete_specz")
    else:
        cap = FAMILY_COMPLETE_SPECZ_CAP
    if np.any(np.isfinite(sed_rms) & (sed_rms > FAMILY_COLOR_RMS_ACCEPTABLE)):
        flags.append("large_sed_residual")
        cap = min(cap, FAMILY_GENERIC_LARGE_SED_CAP)
    if min_pair < 0.5:
        flags.append("low_min_pair_score")
        cap = min(cap, FAMILY_GENERIC_LOW_MIN_PAIR_CAP)
    return float(np.clip(min(base, cap), 0.0, 0.98)), "|".join(flags)


def _family_redshift_summary_from_arrays(
    zspec_values: np.ndarray,
    rank_values: np.ndarray,
    zphot_values: np.ndarray,
) -> tuple[float, str, float]:
    specz_mask = _usable_family_spec_mask(zspec_values)
    min_spec_rank = _finite_min(rank_values[specz_mask])
    specz = zspec_values[specz_mask]
    if specz.size:
        return float(np.median(specz)), "specz_median", min_spec_rank
    return float("nan"), "no_redshift", min_spec_rank


def _family_probability_from_arrays(
    family_size: int,
    pair_scores: np.ndarray,
    sed_rms: np.ndarray,
    zspec_values: np.ndarray,
    rank_values: np.ndarray,
    *,
    is_two_image: bool,
) -> tuple[float, str]:
    min_pair = float(np.nanmin(pair_scores))
    median_pair = float(np.nanmedian(pair_scores))
    base = 0.55 * min_pair + 0.45 * median_pair
    specz_ranks = rank_values[_usable_family_spec_mask(zspec_values)]
    flags: list[str] = []
    if is_two_image:
        flags.append("two_image_candidate")
    if specz_ranks.size == 0:
        cap = DEFAULT_PHOTOZ_ONLY_FAMILY_PROBABILITY_CAP
        flags.append("missing_specz")
    elif len(specz_ranks) < family_size:
        cap = FAMILY_INCOMPLETE_SPECZ_CAP
        flags.append("incomplete_specz")
    else:
        cap = FAMILY_COMPLETE_SPECZ_CAP
    if np.any(np.isfinite(sed_rms) & (sed_rms > FAMILY_COLOR_RMS_ACCEPTABLE)):
        flags.append("large_sed_residual")
        cap = min(cap, FAMILY_GENERIC_LARGE_SED_CAP)
    if min_pair < 0.5:
        flags.append("low_min_pair_score")
        cap = min(cap, FAMILY_GENERIC_LOW_MIN_PAIR_CAP)
    return float(np.clip(min(base, cap), 0.0, 0.98)), "|".join(flags)


def _photoz_only_family_probability_from_arrays(
    pair_scores: np.ndarray,
    sed_rms: np.ndarray,
) -> tuple[float, str]:
    min_pair = float(np.nanmin(pair_scores))
    median_pair = float(np.nanmedian(pair_scores))
    base = 0.55 * min_pair + 0.45 * median_pair
    cap = DEFAULT_PHOTOZ_ONLY_FAMILY_PROBABILITY_CAP
    flags = ["photoz_only_anchor", "missing_specz", "photoz_only_candidate"]
    if np.any(np.isfinite(sed_rms) & (sed_rms > FAMILY_COLOR_RMS_ACCEPTABLE)):
        flags.append("large_sed_residual")
        cap = min(cap, FAMILY_PHOTOZ_ONLY_LARGE_SED_CAP)
    if min_pair < 0.5:
        flags.append("low_min_pair_score")
        cap = min(cap, FAMILY_PHOTOZ_ONLY_LOW_MIN_PAIR_CAP)
    return float(np.clip(min(base, cap), 0.0, cap)), "|".join(flags)


def _pair_object_key(left: Any, right: Any) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def _candidate_pairwise_separations(
    candidates: pd.DataFrame,
    candidate_indices: np.ndarray,
    spec: ClusterSpec,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(candidate_indices, dtype=np.int32)
    if indices.size < 2:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    rows = candidates.iloc[indices]
    x_values = pd.to_numeric(rows.get("image_family_x_kpc", pd.Series(np.nan, index=rows.index)), errors="coerce").to_numpy(
        dtype=float
    )
    y_values = pd.to_numeric(rows.get("image_family_y_kpc", pd.Series(np.nan, index=rows.index)), errors="coerce").to_numpy(
        dtype=float
    )
    sep_kpc: list[float] = []
    sep_arcsec: list[float] = []
    scale = kpc_per_arcsec(spec.z_lens)
    if np.isfinite(x_values).all() and np.isfinite(y_values).all():
        for left_pos in range(indices.size):
            for right_pos in range(left_pos + 1, indices.size):
                separation_kpc = float(np.hypot(x_values[left_pos] - x_values[right_pos], y_values[left_pos] - y_values[right_pos]))
                sep_kpc.append(separation_kpc)
                sep_arcsec.append(separation_kpc / scale if scale > 0.0 else float("nan"))
        return np.asarray(sep_arcsec, dtype=float), np.asarray(sep_kpc, dtype=float)

    ra_values = pd.to_numeric(rows["ra"], errors="coerce").to_numpy(dtype=float)
    dec_values = pd.to_numeric(rows["dec"], errors="coerce").to_numpy(dtype=float)
    coords = SkyCoord(ra_values * u.deg, dec_values * u.deg)
    for left_pos in range(indices.size):
        for right_pos in range(left_pos + 1, indices.size):
            separation_arcsec = float(coords[left_pos].separation(coords[right_pos]).to(u.arcsec).value)
            sep_arcsec.append(separation_arcsec)
            sep_kpc.append(separation_arcsec * scale)
    return np.asarray(sep_arcsec, dtype=float), np.asarray(sep_kpc, dtype=float)


def _anchored_family_probability(
    pair_scores: np.ndarray,
    sed_rms: np.ndarray,
    max_separation_kpc: float,
    max_family_span_kpc: float,
    flags: list[str],
) -> tuple[float, str]:
    finite_scores = pair_scores[np.isfinite(pair_scores)]
    if finite_scores.size == 0:
        return 0.0, "|".join([*flags, "missing_anchor_pair_scores"])
    min_pair = float(np.nanmin(finite_scores))
    median_pair = float(np.nanmedian(finite_scores))
    base = 0.55 * min_pair + 0.45 * median_pair
    cap = FAMILY_ANCHORED_CAP
    if np.any(np.isfinite(sed_rms) & (sed_rms > FAMILY_COLOR_RMS_ACCEPTABLE)):
        flags.append("large_sed_residual")
        cap = min(cap, FAMILY_ANCHORED_LARGE_SED_OR_SPAN_CAP)
    if np.isfinite(max_separation_kpc) and max_separation_kpc > 0.8 * float(max_family_span_kpc):
        flags.append("large_family_span")
        cap = min(cap, FAMILY_ANCHORED_LARGE_SED_OR_SPAN_CAP)
    if min_pair < 0.5:
        flags.append("low_min_pair_score")
        cap = min(cap, FAMILY_ANCHORED_LOW_MIN_PAIR_CAP)
    return float(np.clip(min(base, cap), 0.0, cap)), "|".join(dict.fromkeys(flags))


def _single_specz_anchor_family_summaries(
    candidates: pd.DataFrame,
    accepted: pd.DataFrame,
    pairs: pd.DataFrame,
    spec: ClusterSpec,
    *,
    pair_score_threshold: float,
    family_probability_threshold: float,
    max_family_span_kpc: float,
    family_photoz_delta_max: float,
) -> list[dict[str, Any]]:
    if candidates.empty or accepted.empty:
        return []
    candidate_object_ids = candidates["object_id"].astype(str).to_numpy()
    object_to_index = {object_id: index for index, object_id in enumerate(candidate_object_ids)}
    zspec = _valid_redshift_array(candidates["zspec_best"]) if "zspec_best" in candidates.columns else np.full(len(candidates), np.nan)
    rank = (
        pd.to_numeric(candidates["zspec_best_confidence_rank"], errors="coerce").to_numpy(dtype=float)
        if "zspec_best_confidence_rank" in candidates.columns
        else np.full(len(candidates), np.nan)
    )
    zphot = (
        _valid_redshift_array(candidates["image_zphot_family"])
        if "image_zphot_family" in candidates.columns
        else np.full(len(candidates), np.nan)
    )
    usable_spec = _usable_family_spec_mask(zspec)
    photoz_companion = ~usable_spec & np.isfinite(zphot)

    pair_lookup = {
        _pair_object_key(row.left_object_id, row.right_object_id): row
        for row in pairs.itertuples(index=False)
    }
    edges_by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    accepted_specphoto = accepted.loc[accepted["redshift_relation"].astype(str).eq("specz_photoz")]
    for row in accepted_specphoto.itertuples(index=False):
        left_object_id = str(row.left_object_id)
        right_object_id = str(row.right_object_id)
        if left_object_id not in object_to_index or right_object_id not in object_to_index:
            continue
        left_index = object_to_index[left_object_id]
        right_index = object_to_index[right_object_id]
        left_spec = bool(usable_spec[left_index])
        right_spec = bool(usable_spec[right_index])
        if left_spec == right_spec:
            continue
        anchor_object_id, companion_object_id = (left_object_id, right_object_id) if left_spec else (right_object_id, left_object_id)
        companion_index = object_to_index[companion_object_id]
        if not bool(photoz_companion[companion_index]):
            continue
        pair_score = _to_float(row.pair_score)
        if not np.isfinite(pair_score) or pair_score < pair_score_threshold:
            continue
        edges_by_anchor[anchor_object_id].append(
            {
                "anchor_object_id": anchor_object_id,
                "companion_object_id": companion_object_id,
                "pair_score": pair_score,
                "sed_rms": _to_float(row.sed_rms),
                "separation_arcsec": _to_float(row.separation_arcsec),
                "separation_kpc": _to_float(row.separation_kpc),
                "zphot_delta": _to_float(getattr(row, "zphot_delta", np.nan)),
            }
        )

    summaries: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for anchor_object_id, edges in edges_by_anchor.items():
        if len(edges) < 2:
            continue
        edges = sorted(edges, key=lambda edge: (-float(edge["pair_score"]), str(edge["companion_object_id"])))
        anchor_index = object_to_index[anchor_object_id]
        for seed_edge in edges:
            selected_edges: list[dict[str, Any]] = [seed_edge]
            selected_object_ids = {anchor_object_id, str(seed_edge["companion_object_id"])}
            for edge in edges:
                companion_object_id = str(edge["companion_object_id"])
                if companion_object_id in selected_object_ids:
                    continue
                trial_object_ids = selected_object_ids | {companion_object_id}
                trial_indices = np.asarray([object_to_index[object_id] for object_id in sorted(trial_object_ids)], dtype=np.int32)
                _trial_arcsec, trial_kpc = _candidate_pairwise_separations(candidates, trial_indices, spec)
                if trial_kpc.size and float(np.nanmax(trial_kpc)) > max_family_span_kpc:
                    continue
                selected_edges.append(edge)
                selected_object_ids.add(companion_object_id)

            if len(selected_object_ids) < 3:
                continue
            family_key = tuple(sorted(selected_object_ids))
            if family_key in seen:
                continue
            seen.add(family_key)
            candidate_indices = np.asarray([object_to_index[object_id] for object_id in family_key], dtype=np.int32)
            separations_arcsec, separations_kpc = _candidate_pairwise_separations(candidates, candidate_indices, spec)
            if separations_kpc.size == 0 or float(np.nanmax(separations_kpc)) > max_family_span_kpc:
                continue
            pair_scores = np.asarray([float(edge["pair_score"]) for edge in selected_edges], dtype=float)
            anchor_sed_rms = np.asarray([_to_float(edge["sed_rms"]) for edge in selected_edges], dtype=float)
            flags = ["single_specz_anchor", "incomplete_specz", "photo_photo_diagnostic_only"]
            companion_ids = sorted(selected_object_ids - {anchor_object_id})
            photo_photo_sed_rms: list[float] = []
            companion_indices = np.asarray([object_to_index[object_id] for object_id in companion_ids], dtype=np.int32)
            companion_zphot = zphot[companion_indices]
            companion_zphot_deltas = [
                abs(float(left) - float(right))
                for left, right in itertools.combinations(companion_zphot, 2)
                if np.isfinite(left) and np.isfinite(right)
            ]
            max_companion_zphot_delta = (
                float(np.max(np.asarray(companion_zphot_deltas, dtype=float))) if companion_zphot_deltas else float("nan")
            )
            if np.isfinite(max_companion_zphot_delta) and not (max_companion_zphot_delta < float(family_photoz_delta_max)):
                continue
            if np.isfinite(max_companion_zphot_delta) and max_companion_zphot_delta >= 0.8 * float(family_photoz_delta_max):
                flags.append("photoz_delta_near_limit")
            for left_pos, left_object_id in enumerate(companion_ids):
                for right_object_id in companion_ids[left_pos + 1 :]:
                    photo_pair = pair_lookup.get(_pair_object_key(left_object_id, right_object_id))
                    if photo_pair is None:
                        flags.append("missing_photo_photo_diagnostic")
                        continue
                    photo_sed = _to_float(getattr(photo_pair, "sed_rms", np.nan))
                    if np.isfinite(photo_sed):
                        photo_photo_sed_rms.append(photo_sed)
                    if str(getattr(photo_pair, "hard_reject_reason", "")) == "color_rms_too_large":
                        flags.append("photo_photo_large_sed_residual")
            all_sed_rms = (
                np.concatenate([anchor_sed_rms, np.asarray(photo_photo_sed_rms, dtype=float)])
                if photo_photo_sed_rms
                else anchor_sed_rms
            )
            if np.any(np.asarray(photo_photo_sed_rms, dtype=float) > FAMILY_COLOR_RMS_ACCEPTABLE):
                flags.append("photo_photo_large_sed_residual")
            probability, review_flags = _anchored_family_probability(
                pair_scores,
                all_sed_rms,
                float(np.nanmax(separations_kpc)),
                max_family_span_kpc,
                flags,
            )
            if probability < family_probability_threshold:
                continue
            summaries.append(
                {
                    "source": "single_specz_anchor",
                    "family": set(family_key),
                    "candidate_indices": candidate_indices,
                    "probability": probability,
                    "pair_scores": pair_scores,
                    "sed_rms": all_sed_rms,
                    "separations_arcsec": separations_arcsec,
                    "separations_kpc": separations_kpc,
                    "family_z_best": float(zspec[anchor_index]),
                    "family_z_method": "single_specz_anchor",
                    "min_specz_confidence": float(rank[anchor_index]),
                    "review_flags": review_flags,
                    "anchor_object_id": anchor_object_id,
                    "anchor_edge_scores": {
                        str(edge["companion_object_id"]): float(edge["pair_score"])
                        for edge in selected_edges
                    },
                }
            )
    return summaries


def _empty_family_growth_metrics(elapsed: float = 0.0) -> dict[str, Any]:
    return {
        "family_growth_backend": "jax_dense",
        "family_growth_seconds": float(elapsed),
        "n_family_growth_objects": 0,
        "n_family_growth_seed_edges": 0,
        "family_growth_n_batches": 0,
        "family_growth_unique_masks": 0,
        "family_growth_packed_mask_bytes": 0,
        "family_growth_all_seed_batch": False,
        "n_single_specz_anchor_family_candidates": 0,
        "n_single_specz_anchor_families": 0,
        "n_photoz_only_families": 0,
    }


def _attach_family_growth_metrics(
    families: pd.DataFrame,
    members: pd.DataFrame,
    metrics: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    families.attrs["family_growth_metrics"] = metrics
    members.attrs["family_growth_metrics"] = metrics
    return families, members


@jax.jit
def _grow_complete_link_family_batch_jax(
    score_matrix: Any,
    adjacency_matrix: Any,
    seed_left: Any,
    seed_right: Any,
    valid_seed: Any,
) -> Any:
    n_objects = score_matrix.shape[0]
    left_mask = jax.nn.one_hot(seed_left, n_objects, dtype=jnp.bool_) & valid_seed[:, None]
    right_mask = jax.nn.one_hot(seed_right, n_objects, dtype=jnp.bool_) & valid_seed[:, None]
    family_mask = left_mask | right_mask
    candidate_mask = adjacency_matrix[seed_left] & adjacency_matrix[seed_right] & ~family_mask & valid_seed[:, None]
    score_sum = score_matrix[seed_left] + score_matrix[seed_right]
    family_size = jnp.where(valid_seed, 2, 1).astype(jnp.float32)
    step = jnp.asarray(0, dtype=jnp.int32)

    def cond_fn(state: tuple[Any, Any, Any, Any, Any]) -> Any:
        _family_mask, current_candidates, _score_sum, _family_size, current_step = state
        return jnp.any(current_candidates) & (current_step < n_objects)

    def body_fn(state: tuple[Any, Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any, Any]:
        current_family, current_candidates, current_score_sum, current_size, current_step = state
        mean_scores = current_score_sum / current_size[:, None]
        candidate_scores = jnp.where(current_candidates, mean_scores, -jnp.inf)
        best_scores = jnp.max(candidate_scores, axis=1)
        has_choice = jnp.isfinite(best_scores)
        chosen = jnp.argmax(candidate_scores, axis=1)
        chosen_mask = jax.nn.one_hot(chosen, n_objects, dtype=jnp.bool_) & has_choice[:, None]
        next_family = current_family | chosen_mask
        next_size = current_size + has_choice.astype(jnp.float32)
        chosen_scores = score_matrix[chosen]
        next_score_sum = current_score_sum + jnp.where(has_choice[:, None], chosen_scores, 0.0)
        next_candidates = current_candidates & adjacency_matrix[chosen] & ~next_family & has_choice[:, None]
        return next_family, next_candidates, next_score_sum, next_size, current_step + 1

    final_family, _candidate_mask, _score_sum, _family_size, _step = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (family_mask, candidate_mask, score_sum, family_size, step),
    )
    final_family = final_family & valid_seed[:, None]
    final_size = jnp.sum(final_family, axis=1).astype(jnp.int32)
    return jnp.packbits(final_family, axis=1), final_size


def _build_dense_growth_inputs(accepted: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    object_ids = list(
        dict.fromkeys(
            [
                *accepted["left_object_id"].astype(str).tolist(),
                *accepted["right_object_id"].astype(str).tolist(),
            ]
        )
    )
    if len(object_ids) > DEFAULT_FAMILY_GROWTH_MAX_OBJECTS:
        raise RuntimeError(
            "Accepted image-family graph has "
            f"{len(object_ids):,} objects, exceeding dense JAX family-growth limit "
            f"{DEFAULT_FAMILY_GROWTH_MAX_OBJECTS:,}. Increase pair thresholds or reduce candidate density."
        )
    object_index = {object_id: idx for idx, object_id in enumerate(object_ids)}
    left_idx = accepted["left_object_id"].astype(str).map(object_index).to_numpy(dtype=np.int32)
    right_idx = accepted["right_object_id"].astype(str).map(object_index).to_numpy(dtype=np.int32)
    scores = pd.to_numeric(accepted["pair_score"], errors="coerce").to_numpy(dtype=np.float32)
    n_objects = len(object_ids)
    score_matrix = np.zeros((n_objects, n_objects), dtype=np.float32)
    adjacency_matrix = np.zeros((n_objects, n_objects), dtype=bool)
    score_matrix[left_idx, right_idx] = scores
    score_matrix[right_idx, left_idx] = scores
    adjacency_matrix[left_idx, right_idx] = True
    adjacency_matrix[right_idx, left_idx] = True
    return object_ids, left_idx, right_idx, score_matrix, adjacency_matrix


def _build_dense_growth_inputs_from_arrays(accepted_arrays: dict[str, Any]) -> dict[str, Any]:
    left_original = np.asarray(accepted_arrays["left_idx"], dtype=np.int32)
    right_original = np.asarray(accepted_arrays["right_idx"], dtype=np.int32)
    source_object_ids = np.asarray(accepted_arrays["object_ids"], dtype=object)
    compact_candidate_indices = np.asarray(
        list(dict.fromkeys([*left_original.astype(int).tolist(), *right_original.astype(int).tolist()])),
        dtype=np.int32,
    )
    if len(compact_candidate_indices) > DEFAULT_FAMILY_GROWTH_MAX_OBJECTS:
        raise RuntimeError(
            "Accepted image-family graph has "
            f"{len(compact_candidate_indices):,} objects, exceeding dense JAX family-growth limit "
            f"{DEFAULT_FAMILY_GROWTH_MAX_OBJECTS:,}. Increase pair thresholds or reduce candidate density."
        )
    compact_lookup = np.full(len(source_object_ids), -1, dtype=np.int32)
    compact_lookup[compact_candidate_indices] = np.arange(len(compact_candidate_indices), dtype=np.int32)
    left_idx = compact_lookup[left_original]
    right_idx = compact_lookup[right_original]
    scores = np.asarray(accepted_arrays["pair_score"], dtype=np.float32)
    n_objects = len(compact_candidate_indices)
    score_matrix = np.zeros((n_objects, n_objects), dtype=np.float32)
    adjacency_matrix = np.zeros((n_objects, n_objects), dtype=bool)
    sed_rms_matrix = np.full((n_objects, n_objects), np.nan, dtype=np.float32)
    sep_arcsec_matrix = np.full((n_objects, n_objects), np.nan, dtype=np.float32)
    sep_kpc_matrix = np.full((n_objects, n_objects), np.nan, dtype=np.float32)
    score_matrix[left_idx, right_idx] = scores
    score_matrix[right_idx, left_idx] = scores
    adjacency_matrix[left_idx, right_idx] = True
    adjacency_matrix[right_idx, left_idx] = True
    sed_rms = np.asarray(accepted_arrays["sed_rms"], dtype=np.float32)
    sep_arcsec = np.asarray(accepted_arrays["separation_arcsec"], dtype=np.float32)
    sep_kpc = np.asarray(accepted_arrays["separation_kpc"], dtype=np.float32)
    sed_rms_matrix[left_idx, right_idx] = sed_rms
    sed_rms_matrix[right_idx, left_idx] = sed_rms
    sep_arcsec_matrix[left_idx, right_idx] = sep_arcsec
    sep_arcsec_matrix[right_idx, left_idx] = sep_arcsec
    sep_kpc_matrix[left_idx, right_idx] = sep_kpc
    sep_kpc_matrix[right_idx, left_idx] = sep_kpc
    return {
        "object_ids": source_object_ids[compact_candidate_indices].astype(str).tolist(),
        "compact_candidate_indices": compact_candidate_indices,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "score_matrix": score_matrix,
        "adjacency_matrix": adjacency_matrix,
        "sed_rms_matrix": sed_rms_matrix,
        "separation_arcsec_matrix": sep_arcsec_matrix,
        "separation_kpc_matrix": sep_kpc_matrix,
        "pair_score": scores,
        "specz_score": np.asarray(accepted_arrays["specz_score"], dtype=np.float32),
        "color_score": np.asarray(accepted_arrays["color_score"], dtype=np.float32),
    }


def _family_growth_batch_size(n_seed_edges: int, n_objects: int) -> tuple[int, bool]:
    if n_seed_edges <= 0:
        return 1, False
    safe_cells = max(1, int(DEFAULT_FAMILY_GROWTH_MAX_BATCH_CELLS))
    if n_objects <= 0 or n_seed_edges * n_objects <= safe_cells:
        return max(1, n_seed_edges), True
    return max(1, min(n_seed_edges, safe_cells // max(1, n_objects))), False


def _decode_packed_families(packed_masks: np.ndarray, object_ids: list[str]) -> list[set[str]]:
    if packed_masks.size == 0:
        return []
    decoded = np.unpackbits(packed_masks, axis=1, count=len(object_ids)).astype(bool, copy=False)
    families: list[set[str]] = []
    for mask in decoded:
        families.append({object_ids[int(index)] for index in np.flatnonzero(mask)})
    return families


def _grow_complete_link_family_result_jax(
    accepted_arrays: dict[str, Any],
    *,
    two_image_score_threshold: float,
    progress: Any | None = None,
    progress_label: str = "growing complete-link families",
) -> FamilyGrowthResult:
    start_time = time.perf_counter()
    if len(accepted_arrays.get("left_idx", [])) == 0:
        metrics = _empty_family_growth_metrics()
        return FamilyGrowthResult(
            [],
            np.empty((0, 0), dtype=bool),
            [],
            np.empty(0, dtype=np.int32),
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            metrics,
        )
    order = np.argsort(-np.asarray(accepted_arrays["pair_score"], dtype=np.float32), kind="mergesort")
    sorted_arrays = {
        key: (np.asarray(value)[order] if key != "object_ids" else value)
        for key, value in accepted_arrays.items()
    }
    dense = _build_dense_growth_inputs_from_arrays(sorted_arrays)
    object_ids = dense["object_ids"]
    left_idx = dense["left_idx"]
    right_idx = dense["right_idx"]
    score_matrix = dense["score_matrix"]
    adjacency_matrix = dense["adjacency_matrix"]
    n_seed_edges = int(len(left_idx))
    batch_size, all_seed_batch = _family_growth_batch_size(n_seed_edges, len(object_ids))
    score_matrix_jax = jnp.asarray(score_matrix)
    adjacency_matrix_jax = jnp.asarray(adjacency_matrix)
    pair_score_values = np.asarray(sorted_arrays["pair_score"], dtype=np.float32)
    specz_score_values = np.asarray(sorted_arrays["specz_score"], dtype=np.float32)
    sed_rms_values = np.asarray(sorted_arrays["sed_rms"], dtype=np.float32)

    packed_chunks: list[np.ndarray] = []
    n_batches = 0
    if progress is not None:
        progress.start_step(progress_label, total=n_seed_edges)
    for start in range(0, n_seed_edges, batch_size):
        n_batches += 1
        stop = min(start + batch_size, n_seed_edges)
        valid_count = stop - start
        seed_left = np.zeros(batch_size, dtype=np.int32)
        seed_right = np.zeros(batch_size, dtype=np.int32)
        valid_seed = np.zeros(batch_size, dtype=bool)
        seed_left[:valid_count] = left_idx[start:stop]
        seed_right[:valid_count] = right_idx[start:stop]
        valid_seed[:valid_count] = True
        packed_masks, family_size = jax.device_get(
            _grow_complete_link_family_batch_jax(
                score_matrix_jax,
                adjacency_matrix_jax,
                jnp.asarray(seed_left),
                jnp.asarray(seed_right),
                jnp.asarray(valid_seed),
            )
        )
        packed_masks = np.asarray(packed_masks)[:valid_count]
        family_size = np.asarray(family_size)[:valid_count]
        two_image = family_size == 2
        keep = family_size >= 2
        keep &= ~(two_image & (pair_score_values[start:stop] < float(two_image_score_threshold)))
        keep &= ~(two_image & ((specz_score_values[start:stop] < 0.70) | (sed_rms_values[start:stop] > FAMILY_COLOR_RMS_ACCEPTABLE)))
        if np.any(keep):
            batch_unique, batch_first_indices = np.unique(packed_masks[keep], axis=0, return_index=True)
            packed_chunks.append(batch_unique[np.argsort(batch_first_indices, kind="mergesort")])
        if progress is not None:
            progress.advance_step(valid_count)
    if progress is not None:
        progress.finish_step()
    if packed_chunks:
        packed_candidates = np.vstack(packed_chunks)
        unique_packed, first_indices = np.unique(packed_candidates, axis=0, return_index=True)
        unique_packed = unique_packed[np.argsort(first_indices, kind="mergesort")]
    else:
        packed_width = int(math.ceil(len(object_ids) / 8))
        unique_packed = np.empty((0, packed_width), dtype=np.uint8)
    family_masks = np.unpackbits(unique_packed, axis=1, count=len(object_ids)).astype(bool, copy=False) if unique_packed.size else np.empty((0, len(object_ids)), dtype=bool)
    family_sets = _decode_packed_families(unique_packed, object_ids)
    metrics = {
        "family_growth_backend": "jax_dense",
        "family_growth_seconds": float(time.perf_counter() - start_time),
        "n_family_growth_objects": int(len(object_ids)),
        "n_family_growth_seed_edges": n_seed_edges,
        "family_growth_n_batches": int(n_batches),
        "family_growth_unique_masks": int(len(unique_packed)),
        "family_growth_packed_mask_bytes": int(unique_packed.nbytes),
        "family_growth_all_seed_batch": bool(all_seed_batch),
    }
    return FamilyGrowthResult(
        family_sets,
        family_masks,
        object_ids,
        dense["compact_candidate_indices"],
        dense["score_matrix"],
        dense["sed_rms_matrix"],
        dense["separation_arcsec_matrix"],
        dense["separation_kpc_matrix"],
        metrics,
    )


def _grow_complete_link_families_jax(
    accepted: pd.DataFrame,
    *,
    two_image_score_threshold: float,
    progress: Any | None = None,
    progress_label: str = "growing complete-link families",
) -> tuple[list[set[str]], dict[str, Any]]:
    if accepted.empty:
        return [], _empty_family_growth_metrics()
    accepted_arrays = _accepted_arrays_from_pair_dataframe(
        pd.DataFrame({"object_id": pd.unique(pd.concat([accepted["left_object_id"], accepted["right_object_id"]]).astype(str))}),
        accepted,
    )
    result = _grow_complete_link_family_result_jax(
        accepted_arrays,
        two_image_score_threshold=two_image_score_threshold,
        progress=progress,
        progress_label=progress_label,
    )
    return result.families, result.metrics


def build_families_from_pairs(
    candidates: pd.DataFrame,
    pairs: pd.DataFrame,
    spec: ClusterSpec,
    *,
    pair_score_threshold: float = DEFAULT_PAIR_SCORE_THRESHOLD,
    two_image_score_threshold: float = DEFAULT_TWO_IMAGE_SCORE_THRESHOLD,
    family_probability_threshold: float = DEFAULT_FAMILY_PROBABILITY_THRESHOLD,
    max_family_span_kpc: float = DEFAULT_MAX_FAMILY_SPAN_KPC,
    family_photoz_delta_max: float = DEFAULT_FAMILY_PHOTOZ_DELTA_MAX,
    max_families_per_object: int = DEFAULT_MAX_FAMILIES_PER_OBJECT,
    progress: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty or pairs.empty:
        return _attach_family_growth_metrics(
            pd.DataFrame(columns=FAMILY_COLUMNS),
            pd.DataFrame(columns=MEMBER_COLUMNS),
            _empty_family_growth_metrics(),
        )

    accepted = pairs.loc[
        (pairs["hard_reject_reason"].fillna("") == "")
        & (pd.to_numeric(pairs["pair_score"], errors="coerce") >= pair_score_threshold)
    ].copy()
    if accepted.empty:
        return _attach_family_growth_metrics(
            pd.DataFrame(columns=FAMILY_COLUMNS),
            pd.DataFrame(columns=MEMBER_COLUMNS),
            _empty_family_growth_metrics(),
        )

    accepted_arrays = pairs.attrs.get("accepted_pair_arrays")
    if not isinstance(accepted_arrays, dict) or len(accepted_arrays.get("left_idx", [])) != len(accepted):
        accepted_arrays = _accepted_arrays_from_pair_dataframe(candidates, accepted)
    growth = _grow_complete_link_family_result_jax(
        accepted_arrays,
        two_image_score_threshold=two_image_score_threshold,
        progress=progress,
        progress_label=f"{spec.key}: growing complete-link families",
    )
    growth_metrics = growth.metrics
    candidates_reset = candidates.reset_index(drop=True)
    candidate_zspec = _valid_redshift_array(candidates_reset["zspec_best"]) if "zspec_best" in candidates_reset.columns else np.full(len(candidates_reset), np.nan)
    candidate_rank = (
        pd.to_numeric(candidates_reset["zspec_best_confidence_rank"], errors="coerce").to_numpy(dtype=float)
        if "zspec_best_confidence_rank" in candidates_reset.columns
        else np.full(len(candidates_reset), np.nan)
    )
    candidate_zphot = (
        _valid_redshift_array(candidates_reset["image_zphot_family"])
        if "image_zphot_family" in candidates_reset.columns
        else np.full(len(candidates_reset), np.nan)
    )
    candidate_redshift_conflict = (
        candidates_reset["image_family_review_flags"].astype(str).str.contains("zspec_conflict_candidate", na=False).to_numpy(dtype=bool)
        if "image_family_review_flags" in candidates_reset.columns
        else np.zeros(len(candidates_reset), dtype=bool)
    )

    candidate_summaries: list[dict[str, Any]] = []
    if progress is not None:
        progress.start_step(f"{spec.key}: summarizing candidate families", total=len(growth.families))
    for family_index, family in enumerate(growth.families):
        compact_idx = np.flatnonzero(growth.family_masks[family_index])
        if compact_idx.size < 2:
            if progress is not None:
                progress.advance_step()
            continue
        tri = np.triu_indices(int(compact_idx.size), k=1)
        pair_scores = growth.pair_score_matrix[np.ix_(compact_idx, compact_idx)][tri]
        pair_scores = pair_scores[np.isfinite(pair_scores) & (pair_scores > 0.0)]
        if pair_scores.size == 0:
            if progress is not None:
                progress.advance_step()
            continue
        sed_rms = growth.sed_rms_matrix[np.ix_(compact_idx, compact_idx)][tri]
        separations_arcsec = growth.separation_arcsec_matrix[np.ix_(compact_idx, compact_idx)][tri]
        separations_kpc = growth.separation_kpc_matrix[np.ix_(compact_idx, compact_idx)][tri]
        candidate_indices = growth.compact_candidate_indices[compact_idx]
        zspec_values = candidate_zspec[candidate_indices]
        rank_values = candidate_rank[candidate_indices]
        zphot_values = candidate_zphot[candidate_indices]
        conflict_values = candidate_redshift_conflict[candidate_indices]
        is_two_image = len(family) == 2
        usable_spec_mask = _usable_family_spec_mask(zspec_values)
        usable_spec_values = zspec_values[usable_spec_mask]
        color_guided_spec_conflict = False
        if usable_spec_values.size == 0:
            valid_photoz = zphot_values[np.isfinite(zphot_values)]
            if is_two_image or len(family) < 3 or valid_photoz.size != len(family):
                if progress is not None:
                    progress.advance_step()
                continue
            if float(np.nanmax(valid_photoz) - np.nanmin(valid_photoz)) >= float(family_photoz_delta_max):
                if progress is not None:
                    progress.advance_step()
                continue
            probability, review_flags = _photoz_only_family_probability_from_arrays(pair_scores, sed_rms)
            family_z = float(np.nanmedian(valid_photoz))
            family_z_method = "photoz_median"
            min_spec_rank = float("nan")
        else:
            spec_spread = float(np.nanmax(usable_spec_values) - np.nanmin(usable_spec_values)) if usable_spec_values.size > 1 else 0.0
            spec_spread_norm = spec_spread / max(1.0 + float(np.nanmedian(usable_spec_values)), 1.0e-6)
            if usable_spec_values.size > 1 and spec_spread > ZSPEC_HARD_CONFLICT_TOL and spec_spread_norm > ZSPEC_HARD_CONFLICT_NORM_TOL:
                color_guided_spec_conflict = bool(
                    np.any(conflict_values)
                    and sed_rms.size > 0
                    and np.all(np.isfinite(sed_rms))
                    and float(np.nanmedian(sed_rms)) <= DEFAULT_CONFLICT_COLOR_FAMILY_RMS_MAX
                )
                if not color_guided_spec_conflict:
                    if progress is not None:
                        progress.advance_step()
                    continue
            if is_two_image and (usable_spec_values.size < 2 or np.any(np.isfinite(sed_rms) & (sed_rms > FAMILY_COLOR_RMS_ACCEPTABLE))):
                if progress is not None:
                    progress.advance_step()
                continue
            probability, review_flags = _family_probability_from_arrays(
                len(family),
                pair_scores,
                sed_rms,
                zspec_values,
                rank_values,
                is_two_image=is_two_image,
            )
            family_z, family_z_method, min_spec_rank = _family_redshift_summary_from_arrays(zspec_values, rank_values, zphot_values)
            if color_guided_spec_conflict:
                probability = min(probability, DEFAULT_PHOTOZ_ONLY_FAMILY_PROBABILITY_CAP)
                review_flags = "|".join(
                    dict.fromkeys(
                        [
                            *[flag for flag in review_flags.split("|") if flag],
                            "color_guided_zspec_conflict_family",
                        ]
                    )
                )
                valid_photoz = zphot_values[np.isfinite(zphot_values)]
                if valid_photoz.size:
                    family_z = float(np.nanmedian(valid_photoz))
                    family_z_method = "color_guided_photoz_median"
            if len(family) >= 3 and usable_spec_values.size == 1:
                probability = min(probability, FAMILY_ANCHORED_CAP)
                review_flags = "|".join(
                    dict.fromkeys(
                        [
                            *[flag for flag in review_flags.split("|") if flag],
                            "single_specz_anchor",
                            "photo_photo_complete_link",
                        ]
                    )
                )
                family_z = float(usable_spec_values[0])
                family_z_method = "single_specz_anchor"
        if len(family) >= 3 and probability < family_probability_threshold:
            if progress is not None:
                progress.advance_step()
            continue
        if is_two_image and probability < family_probability_threshold:
            if progress is not None:
                progress.advance_step()
            continue
        candidate_summaries.append(
            {
                "family": family,
                "source": "complete_link",
                "compact_idx": compact_idx,
                "candidate_indices": candidate_indices,
                "probability": probability,
                "pair_scores": pair_scores,
                "sed_rms": sed_rms,
                "separations_arcsec": separations_arcsec,
                "separations_kpc": separations_kpc,
                "family_z_best": family_z,
                "family_z_method": family_z_method,
                "min_specz_confidence": min_spec_rank,
                "review_flags": review_flags,
            }
        )
        if progress is not None:
            progress.advance_step()
    if progress is not None:
        progress.finish_step()

    anchored_summaries = _single_specz_anchor_family_summaries(
        candidates_reset,
        accepted,
        pairs,
        spec,
        pair_score_threshold=pair_score_threshold,
        family_probability_threshold=family_probability_threshold,
        max_family_span_kpc=max_family_span_kpc,
        family_photoz_delta_max=family_photoz_delta_max,
    )
    candidate_summaries.extend(anchored_summaries)
    growth_metrics["n_single_specz_anchor_family_candidates"] = int(len(anchored_summaries))

    selected: list[dict[str, Any]] = []
    object_family_counts: dict[str, int] = defaultdict(int)
    seen_selected_families: set[tuple[str, ...]] = set()
    for item in sorted(candidate_summaries, key=lambda value: value["probability"], reverse=True):
        family = item["family"]
        family_key = tuple(sorted(family))
        if family_key in seen_selected_families:
            continue
        if any(object_family_counts[object_id] >= max_families_per_object for object_id in family):
            continue
        seen_selected_families.add(family_key)
        selected.append(item)
        for object_id in family:
            object_family_counts[object_id] += 1
    growth_metrics["n_single_specz_anchor_families"] = int(
        sum(
            1
            for item in selected
            if item.get("source") == "single_specz_anchor"
            or "single_specz_anchor" in str(item.get("review_flags", ""))
        )
    )
    growth_metrics["n_photoz_only_families"] = int(
        sum(1 for item in selected if "photoz_only_candidate" in str(item.get("review_flags", "")))
    )

    family_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for family_number, item in enumerate(selected, start=1):
        family = set(item["family"])
        compact_idx = np.asarray(item.get("compact_idx", []), dtype=np.int32)
        family_id = f"{spec.key}_IF{family_number:05d}"
        pair_scores = np.asarray(item["pair_scores"], dtype=float)
        sed_rms = np.asarray(item["sed_rms"], dtype=float)
        separations_arcsec = np.asarray(item["separations_arcsec"], dtype=float)
        separations_kpc = np.asarray(item["separations_kpc"], dtype=float)
        candidate_indices = np.asarray(item["candidate_indices"], dtype=np.int32)
        candidate_review_flags: list[str] = []
        if "image_family_review_flags" in candidates_reset.columns:
            for candidate_index in candidate_indices:
                candidate_review_flags.extend(
                    flag
                    for flag in str(candidates_reset.iloc[int(candidate_index)].get("image_family_review_flags", "")).split("|")
                    if flag
                )
        review_flags = "|".join(dict.fromkeys([*[flag for flag in str(item["review_flags"]).split("|") if flag], *candidate_review_flags]))
        family_rows.append(
            {
                "cluster_key": spec.key,
                "candidate_family_id": family_id,
                "n_images": len(family),
                "family_probability": item["probability"],
                "max_separation_arcsec": float(np.nanmax(separations_arcsec)),
                "max_separation_kpc": float(np.nanmax(separations_kpc)),
                "family_z_best": item["family_z_best"],
                "family_z_method": item["family_z_method"],
                "min_specz_confidence": item["min_specz_confidence"],
                "median_sed_rms": float(np.nanmedian(sed_rms)),
                "min_pair_score": float(np.nanmin(pair_scores)),
                "review_flags": review_flags,
            }
        )
        if item.get("source") == "single_specz_anchor":
            ordered_member_refs = [
                (int(candidate_index), None)
                for candidate_index in sorted(
                    np.asarray(item["candidate_indices"], dtype=np.int32).tolist(),
                    key=lambda index: str(candidates_reset.iloc[int(index)].get("object_id", "")),
                )
            ]
        else:
            ordered_member_refs = [
                (int(growth.compact_candidate_indices[int(compact_object_index)]), int(compact_object_index))
                for compact_object_index in sorted(compact_idx.tolist(), key=lambda index: growth.compact_object_ids[int(index)])
            ]
        for image_rank, (candidate_index, compact_object_index) in enumerate(ordered_member_refs, start=1):
            row = candidates_reset.iloc[candidate_index]
            object_id = str(row.get("object_id", f"row:{candidate_index}"))
            if item.get("source") == "single_specz_anchor":
                anchor_edge_scores = item.get("anchor_edge_scores", {})
                if object_id == item.get("anchor_object_id"):
                    object_pair_scores = np.asarray(list(anchor_edge_scores.values()), dtype=float)
                else:
                    object_pair_scores = np.asarray([anchor_edge_scores.get(object_id, item["probability"])], dtype=float)
            else:
                object_pair_scores = growth.pair_score_matrix[int(compact_object_index), compact_idx]
            object_pair_scores = object_pair_scores[np.isfinite(object_pair_scores) & (object_pair_scores > 0.0)]
            mean_pair_score = float(np.nanmean(object_pair_scores)) if object_pair_scores.size else float(item["probability"])
            raw_probability = float(item["probability"] * mean_pair_score)
            member_rows.append(
                {
                    "cluster_key": spec.key,
                    "candidate_family_id": family_id,
                    "object_id": object_id,
                    "ra": _to_float(row.get("ra")),
                    "dec": _to_float(row.get("dec")),
                    "membership_probability": raw_probability,
                    "raw_probability": raw_probability,
                    "zspec_best": valid_redshift(row.get("zspec_best")),
                    "zspec_best_confidence": _confidence_label(row),
                    "zspec_best_native_quality": _valid_number(row.get("zspec_best_native_quality")),
                    "zphot_best": valid_redshift(row.get("zphot_best")),
                    "n_valid_bands": int(row.get("n_valid_bands", 0)),
                    "object_source": str(row.get("object_source", "")),
                    "catalog_sources": str(row.get("catalog_sources", "")),
                    "image_preclean_selected": _bool_value(row.get("image_preclean_selected", True)),
                    "image_preclean_reject_reason": str(row.get("image_preclean_reject_reason", "")),
                    "image_size_arcsec": _to_float(row.get("image_size_arcsec")),
                    "image_ellipticity": _to_float(row.get("image_ellipticity")),
                    "image_photoz_quality_selected": _bool_value(row.get("image_photoz_quality_selected", False)),
                    "image_photoz_reject_reason": str(row.get("image_photoz_reject_reason", "")),
                    "image_zphot_family": valid_redshift(row.get("image_zphot_family")),
                    "image_family_rescue_selected": _bool_value(row.get("image_family_rescue_selected", False)),
                    "image_family_review_flags": str(row.get("image_family_review_flags", "")),
                    "image_rank": image_rank,
                    "mean_pair_score_to_family": mean_pair_score,
                }
            )

    members = normalize_membership_probabilities(pd.DataFrame(member_rows))
    families = pd.DataFrame(family_rows, columns=FAMILY_COLUMNS)
    if members.empty:
        members = pd.DataFrame(columns=[*MEMBER_COLUMNS, "image_rank", "mean_pair_score_to_family"])
    return _attach_family_growth_metrics(families, members, growth_metrics)


def normalize_membership_probabilities(members: pd.DataFrame) -> pd.DataFrame:
    if members.empty or "object_id" not in members.columns:
        return members
    result = members.copy()
    result["raw_probability"] = pd.to_numeric(result["raw_probability"], errors="coerce").fillna(0.0)
    result["membership_probability"] = result["raw_probability"]
    sums = result.groupby("object_id")["raw_probability"].transform("sum")
    over = sums > 1.0
    result.loc[over, "membership_probability"] = result.loc[over, "raw_probability"] / sums.loc[over]
    result["membership_probability"] = result["membership_probability"].clip(lower=0.0, upper=1.0)
    return result


def build_cluster_image_families(
    catalog: pd.DataFrame,
    spec: ClusterSpec,
    *,
    max_family_span_kpc: float = DEFAULT_MAX_FAMILY_SPAN_KPC,
    min_pair_separation_arcsec: float = DEFAULT_MIN_PAIR_SEPARATION_ARCSEC,
    min_common_bands: int = DEFAULT_MIN_COMMON_BANDS,
    pair_score_threshold: float = DEFAULT_PAIR_SCORE_THRESHOLD,
    two_image_score_threshold: float = DEFAULT_TWO_IMAGE_SCORE_THRESHOLD,
    family_color_rms_max: float = DEFAULT_FAMILY_COLOR_RMS_MAX,
    family_photoz_delta_max: float = DEFAULT_FAMILY_PHOTOZ_DELTA_MAX,
    image_family_fov_kpc: float = DEFAULT_IMAGE_FAMILY_FOV_KPC,
    family_probability_threshold: float = DEFAULT_FAMILY_PROBABILITY_THRESHOLD,
    max_families_per_object: int = DEFAULT_MAX_FAMILIES_PER_OBJECT,
    family_pair_batch_size: int = DEFAULT_FAMILY_PAIR_BATCH_SIZE,
    family_pair_diagnostics: str = DEFAULT_FAMILY_PAIR_DIAGNOSTICS,
    image_bright_mag_f814w: float = DEFAULT_IMAGE_BRIGHT_MAG_F814W,
    image_hff_faint_mag_f814w: float = DEFAULT_IMAGE_HFF_FAINT_MAG_F814W,
    image_outer_faint_mag_f814w: float = DEFAULT_IMAGE_OUTER_FAINT_MAG_F814W,
    image_min_size_arcsec: float = DEFAULT_IMAGE_MIN_SIZE_ARCSEC,
    image_size_pixel_scale_arcsec: float = DEFAULT_IMAGE_SIZE_PIXEL_SCALE_ARCSEC,
    image_photoz_min_mag_f160w: float = DEFAULT_IMAGE_PHOTOZ_MIN_MAG_F160W,
    image_photoz_max_mag_f160w: float = DEFAULT_IMAGE_PHOTOZ_MAX_MAG_F160W,
    image_photoz_min_nb_used: float = DEFAULT_IMAGE_PHOTOZ_MIN_NB_USED,
    image_photoz_max_dz_norm: float = DEFAULT_IMAGE_PHOTOZ_MAX_DZ_NORM,
    strong_lensing_rescue_faint_mag_f814w: float = DEFAULT_STRONG_LENSING_RESCUE_FAINT_MAG_F814W,
    strong_lensing_rescue_min_bands: int = DEFAULT_STRONG_LENSING_RESCUE_MIN_BANDS,
    progress: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if progress is not None:
        progress.start_step(f"{spec.key}: preparing image candidates", total=None)
    candidates = prepare_candidates(
        catalog,
        spec,
        min_common_bands=min_common_bands,
        strong_lensing_rescue_faint_mag_f814w=strong_lensing_rescue_faint_mag_f814w,
        strong_lensing_rescue_min_bands=strong_lensing_rescue_min_bands,
        image_family_fov_kpc=image_family_fov_kpc,
        image_bright_mag_f814w=image_bright_mag_f814w,
        image_hff_faint_mag_f814w=image_hff_faint_mag_f814w,
        image_outer_faint_mag_f814w=image_outer_faint_mag_f814w,
        image_min_size_arcsec=image_min_size_arcsec,
        image_size_pixel_scale_arcsec=image_size_pixel_scale_arcsec,
        image_photoz_min_mag_f160w=image_photoz_min_mag_f160w,
        image_photoz_max_mag_f160w=image_photoz_max_mag_f160w,
        image_photoz_min_nb_used=image_photoz_min_nb_used,
    )
    if progress is not None:
        progress.finish_step()
    pair_result = _score_candidate_pair_result_jax(
        candidates,
        spec,
        max_family_span_kpc=max_family_span_kpc,
        min_pair_separation_arcsec=min_pair_separation_arcsec,
        min_common_bands=min_common_bands,
        pair_score_threshold=pair_score_threshold,
        family_pair_batch_size=family_pair_batch_size,
        family_pair_diagnostics=family_pair_diagnostics,
        image_photoz_max_dz_norm=image_photoz_max_dz_norm,
        family_color_rms_max=family_color_rms_max,
        family_photoz_delta_max=family_photoz_delta_max,
        progress=progress,
    )
    pairs = pair_result.pairs
    families, members = build_families_from_pairs(
        candidates,
        pairs,
        spec,
        pair_score_threshold=pair_score_threshold,
        two_image_score_threshold=two_image_score_threshold,
        family_probability_threshold=family_probability_threshold,
        max_family_span_kpc=max_family_span_kpc,
        family_photoz_delta_max=family_photoz_delta_max,
        max_families_per_object=max_families_per_object,
        progress=progress,
    )
    if not pairs.empty:
        accepted_mask = (
            (pairs["hard_reject_reason"].fillna("") == "")
            & (pd.to_numeric(pairs["pair_score"], errors="coerce") >= pair_score_threshold)
        )
        n_accepted_pairs = int(accepted_mask.sum())
        n_photoz_only_accepted_pairs = int(
            (accepted_mask & pairs["redshift_relation"].fillna("").astype(str).eq("photoz_only")).sum()
        )
    else:
        n_accepted_pairs = 0
        n_photoz_only_accepted_pairs = 0
    pair_score_metrics = pairs.attrs.get("pair_score_metrics", {})
    family_growth_metrics = families.attrs.get("family_growth_metrics", _empty_family_growth_metrics())
    image_preclean_metrics = candidates.attrs.get("image_preclean_metrics", {})
    image_photoz_quality_metrics = candidates.attrs.get("image_photoz_quality_metrics", {})
    image_family_selection_metrics = candidates.attrs.get("image_family_selection_metrics", {})
    manifest = {
        "cluster_key": spec.key,
        "z_lens": spec.z_lens,
        "kpc_per_arcsec": kpc_per_arcsec(spec.z_lens),
        "max_family_span_kpc": max_family_span_kpc,
        "max_family_span_arcsec": max_span_arcsec(spec, max_family_span_kpc),
        "n_master_rows": int(len(catalog)),
        "n_candidates": int(len(candidates)),
        "n_image_preclean_rows": int(image_preclean_metrics.get("n_image_preclean_rows", len(catalog))),
        "n_image_preclean_selected": int(image_preclean_metrics.get("n_image_preclean_selected", len(candidates))),
        "n_image_preclean_rejected": int(image_preclean_metrics.get("n_image_preclean_rejected", 0)),
        "n_image_photoz_quality_rows": int(image_photoz_quality_metrics.get("n_image_photoz_quality_rows", len(catalog))),
        "n_image_photoz_quality_selected": int(image_photoz_quality_metrics.get("n_image_photoz_quality_selected", 0)),
        "n_image_photoz_quality_rejected": int(image_photoz_quality_metrics.get("n_image_photoz_quality_rejected", 0)),
        "n_scored_pairs": int(len(pairs)),
        "n_accepted_pairs": n_accepted_pairs,
        "n_photoz_only_accepted_pairs": n_photoz_only_accepted_pairs,
        "n_candidate_families": int(len(families)),
        "n_candidate_family_members": int(len(members)),
        "min_common_bands": int(min_common_bands),
        "pair_score_threshold": float(pair_score_threshold),
        "two_image_score_threshold": float(two_image_score_threshold),
        "family_color_rms_max": float(family_color_rms_max),
        "family_photoz_delta_max": float(family_photoz_delta_max),
        "photoz_only_family_probability_cap": float(DEFAULT_PHOTOZ_ONLY_FAMILY_PROBABILITY_CAP),
        "zspec_excellent_tol": float(ZSPEC_EXCELLENT_TOL),
        "zspec_hard_conflict_tol": float(ZSPEC_HARD_CONFLICT_TOL),
        "zspec_hard_conflict_norm_tol": float(ZSPEC_HARD_CONFLICT_NORM_TOL),
        "family_color_rms_strong": float(FAMILY_COLOR_RMS_STRONG),
        "family_color_rms_acceptable": float(FAMILY_COLOR_RMS_ACCEPTABLE),
        "n_spatial_pairs": int(pair_score_metrics.get("n_spatial_pairs", len(pairs))),
        "n_prefiltered_pairs": int(pair_score_metrics.get("n_prefiltered_pairs", len(pairs))),
        "n_pruned_redshift": int(pair_score_metrics.get("n_pruned_redshift", 0)),
        "n_pruned_photoz_delta": int(pair_score_metrics.get("n_pruned_photoz_delta", 0)),
        "n_pruned_score_upper_bound": int(pair_score_metrics.get("n_pruned_score_upper_bound", 0)),
        "n_pruned_common_bands": int(pair_score_metrics.get("n_pruned_common_bands", 0)),
        "n_prefilter_full_score_pairs": int(pair_score_metrics.get("n_prefilter_full_score_pairs", 0)),
        "n_prefilter_redshift_rejects": int(pair_score_metrics.get("n_prefilter_redshift_rejects", 0)),
        "n_prefilter_photoz_delta_rejects": int(pair_score_metrics.get("n_prefilter_photoz_delta_rejects", 0)),
        "n_accepted_array_pairs": int(pair_score_metrics.get("n_accepted_array_pairs", n_accepted_pairs)),
        "pair_score_backend": str(pair_score_metrics.get("pair_score_backend", "jax")),
        "pair_score_seconds": float(pair_score_metrics.get("pair_score_seconds", 0.0)),
        "image_photoz_min_mag_f160w": float(
            image_photoz_quality_metrics.get("image_photoz_min_mag_f160w", image_photoz_min_mag_f160w)
        ),
        "image_photoz_max_mag_f160w": float(
            image_photoz_quality_metrics.get("image_photoz_max_mag_f160w", image_photoz_max_mag_f160w)
        ),
        "image_photoz_min_nb_used": float(
            image_photoz_quality_metrics.get("image_photoz_min_nb_used", image_photoz_min_nb_used)
        ),
        "image_photoz_max_dz_norm": float(image_photoz_max_dz_norm),
        "image_family_fov_kpc": float(image_family_selection_metrics.get("image_family_fov_kpc", image_family_fov_kpc)),
        "image_family_fov_shape": str(image_family_selection_metrics.get("image_family_fov_shape", "circle")),
        "image_family_fov_radius_kpc": float(
            image_family_selection_metrics.get("image_family_fov_radius_kpc", float(image_family_fov_kpc) / 2.0)
        ),
        "image_family_fov_half_width_kpc": float(
            image_family_selection_metrics.get("image_family_fov_half_width_kpc", float(image_family_fov_kpc) / 2.0)
        ),
        "image_family_center_ra": float(image_family_selection_metrics.get("image_family_center_ra", np.nan)),
        "image_family_center_dec": float(image_family_selection_metrics.get("image_family_center_dec", np.nan)),
        "image_family_center_note": str(image_family_selection_metrics.get("image_family_center_note", "")),
        "n_image_family_fov_input_rows": int(image_family_selection_metrics.get("n_image_family_fov_input_rows", len(catalog))),
        "n_image_family_fov_rows": int(image_family_selection_metrics.get("n_image_family_fov_rows", len(candidates))),
        "n_rejected_missing_strong_specz": int(image_family_selection_metrics.get("n_rejected_missing_strong_specz", 0)),
        "n_rejected_missing_family_redshift_evidence": int(image_family_selection_metrics.get("n_rejected_missing_family_redshift_evidence", 0)),
        "n_image_family_strong_specz_candidates": int(image_family_selection_metrics.get("n_image_family_strong_specz_candidates", 0)),
        "n_image_family_photoz_companion_candidates": int(image_family_selection_metrics.get("n_image_family_photoz_companion_candidates", 0)),
        "n_image_family_rescue_candidates": int(image_family_selection_metrics.get("n_image_family_rescue_candidates", 0)),
        "n_zspec_conflict_candidates": int(image_family_selection_metrics.get("n_zspec_conflict_candidates", 0)),
        "strong_lensing_rescue_faint_mag_f814w": float(
            image_family_selection_metrics.get("strong_lensing_rescue_faint_mag_f814w", strong_lensing_rescue_faint_mag_f814w)
        ),
        "strong_lensing_rescue_min_bands": int(
            image_family_selection_metrics.get("strong_lensing_rescue_min_bands", strong_lensing_rescue_min_bands)
        ),
        "n_rejected_outside_family_fov": int(image_family_selection_metrics.get("n_rejected_outside_family_fov", 0)),
        "n_rejected_zspec_conflict": int(image_family_selection_metrics.get("n_rejected_zspec_conflict", 0)),
        "n_rejected_not_background_specz": int(image_family_selection_metrics.get("n_rejected_not_background_specz", 0)),
        "image_bright_mag_f814w": float(image_preclean_metrics.get("image_bright_mag_f814w", image_bright_mag_f814w)),
        "image_hff_faint_mag_f814w": float(image_preclean_metrics.get("image_hff_faint_mag_f814w", image_hff_faint_mag_f814w)),
        "image_outer_faint_mag_f814w": float(image_preclean_metrics.get("image_outer_faint_mag_f814w", image_outer_faint_mag_f814w)),
        "image_min_size_arcsec": float(image_preclean_metrics.get("image_min_size_arcsec", image_min_size_arcsec)),
        "image_size_pixel_scale_arcsec": float(
            image_preclean_metrics.get("image_size_pixel_scale_arcsec", image_size_pixel_scale_arcsec)
        ),
        "n_image_preclean_size_available": int(image_preclean_metrics.get("n_image_preclean_size_available", 0)),
        "n_image_preclean_shape_available": int(image_preclean_metrics.get("n_image_preclean_shape_available", 0)),
        "n_image_preclean_shipley_use_available": int(image_preclean_metrics.get("n_image_preclean_shipley_use_available", 0)),
        "n_image_preclean_f814w_flag_available": int(image_preclean_metrics.get("n_image_preclean_f814w_flag_available", 0)),
        "n_image_preclean_f814w_error_available": int(image_preclean_metrics.get("n_image_preclean_f814w_error_available", 0)),
        "family_growth_backend": str(family_growth_metrics.get("family_growth_backend", "jax_dense")),
        "family_growth_seconds": float(family_growth_metrics.get("family_growth_seconds", 0.0)),
        "n_family_growth_objects": int(family_growth_metrics.get("n_family_growth_objects", 0)),
        "n_family_growth_seed_edges": int(family_growth_metrics.get("n_family_growth_seed_edges", 0)),
        "family_growth_n_batches": int(family_growth_metrics.get("family_growth_n_batches", 0)),
        "family_growth_unique_masks": int(family_growth_metrics.get("family_growth_unique_masks", 0)),
        "family_growth_packed_mask_bytes": int(family_growth_metrics.get("family_growth_packed_mask_bytes", 0)),
        "family_growth_all_seed_batch": bool(family_growth_metrics.get("family_growth_all_seed_batch", False)),
        "n_single_specz_anchor_family_candidates": int(
            family_growth_metrics.get("n_single_specz_anchor_family_candidates", 0)
        ),
        "n_single_specz_anchor_families": int(family_growth_metrics.get("n_single_specz_anchor_families", 0)),
        "n_photoz_only_families": int(family_growth_metrics.get("n_photoz_only_families", 0)),
    }
    return families, members, pairs, manifest


class TqdmProgressManager:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.disable = not console.is_terminal
        self._bars: list[tqdm] = []

    def __enter__(self) -> "TqdmProgressManager":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        for bar in reversed(self._bars):
            bar.close()
        self._bars.clear()

    def make_bar(
        self,
        *,
        desc: str,
        total: int | None,
        position: int,
        leave: bool,
        colour: str,
        bar_format: str | None = None,
    ) -> tqdm:
        bar = tqdm(
            total=total,
            desc=desc,
            dynamic_ncols=True,
            leave=leave,
            position=position,
            colour=colour,
            file=self.console.file,
            disable=self.disable,
            bar_format=bar_format,
        )
        self._bars.append(bar)
        return bar

    def close_bar(self, bar: tqdm | None) -> None:
        if bar is None:
            return
        bar.close()
        if bar in self._bars:
            self._bars.remove(bar)


def make_progress(console: Console) -> TqdmProgressManager:
    return TqdmProgressManager(console)


class CatalogProgress:
    def __init__(
        self,
        progress: TqdmProgressManager,
        *,
        total_clusters: int,
        update_interval: int = DEFAULT_PROGRESS_UPDATE_INTERVAL,
    ) -> None:
        self.progress = progress
        self.update_interval = max(1, int(update_interval))
        self.cluster_bar = progress.make_bar(
            desc="clusters",
            total=total_clusters,
            position=0,
            leave=True,
            colour="green",
        )
        self.step_bar: tqdm | None = None
        self._step_total: int | None = None
        self._step_pending = 0

    def set_cluster_phase(self, label: str) -> None:
        self.cluster_bar.set_description_str(label)

    def advance_cluster(self, n: int = 1) -> None:
        self.cluster_bar.update(max(0, int(n)))

    def start_step(self, label: str, total: int | None = None) -> None:
        self.finish_step()
        self._step_total = None if total is None else max(0, int(total))
        self._step_pending = 0
        bar_format = "{desc}: {elapsed}" if self._step_total is None else None
        self.step_bar = self.progress.make_bar(
            desc=f"  {label}",
            total=self._step_total,
            position=1,
            leave=False,
            colour="cyan",
            bar_format=bar_format,
        )

    def advance_step(self, n: int = 1) -> None:
        self._step_pending += max(0, int(n))
        if self._step_total is None:
            self._flush_step()
            return
        current = int(self.step_bar.n) if self.step_bar is not None else 0
        if self._step_pending >= self.update_interval or current + self._step_pending >= self._step_total:
            self._flush_step()

    def finish_step(self) -> None:
        self._flush_step()
        if self.step_bar is not None and self._step_total is not None:
            remaining = self._step_total - int(self.step_bar.n)
            if remaining > 0:
                self.step_bar.update(remaining)
        self.progress.close_bar(self.step_bar)
        self.step_bar = None
        self._step_total = None

    def _flush_step(self) -> None:
        if self._step_pending <= 0 or self.step_bar is None:
            return
        self.step_bar.update(self._step_pending)
        self._step_pending = 0


def cluster_output_dir(output_dir: Path, spec: ClusterSpec) -> Path:
    return output_dir / spec.key


def master_catalog_path(output_dir: Path, spec: ClusterSpec) -> Path:
    return cluster_output_dir(output_dir, spec) / f"{spec.key}_master_catalog.csv"


def match_audit_path(output_dir: Path, spec: ClusterSpec) -> Path:
    return cluster_output_dir(output_dir, spec) / f"{spec.key}_match_audit.csv"


def cluster_member_catalog_paths(output_dir: Path, spec: ClusterSpec) -> tuple[Path, Path, Path, Path, Path]:
    cluster_dir = cluster_output_dir(output_dir, spec)
    return (
        cluster_dir / f"{spec.key}_cluster_member_scores.csv",
        cluster_dir / f"{spec.key}_cluster_members.csv",
        cluster_dir / f"{spec.key}_cluster_members_potfile.cat",
        cluster_dir / f"{spec.key}_cluster_member_red_sequence.csv",
        cluster_dir / f"{spec.key}_bcg_special_member_candidates.csv",
    )


def family_catalog_paths(output_dir: Path, spec: ClusterSpec) -> tuple[Path, Path, Path]:
    cluster_dir = cluster_output_dir(output_dir, spec)
    return (
        cluster_dir / f"{spec.key}_candidate_image_families.csv",
        cluster_dir / f"{spec.key}_candidate_family_members.csv",
        cluster_dir / f"{spec.key}_candidate_family_pairs.csv",
    )


def reference_family_catalog_paths(output_dir: Path, spec: ClusterSpec) -> tuple[Path, Path]:
    cluster_dir = cluster_output_dir(output_dir, spec)
    return (
        cluster_dir / f"{spec.key}_reference_family_crossmatch.csv",
        cluster_dir / f"{spec.key}_reference_family_recovery.csv",
    )


def default_reference_family_path(spec: ClusterSpec) -> Path | None:
    path = DEFAULT_REFERENCE_FAMILY_PATHS.get(spec.key)
    if path is not None and path.exists():
        return path
    return None


def parse_reference_family_catalog(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) < 5:
            continue
        image_id = parts[0]
        family_id = image_id.split(".", 1)[0]
        image_rank = image_id.split(".", 1)[1] if "." in image_id else ""
        raw_z = parts[3].strip().strip('"')
        rows.append(
            {
                "reference_line": line_number,
                "reference_image_id": image_id,
                "reference_family_id": family_id,
                "reference_image_rank": image_rank,
                "reference_ra": _to_float(parts[1]),
                "reference_dec": _to_float(parts[2]),
                "reference_z": valid_redshift(raw_z),
                "reference_category": parts[4],
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "reference_line",
            "reference_image_id",
            "reference_family_id",
            "reference_image_rank",
            "reference_ra",
            "reference_dec",
            "reference_z",
            "reference_category",
        ],
    )


def _reference_nearest_match(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    *,
    target_prefix: str,
    radius_arcsec: float,
) -> pd.DataFrame:
    if reference.empty or target.empty or "ra" not in target.columns or "dec" not in target.columns:
        result = pd.DataFrame(index=reference.index)
        result[f"{target_prefix}_index"] = np.nan
        result[f"{target_prefix}_match_sep_arcsec"] = np.nan
        return result
    ref_coords = SkyCoord(
        pd.to_numeric(reference["reference_ra"], errors="coerce").to_numpy(dtype=float) * u.deg,
        pd.to_numeric(reference["reference_dec"], errors="coerce").to_numpy(dtype=float) * u.deg,
    )
    target_ra = pd.to_numeric(target["ra"], errors="coerce")
    target_dec = pd.to_numeric(target["dec"], errors="coerce")
    valid_target = target_ra.notna() & target_dec.notna()
    result = pd.DataFrame(index=reference.index)
    result[f"{target_prefix}_index"] = np.nan
    result[f"{target_prefix}_match_sep_arcsec"] = np.nan
    if not valid_target.any():
        return result
    target_indices = target.index[valid_target].to_numpy()
    target_coords = SkyCoord(
        target_ra[valid_target].to_numpy(dtype=float) * u.deg,
        target_dec[valid_target].to_numpy(dtype=float) * u.deg,
    )
    nearest, sep2d, _ = ref_coords.match_to_catalog_sky(target_coords)
    sep_arcsec = sep2d.arcsec
    matched = sep_arcsec <= float(radius_arcsec)
    result.loc[matched, f"{target_prefix}_index"] = target_indices[nearest[matched]]
    result.loc[matched, f"{target_prefix}_match_sep_arcsec"] = sep_arcsec[matched]
    return result


def build_reference_family_diagnostics(
    *,
    reference_path: Path,
    master: pd.DataFrame,
    families: pd.DataFrame,
    family_members: pd.DataFrame,
    match_radius_arcsec: float = DEFAULT_REFERENCE_MATCH_RADIUS_ARCSEC,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    reference = parse_reference_family_catalog(reference_path)
    if reference.empty:
        empty_recovery = pd.DataFrame()
        return reference, empty_recovery, {"n_reference_images": 0, "n_reference_families": 0}

    crossmatch = reference.copy()
    master_matches = _reference_nearest_match(reference, master, target_prefix="master", radius_arcsec=match_radius_arcsec)
    crossmatch = pd.concat([crossmatch, master_matches], axis=1)
    master_value_columns = [
        "object_id",
        "object_source",
        "catalog_sources",
        "mag_F435W",
        "mag_F606W",
        "mag_F814W",
        "mag_F105W",
        "mag_F125W",
        "mag_F160W",
        "zspec_best",
        "zspec_best_source",
        "zspec_best_confidence_rank",
        "zspec_selection_note",
        "zspec_conflict",
        "zphot_best",
        "pagul_zpdf",
        "pagul_zpdf_low",
        "pagul_zpdf_high",
        "pagul_nb_used",
        "member_probability",
        "member_for_lensing",
        "cluster_member_selected",
    ]
    for column in master_value_columns:
        crossmatch[column] = pd.Series([pd.NA] * len(crossmatch), index=crossmatch.index, dtype=object)
    matched_master = crossmatch["master_index"].notna()
    if matched_master.any():
        master_rows = master.loc[crossmatch.loc[matched_master, "master_index"].astype(int)]
        for column in master_value_columns:
            if column in master_rows.columns:
                crossmatch.loc[matched_master, column] = master_rows[column].to_numpy()
    for blue, red in (("mag_F435W", "mag_F814W"), ("mag_F606W", "mag_F814W"), ("mag_F814W", "mag_F160W")):
        if blue in crossmatch.columns and red in crossmatch.columns:
            crossmatch[f"color_{blue[4:]}_{red[4:]}"] = (
                pd.to_numeric(crossmatch[blue], errors="coerce") - pd.to_numeric(crossmatch[red], errors="coerce")
            )

    member_matches = _reference_nearest_match(
        reference,
        family_members,
        target_prefix="generated_member",
        radius_arcsec=match_radius_arcsec,
    )
    crossmatch = pd.concat([crossmatch, member_matches], axis=1)
    for column in ("candidate_family_id", "membership_probability", "image_rank", "image_family_review_flags"):
        crossmatch[f"generated_{column}"] = pd.Series([pd.NA] * len(crossmatch), index=crossmatch.index, dtype=object)
    generated_members_with_index = family_members.reset_index().rename(columns={"index": "generated_member_index"})
    if "object_id" in generated_members_with_index.columns and "object_id" in crossmatch.columns:
        generated_members_with_index["object_id"] = generated_members_with_index["object_id"].astype(str)
        member_probability = pd.to_numeric(generated_members_with_index.get("membership_probability"), errors="coerce")
        best_generated_by_object = (
            generated_members_with_index.assign(_membership_probability_sort=member_probability.fillna(-np.inf))
            .sort_values(["object_id", "_membership_probability_sort", "candidate_family_id"], ascending=[True, False, True], kind="mergesort")
            .drop_duplicates("object_id")
            .set_index("object_id", drop=False)
        )
        object_matched_member = crossmatch["object_id"].map(lambda value: str(value) in best_generated_by_object.index if pd.notna(value) else False)
        if object_matched_member.any():
            object_member_rows = best_generated_by_object.loc[crossmatch.loc[object_matched_member, "object_id"].astype(str)]
            crossmatch.loc[object_matched_member, "generated_member_index"] = object_member_rows["generated_member_index"].to_numpy()
            crossmatch.loc[object_matched_member, "generated_member_match_sep_arcsec"] = crossmatch.loc[
                object_matched_member, "master_match_sep_arcsec"
            ].to_numpy()
    matched_member = crossmatch["generated_member_index"].notna()
    if matched_member.any():
        member_rows = family_members.loc[crossmatch.loc[matched_member, "generated_member_index"].astype(int)]
        for column in ("candidate_family_id", "membership_probability", "image_rank", "image_family_review_flags"):
            if column in member_rows.columns:
                crossmatch.loc[matched_member, f"generated_{column}"] = member_rows[column].to_numpy()

    object_family_matches = pd.DataFrame()
    if {"object_id", "candidate_family_id"}.issubset(generated_members_with_index.columns) and "object_id" in crossmatch.columns:
        reference_objects = crossmatch.loc[crossmatch["object_id"].notna(), ["reference_line", "reference_family_id", "object_id"]].copy()
        reference_objects["object_id"] = reference_objects["object_id"].astype(str)
        member_families = generated_members_with_index.loc[
            generated_members_with_index["object_id"].notna(),
            ["object_id", "candidate_family_id"],
        ].copy()
        member_families["object_id"] = member_families["object_id"].astype(str)
        object_family_matches = reference_objects.merge(member_families, on="object_id", how="inner")
    generated_counts_source = (
        object_family_matches.rename(columns={"candidate_family_id": "generated_candidate_family_id"})
        if not object_family_matches.empty
        else crossmatch.loc[crossmatch["generated_member_index"].notna(), ["reference_family_id", "generated_candidate_family_id"]]
    )
    generated_counts = (
        generated_counts_source.groupby(["reference_family_id", "generated_candidate_family_id"], dropna=True)
        .size()
        .rename("n_generated_members_in_same_candidate_family")
        .reset_index()
    )
    best_generated = (
        generated_counts.sort_values(
            ["reference_family_id", "n_generated_members_in_same_candidate_family", "generated_candidate_family_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("reference_family_id")
        if not generated_counts.empty
        else pd.DataFrame(columns=["reference_family_id", "generated_candidate_family_id", "n_generated_members_in_same_candidate_family"])
    )
    recovery = (
        crossmatch.groupby("reference_family_id")
        .agg(
            n_reference_images=("reference_image_id", "size"),
            n_master_matches=("master_index", lambda values: int(pd.Series(values).notna().sum())),
            n_generated_member_matches=("generated_member_index", lambda values: int(pd.Series(values).notna().sum())),
            reference_categories=("reference_category", lambda values: "|".join(sorted(set(map(str, values))))),
            reference_z_median=("reference_z", "median"),
        )
        .reset_index()
    )
    recovery = recovery.merge(best_generated, on="reference_family_id", how="left")
    if not object_family_matches.empty:
        generated_reference_lines = object_family_matches["reference_line"].drop_duplicates()
        object_generated_matches = (
            crossmatch.loc[crossmatch["reference_line"].isin(generated_reference_lines)]
            .groupby("reference_family_id")
            .size()
            .rename("n_generated_member_matches_by_object")
            .reset_index()
        )
        recovery = recovery.merge(object_generated_matches, on="reference_family_id", how="left")
        recovery["n_generated_member_matches"] = recovery["n_generated_member_matches_by_object"].fillna(0).astype(int)
        recovery = recovery.drop(columns=["n_generated_member_matches_by_object"])
    recovery["recoverable_family"] = recovery["n_master_matches"] >= 2
    recovery["recovered_family"] = recovery["n_generated_members_in_same_candidate_family"].fillna(0).astype(int) >= 2
    recovery["n_generated_family_rows"] = int(len(families))
    metrics = {
        "reference_family_path": str(reference_path),
        "reference_match_radius_arcsec": float(match_radius_arcsec),
        "n_reference_images": int(len(reference)),
        "n_reference_families": int(reference["reference_family_id"].nunique()),
        "n_reference_master_matches": int(crossmatch["master_index"].notna().sum()),
        "n_reference_generated_member_matches": int(crossmatch["generated_member_index"].notna().sum()),
        "n_reference_recoverable_families": int(recovery["recoverable_family"].sum()),
        "n_reference_recovered_families": int(recovery["recovered_family"].sum()),
    }
    return crossmatch, recovery, metrics


def plot_output_dirs(output_dir: Path, spec: ClusterSpec) -> tuple[Path, Path]:
    plot_dir = cluster_output_dir(output_dir, spec) / "plots"
    return plot_dir / "diagnostic", plot_dir / "publication"


def catalog_plot_manifest_path(output_dir: Path) -> Path:
    return output_dir / "hff_catalog_plot_manifest.csv"


def _safe_numeric_frame_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def _plot_manifest_row(
    *,
    spec: ClusterSpec | None,
    plot_kind: str,
    plot_name: str,
    status: str,
    paths: Iterable[Path] | None = None,
    reason: str = "",
    used_background: bool = False,
    background_path: Path | None = None,
    n_master_rows: int = 0,
    n_member_rows: int = 0,
    n_family_rows: int = 0,
    plot_center_ra: float = np.nan,
    plot_center_dec: float = np.nan,
    plot_fov_half_width_kpc: float = np.nan,
    n_spatial_input_rows: int = 0,
    n_spatial_fov_rows: int = 0,
    n_spatial_plotted_rows: int = 0,
    spatial_selection_note: str = "",
    image_render_mode: str = "",
    image_scale: str = "",
    n_cutout_families: int = 0,
    n_cutout_images: int = 0,
    family_cutout_size_arcsec: float = np.nan,
    family_cutout_circle_radius_arcsec: float = np.nan,
    family_cutout_color_rms_label: bool = False,
    family_cutout_bands: str = "",
    family_cutout_rgb_paths: str = "",
) -> dict[str, Any]:
    path_list = [str(path) for path in (paths or [])]
    return {
        "cluster_key": spec.key if spec is not None else "all",
        "plot_kind": plot_kind,
        "plot_name": plot_name,
        "status": status,
        "path": path_list[0] if path_list else "",
        "paths": "|".join(path_list),
        "reason": reason,
        "used_background": bool(used_background),
        "background_path": str(background_path) if background_path is not None else "",
        "n_master_rows": int(n_master_rows),
        "n_member_rows": int(n_member_rows),
        "n_family_rows": int(n_family_rows),
        "plot_center_ra": plot_center_ra,
        "plot_center_dec": plot_center_dec,
        "plot_fov_half_width_kpc": plot_fov_half_width_kpc,
        "n_spatial_input_rows": int(n_spatial_input_rows),
        "n_spatial_fov_rows": int(n_spatial_fov_rows),
        "n_spatial_plotted_rows": int(n_spatial_plotted_rows),
        "spatial_selection_note": spatial_selection_note,
        "image_render_mode": image_render_mode,
        "image_scale": image_scale,
        "n_cutout_families": int(n_cutout_families),
        "n_cutout_images": int(n_cutout_images),
        "family_cutout_size_arcsec": family_cutout_size_arcsec,
        "family_cutout_circle_radius_arcsec": family_cutout_circle_radius_arcsec,
        "family_cutout_color_rms_label": bool(family_cutout_color_rms_label),
        "family_cutout_bands": family_cutout_bands,
        "family_cutout_rgb_paths": family_cutout_rgb_paths,
    }


def _coerce_plot_output(output: Any) -> tuple[list[Path], dict[str, Any]]:
    if isinstance(output, tuple) and len(output) == 2:
        paths, metadata = output
        return list(paths), dict(metadata or {})
    return list(output), {}


def _write_diagnostic_png(fig: plt.Figure, path: Path) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [path]


def _write_publication_fig(fig: plt.Figure, stem: Path, *, dpi: int = 300) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _finite_sky(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "ra" not in df.columns or "dec" not in df.columns:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    ra = pd.to_numeric(df["ra"], errors="coerce")
    dec = pd.to_numeric(df["dec"], errors="coerce")
    return df.loc[np.isfinite(ra) & np.isfinite(dec)].copy()


def _spatial_metadata(
    *,
    center_ra: float = np.nan,
    center_dec: float = np.nan,
    fov_half_width_kpc: float = PLOT_FOV_HALF_WIDTH_KPC,
    input_rows: int = 0,
    fov_rows: int = 0,
    plotted_rows: int = 0,
    selection_note: str = "",
    image_render_mode: str = "",
) -> dict[str, Any]:
    return {
        "plot_center_ra": center_ra,
        "plot_center_dec": center_dec,
        "plot_fov_half_width_kpc": fov_half_width_kpc,
        "n_spatial_input_rows": int(input_rows),
        "n_spatial_fov_rows": int(fov_rows),
        "n_spatial_plotted_rows": int(plotted_rows),
        "spatial_selection_note": selection_note,
        "image_render_mode": image_render_mode,
    }


def _choose_plot_center(
    master: pd.DataFrame,
    member_scores: pd.DataFrame | None,
) -> tuple[float, float, str]:
    if member_scores is not None and not member_scores.empty:
        member_sky = _finite_sky(member_scores)
        if not member_sky.empty:
            probability = _safe_numeric_frame_column(member_sky, "member_probability").fillna(0.0)
            high_probability = member_sky.loc[probability >= 0.80]
            if not high_probability.empty:
                return (
                    float(pd.to_numeric(high_probability["ra"], errors="coerce").median()),
                    float(pd.to_numeric(high_probability["dec"], errors="coerce").median()),
                    "median_ra_dec_of_member_probability_ge_0.80",
                )
            if "cluster_member_selected" in member_sky.columns:
                selected = member_sky.loc[member_sky["cluster_member_selected"].map(_bool_value)]
                if not selected.empty:
                    return (
                        float(pd.to_numeric(selected["ra"], errors="coerce").median()),
                        float(pd.to_numeric(selected["dec"], errors="coerce").median()),
                        "median_ra_dec_of_selected_members",
                    )
    master_sky = _finite_sky(master)
    if not master_sky.empty:
        return (
            float(pd.to_numeric(master_sky["ra"], errors="coerce").median()),
            float(pd.to_numeric(master_sky["dec"], errors="coerce").median()),
            "median_ra_dec_of_master_catalog",
        )
    return float("nan"), float("nan"), "no_finite_sky_coordinates"


def _add_projected_offsets(
    df: pd.DataFrame,
    spec: ClusterSpec,
    *,
    center_ra: float,
    center_dec: float,
) -> pd.DataFrame:
    sky = _finite_sky(df)
    if sky.empty or not (np.isfinite(center_ra) and np.isfinite(center_dec)):
        result = sky.copy()
        result["x_kpc"] = np.nan
        result["y_kpc"] = np.nan
        result["radius_kpc"] = np.nan
        return result
    coords = SkyCoord(
        pd.to_numeric(sky["ra"], errors="coerce").to_numpy(dtype=float) * u.deg,
        pd.to_numeric(sky["dec"], errors="coerce").to_numpy(dtype=float) * u.deg,
    )
    center = SkyCoord(float(center_ra) * u.deg, float(center_dec) * u.deg)
    dlon, dlat = center.spherical_offsets_to(coords)
    scale = kpc_per_arcsec(spec.z_lens)
    result = sky.copy()
    result["x_kpc"] = dlon.to(u.arcsec).value * scale
    result["y_kpc"] = -dlat.to(u.arcsec).value * scale
    result["radius_kpc"] = np.hypot(result["x_kpc"], result["y_kpc"])
    return result


def _select_spatial_plot_rows(
    df: pd.DataFrame | None,
    spec: ClusterSpec,
    *,
    center_ra: float,
    center_dec: float,
    max_rows: int,
    probability_column: str | None,
    fov_half_width_kpc: float = PLOT_FOV_HALF_WIDTH_KPC,
    extra_sort_columns: Iterable[str] = (),
    label: str = "rows",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df is None:
        return pd.DataFrame(), _spatial_metadata(
            center_ra=center_ra,
            center_dec=center_dec,
            fov_half_width_kpc=fov_half_width_kpc,
            selection_note=f"missing {label}",
        )
    projected = _add_projected_offsets(df, spec, center_ra=center_ra, center_dec=center_dec)
    if projected.empty:
        return projected, _spatial_metadata(
            center_ra=center_ra,
            center_dec=center_dec,
            fov_half_width_kpc=fov_half_width_kpc,
            input_rows=len(df),
            selection_note=f"no finite sky coordinates for {label}",
        )
    fov_mask = (
        np.isfinite(projected["x_kpc"])
        & np.isfinite(projected["y_kpc"])
        & (projected["x_kpc"].abs() <= float(fov_half_width_kpc))
        & (projected["y_kpc"].abs() <= float(fov_half_width_kpc))
    )
    fov = projected.loc[fov_mask].copy()
    sort_columns: list[str] = []
    ascending: list[bool] = []
    sort_note = "deterministic object_id order"
    if probability_column is not None and probability_column in fov.columns:
        fov["_plot_probability"] = pd.to_numeric(fov[probability_column], errors="coerce").fillna(0.0)
        sort_columns.append("_plot_probability")
        ascending.append(False)
        sort_note = f"top {max_rows} by {probability_column}"
    for column in extra_sort_columns:
        if column in fov.columns:
            if column == "family_probability":
                fov["_plot_family_probability"] = pd.to_numeric(fov[column], errors="coerce").fillna(0.0)
                sort_columns.append("_plot_family_probability")
                ascending.append(False)
            else:
                sort_columns.append(column)
                ascending.append(True)
    for column in ("candidate_family_id", "object_id", "ra", "dec"):
        if column in fov.columns and column not in sort_columns:
            sort_columns.append(column)
            ascending.append(True)
    if sort_columns:
        fov = fov.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    selected = fov.head(int(max_rows)).drop(
        columns=["_plot_probability", "_plot_family_probability"],
        errors="ignore",
    )
    metadata = _spatial_metadata(
        center_ra=center_ra,
        center_dec=center_dec,
        fov_half_width_kpc=fov_half_width_kpc,
        input_rows=len(df),
        fov_rows=len(fov),
        plotted_rows=len(selected),
        selection_note=f"{label}: abs(x_kpc)<=500 and abs(y_kpc)<=500; {sort_note}",
    )
    return selected, metadata


def _merge_spatial_metadata(*items: dict[str, Any], selection_note: str = "", image_render_mode: str = "") -> dict[str, Any]:
    valid_items = [item for item in items if item]
    if not valid_items:
        return _spatial_metadata(selection_note=selection_note, image_render_mode=image_render_mode)
    first = valid_items[0]
    notes = [str(item.get("spatial_selection_note", "")) for item in valid_items if item.get("spatial_selection_note")]
    if selection_note:
        notes.append(selection_note)
    return _spatial_metadata(
        center_ra=float(first.get("plot_center_ra", np.nan)),
        center_dec=float(first.get("plot_center_dec", np.nan)),
        fov_half_width_kpc=float(first.get("plot_fov_half_width_kpc", PLOT_FOV_HALF_WIDTH_KPC)),
        input_rows=sum(int(item.get("n_spatial_input_rows", 0)) for item in valid_items),
        fov_rows=sum(int(item.get("n_spatial_fov_rows", 0)) for item in valid_items),
        plotted_rows=sum(int(item.get("n_spatial_plotted_rows", 0)) for item in valid_items),
        selection_note=" | ".join(notes),
        image_render_mode=image_render_mode or str(first.get("image_render_mode", "")),
    )


def _finish_sky_axes(ax: plt.Axes, *, title: str) -> None:
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(title)
    ax.grid(alpha=0.18, linewidth=0.5)
    try:
        ax.invert_xaxis()
    except Exception:
        pass


def _finish_kpc_axes(ax: plt.Axes, *, title: str, dark: bool = False) -> None:
    color = "white" if dark else "black"
    grid_color = "white" if dark else "0.3"
    ax.set_xlim(-PLOT_FOV_HALF_WIDTH_KPC, PLOT_FOV_HALF_WIDTH_KPC)
    ax.set_ylim(-PLOT_FOV_HALF_WIDTH_KPC, PLOT_FOV_HALF_WIDTH_KPC)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Delta x [kpc]", color=color)
    ax.set_ylabel("Delta y [kpc]", color=color)
    ax.set_title(title, color=color)
    ax.tick_params(axis="both", colors=color)
    for spine in ax.spines.values():
        spine.set_color(color)
    ax.grid(alpha=0.16, linewidth=0.5, color=grid_color)


def _find_background_path(image_dir: Path, spec: ClusterSpec) -> Path | None:
    if not image_dir.exists():
        return None
    tokens = {spec.key.lower(), spec.shipley_cluster.lower().replace("-clu", "")}
    tokens.update(slug.lower() for slug in spec.pagul_slugs)
    candidates: list[Path] = []
    for path in image_dir.rglob("*"):
        name = path.name.lower()
        if not path.is_file():
            continue
        if not (name.endswith(".fits") or name.endswith(".fit") or name.endswith(".fits.gz")):
            continue
        if any(token in name for token in tokens):
            candidates.append(path)
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        band_score = 0
        for preferred in ("f160w", "f814w", "f606w"):
            if preferred in name:
                break
            band_score += 1
        return band_score, len(path.parts), str(path)

    return sorted(candidates, key=score)[0]


def _background_from_fits(path: Path) -> dict[str, Any]:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = np.asarray(hdu.data, dtype=float)
            while data.ndim > 2:
                data = data[0]
            if data.ndim != 2 or not np.isfinite(data).any():
                continue
            wcs = WCS(hdu.header)
            if not getattr(wcs, "has_celestial", False):
                continue
            ny, nx = data.shape
            x = np.array([0.0, float(nx - 1), float(nx - 1), 0.0])
            y = np.array([0.0, 0.0, float(ny - 1), float(ny - 1)])
            ra, dec = wcs.celestial.pixel_to_world_values(x, y)
            if not (np.isfinite(ra).all() and np.isfinite(dec).all()):
                continue
            return {
                "path": path,
                "data": data,
                "extent": (float(np.nanmin(ra)), float(np.nanmax(ra)), float(np.nanmin(dec)), float(np.nanmax(dec))),
            }
    raise ValueError("no 2D celestial WCS image extension found")


def _load_plot_background(spec: ClusterSpec, image_dir: Path, mode: str) -> tuple[dict[str, Any] | None, str]:
    if mode == "never":
        return None, "background disabled"
    path = _find_background_path(image_dir, spec)
    if path is None:
        message = f"No usable FITS background found for {spec.key} under {image_dir}"
        if mode == "required":
            raise MissingCatalogError(message)
        return None, message
    try:
        return _background_from_fits(path), ""
    except Exception as exc:
        message = f"Could not load FITS background for {spec.key}: {path} ({exc})"
        if mode == "required":
            raise MissingCatalogError(message) from exc
        return None, message


def _draw_background(ax: plt.Axes, background: dict[str, Any] | None) -> None:
    if background is None:
        return
    data = np.asarray(background["data"], dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    vmin, vmax = np.nanpercentile(finite, [2.0, 98.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    ax.imshow(
        data,
        origin="lower",
        cmap="gray_r",
        extent=background["extent"],
        vmin=vmin,
        vmax=vmax,
        alpha=0.75,
        aspect="auto",
        zorder=0,
    )


def _prefer_image_scale(paths: Sequence[Path], root: Path, image_scale: str) -> list[Path]:
    if image_scale == "auto":
        return list(paths)
    scale_token = image_scale.lower()
    preferred = [path for path in paths if scale_token in str(path.relative_to(root)).lower()]
    return preferred or list(paths)


def _find_background_band_paths(image_dir: Path, spec: ClusterSpec, *, image_scale: str = DEFAULT_IMAGE_SCALE) -> dict[str, Path]:
    if not image_dir.exists():
        return {}
    if image_scale not in IMAGE_SCALE_CHOICES:
        raise ValueError(f"Unsupported image scale {image_scale!r}. Valid choices: {IMAGE_SCALE_CHOICES}.")
    tokens = {spec.key.lower(), spec.shipley_cluster.lower().replace("-clu", "")}
    tokens.update(slug.lower() for slug in spec.pagul_slugs)
    by_band: dict[str, list[Path]] = {band: [] for band in PLOT_RGB_BANDS}
    fallback: list[Path] = []
    for path in image_dir.rglob("*"):
        name = path.name.lower()
        if not path.is_file():
            continue
        if not (name.endswith(".fits") or name.endswith(".fit") or name.endswith(".fits.gz")):
            continue
        if not any(token in name for token in tokens):
            continue
        fallback.append(path)
        for band in PLOT_RGB_BANDS:
            if band.lower() in name:
                by_band[band].append(path)
    result: dict[str, Path] = {}
    for band, paths in by_band.items():
        if paths:
            scaled_paths = _prefer_image_scale(paths, image_dir, image_scale)
            result[band] = sorted(scaled_paths, key=lambda path: (len(path.parts), str(path)))[0]
    if not result and fallback:
        scaled_fallback = _prefer_image_scale(fallback, image_dir, image_scale)
        result["grayscale"] = sorted(scaled_fallback, key=lambda path: (len(path.parts), str(path)))[0]
    return result


def _load_fits_crop_for_overlay(
    path: Path,
    *,
    center_ra: float,
    center_dec: float,
    spec: ClusterSpec,
) -> OverlayCrop:
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = hdu.data
            while getattr(data, "ndim", 0) > 2:
                data = data[0]
            if getattr(data, "ndim", 0) != 2:
                continue
            wcs = WCS(hdu.header)
            if not getattr(wcs, "has_celestial", False):
                continue
            wcs = wcs.celestial
            center_x, center_y = wcs.wcs_world2pix(float(center_ra), float(center_dec), 0)
            pixel_scale_arcsec = float(np.nanmean(np.abs(proj_plane_pixel_scales(wcs)))) * 3600.0
            if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0.0:
                raise ValueError(f"Invalid pixel scale for {path}")
            kpc_per_pixel = pixel_scale_arcsec * kpc_per_arcsec(spec.z_lens)
            half_size_pix = max(1, int(np.ceil(PLOT_FOV_HALF_WIDTH_KPC / kpc_per_pixel)))
            stride = max(1, int(np.ceil((2 * half_size_pix) / PLOT_IMAGE_MAX_PIXELS)))
            ny, nx = data.shape
            x_center = int(np.round(float(center_x)))
            y_center = int(np.round(float(center_y)))
            x_min = max(0, x_center - half_size_pix)
            x_max = min(nx, x_center + half_size_pix)
            y_min = max(0, y_center - half_size_pix)
            y_max = min(ny, y_center + half_size_pix)
            if x_min >= x_max or y_min >= y_max:
                raise ValueError(f"Requested 1 Mpc crop is outside {path}")
            crop = hdu.section[y_min:y_max:stride, x_min:x_max:stride]
            data = np.nan_to_num(np.asarray(crop, dtype=np.float32), nan=0.0)
            return OverlayCrop(
                data=data,
                extent=_overlay_crop_extent(
                    wcs,
                    data.shape,
                    x_min=x_min,
                    y_min=y_min,
                    stride=stride,
                    center_ra=center_ra,
                    center_dec=center_dec,
                    z_lens=spec.z_lens,
                ),
                wcs=wcs,
                x_min=int(x_min),
                y_min=int(y_min),
                stride=int(stride),
                center_ra=float(center_ra),
                center_dec=float(center_dec),
                z_lens=float(spec.z_lens),
            )
    raise ValueError(f"No 2D celestial WCS image found in {path}")


def _overlay_crop_extent(
    wcs: WCS,
    shape: tuple[int, int],
    *,
    x_min: int,
    y_min: int,
    stride: int,
    center_ra: float,
    center_dec: float,
    z_lens: float,
) -> tuple[float, float, float, float]:
    n_y, n_x = int(shape[0]), int(shape[1])
    x0 = float(x_min) - 0.5 * float(stride)
    x1 = float(x_min) + (float(n_x) - 0.5) * float(stride)
    y0 = float(y_min) - 0.5 * float(stride)
    y1 = float(y_min) + (float(n_y) - 0.5) * float(stride)
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    left_ra, left_dec = wcs.wcs_pix2world(x0, y_mid, 0)
    right_ra, right_dec = wcs.wcs_pix2world(x1, y_mid, 0)
    bottom_ra, bottom_dec = wcs.wcs_pix2world(x_mid, y0, 0)
    top_ra, top_dec = wcs.wcs_pix2world(x_mid, y1, 0)
    center = SkyCoord(float(center_ra) * u.deg, float(center_dec) * u.deg)
    scale = kpc_per_arcsec(float(z_lens))

    def _offset(ra: float, dec: float) -> tuple[float, float]:
        coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
        dlon, dlat = center.spherical_offsets_to(coord)
        return float(dlon.to(u.arcsec).value * scale), float(-dlat.to(u.arcsec).value * scale)

    left_x, _left_y = _offset(left_ra, left_dec)
    right_x, _right_y = _offset(right_ra, right_dec)
    _bottom_x, bottom_y = _offset(bottom_ra, bottom_dec)
    _top_x, top_y = _offset(top_ra, top_dec)
    return (left_x, right_x, bottom_y, top_y)


def _overlay_crop_with_shape(crop: OverlayCrop, shape: tuple[int, int]) -> OverlayCrop:
    n_y, n_x = int(shape[0]), int(shape[1])
    data = crop.data[:n_y, :n_x]
    return OverlayCrop(
        data=data,
        extent=_overlay_crop_extent(
            crop.wcs,
            data.shape,
            x_min=crop.x_min,
            y_min=crop.y_min,
            stride=crop.stride,
            center_ra=crop.center_ra,
            center_dec=crop.center_dec,
            z_lens=crop.z_lens,
        ),
        wcs=crop.wcs,
        x_min=crop.x_min,
        y_min=crop.y_min,
        stride=crop.stride,
        center_ra=crop.center_ra,
        center_dec=crop.center_dec,
        z_lens=crop.z_lens,
    )


def _trim_overlay_crops(crops: Iterable[OverlayCrop]) -> list[OverlayCrop]:
    valid = [crop for crop in crops if np.asarray(crop.data).ndim == 2]
    if not valid:
        return []
    min_y = min(crop.data.shape[0] for crop in valid)
    min_x = min(crop.data.shape[1] for crop in valid)
    return [_overlay_crop_with_shape(crop, (min_y, min_x)) for crop in valid]


def _trim_to_common_shape(arrays: Iterable[np.ndarray]) -> list[np.ndarray]:
    valid = [np.asarray(array) for array in arrays if np.asarray(array).ndim == 2]
    if not valid:
        return []
    min_y = min(array.shape[0] for array in valid)
    min_x = min(array.shape[1] for array in valid)
    return [array[:min_y, :min_x] for array in valid]


def _missing_cutout_rgb_message(image_dir: Path, spec: ClusterSpec, missing_bands: Sequence[str]) -> str:
    return (
        f"Missing RGB FITS image(s) for candidate family cutouts for {spec.key} under {image_dir}: "
        f"{', '.join(str(band) for band in missing_bands)}"
    )


def _header_photometry_keyword(headers: Sequence[fits.Header], keyword: str) -> float | None:
    for header in headers:
        value = header.get(keyword)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0.0:
            return number
    return None


def _measure_cutout_image_background(data: Any, *, max_samples: int = 400_000) -> tuple[float, float]:
    ny, nx = data.shape
    stride = max(1, int(math.sqrt(max(1, (ny * nx) / float(max_samples)))))
    sample = np.asarray(data[::stride, ::stride], dtype=np.float32)
    values = sample[np.isfinite(sample) & (sample != 0.0)]
    if values.size == 0:
        return 0.0, 0.0
    sigma = 0.0
    for _ in range(5):
        median = float(np.median(values))
        sigma = 1.4826 * float(np.median(np.abs(values - median)))
        if sigma <= 0.0:
            break
        kept = values[np.abs(values - median) < 3.0 * sigma]
        if kept.size == values.size or kept.size == 0:
            break
        values = kept
    return float(np.median(values)), float(sigma)


def _default_family_cutout_minimum(
    band_images: dict[str, PlotBandImage],
    bands: Sequence[str],
    fluxscales: dict[str, float],
) -> float:
    sigmas = []
    for band in bands:
        band_key = str(band)
        sigma = float(getattr(band_images.get(band_key), "background_sigma", 0.0) or 0.0)
        if sigma > 0.0:
            sigmas.append(sigma * float(fluxscales.get(band_key, 1.0)))
    if not sigmas:
        return 0.0
    return CALIBRATED_RGB_MINIMUM_SKY_SIGMA * float(np.median(sigmas))


def _hff_family_cutout_channel_weights(bands: Sequence[str]) -> dict[str, dict[str, float]]:
    available = {str(band) for band in bands}
    weights: dict[str, dict[str, float]] = {}
    for role, role_weights in DEFAULT_HFF_RGB_CHANNEL_WEIGHTS.items():
        selected = {band: float(weight) for band, weight in role_weights.items() if band in available}
        if selected:
            weights[role] = selected
    return weights


def _build_family_cutout_rgb_display(
    band_images: dict[str, PlotBandImage],
    bands: Sequence[str],
) -> RGBDisplayConfig:
    band_keys = tuple(str(band) for band in bands)
    channel_weights = _hff_family_cutout_channel_weights(band_keys)
    reference_band = DEFAULT_HFF_RGB_REFERENCE_BAND if DEFAULT_HFF_RGB_REFERENCE_BAND in band_keys else band_keys[-1]
    fluxscales = compute_fnu_band_fluxscales(
        {band: getattr(band_images.get(band), "photflam", None) for band in band_keys},
        {band: getattr(band_images.get(band), "photplam", None) for band in band_keys},
        reference_band=reference_band,
    )
    backgrounds = {band: float(getattr(band_images.get(band), "background", 0.0) or 0.0) for band in band_keys}
    if len(band_keys) > 3 and all(channel_weights.get(role) for role in ("blue", "green", "red")):
        return CalibratedRGBDisplayConfig(
            q=DEFAULT_HFF_RGB_Q,
            stretch=DEFAULT_HFF_RGB_STRETCH,
            channel_gains=dict(DEFAULT_HFF_RGB_CHANNEL_GAINS),
            minimum=DEFAULT_HFF_RGB_MINIMUM,
            band_backgrounds=backgrounds,
            band_fluxscales=fluxscales,
            warm_highlight_desaturation=DEFAULT_HFF_RGB_WARM_HIGHLIGHT_DESATURATION,
            channel_weights=channel_weights,
            highlight_knee=DEFAULT_HFF_RGB_HIGHLIGHT_KNEE,
            highlight_ceiling=DEFAULT_HFF_RGB_HIGHLIGHT_CEILING,
            highlight_softness=DEFAULT_HFF_RGB_HIGHLIGHT_SOFTNESS,
        )
    display = CalibratedRGBDisplayConfig(
        q=6.5,
        stretch=0.0165,
        channel_gains={"red": 0.68, "green": 0.75, "blue": 3.5},
        minimum=_default_family_cutout_minimum(band_images, band_keys, fluxscales),
        band_backgrounds=backgrounds,
        band_fluxscales=fluxscales,
    )
    if len(band_keys) == 3:
        return display
    raise ValueError("--family-cutout-bands must either provide exactly three bands or the HFF multiband RGB set.")


def _find_family_cutout_band_paths(
    image_dir: Path,
    spec: ClusterSpec,
    bands: Sequence[str],
    *,
    image_scale: str = DEFAULT_IMAGE_SCALE,
) -> dict[str, Path]:
    if len(tuple(bands)) < 3:
        raise ValueError("--family-cutout-bands must provide at least three bands.")
    if image_scale not in IMAGE_SCALE_CHOICES:
        raise ValueError(f"Unsupported image scale {image_scale!r}. Valid choices: {IMAGE_SCALE_CHOICES}.")
    root = Path(image_dir)
    tokens = {spec.key.lower(), spec.shipley_cluster.lower().replace("-clu", "")}
    tokens.update(slug.lower() for slug in spec.pagul_slugs)
    result: dict[str, Path] = {}
    missing: list[str] = []
    for band in bands:
        band_name = str(band)
        band_token = band_name.lower()
        candidates: list[Path] = []
        if root.exists():
            for path in root.rglob("*"):
                name = path.name.lower()
                if not path.is_file():
                    continue
                if not (name.endswith(".fits") or name.endswith(".fit") or name.endswith(".fits.gz")):
                    continue
                haystack = str(path.relative_to(root)).lower()
                if band_token in haystack and any(token in haystack for token in tokens):
                    candidates.append(path)
        if not candidates:
            missing.append(band_name)
            continue
        candidates = _prefer_image_scale(candidates, root, image_scale)
        result[band_name] = sorted(
            candidates,
            key=lambda path: (0 if "_drz" in path.name.lower() else 1, len(path.parts), str(path)),
        )[0]
    if missing:
        raise FileNotFoundError(_missing_cutout_rgb_message(root, spec, missing))
    return result


def _load_family_cutout_band_image(band: str, path: Path) -> PlotBandImage:
    with fits.open(path, memmap=True) as hdul:
        primary_header = hdul[0].header
        for hdu_index, hdu in enumerate(hdul):
            if hdu.data is None:
                continue
            data = hdu.data
            while getattr(data, "ndim", 0) > 2:
                data = data[0]
            if getattr(data, "ndim", 0) != 2:
                continue
            wcs = WCS(hdu.header)
            if not getattr(wcs, "has_celestial", False):
                continue
            celestial = wcs.celestial
            scales = np.asarray(proj_plane_pixel_scales(celestial), dtype=float) * 3600.0
            pixel_scale = float(np.nanmean(np.abs(scales)))
            if not np.isfinite(pixel_scale) or pixel_scale <= 0.0:
                continue
            ny, nx = data.shape
            headers = (hdu.header, primary_header)
            background, background_sigma = _measure_cutout_image_background(data)
            return PlotBandImage(
                band=str(band),
                path=Path(path),
                hdu_index=int(hdu_index),
                shape=(int(ny), int(nx)),
                wcs=celestial,
                pixel_scale_arcsec=pixel_scale,
                photflam=_header_photometry_keyword(headers, "PHOTFLAM"),
                photplam=_header_photometry_keyword(headers, "PHOTPLAM"),
                background=background,
                background_sigma=background_sigma,
            )
    raise ValueError(f"No 2D celestial WCS image found in {path}")


def _load_family_cutout_band_images(
    paths_by_band: dict[str, Path],
    bands: Sequence[str],
) -> dict[str, PlotBandImage]:
    return {str(band): _load_family_cutout_band_image(str(band), paths_by_band[str(band)]) for band in bands}


def _extract_family_cutout(image: PlotBandImage, coord: SkyCoord, *, cutout_size_arcsec: float) -> np.ndarray:
    if cutout_size_arcsec <= 0.0:
        raise ValueError("family cutout size must be positive.")
    npix = max(1, int(math.ceil(float(cutout_size_arcsec) / image.pixel_scale_arcsec)))
    center_x, center_y = image.wcs.world_to_pixel(coord)
    x_center = int(round(float(center_x)))
    y_center = int(round(float(center_y)))
    half = npix // 2
    x0 = x_center - half
    y0 = y_center - half
    x1 = x0 + npix
    y1 = y0 + npix
    ny, nx = image.shape
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(nx, x1)
    src_y1 = min(ny, y1)
    cutout = np.full((npix, npix), np.nan, dtype=np.float32)
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return cutout
    with fits.open(image.path, memmap=True) as hdul:
        data = hdul[image.hdu_index].section[src_y0:src_y1, src_x0:src_x1]
        values = np.asarray(data, dtype=np.float32)
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    cutout[dst_y0 : dst_y0 + values.shape[0], dst_x0 : dst_x0 + values.shape[1]] = values
    return cutout


def _make_family_rgb_cutout(
    cutouts_by_band: dict[str, np.ndarray],
    bands: Sequence[str],
    rgb_display: RGBDisplayConfig,
) -> np.ndarray:
    return make_natural_rgb(cutouts_by_band, bands=bands, display=rgb_display)


def _load_cluster_overlay_background(
    spec: ClusterSpec,
    image_dir: Path,
    mode: str,
    *,
    center_ra: float,
    center_dec: float,
    image_scale: str = DEFAULT_IMAGE_SCALE,
) -> dict[str, Any]:
    if mode == "never":
        return {"mode": "catalog_only", "reason": "background disabled", "paths": [], "image_scale": image_scale}
    paths_by_band = _find_background_band_paths(image_dir, spec, image_scale=image_scale)
    if not paths_by_band:
        message = f"No usable FITS background found for {spec.key} under {image_dir}"
        if mode == "required":
            raise MissingCatalogError(message)
        return {"mode": "catalog_only", "reason": message, "paths": [], "image_scale": image_scale}

    rgb_paths = [paths_by_band.get(band) for band in PLOT_RGB_BANDS]
    if all(path is not None for path in rgb_paths):
        try:
            rgb_path_by_band = {str(band): Path(path) for band, path in zip(PLOT_RGB_BANDS, rgb_paths, strict=True) if path is not None}
            rgb_display = build_rgb_display_from_band_images(
                _load_family_cutout_band_images(rgb_path_by_band, PLOT_RGB_BANDS),
                PLOT_RGB_BANDS,
            )
            b_crop, g_crop, r_crop = _trim_overlay_crops(
                [
                    _load_fits_crop_for_overlay(rgb_paths[0], center_ra=center_ra, center_dec=center_dec, spec=spec),
                    _load_fits_crop_for_overlay(rgb_paths[1], center_ra=center_ra, center_dec=center_dec, spec=spec),
                    _load_fits_crop_for_overlay(rgb_paths[2], center_ra=center_ra, center_dec=center_dec, spec=spec),
                ]
            )
            rgb = make_natural_rgb(
                {str(PLOT_RGB_BANDS[0]): b_crop.data, str(PLOT_RGB_BANDS[1]): g_crop.data, str(PLOT_RGB_BANDS[2]): r_crop.data},
                bands=PLOT_RGB_BANDS,
                display=rgb_display,
            )
            return {
                "mode": "rgb",
                "image": rgb,
                "extent": b_crop.extent,
                "paths": [path for path in rgb_paths if path is not None],
                "reason": "",
                "image_scale": image_scale,
            }
        except Exception as exc:
            if mode == "required":
                raise MissingCatalogError(f"Could not render RGB background for {spec.key}: {exc}") from exc

    grayscale_path = paths_by_band.get("F814W") or paths_by_band.get("F606W") or paths_by_band.get("F435W") or paths_by_band.get("grayscale")
    if grayscale_path is not None:
        try:
            grayscale = _load_fits_crop_for_overlay(grayscale_path, center_ra=center_ra, center_dec=center_dec, spec=spec)
            return {
                "mode": "grayscale",
                "image": grayscale.data,
                "extent": grayscale.extent,
                "paths": [grayscale_path],
                "reason": "",
                "image_scale": image_scale,
            }
        except Exception as exc:
            if mode == "required":
                raise MissingCatalogError(f"Could not render grayscale background for {spec.key}: {exc}") from exc
            return {"mode": "catalog_only", "reason": str(exc), "paths": [], "image_scale": image_scale}

    message = f"No usable FITS background found for {spec.key} under {image_dir}"
    if mode == "required":
        raise MissingCatalogError(message)
    return {"mode": "catalog_only", "reason": message, "paths": [], "image_scale": image_scale}


def _draw_overlay_background(ax: plt.Axes, image_background: dict[str, Any]) -> None:
    render_mode = str(image_background.get("mode", "catalog_only"))
    extent = image_background.get(
        "extent",
        [-PLOT_FOV_HALF_WIDTH_KPC, PLOT_FOV_HALF_WIDTH_KPC, -PLOT_FOV_HALF_WIDTH_KPC, PLOT_FOV_HALF_WIDTH_KPC],
    )
    if render_mode == "rgb":
        ax.imshow(image_background["image"], origin="lower", extent=extent, interpolation="bilinear", zorder=0)
    elif render_mode == "grayscale":
        data = np.asarray(image_background["image"], dtype=float)
        finite = data[np.isfinite(data)]
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, [2.0, 99.0])
            ax.imshow(
                data,
                origin="lower",
                extent=extent,
                interpolation="bilinear",
                cmap="gray_r",
                vmin=vmin,
                vmax=vmax,
                zorder=0,
            )


def _plot_master_sky(
    master: pd.DataFrame,
    spec: ClusterSpec,
    out_dir: Path,
    *,
    center_ra: float,
    center_dec: float,
) -> tuple[list[Path], dict[str, Any]]:
    sky, metadata = _select_spatial_plot_rows(
        master,
        spec,
        center_ra=center_ra,
        center_dec=center_dec,
        max_rows=PLOT_MASTER_CONTEXT_MAX_ROWS,
        probability_column=None,
        label="master context",
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    if not sky.empty:
        source = sky["object_source"].fillna("unknown").astype(str) if "object_source" in sky.columns else pd.Series("unknown", index=sky.index)
        for label in sorted(source.unique()):
            subset = sky.loc[source == label]
            ax.scatter(subset["x_kpc"], subset["y_kpc"], s=5, alpha=0.45, label=label, linewidths=0)
        ax.legend(loc="best", fontsize=8, frameon=False)
    _finish_kpc_axes(ax, title=f"{spec.key} master catalog footprint")
    return _write_diagnostic_png(fig, out_dir / f"{spec.key}_master_sky_footprint.png"), metadata


def _plot_magnitude_histograms(master: pd.DataFrame, spec: ClusterSpec, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.5), sharey=False)
    for ax, band in zip(axes.ravel(), PLOT_KEY_BANDS):
        column = f"mag_{band}"
        values = _safe_numeric_frame_column(master, column).map(_valid_mag).dropna()
        if not values.empty:
            ax.hist(values, bins=35, color="#2f6f9f", alpha=0.85)
            ax.invert_xaxis()
        ax.set_title(band)
        ax.set_xlabel("AB mag")
        ax.set_ylabel("N")
        ax.grid(alpha=0.15)
    fig.suptitle(f"{spec.key} magnitude distributions")
    return _write_diagnostic_png(fig, out_dir / f"{spec.key}_magnitude_histograms.png")


def _plot_redshift_diagnostics(master: pd.DataFrame, spec: ClusterSpec, out_dir: Path) -> list[Path]:
    zspec = _safe_numeric_frame_column(master, "zspec_best").map(valid_redshift).dropna()
    zphot = _safe_numeric_frame_column(master, "zphot_best").map(valid_redshift).dropna()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for ax, values, label in [(axes[0], zspec, "spec-z"), (axes[1], zphot, "photo-z")]:
        if not values.empty:
            ax.hist(values, bins=40, color="#287c6a", alpha=0.85)
        ax.axvline(spec.z_lens, color="crimson", linestyle="--", linewidth=1.2, label="z_lens")
        ax.set_xlabel(label)
        ax.set_ylabel("N")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.15)
    if "zspec_best" in master.columns and "zphot_best" in master.columns:
        zspec_all = _safe_numeric_frame_column(master, "zspec_best").map(valid_redshift)
        zphot_all = _safe_numeric_frame_column(master, "zphot_best").map(valid_redshift)
        mask = np.isfinite(zspec_all) & np.isfinite(zphot_all)
        axes[2].scatter(zspec_all[mask], zphot_all[mask], s=6, alpha=0.45, linewidths=0)
    axes[2].plot([0, 8], [0, 8], color="0.4", linewidth=1.0)
    axes[2].set_xlim(0, max(1.5, float(np.nanmax(zspec)) if len(zspec) else 1.5))
    axes[2].set_ylim(0, max(1.5, float(np.nanmax(zphot)) if len(zphot) else 1.5))
    axes[2].set_xlabel("spec-z")
    axes[2].set_ylabel("photo-z")
    axes[2].grid(alpha=0.15)
    fig.suptitle(f"{spec.key} redshift diagnostics")
    return _write_diagnostic_png(fig, out_dir / f"{spec.key}_redshift_diagnostics.png")


def _plot_match_separations(match_audit: pd.DataFrame, spec: ClusterSpec, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    sep = _safe_numeric_frame_column(match_audit, "separation_arcsec").dropna()
    if not sep.empty:
        match_type = match_audit["match_type"].fillna("match").astype(str) if "match_type" in match_audit.columns else pd.Series("match", index=match_audit.index)
        for label in sorted(match_type.unique()):
            values = sep.loc[match_type == label]
            ax.hist(values, bins=30, alpha=0.65, label=label)
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("Match separation [arcsec]")
    ax.set_ylabel("N")
    ax.set_title(f"{spec.key} catalog match separations")
    ax.grid(alpha=0.15)
    return _write_diagnostic_png(fig, out_dir / f"{spec.key}_match_separation_histogram.png")


def _plot_member_probability(member_scores: pd.DataFrame, spec: ClusterSpec, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    probs = _safe_numeric_frame_column(member_scores, "member_probability").fillna(0.0)
    evidence = (
        member_scores["member_selection_evidence"].fillna("unselected").replace("", "unselected").astype(str)
        if "member_selection_evidence" in member_scores.columns
        else pd.Series("all", index=member_scores.index)
    )
    for label in sorted(evidence.unique()):
        ax.hist(probs.loc[evidence == label], bins=np.linspace(0, 1, 31), alpha=0.6, label=label)
    ax.set_xlabel("Member probability")
    ax.set_ylabel("N")
    ax.set_title(f"{spec.key} member probabilities")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.15)
    return _write_diagnostic_png(fig, out_dir / f"{spec.key}_member_probability_histogram.png")


def _plot_member_velocity(member_scores: pd.DataFrame, spec: ClusterSpec, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    delta_v = _safe_numeric_frame_column(member_scores, "member_delta_v_kms")
    rank = _safe_numeric_frame_column(member_scores, "zspec_best_confidence_rank")
    probability = _safe_numeric_frame_column(member_scores, "member_probability").fillna(0.0)
    mask = np.isfinite(delta_v) & np.isfinite(rank)
    if mask.any():
        scatter = ax.scatter(rank[mask], delta_v[mask], c=probability[mask], s=12, cmap="viridis", vmin=0, vmax=1, alpha=0.65)
        fig.colorbar(scatter, ax=ax, label="Member probability")
    ax.axhline(0.0, color="0.3", linewidth=1.0)
    ax.set_xlabel("Spec-z confidence rank")
    ax.set_ylabel(r"$\Delta v$ [km/s]")
    ax.set_title(f"{spec.key} velocity offsets")
    ax.grid(alpha=0.15)
    return _write_diagnostic_png(fig, out_dir / f"{spec.key}_member_velocity_confidence.png")


def _plot_red_sequence_diagnostic(member_scores: pd.DataFrame, red_sequence: pd.DataFrame, spec: ClusterSpec, out_dir: Path) -> list[Path]:
    n_planes = min(4, len(red_sequence))
    fig, axes = plt.subplots(1, max(1, n_planes), figsize=(4.8 * max(1, n_planes), 4.4), squeeze=False)
    selected = member_scores["cluster_member_selected"].map(_bool_value) if "cluster_member_selected" in member_scores.columns else pd.Series(False, index=member_scores.index)
    if n_planes == 0:
        axes.ravel()[0].text(0.5, 0.5, "No fitted red sequence", ha="center", va="center", transform=axes.ravel()[0].transAxes)
    for ax, fit in zip(axes.ravel(), red_sequence.head(n_planes).itertuples(index=False)):
        blue = _safe_numeric_frame_column(member_scores, f"mag_{fit.blue_band}").map(_valid_mag)
        red = _safe_numeric_frame_column(member_scores, f"mag_{fit.red_band}").map(_valid_mag)
        mag = _safe_numeric_frame_column(member_scores, f"mag_{fit.mag_band}").map(_valid_mag)
        color = blue - red
        valid = np.isfinite(color) & np.isfinite(mag)
        ax.scatter(mag[valid & ~selected], color[valid & ~selected], s=5, color="0.75", alpha=0.35, linewidths=0, label="not selected")
        ax.scatter(mag[valid & selected], color[valid & selected], s=10, color="#1b9e77", alpha=0.8, linewidths=0, label="selected")
        if valid.any():
            x = np.linspace(float(np.nanmin(mag[valid])), float(np.nanmax(mag[valid])), 100)
            y = float(fit.slope) * x + float(fit.intercept)
            scatter = max(float(fit.scatter_mag), 0.05)
            ax.plot(x, y, color="crimson", linewidth=1.3)
            ax.fill_between(x, y - 3 * scatter, y + 3 * scatter, color="crimson", alpha=0.12, linewidth=0)
        ax.set_xlabel(fit.mag_band)
        ax.set_ylabel(f"{fit.blue_band}-{fit.red_band}")
        ax.invert_xaxis()
        ax.grid(alpha=0.15)
    axes.ravel()[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{spec.key} red-sequence diagnostics")
    return _write_diagnostic_png(fig, out_dir / f"{spec.key}_red_sequence_diagnostics.png")


def _plot_member_sky(
    master: pd.DataFrame,
    member_scores: pd.DataFrame,
    spec: ClusterSpec,
    out_dir: Path,
    *,
    center_ra: float,
    center_dec: float,
) -> tuple[list[Path], dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    sky, master_metadata = _select_spatial_plot_rows(
        master,
        spec,
        center_ra=center_ra,
        center_dec=center_dec,
        max_rows=PLOT_MASTER_CONTEXT_MAX_ROWS,
        probability_column=None,
        label="master context",
    )
    if not sky.empty:
        ax.scatter(sky["x_kpc"], sky["y_kpc"], s=3, color="0.78", alpha=0.25, linewidths=0, label="master")
    member_sky, member_metadata = _select_spatial_plot_rows(
        member_scores,
        spec,
        center_ra=center_ra,
        center_dec=center_dec,
        max_rows=PLOT_MEMBER_MAX_ROWS,
        probability_column="member_probability",
        label="cluster members",
    )
    if not member_sky.empty:
        selected = member_sky["cluster_member_selected"].map(_bool_value) if "cluster_member_selected" in member_sky.columns else pd.Series(False, index=member_sky.index)
        lensing = member_sky["member_for_lensing"].map(_bool_value) if "member_for_lensing" in member_sky.columns else pd.Series(False, index=member_sky.index)
        special = member_sky["bcg_special_member_candidate"].map(_bool_value) if "bcg_special_member_candidate" in member_sky.columns else pd.Series(False, index=member_sky.index)
        ax.scatter(member_sky.loc[selected, "x_kpc"], member_sky.loc[selected, "y_kpc"], s=8, color="#1b9e77", alpha=0.65, linewidths=0, label="members")
        ax.scatter(member_sky.loc[lensing, "x_kpc"], member_sky.loc[lensing, "y_kpc"], s=18, facecolors="none", edgecolors="#377eb8", linewidths=0.8, label="lensing")
        ax.scatter(member_sky.loc[special, "x_kpc"], member_sky.loc[special, "y_kpc"], s=48, marker="*", color="#d95f02", label="special")
        ax.legend(loc="best", fontsize=8, frameon=False)
    _finish_kpc_axes(ax, title=f"{spec.key} cluster-member sky map")
    metadata = dict(member_metadata)
    metadata["spatial_selection_note"] = f"{metadata.get('spatial_selection_note', '')} | member_sky_map"
    return _write_diagnostic_png(fig, out_dir / f"{spec.key}_member_sky_map.png"), metadata


def _plot_pair_diagnostics(pairs: pd.DataFrame, spec: ClusterSpec, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    sep = _safe_numeric_frame_column(pairs, "separation_arcsec")
    score = _safe_numeric_frame_column(pairs, "pair_score")
    rms = _safe_numeric_frame_column(pairs, "sed_rms")
    mask_score = np.isfinite(sep) & np.isfinite(score)
    axes[0].scatter(sep[mask_score], score[mask_score], s=8, alpha=0.45, linewidths=0)
    axes[0].set_xlabel("Separation [arcsec]")
    axes[0].set_ylabel("Pair score")
    mask_rms = np.isfinite(rms) & np.isfinite(score)
    axes[1].scatter(rms[mask_rms], score[mask_rms], s=8, alpha=0.45, linewidths=0, color="#7570b3")
    axes[1].set_xlabel("Color RMS [mag]")
    axes[1].set_ylabel("Pair score")
    if "hard_reject_reason" in pairs.columns:
        counts = pairs["hard_reject_reason"].fillna("").replace("", "accepted/scored").value_counts().head(12)
        axes[2].barh(np.arange(len(counts)), counts.to_numpy(), color="#d95f02", alpha=0.75)
        axes[2].set_yticks(np.arange(len(counts)), counts.index)
    axes[2].set_xlabel("N")
    axes[2].set_title("Reject reasons")
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle(f"{spec.key} image-family pair diagnostics")
    return _write_diagnostic_png(fig, out_dir / f"{spec.key}_family_pair_diagnostics.png")


def _plot_family_diagnostics(families: pd.DataFrame, spec: ClusterSpec, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    probability = _safe_numeric_frame_column(families, "family_probability")
    n_images = _safe_numeric_frame_column(families, "n_images")
    span = _safe_numeric_frame_column(families, "max_separation_kpc")
    mask = np.isfinite(probability) & np.isfinite(n_images)
    axes[0].scatter(n_images[mask], probability[mask], s=28, alpha=0.75, linewidths=0)
    axes[0].set_xlabel("Images per family")
    axes[0].set_ylabel("Family probability")
    mask_span = np.isfinite(span) & np.isfinite(probability)
    axes[1].scatter(span[mask_span], probability[mask_span], s=28, alpha=0.75, linewidths=0, color="#1b9e77")
    axes[1].set_xlabel("Max span [kpc]")
    axes[1].set_ylabel("Family probability")
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle(f"{spec.key} candidate family diagnostics")
    return _write_diagnostic_png(fig, out_dir / f"{spec.key}_family_probability_summary.png")


def _plot_family_sky(
    master: pd.DataFrame,
    family_members: pd.DataFrame,
    families: pd.DataFrame | None,
    spec: ClusterSpec,
    out_dir: Path,
    *,
    center_ra: float,
    center_dec: float,
) -> tuple[list[Path], dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    sky, master_metadata = _select_spatial_plot_rows(
        master,
        spec,
        center_ra=center_ra,
        center_dec=center_dec,
        max_rows=PLOT_MASTER_CONTEXT_MAX_ROWS,
        probability_column=None,
        label="master context",
    )
    if not sky.empty:
        ax.scatter(sky["x_kpc"], sky["y_kpc"], s=3, color="0.82", alpha=0.22, linewidths=0)
    family_frame = family_members.copy()
    if families is not None and not families.empty and "candidate_family_id" in family_frame.columns and "candidate_family_id" in families.columns:
        family_frame = family_frame.merge(
            families[["candidate_family_id", "family_probability"]],
            on="candidate_family_id",
            how="left",
        )
    fam_sky, family_metadata = _select_spatial_plot_rows(
        family_frame,
        spec,
        center_ra=center_ra,
        center_dec=center_dec,
        max_rows=PLOT_FAMILY_MEMBER_MAX_ROWS,
        probability_column="membership_probability",
        extra_sort_columns=("family_probability",),
        label="candidate image-family members",
    )
    if not fam_sky.empty and "candidate_family_id" in fam_sky.columns:
        family_ids = fam_sky["candidate_family_id"].astype(str)
        unique_ids = sorted(family_ids.unique())
        cmap = plt.get_cmap("tab20", max(1, len(unique_ids)))
        for idx, family_id in enumerate(unique_ids):
            subset = fam_sky.loc[family_ids == family_id]
            color = cmap(idx % cmap.N)
            ax.scatter(subset["x_kpc"], subset["y_kpc"], s=24, color=color, alpha=0.9, linewidths=0, label=family_id)
            if len(subset) > 1:
                ordered = subset.sort_values(["x_kpc", "y_kpc"])
                ax.plot(ordered["x_kpc"], ordered["y_kpc"], color=color, alpha=0.35, linewidth=0.8)
        if len(unique_ids) <= 12:
            ax.legend(fontsize=6, frameon=False, loc="best")
    _finish_kpc_axes(ax, title=f"{spec.key} candidate image families")
    metadata = dict(family_metadata)
    metadata["spatial_selection_note"] = f"{metadata.get('spatial_selection_note', '')} | family_sky_map"
    return _write_diagnostic_png(fig, out_dir / f"{spec.key}_family_sky_map.png"), metadata


def _plot_publication_overview(
    master: pd.DataFrame,
    member_scores: pd.DataFrame | None,
    family_members: pd.DataFrame | None,
    families: pd.DataFrame | None,
    spec: ClusterSpec,
    out_dir: Path,
    *,
    center_ra: float,
    center_dec: float,
) -> tuple[list[Path], dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    sky, master_metadata = _select_spatial_plot_rows(
        master,
        spec,
        center_ra=center_ra,
        center_dec=center_dec,
        max_rows=PLOT_MASTER_CONTEXT_MAX_ROWS,
        probability_column=None,
        label="master context",
    )
    if not sky.empty:
        ax.scatter(sky["x_kpc"], sky["y_kpc"], s=2.5, color="0.55", alpha=0.25, linewidths=0, label="master")
    metadata_items = [master_metadata]
    if member_scores is not None and not member_scores.empty:
        member_sky, member_metadata = _select_spatial_plot_rows(
            member_scores,
            spec,
            center_ra=center_ra,
            center_dec=center_dec,
            max_rows=PLOT_MEMBER_MAX_ROWS,
            probability_column="member_probability",
            label="cluster members",
        )
        metadata_items.append(member_metadata)
        if not member_sky.empty:
            selected = member_sky["cluster_member_selected"].map(_bool_value) if "cluster_member_selected" in member_sky.columns else pd.Series(False, index=member_sky.index)
            lensing = member_sky["member_for_lensing"].map(_bool_value) if "member_for_lensing" in member_sky.columns else pd.Series(False, index=member_sky.index)
            special = member_sky["bcg_special_member_candidate"].map(_bool_value) if "bcg_special_member_candidate" in member_sky.columns else pd.Series(False, index=member_sky.index)
            ax.scatter(member_sky.loc[selected, "x_kpc"], member_sky.loc[selected, "y_kpc"], s=10, color="#1b9e77", alpha=0.7, linewidths=0, label="members")
            ax.scatter(member_sky.loc[lensing, "x_kpc"], member_sky.loc[lensing, "y_kpc"], s=22, facecolors="none", edgecolors="#1f78b4", linewidths=0.8, label="lensing members")
            ax.scatter(member_sky.loc[special, "x_kpc"], member_sky.loc[special, "y_kpc"], s=60, marker="*", color="#e7298a", label="BCG/special")
    if family_members is not None and not family_members.empty:
        family_frame = family_members.copy()
        if families is not None and not families.empty and "candidate_family_id" in family_frame.columns and "candidate_family_id" in families.columns:
            family_frame = family_frame.merge(
                families[["candidate_family_id", "family_probability"]],
                on="candidate_family_id",
                how="left",
            )
        fam_sky, family_metadata = _select_spatial_plot_rows(
            family_frame,
            spec,
            center_ra=center_ra,
            center_dec=center_dec,
            max_rows=PLOT_FAMILY_MEMBER_MAX_ROWS,
            probability_column="membership_probability",
            extra_sort_columns=("family_probability",),
            label="candidate image-family members",
        )
        metadata_items.append(family_metadata)
        if not fam_sky.empty and "candidate_family_id" in fam_sky.columns:
            family_ids = fam_sky["candidate_family_id"].astype(str)
            unique_ids = sorted(family_ids.unique())
            cmap = plt.get_cmap("tab20", max(1, len(unique_ids)))
            for idx, family_id in enumerate(unique_ids):
                subset = fam_sky.loc[family_ids == family_id]
                color = cmap(idx % cmap.N)
                ax.scatter(subset["x_kpc"], subset["y_kpc"], s=34, color=color, edgecolors="black", linewidths=0.25, alpha=0.95)
                if len(subset) > 1:
                    ordered = subset.sort_values(["x_kpc", "y_kpc"])
                    ax.plot(ordered["x_kpc"], ordered["y_kpc"], color=color, linewidth=0.8, alpha=0.45)
    circle = Circle((0, 0), PLOT_FOV_HALF_WIDTH_KPC, edgecolor="0.35", facecolor="none", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.add_patch(circle)
    ax.scatter([0], [0], marker="+", s=90, color="black", linewidths=1.4, label="center")
    _finish_kpc_axes(ax, title=f"{spec.key.upper()} catalog overview")
    ax.legend(loc="best", fontsize=7, frameon=True, framealpha=0.72)
    metadata = _merge_spatial_metadata(*metadata_items, selection_note="catalog_overview", image_render_mode="catalog_only")
    return _write_publication_fig(fig, out_dir / f"{spec.key}_catalog_overview"), metadata


def _plot_cluster_image_overlay(
    master: pd.DataFrame,
    member_scores: pd.DataFrame | None,
    family_members: pd.DataFrame | None,
    families: pd.DataFrame | None,
    spec: ClusterSpec,
    out_dir: Path,
    image_background: dict[str, Any],
    *,
    center_ra: float,
    center_dec: float,
) -> tuple[list[Path], dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(10.0, 10.0), dpi=160, facecolor="black")
    ax.set_facecolor("black")
    _draw_overlay_background(ax, image_background)
    metadata_items: list[dict[str, Any]] = []

    member_sky = pd.DataFrame()
    if member_scores is not None:
        member_sky, member_metadata = _select_spatial_plot_rows(
            member_scores,
            spec,
            center_ra=center_ra,
            center_dec=center_dec,
            max_rows=PLOT_MEMBER_MAX_ROWS,
            probability_column="member_probability",
            label="cluster members",
        )
        metadata_items.append(member_metadata)
        if not member_sky.empty:
            selected = member_sky["cluster_member_selected"].map(_bool_value) if "cluster_member_selected" in member_sky.columns else pd.Series(True, index=member_sky.index)
            visible_members = member_sky.loc[selected]
            ax.scatter(
                visible_members["x_kpc"],
                visible_members["y_kpc"],
                s=22,
                marker="o",
                facecolors="none",
                edgecolors="white",
                linewidths=0.7,
                alpha=0.95,
                zorder=4,
            )

    fam_sky = pd.DataFrame()
    if family_members is not None:
        family_frame = family_members.copy()
        if families is not None and not families.empty and "candidate_family_id" in family_frame.columns and "candidate_family_id" in families.columns:
            family_frame = family_frame.merge(
                families[["candidate_family_id", "family_probability"]],
                on="candidate_family_id",
                how="left",
            )
        fam_sky, family_metadata = _select_spatial_plot_rows(
            family_frame,
            spec,
            center_ra=center_ra,
            center_dec=center_dec,
            max_rows=PLOT_FAMILY_MEMBER_MAX_ROWS,
            probability_column="membership_probability",
            extra_sort_columns=("family_probability",),
            label="candidate image-family members",
        )
        metadata_items.append(family_metadata)
        if not fam_sky.empty and "candidate_family_id" in fam_sky.columns:
            family_ids = fam_sky["candidate_family_id"].astype(str)
            unique_ids = sorted(family_ids.unique())
            cmap = plt.get_cmap("tab20", max(1, len(unique_ids)))
            for idx, family_id in enumerate(unique_ids):
                subset = fam_sky.loc[family_ids == family_id]
                color = cmap(idx % cmap.N)
                ax.scatter(
                    subset["x_kpc"],
                    subset["y_kpc"],
                    s=46,
                    marker="o",
                    facecolors="none",
                    edgecolors=[color],
                    linewidths=0.8,
                    zorder=8,
                )
                if len(subset) > 1:
                    ordered = subset.sort_values(["x_kpc", "y_kpc"])
                    ax.plot(ordered["x_kpc"], ordered["y_kpc"], color=color, linewidth=0.45, alpha=0.55, zorder=7)

    circle = Circle(
        (0, 0),
        PLOT_FOV_HALF_WIDTH_KPC,
        edgecolor="white",
        facecolor="none",
        linestyle="--",
        linewidth=1.5,
        alpha=0.85,
        zorder=5,
    )
    ax.add_patch(circle)
    ax.scatter([0], [0], marker="+", s=140, c="white", linewidths=1.8, zorder=9)
    _finish_kpc_axes(ax, title=f"{spec.key.upper()} Cluster Image Overlay", dark=True)
    title = ax.title
    title.set_path_effects([pe.withStroke(linewidth=3.0, foreground="black")])

    member_count = int(len(member_sky.loc[member_sky["cluster_member_selected"].map(_bool_value)])) if not member_sky.empty and "cluster_member_selected" in member_sky.columns else int(len(member_sky))
    family_count = int(fam_sky["candidate_family_id"].nunique()) if not fam_sky.empty and "candidate_family_id" in fam_sky.columns else 0
    legend_handles = [
        Line2D([], [], marker="o", color="white", markerfacecolor="none", linestyle="None", markersize=6, alpha=1, label=f"Members (N={member_count:,})"),
        Line2D([], [], marker="o", color="#1f77b4", markerfacecolor="none", linestyle="None", markersize=7, markeredgewidth=0.8, label=f"Image families (N={family_count:,})"),
        Line2D([], [], color="white", linestyle="--", linewidth=1.5, label="500 kpc"),
        Line2D([], [], marker="+", color="white", linestyle="None", markersize=10, label="plot center"),
    ]
    legend = ax.legend(handles=legend_handles, loc="upper right", facecolor="black", edgecolor="white", framealpha=0.82)
    for text in legend.get_texts():
        text.set_color("white")

    render_mode = str(image_background.get("mode", "catalog_only"))
    metadata = _merge_spatial_metadata(
        *metadata_items,
        selection_note="cluster_image_overlay",
        image_render_mode=render_mode,
    )
    metadata["image_scale"] = str(image_background.get("image_scale", ""))
    return _write_publication_fig(fig, out_dir / f"{spec.key}_cluster_image_overlay", dpi=600), metadata


def _plot_publication_red_sequence(member_scores: pd.DataFrame, red_sequence: pd.DataFrame, spec: ClusterSpec, out_dir: Path) -> list[Path]:
    ranked = red_sequence.sort_values("n_used", ascending=False) if "n_used" in red_sequence.columns else red_sequence
    ranked = ranked.head(2)
    fig, axes = plt.subplots(1, max(1, len(ranked)), figsize=(5.0 * max(1, len(ranked)), 4.4), squeeze=False)
    selected = member_scores["cluster_member_selected"].map(_bool_value) if "cluster_member_selected" in member_scores.columns else pd.Series(False, index=member_scores.index)
    spec_member = _safe_numeric_frame_column(member_scores, "member_specz_score").fillna(0.0) >= 0.60
    if ranked.empty:
        axes.ravel()[0].text(0.5, 0.5, "No fitted red sequence", ha="center", va="center", transform=axes.ravel()[0].transAxes)
    for ax, fit in zip(axes.ravel(), ranked.itertuples(index=False)):
        blue = _safe_numeric_frame_column(member_scores, f"mag_{fit.blue_band}").map(_valid_mag)
        red = _safe_numeric_frame_column(member_scores, f"mag_{fit.red_band}").map(_valid_mag)
        mag = _safe_numeric_frame_column(member_scores, f"mag_{fit.mag_band}").map(_valid_mag)
        color = blue - red
        valid = np.isfinite(color) & np.isfinite(mag)
        ax.scatter(mag[valid & ~selected], color[valid & ~selected], s=5, color="0.75", alpha=0.30, linewidths=0, label="not selected")
        ax.scatter(mag[valid & selected & ~spec_member], color[valid & selected & ~spec_member], s=13, color="#1b9e77", alpha=0.75, linewidths=0, label="photometric members")
        ax.scatter(mag[valid & spec_member], color[valid & spec_member], s=16, color="#377eb8", alpha=0.85, linewidths=0, label="spec members")
        if valid.any():
            x = np.linspace(float(np.nanmin(mag[valid])), float(np.nanmax(mag[valid])), 100)
            y = float(fit.slope) * x + float(fit.intercept)
            scatter = max(float(fit.scatter_mag), 0.05)
            ax.plot(x, y, color="crimson", linewidth=1.4)
            ax.fill_between(x, y - 3 * scatter, y + 3 * scatter, color="crimson", alpha=0.12, linewidth=0)
        ax.set_xlabel(f"{fit.mag_band} [AB]")
        ax.set_ylabel(f"{fit.blue_band}-{fit.red_band} [mag]")
        ax.invert_xaxis()
        ax.grid(alpha=0.15)
    axes.ravel()[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{spec.key.upper()} red-sequence member selection")
    return _write_publication_fig(fig, out_dir / f"{spec.key}_red_sequence_selection")


def _plot_publication_family_summary(families: pd.DataFrame, spec: ClusterSpec, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    probability = _safe_numeric_frame_column(families, "family_probability")
    n_images = _safe_numeric_frame_column(families, "n_images")
    span = _safe_numeric_frame_column(families, "max_separation_kpc")
    family_z = _safe_numeric_frame_column(families, "family_z_best").map(valid_redshift)
    valid = np.isfinite(probability)
    axes[0].hist(probability[valid], bins=np.linspace(0, 1, 16), color="#377eb8", alpha=0.8)
    axes[0].set_xlabel("Family probability")
    axes[0].set_ylabel("N")
    mask = np.isfinite(n_images) & np.isfinite(span)
    axes[1].scatter(n_images[mask], span[mask], c=probability[mask], cmap="viridis", vmin=0, vmax=1, s=32)
    axes[1].set_xlabel("Images")
    axes[1].set_ylabel("Span [kpc]")
    mask_z = np.isfinite(family_z) & np.isfinite(probability)
    axes[2].scatter(family_z[mask_z], probability[mask_z], s=32, color="#1b9e77", alpha=0.8)
    axes[2].set_xlabel("Family redshift")
    axes[2].set_ylabel("Probability")
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle(f"{spec.key.upper()} image-family summary")
    return _write_publication_fig(fig, out_dir / f"{spec.key}_image_family_summary")


def _format_cutout_zlabel(value: Any, prefix: str) -> str:
    numeric = valid_redshift(value)
    if not np.isfinite(numeric):
        return f"{prefix}=na"
    return f"{prefix}={numeric:.3g}"


def _format_cutout_crms(value: Any) -> str:
    numeric = _to_float(value)
    if not np.isfinite(numeric):
        return "crms=na"
    return f"crms={numeric:.2f}"


def _family_cutout_member_crms(
    selected_members: pd.DataFrame,
    pairs: pd.DataFrame | None,
) -> dict[tuple[str, str], float]:
    if selected_members.empty or pairs is None or pairs.empty:
        return {}
    required_member = {"candidate_family_id", "object_id"}
    required_pair = {"left_object_id", "right_object_id", "sed_rms"}
    if not required_member.issubset(selected_members.columns) or not required_pair.issubset(pairs.columns):
        return {}

    family_members: dict[str, set[str]] = defaultdict(set)
    for row in selected_members[["candidate_family_id", "object_id"]].itertuples(index=False):
        family_members[str(row.candidate_family_id)].add(str(row.object_id))

    rms_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for family_id, object_ids in family_members.items():
        if len(object_ids) < 2:
            continue
        pair_subset = pairs.loc[
            pairs["left_object_id"].astype(str).isin(object_ids) & pairs["right_object_id"].astype(str).isin(object_ids)
        ]
        if "hard_reject_reason" in pair_subset.columns:
            reject_reason = pair_subset["hard_reject_reason"].fillna("").astype(str).str.strip()
            pair_subset = pair_subset.loc[reject_reason.eq("")]
        for pair in pair_subset.itertuples(index=False):
            left_id = str(getattr(pair, "left_object_id"))
            right_id = str(getattr(pair, "right_object_id"))
            sed_rms = _to_float(getattr(pair, "sed_rms"))
            if not np.isfinite(sed_rms):
                continue
            rms_values[(family_id, left_id)].append(sed_rms)
            rms_values[(family_id, right_id)].append(sed_rms)

    return {
        key: float(np.nanmedian(np.asarray(values, dtype=float)))
        for key, values in rms_values.items()
        if values and np.any(np.isfinite(np.asarray(values, dtype=float)))
    }


def _select_family_cutout_rows(
    families: pd.DataFrame,
    family_members: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if families.empty or family_members.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "candidate_family_id" not in families.columns or "candidate_family_id" not in family_members.columns:
        return pd.DataFrame(), pd.DataFrame()

    ranked_families = families.copy()
    ranked_families["_family_probability_sort"] = _safe_numeric_frame_column(ranked_families, "family_probability").fillna(0.0)
    ranked_families["_family_id_sort"] = ranked_families["candidate_family_id"].astype(str)
    ranked_families = ranked_families.sort_values(
        ["_family_probability_sort", "_family_id_sort"],
        ascending=[False, True],
        kind="mergesort",
    )

    family_ids = list(ranked_families["candidate_family_id"].astype(str))
    members = _finite_sky(family_members)
    if members.empty:
        return ranked_families.drop(columns=["_family_probability_sort", "_family_id_sort"], errors="ignore"), pd.DataFrame()
    members = members.loc[members["candidate_family_id"].astype(str).isin(family_ids)].copy()
    if members.empty:
        return ranked_families.drop(columns=["_family_probability_sort", "_family_id_sort"], errors="ignore"), pd.DataFrame()
    members["_family_order"] = pd.Categorical(members["candidate_family_id"].astype(str), categories=family_ids, ordered=True)
    members["_membership_sort"] = _safe_numeric_frame_column(members, "membership_probability").fillna(0.0)
    members["_object_id_sort"] = members["object_id"].astype(str) if "object_id" in members.columns else members.index.astype(str)
    selected = (
        members.sort_values(["_family_order", "_membership_sort", "_object_id_sort"], ascending=[True, False, True], kind="mergesort")
        .drop(columns=["_family_order", "_membership_sort", "_object_id_sort"], errors="ignore")
    )
    ranked_families = ranked_families.loc[
        ranked_families["candidate_family_id"].astype(str).isin(selected["candidate_family_id"].astype(str).unique())
    ].drop(columns=["_family_probability_sort", "_family_id_sort"], errors="ignore")
    return ranked_families, selected


def _plot_candidate_family_cutouts(
    families: pd.DataFrame,
    family_members: pd.DataFrame,
    pairs: pd.DataFrame | None,
    spec: ClusterSpec,
    out_dir: Path,
    band_images: dict[str, PlotBandImage],
    paths_by_band: dict[str, Path],
    *,
    bands: Sequence[str],
    cutout_size_arcsec: float,
    circle_radius_arcsec: float,
    image_scale: str,
    families_per_page: int,
) -> tuple[list[Path], dict[str, Any]]:
    if len(tuple(bands)) < 3:
        raise ValueError("--family-cutout-bands must provide at least three bands.")
    if families_per_page <= 0:
        raise ValueError("family_cutout_families_per_page must be positive.")
    if circle_radius_arcsec < 0.0:
        raise ValueError("family_cutout_circle_radius_arcsec must be non-negative.")
    selected_families, selected_members = _select_family_cutout_rows(families, family_members)
    if selected_families.empty or selected_members.empty:
        raise ValueError("No candidate families with finite image coordinates are available for cutouts.")

    output = out_dir / f"{spec.key}_candidate_family_cutouts.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    family_ids = list(selected_families["candidate_family_id"].astype(str))
    family_lookup = selected_families.set_index(selected_families["candidate_family_id"].astype(str), drop=False)
    member_crms = _family_cutout_member_crms(selected_members, pairs)
    n_pages = int(math.ceil(len(family_ids) / int(families_per_page)))
    rendered_images = 0
    rgb_display = _build_family_cutout_rgb_display(band_images, bands)

    with PdfPages(output) as pdf:
        for page_index in range(n_pages):
            page_family_ids = family_ids[page_index * families_per_page : (page_index + 1) * families_per_page]
            n_rows = len(page_family_ids)
            page_members = selected_members.loc[selected_members["candidate_family_id"].astype(str).isin(page_family_ids)]
            page_member_counts = page_members.groupby(page_members["candidate_family_id"].astype(str), sort=False).size()
            n_columns = max(1, int(page_member_counts.max()))
            fig, axes = plt.subplots(
                n_rows,
                n_columns,
                figsize=(2.7 * n_columns, 2.55 * n_rows),
                squeeze=False,
                constrained_layout=True,
            )
            fig.patch.set_facecolor("black")
            fig.suptitle(
                f"{spec.key.upper()} candidate image-family cutouts "
                f"(page {page_index + 1}/{n_pages})",
                fontsize=12,
                color="white",
            )
            for row_index, family_id in enumerate(page_family_ids):
                family_rows = selected_members.loc[selected_members["candidate_family_id"].astype(str) == family_id].reset_index(drop=True)
                family_info = family_lookup.loc[family_id]
                family_probability = _to_float(family_info.get("family_probability", np.nan))
                if not np.isfinite(family_probability):
                    family_probability = 0.0
                for col_index in range(n_columns):
                    ax = axes[row_index, col_index]
                    ax.set_facecolor("black")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_color("white")
                        spine.set_linewidth(0.6)
                    if col_index >= len(family_rows):
                        ax.set_axis_off()
                        continue
                    image_row = family_rows.iloc[col_index]
                    coord = SkyCoord(float(image_row["ra"]) * u.deg, float(image_row["dec"]) * u.deg, frame="icrs")
                    cutouts = {
                        str(band): _extract_family_cutout(
                            band_images[str(band)],
                            coord,
                            cutout_size_arcsec=cutout_size_arcsec,
                        )
                        for band in bands
                    }
                    rgb = _make_family_rgb_cutout(cutouts, bands, rgb_display)
                    ax.imshow(rgb, origin="lower", interpolation="nearest")
                    if circle_radius_arcsec > 0.0:
                        circle_radius_pix = float(circle_radius_arcsec) / float(cutout_size_arcsec) * float(rgb.shape[1])
                        ax.add_patch(
                            Circle(
                                ((rgb.shape[1] - 1.0) / 2.0, (rgb.shape[0] - 1.0) / 2.0),
                                radius=circle_radius_pix,
                                edgecolor="0.72",
                                facecolor="none",
                                linewidth=1.1,
                                alpha=0.65,
                            )
                        )
                    membership = _to_float(image_row.get("membership_probability", np.nan))
                    if not np.isfinite(membership):
                        membership = 0.0
                    object_id = str(image_row.get("object_id", "object"))
                    image_crms = member_crms.get((str(family_id), object_id), np.nan)
                    label = (
                        f"{object_id}\n"
                        f"p={membership:.2f}  "
                        f"{_format_cutout_zlabel(image_row.get('zspec_best', np.nan), 'zs')}  "
                        f"{_format_cutout_zlabel(image_row.get('zphot_best', np.nan), 'zp')}\n"
                        f"{_format_cutout_crms(image_crms)}"
                    )
                    ax.text(
                        0.04,
                        0.94,
                        label,
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=6.8,
                        color="white",
                        bbox={"facecolor": "black", "alpha": 0.62, "edgecolor": "none", "pad": 1.6},
                    )
                    if col_index == 0:
                        n_family_images = int(family_info.get("n_images", len(family_rows)))
                        family_crms = _to_float(family_info.get("median_sed_rms", np.nan))
                        min_pair_score = _to_float(family_info.get("min_pair_score", np.nan))
                        min_pair_label = f"{min_pair_score:.2f}" if np.isfinite(min_pair_score) else "na"
                        family_label = (
                            f"{family_id}  P={family_probability:.2f}  N={n_family_images}\n"
                            f"{_format_cutout_crms(family_crms)}  minpair={min_pair_label}"
                        )
                        ax.text(
                            0.04,
                            0.06,
                            family_label,
                            transform=ax.transAxes,
                            va="bottom",
                            ha="left",
                            fontsize=7.4,
                            color="white",
                            bbox={"facecolor": "black", "alpha": 0.62, "edgecolor": "none", "pad": 1.6},
                        )
                    rendered_images += 1
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)

    metadata = {
        "n_cutout_families": int(len(family_ids)),
        "n_cutout_images": int(rendered_images),
        "family_cutout_size_arcsec": float(cutout_size_arcsec),
        "family_cutout_circle_radius_arcsec": float(circle_radius_arcsec),
        "family_cutout_color_rms_label": True,
        "family_cutout_bands": "|".join(str(band) for band in bands),
        "family_cutout_rgb_paths": "|".join(str(paths_by_band[str(band)]) for band in bands),
        "image_render_mode": "rgb",
        "image_scale": image_scale,
    }
    return [output], metadata


def _plot_all_cluster_summary(cluster_counts: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plot_dir = output_dir / "plots" / "publication"
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    if cluster_counts:
        labels = [str(row["cluster_key"]) for row in cluster_counts]
        x = np.arange(len(labels))
        width = 0.18
        series = [
            ("master_rows", "Master rows"),
            ("selected_members", "Members"),
            ("lensing_members", "Lensing"),
            ("families", "Families"),
            ("family_images", "Family images"),
        ]
        for idx, (key, label) in enumerate(series):
            values = [float(row.get(key, 0)) for row in cluster_counts]
            ax.bar(x + (idx - 2) * width, values, width=width, label=label)
        ax.set_xticks(x, labels)
        ax.set_yscale("log")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, frameon=False, ncols=3)
    ax.set_title("HFF catalog output summary")
    ax.grid(axis="y", alpha=0.18)
    return _write_publication_fig(fig, plot_dir / "hff_all_cluster_catalog_summary")


def render_plot_summary(console: Console, manifest_rows: list[dict[str, Any]], manifest_path: Path) -> None:
    table = RichTable(title="HFF Catalog Plot Summary", title_style="bold bright_green")
    table.add_column("Cluster", style="bold cyan", no_wrap=True)
    table.add_column("Generated", justify="right", style="green")
    table.add_column("Skipped", justify="right", style="yellow")
    table.add_column("Failed", justify="right", style="red")
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        by_cluster[str(row["cluster_key"])].append(row)
    for cluster_key in sorted(by_cluster):
        rows = by_cluster[cluster_key]
        table.add_row(
            cluster_key,
            str(sum(row["status"] == "generated" for row in rows)),
            str(sum(row["status"] == "skipped" for row in rows)),
            str(sum(row["status"] == "failed" for row in rows)),
        )
    console.print(table)
    console.print(f"[bold green]Plot manifest:[/bold green] [cyan]{manifest_path}[/cyan]")


def render_master_run_header(console: Console, args: argparse.Namespace, selected_specs: list[ClusterSpec]) -> None:
    table = RichTable(title="HFF Master Catalog Builder", title_style="bold bright_cyan")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white", overflow="fold")
    table.add_row("Clusters", ", ".join(spec.key for spec in selected_specs))
    table.add_row("Pagul2024", str(args.pagul_dir))
    table.add_row("Shipley2018", str(args.shipley_path))
    table.add_row("Lagattuta22", str(args.lagattuta22_path))
    table.add_row("Output", str(args.output_dir))
    table.add_row("Match radius", f"{args.match_radius_arcsec:.3f} arcsec")
    console.print(table)


def render_master_summary(console: Console, manifest_rows: list[dict[str, Any]], manifest_path: Path) -> None:
    table = RichTable(title="Master Catalog Summary", title_style="bold bright_green")
    table.add_column("Cluster", style="bold cyan", no_wrap=True)
    table.add_column("Rows", justify="right", style="white")
    table.add_column("Pagul", justify="right", style="green")
    table.add_column("Shipley", justify="right", style="blue")
    table.add_column("Matched", justify="right", style="bright_blue")
    table.add_column("Unmatched Added", justify="right", style="magenta")
    table.add_column("Flux Scale", style="cyan", no_wrap=True)
    for row in manifest_rows:
        table.add_row(
            str(row["cluster_key"]),
            f"{int(row['n_output_rows']):,}",
            f"{int(row['n_pagul_rows']):,}",
            f"{int(row['n_shipley_rows']):,}",
            f"{int(row['n_shipley_matched']):,}",
            f"{int(row['n_shipley_unmatched_appended']):,}",
            str(row["pagul_flux_scale"]),
        )
    console.print(table)
    console.print(f"[bold green]Wrote {len(manifest_rows)} cluster catalog(s)[/bold green] to [cyan]{manifest_path.parent}[/cyan]")
    console.print(f"[bold green]Manifest:[/bold green] [cyan]{manifest_path}[/cyan]")


def render_family_summary(console: Console, manifest_rows: list[dict[str, Any]], manifest_path: Path) -> None:
    table = RichTable(title="HFF Image-Family Catalog Summary", title_style="bold bright_green")
    table.add_column("Cluster", style="bold cyan", no_wrap=True)
    table.add_column("Candidates", justify="right")
    table.add_column("Pairs", justify="right")
    table.add_column("Accepted", justify="right")
    table.add_column("Families", justify="right")
    table.add_column("Members", justify="right")
    table.add_column("500 kpc", justify="right")
    for row in manifest_rows:
        table.add_row(
            str(row["cluster_key"]),
            f"{int(row['n_candidates']):,}",
            f"{int(row['n_scored_pairs']):,}",
            f"{int(row['n_accepted_pairs']):,}",
            f"{int(row['n_candidate_families']):,}",
            f"{int(row['n_candidate_family_members']):,}",
            f"{float(row['max_family_span_arcsec']):.1f}\"",
        )
    console.print(table)
    console.print(f"[bold green]Wrote image-family catalog(s)[/bold green] to [cyan]{manifest_path.parent}[/cyan]")
    console.print(f"[bold green]Manifest:[/bold green] [cyan]{manifest_path}[/cyan]")


def render_member_summary(console: Console, manifest_rows: list[dict[str, Any]], manifest_path: Path) -> None:
    table = RichTable(title="HFF Cluster-Member Catalog Summary", title_style="bold bright_green")
    table.add_column("Cluster", style="bold cyan", no_wrap=True)
    table.add_column("Rows", justify="right")
    table.add_column("Members", justify="right", style="green")
    table.add_column("Lensing", justify="right", style="bright_blue")
    table.add_column("Special", justify="right", style="magenta")
    table.add_column("Spec Seeds", justify="right", style="yellow")
    table.add_column("Window", justify="right")
    table.add_column("Red Seq", justify="right")
    for row in manifest_rows:
        table.add_row(
            str(row["cluster_key"]),
            f"{int(row['n_master_rows']):,}",
            f"{int(row['n_cluster_members']):,}",
            f"{int(row['n_lensing_members']):,}",
            f"{int(row['n_bcg_special_candidates']):,}",
            f"{int(row['n_velocity_seed_members']):,}",
            f"{float(row['member_velocity_window_kms']):.0f} km/s",
            f"{int(row['n_red_sequence_planes'])}",
        )
    console.print(table)
    console.print(f"[bold green]Wrote cluster-member catalog(s)[/bold green] to [cyan]{manifest_path.parent}[/cyan]")
    console.print(f"[bold green]Manifest:[/bold green] [cyan]{manifest_path}[/cyan]")


def _run_catalog_plot(
    manifest_rows: list[dict[str, Any]],
    *,
    spec: ClusterSpec,
    plot_kind: str,
    plot_name: str,
    plot_func: Any,
    n_master_rows: int,
    n_member_rows: int,
    n_family_rows: int,
    progress: Any | None = None,
    used_background: bool = False,
    background_path: Path | None = None,
    reason: str = "",
) -> None:
    try:
        paths, metadata = _coerce_plot_output(plot_func())
        manifest_rows.append(
            _plot_manifest_row(
                spec=spec,
                plot_kind=plot_kind,
                plot_name=plot_name,
                status="generated",
                paths=paths,
                reason=reason,
                used_background=used_background,
                background_path=background_path,
                n_master_rows=n_master_rows,
                n_member_rows=n_member_rows,
                n_family_rows=n_family_rows,
                plot_center_ra=metadata.get("plot_center_ra", np.nan),
                plot_center_dec=metadata.get("plot_center_dec", np.nan),
                plot_fov_half_width_kpc=metadata.get("plot_fov_half_width_kpc", np.nan),
                n_spatial_input_rows=int(metadata.get("n_spatial_input_rows", 0)),
                n_spatial_fov_rows=int(metadata.get("n_spatial_fov_rows", 0)),
                n_spatial_plotted_rows=int(metadata.get("n_spatial_plotted_rows", 0)),
                spatial_selection_note=str(metadata.get("spatial_selection_note", "")),
                image_render_mode=str(metadata.get("image_render_mode", "")),
                image_scale=str(metadata.get("image_scale", "")),
                n_cutout_families=int(metadata.get("n_cutout_families", 0)),
                n_cutout_images=int(metadata.get("n_cutout_images", 0)),
                family_cutout_size_arcsec=metadata.get("family_cutout_size_arcsec", np.nan),
                family_cutout_circle_radius_arcsec=metadata.get("family_cutout_circle_radius_arcsec", np.nan),
                family_cutout_color_rms_label=_bool_value(metadata.get("family_cutout_color_rms_label", False)),
                family_cutout_bands=str(metadata.get("family_cutout_bands", "")),
                family_cutout_rgb_paths=str(metadata.get("family_cutout_rgb_paths", "")),
            )
        )
    except Exception as exc:
        plt.close("all")
        manifest_rows.append(
            _plot_manifest_row(
                spec=spec,
                plot_kind=plot_kind,
                plot_name=plot_name,
                status="failed",
                reason=str(exc),
                used_background=used_background,
                background_path=background_path,
                n_master_rows=n_master_rows,
                n_member_rows=n_member_rows,
                n_family_rows=n_family_rows,
            )
        )
    if progress is not None:
        progress.advance_step()


def _record_skipped_plot(
    manifest_rows: list[dict[str, Any]],
    *,
    spec: ClusterSpec,
    plot_kind: str,
    plot_name: str,
    reason: str,
    n_master_rows: int,
    n_member_rows: int,
    n_family_rows: int,
    progress: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    metadata = metadata or {}
    manifest_rows.append(
        _plot_manifest_row(
            spec=spec,
            plot_kind=plot_kind,
            plot_name=plot_name,
            status="skipped",
            reason=reason,
            n_master_rows=n_master_rows,
            n_member_rows=n_member_rows,
            n_family_rows=n_family_rows,
            plot_center_ra=metadata.get("plot_center_ra", np.nan),
            plot_center_dec=metadata.get("plot_center_dec", np.nan),
            plot_fov_half_width_kpc=metadata.get("plot_fov_half_width_kpc", np.nan),
            n_spatial_input_rows=int(metadata.get("n_spatial_input_rows", 0)),
            n_spatial_fov_rows=int(metadata.get("n_spatial_fov_rows", 0)),
            n_spatial_plotted_rows=int(metadata.get("n_spatial_plotted_rows", 0)),
            spatial_selection_note=str(metadata.get("spatial_selection_note", "")),
            image_render_mode=str(metadata.get("image_render_mode", "")),
            image_scale=str(metadata.get("image_scale", "")),
            n_cutout_families=int(metadata.get("n_cutout_families", 0)),
            n_cutout_images=int(metadata.get("n_cutout_images", 0)),
            family_cutout_size_arcsec=metadata.get("family_cutout_size_arcsec", np.nan),
            family_cutout_circle_radius_arcsec=metadata.get("family_cutout_circle_radius_arcsec", np.nan),
            family_cutout_color_rms_label=_bool_value(metadata.get("family_cutout_color_rms_label", False)),
            family_cutout_bands=str(metadata.get("family_cutout_bands", "")),
            family_cutout_rgb_paths=str(metadata.get("family_cutout_rgb_paths", "")),
        )
    )
    if progress is not None:
        progress.advance_step()


def _plot_kind_enabled(requested: str, kind: str) -> bool:
    return requested == "all" or requested == kind


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--clusters",
        nargs="+",
        default=[spec.key for spec in CLUSTER_SPECS],
        choices=[spec.key for spec in CLUSTER_SPECS],
    )


def _add_master_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pagul-dir", type=Path, default=Path("data") / "Pagul2024")
    parser.add_argument("--shipley-path", type=Path, default=Path("data") / "Shipley2018.fit")
    parser.add_argument("--match-radius-arcsec", type=float, default=DEFAULT_MATCH_RADIUS_ARCSEC)
    parser.add_argument("--redshift-dir", type=Path, default=DEFAULT_REDSHIFT_DIR)
    parser.add_argument("--redshift-match-radius-arcsec", type=float, default=DEFAULT_REDSHIFT_MATCH_RADIUS_ARCSEC)
    parser.add_argument("--lagattuta22-path", type=Path, default=DEFAULT_LAGATTUTA22_PATH)


def _add_family_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-family-span-kpc", type=float, default=DEFAULT_MAX_FAMILY_SPAN_KPC)
    parser.add_argument("--min-pair-separation-arcsec", type=float, default=DEFAULT_MIN_PAIR_SEPARATION_ARCSEC)
    parser.add_argument("--min-common-bands", type=int, default=DEFAULT_MIN_COMMON_BANDS)
    parser.add_argument("--image-bright-mag-f814w", type=float, default=DEFAULT_IMAGE_BRIGHT_MAG_F814W)
    parser.add_argument("--image-hff-faint-mag-f814w", type=float, default=DEFAULT_IMAGE_HFF_FAINT_MAG_F814W)
    parser.add_argument("--image-outer-faint-mag-f814w", type=float, default=DEFAULT_IMAGE_OUTER_FAINT_MAG_F814W)
    parser.add_argument("--image-min-size-arcsec", type=float, default=DEFAULT_IMAGE_MIN_SIZE_ARCSEC)
    parser.add_argument("--image-size-pixel-scale-arcsec", type=float, default=DEFAULT_IMAGE_SIZE_PIXEL_SCALE_ARCSEC)
    parser.add_argument("--image-photoz-min-mag-f160w", type=float, default=DEFAULT_IMAGE_PHOTOZ_MIN_MAG_F160W)
    parser.add_argument("--image-photoz-max-mag-f160w", type=float, default=DEFAULT_IMAGE_PHOTOZ_MAX_MAG_F160W)
    parser.add_argument("--image-photoz-min-nb-used", type=float, default=DEFAULT_IMAGE_PHOTOZ_MIN_NB_USED)
    parser.add_argument("--image-photoz-max-dz-norm", type=float, default=DEFAULT_IMAGE_PHOTOZ_MAX_DZ_NORM)
    parser.add_argument("--strong-lensing-rescue-faint-mag-f814w", type=float, default=DEFAULT_STRONG_LENSING_RESCUE_FAINT_MAG_F814W)
    parser.add_argument("--strong-lensing-rescue-min-bands", type=int, default=DEFAULT_STRONG_LENSING_RESCUE_MIN_BANDS)
    parser.add_argument("--image-family-fov-kpc", type=float, default=DEFAULT_IMAGE_FAMILY_FOV_KPC)
    parser.add_argument("--family-color-rms-max", type=float, default=DEFAULT_FAMILY_COLOR_RMS_MAX)
    parser.add_argument("--family-photoz-delta-max", type=float, default=DEFAULT_FAMILY_PHOTOZ_DELTA_MAX)
    parser.add_argument("--reference-family-path", type=Path, default=None)
    parser.add_argument("--reference-match-radius-arcsec", type=float, default=DEFAULT_REFERENCE_MATCH_RADIUS_ARCSEC)
    parser.add_argument("--pair-score-threshold", type=float, default=DEFAULT_PAIR_SCORE_THRESHOLD)
    parser.add_argument("--two-image-score-threshold", type=float, default=DEFAULT_TWO_IMAGE_SCORE_THRESHOLD)
    parser.add_argument("--max-families-per-object", type=int, default=DEFAULT_MAX_FAMILIES_PER_OBJECT)
    parser.add_argument("--family-pair-batch-size", type=int, default=DEFAULT_FAMILY_PAIR_BATCH_SIZE)
    parser.add_argument(
        "--family-pair-diagnostics",
        choices=("scored", "accepted", "all"),
        default=DEFAULT_FAMILY_PAIR_DIAGNOSTICS,
    )


def _add_member_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--member-probability-threshold", type=float, default=DEFAULT_MEMBER_PROBABILITY_THRESHOLD)
    parser.add_argument(
        "--lensing-member-probability-threshold",
        type=float,
        default=DEFAULT_LENSING_MEMBER_PROBABILITY_THRESHOLD,
    )
    parser.add_argument("--lensing-bright-mag-f160w", type=float, default=DEFAULT_LENSING_BRIGHT_MAG_F160W)
    parser.add_argument("--member-faint-mag-f814w", type=float, default=DEFAULT_MEMBER_FAINT_MAG_F814W)
    parser.add_argument("--bcg-special-max", type=int, default=DEFAULT_BCG_SPECIAL_MAX)


def _add_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot-kind", choices=("all", "diagnostic", "publication"), default="all")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--image-scale", choices=IMAGE_SCALE_CHOICES, default=DEFAULT_IMAGE_SCALE)
    parser.add_argument("--image-background", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--family-cutout-size-arcsec", type=float, default=DEFAULT_FAMILY_CUTOUT_SIZE_ARCSEC)
    parser.add_argument(
        "--family-cutout-circle-radius-arcsec",
        type=float,
        default=DEFAULT_FAMILY_CUTOUT_CIRCLE_RADIUS_ARCSEC,
    )
    parser.add_argument("--family-cutout-families-per-page", type=int, default=DEFAULT_FAMILY_CUTOUT_FAMILIES_PER_PAGE)
    parser.add_argument(
        "--family-cutout-bands",
        nargs="+",
        default=list(DEFAULT_FAMILY_CUTOUT_BANDS),
        metavar="BAND",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HFF master, cluster-member, probabilistic image-family catalogs, and catalog plots."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    master_parser = subparsers.add_parser("master", help="Build master catalogs only.")
    _add_common_args(master_parser)
    _add_master_args(master_parser)

    members_parser = subparsers.add_parser("members", help="Build cluster-member catalogs from existing master catalogs.")
    _add_common_args(members_parser)
    _add_member_args(members_parser)

    families_parser = subparsers.add_parser("families", help="Build image-family catalogs from existing master catalogs.")
    _add_common_args(families_parser)
    _add_family_args(families_parser)

    all_parser = subparsers.add_parser("all", help="Build master, cluster-member, image-family catalogs, and plots.")
    _add_common_args(all_parser)
    _add_master_args(all_parser)
    _add_member_args(all_parser)
    _add_family_args(all_parser)
    _add_plot_args(all_parser)

    plots_parser = subparsers.add_parser("plots", help="Generate catalog plots from existing CSV outputs.")
    _add_common_args(plots_parser)
    _add_plot_args(plots_parser)

    return parser.parse_args(argv)


def _write_cluster_member_potfile(path: Path, members: pd.DataFrame) -> None:
    lines = ["#REFERENCE 0"]
    if members.empty:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    sortable_members = members.copy()
    if "mag_F814W" not in sortable_members.columns:
        sortable_members["mag_F814W"] = np.nan
    if "object_id" not in sortable_members.columns:
        sortable_members["object_id"] = ""
    for row in sortable_members.sort_values(["mag_F814W", "object_id"], na_position="last").itertuples(index=False):
        if not _bool_value(getattr(row, "member_for_lensing", False)):
            continue
        if _bool_value(getattr(row, "bcg_special_member_candidate", False)):
            continue
        mag = _valid_mag(getattr(row, "mag_F814W", np.nan))
        ra = _to_float(getattr(row, "ra", np.nan))
        dec = _to_float(getattr(row, "dec", np.nan))
        if not (np.isfinite(mag) and np.isfinite(ra) and np.isfinite(dec)):
            continue
        object_id = str(getattr(row, "object_id", "")).replace(" ", "_")
        lines.append(f"{object_id:>24s} {ra: .8f} {dec: .8f} 1.0000 1.0000 0.0 {mag:.4f} 1.0000")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_member_outputs(
    *,
    output_dir: Path,
    spec: ClusterSpec,
    catalog: pd.DataFrame,
    args: argparse.Namespace,
    master_path: Path,
    progress: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scores, red_sequence, manifest = score_cluster_members(
        catalog,
        spec,
        member_probability_threshold=args.member_probability_threshold,
        lensing_member_probability_threshold=args.lensing_member_probability_threshold,
        lensing_bright_mag_f160w=args.lensing_bright_mag_f160w,
        member_faint_mag_f814w=args.member_faint_mag_f814w,
        bcg_special_max=args.bcg_special_max,
        progress=progress,
    )
    score_path, selected_path, potfile_path, red_sequence_path, special_path = cluster_member_catalog_paths(output_dir, spec)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    selected = scores.loc[scores["cluster_member_selected"].map(_bool_value)].copy() if not scores.empty else scores.copy()
    special = scores.loc[scores["bcg_special_member_candidate"].map(_bool_value)].copy() if not scores.empty else scores.copy()
    if progress is not None:
        progress.start_step(f"{spec.key}: writing member catalogs", total=5)
    scores.to_csv(score_path, index=False)
    if progress is not None:
        progress.advance_step()
    selected.to_csv(selected_path, index=False)
    if progress is not None:
        progress.advance_step()
    red_sequence.to_csv(red_sequence_path, index=False)
    if progress is not None:
        progress.advance_step()
    special.to_csv(special_path, index=False)
    if progress is not None:
        progress.advance_step()
    _write_cluster_member_potfile(potfile_path, scores)
    if progress is not None:
        progress.advance_step()
        progress.finish_step()
    manifest.update(
        {
            "master_path": str(master_path),
            "member_scores_path": str(score_path),
            "cluster_members_path": str(selected_path),
            "member_potfile_path": str(potfile_path),
            "red_sequence_path": str(red_sequence_path),
            "bcg_special_path": str(special_path),
        }
    )
    return scores, manifest


def _write_family_outputs(
    *,
    output_dir: Path,
    spec: ClusterSpec,
    catalog: pd.DataFrame,
    args: argparse.Namespace,
    master_path: Path,
    progress: Any | None = None,
) -> dict[str, Any]:
    families, members, pairs, manifest = build_cluster_image_families(
        catalog,
        spec,
        max_family_span_kpc=args.max_family_span_kpc,
        min_pair_separation_arcsec=args.min_pair_separation_arcsec,
        min_common_bands=args.min_common_bands,
        pair_score_threshold=args.pair_score_threshold,
        two_image_score_threshold=args.two_image_score_threshold,
        family_color_rms_max=args.family_color_rms_max,
        family_photoz_delta_max=args.family_photoz_delta_max,
        image_family_fov_kpc=args.image_family_fov_kpc,
        max_families_per_object=args.max_families_per_object,
        family_pair_batch_size=args.family_pair_batch_size,
        family_pair_diagnostics=args.family_pair_diagnostics,
        image_bright_mag_f814w=args.image_bright_mag_f814w,
        image_hff_faint_mag_f814w=args.image_hff_faint_mag_f814w,
        image_outer_faint_mag_f814w=args.image_outer_faint_mag_f814w,
        image_min_size_arcsec=args.image_min_size_arcsec,
        image_size_pixel_scale_arcsec=args.image_size_pixel_scale_arcsec,
        image_photoz_min_mag_f160w=args.image_photoz_min_mag_f160w,
        image_photoz_max_mag_f160w=args.image_photoz_max_mag_f160w,
        image_photoz_min_nb_used=args.image_photoz_min_nb_used,
        image_photoz_max_dz_norm=args.image_photoz_max_dz_norm,
        strong_lensing_rescue_faint_mag_f814w=args.strong_lensing_rescue_faint_mag_f814w,
        strong_lensing_rescue_min_bands=args.strong_lensing_rescue_min_bands,
        progress=progress,
    )
    family_path, member_path, pair_path = family_catalog_paths(output_dir, spec)
    family_path.parent.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress.start_step(f"{spec.key}: writing image-family catalogs", total=3)
    families.to_csv(family_path, index=False)
    if progress is not None:
        progress.advance_step()
    members.to_csv(member_path, index=False)
    if progress is not None:
        progress.advance_step()
    pairs.to_csv(pair_path, index=False)
    if progress is not None:
        progress.advance_step()
        progress.finish_step()
    reference_path = args.reference_family_path or default_reference_family_path(spec)
    if reference_path is not None and Path(reference_path).exists():
        reference_crossmatch_path, reference_recovery_path = reference_family_catalog_paths(output_dir, spec)
        if progress is not None:
            progress.start_step(f"{spec.key}: writing reference family diagnostics", total=2)
        crossmatch, recovery, reference_metrics = build_reference_family_diagnostics(
            reference_path=Path(reference_path),
            master=catalog,
            families=families,
            family_members=members,
            match_radius_arcsec=args.reference_match_radius_arcsec,
        )
        crossmatch.to_csv(reference_crossmatch_path, index=False)
        if progress is not None:
            progress.advance_step()
        recovery.to_csv(reference_recovery_path, index=False)
        if progress is not None:
            progress.advance_step()
            progress.finish_step()
        manifest.update(reference_metrics)
        manifest.update(
            {
                "reference_crossmatch_path": str(reference_crossmatch_path),
                "reference_recovery_path": str(reference_recovery_path),
            }
        )
    else:
        manifest.update(
            {
                "reference_family_path": str(reference_path) if reference_path is not None else "",
                "reference_match_radius_arcsec": float(args.reference_match_radius_arcsec),
                "n_reference_images": 0,
                "n_reference_families": 0,
                "n_reference_master_matches": 0,
                "n_reference_generated_member_matches": 0,
                "n_reference_recoverable_families": 0,
                "n_reference_recovered_families": 0,
                "reference_crossmatch_path": "",
                "reference_recovery_path": "",
            }
        )
    manifest.update(
        {
            "master_path": str(master_path),
            "family_path": str(family_path),
            "member_path": str(member_path),
            "pair_path": str(pair_path),
        }
    )
    return manifest


def run_master_stage(
    args: argparse.Namespace,
    console: Console,
    selected_specs: list[ClusterSpec],
    *,
    build_members: bool = False,
    build_families: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    render_master_run_header(console, args, selected_specs)

    if not args.shipley_path.exists():
        console.print(f"[bold red]Missing Shipley2018 catalog:[/bold red] {args.shipley_path}")
        raise MissingCatalogError(f"Missing Shipley2018 catalog: {args.shipley_path}")
    with console.status("[bold cyan]Checking required Pagul2024 catalogs...[/bold cyan]"):
        pagul_paths = {spec.key: locate_pagul_catalog(spec, args.pagul_dir) for spec in selected_specs}
    console.print(f"[green]Found[/green] {len(pagul_paths)} Pagul2024 catalog(s).")
    with console.status(f"[bold cyan]Loading Shipley2018 catalog from {args.shipley_path}...[/bold cyan]"):
        shipley = _table_to_dataframe(args.shipley_path)
    console.print(f"[green]Loaded[/green] Shipley2018 rows: [bold]{len(shipley):,}[/bold]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    member_manifest_rows: list[dict[str, Any]] = []
    family_manifest_rows: list[dict[str, Any]] = []
    with make_progress(console) as progress:
        reporter = CatalogProgress(progress, total_clusters=len(selected_specs))
        for spec in selected_specs:
            reporter.set_cluster_phase(f"{spec.key}: cluster")
            pagul_path = pagul_paths[spec.key]
            reporter.start_step(f"{spec.key}: loading inputs", total=1)
            pagul_table = _table_to_dataframe(pagul_path)
            reporter.advance_step()
            reporter.finish_step()
            reporter.start_step(f"{spec.key}: loading external redshifts", total=1)
            external_redshifts = load_external_redshift_catalogs(
                args.redshift_dir,
                spec,
                lagattuta22_path=args.lagattuta22_path,
            )
            reporter.advance_step()
            reporter.finish_step()
            catalog, audit, manifest = build_cluster_catalog(
                spec=spec,
                pagul=pagul_table,
                shipley=shipley,
                external_redshifts=external_redshifts,
                match_radius_arcsec=args.match_radius_arcsec,
                redshift_match_radius_arcsec=args.redshift_match_radius_arcsec,
                progress=reporter,
            )
            cluster_output_dir(args.output_dir, spec).mkdir(parents=True, exist_ok=True)
            catalog_path = master_catalog_path(args.output_dir, spec)
            audit_path = match_audit_path(args.output_dir, spec)
            reporter.start_step(f"{spec.key}: writing master CSVs", total=2)
            catalog.to_csv(catalog_path, index=False)
            reporter.advance_step()
            audit.to_csv(audit_path, index=False)
            reporter.advance_step()
            reporter.finish_step()
            manifest.update(
                {
                    "pagul_path": str(pagul_path),
                    "shipley_path": str(args.shipley_path),
                    "redshift_dir": str(args.redshift_dir),
                    "lagattuta22_path": str(args.lagattuta22_path),
                    "catalog_path": str(catalog_path),
                    "audit_path": str(audit_path),
                }
            )
            manifest_rows.append(manifest)
            family_input_catalog = catalog
            if build_members:
                family_input_catalog, member_manifest = _write_member_outputs(
                    output_dir=args.output_dir,
                    spec=spec,
                    catalog=catalog,
                    args=args,
                    master_path=catalog_path,
                    progress=reporter,
                )
                member_manifest_rows.append(member_manifest)
            if build_families:
                family_manifest_rows.append(
                    _write_family_outputs(
                        output_dir=args.output_dir,
                        spec=spec,
                        catalog=family_input_catalog,
                        args=args,
                        master_path=catalog_path,
                        progress=reporter,
                    )
                )
            reporter.advance_cluster()

    manifest_path = args.output_dir / "hff_master_catalog_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    render_master_summary(console, manifest_rows, manifest_path)
    if build_members:
        member_manifest_path = args.output_dir / "hff_cluster_member_manifest.csv"
        pd.DataFrame(member_manifest_rows).to_csv(member_manifest_path, index=False)
        render_member_summary(console, member_manifest_rows, member_manifest_path)
    if build_families:
        family_manifest_path = args.output_dir / "hff_image_family_manifest.csv"
        pd.DataFrame(family_manifest_rows).to_csv(family_manifest_path, index=False)
        render_family_summary(console, family_manifest_rows, family_manifest_path)
    return manifest_rows, member_manifest_rows, family_manifest_rows


def run_members_stage(
    args: argparse.Namespace,
    console: Console,
    selected_specs: list[ClusterSpec],
) -> list[dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    member_manifest_rows: list[dict[str, Any]] = []
    with make_progress(console) as progress:
        reporter = CatalogProgress(progress, total_clusters=len(selected_specs))
        for spec in selected_specs:
            reporter.set_cluster_phase(f"{spec.key}: cluster")
            catalog_path = master_catalog_path(args.output_dir, spec)
            if not catalog_path.exists():
                raise MissingCatalogError(f"Missing master catalog for {spec.key}: {catalog_path}")
            reporter.start_step(f"{spec.key}: loading master catalog", total=None)
            catalog = pd.read_csv(catalog_path, low_memory=False)
            reporter.finish_step()
            _scores, manifest = _write_member_outputs(
                output_dir=args.output_dir,
                spec=spec,
                catalog=catalog,
                args=args,
                master_path=catalog_path,
                progress=reporter,
            )
            member_manifest_rows.append(manifest)
            reporter.advance_cluster()

    member_manifest_path = args.output_dir / "hff_cluster_member_manifest.csv"
    pd.DataFrame(member_manifest_rows).to_csv(member_manifest_path, index=False)
    render_member_summary(console, member_manifest_rows, member_manifest_path)
    return member_manifest_rows


def run_families_stage(
    args: argparse.Namespace,
    console: Console,
    selected_specs: list[ClusterSpec],
) -> list[dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    family_manifest_rows: list[dict[str, Any]] = []
    with make_progress(console) as progress:
        reporter = CatalogProgress(progress, total_clusters=len(selected_specs))
        for spec in selected_specs:
            reporter.set_cluster_phase(f"{spec.key}: cluster")
            catalog_path = master_catalog_path(args.output_dir, spec)
            if not catalog_path.exists():
                raise MissingCatalogError(f"Missing master catalog for {spec.key}: {catalog_path}")
            score_path, _selected_path, _potfile_path, _red_sequence_path, _special_path = cluster_member_catalog_paths(
                args.output_dir,
                spec,
            )
            if score_path.exists():
                reporter.start_step(f"{spec.key}: loading member scores", total=None)
                catalog = pd.read_csv(score_path, low_memory=False)
            else:
                reporter.start_step(f"{spec.key}: loading master catalog", total=None)
                catalog = pd.read_csv(catalog_path, low_memory=False)
            reporter.finish_step()
            family_manifest_rows.append(
                _write_family_outputs(
                    output_dir=args.output_dir,
                    spec=spec,
                    catalog=catalog,
                    args=args,
                    master_path=catalog_path,
                    progress=reporter,
                )
            )
            reporter.advance_cluster()

    family_manifest_path = args.output_dir / "hff_image_family_manifest.csv"
    pd.DataFrame(family_manifest_rows).to_csv(family_manifest_path, index=False)
    render_family_summary(console, family_manifest_rows, family_manifest_path)
    return family_manifest_rows


def run_plots_stage(
    args: argparse.Namespace,
    console: Console,
    selected_specs: list[ClusterSpec],
) -> list[dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    cluster_counts: list[dict[str, Any]] = []
    diagnostic_enabled = _plot_kind_enabled(args.plot_kind, "diagnostic")
    publication_enabled = _plot_kind_enabled(args.plot_kind, "publication")

    with make_progress(console) as progress:
        total_units = len(selected_specs) + (1 if publication_enabled else 0)
        reporter = CatalogProgress(progress, total_clusters=total_units)
        for spec in selected_specs:
            reporter.set_cluster_phase(f"{spec.key}: plots")
            catalog_path = master_catalog_path(args.output_dir, spec)
            if not catalog_path.exists():
                raise MissingCatalogError(f"Missing master catalog for {spec.key}: {catalog_path}")

            diagnostic_dir, publication_dir = plot_output_dirs(args.output_dir, spec)
            reporter.start_step(f"{spec.key}: loading plot inputs", total=None)
            master = pd.read_csv(catalog_path, low_memory=False)
            match_audit = _read_optional_csv(match_audit_path(args.output_dir, spec))
            score_path, selected_path, _potfile_path, red_sequence_path, _special_path = cluster_member_catalog_paths(
                args.output_dir,
                spec,
            )
            member_scores = _read_optional_csv(score_path)
            if member_scores is None:
                member_scores = _read_optional_csv(selected_path)
            red_sequence = _read_optional_csv(red_sequence_path)
            family_path, family_member_path, pair_path = family_catalog_paths(args.output_dir, spec)
            families = _read_optional_csv(family_path)
            family_members = _read_optional_csv(family_member_path)
            pairs = _read_optional_csv(pair_path)
            center_ra, center_dec, center_note = _choose_plot_center(master, member_scores)
            reporter.finish_step()

            image_background: dict[str, Any] = {"mode": "catalog_only", "reason": "publication plots disabled", "paths": []}
            background_reason = ""
            if publication_enabled:
                reporter.start_step(f"{spec.key}: checking image background", total=None)
                image_background = _load_cluster_overlay_background(
                    spec,
                    args.image_dir,
                    args.image_background,
                    center_ra=center_ra,
                    center_dec=center_dec,
                    image_scale=args.image_scale,
                )
                background_reason = str(image_background.get("reason", ""))
                reporter.finish_step()

            n_master_rows = int(len(master))
            n_member_rows = int(len(member_scores)) if member_scores is not None else 0
            n_family_rows = int(len(families)) if families is not None else 0
            cluster_counts.append(
                {
                    "cluster_key": spec.key,
                    "master_rows": n_master_rows,
                    "selected_members": int(member_scores["cluster_member_selected"].map(_bool_value).sum())
                    if member_scores is not None and "cluster_member_selected" in member_scores.columns
                    else 0,
                    "lensing_members": int(member_scores["member_for_lensing"].map(_bool_value).sum())
                    if member_scores is not None and "member_for_lensing" in member_scores.columns
                    else 0,
                    "families": n_family_rows,
                    "family_images": int(len(family_members)) if family_members is not None else 0,
                }
            )

            n_steps = 0
            if diagnostic_enabled:
                n_steps += 12
            if publication_enabled:
                n_steps += 5
            reporter.start_step(f"{spec.key}: writing plots", total=n_steps)

            if diagnostic_enabled:
                _run_catalog_plot(
                    manifest_rows,
                    spec=spec,
                    plot_kind="diagnostic",
                    plot_name="master_sky_footprint",
                    plot_func=lambda master=master, spec=spec, diagnostic_dir=diagnostic_dir: _plot_master_sky(
                        master,
                        spec,
                        diagnostic_dir,
                        center_ra=center_ra,
                        center_dec=center_dec,
                    ),
                    n_master_rows=n_master_rows,
                    n_member_rows=n_member_rows,
                    n_family_rows=n_family_rows,
                    progress=reporter,
                )
                _run_catalog_plot(
                    manifest_rows,
                    spec=spec,
                    plot_kind="diagnostic",
                    plot_name="magnitude_histograms",
                    plot_func=lambda master=master, spec=spec, diagnostic_dir=diagnostic_dir: _plot_magnitude_histograms(
                        master,
                        spec,
                        diagnostic_dir,
                    ),
                    n_master_rows=n_master_rows,
                    n_member_rows=n_member_rows,
                    n_family_rows=n_family_rows,
                    progress=reporter,
                )
                _run_catalog_plot(
                    manifest_rows,
                    spec=spec,
                    plot_kind="diagnostic",
                    plot_name="redshift_diagnostics",
                    plot_func=lambda master=master, spec=spec, diagnostic_dir=diagnostic_dir: _plot_redshift_diagnostics(
                        master,
                        spec,
                        diagnostic_dir,
                    ),
                    n_master_rows=n_master_rows,
                    n_member_rows=n_member_rows,
                    n_family_rows=n_family_rows,
                    progress=reporter,
                )
                if match_audit is not None:
                    _run_catalog_plot(
                        manifest_rows,
                        spec=spec,
                        plot_kind="diagnostic",
                        plot_name="match_separation_histogram",
                        plot_func=lambda match_audit=match_audit, spec=spec, diagnostic_dir=diagnostic_dir: _plot_match_separations(
                            match_audit,
                            spec,
                            diagnostic_dir,
                        ),
                        n_master_rows=n_master_rows,
                        n_member_rows=n_member_rows,
                        n_family_rows=n_family_rows,
                        progress=reporter,
                    )
                else:
                    _record_skipped_plot(
                        manifest_rows,
                        spec=spec,
                        plot_kind="diagnostic",
                        plot_name="match_separation_histogram",
                        reason=f"Missing match audit: {match_audit_path(args.output_dir, spec)}",
                        n_master_rows=n_master_rows,
                        n_member_rows=n_member_rows,
                        n_family_rows=n_family_rows,
                        progress=reporter,
                    )

                member_plot_specs = [
                    ("member_probability_histogram", _plot_member_probability),
                    ("member_velocity_confidence", _plot_member_velocity),
                    ("member_sky_map", _plot_member_sky),
                ]
                for plot_name, plot_func in member_plot_specs:
                    if member_scores is None:
                        _record_skipped_plot(
                            manifest_rows,
                            spec=spec,
                            plot_kind="diagnostic",
                            plot_name=plot_name,
                            reason=f"Missing member scores: {score_path}",
                            n_master_rows=n_master_rows,
                            n_member_rows=n_member_rows,
                            n_family_rows=n_family_rows,
                            progress=reporter,
                        )
                        continue
                    if plot_name == "member_sky_map":
                        plot_callable = lambda master=master, member_scores=member_scores, spec=spec, diagnostic_dir=diagnostic_dir: _plot_member_sky(
                            master,
                            member_scores,
                            spec,
                            diagnostic_dir,
                            center_ra=center_ra,
                            center_dec=center_dec,
                        )
                    else:
                        plot_callable = lambda member_scores=member_scores, spec=spec, diagnostic_dir=diagnostic_dir, plot_func=plot_func: plot_func(
                            member_scores,
                            spec,
                            diagnostic_dir,
                        )
                    _run_catalog_plot(
                        manifest_rows,
                        spec=spec,
                        plot_kind="diagnostic",
                        plot_name=plot_name,
                        plot_func=plot_callable,
                        n_master_rows=n_master_rows,
                        n_member_rows=n_member_rows,
                        n_family_rows=n_family_rows,
                        progress=reporter,
                    )

                if member_scores is not None and red_sequence is not None:
                    _run_catalog_plot(
                        manifest_rows,
                        spec=spec,
                        plot_kind="diagnostic",
                        plot_name="red_sequence_diagnostics",
                        plot_func=lambda member_scores=member_scores, red_sequence=red_sequence, spec=spec, diagnostic_dir=diagnostic_dir: _plot_red_sequence_diagnostic(
                            member_scores,
                            red_sequence,
                            spec,
                            diagnostic_dir,
                        ),
                        n_master_rows=n_master_rows,
                        n_member_rows=n_member_rows,
                        n_family_rows=n_family_rows,
                        progress=reporter,
                    )
                else:
                    _record_skipped_plot(
                        manifest_rows,
                        spec=spec,
                        plot_kind="diagnostic",
                        plot_name="red_sequence_diagnostics",
                        reason="Missing member scores or red-sequence fit catalog",
                        n_master_rows=n_master_rows,
                        n_member_rows=n_member_rows,
                        n_family_rows=n_family_rows,
                        progress=reporter,
                    )

                family_plot_specs = [
                    ("family_pair_diagnostics", pairs, _plot_pair_diagnostics, pair_path),
                    ("family_probability_summary", families, _plot_family_diagnostics, family_path),
                    ("family_sky_map", family_members, _plot_family_sky, family_member_path),
                ]
                for plot_name, frame, plot_func, required_path in family_plot_specs:
                    if frame is None:
                        _record_skipped_plot(
                            manifest_rows,
                            spec=spec,
                            plot_kind="diagnostic",
                            plot_name=plot_name,
                            reason=f"Missing family catalog: {required_path}",
                            n_master_rows=n_master_rows,
                            n_member_rows=n_member_rows,
                            n_family_rows=n_family_rows,
                            progress=reporter,
                        )
                        continue
                    if plot_name == "family_sky_map":
                        plot_callable = lambda master=master, family_members=family_members, families=families, spec=spec, diagnostic_dir=diagnostic_dir: _plot_family_sky(
                            master,
                            family_members,
                            families,
                            spec,
                            diagnostic_dir,
                            center_ra=center_ra,
                            center_dec=center_dec,
                        )
                    else:
                        plot_callable = lambda frame=frame, spec=spec, diagnostic_dir=diagnostic_dir, plot_func=plot_func: plot_func(
                            frame,
                            spec,
                            diagnostic_dir,
                        )
                    _run_catalog_plot(
                        manifest_rows,
                        spec=spec,
                        plot_kind="diagnostic",
                        plot_name=plot_name,
                        plot_func=plot_callable,
                        n_master_rows=n_master_rows,
                        n_member_rows=n_member_rows,
                        n_family_rows=n_family_rows,
                        progress=reporter,
                    )

            if publication_enabled:
                used_background = str(image_background.get("mode", "catalog_only")) in {"rgb", "grayscale"}
                background_paths = list(image_background.get("paths", []))
                background_path = background_paths[0] if background_paths else None
                _run_catalog_plot(
                    manifest_rows,
                    spec=spec,
                    plot_kind="publication",
                    plot_name="catalog_overview",
                    plot_func=lambda master=master, member_scores=member_scores, family_members=family_members, families=families, spec=spec, publication_dir=publication_dir: _plot_publication_overview(
                        master,
                        member_scores,
                        family_members,
                        families,
                        spec,
                        publication_dir,
                        center_ra=center_ra,
                        center_dec=center_dec,
                    ),
                    n_master_rows=n_master_rows,
                    n_member_rows=n_member_rows,
                    n_family_rows=n_family_rows,
                    progress=reporter,
                    used_background=False,
                    background_path=None,
                    reason=background_reason,
                )
                _run_catalog_plot(
                    manifest_rows,
                    spec=spec,
                    plot_kind="publication",
                    plot_name="cluster_image_overlay",
                    plot_func=lambda master=master, member_scores=member_scores, family_members=family_members, families=families, spec=spec, publication_dir=publication_dir, image_background=image_background: _plot_cluster_image_overlay(
                        master,
                        member_scores,
                        family_members,
                        families,
                        spec,
                        publication_dir,
                        image_background,
                        center_ra=center_ra,
                        center_dec=center_dec,
                    ),
                    n_master_rows=n_master_rows,
                    n_member_rows=n_member_rows,
                    n_family_rows=n_family_rows,
                    progress=reporter,
                    used_background=used_background,
                    background_path=background_path,
                    reason=background_reason or center_note,
                )
                if member_scores is not None and red_sequence is not None:
                    _run_catalog_plot(
                        manifest_rows,
                        spec=spec,
                        plot_kind="publication",
                        plot_name="red_sequence_selection",
                        plot_func=lambda member_scores=member_scores, red_sequence=red_sequence, spec=spec, publication_dir=publication_dir: _plot_publication_red_sequence(
                            member_scores,
                            red_sequence,
                            spec,
                            publication_dir,
                        ),
                        n_master_rows=n_master_rows,
                        n_member_rows=n_member_rows,
                        n_family_rows=n_family_rows,
                        progress=reporter,
                    )
                else:
                    _record_skipped_plot(
                        manifest_rows,
                        spec=spec,
                        plot_kind="publication",
                        plot_name="red_sequence_selection",
                        reason="Missing member scores or red-sequence fit catalog",
                        n_master_rows=n_master_rows,
                        n_member_rows=n_member_rows,
                        n_family_rows=n_family_rows,
                        progress=reporter,
                    )
                if families is not None:
                    _run_catalog_plot(
                        manifest_rows,
                        spec=spec,
                        plot_kind="publication",
                        plot_name="image_family_summary",
                        plot_func=lambda families=families, spec=spec, publication_dir=publication_dir: _plot_publication_family_summary(
                            families,
                            spec,
                            publication_dir,
                        ),
                        n_master_rows=n_master_rows,
                        n_member_rows=n_member_rows,
                        n_family_rows=n_family_rows,
                        progress=reporter,
                    )
                else:
                    _record_skipped_plot(
                        manifest_rows,
                        spec=spec,
                        plot_kind="publication",
                        plot_name="image_family_summary",
                        reason=f"Missing family catalog: {family_path}",
                        n_master_rows=n_master_rows,
                        n_member_rows=n_member_rows,
                        n_family_rows=n_family_rows,
                        progress=reporter,
                    )
                cutout_metadata = {
                    "family_cutout_size_arcsec": float(args.family_cutout_size_arcsec),
                    "family_cutout_circle_radius_arcsec": float(args.family_cutout_circle_radius_arcsec),
                    "family_cutout_color_rms_label": False,
                    "family_cutout_bands": "|".join(str(band) for band in args.family_cutout_bands),
                    "image_scale": args.image_scale,
                }
                if args.image_background == "never":
                    _record_skipped_plot(
                        manifest_rows,
                        spec=spec,
                        plot_kind="publication",
                        plot_name="family_cutouts",
                        reason="image background disabled",
                        n_master_rows=n_master_rows,
                        n_member_rows=n_member_rows,
                        n_family_rows=n_family_rows,
                        progress=reporter,
                        metadata=cutout_metadata,
                    )
                elif families is None or family_members is None or families.empty or family_members.empty:
                    _record_skipped_plot(
                        manifest_rows,
                        spec=spec,
                        plot_kind="publication",
                        plot_name="family_cutouts",
                        reason=f"Missing or empty family-member catalog: {family_member_path}",
                        n_master_rows=n_master_rows,
                        n_member_rows=n_member_rows,
                        n_family_rows=n_family_rows,
                        progress=reporter,
                        metadata=cutout_metadata,
                    )
                else:
                    try:
                        cutout_paths_by_band = _find_family_cutout_band_paths(
                            args.image_dir,
                            spec,
                            tuple(args.family_cutout_bands),
                            image_scale=args.image_scale,
                        )
                        cutout_band_images = _load_family_cutout_band_images(
                            cutout_paths_by_band,
                            tuple(args.family_cutout_bands),
                        )
                    except Exception as exc:
                        if args.image_background == "required":
                            raise MissingCatalogError(f"Could not load RGB FITS for candidate family cutouts: {exc}") from exc
                        _record_skipped_plot(
                            manifest_rows,
                            spec=spec,
                            plot_kind="publication",
                            plot_name="family_cutouts",
                            reason=str(exc),
                            n_master_rows=n_master_rows,
                            n_member_rows=n_member_rows,
                            n_family_rows=n_family_rows,
                            progress=reporter,
                            metadata=cutout_metadata,
                        )
                    else:
                        _run_catalog_plot(
                            manifest_rows,
                            spec=spec,
                            plot_kind="publication",
                            plot_name="family_cutouts",
                            plot_func=lambda families=families, family_members=family_members, pairs=pairs, spec=spec, publication_dir=publication_dir, cutout_band_images=cutout_band_images, cutout_paths_by_band=cutout_paths_by_band: _plot_candidate_family_cutouts(
                                families,
                                family_members,
                                pairs,
                                spec,
                                publication_dir,
                                cutout_band_images,
                                cutout_paths_by_band,
                                bands=tuple(args.family_cutout_bands),
                                cutout_size_arcsec=float(args.family_cutout_size_arcsec),
                                circle_radius_arcsec=float(args.family_cutout_circle_radius_arcsec),
                                image_scale=args.image_scale,
                                families_per_page=int(args.family_cutout_families_per_page),
                            ),
                            n_master_rows=n_master_rows,
                            n_member_rows=n_member_rows,
                            n_family_rows=n_family_rows,
                            progress=reporter,
                            used_background=True,
                            background_path=next(iter(cutout_paths_by_band.values()), None),
                        )

            reporter.finish_step()
            reporter.advance_cluster()

        if publication_enabled:
            reporter.set_cluster_phase("all: plots")
            reporter.start_step("all: writing summary plot", total=1)
            paths = _plot_all_cluster_summary(cluster_counts, args.output_dir)
            manifest_rows.append(
                _plot_manifest_row(
                    spec=None,
                    plot_kind="publication",
                    plot_name="all_cluster_summary",
                    status="generated",
                    paths=paths,
                    n_master_rows=sum(int(row.get("master_rows", 0)) for row in cluster_counts),
                    n_member_rows=sum(int(row.get("selected_members", 0)) for row in cluster_counts),
                    n_family_rows=sum(int(row.get("families", 0)) for row in cluster_counts),
                )
            )
            reporter.advance_step()
            reporter.finish_step()
            reporter.advance_cluster()

    manifest_path = catalog_plot_manifest_path(args.output_dir)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    render_plot_summary(console, manifest_rows, manifest_path)
    return manifest_rows


def run_all_stage(
    args: argparse.Namespace,
    console: Console,
    selected_specs: list[ClusterSpec],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    master_rows, member_rows, family_rows = run_master_stage(
        args,
        console,
        selected_specs,
        build_members=True,
        build_families=True,
    )
    plot_rows = run_plots_stage(args, console, selected_specs)
    return master_rows, member_rows, family_rows, plot_rows


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    console = Console()
    selected_specs = [CLUSTER_BY_KEY[key] for key in args.clusters]
    if args.command == "master":
        run_master_stage(args, console, selected_specs, build_members=False, build_families=False)
    elif args.command == "members":
        run_members_stage(args, console, selected_specs)
    elif args.command == "families":
        run_families_stage(args, console, selected_specs)
    elif args.command == "all":
        run_all_stage(args, console, selected_specs)
    elif args.command == "plots":
        run_plots_stage(args, console, selected_specs)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
