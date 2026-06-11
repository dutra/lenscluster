#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-literature-family-cutouts")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
import astropy.units as u

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_path in (SCRIPT_DIR, REPO_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from analyze_literature_family_diagnostics import (  # noqa: E402
    load_pagul_catalog,
    match_literature_images_to_pagul,
)
from compare_hff_to_literature import LiteratureCatalog, discover_literature_catalogs  # noqa: E402
from lenscluster.rgb import (  # noqa: E402
    DEFAULT_RGB_CHANNEL_GAINS,
    DEFAULT_RGB_MINIMUM,
    DEFAULT_RGB_Q,
    DEFAULT_RGB_STRETCH,
    RGBDisplayConfig,
    build_rgb_display_from_band_images,
    make_natural_rgb,
    trim_to_common_shape,
)


DEFAULT_LITERATURE_ROOT = Path("data") / "literature_lenstool_models"
DEFAULT_IMAGE_DIR = Path("data") / "BUFFALO_Images"
DEFAULT_PAGUL_ROOT = Path("data") / "Pagul2024"
DEFAULT_HFF_CATALOG_ROOT = Path("results") / "hff_master_catalogs"
DEFAULT_OUTPUT_DIR = Path("results") / "literature_family_cutouts"
DEFAULT_CLUSTER = "a370"
DEFAULT_SOURCE_SLUG = "niemiec_buffalo"
DEFAULT_CATALOG_CONTAINS = "sl-final"
PRESET_SINGLE = "single"
PRESET_BERGAMINI = "bergamini"
DEFAULT_BANDS = ("F435W", "F606W", "F814W")
DEFAULT_IMAGE_SCALE = "auto"
IMAGE_SCALE_CHOICES = ("auto", "30mas", "60mas")
DEFAULT_CUTOUT_SIZE_ARCSEC = 10.0
DEFAULT_FAMILIES_PER_PAGE = 6
DEFAULT_MAX_IMAGES_PER_FAMILY = 5
FAMILY_CUTOUT_DETAIL_COLUMNS = 3
MIN_OVERVIEW_SIZE_ARCSEC = 40.0
DEFAULT_PAGUL_MATCH_RADIUS_ARCSEC = 0.5
DEFAULT_PAGUL_MARKER_RADIUS_ARCSEC = 0.2
LITERATURE_MARKER_COLOR = "0.72"
PAGUL_MARKER_COLOR = "#00d5ff"
LITERATURE_MARKER_ALPHA = 0.55
PAGUL_MARKER_ALPHA = 0.85
LITERATURE_MARKER_ZORDER = 5
PAGUL_MARKER_ZORDER = 7
PAGE_FACE_COLOR = "black"
AXIS_FACE_COLOR = "black"
CUTOUT_PANEL_SIZE_INCH = 3.2
CUTOUT_FIGURE_DPI = 300
CUTOUT_LABEL_FONT_SIZE = 8.0
CUTOUT_FAMILY_LABEL_FONT_SIZE = 8.8
CUTOUT_HFF_LABEL_FONT_SIZE = 7.6
CUTOUT_LEGEND_FONT_SIZE = 8.0
CUTOUT_TEXT_BBOX = {"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 0.9}
QUALITY_ORDER = ("Platinum", "Gold", "Silver", "Bronze")
QUALITY_COLORS = {
    "platinum": "#e6ddff",
    "gold": "#ffd84d",
    "silver": "#c8c8c8",
    "bronze": "#cd7f32",
}
UNKNOWN_QUALITY_COLOR = "0.82"
PANEL_SPACING = 0.015
FIGURE_MARGIN = 0.005
SAVEFIG_KWARGS = {"bbox_inches": "tight", "pad_inches": 0.02, "dpi": CUTOUT_FIGURE_DPI}
RGB_RED_GAIN = DEFAULT_RGB_CHANNEL_GAINS["red"]
RGB_GREEN_GAIN = DEFAULT_RGB_CHANNEL_GAINS["green"]
RGB_BLUE_GAIN = DEFAULT_RGB_CHANNEL_GAINS["blue"]
RGB_STRETCH = DEFAULT_RGB_STRETCH
RGB_Q = DEFAULT_RGB_Q
PAGUL_COLOR_BLUE_BAND = "F814W"
PAGUL_COLOR_RED_BAND = "F160W"
PAGUL_SECOND_COLOR_BLUE_BAND = "F606W"
PAGUL_SECOND_COLOR_RED_BAND = "F814W"
PAGUL_RMS_BANDS = (
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
DOWNLOAD_COMMAND_TEMPLATE = (
    "/home/dutra/.conda/envs/lenstronomy/bin/python download_catalogs.py "
    "--catalog buffalo-images --image-scale {image_scale} --output-dir data/BUFFALO_Images"
)
CLUSTER_IMAGE_TOKENS = {
    "a2744": ("a2744", "abell2744"),
    "a370": ("a370", "abell370"),
    "ares": ("ares",),
    "as1063": ("as1063", "abells1063", "rxcj2248"),
    "hera": ("hera",),
    "m0416": ("m0416", "macs0416"),
    "m0717": ("m0717", "macs0717"),
    "m1149": ("m1149", "macs1149"),
}


@dataclass(frozen=True)
class BandImage:
    band: str
    path: Path
    hdu_index: int
    shape: tuple[int, int]
    wcs: WCS
    pixel_scale_arcsec: float


@dataclass(frozen=True)
class CutoutWindow:
    npix: int
    x0: int
    y0: int
    src_x0: int
    src_y0: int
    src_x1: int
    src_y1: int
    dst_x0: int
    dst_y0: int


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "plot"


def default_output_path(cluster: str, source_slug: str, catalog_contains: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{_safe_filename(cluster)}_{_safe_filename(source_slug)}_{_safe_filename(catalog_contains)}_family_cutouts.pdf"


def select_literature_catalog(
    literature_root: Path,
    *,
    cluster: str = DEFAULT_CLUSTER,
    source_slug: str = DEFAULT_SOURCE_SLUG,
    catalog_contains: str = DEFAULT_CATALOG_CONTAINS,
) -> LiteratureCatalog:
    _sources, catalogs = discover_literature_catalogs(Path(literature_root))
    contains = str(catalog_contains).lower()
    candidates = [
        catalog
        for catalog in catalogs
        if catalog.cluster == cluster
        and catalog.source_slug == source_slug
        and catalog.catalog_kind == "image"
        and contains in catalog.path.name.lower()
    ]
    if not candidates:
        available = sorted(
            catalog.path.name
            for catalog in catalogs
            if catalog.cluster == cluster and catalog.source_slug == source_slug and catalog.catalog_kind == "image"
        )
        raise ValueError(
            f"No literature image catalog for cluster={cluster!r}, source_slug={source_slug!r}, "
            f"catalog_contains={catalog_contains!r}. Available image catalogs: {available}"
        )
    return sorted(candidates, key=lambda catalog: str(catalog.path))[0]


def select_bergamini_catalogs(literature_root: Path) -> list[LiteratureCatalog]:
    _sources, catalogs = discover_literature_catalogs(Path(literature_root))
    candidates = [
        catalog
        for catalog in catalogs
        if catalog.catalog_kind == "image" and "bergamini" in catalog.source_slug.lower()
    ]
    if not candidates:
        raise ValueError(f"No Bergamini literature image catalogs found under {literature_root}.")
    return sorted(candidates, key=lambda catalog: (catalog.cluster, catalog.source_slug, catalog.path.name))


def _fits_like(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".fits") or name.endswith(".fit") or name.endswith(".fits.gz")


def _cluster_tokens(cluster: str) -> tuple[str, ...]:
    return CLUSTER_IMAGE_TOKENS.get(str(cluster), (str(cluster),))


def _download_image_scale(image_scale: str) -> str:
    return "60mas" if image_scale == "auto" else str(image_scale)


def _missing_rgb_message(image_dir: Path, cluster: str, missing_bands: Sequence[str], *, image_scale: str = DEFAULT_IMAGE_SCALE) -> str:
    return (
        f"Missing BUFFALO RGB FITS image(s) for cluster {cluster!r} under {image_dir}: "
        f"{', '.join(missing_bands)}. Download them with:\n"
        f"{DOWNLOAD_COMMAND_TEMPLATE.format(image_scale=_download_image_scale(image_scale))}"
    )


def _prefer_image_scale(paths: Sequence[Path], root: Path, image_scale: str) -> list[Path]:
    if image_scale == "auto":
        return list(paths)
    scale_token = str(image_scale).lower()
    preferred = [path for path in paths if scale_token in str(path.relative_to(root)).lower()]
    return preferred or list(paths)


def find_rgb_band_paths(
    image_dir: Path,
    *,
    cluster: str = DEFAULT_CLUSTER,
    bands: Sequence[str] = DEFAULT_BANDS,
    image_scale: str = DEFAULT_IMAGE_SCALE,
) -> dict[str, Path]:
    if image_scale not in IMAGE_SCALE_CHOICES:
        raise ValueError(f"Unsupported image scale {image_scale!r}. Valid choices: {IMAGE_SCALE_CHOICES}.")
    root = Path(image_dir)
    tokens = tuple(token.lower() for token in _cluster_tokens(cluster))
    root_has_cluster_token = any(token in str(root).lower() for token in tokens)
    result: dict[str, Path] = {}
    missing: list[str] = []

    for band in bands:
        band_token = str(band).lower()
        candidates: list[Path] = []
        if root.exists():
            for path in root.rglob("*"):
                if not path.is_file() or not _fits_like(path):
                    continue
                haystack = str(path.relative_to(root)).lower()
                if band_token in haystack and (root_has_cluster_token or any(token in haystack for token in tokens)):
                    candidates.append(path)
        if not candidates:
            missing.append(str(band))
            continue
        candidates = _prefer_image_scale(candidates, root, image_scale)
        result[str(band)] = sorted(candidates, key=lambda path: (0 if "_drz" in path.name.lower() else 1, len(path.parts), str(path)))[0]

    if missing:
        raise FileNotFoundError(_missing_rgb_message(root, cluster, missing, image_scale=image_scale))
    return result


def load_band_metadata(band: str, path: Path) -> BandImage:
    with fits.open(path, memmap=True) as hdul:
        for hdu_index, hdu in enumerate(hdul):
            if hdu.data is None:
                continue
            if getattr(hdu.data, "ndim", 0) != 2:
                continue
            wcs = WCS(hdu.header)
            if not getattr(wcs, "has_celestial", False):
                continue
            celestial = wcs.celestial
            scales = np.asarray(proj_plane_pixel_scales(celestial), dtype=float) * 3600.0
            pixel_scale = float(np.nanmean(np.abs(scales)))
            if not np.isfinite(pixel_scale) or pixel_scale <= 0.0:
                continue
            ny, nx = hdu.data.shape
            return BandImage(
                band=str(band),
                path=Path(path),
                hdu_index=int(hdu_index),
                shape=(int(ny), int(nx)),
                wcs=celestial,
                pixel_scale_arcsec=pixel_scale,
            )
    raise ValueError(f"No 2D celestial WCS image found in {path}")


def load_rgb_metadata(paths_by_band: dict[str, Path], bands: Sequence[str] = DEFAULT_BANDS) -> dict[str, BandImage]:
    return {str(band): load_band_metadata(str(band), paths_by_band[str(band)]) for band in bands}


def _cutout_npixels(image: BandImage, *, cutout_size_arcsec: float) -> int:
    if cutout_size_arcsec <= 0.0:
        raise ValueError("cutout_size_arcsec must be positive.")
    return max(1, int(math.ceil(float(cutout_size_arcsec) / image.pixel_scale_arcsec)))


def _clamped_cutout_origin(raw_origin: int, npix: int, image_size: int) -> int:
    if image_size >= npix:
        return int(np.clip(raw_origin, 0, image_size - npix))
    return int(np.clip(raw_origin, image_size - npix, 0))


def _cutout_window(image: BandImage, coord: SkyCoord, *, cutout_size_arcsec: float) -> CutoutWindow | None:
    npix = _cutout_npixels(image, cutout_size_arcsec=cutout_size_arcsec)
    center_x, center_y = image.wcs.world_to_pixel(coord)
    if not all(np.isfinite(value) for value in (center_x, center_y)):
        return None
    x_center = int(round(float(center_x)))
    y_center = int(round(float(center_y)))
    half = npix // 2
    raw_x0 = x_center - half
    raw_y0 = y_center - half
    raw_x1 = raw_x0 + npix
    raw_y1 = raw_y0 + npix
    ny, nx = image.shape
    if raw_x0 >= nx or raw_x1 <= 0 or raw_y0 >= ny or raw_y1 <= 0:
        return None

    x0 = _clamped_cutout_origin(raw_x0, npix, nx)
    y0 = _clamped_cutout_origin(raw_y0, npix, ny)
    x1 = x0 + npix
    y1 = y0 + npix
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(nx, x1)
    src_y1 = min(ny, y1)
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return None
    return CutoutWindow(
        npix=npix,
        x0=x0,
        y0=y0,
        src_x0=src_x0,
        src_y0=src_y0,
        src_x1=src_x1,
        src_y1=src_y1,
        dst_x0=src_x0 - x0,
        dst_y0=src_y0 - y0,
    )


def extract_band_cutout(image: BandImage, coord: SkyCoord, *, cutout_size_arcsec: float) -> np.ndarray:
    npix = _cutout_npixels(image, cutout_size_arcsec=cutout_size_arcsec)
    cutout = np.full((npix, npix), np.nan, dtype=np.float32)
    window = _cutout_window(image, coord, cutout_size_arcsec=cutout_size_arcsec)
    if window is None:
        return cutout

    with fits.open(image.path, memmap=True) as hdul:
        data = hdul[image.hdu_index].section[window.src_y0 : window.src_y1, window.src_x0 : window.src_x1]
        values = np.asarray(data, dtype=np.float32)

    cutout[
        window.dst_y0 : window.dst_y0 + values.shape[0],
        window.dst_x0 : window.dst_x0 + values.shape[1],
    ] = values
    return cutout


def _trim_to_common_shape(arrays: Iterable[np.ndarray]) -> list[np.ndarray]:
    return trim_to_common_shape([np.asarray(array) for array in arrays])


def build_rgb_display(
    band_images: Mapping[str, BandImage],
    *,
    bands: Sequence[str] = DEFAULT_BANDS,
    q: float = DEFAULT_RGB_Q,
    stretch: float = DEFAULT_RGB_STRETCH,
    channel_gains: Mapping[str, float] = DEFAULT_RGB_CHANNEL_GAINS,
    minimum: float = DEFAULT_RGB_MINIMUM,
) -> RGBDisplayConfig:
    return build_rgb_display_from_band_images(
        band_images,
        bands=bands,
        q=q,
        stretch=stretch,
        channel_gains=channel_gains,
        minimum=minimum,
    )


def make_rgb_cutout(
    cutouts_by_band: dict[str, np.ndarray],
    bands: Sequence[str] = DEFAULT_BANDS,
    *,
    rgb_display: RGBDisplayConfig | None = None,
) -> np.ndarray:
    return make_natural_rgb(cutouts_by_band, bands=bands, display=rgb_display)


def _draw_rgb_cutout(
    ax: plt.Axes,
    band_images: dict[str, BandImage],
    bands: Sequence[str],
    rgb_display: RGBDisplayConfig,
    center_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
) -> np.ndarray:
    cutouts = {
        str(band): extract_band_cutout(band_images[str(band)], center_coord, cutout_size_arcsec=cutout_size_arcsec)
        for band in bands
    }
    rgb = make_rgb_cutout(cutouts, bands=bands, rgb_display=rgb_display)
    height, width = rgb.shape[:2]
    ax.imshow(
        rgb,
        origin="lower",
        interpolation="bilinear",
        extent=(-0.5, width - 0.5, -0.5, height - 0.5),
    )
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_autoscale_on(False)
    return rgb


def _quality_key(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    lower = text.lower()
    for quality in QUALITY_ORDER:
        if lower == quality.lower():
            return quality
    return text


def _quality_rank(value: object) -> int:
    key = _quality_key(value).lower()
    ranks = {quality.lower(): index for index, quality in enumerate(QUALITY_ORDER)}
    return ranks.get(key, len(QUALITY_ORDER))


def _quality_color(value: object) -> str:
    return QUALITY_COLORS.get(_quality_key(value).lower(), UNKNOWN_QUALITY_COLOR)


def _quality_text(value: object) -> str:
    key = _quality_key(value)
    return key if key.lower() in QUALITY_COLORS else ""


def _quality_text_bbox(value: object) -> dict[str, object]:
    return {**CUTOUT_TEXT_BBOX, "edgecolor": _quality_color(value), "linewidth": 0.75}


def _family_ids_in_catalog_order(data: pd.DataFrame) -> list[str]:
    family_ids = list(dict.fromkeys(data["family_id"].astype(str)))
    if "catalog_quality" not in data.columns:
        return family_ids
    family_rank: dict[str, int] = {}
    for family_id, family in data.groupby(data["family_id"].astype(str), sort=False):
        ranks = [_quality_rank(value) for value in family["catalog_quality"]]
        family_rank[str(family_id)] = min(ranks) if ranks else len(QUALITY_ORDER)
    original_order = {family_id: index for index, family_id in enumerate(family_ids)}
    return sorted(family_ids, key=lambda family_id: (family_rank.get(str(family_id), len(QUALITY_ORDER)), original_order[family_id]))


def _format_redshift(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "z=na"
    if not np.isfinite(numeric):
        return "z=na"
    return f"z={numeric:.4g}"


def _row_coord(row: pd.Series) -> SkyCoord:
    return SkyCoord(ra=float(row["ra"]) * u.deg, dec=float(row["dec"]) * u.deg, frame="icrs")


def _safe_row_coord(row: pd.Series) -> SkyCoord | None:
    try:
        coord = _row_coord(row)
    except (KeyError, TypeError, ValueError):
        return None
    pixel_ra = float(coord.ra.deg)
    pixel_dec = float(coord.dec.deg)
    if not np.isfinite(pixel_ra) or not np.isfinite(pixel_dec):
        return None
    return coord


def _pixel_to_skycoord(image: BandImage, x_pixel: float, y_pixel: float) -> SkyCoord | None:
    try:
        coord = image.wcs.pixel_to_world(float(x_pixel), float(y_pixel))
    except Exception:
        return None
    try:
        skycoord = coord if isinstance(coord, SkyCoord) else SkyCoord(coord)
    except Exception:
        return None
    if not np.isfinite(float(skycoord.ra.deg)) or not np.isfinite(float(skycoord.dec.deg)):
        return None
    return skycoord


def _coords_from_rows(rows: pd.DataFrame, *, include_pagul: bool = False) -> list[SkyCoord]:
    coords: list[SkyCoord] = []
    for _, row in rows.iterrows():
        coord = _safe_row_coord(row)
        if coord is not None:
            coords.append(coord)
        if include_pagul:
            pagul_coord = _pagul_coord(row)
            if pagul_coord is not None:
                coords.append(pagul_coord)
    return coords


def _overview_geometry_for_coords(
    image: BandImage,
    coords: Sequence[SkyCoord],
    *,
    minimum_side_arcsec: float = MIN_OVERVIEW_SIZE_ARCSEC,
) -> tuple[SkyCoord, float]:
    pixels: list[tuple[float, float]] = []
    for coord in coords:
        try:
            x_pixel, y_pixel = image.wcs.world_to_pixel(coord)
        except Exception:
            continue
        if np.isfinite(x_pixel) and np.isfinite(y_pixel):
            pixels.append((float(x_pixel), float(y_pixel)))
    if pixels:
        pixel_array = np.asarray(pixels, dtype=float)
        x_min, y_min = np.min(pixel_array, axis=0)
        x_max, y_max = np.max(pixel_array, axis=0)
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        span_arcsec = float(max(x_max - x_min, y_max - y_min) * image.pixel_scale_arcsec)
    else:
        ny, nx = image.shape
        x_center = 0.5 * float(nx - 1)
        y_center = 0.5 * float(ny - 1)
        span_arcsec = 0.0
    center_coord = _pixel_to_skycoord(image, x_center, y_center)
    if center_coord is None and coords:
        center_coord = coords[0]
    if center_coord is None:
        center_coord = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, frame="icrs")
    padding = max(2.0, 0.15 * span_arcsec)
    return center_coord, max(float(minimum_side_arcsec), span_arcsec + 2.0 * padding)


def _ab_magnitude(flux_values: object) -> np.ndarray:
    flux = pd.to_numeric(pd.Series(flux_values), errors="coerce").to_numpy(dtype=float)
    magnitude = np.full(flux.shape, np.nan, dtype=float)
    positive = np.isfinite(flux) & (flux > 0.0)
    magnitude[positive] = -2.5 * np.log10(flux[positive]) - 48.6
    return magnitude


def _decode_strings(values: object) -> pd.Series:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="ignore").strip())
        else:
            decoded.append(str(value).strip())
    return pd.Series(decoded, dtype=object)


def _pagul_color_column_name() -> str:
    return f"pagul_color_{PAGUL_COLOR_BLUE_BAND.lower()}_{PAGUL_COLOR_RED_BAND.lower()}"


def _pagul_color_label() -> str:
    return f"{PAGUL_COLOR_BLUE_BAND}-{PAGUL_COLOR_RED_BAND}"


def _pagul_second_color_column_name() -> str:
    return f"pagul_color_{PAGUL_SECOND_COLOR_BLUE_BAND.lower()}_{PAGUL_SECOND_COLOR_RED_BAND.lower()}"


def _pagul_second_color_label() -> str:
    return f"{PAGUL_SECOND_COLOR_BLUE_BAND}-{PAGUL_SECOND_COLOR_RED_BAND}"


def _pagul_mag_column(band: str) -> str:
    return f"pagul_mag_{str(band).lower()}"


def _pagul_source_mag_column(band: str) -> str:
    return f"mag_{str(band).lower()}"


def _add_pagul_color_columns(pagul: pd.DataFrame) -> pd.DataFrame:
    pagul = pagul.copy()
    for band in PAGUL_RMS_BANDS:
        pagul[_pagul_source_mag_column(band)] = np.nan
    pagul["color_f814w_f160w"] = np.nan
    pagul["color_f606w_f814w"] = np.nan
    if pagul.empty or "catalog_path" not in pagul.columns:
        return pagul

    catalog_paths = [path for path in pagul["catalog_path"].dropna().astype(str).unique() if path]
    if not catalog_paths:
        return pagul
    table = Table.read(catalog_paths[0])
    if "ID" not in table.colnames:
        return pagul

    object_ids = _decode_strings(table["ID"])
    by_id_columns: dict[str, object] = {"object_id": object_ids}
    for band in PAGUL_RMS_BANDS:
        flux_column = f"FLUX_{band}"
        if flux_column in table.colnames:
            by_id_columns[_pagul_source_mag_column(band)] = _ab_magnitude(table[flux_column])
    by_id = pd.DataFrame(by_id_columns).drop_duplicates("object_id")
    by_id_indexed = by_id.set_index("object_id")
    for band in PAGUL_RMS_BANDS:
        source_column = _pagul_source_mag_column(band)
        if source_column in by_id_indexed.columns:
            pagul[source_column] = pagul["object_id"].astype(str).map(by_id_indexed[source_column])

    blue_column = _pagul_source_mag_column(PAGUL_COLOR_BLUE_BAND)
    red_column = _pagul_source_mag_column(PAGUL_COLOR_RED_BAND)
    if blue_column in by_id_indexed.columns and red_column in by_id_indexed.columns:
        color_by_id = by_id_indexed[blue_column] - by_id_indexed[red_column]
        pagul["color_f814w_f160w"] = pagul["object_id"].astype(str).map(color_by_id)
    second_blue_column = _pagul_source_mag_column(PAGUL_SECOND_COLOR_BLUE_BAND)
    second_red_column = _pagul_source_mag_column(PAGUL_SECOND_COLOR_RED_BAND)
    if second_blue_column in by_id_indexed.columns and second_red_column in by_id_indexed.columns:
        second_color_by_id = by_id_indexed[second_blue_column] - by_id_indexed[second_red_column]
        pagul["color_f606w_f814w"] = pagul["object_id"].astype(str).map(second_color_by_id)
    return pagul


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _catalog_with_pagul_match_info(
    catalog: LiteratureCatalog,
    *,
    pagul_root: Path = DEFAULT_PAGUL_ROOT,
    match_radius_arcsec: float = DEFAULT_PAGUL_MATCH_RADIUS_ARCSEC,
) -> LiteratureCatalog:
    data = catalog.data.reset_index(drop=True).copy()
    pagul = _add_pagul_color_columns(load_pagul_catalog(Path(pagul_root), catalog.cluster))
    data["pagul_catalog_present"] = not pagul.empty

    if pagul.empty:
        data["pagul_matched"] = False
        data["pagul_match_separation_arcsec"] = np.nan
        data["pagul_object_id"] = ""
        data["pagul_zspec"] = np.nan
        data["pagul_zphot"] = np.nan
        data[_pagul_color_column_name()] = np.nan
        data[_pagul_second_color_column_name()] = np.nan
        data["pagul_color_rms"] = np.nan
        data["pagul_family_color_rms"] = np.nan
    else:
        matches = match_literature_images_to_pagul(data, pagul, radius_arcsec=match_radius_arcsec)
        pagul_by_id = pagul.set_index(pagul["object_id"].astype(str))
        object_id_to_color = pagul_by_id["color_f814w_f160w"]
        matches[_pagul_color_column_name()] = matches["pagul_object_id"].astype(str).map(object_id_to_color)
        object_id_to_second_color = pagul_by_id["color_f606w_f814w"]
        matches[_pagul_second_color_column_name()] = matches["pagul_object_id"].astype(str).map(object_id_to_second_color)
        for band in PAGUL_RMS_BANDS:
            source_column = _pagul_source_mag_column(band)
            if source_column in pagul_by_id.columns:
                matches[_pagul_mag_column(band)] = matches["pagul_object_id"].astype(str).map(pagul_by_id[source_column])
        matches = matches.set_index("literature_row_index")
        for column in matches.columns:
            data[column] = matches[column].reindex(data.index).to_numpy()
        data = _add_color_rms_columns(data)

    return LiteratureCatalog(
        cluster=catalog.cluster,
        source_slug=catalog.source_slug,
        source_id=catalog.source_id,
        catalog_kind=catalog.catalog_kind,
        path=catalog.path,
        data=data,
        status=catalog.status,
        note=catalog.note,
    )


def _hff_catalog_paths(hff_root: Path, cluster: str) -> tuple[Path, Path, Path]:
    cluster_dir = Path(hff_root) / str(cluster)
    return (
        cluster_dir / f"{cluster}_candidate_image_families.csv",
        cluster_dir / f"{cluster}_candidate_family_members.csv",
        cluster_dir / f"{cluster}_candidate_family_pairs.csv",
    )


def _read_hff_catalogs(hff_root: Path, cluster: str) -> tuple[bool, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_path, member_path, pair_path = _hff_catalog_paths(Path(hff_root), str(cluster))
    if not (family_path.exists() and member_path.exists() and pair_path.exists()):
        return False, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return (
        True,
        pd.read_csv(family_path, low_memory=False),
        pd.read_csv(member_path, low_memory=False),
        pd.read_csv(pair_path, low_memory=False),
    )


def _hff_object_id_from_pagul_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text if text.startswith("pagul2024:") else f"pagul2024:{text}"


def _pair_key(left: object, right: object) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def _compact_reason(value: object, *, max_chars: int = 30) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text if len(text) <= max_chars else f"{text[: max_chars - 1]}..."


def _catalog_with_hff_diagnostics(
    catalog: LiteratureCatalog,
    *,
    hff_root: Path = DEFAULT_HFF_CATALOG_ROOT,
) -> LiteratureCatalog:
    data = catalog.data.reset_index(drop=True).copy()
    catalog_present, families, members, pairs = _read_hff_catalogs(Path(hff_root), catalog.cluster)
    data["hff_catalog_present"] = bool(catalog_present)
    data["hff_object_id"] = data.get("pagul_object_id", pd.Series("", index=data.index)).map(_hff_object_id_from_pagul_id)
    hff_columns: dict[str, Any] = {
        "hff_candidate_family_id": "",
        "hff_membership_probability": np.nan,
        "hff_family_probability": np.nan,
        "hff_min_pair_score": np.nan,
        "hff_median_sed_rms": np.nan,
        "hff_review_flags": "",
        "hff_family_n_images": np.nan,
        "hff_mean_pair_score_to_family": np.nan,
        "hff_best_pair_score": np.nan,
        "hff_pair_reject_reason": "",
    }
    for column, default in hff_columns.items():
        data[column] = default
    if not catalog_present:
        return LiteratureCatalog(
            cluster=catalog.cluster,
            source_slug=catalog.source_slug,
            source_id=catalog.source_id,
            catalog_kind=catalog.catalog_kind,
            path=catalog.path,
            data=data,
            status=catalog.status,
            note=catalog.note,
        )

    if not members.empty and "object_id" in members.columns:
        members = members.copy()
        members["object_id"] = members["object_id"].astype(str)
        members["_membership_sort"] = pd.to_numeric(members.get("membership_probability", np.nan), errors="coerce").fillna(-np.inf)
        best_members = members.sort_values("_membership_sort", ascending=False).drop_duplicates("object_id")
        family_columns = [
            column
            for column in (
                "candidate_family_id",
                "family_probability",
                "min_pair_score",
                "median_sed_rms",
                "review_flags",
                "n_images",
            )
            if column in families.columns
        ]
        if family_columns and "candidate_family_id" in family_columns:
            best_members = best_members.merge(
                families[family_columns],
                on="candidate_family_id",
                how="left",
                suffixes=("", "_family"),
            )
        by_object = best_members.set_index("object_id", drop=False)
        for index, hff_object_id in data["hff_object_id"].items():
            if not hff_object_id or hff_object_id not in by_object.index:
                continue
            row = by_object.loc[hff_object_id]
            data.at[index, "hff_candidate_family_id"] = str(row.get("candidate_family_id", ""))
            data.at[index, "hff_membership_probability"] = _safe_float(row.get("membership_probability"))
            data.at[index, "hff_family_probability"] = _safe_float(row.get("family_probability"))
            data.at[index, "hff_min_pair_score"] = _safe_float(row.get("min_pair_score"))
            data.at[index, "hff_median_sed_rms"] = _safe_float(row.get("median_sed_rms"))
            data.at[index, "hff_review_flags"] = _compact_reason(row.get("review_flags", ""))
            data.at[index, "hff_family_n_images"] = _safe_float(row.get("n_images"))
            data.at[index, "hff_mean_pair_score_to_family"] = _safe_float(row.get("mean_pair_score_to_family"))

    pair_lookup: dict[tuple[str, str], list[pd.Series]] = {}
    if not pairs.empty and {"left_object_id", "right_object_id"}.issubset(pairs.columns):
        for _, pair_row in pairs.iterrows():
            pair_lookup.setdefault(_pair_key(pair_row["left_object_id"], pair_row["right_object_id"]), []).append(pair_row)
    if pair_lookup:
        for _family_id, family_rows in data.groupby(data["family_id"].astype(str), sort=False):
            family_object_ids = [object_id for object_id in family_rows["hff_object_id"].astype(str).tolist() if object_id]
            for index, hff_object_id in family_rows["hff_object_id"].astype(str).items():
                if not hff_object_id:
                    continue
                candidate_pairs: list[pd.Series] = []
                for other_object_id in family_object_ids:
                    if other_object_id == hff_object_id:
                        continue
                    candidate_pairs.extend(pair_lookup.get(_pair_key(hff_object_id, other_object_id), []))
                if not candidate_pairs:
                    continue
                scores = np.asarray([_safe_float(pair.get("pair_score")) for pair in candidate_pairs], dtype=float)
                finite_scores = scores[np.isfinite(scores)]
                if finite_scores.size:
                    data.at[index, "hff_best_pair_score"] = float(np.nanmax(finite_scores))
                reasons = [
                    _compact_reason(pair.get("hard_reject_reason", ""))
                    for pair in candidate_pairs
                    if _compact_reason(pair.get("hard_reject_reason", ""))
                ]
                if reasons:
                    data.at[index, "hff_pair_reject_reason"] = reasons[0]

    return LiteratureCatalog(
        cluster=catalog.cluster,
        source_slug=catalog.source_slug,
        source_id=catalog.source_id,
        catalog_kind=catalog.catalog_kind,
        path=catalog.path,
        data=data,
        status=catalog.status,
        note=catalog.note,
    )


def _pair_color_rms(left: pd.Series, right: pd.Series) -> float:
    left_values = np.asarray([_safe_float(left.get(_pagul_mag_column(band))) for band in PAGUL_RMS_BANDS], dtype=float)
    right_values = np.asarray([_safe_float(right.get(_pagul_mag_column(band))) for band in PAGUL_RMS_BANDS], dtype=float)
    common = np.isfinite(left_values) & np.isfinite(right_values)
    if not common.any():
        return float("nan")
    diff = left_values[common] - right_values[common]
    offset = float(np.median(diff))
    residuals = diff - offset
    return float(np.sqrt(np.mean(residuals**2)))


def _add_color_rms_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["pagul_color_rms"] = np.nan
    data["pagul_family_color_rms"] = np.nan
    if data.empty or "family_id" not in data.columns or "pagul_matched" not in data.columns:
        return data

    for _family_id, family in data.groupby(data["family_id"].astype(str), sort=False):
        matched_indices = [int(index) for index, row in family.iterrows() if _bool_value(row.get("pagul_matched"))]
        if len(matched_indices) < 2:
            continue
        image_values: dict[int, list[float]] = {index: [] for index in matched_indices}
        family_values: list[float] = []
        for left_index, right_index in itertools.combinations(matched_indices, 2):
            rms = _pair_color_rms(data.loc[left_index], data.loc[right_index])
            if not np.isfinite(rms):
                continue
            image_values[left_index].append(rms)
            image_values[right_index].append(rms)
            family_values.append(rms)

        if family_values:
            data.loc[family.index, "pagul_family_color_rms"] = float(np.median(np.asarray(family_values, dtype=float)))
        for index, values in image_values.items():
            if values:
                data.loc[index, "pagul_color_rms"] = float(np.median(np.asarray(values, dtype=float)))
    return data


def _safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def _format_number(value: object, *, precision: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "na"
    if not np.isfinite(numeric):
        return "na"
    return f"{numeric:.{precision}g}"


def _format_pagul_match(row: pd.Series) -> str:
    if "pagul_catalog_present" not in row:
        return ""
    if not _bool_value(row.get("pagul_catalog_present")):
        return "Pagul: no catalog"
    if not _bool_value(row.get("pagul_matched")):
        return "Pagul: no"

    object_id = str(row.get("pagul_object_id", "")).strip()
    separation = row.get("pagul_match_separation_arcsec")
    try:
        separation_value = float(separation)
    except (TypeError, ValueError):
        separation_value = float("nan")
    object_text = f" {object_id}" if object_id else ""
    separation_text = f' ({separation_value:.3f}")' if np.isfinite(separation_value) else ""
    color = _format_number(row.get(_pagul_color_column_name()), precision=3)
    second_color = _format_number(row.get(_pagul_second_color_column_name()), precision=3)
    specz = _format_number(row.get("pagul_zspec"), precision=4)
    photoz = _format_number(row.get("pagul_zphot"), precision=4)
    return (
        f"Pagul: yes{object_text}{separation_text}\n"
        f"{_pagul_color_label()}={color}\n"
        f"{_pagul_second_color_label()}={second_color}\n"
        f"specz={specz}  photoz={photoz}"
    )


def _format_cutout_crms(value: object) -> str:
    numeric = _safe_float(value)
    if not np.isfinite(numeric):
        return "crms=na"
    return f"crms={numeric:.2f}"


def _format_family_label(family_id: str, family: pd.DataFrame, max_images_per_family: int) -> str:
    family_label = f"Family {family_id} ({len(family)} images)"
    if "pagul_family_color_rms" in family.columns:
        family_label += f"\n{_format_cutout_crms(family['pagul_family_color_rms'].iloc[0])}"
    return family_label


def _format_hff_diagnostics(row: pd.Series) -> str:
    if "hff_catalog_present" not in row:
        return ""
    if not _bool_value(row.get("hff_catalog_present")):
        return "HFF: no catalog"
    if not str(row.get("hff_object_id", "")).strip():
        return "HFF: no Pagul"

    family_id = str(row.get("hff_candidate_family_id", "")).strip()
    lines: list[str] = []
    if family_id:
        family_probability = _format_number(row.get("hff_family_probability"), precision=2)
        membership = _format_number(row.get("hff_membership_probability"), precision=2)
        lines.append(f"HFF: {family_id} P={family_probability} mem={membership}")
        min_pair = _format_number(row.get("hff_min_pair_score"), precision=2)
        hff_crms = _format_number(row.get("hff_median_sed_rms"), precision=2)
        lines.append(f"minpair={min_pair} hffcrms={hff_crms}")
    else:
        lines.append("HFF: no family")

    best_pair = _safe_float(row.get("hff_best_pair_score"))
    reason = _compact_reason(row.get("hff_pair_reject_reason", ""))
    pair_parts: list[str] = []
    if np.isfinite(best_pair):
        pair_parts.append(f"bestpair={best_pair:.2f}")
    if reason:
        pair_parts.append(f"reject={reason}")
    if pair_parts:
        lines.append(" ".join(pair_parts))

    flags = _compact_reason(row.get("hff_review_flags", ""))
    if flags:
        lines.append(f"flags={flags}")
    return "\n".join(lines)


def _pagul_coord(row: pd.Series) -> SkyCoord | None:
    if not _bool_value(row.get("pagul_matched")):
        return None
    try:
        ra = float(row.get("pagul_ra"))
        dec = float(row.get("pagul_dec"))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(ra) or not np.isfinite(dec):
        return None
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")


def _cutout_pixel_position(
    image: BandImage,
    center_coord: SkyCoord,
    target_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
) -> tuple[float, float, float] | None:
    if cutout_size_arcsec <= 0.0:
        return None
    window = _cutout_window(image, center_coord, cutout_size_arcsec=cutout_size_arcsec)
    if window is None:
        return None
    target_x, target_y = image.wcs.world_to_pixel(target_coord)
    if not all(np.isfinite(value) for value in (target_x, target_y)):
        return None
    marker_radius_pix = DEFAULT_PAGUL_MARKER_RADIUS_ARCSEC / image.pixel_scale_arcsec
    return float(target_x) - float(window.x0), float(target_y) - float(window.y0), marker_radius_pix


def _draw_marker(
    ax: plt.Axes,
    image: BandImage,
    center_coord: SkyCoord,
    target_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    edgecolor: str,
    alpha: float = 0.70,
    linestyle: str = "-",
    linewidth: float = 1.1,
    zorder: float = 5,
) -> None:
    marker = _cutout_pixel_position(
        image,
        center_coord,
        target_coord,
        cutout_size_arcsec=cutout_size_arcsec,
    )
    if marker is None:
        return
    x, y, radius = marker
    height, width = rendered_shape
    if x < -radius or x > width - 1 + radius or y < -radius or y > height - 1 + radius:
        return
    ax.add_patch(
        Circle(
            (x, y),
            radius=radius,
            edgecolor=edgecolor,
            facecolor="none",
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )
    )


def _draw_pagul_marker(
    ax: plt.Axes,
    image: BandImage,
    image_row: pd.Series,
    center_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
) -> None:
    pagul_coord = _pagul_coord(image_row)
    if pagul_coord is None:
        return
    _draw_marker(
        ax,
        image,
        center_coord,
        pagul_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        edgecolor=PAGUL_MARKER_COLOR,
        alpha=PAGUL_MARKER_ALPHA,
        linestyle="--",
        linewidth=1.15,
        zorder=PAGUL_MARKER_ZORDER,
    )


def _draw_literature_marker_at(
    ax: plt.Axes,
    image: BandImage,
    center_coord: SkyCoord,
    target_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
) -> None:
    _draw_marker(
        ax,
        image,
        center_coord,
        target_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        edgecolor=LITERATURE_MARKER_COLOR,
        alpha=LITERATURE_MARKER_ALPHA,
        linestyle="-",
        linewidth=1.1,
        zorder=LITERATURE_MARKER_ZORDER,
    )


def _draw_literature_marker(
    ax: plt.Axes,
    image: BandImage,
    center_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
) -> None:
    _draw_literature_marker_at(
        ax,
        image,
        center_coord,
        center_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
    )


def _draw_marker_legend(ax: plt.Axes, *, include_pagul: bool = True) -> None:
    legend_bbox = {"facecolor": "black", "alpha": 0.38, "edgecolor": "none", "pad": 0.45}
    entries = [
        ("lit", LITERATURE_MARKER_COLOR, "-", LITERATURE_MARKER_ALPHA, LITERATURE_MARKER_ZORDER),
    ]
    if include_pagul:
        entries.append(("Pagul", PAGUL_MARKER_COLOR, "--", PAGUL_MARKER_ALPHA, PAGUL_MARKER_ZORDER))
    y_values = [0.105, 0.055] if include_pagul else [0.065]
    ax.text(
        0.782,
        0.14 if include_pagul else 0.095,
        " \n " if include_pagul else " ",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=CUTOUT_LEGEND_FONT_SIZE,
        color="white",
        linespacing=0.85,
        clip_on=True,
        bbox=legend_bbox,
    )
    for (label, color, linestyle, alpha, zorder), y in zip(entries, y_values):
        ax.add_patch(
            Circle(
                (0.812, y),
                radius=0.014,
                transform=ax.transAxes,
                edgecolor=color,
                facecolor="none",
                linewidth=1.0,
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
                clip_on=True,
            )
        )
        ax.text(
            0.845,
            y,
            label,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=CUTOUT_LEGEND_FONT_SIZE,
            color="white",
            clip_on=True,
            zorder=zorder,
        )


def _figure_size(n_rows: int, max_images_per_family: int) -> tuple[float, float]:
    return CUTOUT_PANEL_SIZE_INCH * float(max_images_per_family), CUTOUT_PANEL_SIZE_INCH * float(n_rows)


def _style_cutout_figure(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(PAGE_FACE_COLOR)
    fig.subplots_adjust(
        left=FIGURE_MARGIN,
        right=1.0 - FIGURE_MARGIN,
        bottom=FIGURE_MARGIN,
        top=1.0 - FIGURE_MARGIN,
        wspace=PANEL_SPACING,
        hspace=PANEL_SPACING,
    )


def _style_cutout_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(AXIS_FACE_COLOR)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.08")
        spine.set_linewidth(0.35)


def _draw_literature_cluster_overview_panel(
    ax: plt.Axes,
    band_images: dict[str, BandImage],
    bands: Sequence[str],
    rgb_display: RGBDisplayConfig,
    data: pd.DataFrame,
) -> None:
    display_image = band_images[str(bands[-1])]
    center_coord, cutout_size_arcsec = _overview_geometry_for_coords(
        display_image,
        _coords_from_rows(data, include_pagul=False),
    )
    rgb = _draw_rgb_cutout(
        ax,
        band_images,
        bands,
        rgb_display,
        center_coord,
        cutout_size_arcsec=cutout_size_arcsec,
    )
    for _, image_row in data.iterrows():
        coord = _safe_row_coord(image_row)
        if coord is None:
            continue
        _draw_literature_marker_at(
            ax,
            display_image,
            center_coord,
            coord,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
        )
    _draw_marker_legend(ax, include_pagul=False)


def _draw_literature_family_overview_panel(
    ax: plt.Axes,
    band_images: dict[str, BandImage],
    bands: Sequence[str],
    rgb_display: RGBDisplayConfig,
    family_id: str,
    family: pd.DataFrame,
    *,
    max_images_per_family: int,
) -> None:
    display_image = band_images[str(bands[-1])]
    center_coord, cutout_size_arcsec = _overview_geometry_for_coords(
        display_image,
        _coords_from_rows(family, include_pagul=True),
    )
    rgb = _draw_rgb_cutout(
        ax,
        band_images,
        bands,
        rgb_display,
        center_coord,
        cutout_size_arcsec=cutout_size_arcsec,
    )
    for _, image_row in family.iterrows():
        coord = _safe_row_coord(image_row)
        if coord is None:
            continue
        _draw_literature_marker_at(
            ax,
            display_image,
            center_coord,
            coord,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
        )
        _draw_pagul_marker(
            ax,
            display_image,
            image_row,
            center_coord,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
        )
    ax.text(
        0.035,
        0.965,
        _format_family_label(family_id, family, max_images_per_family),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=CUTOUT_FAMILY_LABEL_FONT_SIZE,
        color="white",
        linespacing=0.95,
        clip_on=True,
        bbox=CUTOUT_TEXT_BBOX,
    )
    _draw_marker_legend(ax, include_pagul=True)


def _draw_literature_detail_panel(
    ax: plt.Axes,
    band_images: dict[str, BandImage],
    bands: Sequence[str],
    rgb_display: RGBDisplayConfig,
    image_row: pd.Series,
    *,
    cutout_size_arcsec: float,
) -> None:
    coord = _safe_row_coord(image_row)
    if coord is None:
        ax.set_axis_off()
        return
    display_image = band_images[str(bands[-1])]
    rgb = _draw_rgb_cutout(
        ax,
        band_images,
        bands,
        rgb_display,
        coord,
        cutout_size_arcsec=cutout_size_arcsec,
    )
    _draw_literature_marker(
        ax,
        display_image,
        coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rgb.shape[:2],
    )
    _draw_pagul_marker(
        ax,
        display_image,
        image_row,
        coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rgb.shape[:2],
    )
    quality = _quality_text(image_row.get("catalog_quality", ""))
    quality_suffix = f" {quality}" if quality else ""
    label = f"{image_row.get('literature_id', '')}  {_format_redshift(image_row.get('catalog_z'))}{quality_suffix}"
    pagul_label = _format_pagul_match(image_row)
    if pagul_label:
        label = f"{label}\n{pagul_label}"
    ax.text(
        0.04,
        0.94,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=CUTOUT_LABEL_FONT_SIZE,
        color="white",
        linespacing=0.95,
        clip_on=True,
        bbox=_quality_text_bbox(image_row.get("catalog_quality", "")),
    )
    hff_label = _format_hff_diagnostics(image_row)
    if hff_label:
        ax.text(
            0.96,
            0.06,
            hff_label,
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=CUTOUT_HFF_LABEL_FONT_SIZE,
            color="white",
            linespacing=0.92,
            clip_on=True,
            bbox=CUTOUT_TEXT_BBOX,
        )


def write_family_cutout_pdf(
    catalog: LiteratureCatalog,
    band_images: dict[str, BandImage],
    output: Path,
    *,
    bands: Sequence[str] = DEFAULT_BANDS,
    cutout_size_arcsec: float = DEFAULT_CUTOUT_SIZE_ARCSEC,
    families_per_page: int = DEFAULT_FAMILIES_PER_PAGE,
    max_images_per_family: int = DEFAULT_MAX_IMAGES_PER_FAMILY,
) -> int:
    if catalog.data.empty:
        raise ValueError(f"Selected catalog has no images: {catalog.path}")
    if families_per_page <= 0:
        raise ValueError("families_per_page must be positive.")
    if max_images_per_family <= 0:
        raise ValueError("max_images_per_family must be positive.")
    output = Path(output).with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)

    rgb_display = build_rgb_display(band_images, bands=bands)
    family_ids = _family_ids_in_catalog_order(catalog.data)
    detail_cols = FAMILY_CUTOUT_DETAIL_COLUMNS
    with PdfPages(output) as pdf:
        fig = plt.figure(figsize=_figure_size(detail_cols, detail_cols), dpi=CUTOUT_FIGURE_DPI)
        _style_cutout_figure(fig)
        grid = fig.add_gridspec(detail_cols, detail_cols)
        cluster_ax = fig.add_subplot(grid[:, :])
        _style_cutout_axis(cluster_ax)
        _draw_literature_cluster_overview_panel(
            cluster_ax,
            band_images,
            bands,
            rgb_display,
            catalog.data.reset_index(drop=True),
        )
        pdf.savefig(fig, facecolor=fig.get_facecolor(), **SAVEFIG_KWARGS)
        plt.close(fig)

        for family_id in family_ids:
            family = catalog.data.loc[catalog.data["family_id"].astype(str) == str(family_id)].reset_index(drop=True)
            overview_units = detail_cols
            detail_rows = max(1, int(math.ceil(float(len(family)) / float(detail_cols))))
            n_rows = overview_units + detail_rows
            fig = plt.figure(figsize=_figure_size(n_rows, detail_cols), dpi=CUTOUT_FIGURE_DPI)
            _style_cutout_figure(fig)
            grid = fig.add_gridspec(n_rows, detail_cols)
            overview_ax = fig.add_subplot(grid[:overview_units, :])
            _style_cutout_axis(overview_ax)
            _draw_literature_family_overview_panel(
                overview_ax,
                band_images,
                bands,
                rgb_display,
                str(family_id),
                family,
                max_images_per_family=max_images_per_family,
            )
            for panel_index, image_row in family.iterrows():
                detail_row = overview_units + int(panel_index) // detail_cols
                detail_col = int(panel_index) % detail_cols
                ax = fig.add_subplot(grid[detail_row, detail_col])
                _style_cutout_axis(ax)
                _draw_literature_detail_panel(
                    ax,
                    band_images,
                    bands,
                    rgb_display,
                    image_row,
                    cutout_size_arcsec=cutout_size_arcsec,
                )
            for blank_index in range(len(family), detail_rows * detail_cols):
                detail_row = overview_units + blank_index // detail_cols
                detail_col = blank_index % detail_cols
                ax = fig.add_subplot(grid[detail_row, detail_col])
                _style_cutout_axis(ax)
                ax.set_axis_off()
            pdf.savefig(fig, facecolor=fig.get_facecolor(), **SAVEFIG_KWARGS)
            plt.close(fig)
    return 1 + len(family_ids)


def run(
    *,
    literature_root: Path = DEFAULT_LITERATURE_ROOT,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    output: Path | None = None,
    cluster: str = DEFAULT_CLUSTER,
    source_slug: str = DEFAULT_SOURCE_SLUG,
    catalog_contains: str = DEFAULT_CATALOG_CONTAINS,
    cutout_size_arcsec: float = DEFAULT_CUTOUT_SIZE_ARCSEC,
    families_per_page: int = DEFAULT_FAMILIES_PER_PAGE,
    bands: Sequence[str] = DEFAULT_BANDS,
    image_scale: str = DEFAULT_IMAGE_SCALE,
    pagul_root: Path = DEFAULT_PAGUL_ROOT,
    pagul_match_radius_arcsec: float = DEFAULT_PAGUL_MATCH_RADIUS_ARCSEC,
    include_pagul_match_info: bool = True,
    hff_catalog_root: Path = DEFAULT_HFF_CATALOG_ROOT,
    include_hff_diagnostics: bool = True,
) -> Path:
    if len(bands) != 3:
        raise ValueError("--bands must provide exactly three bands in blue green red order.")
    catalog = select_literature_catalog(
        Path(literature_root),
        cluster=cluster,
        source_slug=source_slug,
        catalog_contains=catalog_contains,
    )
    if include_pagul_match_info:
        catalog = _catalog_with_pagul_match_info(
            catalog,
            pagul_root=Path(pagul_root),
            match_radius_arcsec=pagul_match_radius_arcsec,
        )
    if include_hff_diagnostics:
        catalog = _catalog_with_hff_diagnostics(catalog, hff_root=Path(hff_catalog_root))
    band_paths = find_rgb_band_paths(Path(image_dir), cluster=cluster, bands=bands, image_scale=image_scale)
    band_images = load_rgb_metadata(band_paths, bands=bands)
    output_path = Path(output) if output is not None else default_output_path(cluster, source_slug, catalog_contains)
    write_family_cutout_pdf(
        catalog,
        band_images,
        output_path,
        bands=bands,
        cutout_size_arcsec=cutout_size_arcsec,
        families_per_page=families_per_page,
    )
    return output_path.with_suffix(".pdf")


def _multi_output_path(catalog: LiteratureCatalog, output_dir: Path | None) -> Path:
    if output_dir is None:
        return default_output_path(catalog.cluster, catalog.source_slug, catalog.path.stem)
    return Path(output_dir) / default_output_path(catalog.cluster, catalog.source_slug, catalog.path.stem).name


def run_bergamini(
    *,
    literature_root: Path = DEFAULT_LITERATURE_ROOT,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    output: Path | None = None,
    cutout_size_arcsec: float = DEFAULT_CUTOUT_SIZE_ARCSEC,
    families_per_page: int = DEFAULT_FAMILIES_PER_PAGE,
    bands: Sequence[str] = DEFAULT_BANDS,
    image_scale: str = DEFAULT_IMAGE_SCALE,
    hff_catalog_root: Path = DEFAULT_HFF_CATALOG_ROOT,
    include_hff_diagnostics: bool = True,
) -> list[Path]:
    if len(bands) != 3:
        raise ValueError("--bands must provide exactly three bands in blue green red order.")
    if output is not None and Path(output).suffix:
        raise ValueError("--output must be a directory when --preset bergamini is used.")

    catalogs = select_bergamini_catalogs(Path(literature_root))
    output_dir = Path(output) if output is not None else None
    band_cache: dict[str, dict[str, BandImage]] = {}
    outputs: list[Path] = []
    for catalog in catalogs:
        if include_hff_diagnostics:
            catalog = _catalog_with_hff_diagnostics(catalog, hff_root=Path(hff_catalog_root))
        if catalog.cluster not in band_cache:
            band_paths = find_rgb_band_paths(Path(image_dir), cluster=catalog.cluster, bands=bands, image_scale=image_scale)
            band_cache[catalog.cluster] = load_rgb_metadata(band_paths, bands=bands)
        output_path = _multi_output_path(catalog, output_dir)
        write_family_cutout_pdf(
            catalog,
            band_cache[catalog.cluster],
            output_path,
            bands=bands,
            cutout_size_arcsec=cutout_size_arcsec,
            families_per_page=families_per_page,
        )
        outputs.append(output_path.with_suffix(".pdf"))
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot WCS image cutouts for literature multiple-image families.")
    parser.add_argument(
        "--preset",
        choices=(PRESET_SINGLE, PRESET_BERGAMINI),
        default=PRESET_SINGLE,
        help="Use 'bergamini' to render every staged Bergamini literature image catalog.",
    )
    parser.add_argument("--literature-root", type=Path, default=DEFAULT_LITERATURE_ROOT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--image-scale", choices=IMAGE_SCALE_CHOICES, default=DEFAULT_IMAGE_SCALE)
    parser.add_argument("--pagul-root", type=Path, default=DEFAULT_PAGUL_ROOT)
    parser.add_argument("--hff-catalog-root", type=Path, default=DEFAULT_HFF_CATALOG_ROOT)
    parser.add_argument("--output", type=Path, default=None, help="PDF path for single mode, or output directory for --preset bergamini.")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER)
    parser.add_argument("--source-slug", default=DEFAULT_SOURCE_SLUG)
    parser.add_argument("--catalog-contains", default=DEFAULT_CATALOG_CONTAINS)
    parser.add_argument("--cutout-size-arcsec", type=float, default=DEFAULT_CUTOUT_SIZE_ARCSEC)
    parser.add_argument("--families-per-page", type=int, default=DEFAULT_FAMILIES_PER_PAGE)
    parser.add_argument("--pagul-match-radius-arcsec", type=float, default=DEFAULT_PAGUL_MATCH_RADIUS_ARCSEC)
    parser.add_argument("--no-pagul-match-info", action="store_true", help="Render cutouts without Pagul2024 match labels.")
    parser.add_argument("--no-hff-diagnostics", action="store_true", help="Render cutouts without HFF family selection diagnostics.")
    parser.add_argument("--bands", nargs=3, default=list(DEFAULT_BANDS), metavar=("BLUE", "GREEN", "RED"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preset == PRESET_BERGAMINI:
        outputs = run_bergamini(
            literature_root=args.literature_root,
            image_dir=args.image_dir,
            output=args.output,
            cutout_size_arcsec=args.cutout_size_arcsec,
            families_per_page=args.families_per_page,
            bands=tuple(args.bands),
            image_scale=args.image_scale,
            hff_catalog_root=args.hff_catalog_root,
            include_hff_diagnostics=not args.no_hff_diagnostics,
        )
        for output in outputs:
            print(f"Wrote {output}")
        return 0

    output = run(
        literature_root=args.literature_root,
        image_dir=args.image_dir,
        output=args.output,
        cluster=args.cluster,
        source_slug=args.source_slug,
        catalog_contains=args.catalog_contains,
        cutout_size_arcsec=args.cutout_size_arcsec,
        families_per_page=args.families_per_page,
        bands=tuple(args.bands),
        image_scale=args.image_scale,
        pagul_root=args.pagul_root,
        pagul_match_radius_arcsec=args.pagul_match_radius_arcsec,
        include_pagul_match_info=not args.no_pagul_match_info,
        hff_catalog_root=args.hff_catalog_root,
        include_hff_diagnostics=not args.no_hff_diagnostics,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
