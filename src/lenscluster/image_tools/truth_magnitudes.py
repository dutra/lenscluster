from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS

from .photometry import read_lenstool_image_catalog, write_photometric_image_catalog


HST_BAND_COLOR_REL_F160W: dict[str, float] = {
    "hst_f435w": 1.20,
    "hst_f606w": 0.80,
    "hst_f814w": 0.40,
    "hst_f105w": 0.15,
    "hst_f125w": 0.05,
    "hst_f140w": 0.02,
    "hst_f160w": 0.00,
}


@dataclass(frozen=True)
class TruthMagnitudeConfig:
    z_lens: float = 0.5
    z_map_source: float = 9.0
    h0: float = 70.4
    om0: float = 0.272
    source_mag_f160w_mean: float = 27.0
    source_mag_f160w_sigma: float = 1.0
    color_scatter_sigma: float = 0.15
    magnitude_error: float = 0.01
    mu_floor: float = 1.0e-3
    random_seed: int = 12_345
    band_colors_rel_f160w: dict[str, float] = field(default_factory=lambda: dict(HST_BAND_COLOR_REL_F160W))


def _load_wcs_map(path: str | Path) -> tuple[np.ndarray, WCS]:
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f"{path} does not contain a 2D FITS image.")
    if str(header.get("CTYPE2", "")).upper() == "DEC---TAN":
        header["CTYPE2"] = "DEC--TAN"
    return data, WCS(header)


def _bilinear_sample(data: np.ndarray, x_pix: np.ndarray, y_pix: np.ndarray) -> np.ndarray:
    ny, nx = data.shape
    x = np.clip(np.asarray(x_pix, dtype=float), 0.0, float(nx - 1))
    y = np.clip(np.asarray(y_pix, dtype=float), 0.0, float(ny - 1))
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    wx = x - x0
    wy = y - y0
    return (
        (1.0 - wx) * (1.0 - wy) * data[y0, x0]
        + wx * (1.0 - wy) * data[y0, x1]
        + (1.0 - wx) * wy * data[y1, x0]
        + wx * wy * data[y1, x1]
    )


def _lensing_efficiency(cosmo: FlatLambdaCDM, z_lens: float, z_source: np.ndarray | float) -> np.ndarray:
    z_source_array = np.asarray(z_source, dtype=float)
    efficiency = np.zeros_like(z_source_array, dtype=float)
    valid = np.isfinite(z_source_array) & (z_source_array > float(z_lens))
    if np.any(valid):
        d_s = cosmo.angular_diameter_distance(z_source_array[valid]).value
        d_ls = cosmo.angular_diameter_distance(float(z_lens), z_source_array[valid]).value
        efficiency[valid] = d_ls / np.maximum(d_s, 1.0e-30)
    return efficiency


def _family_source_magnitudes(
    image_catalog: pd.DataFrame,
    bands: list[str],
    config: TruthMagnitudeConfig,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(int(config.random_seed))
    family_ids = sorted(image_catalog["image_label"].astype(str).str.split(".", n=1).str[0].unique().tolist())
    source_magnitudes: dict[str, dict[str, float]] = {}
    for family_id in family_ids:
        f160w = float(rng.normal(config.source_mag_f160w_mean, config.source_mag_f160w_sigma))
        source_magnitudes[family_id] = {}
        for band in bands:
            color = float(config.band_colors_rel_f160w.get(band, 0.0))
            if float(config.color_scatter_sigma) > 0.0:
                color += float(rng.normal(0.0, config.color_scatter_sigma))
            source_magnitudes[family_id][band] = f160w + color
    return source_magnitudes


def build_truth_magnitude_catalog(
    image_catalog_path: str | Path,
    map_dir: str | Path,
    output_catalog_path: str | Path,
    output_band_table_path: str | Path,
    *,
    config: TruthMagnitudeConfig = TruthMagnitudeConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Write ideal apparent magnitudes from true FF-SIMS kappa/shear maps."""
    reference, image_catalog = read_lenstool_image_catalog(image_catalog_path)
    map_root = Path(map_dir)
    kappa_z9, wcs = _load_wcs_map(map_root / "kappa_z9_0.fits")
    gamma_x_z9, _ = _load_wcs_map(map_root / "gammax_z9_0.fits")
    gamma_y_z9, _ = _load_wcs_map(map_root / "gammay_z9_0.fits")

    x_pix, y_pix = wcs.world_to_pixel_values(
        image_catalog["x_arcsec"].to_numpy(dtype=float) / 3600.0,
        image_catalog["y_arcsec"].to_numpy(dtype=float) / 3600.0,
    )
    kappa_ref = _bilinear_sample(kappa_z9, x_pix, y_pix)
    gamma_x_ref = _bilinear_sample(gamma_x_z9, x_pix, y_pix)
    gamma_y_ref = _bilinear_sample(gamma_y_z9, x_pix, y_pix)

    cosmo = FlatLambdaCDM(H0=float(config.h0), Om0=float(config.om0), Tcmb0=2.725)
    z_source = image_catalog["catalog_z"].to_numpy(dtype=float)
    eff = _lensing_efficiency(cosmo, float(config.z_lens), z_source)
    eff_ref = float(_lensing_efficiency(cosmo, float(config.z_lens), np.asarray(float(config.z_map_source))))
    scale = eff / max(eff_ref, 1.0e-30)
    kappa = scale * kappa_ref
    gamma_x = scale * gamma_x_ref
    gamma_y = scale * gamma_y_ref
    det_a = np.square(1.0 - kappa) - np.square(gamma_x) - np.square(gamma_y)
    mu = 1.0 / np.where(np.isfinite(det_a) & (np.abs(det_a) > 1.0e-12), det_a, np.sign(det_a) * 1.0e-12)
    abs_mu = np.maximum(np.abs(mu), float(config.mu_floor))

    bands = sorted(config.band_colors_rel_f160w)
    source_magnitudes = _family_source_magnitudes(image_catalog, bands, config)
    rows: list[dict[str, object]] = []
    combined_mags: list[float] = []
    for row_index, row in enumerate(image_catalog.itertuples(index=False)):
        image_label = str(row.image_label)
        family_id = image_label.split(".", 1)[0]
        output_row: dict[str, object] = {
            "image_label": image_label,
            "x_arcsec": float(row.x_arcsec),
            "y_arcsec": float(row.y_arcsec),
            "catalog_a": float(row.catalog_a),
            "catalog_b": float(row.catalog_b),
            "catalog_theta": float(row.catalog_theta),
            "catalog_z": float(row.catalog_z),
            "family_reliability": float(row.family_reliability)
            if np.isfinite(float(row.family_reliability))
            else float(1.0),
            "truth_mu": float(mu[row_index]),
            "truth_abs_mu": float(abs_mu[row_index]),
        }
        band_mags: list[float] = []
        for band in bands:
            source_mag = float(source_magnitudes[family_id][band])
            apparent_mag = source_mag - 2.5 * np.log10(abs_mu[row_index])
            output_row[f"mag_{band}"] = float(apparent_mag)
            output_row[f"mag_err_{band}"] = float(config.magnitude_error)
            output_row[f"detected_{band}"] = True
            output_row[f"use_for_catalog_{band}"] = True
            band_mags.append(float(apparent_mag))
        combined_mag = float(np.mean(band_mags))
        output_row["catalog_mag"] = combined_mag
        output_row["catalog_mag_err"] = float(config.magnitude_error) / np.sqrt(float(len(band_mags)))
        rows.append(output_row)
        combined_mags.append(combined_mag)

    band_table = pd.DataFrame(rows)
    catalog = image_catalog.copy()
    catalog["catalog_mag"] = np.asarray(combined_mags, dtype=float)
    catalog["catalog_mag_err"] = float(config.magnitude_error) / np.sqrt(float(len(bands)))
    catalog["family_reliability"] = pd.to_numeric(catalog["family_reliability"], errors="coerce").fillna(1.0)
    write_photometric_image_catalog(catalog, output_catalog_path, reference=reference)
    Path(output_band_table_path).parent.mkdir(parents=True, exist_ok=True)
    band_table.to_csv(output_band_table_path, index=False)
    return catalog, band_table, reference
