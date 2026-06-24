from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry


@dataclass(frozen=True)
class BandPhotometryConfig:
    band: str
    fits_path: Path
    zeropoint: float = 0.0
    use_for_catalog: bool = True


@dataclass(frozen=True)
class ImagePhotometryConfig:
    aperture_radius_arcsec: float = 0.35
    annulus_inner_radius_arcsec: float = 0.70
    annulus_outer_radius_arcsec: float = 1.20
    background_sigma_clip: float = 3.0
    background_sigma_clip_maxiters: int = 5
    default_catalog_reliability: float = 1.0
    min_flux_snr: float = 1.0e-12


def discover_simulation_bands(image_dir: str | Path) -> list[BandPhotometryConfig]:
    """Return available FF-SIMS HST photometric FITS bands in a deterministic order."""
    root = Path(image_dir)
    band_paths = sorted(root.glob("simulation_hst_*.fits"))
    configs: list[BandPhotometryConfig] = []
    for path in band_paths:
        stem = path.stem
        band = stem.removeprefix("simulation_")
        configs.append(BandPhotometryConfig(band=band, fits_path=path))
    return configs


_CLGAL_MAG_COLUMNS = {
    "hst_f435w": "mag_f435w",
    "hst_f606w": "mag_f606w",
    "hst_f814w": "mag_f814w",
    "hst_f105w": "mag_f105w",
    "hst_f125w": "mag_f125w",
    "hst_f140w": "mag_f140w",
    "hst_f160w": "mag_f160w",
}


def read_clgal_calibration_catalog(path: str | Path) -> pd.DataFrame:
    columns = [
        "image_label",
        "x_arcsec",
        "y_arcsec",
        "mag_f435w",
        "mag_f606w",
        "mag_f814w",
        "mag_f105w",
        "mag_f125w",
        "mag_f140w",
        "mag_f160w",
    ]
    catalog = pd.read_csv(path, sep=r"\s+", comment="#", names=columns)
    for column in columns:
        if column == "image_label":
            catalog[column] = catalog[column].astype(str)
        else:
            catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
    finite_position = np.isfinite(catalog["x_arcsec"]) & np.isfinite(catalog["y_arcsec"])
    return catalog.loc[finite_position].reset_index(drop=True)


def calibrate_band_zeropoints(
    band_configs: Iterable[BandPhotometryConfig],
    calibration_catalog_path: str | Path,
    config: ImagePhotometryConfig = ImagePhotometryConfig(),
) -> tuple[list[BandPhotometryConfig], pd.DataFrame]:
    """Infer per-band zeropoints from the FF-SIMS cluster-galaxy AB catalog."""
    calibration_catalog = read_clgal_calibration_catalog(calibration_catalog_path)
    calibrated_configs: list[BandPhotometryConfig] = []
    rows: list[dict[str, float | str | int | bool]] = []
    for band_config in band_configs:
        mag_column = _CLGAL_MAG_COLUMNS.get(str(band_config.band))
        if mag_column is None or mag_column not in calibration_catalog:
            calibrated_configs.append(
                BandPhotometryConfig(
                    band=band_config.band,
                    fits_path=band_config.fits_path,
                    zeropoint=band_config.zeropoint,
                    use_for_catalog=False,
                )
            )
            rows.append(
                {
                    "band": band_config.band,
                    "fits_path": str(band_config.fits_path),
                    "zeropoint": np.nan,
                    "n_calibrators": 0,
                    "use_for_catalog": False,
                }
            )
            continue
        instrumental = measure_band_photometry(calibration_catalog, band_config, config)
        known_mag = pd.to_numeric(calibration_catalog[mag_column], errors="coerce").to_numpy(dtype=float)
        flux = pd.to_numeric(instrumental["aperture_flux"], errors="coerce").to_numpy(dtype=float)
        valid = (
            np.isfinite(known_mag)
            & np.isfinite(flux)
            & (known_mag > 0.0)
            & (known_mag < 60.0)
            & (flux > 0.0)
        )
        if not np.any(valid):
            zeropoint = float("nan")
            use_for_catalog = False
        else:
            zeropoint_values = known_mag[valid] + 2.5 * np.log10(flux[valid])
            zeropoint = float(np.nanmedian(zeropoint_values))
            use_for_catalog = np.isfinite(zeropoint)
        calibrated_configs.append(
            BandPhotometryConfig(
                band=band_config.band,
                fits_path=band_config.fits_path,
                zeropoint=zeropoint if use_for_catalog else band_config.zeropoint,
                use_for_catalog=use_for_catalog,
            )
        )
        rows.append(
            {
                "band": band_config.band,
                "fits_path": str(band_config.fits_path),
                "zeropoint": zeropoint,
                "n_calibrators": int(np.sum(valid)),
                "use_for_catalog": bool(use_for_catalog),
            }
        )
    return calibrated_configs, pd.DataFrame(rows)


def read_lenstool_image_catalog(path: str | Path) -> tuple[int, pd.DataFrame]:
    """Read a Lenstool multiple-image catalog while preserving original columns."""
    rows: list[dict[str, float | str | int]] = []
    reference = 3
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#REFERENCE"):
            parts = stripped.split()
            if len(parts) >= 2:
                reference = int(parts[1])
            continue
        if stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            raise ValueError(f"{path}:{line_number} has fewer than 8 columns.")
        rows.append(
            {
                "line_number": line_number,
                "image_label": parts[0],
                "x_arcsec": float(parts[1]),
                "y_arcsec": float(parts[2]),
                "catalog_a": float(parts[3]),
                "catalog_b": float(parts[4]),
                "catalog_theta": float(parts[5]),
                "catalog_z": float(parts[6]),
                "catalog_mag": float(parts[7]),
                "family_reliability": float(parts[8]) if len(parts) >= 9 else np.nan,
                "catalog_mag_err": float(parts[9]) if len(parts) >= 10 else np.nan,
            }
        )
    return reference, pd.DataFrame(rows)


def _pixel_scale_arcsec(wcs: WCS) -> float:
    scales = wcs.proj_plane_pixel_scales()
    if hasattr(scales, "to_value"):
        proj = np.asarray(scales.to_value(u.deg), dtype=float)
    else:
        values = []
        for scale in scales:
            if hasattr(scale, "to_value"):
                values.append(float(scale.to_value(u.deg)))
            else:
                values.append(float(scale))
        proj = np.asarray(values, dtype=float)
    finite = proj[np.isfinite(proj) & (proj > 0.0)]
    if finite.size == 0:
        raise ValueError("FITS image WCS does not define a positive pixel scale.")
    return float(np.median(finite) * 3600.0)


def _arcsec_offsets_to_pixels(wcs: WCS, x_arcsec: np.ndarray, y_arcsec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world_x_deg = np.asarray(x_arcsec, dtype=float) / 3600.0
    world_y_deg = np.asarray(y_arcsec, dtype=float) / 3600.0
    x_pix, y_pix = wcs.world_to_pixel_values(world_x_deg, world_y_deg)
    return np.asarray(x_pix, dtype=float), np.asarray(y_pix, dtype=float)


def _flux_to_mag(flux: np.ndarray, flux_err: np.ndarray, zeropoint: float) -> tuple[np.ndarray, np.ndarray]:
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    finite = np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0.0) & (flux_err > 0.0)
    mag = np.full(flux.shape, np.nan, dtype=float)
    mag_err = np.full(flux.shape, np.nan, dtype=float)
    mag[finite] = float(zeropoint) - 2.5 * np.log10(flux[finite])
    mag_err[finite] = 2.5 / np.log(10.0) * flux_err[finite] / flux[finite]
    return mag, mag_err


def measure_band_photometry(
    image_catalog: pd.DataFrame,
    band_config: BandPhotometryConfig,
    config: ImagePhotometryConfig = ImagePhotometryConfig(),
) -> pd.DataFrame:
    """Measure local-background-subtracted aperture photometry for one FITS band."""
    with fits.open(band_config.fits_path) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f"{band_config.fits_path} does not contain a 2D image in HDU 0.")
    if str(header.get("CTYPE2", "")).upper() == "DEC---TAN":
        header["CTYPE2"] = "DEC--TAN"
    wcs = WCS(header)
    pixel_scale = _pixel_scale_arcsec(wcs)
    aperture_radius_pix = float(config.aperture_radius_arcsec) / pixel_scale
    annulus_inner_pix = float(config.annulus_inner_radius_arcsec) / pixel_scale
    annulus_outer_pix = float(config.annulus_outer_radius_arcsec) / pixel_scale
    x_pix, y_pix = _arcsec_offsets_to_pixels(
        wcs,
        image_catalog["x_arcsec"].to_numpy(dtype=float),
        image_catalog["y_arcsec"].to_numpy(dtype=float),
    )
    positions = np.column_stack([x_pix, y_pix])
    aperture = CircularAperture(positions, r=aperture_radius_pix)
    annulus = CircularAnnulus(positions, r_in=annulus_inner_pix, r_out=annulus_outer_pix)
    phot = aperture_photometry(data, aperture)
    sigma_clip = SigmaClip(
        sigma=float(config.background_sigma_clip),
        maxiters=int(config.background_sigma_clip_maxiters),
    )
    annulus_stats = ApertureStats(data, annulus, sigma_clip=sigma_clip)
    aperture_sum = np.asarray(phot["aperture_sum"], dtype=float)
    bkg_median = np.asarray(annulus_stats.median, dtype=float)
    bkg_std = np.asarray(annulus_stats.std, dtype=float)
    aperture_area = float(aperture.area)
    flux = aperture_sum - bkg_median * aperture_area
    flux_err = np.sqrt(np.maximum(aperture_area * np.square(bkg_std), 0.0))
    mag, mag_err = _flux_to_mag(flux, flux_err, band_config.zeropoint)
    ny, nx = data.shape
    in_bounds = (
        np.isfinite(x_pix)
        & np.isfinite(y_pix)
        & (x_pix >= 0.0)
        & (x_pix < float(nx))
        & (y_pix >= 0.0)
        & (y_pix < float(ny))
    )
    detected = in_bounds & np.isfinite(mag) & np.isfinite(mag_err)
    return pd.DataFrame(
        {
            "image_label": image_catalog["image_label"].astype(str).to_numpy(),
            "band": str(band_config.band),
            "fits_path": str(band_config.fits_path),
            "zeropoint": float(band_config.zeropoint),
            "use_for_catalog": bool(band_config.use_for_catalog),
            "x_pix": x_pix,
            "y_pix": y_pix,
            "pixel_scale_arcsec": pixel_scale,
            "aperture_radius_arcsec": float(config.aperture_radius_arcsec),
            "aperture_flux": flux,
            "aperture_flux_err": flux_err,
            "aperture_mag": mag,
            "aperture_mag_err": mag_err,
            "background_median_per_pix": bkg_median,
            "background_std_per_pix": bkg_std,
            "in_bounds": in_bounds,
            "detected": detected,
        }
    )


def _combine_band_magnitudes(band_table: pd.DataFrame, min_flux_snr: float) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for image_label, group in band_table.groupby("image_label", sort=False):
        flux = pd.to_numeric(group["aperture_flux"], errors="coerce").to_numpy(dtype=float)
        flux_err = pd.to_numeric(group["aperture_flux_err"], errors="coerce").to_numpy(dtype=float)
        mag = pd.to_numeric(group["aperture_mag"], errors="coerce").to_numpy(dtype=float)
        mag_err = pd.to_numeric(group["aperture_mag_err"], errors="coerce").to_numpy(dtype=float)
        use_for_catalog = group.get("use_for_catalog", pd.Series(True, index=group.index)).astype(bool).to_numpy()
        valid = (
            use_for_catalog
            & np.isfinite(flux)
            & np.isfinite(flux_err)
            & np.isfinite(mag)
            & np.isfinite(mag_err)
            & (flux > 0.0)
            & (flux_err > 0.0)
            & (mag_err > 0.0)
        )
        if not np.any(valid):
            rows.append(
                {
                    "image_label": str(image_label),
                    "combined_flux": np.nan,
                    "combined_flux_err": np.nan,
                    "combined_mag": np.nan,
                    "combined_mag_err": np.nan,
                    "n_detected_bands": 0,
                }
            )
            continue
        weights = 1.0 / np.square(np.maximum(flux_err[valid], float(min_flux_snr)))
        combined_flux = float(np.sum(weights * flux[valid]) / np.sum(weights))
        combined_flux_err = float(np.sqrt(1.0 / np.sum(weights)))
        mag_weights = 1.0 / np.square(mag_err[valid])
        combined_mag = float(np.sum(mag_weights * mag[valid]) / np.sum(mag_weights))
        combined_mag_err = float(np.sqrt(1.0 / np.sum(mag_weights)))
        rows.append(
            {
                "image_label": str(image_label),
                "combined_flux": combined_flux,
                "combined_flux_err": combined_flux_err,
                "combined_mag": combined_mag,
                "combined_mag_err": combined_mag_err,
                "n_detected_bands": int(np.sum(valid)),
            }
        )
    return pd.DataFrame(rows)


def band_photometry_wide_table(image_catalog: pd.DataFrame, band_table: pd.DataFrame) -> pd.DataFrame:
    """Return one row per image with flux, magnitude, and error columns for each measured band."""
    base_columns = [
        "image_label",
        "x_arcsec",
        "y_arcsec",
        "catalog_a",
        "catalog_b",
        "catalog_theta",
        "catalog_z",
        "family_reliability",
        "catalog_mag",
        "catalog_mag_err",
    ]
    available_base = [column for column in base_columns if column in image_catalog.columns]
    wide = image_catalog.loc[:, available_base].copy()
    if band_table.empty:
        return wide
    for band in sorted(str(item) for item in band_table["band"].dropna().unique()):
        band_rows = band_table.loc[band_table["band"].astype(str) == band].copy()
        band_rows = band_rows.drop_duplicates(subset=["image_label"], keep="first")
        suffix = band.lower().replace("-", "_")
        band_values = band_rows.set_index("image_label")
        for source_column, output_prefix in (
            ("aperture_mag", "mag"),
            ("aperture_mag_err", "mag_err"),
            ("aperture_flux", "flux"),
            ("aperture_flux_err", "flux_err"),
        ):
            wide[f"{output_prefix}_{suffix}"] = wide["image_label"].map(band_values[source_column])
        wide[f"detected_{suffix}"] = wide["image_label"].map(band_values["detected"]).fillna(False).astype(bool)
        wide[f"use_for_catalog_{suffix}"] = (
            wide["image_label"].map(band_values["use_for_catalog"]).fillna(False).astype(bool)
        )
    return wide


def measure_image_catalog_photometry(
    image_catalog_path: str | Path,
    band_configs: Iterable[BandPhotometryConfig],
    config: ImagePhotometryConfig = ImagePhotometryConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Measure all requested bands and return updated catalog rows plus per-band diagnostics."""
    reference, image_catalog = read_lenstool_image_catalog(image_catalog_path)
    band_tables = [measure_band_photometry(image_catalog, band_config, config) for band_config in band_configs]
    if band_tables:
        band_table = pd.concat(band_tables, ignore_index=True)
    else:
        band_table = pd.DataFrame()
    combined = _combine_band_magnitudes(band_table, config.min_flux_snr) if not band_table.empty else pd.DataFrame()
    merged = image_catalog.merge(combined, on="image_label", how="left")
    measured_mag = pd.to_numeric(merged["combined_mag"], errors="coerce")
    measured_err = pd.to_numeric(merged["combined_mag_err"], errors="coerce")
    merged["catalog_mag"] = np.where(np.isfinite(measured_mag), measured_mag, np.nan)
    merged["catalog_mag_err"] = np.where(np.isfinite(measured_err), measured_err, np.nan)
    merged["family_reliability"] = (
        pd.to_numeric(merged["family_reliability"], errors="coerce")
        .fillna(float(config.default_catalog_reliability))
        .clip(0.0, 1.0)
    )
    return merged, band_table, reference


def write_photometric_image_catalog(catalog: pd.DataFrame, output_path: str | Path, reference: int = 3) -> None:
    """Write a Lenstool-style image catalog with reliability and magnitude error columns."""
    path = Path(output_path)
    lines = [f"#REFERENCE {int(reference)}"]
    for row in catalog.itertuples(index=False):
        lines.append(
            f"{str(row.image_label):>10s} "
            f"{float(row.x_arcsec): .8f} {float(row.y_arcsec): .8f} "
            f"{float(row.catalog_a):.4f} {float(row.catalog_b):.4f} {float(row.catalog_theta):.1f} "
            f"{float(row.catalog_z):.8f} {float(row.catalog_mag):.8f} "
            f"{float(row.family_reliability):.6f} {float(row.catalog_mag_err):.8f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
