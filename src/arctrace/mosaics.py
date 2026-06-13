"""FITS mosaic loading and cutout extraction.

Loader behavior mirrors the family-cutout scripts in ``scripts/`` (which are
not an importable package): first 2D celestial HDU, PHOTFLAM/PHOTPLAM from the
data or primary header, strided sigma-clipped global background with drizzle
zero-padding excluded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.stats import mad_std, sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from .errors import CutoutError


@dataclass(frozen=True)
class BandMosaic:
    band: str
    path: Path
    hdu_index: int
    shape: tuple[int, int]
    wcs: WCS
    pixel_scale_arcsec: float
    photflam: float | None
    photplam: float | None
    background: float
    background_sigma: float


@dataclass(frozen=True)
class CutoutData:
    band: str
    data: np.ndarray
    wcs: WCS
    pixel_scale_arcsec: float
    photflam: float | None
    photplam: float | None
    global_background: float
    global_background_sigma: float


def _header_photometry_keyword(headers, keyword: str) -> float | None:
    for header in headers:
        if header is None:
            continue
        value = header.get(keyword)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0.0:
            return number
    return None


def _find_celestial_hdu(hdul) -> int:
    for index, hdu in enumerate(hdul):
        data = hdu.data
        if data is None or getattr(data, "ndim", 0) != 2:
            continue
        try:
            wcs = WCS(hdu.header)
        except Exception:
            continue
        if wcs.has_celestial:
            return index
    raise CutoutError("No 2D HDU with a celestial WCS found.")


def _global_background(data: np.ndarray, max_samples: int = 4_000_000) -> tuple[float, float]:
    stride = max(1, int(math.sqrt(max(data.shape[0] * data.shape[1] / max_samples, 1.0))))
    sample = np.asarray(data[::stride, ::stride], dtype=float)
    finite = np.isfinite(sample) & (sample != 0.0)
    values = sample[finite]
    if values.size < 100:
        return 0.0, 1.0
    _, median, _ = sigma_clipped_stats(values, sigma=3.0, maxiters=5)
    sigma = float(mad_std(values - median))
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(values)) or 1.0
    return float(median), sigma


def load_band_mosaic(band: str, path: str | Path) -> BandMosaic:
    path = Path(path)
    if not path.exists():
        raise CutoutError(f"Mosaic for band {band} not found: {path}")
    with fits.open(path, memmap=True) as hdul:
        index = _find_celestial_hdu(hdul)
        hdu = hdul[index]
        header = hdu.header
        wcs = WCS(header).celestial
        scales_deg = proj_plane_pixel_scales(wcs)
        scales_arcsec = np.abs(scales_deg) * 3600.0
        if scales_arcsec.size >= 2 and abs(scales_arcsec[0] - scales_arcsec[1]) > 5.0e-3 * scales_arcsec.mean():
            import warnings

            warnings.warn(
                f"Band {band}: anisotropic pixel scale {scales_arcsec.tolist()} arcsec; using the mean.",
                stacklevel=2,
            )
        pixel_scale = float(scales_arcsec.mean())
        primary_header = hdul[0].header if index != 0 else None
        photflam = _header_photometry_keyword([header, primary_header], "PHOTFLAM")
        photplam = _header_photometry_keyword([header, primary_header], "PHOTPLAM")
        background, background_sigma = _global_background(hdu.data)
        shape = (int(hdu.data.shape[0]), int(hdu.data.shape[1]))
    return BandMosaic(
        band=str(band),
        path=path,
        hdu_index=index,
        shape=shape,
        wcs=wcs,
        pixel_scale_arcsec=pixel_scale,
        photflam=photflam,
        photplam=photplam,
        background=background,
        background_sigma=background_sigma,
    )


def load_band_mosaics(paths_by_band: dict[str, Path | str]) -> dict[str, BandMosaic]:
    return {str(band): load_band_mosaic(band, path) for band, path in paths_by_band.items()}


def extract_cutout(mosaic: BandMosaic, center: SkyCoord, size_arcsec: float) -> CutoutData:
    npix = int(math.ceil(float(size_arcsec) / mosaic.pixel_scale_arcsec))
    npix = max(npix, 16)
    if npix % 2 == 0:
        npix += 1
    try:
        with fits.open(mosaic.path, memmap=True) as hdul:
            cutout = Cutout2D(
                hdul[mosaic.hdu_index].data,
                position=center,
                size=(npix, npix),
                wcs=mosaic.wcs,
                mode="partial",
                fill_value=np.nan,
                copy=True,
            )
    except NoOverlapError as exc:
        raise CutoutError(
            f"Seed {center.ra.deg:.6f},{center.dec.deg:.6f} has no overlap with the {mosaic.band} mosaic."
        ) from exc
    return CutoutData(
        band=mosaic.band,
        data=np.asarray(cutout.data, dtype=float),
        wcs=cutout.wcs,
        pixel_scale_arcsec=mosaic.pixel_scale_arcsec,
        photflam=mosaic.photflam,
        photplam=mosaic.photplam,
        global_background=mosaic.background,
        global_background_sigma=mosaic.background_sigma,
    )


def mosaic_center(mosaic: BandMosaic) -> SkyCoord:
    ny, nx = mosaic.shape
    world = mosaic.wcs.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
    return SkyCoord(ra=world.ra, dec=world.dec, frame="icrs")


def mosaic_from_arrays(
    band: str,
    data: np.ndarray,
    wcs: WCS,
    *,
    photflam: float | None = None,
    photplam: float | None = None,
) -> tuple[BandMosaic, np.ndarray]:
    """In-memory mosaic for tests/interactive reuse (no file I/O on extract)."""
    scales_arcsec = np.abs(proj_plane_pixel_scales(wcs)) * 3600.0
    background, background_sigma = _global_background(np.asarray(data, dtype=float))
    mosaic = BandMosaic(
        band=str(band),
        path=Path("<memory>"),
        hdu_index=0,
        shape=(int(data.shape[0]), int(data.shape[1])),
        wcs=wcs.celestial,
        pixel_scale_arcsec=float(scales_arcsec.mean()),
        photflam=photflam,
        photplam=photplam,
        background=background,
        background_sigma=background_sigma,
    )
    return mosaic, np.asarray(data, dtype=float)


def extract_cutout_from_array(
    mosaic: BandMosaic,
    data: np.ndarray,
    center: SkyCoord,
    size_arcsec: float,
) -> CutoutData:
    npix = int(math.ceil(float(size_arcsec) / mosaic.pixel_scale_arcsec))
    npix = max(npix, 16)
    if npix % 2 == 0:
        npix += 1
    try:
        cutout = Cutout2D(
            np.asarray(data, dtype=float),
            position=center,
            size=(npix, npix),
            wcs=mosaic.wcs,
            mode="partial",
            fill_value=np.nan,
            copy=True,
        )
    except NoOverlapError as exc:
        raise CutoutError(
            f"Seed {center.ra.deg:.6f},{center.dec.deg:.6f} has no overlap with the {mosaic.band} array."
        ) from exc
    return CutoutData(
        band=mosaic.band,
        data=np.asarray(cutout.data, dtype=float),
        wcs=cutout.wcs,
        pixel_scale_arcsec=mosaic.pixel_scale_arcsec,
        photflam=mosaic.photflam,
        photplam=mosaic.photplam,
        global_background=mosaic.background,
        global_background_sigma=mosaic.background_sigma,
    )


__all__ = [
    "BandMosaic",
    "CutoutData",
    "load_band_mosaic",
    "load_band_mosaics",
    "extract_cutout",
    "extract_cutout_from_array",
    "mosaic_from_arrays",
    "mosaic_center",
]
