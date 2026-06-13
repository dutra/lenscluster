"""Shared synthetic-data helpers for the arctrace test suite.

Truth values are derived independently of the arctrace package: pixel -> sky
goes through astropy WCS only, and the tangent-plane offsets formula is
written out inline (it is itself validated against lenscluster.lenstool_parser
in test_arctrace_geometry.py).
"""

from __future__ import annotations

import math

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def make_tan_wcs(
    ra0_deg: float,
    dec0_deg: float,
    *,
    pixscale_arcsec: float,
    rotation_deg: float = 0.0,
    flip_ra: bool = True,
    crpix: tuple[float, float] = (50.0, 50.0),
) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [float(ra0_deg), float(dec0_deg)]
    wcs.wcs.crpix = [float(crpix[0]), float(crpix[1])]
    rot = math.radians(float(rotation_deg))
    wcs.wcs.pc = [[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]]
    scale_deg = float(pixscale_arcsec) / 3600.0
    wcs.wcs.cdelt = [(-scale_deg if flip_ra else scale_deg), scale_deg]
    return wcs


def offsets_arcsec(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Inline solver-offsets formula (x westward, y northward), for truth values."""
    cos_dec0 = math.cos(math.radians(dec0_deg))
    x = (ra0_deg - np.asarray(ra_deg, dtype=float)) * cos_dec0 * 3600.0
    y = (np.asarray(dec_deg, dtype=float) - dec0_deg) * 3600.0
    return x, y


def offsets_to_radec(x_arcsec, y_arcsec, ra0_deg, dec0_deg):
    cos_dec0 = math.cos(math.radians(dec0_deg))
    ra = ra0_deg - np.asarray(x_arcsec, dtype=float) / (3600.0 * cos_dec0)
    dec = dec0_deg + np.asarray(y_arcsec, dtype=float) / 3600.0
    return ra, dec


def paint_arc_on_wcs(
    wcs: WCS,
    shape: tuple[int, int],
    *,
    circle_center_offsets_arcsec: tuple[float, float],
    radius_arcsec: float,
    azimuth_center_rad: float,
    half_span_rad: float,
    width_arcsec: float,
    peak: float,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    taper_rad: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """Paint a curved Gaussian ribbon onto an image with the given WCS.

    The ribbon follows a circle defined in the solver offsets frame anchored at
    the WCS reference point (crval). Returns (image, truth) where truth holds
    the analytic anchor position, axial tangent angle in the offsets frame and
    curvature in arcsec^-1.
    """
    ny, nx = shape
    ra0_deg, dec0_deg = (float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1]))
    yy, xx = np.mgrid[0:ny, 0:nx]
    world = wcs.pixel_to_world(xx.astype(float), yy.astype(float))
    x_off, y_off = offsets_arcsec(world.ra.deg, world.dec.deg, ra0_deg, dec0_deg)

    xc, yc = (float(circle_center_offsets_arcsec[0]), float(circle_center_offsets_arcsec[1]))
    dx = x_off - xc
    dy = y_off - yc
    r = np.hypot(dx, dy)
    azimuth = np.arctan2(dy, dx)
    dpsi = np.arctan2(np.sin(azimuth - azimuth_center_rad), np.cos(azimuth - azimuth_center_rad))

    radial_distance = np.abs(r - float(radius_arcsec))
    sigma_w = float(width_arcsec) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    profile = np.exp(-0.5 * (radial_distance / sigma_w) ** 2)
    # Smooth cosine taper at the angular ends of the arc.
    inside = np.abs(dpsi) <= half_span_rad
    edge = (np.abs(dpsi) > half_span_rad) & (np.abs(dpsi) <= half_span_rad + taper_rad)
    taper = np.zeros_like(profile)
    taper[inside] = 1.0
    taper[edge] = 0.5 * (1.0 + np.cos(np.pi * (np.abs(dpsi[edge]) - half_span_rad) / taper_rad))
    image = float(peak) * profile * taper

    if noise_sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng(0)
        image = image + rng.normal(0.0, float(noise_sigma), size=image.shape)

    anchor_x = xc + float(radius_arcsec) * math.cos(azimuth_center_rad)
    anchor_y = yc + float(radius_arcsec) * math.sin(azimuth_center_rad)
    anchor_ra, anchor_dec = offsets_to_radec(anchor_x, anchor_y, ra0_deg, dec0_deg)
    tangent_angle = (azimuth_center_rad + 0.5 * math.pi) % math.pi
    truth = {
        "anchor_ra_deg": float(anchor_ra),
        "anchor_dec_deg": float(anchor_dec),
        "anchor_offsets_arcsec": (anchor_x, anchor_y),
        "tangent_angle_offset_rad": float(tangent_angle),
        "curvature_arcsec_inv": 1.0 / float(radius_arcsec),
        "center_offsets_arcsec": (xc, yc),
        "length_arcsec": 2.0 * half_span_rad * float(radius_arcsec),
        "width_arcsec": float(width_arcsec),
        "ra0_deg": ra0_deg,
        "dec0_deg": dec0_deg,
    }
    return image.astype(float), truth


def write_synthetic_arc_fits(
    path,
    *,
    ra0_deg: float = 39.97,
    dec0_deg: float = -1.58,
    npix: int = 400,
    pixscale_arcsec: float = 0.06,
    rotation_deg: float = 0.0,
    flip_ra: bool = True,
    radius_arcsec: float = 8.0,
    azimuth_center_rad: float = 0.6,
    half_span_rad: float = 0.6,
    width_arcsec: float = 0.35,
    peak: float = 2.0,
    noise_sigma: float = 0.05,
    band: str = "F814W",
    rng: np.random.Generator | None = None,
) -> dict:
    """Write a synthetic single-band FITS mosaic containing one curved arc."""
    wcs = make_tan_wcs(
        ra0_deg,
        dec0_deg,
        pixscale_arcsec=pixscale_arcsec,
        rotation_deg=rotation_deg,
        flip_ra=flip_ra,
        crpix=(npix / 2.0, npix / 2.0),
    )
    image, truth = paint_arc_on_wcs(
        wcs,
        (npix, npix),
        circle_center_offsets_arcsec=(0.0, 0.0),
        radius_arcsec=radius_arcsec,
        azimuth_center_rad=azimuth_center_rad,
        half_span_rad=half_span_rad,
        width_arcsec=width_arcsec,
        peak=peak,
        noise_sigma=noise_sigma,
        rng=rng,
    )
    header = wcs.to_header()
    header["PHOTFLAM"] = 7.0e-20
    header["PHOTPLAM"] = 8045.0
    header["FILTER"] = band
    fits.PrimaryHDU(data=image.astype(np.float32), header=header).writeto(path, overwrite=True)
    truth["noise_sigma"] = float(noise_sigma)
    truth["pixscale_arcsec"] = float(pixscale_arcsec)
    return truth
