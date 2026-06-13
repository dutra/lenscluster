"""Flux-weighted ridge tracing of a segmented arc.

Pass 1 slices the mask perpendicular to its principal axis and takes the
flux-weighted transverse centroid per slice. Pass 2 re-slices in polar
coordinates around the fitted circle center, i.e. in the radial/tangential
eigenframe of the curved-arc basis: azimuthal bins, radial centroids. The
transverse width is the radial-stretch proxy and the arclength the tangential
extent, so width/length tracks the MST-invariant stretch ratio
lambda_tan/lambda_rad (Birrer 2021, Eq. 40).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import RidgeError

_FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))
_MIN_SIGMA_PIX = 0.3


@dataclass(frozen=True)
class RidgePoint:
    x_pix: float
    y_pix: float
    sigma_pix: float
    snr: float
    arclength_pix: float


@dataclass(frozen=True)
class RidgeTrace:
    points: tuple[RidgePoint, ...]
    length_arcsec: float
    width_arcsec: float
    axis_ratio: float
    orientation_pix_rad: float

    @property
    def x_pix(self) -> np.ndarray:
        return np.array([p.x_pix for p in self.points])

    @property
    def y_pix(self) -> np.ndarray:
        return np.array([p.y_pix for p in self.points])

    @property
    def sigma_pix(self) -> np.ndarray:
        return np.array([p.sigma_pix for p in self.points])

    @property
    def arclength_pix(self) -> np.ndarray:
        return np.array([p.arclength_pix for p in self.points])


def _masked_flux_weights(data_sub: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(mask)
    flux = np.clip(data_sub[rows, cols], 0.0, None)
    if flux.sum() <= 0.0:
        raise RidgeError("Arc mask contains no positive flux.")
    return cols.astype(float), rows.astype(float), flux


def _build_trace(
    coords_along: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    flux: np.ndarray,
    transverse: np.ndarray,
    *,
    slice_width: np.ndarray | float,
    bin_index: np.ndarray,
    sky_sigma: float,
    min_snr: float,
    min_pix_per_slice: int,
    pixel_scale_arcsec: float,
    point_from_bin,
    orientation_pix_rad: float,
) -> RidgeTrace:
    """Common slice-statistics machinery for both passes.

    ``point_from_bin(s_mean, t_mean) -> (x_pix, y_pix)`` maps the per-bin
    longitudinal/transverse centroids back to pixel coordinates;
    ``transverse`` holds the per-pixel transverse coordinate in pixels.
    """
    points: list[tuple[float, float, float, float, float]] = []
    widths: list[tuple[float, float]] = []
    for index in np.unique(bin_index):
        sel = bin_index == index
        if int(np.count_nonzero(sel)) < int(min_pix_per_slice):
            continue
        w = flux[sel]
        w_sum = float(w.sum())
        if w_sum <= 0.0:
            continue
        n_pix = int(np.count_nonzero(sel))
        snr = w_sum / (float(sky_sigma) * math.sqrt(n_pix))
        if snr < float(min_snr):
            continue
        s_mean = float(np.sum(w * coords_along[sel]) / w_sum)
        t_mean = float(np.sum(w * transverse[sel]) / w_sum)
        t_rms = float(math.sqrt(max(np.sum(w * (transverse[sel] - t_mean) ** 2) / w_sum, 0.0)))
        sigma = max(t_rms / math.sqrt(max(snr, 1.0e-6)), _MIN_SIGMA_PIX)
        x_pix, y_pix = point_from_bin(s_mean, t_mean)
        points.append((s_mean, float(x_pix), float(y_pix), sigma, snr))
        widths.append((t_rms, w_sum))

    if len(points) < 3:
        raise RidgeError(f"Only {len(points)} valid ridge slices; arc too faint or too short.")

    points.sort(key=lambda item: item[0])
    xs = np.array([p[1] for p in points])
    ys = np.array([p[2] for p in points])
    seglen = np.hypot(np.diff(xs), np.diff(ys))
    arclengths = np.concatenate([[0.0], np.cumsum(seglen)])
    ridge_points = tuple(
        RidgePoint(
            x_pix=points[i][1],
            y_pix=points[i][2],
            sigma_pix=points[i][3],
            snr=points[i][4],
            arclength_pix=float(arclengths[i]),
        )
        for i in range(len(points))
    )
    length_pix = float(arclengths[-1])
    rms_values = np.array([w[0] for w in widths])
    rms_weights = np.array([w[1] for w in widths])
    order = np.argsort(rms_values)
    cum = np.cumsum(rms_weights[order])
    median_rms = float(rms_values[order][np.searchsorted(cum, 0.5 * cum[-1])])
    width_pix = _FWHM_FACTOR * median_rms
    length_arcsec = length_pix * float(pixel_scale_arcsec)
    width_arcsec = width_pix * float(pixel_scale_arcsec)
    axis_ratio = float(np.clip(width_arcsec / max(length_arcsec, 1.0e-9), 1.0e-6, 1.0))
    return RidgeTrace(
        points=ridge_points,
        length_arcsec=length_arcsec,
        width_arcsec=width_arcsec,
        axis_ratio=axis_ratio,
        orientation_pix_rad=float(orientation_pix_rad),
    )


def trace_ridge_moments(
    data_sub: np.ndarray,
    mask: np.ndarray,
    *,
    slice_width_pix: float,
    sky_sigma: float,
    min_snr: float,
    min_pix_per_slice: int = 3,
    pixel_scale_arcsec: float = 1.0,
) -> RidgeTrace:
    """Pass 1: slices perpendicular to the flux-weighted principal axis."""
    x, y, flux = _masked_flux_weights(data_sub, mask)
    w = flux / flux.sum()
    cx = float(np.sum(w * x))
    cy = float(np.sum(w * y))
    sxx = float(np.sum(w * (x - cx) ** 2))
    syy = float(np.sum(w * (y - cy) ** 2))
    sxy = float(np.sum(w * (x - cx) * (y - cy)))
    orientation = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    cos_o, sin_o = math.cos(orientation), math.sin(orientation)
    s = (x - cx) * cos_o + (y - cy) * sin_o
    t = -(x - cx) * sin_o + (y - cy) * cos_o
    width = max(float(slice_width_pix), 1.0)
    bin_index = np.floor(s / width).astype(int)

    def point_from_bin(s_mean: float, t_mean: float) -> tuple[float, float]:
        return (cx + s_mean * cos_o - t_mean * sin_o, cy + s_mean * sin_o + t_mean * cos_o)

    return _build_trace(
        s,
        x,
        y,
        flux,
        t,
        slice_width=width,
        bin_index=bin_index,
        sky_sigma=sky_sigma,
        min_snr=min_snr,
        min_pix_per_slice=min_pix_per_slice,
        pixel_scale_arcsec=pixel_scale_arcsec,
        point_from_bin=point_from_bin,
        orientation_pix_rad=orientation,
    )


def resample_ridge_polar(
    data_sub: np.ndarray,
    mask: np.ndarray,
    *,
    center_xy_pix: tuple[float, float],
    slice_width_pix: float,
    sky_sigma: float,
    min_snr: float,
    min_pix_per_slice: int = 3,
    pixel_scale_arcsec: float = 1.0,
) -> RidgeTrace:
    """Pass 2: azimuthal slices around the fitted curvature center."""
    x, y, flux = _masked_flux_weights(data_sub, mask)
    dx = x - float(center_xy_pix[0])
    dy = y - float(center_xy_pix[1])
    r = np.hypot(dx, dy)
    if np.any(r <= 0.0):
        keep = r > 0.0
        x, y, flux, dx, dy, r = x[keep], y[keep], flux[keep], dx[keep], dy[keep], r[keep]
    azimuth = np.arctan2(dy, dx)
    w = flux / flux.sum()
    azimuth0 = math.atan2(float(np.sum(w * np.sin(azimuth))), float(np.sum(w * np.cos(azimuth))))
    dpsi = np.arctan2(np.sin(azimuth - azimuth0), np.cos(azimuth - azimuth0))
    r0 = float(np.sum(w * r))
    if r0 <= 0.0:
        raise RidgeError("Degenerate polar resampling: zero mean radius.")
    delta = max(float(slice_width_pix), 1.0) / r0
    bin_index = np.floor(dpsi / delta).astype(int)
    # Longitudinal coordinate = arclength along the circle, transverse = radius.
    s = dpsi * r0

    def point_from_bin(s_mean: float, r_mean: float) -> tuple[float, float]:
        psi = azimuth0 + s_mean / r0
        return (
            float(center_xy_pix[0]) + r_mean * math.cos(psi),
            float(center_xy_pix[1]) + r_mean * math.sin(psi),
        )

    return _build_trace(
        s,
        x,
        y,
        flux,
        r,
        slice_width=delta,
        bin_index=bin_index,
        sky_sigma=sky_sigma,
        min_snr=min_snr,
        min_pix_per_slice=min_pix_per_slice,
        pixel_scale_arcsec=pixel_scale_arcsec,
        point_from_bin=point_from_bin,
        orientation_pix_rad=float(np.mod(azimuth0 + 0.5 * math.pi, math.pi)),
    )
