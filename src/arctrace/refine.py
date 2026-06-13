"""Optional pixel-level PSF-convolved refinement of the geometric circle fit.

A lightweight version of the forward-modeling approach of Birrer (2021,
Sect. 5.1): the same circle geometry as the geometric fit, with a Gaussian
transverse profile and per-slice amplitudes solved linearly, convolved with a
Gaussian PSF and fit to the background-subtracted pixels. The free non-linear
parameters are exactly the observables (tangent angle at the anchor, signed
curvature) plus a transverse ridge offset and the intrinsic width, so the
Laplace covariance at the optimum gives sigmas directly in the quantities the
solver consumes. Useful for short arcs (length within a few PSF FWHM) where
endpoint erosion and PSF mixing bias the geometric ridge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, optimize


@dataclass(frozen=True)
class ForwardRefineResult:
    success: bool
    message: str
    tangent_angle_rad: float
    curvature_arcsec_inv: float
    curvature_side: int
    sigma_tangent_rad: float
    sigma_curvature_arcsec_inv: float
    reduced_chi2: float


def _circle_center(
    anchor_xy: tuple[float, float],
    psi: float,
    kappa_signed: float,
    delta: float,
) -> tuple[float, float, float]:
    """Circle through the (transversely shifted) anchor with given tangent/curvature."""
    nx = -math.sin(psi)
    ny = math.cos(psi)
    px = anchor_xy[0] + delta * nx
    py = anchor_xy[1] + delta * ny
    radius = 1.0 / abs(kappa_signed)
    sign = 1.0 if kappa_signed >= 0.0 else -1.0
    cx = px + sign * radius * nx
    cy = py + sign * radius * ny
    return cx, cy, radius


def forward_refine(
    data_sub: np.ndarray,
    invalid_mask: np.ndarray,
    mask: np.ndarray,
    pixel_offsets_x: np.ndarray,
    pixel_offsets_y: np.ndarray,
    *,
    anchor_xy_offsets: tuple[float, float],
    psi0: float,
    kappa0: float,
    side0: int,
    width0_arcsec: float,
    ridge_xy_offsets: tuple[np.ndarray, np.ndarray],
    sky_sigma: float,
    psf_fwhm_arcsec: float,
    pixel_scale_arcsec: float,
    sigma_psi0: float,
    sigma_kappa0: float,
    max_slices: int = 12,
) -> ForwardRefineResult:
    """Refine (tangent angle, curvature) by fitting pixels with a PSF-convolved ribbon.

    ``pixel_offsets_x/y`` give the solver-offsets coordinates (arcsec) of every
    cutout pixel; the fit region is the segmentation mask dilated by ~2 PSF.
    """
    if kappa0 <= 0.0 or side0 == 0:
        return _failed("Forward refinement requires a circle (non-zero curvature) initial fit.")
    kappa_signed0 = float(side0) * float(kappa0)

    psf_sigma_pix = float(psf_fwhm_arcsec) / (2.0 * math.sqrt(2.0 * math.log(2.0))) / float(pixel_scale_arcsec)
    dilate_pix = max(int(round(2.0 * psf_fwhm_arcsec / pixel_scale_arcsec)), 2)
    region = ndimage.binary_dilation(mask, iterations=dilate_pix) & ~invalid_mask
    rows, cols = np.nonzero(region)
    if rows.size < 50:
        return _failed("Too few pixels in the refinement region.")
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    pad = dilate_pix + int(math.ceil(3.0 * psf_sigma_pix))
    r0 = max(r0 - pad, 0)
    c0 = max(c0 - pad, 0)
    r1 = min(r1 + pad, data_sub.shape[0])
    c1 = min(c1 + pad, data_sub.shape[1])

    data_box = data_sub[r0:r1, c0:c1]
    region_box = region[r0:r1, c0:c1]
    x_off_box = pixel_offsets_x[r0:r1, c0:c1]
    y_off_box = pixel_offsets_y[r0:r1, c0:c1]

    # Fixed slice assignment from the base ridge: every pixel belongs to the
    # slice of its nearest ridge point (longitudinally), independent of the
    # fit parameters.
    ridge_x, ridge_y = ridge_xy_offsets
    n_ridge = ridge_x.size
    n_slices = int(np.clip(n_ridge, 3, max_slices))
    slice_of_ridge = np.minimum((np.arange(n_ridge) * n_slices) // max(n_ridge, 1), n_slices - 1)
    flat_x = x_off_box.ravel()
    flat_y = y_off_box.ravel()
    d2 = (flat_x[:, None] - ridge_x[None, :]) ** 2 + (flat_y[:, None] - ridge_y[None, :]) ** 2
    nearest_ridge = np.argmin(d2, axis=1)
    slice_index_box = slice_of_ridge[nearest_ridge].reshape(x_off_box.shape)

    region_flat = region_box.ravel()
    data_flat = data_box.ravel()[region_flat]
    sigma = float(sky_sigma)

    def model_basis(theta: np.ndarray) -> np.ndarray | None:
        psi, kappa_signed, delta, log_w = (float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3]))
        if abs(kappa_signed) < 1.0e-6 or abs(kappa_signed) > 10.0:
            return None
        w = math.exp(log_w)
        if not (0.005 <= w <= 5.0):
            return None
        cx, cy, radius = _circle_center(anchor_xy_offsets, psi, kappa_signed, delta)
        dist = np.abs(np.hypot(x_off_box - cx, y_off_box - cy) - radius)
        profile = np.exp(-0.5 * (dist / w) ** 2)
        basis = np.empty((int(np.count_nonzero(region_flat)), n_slices), dtype=float)
        for j in range(n_slices):
            component = np.where(slice_index_box == j, profile, 0.0)
            component = ndimage.gaussian_filter(component, sigma=psf_sigma_pix)
            basis[:, j] = component.ravel()[region_flat]
        return basis

    def chi2(theta: np.ndarray) -> float:
        basis = model_basis(theta)
        if basis is None:
            return 1.0e30
        try:
            amplitudes, _ = optimize.nnls(basis, data_flat)
        except Exception:
            return 1.0e30
        residual = data_flat - basis @ amplitudes
        return float(np.sum((residual / sigma) ** 2))

    width_sigma0 = max(float(width0_arcsec) / (2.0 * math.sqrt(2.0 * math.log(2.0))), 0.02)
    theta0 = np.array([float(psi0), kappa_signed0, 0.0, math.log(width_sigma0)])
    steps = np.array(
        [
            max(float(sigma_psi0), 0.01),
            max(float(sigma_kappa0), 0.05 * abs(kappa_signed0)),
            0.05,
            0.2,
        ]
    )
    simplex = np.vstack([theta0] + [theta0 + steps * basis_vec for basis_vec in np.eye(4)])
    result = optimize.minimize(
        chi2,
        theta0,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "maxiter": 400, "xatol": 1e-5, "fatol": 1e-2},
    )
    theta_best = result.x
    chi2_best = float(result.fun)
    if not np.isfinite(chi2_best) or chi2_best >= 1.0e29:
        return _failed("Forward refinement optimization failed.")

    # Laplace covariance over (psi, kappa) at the optimum.
    steps_h = np.array([max(float(sigma_psi0), 0.01) / 3.0, max(float(sigma_kappa0), 1e-4) / 3.0])
    hessian = np.zeros((2, 2))
    f0 = chi2_best

    def chi2_at(dpsi: float, dkap: float) -> float:
        theta = theta_best.copy()
        theta[0] += dpsi
        theta[1] += dkap
        return chi2(theta)

    try:
        for i in range(2):
            e = np.zeros(2)
            e[i] = steps_h[i]
            hessian[i, i] = (chi2_at(*(e)) - 2.0 * f0 + chi2_at(*(-e))) / steps_h[i] ** 2
        cross = (
            chi2_at(steps_h[0], steps_h[1])
            - chi2_at(steps_h[0], -steps_h[1])
            - chi2_at(-steps_h[0], steps_h[1])
            + chi2_at(-steps_h[0], -steps_h[1])
        ) / (4.0 * steps_h[0] * steps_h[1])
        hessian[0, 1] = hessian[1, 0] = cross
        cov = np.linalg.inv(0.5 * hessian)
        sigma_psi = math.sqrt(max(cov[0, 0], 0.0))
        sigma_kappa = math.sqrt(max(cov[1, 1], 0.0))
        laplace_ok = (
            math.isfinite(sigma_psi) and math.isfinite(sigma_kappa) and sigma_psi > 0.0 and sigma_kappa > 0.0
        )
    except Exception:
        laplace_ok = False
        sigma_psi = float(sigma_psi0)
        sigma_kappa = float(sigma_kappa0)
    if not laplace_ok:
        sigma_psi = float(sigma_psi0)
        sigma_kappa = float(sigma_kappa0)

    kappa_signed = float(theta_best[1])
    reduced = chi2_best / max(data_flat.size - n_slices - 4, 1)
    return ForwardRefineResult(
        success=True,
        message="ok",
        tangent_angle_rad=float(np.mod(theta_best[0], math.pi)),
        curvature_arcsec_inv=abs(kappa_signed),
        curvature_side=int(np.sign(kappa_signed)),
        sigma_tangent_rad=float(sigma_psi),
        sigma_curvature_arcsec_inv=float(sigma_kappa),
        reduced_chi2=float(reduced),
    )


def _failed(message: str) -> ForwardRefineResult:
    return ForwardRefineResult(
        success=False,
        message=message,
        tangent_angle_rad=float("nan"),
        curvature_arcsec_inv=float("nan"),
        curvature_side=0,
        sigma_tangent_rad=float("nan"),
        sigma_curvature_arcsec_inv=float("nan"),
        reduced_chi2=float("nan"),
    )
