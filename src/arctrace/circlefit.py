"""Weighted circle fitting of arc ridges.

Why a circle: in the curved-arc basis (Birrer 2021, Sect. 2.3-2.4) the minimal
tangential arc model demands the curvature s_tan be constant along the
tangential direction, so the tangential eigenvector's integral curve - the
ridge of a thin arc - is a circle of radius r = 1/s_tan (his Fig. 1) centered
at theta_c = theta_0 - s_tan^-1 e_rad (his Eq. 28). The solver's CAB
likelihood consumes exactly the two parameters of that truncation (tangent
angle and curvature at an anchor), so a weighted circle fit to ridge points is
the maximum-likelihood estimator of precisely those quantities under precisely
that local model. The straight-line fallback is the s_tan -> 0 limit.

The algebraic fit is Taubin's method in the Newton form of Chernov (2010,
"Circular and linear regression", Sect. 5), generalized to weighted moments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircleFitResult:
    xc: float
    yc: float
    r: float
    rms_residual: float
    is_line: bool
    line_angle_rad: float | None
    n_points: int


def _weighted_moments(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    w = weights / np.sum(weights)
    cx = float(np.sum(w * x))
    cy = float(np.sum(w * y))
    u = x - cx
    v = y - cy
    z = u * u + v * v
    return {
        "cx": cx,
        "cy": cy,
        "Mxx": float(np.sum(w * u * u)),
        "Myy": float(np.sum(w * v * v)),
        "Mxy": float(np.sum(w * u * v)),
        "Mxz": float(np.sum(w * u * z)),
        "Myz": float(np.sum(w * v * z)),
        "Mzz": float(np.sum(w * z * z)),
    }


def taubin_circle_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
) -> CircleFitResult:
    """Weighted Taubin algebraic circle fit (Chernov's Newton form)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3:
        raise ValueError("Circle fit requires at least 3 points.")
    if weights is None:
        weights = np.ones(n, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0) or np.sum(weights) <= 0.0:
        raise ValueError("Circle fit weights must be finite, non-negative, and not all zero.")

    m = _weighted_moments(x, y, weights)
    Mxx, Myy, Mxy = m["Mxx"], m["Myy"], m["Mxy"]
    Mxz, Myz, Mzz = m["Mxz"], m["Myz"], m["Mzz"]
    Mz = Mxx + Myy
    cov_xy = Mxx * Myy - Mxy * Mxy
    var_z = Mzz - Mz * Mz

    a3 = 4.0 * Mz
    a2 = -3.0 * Mz * Mz - Mzz
    a1 = var_z * Mz + 4.0 * cov_xy * Mz - Mxz * Mxz - Myz * Myz
    a0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - var_z * cov_xy
    a22 = a2 + a2
    a33 = a3 + a3 + a3

    eta = 0.0
    poly = 1.0e300
    for _ in range(30):
        poly_old = poly
        poly = a0 + eta * (a1 + eta * (a2 + eta * a3))
        if not math.isfinite(poly) or abs(poly) > abs(poly_old):
            eta = 0.0
            break
        dpoly = a1 + eta * (a22 + eta * a33)
        if dpoly == 0.0 or not math.isfinite(dpoly):
            break
        eta_new = eta - poly / dpoly
        if not math.isfinite(eta_new) or eta_new < 0.0:
            break
        if abs(eta_new - eta) <= 1.0e-14 * max(abs(eta_new), 1.0):
            eta = eta_new
            break
        eta = eta_new

    det = eta * eta - eta * Mz + cov_xy
    if det == 0.0 or not math.isfinite(det):
        return _degenerate_line_result(x, y, weights)
    uc = (Mxz * (Myy - eta) - Myz * Mxy) / (2.0 * det)
    vc = (Myz * (Mxx - eta) - Mxz * Mxy) / (2.0 * det)
    r2 = uc * uc + vc * vc + Mz
    if r2 <= 0.0 or not math.isfinite(r2):
        return _degenerate_line_result(x, y, weights)
    r = math.sqrt(r2)
    xc = m["cx"] + uc
    yc = m["cy"] + vc

    radii = np.hypot(x - xc, y - yc)
    w_norm = weights / np.sum(weights)
    rms = float(np.sqrt(np.sum(w_norm * (radii - r) ** 2)))
    return CircleFitResult(
        xc=float(xc),
        yc=float(yc),
        r=float(r),
        rms_residual=rms,
        is_line=False,
        line_angle_rad=None,
        n_points=int(n),
    )


def fit_line_axial(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Axial direction (mod pi) of the weighted principal axis of the points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if weights is None:
        weights = np.ones_like(x)
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    cx = float(np.sum(w * x))
    cy = float(np.sum(w * y))
    sxx = float(np.sum(w * (x - cx) ** 2))
    syy = float(np.sum(w * (y - cy) ** 2))
    sxy = float(np.sum(w * (x - cx) * (y - cy)))
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return float(np.mod(angle, math.pi))


def _degenerate_line_result(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> CircleFitResult:
    angle = fit_line_axial(x, y, weights)
    normal = np.array([-math.sin(angle), math.cos(angle)])
    w = weights / np.sum(weights)
    cx = float(np.sum(w * x))
    cy = float(np.sum(w * y))
    distances = (x - cx) * normal[0] + (y - cy) * normal[1]
    rms = float(np.sqrt(np.sum(w * distances**2)))
    return CircleFitResult(
        xc=float("nan"),
        yc=float("nan"),
        r=float("inf"),
        rms_residual=rms,
        is_line=True,
        line_angle_rad=angle,
        n_points=int(np.size(x)),
    )


def fit_ridge_circle(
    x: np.ndarray,
    y: np.ndarray,
    sigmas: np.ndarray,
    *,
    length: float,
    straightness_sagitta_snr: float = 1.0,
) -> CircleFitResult:
    """Fit a circle to ridge points; fall back to a line for near-straight arcs.

    The sagitta of a chord of length L on a circle of radius r is L^2 / (8 r).
    When it falls below the expected transverse point scatter
    (``straightness_sagitta_snr * median(sigma) / sqrt(N)``), curvature is not
    measurable and the s_tan -> 0 line limit is returned.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    if np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("Ridge point sigmas must be finite and positive.")
    weights = 1.0 / sigmas**2
    fit = taubin_circle_fit(x, y, weights)
    if fit.is_line:
        return fit
    sagitta = float(length) ** 2 / (8.0 * fit.r)
    scatter = float(straightness_sagitta_snr) * float(np.median(sigmas)) / math.sqrt(max(x.size, 1))
    if sagitta < scatter:
        return _degenerate_line_result(x, y, weights)
    return fit


def tangent_and_curvature_at(
    fit: CircleFitResult,
    x0: float,
    y0: float,
) -> tuple[float, float, int]:
    """Axial tangent angle, |curvature| and curvature-center side at a point.

    The side is the sign of the cross product t x (c - p) for the axial
    tangent representative in [0, pi): +1 means the curvature center lies to
    the left of that representative direction, -1 to the right, 0 for the line
    fallback.
    """
    if fit.is_line:
        assert fit.line_angle_rad is not None
        return float(fit.line_angle_rad), 0.0, 0
    rx = float(x0) - fit.xc
    ry = float(y0) - fit.yc
    angle = float(np.mod(math.atan2(rx, -ry), math.pi))  # atan2(ry, rx) + pi/2, axial
    tx = math.cos(angle)
    ty = math.sin(angle)
    cross = tx * (fit.yc - float(y0)) - ty * (fit.xc - float(x0))
    side = int(np.sign(cross)) if cross != 0.0 else 0
    return angle, 1.0 / fit.r, side


def bootstrap_geometry_sigma(
    x: np.ndarray,
    y: np.ndarray,
    sigmas: np.ndarray,
    *,
    anchor_xy: tuple[float, float],
    n_boot: int,
    rng: np.random.Generator,
    length: float,
    straightness_sagitta_snr: float = 1.0,
) -> tuple[float, float]:
    """Bootstrap (sigma_tangent_axial, sigma_curvature) at a fixed anchor.

    Each replicate perturbs the points along their local transverse direction
    (radial for a circle, normal for a line) by N(0, sigma_i) and refits. The
    axial tangent dispersion is the circular std of the doubled angle, halved;
    the curvature dispersion is a robust MAD-based std, with line-fallback
    replicates contributing curvature 0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    base = fit_ridge_circle(x, y, sigmas, length=length, straightness_sagitta_snr=straightness_sagitta_snr)
    if base.is_line:
        assert base.line_angle_rad is not None
        nx = -math.sin(base.line_angle_rad)
        ny = math.cos(base.line_angle_rad)
        ux = np.full_like(x, nx)
        uy = np.full_like(y, ny)
    else:
        dx = x - base.xc
        dy = y - base.yc
        norm = np.hypot(dx, dy)
        norm[norm == 0.0] = 1.0
        ux = dx / norm
        uy = dy / norm

    angles = np.empty(int(n_boot), dtype=float)
    kappas = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        eps = rng.normal(0.0, sigmas)
        fit = fit_ridge_circle(
            x + eps * ux,
            y + eps * uy,
            sigmas,
            length=length,
            straightness_sagitta_snr=straightness_sagitta_snr,
        )
        angle, kappa, _ = tangent_and_curvature_at(fit, anchor_xy[0], anchor_xy[1])
        angles[i] = angle
        kappas[i] = kappa

    mean_vector = abs(np.mean(np.exp(2.0j * angles)))
    mean_vector = min(max(mean_vector, 1.0e-12), 1.0)
    sigma_tangent = 0.5 * math.sqrt(max(-2.0 * math.log(mean_vector), 0.0))
    mad = float(np.median(np.abs(kappas - np.median(kappas))))
    sigma_curvature = 1.4826 * mad
    if sigma_curvature == 0.0:
        sigma_curvature = float(np.std(kappas))
    return float(sigma_tangent), float(sigma_curvature)


def subsegment_curvature_scatter(
    x: np.ndarray,
    y: np.ndarray,
    sigmas: np.ndarray,
    arclengths: np.ndarray,
    *,
    straightness_sagitta_snr: float = 1.0,
) -> tuple[float, float, int]:
    """Constant-curvature validity check (Birrer 2021, Sect. 5.3).

    Splits the ordered ridge into contiguous halves (plus thirds when there
    are >= 12 points), refits each sub-segment, and returns
    (curvature scatter, center scatter, number of sub-segments). Line-fallback
    sub-segments contribute curvature 0; centers are compared only among
    circle fits. A large curvature scatter indicates higher-order differentials
    (d s_tan / d e_tan != 0) that the single-circle model cannot absorb.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    arclengths = np.asarray(arclengths, dtype=float)
    order = np.argsort(arclengths)
    x, y, sigmas, arclengths = x[order], y[order], sigmas[order], arclengths[order]
    n = x.size

    splits: list[np.ndarray] = []
    if n >= 6:
        half = n // 2
        splits.extend([np.arange(0, half), np.arange(half, n)])
    if n >= 12:
        third = n // 3
        splits.extend([np.arange(0, third), np.arange(third, 2 * third), np.arange(2 * third, n)])
    if not splits:
        return 0.0, 0.0, 0

    kappas: list[float] = []
    centers: list[tuple[float, float]] = []
    for idx in splits:
        if idx.size < 3:
            continue
        seg_length = float(arclengths[idx][-1] - arclengths[idx][0])
        if seg_length <= 0.0:
            continue
        fit = fit_ridge_circle(
            x[idx],
            y[idx],
            sigmas[idx],
            length=seg_length,
            straightness_sagitta_snr=straightness_sagitta_snr,
        )
        if fit.is_line:
            kappas.append(0.0)
        else:
            kappas.append(1.0 / fit.r)
            centers.append((fit.xc, fit.yc))

    if len(kappas) < 2:
        return 0.0, 0.0, len(kappas)
    kappa_scatter = float(np.std(np.asarray(kappas), ddof=1))
    center_scatter = 0.0
    if len(centers) >= 2:
        centers_arr = np.asarray(centers)
        center_scatter = float(np.sqrt(np.mean(np.sum((centers_arr - centers_arr.mean(axis=0)) ** 2, axis=1))))
    return kappa_scatter, center_scatter, len(kappas)
