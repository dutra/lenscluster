import math

import numpy as np
import pytest

from arctrace import circlefit
from arctrace.geometry import axial_difference, wrap_axial


def _arc_points(xc, yc, r, theta0, theta1, n):
    theta = np.linspace(theta0, theta1, n)
    return xc + r * np.cos(theta), yc + r * np.sin(theta), theta


def test_taubin_exact_circle() -> None:
    x, y, _ = _arc_points(2.0, -1.0, 5.0, 0.2, 1.4, 25)
    fit = circlefit.taubin_circle_fit(x, y)
    assert not fit.is_line
    assert fit.xc == pytest.approx(2.0, abs=1e-9)
    assert fit.yc == pytest.approx(-1.0, abs=1e-9)
    assert fit.r == pytest.approx(5.0, abs=1e-9)
    assert fit.rms_residual == pytest.approx(0.0, abs=1e-9)


def test_taubin_noisy_circle_recovery() -> None:
    # Transverse noise representative of real ridge centroids (sigma/R ~ 0.1%):
    # the sagitta lever arm on a 60 deg span amplifies it by ~x20 on the radius.
    rng = np.random.default_rng(11)
    r_true = 10.0
    x0, y0, theta = _arc_points(0.0, 0.0, r_true, -0.5, 0.55, 40)  # ~60 deg span
    sigma = 0.001 * r_true
    biases = []
    for _ in range(50):
        eps = rng.normal(0.0, sigma, size=x0.size)
        fit = circlefit.taubin_circle_fit(x0 + eps * np.cos(theta), y0 + eps * np.sin(theta))
        biases.append(fit.r - r_true)
    assert abs(np.mean(biases)) < 0.005 * r_true
    assert np.std(biases) < 0.02 * r_true


def test_taubin_weights_downweight_outlier() -> None:
    x, y, _ = _arc_points(0.0, 0.0, 8.0, 0.0, 1.2, 20)
    x_out = np.append(x, 0.0)
    y_out = np.append(y, 0.0)  # gross outlier at the center
    weights = np.append(np.ones(20), 1.0e-9)
    fit = circlefit.taubin_circle_fit(x_out, y_out, weights)
    assert fit.r == pytest.approx(8.0, abs=1e-3)


def test_fit_ridge_circle_line_fallback() -> None:
    # Nearly collinear points whose sagitta is below the scatter.
    x = np.linspace(0.0, 4.0, 12)
    y = 0.5 * x + 0.0001 * (x - 2.0) ** 2
    sigmas = np.full_like(x, 0.05)
    fit = circlefit.fit_ridge_circle(x, y, sigmas, length=float(np.hypot(4.0, 2.0)))
    assert fit.is_line
    assert fit.line_angle_rad == pytest.approx(math.atan2(1.0, 2.0), abs=5e-3)
    angle, kappa, side = circlefit.tangent_and_curvature_at(fit, 2.0, 1.0)
    assert kappa == 0.0
    assert side == 0
    assert angle == pytest.approx(math.atan2(1.0, 2.0), abs=5e-3)


def test_tangent_and_curvature_at_anchor() -> None:
    x, y, _ = _arc_points(3.0, 4.0, 6.0, 0.1, 1.3, 30)
    fit = circlefit.taubin_circle_fit(x, y)
    # Anchor on the circle at azimuth 0.7.
    ax = 3.0 + 6.0 * math.cos(0.7)
    ay = 4.0 + 6.0 * math.sin(0.7)
    angle, kappa, side = circlefit.tangent_and_curvature_at(fit, ax, ay)
    assert kappa == pytest.approx(1.0 / 6.0, abs=1e-9)
    expected = wrap_axial(0.7 + 0.5 * math.pi)
    assert abs(axial_difference(angle, expected)) < 1e-9
    assert side != 0


def test_bootstrap_sigma_statistically_sane() -> None:
    rng = np.random.default_rng(5)
    r_true = 12.0
    x0, y0, theta = _arc_points(0.0, 0.0, r_true, 0.3, 1.1, 24)
    sigma = 0.08
    sigmas = np.full_like(x0, sigma)
    eps = rng.normal(0.0, sigma, size=x0.size)
    x = x0 + eps * np.cos(theta)
    y = y0 + eps * np.sin(theta)
    length = r_true * 0.8
    anchor_idx = x0.size // 2
    fit = circlefit.fit_ridge_circle(x, y, sigmas, length=length)
    angle, kappa, _ = circlefit.tangent_and_curvature_at(fit, x[anchor_idx], y[anchor_idx])
    sig_phi, sig_kappa = circlefit.bootstrap_geometry_sigma(
        x,
        y,
        sigmas,
        anchor_xy=(float(x[anchor_idx]), float(y[anchor_idx])),
        n_boot=300,
        rng=np.random.default_rng(99),
        length=length,
    )
    assert 0.0 < sig_phi < 0.5
    assert 0.0 < sig_kappa < 0.05
    true_angle = wrap_axial(theta[anchor_idx] + 0.5 * math.pi)
    assert abs(axial_difference(angle, true_angle)) < 4.0 * sig_phi + 0.02
    assert abs(kappa - 1.0 / r_true) < 4.0 * sig_kappa + 1e-4


def test_subsegment_consistency_constant_vs_varying_curvature() -> None:
    rng = np.random.default_rng(21)
    sigma = 0.02
    # Constant-curvature ridge.
    x_c, y_c, theta = _arc_points(0.0, 0.0, 10.0, 0.0, 1.2, 24)
    x_c = x_c + rng.normal(0.0, sigma, x_c.size)
    y_c = y_c + rng.normal(0.0, sigma, y_c.size)
    arclengths = 10.0 * (theta - theta[0])
    sigmas = np.full_like(x_c, sigma)
    k_scatter_const, _, n_seg = circlefit.subsegment_curvature_scatter(x_c, y_c, sigmas, arclengths)
    assert n_seg >= 2

    # Varying-curvature ridge: two tangent circular pieces (R=10 then R=5),
    # joined smoothly at theta=0.6 of the first circle.
    theta1 = np.linspace(0.0, 0.6, 12)
    x1 = 10.0 * np.cos(theta1)
    y1 = 10.0 * np.sin(theta1)
    join = np.array([10.0 * math.cos(0.6), 10.0 * math.sin(0.6)])
    radial = join / np.hypot(*join)
    center2 = join - 5.0 * radial
    theta2 = np.linspace(0.6, 1.6, 12)
    x2 = center2[0] + 5.0 * np.cos(theta2)
    y2 = center2[1] + 5.0 * np.sin(theta2)
    x_v = np.concatenate([x1, x2[1:]]) + rng.normal(0.0, sigma, 23)
    y_v = np.concatenate([y1, y2[1:]]) + rng.normal(0.0, sigma, 23)
    s_v = np.concatenate([10.0 * theta1, 10.0 * 0.6 + 5.0 * (theta2[1:] - 0.6)])
    sig_v = np.full_like(x_v, sigma)
    k_scatter_var, _, _ = circlefit.subsegment_curvature_scatter(x_v, y_v, sig_v, s_v)

    assert k_scatter_var > 5.0 * max(k_scatter_const, 1e-6)
    assert k_scatter_var > 0.02  # of order |1/5 - 1/10| / 2


def test_too_few_points_raises() -> None:
    with pytest.raises(ValueError):
        circlefit.taubin_circle_fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
