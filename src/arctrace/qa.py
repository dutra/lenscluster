"""QA overlay rendering: cutout + mask contour + ridge + fitted circle + tangent.

Uses the lenscluster calibrated Lupton RGB recipe when at least three display
bands are available; falls back to a grayscale asinh stretch of the reference
band otherwise. All overlays are drawn through the WCS, so they are correct
for rotated or unflipped mosaics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np
from astropy.visualization import simple_norm
from skimage import measure as sk_measure

from lenscluster.rgb import (
    CALIBRATED_RGB_MINIMUM_SKY_SIGMA,
    CalibratedRGBDisplayConfig,
    compute_fnu_band_fluxscales,
    make_calibrated_rgb,
    trim_to_common_shape,
)

from . import geometry
from .measure import ArcMeasurement, BandArtifacts
from .mosaics import BandMosaic, CutoutData, extract_cutout


def extract_display_cutouts(
    mosaics: Mapping[str, BandMosaic],
    center,
    size_arcsec: float,
) -> dict[str, CutoutData]:
    cutouts: dict[str, CutoutData] = {}
    for band, mosaic in mosaics.items():
        try:
            cutouts[band] = extract_cutout(mosaic, center, size_arcsec)
        except Exception:
            continue
    return cutouts


def _rgb_band_order(bands: list[str]) -> list[str]:
    def sort_key(band: str) -> float:
        digits = "".join(ch for ch in band if ch.isdigit())
        return float(digits) if digits else 0.0

    return sorted(bands, key=sort_key)


def render_display_image(
    cutouts: Mapping[str, CutoutData],
    reference_band: str,
) -> tuple[np.ndarray, str, bool]:
    """(image, displayed-band description, is_rgb) for imshow(origin='lower')."""
    usable = {
        band: cutout
        for band, cutout in cutouts.items()
        if np.isfinite(cutout.data).any()
    }
    reference = usable.get(reference_band) or next(iter(usable.values()), None)
    if reference is None:
        raise ValueError("No usable display cutouts.")
    scales = {band: cutout.pixel_scale_arcsec for band, cutout in usable.items()}
    same_grid = all(abs(s - reference.pixel_scale_arcsec) < 1e-3 for s in scales.values())
    if len(usable) >= 3 and same_grid:
        ordered = _rgb_band_order(list(usable))
        blue, green, red = ordered[0], ordered[len(ordered) // 2], ordered[-1]
        bands = [blue, green, red]
        fluxscales = compute_fnu_band_fluxscales(
            {b: usable[b].photflam for b in bands},
            {b: usable[b].photplam for b in bands},
            reference_band=reference_band if reference_band in bands else bands[1],
        )
        backgrounds = {b: float(np.nanmedian(usable[b].data)) for b in bands}
        sky_sigmas = [
            usable[b].global_background_sigma * fluxscales.get(b, 1.0) for b in bands
        ]
        display = CalibratedRGBDisplayConfig(
            band_backgrounds=backgrounds,
            band_fluxscales=fluxscales,
            minimum=CALIBRATED_RGB_MINIMUM_SKY_SIGMA * float(min(sky_sigmas)),
        )
        arrays = trim_to_common_shape([usable[b].data for b in bands])
        cutouts_by_band = {b: np.nan_to_num(a, nan=0.0) for b, a in zip(bands, arrays)}
        rgb = make_calibrated_rgb(cutouts_by_band, bands=bands, display=display)
        return rgb, "+".join(bands), True
    data = np.nan_to_num(reference.data, nan=0.0)
    norm = simple_norm(data, "asinh", percent=99.5)
    return norm(data), reference.band, False


def arc_summary_lines(measurement: ArcMeasurement, band_desc: str | None = None) -> list[str]:
    """Human-readable summary of a measurement, shared by the QA PNG and the
    interactive overlay."""
    label = measurement.label or "(unlabeled)"
    head = f"{label}  [{band_desc}]" if band_desc else label
    lines = [
        head,
        rf"$\varphi$ = {measurement.tangent_angle_offset_rad:.4f} $\pm$ {measurement.sigma_tangent_rad:.4f} rad",
        rf"$\kappa$ = {measurement.curvature_arcsec_inv:.4f} $\pm$ {measurement.sigma_curvature_arcsec_inv:.4f} /arcsec",
    ]
    reference = next((b for b in measurement.bands if b.band == measurement.reference_band), None)
    if reference is not None and reference.success:
        radius = "inf" if reference.radius_arcsec is None else f"{reference.radius_arcsec:.1f}\""
        lines.append(
            f"R = {radius}  L = {reference.length_arcsec:.2f}\"  W = {reference.width_arcsec:.2f}\"  "
            f"q = {reference.axis_ratio:.2f}"
        )
        if reference.is_line_fallback:
            lines.append("line fallback (curvature unresolved)")
    lines.append(f"reliability = {measurement.reliability:.2f}")
    for warning in measurement.warnings[:2]:
        lines.append(f"! {warning[:70]}")
    return lines


def render_qa_figure(
    measurement: ArcMeasurement,
    artifacts: BandArtifacts,
    display_cutouts: Mapping[str, CutoutData] | None = None,
    *,
    fig=None,
    ax=None,
):
    import matplotlib.pyplot as plt

    cutout = artifacts.cutout
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    ax.clear()

    cutouts = dict(display_cutouts or {})
    cutouts.setdefault(cutout.band, cutout)
    try:
        image, band_desc, is_rgb = render_display_image(cutouts, cutout.band)
    except ValueError:
        image, band_desc, is_rgb = np.zeros_like(cutout.data), cutout.band, False
    cmap = None if is_rgb else "gray"
    ax.imshow(image, origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_xlim(-0.5, cutout.data.shape[1] - 0.5)
    ax.set_ylim(-0.5, cutout.data.shape[0] - 0.5)

    # Mask boundary.
    for contour in sk_measure.find_contours(artifacts.segmentation.mask.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color="#00e5ff", linewidth=0.9, alpha=0.9)

    # Ridge points.
    ridge = artifacts.ridge
    snr = np.array([p.snr for p in ridge.points])
    sizes = 8.0 + 20.0 * np.clip(snr / max(snr.max(), 1.0), 0.0, 1.0)
    ax.scatter(ridge.x_pix, ridge.y_pix, s=sizes, facecolors="none", edgecolors="#ffd54f", linewidths=0.9)

    ra0, dec0 = artifacts.ra0_deg, artifacts.dec0_deg
    fit = artifacts.fit
    if not fit.is_line and math.isfinite(fit.r):
        theta = np.linspace(0.0, 2.0 * math.pi, 720)
        circle_x = fit.xc + fit.r * np.cos(theta)
        circle_y = fit.yc + fit.r * np.sin(theta)
        px, py = geometry.offset_frame_point_to_pixel(cutout.wcs, circle_x, circle_y, ra0, dec0)
        inside = (px > -0.5) & (px < cutout.data.shape[1] - 0.5) & (py > -0.5) & (py < cutout.data.shape[0] - 0.5)
        ax.plot(np.where(inside, px, np.nan), np.where(inside, py, np.nan), color="#ff5252", linewidth=1.2)

    # Tangent segment at the anchor (in the offsets frame, mapped through WCS).
    anchor_x, anchor_y = artifacts.anchor_offsets
    phi = measurement.tangent_angle_offset_rad
    if math.isfinite(phi):
        half = 1.2  # arcsec
        seg_x = np.array([anchor_x - half * math.cos(phi), anchor_x + half * math.cos(phi)])
        seg_y = np.array([anchor_y - half * math.sin(phi), anchor_y + half * math.sin(phi)])
        px, py = geometry.offset_frame_point_to_pixel(cutout.wcs, seg_x, seg_y, ra0, dec0)
        ax.plot(px, py, color="#69f0ae", linewidth=2.0)
    apx, apy = geometry.offset_frame_point_to_pixel(cutout.wcs, anchor_x, anchor_y, ra0, dec0)
    ax.plot(float(apx), float(apy), marker="+", color="#69f0ae", markersize=12, markeredgewidth=2.0)
    # Seed marker (the offsets frame origin is the seed).
    spx, spy = geometry.offset_frame_point_to_pixel(cutout.wcs, 0.0, 0.0, ra0, dec0)
    ax.plot(float(spx), float(spy), marker="x", color="#ff4081", markersize=10, markeredgewidth=1.6)

    lines = arc_summary_lines(measurement, band_desc)
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "boxstyle": "round,pad=0.35"},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def save_qa_png(
    measurement: ArcMeasurement,
    artifacts: BandArtifacts,
    out_path: str | Path,
    display_cutouts: Mapping[str, CutoutData] | None = None,
    *,
    dpi: int = 150,
) -> Path:
    import matplotlib.pyplot as plt

    fig = render_qa_figure(measurement, artifacts, display_cutouts)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
