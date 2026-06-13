"""Arc measurement orchestrator: seed -> curved-arc-basis observables.

Per band: cutout -> background -> seeded segmentation -> two-pass ridge trace
-> weighted circle fit in the solver offsets frame (anchored at the seed) ->
tangent angle + curvature at the mid-arc anchor, with bootstrap statistical
sigmas, segmentation-variant and sub-segment systematic sigmas, and intrinsic
shape-noise floors. See circlefit.py for why the circle is exactly the
curved-arc-basis measurement model.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord

from . import geometry
from .circlefit import (
    CircleFitResult,
    bootstrap_geometry_sigma,
    fit_ridge_circle,
    subsegment_curvature_scatter,
    tangent_and_curvature_at,
)
from .config import ArcMeasureConfig, SegmentationConfig
from .errors import ArctraceError
from .mosaics import BandMosaic, CutoutData, extract_cutout
from .refine import forward_refine
from .ridge import RidgeTrace, resample_ridge_polar, trace_ridge_moments
from .segmentation import ArcSegmentation, BackgroundResult, segment_arc, subtract_background

_FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))


@dataclass(frozen=True)
class BandArtifacts:
    """Intermediate products kept for QA rendering."""

    cutout: CutoutData
    background: BackgroundResult
    segmentation: ArcSegmentation
    ridge: RidgeTrace
    fit: CircleFitResult
    ra0_deg: float
    dec0_deg: float
    fit_x_offsets: np.ndarray
    fit_y_offsets: np.ndarray
    anchor_offsets: tuple[float, float]


@dataclass(frozen=True)
class BandArcMeasurement:
    band: str
    success: bool
    failure_reason: str | None
    tangent_angle_offset_rad: float = float("nan")
    curvature_arcsec_inv: float = float("nan")
    sigma_tangent_stat_rad: float = float("nan")
    sigma_curvature_stat: float = float("nan")
    sigma_tangent_seg_rad: float = float("nan")
    sigma_curvature_seg: float = float("nan")
    anchor_ra_deg: float = float("nan")
    anchor_dec_deg: float = float("nan")
    center_ra_deg: float | None = None
    center_dec_deg: float | None = None
    radius_arcsec: float | None = None
    curvature_side: int = 0
    length_arcsec: float = float("nan")
    width_arcsec: float = float("nan")
    axis_ratio: float = float("nan")
    n_ridge_points: int = 0
    fit_rms_arcsec: float = float("nan")
    is_line_fallback: bool = False
    bridged_gap: bool = False
    touches_edge: bool = False
    masked_invalid_fraction: float = float("nan")
    contested_fraction: float = float("nan")
    effective_threshold_sigma: float = float("nan")
    cutout_size_used_arcsec: float = float("nan")
    kappa_subsegment_scatter: float = float("nan")
    center_subsegment_scatter_arcsec: float = float("nan")
    refine_applied: bool = False
    refine_reduced_chi2: float = float("nan")
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArcMeasurement:
    label: str | None
    seed_ra_deg: float
    seed_dec_deg: float
    success: bool
    failure_reason: str | None
    anchor_ra_deg: float = float("nan")
    anchor_dec_deg: float = float("nan")
    tangent_angle_offset_rad: float = float("nan")
    curvature_arcsec_inv: float = float("nan")
    sigma_tangent_rad: float = float("nan")
    sigma_curvature_arcsec_inv: float = float("nan")
    reliability: float = float("nan")
    reliability_overridden: bool = False
    reference_band: str = ""
    bands: tuple[BandArcMeasurement, ...] = ()
    multiband_consistent: bool | None = None
    multiband_chi2: float | None = None
    sigma_tangent_floor_rad: float = float("nan")
    warnings: tuple[str, ...] = ()
    config_summary: dict[str, Any] = field(default_factory=dict)


def tangent_sigma_floor(axis_ratio: float, sigma_e: float, cap: float) -> float:
    """Intrinsic shape-noise floor on the tangent angle (Birrer 2021, Sect. 3.3).

    sigma_phi ~ sigma_e (1 - g^2) / (2 g), g = (1 - q) / (1 + q): the observed
    orientation of a weakly stretched image is contaminated by the unknown
    intrinsic source orientation; strong stretching suppresses it.
    """
    q = float(np.clip(axis_ratio, 1.0e-6, 1.0))
    g = (1.0 - q) / (1.0 + q)
    if g <= 1.0e-6:
        return float(cap)
    floor = float(sigma_e) * (1.0 - g * g) / (2.0 * g)
    return float(np.clip(floor, 0.0, cap))


def _config_summary(cfg: ArcMeasureConfig) -> dict[str, Any]:
    summary = dataclasses.asdict(cfg)
    summary["segmentation"] = dataclasses.asdict(cfg.segmentation)
    summary["psf_fwhm_arcsec"] = dict(cfg.psf_fwhm_arcsec)
    summary["measure_bands"] = list(cfg.measure_bands)
    summary["segmentation_variants"] = [list(v) for v in cfg.segmentation_variants]
    return summary


def _ridge_to_offsets(
    cutout: CutoutData,
    trace: RidgeTrace,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_off, y_off = geometry.pixel_points_to_offset_frame(
        cutout.wcs, trace.x_pix, trace.y_pix, ra0_deg, dec0_deg
    )
    sigmas = trace.sigma_pix * cutout.pixel_scale_arcsec
    arclengths = trace.arclength_pix * cutout.pixel_scale_arcsec
    return np.asarray(x_off), np.asarray(y_off), sigmas, arclengths


def _select_fit_window(
    x: np.ndarray,
    y: np.ndarray,
    sigmas: np.ndarray,
    arclengths: np.ndarray,
    anchor_index: int,
    halfspan: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if halfspan is None:
        return x, y, sigmas, arclengths
    keep = np.abs(arclengths - arclengths[anchor_index]) <= float(halfspan)
    if np.count_nonzero(keep) < 3:
        return x, y, sigmas, arclengths
    return x[keep], y[keep], sigmas[keep], arclengths[keep]


def _trace_and_fit(
    cutout: CutoutData,
    bg: BackgroundResult,
    segmentation: ArcSegmentation,
    cfg: ArcMeasureConfig,
    band: str,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[RidgeTrace, CircleFitResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Two-pass ridge trace and circle fit in the offsets frame.

    Returns (trace, fit, x_off, y_off, sigmas, arclengths, anchor_index) where
    the arrays are the (possibly windowed) points used in the final fit and
    anchor_index points into them.
    """
    psf_fwhm = cfg.psf_fwhm(band)
    slice_width_pix = max(psf_fwhm * cfg.slice_width_psf_factor / cutout.pixel_scale_arcsec, 1.0)
    trace = trace_ridge_moments(
        bg.data_sub,
        segmentation.mask,
        slice_width_pix=slice_width_pix,
        sky_sigma=bg.sky_sigma,
        min_snr=cfg.min_slice_snr,
        pixel_scale_arcsec=cutout.pixel_scale_arcsec,
    )
    if len(trace.points) < int(cfg.min_ridge_points):
        raise ArctraceError(
            f"Only {len(trace.points)} ridge points (< {cfg.min_ridge_points}) in band {band}."
        )
    x_off, y_off, sigmas, arclengths = _ridge_to_offsets(cutout, trace, ra0_deg, dec0_deg)
    length = float(arclengths[-1] - arclengths[0])
    fit = fit_ridge_circle(
        x_off,
        y_off,
        sigmas,
        length=length,
        straightness_sagitta_snr=cfg.straightness_sagitta_snr,
    )

    # Pass 2: polar re-slicing around the fitted center (radial/tangential
    # eigenframe), only when the curvature is meaningfully constrained.
    if not fit.is_line and fit.r < 4.0 * cfg.max_cutout_size_arcsec:
        center_x_pix, center_y_pix = geometry.offset_frame_point_to_pixel(
            cutout.wcs, fit.xc, fit.yc, ra0_deg, dec0_deg
        )
        try:
            polar = resample_ridge_polar(
                bg.data_sub,
                segmentation.mask,
                center_xy_pix=(float(center_x_pix), float(center_y_pix)),
                slice_width_pix=slice_width_pix,
                sky_sigma=bg.sky_sigma,
                min_snr=cfg.min_slice_snr,
                pixel_scale_arcsec=cutout.pixel_scale_arcsec,
            )
        except ArctraceError:
            polar = None
        if polar is not None and len(polar.points) >= max(len(trace.points) // 2, 3):
            x2, y2, s2, a2 = _ridge_to_offsets(cutout, polar, ra0_deg, dec0_deg)
            length2 = float(a2[-1] - a2[0])
            fit2 = fit_ridge_circle(
                x2,
                y2,
                s2,
                length=length2,
                straightness_sagitta_snr=cfg.straightness_sagitta_snr,
            )
            trace, fit = polar, fit2
            x_off, y_off, sigmas, arclengths = x2, y2, s2, a2
            length = length2

    anchor_index = int(np.argmin(np.abs(arclengths - 0.5 * (arclengths[0] + arclengths[-1]))))
    anchor_arclength = float(arclengths[anchor_index])
    x_w, y_w, s_w, a_w = _select_fit_window(
        x_off, y_off, sigmas, arclengths, anchor_index, cfg.fit_halfspan_arcsec
    )
    if x_w.size != x_off.size:
        fit = fit_ridge_circle(
            x_w,
            y_w,
            s_w,
            length=float(a_w[-1] - a_w[0]),
            straightness_sagitta_snr=cfg.straightness_sagitta_snr,
        )
        anchor_index = int(np.argmin(np.abs(a_w - anchor_arclength)))
        x_off, y_off, sigmas, arclengths = x_w, y_w, s_w, a_w
    return trace, fit, x_off, y_off, sigmas, arclengths, anchor_index


def measure_band(
    mosaic: BandMosaic,
    seed: SkyCoord,
    cfg: ArcMeasureConfig,
    *,
    rng: np.random.Generator,
) -> tuple[BandArcMeasurement, BandArtifacts | None]:
    band = mosaic.band
    ra0_deg = float(seed.icrs.ra.deg)
    dec0_deg = float(seed.icrs.dec.deg)
    warnings: list[str] = []
    try:
        # Cutout with auto-enlargement while the mask touches the edge.
        size = float(cfg.cutout_size_arcsec)
        cutout = bg = segmentation = None
        for _ in range(4):
            cutout = extract_cutout(mosaic, seed, size)
            bg = subtract_background(cutout, cfg.segmentation)
            seed_x, seed_y = cutout.wcs.world_to_pixel(seed)
            segmentation = segment_arc(
                bg,
                (float(seed_x), float(seed_y)),
                cfg.segmentation,
                psf_fwhm_arcsec=cfg.psf_fwhm(band),
                pixel_scale_arcsec=cutout.pixel_scale_arcsec,
            )
            if not segmentation.touches_edge or size >= cfg.max_cutout_size_arcsec:
                break
            size = min(size * 1.5, float(cfg.max_cutout_size_arcsec))
        assert cutout is not None and bg is not None and segmentation is not None
        if segmentation.touches_edge:
            warnings.append("arc mask touches the cutout edge at the maximum cutout size")

        trace, fit, x_off, y_off, sigmas, arclengths, anchor_index = _trace_and_fit(
            cutout, bg, segmentation, cfg, band, ra0_deg, dec0_deg
        )
        anchor_xy = (float(x_off[anchor_index]), float(y_off[anchor_index]))
        angle, kappa, side = tangent_and_curvature_at(fit, anchor_xy[0], anchor_xy[1])

        sig_phi_stat, sig_kappa_stat = bootstrap_geometry_sigma(
            x_off,
            y_off,
            sigmas,
            anchor_xy=anchor_xy,
            n_boot=int(cfg.n_bootstrap),
            rng=rng,
            length=float(arclengths[-1] - arclengths[0]),
            straightness_sagitta_snr=cfg.straightness_sagitta_snr,
        )

        # Segmentation-variant systematic scatter, evaluated at the base anchor.
        variant_angles = [angle]
        variant_kappas = [kappa]
        for k_mult, gap_mult in cfg.segmentation_variants:
            seg_cfg = replace(
                cfg.segmentation,
                threshold_sigma=cfg.segmentation.threshold_sigma * float(k_mult),
                max_bridge_gap_arcsec=cfg.segmentation.max_bridge_gap_arcsec * float(gap_mult),
            )
            try:
                seed_x, seed_y = cutout.wcs.world_to_pixel(seed)
                seg_v = segment_arc(
                    bg,
                    (float(seed_x), float(seed_y)),
                    seg_cfg,
                    psf_fwhm_arcsec=cfg.psf_fwhm(band),
                    pixel_scale_arcsec=cutout.pixel_scale_arcsec,
                )
                _, fit_v, *_ = _trace_and_fit(cutout, bg, seg_v, cfg, band, ra0_deg, dec0_deg)
                angle_v, kappa_v, _ = tangent_and_curvature_at(fit_v, anchor_xy[0], anchor_xy[1])
                variant_angles.append(angle_v)
                variant_kappas.append(kappa_v)
            except ArctraceError:
                continue
        if len(variant_angles) >= 3:
            doubled = np.exp(2.0j * np.asarray(variant_angles))
            mean_vector = min(max(abs(np.mean(doubled)), 1.0e-12), 1.0)
            sig_phi_seg = 0.5 * math.sqrt(max(-2.0 * math.log(mean_vector), 0.0))
            sig_kappa_seg = float(np.std(np.asarray(variant_kappas), ddof=1))
        else:
            sig_phi_seg = 0.0
            sig_kappa_seg = 0.0
            warnings.append("fewer than 2 segmentation variants succeeded; systematic sigma not estimated")

        # Constant-curvature validity (Birrer 2021, Sect. 5.3).
        kappa_scatter, center_scatter, n_subseg = subsegment_curvature_scatter(
            x_off,
            y_off,
            sigmas,
            arclengths,
            straightness_sagitta_snr=cfg.straightness_sagitta_snr,
        )
        if n_subseg >= 2:
            sig_kappa_seg = math.hypot(sig_kappa_seg, kappa_scatter)
            # Flag only when the along-arc curvature variation is both
            # statistically significant AND a meaningful fraction of the
            # curvature itself, i.e. the single anchor value is unreliable.
            # Well-sampled ridges have tiny formal sigmas, so a pure
            # significance test alone fires on every real arc.
            significant = kappa_scatter > 3.0 * max(sig_kappa_stat, 1.0e-9)
            material = kappa_scatter > 0.3 * max(kappa, 1.0e-6)
            if significant and material:
                warnings.append(
                    "single-circle model strained: curvature varies along the arc "
                    f"(sub-segment scatter {kappa_scatter:.3g} vs kappa {kappa:.3g} arcsec^-1); "
                    "higher-order differentials present - consider --fit-halfspan-arcsec"
                )

        refine_applied = False
        refine_chi2 = float("nan")
        if cfg.refine == "forward" and not fit.is_line:
            yy, xx = np.mgrid[0 : cutout.data.shape[0], 0 : cutout.data.shape[1]]
            px_x, px_y = geometry.pixel_points_to_offset_frame(
                cutout.wcs, xx.ravel().astype(float), yy.ravel().astype(float), ra0_deg, dec0_deg
            )
            refine_result = forward_refine(
                bg.data_sub,
                bg.invalid_mask,
                segmentation.mask,
                px_x.reshape(cutout.data.shape),
                px_y.reshape(cutout.data.shape),
                anchor_xy_offsets=anchor_xy,
                psi0=angle,
                kappa0=kappa,
                side0=side,
                width0_arcsec=trace.width_arcsec,
                ridge_xy_offsets=(x_off, y_off),
                sky_sigma=bg.sky_sigma,
                psf_fwhm_arcsec=cfg.psf_fwhm(band),
                pixel_scale_arcsec=cutout.pixel_scale_arcsec,
                sigma_psi0=sig_phi_stat,
                sigma_kappa0=sig_kappa_stat,
            )
            if refine_result.success:
                angle = refine_result.tangent_angle_rad
                kappa = refine_result.curvature_arcsec_inv
                side = refine_result.curvature_side
                sig_phi_stat = refine_result.sigma_tangent_rad
                sig_kappa_stat = refine_result.sigma_curvature_arcsec_inv
                refine_applied = True
                refine_chi2 = refine_result.reduced_chi2
            else:
                warnings.append(f"forward refinement skipped: {refine_result.message}")

        anchor_x_pix, anchor_y_pix = geometry.offset_frame_point_to_pixel(
            cutout.wcs, anchor_xy[0], anchor_xy[1], ra0_deg, dec0_deg
        )
        anchor_world = cutout.wcs.pixel_to_world(float(anchor_x_pix), float(anchor_y_pix)).icrs
        if fit.is_line:
            center_ra = center_dec = None
            radius = None
        else:
            center_ra_arr, center_dec_arr = geometry.solver_offsets_to_radec(
                fit.xc, fit.yc, ra0_deg, dec0_deg
            )
            center_ra, center_dec = float(center_ra_arr), float(center_dec_arr)
            radius = float(fit.r)

        measurement = BandArcMeasurement(
            band=band,
            success=True,
            failure_reason=None,
            tangent_angle_offset_rad=float(geometry.wrap_axial(angle)),
            curvature_arcsec_inv=float(abs(kappa)),
            sigma_tangent_stat_rad=float(sig_phi_stat),
            sigma_curvature_stat=float(sig_kappa_stat),
            sigma_tangent_seg_rad=float(sig_phi_seg),
            sigma_curvature_seg=float(sig_kappa_seg),
            anchor_ra_deg=float(anchor_world.ra.deg),
            anchor_dec_deg=float(anchor_world.dec.deg),
            center_ra_deg=center_ra,
            center_dec_deg=center_dec,
            radius_arcsec=radius,
            curvature_side=int(side),
            length_arcsec=float(trace.length_arcsec),
            width_arcsec=float(trace.width_arcsec),
            axis_ratio=float(trace.axis_ratio),
            n_ridge_points=int(x_off.size),
            fit_rms_arcsec=float(fit.rms_residual),
            is_line_fallback=bool(fit.is_line),
            bridged_gap=bool(segmentation.bridged_gap),
            touches_edge=bool(segmentation.touches_edge),
            masked_invalid_fraction=float(segmentation.masked_invalid_fraction),
            contested_fraction=float(segmentation.contested_fraction),
            effective_threshold_sigma=float(segmentation.effective_threshold_sigma),
            cutout_size_used_arcsec=float(size),
            kappa_subsegment_scatter=float(kappa_scatter),
            center_subsegment_scatter_arcsec=float(center_scatter),
            refine_applied=refine_applied,
            refine_reduced_chi2=refine_chi2,
            warnings=tuple(warnings),
        )
        artifacts = BandArtifacts(
            cutout=cutout,
            background=bg,
            segmentation=segmentation,
            ridge=trace,
            fit=fit,
            ra0_deg=ra0_deg,
            dec0_deg=dec0_deg,
            fit_x_offsets=x_off,
            fit_y_offsets=y_off,
            anchor_offsets=anchor_xy,
        )
        return measurement, artifacts
    except ArctraceError as exc:
        return (
            BandArcMeasurement(band=band, success=False, failure_reason=str(exc), warnings=tuple(warnings)),
            None,
        )


def combine_band_measurements(
    band_results: tuple[BandArcMeasurement, ...],
    cfg: ArcMeasureConfig,
    *,
    label: str | None,
    seed: SkyCoord,
    reliability_override: float | None = None,
) -> ArcMeasurement:
    reference = next((b for b in band_results if b.band == cfg.reference_band), None)
    if reference is None and band_results:
        reference = band_results[0]
    seed_ra = float(seed.icrs.ra.deg)
    seed_dec = float(seed.icrs.dec.deg)
    if reference is None or not reference.success:
        reason = reference.failure_reason if reference is not None else "no bands measured"
        return ArcMeasurement(
            label=label,
            seed_ra_deg=seed_ra,
            seed_dec_deg=seed_dec,
            success=False,
            failure_reason=reason,
            reference_band=cfg.reference_band,
            bands=band_results,
            config_summary=_config_summary(cfg),
        )

    warnings = list(reference.warnings)
    floor = tangent_sigma_floor(reference.axis_ratio, cfg.sigma_e_floor, cfg.tangent_sigma_cap_rad)
    sigma_tangent = math.sqrt(
        reference.sigma_tangent_stat_rad**2
        + reference.sigma_tangent_seg_rad**2
        + floor**2
        + float(cfg.tangent_method_floor_rad) ** 2
    )
    curvature_floor = max(
        float(cfg.curvature_sigma_floor_arcsec_inv),
        float(cfg.curvature_method_floor_frac) * reference.curvature_arcsec_inv,
    )
    sigma_curvature = math.sqrt(
        reference.sigma_curvature_stat**2 + reference.sigma_curvature_seg**2 + curvature_floor**2
    )
    sigma_tangent = max(sigma_tangent, 1.0e-6)
    sigma_curvature = max(sigma_curvature, 1.0e-6)

    # Achromaticity cross-check between bands. Tangent angle and curvature are
    # checked separately: the tangent is the robust, primary MST-invariant
    # observable, while curvature is a fragile second-order lever-arm quantity
    # that legitimately differs between bands when color gradients shift the
    # flux-weighted ridge endpoints. Consistency (and the reliability bonus)
    # keys on the tangent; curvature disagreement is reported but does not by
    # itself condemn the measurement. Comparisons use combined
    # statistical+segmentation sigma; bootstrap-only sigma is unrealistically
    # small for a well-sampled ridge.
    others = [b for b in band_results if b.band != reference.band and b.success]
    multiband_consistent: bool | None = None
    multiband_chi2: float | None = None
    if others:

        def _band_sigma_phi(band: BandArcMeasurement) -> float:
            return math.sqrt(
                band.sigma_tangent_stat_rad**2
                + band.sigma_tangent_seg_rad**2
                + float(cfg.tangent_method_floor_rad) ** 2
            )

        def _band_sigma_kappa(band: BandArcMeasurement) -> float:
            frac_floor = float(cfg.curvature_method_floor_frac) * band.curvature_arcsec_inv
            return math.sqrt(band.sigma_curvature_stat**2 + band.sigma_curvature_seg**2 + frac_floor**2)

        tangent_chi2 = []
        curvature_chi2 = []
        for other in others:
            dphi = geometry.axial_difference(
                reference.tangent_angle_offset_rad, other.tangent_angle_offset_rad
            )
            var_phi = _band_sigma_phi(reference) ** 2 + _band_sigma_phi(other) ** 2
            tangent_chi2.append(float(dphi) ** 2 / max(var_phi, 1e-12))
            dkap = reference.curvature_arcsec_inv - other.curvature_arcsec_inv
            var_kap = _band_sigma_kappa(reference) ** 2 + _band_sigma_kappa(other) ** 2
            curvature_chi2.append(dkap**2 / max(var_kap, 1e-12))
        max_tangent_chi2 = float(max(tangent_chi2))
        max_curvature_chi2 = float(max(curvature_chi2))
        multiband_chi2 = max_tangent_chi2  # tangent drives the consistency flag
        multiband_consistent = max_tangent_chi2 < 4.0  # ~2 sigma in 1 dof
        if not multiband_consistent:
            warnings.append(
                f"bands disagree on tangent angle beyond their combined errors "
                f"(chi2 {max_tangent_chi2:.1f}); possible contaminant or mis-trace"
            )
        elif max_curvature_chi2 > 4.0:
            warnings.append(
                f"bands agree on tangent angle but not curvature (chi2 {max_curvature_chi2:.1f}); "
                "curvature is color-gradient sensitive - treat |kappa| as the less certain quantity"
            )

    reliability = 0.85
    if multiband_consistent:
        reliability += 0.10
    if reference.bridged_gap:
        reliability -= 0.20
    if reference.contested_fraction > 0.2 or reference.masked_invalid_fraction > 0.2:
        reliability -= 0.15
    if reference.touches_edge:
        reliability -= 0.20
    if reference.axis_ratio > 0.6:
        reliability -= 0.10
    if reference.is_line_fallback:
        reliability -= 0.10
    if multiband_consistent is False:
        # A color gradient (knots brighter in one band) shifts the ridge
        # slightly without invalidating the geometry; penalize modestly.
        reliability -= 0.15
    reliability = float(np.clip(reliability, 0.05, 1.0))
    overridden = False
    if reliability_override is not None:
        reliability = float(np.clip(reliability_override, 0.0, 1.0))
        overridden = True

    return ArcMeasurement(
        label=label,
        seed_ra_deg=seed_ra,
        seed_dec_deg=seed_dec,
        success=True,
        failure_reason=None,
        anchor_ra_deg=reference.anchor_ra_deg,
        anchor_dec_deg=reference.anchor_dec_deg,
        tangent_angle_offset_rad=reference.tangent_angle_offset_rad,
        curvature_arcsec_inv=reference.curvature_arcsec_inv,
        sigma_tangent_rad=float(sigma_tangent),
        sigma_curvature_arcsec_inv=float(sigma_curvature),
        reliability=reliability,
        reliability_overridden=overridden,
        reference_band=reference.band,
        bands=band_results,
        multiband_consistent=multiband_consistent,
        multiband_chi2=multiband_chi2,
        sigma_tangent_floor_rad=float(floor),
        warnings=tuple(warnings),
        config_summary=_config_summary(cfg),
    )


def measure_arc(
    mosaics: dict[str, BandMosaic],
    seed: SkyCoord,
    cfg: ArcMeasureConfig,
    *,
    label: str | None = None,
    rng: np.random.Generator | None = None,
    reliability_override: float | None = None,
    return_artifacts: bool = False,
) -> ArcMeasurement | tuple[ArcMeasurement, dict[str, BandArtifacts]]:
    """Measure one arc from a seed position across the configured bands."""
    if rng is None:
        rng = np.random.default_rng(cfg.rng_seed)
    bands = list(cfg.measure_bands) or [cfg.reference_band]
    if cfg.reference_band not in bands:
        bands.insert(0, cfg.reference_band)
    band_results: list[BandArcMeasurement] = []
    artifacts: dict[str, BandArtifacts] = {}
    for band in bands:
        mosaic = mosaics.get(band)
        if mosaic is None:
            band_results.append(
                BandArcMeasurement(band=band, success=False, failure_reason=f"no mosaic loaded for band {band}")
            )
            continue
        result, band_artifacts = measure_band(mosaic, seed, cfg, rng=rng)
        band_results.append(result)
        if band_artifacts is not None:
            artifacts[band] = band_artifacts
    measurement = combine_band_measurements(
        tuple(band_results),
        cfg,
        label=label,
        seed=seed,
        reliability_override=reliability_override,
    )
    if return_artifacts:
        return measurement, artifacts
    return measurement
