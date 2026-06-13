"""Seeded arc segmentation: background, smoothing, threshold, marker watershed."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from astropy.stats import SigmaClip, mad_std, sigma_clipped_stats
from photutils.background import Background2D, SExtractorBackground
from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources
from scipy import ndimage
from skimage.morphology import disk
from skimage.segmentation import watershed

from .config import SegmentationConfig
from .errors import SeedOffEmissionError, SegmentationError
from .mosaics import CutoutData


@dataclass(frozen=True)
class BackgroundResult:
    data_sub: np.ndarray
    sky_sigma: float
    background_median: float
    invalid_mask: np.ndarray


@dataclass(frozen=True)
class ArcSegmentation:
    mask: np.ndarray
    smoothed: np.ndarray
    seed_xy: tuple[float, float]
    threshold: float
    bridged_gap: bool
    n_components_before_closing: int
    touches_edge: bool
    masked_invalid_fraction: float
    contested_fraction: float
    area_pixels: int
    effective_threshold_sigma: float


def _detect_sources(image, threshold, npixels, mask):
    """detect_sources wrapper tolerant of the photutils >=3.0 keyword rename."""
    try:
        return detect_sources(image, threshold, n_pixels=npixels, mask=mask)
    except TypeError:
        return detect_sources(image, threshold, npixels=npixels, mask=mask)


def _deblend_sources(image, segm, npixels):
    """deblend_sources wrapper tolerant of the photutils >=3.0 keyword rename."""
    try:
        return deblend_sources(image, segm, n_pixels=npixels, n_levels=16, contrast=0.005, progress_bar=False)
    except TypeError:
        return deblend_sources(image, segm, npixels=npixels, nlevels=16, contrast=0.005, progress_bar=False)


def subtract_background(cutout: CutoutData, cfg: SegmentationConfig) -> BackgroundResult:
    data = np.asarray(cutout.data, dtype=float)
    invalid = ~np.isfinite(data) | (data == 0.0)
    npix = min(data.shape)
    box = int(round(cfg.bkg_box_arcsec / cutout.pixel_scale_arcsec))
    box = int(np.clip(box, 8, max(8, npix // 4)))
    background_map: np.ndarray | None = None
    sky_sigma = float("nan")
    background_median = float("nan")
    try:
        bkg = Background2D(
            data,
            box_size=box,
            mask=invalid,
            filter_size=int(cfg.bkg_filter_size),
            sigma_clip=SigmaClip(sigma=3.0),
            bkg_estimator=SExtractorBackground(),
            exclude_percentile=90.0,
        )
        background_map = np.asarray(bkg.background, dtype=float)
        sky_sigma = float(bkg.background_rms_median)
        background_median = float(bkg.background_median)
    except Exception:
        background_map = None
    if background_map is None or not math.isfinite(sky_sigma) or sky_sigma <= 0.0:
        values = data[~invalid]
        if values.size < 50:
            raise SegmentationError("Cutout has too few valid pixels for background estimation.")
        _, median, _ = sigma_clipped_stats(values, sigma=3.0, maxiters=5)
        sigma = float(mad_std(values - median))
        if not math.isfinite(sigma) or sigma <= 0.0:
            sigma = float(np.std(values)) or 1.0
        background_map = np.full_like(data, float(median))
        sky_sigma = sigma
        background_median = float(median)
    data_sub = np.where(invalid, 0.0, data - background_map)
    return BackgroundResult(
        data_sub=data_sub,
        sky_sigma=sky_sigma,
        background_median=background_median,
        invalid_mask=invalid,
    )


def smooth_image(data: np.ndarray, psf_fwhm_arcsec: float, pixel_scale_arcsec: float) -> np.ndarray:
    sigma_pix = float(psf_fwhm_arcsec) / (2.0 * math.sqrt(2.0 * math.log(2.0))) / float(pixel_scale_arcsec)
    if sigma_pix <= 0.0:
        return np.asarray(data, dtype=float)
    return ndimage.gaussian_filter(np.asarray(data, dtype=float), sigma=sigma_pix)


def _snap_seed(
    above: np.ndarray,
    seed_xy_pix: tuple[float, float],
    snap_radius_pix: float,
    smoothed: np.ndarray,
    threshold: float,
) -> tuple[int, int]:
    ny, nx = above.shape
    col = int(round(seed_xy_pix[0]))
    row = int(round(seed_xy_pix[1]))
    if not (0 <= row < ny and 0 <= col < nx):
        raise SeedOffEmissionError("Seed falls outside the cutout.")
    if above[row, col]:
        return row, col
    distances, (rows, cols) = ndimage.distance_transform_edt(~above, return_indices=True)
    distance = float(distances[row, col])
    if distance > snap_radius_pix:
        local = float(smoothed[row, col])
        raise SeedOffEmissionError(
            f"No emission above threshold ({threshold:.4g}) within the snap radius of the seed "
            f"(local smoothed value {local:.4g}, nearest above-threshold pixel {distance:.1f} px away)."
        )
    return int(rows[row, col]), int(cols[row, col])


def _local_arc_axis(
    above: np.ndarray,
    smoothed: np.ndarray,
    seed_rc: tuple[int, int],
    *,
    radius_pix: float,
    seed_segment_mask: np.ndarray | None = None,
) -> tuple[float, float, float, float] | None:
    """Flux-weighted principal axis of the seed's component near the seed.

    Uses the seed's deblended detection segment when available (it excludes
    blended neighbors that would otherwise drag the axis toward themselves);
    falls back to the seed's above-threshold connected component. Returns
    (cx, cy, cos, sin) in pixel coordinates, or None when undefined.
    """
    if seed_segment_mask is not None and seed_segment_mask[seed_rc]:
        member = seed_segment_mask
    else:
        labels, _ = ndimage.label(above)
        seed_label = labels[seed_rc]
        if seed_label == 0:
            return None
        member = labels == seed_label
    rows, cols = np.nonzero(member)
    near = np.hypot(cols - seed_rc[1], rows - seed_rc[0]) <= float(radius_pix)
    rows, cols = rows[near], cols[near]
    if rows.size < 6:
        return None
    flux = np.clip(smoothed[rows, cols], 0.0, None)
    total = float(flux.sum())
    if total <= 0.0:
        return None
    w = flux / total
    cx = float(np.sum(w * cols))
    cy = float(np.sum(w * rows))
    sxx = float(np.sum(w * (cols - cx) ** 2))
    syy = float(np.sum(w * (rows - cy) ** 2))
    sxy = float(np.sum(w * (cols - cx) * (rows - cy)))
    if sxx + syy <= 0.0:
        return None
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return cx, cy, math.cos(angle), math.sin(angle)


def segment_arc(
    bg: BackgroundResult,
    seed_xy_pix: tuple[float, float],
    cfg: SegmentationConfig,
    psf_fwhm_arcsec: float,
    pixel_scale_arcsec: float,
) -> ArcSegmentation:
    smoothed = smooth_image(bg.data_sub, psf_fwhm_arcsec, pixel_scale_arcsec)
    invalid = bg.invalid_mask
    gap_pix = float(cfg.max_bridge_gap_arcsec) / float(pixel_scale_arcsec)
    snap_radius_pix = float(cfg.snap_radius_arcsec) / float(pixel_scale_arcsec)

    # Competitor markers from photutils source detection (computed once; they
    # do not depend on the arc threshold escalation below). Deblending gives
    # neighbors that blend into the arc their own labels and hence their own
    # watershed markers.
    detect_threshold = float(cfg.detect_threshold_sigma) * bg.sky_sigma
    segm = _detect_sources(smoothed, detect_threshold, int(cfg.detect_npixels), invalid)
    if segm is not None:
        try:
            segm = _deblend_sources(smoothed, segm, int(cfg.detect_npixels))
        except Exception:
            pass

    threshold_sigma = float(cfg.threshold_sigma)
    max_threshold_sigma = 4.0 * threshold_sigma
    last_error: SegmentationError | None = None
    while True:
        threshold = threshold_sigma * bg.sky_sigma
        above = (smoothed > threshold) & ~invalid
        if not np.any(above):
            raise SeedOffEmissionError(
                f"No pixels above {threshold_sigma:.2f} sigma in the cutout; arc not detectable."
            )
        seed_rc = _snap_seed(above, seed_xy_pix, snap_radius_pix, smoothed, threshold)

        closing_radius = max(int(round(gap_pix / 2.0)), 0)
        if closing_radius >= 1:
            closed = ndimage.binary_closing(above, structure=disk(closing_radius))
            closed &= ~invalid
            closed |= above
        else:
            closed = above

        markers = np.zeros(above.shape, dtype=np.int32)
        markers[seed_rc] = 1
        competitor_label = 2
        if segm is not None:
            seed_segment = int(segm.data[seed_rc])
            catalog = SourceCatalog(bg.data_sub, segm)
            seed_segment_mask = (segm.data == seed_segment) if seed_segment > 0 else None
            axis = _local_arc_axis(
                above,
                smoothed,
                seed_rc,
                radius_pix=3.0 / pixel_scale_arcsec,
                seed_segment_mask=seed_segment_mask,
            )
            exclusion_pix = float(cfg.competitor_exclusion_arcsec) / float(pixel_scale_arcsec)
            for source in catalog:
                label = int(source.label)
                if label == seed_segment:
                    continue
                peak_row = float(source.max_value_yindex)
                peak_col = float(source.max_value_xindex)
                if axis is not None:
                    # Bright knots lie ON the ridge: their peaks sit close to
                    # the local arc axis. Only transversely displaced peaks
                    # (neighbor galaxies off the ridge) become competitors.
                    cx, cy, cos_o, sin_o = axis
                    t_perp = abs(-(peak_col - cx) * sin_o + (peak_row - cy) * cos_o)
                    if t_perp <= exclusion_pix:
                        continue
                else:
                    distance = math.hypot(peak_col - seed_rc[1], peak_row - seed_rc[0])
                    if distance <= exclusion_pix:
                        continue
                r, c = int(round(peak_row)), int(round(peak_col))
                if 0 <= r < above.shape[0] and 0 <= c < above.shape[1] and closed[r, c] and markers[r, c] == 0:
                    markers[r, c] = competitor_label
                    competitor_label += 1

        basins = watershed(-smoothed, markers=markers, mask=closed)
        candidate = basins == 1
        labels, _ = ndimage.label(candidate)
        seed_label = labels[seed_rc]
        if seed_label == 0:
            raise SegmentationError("Watershed did not assign the seed to any basin.")
        mask = labels == seed_label

        area = int(np.count_nonzero(mask))
        if area * pixel_scale_arcsec**2 <= cfg.max_area_arcsec2:
            break
        threshold_sigma *= 1.25
        last_error = SegmentationError(
            f"Arc mask exceeds max area ({cfg.max_area_arcsec2:.0f} arcsec^2) even at "
            f"{threshold_sigma:.2f} sigma threshold."
        )
        if threshold_sigma > max_threshold_sigma:
            raise last_error

    if area < int(cfg.min_area_pixels):
        raise SeedOffEmissionError(
            f"Emission at the seed yields only {area} mask pixels (< {cfg.min_area_pixels}); "
            "no measurable arc at this position."
        )

    # Count only significant above-threshold islands inside the mask: isolated
    # noise speckles attached by the closing must not flag a bridged gap.
    component_labels, n_raw = ndimage.label(mask & above)
    if n_raw > 1:
        sizes = ndimage.sum_labels(np.ones_like(component_labels), component_labels, np.arange(1, n_raw + 1))
        n_components = int(np.count_nonzero(np.asarray(sizes) >= 5))
        n_components = max(n_components, 1)
    else:
        n_components = int(n_raw)
    bridged_gap = n_components > 1

    touches_edge = bool(
        mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()
    )

    rows, cols = np.nonzero(mask)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    bbox_invalid = invalid[r0:r1, c0:c1]
    masked_invalid_fraction = float(np.count_nonzero(bbox_invalid)) / float(bbox_invalid.size)

    perimeter = ndimage.binary_dilation(mask) & ~mask
    contested = perimeter & (basins > 1)
    contested_fraction = float(np.count_nonzero(contested)) / float(max(np.count_nonzero(perimeter), 1))

    return ArcSegmentation(
        mask=mask,
        smoothed=smoothed,
        seed_xy=(float(seed_rc[1]), float(seed_rc[0])),
        threshold=float(threshold),
        bridged_gap=bool(bridged_gap),
        n_components_before_closing=int(n_components),
        touches_edge=touches_edge,
        masked_invalid_fraction=masked_invalid_fraction,
        contested_fraction=contested_fraction,
        area_pixels=area,
        effective_threshold_sigma=float(threshold_sigma),
    )
