import math

import numpy as np
import pytest

from arctrace.config import SegmentationConfig
from arctrace.errors import SeedOffEmissionError
from arctrace.mosaics import CutoutData
from arctrace.ridge import resample_ridge_polar, trace_ridge_moments
from arctrace.segmentation import segment_arc, subtract_background
from arctrace_synth import make_tan_wcs, paint_arc_on_wcs

PIXSCALE = 0.06
NOISE = 0.05
RA0, DEC0 = 10.0, 0.0


def _make_cutout(npix: int = 300, *, radius_arcsec: float = 6.0, half_span_rad: float = 0.5,
                 width_arcsec: float = 0.35, peak: float = 1.5, seed: int = 4):
    wcs = make_tan_wcs(RA0, DEC0, pixscale_arcsec=PIXSCALE, crpix=(npix / 2.0, npix / 2.0))
    rng = np.random.default_rng(seed)
    image, truth = paint_arc_on_wcs(
        wcs,
        (npix, npix),
        circle_center_offsets_arcsec=(0.0, 0.0),
        radius_arcsec=radius_arcsec,
        azimuth_center_rad=0.4,
        half_span_rad=half_span_rad,
        width_arcsec=width_arcsec,
        peak=peak,
        noise_sigma=NOISE,
        rng=rng,
    )
    return image, truth, wcs


def _cutout_data(image, wcs) -> CutoutData:
    return CutoutData(
        band="F814W",
        data=np.asarray(image, dtype=float),
        wcs=wcs,
        pixel_scale_arcsec=PIXSCALE,
        photflam=None,
        photplam=None,
        global_background=0.0,
        global_background_sigma=NOISE,
    )


def _seed_pixel(wcs, truth):
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    coord = SkyCoord(ra=truth["anchor_ra_deg"] * u.deg, dec=truth["anchor_dec_deg"] * u.deg)
    x, y = wcs.world_to_pixel(coord)
    return float(x), float(y)


def _gaussian_blob(shape, center_rc, sigma_pix, peak):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    d2 = (yy - center_rc[0]) ** 2 + (xx - center_rc[1]) ** 2
    return peak * np.exp(-0.5 * d2 / sigma_pix**2)


def test_segment_simple_arc() -> None:
    image, truth, wcs = _make_cutout()
    cutout = _cutout_data(image, wcs)
    cfg = SegmentationConfig()
    bg = subtract_background(cutout, cfg)
    assert bg.sky_sigma == pytest.approx(NOISE, rel=0.3)
    seed = _seed_pixel(wcs, truth)
    seg = segment_arc(bg, seed, cfg, psf_fwhm_arcsec=0.1, pixel_scale_arcsec=PIXSCALE)
    assert seg.area_pixels > 200
    assert not seg.touches_edge
    assert not seg.bridged_gap
    assert seg.contested_fraction == pytest.approx(0.0, abs=0.05)


def test_watershed_excludes_offset_blob() -> None:
    image, truth, wcs = _make_cutout()
    # Blob 1.2 arcsec radially outside the ridge, blending into the arc.
    azim = 0.4
    blob_r_arcsec = 6.0 + 1.2
    npix = image.shape[0]
    # offsets frame: x westward; pixel frame of this flipped WCS: +x west, +y north.
    blob_col = npix / 2.0 + (blob_r_arcsec * math.cos(azim)) / PIXSCALE - 0.5
    blob_row = npix / 2.0 + (blob_r_arcsec * math.sin(azim)) / PIXSCALE - 0.5
    blob = _gaussian_blob(image.shape, (blob_row, blob_col), sigma_pix=0.35 / PIXSCALE, peak=5.0)
    cutout = _cutout_data(image + blob, wcs)
    cfg = SegmentationConfig()
    bg = subtract_background(cutout, cfg)
    seed = _seed_pixel(wcs, truth)
    seg = segment_arc(bg, seed, cfg, psf_fwhm_arcsec=0.1, pixel_scale_arcsec=PIXSCALE)
    assert not seg.mask[int(round(blob_row)), int(round(blob_col))]
    assert seg.contested_fraction > 0.0


def test_knot_on_ridge_is_not_a_competitor() -> None:
    image, truth, wcs = _make_cutout()
    # Bright knot ON the ridge, 1.5 arcsec along the arc from the seed.
    azim = 0.4 + 1.5 / 6.0
    npix = image.shape[0]
    knot_col = npix / 2.0 + (6.0 * math.cos(azim)) / PIXSCALE - 0.5
    knot_row = npix / 2.0 + (6.0 * math.sin(azim)) / PIXSCALE - 0.5
    knot = _gaussian_blob(image.shape, (knot_row, knot_col), sigma_pix=0.15 / PIXSCALE, peak=6.0)
    cutout = _cutout_data(image + knot, wcs)
    cfg = SegmentationConfig()
    bg = subtract_background(cutout, cfg)
    seed = _seed_pixel(wcs, truth)
    seg = segment_arc(bg, seed, cfg, psf_fwhm_arcsec=0.1, pixel_scale_arcsec=PIXSCALE)
    assert seg.mask[int(round(knot_row)), int(round(knot_col))]


def test_gap_bridging_on_and_off() -> None:
    image, truth, wcs = _make_cutout()
    # Carve a 0.35 arcsec gap across the arc, 1.2 arcsec from the seed.
    azim_gap = 0.4 + 1.2 / 6.0
    gap_halfwidth_rad = 0.5 * 0.35 / 6.0
    npix = image.shape[0]
    yy, xx = np.mgrid[0:npix, 0:npix]
    x_off = (xx - (npix / 2.0 - 0.5)) * PIXSCALE
    y_off = (yy - (npix / 2.0 - 0.5)) * PIXSCALE
    azim = np.arctan2(y_off, x_off)
    in_gap = np.abs(np.arctan2(np.sin(azim - azim_gap), np.cos(azim - azim_gap))) < gap_halfwidth_rad
    gapped = image.copy()
    gapped[in_gap] = np.random.default_rng(2).normal(0.0, NOISE, size=int(in_gap.sum()))

    seed = _seed_pixel(wcs, truth)
    bridged_cfg = SegmentationConfig(max_bridge_gap_arcsec=0.6)
    bg = subtract_background(_cutout_data(gapped, wcs), bridged_cfg)
    seg_bridged = segment_arc(bg, seed, bridged_cfg, psf_fwhm_arcsec=0.1, pixel_scale_arcsec=PIXSCALE)
    assert seg_bridged.bridged_gap

    narrow_cfg = SegmentationConfig(max_bridge_gap_arcsec=0.1)
    bg2 = subtract_background(_cutout_data(gapped, wcs), narrow_cfg)
    seg_narrow = segment_arc(bg2, seed, narrow_cfg, psf_fwhm_arcsec=0.1, pixel_scale_arcsec=PIXSCALE)
    assert not seg_narrow.bridged_gap
    assert seg_narrow.area_pixels < seg_bridged.area_pixels


def test_blank_sky_seed_raises() -> None:
    image, truth, wcs = _make_cutout()
    cutout = _cutout_data(image, wcs)
    cfg = SegmentationConfig()
    bg = subtract_background(cutout, cfg)
    with pytest.raises(SeedOffEmissionError):
        segment_arc(bg, (30.0, 30.0), cfg, psf_fwhm_arcsec=0.1, pixel_scale_arcsec=PIXSCALE)


def test_edge_contact_flag() -> None:
    # Small cutout with the circle center offset so one arm of the arc runs
    # off the frame while the seed stays inside.
    npix = 120
    wcs = make_tan_wcs(RA0, DEC0, pixscale_arcsec=PIXSCALE, crpix=(npix / 2.0, npix / 2.0))
    image, truth = paint_arc_on_wcs(
        wcs,
        (npix, npix),
        circle_center_offsets_arcsec=(1.0, 0.0),
        radius_arcsec=3.0,
        azimuth_center_rad=math.pi / 2.0,
        half_span_rad=1.4,
        width_arcsec=0.35,
        peak=1.5,
        noise_sigma=NOISE,
        rng=np.random.default_rng(8),
    )
    cutout = _cutout_data(image, wcs)
    cfg = SegmentationConfig()
    bg = subtract_background(cutout, cfg)
    seed = _seed_pixel(wcs, truth)
    seg = segment_arc(bg, seed, cfg, psf_fwhm_arcsec=0.1, pixel_scale_arcsec=PIXSCALE)
    assert seg.touches_edge


def test_ridge_traces_circle() -> None:
    image, truth, wcs = _make_cutout()
    cutout = _cutout_data(image, wcs)
    cfg = SegmentationConfig()
    bg = subtract_background(cutout, cfg)
    seed = _seed_pixel(wcs, truth)
    seg = segment_arc(bg, seed, cfg, psf_fwhm_arcsec=0.1, pixel_scale_arcsec=PIXSCALE)
    trace = trace_ridge_moments(
        bg.data_sub,
        seg.mask,
        slice_width_pix=0.1 / PIXSCALE,
        sky_sigma=bg.sky_sigma,
        min_snr=1.5,
        pixel_scale_arcsec=PIXSCALE,
    )
    assert len(trace.points) >= 10
    # Pass 1 slices perpendicular to the straight principal axis, so the arc
    # ends are cut obliquely: allow a couple tenths of an arcsec there.
    npix = image.shape[0]
    cx = npix / 2.0 - 0.5
    cy = npix / 2.0 - 0.5
    radii = np.hypot(trace.x_pix - cx, trace.y_pix - cy) * PIXSCALE
    assert np.max(np.abs(radii - 6.0)) < 0.25
    assert np.median(np.abs(radii - 6.0)) < 0.1
    assert trace.length_arcsec == pytest.approx(truth["length_arcsec"], rel=0.2)
    assert trace.width_arcsec == pytest.approx(truth["width_arcsec"], rel=0.4)

    # Pass 2 re-slices in polar coordinates around the curvature center,
    # eliminating the obliquity bias.
    polar = resample_ridge_polar(
        bg.data_sub,
        seg.mask,
        center_xy_pix=(cx, cy),
        slice_width_pix=0.1 / PIXSCALE,
        sky_sigma=bg.sky_sigma,
        min_snr=1.5,
        pixel_scale_arcsec=PIXSCALE,
    )
    radii_polar = np.hypot(polar.x_pix - cx, polar.y_pix - cy) * PIXSCALE
    assert np.max(np.abs(radii_polar - 6.0)) < 0.1
