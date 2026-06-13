from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# Approximate PSF FWHM of the drizzled BUFFALO/HFF mosaics, per camera.
DEFAULT_PSF_FWHM_ARCSEC: dict[str, float] = {
    # ACS/WFC optical
    "F435W": 0.10,
    "F475W": 0.10,
    "F606W": 0.10,
    "F625W": 0.10,
    "F814W": 0.10,
    # WFC3/IR
    "F105W": 0.18,
    "F110W": 0.18,
    "F125W": 0.18,
    "F140W": 0.18,
    "F160W": 0.18,
}
DEFAULT_PSF_FWHM_FALLBACK_ARCSEC = 0.10


def psf_fwhm_for_band(band: str, overrides: Mapping[str, float] | None = None) -> float:
    key = str(band).upper()
    if overrides:
        upper_overrides = {str(k).upper(): float(v) for k, v in overrides.items()}
        if key in upper_overrides:
            return upper_overrides[key]
        if "*" in upper_overrides:
            return upper_overrides["*"]
    return DEFAULT_PSF_FWHM_ARCSEC.get(key, DEFAULT_PSF_FWHM_FALLBACK_ARCSEC)


@dataclass(frozen=True)
class SegmentationConfig:
    threshold_sigma: float = 1.2
    detect_threshold_sigma: float = 2.0
    detect_npixels: int = 10
    bkg_box_arcsec: float = 3.0
    bkg_filter_size: int = 3
    max_bridge_gap_arcsec: float = 0.5
    snap_radius_arcsec: float = 0.7
    competitor_exclusion_arcsec: float = 1.0
    max_area_arcsec2: float = 80.0
    min_area_pixels: int = 12


@dataclass(frozen=True)
class ArcMeasureConfig:
    cutout_size_arcsec: float = 20.0
    max_cutout_size_arcsec: float = 40.0
    psf_fwhm_arcsec: Mapping[str, float] = field(default_factory=dict)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    reference_band: str = "F814W"
    measure_bands: tuple[str, ...] = ()
    slice_width_psf_factor: float = 1.0
    min_slice_snr: float = 1.5
    min_ridge_points: int = 5
    n_bootstrap: int = 200
    # Multiplicative perturbations of (threshold_sigma, max_bridge_gap_arcsec)
    # used to estimate the segmentation systematic scatter.
    segmentation_variants: tuple[tuple[float, float], ...] = (
        (0.85, 0.5),
        (0.85, 1.5),
        (1.15, 0.5),
        (1.15, 1.5),
    )
    sigma_e_floor: float = 0.25
    tangent_sigma_cap_rad: float = 0.9
    curvature_sigma_floor_arcsec_inv: float = 0.005
    # Irreducible method-systematic floors. Bootstrap formal errors on a
    # well-sampled ridge are far smaller (sub-degree) than the true
    # reproducibility of ridge-based geometry, as multi-band scatter on real
    # arcs demonstrates. These floors encode "do not claim tangent precision
    # below ~1 deg, nor curvature precision below ~10%." Applied to both the
    # reported sigmas and the multi-band consistency check.
    tangent_method_floor_rad: float = 0.02  # ~1.1 deg
    curvature_method_floor_frac: float = 0.10
    straightness_sagitta_snr: float = 1.0
    # Restrict the geometric fit to ridge points within this arclength of the
    # anchor (None = use the full ridge). Birrer (2021) Sect. 5.3: constant
    # curvature breaks down for arcs spanning large azimuth.
    fit_halfspan_arcsec: float | None = None
    # Optional pixel-level PSF-convolved refinement of the geometric fit.
    refine: str = "none"  # "none" | "forward"
    rng_seed: int = 1234

    def psf_fwhm(self, band: str) -> float:
        return psf_fwhm_for_band(band, self.psf_fwhm_arcsec)
