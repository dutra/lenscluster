"""arctrace: seeded measurement of gravitational-arc geometry from HST mosaics.

Produces curved-arc-basis (tangent angle + curvature) constraints in the
arcfile format consumed by the lenscluster solver. See Birrer (2021),
arXiv:2104.09522, for the formalism: the ridge of a thin tangential arc in the
minimal curved-arc model is a circle of radius 1/s_tan, so a circle fit to the
ridge estimates exactly the observables the solver's CAB likelihood consumes.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import ArcMeasureConfig, SegmentationConfig, psf_fwhm_for_band
from .errors import (
    ArcfileWriteError,
    ArctraceError,
    CutoutError,
    RegionParseError,
    RidgeError,
    SeedOffEmissionError,
    SegmentationError,
)
from .measure import ArcMeasurement, BandArcMeasurement, measure_arc
from .mosaics import BandMosaic, CutoutData, load_band_mosaic, load_band_mosaics

__all__ = [
    "ArcMeasureConfig",
    "SegmentationConfig",
    "psf_fwhm_for_band",
    "ArcMeasurement",
    "BandArcMeasurement",
    "measure_arc",
    "BandMosaic",
    "CutoutData",
    "load_band_mosaic",
    "load_band_mosaics",
    "ArctraceError",
    "ArcfileWriteError",
    "CutoutError",
    "RegionParseError",
    "RidgeError",
    "SeedOffEmissionError",
    "SegmentationError",
    "__version__",
]
