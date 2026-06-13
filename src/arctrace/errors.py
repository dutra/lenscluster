from __future__ import annotations


class ArctraceError(Exception):
    """Base class for all arctrace errors."""


class CutoutError(ArctraceError):
    """Raised when a cutout cannot be extracted (seed outside mosaic, bad WCS)."""


class SeedOffEmissionError(ArctraceError):
    """Raised when no above-threshold emission is found near the seed."""


class SegmentationError(ArctraceError):
    """Raised when the arc segmentation fails (area cap, too few pixels)."""


class RidgeError(ArctraceError):
    """Raised when too few valid ridge points can be extracted."""


class RegionParseError(ArctraceError):
    """Raised when a DS9 region file cannot be interpreted."""


class ArcfileWriteError(ArctraceError):
    """Raised when measurements cannot be serialized to a valid arcfile."""
