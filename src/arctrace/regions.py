"""Minimal DS9 region file parser for seed positions.

Supports celestial frames only (fk5/icrs/j2000), shapes point/circle/ellipse/
box (center used as the seed), decimal-degree or sexagesimal coordinates, and
``text={...}`` attributes carrying the arc ID.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord

from .errors import RegionParseError

_CELESTIAL_FRAMES = {"fk5", "icrs", "j2000"}
_REJECTED_FRAMES = {"image", "physical", "galactic", "ecliptic", "amplifier", "detector", "fk4", "b1950"}
_SHAPES = ("point", "circle", "ellipse", "box", "annulus")
_SHAPE_RE = re.compile(r"^([+-]?)(point|circle|ellipse|box|annulus)\s*\(([^)]*)\)\s*(?:#(.*))?$", re.IGNORECASE)
_TEXT_RE = re.compile(r"text\s*=\s*\{([^}]*)\}", re.IGNORECASE)


@dataclass(frozen=True)
class RegionSeed:
    ra_deg: float
    dec_deg: float
    label_raw: str | None
    radius_arcsec: float | None
    line_number: int

    @property
    def coord(self) -> SkyCoord:
        return SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg, frame="icrs")


def normalize_arc_id(raw: str) -> str:
    """Normalize a region label to a standalone arc ID."""
    text = str(raw).strip()
    if not text:
        raise ValueError("Empty arc ID.")
    if any(ch.isspace() for ch in text):
        raise ValueError(f"Arc ID {text!r} must not contain whitespace.")
    return text


def _parse_angle_value(token: str, *, default_unit: u.Unit) -> float:
    """Radius-like value with DS9 unit suffix; returns arcsec."""
    token = token.strip()
    if token.endswith('"'):
        return float(token[:-1])
    if token.endswith("'"):
        return float(token[:-1]) * 60.0
    if token.lower().endswith("r"):
        raise RegionParseError(f"Radian region sizes are not supported: {token!r}")
    return float(token) * default_unit.to(u.arcsec)


def _parse_coordinates(ra_token: str, dec_token: str) -> tuple[float, float]:
    ra_token = ra_token.strip()
    dec_token = dec_token.strip()
    if ":" in ra_token or ":" in dec_token:
        coord = SkyCoord(ra=ra_token, dec=dec_token, unit=(u.hourangle, u.deg), frame="icrs")
    else:
        coord = SkyCoord(ra=float(ra_token) * u.deg, dec=float(dec_token) * u.deg, frame="icrs")
    return float(coord.ra.deg), float(coord.dec.deg)


def parse_ds9_regions(path: str | Path) -> list[RegionSeed]:
    path = Path(path)
    if not path.exists():
        raise RegionParseError(f"Region file not found: {path}")
    seeds: list[RegionSeed] = []
    frame: str | None = None
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("global"):
            continue

        # Inline frame prefix: "fk5; circle(...)".
        while ";" in line:
            head, _, rest = line.partition(";")
            head_l = head.strip().lower()
            if head_l in _CELESTIAL_FRAMES:
                frame = head_l
                line = rest.strip()
            elif head_l in _REJECTED_FRAMES:
                raise RegionParseError(
                    f"{path}:{line_number}: unsupported region frame {head!r}; use fk5/icrs."
                )
            else:
                break
        if not line:
            continue

        lowered = line.lower()
        if lowered in _CELESTIAL_FRAMES:
            frame = lowered
            continue
        if lowered in _REJECTED_FRAMES:
            raise RegionParseError(f"{path}:{line_number}: unsupported region frame {line!r}; use fk5/icrs.")

        match = _SHAPE_RE.match(line)
        if match is None:
            continue
        if match.group(1) == "-":
            continue  # exclusion region
        if frame is None:
            raise RegionParseError(
                f"{path}:{line_number}: region shape before any coordinate frame declaration; "
                "add an 'fk5' or 'icrs' line."
            )
        arguments = [token for token in match.group(3).split(",") if token.strip()]
        if len(arguments) < 2:
            raise RegionParseError(f"{path}:{line_number}: shape needs at least 2 coordinates.")
        try:
            ra_deg, dec_deg = _parse_coordinates(arguments[0], arguments[1])
        except (ValueError, TypeError) as exc:
            raise RegionParseError(f"{path}:{line_number}: cannot parse coordinates: {exc}") from exc

        radius_arcsec: float | None = None
        shape = match.group(2).lower()
        if shape in {"circle", "ellipse", "annulus"} and len(arguments) >= 3:
            try:
                radius_arcsec = _parse_angle_value(arguments[2], default_unit=u.deg)
            except (ValueError, RegionParseError):
                radius_arcsec = None

        label_raw: str | None = None
        comment = match.group(4)
        if comment:
            text_match = _TEXT_RE.search(comment)
            if text_match:
                label_raw = text_match.group(1).strip() or None

        seeds.append(
            RegionSeed(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                label_raw=label_raw,
                radius_arcsec=radius_arcsec,
                line_number=line_number,
            )
        )
    if not seeds:
        raise RegionParseError(f"No usable region shapes found in {path}.")
    return seeds
