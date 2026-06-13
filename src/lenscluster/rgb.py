from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.visualization import LuptonAsinhStretch, make_lupton_rgb
from skimage.transform import resize as resize_image


DEFAULT_RGB_Q = 8.0
DEFAULT_RGB_STRETCH = 0.1
DEFAULT_RGB_MINIMUM = 0.0
DEFAULT_RGB_CHANNEL_GAINS = {
    "red": 1.0,
    "green": 1.0,
    "blue": 1.2,
}
DEFAULT_MULTIBAND_NORMALIZATION_PERCENTILE = 99.7
# Calibrated-path defaults tuned on the AS1063/M0416 BUFFALO 30mas mosaics.
# The seven-band HFF defaults intentionally keep faint sky/noise pixels visible
# while compressing bright cores below pure white.
DEFAULT_CALIBRATED_RGB_Q = 6.5
DEFAULT_CALIBRATED_RGB_STRETCH = 0.0165
DEFAULT_CALIBRATED_RGB_CHANNEL_GAINS = {
    "red": 0.68,
    "green": 0.75,
    "blue": 3.5,
}
DEFAULT_CALIBRATED_RGB_WARM_HIGHLIGHT_DESATURATION = 0.65
CALIBRATED_RGB_MINIMUM_SKY_SIGMA = 1.2
DEFAULT_HFF_RGB_BANDS = ("F435W", "F606W", "F814W", "F105W", "F125W", "F140W", "F160W")
DEFAULT_HFF_RGB_REFERENCE_BAND = "F814W"
DEFAULT_HFF_RGB_Q = 6.4
DEFAULT_HFF_RGB_STRETCH = 0.0145
DEFAULT_HFF_RGB_MINIMUM = -5.5e-4
DEFAULT_HFF_RGB_CHANNEL_GAINS = {
    "red": 0.47,
    "green": 0.91,
    "blue": 3.95,
}
DEFAULT_HFF_RGB_CHANNEL_WEIGHTS = {
    "blue": {
        "F435W": 0.92,
        "F606W": 0.40,
    },
    "green": {
        "F606W": 0.66,
        "F814W": 0.96,
        "F105W": 0.18,
    },
    "red": {
        "F814W": 0.18,
        "F105W": 0.28,
        "F125W": 0.40,
        "F140W": 0.34,
        "F160W": 0.32,
    },
}
DEFAULT_HFF_RGB_WARM_HIGHLIGHT_DESATURATION = 0.20
DEFAULT_HFF_RGB_HIGHLIGHT_KNEE = 0.58
DEFAULT_HFF_RGB_HIGHLIGHT_CEILING = 0.88
DEFAULT_HFF_RGB_HIGHLIGHT_SOFTNESS = 0.44
SPEED_OF_LIGHT_ANGSTROM_PER_S = 2.99792458e18


@dataclass(frozen=True)
class RGBDisplayConfig:
    q: float = DEFAULT_RGB_Q
    stretch: float = DEFAULT_RGB_STRETCH
    channel_gains: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_RGB_CHANNEL_GAINS))
    minimum: float = DEFAULT_RGB_MINIMUM


@dataclass(frozen=True)
class CalibratedRGBDisplayConfig(RGBDisplayConfig):
    """Lupton asinh display with per-band sky subtraction and f_nu flux calibration.

    The weighted HFF path combines native-grid band cutouts directly and uses
    highlight compression to keep bright cores below pure white. The calibrated
    path does not spatially blur or resample pixels.
    """

    q: float = DEFAULT_CALIBRATED_RGB_Q
    stretch: float = DEFAULT_CALIBRATED_RGB_STRETCH
    channel_gains: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_CALIBRATED_RGB_CHANNEL_GAINS))
    band_backgrounds: Mapping[str, float] = field(default_factory=dict)
    band_fluxscales: Mapping[str, float] = field(default_factory=dict)
    warm_highlight_desaturation: float = DEFAULT_CALIBRATED_RGB_WARM_HIGHLIGHT_DESATURATION
    channel_weights: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    highlight_knee: float = 1.0
    highlight_ceiling: float = 1.0
    highlight_softness: float = DEFAULT_HFF_RGB_HIGHLIGHT_SOFTNESS


@dataclass(frozen=True)
class MultiBandRGBDisplayConfig(RGBDisplayConfig):
    channel_weights: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    reference_band: str = "F814W"
    normalization_percentile: float = DEFAULT_MULTIBAND_NORMALIZATION_PERCENTILE
    band_backgrounds: Mapping[str, float] = field(default_factory=dict)
    band_scales: Mapping[str, float] = field(default_factory=dict)


def trim_to_common_shape(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    valid = [np.asarray(array) for array in arrays if np.asarray(array).ndim == 2]
    if not valid:
        return []
    min_y = min(array.shape[0] for array in valid)
    min_x = min(array.shape[1] for array in valid)
    return [array[:min_y, :min_x] for array in valid]


def _validate_bands(bands: Sequence[str]) -> tuple[str, str, str]:
    band_tuple = tuple(str(band) for band in bands)
    if len(band_tuple) != 3:
        raise ValueError("RGB rendering requires exactly three bands in blue, green, red order.")
    return band_tuple[0], band_tuple[1], band_tuple[2]


def _validate_display(display: RGBDisplayConfig) -> None:
    if float(display.q) <= 0.0:
        raise ValueError("Lupton Q must be positive.")
    if float(display.stretch) <= 0.0:
        raise ValueError("Lupton stretch must be positive.")


def _channel_gain(display: RGBDisplayConfig, role: str) -> float:
    return float(display.channel_gains.get(role, DEFAULT_RGB_CHANNEL_GAINS[role]))


def _fill_invalid_with_median(data: np.ndarray, *, role: str) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(f"Cannot render Lupton RGB: no finite {role} channel pixels.")
    filled = np.array(values, copy=True)
    filled[~finite] = float(np.median(filled[finite]))
    return filled


def _multiband_required_bands(display: MultiBandRGBDisplayConfig) -> list[str]:
    required = {str(display.reference_band)}
    for role in ("blue", "green", "red"):
        required.update(str(band) for band in display.channel_weights.get(role, {}))
    return sorted(required)


def _validate_multiband_display(display: MultiBandRGBDisplayConfig) -> None:
    _validate_display(display)
    percentile = float(display.normalization_percentile)
    if percentile <= 0.0 or percentile > 100.0:
        raise ValueError("Multi-band RGB normalization_percentile must be in the interval (0, 100].")
    for role in ("blue", "green", "red"):
        if not display.channel_weights.get(role):
            raise ValueError(f"Multi-band RGB requires at least one {role} channel weight.")


def _resize_cutout_to_shape(data: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("RGB rendering requires 2D cutout arrays.")
    if values.shape == shape:
        return values
    return np.asarray(
        resize_image(
            values,
            shape,
            order=1,
            mode="edge",
            anti_aliasing=True,
            preserve_range=True,
        ),
        dtype=np.float32,
    )


def _measure_band_background_and_scale(data: np.ndarray, *, band: str, percentile: float) -> tuple[float, float]:
    values = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(f"Cannot render Lupton RGB: no finite {band} channel pixels.")
    median = float(np.median(values[finite]))
    centered = values - median
    scale = float(np.nanpercentile(centered[finite], percentile))
    if not np.isfinite(scale) or scale <= 0.0:
        positive = centered[finite & (centered > 0.0)]
        scale = float(np.nanmax(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return median, scale


def measure_multiband_normalization(
    cutouts_by_band: Mapping[str, np.ndarray],
    *,
    bands: Sequence[str],
    percentile: float = DEFAULT_MULTIBAND_NORMALIZATION_PERCENTILE,
) -> tuple[dict[str, float], dict[str, float]]:
    backgrounds: dict[str, float] = {}
    scales: dict[str, float] = {}
    for band in bands:
        band_key = str(band)
        backgrounds[band_key], scales[band_key] = _measure_band_background_and_scale(
            np.asarray(cutouts_by_band[band_key], dtype=np.float32),
            band=band_key,
            percentile=float(percentile),
        )
    return backgrounds, scales


def _normalize_multiband_cutout(
    data: np.ndarray,
    *,
    band: str,
    percentile: float,
    background: float | None = None,
    scale: float | None = None,
) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    filled = _fill_invalid_with_median(values, role=str(band))
    if background is None or scale is None:
        measured_background, measured_scale = _measure_band_background_and_scale(values, band=band, percentile=percentile)
        if background is None:
            background = measured_background
        if scale is None:
            scale = measured_scale
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    centered = filled - float(background)
    return np.clip(centered / scale, 0.0, None).astype(np.float32)


def _make_multiband_channel(
    cutouts_by_band: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    *,
    reference_shape: tuple[int, int],
    percentile: float,
    backgrounds: Mapping[str, float],
    scales: Mapping[str, float],
) -> np.ndarray:
    channel = np.zeros(reference_shape, dtype=np.float32)
    for band, weight in weights.items():
        band_key = str(band)
        resized = _resize_cutout_to_shape(cutouts_by_band[band_key], reference_shape)
        normalized = _normalize_multiband_cutout(
            resized,
            band=band_key,
            percentile=percentile,
            background=backgrounds.get(band_key),
            scale=scales.get(band_key),
        )
        channel += float(weight) * normalized
    return channel


def make_multiband_rgb(
    cutouts_by_band: Mapping[str, np.ndarray],
    *,
    display: MultiBandRGBDisplayConfig,
) -> np.ndarray:
    _validate_multiband_display(display)
    missing = [band for band in _multiband_required_bands(display) if band not in cutouts_by_band]
    if missing:
        raise ValueError(f"Missing source band(s) for multi-band RGB: {', '.join(missing)}.")

    reference = np.asarray(cutouts_by_band[str(display.reference_band)], dtype=np.float32)
    if reference.ndim != 2:
        raise ValueError("Multi-band RGB reference band must be a 2D cutout array.")
    reference_shape = (int(reference.shape[0]), int(reference.shape[1]))
    percentile = float(display.normalization_percentile)

    img_b = _make_multiband_channel(
        cutouts_by_band,
        display.channel_weights["blue"],
        reference_shape=reference_shape,
        percentile=percentile,
        backgrounds=display.band_backgrounds,
        scales=display.band_scales,
    )
    img_g = _make_multiband_channel(
        cutouts_by_band,
        display.channel_weights["green"],
        reference_shape=reference_shape,
        percentile=percentile,
        backgrounds=display.band_backgrounds,
        scales=display.band_scales,
    )
    img_r = _make_multiband_channel(
        cutouts_by_band,
        display.channel_weights["red"],
        reference_shape=reference_shape,
        percentile=percentile,
        backgrounds=display.band_backgrounds,
        scales=display.band_scales,
    )

    return make_lupton_rgb(
        img_r * _channel_gain(display, "red"),
        img_g * _channel_gain(display, "green"),
        img_b * _channel_gain(display, "blue"),
        minimum=float(display.minimum),
        stretch=float(display.stretch),
        Q=float(display.q),
    )


def compute_fnu_band_fluxscales(
    photflam_by_band: Mapping[str, float | None],
    photplam_by_band: Mapping[str, float | None],
    *,
    reference_band: str,
) -> dict[str, float]:
    """Multiplicative factors converting counts to a common f_nu scale.

    HST drizzled mosaics are stored in electrons/s; equal AB surface brightness
    produces very different count rates per filter (the instrument is far more
    sensitive in F606W than in F435W), so combining raw counts skews colors.
    f_nu = counts * PHOTFLAM * PHOTPLAM^2 / c; factors are normalized so the
    reference band keeps a scale of 1. Bands with missing photometry keywords
    fall back to 1.
    """

    raw: dict[str, float] = {}
    for band in photflam_by_band:
        photflam = photflam_by_band.get(band)
        photplam = photplam_by_band.get(band)
        if photflam is None or photplam is None:
            continue
        photflam = float(photflam)
        photplam = float(photplam)
        if not np.isfinite(photflam) or not np.isfinite(photplam) or photflam <= 0.0 or photplam <= 0.0:
            continue
        raw[str(band)] = photflam * photplam * photplam / SPEED_OF_LIGHT_ANGSTROM_PER_S
    reference = raw.get(str(reference_band))
    if reference is None or reference <= 0.0:
        return {band: 1.0 for band in photflam_by_band}
    return {str(band): raw.get(str(band), reference) / reference for band in photflam_by_band}


def _calibrated_channel(
    cutouts_by_band: Mapping[str, np.ndarray],
    band: str,
    role: str,
    display: CalibratedRGBDisplayConfig,
) -> np.ndarray:
    return _calibrated_source_band(cutouts_by_band, band, role, display) * _channel_gain(display, role)


def _calibrated_source_band(
    cutouts_by_band: Mapping[str, np.ndarray],
    band: str,
    role: str,
    display: CalibratedRGBDisplayConfig,
) -> np.ndarray:
    values = np.asarray(cutouts_by_band[str(band)], dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("RGB rendering requires 2D cutout arrays.")
    background = float(display.band_backgrounds.get(str(band), 0.0))
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(f"Cannot render calibrated RGB: no finite {role} channel pixels.")
    filled = np.array(values, copy=True)
    filled[~finite] = background
    fluxscale = float(display.band_fluxscales.get(str(band), 1.0))
    return (filled - background) * fluxscale


def _calibrated_weighted_channel(
    cutouts_by_band: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    role: str,
    display: CalibratedRGBDisplayConfig,
) -> np.ndarray:
    channel: np.ndarray | None = None
    for band, weight in weights.items():
        source = _calibrated_source_band(cutouts_by_band, str(band), role, display) * float(weight)
        channel = source if channel is None else channel + source
    if channel is None:
        raise ValueError(f"Calibrated RGB requires at least one {role} channel weight.")
    return channel * _channel_gain(display, role)


def _cubic_ramp(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    scaled = np.clip((values - float(edge0)) / (float(edge1) - float(edge0)), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _neutralize_warm_highlights(rgb: np.ndarray, *, amount: float) -> np.ndarray:
    amount = float(amount)
    if amount <= 0.0:
        return rgb

    values = np.asarray(rgb, dtype=np.float32)
    red = values[0]
    green = values[1]
    blue = values[2]
    brightness = np.max(values, axis=0)
    warm_excess = np.clip((0.5 * (red + green) - blue) / np.maximum(brightness, 1.0e-6), 0.0, 1.0)
    highlight_weight = _cubic_ramp(0.08, 0.65, brightness)
    warm_blend = np.clip(amount * warm_excess * highlight_weight, 0.0, 1.0)
    clipped_blend = _cubic_ramp(0.92, 1.0, brightness)
    blend = np.maximum(warm_blend, clipped_blend)
    neutral = np.broadcast_to(brightness, values.shape)
    return np.clip(values * (1.0 - blend) + neutral * blend, 0.0, 1.0)


def _gently_neutralize_warm_highlights(rgb: np.ndarray, *, amount: float) -> np.ndarray:
    amount = float(amount)
    if amount <= 0.0:
        return rgb

    values = np.asarray(rgb, dtype=np.float32)
    red = values[0]
    green = values[1]
    blue = values[2]
    brightness = np.max(values, axis=0)
    warm_excess = np.clip((0.5 * (red + green) - blue) / np.maximum(brightness, 1.0e-6), 0.0, 1.0)
    highlight_weight = _cubic_ramp(0.18, 0.80, brightness)
    blend = np.clip(amount * warm_excess * highlight_weight, 0.0, 0.65)
    neutral = np.broadcast_to(brightness, values.shape)
    return values * (1.0 - blend) + neutral * blend


def _compress_highlights(
    rgb: np.ndarray,
    *,
    knee: float,
    ceiling: float,
    softness: float,
) -> np.ndarray:
    ceiling = float(ceiling)
    knee = float(knee)
    if ceiling >= 1.0:
        return rgb
    if not np.isfinite(ceiling) or ceiling <= 0.0 or ceiling > 1.0:
        raise ValueError("Calibrated RGB highlight_ceiling must be in the interval (0, 1].")
    if not np.isfinite(knee) or knee < 0.0 or knee >= ceiling:
        raise ValueError("Calibrated RGB highlight_knee must be finite and smaller than highlight_ceiling.")
    softness = max(float(softness), 1.0e-6)

    values = np.asarray(rgb, dtype=np.float32)
    brightness = np.max(values, axis=0)
    mapped = np.array(brightness, copy=True)
    high = brightness > knee
    if np.any(high):
        excess = (brightness[high] - knee) / softness
        mapped[high] = knee + (ceiling - knee) * (1.0 - np.exp(-excess))
    mapped = np.minimum(mapped, ceiling)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(brightness > 0.0, mapped / brightness, 0.0)
    return np.clip(values * scale, 0.0, ceiling)


def _calibrated_channel_weights_for_bands(bands: Sequence[str]) -> dict[str, dict[str, float]]:
    available = {str(band) for band in bands}
    weights: dict[str, dict[str, float]] = {}
    for role, role_weights in DEFAULT_HFF_RGB_CHANNEL_WEIGHTS.items():
        selected = {band: float(weight) for band, weight in role_weights.items() if band in available}
        if selected:
            weights[role] = selected
    return weights


def make_calibrated_rgb(
    cutouts_by_band: Mapping[str, np.ndarray],
    *,
    bands: Sequence[str],
    display: CalibratedRGBDisplayConfig,
) -> np.ndarray:
    _validate_display(display)

    if display.channel_weights:
        missing_roles = [role for role in ("blue", "green", "red") if not display.channel_weights.get(role)]
        if missing_roles:
            raise ValueError(f"Calibrated RGB missing channel weight(s): {', '.join(missing_roles)}.")
        blue, green, red = trim_to_common_shape(
            [
                _calibrated_weighted_channel(cutouts_by_band, display.channel_weights["blue"], "blue", display),
                _calibrated_weighted_channel(cutouts_by_band, display.channel_weights["green"], "green", display),
                _calibrated_weighted_channel(cutouts_by_band, display.channel_weights["red"], "red", display),
            ]
        )
    else:
        blue_band, green_band, red_band = _validate_bands(bands)
        blue, green, red = trim_to_common_shape(
            [
                _calibrated_channel(cutouts_by_band, blue_band, "blue", display),
                _calibrated_channel(cutouts_by_band, green_band, "green", display),
                _calibrated_channel(cutouts_by_band, red_band, "red", display),
            ]
        )
    channels = np.clip(np.stack([red, green, blue], axis=0) - float(display.minimum), 0.0, None)
    intensity = np.mean(channels, axis=0)
    stretch_fn = LuptonAsinhStretch(stretch=float(display.stretch), Q=float(display.q))
    stretched = stretch_fn(np.clip(intensity, 0.0, None), clip=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(intensity > 0.0, stretched / intensity, 0.0)
    scaled = channels * ratio
    if float(display.highlight_ceiling) < 1.0:
        scaled = _gently_neutralize_warm_highlights(scaled, amount=float(display.warm_highlight_desaturation))
        scaled = _compress_highlights(
            scaled,
            knee=float(display.highlight_knee),
            ceiling=float(display.highlight_ceiling),
            softness=float(display.highlight_softness),
        )
    else:
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled = _neutralize_warm_highlights(scaled, amount=float(display.warm_highlight_desaturation))
    return np.round(np.moveaxis(scaled, 0, -1) * 255.0).astype(np.uint8)


def build_rgb_display_from_band_images(
    band_images: Mapping[str, Any],
    bands: Sequence[str],
    *,
    q: float = DEFAULT_RGB_Q,
    stretch: float = DEFAULT_RGB_STRETCH,
    channel_gains: Mapping[str, float] = DEFAULT_RGB_CHANNEL_GAINS,
    minimum: float = DEFAULT_RGB_MINIMUM,
) -> RGBDisplayConfig:
    _ = band_images
    _validate_bands(bands)
    display = RGBDisplayConfig(q=float(q), stretch=float(stretch), channel_gains=dict(channel_gains), minimum=float(minimum))
    _validate_display(display)
    return display


def build_rgb_display_from_arrays(
    cutouts_by_band: Mapping[str, np.ndarray],
    bands: Sequence[str],
    *,
    q: float = DEFAULT_RGB_Q,
    stretch: float = DEFAULT_RGB_STRETCH,
    channel_gains: Mapping[str, float] = DEFAULT_RGB_CHANNEL_GAINS,
    minimum: float = DEFAULT_RGB_MINIMUM,
) -> RGBDisplayConfig:
    _ = cutouts_by_band
    return build_rgb_display_from_band_images({}, bands, q=q, stretch=stretch, channel_gains=channel_gains, minimum=minimum)


def make_natural_rgb(
    cutouts_by_band: Mapping[str, np.ndarray],
    *,
    bands: Sequence[str],
    display: RGBDisplayConfig | None = None,
) -> np.ndarray:
    if display is None:
        display = RGBDisplayConfig()
    if isinstance(display, MultiBandRGBDisplayConfig):
        return make_multiband_rgb(cutouts_by_band, display=display)
    if isinstance(display, CalibratedRGBDisplayConfig):
        return make_calibrated_rgb(cutouts_by_band, bands=bands, display=display)
    blue_band, green_band, red_band = _validate_bands(bands)
    _validate_display(display)

    blue = np.asarray(cutouts_by_band[str(blue_band)], dtype=np.float32)
    green = np.asarray(cutouts_by_band[str(green_band)], dtype=np.float32)
    red = np.asarray(cutouts_by_band[str(red_band)], dtype=np.float32)
    if blue.ndim != 2 or green.ndim != 2 or red.ndim != 2:
        raise ValueError("RGB rendering requires 2D cutout arrays.")

    blue, green, red = trim_to_common_shape([blue, green, red])
    img_r = _fill_invalid_with_median(red, role="red") * _channel_gain(display, "red")
    img_g = _fill_invalid_with_median(green, role="green") * _channel_gain(display, "green")
    img_b = _fill_invalid_with_median(blue, role="blue") * _channel_gain(display, "blue")

    return make_lupton_rgb(
        img_r,
        img_g,
        img_b,
        minimum=float(display.minimum),
        stretch=float(display.stretch),
        Q=float(display.q),
    )
