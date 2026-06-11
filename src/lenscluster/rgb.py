from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.visualization import make_lupton_rgb
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


@dataclass(frozen=True)
class RGBDisplayConfig:
    q: float = DEFAULT_RGB_Q
    stretch: float = DEFAULT_RGB_STRETCH
    channel_gains: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_RGB_CHANNEL_GAINS))
    minimum: float = DEFAULT_RGB_MINIMUM


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
