"""Source-plane consistency scores for candidate family partitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .family_partition import canonicalize_partition
from .family_partition import _logit as _pair_logit


RayShooter = Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class SourcePlaneScoreResult:
    """Source-plane scores and diagnostics for candidate partitions."""

    log_scores: np.ndarray
    family_table: pd.DataFrame


@dataclass(frozen=True)
class SourcePlaneCache:
    """Cached ray-shooting values over a redshift grid."""

    z_grid: np.ndarray
    beta_x: np.ndarray
    beta_y: np.ndarray
    image_sigma_arcsec: np.ndarray
    reliability: np.ndarray
    image_redshift_mean: np.ndarray
    image_redshift_sigma: np.ndarray
    beta_background_loglike: np.ndarray | None = None


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _best_redshift(row: pd.Series) -> float:
    for column in ("zspec_best", "catalog_z", "zphot_best", "image_zphot_family", "z_source"):
        if column in row:
            value = _finite_float(row[column])
            if value > 0.0:
                return value
    return float("nan")


def _redshift_sigma(row: pd.Series, *, default: float = 0.5) -> float:
    value = _finite_float(row.get("catalog_z_sigma", np.nan))
    if value > 0.0:
        return max(value, 1.0e-3)
    if _finite_float(row.get("zspec_best", np.nan)) > 0.0:
        confidence = str(row.get("zspec_best_confidence", "")).lower()
        rank = _finite_float(row.get("zspec_best_confidence_rank", np.nan))
        if "secure" in confidence or (np.isfinite(rank) and rank >= 3.0):
            return 0.01
        return 0.05
    return default


def _positions_arcsec(images: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if {"x_obs", "y_obs"}.issubset(images.columns):
        return (
            pd.to_numeric(images["x_obs"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(images["y_obs"], errors="coerce").to_numpy(dtype=float),
        )
    if {"x_obs_arcsec", "y_obs_arcsec"}.issubset(images.columns):
        return (
            pd.to_numeric(images["x_obs_arcsec"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(images["y_obs_arcsec"], errors="coerce").to_numpy(dtype=float),
        )
    if {"x", "y"}.issubset(images.columns):
        return (
            pd.to_numeric(images["x"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(images["y"], errors="coerce").to_numpy(dtype=float),
        )
    raise ValueError("images must contain x/y positions as x_obs/y_obs, x_obs_arcsec/y_obs_arcsec, or x/y.")


def _image_sigma_arcsec(images: pd.DataFrame, *, default: float) -> np.ndarray:
    for column in ("sigma_arcsec", "pos_sigma_arcsec", "image_sigma_arcsec"):
        if column in images.columns:
            values = pd.to_numeric(images[column], errors="coerce").to_numpy(dtype=float)
            return np.where(np.isfinite(values) & (values > 0.0), values, float(default))
    return np.full(len(images), float(default), dtype=float)


def _image_redshift_arrays(images: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray([_best_redshift(row) for _, row in images.iterrows()], dtype=float)
    sigmas = np.asarray([_redshift_sigma(row) for _, row in images.iterrows()], dtype=float)
    sigmas = np.where(np.isfinite(sigmas) & (sigmas > 0.0), sigmas, 0.5)
    return means, sigmas


def redshift_grid_from_images(
    images: pd.DataFrame,
    *,
    n_grid: int = 8,
    z_min: float | None = None,
    z_max: float | None = None,
    sigma_padding: float = 2.0,
) -> np.ndarray:
    """Build a compact redshift grid covering image redshift uncertainty."""
    means, sigmas = _image_redshift_arrays(images.reset_index(drop=True))
    finite = np.isfinite(means) & (means > 0.0)
    if not bool(np.any(finite)):
        low = 1.0 if z_min is None else float(z_min)
        high = 6.0 if z_max is None else float(z_max)
    else:
        low = float(np.min(means[finite] - float(sigma_padding) * sigmas[finite])) if z_min is None else float(z_min)
        high = float(np.max(means[finite] + float(sigma_padding) * sigmas[finite])) if z_max is None else float(z_max)
        low = max(0.05, low)
        high = max(low + 0.05, high)
    return np.linspace(low, high, max(2, int(n_grid)), dtype=float)


def precompute_source_plane_cache(
    images: pd.DataFrame,
    ray_shooter: RayShooter,
    *,
    z_grid: np.ndarray | None = None,
    n_redshift_grid: int = 8,
    position_sigma_arcsec: float = 0.15,
) -> SourcePlaneCache:
    """Ray-shoot all images once per redshift grid point."""
    work = images.reset_index(drop=True).copy()
    x_arcsec, y_arcsec = _positions_arcsec(work)
    grid = redshift_grid_from_images(work, n_grid=n_redshift_grid) if z_grid is None else np.asarray(z_grid, dtype=float)
    beta_x = np.empty((grid.size, len(work)), dtype=float)
    beta_y = np.empty((grid.size, len(work)), dtype=float)
    for z_index, z_source in enumerate(grid):
        ray_x, ray_y = ray_shooter(x_arcsec, y_arcsec, float(z_source))
        beta_x[z_index] = np.asarray(ray_x, dtype=float)
        beta_y[z_index] = np.asarray(ray_y, dtype=float)
    beta_background_loglike = np.zeros_like(beta_x, dtype=float)
    for z_index in range(grid.size):
        all_x = beta_x[z_index]
        all_y = beta_y[z_index]
        finite = np.isfinite(all_x) & np.isfinite(all_y)
        if int(np.sum(finite)) <= 1:
            continue
        mean_x = float(np.mean(all_x[finite]))
        mean_y = float(np.mean(all_y[finite]))
        variance_x = max(float(np.var(all_x[finite])), float(position_sigma_arcsec) ** 2, 1.0e-8)
        variance_y = max(float(np.var(all_y[finite])), float(position_sigma_arcsec) ** 2, 1.0e-8)
        beta_background_loglike[z_index] = -0.5 * (
            np.log(2.0 * np.pi * variance_x)
            + np.square(all_x - mean_x) / variance_x
            + np.log(2.0 * np.pi * variance_y)
            + np.square(all_y - mean_y) / variance_y
        )
        beta_background_loglike[z_index, ~finite] = 0.0
    means, sigmas = _image_redshift_arrays(work)
    reliability = (
        pd.to_numeric(work.get("family_reliability", pd.Series(1.0, index=work.index)), errors="coerce")
        .fillna(1.0)
        .clip(0.0, 1.0)
        .to_numpy(dtype=float)
    )
    return SourcePlaneCache(
        z_grid=grid,
        beta_x=beta_x,
        beta_y=beta_y,
        image_sigma_arcsec=_image_sigma_arcsec(work, default=float(position_sigma_arcsec)),
        reliability=reliability,
        image_redshift_mean=means,
        image_redshift_sigma=sigmas,
        beta_background_loglike=beta_background_loglike,
    )


def _normalized_redshift_weights(cache: SourcePlaneCache, members: np.ndarray) -> np.ndarray:
    means = cache.image_redshift_mean[members]
    sigmas = cache.image_redshift_sigma[members]
    finite = np.isfinite(means) & (means > 0.0) & np.isfinite(sigmas) & (sigmas > 0.0)
    if not bool(np.any(finite)):
        return np.full(cache.z_grid.size, 1.0 / cache.z_grid.size, dtype=float)
    dz = cache.z_grid[:, None] - means[finite][None, :]
    log_weights = -0.5 * np.sum(np.square(dz / sigmas[finite][None, :]), axis=1)
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(cache.z_grid.size, 1.0 / cache.z_grid.size, dtype=float)
    return weights / total


def lens_pair_affinity_from_cache(
    cache: SourcePlaneCache,
    *,
    source_scatter_arcsec: float = 0.05,
) -> np.ndarray:
    """Build redshift-marginalized pair affinity from cached source-plane positions."""
    n_images = int(cache.beta_x.shape[1])
    affinity = np.eye(n_images, dtype=float)
    for left in range(n_images):
        for right in range(left + 1, n_images):
            weights = _normalized_redshift_weights(cache, np.asarray([left, right], dtype=int))
            dx = cache.beta_x[:, left] - cache.beta_x[:, right]
            dy = cache.beta_y[:, left] - cache.beta_y[:, right]
            variance = cache.image_sigma_arcsec[left] ** 2 + cache.image_sigma_arcsec[right] ** 2 + float(source_scatter_arcsec) ** 2
            values = np.exp(-0.5 * (np.square(dx) + np.square(dy)) / max(variance, 1.0e-8))
            affinity[left, right] = float(np.sum(weights * values))
            affinity[right, left] = affinity[left, right]
    return np.clip(affinity, 1.0e-6, 1.0 - 1.0e-6)


def combine_catalog_lens_affinity(
    catalog_probability: np.ndarray,
    lens_affinity: np.ndarray,
    *,
    lens_weight: float = 1.0,
) -> np.ndarray:
    """Combine catalog and lens affinities in logit space."""
    catalog = np.asarray(catalog_probability, dtype=float)
    lens = np.asarray(lens_affinity, dtype=float)
    if catalog.shape != lens.shape:
        raise ValueError("catalog_probability and lens_affinity must have matching shape.")
    linear = _pair_logit(catalog) + float(lens_weight) * _pair_logit(lens)
    combined = 1.0 / (1.0 + np.exp(-linear))
    np.fill_diagonal(combined, 1.0)
    return np.clip(combined, 1.0e-6, 1.0 - 1.0e-6)


def _family_redshift(images: pd.DataFrame, member_indices: np.ndarray) -> float:
    redshifts = np.asarray([_best_redshift(images.iloc[int(index)]) for index in member_indices], dtype=float)
    sigmas = np.asarray([_redshift_sigma(images.iloc[int(index)]) for index in member_indices], dtype=float)
    finite = np.isfinite(redshifts) & (redshifts > 0.0) & np.isfinite(sigmas) & (sigmas > 0.0)
    if not bool(np.any(finite)):
        return float("nan")
    weights = 1.0 / np.square(np.maximum(sigmas[finite], 1.0e-3))
    return float(np.average(redshifts[finite], weights=weights))


def source_plane_family_loglike(
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    sigma_arcsec: np.ndarray | float,
    *,
    source_scatter_arcsec: float = 0.05,
    reliability: np.ndarray | float | None = None,
    outlier_sigma_arcsec: float | None = None,
    include_log_normalization: bool = False,
) -> tuple[float, dict[str, float]]:
    """Score whether ray-shot image positions share a common source position."""
    beta_x = np.asarray(beta_x, dtype=float).reshape(-1)
    beta_y = np.asarray(beta_y, dtype=float).reshape(-1)
    if beta_x.size != beta_y.size:
        raise ValueError("beta_x and beta_y must have matching length.")
    n_images = int(beta_x.size)
    if n_images <= 1:
        return 0.0, {
            "source_x": float(beta_x[0]) if n_images == 1 and np.isfinite(beta_x[0]) else float("nan"),
            "source_y": float(beta_y[0]) if n_images == 1 and np.isfinite(beta_y[0]) else float("nan"),
            "source_plane_rms": 0.0,
            "n_images": float(n_images),
        }

    sigma = np.asarray(sigma_arcsec, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(n_images, float(sigma), dtype=float)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, np.nan)
    variance = np.square(sigma) + float(source_scatter_arcsec) ** 2
    finite = np.isfinite(beta_x) & np.isfinite(beta_y) & np.isfinite(variance) & (variance > 0.0)
    if int(np.sum(finite)) <= 1:
        return -np.inf, {
            "source_x": float("nan"),
            "source_y": float("nan"),
            "source_plane_rms": float("nan"),
            "n_images": float(n_images),
        }

    precision = 1.0 / variance[finite]
    source_x = float(np.average(beta_x[finite], weights=precision))
    source_y = float(np.average(beta_y[finite], weights=precision))
    residual2 = np.square(beta_x[finite] - source_x) + np.square(beta_y[finite] - source_y)
    gaussian_loglike = -0.5 * residual2 * precision
    if include_log_normalization:
        gaussian_loglike = gaussian_loglike - np.log(2.0 * np.pi * variance[finite])

    if reliability is None:
        loglike = float(np.sum(gaussian_loglike))
    else:
        rel = np.asarray(reliability, dtype=float)
        if rel.ndim == 0:
            rel = np.full(n_images, float(rel), dtype=float)
        rel = np.clip(np.where(np.isfinite(rel), rel, 1.0), 0.0, 1.0)[finite]
        if outlier_sigma_arcsec is None:
            outlier_sigma_arcsec = max(1.0, 10.0 * float(source_scatter_arcsec))
        outlier_var = float(outlier_sigma_arcsec) ** 2
        outlier_loglike = -0.5 * residual2 / outlier_var
        if include_log_normalization:
            outlier_loglike = outlier_loglike - np.log(2.0 * np.pi * outlier_var)
        loglike = float(
            np.sum(
                np.logaddexp(
                    np.log(np.clip(rel, 1.0e-6, 1.0)) + gaussian_loglike,
                    np.log(np.clip(1.0 - rel, 1.0e-6, 1.0)) + outlier_loglike,
                )
            )
        )

    rms = float(np.sqrt(np.mean(residual2))) if residual2.size else float("nan")
    return loglike, {
        "source_x": source_x,
        "source_y": source_y,
        "source_plane_rms": rms,
        "n_images": float(n_images),
    }


def _source_plane_background_loglike(
    cache: SourcePlaneCache,
    z_index: int,
    members: np.ndarray,
    *,
    source_scatter_arcsec: float,
) -> float:
    """Independent background likelihood for source-plane positions at one redshift."""
    members = np.asarray(members, dtype=int).reshape(-1)
    if members.size <= 1:
        return 0.0
    if cache.beta_background_loglike is not None:
        return float(np.sum(cache.beta_background_loglike[int(z_index), members]))
    all_x = np.asarray(cache.beta_x[int(z_index)], dtype=float)
    all_y = np.asarray(cache.beta_y[int(z_index)], dtype=float)
    finite = np.isfinite(all_x) & np.isfinite(all_y)
    if int(np.sum(finite)) <= 1:
        return 0.0
    mean_x = float(np.mean(all_x[finite]))
    mean_y = float(np.mean(all_y[finite]))
    variance_x = max(float(np.var(all_x[finite])), float(source_scatter_arcsec) ** 2, 1.0e-8)
    variance_y = max(float(np.var(all_y[finite])), float(source_scatter_arcsec) ** 2, 1.0e-8)
    x = all_x[members]
    y = all_y[members]
    member_finite = np.isfinite(x) & np.isfinite(y)
    if not bool(np.any(member_finite)):
        return 0.0
    x = x[member_finite]
    y = y[member_finite]
    log_x = np.log(2.0 * np.pi * variance_x) + np.square(x - mean_x) / variance_x
    log_y = np.log(2.0 * np.pi * variance_y) + np.square(y - mean_y) / variance_y
    return float(-0.5 * np.sum(log_x + log_y))


def score_source_plane_partitions(
    partitions: np.ndarray,
    beta_x_by_partition: np.ndarray,
    beta_y_by_partition: np.ndarray,
    *,
    sigma_arcsec: np.ndarray | float = 0.15,
    source_scatter_arcsec: float = 0.05,
    reliability: np.ndarray | float | None = None,
    outlier_sigma_arcsec: float | None = None,
) -> SourcePlaneScoreResult:
    """Score partitions from per-partition, per-image source-plane positions."""
    partition_array = np.asarray(partitions, dtype=int)
    if partition_array.ndim == 1:
        partition_array = partition_array.reshape(1, -1)
    beta_x = np.asarray(beta_x_by_partition, dtype=float)
    beta_y = np.asarray(beta_y_by_partition, dtype=float)
    if beta_x.shape != partition_array.shape or beta_y.shape != partition_array.shape:
        raise ValueError("beta arrays must have shape (n_partitions, n_images).")

    sigma = np.asarray(sigma_arcsec, dtype=float)
    reliability_array = None if reliability is None else np.asarray(reliability, dtype=float)
    log_scores = np.empty(partition_array.shape[0], dtype=float)
    rows: list[dict[str, Any]] = []
    for partition_index, assignment in enumerate(partition_array):
        canonical = canonicalize_partition(assignment)
        score = 0.0
        for family_label in np.unique(canonical):
            members = np.flatnonzero(canonical == family_label)
            member_sigma = sigma if sigma.ndim == 0 else sigma[members]
            member_reliability = None if reliability_array is None else (reliability_array if reliability_array.ndim == 0 else reliability_array[members])
            family_score, diagnostics = source_plane_family_loglike(
                beta_x[partition_index, members],
                beta_y[partition_index, members],
                member_sigma,
                source_scatter_arcsec=source_scatter_arcsec,
                reliability=member_reliability,
                outlier_sigma_arcsec=outlier_sigma_arcsec,
            )
            score += family_score
            rows.append(
                {
                    "partition_index": int(partition_index),
                    "family_label": int(family_label),
                    "family_log_score": float(family_score),
                    **diagnostics,
                }
            )
        log_scores[partition_index] = float(score)
    return SourcePlaneScoreResult(log_scores=log_scores, family_table=pd.DataFrame(rows))


def score_cached_source_plane_partitions(
    partitions: np.ndarray,
    cache: SourcePlaneCache,
    *,
    source_scatter_arcsec: float = 0.05,
    outlier_sigma_arcsec: float | None = None,
    score_mode: str = "raw",
) -> SourcePlaneScoreResult:
    """Score partitions using cached beta[z, image] and redshift marginalization."""
    if score_mode not in {"raw", "likelihood_ratio"}:
        raise ValueError("score_mode must be 'raw' or 'likelihood_ratio'.")
    partition_array = np.asarray(partitions, dtype=int)
    if partition_array.ndim == 1:
        partition_array = partition_array.reshape(1, -1)
    log_scores = np.empty(partition_array.shape[0], dtype=float)
    rows: list[dict[str, Any]] = []
    family_cache: dict[tuple[int, ...], tuple[float, dict[str, float]]] = {}
    for partition_index, assignment in enumerate(partition_array):
        canonical = canonicalize_partition(assignment)
        partition_score = 0.0
        for family_label in np.unique(canonical):
            members = np.flatnonzero(canonical == family_label)
            member_key = tuple(int(item) for item in members.tolist())
            cached = family_cache.get(member_key)
            if cached is not None:
                family_score, diagnostics = cached
            elif members.size <= 1:
                family_score = 0.0
                diagnostics = {
                    "source_x": float("nan"),
                    "source_y": float("nan"),
                    "source_plane_rms": 0.0,
                    "n_images": float(members.size),
                }
                family_cache[member_key] = (family_score, diagnostics)
            else:
                redshift_weights = _normalized_redshift_weights(cache, members)
                z_scores = np.empty(cache.z_grid.size, dtype=float)
                z_rms = np.empty(cache.z_grid.size, dtype=float)
                for z_index in range(cache.z_grid.size):
                    same_score, diag = source_plane_family_loglike(
                        cache.beta_x[z_index, members],
                        cache.beta_y[z_index, members],
                        cache.image_sigma_arcsec[members],
                        source_scatter_arcsec=source_scatter_arcsec,
                        reliability=cache.reliability[members],
                        outlier_sigma_arcsec=outlier_sigma_arcsec,
                        include_log_normalization=score_mode == "likelihood_ratio",
                    )
                    if score_mode == "likelihood_ratio":
                        same_score -= _source_plane_background_loglike(
                            cache,
                            z_index,
                            members,
                            source_scatter_arcsec=source_scatter_arcsec,
                        )
                    z_scores[z_index] = same_score
                    z_rms[z_index] = float(diag["source_plane_rms"])
                finite = np.isfinite(z_scores) & (redshift_weights > 0.0)
                if bool(np.any(finite)):
                    terms = np.log(redshift_weights[finite]) + z_scores[finite]
                    max_term = float(np.max(terms))
                    family_score = max_term + float(np.log(np.sum(np.exp(terms - max_term))))
                    best_z_index = int(np.argmax(terms))
                    diagnostics = {
                        "source_x": float("nan"),
                        "source_y": float("nan"),
                        "source_plane_rms": float(z_rms[np.flatnonzero(finite)[best_z_index]]),
                        "n_images": float(members.size),
                        "best_z": float(cache.z_grid[np.flatnonzero(finite)[best_z_index]]),
                    }
                else:
                    family_score = -np.inf
                    diagnostics = {
                        "source_x": float("nan"),
                        "source_y": float("nan"),
                        "source_plane_rms": float("nan"),
                        "n_images": float(members.size),
                        "best_z": float("nan"),
                    }
                family_cache[member_key] = (float(family_score), diagnostics)
            partition_score += float(family_score)
            rows.append(
                {
                    "partition_index": int(partition_index),
                    "family_label": int(family_label),
                    "family_log_score": float(family_score),
                    **diagnostics,
                }
            )
        log_scores[partition_index] = float(partition_score)
    return SourcePlaneScoreResult(log_scores=log_scores, family_table=pd.DataFrame(rows))


def cached_source_plane_score_callback(
    cache: SourcePlaneCache,
    *,
    source_scatter_arcsec: float = 0.05,
    outlier_sigma_arcsec: float | None = None,
    score_mode: str = "raw",
) -> Callable[[np.ndarray, pd.DataFrame], np.ndarray]:
    """Build a partition-score callback from a precomputed source-plane cache."""
    def callback(partitions: np.ndarray, _pair_table: pd.DataFrame) -> np.ndarray:
        return score_cached_source_plane_partitions(
            partitions,
            cache,
            source_scatter_arcsec=source_scatter_arcsec,
            outlier_sigma_arcsec=outlier_sigma_arcsec,
            score_mode=score_mode,
        ).log_scores

    return callback


def cached_lens_proposal_matrix_callback(
    lens_affinity: np.ndarray,
    *,
    lens_weight: float = 1.0,
) -> Callable[[np.ndarray, pd.DataFrame], list[np.ndarray]]:
    """Build proposal matrices from cached lens affinity and catalog affinity."""
    lens = np.asarray(lens_affinity, dtype=float)

    def callback(catalog_probability: np.ndarray, _pair_table: pd.DataFrame) -> list[np.ndarray]:
        return [
            lens,
            combine_catalog_lens_affinity(catalog_probability, lens, lens_weight=lens_weight),
        ]

    return callback


def _connected_components_from_adjacency(adjacency: np.ndarray) -> list[list[int]]:
    n_items = int(adjacency.shape[0])
    seen = np.zeros(n_items, dtype=bool)
    components: list[list[int]] = []
    for start in range(n_items):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            item = stack.pop()
            component.append(item)
            neighbors = np.flatnonzero(adjacency[item] & ~seen)
            seen[neighbors] = True
            stack.extend(int(neighbor) for neighbor in neighbors)
        components.append(component)
    return components


def lens_source_cluster_partitions(
    cache: SourcePlaneCache,
    *,
    catalog_probability: np.ndarray | None = None,
    beta_radius_grid: tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.18),
    min_cluster_size: int = 2,
    min_redshift_weight: float = 0.02,
    source_scatter_arcsec: float = 0.05,
    catalog_weight: float = 1.0,
    beam_width: int = 50,
    max_cluster_hypotheses: int = 400,
) -> np.ndarray:
    """Propose full partitions by beam-assembling source-plane cluster hypotheses."""
    n_images = int(cache.beta_x.shape[1])
    if n_images == 0:
        return np.empty((0, 0), dtype=int)
    catalog = None if catalog_probability is None else np.asarray(catalog_probability, dtype=float)
    if catalog is not None and catalog.shape != (n_images, n_images):
        raise ValueError("catalog_probability must have shape (n_images, n_images).")
    catalog_logit = None if catalog is None else _pair_logit(np.clip(catalog, 1.0e-6, 1.0 - 1.0e-6))
    redshift_weights = np.vstack([
        _normalized_redshift_weights(cache, np.asarray([image_index], dtype=int))
        for image_index in range(n_images)
    ])
    candidates: dict[tuple[int, ...], np.ndarray] = {}
    cluster_pool: dict[tuple[int, ...], float] = {}

    def add(assignment: np.ndarray) -> None:
        canonical = canonicalize_partition(assignment)
        if canonical.size == n_images:
            candidates[tuple(canonical.tolist())] = canonical

    def cluster_lens_score(members: np.ndarray) -> float:
        redshift = _normalized_redshift_weights(cache, members)
        z_scores = np.empty(cache.z_grid.size, dtype=float)
        for z_index in range(cache.z_grid.size):
            z_scores[z_index], _diag = source_plane_family_loglike(
                cache.beta_x[z_index, members],
                cache.beta_y[z_index, members],
                cache.image_sigma_arcsec[members],
                source_scatter_arcsec=source_scatter_arcsec,
                reliability=cache.reliability[members],
            )
        finite = np.isfinite(z_scores) & (redshift > 0.0)
        if not bool(np.any(finite)):
            return -np.inf
        terms = np.log(redshift[finite]) + z_scores[finite]
        max_term = float(np.max(terms))
        return max_term + float(np.log(np.sum(np.exp(terms - max_term))))

    def cluster_catalog_score(members: np.ndarray) -> float:
        if catalog_logit is None or members.size < 2:
            return 0.0
        sub = catalog_logit[np.ix_(members, members)]
        upper = np.triu_indices(members.size, k=1)
        values = sub[upper]
        return float(np.sum(values)) if values.size else 0.0

    def cluster_score(component: list[int]) -> float:
        members = np.asarray(sorted(component), dtype=int)
        lens_score = cluster_lens_score(members)
        if not np.isfinite(lens_score):
            return -np.inf
        size_bonus = float(np.log(max(members.size, 1)))
        return float(lens_score + float(catalog_weight) * cluster_catalog_score(members) + size_bonus)

    def remember_cluster(component: list[int]) -> None:
        if len(component) < int(min_cluster_size):
            return
        members = tuple(sorted(int(item) for item in component))
        score = cluster_score(list(members))
        if not np.isfinite(score):
            return
        previous = cluster_pool.get(members)
        if previous is None or score > previous:
            cluster_pool[members] = score

    for radius in beta_radius_grid:
        radius = float(radius)
        for z_index in range(cache.z_grid.size):
            eligible = redshift_weights[:, z_index] >= float(min_redshift_weight)
            if int(np.sum(eligible)) < int(min_cluster_size):
                continue
            coords = np.column_stack([cache.beta_x[z_index], cache.beta_y[z_index]])
            delta = coords[:, None, :] - coords[None, :, :]
            distance = np.sqrt(np.sum(np.square(delta), axis=2))
            adjacency = (distance <= radius) & eligible[:, None] & eligible[None, :]
            np.fill_diagonal(adjacency, True)
            components = _connected_components_from_adjacency(adjacency)
            components = [component for component in components if len(component) >= int(min_cluster_size)]
            if not components:
                continue
            for component in components:
                remember_cluster(component)

    sorted_clusters = sorted(cluster_pool.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    sorted_clusters = sorted_clusters[: max(1, int(max_cluster_hypotheses))]

    def mask_for(members: tuple[int, ...]) -> int:
        mask = 0
        for member in members:
            mask |= 1 << int(member)
        return mask

    cluster_items = [(members, mask_for(members), float(score)) for members, score in sorted_clusters]
    beam: list[tuple[float, int, tuple[tuple[int, ...], ...]]] = [(0.0, 0, tuple())]
    for members, mask, score in cluster_items:
        additions: list[tuple[float, int, tuple[tuple[int, ...], ...]]] = []
        for state_score, used_mask, selected in beam:
            if used_mask & mask:
                continue
            additions.append((state_score + score, used_mask | mask, selected + (members,)))
        if not additions:
            continue
        combined = beam + additions
        deduped: dict[int, tuple[float, int, tuple[tuple[int, ...], ...]]] = {}
        for state in combined:
            current = deduped.get(state[1])
            if current is None or state[0] > current[0]:
                deduped[state[1]] = state
        beam = sorted(deduped.values(), key=lambda item: (-item[0], -int.bit_count(item[1]), item[2]))[: max(1, int(beam_width))]

    for _score, _used_mask, selected in beam:
        if not selected:
            continue
        assignment = np.arange(n_images, dtype=int)
        next_label = n_images
        for members in selected:
            assignment[np.asarray(members, dtype=int)] = next_label
            next_label += 1
        add(assignment)

    return np.asarray(list(candidates.values()), dtype=int)


def cached_lens_cluster_partitions_callback(
    cache: SourcePlaneCache,
    *,
    beta_radius_grid: tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.18),
    min_cluster_size: int = 2,
    min_redshift_weight: float = 0.02,
    source_scatter_arcsec: float = 0.05,
    catalog_weight: float = 1.0,
    beam_width: int = 50,
    max_cluster_hypotheses: int = 400,
) -> Callable[[np.ndarray, pd.DataFrame], np.ndarray]:
    """Build a callback that injects lens-first source-plane cluster partitions."""
    def callback(catalog_probability: np.ndarray, _pair_table: pd.DataFrame) -> np.ndarray:
        return lens_source_cluster_partitions(
            cache,
            catalog_probability=catalog_probability,
            beta_radius_grid=beta_radius_grid,
            min_cluster_size=min_cluster_size,
            min_redshift_weight=min_redshift_weight,
            source_scatter_arcsec=source_scatter_arcsec,
            catalog_weight=catalog_weight,
            beam_width=beam_width,
            max_cluster_hypotheses=max_cluster_hypotheses,
        )

    return callback


def source_plane_score_callback_from_ray_shooter(
    images: pd.DataFrame,
    ray_shooter: RayShooter,
    *,
    position_sigma_arcsec: float = 0.15,
    source_scatter_arcsec: float = 0.05,
    outlier_sigma_arcsec: float | None = None,
) -> Callable[[np.ndarray, pd.DataFrame], np.ndarray]:
    """Build a partition-score callback from a lens ray-shooting function.

    ``ray_shooter`` receives ``(x_arcsec, y_arcsec, z_source)`` and returns
    ray-shot ``(beta_x, beta_y)`` arrays for that source redshift.
    """
    work = images.reset_index(drop=True).copy()
    x_arcsec, y_arcsec = _positions_arcsec(work)
    sigma = _image_sigma_arcsec(work, default=float(position_sigma_arcsec))
    reliability = (
        pd.to_numeric(work.get("family_reliability", pd.Series(1.0, index=work.index)), errors="coerce")
        .fillna(1.0)
        .clip(0.0, 1.0)
        .to_numpy(dtype=float)
    )

    def callback(partitions: np.ndarray, _pair_table: pd.DataFrame) -> np.ndarray:
        partition_array = np.asarray(partitions, dtype=int)
        if partition_array.ndim == 1:
            partition_array = partition_array.reshape(1, -1)
        beta_x = np.full(partition_array.shape, np.nan, dtype=float)
        beta_y = np.full(partition_array.shape, np.nan, dtype=float)
        for partition_index, assignment in enumerate(partition_array):
            canonical = canonicalize_partition(assignment)
            for family_label in np.unique(canonical):
                members = np.flatnonzero(canonical == family_label)
                z_source = _family_redshift(work, members)
                if not np.isfinite(z_source):
                    continue
                family_beta_x, family_beta_y = ray_shooter(x_arcsec[members], y_arcsec[members], float(z_source))
                beta_x[partition_index, members] = np.asarray(family_beta_x, dtype=float)
                beta_y[partition_index, members] = np.asarray(family_beta_y, dtype=float)
        return score_source_plane_partitions(
            partition_array,
            beta_x,
            beta_y,
            sigma_arcsec=sigma,
            source_scatter_arcsec=source_scatter_arcsec,
            reliability=reliability,
            outlier_sigma_arcsec=outlier_sigma_arcsec,
        ).log_scores

    return callback


def marginalized_source_plane_score_callback_from_ray_shooters(
    images: pd.DataFrame,
    ray_shooters: list[RayShooter],
    *,
    position_sigma_arcsec: float = 0.15,
    source_scatter_arcsec: float = 0.05,
    outlier_sigma_arcsec: float | None = None,
) -> Callable[[np.ndarray, pd.DataFrame], np.ndarray]:
    """Build a source-plane callback marginalized over lens posterior samples."""
    if not ray_shooters:
        raise ValueError("At least one ray_shooter is required.")
    callbacks = [
        source_plane_score_callback_from_ray_shooter(
            images,
            ray_shooter,
            position_sigma_arcsec=position_sigma_arcsec,
            source_scatter_arcsec=source_scatter_arcsec,
            outlier_sigma_arcsec=outlier_sigma_arcsec,
        )
        for ray_shooter in ray_shooters
    ]

    def callback(partitions: np.ndarray, pair_table: pd.DataFrame) -> np.ndarray:
        sample_scores = np.vstack([item(partitions, pair_table) for item in callbacks])
        finite = np.isfinite(sample_scores)
        output = np.full(sample_scores.shape[1], -np.inf, dtype=float)
        for partition_index in range(sample_scores.shape[1]):
            values = sample_scores[finite[:, partition_index], partition_index]
            if values.size:
                max_value = float(np.max(values))
                output[partition_index] = max_value + float(np.log(np.mean(np.exp(values - max_value))))
        return output

    return callback
