"""Probability-conserving candidate partition engine for lensed image families."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


LOGIT_EPS = 1.0e-6
PartitionScoreCallback = Callable[[np.ndarray, pd.DataFrame], np.ndarray]
ProposalMatrixCallback = Callable[[np.ndarray, pd.DataFrame], list[np.ndarray]]
ProposalPartitionsCallback = Callable[[np.ndarray, pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class LogisticFit:
    """MAP logistic pair model fit."""

    coefficients: np.ndarray
    feature_names: list[str]
    success: bool
    objective: float


@dataclass(frozen=True)
class PartitionResult:
    """Weighted candidate partitions and derived conserved probabilities."""

    partitions: np.ndarray
    log_scores: np.ndarray
    weights: np.ndarray
    pair_probability: np.ndarray
    pair_table: pd.DataFrame
    map_assignment: np.ndarray
    logistic_fit: LogisticFit
    pair_score_mode: str = "sum"


def canonicalize_partition(assignment: np.ndarray | list[int]) -> np.ndarray:
    """Relabel partition ids by first occurrence."""
    values = np.asarray(assignment, dtype=int).reshape(-1)
    mapping: dict[int, int] = {}
    canonical = np.empty_like(values)
    for index, value in enumerate(values.tolist()):
        if value not in mapping:
            mapping[value] = len(mapping)
        canonical[index] = mapping[value]
    return canonical


def same_family_matrix(assignment: np.ndarray | list[int]) -> np.ndarray:
    """Return an NxN boolean matrix indicating common family membership."""
    canonical = canonicalize_partition(assignment)
    return canonical[:, None] == canonical[None, :]


def weighted_pair_probability(partitions: np.ndarray | list[np.ndarray], weights: np.ndarray | list[float]) -> np.ndarray:
    """Compute pair probabilities from weighted valid partitions."""
    partition_array = np.asarray(partitions, dtype=int)
    if partition_array.ndim == 1:
        partition_array = partition_array.reshape(1, -1)
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    if partition_array.shape[0] != weight_array.size:
        raise ValueError("Number of partitions must match number of weights.")
    if partition_array.shape[0] == 0:
        return np.empty((0, 0), dtype=float)
    total = float(np.sum(weight_array))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Partition weights must have a positive finite sum.")
    normalized = weight_array / total
    n_images = int(partition_array.shape[1])
    probability = np.zeros((n_images, n_images), dtype=float)
    for assignment, weight in zip(partition_array, normalized):
        probability += float(weight) * same_family_matrix(assignment).astype(float)
    return probability


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(
        values >= 0.0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def _logit(probability: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(clipped) - np.log1p(-clipped)


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _best_redshift(row: pd.Series) -> float:
    for column in ("zspec_best", "catalog_z", "zphot_best", "image_zphot_family"):
        if column in row:
            value = _finite_float(row[column])
            if value > 0.0:
                return value
    return float("nan")


def _redshift_sigma(row: pd.Series, *, default: float = 0.5) -> float:
    if "catalog_z_sigma" in row:
        value = _finite_float(row["catalog_z_sigma"])
        if value > 0.0:
            return max(value, 1.0e-3)
    if "zspec_best" in row and _finite_float(row["zspec_best"]) > 0.0:
        confidence = str(row.get("zspec_best_confidence", "")).lower()
        rank = _finite_float(row.get("zspec_best_confidence_rank", np.nan))
        if "secure" in confidence or (np.isfinite(rank) and rank >= 3.0):
            return 0.01
        return 0.05
    if "zphot_best" in row and _finite_float(row["zphot_best"]) > 0.0:
        low = _finite_float(row.get("pagul_zpdf_low", np.nan))
        high = _finite_float(row.get("pagul_zpdf_high", np.nan))
        if np.isfinite(low) and np.isfinite(high) and high > low:
            return max(0.5 * (high - low), 0.05)
        return default
    return default


def _positions_arcsec(images: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if {"x_obs", "y_obs"}.issubset(images.columns):
        return (
            pd.to_numeric(images["x_obs"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(images["y_obs"], errors="coerce").to_numpy(dtype=float),
        )
    if {"x", "y"}.issubset(images.columns):
        return (
            pd.to_numeric(images["x"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(images["y"], errors="coerce").to_numpy(dtype=float),
        )
    if {"ra", "dec"}.issubset(images.columns):
        ra = pd.to_numeric(images["ra"], errors="coerce").to_numpy(dtype=float)
        dec = pd.to_numeric(images["dec"], errors="coerce").to_numpy(dtype=float)
        ra0 = float(np.nanmedian(ra)) if np.isfinite(ra).any() else 0.0
        dec0 = float(np.nanmedian(dec)) if np.isfinite(dec).any() else 0.0
        x = (ra - ra0) * math.cos(math.radians(dec0)) * 3600.0
        y = (dec - dec0) * 3600.0
        return x, y
    return np.full(len(images), np.nan, dtype=float), np.full(len(images), np.nan, dtype=float)


def _id_values(images: pd.DataFrame) -> list[str]:
    for column in ("object_id", "image_label", "id"):
        if column in images.columns:
            return images[column].fillna("").astype(str).tolist()
    return [str(index) for index in range(len(images))]


def _mag_columns(images: pd.DataFrame) -> list[str]:
    return sorted(column for column in images.columns if str(column).startswith("mag_"))


def _color_vector(row: pd.Series, mag_columns: list[str]) -> np.ndarray:
    values = np.asarray([_finite_float(row.get(column, np.nan)) for column in mag_columns], dtype=float)
    finite = np.isfinite(values)
    if int(np.sum(finite)) < 2:
        return np.full_like(values, np.nan, dtype=float)
    centered = values.copy()
    centered[finite] -= float(np.nanmean(values[finite]))
    return centered


def _color_rms(left: pd.Series, right: pd.Series, mag_columns: list[str]) -> tuple[float, int]:
    if not mag_columns:
        return float("nan"), 0
    left_colors = _color_vector(left, mag_columns)
    right_colors = _color_vector(right, mag_columns)
    finite = np.isfinite(left_colors) & np.isfinite(right_colors)
    if int(np.sum(finite)) < 2:
        return float("nan"), int(np.sum(finite))
    diff = left_colors[finite] - right_colors[finite]
    return float(np.sqrt(np.mean(np.square(diff)))), int(np.sum(finite))


def build_pair_table(images: pd.DataFrame) -> pd.DataFrame:
    """Build pair features used by the partition engine."""
    work = images.reset_index(drop=True).copy()
    n_images = int(len(work))
    ids = _id_values(work)
    x_arcsec, y_arcsec = _positions_arcsec(work)
    redshifts = np.asarray([_best_redshift(row) for _, row in work.iterrows()], dtype=float)
    redshift_sigmas = np.asarray([_redshift_sigma(row) for _, row in work.iterrows()], dtype=float)
    reliability = (
        pd.to_numeric(work.get("family_reliability", pd.Series(1.0, index=work.index)), errors="coerce")
        .fillna(1.0)
        .clip(0.0, 1.0)
        .to_numpy(dtype=float)
    )
    mag_columns = _mag_columns(work)

    rows: list[dict[str, Any]] = []
    radii = np.hypot(x_arcsec, y_arcsec)
    angles = np.arctan2(y_arcsec, x_arcsec)
    for left_index in range(n_images):
        left = work.iloc[left_index]
        for right_index in range(left_index + 1, n_images):
            right = work.iloc[right_index]
            separation = math.hypot(float(x_arcsec[left_index] - x_arcsec[right_index]), float(y_arcsec[left_index] - y_arcsec[right_index]))
            if not np.isfinite(separation):
                separation = float("nan")
            radius_left = float(radii[left_index])
            radius_right = float(radii[right_index])
            radius_scale = max(float(np.nanmedian(radii[np.isfinite(radii)])) if np.isfinite(radii).any() else 1.0, 1.0)
            radius_similarity = (
                math.exp(-abs(radius_left - radius_right) / radius_scale)
                if np.isfinite(radius_left) and np.isfinite(radius_right)
                else 0.5
            )
            angle_delta = abs(float(np.angle(np.exp(1j * (angles[left_index] - angles[right_index])))))
            opposite_side_score = 0.5 * (1.0 - math.cos(angle_delta)) if np.isfinite(angle_delta) else 0.5
            z_delta = abs(redshifts[left_index] - redshifts[right_index]) if np.isfinite(redshifts[[left_index, right_index]]).all() else float("nan")
            z_scale = math.sqrt(redshift_sigmas[left_index] ** 2 + redshift_sigmas[right_index] ** 2)
            z_similarity = math.exp(-0.5 * (z_delta / max(z_scale, 1.0e-3)) ** 2) if np.isfinite(z_delta) else 0.5
            color_rms, n_common_bands = _color_rms(left, right, mag_columns)
            color_similarity = math.exp(-float(color_rms)) if np.isfinite(color_rms) else 0.5
            size_delta = abs(_finite_float(left.get("image_size_arcsec", np.nan)) - _finite_float(right.get("image_size_arcsec", np.nan)))
            size_similarity = math.exp(-size_delta) if np.isfinite(size_delta) else 0.5
            ellipticity_delta = abs(_finite_float(left.get("image_ellipticity", np.nan)) - _finite_float(right.get("image_ellipticity", np.nan)))
            ellipticity_similarity = math.exp(-ellipticity_delta) if np.isfinite(ellipticity_delta) else 0.5
            provenance_match = float(str(left.get("object_source", "")) == str(right.get("object_source", "")) and str(left.get("object_source", "")) != "")
            family_left = str(left.get("family_id", "")).strip()
            family_right = str(right.get("family_id", "")).strip()
            same_input_family = bool(family_left and family_right and family_left == family_right)
            rows.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_object_id": ids[left_index],
                    "right_object_id": ids[right_index],
                    "separation_arcsec": separation,
                    "radius_similarity": radius_similarity,
                    "opposite_side_score": opposite_side_score,
                    "z_delta": z_delta,
                    "z_sigma_combined": z_scale,
                    "z_similarity": z_similarity,
                    "color_rms": color_rms,
                    "color_similarity": color_similarity,
                    "n_common_bands": n_common_bands,
                    "size_similarity": size_similarity,
                    "ellipticity_similarity": ellipticity_similarity,
                    "reliability_mean": float(0.5 * (reliability[left_index] + reliability[right_index])),
                    "same_object_source": provenance_match,
                    "same_input_family": same_input_family,
                }
            )
    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "intercept",
    "local_separation",
    "radius_similarity",
    "opposite_side_score",
    "z_similarity",
    "color_similarity",
    "size_similarity",
    "ellipticity_similarity",
    "reliability_mean",
    "same_object_source",
]


def feature_matrix(pair_table: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return logistic feature matrix and feature names."""
    if pair_table.empty:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=float), list(FEATURE_COLUMNS)
    separation = pd.to_numeric(pair_table["separation_arcsec"], errors="coerce").to_numpy(dtype=float)
    local_separation = 1.0 / (1.0 + np.where(np.isfinite(separation), np.maximum(separation, 0.0), 100.0) / 10.0)
    columns = {
        "intercept": np.ones(len(pair_table), dtype=float),
        "local_separation": local_separation,
        "radius_similarity": pd.to_numeric(pair_table["radius_similarity"], errors="coerce").fillna(0.5).to_numpy(dtype=float),
        "opposite_side_score": pd.to_numeric(pair_table["opposite_side_score"], errors="coerce").fillna(0.5).to_numpy(dtype=float),
        "z_similarity": pd.to_numeric(pair_table["z_similarity"], errors="coerce").fillna(0.5).to_numpy(dtype=float),
        "color_similarity": pd.to_numeric(pair_table["color_similarity"], errors="coerce").fillna(0.5).to_numpy(dtype=float),
        "size_similarity": pd.to_numeric(pair_table["size_similarity"], errors="coerce").fillna(0.5).to_numpy(dtype=float),
        "ellipticity_similarity": pd.to_numeric(pair_table["ellipticity_similarity"], errors="coerce").fillna(0.5).to_numpy(dtype=float),
        "reliability_mean": pd.to_numeric(pair_table["reliability_mean"], errors="coerce").fillna(1.0).to_numpy(dtype=float),
        "same_object_source": pd.to_numeric(pair_table["same_object_source"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
    }
    return np.column_stack([columns[name] for name in FEATURE_COLUMNS]), list(FEATURE_COLUMNS)


def create_anchor_labels(
    pair_table: pd.DataFrame,
    *,
    incompatible_z_delta: float = 0.8,
    incompatible_z_sigma: float = 3.0,
    incompatible_color_rms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create high-confidence pair labels and weights from catalog evidence."""
    labels = np.full(len(pair_table), np.nan, dtype=float)
    weights = np.zeros(len(pair_table), dtype=float)
    if pair_table.empty:
        return labels, weights

    same_family = pair_table.get("same_input_family", pd.Series(False, index=pair_table.index)).fillna(False).astype(bool).to_numpy()
    labels[same_family] = 1.0
    weights[same_family] = np.maximum(weights[same_family], 8.0)

    z_delta = pd.to_numeric(pair_table.get("z_delta", pd.Series(np.nan, index=pair_table.index)), errors="coerce").to_numpy(dtype=float)
    z_sigma = pd.to_numeric(pair_table.get("z_sigma_combined", pd.Series(np.nan, index=pair_table.index)), errors="coerce").to_numpy(dtype=float)
    z_significance = z_delta / np.maximum(z_sigma, 1.0e-3)
    z_bad = np.isfinite(z_delta) & np.isfinite(z_significance) & (z_delta >= float(incompatible_z_delta)) & (z_significance >= float(incompatible_z_sigma))
    labels[z_bad] = 0.0
    weights[z_bad] = np.maximum(weights[z_bad], 12.0)

    color_rms = pd.to_numeric(pair_table.get("color_rms", pd.Series(np.nan, index=pair_table.index)), errors="coerce").to_numpy(dtype=float)
    n_common = pd.to_numeric(pair_table.get("n_common_bands", pd.Series(0, index=pair_table.index)), errors="coerce").fillna(0).to_numpy(dtype=float)
    color_bad = np.isfinite(color_rms) & (n_common >= 3) & (color_rms >= float(incompatible_color_rms))
    unset = weights <= 0.0
    labels[color_bad & unset] = 0.0
    weights[color_bad & unset] = 5.0
    return labels, weights


def fit_logistic_map(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    gaussian_prior_sigma: float = 2.0,
    initial_coefficients: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> LogisticFit:
    """Fit MAP logistic regression with soft labels and a Gaussian prior."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("x must be a 2D feature matrix.")
    if x.shape[0] != y.size:
        raise ValueError("x and y must have matching rows.")
    if x.shape[0] == 0:
        coefficients = np.zeros(x.shape[1], dtype=float)
        return LogisticFit(coefficients, feature_names or [f"x{index}" for index in range(x.shape[1])], True, 0.0)
    weights = np.ones(y.size, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    y = np.clip(y, 0.0, 1.0)
    sigma = max(float(gaussian_prior_sigma), 1.0e-6)
    initial = np.zeros(x.shape[1], dtype=float) if initial_coefficients is None else np.asarray(initial_coefficients, dtype=float)
    if initial.size != x.shape[1]:
        initial = np.zeros(x.shape[1], dtype=float)

    def objective_and_gradient(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = x @ beta
        probability = _sigmoid(linear)
        clipped = np.clip(probability, LOGIT_EPS, 1.0 - LOGIT_EPS)
        loss = -np.sum(weights * (y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped)))
        prior_loss = 0.5 * float(np.sum(np.square(beta / sigma)))
        grad = x.T @ (weights * (probability - y)) + beta / (sigma * sigma)
        return float(loss + prior_loss), np.asarray(grad, dtype=float)

    result = minimize(
        fun=lambda beta: objective_and_gradient(beta)[0],
        x0=initial,
        jac=lambda beta: objective_and_gradient(beta)[1],
        method="L-BFGS-B",
    )
    coefficients = np.asarray(result.x if result.x is not None else initial, dtype=float)
    return LogisticFit(
        coefficients=coefficients,
        feature_names=feature_names or [f"x{index}" for index in range(x.shape[1])],
        success=bool(result.success),
        objective=float(result.fun if np.isfinite(result.fun) else np.nan),
    )


def _initial_coefficients(feature_names: list[str]) -> np.ndarray:
    values = {
        "intercept": -2.0,
        "local_separation": 0.2,
        "radius_similarity": 0.8,
        "opposite_side_score": 0.8,
        "z_similarity": 2.0,
        "color_similarity": 1.5,
        "size_similarity": 0.4,
        "ellipticity_similarity": 0.3,
        "reliability_mean": 0.6,
        "same_object_source": 0.1,
    }
    return np.asarray([values.get(name, 0.0) for name in feature_names], dtype=float)


def _assignment_from_threshold(probability_matrix: np.ndarray, threshold: float) -> np.ndarray:
    n_images = int(probability_matrix.shape[0])
    parent = np.arange(n_images, dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(n_images):
        for right in range(left + 1, n_images):
            if float(probability_matrix[left, right]) >= float(threshold):
                union(left, right)
    return canonicalize_partition([find(index) for index in range(n_images)])


def _assignment_from_complete_link_threshold(probability_matrix: np.ndarray, threshold: float) -> np.ndarray:
    n_images = int(probability_matrix.shape[0])
    families: list[list[int]] = []
    for image_index in range(n_images):
        best_family = -1
        best_score = -np.inf
        for family_index, members in enumerate(families):
            pair_values = [float(probability_matrix[image_index, member]) for member in members]
            if pair_values and min(pair_values) >= float(threshold):
                score = float(np.mean(pair_values))
                if score > best_score:
                    best_score = score
                    best_family = family_index
        if best_family >= 0:
            families[best_family].append(image_index)
        else:
            families.append([image_index])
    assignment = np.empty(n_images, dtype=int)
    for family_index, members in enumerate(families):
        assignment[np.asarray(members, dtype=int)] = family_index
    return canonicalize_partition(assignment)


def _cluster_link_score(probability_matrix: np.ndarray, left_members: list[int], right_members: list[int], linkage: str) -> float:
    values = [
        float(probability_matrix[left, right])
        for left in left_members
        for right in right_members
        if left != right
    ]
    if not values:
        return -np.inf
    if linkage == "complete":
        return float(np.min(values))
    if linkage == "average":
        return float(np.mean(values))
    raise ValueError(f"Unknown linkage: {linkage}")


def _assignment_from_agglomerative_threshold(
    probability_matrix: np.ndarray,
    threshold: float,
    *,
    linkage: str,
) -> np.ndarray:
    """Agglomerative threshold clustering with complete or average linkage."""
    clusters = [[index] for index in range(int(probability_matrix.shape[0]))]
    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_score = -np.inf
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                score = _cluster_link_score(probability_matrix, clusters[left_index], clusters[right_index], linkage)
                if score > best_score:
                    best_score = score
                    best_pair = (left_index, right_index)
        if best_pair is None or best_score < float(threshold):
            break
        left_index, right_index = best_pair
        clusters[left_index] = clusters[left_index] + clusters[right_index]
        del clusters[right_index]
    assignment = np.empty(int(probability_matrix.shape[0]), dtype=int)
    for cluster_index, members in enumerate(clusters):
        assignment[np.asarray(members, dtype=int)] = cluster_index
    return canonicalize_partition(assignment)


def _incompatible_pair_set(pair_table: pd.DataFrame) -> set[tuple[int, int]]:
    incompatible: set[tuple[int, int]] = set()
    if pair_table.empty:
        return incompatible
    labels, weights = create_anchor_labels(pair_table)
    for row_index, row in enumerate(pair_table.itertuples(index=False)):
        if row_index >= labels.size:
            continue
        if np.isfinite(labels[row_index]) and labels[row_index] < 0.5 and weights[row_index] > 0.0:
            left = int(row.left_index)
            right = int(row.right_index)
            incompatible.add((min(left, right), max(left, right)))
    return incompatible


def _same_anchor_groups(pair_table: pd.DataFrame, n_images: int) -> list[list[int]]:
    parent = np.arange(n_images, dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    if "same_input_family" in pair_table.columns:
        family_pairs = pair_table[pair_table["same_input_family"].fillna(False).astype(bool)]
        for row in family_pairs.itertuples(index=False):
            union(int(row.left_index), int(row.right_index))

    groups_by_root: dict[int, list[int]] = {}
    for image_index in range(n_images):
        groups_by_root.setdefault(find(image_index), []).append(image_index)
    return [members for members in groups_by_root.values() if len(members) > 1]


def _has_incompatible_cross_pair(left_members: list[int], right_members: list[int], incompatible_pairs: set[tuple[int, int]]) -> bool:
    for left in left_members:
        for right in right_members:
            key = (min(left, right), max(left, right))
            if key in incompatible_pairs:
                return True
    return False


def _assignment_from_anchor_expansion(
    probability_matrix: np.ndarray,
    pair_table: pd.DataFrame,
    *,
    threshold: float,
) -> np.ndarray:
    """Seed known high-confidence families, then attach compatible images by affinity."""
    n_images = int(probability_matrix.shape[0])
    used: set[int] = set()
    clusters: list[list[int]] = []
    for group in _same_anchor_groups(pair_table, n_images):
        clusters.append(list(group))
        used.update(group)
    for image_index in range(n_images):
        if image_index not in used:
            clusters.append([image_index])

    incompatible_pairs = _incompatible_pair_set(pair_table)
    changed = True
    while changed:
        changed = False
        best_pair: tuple[int, int] | None = None
        best_score = -np.inf
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                if _has_incompatible_cross_pair(clusters[left_index], clusters[right_index], incompatible_pairs):
                    continue
                score = _cluster_link_score(probability_matrix, clusters[left_index], clusters[right_index], "average")
                if score > best_score:
                    best_score = score
                    best_pair = (left_index, right_index)
        if best_pair is not None and best_score >= float(threshold):
            left_index, right_index = best_pair
            clusters[left_index] = clusters[left_index] + clusters[right_index]
            del clusters[right_index]
            changed = True

    assignment = np.empty(n_images, dtype=int)
    for cluster_index, members in enumerate(clusters):
        assignment[np.asarray(members, dtype=int)] = cluster_index
    return canonicalize_partition(assignment)


def _local_partition_perturbations(assignment: np.ndarray) -> list[np.ndarray]:
    """One-step split, merge, and single-image reassignment proposals."""
    base = canonicalize_partition(assignment)
    n_images = int(base.size)
    if n_images <= 1:
        return []
    proposals: list[np.ndarray] = []
    labels = sorted(np.unique(base).tolist())

    for source_label in labels:
        members = np.flatnonzero(base == source_label)
        if members.size <= 1:
            continue
        for image_index in members:
            proposal = base.copy()
            proposal[image_index] = int(np.max(base)) + 1
            proposals.append(canonicalize_partition(proposal))

    for left_pos, left_label in enumerate(labels):
        for right_label in labels[left_pos + 1 :]:
            proposal = base.copy()
            proposal[proposal == right_label] = left_label
            proposals.append(canonicalize_partition(proposal))

    for image_index in range(n_images):
        current_label = int(base[image_index])
        for target_label in labels:
            if int(target_label) == current_label:
                continue
            proposal = base.copy()
            proposal[image_index] = int(target_label)
            proposals.append(canonicalize_partition(proposal))
        proposal = base.copy()
        proposal[image_index] = int(np.max(base)) + 1
        proposals.append(canonicalize_partition(proposal))

    return proposals


def _assignment_from_pivot_correlation(
    probability_matrix: np.ndarray,
    *,
    threshold: float = 0.0,
    order: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy pivot correlation-clustering proposal from pair log-odds."""
    probability = np.asarray(probability_matrix, dtype=float)
    n_images = int(probability.shape[0])
    weights = _logit(probability)
    order_array = np.arange(n_images, dtype=int) if order is None else np.asarray(order, dtype=int)
    unassigned = np.ones(n_images, dtype=bool)
    assignment = np.full(n_images, -1, dtype=int)
    family_label = 0
    for pivot in order_array:
        pivot = int(pivot)
        if pivot < 0 or pivot >= n_images or not bool(unassigned[pivot]):
            continue
        members = unassigned & (weights[pivot] >= float(threshold))
        members[pivot] = True
        assignment[members] = family_label
        unassigned[members] = False
        family_label += 1
    for index in np.flatnonzero(unassigned):
        assignment[int(index)] = family_label
        family_label += 1
    return canonicalize_partition(assignment)


def _pivot_orders(probability_matrix: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    n_images = int(probability_matrix.shape[0])
    if n_images == 0:
        return [np.empty(0, dtype=int)]
    offdiag = np.asarray(probability_matrix, dtype=float).copy()
    np.fill_diagonal(offdiag, np.nan)
    mean_affinity = np.nanmean(offdiag, axis=1)
    mean_affinity = np.where(np.isfinite(mean_affinity), mean_affinity, 0.0)
    orders = [
        np.arange(n_images, dtype=int),
        np.argsort(-mean_affinity).astype(int),
        np.argsort(mean_affinity).astype(int),
    ]
    for _ in range(2 if n_images >= 15 else min(4, max(1, n_images))):
        orders.append(rng.permutation(n_images).astype(int))
    return orders


def generate_candidate_partitions(
    pair_table: pd.DataFrame,
    pair_probability: np.ndarray,
    *,
    n_images: int,
    n_thresholds: int = 25,
    n_noisy_partitions: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate valid full partitions from deterministic, noisy, and local proposals."""
    rng = np.random.default_rng() if rng is None else rng
    probability = np.asarray(pair_probability, dtype=float)
    if probability.shape != (n_images, n_images):
        raise ValueError("pair_probability must have shape (n_images, n_images).")
    candidates: dict[tuple[int, ...], np.ndarray] = {}

    def add(assignment: np.ndarray | list[int]) -> None:
        canonical = canonicalize_partition(assignment)
        if canonical.size == n_images:
            candidates[tuple(canonical.tolist())] = canonical

    add(np.arange(n_images, dtype=int))
    add(np.zeros(n_images, dtype=int))
    thresholds = np.linspace(0.05, 0.95, max(2, int(n_thresholds)))
    correlation_thresholds = np.unique(np.concatenate([_logit(thresholds), np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)]))
    orders = _pivot_orders(probability, rng)
    for threshold in thresholds:
        add(_assignment_from_threshold(probability, float(threshold)))
        add(_assignment_from_complete_link_threshold(probability, float(threshold)))
        add(_assignment_from_agglomerative_threshold(probability, float(threshold), linkage="complete"))
        add(_assignment_from_agglomerative_threshold(probability, float(threshold), linkage="average"))
        add(_assignment_from_anchor_expansion(probability, pair_table, threshold=float(threshold)))
    for threshold in correlation_thresholds:
        for order in orders:
            add(_assignment_from_pivot_correlation(probability, threshold=float(threshold), order=order))

    if "same_input_family" in pair_table.columns:
        assignment = np.arange(n_images, dtype=int)
        family_pairs = pair_table[pair_table["same_input_family"].fillna(False).astype(bool)]
        for row in family_pairs.itertuples(index=False):
            assignment[int(row.right_index)] = assignment[int(row.left_index)]
        add(assignment)

    for _ in range(max(0, int(n_noisy_partitions))):
        noisy = rng.beta(
            np.maximum(probability * 20.0, 1.0e-3),
            np.maximum((1.0 - probability) * 20.0, 1.0e-3),
        )
        threshold = float(rng.uniform(0.25, 0.75))
        add(_assignment_from_threshold(noisy, threshold))
        add(_assignment_from_agglomerative_threshold(noisy, threshold, linkage="complete"))
        add(_assignment_from_agglomerative_threshold(noisy, threshold, linkage="average"))
        add(_assignment_from_anchor_expansion(noisy, pair_table, threshold=threshold))
        noisy_order_count = 1 if n_images >= 15 else 3
        for order in _pivot_orders(noisy, rng)[:noisy_order_count]:
            add(_assignment_from_pivot_correlation(noisy, threshold=float(_logit(threshold)), order=order))

    base_candidates = list(candidates.values())
    for assignment in base_candidates:
        for proposal in _local_partition_perturbations(assignment):
            add(proposal)

    return np.asarray(list(candidates.values()), dtype=int)


def _partition_log_prior(
    assignment: np.ndarray,
    *,
    alpha: float = 0.3,
    singleton_penalty: float = 1.0,
    family_size_target: float = 3.0,
    family_size_sigma: float = 2.0,
) -> float:
    canonical = canonicalize_partition(assignment)
    n_images = int(canonical.size)
    if n_images == 0:
        return 0.0
    alpha = max(float(alpha), 1.0e-12)
    counts = np.bincount(canonical).astype(float)
    log_prior = len(counts) * math.log(alpha) + math.lgamma(alpha) - math.lgamma(alpha + n_images)
    log_prior += float(sum(math.lgamma(int(count)) for count in counts))
    log_prior -= float(singleton_penalty) * float(np.sum(counts == 1.0))
    if family_size_sigma > 0.0:
        multi_counts = counts[counts >= 2.0]
        if multi_counts.size:
            target = max(float(family_size_target), 2.0)
            log_prior -= 0.5 * float(np.sum(np.square((multi_counts - target) / float(family_size_sigma))))
    return float(log_prior)


def score_partitions(
    partitions: np.ndarray,
    pair_table: pd.DataFrame,
    pair_probability: np.ndarray,
    *,
    anchor_labels: np.ndarray | None = None,
    anchor_weights: np.ndarray | None = None,
    pair_score_mode: str = "sum",
) -> np.ndarray:
    """Score partitions with pair evidence and a simple partition prior."""
    if pair_score_mode not in {"none", "sum", "same_family_normalized", "block_normalized"}:
        raise ValueError("pair_score_mode must be 'none', 'sum', 'same_family_normalized', or 'block_normalized'.")
    partition_array = np.asarray(partitions, dtype=int)
    if partition_array.ndim == 1:
        partition_array = partition_array.reshape(1, -1)
    pair_probability = np.asarray(pair_probability, dtype=float)
    log_p = np.log(np.clip(pair_probability, LOGIT_EPS, 1.0 - LOGIT_EPS))
    log_not_p = np.log1p(-np.clip(pair_probability, LOGIT_EPS, 1.0 - LOGIT_EPS))
    anchor_labels_arr = None if anchor_labels is None else np.asarray(anchor_labels, dtype=float)
    anchor_weights_arr = None if anchor_weights is None else np.asarray(anchor_weights, dtype=float)
    scores = np.empty(partition_array.shape[0], dtype=float)
    pair_indices = [(int(row.left_index), int(row.right_index)) for row in pair_table.itertuples(index=False)]
    for index, assignment in enumerate(partition_array):
        canonical = canonicalize_partition(assignment)
        same = same_family_matrix(canonical)
        if pair_score_mode == "same_family_normalized":
            counts = np.bincount(canonical).astype(float)
            same_denominators = np.maximum(counts * (counts - 1.0) * 0.5, 1.0)
            family_counts = counts
        elif pair_score_mode == "block_normalized":
            counts = np.bincount(canonical).astype(float)
            same_denominators = np.maximum(counts * (counts - 1.0) * 0.5, 1.0)
            family_counts = np.maximum(counts, 1.0)
        else:
            same_denominators = np.ones(max(1, int(np.max(canonical)) + 1), dtype=float)
            family_counts = np.ones_like(same_denominators)
        score = _partition_log_prior(assignment)
        for pair_row, (left, right) in enumerate(pair_indices):
            is_same = bool(same[left, right])
            if pair_score_mode != "none":
                if is_same:
                    pair_weight = 1.0 / float(same_denominators[int(canonical[left])])
                elif pair_score_mode == "block_normalized":
                    pair_weight = 1.0 / float(family_counts[int(canonical[left])] * family_counts[int(canonical[right])])
                else:
                    pair_weight = 1.0
                score += pair_weight * float(log_p[left, right] if is_same else log_not_p[left, right])
            if anchor_labels_arr is not None and anchor_weights_arr is not None and pair_row < anchor_labels_arr.size:
                label = anchor_labels_arr[pair_row]
                weight = anchor_weights_arr[pair_row]
                if np.isfinite(label) and np.isfinite(weight) and weight > 0.0 and bool(round(float(label))) != is_same:
                    score -= 20.0 * float(weight)
        scores[index] = float(score)
    return scores


def _normalize_log_weights(log_scores: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(log_scores, dtype=float)
    if values.size == 0:
        return np.empty(0, dtype=float)
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("partition temperature must be positive and finite.")
    finite = np.isfinite(values)
    if not bool(np.any(finite)):
        return np.full(values.size, 1.0 / values.size, dtype=float)
    shifted = np.full(values.size, -np.inf, dtype=float)
    tempered = values[finite] / temperature
    shifted[finite] = tempered - float(np.max(tempered))
    raw = np.exp(shifted)
    return raw / float(np.sum(raw))


def _pair_vector_from_probability_matrix(pair_table: pd.DataFrame, matrix: np.ndarray) -> np.ndarray:
    return np.asarray([matrix[int(row.left_index), int(row.right_index)] for row in pair_table.itertuples(index=False)], dtype=float)


def _probability_matrix_from_pair_vector(pair_table: pd.DataFrame, values: np.ndarray, n_images: int) -> np.ndarray:
    matrix = np.eye(n_images, dtype=float)
    for value, row in zip(np.asarray(values, dtype=float), pair_table.itertuples(index=False)):
        matrix[int(row.left_index), int(row.right_index)] = float(value)
        matrix[int(row.right_index), int(row.left_index)] = float(value)
    return matrix


def _anchored_pair_probabilities(
    x: np.ndarray,
    coefficients: np.ndarray,
    anchor_labels: np.ndarray,
    anchor_weights: np.ndarray,
) -> np.ndarray:
    values = _sigmoid(x @ coefficients) if x.size else np.empty(0, dtype=float)
    for row_index, (label, weight) in enumerate(zip(anchor_labels, anchor_weights)):
        if np.isfinite(label) and np.isfinite(weight) and weight > 0.0:
            values[row_index] = 1.0 - LOGIT_EPS if label >= 0.5 else LOGIT_EPS
    return values


def _apply_external_partition_scores(
    log_scores: np.ndarray,
    partitions: np.ndarray,
    pair_table: pd.DataFrame,
    score_callback: PartitionScoreCallback | None,
    *,
    score_weight: float,
) -> np.ndarray:
    if score_callback is None:
        return np.asarray(log_scores, dtype=float)
    external_scores = np.asarray(score_callback(np.asarray(partitions, dtype=int), pair_table), dtype=float).reshape(-1)
    if external_scores.size != len(log_scores):
        raise ValueError("external partition score callback must return one score per partition.")
    return np.asarray(log_scores, dtype=float) + float(score_weight) * external_scores


def _gaussian_shared_mean_loglike(
    values: np.ndarray,
    variances: np.ndarray,
    *,
    prior_mean: float,
    prior_variance: float,
    include_singletons: bool = False,
) -> float:
    """Marginal likelihood for observations sharing one Gaussian latent mean."""
    y = np.asarray(values, dtype=float).reshape(-1)
    var = np.asarray(variances, dtype=float).reshape(-1)
    finite = np.isfinite(y) & np.isfinite(var) & (var > 0.0)
    y = y[finite]
    var = var[finite]
    n_values = int(y.size)
    if n_values == 0 or (n_values == 1 and not include_singletons):
        return 0.0
    prior_var = max(float(prior_variance), 1.0e-8)
    centered = y - float(prior_mean)
    precision = 1.0 / var
    prior_precision = 1.0 / prior_var
    total_precision = prior_precision + float(np.sum(precision))
    logdet = float(np.sum(np.log(var))) + math.log(prior_var) + math.log(total_precision)
    quad = float(np.sum(np.square(centered) * precision))
    weighted_sum = float(np.sum(centered * precision))
    quad -= weighted_sum * weighted_sum / total_precision
    return float(-0.5 * (n_values * math.log(2.0 * math.pi) + logdet + quad))


def _gaussian_independent_background_loglike(
    values: np.ndarray,
    *,
    background_mean: float,
    background_variance: float,
    include_singletons: bool = False,
) -> float:
    """Independent Gaussian background likelihood for observations in one proposed family."""
    y = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(y)
    y = y[finite]
    n_values = int(y.size)
    if n_values == 0 or (n_values == 1 and not include_singletons):
        return 0.0
    variance = max(float(background_variance), 1.0e-8)
    centered = y - float(background_mean)
    return float(-0.5 * np.sum(math.log(2.0 * math.pi * variance) + np.square(centered) / variance))


def _family_catalog_arrays(
    images: pd.DataFrame,
    *,
    redshift_default_sigma: float,
    color_sigma: float,
    morphology_sigma: float,
) -> dict[str, Any]:
    work = images.reset_index(drop=True).copy()
    redshift = np.asarray([_best_redshift(row) for _, row in work.iterrows()], dtype=float)
    redshift_sigma = np.asarray([_redshift_sigma(row, default=redshift_default_sigma) for _, row in work.iterrows()], dtype=float)
    redshift_sigma = np.where(np.isfinite(redshift_sigma) & (redshift_sigma > 0.0), redshift_sigma, float(redshift_default_sigma))
    finite_z = np.isfinite(redshift) & (redshift > 0.0)
    z_prior_mean = float(np.nanmedian(redshift[finite_z])) if bool(np.any(finite_z)) else 2.0
    z_prior_sigma = max(3.0, float(np.nanstd(redshift[finite_z])) if int(np.sum(finite_z)) > 1 else 0.0)

    mag_columns = _mag_columns(work)
    colors = np.asarray([_color_vector(row, mag_columns) for _, row in work.iterrows()], dtype=float) if mag_columns else np.empty((len(work), 0), dtype=float)
    if colors.size:
        color_prior_mean = np.nanmean(colors, axis=0)
        color_prior_mean = np.where(np.isfinite(color_prior_mean), color_prior_mean, 0.0)
        color_prior_sigma = np.full(colors.shape[1], max(float(color_sigma) * 5.0, 1.0), dtype=float)
        color_background_sigma = np.nanstd(colors, axis=0)
        color_background_sigma = np.where(
            np.isfinite(color_background_sigma) & (color_background_sigma > 0.0),
            color_background_sigma,
            color_prior_sigma,
        )
        color_background_sigma = np.maximum(color_background_sigma, float(color_sigma))
    else:
        color_prior_mean = np.empty(0, dtype=float)
        color_prior_sigma = np.empty(0, dtype=float)
        color_background_sigma = np.empty(0, dtype=float)

    morphology_columns = [column for column in ("image_size_arcsec", "image_ellipticity") if column in work.columns]
    morphology = (
        work[morphology_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if morphology_columns
        else np.empty((len(work), 0), dtype=float)
    )
    if morphology.size:
        morphology_prior_mean = np.nanmean(morphology, axis=0)
        morphology_prior_mean = np.where(np.isfinite(morphology_prior_mean), morphology_prior_mean, 0.0)
        morphology_prior_sigma = np.full(morphology.shape[1], max(float(morphology_sigma) * 5.0, 1.0), dtype=float)
        morphology_background_sigma = np.nanstd(morphology, axis=0)
        morphology_background_sigma = np.where(
            np.isfinite(morphology_background_sigma) & (morphology_background_sigma > 0.0),
            morphology_background_sigma,
            morphology_prior_sigma,
        )
        morphology_background_sigma = np.maximum(morphology_background_sigma, float(morphology_sigma))
    else:
        morphology_prior_mean = np.empty(0, dtype=float)
        morphology_prior_sigma = np.empty(0, dtype=float)
        morphology_background_sigma = np.empty(0, dtype=float)

    return {
        "redshift": redshift,
        "redshift_sigma": redshift_sigma,
        "z_prior_mean": z_prior_mean,
        "z_prior_sigma": z_prior_sigma,
        "z_background_mean": z_prior_mean,
        "z_background_sigma": z_prior_sigma,
        "colors": colors,
        "color_sigma": np.full(colors.shape[1], float(color_sigma), dtype=float) if colors.ndim == 2 else np.empty(0, dtype=float),
        "color_prior_mean": color_prior_mean,
        "color_prior_sigma": color_prior_sigma,
        "color_background_mean": color_prior_mean,
        "color_background_sigma": color_background_sigma,
        "morphology": morphology,
        "morphology_sigma": np.full(morphology.shape[1], float(morphology_sigma), dtype=float) if morphology.ndim == 2 else np.empty(0, dtype=float),
        "morphology_prior_mean": morphology_prior_mean,
        "morphology_prior_sigma": morphology_prior_sigma,
        "morphology_background_mean": morphology_prior_mean,
        "morphology_background_sigma": morphology_background_sigma,
    }


def family_catalog_partition_loglike(
    partitions: np.ndarray,
    images: pd.DataFrame,
    *,
    redshift_weight: float = 1.0,
    color_weight: float = 1.0,
    morphology_weight: float = 0.3,
    redshift_default_sigma: float = 0.5,
    color_sigma: float = 0.25,
    morphology_sigma: float = 0.15,
    score_mode: str = "raw",
) -> np.ndarray:
    """Score partitions by family-level redshift, color, and morphology likelihoods."""
    if score_mode not in {"raw", "likelihood_ratio"}:
        raise ValueError("score_mode must be 'raw' or 'likelihood_ratio'.")
    partition_array = np.asarray(partitions, dtype=int)
    if partition_array.ndim == 1:
        partition_array = partition_array.reshape(1, -1)
    arrays = _family_catalog_arrays(
        images,
        redshift_default_sigma=redshift_default_sigma,
        color_sigma=color_sigma,
        morphology_sigma=morphology_sigma,
    )
    scores = np.zeros(partition_array.shape[0], dtype=float)
    family_score_cache: dict[tuple[int, ...], float] = {}
    for partition_index, assignment in enumerate(partition_array):
        canonical = canonicalize_partition(assignment)
        score = 0.0
        for family_label in np.unique(canonical):
            members = np.flatnonzero(canonical == family_label)
            member_key = tuple(int(item) for item in members.tolist())
            cached = family_score_cache.get(member_key)
            if cached is not None:
                score += cached
                continue
            if members.size <= 1:
                family_score_cache[member_key] = 0.0
                continue
            family_score = 0.0
            if redshift_weight != 0.0:
                same_loglike = _gaussian_shared_mean_loglike(
                    arrays["redshift"][members],
                    np.square(arrays["redshift_sigma"][members]),
                    prior_mean=float(arrays["z_prior_mean"]),
                    prior_variance=float(arrays["z_prior_sigma"]) ** 2,
                )
                if score_mode == "likelihood_ratio":
                    same_loglike -= _gaussian_independent_background_loglike(
                        arrays["redshift"][members],
                        background_mean=float(arrays["z_background_mean"]),
                        background_variance=float(arrays["z_background_sigma"]) ** 2,
                    )
                family_score += float(redshift_weight) * same_loglike
            colors = arrays["colors"]
            if color_weight != 0.0 and colors.size:
                for column_index in range(colors.shape[1]):
                    same_loglike = _gaussian_shared_mean_loglike(
                        colors[members, column_index],
                        np.full(members.size, float(arrays["color_sigma"][column_index]) ** 2, dtype=float),
                        prior_mean=float(arrays["color_prior_mean"][column_index]),
                        prior_variance=float(arrays["color_prior_sigma"][column_index]) ** 2,
                    )
                    if score_mode == "likelihood_ratio":
                        same_loglike -= _gaussian_independent_background_loglike(
                            colors[members, column_index],
                            background_mean=float(arrays["color_background_mean"][column_index]),
                            background_variance=float(arrays["color_background_sigma"][column_index]) ** 2,
                        )
                    family_score += float(color_weight) * same_loglike
            morphology = arrays["morphology"]
            if morphology_weight != 0.0 and morphology.size:
                for column_index in range(morphology.shape[1]):
                    same_loglike = _gaussian_shared_mean_loglike(
                        morphology[members, column_index],
                        np.full(members.size, float(arrays["morphology_sigma"][column_index]) ** 2, dtype=float),
                        prior_mean=float(arrays["morphology_prior_mean"][column_index]),
                        prior_variance=float(arrays["morphology_prior_sigma"][column_index]) ** 2,
                    )
                    if score_mode == "likelihood_ratio":
                        same_loglike -= _gaussian_independent_background_loglike(
                            morphology[members, column_index],
                            background_mean=float(arrays["morphology_background_mean"][column_index]),
                            background_variance=float(arrays["morphology_background_sigma"][column_index]) ** 2,
                        )
                    family_score += float(morphology_weight) * same_loglike
            family_score_cache[member_key] = float(family_score)
            score += float(family_score)
        scores[partition_index] = float(score)
    return scores


def family_catalog_score_callback(
    images: pd.DataFrame,
    *,
    redshift_weight: float = 1.0,
    color_weight: float = 1.0,
    morphology_weight: float = 0.3,
    redshift_default_sigma: float = 0.5,
    color_sigma: float = 0.25,
    morphology_sigma: float = 0.15,
    score_mode: str = "raw",
) -> PartitionScoreCallback:
    """Build an external scorer for family-level catalog likelihoods."""
    work = images.reset_index(drop=True).copy()

    def callback(partitions: np.ndarray, _pair_table: pd.DataFrame) -> np.ndarray:
        return family_catalog_partition_loglike(
            partitions,
            work,
            redshift_weight=redshift_weight,
            color_weight=color_weight,
            morphology_weight=morphology_weight,
            redshift_default_sigma=redshift_default_sigma,
            color_sigma=color_sigma,
            morphology_sigma=morphology_sigma,
            score_mode=score_mode,
        )

    return callback


def combine_partition_score_callbacks(
    callbacks: list[tuple[PartitionScoreCallback | None, float]],
) -> PartitionScoreCallback | None:
    """Combine weighted partition-score callbacks into one callback."""
    active = [(callback, float(weight)) for callback, weight in callbacks if callback is not None and float(weight) != 0.0]
    if not active:
        return None

    def callback(partitions: np.ndarray, pair_table: pd.DataFrame) -> np.ndarray:
        total = np.zeros(np.asarray(partitions).reshape((-1, np.asarray(partitions).shape[-1])).shape[0], dtype=float)
        for score_callback, weight in active:
            total += weight * np.asarray(score_callback(partitions, pair_table), dtype=float).reshape(-1)
        return total

    return callback


def _score_partitions_with_callbacks(
    partitions: np.ndarray,
    pair_table: pd.DataFrame,
    pair_probability_matrix: np.ndarray,
    *,
    anchor_labels: np.ndarray,
    anchor_weights: np.ndarray,
    pair_score_mode: str,
    partition_score_callback: PartitionScoreCallback | None,
    partition_score_callback_weight: float,
) -> np.ndarray:
    log_scores = score_partitions(
        partitions,
        pair_table,
        pair_probability_matrix,
        anchor_labels=anchor_labels,
        anchor_weights=anchor_weights,
        pair_score_mode=pair_score_mode,
    )
    return _apply_external_partition_scores(
        log_scores,
        partitions,
        pair_table,
        partition_score_callback,
        score_weight=partition_score_callback_weight,
    )


def repair_candidate_partitions(
    partitions: np.ndarray,
    log_scores: np.ndarray,
    pair_table: pd.DataFrame,
    pair_probability_matrix: np.ndarray,
    *,
    anchor_labels: np.ndarray,
    anchor_weights: np.ndarray,
    pair_score_mode: str,
    partition_score_callback: PartitionScoreCallback | None = None,
    partition_score_callback_weight: float = 1.0,
    top_k: int = 200,
    max_rounds: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair top candidate partitions with scorer-guided local perturbations."""
    partition_array = np.asarray(partitions, dtype=int)
    if partition_array.ndim == 1:
        partition_array = partition_array.reshape(1, -1)
    score_array = np.asarray(log_scores, dtype=float).reshape(-1)
    if partition_array.shape[0] != score_array.size:
        raise ValueError("partitions and log_scores must have matching rows.")
    if partition_array.shape[0] == 0 or int(max_rounds) <= 0 or int(top_k) <= 0:
        return partition_array, score_array
    n_images = int(partition_array.shape[1])
    current = _dedupe_partitions(partition_array, n_images=n_images)
    current_scores = _score_partitions_with_callbacks(
        current,
        pair_table,
        pair_probability_matrix,
        anchor_labels=anchor_labels,
        anchor_weights=anchor_weights,
        pair_score_mode=pair_score_mode,
        partition_score_callback=partition_score_callback,
        partition_score_callback_weight=partition_score_callback_weight,
    )
    for _round in range(int(max_rounds)):
        if current.size == 0:
            break
        order = np.argsort(-current_scores)[: min(int(top_k), current.shape[0])]
        proposals: list[np.ndarray] = []
        for partition_index in order:
            proposals.extend(_local_partition_perturbations(current[int(partition_index)]))
        if not proposals:
            break
        combined = np.vstack([current, np.asarray(proposals, dtype=int)])
        repaired = _dedupe_partitions(combined, n_images=n_images)
        if repaired.shape[0] == current.shape[0]:
            break
        current = repaired
        current_scores = _score_partitions_with_callbacks(
            current,
            pair_table,
            pair_probability_matrix,
            anchor_labels=anchor_labels,
            anchor_weights=anchor_weights,
            pair_score_mode=pair_score_mode,
            partition_score_callback=partition_score_callback,
            partition_score_callback_weight=partition_score_callback_weight,
        )
    return current, current_scores


def _dedupe_partitions(partitions: np.ndarray, *, n_images: int) -> np.ndarray:
    partition_array = np.asarray(partitions, dtype=int)
    if partition_array.ndim == 1:
        partition_array = partition_array.reshape(1, -1)
    candidates: dict[tuple[int, ...], np.ndarray] = {}
    for row in partition_array:
        canonical = canonicalize_partition(row)
        if canonical.size == n_images:
            candidates[tuple(canonical.tolist())] = canonical
    return np.asarray(list(candidates.values()), dtype=int)


def run_family_partition_engine(
    images: pd.DataFrame,
    *,
    random_seed: int | None = None,
    n_iterations: int = 3,
    n_thresholds: int = 25,
    n_noisy_partitions: int = 100,
    gaussian_prior_sigma: float = 2.0,
    partition_temperature: float = 1.0,
    partition_score_callback: PartitionScoreCallback | None = None,
    partition_score_callback_weight: float = 1.0,
    proposal_matrix_callback: ProposalMatrixCallback | None = None,
    proposal_partitions_callback: ProposalPartitionsCallback | None = None,
    initial_coefficients: np.ndarray | None = None,
    refit_logistic: bool = True,
    pair_score_mode: str = "sum",
    repair_top_k: int = 0,
    repair_max_rounds: int = 0,
) -> PartitionResult:
    """Run anchor-seeded, probability-conserving candidate partition inference."""
    work = images.reset_index(drop=True).copy()
    n_images = int(len(work))
    partition_temperature = float(partition_temperature)
    if not np.isfinite(partition_temperature) or partition_temperature <= 0.0:
        raise ValueError("partition_temperature must be positive and finite.")
    rng = np.random.default_rng(random_seed)
    pair_table = build_pair_table(work)
    x, feature_names = feature_matrix(pair_table)
    anchor_labels, anchor_weights = create_anchor_labels(pair_table)

    if n_images == 0:
        logistic_fit = LogisticFit(np.zeros(len(feature_names), dtype=float), feature_names, True, 0.0)
        return PartitionResult(
            partitions=np.empty((0, 0), dtype=int),
            log_scores=np.empty(0, dtype=float),
            weights=np.empty(0, dtype=float),
            pair_probability=np.empty((0, 0), dtype=float),
            pair_table=pair_table,
            map_assignment=np.empty(0, dtype=int),
            logistic_fit=logistic_fit,
            pair_score_mode=pair_score_mode,
        )
    if n_images == 1:
        logistic_fit = LogisticFit(np.zeros(len(feature_names), dtype=float), feature_names, True, 0.0)
        return PartitionResult(
            partitions=np.zeros((1, 1), dtype=int),
            log_scores=np.asarray([0.0], dtype=float),
            weights=np.asarray([1.0], dtype=float),
            pair_probability=np.ones((1, 1), dtype=float),
            pair_table=pair_table,
            map_assignment=np.asarray([0], dtype=int),
            logistic_fit=logistic_fit,
            pair_score_mode=pair_score_mode,
        )

    if initial_coefficients is None:
        coefficients = _initial_coefficients(feature_names)
    else:
        coefficients = np.asarray(initial_coefficients, dtype=float).reshape(-1)
        if coefficients.size != len(feature_names):
            raise ValueError("initial_coefficients length must match the partition feature count.")
    logistic_fit = LogisticFit(coefficients, feature_names, True, float("nan"))
    partitions = np.empty((0, n_images), dtype=int)
    log_scores = np.empty(0, dtype=float)
    weights = np.empty(0, dtype=float)
    partition_pair_probability = np.eye(n_images, dtype=float)

    for iteration in range(max(1, int(n_iterations))):
        model_pair_probs = _anchored_pair_probabilities(x, coefficients, anchor_labels, anchor_weights)
        pair_probability_matrix = _probability_matrix_from_pair_vector(pair_table, model_pair_probs, n_images)
        partitions = generate_candidate_partitions(
            pair_table,
            pair_probability_matrix,
            n_images=n_images,
            n_thresholds=n_thresholds,
            n_noisy_partitions=n_noisy_partitions,
            rng=rng,
        )
        if proposal_matrix_callback is not None:
            for proposal_matrix in proposal_matrix_callback(pair_probability_matrix, pair_table):
                extra_partitions = generate_candidate_partitions(
                    pair_table,
                    proposal_matrix,
                    n_images=n_images,
                    n_thresholds=n_thresholds,
                    n_noisy_partitions=n_noisy_partitions,
                    rng=rng,
                )
                partitions = _dedupe_partitions(np.vstack([partitions, extra_partitions]), n_images=n_images)
        if proposal_partitions_callback is not None:
            extra_partitions = proposal_partitions_callback(pair_probability_matrix, pair_table)
            if np.asarray(extra_partitions).size:
                partitions = _dedupe_partitions(np.vstack([partitions, extra_partitions]), n_images=n_images)
        log_scores = _score_partitions_with_callbacks(
            partitions,
            pair_table,
            pair_probability_matrix,
            anchor_labels=anchor_labels,
            anchor_weights=anchor_weights,
            pair_score_mode=pair_score_mode,
            partition_score_callback=partition_score_callback,
            partition_score_callback_weight=partition_score_callback_weight,
        )
        weights = _normalize_log_weights(log_scores, temperature=partition_temperature)
        partition_pair_probability = weighted_pair_probability(partitions, weights)
        soft_targets = _pair_vector_from_probability_matrix(pair_table, partition_pair_probability)
        if not bool(refit_logistic):
            break
        fit_targets = soft_targets.copy()
        fit_weights = np.ones(len(pair_table), dtype=float)
        anchored = np.isfinite(anchor_labels) & (anchor_weights > 0.0)
        fit_targets[anchored] = anchor_labels[anchored]
        fit_weights[anchored] = np.maximum(anchor_weights[anchored], 1.0)
        logistic_fit = fit_logistic_map(
            x,
            fit_targets,
            sample_weight=fit_weights,
            gaussian_prior_sigma=gaussian_prior_sigma,
            initial_coefficients=coefficients,
            feature_names=feature_names,
        )
        coefficients = logistic_fit.coefficients
        if iteration == max(1, int(n_iterations)) - 1:
            break

    final_pair_probs_for_scoring = _anchored_pair_probabilities(x, coefficients, anchor_labels, anchor_weights)
    final_pair_probability_matrix = _probability_matrix_from_pair_vector(pair_table, final_pair_probs_for_scoring, n_images)
    partitions = generate_candidate_partitions(
        pair_table,
        final_pair_probability_matrix,
        n_images=n_images,
        n_thresholds=n_thresholds,
        n_noisy_partitions=n_noisy_partitions,
        rng=rng,
    )
    if proposal_matrix_callback is not None:
        for proposal_matrix in proposal_matrix_callback(final_pair_probability_matrix, pair_table):
            extra_partitions = generate_candidate_partitions(
                pair_table,
                proposal_matrix,
                n_images=n_images,
                n_thresholds=n_thresholds,
                n_noisy_partitions=n_noisy_partitions,
                rng=rng,
            )
            partitions = _dedupe_partitions(np.vstack([partitions, extra_partitions]), n_images=n_images)
    if proposal_partitions_callback is not None:
        extra_partitions = proposal_partitions_callback(final_pair_probability_matrix, pair_table)
        if np.asarray(extra_partitions).size:
            partitions = _dedupe_partitions(np.vstack([partitions, extra_partitions]), n_images=n_images)
    log_scores = _score_partitions_with_callbacks(
        partitions,
        pair_table,
        final_pair_probability_matrix,
        anchor_labels=anchor_labels,
        anchor_weights=anchor_weights,
        pair_score_mode=pair_score_mode,
        partition_score_callback=partition_score_callback,
        partition_score_callback_weight=partition_score_callback_weight,
    )
    if int(repair_top_k) > 0 and int(repair_max_rounds) > 0:
        partitions, log_scores = repair_candidate_partitions(
            partitions,
            log_scores,
            pair_table,
            final_pair_probability_matrix,
            anchor_labels=anchor_labels,
            anchor_weights=anchor_weights,
            pair_score_mode=pair_score_mode,
            partition_score_callback=partition_score_callback,
            partition_score_callback_weight=partition_score_callback_weight,
            top_k=int(repair_top_k),
            max_rounds=int(repair_max_rounds),
        )
    weights = _normalize_log_weights(log_scores, temperature=partition_temperature)
    partition_pair_probability = weighted_pair_probability(partitions, weights)

    pair_table = pair_table.copy()
    final_pair_probs = _sigmoid(x @ coefficients) if len(pair_table) else np.empty(0, dtype=float)
    pair_table["model_pair_probability"] = final_pair_probs
    pair_table["partition_pair_probability"] = _pair_vector_from_probability_matrix(pair_table, partition_pair_probability)
    pair_table["anchor_label"] = anchor_labels
    pair_table["anchor_weight"] = anchor_weights
    map_assignment = partitions[int(np.argmax(weights))] if weights.size else np.arange(n_images, dtype=int)
    return PartitionResult(
        partitions=np.asarray(partitions, dtype=int),
        log_scores=np.asarray(log_scores, dtype=float),
        weights=np.asarray(weights, dtype=float),
        pair_probability=np.asarray(partition_pair_probability, dtype=float),
        pair_table=pair_table,
        map_assignment=canonicalize_partition(map_assignment),
        logistic_fit=logistic_fit,
        pair_score_mode=pair_score_mode,
    )
