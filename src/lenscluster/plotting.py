from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import astropy.units as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from scipy.stats import norm
from skimage.measure import find_contours

try:
    import corner
except ImportError:  # pragma: no cover
    corner = None

from .image_diagnostics import (
    diagnostic_detail_array as _shared_diagnostic_detail_array,
    extra_image_rows as _shared_extra_image_rows,
    family_image_recovery_rows as _shared_family_image_recovery_rows,
    image_count_info_from_exact_details as _shared_image_count_info_from_exact_details,
    image_count_recovery_row as _shared_image_count_recovery_row,
    image_count_recovery_summary as _shared_image_count_recovery_summary,
    image_count_recovery_table as _shared_image_count_recovery_table,
    model_count_fields_from_count_info as _shared_model_count_fields_from_count_info,
    successful_image_count_info as _shared_successful_image_count_info,
    unavailable_image_count_info as _shared_unavailable_image_count_info,
)
from .jax_cosmology import critical_surface_density_angle_from_config
from .jax_cosmology import kpc_per_arcsec_from_config as _kpc_per_arcsec_from_config
from .lenstool_parser import load_best_par
from .model import BuildState, EvaluationResult, ParameterSpec, PosteriorResults
from .model import convert_theta_to_latent as _convert_theta_to_latent
from .model import display_lower as _display_lower
from .model import display_upper as _display_upper
from .utils import jax_cpu_worker_count
from .utils import log_message as _log
from .utils import run_logged_phase as _run_logged_phase

DEFAULT_NUTS_INIT_BOUNDARY_FRAC = 0.02
DEFAULT_NUTS_INIT_JITTER_FRAC = 0.02
DEFAULT_SVI_STEPS = 2000
DEFAULT_SVI_LEARNING_RATE = 5.0e-3
CORNER_SIGMA_CONTOUR_LEVELS = tuple(float(1.0 - np.exp(-0.5 * sigma**2)) for sigma in (1.0, 2.0, 3.0))
CORNER_PLOT_KWARGS = {
    "show_titles": True,
    "title_fmt": ".3g",
    "quantiles": [0.16, 0.5, 0.84],
    "levels": CORNER_SIGMA_CONTOUR_LEVELS,
    "plot_datapoints": False,
    "fill_contours": True,
    "smooth": 1.0,
    "smooth1d": 1.0,
    "max_n_ticks": 4,
}
CORNER_PLOT_DPI = 300
CORNER_BEST_FIT_COLOR = "#d4a017"
CORNER_BEST_PAR_COLOR = "tab:red"
CORNER_PREVIOUS_STAGE_COLOR = "tab:green"
CORNER_BAYES_OVERLAY_COLOR = "tab:red"
SMC_CORNER_MAX_PARAMS = 8
CAUSTIC_OVERLAY_FOV_ARCSEC = 200.0
CAUSTIC_PLOT_GRID_SCALE_ARCSEC = 0.2
ABSOLUTE_MAGNIFICATION_PLOT_CAP = 25.0
CRITICAL_ARC_CURVE_SUPPORT_RADIUS_ARCSEC = 0.5
CRITICAL_ARC_SINGULAR_THRESHOLD = 0.20
SUBHALO_TOTAL_MASS_RADIUS_FACTOR = 1.0e6
SUBHALO_PROPERTIES_COLUMNS = [
    "component_index",
    "potfile_id",
    "potfile_order",
    "catalog_id",
    "catalog_mag",
    "x_centre",
    "y_centre",
    "radius_arcsec",
    "sigma0",
    "Ra",
    "Rs",
    "mass_within_Rs_msun",
    "mass_within_1e6_Rs_msun",
]


def _caustic_plot_grid_axes(grid_scale_arcsec: float) -> tuple[np.ndarray, np.ndarray]:
    grid_scale = float(grid_scale_arcsec)
    if not np.isfinite(grid_scale) or grid_scale <= 0.0:
        raise ValueError("caustic plot grid scale must be positive.")
    half_fov = CAUSTIC_OVERLAY_FOV_ARCSEC / 2.0
    grid_num_pix = max(2, int(round(CAUSTIC_OVERLAY_FOV_ARCSEC / grid_scale)) + 1)
    axis = np.linspace(-half_fov, half_fov, grid_num_pix)
    return axis, axis.copy()


def _first_int_value(value: Any, default: int) -> int:
    if isinstance(value, (list, tuple)):
        if not value:
            return int(default)
        return int(value[0])
    return int(value)


def plot_path(root: Path, name: str) -> Path:
    """Return an output plot path, creating the output directory first."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        path = path.with_suffix(".pdf")
    return path


def _plot_path(root: Path, name: str) -> Path:
    return plot_path(root, name)


PlotTask = tuple[str, str, Callable[[], Any]]


def _run_plot_tasks_with_progress(args: argparse.Namespace, plot_tasks: list[PlotTask]) -> None:
    if not plot_tasks:
        return
    if bool(getattr(args, "quiet", False)):
        for _display_name, phase_name, task in plot_tasks:
            _run_logged_phase(args, phase_name, task)
        return
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task_id = progress.add_task("plots", total=len(plot_tasks))
        for display_name, phase_name, task in plot_tasks:
            progress.update(task_id, description=f"plots: {display_name}")
            _run_logged_phase(args, phase_name, task)
            progress.advance(task_id)
        progress.update(task_id, description="plots: complete")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    sorter = np.argsort(values)
    sorted_values = values[sorter]
    sorted_weights = weights[sorter]
    cumulative = np.cumsum(sorted_weights)
    cumulative = cumulative / cumulative[-1]
    return np.interp(quantiles, cumulative, sorted_values)


def _normalized_weights(weights: np.ndarray | None, n_samples: int) -> np.ndarray:
    if n_samples <= 0:
        return np.empty((0,), dtype=float)
    if weights is None:
        return np.full(n_samples, 1.0 / n_samples, dtype=float)
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    if weight_array.size != n_samples or not np.all(np.isfinite(weight_array)):
        return np.full(n_samples, 1.0 / n_samples, dtype=float)
    total = float(np.sum(weight_array))
    if total <= 0.0:
        return np.full(n_samples, 1.0 / n_samples, dtype=float)
    return weight_array / total


def _valid_normalized_sample_weights(weights: np.ndarray | None, n_samples: int) -> np.ndarray | None:
    if weights is None or n_samples <= 0:
        return None
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    if weight_array.size != n_samples:
        return None
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        return None
    total = float(np.sum(weight_array))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return weight_array / total


def _effective_sample_size(weights: np.ndarray) -> float:
    normalized = _normalized_weights(weights, len(weights))
    if normalized.size == 0:
        return 0.0
    return float(1.0 / np.sum(np.square(normalized)))


def _family_color_map(family_ids: list[str] | np.ndarray | pd.Series) -> dict[str, tuple[float, float, float, float]]:
    ordered_family_ids = list(dict.fromkeys(str(family_id) for family_id in family_ids))
    cmap = plt.get_cmap("tab20", max(1, len(ordered_family_ids)))
    return {family_id: cmap(idx) for idx, family_id in enumerate(ordered_family_ids)}


def _color_with_alpha(color: Any, alpha: float) -> tuple[float, float, float, float]:
    try:
        rgba = np.asarray(to_rgba(color), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        rgba = np.asarray(color, dtype=float).reshape(-1)
    if rgba.size < 3:
        return (0.0, 0.0, 0.0, float(alpha))
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(alpha))


def _metric_text(value: Any, *, precision: int = 4) -> str:
    if value is None:
        return "na"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value)
        return text if text else "na"
    if not np.isfinite(numeric):
        return "na"
    if abs(numeric - round(numeric)) < 1.0e-10 and abs(numeric) < 1.0e9:
        return str(int(round(numeric)))
    return f"{numeric:.{precision}g}"


def _key_value_lines(items: list[tuple[str, Any]], *, key_width: int = 34) -> list[str]:
    return [f"{label:<{key_width}} {_metric_text(value)}" for label, value in items]


def _table_text(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return ["No rows available."]
    rendered = [
        {header: _metric_text(row.get(key)) for header, key in columns}
        for row in rows
    ]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rendered))
        for header, _key in columns
    }
    lines = [
        " ".join(header.ljust(widths[header]) for header, _key in columns),
        " ".join("-" * widths[header] for header, _key in columns),
    ]
    for row in rendered:
        lines.append(" ".join(row[header].ljust(widths[header]) for header, _key in columns))
    return lines


def _finite_sample_rows(samples: np.ndarray) -> np.ndarray:
    samples_array = np.asarray(samples, dtype=float)
    if samples_array.ndim != 2 or samples_array.size == 0:
        return np.empty((0, samples_array.shape[-1] if samples_array.ndim == 2 else 0), dtype=float)
    return samples_array[np.isfinite(samples_array).all(axis=1)]


def _uniform_normalized_plot_weights(samples: np.ndarray) -> np.ndarray | None:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] <= 0:
        return None
    return np.full(sample_array.shape[0], 1.0 / float(sample_array.shape[0]), dtype=float)


def _summary_table(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    if not parameter_specs:
        return pd.DataFrame(
            columns=[
                "potential_id",
                "profile_type",
                "component_family",
                "prior_kind",
                "parameter",
                "label",
                "map",
                "median",
                "mean",
                "std",
                "p16",
                "p84",
                "lower",
                "upper",
            ]
        )
    weights = _normalized_weights(sample_weights, samples.shape[0])
    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(parameter_specs):
        values = samples[:, idx]
        q16, q50, q84 = _weighted_quantile(values, weights, [0.16, 0.50, 0.84])
        rows.append(
            {
                "potential_id": spec.potential_id,
                "profile_type": spec.profile_type,
                "component_family": spec.component_family,
                "prior_kind": spec.prior_kind,
                "parameter": spec.field,
                "label": spec.name,
                "map": float(best_fit[idx]),
                "median": float(q50),
                "mean": float(np.average(values, weights=weights)),
                "std": float(np.sqrt(np.average(np.square(values - np.average(values, weights=weights)), weights=weights))),
                "p16": float(q16),
                "p84": float(q84),
                "lower": _display_lower(spec),
                "upper": _display_upper(spec),
            }
        )
    return pd.DataFrame(rows)


def _write_potfile_summary_txt(tables_dir: Path, summary_df: pd.DataFrame) -> None:
    scaling_df = summary_df[summary_df["component_family"] == "scaling"].copy()
    if scaling_df.empty:
        return
    scaling_df = scaling_df.sort_values(["potential_id", "parameter", "label"]).reset_index(drop=True)
    lines: list[str] = []
    for potential_id, group_df in scaling_df.groupby("potential_id", sort=False):
        lines.append(str(potential_id))
        for row in group_df.itertuples(index=False):
            lines.append(f"  {row.parameter}: median={float(row.median):.6g}, std={float(row.std):.6g}")
        lines.append("")
    content = "\n".join(lines).rstrip() + "\n"
    (tables_dir / "potfile_summary.txt").write_text(content, encoding="utf-8")


def _potfile_constraint_diagnostics_table(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
    scaling_rank_df: pd.DataFrame,
    sample_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    scaling_specs = [spec for spec in parameter_specs if spec.component_family == "scaling"]
    if not scaling_specs:
        return pd.DataFrame(
            columns=[
                "potfile_id",
                "parameter",
                "label",
                "prior_kind",
                "prior_mean",
                "prior_std",
                "posterior_mean",
                "posterior_median",
                "posterior_std",
                "map",
                "p16",
                "p84",
                "posterior_interval_width",
                "posterior_std_over_prior_std",
                "posterior_mean_minus_prior_mean_over_prior_std",
                "n_components",
                "n_active_components",
                "min_distance_arcsec_min",
                "min_distance_arcsec_median",
                "importance_max",
                "importance_median",
                "brightness_median",
            ]
        )
    weights = _normalized_weights(sample_weights, samples.shape[0])
    leverage_by_potfile: dict[str, dict[str, float]] = {}
    if not scaling_rank_df.empty:
        grouped = scaling_rank_df.groupby("potfile_id", sort=False)
        for potfile_id, group_df in grouped:
            leverage_by_potfile[str(potfile_id)] = {
                "n_components": int(len(group_df)),
                "n_active_components": int(np.sum(group_df["selected_active"].astype(bool))),
                "min_distance_arcsec_min": float(group_df["min_distance_arcsec"].min()),
                "min_distance_arcsec_median": float(group_df["min_distance_arcsec"].median()),
                "importance_max": float(group_df["importance"].max()),
                "importance_median": float(group_df["importance"].median()),
                "brightness_median": float(group_df["brightness"].median()),
            }
    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(parameter_specs):
        if spec.component_family != "scaling":
            continue
        values = np.asarray(samples[:, idx], dtype=float)
        q16, q50, q84 = _weighted_quantile(values, weights, [0.16, 0.50, 0.84])
        posterior_mean = float(np.average(values, weights=weights))
        posterior_std = float(
            np.sqrt(np.average(np.square(values - posterior_mean), weights=weights))
        )
        prior_mean = spec.physical_mean if spec.physical_mean is not None else spec.mean
        prior_std = spec.physical_std if spec.physical_std is not None else spec.std
        prior_mean_value = float(prior_mean) if prior_mean is not None else float("nan")
        prior_std_value = float(prior_std) if prior_std is not None else float("nan")
        leverage = leverage_by_potfile.get(
            str(spec.potential_id),
            {
                "n_components": 0,
                "n_active_components": 0,
                "min_distance_arcsec_min": float("nan"),
                "min_distance_arcsec_median": float("nan"),
                "importance_max": float("nan"),
                "importance_median": float("nan"),
                "brightness_median": float("nan"),
            },
        )
        rows.append(
            {
                "potfile_id": str(spec.potential_id),
                "parameter": str(spec.field),
                "label": str(spec.name),
                "prior_kind": str(spec.prior_kind),
                "prior_mean": prior_mean_value,
                "prior_std": prior_std_value,
                "posterior_mean": posterior_mean,
                "posterior_median": float(q50),
                "posterior_std": posterior_std,
                "map": float(best_fit[idx]),
                "p16": float(q16),
                "p84": float(q84),
                "posterior_interval_width": float(q84 - q16),
                "posterior_std_over_prior_std": (
                    float(posterior_std / prior_std_value)
                    if np.isfinite(prior_std_value) and prior_std_value > 0.0
                    else float("nan")
                ),
                "posterior_mean_minus_prior_mean_over_prior_std": (
                    float((posterior_mean - prior_mean_value) / prior_std_value)
                    if np.isfinite(prior_mean_value) and np.isfinite(prior_std_value) and prior_std_value > 0.0
                    else float("nan")
                ),
                **leverage,
            }
        )
    return pd.DataFrame(rows).sort_values(["potfile_id", "parameter"]).reset_index(drop=True)


def _constraint_strength_label(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 0.60:
        return "strongly constrained"
    if value <= 0.85:
        return "moderately constrained"
    return "weakly constrained"


def _prior_shift_label(value: float) -> str:
    if not np.isfinite(value):
        return "prior shift unknown"
    if abs(value) >= 0.75:
        return "material prior shift"
    if abs(value) >= 0.30:
        return "modest prior shift"
    return "close to prior mean"


def _potfile_leverage_label(
    max_importance: float,
    global_max_importance: float,
    min_distance_arcsec: float,
) -> str:
    importance_ratio = (
        float(max_importance / global_max_importance)
        if np.isfinite(max_importance) and np.isfinite(global_max_importance) and global_max_importance > 0.0
        else float("nan")
    )
    if (np.isfinite(importance_ratio) and importance_ratio >= 0.5) or (np.isfinite(min_distance_arcsec) and min_distance_arcsec <= 5.0):
        return "strong local leverage"
    if (np.isfinite(importance_ratio) and importance_ratio >= 0.1) or (np.isfinite(min_distance_arcsec) and min_distance_arcsec <= 20.0):
        return "moderate local leverage"
    return "weak local leverage"


def _write_potfile_constraint_summary_txt(tables_dir: Path, potfile_diag_df: pd.DataFrame) -> None:
    if potfile_diag_df.empty:
        return
    global_max_importance = float(potfile_diag_df["importance_max"].max()) if potfile_diag_df["importance_max"].notna().any() else float("nan")
    lines: list[str] = []
    grouped = potfile_diag_df.groupby("potfile_id", sort=False)
    for potfile_id, group_df in grouped:
        shrink = float(group_df["posterior_std_over_prior_std"].median())
        shift = float(group_df["posterior_mean_minus_prior_mean_over_prior_std"].abs().max())
        min_distance = float(group_df["min_distance_arcsec_min"].min())
        max_importance = float(group_df["importance_max"].max())
        n_components = int(group_df["n_components"].max())
        n_active = int(group_df["n_active_components"].max())
        lines.append(f"{potfile_id}")
        lines.append(
            f"  constraint: {_constraint_strength_label(shrink)} "
            f"(median posterior/prior std={shrink:.3g})"
        )
        lines.append(
            f"  prior shift: {_prior_shift_label(shift)} "
            f"(max |mean-prior|/prior_std={shift:.3g})"
        )
        lines.append(
            f"  leverage: {_potfile_leverage_label(max_importance, global_max_importance, min_distance)} "
            f"(min_distance={min_distance:.3g}\", max_importance={max_importance:.3g})"
        )
        lines.append(f"  active components: {n_active}/{n_components}")
        for row in group_df.itertuples(index=False):
            lines.append(
                "  "
                + f"{row.parameter}: prior={float(row.prior_mean):.6g}±{float(row.prior_std):.6g}, "
                + f"posterior={float(row.posterior_median):.6g}±{float(row.posterior_std):.6g}, "
                + f"best={float(row.map):.6g}"
            )
        lines.append("")
    content = "\n".join(lines).rstrip() + "\n"
    (tables_dir / "potfile_constraint_summary.txt").write_text(content, encoding="utf-8")


def _plot_potfile_prior_posterior(
    plot_dir: Path,
    potfile_diag_df: pd.DataFrame,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
) -> None:
    if potfile_diag_df.empty or samples.size == 0:
        return
    scaling_indices = [idx for idx, spec in enumerate(parameter_specs) if spec.component_family == "scaling"]
    if not scaling_indices:
        return
    scaling_samples = np.asarray(samples[:, scaling_indices], dtype=float)
    nrows = len(scaling_indices)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, max(4, 2.3 * nrows)), sharex=False)
    if nrows == 1:
        axes = [axes]
    for ax, spec, values in zip(axes, [parameter_specs[idx] for idx in scaling_indices], scaling_samples.T):
        row = potfile_diag_df[(potfile_diag_df["potfile_id"] == str(spec.potential_id)) & (potfile_diag_df["parameter"] == str(spec.field))]
        if row.empty:
            continue
        record = row.iloc[0]
        finite_values = np.asarray(values[np.isfinite(values)], dtype=float)
        if finite_values.size == 0:
            continue
        ax.hist(
            finite_values,
            bins=min(40, max(10, int(np.sqrt(finite_values.size)))),
            density=True,
            color="tab:blue",
            alpha=0.45,
            label="posterior",
        )
        if np.isfinite(record["prior_mean"]) and np.isfinite(record["prior_std"]) and float(record["prior_std"]) > 0.0:
            x_min = min(float(np.min(finite_values)), float(record["prior_mean"] - 4.0 * record["prior_std"]))
            x_max = max(float(np.max(finite_values)), float(record["prior_mean"] + 4.0 * record["prior_std"]))
            x_grid = np.linspace(x_min, x_max, 400)
            ax.plot(
                x_grid,
                norm.pdf(x_grid, loc=float(record["prior_mean"]), scale=float(record["prior_std"])),
                color="tab:orange",
                linewidth=1.8,
                label="prior",
            )
        ax.axvline(float(record["posterior_median"]), color="tab:blue", linewidth=1.5, linestyle="-", label="median")
        ax.axvline(float(record["map"]), color="tab:red", linewidth=1.5, linestyle="--", label="best fit")
        ax.set_title(str(record["label"]))
        ax.set_ylabel("density")
    axes[-1].set_xlabel("parameter value")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "potfile_prior_posterior.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_potfile_constraint_strength(plot_dir: Path, potfile_diag_df: pd.DataFrame) -> None:
    if potfile_diag_df.empty:
        return
    plot_df = potfile_diag_df.copy().sort_values("posterior_std_over_prior_std", ascending=False)
    fig, ax = plt.subplots(1, 1, figsize=(10, max(4, 0.6 * len(plot_df))))
    labels = plot_df["label"].astype(str).tolist()
    values = np.asarray(plot_df["posterior_std_over_prior_std"], dtype=float)
    ax.barh(labels, values, color="tab:green", alpha=0.8)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("posterior std / prior std")
    ax.set_title("Potfile Constraint Brightness")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "potfile_constraint_strength.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_potfile_prior_shift(plot_dir: Path, potfile_diag_df: pd.DataFrame) -> None:
    if potfile_diag_df.empty:
        return
    plot_df = potfile_diag_df.copy().sort_values("posterior_mean_minus_prior_mean_over_prior_std", ascending=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, max(4, 0.6 * len(plot_df))))
    labels = plot_df["label"].astype(str).tolist()
    values = np.asarray(plot_df["posterior_mean_minus_prior_mean_over_prior_std"], dtype=float)
    colors = ["tab:red" if value < 0.0 else "tab:blue" for value in values]
    ax.barh(labels, values, color=colors, alpha=0.8)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("(posterior mean - prior mean) / prior std")
    ax.set_title("Potfile Prior Shift")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "potfile_prior_shift.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_potfile_leverage_summary(plot_dir: Path, potfile_diag_df: pd.DataFrame) -> None:
    if potfile_diag_df.empty:
        return
    pot_df = (
        potfile_diag_df.groupby("potfile_id", sort=False)
        .agg(
            n_components=("n_components", "max"),
            n_active_components=("n_active_components", "max"),
            min_distance_arcsec_min=("min_distance_arcsec_min", "min"),
            min_distance_arcsec_median=("min_distance_arcsec_median", "median"),
            importance_max=("importance_max", "max"),
            importance_median=("importance_median", "median"),
            brightness_median=("brightness_median", "median"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    pot_labels = pot_df["potfile_id"].astype(str).tolist()
    axes[0, 0].bar(pot_labels, np.asarray(pot_df["n_active_components"], dtype=float), color="tab:orange")
    axes[0, 0].set_title("Active Components")
    axes[0, 0].set_ylabel("count")
    axes[0, 1].bar(pot_labels, np.asarray(pot_df["min_distance_arcsec_min"], dtype=float), color="tab:purple")
    axes[0, 1].set_title("Closest Distance To Images")
    axes[0, 1].set_ylabel("arcsec")
    importance_vals, floor = _logsafe_importance_values(pot_df["importance_max"])
    axes[1, 0].bar(pot_labels, importance_vals, color="tab:green")
    axes[1, 0].set_title("Max Importance")
    axes[1, 0].set_ylabel("importance")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(bottom=floor * 0.8)
    axes[1, 1].bar(pot_labels, np.asarray(pot_df["brightness_median"], dtype=float), color="tab:blue")
    axes[1, 1].set_title("Median Brightness")
    axes[1, 1].set_ylabel("brightness")
    for ax in axes.ravel():
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "potfile_leverage_summary.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _family_diagnostics_table(evaluator: ClusterJAXEvaluator, best_eval: EvaluationResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in evaluator.state.family_data:
        cache = evaluator.validation_cache[family.family_id]
        pred = best_eval.family_predictions[family.family_id]
        exact_rms = pred.get("exact_image_rms")
        if exact_rms is None or not np.isfinite(exact_rms):
            rms_residual = pred.get("approx_image_rms_arcsec", pred.get("source_plane_rms"))
        else:
            rms_residual = exact_rms
        rows.append(
            {
                "family_id": family.family_id,
                "z_source": family.z_source,
                "effective_z_source": family.effective_z_source,
                "n_images": family.n_images,
                "sigma_arcsec": family.sigma_arcsec,
                "source_sigma_int_arcsec": pred.get("source_sigma_int_arcsec"),
                "source_sigma_eff_arcsec": pred.get("source_sigma_eff_arcsec"),
                "source_plane_rms_arcsec": pred.get("source_plane_rms"),
                "approx_image_rms_arcsec": pred.get("approx_image_rms_arcsec"),
                "exact_image_rms_arcsec": pred.get("exact_image_rms"),
                "rms_residual_arcsec": rms_residual,
                "max_residual_arcsec": pred.get("residual_max"),
                "used_exact_refresh": pred.get("used_exact_refresh", False),
                "refresh_reason": pred.get("refresh_reason"),
                "exact_validation_count": cache.exact_validation_count,
                "multiplicity_mismatch_count": cache.multiplicity_mismatch_count,
                "match_failure_count": cache.match_failure_count,
            }
        )
    return pd.DataFrame(rows).sort_values(by="rms_residual_arcsec", ascending=False, na_position="last")


def _arc_aware_family_diagnostics_from_image_rows(image_df: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "family_id",
        "arc_aware_image_rms_arcsec",
        "arc_aware_recovered_image_count",
        "arc_aware_missing_image_count",
        "arc_supported_image_count",
    ]
    if image_df is None or image_df.empty or "family_id" not in image_df:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for family_id, group_df in image_df.groupby("family_id", sort=False):
        if "arc_aware_image_residual_arcsec" in group_df:
            residuals = pd.to_numeric(group_df["arc_aware_image_residual_arcsec"], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(residuals)
            recovered_count = int(np.sum(finite))
            missing_count = int(max(0, len(group_df) - recovered_count))
            rms = float(np.sqrt(np.mean(np.square(residuals[finite])))) if np.any(finite) else np.nan
        else:
            recovered_count = 0
            missing_count = int(len(group_df))
            rms = np.nan
        if "arc_supported" in group_df:
            supported_count = int(np.sum(group_df["arc_supported"].astype(bool).to_numpy()))
        elif "arc_recovery_status" in group_df:
            supported_count = int(np.sum(group_df["arc_recovery_status"].astype(str).to_numpy() == "arc_supported"))
        else:
            supported_count = 0
        rows.append(
            {
                "family_id": str(family_id),
                "arc_aware_image_rms_arcsec": rms,
                "arc_aware_recovered_image_count": recovered_count,
                "arc_aware_missing_image_count": missing_count,
                "arc_supported_image_count": supported_count,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _red1_pos_sigma_arcsec(
    residual2: np.ndarray,
    image_sigma_int: np.ndarray | None,
    covariance_floor: np.ndarray | None,
    dof: int,
) -> float | None:
    if dof <= 0 or residual2.size == 0 or image_sigma_int is None or covariance_floor is None:
        return None
    residual2 = np.asarray(residual2, dtype=float).reshape(-1)
    image_sigma_int = np.asarray(image_sigma_int, dtype=float).reshape(-1)
    covariance_floor = np.asarray(covariance_floor, dtype=float).reshape(-1)
    if residual2.shape != image_sigma_int.shape or residual2.shape != covariance_floor.shape:
        return None
    floor_variance = np.square(image_sigma_int) + np.maximum(covariance_floor, 0.0)
    if not np.all(np.isfinite(residual2 + floor_variance)) or np.any(floor_variance < 0.0):
        return None
    if not np.any(residual2 > 0.0):
        return 0.0

    target = float(dof)

    def chi_for(pos_sigma: float) -> float:
        denom = np.square(float(pos_sigma)) + floor_variance
        if np.any(denom <= 0.0):
            return float("inf")
        return float(np.sum(residual2 / denom))

    chi_at_zero = chi_for(0.0)
    if not np.isfinite(chi_at_zero):
        floor_variance = np.maximum(floor_variance, 1.0e-18)
        chi_at_zero = chi_for(0.0)
    if not np.isfinite(chi_at_zero):
        return None
    if chi_at_zero <= target:
        return 0.0

    high = math.sqrt(float(np.sum(residual2)) / target)
    if not np.isfinite(high) or high <= 0.0:
        return None
    low = 0.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if chi_for(mid) > target:
            low = mid
        else:
            high = mid
    return float(high)


def _fit_quality_chi_square_summary(
    image_fit_quality_df: pd.DataFrame | None,
    state: BuildState,
) -> dict[str, Any]:
    parameter_specs = list(getattr(state, "parameter_specs", []))
    n_families = int(len(getattr(state, "family_data", [])))
    n_observed_images = int(sum(int(getattr(family, "n_images", 0)) for family in getattr(state, "family_data", [])))
    sampled_non_source_parameters = int(
        sum(getattr(spec, "component_family", None) != "source_position" for spec in parameter_specs)
    )
    source_position_parameters = int(2 * n_families)
    k_effective = int(sampled_non_source_parameters + source_position_parameters)
    empty = {
        "observed_image_count": n_observed_images,
        "n_effective_parameters": k_effective,
        "sampled_non_source_position_parameters": sampled_non_source_parameters,
        "source_position_parameters": source_position_parameters,
        "chi_square_sigma_basis": "image_sigma_eff_arcsec",
        "chi_square_sigma_eff_median_arcsec": None,
        "chi_square_sigma_eff_min_arcsec": None,
        "chi_square_sigma_eff_max_arcsec": None,
        "chi_square_red1_calibration_note": "post-fit diagnostic; holds image_sigma_int fixed",
        "headline_chi_square_red1_total_sigma_arcsec": None,
        "headline_chi_square_red1_pos_sigma_arcsec": None,
        "arc_aware_chi_square_red1_total_sigma_arcsec": None,
        "arc_aware_chi_square_red1_pos_sigma_arcsec": None,
        "headline_chi_square": None,
        "headline_n_data": 0,
        "headline_dof": int(-k_effective),
        "headline_reduced_chi_square": None,
        "headline_point_image_count": 0,
        "headline_missing_image_count": n_observed_images,
        "arc_aware_chi_square": None,
        "arc_aware_n_data": 0,
        "arc_aware_dof": int(-k_effective),
        "arc_aware_reduced_chi_square": None,
        "arc_aware_point_image_count": 0,
        "arc_aware_arc_supported_image_count": 0,
        "arc_aware_missing_image_count": n_observed_images,
        "image_residual_mean_arcsec": None,
        "image_residual_median_arcsec": None,
        "image_residual_max_arcsec": None,
        "arc_aware_valid_image_count": 0,
        "arc_aware_image_rms_arcsec": None,
        "arc_aware_image_residual_mean_arcsec": None,
        "arc_aware_image_residual_median_arcsec": None,
        "arc_aware_image_residual_max_arcsec": None,
        "covered_xy_1sigma_fraction": None,
    }
    if image_fit_quality_df is None or image_fit_quality_df.empty:
        return empty
    df = image_fit_quality_df.copy()
    observed_image_count = n_observed_images if n_observed_images > 0 else int(len(df))
    empty["observed_image_count"] = observed_image_count
    empty["headline_missing_image_count"] = observed_image_count
    empty["arc_aware_missing_image_count"] = observed_image_count
    required = [
        "x_model_arcsec",
        "y_model_arcsec",
        "x_obs_arcsec",
        "y_obs_arcsec",
        "sigma_arcsec",
        "image_sigma_eff_arcsec",
    ]
    if any(column not in df.columns for column in required):
        return empty
    failed = (
        df["exact_image_prediction_failed"].astype(bool).to_numpy()
        if "exact_image_prediction_failed" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    recovery_status = (
        df["image_recovery_status"].astype(str).to_numpy()
        if "image_recovery_status" in df.columns
        else np.full(len(df), "", dtype=object)
    )
    arc_recovery_status = (
        df["arc_recovery_status"].astype(str).to_numpy()
        if "arc_recovery_status" in df.columns
        else np.full(len(df), "", dtype=object)
    )
    if "arc_supported" in df.columns:
        arc_supported = df["arc_supported"].astype(bool).to_numpy()
    else:
        arc_supported = arc_recovery_status == "arc_supported"
    has_recovery_status = "image_recovery_status" in df.columns or "arc_recovery_status" in df.columns
    point_recovered = (recovery_status == "recovered") | (arc_recovery_status == "point_recovered")
    if not has_recovery_status:
        point_recovered = ~failed
    x_model = pd.to_numeric(df["x_model_arcsec"], errors="coerce").to_numpy(dtype=float)
    y_model = pd.to_numeric(df["y_model_arcsec"], errors="coerce").to_numpy(dtype=float)
    x_obs = pd.to_numeric(df["x_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    y_obs = pd.to_numeric(df["y_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    sigma_meas = pd.to_numeric(df["sigma_arcsec"], errors="coerce").to_numpy(dtype=float)
    sigma_eff = pd.to_numeric(df["image_sigma_eff_arcsec"], errors="coerce").to_numpy(dtype=float)
    image_sigma_int = (
        pd.to_numeric(df["image_sigma_int_arcsec"], errors="coerce").to_numpy(dtype=float)
        if "image_sigma_int_arcsec" in df.columns
        else None
    )
    covariance_floor = None
    if image_sigma_int is not None:
        covariance_floor = np.maximum(np.square(sigma_eff) - np.square(sigma_meas) - np.square(image_sigma_int), 0.0)
    point_valid = (
        point_recovered
        & (~failed)
        & np.isfinite(x_model + y_model + x_obs + y_obs + sigma_eff)
        & (sigma_eff > 0.0)
    )
    dx = x_model[point_valid] - x_obs[point_valid]
    dy = y_model[point_valid] - y_obs[point_valid]
    point_residual2 = np.square(dx) + np.square(dy)
    residuals = np.sqrt(np.square(dx) + np.square(dy))
    point_chi_square = (
        float(np.sum(point_residual2 / np.square(sigma_eff[point_valid])))
        if dx.size
        else 0.0
    )
    point_count = int(np.sum(point_valid))
    headline_n_data = int(2 * point_count)
    headline_dof = int(headline_n_data - k_effective)

    if "arc_aware_image_residual_arcsec" in df.columns:
        arc_residual = pd.to_numeric(df["arc_aware_image_residual_arcsec"], errors="coerce").to_numpy(dtype=float)
    else:
        arc_residual = np.full(len(df), np.nan, dtype=float)
    if "arc_curve_distance_arcsec" in df.columns:
        curve_distance = pd.to_numeric(df["arc_curve_distance_arcsec"], errors="coerce").to_numpy(dtype=float)
        arc_residual = np.where(np.isfinite(arc_residual), arc_residual, curve_distance)
    arc_valid = (
        (~point_recovered)
        & arc_supported
        & np.isfinite(arc_residual + sigma_eff)
        & (sigma_eff > 0.0)
    )
    arc_supported_count = int(np.sum(arc_valid))
    arc_residual2 = np.square(arc_residual[arc_valid])
    arc_chi_square = (
        point_chi_square + float(np.sum(arc_residual2 / np.square(sigma_eff[arc_valid])))
        if point_count or arc_supported_count
        else 0.0
    )
    arc_aware_n_data = int(2 * point_count + arc_supported_count)
    arc_aware_dof = int(arc_aware_n_data - k_effective)
    arc_aware_residuals = (
        np.concatenate([residuals, arc_residual[arc_valid]])
        if point_count or arc_supported_count
        else np.asarray([])
    )
    coverage_fraction = None
    if "covered_xy_1sigma" in df.columns:
        coverage_values = df.loc[point_valid, "covered_xy_1sigma"].astype(bool).to_numpy()
        coverage_fraction = float(np.mean(coverage_values)) if coverage_values.size else None
    chi_sigma_values = sigma_eff[point_valid | arc_valid]
    point_sum_squares = float(np.sum(point_residual2))
    arc_sum_squares = point_sum_squares + float(np.sum(arc_residual2))
    headline_red1_total_sigma = (
        float(math.sqrt(point_sum_squares / headline_dof)) if headline_dof > 0 and point_count else None
    )
    arc_aware_red1_total_sigma = (
        float(math.sqrt(arc_sum_squares / arc_aware_dof))
        if arc_aware_dof > 0 and (point_count or arc_supported_count)
        else None
    )
    point_image_sigma_int = image_sigma_int[point_valid] if image_sigma_int is not None else None
    point_covariance_floor = covariance_floor[point_valid] if covariance_floor is not None else None
    arc_aware_residual2 = (
        np.concatenate([point_residual2, arc_residual2])
        if point_count or arc_supported_count
        else np.asarray([], dtype=float)
    )
    arc_aware_image_sigma_int = (
        np.concatenate([image_sigma_int[point_valid], image_sigma_int[arc_valid]])
        if image_sigma_int is not None and (point_count or arc_supported_count)
        else None
    )
    arc_aware_covariance_floor = (
        np.concatenate([covariance_floor[point_valid], covariance_floor[arc_valid]])
        if covariance_floor is not None and (point_count or arc_supported_count)
        else None
    )
    return {
        "observed_image_count": observed_image_count,
        "n_effective_parameters": k_effective,
        "sampled_non_source_position_parameters": sampled_non_source_parameters,
        "source_position_parameters": source_position_parameters,
        "chi_square_sigma_basis": "image_sigma_eff_arcsec",
        "chi_square_sigma_eff_median_arcsec": float(np.median(chi_sigma_values)) if chi_sigma_values.size else None,
        "chi_square_sigma_eff_min_arcsec": float(np.min(chi_sigma_values)) if chi_sigma_values.size else None,
        "chi_square_sigma_eff_max_arcsec": float(np.max(chi_sigma_values)) if chi_sigma_values.size else None,
        "chi_square_red1_calibration_note": "post-fit diagnostic; holds image_sigma_int fixed",
        "headline_chi_square_red1_total_sigma_arcsec": headline_red1_total_sigma,
        "headline_chi_square_red1_pos_sigma_arcsec": _red1_pos_sigma_arcsec(
            point_residual2,
            point_image_sigma_int,
            point_covariance_floor,
            headline_dof,
        ),
        "arc_aware_chi_square_red1_total_sigma_arcsec": arc_aware_red1_total_sigma,
        "arc_aware_chi_square_red1_pos_sigma_arcsec": _red1_pos_sigma_arcsec(
            arc_aware_residual2,
            arc_aware_image_sigma_int,
            arc_aware_covariance_floor,
            arc_aware_dof,
        ),
        "headline_chi_square": point_chi_square,
        "headline_n_data": headline_n_data,
        "headline_dof": headline_dof,
        "headline_reduced_chi_square": float(point_chi_square / headline_dof) if headline_dof > 0 else None,
        "headline_point_image_count": point_count,
        "headline_missing_image_count": int(max(0, observed_image_count - point_count)),
        "arc_aware_chi_square": arc_chi_square,
        "arc_aware_n_data": arc_aware_n_data,
        "arc_aware_dof": arc_aware_dof,
        "arc_aware_reduced_chi_square": float(arc_chi_square / arc_aware_dof) if arc_aware_dof > 0 else None,
        "arc_aware_point_image_count": point_count,
        "arc_aware_arc_supported_image_count": arc_supported_count,
        "arc_aware_missing_image_count": int(max(0, observed_image_count - point_count - arc_supported_count)),
        "image_residual_mean_arcsec": float(np.mean(residuals)) if residuals.size else None,
        "image_residual_median_arcsec": float(np.median(residuals)) if residuals.size else None,
        "image_residual_max_arcsec": float(np.max(residuals)) if residuals.size else None,
        "arc_aware_valid_image_count": int(point_count + arc_supported_count),
        "arc_aware_image_rms_arcsec": (
            float(np.sqrt(np.mean(np.square(arc_aware_residuals)))) if arc_aware_residuals.size else None
        ),
        "arc_aware_image_residual_mean_arcsec": float(np.mean(arc_aware_residuals)) if arc_aware_residuals.size else None,
        "arc_aware_image_residual_median_arcsec": float(np.median(arc_aware_residuals)) if arc_aware_residuals.size else None,
        "arc_aware_image_residual_max_arcsec": float(np.max(arc_aware_residuals)) if arc_aware_residuals.size else None,
        "covered_xy_1sigma_fraction": coverage_fraction,
    }


def _chain_diagnostics_summary(results: PosteriorResults, parameter_specs: list[ParameterSpec]) -> dict[str, Any]:
    base = {
        "ess_min": None,
        "ess_median": None,
        "ess_worst_parameter": None,
        "rhat_max": None,
        "rhat_median": None,
        "rhat_worst_parameter": None,
    }
    if str(results.sampler) == "numpyro_jaxns":
        ns_ess = None
        if results.init_diagnostics:
            ns_ess = results.init_diagnostics.get("ns_ess")
        if ns_ess is None and results.sample_weights is not None and len(results.sample_weights) > 0:
            ns_ess = _effective_sample_size(results.sample_weights)
        base["ess_min"] = ns_ess
        base["ess_median"] = ns_ess
        return base
    grouped = results.grouped_samples
    if grouped is None:
        return base
    grouped_array = np.asarray(grouped, dtype=float)
    if grouped_array.ndim != 3 or grouped_array.shape[1] < 2 or grouped_array.shape[2] == 0:
        return base
    try:
        from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    except Exception:
        return base
    names = [
        parameter_specs[idx].name if idx < len(parameter_specs) else f"param_{idx}"
        for idx in range(grouped_array.shape[2])
    ]
    ess_values: list[tuple[str, float]] = []
    rhat_values: list[tuple[str, float]] = []
    for idx, name in enumerate(names):
        values = grouped_array[:, :, idx]
        if not np.isfinite(values).all():
            continue
        try:
            ess = float(np.asarray(effective_sample_size(values)).reshape(-1)[0])
        except Exception:
            ess = float("nan")
        if np.isfinite(ess):
            ess_values.append((name, ess))
        if grouped_array.shape[0] >= 2:
            try:
                rhat = float(np.asarray(split_gelman_rubin(values)).reshape(-1)[0])
            except Exception:
                rhat = float("nan")
            if np.isfinite(rhat):
                rhat_values.append((name, rhat))
    if ess_values:
        ess_array = np.asarray([value for _name, value in ess_values], dtype=float)
        worst_name, worst_value = min(ess_values, key=lambda item: item[1])
        base["ess_min"] = float(worst_value)
        base["ess_median"] = float(np.median(ess_array))
        base["ess_worst_parameter"] = worst_name
    if rhat_values:
        rhat_array = np.asarray([value for _name, value in rhat_values], dtype=float)
        worst_name, worst_value = max(rhat_values, key=lambda item: item[1])
        base["rhat_max"] = float(worst_value)
        base["rhat_median"] = float(np.median(rhat_array))
        base["rhat_worst_parameter"] = worst_name
    return base


def _grouped_chain_array(
    value: np.ndarray | None,
    n_chains: int,
    n_draws: int,
    *,
    dtype: Any = float,
) -> np.ndarray | None:
    if value is None or n_chains <= 0 or n_draws <= 0:
        return None
    array = np.asarray(value, dtype=dtype)
    if array.shape == (n_chains, n_draws):
        return array
    flat = array.reshape(-1)
    expected = int(n_chains) * int(n_draws)
    if flat.size != expected:
        return None
    return flat.reshape((n_chains, n_draws))


def _finite_quantiles(values: np.ndarray, quantiles: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return [float("nan") for _quantile in quantiles]
    return [float(value) for value in np.quantile(finite, quantiles)]


def _image_sigma_int_index(parameter_specs: list[ParameterSpec], n_params: int) -> int | None:
    for idx, spec in enumerate(parameter_specs[:n_params]):
        if str(getattr(spec, "sample_name", "")) == "image_sigma_int":
            return idx
    for idx, spec in enumerate(parameter_specs[:n_params]):
        if str(getattr(spec, "name", "")) == "image.sigma_int":
            return idx
    for idx, spec in enumerate(parameter_specs[:n_params]):
        if str(getattr(spec, "component_family", "")) == "image_scatter":
            return idx
    return None


def _chain_health_summary_table(
    results: PosteriorResults,
    parameter_specs: list[ParameterSpec],
    *,
    max_tree_depth: int | None = None,
) -> pd.DataFrame:
    columns = [
        "chain",
        "chain_index",
        "chain_label",
        "n_draws",
        "log_prob_mean",
        "log_prob_median",
        "log_prob_max",
        "accept_prob_mean",
        "divergence_count",
        "max_tree_depth_saturation_fraction",
        "image_sigma_int_q16",
        "image_sigma_int_q50",
        "image_sigma_int_q84",
    ]
    grouped = results.grouped_samples
    if grouped is None:
        return pd.DataFrame(columns=columns)
    grouped_array = np.asarray(grouped, dtype=float)
    if grouped_array.ndim != 3 or grouped_array.shape[0] == 0 or grouped_array.shape[1] == 0:
        return pd.DataFrame(columns=columns)
    n_chains, n_draws, n_params = grouped_array.shape
    grouped_log_prob = (
        np.asarray(results.grouped_log_prob, dtype=float)
        if results.grouped_log_prob is not None
        else _grouped_chain_array(results.log_prob, n_chains, n_draws, dtype=float)
    )
    if grouped_log_prob is not None and grouped_log_prob.shape != (n_chains, n_draws):
        grouped_log_prob = None
    grouped_accept = _grouped_chain_array(results.accept_prob, n_chains, n_draws, dtype=float)
    grouped_diverging = _grouped_chain_array(results.diverging, n_chains, n_draws, dtype=bool)
    grouped_steps = _grouped_chain_array(results.num_steps, n_chains, n_draws, dtype=float)
    step_threshold = (2**int(max_tree_depth) - 1) if max_tree_depth is not None else None
    image_sigma_idx = _image_sigma_int_index(parameter_specs, n_params)
    labels = list((results.init_diagnostics or {}).get("chain_seed_labels", []))
    rows: list[dict[str, Any]] = []
    for chain_idx in range(n_chains):
        log_prob = (
            np.asarray(grouped_log_prob[chain_idx], dtype=float)
            if grouped_log_prob is not None
            else np.asarray([], dtype=float)
        )
        finite_log_prob = log_prob[np.isfinite(log_prob)]
        accept = (
            np.asarray(grouped_accept[chain_idx], dtype=float)
            if grouped_accept is not None
            else np.asarray([], dtype=float)
        )
        finite_accept = accept[np.isfinite(accept)]
        diverging = (
            np.asarray(grouped_diverging[chain_idx], dtype=bool)
            if grouped_diverging is not None
            else np.asarray([], dtype=bool)
        )
        steps = (
            np.asarray(grouped_steps[chain_idx], dtype=float)
            if grouped_steps is not None
            else np.asarray([], dtype=float)
        )
        finite_steps = steps[np.isfinite(steps)]
        if image_sigma_idx is not None:
            sigma_q16, sigma_q50, sigma_q84 = _finite_quantiles(grouped_array[chain_idx, :, image_sigma_idx], [0.16, 0.50, 0.84])
        else:
            sigma_q16 = sigma_q50 = sigma_q84 = float("nan")
        saturation_fraction = float("nan")
        if step_threshold is not None and finite_steps.size:
            saturation_fraction = float(np.mean(finite_steps >= float(step_threshold)))
        rows.append(
            {
                "chain": int(chain_idx + 1),
                "chain_index": int(chain_idx),
                "chain_label": str(labels[chain_idx]) if chain_idx < len(labels) else f"chain {chain_idx + 1}",
                "n_draws": int(n_draws),
                "log_prob_mean": float(np.mean(finite_log_prob)) if finite_log_prob.size else float("nan"),
                "log_prob_median": float(np.median(finite_log_prob)) if finite_log_prob.size else float("nan"),
                "log_prob_max": float(np.max(finite_log_prob)) if finite_log_prob.size else float("nan"),
                "accept_prob_mean": float(np.mean(finite_accept)) if finite_accept.size else float("nan"),
                "divergence_count": int(np.sum(diverging)) if diverging.size else 0,
                "max_tree_depth_saturation_fraction": saturation_fraction,
                "image_sigma_int_q16": sigma_q16,
                "image_sigma_int_q50": sigma_q50,
                "image_sigma_int_q84": sigma_q84,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _chain_parameter_diagnostics_table(
    results: PosteriorResults,
    parameter_specs: list[ParameterSpec],
) -> pd.DataFrame:
    base_columns = [
        "parameter_index",
        "parameter",
        "sample_name",
        "component_family",
        "ess",
        "rhat",
        "chain_median_spread",
        "chain_median_standardized_spread",
    ]
    grouped = results.grouped_samples
    if grouped is None:
        return pd.DataFrame(columns=base_columns)
    grouped_array = np.asarray(grouped, dtype=float)
    if grouped_array.ndim != 3 or grouped_array.shape[0] == 0 or grouped_array.shape[1] == 0 or grouped_array.shape[2] == 0:
        return pd.DataFrame(columns=base_columns)
    n_chains, _n_draws, n_params = grouped_array.shape
    try:
        from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    except Exception:
        effective_sample_size = None
        split_gelman_rubin = None
    rows: list[dict[str, Any]] = []
    for param_idx in range(n_params):
        spec = parameter_specs[param_idx] if param_idx < len(parameter_specs) else None
        values = grouped_array[:, :, param_idx]
        chain_quantiles = [_finite_quantiles(values[chain_idx], [0.16, 0.50, 0.84]) for chain_idx in range(n_chains)]
        chain_medians = np.asarray([item[1] for item in chain_quantiles], dtype=float)
        finite_medians = chain_medians[np.isfinite(chain_medians)]
        finite_values = values[np.isfinite(values)]
        spread = float(np.max(finite_medians) - np.min(finite_medians)) if finite_medians.size else float("nan")
        global_std = float(np.std(finite_values)) if finite_values.size else float("nan")
        standardized_spread = spread / global_std if np.isfinite(spread) and np.isfinite(global_std) and global_std > 0.0 else float("nan")
        ess = float("nan")
        rhat = float("nan")
        if np.isfinite(values).all():
            if effective_sample_size is not None:
                try:
                    ess = float(np.asarray(effective_sample_size(values)).reshape(-1)[0])
                except Exception:
                    ess = float("nan")
            if split_gelman_rubin is not None and n_chains >= 2:
                try:
                    rhat = float(np.asarray(split_gelman_rubin(values)).reshape(-1)[0])
                except Exception:
                    rhat = float("nan")
        row: dict[str, Any] = {
            "parameter_index": int(param_idx),
            "parameter": str(getattr(spec, "name", f"param_{param_idx}")),
            "sample_name": str(getattr(spec, "sample_name", f"param_{param_idx}")),
            "component_family": str(getattr(spec, "component_family", "")),
            "ess": ess,
            "rhat": rhat,
            "chain_median_spread": spread,
            "chain_median_standardized_spread": standardized_spread,
        }
        for chain_idx, (q16, q50, q84) in enumerate(chain_quantiles, start=1):
            row[f"chain_{chain_idx}_q16"] = q16
            row[f"chain_{chain_idx}_q50"] = q50
            row[f"chain_{chain_idx}_q84"] = q84
        rows.append(row)
    return pd.DataFrame(rows)


def _chain_parameter_rank_key(row: Any) -> tuple[float, float, float]:
    rhat = getattr(row, "rhat", float("nan"))
    standardized = getattr(row, "chain_median_standardized_spread", float("nan"))
    spread = getattr(row, "chain_median_spread", float("nan"))
    rhat_rank = float(rhat) if np.isfinite(float(rhat)) else -float("inf")
    standardized_rank = float(standardized) if np.isfinite(float(standardized)) else -float("inf")
    spread_rank = abs(float(spread)) if np.isfinite(float(spread)) else -float("inf")
    return (rhat_rank, standardized_rank, spread_rank)


def _ranked_chain_trace_subset(
    grouped_samples: np.ndarray | None,
    parameter_specs: list[ParameterSpec],
    parameter_diagnostics: pd.DataFrame | None = None,
    *,
    max_params: int = 8,
) -> tuple[np.ndarray, list[ParameterSpec]] | None:
    if grouped_samples is None or not parameter_specs or max_params <= 0:
        return None
    grouped_array = np.asarray(grouped_samples, dtype=float)
    if grouped_array.ndim != 3 or grouped_array.shape[0] == 0 or grouped_array.shape[1] == 0 or grouped_array.shape[2] == 0:
        return None
    if parameter_diagnostics is None or parameter_diagnostics.empty:
        parameter_diagnostics = _chain_parameter_diagnostics_table(
            PosteriorResults(
                samples=grouped_array.reshape((-1, grouped_array.shape[-1])),
                log_prob=np.empty((0,), dtype=float),
                accept_prob=np.empty((0,), dtype=float),
                diverging=np.empty((0,), dtype=bool),
                num_steps=np.empty((0,), dtype=float),
                warmup_steps=0,
                sample_steps=grouped_array.shape[1],
                num_chains=grouped_array.shape[0],
                grouped_samples=grouped_array,
            ),
            parameter_specs,
        )
    if parameter_diagnostics.empty or "parameter_index" not in parameter_diagnostics.columns:
        return None
    rows = [
        row
        for row in parameter_diagnostics.itertuples(index=False)
        if 0 <= int(getattr(row, "parameter_index")) < min(grouped_array.shape[2], len(parameter_specs))
    ]
    if not rows:
        return None
    source_count = sum(
        str(getattr(parameter_specs[int(getattr(row, "parameter_index"))], "component_family", "")) == "source_position"
        for row in rows
    )
    prefer_non_source = source_count > max_params and len(rows) > max_params
    if prefer_non_source:
        non_source_rows = [
            row
            for row in rows
            if str(getattr(parameter_specs[int(getattr(row, "parameter_index"))], "component_family", "")) != "source_position"
        ]
        source_rows = [
            row
            for row in rows
            if str(getattr(parameter_specs[int(getattr(row, "parameter_index"))], "component_family", "")) == "source_position"
        ]
        ranked_rows = sorted(
            non_source_rows,
            key=lambda row: _chain_parameter_rank_key(row),
            reverse=True,
        )
        if len(ranked_rows) < max_params:
            ranked_rows.extend(
                sorted(
                    source_rows,
                    key=lambda row: _chain_parameter_rank_key(row),
                    reverse=True,
                )
            )
    else:
        ranked_rows = sorted(
            rows,
            key=lambda row: _chain_parameter_rank_key(row),
            reverse=True,
        )
    selected_indices = [int(getattr(row, "parameter_index")) for row in ranked_rows[:max_params]]
    if not selected_indices:
        return None
    return grouped_array[:, :, selected_indices], [parameter_specs[idx] for idx in selected_indices]


def _run_summary(
    args: argparse.Namespace,
    state: BuildState,
    runtime_sec: float,
    results: PosteriorResults,
    best_loglike: float,
    evaluator: ClusterJAXEvaluator,
    image_fit_quality_df: pd.DataFrame | None = None,
    image_count_recovery_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    init_diagnostics = dict(results.init_diagnostics or {})
    run_name = str(getattr(args, "run_name", None) or state.run_name)
    geometry_cache = getattr(state, "geometry_cache", None)
    sample_likelihood_mode = str(getattr(args, "sample_likelihood_mode", "source"))
    source_redshifts = np.asarray([float(family.z_source) for family in state.family_data], dtype=float)
    finite_source_redshifts = source_redshifts[np.isfinite(source_redshifts)]
    lens_redshift = getattr(state, "z_lens", None)
    chi_square_summary = _fit_quality_chi_square_summary(image_fit_quality_df, state)
    image_count_summary = _image_count_recovery_summary(image_count_recovery_df)
    chain_summary = _chain_diagnostics_summary(results, state.parameter_specs)
    def _cosmology_summary_fields() -> dict[str, Any]:
        fields: dict[str, Any] = {
            "fit_cosmology_flat_wcdm": bool(getattr(state, "fit_cosmology_flat_wcdm", False)),
            "cosmology_H0_fixed": float(dict(getattr(state, "cosmo_config", {})).get("H0", np.nan)),
            "cosmology_redshift_binning": "fiducial_fixed",
        }
        sample_array = np.asarray(results.samples, dtype=float)
        log_prob = np.asarray(results.log_prob, dtype=float)
        best_index = None
        if sample_array.ndim == 2 and log_prob.shape[0] == sample_array.shape[0] and np.isfinite(log_prob).any():
            best_index = int(np.nanargmax(log_prob))
        for idx, spec in enumerate(state.parameter_specs):
            if spec.sample_name not in {"cosmology_Om0", "cosmology_w0"}:
                continue
            values = sample_array[:, idx] if sample_array.ndim == 2 and sample_array.shape[1] > idx else np.asarray([])
            finite = values[np.isfinite(values)]
            prefix = spec.sample_name
            fields[f"{prefix}_best"] = (
                float(sample_array[best_index, idx])
                if best_index is not None and sample_array.shape[1] > idx and np.isfinite(sample_array[best_index, idx])
                else None
            )
            if finite.size:
                q16, q50, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
                fields[f"{prefix}_q16"] = float(q16)
                fields[f"{prefix}_median"] = float(q50)
                fields[f"{prefix}_q84"] = float(q84)
            else:
                fields[f"{prefix}_q16"] = None
                fields[f"{prefix}_median"] = None
                fields[f"{prefix}_q84"] = None
        return fields

    max_tree_depth = _first_int_value(getattr(args, "max_tree_depth", 10), 10)
    summary = {
        "run_name": run_name,
        "par_path": state.par_path,
        "fit_mode": state.fit_mode,
        "n_parameters": len(state.parameter_specs),
        "n_families": len(state.family_data),
        "n_images": int(sum(f.n_images for f in state.family_data)),
        "n_large_scale_parameters": int(sum(spec.component_family == "large" for spec in state.parameter_specs)),
        "n_scaling_parameters": int(sum(spec.component_family == "scaling" for spec in state.parameter_specs)),
        "n_source_scatter_parameters": int(sum(spec.component_family == "source_scatter" for spec in state.parameter_specs)),
        "n_source_position_parameters": int(sum(spec.component_family == "source_position" for spec in state.parameter_specs)),
        "n_image_scatter_parameters": int(sum(spec.component_family == "image_scatter" for spec in state.parameter_specs)),
        "n_cosmology_parameters": int(sum(spec.component_family == "cosmology" for spec in state.parameter_specs)),
        "n_scaling_galaxy_components": int(np.sum(state.packed_lens_spec.component_family == 1)),
        "z_lens": float(lens_redshift) if lens_redshift is not None and np.isfinite(float(lens_redshift)) else None,
        "z_source_min": float(np.min(finite_source_redshifts)) if finite_source_redshifts.size else None,
        "z_source_max": float(np.max(finite_source_redshifts)) if finite_source_redshifts.size else None,
        "fit_method": str(getattr(args, "fit_method", "svi+nuts")),
        "sample_likelihood_mode": sample_likelihood_mode,
        "image_plane_mode": str(getattr(args, "image_plane_mode", "none")),
        "skip_stage3_image_plane_local_jacobian": bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)),
        "quick_diagnostics": bool(getattr(args, "quick_diagnostics", False)),
        "image_plane_newton_steps": int(getattr(args, "image_plane_newton_steps", 0)),
        "image_plane_scatter_floor_arcsec": float(getattr(args, "image_plane_scatter_floor_arcsec", 0.0)),
        "arc_aware_noncritical_support_radius_arcsec": float(
            getattr(
                args,
                "arc_aware_noncritical_support_radius_arcsec",
                getattr(evaluator, "arc_aware_noncritical_support_radius_arcsec", 0.5),
            )
        ),
        "arc_aware_max_arclength_arcsec": float(
            getattr(
                args,
                "arc_aware_max_arclength_arcsec",
                getattr(evaluator, "arc_aware_max_arclength_arcsec", 5.0),
            )
        ),
        "arc_aware_curve_step_arcsec": float(
            getattr(
                args,
                "arc_aware_curve_step_arcsec",
                getattr(evaluator, "arc_aware_curve_step_arcsec", 0.1),
            )
        ),
        "image_plane_scatter_prior": str(getattr(args, "image_plane_scatter_prior", "log-uniform")),
        "image_plane_scatter_prior_median_arcsec": float(
            getattr(args, "image_plane_scatter_prior_median_arcsec", 0.3)
        ),
        "image_plane_scatter_prior_log_sigma": float(
            getattr(args, "image_plane_scatter_prior_log_sigma", 0.5)
        ),
        "likelihood_stabilizer_max_gain": float(getattr(args, "likelihood_stabilizer_max_gain", 0.0)),
        "likelihood_stabilizer_max_residual_arcsec": float(
            getattr(args, "likelihood_stabilizer_max_residual_arcsec", 0.0)
        ),
        "likelihood_stabilizer_residual_loss": str(getattr(args, "likelihood_stabilizer_residual_loss", "gaussian")),
        "likelihood_stabilizer_student_t_nu": float(getattr(args, "likelihood_stabilizer_student_t_nu", 4.0)),
        "source_position_parameterization": str(getattr(args, "source_position_parameterization", "prior-whitened")),
        "evidence_source_prior_sigma_arcsec": (
            None
            if getattr(args, "evidence_source_prior_sigma_arcsec", None) is None
            else float(getattr(args, "evidence_source_prior_sigma_arcsec"))
        ),
        "evidence_source_prior_mean_x_arcsec": float(getattr(args, "evidence_source_prior_mean_x_arcsec", 0.0)),
        "evidence_source_prior_mean_y_arcsec": float(getattr(args, "evidence_source_prior_mean_y_arcsec", 0.0)),
        **_cosmology_summary_fields(),
        "sampling_engine": args.sampling_engine,
        "fit_sampling_engine": init_diagnostics.get(
            "fit_sampling_engine",
            str(getattr(args, "sampling_engine", getattr(evaluator, "sampling_engine", "full"))),
        ),
        "final_validation_sampling_engine": init_diagnostics.get(
            "final_validation_sampling_engine",
            str(getattr(evaluator, "final_validation_sampling_engine", getattr(evaluator, "sampling_engine", "full"))),
        ),
        "fit_active_subset_loglike": init_diagnostics.get("fit_active_subset_loglike"),
        "full_model_validation_loglike": best_loglike
        if str(
            init_diagnostics.get(
                "fit_sampling_engine",
                getattr(args, "sampling_engine", getattr(evaluator, "sampling_engine", "full")),
            )
        )
        == "active_subset"
        else None,
        "active_scaling_galaxies": list(evaluator.active_scaling_galaxies_by_potfile),
        "active_scaling_components": int(len(evaluator.active_scaling_component_indices)),
        "inactive_scaling_components": int(len(evaluator.inactive_scaling_component_indices)),
        "requested_active_scaling_by_potfile": evaluator.requested_active_scaling_by_potfile,
        "actual_active_scaling_by_potfile": evaluator.actual_active_scaling_by_potfile,
        "total_scaling_by_potfile": evaluator.total_scaling_by_potfile,
        "fit_quality_reference_sample_kind": str(
            init_diagnostics.get("fit_quality_reference_sample_kind", "max_likelihood")
        ),
        "fit_quality_reference_sample_index": init_diagnostics.get("max_likelihood_sample_index"),
        "fit_quality_reference_source_loglike": init_diagnostics.get("max_likelihood_source_loglike"),
        "fit_quality_reference_log_prob": init_diagnostics.get("max_likelihood_sample_log_prob"),
        "sampler": str(results.sampler),
        "nuts_init_strategy_requested": init_diagnostics.get("strategy_requested", getattr(args, "nuts_init_strategy", "svi")),
        "nuts_init_strategy_used": init_diagnostics.get("strategy_used", getattr(args, "nuts_init_strategy", "svi")),
        "nuts_init_settings": {
            "boundary_frac": float(getattr(args, "nuts_init_boundary_frac", DEFAULT_NUTS_INIT_BOUNDARY_FRAC)),
            "jitter_frac": float(getattr(args, "nuts_init_jitter_frac", DEFAULT_NUTS_INIT_JITTER_FRAC)),
            "svi_steps": int(getattr(args, "svi_steps", DEFAULT_SVI_STEPS)),
            "svi_learning_rate": float(getattr(args, "svi_learning_rate", DEFAULT_SVI_LEARNING_RATE)),
        },
        "nuts_init_diagnostics": {
            "distinct_chain_seeds": int(init_diagnostics.get("distinct_chain_seeds", 0)),
            "svi_used": bool(init_diagnostics.get("svi_used", False)),
            "svi_final_elbo_loss": float(init_diagnostics.get("svi_final_elbo_loss", float("nan"))),
            "chain_seed_labels": list(init_diagnostics.get("chain_seed_labels", [])),
            "svi_chain_start_labels": list(init_diagnostics.get("svi_chain_start_labels", [])),
            "requested_chains": int(init_diagnostics.get("requested_chains", results.num_chains)),
            "retained_finite_chains": int(init_diagnostics.get("retained_finite_chains", results.num_chains)),
            "dropped_nonfinite_chains": int(init_diagnostics.get("dropped_nonfinite_chains", 0)),
            "retained_chain_indices": list(init_diagnostics.get("retained_chain_indices", list(range(results.num_chains)))),
            "dropped_chain_indices": list(init_diagnostics.get("dropped_chain_indices", [])),
            "invalid_state_rejection_count": int(init_diagnostics.get("invalid_state_rejection_count", evaluator.invalid_state_rejection_count)),
            "invalid_state_reason_counts": {
                key: int(value)
                for key, value in dict(
                    init_diagnostics.get("invalid_state_reason_counts", evaluator.invalid_state_reason_counts)
                ).items()
            },
            "quality_metrics": dict(init_diagnostics.get("nuts_quality_metrics", {})),
            "quality_warnings": list(init_diagnostics.get("nuts_quality_warnings", [])),
            "svi_health_metrics": dict(init_diagnostics.get("svi_health_metrics") or {}),
            "svi_health_warnings": list(init_diagnostics.get("svi_health_warnings") or []),
        },
        "ns_settings": {
            "num_live_points": init_diagnostics.get("ns_num_live_points", getattr(args, "ns_num_live_points", None)),
            "max_samples": init_diagnostics.get("ns_max_samples", getattr(args, "ns_max_samples", None)),
            "dlogz": init_diagnostics.get("ns_dlogz", getattr(args, "ns_dlogz", None)),
        },
        "ns_log_z_mean": init_diagnostics.get("ns_log_z_mean"),
        "ns_log_z_uncert": init_diagnostics.get("ns_log_z_uncert"),
        "ns_ess": init_diagnostics.get("ns_ess"),
        "ns_total_num_samples": init_diagnostics.get("ns_total_num_samples"),
        "ns_posterior_samples": init_diagnostics.get("ns_posterior_samples"),
        "ns_posterior_resampling": init_diagnostics.get("ns_posterior_resampling"),
        "ns_total_num_likelihood_evaluations": init_diagnostics.get("ns_total_num_likelihood_evaluations"),
        "ns_termination_reason": init_diagnostics.get("ns_termination_reason"),
        "warmup": args.warmup,
        "samples": args.samples,
        "chains": results.num_chains,
        "requested_chains": int(init_diagnostics.get("requested_chains", args.chains)),
        "thin": args.thin,
        "max_tree_depth": max_tree_depth,
        "target_accept": args.target_accept,
        "runtime_sec": runtime_sec,
        "best_loglike": best_loglike,
        "seed": args.seed,
        "packed_fast_path": True,
        "uses_potfile_scaling": bool(state.potfiles and state.fit_mode in {"small-only", "joint"}),
        "surrogate_enabled": bool(evaluator.surrogate_enabled),
        "approximate_eval_count": int(evaluator.approximate_eval_count),
        "full_refresh_count": int(evaluator.full_refresh_count),
        "invalid_state_rejection_count": int(evaluator.invalid_state_rejection_count),
        "invalid_state_reason_counts": {key: int(value) for key, value in evaluator.invalid_state_reason_counts.items()},
        "stage2_large_scale_priors": {
            spec.sample_name: {"mean": spec.mean, "std": spec.std}
            for spec in state.parameter_specs
            if state.fit_mode == "small-only"
            and spec.component_family == "large"
            and spec.prior_kind in {"normal", "truncated_normal"}
        },
        "geometry_setup_timing_sec": {
            "family_redshift_binning": float(geometry_cache.family_redshift_binning_sec)
            if geometry_cache is not None
            else None,
            "geometry_cache_build": float(geometry_cache.geometry_cache_build_sec)
            if geometry_cache is not None
            else None,
        },
        "distinct_effective_source_planes": len({family.effective_z_source for family in state.family_data}),
        "mean_eval_wall_time_sec": float(np.mean(evaluator.eval_wall_times)) if evaluator.eval_wall_times else None,
        "median_eval_wall_time_sec": float(np.median(evaluator.eval_wall_times)) if evaluator.eval_wall_times else None,
        "timing_totals_sec": {key: float(value) for key, value in evaluator.timing_totals.items()},
        "accept_prob_mean": float(np.mean(results.accept_prob)) if results.accept_prob.size else None,
        "divergence_count": int(np.sum(results.diverging)) if results.diverging.size else 0,
        "mean_num_steps": float(np.mean(results.num_steps)) if results.num_steps.size else None,
        "max_tree_depth_saturation_fraction": (
            float(
                np.mean(
                    np.asarray(results.num_steps, dtype=float)
                    >= (2**max_tree_depth - 1)
                )
            )
            if results.num_steps.size
            else None
        ),
        "svi_health_metrics": dict(init_diagnostics.get("svi_health_metrics") or {}),
        "svi_health_warnings": list(init_diagnostics.get("svi_health_warnings") or []),
        "nuts_quality_warnings": list(init_diagnostics.get("nuts_quality_warnings", [])),
        "sample_weight_ess": float(_effective_sample_size(results.sample_weights))
        if results.sample_weights is not None and len(results.sample_weights) > 0
        else None,
        "temperature_schedule": results.temperature_schedule.tolist() if results.temperature_schedule is not None else None,
        "ess_history": results.ess_history.tolist() if results.ess_history is not None else None,
        "move_acceptance_history": results.move_acceptance_history.tolist() if results.move_acceptance_history is not None else None,
        **chi_square_summary,
        **image_count_summary,
        **chain_summary,
    }
    image_scatter_indices = [
        idx for idx, spec in enumerate(state.parameter_specs) if spec.component_family == "image_scatter"
    ]
    fixed_image_sigma_int = getattr(args, "fix_image_sigma_int_arcsec", None)
    summary["fixed_image_sigma_int_arcsec"] = (
        None if fixed_image_sigma_int is None else float(fixed_image_sigma_int)
    )
    summary["image_sigma_int_sampled"] = bool(image_scatter_indices) and fixed_image_sigma_int is None
    if fixed_image_sigma_int is not None:
        q50 = float(fixed_image_sigma_int)
        scatter_floor = float(getattr(args, "image_plane_scatter_floor_arcsec", 0.0))
        summary["image_sigma_int_posterior"] = {
            "q16": q50,
            "median": q50,
            "q84": q50,
            "lower_arcsec": q50,
            "upper_arcsec": q50,
            "scatter_floor_arcsec": scatter_floor,
            "near_lower_bound": False,
            "floor_dominated": bool(q50 <= scatter_floor),
            "near_upper_bound": False,
            "fixed": True,
        }
        sigma_values = []
        for bin_item in getattr(state, "bin_data", []):
            if hasattr(bin_item, "sigma_per_image"):
                sigma_values.extend(np.asarray(bin_item.sigma_per_image, dtype=float).reshape(-1).tolist())
        sigma_array = np.asarray(sigma_values, dtype=float)
        finite_sigma = sigma_array[np.isfinite(sigma_array)]
        if finite_sigma.size:
            covariance_floor = float(getattr(args, "source_plane_covariance_floor", 0.0))
            sigma_eff2 = finite_sigma**2 + q50**2 + covariance_floor
            summary["image_sigma_eff_variance_arcsec2"] = {
                "min": float(np.min(sigma_eff2)),
                "median": float(np.median(sigma_eff2)),
                "max": float(np.max(sigma_eff2)),
            }
    elif image_scatter_indices:
        values = np.asarray(results.samples, dtype=float)[:, image_scatter_indices[0]]
        finite = values[np.isfinite(values)]
        if finite.size:
            spec = state.parameter_specs[image_scatter_indices[0]]
            upper_value = getattr(spec, "physical_upper", None)
            if upper_value is None:
                upper_value = getattr(spec, "upper", np.nan)
            upper = float(upper_value)
            q16, q50, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
            lower_value = getattr(spec, "physical_lower", None)
            lower = float(lower_value) if lower_value is not None else float("nan")
            scatter_floor = float(getattr(args, "image_plane_scatter_floor_arcsec", 0.0))
            near_lower = bool(np.isfinite(lower) and q16 <= max(1.1 * lower, lower + 1.0e-9))
            floor_dominated = bool(scatter_floor > 0.0 and q50 <= scatter_floor)
            summary["image_sigma_int_posterior"] = {
                "q16": float(q16),
                "median": float(q50),
                "q84": float(q84),
                "lower_arcsec": lower,
                "upper_arcsec": upper,
                "scatter_floor_arcsec": scatter_floor,
                "near_lower_bound": near_lower,
                "floor_dominated": floor_dominated,
                "near_upper_bound": bool(np.isfinite(upper) and q84 >= 0.9 * upper),
            }
            sigma_values = []
            for bin_item in getattr(state, "bin_data", []):
                if hasattr(bin_item, "sigma_per_image"):
                    sigma_values.extend(np.asarray(bin_item.sigma_per_image, dtype=float).reshape(-1).tolist())
            sigma_array = np.asarray(sigma_values, dtype=float)
            finite_sigma = sigma_array[np.isfinite(sigma_array)]
            if finite_sigma.size:
                covariance_floor = float(getattr(args, "source_plane_covariance_floor", 0.0))
                sigma_eff2 = finite_sigma**2 + float(q50) ** 2 + covariance_floor
                summary["image_sigma_eff_variance_arcsec2"] = {
                    "min": float(np.min(sigma_eff2)),
                    "median": float(np.median(sigma_eff2)),
                    "max": float(np.max(sigma_eff2)),
                }
            summary["image_sigma_int_posterior"]["fixed"] = False
    return summary


def _format_run_summary_text(summary: dict[str, Any]) -> str:
    source_range = "na"
    if summary.get("z_source_min") is not None and summary.get("z_source_max") is not None:
        source_range = f"{_metric_text(summary.get('z_source_min'))}-{_metric_text(summary.get('z_source_max'))}"
    headline_chi_square = summary.get("headline_chi_square")
    headline_dof = summary.get("headline_dof")
    headline_reduced_chi_square = summary.get("headline_reduced_chi_square")
    arc_aware_chi_square = summary.get("arc_aware_chi_square")
    arc_aware_dof = summary.get("arc_aware_dof")
    arc_aware_reduced_chi_square = summary.get("arc_aware_reduced_chi_square")
    arc_supported_count = summary.get("arc_aware_arc_supported_image_count")
    missing_count = summary.get("arc_aware_missing_image_count")
    support_radius = summary.get("arc_aware_noncritical_support_radius_arcsec")
    caveat = (
        "arc-aware caveat: arc support is censored by "
        f"arc_aware_noncritical_support_radius_arcsec={_metric_text(support_radius)}; "
        "still-missing images have no supporting arc."
    )
    lines = [
        "Cluster Solver Run Summary",
        f"run_name={_metric_text(summary.get('run_name'))}",
        "",
        "Lensing Information",
        *(
            _key_value_lines(
                [
                    ("fit mode", summary.get("fit_mode")),
                    ("likelihood mode", summary.get("sample_likelihood_mode")),
                    ("fit sampling engine", summary.get("fit_sampling_engine", summary.get("sampling_engine"))),
                    ("final validation sampling engine", summary.get("final_validation_sampling_engine")),
                    ("sampler", summary.get("sampler")),
                    ("runtime seconds", summary.get("runtime_sec")),
                    ("families", summary.get("n_families")),
                    ("images", summary.get("n_images")),
                    ("parameters", summary.get("n_parameters")),
                    ("scaling components", summary.get("n_scaling_galaxy_components")),
                    ("active scaling components", summary.get("active_scaling_components")),
                    ("lens redshift", summary.get("z_lens")),
                    ("source redshift range", source_range),
                    ("effective source planes", summary.get("distinct_effective_source_planes")),
                    ("quick diagnostics", summary.get("quick_diagnostics")),
                    ("image scatter floor arcsec", summary.get("image_plane_scatter_floor_arcsec")),
                    ("image scatter prior", summary.get("image_plane_scatter_prior")),
                    ("image sigma int sampled", summary.get("image_sigma_int_sampled")),
                    ("fixed image sigma int arcsec", summary.get("fixed_image_sigma_int_arcsec")),
                    ("likelihood max gain", summary.get("likelihood_stabilizer_max_gain")),
                    ("likelihood max residual arcsec", summary.get("likelihood_stabilizer_max_residual_arcsec")),
                    ("likelihood residual loss", summary.get("likelihood_stabilizer_residual_loss")),
                    ("likelihood Student-t nu", summary.get("likelihood_stabilizer_student_t_nu")),
                    ("fit active-subset log likelihood", summary.get("fit_active_subset_loglike")),
                    ("full-model validation log likelihood", summary.get("full_model_validation_loglike")),
                    ("best log likelihood", summary.get("best_loglike")),
                ]
            )
        ),
        "",
        "Quality Of Fit",
        "chi-square sigma: total image-plane sigma (image_sigma_eff_arcsec)",
        *(
            _key_value_lines(
                [
                    ("headline_chi_square", headline_chi_square),
                    ("headline dof", headline_dof),
                    ("headline_reduced_chi_square", headline_reduced_chi_square),
                    ("arc_aware_chi_square", arc_aware_chi_square),
                    ("arc-aware dof", arc_aware_dof),
                    ("arc_aware_reduced_chi_square", arc_aware_reduced_chi_square),
                    ("chi-square sigma basis", summary.get("chi_square_sigma_basis")),
                    ("chi-square median sigma arcsec", summary.get("chi_square_sigma_eff_median_arcsec")),
                    ("chi-square min sigma arcsec", summary.get("chi_square_sigma_eff_min_arcsec")),
                    ("chi-square max sigma arcsec", summary.get("chi_square_sigma_eff_max_arcsec")),
                    (
                        "headline red1 total sigma arcsec",
                        summary.get("headline_chi_square_red1_total_sigma_arcsec"),
                    ),
                    (
                        "headline red1 pos_sigma_arcsec",
                        summary.get("headline_chi_square_red1_pos_sigma_arcsec"),
                    ),
                    (
                        "arc-aware red1 total sigma arcsec",
                        summary.get("arc_aware_chi_square_red1_total_sigma_arcsec"),
                    ),
                    (
                        "arc-aware red1 pos_sigma_arcsec",
                        summary.get("arc_aware_chi_square_red1_pos_sigma_arcsec"),
                    ),
                    ("chi-square red1 calibration", summary.get("chi_square_red1_calibration_note")),
                    ("N_arc_supported", arc_supported_count),
                    ("N_missing", missing_count),
                    ("effective parameters", summary.get("n_effective_parameters")),
                    ("fit-quality reference", summary.get("fit_quality_reference_sample_kind")),
                    ("fit-quality sample index", summary.get("fit_quality_reference_sample_index")),
                    ("fit-quality source log likelihood", summary.get("fit_quality_reference_source_loglike")),
                    ("fit-quality log probability", summary.get("fit_quality_reference_log_prob")),
                ]
            )
        ),
        caveat,
    ]
    return "\n".join(lines).rstrip() + "\n"


def _format_sequential_run_summary_text(
    stage_summaries: list[dict[str, Any]],
    *,
    run_name: str,
    root_dir: str | Path,
) -> str:
    lines = [
        "Sequential Cluster Solver Run Summary",
        f"run_name={run_name}",
        f"root_dir={Path(root_dir)}",
        "",
        "Stage Quality Comparison",
    ]
    columns = [
        ("stage", "stage"),
        ("fit", "fit_method"),
        ("likelihood", "sample_likelihood_mode"),
        ("sampler", "sampler"),
        ("families", "n_families"),
        ("images", "n_images"),
        ("headline_chi2", "headline_chi_square"),
        ("headline_dof", "headline_dof"),
        ("headline_red", "headline_reduced_chi_square"),
        ("arc_chi2", "arc_aware_chi_square"),
        ("arc_dof", "arc_aware_dof"),
        ("arc_red", "arc_aware_reduced_chi_square"),
        ("N_arc", "arc_aware_arc_supported_image_count"),
        ("N_missing", "arc_aware_missing_image_count"),
        ("arc_RMS", "arc_aware_image_rms_arcsec"),
        ("ESS_min", "ess_min"),
        ("Rhat_max", "rhat_max"),
        ("runtime_s", "runtime_sec"),
    ]
    lines.extend(_table_text(stage_summaries, columns))
    return "\n".join(lines).rstrip() + "\n"


def _corner_dynamic_subset(
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    plot_name: str,
) -> tuple[np.ndarray, list[ParameterSpec]] | None:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] < 2 or sample_array.shape[1] == 0:
        _log(None, f"[plot:corner] skipped {plot_name}: need at least two samples and one parameter")
        return None
    spans = np.nanmax(sample_array, axis=0) - np.nanmin(sample_array, axis=0)
    dynamic_mask = np.isfinite(spans) & (spans > 0.0)
    if int(np.sum(dynamic_mask)) < 2:
        _log(None, f"[plot:corner] skipped {plot_name}: fewer than two parameters have dynamic range")
        return None
    if not np.all(dynamic_mask):
        dropped = [spec.name for spec, keep in zip(parameter_specs, dynamic_mask) if not keep]
        _log(None, f"[plot:corner] {plot_name}: dropped constant parameters={', '.join(dropped)}")
    subset_samples = sample_array[:, dynamic_mask]
    subset_specs = [spec for spec, keep in zip(parameter_specs, dynamic_mask) if keep]
    if subset_samples.shape[1] > subset_samples.shape[0]:
        spans_subset = spans[dynamic_mask]
        keep_count = max(2, int(subset_samples.shape[0]))
        keep_order = np.argsort(-spans_subset)[:keep_count]
        keep_order = np.sort(keep_order)
        dropped = [spec.name for idx, spec in enumerate(subset_specs) if idx not in set(keep_order.tolist())]
        _log(
            None,
            (
                f"[plot:corner] {plot_name}: limited plotted parameters to {keep_count} "
                f"because samples={subset_samples.shape[0]} < parameters={subset_samples.shape[1]}; "
                f"dropped={', '.join(dropped)}"
            ),
        )
        subset_samples = subset_samples[:, keep_order]
        subset_specs = [subset_specs[idx] for idx in keep_order.tolist()]
    return subset_samples, subset_specs


def _corner_without_source_positions(
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    plot_name: str = "corner.pdf",
) -> tuple[np.ndarray, list[ParameterSpec]]:
    sample_array = np.asarray(samples, dtype=float)
    n_columns = sample_array.shape[1] if sample_array.ndim == 2 else len(parameter_specs)
    keep_indices = [
        idx
        for idx, spec in enumerate(parameter_specs)
        if getattr(spec, "component_family", None) != "source_position" and idx < n_columns
    ]
    excluded = [
        getattr(spec, "name", str(spec))
        for idx, spec in enumerate(parameter_specs)
        if getattr(spec, "component_family", None) == "source_position" and idx < n_columns
    ]
    if excluded:
        _log(None, f"[plot:corner] {plot_name}: excluded source-position parameters={', '.join(excluded)}")
    if sample_array.ndim != 2:
        subset_specs = [
            spec
            for spec in parameter_specs
            if getattr(spec, "component_family", None) != "source_position"
        ]
        return sample_array, subset_specs
    return sample_array[:, keep_indices], [parameter_specs[idx] for idx in keep_indices]


def _plot_corner(
    plot_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    truth_values: dict[str, float] | None = None,
    best_fit_values: dict[str, float] | None = None,
    previous_stage_best_values: dict[str, float] | None = None,
    bayes_corner_overlay: BayesCornerOverlay | None = None,
    best_par_marker_values: dict[str, float] | None = None,
    *,
    output_name: str = "corner.pdf",
    plot_datapoints: bool = False,
) -> None:
    if corner is None or not parameter_specs:
        return
    corner_samples, corner_specs = _corner_without_source_positions(samples, parameter_specs, output_name)
    if not corner_specs:
        return
    finite_samples = _finite_sample_rows(corner_samples)
    if finite_samples.shape[0] == 0:
        return
    subset = _corner_dynamic_subset(finite_samples, corner_specs, output_name)
    if subset is None:
        return
    finite_samples, subset_specs = subset
    _log(
        None,
        f"[plot:corner] path={_plot_path(plot_dir, output_name)} ndim={len(subset_specs)} samples_shape={tuple(finite_samples.shape)}",
    )
    labels = [spec.name for spec in subset_specs]
    truths = _corner_values_for_specs(subset_specs, truth_values) if truth_values else None
    corner_kwargs = {**CORNER_PLOT_KWARGS, "plot_datapoints": bool(plot_datapoints)}
    fig = corner.corner(
        finite_samples,
        labels=labels,
        truths=truths,
        weights=_uniform_normalized_plot_weights(finite_samples),
        **corner_kwargs,
    )
    _overplot_bayes_corner_contours(fig, subset_specs, bayes_corner_overlay, output_name)
    _overplot_corner_previous_stage_best_fit(fig, subset_specs, previous_stage_best_values)
    _overplot_corner_best_fit(fig, subset_specs, best_fit_values)
    _overplot_corner_best_par_marker(fig, subset_specs, best_par_marker_values, output_name)
    fig.savefig(_plot_path(plot_dir, output_name), dpi=CORNER_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _best_fit_values_for_specs(
    parameter_specs: list[ParameterSpec],
    best_fit: np.ndarray,
) -> dict[str, float]:
    best_fit_array = np.asarray(best_fit, dtype=float).reshape(-1)
    return {
        spec.name: float(best_fit_array[idx])
        for idx, spec in enumerate(parameter_specs)
        if idx < best_fit_array.size
    }


def _corner_values_for_specs(
    parameter_specs: list[ParameterSpec],
    values_by_name: dict[str, float] | None,
) -> list[float]:
    if not values_by_name:
        return []
    values: list[float] = []
    for spec in parameter_specs:
        value = float("nan")
        for key in (getattr(spec, "name", None), getattr(spec, "sample_name", None)):
            if key is None or key not in values_by_name:
                continue
            try:
                candidate = float(values_by_name[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(candidate):
                value = candidate
                break
        values.append(value)
    return values


BayesCornerOverlay = dict[str, np.ndarray]


def _bayes_dat_headers(path: Path) -> list[str]:
    headers: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                break
            header = stripped[1:].strip()
            if header:
                headers.append(header)
    return headers


def _bayes_kpc_per_arcsec(state: BuildState) -> float:
    try:
        z_lens = float(getattr(state, "z_lens"))
        cosmo_config = dict(getattr(state, "cosmo_config", {}) or {})
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(z_lens) or z_lens <= 0.0 or not cosmo_config:
        return 1.0
    try:
        return float(_kpc_per_arcsec_from_config(z_lens, cosmo_config))
    except Exception as exc:  # pragma: no cover - defensive fallback for malformed old artifacts
        _log(None, f"[plot:corner] bayes.dat radius conversion unavailable; using arcsec values: {exc}")
        return 1.0


def _bayes_potfile_id(state: BuildState, index: int) -> str | None:
    potfiles = list(getattr(state, "potfiles", []) or [])
    if index < 0 or index >= len(potfiles):
        return None
    potfile = potfiles[index]
    return str(potfile.get("id", f"potfile{index}")) if isinstance(potfile, dict) else f"potfile{index}"


def _bayes_dat_column_key(
    header: str,
    state: BuildState,
    *,
    kpc_per_arcsec: float,
) -> tuple[str, float] | None:
    base_header = re.sub(r"\s+", " ", str(header).split("(", 1)[0].strip())
    if not base_header:
        return None
    ignored = {"nsample", "ln(lhood)", "ln likelihood", "ln_lhood", "chi2"}
    if base_header.lower() in ignored:
        return None

    object_match = re.match(r"^O(\d+)\s*:\s*(.+)$", base_header, flags=re.IGNORECASE)
    if object_match:
        potential_id = str(int(object_match.group(1)))
        raw_field = object_match.group(2).strip().lower()
        field_map = {
            "x": ("x_centre", 1.0),
            "y": ("y_centre", 1.0),
            "emass": ("ellipticite", 1.0),
            "ellipticity": ("ellipticite", 1.0),
            "ellipticite": ("ellipticite", 1.0),
            "theta": ("angle_pos", 1.0),
            "angle": ("angle_pos", 1.0),
            "angle_pos": ("angle_pos", 1.0),
            "rc": ("core_radius_kpc", kpc_per_arcsec),
            "core": ("core_radius_kpc", kpc_per_arcsec),
            "core_radius": ("core_radius_kpc", kpc_per_arcsec),
            "rcut": ("cut_radius_kpc", kpc_per_arcsec),
            "cut": ("cut_radius_kpc", kpc_per_arcsec),
            "cut_radius": ("cut_radius_kpc", kpc_per_arcsec),
            "sigma": ("v_disp", 1.0),
            "v_disp": ("v_disp", 1.0),
        }
        mapped = field_map.get(raw_field)
        if mapped is None:
            return None
        field_name, scale = mapped
        return f"{potential_id}.{field_name}", float(scale)

    potfile_match = re.match(r"^Pot(\d+)\s+(.+)$", base_header, flags=re.IGNORECASE)
    if potfile_match:
        potfile_id = _bayes_potfile_id(state, int(potfile_match.group(1)))
        if potfile_id is None:
            return None
        raw_field = potfile_match.group(2).strip().lower()
        field_map = {
            "sigma": ("sigma", 1.0),
            "rcut": ("cutkpc", kpc_per_arcsec),
            "cut": ("cutkpc", kpc_per_arcsec),
            "cut_radius": ("cutkpc", kpc_per_arcsec),
            "rc": ("corekpc", kpc_per_arcsec),
            "core": ("corekpc", kpc_per_arcsec),
            "core_radius": ("corekpc", kpc_per_arcsec),
            "vdslope": ("vdslope", 1.0),
            "slope": ("slope", 1.0),
        }
        mapped = field_map.get(raw_field)
        if mapped is None:
            return None
        field_name, scale = mapped
        return f"{potfile_id}.{field_name}", float(scale)

    return None


def _load_bayes_corner_overlay(path: str | Path | None, state: BuildState) -> BayesCornerOverlay | None:
    if path is None:
        return None
    bayes_path = Path(path)
    if not bayes_path.exists():
        _log(None, f"[plot:corner] bayes.dat overlay skipped; missing file: {bayes_path}")
        return None
    try:
        headers = _bayes_dat_headers(bayes_path)
        data = np.loadtxt(bayes_path, comments="#", dtype=float)
    except Exception as exc:
        _log(None, f"[plot:corner] bayes.dat overlay skipped; failed to read {bayes_path}: {exc}")
        return None
    if data.size == 0:
        _log(None, f"[plot:corner] bayes.dat overlay skipped; no samples in {bayes_path}")
        return None
    data_array = np.asarray(data, dtype=float)
    if data_array.ndim == 1:
        data_array = data_array.reshape(1, -1)
    if not headers:
        _log(None, f"[plot:corner] bayes.dat overlay skipped; no column headers in {bayes_path}")
        return None
    kpc_per_arcsec = _bayes_kpc_per_arcsec(state)
    overlay: BayesCornerOverlay = {}
    for column_idx, header in enumerate(headers[: data_array.shape[1]]):
        mapping = _bayes_dat_column_key(header, state, kpc_per_arcsec=kpc_per_arcsec)
        if mapping is None:
            continue
        key, scale = mapping
        if key in overlay:
            continue
        values = np.asarray(data_array[:, column_idx], dtype=float) * float(scale)
        if np.isfinite(values).any():
            overlay[key] = values
    if not overlay:
        _log(None, f"[plot:corner] bayes.dat overlay skipped; no recognized parameter columns in {bayes_path}")
        return None
    _log(None, f"[plot:corner] bayes.dat overlay loaded path={bayes_path} columns={len(overlay)} samples={data_array.shape[0]}")
    return overlay


def _bayes_overlay_column_for_spec(
    spec: ParameterSpec,
    overlay: BayesCornerOverlay,
) -> np.ndarray | None:
    keys = [
        str(getattr(spec, "name", "")),
        str(getattr(spec, "sample_name", "")),
        f"{getattr(spec, 'potential_id', '')}.{getattr(spec, 'field', '')}",
    ]
    for key in keys:
        if key and key in overlay:
            return np.asarray(overlay[key], dtype=float)
    return None


def _bayes_overlay_samples_for_specs(
    parameter_specs: list[ParameterSpec],
    overlay: BayesCornerOverlay | None,
    plot_name: str,
) -> np.ndarray | None:
    if not overlay or not parameter_specs:
        return None
    columns: list[np.ndarray] = []
    missing: list[str] = []
    expected_rows: int | None = None
    for spec in parameter_specs:
        values = _bayes_overlay_column_for_spec(spec, overlay)
        if values is None:
            missing.append(str(getattr(spec, "name", spec)))
            continue
        if expected_rows is None:
            expected_rows = int(values.shape[0])
        if values.shape[0] != expected_rows:
            missing.append(str(getattr(spec, "name", spec)))
            continue
        columns.append(values)
    if missing:
        _log(None, f"[plot:corner] {plot_name}: bayes.dat overlay skipped; unmatched parameters={', '.join(missing)}")
        return None
    if not columns:
        _log(None, f"[plot:corner] {plot_name}: bayes.dat overlay skipped; no matching columns")
        return None
    samples = _finite_sample_rows(np.column_stack(columns))
    if samples.shape[0] == 0:
        _log(None, f"[plot:corner] {plot_name}: bayes.dat overlay skipped; no finite matching samples")
        return None
    return samples


def _overplot_bayes_corner_contours(
    fig: Any,
    parameter_specs: list[ParameterSpec],
    overlay: BayesCornerOverlay | None,
    plot_name: str,
) -> None:
    if corner is None or overlay is None:
        return
    bayes_samples = _bayes_overlay_samples_for_specs(parameter_specs, overlay, plot_name)
    if bayes_samples is None:
        return
    kwargs = {
        **CORNER_PLOT_KWARGS,
        "fig": fig,
        "color": CORNER_BAYES_OVERLAY_COLOR,
        "fill_contours": False,
        "no_fill_contours": True,
        "plot_datapoints": False,
        "plot_density": False,
        "show_titles": False,
        "quantiles": [],
    }
    corner.corner(
        bayes_samples,
        labels=[spec.name for spec in parameter_specs],
        weights=_uniform_normalized_plot_weights(bayes_samples),
        **kwargs,
    )


def _normalized_best_par_potential_id(potential_id: Any) -> str:
    text = str(potential_id).strip()
    match = re.fullmatch(r"O(\d+)", text, flags=re.IGNORECASE)
    if match:
        return str(int(match.group(1)))
    return text


def _finite_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _finite_median(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _state_potfiles_with_catalogs(state: BuildState) -> list[dict[str, Any]]:
    potfiles = [dict(item) for item in list(getattr(state, "potfiles", []) or [])]
    if any(isinstance(potfile.get("catalog_df"), pd.DataFrame) for potfile in potfiles):
        return potfiles
    par_path = getattr(state, "par_path", None)
    if not par_path:
        return potfiles
    try:
        parsed, _potentials_df, _images_df, _potentials_with_priors = load_best_par(par_path)
    except Exception as exc:
        _log(None, f"[plot:corner] best.par potfile catalog fallback unavailable from {par_path}: {exc}")
        return potfiles
    return [dict(item) for item in list(parsed.get("potfiles", []) or [])]


def _best_par_large_component_values(potentials_df: pd.DataFrame) -> dict[str, float]:
    if potentials_df.empty or "id" not in potentials_df.columns:
        return {}
    fields = (
        "x_centre",
        "y_centre",
        "ellipticite",
        "angle_pos",
        "core_radius_kpc",
        "cut_radius_kpc",
        "v_disp",
        "gamma",
    )
    values: dict[str, float] = {}
    for row in potentials_df.to_dict(orient="records"):
        potential_id = _normalized_best_par_potential_id(row.get("id"))
        for field in fields:
            value = _finite_float_or_none(row.get(field))
            if value is not None:
                values[f"{potential_id}.{field}"] = value
    return values


def _best_par_potfile_values(
    potentials_df: pd.DataFrame,
    state: BuildState,
) -> dict[str, float]:
    if potentials_df.empty or "id" not in potentials_df.columns:
        return {}
    best_by_id = {
        str(row["id"]): row
        for row in potentials_df.to_dict(orient="records")
        if row.get("id") is not None
    }
    values: dict[str, float] = {}
    for potfile_order, potfile in enumerate(_state_potfiles_with_catalogs(state)):
        potfile_id = str(potfile.get("id", f"potfile{potfile_order}"))
        catalog_df = potfile.get("catalog_df")
        if not isinstance(catalog_df, pd.DataFrame) or catalog_df.empty:
            continue
        mag0 = _finite_float_or_none(potfile.get("mag0"))
        vdslope = _finite_float_or_none(potfile.get("vdslope_nominal", potfile.get("vdslope")))
        slope = _finite_float_or_none(potfile.get("slope_nominal", potfile.get("slope")))
        if mag0 is None:
            continue
        sigma_refs: list[float] = []
        cut_refs: list[float] = []
        core_refs: list[float] = []
        for catalog_row in catalog_df.to_dict(orient="records"):
            catalog_id = str(catalog_row.get("id"))
            best_row = best_by_id.get(catalog_id)
            if best_row is None:
                continue
            mag = _finite_float_or_none(catalog_row.get("catalog_mag"))
            if mag is None:
                continue
            luminosity_ratio = float(10.0 ** (-0.4 * (mag - mag0)))
            if not np.isfinite(luminosity_ratio) or luminosity_ratio <= 0.0:
                continue
            v_disp = _finite_float_or_none(best_row.get("v_disp"))
            if v_disp is not None and vdslope is not None and abs(vdslope) > 1.0e-12:
                sigma_refs.append(float(v_disp / (luminosity_ratio ** (1.0 / vdslope))))
            cut_radius_kpc = _finite_float_or_none(best_row.get("cut_radius_kpc"))
            if cut_radius_kpc is not None and slope is not None and abs(slope) > 1.0e-12:
                cut_refs.append(float(cut_radius_kpc / (luminosity_ratio ** (2.0 / slope))))
            core_radius_kpc = _finite_float_or_none(best_row.get("core_radius_kpc"))
            if core_radius_kpc is not None:
                core_refs.append(float(core_radius_kpc / np.sqrt(luminosity_ratio)))
        for field, field_values in (("sigma", sigma_refs), ("cutkpc", cut_refs), ("corekpc", core_refs)):
            median = _finite_median(field_values)
            if median is not None:
                values[f"{potfile_id}.{field}"] = median
        if vdslope is not None:
            values[f"{potfile_id}.vdslope"] = vdslope
        if slope is not None:
            values[f"{potfile_id}.slope"] = slope
    return values


def _load_best_par_marker_values(path: str | Path | None, state: BuildState) -> dict[str, float] | None:
    if path is None:
        return None
    best_par_path = Path(path)
    if not best_par_path.exists():
        _log(None, f"[plot:corner] best.par marker skipped; missing file: {best_par_path}")
        return None
    try:
        _parsed, potentials_df, _images_df, _potentials_with_priors = load_best_par(best_par_path)
    except Exception as exc:
        _log(None, f"[plot:corner] best.par marker skipped; failed to read {best_par_path}: {exc}")
        return None
    values = _best_par_large_component_values(potentials_df)
    values.update(_best_par_potfile_values(potentials_df, state))
    if not values:
        _log(None, f"[plot:corner] best.par marker skipped; no recognized parameter values in {best_par_path}")
        return None
    _log(None, f"[plot:corner] best.par marker loaded path={best_par_path} values={len(values)}")
    return values


def _overplot_corner_best_par_marker(
    fig: Any,
    parameter_specs: list[ParameterSpec],
    marker_values: dict[str, float] | None,
    plot_name: str,
) -> None:
    if corner is None or not marker_values:
        return
    xs = _corner_values_for_specs(parameter_specs, marker_values)
    if not xs or not any(np.isfinite(xs)):
        _log(None, f"[plot:corner] {plot_name}: best.par marker skipped; no matching finite parameters")
        return
    point_xs = [[float(value) if np.isfinite(value) else np.nan for value in xs]]
    corner.overplot_points(
        fig,
        point_xs,
        marker="x",
        color=CORNER_BEST_PAR_COLOR,
        markersize=5,
        markeredgewidth=1.2,
    )


def _overplot_corner_best_fit(
    fig: Any,
    parameter_specs: list[ParameterSpec],
    best_fit_values: dict[str, float] | None,
) -> None:
    if corner is None or not best_fit_values:
        return
    xs = _corner_values_for_specs(parameter_specs, best_fit_values)
    if not xs or not any(np.isfinite(xs)):
        return
    point_xs = [[float(value) if np.isfinite(value) else np.nan for value in xs]]
    corner.overplot_points(
        fig,
        point_xs,
        marker="x",
        color=CORNER_BEST_FIT_COLOR,
        markersize=5,
        markeredgewidth=1.2,
    )


def _overplot_corner_previous_stage_best_fit(
    fig: Any,
    parameter_specs: list[ParameterSpec],
    previous_stage_best_values: dict[str, float] | None,
) -> None:
    if corner is None or not previous_stage_best_values:
        return
    xs = _corner_values_for_specs(parameter_specs, previous_stage_best_values)
    if not xs or not any(np.isfinite(xs)):
        return
    point_xs = [[float(value) if np.isfinite(value) else np.nan for value in xs]]
    corner.overplot_points(
        fig,
        point_xs,
        marker="x",
        color=CORNER_PREVIOUS_STAGE_COLOR,
        markersize=5,
        markeredgewidth=1.2,
    )


def _scaling_parameter_subset(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
) -> tuple[list[ParameterSpec], np.ndarray, np.ndarray]:
    scaling_indices = [idx for idx, spec in enumerate(parameter_specs) if spec.component_family == "scaling"]
    if not scaling_indices:
        return [], np.empty((samples.shape[0], 0), dtype=float), np.empty((0,), dtype=float)
    subset_specs = [parameter_specs[idx] for idx in scaling_indices]
    subset_samples = np.asarray(samples[:, scaling_indices], dtype=float)
    subset_best_fit = np.asarray(best_fit[scaling_indices], dtype=float)
    return subset_specs, subset_samples, subset_best_fit


def _cosmology_parameter_subset(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
) -> tuple[list[ParameterSpec], np.ndarray, np.ndarray]:
    cosmology_indices = [
        idx for idx, spec in enumerate(parameter_specs) if getattr(spec, "component_family", None) == "cosmology"
    ]
    if not cosmology_indices:
        return [], np.empty((samples.shape[0], 0), dtype=float), np.empty((0,), dtype=float)
    subset_specs = [parameter_specs[idx] for idx in cosmology_indices]
    subset_samples = np.asarray(samples[:, cosmology_indices], dtype=float)
    subset_best_fit = np.asarray(best_fit[cosmology_indices], dtype=float)
    return subset_specs, subset_samples, subset_best_fit


def _scaling_grouped_subset(
    parameter_specs: list[ParameterSpec],
    grouped_samples: np.ndarray | None,
) -> tuple[list[ParameterSpec], np.ndarray | None]:
    scaling_indices = [idx for idx, spec in enumerate(parameter_specs) if spec.component_family == "scaling"]
    if grouped_samples is None or not scaling_indices:
        return [], None
    grouped_array = np.asarray(grouped_samples, dtype=float)
    if grouped_array.ndim != 3 or grouped_array.size == 0:
        return [], None
    subset_specs = [parameter_specs[idx] for idx in scaling_indices]
    subset_samples = np.asarray(grouped_array[:, :, scaling_indices], dtype=float)
    return subset_specs, subset_samples


def _plot_potfile_corner(
    plot_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    truth_values: dict[str, float] | None = None,
    best_fit_values: dict[str, float] | None = None,
    previous_stage_best_values: dict[str, float] | None = None,
    bayes_corner_overlay: BayesCornerOverlay | None = None,
    best_par_marker_values: dict[str, float] | None = None,
) -> None:
    if corner is None or samples.size == 0 or not parameter_specs:
        return
    finite_samples = _finite_sample_rows(samples)
    if finite_samples.shape[0] == 0:
        return
    subset = _corner_dynamic_subset(finite_samples, parameter_specs, "potfile_corner.pdf")
    if subset is None:
        return
    finite_samples, subset_specs = subset
    _log(
        None,
        f"[plot:corner] path={_plot_path(plot_dir, 'potfile_corner.pdf')} ndim={len(subset_specs)} samples_shape={tuple(finite_samples.shape)}",
    )
    labels = [spec.name for spec in subset_specs]
    truths = _corner_values_for_specs(subset_specs, truth_values) if truth_values else None
    fig = corner.corner(
        finite_samples,
        labels=labels,
        truths=truths,
        weights=_uniform_normalized_plot_weights(finite_samples),
        **CORNER_PLOT_KWARGS,
    )
    _overplot_bayes_corner_contours(fig, subset_specs, bayes_corner_overlay, "potfile_corner.pdf")
    _overplot_corner_previous_stage_best_fit(fig, subset_specs, previous_stage_best_values)
    _overplot_corner_best_fit(fig, subset_specs, best_fit_values)
    _overplot_corner_best_par_marker(fig, subset_specs, best_par_marker_values, "potfile_corner.pdf")
    fig.savefig(_plot_path(plot_dir, "potfile_corner.pdf"), dpi=CORNER_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_cosmology_corner(
    plot_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    truth_values: dict[str, float] | None = None,
    best_fit_values: dict[str, float] | None = None,
    previous_stage_best_values: dict[str, float] | None = None,
    bayes_corner_overlay: BayesCornerOverlay | None = None,
    best_par_marker_values: dict[str, float] | None = None,
    *,
    plot_datapoints: bool = False,
) -> None:
    if corner is None or samples.size == 0 or not parameter_specs:
        return
    finite_samples = _finite_sample_rows(samples)
    if finite_samples.shape[0] == 0:
        return
    subset = _corner_dynamic_subset(finite_samples, parameter_specs, "cosmology_corner.pdf")
    if subset is None:
        return
    finite_samples, subset_specs = subset
    _log(
        None,
        f"[plot:corner] path={_plot_path(plot_dir, 'cosmology_corner.pdf')} ndim={len(subset_specs)} samples_shape={tuple(finite_samples.shape)}",
    )
    labels = [spec.name for spec in subset_specs]
    truths = _corner_values_for_specs(subset_specs, truth_values) if truth_values else None
    corner_kwargs = {**CORNER_PLOT_KWARGS, "plot_datapoints": bool(plot_datapoints)}
    fig = corner.corner(
        finite_samples,
        labels=labels,
        truths=truths,
        weights=_uniform_normalized_plot_weights(finite_samples),
        **corner_kwargs,
    )
    _overplot_bayes_corner_contours(fig, subset_specs, bayes_corner_overlay, "cosmology_corner.pdf")
    _overplot_corner_previous_stage_best_fit(fig, subset_specs, previous_stage_best_values)
    _overplot_corner_best_fit(fig, subset_specs, best_fit_values)
    _overplot_corner_best_par_marker(fig, subset_specs, best_par_marker_values, "cosmology_corner.pdf")
    fig.savefig(_plot_path(plot_dir, "cosmology_corner.pdf"), dpi=CORNER_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_trace(
    plot_dir: Path,
    grouped_samples: np.ndarray | None,
    parameter_specs: list[ParameterSpec],
    *,
    output_name: str = "trace_plot.png",
) -> None:
    if grouped_samples is None or not parameter_specs:
        return
    grouped_array = np.asarray(grouped_samples, dtype=float)
    if grouped_array.ndim != 3 or grouped_array.size == 0:
        return
    n_chains, n_draws, n_params = grouped_array.shape
    if n_chains == 0 or n_draws == 0 or n_params == 0:
        return
    finite_mask = np.isfinite(grouped_array).all(axis=(1, 2))
    grouped_array = grouped_array[finite_mask]
    if grouped_array.size == 0:
        return
    nrows = len(parameter_specs)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, max(4, 2.2 * nrows)), sharex=True)
    if nrows == 1:
        axes = [axes]
    draw_index = np.arange(grouped_array.shape[1], dtype=int)
    cmap = plt.get_cmap("tab10", grouped_array.shape[0])
    for param_index, (ax, spec) in enumerate(zip(axes, parameter_specs)):
        for chain_index in range(grouped_array.shape[0]):
            ax.plot(
                draw_index,
                grouped_array[chain_index, :, param_index],
                linewidth=1.0,
                alpha=0.85,
                color=cmap(chain_index),
                label=f"chain {chain_index + 1}" if param_index == 0 else None,
            )
        ax.set_ylabel(spec.name)
    axes[-1].set_xlabel("posterior draw")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, output_name), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _ns_series(ns_diagnostics: dict[str, np.ndarray] | None, key: str) -> np.ndarray | None:
    if not ns_diagnostics or key not in ns_diagnostics:
        return None
    array = np.asarray(ns_diagnostics[key], dtype=float).reshape(-1)
    return array if array.size else None


def _ns_aligned_core(ns_diagnostics: dict[str, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    log_l = _ns_series(ns_diagnostics, "log_L_samples")
    log_dp = _ns_series(ns_diagnostics, "log_dp_mean")
    log_x = _ns_series(ns_diagnostics, "log_X_mean")
    if log_l is None or log_dp is None or log_x is None:
        return None
    n = min(log_l.size, log_dp.size, log_x.size)
    if n == 0:
        return None
    log_l = log_l[:n]
    log_dp = log_dp[:n]
    log_x = log_x[:n]
    finite = np.isfinite(log_l) & np.isfinite(log_dp) & np.isfinite(log_x)
    if not np.any(finite):
        return None
    return -log_x[finite], log_l[finite], log_dp[finite], finite


def _ns_normalized_weights(log_dp: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(log_dp)
    if not np.any(finite):
        return None
    weights = np.zeros_like(log_dp, dtype=float)
    max_log = float(np.max(log_dp[finite]))
    weights[finite] = np.exp(log_dp[finite] - max_log)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return weights / total


def _plot_ns_diagnostics(plot_dir: Path, ns_diagnostics: dict[str, np.ndarray] | None) -> None:
    aligned = _ns_aligned_core(ns_diagnostics)
    if aligned is None:
        return
    neg_log_x, log_l, log_dp, finite_mask = aligned
    weights = _ns_normalized_weights(log_dp)
    if weights is None:
        return
    n = neg_log_x.size
    live_points = _ns_series(ns_diagnostics, "num_live_points_per_sample")
    evals = _ns_series(ns_diagnostics, "num_likelihood_evaluations_per_sample")
    if live_points is not None:
        live_points = live_points[:finite_mask.size][finite_mask]
    if evals is not None:
        evals = evals[:finite_mask.size][finite_mask]
    log_efficiency = _ns_series(ns_diagnostics, "log_efficiency")
    mean_eff = float(np.exp(log_efficiency[0])) if log_efficiency is not None and np.isfinite(log_efficiency[0]) else None
    rel_l = np.exp(log_l - np.nanmax(log_l))
    cumulative_weight = np.cumsum(weights)
    log_x = -neg_log_x
    log_xl = log_x + log_l
    rel_xl = np.exp(log_xl - np.nanmax(log_xl[np.isfinite(log_xl)]))

    fig, axes = plt.subplots(6, 1, figsize=(9, 15), sharex=True)
    if live_points is not None and live_points.size == n:
        axes[0].plot(neg_log_x, live_points, color="black", linewidth=1.0)
    axes[0].set_ylabel(r"$n_{\rm live}$")
    axes[0].set_title("Nested Sampling Diagnostics")

    axes[1].plot(neg_log_x, rel_l, color="black", linewidth=1.0)
    axes[1].set_ylabel(r"$L / L_{\rm max}$")
    axes[1].set_yscale("log")

    axes[2].plot(neg_log_x, weights, color="black", linewidth=1.0)
    axes[2].set_ylabel("posterior mass")
    axes[2].set_yscale("log")

    axes[3].plot(neg_log_x, cumulative_weight, color="black", linewidth=1.0)
    axes[3].set_ylabel("cum. mass")
    axes[3].set_ylim(-0.02, 1.02)

    if evals is not None and evals.size == n:
        efficiency = np.where(evals > 0, 1.0 / evals, np.nan)
        axes[4].scatter(neg_log_x, efficiency, s=5, color="black", alpha=0.75)
    if mean_eff is not None:
        axes[4].axhline(mean_eff, color="tab:red", linestyle="--", linewidth=1.0, label=f"mean={mean_eff:.3g}")
        axes[4].legend(loc="best", fontsize=8)
    axes[4].set_ylabel("efficiency")

    axes[5].plot(neg_log_x, rel_xl, color="black", linewidth=1.0)
    axes[5].set_ylabel(r"$X L$ rel.")
    axes[5].set_xlabel(r"$-\log X$")

    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "ns_diagnostics.pdf"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _ns_trace_parameter_subset(
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    *,
    max_params: int = 8,
) -> tuple[np.ndarray, list[ParameterSpec]] | None:
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0 or not parameter_specs:
        return None
    n_params = min(samples.shape[1], len(parameter_specs))
    finite = np.isfinite(samples[:, :n_params])
    variances = np.asarray(
        [
            np.nanvar(np.where(finite[:, idx], samples[:, idx], np.nan))
            for idx in range(n_params)
        ],
        dtype=float,
    )
    candidate_indices = [idx for idx in np.argsort(-variances).tolist() if np.isfinite(variances[idx]) and variances[idx] > 0.0]
    if not candidate_indices:
        candidate_indices = list(range(n_params))
    selected = candidate_indices[:max_params]
    return np.asarray(samples[:, selected], dtype=float), [parameter_specs[idx] for idx in selected]


def _plot_ns_trace(plot_dir: Path, ns_diagnostics: dict[str, np.ndarray] | None, parameter_specs: list[ParameterSpec]) -> None:
    aligned = _ns_aligned_core(ns_diagnostics)
    if aligned is None or not ns_diagnostics or "samples" not in ns_diagnostics:
        return
    neg_log_x, _log_l, log_dp, finite_mask = aligned
    samples = np.asarray(ns_diagnostics["samples"], dtype=float)
    if samples.ndim != 2 or samples.shape[0] < finite_mask.size:
        return
    samples = samples[:finite_mask.size][finite_mask]
    subset = _ns_trace_parameter_subset(samples, parameter_specs)
    if subset is None:
        return
    subset_samples, subset_specs = subset
    weights = _ns_normalized_weights(log_dp)
    color_values = np.log10(np.maximum(weights, np.nextafter(0.0, 1.0))) if weights is not None else None
    nrows = len(subset_specs)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, max(4, 2.2 * nrows)), sharex=True)
    if nrows == 1:
        axes = [axes]
    mappable = None
    for idx, (ax, spec) in enumerate(zip(axes, subset_specs)):
        values = subset_samples[:, idx]
        finite_values = np.isfinite(values)
        if color_values is None:
            ax.plot(neg_log_x[finite_values], values[finite_values], color="tab:blue", linewidth=0.7, alpha=0.75)
        else:
            mappable = ax.scatter(
                neg_log_x[finite_values],
                values[finite_values],
                c=color_values[finite_values],
                cmap="viridis",
                s=8,
                alpha=0.85,
                linewidths=0.0,
                rasterized=True,
            )
        ax.set_ylabel(spec.name)
    axes[-1].set_xlabel(r"$-\log X$")
    if mappable is not None:
        fig.colorbar(mappable, ax=axes, label=r"$\log_{10}$ posterior weight", shrink=0.9)
        fig.subplots_adjust(right=0.86, hspace=0.25)
    else:
        fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "ns_trace_plot.pdf"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_ns_weight_diagnostics(plot_dir: Path, ns_diagnostics: dict[str, np.ndarray] | None) -> None:
    log_dp = _ns_series(ns_diagnostics, "log_dp_mean")
    if log_dp is None:
        return
    weights = _ns_normalized_weights(log_dp)
    if weights is None:
        return
    positive = weights[np.isfinite(weights) & (weights > 0.0)]
    if positive.size == 0:
        return
    sorted_weights = np.sort(positive)[::-1]
    cumulative = np.cumsum(sorted_weights)
    ess = 1.0 / float(np.sum(np.square(positive)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(np.log10(positive), bins=min(60, max(12, int(np.sqrt(positive.size)))), color="tab:blue", alpha=0.8)
    axes[0].set_xlabel(r"$\log_{10}$ normalized weight")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Weight distribution; ESS={ess:.3g}")
    axes[1].plot(np.arange(1, sorted_weights.size + 1), cumulative, color="black", linewidth=1.2)
    axes[1].set_xlabel("samples sorted by descending weight")
    axes[1].set_ylabel("cumulative weight")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xscale("log")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "ns_weight_diagnostics.pdf"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_scaling_rank_bars(plot_dir: Path, scaling_rank_df: pd.DataFrame) -> None:
    if scaling_rank_df.empty:
        return
    potfile_ids = scaling_rank_df["potfile_id"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(potfile_ids), 1, figsize=(12, max(4, 3.0 * len(potfile_ids))), sharex=False)
    if len(potfile_ids) == 1:
        axes = [axes]
    for ax, potfile_id in zip(axes, potfile_ids):
        pot_df = scaling_rank_df[scaling_rank_df["potfile_id"] == potfile_id].copy().sort_values("rank")
        requested = int(pot_df["requested_active_count"].iloc[0]) if not pot_df.empty else 0
        top_n = min(len(pot_df), max(12, min(40, max(3 * max(requested, 1), requested + 8))))
        top_df = pot_df.head(top_n)
        top_importance, _ = _logsafe_importance_values(top_df["importance"])
        colors = ["tab:orange" if active else "tab:gray" for active in top_df["selected_active"].tolist()]
        ax.bar(top_df["rank"].astype(int), top_importance, color=colors)
        if requested > 0:
            ax.axvline(requested + 0.5, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"{potfile_id}: top scaling-galaxy ranks")
        ax.set_xlabel("rank")
        ax.set_ylabel("importance")
        _apply_log_importance_axis(ax, top_df["importance"])
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "scaling_rank_bars.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _logsafe_importance_values(values: pd.Series | np.ndarray | list[float]) -> tuple[np.ndarray, float]:
    importance = np.asarray(values, dtype=float)
    positive = importance[np.isfinite(importance) & (importance > 0.0)]
    floor = float(max(np.min(positive) * 0.5, np.nextafter(0.0, 1.0))) if positive.size else 1.0
    clipped = np.where(np.isfinite(importance) & (importance > 0.0), importance, floor)
    return clipped, floor


def _apply_log_importance_axis(ax: plt.Axes, values: pd.Series | np.ndarray | list[float]) -> None:
    _, floor = _logsafe_importance_values(values)
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor * 0.8)


def _plot_scaling_rank_scatter(plot_dir: Path, scaling_rank_df: pd.DataFrame) -> None:
    if scaling_rank_df.empty:
        return
    potfile_ids = scaling_rank_df["potfile_id"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(potfile_ids), 1, figsize=(10, max(4, 3.4 * len(potfile_ids))), sharex=False)
    if len(potfile_ids) == 1:
        axes = [axes]
    for ax, potfile_id in zip(axes, potfile_ids):
        pot_df = scaling_rank_df[scaling_rank_df["potfile_id"] == potfile_id].copy()
        active_df = pot_df[pot_df["selected_active"]]
        inactive_df = pot_df[~pot_df["selected_active"]]
        inactive_importance, _ = _logsafe_importance_values(inactive_df["importance"])
        active_importance, _ = _logsafe_importance_values(active_df["importance"])
        ax.scatter(
            inactive_df["min_distance_arcsec"],
            inactive_importance,
            color="tab:gray",
            alpha=0.55,
            s=20,
            label="inactive",
        )
        ax.scatter(
            active_df["min_distance_arcsec"],
            active_importance,
            color="tab:orange",
            alpha=0.9,
            s=28,
            label="active",
        )
        top_df = pot_df.sort_values("rank").head(min(5, len(pot_df)))
        top_importance, _ = _logsafe_importance_values(top_df["importance"])
        for row, y_plot in zip(top_df.itertuples(index=False), top_importance.tolist()):
            ax.annotate(str(row.catalog_id), (row.min_distance_arcsec, y_plot), fontsize=7, alpha=0.8)
        ax.set_title(f"{potfile_id}: importance vs distance")
        ax.set_xlabel("min distance to observed images [arcsec]")
        ax.set_ylabel("importance")
        ax.legend(loc="best", fontsize=8)
        _apply_log_importance_axis(ax, pot_df["importance"])
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "scaling_rank_scatter.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_run_diagnostics(plot_dir: Path, results: PosteriorResults) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    axes[0].plot(results.accept_prob.ravel(), color="tab:green")
    axes[0].set_xlabel("Posterior draw")
    axes[0].set_ylabel("Accept prob")
    axes[0].set_title("NUTS Acceptance Probability")
    axes[1].axis("off")
    init_diag = results.init_diagnostics or {}
    summary_lines = [
        f"Init requested: {init_diag.get('strategy_requested', 'unknown')}",
        f"Init used: {init_diag.get('strategy_used', 'unknown')}",
        (
            "Seeds: "
            f"distinct_chains={int(init_diag.get('distinct_chain_seeds', 0))}"
        ),
        (
            "Chain quality: "
            f"retained={int(init_diag.get('retained_finite_chains', results.num_chains))}/"
            f"{int(init_diag.get('requested_chains', results.num_chains))} "
            f"dropped={int(init_diag.get('dropped_nonfinite_chains', 0))}"
        ),
    ]
    if init_diag.get("svi_used", False):
        summary_lines.append(
            "SVI: "
            f"steps={int(init_diag.get('svi_steps', 0))} "
            f"lr={float(init_diag.get('svi_learning_rate', 0.0)):.3g} "
            f"final_loss={float(init_diag.get('svi_final_elbo_loss', float('nan'))):.4g}"
        )
    if "invalid_state_rejection_count" in init_diag:
        summary_lines.append(
            "Invalid states: "
            f"rejected={int(init_diag.get('invalid_state_rejection_count', 0))}"
        )
    labels = list(init_diag.get("chain_seed_labels", []))
    if labels:
        summary_lines.append("Chain sources: " + ", ".join(str(label) for label in labels))
    axes[1].text(0.01, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=9, family="monospace")
    axes[1].set_title("Sampler Initialization")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "run_diagnostics.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_weights_logl(plot_dir: Path, results: PosteriorResults) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(results.log_prob, color="tab:red")
    axes[0].set_ylabel("log posterior")
    axes[0].set_title("Posterior Log Probability")
    axes[1].plot(results.num_steps.ravel(), color="tab:blue")
    axes[1].set_ylabel("NUTS steps")
    axes[1].set_xlabel("Posterior draw")
    axes[1].set_title("NUTS Integrator Steps")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "weights_logl.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_unavailable_axis(ax: Any, label: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, f"{label} unavailable", ha="center", va="center", fontsize=9)


def _plot_grouped_lines(
    ax: Any,
    values: np.ndarray | None,
    *,
    cmap: Any,
    ylabel: str,
    title: str,
    draw_index: np.ndarray,
) -> None:
    if values is None:
        _plot_unavailable_axis(ax, title)
        return
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        _plot_unavailable_axis(ax, title)
        return
    for chain_index in range(array.shape[0]):
        chain_values = array[chain_index]
        finite = np.isfinite(chain_values)
        if finite.any():
            ax.plot(
                draw_index[finite],
                chain_values[finite],
                linewidth=1.0,
                alpha=0.85,
                color=cmap(chain_index),
                label=f"chain {chain_index + 1}",
            )
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _plot_chain_health(
    plot_dir: Path,
    results: PosteriorResults,
    parameter_specs: list[ParameterSpec],
    *,
    max_tree_depth: int | None = None,
) -> None:
    grouped = results.grouped_samples
    if grouped is None:
        return
    grouped_array = np.asarray(grouped, dtype=float)
    if grouped_array.ndim != 3 or grouped_array.shape[0] == 0 or grouped_array.shape[1] == 0:
        return
    n_chains, n_draws, n_params = grouped_array.shape
    grouped_log_prob = (
        np.asarray(results.grouped_log_prob, dtype=float)
        if results.grouped_log_prob is not None
        else _grouped_chain_array(results.log_prob, n_chains, n_draws, dtype=float)
    )
    if grouped_log_prob is not None and grouped_log_prob.shape != (n_chains, n_draws):
        grouped_log_prob = None
    grouped_accept = _grouped_chain_array(results.accept_prob, n_chains, n_draws, dtype=float)
    grouped_steps = _grouped_chain_array(results.num_steps, n_chains, n_draws, dtype=float)
    image_sigma_idx = _image_sigma_int_index(parameter_specs, n_params)
    image_sigma = grouped_array[:, :, image_sigma_idx] if image_sigma_idx is not None else None
    draw_index = np.arange(n_draws, dtype=int)
    cmap = plt.get_cmap("tab10", n_chains)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    _plot_grouped_lines(
        axes[0],
        grouped_log_prob,
        cmap=cmap,
        ylabel="log posterior",
        title="Chain Log Posterior",
        draw_index=draw_index,
    )
    _plot_grouped_lines(
        axes[1],
        grouped_accept,
        cmap=cmap,
        ylabel="accept prob",
        title="Chain Acceptance Probability",
        draw_index=draw_index,
    )
    _plot_grouped_lines(
        axes[2],
        grouped_steps,
        cmap=cmap,
        ylabel="NUTS steps",
        title="Chain NUTS Integrator Steps",
        draw_index=draw_index,
    )
    if max_tree_depth is not None and grouped_steps is not None:
        axes[2].axhline(2**int(max_tree_depth) - 1, color="black", linestyle="--", linewidth=0.8, alpha=0.75)
    _plot_grouped_lines(
        axes[3],
        image_sigma,
        cmap=cmap,
        ylabel="arcsec",
        title="image.sigma_int by Chain",
        draw_index=draw_index,
    )
    axes[-1].set_xlabel("posterior draw")
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes[1:]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "chain_health.pdf"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_chain_ranked_trace(
    plot_dir: Path,
    grouped_samples: np.ndarray | None,
    parameter_specs: list[ParameterSpec],
    parameter_diagnostics: pd.DataFrame | None = None,
    *,
    max_params: int = 8,
) -> None:
    subset = _ranked_chain_trace_subset(
        grouped_samples,
        parameter_specs,
        parameter_diagnostics,
        max_params=max_params,
    )
    if subset is None:
        return
    grouped_array, subset_specs = subset
    if grouped_array.ndim != 3 or grouped_array.shape[0] == 0 or grouped_array.shape[1] == 0 or grouped_array.shape[2] == 0:
        return
    n_chains, n_draws, _n_params = grouped_array.shape
    draw_index = np.arange(n_draws, dtype=int)
    cmap = plt.get_cmap("tab10", n_chains)
    nrows = len(subset_specs)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, max(4, 2.0 * nrows)), sharex=True)
    if nrows == 1:
        axes = [axes]
    for param_index, (ax, spec) in enumerate(zip(axes, subset_specs)):
        for chain_index in range(n_chains):
            values = grouped_array[chain_index, :, param_index]
            finite = np.isfinite(values)
            if finite.any():
                ax.plot(
                    draw_index[finite],
                    values[finite],
                    linewidth=1.0,
                    alpha=0.85,
                    color=cmap(chain_index),
                    label=f"chain {chain_index + 1}" if param_index == 0 else None,
                )
        ax.set_ylabel(spec.name)
    axes[0].set_title("Worst Chain-Mixing Parameter Traces")
    axes[-1].set_xlabel("posterior draw")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "chain_ranked_trace.pdf"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _finite_1d_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return None
    finite = np.isfinite(array)
    if not np.any(finite):
        return None
    return array[finite]


def _has_smc_plot_data(results: PosteriorResults) -> bool:
    return (
        str(getattr(results, "sampler", "")) == "blackjax_smc"
        or results.temperature_schedule is not None
        or results.ess_history is not None
        or results.move_acceptance_history is not None
        or results.sample_weights is not None
    )


def _plot_smc_diagnostics(plot_dir: Path, results: PosteriorResults) -> None:
    temperature = _finite_1d_array(results.temperature_schedule)
    ess = _finite_1d_array(results.ess_history)
    acceptance = _finite_1d_array(results.move_acceptance_history)
    if temperature is None and ess is None and acceptance is None:
        return

    init_diag = results.init_diagnostics or {}
    n_particles = int(init_diag.get("smc_particles", 0) or 0)
    if n_particles <= 0 and results.sample_weights is not None:
        n_particles = int(np.asarray(results.sample_weights).size)
    target_ess_frac = init_diag.get("smc_target_ess_frac")
    try:
        target_ess_frac_value = float(target_ess_frac)
    except (TypeError, ValueError):
        target_ess_frac_value = float("nan")
    mean_acceptance = init_diag.get("smc_mean_move_acceptance")
    try:
        mean_acceptance_value = float(mean_acceptance)
    except (TypeError, ValueError):
        mean_acceptance_value = float("nan")

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=False)
    if temperature is not None:
        steps = np.arange(temperature.size, dtype=int)
        axes[0].plot(steps, temperature, marker="o", markersize=3, color="tab:blue", linewidth=1.1)
        axes[0].set_ylabel("temperature")
        axes[0].set_xlabel("SMC step")
        axes[0].set_ylim(min(-0.02, float(np.nanmin(temperature)) - 0.02), max(1.02, float(np.nanmax(temperature)) + 0.02))
        axes[0].set_title("SMC Adaptive Tempering")
        if temperature.size > 1:
            axes[1].plot(
                np.arange(1, temperature.size, dtype=int),
                np.diff(temperature),
                marker="o",
                markersize=3,
                color="tab:purple",
                linewidth=1.1,
            )
            axes[1].set_ylabel("delta temperature")
            axes[1].set_xlabel("SMC step")
        else:
            axes[1].axis("off")
    else:
        axes[0].axis("off")
        axes[1].axis("off")

    if ess is not None:
        ess_steps = np.arange(ess.size, dtype=int)
        if n_particles > 0:
            ess_values = ess / float(n_particles)
            axes[2].set_ylabel("ESS fraction")
            if np.isfinite(target_ess_frac_value) and target_ess_frac_value > 0.0:
                axes[2].axhline(
                    target_ess_frac_value,
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"target={target_ess_frac_value:.3g}",
                )
                axes[2].legend(loc="best", fontsize=8)
            axes[2].set_ylim(0.0, max(1.02, float(np.nanmax(ess_values)) * 1.05))
        else:
            ess_values = ess
            axes[2].set_ylabel("ESS")
        axes[2].plot(ess_steps, ess_values, marker="o", markersize=3, color="tab:green", linewidth=1.1)
        axes[2].set_xlabel("SMC step")
    else:
        axes[2].axis("off")

    if acceptance is not None:
        acceptance_steps = np.arange(1, acceptance.size + 1, dtype=int)
        axes[3].plot(acceptance_steps, acceptance, marker="o", markersize=3, color="tab:orange", linewidth=1.1)
        if np.isfinite(mean_acceptance_value):
            axes[3].axhline(
                mean_acceptance_value,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label=f"mean={mean_acceptance_value:.3g}",
            )
            axes[3].legend(loc="best", fontsize=8)
        axes[3].set_ylim(-0.02, max(1.02, float(np.nanmax(acceptance)) * 1.05))
        axes[3].set_ylabel("move acceptance")
        axes[3].set_xlabel("SMC mutation step")
    else:
        axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "smc_diagnostics.pdf"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_smc_weight_diagnostics(plot_dir: Path, results: PosteriorResults) -> None:
    n_samples = int(np.asarray(results.samples).shape[0]) if np.asarray(results.samples).ndim >= 1 else 0
    weights = _valid_normalized_sample_weights(results.sample_weights, n_samples)
    if weights is None:
        return
    positive = weights[np.isfinite(weights) & (weights > 0.0)]
    if positive.size == 0:
        return
    log_prob = np.asarray(results.log_prob, dtype=float).reshape(-1)
    log_prob = log_prob if log_prob.size == n_samples else None

    sorted_weights = np.sort(positive)[::-1]
    cumulative = np.cumsum(sorted_weights)
    ess = 1.0 / float(np.sum(np.square(weights)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes[0, 0].hist(np.log10(positive), bins=min(60, max(12, int(np.sqrt(positive.size)))), color="tab:blue", alpha=0.8)
    axes[0, 0].set_xlabel(r"$\log_{10}$ normalized weight")
    axes[0, 0].set_ylabel("particle count")
    axes[0, 0].set_title(f"Final SMC weights; ESS={ess:.3g}")

    axes[0, 1].plot(np.arange(1, sorted_weights.size + 1), cumulative, color="black", linewidth=1.2)
    axes[0, 1].set_xlabel("particles sorted by descending weight")
    axes[0, 1].set_ylabel("cumulative weight")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_xscale("log")

    if log_prob is not None and np.isfinite(log_prob).any():
        finite_log_prob = np.isfinite(log_prob)
        ranked = np.sort(log_prob[finite_log_prob])[::-1]
        axes[1, 0].plot(np.arange(1, ranked.size + 1), ranked, color="tab:red", linewidth=1.0)
        axes[1, 0].set_xlabel("particles sorted by descending log posterior")
        axes[1, 0].set_ylabel("log posterior")
        positive_mask = (weights > 0.0) & finite_log_prob
        axes[1, 1].scatter(
            np.log10(weights[positive_mask]),
            log_prob[positive_mask],
            s=10,
            alpha=0.75,
            color="tab:purple",
            linewidths=0.0,
            rasterized=True,
        )
        axes[1, 1].set_xlabel(r"$\log_{10}$ normalized weight")
        axes[1, 1].set_ylabel("log posterior")
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "smc_weight_diagnostics.pdf"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _weighted_variance(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    means = np.sum(samples * weights[:, None], axis=0)
    return np.sum(np.square(samples - means[None, :]) * weights[:, None], axis=0)


def _smc_corner_subset(
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    sample_weights: np.ndarray | None,
    *,
    max_params: int = SMC_CORNER_MAX_PARAMS,
) -> tuple[np.ndarray, list[ParameterSpec], np.ndarray] | None:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] < 2 or sample_array.shape[1] == 0 or not parameter_specs:
        return None
    weights = _valid_normalized_sample_weights(sample_weights, sample_array.shape[0])
    if weights is None:
        return None
    corner_samples, corner_specs = _corner_without_source_positions(sample_array, parameter_specs, "smc_corner.pdf")
    if corner_samples.ndim != 2 or not corner_specs:
        return None
    n_params = min(corner_samples.shape[1], len(corner_specs))
    corner_samples = np.asarray(corner_samples[:, :n_params], dtype=float)
    corner_specs = corner_specs[:n_params]
    finite_rows = np.isfinite(corner_samples).all(axis=1) & np.isfinite(weights)
    if int(np.sum(finite_rows)) < 2:
        return None
    corner_samples = corner_samples[finite_rows]
    weights = _valid_normalized_sample_weights(weights[finite_rows], int(np.sum(finite_rows)))
    if weights is None:
        return None

    spans = np.nanmax(corner_samples, axis=0) - np.nanmin(corner_samples, axis=0)
    variances = _weighted_variance(corner_samples, weights)
    dynamic = np.isfinite(spans) & (spans > 0.0) & np.isfinite(variances) & (variances > 0.0)
    if int(np.sum(dynamic)) < 2:
        return None
    cosmology_indices = [
        idx
        for idx, spec in enumerate(corner_specs)
        if dynamic[idx] and getattr(spec, "component_family", None) == "cosmology"
    ]
    other_indices = [
        idx
        for idx in np.argsort(-variances).tolist()
        if dynamic[idx] and idx not in set(cosmology_indices)
    ]
    selected = (cosmology_indices + other_indices)[: max(2, int(max_params))]
    if len(selected) < 2:
        return None
    return corner_samples[:, selected], [corner_specs[idx] for idx in selected], weights


def _plot_smc_corner(
    plot_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    sample_weights: np.ndarray | None,
    best_fit_values: dict[str, float] | None = None,
    previous_stage_best_values: dict[str, float] | None = None,
) -> None:
    if corner is None:
        return
    subset = _smc_corner_subset(samples, parameter_specs, sample_weights)
    if subset is None:
        return
    subset_samples, subset_specs, subset_weights = subset
    _log(
        None,
        f"[plot:corner] path={_plot_path(plot_dir, 'smc_corner.pdf')} ndim={len(subset_specs)} samples_shape={tuple(subset_samples.shape)}",
    )
    labels = [spec.name for spec in subset_specs]
    corner_kwargs = {**CORNER_PLOT_KWARGS, "plot_datapoints": True}
    fig = corner.corner(subset_samples, labels=labels, weights=subset_weights, **corner_kwargs)
    _overplot_corner_previous_stage_best_fit(fig, subset_specs, previous_stage_best_values)
    _overplot_corner_best_fit(fig, subset_specs, best_fit_values)
    fig.savefig(_plot_path(plot_dir, "smc_corner.pdf"), dpi=CORNER_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals_by_family(plot_dir: Path, family_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    values = family_df["source_plane_rms_arcsec"].fillna(family_df["rms_residual_arcsec"])
    ax.bar(family_df["family_id"].astype(str), values, color="tab:purple")
    ax.set_xlabel("Family")
    ax.set_ylabel("RMS residual [arcsec]")
    ax.set_title("Source-Plane RMS Residual by Family")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "residuals_by_family.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _source_plane_residual_components(
    state: BuildState,
    best_eval: EvaluationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx_values: list[np.ndarray] = []
    dy_values: list[np.ndarray] = []
    norm_x_values: list[np.ndarray] = []
    norm_y_values: list[np.ndarray] = []
    radial_values: list[np.ndarray] = []
    for family in state.family_data:
        if family.n_images < 2:
            continue
        pred = best_eval.family_predictions.get(family.family_id)
        if not pred or bool(pred.get("failed", False)):
            continue
        source_x = pred.get("source_x")
        source_y = pred.get("source_y")
        if source_x is None or source_y is None or not np.isfinite(source_x) or not np.isfinite(source_y):
            continue
        sigma = float(pred.get("source_sigma_eff_arcsec", family.sigma_arcsec))
        if not np.isfinite(sigma) or sigma <= 0.0:
            continue
        if "source_beta_x" not in pred or "source_beta_y" not in pred:
            continue
        beta_x = np.asarray(pred["source_beta_x"], dtype=float)
        beta_y = np.asarray(pred["source_beta_y"], dtype=float)
        if beta_x.shape != family.x_obs.shape or beta_y.shape != family.y_obs.shape:
            continue
        dx = beta_x - float(source_x)
        dy = beta_y - float(source_y)
        finite = np.isfinite(dx) & np.isfinite(dy)
        if not finite.any():
            continue
        dx = dx[finite]
        dy = dy[finite]
        dx_values.append(dx)
        dy_values.append(dy)
        norm_x_values.append(dx / sigma)
        norm_y_values.append(dy / sigma)
        radial_values.append(np.sqrt(dx**2 + dy**2) / sigma)
    dx = np.concatenate(dx_values) if dx_values else np.asarray([], dtype=float)
    dy = np.concatenate(dy_values) if dy_values else np.asarray([], dtype=float)
    norm_x = np.concatenate(norm_x_values) if norm_x_values else np.asarray([], dtype=float)
    norm_y = np.concatenate(norm_y_values) if norm_y_values else np.asarray([], dtype=float)
    radial = np.concatenate(radial_values) if radial_values else np.asarray([], dtype=float)
    return dx, dy, norm_x, norm_y, radial


def _plot_gaussian_component_histogram(
    ax: Any,
    component: np.ndarray,
    *,
    xlabel: str,
    title: str,
    standard_normal: bool = False,
) -> None:
    bins = min(60, max(16, int(np.sqrt(component.size))))
    ax.hist(component, bins=bins, density=True, alpha=0.65, color="tab:blue", label="components")
    x_min = float(np.nanmin(component))
    x_max = float(np.nanmax(component))
    span = max(abs(x_min), abs(x_max), 1.0)
    x_grid = np.linspace(-1.05 * span, 1.05 * span, 400)
    mu = float(np.mean(component))
    std = float(np.std(component, ddof=1)) if component.size > 1 else float("nan")
    if standard_normal:
        ax.plot(x_grid, norm.pdf(x_grid, 0.0, 1.0), color="black", linestyle="--", label="N(0, 1)")
    if np.isfinite(std) and std > 0:
        ax.plot(x_grid, norm.pdf(x_grid, mu, std), color="tab:red", label=f"fit mu={mu:.2f}, sigma={std:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)


def _plot_source_plane_residual_histogram(
    plot_dir: Path,
    state: BuildState,
    best_eval: EvaluationResult,
) -> None:
    dx, dy, norm_x, norm_y, radial = _source_plane_residual_components(state, best_eval)
    if radial.size == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    arcsec_component = np.concatenate([dx, dy])
    normalized_component = np.concatenate([norm_x, norm_y])
    _plot_gaussian_component_histogram(
        axes[0],
        arcsec_component,
        xlabel="Signed source-plane residual [arcsec]",
        title="Signed Components",
    )
    _plot_gaussian_component_histogram(
        axes[1],
        normalized_component,
        xlabel="Signed source-plane residual / sigma_eff",
        title="Normalized Components",
        standard_normal=True,
    )

    bins = min(60, max(16, int(np.sqrt(radial.size))))
    max_x = max(4.0, float(np.nanmax(radial)) * 1.05)
    axes[2].hist(radial, bins=bins, density=True, alpha=0.65, color="tab:purple", label="radial")
    x_grid = np.linspace(0.0, max_x, 400)
    rayleigh_pdf = x_grid * np.exp(-0.5 * x_grid**2)
    axes[2].plot(x_grid, rayleigh_pdf, color="black", linestyle="--", label="Rayleigh scale=1")
    median = float(np.median(radial))
    p68, p95 = np.quantile(radial, [0.68, 0.95])
    axes[2].axvline(median, color="tab:red", linewidth=1.5, label=f"median={median:.2f}")
    axes[2].text(
        0.98,
        0.95,
        f"N={radial.size}\n68%={p68:.2f}\n95%={p95:.2f}",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    axes[2].set_xlabel("Radial source-plane residual / sigma_eff")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Radial Residuals")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "source_plane_residual_histogram.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_image_plane_fit(plot_dir: Path, state: BuildState, best_eval: EvaluationResult) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = _family_color_map([family.family_id for family in state.family_data])
    for family in state.family_data:
        color = colors[str(family.family_id)]
        pred = best_eval.family_predictions.get(
            str(family.family_id),
            best_eval.family_predictions.get(family.family_id, {}),
        )
        n_images = int(getattr(family, "n_images", len(family.x_obs)))
        x_pred = np.asarray(pred.get("x_pred", np.full(n_images, np.nan)), dtype=float).reshape(-1)
        y_pred = np.asarray(pred.get("y_pred", np.full(n_images, np.nan)), dtype=float).reshape(-1)
        if x_pred.shape != (n_images,):
            resized = np.full(n_images, np.nan, dtype=float)
            resized[: min(n_images, x_pred.size)] = x_pred[: min(n_images, x_pred.size)]
            x_pred = resized
        if y_pred.shape != (n_images,):
            resized = np.full(n_images, np.nan, dtype=float)
            resized[: min(n_images, y_pred.size)] = y_pred[: min(n_images, y_pred.size)]
            y_pred = resized
        ax.scatter(family.x_obs, family.y_obs, marker="x", color=color, label=f"{family.family_id} obs")
        finite_model = np.isfinite(x_pred) & np.isfinite(y_pred)
        if finite_model.any():
            ax.scatter(x_pred[finite_model], y_pred[finite_model], marker="o", color=color, s=36, alpha=0.65)
            for x0, y0, x1, y1 in zip(family.x_obs, family.y_obs, x_pred, y_pred):
                if np.isfinite(x1) and np.isfinite(y1):
                    ax.plot([x0, x1], [y0, y1], color=color, alpha=0.35, linewidth=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_title("Observed vs Exact-Validated Image Positions")
    if len(state.family_data) <= 15:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "image_plane_fit.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fit_quality_quantiles(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    q16, q50, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
    return float(q16), float(q50), float(q84)


def _fit_quality_median_std(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    median = float(np.median(finite))
    std = float(np.std(finite))
    return median - std, median, median + std


def _capped_fit_quality_samples(samples: np.ndarray, max_draws: int) -> np.ndarray:
    sample_array = _finite_sample_rows(samples)
    if sample_array.shape[0] == 0 or int(max_draws) <= 0:
        return np.empty((0, sample_array.shape[1]), dtype=float)
    if sample_array.shape[0] <= int(max_draws):
        return sample_array
    indices = np.linspace(0, sample_array.shape[0] - 1, int(max_draws), dtype=int)
    return sample_array[indices]


def _reported_physical_to_latent_vector(evaluator: Any, theta: np.ndarray) -> np.ndarray:
    if hasattr(evaluator, "reported_physical_to_latent_parameter_vector"):
        return np.asarray(evaluator.reported_physical_to_latent_parameter_vector(theta), dtype=float)
    return _convert_theta_to_latent(np.asarray(theta, dtype=float), evaluator.state.parameter_specs)


def _fit_quality_image_sigma_int(evaluator: Any, params_latent: np.ndarray) -> float:
    if not hasattr(evaluator, "_image_sigma_int_numpy"):
        return 0.0
    try:
        value = float(evaluator._image_sigma_int_numpy(params_latent))
    except Exception:
        return 0.0
    return value if np.isfinite(value) else 0.0


def _fit_quality_image_sigma_eff(
    measurement_sigma_arcsec: float,
    image_sigma_int_arcsec: float,
    covariance_floor: float,
) -> float:
    variance = (
        float(measurement_sigma_arcsec) ** 2
        + float(image_sigma_int_arcsec) ** 2
        + max(float(covariance_floor), 0.0)
    )
    if not np.isfinite(variance) or variance < 0.0:
        return np.nan
    return float(np.sqrt(variance))


def _finite_or(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _covered_by_inflated_interval(
    observed: float,
    lower: float,
    upper: float,
    center: float,
    sigma_eff: float,
) -> bool:
    if not np.isfinite(observed) or not np.isfinite(sigma_eff):
        return False
    lo = lower if np.isfinite(lower) else center
    hi = upper if np.isfinite(upper) else center
    if not np.isfinite(lo) or not np.isfinite(hi):
        return False
    lo, hi = sorted((float(lo), float(hi)))
    return bool((float(observed) >= lo - float(sigma_eff)) and (float(observed) <= hi + float(sigma_eff)))


def _fit_quality_value(df: pd.DataFrame, column: str, fallback_column: str | None = None) -> np.ndarray:
    if column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    else:
        values = np.full(len(df), np.nan, dtype=float)
    if fallback_column is not None and fallback_column in df.columns:
        fallback = pd.to_numeric(df[fallback_column], errors="coerce").to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, fallback)
    return values


def _merge_fit_quality_with_magnification(image_df: pd.DataFrame, magnification_df: pd.DataFrame) -> pd.DataFrame:
    if image_df.empty or magnification_df.empty:
        return pd.DataFrame()
    mag_columns = [
        column
        for column in [
            "family_id",
            "image_label",
            "magnification_model",
            "magnification_model_q16",
            "magnification_model_q50",
            "magnification_model_q84",
        ]
        if column in magnification_df.columns
    ]
    if "image_label" not in mag_columns:
        return pd.DataFrame()
    merge_keys = ["image_label"]
    if "family_id" in image_df.columns and "family_id" in mag_columns:
        merge_keys.insert(0, "family_id")
    return image_df.merge(magnification_df[mag_columns], on=merge_keys, how="inner")


def _unavailable_image_count_info(family: Any, reason: str) -> dict[str, Any]:
    return _shared_unavailable_image_count_info(family, reason)


def _successful_image_count_info(family: Any) -> dict[str, Any]:
    return _shared_successful_image_count_info(family)


def _image_count_info_from_exact_details(family: Any, details: dict[str, Any]) -> dict[str, Any]:
    return _shared_image_count_info_from_exact_details(family, details)


def _model_count_fields_from_count_info(count_info: dict[str, Any]) -> dict[str, Any]:
    return _shared_model_count_fields_from_count_info(count_info)


def _image_count_recovery_row(family: Any, count_info: dict[str, Any]) -> dict[str, Any]:
    return _shared_image_count_recovery_row(family, count_info)


def _image_count_recovery_table(state: BuildState, image_df: pd.DataFrame) -> pd.DataFrame:
    return _shared_image_count_recovery_table(state, image_df)


def _image_count_recovery_summary(image_count_df: pd.DataFrame | None) -> dict[str, Any]:
    return _shared_image_count_recovery_summary(image_count_df)


def _clone_fit_quality_evaluator(evaluator: Any, args: argparse.Namespace) -> Any:
    from .cluster_solver import (  # local import avoids a module import cycle at plotting import time
        ClusterJAXEvaluator,
        DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        DEFAULT_ACTIVE_SCALING_MIN,
        DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
        DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
        DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS,
        DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
        DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC,
        DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC,
        DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC,
        DEFAULT_CRITICAL_ARC_BASE_PROB,
        DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE,
        DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE,
        DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC,
        DEFAULT_CRITICAL_ARC_MAX_PROB,
        DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS,
        DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD,
        DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC,
        DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC,
        SAMPLE_LIKELIHOOD_SOURCE,
    )

    return ClusterJAXEvaluator(
        state=evaluator.state,
        match_tolerance_arcsec=float(
            getattr(args, "match_tolerance_arcsec", getattr(evaluator, "match_tolerance_arcsec", DEFAULT_MATCH_TOLERANCE))
        ),
        sampling_engine=str(getattr(args, "sampling_engine", getattr(evaluator, "sampling_engine", "full"))),
        active_scaling_galaxies=getattr(args, "active_scaling_galaxies", None),
        active_scaling_selection=str(
            getattr(args, "active_scaling_selection", getattr(evaluator, "active_scaling_selection", "adaptive"))
        ),
        active_scaling_cumulative_fraction=float(
            getattr(
                args,
                "active_scaling_cumulative_fraction",
                getattr(evaluator, "active_scaling_cumulative_fraction", DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION),
            )
        ),
        active_scaling_min=int(getattr(args, "active_scaling_min", getattr(evaluator, "active_scaling_min", DEFAULT_ACTIVE_SCALING_MIN))),
        refresh_every=int(getattr(args, "refresh_every", getattr(evaluator, "refresh_every", DEFAULT_REFRESH_EVERY))),
        refresh_param_drift_frac=float(
            getattr(args, "refresh_param_drift_frac", getattr(evaluator, "refresh_param_drift_frac", DEFAULT_REFRESH_PARAM_DRIFT_FRAC))
        ),
        source_plane_covariance_floor=float(
            getattr(args, "source_plane_covariance_floor", getattr(evaluator, "source_plane_covariance_floor", 1.0e-6))
        ),
        source_plane_outlier_sigma_arcsec=float(
            getattr(
                args,
                "source_plane_outlier_sigma_arcsec",
                getattr(evaluator, "source_plane_outlier_sigma_arcsec", DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC),
            )
        ),
        sample_likelihood_mode=str(
            getattr(args, "sample_likelihood_mode", getattr(evaluator, "sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE))
        ),
        image_plane_newton_steps=int(getattr(args, "image_plane_newton_steps", getattr(evaluator, "image_plane_newton_steps", 0))),
        anchored_image_plane_solve_steps=int(
            getattr(
                args,
                "anchored_image_plane_solve_steps",
                getattr(evaluator, "anchored_image_plane_solve_steps", DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS),
            )
        ),
        anchored_image_plane_trust_radius_arcsec=float(
            getattr(
                args,
                "anchored_image_plane_trust_radius_arcsec",
                getattr(evaluator, "anchored_image_plane_trust_radius_arcsec", DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC),
            )
        ),
        anchored_image_plane_lm_damping_relative=float(
            getattr(
                args,
                "anchored_image_plane_lm_damping_relative",
                getattr(evaluator, "anchored_image_plane_lm_damping_relative", DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE),
            )
        ),
        anchored_image_plane_lm_damping_absolute=float(
            getattr(
                args,
                "anchored_image_plane_lm_damping_absolute",
                getattr(evaluator, "anchored_image_plane_lm_damping_absolute", DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE),
            )
        ),
        critical_arc_critical_direction_sigma_arcsec=float(
            getattr(
                args,
                "critical_arc_critical_direction_sigma_arcsec",
                getattr(evaluator, "critical_arc_critical_direction_sigma_arcsec", DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC),
            )
        ),
        critical_arc_base_prob=float(
            getattr(args, "critical_arc_base_prob", getattr(evaluator, "critical_arc_base_prob", DEFAULT_CRITICAL_ARC_BASE_PROB))
        ),
        critical_arc_max_prob=float(
            getattr(args, "critical_arc_max_prob", getattr(evaluator, "critical_arc_max_prob", DEFAULT_CRITICAL_ARC_MAX_PROB))
        ),
        critical_arc_singular_threshold=float(
            getattr(
                args,
                "critical_arc_singular_threshold",
                getattr(evaluator, "critical_arc_singular_threshold", DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD),
            )
        ),
        critical_arc_singular_softness=float(
            getattr(
                args,
                "critical_arc_singular_softness",
                getattr(evaluator, "critical_arc_singular_softness", DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS),
            )
        ),
        critical_arc_lm_damping_relative=float(
            getattr(
                args,
                "critical_arc_lm_damping_relative",
                getattr(evaluator, "critical_arc_lm_damping_relative", DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE),
            )
        ),
        critical_arc_lm_damping_absolute=float(
            getattr(
                args,
                "critical_arc_lm_damping_absolute",
                getattr(evaluator, "critical_arc_lm_damping_absolute", DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE),
            )
        ),
        critical_arc_lm_trust_radius_arcsec=float(
            getattr(
                args,
                "critical_arc_lm_trust_radius_arcsec",
                getattr(evaluator, "critical_arc_lm_trust_radius_arcsec", DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC),
            )
        ),
        arc_aware_noncritical_support_radius_arcsec=float(
            getattr(
                args,
                "arc_aware_noncritical_support_radius_arcsec",
                getattr(
                    evaluator,
                    "arc_aware_noncritical_support_radius_arcsec",
                    DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC,
                ),
            )
        ),
        arc_aware_max_arclength_arcsec=float(
            getattr(
                args,
                "arc_aware_max_arclength_arcsec",
                getattr(evaluator, "arc_aware_max_arclength_arcsec", DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC),
            )
        ),
        arc_aware_curve_step_arcsec=float(
            getattr(
                args,
                "arc_aware_curve_step_arcsec",
                getattr(evaluator, "arc_aware_curve_step_arcsec", DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC),
            )
        ),
        image_plane_scatter_floor_arcsec=float(
            getattr(
                args,
                "image_plane_scatter_floor_arcsec",
                getattr(evaluator, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC),
            )
        ),
        fixed_image_sigma_int_arcsec=getattr(
            args,
            "fix_image_sigma_int_arcsec",
            getattr(evaluator, "fixed_image_sigma_int_arcsec", None),
        ),
        evidence_source_prior_sigma_arcsec=getattr(
            args,
            "evidence_source_prior_sigma_arcsec",
            getattr(evaluator, "evidence_source_prior_sigma_arcsec", None),
        ),
        evidence_source_prior_mean_x_arcsec=float(
            getattr(
                args,
                "evidence_source_prior_mean_x_arcsec",
                getattr(evaluator, "evidence_source_prior_mean_x_arcsec", 0.0),
            )
        ),
        evidence_source_prior_mean_y_arcsec=float(
            getattr(
                args,
                "evidence_source_prior_mean_y_arcsec",
                getattr(evaluator, "evidence_source_prior_mean_y_arcsec", 0.0),
            )
        ),
        quick_diagnostics=bool(getattr(args, "quick_diagnostics", getattr(evaluator, "quick_diagnostics", False))),
    )


def _fit_quality_detail_array(details: dict[str, Any], key: str, size: int, dtype: Any = float) -> np.ndarray | None:
    return _shared_diagnostic_detail_array(details, key, size, dtype)


def _fit_quality_extra_image_rows(
    family: Any,
    details: dict[str, Any],
    model_count_fields: dict[str, Any],
) -> list[dict[str, Any]]:
    return _shared_extra_image_rows(family, details, model_count_fields)


def _fit_quality_prediction_for_family_latent(
    evaluator: Any,
    family: Any,
    params_latent: np.ndarray,
    image_sigma_int: float,
    covariance_floor: float,
    quick_diagnostics: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    magnification_rows: list[dict[str, Any]] = []
    params_latent = np.asarray(params_latent, dtype=float)
    n_images = int(family.n_images)
    exact_details: dict[str, Any] | None = None
    unavailable_reason = "quick_diagnostics" if quick_diagnostics else "not_run"
    unavailable_status = "unknown"
    if not quick_diagnostics:
        try:
            if hasattr(evaluator, "_exact_family_prediction_details"):
                exact_details = evaluator._exact_family_prediction_details(params_latent, family)
            else:
                exact_prediction = evaluator._exact_family_prediction(params_latent, family)
                if exact_prediction is not None:
                    x_exact, y_exact, _exact_rms = exact_prediction
                    x_exact = np.asarray(x_exact, dtype=float)
                    y_exact = np.asarray(y_exact, dtype=float)
                    if x_exact.shape == (n_images,) and y_exact.shape == (n_images,):
                        exact_details = {
                            **_successful_image_count_info(family),
                            "failed": False,
                            "x_pred": x_exact,
                            "y_pred": y_exact,
                            "exact_image_rms": _exact_rms,
                        }
                else:
                    unavailable_reason = "exact_prediction_failed"
                    unavailable_status = "not_recovered"
        except Exception:
            unavailable_reason = "exact_prediction_exception"
            unavailable_status = "unknown"

    mu = np.full(n_images, np.nan, dtype=float)
    magnification_failed = True
    try:
        model, _solver = evaluator._get_exact_model_solver(family.z_source)
        packed_state = evaluator._build_packed_lens_state(jnp.asarray(params_latent, dtype=jnp.float64), family.z_source)
        kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
        mu_values = np.asarray(
            model.magnification(
                jnp.asarray(family.x_obs, dtype=jnp.float64),
                jnp.asarray(family.y_obs, dtype=jnp.float64),
                kwargs_lens,
            ),
            dtype=float,
        )
        if mu_values.shape == (n_images,):
            mu = mu_values
            magnification_failed = False
    except Exception:
        magnification_failed = True

    sigma_arcsec = _finite_or(getattr(family, "sigma_arcsec", np.nan))
    sigma_eff = _fit_quality_image_sigma_eff(sigma_arcsec, image_sigma_int, covariance_floor)
    image_rows, extra_image_rows, count_info = _shared_family_image_recovery_rows(
        family,
        exact_details,
        sigma_arcsec=sigma_arcsec,
        image_sigma_int_arcsec=image_sigma_int,
        image_sigma_eff_arcsec=sigma_eff,
        unavailable_reason=unavailable_reason,
        unavailable_status=unavailable_status,
    )
    magnification_common_keys = [
        "family_id",
        "image_label",
        "x_obs_arcsec",
        "y_obs_arcsec",
        "z_source",
        "effective_z_source",
        "sigma_arcsec",
        "image_sigma_int_arcsec",
        "image_sigma_eff_arcsec",
        "radius_arcsec",
        "angle_deg",
        "image_recovery_status",
        "model_produced_image_count",
        "model_recovered_image_count",
        "model_missing_image_count",
        "model_extra_image_count",
        "model_multiplicity_failed",
        "model_multiplicity_failure_reason",
    ]
    for image_row, mu_value in zip(image_rows, mu):
        common = {key: image_row[key] for key in magnification_common_keys if key in image_row}
        magnification_rows.append(
            {
                **common,
                "magnification_model": float(mu_value),
                "magnification_prediction_failed": bool(magnification_failed),
            }
        )
    return {
        "image_rows": image_rows,
        "magnification_rows": magnification_rows,
        "image_count_rows": [_image_count_recovery_row(family, count_info)],
        "extra_image_rows": extra_image_rows,
    }


def _fit_quality_prediction_for_latent(
    evaluator: Any,
    state: BuildState,
    params_latent: np.ndarray,
    quick_diagnostics: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    params_latent = np.asarray(params_latent, dtype=float)
    image_sigma_int = _fit_quality_image_sigma_int(evaluator, params_latent)
    covariance_floor = _finite_or(getattr(evaluator, "source_plane_covariance_floor", 0.0), 0.0)
    prediction: dict[str, list[dict[str, Any]]] = {
        "image_rows": [],
        "magnification_rows": [],
        "image_count_rows": [],
        "extra_image_rows": [],
    }
    for family in state.family_data:
        family_prediction = _fit_quality_prediction_for_family_latent(
            evaluator,
            family,
            params_latent,
            image_sigma_int,
            covariance_floor,
            quick_diagnostics=quick_diagnostics,
        )
        prediction["image_rows"].extend(family_prediction["image_rows"])
        prediction["magnification_rows"].extend(family_prediction["magnification_rows"])
        prediction["image_count_rows"].extend(family_prediction.get("image_count_rows", []))
        prediction["extra_image_rows"].extend(family_prediction.get("extra_image_rows", []))
    return prediction


def _fit_quality_family_cost_metadata(family: Any) -> dict[str, Any]:
    min_distance = 0.2
    search_window = _finite_or(getattr(family, "search_window", np.nan))
    if not np.isfinite(search_window):
        x_obs = np.asarray(getattr(family, "x_obs", []), dtype=float)
        y_obs = np.asarray(getattr(family, "y_obs", []), dtype=float)
        span_x = float(np.ptp(x_obs)) if x_obs.size > 1 else 0.0
        span_y = float(np.ptp(y_obs)) if y_obs.size > 1 else 0.0
        search_window = max(span_x, span_y) + 10.0
    num_pix = int(round(float(search_window) / float(min_distance)) + 0.5) if min_distance > 0.0 else 0
    grid_points = max(0, num_pix * num_pix)
    return {
        "min_distance": float(min_distance),
        "search_window": float(search_window),
        "num_pix": int(num_pix),
        "grid_points": int(grid_points),
    }


def _posterior_fit_quality_predictions(
    evaluator: Any,
    state: BuildState,
    sample_latents: list[np.ndarray],
    args: argparse.Namespace,
) -> list[dict[str, list[dict[str, Any]]]]:
    if not sample_latents:
        return []
    worker_count = max(1, int(jax_cpu_worker_count()))
    quick_diagnostics = bool(getattr(args, "quick_diagnostics", getattr(evaluator, "quick_diagnostics", False)))
    n_families = len(state.family_data)
    n_tasks = len(sample_latents) * n_families
    sample_latents = [np.asarray(sample, dtype=float) for sample in sample_latents]
    image_sigma_int_by_sample = [_fit_quality_image_sigma_int(evaluator, sample) for sample in sample_latents]
    covariance_floor = _finite_or(getattr(evaluator, "source_plane_covariance_floor", 0.0), 0.0)
    worker_evaluators: list[Any] = []
    worker_lock = threading.Lock()
    worker_local = threading.local()

    def worker_evaluator() -> Any:
        cached = getattr(worker_local, "evaluator", None)
        if cached is None:
            cached = _clone_fit_quality_evaluator(evaluator, args)
            worker_local.evaluator = cached
            with worker_lock:
                worker_evaluators.append(cached)
        return cached

    def evaluate_family(sample_index: int, family_index: int) -> tuple[int, int, dict[str, list[dict[str, Any]]]]:
        return (
            sample_index,
            family_index,
            _fit_quality_prediction_for_family_latent(
                worker_evaluator(),
                state.family_data[family_index],
                sample_latents[sample_index],
                image_sigma_int_by_sample[sample_index],
                covariance_floor,
                quick_diagnostics=quick_diagnostics,
            ),
        )

    family_predictions: list[list[dict[str, list[dict[str, Any]]] | None]] = [
        [None] * n_families for _ in sample_latents
    ]
    family_costs = [_fit_quality_family_cost_metadata(family) for family in state.family_data]
    task_indices = [
        (sample_index, family_index)
        for sample_index in range(len(sample_latents))
        for family_index in range(n_families)
    ]
    task_indices.sort(key=lambda index: family_costs[index[1]]["grid_points"], reverse=True)

    completed_family_tasks = 0
    completed_draws = 0
    completed_families_by_draw = [0] * len(sample_latents)

    def family_progress_description(status: str, sample_index: int, family_index: int) -> str:
        family = state.family_data[family_index]
        cost = family_costs[family_index]
        return (
            f"fit quality exact: {completed_family_tasks}/{n_tasks} family diagnostics | "
            f"{status} draw {sample_index + 1}/{len(sample_latents)} "
            f"family={family.family_id} z={float(family.z_source):.4f} "
            f"window={float(cost['search_window']):.1f} grid={int(cost['num_pix'])}x{int(cost['num_pix'])}"
        )

    def draw_progress_description() -> str:
        return f"draw progress: {completed_draws}/{len(sample_latents)} complete"

    def update_task_progress(
        progress: Progress | None,
        family_task_id: int | None,
        status: str,
        sample_index: int,
        family_index: int,
        *,
        advance: bool = False,
    ) -> None:
        if progress is None or family_task_id is None:
            return
        progress.update(family_task_id, description=family_progress_description(status, sample_index, family_index))
        if advance:
            progress.advance(family_task_id)

    def mark_family_completed(
        progress: Progress | None,
        family_task_id: int | None,
        draw_task_id: int | None,
        sample_index: int,
        family_index: int,
    ) -> None:
        nonlocal completed_family_tasks, completed_draws
        completed_family_tasks += 1
        completed_families_by_draw[sample_index] += 1
        update_task_progress(progress, family_task_id, "completed", sample_index, family_index, advance=True)
        if completed_families_by_draw[sample_index] == n_families:
            completed_draws += 1
            if progress is not None and draw_task_id is not None:
                progress.update(draw_task_id, description=draw_progress_description())
                progress.advance(draw_task_id)
            if not quick_diagnostics:
                _log(
                    args,
                    (
                        f"[plot:fit_quality] draw {sample_index + 1}/{len(sample_latents)} complete "
                        f"families={completed_families_by_draw[sample_index]}/{n_families} "
                        f"completed_tasks={completed_family_tasks}/{n_tasks}"
                    ),
                )

    def combine_predictions() -> list[dict[str, list[dict[str, Any]]]]:
        predictions: list[dict[str, list[dict[str, Any]]]] = []
        for sample_predictions in family_predictions:
            combined = {
                "image_rows": [],
                "magnification_rows": [],
                "image_count_rows": [],
                "extra_image_rows": [],
            }
            for family_prediction in sample_predictions:
                if family_prediction is None:
                    continue
                combined["image_rows"].extend(family_prediction["image_rows"])
                combined["magnification_rows"].extend(family_prediction["magnification_rows"])
                combined["image_count_rows"].extend(family_prediction.get("image_count_rows", []))
                combined["extra_image_rows"].extend(family_prediction.get("extra_image_rows", []))
            predictions.append(combined)
        return predictions

    progress_enabled = n_tasks > 0 and not quick_diagnostics and not bool(getattr(args, "quiet", False))
    max_workers = min(worker_count, n_tasks)
    largest_num_pix = max((int(cost["num_pix"]) for cost in family_costs), default=0)
    total_grid_points = sum(int(cost["grid_points"]) for cost in family_costs) * len(sample_latents)
    _log(
        args,
        (
            f"[plot:fit_quality] family diagnostics tasks={n_tasks} workers={max_workers} "
            f"families={n_families} draws={len(sample_latents)} "
            f"largest_grid={largest_num_pix}x{largest_num_pix} total_grid_points={total_grid_points}"
        ),
    )
    progress_cm = (
        Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=False,
        )
        if progress_enabled
        else None
    )
    progress: Progress | None = None
    family_task_id: int | None = None
    draw_task_id: int | None = None
    try:
        if progress_cm is not None:
            progress = progress_cm.__enter__()
            family_task_id = progress.add_task(f"fit quality exact: 0/{n_tasks} family diagnostics", total=n_tasks)
            draw_task_id = progress.add_task(draw_progress_description(), total=len(sample_latents))
        if worker_count <= 1 or n_tasks <= 1:
            for sample_index, family_index in task_indices:
                sample = sample_latents[sample_index]
                family = state.family_data[family_index]
                update_task_progress(progress, family_task_id, "running", sample_index, family_index)
                _sample_index, _family_index, prediction = (
                    sample_index,
                    family_index,
                    _fit_quality_prediction_for_family_latent(
                        evaluator,
                        family,
                        sample,
                        image_sigma_int_by_sample[sample_index],
                        covariance_floor,
                        quick_diagnostics=quick_diagnostics,
                    ),
                )
                family_predictions[_sample_index][_family_index] = prediction
                mark_family_completed(progress, family_task_id, draw_task_id, _sample_index, _family_index)
            return combine_predictions()
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {}
                for sample_index, family_index in task_indices:
                    update_task_progress(progress, family_task_id, "queued", sample_index, family_index)
                    future_to_index[executor.submit(evaluate_family, sample_index, family_index)] = (
                        sample_index,
                        family_index,
                    )
                for future in as_completed(future_to_index):
                    sample_index, family_index, prediction = future.result()
                    family_predictions[sample_index][family_index] = prediction
                    mark_family_completed(progress, family_task_id, draw_task_id, sample_index, family_index)
        except Exception as exc:
            _log(None, f"[plot:fit_quality] parallel posterior diagnostics failed; retrying serially error={exc}")
            for cached_evaluator in worker_evaluators:
                if hasattr(cached_evaluator, "release_runtime_caches"):
                    cached_evaluator.release_runtime_caches()
            worker_evaluators.clear()
            for sample_index, family_index in task_indices:
                sample = sample_latents[sample_index]
                family = state.family_data[family_index]
                update_task_progress(progress, family_task_id, "running", sample_index, family_index)
                _sample_index, _family_index, prediction = (
                    sample_index,
                    family_index,
                    _fit_quality_prediction_for_family_latent(
                        evaluator,
                        family,
                        sample,
                        image_sigma_int_by_sample[sample_index],
                        covariance_floor,
                        quick_diagnostics=quick_diagnostics,
                    ),
                )
                family_predictions[_sample_index][_family_index] = prediction
                mark_family_completed(progress, family_task_id, draw_task_id, _sample_index, _family_index)
        return combine_predictions()
    finally:
        for cached_evaluator in worker_evaluators:
            if hasattr(cached_evaluator, "release_runtime_caches"):
                cached_evaluator.release_runtime_caches()
        if progress_cm is not None:
            progress_cm.__exit__(None, None, None)


def _fit_quality_tables(
    state: BuildState,
    evaluator: Any,
    best_fit: np.ndarray,
    results: PosteriorResults,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best_fit_latent = _reported_physical_to_latent_vector(evaluator, np.asarray(best_fit, dtype=float))
    quick_diagnostics = bool(getattr(args, "quick_diagnostics", getattr(evaluator, "quick_diagnostics", False)))
    summary_fn = _fit_quality_median_std if quick_diagnostics else _fit_quality_quantiles

    max_draws = max(0, int(getattr(args, "fit_quality_draws", 0)))
    posterior_samples = _capped_fit_quality_samples(results.samples, max_draws)
    sample_latents = [
        _reported_physical_to_latent_vector(evaluator, np.asarray(sample, dtype=float))
        for sample in posterior_samples
    ]
    all_predictions = _posterior_fit_quality_predictions(evaluator, state, [best_fit_latent, *sample_latents], args)
    best_prediction = (
        all_predictions[0]
        if all_predictions
        else {"image_rows": [], "magnification_rows": [], "image_count_rows": [], "extra_image_rows": []}
    )
    posterior_predictions = all_predictions[1:]

    image_draws_by_label: dict[str, list[dict[str, Any]]] = {}
    magnification_draws_by_label: dict[str, list[dict[str, Any]]] = {}
    for prediction in posterior_predictions:
        for row in prediction["image_rows"]:
            image_draws_by_label.setdefault(str(row["image_label"]), []).append(row)
        for row in prediction["magnification_rows"]:
            magnification_draws_by_label.setdefault(str(row["image_label"]), []).append(row)

    image_rows: list[dict[str, Any]] = []
    for row in best_prediction["image_rows"]:
        label = str(row["image_label"])
        draws = image_draws_by_label.get(label, [])
        x16, x50, x84 = summary_fn([draw["x_model_arcsec"] for draw in draws])
        y16, y50, y84 = summary_fn([draw["y_model_arcsec"] for draw in draws])
        r16, r50, r84 = summary_fn([draw["image_residual_arcsec"] for draw in draws])
        arc_r16, arc_r50, arc_r84 = summary_fn(
            [draw.get("arc_aware_image_residual_arcsec", np.nan) for draw in draws]
        )
        sigma_eff = _finite_or(row.get("image_sigma_eff_arcsec", np.nan))
        residual_norm = (
            float(row["image_residual_arcsec"]) / sigma_eff
            if np.isfinite(float(row["image_residual_arcsec"])) and np.isfinite(sigma_eff) and sigma_eff > 0.0
            else np.nan
        )
        residual_norm_q50 = r50 / sigma_eff if np.isfinite(r50) and np.isfinite(sigma_eff) and sigma_eff > 0.0 else np.nan
        covered_x = _covered_by_inflated_interval(
            float(row["x_obs_arcsec"]),
            x16,
            x84,
            float(row["x_model_arcsec"]),
            sigma_eff,
        )
        covered_y = _covered_by_inflated_interval(
            float(row["y_obs_arcsec"]),
            y16,
            y84,
            float(row["y_model_arcsec"]),
            sigma_eff,
        )
        valid_draws = sum(
            1
            for draw in draws
            if np.isfinite(float(draw["x_model_arcsec"]) + float(draw["y_model_arcsec"]))
        )
        image_rows.append(
            {
                **row,
                "x_model_q16": x16,
                "x_model_q50": x50,
                "x_model_q84": x84,
                "y_model_q16": y16,
                "y_model_q50": y50,
                "y_model_q84": y84,
                "image_residual_q16": r16,
                "image_residual_q50": r50,
                "image_residual_q84": r84,
                "arc_aware_image_residual_q16": arc_r16,
                "arc_aware_image_residual_q50": arc_r50,
                "arc_aware_image_residual_q84": arc_r84,
                "residual_norm": residual_norm,
                "residual_norm_q50": residual_norm_q50,
                "covered_x_1sigma": bool(covered_x),
                "covered_y_1sigma": bool(covered_y),
                "covered_xy_1sigma": bool(covered_x and covered_y),
                "posterior_valid_draws": int(valid_draws),
                "posterior_failed_draws": int(len(draws) - valid_draws),
            }
        )
    magnification_rows: list[dict[str, Any]] = []
    for row in best_prediction["magnification_rows"]:
        label = str(row["image_label"])
        draws = magnification_draws_by_label.get(label, [])
        q16, q50, q84 = summary_fn([draw["magnification_model"] for draw in draws])
        valid_draws = sum(1 for draw in draws if np.isfinite(float(draw["magnification_model"])))
        magnification_rows.append(
            {
                **row,
                "magnification_model_q16": q16,
                "magnification_model_q50": q50,
                "magnification_model_q84": q84,
                "posterior_valid_draws": int(valid_draws),
                "posterior_failed_draws": int(len(draws) - valid_draws),
            }
        )
    return pd.DataFrame(image_rows), pd.DataFrame(magnification_rows), pd.DataFrame(best_prediction.get("extra_image_rows", []))


def _plot_image_recovery_fit_quality(
    image_df: pd.DataFrame,
    path: Path,
    extra_image_df: pd.DataFrame | None = None,
) -> None:
    if image_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    x_obs = image_df["x_obs_arcsec"].to_numpy(dtype=float)
    y_obs = image_df["y_obs_arcsec"].to_numpy(dtype=float)
    x_best = _fit_quality_value(image_df, "x_model_arcsec")
    y_best = _fit_quality_value(image_df, "y_model_arcsec")
    x_model = _fit_quality_value(image_df, "x_model_q50", "x_model_arcsec")
    y_model = _fit_quality_value(image_df, "y_model_q50", "y_model_arcsec")
    x_model = np.where(np.isfinite(x_model), x_model, x_best)
    y_model = np.where(np.isfinite(y_model), y_model, y_best)

    if "image_recovery_status" in image_df.columns:
        status = image_df["image_recovery_status"].fillna("unknown").astype(str).to_numpy()
    elif "exact_image_prediction_failed" in image_df.columns:
        failed = image_df["exact_image_prediction_failed"].astype(bool).to_numpy()
        status = np.where(failed, "not_recovered", "recovered")
    else:
        status = np.where(np.isfinite(x_model) & np.isfinite(y_model), "recovered", "not_recovered")
    recovered = status == "recovered"
    not_recovered = ~recovered
    finite_model = np.isfinite(x_model) & np.isfinite(y_model)
    finite_recovered_model = recovered & finite_model

    if recovered.any():
        ax.scatter(
            x_obs[recovered],
            y_obs[recovered],
            marker="x",
            color="tab:green",
            s=18,
            linewidths=0.9,
            label="recovered",
        )
    if not_recovered.any():
        ax.scatter(
            x_obs[not_recovered],
            y_obs[not_recovered],
            marker="x",
            color="tab:red",
            s=18,
            linewidths=0.9,
            label="not recovered",
        )
    if finite_recovered_model.any():
        x16 = _fit_quality_value(image_df, "x_model_q16")
        x84 = _fit_quality_value(image_df, "x_model_q84")
        y16 = _fit_quality_value(image_df, "y_model_q16")
        y84 = _fit_quality_value(image_df, "y_model_q84")
        xerr = [
            np.where(np.isfinite(x16), np.maximum(0.0, x_model - x16), 0.0)[finite_recovered_model],
            np.where(np.isfinite(x84), np.maximum(0.0, x84 - x_model), 0.0)[finite_recovered_model],
        ]
        yerr = [
            np.where(np.isfinite(y16), np.maximum(0.0, y_model - y16), 0.0)[finite_recovered_model],
            np.where(np.isfinite(y84), np.maximum(0.0, y84 - y_model), 0.0)[finite_recovered_model],
        ]
        ax.errorbar(
            x_model[finite_recovered_model],
            y_model[finite_recovered_model],
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=_color_with_alpha("tab:green", 0.75),
            ecolor=_color_with_alpha("tab:green", 0.35),
            markersize=3,
            elinewidth=0.8,
            capsize=1.5,
            label=None,
        )
    if extra_image_df is not None and not extra_image_df.empty:
        x_extra = _fit_quality_value(extra_image_df, "x_model_arcsec")
        y_extra = _fit_quality_value(extra_image_df, "y_model_arcsec")
        finite_extra = np.isfinite(x_extra) & np.isfinite(y_extra)
        if finite_extra.any():
            ax.scatter(
                x_extra[finite_extra],
                y_extra[finite_extra],
                marker="o",
                color="tab:blue",
                s=16,
                linewidths=0.0,
                label="extra",
            )
    for row, x_fit, y_fit, is_recovered in zip(image_df.itertuples(index=False), x_model, y_model, recovered):
        if is_recovered and np.isfinite(x_fit) and np.isfinite(y_fit):
            ax.plot([row.x_obs_arcsec, x_fit], [row.y_obs_arcsec, y_fit], color="0.6", lw=0.7)
    ax.invert_xaxis()
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_title("Observed vs Model Image Positions")
    ax.legend(loc="best", fontsize=8)

    residual_best = image_df["image_residual_arcsec"].to_numpy(dtype=float)
    residual = image_df["image_residual_q50"].to_numpy(dtype=float)
    residual = np.where(np.isfinite(residual), residual, residual_best)
    x_index = np.arange(len(image_df))
    r16 = image_df["image_residual_q16"].to_numpy(dtype=float)
    r84 = image_df["image_residual_q84"].to_numpy(dtype=float)
    finite_residual = np.isfinite(residual)
    if finite_residual.any():
        yerr = [
            np.where(np.isfinite(r16), np.maximum(0.0, residual - r16), 0.0)[finite_residual],
            np.where(np.isfinite(r84), np.maximum(0.0, r84 - residual), 0.0)[finite_residual],
        ]
        axes[1].errorbar(
            x_index[finite_residual],
            residual[finite_residual],
            yerr=yerr,
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            markersize=3,
        )
    arc_residual_best = _fit_quality_value(image_df, "arc_aware_image_residual_arcsec")
    arc_residual = _fit_quality_value(image_df, "arc_aware_image_residual_q50", "arc_aware_image_residual_arcsec")
    arc_residual = np.where(np.isfinite(arc_residual), arc_residual, arc_residual_best)
    finite_arc_residual = np.isfinite(arc_residual)
    if finite_arc_residual.any():
        axes[1].scatter(
            x_index[finite_arc_residual],
            arc_residual[finite_arc_residual],
            color="tab:olive",
            marker="x",
            s=24,
            label="arc-aware",
        )
    axes[1].set_xlabel("image index")
    axes[1].set_ylabel("image residual [arcsec]")
    axes[1].set_title("Image Residuals")
    if finite_arc_residual.any():
        axes[1].legend(loc="best", fontsize=8)
    if len(image_df) <= 40:
        axes[1].set_xticks(x_index)
        axes[1].set_xticklabels(image_df["image_label"].astype(str), rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_image_count_recovery(image_count_df: pd.DataFrame, path: Path) -> None:
    if image_count_df.empty:
        return
    required = {"family_id", "observed_image_count", "recovered_image_count", "produced_image_count"}
    if not required.issubset(image_count_df.columns):
        return
    observed = pd.to_numeric(image_count_df["observed_image_count"], errors="coerce").to_numpy(dtype=float)
    recovered = pd.to_numeric(image_count_df["recovered_image_count"], errors="coerce").to_numpy(dtype=float)
    produced = pd.to_numeric(image_count_df["produced_image_count"], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(recovered).any() or np.isfinite(produced).any()):
        return
    labels = image_count_df["family_id"].astype(str).to_numpy()
    y_index = np.arange(len(image_count_df))
    height = 0.24
    fig, ax = plt.subplots(figsize=(9.5, max(4.2, 0.32 * len(image_count_df))))
    finite_observed = np.isfinite(observed)
    finite_recovered = np.isfinite(recovered)
    finite_produced = np.isfinite(produced)
    if finite_observed.any():
        ax.barh(y_index[finite_observed] - height, observed[finite_observed], height=height, color="0.65", label="observed")
    if finite_recovered.any():
        ax.barh(y_index[finite_recovered], recovered[finite_recovered], height=height, color="tab:green", label="recovered")
    if finite_produced.any():
        ax.barh(y_index[finite_produced] + height, produced[finite_produced], height=height, color="tab:blue", label="produced")
    total_observed = int(np.nansum(observed))
    total_recovered = int(np.nansum(recovered)) if finite_recovered.any() else 0
    total_produced = int(np.nansum(produced)) if finite_produced.any() else 0
    ax.set_yticks(y_index)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("image count")
    ax.set_ylabel("family")
    ax.set_title(
        f"Image Count Recovery: observed={total_observed} recovered={total_recovered} produced={total_produced}"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_model_magnification_fit_quality(magnification_df: pd.DataFrame, path: Path) -> None:
    if magnification_df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.22 * len(magnification_df)), 4.8))
    x_index = np.arange(len(magnification_df))
    best = magnification_df["magnification_model"].to_numpy(dtype=float)
    median = magnification_df["magnification_model_q50"].to_numpy(dtype=float)
    median = np.where(np.isfinite(median), median, best)
    q16 = magnification_df["magnification_model_q16"].to_numpy(dtype=float)
    q84 = magnification_df["magnification_model_q84"].to_numpy(dtype=float)
    finite = np.isfinite(median)
    if finite.any():
        yerr = [
            np.where(np.isfinite(q16), np.maximum(0.0, median - q16), 0.0)[finite],
            np.where(np.isfinite(q84), np.maximum(0.0, q84 - median), 0.0)[finite],
        ]
        ax.errorbar(
            x_index[finite],
            median[finite],
            yerr=yerr,
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            markersize=4,
            label="posterior interval",
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("image")
    ax.set_ylabel("model signed magnification")
    ax.set_title("Model-Only Signed Magnification at Observed Images")
    if len(magnification_df) <= 80:
        ax.set_xticks(x_index)
        ax.set_xticklabels(magnification_df["image_label"].astype(str), rotation=90, fontsize=7)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_normalized_image_residuals(image_df: pd.DataFrame, path: Path) -> None:
    if image_df.empty:
        return
    residual_norm = _fit_quality_value(image_df, "residual_norm_q50", "residual_norm")
    finite = np.isfinite(residual_norm)
    if not finite.any():
        return
    labels = image_df["image_label"].astype(str).to_numpy()
    family_ids = image_df["family_id"].astype(str).to_numpy() if "family_id" in image_df.columns else np.full(len(image_df), "")
    unique_families = list(dict.fromkeys(family_ids.tolist()))
    color_map = {family_id: plt.get_cmap("tab20", max(1, len(unique_families)))(idx) for idx, family_id in enumerate(unique_families)}
    colors = [color_map[family_id] for family_id in family_ids]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = min(60, max(12, int(np.sqrt(int(np.sum(finite))) * 2)))
    axes[0].hist(residual_norm[finite], bins=bins, color="tab:blue", alpha=0.75)
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("image residual / sigma_eff")
    axes[0].set_ylabel("count")
    axes[0].set_title("Normalized Residual Distribution")

    x_index = np.arange(len(image_df))
    axes[1].scatter(x_index[finite], residual_norm[finite], c=np.asarray(colors, dtype=object)[finite].tolist(), s=28)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("image")
    axes[1].set_ylabel("image residual / sigma_eff")
    axes[1].set_title("Normalized Residual by Image")
    if len(image_df) <= 80:
        axes[1].set_xticks(x_index)
        axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_placeholder_plot(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_image_residual_histogram(image_df: pd.DataFrame, path: Path) -> None:
    residual = _fit_quality_value(image_df, "image_residual_q50", "image_residual_arcsec")
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        _write_placeholder_plot(
            path,
            "Image residual histogram",
            "No finite image residuals are available.",
        )
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bin_count = 30
    total_rms = float(np.sqrt(np.mean(np.square(residual))))
    ax.hist(residual, bins=bin_count, color="tab:blue", alpha=0.75)
    ax.axvline(
        float(np.nanmedian(residual)),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="median",
    )
    ax.axvline(
        total_rms,
        color="tab:red",
        linestyle="-.",
        linewidth=1.2,
        label="total RMS",
    )
    rms_annotation = (
        f"Total RMS = $\\sqrt{{\\mathrm{{mean}}(r^2)}}$ = {total_rms:.3g} arcsec\n"
        f"N = {residual.size}"
    )
    ax.text(
        0.98,
        0.95,
        rms_annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.9,
        },
    )
    ax.set_xlabel("image residual [arcsec]")
    ax.set_ylabel("N images")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _finite_plot_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _critical_arc_support_available(image_df: pd.DataFrame | None) -> bool:
    if image_df is None or image_df.empty:
        return False
    for column in (
        "arc_s_min",
        "arc_noncritical_direction_residual_arcsec",
        "arc_aware_image_residual_arcsec",
        "arc_aware_image_residual_q50",
        "arc_prior_probability",
    ):
        if column in image_df.columns and np.isfinite(pd.to_numeric(image_df[column], errors="coerce").to_numpy(dtype=float)).any():
            return True
    return False


def _critical_arc_status_counts(image_df: pd.DataFrame) -> dict[str, int]:
    if "arc_recovery_status" in image_df.columns:
        status = image_df["arc_recovery_status"].fillna("not_recovered").astype(str).to_numpy()
    elif "arc_supported" in image_df.columns:
        supported = image_df["arc_supported"].astype(bool).to_numpy()
        status = np.where(supported, "arc_supported", "not_recovered")
    else:
        status = np.full(len(image_df), "not_recovered", dtype=object)
    return {
        "point_recovered": int(np.sum(status == "point_recovered")),
        "arc_supported": int(np.sum(status == "arc_supported")),
        "not_recovered": int(np.sum(status == "not_recovered")),
    }


def _histogram_bins(values: np.ndarray) -> int:
    count = int(np.sum(np.isfinite(values)))
    return min(40, max(8, int(np.sqrt(max(count, 1)) * 2)))


def _plot_critical_arc_support_histogram(
    image_df: pd.DataFrame,
    path: Path,
    *,
    curve_support_radius_arcsec: float = CRITICAL_ARC_CURVE_SUPPORT_RADIUS_ARCSEC,
    singular_threshold: float = CRITICAL_ARC_SINGULAR_THRESHOLD,
) -> None:
    if not _critical_arc_support_available(image_df):
        _write_placeholder_plot(
            path,
            "Critical-arc support histogram",
            "No finite critical-arc support diagnostics are available.",
        )
        return

    strict_residual = _finite_plot_values(_fit_quality_value(image_df, "image_residual_q50", "image_residual_arcsec"))
    arc_residual = _finite_plot_values(
        _fit_quality_value(image_df, "arc_aware_image_residual_q50", "arc_aware_image_residual_arcsec")
    )
    curve_distance = _finite_plot_values(
        _fit_quality_value(image_df, "arc_curve_distance_arcsec", "arc_noncritical_direction_residual_arcsec")
    )
    critical_direction_residual = _finite_plot_values(_fit_quality_value(image_df, "arc_critical_direction_residual_arcsec"))
    s_min = _finite_plot_values(_fit_quality_value(image_df, "arc_s_min"))
    arc_prior = _finite_plot_values(_fit_quality_value(image_df, "arc_prior_probability"))
    counts = _critical_arc_status_counts(image_df)
    strict_rms = float(np.sqrt(np.mean(np.square(strict_residual)))) if strict_residual.size else np.nan
    arc_rms = float(np.sqrt(np.mean(np.square(arc_residual)))) if arc_residual.size else np.nan

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    residual_ax, noncritical_ax, critical_direction_ax, singular_ax = axes.ravel()
    if strict_residual.size:
        residual_ax.hist(strict_residual, bins=_histogram_bins(strict_residual), color="tab:blue", alpha=0.52, label="strict")
    if arc_residual.size:
        residual_ax.hist(arc_residual, bins=_histogram_bins(arc_residual), color="tab:olive", alpha=0.48, label="arc-aware")
    residual_ax.set_xlabel("image residual [arcsec]")
    residual_ax.set_ylabel("N images")
    residual_ax.set_title("Strict vs Arc-Aware Residual")
    residual_ax.legend(loc="best", fontsize=8)
    residual_ax.text(
        0.98,
        0.95,
        (
            f"strict RMS={strict_rms:.3g}, N={strict_residual.size}\n"
            f"arc RMS={arc_rms:.3g}, N={arc_residual.size}\n"
            f"point={counts['point_recovered']} arc={counts['arc_supported']} missing={counts['not_recovered']}"
        ),
        transform=residual_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.9},
    )

    if curve_distance.size:
        noncritical_ax.hist(curve_distance, bins=_histogram_bins(curve_distance), color="tab:green", alpha=0.75)
    noncritical_ax.axvline(float(curve_support_radius_arcsec), color="black", linestyle="--", linewidth=1.1, label="curve support radius")
    noncritical_ax.set_xlabel("support-curve distance [arcsec]")
    noncritical_ax.set_ylabel("N images")
    noncritical_ax.set_title("Critical-Arc Support-Curve Distance")
    noncritical_ax.legend(loc="best", fontsize=8)

    if critical_direction_residual.size:
        critical_direction_log = np.log10(np.maximum(critical_direction_residual, 1.0e-6))
        critical_direction_ax.hist(critical_direction_log, bins=_histogram_bins(critical_direction_log), color="tab:purple", alpha=0.72)
    critical_direction_ax.set_xlabel("log10 critical-direction residual [arcsec]")
    critical_direction_ax.set_ylabel("N images")
    critical_direction_ax.set_title("Critical-Direction Residual")

    if s_min.size:
        singular_ax.hist(s_min, bins=_histogram_bins(s_min), color="tab:orange", alpha=0.72)
        singular_ax.axvline(float(singular_threshold), color="black", linestyle="--", linewidth=1.1, label="singular threshold")
        singular_ax.set_xlabel("smallest singular value")
    elif arc_prior.size:
        singular_ax.hist(arc_prior, bins=_histogram_bins(arc_prior), color="tab:orange", alpha=0.72)
        singular_ax.set_xlabel("arc prior probability")
    singular_ax.set_ylabel("N images")
    singular_ax.set_title("Local Criticality")
    if s_min.size:
        singular_ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_critical_arc_support_phase_space(
    image_df: pd.DataFrame,
    path: Path,
    *,
    curve_support_radius_arcsec: float = CRITICAL_ARC_CURVE_SUPPORT_RADIUS_ARCSEC,
    singular_threshold: float = CRITICAL_ARC_SINGULAR_THRESHOLD,
) -> None:
    has_curve_distance = "arc_curve_distance_arcsec" in image_df.columns
    has_legacy_distance = "arc_noncritical_direction_residual_arcsec" in image_df.columns
    if image_df.empty or "arc_s_min" not in image_df.columns or not (has_curve_distance or has_legacy_distance):
        _write_placeholder_plot(
            path,
            "Critical-arc support phase space",
            "No finite critical-arc phase-space diagnostics are available.",
        )
        return
    s_min = _fit_quality_value(image_df, "arc_s_min")
    support_curve_distance = _fit_quality_value(image_df, "arc_curve_distance_arcsec", "arc_noncritical_direction_residual_arcsec")
    critical_direction = _fit_quality_value(image_df, "arc_critical_direction_residual_arcsec")
    finite = np.isfinite(s_min) & np.isfinite(support_curve_distance)
    if not finite.any():
        _write_placeholder_plot(
            path,
            "Critical-arc support phase space",
            "No finite critical-arc phase-space diagnostics are available.",
        )
        return
    status = (
        image_df["arc_recovery_status"].fillna("not_recovered").astype(str).to_numpy()
        if "arc_recovery_status" in image_df.columns
        else np.full(len(image_df), "not_recovered", dtype=object)
    )
    colors = {
        "point_recovered": "tab:blue",
        "arc_supported": "tab:green",
        "not_recovered": "tab:red",
    }
    finite_critical_direction = np.where(np.isfinite(critical_direction), critical_direction, 0.0)
    critical_direction_scale = np.log10(np.maximum(finite_critical_direction, 1.0e-6))
    critical_direction_scale = critical_direction_scale - np.nanmin(critical_direction_scale[finite]) if np.any(finite) else critical_direction_scale
    max_scale = np.nanmax(critical_direction_scale[finite]) if np.any(np.isfinite(critical_direction_scale[finite])) else 0.0
    sizes = 24.0 + 55.0 * (critical_direction_scale / max(max_scale, 1.0e-12))
    sizes = np.clip(sizes, 24.0, 90.0)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for status_name, color in colors.items():
        mask = finite & (status == status_name)
        if mask.any():
            ax.scatter(
                s_min[mask],
                support_curve_distance[mask],
                s=sizes[mask],
                color=color,
                alpha=0.72,
                edgecolors="white",
                linewidths=0.35,
                label=status_name.replace("_", " "),
            )
    other_mask = finite & ~np.isin(status, list(colors))
    if other_mask.any():
        ax.scatter(s_min[other_mask], support_curve_distance[other_mask], s=sizes[other_mask], color="0.4", alpha=0.65, label="other")
    ax.axvline(float(singular_threshold), color="black", linestyle="--", linewidth=1.0, label="singular threshold")
    ax.axhline(float(curve_support_radius_arcsec), color="0.25", linestyle=":", linewidth=1.2, label="curve support radius")
    ax.set_xlabel("smallest singular value")
    ax.set_ylabel("support-curve distance [arcsec]")
    ax.set_title("Critical-Arc Support Phase Space")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_critical_arc_recovery_by_family(image_count_df: pd.DataFrame, path: Path) -> None:
    required = {
        "family_id",
        "observed_image_count",
        "recovered_image_count",
        "arc_aware_recovered_image_count",
        "arc_supported_image_count",
    }
    if image_count_df is None or image_count_df.empty or not required.issubset(image_count_df.columns):
        _write_placeholder_plot(
            path,
            "Critical-arc recovery by family",
            "No family-level critical-arc recovery counts are available.",
        )
        return
    df = image_count_df.copy()
    for column in [
        "observed_image_count",
        "recovered_image_count",
        "missing_image_count",
        "arc_aware_recovered_image_count",
        "arc_aware_missing_image_count",
        "arc_supported_image_count",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if not np.isfinite(df["arc_aware_recovered_image_count"].to_numpy(dtype=float)).any():
        _write_placeholder_plot(
            path,
            "Critical-arc recovery by family",
            "No finite family-level critical-arc recovery counts are available.",
        )
        return
    df["_arc_missing_sort"] = np.nan_to_num(df.get("arc_aware_missing_image_count", np.nan), nan=-1.0)
    df["_strict_missing_sort"] = np.nan_to_num(df.get("missing_image_count", np.nan), nan=-1.0)
    df = df.sort_values(["_arc_missing_sort", "_strict_missing_sort", "family_id"], ascending=[False, False, True])
    labels = df["family_id"].astype(str).to_numpy()
    y = np.arange(len(df))
    height = 0.18
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.36 * len(df))))
    series = [
        ("observed", "observed_image_count", "0.65", -1.5 * height),
        ("strict recovered", "recovered_image_count", "tab:blue", -0.5 * height),
        ("arc-aware recovered", "arc_aware_recovered_image_count", "tab:green", 0.5 * height),
        ("arc supported", "arc_supported_image_count", "tab:olive", 1.5 * height),
    ]
    for label, column, color, offset in series:
        values = df[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.any():
            ax.barh(y[finite] + offset, values[finite], height=height, color=color, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("image count")
    ax.set_ylabel("family")
    ax.set_title("Critical-Arc Recovery by Family")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_vs_magnification(image_df: pd.DataFrame, magnification_df: pd.DataFrame, path: Path) -> None:
    merged = _merge_fit_quality_with_magnification(image_df, magnification_df)
    if merged.empty:
        return
    abs_mu = np.abs(_fit_quality_value(merged, "magnification_model_q50", "magnification_model"))
    residual = _fit_quality_value(merged, "image_residual_q50", "image_residual_arcsec")
    residual_norm = _fit_quality_value(merged, "residual_norm_q50", "residual_norm")
    finite_any = np.isfinite(abs_mu) & (np.isfinite(residual) | np.isfinite(residual_norm))
    if not finite_any.any():
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    finite_residual = np.isfinite(abs_mu) & np.isfinite(residual)
    axes[0].scatter(abs_mu[finite_residual], residual[finite_residual], color="tab:blue", s=26)
    axes[0].set_ylabel("image residual [arcsec]")
    axes[0].set_title("Residual vs Magnification")

    finite_norm = np.isfinite(abs_mu) & np.isfinite(residual_norm)
    axes[1].scatter(abs_mu[finite_norm], residual_norm[finite_norm], color="tab:purple", s=26)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("image residual / sigma_eff")
    axes[1].set_title("Normalized Residual vs Magnification")
    for ax in axes:
        ax.set_xlabel("|model signed magnification|")
        positive = abs_mu[np.isfinite(abs_mu) & (abs_mu > 0.0)]
        if positive.size and float(np.nanmax(positive)) / max(float(np.nanmin(positive)), 1.0e-12) > 20.0:
            ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_geometry_trends(image_df: pd.DataFrame, path: Path) -> None:
    if image_df.empty:
        return
    residual = _fit_quality_value(image_df, "image_residual_q50", "image_residual_arcsec")
    x_columns = [
        ("radius_arcsec", "radius [arcsec]", "Residual vs Radius"),
        ("angle_deg", "angle [deg]", "Residual vs Angle"),
        ("z_source", "source redshift", "Residual vs Redshift"),
    ]
    if not any(column in image_df.columns for column, _xlabel, _title in x_columns):
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    plotted = False
    for ax, (column, xlabel, title) in zip(axes, x_columns):
        x_values = _fit_quality_value(image_df, column)
        finite = np.isfinite(x_values) & np.isfinite(residual)
        if finite.any():
            ax.scatter(x_values[finite], residual[finite], color="tab:blue", s=24)
            plotted = True
        ax.set_xlabel(xlabel)
        ax.set_ylabel("image residual [arcsec]")
        ax.set_title(title)
    if not plotted:
        plt.close(fig)
        return
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_posterior_predictive_coverage(image_df: pd.DataFrame, path: Path) -> None:
    if image_df.empty or not {"covered_x_1sigma", "covered_y_1sigma", "covered_xy_1sigma"}.issubset(image_df.columns):
        return
    labels = image_df["image_label"].astype(str).to_numpy()
    coverage = np.vstack(
        [
            image_df["covered_x_1sigma"].astype(bool).to_numpy(),
            image_df["covered_y_1sigma"].astype(bool).to_numpy(),
            image_df["covered_xy_1sigma"].astype(bool).to_numpy(),
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1.0]})
    axes[0].imshow(coverage.astype(float), aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(["x", "y", "x and y"])
    axes[0].set_xlabel("image")
    axes[0].set_title("1 Sigma Predictive Coverage")
    if len(image_df) <= 80:
        x_index = np.arange(len(image_df))
        axes[0].set_xticks(x_index)
        axes[0].set_xticklabels(labels, rotation=90, fontsize=7)

    fractions = coverage.mean(axis=1) if coverage.shape[1] else np.zeros(3, dtype=float)
    axes[1].bar(["x", "y", "x and y"], fractions, color=["tab:blue", "tab:orange", "tab:green"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("covered fraction")
    axes[1].set_title("Coverage Summary")
    for index, value in enumerate(fractions):
        axes[1].text(index, min(0.98, float(value) + 0.03), f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_exact_vs_approx_prediction_error(family_df: pd.DataFrame, path: Path) -> None:
    if family_df.empty or not {"exact_image_rms_arcsec", "approx_image_rms_arcsec"}.issubset(family_df.columns):
        return
    exact = _fit_quality_value(family_df, "exact_image_rms_arcsec")
    approx = _fit_quality_value(family_df, "approx_image_rms_arcsec")
    finite = np.isfinite(exact) & np.isfinite(approx)
    if not finite.any():
        return
    labels = family_df["family_id"].astype(str).to_numpy() if "family_id" in family_df.columns else np.arange(len(family_df)).astype(str)
    exact = exact[finite]
    approx = approx[finite]
    labels = labels[finite]
    diff = exact - approx
    ratio = exact / np.where(np.abs(approx) > 1.0e-12, approx, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].scatter(approx, exact, color="tab:blue", s=30)
    finite_pair = np.concatenate([approx[np.isfinite(approx)], exact[np.isfinite(exact)]])
    if finite_pair.size:
        lo = float(np.nanmin(finite_pair))
        hi = float(np.nanmax(finite_pair))
        axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    axes[0].set_xlabel("approx image RMS [arcsec]")
    axes[0].set_ylabel("exact image RMS [arcsec]")
    axes[0].set_title("Exact vs Approx RMS")

    order = np.argsort(diff)
    axes[1].bar(np.arange(len(diff)), diff[order], color="tab:orange")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_ylabel("exact - approx [arcsec]")
    axes[1].set_title("RMS Difference")

    axes[2].bar(np.arange(len(ratio)), ratio[order], color="tab:purple")
    axes[2].axhline(1.0, color="black", linewidth=1.0)
    axes[2].set_ylabel("exact / approx")
    axes[2].set_title("RMS Ratio")
    if len(labels) <= 40:
        for ax in axes[1:]:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels[order], rotation=90, fontsize=7)
            ax.set_xlabel("family")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_source_plane_scatter(plot_dir: Path, state: BuildState, best_eval: EvaluationResult) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cmap = plt.get_cmap("tab20", len(state.family_data))
    for idx, family in enumerate(state.family_data):
        color = cmap(idx)
        pred = best_eval.family_predictions[family.family_id]
        ax.scatter(pred["source_x"], pred["source_y"], color=color, s=40, label=family.family_id)
    ax.set_xlabel(r"$\beta_x$ [arcsec]")
    ax.set_ylabel(r"$\beta_y$ [arcsec]")
    ax.set_title("Back-Projected Source Positions")
    if len(state.family_data) <= 20:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "source_plane_scatter.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_per_potential_summary(
    plot_dir: Path,
    summary_df: pd.DataFrame,
    best_par_marker_values: dict[str, float] | None = None,
    previous_stage_best_values: dict[str, float] | None = None,
    parameter_specs: list[ParameterSpec] | None = None,
) -> None:
    if summary_df.empty:
        return
    previous_stage_by_label: dict[str, float] = {}
    if previous_stage_best_values and parameter_specs:
        for spec in parameter_specs:
            values = _corner_values_for_specs([spec], previous_stage_best_values)
            if values and np.isfinite(values[0]):
                previous_stage_by_label[str(spec.name)] = float(values[0])
    nrows = len(summary_df)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, max(4, 1.4 * nrows)), sharex=False)
    if nrows == 1:
        axes = [axes]
    for ax, row in zip(axes, summary_df.itertuples(index=False)):
        best_par_value = None
        if best_par_marker_values:
            best_par_value = _finite_float_or_none(best_par_marker_values.get(str(row.label)))
        previous_stage_value = previous_stage_by_label.get(str(row.label))
        ax.hlines(1, row.p16, row.p84, linewidth=4, color="tab:blue")
        ax.scatter([row.median], [1], color="tab:blue", s=35, label="median")
        ax.scatter([row.map], [1], color=CORNER_BEST_FIT_COLOR, marker="x", s=30, label="best fit")
        if best_par_value is not None:
            ax.scatter([best_par_value], [1], color=CORNER_BEST_PAR_COLOR, marker="x", s=30, label="best.par")
        if previous_stage_value is not None:
            ax.scatter(
                [previous_stage_value],
                [1],
                color=CORNER_PREVIOUS_STAGE_COLOR,
                marker="x",
                s=30,
                label="previous stage",
            )
        if np.isfinite(row.lower) and np.isfinite(row.upper):
            x_min = float(row.lower)
            x_max = float(row.upper)
        else:
            width = max(float(row.std), 0.5 * abs(float(row.p84) - float(row.p16)), 1.0e-3)
            x_min = float(min(row.p16, row.median, row.map) - 2.0 * width)
            x_max = float(max(row.p84, row.median, row.map) + 2.0 * width)
        for marker_value in (best_par_value, previous_stage_value):
            if marker_value is None:
                continue
            span = max(abs(x_max - x_min), 1.0e-3)
            pad = 0.05 * span
            x_min = min(x_min, float(marker_value) - pad)
            x_max = max(x_max, float(marker_value) + pad)
        ax.set_xlim(x_min, x_max)
        ax.set_yticks([])
        ax.set_title(row.label)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "per_potential_summary.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_refresh_diagnostics(plot_dir: Path, family_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    families = family_df["family_id"].astype(str)
    axes[0].bar(families, family_df["exact_validation_count"], color="tab:blue")
    axes[0].set_ylabel("Exact validations")
    axes[1].bar(families, family_df["source_plane_rms_arcsec"], color="tab:orange")
    axes[1].set_ylabel("Source RMS")
    mismatch = family_df["multiplicity_mismatch_count"] + family_df["match_failure_count"]
    axes[2].bar(families, mismatch, color="tab:red")
    axes[2].set_ylabel("Match failures")
    axes[2].set_xlabel("Family")
    axes[2].tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "refresh_diagnostics.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_timing_profile(plot_dir: Path, evaluator: ClusterJAXEvaluator) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    names = list(evaluator.timing_totals.keys())
    values = [evaluator.timing_totals[name] for name in names]
    ax.bar(names, values, color="tab:cyan")
    ax.set_ylabel("seconds")
    ax.set_title("Timing Totals")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "timing_profile.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _critical_curve_caustics(
    lens_model: Any,
    kwargs_lens: list[dict[str, float]],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    include_tangential: bool = True,
    include_radial: bool = False,
) -> list[dict[str, np.ndarray]]:
    x_values = np.asarray(x_axis, dtype=float)
    y_values = np.asarray(y_axis, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size < 2 or y_values.size < 2:
        return []
    xx, yy = np.meshgrid(x_values, y_values)
    f_xx, f_xy, f_yx, f_yy = lens_model.hessian(
        xx.ravel(),
        yy.ravel(),
        kwargs_lens,
    )
    shape = xx.shape
    f_xx = np.asarray(f_xx, dtype=float).reshape(shape)
    f_yy = np.asarray(f_yy, dtype=float).reshape(shape)
    f_xy = np.asarray(f_xy, dtype=float).reshape(shape)
    f_yx = np.asarray(f_yx, dtype=float).reshape(shape)
    kappa = 0.5 * (f_xx + f_yy)
    gamma1 = 0.5 * (f_xx - f_yy)
    gamma2 = 0.5 * (f_xy + f_yx)
    contours: list[dict[str, np.ndarray]] = []
    pixel_x = np.arange(x_values.size, dtype=float)
    pixel_y = np.arange(y_values.size, dtype=float)
    contour_fields: list[tuple[str, np.ndarray]] = []
    shear = np.hypot(gamma1, gamma2)
    if include_tangential:
        contour_fields.append(("tangential", 1.0 - kappa - shear))
    if include_radial:
        contour_fields.append(("radial", 1.0 - kappa + shear))
    for kind, field in contour_fields:
        for vertices in find_contours(field, 0.0):
            vertices = np.asarray(vertices, dtype=float)
            if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
                continue
            crit_x = np.interp(vertices[:, 1], pixel_x, x_values)
            crit_y = np.interp(vertices[:, 0], pixel_y, y_values)
            if not np.all(np.isfinite(crit_x)) or not np.all(np.isfinite(crit_y)):
                continue
            beta_x, beta_y = lens_model.ray_shooting(crit_x, crit_y, kwargs_lens)
            beta_x = np.asarray(beta_x, dtype=float)
            beta_y = np.asarray(beta_y, dtype=float)
            if beta_x.shape != crit_x.shape or beta_y.shape != crit_y.shape:
                continue
            if not np.all(np.isfinite(beta_x)) or not np.all(np.isfinite(beta_y)):
                continue
            contours.append(
                {
                    "kind": kind,
                    "critical_x": crit_x,
                    "critical_y": crit_y,
                    "caustic_x": beta_x,
                    "caustic_y": beta_y,
                }
            )
    return contours


def _tangential_critical_curve_caustics(
    lens_model: Any,
    kwargs_lens: list[dict[str, float]],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    return _critical_curve_caustics(
        lens_model,
        kwargs_lens,
        x_axis,
        y_axis,
        include_tangential=True,
        include_radial=False,
    )


def _plot_caustic_overlay(
    plot_dir: Path,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    caustic_plot_grid_scale_arcsec: float,
    caustic_source_redshift: float,
) -> None:
    z_source = float(caustic_source_redshift)
    z_lens = getattr(evaluator.state, "z_lens", None)
    if z_lens is not None and np.isfinite(float(z_lens)) and z_source <= float(z_lens):
        _log(
            None,
            f"[plot:caustic_overlay] skipped: caustic source redshift z={z_source:g} "
            f"is not behind lens redshift z={float(z_lens):g}",
        )
        return
    best_fit_latent = _reported_physical_to_latent_vector(evaluator, np.asarray(best_fit, dtype=float))
    x_grid, y_grid = _caustic_plot_grid_axes(caustic_plot_grid_scale_arcsec)
    exact_models_by_z = getattr(evaluator, "exact_models_by_z", {})
    model = exact_models_by_z.get(z_source) if exact_models_by_z is not None else None
    if model is None:
        model, _ = evaluator._get_exact_model_solver(z_source)
    packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit_latent, dtype=jnp.float64), z_source)
    kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
    contours = _tangential_critical_curve_caustics(model, kwargs_lens, x_grid, y_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    image_ax, source_ax = axes
    for contour in contours:
        image_ax.plot(contour["critical_x"], contour["critical_y"], color="black", linewidth=1.0)
    cmap = plt.get_cmap("tab20", len(evaluator.state.family_data))
    for idx, fam in enumerate(evaluator.state.family_data):
        image_ax.scatter(fam.x_obs, fam.y_obs, color=cmap(idx), s=14, label=fam.family_id)
    image_ax.invert_xaxis()
    image_ax.set_xlabel("x [arcsec]")
    image_ax.set_ylabel("y [arcsec]")
    image_ax.set_title(f"Tangential Critical Lines + Images (z={z_source:g})")

    for contour in contours:
        source_ax.scatter(
            contour["caustic_x"],
            contour["caustic_y"],
            color="black",
            s=2,
            alpha=0.75,
            linewidths=0.0,
        )
    best_source_eval = evaluator.evaluate(best_fit_latent, validate_all_families=False)
    for idx, fam in enumerate(evaluator.state.family_data):
        pred = best_source_eval.family_predictions[fam.family_id]
        source_ax.scatter(pred["source_x"], pred["source_y"], color=cmap(idx), s=14)
    source_ax.set_xlabel(r"$\beta_x$ [arcsec]")
    source_ax.set_ylabel(r"$\beta_y$ [arcsec]")
    source_ax.set_title("Caustics + Source Centroids")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "caustic_overlay.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_absolute_magnification(
    plot_dir: Path,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    caustic_plot_grid_scale_arcsec: float,
    caustic_source_redshift: float,
    *,
    cap: float = ABSOLUTE_MAGNIFICATION_PLOT_CAP,
) -> None:
    z_source = float(caustic_source_redshift)
    z_lens = getattr(evaluator.state, "z_lens", None)
    if z_lens is not None and np.isfinite(float(z_lens)) and z_source <= float(z_lens):
        _log(
            None,
            f"[plot:absolute_magnification] skipped: caustic source redshift z={z_source:g} "
            f"is not behind lens redshift z={float(z_lens):g}",
        )
        return
    if float(cap) <= 0.0:
        raise ValueError("cap must be positive.")

    best_fit_latent = _reported_physical_to_latent_vector(evaluator, np.asarray(best_fit, dtype=float))
    x_grid, y_grid = _caustic_plot_grid_axes(caustic_plot_grid_scale_arcsec)
    xx, yy = np.meshgrid(x_grid, y_grid)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)

    exact_models_by_z = getattr(evaluator, "exact_models_by_z", {})
    model = exact_models_by_z.get(z_source) if exact_models_by_z is not None else None
    if model is None:
        model, _ = evaluator._get_exact_model_solver(z_source)
    packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit_latent, dtype=jnp.float64), z_source)
    kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
    mu = np.asarray(model.magnification(flat_x, flat_y, kwargs_lens), dtype=float).reshape(xx.shape)
    abs_mu = np.minimum(np.abs(mu), float(cap))

    extent = [float(x_grid[0]), float(x_grid[-1]), float(y_grid[0]), float(y_grid[-1])]
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    image = ax.imshow(
        np.ma.masked_invalid(abs_mu),
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=float(cap),
        aspect="equal",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(r"$|\mu|$")
    ax.invert_xaxis()
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_title(f"Absolute Magnification (z={z_source:g})")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "absolute_magnification.pdf"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_kappa_true_fits(path: str | Path) -> tuple[np.ndarray, WCS]:
    fits_path = Path(path)
    with fits.open(fits_path, memmap=True) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            image = np.squeeze(np.asarray(data, dtype=float))
            if image.ndim != 2:
                continue
            wcs = WCS(hdu.header).celestial
            if not wcs.has_celestial:
                continue
            return image, wcs
    raise ValueError(f"No 2D celestial WCS image found in {fits_path}")


def _radec_to_solver_arcsec_offsets(
    ra_deg: Any,
    dec_deg: Any,
    reference: tuple[int, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        _reference_type, ra0_deg, dec0_deg = reference
        ra0 = float(ra0_deg)
        dec0 = float(dec0_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError("State reference must contain reference type, RA, and Dec.") from exc
    cos_dec0 = math.cos(math.radians(dec0))
    if cos_dec0 == 0.0:
        raise ValueError("Reference declination is too close to a pole for offset conversion.")
    ra_values = np.asarray(ra_deg, dtype=float)
    dec_values = np.asarray(dec_deg, dtype=float)
    delta_ra = (ra0 - ra_values + 180.0) % 360.0 - 180.0
    x_arcsec = delta_ra * cos_dec0 * 3600.0
    y_arcsec = (dec_values - dec0) * 3600.0
    return x_arcsec, y_arcsec


def _kappa_model_grid_for_true_fits(
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    kappa_true_shape: tuple[int, int],
    kappa_wcs: WCS,
    caustic_source_redshift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = kappa_true_shape
    y_pixels, x_pixels = np.indices((height, width), dtype=float)
    ra_deg, dec_deg = kappa_wcs.pixel_to_world_values(x_pixels, y_pixels)
    x_arcsec, y_arcsec = _radec_to_solver_arcsec_offsets(ra_deg, dec_deg, evaluator.state.reference)
    z_source = float(caustic_source_redshift)
    best_fit_latent = _reported_physical_to_latent_vector(evaluator, np.asarray(best_fit, dtype=float))
    exact_models_by_z = getattr(evaluator, "exact_models_by_z", {})
    model = exact_models_by_z.get(z_source) if exact_models_by_z is not None else None
    if model is None:
        model, _ = evaluator._get_exact_model_solver(z_source)
    packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit_latent, dtype=jnp.float64), z_source)
    kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
    flat_x = x_arcsec.reshape(-1)
    flat_y = y_arcsec.reshape(-1)
    flat_kappa = np.full(flat_x.shape, np.nan, dtype=float)
    chunk_size = 250_000
    for start in range(0, flat_x.size, chunk_size):
        stop = min(start + chunk_size, flat_x.size)
        finite = np.isfinite(flat_x[start:stop]) & np.isfinite(flat_y[start:stop])
        if not np.any(finite):
            continue
        chunk_values = np.full(stop - start, np.nan, dtype=float)
        chunk_values[finite] = np.asarray(
            model.kappa(
                flat_x[start:stop][finite],
                flat_y[start:stop][finite],
                kwargs_lens,
            ),
            dtype=float,
        )
        flat_kappa[start:stop] = chunk_values
    return flat_kappa.reshape(kappa_true_shape), x_arcsec, y_arcsec


def _plot_kappa_true_comparison(
    plot_dir: Path,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    kappa_true_fits: str | Path,
    caustic_source_redshift: float,
) -> None:
    z_source = float(caustic_source_redshift)
    z_lens = getattr(evaluator.state, "z_lens", None)
    if z_lens is not None and np.isfinite(float(z_lens)) and z_source <= float(z_lens):
        _log(
            None,
            f"[plot:kappa_comparison] skipped: caustic source redshift z={z_source:g} "
            f"is not behind lens redshift z={float(z_lens):g}",
        )
        return
    kappa_true, kappa_wcs = _load_kappa_true_fits(kappa_true_fits)
    model_kappa, x_arcsec, y_arcsec = _kappa_model_grid_for_true_fits(
        evaluator,
        best_fit,
        kappa_true.shape,
        kappa_wcs,
        z_source,
    )
    valid_residual = np.isfinite(model_kappa) & np.isfinite(kappa_true) & (kappa_true > 0.0)
    fractional_residual = np.full(kappa_true.shape, np.nan, dtype=float)
    fractional_residual[valid_residual] = (model_kappa[valid_residual] - kappa_true[valid_residual]) / kappa_true[valid_residual]
    extent = [
        float(np.nanmin(x_arcsec)),
        float(np.nanmax(x_arcsec)),
        float(np.nanmin(y_arcsec)),
        float(np.nanmax(y_arcsec)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    kappa_image = axes[0].imshow(
        np.ma.masked_invalid(model_kappa),
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=3.0,
        aspect="equal",
    )
    kappa_colorbar = fig.colorbar(kappa_image, ax=axes[0], fraction=0.046, pad=0.04)
    kappa_colorbar.set_label(r"$\kappa_{\rm model}$")

    residual_image = axes[1].imshow(
        np.ma.masked_invalid(fractional_residual),
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=2.0),
        aspect="equal",
    )
    residual_colorbar = fig.colorbar(residual_image, ax=axes[1], fraction=0.046, pad=0.04)
    residual_colorbar.set_label(r"$(\kappa_{\rm model} - \kappa_{\rm true}) / \kappa_{\rm true}$")

    axes[0].set_title(fr"Model $\kappa$ (z={z_source:g})")
    axes[1].set_title(r"Fractional $\kappa$ Residual")
    for ax in axes:
        ax.invert_xaxis()
        ax.set_xlabel("x [arcsec]")
        ax.set_ylabel("y [arcsec]")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "kappa_comparison.pdf"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _empty_subhalo_properties_table() -> pd.DataFrame:
    return pd.DataFrame(columns=SUBHALO_PROPERTIES_COLUMNS)


def _finite_float_or_nan(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if np.isfinite(parsed) else float("nan")


def _subhalo_component_records(state: BuildState) -> list[dict[str, Any]]:
    packed = getattr(state, "packed_lens_spec", None)
    if packed is None:
        return []
    component_family = np.asarray(getattr(packed, "component_family", []), dtype=int)
    if component_family.size == 0:
        return []
    records: list[dict[str, Any]] = []
    for raw_record in getattr(state, "scaling_component_records", []) or []:
        if not isinstance(raw_record, dict) or "component_index" not in raw_record:
            continue
        try:
            component_index = int(raw_record["component_index"])
        except (TypeError, ValueError):
            continue
        if component_index < 0 or component_index >= component_family.size:
            continue
        if int(component_family[component_index]) != 1:
            continue
        record = dict(raw_record)
        record["component_index"] = component_index
        records.append(record)
    return records


def _lens_mass_to_msun(mass_angle: Any, sigma_crit_angle: float) -> float:
    mass_value = _finite_float_or_nan(np.asarray(mass_angle).reshape(()))
    if not (np.isfinite(mass_value) and np.isfinite(float(sigma_crit_angle)) and float(sigma_crit_angle) > 0.0):
        return float("nan")
    return float(mass_value * float(sigma_crit_angle))


def _subhalo_properties_table(
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    caustic_source_redshift: float,
) -> pd.DataFrame:
    records = _subhalo_component_records(state)
    if not records:
        return _empty_subhalo_properties_table()

    z_source = float(caustic_source_redshift)
    try:
        sigma_crit_angle = critical_surface_density_angle_from_config(
            float(getattr(state, "z_lens")),
            z_source,
            getattr(state, "cosmo_config", None),
        )
    except Exception as exc:  # pragma: no cover - malformed artifacts should still write diagnostic rows
        _log(None, f"[plot:subhalo_properties] mass conversion unavailable: {exc}")
        sigma_crit_angle = float("nan")

    best_fit_latent = _reported_physical_to_latent_vector(evaluator, np.asarray(best_fit, dtype=float))
    exact_models_by_z = getattr(evaluator, "exact_models_by_z", {})
    model = exact_models_by_z.get(z_source) if exact_models_by_z is not None else None
    if model is None:
        model, _ = evaluator._get_exact_model_solver(z_source)
    packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit_latent, dtype=jnp.float64), z_source)
    kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)

    packed = getattr(state, "packed_lens_spec", None)
    x_center_base = np.asarray(getattr(packed, "x_center_base", []), dtype=float)
    y_center_base = np.asarray(getattr(packed, "y_center_base", []), dtype=float)
    rows: list[dict[str, Any]] = []
    for record in records:
        component_index = int(record["component_index"])
        if component_index >= len(kwargs_lens):
            continue
        kwargs = dict(kwargs_lens[component_index])
        rs = _finite_float_or_nan(kwargs.get("Rs"))
        mass_within_rs = float("nan")
        mass_within_total = float("nan")
        if np.isfinite(rs) and rs > 0.0 and hasattr(model, "mass_3d"):
            bool_list = [idx == component_index for idx in range(len(kwargs_lens))]
            try:
                mass_within_rs = _lens_mass_to_msun(
                    model.mass_3d(rs, kwargs_lens, bool_list=bool_list),
                    sigma_crit_angle,
                )
                mass_within_total = _lens_mass_to_msun(
                    model.mass_3d(SUBHALO_TOTAL_MASS_RADIUS_FACTOR * rs, kwargs_lens, bool_list=bool_list),
                    sigma_crit_angle,
                )
            except Exception as exc:  # pragma: no cover - best-effort diagnostic for corrupt model rows
                _log(None, f"[plot:subhalo_properties] mass_3d failed component={component_index}: {exc}")
        x_centre = _finite_float_or_nan(record.get("x_centre"))
        y_centre = _finite_float_or_nan(record.get("y_centre"))
        if (not np.isfinite(x_centre)) and component_index < x_center_base.size:
            x_centre = _finite_float_or_nan(x_center_base[component_index])
        if (not np.isfinite(y_centre)) and component_index < y_center_base.size:
            y_centre = _finite_float_or_nan(y_center_base[component_index])
        radius = float(np.hypot(x_centre, y_centre)) if np.isfinite(x_centre) and np.isfinite(y_centre) else float("nan")
        rows.append(
            {
                "component_index": component_index,
                "potfile_id": str(record.get("potfile_id", "")),
                "potfile_order": int(record.get("potfile_order", -1)),
                "catalog_id": str(record.get("catalog_id", f"component{component_index}")),
                "catalog_mag": _finite_float_or_nan(record.get("catalog_mag")),
                "x_centre": float(x_centre),
                "y_centre": float(y_centre),
                "radius_arcsec": radius,
                "sigma0": _finite_float_or_nan(kwargs.get("sigma0")),
                "Ra": _finite_float_or_nan(kwargs.get("Ra")),
                "Rs": rs,
                "mass_within_Rs_msun": mass_within_rs,
                "mass_within_1e6_Rs_msun": mass_within_total,
            }
        )
    if not rows:
        return _empty_subhalo_properties_table()
    return pd.DataFrame(rows, columns=SUBHALO_PROPERTIES_COLUMNS).sort_values(
        ["potfile_order", "component_index"],
    ).reset_index(drop=True)


def _finite_positive_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.empty((0,), dtype=float)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def _finite_nonnegative_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.empty((0,), dtype=float)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values) & (values >= 0.0)]


def _subhalo_log_mass_bins(*mass_arrays: np.ndarray) -> np.ndarray | None:
    positive_arrays = [np.asarray(values, dtype=float) for values in mass_arrays if np.asarray(values).size]
    positive_arrays = [values[np.isfinite(values) & (values > 0.0)] for values in positive_arrays]
    positive_arrays = [values for values in positive_arrays if values.size]
    if not positive_arrays:
        return None
    log_mass = np.log10(np.concatenate(positive_arrays))
    log_min = float(np.nanmin(log_mass))
    log_max = float(np.nanmax(log_mass))
    if not np.isfinite(log_min) or not np.isfinite(log_max):
        return None
    if log_max <= log_min:
        log_min -= 0.25
        log_max += 0.25
    n_bins = int(np.clip(np.sqrt(max(log_mass.size, 1)), 6, 18))
    return np.linspace(log_min, log_max, n_bins + 1)


def _subhalo_linear_bins(values: np.ndarray) -> np.ndarray | None:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values) & (finite_values >= 0.0)]
    if finite_values.size == 0:
        return None
    value_min = float(np.nanmin(finite_values))
    value_max = float(np.nanmax(finite_values))
    if not np.isfinite(value_min) or not np.isfinite(value_max):
        return None
    if value_max <= value_min:
        padding = max(0.5, 0.1 * max(abs(value_min), 1.0))
        value_min = max(0.0, value_min - padding)
        value_max += padding
    n_bins = int(np.clip(np.sqrt(max(finite_values.size, 1)), 6, 18))
    return np.linspace(value_min, value_max, n_bins + 1)


def _plot_subhalo_mass_function(subhalo_df: pd.DataFrame, path: Path) -> None:
    mass_total = _finite_positive_column(subhalo_df, "mass_within_1e6_Rs_msun")
    bins = _subhalo_log_mass_bins(mass_total)
    if bins is None:
        _write_placeholder_plot(
            path,
            "Subhalo mass function",
            "No finite fitted subhalo masses are available.",
        )
        return
    bin_width = float(np.mean(np.diff(bins)))
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    if mass_total.size:
        log_mass_total = np.log10(mass_total)
        ax.hist(
            log_mass_total,
            bins=bins,
            weights=np.full(log_mass_total.size, 1.0 / bin_width),
            histtype="step",
            linewidth=2.0,
            color="tab:blue",
            label="Subhalo Mass",
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(M_{\rm sub}/M_\odot)$")
    ax.set_ylabel(r"$dN/d\log_{10}M$")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_subhalo_radial_distribution(subhalo_df: pd.DataFrame, path: Path) -> None:
    radius = _finite_nonnegative_column(subhalo_df, "radius_arcsec")
    bins = _subhalo_linear_bins(radius)
    if bins is None:
        _write_placeholder_plot(
            path,
            "Subhalo radial distribution",
            "No finite fitted subhalo radii are available.",
        )
        return
    bin_width = float(np.mean(np.diff(bins)))
    if not np.isfinite(bin_width) or bin_width <= 0.0:
        _write_placeholder_plot(
            path,
            "Subhalo radial distribution",
            "Subhalo radial bins are invalid.",
        )
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(
        radius,
        bins=bins,
        weights=np.full(radius.size, 1.0 / bin_width),
        histtype="stepfilled",
        color="tab:blue",
        alpha=0.75,
    )
    ax.set_xlabel("cluster-centric radius [arcsec]")
    ax.set_ylabel(r"$dN/dR$ [arcsec$^{-1}$]")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_image_catalog_cutout_helpers() -> Any:
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("plot_literature_family_cutouts")


def _image_catalog_family_cutout_enabled(args: argparse.Namespace, run_dir: Path) -> bool:
    image_dir = getattr(args, "image_catalog_family_cutout_image_dir", None)
    if image_dir is None or not str(image_dir).strip():
        return False
    stage_name = Path(run_dir).name
    if stage_name == "stage4_critical_arc_mixture_image_plane":
        return True
    return stage_name == "stage3_image_plane" and bool(getattr(args, "exact_image_diagnostics_stage3", False))


def _infer_image_catalog_cutout_cluster(state: BuildState) -> str:
    text = " ".join(
        [
            str(getattr(state, "run_name", "")),
            str(getattr(state, "par_path", "")),
        ]
    ).lower()
    candidates = (
        ("a2744", ("a2744", "abell2744")),
        ("ares", ("ares",)),
        ("m0416", ("m0416", "macs0416")),
        ("m1206", ("m1206", "macsj1206")),
        ("as1063", ("as1063", "abells1063", "rxcj2248", "a1063")),
        ("hera", ("hera",)),
        ("a370", ("a370", "abell370", "a307")),
        ("m0717", ("m0717", "macs0717")),
        ("m1149", ("m1149", "macs1149")),
    )
    for cluster, tokens in candidates:
        if any(token in text for token in tokens):
            return cluster
    return str(getattr(state, "run_name", "cluster")).split("_", 1)[0].lower() or "cluster"


def _arcsec_to_skycoord(x_arcsec: Any, y_arcsec: Any, reference: tuple[int, float, float]) -> SkyCoord | None:
    try:
        _reference_type, ra0_deg, dec0_deg = reference
        x_value = float(x_arcsec)
        y_value = float(y_arcsec)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x_value) or not np.isfinite(y_value):
        return None
    cos_dec0 = math.cos(math.radians(float(dec0_deg)))
    if cos_dec0 == 0.0:
        return None
    ra_deg = float(ra0_deg) - x_value / (3600.0 * cos_dec0)
    dec_deg = float(dec0_deg) + y_value / 3600.0
    return SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")


def _cutout_npixels_for_image(image: Any, *, cutout_size_arcsec: float) -> int | None:
    if cutout_size_arcsec <= 0.0:
        return None
    pixel_scale = float(getattr(image, "pixel_scale_arcsec", np.nan))
    if not np.isfinite(pixel_scale) or pixel_scale <= 0.0:
        return None
    return max(1, int(math.ceil(float(cutout_size_arcsec) / pixel_scale)))


def _clamped_cutout_origin(raw_origin: int, npix: int, image_size: int) -> int:
    if image_size >= npix:
        return int(np.clip(raw_origin, 0, image_size - npix))
    return int(np.clip(raw_origin, image_size - npix, 0))


def _cutout_window_origin_xy(
    image: Any,
    center_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
) -> tuple[int, int] | None:
    npix = _cutout_npixels_for_image(image, cutout_size_arcsec=cutout_size_arcsec)
    if npix is None:
        return None
    try:
        ny, nx = tuple(int(value) for value in image.shape[:2])
    except (AttributeError, TypeError, ValueError):
        return None
    center_x, center_y = image.wcs.world_to_pixel(center_coord)
    if not all(np.isfinite(value) for value in (center_x, center_y)):
        return None
    half = npix // 2
    raw_x0 = int(round(float(center_x))) - half
    raw_y0 = int(round(float(center_y))) - half
    if raw_x0 >= nx or raw_x0 + npix <= 0 or raw_y0 >= ny or raw_y0 + npix <= 0:
        return None
    return _clamped_cutout_origin(raw_x0, npix, nx), _clamped_cutout_origin(raw_y0, npix, ny)


def _cutout_pixel_xy(
    image: Any,
    center_coord: SkyCoord,
    target_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
) -> tuple[float, float] | None:
    origin = _cutout_window_origin_xy(image, center_coord, cutout_size_arcsec=cutout_size_arcsec)
    if origin is None:
        return None
    target_x, target_y = image.wcs.world_to_pixel(target_coord)
    if not all(np.isfinite(value) for value in (target_x, target_y)):
        return None
    x0, y0 = origin
    return float(target_x) - float(x0), float(target_y) - float(y0)


def _lock_cutout_axis_to_image(ax: plt.Axes, rendered_shape: tuple[int, ...]) -> None:
    if len(rendered_shape) < 2:
        return
    height = int(rendered_shape[0])
    width = int(rendered_shape[1])
    if height <= 0 or width <= 0:
        return
    ax.set_xlim(-0.5, float(width) - 0.5)
    ax.set_ylim(-0.5, float(height) - 0.5)
    ax.set_autoscale_on(False)


def _draw_cutout_circle(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    target_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    color: str,
    radius_arcsec: float,
    linestyle: str = "-",
    linewidth: float = 0.9,
    alpha: float = 0.85,
    zorder: float = 8,
) -> None:
    pixel = _cutout_pixel_xy(image, center_coord, target_coord, cutout_size_arcsec=cutout_size_arcsec)
    if pixel is None:
        return
    x, y = pixel
    height, width = rendered_shape
    radius = float(radius_arcsec) / float(image.pixel_scale_arcsec)
    if x < -radius or x > width - 1 + radius or y < -radius or y > height - 1 + radius:
        return
    ax.add_patch(
        Circle(
            (x, y),
            radius=radius,
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
            clip_on=True,
        )
    )


def _draw_cutout_segment(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    start_coord: SkyCoord,
    end_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    color: str,
    linewidth: float = 0.9,
    linestyle: str = "-",
    alpha: float = 0.85,
    zorder: float = 8,
) -> tuple[float, float] | None:
    start_pixel = _cutout_pixel_xy(image, center_coord, start_coord, cutout_size_arcsec=cutout_size_arcsec)
    end_pixel = _cutout_pixel_xy(image, center_coord, end_coord, cutout_size_arcsec=cutout_size_arcsec)
    if start_pixel is None or end_pixel is None:
        return None
    sx, sy = start_pixel
    ex, ey = end_pixel
    height, width = rendered_shape
    if max(sx, ex) < 0 or min(sx, ex) > width - 1 or max(sy, ey) < 0 or min(sy, ey) > height - 1:
        return None
    ax.plot(
        [sx, ex],
        [sy, ey],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
        clip_on=True,
    )
    return ex, ey


def _arcsec_curve_values(value: Any) -> np.ndarray:
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return np.asarray([], dtype=float)
        try:
            return np.asarray(json.loads(text), dtype=float).reshape(-1)
        except (json.JSONDecodeError, TypeError, ValueError):
            return np.asarray([], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)


def _image_catalog_finite_polyline_points(x_values: Any, y_values: Any) -> np.ndarray:
    x_array = _arcsec_curve_values(x_values)
    y_array = _arcsec_curve_values(y_values)
    if x_array.size == 0 or x_array.shape != y_array.shape:
        return np.empty((0, 2), dtype=float)
    points = np.column_stack([x_array, y_array])
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] <= 1:
        return points
    keep = [0]
    for index in range(1, points.shape[0]):
        if float(np.linalg.norm(points[index] - points[keep[-1]])) > 0.0:
            keep.append(index)
    return points[np.asarray(keep, dtype=int)]


def _image_catalog_polyline_cumulative_arclength(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.zeros(points.shape[0], dtype=float)
    segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def _image_catalog_project_point_to_polyline(
    point: np.ndarray,
    points: np.ndarray,
    *,
    preferred_s: float | None = None,
) -> dict[str, Any] | None:
    point_array = np.asarray(point, dtype=float).reshape(2)
    if points.shape[0] < 2 or not np.isfinite(point_array).all():
        return None
    segment = points[1:] - points[:-1]
    segment_len2 = np.sum(np.square(segment), axis=1)
    valid = segment_len2 > 0.0
    if not np.any(valid):
        return None
    start = points[:-1][valid]
    direction = segment[valid]
    valid_len2 = segment_len2[valid]
    t_values = np.clip(np.sum((point_array - start) * direction, axis=1) / valid_len2, 0.0, 1.0)
    projections = start + t_values[:, None] * direction
    distances = np.linalg.norm(projections - point_array, axis=1)
    cumulative = _image_catalog_polyline_cumulative_arclength(points)
    valid_indices = np.flatnonzero(valid)
    s_values = cumulative[valid_indices] + t_values * np.sqrt(valid_len2)
    min_distance = float(np.min(distances))
    tolerance = max(1.0e-9, 1.0e-6 * max(1.0, min_distance))
    candidates = np.flatnonzero(distances <= min_distance + tolerance)
    if preferred_s is not None and np.isfinite(preferred_s):
        best = int(candidates[np.argmin(np.abs(s_values[candidates] - float(preferred_s)))])
    else:
        best = int(candidates[0])
    return {
        "point": projections[best],
        "distance": float(distances[best]),
        "s": float(s_values[best]),
    }


def _image_catalog_point_at_polyline_s(points: np.ndarray, cumulative: np.ndarray, s_value: float) -> np.ndarray:
    if points.shape[0] == 0:
        return np.asarray([np.nan, np.nan], dtype=float)
    if points.shape[0] == 1:
        return points[0].copy()
    s_clamped = float(np.clip(s_value, cumulative[0], cumulative[-1]))
    index = int(np.searchsorted(cumulative, s_clamped, side="right") - 1)
    index = max(0, min(index, points.shape[0] - 2))
    segment_length = float(cumulative[index + 1] - cumulative[index])
    if segment_length <= 0.0:
        return points[index].copy()
    t_value = (s_clamped - float(cumulative[index])) / segment_length
    return points[index] + t_value * (points[index + 1] - points[index])


def _image_catalog_polyline_between_s(points: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    if points.shape[0] < 2 or not (np.isfinite(start_s) and np.isfinite(end_s)):
        return np.empty((0, 2), dtype=float)
    cumulative = _image_catalog_polyline_cumulative_arclength(points)
    start = _image_catalog_point_at_polyline_s(points, cumulative, start_s)
    end = _image_catalog_point_at_polyline_s(points, cumulative, end_s)
    if not (np.isfinite(start).all() and np.isfinite(end).all()):
        return np.empty((0, 2), dtype=float)
    if abs(float(end_s) - float(start_s)) <= 1.0e-10:
        return np.vstack([start, end])
    low = min(float(start_s), float(end_s))
    high = max(float(start_s), float(end_s))
    interior_mask = (cumulative > low + 1.0e-10) & (cumulative < high - 1.0e-10)
    interior = points[interior_mask]
    if float(end_s) < float(start_s):
        interior = interior[::-1]
    return np.vstack([start, interior, end])


def _image_catalog_arc_support_geometry(row: pd.Series) -> dict[str, Any] | None:
    observed = _image_catalog_finite_arcsec_pair(row, "x_obs_arcsec", "y_obs_arcsec")
    anchor = _image_catalog_finite_arcsec_pair(row, "arc_support_anchor_x_arcsec", "arc_support_anchor_y_arcsec")
    if observed is None or anchor is None:
        return None
    points = _image_catalog_finite_polyline_points(
        row.get("arc_support_curve_x_arcsec", "[]"),
        row.get("arc_support_curve_y_arcsec", "[]"),
    )
    if points.shape[0] < 2:
        return None
    observed_point = np.asarray(observed, dtype=float)
    anchor_point = np.asarray(anchor, dtype=float)
    anchor_projection = _image_catalog_project_point_to_polyline(anchor_point, points)
    if anchor_projection is None:
        return None
    closest_projection = _image_catalog_project_point_to_polyline(
        observed_point,
        points,
        preferred_s=float(anchor_projection["s"]),
    )
    if closest_projection is None:
        return None
    tangential_curve = _image_catalog_polyline_between_s(
        points,
        float(closest_projection["s"]),
        float(anchor_projection["s"]),
    )
    if tangential_curve.shape[0] < 2:
        tangential_curve = np.vstack([closest_projection["point"], anchor_projection["point"]])
    return {
        "observed_arcsec": observed_point,
        "closest_arcsec": np.asarray(closest_projection["point"], dtype=float),
        "anchor_arcsec": anchor_point,
        "anchor_curve_arcsec": np.asarray(anchor_projection["point"], dtype=float),
        "residual_arcsec": float(closest_projection["distance"]),
        "closest_s_arcsec": float(closest_projection["s"]),
        "anchor_s_arcsec": float(anchor_projection["s"]),
        "tangential_curve_arcsec": np.asarray(tangential_curve, dtype=float),
    }


def _draw_cutout_polyline_arcsec(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    x_arcsec: Any,
    y_arcsec: Any,
    reference: tuple[int, float, float],
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    color: str,
    linewidth: float = 0.9,
    linestyle: str = "-",
    alpha: float = 0.85,
    zorder: float = 8,
) -> bool:
    x_values = _arcsec_curve_values(x_arcsec)
    y_values = _arcsec_curve_values(y_arcsec)
    if x_values.size < 2 or x_values.shape != y_values.shape:
        return False
    pixels: list[tuple[float, float]] = []
    for x_value, y_value in zip(x_values, y_values):
        coord = _arcsec_to_skycoord(float(x_value), float(y_value), reference)
        if coord is None:
            pixels.append((np.nan, np.nan))
            continue
        pixel = _cutout_pixel_xy(image, center_coord, coord, cutout_size_arcsec=cutout_size_arcsec)
        pixels.append(pixel if pixel is not None else (np.nan, np.nan))
    pixel_array = np.asarray(pixels, dtype=float)
    finite = np.isfinite(pixel_array).all(axis=1)
    if np.sum(finite) < 2:
        return False
    height, width = rendered_shape
    visible = (
        finite
        & (pixel_array[:, 0] >= 0.0)
        & (pixel_array[:, 0] <= width - 1)
        & (pixel_array[:, 1] >= 0.0)
        & (pixel_array[:, 1] <= height - 1)
    )
    if not np.any(visible):
        return False
    finite_indices = np.flatnonzero(finite)
    runs = np.split(finite_indices, np.where(np.diff(finite_indices) != 1)[0] + 1)
    drawn = False
    for run in runs:
        if run.size < 2:
            continue
        ax.plot(
            pixel_array[run, 0],
            pixel_array[run, 1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
            clip_on=True,
        )
        drawn = True
    return drawn


def _draw_image_catalog_arc_support_curve(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    row: pd.Series,
    reference: tuple[int, float, float],
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    alpha: float = 0.92,
    linewidth: float = 1.05,
    zorder: float = 10,
) -> bool:
    if not _image_catalog_draw_arc_anchor_overlays(row):
        return False
    return _draw_cutout_polyline_arcsec(
        ax,
        image,
        center_coord,
        row.get("arc_support_curve_x_arcsec", "[]"),
        row.get("arc_support_curve_y_arcsec", "[]"),
        reference,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        color=_image_catalog_status_color("ARC_RECOVERED"),
        linewidth=linewidth,
        linestyle="--",
        alpha=alpha,
        zorder=zorder,
    )


def _draw_cutout_direction_frame(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    row: pd.Series,
    reference: tuple[int, float, float],
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
) -> None:
    try:
        x0 = float(row["x_obs_arcsec"])
        y0 = float(row["y_obs_arcsec"])
        tx = float(row.get("arc_critical_direction_x", np.nan))
        ty = float(row.get("arc_critical_direction_y", np.nan))
        nx = float(row.get("arc_noncritical_direction_x", np.nan))
        ny = float(row.get("arc_noncritical_direction_y", np.nan))
    except (TypeError, ValueError):
        return
    if not np.all(np.isfinite([x0, y0, tx, ty, nx, ny])):
        return
    half_length = min(1.6, max(0.55, 0.16 * float(cutout_size_arcsec)))
    for label, vx, vy, color in (("T", tx, ty, "#00e5ff"), ("N", nx, ny, "#ff4da6")):
        start_coord = _arcsec_to_skycoord(x0 - half_length * vx, y0 - half_length * vy, reference)
        end_coord = _arcsec_to_skycoord(x0 + half_length * vx, y0 + half_length * vy, reference)
        if start_coord is None or end_coord is None:
            continue
        end_pixel = _draw_cutout_segment(
            ax,
            image,
            center_coord,
            start_coord,
            end_coord,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
            color=color,
            linewidth=1.05,
            alpha=0.9,
            zorder=11,
        )
        if end_pixel is not None:
            ax.text(
                end_pixel[0],
                end_pixel[1],
                label,
                color=color,
                fontsize=5.0,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=12,
                clip_on=True,
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 0.35},
            )


def _draw_image_catalog_cab_morphology(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    row: pd.Series,
    reference: tuple[int, float, float],
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    alpha: float = 0.85,
    zorder: float = 12,
) -> None:
    if not bool(row.get("cab_has_constraint", False)):
        return
    try:
        x0 = float(row.get("cab_anchor_x_arcsec", np.nan))
        y0 = float(row.get("cab_anchor_y_arcsec", np.nan))
        angle = float(row.get("cab_tangent_angle_model_rad", np.nan))
        curvature = float(row.get("cab_curvature_model_arcsec_inv", np.nan))
    except (TypeError, ValueError):
        return
    if not np.all(np.isfinite([x0, y0, angle])):
        return
    tangent_x = math.cos(angle)
    tangent_y = math.sin(angle)
    half_length = min(1.8, max(0.45, 0.14 * float(cutout_size_arcsec)))
    start_coord = _arcsec_to_skycoord(x0 - half_length * tangent_x, y0 - half_length * tangent_y, reference)
    end_coord = _arcsec_to_skycoord(x0 + half_length * tangent_x, y0 + half_length * tangent_y, reference)
    if start_coord is not None and end_coord is not None:
        _draw_cutout_segment(
            ax,
            image,
            center_coord,
            start_coord,
            end_coord,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
            color="#ffe04d",
            linewidth=1.25,
            alpha=alpha,
            zorder=zorder,
        )
    if not np.isfinite(curvature) or curvature <= 0.0:
        return
    radius_arcsec = 1.0 / curvature
    if radius_arcsec > 2.0 * float(cutout_size_arcsec):
        return
    normal_x = -tangent_y
    normal_y = tangent_x
    circle_center = _arcsec_to_skycoord(x0 + radius_arcsec * normal_x, y0 + radius_arcsec * normal_y, reference)
    if circle_center is None:
        return
    _draw_cutout_circle(
        ax,
        image,
        center_coord,
        circle_center,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        color="#ffe04d",
        radius_arcsec=radius_arcsec,
        linestyle="--",
        linewidth=0.85,
        alpha=0.55 * alpha,
        zorder=zorder - 0.5,
    )


def _format_cutout_float(value: Any, precision: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "na"
    if not np.isfinite(numeric):
        return "na"
    return f"{numeric:.{precision}g}"


def _format_image_catalog_diagnostic_label(row: pd.Series) -> str:
    status = str(row.get("arc_recovery_status", row.get("image_recovery_status", ""))).replace("_", " ")
    lines = [
        f"{row.get('image_label', '')} z={_format_cutout_float(row.get('z_source', row.get('catalog_z')), 3)}",
        (
            f"r={_format_cutout_float(row.get('image_residual_arcsec'))} "
            f"arc={_format_cutout_float(row.get('arc_aware_image_residual_arcsec'))}"
        ),
        (
            f"N={_format_cutout_float(row.get('arc_noncritical_direction_residual_arcsec'))} "
            f"T={_format_cutout_float(row.get('arc_critical_direction_residual_arcsec'))}"
        ),
        (
            f"curve={_format_cutout_float(row.get('arc_curve_distance_arcsec'))} "
            f"s={_format_cutout_float(row.get('arc_curve_arclength_arcsec'))}"
        ),
        (
            f"s={_format_cutout_float(row.get('arc_s_min'))}/"
            f"{_format_cutout_float(row.get('arc_s_max'))} "
            f"det={_format_cutout_float(row.get('arc_detA'))}"
        ),
    ]
    if bool(row.get("cab_has_constraint", False)):
        lines.append(
            (
                f"CAB dphi={_format_cutout_float(row.get('cab_tangent_residual_rad'))} "
                f"dk={_format_cutout_float(row.get('cab_curvature_residual_arcsec_inv'))}"
            )
        )
    lines.append(f"p_arc={_format_cutout_float(row.get('arc_prior_probability'))} {status}")
    return "\n".join(lines)


def _image_catalog_cutout_rows(state: BuildState, image_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in state.family_data:
        for label, x_obs, y_obs in zip(family.image_labels, family.x_obs, family.y_obs):
            coord = _arcsec_to_skycoord(float(x_obs), float(y_obs), state.reference)
            if coord is None:
                continue
            rows.append(
                {
                    "family_id": str(family.family_id),
                    "image_label": str(label),
                    "x_obs_arcsec": float(x_obs),
                    "y_obs_arcsec": float(y_obs),
                    "ra": float(coord.ra.deg),
                    "dec": float(coord.dec.deg),
                    "z_source": float(family.z_source),
                    "effective_z_source": float(family.effective_z_source),
                    "catalog_z": float(family.z_source),
                }
            )
    base = pd.DataFrame(rows)
    if base.empty or image_df is None or image_df.empty:
        return base
    diagnostics = image_df.copy()
    for column in ("family_id", "image_label"):
        if column in diagnostics.columns:
            diagnostics[column] = diagnostics[column].astype(str)
    merged = base.merge(
        diagnostics.drop(columns=[column for column in ("x_obs_arcsec", "y_obs_arcsec") if column in diagnostics.columns]),
        on=["family_id", "image_label"],
        how="left",
        suffixes=("", "_diagnostic"),
    )
    return merged


def _image_catalog_extra_cutout_rows(state: BuildState, extra_image_df: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "family_id",
        "extra_image_index",
        "image_label",
        "x_model_arcsec",
        "y_model_arcsec",
        "x_center_arcsec",
        "y_center_arcsec",
        "ra",
        "dec",
        "z_source",
        "effective_z_source",
        "catalog_z",
        "image_recovery_status",
    ]
    if extra_image_df is None or extra_image_df.empty:
        return pd.DataFrame(columns=columns)
    family_by_id = {str(family.family_id): family for family in getattr(state, "family_data", [])}
    rows: list[dict[str, Any]] = []
    for _, row in extra_image_df.iterrows():
        family_id = str(row.get("family_id", ""))
        family = family_by_id.get(family_id)
        try:
            x_model = float(row.get("x_model_arcsec"))
            y_model = float(row.get("y_model_arcsec"))
        except (TypeError, ValueError):
            continue
        coord = _arcsec_to_skycoord(x_model, y_model, state.reference)
        if coord is None:
            continue

        def redshift_value(column: str, fallback: Any) -> float:
            try:
                value = float(row.get(column, fallback))
            except (TypeError, ValueError):
                value = float(fallback)
            return value if np.isfinite(value) else float(fallback)

        family_z = redshift_value("z_source", getattr(family, "z_source", np.nan))
        effective_z = redshift_value("effective_z_source", getattr(family, "effective_z_source", family_z))
        extra_index = row.get("extra_image_index", len(rows) + 1)
        extra_label = f"{family_id}.extra{extra_index}"
        extra_row = row.to_dict()
        extra_row.update(
            {
                "family_id": family_id,
                "extra_image_index": extra_index,
                "image_label": str(extra_label),
                "x_model_arcsec": x_model,
                "y_model_arcsec": y_model,
                "x_center_arcsec": x_model,
                "y_center_arcsec": y_model,
                "ra": float(coord.ra.deg),
                "dec": float(coord.dec.deg),
                "z_source": float(family_z),
                "effective_z_source": float(effective_z),
                "catalog_z": float(family_z),
                "image_recovery_status": "extra",
            }
        )
        rows.append(extra_row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def _image_catalog_status_color(status: str) -> str:
    normalized = str(status).strip().upper()
    if normalized in {"POINT_RECOVERED", "OBSERVED"}:
        return "#4da3ff"
    if normalized == "ARC_RECOVERED":
        return "#ffd54f"
    if normalized == "MISSED":
        return "#ff4d5e"
    if normalized == "EXTRA":
        return "tab:purple"
    if normalized == "MODEL":
        return "#4caf50"
    if normalized == "FAMILY":
        return "#00e5ff"
    return "#f5f5f5"


def _image_catalog_status_display_text(status: str) -> str:
    normalized = str(status).strip().upper()
    labels = {
        "POINT_RECOVERED": "point recovered",
        "OBSERVED": "point recovered",
        "ARC_RECOVERED": "arc recovered",
        "MISSED": "not recovered",
        "EXTRA": "extra",
    }
    return labels.get(normalized, str(status).strip().replace("_", " ").lower())


def _image_catalog_finite_arcsec_pair(row: pd.Series, x_column: str, y_column: str) -> tuple[float, float] | None:
    if x_column not in row or y_column not in row:
        return None
    try:
        x_value = float(row.get(x_column, np.nan))
        y_value = float(row.get(y_column, np.nan))
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(x_value) and np.isfinite(y_value)):
        return None
    return x_value, y_value


def _image_catalog_has_finite_model_position(row: pd.Series) -> bool:
    return _image_catalog_finite_arcsec_pair(row, "x_model_arcsec", "y_model_arcsec") is not None


def _image_catalog_truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _image_catalog_point_recovered(row: pd.Series) -> bool:
    status = str(row.get("image_recovery_status", "")).strip().lower()
    if status == "recovered":
        return True
    if _image_catalog_truthy(row.get("exact_image_prediction_failed", False)):
        return False
    if status in {"not_recovered", "missing", "missed", "failed"}:
        return False
    return _image_catalog_has_finite_model_position(row)


def _image_catalog_arc_recovered(row: pd.Series) -> bool:
    if _image_catalog_point_recovered(row):
        return True
    arc_status = str(row.get("arc_recovery_status", "")).strip().lower()
    if arc_status in {"point_recovered", "arc_supported", "arc_recovered"}:
        return True
    try:
        arc_residual = float(row.get("arc_aware_image_residual_arcsec", np.nan))
    except (TypeError, ValueError):
        arc_residual = np.nan
    if np.isfinite(arc_residual):
        return True
    return _image_catalog_truthy(row.get("arc_supported", False))


def _image_catalog_observed_panel_status(row: pd.Series) -> str:
    if _image_catalog_point_recovered(row):
        return "POINT_RECOVERED"
    if _image_catalog_arc_recovered(row):
        return "ARC_RECOVERED"
    return "MISSED"


def _image_catalog_draw_arc_anchor_overlays(row: pd.Series) -> bool:
    panel_status = str(row.get("panel_status", "")).strip().upper()
    if panel_status and panel_status not in {"NAN", "NONE", "NULL"}:
        return panel_status == "ARC_RECOVERED"
    return _image_catalog_observed_panel_status(row) == "ARC_RECOVERED"


def _image_catalog_display_model_arcsec(row: pd.Series) -> tuple[float, float] | None:
    return _image_catalog_finite_arcsec_pair(row, "x_model_arcsec", "y_model_arcsec")


def _image_catalog_display_model_coord(row: pd.Series, reference: tuple[int, float, float]) -> SkyCoord | None:
    pair = _image_catalog_display_model_arcsec(row)
    if pair is None:
        return None
    return _arcsec_to_skycoord(pair[0], pair[1], reference)


def _finite_image_catalog_points(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.empty((0, 2), dtype=float)
    points = [np.asarray(array, dtype=float).reshape(-1, 2) for array in arrays if np.asarray(array).size]
    if not points:
        return np.empty((0, 2), dtype=float)
    stacked = np.vstack(points)
    return stacked[np.isfinite(stacked).all(axis=1)]


def _image_catalog_xy_points(data: pd.DataFrame, x_column: str, y_column: str) -> np.ndarray:
    if data.empty or x_column not in data.columns or y_column not in data.columns:
        return np.empty((0, 2), dtype=float)
    x_values = pd.to_numeric(data[x_column], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(data[y_column], errors="coerce").to_numpy(dtype=float)
    return np.column_stack([x_values, y_values])


def _image_catalog_arc_anchor_overlay_rows(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    mask = np.asarray([_image_catalog_draw_arc_anchor_overlays(row) for _, row in data.iterrows()], dtype=bool)
    return data.loc[mask]


IMAGE_CATALOG_DETAIL_COLUMNS = 3
IMAGE_CATALOG_MIN_OVERVIEW_SIZE_ARCSEC = 40.0
IMAGE_CATALOG_OVERVIEW_LABEL_FONT_SIZE = 8.6
IMAGE_CATALOG_DETAIL_LABEL_FONT_SIZE = 8.0
IMAGE_CATALOG_STATUS_LABEL_FONT_SIZE = 8.0
IMAGE_CATALOG_LEGEND_FONT_SIZE = 8.0


def _image_catalog_geometry_from_points(
    points: np.ndarray,
    *,
    minimum_side_arcsec: float = IMAGE_CATALOG_MIN_OVERVIEW_SIZE_ARCSEC,
) -> tuple[float, float, float]:
    finite = _finite_image_catalog_points(points)
    if finite.size == 0:
        return 0.0, 0.0, float(minimum_side_arcsec)
    x_min, y_min = np.min(finite, axis=0)
    x_max, y_max = np.max(finite, axis=0)
    span = float(max(x_max - x_min, y_max - y_min))
    padding = max(2.0, 0.15 * span)
    cutout_size = max(float(minimum_side_arcsec), span + 2.0 * padding)
    return float(0.5 * (x_min + x_max)), float(0.5 * (y_min + y_max)), float(cutout_size)


def _image_catalog_arc_support_curve_points(data: pd.DataFrame) -> np.ndarray:
    if data.empty:
        return np.empty((0, 2), dtype=float)
    curves: list[np.ndarray] = []
    for _, row in data.iterrows():
        if not _image_catalog_draw_arc_anchor_overlays(row):
            continue
        curve = _image_catalog_finite_polyline_points(
            row.get("arc_support_curve_x_arcsec", "[]"),
            row.get("arc_support_curve_y_arcsec", "[]"),
        )
        if curve.shape[0]:
            curves.append(curve)
    if not curves:
        return np.empty((0, 2), dtype=float)
    return np.vstack(curves)


def _image_catalog_cluster_overview_geometry(catalog_df: pd.DataFrame) -> tuple[float, float, float]:
    return _image_catalog_geometry_from_points(
        _image_catalog_xy_points(catalog_df, "x_obs_arcsec", "y_obs_arcsec"),
        minimum_side_arcsec=IMAGE_CATALOG_MIN_OVERVIEW_SIZE_ARCSEC,
    )


def _image_catalog_overview_geometry(
    observed: pd.DataFrame,
    extras: pd.DataFrame,
    default_cutout_size_arcsec: float,
) -> tuple[float, float, float]:
    arc_anchor_observed = _image_catalog_arc_anchor_overlay_rows(observed)
    points = _finite_image_catalog_points(
        _image_catalog_xy_points(observed, "x_obs_arcsec", "y_obs_arcsec"),
        _image_catalog_xy_points(observed, "x_model_arcsec", "y_model_arcsec"),
        _image_catalog_xy_points(arc_anchor_observed, "arc_support_anchor_x_arcsec", "arc_support_anchor_y_arcsec"),
        _image_catalog_arc_support_curve_points(arc_anchor_observed),
        _image_catalog_xy_points(extras, "x_model_arcsec", "y_model_arcsec"),
    )
    return _image_catalog_geometry_from_points(
        points,
        minimum_side_arcsec=max(float(default_cutout_size_arcsec), IMAGE_CATALOG_MIN_OVERVIEW_SIZE_ARCSEC),
    )


def _image_catalog_family_block_layout(detail_count: int, detail_cols: int) -> dict[str, int]:
    detail_cols = max(1, int(detail_cols))
    detail_count = max(1, int(detail_count))
    overview_units = detail_cols
    detail_row_count = max(1, int(math.ceil(float(detail_count) / float(detail_cols))))
    return {
        "overview_units": int(overview_units),
        "detail_row_count": int(detail_row_count),
        "layout_rowspan": int(overview_units + detail_row_count),
    }


def _image_catalog_family_cutout_blocks(
    state: BuildState,
    catalog_df: pd.DataFrame,
    extra_df: pd.DataFrame | None,
    *,
    detail_cols: int,
    default_cutout_size_arcsec: float,
) -> list[dict[str, Any]]:
    detail_cols = max(1, int(detail_cols))
    if extra_df is None:
        extra_df = pd.DataFrame()
    catalog_family_ids = set(catalog_df.get("family_id", pd.Series(dtype=object)).astype(str))
    extra_family_ids = set(extra_df.get("family_id", pd.Series(dtype=object)).astype(str)) if not extra_df.empty else set()
    available_family_ids = catalog_family_ids | extra_family_ids
    family_ids = [str(family.family_id) for family in getattr(state, "family_data", []) if str(family.family_id) in available_family_ids]
    blocks: list[dict[str, Any]] = []
    family_by_id = {str(family.family_id): family for family in getattr(state, "family_data", [])}
    for family_id in family_ids:
        family = family_by_id.get(family_id)
        observed = (
            catalog_df.loc[catalog_df["family_id"].astype(str) == family_id].reset_index(drop=True)
            if not catalog_df.empty and "family_id" in catalog_df.columns
            else pd.DataFrame()
        )
        extras = (
            extra_df.loc[extra_df["family_id"].astype(str) == family_id].reset_index(drop=True)
            if not extra_df.empty and "family_id" in extra_df.columns
            else pd.DataFrame()
        )
        detail_panels: list[dict[str, Any]] = []
        for panel_index, (_, image_row) in enumerate(observed.iterrows()):
            row = image_row.to_dict()
            row.update(
                {
                    "panel_kind": "observed",
                    "panel_status": _image_catalog_observed_panel_status(image_row),
                    "panel_index": int(panel_index),
                    "x_center_arcsec": float(image_row.get("x_obs_arcsec", np.nan)),
                    "y_center_arcsec": float(image_row.get("y_obs_arcsec", np.nan)),
                    "cutout_size_arcsec": float(default_cutout_size_arcsec),
                }
            )
            detail_panels.append(row)
        for extra_offset, (_, extra_row) in enumerate(extras.iterrows(), start=len(detail_panels)):
            row = extra_row.to_dict()
            row.update(
                {
                    "panel_kind": "extra",
                    "panel_status": "EXTRA",
                    "panel_index": int(extra_offset),
                    "x_center_arcsec": float(extra_row.get("x_center_arcsec", extra_row.get("x_model_arcsec", np.nan))),
                    "y_center_arcsec": float(extra_row.get("y_center_arcsec", extra_row.get("y_model_arcsec", np.nan))),
                    "cutout_size_arcsec": float(default_cutout_size_arcsec),
                }
            )
            detail_panels.append(row)
        detail_count = max(1, len(detail_panels))
        layout = _image_catalog_family_block_layout(detail_count, detail_cols)
        overview_x, overview_y, overview_size = _image_catalog_overview_geometry(
            observed,
            extras,
            float(default_cutout_size_arcsec),
        )
        if family is not None and hasattr(family, "z_source"):
            z_source = float(getattr(family, "z_source"))
        elif not observed.empty and "z_source" in observed.columns:
            z_source = float(observed["z_source"].iloc[0])
        elif not extras.empty and "z_source" in extras.columns:
            z_source = float(extras["z_source"].iloc[0])
        else:
            z_source = np.nan
        if family is not None and hasattr(family, "effective_z_source"):
            effective_z = float(getattr(family, "effective_z_source"))
        elif not observed.empty and "effective_z_source" in observed.columns:
            effective_z = float(observed["effective_z_source"].iloc[0])
        elif not extras.empty and "effective_z_source" in extras.columns:
            effective_z = float(extras["effective_z_source"].iloc[0])
        else:
            effective_z = z_source
        blocks.append(
            {
                "family_id": family_id,
                "family": family,
                "z_source": z_source,
                "effective_z_source": effective_z,
                "observed": observed,
                "extras": extras,
                "detail_panels": detail_panels,
                "detail_cols": detail_cols,
                "overview_rowspan": int(layout["layout_rowspan"]),
                "overview_units": int(layout["overview_units"]),
                "detail_row_count": int(layout["detail_row_count"]),
                "layout_rowspan": int(layout["layout_rowspan"]),
                "overview_center_x_arcsec": float(overview_x),
                "overview_center_y_arcsec": float(overview_y),
                "overview_cutout_size_arcsec": float(overview_size),
            }
        )
    return blocks


def _image_catalog_family_block_pages(blocks: list[dict[str, Any]], row_slot_budget: int) -> list[list[dict[str, Any]]]:
    row_slot_budget = max(1, int(row_slot_budget))
    pages: list[list[dict[str, Any]]] = []
    current_page: list[dict[str, Any]] = []
    current_rows = 0
    for block in blocks:
        block_rows = max(1, int(block.get("layout_rowspan", block.get("overview_rowspan", 1))))
        if current_page and current_rows + block_rows > row_slot_budget:
            pages.append(current_page)
            current_page = []
            current_rows = 0
        current_page.append(block)
        current_rows += block_rows
    if current_page:
        pages.append(current_page)
    return pages


def _draw_image_catalog_critical_lines(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    center_x_arcsec: Any,
    center_y_arcsec: Any,
    reference: tuple[int, float, float],
    model: Any,
    kwargs_lens: list[dict[str, float]],
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
) -> None:
    try:
        x0 = float(center_x_arcsec)
        y0 = float(center_y_arcsec)
    except (TypeError, ValueError):
        return
    if not np.isfinite(x0) or not np.isfinite(y0):
        return
    half = 0.62 * float(cutout_size_arcsec)
    grid_scale = max(0.05, min(0.18, float(cutout_size_arcsec) / 80.0))
    n_grid = max(12, int(round(2.0 * half / grid_scale)) + 1)
    x_axis = np.linspace(x0 - half, x0 + half, n_grid)
    y_axis = np.linspace(y0 - half, y0 + half, n_grid)
    try:
        contours = _critical_curve_caustics(
            model,
            kwargs_lens,
            x_axis,
            y_axis,
            include_tangential=True,
            include_radial=True,
        )
    except Exception:
        return
    colors = {"tangential": "#ffd54f", "radial": "#ff4da6"}
    linestyles = {"tangential": "-", "radial": "--"}
    for contour in contours:
        crit_x = np.asarray(contour.get("critical_x", []), dtype=float)
        crit_y = np.asarray(contour.get("critical_y", []), dtype=float)
        if crit_x.size < 2 or crit_x.shape != crit_y.shape:
            continue
        pixels: list[tuple[float, float]] = []
        for x_value, y_value in zip(crit_x, crit_y):
            coord = _arcsec_to_skycoord(x_value, y_value, reference)
            if coord is None:
                pixels.append((np.nan, np.nan))
                continue
            pixel = _cutout_pixel_xy(image, center_coord, coord, cutout_size_arcsec=cutout_size_arcsec)
            pixels.append(pixel if pixel is not None else (np.nan, np.nan))
        pixel_array = np.asarray(pixels, dtype=float)
        finite = np.isfinite(pixel_array).all(axis=1)
        if not finite.any():
            continue
        kind = str(contour.get("kind", "tangential"))
        ax.plot(
            pixel_array[finite, 0],
            pixel_array[finite, 1],
            color=colors.get(kind, "white"),
            linestyle=linestyles.get(kind, "-"),
            linewidth=0.65,
            alpha=0.78,
            zorder=6,
        )


def _draw_image_catalog_status_label(ax: plt.Axes, status: str) -> None:
    ax.text(
        0.965,
        0.965,
        str(status).upper(),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=IMAGE_CATALOG_STATUS_LABEL_FONT_SIZE,
        fontweight="bold",
        color=_image_catalog_status_color(status),
        clip_on=True,
        zorder=22,
        bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 0.55},
    )


IMAGE_CATALOG_PANEL_TEXT_BBOX = {"facecolor": "black", "alpha": 0.38, "edgecolor": "none", "pad": 0.55}


def _image_catalog_compact_status_text(row: pd.Series, fallback: str) -> str:
    value = row.get("arc_recovery_status", fallback)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return _image_catalog_status_display_text(fallback)
    if text.lower() in {"not_recovered", "missing", "missed", "failed"} and str(fallback).strip().upper() in {
        "POINT_RECOVERED",
        "ARC_RECOVERED",
    }:
        return _image_catalog_status_display_text(fallback)
    return text.replace("_", " ").lower()


def _format_image_catalog_extra_label(row: pd.Series) -> str:
    label = str(row.get("image_label", "")).strip()
    if not label:
        label = f"Extra {_format_cutout_float(row.get('extra_image_index'), 3)}"
    return "\n".join(
        [
            f"{label}  extra",
            (
                f"model x={_format_cutout_float(row.get('x_model_arcsec'), 3)} "
                f"y={_format_cutout_float(row.get('y_model_arcsec'), 3)}"
            ),
        ]
    )


def _format_image_catalog_overview_label(block: dict[str, Any]) -> str:
    observed = block["observed"]
    extras = block["extras"]
    point_recovered = sum(1 for _, row in observed.iterrows() if _image_catalog_point_recovered(row))
    arc_recovered = sum(1 for _, row in observed.iterrows() if _image_catalog_arc_recovered(row))
    return "\n".join(
        [
            f"Family {block['family_id']}  z={_format_cutout_float(block.get('z_source'), 3)}",
            (
                f"Nobs={len(observed)}  Npoint_recovered={point_recovered}  "
                f"Narc_recovered={arc_recovered}  Nextra={len(extras)}"
            ),
        ]
    )


def _format_image_catalog_compact_detail_label(row: pd.Series) -> str:
    panel_status = str(row.get("panel_status", "OBSERVED"))
    lines = [
        f"{row.get('image_label', '')}  {_image_catalog_compact_status_text(row, panel_status)}",
    ]
    metrics: list[str] = []
    for label, column in (
        ("r", "image_residual_arcsec"),
        ("r_arc", "arc_aware_image_residual_arcsec"),
        ("d_curve", "arc_curve_distance_arcsec"),
    ):
        try:
            value = float(row.get(column, np.nan))
        except (TypeError, ValueError):
            value = np.nan
        if np.isfinite(value):
            metrics.append(f"{label}={_format_cutout_float(value)}")
    if metrics:
        lines.append("  ".join(metrics))
    return "\n".join(lines)


def _draw_image_catalog_panel_text(ax: plt.Axes, label: str, *, fontsize: float = IMAGE_CATALOG_DETAIL_LABEL_FONT_SIZE) -> None:
    ax.text(
        0.035,
        0.965,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        color="white",
        linespacing=0.95,
        clip_on=True,
        zorder=20,
        bbox=IMAGE_CATALOG_PANEL_TEXT_BBOX,
    )


def _image_catalog_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=_image_catalog_status_color("POINT_RECOVERED"),
            markeredgewidth=1.15,
            color=_image_catalog_status_color("POINT_RECOVERED"),
            label="point recovered image",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=_image_catalog_status_color("ARC_RECOVERED"),
            markeredgewidth=1.15,
            color=_image_catalog_status_color("ARC_RECOVERED"),
            label="arc recovered image",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=_image_catalog_status_color("MISSED"),
            markeredgewidth=1.25,
            color=_image_catalog_status_color("MISSED"),
            label="missed observed image",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="--",
            markerfacecolor="none",
            markeredgecolor=_image_catalog_status_color("MODEL"),
            markeredgewidth=1.15,
            color=_image_catalog_status_color("MODEL"),
            label="matched model image",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=_image_catalog_status_color("EXTRA"),
            markeredgewidth=1.15,
            color=_image_catalog_status_color("EXTRA"),
            label="extra model image",
        ),
        Line2D(
            [0],
            [0],
            color=_image_catalog_status_color("ARC_RECOVERED"),
            linestyle="--",
            linewidth=1.25,
            label="arc-support curve",
        ),
        Line2D([0], [0], color="#bdbdbd", linewidth=0.9, label="observed-to-model residual"),
        Line2D(
            [0],
            [0],
            color=_image_catalog_status_color("ARC_RECOVERED"),
            linewidth=1.25,
            label="tangential arc displacement",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            markeredgecolor=_image_catalog_status_color("MODEL"),
            color=_image_catalog_status_color("MODEL"),
            markersize=4.0,
            markeredgewidth=0.9,
            label="linearized arc anchor",
        ),
    ]


def _add_image_catalog_axis_legend(ax: plt.Axes) -> None:
    legend = ax.legend(
        handles=_image_catalog_legend_handles(),
        loc="lower right",
        ncol=1,
        fontsize=IMAGE_CATALOG_LEGEND_FONT_SIZE,
        frameon=True,
        framealpha=0.72,
        facecolor="black",
        edgecolor="0.45",
        handlelength=2.4,
        handletextpad=0.6,
        columnspacing=0.9,
        borderpad=0.45,
    )
    for text in legend.get_texts():
        text.set_color("white")


def _image_catalog_panel_center(row: pd.Series, reference: tuple[int, float, float]) -> tuple[SkyCoord | None, float, float]:
    try:
        x_center = float(row.get("x_center_arcsec"))
        y_center = float(row.get("y_center_arcsec"))
    except (TypeError, ValueError):
        x_center = y_center = np.nan
    coord = _arcsec_to_skycoord(x_center, y_center, reference)
    return coord, x_center, y_center


def _image_catalog_draw_rgb_cutout(
    ax: plt.Axes,
    helpers: Any,
    band_images: dict[str, Any],
    bands: Sequence[str],
    rgb_display: Any,
    center_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
) -> np.ndarray:
    cutouts = {
        str(band): helpers.extract_band_cutout(
            band_images[str(band)],
            center_coord,
            cutout_size_arcsec=cutout_size_arcsec,
        )
        for band in bands
    }
    rgb = helpers.make_rgb_cutout(cutouts, bands=bands, rgb_display=rgb_display)
    height, width = rgb.shape[:2]
    ax.imshow(
        rgb,
        origin="lower",
        interpolation="bilinear",
        extent=(-0.5, width - 0.5, -0.5, height - 0.5),
    )
    _lock_cutout_axis_to_image(ax, rgb.shape)
    return rgb


def _draw_image_catalog_observed_marker(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    target_coord: SkyCoord,
    *,
    status: str,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    zorder: float = 10,
) -> None:
    _draw_cutout_circle(
        ax,
        image,
        center_coord,
        target_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        color=_image_catalog_status_color(status),
        radius_arcsec=0.5,
        linewidth=1.6 if str(status).upper() in {"MISSED", "ARC_RECOVERED"} else 1.45,
        alpha=0.95,
        zorder=zorder,
    )


def _draw_image_catalog_model_marker(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    model_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    zorder: float = 10,
) -> None:
    _draw_cutout_circle(
        ax,
        image,
        center_coord,
        model_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        color=_image_catalog_status_color("MODEL"),
        radius_arcsec=0.5,
        linestyle="--",
        linewidth=1.45,
        alpha=0.95,
        zorder=zorder,
    )


def _draw_image_catalog_arc_anchor_marker(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    anchor_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    zorder: float = 13,
) -> bool:
    pixel = _cutout_pixel_xy(image, center_coord, anchor_coord, cutout_size_arcsec=cutout_size_arcsec)
    if pixel is None:
        return False
    x_pixel, y_pixel = pixel
    height, width = rendered_shape
    if x_pixel < 0.0 or x_pixel > width - 1 or y_pixel < 0.0 or y_pixel > height - 1:
        return False
    ax.plot(
        [x_pixel],
        [y_pixel],
        marker="x",
        linestyle="None",
        color=_image_catalog_status_color("MODEL"),
        markersize=5.0,
        markeredgewidth=1.2,
        alpha=0.98,
        zorder=zorder,
        clip_on=True,
    )
    return True


def _draw_image_catalog_arc_supported_components(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    row: pd.Series,
    reference: tuple[int, float, float],
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
) -> bool:
    if not _image_catalog_draw_arc_anchor_overlays(row):
        return False
    geometry = _image_catalog_arc_support_geometry(row)
    if geometry is None:
        return False
    observed_coord = _arcsec_to_skycoord(*geometry["observed_arcsec"], reference)
    closest_coord = _arcsec_to_skycoord(*geometry["closest_arcsec"], reference)
    anchor_coord = _arcsec_to_skycoord(*geometry["anchor_arcsec"], reference)
    if observed_coord is None or closest_coord is None or anchor_coord is None:
        return False
    _draw_cutout_segment(
        ax,
        image,
        center_coord,
        observed_coord,
        closest_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
            color="#bdbdbd",
            linewidth=1.15,
        alpha=0.86,
        zorder=12,
    )
    tangential_curve = np.asarray(geometry["tangential_curve_arcsec"], dtype=float)
    if tangential_curve.shape[0] >= 2:
        _draw_cutout_polyline_arcsec(
            ax,
            image,
            center_coord,
            tangential_curve[:, 0],
            tangential_curve[:, 1],
            reference,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
            color=_image_catalog_status_color("ARC_RECOVERED"),
            linewidth=1.55,
            alpha=0.96,
            zorder=11,
        )
    _draw_image_catalog_arc_anchor_marker(
        ax,
        image,
        center_coord,
        anchor_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
    )
    return True


def _draw_image_catalog_extra_marker(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    extra_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    zorder: float = 10,
) -> None:
    _draw_cutout_circle(
        ax,
        image,
        center_coord,
        extra_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        color=_image_catalog_status_color("EXTRA"),
        radius_arcsec=0.5,
        linewidth=1.45,
        alpha=0.95,
        zorder=zorder,
    )


def _draw_image_catalog_cluster_overview_panel(
    ax: plt.Axes,
    helpers: Any,
    band_images: dict[str, Any],
    bands: Sequence[str],
    rgb_display: Any,
    display_image: Any,
    catalog_df: pd.DataFrame,
    reference: tuple[int, float, float],
) -> None:
    center_x, center_y, cutout_size_arcsec = _image_catalog_cluster_overview_geometry(catalog_df)
    center_coord = _arcsec_to_skycoord(center_x, center_y, reference)
    if center_coord is None:
        ax.set_axis_off()
        return
    rgb = _image_catalog_draw_rgb_cutout(
        ax,
        helpers,
        band_images,
        bands,
        rgb_display,
        center_coord,
        cutout_size_arcsec=cutout_size_arcsec,
    )
    for _, image_row in catalog_df.iterrows():
        target_coord = _arcsec_to_skycoord(image_row.get("x_obs_arcsec"), image_row.get("y_obs_arcsec"), reference)
        if target_coord is None:
            continue
        _draw_image_catalog_observed_marker(
            ax,
            display_image,
            center_coord,
            target_coord,
            status=_image_catalog_observed_panel_status(image_row),
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
            zorder=10,
        )
    _add_image_catalog_axis_legend(ax)


def _draw_image_catalog_overview_panel(
    ax: plt.Axes,
    helpers: Any,
    band_images: dict[str, Any],
    bands: Sequence[str],
    rgb_display: Any,
    display_image: Any,
    block: dict[str, Any],
    reference: tuple[int, float, float],
    _model_pair: tuple[Any, list[dict[str, float]]] | None,
) -> None:
    center_coord = _arcsec_to_skycoord(
        block["overview_center_x_arcsec"],
        block["overview_center_y_arcsec"],
        reference,
    )
    if center_coord is None:
        ax.set_axis_off()
        return
    cutout_size_arcsec = float(block["overview_cutout_size_arcsec"])
    rgb = _image_catalog_draw_rgb_cutout(
        ax,
        helpers,
        band_images,
        bands,
        rgb_display,
        center_coord,
        cutout_size_arcsec=cutout_size_arcsec,
    )
    for _, image_row in block["observed"].iterrows():
        target_coord = _arcsec_to_skycoord(image_row.get("x_obs_arcsec"), image_row.get("y_obs_arcsec"), reference)
        if target_coord is None:
            continue
        status = _image_catalog_observed_panel_status(image_row)
        draw_arc_anchor_overlays = _image_catalog_draw_arc_anchor_overlays(image_row)
        if draw_arc_anchor_overlays:
            _draw_image_catalog_arc_support_curve(
                ax,
                display_image,
                center_coord,
                image_row,
                reference,
                cutout_size_arcsec=cutout_size_arcsec,
                rendered_shape=rgb.shape[:2],
                alpha=0.60,
                linewidth=1.1,
                zorder=7,
            )
        _draw_image_catalog_observed_marker(
            ax,
            display_image,
            center_coord,
            target_coord,
            status=status,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
        )
        if draw_arc_anchor_overlays:
            _draw_image_catalog_arc_supported_components(
                ax,
                display_image,
                center_coord,
                image_row,
                reference,
                cutout_size_arcsec=cutout_size_arcsec,
                rendered_shape=rgb.shape[:2],
            )
            continue
        model_coord = _image_catalog_display_model_coord(image_row, reference)
        if model_coord is None:
            continue
        _draw_image_catalog_model_marker(
            ax,
            display_image,
            center_coord,
            model_coord,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
        )
        _draw_cutout_segment(
            ax,
            display_image,
            center_coord,
            target_coord,
            model_coord,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
            color="#bdbdbd",
            linewidth=1.05,
            alpha=0.72,
            zorder=9,
        )
    for _, extra_row in block["extras"].iterrows():
        extra_coord = _arcsec_to_skycoord(extra_row.get("x_model_arcsec"), extra_row.get("y_model_arcsec"), reference)
        if extra_coord is None:
            continue
        _draw_image_catalog_extra_marker(
            ax,
            display_image,
            center_coord,
            extra_coord,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
        )
    _draw_image_catalog_panel_text(ax, _format_image_catalog_overview_label(block), fontsize=IMAGE_CATALOG_OVERVIEW_LABEL_FONT_SIZE)
    _add_image_catalog_axis_legend(ax)


def _draw_image_catalog_detail_panel(
    ax: plt.Axes,
    helpers: Any,
    band_images: dict[str, Any],
    bands: Sequence[str],
    rgb_display: Any,
    display_image: Any,
    row: pd.Series,
    reference: tuple[int, float, float],
    _model_pair: tuple[Any, list[dict[str, float]]] | None,
) -> None:
    center_coord, _center_x, _center_y = _image_catalog_panel_center(row, reference)
    if center_coord is None:
        ax.set_axis_off()
        return
    cutout_size_arcsec = float(row.get("cutout_size_arcsec", 10.0))
    rgb = _image_catalog_draw_rgb_cutout(
        ax,
        helpers,
        band_images,
        bands,
        rgb_display,
        center_coord,
        cutout_size_arcsec=cutout_size_arcsec,
    )
    panel_kind = str(row.get("panel_kind", "observed"))
    panel_status = str(row.get("panel_status", "OBSERVED"))
    if panel_kind == "extra":
        extra_coord = _arcsec_to_skycoord(row.get("x_model_arcsec"), row.get("y_model_arcsec"), reference)
        if extra_coord is not None:
            _draw_image_catalog_extra_marker(
                ax,
                display_image,
                center_coord,
                extra_coord,
                cutout_size_arcsec=cutout_size_arcsec,
                rendered_shape=rgb.shape[:2],
            )
        label = _format_image_catalog_extra_label(row)
    else:
        observed_coord = _arcsec_to_skycoord(row.get("x_obs_arcsec"), row.get("y_obs_arcsec"), reference)
        draw_arc_anchor_overlays = _image_catalog_draw_arc_anchor_overlays(row)
        if observed_coord is not None:
            if draw_arc_anchor_overlays:
                _draw_image_catalog_arc_support_curve(
                    ax,
                    display_image,
                    center_coord,
                    row,
                    reference,
                    cutout_size_arcsec=cutout_size_arcsec,
                    rendered_shape=rgb.shape[:2],
                    alpha=0.92,
                    linewidth=1.15,
                    zorder=10,
                )
            _draw_image_catalog_observed_marker(
                ax,
                display_image,
                center_coord,
                observed_coord,
                status=panel_status,
                cutout_size_arcsec=cutout_size_arcsec,
                rendered_shape=rgb.shape[:2],
            )
        if draw_arc_anchor_overlays:
            _draw_image_catalog_arc_supported_components(
                ax,
                display_image,
                center_coord,
                row,
                reference,
                cutout_size_arcsec=cutout_size_arcsec,
                rendered_shape=rgb.shape[:2],
            )
        else:
            model_coord = _image_catalog_display_model_coord(row, reference)
            if model_coord is not None:
                _draw_image_catalog_model_marker(
                    ax,
                    display_image,
                    center_coord,
                    model_coord,
                    cutout_size_arcsec=cutout_size_arcsec,
                    rendered_shape=rgb.shape[:2],
                )
                if observed_coord is not None:
                    _draw_cutout_segment(
                        ax,
                        display_image,
                        center_coord,
                        observed_coord,
                        model_coord,
                        cutout_size_arcsec=cutout_size_arcsec,
                        rendered_shape=rgb.shape[:2],
                        color="#bdbdbd",
                        linewidth=1.05,
                        alpha=0.75,
                        zorder=9,
                    )
        label = _format_image_catalog_compact_detail_label(row)
    _draw_image_catalog_panel_text(ax, label, fontsize=IMAGE_CATALOG_DETAIL_LABEL_FONT_SIZE)


def _plot_image_catalog_family_cutouts(
    run_dir: Path,
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    image_df: pd.DataFrame,
    extra_image_df: pd.DataFrame | None,
    args: argparse.Namespace,
) -> None:
    helpers = _load_image_catalog_cutout_helpers()
    image_dir = Path(str(getattr(args, "image_catalog_family_cutout_image_dir")))
    image_scale = str(getattr(args, "image_catalog_family_cutout_image_scale", "60mas"))
    requested_bands = getattr(args, "image_catalog_family_cutout_bands", None)
    bands = tuple(requested_bands) if requested_bands is not None else tuple(getattr(helpers, "DEFAULT_BANDS", ("F435W", "F606W", "F814W")))
    if len(bands) != 3:
        raise ValueError("--image-catalog-family-cutout-bands must contain exactly three bands.")
    cutout_size_arcsec = float(getattr(helpers, "DEFAULT_CUTOUT_SIZE_ARCSEC", 10.0))
    cluster = _infer_image_catalog_cutout_cluster(state)
    band_paths = helpers.find_rgb_band_paths(image_dir, cluster=cluster, bands=bands, image_scale=image_scale)
    band_images = helpers.load_rgb_metadata(band_paths, bands=bands)
    rgb_kwargs: dict[str, Any] = {}
    for arg_name, kwarg_name in (
        ("image_catalog_family_cutout_rgb_q", "q"),
        ("image_catalog_family_cutout_rgb_stretch", "stretch"),
        ("image_catalog_family_cutout_rgb_minimum", "minimum"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            rgb_kwargs[kwarg_name] = float(value)
    channel_gains = dict(getattr(helpers, "DEFAULT_RGB_CHANNEL_GAINS", {"red": 1.0, "green": 1.0, "blue": 1.2}))
    channel_gain_supplied = False
    for arg_name, role in (
        ("image_catalog_family_cutout_rgb_red_gain", "red"),
        ("image_catalog_family_cutout_rgb_green_gain", "green"),
        ("image_catalog_family_cutout_rgb_blue_gain", "blue"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            channel_gains[role] = float(value)
            channel_gain_supplied = True
    if channel_gain_supplied:
        rgb_kwargs["channel_gains"] = channel_gains
    rgb_display = helpers.build_rgb_display(band_images, bands=bands, **rgb_kwargs)
    display_image = band_images[str(bands[-1])]
    catalog_df = _image_catalog_cutout_rows(state, image_df)
    extra_df = _image_catalog_extra_cutout_rows(state, extra_image_df)
    if catalog_df.empty and extra_df.empty:
        _write_placeholder_plot(
            _plot_path(run_dir, "image_catalog_family_cutouts.pdf"),
            "Image-catalog family cutouts",
            "No image catalog rows are available.",
        )
        return

    output = _plot_path(run_dir, "image_catalog_family_cutouts.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    detail_cols = IMAGE_CATALOG_DETAIL_COLUMNS
    blocks = _image_catalog_family_cutout_blocks(
        state,
        catalog_df,
        extra_df,
        detail_cols=detail_cols,
        default_cutout_size_arcsec=cutout_size_arcsec,
    )
    with PdfPages(output) as pdf:
        cluster_units = detail_cols
        fig = plt.figure(
            figsize=helpers._figure_size(cluster_units, detail_cols),
            dpi=getattr(helpers, "CUTOUT_FIGURE_DPI", 300),
        )
        helpers._style_cutout_figure(fig)
        grid = fig.add_gridspec(cluster_units, detail_cols)
        cluster_ax = fig.add_subplot(grid[:, :])
        helpers._style_cutout_axis(cluster_ax)
        _draw_image_catalog_cluster_overview_panel(
            cluster_ax,
            helpers,
            band_images,
            bands,
            rgb_display,
            display_image,
            catalog_df,
            state.reference,
        )
        pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.02, dpi=getattr(helpers, "CUTOUT_FIGURE_DPI", 300))
        plt.close(fig)

        for block in blocks:
            n_rows = max(1, int(block.get("layout_rowspan", block.get("overview_rowspan", 1))))
            n_cols = detail_cols
            fig = plt.figure(
                figsize=helpers._figure_size(n_rows, n_cols),
                dpi=getattr(helpers, "CUTOUT_FIGURE_DPI", 300),
            )
            helpers._style_cutout_figure(fig)
            grid = fig.add_gridspec(n_rows, n_cols)
            overview_units = max(1, int(block.get("overview_units", detail_cols)))
            overview_ax = fig.add_subplot(grid[:overview_units, :])
            helpers._style_cutout_axis(overview_ax)
            _draw_image_catalog_overview_panel(
                overview_ax,
                helpers,
                band_images,
                bands,
                rgb_display,
                display_image,
                block,
                state.reference,
                None,
            )
            for panel_index, panel in enumerate(block["detail_panels"]):
                detail_row = overview_units + panel_index // detail_cols
                detail_col = panel_index % detail_cols
                ax = fig.add_subplot(grid[detail_row, detail_col])
                helpers._style_cutout_axis(ax)
                _draw_image_catalog_detail_panel(
                    ax,
                    helpers,
                    band_images,
                    bands,
                    rgb_display,
                    display_image,
                    pd.Series(panel),
                    state.reference,
                    None,
                )
            pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.02, dpi=getattr(helpers, "CUTOUT_FIGURE_DPI", 300))
            plt.close(fig)


def _generate_plots_and_tables(
    run_dir: Path,
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    best_eval: EvaluationResult,
    results: PosteriorResults,
    runtime_sec: float,
    args: argparse.Namespace,
) -> None:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_df = _run_logged_phase(
        args,
        "plots.summary_table",
        lambda: _summary_table(
            state.parameter_specs,
            results.samples,
            best_fit,
            sample_weights=results.sample_weights,
        ),
    )
    family_df = _run_logged_phase(args, "plots.family_diagnostics_table", lambda: _family_diagnostics_table(evaluator, best_eval))
    image_fit_quality_df, model_magnification_df, image_recovery_extra_df = _run_logged_phase(
        args,
        "plots.fit_quality_tables",
        lambda: _fit_quality_tables(state, evaluator, best_fit, results, args),
    )
    image_count_recovery_df = _run_logged_phase(
        args,
        "plots.image_count_recovery_table",
        lambda: _image_count_recovery_table(state, image_fit_quality_df),
    )
    arc_aware_family_df = _run_logged_phase(
        args,
        "plots.arc_aware_family_diagnostics_table",
        lambda: _arc_aware_family_diagnostics_from_image_rows(image_fit_quality_df),
    )
    if not arc_aware_family_df.empty:
        family_df = family_df.copy()
        family_df["family_id"] = family_df["family_id"].astype(str)
        arc_aware_family_df["family_id"] = arc_aware_family_df["family_id"].astype(str)
        family_df = family_df.merge(arc_aware_family_df, on="family_id", how="left")
    max_tree_depth = _first_int_value(getattr(args, "max_tree_depth", 10), 10)
    chain_health_df = _run_logged_phase(
        args,
        "plots.chain_health_table",
        lambda: _chain_health_summary_table(
            results,
            state.parameter_specs,
            max_tree_depth=max_tree_depth,
        ),
    )
    chain_parameter_diagnostics_df = _run_logged_phase(
        args,
        "plots.chain_parameter_diagnostics_table",
        lambda: _chain_parameter_diagnostics_table(results, state.parameter_specs),
    )
    run_summary = _run_logged_phase(
        args,
        "plots.run_summary",
        lambda: _run_summary(
            args,
            state,
            runtime_sec,
            results,
            best_eval.loglike,
            evaluator,
            image_fit_quality_df=image_fit_quality_df,
            image_count_recovery_df=image_count_recovery_df,
        ),
    )
    run_summary_text = _format_run_summary_text(run_summary)
    scaling_specs, scaling_samples, scaling_best_fit = _run_logged_phase(
        args,
        "plots.scaling_subset",
        lambda: _scaling_parameter_subset(state.parameter_specs, results.samples, best_fit),
    )
    cosmology_specs, cosmology_samples, cosmology_best_fit = _run_logged_phase(
        args,
        "plots.cosmology_subset",
        lambda: _cosmology_parameter_subset(state.parameter_specs, results.samples, best_fit),
    )
    bayes_corner_overlay = _run_logged_phase(
        args,
        "plots.load_bayes_corner_overlay",
        lambda: _load_bayes_corner_overlay(getattr(args, "corner_overlay_bayes_dat", None), state),
    )
    best_par_marker_values = _run_logged_phase(
        args,
        "plots.load_best_par_corner_marker",
        lambda: _load_best_par_marker_values(getattr(args, "corner_overlay_best_par", None), state),
    )
    best_fit_values = _best_fit_values_for_specs(state.parameter_specs, best_fit)
    scaling_best_fit_values = _best_fit_values_for_specs(scaling_specs, scaling_best_fit)
    cosmology_best_fit_values = _best_fit_values_for_specs(cosmology_specs, cosmology_best_fit)
    previous_stage_best_values = getattr(state, "previous_stage_best_values", None)
    potfile_constraint_df = _run_logged_phase(
        args,
        "plots.potfile_constraint_table",
        lambda: _potfile_constraint_diagnostics_table(
            state.parameter_specs,
            results.samples,
            best_fit,
            evaluator.scaling_rank_df,
            sample_weights=results.sample_weights,
        ),
    )
    trace_specs, trace_grouped_samples = _run_logged_phase(
        args,
        "plots.scaling_grouped_subset",
        lambda: _scaling_grouped_subset(state.parameter_specs, results.grouped_samples),
    )
    subhalo_df = _run_logged_phase(
        args,
        "plots.subhalo_properties_table",
        lambda: _subhalo_properties_table(
            state,
            evaluator,
            best_fit,
            getattr(args, "caustic_source_redshift", 9.0),
        ),
    )

    _run_logged_phase(args, "plots.write_potential_summary_csv", lambda: summary_df.to_csv(tables_dir / "potential_summary.csv", index=False))
    _run_logged_phase(args, "plots.write_family_diagnostics_csv", lambda: family_df.to_csv(tables_dir / "family_diagnostics.csv", index=False))
    _run_logged_phase(
        args,
        "plots.write_image_fit_quality_csv",
        lambda: image_fit_quality_df.to_csv(tables_dir / "image_fit_quality.csv", index=False),
    )
    _run_logged_phase(
        args,
        "plots.write_image_count_recovery_csv",
        lambda: image_count_recovery_df.to_csv(tables_dir / "image_count_recovery.csv", index=False),
    )
    _run_logged_phase(
        args,
        "plots.write_image_recovery_extra_images_csv",
        lambda: image_recovery_extra_df.to_csv(tables_dir / "image_recovery_extra_images.csv", index=False),
    )
    _run_logged_phase(
        args,
        "plots.write_model_magnification_csv",
        lambda: model_magnification_df.to_csv(tables_dir / "model_magnification.csv", index=False),
    )
    _run_logged_phase(
        args,
        "plots.write_subhalo_properties_csv",
        lambda: subhalo_df.to_csv(tables_dir / "subhalo_properties.csv", index=False),
    )
    _run_logged_phase(args, "plots.write_potfile_summary_txt", lambda: _write_potfile_summary_txt(tables_dir, summary_df))
    if not potfile_constraint_df.empty:
        _run_logged_phase(
            args,
            "plots.write_potfile_constraint_csv",
            lambda: potfile_constraint_df.to_csv(tables_dir / "potfile_constraint_diagnostics.csv", index=False),
        )
        _run_logged_phase(
            args,
            "plots.write_potfile_constraint_txt",
            lambda: _write_potfile_constraint_summary_txt(tables_dir, potfile_constraint_df),
        )
    if not evaluator.scaling_rank_df.empty:
        _run_logged_phase(
            args,
            "plots.write_scaling_rank_csv",
            lambda: evaluator.scaling_rank_df.to_csv(tables_dir / "scaling_rank_diagnostics.csv", index=False),
        )
    if not chain_health_df.empty:
        _run_logged_phase(
            args,
            "plots.write_chain_health_csv",
            lambda: chain_health_df.to_csv(tables_dir / "chain_health_summary.csv", index=False),
        )
    if not chain_parameter_diagnostics_df.empty:
        _run_logged_phase(
            args,
            "plots.write_chain_parameter_diagnostics_csv",
            lambda: chain_parameter_diagnostics_df.to_csv(tables_dir / "chain_parameter_diagnostics.csv", index=False),
        )
    _run_logged_phase(
        args,
        "plots.write_run_summary_json",
        lambda: (tables_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8"),
    )
    _run_logged_phase(
        args,
        "plots.write_run_summary_txt",
        lambda: (tables_dir / "run_summary.txt").write_text(run_summary_text, encoding="utf-8"),
    )

    plot_tasks: list[PlotTask] = [
        (
            "corner",
            "plots.corner",
            lambda: _plot_corner(
                run_dir,
                results.samples,
                state.parameter_specs,
                best_fit_values=best_fit_values,
                previous_stage_best_values=previous_stage_best_values,
                bayes_corner_overlay=bayes_corner_overlay,
                best_par_marker_values=best_par_marker_values,
            ),
        ),
        (
            "potfile_corner",
            "plots.potfile_corner",
            lambda: _plot_potfile_corner(
                run_dir,
                scaling_samples,
                scaling_specs,
                best_fit_values=scaling_best_fit_values,
                previous_stage_best_values=previous_stage_best_values,
                bayes_corner_overlay=bayes_corner_overlay,
                best_par_marker_values=best_par_marker_values,
            ),
        ),
        (
            "potfile_prior_posterior",
            "plots.potfile_prior_posterior",
            lambda: _plot_potfile_prior_posterior(run_dir, potfile_constraint_df, results.samples, state.parameter_specs),
        ),
        (
            "potfile_constraint_strength",
            "plots.potfile_constraint_strength",
            lambda: _plot_potfile_constraint_strength(run_dir, potfile_constraint_df),
        ),
        ("potfile_prior_shift", "plots.potfile_prior_shift", lambda: _plot_potfile_prior_shift(run_dir, potfile_constraint_df)),
        (
            "potfile_leverage_summary",
            "plots.potfile_leverage_summary",
            lambda: _plot_potfile_leverage_summary(run_dir, potfile_constraint_df),
        ),
        ("trace", "plots.trace", lambda: _plot_trace(run_dir, trace_grouped_samples, trace_specs)),
        ("scaling_rank_bars", "plots.scaling_rank_bars", lambda: _plot_scaling_rank_bars(run_dir, evaluator.scaling_rank_df)),
        (
            "scaling_rank_scatter",
            "plots.scaling_rank_scatter",
            lambda: _plot_scaling_rank_scatter(run_dir, evaluator.scaling_rank_df),
        ),
        ("run_diagnostics", "plots.run_diagnostics", lambda: _plot_run_diagnostics(run_dir, results)),
        ("weights_logl", "plots.weights_logl", lambda: _plot_weights_logl(run_dir, results)),
        (
            "chain_health",
            "plots.chain_health",
            lambda: _plot_chain_health(
                run_dir,
                results,
                state.parameter_specs,
                max_tree_depth=max_tree_depth,
            ),
        ),
        (
            "chain_ranked_trace",
            "plots.chain_ranked_trace",
            lambda: _plot_chain_ranked_trace(
                run_dir,
                results.grouped_samples,
                state.parameter_specs,
                chain_parameter_diagnostics_df,
            ),
        ),
        ("residuals_by_family", "plots.residuals_by_family", lambda: _plot_residuals_by_family(run_dir, family_df)),
        (
            "source_plane_residual_histogram",
            "plots.source_plane_residual_histogram",
            lambda: _plot_source_plane_residual_histogram(run_dir, state, best_eval),
        ),
        (
            "image_recovery",
            "plots.image_recovery",
            lambda: _plot_image_recovery_fit_quality(
                image_fit_quality_df,
                _plot_path(run_dir, "image_recovery.pdf"),
                image_recovery_extra_df,
            ),
        ),
        *(
            [
                (
                    "image_catalog_family_cutouts",
                    "plots.image_catalog_family_cutouts",
                    lambda: _plot_image_catalog_family_cutouts(
                        run_dir,
                        state,
                        evaluator,
                        best_fit,
                        image_fit_quality_df,
                        image_recovery_extra_df,
                        args,
                    ),
                )
            ]
            if _image_catalog_family_cutout_enabled(args, run_dir)
            else []
        ),
        (
            "image_count_recovery",
            "plots.image_count_recovery",
            lambda: _plot_image_count_recovery(image_count_recovery_df, _plot_path(run_dir, "image_count_recovery.pdf")),
        ),
        (
            "model_magnification",
            "plots.model_magnification",
            lambda: _plot_model_magnification_fit_quality(model_magnification_df, _plot_path(run_dir, "model_magnification.pdf")),
        ),
        (
            "normalized_image_residuals",
            "plots.normalized_image_residuals",
            lambda: _plot_normalized_image_residuals(image_fit_quality_df, _plot_path(run_dir, "normalized_image_residuals.pdf")),
        ),
        (
            "image_residual_histogram",
            "plots.image_residual_histogram",
            lambda: _plot_image_residual_histogram(image_fit_quality_df, _plot_path(run_dir, "image_residual_histogram.pdf")),
        ),
        (
            "residual_vs_magnification",
            "plots.residual_vs_magnification",
            lambda: _plot_residual_vs_magnification(
                image_fit_quality_df,
                model_magnification_df,
                _plot_path(run_dir, "residual_vs_magnification.pdf"),
            ),
        ),
        (
            "residual_geometry_trends",
            "plots.residual_geometry_trends",
            lambda: _plot_residual_geometry_trends(image_fit_quality_df, _plot_path(run_dir, "residual_geometry_trends.pdf")),
        ),
        (
            "posterior_predictive_coverage",
            "plots.posterior_predictive_coverage",
            lambda: _plot_posterior_predictive_coverage(
                image_fit_quality_df,
                _plot_path(run_dir, "posterior_predictive_coverage.pdf"),
            ),
        ),
        ("image_plane_fit", "plots.image_plane_fit", lambda: _plot_image_plane_fit(run_dir, state, best_eval)),
        ("source_plane_scatter", "plots.source_plane_scatter", lambda: _plot_source_plane_scatter(run_dir, state, best_eval)),
        (
            "subhalo_mass_function",
            "plots.subhalo_mass_function",
            lambda: _plot_subhalo_mass_function(subhalo_df, _plot_path(run_dir, "subhalo_mass_function.pdf")),
        ),
        (
            "subhalo_radial_distribution",
            "plots.subhalo_radial_distribution",
            lambda: _plot_subhalo_radial_distribution(subhalo_df, _plot_path(run_dir, "subhalo_radial_distribution.pdf")),
        ),
        (
            "per_potential_summary",
            "plots.per_potential_summary",
            lambda: _plot_per_potential_summary(
                run_dir,
                summary_df,
                best_par_marker_values=best_par_marker_values,
                previous_stage_best_values=previous_stage_best_values,
                parameter_specs=state.parameter_specs,
            ),
        ),
        ("refresh_diagnostics", "plots.refresh_diagnostics", lambda: _plot_refresh_diagnostics(run_dir, family_df)),
        ("timing_profile", "plots.timing_profile", lambda: _plot_timing_profile(run_dir, evaluator)),
    ]
    plot_tasks.extend(
        [
            (
                "critical_arc_support_histogram",
                "plots.critical_arc_support_histogram",
                lambda: _plot_critical_arc_support_histogram(
                    image_fit_quality_df,
                    _plot_path(run_dir, "critical_arc_support_histogram.pdf"),
                ),
            ),
            (
                "critical_arc_support_phase_space",
                "plots.critical_arc_support_phase_space",
                lambda: _plot_critical_arc_support_phase_space(
                    image_fit_quality_df,
                    _plot_path(run_dir, "critical_arc_support_phase_space.pdf"),
                ),
            ),
            (
                "critical_arc_recovery_by_family",
                "plots.critical_arc_recovery_by_family",
                lambda: _plot_critical_arc_recovery_by_family(
                    image_count_recovery_df,
                    _plot_path(run_dir, "critical_arc_recovery_by_family.pdf"),
                ),
            ),
        ]
    )
    if cosmology_specs:
        plot_tasks.insert(
            1,
            (
                "cosmology_corner",
                "plots.cosmology_corner",
                lambda: _plot_cosmology_corner(
                    run_dir,
                    cosmology_samples,
                    cosmology_specs,
                    best_fit_values=cosmology_best_fit_values,
                    previous_stage_best_values=previous_stage_best_values,
                    bayes_corner_overlay=bayes_corner_overlay,
                    best_par_marker_values=best_par_marker_values,
                ),
            )
        )
    if results.ns_diagnostics:
        plot_tasks.extend(
            [
                ("ns_diagnostics", "plots.ns_diagnostics", lambda: _plot_ns_diagnostics(run_dir, results.ns_diagnostics)),
                ("ns_trace", "plots.ns_trace", lambda: _plot_ns_trace(run_dir, results.ns_diagnostics, state.parameter_specs)),
                ("ns_weight_diagnostics", "plots.ns_weight_diagnostics", lambda: _plot_ns_weight_diagnostics(run_dir, results.ns_diagnostics)),
            ]
        )
    if _has_smc_plot_data(results):
        plot_tasks.extend(
            [
                ("smc_diagnostics", "plots.smc_diagnostics", lambda: _plot_smc_diagnostics(run_dir, results)),
                ("smc_weight_diagnostics", "plots.smc_weight_diagnostics", lambda: _plot_smc_weight_diagnostics(run_dir, results)),
                (
                    "smc_corner",
                    "plots.smc_corner",
                    lambda: _plot_smc_corner(
                        run_dir,
                        results.samples,
                        state.parameter_specs,
                        results.sample_weights,
                        best_fit_values=best_fit_values,
                        previous_stage_best_values=previous_stage_best_values,
                    ),
                ),
            ]
        )
    kappa_true_fits = getattr(args, "kappa_true_fits", None)
    if kappa_true_fits is not None and str(kappa_true_fits).strip() and not bool(getattr(args, "quick_diagnostics", False)):
        plot_tasks.append(
            (
                "kappa_comparison",
                "plots.kappa_comparison",
                lambda: _plot_kappa_true_comparison(
                    run_dir,
                    evaluator,
                    best_fit,
                    str(kappa_true_fits),
                    getattr(args, "caustic_source_redshift", 9.0),
                ),
            )
        )
    if bool(getattr(args, "plot_caustics", False)) and not bool(getattr(args, "quick_diagnostics", False)):
        caustic_plot_grid_scale_arcsec = getattr(
            args,
            "caustic_plot_grid_scale_arcsec",
            CAUSTIC_PLOT_GRID_SCALE_ARCSEC,
        )
        plot_tasks.append(
            (
                "absolute_magnification",
                "plots.absolute_magnification",
                lambda: _plot_absolute_magnification(
                    run_dir,
                    evaluator,
                    best_fit,
                    caustic_plot_grid_scale_arcsec,
                    getattr(args, "caustic_source_redshift", 9.0),
                ),
            )
        )
        plot_tasks.append(
            (
                "caustic_overlay",
                "plots.caustic_overlay",
                lambda: _plot_caustic_overlay(
                    run_dir,
                    evaluator,
                    best_fit,
                    caustic_plot_grid_scale_arcsec,
                    getattr(args, "caustic_source_redshift", 9.0),
                ),
            )
        )
    _run_plot_tasks_with_progress(args, plot_tasks)
    _log(args, "[done] run summary\n" + run_summary_text.rstrip())


def _infer_stage1_artifacts_dir(args: argparse.Namespace) -> Path:
    if args.stage1_run_dir:
        candidate = Path(args.stage1_run_dir)
        return candidate / "artifacts" if candidate.name != "artifacts" else candidate
    if args.run_name:
        candidate = Path(args.output_dir) / args.run_name / "stage1_large_only" / "artifacts"
        if candidate.exists():
            return candidate
    raise ValueError("Missing stage-1 artifacts for the requested internal stage.")
