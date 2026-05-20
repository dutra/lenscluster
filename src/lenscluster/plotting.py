from __future__ import annotations

import argparse
import json
import math
import threading
from pathlib import Path
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from scipy.stats import norm
from skimage.measure import find_contours

try:
    import corner
except ImportError:  # pragma: no cover
    corner = None

from .model import BuildState, EvaluationResult, ParameterSpec, PosteriorResults
from .model import convert_theta_to_latent as _convert_theta_to_latent
from .model import display_lower as _display_lower
from .model import display_upper as _display_upper
from .utils import log_message as _log
from .utils import run_logged_phase as _run_logged_phase

DEFAULT_NUTS_INIT_BOUNDARY_FRAC = 0.02
DEFAULT_NUTS_INIT_JITTER_FRAC = 0.02
DEFAULT_SVI_STEPS = 2000
DEFAULT_SVI_LEARNING_RATE = 5.0e-3
CORNER_PLOT_KWARGS = {
    "show_titles": True,
    "title_fmt": ".3g",
    "quantiles": [0.16, 0.5, 0.84],
    "plot_datapoints": False,
    "fill_contours": True,
    "smooth": 1.0,
    "smooth1d": 1.0,
    "max_n_ticks": 4,
}
CORNER_PLOT_DPI = 300
CORNER_BEST_FIT_COLOR = "#d4a017"


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


def _fit_quality_chi_square_summary(
    image_fit_quality_df: pd.DataFrame | None,
    state: BuildState,
) -> dict[str, Any]:
    n_parameters = int(len(getattr(state, "parameter_specs", [])))
    n_families = int(len(getattr(state, "family_data", [])))
    n_observed_images = int(sum(int(getattr(family, "n_images", 0)) for family in getattr(state, "family_data", [])))
    explicit_source_parameters = int(
        sum(spec.component_family == "source_position" for spec in getattr(state, "parameter_specs", []))
    )
    implicit_source_parameters = 0 if explicit_source_parameters > 0 else 2 * n_families
    k_effective = int(n_parameters + implicit_source_parameters)
    n_data = int(2 * n_observed_images)
    dof = int(n_data - k_effective)
    empty = {
        "chi_square": None,
        "n_data": n_data,
        "valid_image_count": 0,
        "diagnostic_n_data": 0,
        "n_effective_parameters": k_effective,
        "implicit_source_position_parameters": implicit_source_parameters,
        "dof": dof,
        "diagnostic_dof": int(-k_effective),
        "reduced_chi_square": None,
        "aic": None,
        "bic": None,
        "image_residual_mean_arcsec": None,
        "image_residual_median_arcsec": None,
        "image_residual_max_arcsec": None,
        "covered_xy_1sigma_fraction": None,
    }
    if image_fit_quality_df is None or image_fit_quality_df.empty:
        return empty
    df = image_fit_quality_df.copy()
    required = [
        "x_model_arcsec",
        "y_model_arcsec",
        "x_obs_arcsec",
        "y_obs_arcsec",
        "image_sigma_eff_arcsec",
    ]
    if any(column not in df.columns for column in required):
        return empty
    failed = (
        df["exact_image_prediction_failed"].astype(bool).to_numpy()
        if "exact_image_prediction_failed" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    x_model = pd.to_numeric(df["x_model_arcsec"], errors="coerce").to_numpy(dtype=float)
    y_model = pd.to_numeric(df["y_model_arcsec"], errors="coerce").to_numpy(dtype=float)
    x_obs = pd.to_numeric(df["x_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    y_obs = pd.to_numeric(df["y_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    sigma = pd.to_numeric(df["image_sigma_eff_arcsec"], errors="coerce").to_numpy(dtype=float)
    valid = (~failed) & np.isfinite(x_model + y_model + x_obs + y_obs + sigma) & (sigma > 0.0)
    if not np.any(valid):
        return empty
    dx = x_model[valid] - x_obs[valid]
    dy = y_model[valid] - y_obs[valid]
    chi_square = float(np.sum((np.square(dx) + np.square(dy)) / np.square(sigma[valid])))
    residuals = np.sqrt(np.square(dx) + np.square(dy))
    valid_image_count = int(np.sum(valid))
    diagnostic_n_data = int(2 * valid_image_count)
    diagnostic_dof = int(diagnostic_n_data - k_effective)
    coverage_fraction = None
    if "covered_xy_1sigma" in df.columns:
        coverage_values = df.loc[valid, "covered_xy_1sigma"].astype(bool).to_numpy()
        coverage_fraction = float(np.mean(coverage_values)) if coverage_values.size else None
    return {
        "chi_square": chi_square,
        "n_data": n_data,
        "valid_image_count": valid_image_count,
        "diagnostic_n_data": diagnostic_n_data,
        "n_effective_parameters": k_effective,
        "implicit_source_position_parameters": implicit_source_parameters,
        "dof": dof,
        "diagnostic_dof": diagnostic_dof,
        "reduced_chi_square": float(chi_square / dof) if dof > 0 else None,
        "aic": float(chi_square + 2.0 * k_effective),
        "bic": float(chi_square + k_effective * math.log(n_data)) if n_data > 0 else None,
        "image_residual_mean_arcsec": float(np.mean(residuals)),
        "image_residual_median_arcsec": float(np.median(residuals)),
        "image_residual_max_arcsec": float(np.max(residuals)),
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


def _run_summary(
    args: argparse.Namespace,
    state: BuildState,
    runtime_sec: float,
    results: PosteriorResults,
    best_loglike: float,
    evaluator: ClusterJAXEvaluator,
    image_fit_quality_df: pd.DataFrame | None = None,
    family_df: pd.DataFrame | None = None,
    used_exact_validation: bool | None = None,
) -> dict[str, Any]:
    init_diagnostics = dict(results.init_diagnostics or {})
    run_name = str(getattr(args, "run_name", None) or state.run_name)
    geometry_cache = getattr(state, "geometry_cache", None)
    sample_likelihood_mode = str(getattr(args, "sample_likelihood_mode", "source"))
    source_redshifts = np.asarray([float(family.z_source) for family in state.family_data], dtype=float)
    finite_source_redshifts = source_redshifts[np.isfinite(source_redshifts)]
    lens_redshift = getattr(state, "z_lens", None)
    chi_square_summary = _fit_quality_chi_square_summary(image_fit_quality_df, state)
    chain_summary = _chain_diagnostics_summary(results, state.parameter_specs)
    exact_family_count = None
    failed_or_missing_exact_count = None
    if family_df is not None and not family_df.empty and "exact_image_rms_arcsec" in family_df.columns:
        exact_values = pd.to_numeric(family_df["exact_image_rms_arcsec"], errors="coerce").to_numpy(dtype=float)
        exact_family_count = int(np.sum(np.isfinite(exact_values)))
        failed_or_missing_exact_count = int(len(exact_values) - exact_family_count)

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
        "validation_approx": args.validation_approx,
        "active_scaling_galaxies": list(evaluator.active_scaling_galaxies_by_potfile),
        "active_scaling_components": int(len(evaluator.active_scaling_component_indices)),
        "inactive_scaling_components": int(len(evaluator.inactive_scaling_component_indices)),
        "requested_active_scaling_by_potfile": evaluator.requested_active_scaling_by_potfile,
        "actual_active_scaling_by_potfile": evaluator.actual_active_scaling_by_potfile,
        "total_scaling_by_potfile": evaluator.total_scaling_by_potfile,
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
        "max_tree_depth": args.max_tree_depth,
        "target_accept": args.target_accept,
        "runtime_sec": runtime_sec,
        "best_loglike": best_loglike,
        "used_exact_validation": bool(used_exact_validation) if used_exact_validation is not None else None,
        "exact_family_count": exact_family_count,
        "failed_or_missing_exact": failed_or_missing_exact_count,
        "seed": args.seed,
        "packed_fast_path": True,
        "uses_potfile_scaling": bool(state.potfiles and state.fit_mode in {"small-only", "joint"}),
        "surrogate_enabled": bool(evaluator.surrogate_enabled),
        "approximate_eval_count": int(evaluator.approximate_eval_count),
        "full_refresh_count": int(evaluator.full_refresh_count),
        "validation_fallback_count": int(evaluator.validation_fallback_count),
        "invalid_state_rejection_count": int(evaluator.invalid_state_rejection_count),
        "invalid_state_reason_counts": {key: int(value) for key, value in evaluator.invalid_state_reason_counts.items()},
        "stage2_large_scale_priors": {
            spec.sample_name: {"mean": spec.mean, "std": spec.std}
            for spec in state.parameter_specs
            if state.fit_mode == "small-only" and spec.component_family == "large" and spec.prior_kind == "normal"
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
                    >= (2 ** int(getattr(args, "max_tree_depth", 10)) - 1)
                )
            )
            if results.num_steps.size
            else None
        ),
        "sample_weight_ess": float(_effective_sample_size(results.sample_weights))
        if results.sample_weights is not None and len(results.sample_weights) > 0
        else None,
        "temperature_schedule": results.temperature_schedule.tolist() if results.temperature_schedule is not None else None,
        "ess_history": results.ess_history.tolist() if results.ess_history is not None else None,
        "move_acceptance_history": results.move_acceptance_history.tolist() if results.move_acceptance_history is not None else None,
        **chi_square_summary,
        **chain_summary,
    }
    image_scatter_indices = [
        idx for idx, spec in enumerate(state.parameter_specs) if spec.component_family == "image_scatter"
    ]
    if image_scatter_indices:
        values = np.asarray(results.samples, dtype=float)[:, image_scatter_indices[0]]
        finite = values[np.isfinite(values)]
        if finite.size:
            spec = state.parameter_specs[image_scatter_indices[0]]
            upper_value = getattr(spec, "physical_upper", None)
            if upper_value is None:
                upper_value = getattr(spec, "upper", np.nan)
            upper = float(upper_value)
            q16, q50, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
            summary["image_sigma_int_posterior"] = {
                "q16": float(q16),
                "median": float(q50),
                "q84": float(q84),
                "upper_arcsec": upper,
                "near_upper_bound": bool(np.isfinite(upper) and q84 >= 0.9 * upper),
            }
    return summary


def _format_run_summary_text(summary: dict[str, Any]) -> str:
    source_range = "na"
    if summary.get("z_source_min") is not None and summary.get("z_source_max") is not None:
        source_range = f"{_metric_text(summary.get('z_source_min'))}-{_metric_text(summary.get('z_source_max'))}"
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
                    ("validation mode", summary.get("validation_approx")),
                    ("quick diagnostics", summary.get("quick_diagnostics")),
                    ("used exact validation", summary.get("used_exact_validation")),
                    ("exact families", summary.get("exact_family_count")),
                    ("missing exact families", summary.get("failed_or_missing_exact")),
                    ("best log likelihood", summary.get("best_loglike")),
                ]
            )
        ),
        "",
        "Quality Of Fit",
        *(
            _key_value_lines(
                [
                    ("chi_square", summary.get("chi_square")),
                    ("dof", summary.get("dof")),
                    ("reduced_chi_square", summary.get("reduced_chi_square")),
                    ("AIC", summary.get("aic")),
                    ("BIC", summary.get("bic")),
                    ("valid image count", summary.get("valid_image_count")),
                    ("diagnostic data points", summary.get("diagnostic_n_data")),
                    ("diagnostic dof", summary.get("diagnostic_dof")),
                    ("effective parameters", summary.get("n_effective_parameters")),
                    ("implicit source parameters", summary.get("implicit_source_position_parameters")),
                    ("mean image residual arcsec", summary.get("image_residual_mean_arcsec")),
                    ("median image residual arcsec", summary.get("image_residual_median_arcsec")),
                    ("max image residual arcsec", summary.get("image_residual_max_arcsec")),
                    ("1sigma xy coverage fraction", summary.get("covered_xy_1sigma_fraction")),
                    ("ESS min", summary.get("ess_min")),
                    ("ESS median", summary.get("ess_median")),
                    ("ESS worst parameter", summary.get("ess_worst_parameter")),
                    ("Rhat max", summary.get("rhat_max")),
                    ("Rhat median", summary.get("rhat_median")),
                    ("Rhat worst parameter", summary.get("rhat_worst_parameter")),
                    ("accept prob mean", summary.get("accept_prob_mean")),
                    ("divergence count", summary.get("divergence_count")),
                    ("mean num steps", summary.get("mean_num_steps")),
                ]
            )
        ),
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
        ("chi2", "chi_square"),
        ("dof", "dof"),
        ("chi2_red", "reduced_chi_square"),
        ("AIC", "aic"),
        ("BIC", "bic"),
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
) -> None:
    if corner is None or not parameter_specs:
        return
    corner_samples, corner_specs = _corner_without_source_positions(samples, parameter_specs, "corner.pdf")
    if not corner_specs:
        return
    finite_samples = _finite_sample_rows(corner_samples)
    if finite_samples.shape[0] == 0:
        return
    subset = _corner_dynamic_subset(finite_samples, corner_specs, "corner.pdf")
    if subset is None:
        return
    finite_samples, subset_specs = subset
    _log(
        None,
        f"[plot:corner] path={_plot_path(plot_dir, 'corner.pdf')} ndim={len(subset_specs)} samples_shape={tuple(finite_samples.shape)}",
    )
    labels = [spec.name for spec in subset_specs]
    truths = _corner_values_for_specs(subset_specs, truth_values) if truth_values else None
    fig = corner.corner(finite_samples, labels=labels, truths=truths, **CORNER_PLOT_KWARGS)
    _overplot_corner_best_fit(fig, subset_specs, best_fit_values)
    fig.savefig(_plot_path(plot_dir, "corner.pdf"), dpi=CORNER_PLOT_DPI, bbox_inches="tight")
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
    line_xs = [float(value) if np.isfinite(value) else None for value in xs]
    point_xs = [[float(value) if np.isfinite(value) else np.nan for value in xs]]
    corner.overplot_lines(fig, line_xs, color=CORNER_BEST_FIT_COLOR)
    corner.overplot_points(fig, point_xs, marker="s", color=CORNER_BEST_FIT_COLOR)


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
    fig = corner.corner(finite_samples, labels=labels, truths=truths, **CORNER_PLOT_KWARGS)
    _overplot_corner_best_fit(fig, subset_specs, best_fit_values)
    fig.savefig(_plot_path(plot_dir, "potfile_corner.pdf"), dpi=CORNER_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_cosmology_corner(
    plot_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    truth_values: dict[str, float] | None = None,
    best_fit_values: dict[str, float] | None = None,
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
    fig = corner.corner(finite_samples, labels=labels, truths=truths, **CORNER_PLOT_KWARGS)
    _overplot_corner_best_fit(fig, subset_specs, best_fit_values)
    fig.savefig(_plot_path(plot_dir, "cosmology_corner.pdf"), dpi=CORNER_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_potfile_histograms(
    plot_dir: Path,
    samples: np.ndarray,
    best_fit: np.ndarray,
    parameter_specs: list[ParameterSpec],
) -> None:
    if samples.size == 0 or not parameter_specs:
        return
    nrows = len(parameter_specs)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, max(4, 2.0 * nrows)), sharex=False)
    if nrows == 1:
        axes = [axes]
    for idx, (ax, spec) in enumerate(zip(axes, parameter_specs)):
        values = np.asarray(samples[:, idx], dtype=float)
        ax.hist(values, bins=min(40, max(10, int(np.sqrt(len(values))))), color="tab:blue", alpha=0.75)
        ax.axvline(float(np.median(values)), color="tab:orange", linewidth=1.5, label="median")
        ax.axvline(float(best_fit[idx]), color="tab:red", linewidth=1.5, linestyle="--", label="best fit")
        ax.set_title(spec.name)
        ax.set_ylabel("count")
    axes[-1].set_xlabel("parameter value")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "potfile_histograms.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_trace(plot_dir: Path, grouped_samples: np.ndarray | None, parameter_specs: list[ParameterSpec]) -> None:
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
    fig.savefig(_plot_path(plot_dir, "trace_plot.png"), dpi=180, bbox_inches="tight")
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
        pred = best_eval.family_predictions[family.family_id]
        ax.scatter(family.x_obs, family.y_obs, marker="x", color=color, label=f"{family.family_id} obs")
        if np.isfinite(pred["x_pred"]).any():
            ax.scatter(pred["x_pred"], pred["y_pred"], marker="o", color=color, s=36, alpha=0.65)
            for x0, y0, x1, y1 in zip(family.x_obs, family.y_obs, pred["x_pred"], pred["y_pred"]):
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


def _clone_fit_quality_evaluator(evaluator: Any, args: argparse.Namespace) -> Any:
    from .cluster_solver import (  # local import avoids a module import cycle at plotting import time
        ClusterJAXEvaluator,
        DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        DEFAULT_ACTIVE_SCALING_MIN,
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
        validate_top_k_families=0,
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
        validation_approx=str(getattr(args, "validation_approx", getattr(evaluator, "validation_approx", "exact"))),
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
        exact_image_solver=str(getattr(args, "exact_image_solver", getattr(evaluator, "exact_image_solver", "auto"))),
        quick_diagnostics=bool(getattr(args, "quick_diagnostics", getattr(evaluator, "quick_diagnostics", False))),
    )


def _fit_quality_prediction_for_family_latent(
    evaluator: Any,
    family: Any,
    params_latent: np.ndarray,
    image_sigma_int: float,
    covariance_floor: float,
    quick_diagnostics: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    image_rows: list[dict[str, Any]] = []
    magnification_rows: list[dict[str, Any]] = []
    params_latent = np.asarray(params_latent, dtype=float)
    n_images = int(family.n_images)
    x_pred = np.full(n_images, np.nan, dtype=float)
    y_pred = np.full(n_images, np.nan, dtype=float)
    image_failed = True
    if not quick_diagnostics:
        try:
            exact_prediction = evaluator._exact_family_prediction(params_latent, family)
            if exact_prediction is not None:
                x_exact, y_exact, _exact_rms = exact_prediction
                x_exact = np.asarray(x_exact, dtype=float)
                y_exact = np.asarray(y_exact, dtype=float)
                if x_exact.shape == (n_images,) and y_exact.shape == (n_images,):
                    x_pred = x_exact
                    y_pred = y_exact
                    image_failed = False
        except Exception:
            image_failed = True

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
    for label, x_obs, y_obs, x_model, y_model, mu_value in zip(
        family.image_labels,
        family.x_obs,
        family.y_obs,
        x_pred,
        y_pred,
        mu,
    ):
        residual = (
            math.hypot(float(x_model) - float(x_obs), float(y_model) - float(y_obs))
            if np.isfinite(float(x_model) + float(y_model))
            else np.nan
        )
        common = {
            "family_id": str(family.family_id),
            "image_label": str(label),
            "x_obs_arcsec": float(x_obs),
            "y_obs_arcsec": float(y_obs),
            "z_source": _finite_or(getattr(family, "z_source", np.nan)),
            "sigma_arcsec": sigma_arcsec,
            "image_sigma_int_arcsec": image_sigma_int,
            "image_sigma_eff_arcsec": sigma_eff,
            "radius_arcsec": float(math.hypot(float(x_obs), float(y_obs))),
            "angle_deg": float(np.degrees(np.arctan2(float(y_obs), float(x_obs)))),
        }
        image_rows.append(
            {
                **common,
                "x_model_arcsec": float(x_model),
                "y_model_arcsec": float(y_model),
                "image_residual_arcsec": float(residual),
                "exact_image_prediction_failed": bool(image_failed),
            }
        )
        magnification_rows.append(
            {
                **common,
                "magnification_model": float(mu_value),
                "magnification_prediction_failed": bool(magnification_failed),
            }
        )
    return {"image_rows": image_rows, "magnification_rows": magnification_rows}


def _fit_quality_prediction_for_latent(
    evaluator: Any,
    state: BuildState,
    params_latent: np.ndarray,
    quick_diagnostics: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    params_latent = np.asarray(params_latent, dtype=float)
    image_sigma_int = _fit_quality_image_sigma_int(evaluator, params_latent)
    covariance_floor = _finite_or(getattr(evaluator, "source_plane_covariance_floor", 0.0), 0.0)
    prediction: dict[str, list[dict[str, Any]]] = {"image_rows": [], "magnification_rows": []}
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
    return prediction


def _fit_quality_family_cost_metadata(family: Any) -> dict[str, Any]:
    sigma_arcsec = _finite_or(getattr(family, "sigma_arcsec", np.nan))
    min_distance = max(0.02, sigma_arcsec / 5.0) if np.isfinite(sigma_arcsec) else 0.02
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
    worker_count = max(1, int(getattr(args, "fit_quality_workers", 1)))
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
            combined = {"image_rows": [], "magnification_rows": []}
            for family_prediction in sample_predictions:
                if family_prediction is None:
                    continue
                combined["image_rows"].extend(family_prediction["image_rows"])
                combined["magnification_rows"].extend(family_prediction["magnification_rows"])
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    best_prediction = all_predictions[0] if all_predictions else {"image_rows": [], "magnification_rows": []}
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
    return pd.DataFrame(image_rows), pd.DataFrame(magnification_rows)


def _plot_image_recovery_fit_quality(image_df: pd.DataFrame, path: Path) -> None:
    if image_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    family_ids = (
        image_df["family_id"].astype(str).to_numpy()
        if "family_id" in image_df.columns
        else np.full(len(image_df), "", dtype=object)
    )
    colors = _family_color_map(family_ids)
    for family_index, family_id in enumerate(colors):
        family_mask = family_ids == family_id
        ax.scatter(
            image_df.loc[family_mask, "x_obs_arcsec"],
            image_df.loc[family_mask, "y_obs_arcsec"],
            marker="x",
            color=colors[family_id],
            s=30,
            label="observed" if family_index == 0 else None,
        )
    x_best = image_df["x_model_arcsec"].to_numpy(dtype=float)
    y_best = image_df["y_model_arcsec"].to_numpy(dtype=float)
    x_model = image_df["x_model_q50"].to_numpy(dtype=float)
    y_model = image_df["y_model_q50"].to_numpy(dtype=float)
    x_model = np.where(np.isfinite(x_model), x_model, x_best)
    y_model = np.where(np.isfinite(y_model), y_model, y_best)
    finite_model = np.isfinite(x_model) & np.isfinite(y_model)
    if finite_model.any():
        x16 = image_df["x_model_q16"].to_numpy(dtype=float)
        x84 = image_df["x_model_q84"].to_numpy(dtype=float)
        y16 = image_df["y_model_q16"].to_numpy(dtype=float)
        y84 = image_df["y_model_q84"].to_numpy(dtype=float)
        xerr = [
            np.where(np.isfinite(x16), np.maximum(0.0, x_model - x16), 0.0)[finite_model],
            np.where(np.isfinite(x84), np.maximum(0.0, x84 - x_model), 0.0)[finite_model],
        ]
        yerr = [
            np.where(np.isfinite(y16), np.maximum(0.0, y_model - y16), 0.0)[finite_model],
            np.where(np.isfinite(y84), np.maximum(0.0, y84 - y_model), 0.0)[finite_model],
        ]
        model_label_added = False
        for family_id in colors:
            family_finite = family_ids[finite_model] == family_id
            if not np.any(family_finite):
                continue
            color = colors[family_id]
            ax.errorbar(
                x_model[finite_model][family_finite],
                y_model[finite_model][family_finite],
                xerr=[xerr[0][family_finite], xerr[1][family_finite]],
                yerr=[yerr[0][family_finite], yerr[1][family_finite]],
                fmt="o",
                color=_color_with_alpha(color, 0.65),
                ecolor=_color_with_alpha(color, 0.35),
                markersize=4,
                label="model posterior" if not model_label_added else None,
            )
            model_label_added = True
    for row, x_fit, y_fit in zip(image_df.itertuples(index=False), x_model, y_model):
        if np.isfinite(x_fit) and np.isfinite(y_fit):
            ax.plot([row.x_obs_arcsec, x_fit], [row.y_obs_arcsec, y_fit], color="0.6", lw=0.8)
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
        )
    axes[1].set_xlabel("image index")
    axes[1].set_ylabel("image residual [arcsec]")
    axes[1].set_title("Image Residuals")
    if len(image_df) <= 40:
        axes[1].set_xticks(x_index)
        axes[1].set_xticklabels(image_df["image_label"].astype(str), rotation=90, fontsize=7)
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
) -> None:
    if summary_df.empty:
        return
    nrows = len(summary_df)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, max(4, 1.4 * nrows)), sharex=False)
    if nrows == 1:
        axes = [axes]
    for ax, row in zip(axes, summary_df.itertuples(index=False)):
        ax.hlines(1, row.p16, row.p84, linewidth=4, color="tab:blue")
        ax.scatter([row.median], [1], color="tab:blue", s=35, label="median")
        ax.scatter([row.map], [1], color="tab:red", marker="x", s=50, label="best fit")
        if np.isfinite(row.lower) and np.isfinite(row.upper):
            x_min = float(row.lower)
            x_max = float(row.upper)
        else:
            width = max(float(row.std), 0.5 * abs(float(row.p84) - float(row.p16)), 1.0e-3)
            x_min = float(min(row.p16, row.median, row.map) - 2.0 * width)
            x_max = float(max(row.p84, row.median, row.map) + 2.0 * width)
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


def _tangential_critical_curve_caustics(
    lens_model: Any,
    kwargs_lens: list[dict[str, float]],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
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
    lambda_tan = 1.0 - kappa - np.hypot(gamma1, gamma2)
    contours: list[dict[str, np.ndarray]] = []
    pixel_x = np.arange(x_values.size, dtype=float)
    pixel_y = np.arange(y_values.size, dtype=float)
    for vertices in find_contours(lambda_tan, 0.0):
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
                "critical_x": crit_x,
                "critical_y": crit_y,
                "caustic_x": beta_x,
                "caustic_y": beta_y,
            }
        )
    return contours


def _plot_caustic_overlay(
    plot_dir: Path,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    caustic_num_pix: int,
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
    x_all = np.concatenate([fam.x_obs for fam in evaluator.state.family_data])
    y_all = np.concatenate([fam.y_obs for fam in evaluator.state.family_data])
    center_x = float(np.mean(x_all))
    center_y = float(np.mean(y_all))
    span = max(np.ptp(x_all), np.ptp(y_all), 12.0)
    half = 0.55 * span
    contour_num_pix = max(int(caustic_num_pix), 250)
    x_grid = np.linspace(center_x - half, center_x + half, contour_num_pix)
    y_grid = np.linspace(center_y - half, center_y + half, contour_num_pix)
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
    image_fit_quality_df, model_magnification_df = _run_logged_phase(
        args,
        "plots.fit_quality_tables",
        lambda: _fit_quality_tables(state, evaluator, best_fit, results, args),
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
            family_df=family_df,
            used_exact_validation=getattr(best_eval, "used_exact_validation", None),
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
    best_fit_values = _best_fit_values_for_specs(state.parameter_specs, best_fit)
    scaling_best_fit_values = _best_fit_values_for_specs(scaling_specs, scaling_best_fit)
    cosmology_best_fit_values = _best_fit_values_for_specs(cosmology_specs, cosmology_best_fit)
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

    _run_logged_phase(args, "plots.write_potential_summary_csv", lambda: summary_df.to_csv(tables_dir / "potential_summary.csv", index=False))
    _run_logged_phase(args, "plots.write_family_diagnostics_csv", lambda: family_df.to_csv(tables_dir / "family_diagnostics.csv", index=False))
    _run_logged_phase(
        args,
        "plots.write_image_fit_quality_csv",
        lambda: image_fit_quality_df.to_csv(tables_dir / "image_fit_quality.csv", index=False),
    )
    _run_logged_phase(
        args,
        "plots.write_model_magnification_csv",
        lambda: model_magnification_df.to_csv(tables_dir / "model_magnification.csv", index=False),
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
            lambda: _plot_corner(run_dir, results.samples, state.parameter_specs, best_fit_values=best_fit_values),
        ),
        (
            "potfile_corner",
            "plots.potfile_corner",
            lambda: _plot_potfile_corner(
                run_dir,
                scaling_samples,
                scaling_specs,
                best_fit_values=scaling_best_fit_values,
            ),
        ),
        (
            "potfile_histograms",
            "plots.potfile_histograms",
            lambda: _plot_potfile_histograms(run_dir, scaling_samples, scaling_best_fit, scaling_specs),
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
        ("residuals_by_family", "plots.residuals_by_family", lambda: _plot_residuals_by_family(run_dir, family_df)),
        (
            "source_plane_residual_histogram",
            "plots.source_plane_residual_histogram",
            lambda: _plot_source_plane_residual_histogram(run_dir, state, best_eval),
        ),
        (
            "image_recovery",
            "plots.image_recovery",
            lambda: _plot_image_recovery_fit_quality(image_fit_quality_df, _plot_path(run_dir, "image_recovery.pdf")),
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
        (
            "exact_vs_approx_prediction_error",
            "plots.exact_vs_approx_prediction_error",
            lambda: _plot_exact_vs_approx_prediction_error(
                family_df,
                _plot_path(run_dir, "exact_vs_approx_prediction_error.pdf"),
            ),
        ),
        ("image_plane_fit", "plots.image_plane_fit", lambda: _plot_image_plane_fit(run_dir, state, best_eval)),
        ("source_plane_scatter", "plots.source_plane_scatter", lambda: _plot_source_plane_scatter(run_dir, state, best_eval)),
        ("per_potential_summary", "plots.per_potential_summary", lambda: _plot_per_potential_summary(run_dir, summary_df)),
        ("refresh_diagnostics", "plots.refresh_diagnostics", lambda: _plot_refresh_diagnostics(run_dir, family_df)),
        ("timing_profile", "plots.timing_profile", lambda: _plot_timing_profile(run_dir, evaluator)),
    ]
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
    if bool(getattr(args, "plot_caustics", False)) and not bool(getattr(args, "quick_diagnostics", False)):
        plot_tasks.append(
            (
                "caustic_overlay",
                "plots.caustic_overlay",
                lambda: _plot_caustic_overlay(
                    run_dir,
                    evaluator,
                    best_fit,
                    args.caustic_num_pix,
                    getattr(args, "caustic_source_redshift", 7.0),
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
