from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    import corner
except ImportError:  # pragma: no cover
    corner = None

from .model import BuildState, EvaluationResult, ParameterSpec, PosteriorResults
from .model import display_lower as _display_lower
from .model import display_upper as _display_upper
from .utils import log_message as _log
from .utils import run_logged_phase as _run_logged_phase

PROFILE_VARIANT_ORIGINAL = "original"
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
CORNER_PLOT_DPI = 220


def plot_path(root: Path, name: str) -> Path:
    """Return an output plot path, creating the output directory first."""
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def _plot_path(root: Path, name: str) -> Path:
    return plot_path(root, name)


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


def _run_summary(
    args: argparse.Namespace,
    state: BuildState,
    runtime_sec: float,
    results: PosteriorResults,
    best_loglike: float,
    evaluator: ClusterJAXEvaluator,
) -> dict[str, Any]:
    init_diagnostics = dict(results.init_diagnostics or {})
    run_name = str(getattr(args, "run_name", None) or state.run_name)
    geometry_cache = getattr(state, "geometry_cache", None)
    return {
        "run_name": run_name,
        "par_path": state.par_path,
        "fit_mode": state.fit_mode,
        "profile_variant": getattr(state, "profile_variant", PROFILE_VARIANT_ORIGINAL),
        "compact_skip_factor": float(getattr(state, "compact_skip_factor", 1.0)),
        "n_parameters": len(state.parameter_specs),
        "n_families": len(state.family_data),
        "n_images": int(sum(f.n_images for f in state.family_data)),
        "n_large_scale_parameters": int(sum(spec.component_family == "large" for spec in state.parameter_specs)),
        "n_scaling_parameters": int(sum(spec.component_family == "scaling" for spec in state.parameter_specs)),
        "n_scaling_galaxy_components": int(np.sum(state.packed_lens_spec.component_family == 1)),
        "likelihood_mode": args.likelihood_mode,
        "fit_method": str(getattr(args, "fit_method", "svi+nuts")),
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
        "warmup": args.warmup,
        "samples": args.samples,
        "chains": results.num_chains,
        "requested_chains": int(init_diagnostics.get("requested_chains", args.chains)),
        "thin": args.thin,
        "max_tree_depth": args.max_tree_depth,
        "target_accept": args.target_accept,
        "runtime_sec": runtime_sec,
        "best_loglike": best_loglike,
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
        "sample_weight_ess": float(_effective_sample_size(results.sample_weights))
        if results.sample_weights is not None and len(results.sample_weights) > 0
        else None,
        "temperature_schedule": results.temperature_schedule.tolist() if results.temperature_schedule is not None else None,
        "ess_history": results.ess_history.tolist() if results.ess_history is not None else None,
        "move_acceptance_history": results.move_acceptance_history.tolist() if results.move_acceptance_history is not None else None,
    }


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
    return subset_samples, subset_specs


def _plot_corner(
    plot_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
) -> None:
    if corner is None or not parameter_specs:
        return
    finite_samples = _finite_sample_rows(samples)
    if finite_samples.shape[0] == 0:
        return
    subset = _corner_dynamic_subset(finite_samples, parameter_specs, "corner.png")
    if subset is None:
        return
    finite_samples, subset_specs = subset
    _log(
        None,
        f"[plot:corner] path={_plot_path(plot_dir, 'corner.png')} ndim={len(subset_specs)} samples_shape={tuple(finite_samples.shape)}",
    )
    labels = [spec.name for spec in subset_specs]
    fig = corner.corner(finite_samples, labels=labels, **CORNER_PLOT_KWARGS)
    fig.savefig(_plot_path(plot_dir, "corner.png"), dpi=CORNER_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


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
) -> None:
    if corner is None or samples.size == 0 or not parameter_specs:
        return
    finite_samples = _finite_sample_rows(samples)
    if finite_samples.shape[0] == 0:
        return
    subset = _corner_dynamic_subset(finite_samples, parameter_specs, "potfile_corner.png")
    if subset is None:
        return
    finite_samples, subset_specs = subset
    _log(
        None,
        f"[plot:corner] path={_plot_path(plot_dir, 'potfile_corner.png')} ndim={len(subset_specs)} samples_shape={tuple(finite_samples.shape)}",
    )
    labels = [spec.name for spec in subset_specs]
    fig = corner.corner(finite_samples, labels=labels, **CORNER_PLOT_KWARGS)
    fig.savefig(_plot_path(plot_dir, "potfile_corner.png"), dpi=CORNER_PLOT_DPI, bbox_inches="tight")
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


def _plot_image_plane_fit(plot_dir: Path, state: BuildState, best_eval: EvaluationResult) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cmap = plt.get_cmap("tab20", len(state.family_data))
    for idx, family in enumerate(state.family_data):
        color = cmap(idx)
        pred = best_eval.family_predictions[family.family_id]
        ax.scatter(family.x_obs, family.y_obs, marker="o", color=color, label=f"{family.family_id} obs")
        if np.isfinite(pred["x_pred"]).any():
            ax.scatter(pred["x_pred"], pred["y_pred"], marker="x", color=color, s=50)
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


def _iter_contour_vertices(contour_set: Any):
    if hasattr(contour_set, "get_paths"):
        for path in contour_set.get_paths():
            verts = np.asarray(path.vertices, dtype=float)
            if verts.ndim == 2 and verts.shape[0] >= 3 and verts.shape[1] == 2:
                yield verts
    elif hasattr(contour_set, "collections"):  # pragma: no cover
        for collection in contour_set.collections:
            for path in collection.get_paths():
                verts = np.asarray(path.vertices, dtype=float)
                if verts.ndim == 2 and verts.shape[0] >= 3 and verts.shape[1] == 2:
                    yield verts
    elif hasattr(contour_set, "allsegs"):  # pragma: no cover
        for level_segments in contour_set.allsegs:
            for segment in level_segments:
                verts = np.asarray(segment, dtype=float)
                if verts.ndim == 2 and verts.shape[0] >= 3 and verts.shape[1] == 2:
                    yield verts


def _resample_curve_vertices(vertices: np.ndarray, target_points: int) -> np.ndarray | None:
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[0] < 3 or verts.shape[1] != 2:
        return None
    deltas = np.diff(verts, axis=0)
    seg_lengths = np.sqrt(np.sum(deltas**2, axis=1))
    total_length = float(np.sum(seg_lengths))
    if not np.isfinite(total_length) or total_length <= 1.0e-6:
        return None
    keep = np.concatenate([[True], seg_lengths > 1.0e-10])
    verts = verts[keep]
    if verts.shape[0] < 3:
        return None
    deltas = np.diff(verts, axis=0)
    seg_lengths = np.sqrt(np.sum(deltas**2, axis=1))
    total_length = float(np.sum(seg_lengths))
    if not np.isfinite(total_length) or total_length <= 1.0e-6:
        return None
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    n_points = max(int(target_points), int(verts.shape[0]))
    sample_s = np.linspace(0.0, cumulative[-1], n_points)
    x_resampled = np.interp(sample_s, cumulative, verts[:, 0])
    y_resampled = np.interp(sample_s, cumulative, verts[:, 1])
    return np.column_stack([x_resampled, y_resampled])


def _plot_caustic_overlay(plot_dir: Path, evaluator: ClusterJAXEvaluator, best_fit: np.ndarray, caustic_num_pix: int) -> None:
    x_all = np.concatenate([fam.x_obs for fam in evaluator.state.family_data])
    y_all = np.concatenate([fam.y_obs for fam in evaluator.state.family_data])
    center_x = float(np.mean(x_all))
    center_y = float(np.mean(y_all))
    span = max(np.ptp(x_all), np.ptp(y_all), 12.0)
    half = 0.55 * span
    contour_num_pix = max(int(caustic_num_pix), 250)
    x_grid = np.linspace(center_x - half, center_x + half, contour_num_pix)
    y_grid = np.linspace(center_y - half, center_y + half, contour_num_pix)
    xx, yy = np.meshgrid(x_grid, y_grid)
    family = evaluator.state.family_data[0]
    model = evaluator.exact_models_by_z.get(family.z_source)
    if model is None:
        model, _ = evaluator._get_exact_model_solver(family.z_source)
    packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit, dtype=jnp.float64), family.z_source)
    kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
    inv_mag = np.asarray(
        1.0
        / model.magnification(
            jnp.asarray(xx.ravel(), dtype=jnp.float64),
            jnp.asarray(yy.ravel(), dtype=jnp.float64),
            kwargs_lens,
        ),
        dtype=float,
    ).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    image_ax, source_ax = axes
    contour = image_ax.contour(xx, yy, inv_mag, levels=[0.0], colors="black", linewidths=1.0)
    cmap = plt.get_cmap("tab20", len(evaluator.state.family_data))
    for idx, fam in enumerate(evaluator.state.family_data):
        image_ax.scatter(fam.x_obs, fam.y_obs, color=cmap(idx), s=14, label=fam.family_id)
    image_ax.invert_xaxis()
    image_ax.set_xlabel("x [arcsec]")
    image_ax.set_ylabel("y [arcsec]")
    image_ax.set_title("Critical Curves + Images")

    for verts in _iter_contour_vertices(contour):
        resampled = _resample_curve_vertices(verts, target_points=max(4 * contour_num_pix, 400))
        if resampled is None or resampled.shape[0] < 3:
            continue
        beta_x, beta_y = model.ray_shooting(
            jnp.asarray(resampled[:, 0], dtype=jnp.float64),
            jnp.asarray(resampled[:, 1], dtype=jnp.float64),
            kwargs_lens,
        )
        source_ax.scatter(
            np.asarray(beta_x, dtype=float),
            np.asarray(beta_y, dtype=float),
            color="black",
            s=2,
            alpha=0.75,
            linewidths=0.0,
        )
    best_source_eval = evaluator.evaluate(best_fit, likelihood_mode="source")
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
    run_summary = _run_logged_phase(
        args,
        "plots.run_summary",
        lambda: _run_summary(args, state, runtime_sec, results, best_eval.loglike, evaluator),
    )
    scaling_specs, scaling_samples, scaling_best_fit = _run_logged_phase(
        args,
        "plots.scaling_subset",
        lambda: _scaling_parameter_subset(state.parameter_specs, results.samples, best_fit),
    )
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

    _run_logged_phase(args, "plots.corner", lambda: _plot_corner(run_dir, results.samples, state.parameter_specs))
    _run_logged_phase(
        args,
        "plots.potfile_corner",
        lambda: _plot_potfile_corner(run_dir, scaling_samples, scaling_specs),
    )
    _run_logged_phase(args, "plots.potfile_histograms", lambda: _plot_potfile_histograms(run_dir, scaling_samples, scaling_best_fit, scaling_specs))
    _run_logged_phase(
        args,
        "plots.potfile_prior_posterior",
        lambda: _plot_potfile_prior_posterior(run_dir, potfile_constraint_df, results.samples, state.parameter_specs),
    )
    _run_logged_phase(args, "plots.potfile_constraint_strength", lambda: _plot_potfile_constraint_strength(run_dir, potfile_constraint_df))
    _run_logged_phase(args, "plots.potfile_prior_shift", lambda: _plot_potfile_prior_shift(run_dir, potfile_constraint_df))
    _run_logged_phase(args, "plots.potfile_leverage_summary", lambda: _plot_potfile_leverage_summary(run_dir, potfile_constraint_df))
    _run_logged_phase(args, "plots.trace", lambda: _plot_trace(run_dir, trace_grouped_samples, trace_specs))
    _run_logged_phase(args, "plots.scaling_rank_bars", lambda: _plot_scaling_rank_bars(run_dir, evaluator.scaling_rank_df))
    _run_logged_phase(args, "plots.scaling_rank_scatter", lambda: _plot_scaling_rank_scatter(run_dir, evaluator.scaling_rank_df))
    _run_logged_phase(args, "plots.run_diagnostics", lambda: _plot_run_diagnostics(run_dir, results))
    _run_logged_phase(args, "plots.weights_logl", lambda: _plot_weights_logl(run_dir, results))
    _run_logged_phase(args, "plots.residuals_by_family", lambda: _plot_residuals_by_family(run_dir, family_df))
    _run_logged_phase(args, "plots.image_plane_fit", lambda: _plot_image_plane_fit(run_dir, state, best_eval))
    _run_logged_phase(args, "plots.source_plane_scatter", lambda: _plot_source_plane_scatter(run_dir, state, best_eval))
    _run_logged_phase(
        args,
        "plots.per_potential_summary",
        lambda: _plot_per_potential_summary(run_dir, summary_df),
    )
    _run_logged_phase(args, "plots.refresh_diagnostics", lambda: _plot_refresh_diagnostics(run_dir, family_df))
    _run_logged_phase(args, "plots.timing_profile", lambda: _plot_timing_profile(run_dir, evaluator))
    if args.plot_caustics or state.fit_mode == "large-only":
        _run_logged_phase(
            args,
            "plots.caustic_overlay",
            lambda: _plot_caustic_overlay(run_dir, evaluator, best_fit, args.caustic_num_pix),
        )


def _infer_stage1_artifacts_dir(args: argparse.Namespace) -> Path:
    if args.stage1_run_dir:
        candidate = Path(args.stage1_run_dir)
        return candidate / "artifacts" if candidate.name != "artifacts" else candidate
    if args.run_name:
        candidate = Path(args.output_dir) / args.run_name / "stage1_large_only" / "artifacts"
        if candidate.exists():
            return candidate
    raise ValueError("Missing stage-1 artifacts for the requested internal stage.")
