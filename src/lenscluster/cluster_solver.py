from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import re
import time
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.optim as numpyro_optim
import pandas as pd
import h5py
from astropy import constants as astro_const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, LambdaCDM, wCDM
from jax import config as jax_config
from matplotlib import use as matplotlib_use
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import unconstrain_fn
from scipy.optimize import linear_sum_assignment
from lenstronomy.LensModel.lens_model import LensModel as NPLensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver as NPLensEquationSolver

# Configure matplotlib before pyplot import.
_MPLCONFIGDIR = os.environ.get("MPLCONFIGDIR")
if not _MPLCONFIGDIR:
    _MPLCONFIGDIR = f"/tmp/mpl_cluster_solver_{os.getpid()}"
    os.environ["MPLCONFIGDIR"] = _MPLCONFIGDIR
Path(_MPLCONFIGDIR).mkdir(parents=True, exist_ok=True)
matplotlib_use("Agg")

jax_config.update("jax_enable_x64", True)

from jaxtronomy.LensModel.lens_model_bulk import LensModelBulk
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.Util import param_util

from .lenstool_parser import load_best_par
from .model import (
    BinData,
    BuildState,
    ChainSeed,
    EvaluationResult,
    FamilyData,
    FamilyValidationCache,
    GeometryCache,
    NUTSInitialization,
    PackedLensSpec,
    ParameterSpec,
    PosteriorResults,
    Stage1PriorSummary,
    SurrogateBinCache,
    apply_parameter_transforms_jax as _apply_parameter_transforms_jax,
    convert_sample_matrix_to_latent as _convert_sample_matrix_to_latent,
    convert_sample_matrix_to_physical as _convert_sample_matrix_to_physical,
    convert_theta_to_latent as _convert_theta_to_latent,
    convert_theta_to_physical as _convert_theta_to_physical,
    display_lower as _display_lower,
    display_upper as _display_upper,
    latent_jax_to_physical as _latent_to_physical_jax,
    latent_to_physical as _latent_to_physical_numpy,
    physical_to_latent as _physical_to_latent_numpy,
    positive_lognormal_parameters as _positive_lognormal_parameters,
)
from .plotting import _generate_plots_and_tables
from .utils import (
    close_debug_log as _close_debug_log,
    configure_debug_log as _configure_debug_log,
    fmt_seconds as _fmt_seconds,
    log_exception as _log_exception,
    log_message as _log,
    make_run_name as _make_run_name,
    parse_bool_env as _parse_bool_env,
    run_logged_phase as _run_logged_phase,
)


SUPPORTED_PROFILES = {81, 14}
DP_IE_PROFILE = 81
SHEAR_PROFILE = 14
DEFAULT_MATCH_TOLERANCE = 1.5
DEFAULT_SEARCH_PADDING = 8.0
DEFAULT_Z_BIN_TOL = 0.02
DEFAULT_WARMUP = 300
DEFAULT_SAMPLES = 500
DEFAULT_TARGET_ACCEPT = 0.85
DEFAULT_MAX_TREE_DEPTH = 10
DEFAULT_ACTIVE_SCALING_GALAXIES = 64
DEFAULT_REFRESH_EVERY = 250
DEFAULT_REFRESH_PARAM_DRIFT_FRAC = 0.25
DEFAULT_VALIDATION_RMS_FACTOR = 1.5
DEFAULT_NUTS_INIT_BOUNDARY_FRAC = 0.02
DEFAULT_NUTS_INIT_JITTER_FRAC = 0.02
DEFAULT_SVI_STEPS = 2000
DEFAULT_SVI_LEARNING_RATE = 5.0e-3
DEFAULT_SAMPLER = "numpyro_nuts"
DEFAULT_SOURCE_SIGMA_INT_LOWER_ARCSEC = 1.0e-3
DEFAULT_SOURCE_SIGMA_INT_UPPER_ARCSEC = 2.0
SAFE_SCALING_EXPONENT_ABS_MIN = 1.0e-3
SAFE_RADIUS_MARGIN_ARCSEC = 1.0e-3
BAD_LOG_LIKE = -1.0e30
PROFILE_VARIANT_ORIGINAL = "original"
PROFILE_VARIANT_COMPACT = "compact"
ORIGINAL_DPIE_PROFILE_NAME = "PJAFFE_ELLIPSE_POTENTIAL"
COMPACT_DPIE_PROFILE_NAME = "PJAFFE_ELLIPSE_POTENTIAL_COMPACT"
COMPACT_PROFILE_NAMES = {
    "PJAFFE_COMPACT",
    "PJAFFE_ELLIPSE_POTENTIAL_COMPACT",
}
CORNER_PLOT_KWARGS = {
    "bins": 32,
    "hist_bin_factor": 2,
    "smooth": 1.0,
    "smooth1d": 1.0,
    "show_titles": True,
    "quantiles": [0.16, 0.5, 0.84],
    "title_fmt": ".3g",
    "title_kwargs": {"fontsize": 8},
    "label_kwargs": {"fontsize": 8},
}
CORNER_PLOT_DPI = 220
INVALID_STATE_REASON_NAMES = (
    "nonfinite_sigma0",
    "nonpositive_ra",
    "rs_not_greater_than_ra",
    "nonpositive_vdisp",
    "bad_scaling_exponent",
    "nonfinite_centers",
    "nonfinite_shear",
    "nonfinite_shape",
    "nonfinite_cosmology_factor",
)


def _jax_profile_kwargs_list(lens_model_list: list[str], compact_skip_factor: float) -> list[dict[str, float]]:
    return [
        {"compact_skip_factor": float(compact_skip_factor)} if lens_type in COMPACT_PROFILE_NAMES else {}
        for lens_type in lens_model_list
    ]


def _state_with_run_name(state: BuildState, run_name: str) -> BuildState:
    if is_dataclass(state):
        return replace(state, run_name=run_name)
    cloned = copy.copy(state)
    setattr(cloned, "run_name", run_name)
    return cloned


def _check_physical_sample_matrix(
    samples: np.ndarray | None,
    parameter_specs: list[ParameterSpec],
    *,
    context: str,
) -> None:
    if samples is None:
        return
    array = np.asarray(samples, dtype=float)
    if array.size == 0:
        return
    suspicious: list[str] = []
    for idx, spec in enumerate(parameter_specs):
        if spec.transform_kind not in {"log_positive", "log_offset_positive"}:
            continue
        values = np.asarray(array[..., idx], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        reference = max(
            abs(float(spec.physical_mean or 0.0)),
            abs(float(spec.physical_upper or 0.0)),
            abs(float(spec.transform_offset)),
            1.0,
        )
        max_value = float(np.max(np.abs(finite)))
        if max_value > max(1.0e12 * reference, 1.0e12):
            suspicious.append(f"{spec.name}:max={max_value:.3e}")
    if not suspicious:
        return
    message = (
        f"[posterior:guard] suspiciously large converted values in {context}; "
        "possible double latent->physical conversion for transformed parameters: "
        + ", ".join(suspicious)
    )
    if _parse_bool_env("CLUSTER_SOLVER_STRICT_PHYSICAL_GUARDS"):
        raise ValueError(message)
    _log(None, message)


def _posterior_results_to_physical(results: PosteriorResults, parameter_specs: list[ParameterSpec]) -> PosteriorResults:
    # PosteriorResults are kept in latent space during inference and converted once here for output/reporting.
    grouped_samples = None
    if results.grouped_samples is not None:
        grouped_samples = _convert_sample_matrix_to_physical(results.grouped_samples, parameter_specs)
    init_diagnostics = dict(results.init_diagnostics or {})
    converted = replace(
        results,
        samples=_convert_sample_matrix_to_physical(results.samples, parameter_specs),
        grouped_samples=grouped_samples,
        init_diagnostics=init_diagnostics,
    )
    _check_physical_sample_matrix(converted.samples, parameter_specs, context="posterior.samples")
    _check_physical_sample_matrix(converted.grouped_samples, parameter_specs, context="posterior.grouped_samples")
    return converted


def _loaded_posterior_arrays_need_physical_conversion(
    arrays: dict[str, np.ndarray],
    parameter_specs: list[ParameterSpec],
    init_diagnostics: dict[str, Any] | None,
) -> bool:
    if not parameter_specs or "samples" not in arrays or "best_fit" not in arrays:
        return False
    diagnostics = dict(init_diagnostics or {})
    if not bool(diagnostics.get("legacy_latent_saved_run", False)):
        return False
    samples = np.asarray(arrays["samples"], dtype=float)
    best_fit = np.asarray(arrays["best_fit"], dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0 or best_fit.shape[0] != len(parameter_specs):
        return False
    converted_better = 0
    transformed_count = 0
    for idx, spec in enumerate(parameter_specs):
        if spec.transform_kind not in {"log_positive", "log_offset_positive"}:
            continue
        transformed_count += 1
        sample_median = float(np.median(samples[:, idx]))
        best_fit_value = float(best_fit[idx])
        raw_delta = abs(sample_median - best_fit_value)
        converted_delta = abs(_latent_to_physical_numpy(sample_median, spec) - best_fit_value)
        if np.isfinite(converted_delta) and converted_delta + 1.0e-9 < raw_delta:
            converted_better += 1
    return transformed_count > 0 and converted_better == transformed_count


def _maybe_convert_loaded_posterior_arrays_to_physical(
    arrays: dict[str, np.ndarray],
    parameter_specs: list[ParameterSpec],
    init_diagnostics: dict[str, Any] | None,
) -> tuple[dict[str, np.ndarray], bool]:
    normalized = {key: np.asarray(value) for key, value in arrays.items()}
    if not _loaded_posterior_arrays_need_physical_conversion(normalized, parameter_specs, init_diagnostics):
        return normalized, False
    converted = dict(normalized)
    converted["samples"] = _convert_sample_matrix_to_physical(np.asarray(normalized["samples"], dtype=float), parameter_specs)
    if "grouped_samples" in normalized:
        converted["grouped_samples"] = _convert_sample_matrix_to_physical(
            np.asarray(normalized["grouped_samples"], dtype=float),
            parameter_specs,
        )
    return converted, True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster dPIE solver with JAXtronomy + NumPyro.")
    parser.add_argument("--par-path", required=False, help="Path to input_a_sl.par")
    parser.add_argument("--output-dir", default="plots", help="Base output directory")
    parser.add_argument("--run-name", default=None, help="Optional run name")
    parser.add_argument(
        "--pos-sigma-arcsec",
        type=float,
        default=None,
        help="Override positional uncertainty in arcsec",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Debug mode: skip exact validation after sampling to isolate post-sampling memory spikes.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Debug mode: skip plot/table generation after artifacts are saved to isolate plotting memory spikes.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress stage logs while keeping NUTS progress bars.")
    parser.add_argument(
        "--sampling-engine",
        choices=("full", "refreshing_surrogate"),
        default="refreshing_surrogate",
        help="Use the exact full source-plane likelihood or a first-order surrogate around a refreshed reference point.",
    )
    parser.add_argument(
        "--active-scaling-galaxies",
        type=int,
        nargs="+",
        default=None,
        help="Per-potfile counts of most-important scaling-law galaxies to keep exact in surrogate mode, in potfile order. Negative values mean use all galaxies for that potfile.",
    )
    parser.add_argument(
        "--scaling-scatter",
        action="store_true",
        help="Add hierarchical intrinsic log-scatter to member-galaxy scaling relations.",
    )
    parser.add_argument(
        "--scaling-scatter-fields",
        default="sigma,core,cut",
        help="Comma-separated scaling-relation fields that get intrinsic scatter when --scaling-scatter is set: sigma, core, cut.",
    )
    parser.add_argument(
        "--scaling-scatter-max",
        type=float,
        default=0.5,
        help="Upper bound for each intrinsic log-scatter hyperparameter.",
    )
    parser.add_argument(
        "--profile-variant",
        choices=(PROFILE_VARIANT_ORIGINAL, PROFILE_VARIANT_COMPACT),
        default=PROFILE_VARIANT_ORIGINAL,
        help="Lens profile used for all dPIE / pseudo-Jaffe components; non-dPIE components keep their native profile types.",
    )
    parser.add_argument(
        "--compact-skip-factor",
        type=float,
        default=1.0,
        help="Effective support multiplier for compact JAX profiles; components farther than compact_skip_factor * Rs from an image can be skipped in the JAX inference path.",
    )
    parser.add_argument(
        "--refresh-every",
        type=int,
        default=DEFAULT_REFRESH_EVERY,
        help="Refresh cadence hint for surrogate mode; currently applied between major phases.",
    )
    parser.add_argument(
        "--refresh-param-drift-frac",
        type=float,
        default=DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        help="Fraction of the prior width used for surrogate finite-difference steps and drift thresholds.",
    )
    parser.add_argument(
        "--fit-method",
        choices=("svi", "svi+nuts"),
        default="svi+nuts",
        help="Use SVI only, or use SVI to initialize optional NUTS sampling.",
    )
    parser.add_argument(
        "--likelihood-mode",
        choices=("source", "hybrid", "image"),
        default="source",
        help="Source-plane sampling is always used; this controls final exact validation breadth.",
    )
    parser.add_argument(
        "--validation-approx",
        choices=("exact", "adaptive"),
        default="adaptive",
        help="Use exact lens-equation validation for all selected families or only for degraded families.",
    )
    parser.add_argument(
        "--z-bin-tol",
        type=float,
        default=DEFAULT_Z_BIN_TOL,
        help="Tolerance for grouping close family redshifts into one effective source plane.",
    )
    parser.add_argument(
        "--validate-top-k-families",
        type=int,
        default=0,
        help="Number of informative families to validate exactly in source/hybrid mode. Omit or set to 0 to skip exact validation there; image mode still validates all families.",
    )
    parser.add_argument(
        "--match-tolerance-arcsec",
        type=float,
        default=DEFAULT_MATCH_TOLERANCE,
        help="Maximum assignment residual for exact image matching.",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--thin", type=int, default=1)
    parser.set_defaults(fit_mode="sequential", sampler="numpyro_nuts", stage1_run_dir=None)
    parser.add_argument("--max-tree-depth", type=int, default=DEFAULT_MAX_TREE_DEPTH)
    parser.add_argument("--target-accept", type=float, default=DEFAULT_TARGET_ACCEPT)
    parser.set_defaults(nuts_init_strategy="svi")
    parser.add_argument(
        "--nuts-init-boundary-frac",
        type=float,
        default=DEFAULT_NUTS_INIT_BOUNDARY_FRAC,
        help="Clip seed proposals away from uniform-prior support edges by this fraction of the support width.",
    )
    parser.add_argument(
        "--nuts-init-jitter-frac",
        type=float,
        default=DEFAULT_NUTS_INIT_JITTER_FRAC,
        help="Constrained-space jitter scale for uniform-prior NUTS seeds, as a fraction of prior width.",
    )
    parser.add_argument(
        "--svi-steps",
        type=int,
        default=DEFAULT_SVI_STEPS,
        help="Number of SVI steps used to initialize NUTS chains.",
    )
    parser.add_argument(
        "--svi-learning-rate",
        type=float,
        default=DEFAULT_SVI_LEARNING_RATE,
        help="Learning rate for the SVI initializer.",
    )
    parser.add_argument(
        "--caustic-num-pix",
        type=int,
        default=250,
        help="Grid resolution for caustic overlay plot.",
    )
    parser.add_argument(
        "--plot-caustics",
        action="store_true",
        help="Generate the expensive caustic overlay plot for the joint stage too.",
    )
    return parser.parse_args()


def _safe_signed_min_abs(values: jnp.ndarray, min_abs: float) -> jnp.ndarray:
    min_abs_value = jnp.asarray(float(min_abs), dtype=jnp.float64)
    signs = jnp.where(values < 0.0, -1.0, 1.0)
    return jnp.where(jnp.abs(values) < min_abs_value, signs * min_abs_value, values)


def _clip_value_to_safe_bounds(value: float, spec: ParameterSpec, boundary_frac: float = 0.0) -> float:
    lower = float(spec.lower)
    upper = float(spec.upper)
    clipped = float(np.clip(float(value), lower, upper))
    if spec.prior_kind == "uniform":
        width = float(upper - lower)
        margin = min(max(float(boundary_frac) * width, 0.0), 0.5 * width - 1.0e-12) if width > 0.0 else 0.0
        if margin > 0.0:
            clipped = float(np.clip(clipped, lower + margin, upper - margin))
    if spec.field in {"vdslope", "slope"} and lower < SAFE_SCALING_EXPONENT_ABS_MIN and upper > -SAFE_SCALING_EXPONENT_ABS_MIN:
        if abs(clipped) < SAFE_SCALING_EXPONENT_ABS_MIN:
            pos_candidate = max(SAFE_SCALING_EXPONENT_ABS_MIN, lower)
            neg_candidate = min(-SAFE_SCALING_EXPONENT_ABS_MIN, upper)
            if clipped >= 0.0 and pos_candidate <= upper:
                clipped = pos_candidate
            elif clipped < 0.0 and neg_candidate >= lower:
                clipped = neg_candidate
            elif pos_candidate <= upper:
                clipped = pos_candidate
            elif neg_candidate >= lower:
                clipped = neg_candidate
    return float(np.clip(clipped, lower, upper))


def _clip_theta_to_support(
    theta: np.ndarray,
    parameter_specs: list[ParameterSpec],
    boundary_frac: float = 0.0,
) -> np.ndarray:
    clipped = np.asarray(theta, dtype=float).copy()
    for idx, spec in enumerate(parameter_specs):
        clipped[idx] = _clip_value_to_safe_bounds(clipped[idx], spec, boundary_frac=boundary_frac)
    return clipped


def _default_theta(parameter_specs: list[ParameterSpec]) -> np.ndarray:
    default = np.asarray(
        [
            0.5 * (spec.lower + spec.upper) if spec.prior_kind == "uniform" else float(spec.mean)
            for spec in parameter_specs
        ],
        dtype=float,
    )
    return _clip_theta_to_support(default, parameter_specs, boundary_frac=DEFAULT_NUTS_INIT_BOUNDARY_FRAC)


def _jitter_theta_in_support(
    theta: np.ndarray,
    parameter_specs: list[ParameterSpec],
    jitter_frac: float,
    rng: np.random.Generator,
    boundary_frac: float = DEFAULT_NUTS_INIT_BOUNDARY_FRAC,
) -> np.ndarray:
    theta_array = np.asarray(theta, dtype=float).copy()
    for idx, spec in enumerate(parameter_specs):
        if spec.prior_kind == "normal":
            scale = max(float(spec.std or 0.0) * 0.15, 1.0e-6)
        else:
            scale = max(float(spec.upper - spec.lower) * float(jitter_frac), 1.0e-6)
        theta_array[idx] += float(rng.normal(0.0, scale))
    return _clip_theta_to_support(theta_array, parameter_specs, boundary_frac=boundary_frac)


def _seed_values_to_init_params(
    parameter_specs: list[ParameterSpec],
    chain_seeds: list[ChainSeed],
    model_for_init=None,
) -> dict[str, jnp.ndarray]:
    if not chain_seeds:
        raise ValueError("At least one chain seed is required.")
    if model_for_init is None:
        model_for_init = _sample_site_model(parameter_specs)
    unconstrained_payloads: list[dict[str, np.ndarray]] = []
    for seed in chain_seeds:
        params_dict = {
            spec.sample_name: jnp.asarray(float(seed.values[idx]), dtype=jnp.float64)
            for idx, spec in enumerate(parameter_specs)
        }
        unconstrained = unconstrain_fn(
            model_for_init,
            model_args=(),
            model_kwargs={},
            params=params_dict,
        )
        unconstrained_payloads.append(
            {
                spec.sample_name: np.asarray(unconstrained[spec.sample_name], dtype=float)
                for spec in parameter_specs
            }
        )
    payload: dict[str, jnp.ndarray] = {}
    for spec in parameter_specs:
        payload[spec.sample_name] = jnp.asarray(
            np.stack([item[spec.sample_name] for item in unconstrained_payloads], axis=0),
            dtype=jnp.float64,
        )
    return payload


def _sample_site_model(parameter_specs: list[ParameterSpec]):
    def model():
        sampled: dict[str, Any] = {}
        spec_by_sample = {spec.sample_name: spec for spec in parameter_specs}
        for spec in parameter_specs:
            if spec.prior_kind == "hierarchical_normal":
                if not spec.parent_sample_name:
                    raise ValueError(f"Hierarchical parameter {spec.name} is missing parent_sample_name.")
                parent_value = _latent_to_physical_jax(sampled[spec.parent_sample_name], spec_by_sample[spec.parent_sample_name])
                sampled[spec.sample_name] = numpyro.sample(
                    spec.sample_name,
                    dist.Normal(float(spec.mean or 0.0), jnp.asarray(parent_value, dtype=jnp.float64)),
                )
            else:
                sampled[spec.sample_name] = numpyro.sample(spec.sample_name, _distribution_for_spec(spec))

    return model


def _values_dict_to_theta(
    parameter_specs: list[ParameterSpec],
    values: dict[str, Any],
) -> np.ndarray:
    return np.asarray([float(values[spec.sample_name]) for spec in parameter_specs], dtype=float)


def _svi_initial_value_dict(
    parameter_specs: list[ParameterSpec],
    init_values: dict[str, float] | None,
) -> dict[str, jnp.ndarray] | None:
    if not init_values:
        return None
    payload: dict[str, jnp.ndarray] = {}
    for spec in parameter_specs:
        if spec.sample_name not in init_values:
            continue
        payload[spec.sample_name] = jnp.asarray(float(init_values[spec.sample_name]), dtype=jnp.float64)
    return payload or None


def _make_auto_normal_guide(
    sample_model,
    parameter_specs: list[ParameterSpec],
    init_values: dict[str, float] | None = None,
):
    init_payload = _svi_initial_value_dict(parameter_specs, init_values)
    if init_payload is None:
        return AutoNormal(sample_model)
    return AutoNormal(sample_model, init_loc_fn=init_to_value(values=init_payload))


def _small_svi_nuts_perturbation(
    theta: np.ndarray,
    parameter_specs: list[ParameterSpec],
    rng: np.random.Generator,
    boundary_frac: float = DEFAULT_NUTS_INIT_BOUNDARY_FRAC,
) -> np.ndarray:
    perturbed = np.asarray(theta, dtype=float).copy()
    for idx, spec in enumerate(parameter_specs):
        if spec.prior_kind == "normal":
            scale = max(0.10 * float(spec.std or 0.0), 1.0e-6)
        else:
            scale = max(0.01 * float(spec.upper - spec.lower), 1.0e-6)
        perturbed[idx] += float(rng.normal(0.0, scale))
    return _clip_theta_to_support(perturbed, parameter_specs, boundary_frac=boundary_frac)


def _run_svi_initializer(
    args: argparse.Namespace,
    sample_model,
    parameter_specs: list[ParameterSpec],
    rng_key: jax.Array,
    init_values: dict[str, float] | None = None,
) -> tuple[list[ChainSeed], dict[str, Any], np.ndarray]:
    guide = _make_auto_normal_guide(sample_model, parameter_specs, init_values)
    svi = SVI(sample_model, guide, numpyro_optim.Adam(float(args.svi_learning_rate)), Trace_ELBO())
    svi_result = svi.run(
        rng_key,
        int(args.svi_steps),
        progress_bar=False,
    )
    params = svi.get_params(svi_result.state)
    init_values = guide.median(params)
    center_theta = _clip_theta_to_support(
        _values_dict_to_theta(parameter_specs, init_values),
        parameter_specs,
        boundary_frac=DEFAULT_NUTS_INIT_BOUNDARY_FRAC,
    )
    if not np.all(np.isfinite(center_theta)):
        raise ValueError("SVI initializer produced non-finite guide median values.")
    rng = np.random.default_rng(None if args.seed is None else int(args.seed) + 303)
    chain_seeds: list[ChainSeed] = []
    chain_start_labels: list[str] = []
    for chain_index in range(int(args.chains)):
        seed_theta = _small_svi_nuts_perturbation(
            center_theta,
            parameter_specs,
            rng,
            boundary_frac=float(getattr(args, "nuts_init_boundary_frac", DEFAULT_NUTS_INIT_BOUNDARY_FRAC)),
        )
        if not np.all(np.isfinite(seed_theta)):
            raise ValueError(f"SVI+NUTS initializer produced non-finite values for chain {chain_index + 1}.")
        chain_label = f"svi_nuts_chain_{chain_index + 1}"
        chain_seeds.append(ChainSeed(values=seed_theta, source_label=chain_label))
        chain_start_labels.append(f"{chain_label}:perturbed")
    diagnostics = {
        "svi_steps": int(args.svi_steps),
        "svi_learning_rate": float(args.svi_learning_rate),
        "svi_final_elbo_loss": float(np.asarray(svi_result.losses[-1], dtype=float)) if len(svi_result.losses) else float("nan"),
        "svi_chain_seed_labels": [seed.source_label for seed in chain_seeds],
        "svi_chain_start_labels": chain_start_labels,
    }
    return chain_seeds, diagnostics, center_theta


def _posterior_logprob_matrix(
    parameter_specs: list[ParameterSpec],
    evaluator: ClusterJAXEvaluator,
    samples: np.ndarray,
) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] == 0:
        return np.empty((0,), dtype=float)

    def logprob_one(theta: jnp.ndarray) -> jnp.ndarray:
        total = evaluator._source_loglike_fn(theta)
        return total + _prior_log_prob(parameter_specs, theta)

    return np.asarray(jax.vmap(logprob_one)(jnp.asarray(sample_array, dtype=jnp.float64)), dtype=float)


def _nuts_initialization_from_svi_center(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
    center_theta: np.ndarray,
    svi_diagnostics: dict[str, Any],
) -> NUTSInitialization:
    rng = np.random.default_rng(None if args.seed is None else int(args.seed) + 303)
    chain_seeds: list[ChainSeed] = []
    chain_start_labels: list[str] = []
    for chain_index in range(int(args.chains)):
        seed_theta = _small_svi_nuts_perturbation(
            center_theta,
            parameter_specs,
            rng,
            boundary_frac=float(getattr(args, "nuts_init_boundary_frac", DEFAULT_NUTS_INIT_BOUNDARY_FRAC)),
        )
        chain_label = f"svi_nuts_chain_{chain_index + 1}"
        chain_seeds.append(ChainSeed(values=np.asarray(seed_theta, dtype=float), source_label=chain_label))
        chain_start_labels.append(f"{chain_label}:perturbed")
    diagnostics = {
        "strategy_requested": "svi",
        "strategy_used": "svi",
        "svi_used": True,
        **svi_diagnostics,
        "svi_chain_seed_labels": [seed.source_label for seed in chain_seeds],
        "svi_chain_start_labels": chain_start_labels,
        "chain_seed_labels": [seed.source_label for seed in chain_seeds],
        "distinct_chain_seeds": int(len({tuple(np.round(seed.values, 8)) for seed in chain_seeds})),
    }
    return NUTSInitialization(
        init_params=_seed_values_to_init_params(parameter_specs, chain_seeds, model_for_init=_sample_site_model(parameter_specs)),
        chain_seeds=chain_seeds,
        diagnostics=diagnostics,
        reference_theta=np.asarray(center_theta, dtype=float),
    )


def _run_svi_fit(
    args: argparse.Namespace,
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    sample_model,
) -> tuple[np.ndarray, PosteriorResults, dict[str, Any]]:
    _log(
        args,
        (
            f"[svi] starting steps={int(args.svi_steps)} "
            f"lr={float(args.svi_learning_rate):.3g} init_values={bool(state.svi_init_values)}"
        ),
    )
    guide = _make_auto_normal_guide(sample_model, state.parameter_specs, state.svi_init_values)
    svi = SVI(sample_model, guide, numpyro_optim.Adam(float(args.svi_learning_rate)), Trace_ELBO())
    svi_start = time.time()
    svi_result = _run_logged_phase(
        args,
        "svi.run",
        lambda: svi.run(
            jax.random.PRNGKey(0 if args.seed is None else int(args.seed) + 202),
            int(args.svi_steps),
            progress_bar=True,
        ),
    )
    svi_elapsed = time.time() - svi_start
    evaluator.timing_totals["svi_runtime"] = evaluator.timing_totals.get("svi_runtime", 0.0) + svi_elapsed
    params = svi.get_params(svi_result.state)
    init_values = guide.median(params)
    center_theta = _clip_theta_to_support(
        _values_dict_to_theta(state.parameter_specs, init_values),
        state.parameter_specs,
        boundary_frac=float(getattr(args, "nuts_init_boundary_frac", DEFAULT_NUTS_INIT_BOUNDARY_FRAC)),
    )
    if not np.all(np.isfinite(center_theta)):
        raise ValueError("SVI produced non-finite guide median values.")
    total_draws = max(1, int(args.samples) * max(1, int(args.chains)))
    guide_samples_dict = _run_logged_phase(
        args,
        "svi.sample_posterior",
        lambda: guide.sample_posterior(
            jax.random.PRNGKey(0 if args.seed is None else int(args.seed) + 909),
            params,
            sample_shape=(total_draws,),
        ),
    )
    guide_samples = np.stack(
        [np.asarray(guide_samples_dict[spec.sample_name], dtype=float) for spec in state.parameter_specs],
        axis=-1,
    )
    guide_samples = _clip_theta_matrix_to_support(np.asarray(guide_samples, dtype=float), state.parameter_specs)
    log_prob = _run_logged_phase(
        args,
        "svi.posterior_logprob",
        lambda: _posterior_logprob_matrix(state.parameter_specs, evaluator, guide_samples),
    )
    diagnostics = {
        "strategy_requested": "svi",
        "strategy_used": "svi",
        "svi_used": True,
        "svi_steps": int(args.svi_steps),
        "svi_learning_rate": float(args.svi_learning_rate),
        "svi_final_elbo_loss": float(np.asarray(svi_result.losses[-1], dtype=float)) if len(svi_result.losses) else float("nan"),
        "svi_runtime_sec": float(svi_elapsed),
        "svi_init_values_used": bool(state.svi_init_values),
        "direct_evaluator_startup": True,
        "requested_chains": 0,
        "retained_finite_chains": 0,
        "dropped_nonfinite_chains": 0,
        "retained_chain_indices": [],
        "dropped_chain_indices": [],
        "chain_seed_labels": [],
        "distinct_chain_seeds": 0,
        "invalid_state_rejection_count": int(getattr(evaluator, "invalid_state_rejection_count", 0)),
        "invalid_state_reason_counts": {
            key: int(value) for key, value in dict(getattr(evaluator, "invalid_state_reason_counts", {})).items()
        },
    }
    posterior = PosteriorResults(
        samples=guide_samples,
        log_prob=log_prob,
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=total_draws,
        num_chains=0,
        init_diagnostics=diagnostics,
        grouped_samples=guide_samples[None, :, :],
        grouped_log_prob=log_prob[None, :],
        sampler="svi",
    )
    _log(
        args,
        f"[svi] complete in {_fmt_seconds(svi_elapsed)} final_elbo={diagnostics['svi_final_elbo_loss']:.4g} guide_draws={total_draws}",
    )
    return np.asarray(center_theta, dtype=float), posterior, diagnostics


def _clip_theta_matrix_to_support(
    theta_matrix: np.ndarray,
    parameter_specs: list[ParameterSpec],
    boundary_frac: float = 0.0,
) -> np.ndarray:
    clipped = np.asarray(theta_matrix, dtype=float).copy()
    for row_index in range(clipped.shape[0]):
        clipped[row_index] = _clip_theta_to_support(clipped[row_index], parameter_specs, boundary_frac=boundary_frac)
    return clipped


def _run_numpyro_nuts_sampler(
    args: argparse.Namespace,
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    sample_model,
    nuts_init: NUTSInitialization,
) -> PosteriorResults:
    chain_method = "parallel" if jax.device_count() >= args.chains else "sequential"
    _log(
        args,
        (
            f"[nuts] preparing sampler chains={args.chains} chain_method={chain_method} "
            f"warmup={args.warmup} samples={args.samples} thin={args.thin} "
            f"target_accept={args.target_accept:.2f} max_tree_depth={args.max_tree_depth} "
            f"init={nuts_init.diagnostics['strategy_used']} distinct_seeds={nuts_init.diagnostics['distinct_chain_seeds']}"
        ),
    )
    nuts = NUTS(
        sample_model,
        target_accept_prob=args.target_accept,
        max_tree_depth=args.max_tree_depth,
    )
    mcmc = MCMC(
        nuts,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method=chain_method,
        progress_bar=True,
        progress_rate=1,
    )
    rng_key = jax.random.PRNGKey(0 if args.seed is None else args.seed)
    _log(args, "[nuts] warmup + sampling started")
    nuts_start = time.time()
    _run_logged_phase(
        args,
        "nuts.run",
        lambda: mcmc.run(
            rng_key,
            extra_fields=("accept_prob", "diverging", "num_steps", "potential_energy"),
            init_params=nuts_init.init_params,
        ),
    )
    nuts_elapsed = time.time() - nuts_start
    evaluator.timing_totals["nuts_runtime"] += nuts_elapsed

    samples_dict = _run_logged_phase(
        args,
        "nuts.get_samples",
        lambda: mcmc.get_samples(group_by_chain=True),
    )
    extra = _run_logged_phase(
        args,
        "nuts.get_extra_fields",
        lambda: mcmc.get_extra_fields(group_by_chain=True),
    )
    samples_dict, extra, chain_quality_diag = _run_logged_phase(
        args,
        "posterior.sanitize_grouped",
        lambda: _sanitize_grouped_posterior(samples_dict, extra, state.parameter_specs),
    )
    nuts_init.diagnostics.update(chain_quality_diag)
    nuts_init.diagnostics["invalid_state_rejection_count"] = int(getattr(evaluator, "invalid_state_rejection_count", 0))
    nuts_init.diagnostics["invalid_state_reason_counts"] = {
        key: int(value) for key, value in dict(getattr(evaluator, "invalid_state_reason_counts", {})).items()
    }
    samples = _run_logged_phase(
        args,
        "posterior.extract_samples",
        lambda: _extract_samples(samples_dict, state.parameter_specs, args.thin),
    )
    grouped_samples = _run_logged_phase(
        args,
        "posterior.extract_grouped_samples",
        lambda: _extract_grouped_samples(samples_dict, state.parameter_specs, args.thin),
    )
    accept_prob = np.asarray(extra["accept_prob"], dtype=float).reshape(-1)[:: max(1, args.thin)]
    diverging = np.asarray(extra["diverging"], dtype=bool).reshape(-1)[:: max(1, args.thin)]
    num_steps = np.asarray(extra["num_steps"], dtype=float).reshape(-1)[:: max(1, args.thin)]
    log_prob = -np.asarray(extra["potential_energy"], dtype=float).reshape(-1)[:: max(1, args.thin)]
    grouped_log_prob = -np.asarray(extra["potential_energy"], dtype=float)[:, :: max(1, args.thin)]
    posterior = PosteriorResults(
        samples=samples,
        log_prob=log_prob,
        accept_prob=accept_prob,
        diverging=diverging,
        num_steps=num_steps,
        warmup_steps=args.warmup,
        sample_steps=args.samples,
        num_chains=int(chain_quality_diag["retained_finite_chains"]),
        init_diagnostics=nuts_init.diagnostics,
        grouped_samples=grouped_samples,
        grouped_log_prob=grouped_log_prob,
        sampler="numpyro_nuts",
    )
    _log(
        args,
        (
            "[nuts] chain quality "
            f"retained_finite_chains={chain_quality_diag['retained_finite_chains']}/"
            f"{chain_quality_diag['requested_chains']} "
            f"dropped_nonfinite_chains={chain_quality_diag['dropped_nonfinite_chains']}"
        ),
    )
    _log(
        args,
        (
            "[nuts] invalid-state guards "
            f"rejections={int(getattr(evaluator, 'invalid_state_rejection_count', 0))} "
            f"reasons={json.dumps({key: int(value) for key, value in dict(getattr(evaluator, 'invalid_state_reason_counts', {})).items() if int(value) > 0}, sort_keys=True)}"
        ),
    )
    _log(
        args,
        (
            f"[nuts] complete in {_fmt_seconds(nuts_elapsed)} "
            f"accept_mean={np.mean(accept_prob):.3f} divergences={int(np.sum(diverging))} "
            f"mean_steps={np.mean(num_steps):.2f} retained_samples={samples.shape[0]}"
        ),
    )
    return posterior


def _fail(message: str) -> None:
    raise SystemExit(message)


def _extract_reference(parsed: dict[str, Any]) -> tuple[int, float, float]:
    runmode = parsed.get("runmode")
    if not isinstance(runmode, dict):
        raise ValueError("Missing runmode block in .par file.")
    reference = runmode.get("reference")
    if not isinstance(reference, list) or len(reference) < 3:
        raise ValueError("Missing runmode.reference in .par file.")
    return int(reference[0]), float(reference[1]), float(reference[2])


def _build_cosmology(parsed: dict[str, Any]):
    cosmo_block = parsed.get("cosmologie")
    if not isinstance(cosmo_block, dict):
        return FlatLambdaCDM(H0=70.0, Om0=0.3)
    h0 = float(cosmo_block.get("H0", 70.0))
    om0 = float(cosmo_block.get("omegaM", 0.3))
    ode0 = float(cosmo_block.get("omegaX", 0.7))
    ok0 = float(cosmo_block.get("omegaK", 0.0))
    w0 = float(cosmo_block.get("wX", -1.0))
    wa = float(cosmo_block.get("wa", 0.0))
    if abs(ok0) < 1e-10:
        if abs(w0 + 1.0) < 1e-10 and abs(wa) < 1e-10:
            return FlatLambdaCDM(H0=h0, Om0=om0)
        return FlatwCDM(H0=h0, Om0=om0, w0=w0)
    if abs(w0 + 1.0) < 1e-10 and abs(wa) < 1e-10:
        return LambdaCDM(H0=h0, Om0=om0, Ode0=ode0)
    return wCDM(H0=h0, Om0=om0, Ode0=ode0, w0=w0)


def _build_cosmology_from_config(cosmo_config: dict[str, Any]):
    if not isinstance(cosmo_config, dict):
        return FlatLambdaCDM(H0=70.0, Om0=0.3)
    class_name = str(cosmo_config.get("class", "FlatLambdaCDM"))
    h0 = float(cosmo_config.get("H0", 70.0))
    om0 = float(cosmo_config.get("Om0", 0.3))
    ode0 = float(cosmo_config.get("Ode0", 1.0 - om0))
    w0 = float(cosmo_config.get("w0", -1.0))
    if class_name == "FlatLambdaCDM":
        return FlatLambdaCDM(H0=h0, Om0=om0)
    if class_name == "FlatwCDM":
        return FlatwCDM(H0=h0, Om0=om0, w0=w0)
    if class_name == "LambdaCDM":
        return LambdaCDM(H0=h0, Om0=om0, Ode0=ode0)
    if class_name == "wCDM":
        return wCDM(H0=h0, Om0=om0, Ode0=ode0, w0=w0)
    return FlatLambdaCDM(H0=h0, Om0=om0)


def _radec_to_offsets_arcsec(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    cos_dec0 = math.cos(math.radians(dec0_deg))
    x_arcsec = (ra0_deg - np.asarray(ra_deg, dtype=float)) * cos_dec0 * 3600.0
    y_arcsec = (np.asarray(dec_deg, dtype=float) - dec0_deg) * 3600.0
    return x_arcsec, y_arcsec


def _kpc_to_arcsec(radius_kpc: float, z_lens: float, cosmo: Any) -> float:
    scale = cosmo.kpc_proper_per_arcmin(z_lens).to(u.kpc / u.arcsec).value
    return float(radius_kpc / scale)


def _bin_redshifts(redshifts: list[float], tolerance: float) -> dict[float, float]:
    sorted_unique = sorted(set(float(z) for z in redshifts))
    groups: list[list[float]] = []
    for value in sorted_unique:
        if not groups or abs(value - groups[-1][-1]) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    mapping: dict[float, float] = {}
    for group in groups:
        representative = float(np.mean(group))
        for value in group:
            mapping[value] = representative
    return mapping


def _build_geometry_cache(
    cosmo: Any,
    z_lens: float,
    family_data: list[FamilyData],
    bin_data: list[BinData],
    *,
    family_redshift_binning_sec: float = 0.0,
) -> GeometryCache:
    geometry_start = time.perf_counter()
    effective_z_source_values = sorted({float(bin_item.effective_z_source) for bin_item in bin_data})
    exact_z_source_values = sorted({float(family.z_source) for family in family_data})
    family_z_source_map = {str(family.family_id): float(family.z_source) for family in family_data}
    family_effective_z_source_map = {
        str(family.family_id): float(family.effective_z_source) for family in family_data
    }
    dpie_sigma0_factor_by_effective_z = {
        float(z_source): _dpie_sigma0_factor(z_lens, float(z_source), cosmo) for z_source in effective_z_source_values
    }
    dpie_sigma0_factor_by_exact_z = {
        float(z_source): _dpie_sigma0_factor(z_lens, float(z_source), cosmo) for z_source in exact_z_source_values
    }
    return GeometryCache(
        effective_z_source_values=[float(value) for value in effective_z_source_values],
        exact_z_source_values=[float(value) for value in exact_z_source_values],
        family_z_source_map=family_z_source_map,
        family_effective_z_source_map=family_effective_z_source_map,
        dpie_sigma0_factor_by_effective_z=dpie_sigma0_factor_by_effective_z,
        dpie_sigma0_factor_by_exact_z=dpie_sigma0_factor_by_exact_z,
        family_redshift_binning_sec=float(family_redshift_binning_sec),
        geometry_cache_build_sec=float(time.perf_counter() - geometry_start),
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_f):
        return None
    return value_f


def _coerce_numeric(value: Any, context: str) -> float:
    direct = _safe_float(value)
    if direct is not None:
        return direct
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if match:
            return float(match.group(0))
    raise ValueError(f"Could not parse numeric value for {context}: {value!r}")


def _sample_name(potential_id: str, field: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", f"{potential_id}_{field}")


def _potfile_fixed_or_nominal(value: Any, context: str) -> float:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty potfile value for {context}.")
        flag = int(value[0])
        if flag == 0 and len(value) >= 2:
            return _coerce_numeric(value[1], context)
        if flag == 1 and len(value) >= 3:
            return 0.5 * (_coerce_numeric(value[1], context) + _coerce_numeric(value[2], context))
        if flag == 3 and len(value) >= 3:
            return _coerce_numeric(value[1], context)
    return _coerce_numeric(value, context)


def _decode_parameter_prior(value: Any, context: str) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    if not value:
        raise ValueError(f"Empty prior specification for {context}.")
    mode = int(value[0])
    if mode == 0:
        return None
    if mode == 1:
        if len(value) < 3:
            raise ValueError(f"Uniform prior for {context} requires at least lower and upper bounds.")
        lower = _coerce_numeric(value[1], f"{context}.lower")
        upper = _coerce_numeric(value[2], f"{context}.upper")
        step = _coerce_numeric(value[3], f"{context}.step") if len(value) >= 4 else 0.1 * (upper - lower)
        return {
            "prior_kind": "uniform",
            "lower": lower,
            "upper": upper,
            "step": step,
            "mean": None,
            "std": None,
        }
    if mode == 3:
        if len(value) < 3:
            raise ValueError(f"Normal prior for {context} requires mean and std.")
        mean = _coerce_numeric(value[1], f"{context}.mean")
        std = _coerce_numeric(value[2], f"{context}.std")
        if std <= 0.0:
            raise ValueError(f"Normal prior std must be positive for {context}.")
        return {
            "prior_kind": "normal",
            "lower": float("-inf"),
            "upper": float("inf"),
            "step": std,
            "mean": mean,
            "std": std,
        }
    raise ValueError(f"Unsupported prior mode {mode} for {context}; expected 0, 1, or 3.")


def _scale_potfile_prior_to_kpc(value: Any, kpc_per_arcsec: float) -> Any:
    if value is None:
        return None
    scale = float(kpc_per_arcsec)
    if isinstance(value, list):
        if not value:
            return []
        return [value[0], *[float(item) * scale for item in value[1:]]]
    return float(value) * scale


def _potfile_core_radius_kpc(potfile: dict[str, Any], kpc_per_arcsec: float) -> float:
    if potfile.get("corekpc") is not None:
        return float(potfile["corekpc"])
    if potfile.get("core_arcsec") is not None:
        return float(potfile["core_arcsec"]) * float(kpc_per_arcsec)
    raise ValueError(f"Potfile {potfile.get('id', 'potfile')} is missing core/corekpc.")


def _potfile_cut_radius_kpc(potfile: dict[str, Any], kpc_per_arcsec: float) -> float:
    if potfile.get("cutkpc_nominal") is not None:
        return float(potfile["cutkpc_nominal"])
    if potfile.get("cut_arcsec_nominal") is not None:
        return float(potfile["cut_arcsec_nominal"]) * float(kpc_per_arcsec)
    raise ValueError(f"Potfile {potfile.get('id', 'potfile')} is missing cut/cutkpc.")


def _normalize_component_field_name(field_name: str) -> str:
    if field_name == "ellipticity":
        return "ellipticite"
    return field_name


def _build_parameter_specs(
    potentials_with_priors: list[dict[str, Any]],
    profile_variant: str = PROFILE_VARIANT_ORIGINAL,
) -> tuple[list[ParameterSpec], list[list[tuple[str, int]]], list[str]]:
    specs: list[ParameterSpec] = []
    component_param_assignments: list[list[tuple[str, int]]] = []
    lens_model_list: list[str] = []
    for potential in potentials_with_priors:
        potential_id = str(potential.get("id"))
        profile_type = int(potential.get("profil"))
        if profile_type not in SUPPORTED_PROFILES:
            raise ValueError(f"Unsupported profile {profile_type} for potential {potential_id}.")
        if profile_type == DP_IE_PROFILE:
            lens_model_list.append(
                COMPACT_DPIE_PROFILE_NAME
                if profile_variant == PROFILE_VARIANT_COMPACT
                else ORIGINAL_DPIE_PROFILE_NAME
            )
        else:
            lens_model_list.append("SHEAR")
        assignments: list[tuple[str, int]] = []
        priors = potential.get("priors", {}) or {}
        for field_name, prior in priors.items():
            normalized_field_name = _normalize_component_field_name(str(field_name))
            decoded_prior = _decode_parameter_prior(prior, f"{potential_id}.{normalized_field_name}")
            if decoded_prior is None:
                continue
            index = len(specs)
            specs.append(
                ParameterSpec(
                    name=f"{potential_id}.{normalized_field_name}",
                    sample_name=_sample_name(potential_id, normalized_field_name),
                    potential_id=potential_id,
                    profile_type=profile_type,
                    field=normalized_field_name,
                    prior_kind=str(decoded_prior["prior_kind"]),
                    lower=float(decoded_prior["lower"]),
                    upper=float(decoded_prior["upper"]),
                    step=float(decoded_prior["step"]),
                    mean=None if decoded_prior["mean"] is None else float(decoded_prior["mean"]),
                    std=None if decoded_prior["std"] is None else float(decoded_prior["std"]),
                )
            )
            assignments.append((normalized_field_name, index))
        component_param_assignments.append(assignments)
    return specs, component_param_assignments, lens_model_list


def _build_scaling_parameter_specs(
    potfiles: list[dict[str, Any]],
    profile_variant: str = PROFILE_VARIANT_ORIGINAL,
    start_index: int = 0,
    kpc_per_arcsec: float = 1.0,
) -> tuple[list[ParameterSpec], list[dict[str, int]], list[str]]:
    specs: list[ParameterSpec] = []
    param_index_by_potfile: list[dict[str, int]] = []
    lens_model_list: list[str] = []
    lens_profile_name = (
        COMPACT_DPIE_PROFILE_NAME
        if profile_variant == PROFILE_VARIANT_COMPACT
        else ORIGINAL_DPIE_PROFILE_NAME
    )
    for potfile in potfiles:
        potfile_id = str(potfile["id"])
        potfile_type = int(potfile["type"])
        if potfile_type != DP_IE_PROFILE:
            raise ValueError(f"Unsupported potfile type {potfile_type} for {potfile_id}.")
        for _ in range(len(potfile["catalog_df"])):
            lens_model_list.append(lens_profile_name)
        field_index: dict[str, int] = {}
        core_radius_kpc = _potfile_core_radius_kpc(potfile, kpc_per_arcsec)
        for field_name in ("sigma", "cutkpc", "corekpc", "vdslope", "slope"):
            raw_value = potfile.get(field_name)
            if field_name == "cutkpc" and raw_value is None:
                raw_value = _scale_potfile_prior_to_kpc(potfile.get("cut"), kpc_per_arcsec)
            elif field_name == "corekpc" and raw_value is None:
                raw_value = _scale_potfile_prior_to_kpc(potfile.get("core"), kpc_per_arcsec)
            decoded_prior = _decode_parameter_prior(raw_value, f"{potfile_id}.{field_name}")
            if decoded_prior is None:
                continue
            prior_kind = str(decoded_prior["prior_kind"])
            lower = float(decoded_prior["lower"])
            upper = float(decoded_prior["upper"])
            mean = None if decoded_prior["mean"] is None else float(decoded_prior["mean"])
            std = None if decoded_prior["std"] is None else float(decoded_prior["std"])
            transform_kind = "identity"
            transform_offset = 0.0
            physical_lower = lower
            physical_upper = upper
            physical_mean = mean
            physical_std = std

            if field_name == "sigma":
                transform_kind = "log_positive"
                if prior_kind == "normal":
                    if mean is None or std is None:
                        raise ValueError(f"Scaling prior for {potfile_id}.{field_name} requires mean/std.")
                    mean, std = _positive_lognormal_parameters(mean, std)
                    lower = float("-inf")
                    upper = float("inf")
                else:
                    lower = np.log(max(lower, 1.0e-12))
                    upper = np.log(max(upper, 1.0e-12))
            elif field_name == "cutkpc":
                transform_kind = "log_offset_positive"
                transform_offset = core_radius_kpc
                gap_lower = max(lower - transform_offset, 1.0e-9)
                gap_upper = max(upper - transform_offset, gap_lower + 1.0e-9)
                if prior_kind == "normal":
                    if mean is None or std is None:
                        raise ValueError(f"Scaling prior for {potfile_id}.{field_name} requires mean/std.")
                    gap_mean = max(mean - transform_offset, 1.0e-9)
                    mean, std = _positive_lognormal_parameters(gap_mean, std)
                    lower = float("-inf")
                    upper = float("inf")
                else:
                    lower = float(np.log(gap_lower))
                    upper = float(np.log(gap_upper))
            index = start_index + len(specs)
            specs.append(
                ParameterSpec(
                    name=f"{potfile_id}.{field_name}",
                    sample_name=_sample_name(potfile_id, field_name),
                    potential_id=potfile_id,
                    profile_type=potfile_type,
                    field=field_name,
                    prior_kind=prior_kind,
                    lower=float(lower),
                    upper=float(upper),
                    step=float(decoded_prior["step"]),
                    mean=mean,
                    std=std,
                    component_family="scaling",
                    transform_kind=transform_kind,
                    physical_lower=physical_lower,
                    physical_upper=physical_upper,
                    physical_mean=physical_mean,
                    physical_std=physical_std,
                    transform_offset=transform_offset,
                )
            )
            field_index[field_name] = index
        param_index_by_potfile.append(field_index)
    return specs, param_index_by_potfile, lens_model_list


def _parse_scaling_scatter_fields(raw_fields: str) -> set[str]:
    aliases = {"v_disp": "sigma", "sigma_ref": "sigma", "corekpc": "core", "cutkpc": "cut"}
    fields = {
        aliases.get(item.strip().lower(), item.strip().lower())
        for item in str(raw_fields or "").split(",")
        if item.strip()
    }
    invalid = fields - {"sigma", "core", "cut"}
    if invalid:
        raise ValueError(f"Unsupported scaling scatter fields: {sorted(invalid)}")
    return fields


def _build_scaling_scatter_parameter_specs(
    potfiles: list[dict[str, Any]],
    fields: set[str],
    *,
    start_index: int,
    scatter_max: float,
) -> tuple[list[ParameterSpec], list[dict[str, int]]]:
    specs: list[ParameterSpec] = []
    scatter_indices_by_potfile: list[dict[str, int]] = []
    if not fields:
        return specs, [{} for _ in potfiles]
    upper_physical = max(float(scatter_max), 1.0e-6)
    lower_physical = 1.0e-4
    for potfile in potfiles:
        potfile_id = str(potfile["id"])
        field_index: dict[str, int] = {}
        for field_name in ("sigma", "core", "cut"):
            if field_name not in fields:
                continue
            index = start_index + len(specs)
            sample_name = _sample_name(potfile_id, f"{field_name}_log_scatter")
            specs.append(
                ParameterSpec(
                    name=f"{potfile_id}.{field_name}_log_scatter",
                    sample_name=sample_name,
                    potential_id=potfile_id,
                    profile_type=int(potfile["type"]),
                    field=f"{field_name}_log_scatter",
                    prior_kind="uniform",
                    lower=float(np.log(lower_physical)),
                    upper=float(np.log(upper_physical)),
                    step=0.1,
                    component_family="scaling_scatter",
                    transform_kind="log_positive",
                    physical_lower=lower_physical,
                    physical_upper=upper_physical,
                )
            )
            field_index[field_name] = index
        scatter_indices_by_potfile.append(field_index)
    return specs, scatter_indices_by_potfile


def _build_scaling_offset_parameter_specs(
    potfiles: list[dict[str, Any]],
    fields: set[str],
    scatter_indices_by_potfile: list[dict[str, int]],
    scatter_specs: list[ParameterSpec],
    *,
    start_index: int,
) -> tuple[list[ParameterSpec], list[list[dict[str, int]]]]:
    specs: list[ParameterSpec] = []
    offset_indices_by_component: list[list[dict[str, int]]] = []
    scatter_spec_by_index = {start_index - len(scatter_specs) + idx: spec for idx, spec in enumerate(scatter_specs)}
    for potfile, scatter_lookup in zip(potfiles, scatter_indices_by_potfile):
        potfile_id = str(potfile["id"])
        catalog_offsets: list[dict[str, int]] = []
        for row in potfile["catalog_df"].itertuples(index=False):
            row_lookup: dict[str, int] = {}
            for field_name in ("sigma", "core", "cut"):
                if field_name not in fields:
                    continue
                parent_index = scatter_lookup.get(field_name, -1)
                if parent_index < 0:
                    continue
                index = start_index + len(specs)
                parent_spec = scatter_spec_by_index[parent_index]
                specs.append(
                    ParameterSpec(
                        name=f"{potfile_id}.{row.id}.{field_name}_log_offset",
                        sample_name=_sample_name(f"{potfile_id}_{row.id}", f"{field_name}_log_offset"),
                        potential_id=potfile_id,
                        profile_type=int(potfile["type"]),
                        field=f"{field_name}_log_offset",
                        prior_kind="hierarchical_normal",
                        lower=float("-inf"),
                        upper=float("inf"),
                        step=0.1,
                        mean=0.0,
                        std=0.1,
                        component_family="scaling_offset",
                        parent_sample_name=parent_spec.sample_name,
                    )
                )
                row_lookup[field_name] = index
            catalog_offsets.append(row_lookup)
        offset_indices_by_component.append(catalog_offsets)
    return specs, offset_indices_by_component


def _build_stage2_large_parameter_specs(
    large_specs: list[ParameterSpec],
    stage1_prior_summary: Stage1PriorSummary,
) -> list[ParameterSpec]:
    stage2_specs: list[ParameterSpec] = []
    for spec in large_specs:
        if spec.sample_name not in stage1_prior_summary.means:
            raise ValueError(f"Missing stage-1 posterior mean for parameter {spec.sample_name}.")
        mean = float(stage1_prior_summary.means[spec.sample_name])
        std = float(stage1_prior_summary.stds.get(spec.sample_name, 0.0))
        if np.isfinite(spec.lower) and np.isfinite(spec.upper):
            width_scale = abs(spec.upper - spec.lower)
        else:
            width_scale = abs(float(spec.std or 0.0))
        floor = max(1.0e-3, 0.05 * max(width_scale, 1.0))
        stage2_specs.append(
            ParameterSpec(
                name=spec.name,
                sample_name=spec.sample_name,
                potential_id=spec.potential_id,
                profile_type=spec.profile_type,
                field=spec.field,
                prior_kind="normal",
                lower=spec.lower,
                upper=spec.upper,
                step=spec.step,
                mean=mean,
                std=max(std, floor),
                component_family=spec.component_family,
                transform_kind=spec.transform_kind,
                physical_lower=spec.physical_lower,
                physical_upper=spec.physical_upper,
                physical_mean=spec.physical_mean,
                physical_std=spec.physical_std,
                transform_offset=spec.transform_offset,
            )
        )
    return stage2_specs


def _build_source_scatter_parameter_spec(start_index: int) -> ParameterSpec:
    del start_index
    return ParameterSpec(
        name="source.sigma_int",
        sample_name="source_sigma_int",
        potential_id="source",
        profile_type=0,
        field="sigma_int",
        prior_kind="uniform",
        lower=float(np.log(DEFAULT_SOURCE_SIGMA_INT_LOWER_ARCSEC)),
        upper=float(np.log(DEFAULT_SOURCE_SIGMA_INT_UPPER_ARCSEC)),
        step=0.05,
        component_family="source_scatter",
        transform_kind="log_positive",
        physical_lower=DEFAULT_SOURCE_SIGMA_INT_LOWER_ARCSEC,
        physical_upper=DEFAULT_SOURCE_SIGMA_INT_UPPER_ARCSEC,
    )


def _field_param_index(assignments: list[tuple[str, int]], field: str) -> int:
    for name, index in assignments:
        if name == field:
            return int(index)
    return -1


def _build_packed_lens_spec(
    base_components: list[dict[str, Any]],
    component_param_assignments: list[list[tuple[str, int]]],
    scaling_component_assignments: list[dict[str, Any]] | None = None,
) -> PackedLensSpec:
    n_components = len(base_components)
    profile_type = np.zeros(n_components, dtype=np.int32)
    component_family = np.zeros(n_components, dtype=np.int32)
    x_center_base = np.zeros(n_components, dtype=float)
    y_center_base = np.zeros(n_components, dtype=float)
    ellipticite_base = np.zeros(n_components, dtype=float)
    angle_pos_base = np.zeros(n_components, dtype=float)
    core_radius_kpc_base = np.zeros(n_components, dtype=float)
    cut_radius_kpc_base = np.zeros(n_components, dtype=float)
    v_disp_base = np.zeros(n_components, dtype=float)
    gamma_base = np.zeros(n_components, dtype=float)
    x_center_param_index = np.full(n_components, -1, dtype=np.int32)
    y_center_param_index = np.full(n_components, -1, dtype=np.int32)
    ellipticite_param_index = np.full(n_components, -1, dtype=np.int32)
    angle_pos_param_index = np.full(n_components, -1, dtype=np.int32)
    core_radius_param_index = np.full(n_components, -1, dtype=np.int32)
    cut_radius_param_index = np.full(n_components, -1, dtype=np.int32)
    v_disp_param_index = np.full(n_components, -1, dtype=np.int32)
    gamma_param_index = np.full(n_components, -1, dtype=np.int32)
    luminosity_ratio = np.ones(n_components, dtype=float)
    sigma_ref_base = np.zeros(n_components, dtype=float)
    cut_ref_base = np.zeros(n_components, dtype=float)
    core_ref_base = np.zeros(n_components, dtype=float)
    vdslope_base = np.zeros(n_components, dtype=float)
    slope_base = np.zeros(n_components, dtype=float)
    sigma_ref_param_index = np.full(n_components, -1, dtype=np.int32)
    cut_ref_param_index = np.full(n_components, -1, dtype=np.int32)
    core_ref_param_index = np.full(n_components, -1, dtype=np.int32)
    vdslope_param_index = np.full(n_components, -1, dtype=np.int32)
    slope_param_index = np.full(n_components, -1, dtype=np.int32)
    sigma_log_scatter_param_index = np.full(n_components, -1, dtype=np.int32)
    core_log_scatter_param_index = np.full(n_components, -1, dtype=np.int32)
    cut_log_scatter_param_index = np.full(n_components, -1, dtype=np.int32)

    for idx, (component, assignments) in enumerate(zip(base_components, component_param_assignments)):
        profile_type[idx] = int(component["profil"])
        component_family[idx] = 0
        x_center_base[idx] = float(component.get("x_centre", 0.0))
        y_center_base[idx] = float(component.get("y_centre", 0.0))
        ellipticite_base[idx] = float(component.get("ellipticite", 0.0))
        angle_pos_base[idx] = float(component.get("angle_pos", 0.0))
        core_radius_kpc_base[idx] = float(component.get("core_radius_kpc", 0.0))
        cut_radius_kpc_base[idx] = float(component.get("cut_radius_kpc", 0.0))
        v_disp_base[idx] = float(component.get("v_disp", 0.0))
        gamma_base[idx] = float(component.get("gamma", 0.0))
        x_center_param_index[idx] = _field_param_index(assignments, "x_centre")
        y_center_param_index[idx] = _field_param_index(assignments, "y_centre")
        ellipticite_param_index[idx] = _field_param_index(assignments, "ellipticite")
        angle_pos_param_index[idx] = _field_param_index(assignments, "angle_pos")
        core_radius_param_index[idx] = _field_param_index(assignments, "core_radius_kpc")
        cut_radius_param_index[idx] = _field_param_index(assignments, "cut_radius_kpc")
        v_disp_param_index[idx] = _field_param_index(assignments, "v_disp")
        gamma_param_index[idx] = _field_param_index(assignments, "gamma")

    scaling_component_assignments = scaling_component_assignments or []
    for item in scaling_component_assignments:
        idx = int(item["component_index"])
        component_family[idx] = 1
        luminosity_ratio[idx] = float(item["luminosity_ratio"])
        sigma_ref_base[idx] = float(item["sigma_ref_base"])
        cut_ref_base[idx] = float(item["cut_ref_base"])
        core_ref_base[idx] = float(item["core_ref_base"])
        vdslope_base[idx] = float(item["vdslope_base"])
        slope_base[idx] = float(item["slope_base"])
        sigma_ref_param_index[idx] = int(item.get("sigma_ref_param_index", -1))
        cut_ref_param_index[idx] = int(item.get("cut_ref_param_index", -1))
        core_ref_param_index[idx] = int(item.get("core_ref_param_index", -1))
        vdslope_param_index[idx] = int(item.get("vdslope_param_index", -1))
        slope_param_index[idx] = int(item.get("slope_param_index", -1))
        sigma_log_scatter_param_index[idx] = int(item.get("sigma_log_scatter_param_index", -1))
        core_log_scatter_param_index[idx] = int(item.get("core_log_scatter_param_index", -1))
        cut_log_scatter_param_index[idx] = int(item.get("cut_log_scatter_param_index", -1))

    return PackedLensSpec(
        profile_type=profile_type,
        component_family=component_family,
        x_center_base=x_center_base,
        y_center_base=y_center_base,
        ellipticite_base=ellipticite_base,
        angle_pos_base=angle_pos_base,
        core_radius_kpc_base=core_radius_kpc_base,
        cut_radius_kpc_base=cut_radius_kpc_base,
        v_disp_base=v_disp_base,
        gamma_base=gamma_base,
        x_center_param_index=x_center_param_index,
        y_center_param_index=y_center_param_index,
        ellipticite_param_index=ellipticite_param_index,
        angle_pos_param_index=angle_pos_param_index,
        core_radius_param_index=core_radius_param_index,
        cut_radius_param_index=cut_radius_param_index,
        v_disp_param_index=v_disp_param_index,
        gamma_param_index=gamma_param_index,
        luminosity_ratio=luminosity_ratio,
        sigma_ref_base=sigma_ref_base,
        cut_ref_base=cut_ref_base,
        core_ref_base=core_ref_base,
        vdslope_base=vdslope_base,
        slope_base=slope_base,
        sigma_ref_param_index=sigma_ref_param_index,
        cut_ref_param_index=cut_ref_param_index,
        core_ref_param_index=core_ref_param_index,
        vdslope_param_index=vdslope_param_index,
        slope_param_index=slope_param_index,
        sigma_log_scatter_param_index=sigma_log_scatter_param_index,
        core_log_scatter_param_index=core_log_scatter_param_index,
        cut_log_scatter_param_index=cut_log_scatter_param_index,
    )


def _serialize_component(potential: dict[str, Any]) -> dict[str, Any]:
    serialized = {key: value for key, value in potential.items() if key != "priors"}
    if "ellipticite" not in serialized and "ellipticity" in serialized:
        serialized["ellipticite"] = serialized.pop("ellipticity")
    serialized["id"] = str(serialized["id"])
    serialized["profil"] = int(serialized["profil"])
    return serialized


def _catalog_shape_to_ellipticity(a_value: float, b_value: float, theta_value: float) -> tuple[float, float]:
    a_axis = max(float(a_value), 1.0e-3)
    b_axis = max(min(float(b_value), a_axis), 1.0e-3)
    q = max(1.0e-3, min(1.0, b_axis / a_axis))
    ellipticite = 1.0 - q
    return ellipticite, float(theta_value)


def _build_scaling_components(
    potfiles: list[dict[str, Any]],
    reference: tuple[int, float, float],
    scaling_param_indices: list[dict[str, int]],
    scaling_offset_indices: list[list[dict[str, int]]] | None,
    start_component_index: int,
    kpc_per_arcsec: float = 1.0,
) -> tuple[list[dict[str, Any]], list[list[tuple[str, int]]], list[dict[str, Any]], list[dict[str, Any]]]:
    _, ra0_deg, dec0_deg = reference
    components: list[dict[str, Any]] = []
    assignments: list[list[tuple[str, int]]] = []
    scaling_component_assignments: list[dict[str, Any]] = []
    scaling_component_records: list[dict[str, Any]] = []
    scaling_offset_indices = scaling_offset_indices or [[] for _ in potfiles]
    for potfile_order, (potfile, param_index_lookup, offset_index_lookup) in enumerate(
        zip(potfiles, scaling_param_indices, scaling_offset_indices)
    ):
        catalog_df = potfile["catalog_df"]
        if catalog_df.empty:
            continue
        core_radius_kpc = _potfile_core_radius_kpc(potfile, kpc_per_arcsec)
        cut_radius_kpc = _potfile_cut_radius_kpc(potfile, kpc_per_arcsec)
        x_offsets, y_offsets = _radec_to_offsets_arcsec(
            catalog_df["ra"].to_numpy(dtype=float),
            catalog_df["dec"].to_numpy(dtype=float),
            ra0_deg,
            dec0_deg,
        )
        magnitudes = catalog_df["catalog_mag"].to_numpy(dtype=float)
        luminosity_ratio = np.power(10.0, -0.4 * (magnitudes - float(potfile["mag0"])))
        for row_index, row in enumerate(catalog_df.itertuples(index=False)):
            ellipticite, angle_pos = _catalog_shape_to_ellipticity(row.catalog_a, row.catalog_b, row.catalog_theta)
            component_index = start_component_index + len(components)
            row_offset_lookup = offset_index_lookup[row_index] if row_index < len(offset_index_lookup) else {}
            components.append(
                {
                    "id": f"{potfile['id']}.{row.id}",
                    "profil": DP_IE_PROFILE,
                    "x_centre": float(x_offsets[row_index]),
                    "y_centre": float(y_offsets[row_index]),
                    "ellipticite": ellipticite,
                    "angle_pos": angle_pos,
                    "core_radius_kpc": core_radius_kpc,
                    "cut_radius_kpc": cut_radius_kpc,
                    "v_disp": float(potfile["sigma_nominal"]),
                    "z_lens": float(potfile["zlens"]),
                }
            )
            assignments.append([])
            scaling_component_assignments.append(
                {
                    "component_index": component_index,
                    "luminosity_ratio": float(luminosity_ratio[row_index]),
                    "sigma_ref_base": float(potfile["sigma_nominal"]),
                    "cut_ref_base": cut_radius_kpc,
                    "core_ref_base": core_radius_kpc,
                    "vdslope_base": float(potfile["vdslope_nominal"]),
                    "slope_base": float(potfile["slope_nominal"]),
                    "sigma_ref_param_index": int(param_index_lookup.get("sigma", -1)),
                    "cut_ref_param_index": int(param_index_lookup.get("cutkpc", -1)),
                    "core_ref_param_index": int(param_index_lookup.get("corekpc", -1)),
                    "vdslope_param_index": int(param_index_lookup.get("vdslope", -1)),
                    "slope_param_index": int(param_index_lookup.get("slope", -1)),
                    "sigma_log_scatter_param_index": int(row_offset_lookup.get("sigma", -1)),
                    "core_log_scatter_param_index": int(row_offset_lookup.get("core", -1)),
                    "cut_log_scatter_param_index": int(row_offset_lookup.get("cut", -1)),
                }
            )
            scaling_component_records.append(
                {
                    "potfile_id": str(potfile["id"]),
                    "potfile_order": int(potfile_order),
                    "component_index": int(component_index),
                    "catalog_id": str(row.id),
                    "catalog_mag": float(row.catalog_mag),
                    "x_centre": float(x_offsets[row_index]),
                    "y_centre": float(y_offsets[row_index]),
                }
            )
    return components, assignments, scaling_component_assignments, scaling_component_records


def _normalize_active_scaling_counts(
    active_scaling_galaxies: Any,
    potfiles: list[dict[str, Any]],
) -> list[int]:
    n_potfiles = len(potfiles)
    if n_potfiles == 0:
        return []
    if active_scaling_galaxies is None:
        return [DEFAULT_ACTIVE_SCALING_GALAXIES for _ in range(n_potfiles)]
    if isinstance(active_scaling_galaxies, (int, np.integer)):
        values = [int(active_scaling_galaxies)]
    else:
        values = [int(value) for value in active_scaling_galaxies]
    if len(values) != n_potfiles:
        potfile_ids = [str(potfile.get("id", f"potfile{idx}")) for idx, potfile in enumerate(potfiles)]
        raise ValueError(
            "--active-scaling-galaxies expects exactly one value per potfile. "
            f"Detected {n_potfiles} potfiles {potfile_ids}, received {len(values)} value(s): {values}."
        )
    resolved: list[int] = []
    for idx, value in enumerate(values):
        if value < 0:
            resolved.append(int(len(potfiles[idx].get("catalog_df", []))))
        else:
            resolved.append(int(value))
    return resolved


def _prepare_family_data(
    images_df: pd.DataFrame,
    sigma_arcsec: float,
    reference: tuple[int, float, float],
    z_bin_tol: float,
) -> tuple[list[FamilyData], float]:
    family_start = time.perf_counter()
    _, ra0_deg, dec0_deg = reference
    if "catalog_z" not in images_df.columns:
        raise ValueError("Image catalog is missing required catalog_z column.")
    image_catalog_sources = sorted(images_df["catalog_source"].dropna().astype(str).unique().tolist())
    source_label = image_catalog_sources[0] if len(image_catalog_sources) == 1 else image_catalog_sources
    image_families = sorted(images_df["family_id"].astype(str).unique().tolist())
    family_redshifts: dict[str, float] = {}
    for family_id in image_families:
        family_df = images_df[images_df["family_id"].astype(str) == family_id].copy()
        z_values = pd.to_numeric(family_df["catalog_z"], errors="coerce")
        if z_values.isna().any():
            bad_labels = family_df.loc[z_values.isna(), "image_label"].astype(str).tolist()
            raise ValueError(
                f"Family {family_id} has missing or invalid catalog_z values for images {bad_labels} "
                f"in image catalog {source_label}."
            )
        unique_z = np.unique(np.round(z_values.to_numpy(dtype=float), 8))
        if len(unique_z) != 1:
            raise ValueError(
                f"Family {family_id} has inconsistent catalog_z values {unique_z.tolist()} "
                f"in image catalog {source_label}."
            )
        z_source = float(unique_z[0])
        if not np.isfinite(z_source) or z_source <= 0:
            raise ValueError(
                f"Family {family_id} has non-positive catalog_z={z_source} in image catalog {source_label}."
            )
        family_redshifts[family_id] = z_source
    z_mapping = _bin_redshifts(list(family_redshifts.values()), z_bin_tol)
    families: list[FamilyData] = []
    for family_id in image_families:
        family_df = images_df[images_df["family_id"].astype(str) == family_id].copy()
        z_source = family_redshifts[family_id]
        x_obs, y_obs = _radec_to_offsets_arcsec(
            family_df["ra"].to_numpy(),
            family_df["dec"].to_numpy(),
            ra0_deg,
            dec0_deg,
        )
        families.append(
            FamilyData(
                family_id=family_id,
                z_source=z_source,
                effective_z_source=float(z_mapping[z_source]),
                sigma_arcsec=float(sigma_arcsec),
                image_labels=family_df["image_label"].astype(str).tolist(),
                x_obs=np.asarray(x_obs, dtype=float),
                y_obs=np.asarray(y_obs, dtype=float),
            )
        )
    return families, float(time.perf_counter() - family_start)


def _build_bin_data(families: list[FamilyData]) -> list[BinData]:
    bins: dict[float, list[FamilyData]] = {}
    for family in families:
        bins.setdefault(family.effective_z_source, []).append(family)
    bin_data: list[BinData] = []
    for effective_z in sorted(bins):
        family_list = bins[effective_z]
        x_obs = np.concatenate([family.x_obs for family in family_list])
        y_obs = np.concatenate([family.y_obs for family in family_list])
        sigma_per_image = np.concatenate(
            [np.full(family.n_images, family.sigma_arcsec, dtype=float) for family in family_list]
        )
        family_index_per_image = np.concatenate(
            [np.full(family.n_images, idx, dtype=int) for idx, family in enumerate(family_list)]
        )
        bin_data.append(
            BinData(
                effective_z_source=float(effective_z),
                family_ids=[family.family_id for family in family_list],
                family_index_per_image=family_index_per_image,
                x_obs=x_obs,
                y_obs=y_obs,
                sigma_per_image=sigma_per_image,
            )
        )
    return bin_data


def _dpie_sigma0_from_vel_disp(vel_disp: float, ra_arcsec: float, rs_arcsec: float, z_lens: float, z_source: float, cosmo: Any) -> float:
    ds = cosmo.angular_diameter_distance(z_source).to(u.m).value
    dds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to(u.m).value
    if ds <= 0 or dds <= 0 or rs_arcsec <= ra_arcsec:
        return float("nan")
    arcsec_rad = np.deg2rad(1.0 / 3600.0)
    c_si = astro_const.c.to_value(u.m / u.s)
    sigma0 = (
        (vel_disp * 1000.0 / c_si) ** 2
        * 2.0
        * np.pi
        * dds
        / ds
        * ((rs_arcsec - ra_arcsec) / (rs_arcsec * ra_arcsec))
        / arcsec_rad
    )
    return float(sigma0)


def _dpie_sigma0_factor(z_lens: float, z_source: float, cosmo: Any) -> float:
    ds = cosmo.angular_diameter_distance(z_source).to(u.m).value
    dds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to(u.m).value
    if ds <= 0 or dds <= 0:
        return float("nan")
    arcsec_rad = np.deg2rad(1.0 / 3600.0)
    c_si = astro_const.c.to_value(u.m / u.s)
    return float((1000.0 / c_si) ** 2 * 2.0 * np.pi * dds / ds / arcsec_rad)


class ClusterJAXEvaluator:
    def __init__(
        self,
        state: BuildState,
        match_tolerance_arcsec: float,
        validate_top_k_families: int,
        sampling_engine: str = "full",
        active_scaling_galaxies: list[int] | int | None = None,
        refresh_every: int = DEFAULT_REFRESH_EVERY,
        refresh_param_drift_frac: float = DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        validation_approx: str = "exact",
    ):
        self.state = state
        self.match_tolerance_arcsec = float(match_tolerance_arcsec)
        self.validate_top_k_families = max(0, int(validate_top_k_families))
        self.sampling_engine = str(sampling_engine)
        self.active_scaling_galaxies_by_potfile = _normalize_active_scaling_counts(active_scaling_galaxies, state.potfiles)
        self.refresh_every = max(1, int(refresh_every))
        self.refresh_param_drift_frac = float(refresh_param_drift_frac)
        self.validation_approx = str(validation_approx)
        geometry_setup_start = time.perf_counter()
        if state.cosmo_config:
            self.cosmo = _build_cosmology_from_config(state.cosmo_config)
        else:  # pragma: no cover - legacy fallback
            self.cosmo = _build_cosmology(state.parsed)
        if getattr(self.state, "geometry_cache", None) is None:
            self.state.geometry_cache = _build_geometry_cache(
                self.cosmo,
                state.z_lens,
                state.family_data,
                state.bin_data,
            )
        geometry_cache = self.state.geometry_cache
        self.validation_cache = {family.family_id: FamilyValidationCache() for family in state.family_data}
        self.eval_wall_times: list[float] = []
        self.timing_totals = {
            "initial_jit_compile": 0.0,
            "geometry_cache_setup": 0.0,
            "packed_parameter_update": 0.0,
            "ray_shooting": 0.0,
            "family_aggregate": 0.0,
            "nuts_runtime": 0.0,
            "exact_model_cache_setup": 0.0,
            "exact_solver": 0.0,
            "validation_runtime": 0.0,
            "validation_conversion": 0.0,
            "plot_runtime": 0.0,
        }
        self.approximate_eval_count = 0
        self.full_refresh_count = 0
        self.validation_fallback_count = 0
        self.invalid_state_rejection_count = 0
        self.invalid_state_reason_counts = {name: 0 for name in INVALID_STATE_REASON_NAMES}
        self.use_bulk_ray_shooting = True
        unique_lens_model_list = list(dict.fromkeys(state.lens_model_list))
        self.bulk_index_list = np.asarray([unique_lens_model_list.index(name) for name in state.lens_model_list], dtype=np.int32)
        compact_skip_factor = float(getattr(self.state, "compact_skip_factor", 1.0))
        self.models_by_effective_z = {
            float(effective_z_source): LensModelBulk(
                unique_lens_model_list=unique_lens_model_list,
                multi_plane=False,
                cosmo=self.cosmo,
                profile_kwargs_list=_jax_profile_kwargs_list(unique_lens_model_list, compact_skip_factor),
            )
            for effective_z_source in geometry_cache.effective_z_source_values
        }
        self.kpc_per_arcsec = self.cosmo.kpc_proper_per_arcmin(state.z_lens).to(u.kpc / u.arcsec).value
        self.dpie_sigma0_factors = {
            float(z_source): float(value)
            for z_source, value in geometry_cache.dpie_sigma0_factor_by_effective_z.items()
        }
        self.exact_dpie_sigma0_factors = {
            float(z_source): float(value)
            for z_source, value in geometry_cache.dpie_sigma0_factor_by_exact_z.items()
        }
        self.exact_models_by_z: dict[float, NPLensModel] = {}
        self.exact_solvers_by_z: dict[float, NPLensEquationSolver] = {}
        self.timing_totals["geometry_cache_setup"] += time.perf_counter() - geometry_setup_start
        self.validation_family_ids = (
            {family.family_id for family in self._select_validation_families(self.validate_top_k_families)}
            if self.validate_top_k_families > 0
            else set()
        )
        component_family = np.asarray(self.state.packed_lens_spec.component_family, dtype=np.int32)
        self.scaling_component_indices = np.where(component_family == 1)[0].astype(np.int32)
        self.large_component_indices = np.where(component_family != 1)[0].astype(np.int32)
        self.scaling_param_indices = np.asarray(
            [idx for idx, spec in enumerate(self.state.parameter_specs) if spec.component_family == "scaling"],
            dtype=np.int32,
        )
        source_scatter_indices = [
            idx for idx, spec in enumerate(self.state.parameter_specs) if spec.component_family == "source_scatter"
        ]
        self.source_sigma_int_param_index = int(source_scatter_indices[0]) if source_scatter_indices else -1
        self.scaling_param_indices_jax = jnp.asarray(self.scaling_param_indices, dtype=jnp.int32)
        transform_kind_array = np.asarray(
            [str(spec.transform_kind) for spec in self.state.parameter_specs],
            dtype=object,
        )
        transform_offset_array = np.asarray(
            [float(spec.transform_offset) for spec in self.state.parameter_specs],
            dtype=float,
        )
        self.transform_kind_log_positive_mask = jnp.asarray(transform_kind_array == "log_positive", dtype=bool)
        self.transform_kind_log_offset_positive_mask = jnp.asarray(transform_kind_array == "log_offset_positive", dtype=bool)
        self.transform_offset_array = jnp.asarray(transform_offset_array, dtype=jnp.float64)
        self.scaling_rank_df = self._build_scaling_rank_diagnostics()
        self.active_scaling_component_indices = np.asarray(
            self.scaling_rank_df.loc[self.scaling_rank_df["selected_active"], "component_index"].to_numpy(dtype=np.int32)
            if not self.scaling_rank_df.empty
            else np.asarray([], dtype=np.int32),
            dtype=np.int32,
        )
        self.inactive_scaling_component_indices = np.asarray(
            sorted(set(self.scaling_component_indices.tolist()) - set(self.active_scaling_component_indices.tolist())),
            dtype=np.int32,
        )
        self.requested_active_scaling_by_potfile = {
            str(potfile.get("id", f"potfile{idx}")): int(self.active_scaling_galaxies_by_potfile[idx])
            for idx, potfile in enumerate(self.state.potfiles)
        }
        self.actual_active_scaling_by_potfile = {
            str(potfile.get("id", f"potfile{idx}")): int(
                len(
                    self.scaling_rank_df[
                        (self.scaling_rank_df["potfile_order"] == idx) & (self.scaling_rank_df["selected_active"])
                    ]
                )
            )
            for idx, potfile in enumerate(self.state.potfiles)
        }
        self.total_scaling_by_potfile = {
            str(potfile.get("id", f"potfile{idx}")): int(
                len(self.scaling_rank_df[self.scaling_rank_df["potfile_order"] == idx])
            )
            for idx, potfile in enumerate(self.state.potfiles)
        }
        self.active_component_indices = np.asarray(
            np.concatenate([self.large_component_indices, self.active_scaling_component_indices]).tolist(),
            dtype=np.int32,
        )
        self.surrogate_enabled = (
            self.sampling_engine == "refreshing_surrogate"
            and self.use_bulk_ray_shooting
            and len(self.scaling_component_indices) > 0
            and len(self.inactive_scaling_component_indices) > 0
            and len(self.scaling_param_indices) > 0
        )
        self.surrogate_reference_params: np.ndarray | None = None
        self.surrogate_reference_scaling_params = np.zeros(len(self.scaling_param_indices), dtype=float)
        self.surrogate_cache_by_z: dict[float, SurrogateBinCache] = {}
        self._source_loglike_fn = jax.jit(self._source_loglike_impl)

    def _physical_parameter_vector(self, params: jnp.ndarray) -> jnp.ndarray:
        return _apply_parameter_transforms_jax(
            params,
            self.transform_kind_log_positive_mask,
            self.transform_kind_log_offset_positive_mask,
            self.transform_offset_array,
        )

    def _source_sigma_int_jax(self, params: jnp.ndarray) -> jnp.ndarray:
        if self.source_sigma_int_param_index < 0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        physical_params = self._physical_parameter_vector(jnp.asarray(params, dtype=jnp.float64))
        return jnp.take(physical_params, jnp.asarray(self.source_sigma_int_param_index, dtype=jnp.int32))

    def _source_sigma_int_numpy(self, params: np.ndarray | jnp.ndarray) -> float:
        if self.source_sigma_int_param_index < 0:
            return 0.0
        params_array = np.asarray(params, dtype=float)
        physical_params = _convert_theta_to_physical(params_array, self.state.parameter_specs)
        return float(physical_params[self.source_sigma_int_param_index])

    def _record_invalid_state_callback(self, reason_flags: Any) -> None:
        flags = np.asarray(reason_flags, dtype=bool).reshape(-1)
        if not flags.any():
            return
        self.invalid_state_rejection_count += 1
        for name, flag in zip(INVALID_STATE_REASON_NAMES, flags.tolist()):
            if flag:
                self.invalid_state_reason_counts[name] = int(self.invalid_state_reason_counts.get(name, 0)) + 1

    def _emit_invalid_state_callback(self, reason_flags: jnp.ndarray) -> jnp.ndarray:
        jax.debug.callback(self._record_invalid_state_callback, reason_flags)
        return jnp.int32(0)

    def _bulk_ray_shooting_kwargs(self, packed_state: dict[str, Any]) -> dict[str, Any]:
        return self._bulk_ray_shooting_kwargs_from_indices(packed_state)

    def _bulk_ray_shooting_kwargs_from_indices(
        self,
        packed_state: dict[str, Any],
        component_indices: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if component_indices is None:
            component_indices = np.arange(len(self.bulk_index_list), dtype=np.int32)
        component_indices_jax = jnp.asarray(component_indices, dtype=jnp.int32)
        all_kwargs = {
            "sigma0": jnp.take(jnp.asarray(packed_state["sigma0"], dtype=jnp.float64), component_indices_jax),
            "Ra": jnp.take(jnp.asarray(packed_state["Ra"], dtype=jnp.float64), component_indices_jax),
            "Rs": jnp.take(jnp.asarray(packed_state["Rs"], dtype=jnp.float64), component_indices_jax),
            "e1": jnp.take(jnp.asarray(packed_state["e1"], dtype=jnp.float64), component_indices_jax),
            "e2": jnp.take(jnp.asarray(packed_state["e2"], dtype=jnp.float64), component_indices_jax),
            "center_x": jnp.take(jnp.asarray(packed_state["center_x"], dtype=jnp.float64), component_indices_jax),
            "center_y": jnp.take(jnp.asarray(packed_state["center_y"], dtype=jnp.float64), component_indices_jax),
            "gamma1": jnp.take(jnp.asarray(packed_state["gamma1"], dtype=jnp.float64), component_indices_jax),
            "gamma2": jnp.take(jnp.asarray(packed_state["gamma2"], dtype=jnp.float64), component_indices_jax),
            "ra_0": jnp.zeros_like(jnp.take(jnp.asarray(packed_state["gamma1"], dtype=jnp.float64), component_indices_jax)),
            "dec_0": jnp.zeros_like(jnp.take(jnp.asarray(packed_state["gamma2"], dtype=jnp.float64), component_indices_jax)),
        }
        return {"all_kwargs": all_kwargs, "index_list": jnp.asarray(self.bulk_index_list[component_indices], dtype=jnp.int32)}

    def _select_validation_families(self, top_k: int) -> list[FamilyData]:
        families = sorted(
            self.state.family_data,
            key=lambda fam: (-fam.n_images, -fam.search_window, abs(fam.x_center) + abs(fam.y_center)),
        )
        return families[: min(top_k, len(families))]

    def _build_scaling_rank_diagnostics(self) -> pd.DataFrame:
        scaling_component_records = getattr(self.state, "scaling_component_records", [])
        if len(self.scaling_component_indices) == 0 or not scaling_component_records:
            return pd.DataFrame(
                columns=[
                    "potfile_id",
                    "potfile_order",
                    "component_index",
                    "catalog_id",
                    "rank",
                    "selected_active",
                    "requested_active_count",
                    "importance",
                    "min_distance_arcsec",
                    "brightness",
                    "catalog_mag",
                    "x_centre",
                    "y_centre",
                ]
            )
        x_all = np.concatenate([family.x_obs for family in self.state.family_data])
        y_all = np.concatenate([family.y_obs for family in self.state.family_data])
        centers_x_all = np.asarray(self.state.packed_lens_spec.x_center_base, dtype=float)
        centers_y_all = np.asarray(self.state.packed_lens_spec.y_center_base, dtype=float)
        rows: list[dict[str, Any]] = []
        for potfile_order, potfile in enumerate(self.state.potfiles):
            potfile_id = str(potfile.get("id", f"potfile{potfile_order}"))
            requested_active_count = int(self.active_scaling_galaxies_by_potfile[potfile_order])
            records = [
                record for record in scaling_component_records if int(record["potfile_order"]) == potfile_order
            ]
            if not records:
                continue
            component_indices = np.asarray([int(record["component_index"]) for record in records], dtype=np.int32)
            centers_x = centers_x_all[component_indices]
            centers_y = centers_y_all[component_indices]
            magnitudes = np.asarray([float(record["catalog_mag"]) for record in records], dtype=float)
            mag0 = float(potfile["mag0"])
            dx = centers_x[:, None] - x_all[None, :]
            dy = centers_y[:, None] - y_all[None, :]
            min_dist = np.min(np.sqrt(dx**2 + dy**2), axis=1)
            brightness = np.power(10.0, -0.4 * (magnitudes - mag0))
            importance = brightness / np.square(min_dist + 0.5)
            order = np.argsort(-importance)
            top_k = min(requested_active_count, len(order))
            active_positions = set(order[:top_k].tolist())
            for rank, record_pos in enumerate(order.tolist(), start=1):
                record = records[record_pos]
                rows.append(
                    {
                        "potfile_id": potfile_id,
                        "potfile_order": potfile_order,
                        "component_index": int(record["component_index"]),
                        "catalog_id": str(record["catalog_id"]),
                        "rank": rank,
                        "selected_active": bool(record_pos in active_positions),
                        "requested_active_count": requested_active_count,
                        "importance": float(importance[record_pos]),
                        "min_distance_arcsec": float(min_dist[record_pos]),
                        "brightness": float(brightness[record_pos]),
                        "catalog_mag": float(record["catalog_mag"]),
                        "x_centre": float(record["x_centre"]),
                        "y_centre": float(record["y_centre"]),
                    }
                )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["potfile_order", "rank"]).reset_index(drop=True)

    def _apply_param_updates(
        self,
        base: jnp.ndarray,
        param_index: np.ndarray,
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        if params.shape[0] == 0:
            return base
        param_index_jax = jnp.asarray(param_index, dtype=jnp.int32)
        safe_index = jnp.where(param_index_jax >= 0, param_index_jax, 0)
        updated = jnp.take(params, safe_index)
        return jnp.where(param_index_jax >= 0, updated, base)

    def _dpie_sigma0_factor_for_z_source(self, z_source: float) -> float:
        factor = self.dpie_sigma0_factors.get(float(z_source))
        if factor is not None:
            return float(factor)
        factor = self.exact_dpie_sigma0_factors.get(float(z_source))
        if factor is not None:
            return float(factor)
        return float(_dpie_sigma0_factor(self.state.z_lens, float(z_source), self.cosmo))

    def _build_packed_lens_state_details(
        self,
        params: jnp.ndarray,
        z_source: float,
    ) -> tuple[dict[str, Any], dict[str, jnp.ndarray]]:
        spec = self.state.packed_lens_spec
        transform_kind_log_positive_mask = getattr(self, "transform_kind_log_positive_mask", None)
        if transform_kind_log_positive_mask is None:
            transform_kind_array = np.asarray(
                [str(spec_item.transform_kind) for spec_item in self.state.parameter_specs],
                dtype=object,
            )
            transform_kind_log_positive_mask = jnp.asarray(transform_kind_array == "log_positive", dtype=bool)
            transform_kind_log_offset_positive_mask = jnp.asarray(
                transform_kind_array == "log_offset_positive",
                dtype=bool,
            )
            transform_offset_array = jnp.asarray(
                [float(spec_item.transform_offset) for spec_item in self.state.parameter_specs],
                dtype=jnp.float64,
            )
        else:
            transform_kind_log_offset_positive_mask = self.transform_kind_log_offset_positive_mask
            transform_offset_array = self.transform_offset_array
        physical_params = _apply_parameter_transforms_jax(
            params,
            transform_kind_log_positive_mask,
            transform_kind_log_offset_positive_mask,
            transform_offset_array,
        )
        x_center = self._apply_param_updates(jnp.asarray(spec.x_center_base, dtype=jnp.float64), spec.x_center_param_index, physical_params)
        y_center = self._apply_param_updates(jnp.asarray(spec.y_center_base, dtype=jnp.float64), spec.y_center_param_index, physical_params)
        ellipticite = self._apply_param_updates(
            jnp.asarray(spec.ellipticite_base, dtype=jnp.float64), spec.ellipticite_param_index, physical_params
        )
        angle_pos = self._apply_param_updates(
            jnp.asarray(spec.angle_pos_base, dtype=jnp.float64), spec.angle_pos_param_index, physical_params
        )
        core_radius_kpc = self._apply_param_updates(
            jnp.asarray(spec.core_radius_kpc_base, dtype=jnp.float64), spec.core_radius_param_index, physical_params
        )
        cut_radius_kpc = self._apply_param_updates(
            jnp.asarray(spec.cut_radius_kpc_base, dtype=jnp.float64), spec.cut_radius_param_index, physical_params
        )
        v_disp = self._apply_param_updates(jnp.asarray(spec.v_disp_base, dtype=jnp.float64), spec.v_disp_param_index, physical_params)
        gamma = self._apply_param_updates(jnp.asarray(spec.gamma_base, dtype=jnp.float64), spec.gamma_param_index, physical_params)

        profile_type = jnp.asarray(spec.profile_type, dtype=jnp.int32)
        component_family = jnp.asarray(spec.component_family, dtype=jnp.int32)
        is_dpie = profile_type == DP_IE_PROFILE
        is_shear = profile_type == SHEAR_PROFILE
        is_scaling = component_family == 1

        luminosity_ratio = jnp.asarray(spec.luminosity_ratio, dtype=jnp.float64)
        sigma_ref = self._apply_param_updates(
            jnp.asarray(spec.sigma_ref_base, dtype=jnp.float64), spec.sigma_ref_param_index, physical_params
        )
        cut_ref = self._apply_param_updates(
            jnp.asarray(spec.cut_ref_base, dtype=jnp.float64), spec.cut_ref_param_index, physical_params
        )
        core_ref = self._apply_param_updates(
            jnp.asarray(spec.core_ref_base, dtype=jnp.float64), spec.core_ref_param_index, physical_params
        )
        vdslope = self._apply_param_updates(
            jnp.asarray(spec.vdslope_base, dtype=jnp.float64), spec.vdslope_param_index, physical_params
        )
        slope = self._apply_param_updates(
            jnp.asarray(spec.slope_base, dtype=jnp.float64), spec.slope_param_index, physical_params
        )
        safe_vdslope = _safe_signed_min_abs(vdslope, SAFE_SCALING_EXPONENT_ABS_MIN)
        safe_slope = _safe_signed_min_abs(slope, SAFE_SCALING_EXPONENT_ABS_MIN)
        scaled_vdisp = sigma_ref * jnp.power(luminosity_ratio, 1.0 / safe_vdslope)
        scaled_core = core_ref * jnp.power(luminosity_ratio, 0.5)
        scaled_cut = cut_ref * jnp.power(luminosity_ratio, 2.0 / safe_slope)
        sigma_log_offset = self._apply_param_updates(
            jnp.zeros_like(scaled_vdisp), spec.sigma_log_scatter_param_index, physical_params
        )
        core_log_offset = self._apply_param_updates(
            jnp.zeros_like(scaled_core), spec.core_log_scatter_param_index, physical_params
        )
        cut_log_offset = self._apply_param_updates(
            jnp.zeros_like(scaled_cut), spec.cut_log_scatter_param_index, physical_params
        )
        scaled_vdisp = scaled_vdisp * jnp.exp(sigma_log_offset)
        scaled_core = scaled_core * jnp.exp(core_log_offset)
        scaled_cut = scaled_cut * jnp.exp(cut_log_offset)
        v_disp = jnp.where(is_scaling, scaled_vdisp, v_disp)
        core_radius_kpc = jnp.where(is_scaling, scaled_core, core_radius_kpc)
        cut_radius_kpc = jnp.where(is_scaling, scaled_cut, cut_radius_kpc)

        q = jnp.maximum(1.0e-3, 1.0 - ellipticite)
        phi = jnp.deg2rad(angle_pos)
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        ra_raw = core_radius_kpc / self.kpc_per_arcsec
        rs_raw = cut_radius_kpc / self.kpc_per_arcsec
        ra = jnp.maximum(ra_raw, SAFE_RADIUS_MARGIN_ARCSEC)
        rs = jnp.maximum(rs_raw, ra + SAFE_RADIUS_MARGIN_ARCSEC)
        factor = self._dpie_sigma0_factor_for_z_source(z_source)
        factor_array = jnp.asarray(factor, dtype=jnp.float64)
        safe_factor = jnp.where(jnp.isfinite(factor_array), factor_array, 0.0)
        sigma0 = (v_disp**2) * safe_factor * ((rs - ra) / (rs * ra))
        gamma1, gamma2 = param_util.shear_polar2cartesian(phi, gamma)

        packed_state = {
            "__packed__": True,
            "profile_type": profile_type,
            "sigma0": jnp.where(is_dpie, sigma0, 0.0),
            "Ra": jnp.where(is_dpie, ra, 0.0),
            "Rs": jnp.where(is_dpie, rs, 0.0),
            "e1": jnp.where(is_dpie, e1, 0.0),
            "e2": jnp.where(is_dpie, e2, 0.0),
            "center_x": x_center,
            "center_y": y_center,
            "gamma1": jnp.where(is_shear, gamma1, 0.0),
            "gamma2": jnp.where(is_shear, gamma2, 0.0),
        }
        details = {
            "is_dpie": is_dpie,
            "is_shear": is_shear,
            "is_scaling": is_scaling,
            "sigma0": sigma0,
            "ra_raw": ra_raw,
            "rs_raw": rs_raw,
            "v_disp": v_disp,
            "vdslope": vdslope,
            "slope": slope,
            "x_center": x_center,
            "y_center": y_center,
            "gamma1": gamma1,
            "gamma2": gamma2,
            "e1": e1,
            "e2": e2,
            "factor_array": factor_array,
        }
        return packed_state, details

    def _packed_lens_validity(
        self,
        details: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        reason_flags = jnp.stack(
            [
                jnp.any(details["is_dpie"] & ~jnp.isfinite(details["sigma0"])),
                jnp.any(details["is_dpie"] & (details["ra_raw"] <= 0.0)),
                jnp.any(details["is_dpie"] & ~jnp.isfinite(details["rs_raw"])),
                jnp.any(details["is_dpie"] & ~jnp.isfinite(details["v_disp"])),
                jnp.any(
                    details["is_scaling"]
                    & (
                        ~jnp.isfinite(details["vdslope"])
                        | ~jnp.isfinite(details["slope"])
                        | (jnp.abs(details["vdslope"]) < SAFE_SCALING_EXPONENT_ABS_MIN)
                        | (jnp.abs(details["slope"]) < SAFE_SCALING_EXPONENT_ABS_MIN)
                    )
                ),
                jnp.any(~jnp.isfinite(details["x_center"]) | ~jnp.isfinite(details["y_center"])),
                jnp.any(details["is_shear"] & (~jnp.isfinite(details["gamma1"]) | ~jnp.isfinite(details["gamma2"]))),
                jnp.any(details["is_dpie"] & (~jnp.isfinite(details["e1"]) | ~jnp.isfinite(details["e2"]))),
                ~jnp.isfinite(details["factor_array"]),
            ],
            axis=0,
        )
        return {"is_valid": ~jnp.any(reason_flags), "reason_flags": reason_flags}

    def _stopped_packed_lens_validity(
        self,
        details: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        return self._packed_lens_validity(
            {key: jax.lax.stop_gradient(value) for key, value in details.items()}
        )

    def _packed_lens_validity_from_params(
        self,
        params: jnp.ndarray,
        z_source: float,
        *,
        stop_gradient: bool,
    ) -> dict[str, jnp.ndarray]:
        validity_params = jax.lax.stop_gradient(params) if stop_gradient else params
        _packed_state, details = self._build_packed_lens_state_details(validity_params, z_source)
        return self._packed_lens_validity(details)

    def _build_packed_lens_state_with_validity(
        self,
        params: jnp.ndarray,
        z_source: float,
    ) -> tuple[dict[str, Any], dict[str, jnp.ndarray]]:
        packed_state = self._build_packed_lens_state(params, z_source)
        return packed_state, self._packed_lens_validity_from_params(params, z_source, stop_gradient=False)

    def _build_packed_lens_state(self, params: jnp.ndarray, z_source: float) -> dict[str, Any]:
        pack_start = time.perf_counter()
        spec = self.state.packed_lens_spec
        transform_kind_log_positive_mask = getattr(self, "transform_kind_log_positive_mask", None)
        if transform_kind_log_positive_mask is None:
            transform_kind_array = np.asarray(
                [str(spec_item.transform_kind) for spec_item in self.state.parameter_specs],
                dtype=object,
            )
            transform_kind_log_positive_mask = jnp.asarray(transform_kind_array == "log_positive", dtype=bool)
            transform_kind_log_offset_positive_mask = jnp.asarray(
                transform_kind_array == "log_offset_positive",
                dtype=bool,
            )
            transform_offset_array = jnp.asarray(
                [float(spec_item.transform_offset) for spec_item in self.state.parameter_specs],
                dtype=jnp.float64,
            )
        else:
            transform_kind_log_offset_positive_mask = self.transform_kind_log_offset_positive_mask
            transform_offset_array = self.transform_offset_array
        physical_params = _apply_parameter_transforms_jax(
            params,
            transform_kind_log_positive_mask,
            transform_kind_log_offset_positive_mask,
            transform_offset_array,
        )
        x_center = self._apply_param_updates(jnp.asarray(spec.x_center_base, dtype=jnp.float64), spec.x_center_param_index, physical_params)
        y_center = self._apply_param_updates(jnp.asarray(spec.y_center_base, dtype=jnp.float64), spec.y_center_param_index, physical_params)
        ellipticite = self._apply_param_updates(
            jnp.asarray(spec.ellipticite_base, dtype=jnp.float64), spec.ellipticite_param_index, physical_params
        )
        angle_pos = self._apply_param_updates(
            jnp.asarray(spec.angle_pos_base, dtype=jnp.float64), spec.angle_pos_param_index, physical_params
        )
        core_radius_kpc = self._apply_param_updates(
            jnp.asarray(spec.core_radius_kpc_base, dtype=jnp.float64), spec.core_radius_param_index, physical_params
        )
        cut_radius_kpc = self._apply_param_updates(
            jnp.asarray(spec.cut_radius_kpc_base, dtype=jnp.float64), spec.cut_radius_param_index, physical_params
        )
        v_disp = self._apply_param_updates(jnp.asarray(spec.v_disp_base, dtype=jnp.float64), spec.v_disp_param_index, physical_params)
        gamma = self._apply_param_updates(jnp.asarray(spec.gamma_base, dtype=jnp.float64), spec.gamma_param_index, physical_params)

        profile_type = jnp.asarray(spec.profile_type, dtype=jnp.int32)
        component_family = jnp.asarray(spec.component_family, dtype=jnp.int32)
        is_dpie = profile_type == DP_IE_PROFILE
        is_shear = profile_type == SHEAR_PROFILE
        is_scaling = component_family == 1

        luminosity_ratio = jnp.asarray(spec.luminosity_ratio, dtype=jnp.float64)
        sigma_ref = self._apply_param_updates(
            jnp.asarray(spec.sigma_ref_base, dtype=jnp.float64), spec.sigma_ref_param_index, physical_params
        )
        cut_ref = self._apply_param_updates(
            jnp.asarray(spec.cut_ref_base, dtype=jnp.float64), spec.cut_ref_param_index, physical_params
        )
        core_ref = self._apply_param_updates(
            jnp.asarray(spec.core_ref_base, dtype=jnp.float64), spec.core_ref_param_index, physical_params
        )
        vdslope = self._apply_param_updates(
            jnp.asarray(spec.vdslope_base, dtype=jnp.float64), spec.vdslope_param_index, physical_params
        )
        slope = self._apply_param_updates(
            jnp.asarray(spec.slope_base, dtype=jnp.float64), spec.slope_param_index, physical_params
        )
        safe_vdslope = _safe_signed_min_abs(vdslope, SAFE_SCALING_EXPONENT_ABS_MIN)
        safe_slope = _safe_signed_min_abs(slope, SAFE_SCALING_EXPONENT_ABS_MIN)
        scaled_vdisp = sigma_ref * jnp.power(luminosity_ratio, 1.0 / safe_vdslope)
        scaled_core = core_ref * jnp.power(luminosity_ratio, 0.5)
        scaled_cut = cut_ref * jnp.power(luminosity_ratio, 2.0 / safe_slope)
        sigma_log_offset = self._apply_param_updates(
            jnp.zeros_like(scaled_vdisp), spec.sigma_log_scatter_param_index, physical_params
        )
        core_log_offset = self._apply_param_updates(
            jnp.zeros_like(scaled_core), spec.core_log_scatter_param_index, physical_params
        )
        cut_log_offset = self._apply_param_updates(
            jnp.zeros_like(scaled_cut), spec.cut_log_scatter_param_index, physical_params
        )
        scaled_vdisp = scaled_vdisp * jnp.exp(sigma_log_offset)
        scaled_core = scaled_core * jnp.exp(core_log_offset)
        scaled_cut = scaled_cut * jnp.exp(cut_log_offset)
        v_disp = jnp.where(is_scaling, scaled_vdisp, v_disp)
        core_radius_kpc = jnp.where(is_scaling, scaled_core, core_radius_kpc)
        cut_radius_kpc = jnp.where(is_scaling, scaled_cut, cut_radius_kpc)

        q = jnp.maximum(1.0e-3, 1.0 - ellipticite)
        phi = jnp.deg2rad(angle_pos)
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        ra_raw = core_radius_kpc / self.kpc_per_arcsec
        rs_raw = cut_radius_kpc / self.kpc_per_arcsec
        ra = jnp.maximum(ra_raw, SAFE_RADIUS_MARGIN_ARCSEC)
        rs = jnp.maximum(rs_raw, ra + SAFE_RADIUS_MARGIN_ARCSEC)
        factor = self._dpie_sigma0_factor_for_z_source(z_source)
        factor_array = jnp.asarray(factor, dtype=jnp.float64)
        safe_factor = jnp.where(jnp.isfinite(factor_array), factor_array, 0.0)
        sigma0 = (v_disp**2) * safe_factor * ((rs - ra) / (rs * ra))
        gamma1, gamma2 = param_util.shear_polar2cartesian(phi, gamma)

        packed_state = {
            "__packed__": True,
            "profile_type": profile_type,
            "sigma0": jnp.where(is_dpie, sigma0, 0.0),
            "Ra": jnp.where(is_dpie, ra, 0.0),
            "Rs": jnp.where(is_dpie, rs, 0.0),
            "e1": jnp.where(is_dpie, e1, 0.0),
            "e2": jnp.where(is_dpie, e2, 0.0),
            "center_x": x_center,
            "center_y": y_center,
            "gamma1": jnp.where(is_shear, gamma1, 0.0),
            "gamma2": jnp.where(is_shear, gamma2, 0.0),
        }
        self.timing_totals["packed_parameter_update"] += time.perf_counter() - pack_start
        return packed_state

    def _maybe_record_invalid_state(self, validity: dict[str, jnp.ndarray]) -> None:
        reason_flags = validity["reason_flags"]
        _ = jax.lax.cond(
            jnp.any(reason_flags),
            self._emit_invalid_state_callback,
            lambda flags: jnp.int32(0),
            reason_flags,
        )

    def _ray_shooting_for_components(
        self,
        z_source: float,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: dict[str, Any],
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        model = self.models_by_effective_z[z_source]
        if self.use_bulk_ray_shooting:
            kwargs = self._bulk_ray_shooting_kwargs_from_indices(packed_state, component_indices)
            return model.ray_shooting(x, y, kwargs)
        if component_indices is not None:
            return model.ray_shooting(x, y, packed_state, k=tuple(int(idx) for idx in component_indices.tolist()))
        return model.ray_shooting(x, y, packed_state)

    def _inactive_fd_step(self, spec: ParameterSpec) -> float:
        if spec.prior_kind == "normal" and spec.std is not None:
            return max(abs(float(spec.std)) * self.refresh_param_drift_frac, 1.0e-4)
        return max(abs(spec.upper - spec.lower) * self.refresh_param_drift_frac, abs(spec.step), 1.0e-4)

    def _replace_scaling_params(
        self,
        reference_params: jnp.ndarray,
        scaling_params: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.scaling_param_indices.size == 0:
            return reference_params
        return reference_params.at[self.scaling_param_indices_jax].set(scaling_params)

    def _inactive_alpha_concat(
        self,
        scaling_params: jnp.ndarray,
        reference_params: jnp.ndarray,
        z_source: float,
        x_obs: jnp.ndarray,
        y_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        params = self._replace_scaling_params(reference_params, scaling_params)
        packed_state = self._build_packed_lens_state(params, z_source)
        validity = self._packed_lens_validity_from_params(params, z_source, stop_gradient=True)
        beta_x, beta_y = jax.lax.cond(
            validity["is_valid"],
            lambda current_state: self._ray_shooting_for_components(
                z_source,
                x_obs,
                y_obs,
                current_state,
                self.inactive_scaling_component_indices,
            ),
            lambda _: (x_obs, y_obs),
            packed_state,
        )
        alpha_x = x_obs - beta_x
        alpha_y = y_obs - beta_y
        return jnp.concatenate([alpha_x, alpha_y])

    def _build_inactive_surrogate_jacobian(
        self,
        reference_params: np.ndarray,
        effective_z_source: float,
        x_obs: jnp.ndarray,
        y_obs: jnp.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        reference = np.asarray(reference_params, dtype=float)
        reference_jax = jnp.asarray(reference, dtype=jnp.float64)
        scaling_reference = reference[self.scaling_param_indices]
        scaling_reference_jax = jnp.asarray(scaling_reference, dtype=jnp.float64)
        inactive_alpha_concat = np.asarray(
            self._inactive_alpha_concat(
                scaling_reference_jax,
                reference_jax,
                effective_z_source,
                x_obs,
                y_obs,
            ),
            dtype=float,
        )
        if not np.isfinite(inactive_alpha_concat).all():
            return None

        n_obs = len(np.asarray(x_obs))
        n_scaling = len(self.scaling_param_indices)
        deriv = np.zeros((2 * n_obs, n_scaling), dtype=float)
        x_obs_np = np.asarray(x_obs, dtype=float)
        y_obs_np = np.asarray(y_obs, dtype=float)

        for local_index, param_index in enumerate(self.scaling_param_indices.tolist()):
            spec = self.state.parameter_specs[int(param_index)]
            nominal_step = float(self._inactive_fd_step(spec))
            theta0 = float(reference[param_index])
            lower = float(spec.lower)
            upper = float(spec.upper)
            minus_room = float(theta0 - lower) if np.isfinite(lower) else float("inf")
            plus_room = float(upper - theta0) if np.isfinite(upper) else float("inf")
            minus_step = nominal_step if not np.isfinite(lower) else min(nominal_step, max(0.0, 0.5 * minus_room))
            plus_step = nominal_step if not np.isfinite(upper) else min(nominal_step, max(0.0, 0.5 * plus_room))
            can_use_minus = minus_step > 1.0e-12
            can_use_plus = plus_step > 1.0e-12

            plus_eval: np.ndarray | None = None
            minus_eval: np.ndarray | None = None

            if can_use_plus:
                plus_theta = reference.copy()
                plus_theta[param_index] = _clip_value_to_safe_bounds(theta0 + plus_step, spec, boundary_frac=0.0)
                plus_validity = self._packed_lens_validity_from_params(
                    jnp.asarray(plus_theta, dtype=jnp.float64),
                    effective_z_source,
                    stop_gradient=False,
                )
                if not bool(np.asarray(plus_validity["is_valid"], dtype=bool)):
                    self._record_invalid_state_callback(np.asarray(plus_validity["reason_flags"], dtype=bool))
                    return None
                plus_eval = np.asarray(
                    self._inactive_alpha_concat(
                        jnp.asarray(plus_theta[self.scaling_param_indices], dtype=jnp.float64),
                        jnp.asarray(plus_theta, dtype=jnp.float64),
                        effective_z_source,
                        x_obs,
                        y_obs,
                    ),
                    dtype=float,
                )
                if not np.isfinite(plus_eval).all():
                    return None

            if can_use_minus:
                minus_theta = reference.copy()
                minus_theta[param_index] = _clip_value_to_safe_bounds(theta0 - minus_step, spec, boundary_frac=0.0)
                minus_validity = self._packed_lens_validity_from_params(
                    jnp.asarray(minus_theta, dtype=jnp.float64),
                    effective_z_source,
                    stop_gradient=False,
                )
                if not bool(np.asarray(minus_validity["is_valid"], dtype=bool)):
                    self._record_invalid_state_callback(np.asarray(minus_validity["reason_flags"], dtype=bool))
                    return None
                minus_eval = np.asarray(
                    self._inactive_alpha_concat(
                        jnp.asarray(minus_theta[self.scaling_param_indices], dtype=jnp.float64),
                        jnp.asarray(minus_theta, dtype=jnp.float64),
                        effective_z_source,
                        x_obs,
                        y_obs,
                    ),
                    dtype=float,
                )
                if not np.isfinite(minus_eval).all():
                    return None

            if plus_eval is not None and minus_eval is not None:
                delta = float(plus_theta[param_index] - minus_theta[param_index])
                if delta <= 0.0:
                    return None
                deriv[:, local_index] = (plus_eval - minus_eval) / delta
            elif plus_eval is not None:
                delta = float(plus_theta[param_index] - theta0)
                if delta <= 0.0:
                    return None
                deriv[:, local_index] = (plus_eval - inactive_alpha_concat) / delta
            elif minus_eval is not None:
                delta = float(theta0 - minus_theta[param_index])
                if delta <= 0.0:
                    return None
                deriv[:, local_index] = (inactive_alpha_concat - minus_eval) / delta
            else:
                return None

        inactive_alpha_x = inactive_alpha_concat[:n_obs]
        inactive_alpha_y = inactive_alpha_concat[n_obs:]
        deriv_x = deriv[:n_obs, :].T
        deriv_y = deriv[n_obs:, :].T

        if not (
            np.isfinite(inactive_alpha_x).all()
            and np.isfinite(inactive_alpha_y).all()
            and np.isfinite(deriv_x).all()
            and np.isfinite(deriv_y).all()
            and np.isfinite(x_obs_np).all()
            and np.isfinite(y_obs_np).all()
        ):
            return None
        return inactive_alpha_x, inactive_alpha_y, deriv_x, deriv_y

    def refresh_surrogate(self, reference_params: np.ndarray, reason: str = "manual") -> None:
        if not self.surrogate_enabled:
            return
        reference = np.asarray(reference_params, dtype=float)
        self.surrogate_reference_params = reference.copy()
        self.surrogate_reference_scaling_params = reference[self.scaling_param_indices].copy()
        self.surrogate_cache_by_z = {}
        reference_jax = jnp.asarray(reference, dtype=jnp.float64)
        for bin_data in self.state.bin_data:
            x_obs = jnp.asarray(bin_data.x_obs, dtype=jnp.float64)
            y_obs = jnp.asarray(bin_data.y_obs, dtype=jnp.float64)
            packed_state = self._build_packed_lens_state(reference_jax, bin_data.effective_z_source)
            validity = self._packed_lens_validity_from_params(reference_jax, bin_data.effective_z_source, stop_gradient=False)
            if not bool(np.asarray(validity["is_valid"], dtype=bool)):
                self._record_invalid_state_callback(np.asarray(validity["reason_flags"], dtype=bool))
                self.surrogate_cache_by_z = {}
                self.surrogate_reference_params = None
                self.surrogate_reference_scaling_params = np.zeros(len(self.scaling_param_indices), dtype=float)
                return
            inactive_surrogate = self._build_inactive_surrogate_jacobian(
                reference,
                bin_data.effective_z_source,
                x_obs,
                y_obs,
            )
            if inactive_surrogate is None:
                self.surrogate_cache_by_z = {}
                self.surrogate_reference_params = None
                self.surrogate_reference_scaling_params = np.zeros(len(self.scaling_param_indices), dtype=float)
                return
            inactive_alpha_x, inactive_alpha_y, deriv_x, deriv_y = inactive_surrogate
            self.surrogate_cache_by_z[bin_data.effective_z_source] = SurrogateBinCache(
                effective_z_source=float(bin_data.effective_z_source),
                inactive_alpha_x=inactive_alpha_x,
                inactive_alpha_y=inactive_alpha_y,
                inactive_alpha_dx_dparams=deriv_x,
                inactive_alpha_dy_dparams=deriv_y,
            )
        self.full_refresh_count += 1
        self._source_loglike_fn = jax.jit(self._source_loglike_impl)

    def _surrogate_needs_refresh(self, params: np.ndarray) -> bool:
        if not self.surrogate_enabled or self.surrogate_reference_params is None:
            return False
        current = np.asarray(params, dtype=float)
        for param_index in self.scaling_param_indices.tolist():
            spec = self.state.parameter_specs[int(param_index)]
            scale = self._inactive_fd_step(spec)
            if abs(current[param_index] - self.surrogate_reference_params[param_index]) > scale:
                return True
        return False

    def _surrogate_beta(
        self,
        params: jnp.ndarray,
        bin_data: BinData,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        packed_state = self._build_packed_lens_state(params, bin_data.effective_z_source)
        validity = self._packed_lens_validity_from_params(params, bin_data.effective_z_source, stop_gradient=True)
        self._maybe_record_invalid_state(validity)
        x_obs = jnp.asarray(bin_data.x_obs, dtype=jnp.float64)
        y_obs = jnp.asarray(bin_data.y_obs, dtype=jnp.float64)
        invalid = ~validity["is_valid"]
        beta_active_x, beta_active_y = jax.lax.cond(
            invalid,
            lambda _: (x_obs, y_obs),
            lambda current_state: self._ray_shooting_for_components(
                bin_data.effective_z_source,
                x_obs,
                y_obs,
                current_state,
                self.active_component_indices,
            ),
            packed_state,
        )
        active_alpha_x = x_obs - beta_active_x
        active_alpha_y = y_obs - beta_active_y
        cache = self.surrogate_cache_by_z[bin_data.effective_z_source]
        delta = jnp.take(params, jnp.asarray(self.scaling_param_indices, dtype=jnp.int32)) - jnp.asarray(
            self.surrogate_reference_scaling_params,
            dtype=jnp.float64,
        )
        inactive_alpha_x = jnp.asarray(cache.inactive_alpha_x, dtype=jnp.float64) + jnp.tensordot(
            delta,
            jnp.asarray(cache.inactive_alpha_dx_dparams, dtype=jnp.float64),
            axes=1,
        )
        inactive_alpha_y = jnp.asarray(cache.inactive_alpha_y, dtype=jnp.float64) + jnp.tensordot(
            delta,
            jnp.asarray(cache.inactive_alpha_dy_dparams, dtype=jnp.float64),
            axes=1,
        )
        beta_x = x_obs - active_alpha_x - inactive_alpha_x
        beta_y = y_obs - active_alpha_y - inactive_alpha_y
        return beta_x, beta_y, invalid

    def _packed_to_kwargs_lens(self, packed_state: dict[str, Any]) -> list[dict[str, float]]:
        convert_start = time.perf_counter()
        kwargs_lens: list[dict[str, float]] = []
        profile_type = np.asarray(packed_state["profile_type"], dtype=int)
        for idx, profile in enumerate(profile_type.tolist()):
            if profile == DP_IE_PROFILE:
                kwargs_lens.append(
                    {
                        "sigma0": float(np.asarray(packed_state["sigma0"][idx])),
                        "Ra": float(np.asarray(packed_state["Ra"][idx])),
                        "Rs": float(np.asarray(packed_state["Rs"][idx])),
                        "e1": float(np.asarray(packed_state["e1"][idx])),
                        "e2": float(np.asarray(packed_state["e2"][idx])),
                        "center_x": float(np.asarray(packed_state["center_x"][idx])),
                        "center_y": float(np.asarray(packed_state["center_y"][idx])),
                    }
                )
            elif profile == SHEAR_PROFILE:
                kwargs_lens.append(
                    {
                        "gamma1": float(np.asarray(packed_state["gamma1"][idx])),
                        "gamma2": float(np.asarray(packed_state["gamma2"][idx])),
                    }
                )
            else:  # pragma: no cover
                raise ValueError(f"Unsupported profile type {profile}.")
        self.timing_totals["validation_conversion"] += time.perf_counter() - convert_start
        return kwargs_lens

    def _source_loglike_impl(self, params: jnp.ndarray) -> jnp.ndarray:
        total_loglike = jnp.array(0.0, dtype=jnp.float64)
        invalid_seen = jnp.array(False)
        source_sigma_int = self._source_sigma_int_jax(params)
        for bin_data in self.state.bin_data:
            if self.surrogate_enabled and self.surrogate_cache_by_z:
                beta_x, beta_y, invalid = self._surrogate_beta(params, bin_data)
            else:
                packed_state = self._build_packed_lens_state(params, bin_data.effective_z_source)
                validity = self._packed_lens_validity_from_params(params, bin_data.effective_z_source, stop_gradient=True)
                self._maybe_record_invalid_state(validity)
                x_obs = jnp.asarray(bin_data.x_obs, dtype=jnp.float64)
                y_obs = jnp.asarray(bin_data.y_obs, dtype=jnp.float64)
                invalid = ~validity["is_valid"]
                beta_x, beta_y = jax.lax.cond(
                    invalid,
                    lambda _: (x_obs, y_obs),
                    lambda current_state: self._ray_shooting_for_components(
                        bin_data.effective_z_source,
                        x_obs,
                        y_obs,
                        current_state,
                    ),
                    packed_state,
                )
            family_idx = jnp.asarray(bin_data.family_index_per_image, dtype=jnp.int32)
            sigma_base = jnp.asarray(bin_data.sigma_per_image, dtype=jnp.float64)
            sigma = jnp.sqrt(jnp.square(sigma_base) + jnp.square(source_sigma_int))
            weights = 1.0 / jnp.square(sigma)
            n_families = len(bin_data.family_ids)
            family_counts = jnp.zeros(n_families, dtype=jnp.int32).at[family_idx].add(1)
            image_has_constraint = family_counts[family_idx] > 1
            sum_w = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(weights)
            sum_bx = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(weights * beta_x)
            sum_by = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(weights * beta_y)
            centroid_x = sum_bx / jnp.maximum(sum_w, 1.0e-18)
            centroid_y = sum_by / jnp.maximum(sum_w, 1.0e-18)
            dx = beta_x - centroid_x[family_idx]
            dy = beta_y - centroid_y[family_idx]
            sigma2 = jnp.square(sigma)
            bin_loglike = -0.5 * jnp.sum(
                jnp.where(
                    image_has_constraint,
                    (dx**2 + dy**2) / sigma2 + 2.0 * jnp.log(2.0 * jnp.pi * sigma2),
                    0.0,
                )
            )
            total_loglike = jnp.where(invalid, total_loglike, total_loglike + bin_loglike)
            invalid_seen = jnp.logical_or(invalid_seen, invalid)
        return jnp.where(
            invalid_seen,
            jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64),
            jnp.nan_to_num(total_loglike, nan=BAD_LOG_LIKE, posinf=BAD_LOG_LIKE, neginf=BAD_LOG_LIKE),
        )

    def source_loglike(self, params: np.ndarray | jnp.ndarray) -> float:
        start = time.perf_counter()
        value = float(self._source_loglike_fn(jnp.asarray(params, dtype=jnp.float64)))
        elapsed = time.perf_counter() - start
        self.eval_wall_times.append(elapsed)
        if self.surrogate_enabled and self.surrogate_cache_by_z:
            self.approximate_eval_count += 1
        self.timing_totals["ray_shooting"] += elapsed
        self.timing_totals["family_aggregate"] += elapsed
        return value

    def _family_source_summary(self, params: np.ndarray | jnp.ndarray) -> dict[str, dict[str, Any]]:
        summaries: dict[str, dict[str, Any]] = {}
        params_jax = jnp.asarray(params, dtype=jnp.float64)
        source_sigma_int = self._source_sigma_int_numpy(params)
        for bin_data in self.state.bin_data:
            if self.surrogate_enabled and self.surrogate_cache_by_z:
                beta_x, beta_y, invalid = self._surrogate_beta(params_jax, bin_data)
                if bool(np.asarray(invalid, dtype=bool)):
                    for family_id in bin_data.family_ids:
                        summaries[str(family_id)] = {
                            "source_x": float("nan"),
                            "source_y": float("nan"),
                            "source_plane_rms": float("nan"),
                            "failed": True,
                        }
                    continue
            else:
                packed_state = self._build_packed_lens_state(params_jax, bin_data.effective_z_source)
                validity = self._packed_lens_validity_from_params(params_jax, bin_data.effective_z_source, stop_gradient=False)
                if not bool(np.asarray(validity["is_valid"], dtype=bool)):
                    self._record_invalid_state_callback(np.asarray(validity["reason_flags"], dtype=bool))
                    for family_id in bin_data.family_ids:
                        summaries[str(family_id)] = {
                            "source_x": float("nan"),
                            "source_y": float("nan"),
                            "source_plane_rms": float("nan"),
                            "failed": True,
                        }
                    continue
                beta_x, beta_y = self._ray_shooting_for_components(
                    bin_data.effective_z_source,
                    jnp.asarray(bin_data.x_obs, dtype=jnp.float64),
                    jnp.asarray(bin_data.y_obs, dtype=jnp.float64),
                    packed_state,
                )
            beta_x = np.asarray(beta_x, dtype=float)
            beta_y = np.asarray(beta_y, dtype=float)
            idx = np.asarray(bin_data.family_index_per_image, dtype=int)
            for family_index, family_id in enumerate(bin_data.family_ids):
                mask = idx == family_index
                family = next(item for item in self.state.family_data if item.family_id == family_id)
                sigma_eff = float(np.sqrt(family.sigma_arcsec**2 + source_sigma_int**2))
                weights = np.full(np.sum(mask), 1.0 / (sigma_eff**2), dtype=float)
                source_x = float(np.average(beta_x[mask], weights=weights))
                source_y = float(np.average(beta_y[mask], weights=weights))
                dx = beta_x[mask] - source_x
                dy = beta_y[mask] - source_y
                residuals = np.sqrt(dx**2 + dy**2)
                summaries[family_id] = {
                    "source_x": source_x,
                    "source_y": source_y,
                    "source_beta_x": beta_x[mask],
                    "source_beta_y": beta_y[mask],
                    "source_sigma_int_arcsec": source_sigma_int,
                    "source_sigma_eff_arcsec": sigma_eff,
                    "source_plane_rms": float(np.sqrt(np.mean(residuals**2))),
                    "residual_max": float(np.max(residuals)),
                    "x_pred": np.full(family.n_images, np.nan),
                    "y_pred": np.full(family.n_images, np.nan),
                    "exact_image_rms": np.nan,
                    "failed": False,
                }
        return summaries

    def _get_exact_model_solver(self, z_source: float) -> tuple[NPLensModel, NPLensEquationSolver]:
        model = self.exact_models_by_z.get(z_source)
        if model is None:
            exact_model_setup_start = time.perf_counter()
            model = NPLensModel(
                lens_model_list=self.state.lens_model_list,
                z_lens=self.state.z_lens,
                z_source=z_source,
                cosmo=self.cosmo,
            )
            self.exact_models_by_z[z_source] = model
            self.exact_solvers_by_z[z_source] = NPLensEquationSolver(model)
            self.timing_totals["exact_model_cache_setup"] += time.perf_counter() - exact_model_setup_start
        return model, self.exact_solvers_by_z[z_source]

    def _match_images(self, x_pred: np.ndarray, y_pred: np.ndarray, family: FamilyData) -> tuple[np.ndarray, np.ndarray] | None:
        if len(x_pred) != family.n_images:
            return None
        pred = np.column_stack([x_pred, y_pred])
        obs = np.column_stack([family.x_obs, family.y_obs])
        cost = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        if np.any(cost[row_ind, col_ind] > self.match_tolerance_arcsec):
            return None
        ordered_x = np.empty(family.n_images, dtype=float)
        ordered_y = np.empty(family.n_images, dtype=float)
        for r, c in zip(row_ind, col_ind):
            ordered_x[c] = x_pred[r]
            ordered_y[c] = y_pred[r]
        return ordered_x, ordered_y

    def _exact_family_prediction(self, params: np.ndarray, family: FamilyData) -> tuple[np.ndarray, np.ndarray, float] | None:
        cache = self.validation_cache[family.family_id]
        model, solver = self._get_exact_model_solver(family.z_source)
        packed_state = self._build_packed_lens_state(jnp.asarray(params, dtype=jnp.float64), family.z_source)
        kwargs_lens = self._packed_to_kwargs_lens(packed_state)
        beta_x, beta_y = model.ray_shooting(
            jnp.asarray(family.x_obs, dtype=jnp.float64),
            jnp.asarray(family.y_obs, dtype=jnp.float64),
            kwargs_lens,
        )
        weights = np.full(family.n_images, 1.0 / (family.sigma_arcsec**2), dtype=float)
        source_x = float(np.average(np.asarray(beta_x, dtype=float), weights=weights))
        source_y = float(np.average(np.asarray(beta_y, dtype=float), weights=weights))
        source_residuals = np.sqrt((np.asarray(beta_x, dtype=float) - source_x) ** 2 + (np.asarray(beta_y, dtype=float) - source_y) ** 2)
        cache.source_plane_rms = float(np.sqrt(np.mean(source_residuals**2)))
        cache.last_source_x = source_x
        cache.last_source_y = source_y
        exact_start = time.perf_counter()
        x_pred, y_pred = solver.image_position_from_source(
            source_x,
            source_y,
            kwargs_lens,
            solver="lenstronomy",
            min_distance=max(0.02, family.sigma_arcsec / 5.0),
            search_window=family.search_window,
            x_center=family.x_center,
            y_center=family.y_center,
            num_iter_max=200,
            precision_limit=1e-8,
        )
        self.timing_totals["exact_solver"] += time.perf_counter() - exact_start
        cache.exact_validation_count += 1
        matched = self._match_images(np.asarray(x_pred), np.asarray(y_pred), family)
        if matched is None:
            if len(x_pred) != family.n_images:
                cache.multiplicity_mismatch_count += 1
            else:
                cache.match_failure_count += 1
            return None
        residuals = np.sqrt((matched[0] - family.x_obs) ** 2 + (matched[1] - family.y_obs) ** 2)
        rms = float(np.sqrt(np.mean(residuals**2)))
        cache.exact_image_rms = rms
        return matched[0], matched[1], rms

    def _should_run_exact_validation(self, family: FamilyData, family_prediction: dict[str, Any]) -> tuple[bool, str]:
        if self.validation_approx == "exact":
            return True, "exact_mode"
        cache = self.validation_cache[family.family_id]
        if cache.exact_image_rms is None or cache.source_plane_rms is None:
            return True, "no_cached_exact"
        current_rms = float(family_prediction.get("source_plane_rms", np.inf))
        if not np.isfinite(current_rms):
            return True, "non_finite_rms"
        if current_rms > max(0.05, DEFAULT_VALIDATION_RMS_FACTOR * cache.source_plane_rms):
            return True, "rms_degraded"
        source_x = float(family_prediction.get("source_x", 0.0))
        source_y = float(family_prediction.get("source_y", 0.0))
        if cache.last_source_x is None or cache.last_source_y is None:
            return True, "missing_cached_source"
        source_shift = math.hypot(source_x - cache.last_source_x, source_y - cache.last_source_y)
        if source_shift > max(0.03, 0.5 * family.sigma_arcsec):
            return True, "source_shift"
        return False, "cache_ok"

    def evaluate(self, params: np.ndarray, likelihood_mode: str) -> EvaluationResult:
        if self.surrogate_enabled and self._surrogate_needs_refresh(np.asarray(params, dtype=float)):
            self.refresh_surrogate(np.asarray(params, dtype=float), reason="validation_drift")
        source_loglike = self.source_loglike(params)
        family_predictions = self._family_source_summary(params)
        should_validate_all = likelihood_mode == "image"
        validate_ids = {family.family_id for family in self.state.family_data} if should_validate_all else self.validation_family_ids
        used_exact = False
        total_loglike = source_loglike
        for family in self.state.family_data:
            if family.family_id not in validate_ids:
                continue
            family_predictions[family.family_id]["approx_image_rms_arcsec"] = family_predictions[family.family_id].get(
                "source_plane_rms"
            )
            should_exact, reason = self._should_run_exact_validation(family, family_predictions[family.family_id])
            family_predictions[family.family_id]["refresh_reason"] = reason
            family_predictions[family.family_id]["used_exact_refresh"] = bool(should_exact)
            if not should_exact:
                continue
            _log(
                None,
                f"[validation:family] family={family.family_id} start reason={reason} z={family.z_source:.4f} n_images={family.n_images}",
            )
            family_start = time.time()
            prediction = self._exact_family_prediction(params, family)
            used_exact = True
            if self.validation_approx == "adaptive":
                self.validation_fallback_count += 1
            if prediction is None:
                family_predictions[family.family_id]["failed"] = True
                total_loglike += -1.0e6
                _log(
                    None,
                    f"[validation:family] family={family.family_id} failed elapsed={_fmt_seconds(time.time() - family_start)}",
                )
            else:
                x_pred, y_pred, exact_rms = prediction
                family_predictions[family.family_id]["x_pred"] = x_pred
                family_predictions[family.family_id]["y_pred"] = y_pred
                family_predictions[family.family_id]["exact_image_rms"] = exact_rms
                _log(
                    None,
                    f"[validation:family] family={family.family_id} end elapsed={_fmt_seconds(time.time() - family_start)} exact_rms={exact_rms:.4f}",
                )
        return EvaluationResult(loglike=float(total_loglike), family_predictions=family_predictions, used_exact_validation=used_exact)


def _save_artifacts(
    artifacts_dir: Path,
    state: BuildState,
    args: argparse.Namespace,
    best_fit: np.ndarray,
    results: PosteriorResults,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _save_plot_bundle_h5(artifacts_dir / "plot_bundle.h5", state, args, best_fit, results)


def _approximate_evaluation(evaluator: ClusterJAXEvaluator, params: np.ndarray) -> EvaluationResult:
    return EvaluationResult(
        loglike=float(evaluator.source_loglike(params)),
        family_predictions=evaluator._family_source_summary(params),
        used_exact_validation=False,
    )


def _to_jsonable(payload: Any) -> Any:
    if isinstance(payload, Path):
        return {"__path__": str(payload)}
    if isinstance(payload, pd.DataFrame):
        return {"__dataframe__": payload.to_dict(orient="list")}
    if isinstance(payload, dict):
        return {str(key): _to_jsonable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_to_jsonable(value) for value in payload]
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


def _from_jsonable(payload: Any) -> Any:
    if isinstance(payload, dict):
        if "__path__" in payload:
            return Path(str(payload["__path__"]))
        if "__dataframe__" in payload:
            return pd.DataFrame(payload["__dataframe__"])
        return {str(key): _from_jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_from_jsonable(value) for value in payload]
    return payload


def _write_h5_json(group: h5py.Group, name: str, payload: Any) -> None:
    group.create_dataset(name, data=json.dumps(_to_jsonable(payload)))


def _read_h5_json(group: h5py.Group, name: str, default: Any = None) -> Any:
    if name not in group:
        return default
    raw = group[name][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return _from_jsonable(json.loads(raw))


def _save_plot_bundle_h5(
    path: Path,
    state: BuildState,
    args: argparse.Namespace,
    best_fit: np.ndarray,
    results: PosteriorResults,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 1

        posterior_group = handle.create_group("posterior")
        posterior_group.create_dataset("samples", data=np.asarray(results.samples, dtype=float))
        posterior_group.create_dataset("log_prob", data=np.asarray(results.log_prob, dtype=float))
        posterior_group.create_dataset("accept_prob", data=np.asarray(results.accept_prob, dtype=float))
        posterior_group.create_dataset("diverging", data=np.asarray(results.diverging, dtype=bool))
        posterior_group.create_dataset("num_steps", data=np.asarray(results.num_steps, dtype=float))
        posterior_group.create_dataset("best_fit", data=np.asarray(best_fit, dtype=float))
        posterior_group.attrs["sampler"] = str(results.sampler)
        if results.grouped_samples is not None:
            posterior_group.create_dataset("grouped_samples", data=np.asarray(results.grouped_samples, dtype=float))
        if results.grouped_log_prob is not None:
            posterior_group.create_dataset("grouped_log_prob", data=np.asarray(results.grouped_log_prob, dtype=float))
        if results.sample_weights is not None:
            posterior_group.create_dataset("sample_weights", data=np.asarray(results.sample_weights, dtype=float))
        if results.temperature_schedule is not None:
            posterior_group.create_dataset("temperature_schedule", data=np.asarray(results.temperature_schedule, dtype=float))
        if results.ess_history is not None:
            posterior_group.create_dataset("ess_history", data=np.asarray(results.ess_history, dtype=float))
        if results.move_acceptance_history is not None:
            posterior_group.create_dataset("move_acceptance_history", data=np.asarray(results.move_acceptance_history, dtype=float))

        _write_h5_json(handle, "cli_args_json", vars(args))
        _write_h5_json(handle, "init_diagnostics_json", results.init_diagnostics or {})

        state_group = handle.create_group("state")
        _write_h5_json(
            state_group,
            "build_state_meta_json",
            {
                "run_name": state.run_name,
                "par_path": state.par_path,
                "cosmo_config": state.cosmo_config,
                "z_lens": state.z_lens,
                "sigma_arcsec": state.sigma_arcsec,
                "reference": list(state.reference),
                "fit_mode": state.fit_mode,
                "profile_variant": state.profile_variant,
                "compact_skip_factor": state.compact_skip_factor,
                "lens_model_list": state.lens_model_list,
                "base_components": state.base_components,
                "potfiles": state.potfiles,
                "scaling_component_records": state.scaling_component_records,
                "geometry_cache": None if state.geometry_cache is None else {
                    "effective_z_source_values": state.geometry_cache.effective_z_source_values,
                    "exact_z_source_values": state.geometry_cache.exact_z_source_values,
                    "family_z_source_map": state.geometry_cache.family_z_source_map,
                    "family_effective_z_source_map": state.geometry_cache.family_effective_z_source_map,
                    "dpie_sigma0_factor_by_effective_z": state.geometry_cache.dpie_sigma0_factor_by_effective_z,
                    "dpie_sigma0_factor_by_exact_z": state.geometry_cache.dpie_sigma0_factor_by_exact_z,
                    "family_redshift_binning_sec": state.geometry_cache.family_redshift_binning_sec,
                    "geometry_cache_build_sec": state.geometry_cache.geometry_cache_build_sec,
                },
                "svi_init_values": state.svi_init_values,
                "parameter_specs": [spec.__dict__ for spec in state.parameter_specs],
                "family_data": [
                    {
                        "family_id": family.family_id,
                        "z_source": family.z_source,
                        "effective_z_source": family.effective_z_source,
                        "sigma_arcsec": family.sigma_arcsec,
                        "image_labels": family.image_labels,
                        "x_obs": family.x_obs,
                        "y_obs": family.y_obs,
                    }
                    for family in state.family_data
                ],
                "bin_data": [
                    {
                        "effective_z_source": bin_item.effective_z_source,
                        "family_ids": bin_item.family_ids,
                        "family_index_per_image": bin_item.family_index_per_image,
                        "x_obs": bin_item.x_obs,
                        "y_obs": bin_item.y_obs,
                        "sigma_per_image": bin_item.sigma_per_image,
                    }
                    for bin_item in state.bin_data
                ],
            },
        )
        packed_group = state_group.create_group("packed_lens_spec")
        for field_name in PackedLensSpec.__dataclass_fields__:
            packed_group.create_dataset(field_name, data=np.asarray(getattr(state.packed_lens_spec, field_name)))


def _rebuild_state_from_h5(path: Path) -> tuple[BuildState, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        meta = _read_h5_json(handle["state"], "build_state_meta_json", default={})
        packed_group = handle["state"]["packed_lens_spec"]
        n_components = len(np.asarray(packed_group["profile_type"]))
        packed_lens_spec = PackedLensSpec(
            **{
                field_name: (
                    np.asarray(packed_group[field_name])
                    if field_name in packed_group
                    else np.full(n_components, -1, dtype=np.int32)
                )
                for field_name in PackedLensSpec.__dataclass_fields__
            }
        )
        state = BuildState(
            run_name=str(meta["run_name"]),
            par_path=str(meta["par_path"]),
            cosmo_config=dict(meta["cosmo_config"]),
            z_lens=float(meta["z_lens"]),
            sigma_arcsec=float(meta["sigma_arcsec"]),
            parsed={},
            parameter_specs=[ParameterSpec(**item) for item in meta.get("parameter_specs", [])],
            base_components=list(meta.get("base_components", [])),
            packed_lens_spec=packed_lens_spec,
            family_data=[
                FamilyData(
                    family_id=str(item["family_id"]),
                    z_source=float(item["z_source"]),
                    effective_z_source=float(item["effective_z_source"]),
                    sigma_arcsec=float(item["sigma_arcsec"]),
                    image_labels=[str(label) for label in item["image_labels"]],
                    x_obs=np.asarray(item["x_obs"], dtype=float),
                    y_obs=np.asarray(item["y_obs"], dtype=float),
                )
                for item in meta.get("family_data", [])
            ],
            bin_data=[
                BinData(
                    effective_z_source=float(item["effective_z_source"]),
                    family_ids=[str(family_id) for family_id in item["family_ids"]],
                    family_index_per_image=np.asarray(item["family_index_per_image"], dtype=int),
                    x_obs=np.asarray(item["x_obs"], dtype=float),
                    y_obs=np.asarray(item["y_obs"], dtype=float),
                    sigma_per_image=np.asarray(item["sigma_per_image"], dtype=float),
                )
                for item in meta.get("bin_data", [])
            ],
            lens_model_list=[str(name) for name in meta.get("lens_model_list", [])],
            reference=tuple(meta.get("reference", [0, 0.0, 0.0])),
            fit_mode=str(meta["fit_mode"]),
            profile_variant=str(
                meta.get(
                    "profile_variant",
                    meta.get("scaling_profile_variant", PROFILE_VARIANT_ORIGINAL),
                )
            ),
            compact_skip_factor=float(meta.get("compact_skip_factor", 1.0)),
            potfiles=[dict(item) for item in meta.get("potfiles", [])],
            scaling_component_records=[dict(item) for item in meta.get("scaling_component_records", [])],
            geometry_cache=(
                GeometryCache(**meta["geometry_cache"])
                if isinstance(meta.get("geometry_cache"), dict)
                else None
            ),
            svi_init_values=(
                {str(key): float(value) for key, value in dict(meta.get("svi_init_values") or {}).items()}
                if meta.get("svi_init_values") is not None
                else None
            ),
        )
        cli_args = _read_h5_json(handle, "cli_args_json", default={})
        init_diagnostics = _read_h5_json(handle, "init_diagnostics_json", default={})
        posterior_group = handle["posterior"]
        arrays = {name: np.asarray(posterior_group[name]) for name in posterior_group.keys()}
    return state, cli_args, arrays, init_diagnostics


class _LegacyArtifactUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:  # pragma: no cover - compatibility shim
        if module == "__main__" and name in globals():
            return globals()[name]
        return super().find_class(module, name)


def _load_legacy_artifacts(artifacts_dir: Path) -> tuple[BuildState, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    with (artifacts_dir / "build_state.pkl").open("rb") as handle:
        state = _LegacyArtifactUnpickler(handle).load()
    if not hasattr(state, "scaling_component_records"):
        state.scaling_component_records = []
    if not hasattr(state, "profile_variant"):
        state.profile_variant = getattr(
            state,
            "scaling_profile_variant",
            PROFILE_VARIANT_ORIGINAL,
        )
    if not hasattr(state, "compact_skip_factor"):
        state.compact_skip_factor = 1.0
    if not hasattr(state, "geometry_cache"):
        state.geometry_cache = None
    cli_args = json.loads((artifacts_dir / "cli_args.json").read_text())
    arrays = {key: np.asarray(value) for key, value in np.load(artifacts_dir / "posterior_arrays.npz").items()}
    init_diagnostics_path = artifacts_dir / "init_diagnostics.json"
    if init_diagnostics_path.exists():
        init_diagnostics = json.loads(init_diagnostics_path.read_text())
    else:
        init_diagnostics = {}
    return state, cli_args, arrays, init_diagnostics


def _load_artifacts(artifacts_dir: Path) -> tuple[BuildState, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    h5_path = artifacts_dir / "plot_bundle.h5"
    if h5_path.exists():
        state, cli_args, arrays, init_diagnostics = _rebuild_state_from_h5(h5_path)
    else:
        state, cli_args, arrays, init_diagnostics = _load_legacy_artifacts(artifacts_dir)
    if state.geometry_cache is None:
        cosmo = _build_cosmology_from_config(state.cosmo_config) if state.cosmo_config else _build_cosmology(state.parsed)
        state.geometry_cache = _build_geometry_cache(
            cosmo,
            state.z_lens,
            state.family_data,
            state.bin_data,
        )
    return state, cli_args, arrays, init_diagnostics


def _stage1_summary_from_results(parameter_specs: list[ParameterSpec], samples: np.ndarray, best_fit: np.ndarray) -> Stage1PriorSummary:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    map_values: dict[str, float] = {}
    if not parameter_specs:
        return Stage1PriorSummary(map_values=map_values, means=means, stds=stds)
    for idx, spec in enumerate(parameter_specs):
        means[spec.sample_name] = float(np.mean(samples[:, idx]))
        stds[spec.sample_name] = float(np.std(samples[:, idx]))
        map_values[spec.sample_name] = float(best_fit[idx])
    return Stage1PriorSummary(map_values=map_values, means=means, stds=stds)


def _save_stage1_summary(artifacts_dir: Path, summary: Stage1PriorSummary) -> None:
    payload = {
        "map_values": summary.map_values,
        "means": summary.means,
        "stds": summary.stds,
    }
    with (artifacts_dir / "stage1_prior_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_stage1_summary(artifacts_dir: Path) -> Stage1PriorSummary:
    summary_path = artifacts_dir / "stage1_prior_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text())
        return Stage1PriorSummary(
            map_values={str(key): float(value) for key, value in payload.get("map_values", {}).items()},
            means={str(key): float(value) for key, value in payload.get("means", {}).items()},
            stds={str(key): float(value) for key, value in payload.get("stds", {}).items()},
        )
    state, _cli_args, arrays, _init_diagnostics = _load_artifacts(artifacts_dir)
    return _stage1_summary_from_results(state.parameter_specs, np.asarray(arrays["samples"], dtype=float), np.asarray(arrays["best_fit"], dtype=float))



def _build_state_from_inputs(
    args: argparse.Namespace,
    fit_mode_override: str | None = None,
    stage1_prior_summary: Stage1PriorSummary | None = None,
) -> BuildState:
    fit_mode = fit_mode_override or args.fit_mode
    parsed, _potentials_df, images_df, potentials_with_priors = load_best_par(args.par_path)
    if images_df.empty:
        raise ValueError("No multiple-image constraints found in the parsed image catalog.")
    reference = _extract_reference(parsed)
    cosmo = _build_cosmology(parsed)
    z_lens_values = [float(pot.get("z_lens", 0.0)) for pot in potentials_with_priors if pot.get("z_lens") is not None]
    z_lens = float(z_lens_values[0]) if z_lens_values else 0.0
    scaling_kpc_per_arcsec = (
        float(cosmo.kpc_proper_per_arcmin(z_lens).to(u.kpc / u.arcsec).value)
        if z_lens > 0.0
        else 1.0
    )
    sigma_arcsec = float(args.pos_sigma_arcsec or parsed.get("image", {}).get("sigposArcsec", 0.5))
    potfiles = list(parsed.get("potfiles", []))
    large_parameter_specs, large_component_param_assignments, large_lens_model_list = _build_parameter_specs(
        potentials_with_priors,
        profile_variant=str(args.profile_variant),
    )
    if fit_mode == "small-only" and large_parameter_specs:
        if stage1_prior_summary is None:
            stage1_prior_summary = _load_stage1_summary(_infer_stage1_artifacts_dir(args))
        large_parameter_specs = _build_stage2_large_parameter_specs(large_parameter_specs, stage1_prior_summary)
    svi_init_values: dict[str, float] | None = None
    if fit_mode == "joint" and stage1_prior_summary is not None:
        svi_init_values = {}
        for spec in large_parameter_specs:
            if spec.sample_name not in stage1_prior_summary.map_values:
                continue
            physical_value = float(stage1_prior_summary.map_values[spec.sample_name])
            svi_init_values[spec.sample_name] = _physical_to_latent_numpy(physical_value, spec)
    parameter_specs = list(large_parameter_specs)
    component_param_assignments = list(large_component_param_assignments)
    lens_model_list = list(large_lens_model_list)
    base_components = [_serialize_component(potential) for potential in potentials_with_priors]
    scaling_component_assignments: list[dict[str, Any]] = []
    scaling_component_records: list[dict[str, Any]] = []
    if fit_mode in {"small-only", "joint"}:
        scaling_parameter_specs, scaling_param_indices, scaling_lens_model_list = _build_scaling_parameter_specs(
            potfiles,
            profile_variant=str(args.profile_variant),
            start_index=len(parameter_specs),
            kpc_per_arcsec=scaling_kpc_per_arcsec,
        )
        parameter_specs.extend(scaling_parameter_specs)
        scatter_fields = _parse_scaling_scatter_fields(args.scaling_scatter_fields) if args.scaling_scatter else set()
        scaling_scatter_specs, scaling_scatter_indices = _build_scaling_scatter_parameter_specs(
            potfiles,
            scatter_fields,
            start_index=len(parameter_specs),
            scatter_max=float(args.scaling_scatter_max),
        )
        parameter_specs.extend(scaling_scatter_specs)
        scaling_offset_specs, scaling_offset_indices = _build_scaling_offset_parameter_specs(
            potfiles,
            scatter_fields,
            scaling_scatter_indices,
            scaling_scatter_specs,
            start_index=len(parameter_specs),
        )
        parameter_specs.extend(scaling_offset_specs)
        scaling_components, scaling_assignments, scaling_component_assignments, scaling_component_records = _build_scaling_components(
            potfiles,
            reference,
            scaling_param_indices,
            scaling_offset_indices,
            start_component_index=len(base_components),
            kpc_per_arcsec=scaling_kpc_per_arcsec,
        )
        component_param_assignments.extend(scaling_assignments)
        lens_model_list.extend(scaling_lens_model_list)
        base_components.extend(scaling_components)
    parameter_specs.append(_build_source_scatter_parameter_spec(start_index=len(parameter_specs)))
    packed_lens_spec = _build_packed_lens_spec(base_components, component_param_assignments, scaling_component_assignments)
    family_data, family_redshift_binning_sec = _prepare_family_data(images_df, sigma_arcsec, reference, args.z_bin_tol)
    bin_data = _build_bin_data(family_data)
    cosmo_config = {
        "class": cosmo.__class__.__name__,
        "H0": float(getattr(cosmo, "H0").value),
        "Om0": float(getattr(cosmo, "Om0")),
        "Ode0": float(getattr(cosmo, "Ode0", 1.0 - getattr(cosmo, "Om0"))),
    }
    geometry_cache = _build_geometry_cache(
        cosmo,
        z_lens,
        family_data,
        bin_data,
        family_redshift_binning_sec=family_redshift_binning_sec,
    )
    return BuildState(
        run_name=args.run_name or _make_run_name(args.par_path),
        par_path=str(Path(args.par_path).resolve()),
        cosmo_config=cosmo_config,
        z_lens=z_lens,
        sigma_arcsec=sigma_arcsec,
        parsed=parsed,
        parameter_specs=parameter_specs,
        base_components=base_components,
        packed_lens_spec=packed_lens_spec,
        family_data=family_data,
        bin_data=bin_data,
        lens_model_list=lens_model_list,
        reference=reference,
        fit_mode=fit_mode,
        profile_variant=str(args.profile_variant),
        compact_skip_factor=float(args.compact_skip_factor),
        potfiles=potfiles,
        scaling_component_records=scaling_component_records,
        geometry_cache=geometry_cache,
        svi_init_values=svi_init_values,
    )


def _distribution_for_spec(spec: ParameterSpec):
    if spec.prior_kind == "hierarchical_normal":
        raise ValueError(f"Hierarchical parameter {spec.name} requires a parent distribution.")
    if spec.prior_kind == "normal":
        return dist.Normal(float(spec.mean), float(spec.std))
    return dist.Uniform(float(spec.lower), float(spec.upper))


def _prior_log_prob(parameter_specs: list[ParameterSpec], theta: jnp.ndarray) -> jnp.ndarray:
    theta_array = jnp.asarray(theta, dtype=jnp.float64)
    transform_kind_array = np.asarray([str(spec.transform_kind) for spec in parameter_specs], dtype=object)
    physical_theta = _apply_parameter_transforms_jax(
        theta_array,
        jnp.asarray(transform_kind_array == "log_positive", dtype=bool),
        jnp.asarray(transform_kind_array == "log_offset_positive", dtype=bool),
        jnp.asarray([float(spec.transform_offset) for spec in parameter_specs], dtype=jnp.float64),
    )
    sample_index = {spec.sample_name: idx for idx, spec in enumerate(parameter_specs)}
    total = jnp.array(0.0, dtype=jnp.float64)
    for idx, spec in enumerate(parameter_specs):
        if spec.prior_kind == "hierarchical_normal":
            if not spec.parent_sample_name:
                raise ValueError(f"Hierarchical parameter {spec.name} is missing parent_sample_name.")
            parent_idx = sample_index[spec.parent_sample_name]
            total = total + dist.Normal(float(spec.mean or 0.0), physical_theta[parent_idx]).log_prob(theta_array[idx])
        else:
            total = total + _distribution_for_spec(spec).log_prob(theta_array[idx])
    return total


def _posterior_model(parameter_specs: list[ParameterSpec], evaluator: ClusterJAXEvaluator):
    def model():
        values = []
        sampled: dict[str, Any] = {}
        spec_by_sample = {spec.sample_name: spec for spec in parameter_specs}
        for spec in parameter_specs:
            if spec.prior_kind == "hierarchical_normal":
                if not spec.parent_sample_name:
                    raise ValueError(f"Hierarchical parameter {spec.name} is missing parent_sample_name.")
                parent_value = _latent_to_physical_jax(sampled[spec.parent_sample_name], spec_by_sample[spec.parent_sample_name])
                value = numpyro.sample(
                    spec.sample_name,
                    dist.Normal(float(spec.mean or 0.0), jnp.asarray(parent_value, dtype=jnp.float64)),
                )
            else:
                value = numpyro.sample(spec.sample_name, _distribution_for_spec(spec))
            sampled[spec.sample_name] = value
            values.append(value)
        theta = jnp.stack(values)
        numpyro.factor("position_loglike", evaluator._source_loglike_fn(theta))

    return model


def _extract_samples(samples_dict: dict[str, np.ndarray], parameter_specs: list[ParameterSpec], thin: int) -> np.ndarray:
    if not parameter_specs:
        requested_chains = 0
        requested_draws = 0
        if samples_dict:
            first_array = np.asarray(next(iter(samples_dict.values())))
            if first_array.ndim >= 2:
                requested_chains = int(first_array.shape[0])
                requested_draws = int(first_array.shape[1])
        total_draws = requested_chains * requested_draws
        return np.empty((total_draws // max(1, thin), 0), dtype=float)
    arrays = [np.asarray(samples_dict[spec.sample_name], dtype=float) for spec in parameter_specs]
    stacked = np.stack(arrays, axis=-1)
    flat = stacked.reshape(-1, stacked.shape[-1])
    # Keep posterior samples in latent space; output code owns latent->physical conversion.
    return np.asarray(flat[:: max(1, thin)], dtype=float)


def _extract_grouped_samples(
    samples_dict: dict[str, np.ndarray],
    parameter_specs: list[ParameterSpec],
    thin: int,
) -> np.ndarray | None:
    if not parameter_specs:
        return np.empty((0, 0, 0), dtype=float)
    arrays = [np.asarray(samples_dict[spec.sample_name], dtype=float) for spec in parameter_specs]
    if not arrays:
        return None
    stacked = np.stack(arrays, axis=-1)
    # Keep grouped posterior samples in latent space; output code owns latent->physical conversion.
    return np.asarray(stacked[:, :: max(1, thin), :], dtype=float)


def _sanitize_grouped_posterior(
    samples_dict: dict[str, np.ndarray],
    extra: dict[str, Any],
    parameter_specs: list[ParameterSpec],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    if not parameter_specs:
        requested_chains = int(np.asarray(extra["potential_energy"]).shape[0])
        retained_indices = list(range(requested_chains))
        diagnostics = {
            "requested_chains": requested_chains,
            "retained_finite_chains": requested_chains,
            "dropped_nonfinite_chains": 0,
            "retained_chain_indices": retained_indices,
            "dropped_chain_indices": [],
        }
        return (
            {key: np.asarray(value) for key, value in samples_dict.items()},
            {key: np.asarray(value) for key, value in extra.items()},
            diagnostics,
        )
    arrays = [np.asarray(samples_dict[spec.sample_name], dtype=float) for spec in parameter_specs]
    stacked = np.stack(arrays, axis=-1)
    requested_chains = int(stacked.shape[0])
    sample_finite = np.isfinite(stacked).all(axis=tuple(range(1, stacked.ndim)))
    potential_energy = np.asarray(extra["potential_energy"], dtype=float)
    log_prob_finite = np.isfinite(potential_energy).all(axis=1)
    valid_chain_mask = sample_finite & log_prob_finite
    retained_indices = np.where(valid_chain_mask)[0].astype(int).tolist()
    dropped_indices = np.where(~valid_chain_mask)[0].astype(int).tolist()
    if not retained_indices:
        raise RuntimeError(
            "All NUTS chains produced non-finite posterior samples or log probabilities; no finite chains remain."
        )
    filtered_samples_dict = {
        key: np.asarray(value)[valid_chain_mask]
        for key, value in samples_dict.items()
    }
    filtered_extra = {
        key: np.asarray(value)[valid_chain_mask]
        for key, value in extra.items()
    }
    diagnostics = {
        "requested_chains": requested_chains,
        "retained_finite_chains": len(retained_indices),
        "dropped_nonfinite_chains": len(dropped_indices),
        "retained_chain_indices": retained_indices,
        "dropped_chain_indices": dropped_indices,
    }
    return filtered_samples_dict, filtered_extra, diagnostics


def _prepare_direct_evaluator(
    args: argparse.Namespace,
    state: BuildState,
) -> tuple[ClusterJAXEvaluator, np.ndarray]:
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=args.match_tolerance_arcsec,
        validate_top_k_families=args.validate_top_k_families,
        sampling_engine=args.sampling_engine,
        active_scaling_galaxies=args.active_scaling_galaxies,
        refresh_every=args.refresh_every,
        refresh_param_drift_frac=args.refresh_param_drift_frac,
        validation_approx=args.validation_approx,
    )
    midpoint = _default_theta(state.parameter_specs)
    if evaluator.surrogate_enabled:
        _log(
            args,
            (
                f"[surrogate] initializing active_scaling={len(evaluator.active_scaling_component_indices)} "
                f"inactive_scaling={len(evaluator.inactive_scaling_component_indices)}"
            ),
        )
        evaluator.refresh_surrogate(midpoint, reason="svi_nuts_initial")
    _log(args, "[compile] tracing first JAX likelihood evaluation")
    compile_start = time.time()
    compile_loglike = evaluator.source_loglike(midpoint)
    compile_elapsed = time.time() - compile_start
    evaluator.timing_totals["initial_jit_compile"] += compile_elapsed
    _log(args, f"[compile] initial trace complete in {_fmt_seconds(compile_elapsed)} loglike={compile_loglike:.3f}")
    return evaluator, midpoint


def _run_inference(args: argparse.Namespace, state: BuildState, run_dir: Path) -> None:
    start = time.time()
    _configure_debug_log(args, state.run_name, run_dir)
    _log(args, f"[load] run={state.run_name} par={state.par_path}")
    _log(
        args,
        (
            f"[model] parameters={len(state.parameter_specs)} families={len(state.family_data)} "
            f"images={sum(f.n_images for f in state.family_data)} z_bins={len(state.bin_data)}"
        ),
    )
    if len(state.parameter_specs) == 0:
        _log(args, "[model] no free parameters detected; running fixed-model evaluation only")
        evaluator, best_fit = _prepare_direct_evaluator(args, state)
        init_diagnostics = {
            "strategy_requested": "fixed_model",
            "strategy_used": "fixed_model",
            "svi_used": False,
            "distinct_chain_seeds": 0,
            "chain_seed_labels": [],
            "requested_chains": 0,
            "retained_finite_chains": 0,
            "dropped_nonfinite_chains": 0,
            "retained_chain_indices": [],
            "dropped_chain_indices": [],
            "invalid_state_rejection_count": int(evaluator.invalid_state_rejection_count),
            "invalid_state_reason_counts": {key: int(value) for key, value in evaluator.invalid_state_reason_counts.items()},
        }
        posterior = PosteriorResults(
            samples=np.empty((1, 0), dtype=float),
            log_prob=np.empty((0,), dtype=float),
            accept_prob=np.empty((0,), dtype=float),
            diverging=np.empty((0,), dtype=bool),
            num_steps=np.empty((0,), dtype=float),
            warmup_steps=0,
            sample_steps=0,
            num_chains=0,
            init_diagnostics=init_diagnostics,
            grouped_samples=np.empty((0, 0, 0), dtype=float),
            grouped_log_prob=np.empty((0, 0), dtype=float),
            sampler="fixed_model",
        )
        if args.skip_validation:
            _log(args, "[validation] skipped by --skip-validation; using source-plane summary only")
            best_eval = _run_logged_phase(args, "validation.approximate", lambda: _approximate_evaluation(evaluator, best_fit))
        else:
            validation_start = time.time()
            n_validate = len(state.family_data) if args.likelihood_mode == "image" else len(evaluator.validation_family_ids)
            _log(args, f"[validation] starting exact validation families={n_validate} mode={args.likelihood_mode}")
            best_eval = _run_logged_phase(
                args,
                "validation.evaluate",
                lambda: evaluator.evaluate(best_fit, likelihood_mode=args.likelihood_mode),
            )
            validation_elapsed = time.time() - validation_start
            evaluator.timing_totals["validation_runtime"] += validation_elapsed
            n_failed = sum(1 for info in best_eval.family_predictions.values() if info.get("failed"))
            _log(args, f"[validation] complete in {_fmt_seconds(validation_elapsed)} failed_families={n_failed}")
        runtime_sec = time.time() - start

        artifacts_dir = run_dir / "artifacts"
        _log(args, f"[output] saving artifacts to {artifacts_dir}")
        best_fit_physical = _run_logged_phase(
            args,
            "output.convert_best_fit_to_physical",
            lambda: _convert_theta_to_physical(best_fit, state.parameter_specs),
        )
        posterior_for_output = _run_logged_phase(
            args,
            "output.posterior_results_to_physical",
            lambda: _posterior_results_to_physical(posterior, state.parameter_specs),
        )
        _run_logged_phase(
            args,
            "output.save_artifacts",
            lambda: _save_artifacts(artifacts_dir, state, args, best_fit_physical, posterior_for_output),
        )
        if args.skip_plots:
            _log(args, "[output] plot generation skipped by --skip-plots")
        else:
            plot_start = time.time()
            _log(args, f"[output] generating plots and tables in {run_dir}")
            _run_logged_phase(
                args,
                "output.generate_plots_and_tables",
                lambda: _generate_plots_and_tables(
                    run_dir=run_dir,
                    state=state,
                    evaluator=evaluator,
                    best_fit=best_fit_physical,
                    best_eval=best_eval,
                    results=posterior_for_output,
                    runtime_sec=runtime_sec,
                    args=args,
                ),
            )
            plot_elapsed = time.time() - plot_start
            evaluator.timing_totals["plot_runtime"] += plot_elapsed
            _log(args, f"[output] complete in {_fmt_seconds(plot_elapsed)} run_dir={run_dir}")
        _log(args, f"[done] total_runtime={_fmt_seconds(time.time() - start)}")
        return
    _log(args, f"[model] initializing direct evaluator for {args.fit_method}")
    evaluator, _midpoint = _prepare_direct_evaluator(args, state)
    sample_model = _posterior_model(state.parameter_specs, evaluator)
    best_fit, posterior, svi_diagnostics = _run_svi_fit(args, state, evaluator, sample_model)
    _log(
        args,
        (
            "[svi:init] "
            f"final_elbo={float(svi_diagnostics.get('svi_final_elbo_loss', float('nan'))):.4g}"
        ),
    )
    if str(args.fit_method) == "svi+nuts":
        nuts_init = _nuts_initialization_from_svi_center(args, state.parameter_specs, best_fit, svi_diagnostics)
        _log(
            args,
            (
                "[nuts:init] from svi "
                f"distinct_seeds={int(nuts_init.diagnostics.get('distinct_chain_seeds', 0))}"
            ),
        )
        posterior = _run_numpyro_nuts_sampler(args, state, evaluator, sample_model, nuts_init)
        if posterior.samples.size:
            best_index = int(np.nanargmax(posterior.log_prob))
            best_fit = np.asarray(posterior.samples[best_index], dtype=float)
            _log(args, f"[nuts] best_fit updated from retained posterior sample index={best_index}")

    if args.skip_validation:
        _log(args, "[validation] skipped by --skip-validation; using source-plane summary only")
        best_eval = _run_logged_phase(args, "validation.approximate", lambda: _approximate_evaluation(evaluator, best_fit))
    else:
        validation_start = time.time()
        n_validate = len(state.family_data) if args.likelihood_mode == "image" else len(evaluator.validation_family_ids)
        _log(args, f"[validation] starting exact validation families={n_validate} mode={args.likelihood_mode}")
        best_eval = _run_logged_phase(
            args,
            "validation.evaluate",
            lambda: evaluator.evaluate(best_fit, likelihood_mode=args.likelihood_mode),
        )
        validation_elapsed = time.time() - validation_start
        evaluator.timing_totals["validation_runtime"] += validation_elapsed
        n_failed = sum(1 for info in best_eval.family_predictions.values() if info.get("failed"))
        _log(args, f"[validation] complete in {_fmt_seconds(validation_elapsed)} failed_families={n_failed}")
    runtime_sec = time.time() - start

    artifacts_dir = run_dir / "artifacts"
    _log(args, f"[output] saving artifacts to {artifacts_dir}")
    best_fit_physical = _run_logged_phase(
        args,
        "output.convert_best_fit_to_physical",
        lambda: _convert_theta_to_physical(best_fit, state.parameter_specs),
    )
    posterior_for_output = _run_logged_phase(
        args,
        "output.posterior_results_to_physical",
        lambda: _posterior_results_to_physical(posterior, state.parameter_specs),
    )
    _run_logged_phase(
        args,
        "output.save_artifacts",
        lambda: _save_artifacts(artifacts_dir, state, args, best_fit_physical, posterior_for_output),
    )
    if state.fit_mode == "large-only" and state.parameter_specs:
        _run_logged_phase(
            args,
            "output.save_stage1_summary",
            lambda: _save_stage1_summary(
                artifacts_dir,
                _stage1_summary_from_results(state.parameter_specs, posterior_for_output.samples, best_fit_physical),
            ),
        )
    if args.skip_plots:
        _log(args, "[output] plot generation skipped by --skip-plots")
    else:
        plot_start = time.time()
        _log(args, f"[output] generating plots and tables in {run_dir}")
        _run_logged_phase(
            args,
            "output.generate_plots_and_tables",
            lambda: _generate_plots_and_tables(
                run_dir=run_dir,
                state=state,
                evaluator=evaluator,
                best_fit=best_fit_physical,
                best_eval=best_eval,
                results=posterior_for_output,
                runtime_sec=runtime_sec,
                args=args,
            ),
        )
        plot_elapsed = time.time() - plot_start
        evaluator.timing_totals["plot_runtime"] += plot_elapsed
        _log(args, f"[output] complete in {_fmt_seconds(plot_elapsed)} run_dir={run_dir}")
    _log(args, f"[done] total_runtime={_fmt_seconds(time.time() - start)}")


def _rerender_plots(args: argparse.Namespace, run_dir: Path) -> None:
    _configure_debug_log(args, run_dir.name, run_dir)
    _log(args, f"[plots-only] loading artifacts from {run_dir / 'artifacts'}")
    state, saved_args, arrays, init_diagnostics = _run_logged_phase(
        args,
        "plots_only.load_artifacts",
        lambda: _load_artifacts(run_dir / "artifacts"),
    )
    arrays, converted_legacy_refine = _run_logged_phase(
        args,
        "plots_only.normalize_loaded_posterior",
        lambda: _maybe_convert_loaded_posterior_arrays_to_physical(arrays, state.parameter_specs, init_diagnostics),
    )
    if converted_legacy_refine:
        _log(args, "[plots-only] converted legacy saved posterior arrays from latent to physical units")
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=float(saved_args.get("match_tolerance_arcsec", DEFAULT_MATCH_TOLERANCE)),
        validate_top_k_families=int(saved_args.get("validate_top_k_families", 8)),
        sampling_engine=str(saved_args.get("sampling_engine", "full")),
        active_scaling_galaxies=saved_args.get("active_scaling_galaxies"),
        refresh_every=int(saved_args.get("refresh_every", DEFAULT_REFRESH_EVERY)),
        refresh_param_drift_frac=float(saved_args.get("refresh_param_drift_frac", DEFAULT_REFRESH_PARAM_DRIFT_FRAC)),
        validation_approx=str(saved_args.get("validation_approx", "exact")),
    )
    best_fit = np.asarray(arrays["best_fit"], dtype=float)
    best_fit_latent = _convert_theta_to_latent(best_fit, state.parameter_specs)
    if evaluator.surrogate_enabled:
        evaluator.refresh_surrogate(best_fit_latent, reason="plots_only")
    best_eval = _run_logged_phase(
        args,
        "plots_only.validation.evaluate",
        lambda: evaluator.evaluate(best_fit_latent, likelihood_mode=str(saved_args.get("likelihood_mode", "source"))),
    )
    posterior = PosteriorResults(
        samples=np.asarray(arrays["samples"], dtype=float),
        log_prob=np.asarray(arrays["log_prob"], dtype=float),
        accept_prob=np.asarray(arrays["accept_prob"], dtype=float),
        diverging=np.asarray(arrays["diverging"], dtype=bool),
        num_steps=np.asarray(arrays["num_steps"], dtype=float),
        warmup_steps=int(saved_args.get("warmup", 0)),
        sample_steps=int(saved_args.get("samples", 0)),
        num_chains=int(saved_args.get("chains", 1)),
        init_diagnostics=init_diagnostics,
        grouped_samples=np.asarray(arrays["grouped_samples"], dtype=float) if "grouped_samples" in arrays else None,
        grouped_log_prob=np.asarray(arrays["grouped_log_prob"], dtype=float) if "grouped_log_prob" in arrays else None,
        sampler=str(saved_args.get("sampler", DEFAULT_SAMPLER)),
        sample_weights=np.asarray(arrays["sample_weights"], dtype=float) if "sample_weights" in arrays else None,
        temperature_schedule=np.asarray(arrays["temperature_schedule"], dtype=float) if "temperature_schedule" in arrays else None,
        ess_history=np.asarray(arrays["ess_history"], dtype=float) if "ess_history" in arrays else None,
        move_acceptance_history=np.asarray(arrays["move_acceptance_history"], dtype=float) if "move_acceptance_history" in arrays else None,
    )
    plot_start = time.time()
    _log(args, f"[plots-only] regenerating outputs in {run_dir}")
    _run_logged_phase(
        args,
        "plots_only.generate_plots_and_tables",
        lambda: _generate_plots_and_tables(
            run_dir=run_dir,
            state=state,
            evaluator=evaluator,
            best_fit=best_fit,
            best_eval=best_eval,
            results=posterior,
            runtime_sec=0.0,
            args=args,
        ),
    )
    evaluator.timing_totals["plot_runtime"] += time.time() - plot_start
    _log(args, f"[plots-only] complete in {_fmt_seconds(evaluator.timing_totals['plot_runtime'])}")


def _clone_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    payload = vars(args).copy()
    payload.update(updates)
    return argparse.Namespace(**payload)


def _run_single_stage(
    args: argparse.Namespace,
    fit_mode: str,
    run_name: str,
    stage1_prior_summary: Stage1PriorSummary | None = None,
) -> Path:
    _configure_debug_log(args, run_name, None)
    stage_args = _clone_args(args, fit_mode=fit_mode, run_name=run_name)
    load_start = time.time()
    _log(stage_args, f"[load] parsing input from {stage_args.par_path}")
    state = _run_logged_phase(
        stage_args,
        "state.build",
        lambda: _build_state_from_inputs(stage_args, fit_mode_override=fit_mode, stage1_prior_summary=stage1_prior_summary),
        detail=f"fit_mode={fit_mode}",
    )
    _log(stage_args, f"[load] parser complete in {_fmt_seconds(time.time() - load_start)}")
    run_dir = Path(stage_args.output_dir) / state.run_name
    _run_inference(stage_args, state, run_dir)
    return run_dir


def _run_sequential(args: argparse.Namespace) -> None:
    root_run_name = args.run_name or _make_run_name(args.par_path)
    stage1_run_name = str(Path(root_run_name) / "stage1_large_only")
    stage1_args = _clone_args(args, fit_method="svi")
    stage1_run_dir = _run_single_stage(stage1_args, "large-only", stage1_run_name)
    stage1_summary = _load_stage1_summary(stage1_run_dir / "artifacts")
    stage2_run_name = str(Path(root_run_name) / "stage2_joint")
    stage2_run_dir = _run_single_stage(args, "joint", stage2_run_name, stage1_prior_summary=stage1_summary)
    summary_payload = {
        "fit_mode": "sequential",
        "fit_method": str(args.fit_method),
        "stage1_run_dir": str(stage1_run_dir),
        "stage2_run_dir": str(stage2_run_dir),
    }
    summary_path = Path(args.output_dir) / root_run_name / "sequential_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    _log(args, f"[done] sequential summary written to {summary_path}")


def _has_plot_artifacts(artifacts_dir: Path) -> bool:
    return (artifacts_dir / "plot_bundle.h5").exists() or (artifacts_dir / "posterior_arrays.npz").exists()


def _resolve_run_artifacts_dir(run_dir: str | Path) -> Path:
    candidate = Path(run_dir)
    return candidate if candidate.name == "artifacts" else candidate / "artifacts"


def main() -> None:
    try:
        args = _parse_args()
        if args.seed is not None:
            np.random.seed(args.seed)
        inferred_run_name = args.run_name
        if inferred_run_name is None and args.par_path:
            inferred_run_name = _make_run_name(args.par_path)
        else:
            inferred_run_name = inferred_run_name or "cluster_solver"
        _configure_debug_log(args, inferred_run_name, None)
        _log(args, "[main] startup")
        if not args.par_path and not args.plots_only:
            _fail("--par-path is required unless --plots-only is used.")
        if args.plots_only:
            root_run_name = args.run_name or _make_run_name(args.par_path)
            root_dir = Path(args.output_dir) / root_run_name
            stage_dirs = [root_dir / "stage1_large_only", root_dir / "stage2_joint"]
            if any(_has_plot_artifacts(stage_dir / "artifacts") for stage_dir in stage_dirs):
                for stage_dir in stage_dirs:
                    if _has_plot_artifacts(stage_dir / "artifacts"):
                        _configure_debug_log(args, stage_dir.name, stage_dir)
                        _rerender_plots(args, stage_dir)
                return
            run_dir = Path(args.output_dir) / (args.run_name or _make_run_name(args.par_path))
            _configure_debug_log(args, run_dir.name, run_dir)
            if not _has_plot_artifacts(run_dir / "artifacts"):
                _fail(f"Missing saved artifacts for plots-only mode: {run_dir / 'artifacts'}")
            _rerender_plots(args, run_dir)
            return
        _run_sequential(args)
    except BaseException as exc:
        _log_exception("main", exc)
        raise
    finally:
        _close_debug_log()


if __name__ == "__main__":
    main()
