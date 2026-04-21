from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import re
import resource
import time
import traceback
from dataclasses import dataclass, is_dataclass, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.optim as numpyro_optim
import optax
import pandas as pd
import h5py
from astropy import constants as astro_const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, LambdaCDM, wCDM
from jax import config as jax_config
from matplotlib import use as matplotlib_use
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import constrain_fn, unconstrain_fn
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm, qmc
from lenstronomy.LensModel.lens_model import LensModel as NPLensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver as NPLensEquationSolver

# Configure matplotlib before pyplot import.
_MPLCONFIGDIR = os.environ.get("MPLCONFIGDIR")
if not _MPLCONFIGDIR:
    _MPLCONFIGDIR = f"/tmp/mpl_cluster_solver_{os.getpid()}"
    os.environ["MPLCONFIGDIR"] = _MPLCONFIGDIR
Path(_MPLCONFIGDIR).mkdir(parents=True, exist_ok=True)
matplotlib_use("Agg")
import matplotlib.pyplot as plt

jax_config.update("jax_enable_x64", True)

from jaxtronomy.LensModel.lens_model_bulk import LensModelBulk
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.Util import param_util

from lenstool_parser import load_best_par

try:
    import corner
except ImportError:  # pragma: no cover
    corner = None


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
DEFAULT_MAP_BROAD_SEEDS = 8
DEFAULT_MAP_LOCAL_REFINE_SEEDS = 6
DEFAULT_MAP_LOCAL_JITTER_SCALE = 0.08
DEFAULT_CONTINUATION_SIGMA_SCALE = 2.5
DEFAULT_CONTINUATION_VALIDATION_TOP_K = 3
DEFAULT_NUTS_INIT_TOP_K = 0
DEFAULT_NUTS_INIT_BOUNDARY_FRAC = 0.02
DEFAULT_NUTS_INIT_JITTER_FRAC = 0.02
DEFAULT_NUTS_INIT_DEDUP_DISTANCE = 0.35
DEFAULT_SVI_STEPS = 2000
DEFAULT_SVI_LEARNING_RATE = 5.0e-3
DEFAULT_SAMPLER = "numpyro_nuts"
DEFAULT_SMC_PARTICLES = 512
DEFAULT_SMC_ESS_THRESHOLD = 0.5
DEFAULT_SMC_MOVE_STEPS = 8
DEFAULT_SMC_MOVE_SCALE = 0.03
DEFAULT_SMC_MAX_TEMPERING_STEPS = 64
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
_DEBUG_LOG_PATH: Path | None = None
_DEBUG_LOG_HANDLE = None
_DEBUG_LOG_STDOUT_ENABLED = True


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    sample_name: str
    potential_id: str
    profile_type: int
    field: str
    prior_kind: str
    lower: float
    upper: float
    step: float
    mean: float | None = None
    std: float | None = None
    component_family: str = "large"
    transform_kind: str = "identity"
    physical_lower: float | None = None
    physical_upper: float | None = None
    physical_mean: float | None = None
    physical_std: float | None = None
    transform_offset: float = 0.0


@dataclass
class SurrogateBinCache:
    effective_z_source: float
    inactive_alpha_x: np.ndarray
    inactive_alpha_y: np.ndarray
    inactive_alpha_dx_dparams: np.ndarray
    inactive_alpha_dy_dparams: np.ndarray


@dataclass
class FamilyData:
    family_id: str
    z_source: float
    effective_z_source: float
    sigma_arcsec: float
    image_labels: list[str]
    x_obs: np.ndarray
    y_obs: np.ndarray

    @property
    def n_images(self) -> int:
        return len(self.image_labels)

    @property
    def x_center(self) -> float:
        return float(np.mean(self.x_obs))

    @property
    def y_center(self) -> float:
        return float(np.mean(self.y_obs))

    @property
    def search_window(self) -> float:
        span_x = np.ptp(self.x_obs) if self.n_images > 1 else 0.0
        span_y = np.ptp(self.y_obs) if self.n_images > 1 else 0.0
        return float(max(span_x, span_y) + DEFAULT_SEARCH_PADDING)


@dataclass
class BinData:
    effective_z_source: float
    family_ids: list[str]
    family_index_per_image: np.ndarray
    x_obs: np.ndarray
    y_obs: np.ndarray
    sigma_per_image: np.ndarray


@dataclass(frozen=True)
class GeometryCache:
    effective_z_source_values: list[float]
    exact_z_source_values: list[float]
    family_z_source_map: dict[str, float]
    family_effective_z_source_map: dict[str, float]
    dpie_sigma0_factor_by_effective_z: dict[float, float]
    dpie_sigma0_factor_by_exact_z: dict[float, float]
    family_redshift_binning_sec: float = 0.0
    geometry_cache_build_sec: float = 0.0


@dataclass
class FamilyValidationCache:
    exact_validation_count: int = 0
    multiplicity_mismatch_count: int = 0
    match_failure_count: int = 0
    source_plane_rms: float | None = None
    exact_image_rms: float | None = None
    last_source_x: float | None = None
    last_source_y: float | None = None


@dataclass
class EvaluationResult:
    loglike: float
    family_predictions: dict[str, dict[str, Any]]
    used_exact_validation: bool


@dataclass
class PosteriorResults:
    samples: np.ndarray
    log_prob: np.ndarray
    accept_prob: np.ndarray
    diverging: np.ndarray
    num_steps: np.ndarray
    map_history: list[dict[str, Any]]
    warmup_steps: int
    sample_steps: int
    num_chains: int
    init_diagnostics: dict[str, Any] | None = None
    grouped_samples: np.ndarray | None = None
    grouped_log_prob: np.ndarray | None = None
    sampler: str = DEFAULT_SAMPLER
    sample_weights: np.ndarray | None = None
    temperature_schedule: np.ndarray | None = None
    ess_history: np.ndarray | None = None
    move_acceptance_history: np.ndarray | None = None


@dataclass(frozen=True)
class SMCRunResult:
    particles: np.ndarray
    log_prob: np.ndarray
    sample_weights: np.ndarray
    temperature_schedule: np.ndarray
    ess_history: np.ndarray
    move_acceptance_history: np.ndarray
    init_diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ContinuationPass:
    name: str
    sigma_scale: float
    validate_top_k_families: int
    validation_approx: str
    sampling_engine: str


@dataclass(frozen=True)
class SeededStart:
    values: np.ndarray
    label: str


@dataclass(frozen=True)
class MAPRunResult:
    theta: np.ndarray
    logprob: float
    iterations: int
    label: str


@dataclass(frozen=True)
class ChainSeed:
    values: np.ndarray
    source_label: str


@dataclass(frozen=True)
class NUTSInitialization:
    init_params: dict[str, jnp.ndarray]
    chain_seeds: list[ChainSeed]
    diagnostics: dict[str, Any]
    reference_theta: np.ndarray


@dataclass
class PackedLensSpec:
    profile_type: np.ndarray
    component_family: np.ndarray
    x_center_base: np.ndarray
    y_center_base: np.ndarray
    ellipticite_base: np.ndarray
    angle_pos_base: np.ndarray
    core_radius_kpc_base: np.ndarray
    cut_radius_kpc_base: np.ndarray
    v_disp_base: np.ndarray
    gamma_base: np.ndarray
    x_center_param_index: np.ndarray
    y_center_param_index: np.ndarray
    ellipticite_param_index: np.ndarray
    angle_pos_param_index: np.ndarray
    core_radius_param_index: np.ndarray
    cut_radius_param_index: np.ndarray
    v_disp_param_index: np.ndarray
    gamma_param_index: np.ndarray
    luminosity_ratio: np.ndarray
    sigma_ref_base: np.ndarray
    cut_ref_base: np.ndarray
    core_ref_base: np.ndarray
    vdslope_base: np.ndarray
    slope_base: np.ndarray
    sigma_ref_param_index: np.ndarray
    cut_ref_param_index: np.ndarray
    core_ref_param_index: np.ndarray
    vdslope_param_index: np.ndarray
    slope_param_index: np.ndarray


@dataclass
class Stage1PriorSummary:
    map_values: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]


@dataclass
class BuildState:
    run_name: str
    par_path: str
    cosmo_config: dict[str, Any]
    z_lens: float
    sigma_arcsec: float
    parsed: dict[str, Any]
    parameter_specs: list[ParameterSpec]
    base_components: list[dict[str, Any]]
    packed_lens_spec: PackedLensSpec
    family_data: list[FamilyData]
    bin_data: list[BinData]
    lens_model_list: list[str]
    reference: tuple[int, float, float]
    fit_mode: str
    profile_variant: str
    compact_skip_factor: float
    potfiles: list[dict[str, Any]]
    scaling_component_records: list[dict[str, Any]]
    geometry_cache: GeometryCache | None = None


def _positive_lognormal_parameters(mean: float, std: float, *, floor: float = 1.0e-6) -> tuple[float, float]:
    mean = max(float(mean), float(floor))
    std = max(float(std), 1.0e-9)
    variance = std * std
    sigma2 = np.log1p(variance / (mean * mean))
    return float(np.log(mean) - 0.5 * sigma2), float(np.sqrt(sigma2))


def _jax_profile_kwargs_list(lens_model_list: list[str], compact_skip_factor: float) -> list[dict[str, float]]:
    return [
        {"compact_skip_factor": float(compact_skip_factor)} if lens_type in COMPACT_PROFILE_NAMES else {}
        for lens_type in lens_model_list
    ]


def _physical_to_latent_numpy(value: float, spec: ParameterSpec) -> float:
    value = float(value)
    offset = float(spec.transform_offset)
    if spec.transform_kind == "log_positive":
        return float(np.log(max(value, 1.0e-12)))
    if spec.transform_kind == "log_offset_positive":
        return float(np.log(max(value - offset, 1.0e-12)))
    return value


def _latent_to_physical_numpy(value: float, spec: ParameterSpec) -> float:
    value = float(value)
    if spec.transform_kind == "log_positive":
        return float(np.exp(value))
    if spec.transform_kind == "log_offset_positive":
        return float(spec.transform_offset + np.exp(value))
    return value


def _latent_to_physical_array(values: np.ndarray, spec: ParameterSpec) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if spec.transform_kind == "log_positive":
        return np.exp(array)
    if spec.transform_kind == "log_offset_positive":
        return float(spec.transform_offset) + np.exp(array)
    return array


def _latent_to_physical_jax(values: jnp.ndarray, spec: ParameterSpec) -> jnp.ndarray:
    array = jnp.asarray(values, dtype=jnp.float64)
    if spec.transform_kind == "log_positive":
        return jnp.exp(array)
    if spec.transform_kind == "log_offset_positive":
        return jnp.asarray(spec.transform_offset, dtype=jnp.float64) + jnp.exp(array)
    return array


def _display_lower(spec: ParameterSpec) -> float:
    return float(spec.physical_lower if spec.physical_lower is not None else spec.lower)


def _display_upper(spec: ParameterSpec) -> float:
    return float(spec.physical_upper if spec.physical_upper is not None else spec.upper)


def _convert_theta_to_physical(theta: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    theta_array = np.asarray(theta, dtype=float)
    if theta_array.size == 0:
        return theta_array.copy()
    converted = theta_array.copy()
    for idx, spec in enumerate(parameter_specs):
        converted[idx] = _latent_to_physical_numpy(theta_array[idx], spec)
    return converted


def _convert_theta_to_latent(theta: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    theta_array = np.asarray(theta, dtype=float)
    if theta_array.size == 0:
        return theta_array.copy()
    converted = theta_array.copy()
    for idx, spec in enumerate(parameter_specs):
        converted[idx] = _physical_to_latent_numpy(theta_array[idx], spec)
    return converted


def _convert_sample_matrix_to_latent(samples: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.size == 0:
        return array.copy()
    converted = array.copy()
    for idx, spec in enumerate(parameter_specs):
        converted[..., idx] = np.asarray([_physical_to_latent_numpy(value, spec) for value in array[..., idx].reshape(-1)], dtype=float).reshape(array[..., idx].shape)
    return converted


def _state_with_run_name(state: BuildState, run_name: str) -> BuildState:
    if is_dataclass(state):
        return replace(state, run_name=run_name)
    cloned = copy.copy(state)
    setattr(cloned, "run_name", run_name)
    return cloned


def _apply_parameter_transforms_jax(
    params: jnp.ndarray,
    log_positive_mask: jnp.ndarray,
    log_offset_positive_mask: jnp.ndarray,
    transform_offset_array: jnp.ndarray,
) -> jnp.ndarray:
    params_array = jnp.asarray(params, dtype=jnp.float64)
    safe_log_positive_params = jnp.where(log_positive_mask, params_array, 0.0)
    physical_params = jnp.where(
        log_positive_mask,
        jnp.exp(safe_log_positive_params),
        params_array,
    )
    safe_log_offset_params = jnp.where(log_offset_positive_mask, params_array, 0.0)
    physical_params = jnp.where(
        log_offset_positive_mask,
        transform_offset_array + jnp.exp(safe_log_offset_params),
        physical_params,
    )
    return physical_params


def _convert_sample_matrix_to_physical(samples: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.size == 0:
        return array.copy()
    converted = array.copy()
    for idx, spec in enumerate(parameter_specs):
        converted[..., idx] = _latent_to_physical_array(array[..., idx], spec)
    return converted


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
    chosen_ranked_maps = init_diagnostics.get("chosen_ranked_maps")
    if isinstance(chosen_ranked_maps, list):
        converted_maps: list[dict[str, Any]] = []
        for item in chosen_ranked_maps:
            if not isinstance(item, dict) or "theta" not in item:
                converted_maps.append(item)
                continue
            converted = dict(item)
            converted["theta"] = _convert_theta_to_physical(np.asarray(item["theta"], dtype=float), parameter_specs)
            converted_maps.append(converted)
        init_diagnostics["chosen_ranked_maps"] = converted_maps
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
    if not bool(diagnostics.get("saved_smc_refine", False)):
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


def _parse_bool_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster dPIE solver with JAXtronomy + NumPyro.")
    parser.add_argument("--par-path", required=False, help="Path to input_a_sl.par")
    parser.add_argument("--output-dir", default="plots", help="Base output directory")
    parser.add_argument("--run-name", default=None, help="Optional run name")
    parser.add_argument(
        "--refine-from-run-dir",
        default=None,
        help="Existing run directory whose saved blackjax_smc artifacts should seed a NUTS-only refinement run.",
    )
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
        "--fit-mode",
        choices=("large-only", "small-only", "sequential"),
        default="sequential",
        help="Run only large-scale halos, only potfile scaling-law halos with stage-1 priors, or both sequentially.",
    )
    parser.add_argument(
        "--stage1-run-dir",
        default=None,
        help="Existing stage-1 run directory to seed stage-2 large-scale Gaussian priors.",
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
        "--map-broad-seeds",
        type=int,
        default=DEFAULT_MAP_BROAD_SEEDS,
        help="Number of broad prior-covering MAP start points per continuation pass.",
    )
    parser.add_argument(
        "--map-local-refine-seeds",
        type=int,
        default=DEFAULT_MAP_LOCAL_REFINE_SEEDS,
        help="Number of local refinement MAP starts spawned around the strongest broad solutions.",
    )
    parser.add_argument(
        "--map-local-jitter-scale",
        type=float,
        default=DEFAULT_MAP_LOCAL_JITTER_SCALE,
        help="Jitter scale for local refinement starts, expressed as a fraction of prior width or prior std.",
    )
    parser.add_argument(
        "--continuation-sigma-scale",
        type=float,
        default=DEFAULT_CONTINUATION_SIGMA_SCALE,
        help="Inflation factor applied to positional sigma in the first continuation pass.",
    )
    parser.add_argument(
        "--continuation-validation-top-k",
        type=int,
        default=DEFAULT_CONTINUATION_VALIDATION_TOP_K,
        help="Validation breadth used in the first continuation pass before tightening to the final setting.",
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
    parser.add_argument(
        "--sampler",
        choices=("numpyro_nuts", "blackjax_smc"),
        default=DEFAULT_SAMPLER,
        help="Posterior sampler to run after continuation MAP initialization.",
    )
    parser.add_argument("--max-tree-depth", type=int, default=DEFAULT_MAX_TREE_DEPTH)
    parser.add_argument("--target-accept", type=float, default=DEFAULT_TARGET_ACCEPT)
    parser.add_argument(
        "--nuts-init-strategy",
        choices=("ranked_map", "svi+nuts", "prior_center"),
        default="ranked_map",
        help="Initializer for NUTS chains: ranked_map runs continuation MAP first, svi+nuts skips MAP and uses SVI, prior_center skips MAP and starts from the prior center.",
    )
    parser.add_argument(
        "--nuts-init-top-k",
        type=int,
        default=DEFAULT_NUTS_INIT_TOP_K,
        help="Number of ranked MAP candidates to consider when building NUTS chain starts. 0 uses max(chains*2, 8).",
    )
    parser.add_argument(
        "--nuts-init-boundary-frac",
        type=float,
        default=DEFAULT_NUTS_INIT_BOUNDARY_FRAC,
        help="Reject uniform-prior MAP candidates that sit within this fraction of a support edge.",
    )
    parser.add_argument(
        "--nuts-init-jitter-frac",
        type=float,
        default=DEFAULT_NUTS_INIT_JITTER_FRAC,
        help="Constrained-space jitter scale for uniform-prior NUTS seeds, as a fraction of prior width.",
    )
    parser.add_argument(
        "--nuts-init-dedup-distance",
        type=float,
        default=DEFAULT_NUTS_INIT_DEDUP_DISTANCE,
        help="Scaled-distance threshold below which MAP candidates are treated as duplicates.",
    )
    parser.add_argument(
        "--svi-steps",
        type=int,
        default=DEFAULT_SVI_STEPS,
        help="Number of SVI steps when using the svi+nuts initializer.",
    )
    parser.add_argument(
        "--svi-learning-rate",
        type=float,
        default=DEFAULT_SVI_LEARNING_RATE,
        help="Learning rate for SVI when using the svi+nuts initializer.",
    )
    parser.add_argument(
        "--smc-particles",
        type=int,
        default=DEFAULT_SMC_PARTICLES,
        help="Number of SMC particles for the blackjax_smc sampler.",
    )
    parser.add_argument(
        "--smc-ess-threshold",
        type=float,
        default=DEFAULT_SMC_ESS_THRESHOLD,
        help="Relative ESS threshold used to adapt the next SMC temperature.",
    )
    parser.add_argument(
        "--smc-move-steps",
        type=int,
        default=DEFAULT_SMC_MOVE_STEPS,
        help="Number of random-walk rejuvenation steps per SMC temperature level.",
    )
    parser.add_argument(
        "--smc-move-scale",
        type=float,
        default=DEFAULT_SMC_MOVE_SCALE,
        help="Random-walk proposal scale for SMC moves, as a fraction of the prior width or prior std.",
    )
    parser.add_argument(
        "--smc-seed-mode",
        choices=("prior", "ranked_map"),
        default="ranked_map",
        help="Initialize SMC particles from the prior or jittered ranked MAP seeds.",
    )
    parser.add_argument(
        "--refine-with-nuts",
        action="store_true",
        help="After SMC, run local NUTS chains from the strongest SMC particles.",
    )
    parser.add_argument(
        "--map-maxiter",
        type=int,
        default=250,
        help="Maximum L-BFGS iterations per MAP restart.",
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
        help="Generate the expensive caustic overlay plot even for small-only exploratory runs.",
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


def _parameter_scale(spec: ParameterSpec) -> float:
    if spec.prior_kind == "normal":
        return max(float(spec.std or 0.0), 1.0e-3)
    return max(float(spec.upper - spec.lower), 1.0e-3)


def _resolved_nuts_init_top_k(args: argparse.Namespace) -> int:
    requested = int(getattr(args, "nuts_init_top_k", 0))
    if requested > 0:
        return requested
    return max(int(args.chains) * 2, 8)


def _scaled_candidate_distance(
    theta_a: np.ndarray,
    theta_b: np.ndarray,
    parameter_specs: list[ParameterSpec],
) -> float:
    scales = np.asarray([_parameter_scale(spec) for spec in parameter_specs], dtype=float)
    diff = (np.asarray(theta_a, dtype=float) - np.asarray(theta_b, dtype=float)) / scales
    return float(np.linalg.norm(diff))


def _is_near_uniform_boundary(theta: np.ndarray, parameter_specs: list[ParameterSpec], boundary_frac: float) -> bool:
    theta_array = np.asarray(theta, dtype=float)
    for idx, spec in enumerate(parameter_specs):
        if spec.prior_kind != "uniform":
            continue
        width = float(spec.upper - spec.lower)
        margin = float(boundary_frac) * width
        if theta_array[idx] <= float(spec.lower) + margin or theta_array[idx] >= float(spec.upper) - margin:
            return True
    return False


def _deduplicate_ranked_candidates(
    parameter_specs: list[ParameterSpec],
    ranked_results: list[MAPRunResult],
    top_k: int,
    boundary_frac: float,
    dedup_distance: float,
) -> tuple[list[MAPRunResult], dict[str, int]]:
    selected: list[MAPRunResult] = []
    boundary_rejected = 0
    duplicate_rejected = 0
    considered = min(len(ranked_results), max(0, int(top_k)))
    for result in ranked_results[:considered]:
        if _is_near_uniform_boundary(result.theta, parameter_specs, boundary_frac):
            boundary_rejected += 1
            continue
        if any(
            _scaled_candidate_distance(result.theta, existing.theta, parameter_specs) < float(dedup_distance)
            for existing in selected
        ):
            duplicate_rejected += 1
            continue
        selected.append(result)
    return selected, {
        "ranked_candidates_considered": int(considered),
        "near_boundary_rejected": int(boundary_rejected),
        "duplicate_rejected": int(duplicate_rejected),
    }


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
        for spec in parameter_specs:
            numpyro.sample(spec.sample_name, _distribution_for_spec(spec))

    return model


def _values_dict_to_theta(
    parameter_specs: list[ParameterSpec],
    values: dict[str, Any],
) -> np.ndarray:
    return np.asarray([float(values[spec.sample_name]) for spec in parameter_specs], dtype=float)


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


def _build_ranked_map_chain_seeds(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
    ranked_results: list[MAPRunResult],
    rng: np.random.Generator,
) -> tuple[list[ChainSeed], dict[str, Any], list[MAPRunResult]]:
    top_k = _resolved_nuts_init_top_k(args)
    filtered, counters = _deduplicate_ranked_candidates(
        parameter_specs,
        ranked_results,
        top_k=top_k,
        boundary_frac=float(args.nuts_init_boundary_frac),
        dedup_distance=float(args.nuts_init_dedup_distance),
    )
    if not filtered:
        fallback = sorted(ranked_results, key=lambda item: item.logprob, reverse=True)
        filtered = fallback[: max(1, min(len(fallback), int(args.chains)))]
    chain_seeds: list[ChainSeed] = []
    for chain_index in range(int(args.chains)):
        anchor = filtered[chain_index % len(filtered)]
        seed_values = _jitter_theta_in_support(
            anchor.theta,
            parameter_specs,
            float(args.nuts_init_jitter_frac),
            rng,
            boundary_frac=float(args.nuts_init_boundary_frac),
        )
        chain_seeds.append(ChainSeed(values=seed_values, source_label=anchor.label))
    diagnostics = {
        **counters,
        "resolved_top_k": int(top_k),
        "distinct_ranked_map_candidates": int(len(filtered)),
        "chain_seed_labels": [seed.source_label for seed in chain_seeds],
        "chain_seed_diversity": int(len({tuple(np.round(seed.values, 8)) for seed in chain_seeds})),
        "chosen_ranked_maps": [
            {
                "label": str(result.label),
                "logprob": float(result.logprob),
                "theta": np.asarray(result.theta, dtype=float),
            }
            for result in filtered
        ],
    }
    return chain_seeds, diagnostics, filtered


def _run_svi_initializer(
    args: argparse.Namespace,
    sample_model,
    parameter_specs: list[ParameterSpec],
    rng_key: jax.Array,
) -> tuple[list[ChainSeed], dict[str, Any], np.ndarray]:
    guide = AutoNormal(sample_model)
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


def _build_prior_center_chain_seeds(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
) -> tuple[list[ChainSeed], dict[str, Any], np.ndarray]:
    center_theta = _default_theta(parameter_specs)
    rng = np.random.default_rng(None if args.seed is None else int(args.seed) + 606)
    chain_seeds: list[ChainSeed] = []
    chain_labels: list[str] = []
    for chain_index in range(int(args.chains)):
        seed_theta = _jitter_theta_in_support(
            center_theta,
            parameter_specs,
            float(args.nuts_init_jitter_frac),
            rng,
            boundary_frac=float(args.nuts_init_boundary_frac),
        )
        label = f"prior_center_chain_{chain_index + 1}"
        chain_seeds.append(ChainSeed(values=seed_theta, source_label=label))
        chain_labels.append(label)
    diagnostics = {
        "strategy_requested": str(args.nuts_init_strategy),
        "strategy_used": "prior_center",
        "svi_used": False,
        "chain_seed_labels": chain_labels,
        "distinct_chain_seeds": int(len({tuple(np.round(seed.values, 8)) for seed in chain_seeds})),
    }
    return chain_seeds, diagnostics, center_theta


def _build_nuts_initialization(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
    ranked_results: list[MAPRunResult] | None,
    sample_model,
) -> NUTSInitialization:
    init_model = _sample_site_model(parameter_specs)
    if str(args.nuts_init_strategy) == "ranked_map" and not ranked_results:
        raise ValueError("Ranked MAP results are required to initialize NUTS.")
    if str(args.nuts_init_strategy) == "prior_center":
        chain_seeds, diagnostics, center_theta = _build_prior_center_chain_seeds(args, parameter_specs)
        return NUTSInitialization(
            init_params=_seed_values_to_init_params(parameter_specs, chain_seeds, model_for_init=init_model),
            chain_seeds=chain_seeds,
            diagnostics=diagnostics,
            reference_theta=np.asarray(center_theta, dtype=float),
        )
    ranked_map_seeds: list[ChainSeed] = []
    ranked_diag: dict[str, Any] = {}
    chosen_ranked_map_results: list[MAPRunResult] = []
    reference_theta: np.ndarray
    if ranked_results:
        rng = np.random.default_rng(None if args.seed is None else int(args.seed) + 101)
        ranked_map_seeds, ranked_diag, chosen_ranked_map_results = _build_ranked_map_chain_seeds(
            args, parameter_specs, ranked_results, rng
        )
        reference_theta = np.asarray(ranked_results[0].theta, dtype=float)
    else:
        reference_theta = _default_theta(parameter_specs)
    chain_seeds = ranked_map_seeds
    diagnostics: dict[str, Any] = {
        "strategy_requested": str(args.nuts_init_strategy),
        "strategy_used": "ranked_map",
        **ranked_diag,
        "svi_used": False,
    }
    if str(args.nuts_init_strategy) == "svi+nuts":
        svi_seeds, svi_diag, svi_center_theta = _run_svi_initializer(
            args,
            sample_model,
            parameter_specs,
            jax.random.PRNGKey(0 if args.seed is None else int(args.seed) + 202),
        )
        chain_seeds = svi_seeds
        diagnostics["strategy_used"] = "svi+nuts"
        diagnostics["svi_used"] = True
        diagnostics.update(svi_diag)
        reference_theta = np.asarray(svi_center_theta, dtype=float)
    diagnostics["chain_seed_labels"] = [seed.source_label for seed in chain_seeds]
    diagnostics["distinct_chain_seeds"] = int(len({tuple(np.round(seed.values, 8)) for seed in chain_seeds}))
    if chosen_ranked_map_results:
        diagnostics["chosen_ranked_maps"] = [
            {
                "label": str(result.label),
                "logprob": float(result.logprob),
                "theta": np.asarray(result.theta, dtype=float),
            }
            for result in chosen_ranked_map_results
        ]
    init_params = _seed_values_to_init_params(parameter_specs, chain_seeds, model_for_init=init_model)
    return NUTSInitialization(
        init_params=init_params,
        chain_seeds=chain_seeds,
        diagnostics=diagnostics,
        reference_theta=np.asarray(reference_theta, dtype=float),
    )


def _parameter_scales_array(parameter_specs: list[ParameterSpec], scale_frac: float = 1.0) -> np.ndarray:
    scales = []
    for spec in parameter_specs:
        if spec.prior_kind == "normal":
            scale = max(float(spec.std or 0.0), 1.0e-6)
        else:
            scale = max(float(spec.upper - spec.lower), 1.0e-6)
        scales.append(scale * float(scale_frac))
    return np.asarray(scales, dtype=float)


def _prior_sample_matrix(
    parameter_specs: list[ParameterSpec],
    n_particles: int,
    seed: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(None if seed is None else int(seed) + 404)
    draws = np.empty((int(n_particles), len(parameter_specs)), dtype=float)
    for idx, spec in enumerate(parameter_specs):
        if spec.prior_kind == "normal":
            draws[:, idx] = rng.normal(float(spec.mean), max(float(spec.std or 0.0), 1.0e-6), size=int(n_particles))
        else:
            draws[:, idx] = rng.uniform(float(spec.lower), float(spec.upper), size=int(n_particles))
    return _clip_theta_matrix_to_support(draws, parameter_specs)


def _clip_theta_matrix_to_support(
    theta_matrix: np.ndarray,
    parameter_specs: list[ParameterSpec],
    boundary_frac: float = 0.0,
) -> np.ndarray:
    clipped = np.asarray(theta_matrix, dtype=float).copy()
    for row_index in range(clipped.shape[0]):
        clipped[row_index] = _clip_theta_to_support(clipped[row_index], parameter_specs, boundary_frac=boundary_frac)
    return clipped


def _build_ranked_map_particles(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
    ranked_results: list[MAPRunResult],
    n_particles: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not ranked_results:
        return _prior_sample_matrix(parameter_specs, n_particles, args.seed), {
            "smc_seed_mode_used": "prior_fallback",
            "smc_seed_labels": [],
        }
    rng = np.random.default_rng(None if args.seed is None else int(args.seed) + 505)
    _, ranked_diag, filtered = _build_ranked_map_chain_seeds(args, parameter_specs, ranked_results, rng)
    anchors = filtered or ranked_results
    scales = _parameter_scales_array(parameter_specs, scale_frac=float(args.smc_move_scale))
    particles = np.empty((int(n_particles), len(parameter_specs)), dtype=float)
    labels: list[str] = []
    for particle_index in range(int(n_particles)):
        anchor = anchors[particle_index % len(anchors)]
        particle = np.asarray(anchor.theta, dtype=float) + rng.normal(0.0, scales, size=len(parameter_specs))
        particles[particle_index] = particle
        labels.append(str(anchor.label))
    particles = _clip_theta_matrix_to_support(
        particles,
        parameter_specs,
        boundary_frac=float(getattr(args, "nuts_init_boundary_frac", DEFAULT_NUTS_INIT_BOUNDARY_FRAC)),
    )
    diagnostics = dict(ranked_diag)
    diagnostics.update(
        {
            "smc_seed_mode_used": "ranked_map",
            "smc_seed_labels": labels[: min(len(labels), 16)],
        }
    )
    return particles, diagnostics


def _smc_initial_particles(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
    ranked_results: list[MAPRunResult],
) -> tuple[np.ndarray, dict[str, Any]]:
    if str(args.smc_seed_mode) == "prior":
        return _prior_sample_matrix(parameter_specs, int(args.smc_particles), args.seed), {
            "smc_seed_mode_used": "prior",
            "smc_seed_labels": [],
        }
    if not ranked_results:
        raise ValueError(
            "SMC seed mode 'ranked_map' requires MAP-ranked candidates, but blackjax_smc now skips the MAP stage. "
            "Use '--smc-seed-mode prior' or switch to the numpyro_nuts workflow."
        )
    return _build_ranked_map_particles(args, parameter_specs, ranked_results, int(args.smc_particles))


def _smc_prior_logprob_matrix(theta: jnp.ndarray, parameter_specs: list[ParameterSpec]) -> jnp.ndarray:
    total = jnp.zeros(theta.shape[0], dtype=jnp.float64)
    for idx, spec in enumerate(parameter_specs):
        total = total + _distribution_for_spec(spec).log_prob(theta[:, idx])
    return total


def _smc_clip_proposals(theta: jnp.ndarray, parameter_specs: list[ParameterSpec]) -> jnp.ndarray:
    lower = jnp.asarray([spec.lower for spec in parameter_specs], dtype=jnp.float64)
    upper = jnp.asarray([spec.upper for spec in parameter_specs], dtype=jnp.float64)
    clipped = jnp.clip(theta, lower, upper)
    for idx, spec in enumerate(parameter_specs):
        if spec.field not in {"vdslope", "slope"}:
            continue
        if float(spec.lower) >= SAFE_SCALING_EXPONENT_ABS_MIN or float(spec.upper) <= -SAFE_SCALING_EXPONENT_ABS_MIN:
            continue
        values = clipped[:, idx]
        signs = jnp.where(values < 0.0, -1.0, 1.0)
        values = jnp.where(jnp.abs(values) < SAFE_SCALING_EXPONENT_ABS_MIN, signs * SAFE_SCALING_EXPONENT_ABS_MIN, values)
        clipped = clipped.at[:, idx].set(jnp.clip(values, float(spec.lower), float(spec.upper)))
    return clipped


def _smc_find_next_beta(current_beta: float, loglike: np.ndarray, ess_target: float) -> tuple[float, float]:
    if current_beta >= 1.0:
        return 1.0, float(len(loglike))
    n_particles = len(loglike)
    target_ess = max(1.0, float(ess_target) * float(n_particles))

    def ess_for_beta(beta_value: float) -> float:
        logw = (beta_value - current_beta) * loglike
        logw = logw - np.max(logw)
        weights = np.exp(logw)
        weights = weights / np.sum(weights)
        return float(1.0 / np.sum(np.square(weights)))

    full_ess = ess_for_beta(1.0)
    if full_ess >= target_ess:
        return 1.0, full_ess

    lo = float(current_beta)
    hi = 1.0
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if ess_for_beta(mid) < target_ess:
            hi = mid
        else:
            lo = mid
    beta = max(current_beta, min(1.0, lo))
    return beta, ess_for_beta(beta)


def _smc_device_logsumexp(values: jnp.ndarray) -> jnp.ndarray:
    return jsp.special.logsumexp(values)


def _smc_effective_sample_size_from_log_weights(log_weights: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    logw_norm = log_weights - _smc_device_logsumexp(log_weights)
    weights = jnp.exp(logw_norm)
    ess = 1.0 / jnp.sum(jnp.square(weights))
    return ess, logw_norm


def _smc_find_next_beta_jax(
    current_beta: jnp.ndarray,
    loglike: jnp.ndarray,
    ess_threshold: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n_particles = loglike.shape[0]
    target_ess = jnp.maximum(1.0, jnp.asarray(float(ess_threshold) * float(n_particles), dtype=jnp.float64))

    def ess_and_logw(beta_value: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _smc_effective_sample_size_from_log_weights((beta_value - current_beta) * loglike)

    full_ess, full_logw = ess_and_logw(jnp.asarray(1.0, dtype=jnp.float64))

    def use_full(_):
        return jnp.asarray(1.0, dtype=jnp.float64), full_ess, full_logw

    def search(_):
        def body(_index, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            mid_ess, _mid_logw = ess_and_logw(mid)
            lo = jnp.where(mid_ess >= target_ess, mid, lo)
            hi = jnp.where(mid_ess < target_ess, mid, hi)
            return lo, hi

        lo0 = current_beta
        hi0 = jnp.asarray(1.0, dtype=jnp.float64)
        lo, _hi = jax.lax.fori_loop(0, 32, body, (lo0, hi0))
        beta = jnp.clip(lo, current_beta, 1.0)
        ess, logw_norm = ess_and_logw(beta)
        return beta, ess, logw_norm

    return jax.lax.cond(full_ess >= target_ess, use_full, search, operand=None)


def _run_blackjax_smc(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
    evaluator: ClusterJAXEvaluator,
    ranked_results: list[MAPRunResult],
) -> SMCRunResult:
    if int(args.smc_particles) <= 0:
        raise ValueError("--smc-particles must be positive.")
    if int(args.smc_move_steps) < 0:
        raise ValueError("--smc-move-steps must be non-negative.")
    if not (0.0 < float(args.smc_ess_threshold) <= 1.0):
        raise ValueError("--smc-ess-threshold must be in (0, 1].")

    particles_np, init_diag = _smc_initial_particles(args, parameter_specs, ranked_results)
    particles = jnp.asarray(particles_np, dtype=jnp.float64)
    batched_loglike = jax.jit(jax.vmap(evaluator._source_loglike_fn))
    move_scales = jnp.asarray(_parameter_scales_array(parameter_specs, scale_frac=float(args.smc_move_scale)), dtype=jnp.float64)
    max_tempering_steps = DEFAULT_SMC_MAX_TEMPERING_STEPS

    @jax.jit
    def _move_particles(
        key: jax.Array,
        theta: jnp.ndarray,
        loglike: jnp.ndarray,
        logprior: jnp.ndarray,
        beta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def body(carry, _step_index):
            rng_key, current_theta, current_loglike, current_logprior, accepted_total = carry
            rng_key, noise_key, accept_key = jax.random.split(rng_key, 3)
            proposal = current_theta + jax.random.normal(noise_key, shape=current_theta.shape, dtype=jnp.float64) * move_scales
            proposal = _smc_clip_proposals(proposal, parameter_specs)
            proposal_loglike = batched_loglike(proposal)
            proposal_logprior = _smc_prior_logprob_matrix(proposal, parameter_specs)
            log_alpha = beta * (proposal_loglike - current_loglike) + (proposal_logprior - current_logprior)
            accept_draw = jnp.log(jax.random.uniform(accept_key, shape=log_alpha.shape, dtype=jnp.float64))
            accept = accept_draw < log_alpha
            next_theta = jnp.where(accept[:, None], proposal, current_theta)
            next_loglike = jnp.where(accept, proposal_loglike, current_loglike)
            next_logprior = jnp.where(accept, proposal_logprior, current_logprior)
            accepted_total = accepted_total + accept.astype(jnp.float64)
            return (rng_key, next_theta, next_loglike, next_logprior, accepted_total), None

        init = (
            key,
            theta,
            loglike,
            logprior,
            jnp.zeros(theta.shape[0], dtype=jnp.float64),
        )
        final, _ = jax.lax.scan(body, init, xs=jnp.arange(int(args.smc_move_steps), dtype=jnp.int32))
        rng_key, moved_theta, moved_loglike, moved_logprior, accepted_total = final
        acceptance = accepted_total / max(1, int(args.smc_move_steps))
        return moved_theta, moved_loglike, moved_logprior, acceptance

    @jax.jit
    def _run_smc_device(initial_key: jax.Array, initial_particles: jnp.ndarray):
        initial_loglike = batched_loglike(initial_particles)
        initial_logprior = _smc_prior_logprob_matrix(initial_particles, parameter_specs)
        n_particles = initial_particles.shape[0]
        temperatures = jnp.zeros(max_tempering_steps + 1, dtype=jnp.float64).at[0].set(0.0)
        ess_history = jnp.zeros(max_tempering_steps + 1, dtype=jnp.float64).at[0].set(float(n_particles))
        move_acceptance = jnp.zeros(max_tempering_steps, dtype=jnp.float64)
        final_weights = jnp.full((n_particles,), 1.0 / float(n_particles), dtype=jnp.float64)

        init = (
            initial_key,
            initial_particles,
            initial_loglike,
            initial_logprior,
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            temperatures,
            ess_history,
            move_acceptance,
            final_weights,
        )

        def cond(carry):
            _key, _particles, _loglike, _logprior, beta, step, _move_count, _temps, _ess, _move_acc, _weights = carry
            return jnp.logical_and(beta < (1.0 - 1.0e-12), step < max_tempering_steps)

        def body(carry):
            key, current_particles, current_loglike, current_logprior, beta, step, move_count, temps, ess_hist, move_acc, _weights = carry
            next_beta, ess, logw_norm = _smc_find_next_beta_jax(beta, current_loglike, float(args.smc_ess_threshold))
            weights = jnp.exp(logw_norm)
            temps = temps.at[step + 1].set(next_beta)
            ess_hist = ess_hist.at[step + 1].set(ess)
            is_final = next_beta >= (1.0 - 1.0e-12)
            key, resample_key, move_key = jax.random.split(key, 3)

            def _final_state(_):
                return current_particles, current_loglike, current_logprior, move_acc, move_count, weights

            def _moved_state(_):
                indices = jax.random.categorical(resample_key, logw_norm, shape=(current_particles.shape[0],))
                resampled_particles = current_particles[indices]
                resampled_loglike = current_loglike[indices]
                resampled_logprior = current_logprior[indices]
                moved_particles, moved_loglike, moved_logprior, acceptance = _move_particles(
                    move_key,
                    resampled_particles,
                    resampled_loglike,
                    resampled_logprior,
                    next_beta,
                )
                updated_move_acc = move_acc.at[move_count].set(jnp.mean(acceptance))
                return moved_particles, moved_loglike, moved_logprior, updated_move_acc, move_count + 1, weights

            (
                next_particles,
                next_loglike,
                next_logprior,
                move_acc,
                move_count,
                weights,
            ) = jax.lax.cond(is_final, _final_state, _moved_state, operand=None)
            return key, next_particles, next_loglike, next_logprior, next_beta, step + 1, move_count, temps, ess_hist, move_acc, weights

        return jax.lax.while_loop(cond, body, init)

    smc_start = time.time()
    smc_key = jax.random.PRNGKey(606 if args.seed is None else int(args.seed) + 606)
    (
        _key,
        particles,
        loglike,
        logprior,
        beta,
        step_count,
        move_count,
        temperatures,
        ess_history,
        move_acceptance,
        final_weights,
    ) = _run_smc_device(smc_key, particles)

    evaluator.timing_totals["smc_runtime"] = evaluator.timing_totals.get("smc_runtime", 0.0) + (time.time() - smc_start)
    final_beta = float(np.asarray(beta))
    if final_beta < 1.0 - 1.0e-12:
        raise RuntimeError(
            f"SMC tempering did not reach beta=1 within {max_tempering_steps} steps; final beta={final_beta:.6f}."
        )
    step_count_int = int(np.asarray(step_count))
    move_count_int = int(np.asarray(move_count))
    particles_np = np.asarray(particles, dtype=float)
    log_prob = np.asarray(loglike + logprior, dtype=float)
    temperature_schedule = np.asarray(temperatures[: step_count_int + 1], dtype=float)
    ess_history_np = np.asarray(ess_history[: step_count_int + 1], dtype=float)
    move_acceptance_np = np.asarray(move_acceptance[:move_count_int], dtype=float)
    final_weights_np = np.asarray(final_weights, dtype=float)
    init_diagnostics = dict(init_diag)
    init_diagnostics.update(
        {
            "strategy_requested": "blackjax_smc",
            "strategy_used": "blackjax_smc",
            "smc_backend": _resolve_sampler_backend(),
            "smc_particles": int(args.smc_particles),
            "smc_seed_mode": str(args.smc_seed_mode),
            "smc_tempering_steps": int(max(0, step_count_int)),
            "smc_final_ess": float(_effective_sample_size(final_weights_np)),
            "smc_weighted_logz_estimate": float("nan"),
            "gpu_visible_devices": [str(device) for device in jax.devices()],
            "gpu_enabled": any(str(device.platform).lower() == "gpu" for device in jax.devices()),
        }
    )
    return SMCRunResult(
        particles=particles_np,
        log_prob=log_prob,
        sample_weights=final_weights_np,
        temperature_schedule=temperature_schedule,
        ess_history=ess_history_np,
        move_acceptance_history=move_acceptance_np,
        init_diagnostics=init_diagnostics,
    )


def _best_weighted_index(log_prob: np.ndarray, sample_weights: np.ndarray | None = None) -> int:
    log_prob_array = np.asarray(log_prob, dtype=float).reshape(-1)
    if log_prob_array.size == 0:
        return 0
    if sample_weights is None:
        return int(np.nanargmax(log_prob_array))
    weights = _normalized_weights(sample_weights, len(log_prob_array))
    score = log_prob_array + np.log(np.maximum(weights, np.nextafter(0.0, 1.0)))
    return int(np.nanargmax(score))


def _build_smc_refine_initialization(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
    particles: np.ndarray,
    log_prob: np.ndarray,
    sample_weights: np.ndarray | None,
) -> NUTSInitialization:
    particle_array = np.asarray(particles, dtype=float)
    log_prob_array = np.asarray(log_prob, dtype=float).reshape(-1)
    if particle_array.ndim != 2 or particle_array.shape[0] == 0:
        raise ValueError("SMC refinement requires at least one particle.")
    weights = _normalized_weights(sample_weights, particle_array.shape[0])
    ranking = np.argsort(-(log_prob_array + np.log(np.maximum(weights, np.nextafter(0.0, 1.0)))))
    selected: list[np.ndarray] = []
    labels: list[str] = []
    rng = np.random.default_rng(None if args.seed is None else int(args.seed) + 707)
    for idx in ranking:
        candidate = particle_array[int(idx)]
        if any(
            _scaled_candidate_distance(candidate, existing, parameter_specs) < float(args.nuts_init_dedup_distance)
            for existing in selected
        ):
            continue
        selected.append(
            _jitter_theta_in_support(
                candidate,
                parameter_specs,
                float(args.nuts_init_jitter_frac),
                rng,
                boundary_frac=float(args.nuts_init_boundary_frac),
            )
        )
        labels.append(f"smc_particle_{int(idx)}")
        if len(selected) >= int(args.chains):
            break
    if not selected:
        selected.append(_clip_theta_to_support(particle_array[int(ranking[0])], parameter_specs, boundary_frac=float(args.nuts_init_boundary_frac)))
        labels.append(f"smc_particle_{int(ranking[0])}")
    while len(selected) < int(args.chains):
        base = np.asarray(selected[len(selected) % len(selected)], dtype=float)
        selected.append(
            _jitter_theta_in_support(
                base,
                parameter_specs,
                float(args.nuts_init_jitter_frac),
                rng,
                boundary_frac=float(args.nuts_init_boundary_frac),
            )
        )
        labels.append(labels[len(labels) % max(1, len(labels))] + "_jitter")
    chain_seeds = [ChainSeed(values=np.asarray(theta, dtype=float), source_label=label) for theta, label in zip(selected, labels)]
    diagnostics = {
        "strategy_requested": "smc_refine_nuts",
        "strategy_used": "smc_refine_nuts",
        "svi_used": False,
        "chain_seed_labels": labels,
        "distinct_chain_seeds": int(len({tuple(np.round(seed.values, 8)) for seed in chain_seeds})),
        "chosen_ranked_maps": [
            {
                "label": label,
                "logprob": float(log_prob_array[int(rank_idx)]),
                "theta": np.asarray(particle_array[int(rank_idx)], dtype=float),
            }
            for label, rank_idx in zip(labels[: len(selected)], ranking[: len(selected)])
        ],
        "smc_preconditioning": {
            "particle_count": int(particle_array.shape[0]),
            "weighted_ess": float(_effective_sample_size(weights)),
        },
    }
    init_model = _sample_site_model(parameter_specs)
    return NUTSInitialization(
        init_params=_seed_values_to_init_params(parameter_specs, chain_seeds, model_for_init=init_model),
        chain_seeds=chain_seeds,
        diagnostics=diagnostics,
        reference_theta=np.asarray(selected[0], dtype=float),
    )


def _run_numpyro_nuts_sampler(
    args: argparse.Namespace,
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    sample_model,
    nuts_init: NUTSInitialization,
    map_history: list[dict[str, Any]],
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
        map_history=map_history,
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


def _run_smc_sampler(
    args: argparse.Namespace,
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    ranked_results: list[MAPRunResult],
    map_history: list[dict[str, Any]],
) -> tuple[np.ndarray, PosteriorResults]:
    if not any(str(device.platform).lower() == "gpu" for device in jax.devices()):
        _log(args, f"[smc] GPU not visible to JAX; running on {', '.join(str(device) for device in jax.devices())}")
    _log(
        args,
        (
            f"[smc] preparing sampler particles={args.smc_particles} ess_threshold={args.smc_ess_threshold:.2f} "
            f"move_steps={args.smc_move_steps} move_scale={args.smc_move_scale:.3f} seed_mode={args.smc_seed_mode}"
        ),
    )
    smc_result = _run_blackjax_smc(args, state.parameter_specs, evaluator, ranked_results)
    best_index = _best_weighted_index(smc_result.log_prob, smc_result.sample_weights)
    best_fit = np.asarray(smc_result.particles[best_index], dtype=float)
    posterior = PosteriorResults(
        samples=np.asarray(smc_result.particles, dtype=float),
        log_prob=np.asarray(smc_result.log_prob, dtype=float),
        accept_prob=np.asarray(smc_result.move_acceptance_history, dtype=float),
        diverging=np.zeros_like(np.asarray(smc_result.move_acceptance_history, dtype=float), dtype=bool),
        num_steps=np.asarray(smc_result.ess_history, dtype=float),
        map_history=map_history,
        warmup_steps=0,
        sample_steps=int(args.smc_particles),
        num_chains=1,
        init_diagnostics=smc_result.init_diagnostics,
        grouped_samples=None,
        grouped_log_prob=None,
        sampler="blackjax_smc",
        sample_weights=np.asarray(smc_result.sample_weights, dtype=float),
        temperature_schedule=np.asarray(smc_result.temperature_schedule, dtype=float),
        ess_history=np.asarray(smc_result.ess_history, dtype=float),
        move_acceptance_history=np.asarray(smc_result.move_acceptance_history, dtype=float),
    )
    posterior.init_diagnostics = dict(posterior.init_diagnostics or {})
    posterior.init_diagnostics.update(
        {
            "direct_evaluator_startup": True,
            "map_stage_skipped": True,
            "post_smc_nuts_refine_requested": bool(getattr(args, "refine_with_nuts", False)),
            "post_smc_nuts_refine_used": False,
        }
    )
    _log(
        args,
        (
            f"[smc] complete steps={max(0, len(smc_result.temperature_schedule) - 1)} "
            f"particles={len(smc_result.particles)} final_ess={_effective_sample_size(smc_result.sample_weights):.1f}"
        ),
    )
    return best_fit, posterior


def _state_with_sigma_scale(state: BuildState, sigma_scale: float) -> BuildState:
    if sigma_scale <= 0:
        raise ValueError("sigma_scale must be positive.")
    scaled_families = [
        replace(family, sigma_arcsec=float(state.sigma_arcsec * sigma_scale))
        for family in state.family_data
    ]
    scaled_bins = _build_bin_data(scaled_families)
    return replace(
        state,
        sigma_arcsec=float(state.sigma_arcsec * sigma_scale),
        family_data=scaled_families,
        bin_data=scaled_bins,
        geometry_cache=state.geometry_cache,
    )


def _build_continuation_passes(args: argparse.Namespace) -> list[ContinuationPass]:
    requested_top_k = max(0, int(args.validate_top_k_families))
    initial_top_k = 0 if requested_top_k == 0 else min(int(args.continuation_validation_top_k), requested_top_k)
    return [
        ContinuationPass(
            name="pass1_relaxed",
            sigma_scale=float(args.continuation_sigma_scale),
            validate_top_k_families=initial_top_k,
            validation_approx="adaptive",
            sampling_engine="refreshing_surrogate",
        ),
        ContinuationPass(
            name="pass2_tightened",
            sigma_scale=max(1.0, 0.5 * (1.0 + float(args.continuation_sigma_scale))),
            validate_top_k_families=0
            if requested_top_k == 0
            else max(initial_top_k, min(requested_top_k, initial_top_k * 2)),
            validation_approx="adaptive",
            sampling_engine=str(args.sampling_engine),
        ),
        ContinuationPass(
            name="pass3_final",
            sigma_scale=1.0,
            validate_top_k_families=requested_top_k,
            validation_approx=str(args.validation_approx),
            sampling_engine=str(args.sampling_engine),
        ),
    ]


def _build_broad_starts(
    parameter_specs: list[ParameterSpec],
    num_broad_seeds: int,
    seed: int | None,
    boundary_frac: float = DEFAULT_NUTS_INIT_BOUNDARY_FRAC,
    carry_forward: np.ndarray | None = None,
) -> list[SeededStart]:
    starts: list[SeededStart] = [SeededStart(values=_default_theta(parameter_specs), label="midpoint")]
    if carry_forward is not None:
        starts.append(
            SeededStart(
                values=_clip_theta_to_support(carry_forward, parameter_specs, boundary_frac=boundary_frac),
                label="carry_forward",
            )
        )
    if num_broad_seeds <= 0:
        return starts
    engine = qmc.Halton(d=len(parameter_specs), scramble=True, seed=seed)
    samples = engine.random(n=num_broad_seeds)
    for index, unit_draw in enumerate(np.asarray(samples, dtype=float)):
        values = np.empty(len(parameter_specs), dtype=float)
        for dim, spec in enumerate(parameter_specs):
            if spec.prior_kind == "normal":
                quantile = float(np.clip(unit_draw[dim], 1.0e-4, 1.0 - 1.0e-4))
                base = float(spec.mean) + float(norm.ppf(quantile)) * _parameter_scale(spec)
                values[dim] = base
            else:
                values[dim] = float(spec.lower) + float(unit_draw[dim]) * float(spec.upper - spec.lower)
        starts.append(
            SeededStart(
                values=_clip_theta_to_support(values, parameter_specs, boundary_frac=boundary_frac),
                label=f"broad_{index + 1}",
            )
        )
    return starts


def _build_local_refinement_starts(
    parameter_specs: list[ParameterSpec],
    ranked_results: list[MAPRunResult],
    num_local_seeds: int,
    jitter_scale: float,
    seed: int | None,
    boundary_frac: float = DEFAULT_NUTS_INIT_BOUNDARY_FRAC,
) -> list[SeededStart]:
    if num_local_seeds <= 0 or not ranked_results:
        return []
    rng = np.random.default_rng(seed)
    top_count = min(3, len(ranked_results))
    top_results = ranked_results[:top_count]
    starts: list[SeededStart] = [
        SeededStart(
            values=_clip_theta_to_support(top_results[0].theta.copy(), parameter_specs, boundary_frac=boundary_frac),
            label="best_broad",
        )
    ]
    scales = np.asarray([_parameter_scale(spec) for spec in parameter_specs], dtype=float)
    for index in range(num_local_seeds):
        anchor = top_results[index % top_count]
        jitter = rng.normal(0.0, float(jitter_scale), size=len(parameter_specs)) * scales
        values = _clip_theta_to_support(anchor.theta + jitter, parameter_specs, boundary_frac=boundary_frac)
        starts.append(SeededStart(values=values, label=f"local_{index + 1}_from_{anchor.label}"))
    return starts


def _stable_seed_offset(name: str) -> int:
    return sum(ord(char) for char in name) % 1000


def _fail(message: str) -> None:
    raise SystemExit(message)


def _process_memory_snapshot() -> dict[str, float | None]:
    rss_mb: float | None = None
    vms_mb: float | None = None
    status_path = Path("/proc/self/status")
    if status_path.exists():
        try:
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss_mb = float(line.split()[1]) / 1024.0
                elif line.startswith("VmSize:"):
                    vms_mb = float(line.split()[1]) / 1024.0
        except OSError:
            pass
    ru_maxrss_mb: float | None = None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        ru_maxrss_mb = float(usage.ru_maxrss) / 1024.0
    except Exception:  # pragma: no cover
        ru_maxrss_mb = None
    return {
        "rss_mb": rss_mb,
        "vms_mb": vms_mb,
        "ru_maxrss_mb": ru_maxrss_mb,
    }


def _format_memory_snapshot() -> str:
    snapshot = _process_memory_snapshot()
    parts = []
    for key in ("rss_mb", "vms_mb", "ru_maxrss_mb"):
        value = snapshot.get(key)
        if value is None or not np.isfinite(value):
            parts.append(f"{key}=na")
        else:
            parts.append(f"{key}={value:.1f}")
    return " ".join(parts)


def _close_debug_log() -> None:
    global _DEBUG_LOG_HANDLE
    if _DEBUG_LOG_HANDLE is not None:
        try:
            _DEBUG_LOG_HANDLE.flush()
            _DEBUG_LOG_HANDLE.close()
        finally:
            _DEBUG_LOG_HANDLE = None


def _debug_log_line(line: str) -> None:
    global _DEBUG_LOG_HANDLE
    if _DEBUG_LOG_HANDLE is None or _DEBUG_LOG_HANDLE.closed:
        if _DEBUG_LOG_PATH is None:
            return
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DEBUG_LOG_HANDLE = _DEBUG_LOG_PATH.open("a", encoding="utf-8")
    _DEBUG_LOG_HANDLE.write(f"{line}\n")
    _DEBUG_LOG_HANDLE.flush()


def _set_debug_log_path(path: Path) -> None:
    global _DEBUG_LOG_PATH
    path = path.resolve()
    if _DEBUG_LOG_PATH == path and _DEBUG_LOG_HANDLE is not None:
        return
    previous_path = _DEBUG_LOG_PATH
    previous_handle = _DEBUG_LOG_HANDLE
    _DEBUG_LOG_PATH = path
    if previous_handle is not None:
        try:
            previous_handle.flush()
        except Exception:  # pragma: no cover
            pass
    if previous_path is not None and previous_path != path and previous_path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(previous_path.read_text(encoding="utf-8"), encoding="utf-8")
    _close_debug_log()
    _debug_log_line(f"{datetime.now().isoformat(timespec='seconds')} [debug-log] path={path}")


def _configure_debug_log(args: argparse.Namespace, run_name: str, run_dir: Path | None = None) -> Path:
    if run_dir is None:
        target = Path(args.output_dir) / f"{run_name}.debug.log"
    else:
        target = run_dir / "run_debug.log"
    _set_debug_log_path(target)
    return target


def _log_exception(context: str, exc: BaseException) -> None:
    _debug_log_line(
        f"{datetime.now().isoformat(timespec='seconds')} [exception] context={context} "
        f"type={type(exc).__name__} {str(exc)} {_format_memory_snapshot()}"
    )
    for line in traceback.format_exc().rstrip().splitlines():
        _debug_log_line(f"{datetime.now().isoformat(timespec='seconds')} [traceback] {line}")


def _run_logged_phase(args: argparse.Namespace | None, phase: str, fn, *, detail: str | None = None):
    start = time.time()
    suffix = f" {detail}" if detail else ""
    _log(args, f"[phase] {phase} start{suffix}")
    try:
        result = fn()
    except Exception as exc:
        _log(args, f"[phase] {phase} error elapsed={_fmt_seconds(time.time() - start)} {type(exc).__name__}: {exc}")
        raise
    _log(args, f"[phase] {phase} end elapsed={_fmt_seconds(time.time() - start)}")
    return result


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
    start_component_index: int,
    kpc_per_arcsec: float = 1.0,
) -> tuple[list[dict[str, Any]], list[list[tuple[str, int]]], list[dict[str, Any]], list[dict[str, Any]]]:
    _, ra0_deg, dec0_deg = reference
    components: list[dict[str, Any]] = []
    assignments: list[list[tuple[str, int]]] = []
    scaling_component_assignments: list[dict[str, Any]] = []
    scaling_component_records: list[dict[str, Any]] = []
    for potfile_order, (potfile, param_index_lookup) in enumerate(zip(potfiles, scaling_param_indices)):
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


def _logsumexp_np(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    max_val = float(np.max(finite))
    return max_val + float(np.log(np.sum(np.exp(finite - max_val))))


def _effective_sample_size(weights: np.ndarray) -> float:
    normalized = _normalized_weights(weights, len(weights))
    if normalized.size == 0:
        return 0.0
    return float(1.0 / np.sum(np.square(normalized)))


def _resolve_sampler_backend() -> str:
    try:  # pragma: no cover - exercised only when blackjax is installed.
        import blackjax as _blackjax  # noqa: F401

        return "blackjax"
    except ImportError:
        return "internal_jax_fallback"


def _make_run_name(par_path: str | Path) -> str:
    stem = Path(par_path).stem
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{timestamp}"


def _plot_path(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def _should_log(args: argparse.Namespace | None) -> bool:
    return args is None or not getattr(args, "quiet", False)


def _log(args: argparse.Namespace | None, message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message} {_format_memory_snapshot()}"
    if _should_log(args):
        print(line, flush=True)
    _debug_log_line(line)


def _fmt_seconds(value: float) -> str:
    return f"{value:.2f}s"


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


class BoundedMAPOptimizer:
    def __init__(
        self,
        parameter_specs: list[ParameterSpec],
        logprob_fn,
        maxiter: int,
    ):
        self.parameter_specs = parameter_specs
        self.logprob_fn = logprob_fn
        self.maxiter = int(maxiter)
        self.opt = optax.lbfgs()
        self.value_and_grad_fun = optax.value_and_grad_from_state(self._loss)

    def _numpyro_model(self):
        for spec in self.parameter_specs:
            numpyro.sample(spec.sample_name, self._distribution(spec))

    def _distribution(self, spec: ParameterSpec):
        if spec.prior_kind == "normal":
            return dist.Normal(float(spec.mean), float(spec.std))
        return dist.Uniform(float(spec.lower), float(spec.upper))

    def _vector_to_params_dict(self, theta: np.ndarray | jnp.ndarray) -> dict[str, jnp.ndarray]:
        theta_array = jnp.asarray(theta, dtype=jnp.float64)
        return {spec.sample_name: theta_array[idx] for idx, spec in enumerate(self.parameter_specs)}

    def _params_dict_to_vector(self, params_dict: dict[str, Any]) -> jnp.ndarray:
        return jnp.stack([jnp.asarray(params_dict[spec.sample_name], dtype=jnp.float64) for spec in self.parameter_specs])

    def prior_log_prob(self, theta: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        theta_array = jnp.asarray(theta, dtype=jnp.float64)
        total = jnp.array(0.0, dtype=jnp.float64)
        for idx, spec in enumerate(self.parameter_specs):
            total = total + self._distribution(spec).log_prob(theta_array[idx])
        return total

    def to_unconstrained(self, theta: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        return unconstrain_fn(
            self._numpyro_model,
            model_args=(),
            model_kwargs={},
            params=self._vector_to_params_dict(theta),
        )

    def to_constrained(self, theta_unconstrained: dict[str, jnp.ndarray]) -> jnp.ndarray:
        constrained = constrain_fn(
            self._numpyro_model,
            model_args=(),
            model_kwargs={},
            params=theta_unconstrained,
        )
        return self._params_dict_to_vector(constrained)

    def _loss(self, theta_unconstrained: dict[str, jnp.ndarray]) -> jnp.ndarray:
        theta = self.to_constrained(theta_unconstrained)
        return -(self.logprob_fn(theta) + self.prior_log_prob(theta))

    @partial(jax.jit, static_argnums=0)
    def run_single(self, init_theta_unconstrained: dict[str, jnp.ndarray], tol: float):
        def step(carry):
            theta_u, state, tol_hit = carry
            value, grad = self.value_and_grad_fun(theta_u, state=state)
            updates, state = self.opt.update(
                grad,
                state,
                theta_u,
                value=value,
                grad=grad,
                value_fn=self._loss,
            )
            theta_u = optax.apply_updates(theta_u, updates)
            new_value = optax.tree.get(state, "value")
            diff = jnp.abs(value - new_value)
            tol_hit = jnp.where(diff < tol, tol_hit + 1, 0)
            return theta_u, state, tol_hit

        def cond(carry):
            _, state, tol_hit = carry
            iter_num = optax.tree.get(state, "count")
            return (iter_num < self.maxiter) & (tol_hit < 3)

        init_state = self.opt.init(init_theta_unconstrained)
        final_theta_u, final_state, _ = jax.lax.while_loop(cond, step, (init_theta_unconstrained, init_state, 0))
        return final_theta_u, optax.tree.get(final_state, "count")

    @partial(jax.jit, static_argnums=0)
    def run_batch(self, init_theta_unconstrained_batch: dict[str, jnp.ndarray], tol: float):
        final_theta_u_batch, counts = jax.vmap(
            lambda theta_u: self.run_single(theta_u, tol),
            in_axes=0,
        )(init_theta_unconstrained_batch)
        final_theta_batch = jax.vmap(self.to_constrained, in_axes=0)(final_theta_u_batch)
        final_logprob_batch = jax.vmap(
            lambda theta: self.logprob_fn(theta) + self.prior_log_prob(theta),
            in_axes=0,
        )(final_theta_batch)
        return final_theta_batch, final_logprob_batch, counts

    def run(
        self,
        starts: list[SeededStart],
        stage_label: str,
        log_fn=None,
    ) -> tuple[np.ndarray, list[dict[str, Any]], list[MAPRunResult]]:
        if not starts:
            raise ValueError("Explicit MAP starts are required.")
        history: list[dict[str, Any]] = []
        results: list[MAPRunResult] = []
        best_theta: np.ndarray | None = None
        best_logprob = -np.inf
        best_restart = -1
        theta0_matrix = _clip_theta_matrix_to_support(
            np.asarray([start.values for start in starts], dtype=float),
            self.parameter_specs,
        )
        for idx, start in enumerate(starts):
            if log_fn is not None:
                log_fn(f"[map:{stage_label}] restart {idx + 1}/{len(starts)} seed={start.label} starting")
            restart_start = time.time()
            theta0 = theta0_matrix[idx]
            theta0_u = self.to_unconstrained(theta0)
            final_theta_u, count = self.run_single(theta0_u, 1.0e-6)
            final_theta = np.asarray(self.to_constrained(final_theta_u), dtype=float)
            final_logprob = float(self.logprob_fn(final_theta) + self.prior_log_prob(final_theta))
            restart_elapsed = time.time() - restart_start
            is_best_so_far = final_logprob > best_logprob
            results.append(
                MAPRunResult(
                    theta=final_theta,
                    logprob=final_logprob,
                    iterations=int(np.asarray(count)),
                    label=start.label,
                )
            )
            history.append(
                {
                    "restart": idx,
                    "stage": stage_label,
                    "start_label": start.label,
                    "iterations": int(np.asarray(count)),
                    "best_loglike": final_logprob,
                    "elapsed_sec": float(restart_elapsed),
                    "is_best_so_far": bool(is_best_so_far),
                }
            )
            if is_best_so_far:
                best_logprob = final_logprob
                best_theta = final_theta
                best_restart = idx
            if log_fn is not None:
                log_fn(
                    f"[map:{stage_label}] restart "
                    f"{idx + 1}/{len(starts)} complete in {_fmt_seconds(restart_elapsed)} "
                    f"iterations={int(np.asarray(count))} loglike={final_logprob:.3f} "
                    f"best_so_far={'yes' if is_best_so_far else 'no'}"
                )
        if best_theta is None:
            raise RuntimeError("MAP optimization failed.")
        if log_fn is not None:
            log_fn(
                f"[map:{stage_label}] restarts finished best_restart={best_restart + 1}/{len(starts)} "
                f"best_loglike={best_logprob:.3f}"
            )
        results.sort(key=lambda item: item.logprob, reverse=True)
        return best_theta, history, results


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
            "map_runtime": 0.0,
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
            sigma = jnp.asarray(bin_data.sigma_per_image, dtype=jnp.float64)
            weights = 1.0 / jnp.square(sigma)
            n_families = len(bin_data.family_ids)
            sum_w = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(weights)
            sum_bx = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(weights * beta_x)
            sum_by = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(weights * beta_y)
            centroid_x = sum_bx / jnp.maximum(sum_w, 1.0e-18)
            centroid_y = sum_by / jnp.maximum(sum_w, 1.0e-18)
            dx = beta_x - centroid_x[family_idx]
            dy = beta_y - centroid_y[family_idx]
            sigma2 = jnp.square(sigma)
            bin_loglike = -0.5 * jnp.sum(
                (dx**2 + dy**2) / sigma2 + 2.0 * jnp.log(2.0 * jnp.pi * sigma2)
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
                weights = np.full(np.sum(mask), 1.0 / (family.sigma_arcsec**2), dtype=float)
                source_x = float(np.average(beta_x[mask], weights=weights))
                source_y = float(np.average(beta_y[mask], weights=weights))
                dx = beta_x[mask] - source_x
                dy = beta_y[mask] - source_y
                residuals = np.sqrt(dx**2 + dy**2)
                summaries[family_id] = {
                    "source_x": source_x,
                    "source_y": source_y,
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
        packed_lens_spec = PackedLensSpec(
            **{
                field_name: np.asarray(packed_group[field_name])
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
                + f"map={float(row.map):.6g}"
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
        ax.axvline(float(record["map"]), color="tab:red", linewidth=1.5, linestyle="--", label="MAP")
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
        "sampling_engine": args.sampling_engine,
        "validation_approx": args.validation_approx,
        "active_scaling_galaxies": list(evaluator.active_scaling_galaxies_by_potfile),
        "active_scaling_components": int(len(evaluator.active_scaling_component_indices)),
        "inactive_scaling_components": int(len(evaluator.inactive_scaling_component_indices)),
        "requested_active_scaling_by_potfile": evaluator.requested_active_scaling_by_potfile,
        "actual_active_scaling_by_potfile": evaluator.actual_active_scaling_by_potfile,
        "total_scaling_by_potfile": evaluator.total_scaling_by_potfile,
        "optimizer": "optax_lbfgs",
        "sampler": str(results.sampler),
        "nuts_init_strategy_requested": init_diagnostics.get("strategy_requested", getattr(args, "nuts_init_strategy", "ranked_map")),
        "nuts_init_strategy_used": init_diagnostics.get("strategy_used", getattr(args, "nuts_init_strategy", "ranked_map")),
        "nuts_init_settings": {
            "top_k": int(init_diagnostics.get("resolved_top_k", _resolved_nuts_init_top_k(args))),
            "boundary_frac": float(getattr(args, "nuts_init_boundary_frac", DEFAULT_NUTS_INIT_BOUNDARY_FRAC)),
            "jitter_frac": float(getattr(args, "nuts_init_jitter_frac", DEFAULT_NUTS_INIT_JITTER_FRAC)),
            "dedup_distance": float(getattr(args, "nuts_init_dedup_distance", DEFAULT_NUTS_INIT_DEDUP_DISTANCE)),
            "svi_steps": int(getattr(args, "svi_steps", DEFAULT_SVI_STEPS)),
            "svi_learning_rate": float(getattr(args, "svi_learning_rate", DEFAULT_SVI_LEARNING_RATE)),
        },
        "nuts_init_diagnostics": {
            "ranked_candidates_considered": int(init_diagnostics.get("ranked_candidates_considered", 0)),
            "near_boundary_rejected": int(init_diagnostics.get("near_boundary_rejected", 0)),
            "duplicate_rejected": int(init_diagnostics.get("duplicate_rejected", 0)),
            "distinct_ranked_map_candidates": int(init_diagnostics.get("distinct_ranked_map_candidates", 0)),
            "distinct_chain_seeds": int(init_diagnostics.get("distinct_chain_seeds", 0)),
            "chain_seed_diversity": int(init_diagnostics.get("chain_seed_diversity", 0)),
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
        "map_broad_seeds": args.map_broad_seeds,
        "map_local_refine_seeds": args.map_local_refine_seeds,
        "map_local_jitter_scale": args.map_local_jitter_scale,
        "continuation_sigma_scale": args.continuation_sigma_scale,
        "continuation_validation_top_k": args.continuation_validation_top_k,
        "runtime_sec": runtime_sec,
        "best_loglike": best_loglike,
        "seed": args.seed,
        "packed_fast_path": True,
        "uses_potfile_scaling": bool(state.potfiles and state.fit_mode == "small-only"),
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


def _chosen_ranked_map_matrix(
    parameter_specs: list[ParameterSpec],
    init_diagnostics: dict[str, Any] | None,
) -> np.ndarray | None:
    if not parameter_specs or not init_diagnostics:
        return None
    chosen_maps = list(init_diagnostics.get("chosen_ranked_maps", []))
    rows: list[np.ndarray] = []
    for item in chosen_maps:
        theta = item.get("theta")
        if theta is None:
            continue
        theta_array = np.asarray(theta, dtype=float)
        if theta_array.shape != (len(parameter_specs),):
            continue
        if not np.all(np.isfinite(theta_array)):
            continue
        rows.append(theta_array)
    if not rows:
        return None
    return np.asarray(rows, dtype=float)


def _subset_parameter_matrix(
    parameter_specs: list[ParameterSpec],
    values: np.ndarray | None,
    component_family: str,
) -> tuple[list[ParameterSpec], np.ndarray | None]:
    indices = [idx for idx, spec in enumerate(parameter_specs) if spec.component_family == component_family]
    subset_specs = [parameter_specs[idx] for idx in indices]
    if values is None or not indices:
        return subset_specs, None
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(parameter_specs):
        return subset_specs, None
    return subset_specs, np.asarray(array[:, indices], dtype=float)


def _overlay_corner_points(fig: plt.Figure, points: np.ndarray | None) -> None:
    if points is None:
        return
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[0] == 0 or point_array.shape[1] == 0:
        return
    finite_points = point_array[np.isfinite(point_array).all(axis=1)]
    if finite_points.shape[0] == 0:
        return
    if hasattr(corner, "overplot_points"):
        corner.overplot_points(fig, finite_points, marker="o", color="tab:orange", ms=4, alpha=0.85)
        return
    ndim = finite_points.shape[1]
    axes = np.asarray(fig.axes, dtype=object).reshape((ndim, ndim))
    for row in range(1, ndim):
        for col in range(row):
            ax = axes[row, col]
            ax.scatter(
                finite_points[:, col],
                finite_points[:, row],
                color="tab:orange",
                s=18,
                alpha=0.85,
                zorder=5,
            )


def _plot_corner(
    plot_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    chosen_map_points: np.ndarray | None = None,
) -> None:
    if corner is None or not parameter_specs:
        return
    finite_samples = _finite_sample_rows(samples)
    if finite_samples.shape[0] == 0:
        return
    _log(
        None,
        f"[plot:corner] path={_plot_path(plot_dir, 'corner.png')} ndim={len(parameter_specs)} samples_shape={tuple(finite_samples.shape)}",
    )
    labels = [spec.name for spec in parameter_specs]
    fig = corner.corner(finite_samples, labels=labels, **CORNER_PLOT_KWARGS)
    _overlay_corner_points(fig, chosen_map_points)
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
    chosen_map_points: np.ndarray | None = None,
) -> None:
    if corner is None or samples.size == 0 or not parameter_specs:
        return
    finite_samples = _finite_sample_rows(samples)
    if finite_samples.shape[0] == 0:
        return
    _log(
        None,
        f"[plot:corner] path={_plot_path(plot_dir, 'potfile_corner.png')} ndim={len(parameter_specs)} samples_shape={tuple(finite_samples.shape)}",
    )
    labels = [spec.name for spec in parameter_specs]
    fig = corner.corner(finite_samples, labels=labels, **CORNER_PLOT_KWARGS)
    _overlay_corner_points(fig, chosen_map_points)
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
        ax.axvline(float(best_fit[idx]), color="tab:red", linewidth=1.5, linestyle="--", label="MAP")
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
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=False)
    restart_index = np.arange(len(results.map_history))
    best_vals = [entry["best_loglike"] for entry in results.map_history]
    iter_vals = [entry["iterations"] for entry in results.map_history]
    axes[0].plot(restart_index, best_vals, marker="o")
    axes[0].set_xlabel("Restart")
    axes[0].set_ylabel("Best logL")
    axes[0].set_title("MAP Restart Diagnostics")
    axes[1].bar(restart_index, iter_vals)
    axes[1].set_xlabel("Restart")
    axes[1].set_ylabel("Iterations")
    axes[1].set_title("L-BFGS Iterations")
    if results.sampler == "blackjax_smc":
        if results.temperature_schedule is not None and len(results.temperature_schedule) > 0:
            axes[2].plot(results.temperature_schedule, marker="o", color="tab:green")
        axes[2].set_xlabel("SMC tempering step")
        axes[2].set_ylabel("temperature")
        axes[2].set_title("SMC Temperature Schedule")
    else:
        axes[2].plot(results.accept_prob.ravel(), color="tab:green")
        axes[2].set_xlabel("Posterior draw")
        axes[2].set_ylabel("Accept prob")
        axes[2].set_title("NUTS Acceptance Probability")
    axes[3].axis("off")
    init_diag = results.init_diagnostics or {}
    summary_lines = [
        f"Init requested: {init_diag.get('strategy_requested', 'unknown')}",
        f"Init used: {init_diag.get('strategy_used', 'unknown')}",
        (
            "Candidates: "
            f"considered={int(init_diag.get('ranked_candidates_considered', 0))} "
            f"boundary={int(init_diag.get('near_boundary_rejected', 0))} "
            f"duplicates={int(init_diag.get('duplicate_rejected', 0))}"
        ),
        (
            "Seeds: "
            f"distinct_ranked={int(init_diag.get('distinct_ranked_map_candidates', 0))} "
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
    axes[3].text(0.01, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=9, family="monospace")
    axes[3].set_title("Sampler Initialization")
    fig.tight_layout()
    fig.savefig(_plot_path(plot_dir, "run_diagnostics.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_weights_logl(plot_dir: Path, results: PosteriorResults) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if results.sampler == "blackjax_smc":
        axes[0].scatter(np.arange(len(results.log_prob)), results.log_prob, c=_normalized_weights(results.sample_weights, len(results.log_prob)), s=18, cmap="viridis")
        axes[0].set_ylabel("log posterior")
        axes[0].set_title("SMC Particle Log Probability")
        if results.ess_history is not None and len(results.ess_history) > 0:
            axes[1].plot(results.ess_history, marker="o", color="tab:blue")
        axes[1].set_ylabel("ESS")
        axes[1].set_xlabel("SMC tempering step")
        axes[1].set_title("SMC Effective Sample Size")
    else:
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
    chosen_map_points: np.ndarray | None = None,
) -> None:
    if summary_df.empty:
        return
    nrows = len(summary_df)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, max(4, 1.4 * nrows)), sharex=False)
    if nrows == 1:
        axes = [axes]
    chosen_array = None
    if chosen_map_points is not None:
        candidate = np.asarray(chosen_map_points, dtype=float)
        if candidate.ndim == 2 and candidate.shape[1] == len(summary_df):
            finite_rows = candidate[np.isfinite(candidate).all(axis=1)]
            if finite_rows.size:
                chosen_array = finite_rows
    for param_index, (ax, row) in enumerate(zip(axes, summary_df.itertuples(index=False))):
        ax.hlines(1, row.p16, row.p84, linewidth=4, color="tab:blue")
        ax.scatter([row.median], [1], color="tab:blue", s=35, label="median")
        ax.scatter([row.map], [1], color="tab:red", marker="x", s=50, label="MAP")
        if chosen_array is not None:
            values = np.asarray(chosen_array[:, param_index], dtype=float)
            ax.scatter(values, np.full(values.shape[0], 1.0), color="tab:orange", s=18, alpha=0.75, label="chosen MAPs")
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
    chosen_map_points = _run_logged_phase(
        args,
        "plots.chosen_map_matrix",
        lambda: _chosen_ranked_map_matrix(state.parameter_specs, results.init_diagnostics),
    )
    _, scaling_chosen_map_points = _run_logged_phase(
        args,
        "plots.scaling_map_subset",
        lambda: _subset_parameter_matrix(state.parameter_specs, chosen_map_points, "scaling"),
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

    _run_logged_phase(args, "plots.corner", lambda: _plot_corner(run_dir, results.samples, state.parameter_specs, chosen_map_points=chosen_map_points))
    _run_logged_phase(
        args,
        "plots.potfile_corner",
        lambda: _plot_potfile_corner(run_dir, scaling_samples, scaling_specs, chosen_map_points=scaling_chosen_map_points),
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
        lambda: _plot_per_potential_summary(run_dir, summary_df, chosen_map_points=chosen_map_points),
    )
    _run_logged_phase(args, "plots.refresh_diagnostics", lambda: _plot_refresh_diagnostics(run_dir, family_df))
    _run_logged_phase(args, "plots.timing_profile", lambda: _plot_timing_profile(run_dir, evaluator))
    if args.plot_caustics or state.fit_mode != "small-only":
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
    raise ValueError("small-only mode requires --stage1-run-dir or a sequential run directory with stage1_large_only artifacts.")


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
    parameter_specs = list(large_parameter_specs)
    component_param_assignments = list(large_component_param_assignments)
    lens_model_list = list(large_lens_model_list)
    base_components = [_serialize_component(potential) for potential in potentials_with_priors]
    scaling_component_assignments: list[dict[str, Any]] = []
    scaling_component_records: list[dict[str, Any]] = []
    if fit_mode == "small-only":
        scaling_parameter_specs, scaling_param_indices, scaling_lens_model_list = _build_scaling_parameter_specs(
            potfiles,
            profile_variant=str(args.profile_variant),
            start_index=len(parameter_specs),
            kpc_per_arcsec=scaling_kpc_per_arcsec,
        )
        scaling_components, scaling_assignments, scaling_component_assignments, scaling_component_records = _build_scaling_components(
            potfiles,
            reference,
            scaling_param_indices,
            start_component_index=len(base_components),
            kpc_per_arcsec=scaling_kpc_per_arcsec,
        )
        parameter_specs.extend(scaling_parameter_specs)
        component_param_assignments.extend(scaling_assignments)
        lens_model_list.extend(scaling_lens_model_list)
        base_components.extend(scaling_components)
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
    )


def _distribution_for_spec(spec: ParameterSpec):
    if spec.prior_kind == "normal":
        return dist.Normal(float(spec.mean), float(spec.std))
    return dist.Uniform(float(spec.lower), float(spec.upper))


def _posterior_model(parameter_specs: list[ParameterSpec], evaluator: ClusterJAXEvaluator):
    def model():
        values = []
        for spec in parameter_specs:
            values.append(numpyro.sample(spec.sample_name, _distribution_for_spec(spec)))
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


def _finite_sample_rows(samples: np.ndarray) -> np.ndarray:
    samples_array = np.asarray(samples, dtype=float)
    if samples_array.ndim != 2 or samples_array.size == 0:
        return np.empty((0, samples_array.shape[-1] if samples_array.ndim == 2 else 0), dtype=float)
    return samples_array[np.isfinite(samples_array).all(axis=1)]


def _make_evaluator_for_pass(
    args: argparse.Namespace,
    state: BuildState,
    continuation_pass: ContinuationPass,
) -> ClusterJAXEvaluator:
    pass_state = _state_with_sigma_scale(state, continuation_pass.sigma_scale)
    evaluator = ClusterJAXEvaluator(
        state=pass_state,
        match_tolerance_arcsec=args.match_tolerance_arcsec,
        validate_top_k_families=continuation_pass.validate_top_k_families,
        sampling_engine=continuation_pass.sampling_engine,
        active_scaling_galaxies=args.active_scaling_galaxies,
        refresh_every=args.refresh_every,
        refresh_param_drift_frac=args.refresh_param_drift_frac,
        validation_approx=continuation_pass.validation_approx,
    )
    return evaluator


def _prepare_pass_evaluator(
    args: argparse.Namespace,
    state: BuildState,
    continuation_pass: ContinuationPass,
) -> tuple[ClusterJAXEvaluator, np.ndarray]:
    evaluator = _make_evaluator_for_pass(args, state, continuation_pass)
    midpoint = _default_theta(state.parameter_specs)
    if evaluator.surrogate_enabled:
        _log(
            args,
            (
                f"[surrogate:{continuation_pass.name}] initializing active_scaling={len(evaluator.active_scaling_component_indices)} "
                f"inactive_scaling={len(evaluator.inactive_scaling_component_indices)}"
            ),
        )
        evaluator.refresh_surrogate(midpoint, reason=f"{continuation_pass.name}_initial")
    _log(args, f"[compile:{continuation_pass.name}] tracing first JAX likelihood evaluation")
    compile_start = time.time()
    compile_loglike = evaluator.source_loglike(midpoint)
    compile_elapsed = time.time() - compile_start
    evaluator.timing_totals["initial_jit_compile"] += compile_elapsed
    _log(
        args,
        f"[compile:{continuation_pass.name}] trace complete in {_fmt_seconds(compile_elapsed)} loglike={compile_loglike:.3f}",
    )
    return evaluator, midpoint


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


def _run_map_search_pass(
    args: argparse.Namespace,
    state: BuildState,
    continuation_pass: ContinuationPass,
    carry_forward: np.ndarray | None,
) -> tuple[np.ndarray, list[dict[str, Any]], ClusterJAXEvaluator, list[MAPRunResult]]:
    evaluator, midpoint = _prepare_pass_evaluator(args, state, continuation_pass)
    broad_starts = _build_broad_starts(
        state.parameter_specs,
        num_broad_seeds=max(1, int(args.map_broad_seeds)),
        seed=None if args.seed is None else int(args.seed) + _stable_seed_offset(continuation_pass.name),
        boundary_frac=float(args.nuts_init_boundary_frac),
        carry_forward=carry_forward,
    )
    broad_optimizer = BoundedMAPOptimizer(
        parameter_specs=state.parameter_specs,
        logprob_fn=evaluator._source_loglike_fn,
        maxiter=args.map_maxiter,
    )
    _log(
        args,
        (
            f"[map:{continuation_pass.name}] broad search starts={len(broad_starts)} maxiter={args.map_maxiter} "
            f"sigma_scale={continuation_pass.sigma_scale:.2f} validation_top_k={continuation_pass.validate_top_k_families}"
        ),
    )
    best_broad, broad_history, ranked_broad = broad_optimizer.run(
        starts=broad_starts,
        stage_label=f"{continuation_pass.name}:broad",
        log_fn=(lambda message: _log(args, message)),
    )
    local_starts = _build_local_refinement_starts(
        state.parameter_specs,
        ranked_results=ranked_broad,
        num_local_seeds=max(0, int(args.map_local_refine_seeds)),
        jitter_scale=float(args.map_local_jitter_scale),
        seed=None if args.seed is None else int(args.seed) + 17 + _stable_seed_offset(continuation_pass.name),
        boundary_frac=float(args.nuts_init_boundary_frac),
    )
    if not local_starts:
        if evaluator.surrogate_enabled:
            evaluator.refresh_surrogate(best_broad, reason=f"{continuation_pass.name}_post_map")
        return best_broad, broad_history, evaluator, ranked_broad
    local_optimizer = BoundedMAPOptimizer(
        parameter_specs=state.parameter_specs,
        logprob_fn=evaluator._source_loglike_fn,
        maxiter=args.map_maxiter,
    )
    _log(args, f"[map:{continuation_pass.name}] local refinement starts={len(local_starts)}")
    best_local, local_history, ranked_local = local_optimizer.run(
        starts=local_starts,
        stage_label=f"{continuation_pass.name}:local",
        log_fn=(lambda message: _log(args, message)),
    )
    combined = ranked_broad + ranked_local
    combined.sort(key=lambda item: item.logprob, reverse=True)
    best_fit = combined[0].theta
    if evaluator.surrogate_enabled:
        evaluator.refresh_surrogate(best_fit, reason=f"{continuation_pass.name}_post_map")
    return best_fit, broad_history + local_history, evaluator, combined


def _run_continuation_map(
    args: argparse.Namespace,
    state: BuildState,
) -> tuple[np.ndarray, list[dict[str, Any]], ClusterJAXEvaluator, list[MAPRunResult]]:
    continuation_passes = _build_continuation_passes(args)
    carry_forward: np.ndarray | None = None
    full_history: list[dict[str, Any]] = []
    final_evaluator: ClusterJAXEvaluator | None = None
    best_fit: np.ndarray | None = None
    final_ranked_results: list[MAPRunResult] = []
    map_start = time.time()
    for continuation_pass in continuation_passes:
        pass_best, pass_history, pass_evaluator, pass_ranked_results = _run_map_search_pass(
            args=args,
            state=state,
            continuation_pass=continuation_pass,
            carry_forward=carry_forward,
        )
        full_history.extend(pass_history)
        carry_forward = pass_best
        final_evaluator = pass_evaluator
        best_fit = pass_best
        final_ranked_results = pass_ranked_results
    if best_fit is None or final_evaluator is None:
        raise RuntimeError("Continuation MAP search did not produce a best-fit point.")
    final_evaluator.timing_totals["map_runtime"] += time.time() - map_start
    return best_fit, full_history, final_evaluator, final_ranked_results


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
            map_history=[],
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
    if str(getattr(args, "sampler", DEFAULT_SAMPLER)) == "blackjax_smc":
        _log(args, "[smc] skipping continuation MAP stage and initializing direct evaluator")
        evaluator, _midpoint = _prepare_direct_evaluator(args, state)
        best_fit = None
        map_history = []
        ranked_results = []
        sample_model = _posterior_model(state.parameter_specs, evaluator)
        best_fit, posterior = _run_smc_sampler(args, state, evaluator, ranked_results, map_history)
        if bool(getattr(args, "refine_with_nuts", False)):
            _log(args, "[smc] refining top particles with local NUTS chains")
            nuts_init = _build_smc_refine_initialization(
                args,
                state.parameter_specs,
                posterior.samples,
                posterior.log_prob,
                posterior.sample_weights,
            )
            posterior = _run_numpyro_nuts_sampler(args, state, evaluator, sample_model, nuts_init, map_history)
            if posterior.samples.size:
                best_fit = np.asarray(posterior.samples[int(np.nanargmax(posterior.log_prob))], dtype=float)
            posterior.init_diagnostics = dict(posterior.init_diagnostics or {})
            posterior.init_diagnostics.update(
                {
                    "direct_evaluator_startup": True,
                    "map_stage_skipped": True,
                    "post_smc_nuts_refine_requested": True,
                    "post_smc_nuts_refine_used": True,
                }
            )
    else:
        if str(args.nuts_init_strategy) in {"svi+nuts", "prior_center"}:
            _log(args, f"[model] skipping MAP and initializing direct evaluator for {args.nuts_init_strategy}")
            evaluator, _midpoint = _prepare_direct_evaluator(args, state)
            best_fit = None
            map_history = []
            ranked_results = []
        else:
            _log(args, "[model] initializing continuation MAP search")
            best_fit, map_history, evaluator, ranked_results = _run_logged_phase(
                args,
                "continuation_map",
                lambda: _run_continuation_map(args, state),
            )
            best_map_loglike = max(entry["best_loglike"] for entry in map_history)
            _log(
                args,
                f"[map] complete best_loglike={best_map_loglike:.3f} final_pass={_build_continuation_passes(args)[-1].name} "
                f"evaluations={len(map_history)}",
            )

        sample_model = _posterior_model(state.parameter_specs, evaluator)
        nuts_init = _build_nuts_initialization(args, state.parameter_specs, ranked_results, sample_model)
        if best_fit is None:
            best_fit = np.asarray(nuts_init.reference_theta, dtype=float)
        if str(nuts_init.diagnostics.get("strategy_used", "")) == "prior_center":
            _log(
                args,
                f"[nuts:init] prior_center distinct_seeds={int(nuts_init.diagnostics.get('distinct_chain_seeds', 0))}",
            )
        if str(nuts_init.diagnostics.get("strategy_used", "")) == "ranked_map":
            chosen_ranked_maps = list(nuts_init.diagnostics.get("chosen_ranked_maps", []))
            if chosen_ranked_maps:
                chosen_text = ", ".join(
                    f"{str(item.get('label', 'unknown'))}:{float(item.get('logprob', float('nan'))):.3f}"
                    for item in chosen_ranked_maps
                )
                _log(args, f"[nuts:init] chosen_ranked_maps={chosen_text}")
        posterior = _run_numpyro_nuts_sampler(args, state, evaluator, sample_model, nuts_init, map_history)
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
        _log(args, "[plots-only] converted legacy saved-SMC refine posterior arrays from latent to physical units")
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
        map_history=[],
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
    stage1_run_dir = _run_single_stage(args, "large-only", stage1_run_name)
    stage1_summary = _load_stage1_summary(stage1_run_dir / "artifacts")
    stage2_run_name = str(Path(root_run_name) / "stage2_small_only")
    stage2_run_dir = _run_single_stage(args, "small-only", stage2_run_name, stage1_prior_summary=stage1_summary)
    summary_payload = {
        "fit_mode": "sequential",
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


def _run_refine_from_saved_smc(args: argparse.Namespace) -> None:
    if not args.refine_from_run_dir:
        raise ValueError("--refine-from-run-dir is required for saved-SMC refinement.")
    source_artifacts_dir = _resolve_run_artifacts_dir(args.refine_from_run_dir)
    if not _has_plot_artifacts(source_artifacts_dir):
        _fail(f"Missing saved artifacts for refinement mode: {source_artifacts_dir}")
    _log(args, f"[refine] loading saved SMC artifacts from {source_artifacts_dir}")
    state, saved_args, arrays, _init_diagnostics = _load_artifacts(source_artifacts_dir)
    source_sampler = str(saved_args.get("sampler", DEFAULT_SAMPLER))
    if source_sampler != "blackjax_smc":
        raise ValueError(
            f"--refine-from-run-dir requires a blackjax_smc source run, but found sampler={source_sampler!r}."
        )
    if "samples" not in arrays or "log_prob" not in arrays or "sample_weights" not in arrays:
        raise ValueError("Saved SMC refinement source is missing one of: samples, log_prob, sample_weights.")

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
    sample_model = _posterior_model(state.parameter_specs, evaluator)
    particles_physical = np.asarray(arrays["samples"], dtype=float)
    particles_latent = _convert_sample_matrix_to_latent(particles_physical, state.parameter_specs)
    log_prob = np.asarray(arrays["log_prob"], dtype=float)
    sample_weights = np.asarray(arrays["sample_weights"], dtype=float)
    nuts_init = _build_smc_refine_initialization(args, state.parameter_specs, particles_latent, log_prob, sample_weights)
    nuts_init = replace(
        nuts_init,
        diagnostics={
            **dict(nuts_init.diagnostics),
            "source_run_dir": str(Path(args.refine_from_run_dir)),
            "source_artifacts_dir": str(source_artifacts_dir),
            "source_sampler": source_sampler,
            "strategy_requested": "saved_smc_refine_nuts",
            "strategy_used": "saved_smc_refine_nuts",
        },
    )
    posterior = _run_numpyro_nuts_sampler(args, state, evaluator, sample_model, nuts_init, map_history=[])
    best_fit = nuts_init.reference_theta
    if posterior.samples.size:
        best_index = int(np.nanargmax(posterior.log_prob))
        best_fit = np.asarray(posterior.samples[best_index], dtype=float)
        _log(args, f"[nuts] best_fit updated from retained posterior sample index={best_index}")
    posterior.init_diagnostics = dict(posterior.init_diagnostics or {})
    posterior.init_diagnostics.update(
        {
            "source_run_dir": str(Path(args.refine_from_run_dir)),
            "source_artifacts_dir": str(source_artifacts_dir),
            "source_sampler": source_sampler,
            "saved_smc_refine": True,
        }
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

    source_run_path = Path(args.refine_from_run_dir)
    inferred_run_name = f"{source_run_path.name}_cpu_refine"
    run_name = args.run_name or inferred_run_name
    run_dir = Path(args.output_dir) / run_name
    _configure_debug_log(args, run_name, run_dir)
    runtime_sec = evaluator.timing_totals.get("nuts_runtime", 0.0) + evaluator.timing_totals.get("validation_runtime", 0.0)
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
    state_for_output = _state_with_run_name(state, run_name)
    _run_logged_phase(
        args,
        "output.save_artifacts",
        lambda: _save_artifacts(artifacts_dir, state_for_output, args, best_fit_physical, posterior_for_output),
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
                state=state_for_output,
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


def main() -> None:
    try:
        args = _parse_args()
        if args.seed is not None:
            np.random.seed(args.seed)
        inferred_run_name = args.run_name
        if inferred_run_name is None and args.par_path:
            inferred_run_name = _make_run_name(args.par_path)
        elif inferred_run_name is None and args.refine_from_run_dir:
            inferred_run_name = Path(args.refine_from_run_dir).name
        else:
            inferred_run_name = inferred_run_name or "cluster_solver"
        _configure_debug_log(args, inferred_run_name, None)
        _log(args, "[main] startup")
        if not args.par_path and not args.refine_from_run_dir and not args.plots_only:
            _fail("--par-path is required unless --refine-from-run-dir or --plots-only is used.")
        if args.plots_only:
            if args.fit_mode == "sequential":
                root_run_name = args.run_name or _make_run_name(args.par_path)
                root_dir = Path(args.output_dir) / root_run_name
                stage_dirs = [root_dir / "stage1_large_only", root_dir / "stage2_small_only"]
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
        if args.refine_from_run_dir:
            _run_refine_from_saved_smc(args)
            return
        if args.fit_mode == "sequential":
            _run_sequential(args)
            return
        _run_single_stage(args, args.fit_mode, args.run_name or _make_run_name(args.par_path))
    except BaseException as exc:
        _log_exception("main", exc)
        raise
    finally:
        _close_debug_log()


if __name__ == "__main__":
    main()
