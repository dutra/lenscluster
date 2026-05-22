from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.optim as numpyro_optim
import pandas as pd
import h5py
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, LambdaCDM, wCDM
from jax import config as jax_config
from matplotlib import use as matplotlib_use
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import unconstrain_fn
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal test environments
    class Progress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "Progress":
            return self

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
            return False

        def add_task(self, description: str, total: int | None = None) -> int:
            return 0

    class SpinnerColumn:
        pass

    class TextColumn:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class TimeElapsedColumn:
        pass
from scipy.optimize import linear_sum_assignment

if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = f"/tmp/numba_cache_{os.getuid()}"
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

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
from jaxtronomy.Util import param_util
try:
    from jaxtronomy.LensModel.Solver.lens_equation_solver_jax import (
        image_position_lenstronomy_jax as _jax_image_position_lenstronomy,
    )
except Exception:  # pragma: no cover - optional fast path import guard
    _jax_image_position_lenstronomy = None

from .lenstool_parser import load_best_par
from .jax_cosmology import (
    DEFAULT_JAX_COSMO_DISTANCE_STEPS,
    cosmology_config_from_parsed as _cosmology_config_from_parsed,
    dpie_sigma0_factor as _jax_dpie_sigma0_factor,
    dpie_sigma0_factor_from_config as _dpie_sigma0_factor_from_config,
    dpie_sigma0_from_vel_disp as _jax_dpie_sigma0_from_vel_disp,
    flat_wcdm_comoving_distance_mpc as _jax_flat_wcdm_comoving_distance_mpc,
    flat_wcdm_kpc_per_arcsec as _jax_flat_wcdm_kpc_per_arcsec,
    flat_wcdm_lens_geometry_factors as _jax_flat_wcdm_lens_geometry_factors,
    flat_wcdm_lensing_efficiency as _jax_flat_wcdm_lensing_efficiency,
    h0_from_config as _h0_from_config,
    kpc_per_arcsec_from_config as _kpc_per_arcsec_from_config,
    ode0_from_config as _ode0_from_config,
    om0_from_config as _om0_from_config,
    w0_from_config as _w0_from_config,
)
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
from .plotting import _format_sequential_run_summary_text, _generate_plots_and_tables
from .utils import (
    close_debug_log as _close_debug_log,
    configure_debug_log as _configure_debug_log,
    fmt_seconds as _fmt_seconds,
    log_exception as _log_exception,
    log_message as _log,
    log_stage_banner as _log_stage_banner,
    make_run_name as _make_run_name,
    parse_bool_env as _parse_bool_env,
    run_logged_phase as _run_logged_phase,
)


SUPPORTED_PROFILES = {81, 14}
DP_IE_PROFILE = 81
SHEAR_PROFILE = 14
DEFAULT_MATCH_TOLERANCE = 1.5
DEFAULT_SEARCH_PADDING = 8.0
DEFAULT_Z_BIN_EFFICIENCY_TOL = 0.01
DEFAULT_WARMUP = 300
DEFAULT_SAMPLES = 500
DEFAULT_TARGET_ACCEPT = 0.85
DEFAULT_MAX_TREE_DEPTH = 10
DEFAULT_INITIAL_STEP_SIZE = 1.0e-3
DEFAULT_ACTIVE_SCALING_GALAXIES = 64
DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION = 0.995
DEFAULT_ACTIVE_SCALING_MIN = 4
DEFAULT_REFRESH_EVERY = 250
DEFAULT_REFRESH_PARAM_DRIFT_FRAC = 0.25
DEFAULT_VALIDATION_RMS_FACTOR = 1.5
DEFAULT_NUTS_INIT_BOUNDARY_FRAC = 0.02
DEFAULT_NUTS_INIT_JITTER_FRAC = 0.02
DEFAULT_SVI_STEPS = 2000
DEFAULT_SVI_LEARNING_RATE = 5.0e-3
DEFAULT_NS_MAX_SAMPLES = None
DEFAULT_NS_POSTERIOR_SAMPLES = 4096
DEFAULT_NS_DLOGZ = 1.0e-4
DEFAULT_POSTERIOR_LOGPROB_BATCH_SIZE = 512
DEFAULT_SAMPLER = "numpyro_nuts"
DEFAULT_SOURCE_SIGMA_INT_LOWER_ARCSEC = 1.0e-3
DEFAULT_SOURCE_SIGMA_INT_UPPER_ARCSEC = 2.0
DEFAULT_IMAGE_SIGMA_INT_LOWER_ARCSEC = 1.0e-3
DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC = 2.0
DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC = 0.3
DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC = 10.0
DEFAULT_IMAGE_PRESENCE_STAGE4_PENALTY_WEIGHT = 2.0
DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC = 0.30
DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC = 0.10
DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS = 0.05
DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN = 0.05
DEFAULT_COSMOLOGY_OM0_LOWER = 0.05
DEFAULT_COSMOLOGY_OM0_UPPER = 0.6
DEFAULT_COSMOLOGY_W0_LOWER = -2.0
DEFAULT_COSMOLOGY_W0_UPPER = -0.3
COSMOLOGY_OM0_SAMPLE_NAME = "cosmology_Om0"
COSMOLOGY_W0_SAMPLE_NAME = "cosmology_w0"
SAFE_SCALING_EXPONENT_ABS_MIN = 1.0e-3
SAFE_RADIUS_MARGIN_ARCSEC = 1.0e-3
SAFE_RADIUS_MARGIN_KPC = 1.0e-3
SAFE_VDISP_MARGIN = 1.0e-6
NUTS_MAX_TREE_SATURATION_WARNING = 0.95
NUTS_RHAT_EXTREME_WARNING = 2.0
NUTS_MIN_ESS_PER_CHAIN_WARNING = 2.0
BAD_LOG_LIKE = -1.0e30
SAMPLE_LIKELIHOOD_SOURCE = "source"
SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN = "local-jacobian"
SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE = "linearized-forward-beta-image-plane"
SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE = "linearized-marginal-beta-image-plane"
EVIDENCE_LIKELIHOOD_MODES = (
    SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
)
DEFAULT_EVIDENCE_LIKELIHOOD_MODE = SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE
SOURCE_POSITION_PARAMETERIZATION_DIRECT = "direct"
SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED = "prior-whitened"
SOURCE_POSITION_PARAMETERIZATION_CONDITIONAL_WHITENED = "conditional-whitened"
SOURCE_POSITION_PARAMETERIZATIONS = (
    SOURCE_POSITION_PARAMETERIZATION_DIRECT,
    SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED,
    SOURCE_POSITION_PARAMETERIZATION_CONDITIONAL_WHITENED,
)
IMAGE_PLANE_MODE_NONE = "none"
IMAGE_PLANE_MODE_LOCAL_JACOBIAN = "local-jacobian"
IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA = "linearized-forward-beta-image-plane"
FIT_MODE_SEQUENTIAL = "sequential"
FIT_MODE_LARGE_ONLY = "large-only"
FIT_MODE_JOINT = "joint"
FIT_MODE_EVIDENCE_NS = "evidence-ns"
FIT_METHOD_SVI = "svi"
FIT_METHOD_SVI_NUTS = "svi+nuts"
FIT_METHOD_NS = "ns"
EXACT_IMAGE_SOLVER_AUTO = "auto"
EXACT_IMAGE_SOLVER_JAX = "jax"
EXACT_IMAGE_SOLVER_LENSTRONOMY = "lenstronomy"
EXACT_IMAGE_SOLVERS = (
    EXACT_IMAGE_SOLVER_AUTO,
    EXACT_IMAGE_SOLVER_JAX,
    EXACT_IMAGE_SOLVER_LENSTRONOMY,
)
DEFAULT_EXACT_IMAGE_SOLVER = EXACT_IMAGE_SOLVER_AUTO


@dataclass(frozen=True)
class StageFitControls:
    fit_method: str
    warmup: int
    samples: int

    def to_json(self) -> dict[str, str | int]:
        return {
            "fit_method": self.fit_method,
            "warmup": self.warmup,
            "samples": self.samples,
        }


def _parse_optional_positive_int(value: str) -> int | None:
    text = str(value).strip()
    if text.lower() in {"none", "null"}:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return parsed


@dataclass(frozen=True)
class TracedBinData:
    effective_z_source: float
    family_ids: tuple[str, ...]
    n_families: int
    family_index_per_image: jnp.ndarray
    x_obs: jnp.ndarray
    y_obs: jnp.ndarray
    sigma_per_image: jnp.ndarray
    reliability_per_image: jnp.ndarray
    image_has_constraint: jnp.ndarray
    effective_z_index: int = -1
ORIGINAL_DPIE_PROFILE_NAME = "DPIE_NIE"
UNSUPPORTED_PJAFFE_PROFILE_NAMES = {
    "PJAFFE_COMPACT",
    "PJAFFE_ELLIPSE_POTENTIAL",
    "PJAFFE_ELLIPSE_POTENTIAL_COMPACT",
}


def _validate_supported_lens_model_list(lens_model_list: list[str], context: str) -> None:
    unsupported = sorted(set(str(name) for name in lens_model_list) & UNSUPPORTED_PJAFFE_PROFILE_NAMES)
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(
            f"{context} uses unsupported compact/PJAFFE lens profiles ({joined}). "
            "Regenerate the artifacts with the DPIE_NIE profile."
        )


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
        if spec.transform_kind not in {"log_positive", "log_offset_positive", "affine"}:
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
    if results.grouped_samples is not None and results.sampler != "svi":
        grouped_samples = _convert_sample_matrix_to_physical(results.grouped_samples, parameter_specs)
    ns_diagnostics = _convert_ns_diagnostics_samples(results.ns_diagnostics, parameter_specs)
    init_diagnostics = dict(results.init_diagnostics or {})
    converted = replace(
        results,
        samples=_convert_sample_matrix_to_physical(results.samples, parameter_specs),
        grouped_samples=grouped_samples,
        grouped_log_prob=None if results.sampler == "svi" else results.grouped_log_prob,
        init_diagnostics=init_diagnostics,
        ns_diagnostics=ns_diagnostics,
    )
    _check_physical_sample_matrix(converted.samples, parameter_specs, context="posterior.samples")
    _check_physical_sample_matrix(converted.grouped_samples, parameter_specs, context="posterior.grouped_samples")
    if converted.ns_diagnostics is not None and "samples" in converted.ns_diagnostics:
        _check_physical_sample_matrix(converted.ns_diagnostics["samples"], parameter_specs, context="posterior.ns_diagnostics.samples")
    return converted


def _evaluator_uses_conditional_source_transport(evaluator: Any | None) -> bool:
    return bool(
        evaluator is not None
        and getattr(evaluator, "source_position_parameterization", None)
        == SOURCE_POSITION_PARAMETERIZATION_CONDITIONAL_WHITENED
    )


def _convert_theta_to_reported_physical(
    theta: np.ndarray,
    parameter_specs: list[ParameterSpec],
    evaluator: Any | None = None,
) -> np.ndarray:
    if _evaluator_uses_conditional_source_transport(evaluator):
        return np.asarray(
            evaluator.reported_physical_parameter_vector(jnp.asarray(theta, dtype=jnp.float64)),
            dtype=float,
        )
    return _convert_theta_to_physical(theta, parameter_specs)


def _convert_sample_matrix_to_reported_physical(
    samples: np.ndarray | None,
    parameter_specs: list[ParameterSpec],
    evaluator: Any | None = None,
) -> np.ndarray | None:
    if samples is None:
        return None
    array = np.asarray(samples, dtype=float)
    if array.size == 0:
        return array.copy()
    if not _evaluator_uses_conditional_source_transport(evaluator):
        return _convert_sample_matrix_to_physical(array, parameter_specs)
    original_shape = array.shape
    flat = array.reshape((-1, original_shape[-1]))
    converted = np.asarray(
        jax.vmap(evaluator._reported_physical_parameter_vector)(jnp.asarray(flat, dtype=jnp.float64)),
        dtype=float,
    )
    return converted.reshape(original_shape)


def _convert_ns_diagnostics_samples(
    ns_diagnostics: dict[str, np.ndarray] | None,
    parameter_specs: list[ParameterSpec],
    evaluator: Any | None = None,
) -> dict[str, np.ndarray] | None:
    if not ns_diagnostics:
        return None
    converted = {str(key): np.asarray(value).copy() for key, value in ns_diagnostics.items()}
    samples = converted.get("samples")
    if samples is not None and samples.size:
        converted["samples"] = _convert_sample_matrix_to_reported_physical(samples, parameter_specs, evaluator)
    return converted


def _posterior_results_to_reported_physical(
    results: PosteriorResults,
    parameter_specs: list[ParameterSpec],
    evaluator: Any | None = None,
) -> PosteriorResults:
    if not _evaluator_uses_conditional_source_transport(evaluator):
        return _posterior_results_to_physical(results, parameter_specs)
    grouped_samples = None
    if results.grouped_samples is not None and results.sampler != "svi":
        grouped_samples = _convert_sample_matrix_to_reported_physical(results.grouped_samples, parameter_specs, evaluator)
    ns_diagnostics = _convert_ns_diagnostics_samples(results.ns_diagnostics, parameter_specs, evaluator)
    init_diagnostics = dict(results.init_diagnostics or {})
    init_diagnostics["conditional_source_transport_physical_output"] = True
    converted = replace(
        results,
        samples=_convert_sample_matrix_to_reported_physical(results.samples, parameter_specs, evaluator),
        grouped_samples=grouped_samples,
        grouped_log_prob=None if results.sampler == "svi" else results.grouped_log_prob,
        init_diagnostics=init_diagnostics,
        ns_diagnostics=ns_diagnostics,
    )
    _check_physical_sample_matrix(converted.samples, parameter_specs, context="posterior.samples")
    _check_physical_sample_matrix(converted.grouped_samples, parameter_specs, context="posterior.grouped_samples")
    if converted.ns_diagnostics is not None and "samples" in converted.ns_diagnostics:
        _check_physical_sample_matrix(converted.ns_diagnostics["samples"], parameter_specs, context="posterior.ns_diagnostics.samples")
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
        if spec.transform_kind not in {"log_positive", "log_offset_positive", "affine"}:
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
    normalized = {
        key: (
            {str(subkey): np.asarray(subvalue) for subkey, subvalue in value.items()}
            if isinstance(value, dict)
            else np.asarray(value)
        )
        for key, value in arrays.items()
    }
    if not _loaded_posterior_arrays_need_physical_conversion(normalized, parameter_specs, init_diagnostics):
        return normalized, False
    converted = dict(normalized)
    converted["samples"] = _convert_sample_matrix_to_physical(np.asarray(normalized["samples"], dtype=float), parameter_specs)
    if "grouped_samples" in normalized:
        converted["grouped_samples"] = _convert_sample_matrix_to_physical(
            np.asarray(normalized["grouped_samples"], dtype=float),
            parameter_specs,
        )
    ns_diagnostics = normalized.get("ns_diagnostics")
    if isinstance(ns_diagnostics, dict) and "samples" in ns_diagnostics:
        converted_ns_diagnostics = dict(ns_diagnostics)
        converted_ns_diagnostics["samples"] = _convert_sample_matrix_to_physical(
            np.asarray(ns_diagnostics["samples"], dtype=float),
            parameter_specs,
        )
        converted["ns_diagnostics"] = converted_ns_diagnostics
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
        "--resume",
        action="store_true",
        help="Reuse completed run/stage artifacts and continue from the first incomplete stage.",
    )
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
    parser.add_argument(
        "--quick-diagnostics",
        action="store_true",
        help=(
            "Fast post-fit diagnostics: skip exact image-position validation, caustic overlay, "
            "and exact image-position fit-quality draws while keeping source-plane and magnification summaries."
        ),
    )
    parser.add_argument(
        "--fit-quality-draws",
        type=int,
        default=0,
        help=(
            "Maximum posterior draws used for fit-quality image and model magnification uncertainty intervals. "
            "Defaults to 0, which runs exact image diagnostics for the best fit only."
        ),
    )
    parser.add_argument(
        "--fit-quality-workers",
        type=int,
        default=1,
        help="Worker threads for posterior fit-quality image and model magnification diagnostics.",
    )
    parser.add_argument(
        "--exact-image-solver",
        choices=EXACT_IMAGE_SOLVERS,
        default=DEFAULT_EXACT_IMAGE_SOLVER,
        help=(
            "Exact image-position solver for validation and fit-quality diagnostics. "
            "auto tries the JAX-vectorized solver first and falls back to lenstronomy."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stage logs and NS progress output; NumPyro SVI/NUTS progress bars may still render.",
    )
    parser.add_argument(
        "--sampling-engine",
        choices=("full", "refreshing_surrogate"),
        default="refreshing_surrogate",
        help="Use the exact full source-plane likelihood or a first-order surrogate around a refreshed reference point.",
    )
    parser.add_argument(
        "--source-plane-covariance-floor",
        type=float,
        default=1.0e-6,
        help="Source-plane covariance diagonal floor in arcsec^2 for magnification-weighted source-plane errors.",
    )
    parser.add_argument(
        "--active-scaling-galaxies",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Per-potfile fixed counts, or adaptive maximum counts, of most-important scaling-law galaxies "
            "to keep exact in surrogate mode, in potfile order. Negative values mean use all galaxies."
        ),
    )
    parser.add_argument(
        "--active-scaling-selection",
        choices=("fixed", "adaptive"),
        default="adaptive",
        help=(
            "How to choose active scaling-law galaxies. fixed uses --active-scaling-galaxies directly; "
            "adaptive chooses the cutoff from the ranked importance curve."
        ),
    )
    parser.add_argument(
        "--active-scaling-cumulative-fraction",
        type=float,
        default=DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        help="Adaptive selection target cumulative ranking importance to include exactly.",
    )
    parser.add_argument(
        "--active-scaling-min",
        type=int,
        default=DEFAULT_ACTIVE_SCALING_MIN,
        help="Minimum active galaxies per potfile for adaptive scaling selection.",
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
        nargs="+",
        choices=(FIT_METHOD_SVI, FIT_METHOD_SVI_NUTS, FIT_METHOD_NS),
        default=[FIT_METHOD_SVI_NUTS],
        metavar="{svi,svi+nuts,ns}",
        help=(
            "Use SVI only or use SVI to initialize optional NUTS sampling. "
            "The ns value is accepted only for backwards-compatible parsing and is reserved for --fit-mode evidence-ns. "
            "In sequential image-plane runs, pass one value for all sampled stages, two values for stage 2 and "
            "stage 3, or three values when the linearized stage 4 is enabled."
        ),
    )
    parser.add_argument(
        "--image-plane-mode",
        choices=(
            IMAGE_PLANE_MODE_NONE,
            IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
            IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        ),
        default=IMAGE_PLANE_MODE_NONE,
        help=(
            "Optional image-plane refinement mode. 'none' keeps the standard source-plane workflow; "
            "'local-jacobian' adds a differentiable local image-plane approximation stage; "
            "'linearized-forward-beta-image-plane' adds the explicit-beta linearized image-plane stage."
        ),
    )
    parser.add_argument(
        "--skip-stage3-image-plane-local-jacobian",
        action="store_true",
        help="Skip the local-Jacobian stage 3 before an explicit-beta stage 4.",
    )
    parser.add_argument(
        "--image-plane-newton-steps",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
        help=(
            "Additional Newton updates after the initial linearized image-plane correction in stage 4. "
            "Zero still performs one local linear solve at each observed image."
        ),
    )
    parser.add_argument(
        "--linearized-beta-prior-sigma-arcsec",
        type=float,
        default=DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC,
        help="Normal-prior sigma for explicit stage-4 source coordinates, centered on stage-3 source centroids.",
    )
    parser.add_argument(
        "--source-position-parameterization",
        choices=SOURCE_POSITION_PARAMETERIZATIONS,
        default=SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED,
        help=(
            "Stage-4 explicit source-position sampling coordinate. direct samples beta in arcsec; "
            "prior-whitened samples unit prior offsets and maps beta=mu+sigma*eta; "
            "conditional-whitened is experimental."
        ),
    )
    parser.add_argument(
        "--evidence-source-prior-sigma-arcsec",
        type=float,
        default=None,
        help=(
            "Required for --fit-mode evidence-ns. Isotropic Gaussian source-position prior sigma, "
            "shared by all families."
        ),
    )
    parser.add_argument(
        "--evidence-source-prior-mean-x-arcsec",
        type=float,
        default=0.0,
        help="Gaussian source-position prior mean beta_x for --fit-mode evidence-ns.",
    )
    parser.add_argument(
        "--evidence-source-prior-mean-y-arcsec",
        type=float,
        default=0.0,
        help="Gaussian source-position prior mean beta_y for --fit-mode evidence-ns.",
    )
    parser.add_argument(
        "--evidence-likelihood-mode",
        choices=EVIDENCE_LIKELIHOOD_MODES,
        default=DEFAULT_EVIDENCE_LIKELIHOOD_MODE,
        help=(
            "Likelihood target for --fit-mode evidence-ns. The default analytically marginalizes source "
            "positions; linearized-forward-beta-image-plane samples one source position per family and "
            "fits the linearized image-plane residuals."
        ),
    )
    parser.add_argument(
        "--image-plane-scatter-upper-arcsec",
        type=float,
        default=DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC,
        help="Upper bound for the stage-4 intrinsic image-plane scatter parameter.",
    )
    parser.add_argument(
        "--image-presence-penalty-weight",
        type=float,
        default=None,
        help=(
            "Weight for the smooth observed-image presence penalty in linearized-forward-beta image-plane mode. "
            "Defaults to 2.0 for sequential stage 4 and 0.0 otherwise; set 0.0 to disable."
        ),
    )
    parser.add_argument(
        "--image-presence-match-radius-arcsec",
        type=float,
        default=DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC,
        help="Image-plane residual radius where an observed image is counted as softly present.",
    )
    parser.add_argument(
        "--image-presence-temperature-arcsec",
        type=float,
        default=DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC,
        help="Smooth transition scale for observed-image presence probabilities.",
    )
    parser.add_argument(
        "--image-presence-count-softness",
        type=float,
        default=DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS,
        help="Softplus scale for the family-level observed-image count shortfall.",
    )
    parser.add_argument(
        "--image-presence-count-margin",
        type=float,
        default=DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN,
        help="Reliability-weighted image-count margin before the presence penalty activates.",
    )
    parser.add_argument(
        "--validation-approx",
        choices=("exact", "adaptive"),
        default="adaptive",
        help="Use exact lens-equation validation for all selected families or only for degraded families.",
    )
    parser.add_argument(
        "--source-plane-outlier-sigma-arcsec",
        type=float,
        default=DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC,
        help="Broad source-plane sigma for fixed reliability-weighted candidate-family mixture terms.",
    )
    parser.add_argument(
        "--z-bin-efficiency-tol",
        type=float,
        default=DEFAULT_Z_BIN_EFFICIENCY_TOL,
        help=(
            "Fractional tolerance for grouping source planes by lensing efficiency D_ls / D_s. "
            "Higher-redshift sources are binned more coarsely because their efficiency changes slowly."
        ),
    )
    parser.add_argument(
        "--fit-cosmology-flat-wcdm",
        action="store_true",
        help=(
            "Sample a flat wCDM cosmology with fixed H0 and free Omega_m,w0. "
            "In sequential runs this is applied only to the final fitting stage."
        ),
    )
    parser.add_argument(
        "--cosmology-init-om0",
        type=float,
        default=None,
        help="Optional SVI start value for sampled flat-wCDM Omega_m.",
    )
    parser.add_argument(
        "--cosmology-init-w0",
        type=float,
        default=None,
        help="Optional SVI start value for sampled flat-wCDM w0.",
    )
    parser.add_argument(
        "--validate-top-k-families",
        type=int,
        default=0,
        help="Number of informative families to validate exactly. Omit or set to 0 to skip exact image-plane validation.",
    )
    parser.add_argument(
        "--match-tolerance-arcsec",
        type=float,
        default=DEFAULT_MATCH_TOLERANCE,
        help="Maximum assignment residual for exact image matching.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        nargs="+",
        default=[DEFAULT_WARMUP],
        help=(
            "NUTS warmup steps. In sequential image-plane runs, pass one value for all sampled stages, "
            "two values for stage 2 and stage 3, or three values when the linearized stage 4 is enabled."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[DEFAULT_SAMPLES],
        help=(
            "Posterior draws per chain. In sequential image-plane runs, pass one value for all sampled stages, "
            "two values for stage 2 and stage 3, or three values when the linearized stage 4 is enabled."
        ),
    )
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument(
        "--fit-mode",
        choices=(FIT_MODE_SEQUENTIAL, FIT_MODE_LARGE_ONLY, FIT_MODE_JOINT, FIT_MODE_EVIDENCE_NS),
        default=FIT_MODE_SEQUENTIAL,
        help=(
            "Workflow to run. sequential runs large-only then joint; joint runs one stage with all parameters "
            "and is faster for iteration when stage-1 tightening is not needed. evidence-ns runs one joint "
            "nested-sampling evidence target."
        ),
    )
    parser.set_defaults(sampler="numpyro_nuts", stage1_run_dir=None, sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE)
    parser.add_argument("--max-tree-depth", type=int, default=DEFAULT_MAX_TREE_DEPTH)
    parser.add_argument("--target-accept", type=float, default=DEFAULT_TARGET_ACCEPT)
    parser.add_argument(
        "--initial-step-size",
        type=float,
        default=DEFAULT_INITIAL_STEP_SIZE,
        help="Initial NUTS step size before warmup adaptation. Small defaults avoid invalid dPIE states during early warmup.",
    )
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
        "--ns-num-live-points",
        type=int,
        default=None,
        help="JAXNS live points for --fit-mode evidence-ns. Defaults to 25 times the number of free parameters.",
    )
    parser.add_argument(
        "--ns-max-samples",
        type=_parse_optional_positive_int,
        default=DEFAULT_NS_MAX_SAMPLES,
        help="JAXNS maximum nested-sampling samples for --fit-mode evidence-ns. Defaults to unlimited; pass a positive integer to cap.",
    )
    parser.add_argument(
        "--ns-dlogz",
        type=float,
        default=DEFAULT_NS_DLOGZ,
        help="JAXNS dlogZ termination threshold for --fit-mode evidence-ns.",
    )
    parser.add_argument(
        "--caustic-num-pix",
        type=int,
        default=250,
        help="Grid resolution for caustic overlay plot.",
    )
    parser.add_argument(
        "--caustic-source-redshift",
        type=float,
        default=7.0,
        help="Source redshift used for critical-line and caustic overlay diagnostics.",
    )
    parser.add_argument(
        "--plot-caustics",
        action="store_true",
        help="Generate the expensive caustic overlay plot for the joint stage too.",
    )
    parser.add_argument(
        "--truth",
        default=None,
        help=(
            "Optional path to a truth.json file. When provided, mock-truth recovery PDFs are written "
            "under each completed stage's validation/ directory."
        ),
    )
    return parser.parse_args()


def _format_count_map(values: dict[str, int] | Counter) -> str:
    if not values:
        return "none"
    return ",".join(f"{key}={int(values[key])}" for key in sorted(values))


def _explicit_source_position_parameterization_for_state(state: BuildState) -> str:
    if not hasattr(state, "source_position_parameterization"):
        raise ValueError(
            "BuildState is missing explicit source_position_parameterization metadata; "
            "rerun the solver to regenerate artifacts with explicit stage-4 source-position metadata."
        )
    mode = str(state.source_position_parameterization)
    if mode not in SOURCE_POSITION_PARAMETERIZATIONS:
        raise ValueError(
            f"Unsupported source_position_parameterization={mode!r}; "
            f"expected one of {', '.join(SOURCE_POSITION_PARAMETERIZATIONS)}."
        )
    source_position_specs = [spec for spec in state.parameter_specs if spec.component_family == "source_position"]
    if not source_position_specs:
        return mode
    if mode == SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED and not all(
        str(spec.transform_kind) == "affine" for spec in source_position_specs
    ):
        raise ValueError("prior-whitened source-position mode requires affine source-position parameter specs.")
    if mode == SOURCE_POSITION_PARAMETERIZATION_CONDITIONAL_WHITENED and not all(
        str(spec.transform_kind) == "identity"
        and spec.prior_kind == "normal"
        and abs(float(spec.mean or 0.0)) < 1.0e-12
        and abs(float(spec.std or 0.0) - 1.0) < 1.0e-12
        and spec.physical_mean is not None
        and spec.physical_std is not None
        for spec in source_position_specs
    ):
        raise ValueError("conditional-whitened source-position mode requires unit-normal eta source-position specs.")
    return mode


def _log_runtime_summary(args: argparse.Namespace) -> None:
    try:
        jax_devices = jax.device_count()
        jax_backend = jax.default_backend()
    except Exception as exc:  # pragma: no cover - defensive around runtime initialization
        jax_devices = "unknown"
        jax_backend = f"unknown:{type(exc).__name__}"
    _log(
        args,
        (
            f"[runtime] python={sys.executable} jax_devices={jax_devices} backend={jax_backend} "
            f"output_dir={args.output_dir}"
        ),
    )
    _log(
        args,
        (
            f"[runtime] par_path={args.par_path} fit_mode={args.fit_mode} fit_method={args.fit_method} "
            f"sampling_engine={args.sampling_engine} sample_likelihood_mode={getattr(args, 'sample_likelihood_mode', SAMPLE_LIKELIHOOD_SOURCE)} "
            f"image_plane_mode={getattr(args, 'image_plane_mode', IMAGE_PLANE_MODE_NONE)} warmup={args.warmup} samples={args.samples} "
            f"source_position_parameterization={getattr(args, 'source_position_parameterization', SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED)} "
            f"evidence_likelihood_mode={getattr(args, 'evidence_likelihood_mode', DEFAULT_EVIDENCE_LIKELIHOOD_MODE)} "
            f"evidence_source_prior_sigma_arcsec={getattr(args, 'evidence_source_prior_sigma_arcsec', None)} "
            f"evidence_source_prior_mean=({getattr(args, 'evidence_source_prior_mean_x_arcsec', 0.0)},"
            f"{getattr(args, 'evidence_source_prior_mean_y_arcsec', 0.0)}) "
            f"fit_cosmology_flat_wcdm={bool(getattr(args, 'fit_cosmology_flat_wcdm', False))} "
            f"chains={args.chains} thin={args.thin} skip_validation={args.skip_validation} "
            f"ns_num_live_points={getattr(args, 'ns_num_live_points', None)} "
            f"ns_max_samples={getattr(args, 'ns_max_samples', DEFAULT_NS_MAX_SAMPLES)} "
            f"ns_posterior_samples={DEFAULT_NS_POSTERIOR_SAMPLES} "
            f"ns_dlogz={getattr(args, 'ns_dlogz', DEFAULT_NS_DLOGZ)} "
            f"image_presence_penalty_weight={getattr(args, 'image_presence_penalty_weight', None)} "
            f"image_presence_match_radius_arcsec={getattr(args, 'image_presence_match_radius_arcsec', DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC)} "
            f"image_presence_temperature_arcsec={getattr(args, 'image_presence_temperature_arcsec', DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)} "
            f"skip_plots={args.skip_plots} quick_diagnostics={bool(getattr(args, 'quick_diagnostics', False))} "
            f"plots_only={args.plots_only}"
        ),
    )


def _log_state_summary(args: argparse.Namespace, state: BuildState) -> None:
    n_images = int(sum(family.n_images for family in state.family_data))
    n_scaling_components = int(len(state.scaling_component_records))
    n_large_components = int(max(0, len(state.base_components) - n_scaling_components))
    z_values = [float(family.z_source) for family in state.family_data]
    z_range = f"{min(z_values):.4g}-{max(z_values):.4g}" if z_values else "none"
    potfile_rows = {
        str(potfile.get("id", f"potfile{idx}")): int(len(potfile.get("catalog_df", [])))
        for idx, potfile in enumerate(state.potfiles)
    }
    parameter_families = Counter(str(spec.component_family) for spec in state.parameter_specs)
    prior_kinds = Counter(str(spec.prior_kind) for spec in state.parameter_specs)
    transform_kinds = Counter(str(spec.transform_kind) for spec in state.parameter_specs)
    positive_transforms = sum(
        1 for spec in state.parameter_specs if str(spec.transform_kind) in {"log_positive", "log_offset_positive"}
    )
    _log(
        args,
        (
            f"[input] z_lens={state.z_lens:.5g} sigma_arcsec={state.sigma_arcsec:.4g} "
            f"cosmology={state.cosmo_config.get('class', 'unknown')} "
            f"H0={float(state.cosmo_config.get('H0', float('nan'))):.4g} "
            f"Om0={float(state.cosmo_config.get('Om0', float('nan'))):.4g} "
            f"w0={float(state.cosmo_config.get('w0', -1.0)):.4g} "
            f"fit_cosmology_flat_wcdm={bool(getattr(state, 'fit_cosmology_flat_wcdm', False))}"
        ),
    )
    _log(
        args,
        (
            f"[model] fit_mode={state.fit_mode} lens_profiles={sorted(set(state.lens_model_list))} "
            f"large_components={n_large_components} scaling_components={n_scaling_components} "
            f"potfiles={len(state.potfiles)} families={len(state.family_data)} images={n_images} "
            f"z_bins={len(state.bin_data)} source_z_range={z_range}"
        ),
    )
    _log(
        args,
        (
            f"[parameters] total={len(state.parameter_specs)} by_family={_format_count_map(parameter_families)} "
            f"priors={_format_count_map(prior_kinds)} transforms={_format_count_map(transform_kinds)} "
            f"positive={positive_transforms}"
        ),
    )
    if potfile_rows:
        _log(args, f"[input] potfile_catalog_rows={json.dumps(potfile_rows, sort_keys=True)}")


def _count_items(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(len(value))
    except TypeError:
        return 1


def _solver_active_approximation_items(evaluator: Any) -> list[str]:
    state = evaluator.state
    items: list[str] = []
    family_count = _count_items(getattr(state, "family_data", []))
    bin_count = _count_items(getattr(state, "bin_data", []))
    if bool(getattr(evaluator, "surrogate_enabled", False)):
        items.append(
            "refreshing_surrogate=active "
            f"inactive_scaling={_count_items(getattr(evaluator, 'inactive_scaling_component_indices', []))}"
        )
    if family_count > 0 and bin_count < family_count:
        items.append(f"z_bins=active grouped_families={family_count} bins={bin_count}")
    validation_approx = str(getattr(evaluator, "validation_approx", "exact"))
    if validation_approx != "exact":
        items.append(f"validation_approx={validation_approx} exact image validation may be skipped")
    if bool(getattr(evaluator, "quick_diagnostics", False)):
        items.append("quick_diagnostics=active exact post-fit image-position diagnostics skipped")

    sample_likelihood_mode = str(getattr(evaluator, "sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE))
    if sample_likelihood_mode == SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN:
        items.append("sample_likelihood=local-jacobian local image-plane covariance")
    elif sample_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE:
        items.append("sample_likelihood=linearized-forward-beta-image-plane local linear image correction")
        image_presence_weight = float(getattr(evaluator, "image_presence_penalty_weight", 0.0))
        if image_presence_weight > 0.0:
            items.append(f"image_presence_penalty=active weight={image_presence_weight:.4g}")
    elif sample_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE:
        items.append("sample_likelihood=linearized-marginal-beta-image-plane analytic linearized source marginalization")

    source_position_parameterization = str(
        getattr(evaluator, "source_position_parameterization", SOURCE_POSITION_PARAMETERIZATION_DIRECT)
    )
    source_position_count = sum(
        str(getattr(spec, "component_family", "")) == "source_position"
        for spec in getattr(state, "parameter_specs", [])
    )
    if source_position_count > 0 and source_position_parameterization != SOURCE_POSITION_PARAMETERIZATION_DIRECT:
        items.append(f"source_position_parameterization={source_position_parameterization}")

    inactive_scaling_count = _count_items(getattr(evaluator, "inactive_scaling_component_indices", []))
    if bool(getattr(evaluator, "surrogate_enabled", False)) and inactive_scaling_count > 0:
        items.append(
            "active_scaling_subset=active "
            f"{_count_items(getattr(evaluator, 'active_scaling_component_indices', []))}/"
            f"{_count_items(getattr(evaluator, 'scaling_component_indices', []))}"
        )
    if (
        _count_items(getattr(evaluator, "scaling_scatter_param_indices", [])) > 0
        and _count_items(getattr(evaluator, "scaling_component_indices", [])) > 0
    ):
        items.append("scaling_scatter_cache=linearized scaling-scatter covariance")
    if _count_items(getattr(evaluator, "source_metric_cache_by_z", {})) > 0:
        items.append("source_metric_cache=refreshed local lensing metric")
    return items


def _log_solver_active_approximation_warning(args: argparse.Namespace | None, evaluator: Any) -> None:
    items = _solver_active_approximation_items(evaluator)
    if items:
        _log(args, "[validation] warning approximations active: " + "; ".join(items))


def _log_evaluator_summary(args: argparse.Namespace, evaluator: Any) -> None:
    state = evaluator.state
    _log(
        args,
        (
            f"[source-position] parameterization={getattr(evaluator, 'source_position_parameterization', 'direct')} "
            f"n_parameters={sum(spec.component_family == 'source_position' for spec in state.parameter_specs)}"
        ),
    )
    if str(getattr(evaluator, "sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE)) == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE:
        _log(
            args,
            (
                "[image-presence] "
                f"weight={float(getattr(evaluator, 'image_presence_penalty_weight', 0.0)):.4g} "
                f"match_radius_arcsec={float(getattr(evaluator, 'image_presence_match_radius_arcsec', DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC)):.4g} "
                f"temperature_arcsec={float(getattr(evaluator, 'image_presence_temperature_arcsec', DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)):.4g} "
                f"count_softness={float(getattr(evaluator, 'image_presence_count_softness', DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS)):.4g} "
                f"count_margin={float(getattr(evaluator, 'image_presence_count_margin', DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN)):.4g}"
            ),
        )
    if not state.potfiles:
        _log(args, "[surrogate] no potfiles detected; scaling surrogate inactive")
        return
    _log(
        args,
        (
            f"[surrogate] engine={evaluator.sampling_engine} enabled={evaluator.surrogate_enabled} "
            f"selection={evaluator.active_scaling_selection} "
            f"active={len(evaluator.active_scaling_component_indices)} "
            f"inactive={len(evaluator.inactive_scaling_component_indices)} "
            f"total_scaling={len(evaluator.scaling_component_indices)}"
        ),
    )
    _log(
        args,
        (
            "[surrogate] active_by_potfile "
            f"requested={json.dumps(evaluator.requested_active_scaling_by_potfile, sort_keys=True)} "
            f"actual={json.dumps(evaluator.actual_active_scaling_by_potfile, sort_keys=True)} "
            f"total={json.dumps(evaluator.total_scaling_by_potfile, sort_keys=True)}"
        ),
    )


def _finite_range(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return "none"
    return f"{float(np.min(finite)):.4g}-{float(np.max(finite)):.4g}"


def _log_posterior_summary(args: argparse.Namespace, label: str, posterior: PosteriorResults) -> None:
    samples = np.asarray(posterior.samples, dtype=float)
    n_draws = int(samples.shape[0]) if samples.ndim >= 1 else 0
    n_params = int(samples.shape[-1]) if samples.ndim >= 2 else 0
    accept_prob = np.asarray(posterior.accept_prob, dtype=float)
    finite_accept = accept_prob[np.isfinite(accept_prob)]
    accept_text = f"{float(np.mean(finite_accept)):.3f}" if finite_accept.size else "na"
    diverging = np.asarray(posterior.diverging, dtype=bool)
    divergence_text = str(int(np.sum(diverging))) if diverging.size else "na"
    num_steps = np.asarray(posterior.num_steps, dtype=float)
    finite_steps = num_steps[np.isfinite(num_steps)]
    steps_text = f"{float(np.mean(finite_steps)):.2f}" if finite_steps.size else "na"
    max_steps = float(2 ** int(getattr(args, "max_tree_depth", DEFAULT_MAX_TREE_DEPTH)) - 1)
    saturation_text = f"{float(np.mean(finite_steps >= max_steps)):.3f}" if finite_steps.size else "na"
    _log(
        args,
        (
            f"[posterior] label={label} sampler={posterior.sampler} draws={n_draws} params={n_params} "
            f"chains={posterior.num_chains} warmup={posterior.warmup_steps} sample_steps={posterior.sample_steps} "
            f"log_prob_range={_finite_range(posterior.log_prob)} accept_mean={accept_text} "
            f"divergences={divergence_text} mean_steps={steps_text} max_tree_saturation={saturation_text}"
        ),
    )


def _basic_rhat(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        return float("nan")
    n_draws = int(array.shape[1])
    chain_means = np.nanmean(array, axis=1)
    within = float(np.nanmean(np.nanvar(array, axis=1, ddof=1)))
    between = float(n_draws * np.nanvar(chain_means, ddof=1))
    if not np.isfinite(within) or within <= 0.0:
        return float("inf") if np.isfinite(between) and between > 0.0 else 1.0
    var_hat = ((n_draws - 1.0) / n_draws) * within + between / n_draws
    return float(np.sqrt(var_hat / within))


def _sampling_chain_diagnostics(
    grouped_samples: np.ndarray | None,
    parameter_specs: list[ParameterSpec],
) -> dict[str, Any]:
    grouped = np.asarray(grouped_samples, dtype=float) if grouped_samples is not None else np.asarray([])
    if grouped.ndim != 3 or grouped.shape[0] < 2 or grouped.shape[1] < 2 or grouped.shape[2] == 0:
        return {}
    try:
        from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    except Exception:
        effective_sample_size = None
        split_gelman_rubin = None
    ess_values: list[tuple[str, float]] = []
    rhat_values: list[tuple[str, float]] = []
    for idx in range(grouped.shape[2]):
        values = grouped[:, :, idx]
        if not np.isfinite(values).all():
            continue
        name = parameter_specs[idx].name if idx < len(parameter_specs) else f"param_{idx}"
        if effective_sample_size is not None:
            try:
                ess = float(np.asarray(effective_sample_size(values)).reshape(-1)[0])
            except Exception:
                ess = float("nan")
            if np.isfinite(ess):
                ess_values.append((name, ess))
        rhat = float("nan")
        if split_gelman_rubin is not None:
            try:
                rhat = float(np.asarray(split_gelman_rubin(values)).reshape(-1)[0])
            except Exception:
                rhat = float("nan")
        if not np.isfinite(rhat):
            rhat = _basic_rhat(values)
        if np.isfinite(rhat) or np.isposinf(rhat):
            rhat_values.append((name, rhat))

    diagnostics: dict[str, Any] = {}
    if ess_values:
        worst_name, worst_value = min(ess_values, key=lambda item: item[1])
        diagnostics["ess_min"] = float(worst_value)
        diagnostics["ess_worst_parameter"] = worst_name
    if rhat_values:
        worst_name, worst_value = max(rhat_values, key=lambda item: item[1])
        diagnostics["rhat_max"] = float(worst_value) if np.isfinite(worst_value) else "inf"
        diagnostics["rhat_worst_parameter"] = worst_name
    return diagnostics


def _nuts_quality_diagnostics(
    args: argparse.Namespace,
    posterior: PosteriorResults,
    parameter_specs: list[ParameterSpec],
) -> tuple[dict[str, Any], list[str]]:
    metrics: dict[str, Any] = {}
    warnings: list[str] = []
    if str(posterior.sampler) != "numpyro_nuts":
        return metrics, warnings

    num_steps = np.asarray(posterior.num_steps, dtype=float)
    finite_steps = num_steps[np.isfinite(num_steps)]
    if finite_steps.size:
        max_tree_depth = int(getattr(args, "max_tree_depth", DEFAULT_MAX_TREE_DEPTH))
        max_steps = float(2**max_tree_depth - 1)
        saturation = float(np.mean(finite_steps >= max_steps))
        metrics["max_tree_depth_saturation_fraction"] = saturation
        metrics["max_tree_depth_saturation_warning_threshold"] = NUTS_MAX_TREE_SATURATION_WARNING
        if saturation >= NUTS_MAX_TREE_SATURATION_WARNING:
            warnings.append(
                "max-tree-depth saturation "
                f"{saturation:.3f} >= {NUTS_MAX_TREE_SATURATION_WARNING:.3f} "
                f"(max_tree_depth={max_tree_depth})"
            )

    chain_metrics = _sampling_chain_diagnostics(posterior.grouped_samples, parameter_specs)
    metrics.update(chain_metrics)
    ess_min = chain_metrics.get("ess_min")
    if ess_min is not None:
        ess_threshold = float(NUTS_MIN_ESS_PER_CHAIN_WARNING * max(int(posterior.num_chains), 1))
        metrics["ess_min_warning_threshold"] = ess_threshold
        if float(ess_min) <= ess_threshold:
            warnings.append(
                "minimum ESS "
                f"{float(ess_min):.3g} <= {ess_threshold:.3g} "
                f"({chain_metrics.get('ess_worst_parameter', 'unknown')})"
            )
    rhat_max = chain_metrics.get("rhat_max")
    if rhat_max is not None:
        metrics["rhat_max_warning_threshold"] = NUTS_RHAT_EXTREME_WARNING
        rhat_value = float("inf") if rhat_max == "inf" else float(rhat_max)
        if not np.isfinite(rhat_value) or rhat_value >= NUTS_RHAT_EXTREME_WARNING:
            rhat_text = "inf" if not np.isfinite(rhat_value) else f"{rhat_value:.3g}"
            warnings.append(
                "extreme Rhat "
                f"{rhat_text} >= {NUTS_RHAT_EXTREME_WARNING:.3g} "
                f"({chain_metrics.get('rhat_worst_parameter', 'unknown')})"
            )
    return metrics, warnings


def _apply_nuts_quality_gate(
    args: argparse.Namespace,
    posterior: PosteriorResults,
    parameter_specs: list[ParameterSpec],
) -> None:
    metrics, warnings = _nuts_quality_diagnostics(args, posterior, parameter_specs)
    if not metrics and not warnings:
        return
    if posterior.init_diagnostics is None:
        posterior.init_diagnostics = {}
    posterior.init_diagnostics["nuts_quality_metrics"] = metrics
    posterior.init_diagnostics["nuts_quality_warnings"] = warnings
    for warning in warnings:
        _log(args, f"[nuts:quality] warning {warning}")


def _write_truth_validation_outputs(args: argparse.Namespace, run_dir: Path) -> None:
    truth_path = getattr(args, "truth", None)
    if truth_path is None:
        return
    from .validation import write_recovery_outputs

    truth_path = Path(truth_path)
    output_dir = Path(run_dir) / "validation"
    _log(args, f"[validation] writing truth recovery plots to {output_dir}")
    paths = _run_logged_phase(
        args,
        "truth_validation.write_recovery_outputs",
        lambda: write_recovery_outputs(
            run_dir,
            truth_path,
            mock_images_path=None,
            output_dir=output_dir,
            quick_diagnostics=bool(getattr(args, "quick_diagnostics", False)),
        ),
    )
    if "mass_profile_plot" not in paths:
        _log(args, "[validation] skipped mass_profile_recovery.pdf; truth file lacks mass-profile lens fields")
    if "surface_density_plot" not in paths:
        _log(args, "[validation] skipped surface_density_recovery.pdf; truth file lacks mass-profile lens fields")
    _log(args, f"[validation] truth recovery plots complete files={len(paths)}")


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
        if len(unconstrained_payloads) == 1:
            value = np.asarray(unconstrained_payloads[0][spec.sample_name], dtype=float)
        else:
            value = np.stack([item[spec.sample_name] for item in unconstrained_payloads], axis=0)
        payload[spec.sample_name] = jnp.asarray(value, dtype=jnp.float64)
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


def _local_jacobian_bin_loglike(
    beta_x: jnp.ndarray,
    beta_y: jnp.ndarray,
    family_idx: jnp.ndarray,
    n_families: int,
    sigma_per_image: jnp.ndarray,
    reliability_per_image: jnp.ndarray,
    image_has_constraint: jnp.ndarray,
    source_sigma_int: jnp.ndarray,
    scatter_var_x: jnp.ndarray,
    scatter_var_y: jnp.ndarray,
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
    covariance_floor: float,
    outlier_sigma_arcsec: float,
) -> jnp.ndarray:
    sigma2_img = jnp.square(sigma_per_image)
    source_var = jnp.square(source_sigma_int)
    cov_floor = jnp.asarray(covariance_floor, dtype=jnp.float64)
    c00 = sigma2_img * (jnp.square(jac_a00) + jnp.square(jac_a01)) + source_var + scatter_var_x + cov_floor
    c11 = sigma2_img * (jnp.square(jac_a10) + jnp.square(jac_a11)) + source_var + scatter_var_y + cov_floor
    c01 = sigma2_img * (jac_a00 * jac_a10 + jac_a01 * jac_a11)
    c00 = jnp.maximum(c00, cov_floor)
    c11 = jnp.maximum(c11, cov_floor)
    det = jnp.maximum(c00 * c11 - jnp.square(c01), jnp.square(jnp.maximum(cov_floor, 1.0e-12)))
    inv00 = c11 / det
    inv11 = c00 / det
    inv01 = -c01 / det

    reliability = jnp.clip(reliability_per_image, 1.0e-6, 1.0 - 1.0e-6)
    w_inv00 = reliability * inv00
    w_inv01 = reliability * inv01
    w_inv11 = reliability * inv11
    sum00 = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(w_inv00)
    sum01 = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(w_inv01)
    sum11 = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(w_inv11)
    rhs_x = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(w_inv00 * beta_x + w_inv01 * beta_y)
    rhs_y = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(w_inv01 * beta_x + w_inv11 * beta_y)
    centroid_det = jnp.maximum(sum00 * sum11 - jnp.square(sum01), 1.0e-18)
    centroid_x = (sum11 * rhs_x - sum01 * rhs_y) / centroid_det
    centroid_y = (-sum01 * rhs_x + sum00 * rhs_y) / centroid_det

    dx = beta_x - centroid_x[family_idx]
    dy = beta_y - centroid_y[family_idx]
    quad = dx * (inv00 * dx + inv01 * dy) + dy * (inv01 * dx + inv11 * dy)
    family_ll = -0.5 * (quad + jnp.log(jnp.square(2.0 * jnp.pi) * det))

    outlier_sigma2 = jnp.square(jnp.asarray(outlier_sigma_arcsec, dtype=jnp.float64))
    outlier_ll = -0.5 * (
        (jnp.square(dx) + jnp.square(dy)) / outlier_sigma2 + 2.0 * jnp.log(2.0 * jnp.pi * outlier_sigma2)
    )
    mixture_ll = jnp.logaddexp(jnp.log(reliability) + family_ll, jnp.log1p(-reliability) + outlier_ll)
    bin_loglike = jnp.sum(jnp.where(image_has_constraint, mixture_ll, 0.0))
    finite = (
        jnp.all(jnp.isfinite(c00))
        & jnp.all(jnp.isfinite(c01))
        & jnp.all(jnp.isfinite(c11))
        & jnp.all(jnp.isfinite(det))
        & jnp.all(jnp.isfinite(quad))
        & jnp.isfinite(bin_loglike)
    )
    return jnp.where(finite, bin_loglike, jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64))


def _linearized_image_plane_residual_from_jacobian(
    f_x: jnp.ndarray,
    f_y: jnp.ndarray,
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
    *,
    determinant_floor: float = 1.0e-12,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    det = jac_a00 * jac_a11 - jac_a01 * jac_a10
    floor = jnp.asarray(determinant_floor, dtype=jnp.float64)
    det_abs = jnp.abs(det)
    det_safe = jnp.where(det_abs >= floor, det, jnp.where(det >= 0.0, floor, -floor))
    delta_x = -(jac_a11 * f_x - jac_a01 * f_y) / det_safe
    delta_y = -(-jac_a10 * f_x + jac_a00 * f_y) / det_safe
    finite = (
        jnp.isfinite(f_x)
        & jnp.isfinite(f_y)
        & jnp.isfinite(jac_a00)
        & jnp.isfinite(jac_a01)
        & jnp.isfinite(jac_a10)
        & jnp.isfinite(jac_a11)
        & jnp.isfinite(delta_x)
        & jnp.isfinite(delta_y)
        & (det_abs >= floor)
    )
    return delta_x, delta_y, finite


def _soft_observed_image_presence_loglike(
    residual_x: jnp.ndarray,
    residual_y: jnp.ndarray,
    family_idx: jnp.ndarray,
    n_families: int,
    reliability_per_image: jnp.ndarray,
    image_has_constraint: jnp.ndarray,
    *,
    penalty_weight: float,
    match_radius_arcsec: float,
    temperature_arcsec: float,
    count_softness: float,
    count_margin: float,
) -> jnp.ndarray:
    if float(penalty_weight) <= 0.0:
        return jnp.asarray(0.0, dtype=jnp.float64)

    image_mask = jnp.asarray(image_has_constraint, dtype=bool)
    reliability = jnp.where(image_mask, jnp.clip(reliability_per_image, 0.0, 1.0), 0.0)
    residual2 = jnp.square(residual_x) + jnp.square(residual_y)
    radius2 = jnp.square(jnp.asarray(match_radius_arcsec, dtype=jnp.float64))
    temperature2 = jnp.maximum(
        jnp.square(jnp.asarray(temperature_arcsec, dtype=jnp.float64)),
        jnp.asarray(1.0e-18, dtype=jnp.float64),
    )
    presence_probability = jax.nn.sigmoid((radius2 - residual2) / temperature2)

    zeros = jnp.zeros(int(n_families), dtype=jnp.float64)
    target_count = zeros.at[family_idx].add(reliability)
    found_count = zeros.at[family_idx].add(reliability * presence_probability)
    softness = jnp.maximum(
        jnp.asarray(count_softness, dtype=jnp.float64),
        jnp.asarray(1.0e-12, dtype=jnp.float64),
    )
    margin = jnp.asarray(count_margin, dtype=jnp.float64)
    shortfall = softness * jax.nn.softplus((target_count - found_count - margin) / softness)
    shortfall = jnp.where(target_count > 0.0, shortfall, 0.0)
    penalty = -jnp.asarray(penalty_weight, dtype=jnp.float64) * jnp.sum(jnp.square(shortfall))
    finite = (
        jnp.all(jnp.isfinite(residual_x))
        & jnp.all(jnp.isfinite(residual_y))
        & jnp.all(jnp.isfinite(target_count))
        & jnp.all(jnp.isfinite(found_count))
        & jnp.isfinite(penalty)
    )
    return jnp.where(finite, penalty, jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64))


def _linearized_image_plane_bin_loglike(
    residual_x: jnp.ndarray,
    residual_y: jnp.ndarray,
    family_idx: jnp.ndarray | None,
    n_families: int | None,
    sigma_per_image: jnp.ndarray,
    reliability_per_image: jnp.ndarray,
    image_has_constraint: jnp.ndarray,
    image_sigma_int: jnp.ndarray,
    covariance_floor: float,
    outlier_sigma_arcsec: float,
    image_presence_penalty_weight: float = 0.0,
    image_presence_match_radius_arcsec: float = DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC,
    image_presence_temperature_arcsec: float = DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC,
    image_presence_count_softness: float = DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS,
    image_presence_count_margin: float = DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN,
) -> jnp.ndarray:
    cov_floor = jnp.asarray(covariance_floor, dtype=jnp.float64)
    sigma2 = jnp.maximum(
        jnp.square(sigma_per_image) + jnp.square(image_sigma_int) + cov_floor,
        jnp.maximum(cov_floor, 1.0e-18),
    )
    reliability = jnp.clip(reliability_per_image, 1.0e-6, 1.0 - 1.0e-6)
    quad = (jnp.square(residual_x) + jnp.square(residual_y)) / sigma2
    family_ll = -0.5 * (quad + 2.0 * jnp.log(2.0 * jnp.pi * sigma2))
    outlier_sigma2 = jnp.square(jnp.asarray(outlier_sigma_arcsec, dtype=jnp.float64))
    outlier_ll = -0.5 * (
        (jnp.square(residual_x) + jnp.square(residual_y)) / outlier_sigma2
        + 2.0 * jnp.log(2.0 * jnp.pi * outlier_sigma2)
    )
    mixture_ll = jnp.logaddexp(jnp.log(reliability) + family_ll, jnp.log1p(-reliability) + outlier_ll)
    bin_loglike = jnp.sum(jnp.where(image_has_constraint, mixture_ll, 0.0))
    if (
        float(image_presence_penalty_weight) > 0.0
        and family_idx is not None
        and n_families is not None
    ):
        bin_loglike = bin_loglike + _soft_observed_image_presence_loglike(
            residual_x=residual_x,
            residual_y=residual_y,
            family_idx=family_idx,
            n_families=int(n_families),
            reliability_per_image=reliability,
            image_has_constraint=image_has_constraint,
            penalty_weight=float(image_presence_penalty_weight),
            match_radius_arcsec=float(image_presence_match_radius_arcsec),
            temperature_arcsec=float(image_presence_temperature_arcsec),
            count_softness=float(image_presence_count_softness),
            count_margin=float(image_presence_count_margin),
        )
    finite = (
        jnp.all(jnp.isfinite(residual_x))
        & jnp.all(jnp.isfinite(residual_y))
        & jnp.all(jnp.isfinite(sigma2))
        & jnp.isfinite(bin_loglike)
    )
    return jnp.where(finite, bin_loglike, jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64))


def _linearized_marginal_beta_image_plane_bin_loglike(
    beta_x: jnp.ndarray,
    beta_y: jnp.ndarray,
    jacobian_entries: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    family_idx: jnp.ndarray,
    n_families: int,
    sigma_per_image: jnp.ndarray,
    image_has_constraint: jnp.ndarray,
    image_sigma_int: jnp.ndarray,
    covariance_floor: float,
    source_prior_mean_x: float,
    source_prior_mean_y: float,
    source_prior_sigma_arcsec: float,
) -> jnp.ndarray:
    jac_a00, jac_a01, jac_a10, jac_a11 = jacobian_entries
    det = jac_a00 * jac_a11 - jac_a01 * jac_a10
    det_safe = _safe_signed_min_abs(det, 1.0e-10)
    inv00 = jac_a11 / det_safe
    inv01 = -jac_a01 / det_safe
    inv10 = -jac_a10 / det_safe
    inv11 = jac_a00 / det_safe

    cov_floor = jnp.asarray(covariance_floor, dtype=jnp.float64)
    sigma2 = jnp.maximum(
        jnp.square(sigma_per_image) + jnp.square(image_sigma_int) + cov_floor,
        jnp.maximum(cov_floor, 1.0e-18),
    )
    image_mask = jnp.asarray(image_has_constraint, dtype=bool)
    weight = jnp.where(image_mask, 1.0 / sigma2, 0.0)
    lambda00 = weight * (jnp.square(inv00) + jnp.square(inv10))
    lambda01 = weight * (inv00 * inv01 + inv10 * inv11)
    lambda11 = weight * (jnp.square(inv01) + jnp.square(inv11))
    zeros = jnp.zeros(int(n_families), dtype=jnp.float64)
    sum_lambda00 = zeros.at[family_idx].add(lambda00)
    sum_lambda01 = zeros.at[family_idx].add(lambda01)
    sum_lambda11 = zeros.at[family_idx].add(lambda11)
    rhs_x = zeros.at[family_idx].add(lambda00 * beta_x + lambda01 * beta_y)
    rhs_y = zeros.at[family_idx].add(lambda01 * beta_x + lambda11 * beta_y)
    log_norm_images = zeros.at[family_idx].add(jnp.where(image_mask, 2.0 * jnp.log(2.0 * jnp.pi * sigma2), 0.0))
    n_images = jnp.zeros(int(n_families), dtype=jnp.int32).at[family_idx].add(image_mask.astype(jnp.int32))

    prior_sigma = jnp.asarray(source_prior_sigma_arcsec, dtype=jnp.float64)
    prior_precision = 1.0 / jnp.square(prior_sigma)
    mu_x = jnp.asarray(source_prior_mean_x, dtype=jnp.float64)
    mu_y = jnp.asarray(source_prior_mean_y, dtype=jnp.float64)
    p00 = prior_precision + sum_lambda00
    p01 = sum_lambda01
    p11 = prior_precision + sum_lambda11
    precision_det = jnp.maximum(p00 * p11 - jnp.square(p01), 1.0e-18)
    rhs_x = prior_precision * mu_x + rhs_x
    rhs_y = prior_precision * mu_y + rhs_y
    posterior_mean_x = (p11 * rhs_x - p01 * rhs_y) / precision_det
    posterior_mean_y = (-p01 * rhs_x + p00 * rhs_y) / precision_det
    mean_x_per_image = posterior_mean_x[family_idx]
    mean_y_per_image = posterior_mean_y[family_idx]
    delta_x = beta_x - mean_x_per_image
    delta_y = beta_y - mean_y_per_image
    image_quad = zeros.at[family_idx].add(
        jnp.where(
            image_mask,
            lambda00 * jnp.square(delta_x) + 2.0 * lambda01 * delta_x * delta_y + lambda11 * jnp.square(delta_y),
            0.0,
        )
    )
    prior_quad = prior_precision * (
        jnp.square(posterior_mean_x - mu_x) + jnp.square(posterior_mean_y - mu_y)
    )
    minimized_quad = jnp.maximum(prior_quad + image_quad, 0.0)
    prior_logdet = 2.0 * jnp.log(jnp.square(prior_sigma))
    family_loglike = -0.5 * (
        minimized_quad + prior_logdet + log_norm_images + jnp.log(precision_det)
    )
    family_loglike = jnp.where(n_images > 0, family_loglike, 0.0)
    bin_loglike = jnp.sum(family_loglike)
    finite = (
        jnp.all(jnp.isfinite(beta_x))
        & jnp.all(jnp.isfinite(beta_y))
        & jnp.all(jnp.isfinite(det))
        & jnp.all(jnp.isfinite(sigma2))
        & jnp.all(jnp.isfinite(family_loglike))
        & jnp.isfinite(bin_loglike)
    )
    return jnp.where(finite, bin_loglike, jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64))


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
    jitter_frac: float = DEFAULT_NUTS_INIT_JITTER_FRAC,
    boundary_frac: float = DEFAULT_NUTS_INIT_BOUNDARY_FRAC,
) -> np.ndarray:
    perturbed = np.asarray(theta, dtype=float).copy()
    if float(jitter_frac) <= 0.0:
        return _clip_theta_to_support(perturbed, parameter_specs, boundary_frac=boundary_frac)
    for idx, spec in enumerate(parameter_specs):
        if spec.prior_kind == "normal":
            scale = max(float(jitter_frac) * float(spec.std or 0.0), 1.0e-6)
        else:
            scale = max(float(jitter_frac) * float(spec.upper - spec.lower), 1.0e-6)
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
            jitter_frac=float(getattr(args, "nuts_init_jitter_frac", DEFAULT_NUTS_INIT_JITTER_FRAC)),
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
    *,
    batch_size: int = DEFAULT_POSTERIOR_LOGPROB_BATCH_SIZE,
) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] == 0:
        return np.empty((0,), dtype=float)

    def logprob_one(theta: jnp.ndarray) -> jnp.ndarray:
        total = evaluator._source_loglike_fn(theta)
        return total + _prior_log_prob(parameter_specs, theta)

    chunk_size = max(1, int(batch_size))
    if sample_array.shape[0] <= chunk_size:
        return np.asarray(jax.vmap(logprob_one)(jnp.asarray(sample_array, dtype=jnp.float64)), dtype=float)
    chunks: list[np.ndarray] = []
    for start in range(0, sample_array.shape[0], chunk_size):
        chunk = sample_array[start : start + chunk_size]
        chunks.append(np.asarray(jax.vmap(logprob_one)(jnp.asarray(chunk, dtype=jnp.float64)), dtype=float))
    return np.concatenate(chunks, axis=0)


def _source_loglike_matrix(
    evaluator: Any,
    samples: np.ndarray,
    *,
    batch_size: int = DEFAULT_POSTERIOR_LOGPROB_BATCH_SIZE,
) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] == 0 or not hasattr(evaluator, "_source_loglike_fn"):
        return np.empty((0,), dtype=float)

    def loglike_one(theta: jnp.ndarray) -> jnp.ndarray:
        return evaluator._source_loglike_fn(theta)

    chunk_size = max(1, int(batch_size))
    if sample_array.shape[0] <= chunk_size:
        return np.asarray(jax.vmap(loglike_one)(jnp.asarray(sample_array, dtype=jnp.float64)), dtype=float)
    chunks: list[np.ndarray] = []
    for start in range(0, sample_array.shape[0], chunk_size):
        chunk = sample_array[start : start + chunk_size]
        chunks.append(np.asarray(jax.vmap(loglike_one)(jnp.asarray(chunk, dtype=jnp.float64)), dtype=float))
    return np.concatenate(chunks, axis=0)


def _max_likelihood_best_fit_from_posterior(
    args: argparse.Namespace,
    evaluator: Any,
    posterior: PosteriorResults,
    fallback_best_fit: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(posterior.samples, dtype=float)
    source_loglike = _source_loglike_matrix(evaluator, samples)
    if source_loglike.shape[0] == samples.shape[0] and np.isfinite(source_loglike).any():
        finite_loglike = np.where(np.isfinite(source_loglike), source_loglike, -np.inf)
        best_index = int(np.argmax(finite_loglike))
        best_fit = np.asarray(samples[best_index], dtype=float)
        _log(
            args,
            (
                "[fit] best_fit updated from retained posterior sample with max source likelihood "
                f"index={best_index} source_loglike={float(source_loglike[best_index]):.6g}"
            ),
        )
        return best_fit
    _log(args, "[fit] no finite posterior source likelihoods; retaining existing best_fit")
    return np.asarray(fallback_best_fit, dtype=float)


def _reference_theta_from_init_values(
    parameter_specs: list[ParameterSpec],
    init_values: dict[str, float] | None,
    fallback: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(fallback, dtype=float).copy()
    if init_values:
        for idx, spec in enumerate(parameter_specs):
            if spec.sample_name in init_values:
                reference[idx] = float(init_values[spec.sample_name])
    return _clip_theta_to_support(reference, parameter_specs, boundary_frac=DEFAULT_NUTS_INIT_BOUNDARY_FRAC)


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
            jitter_frac=float(getattr(args, "nuts_init_jitter_frac", DEFAULT_NUTS_INIT_JITTER_FRAC)),
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
    total_steps = int(args.svi_steps)
    refresh_every = max(1, int(getattr(args, "refresh_every", DEFAULT_REFRESH_EVERY)))
    use_blocked_refresh = total_steps > refresh_every
    _log(
        args,
        (
            f"[svi] starting steps={total_steps} lr={float(args.svi_learning_rate):.3g} "
            f"init_values={bool(state.svi_init_values)} blocked_refresh={use_blocked_refresh} "
            f"refresh_every={refresh_every}"
        ),
    )
    svi_start = time.time()
    block_losses: list[np.ndarray] = []
    block_center_shift: list[float] = []
    block_steps: list[int] = []
    block_refresh_count = 0
    init_values = dict(state.svi_init_values or {})
    guide = None
    svi = None
    svi_result = None
    params = None
    center_theta_previous: np.ndarray | None = None
    remaining_steps = total_steps
    block_index = 0
    while remaining_steps > 0:
        block_index += 1
        block_steps_current = min(refresh_every, remaining_steps) if use_blocked_refresh else remaining_steps
        block_steps.append(int(block_steps_current))
        guide = _make_auto_normal_guide(sample_model, state.parameter_specs, init_values)
        svi = SVI(sample_model, guide, numpyro_optim.Adam(float(args.svi_learning_rate)), Trace_ELBO())
        phase_name = "svi.run" if not use_blocked_refresh else f"svi.run.block_{block_index}"
        svi_result = _run_logged_phase(
            args,
            phase_name,
            lambda block_steps_current=block_steps_current, block_index=block_index, svi=svi: svi.run(
                jax.random.PRNGKey(0 if args.seed is None else int(args.seed) + 202 + block_index),
                int(block_steps_current),
                progress_bar=True,
            ),
        )
        params = svi.get_params(svi_result.state)
        block_losses.append(np.asarray(svi_result.losses, dtype=float))
        median_values = guide.median(params)
        center_theta = _clip_theta_to_support(
            _values_dict_to_theta(state.parameter_specs, median_values),
            state.parameter_specs,
            boundary_frac=float(getattr(args, "nuts_init_boundary_frac", DEFAULT_NUTS_INIT_BOUNDARY_FRAC)),
        )
        if not np.all(np.isfinite(center_theta)):
            raise ValueError(f"SVI block {block_index} produced non-finite guide median values.")
        if center_theta_previous is not None:
            block_center_shift.append(float(np.linalg.norm(center_theta - center_theta_previous)))
        center_theta_previous = np.asarray(center_theta, dtype=float)
        init_values = {spec.sample_name: float(center_theta[idx]) for idx, spec in enumerate(state.parameter_specs)}
        remaining_steps -= int(block_steps_current)
        if remaining_steps > 0:
            _log(
                args,
                (
                    f"[svi:refresh] block={block_index} remaining_steps={remaining_steps} "
                    f"center_shift={block_center_shift[-1] if block_center_shift else float('nan'):.4g}"
                ),
            )
            if evaluator.surrogate_enabled:
                evaluator.refresh_surrogate(center_theta, reason=f"svi_block_{block_index}")
            evaluator.refresh_scaling_scatter_cache(center_theta, reason=f"svi_block_{block_index}")
            evaluator.refresh_source_metric_cache(center_theta, reason=f"svi_block_{block_index}")
            block_refresh_count += 1
    if guide is None or svi is None or svi_result is None or params is None or center_theta_previous is None:
        raise RuntimeError("SVI did not run any steps.")
    if evaluator.surrogate_enabled:
        evaluator.refresh_surrogate(center_theta_previous, reason="svi_final")
    evaluator.refresh_scaling_scatter_cache(center_theta_previous, reason="svi_final")
    evaluator.refresh_source_metric_cache(center_theta_previous, reason="svi_final")
    svi_elapsed = time.time() - svi_start
    evaluator.timing_totals["svi_runtime"] = evaluator.timing_totals.get("svi_runtime", 0.0) + svi_elapsed
    losses = np.concatenate(block_losses) if block_losses else np.empty((0,), dtype=float)
    center_theta = np.asarray(center_theta_previous, dtype=float)
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
        "svi_final_elbo_loss": float(losses[-1]) if len(losses) else float("nan"),
        "svi_runtime_sec": float(svi_elapsed),
        "svi_init_values_used": bool(state.svi_init_values),
        "svi_blocked_refresh": bool(use_blocked_refresh),
        "svi_refresh_every": int(refresh_every),
        "svi_block_count": int(len(block_steps)),
        "svi_block_steps": [int(value) for value in block_steps],
        "svi_cache_refresh_count": int(block_refresh_count),
        "svi_block_center_shift": [float(value) for value in block_center_shift],
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
        grouped_samples=None,
        grouped_log_prob=None,
        sampler="svi",
    )
    _log_posterior_summary(args, "svi_guide", posterior)
    del guide_samples_dict, svi_result, params
    gc.collect()
    _log(
        args,
        (
            f"[svi] complete in {_fmt_seconds(svi_elapsed)} final_elbo={diagnostics['svi_final_elbo_loss']:.4g} "
            f"guide_draws={total_draws} blocks={diagnostics['svi_block_count']} "
            f"cache_refreshes={diagnostics['svi_cache_refresh_count']}"
        ),
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
            f"initial_step_size={float(args.initial_step_size):.3g} "
            f"init={nuts_init.diagnostics['strategy_used']} distinct_seeds={nuts_init.diagnostics['distinct_chain_seeds']}"
        ),
    )
    nuts = NUTS(
        sample_model,
        target_accept_prob=args.target_accept,
        max_tree_depth=args.max_tree_depth,
        step_size=float(args.initial_step_size),
        dense_mass=True,
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
    _apply_nuts_quality_gate(args, posterior, state.parameter_specs)
    _log_posterior_summary(args, "nuts", posterior)
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
    del mcmc, samples_dict, extra
    gc.collect()
    return posterior


def _nested_sampler_class():
    try:
        from numpyro.contrib.nested_sampling import NestedSampler
    except ImportError as exc:
        raise RuntimeError(
            "Nested sampling requires numpyro.contrib.nested_sampling and jaxns. "
            "Install jaxns in the active environment to use --fit-mode evidence-ns."
        ) from exc
    return NestedSampler


def _result_scalar(results: Any, name: str, *, integer: bool = False) -> float | int | None:
    if results is None or not hasattr(results, name):
        return None
    value = getattr(results, name)
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    scalar = array.reshape(-1)[0].item()
    return int(scalar) if integer else float(scalar)


def _result_array(results: Any, name: str, *, limit: int | None = None, dtype: Any = float) -> np.ndarray | None:
    if results is None or not hasattr(results, name):
        return None
    value = getattr(results, name)
    if value is None:
        return None
    array = np.asarray(value, dtype=dtype)
    if limit is not None and array.ndim >= 1:
        array = array[: max(0, int(limit))]
    return np.asarray(array)


def _extract_ns_diagnostics(results: Any) -> dict[str, np.ndarray] | None:
    if results is None:
        return None
    total_samples = _result_scalar(results, "total_num_samples", integer=True)
    limit = int(total_samples) if total_samples is not None else None
    diagnostics: dict[str, np.ndarray] = {}
    for name in (
        "log_L_samples",
        "log_dp_mean",
        "log_X_mean",
        "num_live_points_per_sample",
        "num_likelihood_evaluations_per_sample",
        "log_efficiency",
        "log_Z_mean",
        "log_Z_uncert",
        "ESS",
        "H_mean",
    ):
        array = _result_array(results, name, limit=limit)
        if array is not None:
            diagnostics[name] = array
    return diagnostics or None


def _run_ns_phase_with_progress(args: argparse.Namespace, fn):
    if bool(getattr(args, "quiet", False)):
        return _run_logged_phase(args, "ns.run", fn)
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        progress.add_task("ns: nested sampling", total=None)
        return _run_logged_phase(args, "ns.run", fn)


def _run_numpyro_nested_sampler(
    args: argparse.Namespace,
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    sample_model,
    *,
    nested_sampler_factory: Any | None = None,
) -> PosteriorResults:
    nested_sampler_factory = nested_sampler_factory or _nested_sampler_class()
    num_live_points = getattr(args, "ns_num_live_points", None)
    if num_live_points is None:
        num_live_points = max(1, int(len(state.parameter_specs)) * 25)
    num_live_points = int(num_live_points)
    max_samples_arg = getattr(args, "ns_max_samples", DEFAULT_NS_MAX_SAMPLES)
    max_samples = None if max_samples_arg is None else int(max_samples_arg)
    posterior_samples = int(DEFAULT_NS_POSTERIOR_SAMPLES)
    dlogz = float(getattr(args, "ns_dlogz", DEFAULT_NS_DLOGZ))
    _log(
        args,
        (
            f"[ns] preparing sampler num_live_points={num_live_points} "
            f"max_samples={max_samples if max_samples is not None else 'none'} "
            f"dlogZ={dlogz:.4g} "
            f"posterior_samples={posterior_samples} posterior_resampling=jaxns.get_samples"
        ),
    )
    ns_verbose = not bool(getattr(args, "quiet", False))
    ns = nested_sampler_factory(
        sample_model,
        constructor_kwargs={
            "num_live_points": num_live_points,
            "max_samples": max_samples,
            "devices": jax.devices(),
            "verbose": ns_verbose,
        },
        termination_kwargs={"dlogZ": dlogz},
    )
    rng_key = jax.random.PRNGKey(0 if args.seed is None else int(args.seed) + 707)
    _log(args, "[ns] nested sampling started")
    ns_start = time.time()
    _run_ns_phase_with_progress(args, lambda: ns.run(rng_key))
    ns_elapsed = time.time() - ns_start
    evaluator.timing_totals["nested_sampling_runtime"] = evaluator.timing_totals.get("nested_sampling_runtime", 0.0) + ns_elapsed

    ns_results = getattr(ns, "_results", None)
    ns_diagnostics = _extract_ns_diagnostics(ns_results)
    total_raw_samples = _result_scalar(ns_results, "total_num_samples", integer=True)
    resample_key = jax.random.PRNGKey(0 if args.seed is None else int(args.seed) + 1707)
    resampled = _run_logged_phase(
        args,
        "ns.get_samples",
        lambda: ns.get_samples(resample_key, posterior_samples),
    )
    samples = _extract_samples(resampled, state.parameter_specs, 1)
    if samples.shape[0] != posterior_samples:
        raise RuntimeError(
            "NestedSampler.get_samples returned an unexpected number of posterior samples "
            f"({samples.shape[0]} for requested {posterior_samples})."
        )
    sample_weights = np.full(samples.shape[0], 1.0 / float(samples.shape[0]), dtype=float)
    _log(
        args,
        (
            f"[ns] resampled posterior samples={samples.shape[0]} "
            f"raw_samples={total_raw_samples if total_raw_samples is not None else 'unknown'} "
            "method=jaxns.get_samples"
        ),
    )
    log_prob = _run_logged_phase(
        args,
        "ns.posterior_logprob",
        lambda: _posterior_logprob_matrix(state.parameter_specs, evaluator, samples),
    )
    diagnostics = {
        "strategy_requested": FIT_METHOD_NS,
        "strategy_used": FIT_METHOD_NS,
        "svi_used": False,
        "requested_chains": 0,
        "retained_finite_chains": 0,
        "dropped_nonfinite_chains": 0,
        "retained_chain_indices": [],
        "dropped_chain_indices": [],
        "chain_seed_labels": [],
        "distinct_chain_seeds": 0,
        "ns_num_live_points": num_live_points,
        "ns_max_samples": max_samples,
        "ns_posterior_samples": int(samples.shape[0]),
        "ns_posterior_resampling": "jaxns.get_samples",
        "ns_dlogz": dlogz,
        "ns_runtime_sec": float(ns_elapsed),
        "ns_log_z_mean": _result_scalar(ns_results, "log_Z_mean"),
        "ns_log_z_uncert": _result_scalar(ns_results, "log_Z_uncert"),
        "ns_ess": _result_scalar(ns_results, "ESS"),
        "ns_total_num_samples": total_raw_samples,
        "ns_total_num_likelihood_evaluations": _result_scalar(
            ns_results,
            "total_num_likelihood_evaluations",
            integer=True,
        ),
        "ns_termination_reason": _result_scalar(ns_results, "termination_reason", integer=True),
        "invalid_state_rejection_count": int(getattr(evaluator, "invalid_state_rejection_count", 0)),
        "invalid_state_reason_counts": {
            key: int(value) for key, value in dict(getattr(evaluator, "invalid_state_reason_counts", {})).items()
        },
    }
    posterior = PosteriorResults(
        samples=samples,
        log_prob=log_prob,
        sample_weights=sample_weights,
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=int(samples.shape[0]),
        num_chains=0,
        init_diagnostics=diagnostics,
        grouped_samples=None,
        grouped_log_prob=None,
        sampler="numpyro_jaxns",
        ns_diagnostics=ns_diagnostics,
    )
    _log_posterior_summary(args, "ns", posterior)
    _log(
        args,
        (
            f"[ns] complete in {_fmt_seconds(ns_elapsed)} posterior_samples={samples.shape[0]} "
            f"raw_samples={total_raw_samples if total_raw_samples is not None else 'unknown'} "
            f"logZ={diagnostics['ns_log_z_mean']} logZ_uncert={diagnostics['ns_log_z_uncert']}"
        ),
    )
    del ns
    gc.collect()
    return posterior


def _nuts_posterior_is_usable(posterior: PosteriorResults) -> tuple[bool, str]:
    samples = np.asarray(posterior.samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        return False, "fewer than two retained samples"
    if not np.isfinite(samples).all():
        return False, "non-finite posterior samples"
    accept_prob = np.asarray(posterior.accept_prob, dtype=float)
    diverging = np.asarray(posterior.diverging, dtype=bool)
    accept_mean = float(np.nanmean(accept_prob)) if accept_prob.size else float("nan")
    divergence_fraction = float(np.mean(diverging)) if diverging.size else 0.0
    spans = np.nanmax(samples, axis=0) - np.nanmin(samples, axis=0)
    dynamic_parameters = int(np.sum(np.isfinite(spans) & (spans > 1.0e-10)))
    if not np.isfinite(accept_mean) or accept_mean < 1.0e-3:
        return False, f"mean acceptance is {accept_mean:.3g}"
    if divergence_fraction > 0.95:
        return False, f"divergence fraction is {divergence_fraction:.3f}"
    if dynamic_parameters < 2:
        return False, f"only {dynamic_parameters} parameters have dynamic range"
    return True, "ok"


def _fail(message: str) -> None:
    raise SystemExit(message)


def _stage_arg_values(value: Any, *, flag_name: str) -> list[Any]:
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if not values:
        _fail(f"{flag_name} requires one, two, or three values.")
    if len(values) > 3:
        _fail(f"{flag_name} accepts at most three values: stage 2, stage 3, and stage 4.")
    return values


def _linearized_stage_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE)) == IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA


def _stage4_explicit_beta_enabled(args: argparse.Namespace) -> bool:
    return _linearized_stage_enabled(args)


def _sample_likelihood_uses_explicit_beta(sample_likelihood_mode: str) -> bool:
    return sample_likelihood_mode in {
        SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
    }


def _sample_likelihood_uses_marginal_beta(sample_likelihood_mode: str) -> bool:
    return sample_likelihood_mode in {
        SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE,
    }


def _sample_likelihood_uses_image_scatter(sample_likelihood_mode: str) -> bool:
    return _sample_likelihood_uses_explicit_beta(sample_likelihood_mode) or _sample_likelihood_uses_marginal_beta(
        sample_likelihood_mode
    )


def _effective_image_presence_penalty_weight(
    requested_weight: float | None,
    *,
    sample_likelihood_mode: str,
    fit_mode: str,
    image_plane_mode: str,
) -> float:
    if requested_weight is not None:
        return float(requested_weight)
    if str(sample_likelihood_mode) != SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE:
        return 0.0
    if str(fit_mode) == FIT_MODE_EVIDENCE_NS:
        return 0.0
    if str(image_plane_mode) != IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA:
        return 0.0
    return DEFAULT_IMAGE_PRESENCE_STAGE4_PENALTY_WEIGHT


def _validation_source_centroid_weights(
    count: int,
    measurement_sigma_arcsec: float,
    scatter_sigma_arcsec: float,
    covariance_floor: float,
) -> tuple[np.ndarray, float]:
    variance = max(
        float(measurement_sigma_arcsec) ** 2 + float(scatter_sigma_arcsec) ** 2,
        float(covariance_floor),
        1.0e-18,
    )
    return np.full(int(count), 1.0 / variance, dtype=float), float(np.sqrt(variance))


def _stage4_sample_likelihood_mode(args: argparse.Namespace) -> str | None:
    mode = str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE))
    if mode == IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA:
        return SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
    return None


def _stage4_run_directory_name(args: argparse.Namespace) -> str:
    return "stage4_linearized_image_plane"


SEQUENTIAL_STAGE_NAMES = {
    "stage1_large_only",
    "stage2_joint",
    "stage3_image_plane",
    "stage4_linearized_image_plane",
}
SEQUENTIAL_STAGE_ORDER = (
    "stage1_large_only",
    "stage2_joint",
    "stage3_image_plane",
    "stage4_linearized_image_plane",
)


def _sequential_stage_name(value: str | Path) -> str:
    return Path(str(value)).name


def _is_sequential_stage_path(value: str | Path) -> bool:
    return _sequential_stage_name(value) in SEQUENTIAL_STAGE_NAMES


def _final_sequential_exact_diagnostics_stage(*, stage3_enabled: bool, stage4_enabled: bool) -> str:
    if stage4_enabled:
        return "stage4_linearized_image_plane"
    if stage3_enabled:
        return "stage3_image_plane"
    return "stage2_joint"


def _stage_allows_exact_image_diagnostics(value: str | Path, exact_diagnostics_stage: str | Path | None) -> bool:
    if not _is_sequential_stage_path(value):
        return True
    if exact_diagnostics_stage is None:
        return True
    return _sequential_stage_name(value) == _sequential_stage_name(exact_diagnostics_stage)


def _force_quick_diagnostics_for_nonfinal_stage(
    stage_args: argparse.Namespace,
    run_name: str | Path,
    exact_diagnostics_stage: str | Path | None,
) -> argparse.Namespace:
    if _is_sequential_stage_path(run_name) and not _stage_allows_exact_image_diagnostics(run_name, exact_diagnostics_stage):
        return _clone_args(stage_args, quick_diagnostics=True)
    return stage_args


def _final_available_sequential_stage(stage_dirs: list[Path]) -> str | None:
    available = {
        _sequential_stage_name(stage_dir)
        for stage_dir in stage_dirs
        if _is_sequential_stage_path(stage_dir) and _has_plot_artifacts(stage_dir / "artifacts")
    }
    for stage_name in reversed(SEQUENTIAL_STAGE_ORDER):
        if stage_name in available:
            return stage_name
    return None


def _plots_only_exact_diagnostics_stage(run_dir: Path) -> str | None:
    if not _is_sequential_stage_path(run_dir):
        return None
    sibling_stage_dirs = [run_dir.parent / stage_name for stage_name in SEQUENTIAL_STAGE_ORDER]
    return _final_available_sequential_stage(sibling_stage_dirs) or _sequential_stage_name(run_dir)


def _local_jacobian_stage_enabled(args: argparse.Namespace) -> bool:
    mode = str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE))
    if mode == IMAGE_PLANE_MODE_LOCAL_JACOBIAN:
        return True
    if mode == IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA:
        return not bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False))
    return False


def _normalize_stage_fit_controls(args: argparse.Namespace) -> dict[str, StageFitControls]:
    fit_mode = str(getattr(args, "fit_mode", FIT_MODE_SEQUENTIAL))
    mode = str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE))
    ns_num_live_points = getattr(args, "ns_num_live_points", None)
    if ns_num_live_points is not None and int(ns_num_live_points) <= 0:
        _fail("--ns-num-live-points must be positive when provided.")
    ns_max_samples = getattr(args, "ns_max_samples", DEFAULT_NS_MAX_SAMPLES)
    if ns_max_samples is not None:
        try:
            ns_max_samples_int = int(ns_max_samples)
        except (TypeError, ValueError):
            _fail("--ns-max-samples must be a positive integer or 'none'.")
        if ns_max_samples_int <= 0:
            _fail("--ns-max-samples must be positive.")
    if float(getattr(args, "ns_dlogz", DEFAULT_NS_DLOGZ)) <= 0.0:
        _fail("--ns-dlogz must be positive.")
    if float(getattr(args, "linearized_beta_prior_sigma_arcsec", DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC)) <= 0.0:
        _fail("--linearized-beta-prior-sigma-arcsec must be positive.")
    if float(getattr(args, "image_plane_scatter_upper_arcsec", DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC)) <= DEFAULT_IMAGE_SIGMA_INT_LOWER_ARCSEC:
        _fail(
            "--image-plane-scatter-upper-arcsec must be greater than "
            f"{DEFAULT_IMAGE_SIGMA_INT_LOWER_ARCSEC:g}."
        )
    image_presence_penalty_weight = getattr(args, "image_presence_penalty_weight", None)
    if image_presence_penalty_weight is not None and (
        not np.isfinite(float(image_presence_penalty_weight)) or float(image_presence_penalty_weight) < 0.0
    ):
        _fail("--image-presence-penalty-weight must be non-negative when provided.")
    if (
        not np.isfinite(float(getattr(args, "image_presence_match_radius_arcsec", DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC)))
        or float(getattr(args, "image_presence_match_radius_arcsec", DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC)) <= 0.0
    ):
        _fail("--image-presence-match-radius-arcsec must be positive.")
    if (
        not np.isfinite(float(getattr(args, "image_presence_temperature_arcsec", DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)))
        or float(getattr(args, "image_presence_temperature_arcsec", DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)) <= 0.0
    ):
        _fail("--image-presence-temperature-arcsec must be positive.")
    if (
        not np.isfinite(float(getattr(args, "image_presence_count_softness", DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS)))
        or float(getattr(args, "image_presence_count_softness", DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS)) <= 0.0
    ):
        _fail("--image-presence-count-softness must be positive.")
    if (
        not np.isfinite(float(getattr(args, "image_presence_count_margin", DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN)))
        or float(getattr(args, "image_presence_count_margin", DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN)) < 0.0
    ):
        _fail("--image-presence-count-margin must be non-negative.")
    if int(getattr(args, "fit_quality_draws", 0)) < 0:
        _fail("--fit-quality-draws must be non-negative.")
    if int(getattr(args, "fit_quality_workers", 1)) <= 0:
        _fail("--fit-quality-workers must be positive.")
    if _has_cosmology_init_overrides(args) and not bool(getattr(args, "fit_cosmology_flat_wcdm", False)):
        _fail("--cosmology-init-om0 and --cosmology-init-w0 require --fit-cosmology-flat-wcdm.")
    try:
        _cosmology_init_overrides_from_args(args)
    except ValueError as exc:
        _fail(str(exc))
    caustic_source_redshift = float(getattr(args, "caustic_source_redshift", 7.0))
    if not np.isfinite(caustic_source_redshift) or caustic_source_redshift <= 0.0:
        _fail("--caustic-source-redshift must be finite and positive.")
    evidence_prior_sigma = getattr(args, "evidence_source_prior_sigma_arcsec", None)
    if evidence_prior_sigma is not None and float(evidence_prior_sigma) <= 0.0:
        _fail("--evidence-source-prior-sigma-arcsec must be positive.")
    evidence_likelihood_mode = str(
        getattr(args, "evidence_likelihood_mode", DEFAULT_EVIDENCE_LIKELIHOOD_MODE)
    )
    if evidence_likelihood_mode not in EVIDENCE_LIKELIHOOD_MODES:
        _fail(
            "--evidence-likelihood-mode must be one of "
            f"{', '.join(EVIDENCE_LIKELIHOOD_MODES)}."
        )
    if fit_mode == FIT_MODE_EVIDENCE_NS:
        sampled_source_evidence = (
            evidence_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
        )
        if evidence_prior_sigma is None:
            _fail("--fit-mode evidence-ns requires --evidence-source-prior-sigma-arcsec.")
        if mode != IMAGE_PLANE_MODE_NONE:
            _fail("--fit-mode evidence-ns owns its likelihood and requires --image-plane-mode none.")
        if bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)):
            _fail("--skip-stage3-image-plane-local-jacobian is not valid with --fit-mode evidence-ns.")
        if int(getattr(args, "image_plane_newton_steps", 0)) != 0 and not sampled_source_evidence:
            _fail(
                "--image-plane-newton-steps is only valid with --fit-mode evidence-ns "
                "--evidence-likelihood-mode linearized-forward-beta-image-plane."
            )
        if (
            sampled_source_evidence
            and str(getattr(args, "sampling_engine", "full")) == "refreshing_surrogate"
            and int(getattr(args, "image_plane_newton_steps", 0)) > 0
        ):
            _fail(
                "--sampling-engine refreshing_surrogate with linearized-forward-beta-image-plane "
                "requires --image-plane-newton-steps 0."
            )
        if (
            float(getattr(args, "linearized_beta_prior_sigma_arcsec", DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC))
            != DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC
        ):
            _fail("--linearized-beta-prior-sigma-arcsec is not valid with --fit-mode evidence-ns; use --evidence-source-prior-sigma-arcsec.")
        if (
            str(getattr(args, "source_position_parameterization", SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED))
            != SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED
            and not sampled_source_evidence
        ):
            _fail(
                "--source-position-parameterization is only valid with --fit-mode evidence-ns "
                "--evidence-likelihood-mode linearized-forward-beta-image-plane."
            )
        source_position_parameterization = str(
            getattr(args, "source_position_parameterization", SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED)
        )
        if source_position_parameterization not in SOURCE_POSITION_PARAMETERIZATIONS:
            _fail(
                "--source-position-parameterization must be one of "
                f"{', '.join(SOURCE_POSITION_PARAMETERIZATIONS)}."
            )
        controls = {
            "stage2": StageFitControls(fit_method=FIT_METHOD_NS, warmup=0, samples=0),
            "stage3": StageFitControls(fit_method=FIT_METHOD_NS, warmup=0, samples=0),
            "stage4": StageFitControls(fit_method=FIT_METHOD_NS, warmup=0, samples=0),
        }
        return controls
    if evidence_likelihood_mode != DEFAULT_EVIDENCE_LIKELIHOOD_MODE:
        _fail("--evidence-likelihood-mode is only valid with --fit-mode evidence-ns.")

    fit_methods = [str(value) for value in _stage_arg_values(getattr(args, "fit_method", FIT_METHOD_SVI_NUTS), flag_name="--fit-method")]
    warmups = [int(value) for value in _stage_arg_values(getattr(args, "warmup", DEFAULT_WARMUP), flag_name="--warmup")]
    samples = [int(value) for value in _stage_arg_values(getattr(args, "samples", DEFAULT_SAMPLES), flag_name="--samples")]

    invalid_fit_methods = sorted(set(fit_methods).difference({FIT_METHOD_SVI, FIT_METHOD_SVI_NUTS, FIT_METHOD_NS}))
    if invalid_fit_methods:
        _fail(f"--fit-method has unsupported value(s): {', '.join(invalid_fit_methods)}")
    if any(value == FIT_METHOD_NS for value in fit_methods):
        _fail("--fit-method ns is only valid with --fit-mode evidence-ns.")
    if any(value < 0 for value in warmups):
        _fail("--warmup values must be non-negative.")
    if any(value <= 0 for value in samples):
        _fail("--samples values must be positive.")

    max_value_count = max(len(fit_methods), len(warmups), len(samples))
    has_stage_specific_values = max_value_count >= 2
    has_three_stage_values = max_value_count == 3
    is_sequential = fit_mode == FIT_MODE_SEQUENTIAL
    has_stage3_or_stage4 = is_sequential and mode in {
        IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    }
    has_stage4 = _stage4_explicit_beta_enabled(args)
    if bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)) and not has_stage4:
        _fail("--skip-stage3-image-plane-local-jacobian is only valid with an explicit-beta stage-4 image-plane mode.")
    if has_stage4 and not is_sequential:
        _fail("Explicit-beta stage-4 image-plane modes require --fit-mode sequential.")
    if (
        has_stage4
        and str(getattr(args, "sampling_engine", "full")) == "refreshing_surrogate"
        and int(getattr(args, "image_plane_newton_steps", 0)) > 0
    ):
        _fail(
            "--sampling-engine refreshing_surrogate with linearized-forward-beta-image-plane "
            "requires --image-plane-newton-steps 0."
        )
    if has_stage_specific_values and not has_stage3_or_stage4:
        _fail(
            "Two-value --fit-method, --warmup, or --samples is only valid with "
            "--fit-mode sequential and an image-plane mode."
        )
    if has_three_stage_values and not has_stage4:
        _fail(
            "Three-value --fit-method, --warmup, or --samples is only valid with "
            "an explicit-beta stage-4 image-plane mode."
        )

    def stage_value(values: list[Any], index: int) -> Any:
        return values[index] if len(values) > index else values[0]

    def stage4_value(values: list[Any]) -> Any:
        if len(values) > 2:
            return values[2]
        if bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)) and len(values) > 1:
            return values[1]
        if len(values) > 1:
            return values[1]
        return values[0]

    controls = {
        "stage2": StageFitControls(
            fit_method=str(stage_value(fit_methods, 0)),
            warmup=int(stage_value(warmups, 0)),
            samples=int(stage_value(samples, 0)),
        ),
        "stage3": StageFitControls(
            fit_method=str(stage_value(fit_methods, 1)),
            warmup=int(stage_value(warmups, 1)),
            samples=int(stage_value(samples, 1)),
        ),
        "stage4": StageFitControls(
            fit_method=str(stage4_value(fit_methods)),
            warmup=int(stage4_value(warmups)),
            samples=int(stage4_value(samples)),
        ),
    }
    return controls


def _args_with_fit_controls(args: argparse.Namespace, controls: StageFitControls, **updates: Any) -> argparse.Namespace:
    payload: dict[str, Any] = {
        "fit_method": controls.fit_method,
        "warmup": controls.warmup,
        "samples": controls.samples,
    }
    payload.update(updates)
    if "fit_cosmology_flat_wcdm" in payload and not bool(payload["fit_cosmology_flat_wcdm"]):
        payload["cosmology_init_om0"] = None
        payload["cosmology_init_w0"] = None
    return _clone_args(args, **payload)


def _extract_reference(parsed: dict[str, Any]) -> tuple[int, float, float]:
    runmode = parsed.get("runmode")
    if not isinstance(runmode, dict):
        raise ValueError("Missing runmode block in .par file.")
    reference = runmode.get("reference")
    if not isinstance(reference, list) or len(reference) < 3:
        raise ValueError("Missing runmode.reference in .par file.")
    return int(reference[0]), float(reference[1]), float(reference[2])


def _build_cosmology(parsed: dict[str, Any]):
    return _cosmology_config_from_parsed(parsed)


def _build_cosmology_from_config(cosmo_config: dict[str, Any]):
    if not isinstance(cosmo_config, dict):
        return FlatLambdaCDM(H0=70.0, Om0=0.3)
    class_name = str(cosmo_config.get("class", "FlatLambdaCDM"))
    h0 = _h0_from_config(cosmo_config)
    om0 = _om0_from_config(cosmo_config)
    ode0 = _ode0_from_config(cosmo_config)
    w0 = _w0_from_config(cosmo_config)
    if class_name == "FlatLambdaCDM":
        return FlatLambdaCDM(H0=h0, Om0=om0)
    if class_name == "FlatwCDM":
        return FlatwCDM(H0=h0, Om0=om0, w0=w0)
    if class_name == "LambdaCDM":
        return LambdaCDM(H0=h0, Om0=om0, Ode0=ode0)
    if class_name == "wCDM":
        return wCDM(H0=h0, Om0=om0, Ode0=ode0, w0=w0)
    return FlatLambdaCDM(H0=h0, Om0=om0)


def _cosmology_w0_value(cosmo: Any) -> float:
    if isinstance(cosmo, dict):
        return _w0_from_config(cosmo)
    return float(getattr(cosmo, "w0", -1.0))


def _cosmology_config_from_any(cosmo: Any) -> dict[str, float | str]:
    if isinstance(cosmo, dict):
        return dict(cosmo)
    om0 = float(getattr(cosmo, "Om0", 0.3))
    return {
        "class": cosmo.__class__.__name__,
        "H0": float(getattr(cosmo, "H0").value),
        "Om0": om0,
        "Ode0": float(getattr(cosmo, "Ode0", 1.0 - om0)),
        "w0": _cosmology_w0_value(cosmo),
    }


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


def _bin_redshifts_by_lensing_efficiency(
    redshifts: list[float],
    *,
    z_lens: float,
    cosmo_config: dict[str, Any],
    fractional_tolerance: float,
) -> dict[float, float]:
    tolerance = max(float(fractional_tolerance), 0.0)
    unique_redshifts = sorted(set(float(z) for z in redshifts))
    entries: list[tuple[float, float]] = []
    h0 = _h0_from_config(cosmo_config)
    om0 = _om0_from_config(cosmo_config)
    w0 = _w0_from_config(cosmo_config)
    for z_source in unique_redshifts:
        efficiency = float(np.asarray(_jax_flat_wcdm_lensing_efficiency(z_lens, z_source, h0, om0, w0)))
        if not np.isfinite(efficiency) or efficiency <= 0.0:
            raise ValueError(
                f"Cannot bin source redshift z={z_source:.8g}: lensing efficiency D_ls/D_s is invalid "
                f"for lens redshift z_lens={z_lens:.8g}."
            )
        entries.append((z_source, efficiency))
    entries.sort(key=lambda item: item[1])

    groups: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for entry in entries:
        candidate = current + [entry]
        efficiencies = np.asarray([value[1] for value in candidate], dtype=float)
        representative = float(np.median(efficiencies))
        fractional_spread = float((np.max(efficiencies) - np.min(efficiencies)) / max(abs(representative), 1.0e-12))
        if current and fractional_spread > tolerance:
            groups.append(current)
            current = [entry]
        else:
            current = candidate
    if current:
        groups.append(current)

    mapping: dict[float, float] = {}
    for group in groups:
        efficiencies = np.asarray([value[1] for value in group], dtype=float)
        representative_efficiency = float(np.median(efficiencies))
        representative_z = float(
            min(group, key=lambda item: abs(float(item[1]) - representative_efficiency))[0]
        )
        for z_source, _efficiency in group:
            mapping[float(z_source)] = representative_z
    return mapping


def _build_geometry_cache(
    cosmo_config: dict[str, Any],
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
        float(z_source): _dpie_sigma0_factor_from_config(z_lens, float(z_source), cosmo_config)
        for z_source in effective_z_source_values
    }
    dpie_sigma0_factor_by_exact_z = {
        float(z_source): _dpie_sigma0_factor_from_config(z_lens, float(z_source), cosmo_config)
        for z_source in exact_z_source_values
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
        flat_wcdm_quadrature_order=DEFAULT_JAX_COSMO_DISTANCE_STEPS,
        lens_quadrature_z=None,
        lens_quadrature_weights=None,
        effective_z_quadrature_z=None,
        effective_z_quadrature_weights=None,
        exact_z_quadrature_z=None,
        exact_z_quadrature_weights=None,
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


def _radius_transform_for_component_prior(
    potential: dict[str, Any],
    field_name: str,
    decoded_prior: dict[str, Any],
) -> dict[str, Any]:
    prior_kind = str(decoded_prior["prior_kind"])
    lower = float(decoded_prior["lower"])
    upper = float(decoded_prior["upper"])
    mean = None if decoded_prior["mean"] is None else float(decoded_prior["mean"])
    std = None if decoded_prior["std"] is None else float(decoded_prior["std"])
    step = float(decoded_prior["step"])
    transform_kind = "identity"
    transform_offset = 0.0
    physical_lower = lower
    physical_upper = upper
    physical_mean = mean
    physical_std = std

    if field_name == "core_radius_kpc":
        transform_kind = "log_positive"
        safe_lower = max(lower, SAFE_RADIUS_MARGIN_KPC)
        safe_upper = max(upper, safe_lower * (1.0 + 1.0e-9))
        if prior_kind == "normal":
            if mean is None or std is None:
                raise ValueError(f"Radius prior for {potential.get('id', 'potential')}.{field_name} requires mean/std.")
            mean, std = _positive_lognormal_parameters(max(mean, safe_lower), std, floor=safe_lower)
            lower = float("-inf")
            upper = float("inf")
        else:
            lower = float(np.log(safe_lower))
            upper = float(np.log(safe_upper))
            step = float(max(step, SAFE_RADIUS_MARGIN_KPC))
    elif field_name == "cut_radius_kpc":
        core_radius = _coerce_numeric(potential.get("core_radius_kpc", SAFE_RADIUS_MARGIN_KPC), "core_radius_kpc")
        transform_kind = "log_offset_positive"
        transform_offset = max(float(core_radius), 0.0)
        gap_lower = max(lower - transform_offset, SAFE_RADIUS_MARGIN_KPC)
        gap_upper = max(upper - transform_offset, gap_lower * (1.0 + 1.0e-9))
        if prior_kind == "normal":
            if mean is None or std is None:
                raise ValueError(f"Radius prior for {potential.get('id', 'potential')}.{field_name} requires mean/std.")
            mean, std = _positive_lognormal_parameters(max(mean - transform_offset, gap_lower), std, floor=gap_lower)
            lower = float("-inf")
            upper = float("inf")
        else:
            lower = float(np.log(gap_lower))
            upper = float(np.log(gap_upper))
            step = float(max(step, SAFE_RADIUS_MARGIN_KPC))

    return {
        "prior_kind": prior_kind,
        "lower": lower,
        "upper": upper,
        "step": step,
        "mean": mean,
        "std": std,
        "transform_kind": transform_kind,
        "transform_offset": transform_offset,
        "physical_lower": physical_lower,
        "physical_upper": physical_upper,
        "physical_mean": physical_mean,
        "physical_std": physical_std,
    }


def _positive_log_transform_for_component_prior(
    decoded_prior: dict[str, Any],
    context: str,
    *,
    floor: float = SAFE_VDISP_MARGIN,
) -> dict[str, Any]:
    prior_kind = str(decoded_prior["prior_kind"])
    lower = float(decoded_prior["lower"])
    upper = float(decoded_prior["upper"])
    mean = None if decoded_prior["mean"] is None else float(decoded_prior["mean"])
    std = None if decoded_prior["std"] is None else float(decoded_prior["std"])
    step = float(decoded_prior["step"])
    physical_lower = max(lower, floor) if np.isfinite(lower) else float(floor)
    physical_upper = upper if np.isfinite(upper) else None
    physical_mean = mean
    physical_std = std

    if prior_kind == "normal":
        if mean is None or std is None:
            raise ValueError(f"Positive prior for {context} requires mean/std.")
        mean, std = _positive_lognormal_parameters(mean, std, floor=floor)
        lower = float("-inf")
        upper = float("inf")
    else:
        safe_lower = max(lower, floor)
        safe_upper = max(upper, safe_lower * (1.0 + 1.0e-9))
        lower = float(np.log(safe_lower))
        upper = float(np.log(safe_upper))
        step = float(max(step, floor))

    return {
        "prior_kind": prior_kind,
        "lower": lower,
        "upper": upper,
        "step": step,
        "mean": mean,
        "std": std,
        "transform_kind": "log_positive",
        "transform_offset": 0.0,
        "physical_lower": physical_lower,
        "physical_upper": physical_upper,
        "physical_mean": physical_mean,
        "physical_std": physical_std,
    }


def _build_parameter_specs(
    potentials_with_priors: list[dict[str, Any]],
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
            lens_model_list.append(ORIGINAL_DPIE_PROFILE_NAME)
        else:
            lens_model_list.append("SHEAR")
        assignments: list[tuple[str, int]] = []
        priors = potential.get("priors", {}) or {}
        for field_name, prior in priors.items():
            normalized_field_name = _normalize_component_field_name(str(field_name))
            decoded_prior = _decode_parameter_prior(prior, f"{potential_id}.{normalized_field_name}")
            if decoded_prior is None:
                continue
            prior_spec = (
                _radius_transform_for_component_prior(potential, normalized_field_name, decoded_prior)
                if profile_type == DP_IE_PROFILE and normalized_field_name in {"core_radius_kpc", "cut_radius_kpc"}
                else _positive_log_transform_for_component_prior(
                    decoded_prior,
                    f"{potential_id}.{normalized_field_name}",
                )
                if profile_type == DP_IE_PROFILE and normalized_field_name == "v_disp"
                else {
                    "prior_kind": str(decoded_prior["prior_kind"]),
                    "lower": float(decoded_prior["lower"]),
                    "upper": float(decoded_prior["upper"]),
                    "step": float(decoded_prior["step"]),
                    "mean": None if decoded_prior["mean"] is None else float(decoded_prior["mean"]),
                    "std": None if decoded_prior["std"] is None else float(decoded_prior["std"]),
                    "transform_kind": "identity",
                    "transform_offset": 0.0,
                    "physical_lower": None,
                    "physical_upper": None,
                    "physical_mean": None,
                    "physical_std": None,
                }
            )
            index = len(specs)
            specs.append(
                ParameterSpec(
                    name=f"{potential_id}.{normalized_field_name}",
                    sample_name=_sample_name(potential_id, normalized_field_name),
                    potential_id=potential_id,
                    profile_type=profile_type,
                    field=normalized_field_name,
                    prior_kind=str(prior_spec["prior_kind"]),
                    lower=float(prior_spec["lower"]),
                    upper=float(prior_spec["upper"]),
                    step=float(prior_spec["step"]),
                    mean=prior_spec["mean"],
                    std=prior_spec["std"],
                    transform_kind=str(prior_spec["transform_kind"]),
                    physical_lower=prior_spec["physical_lower"],
                    physical_upper=prior_spec["physical_upper"],
                    physical_mean=prior_spec["physical_mean"],
                    physical_std=prior_spec["physical_std"],
                    transform_offset=float(prior_spec["transform_offset"]),
                )
            )
            assignments.append((normalized_field_name, index))
        component_param_assignments.append(assignments)
    return specs, component_param_assignments, lens_model_list


def _build_scaling_parameter_specs(
    potfiles: list[dict[str, Any]],
    start_index: int = 0,
    kpc_per_arcsec: float = 1.0,
) -> tuple[list[ParameterSpec], list[dict[str, int]], list[str]]:
    specs: list[ParameterSpec] = []
    param_index_by_potfile: list[dict[str, int]] = []
    lens_model_list: list[str] = []
    for potfile in potfiles:
        potfile_id = str(potfile["id"])
        potfile_type = int(potfile["type"])
        if potfile_type != DP_IE_PROFILE:
            raise ValueError(f"Unsupported potfile type {potfile_type} for {potfile_id}.")
        for _ in range(len(potfile["catalog_df"])):
            lens_model_list.append(ORIGINAL_DPIE_PROFILE_NAME)
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


def _physical_normal_moments_to_latent(
    spec: ParameterSpec,
    mean: float,
    std: float,
) -> tuple[float, float]:
    physical_mean = float(mean)
    physical_std = max(float(std), 1.0e-12)
    if spec.transform_kind == "log_positive":
        floor = max(float(spec.physical_lower or SAFE_VDISP_MARGIN), SAFE_VDISP_MARGIN)
        return _positive_lognormal_parameters(physical_mean, physical_std, floor=floor)
    if spec.transform_kind == "log_offset_positive":
        offset = float(spec.transform_offset)
        floor = max(float(spec.physical_lower or SAFE_RADIUS_MARGIN_KPC) - offset, SAFE_RADIUS_MARGIN_KPC)
        return _positive_lognormal_parameters(physical_mean - offset, physical_std, floor=floor)
    if spec.transform_kind == "affine":
        scale = float(getattr(spec, "transform_scale", 1.0))
        if abs(scale) <= 0.0:
            raise ValueError(f"Affine transform scale for {spec.name} must be non-zero.")
        return (physical_mean - float(spec.transform_offset)) / scale, physical_std / abs(scale)
    return physical_mean, physical_std


def _build_stage2_large_parameter_specs(
    large_specs: list[ParameterSpec],
    stage1_prior_summary: Stage1PriorSummary,
) -> list[ParameterSpec]:
    stage2_specs: list[ParameterSpec] = []
    for spec in large_specs:
        if spec.sample_name not in stage1_prior_summary.means:
            raise ValueError(f"Missing stage-1 posterior mean for parameter {spec.sample_name}.")
        physical_mean = float(stage1_prior_summary.means[spec.sample_name])
        physical_std = float(stage1_prior_summary.stds.get(spec.sample_name, 0.0))
        mean, std = _physical_normal_moments_to_latent(spec, physical_mean, physical_std)
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
                physical_mean=physical_mean,
                physical_std=max(physical_std, 1.0e-12),
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


def _build_image_scatter_parameter_spec(start_index: int, upper_arcsec: float) -> ParameterSpec:
    del start_index
    upper = max(float(upper_arcsec), DEFAULT_IMAGE_SIGMA_INT_LOWER_ARCSEC * 1.01)
    return ParameterSpec(
        name="image.sigma_int",
        sample_name="image_sigma_int",
        potential_id="image",
        profile_type=0,
        field="sigma_int",
        prior_kind="uniform",
        lower=float(np.log(DEFAULT_IMAGE_SIGMA_INT_LOWER_ARCSEC)),
        upper=float(np.log(upper)),
        step=0.05,
        component_family="image_scatter",
        transform_kind="log_positive",
        physical_lower=DEFAULT_IMAGE_SIGMA_INT_LOWER_ARCSEC,
        physical_upper=upper,
    )


def _build_cosmology_parameter_specs(start_index: int, cosmo: Any) -> list[ParameterSpec]:
    del start_index
    om0_value = _om0_from_config(cosmo) if isinstance(cosmo, dict) else float(getattr(cosmo, "Om0", 0.3))
    om0 = float(np.clip(om0_value, DEFAULT_COSMOLOGY_OM0_LOWER, DEFAULT_COSMOLOGY_OM0_UPPER))
    w0 = float(np.clip(_cosmology_w0_value(cosmo), DEFAULT_COSMOLOGY_W0_LOWER, DEFAULT_COSMOLOGY_W0_UPPER))
    return [
        ParameterSpec(
            name="cosmology.Om0",
            sample_name=COSMOLOGY_OM0_SAMPLE_NAME,
            potential_id="cosmology",
            profile_type=0,
            field="Om0",
            prior_kind="uniform",
            lower=DEFAULT_COSMOLOGY_OM0_LOWER,
            upper=DEFAULT_COSMOLOGY_OM0_UPPER,
            step=0.01,
            mean=om0,
            component_family="cosmology",
            transform_kind="identity",
            physical_lower=DEFAULT_COSMOLOGY_OM0_LOWER,
            physical_upper=DEFAULT_COSMOLOGY_OM0_UPPER,
            physical_mean=om0,
        ),
        ParameterSpec(
            name="cosmology.w0",
            sample_name=COSMOLOGY_W0_SAMPLE_NAME,
            potential_id="cosmology",
            profile_type=0,
            field="w0",
            prior_kind="uniform",
            lower=DEFAULT_COSMOLOGY_W0_LOWER,
            upper=DEFAULT_COSMOLOGY_W0_UPPER,
            step=0.05,
            mean=w0,
            component_family="cosmology",
            transform_kind="identity",
            physical_lower=DEFAULT_COSMOLOGY_W0_LOWER,
            physical_upper=DEFAULT_COSMOLOGY_W0_UPPER,
            physical_mean=w0,
        ),
    ]


def _source_position_sample_name(family_id: str, axis: str) -> str:
    safe_family_id = str(family_id).replace(".", "_").replace("-", "_")
    return f"source_{safe_family_id}_beta_{axis}"


def _shared_source_position_prior_values(
    family_data: list[FamilyData],
    mean_x: float,
    mean_y: float,
) -> dict[str, tuple[float, float]]:
    return {
        str(family.family_id): (float(mean_x), float(mean_y))
        for family in family_data
    }


def _build_source_position_parameter_specs(
    family_data: list[FamilyData],
    source_position_prior_values: dict[str, tuple[float, float]],
    *,
    start_index: int,
    beta_prior_sigma_arcsec: float,
    parameterization: str = SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED,
) -> list[ParameterSpec]:
    del start_index
    sigma = max(float(beta_prior_sigma_arcsec), 1.0e-6)
    mode = str(parameterization)
    if mode not in SOURCE_POSITION_PARAMETERIZATIONS:
        raise ValueError(
            f"Unsupported source-position parameterization {mode!r}; "
            f"expected one of {', '.join(SOURCE_POSITION_PARAMETERIZATIONS)}."
        )
    specs: list[ParameterSpec] = []
    for family in family_data:
        if family.family_id not in source_position_prior_values:
            raise ValueError(
                f"Missing stage-3 source centroid for family {family.family_id!r}; "
                "stage 4 requires source-position initialization from stage 3."
            )
        center_x, center_y = source_position_prior_values[family.family_id]
        for axis, center in (("x", center_x), ("y", center_y)):
            specs.append(
                ParameterSpec(
                    name=f"source.{family.family_id}.beta_{axis}",
                    sample_name=_source_position_sample_name(family.family_id, axis),
                    potential_id=str(family.family_id),
                    profile_type=0,
                    field=f"beta_{axis}",
                    prior_kind="normal",
                    lower=float("-inf"),
                    upper=float("inf"),
                    step=0.1 if mode != SOURCE_POSITION_PARAMETERIZATION_DIRECT else max(0.02, 0.1 * sigma),
                    mean=0.0 if mode != SOURCE_POSITION_PARAMETERIZATION_DIRECT else float(center),
                    std=1.0 if mode != SOURCE_POSITION_PARAMETERIZATION_DIRECT else sigma,
                    component_family="source_position",
                    transform_kind="affine" if mode == SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED else "identity",
                    physical_mean=float(center),
                    physical_std=sigma,
                    transform_offset=float(center) if mode == SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED else 0.0,
                    transform_scale=sigma if mode == SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED else 1.0,
                )
            )
    return specs


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
    scaling_scatter_indices: list[dict[str, int]] | None,
    start_component_index: int,
    kpc_per_arcsec: float = 1.0,
) -> tuple[list[dict[str, Any]], list[list[tuple[str, int]]], list[dict[str, Any]], list[dict[str, Any]]]:
    _, ra0_deg, dec0_deg = reference
    components: list[dict[str, Any]] = []
    assignments: list[list[tuple[str, int]]] = []
    scaling_component_assignments: list[dict[str, Any]] = []
    scaling_component_records: list[dict[str, Any]] = []
    scaling_scatter_indices = scaling_scatter_indices or [{} for _ in potfiles]
    for potfile_order, (potfile, param_index_lookup, scatter_index_lookup) in enumerate(
        zip(potfiles, scaling_param_indices, scaling_scatter_indices)
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
                    "sigma_log_scatter_param_index": int(scatter_index_lookup.get("sigma", -1)),
                    "core_log_scatter_param_index": int(scatter_index_lookup.get("core", -1)),
                    "cut_log_scatter_param_index": int(scatter_index_lookup.get("cut", -1)),
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


def _adaptive_active_scaling_count(
    importance: np.ndarray,
    *,
    cumulative_fraction: float,
    min_count: int,
    max_count: int,
) -> tuple[int, int, int]:
    values = np.asarray(importance, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    n_values = int(values.size)
    if n_values == 0:
        return 0, 0, 0
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        return min(max(int(min_count), 1), n_values), 0, 0
    cumulative = np.cumsum(values) / total
    target = float(np.clip(cumulative_fraction, 0.0, 1.0))
    cumulative_count = int(np.searchsorted(cumulative, target, side="left") + 1)
    if n_values <= 2:
        knee_count = n_values
    else:
        rank_fraction = (np.arange(n_values, dtype=float) + 1.0) / float(n_values)
        knee_count = int(np.argmax(cumulative - rank_fraction) + 1)
    if max_count < 0:
        cap = n_values
    elif max_count == 0:
        cap = min(DEFAULT_ACTIVE_SCALING_GALAXIES, n_values)
    else:
        cap = min(int(max_count), n_values)
    selected = max(int(min_count), knee_count, cumulative_count)
    selected = min(max(selected, 1), cap)
    return selected, cumulative_count, knee_count


def _prepare_family_data(
    images_df: pd.DataFrame,
    sigma_arcsec: float,
    reference: tuple[int, float, float],
    *,
    z_lens: float,
    cosmo_config: dict[str, Any],
    z_bin_efficiency_tol: float,
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
    z_mapping = _bin_redshifts_by_lensing_efficiency(
        list(family_redshifts.values()),
        z_lens=float(z_lens),
        cosmo_config=cosmo_config,
        fractional_tolerance=float(z_bin_efficiency_tol),
    )
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
                reliability=np.asarray(
                    pd.to_numeric(family_df.get("family_reliability", 1.0), errors="coerce")
                    .fillna(1.0)
                    .clip(0.0, 1.0)
                    .to_numpy(dtype=float),
                    dtype=float,
                ),
            )
        )
    return families, float(time.perf_counter() - family_start)


def _filter_singleton_families(images_df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    if images_df.empty:
        return images_df.copy(), 0, 0
    family_counts = images_df["family_id"].astype(str).value_counts()
    constrained_family_ids = set(family_counts[family_counts > 1].index.astype(str))
    filtered = images_df[images_df["family_id"].astype(str).isin(constrained_family_ids)].copy()
    return (
        filtered.reset_index(drop=True),
        int(len(images_df) - len(filtered)),
        int(len(family_counts) - len(constrained_family_ids)),
    )


def _filter_non_positive_redshift_families(images_df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, list[str]]:
    if images_df.empty or "catalog_z" not in images_df.columns:
        return images_df.copy(), 0, 0, []

    family_ids = images_df["family_id"].astype(str)
    z_values = pd.to_numeric(images_df["catalog_z"], errors="coerce").to_numpy(dtype=float)
    bad_row_mask = np.isfinite(z_values) & (z_values <= 0.0)
    bad_family_ids = sorted(family_ids[bad_row_mask].unique().astype(str).tolist())
    if not bad_family_ids:
        return images_df.copy(), 0, 0, []

    bad_family_mask = family_ids.isin(bad_family_ids)
    filtered = images_df.loc[~bad_family_mask].copy()
    return (
        filtered.reset_index(drop=True),
        int(bad_family_mask.sum()),
        int(len(bad_family_ids)),
        bad_family_ids,
    )


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
        reliability_per_image = np.concatenate([family.reliability for family in family_list])
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
                reliability_per_image=reliability_per_image,
            )
        )
    return bin_data


def _dpie_sigma0_from_vel_disp(vel_disp: float, ra_arcsec: float, rs_arcsec: float, z_lens: float, z_source: float, cosmo: Any) -> float:
    return _jax_dpie_sigma0_from_vel_disp(
        vel_disp,
        ra_arcsec,
        rs_arcsec,
        z_lens,
        z_source,
        _cosmology_config_from_any(cosmo),
    )


def _dpie_sigma0_factor(z_lens: float, z_source: float, cosmo: Any) -> float:
    return _dpie_sigma0_factor_from_config(z_lens, z_source, _cosmology_config_from_any(cosmo))


class ClusterJAXEvaluator:
    def __init__(
        self,
        state: BuildState,
        match_tolerance_arcsec: float,
        validate_top_k_families: int,
        sampling_engine: str = "full",
        active_scaling_galaxies: list[int] | int | None = None,
        active_scaling_selection: str = "adaptive",
        active_scaling_cumulative_fraction: float = DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        active_scaling_min: int = DEFAULT_ACTIVE_SCALING_MIN,
        refresh_every: int = DEFAULT_REFRESH_EVERY,
        refresh_param_drift_frac: float = DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        validation_approx: str = "exact",
        source_plane_covariance_floor: float = 1.0e-6,
        source_plane_outlier_sigma_arcsec: float = DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC,
        sample_likelihood_mode: str = SAMPLE_LIKELIHOOD_SOURCE,
        image_plane_newton_steps: int = 0,
        image_presence_penalty_weight: float = 0.0,
        image_presence_match_radius_arcsec: float = DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC,
        image_presence_temperature_arcsec: float = DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC,
        image_presence_count_softness: float = DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS,
        image_presence_count_margin: float = DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN,
        evidence_source_prior_sigma_arcsec: float | None = None,
        evidence_source_prior_mean_x_arcsec: float = 0.0,
        evidence_source_prior_mean_y_arcsec: float = 0.0,
        exact_image_solver: str = DEFAULT_EXACT_IMAGE_SOLVER,
        quick_diagnostics: bool = False,
    ):
        self.state = state
        self.match_tolerance_arcsec = float(match_tolerance_arcsec)
        self.validate_top_k_families = max(0, int(validate_top_k_families))
        self.sampling_engine = str(sampling_engine)
        self.active_scaling_galaxies_by_potfile = _normalize_active_scaling_counts(active_scaling_galaxies, state.potfiles)
        self.active_scaling_selection = str(active_scaling_selection)
        self.active_scaling_cumulative_fraction = float(active_scaling_cumulative_fraction)
        self.active_scaling_min = max(1, int(active_scaling_min))
        self.refresh_every = max(1, int(refresh_every))
        self.refresh_param_drift_frac = float(refresh_param_drift_frac)
        self.validation_approx = str(validation_approx)
        self.source_plane_covariance_floor = max(float(source_plane_covariance_floor), 0.0)
        self.source_plane_outlier_sigma_arcsec = max(float(source_plane_outlier_sigma_arcsec), 1.0e-6)
        self.sample_likelihood_mode = str(sample_likelihood_mode)
        if self.sample_likelihood_mode not in {
            SAMPLE_LIKELIHOOD_SOURCE,
            SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
            SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
            SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE,
        }:
            raise ValueError(f"Unsupported sample_likelihood_mode={self.sample_likelihood_mode!r}.")
        requested_image_plane_newton_steps = int(image_plane_newton_steps)
        if (
            self.sampling_engine == "refreshing_surrogate"
            and self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
            and requested_image_plane_newton_steps > 0
        ):
            raise ValueError(
                "refreshing_surrogate with linearized-forward-beta-image-plane requires "
                "image_plane_newton_steps=0; Newton updates move image positions away from the observed-position cache."
            )
        self.image_plane_newton_steps = max(0, min(3, requested_image_plane_newton_steps))
        self.image_presence_penalty_weight = max(float(image_presence_penalty_weight), 0.0)
        self.image_presence_match_radius_arcsec = max(float(image_presence_match_radius_arcsec), 1.0e-12)
        self.image_presence_temperature_arcsec = max(float(image_presence_temperature_arcsec), 1.0e-12)
        self.image_presence_count_softness = max(float(image_presence_count_softness), 1.0e-12)
        self.image_presence_count_margin = max(float(image_presence_count_margin), 0.0)
        self.evidence_source_prior_sigma_arcsec = (
            None if evidence_source_prior_sigma_arcsec is None else float(evidence_source_prior_sigma_arcsec)
        )
        self.evidence_source_prior_mean_x_arcsec = float(evidence_source_prior_mean_x_arcsec)
        self.evidence_source_prior_mean_y_arcsec = float(evidence_source_prior_mean_y_arcsec)
        self.exact_image_solver = str(exact_image_solver)
        if self.exact_image_solver not in EXACT_IMAGE_SOLVERS:
            raise ValueError(f"Unsupported exact_image_solver={self.exact_image_solver!r}.")
        self.quick_diagnostics = bool(quick_diagnostics)
        if _sample_likelihood_uses_marginal_beta(self.sample_likelihood_mode):
            if self.evidence_source_prior_sigma_arcsec is None or self.evidence_source_prior_sigma_arcsec <= 0.0:
                raise ValueError(
                    "linearized marginalized beta evidence requires a positive evidence_source_prior_sigma_arcsec."
                )
            for family in state.family_data:
                reliability = np.asarray(family.reliability, dtype=float)
                if reliability.size and not np.allclose(reliability, 1.0, rtol=0.0, atol=1.0e-8):
                    raise ValueError(
                        "linearized marginalized beta evidence requires all image reliabilities to be 1.0; "
                        "reliability-mixture marginalization is not implemented."
                    )
        geometry_setup_start = time.perf_counter()
        if state.cosmo_config:
            self.cosmo_config = dict(state.cosmo_config)
        else:  # pragma: no cover - legacy fallback
            self.cosmo_config = _build_cosmology(state.parsed)
            self.state.cosmo_config = dict(self.cosmo_config)
        self.cosmo = _build_cosmology_from_config(self.cosmo_config)
        if getattr(self.state, "geometry_cache", None) is None:
            self.state.geometry_cache = _build_geometry_cache(
                self.cosmo_config,
                state.z_lens,
                state.family_data,
                state.bin_data,
            )
        geometry_cache = self.state.geometry_cache
        if geometry_cache is None or geometry_cache.dpie_sigma0_factor_by_effective_z is None:
            self.state.geometry_cache = _build_geometry_cache(
                self.cosmo_config,
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
            "exact_solver_jax": 0.0,
            "exact_solver_lenstronomy": 0.0,
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
        _validate_supported_lens_model_list(state.lens_model_list, "BuildState")
        unique_lens_model_list = list(dict.fromkeys(state.lens_model_list))
        self.unique_lens_model_list = unique_lens_model_list
        self.bulk_index_list = np.asarray([unique_lens_model_list.index(name) for name in state.lens_model_list], dtype=np.int32)
        self.models_by_effective_z = {
            float(effective_z_source): LensModelBulk(
                unique_lens_model_list=unique_lens_model_list,
                multi_plane=False,
                cosmo=self.cosmo,
            )
            for effective_z_source in geometry_cache.effective_z_source_values
        }
        self.kpc_per_arcsec = _kpc_per_arcsec_from_config(state.z_lens, self.cosmo_config)
        self.dpie_sigma0_factors = {
            float(z_source): float(value)
            for z_source, value in geometry_cache.dpie_sigma0_factor_by_effective_z.items()
        }
        self.exact_dpie_sigma0_factors = {
            float(z_source): float(value)
            for z_source, value in geometry_cache.dpie_sigma0_factor_by_exact_z.items()
        }
        self.fit_cosmology_flat_wcdm = bool(getattr(self.state, "fit_cosmology_flat_wcdm", False))
        self.cosmology_h0 = _h0_from_config(self.cosmo_config)
        self.cosmology_fiducial_om0 = _om0_from_config(self.cosmo_config)
        self.cosmology_fiducial_w0 = _w0_from_config(self.cosmo_config)
        self.cosmology_effective_z_values = np.asarray(geometry_cache.effective_z_source_values, dtype=float)
        self.cosmology_effective_z_to_index = {
            float(z_source): int(index) for index, z_source in enumerate(self.cosmology_effective_z_values)
        }
        self.cosmology_effective_z_values_jax = jnp.asarray(self.cosmology_effective_z_values, dtype=jnp.float64)
        self.cosmology_exact_z_values = np.asarray(geometry_cache.exact_z_source_values, dtype=float)
        self.cosmology_exact_z_to_index = {
            float(z_source): int(index) for index, z_source in enumerate(self.cosmology_exact_z_values)
        }
        self.exact_models_by_z: dict[float, NPLensModel] = {}
        self.exact_solvers_by_z: dict[float, NPLensEquationSolver] = {}
        self.jax_exact_models_by_z: dict[float, LensModelBulk] = {}
        self.exact_jax_attempt_count = 0
        self.exact_jax_fallback_count = 0
        self.exact_lenstronomy_count = 0
        self.timing_totals["geometry_cache_setup"] += time.perf_counter() - geometry_setup_start
        self.traced_bin_data = tuple(self._prepare_traced_bin_data(bin_item) for bin_item in state.bin_data)
        self.traced_bin_data_by_z = {bin_item.effective_z_source: bin_item for bin_item in self.traced_bin_data}
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
        self.scaling_scatter_param_indices = np.asarray(
            [idx for idx, spec in enumerate(self.state.parameter_specs) if spec.component_family == "scaling_scatter"],
            dtype=np.int32,
        )
        source_scatter_indices = [
            idx for idx, spec in enumerate(self.state.parameter_specs) if spec.component_family == "source_scatter"
        ]
        self.source_sigma_int_param_index = int(source_scatter_indices[0]) if source_scatter_indices else -1
        image_scatter_indices = [
            idx for idx, spec in enumerate(self.state.parameter_specs) if spec.component_family == "image_scatter"
        ]
        self.image_sigma_int_param_index = int(image_scatter_indices[0]) if image_scatter_indices else -1
        self.cosmology_om0_param_index = next(
            (
                idx
                for idx, spec in enumerate(self.state.parameter_specs)
                if spec.sample_name == COSMOLOGY_OM0_SAMPLE_NAME
            ),
            -1,
        )
        self.cosmology_w0_param_index = next(
            (
                idx
                for idx, spec in enumerate(self.state.parameter_specs)
                if spec.sample_name == COSMOLOGY_W0_SAMPLE_NAME
            ),
            -1,
        )
        self.fit_cosmology_flat_wcdm = bool(
            self.fit_cosmology_flat_wcdm
            and self.cosmology_om0_param_index >= 0
            and self.cosmology_w0_param_index >= 0
        )
        self.source_position_param_indices_by_family = {
            str(family.family_id): (
                next(
                    (
                        idx
                        for idx, spec in enumerate(self.state.parameter_specs)
                        if spec.component_family == "source_position"
                        and spec.potential_id == str(family.family_id)
                        and spec.field == "beta_x"
                    ),
                    -1,
                ),
                next(
                    (
                        idx
                        for idx, spec in enumerate(self.state.parameter_specs)
                        if spec.component_family == "source_position"
                        and spec.potential_id == str(family.family_id)
                        and spec.field == "beta_y"
                    ),
                    -1,
                ),
            )
            for family in self.state.family_data
        }
        self.source_position_parameterization = _explicit_source_position_parameterization_for_state(self.state)
        self.source_position_conditional = (
            self.source_position_parameterization == SOURCE_POSITION_PARAMETERIZATION_CONDITIONAL_WHITENED
        )
        self._conditional_source_inverse_basis_cache: dict[tuple[tuple[int, ...], str, bytes], tuple[dict[str, Any], ...]] = {}
        self.scaling_param_indices_jax = jnp.asarray(self.scaling_param_indices, dtype=jnp.int32)
        surrogate_param_indices = self.scaling_param_indices.tolist()
        if bool(getattr(self, "fit_cosmology_flat_wcdm", False)):
            surrogate_param_indices.extend([self.cosmology_om0_param_index, self.cosmology_w0_param_index])
        self.surrogate_param_indices = np.asarray(
            [idx for idx in dict.fromkeys(int(value) for value in surrogate_param_indices) if idx >= 0],
            dtype=np.int32,
        )
        self.surrogate_param_indices_jax = jnp.asarray(self.surrogate_param_indices, dtype=jnp.int32)
        transform_kind_array = np.asarray(
            [str(spec.transform_kind) for spec in self.state.parameter_specs],
            dtype=object,
        )
        transform_offset_array = np.asarray(
            [float(spec.transform_offset) for spec in self.state.parameter_specs],
            dtype=float,
        )
        transform_scale_array = np.asarray(
            [float(getattr(spec, "transform_scale", 1.0)) for spec in self.state.parameter_specs],
            dtype=float,
        )
        self.transform_kind_log_positive_mask = jnp.asarray(transform_kind_array == "log_positive", dtype=bool)
        self.transform_kind_log_offset_positive_mask = jnp.asarray(transform_kind_array == "log_offset_positive", dtype=bool)
        self.transform_kind_affine_mask = jnp.asarray(transform_kind_array == "affine", dtype=bool)
        self.transform_offset_array = jnp.asarray(transform_offset_array, dtype=jnp.float64)
        self.transform_scale_array = jnp.asarray(transform_scale_array, dtype=jnp.float64)
        self.packed_spec_jax = self._prepare_packed_spec_arrays()
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
        stage4_refreshing_surrogate_supported = (
            self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
            and self.image_plane_newton_steps == 0
        )
        source_plane_refreshing_surrogate_supported = not _sample_likelihood_uses_image_scatter(
            self.sample_likelihood_mode
        )
        self.surrogate_enabled = (
            self.sampling_engine == "refreshing_surrogate"
            and self.use_bulk_ray_shooting
            and len(self.scaling_component_indices) > 0
            and len(self.inactive_scaling_component_indices) > 0
            and len(self.surrogate_param_indices) > 0
            and (source_plane_refreshing_surrogate_supported or stage4_refreshing_surrogate_supported)
        )
        self.surrogate_reference_params: np.ndarray | None = None
        self.surrogate_reference_param_values = np.zeros(len(self.surrogate_param_indices), dtype=float)
        self.surrogate_cache_by_z: dict[float, SurrogateBinCache] = {}
        self.scaling_scatter_reference_params: np.ndarray | None = None
        self.scaling_scatter_cache_by_z: dict[float, dict[str, np.ndarray]] = {}
        self.source_metric_reference_params: np.ndarray | None = None
        self.source_metric_cache_by_z: dict[float, dict[str, np.ndarray]] = {}
        self._source_loglike_fn = jax.jit(self._source_loglike_impl)

    def _prepare_traced_bin_data(self, bin_data: BinData) -> TracedBinData:
        family_idx = np.asarray(bin_data.family_index_per_image, dtype=np.int32)
        n_families = len(bin_data.family_ids)
        family_counts = np.zeros(n_families, dtype=np.int32)
        np.add.at(family_counts, family_idx, 1)
        return TracedBinData(
            effective_z_source=float(bin_data.effective_z_source),
            family_ids=tuple(str(family_id) for family_id in bin_data.family_ids),
            n_families=int(n_families),
            family_index_per_image=jnp.asarray(family_idx, dtype=jnp.int32),
            x_obs=jnp.asarray(bin_data.x_obs, dtype=jnp.float64),
            y_obs=jnp.asarray(bin_data.y_obs, dtype=jnp.float64),
            sigma_per_image=jnp.asarray(bin_data.sigma_per_image, dtype=jnp.float64),
            reliability_per_image=jnp.asarray(bin_data.reliability_per_image, dtype=jnp.float64),
            image_has_constraint=jnp.asarray(family_counts[family_idx] > 1, dtype=bool),
            effective_z_index=int(self.cosmology_effective_z_to_index.get(float(bin_data.effective_z_source), -1)),
        )

    def _prepare_packed_spec_arrays(self) -> dict[str, jnp.ndarray]:
        spec = self.state.packed_lens_spec
        return {
            "x_center_base": jnp.asarray(spec.x_center_base, dtype=jnp.float64),
            "x_center_param_index": jnp.asarray(spec.x_center_param_index, dtype=jnp.int32),
            "y_center_base": jnp.asarray(spec.y_center_base, dtype=jnp.float64),
            "y_center_param_index": jnp.asarray(spec.y_center_param_index, dtype=jnp.int32),
            "ellipticite_base": jnp.asarray(spec.ellipticite_base, dtype=jnp.float64),
            "ellipticite_param_index": jnp.asarray(spec.ellipticite_param_index, dtype=jnp.int32),
            "angle_pos_base": jnp.asarray(spec.angle_pos_base, dtype=jnp.float64),
            "angle_pos_param_index": jnp.asarray(spec.angle_pos_param_index, dtype=jnp.int32),
            "core_radius_kpc_base": jnp.asarray(spec.core_radius_kpc_base, dtype=jnp.float64),
            "core_radius_param_index": jnp.asarray(spec.core_radius_param_index, dtype=jnp.int32),
            "cut_radius_kpc_base": jnp.asarray(spec.cut_radius_kpc_base, dtype=jnp.float64),
            "cut_radius_param_index": jnp.asarray(spec.cut_radius_param_index, dtype=jnp.int32),
            "v_disp_base": jnp.asarray(spec.v_disp_base, dtype=jnp.float64),
            "v_disp_param_index": jnp.asarray(spec.v_disp_param_index, dtype=jnp.int32),
            "gamma_base": jnp.asarray(spec.gamma_base, dtype=jnp.float64),
            "gamma_param_index": jnp.asarray(spec.gamma_param_index, dtype=jnp.int32),
            "profile_type": jnp.asarray(spec.profile_type, dtype=jnp.int32),
            "component_family": jnp.asarray(spec.component_family, dtype=jnp.int32),
            "luminosity_ratio": jnp.asarray(spec.luminosity_ratio, dtype=jnp.float64),
            "sigma_ref_base": jnp.asarray(spec.sigma_ref_base, dtype=jnp.float64),
            "sigma_ref_param_index": jnp.asarray(spec.sigma_ref_param_index, dtype=jnp.int32),
            "cut_ref_base": jnp.asarray(spec.cut_ref_base, dtype=jnp.float64),
            "cut_ref_param_index": jnp.asarray(spec.cut_ref_param_index, dtype=jnp.int32),
            "core_ref_base": jnp.asarray(spec.core_ref_base, dtype=jnp.float64),
            "core_ref_param_index": jnp.asarray(spec.core_ref_param_index, dtype=jnp.int32),
            "vdslope_base": jnp.asarray(spec.vdslope_base, dtype=jnp.float64),
            "vdslope_param_index": jnp.asarray(spec.vdslope_param_index, dtype=jnp.int32),
            "slope_base": jnp.asarray(spec.slope_base, dtype=jnp.float64),
            "slope_param_index": jnp.asarray(spec.slope_param_index, dtype=jnp.int32),
            "sigma_log_scatter_param_index": jnp.asarray(spec.sigma_log_scatter_param_index, dtype=jnp.int32),
            "core_log_scatter_param_index": jnp.asarray(spec.core_log_scatter_param_index, dtype=jnp.int32),
            "cut_log_scatter_param_index": jnp.asarray(spec.cut_log_scatter_param_index, dtype=jnp.int32),
        }

    def _physical_parameter_vector(self, params: jnp.ndarray) -> jnp.ndarray:
        return _apply_parameter_transforms_jax(
            params,
            self.transform_kind_log_positive_mask,
            self.transform_kind_log_offset_positive_mask,
            self.transform_offset_array,
            self.transform_kind_affine_mask,
            self.transform_scale_array,
        )

    def _source_sigma_int_jax(self, params: jnp.ndarray) -> jnp.ndarray:
        physical_params = self._physical_parameter_vector(jnp.asarray(params, dtype=jnp.float64))
        return self._source_sigma_int_from_physical(physical_params)

    def _source_sigma_int_from_physical(self, physical_params: jnp.ndarray) -> jnp.ndarray:
        if self.source_sigma_int_param_index < 0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        return jnp.take(physical_params, jnp.asarray(self.source_sigma_int_param_index, dtype=jnp.int32))

    def _source_sigma_int_numpy(self, params: np.ndarray | jnp.ndarray) -> float:
        if self.source_sigma_int_param_index < 0:
            return 0.0
        params_array = np.asarray(params, dtype=float)
        physical_params = _convert_theta_to_physical(params_array, self.state.parameter_specs)
        return float(physical_params[self.source_sigma_int_param_index])

    def _image_sigma_int_from_physical(self, physical_params: jnp.ndarray) -> jnp.ndarray:
        if self.image_sigma_int_param_index < 0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        return jnp.take(physical_params, jnp.asarray(self.image_sigma_int_param_index, dtype=jnp.int32))

    def _image_sigma_int_numpy(self, params: np.ndarray | jnp.ndarray) -> float:
        if self.image_sigma_int_param_index < 0:
            return 0.0
        params_array = np.asarray(params, dtype=float)
        physical_params = _convert_theta_to_physical(params_array, self.state.parameter_specs)
        return float(physical_params[self.image_sigma_int_param_index])

    def _source_position_vectors_for_bin(
        self,
        physical_params: jnp.ndarray,
        bin_data: TracedBinData,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x_indices = []
        y_indices = []
        for family_id in bin_data.family_ids:
            idx_x, idx_y = self.source_position_param_indices_by_family.get(str(family_id), (-1, -1))
            x_indices.append(int(idx_x))
            y_indices.append(int(idx_y))
        x_indices_jax = jnp.asarray(x_indices, dtype=jnp.int32)
        y_indices_jax = jnp.asarray(y_indices, dtype=jnp.int32)
        has_source_positions = jnp.all((x_indices_jax >= 0) & (y_indices_jax >= 0))
        safe_x_indices = jnp.maximum(x_indices_jax, 0)
        safe_y_indices = jnp.maximum(y_indices_jax, 0)
        source_x = jnp.take(physical_params, safe_x_indices)
        source_y = jnp.take(physical_params, safe_y_indices)
        return source_x[bin_data.family_index_per_image], source_y[bin_data.family_index_per_image], has_source_positions

    def _conditional_source_position_transport_for_bin(
        self,
        params: jnp.ndarray,
        bin_data: TracedBinData,
        beta_x: jnp.ndarray,
        beta_y: jnp.ndarray,
        image_sigma_int: jnp.ndarray,
        jacobian_entries: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x_indices = []
        y_indices = []
        prior_x = []
        prior_y = []
        prior_sigma = []
        spec_by_index = {idx: spec for idx, spec in enumerate(self.state.parameter_specs)}
        for family_id in bin_data.family_ids:
            idx_x, idx_y = self.source_position_param_indices_by_family.get(str(family_id), (-1, -1))
            x_indices.append(int(idx_x))
            y_indices.append(int(idx_y))
            spec_x = spec_by_index.get(int(idx_x))
            spec_y = spec_by_index.get(int(idx_y))
            prior_x.append(float(spec_x.physical_mean if spec_x is not None and spec_x.physical_mean is not None else 0.0))
            prior_y.append(float(spec_y.physical_mean if spec_y is not None and spec_y.physical_mean is not None else 0.0))
            sigma_x = float(spec_x.physical_std if spec_x is not None and spec_x.physical_std is not None else 1.0)
            sigma_y = float(spec_y.physical_std if spec_y is not None and spec_y.physical_std is not None else sigma_x)
            prior_sigma.append(max(0.5 * (sigma_x + sigma_y), 1.0e-9))

        x_indices_jax = jnp.asarray(x_indices, dtype=jnp.int32)
        y_indices_jax = jnp.asarray(y_indices, dtype=jnp.int32)
        has_source_positions = jnp.all((x_indices_jax >= 0) & (y_indices_jax >= 0))
        safe_x_indices = jnp.maximum(x_indices_jax, 0)
        safe_y_indices = jnp.maximum(y_indices_jax, 0)
        eta_x = jnp.take(params, safe_x_indices)
        eta_y = jnp.take(params, safe_y_indices)
        mu_x = jnp.asarray(prior_x, dtype=jnp.float64)
        mu_y = jnp.asarray(prior_y, dtype=jnp.float64)
        sigma_prior = jnp.asarray(prior_sigma, dtype=jnp.float64)
        prior_precision = 1.0 / jnp.square(sigma_prior)

        jac_a00, jac_a01, jac_a10, jac_a11 = jacobian_entries
        det = jac_a00 * jac_a11 - jac_a01 * jac_a10
        det_safe = _safe_signed_min_abs(det, 1.0e-10)
        inv00 = jac_a11 / det_safe
        inv01 = -jac_a01 / det_safe
        inv10 = -jac_a10 / det_safe
        inv11 = jac_a00 / det_safe

        sigma2 = jnp.maximum(
            jnp.square(bin_data.sigma_per_image) + jnp.square(image_sigma_int) + self.source_plane_covariance_floor,
            1.0e-18,
        )
        reliability = jnp.clip(bin_data.reliability_per_image, 1.0e-6, 1.0)
        weight = reliability / sigma2
        lambda00 = weight * (jnp.square(inv00) + jnp.square(inv10))
        lambda01 = weight * (inv00 * inv01 + inv10 * inv11)
        lambda11 = weight * (jnp.square(inv01) + jnp.square(inv11))
        family_idx = bin_data.family_index_per_image
        n_families = bin_data.n_families
        image_mask = bin_data.image_has_constraint
        lambda00 = jnp.where(image_mask, lambda00, 0.0)
        lambda01 = jnp.where(image_mask, lambda01, 0.0)
        lambda11 = jnp.where(image_mask, lambda11, 0.0)
        p00 = prior_precision + jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(lambda00)
        p01 = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(lambda01)
        p11 = prior_precision + jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(lambda11)
        rhs_x = prior_precision * mu_x + jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(
            lambda00 * beta_x + lambda01 * beta_y
        )
        rhs_y = prior_precision * mu_y + jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(
            lambda01 * beta_x + lambda11 * beta_y
        )
        precision_det = jnp.maximum(p00 * p11 - jnp.square(p01), 1.0e-18)
        mean_x = (p11 * rhs_x - p01 * rhs_y) / precision_det
        mean_y = (-p01 * rhs_x + p00 * rhs_y) / precision_det
        cov00 = p11 / precision_det
        cov01 = -p01 / precision_det
        cov11 = p00 / precision_det
        chol00 = jnp.sqrt(jnp.maximum(cov00, 1.0e-18))
        chol10 = cov01 / chol00
        chol11 = jnp.sqrt(jnp.maximum(cov11 - jnp.square(chol10), 1.0e-18))
        source_x_by_family = mean_x + chol00 * eta_x
        source_y_by_family = mean_y + chol10 * eta_x + chol11 * eta_y

        prior_quad = (jnp.square(source_x_by_family - mu_x) + jnp.square(source_y_by_family - mu_y)) * prior_precision
        log_prior_beta = -0.5 * (prior_quad + 2.0 * jnp.log(2.0 * jnp.pi * jnp.square(sigma_prior)))
        log_eta = -0.5 * (jnp.square(eta_x) + jnp.square(eta_y) + 2.0 * jnp.log(2.0 * jnp.pi))
        log_det_transport = jnp.log(chol00) + jnp.log(chol11)
        correction = jnp.sum(log_prior_beta + log_det_transport - log_eta)
        finite = (
            has_source_positions
            & jnp.all(jnp.isfinite(source_x_by_family))
            & jnp.all(jnp.isfinite(source_y_by_family))
            & jnp.all(jnp.isfinite(correction))
        )
        return (
            source_x_by_family[family_idx],
            source_y_by_family[family_idx],
            finite,
            jnp.where(finite, correction, jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64)),
        )

    def _explicit_source_position_vectors_for_bin(
        self,
        params: jnp.ndarray,
        physical_params: jnp.ndarray,
        bin_data: TracedBinData,
        beta_x: jnp.ndarray,
        beta_y: jnp.ndarray,
        image_sigma_int: jnp.ndarray,
        jacobian_entries: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if self.source_position_conditional:
            if jacobian_entries is None:
                raise ValueError("conditional-whitened source transport requires proposal-current Jacobian entries.")
            return self._conditional_source_position_transport_for_bin(
                params,
                bin_data,
                beta_x,
                beta_y,
                image_sigma_int,
                jacobian_entries,
            )
        source_x, source_y, finite = self._source_position_vectors_for_bin(physical_params, bin_data)
        return source_x, source_y, finite, jnp.asarray(0.0, dtype=jnp.float64)

    def _reported_physical_parameter_vector(self, params: jnp.ndarray) -> jnp.ndarray:
        params_jax = jnp.asarray(params, dtype=jnp.float64)
        physical_params = self._physical_parameter_vector(params_jax)
        if not self.source_position_conditional:
            return physical_params
        reported = physical_params
        image_sigma_int = self._image_sigma_int_from_physical(physical_params)
        for bin_data in self.traced_bin_data:
            packed_state, validity = self._build_packed_lens_state_with_validity_from_physical(
                physical_params,
                bin_data.effective_z_source,
                stop_gradient=True,
            )
            beta_x, beta_y = jax.lax.cond(
                validity["is_valid"],
                lambda current_state: self._ray_shooting_for_components(
                    bin_data.effective_z_source,
                    bin_data.x_obs,
                    bin_data.y_obs,
                    current_state,
                ),
                lambda _: (bin_data.x_obs, bin_data.y_obs),
                packed_state,
            )
            jacobian_entries = self._lensing_jacobian_for_components(
                bin_data.effective_z_source,
                bin_data.x_obs,
                bin_data.y_obs,
                packed_state,
            )
            source_x_per_image, source_y_per_image, _finite, _correction = self._conditional_source_position_transport_for_bin(
                params_jax,
                bin_data,
                beta_x,
                beta_y,
                image_sigma_int,
                jacobian_entries,
            )
            family_idx_np = np.asarray(bin_data.family_index_per_image, dtype=int)
            for family_index, family_id in enumerate(bin_data.family_ids):
                matches = np.where(family_idx_np == family_index)[0]
                if matches.size == 0:
                    continue
                idx_x, idx_y = self.source_position_param_indices_by_family.get(str(family_id), (-1, -1))
                if idx_x < 0 or idx_y < 0:
                    continue
                first_image = int(matches[0])
                reported = reported.at[int(idx_x)].set(source_x_per_image[first_image])
                reported = reported.at[int(idx_y)].set(source_y_per_image[first_image])
        return reported

    def reported_physical_parameter_vector(self, params: np.ndarray | jnp.ndarray) -> np.ndarray:
        return np.asarray(self._reported_physical_parameter_vector(jnp.asarray(params, dtype=jnp.float64)), dtype=float)

    def _conditional_source_inverse_cache_key(self, base_latent: np.ndarray) -> tuple[tuple[int, ...], str, bytes]:
        contiguous = np.ascontiguousarray(base_latent)
        return tuple(int(dim) for dim in contiguous.shape), contiguous.dtype.str, contiguous.tobytes()

    def _conditional_source_inverse_basis(self, base_latent: np.ndarray) -> tuple[dict[str, Any], ...]:
        cache = getattr(self, "_conditional_source_inverse_basis_cache", None)
        if cache is None:
            cache = {}
            self._conditional_source_inverse_basis_cache = cache
        key = self._conditional_source_inverse_cache_key(base_latent)
        cached = cache.get(key)
        if cached is not None:
            return cached

        base_latent = np.asarray(base_latent, dtype=float)
        mean_physical = self.reported_physical_parameter_vector(base_latent)
        basis_rows: list[dict[str, Any]] = []
        for family_id, (idx_x, idx_y) in self.source_position_param_indices_by_family.items():
            if idx_x < 0 or idx_y < 0:
                continue
            eta_x_latent = base_latent.copy()
            eta_y_latent = base_latent.copy()
            eta_x_latent[int(idx_x)] = 1.0
            eta_y_latent[int(idx_y)] = 1.0
            basis_x = self.reported_physical_parameter_vector(eta_x_latent) - mean_physical
            basis_y = self.reported_physical_parameter_vector(eta_y_latent) - mean_physical
            basis_rows.append(
                {
                    "family_id": str(family_id),
                    "idx_x": int(idx_x),
                    "idx_y": int(idx_y),
                    "mean": np.asarray(
                        [mean_physical[int(idx_x)], mean_physical[int(idx_y)]],
                        dtype=float,
                    ),
                    "transform": np.asarray(
                        [
                            [basis_x[int(idx_x)], basis_y[int(idx_x)]],
                            [basis_x[int(idx_y)], basis_y[int(idx_y)]],
                        ],
                        dtype=float,
                    ),
                }
            )
        cached = tuple(basis_rows)
        cache[key] = cached
        return cached

    def reported_physical_to_latent_parameter_vector(self, physical_params: np.ndarray | jnp.ndarray) -> np.ndarray:
        physical_array = np.asarray(physical_params, dtype=float)
        latent = _convert_theta_to_latent(physical_array, self.state.parameter_specs)
        if not self.source_position_conditional:
            return latent
        source_indices = sorted(
            {
                int(idx)
                for pair in self.source_position_param_indices_by_family.values()
                for idx in pair
                if int(idx) >= 0
            }
        )
        base_latent = np.asarray(latent, dtype=float).copy()
        for idx in source_indices:
            base_latent[idx] = 0.0
        for basis in self._conditional_source_inverse_basis(base_latent):
            idx_x = int(basis["idx_x"])
            idx_y = int(basis["idx_y"])
            rhs = np.asarray(
                [
                    physical_array[idx_x] - float(basis["mean"][0]),
                    physical_array[idx_y] - float(basis["mean"][1]),
                ],
                dtype=float,
            )
            try:
                eta = np.linalg.solve(np.asarray(basis["transform"], dtype=float), rhs)
            except np.linalg.LinAlgError:
                eta = np.zeros(2, dtype=float)
            latent[idx_x] = float(eta[0])
            latent[idx_y] = float(eta[1])
        return latent

    def _source_position_for_family_numpy(self, params: np.ndarray | jnp.ndarray, family_id: str) -> tuple[float, float] | None:
        idx_x, idx_y = self.source_position_param_indices_by_family.get(str(family_id), (-1, -1))
        if idx_x < 0 or idx_y < 0:
            return None
        params_array = np.asarray(params, dtype=float)
        physical_params = self.reported_physical_parameter_vector(params_array)
        return float(physical_params[idx_x]), float(physical_params[idx_y])

    def _scaling_scatter_field_scales(self, params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        physical_params = self._physical_parameter_vector(jnp.asarray(params, dtype=jnp.float64))
        return self._scaling_scatter_field_scales_from_physical(physical_params)

    def _scaling_scatter_field_scales_from_physical(
        self,
        physical_params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        spec_jax = self.packed_spec_jax
        is_scaling = spec_jax["component_family"] == 1

        def _field_scale(index_array: np.ndarray) -> jnp.ndarray:
            index_jax = jnp.asarray(index_array, dtype=jnp.int32)
            values = self._apply_param_updates(jnp.zeros_like(spec_jax["luminosity_ratio"]), index_array, physical_params)
            mask = is_scaling & (index_jax >= 0)
            squared_sum = jnp.sum(jnp.where(mask, jnp.square(values), 0.0))
            count = jnp.sum(jnp.where(mask, 1.0, 0.0))
            return jnp.sqrt(squared_sum / jnp.maximum(count, 1.0))

        return (
            _field_scale(spec_jax["sigma_log_scatter_param_index"]),
            _field_scale(spec_jax["core_log_scatter_param_index"]),
            _field_scale(spec_jax["cut_log_scatter_param_index"]),
        )

    def _scaling_scatter_extra_variance(
        self,
        params: jnp.ndarray,
        bin_data: BinData | TracedBinData,
        beta_x: jnp.ndarray,
        beta_y: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        physical_params = self._physical_parameter_vector(jnp.asarray(params, dtype=jnp.float64))
        return self._scaling_scatter_extra_variance_from_physical(
            physical_params,
            bin_data,
            beta_x,
            beta_y,
        )

    def _scaling_scatter_extra_variance_from_physical(
        self,
        physical_params: jnp.ndarray,
        bin_data: BinData | TracedBinData,
        beta_x: jnp.ndarray,
        beta_y: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if len(self.scaling_component_indices) == 0 or len(self.scaling_scatter_param_indices) == 0:
            zeros = jnp.zeros_like(beta_x)
            return zeros, zeros
        cache = self.scaling_scatter_cache_by_z.get(float(bin_data.effective_z_source))
        if cache is None:
            zeros = jnp.zeros_like(beta_x)
            return zeros, zeros
        sigma_scatter, core_scatter, cut_scatter = self._scaling_scatter_field_scales_from_physical(physical_params)
        sigma_dx = jnp.asarray(cache["sigma_x"], dtype=jnp.float64)
        sigma_dy = jnp.asarray(cache["sigma_y"], dtype=jnp.float64)
        core_dx = jnp.asarray(cache["core_x"], dtype=jnp.float64)
        core_dy = jnp.asarray(cache["core_y"], dtype=jnp.float64)
        cut_dx = jnp.asarray(cache["cut_x"], dtype=jnp.float64)
        cut_dy = jnp.asarray(cache["cut_y"], dtype=jnp.float64)
        var_x = (
            jnp.square(sigma_scatter * sigma_dx)
            + jnp.square(core_scatter * core_dx)
            + jnp.square(cut_scatter * cut_dx)
        )
        var_y = (
            jnp.square(sigma_scatter * sigma_dy)
            + jnp.square(core_scatter * core_dy)
            + jnp.square(cut_scatter * cut_dy)
        )
        return var_x, var_y

    def _scaling_scatter_image_covariance_from_physical(
        self,
        physical_params: jnp.ndarray,
        bin_data: BinData | TracedBinData,
        jac_a00: jnp.ndarray,
        jac_a01: jnp.ndarray,
        jac_a10: jnp.ndarray,
        jac_a11: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        zeros = jnp.zeros_like(jac_a00)
        if len(self.scaling_component_indices) == 0 or len(self.scaling_scatter_param_indices) == 0:
            return zeros, zeros, zeros, jnp.ones_like(jac_a00, dtype=bool)
        cache = self.scaling_scatter_cache_by_z.get(float(bin_data.effective_z_source))
        if cache is None:
            return zeros, zeros, zeros, jnp.ones_like(jac_a00, dtype=bool)
        sigma_scatter, core_scatter, cut_scatter = self._scaling_scatter_field_scales_from_physical(physical_params)
        det = jac_a00 * jac_a11 - jac_a01 * jac_a10
        floor = jnp.asarray(1.0e-12, dtype=jnp.float64)
        det_abs = jnp.abs(det)
        det_safe = jnp.where(det_abs >= floor, det, jnp.where(det >= 0.0, floor, -floor))

        def _image_displacement_covariance(scale: jnp.ndarray, key_x: str, key_y: str) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            source_dx = scale * jnp.asarray(cache[key_x], dtype=jnp.float64)
            source_dy = scale * jnp.asarray(cache[key_y], dtype=jnp.float64)
            image_dx = (jac_a11 * source_dx - jac_a01 * source_dy) / det_safe
            image_dy = (-jac_a10 * source_dx + jac_a00 * source_dy) / det_safe
            return jnp.square(image_dx), image_dx * image_dy, jnp.square(image_dy)

        sigma00, sigma01, sigma11 = _image_displacement_covariance(sigma_scatter, "sigma_x", "sigma_y")
        core00, core01, core11 = _image_displacement_covariance(core_scatter, "core_x", "core_y")
        cut00, cut01, cut11 = _image_displacement_covariance(cut_scatter, "cut_x", "cut_y")
        cov00 = sigma00 + core00 + cut00
        cov01 = sigma01 + core01 + cut01
        cov11 = sigma11 + core11 + cut11
        finite = (
            (det_abs >= floor)
            & jnp.isfinite(cov00)
            & jnp.isfinite(cov01)
            & jnp.isfinite(cov11)
        )
        return cov00, cov01, cov11, finite

    def refresh_scaling_scatter_cache(self, reference_params: np.ndarray, reason: str = "manual") -> None:
        self.scaling_scatter_cache_by_z = {}
        self.scaling_scatter_reference_params = None
        if len(self.scaling_component_indices) == 0 or len(self.scaling_scatter_param_indices) == 0:
            return
        reference = np.asarray(reference_params, dtype=float)
        reference_jax = jnp.asarray(reference, dtype=jnp.float64)
        eps = 1.0e-3

        def _derivative_for_field(
            bin_data: BinData,
            x_obs: jnp.ndarray,
            y_obs: jnp.ndarray,
            beta_x: jnp.ndarray,
            beta_y: jnp.ndarray,
            sigma_offset: float,
            core_offset: float,
            cut_offset: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            packed_plus = self._build_packed_lens_state(
                reference_jax,
                bin_data.effective_z_source,
                sigma_log_offset=sigma_offset,
                core_log_offset=core_offset,
                cut_log_offset=cut_offset,
            )
            beta_plus_x, beta_plus_y = self._ray_shooting_for_components(
                bin_data.effective_z_source,
                x_obs,
                y_obs,
                packed_plus,
            )
            deriv_x = np.asarray((beta_plus_x - beta_x) / eps, dtype=float)
            deriv_y = np.asarray((beta_plus_y - beta_y) / eps, dtype=float)
            return deriv_x, deriv_y

        for bin_data in self.state.bin_data:
            x_obs = jnp.asarray(bin_data.x_obs, dtype=jnp.float64)
            y_obs = jnp.asarray(bin_data.y_obs, dtype=jnp.float64)
            packed_state = self._build_packed_lens_state(reference_jax, bin_data.effective_z_source)
            validity = self._packed_lens_validity_from_params(reference_jax, bin_data.effective_z_source, stop_gradient=False)
            if not bool(np.asarray(validity["is_valid"], dtype=bool)):
                self._record_invalid_state_callback(np.asarray(validity["reason_flags"], dtype=bool))
                self.scaling_scatter_cache_by_z = {}
                return
            beta_x, beta_y = self._ray_shooting_for_components(
                bin_data.effective_z_source,
                x_obs,
                y_obs,
                packed_state,
            )
            sigma_dx, sigma_dy = _derivative_for_field(bin_data, x_obs, y_obs, beta_x, beta_y, eps, 0.0, 0.0)
            core_dx, core_dy = _derivative_for_field(bin_data, x_obs, y_obs, beta_x, beta_y, 0.0, eps, 0.0)
            cut_dx, cut_dy = _derivative_for_field(bin_data, x_obs, y_obs, beta_x, beta_y, 0.0, 0.0, eps)
            derivatives = {
                "sigma_x": sigma_dx,
                "sigma_y": sigma_dy,
                "core_x": core_dx,
                "core_y": core_dy,
                "cut_x": cut_dx,
                "cut_y": cut_dy,
            }
            if not all(np.isfinite(value).all() for value in derivatives.values()):
                self.scaling_scatter_cache_by_z = {}
                return
            self.scaling_scatter_cache_by_z[float(bin_data.effective_z_source)] = derivatives
        self.scaling_scatter_reference_params = reference.copy()
        self._source_loglike_fn = jax.jit(self._source_loglike_impl)

    def refresh_source_metric_cache(self, reference_params: np.ndarray, reason: str = "manual") -> None:
        reference = np.asarray(reference_params, dtype=float)
        reference_jax = jnp.asarray(reference, dtype=jnp.float64)
        physical_params = self._physical_parameter_vector(reference_jax)
        refreshed_cache: dict[float, dict[str, np.ndarray]] = {}
        for bin_data in self.traced_bin_data:
            packed_state, validity = self._build_packed_lens_state_with_validity_from_physical(
                physical_params,
                bin_data.effective_z_source,
                stop_gradient=False,
            )
            if not bool(np.asarray(validity["is_valid"], dtype=bool)):
                self._record_invalid_state_callback(np.asarray(validity["reason_flags"], dtype=bool))
                return
            jac_a00, jac_a01, jac_a10, jac_a11 = self._lensing_jacobian_for_components(
                bin_data.effective_z_source,
                bin_data.x_obs,
                bin_data.y_obs,
                packed_state,
            )
            inv_abs_mu = jnp.abs(jac_a00 * jac_a11 - jac_a01 * jac_a10)
            inv_abs_mu = np.clip(np.asarray(inv_abs_mu, dtype=float), 1.0e-6, 1.0e6)
            jacobian_entries = {
                "jac_a00": np.asarray(jac_a00, dtype=float),
                "jac_a01": np.asarray(jac_a01, dtype=float),
                "jac_a10": np.asarray(jac_a10, dtype=float),
                "jac_a11": np.asarray(jac_a11, dtype=float),
            }
            if not np.isfinite(inv_abs_mu).all() or not all(np.isfinite(value).all() for value in jacobian_entries.values()):
                return
            refreshed_cache[float(bin_data.effective_z_source)] = {
                "inv_abs_mu": inv_abs_mu,
                **jacobian_entries,
            }
        self.source_metric_cache_by_z = refreshed_cache
        self.source_metric_reference_params = reference.copy()
        self._source_loglike_fn = jax.jit(self._source_loglike_impl)

    def _magnification_inv_abs_mu(self, bin_data: TracedBinData) -> jnp.ndarray:
        cache = self.source_metric_cache_by_z.get(float(bin_data.effective_z_source))
        if cache is None:
            return jnp.ones_like(bin_data.sigma_per_image)
        return jnp.asarray(cache["inv_abs_mu"], dtype=jnp.float64)

    def _source_metric_jacobian_entries(
        self,
        bin_data: TracedBinData,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        cache = self.source_metric_cache_by_z.get(float(bin_data.effective_z_source))
        if cache is None or not all(key in cache for key in ("jac_a00", "jac_a01", "jac_a10", "jac_a11")):
            ones = jnp.ones_like(bin_data.sigma_per_image)
            zeros = jnp.zeros_like(bin_data.sigma_per_image)
            return ones, zeros, zeros, ones
        return (
            jnp.asarray(cache["jac_a00"], dtype=jnp.float64),
            jnp.asarray(cache["jac_a01"], dtype=jnp.float64),
            jnp.asarray(cache["jac_a10"], dtype=jnp.float64),
            jnp.asarray(cache["jac_a11"], dtype=jnp.float64),
        )

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

    def _packed_state_to_kwargs_lens_jax(
        self,
        packed_state: dict[str, Any],
        component_indices: np.ndarray | None = None,
    ) -> list[dict[str, jnp.ndarray]]:
        profile_type = np.asarray(self.state.packed_lens_spec.profile_type, dtype=int)
        kwargs_lens: list[dict[str, jnp.ndarray]] = []
        indices = (
            component_indices.tolist()
            if component_indices is not None
            else list(range(len(profile_type)))
        )
        for idx in indices:
            idx = int(idx)
            profile = int(profile_type[idx])
            if profile == DP_IE_PROFILE:
                kwargs_lens.append(
                    {
                        "sigma0": packed_state["sigma0"][idx],
                        "Ra": packed_state["Ra"][idx],
                        "Rs": packed_state["Rs"][idx],
                        "e1": packed_state["e1"][idx],
                        "e2": packed_state["e2"][idx],
                        "center_x": packed_state["center_x"][idx],
                        "center_y": packed_state["center_y"][idx],
                    }
                )
            elif profile == SHEAR_PROFILE:
                kwargs_lens.append(
                    {
                        "gamma1": packed_state["gamma1"][idx],
                        "gamma2": packed_state["gamma2"][idx],
                    }
                )
            else:  # pragma: no cover
                raise ValueError(f"Unsupported profile type {profile}.")
        return kwargs_lens

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
                    "selection_mode",
                    "adaptive_cumulative_count",
                    "adaptive_knee_count",
                    "cumulative_importance_fraction",
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
            if self.active_scaling_selection == "adaptive":
                top_k, cumulative_count, knee_count = _adaptive_active_scaling_count(
                    importance[order],
                    cumulative_fraction=self.active_scaling_cumulative_fraction,
                    min_count=self.active_scaling_min,
                    max_count=requested_active_count,
                )
            else:
                top_k = min(requested_active_count, len(order))
                cumulative_count = top_k
                knee_count = top_k
            active_positions = set(order[:top_k].tolist())
            ordered_importance = importance[order]
            total_importance = float(np.sum(ordered_importance[np.isfinite(ordered_importance)]))
            cumulative_importance = np.cumsum(np.nan_to_num(ordered_importance, nan=0.0, posinf=0.0, neginf=0.0))
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
                        "selection_mode": self.active_scaling_selection,
                        "adaptive_cumulative_count": int(cumulative_count),
                        "adaptive_knee_count": int(knee_count),
                        "cumulative_importance_fraction": float(cumulative_importance[rank - 1] / total_importance)
                        if total_importance > 0.0
                        else float("nan"),
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
        return _dpie_sigma0_factor_from_config(self.state.z_lens, float(z_source), self.cosmo_config)

    def _cosmology_parameters_from_physical(self, physical_params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if not self.fit_cosmology_flat_wcdm:
            return (
                jnp.asarray(self.cosmology_fiducial_om0, dtype=jnp.float64),
                jnp.asarray(self.cosmology_fiducial_w0, dtype=jnp.float64),
            )
        om0 = jnp.take(physical_params, jnp.asarray(self.cosmology_om0_param_index, dtype=jnp.int32))
        w0 = jnp.take(physical_params, jnp.asarray(self.cosmology_w0_param_index, dtype=jnp.int32))
        return om0, w0

    def _cosmology_lens_chi_mpc(self, physical_params: jnp.ndarray) -> jnp.ndarray:
        om0, w0 = self._cosmology_parameters_from_physical(physical_params)
        return _jax_flat_wcdm_comoving_distance_mpc(
            self.state.z_lens,
            self.cosmology_h0,
            om0,
            w0,
        )

    def _kpc_per_arcsec_for_physical(self, physical_params: jnp.ndarray) -> jnp.ndarray:
        if not self.fit_cosmology_flat_wcdm:
            return jnp.asarray(self.kpc_per_arcsec, dtype=jnp.float64)
        om0, w0 = self._cosmology_parameters_from_physical(physical_params)
        return _jax_flat_wcdm_kpc_per_arcsec(
            self.state.z_lens,
            self.cosmology_h0,
            om0,
            w0,
        )

    def _sampled_cosmology_geometry_for_physical(
        self,
        physical_params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        om0, w0 = self._cosmology_parameters_from_physical(physical_params)
        kpc_per_arcsec, _efficiency, dpie_sigma0_factors = _jax_flat_wcdm_lens_geometry_factors(
            self.state.z_lens,
            self.cosmology_effective_z_values_jax,
            self.cosmology_h0,
            om0,
            w0,
        )
        return kpc_per_arcsec, dpie_sigma0_factors

    def _dpie_sigma0_factor_for_physical_z_source(
        self,
        physical_params: jnp.ndarray,
        z_source: float,
    ) -> jnp.ndarray:
        if not self.fit_cosmology_flat_wcdm:
            return jnp.asarray(self._dpie_sigma0_factor_for_z_source(z_source), dtype=jnp.float64)
        om0, w0 = self._cosmology_parameters_from_physical(physical_params)
        return _jax_dpie_sigma0_factor(
            self.state.z_lens,
            z_source,
            self.cosmology_h0,
            om0,
            w0,
        )

    def _build_packed_lens_state_details(
        self,
        params: jnp.ndarray,
        z_source: float,
    ) -> tuple[dict[str, Any], dict[str, jnp.ndarray]]:
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
            transform_kind_affine_mask = jnp.asarray(transform_kind_array == "affine", dtype=bool)
            transform_scale_array = jnp.asarray(
                [float(getattr(spec_item, "transform_scale", 1.0)) for spec_item in self.state.parameter_specs],
                dtype=jnp.float64,
            )
        else:
            transform_kind_log_offset_positive_mask = self.transform_kind_log_offset_positive_mask
            transform_offset_array = self.transform_offset_array
            transform_kind_affine_mask = self.transform_kind_affine_mask
            transform_scale_array = self.transform_scale_array
        physical_params = _apply_parameter_transforms_jax(
            params,
            transform_kind_log_positive_mask,
            transform_kind_log_offset_positive_mask,
            transform_offset_array,
            transform_kind_affine_mask,
            transform_scale_array,
        )
        return self._build_packed_lens_state_details_from_physical(physical_params, z_source)

    def _build_packed_lens_state_details_from_physical(
        self,
        physical_params: jnp.ndarray,
        z_source: float,
        *,
        kpc_per_arcsec: jnp.ndarray | None = None,
        dpie_sigma0_factor: jnp.ndarray | None = None,
    ) -> tuple[dict[str, Any], dict[str, jnp.ndarray]]:
        spec = self.state.packed_lens_spec
        spec_jax = self.packed_spec_jax
        x_center = self._apply_param_updates(spec_jax["x_center_base"], spec_jax["x_center_param_index"], physical_params)
        y_center = self._apply_param_updates(spec_jax["y_center_base"], spec_jax["y_center_param_index"], physical_params)
        ellipticite = self._apply_param_updates(
            spec_jax["ellipticite_base"], spec_jax["ellipticite_param_index"], physical_params
        )
        angle_pos = self._apply_param_updates(
            spec_jax["angle_pos_base"], spec_jax["angle_pos_param_index"], physical_params
        )
        core_radius_kpc = self._apply_param_updates(
            spec_jax["core_radius_kpc_base"], spec_jax["core_radius_param_index"], physical_params
        )
        cut_radius_kpc = self._apply_param_updates(
            spec_jax["cut_radius_kpc_base"], spec_jax["cut_radius_param_index"], physical_params
        )
        v_disp = self._apply_param_updates(spec_jax["v_disp_base"], spec_jax["v_disp_param_index"], physical_params)
        gamma = self._apply_param_updates(spec_jax["gamma_base"], spec_jax["gamma_param_index"], physical_params)

        profile_type = spec_jax["profile_type"]
        component_family = spec_jax["component_family"]
        is_dpie = profile_type == DP_IE_PROFILE
        is_shear = profile_type == SHEAR_PROFILE
        is_scaling = component_family == 1

        luminosity_ratio = spec_jax["luminosity_ratio"]
        sigma_ref = self._apply_param_updates(
            spec_jax["sigma_ref_base"], spec_jax["sigma_ref_param_index"], physical_params
        )
        cut_ref = self._apply_param_updates(
            spec_jax["cut_ref_base"], spec_jax["cut_ref_param_index"], physical_params
        )
        core_ref = self._apply_param_updates(
            spec_jax["core_ref_base"], spec_jax["core_ref_param_index"], physical_params
        )
        vdslope = self._apply_param_updates(
            spec_jax["vdslope_base"], spec_jax["vdslope_param_index"], physical_params
        )
        slope = self._apply_param_updates(
            spec_jax["slope_base"], spec_jax["slope_param_index"], physical_params
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
        if kpc_per_arcsec is None:
            kpc_per_arcsec = self._kpc_per_arcsec_for_physical(physical_params)
        kpc_per_arcsec = jnp.asarray(kpc_per_arcsec, dtype=jnp.float64)
        ra_raw = core_radius_kpc / kpc_per_arcsec
        rs_raw = cut_radius_kpc / kpc_per_arcsec
        ra = jnp.maximum(ra_raw, SAFE_RADIUS_MARGIN_ARCSEC)
        rs = jnp.maximum(rs_raw, ra + SAFE_RADIUS_MARGIN_ARCSEC)
        if dpie_sigma0_factor is None:
            factor_array = self._dpie_sigma0_factor_for_physical_z_source(physical_params, z_source)
        else:
            factor_array = jnp.asarray(dpie_sigma0_factor, dtype=jnp.float64)
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
                jnp.any(details["is_dpie"] & (~jnp.isfinite(details["v_disp"]) | (details["v_disp"] <= 0.0))),
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
        packed_state, details = self._build_packed_lens_state_details(params, z_source)
        return packed_state, self._packed_lens_validity(details)

    def _build_packed_lens_state_with_validity_from_physical(
        self,
        physical_params: jnp.ndarray,
        z_source: float,
        *,
        stop_gradient: bool,
        kpc_per_arcsec: jnp.ndarray | None = None,
        dpie_sigma0_factor: jnp.ndarray | None = None,
    ) -> tuple[dict[str, Any], dict[str, jnp.ndarray]]:
        packed_state, details = self._build_packed_lens_state_details_from_physical(
            physical_params,
            z_source,
            kpc_per_arcsec=kpc_per_arcsec,
            dpie_sigma0_factor=dpie_sigma0_factor,
        )
        validity = self._stopped_packed_lens_validity(details) if stop_gradient else self._packed_lens_validity(details)
        return packed_state, validity

    def _build_packed_lens_state(
        self,
        params: jnp.ndarray,
        z_source: float,
        *,
        sigma_log_offset: float | jnp.ndarray = 0.0,
        core_log_offset: float | jnp.ndarray = 0.0,
        cut_log_offset: float | jnp.ndarray = 0.0,
    ) -> dict[str, Any]:
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
            transform_kind_affine_mask = jnp.asarray(transform_kind_array == "affine", dtype=bool)
            transform_scale_array = jnp.asarray(
                [float(getattr(spec_item, "transform_scale", 1.0)) for spec_item in self.state.parameter_specs],
                dtype=jnp.float64,
            )
        else:
            transform_kind_log_offset_positive_mask = self.transform_kind_log_offset_positive_mask
            transform_offset_array = self.transform_offset_array
            transform_kind_affine_mask = self.transform_kind_affine_mask
            transform_scale_array = self.transform_scale_array
        physical_params = _apply_parameter_transforms_jax(
            params,
            transform_kind_log_positive_mask,
            transform_kind_log_offset_positive_mask,
            transform_offset_array,
            transform_kind_affine_mask,
            transform_scale_array,
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
        scaled_vdisp = scaled_vdisp * jnp.exp(jnp.asarray(sigma_log_offset, dtype=jnp.float64))
        scaled_core = scaled_core * jnp.exp(jnp.asarray(core_log_offset, dtype=jnp.float64))
        scaled_cut = scaled_cut * jnp.exp(jnp.asarray(cut_log_offset, dtype=jnp.float64))
        v_disp = jnp.where(is_scaling, scaled_vdisp, v_disp)
        core_radius_kpc = jnp.where(is_scaling, scaled_core, core_radius_kpc)
        cut_radius_kpc = jnp.where(is_scaling, scaled_cut, cut_radius_kpc)

        q = jnp.maximum(1.0e-3, 1.0 - ellipticite)
        phi = jnp.deg2rad(angle_pos)
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kpc_per_arcsec = self._kpc_per_arcsec_for_physical(physical_params)
        ra_raw = core_radius_kpc / kpc_per_arcsec
        rs_raw = cut_radius_kpc / kpc_per_arcsec
        ra = jnp.maximum(ra_raw, SAFE_RADIUS_MARGIN_ARCSEC)
        rs = jnp.maximum(rs_raw, ra + SAFE_RADIUS_MARGIN_ARCSEC)
        factor_array = self._dpie_sigma0_factor_for_physical_z_source(physical_params, z_source)
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

    def _lensing_jacobian_for_components(
        self,
        z_source: float,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: dict[str, Any],
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if not self.use_bulk_ray_shooting:
            raise RuntimeError("Analytic image-plane Jacobians require LensModelBulk.hessian.")
        model = self.models_by_effective_z[z_source]
        kwargs = self._bulk_ray_shooting_kwargs_from_indices(packed_state, component_indices)
        f_xx, f_xy, f_yx, f_yy = model.hessian(x, y, kwargs)
        return 1.0 - f_xx, -f_xy, -f_yx, 1.0 - f_yy

    def _linearized_image_plane_residuals_for_components(
        self,
        z_source: float,
        x_obs: jnp.ndarray,
        y_obs: jnp.ndarray,
        beta_family_x: jnp.ndarray,
        beta_family_y: jnp.ndarray,
        packed_state: dict[str, Any],
        initial_jacobian_entries: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        current_x = x_obs
        current_y = y_obs
        total_dx = jnp.zeros_like(x_obs)
        total_dy = jnp.zeros_like(y_obs)
        finite = jnp.ones_like(x_obs, dtype=bool)
        for _step in range(1 + self.image_plane_newton_steps):
            beta_x, beta_y = self._ray_shooting_for_components(z_source, current_x, current_y, packed_state)
            f_x = beta_x - beta_family_x
            f_y = beta_y - beta_family_y
            if _step == 0 and initial_jacobian_entries is not None:
                jac_a00, jac_a01, jac_a10, jac_a11 = initial_jacobian_entries
            else:
                jac_a00, jac_a01, jac_a10, jac_a11 = self._lensing_jacobian_for_components(
                    z_source,
                    current_x,
                    current_y,
                    packed_state,
                )
            delta_x, delta_y, step_finite = _linearized_image_plane_residual_from_jacobian(
                f_x,
                f_y,
                jac_a00,
                jac_a01,
                jac_a10,
                jac_a11,
            )
            current_x = current_x + delta_x
            current_y = current_y + delta_y
            total_dx = total_dx + delta_x
            total_dy = total_dy + delta_y
            finite = finite & step_finite
        return total_dx, total_dy, finite

    def _inactive_fd_step(self, spec: ParameterSpec) -> float:
        if spec.prior_kind == "normal" and spec.std is not None:
            return max(abs(float(spec.std)) * self.refresh_param_drift_frac, 1.0e-4)
        return max(abs(spec.upper - spec.lower) * self.refresh_param_drift_frac, abs(spec.step), 1.0e-4)

    def _replace_surrogate_params(
        self,
        reference_params: jnp.ndarray,
        surrogate_params: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.surrogate_param_indices.size == 0:
            return reference_params
        return reference_params.at[self.surrogate_param_indices_jax].set(surrogate_params)

    def _inactive_surrogate_concat(
        self,
        surrogate_params: jnp.ndarray,
        reference_params: jnp.ndarray,
        z_source: float,
        x_obs: jnp.ndarray,
        y_obs: jnp.ndarray,
        include_jacobian: bool = False,
    ) -> jnp.ndarray:
        params = self._replace_surrogate_params(reference_params, surrogate_params)
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
        pieces = [alpha_x, alpha_y]
        if include_jacobian:
            inactive_jacobian = jax.lax.cond(
                validity["is_valid"],
                lambda current_state: self._lensing_jacobian_for_components(
                    z_source,
                    x_obs,
                    y_obs,
                    current_state,
                    self.inactive_scaling_component_indices,
                ),
                lambda _: (
                    jnp.ones_like(x_obs),
                    jnp.zeros_like(x_obs),
                    jnp.zeros_like(y_obs),
                    jnp.ones_like(y_obs),
                ),
                packed_state,
            )
            jac_a00, jac_a01, jac_a10, jac_a11 = inactive_jacobian
            pieces.extend([jac_a00 - 1.0, jac_a01, jac_a10, jac_a11 - 1.0])
        return jnp.concatenate(pieces)

    def _inactive_alpha_concat(
        self,
        surrogate_params: jnp.ndarray,
        reference_params: jnp.ndarray,
        z_source: float,
        x_obs: jnp.ndarray,
        y_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        return self._inactive_surrogate_concat(
            surrogate_params,
            reference_params,
            z_source,
            x_obs,
            y_obs,
            include_jacobian=False,
        )

    def _build_inactive_surrogate_jacobian(
        self,
        reference_params: np.ndarray,
        effective_z_source: float,
        x_obs: jnp.ndarray,
        y_obs: jnp.ndarray,
        include_jacobian: bool = False,
    ) -> dict[str, np.ndarray] | None:
        reference = np.asarray(reference_params, dtype=float)
        reference_jax = jnp.asarray(reference, dtype=jnp.float64)
        surrogate_reference = reference[self.surrogate_param_indices]
        surrogate_reference_jax = jnp.asarray(surrogate_reference, dtype=jnp.float64)
        inactive_surrogate_concat = np.asarray(
            self._inactive_surrogate_concat(
                surrogate_reference_jax,
                reference_jax,
                effective_z_source,
                x_obs,
                y_obs,
                include_jacobian=include_jacobian,
            ),
            dtype=float,
        )
        if not np.isfinite(inactive_surrogate_concat).all():
            return None

        n_obs = len(np.asarray(x_obs))
        n_surrogate = len(self.surrogate_param_indices)
        n_blocks = 6 if include_jacobian else 2
        deriv = np.zeros((n_blocks * n_obs, n_surrogate), dtype=float)
        x_obs_np = np.asarray(x_obs, dtype=float)
        y_obs_np = np.asarray(y_obs, dtype=float)

        for local_index, param_index in enumerate(self.surrogate_param_indices.tolist()):
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
                    self._inactive_surrogate_concat(
                        jnp.asarray(plus_theta[self.surrogate_param_indices], dtype=jnp.float64),
                        jnp.asarray(plus_theta, dtype=jnp.float64),
                        effective_z_source,
                        x_obs,
                        y_obs,
                        include_jacobian=include_jacobian,
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
                    self._inactive_surrogate_concat(
                        jnp.asarray(minus_theta[self.surrogate_param_indices], dtype=jnp.float64),
                        jnp.asarray(minus_theta, dtype=jnp.float64),
                        effective_z_source,
                        x_obs,
                        y_obs,
                        include_jacobian=include_jacobian,
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
                deriv[:, local_index] = (plus_eval - inactive_surrogate_concat) / delta
            elif minus_eval is not None:
                delta = float(theta0 - minus_theta[param_index])
                if delta <= 0.0:
                    return None
                deriv[:, local_index] = (inactive_surrogate_concat - minus_eval) / delta
            else:
                return None

        result: dict[str, np.ndarray] = {
            "inactive_alpha_x": inactive_surrogate_concat[:n_obs],
            "inactive_alpha_y": inactive_surrogate_concat[n_obs : 2 * n_obs],
            "inactive_alpha_dx_dparams": deriv[:n_obs, :].T,
            "inactive_alpha_dy_dparams": deriv[n_obs : 2 * n_obs, :].T,
        }
        if include_jacobian:
            result.update(
                {
                    "inactive_jacobian_delta_a00": inactive_surrogate_concat[2 * n_obs : 3 * n_obs],
                    "inactive_jacobian_delta_a01": inactive_surrogate_concat[3 * n_obs : 4 * n_obs],
                    "inactive_jacobian_delta_a10": inactive_surrogate_concat[4 * n_obs : 5 * n_obs],
                    "inactive_jacobian_delta_a11": inactive_surrogate_concat[5 * n_obs : 6 * n_obs],
                    "inactive_jacobian_delta_da00_dparams": deriv[2 * n_obs : 3 * n_obs, :].T,
                    "inactive_jacobian_delta_da01_dparams": deriv[3 * n_obs : 4 * n_obs, :].T,
                    "inactive_jacobian_delta_da10_dparams": deriv[4 * n_obs : 5 * n_obs, :].T,
                    "inactive_jacobian_delta_da11_dparams": deriv[5 * n_obs : 6 * n_obs, :].T,
                }
            )

        if not (
            all(np.isfinite(value).all() for value in result.values())
            and np.isfinite(x_obs_np).all()
            and np.isfinite(y_obs_np).all()
        ):
            return None
        return result

    def refresh_surrogate(self, reference_params: np.ndarray, reason: str = "manual") -> None:
        if not self.surrogate_enabled:
            return
        reference = np.asarray(reference_params, dtype=float)
        self.surrogate_reference_params = reference.copy()
        self.surrogate_reference_param_values = reference[self.surrogate_param_indices].copy()
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
                self.surrogate_reference_param_values = np.zeros(len(self.surrogate_param_indices), dtype=float)
                return
            inactive_surrogate = self._build_inactive_surrogate_jacobian(
                reference,
                bin_data.effective_z_source,
                x_obs,
                y_obs,
                include_jacobian=(
                    self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
                ),
            )
            if inactive_surrogate is None:
                self.surrogate_cache_by_z = {}
                self.surrogate_reference_params = None
                self.surrogate_reference_param_values = np.zeros(len(self.surrogate_param_indices), dtype=float)
                return
            self.surrogate_cache_by_z[bin_data.effective_z_source] = SurrogateBinCache(
                effective_z_source=float(bin_data.effective_z_source),
                **inactive_surrogate,
            )
        self.full_refresh_count += 1
        self._source_loglike_fn = jax.jit(self._source_loglike_impl)

    def _surrogate_needs_refresh(self, params: np.ndarray) -> bool:
        if not self.surrogate_enabled or self.surrogate_reference_params is None:
            return False
        current = np.asarray(params, dtype=float)
        for param_index in self.surrogate_param_indices.tolist():
            spec = self.state.parameter_specs[int(param_index)]
            scale = self._inactive_fd_step(spec)
            if abs(current[param_index] - self.surrogate_reference_params[param_index]) > scale:
                return True
        return False

    def _surrogate_parameter_delta(self, params: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(params, self.surrogate_param_indices_jax) - jnp.asarray(
            self.surrogate_reference_param_values,
            dtype=jnp.float64,
        )

    def _cached_surrogate_vector(
        self,
        base: np.ndarray | None,
        derivative: np.ndarray | None,
        delta: jnp.ndarray,
        *,
        field_name: str,
    ) -> jnp.ndarray:
        if base is None or derivative is None:
            raise RuntimeError(
                f"Surrogate cache is missing {field_name}; refresh the stage-specific surrogate first."
            )
        return jnp.asarray(base, dtype=jnp.float64) + jnp.tensordot(
            delta,
            jnp.asarray(derivative, dtype=jnp.float64),
            axes=1,
        )

    def _surrogate_inactive_alpha(
        self,
        params: jnp.ndarray,
        cache: SurrogateBinCache,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        delta = self._surrogate_parameter_delta(params)
        inactive_alpha_x = self._cached_surrogate_vector(
            cache.inactive_alpha_x,
            cache.inactive_alpha_dx_dparams,
            delta,
            field_name="inactive_alpha_x",
        )
        inactive_alpha_y = self._cached_surrogate_vector(
            cache.inactive_alpha_y,
            cache.inactive_alpha_dy_dparams,
            delta,
            field_name="inactive_alpha_y",
        )
        return inactive_alpha_x, inactive_alpha_y

    def _surrogate_jacobian_entries(
        self,
        params: jnp.ndarray,
        bin_data: TracedBinData,
        packed_state: dict[str, Any],
        invalid: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x_obs = bin_data.x_obs
        y_obs = bin_data.y_obs
        active_jacobian_entries = jax.lax.cond(
            invalid,
            lambda _: (
                jnp.ones_like(x_obs),
                jnp.zeros_like(x_obs),
                jnp.zeros_like(y_obs),
                jnp.ones_like(y_obs),
            ),
            lambda current_state: self._lensing_jacobian_for_components(
                bin_data.effective_z_source,
                x_obs,
                y_obs,
                current_state,
                self.active_component_indices,
            ),
            packed_state,
        )
        cache = self.surrogate_cache_by_z[bin_data.effective_z_source]
        delta = self._surrogate_parameter_delta(params)
        inactive_delta_a00 = self._cached_surrogate_vector(
            cache.inactive_jacobian_delta_a00,
            cache.inactive_jacobian_delta_da00_dparams,
            delta,
            field_name="inactive_jacobian_delta_a00",
        )
        inactive_delta_a01 = self._cached_surrogate_vector(
            cache.inactive_jacobian_delta_a01,
            cache.inactive_jacobian_delta_da01_dparams,
            delta,
            field_name="inactive_jacobian_delta_a01",
        )
        inactive_delta_a10 = self._cached_surrogate_vector(
            cache.inactive_jacobian_delta_a10,
            cache.inactive_jacobian_delta_da10_dparams,
            delta,
            field_name="inactive_jacobian_delta_a10",
        )
        inactive_delta_a11 = self._cached_surrogate_vector(
            cache.inactive_jacobian_delta_a11,
            cache.inactive_jacobian_delta_da11_dparams,
            delta,
            field_name="inactive_jacobian_delta_a11",
        )
        active_a00, active_a01, active_a10, active_a11 = active_jacobian_entries
        return (
            active_a00 + inactive_delta_a00,
            active_a01 + inactive_delta_a01,
            active_a10 + inactive_delta_a10,
            active_a11 + inactive_delta_a11,
        )

    def _linearized_image_plane_residuals_from_observed_beta(
        self,
        observed_beta_x: jnp.ndarray,
        observed_beta_y: jnp.ndarray,
        beta_family_x: jnp.ndarray,
        beta_family_y: jnp.ndarray,
        jacobian_entries: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return _linearized_image_plane_residual_from_jacobian(
            observed_beta_x - beta_family_x,
            observed_beta_y - beta_family_y,
            *jacobian_entries,
        )

    def _surrogate_beta(
        self,
        params: jnp.ndarray,
        physical_params: jnp.ndarray,
        bin_data: TracedBinData,
        *,
        kpc_per_arcsec: jnp.ndarray | None = None,
        dpie_sigma0_factor: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, Any]]:
        packed_state, validity = self._build_packed_lens_state_with_validity_from_physical(
            physical_params,
            bin_data.effective_z_source,
            stop_gradient=True,
            kpc_per_arcsec=kpc_per_arcsec,
            dpie_sigma0_factor=dpie_sigma0_factor,
        )
        self._maybe_record_invalid_state(validity)
        x_obs = bin_data.x_obs
        y_obs = bin_data.y_obs
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
        inactive_alpha_x, inactive_alpha_y = self._surrogate_inactive_alpha(params, cache)
        beta_x = x_obs - active_alpha_x - inactive_alpha_x
        beta_y = y_obs - active_alpha_y - inactive_alpha_y
        return beta_x, beta_y, invalid, packed_state

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
        physical_params = self._physical_parameter_vector(jnp.asarray(params, dtype=jnp.float64))
        source_sigma_int = self._source_sigma_int_from_physical(physical_params)
        image_sigma_int = self._image_sigma_int_from_physical(physical_params)
        sampled_kpc_per_arcsec = None
        sampled_dpie_sigma0_factors = None
        if bool(getattr(self, "fit_cosmology_flat_wcdm", False)):
            sampled_kpc_per_arcsec, sampled_dpie_sigma0_factors = self._sampled_cosmology_geometry_for_physical(
                physical_params
            )
        for bin_data in self.traced_bin_data:
            x_obs = bin_data.x_obs
            y_obs = bin_data.y_obs
            bin_kpc_per_arcsec = sampled_kpc_per_arcsec
            bin_dpie_sigma0_factor = None
            if sampled_dpie_sigma0_factors is not None and bin_data.effective_z_index >= 0:
                bin_dpie_sigma0_factor = jnp.take(
                    sampled_dpie_sigma0_factors,
                    jnp.asarray(bin_data.effective_z_index, dtype=jnp.int32),
                )
            if self.surrogate_enabled and self.surrogate_cache_by_z:
                if bin_dpie_sigma0_factor is None:
                    beta_x, beta_y, invalid, packed_state = self._surrogate_beta(params, physical_params, bin_data)
                else:
                    beta_x, beta_y, invalid, packed_state = self._surrogate_beta(
                        params,
                        physical_params,
                        bin_data,
                        kpc_per_arcsec=bin_kpc_per_arcsec,
                        dpie_sigma0_factor=bin_dpie_sigma0_factor,
                    )
            else:
                if bin_dpie_sigma0_factor is None:
                    packed_state, validity = self._build_packed_lens_state_with_validity_from_physical(
                        physical_params,
                        bin_data.effective_z_source,
                        stop_gradient=True,
                    )
                else:
                    packed_state, validity = self._build_packed_lens_state_with_validity_from_physical(
                        physical_params,
                        bin_data.effective_z_source,
                        stop_gradient=True,
                        kpc_per_arcsec=bin_kpc_per_arcsec,
                        dpie_sigma0_factor=bin_dpie_sigma0_factor,
                    )
                self._maybe_record_invalid_state(validity)
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
            family_idx = bin_data.family_index_per_image
            sigma_base = bin_data.sigma_per_image
            scatter_var_x, scatter_var_y = self._scaling_scatter_extra_variance_from_physical(
                physical_params,
                bin_data,
                beta_x,
                beta_y,
            )
            n_families = bin_data.n_families
            image_has_constraint = bin_data.image_has_constraint

            reliability = jnp.clip(bin_data.reliability_per_image, 1.0e-6, 1.0 - 1.0e-6)

            if self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE:
                observed_jacobian_entries = self._lensing_jacobian_for_components(
                    bin_data.effective_z_source,
                    x_obs,
                    y_obs,
                    packed_state,
                )
                bin_loglike = _linearized_marginal_beta_image_plane_bin_loglike(
                    beta_x=beta_x,
                    beta_y=beta_y,
                    jacobian_entries=observed_jacobian_entries,
                    family_idx=family_idx,
                    n_families=n_families,
                    sigma_per_image=sigma_base,
                    image_has_constraint=image_has_constraint,
                    image_sigma_int=image_sigma_int,
                    covariance_floor=self.source_plane_covariance_floor,
                    source_prior_mean_x=self.evidence_source_prior_mean_x_arcsec,
                    source_prior_mean_y=self.evidence_source_prior_mean_y_arcsec,
                    source_prior_sigma_arcsec=float(self.evidence_source_prior_sigma_arcsec or 1.0),
                )
            elif self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE:
                observed_beta_x = beta_x
                observed_beta_y = beta_y
                if self.surrogate_enabled and self.surrogate_cache_by_z:
                    observed_jacobian_entries = self._surrogate_jacobian_entries(
                        jnp.asarray(params, dtype=jnp.float64),
                        bin_data,
                        packed_state,
                        invalid,
                    )
                else:
                    observed_jacobian_entries = self._lensing_jacobian_for_components(
                        bin_data.effective_z_source,
                        x_obs,
                        y_obs,
                        packed_state,
                    )
                beta_family_x, beta_family_y, has_source_positions, source_transport_correction = self._explicit_source_position_vectors_for_bin(
                    jnp.asarray(params, dtype=jnp.float64),
                    physical_params,
                    bin_data,
                    observed_beta_x,
                    observed_beta_y,
                    image_sigma_int,
                    observed_jacobian_entries,
                )
                if self.image_plane_newton_steps == 0:
                    residual_x, residual_y, residual_finite = self._linearized_image_plane_residuals_from_observed_beta(
                        observed_beta_x,
                        observed_beta_y,
                        beta_family_x,
                        beta_family_y,
                        observed_jacobian_entries,
                    )
                else:
                    residual_x, residual_y, residual_finite = self._linearized_image_plane_residuals_for_components(
                        bin_data.effective_z_source,
                        x_obs,
                        y_obs,
                        beta_family_x,
                        beta_family_y,
                        packed_state,
                        initial_jacobian_entries=observed_jacobian_entries,
                    )
                invalid = invalid | (~has_source_positions) | (~jnp.all(residual_finite))
                bin_loglike = _linearized_image_plane_bin_loglike(
                    residual_x=residual_x,
                    residual_y=residual_y,
                    family_idx=family_idx,
                    n_families=n_families,
                    sigma_per_image=sigma_base,
                    reliability_per_image=reliability,
                    image_has_constraint=image_has_constraint,
                    image_sigma_int=image_sigma_int,
                    covariance_floor=self.source_plane_covariance_floor,
                    outlier_sigma_arcsec=self.source_plane_outlier_sigma_arcsec,
                    image_presence_penalty_weight=self.image_presence_penalty_weight,
                    image_presence_match_radius_arcsec=self.image_presence_match_radius_arcsec,
                    image_presence_temperature_arcsec=self.image_presence_temperature_arcsec,
                    image_presence_count_softness=self.image_presence_count_softness,
                    image_presence_count_margin=self.image_presence_count_margin,
                )
                bin_loglike = bin_loglike + source_transport_correction
            elif self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN:
                jac_a00, jac_a01, jac_a10, jac_a11 = self._source_metric_jacobian_entries(bin_data)
                bin_loglike = _local_jacobian_bin_loglike(
                    beta_x=beta_x,
                    beta_y=beta_y,
                    family_idx=family_idx,
                    n_families=n_families,
                    sigma_per_image=sigma_base,
                    reliability_per_image=reliability,
                    image_has_constraint=image_has_constraint,
                    source_sigma_int=source_sigma_int,
                    scatter_var_x=scatter_var_x,
                    scatter_var_y=scatter_var_y,
                    jac_a00=jac_a00,
                    jac_a01=jac_a01,
                    jac_a10=jac_a10,
                    jac_a11=jac_a11,
                    covariance_floor=self.source_plane_covariance_floor,
                    outlier_sigma_arcsec=self.source_plane_outlier_sigma_arcsec,
                )
            else:
                sigma2_image = jnp.square(sigma_base) * self._magnification_inv_abs_mu(bin_data)
                cov_floor = jnp.asarray(self.source_plane_covariance_floor, dtype=jnp.float64)
                sigma2_x = sigma2_image + jnp.square(source_sigma_int) + scatter_var_x + cov_floor
                sigma2_y = sigma2_image + jnp.square(source_sigma_int) + scatter_var_y + cov_floor
                sigma2_weight = 0.5 * (sigma2_x + sigma2_y)
                weights = reliability / jnp.maximum(sigma2_weight, 1.0e-18)
                sum_w = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(weights)
                sum_bx = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(weights * beta_x)
                sum_by = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(weights * beta_y)
                centroid_x = sum_bx / jnp.maximum(sum_w, 1.0e-18)
                centroid_y = sum_by / jnp.maximum(sum_w, 1.0e-18)
                dx = beta_x - centroid_x[family_idx]
                dy = beta_y - centroid_y[family_idx]
                family_ll = -0.5 * (
                    (dx**2) / sigma2_x
                    + (dy**2) / sigma2_y
                    + jnp.log(2.0 * jnp.pi * sigma2_x)
                    + jnp.log(2.0 * jnp.pi * sigma2_y)
                )
                outlier_sigma2 = jnp.square(jnp.asarray(self.source_plane_outlier_sigma_arcsec, dtype=jnp.float64))
                outlier_ll = -0.5 * (
                    (dx**2 + dy**2) / outlier_sigma2 + 2.0 * jnp.log(2.0 * jnp.pi * outlier_sigma2)
                )
                mixture_ll = jnp.logaddexp(jnp.log(reliability) + family_ll, jnp.log1p(-reliability) + outlier_ll)
                bin_loglike = jnp.sum(
                    jnp.where(
                        image_has_constraint,
                        mixture_ll,
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
        physical_params = self._physical_parameter_vector(params_jax)
        source_sigma_int = self._source_sigma_int_numpy(params)
        image_sigma_int = self._image_sigma_int_numpy(params)
        family_by_id = {str(family.family_id): family for family in self.state.family_data}

        def failed_prediction(family_id: str) -> dict[str, Any]:
            family = family_by_id.get(str(family_id))
            n_images = int(getattr(family, "n_images", 0))
            return {
                "source_x": float("nan"),
                "source_y": float("nan"),
                "source_plane_rms": float("nan"),
                "x_pred": np.full(n_images, np.nan),
                "y_pred": np.full(n_images, np.nan),
                "exact_image_rms": np.nan,
                "failed": True,
            }

        for bin_data in self.state.bin_data:
            if self.surrogate_enabled and self.surrogate_cache_by_z:
                traced_bin_data = self.traced_bin_data_by_z[float(bin_data.effective_z_source)]
                beta_x, beta_y, invalid, _packed_state = self._surrogate_beta(params_jax, physical_params, traced_bin_data)
                if bool(np.asarray(invalid, dtype=bool)):
                    for family_id in bin_data.family_ids:
                        summaries[str(family_id)] = failed_prediction(str(family_id))
                    continue
            else:
                packed_state = self._build_packed_lens_state(params_jax, bin_data.effective_z_source)
                validity = self._packed_lens_validity_from_params(params_jax, bin_data.effective_z_source, stop_gradient=False)
                if not bool(np.asarray(validity["is_valid"], dtype=bool)):
                    self._record_invalid_state_callback(np.asarray(validity["reason_flags"], dtype=bool))
                    for family_id in bin_data.family_ids:
                        summaries[str(family_id)] = failed_prediction(str(family_id))
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
                weights, sigma_eff = _validation_source_centroid_weights(
                    np.sum(mask),
                    family.sigma_arcsec,
                    source_sigma_int,
                    self.source_plane_covariance_floor,
                )
                sampled_source = self._source_position_for_family_numpy(params, str(family_id))
                if _sample_likelihood_uses_explicit_beta(self.sample_likelihood_mode) and sampled_source is not None:
                    source_x, source_y = sampled_source
                else:
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
                    "image_sigma_int_arcsec": image_sigma_int,
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

    def _get_jax_exact_model(self, z_source: float) -> LensModelBulk:
        model = self.jax_exact_models_by_z.get(z_source)
        if model is None:
            exact_model_setup_start = time.perf_counter()
            model = LensModelBulk(
                unique_lens_model_list=self.unique_lens_model_list,
                multi_plane=False,
                cosmo=self.cosmo,
            )
            self.jax_exact_models_by_z[z_source] = model
            self.timing_totals["exact_model_cache_setup"] += time.perf_counter() - exact_model_setup_start
        return model

    def _jax_exact_kwargs_lens(self, packed_state: dict[str, Any]) -> dict[str, Any]:
        return self._bulk_ray_shooting_kwargs_from_indices(packed_state)

    def _exact_source_ray_shooting(
        self,
        family: FamilyData,
        packed_state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        solver_mode = getattr(self, "exact_image_solver", EXACT_IMAGE_SOLVER_LENSTRONOMY)
        if solver_mode in {EXACT_IMAGE_SOLVER_AUTO, EXACT_IMAGE_SOLVER_JAX} and _jax_image_position_lenstronomy is not None:
            try:
                model = self._get_jax_exact_model(family.z_source)
                kwargs_lens = self._jax_exact_kwargs_lens(packed_state)
                beta_x, beta_y = model.ray_shooting(
                    jnp.asarray(family.x_obs, dtype=jnp.float64),
                    jnp.asarray(family.y_obs, dtype=jnp.float64),
                    kwargs_lens,
                )
                return np.asarray(beta_x, dtype=float), np.asarray(beta_y, dtype=float)
            except Exception:
                if solver_mode == EXACT_IMAGE_SOLVER_JAX:
                    raise
        model, _solver = self._get_exact_model_solver(family.z_source)
        kwargs_lens = self._packed_to_kwargs_lens(packed_state)
        beta_x, beta_y = model.ray_shooting(
            jnp.asarray(family.x_obs, dtype=jnp.float64),
            jnp.asarray(family.y_obs, dtype=jnp.float64),
            kwargs_lens,
        )
        return np.asarray(beta_x, dtype=float), np.asarray(beta_y, dtype=float)

    def _solve_exact_images_jax(
        self,
        family: FamilyData,
        packed_state: dict[str, Any],
        source_x: float,
        source_y: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if _jax_image_position_lenstronomy is None:
            raise RuntimeError("JAXtronomy lenstronomy_jax exact image solver is unavailable.")
        exact_start = time.perf_counter()
        try:
            model = self._get_jax_exact_model(family.z_source)
            kwargs_lens = self._jax_exact_kwargs_lens(packed_state)
            x_pred, y_pred = _jax_image_position_lenstronomy(
                model,
                source_x,
                source_y,
                kwargs_lens,
                min_distance=max(0.02, family.sigma_arcsec / 5.0),
                search_window=family.search_window,
                x_center=family.x_center,
                y_center=family.y_center,
                num_iter_max=200,
                precision_limit=1e-8,
                arrival_time_sort=False,
            )
            self.exact_jax_attempt_count = int(getattr(self, "exact_jax_attempt_count", 0)) + 1
            return np.asarray(x_pred, dtype=float), np.asarray(y_pred, dtype=float)
        finally:
            elapsed = time.perf_counter() - exact_start
            self.timing_totals["exact_solver"] = self.timing_totals.get("exact_solver", 0.0) + elapsed
            self.timing_totals["exact_solver_jax"] = self.timing_totals.get("exact_solver_jax", 0.0) + elapsed

    def _solve_exact_images_lenstronomy(
        self,
        family: FamilyData,
        packed_state: dict[str, Any],
        source_x: float,
        source_y: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        model, solver = self._get_exact_model_solver(family.z_source)
        kwargs_lens = self._packed_to_kwargs_lens(packed_state)
        exact_start = time.perf_counter()
        try:
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
            self.exact_lenstronomy_count = int(getattr(self, "exact_lenstronomy_count", 0)) + 1
            return np.asarray(x_pred, dtype=float), np.asarray(y_pred, dtype=float)
        finally:
            elapsed = time.perf_counter() - exact_start
            self.timing_totals["exact_solver"] = self.timing_totals.get("exact_solver", 0.0) + elapsed
            self.timing_totals["exact_solver_lenstronomy"] = self.timing_totals.get("exact_solver_lenstronomy", 0.0) + elapsed

    def _match_images(self, x_pred: np.ndarray, y_pred: np.ndarray, family: FamilyData) -> tuple[np.ndarray, np.ndarray] | None:
        match_details = self._image_match_diagnostics(x_pred, y_pred, family)
        if (
            int(match_details["produced_image_count"]) != family.n_images
            or int(match_details["recovered_image_count"]) != family.n_images
        ):
            return None
        ordered_x = np.asarray(match_details["_matched_x"], dtype=float)
        ordered_y = np.asarray(match_details["_matched_y"], dtype=float)
        if ordered_x.shape != (family.n_images,) or ordered_y.shape != (family.n_images,):
            return None
        if not np.all(np.isfinite(ordered_x) & np.isfinite(ordered_y)):
            return None
        return ordered_x, ordered_y

    def _image_match_diagnostics(self, x_pred: np.ndarray, y_pred: np.ndarray, family: FamilyData) -> dict[str, Any]:
        x_array = np.asarray(x_pred, dtype=float).reshape(-1)
        y_array = np.asarray(y_pred, dtype=float).reshape(-1)
        n_observed = int(family.n_images)
        n_produced = int(min(x_array.size, y_array.size))
        matched_x = np.full(n_observed, np.nan, dtype=float)
        matched_y = np.full(n_observed, np.nan, dtype=float)
        diagnostics: dict[str, Any] = {
            "produced_image_count": n_produced,
            "recovered_image_count": 0,
            "missing_image_count": n_observed,
            "extra_image_count": n_produced,
            "multiplicity_failed": True,
            "multiplicity_failure_reason": "no_model_images" if n_produced == 0 else "no_matches",
            "_matched_x": matched_x,
            "_matched_y": matched_y,
        }
        if x_array.size != y_array.size:
            diagnostics["produced_image_count"] = int(max(x_array.size, y_array.size))
            diagnostics["extra_image_count"] = int(max(x_array.size, y_array.size))
            diagnostics["multiplicity_failure_reason"] = "prediction_shape_mismatch"
            return diagnostics
        if n_observed == 0:
            diagnostics.update(
                {
                    "recovered_image_count": 0,
                    "missing_image_count": 0,
                    "extra_image_count": n_produced,
                    "multiplicity_failed": bool(n_produced != 0),
                    "multiplicity_failure_reason": "extra_model_images" if n_produced else "",
                }
            )
            return diagnostics
        if n_produced == 0:
            return diagnostics
        pred = np.column_stack([x_array, y_array])
        obs = np.column_stack([np.asarray(family.x_obs, dtype=float), np.asarray(family.y_obs, dtype=float)])
        if pred.shape[0] != n_produced or obs.shape[0] != n_observed:
            diagnostics["multiplicity_failure_reason"] = "prediction_shape_mismatch"
            return diagnostics
        finite_pred = np.isfinite(pred).all(axis=1)
        finite_obs = np.isfinite(obs).all(axis=1)
        cost = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=2)
        cost = np.where(finite_pred[:, None] & finite_obs[None, :], cost, np.inf)
        finite_cost = np.isfinite(cost)
        if not np.any(finite_cost):
            diagnostics["multiplicity_failure_reason"] = "nonfinite_prediction"
            return diagnostics
        finite_max = float(np.max(cost[finite_cost]))
        match_tolerance = float(getattr(self, "match_tolerance_arcsec", DEFAULT_MATCH_TOLERANCE))
        assignment_cost = np.where(finite_cost, cost, finite_max + match_tolerance + 1.0)
        row_ind, col_ind = linear_sum_assignment(assignment_cost)
        accepted = np.isfinite(cost[row_ind, col_ind]) & (cost[row_ind, col_ind] <= match_tolerance)
        recovered_count = int(np.sum(accepted))
        for pred_idx, obs_idx in zip(row_ind[accepted], col_ind[accepted]):
            matched_x[int(obs_idx)] = x_array[int(pred_idx)]
            matched_y[int(obs_idx)] = y_array[int(pred_idx)]
        missing_count = int(max(0, n_observed - recovered_count))
        extra_count = int(max(0, n_produced - recovered_count))
        if missing_count == 0 and extra_count == 0:
            reason = ""
        elif missing_count > 0 and extra_count > 0:
            reason = "missing_and_extra_model_images" if n_produced != n_observed else "match_tolerance_exceeded"
        elif missing_count > 0:
            reason = "missing_model_images"
        else:
            reason = "extra_model_images"
        diagnostics.update(
            {
                "recovered_image_count": recovered_count,
                "missing_image_count": missing_count,
                "extra_image_count": extra_count,
                "multiplicity_failed": bool(missing_count > 0 or extra_count > 0),
                "multiplicity_failure_reason": reason,
                "_matched_x": matched_x,
                "_matched_y": matched_y,
            }
        )
        return diagnostics

    def _record_exact_prediction_details(self, family_id: str, details: dict[str, Any]) -> None:
        if not hasattr(self, "_last_exact_prediction_details"):
            self._last_exact_prediction_details = {}
        self._last_exact_prediction_details[str(family_id)] = details

    def _exact_family_prediction_details(self, params: np.ndarray, family: FamilyData) -> dict[str, Any]:
        cache = self.validation_cache[family.family_id]
        base_details: dict[str, Any] = {
            "produced_image_count": np.nan,
            "recovered_image_count": np.nan,
            "missing_image_count": np.nan,
            "extra_image_count": np.nan,
            "multiplicity_failed": True,
            "multiplicity_failure_reason": "",
            "failed": True,
        }
        packed_state = self._build_packed_lens_state(jnp.asarray(params, dtype=jnp.float64), family.z_source)
        try:
            beta_x, beta_y = self._exact_source_ray_shooting(family, packed_state)
        except Exception:
            cache.multiplicity_mismatch_count += 1
            details = {**base_details, "multiplicity_failure_reason": "source_ray_shooting_failed"}
            self._record_exact_prediction_details(family.family_id, details)
            return details
        source_sigma_int = self._source_sigma_int_numpy(params)
        weights, _sigma_eff = _validation_source_centroid_weights(
            family.n_images,
            family.sigma_arcsec,
            source_sigma_int,
            self.source_plane_covariance_floor,
        )
        sampled_source = self._source_position_for_family_numpy(params, family.family_id)
        if _sample_likelihood_uses_explicit_beta(self.sample_likelihood_mode) and sampled_source is not None:
            source_x, source_y = sampled_source
        else:
            source_x = float(np.average(np.asarray(beta_x, dtype=float), weights=weights))
            source_y = float(np.average(np.asarray(beta_y, dtype=float), weights=weights))
        source_residuals = np.sqrt((np.asarray(beta_x, dtype=float) - source_x) ** 2 + (np.asarray(beta_y, dtype=float) - source_y) ** 2)
        cache.source_plane_rms = float(np.sqrt(np.mean(source_residuals**2)))
        cache.last_source_x = source_x
        cache.last_source_y = source_y
        cache.exact_validation_count += 1
        solver_mode = getattr(self, "exact_image_solver", EXACT_IMAGE_SOLVER_LENSTRONOMY)
        x_pred: np.ndarray
        y_pred: np.ndarray
        used_jax = False
        try:
            if solver_mode in {EXACT_IMAGE_SOLVER_AUTO, EXACT_IMAGE_SOLVER_JAX}:
                try:
                    x_pred, y_pred = self._solve_exact_images_jax(family, packed_state, source_x, source_y)
                    used_jax = True
                except Exception:
                    if solver_mode == EXACT_IMAGE_SOLVER_JAX:
                        cache.multiplicity_mismatch_count += 1
                        details = {**base_details, "multiplicity_failure_reason": "jax_exact_solver_failed"}
                        self._record_exact_prediction_details(family.family_id, details)
                        return details
                    x_pred, y_pred = self._solve_exact_images_lenstronomy(family, packed_state, source_x, source_y)
            else:
                x_pred, y_pred = self._solve_exact_images_lenstronomy(family, packed_state, source_x, source_y)
        except Exception:
            cache.multiplicity_mismatch_count += 1
            details = {**base_details, "multiplicity_failure_reason": "exact_image_solver_failed"}
            self._record_exact_prediction_details(family.family_id, details)
            return details

        match_details = self._image_match_diagnostics(np.asarray(x_pred), np.asarray(y_pred), family)
        matched = self._match_images(np.asarray(x_pred), np.asarray(y_pred), family)
        if matched is None and used_jax and solver_mode == EXACT_IMAGE_SOLVER_AUTO:
            self.exact_jax_fallback_count = int(getattr(self, "exact_jax_fallback_count", 0)) + 1
            try:
                x_pred, y_pred = self._solve_exact_images_lenstronomy(family, packed_state, source_x, source_y)
            except Exception:
                cache.multiplicity_mismatch_count += 1
                details = {**base_details, "multiplicity_failure_reason": "exact_image_solver_failed"}
                self._record_exact_prediction_details(family.family_id, details)
                return details
            match_details = self._image_match_diagnostics(np.asarray(x_pred), np.asarray(y_pred), family)
            matched = self._match_images(np.asarray(x_pred), np.asarray(y_pred), family)
        if matched is None:
            if len(x_pred) != family.n_images:
                cache.multiplicity_mismatch_count += 1
            else:
                cache.match_failure_count += 1
            details = {
                **base_details,
                **{key: value for key, value in match_details.items() if not str(key).startswith("_")},
                "failed": True,
            }
            self._record_exact_prediction_details(family.family_id, details)
            return details
        residuals = np.sqrt((matched[0] - family.x_obs) ** 2 + (matched[1] - family.y_obs) ** 2)
        rms = float(np.sqrt(np.mean(residuals**2)))
        cache.exact_image_rms = rms
        details = {
            **base_details,
            **{key: value for key, value in match_details.items() if not str(key).startswith("_")},
            "failed": False,
            "x_pred": matched[0],
            "y_pred": matched[1],
            "exact_image_rms": rms,
        }
        self._record_exact_prediction_details(family.family_id, details)
        return details

    def _exact_family_prediction(self, params: np.ndarray, family: FamilyData) -> tuple[np.ndarray, np.ndarray, float] | None:
        details = self._exact_family_prediction_details(params, family)
        if bool(details.get("failed", True)):
            return None
        return (
            np.asarray(details["x_pred"], dtype=float),
            np.asarray(details["y_pred"], dtype=float),
            float(details["exact_image_rms"]),
        )

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

    def evaluate(self, params: np.ndarray, validate_all_families: bool = False) -> EvaluationResult:
        if self.surrogate_enabled and self._surrogate_needs_refresh(np.asarray(params, dtype=float)):
            self.refresh_surrogate(np.asarray(params, dtype=float), reason="validation_drift")
            self.refresh_scaling_scatter_cache(np.asarray(params, dtype=float), reason="validation_drift")
            self.refresh_source_metric_cache(np.asarray(params, dtype=float), reason="validation_drift")
        source_loglike = self.source_loglike(params)
        family_predictions = self._family_source_summary(params)
        _ensure_family_prediction_image_arrays(self.state, family_predictions)
        if bool(getattr(self, "quick_diagnostics", False)):
            for prediction in family_predictions.values():
                prediction["approx_image_rms_arcsec"] = prediction.get("source_plane_rms")
                prediction["used_exact_refresh"] = False
                prediction["refresh_reason"] = "quick_diagnostics"
            return EvaluationResult(
                loglike=float(source_loglike),
                family_predictions=family_predictions,
                used_exact_validation=False,
            )
        validate_ids = {family.family_id for family in self.state.family_data} if validate_all_families else self.validation_family_ids
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
                details = getattr(self, "_last_exact_prediction_details", {}).get(family.family_id, {})
                family_predictions[family.family_id].update(
                    {key: value for key, value in details.items() if key in {
                        "produced_image_count",
                        "recovered_image_count",
                        "missing_image_count",
                        "extra_image_count",
                        "multiplicity_failed",
                        "multiplicity_failure_reason",
                    }}
                )
                _log(
                    None,
                    (
                        f"[validation:family] family={family.family_id} failed diagnostic_only=true "
                        f"elapsed={_fmt_seconds(time.time() - family_start)}"
                    ),
                )
            else:
                x_pred, y_pred, exact_rms = prediction
                family_predictions[family.family_id]["x_pred"] = x_pred
                family_predictions[family.family_id]["y_pred"] = y_pred
                family_predictions[family.family_id]["exact_image_rms"] = exact_rms
                details = getattr(self, "_last_exact_prediction_details", {}).get(family.family_id, {})
                family_predictions[family.family_id].update(
                    {key: value for key, value in details.items() if key in {
                        "produced_image_count",
                        "recovered_image_count",
                        "missing_image_count",
                        "extra_image_count",
                        "multiplicity_failed",
                        "multiplicity_failure_reason",
                    }}
                )
                _log(
                    None,
                    f"[validation:family] family={family.family_id} end elapsed={_fmt_seconds(time.time() - family_start)} exact_rms={exact_rms:.4f}",
                )
        return EvaluationResult(loglike=float(total_loglike), family_predictions=family_predictions, used_exact_validation=used_exact)

    def release_runtime_caches(self) -> None:
        self.models_by_effective_z = {}
        self.exact_models_by_z = {}
        self.exact_solvers_by_z = {}
        self.jax_exact_models_by_z = {}
        self.surrogate_cache_by_z = {}
        self.scaling_scatter_cache_by_z = {}
        self.source_metric_cache_by_z = {}
        self.surrogate_reference_params = None
        self.scaling_scatter_reference_params = None
        self.source_metric_reference_params = None
        if hasattr(self, "_conditional_source_inverse_basis_cache"):
            self._conditional_source_inverse_basis_cache.clear()
        gc.collect()


def _save_artifacts(
    artifacts_dir: Path,
    state: BuildState,
    args: argparse.Namespace,
    best_fit: np.ndarray,
    results: PosteriorResults,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    final_path = artifacts_dir / "plot_bundle.h5"
    tmp_path = artifacts_dir / ".plot_bundle.h5.tmp"
    try:
        _save_plot_bundle_h5(tmp_path, state, args, best_fit, results)
        os.replace(tmp_path, final_path)
    except BaseException:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _save_inference_checkpoint(
    args: argparse.Namespace,
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    run_dir: Path,
    best_fit: np.ndarray,
    posterior: PosteriorResults,
) -> tuple[Path, np.ndarray, PosteriorResults]:
    artifacts_dir = run_dir / "artifacts"
    _log(args, f"[output] saving artifacts to {artifacts_dir}")
    best_fit_physical = _run_logged_phase(
        args,
        "output.convert_best_fit_to_physical",
        lambda: _convert_theta_to_reported_physical(best_fit, state.parameter_specs, evaluator),
    )
    posterior_for_output = _run_logged_phase(
        args,
        "output.posterior_results_to_physical",
        lambda: _posterior_results_to_reported_physical(posterior, state.parameter_specs, evaluator),
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
    return artifacts_dir, best_fit_physical, posterior_for_output


def _coerce_prediction_image_array(value: Any, n_images: int) -> np.ndarray:
    array = np.asarray(value if value is not None else np.full(n_images, np.nan), dtype=float).reshape(-1)
    if array.shape == (n_images,):
        return array
    coerced = np.full(n_images, np.nan, dtype=float)
    count = min(n_images, array.size)
    if count:
        coerced[:count] = array[:count]
    return coerced


def _ensure_family_prediction_image_arrays(state: BuildState, family_predictions: dict[str, dict[str, Any]]) -> None:
    for family in state.family_data:
        key = str(family.family_id)
        prediction = family_predictions.get(key, family_predictions.get(family.family_id))
        if prediction is None:
            prediction = {
                "source_x": float("nan"),
                "source_y": float("nan"),
                "source_plane_rms": float("nan"),
                "failed": True,
            }
            family_predictions[key] = prediction
        n_images = int(getattr(family, "n_images", len(getattr(family, "x_obs", []))))
        prediction["x_pred"] = _coerce_prediction_image_array(prediction.get("x_pred"), n_images)
        prediction["y_pred"] = _coerce_prediction_image_array(prediction.get("y_pred"), n_images)


def _approximate_evaluation(evaluator: ClusterJAXEvaluator, params: np.ndarray) -> EvaluationResult:
    loglike = float(evaluator.source_loglike(params))
    family_predictions = evaluator._family_source_summary(params)
    _ensure_family_prediction_image_arrays(evaluator.state, family_predictions)
    for prediction in family_predictions.values():
        prediction["approx_image_rms_arcsec"] = prediction.get("source_plane_rms")
        prediction["used_exact_refresh"] = False
        prediction["refresh_reason"] = "quick_diagnostics" if bool(getattr(evaluator, "quick_diagnostics", False)) else "validation_skipped"
    return EvaluationResult(
        loglike=loglike,
        family_predictions=family_predictions,
        used_exact_validation=False,
    )


def _finite_prediction_values(best_eval: EvaluationResult, key: str) -> np.ndarray:
    values: list[float] = []
    for info in best_eval.family_predictions.values():
        if key == "exact_image_rms" and info.get("failed"):
            continue
        try:
            value = float(info.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _validation_metrics_summary(best_eval: EvaluationResult) -> str:
    fields: list[str] = [f"validated_families={len(best_eval.family_predictions)}"]
    exact_rms = _finite_prediction_values(best_eval, "exact_image_rms")
    if exact_rms.size:
        fields.append(f"exact_families={int(exact_rms.size)}")
        fields.append(f"exact_image_rms_mean={float(np.mean(exact_rms)):.4g}")
        fields.append(f"exact_image_rms_median={float(np.median(exact_rms)):.4g}")
    approx_rms = _finite_prediction_values(best_eval, "approx_image_rms_arcsec")
    if approx_rms.size:
        fields.append(f"approx_image_rms_mean={float(np.mean(approx_rms)):.4g}")
    source_rms = _finite_prediction_values(best_eval, "source_plane_rms")
    if source_rms.size:
        fields.append(f"source_rms_mean={float(np.mean(source_rms)):.4g}")
    return " ".join(fields)


def _validation_complete_message(validation_elapsed: float, best_eval: EvaluationResult) -> str:
    n_failed = sum(1 for info in best_eval.family_predictions.values() if info.get("failed"))
    metrics = _validation_metrics_summary(best_eval)
    return f"[validation] complete in {_fmt_seconds(validation_elapsed)} failed_families={n_failed} {metrics}"


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


def _slim_potfiles_for_artifact(potfiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slim: list[dict[str, Any]] = []
    for potfile in potfiles:
        item = {
            key: value
            for key, value in dict(potfile).items()
            if key not in {"catalog_df", "catalog", "dataframe"}
        }
        catalog_df = potfile.get("catalog_df")
        if isinstance(catalog_df, pd.DataFrame):
            item["catalog_n_rows"] = int(len(catalog_df))
            if "id" in catalog_df:
                item["catalog_ids"] = catalog_df["id"].astype(str).tolist()
        slim.append(item)
    return slim


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
        if results.ns_diagnostics:
            ns_group = handle.create_group("ns_diagnostics")
            for key, value in sorted(results.ns_diagnostics.items()):
                array = np.asarray(value)
                if array.dtype.kind in {"b", "i", "u", "f", "c"}:
                    ns_group.create_dataset(str(key), data=array)

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
                "lens_model_list": state.lens_model_list,
                "base_components": state.base_components,
                "potfiles": _slim_potfiles_for_artifact(state.potfiles),
                "scaling_component_records": state.scaling_component_records,
                "fit_cosmology_flat_wcdm": bool(getattr(state, "fit_cosmology_flat_wcdm", False)),
                "source_position_parameterization": str(state.source_position_parameterization),
                "geometry_cache": None if state.geometry_cache is None else {
                    "effective_z_source_values": state.geometry_cache.effective_z_source_values,
                    "exact_z_source_values": state.geometry_cache.exact_z_source_values,
                    "family_z_source_map": state.geometry_cache.family_z_source_map,
                    "family_effective_z_source_map": state.geometry_cache.family_effective_z_source_map,
                    "dpie_sigma0_factor_by_effective_z": state.geometry_cache.dpie_sigma0_factor_by_effective_z,
                    "dpie_sigma0_factor_by_exact_z": state.geometry_cache.dpie_sigma0_factor_by_exact_z,
                    "family_redshift_binning_sec": state.geometry_cache.family_redshift_binning_sec,
                    "geometry_cache_build_sec": state.geometry_cache.geometry_cache_build_sec,
                    "flat_wcdm_quadrature_order": state.geometry_cache.flat_wcdm_quadrature_order,
                    "lens_quadrature_z": state.geometry_cache.lens_quadrature_z,
                    "lens_quadrature_weights": state.geometry_cache.lens_quadrature_weights,
                    "effective_z_quadrature_z": state.geometry_cache.effective_z_quadrature_z,
                    "effective_z_quadrature_weights": state.geometry_cache.effective_z_quadrature_weights,
                    "exact_z_quadrature_z": state.geometry_cache.exact_z_quadrature_z,
                    "exact_z_quadrature_weights": state.geometry_cache.exact_z_quadrature_weights,
                },
                "svi_init_values": state.svi_init_values,
                "previous_stage_best_values": state.previous_stage_best_values,
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
                        "reliability": family.reliability,
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
                        "reliability_per_image": bin_item.reliability_per_image,
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
        if "source_position_parameterization" not in meta:
            raise ValueError(
                f"{path} is missing explicit source_position_parameterization metadata; "
                "rerun the solver to regenerate artifacts."
            )
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
        lens_model_list = [str(name) for name in meta.get("lens_model_list", [])]
        _validate_supported_lens_model_list(lens_model_list, str(path))
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
                    reliability=np.asarray(
                        item.get("reliability", np.ones(len(item["image_labels"]), dtype=float)),
                        dtype=float,
                    ),
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
                    reliability_per_image=np.asarray(
                        item.get("reliability_per_image", np.ones(len(item["x_obs"]), dtype=float)),
                        dtype=float,
                    ),
                )
                for item in meta.get("bin_data", [])
            ],
            lens_model_list=lens_model_list,
            reference=tuple(meta.get("reference", [0, 0.0, 0.0])),
            fit_mode=str(meta["fit_mode"]),
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
            previous_stage_best_values=(
                {str(key): float(value) for key, value in dict(meta.get("previous_stage_best_values") or {}).items()}
                if meta.get("previous_stage_best_values") is not None
                else None
            ),
            fit_cosmology_flat_wcdm=bool(meta.get("fit_cosmology_flat_wcdm", False)),
            source_position_parameterization=str(meta["source_position_parameterization"]),
        )
        cli_args = _read_h5_json(handle, "cli_args_json", default={})
        init_diagnostics = _read_h5_json(handle, "init_diagnostics_json", default={})
        posterior_group = handle["posterior"]
        arrays = {name: np.asarray(posterior_group[name]) for name in posterior_group.keys()}
        if "ns_diagnostics" in handle:
            arrays["ns_diagnostics"] = {
                name: np.asarray(handle["ns_diagnostics"][name])
                for name in handle["ns_diagnostics"].keys()
            }
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
    _validate_supported_lens_model_list([str(name) for name in getattr(state, "lens_model_list", [])], str(artifacts_dir))
    if not hasattr(state, "geometry_cache"):
        state.geometry_cache = None
    if not hasattr(state, "fit_cosmology_flat_wcdm"):
        state.fit_cosmology_flat_wcdm = False
    if not hasattr(state, "previous_stage_best_values"):
        state.previous_stage_best_values = None
    if not hasattr(state, "source_position_parameterization"):
        raise ValueError(
            "Legacy artifacts are missing explicit source_position_parameterization metadata; "
            "rerun the solver to regenerate artifacts."
        )
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
        cosmo_config = dict(state.cosmo_config) if state.cosmo_config else _build_cosmology(state.parsed)
        state.cosmo_config = dict(cosmo_config)
        state.geometry_cache = _build_geometry_cache(
            cosmo_config,
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


def _physical_best_fit_values_from_artifacts(artifacts_dir: Path) -> dict[str, float]:
    state, _cli_args, arrays, init_diagnostics = _load_artifacts(artifacts_dir)
    arrays, _converted = _maybe_convert_loaded_posterior_arrays_to_physical(arrays, state.parameter_specs, init_diagnostics)
    best_fit = np.asarray(arrays["best_fit"], dtype=float)
    if best_fit.shape[0] != len(state.parameter_specs):
        raise ValueError(
            f"Cannot initialize from {artifacts_dir}: best_fit length {best_fit.shape[0]} "
            f"does not match {len(state.parameter_specs)} parameters."
        )
    return {spec.sample_name: float(best_fit[idx]) for idx, spec in enumerate(state.parameter_specs)}


def _source_position_prior_values_from_artifacts(artifacts_dir: Path) -> dict[str, tuple[float, float]]:
    state, saved_args, arrays, init_diagnostics = _load_artifacts(artifacts_dir)
    arrays, _converted = _maybe_convert_loaded_posterior_arrays_to_physical(arrays, state.parameter_specs, init_diagnostics)
    best_fit_physical = np.asarray(arrays["best_fit"], dtype=float)
    if best_fit_physical.shape[0] != len(state.parameter_specs):
        raise ValueError(
            f"Cannot initialize source positions from {artifacts_dir}: best_fit length {best_fit_physical.shape[0]} "
            f"does not match {len(state.parameter_specs)} parameters."
        )
    sample_likelihood_mode = str(saved_args.get("sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE))
    image_presence_penalty_weight = _effective_image_presence_penalty_weight(
        saved_args.get("image_presence_penalty_weight"),
        sample_likelihood_mode=sample_likelihood_mode,
        fit_mode=str(saved_args.get("fit_mode", FIT_MODE_SEQUENTIAL)),
        image_plane_mode=str(saved_args.get("image_plane_mode", IMAGE_PLANE_MODE_NONE)),
    )
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=float(saved_args.get("match_tolerance_arcsec", DEFAULT_MATCH_TOLERANCE)),
        validate_top_k_families=0,
        sampling_engine="full",
        active_scaling_galaxies=saved_args.get("active_scaling_galaxies"),
        active_scaling_selection=str(saved_args.get("active_scaling_selection", "adaptive")),
        active_scaling_cumulative_fraction=float(
            saved_args.get("active_scaling_cumulative_fraction", DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION)
        ),
        active_scaling_min=int(saved_args.get("active_scaling_min", DEFAULT_ACTIVE_SCALING_MIN)),
        refresh_every=int(saved_args.get("refresh_every", DEFAULT_REFRESH_EVERY)),
        refresh_param_drift_frac=float(saved_args.get("refresh_param_drift_frac", DEFAULT_REFRESH_PARAM_DRIFT_FRAC)),
        validation_approx=str(saved_args.get("validation_approx", "adaptive")),
        source_plane_covariance_floor=float(saved_args.get("source_plane_covariance_floor", 1.0e-6)),
        source_plane_outlier_sigma_arcsec=float(
            saved_args.get("source_plane_outlier_sigma_arcsec", DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC)
        ),
        sample_likelihood_mode=sample_likelihood_mode,
        image_plane_newton_steps=int(saved_args.get("image_plane_newton_steps", 0)),
        image_presence_penalty_weight=image_presence_penalty_weight,
        image_presence_match_radius_arcsec=float(
            saved_args.get("image_presence_match_radius_arcsec", DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC)
        ),
        image_presence_temperature_arcsec=float(
            saved_args.get("image_presence_temperature_arcsec", DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)
        ),
        image_presence_count_softness=float(
            saved_args.get("image_presence_count_softness", DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS)
        ),
        image_presence_count_margin=float(
            saved_args.get("image_presence_count_margin", DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN)
        ),
        evidence_source_prior_sigma_arcsec=saved_args.get("evidence_source_prior_sigma_arcsec"),
        evidence_source_prior_mean_x_arcsec=float(saved_args.get("evidence_source_prior_mean_x_arcsec", 0.0)),
        evidence_source_prior_mean_y_arcsec=float(saved_args.get("evidence_source_prior_mean_y_arcsec", 0.0)),
        exact_image_solver=str(saved_args.get("exact_image_solver", DEFAULT_EXACT_IMAGE_SOLVER)),
    )
    best_fit_latent = evaluator.reported_physical_to_latent_parameter_vector(best_fit_physical)
    summaries = evaluator._family_source_summary(best_fit_latent)
    priors: dict[str, tuple[float, float]] = {}
    for family_id, summary in summaries.items():
        source_x = float(summary.get("source_x", float("nan")))
        source_y = float(summary.get("source_y", float("nan")))
        if np.isfinite(source_x) and np.isfinite(source_y):
            priors[str(family_id)] = (source_x, source_y)
    evaluator.release_runtime_caches()
    return priors


def _finite_float_values(values: dict[str, float] | None) -> dict[str, float] | None:
    if not values:
        return None
    finite_values: dict[str, float] = {}
    for key, value in values.items():
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_f):
            finite_values[str(key)] = value_f
    return finite_values or None


def _has_cosmology_init_overrides(args: argparse.Namespace) -> bool:
    return (
        getattr(args, "cosmology_init_om0", None) is not None
        or getattr(args, "cosmology_init_w0", None) is not None
    )


def _validate_cosmology_init_value(
    value: Any,
    *,
    flag_name: str,
    lower: float,
    upper: float,
) -> float:
    value_f = float(value)
    if not np.isfinite(value_f):
        raise ValueError(f"{flag_name} must be finite.")
    if value_f < float(lower) or value_f > float(upper):
        raise ValueError(f"{flag_name} must be within [{float(lower):g}, {float(upper):g}].")
    return value_f


def _cosmology_init_overrides_from_args(args: argparse.Namespace) -> dict[str, float]:
    overrides: dict[str, float] = {}
    om0 = getattr(args, "cosmology_init_om0", None)
    if om0 is not None:
        overrides[COSMOLOGY_OM0_SAMPLE_NAME] = _validate_cosmology_init_value(
            om0,
            flag_name="--cosmology-init-om0",
            lower=DEFAULT_COSMOLOGY_OM0_LOWER,
            upper=DEFAULT_COSMOLOGY_OM0_UPPER,
        )
    w0 = getattr(args, "cosmology_init_w0", None)
    if w0 is not None:
        overrides[COSMOLOGY_W0_SAMPLE_NAME] = _validate_cosmology_init_value(
            w0,
            flag_name="--cosmology-init-w0",
            lower=DEFAULT_COSMOLOGY_W0_LOWER,
            upper=DEFAULT_COSMOLOGY_W0_UPPER,
        )
    return overrides


def _apply_cosmology_init_overrides(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
    init_values: dict[str, float] | None,
) -> dict[str, float] | None:
    overrides = _cosmology_init_overrides_from_args(args)
    if not overrides:
        return init_values
    sample_names = {spec.sample_name for spec in parameter_specs}
    missing = sorted(set(overrides).difference(sample_names))
    if missing:
        raise ValueError(
            "Cannot apply cosmology initial values because sampled flat-wCDM cosmology "
            f"parameter(s) are not present: {', '.join(missing)}. "
            "Pass --fit-cosmology-flat-wcdm to sample them."
        )
    merged = dict(init_values or {})
    merged.update(overrides)
    return merged


def _build_state_from_inputs(
    args: argparse.Namespace,
    fit_mode_override: str | None = None,
    stage1_prior_summary: Stage1PriorSummary | None = None,
    svi_init_physical_values: dict[str, float] | None = None,
    source_position_prior_values: dict[str, tuple[float, float]] | None = None,
    previous_stage_best_values: dict[str, float] | None = None,
) -> BuildState:
    fit_mode = fit_mode_override or args.fit_mode
    model_fit_mode = FIT_MODE_JOINT if fit_mode == FIT_MODE_EVIDENCE_NS else fit_mode
    fit_cosmology_flat_wcdm = bool(getattr(args, "fit_cosmology_flat_wcdm", False))
    parsed, _potentials_df, images_df, potentials_with_priors = load_best_par(args.par_path)
    if images_df.empty:
        raise ValueError("No multiple-image constraints found in the parsed image catalog.")
    reference = _extract_reference(parsed)
    cosmo_config = _build_cosmology(parsed)
    z_lens_values = [float(pot.get("z_lens", 0.0)) for pot in potentials_with_priors if pot.get("z_lens") is not None]
    z_lens = float(z_lens_values[0]) if z_lens_values else 0.0
    scaling_kpc_per_arcsec = (
        _kpc_per_arcsec_from_config(z_lens, cosmo_config)
        if z_lens > 0.0
        else 1.0
    )
    sigma_arg = getattr(args, "pos_sigma_arcsec", None)
    sigma_arcsec = float(parsed.get("image", {}).get("sigposArcsec", 0.5) if sigma_arg is None else sigma_arg)
    potfiles = list(parsed.get("potfiles", []))
    large_parameter_specs, large_component_param_assignments, large_lens_model_list = _build_parameter_specs(potentials_with_priors)
    if model_fit_mode == "small-only" and large_parameter_specs:
        if stage1_prior_summary is None:
            stage1_prior_summary = _load_stage1_summary(_infer_stage1_artifacts_dir(args))
        large_parameter_specs = _build_stage2_large_parameter_specs(large_parameter_specs, stage1_prior_summary)
    svi_init_values: dict[str, float] | None = None
    if model_fit_mode == FIT_MODE_JOINT and stage1_prior_summary is not None:
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
    if model_fit_mode in {"small-only", FIT_MODE_JOINT}:
        scaling_parameter_specs, scaling_param_indices, scaling_lens_model_list = _build_scaling_parameter_specs(
            potfiles,
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
        scaling_components, scaling_assignments, scaling_component_assignments, scaling_component_records = _build_scaling_components(
            potfiles,
            reference,
            scaling_param_indices,
            scaling_scatter_indices,
            start_component_index=len(base_components),
            kpc_per_arcsec=scaling_kpc_per_arcsec,
        )
        component_param_assignments.extend(scaling_assignments)
        lens_model_list.extend(scaling_lens_model_list)
        base_components.extend(scaling_components)
    packed_lens_spec = _build_packed_lens_spec(base_components, component_param_assignments, scaling_component_assignments)
    fit_images_df, n_nonpositive_images_skipped, n_nonpositive_families_skipped, nonpositive_family_ids = (
        _filter_non_positive_redshift_families(images_df)
    )
    if fit_images_df.empty:
        raise ValueError(
            "No positive-redshift image families remain after dropping non-positive-redshift families. "
            "At least one family must have finite catalog_z > 0."
        )
    if n_nonpositive_images_skipped > 0:
        _log(
            args,
            (
                f"[input] dropped non-positive-redshift families before fitting: "
                f"families={n_nonpositive_families_skipped} images={n_nonpositive_images_skipped} "
                f"ids={','.join(nonpositive_family_ids)}"
            ),
        )

    fit_images_df, n_singleton_images_skipped, n_singleton_families_skipped = _filter_singleton_families(fit_images_df)
    if fit_images_df.empty:
        raise ValueError(
            "No multi-image families remain after dropping singleton pseudo-families. "
            "At least one family must contain two or more images."
        )
    if n_singleton_images_skipped > 0:
        _log(
            args,
            (
                f"[input] dropped singleton pseudo-families before fitting: "
                f"families={n_singleton_families_skipped} images={n_singleton_images_skipped}"
            ),
        )
    family_data, family_redshift_binning_sec = _prepare_family_data(
        fit_images_df,
        sigma_arcsec,
        reference,
        z_lens=z_lens,
        cosmo_config=cosmo_config,
        z_bin_efficiency_tol=float(args.z_bin_efficiency_tol),
    )
    bin_data = _build_bin_data(family_data)
    sample_likelihood_mode = str(getattr(args, "sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE))
    source_position_parameterization = SOURCE_POSITION_PARAMETERIZATION_DIRECT
    if _sample_likelihood_uses_explicit_beta(sample_likelihood_mode):
        source_position_parameterization = str(
            getattr(
                args,
                "source_position_parameterization",
                SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED,
            )
        )
        explicit_source_position_prior_values = source_position_prior_values
        beta_prior_sigma_arcsec = float(
            getattr(
                args,
                "linearized_beta_prior_sigma_arcsec",
                DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC,
            )
        )
        if fit_mode == FIT_MODE_EVIDENCE_NS:
            evidence_prior_sigma = getattr(args, "evidence_source_prior_sigma_arcsec", None)
            if evidence_prior_sigma is None:
                raise ValueError("sampled-source evidence requires evidence_source_prior_sigma_arcsec.")
            beta_prior_sigma_arcsec = float(evidence_prior_sigma)
            if explicit_source_position_prior_values is None:
                explicit_source_position_prior_values = _shared_source_position_prior_values(
                    family_data,
                    float(getattr(args, "evidence_source_prior_mean_x_arcsec", 0.0)),
                    float(getattr(args, "evidence_source_prior_mean_y_arcsec", 0.0)),
                )
        parameter_specs.extend(
            _build_source_position_parameter_specs(
                family_data,
                explicit_source_position_prior_values or {},
                start_index=len(parameter_specs),
                beta_prior_sigma_arcsec=beta_prior_sigma_arcsec,
                parameterization=source_position_parameterization,
            )
        )
        parameter_specs.append(
            _build_image_scatter_parameter_spec(
                start_index=len(parameter_specs),
                upper_arcsec=float(getattr(args, "image_plane_scatter_upper_arcsec", DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC)),
            )
        )
    elif _sample_likelihood_uses_marginal_beta(sample_likelihood_mode):
        parameter_specs.append(
            _build_image_scatter_parameter_spec(
                start_index=len(parameter_specs),
                upper_arcsec=float(getattr(args, "image_plane_scatter_upper_arcsec", DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC)),
            )
        )
    else:
        parameter_specs.append(_build_source_scatter_parameter_spec(start_index=len(parameter_specs)))
    if fit_cosmology_flat_wcdm:
        parameter_specs.extend(_build_cosmology_parameter_specs(len(parameter_specs), cosmo_config))
        if svi_init_values is None:
            svi_init_values = {}
        svi_init_values.setdefault(COSMOLOGY_OM0_SAMPLE_NAME, float(np.clip(_om0_from_config(cosmo_config), DEFAULT_COSMOLOGY_OM0_LOWER, DEFAULT_COSMOLOGY_OM0_UPPER)))
        svi_init_values.setdefault(COSMOLOGY_W0_SAMPLE_NAME, float(np.clip(_w0_from_config(cosmo_config), DEFAULT_COSMOLOGY_W0_LOWER, DEFAULT_COSMOLOGY_W0_UPPER)))
    if svi_init_physical_values or source_position_prior_values:
        if svi_init_values is None:
            svi_init_values = {}
        for spec in parameter_specs:
            physical_value: float | None = None
            if svi_init_physical_values and spec.sample_name in svi_init_physical_values:
                physical_value = float(svi_init_physical_values[spec.sample_name])
            elif source_position_prior_values and spec.component_family == "source_position":
                family_id = str(spec.potential_id)
                if family_id in source_position_prior_values:
                    center_x, center_y = source_position_prior_values[family_id]
                    physical_value = float(center_x if spec.field == "beta_x" else center_y)
            elif (
                svi_init_physical_values
                and spec.sample_name == "image_sigma_int"
                and "source_sigma_int" in svi_init_physical_values
            ):
                physical_value = float(
                    np.clip(
                        float(svi_init_physical_values["source_sigma_int"]),
                        DEFAULT_IMAGE_SIGMA_INT_LOWER_ARCSEC,
                        float(getattr(args, "image_plane_scatter_upper_arcsec", DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC)),
                    )
                )
            if physical_value is None:
                continue
            if (
                spec.component_family == "source_position"
                and str(getattr(args, "source_position_parameterization", SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED))
                == SOURCE_POSITION_PARAMETERIZATION_CONDITIONAL_WHITENED
            ):
                svi_init_values[spec.sample_name] = 0.0
            else:
                svi_init_values[spec.sample_name] = _physical_to_latent_numpy(physical_value, spec)
    svi_init_values = _apply_cosmology_init_overrides(args, parameter_specs, svi_init_values)
    geometry_cache = _build_geometry_cache(
        cosmo_config,
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
        potfiles=potfiles,
        scaling_component_records=scaling_component_records,
        geometry_cache=geometry_cache,
        svi_init_values=svi_init_values,
        previous_stage_best_values=_finite_float_values(previous_stage_best_values),
        fit_cosmology_flat_wcdm=fit_cosmology_flat_wcdm,
        source_position_parameterization=source_position_parameterization,
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
        jnp.asarray(transform_kind_array == "affine", dtype=bool),
        jnp.asarray([float(getattr(spec, "transform_scale", 1.0)) for spec in parameter_specs], dtype=jnp.float64),
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
    sample_likelihood_mode = str(getattr(args, "sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE))
    image_presence_penalty_weight = _effective_image_presence_penalty_weight(
        getattr(args, "image_presence_penalty_weight", None),
        sample_likelihood_mode=sample_likelihood_mode,
        fit_mode=str(getattr(args, "fit_mode", FIT_MODE_SEQUENTIAL)),
        image_plane_mode=str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE)),
    )
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=args.match_tolerance_arcsec,
        validate_top_k_families=args.validate_top_k_families,
        sampling_engine=args.sampling_engine,
        active_scaling_galaxies=args.active_scaling_galaxies,
        active_scaling_selection=args.active_scaling_selection,
        active_scaling_cumulative_fraction=args.active_scaling_cumulative_fraction,
        active_scaling_min=args.active_scaling_min,
        refresh_every=args.refresh_every,
        refresh_param_drift_frac=args.refresh_param_drift_frac,
        validation_approx=args.validation_approx,
        source_plane_covariance_floor=args.source_plane_covariance_floor,
        source_plane_outlier_sigma_arcsec=args.source_plane_outlier_sigma_arcsec,
        sample_likelihood_mode=sample_likelihood_mode,
        image_plane_newton_steps=int(getattr(args, "image_plane_newton_steps", 0)),
        image_presence_penalty_weight=image_presence_penalty_weight,
        image_presence_match_radius_arcsec=float(
            getattr(args, "image_presence_match_radius_arcsec", DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC)
        ),
        image_presence_temperature_arcsec=float(
            getattr(args, "image_presence_temperature_arcsec", DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)
        ),
        image_presence_count_softness=float(
            getattr(args, "image_presence_count_softness", DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS)
        ),
        image_presence_count_margin=float(
            getattr(args, "image_presence_count_margin", DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN)
        ),
        evidence_source_prior_sigma_arcsec=getattr(args, "evidence_source_prior_sigma_arcsec", None),
        evidence_source_prior_mean_x_arcsec=float(getattr(args, "evidence_source_prior_mean_x_arcsec", 0.0)),
        evidence_source_prior_mean_y_arcsec=float(getattr(args, "evidence_source_prior_mean_y_arcsec", 0.0)),
        exact_image_solver=str(getattr(args, "exact_image_solver", DEFAULT_EXACT_IMAGE_SOLVER)),
        quick_diagnostics=bool(getattr(args, "quick_diagnostics", False)),
    )
    _log_evaluator_summary(args, evaluator)
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
    evaluator.refresh_scaling_scatter_cache(midpoint, reason="svi_nuts_initial")
    evaluator.refresh_source_metric_cache(midpoint, reason="svi_nuts_initial")
    _log_solver_active_approximation_warning(args, evaluator)
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
        _log_posterior_summary(args, "fixed_model", posterior)
        if args.skip_validation or bool(getattr(args, "quick_diagnostics", False)):
            reason = "--quick-diagnostics" if bool(getattr(args, "quick_diagnostics", False)) else "--skip-validation"
            _log(args, f"[validation] skipped by {reason}; using source-plane summary only")
            best_eval = _run_logged_phase(args, "validation.approximate", lambda: _approximate_evaluation(evaluator, best_fit))
        else:
            validation_start = time.time()
            n_validate = len(evaluator.validation_family_ids)
            _log(args, f"[validation] starting exact validation families={n_validate}")
            best_eval = _run_logged_phase(
                args,
                "validation.evaluate",
                lambda: evaluator.evaluate(best_fit, validate_all_families=False),
            )
            validation_elapsed = time.time() - validation_start
            evaluator.timing_totals["validation_runtime"] += validation_elapsed
            _log(args, _validation_complete_message(validation_elapsed, best_eval))
        runtime_sec = time.time() - start

        artifacts_dir = run_dir / "artifacts"
        _log(args, f"[output] saving artifacts to {artifacts_dir}")
        best_fit_physical = _run_logged_phase(
            args,
            "output.convert_best_fit_to_physical",
            lambda: _convert_theta_to_reported_physical(best_fit, state.parameter_specs, evaluator),
        )
        posterior_for_output = _run_logged_phase(
            args,
            "output.posterior_results_to_physical",
            lambda: _posterior_results_to_reported_physical(posterior, state.parameter_specs, evaluator),
        )
        del posterior
        gc.collect()
        _run_logged_phase(
            args,
            "output.save_artifacts",
            lambda: _save_artifacts(artifacts_dir, state, args, best_fit_physical, posterior_for_output),
        )
        _write_truth_validation_outputs(args, run_dir)
        if args.skip_plots:
            _log(args, "[output] plot generation skipped by --skip-plots")
            evaluator.release_runtime_caches()
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
            evaluator.release_runtime_caches()
        _log(args, f"[done] total_runtime={_fmt_seconds(time.time() - start)}")
        return
    _log(args, f"[model] initializing direct evaluator for {args.fit_method}")
    evaluator, _midpoint = _prepare_direct_evaluator(args, state)
    sample_model = _posterior_model(state.parameter_specs, evaluator)
    if str(args.fit_method) == FIT_METHOD_NS:
        best_fit = _reference_theta_from_init_values(state.parameter_specs, state.svi_init_values, _midpoint)
        if evaluator.surrogate_enabled:
            evaluator.refresh_surrogate(best_fit, reason="ns_initial")
        evaluator.refresh_scaling_scatter_cache(best_fit, reason="ns_initial")
        evaluator.refresh_source_metric_cache(best_fit, reason="ns_initial")
        posterior = _run_numpyro_nested_sampler(args, state, evaluator, sample_model)
        if posterior.samples.ndim != 2 or posterior.samples.shape[0] == 0:
            raise RuntimeError("Nested sampler returned no posterior samples.")
        if np.asarray(posterior.log_prob).size and np.isfinite(posterior.log_prob).any():
            best_index = int(np.nanargmax(posterior.log_prob))
        else:
            best_index = 0
        best_fit = np.asarray(posterior.samples[best_index], dtype=float)
        evaluator.refresh_scaling_scatter_cache(best_fit, reason="post_ns")
        evaluator.refresh_source_metric_cache(best_fit, reason="post_ns")
        _log(args, f"[ns] best_fit updated from resampled posterior draw index={best_index}")
    else:
        best_fit, posterior, svi_diagnostics = _run_svi_fit(args, state, evaluator, sample_model)
        svi_posterior = posterior
        evaluator.refresh_scaling_scatter_cache(best_fit, reason="post_svi")
        evaluator.refresh_source_metric_cache(best_fit, reason="post_svi")
        _log(
            args,
            (
                "[svi:init] "
                f"final_elbo={float(svi_diagnostics.get('svi_final_elbo_loss', float('nan'))):.4g}"
            ),
        )
        if str(args.fit_method) == FIT_METHOD_SVI_NUTS:
            nuts_init = _nuts_initialization_from_svi_center(args, state.parameter_specs, best_fit, svi_diagnostics)
            _log(
                args,
                (
                    "[nuts:init] from svi "
                    f"distinct_seeds={int(nuts_init.diagnostics.get('distinct_chain_seeds', 0))}"
                ),
            )
            nuts_posterior = _run_numpyro_nuts_sampler(args, state, evaluator, sample_model, nuts_init)
            nuts_usable, nuts_reason = _nuts_posterior_is_usable(nuts_posterior)
            if not nuts_usable:
                _log(
                    args,
                    (
                        "[nuts] posterior rejected; falling back to SVI guide posterior "
                        f"reason={nuts_reason}"
                    ),
                )
                posterior = svi_posterior
                posterior.sampler = "svi_fallback_after_failed_nuts"
                posterior.init_diagnostics["nuts_rejected"] = True
                posterior.init_diagnostics["nuts_rejection_reason"] = nuts_reason
                posterior.init_diagnostics["nuts_accept_mean"] = (
                    float(np.nanmean(nuts_posterior.accept_prob)) if np.asarray(nuts_posterior.accept_prob).size else float("nan")
                )
                posterior.init_diagnostics["nuts_divergence_fraction"] = (
                    float(np.mean(nuts_posterior.diverging)) if np.asarray(nuts_posterior.diverging).size else float("nan")
                )
            elif nuts_posterior.samples.size:
                posterior = nuts_posterior
                best_index = int(np.nanargmax(posterior.log_prob))
                best_fit = np.asarray(posterior.samples[best_index], dtype=float)
                evaluator.refresh_scaling_scatter_cache(best_fit, reason="post_nuts")
                evaluator.refresh_source_metric_cache(best_fit, reason="post_nuts")
                _log(args, f"[nuts] best_fit updated from retained posterior sample index={best_index}")
            if posterior is not svi_posterior:
                del svi_posterior
                gc.collect()
            if "nuts_posterior" in locals() and posterior is not nuts_posterior:
                del nuts_posterior
                gc.collect()
    best_fit = _max_likelihood_best_fit_from_posterior(args, evaluator, posterior, best_fit)
    if evaluator.surrogate_enabled:
        evaluator.refresh_surrogate(best_fit, reason="post_max_likelihood")
    evaluator.refresh_scaling_scatter_cache(best_fit, reason="post_max_likelihood")
    evaluator.refresh_source_metric_cache(best_fit, reason="post_max_likelihood")
    _log_posterior_summary(args, "selected", posterior)
    _artifacts_dir, best_fit_physical, posterior_for_output = _save_inference_checkpoint(
        args,
        state,
        evaluator,
        run_dir,
        best_fit,
        posterior,
    )
    del posterior
    gc.collect()

    if args.skip_validation or bool(getattr(args, "quick_diagnostics", False)):
        reason = "--quick-diagnostics" if bool(getattr(args, "quick_diagnostics", False)) else "--skip-validation"
        _log(args, f"[validation] skipped by {reason}; using source-plane summary only")
        best_eval = _run_logged_phase(args, "validation.approximate", lambda: _approximate_evaluation(evaluator, best_fit))
    else:
        validation_start = time.time()
        n_validate = len(evaluator.validation_family_ids)
        _log(args, f"[validation] starting exact validation families={n_validate}")
        best_eval = _run_logged_phase(
            args,
            "validation.evaluate",
            lambda: evaluator.evaluate(best_fit, validate_all_families=False),
        )
        validation_elapsed = time.time() - validation_start
        evaluator.timing_totals["validation_runtime"] += validation_elapsed
        _log(args, _validation_complete_message(validation_elapsed, best_eval))
    runtime_sec = time.time() - start

    _write_truth_validation_outputs(args, run_dir)
    if args.skip_plots:
        _log(args, "[output] plot generation skipped by --skip-plots")
        evaluator.release_runtime_caches()
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
        evaluator.release_runtime_caches()
    _log(args, f"[done] total_runtime={_fmt_seconds(time.time() - start)}")


def _previous_stage_artifacts_dir_for_run_dir(run_dir: Path) -> Path | None:
    if not _is_sequential_stage_path(run_dir):
        return None
    stage_name = _sequential_stage_name(run_dir)
    if stage_name == "stage2_joint":
        return run_dir.parent / "stage1_large_only" / "artifacts"
    if stage_name == "stage3_image_plane":
        return run_dir.parent / "stage2_joint" / "artifacts"
    if stage_name == "stage4_linearized_image_plane":
        stage3_artifacts = run_dir.parent / "stage3_image_plane" / "artifacts"
        if _has_plot_artifacts(stage3_artifacts):
            return stage3_artifacts
        return run_dir.parent / "stage2_joint" / "artifacts"
    return None


def _infer_previous_stage_best_values_for_plots(args: argparse.Namespace, run_dir: Path) -> dict[str, float] | None:
    artifacts_dir = _previous_stage_artifacts_dir_for_run_dir(run_dir)
    if artifacts_dir is None or not _has_plot_artifacts(artifacts_dir):
        return None
    try:
        return _finite_float_values(_physical_best_fit_values_from_artifacts(artifacts_dir))
    except Exception as exc:  # pragma: no cover - best-effort compatibility for older/corrupt artifacts
        _log(args, f"[plots-only] previous-stage best-fit guides unavailable from {artifacts_dir}: {exc}")
        return None


def _rerender_plots(
    args: argparse.Namespace,
    run_dir: Path,
    exact_diagnostics_stage: str | Path | None = None,
) -> None:
    _configure_debug_log(args, run_dir.name, run_dir)
    _log_stage_banner(args, f"PLOTS ONLY: {_stage_banner_title_from_run_name(str(run_dir))}", f"run_dir={run_dir}")
    _log(args, f"[stage] plots-only start run_dir={run_dir}")
    _log(args, f"[plots-only] loading artifacts from {run_dir / 'artifacts'}")
    state, saved_args, arrays, init_diagnostics = _run_logged_phase(
        args,
        "plots_only.load_artifacts",
        lambda: _load_artifacts(run_dir / "artifacts"),
    )
    if not getattr(state, "previous_stage_best_values", None):
        inferred_previous_values = _infer_previous_stage_best_values_for_plots(args, run_dir)
        if inferred_previous_values:
            state.previous_stage_best_values = inferred_previous_values
    if exact_diagnostics_stage is None:
        exact_diagnostics_stage = _plots_only_exact_diagnostics_stage(run_dir)
    force_quick_diagnostics = (
        _is_sequential_stage_path(run_dir)
        and not _stage_allows_exact_image_diagnostics(run_dir, exact_diagnostics_stage)
    )
    quick_diagnostics = bool(
        getattr(args, "quick_diagnostics", False)
        or saved_args.get("quick_diagnostics", False)
        or force_quick_diagnostics
    )
    if force_quick_diagnostics and not bool(saved_args.get("quick_diagnostics", False)):
        _log(args, "[plots-only] quick diagnostics forced for pre-stage4 sequential stage")
    plot_saved_args = dict(saved_args)
    plot_saved_args["quick_diagnostics"] = quick_diagnostics
    plot_args = _clone_args(args, **plot_saved_args)
    _log_state_summary(args, state)
    arrays, converted_legacy_refine = _run_logged_phase(
        args,
        "plots_only.normalize_loaded_posterior",
        lambda: _maybe_convert_loaded_posterior_arrays_to_physical(arrays, state.parameter_specs, init_diagnostics),
    )
    if converted_legacy_refine:
        _log(args, "[plots-only] converted legacy saved posterior arrays from latent to physical units")
    sample_likelihood_mode = str(saved_args.get("sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE))
    image_presence_penalty_weight = _effective_image_presence_penalty_weight(
        saved_args.get("image_presence_penalty_weight"),
        sample_likelihood_mode=sample_likelihood_mode,
        fit_mode=str(saved_args.get("fit_mode", FIT_MODE_SEQUENTIAL)),
        image_plane_mode=str(saved_args.get("image_plane_mode", IMAGE_PLANE_MODE_NONE)),
    )
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=float(saved_args.get("match_tolerance_arcsec", DEFAULT_MATCH_TOLERANCE)),
        validate_top_k_families=int(saved_args.get("validate_top_k_families", 8)),
        sampling_engine=str(saved_args.get("sampling_engine", "full")),
        active_scaling_galaxies=saved_args.get("active_scaling_galaxies"),
        active_scaling_selection=str(saved_args.get("active_scaling_selection", "adaptive")),
        active_scaling_cumulative_fraction=float(
            saved_args.get("active_scaling_cumulative_fraction", DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION)
        ),
        active_scaling_min=int(saved_args.get("active_scaling_min", DEFAULT_ACTIVE_SCALING_MIN)),
        refresh_every=int(saved_args.get("refresh_every", DEFAULT_REFRESH_EVERY)),
        refresh_param_drift_frac=float(saved_args.get("refresh_param_drift_frac", DEFAULT_REFRESH_PARAM_DRIFT_FRAC)),
        validation_approx=str(saved_args.get("validation_approx", "exact")),
        source_plane_covariance_floor=float(saved_args.get("source_plane_covariance_floor", 1.0e-6)),
        source_plane_outlier_sigma_arcsec=float(
            saved_args.get("source_plane_outlier_sigma_arcsec", DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC)
        ),
        sample_likelihood_mode=sample_likelihood_mode,
        image_plane_newton_steps=int(saved_args.get("image_plane_newton_steps", 0)),
        image_presence_penalty_weight=image_presence_penalty_weight,
        image_presence_match_radius_arcsec=float(
            saved_args.get("image_presence_match_radius_arcsec", DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC)
        ),
        image_presence_temperature_arcsec=float(
            saved_args.get("image_presence_temperature_arcsec", DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)
        ),
        image_presence_count_softness=float(
            saved_args.get("image_presence_count_softness", DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS)
        ),
        image_presence_count_margin=float(
            saved_args.get("image_presence_count_margin", DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN)
        ),
        evidence_source_prior_sigma_arcsec=saved_args.get("evidence_source_prior_sigma_arcsec"),
        evidence_source_prior_mean_x_arcsec=float(saved_args.get("evidence_source_prior_mean_x_arcsec", 0.0)),
        evidence_source_prior_mean_y_arcsec=float(saved_args.get("evidence_source_prior_mean_y_arcsec", 0.0)),
        exact_image_solver=str(saved_args.get("exact_image_solver", DEFAULT_EXACT_IMAGE_SOLVER)),
        quick_diagnostics=quick_diagnostics,
    )
    best_fit = np.asarray(arrays["best_fit"], dtype=float)
    best_fit_latent = _run_logged_phase(
        args,
        "plots_only.convert_best_fit_to_latent",
        lambda: evaluator.reported_physical_to_latent_parameter_vector(best_fit),
    )
    if evaluator.surrogate_enabled:
        evaluator.refresh_surrogate(best_fit_latent, reason="plots_only")
    evaluator.refresh_scaling_scatter_cache(best_fit_latent, reason="plots_only")
    evaluator.refresh_source_metric_cache(best_fit_latent, reason="plots_only")
    _log_solver_active_approximation_warning(args, evaluator)
    if quick_diagnostics:
        _log(args, "[plots-only] quick diagnostics enabled; using source-plane summary only")
        best_eval = _run_logged_phase(
            args,
            "plots_only.validation.approximate",
            lambda: _approximate_evaluation(evaluator, best_fit_latent),
        )
    else:
        best_eval = _run_logged_phase(
            args,
            "plots_only.validation.evaluate",
            lambda: evaluator.evaluate(best_fit_latent, validate_all_families=False),
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
        ns_diagnostics=(
            {str(key): np.asarray(value) for key, value in arrays["ns_diagnostics"].items()}
            if isinstance(arrays.get("ns_diagnostics"), dict)
            else None
        ),
    )
    _log_posterior_summary(args, "plots_only_loaded", posterior)
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
            args=plot_args,
        ),
    )
    evaluator.timing_totals["plot_runtime"] += time.time() - plot_start
    _log(args, f"[plots-only] complete in {_fmt_seconds(evaluator.timing_totals['plot_runtime'])}")
    _log(args, f"[stage] plots-only end run_dir={run_dir}")


def _clone_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    payload = vars(args).copy()
    payload.update(updates)
    return argparse.Namespace(**payload)


def _stage_banner_title_from_run_name(run_name: str) -> str:
    stage_name = Path(str(run_name)).name
    labels = {
        "stage1_large_only": "STAGE 1: stage1_large_only",
        "stage2_joint": "STAGE 2: stage2_joint",
        "stage3_image_plane": "STAGE 3: stage3_image_plane",
        "stage4_linearized_image_plane": "STAGE 4: stage4_linearized_image_plane",
    }
    return labels.get(stage_name, stage_name or str(run_name))


def _run_single_stage(
    args: argparse.Namespace,
    fit_mode: str,
    run_name: str,
    stage1_prior_summary: Stage1PriorSummary | None = None,
    sample_likelihood_mode: str = SAMPLE_LIKELIHOOD_SOURCE,
    svi_init_physical_values: dict[str, float] | None = None,
    source_position_prior_values: dict[str, tuple[float, float]] | None = None,
    previous_stage_best_values: dict[str, float] | None = None,
) -> Path:
    _configure_debug_log(args, run_name, None)
    stage_args = _clone_args(args, fit_mode=fit_mode, run_name=run_name, sample_likelihood_mode=sample_likelihood_mode)
    _log_stage_banner(
        stage_args,
        _stage_banner_title_from_run_name(run_name),
        (
            f"run_name={run_name} fit_mode={fit_mode} fit_method={stage_args.fit_method} "
            f"sample_likelihood_mode={sample_likelihood_mode} "
            f"fit_cosmology_flat_wcdm={bool(getattr(stage_args, 'fit_cosmology_flat_wcdm', False))}"
        ),
    )
    _log(
        stage_args,
        (
            f"[stage] start run_name={run_name} fit_mode={fit_mode} fit_method={stage_args.fit_method} "
            f"sample_likelihood_mode={sample_likelihood_mode} "
            f"fit_cosmology_flat_wcdm={bool(getattr(stage_args, 'fit_cosmology_flat_wcdm', False))}"
        ),
    )
    load_start = time.time()
    _log(stage_args, f"[load] parsing input from {stage_args.par_path}")
    state = _run_logged_phase(
        stage_args,
        "state.build",
        lambda: _build_state_from_inputs(
            stage_args,
            fit_mode_override=fit_mode,
            stage1_prior_summary=stage1_prior_summary,
            svi_init_physical_values=svi_init_physical_values,
            source_position_prior_values=source_position_prior_values,
            previous_stage_best_values=previous_stage_best_values,
        ),
        detail=f"fit_mode={fit_mode}",
    )
    _log_state_summary(stage_args, state)
    _log(stage_args, f"[load] parser complete in {_fmt_seconds(time.time() - load_start)}")
    run_dir = Path(stage_args.output_dir) / state.run_name
    _run_inference(stage_args, state, run_dir)
    _log(stage_args, f"[stage] end run_name={run_name} run_dir={run_dir}")
    return run_dir


def _load_stage_run_summary_for_aggregate(stage_dir: Path) -> dict[str, Any] | None:
    summary_path = stage_dir / "tables" / "run_summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    payload = dict(payload)
    payload["stage"] = stage_dir.name
    return payload


def _write_sequential_run_summary_txt(
    root_dir: Path,
    root_run_name: str,
    stage_dirs: list[Path | None],
) -> tuple[Path | None, str]:
    rows: list[dict[str, Any]] = []
    for stage_dir in stage_dirs:
        if stage_dir is None:
            continue
        summary = _load_stage_run_summary_for_aggregate(Path(stage_dir))
        if summary is not None:
            rows.append(summary)
    if not rows:
        return None, ""
    text = _format_sequential_run_summary_text(rows, run_name=root_run_name, root_dir=root_dir)
    path = root_dir / "run_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path, text


def _run_sequential(args: argparse.Namespace) -> None:
    stage_fit_controls = _normalize_stage_fit_controls(args)
    stage2_controls = stage_fit_controls["stage2"]
    stage3_controls = stage_fit_controls["stage3"]
    stage4_controls = stage_fit_controls["stage4"]
    stage4_enabled = _stage4_explicit_beta_enabled(args)
    stage3_enabled = _local_jacobian_stage_enabled(args)
    exact_diagnostics_stage = _final_sequential_exact_diagnostics_stage(
        stage3_enabled=stage3_enabled,
        stage4_enabled=stage4_enabled,
    )
    stage4_sample_likelihood_mode = _stage4_sample_likelihood_mode(args)
    fit_cosmology_flat_wcdm = bool(getattr(args, "fit_cosmology_flat_wcdm", False))
    root_run_name = args.run_name or _make_run_name(args.par_path)
    _log_stage_banner(
        args,
        "SEQUENTIAL WORKFLOW",
        (
            f"run_name={root_run_name} image_plane_mode={getattr(args, 'image_plane_mode', IMAGE_PLANE_MODE_NONE)} "
            f"stage3={'enabled' if stage3_enabled else 'disabled'} stage4={'enabled' if stage4_enabled else 'disabled'}"
        ),
    )
    _log(
        args,
        (
            f"[stage] sequential start run_name={args.run_name or '<auto>'} "
            f"stage2_fit_method={stage2_controls.fit_method} "
            f"stage3_fit_method={stage3_controls.fit_method if stage3_enabled else '<disabled>'} "
            f"stage4_fit_method={stage4_controls.fit_method if stage4_enabled else '<disabled>'}"
        ),
    )
    resume = bool(getattr(args, "resume", False))

    def maybe_run_stage(
        stage_args: argparse.Namespace,
        fit_mode: str,
        run_name: str,
        **kwargs: Any,
    ) -> Path:
        run_dir = Path(stage_args.output_dir) / run_name
        if resume:
            if _stage_run_complete(run_dir):
                if bool(getattr(stage_args, "skip_plots", False)):
                    _log(args, f"[resume] reusing completed stage run_name={run_name} run_dir={run_dir} skip_plots=True")
                else:
                    _log(args, f"[resume] refreshing completed stage outputs run_name={run_name} run_dir={run_dir}")
                    _rerender_plots(stage_args, run_dir)
                return run_dir
            if _stage_run_checkpointed(run_dir):
                _log(args, f"[resume] finalizing checkpointed stage run_name={run_name} run_dir={run_dir}")
                _rerender_plots(stage_args, run_dir)
                return run_dir
        return _run_single_stage(stage_args, fit_mode, run_name, **kwargs)

    stage1_run_name = str(Path(root_run_name) / "stage1_large_only")
    stage1_args = _args_with_fit_controls(args, stage2_controls, fit_method=FIT_METHOD_SVI, fit_cosmology_flat_wcdm=False)
    stage1_args = _force_quick_diagnostics_for_nonfinal_stage(stage1_args, stage1_run_name, exact_diagnostics_stage)
    stage1_run_dir = maybe_run_stage(stage1_args, "large-only", stage1_run_name, sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE)
    stage1_summary = _load_stage1_summary(stage1_run_dir / "artifacts")
    stage2_run_name = str(Path(root_run_name) / "stage2_joint")
    stage2_args = _args_with_fit_controls(
        args,
        stage2_controls,
        fit_cosmology_flat_wcdm=fit_cosmology_flat_wcdm and not stage3_enabled and not stage4_enabled,
    )
    stage2_args = _force_quick_diagnostics_for_nonfinal_stage(stage2_args, stage2_run_name, exact_diagnostics_stage)
    stage2_run_dir = maybe_run_stage(
        stage2_args,
        "joint",
        stage2_run_name,
        stage1_prior_summary=stage1_summary,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        previous_stage_best_values=stage1_summary.map_values,
    )
    summary_payload = {
        "fit_mode": "sequential",
        "fit_method": stage2_controls.fit_method,
        "image_plane_mode": str(args.image_plane_mode),
        "stage_fit_controls": {
            "stage2": stage2_controls.to_json(),
            "stage3": stage3_controls.to_json(),
            "stage4": stage4_controls.to_json(),
        },
        "skip_stage3_image_plane_local_jacobian": bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)),
        "image_plane_newton_steps": int(getattr(args, "image_plane_newton_steps", 0)),
        "image_presence_penalty_weight": getattr(args, "image_presence_penalty_weight", None),
        "image_presence_effective_stage4_penalty_weight": _effective_image_presence_penalty_weight(
            getattr(args, "image_presence_penalty_weight", None),
            sample_likelihood_mode=str(stage4_sample_likelihood_mode or SAMPLE_LIKELIHOOD_SOURCE),
            fit_mode=FIT_MODE_JOINT,
            image_plane_mode=str(args.image_plane_mode),
        ),
        "image_presence_match_radius_arcsec": float(
            getattr(args, "image_presence_match_radius_arcsec", DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC)
        ),
        "image_presence_temperature_arcsec": float(
            getattr(args, "image_presence_temperature_arcsec", DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)
        ),
        "image_presence_count_softness": float(
            getattr(args, "image_presence_count_softness", DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS)
        ),
        "image_presence_count_margin": float(
            getattr(args, "image_presence_count_margin", DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN)
        ),
        "source_position_parameterization": str(
            getattr(args, "source_position_parameterization", SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED)
        ),
        "fit_cosmology_flat_wcdm": fit_cosmology_flat_wcdm,
        "cosmology_redshift_binning": "fiducial_fixed",
        "stage1_run_dir": str(stage1_run_dir),
        "stage2_run_dir": str(stage2_run_dir),
    }
    stage3_run_dir: Path | None = None
    if stage3_enabled:
        stage3_run_name = str(Path(root_run_name) / "stage3_image_plane")
        stage3_init_values = _physical_best_fit_values_from_artifacts(stage2_run_dir / "artifacts")
        stage3_args = _args_with_fit_controls(
            args,
            stage3_controls,
            fit_cosmology_flat_wcdm=fit_cosmology_flat_wcdm and not stage4_enabled,
        )
        stage3_args = _force_quick_diagnostics_for_nonfinal_stage(stage3_args, stage3_run_name, exact_diagnostics_stage)
        stage3_run_dir = maybe_run_stage(
            stage3_args,
            "joint",
            stage3_run_name,
            stage1_prior_summary=stage1_summary,
            sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
            svi_init_physical_values=stage3_init_values,
            previous_stage_best_values=stage3_init_values,
        )
        summary_payload["stage3_run_dir"] = str(stage3_run_dir)
    if stage4_enabled:
        stage4_init_artifacts_dir = (stage3_run_dir or stage2_run_dir) / "artifacts"
        stage4_run_name = str(Path(root_run_name) / _stage4_run_directory_name(args))
        stage4_init_values = _physical_best_fit_values_from_artifacts(stage4_init_artifacts_dir)
        source_position_priors = _source_position_prior_values_from_artifacts(stage4_init_artifacts_dir)
        stage4_args = _args_with_fit_controls(
            args,
            stage4_controls,
            fit_cosmology_flat_wcdm=fit_cosmology_flat_wcdm,
        )
        stage4_run_dir = maybe_run_stage(
            stage4_args,
            "joint",
            stage4_run_name,
            stage1_prior_summary=stage1_summary,
            sample_likelihood_mode=str(stage4_sample_likelihood_mode),
            svi_init_physical_values=stage4_init_values,
            source_position_prior_values=source_position_priors,
            previous_stage_best_values=stage4_init_values,
        )
        summary_payload["stage4_run_dir"] = str(stage4_run_dir)
    summary_path = Path(args.output_dir) / root_run_name / "sequential_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    _log(args, f"[done] sequential summary written to {summary_path}")
    aggregate_stage_dirs: list[Path | None] = [stage1_run_dir, stage2_run_dir, stage3_run_dir]
    if "stage4_run_dir" in summary_payload:
        aggregate_stage_dirs.append(Path(str(summary_payload["stage4_run_dir"])))
    summary_txt_path, summary_text = _run_logged_phase(
        args,
        "output.write_sequential_run_summary_txt",
        lambda: _write_sequential_run_summary_txt(
            Path(args.output_dir) / root_run_name,
            root_run_name,
            aggregate_stage_dirs,
        ),
    )
    if summary_txt_path is not None:
        _log(args, f"[done] sequential run summary written to {summary_txt_path}")
        _log(args, "[done] sequential run summary\n" + summary_text.rstrip())
    _log_stage_banner(args, "SEQUENTIAL WORKFLOW COMPLETE", f"run_name={root_run_name}")
    _log(args, f"[stage] sequential end run_name={root_run_name}")


def _run_evidence_ns(args: argparse.Namespace) -> None:
    stage_fit_controls = _normalize_stage_fit_controls(args)
    controls = stage_fit_controls["stage2"]
    run_name = args.run_name or _make_run_name(args.par_path)
    run_dir = Path(args.output_dir) / run_name
    evidence_likelihood_mode = str(
        getattr(args, "evidence_likelihood_mode", DEFAULT_EVIDENCE_LIKELIHOOD_MODE)
    )
    _log_stage_banner(
        args,
        "EVIDENCE NS WORKFLOW",
        (
            f"run_name={run_name} fit_method={controls.fit_method} "
            f"evidence_likelihood_mode={evidence_likelihood_mode} "
            f"source_prior_sigma_arcsec={float(getattr(args, 'evidence_source_prior_sigma_arcsec'))} "
            f"source_prior_mean=({float(getattr(args, 'evidence_source_prior_mean_x_arcsec', 0.0))},"
            f"{float(getattr(args, 'evidence_source_prior_mean_y_arcsec', 0.0))})"
        ),
    )
    if bool(getattr(args, "resume", False)):
        if _stage_run_complete(run_dir):
            if bool(getattr(args, "skip_plots", False)):
                _log(args, f"[resume] reusing completed evidence run run_name={run_name} run_dir={run_dir} skip_plots=True")
            else:
                _log(args, f"[resume] refreshing completed evidence run outputs run_name={run_name} run_dir={run_dir}")
                _rerender_plots(args, run_dir)
            return
        if _stage_run_checkpointed(run_dir):
            _log(args, f"[resume] finalizing checkpointed evidence run run_name={run_name} run_dir={run_dir}")
            _rerender_plots(args, run_dir)
            return
    stage_args = _args_with_fit_controls(args, controls)
    _run_single_stage(
        stage_args,
        FIT_MODE_EVIDENCE_NS,
        run_name,
        sample_likelihood_mode=evidence_likelihood_mode,
    )


def _has_plot_artifacts(artifacts_dir: Path) -> bool:
    return (artifacts_dir / "plot_bundle.h5").exists() or (artifacts_dir / "posterior_arrays.npz").exists()


def _stage_run_complete(run_dir: str | Path) -> bool:
    path = Path(run_dir)
    return (path / "artifacts" / "plot_bundle.h5").exists() and (path / "tables" / "run_summary.json").exists()


def _stage_run_checkpointed(run_dir: str | Path) -> bool:
    path = Path(run_dir)
    return _has_plot_artifacts(path / "artifacts") and not (path / "tables" / "run_summary.json").exists()


def _resolve_run_artifacts_dir(run_dir: str | Path) -> Path:
    candidate = Path(run_dir)
    return candidate if candidate.name == "artifacts" else candidate / "artifacts"


def main() -> None:
    try:
        args = _parse_args()
        stage_fit_controls = _normalize_stage_fit_controls(args)
        if args.seed is not None:
            np.random.seed(args.seed)
        inferred_run_name = args.run_name
        if inferred_run_name is None and args.par_path:
            inferred_run_name = _make_run_name(args.par_path)
        else:
            inferred_run_name = inferred_run_name or "cluster_solver"
        _configure_debug_log(args, inferred_run_name, None)
        _log(args, "[main] startup")
        _log_runtime_summary(args)
        if not args.par_path and not args.plots_only:
            _fail("--par-path is required unless --plots-only is used.")
        if args.plots_only:
            root_run_name = args.run_name or _make_run_name(args.par_path)
            root_dir = Path(args.output_dir) / root_run_name
            stage_dirs = [
                root_dir / "stage1_large_only",
                root_dir / "stage2_joint",
                root_dir / "stage3_image_plane",
                root_dir / "stage4_linearized_image_plane",
            ]
            if any(_has_plot_artifacts(stage_dir / "artifacts") for stage_dir in stage_dirs):
                exact_diagnostics_stage = _final_available_sequential_stage(stage_dirs)
                for stage_dir in stage_dirs:
                    if _has_plot_artifacts(stage_dir / "artifacts"):
                        _configure_debug_log(args, stage_dir.name, stage_dir)
                        _rerender_plots(args, stage_dir, exact_diagnostics_stage=exact_diagnostics_stage)
                return
            run_dir = Path(args.output_dir) / (args.run_name or _make_run_name(args.par_path))
            _configure_debug_log(args, run_dir.name, run_dir)
            if not _has_plot_artifacts(run_dir / "artifacts"):
                _fail(f"Missing saved artifacts for plots-only mode: {run_dir / 'artifacts'}")
            _rerender_plots(args, run_dir)
            return
        if args.fit_mode == FIT_MODE_SEQUENTIAL:
            _run_sequential(args)
        elif args.fit_mode == FIT_MODE_EVIDENCE_NS:
            _run_evidence_ns(args)
        else:
            run_name = args.run_name or _make_run_name(args.par_path)
            run_dir = Path(args.output_dir) / run_name
            if bool(getattr(args, "resume", False)):
                if _stage_run_complete(run_dir):
                    if bool(getattr(args, "skip_plots", False)):
                        _log(args, f"[resume] reusing completed run run_name={run_name} run_dir={run_dir} skip_plots=True")
                    else:
                        _log(args, f"[resume] refreshing completed run outputs run_name={run_name} run_dir={run_dir}")
                        _rerender_plots(args, run_dir)
                    return
                if _stage_run_checkpointed(run_dir):
                    _log(args, f"[resume] finalizing checkpointed run run_name={run_name} run_dir={run_dir}")
                    _rerender_plots(args, run_dir)
                    return
            sample_likelihood_mode = (
                SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN
                if str(args.image_plane_mode) == IMAGE_PLANE_MODE_LOCAL_JACOBIAN
                else SAMPLE_LIKELIHOOD_SOURCE
            )
            stage_args = _args_with_fit_controls(args, stage_fit_controls["stage2"])
            _run_single_stage(stage_args, str(args.fit_mode), run_name, sample_likelihood_mode=sample_likelihood_mode)
    except BaseException as exc:
        _log_exception("main", exc)
        raise
    finally:
        _close_debug_log()


if __name__ == "__main__":
    main()
