from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
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
    ArcConstraintData,
    BinData,
    BuildState,
    ChainSeed,
    COUPLED_ROLE_MASS_NORM,
    COUPLED_ROLE_SIZE,
    COUPLED_TRANSFORM_POTFILE_MASS_SIZE,
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
    potfile_mass_size_coupling_arrays as _potfile_mass_size_coupling_arrays,
    positive_lognormal_parameters as _positive_lognormal_parameters,
)
from .plotting import CAUSTIC_PLOT_GRID_SCALE_ARCSEC, _format_sequential_run_summary_text, _generate_plots_and_tables
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
DEFAULT_NUTS_INIT_BOUNDARY_FRAC = 0.02
DEFAULT_NUTS_INIT_JITTER_FRAC = 0.02
DEFAULT_SVI_STEPS = 2000
DEFAULT_SVI_LEARNING_RATE = 5.0e-3
DEFAULT_NS_MAX_SAMPLES = None
DEFAULT_NS_POSTERIOR_SAMPLES = 4096
DEFAULT_NS_DLOGZ = 1.0e-4
DEFAULT_POSTERIOR_LOGPROB_BATCH_SIZE = 512
DEFAULT_SMC_PARTICLES = 4096
DEFAULT_SMC_MCMC_KERNEL = "rmh"
SMC_MCMC_KERNELS = ("rmh", "mala")
DEFAULT_SMC_MCMC_STEPS = 4
DEFAULT_SMC_TARGET_ESS_FRAC = 0.8
DEFAULT_SMC_MAX_TEMPERATURE_STEPS = 256
DEFAULT_SMC_RMH_SCALE = 1.0
DEFAULT_SMC_MALA_STEP_SIZE = 0.05
SMC_FINAL_TEMPERATURE_TOL = 1.0e-6
SAMPLING_ENGINE_FULL = "full"
SAMPLING_ENGINE_REFRESHING_SURROGATE = "refreshing_surrogate"
SAMPLING_ENGINE_ACTIVE_SUBSET = "active_subset"
SAMPLING_ENGINES = (
    SAMPLING_ENGINE_FULL,
    SAMPLING_ENGINE_REFRESHING_SURROGATE,
    SAMPLING_ENGINE_ACTIVE_SUBSET,
)
DEFAULT_SAMPLER = "numpyro_nuts"
DEFAULT_SOURCE_SIGMA_INT_LOWER_ARCSEC = 1.0e-3
DEFAULT_SOURCE_SIGMA_INT_UPPER_ARCSEC = 2.0
DEFAULT_IMAGE_SIGMA_INT_LOWER_ARCSEC = 1.0e-3
DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC = 2.0
DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC = 1.0e-3
IMAGE_PLANE_SCATTER_PRIOR_LOG_UNIFORM = "log-uniform"
IMAGE_PLANE_SCATTER_PRIOR_LOGNORMAL = "lognormal"
IMAGE_PLANE_SCATTER_PRIORS = (
    IMAGE_PLANE_SCATTER_PRIOR_LOG_UNIFORM,
    IMAGE_PLANE_SCATTER_PRIOR_LOGNORMAL,
)
DEFAULT_IMAGE_PLANE_SCATTER_PRIOR = IMAGE_PLANE_SCATTER_PRIOR_LOG_UNIFORM
DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC = 0.3
DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA = 0.5
DEFAULT_SCALING_SCATTER_PRIOR_MEDIAN = 0.02
DEFAULT_SCALING_SCATTER_PRIOR_LOG_SIGMA = 0.5
DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC = 2.0
DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC = 10.0
DEFAULT_IMAGE_PRESENCE_STAGE4_PENALTY_WEIGHT = 2.0
DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC = 0.30
DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC = 0.10
DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS = 0.05
DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN = 0.05
DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN = 0.0
DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC = 0.0
DEFAULT_CRITICAL_DET_DIAGNOSTIC_THRESHOLD = 1.0e-2
DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS = 3
DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC = 0.3
DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE = 1.0e-3
DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE = 1.0e-6
DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC = 5.0
DEFAULT_CRITICAL_ARC_BASE_PROB = 0.10
DEFAULT_CRITICAL_ARC_MAX_PROB = 0.80
DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD = 0.20
DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS = 0.05
DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE = 1.0e-3
DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE = 1.0e-6
DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC = 20.0
CRITICAL_ARC_EIGENGAP_RELATIVE_SOFTENING = 1.0e-3
CRITICAL_ARC_EIGENGAP_VALUE_ABSOLUTE_SOFTENING = 1.0e-18
CRITICAL_ARC_SINGULAR_VALUE_FLOOR = 1.0e-12
DEFAULT_FOLD_CURVATURE_ARCSEC_INV = 1.0
DEFAULT_FOLD_CURVATURE_FINITE_DIFFERENCE_STEP_ARCSEC = 1.0e-3
DEFAULT_CAB_FINITE_DIFFERENCE_STEP_ARCSEC = 1.0e-3
DEFAULT_CAB_TANGENT_SIGMA_FLOOR_RAD = 1.0e-3
DEFAULT_CAB_CURVATURE_SIGMA_FLOOR_ARCSEC_INV = 1.0e-4
DEFAULT_CAB_LIKELIHOOD_WEIGHT_WITH_ARCS = 1.0
DEFAULT_CAB_LIKELIHOOD_WEIGHT_NO_ARCS = 0.0
CAB_MORPHOLOGY_MODEL_KEY = -1.0
CAB_FRAME_RELATIVE_SOFTENING = 1.0e-3
CAB_FRAME_VALUE_ABSOLUTE_SOFTENING = 1.0e-18
CAB_FRAME_PHYSICAL_AMBIGUITY_FRACTION = 1.0e-1
# Spread of plausible observed arc curvatures (~1/R for R ~ 5-50 arcsec); sets the
# curvature dimension of the CAB outlier density so the inlier/outlier handoff does
# not depend on the absolute unit scale of the per-row curvature sigma.
CAB_OUTLIER_CURVATURE_SIGMA_ARCSEC_INV = 0.1
DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC = 0.5
DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC = 5.0
DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC = 0.1
DEFAULT_COVARIANCE_DIAGONAL_JITTER_RELATIVE = 1.0e-8
DEFAULT_COVARIANCE_DIAGONAL_JITTER_ABSOLUTE = 1.0e-12
MIN_COVARIANCE_DETERMINANT_GUARD = 1.0e-30
DEFAULT_JACOBIAN_INVERSE_DAMPING_RELATIVE = 1.0e-12
DEFAULT_JACOBIAN_INVERSE_DAMPING_ABSOLUTE = 1.0e-24
MIN_JACOBIAN_NORMAL_DETERMINANT_GUARD = 1.0e-30
LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_GAUSSIAN = "gaussian"
LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_STUDENT_T = "student-t"
LIKELIHOOD_STABILIZER_RESIDUAL_LOSSES = (
    LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_GAUSSIAN,
    LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_STUDENT_T,
)
DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS = LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_GAUSSIAN
DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU = 4.0
DEFAULT_COSMOLOGY_OM0_LOWER = 0.05
DEFAULT_COSMOLOGY_OM0_UPPER = 0.6
DEFAULT_COSMOLOGY_W0_LOWER = -2.0
DEFAULT_COSMOLOGY_W0_UPPER = -0.3
SEQUENTIAL_FIDUCIAL_COSMOLOGY_CONFIG: dict[str, float | str] = {
    "class": "FlatLambdaCDM",
    "H0": 70.0,
    "Om0": 0.3,
    "Ode0": 0.7,
    "w0": -1.0,
}
COSMOLOGY_OM0_SAMPLE_NAME = "cosmology_Om0"
COSMOLOGY_W0_SAMPLE_NAME = "cosmology_w0"
SAFE_SCALING_EXPONENT_ABS_MIN = 1.0e-3
SAFE_RADIUS_MARGIN_ARCSEC = 1.0e-3
SAFE_RADIUS_MARGIN_KPC = 1.0e-3
SAFE_VDISP_MARGIN = 1.0e-6
VDISP_TRUNCATION_FLOOR_KM_S = 1.0e-4
NUTS_MAX_TREE_SATURATION_WARNING = 0.95
NUTS_RHAT_EXTREME_WARNING = 2.0
NUTS_MIN_ESS_PER_CHAIN_WARNING = 2.0
SVI_HEALTH_FINITE_DRAW_FRACTION_WARNING = 1.0
SVI_HEALTH_LOGPROB_SPREAD_WARNING = 1.0e3
SVI_HEALTH_CENTER_DROP_WARNING = 50.0


SVI_HEALTH_CHAIN_START_DROP_WARNING = 50.0
SVI_HEALTH_LOW_SPREAD_RATIO_WARNING = 1.0e-4
SVI_HEALTH_WORST_PARAMETER_COUNT = 5
BAD_LOG_LIKE = -1.0e30
SAMPLE_LIKELIHOOD_SOURCE = "source"
SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN = "local-jacobian"
SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE = "linearized-forward-beta-image-plane"
SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE = "forward-metric-image-plane"
SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE = "anchored-solved-forward-beta-image-plane"
SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE = "critical-arc-mixture-image-plane"
SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE = "fold-regularized-forward-beta-image-plane"
EVIDENCE_LIKELIHOOD_MODES = (
    SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
)
DEFAULT_EVIDENCE_LIKELIHOOD_MODE = SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
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
IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED = "linearized-forward-beta-blocked-image-plane"
IMAGE_PLANE_MODE_FORWARD_METRIC = "forward-metric-image-plane"
IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA = "anchored-solved-forward-beta-image-plane"
IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE = "critical-arc-mixture-image-plane"
IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA = "fold-regularized-forward-beta-image-plane"
FIT_MODE_SEQUENTIAL = "sequential"
FIT_MODE_LARGE_ONLY = "large-only"
FIT_MODE_JOINT = "joint"
FIT_MODE_EVIDENCE_NS = "evidence-ns"
FIT_METHOD_SVI = "svi"
FIT_METHOD_SVI_NUTS = "svi+nuts"
FIT_METHOD_NUTS = "nuts"
FIT_METHOD_NS = "ns"
FIT_METHOD_SMC = "smc"
JAX_DEVICE_AUTO = "auto"
JAX_DEVICE_CPU = "cpu"
JAX_DEVICE_GPU = "gpu"
JAX_DEVICE_CHOICES = (JAX_DEVICE_AUTO, JAX_DEVICE_CPU, JAX_DEVICE_GPU)


def _lenstool_ellipticite_to_axis_ratio_jax(ellipticite: float | jnp.ndarray) -> jnp.ndarray:
    safe_e = jnp.clip(jnp.asarray(ellipticite, dtype=jnp.float64), 0.0, 1.0 - 1.0e-9)
    q = jnp.sqrt((1.0 - safe_e) / (1.0 + safe_e))
    return jnp.clip(q, 1.0e-3, 1.0)


def _axis_ratio_to_lenstool_ellipticite(q: float) -> float:
    safe_q = max(1.0e-3, min(1.0, float(q)))
    return float((1.0 - safe_q * safe_q) / (1.0 + safe_q * safe_q))


@dataclass(frozen=True)
class FOVLimit:
    radius_arcsec: float | None = None
    x_min_arcsec: float | None = None
    x_max_arcsec: float | None = None
    y_min_arcsec: float | None = None
    y_max_arcsec: float | None = None

    @property
    def is_active(self) -> bool:
        return (
            self.radius_arcsec is not None
            or self.x_min_arcsec is not None
            or self.x_max_arcsec is not None
            or self.y_min_arcsec is not None
            or self.y_max_arcsec is not None
        )


@dataclass(frozen=True)
class StageFitControls:
    fit_method: str
    svi_steps: int
    warmup: int
    samples: int
    max_tree_depth: int

    def to_json(self) -> dict[str, str | int]:
        return {
            "fit_method": self.fit_method,
            "svi_steps": self.svi_steps,
            "warmup": self.warmup,
            "samples": self.samples,
            "max_tree_depth": self.max_tree_depth,
        }


@dataclass(frozen=True)
class BlockedNUTSParameterBlock:
    name: str
    indices: tuple[int, ...]


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
    constrained_image_indices: jnp.ndarray | None = None


@dataclass(frozen=True)
class TracedArcConstraintData:
    arc_ids: tuple[str, ...]
    z_arc: jnp.ndarray
    anchor_x: jnp.ndarray
    anchor_y: jnp.ndarray
    tangent_angle_rad: jnp.ndarray
    curvature_arcsec_inv: jnp.ndarray
    sigma_tangent_angle_rad: jnp.ndarray
    sigma_curvature_arcsec_inv: jnp.ndarray
    reliability: jnp.ndarray
    n_arcs: int = 0
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
    "nonfinite_rs",
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


def _positive_float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive float") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive float")
    return parsed


def _nonnegative_float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite nonnegative float") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative float")
    return parsed


def _finite_float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite float") from exc
    if not np.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite float")
    return parsed


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
    parser.add_argument(
        "--fov-limit-radius",
        type=float,
        default=None,
        metavar="ARCSEC",
        help="Ignore catalog image/member rows outside this inclusive circular FOV radius in solver arcsec coordinates.",
    )
    parser.add_argument(
        "--fov-limit-x",
        type=float,
        nargs=2,
        default=None,
        metavar=("X_LEFT", "X_RIGHT"),
        help=(
            "Ignore catalog image/member rows outside these inclusive x arcsec bounds. "
            "Values are order-insensitive."
        ),
    )
    parser.add_argument(
        "--fov-limit-y",
        type=float,
        nargs=2,
        default=None,
        metavar=("Y_BOTTOM", "Y_TOP"),
        help=(
            "Ignore catalog image/member rows outside these inclusive y arcsec bounds. "
            "Values are order-insensitive."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--jax-default-device",
        choices=JAX_DEVICE_CHOICES,
        default=JAX_DEVICE_AUTO,
        help="Default JAX device for solver setup, non-SMC sampling, validation, and plotting.",
    )
    parser.add_argument(
        "--smc-device",
        choices=JAX_DEVICE_CHOICES,
        default=JAX_DEVICE_AUTO,
        help="JAX device used only by the BlackJAX SMC sampler.",
    )
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed run/stage artifacts and continue from the first incomplete stage.",
    )
    parser.add_argument(
        "--resume-fast",
        action="store_true",
        help=(
            "Sequential-only shortcut: skip earlier stages, load their existing artifacts, "
            "and run only the final enabled stage."
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Debug mode: skip the post-fit source-plane validation summary after sampling.",
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
            "Fast post-fit diagnostics: skip exact image-position fit-quality diagnostics, caustic overlay, "
            "and exact image-position fit-quality draws while keeping source-plane and magnification summaries."
        ),
    )
    parser.add_argument(
        "--debug-sampler-diagnostics",
        action="store_true",
        help=(
            "Write bounded post-hoc NUTS diagnostics for stuck chains, including adapted step size, "
            "mass-matrix summaries, local gradients, direction scans, and critical-arc likelihood terms."
        ),
    )
    parser.add_argument(
        "--exact-image-diagnostics-stage3",
        action="store_true",
        help=(
            "Sequential-only diagnostic override: run exact image matching and residual diagnostics for "
            "stage3_image_plane even when a stage 4 image-plane stage is enabled."
        ),
    )
    parser.add_argument(
        "--image-catalog-family-cutout-image-dir",
        default=None,
        help=(
            "Optional BUFFALO/HFF image directory. When set, write an image-catalog family cutout diagnostic "
            "PDF for eligible exact image-plane stages."
        ),
    )
    parser.add_argument(
        "--image-catalog-family-cutout-image-scale",
        choices=("auto", "30mas", "60mas"),
        default="60mas",
        help="Image scale for --image-catalog-family-cutout-image-dir.",
    )
    parser.add_argument(
        "--image-catalog-family-cutout-bands",
        nargs="+",
        default=None,
        metavar="BAND",
        help=(
            "Optional three or more FITS bands for image-catalog family cutout diagnostics. "
            "Defaults to the plotting helper's RGB bands."
        ),
    )
    parser.add_argument("--image-catalog-family-cutout-rgb-q", type=_positive_float_arg, default=None)
    parser.add_argument("--image-catalog-family-cutout-rgb-stretch", type=_positive_float_arg, default=None)
    parser.add_argument("--image-catalog-family-cutout-rgb-minimum", type=_finite_float_arg, default=None)
    parser.add_argument("--image-catalog-family-cutout-rgb-red-gain", type=_positive_float_arg, default=None)
    parser.add_argument("--image-catalog-family-cutout-rgb-green-gain", type=_positive_float_arg, default=None)
    parser.add_argument("--image-catalog-family-cutout-rgb-blue-gain", type=_positive_float_arg, default=None)
    parser.add_argument(
        "--kappa-true-fits",
        default=None,
        help=(
            "Optional true convergence FITS image. When set, write kappa_comparison.pdf "
            "at --caustic-source-redshift."
        ),
    )
    parser.add_argument(
        "--corner-overlay-bayes-dat",
        default=None,
        help=(
            "Optional Lenstool bayes.dat chain to overlay as unfilled contours on matching corner plots. "
            "Missing or unmatched columns are skipped during plotting."
        ),
    )
    parser.add_argument(
        "--corner-overlay-best-par",
        default=None,
        help=(
            "Optional Lenstool best.par file to overlay as a gold marker on matching corner plots. "
            "Potfile reference values are inferred from optimized member potentials when possible."
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
        "--quiet",
        action="store_true",
        help="Suppress stage logs and NS progress output; NumPyro SVI/NUTS progress bars may still render.",
    )
    parser.add_argument(
        "--sampling-engine",
        choices=SAMPLING_ENGINES,
        default=SAMPLING_ENGINE_REFRESHING_SURROGATE,
        help=(
            "Use the exact full likelihood, a first-order inactive-scaling surrogate, "
            "or an exact active-scaling subset that omits inactive scaling galaxies during fitting."
        ),
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
        help="Legacy no-op; scaling-scatter hyperparameters now use unbounded lognormal priors.",
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
        choices=(FIT_METHOD_SVI, FIT_METHOD_SVI_NUTS, FIT_METHOD_NUTS, FIT_METHOD_NS, FIT_METHOD_SMC),
        default=[FIT_METHOD_SVI_NUTS],
        metavar="{svi,svi+nuts,nuts,ns,smc}",
        help=(
            "Use SVI only or use SVI to initialize optional NUTS sampling. "
            "The ns value is accepted only for backwards-compatible parsing and is reserved for --fit-mode evidence-ns; "
            "nuts and smc are accepted only for non-blocked sequential stage 4 image-plane modes. "
            "In sequential image-plane runs, pass one value for all sampled stages, two values for stage 2 and "
            "stage 3, or three values when a final stage 4 image-plane mode is enabled."
        ),
    )
    parser.add_argument(
        "--image-plane-mode",
        choices=(
            IMAGE_PLANE_MODE_NONE,
            IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
            IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
            IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED,
            IMAGE_PLANE_MODE_FORWARD_METRIC,
            IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
            IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
            IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
        ),
        default=IMAGE_PLANE_MODE_NONE,
        help=(
            "Optional image-plane refinement mode. 'none' keeps the standard source-plane workflow; "
            "'local-jacobian' adds a differentiable local image-plane approximation stage; "
            "'linearized-forward-beta-image-plane' adds the explicit-beta linearized image-plane stage; "
            "'linearized-forward-beta-blocked-image-plane' uses the same explicit-beta stage with blocked NUTS; "
            "'forward-metric-image-plane' adds an explicit-beta stage that scores source residuals "
            "with the proposal-current forward image covariance; "
            "'anchored-solved-forward-beta-image-plane' adds a fixed-step local image-plane solve seeded at "
            "each observed image; "
            "'critical-arc-mixture-image-plane' adds a fast point/critical-arc mixture image-plane stage; "
            "'fold-regularized-forward-beta-image-plane' adds an experimental local signed-fold "
            "root-distance image-plane stage near singular Jacobians."
        ),
    )
    parser.add_argument(
        "--skip-stage3-image-plane-local-jacobian",
        action="store_true",
        help="Skip the local-Jacobian stage 3 before a final stage 4 image-plane mode.",
    )
    parser.add_argument(
        "--critical-det-diagnostic-threshold",
        type=_positive_float_arg,
        default=DEFAULT_CRITICAL_DET_DIAGNOSTIC_THRESHOLD,
        help=(
            "Post-stage-3 diagnostic threshold for tiny local lensing Jacobian determinants. "
            "Images with abs(det A) below this value are logged and written to critical_det_images.csv."
        ),
    )
    parser.add_argument(
        "--skip-critical-det-diagnostic",
        action="store_true",
        help="Disable the post-stage-3 tiny-detA image diagnostic before stage 4.",
    )
    parser.add_argument(
        "--start-at-stage3",
        action="store_true",
        help="Start a sequential image-plane workflow at stage 3; with --resume-fast, reuse stage 3 artifacts for later stages.",
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
        "--anchored-image-plane-solve-steps",
        type=int,
        default=DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS,
        help=(
            "Fixed damped Newton/LM iterations per observed image for anchored-solved stage 4. "
            "Use 0 for the fast observed-anchor linearized LM approximation."
        ),
    )
    parser.add_argument(
        "--anchored-image-plane-trust-radius-arcsec",
        type=_positive_float_arg,
        default=DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
        help="Smooth per-iteration image-plane trust radius for anchored-solved stage 4.",
    )
    parser.add_argument(
        "--anchored-image-plane-lm-damping-relative",
        type=_positive_float_arg,
        default=DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
        help="Relative LM damping added to A.T A in anchored-solved stage 4.",
    )
    parser.add_argument(
        "--anchored-image-plane-lm-damping-absolute",
        type=_positive_float_arg,
        default=DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
        help="Absolute LM damping added to A.T A in anchored-solved stage 4.",
    )
    parser.add_argument(
        "--critical-arc-critical-direction-sigma-arcsec",
        type=_positive_float_arg,
        default=DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC,
        help="Broad along-arc image-plane sigma for critical-arc mixture stage 4.",
    )
    parser.add_argument(
        "--critical-arc-base-prob",
        type=float,
        default=DEFAULT_CRITICAL_ARC_BASE_PROB,
        help="Baseline prior probability that a catalog row is a critical-arc support point.",
    )
    parser.add_argument(
        "--critical-arc-max-prob",
        type=float,
        default=DEFAULT_CRITICAL_ARC_MAX_PROB,
        help="Maximum prior probability for the critical-arc branch near singular local Jacobians.",
    )
    parser.add_argument(
        "--critical-arc-singular-threshold",
        type=_positive_float_arg,
        default=DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD,
        help="Smallest-singular-value threshold where the critical-arc branch prior starts increasing.",
    )
    parser.add_argument(
        "--critical-arc-singular-softness",
        type=_positive_float_arg,
        default=DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS,
        help="Softness for the critical-arc prior transition as the smallest singular value approaches zero.",
    )
    parser.add_argument(
        "--critical-arc-lm-damping-relative",
        type=_positive_float_arg,
        default=DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE,
        help="Relative LM damping added to A.T A for critical-arc mixture image-plane displacements.",
    )
    parser.add_argument(
        "--critical-arc-lm-damping-absolute",
        type=_positive_float_arg,
        default=DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE,
        help="Absolute LM damping added to A.T A for critical-arc mixture image-plane displacements.",
    )
    parser.add_argument(
        "--critical-arc-lm-trust-radius-arcsec",
        type=_positive_float_arg,
        default=DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC,
        help="Large smooth finite guard radius for critical-arc mixture LM image-plane displacements.",
    )
    parser.add_argument(
        "--arc-aware-noncritical-support-radius-arcsec",
        type=_positive_float_arg,
        default=DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC,
        help="Maximum support-curve distance for arc-aware image recovery validation.",
    )
    parser.add_argument(
        "--arc-aware-max-arclength-arcsec",
        type=_positive_float_arg,
        default=DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC,
        help="Maximum traced arclength in each direction for arc-aware image recovery validation.",
    )
    parser.add_argument(
        "--arc-aware-curve-step-arcsec",
        type=_positive_float_arg,
        default=DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC,
        help="Curve tracing step size for arc-aware image recovery validation.",
    )
    parser.add_argument(
        "--fold-curvature-arcsec-inv",
        type=_positive_float_arg,
        default=DEFAULT_FOLD_CURVATURE_ARCSEC_INV,
        help=(
            "Fallback local fold curvature scale in arcsec^-1 for direct fold-regularized helper use."
        ),
    )
    parser.add_argument(
        "--cab-likelihood-weight",
        type=float,
        default=None,
        help=(
            "Weight for optional CAB-informed arc morphology constraints. "
            "Defaults to 1.0 when parsed arc constraints exist and 0.0 otherwise."
        ),
    )
    parser.add_argument(
        "--cab-finite-difference-step-arcsec",
        type=_positive_float_arg,
        default=DEFAULT_CAB_FINITE_DIFFERENCE_STEP_ARCSEC,
        help="Finite-difference step in arcsec for CAB tangent-direction curvature.",
    )
    parser.add_argument(
        "--cab-tangent-sigma-floor-rad",
        type=_positive_float_arg,
        default=DEFAULT_CAB_TANGENT_SIGMA_FLOOR_RAD,
        help="Minimum tangent-angle sigma in radians for CAB morphology constraints.",
    )
    parser.add_argument(
        "--cab-curvature-sigma-floor-arcsec-inv",
        type=_positive_float_arg,
        default=DEFAULT_CAB_CURVATURE_SIGMA_FLOOR_ARCSEC_INV,
        help="Minimum curvature sigma in arcsec^-1 for CAB morphology constraints.",
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
            "Likelihood target for --fit-mode evidence-ns. "
            "linearized-forward-beta-image-plane samples one source position per family and fits "
            "the linearized image-plane residuals."
        ),
    )
    parser.add_argument(
        "--image-plane-scatter-upper-arcsec",
        type=float,
        default=DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC,
        help="Upper bound for the sampled stage-4 intrinsic image-plane scatter parameter.",
    )
    parser.add_argument(
        "--image-plane-scatter-floor-arcsec",
        type=float,
        default=DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC,
        help="Lower bound for the sampled stage-4 intrinsic image-plane scatter parameter.",
    )
    parser.add_argument(
        "--image-plane-scatter-prior",
        choices=IMAGE_PLANE_SCATTER_PRIORS,
        default=DEFAULT_IMAGE_PLANE_SCATTER_PRIOR,
        help=(
            "Prior for the stage-4 intrinsic image-plane scatter. log-uniform preserves legacy behavior; "
            "lognormal applies a Normal prior to log(image_sigma_int)."
        ),
    )
    parser.add_argument(
        "--image-plane-scatter-prior-median-arcsec",
        type=float,
        default=DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC,
        help="Median image_sigma_int for --image-plane-scatter-prior lognormal.",
    )
    parser.add_argument(
        "--image-plane-scatter-prior-log-sigma",
        type=float,
        default=DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA,
        help="Standard deviation of log(image_sigma_int) for --image-plane-scatter-prior lognormal.",
    )
    parser.add_argument(
        "--fix-image-sigma-int-arcsec",
        type=_nonnegative_float_arg,
        default=None,
        help=(
            "Use a deterministic intrinsic image-plane scatter instead of sampling image.sigma_int. "
            "Accepts 0.0 to disable extra intrinsic image scatter."
        ),
    )
    parser.add_argument(
        "--image-presence-penalty-weight",
        type=float,
        default=None,
        help=(
            "Weight for the smooth observed-image presence penalty in explicit-beta image-plane modes. "
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
        "--likelihood-stabilizer-max-gain",
        type=float,
        default=DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN,
        help=(
            "Shared optional likelihood gain cap for source-plane, local-Jacobian, and linearized image-plane modes. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--likelihood-stabilizer-max-residual-arcsec",
        type=float,
        default=DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC,
        help=(
            "Shared optional smooth tanh cap for source-plane, local-Jacobian, and linearized image-plane residuals in arcsec. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--likelihood-stabilizer-residual-loss",
        choices=LIKELIHOOD_STABILIZER_RESIDUAL_LOSSES,
        default=DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS,
        help=(
            "Shared inlier residual density for source-plane, local-Jacobian, and linearized-forward-beta "
            "image-plane modes. Marginalized evidence remains Gaussian."
        ),
    )
    parser.add_argument(
        "--likelihood-stabilizer-student-t-nu",
        type=float,
        default=DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU,
        help="Degrees of freedom for --likelihood-stabilizer-residual-loss student-t.",
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
            "In sequential runs this is applied only to stage 3 and stage 4; "
            "stage 1 and stage 2 use fixed FlatLambdaCDM(H0=70, Om0=0.3)."
        ),
    )
    parser.add_argument(
        "--cosmology-init-om0",
        type=float,
        default=None,
        help="Optional SVI start value for sampled flat-wCDM Omega_m; ignored unless --fit-cosmology-flat-wcdm is on.",
    )
    parser.add_argument(
        "--cosmology-init-w0",
        type=float,
        default=None,
        help="Optional SVI start value for sampled flat-wCDM w0; ignored unless --fit-cosmology-flat-wcdm is on.",
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
            "two values for stage 2 and stage 3, or three values when a final stage 4 image-plane mode is enabled."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[DEFAULT_SAMPLES],
        help=(
            "Posterior draws per chain. In sequential image-plane runs, pass one value for all sampled stages, "
            "two values for stage 2 and stage 3, or three values when a final stage 4 image-plane mode is enabled."
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
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        nargs="+",
        default=[DEFAULT_MAX_TREE_DEPTH],
        help=(
            "NUTS max tree depth. In sequential image-plane runs, pass one value for all sampled stages, "
            "two values for stage 2 and stage 3, or three values when a final stage 4 image-plane mode is enabled."
        ),
    )
    parser.add_argument("--target-accept", type=float, default=DEFAULT_TARGET_ACCEPT)
    parser.add_argument(
        "--initial-step-size",
        type=float,
        default=DEFAULT_INITIAL_STEP_SIZE,
        help="Initial NUTS step size before warmup adaptation. Small defaults avoid invalid dPIE states during early warmup.",
    )
    parser.add_argument(
        "--dense-mass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dense mass-matrix adaptation for NumPyro NUTS. Pass --no-dense-mass for diagonal mass.",
    )
    parser.add_argument(
        "--potfile-mass-size-reparam",
        action="store_true",
        default=False,
        help=(
            "Opt in to a NumPyro SVI/NUTS-only potfile sigma/cutkpc reparameterization that samples "
            "prior-standardized mass-size coordinates while preserving the original sigma and cutkpc priors."
        ),
    )
    parser.add_argument(
        "--blocked-nuts-cycles",
        type=int,
        default=None,
        help=(
            "Production blocked-NUTS cycles for linearized-forward-beta-blocked-image-plane. "
            "Defaults to the stage sample count."
        ),
    )
    parser.add_argument(
        "--blocked-nuts-pilot-warmup",
        type=int,
        default=None,
        help=(
            "Pilot warmup steps for each blocked-NUTS conditional block. "
            "Defaults to the stage warmup count."
        ),
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
        nargs="+",
        default=[DEFAULT_SVI_STEPS],
        help=(
            "Number of SVI steps used to initialize NUTS chains. In sequential image-plane runs, "
            "pass one value for all sampled stages, two values for stage 2 and stage 3, or three "
            "values when a final stage 4 image-plane mode is enabled."
        ),
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
        "--smc-particles",
        type=int,
        default=DEFAULT_SMC_PARTICLES,
        help="BlackJAX SMC particle count for non-blocked sequential stage 4 when --fit-method smc is selected.",
    )
    parser.add_argument(
        "--smc-mcmc-kernel",
        choices=SMC_MCMC_KERNELS,
        default=DEFAULT_SMC_MCMC_KERNEL,
        help="BlackJAX mutation kernel for stage-4 SMC.",
    )
    parser.add_argument(
        "--smc-mcmc-steps",
        type=int,
        default=DEFAULT_SMC_MCMC_STEPS,
        help="Mutation steps per adaptive SMC temperature update.",
    )
    parser.add_argument(
        "--smc-target-ess-frac",
        type=float,
        default=DEFAULT_SMC_TARGET_ESS_FRAC,
        help="Target ESS fraction used to choose adaptive SMC temperature increments.",
    )
    parser.add_argument(
        "--smc-max-temperature-steps",
        type=int,
        default=DEFAULT_SMC_MAX_TEMPERATURE_STEPS,
        help="Maximum adaptive SMC temperature updates before requiring posterior temperature 1.",
    )
    parser.add_argument(
        "--smc-rmh-scale",
        type=float,
        default=DEFAULT_SMC_RMH_SCALE,
        help="Isotropic normalized-space proposal scale for the RMH SMC mutation kernel.",
    )
    parser.add_argument(
        "--smc-mala-step-size",
        type=float,
        default=DEFAULT_SMC_MALA_STEP_SIZE,
        help="Normalized-space MALA step size for the SMC mutation kernel.",
    )
    parser.add_argument(
        "--caustic-plot-grid-scale-arcsec",
        type=_positive_float_arg,
        default=CAUSTIC_PLOT_GRID_SCALE_ARCSEC,
        help=(
            "Arcsec grid spacing for caustic overlay and absolute magnification plots. "
            f"Defaults to {CAUSTIC_PLOT_GRID_SCALE_ARCSEC:g} arcsec."
        ),
    )
    parser.add_argument(
        "--caustic-source-redshift",
        type=float,
        default=9.0,
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
            f"jax_default_device={getattr(args, 'jax_default_device', JAX_DEVICE_AUTO)} "
            f"smc_device={getattr(args, 'smc_device', JAX_DEVICE_AUTO)} output_dir={args.output_dir}"
        ),
    )
    _log(
        args,
        (
            f"[runtime] par_path={args.par_path} fit_mode={args.fit_mode} fit_method={args.fit_method} "
            f"sampling_engine={args.sampling_engine} sample_likelihood_mode={getattr(args, 'sample_likelihood_mode', SAMPLE_LIKELIHOOD_SOURCE)} "
            f"image_plane_mode={getattr(args, 'image_plane_mode', IMAGE_PLANE_MODE_NONE)} warmup={args.warmup} samples={args.samples} "
            f"anchored_image_plane_solve_steps={getattr(args, 'anchored_image_plane_solve_steps', DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS)} "
            f"anchored_image_plane_trust_radius_arcsec={getattr(args, 'anchored_image_plane_trust_radius_arcsec', DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC)} "
            f"anchored_image_plane_lm_damping_relative={getattr(args, 'anchored_image_plane_lm_damping_relative', DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE)} "
            f"anchored_image_plane_lm_damping_absolute={getattr(args, 'anchored_image_plane_lm_damping_absolute', DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE)} "
            f"critical_arc_critical_direction_sigma_arcsec={getattr(args, 'critical_arc_critical_direction_sigma_arcsec', DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC)} "
            f"critical_arc_prob=({getattr(args, 'critical_arc_base_prob', DEFAULT_CRITICAL_ARC_BASE_PROB)},"
            f"{getattr(args, 'critical_arc_max_prob', DEFAULT_CRITICAL_ARC_MAX_PROB)}) "
            f"arc_aware_noncritical_support_radius_arcsec={getattr(args, 'arc_aware_noncritical_support_radius_arcsec', DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC)} "
            f"arc_aware_max_arclength_arcsec={getattr(args, 'arc_aware_max_arclength_arcsec', DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC)} "
            f"arc_aware_curve_step_arcsec={getattr(args, 'arc_aware_curve_step_arcsec', DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC)} "
            f"fold_curvature_arcsec_inv={getattr(args, 'fold_curvature_arcsec_inv', DEFAULT_FOLD_CURVATURE_ARCSEC_INV)} "
            f"source_position_parameterization={getattr(args, 'source_position_parameterization', SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED)} "
            f"evidence_likelihood_mode={getattr(args, 'evidence_likelihood_mode', DEFAULT_EVIDENCE_LIKELIHOOD_MODE)} "
            f"evidence_source_prior_sigma_arcsec={getattr(args, 'evidence_source_prior_sigma_arcsec', None)} "
            f"evidence_source_prior_mean=({getattr(args, 'evidence_source_prior_mean_x_arcsec', 0.0)},"
            f"{getattr(args, 'evidence_source_prior_mean_y_arcsec', 0.0)}) "
            f"fit_cosmology_flat_wcdm={bool(getattr(args, 'fit_cosmology_flat_wcdm', False))} "
            f"potfile_mass_size_reparam={bool(getattr(args, 'potfile_mass_size_reparam', False))} "
            f"chains={args.chains} thin={args.thin} skip_validation={args.skip_validation} "
            f"ns_num_live_points={getattr(args, 'ns_num_live_points', None)} "
            f"ns_max_samples={getattr(args, 'ns_max_samples', DEFAULT_NS_MAX_SAMPLES)} "
            f"ns_posterior_samples={DEFAULT_NS_POSTERIOR_SAMPLES} "
            f"ns_dlogz={getattr(args, 'ns_dlogz', DEFAULT_NS_DLOGZ)} "
            f"image_plane_scatter_floor_arcsec={getattr(args, 'image_plane_scatter_floor_arcsec', DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)} "
            f"image_plane_scatter_prior={getattr(args, 'image_plane_scatter_prior', DEFAULT_IMAGE_PLANE_SCATTER_PRIOR)} "
            f"image_plane_scatter_prior_median_arcsec={getattr(args, 'image_plane_scatter_prior_median_arcsec', DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC)} "
            f"image_plane_scatter_prior_log_sigma={getattr(args, 'image_plane_scatter_prior_log_sigma', DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA)} "
            f"fix_image_sigma_int_arcsec={getattr(args, 'fix_image_sigma_int_arcsec', None)} "
            f"image_presence_penalty_weight={getattr(args, 'image_presence_penalty_weight', None)} "
            f"image_presence_match_radius_arcsec={getattr(args, 'image_presence_match_radius_arcsec', DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC)} "
            f"image_presence_temperature_arcsec={getattr(args, 'image_presence_temperature_arcsec', DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)} "
            f"likelihood_stabilizer_max_gain={getattr(args, 'likelihood_stabilizer_max_gain', DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN)} "
            f"likelihood_stabilizer_max_residual_arcsec={getattr(args, 'likelihood_stabilizer_max_residual_arcsec', DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC)} "
            f"likelihood_stabilizer_residual_loss={getattr(args, 'likelihood_stabilizer_residual_loss', DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS)} "
            f"likelihood_stabilizer_student_t_nu={getattr(args, 'likelihood_stabilizer_student_t_nu', DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU)} "
            f"skip_plots={args.skip_plots} quick_diagnostics={bool(getattr(args, 'quick_diagnostics', False))} "
            f"plots_only={args.plots_only}"
        ),
    )


def _resolve_jax_device(kind: str, *, flag_name: str) -> Any | None:
    requested = str(kind)
    if requested == JAX_DEVICE_AUTO:
        return None
    if requested not in JAX_DEVICE_CHOICES:
        raise ValueError(f"{flag_name} must be one of {', '.join(JAX_DEVICE_CHOICES)}.")
    try:
        devices = jax.devices(requested)
    except RuntimeError as exc:
        available = ", ".join(str(device) for device in jax.devices()) or "none"
        raise ValueError(
            f"{flag_name}={requested} requested, but no JAX {requested!r} backend is available. "
            f"Visible devices: {available}."
        ) from exc
    if not devices:
        available = ", ".join(str(device) for device in jax.devices()) or "none"
        raise ValueError(
            f"{flag_name}={requested} requested, but no JAX {requested!r} device is available. "
            f"Visible devices: {available}."
        )
    return devices[0]


def _resolve_jax_device_for_args(args: argparse.Namespace, attr_name: str, *, flag_name: str) -> Any | None:
    return _resolve_jax_device(str(getattr(args, attr_name, JAX_DEVICE_AUTO)), flag_name=flag_name)


def _validate_jax_device_controls(args: argparse.Namespace) -> None:
    for attr_name, flag_name in (
        ("jax_default_device", "--jax-default-device"),
        ("smc_device", "--smc-device"),
    ):
        try:
            _resolve_jax_device_for_args(args, attr_name, flag_name=flag_name)
        except ValueError as exc:
            _fail(str(exc))


def _jax_device_context(device: Any | None):
    return nullcontext() if device is None else jax.default_device(device)


def _jax_device_label(device: Any | None) -> str:
    if device is None:
        return JAX_DEVICE_AUTO
    platform = str(getattr(device, "platform", "unknown"))
    device_id = getattr(device, "id", None)
    kind = str(getattr(device, "device_kind", type(device).__name__))
    id_part = "" if device_id is None else f":{device_id}"
    return f"{platform}{id_part} ({kind})"


def _jax_device_backend(device: Any | None) -> str:
    return JAX_DEVICE_AUTO if device is None else str(getattr(device, "platform", "unknown"))


def _log_jax_device_policy(args: argparse.Namespace, default_device: Any | None, smc_device: Any | None) -> None:
    _log(
        args,
        (
            f"[runtime] jax_default_device={getattr(args, 'jax_default_device', JAX_DEVICE_AUTO)}"
            f" resolved={_jax_device_label(default_device)} "
            f"smc_device={getattr(args, 'smc_device', JAX_DEVICE_AUTO)}"
            f" resolved={_jax_device_label(smc_device)}"
        ),
    )


def _compile_evaluator_source_loglike(evaluator: Any, *, device: Any | None = None) -> bool:
    impl = getattr(evaluator, "_source_loglike_impl", None)
    if impl is None:
        return False
    with _jax_device_context(device):
        evaluator._source_loglike_fn = jax.jit(impl)
    return True


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
    potfile_mass_size_groups = _potfile_mass_size_group_count(state.parameter_specs)
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
            f"positive={positive_transforms} "
            f"potfile_mass_size_reparam={bool(getattr(args, 'potfile_mass_size_reparam', False))} "
            f"potfile_mass_size_groups={potfile_mass_size_groups}"
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


def _image_presence_curvature_warning_item(evaluator: Any, image_presence_weight: float) -> str | None:
    if image_presence_weight <= 0.0:
        return None
    try:
        temperature = float(
            getattr(evaluator, "image_presence_temperature_arcsec", DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)
        )
    except (TypeError, ValueError):
        return None
    if not np.isfinite(temperature) or temperature <= 0.0:
        return None
    sigma_values: list[float] = []
    for bin_item in getattr(getattr(evaluator, "state", None), "bin_data", []) or []:
        if hasattr(bin_item, "sigma_per_image"):
            sigma_values.extend(np.asarray(bin_item.sigma_per_image, dtype=float).reshape(-1).tolist())
    sigma_array = np.asarray(sigma_values, dtype=float)
    finite_sigma = sigma_array[np.isfinite(sigma_array)]
    if finite_sigma.size == 0:
        return None
    scatter_floor = max(
        float(getattr(evaluator, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)),
        0.0,
    )
    covariance_floor = max(float(getattr(evaluator, "source_plane_covariance_floor", 0.0)), 0.0)
    sigma_eff2_floor = float(np.median(finite_sigma**2 + scatter_floor**2 + covariance_floor))
    curvature_ratio = image_presence_weight * sigma_eff2_floor / max(temperature**2, 1.0e-18)
    if not np.isfinite(curvature_ratio) or curvature_ratio < 1.0:
        return None
    return (
        "image_presence_penalty=curvature may dominate Gaussian beta transport "
        f"ratio~{curvature_ratio:.4g} temperature_arcsec={temperature:.4g}"
    )


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
    if str(getattr(evaluator, "sampling_engine", SAMPLING_ENGINE_FULL)) == SAMPLING_ENGINE_ACTIVE_SUBSET:
        inactive = _count_items(getattr(evaluator, "inactive_scaling_component_indices", []))
        if inactive > 0:
            items.append(
                "active_subset=fit target omits inactive scaling potentials "
                f"inactive_scaling={inactive}"
            )
    if family_count > 0 and bin_count < family_count:
        items.append(f"z_bins=active grouped_families={family_count} bins={bin_count}")
    if bool(getattr(evaluator, "quick_diagnostics", False)):
        items.append("quick_diagnostics=active exact post-fit image-position diagnostics skipped")

    sample_likelihood_mode = str(getattr(evaluator, "sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE))
    if sample_likelihood_mode == SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN:
        items.append("sample_likelihood=local-jacobian local image-plane covariance")
    elif sample_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE:
        items.append("sample_likelihood=linearized-forward-beta-image-plane local linear image correction")
    elif sample_likelihood_mode == SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE:
        items.append("sample_likelihood=forward-metric-image-plane current forward image covariance")
    elif sample_likelihood_mode == SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE:
        items.append(
            "sample_likelihood=fold-regularized-forward-beta-image-plane "
            f"fold_curvature_arcsec_inv={float(getattr(evaluator, 'fold_curvature_arcsec_inv', DEFAULT_FOLD_CURVATURE_ARCSEC_INV)):.4g}"
        )
    elif sample_likelihood_mode == SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE:
        items.append(
            "sample_likelihood=anchored-solved-forward-beta-image-plane "
            f"fixed_step_local_solve steps={int(getattr(evaluator, 'anchored_image_plane_solve_steps', DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS))}"
        )
    elif sample_likelihood_mode == SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE:
        items.append(
            "sample_likelihood=critical-arc-mixture-image-plane "
            f"critical_direction_sigma_arcsec={float(getattr(evaluator, 'critical_arc_critical_direction_sigma_arcsec', DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC)):.4g} "
            f"arc_support_radius_arcsec={float(getattr(evaluator, 'arc_aware_noncritical_support_radius_arcsec', DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC)):.4g} "
            f"arc_max_arclength_arcsec={float(getattr(evaluator, 'arc_aware_max_arclength_arcsec', DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC)):.4g}"
        )
    if sample_likelihood_mode in {
        SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
        SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
        SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
    }:
        image_presence_weight = float(getattr(evaluator, "image_presence_penalty_weight", 0.0))
        if image_presence_weight > 0.0:
            items.append(f"image_presence_penalty=active weight={image_presence_weight:.4g}")
            if (
                str(getattr(evaluator, "source_position_parameterization", SOURCE_POSITION_PARAMETERIZATION_DIRECT))
                == SOURCE_POSITION_PARAMETERIZATION_CONDITIONAL_WHITENED
            ):
                items.append("image_presence_penalty=non-Gaussian conditional transport is approximate")
            curvature_warning = _image_presence_curvature_warning_item(evaluator, image_presence_weight)
            if curvature_warning is not None:
                items.append(curvature_warning)
    max_gain = float(getattr(evaluator, "likelihood_stabilizer_max_gain", 0.0))
    if max_gain > 0.0:
        items.append(f"likelihood_gain_stabilizer=active max_gain={max_gain:.4g}")
    scatter_floor = float(getattr(evaluator, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC))
    if scatter_floor > 0.0 and _sample_likelihood_uses_image_scatter(sample_likelihood_mode):
        items.append(f"image_scatter_support_floor=active floor_arcsec={scatter_floor:.4g}")
    max_residual = float(getattr(evaluator, "likelihood_stabilizer_max_residual_arcsec", 0.0))
    if max_residual > 0.0:
        items.append(f"likelihood_residual_cap=tanh max_arcsec={max_residual:.4g}")
    residual_loss = str(
        getattr(evaluator, "likelihood_stabilizer_residual_loss", DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS)
    )
    if residual_loss != DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS:
        items.append(
            f"likelihood_residual_loss={residual_loss} "
            f"nu={float(getattr(evaluator, 'likelihood_stabilizer_student_t_nu', DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU)):.4g}"
        )

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
    if (
        (
            bool(getattr(evaluator, "surrogate_enabled", False))
            or str(getattr(evaluator, "sampling_engine", SAMPLING_ENGINE_FULL)) == SAMPLING_ENGINE_ACTIVE_SUBSET
        )
        and inactive_scaling_count > 0
    ):
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
    max_gain = float(getattr(evaluator, "likelihood_stabilizer_max_gain", DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN))
    max_residual = float(
        getattr(evaluator, "likelihood_stabilizer_max_residual_arcsec", DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC)
    )
    residual_loss = str(
        getattr(evaluator, "likelihood_stabilizer_residual_loss", DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS)
    )
    stabilizer_active = (
        max_gain > 0.0
        or max_residual > 0.0
        or residual_loss != DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS
    )
    sample_likelihood_mode = str(getattr(evaluator, "sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE))
    image_presence_weight = float(getattr(evaluator, "image_presence_penalty_weight", 0.0))
    if (
        stabilizer_active
        or image_presence_weight > 0.0
        or sample_likelihood_mode in {
            SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
            SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
            SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
            SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
            SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
        }
    ):
        _log(
            args,
            (
                "[likelihood-stabilizer] "
                f"sample_likelihood={sample_likelihood_mode} "
                f"weight={image_presence_weight:.4g} "
                f"match_radius_arcsec={float(getattr(evaluator, 'image_presence_match_radius_arcsec', DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC)):.4g} "
                f"temperature_arcsec={float(getattr(evaluator, 'image_presence_temperature_arcsec', DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)):.4g} "
                f"count_softness={float(getattr(evaluator, 'image_presence_count_softness', DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS)):.4g} "
                f"count_margin={float(getattr(evaluator, 'image_presence_count_margin', DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN)):.4g} "
                f"image_scatter_support_floor_arcsec={float(getattr(evaluator, 'image_plane_scatter_floor_arcsec', DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)):.4g} "
                f"max_gain={max_gain:.4g} "
                f"max_residual_arcsec={max_residual:.4g} "
                f"residual_loss={residual_loss} "
                f"student_t_nu={float(getattr(evaluator, 'likelihood_stabilizer_student_t_nu', DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU)):.4g} "
                f"anchored_steps={int(getattr(evaluator, 'anchored_image_plane_solve_steps', DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS))} "
                f"anchored_trust_radius_arcsec={float(getattr(evaluator, 'anchored_image_plane_trust_radius_arcsec', DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC)):.4g} "
                f"critical_arc_critical_direction_sigma_arcsec={float(getattr(evaluator, 'critical_arc_critical_direction_sigma_arcsec', DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC)):.4g} "
                f"arc_aware_noncritical_support_radius_arcsec={float(getattr(evaluator, 'arc_aware_noncritical_support_radius_arcsec', DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC)):.4g} "
                f"arc_aware_max_arclength_arcsec={float(getattr(evaluator, 'arc_aware_max_arclength_arcsec', DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC)):.4g} "
                f"arc_aware_curve_step_arcsec={float(getattr(evaluator, 'arc_aware_curve_step_arcsec', DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC)):.4g} "
                f"fold_curvature_arcsec_inv={float(getattr(evaluator, 'fold_curvature_arcsec_inv', DEFAULT_FOLD_CURVATURE_ARCSEC_INV)):.4g}"
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
    if str(getattr(evaluator, "sampling_engine", SAMPLING_ENGINE_FULL)) == SAMPLING_ENGINE_ACTIVE_SUBSET:
        _log(
            args,
            (
                "[active-subset] posterior target omits inactive scaling potentials "
                f"active={len(evaluator.active_scaling_component_indices)} "
                f"ignored_inactive={len(evaluator.inactive_scaling_component_indices)} "
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
    max_steps = float(2 ** _max_tree_depth_for_args(args) - 1)
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
        max_tree_depth = _max_tree_depth_for_args(args)
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


NUTS_BASE_EXTRA_FIELDS: tuple[str, ...] = ("accept_prob", "diverging", "num_steps", "potential_energy")
NUTS_DEBUG_EXTRA_FIELDS: tuple[str, ...] = (
    "energy",
    "adapt_state.step_size",
    "adapt_state.inverse_mass_matrix",
)


def _sampler_debug_extra_field_names(args: argparse.Namespace) -> tuple[str, ...]:
    fields = list(NUTS_BASE_EXTRA_FIELDS)
    if bool(getattr(args, "debug_sampler_diagnostics", False)):
        fields.extend(field for field in NUTS_DEBUG_EXTRA_FIELDS if field not in fields)
    return tuple(fields)


def _filter_extra_field_by_chain_mask(value: Any, valid_chain_mask: np.ndarray) -> Any:
    if isinstance(value, dict):
        return {key: _filter_extra_field_by_chain_mask(item, valid_chain_mask) for key, item in value.items()}
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] == valid_chain_mask.shape[0]:
        return array[valid_chain_mask]
    return array


def _finite_array_stats(values: Any) -> dict[str, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"min": float("nan"), "median": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _chain_draw_array(values: Any, n_chains: int, n_draws: int) -> np.ndarray | None:
    if values is None:
        return None
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim == 0:
        return np.full((n_chains, n_draws), float(array), dtype=float)
    if array.ndim == 1:
        if n_chains == 1 and array.size == n_draws:
            return array.reshape(1, n_draws)
        if n_chains > 0 and n_draws > 0 and array.size == n_chains * n_draws:
            return array.reshape(n_chains, n_draws)
        if array.size == n_draws:
            return np.tile(array.reshape(1, n_draws), (n_chains, 1))
        return None
    if array.ndim >= 2 and array.shape[0] == n_chains:
        if array.shape[1] == n_draws:
            return array.reshape(n_chains, n_draws, *array.shape[2:])
        return array
    return None


def _mass_matrix_chain_summary(value: Any, chain_index: int, n_chains: int) -> dict[str, Any]:
    if value is None:
        return {
            "inverse_mass_block_count": 0,
            "inverse_mass_total_dim": 0,
            "inverse_mass_diag_min": float("nan"),
            "inverse_mass_diag_median": float("nan"),
            "inverse_mass_diag_max": float("nan"),
            "inverse_mass_eig_min": float("nan"),
            "inverse_mass_eig_max": float("nan"),
        }
    blocks = value if isinstance(value, dict) else {"mass": value}
    diag_values: list[np.ndarray] = []
    eig_min_values: list[float] = []
    eig_max_values: list[float] = []
    total_dim = 0
    block_count = 0
    for block_value in blocks.values():
        try:
            array = np.asarray(block_value, dtype=float)
        except (TypeError, ValueError):
            continue
        if array.size == 0:
            continue
        if array.ndim >= 3 and array.shape[0] == n_chains:
            block = np.asarray(array[chain_index, -1], dtype=float)
        elif array.ndim >= 2 and n_chains == 1 and array.shape[0] > 1:
            block = np.asarray(array[-1], dtype=float)
        else:
            block = np.asarray(array, dtype=float)
        while block.ndim > 2:
            block = np.asarray(block[-1], dtype=float)
        if block.ndim == 2 and block.shape[0] == block.shape[1]:
            diag = np.diag(block)
            total_dim += int(block.shape[0])
            block_count += 1
            finite_block = np.isfinite(block).all()
            if finite_block and block.shape[0] <= 512:
                try:
                    eig = np.linalg.eigvalsh(block)
                    finite_eig = eig[np.isfinite(eig)]
                    if finite_eig.size:
                        eig_min_values.append(float(np.min(finite_eig)))
                        eig_max_values.append(float(np.max(finite_eig)))
                except np.linalg.LinAlgError:
                    pass
        else:
            diag = block.reshape(-1)
            total_dim += int(diag.size)
            block_count += 1
        finite_diag = np.asarray(diag, dtype=float).reshape(-1)
        finite_diag = finite_diag[np.isfinite(finite_diag)]
        if finite_diag.size:
            diag_values.append(finite_diag)
    all_diag = np.concatenate(diag_values) if diag_values else np.asarray([], dtype=float)
    diag_stats = _finite_array_stats(all_diag)
    return {
        "inverse_mass_block_count": int(block_count),
        "inverse_mass_total_dim": int(total_dim),
        "inverse_mass_diag_min": diag_stats["min"],
        "inverse_mass_diag_median": diag_stats["median"],
        "inverse_mass_diag_max": diag_stats["max"],
        "inverse_mass_eig_min": float(np.min(eig_min_values)) if eig_min_values else float("nan"),
        "inverse_mass_eig_max": float(np.max(eig_max_values)) if eig_max_values else float("nan"),
    }


def _debug_chain_label(init_diagnostics: dict[str, Any] | None, chain_index: int) -> str:
    labels = list((init_diagnostics or {}).get("chain_seed_labels", []))
    if 0 <= chain_index < len(labels):
        return str(labels[chain_index])
    return f"chain_{chain_index + 1}"


def _write_nuts_integrator_debug_table(
    path: Path,
    args: argparse.Namespace,
    posterior: PosteriorResults,
    extra: dict[str, Any],
    retained_indices: list[int],
) -> pd.DataFrame:
    grouped = np.asarray(posterior.grouped_samples, dtype=float) if posterior.grouped_samples is not None else np.asarray([])
    n_chains = int(grouped.shape[0]) if grouped.ndim == 3 else int(posterior.num_chains)
    n_draws = int(grouped.shape[1]) if grouped.ndim == 3 else int(posterior.sample_steps)
    accept = _chain_draw_array(extra.get("accept_prob", posterior.accept_prob), n_chains, n_draws)
    steps = _chain_draw_array(extra.get("num_steps", posterior.num_steps), n_chains, n_draws)
    energy = _chain_draw_array(extra.get("energy"), n_chains, n_draws)
    log_prob = _chain_draw_array(-np.asarray(extra.get("potential_energy"), dtype=float), n_chains, n_draws)
    step_size = _chain_draw_array(extra.get("adapt_state.step_size"), n_chains, n_draws)
    max_steps = float(2 ** _max_tree_depth_for_args(args) - 1)
    rows: list[dict[str, Any]] = []
    init_diag = posterior.init_diagnostics or {}
    for chain_index in range(n_chains):
        original_index = int(retained_indices[chain_index]) if chain_index < len(retained_indices) else chain_index
        row: dict[str, Any] = {
            "chain": chain_index + 1,
            "chain_index": original_index,
            "chain_label": _debug_chain_label(init_diag, original_index),
            "n_draws": n_draws,
        }
        if accept is not None:
            stats = _finite_array_stats(accept[chain_index])
            row.update({f"accept_prob_{key}": value for key, value in stats.items()})
        if steps is not None:
            chain_steps = np.asarray(steps[chain_index], dtype=float)
            stats = _finite_array_stats(chain_steps)
            row.update({f"num_steps_{key}": value for key, value in stats.items()})
            finite_steps = chain_steps[np.isfinite(chain_steps)]
            row["max_tree_depth_saturation_fraction"] = (
                float(np.mean(finite_steps >= max_steps)) if finite_steps.size else float("nan")
            )
        if log_prob is not None:
            stats = _finite_array_stats(log_prob[chain_index])
            row.update({f"log_prob_{key}": value for key, value in stats.items()})
        if energy is not None:
            chain_energy = np.asarray(energy[chain_index], dtype=float).reshape(-1)
            stats = _finite_array_stats(chain_energy)
            row.update({f"energy_{key}": value for key, value in stats.items()})
            finite_energy = chain_energy[np.isfinite(chain_energy)]
            if finite_energy.size > 1:
                row["energy_variance"] = float(np.var(finite_energy, ddof=1))
                row["energy_diff_mean_square"] = float(np.mean(np.square(np.diff(finite_energy))))
                row["energy_bfmi_like"] = (
                    float(row["energy_diff_mean_square"] / row["energy_variance"])
                    if row["energy_variance"] > 0.0
                    else float("nan")
                )
        if step_size is not None:
            chain_step_size = np.asarray(step_size[chain_index], dtype=float).reshape(-1)
            stats = _finite_array_stats(chain_step_size)
            row.update({f"step_size_{key}": value for key, value in stats.items()})
            finite_step_size = chain_step_size[np.isfinite(chain_step_size)]
            row["step_size_first"] = float(finite_step_size[0]) if finite_step_size.size else float("nan")
            row["step_size_last"] = float(finite_step_size[-1]) if finite_step_size.size else float("nan")
        if grouped.ndim == 3 and chain_index < grouped.shape[0]:
            ranges = np.ptp(np.asarray(grouped[chain_index], dtype=float), axis=0)
            finite_ranges = ranges[np.isfinite(ranges)]
            row["parameter_range_min"] = float(np.min(finite_ranges)) if finite_ranges.size else float("nan")
            row["parameter_range_median"] = float(np.median(finite_ranges)) if finite_ranges.size else float("nan")
            row["stuck_parameter_count_range_lt_1e_9"] = int(np.sum(finite_ranges < 1.0e-9))
            row["stuck_parameter_count_range_lt_1e_6"] = int(np.sum(finite_ranges < 1.0e-6))
        row.update(_mass_matrix_chain_summary(extra.get("adapt_state.inverse_mass_matrix"), chain_index, n_chains))
        rows.append(row)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _debug_prior_likelihood_gradient(
    state: BuildState,
    evaluator: Any,
    theta: np.ndarray,
    *,
    posterior_value_grad_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]] | None = None,
    prior_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    likelihood_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> dict[str, Any]:
    theta_jax = jnp.asarray(theta, dtype=jnp.float64)
    if posterior_value_grad_fn is None:
        posterior_value_grad_fn = jax.jit(
            jax.value_and_grad(lambda current: _prior_log_prob(state.parameter_specs, current) + evaluator._source_loglike_fn(current))
        )
    if prior_fn is None:
        prior_fn = jax.jit(lambda current: _prior_log_prob(state.parameter_specs, current))
    if likelihood_fn is None:
        likelihood_fn = jax.jit(lambda current: evaluator._source_loglike_fn(current))
    posterior_value, grad = posterior_value_grad_fn(theta_jax)
    prior_value = prior_fn(theta_jax)
    likelihood_value = likelihood_fn(theta_jax)
    grad_np = np.asarray(grad, dtype=float)
    finite_grad = grad_np[np.isfinite(grad_np)]
    return {
        "prior_log_prob": float(prior_value),
        "likelihood_log_prob": float(likelihood_value),
        "posterior_log_prob": float(posterior_value),
        "gradient": grad_np,
        "gradient_norm": float(np.linalg.norm(finite_grad)) if finite_grad.size else float("nan"),
        "gradient_max_abs": float(np.max(np.abs(finite_grad))) if finite_grad.size else float("nan"),
        "gradient_finite": bool(np.isfinite(grad_np).all()),
    }


def _top_gradient_indices(gradient: np.ndarray, max_count: int = 8) -> list[int]:
    grad = np.asarray(gradient, dtype=float)
    if grad.size == 0:
        return []
    finite_abs = np.where(np.isfinite(grad), np.abs(grad), -np.inf)
    valid = np.where(np.isfinite(finite_abs) & (finite_abs >= 0.0))[0]
    if valid.size == 0:
        return []
    order = valid[np.argsort(finite_abs[valid])[::-1]]
    return [int(index) for index in order[:max_count]]


def _image_sigma_int_from_theta(theta: np.ndarray, parameter_specs: list[ParameterSpec]) -> float:
    for idx, spec in enumerate(parameter_specs):
        if spec.sample_name == "image_sigma_int":
            physical = _convert_theta_to_physical(np.asarray(theta, dtype=float), parameter_specs)
            return float(physical[idx])
    return float("nan")


def _sampler_debug_state_catalog(
    posterior: PosteriorResults,
    nuts_init: NUTSInitialization,
    parameter_specs: list[ParameterSpec],
) -> list[dict[str, Any]]:
    grouped = np.asarray(posterior.grouped_samples, dtype=float) if posterior.grouped_samples is not None else np.asarray([])
    grouped_log_prob = (
        np.asarray(posterior.grouped_log_prob, dtype=float)
        if posterior.grouped_log_prob is not None
        else np.asarray([])
    )
    retained_indices = list((posterior.init_diagnostics or {}).get("retained_chain_indices", range(posterior.num_chains)))
    states: list[dict[str, Any]] = []
    for local_chain_index, original_chain_index in enumerate(retained_indices):
        if int(original_chain_index) < len(nuts_init.chain_seeds):
            states.append(
                {
                    "state_label": "chain_start",
                    "chain": local_chain_index + 1,
                    "chain_index": int(original_chain_index),
                    "draw": -1,
                    "theta": np.asarray(nuts_init.chain_seeds[int(original_chain_index)].values, dtype=float),
                }
            )
    if grouped.ndim == 3:
        for chain_index in range(grouped.shape[0]):
            original_index = int(retained_indices[chain_index]) if chain_index < len(retained_indices) else chain_index
            states.append(
                {
                    "state_label": "first_draw",
                    "chain": chain_index + 1,
                    "chain_index": original_index,
                    "draw": 0,
                    "theta": np.asarray(grouped[chain_index, 0], dtype=float),
                }
            )
            states.append(
                {
                    "state_label": "chain_median",
                    "chain": chain_index + 1,
                    "chain_index": original_index,
                    "draw": -1,
                    "theta": np.nanmedian(np.asarray(grouped[chain_index], dtype=float), axis=0),
                }
            )
            if grouped_log_prob.ndim == 2 and chain_index < grouped_log_prob.shape[0]:
                finite_lp = np.asarray(grouped_log_prob[chain_index], dtype=float)
                if np.isfinite(finite_lp).any():
                    best_draw = int(np.nanargmax(finite_lp))
                    states.append(
                        {
                            "state_label": "chain_best",
                            "chain": chain_index + 1,
                            "chain_index": original_index,
                            "draw": best_draw,
                            "theta": np.asarray(grouped[chain_index, best_draw], dtype=float),
                        }
                    )
        if grouped_log_prob.ndim == 2 and np.isfinite(grouped_log_prob).any():
            best_flat = int(np.nanargmax(grouped_log_prob))
            best_chain, best_draw = np.unravel_index(best_flat, grouped_log_prob.shape)
            original_index = int(retained_indices[best_chain]) if best_chain < len(retained_indices) else int(best_chain)
            states.append(
                {
                    "state_label": "global_best",
                    "chain": int(best_chain) + 1,
                    "chain_index": original_index,
                    "draw": int(best_draw),
                    "theta": np.asarray(grouped[best_chain, best_draw], dtype=float),
                }
            )
    return states


def _write_sampler_state_debug_table(
    path: Path,
    state: BuildState,
    evaluator: Any,
    debug_states: list[dict[str, Any]],
    posterior_value_grad_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    prior_fn: Callable[[jnp.ndarray], jnp.ndarray],
    likelihood_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    rows: list[dict[str, Any]] = []
    top_indices_by_state: dict[int, list[int]] = {}
    parameter_names = [spec.name for spec in state.parameter_specs]
    for state_index, item in enumerate(debug_states):
        theta = np.asarray(item["theta"], dtype=float)
        metrics = _debug_prior_likelihood_gradient(
            state,
            evaluator,
            theta,
            posterior_value_grad_fn=posterior_value_grad_fn,
            prior_fn=prior_fn,
            likelihood_fn=likelihood_fn,
        )
        top_indices = _top_gradient_indices(np.asarray(metrics["gradient"], dtype=float), max_count=8)
        top_indices_by_state[state_index] = top_indices
        image_sigma_int = (
            float(evaluator._image_sigma_int_numpy(theta))
            if hasattr(evaluator, "_image_sigma_int_numpy")
            else _image_sigma_int_from_theta(theta, state.parameter_specs)
        )
        fixed_image_sigma_int = getattr(evaluator, "fixed_image_sigma_int_arcsec", None)
        row = {
            "state_index": state_index,
            "state_label": item["state_label"],
            "chain": item["chain"],
            "chain_index": item["chain_index"],
            "draw": item["draw"],
            "prior_log_prob": metrics["prior_log_prob"],
            "likelihood_log_prob": metrics["likelihood_log_prob"],
            "posterior_log_prob": metrics["posterior_log_prob"],
            "gradient_norm": metrics["gradient_norm"],
            "gradient_max_abs": metrics["gradient_max_abs"],
            "gradient_finite": metrics["gradient_finite"],
            "theta_finite": bool(np.isfinite(theta).all()),
            "is_bad_loglike": bool(float(metrics["likelihood_log_prob"]) <= 0.5 * BAD_LOG_LIKE),
            "image_sigma_int_arcsec": image_sigma_int,
            "image_sigma_int_sampled": bool(getattr(evaluator, "image_sigma_int_sampled", False)),
            "fixed_image_sigma_int_arcsec": (
                float(fixed_image_sigma_int) if fixed_image_sigma_int is not None else float("nan")
            ),
            "effective_image_sigma_int_arcsec": image_sigma_int,
            "top_gradient_parameters": json.dumps([parameter_names[idx] for idx in top_indices]),
            "top_gradient_values": json.dumps([float(metrics["gradient"][idx]) for idx in top_indices]),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df, top_indices_by_state


def _debug_direction_scan_indices(
    parameter_specs: list[ParameterSpec],
    top_indices: list[int],
) -> list[int]:
    preferred_sample_names = {
        "image_sigma_int",
        "source_14_beta_y",
        "2_v_disp",
        "potfile_sigma",
        "potfile_cutkpc",
    }
    indices = list(top_indices[:6])
    for idx, spec in enumerate(parameter_specs):
        if spec.sample_name in preferred_sample_names or (
            spec.component_family in {"large", "scaling"} and spec.field in {"v_disp", "ellipticite", "core_radius_kpc", "sigma", "cutkpc"}
        ):
            indices.append(idx)
    result: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        if idx not in seen and 0 <= idx < len(parameter_specs):
            seen.add(idx)
            result.append(int(idx))
        if len(result) >= 12:
            break
    return result


def _write_sampler_direction_scan_debug_table(
    path: Path,
    state: BuildState,
    debug_states: list[dict[str, Any]],
    top_indices_by_state: dict[int, list[int]],
    prior_fn: Callable[[jnp.ndarray], jnp.ndarray],
    likelihood_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> pd.DataFrame:
    deltas = (-1.0e-2, -1.0e-3, 0.0, 1.0e-3, 1.0e-2)
    rows: list[dict[str, Any]] = []
    for state_index, item in enumerate(debug_states):
        theta = np.asarray(item["theta"], dtype=float)
        for param_index in _debug_direction_scan_indices(state.parameter_specs, top_indices_by_state.get(state_index, [])):
            spec = state.parameter_specs[param_index]
            for delta in deltas:
                probe = theta.copy()
                probe[param_index] += float(delta)
                probe_jax = jnp.asarray(probe, dtype=jnp.float64)
                prior_value = float(prior_fn(probe_jax))
                likelihood_value = float(likelihood_fn(probe_jax))
                rows.append(
                    {
                        "state_index": state_index,
                        "state_label": item["state_label"],
                        "chain": item["chain"],
                        "draw": item["draw"],
                        "parameter_index": param_index,
                        "parameter": spec.name,
                        "sample_name": spec.sample_name,
                        "delta": float(delta),
                        "prior_log_prob": prior_value,
                        "likelihood_log_prob": likelihood_value,
                        "posterior_log_prob": prior_value + likelihood_value,
                    }
                )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _write_sampler_debug_diagnostics(
    args: argparse.Namespace,
    state: BuildState,
    evaluator: Any,
    nuts_init: NUTSInitialization,
    posterior: PosteriorResults,
    extra: dict[str, Any],
) -> dict[str, Any]:
    if not bool(getattr(args, "debug_sampler_diagnostics", False)):
        return {}
    if str(posterior.sampler) != "numpyro_nuts":
        _log(args, f"[sampler-debug] skipped sampler={posterior.sampler}")
        return {}
    tables_dir = Path(args.output_dir) / state.run_name / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    retained_indices = list((posterior.init_diagnostics or {}).get("retained_chain_indices", range(posterior.num_chains)))
    diagnostics: dict[str, Any] = {}
    integrator_df = _write_nuts_integrator_debug_table(
        tables_dir / "nuts_integrator_diagnostics.csv",
        args,
        posterior,
        extra,
        [int(index) for index in retained_indices],
    )
    diagnostics["nuts_integrator_diagnostics_rows"] = int(len(integrator_df))

    posterior_value_grad_fn = jax.jit(
        jax.value_and_grad(
            lambda current: _prior_log_prob(state.parameter_specs, current) + evaluator._source_loglike_fn(current)
        )
    )
    prior_fn = jax.jit(lambda current: _prior_log_prob(state.parameter_specs, current))
    likelihood_fn = jax.jit(lambda current: evaluator._source_loglike_fn(current))
    debug_states = _sampler_debug_state_catalog(posterior, nuts_init, state.parameter_specs)
    state_df, top_indices_by_state = _write_sampler_state_debug_table(
        tables_dir / "sampler_state_diagnostics.csv",
        state,
        evaluator,
        debug_states,
        posterior_value_grad_fn,
        prior_fn,
        likelihood_fn,
    )
    diagnostics["sampler_state_diagnostics_rows"] = int(len(state_df))
    scan_df = _write_sampler_direction_scan_debug_table(
        tables_dir / "sampler_direction_scan.csv",
        state,
        debug_states,
        top_indices_by_state,
        prior_fn,
        likelihood_fn,
    )
    diagnostics["sampler_direction_scan_rows"] = int(len(scan_df))

    if str(getattr(evaluator, "sample_likelihood_mode", "")) == SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE:
        image_rows: list[dict[str, Any]] = []
        bin_rows: list[dict[str, Any]] = []
        for state_index, item in enumerate(debug_states):
            state_image_rows, state_bin_rows = _critical_arc_debug_terms_for_state(
                state,
                evaluator,
                np.asarray(item["theta"], dtype=float),
                state_index=state_index,
                state_label=str(item["state_label"]),
                chain=int(item["chain"]),
                draw=int(item["draw"]),
            )
            image_rows.extend(state_image_rows)
            bin_rows.extend(state_bin_rows)
        image_df = pd.DataFrame(image_rows)
        bin_df = pd.DataFrame(bin_rows)
        image_df.to_csv(tables_dir / "critical_arc_image_terms.csv", index=False)
        bin_df.to_csv(tables_dir / "critical_arc_bin_terms.csv", index=False)
        diagnostics["critical_arc_image_terms_rows"] = int(len(image_df))
        diagnostics["critical_arc_bin_terms_rows"] = int(len(bin_df))

    _log(
        args,
        (
            "[sampler-debug] wrote diagnostics "
            f"tables_dir={tables_dir} "
            f"states={diagnostics.get('sampler_state_diagnostics_rows', 0)} "
            f"direction_rows={diagnostics.get('sampler_direction_scan_rows', 0)}"
        ),
    )
    return diagnostics


def _finite_json_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _format_svi_health_value(value: Any) -> str:
    numeric = _finite_json_float(value)
    if numeric is None:
        return "na"
    return f"{numeric:.4g}"


def _svi_health_prior_scale(spec: ParameterSpec) -> float | None:
    if spec.prior_kind in {"normal", "truncated_normal"} and spec.std is not None:
        scale = abs(float(spec.std))
        return scale if np.isfinite(scale) and scale > 0.0 else None
    lower = float(spec.lower)
    upper = float(spec.upper)
    if np.isfinite(lower) and np.isfinite(upper):
        scale = abs(upper - lower)
        return scale if scale > 0.0 else None
    return None


def _svi_health_logprob_stats(values: np.ndarray, prefix: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    stats: dict[str, Any] = {
        f"{prefix}_count": int(array.size),
        f"{prefix}_finite_count": int(finite.size),
        f"{prefix}_finite_fraction": float(finite.size / array.size) if array.size else None,
        f"{prefix}_min": None,
        f"{prefix}_q05": None,
        f"{prefix}_median": None,
        f"{prefix}_q95": None,
        f"{prefix}_max": None,
        f"{prefix}_q95_q05_width": None,
    }
    if finite.size:
        q05, median, q95 = np.quantile(finite, [0.05, 0.5, 0.95])
        stats.update(
            {
                f"{prefix}_min": _finite_json_float(np.min(finite)),
                f"{prefix}_q05": _finite_json_float(q05),
                f"{prefix}_median": _finite_json_float(median),
                f"{prefix}_q95": _finite_json_float(q95),
                f"{prefix}_max": _finite_json_float(np.max(finite)),
                f"{prefix}_q95_q05_width": _finite_json_float(q95 - q05),
            }
        )
    return stats


def _svi_health_diagnostics(
    parameter_specs: list[ParameterSpec],
    guide_samples: np.ndarray,
    guide_log_prob: np.ndarray,
    center_theta: np.ndarray,
    center_log_prob: float,
    chain_seeds: list[ChainSeed],
    chain_start_log_prob: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    samples = np.asarray(guide_samples, dtype=float)
    log_prob = np.asarray(guide_log_prob, dtype=float).reshape(-1)
    if samples.ndim != 2:
        samples = np.empty((0, len(parameter_specs)), dtype=float)
    draw_count = int(min(samples.shape[0], log_prob.size))
    sample_finite = np.all(np.isfinite(samples[:draw_count]), axis=1) if draw_count else np.empty((0,), dtype=bool)
    logprob_finite = np.isfinite(log_prob[:draw_count]) if draw_count else np.empty((0,), dtype=bool)
    finite_draw_mask = sample_finite & logprob_finite
    finite_draw_fraction = float(np.mean(finite_draw_mask)) if draw_count else None

    guide_stats = _svi_health_logprob_stats(log_prob[:draw_count], "guide_log_prob")
    chain_stats = _svi_health_logprob_stats(chain_start_log_prob, "chain_start_log_prob")
    finite_guide_log_prob = log_prob[:draw_count][logprob_finite]
    center_array = np.asarray(center_theta, dtype=float).reshape(-1)
    center_finite = bool(center_array.size and np.all(np.isfinite(center_array)))
    center_lp = _finite_json_float(center_log_prob)
    guide_median = guide_stats.get("guide_log_prob_median")
    guide_q05 = guide_stats.get("guide_log_prob_q05")
    center_rank = None
    center_percentile = None
    if center_lp is not None and finite_guide_log_prob.size:
        center_rank = int(np.sum(finite_guide_log_prob <= center_lp))
        center_percentile = float(100.0 * center_rank / finite_guide_log_prob.size)

    chain_start_count = int(len(chain_seeds))
    distinct_chain_starts = int(len({tuple(np.round(np.asarray(seed.values, dtype=float), 8)) for seed in chain_seeds}))
    chain_finite_fraction = chain_stats.get("chain_start_log_prob_finite_fraction")
    chain_min = chain_stats.get("chain_start_log_prob_min")

    spread_items: list[dict[str, Any]] = []
    n_columns = min(samples.shape[1] if samples.ndim == 2 else 0, len(parameter_specs))
    for idx, spec in enumerate(parameter_specs[:n_columns]):
        prior_scale = _svi_health_prior_scale(spec)
        if prior_scale is None:
            continue
        values = samples[:draw_count, idx] if draw_count else np.empty((0,), dtype=float)
        finite_values = values[np.isfinite(values)]
        if not finite_values.size:
            continue
        std = float(np.std(finite_values))
        ratio = float(std / prior_scale)
        spread_items.append(
            {
                "parameter": spec.name,
                "sample_name": spec.sample_name,
                "std": _finite_json_float(std),
                "prior_scale": _finite_json_float(prior_scale),
                "std_over_prior_scale": _finite_json_float(ratio),
            }
        )
    spread_items.sort(
        key=lambda item: (
            float("inf") if item.get("std_over_prior_scale") is None else float(item["std_over_prior_scale"]),
            str(item.get("parameter", "")),
        )
    )
    worst_spread_items = spread_items[:SVI_HEALTH_WORST_PARAMETER_COUNT]

    metrics: dict[str, Any] = {
        "guide_draws": draw_count,
        "guide_sample_count": int(samples.shape[0]),
        "guide_log_prob_count": int(log_prob.size),
        "guide_finite_draw_count": int(np.sum(finite_draw_mask)),
        "guide_finite_draw_fraction": finite_draw_fraction,
        "guide_finite_draw_fraction_warning_threshold": SVI_HEALTH_FINITE_DRAW_FRACTION_WARNING,
        **guide_stats,
        "guide_log_prob_spread_warning_threshold": SVI_HEALTH_LOGPROB_SPREAD_WARNING,
        "center_log_prob": center_lp,
        "center_finite": center_finite,
        "center_log_prob_rank": center_rank,
        "center_log_prob_rank_total": int(finite_guide_log_prob.size),
        "center_log_prob_percentile": center_percentile,
        "center_log_prob_delta_from_guide_median": (
            _finite_json_float(center_lp - float(guide_median))
            if center_lp is not None and guide_median is not None
            else None
        ),
        "center_log_prob_drop_warning_threshold": SVI_HEALTH_CENTER_DROP_WARNING,
        "chain_start_count": chain_start_count,
        "chain_start_distinct_count": distinct_chain_starts,
        **chain_stats,
        "chain_start_log_prob_drop_warning_threshold": SVI_HEALTH_CHAIN_START_DROP_WARNING,
        "chain_start_log_prob_delta_min_from_guide_q05": (
            _finite_json_float(float(chain_min) - float(guide_q05))
            if chain_min is not None and guide_q05 is not None
            else None
        ),
        "guide_low_spread_ratio_warning_threshold": SVI_HEALTH_LOW_SPREAD_RATIO_WARNING,
        "guide_worst_std_over_prior_scale": worst_spread_items,
    }

    warnings: list[str] = []
    if samples.shape[0] != log_prob.size:
        warnings.append(
            "SVI guide sample/log-prob count mismatch "
            f"samples={samples.shape[0]} log_prob={log_prob.size}"
        )
    if finite_draw_fraction is None:
        warnings.append("no SVI guide draws available for health diagnostics")
    elif finite_draw_fraction < SVI_HEALTH_FINITE_DRAW_FRACTION_WARNING:
        warnings.append(
            "non-finite SVI guide draws "
            f"finite_fraction={finite_draw_fraction:.3f}"
        )
    guide_spread = metrics.get("guide_log_prob_q95_q05_width")
    if guide_spread is not None and float(guide_spread) >= SVI_HEALTH_LOGPROB_SPREAD_WARNING:
        warnings.append(
            "wide SVI guide log-prob spread "
            f"q95-q05={float(guide_spread):.4g} >= {SVI_HEALTH_LOGPROB_SPREAD_WARNING:.4g}"
        )
    if not center_finite:
        warnings.append("non-finite SVI center parameters")
    if center_lp is None:
        warnings.append("non-finite SVI center log-prob")
    elif guide_median is not None:
        center_drop = float(guide_median) - center_lp
        if center_drop >= SVI_HEALTH_CENTER_DROP_WARNING:
            warnings.append(
                "SVI center below guide median "
                f"drop={center_drop:.4g} >= {SVI_HEALTH_CENTER_DROP_WARNING:.4g}"
            )
    if chain_start_count and chain_finite_fraction is not None and float(chain_finite_fraction) < 1.0:
        warnings.append(
            "non-finite SVI chain-start log probabilities "
            f"finite_fraction={float(chain_finite_fraction):.3f}"
        )
    if chain_min is not None and guide_q05 is not None:
        chain_drop = float(guide_q05) - float(chain_min)
        if chain_drop >= SVI_HEALTH_CHAIN_START_DROP_WARNING:
            warnings.append(
                "poor SVI chain-start log-prob "
                f"drop_below_guide_q05={chain_drop:.4g} >= {SVI_HEALTH_CHAIN_START_DROP_WARNING:.4g}"
            )
    low_spread_items = [
        item
        for item in worst_spread_items
        if item.get("std_over_prior_scale") is not None
        and float(item["std_over_prior_scale"]) <= SVI_HEALTH_LOW_SPREAD_RATIO_WARNING
    ]
    if low_spread_items:
        worst = low_spread_items[0]
        warnings.append(
            "near-zero SVI guide spread "
            f"std/prior_scale={float(worst['std_over_prior_scale']):.4g} "
            f"<= {SVI_HEALTH_LOW_SPREAD_RATIO_WARNING:.4g} ({worst['parameter']})"
        )
    return metrics, warnings


def _record_svi_health_diagnostics(
    posterior: PosteriorResults,
    svi_diagnostics: dict[str, Any],
    metrics: dict[str, Any],
    warnings: list[str],
) -> None:
    svi_diagnostics["svi_health_metrics"] = dict(metrics)
    svi_diagnostics["svi_health_warnings"] = list(warnings)
    if posterior.init_diagnostics is None:
        posterior.init_diagnostics = {}
    posterior.init_diagnostics["svi_health_metrics"] = dict(metrics)
    posterior.init_diagnostics["svi_health_warnings"] = list(warnings)


def _log_svi_health(args: argparse.Namespace, metrics: dict[str, Any], warnings: list[str]) -> None:
    worst_items = list(metrics.get("guide_worst_std_over_prior_scale", []))[:3]
    worst_text = ",".join(
        f"{item.get('sample_name') or item.get('parameter')}:{_format_svi_health_value(item.get('std_over_prior_scale'))}"
        for item in worst_items
    )
    if not worst_text:
        worst_text = "none"
    center_rank = metrics.get("center_log_prob_rank")
    center_total = metrics.get("center_log_prob_rank_total")
    center_rank_text = "na" if center_rank is None or center_total is None else f"{int(center_rank)}/{int(center_total)}"
    warning_text = "; ".join(warnings) if warnings else "none"
    _log(
        args,
        (
            "[svi:health] "
            f"guide_finite={_format_svi_health_value(metrics.get('guide_finite_draw_fraction'))} "
            "guide_logprob_min/median/max="
            f"{_format_svi_health_value(metrics.get('guide_log_prob_min'))}/"
            f"{_format_svi_health_value(metrics.get('guide_log_prob_median'))}/"
            f"{_format_svi_health_value(metrics.get('guide_log_prob_max'))} "
            "guide_logprob_q05/q95="
            f"{_format_svi_health_value(metrics.get('guide_log_prob_q05'))}/"
            f"{_format_svi_health_value(metrics.get('guide_log_prob_q95'))} "
            f"spread={_format_svi_health_value(metrics.get('guide_log_prob_q95_q05_width'))} "
            f"center_logprob={_format_svi_health_value(metrics.get('center_log_prob'))} "
            f"center_pct={_format_svi_health_value(metrics.get('center_log_prob_percentile'))} "
            f"center_rank={center_rank_text} "
            f"distinct_chain_starts={int(metrics.get('chain_start_distinct_count') or 0)} "
            "chain_logprob_min/median/max="
            f"{_format_svi_health_value(metrics.get('chain_start_log_prob_min'))}/"
            f"{_format_svi_health_value(metrics.get('chain_start_log_prob_median'))}/"
            f"{_format_svi_health_value(metrics.get('chain_start_log_prob_max'))} "
            f"worst_spread={worst_text} "
            f"warnings={warning_text}"
        ),
    )


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
    elif spec.prior_kind == "truncated_normal" and np.isfinite(lower):
        margin = max(float(boundary_frac) * max(abs(lower), 1.0e-12), 0.0)
        if np.isfinite(upper):
            width = float(upper - lower)
            margin = min(margin, 0.5 * width - 1.0e-12) if width > 0.0 else 0.0
        if margin > 0.0:
            clipped = max(clipped, lower + margin)
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
        if _is_potfile_mass_size_spec(spec):
            continue
        clipped[idx] = _clip_value_to_safe_bounds(clipped[idx], spec, boundary_frac=boundary_frac)
    for site in _parameter_sample_sites(parameter_specs):
        if len(site.indices) != 2:
            continue
        mass_idx, size_idx = site.indices
        mass_spec = parameter_specs[mass_idx]
        size_spec = parameter_specs[size_idx]
        if not (_is_potfile_mass_size_spec(mass_spec) and _is_potfile_mass_size_spec(size_spec)):
            continue
        log_cut_gap = (
            float(mass_spec.coupled_size_center)
            + float(mass_spec.coupled_size_scale) * float(clipped[size_idx])
        )
        log_cut_gap = _clip_value_to_safe_bounds(log_cut_gap, size_spec, boundary_frac=boundary_frac)
        clipped[size_idx] = (
            log_cut_gap - float(mass_spec.coupled_size_center)
        ) / float(mass_spec.coupled_size_scale)
        mass_raw = (
            float(mass_spec.coupled_mass_center)
            + float(mass_spec.coupled_mass_scale) * float(clipped[mass_idx])
        )
        log_sigma = 0.5 * (mass_raw - log_cut_gap)
        log_sigma = _clip_value_to_safe_bounds(log_sigma, mass_spec, boundary_frac=boundary_frac)
        mass_raw = 2.0 * log_sigma + log_cut_gap
        clipped[mass_idx] = (
            mass_raw - float(mass_spec.coupled_mass_center)
        ) / float(mass_spec.coupled_mass_scale)
    return clipped


def _default_theta(parameter_specs: list[ParameterSpec]) -> np.ndarray:
    default = np.asarray(
        [
            0.0
            if _is_potfile_mass_size_spec(spec)
            else
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
        if _is_potfile_mass_size_spec(spec):
            scale = max(float(jitter_frac), 1.0e-6)
        elif spec.prior_kind == "normal":
            scale = max(float(spec.std or 0.0) * 0.15, 1.0e-6)
        elif spec.prior_kind == "truncated_normal":
            scale = max(float(spec.std or 0.0) * 0.15, 1.0e-6)
        else:
            scale = max(float(spec.upper - spec.lower) * float(jitter_frac), 1.0e-6)
        theta_array[idx] += float(rng.normal(0.0, scale))
    return _clip_theta_to_support(theta_array, parameter_specs, boundary_frac=boundary_frac)


def _initial_latent_value_from_physical(physical_value: float, spec: ParameterSpec) -> float:
    latent_value = _physical_to_latent_numpy(float(physical_value), spec)
    if spec.prior_kind != "truncated_normal":
        return latent_value
    return _clip_value_to_safe_bounds(
        latent_value,
        spec,
        boundary_frac=DEFAULT_NUTS_INIT_BOUNDARY_FRAC,
    )


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
    sites = _parameter_sample_sites(parameter_specs)
    for seed in chain_seeds:
        params_dict = {
            site.name: _site_value_from_theta(seed.values, site)
            for site in sites
        }
        unconstrained = unconstrain_fn(
            model_for_init,
            model_args=(),
            model_kwargs={},
            params=params_dict,
        )
        unconstrained_payloads.append(
            {
                site.name: np.asarray(unconstrained[site.name], dtype=float)
                for site in sites
            }
        )
    payload: dict[str, jnp.ndarray] = {}
    for site in sites:
        if len(unconstrained_payloads) == 1:
            value = np.asarray(unconstrained_payloads[0][site.name], dtype=float)
        else:
            value = np.stack([item[site.name] for item in unconstrained_payloads], axis=0)
        payload[site.name] = jnp.asarray(value, dtype=jnp.float64)
    return payload


def _sample_site_model(parameter_specs: list[ParameterSpec]):
    def model():
        sampled: dict[str, Any] = {}
        spec_by_sample = {spec.sample_name: spec for spec in parameter_specs}
        for site in _parameter_sample_sites(parameter_specs):
            spec = parameter_specs[site.indices[0]]
            if spec.prior_kind == "hierarchical_normal":
                if len(site.indices) != 1:
                    raise ValueError(f"Hierarchical vector sample site {site.name!r} is unsupported.")
                if not spec.parent_sample_name:
                    raise ValueError(f"Hierarchical parameter {spec.name} is missing parent_sample_name.")
                parent_value = _latent_to_physical_jax(sampled[spec.parent_sample_name], spec_by_sample[spec.parent_sample_name])
                value = numpyro.sample(
                    site.name,
                    dist.Normal(float(spec.mean or 0.0), jnp.asarray(parent_value, dtype=jnp.float64)),
                )
            else:
                value = numpyro.sample(site.name, _distribution_for_sample_site(site, parameter_specs))
            if len(site.indices) == 1:
                sampled[spec.sample_name] = value
            else:
                for offset, idx in enumerate(site.indices):
                    sampled[parameter_specs[idx].sample_name] = value[..., offset]

    return model


def _jittered_2x2_covariance_det(
    c00: jnp.ndarray,
    c01: jnp.ndarray,
    c11: jnp.ndarray,
    *,
    relative_jitter: float = DEFAULT_COVARIANCE_DIAGONAL_JITTER_RELATIVE,
    absolute_jitter: float = DEFAULT_COVARIANCE_DIAGONAL_JITTER_ABSOLUTE,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    diag_scale = jnp.sqrt(jnp.maximum(c00 * c11, jnp.asarray(0.0, dtype=jnp.float64)))
    jitter = (
        jnp.asarray(float(relative_jitter), dtype=jnp.float64) * diag_scale
        + jnp.asarray(float(absolute_jitter), dtype=jnp.float64)
    )
    c00_j = c00 + jitter
    c11_j = c11 + jitter
    det = c00_j * c11_j - jnp.square(c01)
    det = jnp.maximum(det, jnp.asarray(MIN_COVARIANCE_DETERMINANT_GUARD, dtype=jnp.float64))
    return c00_j, c11_j, det


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
    max_gain: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN,
    max_residual_arcsec: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC,
    residual_loss: str = DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS,
    student_t_nu: float = DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU,
) -> jnp.ndarray:
    sigma2_img = jnp.square(sigma_per_image)
    source_var = jnp.square(source_sigma_int)
    cov_floor = jnp.asarray(covariance_floor, dtype=jnp.float64)
    c00 = sigma2_img * (jnp.square(jac_a00) + jnp.square(jac_a01)) + source_var + scatter_var_x + cov_floor
    c11 = sigma2_img * (jnp.square(jac_a10) + jnp.square(jac_a11)) + source_var + scatter_var_y + cov_floor
    c01 = sigma2_img * (jac_a00 * jac_a10 + jac_a01 * jac_a11)
    if float(max_gain) > 0.0:
        gain_floor = jnp.square(sigma_per_image / jnp.asarray(float(max_gain), dtype=jnp.float64))
        c00 = c00 + gain_floor
        c11 = c11 + gain_floor
    c00 = jnp.maximum(c00, cov_floor)
    c11 = jnp.maximum(c11, cov_floor)
    c00, c11, det = _jittered_2x2_covariance_det(c00, c01, c11)
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
    dx, dy = _smooth_residual_cap(dx, dy, max_residual_arcsec)
    quad = dx * (inv00 * dx + inv01 * dy) + dy * (inv01 * dx + inv11 * dy)
    logdet = jnp.log(det)
    if str(residual_loss) == LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_STUDENT_T:
        family_ll = _student_t_2d_loglike_from_quad_logdet(quad, logdet, student_t_nu)
    else:
        family_ll = -0.5 * (quad + jnp.log(jnp.square(2.0 * jnp.pi)) + logdet)

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


def _cab_tangent_angle_residual(
    predicted_angle: jnp.ndarray,
    observed_angle: jnp.ndarray,
) -> jnp.ndarray:
    delta = predicted_angle - observed_angle
    return 0.5 * jnp.arctan2(jnp.sin(2.0 * delta), jnp.cos(2.0 * delta))


class _CabTangentFrame(NamedTuple):
    tangent_angle_rad: jnp.ndarray
    tangent_x: jnp.ndarray
    tangent_y: jnp.ndarray
    lambda_low: jnp.ndarray
    lambda_high: jnp.ndarray
    branch_weight: jnp.ndarray
    frame_weight: jnp.ndarray
    finite: jnp.ndarray


class _CabMorphologyPrediction(NamedTuple):
    tangent_angle_rad: jnp.ndarray
    curvature_arcsec_inv: jnp.ndarray
    branch_weight: jnp.ndarray
    frame_weight: jnp.ndarray
    finite: jnp.ndarray


class _CabMorphologyTerms(NamedTuple):
    sigma_tangent: jnp.ndarray
    sigma_curvature: jnp.ndarray
    branch_weight: jnp.ndarray
    frame_weight: jnp.ndarray
    effective_reliability: jnp.ndarray
    tangent_residual_by_branch: jnp.ndarray
    curvature_residual_by_branch: jnp.ndarray
    tangent_residual: jnp.ndarray
    curvature_residual: jnp.ndarray
    branch_inlier_ll: jnp.ndarray
    inlier_ll: jnp.ndarray
    outlier_ll: jnp.ndarray
    row_loglike: jnp.ndarray
    finite_row: jnp.ndarray


def _cab_tangent_frame_from_jacobian_entries(
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
) -> _CabTangentFrame:
    offdiag = 0.5 * (jac_a01 + jac_a10)
    trace = jac_a00 + jac_a11
    diff = jac_a00 - jac_a11
    twice_offdiag = 2.0 * offdiag
    gap2 = jnp.square(diff) + jnp.square(twice_offdiag)
    raw_scale = jnp.sqrt(gap2 + jnp.square(jnp.asarray(CAB_FRAME_VALUE_ABSOLUTE_SOFTENING, dtype=jnp.float64)))
    value_scale = jnp.maximum(jnp.maximum(jnp.abs(trace), raw_scale), 1.0)
    gap_floor = (
        jnp.asarray(CAB_FRAME_RELATIVE_SOFTENING, dtype=jnp.float64) * value_scale
        + jnp.asarray(CAB_FRAME_VALUE_ABSOLUTE_SOFTENING, dtype=jnp.float64)
    )
    gap = jnp.sqrt(gap2 + jnp.square(gap_floor))
    lambda_low = 0.5 * (trace - gap)
    lambda_high = 0.5 * (trace + gap)

    high_x_a = gap + diff
    high_y_a = twice_offdiag
    high_x_b = twice_offdiag
    high_y_b = gap - diff
    use_a = diff >= 0.0
    high_x = jnp.where(use_a, high_x_a, high_x_b)
    high_y = jnp.where(use_a, high_y_a, high_y_b)
    high_norm = jnp.sqrt(jnp.square(high_x) + jnp.square(high_y))
    high_x = high_x / high_norm
    high_y = high_y / high_norm
    low_x = -high_y
    low_y = high_x

    tangent_x = jnp.stack([low_x, high_x], axis=-1)
    tangent_y = jnp.stack([low_y, high_y], axis=-1)
    tangent_angle = jnp.arctan2(tangent_y, tangent_x)

    abs_low = jnp.sqrt(jnp.square(lambda_low) + jnp.square(gap_floor))
    abs_high = jnp.sqrt(jnp.square(lambda_high) + jnp.square(gap_floor))
    signed_asym = (abs_high - abs_low) / jnp.maximum(abs_high + abs_low, gap_floor)
    physical_frac = jnp.asarray(CAB_FRAME_PHYSICAL_AMBIGUITY_FRACTION, dtype=jnp.float64)
    low_branch_weight = jax.nn.sigmoid(signed_asym / physical_frac)
    branch_weight = jnp.stack([low_branch_weight, 1.0 - low_branch_weight], axis=-1)

    direction_confidence = gap2 / (gap2 + jnp.square(gap_floor))
    selection_confidence = jnp.square(signed_asym) / (jnp.square(signed_asym) + jnp.square(physical_frac))
    frame_weight = jnp.clip(direction_confidence * selection_confidence, 0.0, 1.0)
    finite = (
        jnp.isfinite(jac_a00)
        & jnp.isfinite(jac_a01)
        & jnp.isfinite(jac_a10)
        & jnp.isfinite(jac_a11)
        & jnp.all(jnp.isfinite(tangent_angle), axis=-1)
        & jnp.all(jnp.isfinite(tangent_x), axis=-1)
        & jnp.all(jnp.isfinite(tangent_y), axis=-1)
        & jnp.all(jnp.isfinite(branch_weight), axis=-1)
        & jnp.isfinite(lambda_low)
        & jnp.isfinite(lambda_high)
        & jnp.isfinite(frame_weight)
    )
    return _CabTangentFrame(
        tangent_angle_rad=tangent_angle,
        tangent_x=tangent_x,
        tangent_y=tangent_y,
        lambda_low=lambda_low,
        lambda_high=lambda_high,
        branch_weight=branch_weight,
        frame_weight=frame_weight,
        finite=finite,
    )


def _cab_with_branch_axis(values: jnp.ndarray, *, dtype: Any = jnp.float64) -> jnp.ndarray:
    arr = jnp.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        return arr[jnp.newaxis, jnp.newaxis]
    if arr.ndim == 1:
        return arr[..., jnp.newaxis]
    return arr


def _cab_morphology_terms(
    predicted_tangent_angle_rad: jnp.ndarray,
    predicted_curvature_arcsec_inv: jnp.ndarray,
    prediction_finite: jnp.ndarray,
    observed_tangent_angle_rad: jnp.ndarray,
    observed_curvature_arcsec_inv: jnp.ndarray,
    sigma_tangent_angle_rad: jnp.ndarray,
    sigma_curvature_arcsec_inv: jnp.ndarray,
    reliability: jnp.ndarray,
    active_arcs: jnp.ndarray,
    tangent_sigma_floor_rad: float = DEFAULT_CAB_TANGENT_SIGMA_FLOOR_RAD,
    curvature_sigma_floor_arcsec_inv: float = DEFAULT_CAB_CURVATURE_SIGMA_FLOOR_ARCSEC_INV,
    branch_weight: jnp.ndarray | None = None,
    frame_weight: jnp.ndarray | None = None,
) -> _CabMorphologyTerms:
    predicted_angle = _cab_with_branch_axis(predicted_tangent_angle_rad)
    predicted_curvature = _cab_with_branch_axis(predicted_curvature_arcsec_inv)
    branch_finite = _cab_with_branch_axis(prediction_finite, dtype=bool)
    n_branch = int(predicted_angle.shape[-1])
    if branch_weight is None:
        branch_weights = jnp.ones_like(predicted_angle) / float(n_branch)
    else:
        branch_weights = _cab_with_branch_axis(branch_weight)
    if frame_weight is None:
        branch_frame_weight = jnp.ones_like(predicted_angle)
    else:
        branch_frame_weight = _cab_with_branch_axis(frame_weight)
    branch_weights = jnp.broadcast_to(branch_weights, predicted_angle.shape)
    branch_frame_weight = jnp.broadcast_to(branch_frame_weight, predicted_angle.shape)
    branch_weights = jnp.clip(branch_weights, 0.0, 1.0)
    branch_weight_sum = jnp.sum(branch_weights, axis=-1, keepdims=True)
    uniform_branch_weight = jnp.ones_like(branch_weights) / float(n_branch)
    branch_weights = jnp.where(
        branch_weight_sum > 0.0,
        branch_weights / jnp.maximum(branch_weight_sum, 1.0e-300),
        uniform_branch_weight,
    )
    branch_frame_weight = jnp.clip(branch_frame_weight, 0.0, 1.0)
    sigma_tangent = jnp.maximum(
        jnp.asarray(sigma_tangent_angle_rad, dtype=jnp.float64),
        jnp.asarray(float(tangent_sigma_floor_rad), dtype=jnp.float64),
    )
    sigma_curvature = jnp.maximum(
        jnp.asarray(sigma_curvature_arcsec_inv, dtype=jnp.float64),
        jnp.asarray(float(curvature_sigma_floor_arcsec_inv), dtype=jnp.float64),
    )
    tangent_residual = _cab_tangent_angle_residual(
        predicted_angle,
        jnp.asarray(observed_tangent_angle_rad, dtype=jnp.float64)[..., jnp.newaxis],
    )
    curvature_residual = predicted_curvature - jnp.asarray(observed_curvature_arcsec_inv, dtype=jnp.float64)[
        ..., jnp.newaxis
    ]
    # Inlier density lives on the same observation support as the outlier,
    # [0, pi) x [0, inf): an axial von Mises on the doubled tangent angle and a
    # normal on curvature lower-truncated at 0. Both reduce to the previous
    # Gaussians for small sigma / predicted curvature far from 0.
    sigma_tangent_b = sigma_tangent[..., jnp.newaxis]
    sigma_curvature_b = sigma_curvature[..., jnp.newaxis]
    # Axial von Mises: g(theta) = exp(kappa cos 2d) / (pi I0(kappa)), kappa =
    # 1/(4 sigma^2) so it matches the Gaussian -0.5 (d/sigma)^2 - 0.5 log(2 pi
    # sigma^2) as sigma -> 0. log I0(kappa) = kappa + log i0e(kappa); fold the
    # kappa into (cos 2d - 1) so I0 is never formed and there is no large cancellation.
    kappa_tangent = 1.0 / (4.0 * jnp.square(sigma_tangent_b))
    tangent_ll = (
        kappa_tangent * (jnp.cos(2.0 * tangent_residual) - 1.0)
        - jnp.log(jnp.asarray(jnp.pi, dtype=jnp.float64))
        - jnp.log(jsp_special.i0e(kappa_tangent))
    )
    # Normal lower-truncated to observed curvature >= 0; the truncation mass is
    # Phi(predicted/sigma) (predicted curvature is a nonnegative magnitude), so the
    # normalizer -log Phi(predicted/sigma) is model-dependent.
    curvature_ll = (
        -0.5 * jnp.square(curvature_residual / sigma_curvature_b)
        - 0.5 * jnp.log(2.0 * jnp.pi * jnp.square(sigma_curvature_b))
        - jsp_special.log_ndtr(predicted_curvature / sigma_curvature_b)
    )
    branch_inlier_ll = tangent_ll + curvature_ll
    inlier_ll = jsp_special.logsumexp(jnp.log(jnp.clip(branch_weights, 1.0e-300, 1.0)) + branch_inlier_ll, axis=-1)
    row_frame_weight = jnp.sum(branch_weights * branch_frame_weight, axis=-1)
    reliability_value = jnp.clip(jnp.asarray(reliability, dtype=jnp.float64), 0.0, 1.0)
    effective_reliability = jnp.clip(reliability_value * row_frame_weight, 0.0, 1.0)
    # Outlier models a bad arc-morphology measurement: uniform over the axial angle
    # times a half-normal over the observed (non-negative) curvature. It scores the
    # observation, not the model residual, so it is parameter-independent by design.
    observed_curvature = jnp.asarray(observed_curvature_arcsec_inv, dtype=jnp.float64)
    sigma_curvature_outlier = jnp.maximum(
        jnp.asarray(CAB_OUTLIER_CURVATURE_SIGMA_ARCSEC_INV, dtype=jnp.float64),
        3.0 * sigma_curvature,
    )
    outlier_ll = (
        -jnp.log(jnp.asarray(jnp.pi, dtype=jnp.float64))
        + 0.5 * jnp.log(2.0 / (jnp.pi * jnp.square(sigma_curvature_outlier)))
        - 0.5 * jnp.square(observed_curvature / sigma_curvature_outlier)
    )
    row_loglike = jnp.logaddexp(
        jnp.log(jnp.clip(effective_reliability, 1.0e-300, 1.0)) + inlier_ll,
        jnp.log(jnp.clip(1.0 - effective_reliability, 1.0e-300, 1.0)) + outlier_ll,
    )
    tangent_residual_mean = jnp.sum(branch_weights * tangent_residual, axis=-1)
    curvature_residual_mean = jnp.sum(branch_weights * curvature_residual, axis=-1)
    active = jnp.asarray(active_arcs, dtype=bool)
    finite_inputs = (
        jnp.isfinite(observed_tangent_angle_rad)
        & jnp.isfinite(observed_curvature_arcsec_inv)
        & jnp.isfinite(sigma_tangent)
        & jnp.isfinite(sigma_curvature)
        & (sigma_tangent > 0.0)
        & (sigma_curvature > 0.0)
        & jnp.isfinite(reliability)
        & jnp.isfinite(effective_reliability)
        & jnp.isfinite(row_frame_weight)
    )
    finite_branch = (
        branch_finite
        & jnp.isfinite(predicted_angle)
        & jnp.isfinite(predicted_curvature)
        & jnp.isfinite(branch_weights)
        & jnp.isfinite(branch_frame_weight)
        & jnp.isfinite(branch_inlier_ll)
    )
    finite_row = (
        active
        & finite_inputs
        & jnp.all(finite_branch, axis=-1)
        & jnp.isfinite(inlier_ll)
        & jnp.isfinite(outlier_ll)
        & jnp.isfinite(row_loglike)
        & jnp.isfinite(tangent_residual_mean)
        & jnp.isfinite(curvature_residual_mean)
    )
    return _CabMorphologyTerms(
        sigma_tangent=sigma_tangent,
        sigma_curvature=sigma_curvature,
        branch_weight=branch_weights,
        frame_weight=branch_frame_weight,
        effective_reliability=effective_reliability,
        tangent_residual_by_branch=tangent_residual,
        curvature_residual_by_branch=curvature_residual,
        tangent_residual=tangent_residual_mean,
        curvature_residual=curvature_residual_mean,
        branch_inlier_ll=branch_inlier_ll,
        inlier_ll=inlier_ll,
        outlier_ll=outlier_ll,
        row_loglike=row_loglike,
        finite_row=finite_row,
    )


def _cab_morphology_arc_catalog_loglike(
    predicted_tangent_angle_rad: jnp.ndarray,
    predicted_curvature_arcsec_inv: jnp.ndarray,
    prediction_finite: jnp.ndarray,
    observed_tangent_angle_rad: jnp.ndarray,
    observed_curvature_arcsec_inv: jnp.ndarray,
    sigma_tangent_angle_rad: jnp.ndarray,
    sigma_curvature_arcsec_inv: jnp.ndarray,
    reliability: jnp.ndarray,
    active_arcs: jnp.ndarray,
    tangent_sigma_floor_rad: float = DEFAULT_CAB_TANGENT_SIGMA_FLOOR_RAD,
    curvature_sigma_floor_arcsec_inv: float = DEFAULT_CAB_CURVATURE_SIGMA_FLOOR_ARCSEC_INV,
    branch_weight: jnp.ndarray | None = None,
    frame_weight: jnp.ndarray | None = None,
) -> jnp.ndarray:
    terms = _cab_morphology_terms(
        predicted_tangent_angle_rad=predicted_tangent_angle_rad,
        predicted_curvature_arcsec_inv=predicted_curvature_arcsec_inv,
        prediction_finite=prediction_finite,
        observed_tangent_angle_rad=observed_tangent_angle_rad,
        observed_curvature_arcsec_inv=observed_curvature_arcsec_inv,
        sigma_tangent_angle_rad=sigma_tangent_angle_rad,
        sigma_curvature_arcsec_inv=sigma_curvature_arcsec_inv,
        reliability=reliability,
        active_arcs=active_arcs,
        tangent_sigma_floor_rad=tangent_sigma_floor_rad,
        curvature_sigma_floor_arcsec_inv=curvature_sigma_floor_arcsec_inv,
        branch_weight=branch_weight,
        frame_weight=frame_weight,
    )
    active = jnp.asarray(active_arcs, dtype=bool)
    arc_loglike = jnp.sum(jnp.where(active, terms.row_loglike, 0.0))
    finite = (
        jnp.all(jnp.where(active, terms.finite_row, True))
        & jnp.isfinite(arc_loglike)
    )
    return jnp.where(finite, arc_loglike, jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64))


def _forward_metric_image_plane_bin_loglike(
    residual_beta_x: jnp.ndarray,
    residual_beta_y: jnp.ndarray,
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
    sigma_per_image: jnp.ndarray,
    reliability_per_image: jnp.ndarray,
    image_has_constraint: jnp.ndarray,
    image_sigma_int: jnp.ndarray,
    scatter_var_x: jnp.ndarray,
    scatter_var_y: jnp.ndarray,
    covariance_floor: float,
    outlier_sigma_arcsec: float,
    max_gain: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN,
    max_residual_arcsec: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC,
    residual_loss: str = DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS,
    student_t_nu: float = DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU,
    family_idx: jnp.ndarray | None = None,
    n_families: int | None = None,
    image_presence_penalty_weight: float = 0.0,
    image_presence_match_radius_arcsec: float = DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC,
    image_presence_temperature_arcsec: float = DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC,
    image_presence_count_softness: float = DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS,
    image_presence_count_margin: float = DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN,
) -> jnp.ndarray:
    image_sigma2 = jnp.square(sigma_per_image) + jnp.square(image_sigma_int)
    cov_floor = jnp.asarray(covariance_floor, dtype=jnp.float64)
    c00 = image_sigma2 * (jnp.square(jac_a00) + jnp.square(jac_a01)) + scatter_var_x + cov_floor
    c11 = image_sigma2 * (jnp.square(jac_a10) + jnp.square(jac_a11)) + scatter_var_y + cov_floor
    c01 = image_sigma2 * (jac_a00 * jac_a10 + jac_a01 * jac_a11)
    if float(max_gain) > 0.0:
        gain_floor = image_sigma2 / jnp.square(jnp.asarray(float(max_gain), dtype=jnp.float64))
        c00 = c00 + gain_floor
        c11 = c11 + gain_floor
    c00 = jnp.maximum(c00, cov_floor)
    c11 = jnp.maximum(c11, cov_floor)
    c00, c11, det = _jittered_2x2_covariance_det(c00, c01, c11)
    inv00 = c11 / det
    inv11 = c00 / det
    inv01 = -c01 / det

    dx, dy = _smooth_residual_cap(residual_beta_x, residual_beta_y, max_residual_arcsec)
    quad = dx * (inv00 * dx + inv01 * dy) + dy * (inv01 * dx + inv11 * dy)
    logdet = jnp.log(det)
    if str(residual_loss) == LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_STUDENT_T:
        family_ll = _student_t_2d_loglike_from_quad_logdet(quad, logdet, student_t_nu)
    else:
        family_ll = -0.5 * (quad + jnp.log(jnp.square(2.0 * jnp.pi)) + logdet)

    reliability = jnp.clip(reliability_per_image, 1.0e-6, 1.0 - 1.0e-6)
    outlier_sigma2 = jnp.square(jnp.asarray(outlier_sigma_arcsec, dtype=jnp.float64))
    outlier_ll = -0.5 * (
        (jnp.square(dx) + jnp.square(dy)) / outlier_sigma2 + 2.0 * jnp.log(2.0 * jnp.pi * outlier_sigma2)
    )
    mixture_ll = jnp.logaddexp(jnp.log(reliability) + family_ll, jnp.log1p(-reliability) + outlier_ll)
    bin_loglike = jnp.sum(jnp.where(image_has_constraint, mixture_ll, 0.0))
    presence_finite = jnp.asarray(True)
    if (
        float(image_presence_penalty_weight) > 0.0
        and family_idx is not None
        and n_families is not None
    ):
        presence_residual2, presence_residual_finite = _forward_metric_image_presence_residual2(
            residual_beta_x,
            residual_beta_y,
            jac_a00,
            jac_a01,
            jac_a10,
            jac_a11,
            sigma_per_image,
            image_sigma_int,
            covariance_floor,
            max_gain=max_gain,
        )
        presence_finite = jnp.all(presence_residual_finite)
        bin_loglike = bin_loglike + _soft_observed_image_presence_loglike_from_residual2(
            residual2=presence_residual2,
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
        jnp.all(jnp.isfinite(c00))
        & jnp.all(jnp.isfinite(c01))
        & jnp.all(jnp.isfinite(c11))
        & jnp.all(jnp.isfinite(det))
        & jnp.all(jnp.isfinite(dx))
        & jnp.all(jnp.isfinite(dy))
        & jnp.all(jnp.isfinite(quad))
        & presence_finite
        & jnp.isfinite(bin_loglike)
    )
    return jnp.where(finite, bin_loglike, jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64))


def _symmetric_2x2_min_max_frame(
    m00: jnp.ndarray,
    m01: jnp.ndarray,
    m11: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    trace = jnp.maximum(m00 + m11, jnp.asarray(0.0, dtype=jnp.float64))
    diff = m00 - m11
    gap = jnp.sqrt(jnp.maximum(jnp.square(diff) + 4.0 * jnp.square(m01), 0.0))
    lambda_min = jnp.maximum(0.5 * (trace - gap), jnp.asarray(0.0, dtype=jnp.float64))
    lambda_max = jnp.maximum(0.5 * (trace + gap), jnp.asarray(0.0, dtype=jnp.float64))
    max_angle = 0.5 * jnp.arctan2(2.0 * m01, diff)
    max_x = jnp.cos(max_angle)
    max_y = jnp.sin(max_angle)
    min_x = -max_y
    min_y = max_x
    finite = (
        jnp.isfinite(m00)
        & jnp.isfinite(m01)
        & jnp.isfinite(m11)
        & jnp.isfinite(lambda_min)
        & jnp.isfinite(lambda_max)
        & jnp.isfinite(min_x)
        & jnp.isfinite(min_y)
        & jnp.isfinite(max_x)
        & jnp.isfinite(max_y)
    )
    return min_x, min_y, max_x, max_y, lambda_min, lambda_max, finite


def _fold_regularized_singular_frame_from_jacobian(
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    source00 = jnp.square(jac_a00) + jnp.square(jac_a01)
    source01 = jac_a00 * jac_a10 + jac_a01 * jac_a11
    source11 = jnp.square(jac_a10) + jnp.square(jac_a11)
    (
        source_critical_x,
        source_critical_y,
        source_noncritical_x,
        source_noncritical_y,
        lambda_min,
        lambda_max,
        source_finite,
    ) = _symmetric_2x2_min_max_frame(source00, source01, source11)

    image00 = jnp.square(jac_a00) + jnp.square(jac_a10)
    image01 = jac_a00 * jac_a01 + jac_a10 * jac_a11
    image11 = jnp.square(jac_a01) + jnp.square(jac_a11)
    (
        image_critical_x,
        image_critical_y,
        image_noncritical_x,
        image_noncritical_y,
        _image_lambda_min,
        _image_lambda_max,
        image_finite,
    ) = _symmetric_2x2_min_max_frame(image00, image01, image11)
    critical_av_x = jac_a00 * image_critical_x + jac_a01 * image_critical_y
    critical_av_y = jac_a10 * image_critical_x + jac_a11 * image_critical_y
    critical_alignment = source_critical_x * critical_av_x + source_critical_y * critical_av_y
    critical_sign = jnp.where(critical_alignment < 0.0, -1.0, 1.0)
    source_critical_x = source_critical_x * critical_sign
    source_critical_y = source_critical_y * critical_sign
    noncritical_av_x = jac_a00 * image_noncritical_x + jac_a01 * image_noncritical_y
    noncritical_av_y = jac_a10 * image_noncritical_x + jac_a11 * image_noncritical_y
    noncritical_alignment = (
        source_noncritical_x * noncritical_av_x
        + source_noncritical_y * noncritical_av_y
    )
    noncritical_sign = jnp.where(noncritical_alignment < 0.0, -1.0, 1.0)
    source_noncritical_x = source_noncritical_x * noncritical_sign
    source_noncritical_y = source_noncritical_y * noncritical_sign
    singular_min = jnp.sqrt(jnp.maximum(lambda_min, jnp.asarray(0.0, dtype=jnp.float64)))
    singular_max = jnp.sqrt(jnp.maximum(lambda_max, jnp.asarray(0.0, dtype=jnp.float64)))
    finite = (
        source_finite
        & image_finite
        & jnp.isfinite(critical_alignment)
        & jnp.isfinite(noncritical_alignment)
        & jnp.isfinite(singular_min)
        & jnp.isfinite(singular_max)
    )
    return (
        source_critical_x,
        source_critical_y,
        source_noncritical_x,
        source_noncritical_y,
        image_critical_x,
        image_critical_y,
        image_noncritical_x,
        image_noncritical_y,
        singular_min,
        singular_max,
        finite,
    )


def _fold_regularized_image_plane_bin_loglike(
    residual_beta_x: jnp.ndarray,
    residual_beta_y: jnp.ndarray,
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
    sigma_per_image: jnp.ndarray,
    reliability_per_image: jnp.ndarray,
    image_has_constraint: jnp.ndarray,
    image_sigma_int: jnp.ndarray,
    scatter_var_x: jnp.ndarray,
    scatter_var_y: jnp.ndarray,
    covariance_floor: float,
    outlier_sigma_arcsec: float,
    fold_curvature_arcsec_inv: float = DEFAULT_FOLD_CURVATURE_ARCSEC_INV,
    fold_kappa_eff: jnp.ndarray | None = None,
    fold_frame: tuple[jnp.ndarray, ...] | None = None,
    max_gain: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN,
    max_residual_arcsec: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC,
    residual_loss: str = DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS,
    student_t_nu: float = DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU,
    family_idx: jnp.ndarray | None = None,
    n_families: int | None = None,
    image_presence_penalty_weight: float = 0.0,
    image_presence_match_radius_arcsec: float = DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC,
    image_presence_temperature_arcsec: float = DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC,
    image_presence_count_softness: float = DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS,
    image_presence_count_margin: float = DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN,
) -> jnp.ndarray:
    image_sigma2 = jnp.square(sigma_per_image) + jnp.square(image_sigma_int)
    sigma_eff2 = _image_plane_effective_sigma2(
        sigma_per_image,
        image_sigma_int,
        covariance_floor,
    )
    cov_floor = jnp.asarray(covariance_floor, dtype=jnp.float64)
    dx, dy = _smooth_residual_cap(residual_beta_x, residual_beta_y, max_residual_arcsec)

    source00 = jnp.square(jac_a00) + jnp.square(jac_a01)
    source01 = jac_a00 * jac_a10 + jac_a01 * jac_a11
    source11 = jnp.square(jac_a10) + jnp.square(jac_a11)
    c00 = image_sigma2 * source00 + scatter_var_x + cov_floor
    c11 = image_sigma2 * source11 + scatter_var_y + cov_floor
    c01 = image_sigma2 * source01
    if float(max_gain) > 0.0:
        gain_floor = image_sigma2 / jnp.square(jnp.asarray(float(max_gain), dtype=jnp.float64))
        c00 = c00 + gain_floor
        c11 = c11 + gain_floor
    c00 = jnp.maximum(c00, cov_floor)
    c11 = jnp.maximum(c11, cov_floor)
    c00, c11, det = _jittered_2x2_covariance_det(c00, c01, c11)
    inv00 = c11 / det
    inv11 = c00 / det
    inv01 = -c01 / det
    forward_quad = dx * (inv00 * dx + inv01 * dy) + dy * (inv01 * dx + inv11 * dy)
    forward_logdet = jnp.log(det)
    if str(residual_loss) == LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_STUDENT_T:
        forward_ll = _student_t_2d_loglike_from_quad_logdet(forward_quad, forward_logdet, student_t_nu)
    else:
        forward_ll = -0.5 * (forward_quad + 2.0 * jnp.log(2.0 * jnp.pi) + forward_logdet)

    if fold_frame is None:
        fold_frame = _fold_regularized_singular_frame_from_jacobian(
            jac_a00,
            jac_a01,
            jac_a10,
            jac_a11,
        )
    (
        source_critical_x,
        source_critical_y,
        source_noncritical_x,
        source_noncritical_y,
        _image_critical_x,
        _image_critical_y,
        _image_noncritical_x,
        _image_noncritical_y,
        singular_min,
        singular_max,
        frame_finite,
    ) = fold_frame
    source_critical_residual = dx * source_critical_x + dy * source_critical_y
    source_noncritical_residual = dx * source_noncritical_x + dy * source_noncritical_y
    singular_floor = jnp.asarray(1.0e-12, dtype=jnp.float64)
    singular_max_eff = jnp.maximum(singular_max, singular_floor)
    theta_noncritical = source_noncritical_residual / singular_max_eff

    if fold_kappa_eff is None:
        kappa_eff = jnp.ones_like(dx) * jnp.asarray(float(fold_curvature_arcsec_inv), dtype=jnp.float64)
    else:
        kappa_eff = jnp.asarray(fold_kappa_eff, dtype=jnp.float64)
    curvature_floor = jnp.asarray(1.0e-12, dtype=jnp.float64)
    curvature_ok = jnp.isfinite(kappa_eff) & (jnp.abs(kappa_eff) > curvature_floor)
    discriminant = jnp.square(singular_min) - 2.0 * kappa_eff * source_critical_residual
    discriminant_tolerance = (
        jnp.asarray(1.0e-12, dtype=jnp.float64)
        * (
            1.0
            + jnp.square(singular_min)
            + jnp.abs(2.0 * kappa_eff * source_critical_residual)
        )
    )
    root_ok = curvature_ok & jnp.isfinite(discriminant) & (discriminant >= -discriminant_tolerance)
    discriminant_safe = jnp.where(
        root_ok,
        jnp.maximum(discriminant, jnp.asarray(0.0, dtype=jnp.float64)),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    sqrt_discriminant = jnp.sqrt(discriminant_safe)
    kappa_safe = jnp.where(curvature_ok, kappa_eff, jnp.ones_like(kappa_eff))
    theta_critical_root_a = (-singular_min + sqrt_discriminant) / kappa_safe
    theta_critical_root_b = (-singular_min - sqrt_discriminant) / kappa_safe
    theta_critical2 = jnp.minimum(jnp.square(theta_critical_root_a), jnp.square(theta_critical_root_b))
    fold_image_residual2 = jnp.maximum(jnp.square(theta_noncritical) + theta_critical2, 0.0)
    fold_quad = fold_image_residual2 / sigma_eff2
    fold_logdet = 2.0 * jnp.log(sigma_eff2)
    if str(residual_loss) == LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_STUDENT_T:
        fold_ll = _student_t_2d_loglike_from_quad_logdet(fold_quad, fold_logdet, student_t_nu)
    else:
        fold_ll = -0.5 * (fold_quad + 2.0 * jnp.log(2.0 * jnp.pi * sigma_eff2))

    singular_threshold = jnp.asarray(DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD, dtype=jnp.float64)
    singular_softness = jnp.asarray(DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS, dtype=jnp.float64)
    near_critical_weight = jax.nn.sigmoid(
        (
            singular_threshold
            - singular_min
        )
        / singular_softness
    )
    near_critical_weight = jnp.where(
        singular_min <= singular_floor,
        jnp.asarray(1.0, dtype=jnp.float64),
        near_critical_weight,
    )
    fold_weight = jnp.where(root_ok & frame_finite, near_critical_weight, jnp.asarray(0.0, dtype=jnp.float64))
    fold_weight = jnp.clip(fold_weight, jnp.asarray(0.0, dtype=jnp.float64), jnp.asarray(1.0 - 1.0e-12, dtype=jnp.float64))
    blended_family_ll = jnp.where(
        fold_weight > 0.0,
        jnp.logaddexp(
            jnp.log1p(-fold_weight) + forward_ll,
            jnp.log(fold_weight) + fold_ll,
        ),
        forward_ll,
    )
    family_ll = jnp.where(fold_weight >= 1.0 - 1.0e-9, fold_ll, blended_family_ll)

    reliability = jnp.clip(reliability_per_image, 1.0e-6, 1.0 - 1.0e-6)
    outlier_sigma2 = jnp.square(jnp.asarray(outlier_sigma_arcsec, dtype=jnp.float64))
    outlier_residual2 = jnp.square(dx) + jnp.square(dy)
    outlier_ll = -0.5 * (outlier_residual2 / outlier_sigma2 + 2.0 * jnp.log(2.0 * jnp.pi * outlier_sigma2))
    mixture_ll = jnp.logaddexp(jnp.log(reliability) + family_ll, jnp.log1p(-reliability) + outlier_ll)
    bin_loglike = jnp.sum(jnp.where(image_has_constraint, mixture_ll, 0.0))
    presence_finite = jnp.asarray(True)
    if (
        float(image_presence_penalty_weight) > 0.0
        and family_idx is not None
        and n_families is not None
    ):
        forward_presence_residual2, forward_presence_finite = _forward_metric_image_presence_residual2(
            residual_beta_x,
            residual_beta_y,
            jac_a00,
            jac_a01,
            jac_a10,
            jac_a11,
            sigma_per_image,
            image_sigma_int,
            covariance_floor,
            max_gain=max_gain,
        )
        image_residual2 = jnp.where(
            fold_weight >= 0.5,
            fold_image_residual2,
            forward_presence_residual2,
        )
        presence_finite = jnp.all(jnp.isfinite(image_residual2)) & forward_presence_finite
        bin_loglike = bin_loglike + _soft_observed_image_presence_loglike_from_residual2(
            residual2=image_residual2,
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
        jnp.all(frame_finite)
        & jnp.all(jnp.isfinite(c00))
        & jnp.all(jnp.isfinite(c01))
        & jnp.all(jnp.isfinite(c11))
        & jnp.all(jnp.isfinite(det))
        & jnp.all(jnp.isfinite(dx))
        & jnp.all(jnp.isfinite(dy))
        & jnp.all(jnp.isfinite(forward_quad))
        & jnp.all(jnp.isfinite(fold_quad))
        & jnp.all(jnp.isfinite(fold_image_residual2))
        & jnp.all(jnp.isfinite(sigma_eff2))
        & presence_finite
        & jnp.isfinite(bin_loglike)
    )
    return jnp.where(finite, bin_loglike, jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64))


def _smooth_residual_cap(
    residual_x: jnp.ndarray,
    residual_y: jnp.ndarray,
    max_residual_arcsec: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if float(max_residual_arcsec) <= 0.0:
        return residual_x, residual_y
    radius = jnp.sqrt(jnp.square(residual_x) + jnp.square(residual_y) + jnp.asarray(1.0e-30, dtype=jnp.float64))
    scaled_radius = radius / jnp.asarray(float(max_residual_arcsec), dtype=jnp.float64)
    scale = jnp.tanh(scaled_radius) / jnp.maximum(scaled_radius, jnp.asarray(1.0e-12, dtype=jnp.float64))
    return residual_x * scale, residual_y * scale


def _student_t_2d_loglike_from_quad_logdet(
    quad: jnp.ndarray,
    logdet: jnp.ndarray,
    student_t_nu: float = DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU,
) -> jnp.ndarray:
    nu = jnp.asarray(float(student_t_nu), dtype=jnp.float64)
    safe_quad = jnp.maximum(quad, jnp.asarray(0.0, dtype=jnp.float64))
    return (
        jsp_special.gammaln(0.5 * (nu + 2.0))
        - jsp_special.gammaln(0.5 * nu)
        - jnp.log(nu * jnp.pi)
        - 0.5 * logdet
        - 0.5 * (nu + 2.0) * jnp.log1p(safe_quad / nu)
    )


def _source_plane_bin_loglike(
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
    inv_abs_mu: jnp.ndarray,
    covariance_floor: float,
    outlier_sigma_arcsec: float,
    max_gain: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN,
    max_residual_arcsec: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC,
    residual_loss: str = DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS,
    student_t_nu: float = DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU,
) -> jnp.ndarray:
    if float(max_gain) > 0.0:
        inv_abs_mu = jnp.maximum(
            inv_abs_mu,
            jnp.asarray(1.0 / float(max_gain) ** 2, dtype=jnp.float64),
        )
    sigma2_image = jnp.square(sigma_per_image) * inv_abs_mu
    cov_floor = jnp.asarray(covariance_floor, dtype=jnp.float64)
    sigma2_x = sigma2_image + jnp.square(source_sigma_int) + scatter_var_x + cov_floor
    sigma2_y = sigma2_image + jnp.square(source_sigma_int) + scatter_var_y + cov_floor
    sigma2_weight = 0.5 * (sigma2_x + sigma2_y)
    reliability = jnp.clip(reliability_per_image, 1.0e-6, 1.0 - 1.0e-6)
    weights = reliability / jnp.maximum(sigma2_weight, 1.0e-18)
    sum_w = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(weights)
    sum_bx = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(weights * beta_x)
    sum_by = jnp.zeros(n_families, dtype=jnp.float64).at[family_idx].add(weights * beta_y)
    centroid_x = sum_bx / jnp.maximum(sum_w, 1.0e-18)
    centroid_y = sum_by / jnp.maximum(sum_w, 1.0e-18)
    dx = beta_x - centroid_x[family_idx]
    dy = beta_y - centroid_y[family_idx]
    dx, dy = _smooth_residual_cap(dx, dy, max_residual_arcsec)
    quad = (dx**2) / sigma2_x + (dy**2) / sigma2_y
    if str(residual_loss) == LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_STUDENT_T:
        family_ll = _student_t_2d_loglike_from_quad_logdet(
            quad,
            jnp.log(sigma2_x) + jnp.log(sigma2_y),
            student_t_nu,
        )
    else:
        family_ll = -0.5 * (
            quad
            + jnp.log(2.0 * jnp.pi * sigma2_x)
            + jnp.log(2.0 * jnp.pi * sigma2_y)
        )
    outlier_sigma2 = jnp.square(jnp.asarray(outlier_sigma_arcsec, dtype=jnp.float64))
    outlier_ll = -0.5 * (
        (dx**2 + dy**2) / outlier_sigma2 + 2.0 * jnp.log(2.0 * jnp.pi * outlier_sigma2)
    )
    mixture_ll = jnp.logaddexp(jnp.log(reliability) + family_ll, jnp.log1p(-reliability) + outlier_ll)
    bin_loglike = jnp.sum(jnp.where(image_has_constraint, mixture_ll, 0.0))
    finite = (
        jnp.all(jnp.isfinite(inv_abs_mu))
        & jnp.all(jnp.isfinite(sigma2_x))
        & jnp.all(jnp.isfinite(sigma2_y))
        & jnp.all(jnp.isfinite(dx))
        & jnp.all(jnp.isfinite(dy))
        & jnp.all(jnp.isfinite(family_ll))
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
    max_gain: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN,
    max_residual_arcsec: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    inv00, inv01, inv10, inv11, finite = _linearized_image_plane_inverse_operator(
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
        determinant_floor=determinant_floor,
        max_gain=max_gain,
    )
    delta_x = -(inv00 * f_x + inv01 * f_y)
    delta_y = -(inv10 * f_x + inv11 * f_y)
    delta_x, delta_y = _smooth_residual_cap(delta_x, delta_y, max_residual_arcsec)
    finite = (
        finite
        & jnp.isfinite(f_x)
        & jnp.isfinite(f_y)
        & jnp.isfinite(delta_x)
        & jnp.isfinite(delta_y)
    )
    return delta_x, delta_y, finite


def _linearized_image_plane_inverse_operator(
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
    *,
    determinant_floor: float = 1.0e-12,
    max_gain: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN,
    require_determinant_floor: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    del determinant_floor, require_determinant_floor
    finite_entries = (
        jnp.isfinite(jac_a00)
        & jnp.isfinite(jac_a01)
        & jnp.isfinite(jac_a10)
        & jnp.isfinite(jac_a11)
    )
    m00 = jnp.square(jac_a00) + jnp.square(jac_a10)
    m01 = jac_a00 * jac_a01 + jac_a10 * jac_a11
    m11 = jnp.square(jac_a01) + jnp.square(jac_a11)
    if float(max_gain) > 0.0:
        lam = jnp.asarray(0.25 / float(max_gain) ** 2, dtype=jnp.float64)
    else:
        normal_scale = 0.5 * (m00 + m11)
        lam = (
            jnp.asarray(DEFAULT_JACOBIAN_INVERSE_DAMPING_RELATIVE, dtype=jnp.float64) * normal_scale
            + jnp.asarray(DEFAULT_JACOBIAN_INVERSE_DAMPING_ABSOLUTE, dtype=jnp.float64)
        )
    m00 = m00 + lam
    m11 = m11 + lam
    det_m = m00 * m11 - jnp.square(m01)
    det_m_safe = jnp.maximum(det_m, jnp.asarray(MIN_JACOBIAN_NORMAL_DETERMINANT_GUARD, dtype=jnp.float64))
    inv00 = (m11 * jac_a00 - m01 * jac_a01) / det_m_safe
    inv01 = (m11 * jac_a10 - m01 * jac_a11) / det_m_safe
    inv10 = (-m01 * jac_a00 + m00 * jac_a01) / det_m_safe
    inv11 = (-m01 * jac_a10 + m00 * jac_a11) / det_m_safe
    finite = (
        finite_entries
        & jnp.isfinite(lam)
        & jnp.isfinite(det_m)
        & jnp.isfinite(inv00)
        & jnp.isfinite(inv01)
        & jnp.isfinite(inv10)
        & jnp.isfinite(inv11)
    )
    return inv00, inv01, inv10, inv11, finite


def _anchored_solved_image_plane_step_from_jacobian(
    f_x: jnp.ndarray,
    f_y: jnp.ndarray,
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
    *,
    trust_radius_arcsec: float = DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
    lm_damping_relative: float = DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
    lm_damping_absolute: float = DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    finite_entries = (
        jnp.isfinite(f_x)
        & jnp.isfinite(f_y)
        & jnp.isfinite(jac_a00)
        & jnp.isfinite(jac_a01)
        & jnp.isfinite(jac_a10)
        & jnp.isfinite(jac_a11)
    )
    normal00 = jnp.square(jac_a00) + jnp.square(jac_a10)
    normal01 = jac_a00 * jac_a01 + jac_a10 * jac_a11
    normal11 = jnp.square(jac_a01) + jnp.square(jac_a11)
    trace = jnp.maximum(normal00 + normal11, jnp.asarray(0.0, dtype=jnp.float64))
    lam = (
        jnp.asarray(float(lm_damping_absolute), dtype=jnp.float64)
        + jnp.asarray(float(lm_damping_relative), dtype=jnp.float64) * trace
    )
    h00 = normal00 + lam
    h11 = normal11 + lam
    h01 = normal01
    rhs0 = -(jac_a00 * f_x + jac_a10 * f_y)
    rhs1 = -(jac_a01 * f_x + jac_a11 * f_y)
    det_h = h00 * h11 - jnp.square(h01)
    det_h_safe = jnp.maximum(det_h, jnp.asarray(MIN_JACOBIAN_NORMAL_DETERMINANT_GUARD, dtype=jnp.float64))
    delta_x = (h11 * rhs0 - h01 * rhs1) / det_h_safe
    delta_y = (-h01 * rhs0 + h00 * rhs1) / det_h_safe
    delta_x, delta_y = _smooth_residual_cap(delta_x, delta_y, trust_radius_arcsec)
    finite = (
        finite_entries
        & jnp.isfinite(lam)
        & jnp.isfinite(det_h)
        & jnp.isfinite(delta_x)
        & jnp.isfinite(delta_y)
    )
    return delta_x, delta_y, finite


def _critical_arc_normal_matrix_entries(
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    normal00 = jnp.square(jac_a00) + jnp.square(jac_a10)
    normal01 = jac_a00 * jac_a01 + jac_a10 * jac_a11
    normal11 = jnp.square(jac_a01) + jnp.square(jac_a11)
    trace = jnp.maximum(normal00 + normal11, jnp.asarray(0.0, dtype=jnp.float64))
    diff = normal00 - normal11
    gap2 = jnp.square(diff) + 4.0 * jnp.square(normal01)
    value_gap_floor = (
        jnp.asarray(CRITICAL_ARC_EIGENGAP_RELATIVE_SOFTENING, dtype=jnp.float64) * trace
        + jnp.asarray(CRITICAL_ARC_EIGENGAP_VALUE_ABSOLUTE_SOFTENING, dtype=jnp.float64)
    )
    gap = jnp.sqrt(gap2 + jnp.square(value_gap_floor))
    lambda_min = jnp.maximum(0.5 * (trace - gap), jnp.asarray(0.0, dtype=jnp.float64))
    lambda_max = jnp.maximum(0.5 * (trace + gap), jnp.asarray(0.0, dtype=jnp.float64))
    return normal00, normal01, normal11, trace, gap, lambda_min, lambda_max, diff


def _critical_arc_singular_values_from_lambdas(
    lambda_min: jnp.ndarray,
    lambda_max: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    singular_floor2 = jnp.square(jnp.asarray(CRITICAL_ARC_SINGULAR_VALUE_FLOOR, dtype=jnp.float64))
    return jnp.sqrt(lambda_min + singular_floor2), jnp.sqrt(lambda_max + singular_floor2)


def _critical_arc_critical_direction_projector_from_normal_entries(
    normal00: jnp.ndarray,
    normal01: jnp.ndarray,
    normal11: jnp.ndarray,
    trace: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    diff = normal00 - normal11
    projector_gap_floor = jnp.asarray(CRITICAL_ARC_EIGENGAP_RELATIVE_SOFTENING, dtype=jnp.float64) * jnp.maximum(
        trace,
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    projector_gap = jnp.sqrt(jnp.square(diff) + 4.0 * jnp.square(normal01) + jnp.square(projector_gap_floor))
    critical_p00 = 0.5 - 0.5 * diff / projector_gap
    critical_p01 = -normal01 / projector_gap
    critical_p11 = 0.5 + 0.5 * diff / projector_gap
    return critical_p00, critical_p01, critical_p11


def _critical_arc_geometry_from_jacobian(
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    normal00, normal01, normal11, trace, _gap, lambda_min, lambda_max, _diff = _critical_arc_normal_matrix_entries(
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
    )
    singular_min, singular_max = _critical_arc_singular_values_from_lambdas(lambda_min, lambda_max)
    critical_p00, critical_p01, critical_p11 = _critical_arc_critical_direction_projector_from_normal_entries(
        normal00,
        normal01,
        normal11,
        trace,
    )
    finite = (
        jnp.isfinite(normal00)
        & jnp.isfinite(normal01)
        & jnp.isfinite(normal11)
        & jnp.isfinite(singular_min)
        & jnp.isfinite(singular_max)
        & jnp.isfinite(critical_p00)
        & jnp.isfinite(critical_p01)
        & jnp.isfinite(critical_p11)
    )
    return singular_min, singular_max, critical_p00, critical_p01, critical_p11, finite


def _critical_arc_lm_geometry_from_jacobian(
    f_x: jnp.ndarray,
    f_y: jnp.ndarray,
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
    *,
    trust_radius_arcsec: float = DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC,
    lm_damping_relative: float = DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE,
    lm_damping_absolute: float = DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    finite_entries = (
        jnp.isfinite(f_x)
        & jnp.isfinite(f_y)
        & jnp.isfinite(jac_a00)
        & jnp.isfinite(jac_a01)
        & jnp.isfinite(jac_a10)
        & jnp.isfinite(jac_a11)
    )
    normal00, normal01, normal11, trace, _gap, lambda_min, lambda_max, _diff = _critical_arc_normal_matrix_entries(
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
    )
    singular_min, singular_max = _critical_arc_singular_values_from_lambdas(lambda_min, lambda_max)
    critical_p00, critical_p01, critical_p11 = _critical_arc_critical_direction_projector_from_normal_entries(
        normal00,
        normal01,
        normal11,
        trace,
    )
    lam = (
        jnp.asarray(float(lm_damping_absolute), dtype=jnp.float64)
        + jnp.asarray(float(lm_damping_relative), dtype=jnp.float64) * trace
    )
    h00 = normal00 + lam
    h11 = normal11 + lam
    h01 = normal01
    rhs0 = -(jac_a00 * f_x + jac_a10 * f_y)
    rhs1 = -(jac_a01 * f_x + jac_a11 * f_y)
    det_h = h00 * h11 - jnp.square(h01)
    det_h_safe = jnp.maximum(det_h, jnp.asarray(MIN_JACOBIAN_NORMAL_DETERMINANT_GUARD, dtype=jnp.float64))
    delta_x = (h11 * rhs0 - h01 * rhs1) / det_h_safe
    delta_y = (-h01 * rhs0 + h00 * rhs1) / det_h_safe
    delta_x, delta_y = _smooth_residual_cap(delta_x, delta_y, trust_radius_arcsec)
    finite = (
        finite_entries
        & jnp.isfinite(lam)
        & jnp.isfinite(det_h)
        & jnp.isfinite(delta_x)
        & jnp.isfinite(delta_y)
        & jnp.isfinite(singular_min)
        & jnp.isfinite(singular_max)
        & jnp.isfinite(critical_p00)
        & jnp.isfinite(critical_p01)
        & jnp.isfinite(critical_p11)
    )
    return (
        delta_x,
        delta_y,
        singular_min,
        singular_max,
        critical_p00,
        critical_p01,
        critical_p11,
        finite,
    )


def _critical_arc_lm_step_from_jacobian(
    f_x: jnp.ndarray,
    f_y: jnp.ndarray,
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
    *,
    trust_radius_arcsec: float = DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC,
    lm_damping_relative: float = DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE,
    lm_damping_absolute: float = DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    delta_x, delta_y, *_unused, finite = _critical_arc_lm_geometry_from_jacobian(
        f_x,
        f_y,
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
        trust_radius_arcsec=trust_radius_arcsec,
        lm_damping_relative=lm_damping_relative,
        lm_damping_absolute=lm_damping_absolute,
    )
    return delta_x, delta_y, finite


def _critical_arc_projected_quadratics(
    residual_x: jnp.ndarray,
    residual_y: jnp.ndarray,
    critical_p00: jnp.ndarray,
    critical_p01: jnp.ndarray,
    critical_p11: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    residual2 = jnp.square(residual_x) + jnp.square(residual_y)
    critical_direction_quad_raw = (
        jnp.square(residual_x) * critical_p00
        + 2.0 * residual_x * residual_y * critical_p01
        + jnp.square(residual_y) * critical_p11
    )
    critical_direction_quad = jnp.maximum(critical_direction_quad_raw, jnp.asarray(0.0, dtype=jnp.float64))
    noncritical_direction_quad = jnp.maximum(residual2 - critical_direction_quad, jnp.asarray(0.0, dtype=jnp.float64))
    return critical_direction_quad, noncritical_direction_quad


def _critical_arc_jacobian_frame(
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    normal00, normal01, normal11, _trace, _gap, lambda_min, lambda_max, diff = _critical_arc_normal_matrix_entries(
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
    )
    singular_min, singular_max = _critical_arc_singular_values_from_lambdas(lambda_min, lambda_max)
    angle = 0.5 * jnp.arctan2(2.0 * normal01, diff)
    noncritical_direction_x = jnp.cos(angle)
    noncritical_direction_y = jnp.sin(angle)
    critical_direction_x = -noncritical_direction_y
    critical_direction_y = noncritical_direction_x
    finite = (
        jnp.isfinite(normal00)
        & jnp.isfinite(normal01)
        & jnp.isfinite(normal11)
        & jnp.isfinite(singular_min)
        & jnp.isfinite(singular_max)
        & jnp.isfinite(critical_direction_x)
        & jnp.isfinite(critical_direction_y)
        & jnp.isfinite(noncritical_direction_x)
        & jnp.isfinite(noncritical_direction_y)
    )
    return critical_direction_x, critical_direction_y, noncritical_direction_x, noncritical_direction_y, singular_min, singular_max, finite


def _critical_arc_branch_probability(
    singular_min: jnp.ndarray,
    *,
    base_prob: float = DEFAULT_CRITICAL_ARC_BASE_PROB,
    max_prob: float = DEFAULT_CRITICAL_ARC_MAX_PROB,
    singular_threshold: float = DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD,
    singular_softness: float = DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS,
) -> jnp.ndarray:
    base = jnp.asarray(float(base_prob), dtype=jnp.float64)
    high = jnp.asarray(float(max_prob), dtype=jnp.float64)
    transition = jax.nn.sigmoid(
        (jnp.asarray(float(singular_threshold), dtype=jnp.float64) - singular_min)
        / jnp.asarray(float(singular_softness), dtype=jnp.float64)
    )
    prob = base + (high - base) * transition
    return jnp.clip(prob, 1.0e-6, 1.0 - 1.0e-6)


class _CriticalArcMixtureTerms(NamedTuple):
    sigma2: jnp.ndarray
    reliability: jnp.ndarray
    critical_quad: jnp.ndarray
    noncritical_quad: jnp.ndarray
    arc_prob: jnp.ndarray
    point_ll: jnp.ndarray
    arc_ll: jnp.ndarray
    outlier_ll: jnp.ndarray
    inlier_ll: jnp.ndarray
    mixture_ll: jnp.ndarray


class _CriticalArcMixtureResponsibilities(NamedTuple):
    point_log_weight: jnp.ndarray
    arc_log_weight: jnp.ndarray
    inlier_log_weight: jnp.ndarray
    outlier_log_weight: jnp.ndarray
    point_inlier_responsibility: jnp.ndarray
    arc_inlier_responsibility: jnp.ndarray
    point_mixture_responsibility: jnp.ndarray
    arc_mixture_responsibility: jnp.ndarray
    inlier_responsibility: jnp.ndarray
    outlier_responsibility: jnp.ndarray


def _critical_arc_mixture_image_plane_terms(
    residual_x: jnp.ndarray,
    residual_y: jnp.ndarray,
    sigma_per_image: jnp.ndarray,
    reliability_per_image: jnp.ndarray,
    image_sigma_int: jnp.ndarray,
    covariance_floor: float,
    outlier_sigma_arcsec: float,
    singular_min: jnp.ndarray,
    critical_direction_projector_entries: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    *,
    residual_loss: str = DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS,
    student_t_nu: float = DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU,
    critical_direction_sigma_arcsec: float = DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC,
    base_prob: float = DEFAULT_CRITICAL_ARC_BASE_PROB,
    max_prob: float = DEFAULT_CRITICAL_ARC_MAX_PROB,
    singular_threshold: float = DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD,
    singular_softness: float = DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS,
) -> _CriticalArcMixtureTerms:
    sigma2 = _image_plane_effective_sigma2(
        sigma_per_image,
        image_sigma_int,
        covariance_floor,
    )
    reliability = jnp.clip(reliability_per_image, 1.0e-6, 1.0 - 1.0e-6)
    critical_p00, critical_p01, critical_p11 = critical_direction_projector_entries
    critical_quad, noncritical_quad = _critical_arc_projected_quadratics(
        residual_x,
        residual_y,
        critical_p00,
        critical_p01,
        critical_p11,
    )
    point_quad = (jnp.square(residual_x) + jnp.square(residual_y)) / sigma2
    point_logdet = 2.0 * jnp.log(sigma2)
    # Covariance-side arc branch: Sigma = sigma2 I + sigma_arc^2 P keeps the branch a
    # normalized density for the regularized (non-idempotent) projector. It reduces
    # exactly to the two-axis (sigma2, sigma2 + sigma_arc^2) form when P is an exact
    # projector and to the broad-isotropic sigma2 + sigma_arc^2/2 at degenerate frames.
    arc_extra_var = jnp.square(jnp.asarray(float(critical_direction_sigma_arcsec), dtype=jnp.float64))
    arc_sigma00 = sigma2 + arc_extra_var * critical_p00
    arc_sigma01 = arc_extra_var * critical_p01
    arc_sigma11 = sigma2 + arc_extra_var * critical_p11
    projector_det = jnp.maximum(
        critical_p00 * critical_p11 - jnp.square(critical_p01),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    arc_det = (
        jnp.square(sigma2)
        + sigma2 * arc_extra_var * (critical_p00 + critical_p11)
        + jnp.square(arc_extra_var) * projector_det
    )
    arc_quad = jnp.maximum(
        (
            arc_sigma11 * jnp.square(residual_x)
            - 2.0 * arc_sigma01 * residual_x * residual_y
            + arc_sigma00 * jnp.square(residual_y)
        )
        / arc_det,
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    arc_logdet = jnp.log(arc_det)
    arc_prob = _critical_arc_branch_probability(
        singular_min,
        base_prob=base_prob,
        max_prob=max_prob,
        singular_threshold=singular_threshold,
        singular_softness=singular_softness,
    )
    if str(residual_loss) == LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_STUDENT_T:
        point_ll = _student_t_2d_loglike_from_quad_logdet(point_quad, point_logdet, student_t_nu)
        arc_ll = _student_t_2d_loglike_from_quad_logdet(arc_quad, arc_logdet, student_t_nu)
    else:
        point_ll = -0.5 * (point_quad + 2.0 * jnp.log(2.0 * jnp.pi) + point_logdet)
        arc_ll = -0.5 * (arc_quad + 2.0 * jnp.log(2.0 * jnp.pi) + arc_logdet)
    inlier_ll = jnp.logaddexp(jnp.log1p(-arc_prob) + point_ll, jnp.log(arc_prob) + arc_ll)
    outlier_sigma2 = jnp.square(jnp.asarray(outlier_sigma_arcsec, dtype=jnp.float64))
    outlier_ll = -0.5 * (
        (jnp.square(residual_x) + jnp.square(residual_y)) / outlier_sigma2
        + 2.0 * jnp.log(2.0 * jnp.pi * outlier_sigma2)
    )
    mixture_ll = jnp.logaddexp(jnp.log(reliability) + inlier_ll, jnp.log1p(-reliability) + outlier_ll)
    return _CriticalArcMixtureTerms(
        sigma2=sigma2,
        reliability=reliability,
        critical_quad=critical_quad,
        noncritical_quad=noncritical_quad,
        arc_prob=arc_prob,
        point_ll=point_ll,
        arc_ll=arc_ll,
        outlier_ll=outlier_ll,
        inlier_ll=inlier_ll,
        mixture_ll=mixture_ll,
    )


def _critical_arc_mixture_image_plane_responsibilities(
    terms: _CriticalArcMixtureTerms,
) -> _CriticalArcMixtureResponsibilities:
    point_log_weight = (
        jnp.log(terms.reliability)
        + jnp.log(jnp.clip(1.0 - terms.arc_prob, 1.0e-300, 1.0))
        + terms.point_ll
    )
    arc_log_weight = (
        jnp.log(terms.reliability)
        + jnp.log(jnp.clip(terms.arc_prob, 1.0e-300, 1.0))
        + terms.arc_ll
    )
    inlier_log_weight = jnp.log(terms.reliability) + terms.inlier_ll
    outlier_log_weight = jnp.log1p(-terms.reliability) + terms.outlier_ll
    point_inlier_responsibility = jnp.exp(
        jnp.log(jnp.clip(1.0 - terms.arc_prob, 1.0e-300, 1.0))
        + terms.point_ll
        - terms.inlier_ll
    )
    arc_inlier_responsibility = jnp.exp(
        jnp.log(jnp.clip(terms.arc_prob, 1.0e-300, 1.0))
        + terms.arc_ll
        - terms.inlier_ll
    )
    point_mixture_responsibility = jnp.exp(point_log_weight - terms.mixture_ll)
    arc_mixture_responsibility = jnp.exp(arc_log_weight - terms.mixture_ll)
    inlier_responsibility = jnp.exp(inlier_log_weight - terms.mixture_ll)
    outlier_responsibility = jnp.exp(outlier_log_weight - terms.mixture_ll)
    return _CriticalArcMixtureResponsibilities(
        point_log_weight=point_log_weight,
        arc_log_weight=arc_log_weight,
        inlier_log_weight=inlier_log_weight,
        outlier_log_weight=outlier_log_weight,
        point_inlier_responsibility=point_inlier_responsibility,
        arc_inlier_responsibility=arc_inlier_responsibility,
        point_mixture_responsibility=point_mixture_responsibility,
        arc_mixture_responsibility=arc_mixture_responsibility,
        inlier_responsibility=inlier_responsibility,
        outlier_responsibility=outlier_responsibility,
    )


def _json_arcsec_array(values: Any) -> str:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        array = np.asarray([], dtype=float)
    finite_values = [float(value) for value in array if np.isfinite(value)]
    return json.dumps(finite_values, separators=(",", ":"))


def _critical_arc_frame_numpy_from_jacobian_entries(
    jac_a00: float,
    jac_a01: float,
    jac_a10: float,
    jac_a11: float,
) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
    (
        critical_x,
        critical_y,
        _noncritical_x,
        _noncritical_y,
        singular_min,
        singular_max,
        finite,
    ) = _critical_arc_jacobian_frame(
        jnp.asarray([jac_a00], dtype=jnp.float64),
        jnp.asarray([jac_a01], dtype=jnp.float64),
        jnp.asarray([jac_a10], dtype=jnp.float64),
        jnp.asarray([jac_a11], dtype=jnp.float64),
    )
    direction = np.asarray([float(np.asarray(critical_x)[0]), float(np.asarray(critical_y)[0])], dtype=float)
    norm = float(np.linalg.norm(direction))
    frame_finite = bool(np.asarray(finite)[0]) and np.isfinite(norm) and norm > 0.0
    if frame_finite:
        direction = direction / norm
    return (
        direction,
        np.asarray([float(np.asarray(_noncritical_x)[0]), float(np.asarray(_noncritical_y)[0])], dtype=float),
        float(np.asarray(singular_min)[0]),
        float(np.asarray(singular_max)[0]),
        frame_finite,
    )


def _critical_arc_direction_at_point(
    jacobian_at: Callable[[np.ndarray, np.ndarray], tuple[Any, Any, Any, Any]],
    x_value: float,
    y_value: float,
    previous_direction: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    try:
        entries = jacobian_at(
            np.asarray([float(x_value)], dtype=float),
            np.asarray([float(y_value)], dtype=float),
        )
        a00, a01, a10, a11 = (float(np.asarray(entry, dtype=float).reshape(-1)[0]) for entry in entries)
    except Exception:
        return np.asarray([np.nan, np.nan], dtype=float), False
    direction, _noncritical, _s_min, _s_max, finite = _critical_arc_frame_numpy_from_jacobian_entries(
        a00,
        a01,
        a10,
        a11,
    )
    if not finite or not np.isfinite(direction).all():
        return np.asarray([np.nan, np.nan], dtype=float), False
    if previous_direction is not None and np.isfinite(previous_direction).all():
        if float(np.dot(direction, previous_direction)) < 0.0:
            direction = -direction
    return direction, True


def _trace_critical_arc_direction_branch(
    start_x: float,
    start_y: float,
    initial_direction: np.ndarray,
    jacobian_at: Callable[[np.ndarray, np.ndarray], tuple[Any, Any, Any, Any]],
    *,
    max_arclength_arcsec: float = DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC,
    curve_step_arcsec: float = DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC,
) -> tuple[np.ndarray, bool]:
    max_arclength = float(max_arclength_arcsec)
    if not np.isfinite(max_arclength) or max_arclength <= 0.0:
        return np.asarray([[float(start_x), float(start_y)]], dtype=float), False
    requested_step = float(curve_step_arcsec)
    if not np.isfinite(requested_step) or requested_step <= 0.0:
        requested_step = DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC
    n_steps = max(1, int(math.ceil(max_arclength / requested_step)))
    step = max_arclength / float(n_steps)
    points = [np.asarray([float(start_x), float(start_y)], dtype=float)]
    direction = np.asarray(initial_direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        return np.asarray(points, dtype=float), False
    direction = direction / norm
    finite = True
    current = points[0].copy()
    for _step_index in range(n_steps):
        current_direction, current_finite = _critical_arc_direction_at_point(
            jacobian_at,
            float(current[0]),
            float(current[1]),
            direction,
        )
        if not current_finite:
            finite = False
            break
        midpoint = current + 0.5 * step * current_direction
        midpoint_direction, midpoint_finite = _critical_arc_direction_at_point(
            jacobian_at,
            float(midpoint[0]),
            float(midpoint[1]),
            current_direction,
        )
        if not midpoint_finite:
            finite = False
            break
        current = current + step * midpoint_direction
        if not np.isfinite(current).all():
            finite = False
            break
        points.append(current.copy())
        direction = midpoint_direction
    return np.asarray(points, dtype=float), finite


def _trace_arc_support_curve_from_anchor(
    anchor_x: float,
    anchor_y: float,
    jacobian_at: Callable[[np.ndarray, np.ndarray], tuple[Any, Any, Any, Any]],
    *,
    max_arclength_arcsec: float = DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC,
    curve_step_arcsec: float = DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC,
) -> tuple[np.ndarray, np.ndarray, bool]:
    base_direction, base_finite = _critical_arc_direction_at_point(
        jacobian_at,
        float(anchor_x),
        float(anchor_y),
        None,
    )
    if not base_finite:
        empty = np.asarray([float(anchor_x)], dtype=float)
        return empty, np.asarray([float(anchor_y)], dtype=float), False
    positive, positive_finite = _trace_critical_arc_direction_branch(
        anchor_x,
        anchor_y,
        base_direction,
        jacobian_at,
        max_arclength_arcsec=max_arclength_arcsec,
        curve_step_arcsec=curve_step_arcsec,
    )
    negative, negative_finite = _trace_critical_arc_direction_branch(
        anchor_x,
        anchor_y,
        -base_direction,
        jacobian_at,
        max_arclength_arcsec=max_arclength_arcsec,
        curve_step_arcsec=curve_step_arcsec,
    )
    if negative.shape[0] > 1:
        points = np.vstack([negative[:0:-1], positive])
    else:
        points = positive
    finite = bool(base_finite and positive_finite and negative_finite and points.shape[0] >= 2)
    return points[:, 0], points[:, 1], finite


def _polyline_distance_and_anchor_arclength(
    point_x: float,
    point_y: float,
    curve_x: Any,
    curve_y: Any,
    anchor_x: float,
    anchor_y: float,
) -> tuple[float, float, bool]:
    try:
        x_values = np.asarray(curve_x, dtype=float).reshape(-1)
        y_values = np.asarray(curve_y, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.nan, np.nan, False
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if x_values.size == 0 or x_values.shape != y_values.shape:
        return np.nan, np.nan, False
    points = np.column_stack([x_values, y_values])
    point = np.asarray([float(point_x), float(point_y)], dtype=float)
    anchor = np.asarray([float(anchor_x), float(anchor_y)], dtype=float)
    if not np.isfinite(point).all() or not np.isfinite(anchor).all():
        return np.nan, np.nan, False
    if points.shape[0] == 1:
        return float(np.linalg.norm(point - points[0])), 0.0, True
    segment = points[1:] - points[:-1]
    segment_len2 = np.sum(np.square(segment), axis=1)
    valid = segment_len2 > 0.0
    if not np.any(valid):
        return float(np.min(np.linalg.norm(points - point, axis=1))), 0.0, True
    start = points[:-1][valid]
    direction = segment[valid]
    length2 = segment_len2[valid]
    t = np.clip(np.sum((point - start) * direction, axis=1) / length2, 0.0, 1.0)
    closest = start + t[:, None] * direction
    distances = np.linalg.norm(closest - point, axis=1)
    anchor_distances = np.linalg.norm(points - anchor, axis=1)
    anchor_index = int(np.argmin(anchor_distances))
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(segment, axis=1))])
    valid_indices = np.flatnonzero(valid)
    anchor_s = float(cumulative[anchor_index])
    closest_s_values = cumulative[valid_indices] + t * np.sqrt(segment_len2[valid_indices])
    arclength_values = np.abs(closest_s_values - anchor_s)
    min_distance = float(np.min(distances))
    distance_tolerance = max(1.0e-9, 1.0e-6 * max(1.0, min_distance))
    candidates = np.flatnonzero(distances <= min_distance + distance_tolerance)
    if candidates.size:
        best = int(candidates[np.argmin(arclength_values[candidates])])
    else:
        best = int(np.argmin(distances))
    return float(distances[best]), float(arclength_values[best]), True


def _critical_arc_mixture_image_plane_bin_loglike(
    residual_x: jnp.ndarray,
    residual_y: jnp.ndarray,
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
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
    residual_loss: str = DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS,
    student_t_nu: float = DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU,
    critical_direction_sigma_arcsec: float = DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC,
    base_prob: float = DEFAULT_CRITICAL_ARC_BASE_PROB,
    max_prob: float = DEFAULT_CRITICAL_ARC_MAX_PROB,
    singular_threshold: float = DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD,
    singular_softness: float = DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS,
    singular_min_precomputed: jnp.ndarray | None = None,
    singular_max_precomputed: jnp.ndarray | None = None,
    critical_direction_projector_entries: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    if (
        singular_min_precomputed is None
        or singular_max_precomputed is None
        or critical_direction_projector_entries is None
    ):
        (
            singular_min,
            singular_max,
            critical_p00,
            critical_p01,
            critical_p11,
            frame_finite,
        ) = _critical_arc_geometry_from_jacobian(
            jac_a00,
            jac_a01,
            jac_a10,
            jac_a11,
        )
    else:
        singular_min = singular_min_precomputed
        singular_max = singular_max_precomputed
        critical_p00, critical_p01, critical_p11 = critical_direction_projector_entries
        frame_finite = (
            jnp.isfinite(singular_min)
            & jnp.isfinite(singular_max)
            & jnp.isfinite(critical_p00)
            & jnp.isfinite(critical_p01)
            & jnp.isfinite(critical_p11)
        )
    terms = _critical_arc_mixture_image_plane_terms(
        residual_x=residual_x,
        residual_y=residual_y,
        sigma_per_image=sigma_per_image,
        reliability_per_image=reliability_per_image,
        image_sigma_int=image_sigma_int,
        covariance_floor=covariance_floor,
        outlier_sigma_arcsec=outlier_sigma_arcsec,
        singular_min=singular_min,
        critical_direction_projector_entries=(critical_p00, critical_p01, critical_p11),
        residual_loss=residual_loss,
        student_t_nu=student_t_nu,
        critical_direction_sigma_arcsec=critical_direction_sigma_arcsec,
        base_prob=base_prob,
        max_prob=max_prob,
        singular_threshold=singular_threshold,
        singular_softness=singular_softness,
    )
    bin_loglike = jnp.sum(jnp.where(image_has_constraint, terms.mixture_ll, 0.0))
    if (
        float(image_presence_penalty_weight) > 0.0
        and family_idx is not None
        and n_families is not None
    ):
        point_residual2 = jnp.square(residual_x) + jnp.square(residual_y)
        radius2 = jnp.square(jnp.asarray(image_presence_match_radius_arcsec, dtype=jnp.float64))
        temperature2 = jnp.maximum(
            jnp.square(jnp.asarray(image_presence_temperature_arcsec, dtype=jnp.float64)),
            jnp.asarray(1.0e-18, dtype=jnp.float64),
        )
        point_presence_probability = jax.nn.sigmoid((radius2 - point_residual2) / temperature2)
        arc_presence_probability = jax.nn.sigmoid(
            (radius2 - terms.noncritical_quad) / temperature2
        )
        probability_span = jnp.asarray(float(max_prob) - float(base_prob), dtype=jnp.float64)
        probability_span_safe = jnp.maximum(probability_span, jnp.asarray(1.0e-12, dtype=jnp.float64))
        normalized_arc_gate = (
            terms.arc_prob - jnp.asarray(float(base_prob), dtype=jnp.float64)
        ) / probability_span_safe
        arc_gate = jnp.where(probability_span > 1.0e-12, normalized_arc_gate, terms.arc_prob)
        arc_gate = jnp.clip(arc_gate, 0.0, 1.0)
        presence_probability = point_presence_probability + arc_gate * (
            arc_presence_probability - point_presence_probability
        )
        bin_loglike = bin_loglike + _soft_observed_image_presence_loglike_from_probability(
            presence_probability=presence_probability,
            family_idx=family_idx,
            n_families=int(n_families),
            reliability_per_image=terms.reliability,
            image_has_constraint=image_has_constraint,
            penalty_weight=float(image_presence_penalty_weight),
            count_softness=float(image_presence_count_softness),
            count_margin=float(image_presence_count_margin),
        )
    finite = (
        jnp.all(frame_finite)
        & jnp.all(jnp.isfinite(residual_x))
        & jnp.all(jnp.isfinite(residual_y))
        & jnp.all(jnp.isfinite(terms.sigma2))
        & jnp.all(jnp.isfinite(singular_min))
        & jnp.all(jnp.isfinite(singular_max))
        & jnp.isfinite(bin_loglike)
    )
    return jnp.where(finite, bin_loglike, jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64))


def _critical_arc_debug_terms_for_state(
    state: BuildState,
    evaluator: Any,
    theta: np.ndarray,
    *,
    state_index: int,
    state_label: str,
    chain: int,
    draw: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if str(getattr(evaluator, "sample_likelihood_mode", "")) != SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE:
        return [], []
    params = jnp.asarray(theta, dtype=jnp.float64)
    physical_params = evaluator._physical_parameter_vector(params)
    image_sigma_int = evaluator._image_sigma_int_from_physical(physical_params)
    sampled_kpc_per_arcsec = None
    sampled_dpie_sigma0_factors = None
    if bool(getattr(evaluator, "fit_cosmology_flat_wcdm", False)):
        sampled_kpc_per_arcsec, sampled_dpie_sigma0_factors = evaluator._sampled_cosmology_geometry_for_physical(
            physical_params
        )

    image_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    global_image_index = 0
    for bin_index, bin_data in enumerate(evaluator.traced_bin_data):
        x_obs = bin_data.x_obs
        y_obs = bin_data.y_obs
        bin_kpc_per_arcsec = sampled_kpc_per_arcsec
        bin_dpie_sigma0_factor = None
        if sampled_dpie_sigma0_factors is not None and int(getattr(bin_data, "effective_z_index", -1)) >= 0:
            bin_dpie_sigma0_factor = jnp.take(
                sampled_dpie_sigma0_factors,
                jnp.asarray(int(bin_data.effective_z_index), dtype=jnp.int32),
            )
        if evaluator.surrogate_enabled and evaluator.surrogate_cache_by_z:
            beta_x, beta_y, invalid, packed_state = evaluator._surrogate_beta(
                params,
                physical_params,
                bin_data,
                kpc_per_arcsec=bin_kpc_per_arcsec,
                dpie_sigma0_factor=bin_dpie_sigma0_factor,
            )
            observed_jacobian_entries = evaluator._surrogate_jacobian_entries(
                params,
                bin_data,
                packed_state,
                invalid,
            )
        else:
            if bin_dpie_sigma0_factor is None:
                packed_state, validity = evaluator._build_packed_lens_state_with_validity_from_physical(
                    physical_params,
                    bin_data.effective_z_source,
                    stop_gradient=True,
                )
            else:
                packed_state, validity = evaluator._build_packed_lens_state_with_validity_from_physical(
                    physical_params,
                    bin_data.effective_z_source,
                    stop_gradient=True,
                    kpc_per_arcsec=bin_kpc_per_arcsec,
                    dpie_sigma0_factor=bin_dpie_sigma0_factor,
                )
            invalid = ~validity["is_valid"]
            beta_x, beta_y = jax.lax.cond(
                invalid,
                lambda _: (x_obs, y_obs),
                lambda current_state: evaluator._ray_shooting_for_components(
                    bin_data.effective_z_source,
                    x_obs,
                    y_obs,
                    current_state,
                ),
                packed_state,
            )
            observed_jacobian_entries = evaluator._lensing_jacobian_for_components(
                bin_data.effective_z_source,
                x_obs,
                y_obs,
                packed_state,
            )
        beta_family_x, beta_family_y, has_source_positions, source_transport_correction = evaluator._explicit_source_position_vectors_for_bin(
            params,
            physical_params,
            bin_data,
            beta_x,
            beta_y,
            image_sigma_int,
            observed_jacobian_entries,
        )
        (
            residual_x,
            residual_y,
            singular_min,
            singular_max,
            critical_p00,
            critical_p01,
            critical_p11,
            residual_finite,
        ) = _critical_arc_lm_geometry_from_jacobian(
            beta_x - beta_family_x,
            beta_y - beta_family_y,
            *observed_jacobian_entries,
            trust_radius_arcsec=evaluator.critical_arc_lm_trust_radius_arcsec,
            lm_damping_relative=evaluator.critical_arc_lm_damping_relative,
            lm_damping_absolute=evaluator.critical_arc_lm_damping_absolute,
        )
        invalid = invalid | (~has_source_positions) | (~jnp.all(residual_finite))
        terms = _critical_arc_mixture_image_plane_terms(
            residual_x=residual_x,
            residual_y=residual_y,
            sigma_per_image=bin_data.sigma_per_image,
            reliability_per_image=bin_data.reliability_per_image,
            image_sigma_int=image_sigma_int,
            covariance_floor=evaluator.source_plane_covariance_floor,
            outlier_sigma_arcsec=evaluator.source_plane_outlier_sigma_arcsec,
            singular_min=singular_min,
            critical_direction_projector_entries=(critical_p00, critical_p01, critical_p11),
            residual_loss=evaluator.likelihood_stabilizer_residual_loss,
            student_t_nu=evaluator.likelihood_stabilizer_student_t_nu,
            critical_direction_sigma_arcsec=evaluator.critical_arc_critical_direction_sigma_arcsec,
            base_prob=evaluator.critical_arc_base_prob,
            max_prob=evaluator.critical_arc_max_prob,
            singular_threshold=evaluator.critical_arc_singular_threshold,
            singular_softness=evaluator.critical_arc_singular_softness,
        )
        responsibilities = _critical_arc_mixture_image_plane_responsibilities(terms)
        image_contribution = jnp.where(bin_data.image_has_constraint, terms.mixture_ll, 0.0)
        mixture_sum = jnp.sum(image_contribution)
        bin_loglike_without_transport = _critical_arc_mixture_image_plane_bin_loglike(
            residual_x=residual_x,
            residual_y=residual_y,
            jac_a00=observed_jacobian_entries[0],
            jac_a01=observed_jacobian_entries[1],
            jac_a10=observed_jacobian_entries[2],
            jac_a11=observed_jacobian_entries[3],
            family_idx=bin_data.family_index_per_image,
            n_families=bin_data.n_families,
            sigma_per_image=bin_data.sigma_per_image,
            reliability_per_image=bin_data.reliability_per_image,
            image_has_constraint=bin_data.image_has_constraint,
            image_sigma_int=image_sigma_int,
            covariance_floor=evaluator.source_plane_covariance_floor,
            outlier_sigma_arcsec=evaluator.source_plane_outlier_sigma_arcsec,
            image_presence_penalty_weight=evaluator.image_presence_penalty_weight,
            image_presence_match_radius_arcsec=evaluator.image_presence_match_radius_arcsec,
            image_presence_temperature_arcsec=evaluator.image_presence_temperature_arcsec,
            image_presence_count_softness=evaluator.image_presence_count_softness,
            image_presence_count_margin=evaluator.image_presence_count_margin,
            residual_loss=evaluator.likelihood_stabilizer_residual_loss,
            student_t_nu=evaluator.likelihood_stabilizer_student_t_nu,
            critical_direction_sigma_arcsec=evaluator.critical_arc_critical_direction_sigma_arcsec,
            base_prob=evaluator.critical_arc_base_prob,
            max_prob=evaluator.critical_arc_max_prob,
            singular_threshold=evaluator.critical_arc_singular_threshold,
            singular_softness=evaluator.critical_arc_singular_softness,
            singular_min_precomputed=singular_min,
            singular_max_precomputed=singular_max,
            critical_direction_projector_entries=(critical_p00, critical_p01, critical_p11),
        )
        bin_total = bin_loglike_without_transport + source_transport_correction
        presence_penalty = bin_loglike_without_transport - mixture_sum

        def np1(value: Any, dtype: type = float) -> np.ndarray:
            return np.asarray(value, dtype=dtype).reshape(-1)

        family_idx_np = np1(bin_data.family_index_per_image, dtype=int)
        family_ids = [bin_data.family_ids[int(idx)] if 0 <= int(idx) < len(bin_data.family_ids) else "" for idx in family_idx_np]
        residual_x_np = np1(residual_x)
        residual_y_np = np1(residual_y)
        det_a_np = np1(observed_jacobian_entries[0] * observed_jacobian_entries[3] - observed_jacobian_entries[1] * observed_jacobian_entries[2])
        image_has_np = np1(bin_data.image_has_constraint, dtype=bool)
        for local_image_index in range(residual_x_np.size):
            image_rows.append(
                {
                    "state_index": state_index,
                    "state_label": state_label,
                    "chain": int(chain),
                    "draw": int(draw),
                    "bin_index": int(bin_index),
                    "effective_z_source": float(bin_data.effective_z_source),
                    "image_global_index": int(global_image_index + local_image_index),
                    "image_index": int(local_image_index),
                    "family_id": family_ids[local_image_index],
                    "image_has_constraint": bool(image_has_np[local_image_index]),
                    "x_obs": float(np1(x_obs)[local_image_index]),
                    "y_obs": float(np1(y_obs)[local_image_index]),
                    "residual_x": float(residual_x_np[local_image_index]),
                    "residual_y": float(residual_y_np[local_image_index]),
                    "residual_norm": float(np.hypot(residual_x_np[local_image_index], residual_y_np[local_image_index])),
                    "sigma_eff": float(np.sqrt(np1(terms.sigma2)[local_image_index])),
                    "image_sigma_int": float(np.asarray(image_sigma_int, dtype=float)),
                    "image_sigma_int_sampled": bool(getattr(evaluator, "image_sigma_int_sampled", False)),
                    "fixed_image_sigma_int_arcsec": (
                        float(getattr(evaluator, "fixed_image_sigma_int_arcsec"))
                        if getattr(evaluator, "fixed_image_sigma_int_arcsec", None) is not None
                        else float("nan")
                    ),
                    "reliability": float(np1(terms.reliability)[local_image_index]),
                    "det_a": float(det_a_np[local_image_index]),
                    "singular_min": float(np1(singular_min)[local_image_index]),
                    "singular_max": float(np1(singular_max)[local_image_index]),
                    "arc_probability": float(np1(terms.arc_prob)[local_image_index]),
                    "critical_direction_residual": float(np.sqrt(np1(terms.critical_quad)[local_image_index])),
                    "noncritical_direction_residual": float(np.sqrt(np1(terms.noncritical_quad)[local_image_index])),
                    "point_loglike": float(np1(terms.point_ll)[local_image_index]),
                    "arc_loglike": float(np1(terms.arc_ll)[local_image_index]),
                    "outlier_loglike": float(np1(terms.outlier_ll)[local_image_index]),
                    "inlier_loglike": float(np1(terms.inlier_ll)[local_image_index]),
                    "mixture_loglike": float(np1(terms.mixture_ll)[local_image_index]),
                    "inlier_responsibility": float(np1(responsibilities.inlier_responsibility)[local_image_index]),
                    "outlier_responsibility": float(np1(responsibilities.outlier_responsibility)[local_image_index]),
                    "point_inlier_responsibility": float(np1(responsibilities.point_inlier_responsibility)[local_image_index]),
                    "arc_inlier_responsibility": float(np1(responsibilities.arc_inlier_responsibility)[local_image_index]),
                    "point_mixture_responsibility": float(np1(responsibilities.point_mixture_responsibility)[local_image_index]),
                    "arc_mixture_responsibility": float(np1(responsibilities.arc_mixture_responsibility)[local_image_index]),
                    "inlier_log_weight": float(np1(responsibilities.inlier_log_weight)[local_image_index]),
                    "outlier_log_weight": float(np1(responsibilities.outlier_log_weight)[local_image_index]),
                    "point_log_weight": float(np1(responsibilities.point_log_weight)[local_image_index]),
                    "arc_log_weight": float(np1(responsibilities.arc_log_weight)[local_image_index]),
                    "outlier_margin_log_weight_minus_inlier": float(
                        np1(responsibilities.outlier_log_weight - responsibilities.inlier_log_weight)[local_image_index]
                    ),
                    "arc_margin_log_weight_minus_point": float(
                        np1(responsibilities.arc_log_weight - responsibilities.point_log_weight)[local_image_index]
                    ),
                    "final_image_mixture_contribution": float(np1(image_contribution)[local_image_index]),
                    "branch_margin_arc_minus_point": float(np1(terms.arc_ll - terms.point_ll)[local_image_index]),
                    "residual_finite": bool(np1(residual_finite, dtype=bool)[local_image_index]),
                }
            )
        finite_contrib = np1(image_contribution)
        outlier_resp_np = np1(responsibilities.outlier_responsibility)
        worst_indices = np.argsort(finite_contrib)[: min(5, finite_contrib.size)] if finite_contrib.size else np.asarray([], dtype=int)
        constrained_indices = np.where(image_has_np)[0]
        finite_outlier_indices = constrained_indices[np.isfinite(outlier_resp_np[constrained_indices])] if constrained_indices.size else np.asarray([], dtype=int)
        outlier_resp_constrained = outlier_resp_np[finite_outlier_indices] if finite_outlier_indices.size else np.asarray([], dtype=float)
        worst_outlier_order = (
            finite_outlier_indices[np.argsort(outlier_resp_np[finite_outlier_indices])[::-1]]
            if finite_outlier_indices.size
            else np.asarray([], dtype=int)
        )
        worst_outlier_indices = worst_outlier_order[: min(5, worst_outlier_order.size)]
        bin_rows.append(
            {
                "state_index": state_index,
                "state_label": state_label,
                "chain": int(chain),
                "draw": int(draw),
                "bin_index": int(bin_index),
                "effective_z_source": float(bin_data.effective_z_source),
                "n_images": int(residual_x_np.size),
                "invalid": bool(np.asarray(invalid, dtype=bool)),
                "has_source_positions": bool(np.asarray(has_source_positions, dtype=bool)),
                "all_residual_finite": bool(np.asarray(jnp.all(residual_finite), dtype=bool)),
                "image_sigma_int": float(np.asarray(image_sigma_int, dtype=float)),
                "image_sigma_int_sampled": bool(getattr(evaluator, "image_sigma_int_sampled", False)),
                "fixed_image_sigma_int_arcsec": (
                    float(getattr(evaluator, "fixed_image_sigma_int_arcsec"))
                    if getattr(evaluator, "fixed_image_sigma_int_arcsec", None) is not None
                    else float("nan")
                ),
                "image_mixture_sum": float(mixture_sum),
                "source_transport_correction": float(source_transport_correction),
                "presence_penalty": float(presence_penalty),
                "bin_loglike_without_transport": float(bin_loglike_without_transport),
                "bin_loglike_with_transport": float(bin_total),
                "outlier_responsibility_mean": float(np.mean(outlier_resp_constrained)) if outlier_resp_constrained.size else float("nan"),
                "outlier_responsibility_max": float(np.max(outlier_resp_constrained)) if outlier_resp_constrained.size else float("nan"),
                "outlier_responsibility_sum": float(np.sum(outlier_resp_constrained)) if outlier_resp_constrained.size else 0.0,
                "outlier_responsibility_count_gt_0p1": int(np.sum(outlier_resp_constrained > 0.1)),
                "outlier_responsibility_count_gt_0p5": int(np.sum(outlier_resp_constrained > 0.5)),
                "worst_image_indices": json.dumps([int(idx) for idx in worst_indices]),
                "worst_image_contributions": json.dumps([float(finite_contrib[int(idx)]) for idx in worst_indices]),
                "worst_outlier_responsibility_image_indices": json.dumps([int(idx) for idx in worst_outlier_indices]),
                "worst_outlier_responsibilities": json.dumps(
                    [float(outlier_resp_np[int(idx)]) for idx in worst_outlier_indices]
                ),
            }
        )
        global_image_index += residual_x_np.size
    return image_rows, bin_rows


def _arc_aware_image_support_from_local_linearization(
    beta_residual_x: Any,
    beta_residual_y: Any,
    jac_a00: Any,
    jac_a01: Any,
    jac_a10: Any,
    jac_a11: Any,
    *,
    theta_obs_x: Any | None = None,
    theta_obs_y: Any | None = None,
    jacobian_at: Callable[[np.ndarray, np.ndarray], tuple[Any, Any, Any, Any]] | None = None,
    point_recovered_mask: Any | None = None,
    point_residual_arcsec: Any | None = None,
    noncritical_support_radius_arcsec: float = DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC,
    max_arclength_arcsec: float = DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC,
    curve_step_arcsec: float = DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC,
    trust_radius_arcsec: float = DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC,
    lm_damping_relative: float = DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE,
    lm_damping_absolute: float = DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE,
    base_prob: float = DEFAULT_CRITICAL_ARC_BASE_PROB,
    max_prob: float = DEFAULT_CRITICAL_ARC_MAX_PROB,
    singular_threshold: float = DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD,
    singular_softness: float = DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS,
) -> dict[str, Any]:
    f_x = jnp.asarray(beta_residual_x, dtype=jnp.float64)
    f_y = jnp.asarray(beta_residual_y, dtype=jnp.float64)
    a00 = jnp.asarray(jac_a00, dtype=jnp.float64)
    a01 = jnp.asarray(jac_a01, dtype=jnp.float64)
    a10 = jnp.asarray(jac_a10, dtype=jnp.float64)
    a11 = jnp.asarray(jac_a11, dtype=jnp.float64)
    shape = np.asarray(f_x).reshape(-1).shape
    point_recovered = (
        np.asarray(point_recovered_mask, dtype=bool).reshape(-1)
        if point_recovered_mask is not None
        else np.zeros(shape, dtype=bool)
    )
    if point_recovered.shape != shape:
        point_recovered = np.zeros(shape, dtype=bool)
    point_residual = (
        np.asarray(point_residual_arcsec, dtype=float).reshape(-1)
        if point_residual_arcsec is not None
        else np.full(shape, np.nan, dtype=float)
    )
    if point_residual.shape != shape:
        point_residual = np.full(shape, np.nan, dtype=float)
    point_valid = point_recovered & np.isfinite(point_residual)
    point_residual = np.where(point_valid, point_residual, np.nan)
    obs_x = (
        np.asarray(theta_obs_x, dtype=float).reshape(-1)
        if theta_obs_x is not None
        else np.full(shape, np.nan, dtype=float)
    )
    obs_y = (
        np.asarray(theta_obs_y, dtype=float).reshape(-1)
        if theta_obs_y is not None
        else np.full(shape, np.nan, dtype=float)
    )
    if obs_x.shape != shape:
        obs_x = np.full(shape, np.nan, dtype=float)
    if obs_y.shape != shape:
        obs_y = np.full(shape, np.nan, dtype=float)

    delta_x, delta_y, step_finite = _critical_arc_lm_step_from_jacobian(
        f_x,
        f_y,
        a00,
        a01,
        a10,
        a11,
        trust_radius_arcsec=trust_radius_arcsec,
        lm_damping_relative=lm_damping_relative,
        lm_damping_absolute=lm_damping_absolute,
    )
    critical_direction_x, critical_direction_y, noncritical_direction_x, noncritical_direction_y, singular_min, singular_max, frame_finite = _critical_arc_jacobian_frame(
        a00,
        a01,
        a10,
        a11,
    )
    arc_prior = _critical_arc_branch_probability(
        singular_min,
        base_prob=base_prob,
        max_prob=max_prob,
        singular_threshold=singular_threshold,
        singular_softness=singular_softness,
    )
    delta_critical_direction = delta_x * critical_direction_x + delta_y * critical_direction_y
    delta_noncritical_direction = delta_x * noncritical_direction_x + delta_y * noncritical_direction_y
    noncritical_direction_residual = np.abs(np.asarray(delta_noncritical_direction, dtype=float).reshape(-1))
    critical_direction_residual = np.abs(np.asarray(delta_critical_direction, dtype=float).reshape(-1))
    critical_direction_x_np = np.asarray(critical_direction_x, dtype=float).reshape(-1)
    critical_direction_y_np = np.asarray(critical_direction_y, dtype=float).reshape(-1)
    noncritical_direction_x_np = np.asarray(noncritical_direction_x, dtype=float).reshape(-1)
    noncritical_direction_y_np = np.asarray(noncritical_direction_y, dtype=float).reshape(-1)
    s_min = np.asarray(singular_min, dtype=float).reshape(-1)
    s_max = np.asarray(singular_max, dtype=float).reshape(-1)
    det_a = np.asarray(a00 * a11 - a01 * a10, dtype=float).reshape(-1)
    arc_prior_np = np.asarray(arc_prior, dtype=float).reshape(-1)
    finite = (
        np.asarray(step_finite & frame_finite, dtype=bool).reshape(-1)
        & np.isfinite(noncritical_direction_residual)
        & np.isfinite(critical_direction_residual)
        & np.isfinite(critical_direction_x_np)
        & np.isfinite(critical_direction_y_np)
        & np.isfinite(noncritical_direction_x_np)
        & np.isfinite(noncritical_direction_y_np)
        & np.isfinite(s_min)
        & np.isfinite(s_max)
        & np.isfinite(det_a)
        & np.isfinite(arc_prior_np)
    )
    critical_probability = 0.5 * (float(base_prob) + float(max_prob))
    delta_x_np = np.asarray(delta_x, dtype=float).reshape(-1)
    delta_y_np = np.asarray(delta_y, dtype=float).reshape(-1)
    support_anchor_x = obs_x + delta_x_np
    support_anchor_y = obs_y + delta_y_np
    curve_distance = np.full(shape, np.nan, dtype=float)
    curve_arclength = np.full(shape, np.nan, dtype=float)
    curve_finite = np.zeros(shape, dtype=bool)
    curve_x_json = np.asarray([_json_arcsec_array([]) for _ in range(int(np.prod(shape)))], dtype=object).reshape(shape)
    curve_y_json = np.asarray([_json_arcsec_array([]) for _ in range(int(np.prod(shape)))], dtype=object).reshape(shape)
    if jacobian_at is not None:
        for index in range(shape[0]):
            if not (
                finite[index]
                and np.isfinite(obs_x[index])
                and np.isfinite(obs_y[index])
                and np.isfinite(support_anchor_x[index])
                and np.isfinite(support_anchor_y[index])
            ):
                continue
            curve_x, curve_y, trace_finite = _trace_arc_support_curve_from_anchor(
                float(support_anchor_x[index]),
                float(support_anchor_y[index]),
                jacobian_at,
                max_arclength_arcsec=max_arclength_arcsec,
                curve_step_arcsec=curve_step_arcsec,
            )
            distance, arclength, distance_finite = _polyline_distance_and_anchor_arclength(
                float(obs_x[index]),
                float(obs_y[index]),
                curve_x,
                curve_y,
                float(support_anchor_x[index]),
                float(support_anchor_y[index]),
            )
            curve_ok = bool(trace_finite and distance_finite and np.isfinite(distance) and np.isfinite(arclength))
            curve_finite[index] = curve_ok
            if curve_ok:
                curve_distance[index] = float(distance)
                curve_arclength[index] = float(arclength)
                curve_x_json[index] = _json_arcsec_array(curve_x)
                curve_y_json[index] = _json_arcsec_array(curve_y)
    arc_candidate_supported = (
        finite
        & curve_finite
        & (arc_prior_np >= critical_probability)
        & (curve_distance <= float(noncritical_support_radius_arcsec))
    )
    arc_candidate_residual = np.where(arc_candidate_supported, curve_distance, np.nan)
    arc_supported = (~point_valid) & arc_candidate_supported & np.isfinite(arc_candidate_residual)
    status = np.where(
        point_valid,
        "point_recovered",
        np.where(arc_supported, "arc_supported", "not_recovered"),
    )
    arc_aware_residual = np.where(
        point_valid,
        point_residual,
        np.where(arc_supported, arc_candidate_residual, np.nan),
    )
    supported_or_recovered = np.isfinite(arc_aware_residual)
    arc_aware_rms = (
        float(np.sqrt(np.mean(np.square(arc_aware_residual[supported_or_recovered]))))
        if np.any(supported_or_recovered)
        else np.nan
    )
    return {
        "point_image_residual_arcsec": point_residual.astype(float),
        "arc_candidate_supported": arc_candidate_supported.astype(bool),
        "arc_candidate_image_residual_arcsec": arc_candidate_residual.astype(float),
        "preferred_recovery_status": status.astype(object),
        "preferred_image_residual_arcsec": arc_aware_residual.astype(float),
        "arc_recovery_status": status.astype(object),
        "arc_aware_image_residual_arcsec": arc_aware_residual.astype(float),
        "arc_noncritical_direction_residual_arcsec": noncritical_direction_residual.astype(float),
        "arc_critical_direction_residual_arcsec": critical_direction_residual.astype(float),
        "arc_critical_direction_x": critical_direction_x_np.astype(float),
        "arc_critical_direction_y": critical_direction_y_np.astype(float),
        "arc_noncritical_direction_x": noncritical_direction_x_np.astype(float),
        "arc_noncritical_direction_y": noncritical_direction_y_np.astype(float),
        "arc_s_min": s_min.astype(float),
        "arc_s_max": s_max.astype(float),
        "arc_detA": det_a.astype(float),
        "arc_prior_probability": arc_prior_np.astype(float),
        "arc_curve_distance_arcsec": curve_distance.astype(float),
        "arc_curve_arclength_arcsec": curve_arclength.astype(float),
        "arc_curve_finite": curve_finite.astype(bool),
        "arc_support_anchor_x_arcsec": support_anchor_x.astype(float),
        "arc_support_anchor_y_arcsec": support_anchor_y.astype(float),
        "arc_support_curve_x_arcsec": curve_x_json.astype(object),
        "arc_support_curve_y_arcsec": curve_y_json.astype(object),
        "arc_supported_mask": arc_supported.astype(bool),
        "arc_supported": arc_supported.astype(bool),
        "arc_support_finite_mask": (finite & curve_finite).astype(bool),
        "arc_aware_image_rms_arcsec": arc_aware_rms,
        "arc_aware_recovered_image_count": int(np.sum(supported_or_recovered)),
        "arc_aware_missing_image_count": int(max(0, len(arc_aware_residual) - int(np.sum(supported_or_recovered)))),
        "arc_supported_image_count": int(np.sum(arc_supported)),
        "arc_candidate_supported_image_count": int(np.sum(arc_candidate_supported)),
    }


def _forward_metric_image_presence_residual2(
    residual_beta_x: jnp.ndarray,
    residual_beta_y: jnp.ndarray,
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
    sigma_per_image: jnp.ndarray,
    image_sigma_int: jnp.ndarray,
    covariance_floor: float,
    *,
    max_gain: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    image_sigma2 = jnp.square(sigma_per_image) + jnp.square(image_sigma_int)
    cov_floor = jnp.asarray(covariance_floor, dtype=jnp.float64)
    c00 = image_sigma2 * (jnp.square(jac_a00) + jnp.square(jac_a01)) + cov_floor
    c11 = image_sigma2 * (jnp.square(jac_a10) + jnp.square(jac_a11)) + cov_floor
    c01 = image_sigma2 * (jac_a00 * jac_a10 + jac_a01 * jac_a11)
    if float(max_gain) > 0.0:
        gain_floor = image_sigma2 / jnp.square(jnp.asarray(float(max_gain), dtype=jnp.float64))
        c00 = c00 + gain_floor
        c11 = c11 + gain_floor
    c00 = jnp.maximum(c00, cov_floor)
    c11 = jnp.maximum(c11, cov_floor)
    c00, c11, det = _jittered_2x2_covariance_det(c00, c01, c11)
    inv00 = c11 / det
    inv11 = c00 / det
    inv01 = -c01 / det
    quad = (
        residual_beta_x * (inv00 * residual_beta_x + inv01 * residual_beta_y)
        + residual_beta_y * (inv01 * residual_beta_x + inv11 * residual_beta_y)
    )
    image_scale2 = jnp.maximum(
        image_sigma2 + cov_floor,
        jnp.maximum(cov_floor, jnp.asarray(1.0e-18, dtype=jnp.float64)),
    )
    residual2 = jnp.maximum(quad, 0.0) * image_scale2
    finite = (
        jnp.isfinite(residual_beta_x)
        & jnp.isfinite(residual_beta_y)
        & jnp.isfinite(c00)
        & jnp.isfinite(c01)
        & jnp.isfinite(c11)
        & jnp.isfinite(det)
        & jnp.isfinite(quad)
        & jnp.isfinite(residual2)
    )
    return residual2, finite


def _soft_observed_image_presence_loglike_from_probability(
    presence_probability: jnp.ndarray,
    family_idx: jnp.ndarray,
    n_families: int,
    reliability_per_image: jnp.ndarray,
    image_has_constraint: jnp.ndarray,
    *,
    penalty_weight: float,
    count_softness: float,
    count_margin: float,
) -> jnp.ndarray:
    if float(penalty_weight) <= 0.0:
        return jnp.asarray(0.0, dtype=jnp.float64)

    image_mask = jnp.asarray(image_has_constraint, dtype=bool)
    reliability = jnp.where(image_mask, jnp.clip(reliability_per_image, 0.0, 1.0), 0.0)
    presence_probability = jnp.where(
        image_mask,
        jnp.clip(jnp.asarray(presence_probability, dtype=jnp.float64), 0.0, 1.0),
        0.0,
    )

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
        jnp.all(jnp.isfinite(presence_probability))
        & jnp.all(jnp.isfinite(target_count))
        & jnp.all(jnp.isfinite(found_count))
        & jnp.isfinite(penalty)
    )
    return jnp.where(finite, penalty, jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64))


def _soft_observed_image_presence_loglike_from_residual2(
    residual2: jnp.ndarray,
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

    residual2 = jnp.maximum(jnp.asarray(residual2, dtype=jnp.float64), 0.0)
    radius2 = jnp.square(jnp.asarray(match_radius_arcsec, dtype=jnp.float64))
    temperature2 = jnp.maximum(
        jnp.square(jnp.asarray(temperature_arcsec, dtype=jnp.float64)),
        jnp.asarray(1.0e-18, dtype=jnp.float64),
    )
    presence_probability = jax.nn.sigmoid((radius2 - residual2) / temperature2)
    penalty = _soft_observed_image_presence_loglike_from_probability(
        presence_probability=presence_probability,
        family_idx=family_idx,
        n_families=n_families,
        reliability_per_image=reliability_per_image,
        image_has_constraint=image_has_constraint,
        penalty_weight=penalty_weight,
        count_softness=count_softness,
        count_margin=count_margin,
    )
    finite = jnp.all(jnp.isfinite(residual2)) & jnp.isfinite(penalty)
    return jnp.where(finite, penalty, jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64))


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
    return _soft_observed_image_presence_loglike_from_residual2(
        residual2=jnp.square(residual_x) + jnp.square(residual_y),
        family_idx=family_idx,
        n_families=n_families,
        reliability_per_image=reliability_per_image,
        image_has_constraint=image_has_constraint,
        penalty_weight=penalty_weight,
        match_radius_arcsec=match_radius_arcsec,
        temperature_arcsec=temperature_arcsec,
        count_softness=count_softness,
        count_margin=count_margin,
    )


def _image_plane_effective_sigma2(
    sigma_per_image: jnp.ndarray,
    image_sigma_int: jnp.ndarray,
    covariance_floor: float,
) -> jnp.ndarray:
    cov_floor = jnp.asarray(covariance_floor, dtype=jnp.float64)
    return jnp.maximum(
        jnp.square(sigma_per_image) + jnp.square(image_sigma_int) + cov_floor,
        jnp.maximum(cov_floor, jnp.asarray(1.0e-18, dtype=jnp.float64)),
    )


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
    residual_loss: str = DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS,
    student_t_nu: float = DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU,
) -> jnp.ndarray:
    sigma2 = _image_plane_effective_sigma2(
        sigma_per_image,
        image_sigma_int,
        covariance_floor,
    )
    reliability = jnp.clip(reliability_per_image, 1.0e-6, 1.0 - 1.0e-6)
    quad = (jnp.square(residual_x) + jnp.square(residual_y)) / sigma2
    if str(residual_loss) == LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_STUDENT_T:
        family_ll = _student_t_2d_loglike_from_quad_logdet(quad, 2.0 * jnp.log(sigma2), student_t_nu)
    else:
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


def _values_dict_to_theta(
    parameter_specs: list[ParameterSpec],
    values: dict[str, Any],
) -> np.ndarray:
    return np.asarray([float(_site_array_for_spec(values, spec)) for spec in parameter_specs], dtype=float)


def _svi_initial_value_dict(
    parameter_specs: list[ParameterSpec],
    init_values: dict[str, float] | None,
) -> dict[str, jnp.ndarray] | None:
    if not init_values:
        return None
    payload: dict[str, jnp.ndarray] = {}
    for site in _parameter_sample_sites(parameter_specs):
        if not all(parameter_specs[idx].sample_name in init_values for idx in site.indices):
            continue
        if len(site.indices) == 1:
            spec = parameter_specs[site.indices[0]]
            payload[site.name] = jnp.asarray(float(init_values[spec.sample_name]), dtype=jnp.float64)
        else:
            payload[site.name] = jnp.asarray(
                [float(init_values[parameter_specs[idx].sample_name]) for idx in site.indices],
                dtype=jnp.float64,
            )
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
        if _is_potfile_mass_size_spec(spec):
            scale = max(float(jitter_frac), 1.0e-6)
        elif spec.prior_kind in {"normal", "truncated_normal"}:
            scale = max(float(jitter_frac) * float(spec.std or 0.0), 1.0e-6)
        else:
            scale = max(float(jitter_frac) * float(spec.upper - spec.lower), 1.0e-6)
        perturbed[idx] += float(rng.normal(0.0, scale))
    return _clip_theta_to_support(perturbed, parameter_specs, boundary_frac=boundary_frac)


def _make_svi_nuts_chain_seeds(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
    center_theta: np.ndarray,
) -> list[ChainSeed]:
    rng = np.random.default_rng(None if args.seed is None else int(args.seed) + 303)
    chain_seeds: list[ChainSeed] = []
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
        chain_seeds.append(
            ChainSeed(
                values=np.asarray(seed_theta, dtype=float),
                source_label=f"svi_nuts_chain_{chain_index + 1}",
            )
        )
    return chain_seeds


def _make_direct_nuts_chain_seeds(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
    center_theta: np.ndarray,
) -> list[ChainSeed]:
    rng = np.random.default_rng(None if args.seed is None else int(args.seed) + 1303)
    chain_seeds: list[ChainSeed] = []
    for chain_index in range(int(args.chains)):
        seed_theta = _small_svi_nuts_perturbation(
            center_theta,
            parameter_specs,
            rng,
            jitter_frac=float(getattr(args, "nuts_init_jitter_frac", DEFAULT_NUTS_INIT_JITTER_FRAC)),
            boundary_frac=float(getattr(args, "nuts_init_boundary_frac", DEFAULT_NUTS_INIT_BOUNDARY_FRAC)),
        )
        if not np.all(np.isfinite(seed_theta)):
            raise ValueError(f"Direct NUTS initializer produced non-finite values for chain {chain_index + 1}.")
        chain_seeds.append(
            ChainSeed(
                values=np.asarray(seed_theta, dtype=float),
                source_label=f"direct_nuts_chain_{chain_index + 1}",
            )
        )
    return chain_seeds


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
    chain_seeds = _make_svi_nuts_chain_seeds(args, parameter_specs, center_theta)
    chain_start_labels = [f"{seed.source_label}:perturbed" for seed in chain_seeds]
    diagnostics = {
        "svi_steps": int(args.svi_steps),
        "svi_learning_rate": float(args.svi_learning_rate),
        "svi_final_elbo_loss": float(np.asarray(svi_result.losses[-1], dtype=float)) if len(svi_result.losses) else float("nan"),
        "potfile_mass_size_reparam_enabled": bool(_potfile_mass_size_group_count(parameter_specs) > 0),
        "potfile_mass_size_reparam_group_count": int(_potfile_mass_size_group_count(parameter_specs)),
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


def _smc_standard_normal_log_prob(values: jnp.ndarray) -> jnp.ndarray:
    array = jnp.asarray(values, dtype=jnp.float64)
    return jnp.sum(-0.5 * jnp.square(array) - 0.5 * jnp.log(2.0 * jnp.pi))


def _smc_normalization_arrays(parameter_specs: list[ParameterSpec]) -> dict[str, jnp.ndarray]:
    if any(spec.prior_kind == "hierarchical_normal" for spec in parameter_specs):
        raise ValueError("BlackJAX SMC does not support hierarchical_normal priors.")
    kind_code = []
    lower = []
    upper = []
    mean = []
    std = []
    for spec in parameter_specs:
        prior_kind = str(spec.prior_kind)
        if prior_kind == "uniform":
            if not (np.isfinite(float(spec.lower)) and np.isfinite(float(spec.upper))):
                raise ValueError(f"SMC uniform prior for {spec.name} requires finite bounds.")
            if float(spec.upper) <= float(spec.lower):
                raise ValueError(f"SMC uniform prior for {spec.name} requires upper > lower.")
            kind_code.append(0)
            lower.append(float(spec.lower))
            upper.append(float(spec.upper))
            mean.append(0.0)
            std.append(1.0)
        elif prior_kind in {"normal", "truncated_normal"}:
            sigma = float(spec.std or 0.0)
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(f"SMC {prior_kind} prior for {spec.name} requires positive std.")
            kind_code.append(1 if prior_kind == "normal" else 2)
            lower.append(float(spec.lower))
            upper.append(float(spec.upper))
            mean.append(float(spec.mean or 0.0))
            std.append(sigma)
        else:
            raise ValueError(f"SMC does not support prior kind {prior_kind!r} for {spec.name}.")
    return {
        "kind_code": jnp.asarray(kind_code, dtype=jnp.int32),
        "lower": jnp.asarray(lower, dtype=jnp.float64),
        "upper": jnp.asarray(upper, dtype=jnp.float64),
        "mean": jnp.asarray(mean, dtype=jnp.float64),
        "std": jnp.asarray(std, dtype=jnp.float64),
    }


def _smc_normalized_to_theta(
    normalized: jnp.ndarray,
    normalization: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    z = jnp.asarray(normalized, dtype=jnp.float64)
    kind_code = normalization["kind_code"]
    lower = normalization["lower"]
    upper = normalization["upper"]
    mean = normalization["mean"]
    std = normalization["std"]
    probability = jnp.clip(jsp_special.ndtr(z), 1.0e-12, 1.0 - 1.0e-12)
    uniform_theta = lower + (upper - lower) * probability
    normal_theta = mean + std * z
    trunc_low_probability = jsp_special.ndtr((lower - mean) / std)
    trunc_high_probability = jsp_special.ndtr((upper - mean) / std)
    trunc_probability = jnp.clip(
        trunc_low_probability + probability * (trunc_high_probability - trunc_low_probability),
        1.0e-12,
        1.0 - 1.0e-12,
    )
    truncated_theta = mean + std * jsp_special.ndtri(trunc_probability)
    theta = jnp.where(kind_code == 0, uniform_theta, normal_theta)
    theta = jnp.where(kind_code == 2, truncated_theta, theta)
    return theta


def _smc_theta_to_normalized(
    theta: jnp.ndarray,
    normalization: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    theta_array = jnp.asarray(theta, dtype=jnp.float64)
    kind_code = normalization["kind_code"]
    lower = normalization["lower"]
    upper = normalization["upper"]
    mean = normalization["mean"]
    std = normalization["std"]
    uniform_probability = jnp.clip((theta_array - lower) / (upper - lower), 1.0e-12, 1.0 - 1.0e-12)
    uniform_z = jsp_special.ndtri(uniform_probability)
    normal_z = (theta_array - mean) / std
    trunc_low_probability = jsp_special.ndtr((lower - mean) / std)
    trunc_high_probability = jsp_special.ndtr((upper - mean) / std)
    trunc_probability = jnp.clip(
        (jsp_special.ndtr(normal_z) - trunc_low_probability)
        / jnp.maximum(trunc_high_probability - trunc_low_probability, 1.0e-300),
        1.0e-12,
        1.0 - 1.0e-12,
    )
    truncated_z = jsp_special.ndtri(trunc_probability)
    z = jnp.where(kind_code == 0, uniform_z, normal_z)
    z = jnp.where(kind_code == 2, truncated_z, z)
    return z


def _smc_prior_particles(
    rng_key: jax.Array,
    parameter_specs: list[ParameterSpec],
    num_particles: int,
) -> jnp.ndarray:
    if int(num_particles) <= 0:
        raise ValueError("SMC particle count must be positive.")
    return jax.random.normal(
        rng_key,
        (int(num_particles), len(parameter_specs)),
        dtype=jnp.float64,
    )


def _smc_weighted_ess(weights: np.ndarray | jnp.ndarray) -> float:
    array = np.asarray(weights, dtype=float).reshape(-1)
    finite = np.isfinite(array)
    if array.size == 0 or not finite.any():
        return float("nan")
    safe = np.where(finite, np.maximum(array, 0.0), 0.0)
    total = float(np.sum(safe))
    if total <= 0.0:
        return float("nan")
    normalized = safe / total
    return float(1.0 / np.sum(np.square(normalized)))


def _smc_move_acceptance(info: Any) -> float:
    update_info = getattr(info, "update_info", None)
    for field in ("acceptance_rate", "is_accepted"):
        if update_info is not None and hasattr(update_info, field):
            values = np.asarray(getattr(update_info, field), dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                return float(np.mean(finite))
    return float("nan")


def _smc_fixed_surrogate_drift_summary(
    evaluator: ClusterJAXEvaluator,
    samples: np.ndarray,
) -> dict[str, Any]:
    if not bool(getattr(evaluator, "surrogate_enabled", False)):
        return {"surrogate_enabled": False}
    reference = getattr(evaluator, "surrogate_reference_params", None)
    indices = np.asarray(getattr(evaluator, "surrogate_param_indices", []), dtype=int)
    if reference is None or indices.size == 0:
        return {"surrogate_enabled": True, "surrogate_reference_available": False}
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] == 0:
        return {"surrogate_enabled": True, "surrogate_reference_available": True, "surrogate_drift_samples": 0}
    scales = np.asarray(
        [
            float(evaluator._inactive_fd_step(evaluator.state.parameter_specs[int(index)]))
            for index in indices
        ],
        dtype=float,
    )
    scales = np.where(np.isfinite(scales) & (scales > 0.0), scales, 1.0)
    drift = np.abs(sample_array[:, indices] - np.asarray(reference, dtype=float)[indices]) / scales
    finite = drift[np.isfinite(drift)]
    return {
        "surrogate_enabled": True,
        "surrogate_reference_available": True,
        "surrogate_drift_samples": int(sample_array.shape[0]),
        "surrogate_drift_param_count": int(indices.size),
        "surrogate_drift_median_fd_steps": float(np.median(finite)) if finite.size else float("nan"),
        "surrogate_drift_q95_fd_steps": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
        "surrogate_drift_max_fd_steps": float(np.max(finite)) if finite.size else float("nan"),
    }


def _blackjax_smc_components() -> tuple[Any, Any, Any, Any]:
    try:
        import blackjax
        from blackjax.mcmc import mala as blackjax_mala
        from blackjax.mcmc import random_walk as blackjax_random_walk
        from blackjax.smc import resampling as blackjax_resampling
    except ImportError as exc:
        raise RuntimeError(
            "BlackJAX SMC requires blackjax>=1.5 in the active lenstronomy environment."
        ) from exc
    version = getattr(blackjax, "__version__", None)
    if version is not None:
        try:
            major, minor, *_rest = [int(part) for part in str(version).split(".")]
        except ValueError:
            major, minor = 1, 5
        if (major, minor) < (1, 5):
            raise RuntimeError(f"BlackJAX SMC requires blackjax>=1.5; found {version}.")
    return blackjax, blackjax_resampling, blackjax_random_walk, blackjax_mala


def _run_blackjax_smc_sampler(
    args: argparse.Namespace,
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    *,
    smc_algorithm_factory: Any | None = None,
) -> PosteriorResults:
    parameter_specs = state.parameter_specs
    if not parameter_specs:
        raise RuntimeError("BlackJAX SMC requires at least one free parameter.")
    particles = int(getattr(args, "smc_particles", DEFAULT_SMC_PARTICLES))
    mcmc_steps = int(getattr(args, "smc_mcmc_steps", DEFAULT_SMC_MCMC_STEPS))
    kernel_name = str(getattr(args, "smc_mcmc_kernel", DEFAULT_SMC_MCMC_KERNEL))
    target_ess_frac = float(getattr(args, "smc_target_ess_frac", DEFAULT_SMC_TARGET_ESS_FRAC))
    max_temperature_steps = int(getattr(args, "smc_max_temperature_steps", DEFAULT_SMC_MAX_TEMPERATURE_STEPS))
    if particles <= 0 or mcmc_steps <= 0 or max_temperature_steps <= 0:
        raise ValueError("SMC particle, MCMC step, and temperature-step counts must be positive.")
    if kernel_name not in SMC_MCMC_KERNELS:
        raise ValueError(f"Unsupported SMC mutation kernel {kernel_name!r}.")
    default_device = _resolve_jax_device_for_args(args, "jax_default_device", flag_name="--jax-default-device")
    smc_device = _resolve_jax_device_for_args(args, "smc_device", flag_name="--smc-device")
    _log(
        args,
        (
            f"[smc] preparing sampler particles={particles} kernel={kernel_name} "
            f"mcmc_steps={mcmc_steps} target_ess={target_ess_frac:.3g} "
            f"max_temperature_steps={max_temperature_steps} device={_jax_device_label(smc_device)}"
        ),
    )
    target_ess = float(target_ess_frac) * float(particles)
    original_source_loglike_fn = getattr(evaluator, "_source_loglike_fn", None)
    recompiled_for_smc = False
    first_step_elapsed = float("nan")
    try:
        with _jax_device_context(smc_device):
            if smc_device is not None:
                recompiled_for_smc = _compile_evaluator_source_loglike(evaluator, device=smc_device)
            normalization = _smc_normalization_arrays(parameter_specs)

            def normalized_to_theta(position: jnp.ndarray) -> jnp.ndarray:
                return _smc_normalized_to_theta(position, normalization)

            def logprior_fn(position: jnp.ndarray) -> jnp.ndarray:
                return _smc_standard_normal_log_prob(position)

            def loglikelihood_fn(position: jnp.ndarray) -> jnp.ndarray:
                theta = normalized_to_theta(position)
                return evaluator._source_loglike_fn(theta)

            blackjax = None
            resampling = None
            random_walk = None
            mala = None
            if smc_algorithm_factory is None:
                blackjax, resampling, random_walk, mala = _blackjax_smc_components()
                smc_algorithm_factory = blackjax.adaptive_tempered_smc

            dim = len(parameter_specs)
            if kernel_name == "rmh":
                if random_walk is None:
                    _, _, random_walk, _ = _blackjax_smc_components()
                rmh_kernel = random_walk.build_rmh()

                def mcmc_step_fn(rng_key, mcmc_state, logdensity_fn, step_scale):
                    transition_generator = random_walk.normal(step_scale)
                    return rmh_kernel(rng_key, mcmc_state, logdensity_fn, transition_generator)

                mcmc_init_fn = random_walk.init
                mcmc_parameters = {
                    "step_scale": jnp.full(
                        (1, dim),
                        float(getattr(args, "smc_rmh_scale", DEFAULT_SMC_RMH_SCALE)),
                        dtype=jnp.float64,
                    )
                }
            else:
                if mala is None:
                    _, _, _, mala = _blackjax_smc_components()
                mcmc_step_fn = mala.build_kernel()
                mcmc_init_fn = mala.init
                mcmc_parameters = {
                    "step_size": jnp.asarray(
                        [float(getattr(args, "smc_mala_step_size", DEFAULT_SMC_MALA_STEP_SIZE))],
                        dtype=jnp.float64,
                    )
                }

            if resampling is None:
                _, resampling, _, _ = _blackjax_smc_components()
            rng_seed = 0 if args.seed is None else int(args.seed)
            init_key, run_key = jax.random.split(jax.random.PRNGKey(rng_seed + 2303))
            if smc_device is not None:
                init_key = jax.device_put(init_key, smc_device)
                run_key = jax.device_put(run_key, smc_device)
            initial_particles = _smc_prior_particles(init_key, parameter_specs, particles)
            if smc_device is not None:
                initial_particles = jax.device_put(initial_particles, smc_device)
            algorithm = smc_algorithm_factory(
                logprior_fn=logprior_fn,
                loglikelihood_fn=loglikelihood_fn,
                mcmc_step_fn=mcmc_step_fn,
                mcmc_init_fn=mcmc_init_fn,
                mcmc_parameters=mcmc_parameters,
                resampling_fn=resampling.systematic,
                target_ess=float(target_ess_frac),
                num_mcmc_steps=mcmc_steps,
            )
            smc_state = algorithm.init(initial_particles)
            smc_step = jax.jit(algorithm.step)
            temperature_schedule: list[float] = [float(np.asarray(smc_state.tempering_param))]
            ess_history: list[float] = [_smc_weighted_ess(smc_state.weights)]
            move_acceptance_history: list[float] = []
            logz_estimate = 0.0
            smc_start = time.time()
            _log(args, "[smc] adaptive tempering started")
            for step_index in range(1, max_temperature_steps + 1):
                run_key, step_key = jax.random.split(run_key)
                step_start = time.time()
                smc_state, info = smc_step(step_key, smc_state)
                jax.tree_util.tree_map(
                    lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
                    smc_state,
                )
                if step_index == 1:
                    first_step_elapsed = time.time() - step_start
                temperature = float(np.asarray(smc_state.tempering_param))
                ess = _smc_weighted_ess(smc_state.weights)
                move_acceptance = _smc_move_acceptance(info)
                increment = float(np.asarray(info.log_likelihood_increment))
                logz_estimate += increment
                temperature_schedule.append(temperature)
                ess_history.append(ess)
                move_acceptance_history.append(move_acceptance)
                if not np.isfinite(temperature):
                    raise RuntimeError(f"BlackJAX SMC produced non-finite temperature at step {step_index}.")
                if step_index <= 5 or step_index % 10 == 0 or temperature >= 1.0 - SMC_FINAL_TEMPERATURE_TOL:
                    _log(
                        args,
                        (
                            f"[smc] step={step_index} temperature={temperature:.6g} "
                            f"ess={ess:.4g}/{particles} move_acceptance={move_acceptance:.3g} "
                            f"logz_increment={increment:.6g}"
                        ),
                    )
                if temperature >= 1.0 - SMC_FINAL_TEMPERATURE_TOL:
                    break
            smc_elapsed = time.time() - smc_start
            evaluator.timing_totals["smc_runtime"] = evaluator.timing_totals.get("smc_runtime", 0.0) + smc_elapsed
            final_temperature = float(np.asarray(smc_state.tempering_param))
            if not np.isfinite(final_temperature) or final_temperature < 1.0 - SMC_FINAL_TEMPERATURE_TOL:
                raise RuntimeError(
                    "BlackJAX SMC did not reach posterior temperature 1.0 "
                    f"after {max_temperature_steps} steps (temperature={final_temperature:.6g})."
                )
            normalized_samples = np.asarray(smc_state.particles, dtype=float)
            samples = np.asarray(
                jax.vmap(normalized_to_theta)(jnp.asarray(normalized_samples, dtype=jnp.float64)),
                dtype=float,
            )
            weights = np.asarray(smc_state.weights, dtype=float).reshape(-1)
    finally:
        if recompiled_for_smc:
            if default_device is None:
                if original_source_loglike_fn is not None:
                    evaluator._source_loglike_fn = original_source_loglike_fn
            else:
                _compile_evaluator_source_loglike(evaluator, device=default_device)
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        weights = np.full(samples.shape[0], 1.0 / float(samples.shape[0]), dtype=float)
    else:
        weights = weights / weight_sum
    with _jax_device_context(default_device):
        log_prob = _posterior_logprob_matrix(parameter_specs, evaluator, samples)
    diagnostics = {
        "strategy_requested": FIT_METHOD_SMC,
        "strategy_used": FIT_METHOD_SMC,
        "svi_used": False,
        "jax_default_device": _jax_device_label(default_device),
        "smc_device": _jax_device_label(smc_device),
        "smc_device_backend": _jax_device_backend(smc_device),
        "smc_particles": int(particles),
        "smc_mcmc_kernel": kernel_name,
        "smc_mcmc_steps": int(mcmc_steps),
        "smc_target_ess_frac": float(target_ess_frac),
        "smc_target_ess": float(target_ess),
        "smc_max_temperature_steps": int(max_temperature_steps),
        "smc_temperature_steps": int(len(temperature_schedule) - 1),
        "smc_final_temperature": float(final_temperature),
        "smc_final_weighted_ess": _smc_weighted_ess(weights),
        "smc_mean_move_acceptance": float(np.nanmean(move_acceptance_history)) if move_acceptance_history else float("nan"),
        "smc_logz_estimate": float(logz_estimate),
        "smc_runtime_sec": float(smc_elapsed),
        "smc_first_step_compile_run_sec": float(first_step_elapsed),
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
        "fixed_surrogate_drift": _smc_fixed_surrogate_drift_summary(evaluator, samples),
    }
    posterior = PosteriorResults(
        samples=samples,
        log_prob=log_prob,
        accept_prob=np.asarray(move_acceptance_history, dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=int(samples.shape[0]),
        num_chains=0,
        init_diagnostics=diagnostics,
        grouped_samples=None,
        grouped_log_prob=None,
        sampler="blackjax_smc",
        sample_weights=weights,
        temperature_schedule=np.asarray(temperature_schedule, dtype=float),
        ess_history=np.asarray(ess_history, dtype=float),
        move_acceptance_history=np.asarray(move_acceptance_history, dtype=float),
    )
    _log_posterior_summary(args, "smc", posterior)
    _log(
        args,
        (
            f"[smc] complete in {_fmt_seconds(smc_elapsed)} particles={samples.shape[0]} "
            f"temperature_steps={diagnostics['smc_temperature_steps']} "
            f"weighted_ess={diagnostics['smc_final_weighted_ess']:.4g} "
            f"logZ={diagnostics['smc_logz_estimate']:.6g}"
        ),
    )
    gc.collect()
    return posterior


def _prepare_svi_health_for_nuts(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
    evaluator: ClusterJAXEvaluator,
    center_theta: np.ndarray,
    svi_posterior: PosteriorResults,
    svi_diagnostics: dict[str, Any],
) -> list[ChainSeed]:
    chain_seeds = _make_svi_nuts_chain_seeds(args, parameter_specs, center_theta)
    health_points = np.vstack(
        [np.asarray(center_theta, dtype=float)]
        + [np.asarray(seed.values, dtype=float) for seed in chain_seeds]
    )
    health_log_prob = _run_logged_phase(
        args,
        "svi.health_logprob",
        lambda: _posterior_logprob_matrix(parameter_specs, evaluator, health_points),
    )
    center_log_prob = float(health_log_prob[0]) if health_log_prob.size else float("nan")
    chain_start_log_prob = np.asarray(health_log_prob[1:], dtype=float)
    metrics, warnings = _svi_health_diagnostics(
        parameter_specs,
        svi_posterior.samples,
        svi_posterior.log_prob,
        np.asarray(center_theta, dtype=float),
        center_log_prob,
        chain_seeds,
        chain_start_log_prob,
    )
    _record_svi_health_diagnostics(svi_posterior, svi_diagnostics, metrics, warnings)
    _log_svi_health(args, metrics, warnings)
    return chain_seeds


_SVI_REFRESH_MAX_ARRAYS = 64
_SVI_REFRESH_MAX_VALUES_PER_ARRAY = 256
_SVI_REFRESH_TOP_PARAMETER_CHANGES = 3


@dataclass(frozen=True)
class _SviRefreshArrayProbe:
    shape: tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True)
class _SviRefreshCacheSnapshot:
    enabled: bool
    reference_params: np.ndarray | None
    shapes: dict[str, tuple[int, ...]]
    probes: dict[str, _SviRefreshArrayProbe]


@dataclass(frozen=True)
class _SviRefreshCacheDelta:
    enabled: bool
    before_arrays: int
    after_arrays: int
    compared_arrays: int
    compared_values: int
    added_arrays: int
    removed_arrays: int
    shape_changed_arrays: int
    max_abs: float
    rms: float


@dataclass(frozen=True)
class _SviRefreshParameterDelta:
    theta_l2: float
    theta_linf: float
    top_changes: str


def _svi_refresh_float(value: Any) -> str:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return "na"
    if not np.isfinite(scalar):
        return "na"
    if scalar == 0.0:
        return "0"
    return f"{scalar:.3g}"


def _svi_refresh_key(value: Any) -> str:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isfinite(scalar):
        return f"{scalar:.12g}"
    return str(value)


def _svi_refresh_reference(reference_params: Any) -> np.ndarray | None:
    if reference_params is None:
        return None
    try:
        reference = np.asarray(reference_params, dtype=float)
    except (TypeError, ValueError):
        return None
    if reference.ndim != 1:
        return None
    return reference.copy()


def _svi_refresh_numeric_array(value: Any) -> Any | None:
    if value is None:
        return None
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            shape = tuple(int(dim) for dim in value.shape)
            dtype = np.dtype(value.dtype)
        except (TypeError, ValueError):
            return None
        if len(shape) == 0 or not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
            return None
        return value
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.ndim == 0 or not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        return None
    return array


def _svi_refresh_cache_entry_items(entry: Any) -> list[tuple[str, Any]]:
    if isinstance(entry, dict):
        return [(str(key), value) for key, value in entry.items()]
    if is_dataclass(entry) and not isinstance(entry, type):
        return [(field.name, getattr(entry, field.name)) for field in fields(entry)]
    return [("value", entry)]


def _svi_refresh_cache_arrays(cache_by_z: Any) -> list[tuple[str, Any]]:
    if not isinstance(cache_by_z, dict):
        return []
    arrays: list[tuple[str, Any]] = []
    for z_key, entry in cache_by_z.items():
        z_label = _svi_refresh_key(z_key)
        for field_name, value in _svi_refresh_cache_entry_items(entry):
            array = _svi_refresh_numeric_array(value)
            if array is None:
                continue
            arrays.append((f"z={z_label}.{field_name}", array))
    return sorted(arrays, key=lambda item: item[0])


def _svi_refresh_sample_array(array: Any, max_values_per_array: int) -> _SviRefreshArrayProbe:
    max_values = max(0, int(max_values_per_array))
    sample_count = min(max_values, int(array.size))
    if sample_count == 0:
        values = np.empty((0,), dtype=float)
    else:
        flat = array.reshape(-1)
        if int(array.size) <= sample_count:
            values = np.asarray(flat, dtype=float).copy()
        else:
            indices = np.linspace(0, int(array.size) - 1, num=sample_count, dtype=np.int64)
            values = np.asarray(flat[indices], dtype=float)
    return _SviRefreshArrayProbe(shape=tuple(int(dim) for dim in array.shape), values=values)


def _svi_refresh_cache_snapshot(
    cache_by_z: Any,
    reference_params: Any,
    *,
    enabled: bool = True,
    max_arrays: int = _SVI_REFRESH_MAX_ARRAYS,
    max_values_per_array: int = _SVI_REFRESH_MAX_VALUES_PER_ARRAY,
) -> _SviRefreshCacheSnapshot:
    if not enabled:
        return _SviRefreshCacheSnapshot(enabled=False, reference_params=None, shapes={}, probes={})
    arrays = _svi_refresh_cache_arrays(cache_by_z)
    shapes = {key: tuple(int(dim) for dim in array.shape) for key, array in arrays}
    probes = {
        key: _svi_refresh_sample_array(array, max_values_per_array)
        for key, array in arrays[: max(0, int(max_arrays))]
    }
    return _SviRefreshCacheSnapshot(
        enabled=True,
        reference_params=_svi_refresh_reference(reference_params),
        shapes=shapes,
        probes=probes,
    )


def _svi_refresh_evaluator_snapshot(evaluator: Any) -> dict[str, _SviRefreshCacheSnapshot]:
    return {
        "surrogate": _svi_refresh_cache_snapshot(
            getattr(evaluator, "surrogate_cache_by_z", {}),
            getattr(evaluator, "surrogate_reference_params", None),
            enabled=bool(getattr(evaluator, "surrogate_enabled", False)),
        ),
        "scaling": _svi_refresh_cache_snapshot(
            getattr(evaluator, "scaling_scatter_cache_by_z", {}),
            getattr(evaluator, "scaling_scatter_reference_params", None),
        ),
        "source_metric": _svi_refresh_cache_snapshot(
            getattr(evaluator, "source_metric_cache_by_z", {}),
            getattr(evaluator, "source_metric_reference_params", None),
        ),
    }


def _svi_refresh_cache_delta(
    before: _SviRefreshCacheSnapshot,
    after: _SviRefreshCacheSnapshot,
) -> _SviRefreshCacheDelta:
    if not before.enabled or not after.enabled:
        return _SviRefreshCacheDelta(
            enabled=False,
            before_arrays=0,
            after_arrays=0,
            compared_arrays=0,
            compared_values=0,
            added_arrays=0,
            removed_arrays=0,
            shape_changed_arrays=0,
            max_abs=float("nan"),
            rms=float("nan"),
        )
    before_keys = set(before.shapes)
    after_keys = set(after.shapes)
    common_keys = before_keys & after_keys
    shape_changed = sum(1 for key in common_keys if before.shapes[key] != after.shapes[key])
    compared_arrays = 0
    compared_values = 0
    max_abs = 0.0
    sum_sq = 0.0
    for key in sorted(common_keys):
        if before.shapes[key] != after.shapes[key] or key not in before.probes or key not in after.probes:
            continue
        before_values = before.probes[key].values
        after_values = after.probes[key].values
        value_count = min(int(before_values.size), int(after_values.size))
        compared_arrays += 1
        if value_count == 0:
            continue
        diff = after_values[:value_count] - before_values[:value_count]
        finite = np.isfinite(diff)
        if not finite.any():
            continue
        finite_diff = diff[finite]
        compared_values += int(finite_diff.size)
        abs_diff = np.abs(finite_diff)
        max_abs = max(max_abs, float(np.max(abs_diff)))
        sum_sq += float(np.sum(np.square(finite_diff)))
    rms = math.sqrt(sum_sq / compared_values) if compared_values else float("nan")
    return _SviRefreshCacheDelta(
        enabled=True,
        before_arrays=len(before_keys),
        after_arrays=len(after_keys),
        compared_arrays=compared_arrays,
        compared_values=compared_values,
        added_arrays=len(after_keys - before_keys),
        removed_arrays=len(before_keys - after_keys),
        shape_changed_arrays=shape_changed,
        max_abs=max_abs if compared_values else float("nan"),
        rms=rms,
    )


def _format_svi_refresh_cache_delta(label: str, delta: _SviRefreshCacheDelta) -> str:
    if not delta.enabled:
        return f"{label}=disabled"
    if delta.before_arrays == 0 and delta.after_arrays == 0:
        return f"{label}=empty arrays=0"
    return (
        f"{label}=max={_svi_refresh_float(delta.max_abs)} "
        f"rms={_svi_refresh_float(delta.rms)} "
        f"probes={delta.compared_values} "
        f"arrays={delta.compared_arrays}/{delta.after_arrays} "
        f"added={delta.added_arrays} removed={delta.removed_arrays} "
        f"shape={delta.shape_changed_arrays}"
    )


def _first_svi_refresh_reference(snapshots: dict[str, _SviRefreshCacheSnapshot]) -> np.ndarray | None:
    for label in ("surrogate", "scaling", "source_metric"):
        reference = snapshots.get(label)
        if reference is not None and reference.reference_params is not None:
            return reference.reference_params
    return None


def _svi_refresh_parameter_delta(
    before: np.ndarray | None,
    after: np.ndarray | None,
    parameter_specs: list[ParameterSpec],
) -> _SviRefreshParameterDelta:
    if before is None or after is None or before.shape != after.shape:
        return _SviRefreshParameterDelta(theta_l2=float("nan"), theta_linf=float("nan"), top_changes="none")
    diff = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    finite = np.isfinite(diff)
    if not finite.any():
        return _SviRefreshParameterDelta(theta_l2=float("nan"), theta_linf=float("nan"), top_changes="none")
    finite_diff = np.where(finite, diff, 0.0)
    abs_diff = np.abs(finite_diff)
    theta_l2 = float(np.sqrt(np.sum(np.square(finite_diff))))
    theta_linf = float(np.max(abs_diff))
    top_indices = [
        int(index)
        for index in np.argsort(-abs_diff)[:_SVI_REFRESH_TOP_PARAMETER_CHANGES]
        if abs_diff[int(index)] > 0.0
    ]
    top_changes = []
    for index in top_indices:
        if index < len(parameter_specs):
            name = str(parameter_specs[index].sample_name)
        else:
            name = f"theta_{index}"
        top_changes.append(f"{name}:{_svi_refresh_float(abs_diff[index])}")
    return _SviRefreshParameterDelta(
        theta_l2=theta_l2,
        theta_linf=theta_linf,
        top_changes=",".join(top_changes) if top_changes else "none",
    )


def _svi_refresh_delta_message(
    state: BuildState,
    *,
    reason: str,
    block_index: int,
    remaining_steps: int,
    center_shift: float,
    current_theta: np.ndarray,
    before: dict[str, _SviRefreshCacheSnapshot],
    after: dict[str, _SviRefreshCacheSnapshot],
) -> str:
    before_reference = _first_svi_refresh_reference(before)
    after_reference = _first_svi_refresh_reference(after)
    parameter_delta = _svi_refresh_parameter_delta(
        before_reference,
        after_reference if after_reference is not None else np.asarray(current_theta, dtype=float),
        state.parameter_specs,
    )
    cache_parts = [
        _format_svi_refresh_cache_delta(label, _svi_refresh_cache_delta(before[label], after[label]))
        for label in ("surrogate", "scaling", "source_metric")
    ]
    return " ".join(
        [
            f"[svi] refresh reason={reason}",
            f"block={block_index}",
            f"remaining_steps={remaining_steps}",
            f"center_shift={_svi_refresh_float(center_shift)}",
            f"theta_l2={_svi_refresh_float(parameter_delta.theta_l2)}",
            f"theta_linf={_svi_refresh_float(parameter_delta.theta_linf)}",
            f"top={parameter_delta.top_changes}",
            *cache_parts,
        ]
    )


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
    if posterior.init_diagnostics is None:
        posterior.init_diagnostics = {}
    diagnostics = posterior.init_diagnostics
    samples = np.asarray(posterior.samples, dtype=float)
    source_loglike = _source_loglike_matrix(evaluator, samples)
    if source_loglike.shape[0] == samples.shape[0] and np.isfinite(source_loglike).any():
        finite_loglike = np.where(np.isfinite(source_loglike), source_loglike, -np.inf)
        best_index = int(np.argmax(finite_loglike))
        best_fit = np.asarray(samples[best_index], dtype=float)
        log_prob = np.asarray(posterior.log_prob, dtype=float).reshape(-1)
        sample_log_prob = (
            float(log_prob[best_index])
            if best_index < log_prob.size and np.isfinite(log_prob[best_index])
            else None
        )
        diagnostics["fit_quality_reference_sample_kind"] = "max_likelihood"
        diagnostics["max_likelihood_sample_index"] = best_index
        diagnostics["max_likelihood_source_loglike"] = float(source_loglike[best_index])
        diagnostics["max_likelihood_sample_log_prob"] = sample_log_prob
        _log(
            args,
            (
                "[fit] best_fit updated from retained posterior sample with max source likelihood "
                f"index={best_index} source_loglike={float(source_loglike[best_index]):.6g}"
            ),
        )
        return best_fit
    diagnostics["fit_quality_reference_sample_kind"] = "fallback_best_fit"
    diagnostics["max_likelihood_sample_index"] = None
    diagnostics["max_likelihood_source_loglike"] = None
    diagnostics["max_likelihood_sample_log_prob"] = None
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
    *,
    chain_seeds: list[ChainSeed] | None = None,
) -> NUTSInitialization:
    if chain_seeds is None:
        chain_seeds = _make_svi_nuts_chain_seeds(args, parameter_specs, center_theta)
    chain_seeds = [
        ChainSeed(values=np.asarray(seed.values, dtype=float), source_label=str(seed.source_label))
        for seed in chain_seeds
    ]
    for chain_index, seed in enumerate(chain_seeds):
        if not np.all(np.isfinite(seed.values)):
            raise ValueError(f"SVI+NUTS initializer received non-finite values for chain {chain_index + 1}.")
    chain_start_labels = [f"{seed.source_label}:perturbed" for seed in chain_seeds]
    diagnostics = {
        "strategy_requested": "svi",
        "strategy_used": "svi",
        "svi_used": True,
        **svi_diagnostics,
        "potfile_mass_size_reparam_enabled": bool(_potfile_mass_size_group_count(parameter_specs) > 0),
        "potfile_mass_size_reparam_group_count": int(_potfile_mass_size_group_count(parameter_specs)),
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


def _nuts_initialization_from_reference(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
    reference_theta: np.ndarray,
    *,
    chain_seeds: list[ChainSeed] | None = None,
) -> NUTSInitialization:
    reference = _clip_theta_to_support(
        np.asarray(reference_theta, dtype=float),
        parameter_specs,
        boundary_frac=DEFAULT_NUTS_INIT_BOUNDARY_FRAC,
    )
    if chain_seeds is None:
        chain_seeds = _make_direct_nuts_chain_seeds(args, parameter_specs, reference)
    chain_seeds = [
        ChainSeed(values=np.asarray(seed.values, dtype=float), source_label=str(seed.source_label))
        for seed in chain_seeds
    ]
    for chain_index, seed in enumerate(chain_seeds):
        if not np.all(np.isfinite(seed.values)):
            raise ValueError(f"Direct NUTS initializer received non-finite values for chain {chain_index + 1}.")
    chain_start_labels = [f"{seed.source_label}:perturbed" for seed in chain_seeds]
    diagnostics = {
        "strategy_requested": FIT_METHOD_NUTS,
        "strategy_used": "previous_stage",
        "svi_used": False,
        "potfile_mass_size_reparam_enabled": bool(_potfile_mass_size_group_count(parameter_specs) > 0),
        "potfile_mass_size_reparam_group_count": int(_potfile_mass_size_group_count(parameter_specs)),
        "nuts_init_source": "state_init_values_or_midpoint",
        "direct_nuts_chain_seed_labels": [seed.source_label for seed in chain_seeds],
        "direct_nuts_chain_start_labels": chain_start_labels,
        "chain_seed_labels": [seed.source_label for seed in chain_seeds],
        "distinct_chain_seeds": int(len({tuple(np.round(seed.values, 8)) for seed in chain_seeds})),
    }
    return NUTSInitialization(
        init_params=_seed_values_to_init_params(parameter_specs, chain_seeds, model_for_init=_sample_site_model(parameter_specs)),
        chain_seeds=chain_seeds,
        diagnostics=diagnostics,
        reference_theta=reference,
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
            refresh_reason = f"svi_block_{block_index}"
            before_refresh = _svi_refresh_evaluator_snapshot(evaluator)
            if evaluator.surrogate_enabled:
                evaluator.refresh_surrogate(center_theta, reason=refresh_reason)
            evaluator.refresh_scaling_scatter_cache(center_theta, reason=refresh_reason)
            evaluator.refresh_source_metric_cache(center_theta, reason=refresh_reason)
            after_refresh = _svi_refresh_evaluator_snapshot(evaluator)
            _log(
                args,
                _svi_refresh_delta_message(
                    state,
                    reason=refresh_reason,
                    block_index=block_index,
                    remaining_steps=remaining_steps,
                    center_shift=block_center_shift[-1] if block_center_shift else float("nan"),
                    current_theta=center_theta,
                    before=before_refresh,
                    after=after_refresh,
                ),
            )
            block_refresh_count += 1
    if guide is None or svi is None or svi_result is None or params is None or center_theta_previous is None:
        raise RuntimeError("SVI did not run any steps.")
    before_refresh = _svi_refresh_evaluator_snapshot(evaluator)
    if evaluator.surrogate_enabled:
        evaluator.refresh_surrogate(center_theta_previous, reason="svi_final")
    evaluator.refresh_scaling_scatter_cache(center_theta_previous, reason="svi_final")
    evaluator.refresh_source_metric_cache(center_theta_previous, reason="svi_final")
    after_refresh = _svi_refresh_evaluator_snapshot(evaluator)
    _log(
        args,
        _svi_refresh_delta_message(
            state,
            reason="svi_final",
            block_index=block_index,
            remaining_steps=0,
            center_shift=block_center_shift[-1] if block_center_shift else float("nan"),
            current_theta=center_theta_previous,
            before=before_refresh,
            after=after_refresh,
        ),
    )
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
    guide_samples = np.stack([_site_array_for_spec(guide_samples_dict, spec) for spec in state.parameter_specs], axis=-1)
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
        "potfile_mass_size_reparam_enabled": bool(_potfile_mass_size_group_count(state.parameter_specs) > 0),
        "potfile_mass_size_reparam_group_count": int(_potfile_mass_size_group_count(state.parameter_specs)),
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
        "fixed_image_sigma_int_arcsec": (
            None
            if getattr(evaluator, "fixed_image_sigma_int_arcsec", None) is None
            else float(getattr(evaluator, "fixed_image_sigma_int_arcsec"))
        ),
        "image_sigma_int_sampled": bool(getattr(evaluator, "image_sigma_int_sampled", False)),
        "effective_image_sigma_int_arcsec": (
            None
            if getattr(evaluator, "fixed_image_sigma_int_arcsec", None) is None
            else float(getattr(evaluator, "fixed_image_sigma_int_arcsec"))
        ),
    }
    if bool(getattr(args, "debug_sampler_diagnostics", False)):
        diagnostics["sampler_debug_diagnostics_skipped"] = True
        diagnostics["sampler_debug_diagnostics_skip_reason"] = "sampler_debug diagnostics are NUTS-only"
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
    if bool(getattr(args, "debug_sampler_diagnostics", False)):
        _log(args, "[sampler-debug] skipped sampler=svi; diagnostics are only written for NUTS runs")
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
    max_tree_depth = _max_tree_depth_for_args(args)
    dense_mass = bool(getattr(args, "dense_mass", True))
    nuts_init.diagnostics["dense_mass"] = dense_mass
    nuts_init.diagnostics["nuts_dense_mass"] = dense_mass
    nuts_init.diagnostics["potfile_mass_size_reparam_enabled"] = bool(
        _potfile_mass_size_group_count(state.parameter_specs) > 0
    )
    nuts_init.diagnostics["potfile_mass_size_reparam_group_count"] = int(
        _potfile_mass_size_group_count(state.parameter_specs)
    )
    nuts_init.diagnostics["fixed_image_sigma_int_arcsec"] = (
        None
        if getattr(evaluator, "fixed_image_sigma_int_arcsec", None) is None
        else float(getattr(evaluator, "fixed_image_sigma_int_arcsec"))
    )
    nuts_init.diagnostics["image_sigma_int_sampled"] = bool(getattr(evaluator, "image_sigma_int_sampled", False))
    nuts_init.diagnostics["effective_image_sigma_int_arcsec"] = (
        None
        if getattr(evaluator, "fixed_image_sigma_int_arcsec", None) is None
        else float(getattr(evaluator, "fixed_image_sigma_int_arcsec"))
    )
    _log(
        args,
        (
            f"[nuts] preparing sampler chains={args.chains} chain_method={chain_method} "
            f"warmup={args.warmup} samples={args.samples} thin={args.thin} "
            f"target_accept={args.target_accept:.2f} max_tree_depth={max_tree_depth} "
            f"dense_mass={dense_mass} "
            f"potfile_mass_size_reparam_groups={_potfile_mass_size_group_count(state.parameter_specs)} "
            f"image_sigma_int_sampled={bool(getattr(evaluator, 'image_sigma_int_sampled', False))} "
            f"fixed_image_sigma_int_arcsec={getattr(evaluator, 'fixed_image_sigma_int_arcsec', None)} "
            f"initial_step_size={float(args.initial_step_size):.3g} "
            f"init={nuts_init.diagnostics['strategy_used']} distinct_seeds={nuts_init.diagnostics['distinct_chain_seeds']}"
        ),
    )
    nuts = NUTS(
        sample_model,
        target_accept_prob=args.target_accept,
        max_tree_depth=max_tree_depth,
        step_size=float(args.initial_step_size),
        dense_mass=dense_mass,
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
            extra_fields=_sampler_debug_extra_field_names(args),
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
    if bool(getattr(args, "debug_sampler_diagnostics", False)):
        try:
            debug_diag = _run_logged_phase(
                args,
                "sampler_debug.write_diagnostics",
                lambda: _write_sampler_debug_diagnostics(args, state, evaluator, nuts_init, posterior, extra),
            )
            if debug_diag:
                nuts_init.diagnostics["sampler_debug_diagnostics"] = dict(debug_diag)
        except Exception as exc:  # pragma: no cover - diagnostics should not change sampler behavior
            _log(args, f"[sampler-debug] failed to write diagnostics: {exc}")
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


def _blocked_nuts_parameter_blocks(parameter_specs: list[ParameterSpec]) -> tuple[BlockedNUTSParameterBlock, ...]:
    non_source_indices = tuple(
        idx for idx, spec in enumerate(parameter_specs) if str(spec.component_family) != "source_position"
    )
    source_indices = tuple(
        idx for idx, spec in enumerate(parameter_specs) if str(spec.component_family) == "source_position"
    )
    if not non_source_indices:
        raise ValueError("Blocked NUTS requires at least one non-source parameter.")
    if not source_indices:
        raise ValueError("Blocked NUTS requires at least one source-position parameter.")
    return (
        BlockedNUTSParameterBlock("non_source", non_source_indices),
        BlockedNUTSParameterBlock("source_position", source_indices),
    )


def _conditioned_posterior_model_for_block(
    parameter_specs: list[ParameterSpec],
    evaluator: ClusterJAXEvaluator,
    fixed_indices: tuple[int, ...],
    theta: np.ndarray,
):
    base_model = _posterior_model(parameter_specs, evaluator)
    fixed_data = {
        parameter_specs[idx].sample_name: jnp.asarray(float(theta[idx]), dtype=jnp.float64)
        for idx in fixed_indices
    }
    return numpyro.handlers.condition(base_model, data=fixed_data)


def _block_init_params(
    parameter_specs: list[ParameterSpec],
    block: BlockedNUTSParameterBlock,
    block_model,
    theta: np.ndarray,
) -> dict[str, jnp.ndarray]:
    constrained_params = {
        parameter_specs[idx].sample_name: jnp.asarray(float(theta[idx]), dtype=jnp.float64)
        for idx in block.indices
    }
    unconstrained = unconstrain_fn(
        block_model,
        model_args=(),
        model_kwargs={},
        params=constrained_params,
    )
    return {
        parameter_specs[idx].sample_name: jnp.asarray(unconstrained[parameter_specs[idx].sample_name])
        for idx in block.indices
    }


def _update_theta_from_block_samples(
    theta: np.ndarray,
    parameter_specs: list[ParameterSpec],
    block: BlockedNUTSParameterBlock,
    samples_dict: dict[str, Any],
) -> np.ndarray:
    updated = np.asarray(theta, dtype=float).copy()
    for idx in block.indices:
        name = parameter_specs[idx].sample_name
        values = np.asarray(samples_dict[name], dtype=float).reshape(-1)
        if values.size:
            updated[idx] = float(values[-1])
    return _clip_theta_to_support(updated, parameter_specs, boundary_frac=DEFAULT_NUTS_INIT_BOUNDARY_FRAC)


def _copy_inverse_mass_matrix(value: Any) -> Any:
    return jax.tree_util.tree_map(lambda item: jnp.asarray(item), value)


def _block_extra_scalar(extra: dict[str, Any], key: str, default: float = float("nan")) -> float:
    if key not in extra:
        return float(default)
    values = np.asarray(extra[key], dtype=float).reshape(-1)
    return float(values[-1]) if values.size else float(default)


def _run_block_nuts_once(
    args: argparse.Namespace,
    parameter_specs: list[ParameterSpec],
    evaluator: ClusterJAXEvaluator,
    theta: np.ndarray,
    block: BlockedNUTSParameterBlock,
    rng_key: jax.Array,
    *,
    num_warmup: int,
    step_size: float | None = None,
    inverse_mass_matrix: Any | None = None,
    adapt: bool = True,
    phase_name: str | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    fixed_indices = tuple(idx for idx in range(len(parameter_specs)) if idx not in set(block.indices))
    block_model = _conditioned_posterior_model_for_block(parameter_specs, evaluator, fixed_indices, theta)
    init_params = _block_init_params(parameter_specs, block, block_model, theta)
    dense_mass = bool(getattr(args, "dense_mass", True))
    nuts = NUTS(
        block_model,
        target_accept_prob=float(args.target_accept),
        max_tree_depth=_max_tree_depth_for_args(args),
        step_size=float(step_size if step_size is not None else args.initial_step_size),
        inverse_mass_matrix=None if inverse_mass_matrix is None else _copy_inverse_mass_matrix(inverse_mass_matrix),
        adapt_step_size=bool(adapt),
        adapt_mass_matrix=bool(adapt),
        dense_mass=dense_mass,
    )
    mcmc = MCMC(
        nuts,
        num_warmup=int(num_warmup),
        num_samples=1,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
    )

    def run() -> None:
        mcmc.run(
            rng_key,
            extra_fields=("accept_prob", "diverging", "num_steps", "potential_energy"),
            init_params=init_params,
        )

    if phase_name:
        _run_logged_phase(args, phase_name, run)
    else:
        run()
    samples_dict = mcmc.get_samples(group_by_chain=False)
    extra = mcmc.get_extra_fields(group_by_chain=False)
    updated_theta = _update_theta_from_block_samples(theta, parameter_specs, block, samples_dict)
    last_state = mcmc.last_state
    adapt_state = getattr(last_state, "adapt_state", None)
    adapted = {
        "step_size": float(np.asarray(adapt_state.step_size)) if adapt_state is not None else float(args.initial_step_size),
        "inverse_mass_matrix": (
            _copy_inverse_mass_matrix(adapt_state.inverse_mass_matrix)
            if adapt_state is not None
            else None
        ),
    }
    metrics = {
        "accept_prob": _block_extra_scalar(extra, "accept_prob"),
        "diverging": bool(_block_extra_scalar(extra, "diverging", 0.0)),
        "num_steps": _block_extra_scalar(extra, "num_steps", 0.0),
        "potential_energy": _block_extra_scalar(extra, "potential_energy"),
    }
    del mcmc, samples_dict, extra
    gc.collect()
    return updated_theta, metrics, adapted


def _summarize_blocked_nuts_metrics(values: dict[str, list[float]]) -> dict[str, float | int]:
    accept = np.asarray(values.get("accept_prob", []), dtype=float)
    diverging = np.asarray(values.get("diverging", []), dtype=bool)
    num_steps = np.asarray(values.get("num_steps", []), dtype=float)
    return {
        "draws": int(max(accept.size, diverging.size, num_steps.size)),
        "accept_mean": float(np.nanmean(accept)) if accept.size else float("nan"),
        "divergences": int(np.sum(diverging)) if diverging.size else 0,
        "divergence_fraction": float(np.mean(diverging)) if diverging.size else 0.0,
        "num_steps_mean": float(np.nanmean(num_steps)) if num_steps.size else float("nan"),
        "num_steps_max": float(np.nanmax(num_steps)) if num_steps.size else float("nan"),
    }


def _run_blocked_numpyro_nuts_sampler(
    args: argparse.Namespace,
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    nuts_init: NUTSInitialization,
) -> PosteriorResults:
    blocks = _blocked_nuts_parameter_blocks(state.parameter_specs)
    block_by_name = {block.name: block for block in blocks}
    cycles = int(getattr(args, "blocked_nuts_cycles", None) or args.samples)
    pilot_warmup = int(getattr(args, "blocked_nuts_pilot_warmup", None) or args.warmup)
    if cycles <= 0:
        raise ValueError("--blocked-nuts-cycles must be positive.")
    if pilot_warmup < 0:
        raise ValueError("--blocked-nuts-pilot-warmup must be non-negative.")
    _log(
        args,
        (
            "[blocked-nuts] preparing sampler "
            f"chains={args.chains} cycles={cycles} pilot_warmup={pilot_warmup} "
            f"thin={args.thin} target_accept={args.target_accept:.2f} "
            f"max_tree_depth={_max_tree_depth_for_args(args)} "
            f"dense_mass={bool(getattr(args, 'dense_mass', True))} "
            f"blocks={{non_source:{len(block_by_name['non_source'].indices)}, "
            f"source_position:{len(block_by_name['source_position'].indices)}}}"
        ),
    )
    start = time.time()
    base_seed = 0 if args.seed is None else int(args.seed)
    grouped_samples: list[np.ndarray] = []
    grouped_accept: list[np.ndarray] = []
    grouped_diverging: list[np.ndarray] = []
    grouped_num_steps: list[np.ndarray] = []
    per_block_metrics: dict[str, dict[str, list[float]]] = {
        block.name: {"accept_prob": [], "diverging": [], "num_steps": []}
        for block in blocks
    }

    chain_seeds = list(nuts_init.chain_seeds)
    if len(chain_seeds) < int(args.chains):
        raise ValueError("Blocked NUTS received fewer chain seeds than requested chains.")

    for chain_index in range(int(args.chains)):
        theta = np.asarray(chain_seeds[chain_index].values, dtype=float).copy()
        if not np.all(np.isfinite(theta)):
            raise ValueError(f"Blocked NUTS initializer received non-finite values for chain {chain_index + 1}.")
        _log(args, f"[blocked-nuts] chain={chain_index + 1}/{args.chains} pilot warmup started")
        rng_key = jax.random.PRNGKey(base_seed + 5003 + chain_index * 1009)
        rng_key, non_source_key, source_key = jax.random.split(rng_key, 3)
        theta, pilot_metrics_non_source, non_source_adapted = _run_block_nuts_once(
            args,
            state.parameter_specs,
            evaluator,
            theta,
            block_by_name["non_source"],
            non_source_key,
            num_warmup=pilot_warmup,
            adapt=True,
            phase_name=f"blocked_nuts.chain_{chain_index + 1}.pilot_non_source",
        )
        theta, pilot_metrics_source, source_adapted = _run_block_nuts_once(
            args,
            state.parameter_specs,
            evaluator,
            theta,
            block_by_name["source_position"],
            source_key,
            num_warmup=pilot_warmup,
            adapt=True,
            phase_name=f"blocked_nuts.chain_{chain_index + 1}.pilot_source_position",
        )
        per_block_metrics["non_source"]["accept_prob"].append(float(pilot_metrics_non_source["accept_prob"]))
        per_block_metrics["non_source"]["diverging"].append(float(pilot_metrics_non_source["diverging"]))
        per_block_metrics["non_source"]["num_steps"].append(float(pilot_metrics_non_source["num_steps"]))
        per_block_metrics["source_position"]["accept_prob"].append(float(pilot_metrics_source["accept_prob"]))
        per_block_metrics["source_position"]["diverging"].append(float(pilot_metrics_source["diverging"]))
        per_block_metrics["source_position"]["num_steps"].append(float(pilot_metrics_source["num_steps"]))

        chain_samples = np.empty((cycles, len(state.parameter_specs)), dtype=float)
        chain_accept = np.empty((cycles,), dtype=float)
        chain_diverging = np.empty((cycles,), dtype=bool)
        chain_num_steps = np.empty((cycles,), dtype=float)
        report_every = max(1, cycles // 5)
        _log(args, f"[blocked-nuts] chain={chain_index + 1}/{args.chains} production started")
        for cycle in range(cycles):
            rng_key, non_source_key, source_key = jax.random.split(rng_key, 3)
            theta, non_source_metrics, _ = _run_block_nuts_once(
                args,
                state.parameter_specs,
                evaluator,
                theta,
                block_by_name["non_source"],
                non_source_key,
                num_warmup=0,
                step_size=float(non_source_adapted["step_size"]),
                inverse_mass_matrix=non_source_adapted["inverse_mass_matrix"],
                adapt=False,
            )
            theta, source_metrics, _ = _run_block_nuts_once(
                args,
                state.parameter_specs,
                evaluator,
                theta,
                block_by_name["source_position"],
                source_key,
                num_warmup=0,
                step_size=float(source_adapted["step_size"]),
                inverse_mass_matrix=source_adapted["inverse_mass_matrix"],
                adapt=False,
            )
            for block_name, metrics in (
                ("non_source", non_source_metrics),
                ("source_position", source_metrics),
            ):
                per_block_metrics[block_name]["accept_prob"].append(float(metrics["accept_prob"]))
                per_block_metrics[block_name]["diverging"].append(float(metrics["diverging"]))
                per_block_metrics[block_name]["num_steps"].append(float(metrics["num_steps"]))
            chain_samples[cycle] = theta
            chain_accept[cycle] = float(np.nanmean([non_source_metrics["accept_prob"], source_metrics["accept_prob"]]))
            chain_diverging[cycle] = bool(non_source_metrics["diverging"]) or bool(source_metrics["diverging"])
            chain_num_steps[cycle] = float(non_source_metrics["num_steps"]) + float(source_metrics["num_steps"])
            if (cycle + 1) % report_every == 0 or cycle + 1 == cycles:
                _log(args, f"[blocked-nuts] chain={chain_index + 1}/{args.chains} cycle={cycle + 1}/{cycles}")
        grouped_samples.append(chain_samples)
        grouped_accept.append(chain_accept)
        grouped_diverging.append(chain_diverging)
        grouped_num_steps.append(chain_num_steps)

    grouped_samples_array = np.asarray(grouped_samples, dtype=float)
    grouped_accept_array = np.asarray(grouped_accept, dtype=float)
    grouped_diverging_array = np.asarray(grouped_diverging, dtype=bool)
    grouped_num_steps_array = np.asarray(grouped_num_steps, dtype=float)
    flat_unthinned = grouped_samples_array.reshape(-1, grouped_samples_array.shape[-1])
    log_prob_unthinned = _run_logged_phase(
        args,
        "blocked_nuts.full_posterior_logprob",
        lambda: _posterior_logprob_matrix(state.parameter_specs, evaluator, flat_unthinned),
    )
    grouped_log_prob_array = np.asarray(log_prob_unthinned, dtype=float).reshape(grouped_samples_array.shape[:2])
    sample_finite = np.isfinite(grouped_samples_array).all(axis=(1, 2))
    log_prob_finite = np.isfinite(grouped_log_prob_array).all(axis=1)
    valid_chain_mask = sample_finite & log_prob_finite
    retained_indices = np.where(valid_chain_mask)[0].astype(int).tolist()
    dropped_indices = np.where(~valid_chain_mask)[0].astype(int).tolist()
    if not retained_indices:
        raise RuntimeError(
            "All blocked NUTS chains produced non-finite posterior samples or log probabilities; no finite chains remain."
        )
    grouped_samples_array = grouped_samples_array[valid_chain_mask]
    grouped_log_prob_array = grouped_log_prob_array[valid_chain_mask]
    grouped_accept_array = grouped_accept_array[valid_chain_mask]
    grouped_diverging_array = grouped_diverging_array[valid_chain_mask]
    grouped_num_steps_array = grouped_num_steps_array[valid_chain_mask]
    thin = max(1, int(args.thin))
    grouped_samples_thinned = grouped_samples_array[:, ::thin, :]
    grouped_log_prob_thinned = grouped_log_prob_array[:, ::thin]
    samples = grouped_samples_array.reshape(-1, grouped_samples_array.shape[-1])[::thin]
    log_prob = grouped_log_prob_array.reshape(-1)[::thin]
    accept_prob = grouped_accept_array.reshape(-1)[::thin]
    diverging = grouped_diverging_array.reshape(-1)[::thin]
    num_steps = grouped_num_steps_array.reshape(-1)[::thin]
    elapsed = time.time() - start
    evaluator.timing_totals["nuts_runtime"] += elapsed
    block_summaries = {
        block_name: _summarize_blocked_nuts_metrics(values)
        for block_name, values in per_block_metrics.items()
    }
    diagnostics = {
        **nuts_init.diagnostics,
        "sampler": "numpyro_blocked_nuts",
        "blocked_nuts": True,
        "dense_mass": bool(getattr(args, "dense_mass", True)),
        "nuts_dense_mass": bool(getattr(args, "dense_mass", True)),
        "blocked_nuts_cycles": int(cycles),
        "blocked_nuts_pilot_warmup": int(pilot_warmup),
        "blocked_nuts_blocks": {
            block.name: [state.parameter_specs[idx].sample_name for idx in block.indices]
            for block in blocks
        },
        "blocked_nuts_block_sizes": {block.name: int(len(block.indices)) for block in blocks},
        "blocked_nuts_block_summaries": block_summaries,
        "requested_chains": int(args.chains),
        "retained_finite_chains": int(len(retained_indices)),
        "dropped_nonfinite_chains": int(len(dropped_indices)),
        "retained_chain_indices": retained_indices,
        "dropped_chain_indices": dropped_indices,
        "invalid_state_rejection_count": int(getattr(evaluator, "invalid_state_rejection_count", 0)),
        "invalid_state_reason_counts": {
            key: int(value) for key, value in dict(getattr(evaluator, "invalid_state_reason_counts", {})).items()
        },
    }
    posterior = PosteriorResults(
        samples=samples,
        log_prob=log_prob,
        accept_prob=accept_prob,
        diverging=diverging,
        num_steps=num_steps,
        warmup_steps=pilot_warmup,
        sample_steps=cycles,
        num_chains=int(len(retained_indices)),
        init_diagnostics=diagnostics,
        grouped_samples=grouped_samples_thinned,
        grouped_log_prob=grouped_log_prob_thinned,
        sampler="numpyro_blocked_nuts",
    )
    _apply_nuts_quality_gate(args, posterior, state.parameter_specs)
    _log_posterior_summary(args, "blocked_nuts", posterior)
    _log(
        args,
        (
            "[blocked-nuts] complete "
            f"in {_fmt_seconds(elapsed)} accept_mean={np.mean(accept_prob):.3f} "
            f"divergences={int(np.sum(diverging))} mean_steps={np.mean(num_steps):.2f} "
            f"retained_samples={samples.shape[0]}"
        ),
    )
    _log(
        args,
        (
            "[blocked-nuts] invalid-state guards "
            f"rejections={int(getattr(evaluator, 'invalid_state_rejection_count', 0))} "
            f"reasons={json.dumps({key: int(value) for key, value in dict(getattr(evaluator, 'invalid_state_reason_counts', {})).items() if int(value) > 0}, sort_keys=True)}"
        ),
    )
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
            "devices": (
                [_resolve_jax_device_for_args(args, "jax_default_device", flag_name="--jax-default-device")]
                if str(getattr(args, "jax_default_device", JAX_DEVICE_AUTO)) != JAX_DEVICE_AUTO
                else jax.devices()
            ),
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
        _fail(f"{flag_name} requires one to three values.")
    if len(values) > 3:
        _fail(f"{flag_name} accepts at most three values: stage 2, stage 3, and stage 4.")
    return values


def _max_tree_depth_for_args(args: argparse.Namespace) -> int:
    values = _stage_arg_values(
        getattr(args, "max_tree_depth", DEFAULT_MAX_TREE_DEPTH),
        flag_name="--max-tree-depth",
    )
    return int(values[0])


def _linearized_stage_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE)) in {
        IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED,
    }


def _forward_metric_stage_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE)) == IMAGE_PLANE_MODE_FORWARD_METRIC


def _anchored_solved_stage_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE)) == IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA


def _critical_arc_mixture_stage_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE)) == IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE


def _fold_regularized_stage_enabled(args: argparse.Namespace) -> bool:
    return (
        str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE))
        == IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA
    )


def _blocked_linearized_stage_enabled(args: argparse.Namespace) -> bool:
    return (
        str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE))
        == IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED
    )


def _stage4_image_plane_enabled(args: argparse.Namespace) -> bool:
    return (
        _linearized_stage_enabled(args)
        or _forward_metric_stage_enabled(args)
        or _anchored_solved_stage_enabled(args)
        or _critical_arc_mixture_stage_enabled(args)
        or _fold_regularized_stage_enabled(args)
    )


def _sample_likelihood_uses_explicit_beta(sample_likelihood_mode: str) -> bool:
    return sample_likelihood_mode in {
        SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
        SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
        SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
    }


def _sample_likelihood_uses_image_scatter(sample_likelihood_mode: str) -> bool:
    return _sample_likelihood_uses_explicit_beta(sample_likelihood_mode)


def _effective_image_presence_penalty_weight(
    requested_weight: float | None,
    *,
    sample_likelihood_mode: str,
    fit_mode: str,
    image_plane_mode: str,
) -> float:
    if requested_weight is not None:
        return float(requested_weight)
    if str(sample_likelihood_mode) not in {
        SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
        SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
        SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
    }:
        return 0.0
    if str(sample_likelihood_mode) == SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE:
        return 0.0
    if str(fit_mode) == FIT_MODE_EVIDENCE_NS:
        return 0.0
    if str(image_plane_mode) not in {
        IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED,
        IMAGE_PLANE_MODE_FORWARD_METRIC,
        IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
        IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
        IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
    }:
        return 0.0
    return DEFAULT_IMAGE_PRESENCE_STAGE4_PENALTY_WEIGHT


def _effective_cab_likelihood_weight(
    requested_weight: float | None,
    state: BuildState,
) -> float:
    if requested_weight is not None:
        return float(requested_weight)
    if _has_arc_constraints_in_state(state):
        return DEFAULT_CAB_LIKELIHOOD_WEIGHT_WITH_ARCS
    return DEFAULT_CAB_LIKELIHOOD_WEIGHT_NO_ARCS


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
    if mode in {IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA, IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED}:
        return SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
    if mode == IMAGE_PLANE_MODE_FORWARD_METRIC:
        return SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE
    if mode == IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA:
        return SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE
    if mode == IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE:
        return SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE
    if mode == IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA:
        return SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE
    return None


def _stage4_run_directory_name(args: argparse.Namespace) -> str:
    if _blocked_linearized_stage_enabled(args):
        return "stage4_blocked_linearized_image_plane"
    if _forward_metric_stage_enabled(args):
        return "stage4_forward_metric_image_plane"
    if _anchored_solved_stage_enabled(args):
        return "stage4_anchored_solved_image_plane"
    if _critical_arc_mixture_stage_enabled(args):
        return "stage4_critical_arc_mixture_image_plane"
    if _fold_regularized_stage_enabled(args):
        return "stage4_fold_regularized_image_plane"
    return "stage4_linearized_image_plane"


SEQUENTIAL_STAGE_NAMES = {
    "stage1_large_only",
    "stage2_joint",
    "stage3_image_plane",
    "stage4_linearized_image_plane",
    "stage4_blocked_linearized_image_plane",
    "stage4_forward_metric_image_plane",
    "stage4_anchored_solved_image_plane",
    "stage4_critical_arc_mixture_image_plane",
    "stage4_fold_regularized_image_plane",
}
SEQUENTIAL_STAGE_ORDER = (
    "stage1_large_only",
    "stage2_joint",
    "stage3_image_plane",
    "stage4_linearized_image_plane",
    "stage4_blocked_linearized_image_plane",
    "stage4_forward_metric_image_plane",
    "stage4_anchored_solved_image_plane",
    "stage4_critical_arc_mixture_image_plane",
)


def _sequential_stage_name(value: str | Path) -> str:
    return Path(str(value)).name


def _is_sequential_stage_path(value: str | Path) -> bool:
    return _sequential_stage_name(value) in SEQUENTIAL_STAGE_NAMES


def _final_sequential_exact_diagnostics_stage(args: argparse.Namespace, *, stage3_enabled: bool, stage4_enabled: bool) -> str:
    if stage4_enabled:
        return _stage4_run_directory_name(args)
    if stage3_enabled:
        return "stage3_image_plane"
    return "stage2_joint"


def _stage_allows_exact_image_diagnostics(value: str | Path, exact_diagnostics_stage: str | Path | None) -> bool:
    if not _is_sequential_stage_path(value):
        return True
    if exact_diagnostics_stage is None:
        return True
    return _sequential_stage_name(value) == _sequential_stage_name(exact_diagnostics_stage)


def _stage3_exact_image_diagnostics_enabled(args: argparse.Namespace | None, value: str | Path) -> bool:
    return (
        bool(getattr(args, "exact_image_diagnostics_stage3", False))
        and _sequential_stage_name(value) == "stage3_image_plane"
    )


def _force_quick_diagnostics_for_nonfinal_stage(
    stage_args: argparse.Namespace,
    run_name: str | Path,
    exact_diagnostics_stage: str | Path | None,
) -> argparse.Namespace:
    if _stage3_exact_image_diagnostics_enabled(stage_args, run_name):
        return _clone_args(stage_args, quick_diagnostics=False)
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
    if _sequential_stage_name(run_dir) in {
        "stage4_linearized_image_plane",
        "stage4_blocked_linearized_image_plane",
        "stage4_forward_metric_image_plane",
        "stage4_anchored_solved_image_plane",
        "stage4_critical_arc_mixture_image_plane",
        "stage4_fold_regularized_image_plane",
    }:
        return _sequential_stage_name(run_dir)
    sibling_stage_dirs = [run_dir.parent / stage_name for stage_name in SEQUENTIAL_STAGE_ORDER]
    return _final_available_sequential_stage(sibling_stage_dirs) or _sequential_stage_name(run_dir)


def _local_jacobian_stage_enabled(args: argparse.Namespace) -> bool:
    mode = str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE))
    if mode == IMAGE_PLANE_MODE_LOCAL_JACOBIAN:
        return True
    if mode in {
        IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED,
        IMAGE_PLANE_MODE_FORWARD_METRIC,
        IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
        IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
        IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
    }:
        return not bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False))
    return False


def _normalize_stage_fit_controls(args: argparse.Namespace) -> dict[str, StageFitControls]:
    fit_mode = str(getattr(args, "fit_mode", FIT_MODE_SEQUENTIAL))
    mode = str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE))
    start_at_stage3 = bool(getattr(args, "start_at_stage3", False))
    _validate_jax_device_controls(args)
    if bool(getattr(args, "quick_diagnostics", False)) and bool(getattr(args, "exact_image_diagnostics_stage3", False)):
        _fail("--exact-image-diagnostics-stage3 cannot be combined with --quick-diagnostics.")
    if bool(getattr(args, "resume_fast", False)) and fit_mode != FIT_MODE_SEQUENTIAL:
        _fail("--resume-fast is only valid with --fit-mode sequential.")
    if start_at_stage3:
        if fit_mode != FIT_MODE_SEQUENTIAL:
            _fail("--start-at-stage3 is only valid with --fit-mode sequential.")
        if mode not in {
            IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
            IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
            IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED,
            IMAGE_PLANE_MODE_FORWARD_METRIC,
            IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
            IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
            IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
        }:
            _fail("--start-at-stage3 requires a stage-3-capable --image-plane-mode.")
        if bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)):
            _fail("--start-at-stage3 requires stage 3 and is incompatible with --skip-stage3-image-plane-local-jacobian.")
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
    if int(getattr(args, "smc_particles", DEFAULT_SMC_PARTICLES)) <= 0:
        _fail("--smc-particles must be positive.")
    if str(getattr(args, "smc_mcmc_kernel", DEFAULT_SMC_MCMC_KERNEL)) not in SMC_MCMC_KERNELS:
        _fail(f"--smc-mcmc-kernel must be one of {', '.join(SMC_MCMC_KERNELS)}.")
    if int(getattr(args, "smc_mcmc_steps", DEFAULT_SMC_MCMC_STEPS)) <= 0:
        _fail("--smc-mcmc-steps must be positive.")
    smc_target_ess_frac = float(getattr(args, "smc_target_ess_frac", DEFAULT_SMC_TARGET_ESS_FRAC))
    if not np.isfinite(smc_target_ess_frac) or smc_target_ess_frac <= 0.0 or smc_target_ess_frac > 1.0:
        _fail("--smc-target-ess-frac must be in (0, 1].")
    if int(getattr(args, "smc_max_temperature_steps", DEFAULT_SMC_MAX_TEMPERATURE_STEPS)) <= 0:
        _fail("--smc-max-temperature-steps must be positive.")
    if (
        not np.isfinite(float(getattr(args, "smc_rmh_scale", DEFAULT_SMC_RMH_SCALE)))
        or float(getattr(args, "smc_rmh_scale", DEFAULT_SMC_RMH_SCALE)) <= 0.0
    ):
        _fail("--smc-rmh-scale must be positive.")
    if (
        not np.isfinite(float(getattr(args, "smc_mala_step_size", DEFAULT_SMC_MALA_STEP_SIZE)))
        or float(getattr(args, "smc_mala_step_size", DEFAULT_SMC_MALA_STEP_SIZE)) <= 0.0
    ):
        _fail("--smc-mala-step-size must be positive.")
    if float(getattr(args, "linearized_beta_prior_sigma_arcsec", DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC)) <= 0.0:
        _fail("--linearized-beta-prior-sigma-arcsec must be positive.")
    critical_det_threshold = float(
        getattr(args, "critical_det_diagnostic_threshold", DEFAULT_CRITICAL_DET_DIAGNOSTIC_THRESHOLD)
    )
    if not np.isfinite(critical_det_threshold) or critical_det_threshold <= 0.0:
        _fail("--critical-det-diagnostic-threshold must be finite and positive.")
    image_catalog_cutout_dir = getattr(args, "image_catalog_family_cutout_image_dir", None)
    if image_catalog_cutout_dir is not None and not str(image_catalog_cutout_dir).strip():
        _fail("--image-catalog-family-cutout-image-dir must be a non-empty path when provided.")
    for option_name in (
        "image_catalog_family_cutout_rgb_q",
        "image_catalog_family_cutout_rgb_stretch",
        "image_catalog_family_cutout_rgb_red_gain",
        "image_catalog_family_cutout_rgb_green_gain",
        "image_catalog_family_cutout_rgb_blue_gain",
    ):
        option_value = getattr(args, option_name, None)
        if option_value is not None and (not np.isfinite(float(option_value)) or float(option_value) <= 0.0):
            _fail(f"--{option_name.replace('_', '-')} must be finite and positive.")
    cutout_rgb_minimum = getattr(args, "image_catalog_family_cutout_rgb_minimum", None)
    if cutout_rgb_minimum is not None and not np.isfinite(float(cutout_rgb_minimum)):
        _fail("--image-catalog-family-cutout-rgb-minimum must be finite.")
    anchored_solve_steps = int(
        getattr(args, "anchored_image_plane_solve_steps", DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS)
    )
    if anchored_solve_steps < 0:
        _fail("--anchored-image-plane-solve-steps must be non-negative.")
    anchored_trust_radius = float(
        getattr(args, "anchored_image_plane_trust_radius_arcsec", DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC)
    )
    if not np.isfinite(anchored_trust_radius) or anchored_trust_radius <= 0.0:
        _fail("--anchored-image-plane-trust-radius-arcsec must be finite and positive.")
    anchored_lm_relative = float(
        getattr(args, "anchored_image_plane_lm_damping_relative", DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE)
    )
    if not np.isfinite(anchored_lm_relative) or anchored_lm_relative <= 0.0:
        _fail("--anchored-image-plane-lm-damping-relative must be finite and positive.")
    anchored_lm_absolute = float(
        getattr(args, "anchored_image_plane_lm_damping_absolute", DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE)
    )
    if not np.isfinite(anchored_lm_absolute) or anchored_lm_absolute <= 0.0:
        _fail("--anchored-image-plane-lm-damping-absolute must be finite and positive.")
    critical_arc_critical_direction_sigma = float(
        getattr(args, "critical_arc_critical_direction_sigma_arcsec", DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC)
    )
    if not np.isfinite(critical_arc_critical_direction_sigma) or critical_arc_critical_direction_sigma <= 0.0:
        _fail("--critical-arc-critical-direction-sigma-arcsec must be finite and positive.")
    critical_arc_base_prob = float(getattr(args, "critical_arc_base_prob", DEFAULT_CRITICAL_ARC_BASE_PROB))
    critical_arc_max_prob = float(getattr(args, "critical_arc_max_prob", DEFAULT_CRITICAL_ARC_MAX_PROB))
    if (
        not np.isfinite(critical_arc_base_prob)
        or not np.isfinite(critical_arc_max_prob)
        or critical_arc_base_prob < 0.0
        or critical_arc_max_prob > 1.0
        or critical_arc_base_prob > critical_arc_max_prob
    ):
        _fail("--critical-arc-base-prob and --critical-arc-max-prob must satisfy 0 <= base <= max <= 1.")
    critical_arc_singular_threshold = float(
        getattr(args, "critical_arc_singular_threshold", DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD)
    )
    if not np.isfinite(critical_arc_singular_threshold) or critical_arc_singular_threshold <= 0.0:
        _fail("--critical-arc-singular-threshold must be finite and positive.")
    critical_arc_singular_softness = float(
        getattr(args, "critical_arc_singular_softness", DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS)
    )
    if not np.isfinite(critical_arc_singular_softness) or critical_arc_singular_softness <= 0.0:
        _fail("--critical-arc-singular-softness must be finite and positive.")
    critical_arc_lm_relative = float(
        getattr(args, "critical_arc_lm_damping_relative", DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE)
    )
    if not np.isfinite(critical_arc_lm_relative) or critical_arc_lm_relative <= 0.0:
        _fail("--critical-arc-lm-damping-relative must be finite and positive.")
    critical_arc_lm_absolute = float(
        getattr(args, "critical_arc_lm_damping_absolute", DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE)
    )
    if not np.isfinite(critical_arc_lm_absolute) or critical_arc_lm_absolute <= 0.0:
        _fail("--critical-arc-lm-damping-absolute must be finite and positive.")
    critical_arc_lm_trust_radius = float(
        getattr(args, "critical_arc_lm_trust_radius_arcsec", DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC)
    )
    if not np.isfinite(critical_arc_lm_trust_radius) or critical_arc_lm_trust_radius <= 0.0:
        _fail("--critical-arc-lm-trust-radius-arcsec must be finite and positive.")
    arc_aware_noncritical_support_radius = float(
        getattr(
            args,
            "arc_aware_noncritical_support_radius_arcsec",
            DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC,
        )
    )
    if not np.isfinite(arc_aware_noncritical_support_radius) or arc_aware_noncritical_support_radius <= 0.0:
        _fail("--arc-aware-noncritical-support-radius-arcsec must be finite and positive.")
    arc_aware_max_arclength = float(
        getattr(args, "arc_aware_max_arclength_arcsec", DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC)
    )
    if not np.isfinite(arc_aware_max_arclength) or arc_aware_max_arclength <= 0.0:
        _fail("--arc-aware-max-arclength-arcsec must be finite and positive.")
    arc_aware_curve_step = float(getattr(args, "arc_aware_curve_step_arcsec", DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC))
    if not np.isfinite(arc_aware_curve_step) or arc_aware_curve_step <= 0.0:
        _fail("--arc-aware-curve-step-arcsec must be finite and positive.")
    fold_curvature = float(getattr(args, "fold_curvature_arcsec_inv", DEFAULT_FOLD_CURVATURE_ARCSEC_INV))
    if not np.isfinite(fold_curvature) or fold_curvature <= 0.0:
        _fail("--fold-curvature-arcsec-inv must be finite and positive.")
    cab_likelihood_weight = getattr(args, "cab_likelihood_weight", None)
    if cab_likelihood_weight is not None and (
        not np.isfinite(float(cab_likelihood_weight)) or float(cab_likelihood_weight) < 0.0
    ):
        _fail("--cab-likelihood-weight must be finite and non-negative when provided.")
    cab_fd_step = float(getattr(args, "cab_finite_difference_step_arcsec", DEFAULT_CAB_FINITE_DIFFERENCE_STEP_ARCSEC))
    if not np.isfinite(cab_fd_step) or cab_fd_step <= 0.0:
        _fail("--cab-finite-difference-step-arcsec must be finite and positive.")
    cab_tangent_floor = float(getattr(args, "cab_tangent_sigma_floor_rad", DEFAULT_CAB_TANGENT_SIGMA_FLOOR_RAD))
    if not np.isfinite(cab_tangent_floor) or cab_tangent_floor <= 0.0:
        _fail("--cab-tangent-sigma-floor-rad must be finite and positive.")
    cab_curvature_floor = float(
        getattr(args, "cab_curvature_sigma_floor_arcsec_inv", DEFAULT_CAB_CURVATURE_SIGMA_FLOOR_ARCSEC_INV)
    )
    if not np.isfinite(cab_curvature_floor) or cab_curvature_floor <= 0.0:
        _fail("--cab-curvature-sigma-floor-arcsec-inv must be finite and positive.")
    image_scatter_floor = float(getattr(args, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC))
    if not np.isfinite(image_scatter_floor) or image_scatter_floor <= 0.0:
        _fail("--image-plane-scatter-floor-arcsec must be positive.")
    image_scatter_upper = float(getattr(args, "image_plane_scatter_upper_arcsec", DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC))
    if not np.isfinite(image_scatter_upper) or image_scatter_upper <= image_scatter_floor:
        _fail(
            "--image-plane-scatter-upper-arcsec must be greater than "
            "--image-plane-scatter-floor-arcsec."
        )
    if str(getattr(args, "image_plane_scatter_prior", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR)) not in IMAGE_PLANE_SCATTER_PRIORS:
        _fail(
            "--image-plane-scatter-prior must be one of "
            f"{', '.join(IMAGE_PLANE_SCATTER_PRIORS)}."
        )
    if (
        not np.isfinite(float(getattr(args, "image_plane_scatter_prior_median_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC)))
        or float(getattr(args, "image_plane_scatter_prior_median_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC)) <= 0.0
    ):
        _fail("--image-plane-scatter-prior-median-arcsec must be positive.")
    image_scatter_prior = str(getattr(args, "image_plane_scatter_prior", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR))
    image_scatter_prior_median = float(
        getattr(args, "image_plane_scatter_prior_median_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC)
    )
    if (
        image_scatter_prior == IMAGE_PLANE_SCATTER_PRIOR_LOGNORMAL
        and not (image_scatter_floor < image_scatter_prior_median < image_scatter_upper)
    ):
        _fail(
            "--image-plane-scatter-prior-median-arcsec must be between "
            "--image-plane-scatter-floor-arcsec and --image-plane-scatter-upper-arcsec for lognormal scatter priors."
        )
    if (
        not np.isfinite(float(getattr(args, "image_plane_scatter_prior_log_sigma", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA)))
        or float(getattr(args, "image_plane_scatter_prior_log_sigma", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA)) <= 0.0
    ):
        _fail("--image-plane-scatter-prior-log-sigma must be positive.")
    fixed_image_sigma_int = getattr(args, "fix_image_sigma_int_arcsec", None)
    if fixed_image_sigma_int is not None and (
        not np.isfinite(float(fixed_image_sigma_int)) or float(fixed_image_sigma_int) < 0.0
    ):
        _fail("--fix-image-sigma-int-arcsec must be finite and nonnegative.")
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
    if (
        not np.isfinite(float(getattr(args, "likelihood_stabilizer_max_gain", DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN)))
        or float(getattr(args, "likelihood_stabilizer_max_gain", DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN)) < 0.0
    ):
        _fail("--likelihood-stabilizer-max-gain must be non-negative.")
    if (
        not np.isfinite(float(getattr(args, "likelihood_stabilizer_max_residual_arcsec", DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC)))
        or float(getattr(args, "likelihood_stabilizer_max_residual_arcsec", DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC)) < 0.0
    ):
        _fail("--likelihood-stabilizer-max-residual-arcsec must be non-negative.")
    if str(getattr(args, "likelihood_stabilizer_residual_loss", DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS)) not in LIKELIHOOD_STABILIZER_RESIDUAL_LOSSES:
        _fail(
            "--likelihood-stabilizer-residual-loss must be one of "
            f"{', '.join(LIKELIHOOD_STABILIZER_RESIDUAL_LOSSES)}."
        )
    if (
        not np.isfinite(float(getattr(args, "likelihood_stabilizer_student_t_nu", DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU)))
        or float(getattr(args, "likelihood_stabilizer_student_t_nu", DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU)) <= 0.0
    ):
        _fail("--likelihood-stabilizer-student-t-nu must be positive.")
    if int(getattr(args, "fit_quality_draws", 0)) < 0:
        _fail("--fit-quality-draws must be non-negative.")
    try:
        _cosmology_init_overrides_from_args(args)
    except ValueError as exc:
        _fail(str(exc))
    caustic_source_redshift = float(getattr(args, "caustic_source_redshift", 9.0))
    if not np.isfinite(caustic_source_redshift) or caustic_source_redshift <= 0.0:
        _fail("--caustic-source-redshift must be finite and positive.")
    caustic_plot_grid_scale_arcsec = float(
        getattr(args, "caustic_plot_grid_scale_arcsec", CAUSTIC_PLOT_GRID_SCALE_ARCSEC)
    )
    if not np.isfinite(caustic_plot_grid_scale_arcsec) or caustic_plot_grid_scale_arcsec <= 0.0:
        _fail("--caustic-plot-grid-scale-arcsec must be finite and positive.")
    kappa_true_fits = getattr(args, "kappa_true_fits", None)
    if kappa_true_fits is not None:
        kappa_true_text = str(kappa_true_fits).strip()
        if not kappa_true_text:
            _fail("--kappa-true-fits must be a non-empty path when provided.")
        if not Path(kappa_true_text).is_file():
            _fail(f"--kappa-true-fits does not exist: {kappa_true_text}")
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
    max_tree_depths = [
        int(value)
        for value in _stage_arg_values(
            getattr(args, "max_tree_depth", DEFAULT_MAX_TREE_DEPTH),
            flag_name="--max-tree-depth",
        )
    ]
    if any(value < 0 for value in max_tree_depths):
        _fail("--max-tree-depth values must be non-negative.")
    if fit_mode == FIT_MODE_EVIDENCE_NS:
        if bool(getattr(args, "potfile_mass_size_reparam", False)):
            _fail("--potfile-mass-size-reparam is only supported with NumPyro SVI/NUTS samplers, not --fit-mode evidence-ns.")
        if evidence_prior_sigma is None:
            _fail("--fit-mode evidence-ns requires --evidence-source-prior-sigma-arcsec.")
        if mode != IMAGE_PLANE_MODE_NONE:
            _fail("--fit-mode evidence-ns owns its likelihood and requires --image-plane-mode none.")
        if str(getattr(args, "sampling_engine", SAMPLING_ENGINE_FULL)) == SAMPLING_ENGINE_ACTIVE_SUBSET:
            _fail("--sampling-engine active_subset is not valid with --fit-mode evidence-ns.")
        if bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)):
            _fail("--skip-stage3-image-plane-local-jacobian is not valid with --fit-mode evidence-ns.")
        if (
            str(getattr(args, "sampling_engine", SAMPLING_ENGINE_FULL)) == SAMPLING_ENGINE_REFRESHING_SURROGATE
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
        source_position_parameterization = str(
            getattr(args, "source_position_parameterization", SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED)
        )
        if source_position_parameterization not in SOURCE_POSITION_PARAMETERIZATIONS:
            _fail(
                "--source-position-parameterization must be one of "
                f"{', '.join(SOURCE_POSITION_PARAMETERIZATIONS)}."
            )
        controls = {
            "stage2": StageFitControls(
                fit_method=FIT_METHOD_NS,
                svi_steps=0,
                warmup=0,
                samples=0,
                max_tree_depth=int(max_tree_depths[0]),
            ),
            "stage3": StageFitControls(
                fit_method=FIT_METHOD_NS,
                svi_steps=0,
                warmup=0,
                samples=0,
                max_tree_depth=int(max_tree_depths[0]),
            ),
            "stage4": StageFitControls(
                fit_method=FIT_METHOD_NS,
                svi_steps=0,
                warmup=0,
                samples=0,
                max_tree_depth=int(max_tree_depths[0]),
            ),
        }
        return controls
    if evidence_likelihood_mode != DEFAULT_EVIDENCE_LIKELIHOOD_MODE:
        _fail("--evidence-likelihood-mode is only valid with --fit-mode evidence-ns.")
    if mode in {
        IMAGE_PLANE_MODE_FORWARD_METRIC,
        IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
        IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
        IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
    }:
        if int(getattr(args, "image_plane_newton_steps", 0)) != 0:
            _fail(f"--image-plane-newton-steps must be 0 for --image-plane-mode {mode}.")
        if (
            str(getattr(args, "source_position_parameterization", SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED))
            == SOURCE_POSITION_PARAMETERIZATION_CONDITIONAL_WHITENED
        ):
            _fail(
                "--source-position-parameterization conditional-whitened is not supported with "
                f"--image-plane-mode {mode}."
            )
    if (
        mode == IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA
        and str(getattr(args, "sampling_engine", SAMPLING_ENGINE_FULL)) == SAMPLING_ENGINE_REFRESHING_SURROGATE
        and anchored_solve_steps > 0
    ):
        _fail(
            "--sampling-engine refreshing_surrogate is not supported with "
            "--image-plane-mode anchored-solved-forward-beta-image-plane unless "
            "--anchored-image-plane-solve-steps is 0."
        )

    fit_methods = [str(value) for value in _stage_arg_values(getattr(args, "fit_method", FIT_METHOD_SVI_NUTS), flag_name="--fit-method")]
    svi_steps = [int(value) for value in _stage_arg_values(getattr(args, "svi_steps", DEFAULT_SVI_STEPS), flag_name="--svi-steps")]
    warmups = [int(value) for value in _stage_arg_values(getattr(args, "warmup", DEFAULT_WARMUP), flag_name="--warmup")]
    samples = [int(value) for value in _stage_arg_values(getattr(args, "samples", DEFAULT_SAMPLES), flag_name="--samples")]

    invalid_fit_methods = sorted(
        set(fit_methods).difference({FIT_METHOD_SVI, FIT_METHOD_SVI_NUTS, FIT_METHOD_NUTS, FIT_METHOD_NS, FIT_METHOD_SMC})
    )
    if invalid_fit_methods:
        _fail(f"--fit-method has unsupported value(s): {', '.join(invalid_fit_methods)}")
    if any(value == FIT_METHOD_NS for value in fit_methods):
        _fail("--fit-method ns is only valid with --fit-mode evidence-ns.")
    if any(value <= 0 for value in svi_steps):
        _fail("--svi-steps values must be positive.")
    if any(value < 0 for value in warmups):
        _fail("--warmup values must be non-negative.")
    if any(value <= 0 for value in samples):
        _fail("--samples values must be positive.")
    if getattr(args, "blocked_nuts_cycles", None) is not None and int(args.blocked_nuts_cycles) <= 0:
        _fail("--blocked-nuts-cycles must be positive when provided.")
    if getattr(args, "blocked_nuts_pilot_warmup", None) is not None and int(args.blocked_nuts_pilot_warmup) < 0:
        _fail("--blocked-nuts-pilot-warmup must be non-negative when provided.")

    max_value_count = max(len(fit_methods), len(svi_steps), len(warmups), len(samples), len(max_tree_depths))
    has_stage_specific_values = max_value_count >= 2
    has_three_stage_values = max_value_count == 3
    is_sequential = fit_mode == FIT_MODE_SEQUENTIAL
    has_stage3_or_stage4 = is_sequential and mode in {
        IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED,
        IMAGE_PLANE_MODE_FORWARD_METRIC,
        IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
        IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
        IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
    }
    has_stage4 = _stage4_image_plane_enabled(args)
    has_stage3 = _local_jacobian_stage_enabled(args)
    if bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)) and not has_stage4:
        _fail("--skip-stage3-image-plane-local-jacobian is only valid with a final stage-4 image-plane mode.")
    if has_stage4 and not is_sequential:
        _fail("Stage-4 image-plane modes require --fit-mode sequential.")
    if (
        _linearized_stage_enabled(args)
        and str(getattr(args, "sampling_engine", SAMPLING_ENGINE_FULL)) == SAMPLING_ENGINE_REFRESHING_SURROGATE
        and int(getattr(args, "image_plane_newton_steps", 0)) > 0
    ):
        _fail(
            "--sampling-engine refreshing_surrogate with linearized-forward-beta-image-plane "
            "requires --image-plane-newton-steps 0."
        )
    if has_stage_specific_values and not has_stage3_or_stage4:
        _fail(
            "Two-value --fit-method, --svi-steps, --warmup, --samples, or --max-tree-depth is only valid with "
            "--fit-mode sequential and an image-plane mode."
        )
    if has_three_stage_values and not has_stage4:
        _fail(
            "Three-value --fit-method, --svi-steps, --warmup, --samples, or --max-tree-depth is only valid with "
            "a final stage-4 image-plane mode."
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
            svi_steps=int(stage_value(svi_steps, 0)),
            warmup=int(stage_value(warmups, 0)),
            samples=int(stage_value(samples, 0)),
            max_tree_depth=int(stage_value(max_tree_depths, 0)),
        ),
        "stage3": StageFitControls(
            fit_method=str(stage_value(fit_methods, 1)),
            svi_steps=int(stage_value(svi_steps, 1)),
            warmup=int(stage_value(warmups, 1)),
            samples=int(stage_value(samples, 1)),
            max_tree_depth=int(stage_value(max_tree_depths, 1)),
        ),
        "stage4": StageFitControls(
            fit_method=str(stage4_value(fit_methods)),
            svi_steps=int(stage4_value(svi_steps)),
            warmup=int(stage4_value(warmups)),
            samples=int(stage4_value(samples)),
            max_tree_depth=int(stage4_value(max_tree_depths)),
        ),
    }
    stage4_direct_sampler_modes = {
        IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        IMAGE_PLANE_MODE_FORWARD_METRIC,
        IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
        IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
        IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
    }
    smc_stages: list[str] = []
    if controls["stage2"].fit_method == FIT_METHOD_SMC:
        smc_stages.append("stage2")
    if _local_jacobian_stage_enabled(args) and controls["stage3"].fit_method == FIT_METHOD_SMC:
        smc_stages.append("stage3")
    if has_stage4 and controls["stage4"].fit_method == FIT_METHOD_SMC:
        smc_stages.append("stage4")
    nuts_stages: list[str] = []
    if controls["stage2"].fit_method == FIT_METHOD_NUTS:
        nuts_stages.append("stage2")
    if _local_jacobian_stage_enabled(args) and controls["stage3"].fit_method == FIT_METHOD_NUTS:
        nuts_stages.append("stage3")
    if has_stage4 and controls["stage4"].fit_method == FIT_METHOD_NUTS:
        nuts_stages.append("stage4")
    if _blocked_linearized_stage_enabled(args) and controls["stage4"].fit_method != FIT_METHOD_SVI_NUTS:
        _fail(
            "--image-plane-mode linearized-forward-beta-blocked-image-plane requires "
            "stage-4 --fit-method svi+nuts."
        )
    if smc_stages:
        if smc_stages != ["stage4"] or str(mode) not in stage4_direct_sampler_modes:
            _fail("--fit-method smc is only valid for non-blocked sequential stage 4 image-plane modes.")
    if nuts_stages:
        if nuts_stages != ["stage4"] or str(mode) not in stage4_direct_sampler_modes:
            _fail("--fit-method nuts is only valid for non-blocked sequential stage 4 image-plane modes.")
    if bool(getattr(args, "potfile_mass_size_reparam", False)):
        if _blocked_linearized_stage_enabled(args):
            _fail("--potfile-mass-size-reparam is not supported with blocked linearized stage 4 NUTS.")
        if smc_stages:
            _fail("--potfile-mass-size-reparam is only supported with NumPyro SVI/NUTS samplers, not --fit-method smc.")
    return controls


def _args_with_fit_controls(args: argparse.Namespace, controls: StageFitControls, **updates: Any) -> argparse.Namespace:
    payload: dict[str, Any] = {
        "fit_method": controls.fit_method,
        "svi_steps": controls.svi_steps,
        "warmup": controls.warmup,
        "samples": controls.samples,
        "max_tree_depth": controls.max_tree_depth,
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


def _sequential_fiducial_cosmology_config() -> dict[str, float | str]:
    return dict(SEQUENTIAL_FIDUCIAL_COSMOLOGY_CONFIG)


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


def _cosmology_config_override_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    override = getattr(args, "cosmology_config_override", None)
    if not isinstance(override, dict):
        return None
    return dict(override)


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


def _finite_cli_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _fov_limit_from_args(args: argparse.Namespace) -> FOVLimit | None:
    radius_value = getattr(args, "fov_limit_radius", None)
    x_values = getattr(args, "fov_limit_x", None)
    y_values = getattr(args, "fov_limit_y", None)
    if radius_value is None and x_values is None and y_values is None:
        return None

    radius_arcsec: float | None = None
    if radius_value is not None:
        radius_arcsec = _finite_cli_float(radius_value, "--fov-limit-radius")
        if radius_arcsec < 0.0:
            raise ValueError("--fov-limit-radius must be non-negative.")

    def bounds(values: Any, name: str) -> tuple[float, float] | tuple[None, None]:
        if values is None:
            return None, None
        if len(values) != 2:
            raise ValueError(f"{name} requires exactly two values.")
        first = _finite_cli_float(values[0], name)
        second = _finite_cli_float(values[1], name)
        return min(first, second), max(first, second)

    x_min, x_max = bounds(x_values, "--fov-limit-x")
    y_min, y_max = bounds(y_values, "--fov-limit-y")
    limit = FOVLimit(
        radius_arcsec=radius_arcsec,
        x_min_arcsec=x_min,
        x_max_arcsec=x_max,
        y_min_arcsec=y_min,
        y_max_arcsec=y_max,
    )
    return limit if limit.is_active else None


def _format_fov_limit(limit: FOVLimit) -> str:
    parts: list[str] = []
    if limit.radius_arcsec is not None:
        parts.append(f"radius<={limit.radius_arcsec:.6g}")
    if limit.x_min_arcsec is not None and limit.x_max_arcsec is not None:
        parts.append(f"x=[{limit.x_min_arcsec:.6g},{limit.x_max_arcsec:.6g}]")
    if limit.y_min_arcsec is not None and limit.y_max_arcsec is not None:
        parts.append(f"y=[{limit.y_min_arcsec:.6g},{limit.y_max_arcsec:.6g}]")
    return " ".join(parts)


def _fov_mask_from_offsets(x_arcsec: np.ndarray, y_arcsec: np.ndarray, limit: FOVLimit | None) -> np.ndarray:
    x_values = np.asarray(x_arcsec, dtype=float)
    y_values = np.asarray(y_arcsec, dtype=float)
    if limit is None:
        return np.ones_like(x_values, dtype=bool)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if limit.radius_arcsec is not None:
        mask &= np.hypot(x_values, y_values) <= float(limit.radius_arcsec)
    if limit.x_min_arcsec is not None:
        mask &= x_values >= float(limit.x_min_arcsec)
    if limit.x_max_arcsec is not None:
        mask &= x_values <= float(limit.x_max_arcsec)
    if limit.y_min_arcsec is not None:
        mask &= y_values >= float(limit.y_min_arcsec)
    if limit.y_max_arcsec is not None:
        mask &= y_values <= float(limit.y_max_arcsec)
    return mask


def _fov_mask_for_catalog(catalog_df: pd.DataFrame, reference: tuple[int, float, float], limit: FOVLimit | None) -> np.ndarray:
    if catalog_df.empty:
        return np.zeros(0, dtype=bool)
    _reference_type, ra0_deg, dec0_deg = reference
    x_arcsec, y_arcsec = _radec_to_offsets_arcsec(
        catalog_df["ra"].to_numpy(dtype=float),
        catalog_df["dec"].to_numpy(dtype=float),
        ra0_deg,
        dec0_deg,
    )
    return _fov_mask_from_offsets(x_arcsec, y_arcsec, limit)


def _filter_images_by_fov(
    images_df: pd.DataFrame,
    reference: tuple[int, float, float],
    limit: FOVLimit | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if limit is None:
        return images_df, {}
    mask = _fov_mask_for_catalog(images_df, reference, limit)
    dropped = images_df.loc[~mask]
    filtered = images_df.loc[mask].copy().reset_index(drop=True)
    before_family_ids = set(images_df["family_id"].astype(str).unique().tolist()) if "family_id" in images_df else set()
    after_family_ids = set(filtered["family_id"].astype(str).unique().tolist()) if "family_id" in filtered else set()
    affected_family_ids = (
        sorted(dropped["family_id"].astype(str).unique().tolist()) if "family_id" in dropped and not dropped.empty else []
    )
    removed_family_ids = sorted(before_family_ids - after_family_ids)
    return filtered, {
        "total": int(len(images_df)),
        "kept": int(len(filtered)),
        "dropped": int(len(dropped)),
        "affected_family_ids": affected_family_ids,
        "removed_family_ids": removed_family_ids,
    }


def _filter_potfiles_by_fov(
    potfiles: list[dict[str, Any]],
    reference: tuple[int, float, float],
    limit: FOVLimit | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    if limit is None:
        return potfiles, {}
    filtered_potfiles: list[dict[str, Any]] = []
    summary: dict[str, dict[str, int]] = {}
    for index, potfile in enumerate(potfiles):
        catalog_df = potfile.get("catalog_df")
        if not isinstance(catalog_df, pd.DataFrame):
            filtered_potfiles.append(potfile)
            continue
        mask = _fov_mask_for_catalog(catalog_df, reference, limit)
        filtered_catalog = catalog_df.loc[mask].copy().reset_index(drop=True)
        filtered_potfile = dict(potfile)
        filtered_potfile["catalog_df"] = filtered_catalog
        filtered_potfiles.append(filtered_potfile)
        potfile_id = str(potfile.get("id", f"potfile{index}"))
        summary[potfile_id] = {
            "total": int(len(catalog_df)),
            "kept": int(len(filtered_catalog)),
            "dropped": int(len(catalog_df) - len(filtered_catalog)),
        }
    return filtered_potfiles, summary


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


def _spec_sample_site_name(spec: ParameterSpec) -> str:
    return str(spec.sample_site_name or spec.sample_name)


def _spec_sample_site_index(spec: ParameterSpec) -> int | None:
    value = spec.sample_site_index
    return None if value is None else int(value)


def _is_potfile_mass_size_spec(spec: ParameterSpec) -> bool:
    return str(getattr(spec, "coupled_transform_kind", "none")) == COUPLED_TRANSFORM_POTFILE_MASS_SIZE


def _potfile_mass_size_group_count(parameter_specs: list[ParameterSpec]) -> int:
    groups = {
        str(spec.coupled_group or spec.sample_site_name or spec.potential_id)
        for spec in parameter_specs
        if _is_potfile_mass_size_spec(spec)
    }
    return len(groups)


@dataclass(frozen=True)
class _SampleSiteSpec:
    name: str
    indices: tuple[int, ...]


def _parameter_sample_sites(parameter_specs: list[ParameterSpec]) -> list[_SampleSiteSpec]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for idx, spec in enumerate(parameter_specs):
        site_name = _spec_sample_site_name(spec)
        site_index = _spec_sample_site_index(spec)
        grouped.setdefault(site_name, []).append((idx, idx if site_index is None else site_index))
    sites: list[_SampleSiteSpec] = []
    seen: set[str] = set()
    for spec in parameter_specs:
        site_name = _spec_sample_site_name(spec)
        if site_name in seen:
            continue
        seen.add(site_name)
        ordered = sorted(grouped[site_name], key=lambda item: item[1])
        expected = list(range(len(ordered)))
        actual = [int(item[1]) for item in ordered]
        if len(ordered) > 1 and actual != expected:
            raise ValueError(f"Vector sample site {site_name!r} has non-contiguous indices {actual}; expected {expected}.")
        sites.append(_SampleSiteSpec(name=site_name, indices=tuple(int(item[0]) for item in ordered)))
    return sites


def _site_value_from_theta(theta: np.ndarray | jnp.ndarray, site: _SampleSiteSpec) -> jnp.ndarray:
    theta_array = jnp.asarray(theta, dtype=jnp.float64)
    if len(site.indices) == 1:
        return jnp.asarray(theta_array[int(site.indices[0])], dtype=jnp.float64)
    return jnp.stack([theta_array[int(idx)] for idx in site.indices])


def _site_array_for_spec(samples_dict: dict[str, Any], spec: ParameterSpec) -> np.ndarray:
    site_name = _spec_sample_site_name(spec)
    if site_name in samples_dict:
        values = np.asarray(samples_dict[site_name], dtype=float)
        site_index = _spec_sample_site_index(spec)
        if site_index is None:
            return values
        return np.asarray(values[..., int(site_index)], dtype=float)
    return np.asarray(samples_dict[spec.sample_name], dtype=float)


class _PotfileMassSizeReparamDistribution(dist.Distribution):
    arg_constraints: dict[str, Any] = {}
    support = dist.constraints.real_vector
    reparametrized_params: list[str] = []

    def __init__(
        self,
        sigma_distribution: dist.Distribution,
        cut_distribution: dist.Distribution,
        *,
        mass_center: float,
        mass_scale: float,
        size_center: float,
        size_scale: float,
        validate_args: bool | None = None,
    ) -> None:
        self.sigma_distribution = sigma_distribution
        self.cut_distribution = cut_distribution
        self.mass_center = jnp.asarray(float(mass_center), dtype=jnp.float64)
        self.mass_scale = jnp.asarray(float(mass_scale), dtype=jnp.float64)
        self.size_center = jnp.asarray(float(size_center), dtype=jnp.float64)
        self.size_scale = jnp.asarray(float(size_scale), dtype=jnp.float64)
        self.log_abs_det_jacobian = jnp.log(jnp.abs(0.5 * self.mass_scale * self.size_scale))
        super().__init__(batch_shape=(), event_shape=(2,), validate_args=validate_args)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jnp.ndarray:
        sigma_key, cut_key = jax.random.split(key)
        log_sigma = self.sigma_distribution.sample(sigma_key, sample_shape=sample_shape)
        log_cut_gap = self.cut_distribution.sample(cut_key, sample_shape=sample_shape)
        mass_raw = 2.0 * log_sigma + log_cut_gap
        u_m = (mass_raw - self.mass_center) / self.mass_scale
        u_s = (log_cut_gap - self.size_center) / self.size_scale
        return jnp.stack([u_m, u_s], axis=-1)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value_array = jnp.asarray(value, dtype=jnp.float64)
        u_m = value_array[..., 0]
        u_s = value_array[..., 1]
        log_cut_gap = self.size_center + self.size_scale * u_s
        mass_raw = self.mass_center + self.mass_scale * u_m
        log_sigma = 0.5 * (mass_raw - log_cut_gap)
        return (
            self.sigma_distribution.log_prob(log_sigma)
            + self.cut_distribution.log_prob(log_cut_gap)
            + self.log_abs_det_jacobian
        )


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
        if flag == 9 and len(value) >= 2:
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
    if mode == 9:
        if len(value) < 4:
            raise ValueError(f"Truncated normal prior for {context} requires mean, std, and lower bound.")
        mean = _coerce_numeric(value[1], f"{context}.mean")
        std = _coerce_numeric(value[2], f"{context}.std")
        lower = _coerce_numeric(value[3], f"{context}.lower")
        upper = _coerce_numeric(value[4], f"{context}.upper") if len(value) >= 5 else float("inf")
        if std <= 0.0:
            raise ValueError(f"Truncated normal prior std must be positive for {context}.")
        if np.isfinite(float(upper)) and lower >= upper:
            raise ValueError(f"Truncated normal prior lower bound must be less than upper bound for {context}.")
        return {
            "prior_kind": "truncated_normal",
            "lower": lower,
            "upper": upper,
            "step": std,
            "mean": mean,
            "std": std,
        }
    raise ValueError(f"Unsupported prior mode {mode} for {context}; expected 0, 1, 3, or 9.")


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


def _log_positive_bound(value: float, floor: float) -> float:
    if not np.isfinite(value):
        return float(value)
    return float(np.log(max(float(value), float(floor))))


def _transform_positive_prior_to_log_space(
    prior_kind: str,
    lower: float,
    upper: float,
    mean: float | None,
    std: float | None,
    *,
    floor: float,
    context: str,
) -> tuple[float, float, float | None, float | None]:
    if prior_kind in {"normal", "truncated_normal"}:
        if mean is None or std is None:
            raise ValueError(f"Positive prior for {context} requires mean/std.")
        safe_mean_floor = max(float(lower), float(floor)) if np.isfinite(lower) else float(floor)
        latent_mean, latent_std = _positive_lognormal_parameters(max(float(mean), safe_mean_floor), float(std), floor=floor)
        if prior_kind == "normal":
            return float("-inf"), float("inf"), latent_mean, latent_std
        return _log_positive_bound(lower, floor), _log_positive_bound(upper, floor), latent_mean, latent_std
    safe_lower = max(float(lower), float(floor))
    safe_upper = max(float(upper), safe_lower * (1.0 + 1.0e-9)) if np.isfinite(upper) else float("inf")
    return _log_positive_bound(safe_lower, floor), _log_positive_bound(safe_upper, floor), mean, std


def _transform_offset_positive_prior_to_log_space(
    prior_kind: str,
    lower: float,
    upper: float,
    mean: float | None,
    std: float | None,
    *,
    offset: float,
    floor: float,
    context: str,
) -> tuple[float, float, float | None, float | None]:
    offset = float(offset)
    gap_lower = max(float(lower) - offset, float(floor)) if np.isfinite(lower) else float(floor)
    if np.isfinite(upper):
        gap_upper = max(float(upper) - offset, gap_lower * (1.0 + 1.0e-9))
    else:
        gap_upper = float("inf")
    if prior_kind in {"normal", "truncated_normal"}:
        if mean is None or std is None:
            raise ValueError(f"Offset-positive prior for {context} requires mean/std.")
        gap_mean = max(float(mean) - offset, gap_lower)
        latent_mean, latent_std = _positive_lognormal_parameters(gap_mean, float(std), floor=gap_lower)
        if prior_kind == "normal":
            return float("-inf"), float("inf"), latent_mean, latent_std
        return _log_positive_bound(gap_lower, floor), _log_positive_bound(gap_upper, floor), latent_mean, latent_std
    return _log_positive_bound(gap_lower, floor), _log_positive_bound(gap_upper, floor), mean, std


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
        context = f"{potential.get('id', 'potential')}.{field_name}"
        lower, upper, mean, std = _transform_positive_prior_to_log_space(
            prior_kind,
            lower,
            upper,
            mean,
            std,
            floor=SAFE_RADIUS_MARGIN_KPC,
            context=context,
        )
        if prior_kind == "uniform":
            step = float(max(step, SAFE_RADIUS_MARGIN_KPC))
    elif field_name == "cut_radius_kpc":
        core_radius = _coerce_numeric(potential.get("core_radius_kpc", SAFE_RADIUS_MARGIN_KPC), "core_radius_kpc")
        transform_kind = "log_offset_positive"
        transform_offset = max(float(core_radius), 0.0)
        context = f"{potential.get('id', 'potential')}.{field_name}"
        lower, upper, mean, std = _transform_offset_positive_prior_to_log_space(
            prior_kind,
            lower,
            upper,
            mean,
            std,
            offset=transform_offset,
            floor=SAFE_RADIUS_MARGIN_KPC,
            context=context,
        )
        if prior_kind == "uniform":
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


def _latent_prior_center_scale(spec: ParameterSpec) -> tuple[float, float]:
    if spec.prior_kind == "uniform" and np.isfinite(float(spec.lower)) and np.isfinite(float(spec.upper)):
        lower = float(spec.lower)
        upper = float(spec.upper)
        width = max(upper - lower, 1.0e-12)
        return 0.5 * (lower + upper), width / math.sqrt(12.0)
    if spec.mean is not None and spec.std is not None and np.isfinite(float(spec.mean)) and np.isfinite(float(spec.std)):
        return float(spec.mean), max(float(spec.std), 1.0e-12)
    if np.isfinite(float(spec.lower)) and np.isfinite(float(spec.upper)):
        lower = float(spec.lower)
        upper = float(spec.upper)
        width = max(upper - lower, 1.0e-12)
        return 0.5 * (lower + upper), width / math.sqrt(12.0)
    return 0.0, 1.0


def _apply_potfile_mass_size_reparameterization(
    specs: list[ParameterSpec],
    field_index: dict[str, int],
    *,
    start_index: int,
    potfile_id: str,
) -> None:
    if "sigma" not in field_index or "cutkpc" not in field_index:
        return
    sigma_list_index = int(field_index["sigma"]) - int(start_index)
    cut_list_index = int(field_index["cutkpc"]) - int(start_index)
    if sigma_list_index < 0 or cut_list_index < 0:
        return
    sigma_spec = specs[sigma_list_index]
    cut_spec = specs[cut_list_index]
    sigma_center, sigma_scale = _latent_prior_center_scale(sigma_spec)
    cut_center, cut_scale = _latent_prior_center_scale(cut_spec)
    mass_center = 2.0 * sigma_center + cut_center
    mass_scale = max(math.sqrt(4.0 * sigma_scale * sigma_scale + cut_scale * cut_scale), 1.0e-12)
    size_center = cut_center
    size_scale = max(cut_scale, 1.0e-12)
    site_name = _sample_name(potfile_id, "mass_size")
    group_name = f"{potfile_id}.mass_size"
    common = {
        "sample_site_name": site_name,
        "coupled_transform_kind": COUPLED_TRANSFORM_POTFILE_MASS_SIZE,
        "coupled_group": group_name,
        "coupled_mass_center": float(mass_center),
        "coupled_mass_scale": float(mass_scale),
        "coupled_size_center": float(size_center),
        "coupled_size_scale": float(size_scale),
    }
    specs[sigma_list_index] = replace(
        sigma_spec,
        sample_site_index=0,
        coupled_role=COUPLED_ROLE_MASS_NORM,
        **common,
    )
    specs[cut_list_index] = replace(
        cut_spec,
        sample_site_index=1,
        coupled_role=COUPLED_ROLE_SIZE,
        **common,
    )


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
            if profile_type == DP_IE_PROFILE and normalized_field_name in {"core_radius_kpc", "cut_radius_kpc"}:
                prior_spec = _radius_transform_for_component_prior(potential, normalized_field_name, decoded_prior)
            elif (
                profile_type == DP_IE_PROFILE
                and normalized_field_name == "v_disp"
                and str(decoded_prior["prior_kind"]) in {"normal", "truncated_normal"}
            ):
                decoded_prior_kind = str(decoded_prior["prior_kind"])
                lower = (
                    VDISP_TRUNCATION_FLOOR_KM_S
                    if decoded_prior_kind == "normal"
                    else float(decoded_prior["lower"])
                )
                upper = float("inf") if decoded_prior_kind == "normal" else float(decoded_prior["upper"])
                prior_spec = {
                    "prior_kind": "truncated_normal",
                    "lower": lower,
                    "upper": upper,
                    "step": float(decoded_prior["step"]),
                    "mean": float(decoded_prior["mean"]),
                    "std": float(decoded_prior["std"]),
                    "transform_kind": "identity",
                    "transform_offset": 0.0,
                    "physical_lower": lower,
                    "physical_upper": None if not np.isfinite(upper) else upper,
                    "physical_mean": float(decoded_prior["mean"]),
                    "physical_std": float(decoded_prior["std"]),
                }
            else:
                prior_spec = {
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
    potfile_mass_size_reparam: bool = False,
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
                lower, upper, mean, std = _transform_positive_prior_to_log_space(
                    prior_kind,
                    lower,
                    upper,
                    mean,
                    std,
                    floor=1.0e-12,
                    context=f"{potfile_id}.{field_name}",
                )
            elif field_name == "cutkpc":
                transform_kind = "log_offset_positive"
                transform_offset = core_radius_kpc
                lower, upper, mean, std = _transform_offset_positive_prior_to_log_space(
                    prior_kind,
                    lower,
                    upper,
                    mean,
                    std,
                    offset=transform_offset,
                    floor=1.0e-9,
                    context=f"{potfile_id}.{field_name}",
                )
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
        if bool(potfile_mass_size_reparam):
            _apply_potfile_mass_size_reparameterization(
                specs,
                field_index,
                start_index=start_index,
                potfile_id=potfile_id,
            )
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
    del scatter_max
    prior_median = float(DEFAULT_SCALING_SCATTER_PRIOR_MEDIAN)
    prior_log_sigma = float(DEFAULT_SCALING_SCATTER_PRIOR_LOG_SIGMA)
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
                    prior_kind="normal",
                    lower=float("-inf"),
                    upper=float("inf"),
                    step=0.1,
                    mean=float(np.log(prior_median)),
                    std=prior_log_sigma,
                    component_family="scaling_scatter",
                    transform_kind="log_positive",
                    physical_lower=0.0,
                    physical_upper=None,
                    physical_mean=prior_median,
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
        if spec.prior_kind == "truncated_normal":
            prior_kind = "truncated_normal"
            lower = (
                max(float(spec.lower), VDISP_TRUNCATION_FLOOR_KM_S)
                if spec.field == "v_disp" and spec.transform_kind == "identity"
                else float(spec.lower)
            )
            upper = float(spec.upper)
            if spec.transform_kind == "identity":
                mean = physical_mean
                std = max(physical_std, 1.0e-12)
            else:
                mean, std = _physical_normal_moments_to_latent(spec, physical_mean, physical_std)
                std = max(std, 1.0e-12)
        else:
            prior_kind = "normal"
            lower = float(spec.lower)
            upper = float(spec.upper)
            mean, std = _physical_normal_moments_to_latent(spec, physical_mean, physical_std)
            if np.isfinite(spec.lower) and np.isfinite(spec.upper):
                width_scale = abs(spec.upper - spec.lower)
            else:
                width_scale = abs(float(spec.std or 0.0))
            floor = max(1.0e-3, 0.05 * max(width_scale, 1.0))
            std = max(std, floor)
        stage2_specs.append(
            ParameterSpec(
                name=spec.name,
                sample_name=spec.sample_name,
                potential_id=spec.potential_id,
                profile_type=spec.profile_type,
                field=spec.field,
                prior_kind=prior_kind,
                lower=lower,
                upper=upper,
                step=spec.step,
                mean=mean,
                std=std,
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


def _build_image_scatter_parameter_spec(
    start_index: int,
    upper_arcsec: float,
    floor_arcsec: float = DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC,
    prior: str = DEFAULT_IMAGE_PLANE_SCATTER_PRIOR,
    prior_median_arcsec: float = DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC,
    prior_log_sigma: float = DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA,
) -> ParameterSpec:
    del start_index
    lower = float(floor_arcsec)
    upper = float(upper_arcsec)
    if not np.isfinite(lower) or lower <= 0.0:
        raise ValueError("image scatter floor must be finite and positive.")
    if not np.isfinite(upper) or upper <= lower:
        raise ValueError("image scatter upper must be greater than image scatter floor.")
    prior_kind = str(prior)
    if prior_kind == IMAGE_PLANE_SCATTER_PRIOR_LOGNORMAL:
        median = float(prior_median_arcsec)
        if not np.isfinite(median) or not (lower < median < upper):
            raise ValueError("lognormal image scatter median must be between floor and upper.")
        log_sigma = float(prior_log_sigma)
        if not np.isfinite(log_sigma) or log_sigma <= 0.0:
            raise ValueError("lognormal image scatter prior log sigma must be positive.")
        return ParameterSpec(
            name="image.sigma_int",
            sample_name="image_sigma_int",
            potential_id="image",
            profile_type=0,
            field="sigma_int",
            prior_kind="truncated_normal",
            lower=float(np.log(lower)),
            upper=float(np.log(upper)),
            step=0.05,
            mean=float(np.log(median)),
            std=float(log_sigma),
            component_family="image_scatter",
            transform_kind="log_positive",
            physical_lower=lower,
            physical_upper=upper,
            physical_mean=median,
            physical_std=None,
        )
    return ParameterSpec(
        name="image.sigma_int",
        sample_name="image_sigma_int",
        potential_id="image",
        profile_type=0,
        field="sigma_int",
        prior_kind="uniform",
        lower=float(np.log(lower)),
        upper=float(np.log(upper)),
        step=0.05,
        component_family="image_scatter",
        transform_kind="log_positive",
        physical_lower=lower,
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
    ellipticite = _axis_ratio_to_lenstool_ellipticite(q)
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


def _has_arc_constraints_in_state(state: BuildState) -> bool:
    arc_data = getattr(state, "arc_data", None)
    if arc_data is not None and int(getattr(arc_data, "n_arcs", 0)) > 0:
        return True
    return False


def _prepare_arc_constraint_data(
    arcs_df: pd.DataFrame,
    reference: tuple[int, float, float],
) -> ArcConstraintData | None:
    if arcs_df.empty:
        return None
    _, ra0_deg, dec0_deg = reference
    required_columns = [
        "arc_id",
        "arc_anchor_ra",
        "arc_anchor_dec",
        "z_arc",
        "arc_tangent_angle_rad",
        "arc_curvature_arcsec_inv",
        "arc_sigma_tangent_angle_rad",
        "arc_sigma_curvature_arcsec_inv",
        "arc_reliability",
    ]
    missing_columns = [column for column in required_columns if column not in arcs_df.columns]
    if missing_columns:
        raise ValueError(f"Arc catalog is missing required columns {missing_columns}.")
    arc_ids = arcs_df["arc_id"].astype(str).tolist()
    bad_ids = [arc_id for arc_id in arc_ids if not arc_id or arc_id.strip() != arc_id or any(ch.isspace() for ch in arc_id)]
    if bad_ids:
        raise ValueError(f"Arc catalog has invalid arc_id values {bad_ids}; IDs must be non-empty and contain no whitespace.")
    duplicate_ids = sorted(arcs_df.loc[arcs_df["arc_id"].astype(str).duplicated(keep=False), "arc_id"].astype(str).unique().tolist())
    if duplicate_ids:
        raise ValueError(f"Arc catalog has duplicate arc_id values {duplicate_ids}.")
    numeric_columns = [column for column in required_columns if column != "arc_id"]
    numeric = {
        column: pd.to_numeric(arcs_df[column], errors="coerce").to_numpy(dtype=float)
        for column in numeric_columns
    }
    finite_mask = np.ones(len(arcs_df), dtype=bool)
    for values in numeric.values():
        finite_mask &= np.isfinite(values)
    if not bool(np.all(finite_mask)):
        bad_ids = arcs_df.loc[~finite_mask, "arc_id"].astype(str).tolist()
        raise ValueError(f"Arc catalog has non-finite numeric values for arc IDs {bad_ids}.")
    z_arc = numeric["z_arc"]
    bad_z = ~((z_arc >= 0.0) | (z_arc == -1.0))
    if bool(np.any(bad_z)):
        bad_ids = arcs_df.loc[bad_z, "arc_id"].astype(str).tolist()
        raise ValueError(f"Arc catalog has invalid z_arc values for arc IDs {bad_ids}; use z_arc >= 0 or -1.")
    sigma_tangent = numeric["arc_sigma_tangent_angle_rad"]
    sigma_curvature = numeric["arc_sigma_curvature_arcsec_inv"]
    if bool(np.any(sigma_tangent <= 0.0) or np.any(sigma_curvature <= 0.0)):
        bad_ids = arcs_df.loc[(sigma_tangent <= 0.0) | (sigma_curvature <= 0.0), "arc_id"].astype(str).tolist()
        raise ValueError(f"Arc catalog has non-positive sigmas for arc IDs {bad_ids}.")
    if bool(np.any(numeric["arc_curvature_arcsec_inv"] < 0.0)):
        bad_ids = arcs_df.loc[numeric["arc_curvature_arcsec_inv"] < 0.0, "arc_id"].astype(str).tolist()
        raise ValueError(f"Arc catalog has negative curvature magnitudes for arc IDs {bad_ids}.")
    anchor_x, anchor_y = _radec_to_offsets_arcsec(
        numeric["arc_anchor_ra"],
        numeric["arc_anchor_dec"],
        ra0_deg,
        dec0_deg,
    )
    return ArcConstraintData(
        arc_ids=arc_ids,
        z_arc=np.asarray(z_arc, dtype=float),
        anchor_x=np.asarray(anchor_x, dtype=float),
        anchor_y=np.asarray(anchor_y, dtype=float),
        tangent_angle_rad=np.asarray(numeric["arc_tangent_angle_rad"], dtype=float),
        curvature_arcsec_inv=np.asarray(numeric["arc_curvature_arcsec_inv"], dtype=float),
        sigma_tangent_angle_rad=np.asarray(sigma_tangent, dtype=float),
        sigma_curvature_arcsec_inv=np.asarray(sigma_curvature, dtype=float),
        reliability=np.asarray(np.clip(numeric["arc_reliability"], 0.0, 1.0), dtype=float),
    )


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
    legacy_arc_columns = [
        str(column)
        for column in images_df.columns
        if str(column) == "arc_has_constraint" or str(column).startswith("arc_")
    ]
    if legacy_arc_columns:
        raise ValueError(
            "Image catalog contains legacy CAB arc columns "
            f"{legacy_arc_columns}; declare CAB morphology constraints in image.arcfile instead."
        )
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
        sampling_engine: str = "full",
        active_scaling_galaxies: list[int] | int | None = None,
        active_scaling_selection: str = "adaptive",
        active_scaling_cumulative_fraction: float = DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        active_scaling_min: int = DEFAULT_ACTIVE_SCALING_MIN,
        refresh_every: int = DEFAULT_REFRESH_EVERY,
        refresh_param_drift_frac: float = DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        source_plane_covariance_floor: float = 1.0e-6,
        source_plane_outlier_sigma_arcsec: float = DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC,
        sample_likelihood_mode: str = SAMPLE_LIKELIHOOD_SOURCE,
        image_plane_newton_steps: int = 0,
        anchored_image_plane_solve_steps: int = DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS,
        anchored_image_plane_trust_radius_arcsec: float = DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
        anchored_image_plane_lm_damping_relative: float = DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
        anchored_image_plane_lm_damping_absolute: float = DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
        critical_arc_critical_direction_sigma_arcsec: float = DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC,
        critical_arc_base_prob: float = DEFAULT_CRITICAL_ARC_BASE_PROB,
        critical_arc_max_prob: float = DEFAULT_CRITICAL_ARC_MAX_PROB,
        critical_arc_singular_threshold: float = DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD,
        critical_arc_singular_softness: float = DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS,
        critical_arc_lm_damping_relative: float = DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE,
        critical_arc_lm_damping_absolute: float = DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE,
        critical_arc_lm_trust_radius_arcsec: float = DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC,
        arc_aware_noncritical_support_radius_arcsec: float = DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC,
        arc_aware_max_arclength_arcsec: float = DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC,
        arc_aware_curve_step_arcsec: float = DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC,
        fold_curvature_arcsec_inv: float = DEFAULT_FOLD_CURVATURE_ARCSEC_INV,
        cab_likelihood_weight: float = DEFAULT_CAB_LIKELIHOOD_WEIGHT_NO_ARCS,
        cab_finite_difference_step_arcsec: float = DEFAULT_CAB_FINITE_DIFFERENCE_STEP_ARCSEC,
        cab_tangent_sigma_floor_rad: float = DEFAULT_CAB_TANGENT_SIGMA_FLOOR_RAD,
        cab_curvature_sigma_floor_arcsec_inv: float = DEFAULT_CAB_CURVATURE_SIGMA_FLOOR_ARCSEC_INV,
        image_plane_scatter_floor_arcsec: float = DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC,
        fixed_image_sigma_int_arcsec: float | None = None,
        image_presence_penalty_weight: float = 0.0,
        image_presence_match_radius_arcsec: float = DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC,
        image_presence_temperature_arcsec: float = DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC,
        image_presence_count_softness: float = DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS,
        image_presence_count_margin: float = DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN,
        likelihood_stabilizer_max_gain: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN,
        likelihood_stabilizer_max_residual_arcsec: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC,
        likelihood_stabilizer_residual_loss: str = DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS,
        likelihood_stabilizer_student_t_nu: float = DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU,
        evidence_source_prior_sigma_arcsec: float | None = None,
        evidence_source_prior_mean_x_arcsec: float = 0.0,
        evidence_source_prior_mean_y_arcsec: float = 0.0,
        quick_diagnostics: bool = False,
    ):
        self.state = state
        self.match_tolerance_arcsec = float(match_tolerance_arcsec)
        self.sampling_engine = str(sampling_engine)
        if self.sampling_engine not in SAMPLING_ENGINES:
            raise ValueError(
                f"Unsupported sampling_engine={self.sampling_engine!r}; "
                f"expected one of {', '.join(SAMPLING_ENGINES)}."
            )
        self.active_scaling_galaxies_by_potfile = _normalize_active_scaling_counts(active_scaling_galaxies, state.potfiles)
        self.active_scaling_selection = str(active_scaling_selection)
        self.active_scaling_cumulative_fraction = float(active_scaling_cumulative_fraction)
        self.active_scaling_min = max(1, int(active_scaling_min))
        self.refresh_every = max(1, int(refresh_every))
        self.refresh_param_drift_frac = float(refresh_param_drift_frac)
        self.source_plane_covariance_floor = max(float(source_plane_covariance_floor), 0.0)
        self.source_plane_outlier_sigma_arcsec = max(float(source_plane_outlier_sigma_arcsec), 1.0e-6)
        self.sample_likelihood_mode = str(sample_likelihood_mode)
        if self.sample_likelihood_mode not in {
            SAMPLE_LIKELIHOOD_SOURCE,
            SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
            SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
            SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
            SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
            SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
            SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
        }:
            raise ValueError(f"Unsupported sample_likelihood_mode={self.sample_likelihood_mode!r}.")
        requested_image_plane_newton_steps = int(image_plane_newton_steps)
        if (
            self.sample_likelihood_mode
            in {
                SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
                SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
                SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
                SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
            }
            and requested_image_plane_newton_steps != 0
        ):
            raise ValueError(f"{self.sample_likelihood_mode} requires image_plane_newton_steps=0.")
        if (
            self.sampling_engine == SAMPLING_ENGINE_REFRESHING_SURROGATE
            and self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
            and requested_image_plane_newton_steps > 0
        ):
            raise ValueError(
                "refreshing_surrogate with linearized-forward-beta-image-plane requires "
                "image_plane_newton_steps=0; Newton updates move image positions away from the observed-position cache."
            )
        self.image_plane_newton_steps = max(0, min(3, requested_image_plane_newton_steps))
        requested_anchored_steps = int(anchored_image_plane_solve_steps)
        if requested_anchored_steps < 0:
            raise ValueError("anchored_image_plane_solve_steps must be non-negative.")
        if (
            self.sampling_engine == SAMPLING_ENGINE_REFRESHING_SURROGATE
            and self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE
            and requested_anchored_steps > 0
        ):
            raise ValueError(
                "anchored-solved-forward-beta-image-plane with refreshing_surrogate requires "
                "anchored_image_plane_solve_steps=0."
            )
        self.anchored_image_plane_solve_steps = requested_anchored_steps
        self.anchored_image_plane_trust_radius_arcsec = float(anchored_image_plane_trust_radius_arcsec)
        self.anchored_image_plane_lm_damping_relative = float(anchored_image_plane_lm_damping_relative)
        self.anchored_image_plane_lm_damping_absolute = float(anchored_image_plane_lm_damping_absolute)
        if (
            not np.isfinite(self.anchored_image_plane_trust_radius_arcsec)
            or self.anchored_image_plane_trust_radius_arcsec <= 0.0
        ):
            raise ValueError("anchored_image_plane_trust_radius_arcsec must be finite and positive.")
        if (
            not np.isfinite(self.anchored_image_plane_lm_damping_relative)
            or self.anchored_image_plane_lm_damping_relative <= 0.0
        ):
            raise ValueError("anchored_image_plane_lm_damping_relative must be finite and positive.")
        if (
            not np.isfinite(self.anchored_image_plane_lm_damping_absolute)
            or self.anchored_image_plane_lm_damping_absolute <= 0.0
        ):
            raise ValueError("anchored_image_plane_lm_damping_absolute must be finite and positive.")
        self.critical_arc_critical_direction_sigma_arcsec = float(critical_arc_critical_direction_sigma_arcsec)
        self.critical_arc_base_prob = float(critical_arc_base_prob)
        self.critical_arc_max_prob = float(critical_arc_max_prob)
        self.critical_arc_singular_threshold = float(critical_arc_singular_threshold)
        self.critical_arc_singular_softness = float(critical_arc_singular_softness)
        self.critical_arc_lm_damping_relative = float(critical_arc_lm_damping_relative)
        self.critical_arc_lm_damping_absolute = float(critical_arc_lm_damping_absolute)
        self.critical_arc_lm_trust_radius_arcsec = float(critical_arc_lm_trust_radius_arcsec)
        self.arc_aware_noncritical_support_radius_arcsec = float(arc_aware_noncritical_support_radius_arcsec)
        self.arc_aware_max_arclength_arcsec = float(arc_aware_max_arclength_arcsec)
        self.arc_aware_curve_step_arcsec = float(arc_aware_curve_step_arcsec)
        self.fold_curvature_arcsec_inv = float(fold_curvature_arcsec_inv)
        self.cab_likelihood_weight = float(cab_likelihood_weight)
        self.cab_finite_difference_step_arcsec = float(cab_finite_difference_step_arcsec)
        self.cab_tangent_sigma_floor_rad = float(cab_tangent_sigma_floor_rad)
        self.cab_curvature_sigma_floor_arcsec_inv = float(cab_curvature_sigma_floor_arcsec_inv)
        if (
            not np.isfinite(self.critical_arc_critical_direction_sigma_arcsec)
            or self.critical_arc_critical_direction_sigma_arcsec <= 0.0
        ):
            raise ValueError("critical_arc_critical_direction_sigma_arcsec must be finite and positive.")
        if (
            not np.isfinite(self.critical_arc_base_prob)
            or not np.isfinite(self.critical_arc_max_prob)
            or self.critical_arc_base_prob < 0.0
            or self.critical_arc_max_prob > 1.0
            or self.critical_arc_base_prob > self.critical_arc_max_prob
        ):
            raise ValueError("critical arc branch probabilities must satisfy 0 <= base <= max <= 1.")
        if (
            not np.isfinite(self.critical_arc_singular_threshold)
            or self.critical_arc_singular_threshold <= 0.0
        ):
            raise ValueError("critical_arc_singular_threshold must be finite and positive.")
        if (
            not np.isfinite(self.critical_arc_singular_softness)
            or self.critical_arc_singular_softness <= 0.0
        ):
            raise ValueError("critical_arc_singular_softness must be finite and positive.")
        if (
            not np.isfinite(self.critical_arc_lm_damping_relative)
            or self.critical_arc_lm_damping_relative <= 0.0
        ):
            raise ValueError("critical_arc_lm_damping_relative must be finite and positive.")
        if (
            not np.isfinite(self.critical_arc_lm_damping_absolute)
            or self.critical_arc_lm_damping_absolute <= 0.0
        ):
            raise ValueError("critical_arc_lm_damping_absolute must be finite and positive.")
        if (
            not np.isfinite(self.critical_arc_lm_trust_radius_arcsec)
            or self.critical_arc_lm_trust_radius_arcsec <= 0.0
        ):
            raise ValueError("critical_arc_lm_trust_radius_arcsec must be finite and positive.")
        if (
            not np.isfinite(self.arc_aware_noncritical_support_radius_arcsec)
            or self.arc_aware_noncritical_support_radius_arcsec <= 0.0
        ):
            raise ValueError("arc_aware_noncritical_support_radius_arcsec must be finite and positive.")
        if not np.isfinite(self.arc_aware_max_arclength_arcsec) or self.arc_aware_max_arclength_arcsec <= 0.0:
            raise ValueError("arc_aware_max_arclength_arcsec must be finite and positive.")
        if not np.isfinite(self.arc_aware_curve_step_arcsec) or self.arc_aware_curve_step_arcsec <= 0.0:
            raise ValueError("arc_aware_curve_step_arcsec must be finite and positive.")
        if (
            not np.isfinite(self.fold_curvature_arcsec_inv)
            or self.fold_curvature_arcsec_inv <= 0.0
        ):
            raise ValueError("fold_curvature_arcsec_inv must be finite and positive.")
        if not np.isfinite(self.cab_likelihood_weight) or self.cab_likelihood_weight < 0.0:
            raise ValueError("cab_likelihood_weight must be finite and non-negative.")
        if not np.isfinite(self.cab_finite_difference_step_arcsec) or self.cab_finite_difference_step_arcsec <= 0.0:
            raise ValueError("cab_finite_difference_step_arcsec must be finite and positive.")
        if not np.isfinite(self.cab_tangent_sigma_floor_rad) or self.cab_tangent_sigma_floor_rad <= 0.0:
            raise ValueError("cab_tangent_sigma_floor_rad must be finite and positive.")
        if (
            not np.isfinite(self.cab_curvature_sigma_floor_arcsec_inv)
            or self.cab_curvature_sigma_floor_arcsec_inv <= 0.0
        ):
            raise ValueError("cab_curvature_sigma_floor_arcsec_inv must be finite and positive.")
        image_scatter_floor = float(image_plane_scatter_floor_arcsec)
        if not np.isfinite(image_scatter_floor) or image_scatter_floor <= 0.0:
            raise ValueError("image_plane_scatter_floor_arcsec must be finite and positive.")
        self.image_plane_scatter_floor_arcsec = image_scatter_floor
        if fixed_image_sigma_int_arcsec is None:
            self.fixed_image_sigma_int_arcsec = None
        else:
            fixed_sigma = float(fixed_image_sigma_int_arcsec)
            if not np.isfinite(fixed_sigma) or fixed_sigma < 0.0:
                raise ValueError("fixed_image_sigma_int_arcsec must be finite and nonnegative.")
            self.fixed_image_sigma_int_arcsec = fixed_sigma
        self.image_presence_penalty_weight = max(float(image_presence_penalty_weight), 0.0)
        self.image_presence_match_radius_arcsec = max(float(image_presence_match_radius_arcsec), 1.0e-12)
        self.image_presence_temperature_arcsec = max(float(image_presence_temperature_arcsec), 1.0e-12)
        self.image_presence_count_softness = max(float(image_presence_count_softness), 1.0e-12)
        self.image_presence_count_margin = max(float(image_presence_count_margin), 0.0)
        self.likelihood_stabilizer_max_gain = max(float(likelihood_stabilizer_max_gain), 0.0)
        self.likelihood_stabilizer_max_residual_arcsec = max(float(likelihood_stabilizer_max_residual_arcsec), 0.0)
        self.likelihood_stabilizer_residual_loss = str(likelihood_stabilizer_residual_loss)
        if self.likelihood_stabilizer_residual_loss not in LIKELIHOOD_STABILIZER_RESIDUAL_LOSSES:
            raise ValueError(f"Unsupported likelihood_stabilizer_residual_loss={self.likelihood_stabilizer_residual_loss!r}.")
        self.likelihood_stabilizer_student_t_nu = max(float(likelihood_stabilizer_student_t_nu), 1.0e-6)
        self.evidence_source_prior_sigma_arcsec = (
            None if evidence_source_prior_sigma_arcsec is None else float(evidence_source_prior_sigma_arcsec)
        )
        self.evidence_source_prior_mean_x_arcsec = float(evidence_source_prior_mean_x_arcsec)
        self.evidence_source_prior_mean_y_arcsec = float(evidence_source_prior_mean_y_arcsec)
        self.quick_diagnostics = bool(quick_diagnostics)
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
        if getattr(state.arc_data, "n_arcs", 0):
            self.models_by_effective_z[float(CAB_MORPHOLOGY_MODEL_KEY)] = LensModelBulk(
                unique_lens_model_list=unique_lens_model_list,
                multi_plane=False,
                cosmo=self.cosmo,
            )
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
        self.exact_lenstronomy_count = 0
        self.timing_totals["geometry_cache_setup"] += time.perf_counter() - geometry_setup_start
        self.traced_bin_data = tuple(self._prepare_traced_bin_data(bin_item) for bin_item in state.bin_data)
        self.traced_bin_data_by_z = {bin_item.effective_z_source: bin_item for bin_item in self.traced_bin_data}
        self.traced_arc_data = self._prepare_traced_arc_constraint_data(state.arc_data)
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
        self.image_sigma_int_sampled = self.fixed_image_sigma_int_arcsec is None and self.image_sigma_int_param_index >= 0
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
        if (
            self.sample_likelihood_mode
            in {
                SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
                SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
                SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
                SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
            }
            and self.source_position_conditional
        ):
            raise ValueError(
                f"{self.sample_likelihood_mode} does not support "
                "source_position_parameterization=conditional-whitened."
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
        coupling_arrays = _potfile_mass_size_coupling_arrays(self.state.parameter_specs)
        self.potfile_mass_size_mass_indices = jnp.asarray(coupling_arrays["mass_indices"], dtype=jnp.int32)
        self.potfile_mass_size_size_indices = jnp.asarray(coupling_arrays["size_indices"], dtype=jnp.int32)
        self.potfile_mass_size_mass_centers = jnp.asarray(coupling_arrays["mass_centers"], dtype=jnp.float64)
        self.potfile_mass_size_mass_scales = jnp.asarray(coupling_arrays["mass_scales"], dtype=jnp.float64)
        self.potfile_mass_size_size_centers = jnp.asarray(coupling_arrays["size_centers"], dtype=jnp.float64)
        self.potfile_mass_size_size_scales = jnp.asarray(coupling_arrays["size_scales"], dtype=jnp.float64)
        self.potfile_mass_size_cut_offsets = jnp.asarray(coupling_arrays["cut_offsets"], dtype=jnp.float64)
        self.potfile_mass_size_reparam_group_count = int(len(coupling_arrays["mass_indices"]))
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
        self.active_scaling_component_indices_jax = jnp.asarray(self.active_scaling_component_indices, dtype=jnp.int32)
        self.active_component_indices_jax = jnp.asarray(self.active_component_indices, dtype=jnp.int32)
        stage4_refreshing_surrogate_supported = (
            self.sample_likelihood_mode
            in {
                SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
                SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
                SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
                SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
                SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
            }
            and self.image_plane_newton_steps == 0
            and (
                self.sample_likelihood_mode != SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE
                or self.anchored_image_plane_solve_steps == 0
            )
        )
        source_plane_refreshing_surrogate_supported = not _sample_likelihood_uses_image_scatter(
            self.sample_likelihood_mode
        )
        self.surrogate_enabled = (
            self.sampling_engine == SAMPLING_ENGINE_REFRESHING_SURROGATE
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

    def _active_subset_effective(self) -> bool:
        return (
            str(getattr(self, "sampling_engine", SAMPLING_ENGINE_FULL)) == SAMPLING_ENGINE_ACTIVE_SUBSET
            and len(getattr(self, "scaling_component_indices", [])) > 0
            and len(getattr(self, "inactive_scaling_component_indices", [])) > 0
        )

    def _fit_component_indices(self) -> np.ndarray | None:
        if self._active_subset_effective():
            return np.asarray(self.active_component_indices, dtype=np.int32)
        return None

    def _fit_scaling_component_indices(self) -> np.ndarray:
        if self._active_subset_effective():
            return np.asarray(self.active_scaling_component_indices, dtype=np.int32)
        return np.asarray(self.scaling_component_indices, dtype=np.int32)

    def _prepare_traced_bin_data(self, bin_data: BinData) -> TracedBinData:
        family_idx = np.asarray(bin_data.family_index_per_image, dtype=np.int32)
        n_families = len(bin_data.family_ids)
        family_counts = np.zeros(n_families, dtype=np.int32)
        np.add.at(family_counts, family_idx, 1)
        image_has_constraint = family_counts[family_idx] > 1
        n_images = int(len(family_idx))

        return TracedBinData(
            effective_z_source=float(bin_data.effective_z_source),
            family_ids=tuple(str(family_id) for family_id in bin_data.family_ids),
            n_families=int(n_families),
            family_index_per_image=jnp.asarray(family_idx, dtype=jnp.int32),
            x_obs=jnp.asarray(bin_data.x_obs, dtype=jnp.float64),
            y_obs=jnp.asarray(bin_data.y_obs, dtype=jnp.float64),
            sigma_per_image=jnp.asarray(bin_data.sigma_per_image, dtype=jnp.float64),
            reliability_per_image=jnp.asarray(bin_data.reliability_per_image, dtype=jnp.float64),
            image_has_constraint=jnp.asarray(image_has_constraint, dtype=bool),
            effective_z_index=int(self.cosmology_effective_z_to_index.get(float(bin_data.effective_z_source), -1)),
            constrained_image_indices=jnp.asarray(np.flatnonzero(image_has_constraint), dtype=jnp.int32),
        )

    def _prepare_traced_arc_constraint_data(self, arc_data: ArcConstraintData | None) -> TracedArcConstraintData:
        if arc_data is None or int(getattr(arc_data, "n_arcs", 0)) <= 0:
            empty = jnp.asarray([], dtype=jnp.float64)
            return TracedArcConstraintData(
                arc_ids=(),
                z_arc=empty,
                anchor_x=empty,
                anchor_y=empty,
                tangent_angle_rad=empty,
                curvature_arcsec_inv=empty,
                sigma_tangent_angle_rad=empty,
                sigma_curvature_arcsec_inv=empty,
                reliability=empty,
                n_arcs=0,
            )
        n_arcs = int(arc_data.n_arcs)

        def traced_array(field_name: str) -> np.ndarray:
            values = np.asarray(getattr(arc_data, field_name), dtype=float)
            if values.shape[0] != n_arcs:
                raise ValueError(
                    f"ArcConstraintData {field_name} length {values.shape[0]} does not match arc count {n_arcs}."
                )
            return values

        return TracedArcConstraintData(
            arc_ids=tuple(str(arc_id) for arc_id in arc_data.arc_ids),
            z_arc=jnp.asarray(traced_array("z_arc"), dtype=jnp.float64),
            anchor_x=jnp.asarray(traced_array("anchor_x"), dtype=jnp.float64),
            anchor_y=jnp.asarray(traced_array("anchor_y"), dtype=jnp.float64),
            tangent_angle_rad=jnp.asarray(traced_array("tangent_angle_rad"), dtype=jnp.float64),
            curvature_arcsec_inv=jnp.asarray(traced_array("curvature_arcsec_inv"), dtype=jnp.float64),
            sigma_tangent_angle_rad=jnp.asarray(traced_array("sigma_tangent_angle_rad"), dtype=jnp.float64),
            sigma_curvature_arcsec_inv=jnp.asarray(traced_array("sigma_curvature_arcsec_inv"), dtype=jnp.float64),
            reliability=jnp.asarray(traced_array("reliability"), dtype=jnp.float64),
            n_arcs=n_arcs,
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
            self.potfile_mass_size_mass_indices,
            self.potfile_mass_size_size_indices,
            self.potfile_mass_size_mass_centers,
            self.potfile_mass_size_mass_scales,
            self.potfile_mass_size_size_centers,
            self.potfile_mass_size_size_scales,
            self.potfile_mass_size_cut_offsets,
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
        if self.fixed_image_sigma_int_arcsec is not None:
            return jnp.asarray(float(self.fixed_image_sigma_int_arcsec), dtype=jnp.float64)
        if self.image_sigma_int_param_index < 0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        return jnp.take(physical_params, jnp.asarray(self.image_sigma_int_param_index, dtype=jnp.int32))

    def _image_sigma_int_numpy(self, params: np.ndarray | jnp.ndarray) -> float:
        if self.fixed_image_sigma_int_arcsec is not None:
            return float(self.fixed_image_sigma_int_arcsec)
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
        inv00, inv01, inv10, inv11, inverse_finite = _linearized_image_plane_inverse_operator(
            jac_a00,
            jac_a01,
            jac_a10,
            jac_a11,
            determinant_floor=1.0e-10,
            max_gain=self.likelihood_stabilizer_max_gain,
            require_determinant_floor=False,
        )

        sigma2 = _image_plane_effective_sigma2(
            bin_data.sigma_per_image,
            image_sigma_int,
            self.source_plane_covariance_floor,
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
            & jnp.all(inverse_finite)
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
        active_subset_indices = (
            jnp.asarray(self.active_scaling_component_indices, dtype=jnp.int32)
            if self._active_subset_effective()
            else None
        )

        def _field_scale(index_array: np.ndarray) -> jnp.ndarray:
            index_jax = jnp.asarray(index_array, dtype=jnp.int32)
            values = self._apply_param_updates(jnp.zeros_like(spec_jax["luminosity_ratio"]), index_array, physical_params)
            mask = is_scaling & (index_jax >= 0)
            if active_subset_indices is not None:
                component_ids = jnp.arange(mask.shape[0], dtype=jnp.int32)
                active_mask = jnp.any(component_ids[:, None] == active_subset_indices[None, :], axis=1)
                mask = mask & active_mask
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
        if len(self._fit_scaling_component_indices()) == 0 or len(self.scaling_scatter_param_indices) == 0:
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
        if len(self._fit_scaling_component_indices()) == 0 or len(self.scaling_scatter_param_indices) == 0:
            return zeros, zeros, zeros, jnp.ones_like(jac_a00, dtype=bool)
        cache = self.scaling_scatter_cache_by_z.get(float(bin_data.effective_z_source))
        if cache is None:
            return zeros, zeros, zeros, jnp.ones_like(jac_a00, dtype=bool)
        sigma_scatter, core_scatter, cut_scatter = self._scaling_scatter_field_scales_from_physical(physical_params)
        inv00, inv01, inv10, inv11, inverse_finite = _linearized_image_plane_inverse_operator(
            jac_a00,
            jac_a01,
            jac_a10,
            jac_a11,
            max_gain=self.likelihood_stabilizer_max_gain,
            require_determinant_floor=False,
        )

        def _image_displacement_covariance(scale: jnp.ndarray, key_x: str, key_y: str) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            source_dx = scale * jnp.asarray(cache[key_x], dtype=jnp.float64)
            source_dy = scale * jnp.asarray(cache[key_y], dtype=jnp.float64)
            image_dx = inv00 * source_dx + inv01 * source_dy
            image_dy = inv10 * source_dx + inv11 * source_dy
            return jnp.square(image_dx), image_dx * image_dy, jnp.square(image_dy)

        sigma00, sigma01, sigma11 = _image_displacement_covariance(sigma_scatter, "sigma_x", "sigma_y")
        core00, core01, core11 = _image_displacement_covariance(core_scatter, "core_x", "core_y")
        cut00, cut01, cut11 = _image_displacement_covariance(cut_scatter, "cut_x", "cut_y")
        cov00 = sigma00 + core00 + cut00
        cov01 = sigma01 + core01 + cut01
        cov11 = sigma11 + core11 + cut11
        finite = (
            inverse_finite
            & jnp.isfinite(cov00)
            & jnp.isfinite(cov01)
            & jnp.isfinite(cov11)
        )
        return cov00, cov01, cov11, finite

    def refresh_scaling_scatter_cache(self, reference_params: np.ndarray, reason: str = "manual") -> None:
        self.scaling_scatter_cache_by_z = {}
        self.scaling_scatter_reference_params = None
        if len(self._fit_scaling_component_indices()) == 0 or len(self.scaling_scatter_param_indices) == 0:
            return
        reference = np.asarray(reference_params, dtype=float)
        reference_jax = jnp.asarray(reference, dtype=jnp.float64)
        eps = 1.0e-3
        fit_component_indices = self._fit_component_indices()

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
                fit_component_indices,
            )
            deriv_x = np.asarray((beta_plus_x - beta_x) / eps, dtype=float)
            deriv_y = np.asarray((beta_plus_y - beta_y) / eps, dtype=float)
            return deriv_x, deriv_y

        for bin_data in self.traced_bin_data:
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
                fit_component_indices,
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
                self._fit_component_indices(),
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
            coupling_arrays = _potfile_mass_size_coupling_arrays(self.state.parameter_specs)
            potfile_mass_size_mass_indices = jnp.asarray(coupling_arrays["mass_indices"], dtype=jnp.int32)
            potfile_mass_size_size_indices = jnp.asarray(coupling_arrays["size_indices"], dtype=jnp.int32)
            potfile_mass_size_mass_centers = jnp.asarray(coupling_arrays["mass_centers"], dtype=jnp.float64)
            potfile_mass_size_mass_scales = jnp.asarray(coupling_arrays["mass_scales"], dtype=jnp.float64)
            potfile_mass_size_size_centers = jnp.asarray(coupling_arrays["size_centers"], dtype=jnp.float64)
            potfile_mass_size_size_scales = jnp.asarray(coupling_arrays["size_scales"], dtype=jnp.float64)
            potfile_mass_size_cut_offsets = jnp.asarray(coupling_arrays["cut_offsets"], dtype=jnp.float64)
        else:
            transform_kind_log_offset_positive_mask = self.transform_kind_log_offset_positive_mask
            transform_offset_array = self.transform_offset_array
            transform_kind_affine_mask = self.transform_kind_affine_mask
            transform_scale_array = self.transform_scale_array
            potfile_mass_size_mass_indices = self.potfile_mass_size_mass_indices
            potfile_mass_size_size_indices = self.potfile_mass_size_size_indices
            potfile_mass_size_mass_centers = self.potfile_mass_size_mass_centers
            potfile_mass_size_mass_scales = self.potfile_mass_size_mass_scales
            potfile_mass_size_size_centers = self.potfile_mass_size_size_centers
            potfile_mass_size_size_scales = self.potfile_mass_size_size_scales
            potfile_mass_size_cut_offsets = self.potfile_mass_size_cut_offsets
        physical_params = _apply_parameter_transforms_jax(
            params,
            transform_kind_log_positive_mask,
            transform_kind_log_offset_positive_mask,
            transform_offset_array,
            transform_kind_affine_mask,
            transform_scale_array,
            potfile_mass_size_mass_indices,
            potfile_mass_size_size_indices,
            potfile_mass_size_mass_centers,
            potfile_mass_size_mass_scales,
            potfile_mass_size_size_centers,
            potfile_mass_size_size_scales,
            potfile_mass_size_cut_offsets,
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

        q = _lenstool_ellipticite_to_axis_ratio_jax(ellipticite)
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
        sigma0 = (v_disp**2) * safe_factor / ra
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
                jnp.any(details["is_dpie"] & (details["rs_raw"] <= details["ra_raw"])),
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
            coupling_arrays = _potfile_mass_size_coupling_arrays(self.state.parameter_specs)
            potfile_mass_size_mass_indices = jnp.asarray(coupling_arrays["mass_indices"], dtype=jnp.int32)
            potfile_mass_size_size_indices = jnp.asarray(coupling_arrays["size_indices"], dtype=jnp.int32)
            potfile_mass_size_mass_centers = jnp.asarray(coupling_arrays["mass_centers"], dtype=jnp.float64)
            potfile_mass_size_mass_scales = jnp.asarray(coupling_arrays["mass_scales"], dtype=jnp.float64)
            potfile_mass_size_size_centers = jnp.asarray(coupling_arrays["size_centers"], dtype=jnp.float64)
            potfile_mass_size_size_scales = jnp.asarray(coupling_arrays["size_scales"], dtype=jnp.float64)
            potfile_mass_size_cut_offsets = jnp.asarray(coupling_arrays["cut_offsets"], dtype=jnp.float64)
        else:
            transform_kind_log_offset_positive_mask = self.transform_kind_log_offset_positive_mask
            transform_offset_array = self.transform_offset_array
            transform_kind_affine_mask = self.transform_kind_affine_mask
            transform_scale_array = self.transform_scale_array
            potfile_mass_size_mass_indices = self.potfile_mass_size_mass_indices
            potfile_mass_size_size_indices = self.potfile_mass_size_size_indices
            potfile_mass_size_mass_centers = self.potfile_mass_size_mass_centers
            potfile_mass_size_mass_scales = self.potfile_mass_size_mass_scales
            potfile_mass_size_size_centers = self.potfile_mass_size_size_centers
            potfile_mass_size_size_scales = self.potfile_mass_size_size_scales
            potfile_mass_size_cut_offsets = self.potfile_mass_size_cut_offsets
        physical_params = _apply_parameter_transforms_jax(
            params,
            transform_kind_log_positive_mask,
            transform_kind_log_offset_positive_mask,
            transform_offset_array,
            transform_kind_affine_mask,
            transform_scale_array,
            potfile_mass_size_mass_indices,
            potfile_mass_size_size_indices,
            potfile_mass_size_mass_centers,
            potfile_mass_size_mass_scales,
            potfile_mass_size_size_centers,
            potfile_mass_size_size_scales,
            potfile_mass_size_cut_offsets,
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

        q = _lenstool_ellipticite_to_axis_ratio_jax(ellipticite)
        phi = jnp.deg2rad(angle_pos)
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kpc_per_arcsec = self._kpc_per_arcsec_for_physical(physical_params)
        ra_raw = core_radius_kpc / kpc_per_arcsec
        rs_raw = cut_radius_kpc / kpc_per_arcsec
        ra = jnp.maximum(ra_raw, SAFE_RADIUS_MARGIN_ARCSEC)
        rs = jnp.maximum(rs_raw, ra + SAFE_RADIUS_MARGIN_ARCSEC)
        factor_array = self._dpie_sigma0_factor_for_physical_z_source(physical_params, z_source)
        safe_factor = jnp.where(jnp.isfinite(factor_array), factor_array, 0.0)
        sigma0 = (v_disp**2) * safe_factor / ra
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

    def _cab_morphology_predictions_for_anchors(
        self,
        z_source: float,
        anchor_x: jnp.ndarray,
        anchor_y: jnp.ndarray,
        packed_state: dict[str, Any],
        component_indices: np.ndarray | None = None,
    ) -> _CabMorphologyPrediction:
        base_entries = self._lensing_jacobian_for_components(
            z_source,
            anchor_x,
            anchor_y,
            packed_state,
            component_indices,
        )
        base_frame = _cab_tangent_frame_from_jacobian_entries(*base_entries)
        step = jnp.asarray(self.cab_finite_difference_step_arcsec, dtype=jnp.float64)
        n_anchor = int(anchor_x.shape[0])
        probe_x = jnp.concatenate(
            [
                anchor_x + step * base_frame.tangent_x[:, 0],
                anchor_x - step * base_frame.tangent_x[:, 0],
                anchor_x + step * base_frame.tangent_x[:, 1],
                anchor_x - step * base_frame.tangent_x[:, 1],
            ]
        )
        probe_y = jnp.concatenate(
            [
                anchor_y + step * base_frame.tangent_y[:, 0],
                anchor_y - step * base_frame.tangent_y[:, 0],
                anchor_y + step * base_frame.tangent_y[:, 1],
                anchor_y - step * base_frame.tangent_y[:, 1],
            ]
        )
        probe_entries = self._lensing_jacobian_for_components(
            z_source,
            probe_x,
            probe_y,
            packed_state,
            component_indices,
        )
        probe_frame = _cab_tangent_frame_from_jacobian_entries(*probe_entries)
        plus_low_angle = probe_frame.tangent_angle_rad[:n_anchor, 0]
        minus_low_angle = probe_frame.tangent_angle_rad[n_anchor : 2 * n_anchor, 0]
        plus_high_angle = probe_frame.tangent_angle_rad[2 * n_anchor : 3 * n_anchor, 1]
        minus_high_angle = probe_frame.tangent_angle_rad[3 * n_anchor :, 1]
        curvature_low = jnp.abs(_cab_tangent_angle_residual(plus_low_angle, minus_low_angle) / (2.0 * step))
        curvature_high = jnp.abs(_cab_tangent_angle_residual(plus_high_angle, minus_high_angle) / (2.0 * step))
        curvature = jnp.stack([curvature_low, curvature_high], axis=-1)
        plus_low_finite = probe_frame.finite[:n_anchor]
        minus_low_finite = probe_frame.finite[n_anchor : 2 * n_anchor]
        plus_high_finite = probe_frame.finite[2 * n_anchor : 3 * n_anchor]
        minus_high_finite = probe_frame.finite[3 * n_anchor :]
        branch_finite = jnp.stack(
            [
                base_frame.finite & plus_low_finite & minus_low_finite,
                base_frame.finite & plus_high_finite & minus_high_finite,
            ],
            axis=-1,
        )
        branch_frame_weight = jnp.stack(
            [
                base_frame.frame_weight
                * probe_frame.frame_weight[:n_anchor]
                * probe_frame.frame_weight[n_anchor : 2 * n_anchor],
                base_frame.frame_weight
                * probe_frame.frame_weight[2 * n_anchor : 3 * n_anchor]
                * probe_frame.frame_weight[3 * n_anchor :],
            ],
            axis=-1,
        )
        finite = (
            branch_finite
            & jnp.isfinite(curvature)
            & jnp.isfinite(base_frame.tangent_angle_rad)
            & jnp.isfinite(anchor_x)[..., jnp.newaxis]
            & jnp.isfinite(anchor_y)[..., jnp.newaxis]
        )
        return _CabMorphologyPrediction(
            tangent_angle_rad=base_frame.tangent_angle_rad,
            curvature_arcsec_inv=curvature,
            branch_weight=base_frame.branch_weight,
            frame_weight=branch_frame_weight,
            finite=finite,
        )

    def _build_cab_packed_lens_state_with_validity_from_physical(
        self,
        physical_params: jnp.ndarray,
        *,
        stop_gradient: bool,
    ) -> tuple[dict[str, Any], dict[str, jnp.ndarray]]:
        return self._build_packed_lens_state_with_validity_from_physical(
            physical_params,
            float(CAB_MORPHOLOGY_MODEL_KEY),
            stop_gradient=stop_gradient,
            kpc_per_arcsec=self._kpc_per_arcsec_for_physical(physical_params),
            dpie_sigma0_factor=jnp.asarray(1.0, dtype=jnp.float64),
        )

    def _cab_morphology_loglike_for_arcs(
        self,
        arc_data: TracedArcConstraintData,
        packed_state: dict[str, Any],
        component_indices: np.ndarray | None = None,
    ) -> jnp.ndarray:
        if float(self.cab_likelihood_weight) <= 0.0 or int(getattr(arc_data, "n_arcs", 0)) <= 0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        active = jnp.ones((int(arc_data.n_arcs),), dtype=bool)
        prediction = self._cab_morphology_predictions_for_anchors(
            float(CAB_MORPHOLOGY_MODEL_KEY),
            arc_data.anchor_x,
            arc_data.anchor_y,
            packed_state,
            component_indices,
        )
        arc_loglike = _cab_morphology_arc_catalog_loglike(
            predicted_tangent_angle_rad=prediction.tangent_angle_rad,
            predicted_curvature_arcsec_inv=prediction.curvature_arcsec_inv,
            prediction_finite=prediction.finite,
            observed_tangent_angle_rad=arc_data.tangent_angle_rad,
            observed_curvature_arcsec_inv=arc_data.curvature_arcsec_inv,
            sigma_tangent_angle_rad=arc_data.sigma_tangent_angle_rad,
            sigma_curvature_arcsec_inv=arc_data.sigma_curvature_arcsec_inv,
            reliability=arc_data.reliability,
            active_arcs=active,
            tangent_sigma_floor_rad=self.cab_tangent_sigma_floor_rad,
            curvature_sigma_floor_arcsec_inv=self.cab_curvature_sigma_floor_arcsec_inv,
            branch_weight=prediction.branch_weight,
            frame_weight=prediction.frame_weight,
        )
        return jnp.asarray(float(self.cab_likelihood_weight), dtype=jnp.float64) * arc_loglike

    def _fold_signed_curvature_from_observed_jacobian(
        self,
        z_source: float,
        x_obs: jnp.ndarray,
        y_obs: jnp.ndarray,
        packed_state: dict[str, Any],
        observed_jacobian_entries: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        component_indices: np.ndarray | None = None,
        fold_frame: tuple[jnp.ndarray, ...] | None = None,
        image_indices: jnp.ndarray | None = None,
        fill_value: float | jnp.ndarray = DEFAULT_FOLD_CURVATURE_ARCSEC_INV,
    ) -> jnp.ndarray:
        if not self.use_bulk_ray_shooting:
            return jnp.full_like(x_obs, jnp.nan)
        if fold_frame is None:
            fold_frame = _fold_regularized_singular_frame_from_jacobian(*observed_jacobian_entries)
        if image_indices is not None:
            image_indices = jnp.asarray(image_indices, dtype=jnp.int32)
            if int(image_indices.shape[0]) == 0:
                return jnp.full_like(x_obs, jnp.asarray(fill_value, dtype=jnp.float64))

            def take_rows(value: jnp.ndarray) -> jnp.ndarray:
                return jnp.take(value, image_indices, axis=0)

            x_base = take_rows(x_obs)
            y_base = take_rows(y_obs)
            frame_for_probe = tuple(take_rows(value) for value in fold_frame)
        else:
            x_base = x_obs
            y_base = y_obs
            frame_for_probe = fold_frame
        (
            source_critical_x,
            source_critical_y,
            _source_noncritical_x,
            _source_noncritical_y,
            image_critical_x,
            image_critical_y,
            _image_noncritical_x,
            _image_noncritical_y,
            _singular_min,
            _singular_max,
            frame_finite,
        ) = frame_for_probe
        step = jnp.asarray(DEFAULT_FOLD_CURVATURE_FINITE_DIFFERENCE_STEP_ARCSEC, dtype=jnp.float64)
        n_probe = int(x_base.shape[0])
        probe_x = jnp.concatenate(
            [
                x_base + step * image_critical_x,
                x_base - step * image_critical_x,
            ]
        )
        probe_y = jnp.concatenate(
            [
                y_base + step * image_critical_y,
                y_base - step * image_critical_y,
            ]
        )
        packed_entries = self._lensing_jacobian_for_components(
            z_source,
            probe_x,
            probe_y,
            packed_state,
            component_indices,
        )
        plus_entries = tuple(entry[:n_probe] for entry in packed_entries)
        minus_entries = tuple(entry[n_probe:] for entry in packed_entries)
        da00_dcrit = (plus_entries[0] - minus_entries[0]) / (2.0 * step)
        da01_dcrit = (plus_entries[1] - minus_entries[1]) / (2.0 * step)
        da10_dcrit = (plus_entries[2] - minus_entries[2]) / (2.0 * step)
        da11_dcrit = (plus_entries[3] - minus_entries[3]) / (2.0 * step)
        d_a_v_x = da00_dcrit * image_critical_x + da01_dcrit * image_critical_y
        d_a_v_y = da10_dcrit * image_critical_x + da11_dcrit * image_critical_y
        kappa_eff = source_critical_x * d_a_v_x + source_critical_y * d_a_v_y
        finite = frame_finite & jnp.isfinite(kappa_eff)
        kappa_eff = jnp.where(finite, kappa_eff, jnp.full_like(kappa_eff, jnp.nan))
        if image_indices is None:
            return kappa_eff
        return jnp.full_like(x_obs, jnp.asarray(fill_value, dtype=jnp.float64)).at[image_indices].set(kappa_eff)

    def _linearized_image_plane_residuals_for_components(
        self,
        z_source: float,
        x_obs: jnp.ndarray,
        y_obs: jnp.ndarray,
        beta_family_x: jnp.ndarray,
        beta_family_y: jnp.ndarray,
        packed_state: dict[str, Any],
        initial_jacobian_entries: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        current_x = x_obs
        current_y = y_obs
        total_dx = jnp.zeros_like(x_obs)
        total_dy = jnp.zeros_like(y_obs)
        finite = jnp.ones_like(x_obs, dtype=bool)
        for _step in range(1 + self.image_plane_newton_steps):
            if component_indices is None:
                beta_x, beta_y = self._ray_shooting_for_components(z_source, current_x, current_y, packed_state)
            else:
                beta_x, beta_y = self._ray_shooting_for_components(
                    z_source,
                    current_x,
                    current_y,
                    packed_state,
                    component_indices,
                )
            f_x = beta_x - beta_family_x
            f_y = beta_y - beta_family_y
            if _step == 0 and initial_jacobian_entries is not None:
                jac_a00, jac_a01, jac_a10, jac_a11 = initial_jacobian_entries
            else:
                if component_indices is None:
                    jac_a00, jac_a01, jac_a10, jac_a11 = self._lensing_jacobian_for_components(
                        z_source,
                        current_x,
                        current_y,
                        packed_state,
                    )
                else:
                    jac_a00, jac_a01, jac_a10, jac_a11 = self._lensing_jacobian_for_components(
                        z_source,
                        current_x,
                        current_y,
                        packed_state,
                        component_indices,
                    )
            delta_x, delta_y, step_finite = _linearized_image_plane_residual_from_jacobian(
                f_x,
                f_y,
                jac_a00,
                jac_a01,
                jac_a10,
                jac_a11,
                max_gain=self.likelihood_stabilizer_max_gain,
                max_residual_arcsec=self.likelihood_stabilizer_max_residual_arcsec,
            )
            current_x = current_x + delta_x
            current_y = current_y + delta_y
            total_dx = total_dx + delta_x
            total_dy = total_dy + delta_y
            finite = finite & step_finite
        return total_dx, total_dy, finite

    def _anchored_solved_image_plane_residuals_for_components(
        self,
        z_source: float,
        x_obs: jnp.ndarray,
        y_obs: jnp.ndarray,
        beta_family_x: jnp.ndarray,
        beta_family_y: jnp.ndarray,
        packed_state: dict[str, Any],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        current_x = x_obs
        current_y = y_obs
        finite = jnp.ones_like(x_obs, dtype=bool)
        for _step in range(int(self.anchored_image_plane_solve_steps)):
            beta_x, beta_y = self._ray_shooting_for_components(
                z_source,
                current_x,
                current_y,
                packed_state,
            )
            jac_a00, jac_a01, jac_a10, jac_a11 = self._lensing_jacobian_for_components(
                z_source,
                current_x,
                current_y,
                packed_state,
            )
            delta_x, delta_y, step_finite = _anchored_solved_image_plane_step_from_jacobian(
                beta_x - beta_family_x,
                beta_y - beta_family_y,
                jac_a00,
                jac_a01,
                jac_a10,
                jac_a11,
                trust_radius_arcsec=self.anchored_image_plane_trust_radius_arcsec,
                lm_damping_relative=self.anchored_image_plane_lm_damping_relative,
                lm_damping_absolute=self.anchored_image_plane_lm_damping_absolute,
            )
            current_x = current_x + delta_x
            current_y = current_y + delta_y
            finite = (
                finite
                & step_finite
                & jnp.isfinite(current_x)
                & jnp.isfinite(current_y)
            )
        return current_x - x_obs, current_y - y_obs, finite

    def _inactive_fd_step(self, spec: ParameterSpec) -> float:
        if spec.prior_kind in {"normal", "truncated_normal"} and spec.std is not None:
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
        for bin_data in self.traced_bin_data:
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
                    self.sample_likelihood_mode
                    in {
                        SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
                        SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
                        SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
                        SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
                        SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
                    }
                ),
            )
            if inactive_surrogate is None:
                self.surrogate_cache_by_z = {}
                self.surrogate_reference_params = None
                self.surrogate_reference_param_values = np.zeros(len(self.surrogate_param_indices), dtype=float)
                return
            reference_jacobian_entries = None
            if self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE:
                reference_jacobian_entries = self._lensing_jacobian_for_components(
                    bin_data.effective_z_source,
                    x_obs,
                    y_obs,
                    packed_state,
                )
            fold_regularized_kappa_eff = None
            fold_regularized_near_indices = None
            fold_regularized_far_indices = None
            if (
                self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE
                and reference_jacobian_entries is not None
            ):
                fold_regularized_near_indices, fold_regularized_far_indices = self._fold_regularized_mask_indices_from_jacobian_entries(
                    *reference_jacobian_entries
                )
                fold_regularized_kappa_eff = np.full(
                    int(np.asarray(x_obs).size),
                    float(self.fold_curvature_arcsec_inv),
                    dtype=float,
                )
                constrained_indices = np.asarray(bin_data.constrained_image_indices, dtype=np.int32)
                near_indices = np.asarray(fold_regularized_near_indices, dtype=np.int32)
                curvature_indices = np.intersect1d(near_indices, constrained_indices, assume_unique=True).astype(np.int32)
                if curvature_indices.size > 0:
                    fold_frame = _fold_regularized_singular_frame_from_jacobian(*reference_jacobian_entries)
                    fold_kappa_eff = self._fold_signed_curvature_from_observed_jacobian(
                        bin_data.effective_z_source,
                        x_obs,
                        y_obs,
                        packed_state,
                        reference_jacobian_entries,
                        component_indices=None,
                        fold_frame=fold_frame,
                        image_indices=jnp.asarray(curvature_indices, dtype=jnp.int32),
                        fill_value=jnp.asarray(self.fold_curvature_arcsec_inv, dtype=jnp.float64),
                    )
                    fold_kappa_eff_np = np.asarray(fold_kappa_eff, dtype=float)
                    fold_regularized_kappa_eff = np.where(
                        np.isfinite(fold_kappa_eff_np),
                        fold_kappa_eff_np,
                        float(self.fold_curvature_arcsec_inv),
                    )
            self.surrogate_cache_by_z[bin_data.effective_z_source] = SurrogateBinCache(
                effective_z_source=float(bin_data.effective_z_source),
                fold_regularized_kappa_eff=fold_regularized_kappa_eff,
                fold_regularized_near_indices=fold_regularized_near_indices,
                fold_regularized_far_indices=fold_regularized_far_indices,
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

    def _singular_near_far_indices_from_jacobian_entries(
        self,
        jac_a00: Any,
        jac_a01: Any,
        jac_a10: Any,
        jac_a11: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        a00 = np.asarray(jac_a00, dtype=float).reshape(-1)
        a01 = np.asarray(jac_a01, dtype=float).reshape(-1)
        a10 = np.asarray(jac_a10, dtype=float).reshape(-1)
        a11 = np.asarray(jac_a11, dtype=float).reshape(-1)
        n_items = int(a00.size)
        if not (a01.size == n_items and a10.size == n_items and a11.size == n_items):
            indices = np.arange(n_items, dtype=np.int32)
            return indices, np.zeros(0, dtype=np.int32)
        normal00 = np.square(a00) + np.square(a10)
        normal01 = a00 * a01 + a10 * a11
        normal11 = np.square(a01) + np.square(a11)
        trace = np.maximum(normal00 + normal11, 0.0)
        diff = normal00 - normal11
        gap = np.sqrt(np.maximum(np.square(diff) + 4.0 * np.square(normal01), 0.0))
        metric = np.sqrt(np.maximum(0.5 * (trace - gap), 0.0))
        cutoff = float(self.critical_arc_singular_threshold) + 3.0 * float(self.critical_arc_singular_softness)
        finite = (
            np.isfinite(a00)
            & np.isfinite(a01)
            & np.isfinite(a10)
            & np.isfinite(a11)
            & np.isfinite(metric)
        )
        near_mask = (~finite) | (metric <= cutoff)
        indices = np.arange(n_items, dtype=np.int32)
        return indices[near_mask], indices[~near_mask]

    def _fold_regularized_mask_indices_from_jacobian_entries(
        self,
        jac_a00: Any,
        jac_a01: Any,
        jac_a10: Any,
        jac_a11: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._singular_near_far_indices_from_jacobian_entries(
            jac_a00,
            jac_a01,
            jac_a10,
            jac_a11,
        )

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
            max_gain=self.likelihood_stabilizer_max_gain,
            max_residual_arcsec=self.likelihood_stabilizer_max_residual_arcsec,
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
        fit_component_indices = self._fit_component_indices() if hasattr(self, "_fit_component_indices") else None
        if self.sample_likelihood_mode in {
            SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
            SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        }:
            fit_component_indices = None
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
                if fit_component_indices is None:
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
                else:
                    beta_x, beta_y = jax.lax.cond(
                        invalid,
                        lambda _: (x_obs, y_obs),
                        lambda current_state: self._ray_shooting_for_components(
                            bin_data.effective_z_source,
                            x_obs,
                            y_obs,
                            current_state,
                            fit_component_indices,
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

            if self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE:
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
                    if fit_component_indices is None:
                        observed_jacobian_entries = self._lensing_jacobian_for_components(
                            bin_data.effective_z_source,
                            x_obs,
                            y_obs,
                            packed_state,
                        )
                    else:
                        observed_jacobian_entries = self._lensing_jacobian_for_components(
                            bin_data.effective_z_source,
                            x_obs,
                            y_obs,
                            packed_state,
                            fit_component_indices,
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
                        component_indices=fit_component_indices,
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
                    residual_loss=self.likelihood_stabilizer_residual_loss,
                    student_t_nu=self.likelihood_stabilizer_student_t_nu,
                )
                bin_loglike = bin_loglike + source_transport_correction
            elif self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE:
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
                if self.anchored_image_plane_solve_steps == 0:
                    residual_x, residual_y, residual_finite = _anchored_solved_image_plane_step_from_jacobian(
                        observed_beta_x - beta_family_x,
                        observed_beta_y - beta_family_y,
                        *observed_jacobian_entries,
                        trust_radius_arcsec=self.anchored_image_plane_trust_radius_arcsec,
                        lm_damping_relative=self.anchored_image_plane_lm_damping_relative,
                        lm_damping_absolute=self.anchored_image_plane_lm_damping_absolute,
                    )
                else:
                    residual_x, residual_y, residual_finite = self._anchored_solved_image_plane_residuals_for_components(
                        bin_data.effective_z_source,
                        x_obs,
                        y_obs,
                        beta_family_x,
                        beta_family_y,
                        packed_state,
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
                    residual_loss=self.likelihood_stabilizer_residual_loss,
                    student_t_nu=self.likelihood_stabilizer_student_t_nu,
                )
                bin_loglike = bin_loglike + source_transport_correction
            elif self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE:
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
                (
                    residual_x,
                    residual_y,
                    singular_min,
                    singular_max,
                    critical_p00,
                    critical_p01,
                    critical_p11,
                    residual_finite,
                ) = _critical_arc_lm_geometry_from_jacobian(
                    observed_beta_x - beta_family_x,
                    observed_beta_y - beta_family_y,
                    *observed_jacobian_entries,
                    trust_radius_arcsec=self.critical_arc_lm_trust_radius_arcsec,
                    lm_damping_relative=self.critical_arc_lm_damping_relative,
                    lm_damping_absolute=self.critical_arc_lm_damping_absolute,
                )
                invalid = invalid | (~has_source_positions) | (~jnp.all(residual_finite))
                bin_loglike = _critical_arc_mixture_image_plane_bin_loglike(
                    residual_x=residual_x,
                    residual_y=residual_y,
                    jac_a00=observed_jacobian_entries[0],
                    jac_a01=observed_jacobian_entries[1],
                    jac_a10=observed_jacobian_entries[2],
                    jac_a11=observed_jacobian_entries[3],
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
                    residual_loss=self.likelihood_stabilizer_residual_loss,
                    student_t_nu=self.likelihood_stabilizer_student_t_nu,
                    critical_direction_sigma_arcsec=self.critical_arc_critical_direction_sigma_arcsec,
                    base_prob=self.critical_arc_base_prob,
                    max_prob=self.critical_arc_max_prob,
                    singular_threshold=self.critical_arc_singular_threshold,
                    singular_softness=self.critical_arc_singular_softness,
                    singular_min_precomputed=singular_min,
                    singular_max_precomputed=singular_max,
                    critical_direction_projector_entries=(critical_p00, critical_p01, critical_p11),
                )
                bin_loglike = bin_loglike + source_transport_correction
            elif self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE:
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
                    if fit_component_indices is None:
                        observed_jacobian_entries = self._lensing_jacobian_for_components(
                            bin_data.effective_z_source,
                            x_obs,
                            y_obs,
                            packed_state,
                        )
                    else:
                        observed_jacobian_entries = self._lensing_jacobian_for_components(
                            bin_data.effective_z_source,
                            x_obs,
                            y_obs,
                            packed_state,
                            fit_component_indices,
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
                invalid = invalid | (~has_source_positions)
                residual_beta_x = observed_beta_x - beta_family_x
                residual_beta_y = observed_beta_y - beta_family_y
                fold_cache = None
                use_cached_fold_regularization = False
                if (
                    self.surrogate_enabled
                    and self.surrogate_cache_by_z
                    and float(self.image_presence_penalty_weight) <= 0.0
                ):
                    fold_cache = self.surrogate_cache_by_z.get(bin_data.effective_z_source)
                    use_cached_fold_regularization = (
                        fold_cache is not None
                        and fold_cache.fold_regularized_kappa_eff is not None
                        and fold_cache.fold_regularized_near_indices is not None
                        and fold_cache.fold_regularized_far_indices is not None
                    )
                if use_cached_fold_regularization and fold_cache is not None:
                    def take_rows(value: jnp.ndarray, indices: np.ndarray) -> jnp.ndarray:
                        return jnp.take(value, jnp.asarray(indices, dtype=jnp.int32), axis=0)

                    fold_near_indices = np.asarray(fold_cache.fold_regularized_near_indices, dtype=np.int32)
                    fold_far_indices = np.asarray(fold_cache.fold_regularized_far_indices, dtype=np.int32)
                    fold_kappa_eff = jnp.asarray(fold_cache.fold_regularized_kappa_eff, dtype=jnp.float64)
                    bin_loglike = jnp.asarray(0.0, dtype=jnp.float64)
                    if int(fold_near_indices.size) > 0:
                        bin_loglike = bin_loglike + _fold_regularized_image_plane_bin_loglike(
                            residual_beta_x=take_rows(residual_beta_x, fold_near_indices),
                            residual_beta_y=take_rows(residual_beta_y, fold_near_indices),
                            jac_a00=take_rows(observed_jacobian_entries[0], fold_near_indices),
                            jac_a01=take_rows(observed_jacobian_entries[1], fold_near_indices),
                            jac_a10=take_rows(observed_jacobian_entries[2], fold_near_indices),
                            jac_a11=take_rows(observed_jacobian_entries[3], fold_near_indices),
                            family_idx=None,
                            n_families=None,
                            sigma_per_image=take_rows(sigma_base, fold_near_indices),
                            reliability_per_image=take_rows(reliability, fold_near_indices),
                            image_has_constraint=take_rows(image_has_constraint, fold_near_indices),
                            image_sigma_int=image_sigma_int,
                            scatter_var_x=take_rows(scatter_var_x, fold_near_indices),
                            scatter_var_y=take_rows(scatter_var_y, fold_near_indices),
                            covariance_floor=self.source_plane_covariance_floor,
                            outlier_sigma_arcsec=self.source_plane_outlier_sigma_arcsec,
                            fold_curvature_arcsec_inv=self.fold_curvature_arcsec_inv,
                            fold_kappa_eff=take_rows(fold_kappa_eff, fold_near_indices),
                            max_gain=self.likelihood_stabilizer_max_gain,
                            max_residual_arcsec=self.likelihood_stabilizer_max_residual_arcsec,
                            residual_loss=self.likelihood_stabilizer_residual_loss,
                            student_t_nu=self.likelihood_stabilizer_student_t_nu,
                            image_presence_penalty_weight=0.0,
                        )
                    if int(fold_far_indices.size) > 0:
                        bin_loglike = bin_loglike + _forward_metric_image_plane_bin_loglike(
                            residual_beta_x=take_rows(residual_beta_x, fold_far_indices),
                            residual_beta_y=take_rows(residual_beta_y, fold_far_indices),
                            jac_a00=take_rows(observed_jacobian_entries[0], fold_far_indices),
                            jac_a01=take_rows(observed_jacobian_entries[1], fold_far_indices),
                            jac_a10=take_rows(observed_jacobian_entries[2], fold_far_indices),
                            jac_a11=take_rows(observed_jacobian_entries[3], fold_far_indices),
                            family_idx=None,
                            n_families=None,
                            sigma_per_image=take_rows(sigma_base, fold_far_indices),
                            reliability_per_image=take_rows(reliability, fold_far_indices),
                            image_has_constraint=take_rows(image_has_constraint, fold_far_indices),
                            image_sigma_int=image_sigma_int,
                            scatter_var_x=take_rows(scatter_var_x, fold_far_indices),
                            scatter_var_y=take_rows(scatter_var_y, fold_far_indices),
                            covariance_floor=self.source_plane_covariance_floor,
                            outlier_sigma_arcsec=self.source_plane_outlier_sigma_arcsec,
                            max_gain=self.likelihood_stabilizer_max_gain,
                            max_residual_arcsec=self.likelihood_stabilizer_max_residual_arcsec,
                            residual_loss=self.likelihood_stabilizer_residual_loss,
                            student_t_nu=self.likelihood_stabilizer_student_t_nu,
                            image_presence_penalty_weight=0.0,
                        )
                else:
                    curvature_component_indices = fit_component_indices if fit_component_indices is not None else None
                    fold_frame = _fold_regularized_singular_frame_from_jacobian(*observed_jacobian_entries)
                    fold_curvature_fill = jnp.asarray(self.fold_curvature_arcsec_inv, dtype=jnp.float64)
                    curvature_image_indices = getattr(bin_data, "constrained_image_indices", None)
                    fold_kappa_eff = jax.lax.cond(
                        invalid,
                        lambda _: jnp.full_like(x_obs, fold_curvature_fill),
                        lambda _: self._fold_signed_curvature_from_observed_jacobian(
                            bin_data.effective_z_source,
                            x_obs,
                            y_obs,
                            packed_state,
                            observed_jacobian_entries,
                            curvature_component_indices,
                            fold_frame=fold_frame,
                            image_indices=curvature_image_indices,
                            fill_value=fold_curvature_fill,
                        ),
                        operand=None,
                    )
                    bin_loglike = _fold_regularized_image_plane_bin_loglike(
                        residual_beta_x=residual_beta_x,
                        residual_beta_y=residual_beta_y,
                        jac_a00=observed_jacobian_entries[0],
                        jac_a01=observed_jacobian_entries[1],
                        jac_a10=observed_jacobian_entries[2],
                        jac_a11=observed_jacobian_entries[3],
                        family_idx=family_idx,
                        n_families=n_families,
                        sigma_per_image=sigma_base,
                        reliability_per_image=reliability,
                        image_has_constraint=image_has_constraint,
                        image_sigma_int=image_sigma_int,
                        scatter_var_x=scatter_var_x,
                        scatter_var_y=scatter_var_y,
                        covariance_floor=self.source_plane_covariance_floor,
                        outlier_sigma_arcsec=self.source_plane_outlier_sigma_arcsec,
                        fold_curvature_arcsec_inv=self.fold_curvature_arcsec_inv,
                        fold_kappa_eff=fold_kappa_eff,
                        fold_frame=fold_frame,
                        max_gain=self.likelihood_stabilizer_max_gain,
                        max_residual_arcsec=self.likelihood_stabilizer_max_residual_arcsec,
                        residual_loss=self.likelihood_stabilizer_residual_loss,
                        student_t_nu=self.likelihood_stabilizer_student_t_nu,
                        image_presence_penalty_weight=self.image_presence_penalty_weight,
                        image_presence_match_radius_arcsec=self.image_presence_match_radius_arcsec,
                        image_presence_temperature_arcsec=self.image_presence_temperature_arcsec,
                        image_presence_count_softness=self.image_presence_count_softness,
                        image_presence_count_margin=self.image_presence_count_margin,
                    )
                bin_loglike = bin_loglike + source_transport_correction
            elif self.sample_likelihood_mode == SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE:
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
                    if fit_component_indices is None:
                        observed_jacobian_entries = self._lensing_jacobian_for_components(
                            bin_data.effective_z_source,
                            x_obs,
                            y_obs,
                            packed_state,
                        )
                    else:
                        observed_jacobian_entries = self._lensing_jacobian_for_components(
                            bin_data.effective_z_source,
                            x_obs,
                            y_obs,
                            packed_state,
                            fit_component_indices,
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
                invalid = invalid | (~has_source_positions)
                bin_loglike = _forward_metric_image_plane_bin_loglike(
                    residual_beta_x=observed_beta_x - beta_family_x,
                    residual_beta_y=observed_beta_y - beta_family_y,
                    jac_a00=observed_jacobian_entries[0],
                    jac_a01=observed_jacobian_entries[1],
                    jac_a10=observed_jacobian_entries[2],
                    jac_a11=observed_jacobian_entries[3],
                    family_idx=family_idx,
                    n_families=n_families,
                    sigma_per_image=sigma_base,
                    reliability_per_image=reliability,
                    image_has_constraint=image_has_constraint,
                    image_sigma_int=image_sigma_int,
                    scatter_var_x=scatter_var_x,
                    scatter_var_y=scatter_var_y,
                    covariance_floor=self.source_plane_covariance_floor,
                    outlier_sigma_arcsec=self.source_plane_outlier_sigma_arcsec,
                    max_gain=self.likelihood_stabilizer_max_gain,
                    max_residual_arcsec=self.likelihood_stabilizer_max_residual_arcsec,
                    residual_loss=self.likelihood_stabilizer_residual_loss,
                    student_t_nu=self.likelihood_stabilizer_student_t_nu,
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
                    max_gain=self.likelihood_stabilizer_max_gain,
                    max_residual_arcsec=self.likelihood_stabilizer_max_residual_arcsec,
                    residual_loss=self.likelihood_stabilizer_residual_loss,
                    student_t_nu=self.likelihood_stabilizer_student_t_nu,
                )
            else:
                bin_loglike = _source_plane_bin_loglike(
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
                    inv_abs_mu=self._magnification_inv_abs_mu(bin_data),
                    covariance_floor=self.source_plane_covariance_floor,
                    outlier_sigma_arcsec=self.source_plane_outlier_sigma_arcsec,
                    max_gain=self.likelihood_stabilizer_max_gain,
                    max_residual_arcsec=self.likelihood_stabilizer_max_residual_arcsec,
                    residual_loss=self.likelihood_stabilizer_residual_loss,
                    student_t_nu=self.likelihood_stabilizer_student_t_nu,
                )
            total_loglike = jnp.where(invalid, total_loglike, total_loglike + bin_loglike)
            invalid_seen = jnp.logical_or(invalid_seen, invalid)
        arc_data = getattr(self, "traced_arc_data", None)
        cab_likelihood_weight = float(getattr(self, "cab_likelihood_weight", 0.0))
        if arc_data is not None and int(getattr(arc_data, "n_arcs", 0)) > 0 and cab_likelihood_weight > 0.0:
            cab_packed_state, cab_validity = self._build_cab_packed_lens_state_with_validity_from_physical(
                physical_params,
                stop_gradient=True,
            )
            self._maybe_record_invalid_state(cab_validity)
            cab_invalid = ~cab_validity["is_valid"]
            cab_loglike = self._cab_morphology_loglike_for_arcs(
                arc_data,
                cab_packed_state,
                fit_component_indices,
            )
            total_loglike = jnp.where(cab_invalid, total_loglike, total_loglike + cab_loglike)
            invalid_seen = jnp.logical_or(invalid_seen, cab_invalid)
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
        fit_component_indices = self._fit_component_indices()

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
                beta_x, beta_y, invalid, packed_state = self._surrogate_beta(params_jax, physical_params, traced_bin_data)
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
                if fit_component_indices is None:
                    beta_x, beta_y = self._ray_shooting_for_components(
                        bin_data.effective_z_source,
                        jnp.asarray(bin_data.x_obs, dtype=jnp.float64),
                        jnp.asarray(bin_data.y_obs, dtype=jnp.float64),
                        packed_state,
                    )
                else:
                    beta_x, beta_y = self._ray_shooting_for_components(
                        bin_data.effective_z_source,
                        jnp.asarray(bin_data.x_obs, dtype=jnp.float64),
                        jnp.asarray(bin_data.y_obs, dtype=jnp.float64),
                        packed_state,
                        fit_component_indices,
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

    def _exact_source_ray_shooting(
        self,
        family: FamilyData,
        packed_state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        model, _solver = self._get_exact_model_solver(family.z_source)
        kwargs_lens = self._packed_to_kwargs_lens(packed_state)
        beta_x, beta_y = model.ray_shooting(
            jnp.asarray(family.x_obs, dtype=jnp.float64),
            jnp.asarray(family.y_obs, dtype=jnp.float64),
            kwargs_lens,
        )
        return np.asarray(beta_x, dtype=float), np.asarray(beta_y, dtype=float)

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
                min_distance=0.2,
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
        recovered_mask = np.zeros(n_observed, dtype=bool)
        diagnostics: dict[str, Any] = {
            "produced_image_count": n_produced,
            "recovered_image_count": 0,
            "missing_image_count": n_observed,
            "extra_image_count": n_produced,
            "multiplicity_failed": True,
            "multiplicity_failure_reason": "no_model_images" if n_produced == 0 else "no_matches",
            "matched_model_x_arcsec": matched_x,
            "matched_model_y_arcsec": matched_y,
            "recovered_image_mask": recovered_mask,
            "extra_model_x_arcsec": np.asarray([], dtype=float),
            "extra_model_y_arcsec": np.asarray([], dtype=float),
            "_matched_x": matched_x,
            "_matched_y": matched_y,
        }
        if x_array.size != y_array.size:
            n_extra = int(max(x_array.size, y_array.size))
            extra_x = np.full(n_extra, np.nan, dtype=float)
            extra_y = np.full(n_extra, np.nan, dtype=float)
            extra_x[: x_array.size] = x_array
            extra_y[: y_array.size] = y_array
            diagnostics.update(
                {
                    "produced_image_count": n_extra,
                    "extra_image_count": n_extra,
                    "multiplicity_failure_reason": "prediction_shape_mismatch",
                    "extra_model_x_arcsec": extra_x,
                    "extra_model_y_arcsec": extra_y,
                }
            )
            return diagnostics
        if n_observed == 0:
            diagnostics.update(
                {
                    "recovered_image_count": 0,
                    "missing_image_count": 0,
                    "extra_image_count": n_produced,
                    "multiplicity_failed": bool(n_produced != 0),
                    "multiplicity_failure_reason": "extra_model_images" if n_produced else "",
                    "extra_model_x_arcsec": x_array.copy(),
                    "extra_model_y_arcsec": y_array.copy(),
                }
            )
            return diagnostics
        if n_produced == 0:
            return diagnostics
        pred = np.column_stack([x_array, y_array])
        obs = np.column_stack([np.asarray(family.x_obs, dtype=float), np.asarray(family.y_obs, dtype=float)])
        if pred.shape[0] != n_produced or obs.shape[0] != n_observed:
            diagnostics.update(
                {
                    "multiplicity_failure_reason": "prediction_shape_mismatch",
                    "extra_model_x_arcsec": x_array.copy(),
                    "extra_model_y_arcsec": y_array.copy(),
                }
            )
            return diagnostics
        finite_pred = np.isfinite(pred).all(axis=1)
        finite_obs = np.isfinite(obs).all(axis=1)
        cost = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=2)
        cost = np.where(finite_pred[:, None] & finite_obs[None, :], cost, np.inf)
        finite_cost = np.isfinite(cost)
        if not np.any(finite_cost):
            diagnostics.update(
                {
                    "multiplicity_failure_reason": "nonfinite_prediction",
                    "extra_model_x_arcsec": x_array.copy(),
                    "extra_model_y_arcsec": y_array.copy(),
                }
            )
            return diagnostics
        finite_max = float(np.max(cost[finite_cost]))
        match_tolerance = float(getattr(self, "match_tolerance_arcsec", DEFAULT_MATCH_TOLERANCE))
        assignment_cost = np.where(finite_cost, cost, finite_max + match_tolerance + 1.0)
        row_ind, col_ind = linear_sum_assignment(assignment_cost)
        accepted = np.isfinite(cost[row_ind, col_ind]) & (cost[row_ind, col_ind] <= match_tolerance)
        recovered_count = int(np.sum(accepted))
        accepted_pred_indices = np.asarray(row_ind[accepted], dtype=int)
        for pred_idx, obs_idx in zip(row_ind[accepted], col_ind[accepted]):
            matched_x[int(obs_idx)] = x_array[int(pred_idx)]
            matched_y[int(obs_idx)] = y_array[int(pred_idx)]
            recovered_mask[int(obs_idx)] = True
        extra_mask = np.ones(n_produced, dtype=bool)
        extra_mask[accepted_pred_indices] = False
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
                "matched_model_x_arcsec": matched_x,
                "matched_model_y_arcsec": matched_y,
                "recovered_image_mask": recovered_mask,
                "extra_model_x_arcsec": x_array[extra_mask],
                "extra_model_y_arcsec": y_array[extra_mask],
                "_matched_x": matched_x,
                "_matched_y": matched_y,
            }
        )
        return diagnostics

    def _record_exact_prediction_details(self, family_id: str, details: dict[str, Any]) -> None:
        if not hasattr(self, "_last_exact_prediction_details"):
            self._last_exact_prediction_details = {}
        self._last_exact_prediction_details[str(family_id)] = details

    def _empty_arc_aware_image_support_details(
        self,
        family: FamilyData,
        match_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        n_images = int(getattr(family, "n_images", 0))
        point_recovered = np.zeros(n_images, dtype=bool)
        point_residual = np.full(n_images, np.nan, dtype=float)
        if isinstance(match_details, dict):
            recovered = np.asarray(match_details.get("recovered_image_mask", point_recovered), dtype=bool).reshape(-1)
            matched_x = np.asarray(match_details.get("matched_model_x_arcsec", point_residual), dtype=float).reshape(-1)
            matched_y = np.asarray(match_details.get("matched_model_y_arcsec", point_residual), dtype=float).reshape(-1)
            if recovered.shape == (n_images,) and matched_x.shape == (n_images,) and matched_y.shape == (n_images,):
                point_recovered = recovered
                point_residual = np.sqrt(
                    np.square(matched_x - np.asarray(family.x_obs, dtype=float))
                    + np.square(matched_y - np.asarray(family.y_obs, dtype=float))
                )
                point_residual = np.where(point_recovered, point_residual, np.nan)
        point_valid = point_recovered & np.isfinite(point_residual)
        point_residual = np.where(point_valid, point_residual, np.nan)
        supported_or_recovered = np.isfinite(point_residual)
        arc_aware_rms = (
            float(np.sqrt(np.mean(np.square(point_residual[supported_or_recovered]))))
            if np.any(supported_or_recovered)
            else np.nan
        )
        status = np.where(point_valid, "point_recovered", "not_recovered").astype(object)
        return {
            "point_image_residual_arcsec": point_residual,
            "arc_candidate_supported": np.zeros(n_images, dtype=bool),
            "arc_candidate_image_residual_arcsec": np.full(n_images, np.nan, dtype=float),
            "preferred_recovery_status": status,
            "preferred_image_residual_arcsec": point_residual,
            "arc_recovery_status": status,
            "arc_aware_image_residual_arcsec": point_residual,
            "arc_noncritical_direction_residual_arcsec": np.full(n_images, np.nan, dtype=float),
            "arc_critical_direction_residual_arcsec": np.full(n_images, np.nan, dtype=float),
            "arc_critical_direction_x": np.full(n_images, np.nan, dtype=float),
            "arc_critical_direction_y": np.full(n_images, np.nan, dtype=float),
            "arc_noncritical_direction_x": np.full(n_images, np.nan, dtype=float),
            "arc_noncritical_direction_y": np.full(n_images, np.nan, dtype=float),
            "arc_s_min": np.full(n_images, np.nan, dtype=float),
            "arc_s_max": np.full(n_images, np.nan, dtype=float),
            "arc_detA": np.full(n_images, np.nan, dtype=float),
            "arc_prior_probability": np.full(n_images, np.nan, dtype=float),
            "arc_curve_distance_arcsec": np.full(n_images, np.nan, dtype=float),
            "arc_curve_arclength_arcsec": np.full(n_images, np.nan, dtype=float),
            "arc_curve_finite": np.zeros(n_images, dtype=bool),
            "arc_support_anchor_x_arcsec": np.full(n_images, np.nan, dtype=float),
            "arc_support_anchor_y_arcsec": np.full(n_images, np.nan, dtype=float),
            "arc_support_curve_x_arcsec": np.asarray([_json_arcsec_array([]) for _ in range(n_images)], dtype=object),
            "arc_support_curve_y_arcsec": np.asarray([_json_arcsec_array([]) for _ in range(n_images)], dtype=object),
            "arc_supported_mask": np.zeros(n_images, dtype=bool),
            "arc_supported": np.zeros(n_images, dtype=bool),
            "arc_support_finite_mask": np.zeros(n_images, dtype=bool),
            "arc_aware_image_rms_arcsec": arc_aware_rms,
            "arc_aware_recovered_image_count": int(np.sum(supported_or_recovered)),
            "arc_aware_missing_image_count": int(max(0, n_images - int(np.sum(supported_or_recovered)))),
            "arc_supported_image_count": 0,
            "arc_candidate_supported_image_count": 0,
        }

    def _cab_morphology_details_for_arcs(self, params: np.ndarray) -> pd.DataFrame:
        arc_data = getattr(self.state, "arc_data", None)
        if arc_data is None or int(getattr(arc_data, "n_arcs", 0)) <= 0:
            return pd.DataFrame(
                columns=[
                    "arc_id",
                    "z_arc",
                    "cab_anchor_x_arcsec",
                    "cab_anchor_y_arcsec",
                    "cab_tangent_angle_obs_rad",
                    "cab_tangent_angle_model_rad",
                    "cab_tangent_residual_rad",
                    "cab_curvature_obs_arcsec_inv",
                    "cab_curvature_model_arcsec_inv",
                    "cab_curvature_residual_arcsec_inv",
                    "cab_loglike",
                    "cab_finite",
                ]
            )
        n_arcs = int(arc_data.n_arcs)
        base = pd.DataFrame(
            {
                "arc_id": [str(value) for value in arc_data.arc_ids],
                "z_arc": np.asarray(arc_data.z_arc, dtype=float),
                "cab_anchor_x_arcsec": np.asarray(arc_data.anchor_x, dtype=float),
                "cab_anchor_y_arcsec": np.asarray(arc_data.anchor_y, dtype=float),
                "cab_tangent_angle_obs_rad": np.asarray(arc_data.tangent_angle_rad, dtype=float),
                "cab_tangent_angle_model_rad": np.full(n_arcs, np.nan, dtype=float),
                "cab_tangent_residual_rad": np.full(n_arcs, np.nan, dtype=float),
                "cab_curvature_obs_arcsec_inv": np.asarray(arc_data.curvature_arcsec_inv, dtype=float),
                "cab_curvature_model_arcsec_inv": np.full(n_arcs, np.nan, dtype=float),
                "cab_curvature_residual_arcsec_inv": np.full(n_arcs, np.nan, dtype=float),
                "cab_loglike": np.zeros(n_arcs, dtype=float),
                "cab_finite": np.zeros(n_arcs, dtype=bool),
            }
        )
        if float(self.cab_likelihood_weight) <= 0.0:
            return base
        try:
            params_jax = jnp.asarray(params, dtype=jnp.float64)
            physical_params = self._physical_parameter_vector(params_jax)
            packed_state, validity = self._build_cab_packed_lens_state_with_validity_from_physical(
                physical_params,
                stop_gradient=False,
            )
            if not bool(np.asarray(validity["is_valid"], dtype=bool)):
                return base
            traced = self._prepare_traced_arc_constraint_data(arc_data)
            prediction = self._cab_morphology_predictions_for_anchors(
                float(CAB_MORPHOLOGY_MODEL_KEY),
                traced.anchor_x,
                traced.anchor_y,
                packed_state,
                self._fit_component_indices(),
            )
            active = jnp.ones((n_arcs,), dtype=bool)
            terms = _cab_morphology_terms(
                predicted_tangent_angle_rad=prediction.tangent_angle_rad,
                predicted_curvature_arcsec_inv=prediction.curvature_arcsec_inv,
                prediction_finite=prediction.finite,
                observed_tangent_angle_rad=traced.tangent_angle_rad,
                observed_curvature_arcsec_inv=traced.curvature_arcsec_inv,
                sigma_tangent_angle_rad=traced.sigma_tangent_angle_rad,
                sigma_curvature_arcsec_inv=traced.sigma_curvature_arcsec_inv,
                reliability=traced.reliability,
                active_arcs=active,
                tangent_sigma_floor_rad=self.cab_tangent_sigma_floor_rad,
                curvature_sigma_floor_arcsec_inv=self.cab_curvature_sigma_floor_arcsec_inv,
                branch_weight=prediction.branch_weight,
                frame_weight=prediction.frame_weight,
            )
            angle_by_branch = np.asarray(prediction.tangent_angle_rad, dtype=float)
            curvature_by_branch = np.asarray(prediction.curvature_arcsec_inv, dtype=float)
            branch_score = np.asarray(terms.branch_weight * terms.frame_weight, dtype=float)
            dominant_branch = np.argmax(branch_score, axis=1)
            row_index = np.arange(n_arcs)
            finite = np.asarray(terms.finite_row, dtype=bool)
            base["cab_tangent_angle_model_rad"] = np.where(finite, angle_by_branch[row_index, dominant_branch], np.nan)
            base["cab_curvature_model_arcsec_inv"] = np.where(finite, curvature_by_branch[row_index, dominant_branch], np.nan)
            base["cab_tangent_residual_rad"] = np.where(finite, np.asarray(terms.tangent_residual, dtype=float), np.nan)
            base["cab_curvature_residual_arcsec_inv"] = np.where(finite, np.asarray(terms.curvature_residual, dtype=float), np.nan)
            base["cab_loglike"] = np.where(
                finite,
                float(self.cab_likelihood_weight) * np.asarray(terms.row_loglike, dtype=float),
                0.0,
            )
            base["cab_finite"] = finite
        except Exception:
            return base
        return base

    def _arc_aware_image_support_details(
        self,
        params: np.ndarray,
        family: FamilyData,
        source_x: float,
        source_y: float,
        match_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        n_images = int(getattr(family, "n_images", 0))
        if n_images <= 0:
            return self._empty_arc_aware_image_support_details(family)
        point_recovered = np.zeros(n_images, dtype=bool)
        point_residual = np.full(n_images, np.nan, dtype=float)
        if isinstance(match_details, dict):
            recovered = np.asarray(match_details.get("recovered_image_mask", point_recovered), dtype=bool).reshape(-1)
            matched_x = np.asarray(match_details.get("matched_model_x_arcsec", point_residual), dtype=float).reshape(-1)
            matched_y = np.asarray(match_details.get("matched_model_y_arcsec", point_residual), dtype=float).reshape(-1)
            if recovered.shape == (n_images,) and matched_x.shape == (n_images,) and matched_y.shape == (n_images,):
                point_recovered = recovered
                point_residual = np.sqrt(
                    np.square(matched_x - np.asarray(family.x_obs, dtype=float))
                    + np.square(matched_y - np.asarray(family.y_obs, dtype=float))
                )
                point_residual = np.where(point_recovered, point_residual, np.nan)
        effective_z = float(getattr(family, "effective_z_source", family.z_source))
        models_by_effective_z = getattr(self, "models_by_effective_z", None)
        if (
            isinstance(models_by_effective_z, dict)
            and effective_z not in models_by_effective_z
            and float(family.z_source) in models_by_effective_z
        ):
            effective_z = float(family.z_source)
        try:
            params_jax = jnp.asarray(params, dtype=jnp.float64)
            packed_state = self._build_packed_lens_state(params_jax, effective_z)
            x_obs = jnp.asarray(family.x_obs, dtype=jnp.float64)
            y_obs = jnp.asarray(family.y_obs, dtype=jnp.float64)
            beta_x, beta_y = self._ray_shooting_for_components(effective_z, x_obs, y_obs, packed_state)
            jac_a00, jac_a01, jac_a10, jac_a11 = self._lensing_jacobian_for_components(effective_z, x_obs, y_obs, packed_state)

            def jacobian_at(curve_x: np.ndarray, curve_y: np.ndarray) -> tuple[Any, Any, Any, Any]:
                return self._lensing_jacobian_for_components(
                    effective_z,
                    jnp.asarray(curve_x, dtype=jnp.float64),
                    jnp.asarray(curve_y, dtype=jnp.float64),
                    packed_state,
                )

            details = _arc_aware_image_support_from_local_linearization(
                beta_x - jnp.asarray(float(source_x), dtype=jnp.float64),
                beta_y - jnp.asarray(float(source_y), dtype=jnp.float64),
                jac_a00,
                jac_a01,
                jac_a10,
                jac_a11,
                theta_obs_x=x_obs,
                theta_obs_y=y_obs,
                jacobian_at=jacobian_at,
                point_recovered_mask=point_recovered,
                point_residual_arcsec=point_residual,
                noncritical_support_radius_arcsec=float(
                    getattr(
                        self,
                        "arc_aware_noncritical_support_radius_arcsec",
                        DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC,
                    )
                ),
                max_arclength_arcsec=float(
                    getattr(self, "arc_aware_max_arclength_arcsec", DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC)
                ),
                curve_step_arcsec=float(
                    getattr(self, "arc_aware_curve_step_arcsec", DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC)
                ),
                trust_radius_arcsec=self.critical_arc_lm_trust_radius_arcsec,
                lm_damping_relative=self.critical_arc_lm_damping_relative,
                lm_damping_absolute=self.critical_arc_lm_damping_absolute,
                base_prob=self.critical_arc_base_prob,
                max_prob=self.critical_arc_max_prob,
                singular_threshold=self.critical_arc_singular_threshold,
                singular_softness=self.critical_arc_singular_softness,
            )
        except Exception:
            details = self._empty_arc_aware_image_support_details(family)
        return details

    def _exact_family_prediction_details(self, params: np.ndarray, family: FamilyData) -> dict[str, Any]:
        cache = self.validation_cache[family.family_id]
        base_details: dict[str, Any] = {
            "produced_image_count": np.nan,
            "recovered_image_count": np.nan,
            "missing_image_count": np.nan,
            "extra_image_count": np.nan,
            "multiplicity_failed": True,
            "multiplicity_failure_reason": "",
            "matched_model_x_arcsec": np.full(int(getattr(family, "n_images", 0)), np.nan, dtype=float),
            "matched_model_y_arcsec": np.full(int(getattr(family, "n_images", 0)), np.nan, dtype=float),
            "recovered_image_mask": np.zeros(int(getattr(family, "n_images", 0)), dtype=bool),
            "extra_model_x_arcsec": np.asarray([], dtype=float),
            "extra_model_y_arcsec": np.asarray([], dtype=float),
            "failed": True,
        }
        base_details.update(self._empty_arc_aware_image_support_details(family))
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
        arc_details = self._arc_aware_image_support_details(params, family, source_x, source_y)
        x_pred: np.ndarray
        y_pred: np.ndarray
        try:
            x_pred, y_pred = self._solve_exact_images_lenstronomy(family, packed_state, source_x, source_y)
        except Exception:
            cache.multiplicity_mismatch_count += 1
            details = {
                **base_details,
                **arc_details,
                "multiplicity_failure_reason": "exact_image_prediction_failed",
            }
            self._record_exact_prediction_details(family.family_id, details)
            return details

        match_details = self._image_match_diagnostics(np.asarray(x_pred), np.asarray(y_pred), family)
        arc_details = self._arc_aware_image_support_details(params, family, source_x, source_y, match_details)
        matched = self._match_images(np.asarray(x_pred), np.asarray(y_pred), family)
        if matched is None:
            if len(x_pred) != family.n_images:
                cache.multiplicity_mismatch_count += 1
            else:
                cache.match_failure_count += 1
            details = {
                **base_details,
                **{key: value for key, value in match_details.items() if not str(key).startswith("_")},
                **arc_details,
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
            **arc_details,
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

    def evaluate(self, params: np.ndarray, validate_all_families: bool = False) -> EvaluationResult:
        del validate_all_families
        if self.surrogate_enabled and self._surrogate_needs_refresh(np.asarray(params, dtype=float)):
            self.refresh_surrogate(np.asarray(params, dtype=float), reason="validation_drift")
            self.refresh_scaling_scatter_cache(np.asarray(params, dtype=float), reason="validation_drift")
            self.refresh_source_metric_cache(np.asarray(params, dtype=float), reason="validation_drift")
        source_loglike = self.source_loglike(params)
        family_predictions = self._family_source_summary(params)
        _ensure_family_prediction_image_arrays(self.state, family_predictions)
        refresh_reason = "quick_diagnostics" if bool(getattr(self, "quick_diagnostics", False)) else "source_summary"
        for prediction in family_predictions.values():
            prediction["approx_image_rms_arcsec"] = prediction.get("source_plane_rms")
            prediction["used_exact_refresh"] = False
            prediction["refresh_reason"] = refresh_reason
        return EvaluationResult(
            loglike=float(source_loglike),
            family_predictions=family_predictions,
        )

    def release_runtime_caches(self) -> None:
        self.models_by_effective_z = {}
        self.exact_models_by_z = {}
        self.exact_solvers_by_z = {}
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
                "potfile_mass_size_reparam": bool(_potfile_mass_size_group_count(state.parameter_specs) > 0),
                "potfile_mass_size_reparam_group_count": int(_potfile_mass_size_group_count(state.parameter_specs)),
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
                "arc_data": (
                    {
                        "arc_ids": state.arc_data.arc_ids,
                        "z_arc": state.arc_data.z_arc,
                        "anchor_x": state.arc_data.anchor_x,
                        "anchor_y": state.arc_data.anchor_y,
                        "tangent_angle_rad": state.arc_data.tangent_angle_rad,
                        "curvature_arcsec_inv": state.arc_data.curvature_arcsec_inv,
                        "sigma_tangent_angle_rad": state.arc_data.sigma_tangent_angle_rad,
                        "sigma_curvature_arcsec_inv": state.arc_data.sigma_curvature_arcsec_inv,
                        "reliability": state.arc_data.reliability,
                    }
                    if state.arc_data is not None
                    else None
                ),
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
            arc_data=(
                ArcConstraintData(
                    arc_ids=[str(value) for value in meta["arc_data"]["arc_ids"]],
                    z_arc=np.asarray(meta["arc_data"]["z_arc"], dtype=float),
                    anchor_x=np.asarray(meta["arc_data"]["anchor_x"], dtype=float),
                    anchor_y=np.asarray(meta["arc_data"]["anchor_y"], dtype=float),
                    tangent_angle_rad=np.asarray(meta["arc_data"]["tangent_angle_rad"], dtype=float),
                    curvature_arcsec_inv=np.asarray(meta["arc_data"]["curvature_arcsec_inv"], dtype=float),
                    sigma_tangent_angle_rad=np.asarray(meta["arc_data"]["sigma_tangent_angle_rad"], dtype=float),
                    sigma_curvature_arcsec_inv=np.asarray(meta["arc_data"]["sigma_curvature_arcsec_inv"], dtype=float),
                    reliability=np.asarray(meta["arc_data"]["reliability"], dtype=float),
                )
                if isinstance(meta.get("arc_data"), dict)
                else None
            ),
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


def _load_artifacts(artifacts_dir: Path) -> tuple[BuildState, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    h5_path = artifacts_dir / "plot_bundle.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing current-format artifact bundle: {h5_path}")
    state, cli_args, arrays, init_diagnostics = _rebuild_state_from_h5(h5_path)
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


_LIKELIHOOD_STABILIZER_SAVED_ARG_DEFAULTS: dict[str, Any] = {
    "likelihood_stabilizer_max_gain": DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN,
    "likelihood_stabilizer_max_residual_arcsec": DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC,
    "likelihood_stabilizer_residual_loss": DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS,
    "likelihood_stabilizer_student_t_nu": DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU,
}


def _saved_likelihood_stabilizer_arg(saved_args: dict[str, Any], key: str, default: Any) -> Any:
    legacy_key = key.replace("likelihood_stabilizer", "linearized_image_plane")
    return saved_args.get(key, saved_args.get(legacy_key, default))


def _normalized_saved_likelihood_stabilizer_args(saved_args: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _saved_likelihood_stabilizer_arg(saved_args, key, default)
        for key, default in _LIKELIHOOD_STABILIZER_SAVED_ARG_DEFAULTS.items()
    }


def _drop_legacy_likelihood_stabilizer_args(args: dict[str, Any]) -> None:
    for key in _LIKELIHOOD_STABILIZER_SAVED_ARG_DEFAULTS:
        args.pop(key.replace("likelihood_stabilizer", "linearized_image_plane"), None)


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
    cab_likelihood_weight = _effective_cab_likelihood_weight(saved_args.get("cab_likelihood_weight"), state)
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=float(saved_args.get("match_tolerance_arcsec", DEFAULT_MATCH_TOLERANCE)),
        sampling_engine="full",
        active_scaling_galaxies=saved_args.get("active_scaling_galaxies"),
        active_scaling_selection=str(saved_args.get("active_scaling_selection", "adaptive")),
        active_scaling_cumulative_fraction=float(
            saved_args.get("active_scaling_cumulative_fraction", DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION)
        ),
        active_scaling_min=int(saved_args.get("active_scaling_min", DEFAULT_ACTIVE_SCALING_MIN)),
        refresh_every=int(saved_args.get("refresh_every", DEFAULT_REFRESH_EVERY)),
        refresh_param_drift_frac=float(saved_args.get("refresh_param_drift_frac", DEFAULT_REFRESH_PARAM_DRIFT_FRAC)),
        source_plane_covariance_floor=float(saved_args.get("source_plane_covariance_floor", 1.0e-6)),
        source_plane_outlier_sigma_arcsec=float(
            saved_args.get("source_plane_outlier_sigma_arcsec", DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC)
        ),
        sample_likelihood_mode=sample_likelihood_mode,
        image_plane_newton_steps=int(saved_args.get("image_plane_newton_steps", 0)),
        anchored_image_plane_solve_steps=int(
            saved_args.get("anchored_image_plane_solve_steps", DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS)
        ),
        anchored_image_plane_trust_radius_arcsec=float(
            saved_args.get(
                "anchored_image_plane_trust_radius_arcsec",
                DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
            )
        ),
        anchored_image_plane_lm_damping_relative=float(
            saved_args.get(
                "anchored_image_plane_lm_damping_relative",
                DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
            )
        ),
        anchored_image_plane_lm_damping_absolute=float(
            saved_args.get(
                "anchored_image_plane_lm_damping_absolute",
                DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
            )
        ),
        critical_arc_critical_direction_sigma_arcsec=float(
            saved_args.get("critical_arc_critical_direction_sigma_arcsec", DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC)
        ),
        critical_arc_base_prob=float(
            saved_args.get("critical_arc_base_prob", DEFAULT_CRITICAL_ARC_BASE_PROB)
        ),
        critical_arc_max_prob=float(
            saved_args.get("critical_arc_max_prob", DEFAULT_CRITICAL_ARC_MAX_PROB)
        ),
        critical_arc_singular_threshold=float(
            saved_args.get("critical_arc_singular_threshold", DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD)
        ),
        critical_arc_singular_softness=float(
            saved_args.get("critical_arc_singular_softness", DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS)
        ),
        critical_arc_lm_damping_relative=float(
            saved_args.get("critical_arc_lm_damping_relative", DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE)
        ),
        critical_arc_lm_damping_absolute=float(
            saved_args.get("critical_arc_lm_damping_absolute", DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE)
        ),
        critical_arc_lm_trust_radius_arcsec=float(
            saved_args.get("critical_arc_lm_trust_radius_arcsec", DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC)
        ),
        arc_aware_noncritical_support_radius_arcsec=float(
            saved_args.get(
                "arc_aware_noncritical_support_radius_arcsec",
                DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC,
            )
        ),
        arc_aware_max_arclength_arcsec=float(
            saved_args.get("arc_aware_max_arclength_arcsec", DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC)
        ),
        arc_aware_curve_step_arcsec=float(
            saved_args.get("arc_aware_curve_step_arcsec", DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC)
        ),
        fold_curvature_arcsec_inv=float(
            saved_args.get("fold_curvature_arcsec_inv", DEFAULT_FOLD_CURVATURE_ARCSEC_INV)
        ),
        cab_likelihood_weight=cab_likelihood_weight,
        cab_finite_difference_step_arcsec=float(
            saved_args.get("cab_finite_difference_step_arcsec", DEFAULT_CAB_FINITE_DIFFERENCE_STEP_ARCSEC)
        ),
        cab_tangent_sigma_floor_rad=float(
            saved_args.get("cab_tangent_sigma_floor_rad", DEFAULT_CAB_TANGENT_SIGMA_FLOOR_RAD)
        ),
        cab_curvature_sigma_floor_arcsec_inv=float(
            saved_args.get("cab_curvature_sigma_floor_arcsec_inv", DEFAULT_CAB_CURVATURE_SIGMA_FLOOR_ARCSEC_INV)
        ),
        image_plane_scatter_floor_arcsec=float(
            saved_args.get("image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)
        ),
        fixed_image_sigma_int_arcsec=saved_args.get("fix_image_sigma_int_arcsec"),
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
        **_normalized_saved_likelihood_stabilizer_args(saved_args),
        evidence_source_prior_sigma_arcsec=saved_args.get("evidence_source_prior_sigma_arcsec"),
        evidence_source_prior_mean_x_arcsec=float(saved_args.get("evidence_source_prior_mean_x_arcsec", 0.0)),
        evidence_source_prior_mean_y_arcsec=float(saved_args.get("evidence_source_prior_mean_y_arcsec", 0.0)),
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


CRITICAL_DET_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "family_id",
    "image_label",
    "x_obs",
    "y_obs",
    "z_source",
    "effective_z_source",
    "detA",
    "abs_detA",
    "a00",
    "a01",
    "a10",
    "a11",
)


@dataclass(frozen=True)
class CriticalDetDiagnosticResult:
    flagged: pd.DataFrame
    min_abs_detA: float
    total_images: int


def _empty_critical_det_diagnostic_table() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CRITICAL_DET_DIAGNOSTIC_COLUMNS))


def _stage3_critical_det_metadata_for_bin(state: BuildState, bin_data: TracedBinData) -> list[dict[str, Any]]:
    family_by_id = {str(family.family_id): family for family in state.family_data}
    family_ids = tuple(str(family_id) for family_id in bin_data.family_ids)
    family_index_per_image = np.asarray(bin_data.family_index_per_image, dtype=int)
    x_obs = np.asarray(bin_data.x_obs, dtype=float)
    y_obs = np.asarray(bin_data.y_obs, dtype=float)
    occurrence_by_family = {family_id: 0 for family_id in family_ids}
    metadata: list[dict[str, Any]] = []
    for image_index, family_index in enumerate(family_index_per_image.tolist()):
        if family_index < 0 or family_index >= len(family_ids):
            raise ValueError(f"Invalid family index {family_index} in effective_z_source={bin_data.effective_z_source}.")
        family_id = family_ids[family_index]
        family = family_by_id.get(family_id)
        if family is None:
            raise ValueError(f"Bin effective_z_source={bin_data.effective_z_source} references unknown family_id={family_id!r}.")
        within_family_index = occurrence_by_family[family_id]
        occurrence_by_family[family_id] = within_family_index + 1
        if within_family_index >= family.n_images:
            raise ValueError(f"Bin effective_z_source={bin_data.effective_z_source} has too many images for family_id={family_id!r}.")
        metadata.append(
            {
                "family_id": family_id,
                "image_label": str(family.image_labels[within_family_index]),
                "x_obs": float(x_obs[image_index]),
                "y_obs": float(y_obs[image_index]),
                "z_source": float(family.z_source),
                "effective_z_source": float(family.effective_z_source),
            }
        )
    return metadata


def _stage3_critical_det_diagnostic_from_evaluator(
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    best_fit_physical: np.ndarray,
    *,
    threshold: float,
) -> CriticalDetDiagnosticResult:
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("critical det diagnostic threshold must be finite and positive.")
    best_fit_physical = np.asarray(best_fit_physical, dtype=float)
    if best_fit_physical.shape[0] != len(state.parameter_specs):
        raise ValueError(
            f"Cannot run critical det diagnostic: best_fit length {best_fit_physical.shape[0]} "
            f"does not match {len(state.parameter_specs)} parameters."
        )
    best_fit_latent = np.asarray(evaluator.reported_physical_to_latent_parameter_vector(best_fit_physical), dtype=float)
    physical_params = evaluator._physical_parameter_vector(jnp.asarray(best_fit_latent, dtype=jnp.float64))
    sampled_kpc_per_arcsec = None
    sampled_dpie_sigma0_factors = None
    if bool(getattr(evaluator, "fit_cosmology_flat_wcdm", False)):
        sampled_kpc_per_arcsec, sampled_dpie_sigma0_factors = evaluator._sampled_cosmology_geometry_for_physical(
            physical_params
        )

    rows: list[dict[str, Any]] = []
    total_images = 0
    finite_abs_det_values: list[np.ndarray] = []
    for bin_data in evaluator.traced_bin_data:
        metadata = _stage3_critical_det_metadata_for_bin(state, bin_data)
        total_images += len(metadata)
        packed_kwargs: dict[str, Any] = {}
        if sampled_dpie_sigma0_factors is not None and int(getattr(bin_data, "effective_z_index", -1)) >= 0:
            packed_kwargs["kpc_per_arcsec"] = sampled_kpc_per_arcsec
            packed_kwargs["dpie_sigma0_factor"] = jnp.take(
                sampled_dpie_sigma0_factors,
                jnp.asarray(int(bin_data.effective_z_index), dtype=jnp.int32),
            )
        packed_state, validity = evaluator._build_packed_lens_state_with_validity_from_physical(
            physical_params,
            float(bin_data.effective_z_source),
            stop_gradient=True,
            **packed_kwargs,
        )
        is_valid = bool(np.asarray(validity.get("is_valid", True), dtype=bool))
        if is_valid:
            a00, a01, a10, a11 = evaluator._lensing_jacobian_for_components(
                float(bin_data.effective_z_source),
                bin_data.x_obs,
                bin_data.y_obs,
                packed_state,
            )
            a00_np = np.atleast_1d(np.asarray(a00, dtype=float))
            a01_np = np.atleast_1d(np.asarray(a01, dtype=float))
            a10_np = np.atleast_1d(np.asarray(a10, dtype=float))
            a11_np = np.atleast_1d(np.asarray(a11, dtype=float))
        else:
            n_images = len(metadata)
            a00_np = np.full(n_images, np.nan, dtype=float)
            a01_np = np.full(n_images, np.nan, dtype=float)
            a10_np = np.full(n_images, np.nan, dtype=float)
            a11_np = np.full(n_images, np.nan, dtype=float)
        det_np = a00_np * a11_np - a01_np * a10_np
        abs_det_np = np.abs(det_np)
        finite_abs_det = abs_det_np[np.isfinite(abs_det_np)]
        if finite_abs_det.size:
            finite_abs_det_values.append(finite_abs_det)
        if len(metadata) != det_np.shape[0]:
            raise ValueError(
                "Critical det diagnostic metadata/image count mismatch for "
                f"effective_z_source={bin_data.effective_z_source}: metadata={len(metadata)} detA={det_np.shape[0]}."
            )
        for image_index, base_row in enumerate(metadata):
            abs_det = float(abs_det_np[image_index])
            if not np.isfinite(abs_det) or abs_det >= threshold:
                continue
            rows.append(
                {
                    **base_row,
                    "detA": float(det_np[image_index]),
                    "abs_detA": abs_det,
                    "a00": float(a00_np[image_index]),
                    "a01": float(a01_np[image_index]),
                    "a10": float(a10_np[image_index]),
                    "a11": float(a11_np[image_index]),
                }
            )

    if finite_abs_det_values:
        min_abs_det = float(np.min(np.concatenate(finite_abs_det_values)))
    else:
        min_abs_det = float("nan")
    flagged = pd.DataFrame(rows, columns=list(CRITICAL_DET_DIAGNOSTIC_COLUMNS))
    return CriticalDetDiagnosticResult(flagged=flagged, min_abs_detA=min_abs_det, total_images=total_images)


def _log_stage3_critical_det_diagnostic(
    args: argparse.Namespace,
    *,
    result: CriticalDetDiagnosticResult,
    threshold: float,
    stage3_run_dir: Path,
    table_path: Path,
) -> None:
    min_abs = result.min_abs_detA
    min_abs_text = f"{min_abs:.6e}" if np.isfinite(min_abs) else "nan"
    _log(
        args,
        (
            f"[critical-det] flagged={len(result.flagged)} threshold={float(threshold):.6g} "
            f"min_abs_detA={min_abs_text} stage={stage3_run_dir}"
        ),
    )
    if result.flagged.empty:
        _log(args, f"[critical-det] no images with abs(detA) < {float(threshold):.6g}")
    else:
        for row in result.flagged.itertuples(index=False):
            _log(
                args,
                (
                    f"[critical-det] family_id={row.family_id} image_label={row.image_label} "
                    f"x_obs={float(row.x_obs):.6g} y_obs={float(row.y_obs):.6g} "
                    f"z_source={float(row.z_source):.6g} effective_z_source={float(row.effective_z_source):.6g} "
                    f"detA={float(row.detA):.6e} abs_detA={float(row.abs_detA):.6e}"
                ),
            )
    _log(args, f"[critical-det] table written to {table_path}")


def _write_critical_det_diagnostic_table(result: CriticalDetDiagnosticResult, table_path: Path) -> Path:
    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    result.flagged.to_csv(table_path, index=False)
    return table_path


def _run_stage3_critical_det_diagnostic(
    args: argparse.Namespace,
    stage3_run_dir: Path,
) -> CriticalDetDiagnosticResult:
    threshold = float(getattr(args, "critical_det_diagnostic_threshold", DEFAULT_CRITICAL_DET_DIAGNOSTIC_THRESHOLD))
    artifacts_dir = Path(stage3_run_dir) / "artifacts"
    h5_path = artifacts_dir / "plot_bundle.h5"
    if not h5_path.exists():
        _log(args, f"[critical-det] skipped; missing stage3 plot bundle at {h5_path}")
        return CriticalDetDiagnosticResult(
            flagged=_empty_critical_det_diagnostic_table(),
            min_abs_detA=float("nan"),
            total_images=0,
        )
    state, saved_args, arrays, init_diagnostics = _load_artifacts(artifacts_dir)
    arrays, _converted = _maybe_convert_loaded_posterior_arrays_to_physical(arrays, state.parameter_specs, init_diagnostics)
    best_fit_physical = np.asarray(arrays["best_fit"], dtype=float)
    diagnostic_saved_args = dict(saved_args)
    diagnostic_saved_args.update(_normalized_saved_likelihood_stabilizer_args(saved_args))
    _drop_legacy_likelihood_stabilizer_args(diagnostic_saved_args)
    diagnostic_saved_args["quick_diagnostics"] = False
    diagnostic_args = _clone_args(args, **diagnostic_saved_args)
    evaluator = _build_cluster_evaluator_from_args(
        diagnostic_args,
        state,
        sampling_engine=SAMPLING_ENGINE_FULL,
        quick_diagnostics=False,
    )
    try:
        result = _stage3_critical_det_diagnostic_from_evaluator(
            state,
            evaluator,
            best_fit_physical,
            threshold=threshold,
        )
    finally:
        release = getattr(evaluator, "release_runtime_caches", None)
        if callable(release):
            release()
    table_path = _write_critical_det_diagnostic_table(
        result,
        Path(stage3_run_dir) / "tables" / "critical_det_images.csv",
    )
    _log_stage3_critical_det_diagnostic(
        args,
        result=result,
        threshold=threshold,
        stage3_run_dir=Path(stage3_run_dir),
        table_path=table_path,
    )
    return result


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
    if not bool(getattr(args, "fit_cosmology_flat_wcdm", False)):
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


def _apply_coupled_physical_init_values(
    parameter_specs: list[ParameterSpec],
    init_values: dict[str, float] | None,
    physical_values: dict[str, float],
) -> dict[str, float] | None:
    if not physical_values:
        return init_values
    merged = dict(init_values or {})
    default_physical = _convert_theta_to_physical(_default_theta(parameter_specs), parameter_specs)
    for site in _parameter_sample_sites(parameter_specs):
        if len(site.indices) != 2:
            continue
        if not all(_is_potfile_mass_size_spec(parameter_specs[idx]) for idx in site.indices):
            continue
        if not any(parameter_specs[idx].sample_name in physical_values for idx in site.indices):
            continue
        physical_theta = np.asarray(default_physical, dtype=float).copy()
        for idx in site.indices:
            spec = parameter_specs[idx]
            if spec.sample_name in physical_values:
                physical_theta[idx] = float(physical_values[spec.sample_name])
        latent_theta = _clip_theta_to_support(
            _convert_theta_to_latent(physical_theta, parameter_specs),
            parameter_specs,
            boundary_frac=DEFAULT_NUTS_INIT_BOUNDARY_FRAC,
        )
        for idx in site.indices:
            spec = parameter_specs[idx]
            merged[spec.sample_name] = float(latent_theta[idx])
    return merged or None


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
    parsed, _potentials_df, images_df, arcs_df, potentials_with_priors = load_best_par(args.par_path)
    if images_df.empty and arcs_df.empty:
        raise ValueError("No multiple-image or CAB arc constraints found in the parsed catalogs.")
    reference = _extract_reference(parsed)
    if reference is None:
        raise ValueError("runmode.reference is required to convert image and CAB arc coordinates into solver offsets.")
    cosmo_config = _cosmology_config_override_from_args(args) or _build_cosmology(parsed)
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
    fov_limit = _fov_limit_from_args(args)
    if fov_limit is not None:
        fov_description = _format_fov_limit(fov_limit)
        image_fov_summary: dict[str, Any] = {}
        if not images_df.empty:
            images_df, image_fov_summary = _filter_images_by_fov(images_df, reference, fov_limit)
        potfiles, potfile_fov_summary = _filter_potfiles_by_fov(potfiles, reference, fov_limit)
        if image_fov_summary:
            _log(
                args,
                (
                    f"[input] fov_limit={fov_description} "
                    f"images_kept={image_fov_summary.get('kept', 0)}/{image_fov_summary.get('total', 0)} "
                    f"image_rows_dropped={image_fov_summary.get('dropped', 0)} "
                    f"affected_families={len(image_fov_summary.get('affected_family_ids', []))} "
                    f"removed_families={len(image_fov_summary.get('removed_family_ids', []))} "
                    f"removed_ids={','.join(image_fov_summary.get('removed_family_ids', []))}"
                ),
            )
        if potfile_fov_summary:
            _log(args, f"[input] fov_limit={fov_description} potfile_catalog_rows={json.dumps(potfile_fov_summary, sort_keys=True)}")
        if images_df.empty and arcs_df.empty:
            raise ValueError("No multiple-image or CAB arc constraints remain after applying FOV limits.")
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
            svi_init_values[spec.sample_name] = _initial_latent_value_from_physical(physical_value, spec)
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
            potfile_mass_size_reparam=bool(getattr(args, "potfile_mass_size_reparam", False)),
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
    family_data: list[FamilyData] = []
    family_redshift_binning_sec = 0.0
    if not images_df.empty:
        fit_images_df, n_nonpositive_images_skipped, n_nonpositive_families_skipped, nonpositive_family_ids = (
            _filter_non_positive_redshift_families(images_df)
        )
        if fit_images_df.empty:
            if arcs_df.empty:
                raise ValueError(
                    "No positive-redshift image families remain after dropping non-positive-redshift families. "
                    "At least one family must have finite catalog_z > 0."
                )
        else:
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
                if arcs_df.empty:
                    raise ValueError(
                        "No multi-image families remain after dropping singleton pseudo-families. "
                        "At least one family must contain two or more images."
                    )
            else:
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
    arc_data = _prepare_arc_constraint_data(arcs_df, reference)
    sample_likelihood_mode = str(getattr(args, "sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE))
    source_position_parameterization = SOURCE_POSITION_PARAMETERIZATION_DIRECT
    if family_data and _sample_likelihood_uses_explicit_beta(sample_likelihood_mode):
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
        if getattr(args, "fix_image_sigma_int_arcsec", None) is None:
            parameter_specs.append(
                _build_image_scatter_parameter_spec(
                    start_index=len(parameter_specs),
                    upper_arcsec=float(getattr(args, "image_plane_scatter_upper_arcsec", DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC)),
                    floor_arcsec=float(getattr(args, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)),
                    prior=str(getattr(args, "image_plane_scatter_prior", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR)),
                    prior_median_arcsec=float(
                        getattr(args, "image_plane_scatter_prior_median_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC)
                    ),
                    prior_log_sigma=float(
                        getattr(args, "image_plane_scatter_prior_log_sigma", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA)
                    ),
                )
            )
    elif family_data:
        parameter_specs.append(_build_source_scatter_parameter_spec(start_index=len(parameter_specs)))
    if fit_cosmology_flat_wcdm:
        parameter_specs.extend(_build_cosmology_parameter_specs(len(parameter_specs), cosmo_config))
        if svi_init_values is None:
            svi_init_values = {}
        stage1_map_values = stage1_prior_summary.map_values if stage1_prior_summary is not None else {}
        svi_init_values.setdefault(
            COSMOLOGY_OM0_SAMPLE_NAME,
            float(
                np.clip(
                    stage1_map_values.get(COSMOLOGY_OM0_SAMPLE_NAME, _om0_from_config(cosmo_config)),
                    DEFAULT_COSMOLOGY_OM0_LOWER,
                    DEFAULT_COSMOLOGY_OM0_UPPER,
                )
            ),
        )
        svi_init_values.setdefault(
            COSMOLOGY_W0_SAMPLE_NAME,
            float(
                np.clip(
                    stage1_map_values.get(COSMOLOGY_W0_SAMPLE_NAME, _w0_from_config(cosmo_config)),
                    DEFAULT_COSMOLOGY_W0_LOWER,
                    DEFAULT_COSMOLOGY_W0_UPPER,
                )
            ),
        )
    if svi_init_physical_values or source_position_prior_values:
        if svi_init_values is None:
            svi_init_values = {}
        coupled_physical_init_values: dict[str, float] = {}
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
                        float(getattr(args, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)),
                        float(getattr(args, "image_plane_scatter_upper_arcsec", DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC)),
                    )
                )
            if physical_value is None:
                continue
            if _is_potfile_mass_size_spec(spec):
                coupled_physical_init_values[spec.sample_name] = float(physical_value)
                continue
            if (
                spec.component_family == "source_position"
                and str(getattr(args, "source_position_parameterization", SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED))
                == SOURCE_POSITION_PARAMETERIZATION_CONDITIONAL_WHITENED
            ):
                svi_init_values[spec.sample_name] = 0.0
            else:
                svi_init_values[spec.sample_name] = _initial_latent_value_from_physical(physical_value, spec)
        svi_init_values = _apply_coupled_physical_init_values(
            parameter_specs,
            svi_init_values,
            coupled_physical_init_values,
        )
    svi_init_values = _apply_cosmology_init_overrides(args, parameter_specs, svi_init_values)
    if not family_data and arc_data is not None and int(getattr(arc_data, "n_arcs", 0)) > 0 and not parameter_specs:
        raise ValueError("CAB arc-only runs require at least one model parameter to constrain.")
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
        arc_data=arc_data,
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
    if spec.prior_kind == "truncated_normal":
        high = None if not np.isfinite(float(spec.upper)) else float(spec.upper)
        return dist.TruncatedNormal(
            float(spec.mean),
            float(spec.std),
            low=float(spec.lower),
            high=high,
            validate_args=True,
        )
    return dist.Uniform(float(spec.lower), float(spec.upper))


def _distribution_for_sample_site(site: _SampleSiteSpec, parameter_specs: list[ParameterSpec]):
    if len(site.indices) == 1:
        return _distribution_for_spec(parameter_specs[site.indices[0]])
    if len(site.indices) == 2:
        first_idx, second_idx = site.indices
        first_spec = parameter_specs[first_idx]
        second_spec = parameter_specs[second_idx]
        if _is_potfile_mass_size_spec(first_spec) and _is_potfile_mass_size_spec(second_spec):
            role_to_spec = {
                str(first_spec.coupled_role): first_spec,
                str(second_spec.coupled_role): second_spec,
            }
            mass_spec = role_to_spec.get(COUPLED_ROLE_MASS_NORM)
            size_spec = role_to_spec.get(COUPLED_ROLE_SIZE)
            if mass_spec is None or size_spec is None:
                raise ValueError(f"Potfile mass-size sample site {site.name!r} is missing mass or size role.")
            return _PotfileMassSizeReparamDistribution(
                _distribution_for_spec(mass_spec),
                _distribution_for_spec(size_spec),
                mass_center=float(mass_spec.coupled_mass_center),
                mass_scale=float(mass_spec.coupled_mass_scale),
                size_center=float(mass_spec.coupled_size_center),
                size_scale=float(mass_spec.coupled_size_scale),
            )
    raise ValueError(f"Unsupported vector sample site {site.name!r} with {len(site.indices)} components.")


def _prior_log_prob(parameter_specs: list[ParameterSpec], theta: jnp.ndarray) -> jnp.ndarray:
    theta_array = jnp.asarray(theta, dtype=jnp.float64)
    transform_kind_array = np.asarray([str(spec.transform_kind) for spec in parameter_specs], dtype=object)
    coupling_arrays = _potfile_mass_size_coupling_arrays(parameter_specs)
    physical_theta = _apply_parameter_transforms_jax(
        theta_array,
        jnp.asarray(transform_kind_array == "log_positive", dtype=bool),
        jnp.asarray(transform_kind_array == "log_offset_positive", dtype=bool),
        jnp.asarray([float(spec.transform_offset) for spec in parameter_specs], dtype=jnp.float64),
        jnp.asarray(transform_kind_array == "affine", dtype=bool),
        jnp.asarray([float(getattr(spec, "transform_scale", 1.0)) for spec in parameter_specs], dtype=jnp.float64),
        jnp.asarray(coupling_arrays["mass_indices"], dtype=jnp.int32),
        jnp.asarray(coupling_arrays["size_indices"], dtype=jnp.int32),
        jnp.asarray(coupling_arrays["mass_centers"], dtype=jnp.float64),
        jnp.asarray(coupling_arrays["mass_scales"], dtype=jnp.float64),
        jnp.asarray(coupling_arrays["size_centers"], dtype=jnp.float64),
        jnp.asarray(coupling_arrays["size_scales"], dtype=jnp.float64),
        jnp.asarray(coupling_arrays["cut_offsets"], dtype=jnp.float64),
    )
    sample_index = {spec.sample_name: idx for idx, spec in enumerate(parameter_specs)}
    total = jnp.array(0.0, dtype=jnp.float64)
    for site in _parameter_sample_sites(parameter_specs):
        idx = int(site.indices[0])
        spec = parameter_specs[idx]
        if spec.prior_kind == "hierarchical_normal":
            if len(site.indices) != 1:
                raise ValueError(f"Hierarchical vector sample site {site.name!r} is unsupported.")
            if not spec.parent_sample_name:
                raise ValueError(f"Hierarchical parameter {spec.name} is missing parent_sample_name.")
            parent_idx = sample_index[spec.parent_sample_name]
            total = total + dist.Normal(float(spec.mean or 0.0), physical_theta[parent_idx]).log_prob(theta_array[idx])
        else:
            total = total + _distribution_for_sample_site(site, parameter_specs).log_prob(
                _site_value_from_theta(theta_array, site)
            )
    return total


def _posterior_model(parameter_specs: list[ParameterSpec], evaluator: ClusterJAXEvaluator):
    def model():
        values: list[Any] = [None for _ in parameter_specs]
        sampled: dict[str, Any] = {}
        spec_by_sample = {spec.sample_name: spec for spec in parameter_specs}
        for site in _parameter_sample_sites(parameter_specs):
            spec = parameter_specs[site.indices[0]]
            if spec.prior_kind == "hierarchical_normal":
                if len(site.indices) != 1:
                    raise ValueError(f"Hierarchical vector sample site {site.name!r} is unsupported.")
                if not spec.parent_sample_name:
                    raise ValueError(f"Hierarchical parameter {spec.name} is missing parent_sample_name.")
                parent_value = _latent_to_physical_jax(sampled[spec.parent_sample_name], spec_by_sample[spec.parent_sample_name])
                value = numpyro.sample(
                    site.name,
                    dist.Normal(float(spec.mean or 0.0), jnp.asarray(parent_value, dtype=jnp.float64)),
                )
            else:
                value = numpyro.sample(site.name, _distribution_for_sample_site(site, parameter_specs))
            if len(site.indices) == 1:
                sampled[spec.sample_name] = value
                values[site.indices[0]] = value
            else:
                for offset, value_idx in enumerate(site.indices):
                    component_value = value[..., offset]
                    sampled[parameter_specs[value_idx].sample_name] = component_value
                    values[value_idx] = component_value
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
    arrays = [_site_array_for_spec(samples_dict, spec) for spec in parameter_specs]
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
    arrays = [_site_array_for_spec(samples_dict, spec) for spec in parameter_specs]
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
            {key: _filter_extra_field_by_chain_mask(value, np.ones(requested_chains, dtype=bool)) for key, value in extra.items()},
            diagnostics,
        )
    arrays = [_site_array_for_spec(samples_dict, spec) for spec in parameter_specs]
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
        key: _filter_extra_field_by_chain_mask(value, valid_chain_mask)
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


def _build_cluster_evaluator_from_args(
    args: argparse.Namespace,
    state: BuildState,
    *,
    sampling_engine: str | None = None,
    quick_diagnostics: bool | None = None,
) -> ClusterJAXEvaluator:
    sample_likelihood_mode = str(getattr(args, "sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE))
    image_presence_penalty_weight = _effective_image_presence_penalty_weight(
        getattr(args, "image_presence_penalty_weight", None),
        sample_likelihood_mode=sample_likelihood_mode,
        fit_mode=str(getattr(args, "fit_mode", FIT_MODE_SEQUENTIAL)),
        image_plane_mode=str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE)),
    )
    cab_likelihood_weight = _effective_cab_likelihood_weight(
        getattr(args, "cab_likelihood_weight", None),
        state,
    )
    return ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=args.match_tolerance_arcsec,
        sampling_engine=str(sampling_engine if sampling_engine is not None else getattr(args, "sampling_engine", SAMPLING_ENGINE_FULL)),
        active_scaling_galaxies=args.active_scaling_galaxies,
        active_scaling_selection=args.active_scaling_selection,
        active_scaling_cumulative_fraction=args.active_scaling_cumulative_fraction,
        active_scaling_min=args.active_scaling_min,
        refresh_every=args.refresh_every,
        refresh_param_drift_frac=args.refresh_param_drift_frac,
        source_plane_covariance_floor=args.source_plane_covariance_floor,
        source_plane_outlier_sigma_arcsec=args.source_plane_outlier_sigma_arcsec,
        sample_likelihood_mode=sample_likelihood_mode,
        image_plane_newton_steps=int(getattr(args, "image_plane_newton_steps", 0)),
        anchored_image_plane_solve_steps=int(
            getattr(args, "anchored_image_plane_solve_steps", DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS)
        ),
        anchored_image_plane_trust_radius_arcsec=float(
            getattr(
                args,
                "anchored_image_plane_trust_radius_arcsec",
                DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
            )
        ),
        anchored_image_plane_lm_damping_relative=float(
            getattr(
                args,
                "anchored_image_plane_lm_damping_relative",
                DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
            )
        ),
        anchored_image_plane_lm_damping_absolute=float(
            getattr(
                args,
                "anchored_image_plane_lm_damping_absolute",
                DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
            )
        ),
        critical_arc_critical_direction_sigma_arcsec=float(
            getattr(args, "critical_arc_critical_direction_sigma_arcsec", DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC)
        ),
        critical_arc_base_prob=float(
            getattr(args, "critical_arc_base_prob", DEFAULT_CRITICAL_ARC_BASE_PROB)
        ),
        critical_arc_max_prob=float(
            getattr(args, "critical_arc_max_prob", DEFAULT_CRITICAL_ARC_MAX_PROB)
        ),
        critical_arc_singular_threshold=float(
            getattr(args, "critical_arc_singular_threshold", DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD)
        ),
        critical_arc_singular_softness=float(
            getattr(args, "critical_arc_singular_softness", DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS)
        ),
        critical_arc_lm_damping_relative=float(
            getattr(args, "critical_arc_lm_damping_relative", DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE)
        ),
        critical_arc_lm_damping_absolute=float(
            getattr(args, "critical_arc_lm_damping_absolute", DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE)
        ),
        critical_arc_lm_trust_radius_arcsec=float(
            getattr(args, "critical_arc_lm_trust_radius_arcsec", DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC)
        ),
        arc_aware_noncritical_support_radius_arcsec=float(
            getattr(
                args,
                "arc_aware_noncritical_support_radius_arcsec",
                DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC,
            )
        ),
        arc_aware_max_arclength_arcsec=float(
            getattr(args, "arc_aware_max_arclength_arcsec", DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC)
        ),
        arc_aware_curve_step_arcsec=float(
            getattr(args, "arc_aware_curve_step_arcsec", DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC)
        ),
        fold_curvature_arcsec_inv=float(
            getattr(args, "fold_curvature_arcsec_inv", DEFAULT_FOLD_CURVATURE_ARCSEC_INV)
        ),
        cab_likelihood_weight=cab_likelihood_weight,
        cab_finite_difference_step_arcsec=float(
            getattr(args, "cab_finite_difference_step_arcsec", DEFAULT_CAB_FINITE_DIFFERENCE_STEP_ARCSEC)
        ),
        cab_tangent_sigma_floor_rad=float(
            getattr(args, "cab_tangent_sigma_floor_rad", DEFAULT_CAB_TANGENT_SIGMA_FLOOR_RAD)
        ),
        cab_curvature_sigma_floor_arcsec_inv=float(
            getattr(args, "cab_curvature_sigma_floor_arcsec_inv", DEFAULT_CAB_CURVATURE_SIGMA_FLOOR_ARCSEC_INV)
        ),
        image_plane_scatter_floor_arcsec=float(
            getattr(args, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)
        ),
        fixed_image_sigma_int_arcsec=getattr(args, "fix_image_sigma_int_arcsec", None),
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
        likelihood_stabilizer_max_gain=float(
            getattr(args, "likelihood_stabilizer_max_gain", DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN)
        ),
        likelihood_stabilizer_max_residual_arcsec=float(
            getattr(args, "likelihood_stabilizer_max_residual_arcsec", DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC)
        ),
        likelihood_stabilizer_residual_loss=str(
            getattr(args, "likelihood_stabilizer_residual_loss", DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS)
        ),
        likelihood_stabilizer_student_t_nu=float(
            getattr(args, "likelihood_stabilizer_student_t_nu", DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU)
        ),
        evidence_source_prior_sigma_arcsec=getattr(args, "evidence_source_prior_sigma_arcsec", None),
        evidence_source_prior_mean_x_arcsec=float(getattr(args, "evidence_source_prior_mean_x_arcsec", 0.0)),
        evidence_source_prior_mean_y_arcsec=float(getattr(args, "evidence_source_prior_mean_y_arcsec", 0.0)),
        quick_diagnostics=bool(
            quick_diagnostics if quick_diagnostics is not None else getattr(args, "quick_diagnostics", False)
        ),
    )


def _prepare_direct_evaluator(
    args: argparse.Namespace,
    state: BuildState,
) -> tuple[ClusterJAXEvaluator, np.ndarray]:
    evaluator = _build_cluster_evaluator_from_args(args, state)
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


def _output_evaluator_for_validation(
    args: argparse.Namespace,
    state: BuildState,
    fit_evaluator: ClusterJAXEvaluator,
    best_fit_latent: np.ndarray,
) -> ClusterJAXEvaluator:
    if (
        str(getattr(fit_evaluator, "sampling_engine", SAMPLING_ENGINE_FULL)) != SAMPLING_ENGINE_ACTIVE_SUBSET
        or not fit_evaluator._active_subset_effective()
    ):
        return fit_evaluator
    _log(
        args,
        (
            "[active-subset] building full-model evaluator for final validation and plots; "
            "posterior log_prob remains the active-subset fit target"
        ),
    )
    full_args = _clone_args(args, sampling_engine=SAMPLING_ENGINE_FULL)
    output_evaluator = _build_cluster_evaluator_from_args(
        full_args,
        state,
        sampling_engine=SAMPLING_ENGINE_FULL,
    )
    output_evaluator.fit_sampling_engine = str(getattr(fit_evaluator, "sampling_engine", SAMPLING_ENGINE_ACTIVE_SUBSET))
    output_evaluator.fit_active_scaling_components = int(len(fit_evaluator.active_scaling_component_indices))
    output_evaluator.fit_ignored_inactive_scaling_components = int(len(fit_evaluator.inactive_scaling_component_indices))
    output_evaluator.final_validation_sampling_engine = SAMPLING_ENGINE_FULL
    _log_evaluator_summary(args, output_evaluator)
    output_evaluator.refresh_scaling_scatter_cache(best_fit_latent, reason="active_subset_full_output")
    output_evaluator.refresh_source_metric_cache(best_fit_latent, reason="active_subset_full_output")
    return output_evaluator


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
        validation_evaluator = _output_evaluator_for_validation(args, state, evaluator, best_fit)
        if args.skip_validation or bool(getattr(args, "quick_diagnostics", False)):
            reason = "--quick-diagnostics" if bool(getattr(args, "quick_diagnostics", False)) else "--skip-validation"
            _log(args, f"[validation] skipped by {reason}; using source-plane summary only")
            best_eval = _run_logged_phase(
                args,
                "validation.approximate",
                lambda: _approximate_evaluation(validation_evaluator, best_fit),
            )
        else:
            validation_start = time.time()
            _log(args, "[validation] computing source-plane summary")
            best_eval = _run_logged_phase(
                args,
                "validation.source_summary",
                lambda: validation_evaluator.evaluate(best_fit),
            )
            validation_elapsed = time.time() - validation_start
            validation_evaluator.timing_totals["validation_runtime"] += validation_elapsed
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
            validation_evaluator.release_runtime_caches()
            if validation_evaluator is not evaluator:
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
                    evaluator=validation_evaluator,
                    best_fit=best_fit_physical,
                    best_eval=best_eval,
                    results=posterior_for_output,
                    runtime_sec=runtime_sec,
                    args=args,
                ),
            )
            plot_elapsed = time.time() - plot_start
            validation_evaluator.timing_totals["plot_runtime"] += plot_elapsed
            _log(args, f"[output] complete in {_fmt_seconds(plot_elapsed)} run_dir={run_dir}")
            validation_evaluator.release_runtime_caches()
            if validation_evaluator is not evaluator:
                evaluator.release_runtime_caches()
        _log(args, f"[done] total_runtime={_fmt_seconds(time.time() - start)}")
        return
    _log(args, f"[model] initializing direct evaluator for {args.fit_method}")
    evaluator, _midpoint = _prepare_direct_evaluator(args, state)
    sample_model = _posterior_model(state.parameter_specs, evaluator)
    if str(args.fit_method) == FIT_METHOD_SMC:
        best_fit = _reference_theta_from_init_values(state.parameter_specs, state.svi_init_values, _midpoint)
        if evaluator.surrogate_enabled:
            evaluator.refresh_surrogate(best_fit, reason="smc_initial")
        evaluator.refresh_scaling_scatter_cache(best_fit, reason="smc_initial")
        evaluator.refresh_source_metric_cache(best_fit, reason="smc_initial")
        posterior = _run_blackjax_smc_sampler(args, state, evaluator)
        if posterior.samples.ndim != 2 or posterior.samples.shape[0] == 0:
            raise RuntimeError("BlackJAX SMC returned no posterior particles.")
        if np.asarray(posterior.log_prob).size and np.isfinite(posterior.log_prob).any():
            best_index = int(np.nanargmax(posterior.log_prob))
        else:
            best_index = 0
        best_fit = np.asarray(posterior.samples[best_index], dtype=float)
        evaluator.refresh_scaling_scatter_cache(best_fit, reason="post_smc")
        evaluator.refresh_source_metric_cache(best_fit, reason="post_smc")
        _log(args, f"[smc] best_fit updated from retained particle index={best_index}")
    elif str(args.fit_method) == FIT_METHOD_NS:
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
    elif str(args.fit_method) == FIT_METHOD_NUTS:
        best_fit = _reference_theta_from_init_values(state.parameter_specs, state.svi_init_values, _midpoint)
        if evaluator.surrogate_enabled:
            evaluator.refresh_surrogate(best_fit, reason="nuts_initial")
        evaluator.refresh_scaling_scatter_cache(best_fit, reason="nuts_initial")
        evaluator.refresh_source_metric_cache(best_fit, reason="nuts_initial")
        nuts_init = _nuts_initialization_from_reference(args, state.parameter_specs, best_fit)
        _log(
            args,
            (
                "[nuts:init] from previous-stage/reference "
                f"distinct_seeds={int(nuts_init.diagnostics.get('distinct_chain_seeds', 0))}"
            ),
        )
        posterior = _run_numpyro_nuts_sampler(args, state, evaluator, sample_model, nuts_init)
        nuts_usable, nuts_reason = _nuts_posterior_is_usable(posterior)
        if not nuts_usable:
            raise RuntimeError(f"Direct NUTS posterior is unusable: {nuts_reason}")
        if posterior.samples.ndim != 2 or posterior.samples.shape[0] == 0:
            raise RuntimeError("Direct NUTS returned no posterior samples.")
        if np.asarray(posterior.log_prob).size and np.isfinite(posterior.log_prob).any():
            best_index = int(np.nanargmax(posterior.log_prob))
        else:
            best_index = 0
        best_fit = np.asarray(posterior.samples[best_index], dtype=float)
        evaluator.refresh_scaling_scatter_cache(best_fit, reason="post_nuts")
        evaluator.refresh_source_metric_cache(best_fit, reason="post_nuts")
        _log(args, f"[nuts] best_fit updated from retained posterior sample index={best_index}")
    else:
        best_fit, posterior, svi_diagnostics = _run_svi_fit(args, state, evaluator, sample_model)
        svi_posterior = posterior
        evaluator.refresh_scaling_scatter_cache(best_fit, reason="post_svi")
        evaluator.refresh_source_metric_cache(best_fit, reason="post_svi")
        svi_chain_seeds: list[ChainSeed] | None = None
        if str(args.fit_method) == FIT_METHOD_SVI_NUTS:
            svi_chain_seeds = _prepare_svi_health_for_nuts(
                args,
                state.parameter_specs,
                evaluator,
                best_fit,
                svi_posterior,
                svi_diagnostics,
            )
        _log(
            args,
            (
                "[svi:init] "
                f"final_elbo={float(svi_diagnostics.get('svi_final_elbo_loss', float('nan'))):.4g}"
            ),
        )
        if str(args.fit_method) == FIT_METHOD_SVI_NUTS:
            nuts_init = _nuts_initialization_from_svi_center(
                args,
                state.parameter_specs,
                best_fit,
                svi_diagnostics,
                chain_seeds=svi_chain_seeds,
            )
            _log(
                args,
                (
                    "[nuts:init] from svi "
                    f"distinct_seeds={int(nuts_init.diagnostics.get('distinct_chain_seeds', 0))}"
                ),
            )
            if _blocked_linearized_stage_enabled(args):
                nuts_posterior = _run_blocked_numpyro_nuts_sampler(args, state, evaluator, nuts_init)
            else:
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
    if str(getattr(evaluator, "sampling_engine", SAMPLING_ENGINE_FULL)) == SAMPLING_ENGINE_ACTIVE_SUBSET:
        try:
            posterior.init_diagnostics["fit_sampling_engine"] = SAMPLING_ENGINE_ACTIVE_SUBSET
            posterior.init_diagnostics["final_validation_sampling_engine"] = SAMPLING_ENGINE_FULL
            posterior.init_diagnostics["fit_active_subset_loglike"] = float(evaluator.source_loglike(best_fit))
            posterior.init_diagnostics["fit_active_scaling_components"] = int(len(evaluator.active_scaling_component_indices))
            posterior.init_diagnostics["fit_ignored_inactive_scaling_components"] = int(
                len(evaluator.inactive_scaling_component_indices)
            )
        except Exception as exc:  # pragma: no cover - diagnostics should never fail the fit
            _log(args, f"[active-subset] failed to compute fit-target diagnostic loglike: {exc}")
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
    validation_evaluator = _output_evaluator_for_validation(args, state, evaluator, best_fit)

    if args.skip_validation or bool(getattr(args, "quick_diagnostics", False)):
        reason = "--quick-diagnostics" if bool(getattr(args, "quick_diagnostics", False)) else "--skip-validation"
        _log(args, f"[validation] skipped by {reason}; using source-plane summary only")
        best_eval = _run_logged_phase(
            args,
            "validation.approximate",
            lambda: _approximate_evaluation(validation_evaluator, best_fit),
        )
    else:
        validation_start = time.time()
        _log(args, "[validation] computing source-plane summary")
        best_eval = _run_logged_phase(
            args,
            "validation.source_summary",
            lambda: validation_evaluator.evaluate(best_fit),
        )
        validation_elapsed = time.time() - validation_start
        validation_evaluator.timing_totals["validation_runtime"] += validation_elapsed
        _log(args, _validation_complete_message(validation_elapsed, best_eval))
    runtime_sec = time.time() - start

    _write_truth_validation_outputs(args, run_dir)
    if args.skip_plots:
        _log(args, "[output] plot generation skipped by --skip-plots")
        validation_evaluator.release_runtime_caches()
        if validation_evaluator is not evaluator:
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
                evaluator=validation_evaluator,
                best_fit=best_fit_physical,
                best_eval=best_eval,
                results=posterior_for_output,
                runtime_sec=runtime_sec,
                args=args,
            ),
        )
        plot_elapsed = time.time() - plot_start
        validation_evaluator.timing_totals["plot_runtime"] += plot_elapsed
        _log(args, f"[output] complete in {_fmt_seconds(plot_elapsed)} run_dir={run_dir}")
        validation_evaluator.release_runtime_caches()
        if validation_evaluator is not evaluator:
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
    if stage_name in {
        "stage4_linearized_image_plane",
        "stage4_blocked_linearized_image_plane",
        "stage4_forward_metric_image_plane",
        "stage4_anchored_solved_image_plane",
        "stage4_critical_arc_mixture_image_plane",
        "stage4_fold_regularized_image_plane",
    }:
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
    exact_stage3_diagnostics = _stage3_exact_image_diagnostics_enabled(args, run_dir)
    force_quick_diagnostics = (
        _is_sequential_stage_path(run_dir)
        and not exact_stage3_diagnostics
        and not _stage_allows_exact_image_diagnostics(run_dir, exact_diagnostics_stage)
    )
    quick_diagnostics = bool(
        getattr(args, "quick_diagnostics", False)
        or (saved_args.get("quick_diagnostics", False) and not exact_stage3_diagnostics)
        or force_quick_diagnostics
    )
    if force_quick_diagnostics and not bool(saved_args.get("quick_diagnostics", False)):
        _log(args, "[plots-only] quick diagnostics forced for pre-stage4 sequential stage")
    if exact_stage3_diagnostics and bool(saved_args.get("quick_diagnostics", False)):
        _log(args, "[plots-only] exact stage3 image diagnostics requested; ignoring saved quick_diagnostics=True")
    plot_saved_args = dict(saved_args)
    plot_saved_args.update(_normalized_saved_likelihood_stabilizer_args(saved_args))
    _drop_legacy_likelihood_stabilizer_args(plot_saved_args)
    plot_saved_args["quick_diagnostics"] = quick_diagnostics
    plot_saved_args["exact_image_diagnostics_stage3"] = bool(getattr(args, "exact_image_diagnostics_stage3", False))
    plot_saved_args["caustic_source_redshift"] = float(getattr(args, "caustic_source_redshift", 9.0))
    current_kappa_true_fits = getattr(args, "kappa_true_fits", None)
    if current_kappa_true_fits is not None and str(current_kappa_true_fits).strip():
        plot_saved_args["kappa_true_fits"] = str(current_kappa_true_fits)
    current_cutout_dir = getattr(args, "image_catalog_family_cutout_image_dir", None)
    if current_cutout_dir is not None and str(current_cutout_dir).strip():
        plot_saved_args["image_catalog_family_cutout_image_dir"] = current_cutout_dir
        plot_saved_args["image_catalog_family_cutout_image_scale"] = str(
            getattr(args, "image_catalog_family_cutout_image_scale", "60mas")
        )
        current_cutout_bands = getattr(args, "image_catalog_family_cutout_bands", None)
        if current_cutout_bands is not None:
            plot_saved_args["image_catalog_family_cutout_bands"] = list(current_cutout_bands)
    for cutout_rgb_key in (
        "image_catalog_family_cutout_rgb_q",
        "image_catalog_family_cutout_rgb_stretch",
        "image_catalog_family_cutout_rgb_minimum",
        "image_catalog_family_cutout_rgb_red_gain",
        "image_catalog_family_cutout_rgb_green_gain",
        "image_catalog_family_cutout_rgb_blue_gain",
    ):
        current_cutout_rgb_value = getattr(args, cutout_rgb_key, None)
        if current_cutout_rgb_value is not None:
            plot_saved_args[cutout_rgb_key] = float(current_cutout_rgb_value)
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
    cab_likelihood_weight = _effective_cab_likelihood_weight(saved_args.get("cab_likelihood_weight"), state)
    saved_sampling_engine = str(saved_args.get("sampling_engine", SAMPLING_ENGINE_FULL))
    plot_sampling_engine = (
        SAMPLING_ENGINE_FULL if saved_sampling_engine == SAMPLING_ENGINE_ACTIVE_SUBSET else saved_sampling_engine
    )
    if saved_sampling_engine == SAMPLING_ENGINE_ACTIVE_SUBSET:
        _log(
            args,
            "[plots-only] active_subset artifacts detected; using full lens model for validation and plots",
        )
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=float(saved_args.get("match_tolerance_arcsec", DEFAULT_MATCH_TOLERANCE)),
        sampling_engine=plot_sampling_engine,
        active_scaling_galaxies=saved_args.get("active_scaling_galaxies"),
        active_scaling_selection=str(saved_args.get("active_scaling_selection", "adaptive")),
        active_scaling_cumulative_fraction=float(
            saved_args.get("active_scaling_cumulative_fraction", DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION)
        ),
        active_scaling_min=int(saved_args.get("active_scaling_min", DEFAULT_ACTIVE_SCALING_MIN)),
        refresh_every=int(saved_args.get("refresh_every", DEFAULT_REFRESH_EVERY)),
        refresh_param_drift_frac=float(saved_args.get("refresh_param_drift_frac", DEFAULT_REFRESH_PARAM_DRIFT_FRAC)),
        source_plane_covariance_floor=float(saved_args.get("source_plane_covariance_floor", 1.0e-6)),
        source_plane_outlier_sigma_arcsec=float(
            saved_args.get("source_plane_outlier_sigma_arcsec", DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC)
        ),
        sample_likelihood_mode=sample_likelihood_mode,
        image_plane_newton_steps=int(saved_args.get("image_plane_newton_steps", 0)),
        anchored_image_plane_solve_steps=int(
            saved_args.get("anchored_image_plane_solve_steps", DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS)
        ),
        anchored_image_plane_trust_radius_arcsec=float(
            saved_args.get(
                "anchored_image_plane_trust_radius_arcsec",
                DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
            )
        ),
        anchored_image_plane_lm_damping_relative=float(
            saved_args.get(
                "anchored_image_plane_lm_damping_relative",
                DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
            )
        ),
        anchored_image_plane_lm_damping_absolute=float(
            saved_args.get(
                "anchored_image_plane_lm_damping_absolute",
                DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
            )
        ),
        critical_arc_critical_direction_sigma_arcsec=float(
            saved_args.get("critical_arc_critical_direction_sigma_arcsec", DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC)
        ),
        critical_arc_base_prob=float(
            saved_args.get("critical_arc_base_prob", DEFAULT_CRITICAL_ARC_BASE_PROB)
        ),
        critical_arc_max_prob=float(
            saved_args.get("critical_arc_max_prob", DEFAULT_CRITICAL_ARC_MAX_PROB)
        ),
        critical_arc_singular_threshold=float(
            saved_args.get("critical_arc_singular_threshold", DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD)
        ),
        critical_arc_singular_softness=float(
            saved_args.get("critical_arc_singular_softness", DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS)
        ),
        critical_arc_lm_damping_relative=float(
            saved_args.get("critical_arc_lm_damping_relative", DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE)
        ),
        critical_arc_lm_damping_absolute=float(
            saved_args.get("critical_arc_lm_damping_absolute", DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE)
        ),
        critical_arc_lm_trust_radius_arcsec=float(
            saved_args.get("critical_arc_lm_trust_radius_arcsec", DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC)
        ),
        arc_aware_noncritical_support_radius_arcsec=float(
            saved_args.get(
                "arc_aware_noncritical_support_radius_arcsec",
                DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC,
            )
        ),
        arc_aware_max_arclength_arcsec=float(
            saved_args.get("arc_aware_max_arclength_arcsec", DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC)
        ),
        arc_aware_curve_step_arcsec=float(
            saved_args.get("arc_aware_curve_step_arcsec", DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC)
        ),
        fold_curvature_arcsec_inv=float(
            saved_args.get("fold_curvature_arcsec_inv", DEFAULT_FOLD_CURVATURE_ARCSEC_INV)
        ),
        cab_likelihood_weight=cab_likelihood_weight,
        cab_finite_difference_step_arcsec=float(
            saved_args.get("cab_finite_difference_step_arcsec", DEFAULT_CAB_FINITE_DIFFERENCE_STEP_ARCSEC)
        ),
        cab_tangent_sigma_floor_rad=float(
            saved_args.get("cab_tangent_sigma_floor_rad", DEFAULT_CAB_TANGENT_SIGMA_FLOOR_RAD)
        ),
        cab_curvature_sigma_floor_arcsec_inv=float(
            saved_args.get("cab_curvature_sigma_floor_arcsec_inv", DEFAULT_CAB_CURVATURE_SIGMA_FLOOR_ARCSEC_INV)
        ),
        image_plane_scatter_floor_arcsec=float(
            saved_args.get("image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)
        ),
        fixed_image_sigma_int_arcsec=saved_args.get("fix_image_sigma_int_arcsec"),
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
        **_normalized_saved_likelihood_stabilizer_args(saved_args),
        evidence_source_prior_sigma_arcsec=saved_args.get("evidence_source_prior_sigma_arcsec"),
        evidence_source_prior_mean_x_arcsec=float(saved_args.get("evidence_source_prior_mean_x_arcsec", 0.0)),
        evidence_source_prior_mean_y_arcsec=float(saved_args.get("evidence_source_prior_mean_y_arcsec", 0.0)),
        quick_diagnostics=quick_diagnostics,
    )
    if saved_sampling_engine == SAMPLING_ENGINE_ACTIVE_SUBSET:
        evaluator.fit_sampling_engine = SAMPLING_ENGINE_ACTIVE_SUBSET
        evaluator.final_validation_sampling_engine = SAMPLING_ENGINE_FULL
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
        _log(args, "[plots-only] computing source-plane summary")
        best_eval = _run_logged_phase(
            args,
            "plots_only.validation.source_summary",
            lambda: evaluator.evaluate(best_fit_latent),
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
        "stage4_blocked_linearized_image_plane": "STAGE 4: stage4_blocked_linearized_image_plane",
        "stage4_forward_metric_image_plane": "STAGE 4: stage4_forward_metric_image_plane",
        "stage4_anchored_solved_image_plane": "STAGE 4: stage4_anchored_solved_image_plane",
        "stage4_critical_arc_mixture_image_plane": "STAGE 4: stage4_critical_arc_mixture_image_plane",
        "stage4_fold_regularized_image_plane": "STAGE 4: stage4_fold_regularized_image_plane",
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
            f"max_tree_depth={stage_args.max_tree_depth} "
            f"fit_cosmology_flat_wcdm={bool(getattr(stage_args, 'fit_cosmology_flat_wcdm', False))}"
        ),
    )
    _log(
        stage_args,
        (
            f"[stage] start run_name={run_name} fit_mode={fit_mode} fit_method={stage_args.fit_method} "
            f"sample_likelihood_mode={sample_likelihood_mode} "
            f"max_tree_depth={stage_args.max_tree_depth} "
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


def _final_enabled_sequential_stage_name(*, stage3_enabled: bool, stage4_enabled: bool) -> str:
    if stage4_enabled:
        return "stage4"
    if stage3_enabled:
        return "stage3"
    return "stage2"


def _require_resume_fast_stage1_artifacts(stage1_run_dir: Path) -> None:
    artifacts_dir = stage1_run_dir / "artifacts"
    if (artifacts_dir / "stage1_prior_summary.json").exists() or _has_plot_artifacts(artifacts_dir):
        return
    _fail(
        "--resume-fast requires existing stage1 artifacts at "
        f"{artifacts_dir}. Run the sequential workflow without --resume-fast first, "
        "or provide a completed stage1_large_only run."
    )


def _require_resume_fast_plot_artifacts(stage_run_dir: Path, stage_name: str) -> None:
    artifacts_dir = stage_run_dir / "artifacts"
    if _has_plot_artifacts(artifacts_dir):
        return
    _fail(
        "--resume-fast requires existing "
        f"{stage_name} plot artifacts at {artifacts_dir}. Run the sequential workflow "
        "without --resume-fast first, or restore the previous-stage artifacts."
    )


def _stage_artifact_cosmology_metadata(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir)
    metadata: dict[str, Any] = {}
    summary = _load_stage_run_summary_for_aggregate(path)
    if summary is not None and "fit_cosmology_flat_wcdm" in summary:
        metadata["fit_cosmology_flat_wcdm"] = bool(summary.get("fit_cosmology_flat_wcdm"))
    h5_path = path / "artifacts" / "plot_bundle.h5"
    if not h5_path.exists():
        return metadata
    try:
        with h5py.File(h5_path, "r") as handle:
            if "state" not in handle:
                return metadata
            state_meta = _read_h5_json(handle["state"], "build_state_meta_json", default={})
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return metadata
    if not isinstance(state_meta, dict):
        return metadata
    cosmo_config = state_meta.get("cosmo_config")
    if isinstance(cosmo_config, dict):
        metadata["cosmo_config"] = dict(cosmo_config)
    if "fit_cosmology_flat_wcdm" in state_meta:
        metadata["fit_cosmology_flat_wcdm"] = bool(state_meta.get("fit_cosmology_flat_wcdm"))
    elif any(
        isinstance(item, dict) and str(item.get("component_family")) == "cosmology"
        for item in state_meta.get("parameter_specs", [])
    ):
        metadata["fit_cosmology_flat_wcdm"] = True
    return metadata


def _cosmology_configs_close(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    if str(actual.get("class", "FlatLambdaCDM")) != str(expected.get("class", "FlatLambdaCDM")):
        return False
    for key in ("H0", "Om0", "Ode0", "w0"):
        if key not in actual or key not in expected:
            return False
        try:
            actual_value = float(actual[key])
            expected_value = float(expected[key])
        except (TypeError, ValueError):
            return False
        if not math.isclose(actual_value, expected_value, rel_tol=1.0e-9, abs_tol=1.0e-9):
            return False
    return True


def _stage_run_cosmology_compatible(stage_args: argparse.Namespace, run_dir: str | Path) -> tuple[bool, str]:
    expected_fit = bool(getattr(stage_args, "fit_cosmology_flat_wcdm", False))
    expected_config = _cosmology_config_override_from_args(stage_args)
    if not expected_fit and expected_config is None:
        return True, ""
    metadata = _stage_artifact_cosmology_metadata(run_dir)
    actual_fit = metadata.get("fit_cosmology_flat_wcdm")
    if actual_fit is not None and bool(actual_fit) != expected_fit:
        return (
            False,
            f"fit_cosmology_flat_wcdm is {bool(actual_fit)} but this stage now requires {expected_fit}",
        )
    if actual_fit is None and expected_fit:
        return False, "stored cosmology-fit metadata is missing"
    if expected_config is not None:
        actual_config = metadata.get("cosmo_config")
        if not isinstance(actual_config, dict):
            return False, "stored cosmology config metadata is missing"
        if not _cosmology_configs_close(actual_config, expected_config):
            return False, "stored cosmology config does not match the required fiducial config"
    return True, ""


def _require_resume_fast_cosmology_compatibility(stage_args: argparse.Namespace, run_dir: Path, stage_name: str) -> None:
    compatible, reason = _stage_run_cosmology_compatible(stage_args, run_dir)
    if compatible:
        return
    _fail(
        f"--resume-fast cannot reuse {stage_name} at {run_dir}: {reason}. "
        "Run again without --resume-fast so the stage can be regenerated."
    )


def _run_sequential(args: argparse.Namespace) -> None:
    stage_fit_controls = _normalize_stage_fit_controls(args)
    stage2_controls = stage_fit_controls["stage2"]
    stage3_controls = stage_fit_controls["stage3"]
    stage4_controls = stage_fit_controls["stage4"]
    stage4_enabled = _stage4_image_plane_enabled(args)
    stage3_enabled = _local_jacobian_stage_enabled(args)
    exact_diagnostics_stage = _final_sequential_exact_diagnostics_stage(
        args,
        stage3_enabled=stage3_enabled,
        stage4_enabled=stage4_enabled,
    )
    stage4_sample_likelihood_mode = _stage4_sample_likelihood_mode(args)
    fit_cosmology_requested = bool(getattr(args, "fit_cosmology_flat_wcdm", False))
    sequential_cosmology_config_override = (
        _sequential_fiducial_cosmology_config() if fit_cosmology_requested else None
    )

    def stage_cosmology_updates(*, fit_stage_cosmology: bool) -> dict[str, Any]:
        updates: dict[str, Any] = {"fit_cosmology_flat_wcdm": bool(fit_stage_cosmology)}
        if sequential_cosmology_config_override is not None:
            updates["cosmology_config_override"] = _sequential_fiducial_cosmology_config()
        return updates

    root_run_name = args.run_name or _make_run_name(args.par_path)
    _log_stage_banner(
        args,
        "SEQUENTIAL WORKFLOW",
        (
            f"run_name={root_run_name} image_plane_mode={getattr(args, 'image_plane_mode', IMAGE_PLANE_MODE_NONE)} "
            f"stage3={'enabled' if stage3_enabled else 'disabled'} "
            f"stage4={'enabled' if stage4_enabled else 'disabled'} "
            f"start_at_stage3={bool(getattr(args, 'start_at_stage3', False))} "
            f"resume_fast={bool(getattr(args, 'resume_fast', False))}"
        ),
    )
    _log(
        args,
        (
            f"[stage] sequential start run_name={args.run_name or '<auto>'} "
            f"stage2_fit_method={stage2_controls.fit_method} "
            f"stage2_max_tree_depth={stage2_controls.max_tree_depth} "
            f"stage3_fit_method={stage3_controls.fit_method if stage3_enabled else '<disabled>'} "
            f"stage3_max_tree_depth={stage3_controls.max_tree_depth if stage3_enabled else '<disabled>'} "
            f"stage4_fit_method={stage4_controls.fit_method if stage4_enabled else '<disabled>'}"
            f" stage4_max_tree_depth={stage4_controls.max_tree_depth if stage4_enabled else '<disabled>'}"
        ),
    )
    resume = bool(getattr(args, "resume", False))
    resume_fast = bool(getattr(args, "resume_fast", False))
    start_at_stage3 = bool(getattr(args, "start_at_stage3", False))
    final_stage_name = _final_enabled_sequential_stage_name(
        stage3_enabled=stage3_enabled,
        stage4_enabled=stage4_enabled,
    )

    def maybe_run_stage(
        stage_args: argparse.Namespace,
        fit_mode: str,
        run_name: str,
        **kwargs: Any,
    ) -> Path:
        run_dir = Path(stage_args.output_dir) / run_name
        if resume:
            if _stage_run_complete(run_dir):
                compatible, reason = _stage_run_cosmology_compatible(stage_args, run_dir)
                if not compatible:
                    _log(args, f"[resume] not reusing completed stage run_name={run_name} run_dir={run_dir}: {reason}")
                elif bool(getattr(stage_args, "skip_plots", False)):
                    _log(args, f"[resume] reusing completed stage run_name={run_name} run_dir={run_dir} skip_plots=True")
                    return run_dir
                else:
                    _log(args, f"[resume] refreshing completed stage outputs run_name={run_name} run_dir={run_dir}")
                    _rerender_plots(stage_args, run_dir)
                    return run_dir
            if _stage_run_checkpointed(run_dir):
                compatible, reason = _stage_run_cosmology_compatible(stage_args, run_dir)
                if not compatible:
                    _log(args, f"[resume] not finalizing checkpointed stage run_name={run_name} run_dir={run_dir}: {reason}")
                else:
                    _log(args, f"[resume] finalizing checkpointed stage run_name={run_name} run_dir={run_dir}")
                    _rerender_plots(stage_args, run_dir)
                    return run_dir
        return _run_single_stage(stage_args, fit_mode, run_name, **kwargs)

    stage1_run_dir: Path | None = None
    stage2_run_dir: Path | None = None
    stage1_summary: Stage1PriorSummary | None = None
    if start_at_stage3:
        _log(
            args,
            (
                "[stage] start_at_stage3=True; sequential workflow starts at stage3_image_plane; "
                "skipping stage1_large_only and stage2_joint"
            ),
        )
    else:
        stage1_run_name = str(Path(root_run_name) / "stage1_large_only")
        stage1_args = _args_with_fit_controls(
            args,
            stage2_controls,
            fit_method=FIT_METHOD_SVI,
            **stage_cosmology_updates(fit_stage_cosmology=False),
        )
        stage1_args = _force_quick_diagnostics_for_nonfinal_stage(stage1_args, stage1_run_name, exact_diagnostics_stage)
        if resume_fast:
            stage1_run_dir = Path(args.output_dir) / stage1_run_name
            _require_resume_fast_stage1_artifacts(stage1_run_dir)
            _require_resume_fast_cosmology_compatibility(stage1_args, stage1_run_dir, "stage1_large_only")
            _log(args, f"[resume-fast] skipping stage1 run_name={stage1_run_name} run_dir={stage1_run_dir}")
        else:
            stage1_run_dir = maybe_run_stage(
                stage1_args,
                "large-only",
                stage1_run_name,
                sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
            )
        stage1_summary = _load_stage1_summary(stage1_run_dir / "artifacts")
        stage2_run_name = str(Path(root_run_name) / "stage2_joint")
        stage2_args = _args_with_fit_controls(
            args,
            stage2_controls,
            **stage_cosmology_updates(fit_stage_cosmology=False),
        )
        stage2_args = _force_quick_diagnostics_for_nonfinal_stage(stage2_args, stage2_run_name, exact_diagnostics_stage)
        if resume_fast and final_stage_name != "stage2":
            stage2_run_dir = Path(args.output_dir) / stage2_run_name
            _require_resume_fast_plot_artifacts(stage2_run_dir, "stage2_joint")
            _require_resume_fast_cosmology_compatibility(stage2_args, stage2_run_dir, "stage2_joint")
            _log(args, f"[resume-fast] skipping stage2 run_name={stage2_run_name} run_dir={stage2_run_dir}")
        else:
            stage2_run_dir = maybe_run_stage(
                stage2_args,
                "joint",
                stage2_run_name,
                stage1_prior_summary=stage1_summary,
                sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
                previous_stage_best_values=stage1_summary.map_values,
            )
    stage_fit_controls_payload = {
        "stage2": stage2_controls.to_json(),
        "stage3": stage3_controls.to_json(),
        "stage4": stage4_controls.to_json(),
    }

    summary_payload = {
        "fit_mode": "sequential",
        "fit_method": stage2_controls.fit_method,
        "image_plane_mode": str(args.image_plane_mode),
        "potfile_mass_size_reparam": bool(getattr(args, "potfile_mass_size_reparam", False)),
        "resume_fast": bool(resume_fast),
        "start_at_stage3": bool(start_at_stage3),
        "stage_fit_controls": stage_fit_controls_payload,
        "skip_stage3_image_plane_local_jacobian": bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)),
        "image_plane_newton_steps": int(getattr(args, "image_plane_newton_steps", 0)),
        "anchored_image_plane_solve_steps": int(
            getattr(args, "anchored_image_plane_solve_steps", DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS)
        ),
        "anchored_image_plane_trust_radius_arcsec": float(
            getattr(
                args,
                "anchored_image_plane_trust_radius_arcsec",
                DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
            )
        ),
        "anchored_image_plane_lm_damping_relative": float(
            getattr(
                args,
                "anchored_image_plane_lm_damping_relative",
                DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
            )
        ),
        "anchored_image_plane_lm_damping_absolute": float(
            getattr(
                args,
                "anchored_image_plane_lm_damping_absolute",
                DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
            )
        ),
        "critical_arc_critical_direction_sigma_arcsec": float(
            getattr(args, "critical_arc_critical_direction_sigma_arcsec", DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC)
        ),
        "critical_arc_base_prob": float(getattr(args, "critical_arc_base_prob", DEFAULT_CRITICAL_ARC_BASE_PROB)),
        "critical_arc_max_prob": float(getattr(args, "critical_arc_max_prob", DEFAULT_CRITICAL_ARC_MAX_PROB)),
        "critical_arc_singular_threshold": float(
            getattr(args, "critical_arc_singular_threshold", DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD)
        ),
        "critical_arc_singular_softness": float(
            getattr(args, "critical_arc_singular_softness", DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS)
        ),
        "critical_arc_lm_damping_relative": float(
            getattr(args, "critical_arc_lm_damping_relative", DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE)
        ),
        "critical_arc_lm_damping_absolute": float(
            getattr(args, "critical_arc_lm_damping_absolute", DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE)
        ),
        "critical_arc_lm_trust_radius_arcsec": float(
            getattr(args, "critical_arc_lm_trust_radius_arcsec", DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC)
        ),
        "arc_aware_noncritical_support_radius_arcsec": float(
            getattr(
                args,
                "arc_aware_noncritical_support_radius_arcsec",
                DEFAULT_ARC_AWARE_NONCRITICAL_SUPPORT_RADIUS_ARCSEC,
            )
        ),
        "arc_aware_max_arclength_arcsec": float(
            getattr(args, "arc_aware_max_arclength_arcsec", DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC)
        ),
        "arc_aware_curve_step_arcsec": float(
            getattr(args, "arc_aware_curve_step_arcsec", DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC)
        ),
        "fold_curvature_arcsec_inv": float(
            getattr(args, "fold_curvature_arcsec_inv", DEFAULT_FOLD_CURVATURE_ARCSEC_INV)
        ),
        "cab_likelihood_weight": getattr(args, "cab_likelihood_weight", None),
        "cab_finite_difference_step_arcsec": float(
            getattr(args, "cab_finite_difference_step_arcsec", DEFAULT_CAB_FINITE_DIFFERENCE_STEP_ARCSEC)
        ),
        "cab_tangent_sigma_floor_rad": float(
            getattr(args, "cab_tangent_sigma_floor_rad", DEFAULT_CAB_TANGENT_SIGMA_FLOOR_RAD)
        ),
        "cab_curvature_sigma_floor_arcsec_inv": float(
            getattr(args, "cab_curvature_sigma_floor_arcsec_inv", DEFAULT_CAB_CURVATURE_SIGMA_FLOOR_ARCSEC_INV)
        ),
        "image_plane_scatter_floor_arcsec": float(
            getattr(args, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)
        ),
        "image_plane_scatter_prior": str(getattr(args, "image_plane_scatter_prior", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR)),
        "image_plane_scatter_prior_median_arcsec": float(
            getattr(args, "image_plane_scatter_prior_median_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC)
        ),
        "image_plane_scatter_prior_log_sigma": float(
            getattr(args, "image_plane_scatter_prior_log_sigma", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA)
        ),
        "fix_image_sigma_int_arcsec": (
            None
            if getattr(args, "fix_image_sigma_int_arcsec", None) is None
            else float(getattr(args, "fix_image_sigma_int_arcsec"))
        ),
        "image_sigma_int_sampled": getattr(args, "fix_image_sigma_int_arcsec", None) is None
        and _sample_likelihood_uses_image_scatter(str(stage4_sample_likelihood_mode or SAMPLE_LIKELIHOOD_SOURCE)),
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
        "likelihood_stabilizer_max_gain": float(
            getattr(args, "likelihood_stabilizer_max_gain", DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN)
        ),
        "likelihood_stabilizer_max_residual_arcsec": float(
            getattr(args, "likelihood_stabilizer_max_residual_arcsec", DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC)
        ),
        "likelihood_stabilizer_residual_loss": str(
            getattr(args, "likelihood_stabilizer_residual_loss", DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS)
        ),
        "likelihood_stabilizer_student_t_nu": float(
            getattr(args, "likelihood_stabilizer_student_t_nu", DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU)
        ),
        "source_position_parameterization": str(
            getattr(args, "source_position_parameterization", SOURCE_POSITION_PARAMETERIZATION_PRIOR_WHITENED)
        ),
        "fit_cosmology_flat_wcdm": fit_cosmology_requested,
        "stage_cosmology_fit": {
            "stage1": False,
            "stage2": False,
            "stage3": bool(stage3_enabled and fit_cosmology_requested),
            "stage4": bool(stage4_enabled and fit_cosmology_requested),
        },
        "cosmology_redshift_binning": "fiducial_fixed",
    }
    if sequential_cosmology_config_override is not None:
        summary_payload["sequential_fiducial_cosmology_config"] = dict(sequential_cosmology_config_override)
    if stage1_run_dir is not None:
        summary_payload["stage1_run_dir"] = str(stage1_run_dir)
    if stage2_run_dir is not None:
        summary_payload["stage2_run_dir"] = str(stage2_run_dir)
    stage3_run_dir: Path | None = None
    if stage3_enabled:
        stage3_run_name = str(Path(root_run_name) / "stage3_image_plane")
        stage3_args = _args_with_fit_controls(
            args,
            stage3_controls,
            **stage_cosmology_updates(fit_stage_cosmology=fit_cosmology_requested),
        )
        stage3_args = _force_quick_diagnostics_for_nonfinal_stage(stage3_args, stage3_run_name, exact_diagnostics_stage)
        if resume_fast and final_stage_name != "stage3":
            stage3_run_dir = Path(args.output_dir) / stage3_run_name
            _require_resume_fast_plot_artifacts(stage3_run_dir, "stage3_image_plane")
            _require_resume_fast_cosmology_compatibility(stage3_args, stage3_run_dir, "stage3_image_plane")
            _log(args, f"[resume-fast] skipping stage3 run_name={stage3_run_name} run_dir={stage3_run_dir}")
        else:
            stage3_init_values = None if start_at_stage3 else _physical_best_fit_values_from_artifacts(stage2_run_dir / "artifacts")
            stage3_run_dir = maybe_run_stage(
                stage3_args,
                "joint",
                stage3_run_name,
                stage1_prior_summary=None if start_at_stage3 else stage1_summary,
                sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
                svi_init_physical_values=stage3_init_values,
                previous_stage_best_values=stage3_init_values,
            )
        summary_payload["stage3_run_dir"] = str(stage3_run_dir)
        if not bool(getattr(args, "skip_critical_det_diagnostic", True)):
            _run_logged_phase(
                args,
                "diagnostics.stage3_critical_det",
                lambda: _run_stage3_critical_det_diagnostic(args, stage3_run_dir),
                detail=f"stage={stage3_run_dir}",
            )
    if stage4_enabled:
        stage4_init_stage_dir = stage3_run_dir or stage2_run_dir
        if stage4_init_stage_dir is None:
            raise RuntimeError("Stage 4 initialization requires a completed stage 3 or stage 2 run.")
        stage4_init_artifacts_dir = stage4_init_stage_dir / "artifacts"
        if resume_fast:
            _require_resume_fast_plot_artifacts(stage4_init_artifacts_dir.parent, stage4_init_artifacts_dir.parent.name)
        stage4_run_name = str(Path(root_run_name) / _stage4_run_directory_name(args))
        stage4_init_values = _physical_best_fit_values_from_artifacts(stage4_init_artifacts_dir)
        source_position_priors = _source_position_prior_values_from_artifacts(stage4_init_artifacts_dir)
        stage4_updates = stage_cosmology_updates(fit_stage_cosmology=fit_cosmology_requested)
        stage4_args = _args_with_fit_controls(
            args,
            stage4_controls,
            **stage4_updates,
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


def _main_dispatch(args: argparse.Namespace, stage_fit_controls: dict[str, StageFitControls]) -> None:
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
            root_dir / "stage4_blocked_linearized_image_plane",
            root_dir / "stage4_forward_metric_image_plane",
            root_dir / "stage4_anchored_solved_image_plane",
            root_dir / "stage4_critical_arc_mixture_image_plane",
            root_dir / "stage4_fold_regularized_image_plane",
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
        default_device = _resolve_jax_device_for_args(args, "jax_default_device", flag_name="--jax-default-device")
        smc_device = _resolve_jax_device_for_args(args, "smc_device", flag_name="--smc-device")
        _log_jax_device_policy(args, default_device, smc_device)
        with _jax_device_context(default_device):
            _main_dispatch(args, stage_fit_controls)
    except BaseException as exc:
        _log_exception("main", exc)
        raise
    finally:
        _close_debug_log()


if __name__ == "__main__":
    main()
