from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax.numpy as jnp
import numpy as np

DEFAULT_SEARCH_PADDING = 8.0
DEFAULT_SAMPLER = "numpyro_nuts"
COUPLED_TRANSFORM_NONE = "none"
COUPLED_TRANSFORM_POTFILE_MASS_SIZE = "potfile_mass_size"
COUPLED_ROLE_MASS_NORM = "mass_norm"
COUPLED_ROLE_SIZE = "size"


class ParameterTransformSpec(Protocol):
    transform_kind: str
    transform_offset: float
    transform_scale: float


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
    transform_scale: float = 1.0
    parent_sample_name: str | None = None
    sample_site_name: str | None = None
    sample_site_index: int | None = None
    coupled_transform_kind: str = COUPLED_TRANSFORM_NONE
    coupled_role: str | None = None
    coupled_group: str | None = None
    coupled_mass_center: float = 0.0
    coupled_mass_scale: float = 1.0
    coupled_size_center: float = 0.0
    coupled_size_scale: float = 1.0


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
    sigma_log_scatter_param_index: np.ndarray
    core_log_scatter_param_index: np.ndarray
    cut_log_scatter_param_index: np.ndarray


@dataclass
class SurrogateBinCache:
    effective_z_source: float
    inactive_alpha_x: np.ndarray
    inactive_alpha_y: np.ndarray
    # Derivative rows follow the evaluator's surrogate-parameter basis: scaling parameters,
    # plus sampled cosmology parameters when flat-wCDM cosmology is fitted.
    inactive_alpha_dx_dparams: np.ndarray
    inactive_alpha_dy_dparams: np.ndarray
    inactive_jacobian_delta_a00: np.ndarray | None = None
    inactive_jacobian_delta_a01: np.ndarray | None = None
    inactive_jacobian_delta_a10: np.ndarray | None = None
    inactive_jacobian_delta_a11: np.ndarray | None = None
    inactive_jacobian_delta_da00_dparams: np.ndarray | None = None
    inactive_jacobian_delta_da01_dparams: np.ndarray | None = None
    inactive_jacobian_delta_da10_dparams: np.ndarray | None = None
    inactive_jacobian_delta_da11_dparams: np.ndarray | None = None
    fold_regularized_kappa_eff: np.ndarray | None = None
    fold_regularized_near_indices: np.ndarray | None = None
    fold_regularized_far_indices: np.ndarray | None = None


@dataclass
class FamilyData:
    family_id: str
    z_source: float
    effective_z_source: float
    sigma_arcsec: float
    image_labels: list[str]
    x_obs: np.ndarray
    y_obs: np.ndarray
    reliability: np.ndarray

    @property
    def n_images(self) -> int:
        return len(self.image_labels)

    @property
    def x_center(self) -> float:
        if self.n_images == 0:
            return float("nan")
        return float(0.5 * (np.min(self.x_obs) + np.max(self.x_obs)))

    @property
    def y_center(self) -> float:
        if self.n_images == 0:
            return float("nan")
        return float(0.5 * (np.min(self.y_obs) + np.max(self.y_obs)))

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
    reliability_per_image: np.ndarray


@dataclass
class ArcConstraintData:
    arc_ids: list[str]
    z_arc: np.ndarray
    anchor_x: np.ndarray
    anchor_y: np.ndarray
    tangent_angle_rad: np.ndarray
    curvature_arcsec_inv: np.ndarray
    sigma_tangent_angle_rad: np.ndarray
    sigma_curvature_arcsec_inv: np.ndarray
    reliability: np.ndarray

    @property
    def n_arcs(self) -> int:
        return len(self.arc_ids)


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
    flat_wcdm_quadrature_order: int = 64
    lens_quadrature_z: list[float] | None = None
    lens_quadrature_weights: list[float] | None = None
    effective_z_quadrature_z: list[list[float]] | None = None
    effective_z_quadrature_weights: list[list[float]] | None = None
    exact_z_quadrature_z: list[list[float]] | None = None
    exact_z_quadrature_weights: list[list[float]] | None = None


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


@dataclass
class PosteriorResults:
    samples: np.ndarray
    log_prob: np.ndarray
    accept_prob: np.ndarray
    diverging: np.ndarray
    num_steps: np.ndarray
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
    ns_diagnostics: dict[str, np.ndarray] | None = None


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
    arc_data: ArcConstraintData | None
    lens_model_list: list[str]
    reference: tuple[int, float, float]
    fit_mode: str
    potfiles: list[dict[str, Any]]
    scaling_component_records: list[dict[str, Any]]
    geometry_cache: GeometryCache | None = None
    svi_init_values: dict[str, float] | None = None
    previous_stage_best_values: dict[str, float] | None = None
    fit_cosmology_flat_wcdm: bool = False
    source_position_parameterization: str = "direct"


def positive_lognormal_parameters(mean: float, std: float, *, floor: float = 1.0e-6) -> tuple[float, float]:
    """Convert physical positive-normal moments into latent log-normal moments."""
    mean = max(float(mean), float(floor))
    std = max(float(std), 1.0e-9)
    variance = std * std
    sigma2 = np.log1p(variance / (mean * mean))
    return float(np.log(mean) - 0.5 * sigma2), float(np.sqrt(sigma2))


def physical_to_latent(value: float, spec: ParameterTransformSpec) -> float:
    value = float(value)
    offset = float(spec.transform_offset)
    scale = float(getattr(spec, "transform_scale", 1.0))
    if spec.transform_kind == "log_positive":
        return float(np.log(max(value, 1.0e-12)))
    if spec.transform_kind == "log_offset_positive":
        return float(np.log(max(value - offset, 1.0e-12)))
    if spec.transform_kind == "affine":
        return float((value - offset) / scale)
    return value


def latent_to_physical(value: float, spec: ParameterTransformSpec) -> float:
    value = float(value)
    if spec.transform_kind == "log_positive":
        return float(np.exp(value))
    if spec.transform_kind == "log_offset_positive":
        return float(spec.transform_offset + np.exp(value))
    if spec.transform_kind == "affine":
        return float(spec.transform_offset + float(getattr(spec, "transform_scale", 1.0)) * value)
    return value


def latent_array_to_physical(values: np.ndarray, spec: ParameterTransformSpec) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if spec.transform_kind == "log_positive":
        return np.exp(array)
    if spec.transform_kind == "log_offset_positive":
        return float(spec.transform_offset) + np.exp(array)
    if spec.transform_kind == "affine":
        return float(spec.transform_offset) + float(getattr(spec, "transform_scale", 1.0)) * array
    return array


def latent_jax_to_physical(values: jnp.ndarray, spec: ParameterTransformSpec) -> jnp.ndarray:
    array = jnp.asarray(values, dtype=jnp.float64)
    if spec.transform_kind == "log_positive":
        return jnp.exp(array)
    if spec.transform_kind == "log_offset_positive":
        return jnp.asarray(spec.transform_offset, dtype=jnp.float64) + jnp.exp(array)
    if spec.transform_kind == "affine":
        return jnp.asarray(spec.transform_offset, dtype=jnp.float64) + jnp.asarray(
            getattr(spec, "transform_scale", 1.0),
            dtype=jnp.float64,
        ) * array
    return array


def _is_potfile_mass_size_spec(spec: ParameterSpec) -> bool:
    return str(getattr(spec, "coupled_transform_kind", COUPLED_TRANSFORM_NONE)) == COUPLED_TRANSFORM_POTFILE_MASS_SIZE


def _potfile_mass_size_groups(parameter_specs: list[ParameterSpec]) -> list[tuple[int, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for idx, spec in enumerate(parameter_specs):
        if not _is_potfile_mass_size_spec(spec):
            continue
        group_name = str(spec.coupled_group or spec.sample_site_name or spec.potential_id)
        role = str(spec.coupled_role or "")
        grouped.setdefault(group_name, {})[role] = idx
    result: list[tuple[int, int]] = []
    for role_map in grouped.values():
        if COUPLED_ROLE_MASS_NORM in role_map and COUPLED_ROLE_SIZE in role_map:
            result.append((role_map[COUPLED_ROLE_MASS_NORM], role_map[COUPLED_ROLE_SIZE]))
    return result


def _apply_potfile_mass_size_physical_transform(
    values: np.ndarray,
    parameter_specs: list[ParameterSpec],
) -> np.ndarray:
    converted = np.asarray(values, dtype=float).copy()
    source = np.asarray(values, dtype=float)
    for mass_idx, size_idx in _potfile_mass_size_groups(parameter_specs):
        mass_spec = parameter_specs[mass_idx]
        size_spec = parameter_specs[size_idx]
        u_m = source[..., mass_idx]
        u_s = source[..., size_idx]
        mass_raw = float(mass_spec.coupled_mass_center) + float(mass_spec.coupled_mass_scale) * u_m
        size_raw = float(mass_spec.coupled_size_center) + float(mass_spec.coupled_size_scale) * u_s
        log_sigma = 0.5 * (mass_raw - size_raw)
        converted[..., mass_idx] = np.exp(log_sigma)
        converted[..., size_idx] = float(size_spec.transform_offset) + np.exp(size_raw)
    return converted


def _apply_potfile_mass_size_latent_transform(
    values: np.ndarray,
    parameter_specs: list[ParameterSpec],
) -> np.ndarray:
    converted = np.asarray(values, dtype=float).copy()
    source = np.asarray(values, dtype=float)
    for mass_idx, size_idx in _potfile_mass_size_groups(parameter_specs):
        mass_spec = parameter_specs[mass_idx]
        size_spec = parameter_specs[size_idx]
        sigma = np.maximum(source[..., mass_idx], 1.0e-300)
        cut_gap = np.maximum(source[..., size_idx] - float(size_spec.transform_offset), 1.0e-300)
        log_sigma = np.log(sigma)
        log_cut_gap = np.log(cut_gap)
        mass_raw = 2.0 * log_sigma + log_cut_gap
        converted[..., mass_idx] = (
            mass_raw - float(mass_spec.coupled_mass_center)
        ) / float(mass_spec.coupled_mass_scale)
        converted[..., size_idx] = (
            log_cut_gap - float(mass_spec.coupled_size_center)
        ) / float(mass_spec.coupled_size_scale)
    return converted


def potfile_mass_size_coupling_arrays(parameter_specs: list[ParameterSpec]) -> dict[str, np.ndarray]:
    groups = _potfile_mass_size_groups(parameter_specs)
    mass_indices: list[int] = []
    size_indices: list[int] = []
    mass_centers: list[float] = []
    mass_scales: list[float] = []
    size_centers: list[float] = []
    size_scales: list[float] = []
    cut_offsets: list[float] = []
    for mass_idx, size_idx in groups:
        mass_spec = parameter_specs[mass_idx]
        size_spec = parameter_specs[size_idx]
        mass_indices.append(int(mass_idx))
        size_indices.append(int(size_idx))
        mass_centers.append(float(mass_spec.coupled_mass_center))
        mass_scales.append(float(mass_spec.coupled_mass_scale))
        size_centers.append(float(mass_spec.coupled_size_center))
        size_scales.append(float(mass_spec.coupled_size_scale))
        cut_offsets.append(float(size_spec.transform_offset))
    return {
        "mass_indices": np.asarray(mass_indices, dtype=np.int32),
        "size_indices": np.asarray(size_indices, dtype=np.int32),
        "mass_centers": np.asarray(mass_centers, dtype=float),
        "mass_scales": np.asarray(mass_scales, dtype=float),
        "size_centers": np.asarray(size_centers, dtype=float),
        "size_scales": np.asarray(size_scales, dtype=float),
        "cut_offsets": np.asarray(cut_offsets, dtype=float),
    }


def apply_potfile_mass_size_couplings_jax(
    physical_params: jnp.ndarray,
    latent_params: jnp.ndarray,
    *,
    mass_indices: jnp.ndarray | None = None,
    size_indices: jnp.ndarray | None = None,
    mass_centers: jnp.ndarray | None = None,
    mass_scales: jnp.ndarray | None = None,
    size_centers: jnp.ndarray | None = None,
    size_scales: jnp.ndarray | None = None,
    cut_offsets: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if mass_indices is None or size_indices is None:
        return physical_params
    mass_indices = jnp.asarray(mass_indices, dtype=jnp.int32)
    size_indices = jnp.asarray(size_indices, dtype=jnp.int32)
    if mass_indices.size == 0:
        return physical_params
    latent = jnp.asarray(latent_params, dtype=jnp.float64)
    u_m = jnp.take(latent, mass_indices)
    u_s = jnp.take(latent, size_indices)
    mass_raw = jnp.asarray(mass_centers, dtype=jnp.float64) + jnp.asarray(mass_scales, dtype=jnp.float64) * u_m
    size_raw = jnp.asarray(size_centers, dtype=jnp.float64) + jnp.asarray(size_scales, dtype=jnp.float64) * u_s
    log_sigma = 0.5 * (mass_raw - size_raw)
    sigma = jnp.exp(log_sigma)
    cut = jnp.asarray(cut_offsets, dtype=jnp.float64) + jnp.exp(size_raw)
    result = jnp.asarray(physical_params, dtype=jnp.float64)
    result = result.at[mass_indices].set(sigma)
    result = result.at[size_indices].set(cut)
    return result


def display_lower(spec: ParameterSpec) -> float:
    return float(spec.physical_lower if spec.physical_lower is not None else spec.lower)


def display_upper(spec: ParameterSpec) -> float:
    return float(spec.physical_upper if spec.physical_upper is not None else spec.upper)


def convert_theta_to_physical(theta: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    theta_array = np.asarray(theta, dtype=float)
    if theta_array.size == 0:
        return theta_array.copy()
    converted = theta_array.copy()
    for idx, spec in enumerate(parameter_specs):
        if _is_potfile_mass_size_spec(spec):
            continue
        converted[idx] = latent_to_physical(theta_array[idx], spec)
    return _apply_potfile_mass_size_physical_transform(converted, parameter_specs)


def convert_theta_to_latent(theta: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    theta_array = np.asarray(theta, dtype=float)
    if theta_array.size == 0:
        return theta_array.copy()
    converted = theta_array.copy()
    for idx, spec in enumerate(parameter_specs):
        if _is_potfile_mass_size_spec(spec):
            continue
        converted[idx] = physical_to_latent(theta_array[idx], spec)
    return _apply_potfile_mass_size_latent_transform(converted, parameter_specs)


def convert_sample_matrix_to_latent(samples: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.size == 0:
        return array.copy()
    converted = array.copy()
    for idx, spec in enumerate(parameter_specs):
        if _is_potfile_mass_size_spec(spec):
            continue
        converted[..., idx] = np.asarray(
            [physical_to_latent(value, spec) for value in array[..., idx].reshape(-1)],
            dtype=float,
        ).reshape(array[..., idx].shape)
    return _apply_potfile_mass_size_latent_transform(converted, parameter_specs)


def apply_parameter_transforms_jax(
    params: jnp.ndarray,
    log_positive_mask: jnp.ndarray,
    log_offset_positive_mask: jnp.ndarray,
    transform_offset_array: jnp.ndarray,
    affine_mask: jnp.ndarray | None = None,
    transform_scale_array: jnp.ndarray | None = None,
    potfile_mass_size_mass_indices: jnp.ndarray | None = None,
    potfile_mass_size_size_indices: jnp.ndarray | None = None,
    potfile_mass_size_mass_centers: jnp.ndarray | None = None,
    potfile_mass_size_mass_scales: jnp.ndarray | None = None,
    potfile_mass_size_size_centers: jnp.ndarray | None = None,
    potfile_mass_size_size_scales: jnp.ndarray | None = None,
    potfile_mass_size_cut_offsets: jnp.ndarray | None = None,
) -> jnp.ndarray:
    params_array = jnp.asarray(params, dtype=jnp.float64)
    if affine_mask is None:
        affine_mask = jnp.zeros_like(log_positive_mask, dtype=bool)
    if transform_scale_array is None:
        transform_scale_array = jnp.ones_like(transform_offset_array, dtype=jnp.float64)
    if potfile_mass_size_mass_indices is not None and potfile_mass_size_size_indices is not None:
        coupled_indices = jnp.concatenate(
            [
                jnp.asarray(potfile_mass_size_mass_indices, dtype=jnp.int32),
                jnp.asarray(potfile_mass_size_size_indices, dtype=jnp.int32),
            ]
        )
        coupled_mask = jnp.zeros_like(log_positive_mask, dtype=bool).at[coupled_indices].set(True)
        active_mask = jnp.logical_not(coupled_mask)
        log_positive_mask = jnp.logical_and(log_positive_mask, active_mask)
        log_offset_positive_mask = jnp.logical_and(log_offset_positive_mask, active_mask)
        affine_mask = jnp.logical_and(affine_mask, active_mask)
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
    physical_params = jnp.where(
        affine_mask,
        transform_offset_array + transform_scale_array * params_array,
        physical_params,
    )
    return apply_potfile_mass_size_couplings_jax(
        physical_params,
        params_array,
        mass_indices=potfile_mass_size_mass_indices,
        size_indices=potfile_mass_size_size_indices,
        mass_centers=potfile_mass_size_mass_centers,
        mass_scales=potfile_mass_size_mass_scales,
        size_centers=potfile_mass_size_size_centers,
        size_scales=potfile_mass_size_size_scales,
        cut_offsets=potfile_mass_size_cut_offsets,
    )


def convert_sample_matrix_to_physical(samples: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.size == 0:
        return array.copy()
    converted = array.copy()
    for idx, spec in enumerate(parameter_specs):
        if _is_potfile_mass_size_spec(spec):
            continue
        converted[..., idx] = latent_array_to_physical(array[..., idx], spec)
    return _apply_potfile_mass_size_physical_transform(converted, parameter_specs)
