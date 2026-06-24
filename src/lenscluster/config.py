from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from numbers import Integral
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunPathsConfig:
    output_dir: str | Path = "plots"
    run_name: str | None = None
    corner_overlay_bayes_dat: str | Path | None = None


@dataclass(frozen=True)
class PriorConfig:
    kind: str = "fixed"
    lower: float | None = None
    upper: float | None = None
    mean: float | None = None
    std: float | None = None
    step: float | None = None


@dataclass(frozen=True)
class ReferenceFrameConfig:
    reference: int = 3
    ra0_deg: float = 0.0
    dec0_deg: float = 0.0


@dataclass(frozen=True)
class CosmologyConfig:
    H0: float = 70.0
    Om0: float = 0.3
    Ode0: float | None = None
    w0: float = -1.0
    class_name: str = "FlatLambdaCDM"


@dataclass(frozen=True)
class DPIEHaloConfig:
    id: str
    x_centre: float
    y_centre: float
    ellipticite: float
    angle_pos: float
    core_radius_kpc: float
    cut_radius_kpc: float
    v_disp: float
    z_lens: float
    profile_type: int = 81
    priors: dict[str, PriorConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class IndependentMemberHaloConfig:
    population_id: str
    catalog_id: str
    id: str | None = None
    priors: dict[str, PriorConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class MemberPopulationConfig:
    id: str
    catalog_path: str | Path
    mag0: float
    corekpc: float
    sigma: float
    cutkpc: float
    z_lens: float
    profile_type: int = 81
    sigma_prior: PriorConfig = field(default_factory=PriorConfig)
    cutkpc_prior: PriorConfig = field(default_factory=PriorConfig)
    exclude_catalog_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageConstraintsConfig:
    catalog_path: str | Path
    sigma_arcsec: float = 0.5


@dataclass(frozen=True)
class ArcConstraintsConfig:
    catalog_path: str | Path


@dataclass(frozen=True)
class LensModelConfig:
    reference: ReferenceFrameConfig = field(default_factory=ReferenceFrameConfig)
    cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
    large_halos: tuple[DPIEHaloConfig, ...] = ()
    member_populations: tuple[MemberPopulationConfig, ...] = ()
    independent_member_halos: tuple[IndependentMemberHaloConfig, ...] = ()
    image_constraints: ImageConstraintsConfig | None = None
    arc_constraints: tuple[ArcConstraintsConfig, ...] = ()


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int | None = None
    chains: int = 1
    resume: str | bool = False
    plots_only: bool = False
    skip_validation: bool = False
    skip_plots: bool = False
    plot_numpyro_model: bool = False
    display_plots_in_notebook: bool = False
    quick_diagnostics: bool = False
    quiet: bool = False
    debug_sampler_diagnostics: bool = False
    numpyro_print_summary: bool = False
    jax_default_device: str = "auto"
    smc_device: str = "auto"
    nuts_chain_method: str = "parallel"
    dense_mass: str = "structured"
    jax_clear_caches_after_svi_refresh: bool = True


@dataclass(frozen=True)
class WorkflowConfig:
    fit_mode: str = "sequential"
    sampling_engine: str = "refreshing_surrogate"
    stage1_sampling_engine: str = "refreshing_surrogate_flat"
    stage0_likelihood: str = "source"
    stage1_likelihood: str = "local-jacobian"
    stage2_forward_mode: str = "none"
    stage2_sampling_engine: str = "refreshing_surrogate_flat"
    stage2_fresh_process: bool = True
    exact_image_diagnostics_stage2: bool = False
    exact_image_diagnostics_stage3: bool = False
    best_value: str = "map"
    image_plane_mode: str = "none"
    image_plane_newton_steps: int = 0
    linearized_beta_prior_sigma_arcsec: float = 2.0
    source_position_parameterization: str = "prior-whitened"


@dataclass(frozen=True)
class StageScheduleConfig:
    fit_method: tuple[str, ...] = ("svi+nuts",)
    svi_steps: tuple[int, ...] = (2000, 2000)
    refresh_every: tuple[int | None, ...] = (250, 250)
    warmup: tuple[int, ...] = (300,)
    samples: tuple[int, ...] = (500,)
    sampling_refresh_runs: tuple[int, ...] = (1,)
    max_tree_depth: tuple[int, ...] = (10,)
    target_accept: float = 0.85
    z_bin_efficiency_tol: float = 0.01
    initial_step_size: float = 1.0e-3
    svi_learning_rate: float = 5.0e-3


@dataclass(frozen=True)
class MemberSelectionConfig:
    fov_limit_radius: float | None = None
    fov_limit_x: tuple[float, float] | None = None
    fov_limit_y: tuple[float, float] | None = None
    potfile_member_brightest_n: tuple[int, ...] | None = None
    potfile_member_mag_max: tuple[float, ...] | None = None


@dataclass(frozen=True)
class PerturbationDiscoveryConfig:
    perturbation_discovery_alpha_tol_arcsec: float = 0.01
    perturbation_discovery_jacobian_tol: float = 0.01
    perturbation_discovery_jacobian_weight: float = 1.0
    perturbation_discovery_top_k: int | None = None


@dataclass(frozen=True)
class ScalingModelConfig:
    independent_scaling_free_log_sigma_tau_prior_median: float = 0.10
    independent_scaling_free_log_mass_tau_prior_median: float = 0.20
    independent_scaling_free_log_tau_prior_sigma: float = 0.25
    scaling_relation_mode: str = "direct-exponents"
    potfile_alpha_sigma_prior_mean: float = 0.25
    potfile_alpha_sigma_prior_std: float = 0.04
    potfile_alpha_sigma_prior_lower: float = 0.10
    potfile_alpha_sigma_prior_upper: float = 0.40
    potfile_gamma_ml_prior_mean: float = 0.00
    potfile_gamma_ml_prior_std: float = 0.12
    potfile_gamma_ml_prior_lower: float = -0.40
    potfile_gamma_ml_prior_upper: float = 0.40
    scaling_scatter: bool = False
    softening_length_kpc: float = 0.0
    softening_length_prior_log_sigma: float = 0.15


@dataclass(frozen=True)
class LikelihoodConfig:
    pos_sigma_arcsec: float | None = None
    source_plane_covariance_floor: float = 1.0e-6
    source_plane_covariance_mode: str = "magnification"
    image_presence_penalty_weight: float | None = None
    image_presence_match_radius_arcsec: float = 0.30
    image_presence_temperature_arcsec: float = 0.10
    image_presence_count_softness: float = 0.05
    image_presence_count_margin: float = 0.05
    image_plane_scatter_prior: str = "log-uniform"
    image_plane_scatter_floor_arcsec: float = 1.0e-3
    image_plane_scatter_upper_arcsec: float = 2.0
    fix_image_sigma_int_arcsec: float | None = None


@dataclass(frozen=True)
class ImageDiagnosticsConfig:
    fit_quality_draws: int = 0
    exact_image_min_distance_arcsec: float = 0.1
    exact_image_precision_limit: float = 1.0e-8
    exact_image_num_iter_max: int = 200
    exact_image_finder: str = "lenstronomy"
    exact_image_displacement_tol_arcsec: float = 1.0e-4
    exact_image_identification_tol_arcsec: float = 1.0e-3
    exact_image_lm_max_iter: int = 30
    exact_image_lm_trust_radius_arcsec: float = 1.0
    exact_image_adaptive_max_levels: int = 8
    match_tolerance_arcsec: float = 2.0


@dataclass(frozen=True)
class TruthRecoveryConfig:
    kappa_true_fits: str | Path | None = None
    gammax_true_fits: str | Path | None = None
    gammay_true_fits: str | Path | None = None
    truth_grid_mode: str = "median"
    truth_grid_draws: int | None = 64
    truth_grid_size: int = 256
    caustic_source_redshift: float = 9.0
    caustic_plot_grid_scale_arcsec: float | None = None
    skip_grid_diagnostics: bool = False


@dataclass(frozen=True)
class RGBDisplayConfig:
    q: float | None = None
    stretch: float | None = None
    minimum: float | None = None
    red_gain: float | None = None
    green_gain: float | None = None
    blue_gain: float | None = None


@dataclass(frozen=True)
class ImageCatalogCutoutConfig:
    image_dir: str | Path | None = None
    image_scale: str = "60mas"
    bands: tuple[str, ...] | None = None
    rgb: RGBDisplayConfig = field(default_factory=RGBDisplayConfig)
    mode: str = "full"
    dpi: int | None = None
    max_side_pixels: int | None = None
    critical_lines: str = "auto"
    cutouts: bool = True


@dataclass(frozen=True)
class LensClusterSolverConfig:
    model: LensModelConfig | None = None
    paths: RunPathsConfig = field(default_factory=RunPathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    schedule: StageScheduleConfig = field(default_factory=StageScheduleConfig)
    members: MemberSelectionConfig = field(default_factory=MemberSelectionConfig)
    perturbation: PerturbationDiscoveryConfig = field(default_factory=PerturbationDiscoveryConfig)
    scaling: ScalingModelConfig = field(default_factory=ScalingModelConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    image_diagnostics: ImageDiagnosticsConfig = field(default_factory=ImageDiagnosticsConfig)
    truth: TruthRecoveryConfig = field(default_factory=TruthRecoveryConfig)
    image_catalog: ImageCatalogCutoutConfig = field(default_factory=ImageCatalogCutoutConfig)

    def with_updates(self, **updates: Any) -> "LensClusterSolverConfig":
        return replace(self, **updates)

    def validate(self) -> "LensClusterSolverConfig":
        validate_config(self)
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def to_run_dict(self) -> dict[str, Any]:
        return self.to_json_dict()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def validate_config(config: LensClusterSolverConfig) -> None:
    _validate_model_config(config.model)
    workflow = config.workflow
    if workflow.stage0_likelihood not in {"source", "local-jacobian", "critical-arc"}:
        raise ValueError("stage0_likelihood must be 'source', 'local-jacobian', or 'critical-arc'.")
    if workflow.stage1_likelihood not in {"source", "local-jacobian", "critical-arc"}:
        raise ValueError("stage1_likelihood must be 'source', 'local-jacobian', or 'critical-arc'.")
    if workflow.stage2_forward_mode not in {"none", "linearized", "critical-arc"}:
        raise ValueError("stage2_forward_mode must be 'none', 'linearized', or 'critical-arc'.")
    schedule = config.schedule
    expected_stages = _expected_stage_count(workflow)
    if len(schedule.svi_steps) != expected_stages:
        raise ValueError(f"svi_steps requires exactly {expected_stages} values for this workflow.")
    if len(schedule.refresh_every) != expected_stages:
        raise ValueError(f"refresh_every requires exactly {expected_stages} values for this workflow.")
    _validate_positive_int_sequence("svi_steps", schedule.svi_steps)
    _validate_nonnegative_refresh_sequence(schedule.refresh_every)
    _validate_positive_int_sequence("warmup", schedule.warmup)
    _validate_positive_int_sequence("samples", schedule.samples)
    _validate_positive_int_sequence("sampling_refresh_runs", schedule.sampling_refresh_runs)
    _validate_positive_int_sequence("max_tree_depth", schedule.max_tree_depth)
    if not schedule.fit_method:
        raise ValueError("fit_method must contain at least one stage method.")
    if schedule.target_accept <= 0.0 or schedule.target_accept >= 1.0:
        raise ValueError("target_accept must lie between 0 and 1.")
    if schedule.z_bin_efficiency_tol < 0.0:
        raise ValueError("z_bin_efficiency_tol must be nonnegative.")
    if schedule.initial_step_size <= 0.0:
        raise ValueError("initial_step_size must be positive.")
    if schedule.svi_learning_rate <= 0.0:
        raise ValueError("svi_learning_rate must be positive.")
    if config.perturbation.perturbation_discovery_top_k is not None:
        top_k = config.perturbation.perturbation_discovery_top_k
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or int(top_k) <= 0:
            raise ValueError("perturbation_discovery_top_k must be a positive integer or None.")
    if config.scaling.softening_length_kpc < 0.0:
        raise ValueError("softening_length_kpc must be nonnegative.")
    if config.scaling.softening_length_prior_log_sigma <= 0.0:
        raise ValueError("softening_length_prior_log_sigma must be positive.")
    _validate_truncated_normal(
        "potfile_alpha_sigma_prior",
        mean=config.scaling.potfile_alpha_sigma_prior_mean,
        std=config.scaling.potfile_alpha_sigma_prior_std,
        lower=config.scaling.potfile_alpha_sigma_prior_lower,
        upper=config.scaling.potfile_alpha_sigma_prior_upper,
    )
    _validate_truncated_normal(
        "potfile_gamma_ml_prior",
        mean=config.scaling.potfile_gamma_ml_prior_mean,
        std=config.scaling.potfile_gamma_ml_prior_std,
        lower=config.scaling.potfile_gamma_ml_prior_lower,
        upper=config.scaling.potfile_gamma_ml_prior_upper,
    )
    if config.scaling.independent_scaling_free_log_sigma_tau_prior_median <= 0.0:
        raise ValueError("independent_scaling_free_log_sigma_tau_prior_median must be positive.")
    if config.scaling.independent_scaling_free_log_mass_tau_prior_median <= 0.0:
        raise ValueError("independent_scaling_free_log_mass_tau_prior_median must be positive.")
    if config.scaling.independent_scaling_free_log_tau_prior_sigma <= 0.0:
        raise ValueError("independent_scaling_free_log_tau_prior_sigma must be positive.")
    if config.truth.truth_grid_mode not in {"median", "posterior"}:
        raise ValueError("truth_grid_mode must be 'median' or 'posterior'.")
    if config.truth.truth_grid_draws is not None and int(config.truth.truth_grid_draws) <= 0:
        raise ValueError("truth_grid_draws must be a positive integer or None.")
    if config.truth.truth_grid_size < 0:
        raise ValueError("truth_grid_size must be nonnegative.")


def _validate_model_config(model: LensModelConfig | None) -> None:
    if model is None:
        raise ValueError("LensClusterSolverConfig.model is required; Lenstool par files are no longer supported.")
    if not model.large_halos:
        raise ValueError("model.large_halos must contain at least one halo.")
    if not model.member_populations:
        raise ValueError("model.member_populations must contain at least one member population.")
    if model.image_constraints is None and not model.arc_constraints:
        raise ValueError("model must declare image_constraints or arc_constraints.")
    if model.cosmology.H0 <= 0.0:
        raise ValueError("model.cosmology.H0 must be positive.")
    if model.cosmology.Om0 < 0.0:
        raise ValueError("model.cosmology.Om0 must be nonnegative.")
    ode0 = 1.0 - model.cosmology.Om0 if model.cosmology.Ode0 is None else model.cosmology.Ode0
    if ode0 < 0.0:
        raise ValueError("model.cosmology.Ode0 must be nonnegative.")
    for halo in model.large_halos:
        if halo.core_radius_kpc <= 0.0 or halo.cut_radius_kpc <= 0.0 or halo.v_disp <= 0.0:
            raise ValueError(f"large halo {halo.id!r} must have positive core, cut, and velocity dispersion.")
        if halo.z_lens <= 0.0:
            raise ValueError(f"large halo {halo.id!r} must have positive z_lens.")
    populations_by_id: dict[str, MemberPopulationConfig] = {}
    catalog_ids_by_population: dict[str, set[str]] = {}
    for population in model.member_populations:
        if population.id in populations_by_id:
            raise ValueError(f"duplicate member population id {population.id!r}.")
        populations_by_id[population.id] = population
        if not Path(population.catalog_path).is_file():
            raise ValueError(f"member population catalog does not exist: {population.catalog_path}")
        if population.corekpc <= 0.0 or population.sigma <= 0.0 or population.cutkpc <= 0.0:
            raise ValueError(f"member population {population.id!r} must have positive corekpc, sigma, and cutkpc.")
        if population.z_lens <= 0.0:
            raise ValueError(f"member population {population.id!r} must have positive z_lens.")
        exclude_ids = tuple(str(value).strip() for value in population.exclude_catalog_ids)
        if any(not value for value in exclude_ids):
            raise ValueError(f"member population {population.id!r} has an empty exclude_catalog_ids entry.")
        if len(set(exclude_ids)) != len(exclude_ids):
            raise ValueError(f"member population {population.id!r} has duplicate exclude_catalog_ids entries.")
        catalog_ids_by_population[population.id] = _catalog_ids_from_member_catalog(population.catalog_path)
    independent_ids: set[str] = set()
    independent_member_keys: set[tuple[str, str]] = set()
    for independent in model.independent_member_halos:
        population_id = str(independent.population_id).strip()
        catalog_id = str(independent.catalog_id).strip()
        if not population_id:
            raise ValueError("independent_member_halos entries require a nonempty population_id.")
        if not catalog_id:
            raise ValueError("independent_member_halos entries require a nonempty catalog_id.")
        if population_id not in populations_by_id:
            raise ValueError(f"independent member halo references unknown population_id {population_id!r}.")
        resolved_id = str(independent.id or f"independent_member_{population_id}_{catalog_id}")
        if resolved_id in independent_ids:
            raise ValueError(f"duplicate independent member halo id {resolved_id!r}.")
        independent_ids.add(resolved_id)
        key = (population_id, catalog_id)
        if key in independent_member_keys:
            raise ValueError(f"duplicate independent member halo for population_id={population_id!r} catalog_id={catalog_id!r}.")
        independent_member_keys.add(key)
        population = populations_by_id[population_id]
        if catalog_id in {str(value).strip() for value in population.exclude_catalog_ids}:
            raise ValueError(
                f"independent member halo population_id={population_id!r} catalog_id={catalog_id!r} "
                "also appears in exclude_catalog_ids."
            )
        if catalog_id not in catalog_ids_by_population[population_id]:
            raise ValueError(
                f"independent member halo population_id={population_id!r} catalog_id={catalog_id!r} "
                f"was not found in {population.catalog_path}."
            )
    if model.image_constraints is not None:
        if not Path(model.image_constraints.catalog_path).is_file():
            raise ValueError(f"image constraint catalog does not exist: {model.image_constraints.catalog_path}")
        if model.image_constraints.sigma_arcsec <= 0.0:
            raise ValueError("image_constraints.sigma_arcsec must be positive.")
    for arcs in model.arc_constraints:
        if not Path(arcs.catalog_path).is_file():
            raise ValueError(f"arc constraint catalog does not exist: {arcs.catalog_path}")


def _catalog_ids_from_member_catalog(path: str | Path) -> set[str]:
    catalog_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                token = line.split(",", 1)[0]
            else:
                token = line.split(maxsplit=1)[0]
            token = token.strip().strip("\"'")
            if token and token.lower() != "id":
                catalog_ids.add(token)
    return catalog_ids


def _expected_stage_count(workflow: WorkflowConfig) -> int:
    if workflow.fit_mode == "sequential":
        return 3 if workflow.stage2_forward_mode != "none" else 2
    return 1


def _validate_positive_int_sequence(name: str, values: tuple[int, ...]) -> None:
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    for value in values:
        if int(value) <= 0:
            raise ValueError(f"{name} values must be positive integers.")


def _validate_nonnegative_refresh_sequence(values: tuple[int | None, ...]) -> None:
    if not values:
        raise ValueError("refresh_every must contain at least one value.")
    for value in values:
        if value is not None and int(value) < 0:
            raise ValueError("refresh_every values must be positive, zero, or None.")


def _validate_truncated_normal(name: str, *, mean: float, std: float, lower: float, upper: float) -> None:
    if std <= 0.0:
        raise ValueError(f"{name}_std must be positive.")
    if lower >= upper:
        raise ValueError(f"{name}_lower must be less than {name}_upper.")
    if not lower <= mean <= upper:
        raise ValueError(f"{name}_mean must lie inside the configured bounds.")
