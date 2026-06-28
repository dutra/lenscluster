from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ..config import (
    CosmologyConfig,
    DPIEHaloConfig,
    ImageConstraintsConfig,
    LensClusterSolverConfig,
    LensModelConfig,
    MemberPopulationConfig,
    PriorConfig,
    ReferenceFrameConfig,
    RunPathsConfig,
    RuntimeConfig,
)
from ..jax_cosmology import flat_wcdm_config, kpc_per_arcsec_from_config
from .generation import MockClusterPaths, SingleBCGMockConfig


@dataclass(frozen=True)
class MockValidationPathsConfig:
    output_dir: str | Path = "validation_runs"
    run_name: str = "single_bcg_recovery"
    campaign_name: str | None = None
    variant_name: str | None = None


@dataclass(frozen=True)
class MockValidationRuntimeConfig:
    realizations: int = 1
    seed: int = 12345
    resume: str | bool = False
    quiet: bool = False


@dataclass(frozen=True)
class MockValidationSolverConfig:
    template: LensClusterSolverConfig = field(default_factory=LensClusterSolverConfig)
    run_name: str = "fit"


@dataclass(frozen=True)
class MockValidationRecoveryConfig:
    posterior_diagnostic_draws: int = 8
    posterior_diagnostic_mode: str = "exact"
    critical_caustic_plot_grid_scale_arcsec: float = 0.2
    recovery_profile_draws: int = 128


@dataclass(frozen=True)
class MockValidationConfig:
    mock: SingleBCGMockConfig = field(default_factory=SingleBCGMockConfig)
    paths: MockValidationPathsConfig = field(default_factory=MockValidationPathsConfig)
    runtime: MockValidationRuntimeConfig = field(default_factory=MockValidationRuntimeConfig)
    solver: MockValidationSolverConfig = field(default_factory=MockValidationSolverConfig)
    recovery: MockValidationRecoveryConfig = field(default_factory=MockValidationRecoveryConfig)

    @property
    def quiet(self) -> bool:
        return bool(self.runtime.quiet)

    def with_updates(self, **updates: Any) -> "MockValidationConfig":
        return replace(self, **updates)

    def validate(self) -> "MockValidationConfig":
        validate_mock_validation_config(self)
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def validate_mock_validation_config(config: MockValidationConfig) -> None:
    config.mock.validate()
    runtime = config.runtime
    if isinstance(runtime.realizations, bool) or int(runtime.realizations) <= 0:
        raise ValueError("runtime.realizations must be a positive integer.")
    if isinstance(runtime.seed, bool) or int(runtime.seed) < 0:
        raise ValueError("runtime.seed must be a nonnegative integer.")
    if runtime.resume not in (False, True, "all", "fast"):
        raise ValueError("runtime.resume must be False, True, 'all', or 'fast'.")
    if not str(config.paths.run_name).strip():
        raise ValueError("paths.run_name must be nonempty.")
    _validate_path_segment(config.paths.campaign_name, "paths.campaign_name")
    _validate_path_segment(config.paths.variant_name, "paths.variant_name")
    if not str(config.solver.run_name).strip():
        raise ValueError("solver.run_name must be nonempty.")
    recovery = config.recovery
    if int(recovery.posterior_diagnostic_draws) <= 0:
        raise ValueError("recovery.posterior_diagnostic_draws must be positive.")
    if int(recovery.recovery_profile_draws) <= 0:
        raise ValueError("recovery.recovery_profile_draws must be positive.")
    if recovery.posterior_diagnostic_mode not in {"exact", "approximate"}:
        raise ValueError("recovery.posterior_diagnostic_mode must be 'exact' or 'approximate'.")
    if not math.isfinite(float(recovery.critical_caustic_plot_grid_scale_arcsec)) or (
        float(recovery.critical_caustic_plot_grid_scale_arcsec) <= 0.0
    ):
        raise ValueError("recovery.critical_caustic_plot_grid_scale_arcsec must be positive and finite.")
    _validate_solver_schedule_shape(config.solver.template)


def _validate_path_segment(value: str | None, field_name: str) -> None:
    if value is None:
        return
    text = str(value)
    if not text.strip():
        raise ValueError(f"{field_name} must be nonempty when provided.")
    path = Path(text)
    if text in {".", ".."} or path.name != text or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a single path segment.")


def _validate_solver_schedule_shape(template: LensClusterSolverConfig) -> None:
    workflow = template.workflow
    expected_stages = 3 if workflow.fit_mode == "sequential" and workflow.stage2_forward_mode != "none" else 2
    if workflow.fit_mode != "sequential":
        expected_stages = 1
    expected_production_controls = 2 if workflow.fit_mode == "sequential" and workflow.stage2_forward_mode != "none" else 1
    schedule = template.schedule
    if len(schedule.svi_steps) != expected_stages:
        raise ValueError(f"solver.template.schedule.svi_steps requires exactly {expected_stages} values.")
    if len(schedule.refresh_every) != expected_stages:
        raise ValueError(f"solver.template.schedule.refresh_every requires exactly {expected_stages} values.")
    for field_name, values in (
        ("fit_method", schedule.fit_method),
        ("warmup", schedule.warmup),
        ("samples", schedule.samples),
        ("sampling_refresh_runs", schedule.sampling_refresh_runs),
        ("max_tree_depth", schedule.max_tree_depth),
    ):
        if len(values) != expected_production_controls:
            raise ValueError(
                "solver.template.schedule."
                f"{field_name} requires exactly {expected_production_controls} production control value"
                f"{'' if expected_production_controls == 1 else 's'}."
            )


def single_bcg_mock_lens_model_config(
    mock_config: SingleBCGMockConfig,
    paths: MockClusterPaths,
) -> LensModelConfig:
    kpc_per_arcsec = float(kpc_per_arcsec_from_config(mock_config.z_lens, flat_wcdm_config(h0=70.0, om0=0.3)))
    large_halos = (
        _dpie_halo_config(mock_config.halo, mock_config, kpc_per_arcsec, position_half_width_arcsec=8.0),
        _dpie_halo_config(
            mock_config.bcg,
            mock_config,
            kpc_per_arcsec,
            position_half_width_arcsec=float(mock_config.bcg_position_prior_half_width_arcsec),
        ),
    )
    member_populations: tuple[MemberPopulationConfig, ...] = ()
    member_catalog_path = paths.root / "members.cat"
    if int(mock_config.n_subhalos) > 0 and member_catalog_path.is_file():
        member_populations = (
            MemberPopulationConfig(
                id="potfile",
                catalog_path=member_catalog_path,
                mag0=float(mock_config.subhalo_mag0),
                corekpc=float(mock_config.subhalo_core_radius_arcsec) * kpc_per_arcsec,
                sigma=float(mock_config.subhalo_sigma_ref),
                cutkpc=float(mock_config.subhalo_cut_radius_arcsec) * kpc_per_arcsec,
                z_lens=float(mock_config.z_lens),
                sigma_prior=PriorConfig(
                    "normal",
                    mean=float(mock_config.subhalo_sigma_ref),
                    std=float(mock_config.subhalo_sigma_ref_std),
                ),
                cutkpc_prior=PriorConfig(
                    "uniform",
                    lower=float(mock_config.subhalo_cut_lower_arcsec) * kpc_per_arcsec,
                    upper=float(mock_config.subhalo_cut_upper_arcsec) * kpc_per_arcsec,
                ),
            ),
        )
    return LensModelConfig(
        reference=ReferenceFrameConfig(
            reference=3,
            ra0_deg=float(mock_config.reference_ra_deg),
            dec0_deg=float(mock_config.reference_dec_deg),
        ),
        cosmology=CosmologyConfig(H0=70.0, Om0=0.3, Ode0=0.7),
        large_halos=large_halos,
        member_populations=member_populations,
        image_constraints=ImageConstraintsConfig(
            catalog_path=paths.image_catalog_path,
            sigma_arcsec=float(mock_config.pos_sigma_arcsec),
        ),
    )


def solver_config_for_single_bcg_mock(
    config: MockValidationConfig,
    *,
    paths: MockClusterPaths,
    seed: int,
    output_dir: str | Path,
) -> LensClusterSolverConfig:
    template = config.solver.template
    return replace(
        template,
        model=single_bcg_mock_lens_model_config(config.mock, paths),
        paths=RunPathsConfig(
            output_dir=output_dir,
            run_name=str(config.solver.run_name),
            corner_overlay_bayes_dat=template.paths.corner_overlay_bayes_dat,
        ),
        runtime=replace(
            template.runtime,
            seed=int(seed),
            resume=config.runtime.resume,
            quiet=bool(config.runtime.quiet),
        ),
    )


def _dpie_halo_config(
    component: Any,
    mock_config: SingleBCGMockConfig,
    kpc_per_arcsec: float,
    *,
    position_half_width_arcsec: float,
) -> DPIEHaloConfig:
    core_radius_kpc = float(component.core_radius_arcsec) * kpc_per_arcsec
    cut_radius_kpc = float(component.cut_radius_arcsec) * kpc_per_arcsec
    return DPIEHaloConfig(
        id=str(component.potential_id),
        x_centre=float(component.x_centre),
        y_centre=float(component.y_centre),
        ellipticite=float(component.ellipticite),
        angle_pos=float(component.angle_pos),
        core_radius_kpc=core_radius_kpc,
        cut_radius_kpc=cut_radius_kpc,
        v_disp=float(component.v_disp),
        z_lens=float(mock_config.z_lens),
        priors={
            "x_centre": PriorConfig(
                "uniform",
                lower=float(component.x_centre) - float(position_half_width_arcsec),
                upper=float(component.x_centre) + float(position_half_width_arcsec),
                step=0.05,
            ),
            "y_centre": PriorConfig(
                "uniform",
                lower=float(component.y_centre) - float(position_half_width_arcsec),
                upper=float(component.y_centre) + float(position_half_width_arcsec),
                step=0.05,
            ),
            "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.75, step=0.02),
            "angle_pos": PriorConfig("uniform", lower=-90.0, upper=90.0, step=0.5),
            "core_radius_kpc": PriorConfig(
                "uniform",
                lower=max(0.001, float(component.core_radius_arcsec) * 0.2) * kpc_per_arcsec,
                upper=float(component.core_radius_arcsec) * 3.0 * kpc_per_arcsec,
                step=0.02 * kpc_per_arcsec,
            ),
            "cut_radius_kpc": PriorConfig(
                "uniform",
                lower=float(component.cut_radius_arcsec) * 0.4 * kpc_per_arcsec,
                upper=float(component.cut_radius_arcsec) * 2.0 * kpc_per_arcsec,
                step=0.5 * kpc_per_arcsec,
            ),
            "v_disp": PriorConfig(
                "uniform",
                lower=float(component.v_disp) * 0.55,
                upper=float(component.v_disp) * 1.45,
                step=1.0,
            ),
        },
    )
