$XONSH_SHOW_TRACEBACK = True

import os
import sys
import time
from pathlib import Path

cores = 4  # Fixed to match JAX CPU devices and NUTS chains.
seed_default = 12345
output_dir_default = "validation_runs/mock_recovery"

$JAX_NUM_CPU_DEVICES = str(cores)
$MPLCONFIGDIR = "/tmp/matplotlib-lenscluster-mock"
os.environ["JAX_NUM_CPU_DEVICES"] = str(cores)
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-lenscluster-mock"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lenscluster.config import (
    LensClusterSolverConfig,
    LikelihoodConfig,
    PerturbationDiscoveryConfig,
    RuntimeConfig,
    StageScheduleConfig,
    WorkflowConfig,
)
from lenscluster.mock_validation import (
    MockValidationConfig,
    MockValidationPathsConfig,
    MockValidationRecoveryConfig,
    MockValidationRuntimeConfig,
    MockValidationSolverConfig,
    SingleBCGMockConfig,
    run_single_bcg_validation,
)

VALID_PRESETS = ("smoke", "production")
VALID_COVARIANCE_MODES = ("anisotropic", "isotropic")


def _usage() -> str:
    return (
        "Usage: xonsh run_mock.xsh --campaign NAME "
        "[--preset smoke|production] [--seed INT] "
        "[--covariance anisotropic|isotropic] [--output-dir PATH]"
    )


def _parse_args(argv: list[str]) -> tuple[str, int, str, str, str]:
    values = {
        "campaign": None,
        "preset": "smoke",
        "seed": str(seed_default),
        "covariance": "anisotropic",
        "output_dir": output_dir_default,
    }
    flag_to_key = {
        "--campaign": "campaign",
        "--preset": "preset",
        "--seed": "seed",
        "--covariance": "covariance",
        "--output-dir": "output_dir",
    }
    index = 1
    while index < len(argv):
        flag = argv[index]
        if flag in {"--help", "-h"}:
            print(_usage())
            raise SystemExit(0)
        if flag not in flag_to_key:
            print(f"Unknown argument: {flag!r}\n{_usage()}")
            raise SystemExit(2)
        if index + 1 >= len(argv):
            print(f"Missing value for {flag}.\n{_usage()}")
            raise SystemExit(2)
        values[flag_to_key[flag]] = argv[index + 1]
        index += 2
    if values["campaign"] is None:
        print(f"Missing required --campaign.\n{_usage()}")
        raise SystemExit(2)
    preset = str(values["preset"]).strip().lower()
    if preset not in VALID_PRESETS:
        print(f"Preset must be 'smoke' or 'production'; got {values['preset']!r}.\n{_usage()}")
        raise SystemExit(2)
    try:
        seed = int(str(values["seed"]))
    except ValueError:
        print(f"Seed must be an integer; got {values['seed']!r}.\n{_usage()}")
        raise SystemExit(2)
    if seed < 0:
        print(f"Seed must be nonnegative; got {seed}.\n{_usage()}")
        raise SystemExit(2)
    covariance_mode = str(values["covariance"]).strip().lower()
    if covariance_mode not in VALID_COVARIANCE_MODES:
        print(f"Covariance mode must be 'anisotropic' or 'isotropic'; got {values['covariance']!r}.\n{_usage()}")
        raise SystemExit(2)
    campaign = str(values["campaign"]).strip()
    output_dir = str(values["output_dir"])
    return preset, seed, covariance_mode, campaign, output_dir


def _solver_template(*, production: bool, seed: int, critical_arc_anisotropic_covariance: bool) -> LensClusterSolverConfig:
    if production:
        warmup = 3000, 3000
        samples = 500, 500
        max_tree_depth = 8, 8
        return LensClusterSolverConfig(
            runtime=RuntimeConfig(
                seed=seed,
                chains=cores,
                resume="all",
                quick_diagnostics=False,
                debug_sampler_diagnostics=True,
                numpyro_print_summary=True,
                nuts_chain_method="parallel",
                dense_mass="structured",
                jax_clear_caches_after_svi_refresh=False,
            ),
            workflow=WorkflowConfig(
                fit_mode="sequential",
                stage0_likelihood="source",
                stage1_likelihood="source",
                stage2_forward_mode="critical-arc-anisotropic",
                stage1_sampling_engine="full_flat",
                stage2_sampling_engine="full_flat",
                stage2_fresh_process=True,
                exact_image_diagnostics_stage2=True,
                best_value="maximum-likelihood",
                image_plane_newton_steps=0,
                linearized_beta_prior_sigma_arcsec=3.0,
                source_position_parameterization="prior-whitened",
            ),
            schedule=StageScheduleConfig(
                fit_method=("svi+nuts", "svi+nuts"),
                refresh_every=(None, None, None),
                svi_steps=(10_000, 20_000, 20_000),
                warmup=warmup,
                samples=samples,
                sampling_refresh_runs=(1, 1),
                max_tree_depth=max_tree_depth,
                target_accept=0.8,
                z_bin_efficiency_tol=0.0,
                svi_learning_rate=0.0005,
            ),
            likelihood=LikelihoodConfig(
                critical_arc_anisotropic_covariance=critical_arc_anisotropic_covariance,
            ),
            perturbation=PerturbationDiscoveryConfig(perturbation_discovery_top_k=None),
        )

    return LensClusterSolverConfig(
        runtime=RuntimeConfig(
            seed=seed,
            chains=cores,
            resume="all",
            skip_plots=True,
            quick_diagnostics=False,
            debug_sampler_diagnostics=True,
            numpyro_print_summary=False,
            nuts_chain_method="parallel",
            dense_mass="structured",
        ),
        workflow=WorkflowConfig(
            fit_mode="sequential",
            stage0_likelihood="local-jacobian",
            stage1_likelihood="local-jacobian",
            stage2_forward_mode="critical-arc-anisotropic",
            stage1_sampling_engine="full_flat",
            stage2_sampling_engine="full_flat",
            stage2_fresh_process=True,
            exact_image_diagnostics_stage2=True,
            best_value="maximum-likelihood",
            image_plane_newton_steps=0,
            linearized_beta_prior_sigma_arcsec=3.0,
            source_position_parameterization="prior-whitened",
        ),
        schedule=StageScheduleConfig(
            fit_method=("svi+nuts", "svi+nuts"),
            refresh_every=(None, None, None),
            svi_steps=(250, 500, 500),
            warmup=(100, 100),
            samples=(100, 100),
            sampling_refresh_runs=(1, 1),
            max_tree_depth=(6, 6),
            target_accept=0.8,
            z_bin_efficiency_tol=0.0,
            svi_learning_rate=0.001,
        ),
        likelihood=LikelihoodConfig(
            critical_arc_anisotropic_covariance=critical_arc_anisotropic_covariance,
        ),
        perturbation=PerturbationDiscoveryConfig(perturbation_discovery_top_k=None),
    )


def build_config(
    preset: str,
    *,
    seed: int,
    covariance_mode: str,
    campaign_name: str,
    output_dir: str,
) -> MockValidationConfig:
    production = preset == "production"
    critical_arc_anisotropic_covariance = covariance_mode == "anisotropic"
    mock = SingleBCGMockConfig(
        seed=seed,
        n_primary_families=20 if production else 5,
        n_subhalo_families=0,
        n_subhalos=50 if production else 0,
        min_images_per_family=3,
        max_images_per_family=None,
        primary_source_redshifts=(1.5, 2.0, 3.0),
        subhalo_source_redshifts=(1.5, 2.0, 3.0),
        pos_sigma_arcsec=0.05,
    )
    recovery = MockValidationRecoveryConfig(
        posterior_diagnostic_draws=8 if production else 2,
        posterior_diagnostic_mode="exact",
        recovery_profile_draws=128 if production else 8,
    )
    return MockValidationConfig(
        mock=mock,
        paths=MockValidationPathsConfig(
            output_dir=output_dir,
            run_name=f"single_bcg_{preset}_source",
            campaign_name=campaign_name,
            variant_name=covariance_mode,
        ),
        runtime=MockValidationRuntimeConfig(
            realizations=1,
            seed=seed,
            resume="all",
            quiet=False,
        ),
        solver=MockValidationSolverConfig(
            template=_solver_template(
                production=production,
                seed=seed,
                critical_arc_anisotropic_covariance=critical_arc_anisotropic_covariance,
            ),
            run_name="fit",
        ),
        recovery=recovery,
    )


preset, seed, covariance_mode, campaign_name, output_dir = _parse_args(sys.argv)
config = build_config(
    preset,
    seed=seed,
    covariance_mode=covariance_mode,
    campaign_name=campaign_name,
    output_dir=output_dir,
).validate()

print(
    f"[config] preset={preset} seed={seed} "
    f"covariance={covariance_mode} "
    f"critical_arc_anisotropic_covariance={config.solver.template.likelihood.critical_arc_anisotropic_covariance} "
    f"output_dir={config.paths.output_dir} campaign={config.paths.campaign_name} "
    f"run_name={config.paths.run_name} variant={config.paths.variant_name} chains={cores}"
)

start = time.monotonic()
try:
    run_single_bcg_validation(config)
finally:
    print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
