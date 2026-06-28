$XONSH_SHOW_TRACEBACK = True

import os
import sys
import time
from pathlib import Path

cores = 4  # Fixed to match JAX CPU devices and NUTS chains.
seed_default = 12345

$JAX_NUM_CPU_DEVICES = str(cores)
$MPLCONFIGDIR = "/tmp/matplotlib-lenscluster-mock"
os.environ["JAX_NUM_CPU_DEVICES"] = str(cores)
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-lenscluster-mock"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lenscluster.config import (
    LensClusterSolverConfig,
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


def _usage() -> str:
    return "Usage: xonsh run_mock.xsh [smoke|production] [seed]"


def _parse_args(argv: list[str]) -> tuple[str, int]:
    if len(argv) > 3:
        print(_usage())
        raise SystemExit(2)
    preset = argv[1].strip().lower() if len(argv) >= 2 else "smoke"
    if preset not in VALID_PRESETS:
        print(f"Unknown preset: {argv[1]!r}\n{_usage()}")
        raise SystemExit(2)
    if len(argv) >= 3:
        try:
            seed = int(argv[2])
        except ValueError:
            print(f"Seed must be an integer; got {argv[2]!r}.\n{_usage()}")
            raise SystemExit(2)
        if seed < 0:
            print(f"Seed must be nonnegative; got {seed}.\n{_usage()}")
            raise SystemExit(2)
        return preset, seed
    return preset, seed_default


def _solver_template(*, production: bool, seed: int) -> LensClusterSolverConfig:
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
        perturbation=PerturbationDiscoveryConfig(perturbation_discovery_top_k=None),
    )


def build_config(preset: str, *, seed: int) -> MockValidationConfig:
    production = preset == "production"
    mock = SingleBCGMockConfig(
        seed=seed,
        n_primary_families=20 if production else 5,
        n_subhalo_families=0,
        n_subhalos=10 if production else 0,
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
            output_dir="validation_runs/mock_recovery",
            run_name=f"single_bcg_{preset}_source",
        ),
        runtime=MockValidationRuntimeConfig(
            realizations=1,
            seed=seed,
            resume="all",
            quiet=False,
        ),
        solver=MockValidationSolverConfig(
            template=_solver_template(production=production, seed=seed),
            run_name="fit",
        ),
        recovery=recovery,
    )


preset, seed = _parse_args(sys.argv)
config = build_config(preset, seed=seed).validate()

print(
    f"[config] preset={preset} seed={seed} "
    f"output_dir={config.paths.output_dir} run_name={config.paths.run_name} chains={cores}"
)

start = time.monotonic()
try:
    run_single_bcg_validation(config)
finally:
    print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
