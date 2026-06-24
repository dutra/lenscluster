from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import nbformat
import pytest

from lenscluster import cluster_solver
from lenscluster.config import (
    CosmologyConfig,
    DPIEHaloConfig,
    ImageCatalogCutoutConfig,
    ImageConstraintsConfig,
    ImageDiagnosticsConfig,
    IndependentMemberHaloConfig,
    LensClusterSolverConfig,
    LensModelConfig,
    PerturbationDiscoveryConfig,
    RGBDisplayConfig,
    MemberPopulationConfig,
    PriorConfig,
    ReferenceFrameConfig,
    RunPathsConfig,
    ScalingModelConfig,
    StageScheduleConfig,
    TruthRecoveryConfig,
    WorkflowConfig,
)
from lenscluster.planning import RunPlan, compile_run_plan
from lenscluster.runner import LensClusterRunner
from lenscluster.stages import StageExecutionResult


def _notebook_source(path: str | Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    return "\n\n".join(str(cell.get("source", "")) for cell in notebook.cells)


def _minimal_sequential_config() -> LensClusterSolverConfig:
    return LensClusterSolverConfig(
        model=_minimal_model_config(),
        paths=RunPathsConfig(output_dir="results/demo", run_name="hera_demo"),
        workflow=WorkflowConfig(fit_mode="sequential", stage0_likelihood="local-jacobian"),
        schedule=StageScheduleConfig(
            fit_method=("svi+nuts",),
            svi_steps=(10, 20),
            refresh_every=(None, 100),
            warmup=(1,),
            samples=(2,),
            sampling_refresh_runs=(1,),
            max_tree_depth=(8,),
        ),
    )


def _minimal_model_config() -> LensModelConfig:
    return LensModelConfig(
        reference=ReferenceFrameConfig(reference=3, ra0_deg=0.0, dec0_deg=0.0),
        cosmology=CosmologyConfig(H0=72.0, Om0=0.24, Ode0=0.76),
        large_halos=(
            DPIEHaloConfig(
                id="1",
                x_centre=0.0,
                y_centre=0.0,
                ellipticite=0.2,
                angle_pos=0.0,
                core_radius_kpc=5.0,
                cut_radius_kpc=1500.0,
                v_disp=700.0,
                z_lens=0.507,
                priors={
                    "x_centre": PriorConfig("uniform", lower=-5.0, upper=5.0),
                    "y_centre": PriorConfig("uniform", lower=-5.0, upper=5.0),
                    "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.8),
                    "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0),
                    "core_radius_kpc": PriorConfig("uniform", lower=2.0, upper=15.0),
                    "v_disp": PriorConfig("truncated_normal", mean=700.0, std=245.0, lower=100.0, upper=2200.0),
                },
            ),
        ),
        member_populations=(
            MemberPopulationConfig(
                id="potfile_1",
                catalog_path="data/ff_sims/hera/hera_cluster_members_potfile.cat",
                mag0=19.82,
                corekpc=0.15,
                sigma=96.7,
                cutkpc=33.0,
                z_lens=0.507,
            ),
        ),
        image_constraints=ImageConstraintsConfig(catalog_path="data/ff_sims/hera/hera_obs_arcs.cat", sigma_arcsec=0.5),
    )


def _with_independent_members(
    model: LensModelConfig,
    independent_members: tuple[IndependentMemberHaloConfig, ...],
) -> LensModelConfig:
    return replace(model, independent_member_halos=independent_members)


def test_config_module_is_data_only() -> None:
    import lenscluster.config as config_module

    text = Path("src/lenscluster/config.py").read_text(encoding="utf-8")
    assert "cluster_solver" not in text
    assert "argparse" not in text
    assert not hasattr(config_module.LensClusterSolverConfig, "to_namespace")
    assert not hasattr(config_module, "config_from_namespace")
    assert not hasattr(config_module, "run_from_config")


def test_cluster_solver_exposes_no_cli_parser() -> None:
    assert not hasattr(cluster_solver, "_parse_args")


def test_config_defaults_validate_without_solver_namespace() -> None:
    config = LensClusterSolverConfig(model=_minimal_model_config())

    assert config.workflow.fit_mode == "sequential"
    assert config.workflow.stage1_likelihood == "local-jacobian"
    assert config.workflow.stage2_forward_mode == "none"
    assert config.workflow.best_value == "map"
    assert config.schedule.svi_steps == (2000, 2000)
    assert config.schedule.refresh_every == (250, 250)
    assert config.truth.truth_grid_mode == "median"
    assert config.truth.truth_grid_draws == 64
    assert config.truth.truth_grid_size == 256
    config.validate()


def test_independent_member_halo_config_validation() -> None:
    base_model = _minimal_model_config()
    valid_model = _with_independent_members(
        base_model,
        (IndependentMemberHaloConfig(population_id="potfile_1", catalog_id="1"),),
    )
    LensClusterSolverConfig(model=valid_model).validate()

    missing_population = _with_independent_members(
        base_model,
        (IndependentMemberHaloConfig(population_id="missing", catalog_id="1"),),
    )
    with pytest.raises(ValueError, match="unknown population_id"):
        LensClusterSolverConfig(model=missing_population).validate()

    missing_catalog_id = _with_independent_members(
        base_model,
        (IndependentMemberHaloConfig(population_id="potfile_1", catalog_id="not-a-member"),),
    )
    with pytest.raises(ValueError, match="was not found"):
        LensClusterSolverConfig(model=missing_catalog_id).validate()

    duplicate_member = _with_independent_members(
        base_model,
        (
            IndependentMemberHaloConfig(population_id="potfile_1", catalog_id="1"),
            IndependentMemberHaloConfig(population_id="potfile_1", catalog_id="1"),
        ),
    )
    with pytest.raises(ValueError, match="duplicate independent member halo"):
        LensClusterSolverConfig(model=duplicate_member).validate()

    excluded_population = replace(
        base_model.member_populations[0],
        exclude_catalog_ids=("1",),
    )
    excluded_overlap = replace(
        base_model,
        member_populations=(excluded_population,),
        independent_member_halos=(IndependentMemberHaloConfig(population_id="potfile_1", catalog_id="1"),),
    )
    with pytest.raises(ValueError, match="also appears in exclude_catalog_ids"):
        LensClusterSolverConfig(model=excluded_overlap).validate()


def test_independent_member_halo_is_free_dpie_not_scaling_member() -> None:
    model = _with_independent_members(
        _minimal_model_config(),
        (IndependentMemberHaloConfig(population_id="potfile_1", catalog_id="1"),),
    )
    plan = compile_run_plan(LensClusterSolverConfig(model=model))

    state = cluster_solver._build_state_from_inputs(plan.runtime_args, fit_mode_override="joint")
    evaluator = cluster_solver.ClusterJAXEvaluator(state, match_tolerance_arcsec=cluster_solver.DEFAULT_MATCH_TOLERANCE)

    assert "1" not in {str(record["catalog_id"]) for record in state.scaling_component_records}
    independent_component = next(
        component
        for component in state.base_components
        if component.get("independent_member_catalog_id") == "1"
    )
    assert independent_component["id"] == "independent_member_potfile_1_1"
    independent_index = state.base_components.index(independent_component)
    assert independent_index in set(evaluator.large_component_indices.tolist())
    assert independent_index not in set(evaluator.scaling_component_indices.tolist())

    specs_by_field = {
        spec.field: spec
        for spec in state.parameter_specs
        if spec.potential_id == "independent_member_potfile_1_1"
    }
    for field in ("x_centre", "y_centre", "e1", "e2", "core_radius_kpc", "cut_radius_kpc", "v_disp"):
        assert specs_by_field[field].component_family == "large"


def test_compile_run_plan_resolves_runtime_stages_and_outputs() -> None:
    plan = compile_run_plan(_minimal_sequential_config())

    assert isinstance(plan, RunPlan)
    assert plan.output.run_name == "hera_demo"
    assert plan.output.output_dir == Path("results/demo")
    assert [stage.name for stage in plan.stages] == ["stage0_fast_initializer", "stage1_backprojected_centroid_fit"]
    assert plan.stages[0].sampling_engine == "full_flat"
    assert plan.stages[0].sample_likelihood_mode == "local-jacobian"
    assert plan.stages[0].output_plan.stage0_minimal_outputs is True
    assert plan.stages[1].svi_steps == 20
    assert plan.stages[1].refresh_every == 100
    assert plan.runtime.chains == 1
    assert plan.diagnostics.image_recovery_enabled is True
    assert plan.run_metadata["paths"]["run_name"] == "hera_demo"


def test_compile_run_plan_resolves_unified_critical_arc_stage_policies() -> None:
    config = _minimal_sequential_config().with_updates(
        workflow=WorkflowConfig(
            fit_mode="sequential",
            stage0_likelihood="critical-arc",
            stage1_likelihood="critical-arc",
            stage2_forward_mode="critical-arc",
        ),
        schedule=StageScheduleConfig(
            fit_method=("svi+nuts", "svi+nuts"),
            svi_steps=(10, 20, 30),
            refresh_every=(None, 100, 100),
            warmup=(1, 1),
            samples=(2, 2),
            sampling_refresh_runs=(1,),
            max_tree_depth=(8, 8),
        ),
    )

    plan = compile_run_plan(config)

    assert [stage.sample_likelihood_mode for stage in plan.stages] == [
        "critical-arc-mixture-image-plane",
        "critical-arc-mixture-image-plane",
        "critical-arc-mixture-image-plane",
    ]
    assert [stage.likelihood_family for stage in plan.stages] == [
        "critical-arc",
        "critical-arc",
        "critical-arc",
    ]
    assert [stage.source_position_policy for stage in plan.stages] == [
        "centroid-fixed",
        "centroid-fixed",
        "sampled",
    ]


def test_compile_run_plan_resolves_critical_arc_anisotropic_stage_policies() -> None:
    config = _minimal_sequential_config().with_updates(
        workflow=WorkflowConfig(
            fit_mode="sequential",
            stage0_likelihood="critical-arc-anisotropic",
            stage1_likelihood="critical-arc-anisotropic",
            stage2_forward_mode="critical-arc-anisotropic",
        ),
        schedule=StageScheduleConfig(
            fit_method=("svi+nuts", "svi+nuts"),
            svi_steps=(10, 20, 30),
            refresh_every=(None, 100, 100),
            warmup=(1, 1),
            samples=(2, 2),
            sampling_refresh_runs=(1,),
            max_tree_depth=(8, 8),
        ),
    )

    plan = compile_run_plan(config)

    assert [stage.sample_likelihood_mode for stage in plan.stages] == [
        "critical-arc-anisotropic-image-plane",
        "critical-arc-anisotropic-image-plane",
        "critical-arc-anisotropic-image-plane",
    ]
    assert [stage.likelihood_family for stage in plan.stages] == [
        "critical-arc",
        "critical-arc",
        "critical-arc",
    ]
    assert [stage.source_position_policy for stage in plan.stages] == [
        "centroid-fixed",
        "centroid-fixed",
        "sampled",
    ]


def test_config_validation_rejects_old_stage1_critical_arc_mixture_name() -> None:
    config = _minimal_sequential_config().with_updates(
        workflow=WorkflowConfig(
            fit_mode="sequential",
            stage0_likelihood="local-jacobian",
            stage1_likelihood="critical-arc-mixture",
        ),
    )

    with pytest.raises(ValueError, match="stage1_likelihood"):
        config.validate()


def test_config_validation_requires_stage0_likelihood() -> None:
    for value in ("", None):
        config = _minimal_sequential_config().with_updates(
            workflow=WorkflowConfig(fit_mode="sequential", stage0_likelihood=value),
        )
        with pytest.raises(ValueError, match="stage0_likelihood"):
            config.validate()


def test_workflow_config_defaults_stage0_likelihood_to_source() -> None:
    config = LensClusterSolverConfig(
        model=_minimal_model_config(),
        workflow=WorkflowConfig(),
    )

    config.validate()
    plan = compile_run_plan(config)

    assert config.workflow.stage0_likelihood == "source"
    assert plan.stages[0].sample_likelihood_mode == "source"
    assert plan.stages[0].source_position_policy == "sampled"


def test_config_validation_accepts_stage0_likelihood_values() -> None:
    for value in ("source", "local-jacobian", "critical-arc", "critical-arc-anisotropic"):
        _minimal_sequential_config().with_updates(
            workflow=WorkflowConfig(fit_mode="sequential", stage0_likelihood=value),
        ).validate()


def test_compile_run_plan_allows_stage0_likelihood_different_from_stage1() -> None:
    config = _minimal_sequential_config().with_updates(
        workflow=WorkflowConfig(
            fit_mode="sequential",
            stage0_likelihood="source",
            stage1_likelihood="critical-arc",
        )
    )

    plan = compile_run_plan(config)

    assert [stage.sample_likelihood_mode for stage in plan.stages[:2]] == [
        "source",
        "critical-arc-mixture-image-plane",
    ]
    assert [stage.source_position_policy for stage in plan.stages[:2]] == ["sampled", "centroid-fixed"]
    assert plan.runtime_args.stage0_likelihood == "source"
    assert plan.runtime_args.stage1_likelihood == "critical-arc"


def test_config_validation_accepts_perturbation_discovery_top_k() -> None:
    _minimal_sequential_config().with_updates(
        perturbation=PerturbationDiscoveryConfig(perturbation_discovery_top_k=None),
    ).validate()
    config = _minimal_sequential_config().with_updates(
        perturbation=PerturbationDiscoveryConfig(perturbation_discovery_top_k=5),
    )
    config.validate()
    plan = compile_run_plan(config)
    assert plan.runtime_args.perturbation_discovery_top_k == 5


def test_config_validation_rejects_nonpositive_perturbation_discovery_top_k() -> None:
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="perturbation_discovery_top_k"):
            _minimal_sequential_config().with_updates(
                perturbation=PerturbationDiscoveryConfig(perturbation_discovery_top_k=value),
            ).validate()


def test_critical_arc_source_position_specs_follow_stage_policy() -> None:
    plan = compile_run_plan(
        _minimal_sequential_config().with_updates(
            workflow=WorkflowConfig(
                fit_mode="sequential",
                stage0_likelihood="local-jacobian",
                stage1_likelihood="critical-arc",
            ),
        )
    )
    stage1_args = cluster_solver._clone_args(
        plan.runtime_args,
        sample_likelihood_mode="critical-arc-mixture-image-plane",
        critical_arc_source_position_policy="centroid-fixed",
    )
    stage2_args = cluster_solver._clone_args(
        plan.runtime_args,
        sample_likelihood_mode="critical-arc-mixture-image-plane",
        critical_arc_source_position_policy="sampled",
    )

    stage1_state = cluster_solver._build_state_from_inputs(stage1_args, fit_mode_override="joint")
    source_position_priors = {
        str(family.family_id): (0.0, 0.0)
        for family in stage1_state.family_data
    }
    stage2_state = cluster_solver._build_state_from_inputs(
        stage2_args,
        fit_mode_override="joint",
        source_position_prior_values=source_position_priors,
    )

    assert not any(spec.component_family == "source_position" for spec in stage1_state.parameter_specs)
    assert any(spec.component_family == "source_position" for spec in stage2_state.parameter_specs)


def test_solver_runtime_payload_is_flat_and_clone_preserves_model_config() -> None:
    plan = compile_run_plan(_minimal_sequential_config())

    payload = cluster_solver._args_payload(plan.runtime_args)
    assert "values" not in payload
    assert payload["model_config"] is plan.config.model

    cloned = cluster_solver._clone_args(plan.runtime_args, model_config=None, quick_diagnostics=True)
    assert cloned.model_config is plan.config.model
    assert cloned.quick_diagnostics is True


def test_args_with_fit_controls_preserves_model_config() -> None:
    plan = compile_run_plan(_minimal_sequential_config())
    controls = cluster_solver.StageFitControls(
        fit_method="svi",
        svi_steps=11,
        refresh_every=None,
        warmup=0,
        samples=0,
        sampling_refresh_runs=1,
        max_tree_depth=6,
    )

    staged = cluster_solver._args_with_fit_controls(plan.runtime_args, controls, fit_mode="stage0_fast_initializer")

    assert staged.model_config is plan.config.model
    assert staged.fit_method == "svi"
    assert staged.svi_steps == 11
    assert staged.fit_mode == "stage0_fast_initializer"


def test_old_cli_artifact_bundle_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "plot_bundle.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cli_args_json", data="{}")

    with pytest.raises(ValueError, match="old unsupported artifact bundle"):
        cluster_solver._rebuild_state_from_h5(path)


def test_compile_run_plan_adds_stage2_when_enabled() -> None:
    config = _minimal_sequential_config().with_updates(
        workflow=WorkflowConfig(
            fit_mode="sequential",
            stage0_likelihood="local-jacobian",
            stage2_forward_mode="linearized",
        ),
        schedule=StageScheduleConfig(
            fit_method=("svi+nuts", "svi+nuts"),
            svi_steps=(10, 20, 30),
            refresh_every=(None, 100, 100),
            warmup=(1, 1),
            samples=(2, 2),
            sampling_refresh_runs=(1,),
            max_tree_depth=(8, 8),
        ),
    )

    plan = compile_run_plan(config)

    assert [stage.name for stage in plan.stages] == [
        "stage0_fast_initializer",
        "stage1_backprojected_centroid_fit",
        "stage2_free_source_forward_fit",
    ]
    assert plan.stages[2].svi_steps == 30


def test_config_validation_rejects_wrong_stage_counts_and_bad_priors() -> None:
    bad_counts = LensClusterSolverConfig(
        model=_minimal_model_config(),
        workflow=WorkflowConfig(fit_mode="sequential", stage0_likelihood="local-jacobian"),
        schedule=StageScheduleConfig(svi_steps=(10,), refresh_every=(None,), fit_method=("svi+nuts",)),
    )
    with pytest.raises(ValueError, match="svi_steps requires exactly 2 values"):
        bad_counts.validate()

    bad_prior = LensClusterSolverConfig(
        model=_minimal_model_config(),
        workflow=WorkflowConfig(fit_mode="sequential", stage0_likelihood="local-jacobian"),
        schedule=StageScheduleConfig(svi_steps=(10, 20), refresh_every=(None, 100), fit_method=("svi+nuts",)),
        scaling=ScalingModelConfig(potfile_gamma_ml_prior_lower=0.5, potfile_gamma_ml_prior_upper=-0.5),
    )
    with pytest.raises(ValueError, match="potfile_gamma_ml_prior_lower"):
        bad_prior.validate()


def test_runner_owns_execution_setup_and_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = compile_run_plan(_minimal_sequential_config())
    captured: dict[str, Any] = {}

    monkeypatch.setattr(cluster_solver, "_configure_debug_log", lambda *_args, **_kwargs: captured.setdefault("debug_log", True))
    monkeypatch.setattr(cluster_solver, "_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log_runtime_summary", lambda *_args, **_kwargs: captured.setdefault("runtime_log", True))
    monkeypatch.setattr(cluster_solver, "_log_jax_device_policy", lambda *_args, **_kwargs: captured.setdefault("device_log", True))

    class FakeExecutor:
        def execute(self, passed_plan: RunPlan, stage_fit_controls: dict[str, Any]) -> StageExecutionResult:
            captured["plan"] = passed_plan
            captured["stage_fit_controls"] = stage_fit_controls
            return StageExecutionResult(completed=True)

    result = LensClusterRunner(stage_executor=FakeExecutor()).run(plan)

    assert result.run_name == "hera_demo"
    assert result.run_dir == Path("results/demo/hera_demo")
    assert result.completed is True
    assert captured["plan"] is plan
    assert not hasattr(plan, "_internal_args")
    assert captured["stage_fit_controls"]["stage0"].svi_steps == 10
    assert captured["stage_fit_controls"]["stage1"].svi_steps == 20
    assert captured["debug_log"] is True
    assert captured["runtime_log"] is True
    assert captured["device_log"] is True


def test_ff_sims_script_runner_has_been_removed() -> None:
    assert not Path("scripts/run_ff_sims_fit.py").exists()


def test_run_xsh_is_self_contained_ff_sims_runner() -> None:
    text = Path("run.xsh").read_text(encoding="utf-8")

    assert "scripts/run_ff_sims_fit.py" not in text
    assert "scripts.run_ff_sims_fit" not in text
    assert "from scripts" not in text
    assert "runner_code" not in text
    assert "textwrap" not in text
    assert "-c @(runner_code)" not in text
    assert "LensClusterSolverConfig" in text
    assert "compile_run_plan" in text
    assert "LensClusterRunner" in text
    assert '"ARES"' in text
    assert '"HERA"' in text
    assert "data/ff_sims/ares/ares_obs_arcs.cat" in text
    assert "data/ff_sims/hera/hera_obs_arcs.cat" in text
    assert "data/ff_sims/published/ares/kappa_z9_0.fits" in text
    assert "data/ff_sims/published/hera/kappa_z9_0.fits" in text
    assert "IndependentMemberHaloConfig(population_id=\"potfile_1\", catalog_id=\"2\")" in text
    assert "IndependentMemberHaloConfig(population_id=\"potfile_1\", catalog_id=\"3\")" in text
    assert "IndependentMemberHaloConfig(population_id=\"potfile_1\", catalog_id=\"1\")" in text
    assert "exclude_catalog_ids=(\"2\", \"3\")" not in text
    assert "exclude_catalog_ids=(\"1\", \"2\")" not in text
    assert "2.3 / 0.72" in text
    assert "cores = 4" in text
    assert "chains=cores" in text
    assert 'stage0_likelihood = "source"' in text
    assert "stage0_likelihood=stage0_likelihood" in text
    assert "perturbation_discovery_top_k=perturbation_top_k" in text
    assert 'stage1_likelihood = "critical-arc"' in text
    assert "critical-arc-centroid" not in text
    assert "critical-arc-mixture" not in text


def test_ff_sims_notebook_is_self_contained_and_config_native() -> None:
    notebook = nbformat.read("notebooks/run_ff_sims_fit.ipynb", as_version=4)
    source = _notebook_source("notebooks/run_ff_sims_fit.ipynb")

    assert "scripts/run_ff_sims_fit.py" not in source
    assert "scripts.run_ff_sims_fit" not in source
    assert "from scripts" not in source
    assert "LensClusterSolverConfig" in source
    assert "compile_run_plan" in source
    assert "LensClusterRunner" in source
    assert "tqdm" in source
    assert "quiet=False" in source
    assert '"ARES"' in source
    assert '"HERA"' in source
    assert "softening_length_kpc" in source
    assert "2.3 / 0.72" in source
    assert "IndependentMemberHaloConfig(population_id=\"potfile_1\", catalog_id=\"2\")" in source
    assert "IndependentMemberHaloConfig(population_id=\"potfile_1\", catalog_id=\"3\")" in source
    assert "IndependentMemberHaloConfig(population_id=\"potfile_1\", catalog_id=\"1\")" in source
    assert "exclude_catalog_ids=(\"2\", \"3\")" not in source
    assert "exclude_catalog_ids=(\"1\", \"2\")" not in source
    assert "cores = 4" in source
    assert "os.environ[\"JAX_NUM_CPU_DEVICES\"] = str(cores)" in source
    assert "RuntimeConfig" in source
    assert "chains=cores" in source
    assert 'stage0_likelihood = "source"' in source
    assert "stage0_likelihood=stage0_likelihood" in source
    assert "perturbation_discovery_top_k=perturbation_top_k" in source
    assert 'stage1_likelihood = "critical-arc"' in source
    assert "critical-arc-centroid" not in source
    assert "critical-arc-mixture" not in source
    assert "available_cpu_cores" not in source
    assert "os.sched_getaffinity" not in source
    assert "os.cpu_count" not in source


def test_dataset_specific_runs_are_composed_from_generic_config_groups() -> None:
    cluster_config = {
        "cluster_key": "hera",
        "output_dir": "results/demo/hera",
        "truth_dir": "data/ff_sims/published/hera",
        "softening_length_kpc": 2.3 / 0.72,
        "rgb": {"q": 6.8, "stretch": 0.0158, "minimum": 0.00105, "red_gain": 0.62, "green_gain": 0.78, "blue_gain": 3.65},
    }

    config = LensClusterSolverConfig(
        model=_minimal_model_config(),
        paths=RunPathsConfig(
            output_dir=cluster_config["output_dir"],
            run_name=f"{cluster_config['cluster_key']}_demo",
        ),
        workflow=WorkflowConfig(fit_mode="sequential", stage0_likelihood="local-jacobian"),
        schedule=StageScheduleConfig(
            fit_method=("svi+nuts",),
            svi_steps=(100, 200),
            refresh_every=(None, 50),
            warmup=(10,),
            samples=(20,),
            sampling_refresh_runs=(1,),
            max_tree_depth=(8,),
        ),
        scaling=ScalingModelConfig(softening_length_kpc=cluster_config["softening_length_kpc"], scaling_scatter=True),
        truth=TruthRecoveryConfig(
            kappa_true_fits=f"{cluster_config['truth_dir']}/kappa_z9_0.fits",
            gammax_true_fits=f"{cluster_config['truth_dir']}/gammax_z9_0.fits",
            gammay_true_fits=f"{cluster_config['truth_dir']}/gammay_z9_0.fits",
            truth_grid_mode="posterior",
            truth_grid_draws=64,
        ),
        image_catalog=ImageCatalogCutoutConfig(
            image_dir="data/ff_sims",
            image_scale="auto",
            bands=("F435W", "F606W", "F814W"),
            rgb=RGBDisplayConfig(**cluster_config["rgb"]),
        ),
        image_diagnostics=ImageDiagnosticsConfig(exact_image_finder="local-lm-adaptive"),
    )

    plan = compile_run_plan(config)

    assert plan.runtime_args.kappa_true_fits == "data/ff_sims/published/hera/kappa_z9_0.fits"
    assert plan.runtime_args.softening_length_kpc == pytest.approx(2.3 / 0.72)
    assert plan.runtime_args.image_catalog_family_cutout_image_dir == "data/ff_sims"


def test_run_xsh_is_ff_sims_only_but_keeps_hff_rgb_constants() -> None:
    text = Path("run.xsh").read_text(encoding="utf-8")

    assert "HFF_RGB_BANDS" in text
    assert "HFF_RGB_DISPLAY" in text
    assert "ARES" in text
    assert "HERA" in text
    for removed_cluster in ("A2744", "M0416", "M1206", "AS1063", "A307"):
        assert removed_cluster not in text
