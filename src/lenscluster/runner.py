from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import root_run_dir
from .planning import RunPlan
from .stages import ClusterSolverStageExecutor, StageExecutor
from .utils import install_astropy_wcs_warning_filters


@dataclass(frozen=True)
class RunResult:
    run_name: str
    output_dir: Path
    run_dir: Path
    completed: bool


class LensClusterRunner:
    def __init__(self, *, stage_executor: StageExecutor | None = None) -> None:
        self._stage_executor = stage_executor or ClusterSolverStageExecutor()

    def run(self, plan: RunPlan) -> RunResult:
        install_astropy_wcs_warning_filters()

        from . import cluster_solver

        args = plan.runtime_args
        stage_fit_controls = {} if bool(getattr(args, "plots_only", False)) else _stage_fit_controls_from_plan(plan, cluster_solver)
        if plan.runtime.seed is not None:
            cluster_solver.np.random.seed(plan.runtime.seed)
        cluster_solver._configure_debug_log(args, plan.output.run_name, None)
        cluster_solver._log(args, "[main] startup")
        cluster_solver._log_runtime_summary(args)
        default_device = cluster_solver._resolve_jax_device_for_args(
            args,
            "jax_default_device",
            flag_name="jax_default_device",
        )
        smc_device = cluster_solver._resolve_jax_device_for_args(args, "smc_device", flag_name="smc_device")
        cluster_solver._log_jax_device_policy(args, default_device, smc_device)
        with cluster_solver._jax_device_context(default_device):
            result = self._stage_executor.execute(plan, stage_fit_controls)
        return RunResult(
            run_name=plan.output.run_name,
            output_dir=plan.output.output_dir,
            run_dir=root_run_dir(plan),
            completed=bool(getattr(result, "completed", True)),
        )


def _stage_fit_controls_from_plan(plan: RunPlan, cluster_solver_module: Any) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    for stage in plan.stages:
        if stage.name == "stage0_fast_initializer":
            key = "stage0"
        elif stage.name == "stage1_backprojected_centroid_fit":
            key = "stage1"
        elif stage.name == "stage2_free_source_forward_fit":
            key = "stage2"
        else:
            key = stage.name
        controls[key] = cluster_solver_module.StageFitControls(
            fit_method=stage.fit_method,
            svi_steps=stage.svi_steps,
            refresh_every=stage.refresh_every,
            warmup=stage.warmup,
            samples=stage.samples,
            sampling_refresh_runs=stage.sampling_refresh_runs,
            max_tree_depth=stage.max_tree_depth,
        )
    return controls
