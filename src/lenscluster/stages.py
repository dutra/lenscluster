from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .planning import RunPlan


@dataclass(frozen=True)
class StageExecutionResult:
    completed: bool = True


class StageExecutor(Protocol):
    def execute(self, plan: RunPlan, stage_fit_controls: dict[str, Any]) -> StageExecutionResult:
        ...


class ClusterSolverStageExecutor:
    def execute(self, plan: RunPlan, stage_fit_controls: dict[str, Any]) -> StageExecutionResult:
        from . import cluster_solver

        cluster_solver._run_typed_plan_dispatch(plan.runtime_args, stage_fit_controls)
        return StageExecutionResult(completed=True)
