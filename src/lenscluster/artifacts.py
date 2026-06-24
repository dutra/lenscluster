from __future__ import annotations

from pathlib import Path

from .planning import RunPlan


def root_run_dir(plan: RunPlan) -> Path:
    return plan.output.output_dir / plan.output.run_name


def artifacts_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    return path if path.name == "artifacts" else path / "artifacts"
