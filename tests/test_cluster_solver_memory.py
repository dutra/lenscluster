from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from lenscluster import cluster_solver as solver


def _controls() -> dict[str, solver.StageFitControls]:
    return {
        "stage2": solver.StageFitControls("svi", 1, 0, 1, 1),
        "stage3": solver.StageFitControls("svi", 1, 0, 1, 1),
        "stage4": solver.StageFitControls("svi+nuts", 1, 0, 1, 1),
    }


def _sequential_args(tmp_path: Path, **updates: Any) -> argparse.Namespace:
    payload: dict[str, Any] = {
        "output_dir": str(tmp_path),
        "run_name": "root",
        "par_path": "input.par",
        "fit_mode": solver.FIT_MODE_SEQUENTIAL,
        "image_plane_mode": solver.IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
        "stage3_image_plane_mode": solver.STAGE3_IMAGE_PLANE_MODE_AUTO,
        "start_at_stage3": True,
        "resume": False,
        "skip_plots": False,
        "quick_diagnostics": False,
        "plots_only": False,
        "skip_validation": True,
        "exact_image_diagnostics_stage3": False,
        "skip_critical_det_diagnostic": True,
        "skip_stage3_image_plane_local_jacobian": False,
        "stage4_fresh_process": True,
        "stage4_sampling_engine": solver.STAGE4_SAMPLING_ENGINE_INHERIT,
        "sampling_engine": solver.SAMPLING_ENGINE_REFRESHING_SURROGATE,
        "potfile_mass_size_reparam": False,
        "fit_cosmology_flat_wcdm": False,
    }
    payload.update(updates)
    return argparse.Namespace(**payload)


def _patch_sequential_fast_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(solver, "_normalize_stage_fit_controls", lambda _args: _controls())
    monkeypatch.setattr(solver, "_stage3_image_plane_enabled", lambda _args: True)
    monkeypatch.setattr(solver, "_stage4_image_plane_enabled", lambda _args: True)
    monkeypatch.setattr(solver, "_stage3_sample_likelihood_mode", lambda _args: solver.SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN)
    monkeypatch.setattr(
        solver,
        "_stage4_sample_likelihood_mode",
        lambda _args: solver.SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
    )
    monkeypatch.setattr(solver, "_physical_best_fit_values_from_artifacts", lambda _path: {"theta": 1.0})
    monkeypatch.setattr(solver, "_source_position_prior_values_from_artifacts", lambda _path: {"1": (0.0, 0.0)})
    monkeypatch.setattr(solver, "_write_sequential_run_summary_txt", lambda *_args, **_kwargs: (None, ""))


def test_stage4_runs_in_spawned_process_after_stage3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_sequential_fast_paths(monkeypatch)
    args = _sequential_args(tmp_path, stage4_sampling_engine=solver.SAMPLING_ENGINE_FULL)
    parent_calls: list[tuple[str, str, str]] = []
    fresh_calls: list[tuple[argparse.Namespace, str, str, dict[str, Any]]] = []

    def fake_run_single_stage(stage_args: argparse.Namespace, fit_mode: str, run_name: str, **kwargs: Any) -> Path:
        parent_calls.append((fit_mode, run_name, str(kwargs.get("sample_likelihood_mode"))))
        return Path(stage_args.output_dir) / run_name

    def fake_fresh(stage_args: argparse.Namespace, fit_mode: str, run_name: str, **kwargs: Any) -> Path:
        fresh_calls.append((stage_args, fit_mode, run_name, kwargs))
        return Path(stage_args.output_dir) / run_name

    monkeypatch.setattr(solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(solver, "_run_single_stage_in_fresh_process", fake_fresh)

    solver._run_sequential(args)

    assert parent_calls == [("joint", "root/stage3_image_plane", solver.SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN)]
    assert len(fresh_calls) == 1
    stage4_args, fit_mode, run_name, kwargs = fresh_calls[0]
    assert fit_mode == "joint"
    assert run_name == "root/stage4_critical_arc_mixture_image_plane"
    assert stage4_args.sampling_engine == solver.SAMPLING_ENGINE_FULL
    assert kwargs["sample_likelihood_mode"] == solver.SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE
    summary = json.loads((tmp_path / "root" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["stage4_fresh_process"] is True
    assert summary["stage4_sampling_engine_effective"] == solver.SAMPLING_ENGINE_FULL


def test_stage4_fast_resume_does_not_respawn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_sequential_fast_paths(monkeypatch)
    monkeypatch.setattr(solver, "_require_fast_resume_plot_artifacts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(solver, "_require_fast_resume_cosmology_compatibility", lambda *_args, **_kwargs: None)
    args = _sequential_args(tmp_path, resume=solver.RESUME_MODE_FAST)
    parent_calls: list[str] = []

    def fake_run_single_stage(stage_args: argparse.Namespace, _fit_mode: str, run_name: str, **_kwargs: Any) -> Path:
        parent_calls.append(run_name)
        return Path(stage_args.output_dir) / run_name

    monkeypatch.setattr(solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(
        solver,
        "_run_single_stage_in_fresh_process",
        lambda *_args, **_kwargs: pytest.fail("stage4 should not respawn during --resume fast"),
    )

    solver._run_sequential(args)

    assert parent_calls == ["root/stage4_critical_arc_mixture_image_plane"]


def test_stage4_no_fresh_process_preserves_in_process_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_sequential_fast_paths(monkeypatch)
    args = _sequential_args(tmp_path, stage4_fresh_process=False)
    parent_calls: list[str] = []

    def fake_run_single_stage(stage_args: argparse.Namespace, _fit_mode: str, run_name: str, **_kwargs: Any) -> Path:
        parent_calls.append(run_name)
        return Path(stage_args.output_dir) / run_name

    monkeypatch.setattr(solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(
        solver,
        "_run_single_stage_in_fresh_process",
        lambda *_args, **_kwargs: pytest.fail("stage4 should not respawn with --no-stage4-fresh-process"),
    )

    solver._run_sequential(args)

    assert parent_calls == ["root/stage3_image_plane", "root/stage4_critical_arc_mixture_image_plane"]


def test_fresh_stage_process_error_is_propagated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeQueue:
        def get(self, timeout: float) -> dict[str, str]:
            del timeout
            return {"status": "error", "traceback": "child boom"}

        def close(self) -> None:
            pass

        def join_thread(self) -> None:
            pass

    class FakeProcess:
        exitcode = 0

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    class FakeContext:
        def Queue(self) -> FakeQueue:
            return FakeQueue()

        def Process(self, **_kwargs: Any) -> FakeProcess:
            return FakeProcess()

    monkeypatch.setattr(solver.multiprocessing, "get_context", lambda _method: FakeContext())
    with pytest.raises(RuntimeError, match="child boom"):
        solver._run_single_stage_in_fresh_process(
            _sequential_args(tmp_path),
            "joint",
            "root/stage4_critical_arc_mixture_image_plane",
        )


def test_prepare_direct_evaluator_clears_after_initial_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeEvaluator:
        surrogate_enabled = True
        active_scaling_component_indices: list[int] = []
        inactive_scaling_component_indices: list[int] = []
        traced_bin_data: list[Any] = []
        sample_likelihood_mode = solver.SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE
        sampling_engine = solver.SAMPLING_ENGINE_REFRESHING_SURROGATE
        surrogate_cache_by_z: dict[float, Any] = {}
        scaling_scatter_cache_by_z: dict[float, Any] = {}
        source_metric_cache_by_z: dict[float, Any] = {}
        surrogate_reference_params = None
        scaling_scatter_reference_params = None
        source_metric_reference_params = None
        state = argparse.Namespace(family_data=[], bin_data=[], parameter_specs=[])
        timing_totals = {"initial_jit_compile": 0.0}

        def refresh_surrogate(self, _params: Any, reason: str) -> None:
            events.append(f"surrogate:{reason}")

        def refresh_scaling_scatter_cache(self, _params: Any, reason: str) -> None:
            events.append(f"scaling:{reason}")

        def refresh_source_metric_cache(self, _params: Any, reason: str) -> None:
            events.append(f"source_metric:{reason}")

        def source_loglike(self, _params: Any) -> float:
            events.append("compile")
            return -1.0

    monkeypatch.setattr(solver, "_build_cluster_evaluator_from_args", lambda *_args, **_kwargs: FakeEvaluator())
    monkeypatch.setattr(solver, "_log_evaluator_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(solver.jax, "clear_caches", lambda: events.append("clear"))
    args = argparse.Namespace(jax_clear_caches_after_svi_refresh=True)
    state = argparse.Namespace(parameter_specs=[])

    solver._prepare_direct_evaluator(args, state)

    assert events == [
        "surrogate:svi_nuts_initial",
        "scaling:svi_nuts_initial",
        "source_metric:svi_nuts_initial",
        "clear",
        "compile",
    ]


def test_evaluator_shape_diagnostics_report_one_dimensional_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    bin_data = solver.TracedBinData(
        effective_z_source=1.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.array([0, 0, 0]),
        x_obs=jnp.array([1.0, 2.0, 3.0]),
        y_obs=jnp.array([4.0, 5.0, 6.0]),
        sigma_per_image=jnp.ones(3),
        reliability_per_image=jnp.ones(3),
        image_has_constraint=jnp.array([True, True, True]),
    )
    evaluator = argparse.Namespace(
        traced_bin_data=[bin_data],
        sampling_engine=solver.SAMPLING_ENGINE_REFRESHING_SURROGATE,
        sample_likelihood_mode=solver.SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        surrogate_enabled=True,
        active_scaling_component_indices=[1, 2],
        inactive_scaling_component_indices=[3],
        surrogate_cache_by_z={},
        scaling_scatter_cache_by_z={},
        source_metric_cache_by_z={},
        surrogate_reference_params=None,
        scaling_scatter_reference_params=None,
        source_metric_reference_params=None,
    )
    monkeypatch.setattr(solver, "_log", lambda _args, message: messages.append(message))

    solver._log_evaluator_memory_shape_diagnostics(argparse.Namespace(), evaluator, reason="test")

    assert messages
    assert "images=3" in messages[0]
    assert "x_obs_ndim=1" in messages[0]
    assert "y_obs_ndim=1" in messages[0]
    assert "jacobian_surrogate=True" in messages[0]
