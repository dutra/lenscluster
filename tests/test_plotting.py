import argparse
import math
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import lenscluster.plotting as plotting
from lenscluster.plotting import plot_path
from lenscluster.model import PosteriorResults, ParameterSpec


def test_plot_path_creates_directory(tmp_path: Path) -> None:
    output = plot_path(tmp_path / "plots", "summary.png")

    assert output == tmp_path / "plots" / "summary.pdf"
    assert output.parent.is_dir()


def test_run_plot_tasks_with_progress_tracks_plot_names(monkeypatch: Any) -> None:
    calls: list[str] = []
    phases: list[str] = []
    progress_instances: list[Any] = []

    def fake_logged_phase(args: argparse.Namespace, phase_name: str, fn: Any) -> Any:
        phases.append(phase_name)
        return fn()

    class FakeProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.descriptions: list[str] = []
            self.total: int | None = None
            progress_instances.append(self)

        def __enter__(self) -> "FakeProgress":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def add_task(self, description: str, *, total: int) -> int:
            self.descriptions.append(description)
            self.total = total
            return 1

        def update(self, task_id: int, **kwargs: Any) -> None:
            if "description" in kwargs:
                self.descriptions.append(kwargs["description"])

        def advance(self, task_id: int) -> None:
            return None

    monkeypatch.setattr(plotting, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(plotting, "Progress", FakeProgress)

    tasks: list[plotting.PlotTask] = [
        ("corner", "plots.corner", lambda: calls.append("corner")),
        ("image_plane_fit", "plots.image_plane_fit", lambda: calls.append("image_plane_fit")),
    ]

    plotting._run_plot_tasks_with_progress(argparse.Namespace(quiet=False), tasks)

    assert calls == ["corner", "image_plane_fit"]
    assert phases == ["plots.corner", "plots.image_plane_fit"]
    assert len(progress_instances) == 1
    assert progress_instances[0].total == 2
    assert progress_instances[0].descriptions == [
        "plots",
        "plots: corner",
        "plots: image_plane_fit",
        "plots: complete",
    ]


def test_run_plot_tasks_with_progress_quiet_skips_progress(monkeypatch: Any) -> None:
    calls: list[str] = []
    phases: list[str] = []

    def fake_logged_phase(args: argparse.Namespace, phase_name: str, fn: Any) -> Any:
        phases.append(phase_name)
        return fn()

    def fail_progress(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("quiet plot execution should not create a progress bar")

    monkeypatch.setattr(plotting, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(plotting, "Progress", fail_progress)

    tasks: list[plotting.PlotTask] = [
        ("corner", "plots.corner", lambda: calls.append("corner")),
        ("trace", "plots.trace", lambda: calls.append("trace")),
    ]

    plotting._run_plot_tasks_with_progress(argparse.Namespace(quiet=True), tasks)

    assert calls == ["corner", "trace"]
    assert phases == ["plots.corner", "plots.trace"]


def _corner_test_specs(component_family: str = "large") -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="x",
            sample_name="x",
            potential_id="mock",
            profile_type=81,
            field="x",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family=component_family,
        ),
        ParameterSpec(
            name="y",
            sample_name="y",
            potential_id="mock",
            profile_type=81,
            field="y",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family=component_family,
        ),
    ]


def _mixed_cosmology_test_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="halo.x",
            sample_name="halo_x",
            potential_id="halo",
            profile_type=81,
            field="x",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family="large",
        ),
        ParameterSpec(
            name="cosmology.Om0",
            sample_name="cosmology_Om0",
            potential_id="cosmology",
            profile_type=0,
            field="Om0",
            prior_kind="uniform",
            lower=0.05,
            upper=0.6,
            step=0.01,
            component_family="cosmology",
        ),
        ParameterSpec(
            name="cosmology.w0",
            sample_name="cosmology_w0",
            potential_id="cosmology",
            profile_type=0,
            field="w0",
            prior_kind="uniform",
            lower=-2.0,
            upper=-0.3,
            step=0.05,
            component_family="cosmology",
        ),
    ]


def test_cosmology_parameter_subset_keeps_only_cosmology_columns() -> None:
    samples = np.asarray(
        [
            [10.0, 0.28, -1.1],
            [11.0, 0.30, -1.0],
            [12.0, 0.32, -0.9],
        ],
        dtype=float,
    )
    best_fit = np.asarray([11.5, 0.31, -0.95], dtype=float)

    subset_specs, subset_samples, subset_best_fit = plotting._cosmology_parameter_subset(
        _mixed_cosmology_test_specs(),
        samples,
        best_fit,
    )

    assert [spec.name for spec in subset_specs] == ["cosmology.Om0", "cosmology.w0"]
    np.testing.assert_allclose(subset_samples, samples[:, [1, 2]])
    np.testing.assert_allclose(subset_best_fit, best_fit[[1, 2]])


def test_corner_overlays_gold_best_fit_and_preserves_truths(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []

    class FakeFig:
        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    class FakeCorner:
        def corner(self, samples: np.ndarray, **kwargs: Any) -> FakeFig:
            calls.append(("corner", np.asarray(samples), kwargs))
            return FakeFig()

        def overplot_lines(self, _fig: FakeFig, xs: list[float | None], **kwargs: Any) -> None:
            calls.append(("lines", xs, kwargs))

        def overplot_points(self, _fig: FakeFig, xs: list[list[float]], **kwargs: Any) -> None:
            calls.append(("points", xs, kwargs))

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)

    plotting._plot_corner(
        tmp_path,
        np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]], dtype=float),
        _corner_test_specs(),
        truth_values={"x": 0.5, "y": 2.5},
        best_fit_values={"x": 1.5, "y": 3.5},
        previous_stage_best_values={"x": 1.25, "y": 3.25},
    )

    assert calls[0][0] == "corner"
    assert calls[0][2]["truths"] == [0.5, 2.5]
    assert calls[1] == ("lines", [1.25, 3.25], {"color": plotting.CORNER_PREVIOUS_STAGE_COLOR})
    assert calls[2] == ("lines", [1.5, 3.5], {"color": plotting.CORNER_BEST_FIT_COLOR})
    assert calls[3] == (
        "points",
        [[1.5, 3.5]],
        {"marker": "s", "color": plotting.CORNER_BEST_FIT_COLOR},
    )


def test_corner_excludes_source_positions_before_finite_filtering(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []
    specs = [
        ParameterSpec(
            name="x",
            sample_name="x",
            potential_id="mock",
            profile_type=81,
            field="x",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family="large",
        ),
        ParameterSpec(
            name="source.1.beta_x",
            sample_name="source_1_beta_x",
            potential_id="1",
            profile_type=0,
            field="beta_x",
            prior_kind="normal",
            lower=-np.inf,
            upper=np.inf,
            step=0.1,
            component_family="source_position",
        ),
        ParameterSpec(
            name="y",
            sample_name="y",
            potential_id="mock",
            profile_type=81,
            field="y",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family="large",
        ),
        ParameterSpec(
            name="source.1.beta_y",
            sample_name="source_1_beta_y",
            potential_id="1",
            profile_type=0,
            field="beta_y",
            prior_kind="normal",
            lower=-np.inf,
            upper=np.inf,
            step=0.1,
            component_family="source_position",
        ),
    ]

    class FakeFig:
        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    class FakeCorner:
        def corner(self, samples: np.ndarray, **kwargs: Any) -> FakeFig:
            calls.append(("corner", np.asarray(samples), kwargs))
            return FakeFig()

        def overplot_lines(self, _fig: FakeFig, xs: list[float | None], **kwargs: Any) -> None:
            calls.append(("lines", xs, kwargs))

        def overplot_points(self, _fig: FakeFig, xs: list[list[float]], **kwargs: Any) -> None:
            calls.append(("points", xs, kwargs))

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)

    plotting._plot_corner(
        tmp_path,
        np.asarray(
            [
                [0.0, np.nan, 1.0, 10.0],
                [1.0, np.inf, 2.0, 11.0],
                [2.0, -np.inf, 4.0, 12.0],
            ],
            dtype=float,
        ),
        specs,
        truth_values={"x": 0.5, "y": 2.5, "source.1.beta_x": 9.0, "source.1.beta_y": 9.5},
        best_fit_values={"x": 1.5, "y": 3.5, "source.1.beta_x": 10.0, "source.1.beta_y": 10.5},
    )

    assert calls[0][0] == "corner"
    np.testing.assert_allclose(calls[0][1], np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]]))
    assert calls[0][2]["labels"] == ["x", "y"]
    assert calls[0][2]["truths"] == [0.5, 2.5]
    assert calls[1] == ("lines", [1.5, 3.5], {"color": plotting.CORNER_BEST_FIT_COLOR})
    assert calls[2] == (
        "points",
        [[1.5, 3.5]],
        {"marker": "s", "color": plotting.CORNER_BEST_FIT_COLOR},
    )


def test_potfile_corner_uses_scaling_best_fit_values(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []

    class FakeFig:
        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    class FakeCorner:
        def corner(self, samples: np.ndarray, **kwargs: Any) -> FakeFig:
            calls.append(("corner", np.asarray(samples), kwargs))
            return FakeFig()

        def overplot_lines(self, _fig: FakeFig, xs: list[float | None], **kwargs: Any) -> None:
            calls.append(("lines", xs, kwargs))

        def overplot_points(self, _fig: FakeFig, xs: list[list[float]], **kwargs: Any) -> None:
            calls.append(("points", xs, kwargs))

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)

    plotting._plot_potfile_corner(
        tmp_path,
        np.asarray([[10.0, 20.0], [11.0, 22.0], [13.0, 24.0]], dtype=float),
        _corner_test_specs(component_family="scaling"),
        truth_values={"x": 10.5, "y": 22.5},
        best_fit_values={"x": 12.0, "y": 23.0},
        previous_stage_best_values={"x": 11.5, "y": 22.0},
    )

    assert calls[0][2]["truths"] == [10.5, 22.5]
    assert calls[1] == ("lines", [11.5, 22.0], {"color": plotting.CORNER_PREVIOUS_STAGE_COLOR})
    assert calls[2] == ("lines", [12.0, 23.0], {"color": plotting.CORNER_BEST_FIT_COLOR})
    assert calls[3] == (
        "points",
        [[12.0, 23.0]],
        {"marker": "s", "color": plotting.CORNER_BEST_FIT_COLOR},
    )


def test_cosmology_corner_uses_sample_name_truths_and_best_fit(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []
    samples = np.asarray(
        [
            [10.0, 0.28, -1.1],
            [11.0, 0.30, -1.0],
            [12.0, 0.32, -0.9],
        ],
        dtype=float,
    )
    best_fit = np.asarray([11.5, 0.31, -0.95], dtype=float)
    cosmology_specs, cosmology_samples, cosmology_best_fit = plotting._cosmology_parameter_subset(
        _mixed_cosmology_test_specs(),
        samples,
        best_fit,
    )

    class FakeFig:
        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    class FakeCorner:
        def corner(self, samples: np.ndarray, **kwargs: Any) -> FakeFig:
            calls.append(("corner", np.asarray(samples), kwargs))
            return FakeFig()

        def overplot_lines(self, _fig: FakeFig, xs: list[float | None], **kwargs: Any) -> None:
            calls.append(("lines", xs, kwargs))

        def overplot_points(self, _fig: FakeFig, xs: list[list[float]], **kwargs: Any) -> None:
            calls.append(("points", xs, kwargs))

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)

    plotting._plot_cosmology_corner(
        tmp_path,
        cosmology_samples,
        cosmology_specs,
        truth_values={"cosmology_Om0": 0.3, "cosmology_w0": -1.0},
        best_fit_values=plotting._best_fit_values_for_specs(cosmology_specs, cosmology_best_fit),
        previous_stage_best_values={"cosmology_Om0": 0.29, "cosmology_w0": -1.05},
    )

    assert calls[0][0] == "corner"
    np.testing.assert_allclose(calls[0][1], samples[:, [1, 2]])
    assert calls[0][2]["labels"] == ["cosmology.Om0", "cosmology.w0"]
    assert calls[0][2]["truths"] == [0.3, -1.0]
    assert calls[1] == ("lines", [0.29, -1.05], {"color": plotting.CORNER_PREVIOUS_STAGE_COLOR})
    assert calls[2] == ("lines", [0.31, -0.95], {"color": plotting.CORNER_BEST_FIT_COLOR})
    assert calls[3] == (
        "points",
        [[0.31, -0.95]],
        {"marker": "s", "color": plotting.CORNER_BEST_FIT_COLOR},
    )


def test_fit_quality_tables_cap_draws_convert_physical_and_quantile() -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        sigma_arcsec=5.0,
        n_images=2,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 2.0], dtype=float),
        y_obs=np.asarray([0.0, 1.0], dtype=float),
    )
    state = SimpleNamespace(parameter_specs=[], family_data=[family])

    class FakeModel:
        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            offset = float(kwargs_lens[0]["offset"])
            return np.asarray(x, dtype=float) + np.asarray(y, dtype=float) + offset

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.source_plane_covariance_floor = 7.0
            self.converted: list[np.ndarray] = []
            self.exact_latents: list[float] = []

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            theta_array = np.asarray(theta, dtype=float)
            self.converted.append(theta_array.copy())
            return theta_array + 1.0

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 2.0

        def _exact_family_prediction(self, sample_latent: np.ndarray, family_data: Any) -> tuple[np.ndarray, np.ndarray, float]:
            offset = float(np.asarray(sample_latent, dtype=float)[0])
            self.exact_latents.append(offset)
            return family_data.x_obs + offset, family_data.y_obs - offset, offset

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {"offset": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    evaluator = FakeEvaluator()
    results = SimpleNamespace(samples=np.asarray([[0.0], [10.0], [20.0], [30.0]], dtype=float))

    image_df, magnification_df = plotting._fit_quality_tables(
        state,
        evaluator,
        np.asarray([5.0], dtype=float),
        results,
        argparse.Namespace(fit_quality_draws=2, fit_quality_workers=1),
    )

    assert [float(item[0]) for item in evaluator.converted] == [5.0, 0.0, 30.0]
    assert evaluator.exact_latents == [6.0, 1.0, 31.0]

    row = image_df.set_index("image_label").loc["1.1"]
    assert row["z_source"] == pytest.approx(2.0)
    assert row["sigma_arcsec"] == pytest.approx(5.0)
    assert row["image_sigma_int_arcsec"] == pytest.approx(2.0)
    assert row["image_sigma_eff_arcsec"] == pytest.approx(6.0)
    assert row["radius_arcsec"] == pytest.approx(0.0)
    assert row["angle_deg"] == pytest.approx(0.0)
    assert row["x_model_arcsec"] == pytest.approx(6.0)
    assert row["y_model_arcsec"] == pytest.approx(-6.0)
    assert row["x_model_q16"] == pytest.approx(5.8)
    assert row["x_model_q50"] == pytest.approx(16.0)
    assert row["x_model_q84"] == pytest.approx(26.2)
    assert row["image_residual_q50"] == pytest.approx(math.sqrt(2.0) * 16.0)
    assert row["residual_norm"] == pytest.approx(math.sqrt(2.0))
    assert row["residual_norm_q50"] == pytest.approx(math.sqrt(2.0) * 16.0 / 6.0)
    assert bool(row["covered_x_1sigma"]) is True
    assert bool(row["covered_y_1sigma"]) is True
    assert bool(row["covered_xy_1sigma"]) is True
    assert int(row["posterior_valid_draws"]) == 2
    assert int(row["model_produced_image_count"]) == 2
    assert int(row["model_recovered_image_count"]) == 2
    assert int(row["model_missing_image_count"]) == 0
    assert int(row["model_extra_image_count"]) == 0
    assert bool(row["model_multiplicity_failed"]) is False

    mag_row = magnification_df.set_index("image_label").loc["1.2"]
    assert mag_row["magnification_model"] == pytest.approx(9.0)
    assert mag_row["magnification_model_q50"] == pytest.approx(19.0)
    assert int(mag_row["posterior_valid_draws"]) == 2


def test_fit_quality_tables_defaults_to_best_fit_only() -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        sigma_arcsec=1.0,
        n_images=1,
        image_labels=["1.1"],
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
    )
    state = SimpleNamespace(parameter_specs=[], family_data=[family])

    class FakeModel:
        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            del x, y
            return np.asarray([float(kwargs_lens[0]["offset"])], dtype=float)

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.source_plane_covariance_floor = 0.0
            self.converted: list[float] = []
            self.exact_latents: list[float] = []

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            value = float(np.asarray(theta, dtype=float)[0])
            self.converted.append(value)
            return np.asarray([value + 1.0], dtype=float)

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 0.0

        def _exact_family_prediction(self, sample_latent: np.ndarray, family_data: Any) -> tuple[np.ndarray, np.ndarray, float]:
            offset = float(np.asarray(sample_latent, dtype=float)[0])
            self.exact_latents.append(offset)
            return family_data.x_obs + offset, family_data.y_obs, offset

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {"offset": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    evaluator = FakeEvaluator()
    results = SimpleNamespace(samples=np.asarray([[0.0], [10.0], [20.0]], dtype=float))

    image_df, magnification_df = plotting._fit_quality_tables(
        state,
        evaluator,
        np.asarray([5.0], dtype=float),
        results,
        argparse.Namespace(fit_quality_workers=1),
    )

    assert evaluator.converted == [5.0]
    assert evaluator.exact_latents == [6.0]

    image_row = image_df.set_index("image_label").loc["1.1"]
    assert image_row["x_model_arcsec"] == pytest.approx(6.0)
    assert np.isnan(image_row["x_model_q50"])
    assert np.isnan(image_row["image_residual_q50"])
    assert int(image_row["posterior_valid_draws"]) == 0

    mag_row = magnification_df.set_index("image_label").loc["1.1"]
    assert mag_row["magnification_model"] == pytest.approx(6.0)
    assert np.isnan(mag_row["magnification_model_q50"])
    assert int(mag_row["posterior_valid_draws"]) == 0


def test_fit_quality_tables_quick_diagnostics_skips_exact_and_uses_median_std() -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        sigma_arcsec=1.0,
        n_images=1,
        image_labels=["1.1"],
        x_obs=np.asarray([1.0], dtype=float),
        y_obs=np.asarray([2.0], dtype=float),
    )
    state = SimpleNamespace(parameter_specs=[], family_data=[family])

    class FakeModel:
        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            offset = float(kwargs_lens[0]["offset"])
            return np.asarray(x, dtype=float) + np.asarray(y, dtype=float) + offset

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.source_plane_covariance_floor = 0.0
            self.quick_diagnostics = True
            self.exact_calls = 0

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float)

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 0.0

        def _exact_family_prediction(self, _sample_latent: np.ndarray, _family_data: Any) -> tuple[np.ndarray, np.ndarray, float]:
            self.exact_calls += 1
            raise AssertionError("quick diagnostics should not solve exact image positions")

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {"offset": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    evaluator = FakeEvaluator()
    results = SimpleNamespace(samples=np.asarray([[0.0], [2.0], [4.0]], dtype=float))

    image_df, magnification_df = plotting._fit_quality_tables(
        state,
        evaluator,
        np.asarray([10.0], dtype=float),
        results,
        argparse.Namespace(fit_quality_draws=3, fit_quality_workers=1, quick_diagnostics=True),
    )

    assert evaluator.exact_calls == 0
    image_row = image_df.set_index("image_label").loc["1.1"]
    assert bool(image_row["exact_image_prediction_failed"]) is True
    assert np.isnan(image_row["x_model_arcsec"])
    assert np.isnan(image_row["image_residual_arcsec"])
    assert np.isnan(image_row["model_produced_image_count"])
    assert image_row["model_multiplicity_failure_reason"] == "quick_diagnostics"

    mag_row = magnification_df.set_index("image_label").loc["1.1"]
    mag_values = np.asarray([3.0, 5.0, 7.0], dtype=float)
    assert mag_row["magnification_model"] == pytest.approx(13.0)
    assert mag_row["magnification_model_q16"] == pytest.approx(float(np.median(mag_values) - np.std(mag_values)))
    assert mag_row["magnification_model_q50"] == pytest.approx(float(np.median(mag_values)))
    assert mag_row["magnification_model_q84"] == pytest.approx(float(np.median(mag_values) + np.std(mag_values)))


def test_posterior_fit_quality_predictions_parallelizes_per_sample_family(monkeypatch: Any) -> None:
    families = [
        SimpleNamespace(
            family_id=f"fam-{index}",
            z_source=2.0 + index,
            sigma_arcsec=0.5,
            search_window=[10.0, 30.0, 20.0][index],
            n_images=1,
            image_labels=[f"{index}.1"],
            x_obs=np.asarray([float(index)], dtype=float),
            y_obs=np.asarray([float(index + 1)], dtype=float),
        )
        for index in range(3)
    ]
    state = SimpleNamespace(family_data=families)

    class FakeModel:
        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            offset = float(kwargs_lens[0]["offset"])
            return np.asarray(x, dtype=float) + np.asarray(y, dtype=float) + offset

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.source_plane_covariance_floor = 0.0
            self.exact_calls: list[tuple[float, str]] = []

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 0.0

        def _exact_family_prediction(self, sample_latent: np.ndarray, family_data: Any) -> tuple[np.ndarray, np.ndarray, float]:
            offset = float(np.asarray(sample_latent, dtype=float)[0])
            self.exact_calls.append((offset, family_data.family_id))
            return family_data.x_obs + offset, family_data.y_obs - offset, offset

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {"offset": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    submitted: list[tuple[int, int]] = []
    progress_instances: list[Any] = []

    class InlineExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> "InlineExecutor":
            return self

        def __exit__(self, *_exc: Any) -> bool:
            return False

        def submit(self, fn: Any, sample_index: int, family_index: int) -> Future:
            submitted.append((sample_index, family_index))
            future: Future = Future()
            try:
                future.set_result(fn(sample_index, family_index))
            except BaseException as exc:  # pragma: no cover - exercised by Future consumers if it fails
                future.set_exception(exc)
            return future

    class FakeProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.descriptions: list[tuple[int, str]] = []
            self.advances: dict[int, int] = {}
            self.totals: dict[int, int] = {}
            self.transient = kwargs.get("transient")
            self._next_task_id = 0
            progress_instances.append(self)

        def __enter__(self) -> "FakeProgress":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def add_task(self, description: str, *, total: int) -> int:
            self._next_task_id += 1
            task_id = self._next_task_id
            self.descriptions.append((task_id, description))
            self.totals[task_id] = total
            self.advances[task_id] = 0
            return task_id

        def update(self, task_id: int, **kwargs: Any) -> None:
            if "description" in kwargs:
                self.descriptions.append((task_id, kwargs["description"]))

        def advance(self, task_id: int) -> None:
            self.advances[task_id] += 1

    evaluator = FakeEvaluator()
    log_messages: list[str] = []
    monkeypatch.setattr(plotting, "_clone_fit_quality_evaluator", lambda *_args, **_kwargs: evaluator)
    monkeypatch.setattr(plotting, "ThreadPoolExecutor", InlineExecutor)
    monkeypatch.setattr(plotting, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(plotting, "Progress", FakeProgress)
    monkeypatch.setattr(plotting, "_log", lambda _args, message: log_messages.append(str(message)))

    predictions = plotting._posterior_fit_quality_predictions(
        evaluator,
        state,
        [np.asarray([1.0], dtype=float), np.asarray([2.0], dtype=float)],
        argparse.Namespace(fit_quality_workers=4),
    )

    assert submitted == [(0, 1), (1, 1), (0, 2), (1, 2), (0, 0), (1, 0)]
    assert evaluator.exact_calls == [
        (1.0, "fam-1"),
        (2.0, "fam-1"),
        (1.0, "fam-2"),
        (2.0, "fam-2"),
        (1.0, "fam-0"),
        (2.0, "fam-0"),
    ]
    assert len(predictions) == 2
    assert [row["image_label"] for row in predictions[0]["image_rows"]] == ["0.1", "1.1", "2.1"]
    assert len(predictions[0]["magnification_rows"]) == 3
    assert len(progress_instances) == 1
    progress = progress_instances[0]
    assert progress.transient is False
    assert progress.totals == {1: 6, 2: 2}
    assert progress.advances == {1: 6, 2: 2}
    assert progress.descriptions[0:2] == [
        (1, "fit quality exact: 0/6 family diagnostics"),
        (2, "draw progress: 0/2 complete"),
    ]
    assert (2, "draw progress: 1/2 complete") in progress.descriptions
    assert (2, "draw progress: 2/2 complete") in progress.descriptions
    assert (
        1,
        "fit quality exact: 6/6 family diagnostics | completed draw 2/2 family=fam-0 z=2.0000 window=10.0 grid=100x100",
    ) in progress.descriptions
    assert log_messages == [
        (
            "[plot:fit_quality] family diagnostics tasks=6 workers=4 families=3 draws=2 "
            "largest_grid=300x300 total_grid_points=280000"
        ),
        "[plot:fit_quality] draw 1/2 complete families=3/3 completed_tasks=5/6",
        "[plot:fit_quality] draw 2/2 complete families=3/3 completed_tasks=6/6",
    ]


def test_posterior_fit_quality_predictions_tracks_serial_progress(monkeypatch: Any) -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        sigma_arcsec=1.0,
        search_window=10.0,
        n_images=1,
        image_labels=["1.1"],
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
    )
    state = SimpleNamespace(family_data=[family])

    class FakeModel:
        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            del x, y
            return np.asarray([float(kwargs_lens[0]["offset"])], dtype=float)

    class FakeEvaluator:
        def __init__(self) -> None:
            self.source_plane_covariance_floor = 0.0

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 0.0

        def _exact_family_prediction(self, sample_latent: np.ndarray, family_data: Any) -> tuple[np.ndarray, np.ndarray, float]:
            offset = float(np.asarray(sample_latent, dtype=float)[0])
            return family_data.x_obs + offset, family_data.y_obs, offset

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {"offset": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    progress_instances: list[Any] = []

    class FakeProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.descriptions: list[tuple[int, str]] = []
            self.advances: dict[int, int] = {}
            self.totals: dict[int, int] = {}
            self.transient = kwargs.get("transient")
            self._next_task_id = 0
            progress_instances.append(self)

        def __enter__(self) -> "FakeProgress":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def add_task(self, description: str, *, total: int) -> int:
            self._next_task_id += 1
            task_id = self._next_task_id
            self.descriptions.append((task_id, description))
            self.totals[task_id] = total
            self.advances[task_id] = 0
            return task_id

        def update(self, task_id: int, **kwargs: Any) -> None:
            if "description" in kwargs:
                self.descriptions.append((task_id, kwargs["description"]))

        def advance(self, task_id: int) -> None:
            self.advances[task_id] += 1

    monkeypatch.setattr(plotting, "Progress", FakeProgress)
    log_messages: list[str] = []
    monkeypatch.setattr(plotting, "_log", lambda _args, message: log_messages.append(str(message)))

    predictions = plotting._posterior_fit_quality_predictions(
        FakeEvaluator(),
        state,
        [np.asarray([1.0], dtype=float), np.asarray([2.0], dtype=float)],
        argparse.Namespace(fit_quality_workers=1),
    )

    assert len(predictions) == 2
    assert len(progress_instances) == 1
    progress = progress_instances[0]
    assert progress.transient is False
    assert progress.totals == {1: 2, 2: 2}
    assert progress.advances == {1: 2, 2: 2}
    assert progress.descriptions == [
        (1, "fit quality exact: 0/2 family diagnostics"),
        (2, "draw progress: 0/2 complete"),
        (1, "fit quality exact: 0/2 family diagnostics | running draw 1/2 family=1 z=2.0000 window=10.0 grid=50x50"),
        (1, "fit quality exact: 1/2 family diagnostics | completed draw 1/2 family=1 z=2.0000 window=10.0 grid=50x50"),
        (2, "draw progress: 1/2 complete"),
        (1, "fit quality exact: 1/2 family diagnostics | running draw 2/2 family=1 z=2.0000 window=10.0 grid=50x50"),
        (1, "fit quality exact: 2/2 family diagnostics | completed draw 2/2 family=1 z=2.0000 window=10.0 grid=50x50"),
        (2, "draw progress: 2/2 complete"),
    ]
    assert log_messages == [
        (
            "[plot:fit_quality] family diagnostics tasks=2 workers=1 families=1 draws=2 "
            "largest_grid=50x50 total_grid_points=5000"
        ),
        "[plot:fit_quality] draw 1/2 complete families=1/1 completed_tasks=1/2",
        "[plot:fit_quality] draw 2/2 complete families=1/1 completed_tasks=2/2",
    ]


def test_posterior_fit_quality_predictions_quiet_skips_progress(monkeypatch: Any) -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        sigma_arcsec=1.0,
        search_window=10.0,
        n_images=1,
        image_labels=["1.1"],
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
    )
    state = SimpleNamespace(family_data=[family])

    class FakeModel:
        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            del x, y
            return np.asarray([float(kwargs_lens[0]["offset"])], dtype=float)

    class FakeEvaluator:
        def __init__(self) -> None:
            self.source_plane_covariance_floor = 0.0

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 0.0

        def _exact_family_prediction(self, sample_latent: np.ndarray, family_data: Any) -> tuple[np.ndarray, np.ndarray, float]:
            offset = float(np.asarray(sample_latent, dtype=float)[0])
            return family_data.x_obs + offset, family_data.y_obs, offset

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {"offset": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    def fail_progress(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("quiet fit-quality diagnostics should not create a progress bar")

    monkeypatch.setattr(plotting, "Progress", fail_progress)

    predictions = plotting._posterior_fit_quality_predictions(
        FakeEvaluator(),
        state,
        [np.asarray([1.0], dtype=float)],
        argparse.Namespace(fit_quality_workers=1, quiet=True),
    )

    assert len(predictions) == 1


def test_generate_plots_and_tables_writes_fit_quality_outputs(tmp_path: Path, monkeypatch: Any) -> None:
    image_df = pd.DataFrame(
        {
            "family_id": ["1"],
            "image_label": ["1.1"],
            "x_obs_arcsec": [0.0],
            "y_obs_arcsec": [0.0],
            "z_source": [2.0],
            "sigma_arcsec": [0.1],
            "image_sigma_int_arcsec": [0.0],
            "image_sigma_eff_arcsec": [0.1],
            "radius_arcsec": [0.0],
            "angle_deg": [0.0],
            "x_model_arcsec": [0.1],
            "y_model_arcsec": [0.2],
            "image_residual_arcsec": [0.3],
            "exact_image_prediction_failed": [False],
            "x_model_q16": [0.0],
            "x_model_q50": [0.1],
            "x_model_q84": [0.2],
            "y_model_q16": [0.1],
            "y_model_q50": [0.2],
            "y_model_q84": [0.3],
            "image_residual_q16": [0.2],
            "image_residual_q50": [0.3],
            "image_residual_q84": [0.4],
            "residual_norm": [3.0],
            "residual_norm_q50": [3.0],
            "covered_x_1sigma": [True],
            "covered_y_1sigma": [False],
            "covered_xy_1sigma": [False],
            "posterior_valid_draws": [1],
            "posterior_failed_draws": [0],
        }
    )
    magnification_df = pd.DataFrame(
        {
            "family_id": ["1"],
            "image_label": ["1.1"],
            "x_obs_arcsec": [0.0],
            "y_obs_arcsec": [0.0],
            "magnification_model": [2.0],
            "magnification_prediction_failed": [False],
            "magnification_model_q16": [1.5],
            "magnification_model_q50": [2.0],
            "magnification_model_q84": [2.5],
            "posterior_valid_draws": [1],
            "posterior_failed_draws": [0],
        }
    )
    captured_tasks: list[str] = []

    monkeypatch.setattr(plotting, "_summary_table", lambda *_args, **_kwargs: pd.DataFrame({"label": ["mock"]}))
    monkeypatch.setattr(plotting, "_family_diagnostics_table", lambda *_args, **_kwargs: pd.DataFrame({"family_id": ["1"]}))
    monkeypatch.setattr(plotting, "_fit_quality_tables", lambda *_args, **_kwargs: (image_df, magnification_df))
    monkeypatch.setattr(plotting, "_run_summary", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(plotting, "_potfile_constraint_diagnostics_table", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(plotting, "_scaling_parameter_subset", lambda *_args, **_kwargs: ([], np.empty((1, 0)), np.empty((0,))))
    monkeypatch.setattr(plotting, "_cosmology_parameter_subset", lambda *_args, **_kwargs: ([], np.empty((1, 0)), np.empty((0,))))
    monkeypatch.setattr(plotting, "_scaling_grouped_subset", lambda *_args, **_kwargs: ([], np.empty((0, 0, 0))))
    monkeypatch.setattr(plotting, "_best_fit_values_for_specs", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(plotting, "_write_potfile_summary_txt", lambda *_args, **_kwargs: None)

    def capture_tasks(_args: argparse.Namespace, plot_tasks: list[plotting.PlotTask]) -> None:
        captured_tasks.extend(task[0] for task in plot_tasks)

    monkeypatch.setattr(plotting, "_run_plot_tasks_with_progress", capture_tasks)
    state = SimpleNamespace(parameter_specs=[], family_data=[], fit_mode="joint")
    evaluator = SimpleNamespace(scaling_rank_df=pd.DataFrame())
    results = PosteriorResults(
        samples=np.empty((1, 0), dtype=float),
        log_prob=np.asarray([0.0]),
        accept_prob=np.asarray([1.0]),
        diverging=np.asarray([False]),
        num_steps=np.asarray([1.0]),
        warmup_steps=0,
        sample_steps=1,
        num_chains=1,
    )

    plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=0.0),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(quiet=True, plot_caustics=False),
    )

    assert (tmp_path / "tables" / "image_fit_quality.csv").exists()
    assert (tmp_path / "tables" / "image_count_recovery.csv").exists()
    assert (tmp_path / "tables" / "model_magnification.csv").exists()
    assert (tmp_path / "tables" / "run_summary.txt").exists()
    assert "image_recovery" in captured_tasks
    assert "image_count_recovery" in captured_tasks
    assert "model_magnification" in captured_tasks
    assert "normalized_image_residuals" in captured_tasks
    assert "residual_vs_magnification" in captured_tasks
    assert "residual_geometry_trends" in captured_tasks
    assert "posterior_predictive_coverage" in captured_tasks
    assert "exact_vs_approx_prediction_error" in captured_tasks
    assert "caustic_overlay" not in captured_tasks

    captured_tasks.clear()
    state.fit_mode = "large-only"
    plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=0.0),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(quiet=True, plot_caustics=False, caustic_num_pix=250, caustic_source_redshift=7.0),
    )
    assert "caustic_overlay" not in captured_tasks

    captured_tasks.clear()
    plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=0.0),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(quiet=True, plot_caustics=True, caustic_num_pix=250, caustic_source_redshift=7.0),
    )
    assert "caustic_overlay" in captured_tasks

    captured_tasks.clear()
    plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=0.0),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(
            quiet=True,
            plot_caustics=True,
            quick_diagnostics=True,
            caustic_num_pix=250,
            caustic_source_redshift=7.0,
        ),
    )
    assert "caustic_overlay" not in captured_tasks


def test_run_summary_quality_metrics_from_image_fit_quality() -> None:
    state = SimpleNamespace(
        parameter_specs=[ParameterSpec("p", "p", "mock", 81, "x", "uniform", -1.0, 1.0, 0.1)],
        family_data=[SimpleNamespace(family_id="1", n_images=4)],
    )
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0, 0.0, 0.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0, 0.0, 0.0],
            "x_model_arcsec": [1.0, 0.0, 1.0, 10.0],
            "y_model_arcsec": [0.0, 2.0, 1.0, 10.0],
            "image_sigma_eff_arcsec": [1.0, 1.0, 1.0, 1.0],
            "exact_image_prediction_failed": [False, False, False, True],
            "covered_xy_1sigma": [True, False, True, False],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state)

    assert summary["chi_square"] == pytest.approx(7.0)
    assert summary["valid_image_count"] == 3
    assert summary["n_data"] == 8
    assert summary["diagnostic_n_data"] == 6
    assert summary["n_effective_parameters"] == 3
    assert summary["implicit_source_position_parameters"] == 2
    assert summary["dof"] == 5
    assert summary["diagnostic_dof"] == 3
    assert summary["reduced_chi_square"] == pytest.approx(7.0 / 5.0)
    assert summary["aic"] == pytest.approx(13.0)
    assert summary["bic"] == pytest.approx(7.0 + 3.0 * math.log(8.0))
    assert summary["covered_xy_1sigma_fraction"] == pytest.approx(2.0 / 3.0)


def test_image_count_recovery_table_and_plot_write_pdf(tmp_path: Path) -> None:
    state = SimpleNamespace(
        family_data=[
            SimpleNamespace(family_id="1", n_images=3, z_source=2.0, effective_z_source=2.0),
            SimpleNamespace(family_id="2", n_images=2, z_source=3.0, effective_z_source=3.0),
        ]
    )
    image_df = pd.DataFrame(
        {
            "family_id": ["1", "1", "1", "2", "2"],
            "z_source": [2.0, 2.0, 2.0, 3.0, 3.0],
            "model_produced_image_count": [4, 4, 4, 1, 1],
            "model_recovered_image_count": [3, 3, 3, 1, 1],
            "model_missing_image_count": [0, 0, 0, 1, 1],
            "model_extra_image_count": [1, 1, 1, 0, 0],
            "model_multiplicity_failed": [True, True, True, True, True],
            "model_multiplicity_failure_reason": ["extra_model_images"] * 3 + ["missing_model_images"] * 2,
        }
    )

    count_df = plotting._image_count_recovery_table(state, image_df)
    plotting._plot_image_count_recovery(count_df, tmp_path / "image_count_recovery.pdf")
    summary = plotting._image_count_recovery_summary(count_df)

    assert count_df["family_id"].tolist() == ["1", "2"]
    assert count_df.set_index("family_id").loc["1", "produced_image_count"] == 4
    assert count_df.set_index("family_id").loc["2", "missing_image_count"] == 1
    assert summary["model_recovered_image_count"] == 4
    assert summary["model_produced_image_count"] == 5
    assert summary["model_missing_image_count"] == 1
    assert summary["model_extra_image_count"] == 1
    assert (tmp_path / "image_count_recovery.pdf").exists()


def test_chain_diagnostics_summary_uses_grouped_samples() -> None:
    specs = _corner_test_specs()[:2]
    grouped = np.asarray(
        [
            [[0.0, 1.0], [0.1, 1.1], [0.2, 1.2], [0.3, 1.3], [0.4, 1.4], [0.5, 1.5]],
            [[0.05, 1.05], [0.15, 1.15], [0.25, 1.25], [0.35, 1.35], [0.45, 1.45], [0.55, 1.55]],
        ],
        dtype=float,
    )
    posterior = PosteriorResults(
        samples=grouped.reshape((-1, 2)),
        log_prob=np.zeros(12, dtype=float),
        accept_prob=np.ones(12, dtype=float),
        diverging=np.zeros(12, dtype=bool),
        num_steps=np.ones(12, dtype=float),
        warmup_steps=0,
        sample_steps=6,
        num_chains=2,
        grouped_samples=grouped,
    )

    summary = plotting._chain_diagnostics_summary(posterior, specs)

    assert summary["ess_min"] is not None
    assert summary["ess_median"] is not None
    assert summary["rhat_max"] is not None
    assert summary["rhat_median"] is not None
    assert summary["ess_worst_parameter"] in {spec.name for spec in specs}
    assert summary["rhat_worst_parameter"] in {spec.name for spec in specs}


def test_format_run_summary_text_contains_lensing_and_quality_sections() -> None:
    text = plotting._format_run_summary_text(
        {
            "run_name": "mock",
            "fit_mode": "joint",
            "sample_likelihood_mode": "source",
            "sampler": "numpyro_nuts",
            "n_families": 2,
            "n_images": 6,
            "n_parameters": 4,
            "chi_square": 12.0,
            "dof": 4,
            "diagnostic_n_data": 8,
            "diagnostic_dof": 2,
            "reduced_chi_square": 3.0,
            "aic": 20.0,
            "bic": 21.0,
            "ess_min": 10.0,
            "rhat_max": 1.02,
        }
    )

    assert "Lensing Information" in text
    assert "Quality Of Fit" in text
    assert "chi_square" in text
    assert "diagnostic data points" in text
    assert "diagnostic dof" in text
    assert "Rhat max" in text


def test_parse_args_caustic_source_redshift_default_and_explicit(monkeypatch: Any) -> None:
    from lenscluster import cluster_solver

    monkeypatch.setattr(sys, "argv", ["cluster_solver"])
    args = cluster_solver._parse_args()
    assert args.caustic_source_redshift == pytest.approx(7.0)

    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--caustic-source-redshift", "9.5"])
    args = cluster_solver._parse_args()
    assert args.caustic_source_redshift == pytest.approx(9.5)


def test_tangential_critical_curve_caustics_converts_and_rayshoots(monkeypatch: Any) -> None:
    contour_vertices = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    contour_inputs: list[tuple[tuple[int, int], float]] = []

    def fake_find_contours(lambda_tan: np.ndarray, level: float) -> list[np.ndarray]:
        contour_inputs.append((lambda_tan.shape, level))
        return [contour_vertices]

    class FakeModel:
        def __init__(self) -> None:
            self.ray_inputs: list[tuple[np.ndarray, np.ndarray]] = []

        def hessian(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            x_array = np.asarray(x, dtype=float)
            return (
                np.zeros_like(x_array),
                np.zeros_like(x_array),
                np.zeros_like(x_array),
                np.zeros_like(x_array),
            )

        def ray_shooting(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
            x_array = np.asarray(x, dtype=float)
            y_array = np.asarray(y, dtype=float)
            self.ray_inputs.append((x_array.copy(), y_array.copy()))
            return x_array + 10.0, y_array - 10.0

    monkeypatch.setattr(plotting, "find_contours", fake_find_contours)
    model = FakeModel()

    contours = plotting._tangential_critical_curve_caustics(
        model,
        [{"mock": 1.0}],
        np.asarray([-2.0, 0.0, 2.0], dtype=float),
        np.asarray([10.0, 20.0, 30.0], dtype=float),
    )

    assert contour_inputs == [((3, 3), 0.0)]
    assert len(contours) == 1
    contour = contours[0]
    np.testing.assert_allclose(contour["critical_x"], [-2.0, 2.0, 2.0, -2.0])
    np.testing.assert_allclose(contour["critical_y"], [10.0, 10.0, 30.0, 30.0])
    np.testing.assert_allclose(contour["caustic_x"], [8.0, 12.0, 12.0, 8.0])
    np.testing.assert_allclose(contour["caustic_y"], [0.0, 0.0, 20.0, 20.0])
    np.testing.assert_allclose(model.ray_inputs[0][0], contour["critical_x"])
    np.testing.assert_allclose(model.ray_inputs[0][1], contour["critical_y"])


def test_plot_caustic_overlay_uses_configured_redshift_and_scatter(monkeypatch: Any, tmp_path: Path) -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, -1.0], dtype=float),
    )
    state = SimpleNamespace(z_lens=0.3, parameter_specs=[], family_data=[family])

    class FakeModel:
        pass

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, FakeModel] = {}
            self.converted: list[np.ndarray] = []
            self.model_z: list[float] = []
            self.packed_z: list[float] = []

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            theta_array = np.asarray(theta, dtype=float)
            self.converted.append(theta_array.copy())
            return theta_array + 1.0

        def _get_exact_model_solver(self, z_source: float) -> tuple[FakeModel, None]:
            self.model_z.append(float(z_source))
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: Any, z_source: float) -> dict[str, float]:
            self.packed_z.append(float(z_source))
            return {"latent": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

        def evaluate(self, params_latent: np.ndarray, validate_all_families: bool = False) -> Any:
            return SimpleNamespace(family_predictions={"1": {"source_x": 0.5, "source_y": -0.25}})

    class FakeAxis:
        def __init__(self) -> None:
            self.plots: list[tuple[Any, Any, dict[str, Any]]] = []
            self.scatters: list[tuple[Any, Any, dict[str, Any]]] = []

        def plot(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.plots.append((x, y, kwargs))

        def scatter(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.scatters.append((x, y, kwargs))

        def invert_xaxis(self) -> None:
            return None

        def set_xlabel(self, _label: str) -> None:
            return None

        def set_ylabel(self, _label: str) -> None:
            return None

        def set_title(self, _title: str) -> None:
            return None

    class FakeFig:
        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).touch()

    image_ax = FakeAxis()
    source_ax = FakeAxis()
    helper_calls: list[tuple[list[dict[str, float]], int, int]] = []

    def fake_subplots(*_args: Any, **_kwargs: Any) -> tuple[FakeFig, list[FakeAxis]]:
        return FakeFig(), [image_ax, source_ax]

    def fake_contours(
        model: FakeModel,
        kwargs_lens: list[dict[str, float]],
        x_axis: np.ndarray,
        y_axis: np.ndarray,
    ) -> list[dict[str, np.ndarray]]:
        helper_calls.append((kwargs_lens, len(x_axis), len(y_axis)))
        return [
            {
                "critical_x": np.asarray([1.0, 2.0], dtype=float),
                "critical_y": np.asarray([3.0, 4.0], dtype=float),
                "caustic_x": np.asarray([0.1, 0.2], dtype=float),
                "caustic_y": np.asarray([-0.1, -0.2], dtype=float),
            }
        ]

    monkeypatch.setattr(plotting.plt, "subplots", fake_subplots)
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plotting, "_tangential_critical_curve_caustics", fake_contours)
    evaluator = FakeEvaluator()

    plotting._plot_caustic_overlay(
        tmp_path,
        evaluator,
        np.asarray([4.0], dtype=float),
        caustic_num_pix=12,
        caustic_source_redshift=7.0,
    )

    assert (tmp_path / "caustic_overlay.pdf").exists()
    assert [float(item[0]) for item in evaluator.converted] == [4.0]
    assert evaluator.model_z == [7.0]
    assert evaluator.packed_z == [7.0]
    assert helper_calls == [([{"latent": 5.0}], 250, 250)]
    assert len(image_ax.plots) == 1
    assert source_ax.plots == []
    assert len(source_ax.scatters) == 2
    np.testing.assert_allclose(source_ax.scatters[0][0], [0.1, 0.2])
    np.testing.assert_allclose(source_ax.scatters[0][1], [-0.1, -0.2])


def test_image_plane_fit_uses_family_colored_observed_cross_and_model_point(monkeypatch: Any, tmp_path: Path) -> None:
    class FakeAxis:
        def __init__(self) -> None:
            self.scatters: list[tuple[Any, Any, dict[str, Any]]] = []
            self.plots: list[tuple[Any, Any, dict[str, Any]]] = []

        def scatter(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.scatters.append((x, y, kwargs))

        def plot(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.plots.append((x, y, kwargs))

        def invert_xaxis(self) -> None:
            return None

        def set_xlabel(self, _label: str) -> None:
            return None

        def set_ylabel(self, _label: str) -> None:
            return None

        def set_title(self, _title: str) -> None:
            return None

        def legend(self, **_kwargs: Any) -> None:
            return None

    class FakeFig:
        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).touch()

    axis = FakeAxis()
    monkeypatch.setattr(plotting.plt, "subplots", lambda *_args, **_kwargs: (FakeFig(), axis))
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)

    family = SimpleNamespace(
        family_id="1",
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([2.0, 3.0], dtype=float),
    )
    state = SimpleNamespace(family_data=[family])
    best_eval = SimpleNamespace(
        family_predictions={"1": {"x_pred": np.asarray([0.1, 1.1]), "y_pred": np.asarray([2.1, 3.1])}}
    )

    plotting._plot_image_plane_fit(tmp_path, state, best_eval)

    assert (tmp_path / "image_plane_fit.pdf").exists()
    observed_kwargs = axis.scatters[0][2]
    model_kwargs = axis.scatters[1][2]
    assert observed_kwargs["marker"] == "x"
    assert model_kwargs["marker"] == "o"
    assert model_kwargs["alpha"] < 1.0
    assert observed_kwargs["color"] == model_kwargs["color"]


def test_image_plane_fit_handles_missing_model_predictions(monkeypatch: Any, tmp_path: Path) -> None:
    class FakeAxis:
        def __init__(self) -> None:
            self.scatters: list[tuple[Any, Any, dict[str, Any]]] = []
            self.plots: list[tuple[Any, Any, dict[str, Any]]] = []

        def scatter(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.scatters.append((x, y, kwargs))

        def plot(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.plots.append((x, y, kwargs))

        def invert_xaxis(self) -> None:
            return None

        def set_xlabel(self, _label: str) -> None:
            return None

        def set_ylabel(self, _label: str) -> None:
            return None

        def set_title(self, _title: str) -> None:
            return None

        def legend(self, **_kwargs: Any) -> None:
            return None

    class FakeFig:
        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).touch()

    axis = FakeAxis()
    monkeypatch.setattr(plotting.plt, "subplots", lambda *_args, **_kwargs: (FakeFig(), axis))
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)

    family = SimpleNamespace(
        family_id="1",
        n_images=2,
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([2.0, 3.0], dtype=float),
    )
    state = SimpleNamespace(family_data=[family])
    best_eval = SimpleNamespace(family_predictions={"1": {"failed": True}})

    plotting._plot_image_plane_fit(tmp_path, state, best_eval)

    assert (tmp_path / "image_plane_fit.pdf").exists()
    assert len(axis.scatters) == 1
    assert axis.scatters[0][2]["marker"] == "x"
    assert axis.plots == []


def test_image_recovery_uses_family_colored_observed_cross_and_model_point(monkeypatch: Any, tmp_path: Path) -> None:
    class FakeAxis:
        def __init__(self) -> None:
            self.scatters: list[tuple[Any, Any, dict[str, Any]]] = []
            self.errorbars: list[tuple[Any, Any, dict[str, Any]]] = []
            self.plots: list[tuple[Any, Any, dict[str, Any]]] = []

        def scatter(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.scatters.append((x, y, kwargs))

        def errorbar(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.errorbars.append((x, y, kwargs))

        def plot(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.plots.append((x, y, kwargs))

        def invert_xaxis(self) -> None:
            return None

        def set_xlabel(self, _label: str) -> None:
            return None

        def set_ylabel(self, _label: str) -> None:
            return None

        def set_title(self, _title: str) -> None:
            return None

        def legend(self, **_kwargs: Any) -> None:
            return None

        def set_xticks(self, _ticks: Any) -> None:
            return None

        def set_xticklabels(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    class FakeFig:
        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).touch()

    image_axis = FakeAxis()
    residual_axis = FakeAxis()
    monkeypatch.setattr(plotting.plt, "subplots", lambda *_args, **_kwargs: (FakeFig(), [image_axis, residual_axis]))
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)
    image_df = pd.DataFrame(
        {
            "family_id": ["1", "1", "2"],
            "image_label": ["1.1", "1.2", "2.1"],
            "x_obs_arcsec": [0.0, 2.0, 5.0],
            "y_obs_arcsec": [0.0, 1.0, -1.0],
            "x_model_arcsec": [0.1, 2.2, 4.8],
            "y_model_arcsec": [0.2, 1.1, -1.1],
            "x_model_q16": [0.0, 2.1, 4.7],
            "x_model_q50": [0.1, 2.2, 4.8],
            "x_model_q84": [0.2, 2.3, 4.9],
            "y_model_q16": [0.1, 1.0, -1.2],
            "y_model_q50": [0.2, 1.1, -1.1],
            "y_model_q84": [0.3, 1.2, -1.0],
            "image_residual_arcsec": [0.2, 0.3, 0.4],
            "image_residual_q16": [0.1, 0.2, 0.3],
            "image_residual_q50": [0.2, 0.3, 0.4],
            "image_residual_q84": [0.3, 0.4, 0.5],
        }
    )

    plotting._plot_image_recovery_fit_quality(image_df, tmp_path / "image_recovery.pdf")

    assert (tmp_path / "image_recovery.pdf").exists()
    assert image_axis.scatters[0][2]["marker"] == "x"
    assert image_axis.errorbars[0][2]["fmt"] == "o"
    assert image_axis.errorbars[0][2]["color"][3] < 1.0
    assert image_axis.errorbars[0][2]["ecolor"][3] < image_axis.errorbars[0][2]["color"][3]
    np.testing.assert_allclose(image_axis.scatters[0][2]["color"][:3], image_axis.errorbars[0][2]["color"][:3])
    np.testing.assert_allclose(image_axis.scatters[1][2]["color"][:3], image_axis.errorbars[1][2]["color"][:3])


def test_fit_quality_diagnostic_plots_write_pdfs_and_merge_tables(tmp_path: Path) -> None:
    image_df = pd.DataFrame(
        {
            "family_id": ["1", "1", "2"],
            "image_label": ["1.1", "1.2", "2.1"],
            "x_obs_arcsec": [0.0, 2.0, 5.0],
            "y_obs_arcsec": [0.0, 1.0, -1.0],
            "z_source": [2.0, 2.0, 3.0],
            "radius_arcsec": [0.0, 2.2, 5.1],
            "angle_deg": [0.0, 26.6, -11.3],
            "image_residual_arcsec": [0.1, 0.3, 0.8],
            "image_residual_q50": [0.12, 0.35, 0.75],
            "residual_norm": [0.5, 1.2, 2.4],
            "residual_norm_q50": [0.6, 1.4, 2.2],
            "covered_x_1sigma": [True, True, False],
            "covered_y_1sigma": [True, False, False],
            "covered_xy_1sigma": [True, False, False],
        }
    )
    magnification_df = pd.DataFrame(
        {
            "family_id": ["1", "2", "3"],
            "image_label": ["1.1", "2.1", "3.1"],
            "magnification_model": [2.0, -5.0, 10.0],
            "magnification_model_q50": [2.2, -4.5, 9.5],
        }
    )
    merged = plotting._merge_fit_quality_with_magnification(image_df, magnification_df)
    assert merged["image_label"].tolist() == ["1.1", "2.1"]

    plotting._plot_normalized_image_residuals(image_df, tmp_path / "normalized_image_residuals.pdf")
    plotting._plot_residual_vs_magnification(image_df, magnification_df, tmp_path / "residual_vs_magnification.pdf")
    plotting._plot_residual_geometry_trends(image_df, tmp_path / "residual_geometry_trends.pdf")
    plotting._plot_posterior_predictive_coverage(image_df, tmp_path / "posterior_predictive_coverage.pdf")

    for filename in [
        "normalized_image_residuals.pdf",
        "residual_vs_magnification.pdf",
        "residual_geometry_trends.pdf",
        "posterior_predictive_coverage.pdf",
    ]:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_exact_vs_approx_prediction_error_skips_missing_rows(tmp_path: Path) -> None:
    family_df = pd.DataFrame(
        {
            "family_id": ["1", "2", "3"],
            "exact_image_rms_arcsec": [0.2, np.nan, 0.6],
            "approx_image_rms_arcsec": [0.1, 0.4, np.nan],
        }
    )

    plotting._plot_exact_vs_approx_prediction_error(family_df, tmp_path / "exact_vs_approx_prediction_error.pdf")

    path = tmp_path / "exact_vs_approx_prediction_error.pdf"
    assert path.exists()
    assert path.stat().st_size > 0


def test_ns_diagnostic_plots_write_pdfs(tmp_path: Path) -> None:
    n_samples = 24
    ns_diagnostics = {
        "log_L_samples": np.linspace(-20.0, -3.0, n_samples),
        "log_dp_mean": np.linspace(-12.0, -2.0, n_samples),
        "log_X_mean": -np.linspace(0.1, 8.0, n_samples),
        "num_live_points_per_sample": np.full(n_samples, 12),
        "num_likelihood_evaluations_per_sample": np.arange(1, n_samples + 1),
        "log_efficiency": np.asarray([-4.0]),
    }
    specs = [
        ParameterSpec(
            name="x",
            sample_name="x",
            potential_id="mock",
            profile_type=81,
            field="x",
            prior_kind="uniform",
            lower=-1.0,
            upper=1.0,
            step=0.1,
        ),
        ParameterSpec(
            name="y",
            sample_name="y",
            potential_id="mock",
            profile_type=81,
            field="y",
            prior_kind="uniform",
            lower=-1.0,
            upper=1.0,
            step=0.1,
        ),
    ]

    plotting._plot_ns_diagnostics(tmp_path, ns_diagnostics)
    plotting._plot_ns_trace(tmp_path, ns_diagnostics, specs)
    plotting._plot_ns_weight_diagnostics(tmp_path, ns_diagnostics)

    for filename in ["ns_diagnostics.pdf", "ns_weight_diagnostics.pdf"]:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0
    assert not (tmp_path / "ns_trace_plot.pdf").exists()


def test_ns_diagnostic_plots_skip_missing_inputs(tmp_path: Path) -> None:
    plotting._plot_ns_diagnostics(tmp_path, {})
    plotting._plot_ns_trace(tmp_path, {"samples": np.empty((0, 0))}, [])
    plotting._plot_ns_weight_diagnostics(tmp_path, {"log_dp_mean": np.asarray([])})

    assert not any(tmp_path.iterdir())
