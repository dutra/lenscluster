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
from astropy.io import fits
from astropy.wcs import WCS

import lenscluster.plotting as plotting
from lenscluster.plotting import plot_path
from lenscluster.model import PosteriorResults, ParameterSpec


def test_plot_path_creates_directory(tmp_path: Path) -> None:
    output = plot_path(tmp_path / "plots", "summary.png")

    assert output == tmp_path / "plots" / "summary.pdf"
    assert output.parent.is_dir()


def test_image_catalog_family_cutout_stage_eligibility(tmp_path: Path) -> None:
    args = SimpleNamespace(
        image_catalog_family_cutout_image_dir=tmp_path / "images",
        exact_image_diagnostics_stage3=False,
    )

    assert plotting._image_catalog_family_cutout_enabled(
        args,
        tmp_path / "fit" / "stage4_critical_arc_mixture_image_plane",
    )
    assert not plotting._image_catalog_family_cutout_enabled(
        args,
        tmp_path / "fit" / "stage3_image_plane",
    )
    args.exact_image_diagnostics_stage3 = True
    assert plotting._image_catalog_family_cutout_enabled(
        args,
        tmp_path / "fit" / "stage3_image_plane",
    )
    args.image_catalog_family_cutout_image_dir = None
    assert not plotting._image_catalog_family_cutout_enabled(
        args,
        tmp_path / "fit" / "stage4_critical_arc_mixture_image_plane",
    )


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


def test_corner_uses_one_two_three_sigma_contour_levels(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []

    class FakeFig:
        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    class FakeCorner:
        def corner(self, samples: np.ndarray, **kwargs: Any) -> FakeFig:
            calls.append(("corner", np.asarray(samples), kwargs))
            return FakeFig()

        def overplot_points(self, _fig: FakeFig, _xs: list[list[float]], **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)

    plotting._plot_corner(
        tmp_path,
        np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]], dtype=float),
        _corner_test_specs(),
    )

    assert calls[0][0] == "corner"
    expected = [1.0 - math.exp(-0.5 * sigma**2) for sigma in (1.0, 2.0, 3.0)]
    np.testing.assert_allclose(plotting.CORNER_SIGMA_CONTOUR_LEVELS, expected)
    np.testing.assert_allclose(calls[0][2]["levels"], expected)


def _image_scatter_test_spec() -> ParameterSpec:
    return ParameterSpec(
        name="image.sigma_int",
        sample_name="image_sigma_int",
        potential_id="image",
        profile_type=0,
        field="sigma_int",
        prior_kind="uniform",
        lower=0.0,
        upper=2.0,
        step=0.01,
        component_family="image_scatter",
    )


def _source_position_test_spec(index: int) -> ParameterSpec:
    return ParameterSpec(
        name=f"source.{index}.beta_x",
        sample_name=f"source_{index}_beta_x",
        potential_id=str(index),
        profile_type=0,
        field="beta_x",
        prior_kind="uniform",
        lower=-10.0,
        upper=10.0,
        step=0.01,
        component_family="source_position",
    )


def _synthetic_stuck_chain_posterior() -> tuple[PosteriorResults, list[ParameterSpec]]:
    specs = [
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
            name="halo.y",
            sample_name="halo_y",
            potential_id="halo",
            profile_type=81,
            field="y",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family="large",
        ),
        _image_scatter_test_spec(),
    ]
    draws = np.linspace(-0.02, 0.02, 12)
    grouped = np.zeros((4, draws.size, len(specs)), dtype=float)
    grouped[:, :, 0] = np.asarray([0.0, 0.05, 0.08, 0.1], dtype=float)[:, None] + draws[None, :]
    grouped[:, :, 1] = np.asarray([0.0, 0.02, 0.03, 0.04], dtype=float)[:, None] + 0.5 * draws[None, :]
    grouped[0, :, 2] = 1.23 + 0.002 * np.linspace(-1.0, 1.0, draws.size)
    grouped[1, :, 2] = 0.10 + 0.01 * np.linspace(-1.0, 1.0, draws.size)
    grouped[2, :, 2] = 0.11 + 0.01 * np.linspace(-1.0, 1.0, draws.size)
    grouped[3, :, 2] = 0.09 + 0.01 * np.linspace(-1.0, 1.0, draws.size)
    grouped_log_prob = np.vstack(
        [
            np.full(draws.size, -230.0),
            np.linspace(-10.0, 5.0, draws.size),
            np.linspace(-8.0, 7.0, draws.size),
            np.linspace(0.0, 20.0, draws.size),
        ]
    )
    accept_prob = np.vstack(
        [
            np.full(draws.size, 0.95),
            np.full(draws.size, 0.98),
            np.full(draws.size, 0.97),
            np.full(draws.size, 0.90),
        ]
    )
    num_steps = np.vstack(
        [
            np.full(draws.size, 255.0),
            np.full(draws.size, 64.0),
            np.full(draws.size, 128.0),
            np.full(draws.size, 200.0),
        ]
    )
    posterior = PosteriorResults(
        samples=grouped.reshape((-1, grouped.shape[-1])),
        log_prob=grouped_log_prob.reshape(-1),
        accept_prob=accept_prob.reshape(-1),
        diverging=np.zeros(grouped.shape[0] * grouped.shape[1], dtype=bool),
        num_steps=num_steps.reshape(-1),
        warmup_steps=0,
        sample_steps=grouped.shape[1],
        num_chains=grouped.shape[0],
        init_diagnostics={"chain_seed_labels": ["stuck", "ok-1", "ok-2", "ok-3"]},
        grouped_samples=grouped,
        grouped_log_prob=grouped_log_prob,
    )
    return posterior, specs


def test_load_bayes_corner_overlay_maps_object_and_potfile_columns(tmp_path: Path, monkeypatch: Any) -> None:
    bayes_path = tmp_path / "bayes.dat"
    bayes_path.write_text(
        "\n".join(
            [
                "#Nsample",
                "#ln(Lhood)",
                "#O1 : x (arcsec)",
                "#O1 : rc (arcsec)",
                "#O2 : sigma (km/s)",
                "#Pot0 rcut (arcsec)",
                "#Pot0 sigma (km/s)",
                "#Chi2",
                "1 -10 1.5 2.0 300 4.0 200 9",
                "2 -11 1.7 2.5 310 5.0 220 8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state = SimpleNamespace(z_lens=0.4, cosmo_config={}, potfiles=[{"id": "potfile"}])
    monkeypatch.setattr(plotting, "_bayes_kpc_per_arcsec", lambda _state: 6.0)

    overlay = plotting._load_bayes_corner_overlay(bayes_path, state)

    assert overlay is not None
    np.testing.assert_allclose(overlay["1.x_centre"], [1.5, 1.7])
    np.testing.assert_allclose(overlay["1.core_radius_kpc"], [12.0, 15.0])
    np.testing.assert_allclose(overlay["2.v_disp"], [300.0, 310.0])
    np.testing.assert_allclose(overlay["potfile.cutkpc"], [24.0, 30.0])
    np.testing.assert_allclose(overlay["potfile.sigma"], [200.0, 220.0])
    assert "Chi2" not in overlay


def test_bayes_corner_overlay_uses_existing_corner_figure(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []
    figures: list[Any] = []
    specs = [
        ParameterSpec(
            name="potfile.cutkpc",
            sample_name="potfile_cutkpc",
            potential_id="potfile",
            profile_type=81,
            field="cutkpc",
            prior_kind="uniform",
            lower=1.0,
            upper=50.0,
            step=0.1,
            component_family="scaling",
        ),
        ParameterSpec(
            name="potfile.sigma",
            sample_name="potfile_sigma",
            potential_id="potfile",
            profile_type=81,
            field="sigma",
            prior_kind="uniform",
            lower=100.0,
            upper=400.0,
            step=1.0,
            component_family="scaling",
        ),
    ]

    class FakeFig:
        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    class FakeCorner:
        def corner(self, samples: np.ndarray, **kwargs: Any) -> FakeFig:
            fig = kwargs.get("fig") or FakeFig()
            figures.append(fig)
            calls.append(("corner", np.asarray(samples), kwargs))
            return fig

        def overplot_lines(self, _fig: FakeFig, xs: list[float | None], **kwargs: Any) -> None:
            calls.append(("lines", xs, kwargs))

        def overplot_points(self, _fig: FakeFig, xs: list[list[float]], **kwargs: Any) -> None:
            calls.append(("points", xs, kwargs))

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)

    plotting._plot_potfile_corner(
        tmp_path,
        np.asarray([[20.0, 180.0], [25.0, 200.0], [30.0, 220.0]], dtype=float),
        specs,
        bayes_corner_overlay={
            "potfile.cutkpc": np.asarray([21.0, 26.0, 31.0], dtype=float),
            "potfile.sigma": np.asarray([190.0, 210.0, 230.0], dtype=float),
        },
    )

    assert calls[0][0] == "corner"
    assert calls[1][0] == "corner"
    np.testing.assert_allclose(calls[0][2]["weights"], np.full(3, 1.0 / 3.0))
    assert np.sum(calls[0][2]["weights"]) == pytest.approx(1.0)
    np.testing.assert_allclose(calls[1][1], np.asarray([[21.0, 190.0], [26.0, 210.0], [31.0, 230.0]]))
    np.testing.assert_allclose(calls[1][2]["weights"], np.full(3, 1.0 / 3.0))
    assert np.sum(calls[1][2]["weights"]) == pytest.approx(1.0)
    assert calls[1][2]["fig"] is figures[0]
    assert calls[1][2]["color"] == plotting.CORNER_BAYES_OVERLAY_COLOR
    assert calls[1][2]["fill_contours"] is False
    assert calls[1][2]["no_fill_contours"] is True
    assert "contourf_kwargs" not in calls[1][2]
    assert calls[1][2]["plot_datapoints"] is False


def test_load_best_par_marker_values_maps_large_and_potfile_values(tmp_path: Path) -> None:
    lum2 = 10.0 ** (-0.4)
    best_path = tmp_path / "best.par"
    best_path.write_text(
        f"""
runmode
    reference 3 342.0 -44.0
    end
cosmology
    H0 70
    omega 0.3
    lambda 0.7
    end
potentiel O1
    profil 81
    x_centre 1.5
    y_centre -0.5
    ellipticite 0.6
    angle_pos -40
    core_radius 10
    core_radius_kpc 50
    cut_radius 2000
    cut_radius_kpc 10000
    v_disp 1100
    z_lens 0.35
    end
potentiel 101
    profil 81
    x_centre 0
    y_centre 0
    ellipticite 0
    angle_pos 0
    core_radius 0.1
    core_radius_kpc 2
    cut_radius 10
    cut_radius_kpc 50
    v_disp 300
    z_lens 0.35
    end
potentiel 102
    profil 81
    x_centre 0
    y_centre 0
    ellipticite 0
    angle_pos 0
    core_radius 0.1
    core_radius_kpc {2 * np.sqrt(lum2):.12g}
    cut_radius 10
    cut_radius_kpc {50 * lum2:.12g}
    v_disp {300 * lum2 ** 0.25:.12g}
    z_lens 0.35
    end
fini
""",
        encoding="utf-8",
    )
    state = SimpleNamespace(
        par_path=None,
        potfiles=[
            {
                "id": "potfile",
                "mag0": 20.0,
                "vdslope_nominal": 4.0,
                "slope_nominal": 2.0,
                "catalog_df": pd.DataFrame(
                    {
                        "id": ["101", "102"],
                        "catalog_mag": [20.0, 21.0],
                    }
                ),
            }
        ],
    )

    values = plotting._load_best_par_marker_values(best_path, state)

    assert values is not None
    assert values["1.x_centre"] == pytest.approx(1.5)
    assert values["1.core_radius_kpc"] == pytest.approx(50.0)
    assert values["1.v_disp"] == pytest.approx(1100.0)
    assert values["potfile.sigma"] == pytest.approx(300.0)
    assert values["potfile.cutkpc"] == pytest.approx(50.0)
    assert values["potfile.corekpc"] == pytest.approx(2.0)


def test_best_par_marker_draws_without_fit_markers(tmp_path: Path, monkeypatch: Any) -> None:
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
        best_fit_values=None,
        previous_stage_best_values=None,
        best_par_marker_values={"x": 1.5, "y": 3.5},
    )

    assert calls[0][0] == "corner"
    assert calls[1] == (
        "points",
        [[1.5, 3.5]],
        {
            "marker": "x",
            "color": plotting.CORNER_BEST_PAR_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert len(calls) == 2


def test_per_potential_summary_uses_corner_marker_colors_and_limits(tmp_path: Path, monkeypatch: Any) -> None:
    class FakeAxis:
        def __init__(self) -> None:
            self.hlines_calls: list[tuple[Any, Any, Any, dict[str, Any]]] = []
            self.scatters: list[tuple[Any, Any, dict[str, Any]]] = []
            self.xlim: tuple[float, float] | None = None
            self.title: str | None = None

        def hlines(self, y: Any, xmin: Any, xmax: Any, **kwargs: Any) -> None:
            self.hlines_calls.append((y, xmin, xmax, kwargs))

        def scatter(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.scatters.append((x, y, kwargs))

        def set_xlim(self, x_min: float, x_max: float) -> None:
            self.xlim = (x_min, x_max)

        def set_yticks(self, _ticks: list[Any]) -> None:
            return None

        def set_title(self, title: str) -> None:
            self.title = title

        def get_legend_handles_labels(self) -> tuple[list[str], list[str]]:
            labels = [str(kwargs["label"]) for _x, _y, kwargs in self.scatters if "label" in kwargs]
            return labels, labels

    class FakeFig:
        def __init__(self) -> None:
            self.legend_calls: list[tuple[Any, Any, dict[str, Any]]] = []
            self.saved_path: Path | None = None

        def legend(self, handles: Any, labels: Any, **kwargs: Any) -> None:
            self.legend_calls.append((handles, labels, kwargs))

        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            self.saved_path = Path(path)
            self.saved_path.write_text("pdf", encoding="utf-8")

    axis = FakeAxis()
    fig = FakeFig()
    monkeypatch.setattr(plotting.plt, "subplots", lambda *_args, **_kwargs: (fig, axis))
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)
    summary_df = pd.DataFrame(
        [
            {
                "label": "potfile.sigma",
                "p16": 2.0,
                "p84": 8.0,
                "median": 5.0,
                "map": 6.0,
                "lower": 0.0,
                "upper": 10.0,
                "std": 1.0,
            }
        ]
    )

    plotting._plot_per_potential_summary(
        tmp_path,
        summary_df,
        best_par_marker_values={"potfile.sigma": 20.0},
        previous_stage_best_values={"potfile_sigma": -5.0},
        parameter_specs=[
            ParameterSpec(
                name="potfile.sigma",
                sample_name="potfile_sigma",
                potential_id="potfile",
                profile_type=81,
                field="sigma",
                prior_kind="uniform",
                lower=0.0,
                upper=10.0,
                step=1.0,
                component_family="scaling",
            )
        ],
    )

    assert axis.hlines_calls == [(1, 2.0, 8.0, {"linewidth": 4, "color": "tab:blue"})]
    assert axis.scatters[0] == ([5.0], [1], {"color": "tab:blue", "s": 35, "label": "median"})
    assert axis.scatters[1] == (
        [6.0],
        [1],
        {"color": plotting.CORNER_BEST_FIT_COLOR, "marker": "x", "s": 30, "label": "best fit"},
    )
    assert axis.scatters[2] == (
        [20.0],
        [1],
        {"color": plotting.CORNER_BEST_PAR_COLOR, "marker": "x", "s": 30, "label": "best.par"},
    )
    assert axis.scatters[3] == (
        [-5.0],
        [1],
        {
            "color": plotting.CORNER_PREVIOUS_STAGE_COLOR,
            "marker": "x",
            "s": 30,
            "label": "previous stage",
        },
    )
    assert axis.xlim is not None
    assert axis.xlim[0] < -5.0
    assert axis.xlim[1] > 20.0
    assert axis.title == "potfile.sigma"
    assert fig.saved_path == tmp_path / "per_potential_summary.pdf"


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
    assert calls[1] == (
        "points",
        [[1.25, 3.25]],
        {
            "marker": "x",
            "color": plotting.CORNER_PREVIOUS_STAGE_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert calls[2] == (
        "points",
        [[1.5, 3.5]],
        {
            "marker": "x",
            "color": plotting.CORNER_BEST_FIT_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
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
    assert calls[1] == (
        "points",
        [[1.5, 3.5]],
        {
            "marker": "x",
            "color": plotting.CORNER_BEST_FIT_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
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
    assert calls[1] == (
        "points",
        [[11.5, 22.0]],
        {
            "marker": "x",
            "color": plotting.CORNER_PREVIOUS_STAGE_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert calls[2] == (
        "points",
        [[12.0, 23.0]],
        {
            "marker": "x",
            "color": plotting.CORNER_BEST_FIT_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
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
    assert calls[1] == (
        "points",
        [[0.29, -1.05]],
        {
            "marker": "x",
            "color": plotting.CORNER_PREVIOUS_STAGE_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert calls[2] == (
        "points",
        [[0.31, -0.95]],
        {
            "marker": "x",
            "color": plotting.CORNER_BEST_FIT_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )


def test_fit_quality_tables_cap_draws_convert_physical_and_quantile(monkeypatch: Any) -> None:
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 1)
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

    image_df, magnification_df, extra_image_df = plotting._fit_quality_tables(
        state,
        evaluator,
        np.asarray([5.0], dtype=float),
        results,
        argparse.Namespace(fit_quality_draws=2),
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
    assert row["image_recovery_status"] == "recovered"
    assert extra_image_df.empty

    mag_row = magnification_df.set_index("image_label").loc["1.2"]
    assert mag_row["magnification_model"] == pytest.approx(9.0)
    assert mag_row["magnification_model_q50"] == pytest.approx(19.0)
    assert int(mag_row["posterior_valid_draws"]) == 2


def test_fit_quality_tables_defaults_to_best_fit_only(monkeypatch: Any) -> None:
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 1)
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

    image_df, magnification_df, extra_image_df = plotting._fit_quality_tables(
        state,
        evaluator,
        np.asarray([5.0], dtype=float),
        results,
        argparse.Namespace(),
    )

    assert evaluator.converted == [5.0]
    assert evaluator.exact_latents == [6.0]

    image_row = image_df.set_index("image_label").loc["1.1"]
    assert image_row["x_model_arcsec"] == pytest.approx(6.0)
    assert np.isnan(image_row["x_model_q50"])
    assert np.isnan(image_row["image_residual_q50"])
    assert int(image_row["posterior_valid_draws"]) == 0
    assert image_row["image_recovery_status"] == "recovered"
    assert extra_image_df.empty

    mag_row = magnification_df.set_index("image_label").loc["1.1"]
    assert mag_row["magnification_model"] == pytest.approx(6.0)
    assert np.isnan(mag_row["magnification_model_q50"])
    assert int(mag_row["posterior_valid_draws"]) == 0


def test_fit_quality_tables_quick_diagnostics_skips_exact_and_uses_median_std(monkeypatch: Any) -> None:
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 1)
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

    image_df, magnification_df, extra_image_df = plotting._fit_quality_tables(
        state,
        evaluator,
        np.asarray([10.0], dtype=float),
        results,
        argparse.Namespace(fit_quality_draws=3, quick_diagnostics=True),
    )

    assert evaluator.exact_calls == 0
    image_row = image_df.set_index("image_label").loc["1.1"]
    assert bool(image_row["exact_image_prediction_failed"]) is True
    assert np.isnan(image_row["x_model_arcsec"])
    assert np.isnan(image_row["image_residual_arcsec"])
    assert np.isnan(image_row["model_produced_image_count"])
    assert image_row["model_multiplicity_failure_reason"] == "quick_diagnostics"
    assert image_row["image_recovery_status"] == "unknown"
    assert extra_image_df.empty

    mag_row = magnification_df.set_index("image_label").loc["1.1"]
    mag_values = np.asarray([3.0, 5.0, 7.0], dtype=float)
    assert mag_row["magnification_model"] == pytest.approx(13.0)
    assert mag_row["magnification_model_q16"] == pytest.approx(float(np.median(mag_values) - np.std(mag_values)))
    assert mag_row["magnification_model_q50"] == pytest.approx(float(np.median(mag_values)))
    assert mag_row["magnification_model_q84"] == pytest.approx(float(np.median(mag_values) + np.std(mag_values)))


def test_fit_quality_tables_tracks_partial_recovery_and_extra_images(monkeypatch: Any) -> None:
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 1)
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=1.0,
        n_images=3,
        image_labels=["1.1", "1.2", "1.3"],
        x_obs=np.asarray([0.0, 1.0, 2.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0, 0.0], dtype=float),
    )
    state = SimpleNamespace(parameter_specs=[], family_data=[family])

    class FakeModel:
        def magnification(self, x: Any, y: Any, _kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            return np.asarray(x, dtype=float) + np.asarray(y, dtype=float)

    class FakeEvaluator:
        source_plane_covariance_floor = 0.0

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float)

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 0.0

        def _exact_family_prediction_details(self, _sample_latent: np.ndarray, _family_data: Any) -> dict[str, Any]:
            return {
                "produced_image_count": 3,
                "recovered_image_count": 2,
                "missing_image_count": 1,
                "extra_image_count": 1,
                "multiplicity_failed": True,
                "multiplicity_failure_reason": "match_tolerance_exceeded",
                "matched_model_x_arcsec": np.asarray([0.05, np.nan, 2.05], dtype=float),
                "matched_model_y_arcsec": np.asarray([0.0, np.nan, 0.0], dtype=float),
                "recovered_image_mask": np.asarray([True, False, True], dtype=bool),
                "extra_model_x_arcsec": np.asarray([8.0], dtype=float),
                "extra_model_y_arcsec": np.asarray([-3.0], dtype=float),
                "failed": True,
            }

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, _sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {}

        def _packed_to_kwargs_lens(self, _packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [{}]

    image_df, _magnification_df, extra_image_df = plotting._fit_quality_tables(
        state,
        FakeEvaluator(),
        np.asarray([0.0], dtype=float),
        SimpleNamespace(samples=np.empty((0, 1), dtype=float)),
        argparse.Namespace(fit_quality_draws=0),
    )

    indexed = image_df.set_index("image_label")
    assert indexed["image_recovery_status"].tolist() == ["recovered", "not_recovered", "recovered"]
    assert indexed["exact_image_prediction_failed"].tolist() == [False, True, False]
    assert indexed.loc["1.1", "x_model_arcsec"] == pytest.approx(0.05)
    assert np.isnan(indexed.loc["1.2", "x_model_arcsec"])
    assert indexed.loc["1.3", "image_residual_arcsec"] == pytest.approx(0.05)

    assert extra_image_df["family_id"].tolist() == ["1"]
    assert extra_image_df.loc[0, "image_recovery_status"] == "extra"
    assert extra_image_df.loc[0, "x_model_arcsec"] == pytest.approx(8.0)
    assert extra_image_df.loc[0, "y_model_arcsec"] == pytest.approx(-3.0)
    assert int(extra_image_df.loc[0, "model_extra_image_count"]) == 1


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
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 4)
    monkeypatch.setattr(plotting, "_log", lambda _args, message: log_messages.append(str(message)))

    predictions = plotting._posterior_fit_quality_predictions(
        evaluator,
        state,
        [np.asarray([1.0], dtype=float), np.asarray([2.0], dtype=float)],
        argparse.Namespace(),
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
        "fit quality exact: 6/6 family diagnostics | completed draw 2/2 family=fam-0 z=2.0000 window=10.0 grid=50x50",
    ) in progress.descriptions
    assert log_messages == [
        (
            "[plot:fit_quality] family diagnostics tasks=6 workers=4 families=3 draws=2 "
            "largest_grid=150x150 total_grid_points=70000"
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
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 1)
    log_messages: list[str] = []
    monkeypatch.setattr(plotting, "_log", lambda _args, message: log_messages.append(str(message)))

    predictions = plotting._posterior_fit_quality_predictions(
        FakeEvaluator(),
        state,
        [np.asarray([1.0], dtype=float), np.asarray([2.0], dtype=float)],
        argparse.Namespace(),
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
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 1)

    predictions = plotting._posterior_fit_quality_predictions(
        FakeEvaluator(),
        state,
        [np.asarray([1.0], dtype=float)],
        argparse.Namespace(quiet=True),
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
    extra_image_df = pd.DataFrame(
        {
            "family_id": ["1"],
            "extra_image_index": [1],
            "image_recovery_status": ["extra"],
            "x_model_arcsec": [3.0],
            "y_model_arcsec": [4.0],
        }
    )
    captured_tasks: list[str] = []

    monkeypatch.setattr(plotting, "_summary_table", lambda *_args, **_kwargs: pd.DataFrame({"label": ["mock"]}))
    monkeypatch.setattr(plotting, "_family_diagnostics_table", lambda *_args, **_kwargs: pd.DataFrame({"family_id": ["1"]}))
    monkeypatch.setattr(plotting, "_fit_quality_tables", lambda *_args, **_kwargs: (image_df, magnification_df, extra_image_df))
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
    assert (tmp_path / "tables" / "image_recovery_extra_images.csv").exists()
    assert (tmp_path / "tables" / "model_magnification.csv").exists()
    assert (tmp_path / "tables" / "subhalo_properties.csv").exists()
    assert (tmp_path / "tables" / "run_summary.txt").exists()
    assert "image_recovery" in captured_tasks
    assert "image_count_recovery" in captured_tasks
    assert "model_magnification" in captured_tasks
    assert "normalized_image_residuals" in captured_tasks
    assert "image_residual_histogram" in captured_tasks
    assert "residual_vs_magnification" in captured_tasks
    assert "residual_geometry_trends" in captured_tasks
    assert "posterior_predictive_coverage" in captured_tasks
    assert "subhalo_mass_function" in captured_tasks
    assert "subhalo_radial_distribution" in captured_tasks
    assert "kappa_comparison" not in captured_tasks
    assert "absolute_magnification" not in captured_tasks
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
        args=argparse.Namespace(
            quiet=True,
            plot_caustics=False,
            caustic_plot_grid_scale_arcsec=0.2,
            caustic_source_redshift=9.0,
        ),
    )
    assert "absolute_magnification" not in captured_tasks
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
        args=argparse.Namespace(
            quiet=True,
            plot_caustics=False,
            kappa_true_fits="data/ff_sims/hera/kappa_z9_0.fits",
            caustic_source_redshift=9.0,
        ),
    )
    assert "kappa_comparison" in captured_tasks
    assert "absolute_magnification" not in captured_tasks
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
        args=argparse.Namespace(
            quiet=True,
            plot_caustics=True,
            caustic_plot_grid_scale_arcsec=0.2,
            caustic_source_redshift=9.0,
        ),
    )
    assert "absolute_magnification" in captured_tasks
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
            caustic_plot_grid_scale_arcsec=0.2,
            caustic_source_redshift=9.0,
            kappa_true_fits="data/ff_sims/hera/kappa_z9_0.fits",
        ),
    )
    assert "kappa_comparison" not in captured_tasks
    assert "absolute_magnification" not in captured_tasks
    assert "caustic_overlay" not in captured_tasks


def test_subhalo_properties_table_uses_all_potfile_members_and_mass_radii(monkeypatch: pytest.MonkeyPatch) -> None:
    state = SimpleNamespace(
        z_lens=0.4,
        cosmo_config={},
        parameter_specs=[],
        packed_lens_spec=SimpleNamespace(
            component_family=np.asarray([0, 1, 1], dtype=int),
            x_center_base=np.asarray([0.0, 3.0, 6.0], dtype=float),
            y_center_base=np.asarray([0.0, 4.0, 8.0], dtype=float),
        ),
        scaling_component_records=[
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "component_index": 1,
                "catalog_id": "member001",
                "catalog_mag": 20.0,
                "x_centre": 3.0,
                "y_centre": 4.0,
            },
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "component_index": 2,
                "catalog_id": "member002",
                "catalog_mag": 21.0,
                "x_centre": 6.0,
                "y_centre": 8.0,
            },
        ],
    )

    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[float, int]] = []

        def mass_3d(self, radius: float, kwargs_lens: list[dict[str, float]], bool_list: list[bool]) -> float:
            component_index = bool_list.index(True)
            self.calls.append((float(radius), component_index))
            return float(kwargs_lens[component_index]["sigma0"]) * float(radius)

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, FakeModel] = {}
            self.model = FakeModel()
            self.converted: list[np.ndarray] = []
            self.model_z: list[float] = []
            self.packed_z: list[float] = []

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            theta_array = np.asarray(theta, dtype=float)
            self.converted.append(theta_array.copy())
            return theta_array + 1.0

        def _get_exact_model_solver(self, z_source: float) -> tuple[FakeModel, None]:
            self.model_z.append(float(z_source))
            return self.model, None

        def _build_packed_lens_state(self, sample_latent: Any, z_source: float) -> dict[str, float]:
            self.packed_z.append(float(z_source))
            return {"latent": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            assert packed_state == {"latent": 5.0}
            return [
                {"sigma0": 1.0, "Ra": 0.1, "Rs": 1.0},
                {"sigma0": 3.0, "Ra": 0.2, "Rs": 2.0},
                {"sigma0": 5.0, "Ra": 0.3, "Rs": 4.0},
            ]

    monkeypatch.setattr(plotting, "critical_surface_density_angle_from_config", lambda *_args, **_kwargs: 10.0)
    evaluator = FakeEvaluator()

    table = plotting._subhalo_properties_table(
        state,
        evaluator,
        np.asarray([4.0], dtype=float),
        caustic_source_redshift=9.0,
    )

    assert table["component_index"].tolist() == [1, 2]
    assert table["catalog_id"].tolist() == ["member001", "member002"]
    assert table["radius_arcsec"].tolist() == pytest.approx([5.0, 10.0])
    assert table["Rs"].tolist() == pytest.approx([2.0, 4.0])
    assert table["mass_within_Rs_msun"].tolist() == pytest.approx([60.0, 200.0])
    assert table["mass_within_1e6_Rs_msun"].tolist() == pytest.approx([60.0e6, 200.0e6])
    assert [float(item[0]) for item in evaluator.converted] == [4.0]
    assert evaluator.model_z == [9.0]
    assert evaluator.packed_z == [9.0]
    assert evaluator.model.calls == [
        (2.0, 1),
        (2.0e6, 1),
        (4.0, 2),
        (4.0e6, 2),
    ]


def test_subhalo_distribution_plots_write_pdfs_and_mass_function_has_subhalo_mass_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    subhalo_df = pd.DataFrame(
        {
            "mass_within_Rs_msun": [1.0e10, 3.0e10, 1.0e11],
            "mass_within_1e6_Rs_msun": [2.0e11, 4.0e11, 1.2e12],
            "radius_arcsec": [5.0, 15.0, 30.0],
        }
    )
    hist_calls: list[dict[str, Any]] = []
    original_hist = Axes.hist

    def record_hist(self: Axes, values: Any, *args: Any, **kwargs: Any) -> Any:
        hist_calls.append(
            {
                "values": np.asarray(values, dtype=float).copy(),
                "label": kwargs.get("label"),
                "histtype": kwargs.get("histtype"),
            }
        )
        return original_hist(self, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", record_hist)

    mass_path = tmp_path / "subhalo_mass_function.pdf"
    radial_path = tmp_path / "subhalo_radial_distribution.pdf"
    plotting._plot_subhalo_mass_function(subhalo_df, mass_path)
    plotting._plot_subhalo_radial_distribution(subhalo_df, radial_path)

    assert mass_path.exists()
    assert mass_path.stat().st_size > 0
    assert radial_path.exists()
    assert radial_path.stat().st_size > 0
    mass_hist_calls = [call for call in hist_calls if call["label"] is not None]
    assert [call["label"] for call in mass_hist_calls] == ["Subhalo Mass"]
    assert [call["histtype"] for call in mass_hist_calls] == ["step"]
    np.testing.assert_allclose(mass_hist_calls[0]["values"], np.log10(subhalo_df["mass_within_1e6_Rs_msun"]))


def test_subhalo_distribution_plots_write_placeholders_without_finite_values(tmp_path: Path) -> None:
    subhalo_df = pd.DataFrame(
        {
            "mass_within_Rs_msun": [np.nan, np.inf],
            "mass_within_1e6_Rs_msun": [np.nan, -np.inf],
            "radius_arcsec": [np.nan, np.inf],
        }
    )
    mass_path = tmp_path / "subhalo_mass_function.pdf"
    radial_path = tmp_path / "subhalo_radial_distribution.pdf"

    plotting._plot_subhalo_mass_function(subhalo_df, mass_path)
    plotting._plot_subhalo_radial_distribution(subhalo_df, radial_path)

    assert mass_path.exists()
    assert mass_path.stat().st_size > 0
    assert radial_path.exists()
    assert radial_path.stat().st_size > 0


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
            "sigma_arcsec": [1.0, 1.0, 1.0, 1.0],
            "image_sigma_eff_arcsec": [10.0, 10.0, 10.0, 10.0],
            "image_recovery_status": ["recovered", "recovered", "recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "point_recovered", "point_recovered", "not_recovered"],
            "arc_supported": [False, False, False, False],
            "exact_image_prediction_failed": [False, False, False, True],
            "covered_xy_1sigma": [True, False, True, False],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state)

    assert summary["headline_chi_square"] == pytest.approx(0.07)
    assert summary["headline_point_image_count"] == 3
    assert summary["headline_missing_image_count"] == 1
    assert summary["headline_n_data"] == 6
    assert summary["headline_dof"] == 3
    assert summary["headline_reduced_chi_square"] == pytest.approx(0.07 / 3.0)
    assert summary["arc_aware_chi_square"] == pytest.approx(0.07)
    assert summary["arc_aware_point_image_count"] == 3
    assert summary["arc_aware_arc_supported_image_count"] == 0
    assert summary["arc_aware_missing_image_count"] == 1
    assert summary["arc_aware_n_data"] == 6
    assert summary["arc_aware_dof"] == 3
    assert summary["arc_aware_reduced_chi_square"] == pytest.approx(0.07 / 3.0)
    assert summary["n_effective_parameters"] == 3
    assert summary["sampled_non_source_position_parameters"] == 1
    assert summary["source_position_parameters"] == 2
    assert summary["chi_square_sigma_basis"] == "image_sigma_eff_arcsec"
    assert summary["chi_square_sigma_eff_median_arcsec"] == pytest.approx(10.0)
    assert summary["chi_square_sigma_eff_min_arcsec"] == pytest.approx(10.0)
    assert summary["chi_square_sigma_eff_max_arcsec"] == pytest.approx(10.0)
    assert summary["headline_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(7.0 / 3.0))
    assert summary["arc_aware_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(7.0 / 3.0))
    assert summary["headline_chi_square_red1_pos_sigma_arcsec"] is None
    assert summary["arc_aware_chi_square_red1_pos_sigma_arcsec"] is None
    assert summary["chi_square_red1_calibration_note"] == "post-fit diagnostic; holds image_sigma_int fixed"
    assert summary["covered_xy_1sigma_fraction"] == pytest.approx(2.0 / 3.0)
    assert "chi_square" not in summary
    assert "reduced_chi_square" not in summary
    assert "diagnostic_n_data" not in summary
    assert "diagnostic_dof" not in summary
    assert "aic" not in summary
    assert "bic" not in summary


def test_run_summary_arc_aware_chi_square_counts_arc_supported_rows() -> None:
    state = SimpleNamespace(
        parameter_specs=[ParameterSpec("p", "p", "mock", 81, "x", "uniform", -1.0, 1.0, 0.1)],
        family_data=[SimpleNamespace(family_id="1", n_images=4)],
    )
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0, 0.0, 0.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0, 0.0, 0.0],
            "x_model_arcsec": [1.0, 0.0, np.nan, np.nan],
            "y_model_arcsec": [0.0, 2.0, np.nan, np.nan],
            "sigma_arcsec": [1.0, 2.0, 0.5, 1.0],
            "image_sigma_eff_arcsec": [1.0, 4.0, 0.25, 100.0],
            "image_recovery_status": ["recovered", "recovered", "not_recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "point_recovered", "arc_supported", "not_recovered"],
            "arc_supported": [False, False, True, False],
            "arc_aware_image_residual_arcsec": [1.0, 2.0, 0.5, np.nan],
            "exact_image_prediction_failed": [False, False, True, True],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state)

    assert summary["headline_chi_square"] == pytest.approx(1.25)
    assert summary["headline_n_data"] == 4
    assert summary["headline_dof"] == 1
    assert summary["headline_reduced_chi_square"] == pytest.approx(1.25)
    assert summary["headline_missing_image_count"] == 2
    assert summary["arc_aware_chi_square"] == pytest.approx(5.25)
    assert summary["arc_aware_n_data"] == 5
    assert summary["arc_aware_dof"] == 2
    assert summary["arc_aware_reduced_chi_square"] == pytest.approx(2.625)
    assert summary["arc_aware_point_image_count"] == 2
    assert summary["arc_aware_arc_supported_image_count"] == 1
    assert summary["arc_aware_missing_image_count"] == 1
    assert summary["chi_square_sigma_eff_median_arcsec"] == pytest.approx(1.0)
    assert summary["chi_square_sigma_eff_min_arcsec"] == pytest.approx(0.25)
    assert summary["chi_square_sigma_eff_max_arcsec"] == pytest.approx(4.0)
    assert summary["headline_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(5.0))
    assert summary["arc_aware_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(5.25 / 2.0))


def test_run_summary_chi_square_requires_image_sigma_eff_arcsec() -> None:
    state = SimpleNamespace(
        parameter_specs=[ParameterSpec("p", "p", "mock", 81, "x", "uniform", -1.0, 1.0, 0.1)],
        family_data=[SimpleNamespace(family_id="1", n_images=1)],
    )
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0],
            "y_obs_arcsec": [0.0],
            "x_model_arcsec": [1.0],
            "y_model_arcsec": [0.0],
            "sigma_arcsec": [1.0],
            "image_recovery_status": ["recovered"],
            "arc_recovery_status": ["point_recovered"],
            "exact_image_prediction_failed": [False],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state)

    assert summary["headline_chi_square"] is None
    assert summary["headline_n_data"] == 0
    assert summary["arc_aware_chi_square"] is None
    assert summary["arc_aware_n_data"] == 0
    assert summary["chi_square_sigma_basis"] == "image_sigma_eff_arcsec"
    assert summary["headline_chi_square_red1_total_sigma_arcsec"] is None
    assert summary["headline_chi_square_red1_pos_sigma_arcsec"] is None
    assert summary["arc_aware_chi_square_red1_total_sigma_arcsec"] is None
    assert summary["arc_aware_chi_square_red1_pos_sigma_arcsec"] is None


def test_run_summary_chi_square_excludes_invalid_image_sigma_eff_rows() -> None:
    state = SimpleNamespace(
        parameter_specs=[ParameterSpec("p", "p", "mock", 81, "x", "uniform", -1.0, 1.0, 0.1)],
        family_data=[SimpleNamespace(family_id="1", n_images=3)],
    )
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0, 0.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0, 0.0],
            "x_model_arcsec": [1.0, 2.0, np.nan],
            "y_model_arcsec": [0.0, 0.0, np.nan],
            "sigma_arcsec": [0.1, 0.1, 0.1],
            "image_sigma_eff_arcsec": [0.5, 0.0, np.nan],
            "image_recovery_status": ["recovered", "recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "point_recovered", "arc_supported"],
            "arc_supported": [False, False, True],
            "arc_aware_image_residual_arcsec": [1.0, 2.0, 0.5],
            "exact_image_prediction_failed": [False, False, True],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state)

    assert summary["headline_chi_square"] == pytest.approx(4.0)
    assert summary["headline_point_image_count"] == 1
    assert summary["headline_missing_image_count"] == 2
    assert summary["headline_n_data"] == 2
    assert summary["arc_aware_chi_square"] == pytest.approx(4.0)
    assert summary["arc_aware_arc_supported_image_count"] == 0
    assert summary["arc_aware_missing_image_count"] == 2
    assert summary["arc_aware_n_data"] == 2
    assert summary["headline_chi_square_red1_total_sigma_arcsec"] is None
    assert summary["headline_chi_square_red1_pos_sigma_arcsec"] is None


def test_run_summary_chi_square_red1_calibration_solves_pos_sigma_with_intrinsic_scatter() -> None:
    state = SimpleNamespace(parameter_specs=[], family_data=[])
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0],
            "x_model_arcsec": [2.0, np.nan],
            "y_model_arcsec": [0.0, np.nan],
            "sigma_arcsec": [1.0, 1.0],
            "image_sigma_int_arcsec": [1.0, 1.0],
            "image_sigma_eff_arcsec": [math.sqrt(2.0), math.sqrt(2.0)],
            "image_recovery_status": ["recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "arc_supported"],
            "arc_supported": [False, True],
            "arc_aware_image_residual_arcsec": [2.0, math.sqrt(5.0)],
            "exact_image_prediction_failed": [False, True],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state)

    assert summary["headline_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(2.0))
    assert summary["headline_chi_square_red1_pos_sigma_arcsec"] == pytest.approx(1.0)
    assert summary["arc_aware_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(3.0))
    assert summary["arc_aware_chi_square_red1_pos_sigma_arcsec"] == pytest.approx(math.sqrt(2.0))


def test_run_summary_chi_square_red1_pos_sigma_zero_when_intrinsic_scatter_is_sufficient() -> None:
    state = SimpleNamespace(parameter_specs=[], family_data=[])
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0],
            "y_obs_arcsec": [0.0],
            "x_model_arcsec": [1.0],
            "y_model_arcsec": [0.0],
            "sigma_arcsec": [0.1],
            "image_sigma_int_arcsec": [1.0],
            "image_sigma_eff_arcsec": [math.sqrt(1.01)],
            "image_recovery_status": ["recovered"],
            "arc_recovery_status": ["point_recovered"],
            "arc_supported": [False],
            "exact_image_prediction_failed": [False],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state)

    assert summary["headline_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(0.5))
    assert summary["headline_chi_square_red1_pos_sigma_arcsec"] == pytest.approx(0.0)
    assert summary["arc_aware_chi_square_red1_pos_sigma_arcsec"] == pytest.approx(0.0)


def test_run_summary_effective_parameter_count_does_not_double_count_explicit_sources() -> None:
    specs = [
        ParameterSpec("halo.x", "halo_x", "mock", 81, "x", "uniform", -1.0, 1.0, 0.1),
        ParameterSpec(
            "source.1.beta_x",
            "source_1_beta_x",
            "1",
            0,
            "beta_x",
            "uniform",
            -1.0,
            1.0,
            0.1,
            component_family="source_position",
        ),
        ParameterSpec(
            "source.1.beta_y",
            "source_1_beta_y",
            "1",
            0,
            "beta_y",
            "uniform",
            -1.0,
            1.0,
            0.1,
            component_family="source_position",
        ),
        ParameterSpec(
            "image.sigma_int",
            "image_sigma_int",
            "image",
            0,
            "sigma_int",
            "uniform",
            0.0,
            1.0,
            0.1,
            component_family="image_scatter",
        ),
    ]
    state = SimpleNamespace(
        parameter_specs=specs,
        family_data=[
            SimpleNamespace(family_id="1", n_images=0),
            SimpleNamespace(family_id="2", n_images=0),
        ],
    )

    summary = plotting._fit_quality_chi_square_summary(None, state)

    assert summary["sampled_non_source_position_parameters"] == 2
    assert summary["source_position_parameters"] == 4
    assert summary["n_effective_parameters"] == 6


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


def test_chain_health_summary_table_identifies_stuck_chain() -> None:
    posterior, specs = _synthetic_stuck_chain_posterior()

    table = plotting._chain_health_summary_table(posterior, specs, max_tree_depth=8)

    assert list(table["chain"]) == [1, 2, 3, 4]
    stuck = table.iloc[0]
    assert stuck["chain_label"] == "stuck"
    assert stuck["log_prob_median"] == pytest.approx(-230.0)
    assert stuck["log_prob_median"] < table.iloc[1]["log_prob_median"]
    assert stuck["max_tree_depth_saturation_fraction"] == pytest.approx(1.0)
    assert stuck["image_sigma_int_q50"] == pytest.approx(1.23, abs=0.01)
    assert table.iloc[1]["image_sigma_int_q50"] == pytest.approx(0.10, abs=0.01)


def test_chain_parameter_diagnostics_table_reports_per_chain_quantiles() -> None:
    posterior, specs = _synthetic_stuck_chain_posterior()

    table = plotting._chain_parameter_diagnostics_table(posterior, specs)
    sigma_row = table.loc[table["sample_name"] == "image_sigma_int"].iloc[0]

    assert sigma_row["parameter"] == "image.sigma_int"
    assert sigma_row["chain_1_q50"] == pytest.approx(1.23, abs=0.01)
    assert sigma_row["chain_2_q50"] == pytest.approx(0.10, abs=0.01)
    assert sigma_row["chain_median_spread"] > 1.0
    assert sigma_row["chain_median_standardized_spread"] > 1.0


def test_ranked_chain_trace_subset_prefers_worst_non_source_parameters_when_sources_dominate() -> None:
    posterior, base_specs = _synthetic_stuck_chain_posterior()
    source_specs = [_source_position_test_spec(index) for index in range(8)]
    specs = [*base_specs, *source_specs]
    source_values = np.zeros((posterior.grouped_samples.shape[0], posterior.grouped_samples.shape[1], len(source_specs)))
    for source_index in range(len(source_specs)):
        source_values[:, :, source_index] = source_index + np.asarray([0.0, 0.2, 0.4, 0.6])[:, None]
    grouped = np.concatenate([posterior.grouped_samples, source_values], axis=2)
    table = plotting._chain_parameter_diagnostics_table(
        PosteriorResults(
            samples=grouped.reshape((-1, grouped.shape[-1])),
            log_prob=posterior.log_prob,
            accept_prob=posterior.accept_prob,
            diverging=posterior.diverging,
            num_steps=posterior.num_steps,
            warmup_steps=0,
            sample_steps=grouped.shape[1],
            num_chains=grouped.shape[0],
            grouped_samples=grouped,
            grouped_log_prob=posterior.grouped_log_prob,
        ),
        specs,
    )

    subset = plotting._ranked_chain_trace_subset(grouped, specs, table, max_params=3)

    assert subset is not None
    _subset_samples, subset_specs = subset
    assert "image_sigma_int" in {spec.sample_name for spec in subset_specs}
    assert all(spec.component_family != "source_position" for spec in subset_specs)


def test_chain_diagnostic_plots_create_pdfs_and_skip_missing_grouped_samples(tmp_path: Path) -> None:
    posterior, specs = _synthetic_stuck_chain_posterior()
    diagnostics = plotting._chain_parameter_diagnostics_table(posterior, specs)

    plotting._plot_chain_health(tmp_path, posterior, specs, max_tree_depth=8)
    plotting._plot_chain_ranked_trace(tmp_path, posterior.grouped_samples, specs, diagnostics)

    assert (tmp_path / "chain_health.pdf").exists()
    assert (tmp_path / "chain_ranked_trace.pdf").exists()

    skip_dir = tmp_path / "skip"
    missing_grouped = PosteriorResults(
        samples=posterior.samples,
        log_prob=posterior.log_prob,
        accept_prob=posterior.accept_prob,
        diverging=posterior.diverging,
        num_steps=posterior.num_steps,
        warmup_steps=0,
        sample_steps=posterior.sample_steps,
        num_chains=posterior.num_chains,
    )
    plotting._plot_chain_health(skip_dir, missing_grouped, specs, max_tree_depth=8)
    plotting._plot_chain_ranked_trace(skip_dir, None, specs, diagnostics)

    assert not (skip_dir / "chain_health.pdf").exists()
    assert not (skip_dir / "chain_ranked_trace.pdf").exists()


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
            "headline_chi_square": 12.0,
            "headline_dof": 4,
            "headline_reduced_chi_square": 3.0,
            "arc_aware_chi_square": 9.0,
            "arc_aware_dof": 5,
            "arc_aware_reduced_chi_square": 1.8,
            "arc_aware_arc_supported_image_count": 2,
            "arc_aware_missing_image_count": 1,
            "arc_aware_noncritical_support_radius_arcsec": 0.5,
            "chi_square_sigma_basis": "image_sigma_eff_arcsec",
            "chi_square_sigma_eff_median_arcsec": 0.59,
            "chi_square_sigma_eff_min_arcsec": 0.42,
            "chi_square_sigma_eff_max_arcsec": 0.71,
            "headline_chi_square_red1_total_sigma_arcsec": 0.6,
            "headline_chi_square_red1_pos_sigma_arcsec": 0.58,
            "arc_aware_chi_square_red1_total_sigma_arcsec": 0.59,
            "arc_aware_chi_square_red1_pos_sigma_arcsec": 0.57,
            "chi_square_red1_calibration_note": "post-fit diagnostic; holds image_sigma_int fixed",
            "n_effective_parameters": 4,
            "fit_quality_reference_sample_kind": "max_likelihood",
            "fit_quality_reference_sample_index": 7,
            "fit_quality_reference_source_loglike": -11.0,
            "fit_quality_reference_log_prob": -12.0,
            "ess_min": 10.0,
            "rhat_max": 1.02,
            "svi_health_warnings": ["near-zero SVI guide spread"],
        }
    )

    assert "Lensing Information" in text
    assert "Quality Of Fit" in text
    assert "headline_chi_square" in text
    assert "arc_aware_chi_square" in text
    assert "total image-plane sigma" in text
    assert "chi-square sigma basis" in text
    assert "chi-square median sigma arcsec" in text
    assert "headline red1 total sigma arcsec" in text
    assert "headline red1 pos_sigma_arcsec" in text
    assert "arc-aware red1 total sigma arcsec" in text
    assert "arc-aware red1 pos_sigma_arcsec" in text
    assert "post-fit diagnostic; holds image_sigma_int fixed" in text
    assert "N_arc_supported" in text
    assert "N_missing" in text
    assert "fit-quality reference" in text
    assert "arc-aware caveat" in text
    assert "diagnostic data points" not in text
    assert "diagnostic dof" not in text
    assert "AIC" not in text
    assert "BIC" not in text
    assert "Rhat max" not in text
    assert "SVI health warnings" not in text


def test_parse_args_caustic_source_redshift_default_and_explicit(monkeypatch: Any) -> None:
    from lenscluster import cluster_solver

    monkeypatch.setattr(sys, "argv", ["cluster_solver"])
    args = cluster_solver._parse_args()
    assert args.caustic_source_redshift == pytest.approx(9.0)
    assert args.caustic_plot_grid_scale_arcsec == pytest.approx(0.2)
    assert args.kappa_true_fits is None
    assert not hasattr(args, "caustic_num_pix")
    assert not hasattr(args, "validate_top_k_families")
    assert not hasattr(args, "validation_approx")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--caustic-source-redshift",
            "9.5",
            "--kappa-true-fits",
            "data/ff_sims/hera/kappa_z9_0.fits",
        ],
    )
    args = cluster_solver._parse_args()
    assert args.caustic_source_redshift == pytest.approx(9.5)
    assert args.kappa_true_fits == "data/ff_sims/hera/kappa_z9_0.fits"


def test_parse_args_image_catalog_rgb_display_controls(monkeypatch: Any) -> None:
    from lenscluster import cluster_solver

    monkeypatch.setattr(sys, "argv", ["cluster_solver"])
    args = cluster_solver._parse_args()
    assert args.image_catalog_family_cutout_rgb_q is None
    assert args.image_catalog_family_cutout_rgb_stretch is None
    assert args.image_catalog_family_cutout_rgb_minimum is None
    assert args.image_catalog_family_cutout_rgb_red_gain is None
    assert args.image_catalog_family_cutout_rgb_green_gain is None
    assert args.image_catalog_family_cutout_rgb_blue_gain is None

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--image-catalog-family-cutout-rgb-q",
            "6.5",
            "--image-catalog-family-cutout-rgb-stretch",
            "0.0165",
            "--image-catalog-family-cutout-rgb-minimum",
            "-0.001",
            "--image-catalog-family-cutout-rgb-red-gain",
            "0.68",
            "--image-catalog-family-cutout-rgb-green-gain",
            "0.69",
            "--image-catalog-family-cutout-rgb-blue-gain",
            "2.75",
        ],
    )
    args = cluster_solver._parse_args()
    assert args.image_catalog_family_cutout_rgb_q == pytest.approx(6.5)
    assert args.image_catalog_family_cutout_rgb_stretch == pytest.approx(0.0165)
    assert args.image_catalog_family_cutout_rgb_minimum == pytest.approx(-0.001)
    assert args.image_catalog_family_cutout_rgb_red_gain == pytest.approx(0.68)
    assert args.image_catalog_family_cutout_rgb_green_gain == pytest.approx(0.69)
    assert args.image_catalog_family_cutout_rgb_blue_gain == pytest.approx(2.75)

    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--image-catalog-family-cutout-rgb-q", "0"])
    with pytest.raises(SystemExit):
        cluster_solver._parse_args()

    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--image-catalog-family-cutout-rgb-minimum", "nan"])
    with pytest.raises(SystemExit):
        cluster_solver._parse_args()


def test_parse_args_caustic_plot_grid_scale_and_removed_num_pix(monkeypatch: Any) -> None:
    from lenscluster import cluster_solver

    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--caustic-plot-grid-scale-arcsec", "0.5"])
    args = cluster_solver._parse_args()
    assert args.caustic_plot_grid_scale_arcsec == pytest.approx(0.5)

    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--caustic-plot-grid-scale-arcsec", "0"])
    with pytest.raises(SystemExit):
        cluster_solver._parse_args()

    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--caustic-num-pix", "250"])
    with pytest.raises(SystemExit):
        cluster_solver._parse_args()


def test_parse_args_rejects_removed_main_validation_flags(monkeypatch: Any) -> None:
    from lenscluster import cluster_solver

    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--validate-top-k-families", "1"])
    with pytest.raises(SystemExit):
        cluster_solver._parse_args()

    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--validation-approx", "exact"])
    with pytest.raises(SystemExit):
        cluster_solver._parse_args()


def test_parse_args_rejects_removed_corner_suppress_fit_markers(monkeypatch: Any) -> None:
    from lenscluster import cluster_solver

    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--corner-suppress-fit-markers"])
    with pytest.raises(SystemExit):
        cluster_solver._parse_args()


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
    helper_calls: list[tuple[list[dict[str, float]], np.ndarray, np.ndarray]] = []

    def fake_subplots(*_args: Any, **_kwargs: Any) -> tuple[FakeFig, list[FakeAxis]]:
        return FakeFig(), [image_ax, source_ax]

    def fake_contours(
        model: FakeModel,
        kwargs_lens: list[dict[str, float]],
        x_axis: np.ndarray,
        y_axis: np.ndarray,
    ) -> list[dict[str, np.ndarray]]:
        helper_calls.append(
            (kwargs_lens, np.asarray(x_axis, dtype=float).copy(), np.asarray(y_axis, dtype=float).copy())
        )
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
        caustic_plot_grid_scale_arcsec=0.2,
        caustic_source_redshift=9.0,
    )

    assert (tmp_path / "caustic_overlay.pdf").exists()
    assert [float(item[0]) for item in evaluator.converted] == [4.0]
    assert evaluator.model_z == [9.0]
    assert evaluator.packed_z == [9.0]
    assert len(helper_calls) == 1
    kwargs_lens, x_axis, y_axis = helper_calls[0]
    assert kwargs_lens == [{"latent": 5.0}]
    assert len(x_axis) == 1001
    assert len(y_axis) == 1001
    np.testing.assert_allclose([x_axis[0], x_axis[-1]], [-100.0, 100.0])
    np.testing.assert_allclose([y_axis[0], y_axis[-1]], [-100.0, 100.0])
    assert x_axis[1] - x_axis[0] == pytest.approx(0.2)
    assert y_axis[1] - y_axis[0] == pytest.approx(0.2)
    assert len(image_ax.plots) == 1
    assert source_ax.plots == []
    assert len(source_ax.scatters) == 2
    np.testing.assert_allclose(source_ax.scatters[0][0], [0.1, 0.2])
    np.testing.assert_allclose(source_ax.scatters[0][1], [-0.1, -0.2])


def test_plot_absolute_magnification_uses_configured_grid_redshift_and_capped_abs(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    state = SimpleNamespace(z_lens=0.3, parameter_specs=[], family_data=[])

    class FakeModel:
        def __init__(self) -> None:
            self.inputs: list[tuple[np.ndarray, np.ndarray, list[dict[str, float]]]] = []

        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            x_array = np.asarray(x, dtype=float)
            y_array = np.asarray(y, dtype=float)
            self.inputs.append((x_array.copy(), y_array.copy(), kwargs_lens))
            values = np.zeros_like(x_array)
            values[0] = -2.0
            values[1] = 100.0
            values[2] = np.nan
            return values

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, FakeModel] = {}
            self.converted: list[np.ndarray] = []
            self.model_z: list[float] = []
            self.packed_z: list[float] = []
            self.model = FakeModel()

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            theta_array = np.asarray(theta, dtype=float)
            self.converted.append(theta_array.copy())
            return theta_array + 1.0

        def _get_exact_model_solver(self, z_source: float) -> tuple[FakeModel, None]:
            self.model_z.append(float(z_source))
            return self.model, None

        def _build_packed_lens_state(self, sample_latent: Any, z_source: float) -> dict[str, float]:
            self.packed_z.append(float(z_source))
            return {"latent": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    class FakeColorbar:
        def __init__(self) -> None:
            self.labels: list[str] = []

        def set_label(self, label: str) -> None:
            self.labels.append(label)

    class FakeAxis:
        def __init__(self) -> None:
            self.imshow_calls: list[tuple[np.ndarray, dict[str, Any]]] = []
            self.inverted = False
            self.xlabel: str | None = None
            self.ylabel: str | None = None
            self.title: str | None = None

        def imshow(self, data: Any, **kwargs: Any) -> str:
            self.imshow_calls.append((np.ma.asarray(data).filled(np.nan), dict(kwargs)))
            return "image"

        def invert_xaxis(self) -> None:
            self.inverted = True

        def set_xlabel(self, label: str) -> None:
            self.xlabel = label

        def set_ylabel(self, label: str) -> None:
            self.ylabel = label

        def set_title(self, title: str) -> None:
            self.title = title

    class FakeFig:
        def __init__(self, colorbar: FakeColorbar) -> None:
            self.colorbar_obj = colorbar
            self.saved_paths: list[Path] = []

        def colorbar(self, image: Any, ax: FakeAxis, **kwargs: Any) -> FakeColorbar:
            assert image == "image"
            assert isinstance(ax, FakeAxis)
            assert kwargs["fraction"] == pytest.approx(0.046)
            assert kwargs["pad"] == pytest.approx(0.04)
            return self.colorbar_obj

        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            self.saved_paths.append(Path(path))
            Path(path).touch()

    axis = FakeAxis()
    colorbar = FakeColorbar()
    fig = FakeFig(colorbar)

    monkeypatch.setattr(plotting.plt, "subplots", lambda *_args, **_kwargs: (fig, axis))
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)
    evaluator = FakeEvaluator()

    plotting._plot_absolute_magnification(
        tmp_path,
        evaluator,
        np.asarray([4.0], dtype=float),
        caustic_plot_grid_scale_arcsec=0.2,
        caustic_source_redshift=9.0,
    )

    assert (tmp_path / "absolute_magnification.pdf").exists()
    assert fig.saved_paths == [tmp_path / "absolute_magnification.pdf"]
    assert [float(item[0]) for item in evaluator.converted] == [4.0]
    assert evaluator.model_z == [9.0]
    assert evaluator.packed_z == [9.0]
    assert len(evaluator.model.inputs) == 1
    x_input, y_input, kwargs_lens = evaluator.model.inputs[0]
    assert kwargs_lens == [{"latent": 5.0}]
    assert x_input.size == 1001 * 1001
    assert y_input.size == 1001 * 1001
    np.testing.assert_allclose([np.nanmin(x_input), np.nanmax(x_input)], [-100.0, 100.0])
    np.testing.assert_allclose([np.nanmin(y_input), np.nanmax(y_input)], [-100.0, 100.0])
    assert x_input[1] - x_input[0] == pytest.approx(0.2)
    assert y_input[1001] - y_input[0] == pytest.approx(0.2)
    assert len(axis.imshow_calls) == 1
    image_data, image_kwargs = axis.imshow_calls[0]
    assert image_data.shape == (1001, 1001)
    assert image_data[0, 0] == pytest.approx(2.0)
    assert image_data[0, 1] == pytest.approx(plotting.ABSOLUTE_MAGNIFICATION_PLOT_CAP)
    assert np.isnan(image_data[0, 2])
    assert image_kwargs["cmap"] == "viridis"
    assert image_kwargs["vmin"] == pytest.approx(0.0)
    assert image_kwargs["vmax"] == pytest.approx(plotting.ABSOLUTE_MAGNIFICATION_PLOT_CAP)
    assert axis.inverted is True
    assert axis.xlabel == "x [arcsec]"
    assert axis.ylabel == "y [arcsec]"
    assert axis.title == "Absolute Magnification (z=9)"
    assert colorbar.labels == [r"$|\mu|$"]


def test_plot_kappa_true_comparison_uses_fits_grid_redshift_and_fixed_limits(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    true_path = tmp_path / "kappa_true.fits"
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    true_kappa = np.asarray([[1.0, 0.0], [np.nan, 2.0]], dtype=np.float32)
    fits.PrimaryHDU(true_kappa, header=wcs.to_header()).writeto(true_path)

    state = SimpleNamespace(z_lens=0.3, reference=(3, 0.0, 0.0), parameter_specs=[])

    class FakeModel:
        def __init__(self) -> None:
            self.inputs: list[tuple[np.ndarray, np.ndarray, list[dict[str, float]]]] = []

        def kappa(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            x_array = np.asarray(x, dtype=float)
            y_array = np.asarray(y, dtype=float)
            self.inputs.append((x_array.copy(), y_array.copy(), kwargs_lens))
            return np.asarray([2.0, 2.0, 2.0, 5.0], dtype=float)[: x_array.size]

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, FakeModel] = {}
            self.converted: list[np.ndarray] = []
            self.model_z: list[float] = []
            self.packed_z: list[float] = []
            self.model = FakeModel()

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            theta_array = np.asarray(theta, dtype=float)
            self.converted.append(theta_array.copy())
            return theta_array + 1.0

        def _get_exact_model_solver(self, z_source: float) -> tuple[FakeModel, None]:
            self.model_z.append(float(z_source))
            return self.model, None

        def _build_packed_lens_state(self, sample_latent: Any, z_source: float) -> dict[str, float]:
            self.packed_z.append(float(z_source))
            return {"latent": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    class FakeColorbar:
        def __init__(self) -> None:
            self.labels: list[str] = []

        def set_label(self, label: str) -> None:
            self.labels.append(label)

    class FakeAxis:
        def __init__(self) -> None:
            self.imshow_calls: list[tuple[np.ndarray, dict[str, Any]]] = []
            self.inverted = False
            self.xlabel: str | None = None
            self.ylabel: str | None = None
            self.title: str | None = None

        def imshow(self, data: Any, **kwargs: Any) -> str:
            self.imshow_calls.append((np.ma.asarray(data).filled(np.nan), dict(kwargs)))
            return f"image-{len(self.imshow_calls)}"

        def invert_xaxis(self) -> None:
            self.inverted = True

        def set_xlabel(self, label: str) -> None:
            self.xlabel = label

        def set_ylabel(self, label: str) -> None:
            self.ylabel = label

        def set_title(self, title: str) -> None:
            self.title = title

    class FakeFig:
        def __init__(self) -> None:
            self.colorbars: list[FakeColorbar] = []
            self.saved_paths: list[Path] = []

        def colorbar(self, image: Any, ax: FakeAxis, **kwargs: Any) -> FakeColorbar:
            assert image.startswith("image-")
            assert isinstance(ax, FakeAxis)
            assert kwargs["fraction"] == pytest.approx(0.046)
            assert kwargs["pad"] == pytest.approx(0.04)
            colorbar = FakeColorbar()
            self.colorbars.append(colorbar)
            return colorbar

        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            self.saved_paths.append(Path(path))
            Path(path).touch()

    axes = [FakeAxis(), FakeAxis()]
    fig = FakeFig()
    monkeypatch.setattr(plotting.plt, "subplots", lambda *_args, **_kwargs: (fig, axes))
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)
    evaluator = FakeEvaluator()

    plotting._plot_kappa_true_comparison(
        tmp_path,
        evaluator,
        np.asarray([4.0], dtype=float),
        true_path,
        caustic_source_redshift=9.0,
    )

    assert (tmp_path / "kappa_comparison.pdf").exists()
    assert fig.saved_paths == [tmp_path / "kappa_comparison.pdf"]
    assert [float(item[0]) for item in evaluator.converted] == [4.0]
    assert evaluator.model_z == [9.0]
    assert evaluator.packed_z == [9.0]
    assert len(evaluator.model.inputs) == 1
    _x_input, _y_input, kwargs_lens = evaluator.model.inputs[0]
    assert kwargs_lens == [{"latent": 5.0}]

    model_data, model_kwargs = axes[0].imshow_calls[0]
    residual_data, residual_kwargs = axes[1].imshow_calls[0]
    np.testing.assert_allclose(model_data, [[2.0, 2.0], [2.0, 5.0]])
    np.testing.assert_allclose(residual_data, [[1.0, np.nan], [np.nan, 1.5]], equal_nan=True)
    assert model_kwargs["vmin"] == pytest.approx(0.0)
    assert model_kwargs["vmax"] == pytest.approx(3.0)
    assert "vmin" not in residual_kwargs
    assert "vmax" not in residual_kwargs
    residual_norm = residual_kwargs["norm"]
    assert isinstance(residual_norm, plotting.TwoSlopeNorm)
    assert residual_norm.vmin == pytest.approx(-1.0)
    assert residual_norm.vcenter == pytest.approx(0.0)
    assert residual_norm.vmax == pytest.approx(2.0)
    assert axes[0].inverted is True
    assert axes[1].inverted is True
    assert axes[0].title == r"Model $\kappa$ (z=9)"
    assert axes[1].title == r"Fractional $\kappa$ Residual"
    assert [colorbar.labels for colorbar in fig.colorbars] == [
        [r"$\kappa_{\rm model}$"],
        [r"$(\kappa_{\rm model} - \kappa_{\rm true}) / \kappa_{\rm true}$"],
    ]


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


def test_image_recovery_uses_status_colors_and_small_points(monkeypatch: Any, tmp_path: Path) -> None:
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
            "image_recovery_status": ["recovered", "not_recovered", "recovered"],
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
    extra_df = pd.DataFrame(
        {
            "family_id": ["1"],
            "extra_image_index": [1],
            "image_recovery_status": ["extra"],
            "x_model_arcsec": [8.0],
            "y_model_arcsec": [-3.0],
        }
    )

    plotting._plot_image_recovery_fit_quality(image_df, tmp_path / "image_recovery.pdf", extra_df)

    assert (tmp_path / "image_recovery.pdf").exists()
    assert image_axis.scatters[0][2]["marker"] == "x"
    assert image_axis.scatters[0][2]["color"] == "tab:green"
    assert image_axis.scatters[0][2]["s"] < 30
    assert image_axis.scatters[0][2]["label"] == "recovered"
    assert image_axis.scatters[1][2]["marker"] == "x"
    assert image_axis.scatters[1][2]["color"] == "tab:red"
    assert image_axis.scatters[1][2]["label"] == "not recovered"
    assert image_axis.scatters[2][2]["marker"] == "o"
    assert image_axis.scatters[2][2]["color"] == "tab:blue"
    assert image_axis.scatters[2][2]["s"] < 20
    assert image_axis.scatters[2][2]["label"] == "extra"
    assert image_axis.errorbars[0][2]["fmt"] == "o"
    np.testing.assert_allclose(image_axis.errorbars[0][2]["color"], plotting._color_with_alpha("tab:green", 0.75))
    assert image_axis.errorbars[0][2]["ecolor"][3] < image_axis.errorbars[0][2]["color"][3]
    assert image_axis.errorbars[0][2]["markersize"] < 4
    assert len(image_axis.plots) == 2


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


def test_plot_image_residual_histogram_prefers_q50_and_writes_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_df = pd.DataFrame(
        {
            "image_residual_arcsec": [9.0, 9.0, 0.30, 9.0],
            "image_residual_q50": [0.04, np.nan, 0.08, np.inf],
        }
    )
    path = tmp_path / "image_residual_histogram.pdf"
    captured: dict[str, Any] = {"vertical_lines": [], "texts": []}
    original_subplots = plotting.plt.subplots

    def spy_subplots(*args: Any, **kwargs: Any) -> Any:
        fig, ax = original_subplots(*args, **kwargs)
        original_hist = ax.hist
        original_axvline = ax.axvline
        original_text = ax.text

        def spy_hist(values: Any, *hist_args: Any, **hist_kwargs: Any) -> Any:
            captured["residual"] = np.asarray(values, dtype=float).copy()
            return original_hist(values, *hist_args, **hist_kwargs)

        def spy_axvline(x: float = 0, *line_args: Any, **line_kwargs: Any) -> Any:
            captured["vertical_lines"].append((float(x), line_kwargs.get("label")))
            return original_axvline(x, *line_args, **line_kwargs)

        def spy_text(*text_args: Any, **text_kwargs: Any) -> Any:
            if len(text_args) >= 3:
                captured["texts"].append(str(text_args[2]))
            return original_text(*text_args, **text_kwargs)

        ax.hist = spy_hist
        ax.axvline = spy_axvline
        ax.text = spy_text
        return fig, ax

    monkeypatch.setattr(plotting.plt, "subplots", spy_subplots)

    plotting._plot_image_residual_histogram(image_df, path)

    assert path.exists()
    assert path.stat().st_size > 0
    expected_residual = np.asarray([0.04, 9.0, 0.08, 9.0])
    expected_rms = float(np.sqrt(np.mean(np.square(expected_residual))))
    np.testing.assert_allclose(captured["residual"], expected_residual)
    rms_line = next(x for x, label in captured["vertical_lines"] if label == "total RMS")
    assert rms_line == pytest.approx(expected_rms)
    assert any("Total RMS" in text and "N = 4" in text for text in captured["texts"])


def test_plot_image_residual_histogram_writes_placeholder_without_finite_values(tmp_path: Path) -> None:
    image_df = pd.DataFrame(
        {
            "image_residual_arcsec": [np.nan, np.inf, -np.inf],
            "image_residual_q50": [np.nan, np.inf, -np.inf],
        }
    )
    path = tmp_path / "image_residual_histogram.pdf"

    plotting._plot_image_residual_histogram(image_df, path)

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


def _smc_plot_posterior() -> PosteriorResults:
    samples = np.asarray(
        [
            [10.0, 0.28, -1.10],
            [11.0, 0.30, -1.00],
            [13.0, 0.33, -0.90],
            [16.0, 0.36, -0.75],
        ],
        dtype=float,
    )
    return PosteriorResults(
        samples=samples,
        log_prob=np.asarray([-12.0, -7.5, -5.0, -6.0], dtype=float),
        accept_prob=np.zeros(samples.shape[0], dtype=float),
        diverging=np.zeros(samples.shape[0], dtype=bool),
        num_steps=np.zeros(samples.shape[0], dtype=float),
        warmup_steps=0,
        sample_steps=samples.shape[0],
        num_chains=1,
        sampler="blackjax_smc",
        sample_weights=np.asarray([0.10, 0.20, 0.45, 0.25], dtype=float),
        temperature_schedule=np.asarray([0.0, 0.15, 0.55, 1.0], dtype=float),
        ess_history=np.asarray([4.0, 3.4, 3.0, 4.0], dtype=float),
        move_acceptance_history=np.asarray([0.72, 0.61, 0.58], dtype=float),
        init_diagnostics={
            "smc_particles": 4,
            "smc_target_ess_frac": 0.8,
            "smc_mean_move_acceptance": 0.6367,
        },
    )


def test_smc_diagnostic_plots_write_pdfs(tmp_path: Path) -> None:
    posterior = _smc_plot_posterior()

    plotting._plot_smc_diagnostics(tmp_path, posterior)
    plotting._plot_smc_weight_diagnostics(tmp_path, posterior)

    for filename in ["smc_diagnostics.pdf", "smc_weight_diagnostics.pdf"]:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_smc_plots_skip_missing_or_invalid_inputs(tmp_path: Path, monkeypatch: Any) -> None:
    empty = PosteriorResults(
        samples=np.empty((0, 2), dtype=float),
        log_prob=np.empty((0,), dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=0,
        num_chains=1,
        sampler="blackjax_smc",
    )

    class RaisingCorner:
        def corner(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("SMC corner should skip invalid weights")

    plotting._plot_smc_diagnostics(tmp_path, empty)
    plotting._plot_smc_weight_diagnostics(tmp_path, empty)
    monkeypatch.setattr(plotting, "corner", RaisingCorner())
    plotting._plot_smc_corner(
        tmp_path,
        _smc_plot_posterior().samples,
        _mixed_cosmology_test_specs(),
        np.asarray([0.5, np.nan, 0.25, 0.25], dtype=float),
    )

    assert not any(tmp_path.iterdir())


def test_smc_corner_uses_particle_weights_and_overlays(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []
    posterior = _smc_plot_posterior()

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

    plotting._plot_smc_corner(
        tmp_path,
        posterior.samples,
        _mixed_cosmology_test_specs(),
        posterior.sample_weights,
        best_fit_values={"halo.x": 13.0, "cosmology.Om0": 0.33, "cosmology.w0": -0.9},
        previous_stage_best_values={"halo.x": 11.0, "cosmology.Om0": 0.30, "cosmology.w0": -1.0},
    )

    assert calls[0][0] == "corner"
    np.testing.assert_allclose(calls[0][1], posterior.samples[:, [1, 2, 0]])
    assert calls[0][2]["labels"] == ["cosmology.Om0", "cosmology.w0", "halo.x"]
    np.testing.assert_allclose(calls[0][2]["weights"], posterior.sample_weights)
    assert calls[0][2]["plot_datapoints"] is True
    assert calls[1] == (
        "points",
        [[0.30, -1.0, 11.0]],
        {
            "marker": "x",
            "color": plotting.CORNER_PREVIOUS_STAGE_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert calls[2] == (
        "points",
        [[0.33, -0.9, 13.0]],
        {
            "marker": "x",
            "color": plotting.CORNER_BEST_FIT_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert (tmp_path / "smc_corner.pdf").exists()


def test_smc_corner_subset_prefers_cosmology_and_caps_dimensions() -> None:
    specs = [
        ParameterSpec("large.low", "large_low", "mock", 81, "x", "uniform", -5.0, 5.0, 0.1, component_family="large"),
        ParameterSpec("cosmology.Om0", "cosmology_Om0", "cosmology", 0, "Om0", "uniform", 0.05, 0.6, 0.01, component_family="cosmology"),
        ParameterSpec("source.1.beta_x", "source_1_beta_x", "1", 0, "beta_x", "normal", -np.inf, np.inf, 0.1, component_family="source_position"),
        ParameterSpec("large.high", "large_high", "mock", 81, "y", "uniform", -50.0, 50.0, 0.1, component_family="large"),
        ParameterSpec("cosmology.w0", "cosmology_w0", "cosmology", 0, "w0", "uniform", -2.0, -0.3, 0.05, component_family="cosmology"),
        ParameterSpec("large.mid", "large_mid", "mock", 81, "angle", "uniform", -10.0, 10.0, 0.1, component_family="large"),
    ]
    base = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    samples = np.column_stack(
        [
            base,
            0.28 + 0.01 * base,
            100.0 * base,
            20.0 * base,
            -1.1 + 0.05 * base,
            5.0 * base,
        ]
    )

    subset = plotting._smc_corner_subset(samples, specs, np.ones(samples.shape[0], dtype=float), max_params=3)

    assert subset is not None
    subset_samples, subset_specs, subset_weights = subset
    assert [spec.name for spec in subset_specs] == ["cosmology.Om0", "cosmology.w0", "large.high"]
    assert subset_samples.shape == (5, 3)
    np.testing.assert_allclose(subset_weights, np.full(5, 0.2))
