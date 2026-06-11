import argparse
import json
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
import jax
import jax.numpy as jnp
from astropy.cosmology import FlatwCDM

import lenscluster.cluster_solver as cluster_solver
import lenscluster.image_diagnostics as image_diagnostics
import lenscluster.mock_cluster as mock_cluster
import lenscluster.multi_cluster_solver as multi_cluster_solver
import lenscluster.validation as validation
from lenscluster.jax_cosmology import (
    ARCSEC_TO_RAD,
    C_LIGHT_KM_S,
    dpie_sigma0_factor,
    dpie_sigma0_factor_from_lensing_efficiency,
    dpie_sigma0_from_vel_disp,
    flat_wcdm_comoving_distance_mpc,
    flat_wcdm_kpc_per_arcsec,
    flat_wcdm_lens_geometry_factors,
    flat_wcdm_lensing_efficiency,
)
from lenscluster.cluster_solver import (
    IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
    IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
    IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
    IMAGE_PLANE_MODE_FORWARD_METRIC,
    IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED,
    IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    IMAGE_PLANE_MODE_NONE,
    SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    SAMPLE_LIKELIHOOD_SOURCE,
    _adaptive_active_scaling_count,
    _anchored_solved_image_plane_step_from_jacobian,
    _build_cosmology,
    _build_parameter_specs,
    _critical_arc_branch_probability,
    _critical_arc_geometry_from_jacobian,
    _critical_arc_lm_step_from_jacobian,
    _critical_arc_lm_geometry_from_jacobian,
    _critical_arc_mixture_image_plane_bin_loglike,
    _critical_arc_projected_quadratics,
    _effective_image_presence_penalty_weight,
    _fold_regularized_image_plane_bin_loglike,
    _forward_metric_image_plane_bin_loglike,
    _forward_metric_image_presence_residual2,
    _jittered_2x2_covariance_det,
    _linearized_image_plane_bin_loglike,
    _linearized_image_plane_residual_from_jacobian,
    _smooth_residual_cap,
    _source_plane_bin_loglike,
    _soft_observed_image_presence_loglike_from_residual2,
    _soft_observed_image_presence_loglike,
    _local_jacobian_bin_loglike,
    _normalize_stage_fit_controls,
    _parse_args,
    _validation_metrics_summary,
)
from lenscluster.lenstool_parser import _split_image_label, load_best_par
from lenscluster.model import (
    BuildState,
    ChainSeed,
    GeometryCache,
    EvaluationResult,
    FamilyData,
    FamilyValidationCache,
    PackedLensSpec,
    ParameterSpec,
    NUTSInitialization,
    PosteriorResults,
    Stage1PriorSummary,
    SurrogateBinCache,
)
from lenscluster.plotting import _run_summary
from lenscluster.validation import (
    PARAMETER_RECOVERY_LOG_ABS_FLOOR,
    SingleBCGMockConfig,
    generate_single_bcg_mock,
    load_chires_family_summary,
    load_chires_table,
    magnification_recovery_table,
    _log10_abs_parameter_values,
    _parameter_truth_with_source_positions,
    _normalize_validation_stage_fit_controls,
    parameter_recovery_table,
)
from lenscluster.utils import _rich_log_text, _should_log_to_console, format_stage_banner


def _install_recording_progress(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    instances: list[Any] = []

    class RecordingProgress:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs
            self.events: list[tuple[Any, ...]] = []
            self._next_task_id = 1
            instances.append(self)

        def __enter__(self):
            self.events.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.events.append(("exit", exc_type))
            return False

        def add_task(self, description: str, total: int | None = None) -> int:
            task_id = self._next_task_id
            self._next_task_id += 1
            self.events.append(("add_task", task_id, description, total))
            return task_id

        def update(self, task_id: int, **kwargs: object) -> None:
            self.events.append(("update", task_id, kwargs))

        def advance(self, task_id: int, advance: int = 1) -> None:
            self.events.append(("advance", task_id, advance))

    monkeypatch.setattr(validation, "Progress", RecordingProgress)
    return instances


def test_load_chires_table_parses_image_and_summary_rows(tmp_path: Path) -> None:
    path = tmp_path / "chires.dat"
    path.write_text(
        "\n".join(
            [
                "chi multiples",
                " N    ID    z   Narcs    chip    chix    chiy    chia   rmss     rmsi    dx      dy    nwarn",
                " 6    13c 1.005   1     21.67    0.00    0.00    0.00   0.343    0.00    0.12   -0.32  1",
                " 6     13 1.005   3     43.20    0.00    0.00    0.00   0.279    0.00    N/A     N/A   3",
            ]
        ),
        encoding="utf-8",
    )

    table = load_chires_table(path)
    summary = load_chires_family_summary(path)

    assert table.shape[0] == 2
    assert summary.shape[0] == 1
    assert summary.loc[0, "family_id"] == "13"
    assert summary.loc[0, "n_arcs"] == 3
    assert summary.loc[0, "source_rms_arcsec"] == 0.279
    assert pd.isna(summary.loc[0, "dx_arcsec"])


def test_stage_banner_formatter_renders_delimiter_and_details() -> None:
    lines = format_stage_banner("STAGE 2: stage2_joint", "fit_method=svi run_name=fit/stage2_joint")

    assert lines[0].startswith("[stage] ====")
    assert "STAGE 2: stage2_joint" in lines[0]
    assert lines[0].endswith("====")
    assert lines[1] == "[stage] fit_method=svi run_name=fit/stage2_joint"


def test_stage_banner_rich_rendering_styles_delimiter_and_title() -> None:
    banner = format_stage_banner("STAGE 2: stage2_joint")[0]
    rendered = _rich_log_text("2026-05-15T12:00:00", banner)
    styles = [str(span.style) for span in rendered.spans]

    assert rendered.plain == f"2026-05-15T12:00:00 {banner}"
    assert "bold white on magenta" in styles
    assert styles.count("bold magenta") >= 3


def test_stage_start_rich_rendering_keeps_normal_stage_style() -> None:
    rendered = _rich_log_text("2026-05-15T12:00:00", "[stage] start run_name=fit/stage2_joint")
    styles = [str(span.style) for span in rendered.spans]

    assert rendered.plain == "2026-05-15T12:00:00 [stage] start run_name=fit/stage2_joint"
    assert "bold white on magenta" not in styles
    assert styles.count("bold magenta") == 1


def test_smc_progress_logs_are_console_visible() -> None:
    rendered = _rich_log_text("2026-05-15T12:00:00", "[smc] step=1 temperature=0.1")
    styles = [str(span.style) for span in rendered.spans]

    assert _should_log_to_console("[smc] step=1 temperature=0.1")
    assert rendered.plain == "2026-05-15T12:00:00 [smc] step=1 temperature=0.1"
    assert "bold yellow" in styles


def test_stage_banner_title_falls_back_to_run_name() -> None:
    assert cluster_solver._stage_banner_title_from_run_name("custom_run") == "custom_run"


def test_generate_single_bcg_mock_parses_and_has_finite_magnifications(tmp_path: Path) -> None:
    config = SingleBCGMockConfig(seed=7, n_primary_families=2, pos_sigma_arcsec=0.0)

    paths, images, truth = generate_single_bcg_mock(tmp_path, config)
    _parsed, _potentials_df, images_df, potentials_with_priors = load_best_par(paths.par_path)

    assert paths.par_path.exists()
    assert paths.image_catalog_path.exists()
    arc_rows = [
        line for line in paths.image_catalog_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    arc_labels = [row.split()[0] for row in arc_rows]
    assert arc_rows
    assert all(not row.startswith((" ", "	")) for row in arc_rows)
    assert all(
        label.split(".", 1)[0].isdigit()
        and label.split(".", 1)[1].isalpha()
        and label.split(".", 1)[1].islower()
        for label in arc_labels
    )
    assert len(potentials_with_priors) == 2
    assert images_df["family_id"].nunique() == 2
    assert sorted(images_df.groupby("family_id")["catalog_z"].first().round(3).tolist()) == [1.5, 2.0]
    assert (images.groupby("family_id").size() >= config.min_images_per_family).all()
    assert {source["caustic_class"] for source in truth["sources"]} == {"primary"}
    assert all(int(source["n_images"]) >= 3 for source in truth["sources"])
    assert truth["config"]["primary_source_redshifts"] == [1.5, 2.0, 3.0]
    assert truth["config"]["subhalo_source_redshifts"] == [1.5, 2.0, 3.0]
    assert "source_redshifts" not in truth["config"]
    assert np.isfinite(images["magnification_true"].to_numpy(dtype=float)).all()
    assert set(truth["parameter_truth"]) >= {"halo.v_disp", "bcg.v_disp", "source.sigma_int"}
    assert truth["parameter_truth"]["source.sigma_int"] == config.source_sigma_int_arcsec
    first_source = truth["sources"][0]
    assert truth["parameter_truth"]["source.1.beta_x"] == pytest.approx(first_source["beta_x"])
    assert truth["parameter_truth"]["source.1.beta_y"] == pytest.approx(first_source["beta_y"])
    first_z_key = sorted(truth["caustics_by_source_redshift"])[0]
    first_caustic = truth["caustics_by_source_redshift"][first_z_key][0]
    for key in ("critical_x", "critical_y", "caustic_beta_x", "caustic_beta_y"):
        values = np.asarray(first_caustic[key], dtype=float)
        assert values.ndim == 1
        assert values.size >= 3
        assert np.isfinite(values).all()


def test_write_single_bcg_mock_par_can_tighten_bcg_position_prior(tmp_path: Path) -> None:
    config = SingleBCGMockConfig(bcg_position_prior_half_width_arcsec=2.0)
    path = tmp_path / "single_bcg_mock.par"
    images = pd.DataFrame({"x_obs_arcsec": [0.0], "y_obs_arcsec": [0.0]})

    mock_cluster._write_single_bcg_par(path, config, images)
    par_lines = path.read_text(encoding="utf-8").splitlines()

    def position_bounds(component_id: str, field_name: str) -> tuple[float, float]:
        in_limit_block = False
        for line in par_lines:
            stripped = line.strip()
            if stripped == f"limit {component_id}":
                in_limit_block = True
                continue
            if in_limit_block and stripped == "end":
                break
            if in_limit_block and stripped.startswith(field_name):
                parts = stripped.split()
                return float(parts[2]), float(parts[3])
        raise AssertionError(f"missing {field_name} limit for {component_id}")

    assert position_bounds("halo", "x_centre") == pytest.approx((-8.0, 8.0))
    assert position_bounds("halo", "y_centre") == pytest.approx((-8.0, 8.0))
    assert position_bounds("bcg", "x_centre") == pytest.approx((-1.65, 2.35))
    assert position_bounds("bcg", "y_centre") == pytest.approx((-2.22, 1.78))


def test_generate_single_bcg_mock_respects_max_images_per_family(tmp_path: Path) -> None:
    config = SingleBCGMockConfig(
        seed=7,
        n_primary_families=1,
        min_images_per_family=2,
        max_images_per_family=3,
        pos_sigma_arcsec=0.0,
    )

    _paths, images, truth = generate_single_bcg_mock(tmp_path, config)

    family_sizes = images.groupby("family_id").size()
    assert (family_sizes >= config.min_images_per_family).all()
    assert (family_sizes <= config.max_images_per_family).all()
    assert all(
        config.min_images_per_family <= int(source["n_images"]) <= int(config.max_images_per_family)
        for source in truth["sources"]
    )
    assert truth["config"]["max_images_per_family"] == config.max_images_per_family
    restored_config = validation._caustic_config_from_truth(truth)
    assert restored_config.max_images_per_family == config.max_images_per_family


def test_caustic_config_from_truth_maps_legacy_source_redshifts_to_split_lists() -> None:
    restored_config = validation._caustic_config_from_truth(
        {
            "config": {
                "source_redshift": 2.0,
                "source_redshifts": [2.0, 4.0],
            }
        }
    )

    assert restored_config.primary_source_redshifts == (2.0, 4.0)
    assert restored_config.subhalo_source_redshifts == (2.0, 4.0)


def test_caustic_config_from_truth_restores_mock_image_min_distances() -> None:
    restored_config = validation._caustic_config_from_truth(
        {
            "config": {
                "primary_image_min_distance_arcsec": 2.5,
                "subhalo_image_min_distance_arcsec": 0.75,
            }
        }
    )

    assert restored_config.primary_image_min_distance_arcsec == pytest.approx(2.5)
    assert restored_config.subhalo_image_min_distance_arcsec == pytest.approx(0.75)


def test_generate_single_bcg_mock_emits_progress_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_min_distances: list[float] = []

    class FakeModel:
        def magnification(self, x, _y, _kwargs_lens):
            return np.ones_like(np.asarray(x, dtype=float))

    class FakeSolver:
        def __init__(self, _model) -> None:
            pass

        def image_position_from_source(self, _source_x, _source_y, _kwargs_lens, **kwargs):
            captured_min_distances.append(float(kwargs["min_distance"]))
            return np.asarray([0.0, 1.0, 2.0], dtype=float), np.asarray([0.0, 0.1, 0.0], dtype=float)

    contour = mock_cluster.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([0.0, 0.1, 0.0, 0.0], dtype=float),
        beta_y=np.asarray([0.0, 0.0, 0.1, 0.0], dtype=float),
        critical_x=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
        critical_y=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float),
        caustic_area_arcsec2=0.01,
        critical_area_arcsec2=1.0,
    )

    monkeypatch.setattr(
        mock_cluster,
        "_mock_model_and_kwargs",
        lambda _config, _subhalo_components, _z_source, _cosmo, _cosmo_config: (FakeModel(), []),
    )
    monkeypatch.setattr(mock_cluster, "LensEquationSolver", FakeSolver)
    monkeypatch.setattr(
        mock_cluster,
        "_compute_tangential_caustic_contours",
        lambda _model, _kwargs_lens, _config: [contour],
    )
    monkeypatch.setattr(
        mock_cluster,
        "_sample_caustic_source_candidate",
        lambda contours, _caustic_class, _rng, *, inside_caustic=True: (0.01, 0.02, contours[0]),
    )
    events: list[tuple[str, dict[str, Any]]] = []

    config = SingleBCGMockConfig(
        seed=7,
        n_primary_families=1,
        source_redshift=2.0,
        primary_source_redshifts=(2.0,),
        pos_sigma_arcsec=0.0,
        n_subhalos=0,
    )

    generate_single_bcg_mock(tmp_path, config, progress_callback=lambda event, payload: events.append((event, payload)))

    event_names = [event for event, _payload in events]
    assert event_names[:2] == ["subhalos_start", "subhalos_complete"]
    assert "redshift_start" in event_names
    assert "redshift_complete" in event_names
    assert "family_attempt" in event_names
    assert "family_accept" in event_names
    assert event_names[-1] == "outputs_complete"
    assert next(payload for event, payload in events if event == "redshift_complete")["caustic_count"] == 1
    accepted_payload = next(payload for event, payload in events if event == "family_accept")
    assert accepted_payload["family_index"] == 1
    assert accepted_payload["image_count"] == 3
    assert accepted_payload["image_min_distance_arcsec"] == pytest.approx(
        mock_cluster.DEFAULT_PRIMARY_IMAGE_MIN_DISTANCE_ARCSEC
    )
    assert captured_min_distances == [pytest.approx(mock_cluster.DEFAULT_PRIMARY_IMAGE_MIN_DISTANCE_ARCSEC)]


def test_generate_single_bcg_mock_uses_family_class_image_min_distances(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_min_distances: list[tuple[float, float]] = []

    class FakeModel:
        def magnification(self, x, _y, _kwargs_lens):
            return np.ones_like(np.asarray(x, dtype=float))

    class FakeSolver:
        def __init__(self, _model) -> None:
            pass

        def image_position_from_source(self, source_x, _source_y, _kwargs_lens, **kwargs):
            captured_min_distances.append((float(source_x), float(kwargs["min_distance"])))
            return np.asarray([0.0, 1.0, 2.0], dtype=float), np.asarray([0.0, 0.1, 0.0], dtype=float)

    primary_contour = mock_cluster.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([0.0, 0.1, 0.0, 0.0], dtype=float),
        beta_y=np.asarray([0.0, 0.0, 0.1, 0.0], dtype=float),
        critical_x=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
        critical_y=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float),
        caustic_area_arcsec2=0.02,
        critical_area_arcsec2=1.0,
    )
    subhalo_contour = mock_cluster.CausticContour(
        caustic_index=1,
        caustic_class="subhalo",
        beta_x=np.asarray([0.2, 0.3, 0.2, 0.2], dtype=float),
        beta_y=np.asarray([0.2, 0.2, 0.3, 0.2], dtype=float),
        critical_x=np.asarray([2.0, 3.0, 2.0, 2.0], dtype=float),
        critical_y=np.asarray([2.0, 2.0, 3.0, 2.0], dtype=float),
        caustic_area_arcsec2=0.01,
        critical_area_arcsec2=0.5,
    )

    monkeypatch.setattr(
        mock_cluster,
        "_mock_model_and_kwargs",
        lambda _config, _subhalo_components, _z_source, _cosmo, _cosmo_config: (FakeModel(), []),
    )
    monkeypatch.setattr(mock_cluster, "LensEquationSolver", FakeSolver)
    monkeypatch.setattr(
        mock_cluster,
        "_compute_tangential_caustic_contours",
        lambda _model, _kwargs_lens, _config: [primary_contour, subhalo_contour],
    )

    def sample_candidate(contours, caustic_class, _rng, *, inside_caustic=True):
        del inside_caustic
        contour = next(contour for contour in contours if contour.caustic_class == caustic_class)
        beta_x = 0.01 if caustic_class == "primary" else 0.02
        return beta_x, 0.03, contour

    monkeypatch.setattr(mock_cluster, "_sample_caustic_source_candidate", sample_candidate)
    config = SingleBCGMockConfig(
        seed=7,
        n_primary_families=1,
        n_subhalo_families=1,
        primary_source_redshifts=(2.0,),
        subhalo_source_redshifts=(5.0,),
        primary_image_min_distance_arcsec=2.5,
        subhalo_image_min_distance_arcsec=0.75,
        pos_sigma_arcsec=0.0,
    )

    _paths, _images, truth = generate_single_bcg_mock(tmp_path, config)

    distances_by_source_x = {round(source_x, 2): min_distance for source_x, min_distance in captured_min_distances}
    assert distances_by_source_x[0.01] == pytest.approx(2.5)
    assert distances_by_source_x[0.02] == pytest.approx(0.75)
    assert truth["config"]["primary_image_min_distance_arcsec"] == pytest.approx(2.5)
    assert truth["config"]["subhalo_image_min_distance_arcsec"] == pytest.approx(0.75)


def test_generate_single_bcg_mock_caps_workers_and_accepts_first_valid_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeModel:
        def magnification(self, x, _y, _kwargs_lens):
            return np.ones_like(np.asarray(x, dtype=float))

    class FakeSolver:
        def __init__(self, _model) -> None:
            pass

        def image_position_from_source(self, source_x, _source_y, _kwargs_lens, **_kwargs):
            image_count_by_source = {0.01: 2, 0.02: 4, 0.03: 3, 0.04: 3}
            n_images = image_count_by_source[round(float(source_x), 2)]
            return np.arange(n_images, dtype=float), np.zeros(n_images, dtype=float)

    executor_worker_counts: list[int] = []
    submitted_sources: list[float] = []

    class InlineExecutor:
        def __init__(self, *, max_workers: int) -> None:
            executor_worker_counts.append(int(max_workers))

        def __enter__(self) -> "InlineExecutor":
            return self

        def __exit__(self, *_exc: Any) -> bool:
            return False

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            pass

        def submit(
            self,
            fn: Any,
            z_source: float,
            attempt_index: int,
            beta_x: float,
            beta_y: float,
            contour: Any,
            image_min_distance_arcsec: float,
        ) -> Future[Any]:
            submitted_sources.append(float(beta_x))
            future: Future[Any] = Future()
            try:
                future.set_result(fn(z_source, attempt_index, beta_x, beta_y, contour, image_min_distance_arcsec))
            except BaseException as exc:
                future.set_exception(exc)
            return future

    contour = mock_cluster.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([0.0, 0.1, 0.0, 0.0], dtype=float),
        beta_y=np.asarray([0.0, 0.0, 0.1, 0.0], dtype=float),
        critical_x=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
        critical_y=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float),
        caustic_area_arcsec2=0.01,
        critical_area_arcsec2=1.0,
    )
    candidates = iter((0.01, 0.02, 0.03, 0.04))

    monkeypatch.setattr(mock_cluster, "jax_cpu_worker_count", lambda: 3)
    monkeypatch.setattr(mock_cluster, "ThreadPoolExecutor", InlineExecutor)
    monkeypatch.setattr(
        mock_cluster,
        "_mock_model_and_kwargs",
        lambda _config, _subhalo_components, _z_source, _cosmo, _cosmo_config: (FakeModel(), []),
    )
    monkeypatch.setattr(mock_cluster, "LensEquationSolver", FakeSolver)
    monkeypatch.setattr(
        mock_cluster,
        "_compute_tangential_caustic_contours",
        lambda _model, _kwargs_lens, _config: [contour],
    )
    monkeypatch.setattr(
        mock_cluster,
        "_sample_caustic_source_candidate",
        lambda contours, _caustic_class, _rng, *, inside_caustic=True: (next(candidates), 0.02, contours[0]),
    )

    config = SingleBCGMockConfig(
        seed=7,
        n_primary_families=1,
        primary_source_redshifts=(2.0,),
        pos_sigma_arcsec=0.0,
        max_sources_to_try=4,
    )

    _paths, _images, truth = generate_single_bcg_mock(tmp_path, config)

    assert executor_worker_counts == [1]
    assert submitted_sources == [0.01, 0.02]
    assert truth["sources"][0]["beta_x"] == pytest.approx(0.02)
    assert truth["sources"][0]["n_images"] == 4


def test_generate_single_bcg_mock_queues_first_attempts_across_families(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeModel:
        def __init__(self, z_source: float) -> None:
            self.z_source = float(z_source)

        def magnification(self, x, _y, _kwargs_lens):
            return np.ones_like(np.asarray(x, dtype=float))

    class FakeSolver:
        def __init__(self, _model) -> None:
            pass

        def image_position_from_source(self, source_x, _source_y, _kwargs_lens, **_kwargs):
            marker = int(float(source_x))
            attempt = int(round((float(source_x) - marker) * 100.0))
            n_images = 2 if marker == 20 and attempt == 1 else 3
            return np.arange(n_images, dtype=float), np.zeros(n_images, dtype=float)

    executor_worker_counts: list[int] = []
    submitted: list[tuple[float, int]] = []

    class InlineExecutor:
        def __init__(self, *, max_workers: int) -> None:
            executor_worker_counts.append(int(max_workers))

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            pass

        def submit(
            self,
            fn: Any,
            z_source: float,
            attempt_index: int,
            beta_x: float,
            beta_y: float,
            contour: Any,
            image_min_distance_arcsec: float,
        ) -> Future[Any]:
            submitted.append((float(z_source), int(attempt_index)))
            future: Future[Any] = Future()
            try:
                future.set_result(fn(z_source, attempt_index, beta_x, beta_y, contour, image_min_distance_arcsec))
            except BaseException as exc:
                future.set_exception(exc)
            return future

    def fake_model_and_kwargs(_config, _subhalo_components, z_source, _cosmo, _cosmo_config):
        return FakeModel(float(z_source)), []

    def fake_contours(model, _kwargs_lens, _config):
        marker = int(round(float(model.z_source) * 10.0))
        return [
            mock_cluster.CausticContour(
                caustic_index=marker,
                caustic_class="primary",
                beta_x=np.asarray([0.0, 0.1, 0.0, 0.0], dtype=float),
                beta_y=np.asarray([0.0, 0.0, 0.1, 0.0], dtype=float),
                critical_x=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
                critical_y=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float),
                caustic_area_arcsec2=0.01,
                critical_area_arcsec2=1.0,
            )
        ]

    attempt_by_marker: dict[int, int] = {}

    def fake_sample(contours, _caustic_class, _rng, *, inside_caustic=True):
        marker = int(contours[0].caustic_index)
        attempt_by_marker[marker] = attempt_by_marker.get(marker, 0) + 1
        return float(marker) + 0.01 * attempt_by_marker[marker], 0.02, contours[0]

    monkeypatch.setattr(mock_cluster, "jax_cpu_worker_count", lambda: 2)
    monkeypatch.setattr(mock_cluster, "ThreadPoolExecutor", InlineExecutor)
    monkeypatch.setattr(mock_cluster, "_mock_model_and_kwargs", fake_model_and_kwargs)
    monkeypatch.setattr(mock_cluster, "LensEquationSolver", FakeSolver)
    monkeypatch.setattr(mock_cluster, "_compute_tangential_caustic_contours", fake_contours)
    monkeypatch.setattr(mock_cluster, "_sample_caustic_source_candidate", fake_sample)

    config = SingleBCGMockConfig(
        seed=7,
        n_primary_families=3,
        primary_source_redshifts=(2.0, 3.0, 4.0),
        pos_sigma_arcsec=0.0,
        n_subhalos=0,
        max_sources_to_try=2,
    )

    _paths, _images, truth = generate_single_bcg_mock(tmp_path, config)

    assert executor_worker_counts == [2]
    assert submitted == [(2.0, 1), (3.0, 1), (4.0, 1), (2.0, 2)]
    assert [source["family_id"] for source in truth["sources"]] == ["1", "2", "3"]
    assert [source["n_images"] for source in truth["sources"]] == [3, 3, 3]


def test_generate_single_bcg_mock_queue_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeModel:
        def magnification(self, x, _y, _kwargs_lens):
            return np.ones_like(np.asarray(x, dtype=float))

    class FakeSolver:
        def __init__(self, _model) -> None:
            pass

        def image_position_from_source(self, _source_x, _source_y, _kwargs_lens, **_kwargs):
            return np.asarray([0.0, 1.0, 2.0], dtype=float), np.asarray([0.0, 0.1, 0.0], dtype=float)

    contour = mock_cluster.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([0.0, 0.1, 0.0, 0.0], dtype=float),
        beta_y=np.asarray([0.0, 0.0, 0.1, 0.0], dtype=float),
        critical_x=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
        critical_y=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float),
        caustic_area_arcsec2=0.01,
        critical_area_arcsec2=1.0,
    )

    monkeypatch.setattr(mock_cluster, "jax_cpu_worker_count", lambda: 3)
    monkeypatch.setattr(
        mock_cluster,
        "_mock_model_and_kwargs",
        lambda _config, _subhalo_components, _z_source, _cosmo, _cosmo_config: (FakeModel(), []),
    )
    monkeypatch.setattr(mock_cluster, "LensEquationSolver", FakeSolver)
    monkeypatch.setattr(
        mock_cluster,
        "_compute_tangential_caustic_contours",
        lambda _model, _kwargs_lens, _config: [contour],
    )
    monkeypatch.setattr(
        mock_cluster,
        "_sample_caustic_source_candidate",
        lambda contours, _caustic_class, rng, *, inside_caustic=True: (
            float(rng.uniform(-0.1, 0.1)),
            float(rng.uniform(-0.1, 0.1)),
            contours[0],
        ),
    )

    config = SingleBCGMockConfig(
        seed=17,
        n_primary_families=4,
        primary_source_redshifts=(2.0, 3.0),
        pos_sigma_arcsec=0.05,
        n_subhalos=0,
    )

    _paths_a, images_a, truth_a = generate_single_bcg_mock(tmp_path / "a", config)
    _paths_b, images_b, truth_b = generate_single_bcg_mock(tmp_path / "b", config)

    assert images_a.to_dict("records") == images_b.to_dict("records")
    assert truth_a["sources"] == truth_b["sources"]
    assert truth_a["images"] == truth_b["images"]


def test_generate_single_bcg_mock_queue_raises_when_family_exhausts_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeModel:
        def magnification(self, x, _y, _kwargs_lens):
            return np.ones_like(np.asarray(x, dtype=float))

    class FakeSolver:
        def __init__(self, _model) -> None:
            pass

        def image_position_from_source(self, _source_x, _source_y, _kwargs_lens, **_kwargs):
            return np.asarray([0.0, 1.0], dtype=float), np.asarray([0.0, 0.1], dtype=float)

    contour = mock_cluster.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([0.0, 0.1, 0.0, 0.0], dtype=float),
        beta_y=np.asarray([0.0, 0.0, 0.1, 0.0], dtype=float),
        critical_x=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
        critical_y=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float),
        caustic_area_arcsec2=0.01,
        critical_area_arcsec2=1.0,
    )

    monkeypatch.setattr(mock_cluster, "jax_cpu_worker_count", lambda: 2)
    monkeypatch.setattr(
        mock_cluster,
        "_mock_model_and_kwargs",
        lambda _config, _subhalo_components, _z_source, _cosmo, _cosmo_config: (FakeModel(), []),
    )
    monkeypatch.setattr(mock_cluster, "LensEquationSolver", FakeSolver)
    monkeypatch.setattr(
        mock_cluster,
        "_compute_tangential_caustic_contours",
        lambda _model, _kwargs_lens, _config: [contour],
    )
    monkeypatch.setattr(
        mock_cluster,
        "_sample_caustic_source_candidate",
        lambda contours, _caustic_class, rng, *, inside_caustic=True: (
            float(rng.uniform(-0.1, 0.1)),
            float(rng.uniform(-0.1, 0.1)),
            contours[0],
        ),
    )
    config = SingleBCGMockConfig(
        seed=19,
        n_primary_families=2,
        primary_source_redshifts=(2.0,),
        n_subhalos=0,
        max_sources_to_try=2,
    )

    with pytest.raises(RuntimeError, match="Failed to generate a primary source"):
        generate_single_bcg_mock(tmp_path, config)


def test_generate_single_bcg_mock_cycles_primary_and_subhalo_redshifts_independently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeModel:
        def magnification(self, x, _y, _kwargs_lens):
            return np.ones_like(np.asarray(x, dtype=float))

    class FakeSolver:
        def __init__(self, _model) -> None:
            pass

        def image_position_from_source(self, _source_x, _source_y, _kwargs_lens, **_kwargs):
            return np.asarray([0.0, 1.0, 2.0], dtype=float), np.asarray([0.0, 0.1, 0.0], dtype=float)

    primary_contour = mock_cluster.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([0.0, 0.1, 0.0, 0.0], dtype=float),
        beta_y=np.asarray([0.0, 0.0, 0.1, 0.0], dtype=float),
        critical_x=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float),
        critical_y=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float),
        caustic_area_arcsec2=0.02,
        critical_area_arcsec2=1.0,
    )
    subhalo_contour = mock_cluster.CausticContour(
        caustic_index=1,
        caustic_class="subhalo",
        beta_x=np.asarray([0.2, 0.3, 0.2, 0.2], dtype=float),
        beta_y=np.asarray([0.2, 0.2, 0.3, 0.2], dtype=float),
        critical_x=np.asarray([2.0, 3.0, 2.0, 2.0], dtype=float),
        critical_y=np.asarray([2.0, 2.0, 3.0, 2.0], dtype=float),
        caustic_area_arcsec2=0.01,
        critical_area_arcsec2=0.5,
    )

    monkeypatch.setattr(
        mock_cluster,
        "_mock_model_and_kwargs",
        lambda _config, _subhalo_components, _z_source, _cosmo, _cosmo_config: (FakeModel(), []),
    )
    monkeypatch.setattr(mock_cluster, "LensEquationSolver", FakeSolver)
    monkeypatch.setattr(
        mock_cluster,
        "_compute_tangential_caustic_contours",
        lambda _model, _kwargs_lens, _config: [primary_contour, subhalo_contour],
    )

    def sample_candidate(contours, caustic_class, _rng, *, inside_caustic=True):
        contour = next(contour for contour in contours if contour.caustic_class == caustic_class)
        return 0.01, 0.02, contour

    monkeypatch.setattr(mock_cluster, "_sample_caustic_source_candidate", sample_candidate)
    config = SingleBCGMockConfig(
        seed=7,
        n_primary_families=3,
        n_subhalo_families=2,
        primary_source_redshifts=(2.0, 3.0),
        subhalo_source_redshifts=(5.0,),
        pos_sigma_arcsec=0.0,
    )

    _paths, images, truth = generate_single_bcg_mock(tmp_path, config)

    primary_sources = [source for source in truth["sources"] if source["caustic_class"] == "primary"]
    subhalo_sources = [source for source in truth["sources"] if source["caustic_class"] == "subhalo"]
    assert [source["z_source"] for source in primary_sources] == [2.0, 3.0, 2.0]
    assert [source["z_source"] for source in subhalo_sources] == [5.0, 5.0]
    assert images["family_id"].nunique() == 5
    assert sorted(truth["caustics_by_source_redshift"]) == ["2.00000000", "3.00000000", "5.00000000"]


def test_load_best_par_defaults_missing_potfile_slopes_to_four(tmp_path: Path) -> None:
    catalog_path = tmp_path / "members.cat"
    catalog_path.write_text(
        "#REFERENCE 0\n"
        "1 39.970000 -1.580000 1.0 1.0 0.0 19.5000 1.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "missing_slopes.par"
    par_path.write_text(
        """
runmode
    reference 3 39.971340 -1.582260
    end

potfile
    filein 3 members.cat
    zlens 0.375000
    type 81
    corekpc 0.150000
    mag0 19.5
    sigma 1 10. 200.
    cutkpc 1 1. 40.
    end
fini
""",
        encoding="utf-8",
    )

    parsed, _potentials_df, _images_df, _potentials_with_priors = load_best_par(par_path)

    potfile = parsed["potfiles"][0]
    assert potfile["vdslope"] == [0, 4.0, 0.0]
    assert potfile["slope"] == [0, 4.0, 0.0]
    assert potfile["vdslope_nominal"] == pytest.approx(4.0)
    assert potfile["slope_nominal"] == pytest.approx(4.0)


def test_load_best_par_mode9_potfile_nominal_values_use_mean(tmp_path: Path) -> None:
    catalog_path = tmp_path / "members.cat"
    catalog_path.write_text(
        "#REFERENCE 0\n"
        "1 39.970000 -1.580000 1.0 1.0 0.0 19.5000 1.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "mode9_potfile.par"
    par_path.write_text(
        """
runmode
    reference 3 39.971340 -1.582260
    end

potfile
    filein 3 members.cat
    zlens 0.375000
    type 81
    corekpc 0.150000
    mag0 19.5
    sigma 9 245.0 55.0 190.0 300.0
    cutkpc 9 20.0 5.0 1.0 40.0
    end
fini
""",
        encoding="utf-8",
    )

    parsed, _potentials_df, _images_df, _potentials_with_priors = load_best_par(par_path)

    potfile = parsed["potfiles"][0]
    assert potfile["sigma_nominal"] == pytest.approx(245.0)
    assert potfile["cutkpc_nominal"] == pytest.approx(20.0)


def test_load_best_par_strips_inline_comments_in_named_blocks(tmp_path: Path) -> None:
    catalog_path = tmp_path / "members.cat"
    catalog_path.write_text(
        "#REFERENCE 0\n"
        "1 39.970000 -1.580000 1.0 1.0 0.0 19.5000 1.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "inline_comments.par"
    par_path.write_text(
        """
runmode
    reference 3 39.971340 -1.582260
    end

potentiel 1 # main halo
    profil 81
    x_centre 0 #-0.853
    y_centre 0 # -1.999
    ellipticite 0.1
    angle_pos 12.0
    core_radius 1.0
    cut_radius 20.0
    v_disp 700.0
    z_lens 0.375000
    # full-line comment inside a block
    end

potfile # member scaling
    filein 3 members.cat # cluster members
    zlens 0.375000
    type 81
    corekpc 0.150000
    mag0 19.5 # BCG mag
    sigma 1 10. 200. # real prior list
    cutkpc 1 1. 40. # real prior list
    vdslope 0 3.5 0
    slope 0 3.0 0
    end
fini
""",
        encoding="utf-8",
    )

    parsed, potentials_df, _images_df, potentials_with_priors = load_best_par(par_path)

    assert potentials_df.loc[0, "id"] == "1"
    assert potentials_df.loc[0, "x_centre"] == pytest.approx(0.0)
    assert potentials_df.loc[0, "y_centre"] == pytest.approx(0.0)
    assert potentials_with_priors[0]["id"] == "1"
    potfile = parsed["potfiles"][0]
    assert potfile["mag0"] == pytest.approx(19.5)
    assert potfile["sigma"] == [1, 10.0, 200.0]
    assert potfile["sigma_nominal"] == pytest.approx(105.0)
    assert potfile["cutkpc_nominal"] == pytest.approx(20.5)


def test_load_best_par_preserves_repeated_limit_blocks_after_pluralization(tmp_path: Path) -> None:
    par_path = tmp_path / "repeated_limits.par"
    par_path.write_text(
        """
runmode
    reference 3 181.55062 -8.8009361
    end

cosmology
    H0 70.0
    omega 0.3
    lambda 0.7
    end

potentiel 1
    profil 81
    x_centre 0.0
    y_centre 0.0
    ellipticite 0.0
    angle_pos 0.0
    core_radius 1.0
    cut_radius 200.0
    v_disp 650.0
    z_lens 0.439
    end

limit 1
    v_disp 1 450.0 1200.0 0.1
    end

potentiel 2
    profil 81
    x_centre 10.0
    y_centre 0.0
    ellipticite 0.0
    angle_pos 0.0
    core_radius 1.0
    cut_radius 200.0
    v_disp 650.0
    z_lens 0.439
    end

limit 2
    x_centre 1 0.0 30.0 0.1
    end

potentiel 3
    profil 81
    x_centre -35.0
    y_centre -12.0
    ellipticite 0.0
    angle_pos 0.0
    core_radius 0.0
    cut_radius 200.0
    v_disp 650.0
    z_lens 0.439
    end

limit 3
    core_radius 1 0.0 40.0 0.1
    end

potentiel 4
    profil 14
    gamma 0.1
    angle_pos 0.0
    z_lens 0.439
    end

limit 4
    gamma 1 0.0 1.0 0.1
    end
fini
""",
        encoding="utf-8",
    )

    parsed, _potentials_df, _images_df, potentials_with_priors = load_best_par(par_path)

    assert "limit" not in parsed
    assert [item["id"] for item in parsed["limits"]] == ["1", "2", "3", "4"]
    priors_by_id = {item["id"]: item["priors"] for item in potentials_with_priors}
    assert "v_disp" in priors_by_id["1"]
    assert "x_centre" in priors_by_id["2"]
    assert "core_radius_kpc" in priors_by_id["3"]
    assert "gamma" in priors_by_id["4"]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("2.a", ("2", "a")),
        ("A.b", ("A", "b")),
        ("A200.ab", ("A200", "ab")),
        ("AB12.c", ("AB12", "c")),
    ],
)
def test_split_image_label_requires_strict_family_dot_image(label: str, expected: tuple[str, str]) -> None:
    assert _split_image_label(label) == expected


@pytest.mark.parametrize("label", ["1a", "A200a", "2.1a", "2.A", "fam.a", "candidate", "A-1.a"])
def test_split_image_label_rejects_invalid_labels(label: str) -> None:
    with pytest.raises(ValueError, match=r"expected FAMILY\.image"):
        _split_image_label(label)


def test_load_best_par_groups_strict_image_labels(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "1.a 10.0000 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "1.b 10.0002 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "1.c 10.0004 20.0000 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "strict_labels.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    end
fini
""",
        encoding="utf-8",
    )

    _parsed, _potentials_df, images_df, _potentials_with_priors = load_best_par(par_path)

    assert images_df["image_label"].astype(str).tolist() == ["1.a", "1.b", "1.c"]
    assert images_df["family_id"].astype(str).tolist() == ["1", "1", "1"]
    assert images_df["image_id"].astype(str).tolist() == ["a", "b", "c"]


def test_load_best_par_arcfile_reference3_merges_constraints(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 3\n"
        "1.a 0.0 0.0 1.0 1.0 0.0 2.0 25.0\n"
        "1.b 1.0 0.0 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    arc_catalog_path = tmp_path / "arc_constraints.dat"
    arc_catalog_path.write_text(
        "#REFERENCE 3\n"
        "1.a 0.25 -0.50 0.82 0.09 0.05 0.02 0.7\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "with_arcs.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    arcfile 1 arc_constraints.dat
    end
fini
""",
        encoding="utf-8",
    )

    _parsed, _potentials_df, images_df, _potentials_with_priors = load_best_par(par_path)

    assert images_df["arc_has_constraint"].astype(bool).tolist() == [True, False]
    first = images_df.iloc[0]
    x_anchor, y_anchor = cluster_solver._radec_to_offsets_arcsec(
        np.asarray([first["arc_anchor_ra"]], dtype=float),
        np.asarray([first["arc_anchor_dec"]], dtype=float),
        10.0,
        20.0,
    )
    assert float(x_anchor[0]) == pytest.approx(0.25)
    assert float(y_anchor[0]) == pytest.approx(-0.50)
    assert float(first["arc_tangent_angle_rad"]) == pytest.approx(0.82)
    assert float(first["arc_curvature_arcsec_inv"]) == pytest.approx(0.09)
    assert float(first["arc_sigma_tangent_angle_rad"]) == pytest.approx(0.05)
    assert float(first["arc_sigma_curvature_arcsec_inv"]) == pytest.approx(0.02)
    assert float(first["arc_reliability"]) == pytest.approx(0.7)


def test_load_best_par_arcfile_rejects_unknown_image_label(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "1.a 10.0000 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "1.b 10.0002 20.0000 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    arc_catalog_path = tmp_path / "arc_constraints.dat"
    arc_catalog_path.write_text(
        "#REFERENCE 0\n"
        "2.a 10.0000 20.0000 0.82 0.09 0.05 0.02 1.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "unknown_arc.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    arcfile 1 arc_constraints.dat
    end
fini
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown image_label"):
        load_best_par(par_path)


def test_cab_tangent_angle_residual_wraps_modulo_pi() -> None:
    residual = cluster_solver._cab_tangent_angle_residual(
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(np.pi - 0.05, dtype=jnp.float64),
    )

    assert float(residual) == pytest.approx(0.05)


def test_cab_morphology_loglike_rewards_matching_constraints_and_masks_rows() -> None:
    observed_angle = jnp.asarray([0.25, 1.0], dtype=jnp.float64)
    observed_curvature = jnp.asarray([0.08, 0.9], dtype=jnp.float64)
    sigma_angle = jnp.asarray([0.05, 0.05], dtype=jnp.float64)
    sigma_curvature = jnp.asarray([0.02, 0.02], dtype=jnp.float64)
    reliability = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    mask = jnp.asarray([True, False])
    exact = cluster_solver._cab_morphology_bin_loglike(
        predicted_tangent_angle_rad=observed_angle,
        predicted_curvature_arcsec_inv=observed_curvature,
        prediction_finite=jnp.asarray([True, True]),
        observed_tangent_angle_rad=observed_angle,
        observed_curvature_arcsec_inv=observed_curvature,
        sigma_tangent_angle_rad=sigma_angle,
        sigma_curvature_arcsec_inv=sigma_curvature,
        reliability=reliability,
        arc_has_constraint=mask,
    )
    perturbed = cluster_solver._cab_morphology_bin_loglike(
        predicted_tangent_angle_rad=observed_angle + jnp.asarray([0.10, 10.0], dtype=jnp.float64),
        predicted_curvature_arcsec_inv=observed_curvature + jnp.asarray([0.04, 10.0], dtype=jnp.float64),
        prediction_finite=jnp.asarray([True, True]),
        observed_tangent_angle_rad=observed_angle,
        observed_curvature_arcsec_inv=observed_curvature,
        sigma_tangent_angle_rad=sigma_angle,
        sigma_curvature_arcsec_inv=sigma_curvature,
        reliability=reliability,
        arc_has_constraint=mask,
    )
    no_constraints = cluster_solver._cab_morphology_bin_loglike(
        predicted_tangent_angle_rad=observed_angle + 10.0,
        predicted_curvature_arcsec_inv=observed_curvature + 10.0,
        prediction_finite=jnp.asarray([False, False]),
        observed_tangent_angle_rad=observed_angle,
        observed_curvature_arcsec_inv=observed_curvature,
        sigma_tangent_angle_rad=sigma_angle,
        sigma_curvature_arcsec_inv=sigma_curvature,
        reliability=reliability,
        arc_has_constraint=jnp.asarray([False, False]),
    )

    assert float(exact) > float(perturbed)
    assert float(no_constraints) == pytest.approx(0.0)


def test_cab_likelihood_weight_defaults_to_one_when_arcfile_is_declared() -> None:
    state_with_arcfile = SimpleNamespace(parsed={"image": {"arcfile": [1, "arcs.dat"]}}, family_data=[], bin_data=[])
    state_without_arcfile = SimpleNamespace(parsed={"image": {}}, family_data=[], bin_data=[])

    assert cluster_solver._effective_cab_likelihood_weight(None, state_with_arcfile) == pytest.approx(1.0)
    assert cluster_solver._effective_cab_likelihood_weight(None, state_without_arcfile) == pytest.approx(0.0)
    assert cluster_solver._effective_cab_likelihood_weight(0.25, state_with_arcfile) == pytest.approx(0.25)


def test_cab_arc_arrays_survive_family_bin_and_traced_conversion(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 3\n"
        "1.a 0.0 0.0 1.0 1.0 0.0 2.0 25.0\n"
        "1.b 1.0 0.0 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    arc_catalog_path = tmp_path / "arc_constraints.dat"
    arc_catalog_path.write_text(
        "#REFERENCE 3\n"
        "1.a 0.25 -0.50 0.82 0.09 0.05 0.02 0.7\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "with_arcs.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    arcfile 1 arc_constraints.dat
    end
fini
""",
        encoding="utf-8",
    )
    parsed, _potentials_df, images_df, _potentials_with_priors = load_best_par(par_path)
    reference = cluster_solver._extract_reference(parsed)
    families, _elapsed = cluster_solver._prepare_family_data(
        images_df,
        0.2,
        reference,
        z_lens=0.4,
        cosmo_config=cluster_solver._build_cosmology(parsed),
        z_bin_efficiency_tol=0.0,
    )
    bins = cluster_solver._build_bin_data(families)
    fake = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    fake.cosmology_effective_z_to_index = {}

    traced = cluster_solver.ClusterJAXEvaluator._prepare_traced_bin_data(fake, bins[0])

    assert families[0].arc_has_constraint.tolist() == [True, False]
    np.testing.assert_allclose(families[0].arc_anchor_x, [0.25, 1.0])
    np.testing.assert_allclose(families[0].arc_anchor_y, [-0.5, 0.0])
    assert bins[0].arc_has_constraint.tolist() == [True, False]
    assert traced.arc_constraint_count == 1
    np.testing.assert_allclose(np.asarray(traced.arc_tangent_angle_rad), [0.82, 0.0])
    np.testing.assert_allclose(np.asarray(traced.arc_curvature_arcsec_inv), [0.09, 0.0])


def test_load_best_par_rejects_invalid_image_label(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "2.1a 10.0000 20.0000 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "invalid_label.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    end
fini
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Invalid image label '2\.1a'.*expected FAMILY\.image"):
        load_best_par(par_path)


def test_load_best_par_allows_family_redshifts_equal_after_rounding(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "2.a 10.0000 20.0000 1.0 1.0 0.0 2.000000001 25.0\n"
        "2.b 10.0002 20.0000 1.0 1.0 0.0 2.000000002 25.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "consistent_redshift.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    end
fini
""",
        encoding="utf-8",
    )

    _parsed, _potentials_df, images_df, _potentials_with_priors = load_best_par(par_path)

    assert images_df["family_id"].astype(str).tolist() == ["2", "2"]
    assert images_df["image_id"].astype(str).tolist() == ["a", "b"]


def test_load_best_par_rejects_inconsistent_family_redshifts(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "2.a 10.0000 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "2.b 10.0002 20.0000 1.0 1.0 0.0 2.1 25.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "inconsistent_redshift.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    end
fini
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Family 2 has inconsistent catalog_z values \[2.0, 2.1\]"):
        load_best_par(par_path)


def test_load_best_par_rejects_cross_catalog_family_redshift_mismatch(tmp_path: Path) -> None:
    first_catalog_path = tmp_path / "obs_a.dat"
    first_catalog_path.write_text(
        "#REFERENCE 0\n"
        "2.a 10.0000 20.0000 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    second_catalog_path = tmp_path / "obs_b.dat"
    second_catalog_path.write_text(
        "#REFERENCE 0\n"
        "2.b 10.0002 20.0000 1.0 1.0 0.0 2.1 25.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "cross_catalog_inconsistent_redshift.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    image 1 obs_a.dat
    end

image
    multfile 1 obs_b.dat
    end
fini
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Family 2 has inconsistent catalog_z values \[2.0, 2.1\]"):
        load_best_par(par_path)


def test_prepare_family_data_rejects_inconsistent_family_redshift_backstop() -> None:
    images = pd.DataFrame(
        {
            "family_id": ["2", "2"],
            "image_label": ["2.a", "2.b"],
            "catalog_z": [2.0, 2.1],
            "catalog_source": ["manual", "manual"],
            "ra": [10.0, 10.0002],
            "dec": [20.0, 20.0],
        }
    )

    with pytest.raises(ValueError, match=r"Family 2 has inconsistent catalog_z values \[2.0, 2.1\]"):
        cluster_solver._prepare_family_data(
            images,
            sigma_arcsec=0.5,
            reference=(3, 10.0, 20.0),
            z_lens=0.3,
            cosmo_config={},
            z_bin_efficiency_tol=0.01,
        )


def test_filter_non_positive_redshift_families_keeps_singletons_separate() -> None:
    images = pd.DataFrame(
        {
            "family_id": ["valid", "valid", "zero", "zero", "negative", "negative", "single"],
            "image_label": ["v.1", "v.2", "z.1", "z.2", "n.1", "n.2", "s.1"],
            "catalog_z": [2.0, 2.0, 0.0, 0.0, -1.0, -1.0, 3.0],
        }
    )

    filtered, n_images, n_families, family_ids = cluster_solver._filter_non_positive_redshift_families(images)

    assert n_images == 4
    assert n_families == 2
    assert family_ids == ["negative", "zero"]
    assert filtered["family_id"].astype(str).tolist() == ["valid", "valid", "single"]

    singleton_filtered, n_singleton_images, n_singleton_families = cluster_solver._filter_singleton_families(filtered)
    assert n_singleton_images == 1
    assert n_singleton_families == 1
    assert singleton_filtered["family_id"].astype(str).tolist() == ["valid", "valid"]


def test_fov_limit_mask_combines_radius_and_order_insensitive_bounds() -> None:
    limit = cluster_solver._fov_limit_from_args(
        argparse.Namespace(
            fov_limit_radius=5.0,
            fov_limit_x=[2.0, -2.0],
            fov_limit_y=[1.0, -1.0],
        )
    )

    assert limit is not None
    mask = cluster_solver._fov_mask_from_offsets(
        np.asarray([0.0, 2.0, -2.0, 0.0, 3.0, 5.0]),
        np.asarray([0.0, 0.0, 0.0, 2.0, 4.0, 0.0]),
        limit,
    )

    assert mask.tolist() == [True, True, True, False, False, False]


def test_cluster_solver_parses_fov_limit_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "input.par",
            "--fov-limit-radius",
            "42",
            "--fov-limit-x",
            "10",
            "-10",
            "--fov-limit-y",
            "-5",
            "5",
        ],
    )

    args = _parse_args()

    assert args.fov_limit_radius == pytest.approx(42.0)
    assert args.fov_limit_x == [10.0, -10.0]
    assert args.fov_limit_y == [-5.0, 5.0]


def test_cluster_solver_parses_sampler_debug_diagnostics_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "input.par"])
    default_args = _parse_args()
    assert default_args.debug_sampler_diagnostics is False

    monkeypatch.setattr(
        sys,
        "argv",
        ["cluster_solver", "--par-path", "input.par", "--debug-sampler-diagnostics"],
    )
    debug_args = _parse_args()
    assert debug_args.debug_sampler_diagnostics is True


def test_cluster_solver_parses_fixed_image_sigma_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "input.par"])
    default_args = _parse_args()
    assert default_args.fix_image_sigma_int_arcsec is None

    monkeypatch.setattr(
        sys,
        "argv",
        ["cluster_solver", "--par-path", "input.par", "--fix-image-sigma-int-arcsec", "0.35"],
    )
    fixed_args = _parse_args()
    assert fixed_args.fix_image_sigma_int_arcsec == pytest.approx(0.35)

    monkeypatch.setattr(
        sys,
        "argv",
        ["cluster_solver", "--par-path", "input.par", "--fix-image-sigma-int-arcsec", "0.0"],
    )
    zero_args = _parse_args()
    assert zero_args.fix_image_sigma_int_arcsec == pytest.approx(0.0)

    for invalid_value in ("-0.1", "nan"):
        monkeypatch.setattr(
            sys,
            "argv",
            ["cluster_solver", "--par-path", "input.par", "--fix-image-sigma-int-arcsec", invalid_value],
        )
        with pytest.raises(SystemExit):
            _parse_args()


def test_cluster_solver_parses_dense_mass_boolean_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "input.par"])
    default_args = _parse_args()
    assert default_args.dense_mass is True

    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "input.par", "--no-dense-mass"])
    diagonal_args = _parse_args()
    assert diagonal_args.dense_mass is False

    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "input.par", "--dense-mass"])
    dense_args = _parse_args()
    assert dense_args.dense_mass is True


def test_cluster_solver_parses_potfile_mass_size_reparam_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "input.par"])
    default_args = _parse_args()
    assert default_args.potfile_mass_size_reparam is False

    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "input.par", "--potfile-mass-size-reparam"])
    reparam_args = _parse_args()
    assert reparam_args.potfile_mass_size_reparam is True


def test_multi_cluster_solver_parses_fov_limit_args() -> None:
    args = multi_cluster_solver._parse_args(
        [
            "--cluster",
            "a",
            "a.par",
            "warm/a",
            "--cluster",
            "b",
            "b.par",
            "warm/b",
            "--fov-limit-radius",
            "30",
            "--fov-limit-x",
            "12",
            "-12",
            "--fov-limit-y",
            "-8",
            "8",
        ]
    )

    assert args.fov_limit_radius == pytest.approx(30.0)
    assert args.fov_limit_x == [12.0, -12.0]
    assert args.fov_limit_y == [-8.0, 8.0]


def test_build_state_ignores_non_positive_redshift_families(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_catalog_path = tmp_path / "obs_arcs.cat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "1.a 10.0000 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "1.b 10.0002 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "2.a 10.0100 20.0000 1.0 1.0 0.0 0.0 25.0\n"
        "2.b 10.0102 20.0000 1.0 1.0 0.0 0.0 25.0\n"
        "3.a 10.0200 20.0000 1.0 1.0 0.0 3.0 25.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "input.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.cat
    end

cosmology
    H0 70.0
    omega 0.3
    lambda 0.7
    end

potentiel 1
    profil 81
    x_centre 0.0
    y_centre 0.0
    ellipticite 0.0
    angle_pos 0.0
    core_radius 1.0
    cut_radius 100.0
    v_disp 700.0
    z_lens 0.3
    end
fini
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", str(par_path), "--fit-mode", "large-only"])
    args = _parse_args()

    state = cluster_solver._build_state_from_inputs(args)

    assert [family.family_id for family in state.family_data] == ["1"]
    assert [family.n_images for family in state.family_data] == [2]


def test_build_state_applies_fov_to_images_and_potfile_members(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_catalog_path = tmp_path / "obs_arcs.cat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "1.a 10.0000 0.0000 1.0 1.0 0.0 2.0 25.0\n"
        "1.b 9.9995 0.0000 1.0 1.0 0.0 2.0 25.0\n"
        "2.a 10.0000 0.0002 1.0 1.0 0.0 3.0 25.0\n"
        "2.b 9.9970 0.0000 1.0 1.0 0.0 3.0 25.0\n",
        encoding="utf-8",
    )
    member_catalog_path = tmp_path / "members.cat"
    member_catalog_path.write_text(
        "#REFERENCE 0\n"
        "member-in 10.0000 0.0000 1.0 1.0 0.0 19.5 1.0\n"
        "member-out 9.9970 0.0000 1.0 1.0 0.0 19.5 1.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "input.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 0.0
    end

image
    multfile 1 obs_arcs.cat
    end

cosmology
    H0 70.0
    omega 0.3
    lambda 0.7
    end

potentiel 1
    profil 81
    x_centre 100.0
    y_centre 100.0
    ellipticite 0.0
    angle_pos 0.0
    core_radius 1.0
    cut_radius 100.0
    v_disp 700.0
    z_lens 0.3
    end

potfile
    filein 3 members.cat
    zlens 0.3
    type 81
    corekpc 0.15
    mag0 19.5
    sigma 1 10.0 200.0
    cutkpc 1 1.0 40.0
    vdslope 0 4.0 0.0
    slope 0 4.0 0.0
    end
fini
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            str(par_path),
            "--fit-mode",
            "joint",
            "--fov-limit-radius",
            "5.0",
        ],
    )
    args = _parse_args()

    state = cluster_solver._build_state_from_inputs(args)

    assert [family.family_id for family in state.family_data] == ["1"]
    assert [family.n_images for family in state.family_data] == [2]
    assert len(state.potfiles[0]["catalog_df"]) == 1
    assert state.potfiles[0]["catalog_df"].iloc[0]["id"] == "member-in"
    assert [record["catalog_id"] for record in state.scaling_component_records] == ["member-in"]
    assert state.base_components[0]["x_centre"] == pytest.approx(100.0)
    assert state.base_components[0]["y_centre"] == pytest.approx(100.0)


def test_generate_single_bcg_mock_with_subhalos_uses_potfile(tmp_path: Path) -> None:
    config = SingleBCGMockConfig(seed=11, n_primary_families=1, n_subhalos=8, pos_sigma_arcsec=0.0)

    paths, images, truth = generate_single_bcg_mock(tmp_path, config)
    parsed, _potentials_df, images_df, potentials_with_priors = load_best_par(paths.par_path)

    assert (tmp_path / "members.cat").exists()
    assert len(parsed["potfiles"]) == 1
    assert len(parsed["potfiles"][0]["catalog_df"]) == config.n_subhalos
    assert parsed["grille"]["nlens"] == 2 + config.n_subhalos
    assert parsed["grille"]["nlens_opt"] == 2
    image_radii = np.hypot(images["x_obs_arcsec"].to_numpy(dtype=float), images["y_obs_arcsec"].to_numpy(dtype=float))
    subhalo_radii = np.asarray(
        [np.hypot(row["x_arcsec"], row["y_arcsec"]) for row in truth["subhalos"]],
        dtype=float,
    )
    assert parsed["champ"]["dmax"] == int(np.ceil(max(image_radii.max(), subhalo_radii.max()) + 10.0))
    assert len(potentials_with_priors) == 2
    assert images_df["family_id"].nunique() == 1
    par_text = paths.par_path.read_text(encoding="utf-8")
    assert "vdslope 0 4.00000000 0" in par_text
    assert "slope 0 4.00000000 0" in par_text
    assert len(truth["subhalos"]) == config.n_subhalos
    assert len(truth["subhalo_components"]) == config.n_subhalos
    assert all(float(row["catalog_mag"]) <= config.subhalo_mag_faint_limit for row in truth["subhalos"])
    assert all("subhalo_mass_msun" in row for row in truth["subhalos"])
    assert all("subhalo_parent_rank" in row for row in truth["subhalos"])
    assert all(row["selected_by_mag_cut"] is True for row in truth["subhalos"])
    selection = truth["subhalo_selection"]
    assert selection["schechter_alpha"] == pytest.approx(config.subhalo_schechter_alpha)
    assert selection["luminosity_star"] == pytest.approx(1.0)
    assert selection["luminosity_peak"] == pytest.approx(config.subhalo_schechter_alpha + 1.0)
    assert selection["mass_luminosity_exponent"] == pytest.approx(1.0)
    assert selection["mass_peak"] == pytest.approx(
        config.subhalo_mass_ref * (config.subhalo_schechter_alpha + 1.0) ** selection["mass_luminosity_exponent"]
    )
    assert selection["mass_peak"] == pytest.approx(3.0e11)
    assert selection["parent_count"] >= config.n_subhalos
    assert selection["selected_count"] == config.n_subhalos
    candidates = selection["candidates"]
    selected_candidates = [row for row in candidates if row["selected"]]
    assert len(selected_candidates) == config.n_subhalos
    assert all(row["passes_mag_cut"] for row in selected_candidates)
    candidate_indices = {row["subhalo_candidate_index"] for row in candidates}
    selected_indices = {row["subhalo_candidate_index"] for row in selected_candidates}
    assert {row["subhalo_candidate_index"] for row in truth["subhalos"]} == selected_indices
    assert selected_indices <= candidate_indices
    assert {source["caustic_class"] for source in truth["sources"]} == {"primary"}
    assert np.isfinite(images["magnification_true"].to_numpy(dtype=float)).all()


def test_subhalo_schechter_sampler_has_log_space_peak_and_high_mass_tail() -> None:
    config = SingleBCGMockConfig()
    rng = np.random.default_rng(123)
    luminosity_min, luminosity_max = mock_cluster._subhalo_schechter_luminosity_bounds(config)

    luminosities = mock_cluster._sample_truncated_schechter_luminosities(
        rng,
        180_000,
        luminosity_min=luminosity_min,
        luminosity_max=luminosity_max,
        alpha=config.subhalo_schechter_alpha,
    )
    masses = mock_cluster._subhalo_mass_from_luminosity_ratio(luminosities, config)

    assert np.isfinite(luminosities).all()
    assert luminosities.min() >= luminosity_min
    assert luminosities.max() <= luminosity_max
    assert np.isfinite(masses).all()
    assert masses.min() >= config.subhalo_mass_min
    assert masses.max() <= config.subhalo_mass_max
    log_edges = np.linspace(np.log10(luminosity_min), np.log10(luminosity_max), 40)
    counts, edges = np.histogram(np.log10(luminosities), bins=log_edges)
    peak_center = 10.0 ** (0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]))

    assert peak_center == pytest.approx(config.subhalo_schechter_alpha + 1.0, rel=0.25)
    assert np.count_nonzero(masses > config.subhalo_mass_ref) > 0


def test_subhalo_schechter_selection_draws_from_observable_distribution() -> None:
    config = SingleBCGMockConfig(n_subhalos=8, subhalo_parent_factor=5)

    subhalos, selection = mock_cluster._generate_schechter_subhalo_candidates(config, np.random.default_rng(0))

    assert len(subhalos) == config.n_subhalos
    assert selection is not None
    candidates = selection["candidates"]
    selected = [row for row in candidates if row["selected"]]
    unselected_observable = [row for row in candidates if not row["selected"] and row["passes_mag_cut"]]
    assert len(selected) == config.n_subhalos
    assert all(row["catalog_mag"] <= config.subhalo_mag_faint_limit for row in selected)
    assert all(row["passes_mag_cut"] for row in selected)
    assert unselected_observable

    faintest_selected_mag = max(float(row["catalog_mag"]) for row in selected)
    brightest_unselected_mag = min(float(row["catalog_mag"]) for row in unselected_observable)
    assert brightest_unselected_mag < faintest_selected_mag


def test_subhalo_schechter_luminosity_to_mass_conversion_uses_dpie_scaling() -> None:
    config = SingleBCGMockConfig()
    exponent = mock_cluster._subhalo_mass_luminosity_exponent(config)
    luminosity_peak = config.subhalo_schechter_alpha + 1.0
    one_mag_fainter_luminosity = 10.0**-0.4
    luminosities = np.asarray([1.0, luminosity_peak, 10.0, one_mag_fainter_luminosity])

    masses = mock_cluster._subhalo_mass_from_luminosity_ratio(luminosities, config)
    mags = mock_cluster._subhalo_magnitude_from_luminosity_ratio(luminosities, config)

    assert config.subhalo_vdslope == pytest.approx(4.0)
    assert config.subhalo_slope == pytest.approx(4.0)
    assert exponent == pytest.approx(1.0)
    assert masses[0] == pytest.approx(config.subhalo_mass_ref)
    assert mags[0] == pytest.approx(config.subhalo_mag0)
    assert masses[1] == pytest.approx(config.subhalo_mass_ref * luminosity_peak**exponent)
    assert masses[2] > masses[0]
    assert mags[2] < mags[0]
    assert masses[3] / masses[0] == pytest.approx(10.0**-0.4)
    assert mags[3] == pytest.approx(config.subhalo_mag0 + 1.0)


def test_subhalo_schechter_selection_raises_when_mag_cut_rejects_candidates() -> None:
    config = SingleBCGMockConfig(
        n_primary_families=1,
        n_subhalos=3,
        subhalo_mag_faint_limit=-100.0,
    )

    with pytest.raises(RuntimeError, match="Failed to select 3 Schechter subhalos"):
        mock_cluster._generate_subhalo_catalog(config, np.random.default_rng(5))


def test_plot_subhalo_shmf_writes_pdf(tmp_path: Path) -> None:
    truth = {
        "subhalo_selection": {
            "schechter_alpha": -0.7,
            "mass_ref": 1.0e12,
            "mass_luminosity_exponent": 1.2,
            "mag_faint_limit": 24.0,
            "candidates": [
                {
                    "subhalo_candidate_index": 0,
                    "subhalo_mass_msun": 1.0e12,
                    "catalog_mag": 17.0,
                    "luminosity_ratio": 1.0,
                    "subhalo_parent_rank": 1,
                    "passes_mag_cut": True,
                    "selected": True,
                    "selected_member_id": "member001",
                },
                {
                    "subhalo_candidate_index": 1,
                    "subhalo_mass_msun": 2.4e11,
                    "catalog_mag": 18.3,
                    "luminosity_ratio": 0.3,
                    "subhalo_parent_rank": 2,
                    "passes_mag_cut": True,
                    "selected": False,
                },
                {
                    "subhalo_candidate_index": 2,
                    "subhalo_mass_msun": 1.0e9,
                    "catalog_mag": 25.4,
                    "luminosity_ratio": 0.003,
                    "subhalo_parent_rank": 3,
                    "passes_mag_cut": False,
                    "selected": False,
                },
            ],
        }
    }
    path = tmp_path / "subhalo_shmf.pdf"

    validation._plot_subhalo_selection(truth, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_subhalo_shmf_writes_placeholder_without_data(tmp_path: Path) -> None:
    path = tmp_path / "subhalo_shmf.pdf"

    validation._plot_subhalo_selection({}, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_recovered_subhalo_mass_table_uses_recovered_scaling_parameters() -> None:
    packed = SimpleNamespace(
        component_family=np.asarray([0, 1, 1], dtype=int),
        x_center_base=np.asarray([0.0, -3.0, 6.0], dtype=float),
        y_center_base=np.asarray([0.0, 4.0, 8.0], dtype=float),
        luminosity_ratio=np.asarray([1.0, 1.0, 0.25], dtype=float),
        sigma_ref_base=np.asarray([0.0, 245.0, 245.0], dtype=float),
        cut_ref_base=np.asarray([0.0, 40.0, 40.0], dtype=float),
        vdslope_base=np.asarray([0.0, 4.0, 4.0], dtype=float),
        slope_base=np.asarray([0.0, 4.0, 4.0], dtype=float),
        sigma_ref_param_index=np.asarray([-1, 0, 0], dtype=int),
        cut_ref_param_index=np.asarray([-1, 1, 1], dtype=int),
        vdslope_param_index=np.asarray([-1, 2, 2], dtype=int),
        slope_param_index=np.asarray([-1, 3, 3], dtype=int),
    )
    state = SimpleNamespace(
        packed_lens_spec=packed,
        scaling_component_records=[
            {
                "component_index": 1,
                "potfile_id": "potfile",
                "catalog_id": "member001",
                "x_centre": 3.0,
                "y_centre": 4.0,
            },
            {"component_index": 2, "potfile_id": "potfile", "catalog_id": "member002"},
        ],
    )
    best_fit = np.asarray([300.0, 60.0, 5.0, 4.0], dtype=float)
    truth = {
        "parameter_truth": {"potfile.sigma": 245.0, "potfile.cutkpc": 40.0},
        "subhalo_selection": {"mass_ref": 1.0e12},
    }

    table = validation._recovered_subhalo_mass_table(state, best_fit, truth)

    normalization = (300.0 / 245.0) ** 2 * (60.0 / 40.0)
    exponent = 2.0 / 5.0 + 2.0 / 4.0
    assert table["catalog_id"].tolist() == ["member001", "member002"]
    assert table["recovered_subhalo_mass_msun"].tolist() == pytest.approx(
        [
            1.0e12 * normalization,
            1.0e12 * normalization * 0.25**exponent,
        ]
    )
    assert table["x_arcsec"].tolist() == pytest.approx([3.0, 6.0])
    assert table["y_arcsec"].tolist() == pytest.approx([4.0, 8.0])
    assert table["recovered_radius_arcsec"].tolist() == pytest.approx([5.0, 10.0])


def test_plot_absolute_magnification_recovery_writes_pdf(tmp_path: Path) -> None:
    truth_abs_mu_raw = np.abs(
        np.asarray(
            [
                [1.0, -8.0, 120.0],
                [3.0, 12.0, 25.0],
                [np.nan, 45.0, 70.0],
            ]
        )
    )
    recovered_abs_mu_raw = np.abs(
        np.asarray(
            [
                [2.0, -10.0, 100.0],
                [5.0, 8.0, 20.0],
                [4.0, np.inf, 60.0],
            ]
        )
    )
    truth_abs_mu = np.minimum(truth_abs_mu_raw, 25.0)
    recovered_abs_mu = np.minimum(recovered_abs_mu_raw, 25.0)
    grid = validation._AbsoluteMagnificationRecoveryGrid(
        x_axis_arcsec=np.asarray([-1.0, 0.0, 1.0]),
        y_axis_arcsec=np.asarray([-1.0, 0.0, 1.0]),
        truth_abs_mu_raw=truth_abs_mu_raw,
        recovered_abs_mu_raw=recovered_abs_mu_raw,
        truth_abs_mu=truth_abs_mu,
        recovered_abs_mu=recovered_abs_mu,
        residual_abs_mu=recovered_abs_mu_raw - truth_abs_mu_raw,
        z_source=7.0,
        cap=25.0,
    )
    path = tmp_path / "absolute_magnification_recovery.pdf"

    validation._plot_absolute_magnification_recovery(grid, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_absolute_magnification_recovery_uses_capped_viridis_and_centered_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import TwoSlopeNorm

    truth_abs_mu_raw = np.abs(np.asarray([[5.0, -80.0], [20.0, np.nan]]))
    recovered_abs_mu_raw = np.abs(np.asarray([[10.0, 100.0], [-5.0, np.inf]]))
    truth_abs_mu = np.minimum(truth_abs_mu_raw, 25.0)
    recovered_abs_mu = np.minimum(recovered_abs_mu_raw, 25.0)
    residual_abs_mu = recovered_abs_mu_raw - truth_abs_mu_raw
    grid = validation._AbsoluteMagnificationRecoveryGrid(
        x_axis_arcsec=np.asarray([-1.0, 1.0]),
        y_axis_arcsec=np.asarray([-1.0, 1.0]),
        truth_abs_mu_raw=truth_abs_mu_raw,
        recovered_abs_mu_raw=recovered_abs_mu_raw,
        truth_abs_mu=truth_abs_mu,
        recovered_abs_mu=recovered_abs_mu,
        residual_abs_mu=residual_abs_mu,
        z_source=7.0,
        cap=25.0,
    )
    imshow_calls: list[tuple[np.ndarray, dict[str, Any]]] = []
    colorbar_labels: list[str] = []
    subplot_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    layout_axes: list[Axes] = []
    xlabels: list[tuple[Axes, str]] = []
    ylabels: list[tuple[Axes, str]] = []
    tick_params_calls: list[tuple[Axes, dict[str, Any]]] = []
    original_subplots = validation.plt.subplots
    original_imshow = Axes.imshow
    original_colorbar_set_label = Colorbar.set_label
    original_set_xlabel = Axes.set_xlabel
    original_set_ylabel = Axes.set_ylabel
    original_tick_params = Axes.tick_params

    def record_subplots(*args: Any, **kwargs: Any) -> Any:
        subplot_calls.append((args, dict(kwargs)))
        fig, axes = original_subplots(*args, **kwargs)
        layout_axes.extend(np.ravel(axes).tolist())
        return fig, axes

    def record_imshow(self: Axes, data: Any, *args: Any, **kwargs: Any) -> Any:
        imshow_calls.append((np.ma.asarray(data).filled(np.nan), dict(kwargs)))
        return original_imshow(self, data, *args, **kwargs)

    def fail_set_title(self: Axes, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("absolute magnification recovery plot should not draw panel titles")

    def record_colorbar_label(self: Colorbar, label: str, *args: Any, **kwargs: Any) -> Any:
        colorbar_labels.append(str(label))
        return original_colorbar_set_label(self, label, *args, **kwargs)

    def record_set_xlabel(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        xlabels.append((self, str(label)))
        return original_set_xlabel(self, label, *args, **kwargs)

    def record_set_ylabel(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        ylabels.append((self, str(label)))
        return original_set_ylabel(self, label, *args, **kwargs)

    def record_tick_params(self: Axes, *args: Any, **kwargs: Any) -> Any:
        tick_params_calls.append((self, dict(kwargs)))
        return original_tick_params(self, *args, **kwargs)

    monkeypatch.setattr(validation.plt, "subplots", record_subplots)
    monkeypatch.setattr(Axes, "imshow", record_imshow)
    monkeypatch.setattr(Axes, "set_title", fail_set_title)
    monkeypatch.setattr(Colorbar, "set_label", record_colorbar_label)
    monkeypatch.setattr(Axes, "set_xlabel", record_set_xlabel)
    monkeypatch.setattr(Axes, "set_ylabel", record_set_ylabel)
    monkeypatch.setattr(Axes, "tick_params", record_tick_params)

    validation._plot_absolute_magnification_recovery(grid, tmp_path / "absolute_magnification_recovery.pdf")

    assert subplot_calls == [((2, 1), {"sharex": True, "figsize": (6.2, 8.6)})]
    assert len(layout_axes) == 2
    assert len(imshow_calls) == 2
    left_data, left_kwargs = imshow_calls[0]
    right_data, right_kwargs = imshow_calls[1]
    assert np.nanmax(left_data) <= 25.0
    assert left_kwargs["cmap"] == "viridis"
    assert left_kwargs["vmin"] == 0.0
    assert left_kwargs["vmax"] == 25.0
    np.testing.assert_allclose(right_data, residual_abs_mu, equal_nan=True)
    assert right_kwargs["cmap"] == "RdBu"
    assert isinstance(right_kwargs["norm"], TwoSlopeNorm)
    assert right_kwargs["norm"].vmin == pytest.approx(-25.0)
    assert right_kwargs["norm"].vcenter == pytest.approx(0.0)
    assert right_kwargs["norm"].vmax == pytest.approx(25.0)
    nonempty_colorbar_labels = [label for label in colorbar_labels if label]
    assert nonempty_colorbar_labels == [
        r"$|\mu_{\rm rec}|$",
        r"$|\mu_{\rm rec}| - |\mu_{\rm truth}|$",
    ]
    assert not any("capped at" in label for label in nonempty_colorbar_labels)
    assert not any("min(" in label for label in nonempty_colorbar_labels)
    panel_xlabels = [(ax, label) for ax, label in xlabels if ax in layout_axes]
    panel_ylabels = [(ax, label) for ax, label in ylabels if ax in layout_axes]
    assert layout_axes[0].get_xlabel() == ""
    assert layout_axes[1].get_xlabel() == "x [arcsec]"
    assert not any(ax is layout_axes[0] and label == "x [arcsec]" for ax, label in panel_xlabels)
    assert any(ax is layout_axes[1] and label == "x [arcsec]" for ax, label in panel_xlabels)
    assert panel_ylabels == [(layout_axes[0], "y [arcsec]"), (layout_axes[1], "y [arcsec]")]
    assert any(
        ax is layout_axes[0]
        and kwargs.get("axis") == "x"
        and kwargs.get("labelbottom") is False
        for ax, kwargs in tick_params_calls
    )


def test_absolute_magnification_grid_source_redshift_prefers_z9_then_highest() -> None:
    assert validation._absolute_magnification_grid_source_redshift(
        {"kwargs_lens_by_source_redshift": {"2.00000000": [], "9.00000000": []}}
    ) == pytest.approx(9.0)
    assert validation._absolute_magnification_grid_source_redshift(
        {"kwargs_lens_by_source_redshift": {"2.00000000": [], "5.00000000": []}}
    ) == pytest.approx(5.0)
    assert validation._absolute_magnification_grid_source_redshift(
        {"config": {"source_redshift": 3.0}}
    ) == pytest.approx(3.0)


def test_plot_image_residual_histogram_writes_pdf(tmp_path: Path) -> None:
    image_df = pd.DataFrame({"image_residual_arcsec": [0.02, 0.05, 0.08, 0.13]})
    path = tmp_path / "image_residual_histogram.pdf"

    validation._plot_image_residual_histogram(image_df, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_image_residual_histogram_uses_q50_and_filters_nonfinite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    image_df = pd.DataFrame(
        {
            "image_residual_arcsec": [9.0, 9.0, 9.0, 9.0, 9.0],
            "image_residual_q50": [0.04, np.nan, 0.08, np.inf, 0.12],
        }
    )
    hist_values: list[np.ndarray] = []
    hist_bins: list[Any] = []
    axvline_values: list[float] = []
    xlabels: list[str] = []
    ylabels: list[str] = []
    original_hist = Axes.hist
    original_axvline = Axes.axvline
    original_set_xlabel = Axes.set_xlabel
    original_set_ylabel = Axes.set_ylabel

    def record_hist(self: Axes, values: Any, *args: Any, **kwargs: Any) -> Any:
        hist_values.append(np.asarray(values, dtype=float))
        hist_bins.append(kwargs.get("bins"))
        return original_hist(self, values, *args, **kwargs)

    def record_axvline(self: Axes, x: Any = 0, *args: Any, **kwargs: Any) -> Any:
        axvline_values.append(float(x))
        return original_axvline(self, x, *args, **kwargs)

    def record_set_xlabel(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        xlabels.append(str(label))
        return original_set_xlabel(self, label, *args, **kwargs)

    def record_set_ylabel(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        ylabels.append(str(label))
        return original_set_ylabel(self, label, *args, **kwargs)

    def fail_set_title(self: Axes, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("image residual histogram should not draw a title")

    monkeypatch.setattr(Axes, "hist", record_hist)
    monkeypatch.setattr(Axes, "axvline", record_axvline)
    monkeypatch.setattr(Axes, "set_xlabel", record_set_xlabel)
    monkeypatch.setattr(Axes, "set_ylabel", record_set_ylabel)
    monkeypatch.setattr(Axes, "set_title", fail_set_title)

    validation._plot_image_residual_histogram(image_df, tmp_path / "image_residual_histogram.pdf")

    assert len(hist_values) == 1
    np.testing.assert_allclose(hist_values[0], [0.04, 9.0, 0.08, 9.0, 0.12])
    assert hist_bins == [30]
    assert any(value == pytest.approx(0.12) for value in axvline_values)
    assert any(value == pytest.approx(float(np.sqrt(np.mean(np.square(hist_values[0]))))) for value in axvline_values)
    assert xlabels == ["image residual [arcsec]"]
    assert ylabels == ["N images"]


def test_plot_image_recovery_value_fallback_uses_best_fit_for_nan_q50() -> None:
    image_df = pd.DataFrame(
        {
            "x_model_arcsec": [1.0, 2.0, 3.0],
            "x_model_q50": [np.nan, 20.0, np.inf],
        }
    )

    values = validation._plot_value_with_fallback(image_df, "x_model_q50", "x_model_arcsec")

    np.testing.assert_allclose(values, [1.0, 20.0, 3.0])


def test_plot_image_residual_histogram_writes_placeholder_without_finite_values(tmp_path: Path) -> None:
    image_df = pd.DataFrame({"image_residual_arcsec": [np.nan, np.inf, -np.inf]})
    path = tmp_path / "image_residual_histogram.pdf"

    validation._plot_image_residual_histogram(image_df, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_critical_arc_support_histogram_uses_q50_and_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    image_df = pd.DataFrame(
        {
            "image_residual_arcsec": [9.0, 9.0],
            "image_residual_q50": [0.4, 0.8],
            "arc_aware_image_residual_arcsec": [0.3, 0.2],
            "arc_aware_image_residual_q50": [0.02, np.nan],
            "arc_curve_distance_arcsec": [0.04, 0.45],
            "arc_noncritical_direction_residual_arcsec": [0.01, 0.25],
            "arc_critical_direction_residual_arcsec": [4.0, 0.5],
            "arc_s_min": [0.04, 0.5],
            "arc_prior_probability": [0.75, 0.12],
            "arc_recovery_status": ["arc_supported", "not_recovered"],
        }
    )
    hist_values: list[np.ndarray] = []
    axvline_values: list[float] = []
    original_hist = Axes.hist
    original_axvline = Axes.axvline

    def record_hist(self: Axes, values: Any, *args: Any, **kwargs: Any) -> Any:
        hist_values.append(np.asarray(values, dtype=float))
        return original_hist(self, values, *args, **kwargs)

    def record_axvline(self: Axes, x: Any = 0, *args: Any, **kwargs: Any) -> Any:
        axvline_values.append(float(x))
        return original_axvline(self, x, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", record_hist)
    monkeypatch.setattr(Axes, "axvline", record_axvline)

    validation._plot_critical_arc_support_histogram(
        image_df,
        tmp_path / "critical_arc_support_histogram.pdf",
    )

    assert any(np.allclose(values, [0.02, 0.2]) for values in hist_values)
    assert any(np.allclose(values, [0.04, 0.45]) for values in hist_values)
    assert any(value == pytest.approx(0.5) for value in axvline_values)
    assert any(value == pytest.approx(0.2) for value in axvline_values)


def test_plot_critical_arc_support_phase_space_draws_support_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    image_df = pd.DataFrame(
        {
            "arc_s_min": [0.04, 0.5, np.nan],
            "arc_curve_distance_arcsec": [0.04, 0.45, 0.1],
            "arc_noncritical_direction_residual_arcsec": [0.03, 0.6, 0.1],
            "arc_critical_direction_residual_arcsec": [5.0, 0.1, 1.0],
            "arc_recovery_status": ["arc_supported", "not_recovered", "point_recovered"],
        }
    )
    scatter_xy: list[tuple[np.ndarray, np.ndarray]] = []
    axvline_values: list[float] = []
    axhline_values: list[float] = []
    original_scatter = Axes.scatter
    original_axvline = Axes.axvline
    original_axhline = Axes.axhline

    def record_scatter(self: Axes, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        scatter_xy.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        return original_scatter(self, x, y, *args, **kwargs)

    def record_axvline(self: Axes, x: Any = 0, *args: Any, **kwargs: Any) -> Any:
        axvline_values.append(float(x))
        return original_axvline(self, x, *args, **kwargs)

    def record_axhline(self: Axes, y: Any = 0, *args: Any, **kwargs: Any) -> Any:
        axhline_values.append(float(y))
        return original_axhline(self, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", record_scatter)
    monkeypatch.setattr(Axes, "axvline", record_axvline)
    monkeypatch.setattr(Axes, "axhline", record_axhline)

    validation._plot_critical_arc_support_phase_space(
        image_df,
        tmp_path / "critical_arc_support_phase_space.pdf",
    )

    all_x = np.concatenate([item[0] for item in scatter_xy])
    all_y = np.concatenate([item[1] for item in scatter_xy])
    np.testing.assert_allclose(np.sort(all_x), [0.04, 0.5])
    np.testing.assert_allclose(np.sort(all_y), [0.04, 0.45])
    assert axvline_values == [pytest.approx(0.2)]
    assert axhline_values == [pytest.approx(0.5)]


def test_plot_critical_arc_support_placeholder_without_finite_values(tmp_path: Path) -> None:
    image_df = pd.DataFrame({"image_residual_arcsec": [0.1], "arc_s_min": [np.nan]})
    path = tmp_path / "critical_arc_support_histogram.pdf"

    validation._plot_critical_arc_support_histogram(image_df, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_subhalo_recovery_shmf_writes_pdf(tmp_path: Path) -> None:
    truth = {
        "subhalos": [
            {"subhalo_mass_msun": 1.0e11},
            {"subhalo_mass_msun": 3.0e11},
            {"subhalo_mass_msun": 1.2e12},
        ]
    }
    recovered = pd.DataFrame(
        {
            "recovered_subhalo_mass_msun": [1.2e11, 2.5e11, 9.0e11],
        }
    )
    path = tmp_path / "subhalo_recovery_shmf.pdf"

    validation._plot_subhalo_recovery_shmf(truth, recovered, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_subhalo_recovery_shmf_uses_truth_schechter_without_parent_or_rugs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = {
        "subhalos": [
            {"subhalo_mass_msun": 1.0e11},
            {"subhalo_mass_msun": 3.0e11},
            {"subhalo_mass_msun": 1.2e12},
        ],
        "subhalo_selection": {
            "schechter_alpha": -0.7,
            "mass_ref": 1.0e12,
            "mass_luminosity_exponent": 1.0,
            "candidates": [
                {"subhalo_mass_msun": 5.0e10},
                {"subhalo_mass_msun": 1.0e11},
                {"subhalo_mass_msun": 3.0e11},
                {"subhalo_mass_msun": 1.2e12},
                {"subhalo_mass_msun": 2.0e12},
            ],
        },
    }
    recovered = pd.DataFrame(
        {
            "recovered_subhalo_mass_msun": [1.2e11, 2.5e11, 9.0e11],
        }
    )
    hist_kwargs: list[dict[str, Any]] = []
    plot_labels: list[str | None] = []
    schechter_integrals: list[float] = []
    xlabel_fontsizes: list[float | None] = []
    ylabel_fontsizes: list[float | None] = []
    legend_fontsizes: list[float | None] = []
    legend_locs: list[str | None] = []
    minor_locator_args: list[tuple[Any, ...]] = []
    original_hist = validation.plt.Axes.hist
    original_plot = validation.plt.Axes.plot
    original_set_xlabel = validation.plt.Axes.set_xlabel
    original_set_ylabel = validation.plt.Axes.set_ylabel
    original_legend = validation.plt.Axes.legend
    original_minor_locator = validation.AutoMinorLocator

    def record_hist(self: Any, *args: Any, **kwargs: Any) -> Any:
        hist_kwargs.append(dict(kwargs))
        return original_hist(self, *args, **kwargs)

    def record_plot(self: Any, *args: Any, **kwargs: Any) -> Any:
        plot_labels.append(kwargs.get("label"))
        if len(args) >= 2:
            schechter_integrals.append(float(np.trapezoid(np.asarray(args[1], dtype=float), np.asarray(args[0], dtype=float))))
        return original_plot(self, *args, **kwargs)

    def fail_scatter(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subhalo recovery SHMF should not draw scatter rug markers")

    def fail_title(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subhalo recovery SHMF should not draw a title")

    def record_set_xlabel(self: Any, *args: Any, **kwargs: Any) -> Any:
        xlabel_fontsizes.append(kwargs.get("fontsize"))
        return original_set_xlabel(self, *args, **kwargs)

    def record_set_ylabel(self: Any, *args: Any, **kwargs: Any) -> Any:
        ylabel_fontsizes.append(kwargs.get("fontsize"))
        return original_set_ylabel(self, *args, **kwargs)

    def record_legend(self: Any, *args: Any, **kwargs: Any) -> Any:
        legend_fontsizes.append(kwargs.get("fontsize"))
        legend_locs.append(kwargs.get("loc"))
        return original_legend(self, *args, **kwargs)

    def record_minor_locator(*args: Any, **kwargs: Any) -> Any:
        minor_locator_args.append(args)
        return original_minor_locator(*args, **kwargs)

    monkeypatch.setattr(validation.plt.Axes, "hist", record_hist)
    monkeypatch.setattr(validation.plt.Axes, "plot", record_plot)
    monkeypatch.setattr(validation.plt.Axes, "scatter", fail_scatter)
    monkeypatch.setattr(validation.plt.Axes, "set_title", fail_title)
    monkeypatch.setattr(validation.plt.Axes, "set_xlabel", record_set_xlabel)
    monkeypatch.setattr(validation.plt.Axes, "set_ylabel", record_set_ylabel)
    monkeypatch.setattr(validation.plt.Axes, "legend", record_legend)
    monkeypatch.setattr(validation, "AutoMinorLocator", record_minor_locator)

    path = tmp_path / "subhalo_recovery_shmf.pdf"

    validation._plot_subhalo_recovery_shmf(truth, recovered, path)

    hist_labels = [kwargs.get("label") for kwargs in hist_kwargs]
    assert "parent candidates" not in hist_labels
    assert "Truth subhalos" in hist_labels
    truth_hist = next(kwargs for kwargs in hist_kwargs if kwargs.get("label") == "Truth subhalos")
    assert truth_hist["color"] == "lightgray"
    assert truth_hist["alpha"] == pytest.approx(0.8)
    assert any(label is not None and label.startswith("Schechter") for label in plot_labels)
    assert schechter_integrals == [pytest.approx(3.0)]
    assert minor_locator_args == [(5,)]
    assert xlabel_fontsizes == [14]
    assert ylabel_fontsizes == [14]
    assert legend_fontsizes == [11]
    assert legend_locs == ["upper left"]
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_subhalo_recovery_radial_writes_pdf(tmp_path: Path) -> None:
    truth = {
        "subhalos": [
            {"x_arcsec": 3.0, "y_arcsec": 4.0},
            {"x_arcsec": 8.0, "y_arcsec": 6.0},
            {"x_arcsec": 12.0, "y_arcsec": 16.0},
        ]
    }
    recovered = pd.DataFrame(
        {
            "recovered_radius_arcsec": [6.0, 12.0, 18.0],
        }
    )
    path = tmp_path / "subhalo_recovery_radial.pdf"

    validation._plot_subhalo_recovery_radial(truth, recovered, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_subhalo_recovery_radial_uses_truth_style_without_parent_or_rugs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = {
        "subhalos": [
            {"x_arcsec": 3.0, "y_arcsec": 4.0},
            {"x_arcsec": 8.0, "y_arcsec": 6.0},
            {"x_arcsec": 12.0, "y_arcsec": 16.0},
        ],
        "subhalo_selection": {
            "candidates": [
                {"subhalo_mass_msun": 5.0e10},
                {"subhalo_mass_msun": 1.0e11},
            ],
        },
    }
    recovered = pd.DataFrame(
        {
            "recovered_radius_arcsec": [6.0, 12.0, 18.0],
        }
    )
    hist_kwargs: list[dict[str, Any]] = []
    xlabel_fontsizes: list[float | None] = []
    ylabel_fontsizes: list[float | None] = []
    legend_fontsizes: list[float | None] = []
    legend_locs: list[str | None] = []
    minor_locator_args: list[tuple[Any, ...]] = []
    original_hist = validation.plt.Axes.hist
    original_set_xlabel = validation.plt.Axes.set_xlabel
    original_set_ylabel = validation.plt.Axes.set_ylabel
    original_legend = validation.plt.Axes.legend
    original_minor_locator = validation.AutoMinorLocator

    def record_hist(self: Any, *args: Any, **kwargs: Any) -> Any:
        hist_kwargs.append(dict(kwargs))
        return original_hist(self, *args, **kwargs)

    def fail_scatter(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subhalo radial recovery should not draw scatter rug markers")

    def fail_title(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subhalo radial recovery should not draw a title")

    def fail_plot(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subhalo radial recovery should not draw line plots")

    def record_set_xlabel(self: Any, *args: Any, **kwargs: Any) -> Any:
        xlabel_fontsizes.append(kwargs.get("fontsize"))
        return original_set_xlabel(self, *args, **kwargs)

    def record_set_ylabel(self: Any, *args: Any, **kwargs: Any) -> Any:
        ylabel_fontsizes.append(kwargs.get("fontsize"))
        return original_set_ylabel(self, *args, **kwargs)

    def record_legend(self: Any, *args: Any, **kwargs: Any) -> Any:
        legend_fontsizes.append(kwargs.get("fontsize"))
        legend_locs.append(kwargs.get("loc"))
        return original_legend(self, *args, **kwargs)

    def record_minor_locator(*args: Any, **kwargs: Any) -> Any:
        minor_locator_args.append(args)
        return original_minor_locator(*args, **kwargs)

    monkeypatch.setattr(validation.plt.Axes, "hist", record_hist)
    monkeypatch.setattr(validation.plt.Axes, "plot", fail_plot)
    monkeypatch.setattr(validation.plt.Axes, "scatter", fail_scatter)
    monkeypatch.setattr(validation.plt.Axes, "set_title", fail_title)
    monkeypatch.setattr(validation.plt.Axes, "set_xlabel", record_set_xlabel)
    monkeypatch.setattr(validation.plt.Axes, "set_ylabel", record_set_ylabel)
    monkeypatch.setattr(validation.plt.Axes, "legend", record_legend)
    monkeypatch.setattr(validation, "AutoMinorLocator", record_minor_locator)

    path = tmp_path / "subhalo_recovery_radial.pdf"

    validation._plot_subhalo_recovery_radial(truth, recovered, path)

    hist_labels = [kwargs.get("label") for kwargs in hist_kwargs]
    assert "parent candidates" not in hist_labels
    assert "Truth subhalos" in hist_labels
    assert "recovered subhalos" in hist_labels
    truth_hist = next(kwargs for kwargs in hist_kwargs if kwargs.get("label") == "Truth subhalos")
    recovered_hist = next(kwargs for kwargs in hist_kwargs if kwargs.get("label") == "recovered subhalos")
    assert truth_hist["color"] == "lightgray"
    assert truth_hist["alpha"] == pytest.approx(0.8)
    assert recovered_hist["histtype"] == "step"
    assert recovered_hist["color"] == "tab:blue"
    assert minor_locator_args == [(5,)]
    assert xlabel_fontsizes == [14]
    assert ylabel_fontsizes == [14]
    assert legend_fontsizes == [11]
    assert legend_locs == ["upper left"]
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_prefit_subhalo_spatial_distribution_writes_partial_pdf(tmp_path: Path) -> None:
    subhalo_df = pd.DataFrame(
        {
            "x_arcsec": [2.5, -1.0],
            "y_arcsec": [-0.5, 1.5],
            "luminosity_ratio": [1.0, 0.25],
            "catalog_mag": [17.0, 19.0],
        }
    )
    images = pd.DataFrame({"x_obs_arcsec": [-1.0, 1.0], "y_obs_arcsec": [0.0, 0.5]})
    path = tmp_path / "prefit_subhalo_spatial_distribution.pdf"

    validation._plot_prefit_subhalo_spatial_distribution(subhalo_df, images, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_select_prefit_critical_line_contours_prefers_z9_then_highest() -> None:
    z2_contour = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.1, 0.0, 0.1]),
        beta_y=np.asarray([0.0, 0.1, 0.0]),
        critical_x=np.asarray([-1.0, 0.0, 1.0]),
        critical_y=np.asarray([0.0, 1.0, 0.0]),
        caustic_area_arcsec2=0.01,
        critical_area_arcsec2=1.0,
    )
    z9_contour = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.2, 0.0, 0.2]),
        beta_y=np.asarray([0.0, 0.2, 0.0]),
        critical_x=np.asarray([-2.0, 0.0, 2.0]),
        critical_y=np.asarray([0.0, 2.0, 0.0]),
        caustic_area_arcsec2=0.04,
        critical_area_arcsec2=4.0,
    )

    selected = validation._select_prefit_critical_line_contours(
        {"2.00000000": [z2_contour], "9.00000000": [z9_contour]}
    )
    fallback = validation._select_prefit_critical_line_contours(
        {"2.00000000": [z2_contour], "5.00000000": [z9_contour]}
    )

    assert list(selected) == ["9.00000000"]
    assert selected["9.00000000"][0] is z9_contour
    assert list(fallback) == ["5.00000000"]
    assert fallback["5.00000000"][0] is z9_contour


def test_plot_prefit_critical_lines_writes_pdf(tmp_path: Path) -> None:
    contour_payload = {
        "caustic_index": 0,
        "caustic_class": "primary",
        "critical_x": [-2.0, 0.0, 2.0, -2.0],
        "critical_y": [0.0, 2.0, 0.0, 0.0],
        "caustic_beta_x": [-0.2, 0.0, 0.2, -0.2],
        "caustic_beta_y": [0.0, 0.2, 0.0, 0.0],
        "caustic_area_arcsec2": 0.04,
        "critical_area_arcsec2": 4.0,
    }
    path = tmp_path / "prefit_critical_lines.pdf"

    validation._plot_prefit_critical_lines({"caustics_by_source_redshift": {"9.00000000": [contour_payload]}}, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_write_prefit_validation_diagnostics_writes_all_outputs(tmp_path: Path) -> None:
    contour_payload = {
        "caustic_index": 0,
        "caustic_class": "primary",
        "critical_x": [-2.0, 0.0, 2.0, -2.0],
        "critical_y": [0.0, 2.0, 0.0, 0.0],
        "caustic_beta_x": [-0.2, 0.0, 0.2, -0.2],
        "caustic_beta_y": [0.0, 0.2, 0.0, 0.0],
    }
    truth = {
        "subhalos": [
            {
                "subhalo_mass_msun": 1.0e11,
                "x_arcsec": 2.0,
                "y_arcsec": -1.0,
                "luminosity_ratio": 1.0,
                "catalog_mag": 17.0,
            }
        ],
        "subhalo_selection": {"candidates": [{"subhalo_mass_msun": 1.0e11, "selected": True}]},
        "caustics_by_source_redshift": {"9.00000000": [contour_payload]},
    }
    images = pd.DataFrame({"x_obs_arcsec": [0.0], "y_obs_arcsec": [1.0]})

    outputs = validation.write_prefit_validation_diagnostics(truth, images, tmp_path)

    assert outputs == {
        "subhalo_shmf_plot": tmp_path / "subhalo_shmf.pdf",
        "prefit_subhalo_spatial_distribution_plot": tmp_path / "prefit_subhalo_spatial_distribution.pdf",
        "prefit_critical_lines_plot": tmp_path / "prefit_critical_lines.pdf",
    }
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_generate_single_bcg_mock_can_request_subhalo_caustic_family(tmp_path: Path) -> None:
    config = SingleBCGMockConfig(
        seed=11,
        n_primary_families=1,
        n_subhalo_families=1,
        n_subhalos=8,
        pos_sigma_arcsec=0.0,
        primary_source_redshifts=(2.0,),
        subhalo_source_redshifts=(2.0,),
    )

    _paths, images, truth = generate_single_bcg_mock(tmp_path, config)

    assert [source["caustic_class"] for source in truth["sources"]] == ["primary", "subhalo"]
    assert images["family_id"].nunique() == 2
    assert (images.groupby("family_id").size() >= config.min_images_per_family).all()


def test_caustic_classifier_marks_largest_closed_curve_primary() -> None:
    small = validation.CausticContour(
        caustic_index=0,
        caustic_class="subhalo",
        beta_x=np.asarray([-0.1, 0.1, 0.1, -0.1, -0.1]),
        beta_y=np.asarray([-0.1, -0.1, 0.1, 0.1, -0.1]),
        critical_x=np.asarray([-1.0, 1.0, 1.0, -1.0, -1.0]),
        critical_y=np.asarray([-1.0, -1.0, 1.0, 1.0, -1.0]),
        caustic_area_arcsec2=0.04,
        critical_area_arcsec2=4.0,
    )
    large = validation.CausticContour(
        caustic_index=1,
        caustic_class="primary",
        beta_x=np.asarray([-0.3, 0.3, 0.3, -0.3, -0.3]),
        beta_y=np.asarray([-0.3, -0.3, 0.3, 0.3, -0.3]),
        critical_x=np.asarray([-3.0, 3.0, 3.0, -3.0, -3.0]),
        critical_y=np.asarray([-3.0, -3.0, 3.0, 3.0, -3.0]),
        caustic_area_arcsec2=0.36,
        critical_area_arcsec2=36.0,
    )

    assert max([small, large], key=lambda item: item.caustic_area_arcsec2).caustic_class == "primary"


def test_sample_point_in_caustic_returns_inside_point() -> None:
    contour = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([0.0, 1.0, 1.0, 0.0, 0.0]),
        beta_y=np.asarray([0.0, 0.0, 1.0, 1.0, 0.0]),
        critical_x=np.asarray([0.0, 1.0, 1.0, 0.0, 0.0]),
        critical_y=np.asarray([0.0, 0.0, 1.0, 1.0, 0.0]),
        caustic_area_arcsec2=1.0,
        critical_area_arcsec2=1.0,
    )

    x, y = validation._sample_point_in_caustic(contour, np.random.default_rng(1))

    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
    assert validation.MplPath(np.column_stack([contour.beta_x, contour.beta_y])).contains_point((x, y))


def test_cluster_solver_cosmology_accepts_lenstool_cosmology_keys() -> None:
    cosmo = _build_cosmology({"cosmology": {"H0": 67.74, "omega": 0.3089, "lambda": 0.6911}})

    assert cosmo["class"] == "FlatLambdaCDM"
    np.testing.assert_allclose(cosmo["H0"], 67.74)
    np.testing.assert_allclose(cosmo["Om0"], 0.3089)


def test_cluster_solver_cosmology_accepts_cosmologie_modern_keys() -> None:
    cosmo = _build_cosmology({"cosmologie": {"H0": 67.74, "omegaM": 0.3089, "omegaX": 0.6911}})

    assert cosmo["class"] == "FlatLambdaCDM"
    np.testing.assert_allclose(cosmo["H0"], 67.74)
    np.testing.assert_allclose(cosmo["Om0"], 0.3089)


def test_cluster_solver_cosmology_defaults_without_block() -> None:
    cosmo = _build_cosmology({})

    assert cosmo["class"] == "FlatLambdaCDM"
    np.testing.assert_allclose(cosmo["H0"], 70.0)
    np.testing.assert_allclose(cosmo["Om0"], 0.3)


def test_flat_wcdm_jax_distances_match_astropy() -> None:
    z_lens = 0.3734
    source_redshifts = np.asarray([1.5, 3.0, 7.0], dtype=float)
    h0 = 70.0
    om0 = 0.31
    w0 = -0.8
    cosmo = FlatwCDM(H0=h0, Om0=om0, w0=w0)

    chi_source = flat_wcdm_comoving_distance_mpc(source_redshifts, h0, om0, w0)
    efficiency = flat_wcdm_lensing_efficiency(z_lens, source_redshifts, h0, om0, w0)
    kpc_per_arcsec = flat_wcdm_kpc_per_arcsec(z_lens, h0, om0, w0)

    np.testing.assert_allclose(
        np.asarray(chi_source),
        cosmo.comoving_distance(source_redshifts).value,
        rtol=1.0e-3,
        atol=1.0e-3,
    )
    np.testing.assert_allclose(
        float(kpc_per_arcsec),
        cosmo.kpc_proper_per_arcmin(z_lens).to("kpc/arcsec").value,
        rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        np.asarray(efficiency),
        (cosmo.angular_diameter_distance_z1z2(z_lens, source_redshifts) / cosmo.angular_diameter_distance(source_redshifts)).value,
        rtol=1.0e-3,
    )
    assert np.all(np.asarray(dpie_sigma0_factor_from_lensing_efficiency(efficiency)) > 0.0)


def test_flat_wcdm_vectorized_lens_geometry_matches_scalar_helpers() -> None:
    z_lens = 0.3734
    source_redshifts = np.asarray([1.5, 3.0, 7.0], dtype=float)
    h0 = 70.0
    om0 = 0.31
    w0 = -0.8

    kpc_per_arcsec, efficiency, sigma0_factors = flat_wcdm_lens_geometry_factors(
        z_lens,
        source_redshifts,
        h0,
        om0,
        w0,
    )

    np.testing.assert_allclose(
        float(kpc_per_arcsec),
        float(flat_wcdm_kpc_per_arcsec(z_lens, h0, om0, w0)),
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(efficiency),
        np.asarray(flat_wcdm_lensing_efficiency(z_lens, source_redshifts, h0, om0, w0)),
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(sigma0_factors),
        np.asarray(dpie_sigma0_factor(z_lens, source_redshifts, h0, om0, w0)),
        rtol=1.0e-10,
    )


def test_dpie_sigma0_from_vel_disp_uses_lenstool_profile81_normalization() -> None:
    z_lens = 0.396
    z_source = 2.0
    h0 = 70.0
    om0 = 0.3
    w0 = -1.0
    vel_disp = 760.0
    ra_arcsec = 5.0
    rs_arcsec = 220.0
    cosmo_config = {"class": "FlatLambdaCDM", "H0": h0, "Om0": om0}

    efficiency = float(flat_wcdm_lensing_efficiency(z_lens, z_source, h0, om0, w0))
    expected_factor = (1.0 / C_LIGHT_KM_S) ** 2 * 3.0 * np.pi * efficiency / ARCSEC_TO_RAD
    expected_sigma0 = vel_disp**2 * expected_factor / ra_arcsec

    assert float(dpie_sigma0_factor(z_lens, z_source, h0, om0, w0)) == pytest.approx(expected_factor)
    assert dpie_sigma0_from_vel_disp(
        vel_disp,
        ra_arcsec,
        rs_arcsec,
        z_lens,
        z_source,
        cosmo_config,
    ) == pytest.approx(expected_sigma0)
    assert dpie_sigma0_from_vel_disp(
        vel_disp,
        ra_arcsec,
        400.0,
        z_lens,
        z_source,
        cosmo_config,
    ) == pytest.approx(expected_sigma0)
    assert math.isnan(dpie_sigma0_from_vel_disp(vel_disp, ra_arcsec, ra_arcsec, z_lens, z_source, cosmo_config))


def test_catalog_shape_to_ellipticity_round_trips_lenstool_axis_ratio() -> None:
    ellipticite, theta = cluster_solver._catalog_shape_to_ellipticity(2.0, 1.0, 12.0)

    assert ellipticite == pytest.approx(0.6)
    assert theta == pytest.approx(12.0)
    q = math.sqrt((1.0 - ellipticite) / (1.0 + ellipticite))
    assert q == pytest.approx(0.5)


def test_cosmology_parameter_specs_use_broad_uniform_priors() -> None:
    specs = cluster_solver._build_cosmology_parameter_specs(0, FlatwCDM(H0=70.0, Om0=0.31, w0=-0.8))

    assert [spec.sample_name for spec in specs] == ["cosmology_Om0", "cosmology_w0"]
    assert [spec.component_family for spec in specs] == ["cosmology", "cosmology"]
    assert specs[0].prior_kind == "uniform"
    assert specs[0].lower == pytest.approx(0.05)
    assert specs[0].upper == pytest.approx(0.6)
    assert specs[0].physical_mean == pytest.approx(0.31)
    assert specs[1].lower == pytest.approx(-2.0)
    assert specs[1].upper == pytest.approx(-0.3)
    assert specs[1].physical_mean == pytest.approx(-0.8)


def test_image_plane_scatter_default_log_uniform_prior_uses_default_floor() -> None:
    spec = cluster_solver._build_image_scatter_parameter_spec(
        start_index=0,
        upper_arcsec=2.0,
    )

    assert spec.sample_name == "image_sigma_int"
    assert spec.prior_kind == "uniform"
    assert spec.lower == pytest.approx(np.log(cluster_solver.DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC))
    assert spec.upper == pytest.approx(np.log(2.0))
    assert spec.transform_kind == "log_positive"
    assert spec.physical_lower == pytest.approx(cluster_solver.DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)
    assert spec.physical_upper == pytest.approx(2.0)


def test_image_plane_scatter_log_uniform_prior_uses_floor_as_lower_support() -> None:
    spec = cluster_solver._build_image_scatter_parameter_spec(
        start_index=0,
        upper_arcsec=0.5,
        floor_arcsec=0.05,
    )

    assert spec.prior_kind == "uniform"
    assert spec.lower == pytest.approx(np.log(0.05))
    assert spec.upper == pytest.approx(np.log(0.5))
    assert spec.physical_lower == pytest.approx(0.05)
    assert spec.physical_upper == pytest.approx(0.5)


def test_image_plane_scatter_lognormal_prior_uses_truncated_normal_latent_log_sigma() -> None:
    spec = cluster_solver._build_image_scatter_parameter_spec(
        start_index=0,
        upper_arcsec=0.5,
        floor_arcsec=0.05,
        prior=cluster_solver.IMAGE_PLANE_SCATTER_PRIOR_LOGNORMAL,
        prior_median_arcsec=0.25,
        prior_log_sigma=0.4,
    )

    physical = cluster_solver._convert_theta_to_physical(np.asarray([np.log(0.25)]), [spec])
    log_prob = cluster_solver._prior_log_prob([spec], jnp.asarray([np.log(0.25)], dtype=jnp.float64))

    assert spec.prior_kind == "truncated_normal"
    assert spec.lower == pytest.approx(np.log(0.05))
    assert spec.upper == pytest.approx(np.log(0.5))
    assert spec.mean == pytest.approx(np.log(0.25))
    assert spec.std == pytest.approx(0.4)
    assert spec.transform_kind == "log_positive"
    assert spec.physical_lower == pytest.approx(0.05)
    assert spec.physical_upper == pytest.approx(0.5)
    assert spec.physical_mean == pytest.approx(0.25)
    assert physical.tolist() == pytest.approx([0.25])
    assert np.isfinite(float(log_prob))


@pytest.mark.parametrize(
    ("floor_arcsec", "upper_arcsec", "match"),
    [
        (0.0, 2.0, "floor"),
        (float("nan"), 2.0, "floor"),
        (0.1, 0.1, "upper"),
        (0.1, float("nan"), "upper"),
    ],
)
def test_image_plane_scatter_parameter_spec_rejects_invalid_direct_bounds(
    floor_arcsec: float,
    upper_arcsec: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        cluster_solver._build_image_scatter_parameter_spec(
            start_index=0,
            upper_arcsec=upper_arcsec,
            floor_arcsec=floor_arcsec,
        )


def test_family_data_search_center_uses_observed_bounding_box_midpoint() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.1,
        image_labels=["1.a", "1.b", "1.c", "1.d", "1.e"],
        x_obs=np.asarray([7.9, -19.9, -22.2, 9.5, 0.6], dtype=float),
        y_obs=np.asarray([-26.4, 13.1, -0.3, 12.6, 0.9], dtype=float),
        reliability=np.ones(5, dtype=float),
    )

    assert family.x_center == pytest.approx(0.5 * (np.min(family.x_obs) + np.max(family.x_obs)))
    assert family.y_center == pytest.approx(0.5 * (np.min(family.y_obs) + np.max(family.y_obs)))
    assert family.x_center != pytest.approx(float(np.mean(family.x_obs)))
    assert family.y_center != pytest.approx(float(np.mean(family.y_obs)))
    half_window = 0.5 * family.search_window
    assert np.all(np.abs(family.x_obs - family.x_center) <= half_window)
    assert np.all(np.abs(family.y_obs - family.y_center) <= half_window)
    assert np.any(np.abs(family.y_obs - float(np.mean(family.y_obs))) > half_window)


def test_prior_whitened_source_position_specs_map_unit_offsets_to_physical_beta() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.15,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 1.0]),
        reliability=np.ones(2),
    )

    specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        {"1": (0.25, -0.5)},
        start_index=0,
        beta_prior_sigma_arcsec=0.3,
        parameterization="prior-whitened",
    )

    assert [spec.sample_name for spec in specs] == ["source_1_beta_x", "source_1_beta_y"]
    assert [spec.mean for spec in specs] == [0.0, 0.0]
    assert [spec.std for spec in specs] == [1.0, 1.0]
    assert [spec.transform_kind for spec in specs] == ["affine", "affine"]
    assert specs[0].physical_mean == pytest.approx(0.25)
    assert specs[1].physical_mean == pytest.approx(-0.5)
    assert specs[0].transform_offset == pytest.approx(0.25)
    assert specs[1].transform_offset == pytest.approx(-0.5)
    assert specs[0].transform_scale == pytest.approx(0.3)
    assert specs[1].transform_scale == pytest.approx(0.3)
    assert cluster_solver._convert_theta_to_physical(np.asarray([0.0, 0.0]), specs).tolist() == pytest.approx([0.25, -0.5])
    assert cluster_solver._convert_theta_to_physical(np.asarray([1.0, 1.0]), specs).tolist() == pytest.approx([0.55, -0.2])


def test_direct_source_position_specs_keep_physical_beta_sampling() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.15,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 1.0]),
        reliability=np.ones(2),
    )

    specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        {"1": (0.25, -0.5)},
        start_index=0,
        beta_prior_sigma_arcsec=0.3,
        parameterization="direct",
    )

    assert [spec.mean for spec in specs] == pytest.approx([0.25, -0.5])
    assert [spec.std for spec in specs] == pytest.approx([0.3, 0.3])
    assert [spec.transform_kind for spec in specs] == ["identity", "identity"]


def test_explicit_source_position_parameterization_uses_state_metadata_not_spec_shape() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.15,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 1.0]),
        reliability=np.ones(2),
    )
    specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        {"1": (0.0, 0.0)},
        start_index=0,
        beta_prior_sigma_arcsec=1.0,
        parameterization="direct",
    )

    state = SimpleNamespace(parameter_specs=specs, source_position_parameterization="direct")

    assert cluster_solver._explicit_source_position_parameterization_for_state(state) == "direct"


def test_explicit_source_position_parameterization_requires_metadata() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.15,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 1.0]),
        reliability=np.ones(2),
    )
    specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        {"1": (0.0, 0.0)},
        start_index=0,
        beta_prior_sigma_arcsec=1.0,
        parameterization="conditional-whitened",
    )

    with pytest.raises(ValueError, match="missing explicit source_position_parameterization"):
        cluster_solver._explicit_source_position_parameterization_for_state(SimpleNamespace(parameter_specs=specs))


def test_prior_whitened_source_position_prior_matches_direct_prior_up_to_jacobian() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.15,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 1.0]),
        reliability=np.ones(2),
    )
    centers = {"1": (0.25, -0.5)}
    sigma = 0.3
    direct_specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        centers,
        start_index=0,
        beta_prior_sigma_arcsec=sigma,
        parameterization="direct",
    )
    whitened_specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        centers,
        start_index=0,
        beta_prior_sigma_arcsec=sigma,
        parameterization="prior-whitened",
    )
    eta = np.asarray([0.7, -1.2])
    beta = cluster_solver._convert_theta_to_physical(eta, whitened_specs)

    direct_logp = float(cluster_solver._prior_log_prob(direct_specs, jnp.asarray(beta)))
    whitened_logp = float(cluster_solver._prior_log_prob(whitened_specs, jnp.asarray(eta)))

    assert whitened_logp - direct_logp == pytest.approx(2.0 * np.log(sigma))


def test_conditional_whitened_source_transport_correction_matches_change_of_variables() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.15,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 1.0]),
        reliability=np.ones(2),
    )
    specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        {"1": (0.25, -0.5)},
        start_index=0,
        beta_prior_sigma_arcsec=0.3,
        parameterization="conditional-whitened",
    )
    traced_bin = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0, 0], dtype=jnp.int32),
        x_obs=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        y_obs=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.2, 0.4], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0, 0.8], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True], dtype=bool),
    )
    fake_evaluator = SimpleNamespace(
        state=SimpleNamespace(parameter_specs=specs),
        source_position_param_indices_by_family={"1": (0, 1)},
        source_plane_covariance_floor=0.01,
        image_plane_scatter_floor_arcsec=0.3,
        likelihood_stabilizer_max_gain=0.0,
        likelihood_stabilizer_max_residual_arcsec=0.0,
    )
    eta = jnp.asarray([0.7, -1.2], dtype=jnp.float64)
    beta_x = jnp.asarray([0.1, 0.3], dtype=jnp.float64)
    beta_y = jnp.asarray([-0.2, -0.1], dtype=jnp.float64)
    image_sigma_int = jnp.asarray(0.05, dtype=jnp.float64)
    jacobian_entries = (
        jnp.asarray([1.2, 0.8], dtype=jnp.float64),
        jnp.asarray([0.15, -0.05], dtype=jnp.float64),
        jnp.asarray([0.2, 0.1], dtype=jnp.float64),
        jnp.asarray([0.9, 1.1], dtype=jnp.float64),
    )

    source_x, source_y, finite, correction = (
        cluster_solver.ClusterJAXEvaluator._conditional_source_position_transport_for_bin(
            fake_evaluator,
            eta,
            traced_bin,
            beta_x,
            beta_y,
            image_sigma_int,
            jacobian_entries,
        )
    )

    sigma2 = np.asarray(traced_bin.sigma_per_image) ** 2 + float(image_sigma_int) ** 2 + 0.01
    weights = np.asarray(traced_bin.reliability_per_image) / sigma2
    prior_precision = 1.0 / 0.3**2
    precision_matrix = prior_precision * np.eye(2)
    rhs = prior_precision * np.asarray([0.25, -0.5])
    jac_arrays = [np.asarray(item, dtype=float) for item in jacobian_entries]
    for image_index, weight in enumerate(weights):
        a_matrix = np.asarray(
            [
                [jac_arrays[0][image_index], jac_arrays[1][image_index]],
                [jac_arrays[2][image_index], jac_arrays[3][image_index]],
            ],
            dtype=float,
        )
        inv_a = np.linalg.inv(a_matrix)
        lambda_i = float(weight) * inv_a.T @ inv_a
        beta_i = np.asarray([float(beta_x[image_index]), float(beta_y[image_index])])
        precision_matrix += lambda_i
        rhs += lambda_i @ beta_i
    covariance = np.linalg.inv(precision_matrix)
    mean = covariance @ rhs
    chol = np.linalg.cholesky(covariance)
    expected_source = mean + chol @ np.asarray(eta, dtype=float)

    assert bool(finite)
    assert np.asarray(source_x).tolist() == pytest.approx([expected_source[0], expected_source[0]])
    assert np.asarray(source_y).tolist() == pytest.approx([expected_source[1], expected_source[1]])

    fake_evaluator.source_metric_cache_by_z = {2.0: {"jac_a00": np.full(2, 99.0)}}
    source_x_cached, source_y_cached, finite_cached, correction_cached = (
        cluster_solver.ClusterJAXEvaluator._conditional_source_position_transport_for_bin(
            fake_evaluator,
            eta,
            traced_bin,
            beta_x,
            beta_y,
            image_sigma_int,
            jacobian_entries,
        )
    )
    assert bool(finite_cached)
    np.testing.assert_allclose(np.asarray(source_x_cached), np.asarray(source_x))
    np.testing.assert_allclose(np.asarray(source_y_cached), np.asarray(source_y))
    assert float(correction_cached) == pytest.approx(float(correction))

    beta_prior_logp = -0.5 * (
        np.sum(((expected_source - np.asarray([0.25, -0.5])) / 0.3) ** 2)
        + 2.0 * np.log(2.0 * np.pi * 0.3**2)
    )
    eta_prior_logp = -0.5 * (float(np.sum(np.asarray(eta) ** 2)) + 2.0 * np.log(2.0 * np.pi))
    log_det = float(np.log(np.linalg.det(chol)))

    assert eta_prior_logp + float(correction) == pytest.approx(beta_prior_logp + log_det)


def _conditional_inverse_cache_fake_evaluator() -> Any:
    specs = [
        ParameterSpec(
            name="scale",
            sample_name="scale",
            potential_id="scale",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="sigma_ref",
            prior_kind="uniform",
            lower=-100.0,
            upper=100.0,
            step=0.1,
            component_family="scaling",
        ),
        ParameterSpec(
            name="1.beta_x",
            sample_name="source_1_beta_x",
            potential_id="1",
            profile_type=0,
            field="beta_x",
            prior_kind="normal",
            lower=-100.0,
            upper=100.0,
            step=0.1,
            component_family="source_position",
        ),
        ParameterSpec(
            name="1.beta_y",
            sample_name="source_1_beta_y",
            potential_id="1",
            profile_type=0,
            field="beta_y",
            prior_kind="normal",
            lower=-100.0,
            upper=100.0,
            step=0.1,
            component_family="source_position",
        ),
    ]
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.state = SimpleNamespace(parameter_specs=specs)
    evaluator.source_position_conditional = True
    evaluator.source_position_param_indices_by_family = {"1": (1, 2)}
    evaluator._conditional_source_inverse_basis_cache = {}
    evaluator.reported_calls = 0
    transform = np.asarray([[2.0, 0.5], [-0.25, 1.5]], dtype=float)

    def reported_physical_parameter_vector(params: np.ndarray) -> np.ndarray:
        evaluator.reported_calls += 1
        params = np.asarray(params, dtype=float)
        physical = params.copy()
        mean = np.asarray([10.0 + 2.0 * params[0], -5.0 + 3.0 * params[0]], dtype=float)
        physical[1:3] = mean + transform @ params[1:3]
        return physical

    evaluator.reported_physical_parameter_vector = reported_physical_parameter_vector
    return evaluator


def test_conditional_reported_physical_to_latent_reuses_cached_source_basis() -> None:
    evaluator = _conditional_inverse_cache_fake_evaluator()
    target_latent = np.asarray([4.0, 0.6, -0.2], dtype=float)
    reported_physical = evaluator.reported_physical_parameter_vector(target_latent)
    evaluator.reported_calls = 0

    converted_once = evaluator.reported_physical_to_latent_parameter_vector(reported_physical)
    converted_twice = evaluator.reported_physical_to_latent_parameter_vector(reported_physical)

    np.testing.assert_allclose(converted_once, target_latent)
    np.testing.assert_allclose(converted_twice, target_latent)
    assert len(evaluator._conditional_source_inverse_basis_cache) == 1
    assert evaluator.reported_calls == 3


def test_conditional_reported_physical_to_latent_cache_keys_non_source_state() -> None:
    evaluator = _conditional_inverse_cache_fake_evaluator()
    first_latent = np.asarray([4.0, 0.6, -0.2], dtype=float)
    second_latent = np.asarray([5.0, 0.6, -0.2], dtype=float)
    first_reported = evaluator.reported_physical_parameter_vector(first_latent)
    second_reported = evaluator.reported_physical_parameter_vector(second_latent)
    evaluator.reported_calls = 0

    first_converted = evaluator.reported_physical_to_latent_parameter_vector(first_reported)
    second_converted = evaluator.reported_physical_to_latent_parameter_vector(second_reported)

    np.testing.assert_allclose(first_converted, first_latent)
    np.testing.assert_allclose(second_converted, second_latent)
    assert len(evaluator._conditional_source_inverse_basis_cache) == 2
    assert evaluator.reported_calls == 6


def _minimal_stage4_surrogate_state(*, fit_cosmology_flat_wcdm: bool = False) -> BuildState:
    n_components = 2
    int_minus_one = np.full(n_components, -1, dtype=np.int32)
    float_zero = np.zeros(n_components, dtype=float)
    packed = PackedLensSpec(
        profile_type=np.asarray([cluster_solver.DP_IE_PROFILE, cluster_solver.DP_IE_PROFILE], dtype=np.int32),
        component_family=np.asarray([0, 1], dtype=np.int32),
        x_center_base=np.asarray([0.0, 1.0], dtype=float),
        y_center_base=np.asarray([0.0, 1.0], dtype=float),
        ellipticite_base=float_zero.copy(),
        angle_pos_base=float_zero.copy(),
        core_radius_kpc_base=np.asarray([5.0, 1.0], dtype=float),
        cut_radius_kpc_base=np.asarray([100.0, 20.0], dtype=float),
        v_disp_base=np.asarray([900.0, 120.0], dtype=float),
        gamma_base=float_zero.copy(),
        x_center_param_index=int_minus_one.copy(),
        y_center_param_index=int_minus_one.copy(),
        ellipticite_param_index=int_minus_one.copy(),
        angle_pos_param_index=int_minus_one.copy(),
        core_radius_param_index=int_minus_one.copy(),
        cut_radius_param_index=int_minus_one.copy(),
        v_disp_param_index=int_minus_one.copy(),
        gamma_param_index=int_minus_one.copy(),
        luminosity_ratio=np.ones(n_components, dtype=float),
        sigma_ref_base=np.asarray([0.0, 120.0], dtype=float),
        cut_ref_base=np.asarray([0.0, 20.0], dtype=float),
        core_ref_base=np.asarray([0.0, 1.0], dtype=float),
        vdslope_base=np.ones(n_components, dtype=float),
        slope_base=np.ones(n_components, dtype=float),
        sigma_ref_param_index=int_minus_one.copy(),
        cut_ref_param_index=int_minus_one.copy(),
        core_ref_param_index=int_minus_one.copy(),
        vdslope_param_index=int_minus_one.copy(),
        slope_param_index=int_minus_one.copy(),
        sigma_log_scatter_param_index=int_minus_one.copy(),
        core_log_scatter_param_index=int_minus_one.copy(),
        cut_log_scatter_param_index=int_minus_one.copy(),
    )
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.1,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0], dtype=float),
        reliability=np.ones(2, dtype=float),
    )
    bin_data = cluster_solver.BinData(
        effective_z_source=2.0,
        family_ids=["1"],
        family_index_per_image=np.asarray([0, 0], dtype=np.int32),
        x_obs=family.x_obs,
        y_obs=family.y_obs,
        sigma_per_image=np.full(2, 0.1, dtype=float),
        reliability_per_image=np.ones(2, dtype=float),
    )
    geometry_cache = GeometryCache(
        effective_z_source_values=[2.0],
        exact_z_source_values=[2.0],
        family_z_source_map={"1": 2.0},
        family_effective_z_source_map={"1": 2.0},
        dpie_sigma0_factor_by_effective_z={2.0: 1.0},
        dpie_sigma0_factor_by_exact_z={2.0: 1.0},
        lens_quadrature_z=[0.4],
        lens_quadrature_weights=[1.0],
        effective_z_quadrature_z=[[2.0]],
        effective_z_quadrature_weights=[[1.0]],
        exact_z_quadrature_z=[[2.0]],
        exact_z_quadrature_weights=[[1.0]],
    )
    parameter_specs = [
        ParameterSpec(
            name="sub_scale",
            sample_name="sub_scale",
            potential_id="sub",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="sigma_ref",
            prior_kind="uniform",
            lower=0.0,
            upper=2.0,
            step=0.1,
            component_family="scaling",
        )
    ]
    if fit_cosmology_flat_wcdm:
        parameter_specs.extend(
            [
                ParameterSpec(
                    name="cosmology.Om0",
                    sample_name=cluster_solver.COSMOLOGY_OM0_SAMPLE_NAME,
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
                    sample_name=cluster_solver.COSMOLOGY_W0_SAMPLE_NAME,
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
        )

    return BuildState(
        run_name="surrogate",
        par_path="mock.par",
        cosmo_config={"class": "FlatLambdaCDM", "H0": 70.0, "Om0": 0.3},
        z_lens=0.4,
        sigma_arcsec=0.1,
        parsed={},
        parameter_specs=parameter_specs,
        base_components=[],
        packed_lens_spec=packed,
        family_data=[family],
        bin_data=[bin_data],
        lens_model_list=["DPIE_NIE", "DPIE_NIE"],
        reference=(0, 0.0, 0.0),
        fit_mode="joint",
        potfiles=[],
        scaling_component_records=[],
        geometry_cache=geometry_cache,
        fit_cosmology_flat_wcdm=fit_cosmology_flat_wcdm,
        source_position_parameterization="prior-whitened",
    )


def test_stage4_refreshing_surrogate_enables_with_zero_newton_steps() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=_minimal_stage4_surrogate_state(),
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        image_plane_newton_steps=0,
    )

    assert evaluator.surrogate_enabled is True


def test_forward_metric_stage4_refreshing_surrogate_enables_with_zero_newton_steps() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=_minimal_stage4_surrogate_state(),
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
        image_plane_newton_steps=0,
        image_presence_penalty_weight=2.0,
    )

    assert evaluator.surrogate_enabled is True
    assert evaluator.image_presence_penalty_weight == pytest.approx(2.0)


def test_fold_regularized_stage4_refreshing_surrogate_enables_with_zero_newton_steps() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=_minimal_stage4_surrogate_state(),
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
        image_plane_newton_steps=0,
        fold_curvature_arcsec_inv=1.25,
        image_presence_penalty_weight=2.0,
    )

    assert evaluator.surrogate_enabled is True
    assert evaluator.fold_curvature_arcsec_inv == pytest.approx(1.25)
    assert evaluator.image_presence_penalty_weight == pytest.approx(2.0)


def test_zero_step_anchored_stage4_refreshing_surrogate_enables() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=_minimal_stage4_surrogate_state(),
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
        image_plane_newton_steps=0,
        anchored_image_plane_solve_steps=0,
    )

    assert evaluator.surrogate_enabled is True


def test_critical_arc_stage4_refreshing_surrogate_enables() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=_minimal_stage4_surrogate_state(),
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        image_plane_newton_steps=0,
    )

    assert evaluator.surrogate_enabled is True
    assert evaluator.image_presence_penalty_weight == pytest.approx(0.0)


def test_critical_arc_stage4_explicit_image_presence_penalty_is_honored() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=_minimal_stage4_surrogate_state(),
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        image_plane_newton_steps=0,
        image_presence_penalty_weight=2.0,
    )

    assert evaluator.image_presence_penalty_weight == pytest.approx(2.0)


def test_iterative_anchored_stage4_refreshing_surrogate_rejects() -> None:
    with pytest.raises(ValueError, match="anchored_image_plane_solve_steps=0"):
        cluster_solver.ClusterJAXEvaluator(
            state=_minimal_stage4_surrogate_state(),
            match_tolerance_arcsec=0.1,
            sampling_engine="refreshing_surrogate",
            sample_likelihood_mode=SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
            image_plane_newton_steps=0,
            anchored_image_plane_solve_steps=1,
        )


def test_forward_metric_refresh_surrogate_populates_inactive_jacobian_cache() -> None:
    state = _minimal_stage4_surrogate_state()
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
        image_plane_newton_steps=0,
    )

    evaluator.refresh_surrogate(cluster_solver._default_theta(state.parameter_specs), reason="test")

    assert evaluator.surrogate_cache_by_z
    cache = next(iter(evaluator.surrogate_cache_by_z.values()))
    n_obs = len(state.bin_data[0].x_obs)
    n_surrogate = len(evaluator.surrogate_param_indices)
    for field_name in (
        "inactive_jacobian_delta_a00",
        "inactive_jacobian_delta_a01",
        "inactive_jacobian_delta_a10",
        "inactive_jacobian_delta_a11",
    ):
        value = getattr(cache, field_name)
        assert value is not None
        np.testing.assert_equal(np.asarray(value).shape, (n_obs,))
        assert np.isfinite(value).all()
    for field_name in (
        "inactive_jacobian_delta_da00_dparams",
        "inactive_jacobian_delta_da01_dparams",
        "inactive_jacobian_delta_da10_dparams",
        "inactive_jacobian_delta_da11_dparams",
    ):
        value = getattr(cache, field_name)
        assert value is not None
        np.testing.assert_equal(np.asarray(value).shape, (n_surrogate, n_obs))
        assert np.isfinite(value).all()


def test_fold_regularized_refresh_surrogate_populates_cached_curvature_and_masks() -> None:
    state = _minimal_stage4_surrogate_state()
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
        image_plane_newton_steps=0,
    )

    evaluator.refresh_surrogate(cluster_solver._default_theta(state.parameter_specs), reason="test")

    assert evaluator.surrogate_cache_by_z
    cache = next(iter(evaluator.surrogate_cache_by_z.values()))
    n_obs = len(state.bin_data[0].x_obs)
    assert cache.fold_regularized_kappa_eff is not None
    assert cache.fold_regularized_near_indices is not None
    assert cache.fold_regularized_far_indices is not None
    np.testing.assert_equal(np.asarray(cache.fold_regularized_kappa_eff).shape, (n_obs,))
    assert np.isfinite(cache.fold_regularized_kappa_eff).all()
    near_indices = np.asarray(cache.fold_regularized_near_indices, dtype=np.int32)
    far_indices = np.asarray(cache.fold_regularized_far_indices, dtype=np.int32)
    np.testing.assert_array_equal(np.sort(np.concatenate([near_indices, far_indices])), np.arange(n_obs, dtype=np.int32))
    assert np.intersect1d(near_indices, far_indices).size == 0


def _fold_regularized_source_loglike_fake(
    surrogate_cache_by_z: dict[float, object],
    *,
    image_presence_penalty_weight: float,
    signed_curvature_calls: list[int],
) -> object:
    traced_bin = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        x_obs=jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float64),
        y_obs=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.2, 0.2, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999, 0.999, 0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True, True], dtype=bool),
        constrained_image_indices=jnp.asarray([0, 1, 2], dtype=jnp.int32),
    )

    class FakeEvaluator:
        source_position_conditional = False
        source_position_param_indices_by_family = {"1": (0, 1)}
        surrogate_enabled = True
        sample_likelihood_mode = SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE
        fold_curvature_arcsec_inv = 1.0
        image_plane_scatter_floor_arcsec = 1.0e-3
        source_plane_covariance_floor = 1.0e-6
        source_plane_outlier_sigma_arcsec = 10.0
        image_presence_match_radius_arcsec = 0.30
        image_presence_temperature_arcsec = 0.10
        image_presence_count_softness = 0.05
        image_presence_count_margin = 0.05
        likelihood_stabilizer_max_gain = 0.0
        likelihood_stabilizer_max_residual_arcsec = 0.0
        likelihood_stabilizer_residual_loss = "gaussian"
        likelihood_stabilizer_student_t_nu = 4.0
        fit_cosmology_flat_wcdm = False

        def __init__(self) -> None:
            self.surrogate_cache_by_z = surrogate_cache_by_z
            self.image_presence_penalty_weight = image_presence_penalty_weight
            self.traced_bin_data = [traced_bin]

        def _physical_parameter_vector(self, params):
            return params

        def _source_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.0, dtype=jnp.float64)

        def _image_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.05, dtype=jnp.float64)

        def _surrogate_beta(self, _params, _physical_params, bin_data):
            return bin_data.x_obs + 0.4, bin_data.y_obs - 0.1, jnp.asarray(False), {}

        def _surrogate_jacobian_entries(self, _params, bin_data, _packed_state, _invalid):
            return (
                jnp.asarray([0.05, 1.0, 0.05], dtype=jnp.float64),
                jnp.zeros_like(bin_data.x_obs),
                jnp.zeros_like(bin_data.y_obs),
                jnp.ones_like(bin_data.y_obs),
            )

        def _scaling_scatter_extra_variance_from_physical(self, _physical_params, bin_data, _beta_x, _beta_y):
            return jnp.zeros_like(bin_data.x_obs), jnp.zeros_like(bin_data.y_obs)

        def _fold_signed_curvature_from_observed_jacobian(
            self,
            _z_source,
            x_obs,
            _y_obs,
            _packed_state,
            _observed_jacobian_entries,
            _component_indices=None,
            *,
            fold_frame=None,
            image_indices=None,
            fill_value=1.0,
        ):
            del fold_frame, fill_value
            signed_curvature_calls.append(
                int(image_indices.shape[0]) if image_indices is not None else int(x_obs.shape[0])
            )
            return jnp.full_like(x_obs, 2.0)

        def _cab_morphology_loglike_for_bin(self, *_args, **_kwargs):
            return jnp.asarray(0.0, dtype=jnp.float64)

    fake = FakeEvaluator()
    fake._source_position_vectors_for_bin = cluster_solver.ClusterJAXEvaluator._source_position_vectors_for_bin.__get__(
        fake,
        FakeEvaluator,
    )
    fake._explicit_source_position_vectors_for_bin = (
        cluster_solver.ClusterJAXEvaluator._explicit_source_position_vectors_for_bin.__get__(fake, FakeEvaluator)
    )
    return fake


def test_fold_regularized_cached_fast_path_skips_per_loglike_curvature_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_fold_loglike(**kwargs):
        captured["near_n"] = int(np.asarray(kwargs["residual_beta_x"]).size)
        captured["near_kappa_eff"] = np.asarray(kwargs["fold_kappa_eff"], dtype=float)
        captured["near_presence_weight"] = float(kwargs["image_presence_penalty_weight"])
        return jnp.asarray(100.0, dtype=jnp.float64)

    def fake_forward_loglike(**kwargs):
        captured["far_n"] = int(np.asarray(kwargs["residual_beta_x"]).size)
        captured["far_presence_weight"] = float(kwargs["image_presence_penalty_weight"])
        return jnp.asarray(10.0, dtype=jnp.float64)

    monkeypatch.setattr(cluster_solver, "_fold_regularized_image_plane_bin_loglike", fake_fold_loglike)
    monkeypatch.setattr(cluster_solver, "_forward_metric_image_plane_bin_loglike", fake_forward_loglike)
    signed_curvature_calls: list[int] = []
    fake = _fold_regularized_source_loglike_fake(
        {
            2.0: SimpleNamespace(
                fold_regularized_kappa_eff=np.asarray([1.5, 9.0, 2.5], dtype=float),
                fold_regularized_near_indices=np.asarray([0, 2], dtype=np.int32),
                fold_regularized_far_indices=np.asarray([1], dtype=np.int32),
            )
        },
        image_presence_penalty_weight=0.0,
        signed_curvature_calls=signed_curvature_calls,
    )

    loglike = cluster_solver.ClusterJAXEvaluator._source_loglike_impl(fake, jnp.asarray([0.1, -0.2], dtype=jnp.float64))

    assert float(loglike) == pytest.approx(110.0)
    assert signed_curvature_calls == []
    assert captured["near_n"] == 2
    assert captured["far_n"] == 1
    np.testing.assert_allclose(captured["near_kappa_eff"], np.asarray([1.5, 2.5], dtype=float))
    assert captured["near_presence_weight"] == pytest.approx(0.0)
    assert captured["far_presence_weight"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("surrogate_cache_by_z", "image_presence_penalty_weight"),
    [
        (
            {
                2.0: SimpleNamespace(
                    fold_regularized_kappa_eff=None,
                    fold_regularized_near_indices=None,
                    fold_regularized_far_indices=None,
                )
            },
            0.0,
        ),
        (
            {
                2.0: SimpleNamespace(
                    fold_regularized_kappa_eff=np.asarray([1.5, 9.0, 2.5], dtype=float),
                    fold_regularized_near_indices=np.asarray([0, 2], dtype=np.int32),
                    fold_regularized_far_indices=np.asarray([1], dtype=np.int32),
                )
            },
            2.0,
        ),
    ],
)
def test_fold_regularized_exact_fallback_when_cache_unavailable_or_presence_active(
    monkeypatch: pytest.MonkeyPatch,
    surrogate_cache_by_z: dict[float, object],
    image_presence_penalty_weight: float,
) -> None:
    captured: dict[str, Any] = {}

    def fake_fold_loglike(**kwargs):
        captured["n"] = int(np.asarray(kwargs["residual_beta_x"]).size)
        captured["presence_weight"] = float(kwargs["image_presence_penalty_weight"])
        captured["kappa_eff"] = np.asarray(kwargs["fold_kappa_eff"], dtype=float)
        return jnp.asarray(123.0, dtype=jnp.float64)

    def fail_forward_loglike(**_kwargs):
        raise AssertionError("forward-metric split should not be used on the exact fold fallback path")

    monkeypatch.setattr(cluster_solver, "_fold_regularized_image_plane_bin_loglike", fake_fold_loglike)
    monkeypatch.setattr(cluster_solver, "_forward_metric_image_plane_bin_loglike", fail_forward_loglike)
    signed_curvature_calls: list[int] = []
    fake = _fold_regularized_source_loglike_fake(
        surrogate_cache_by_z,
        image_presence_penalty_weight=image_presence_penalty_weight,
        signed_curvature_calls=signed_curvature_calls,
    )

    loglike = cluster_solver.ClusterJAXEvaluator._source_loglike_impl(fake, jnp.asarray([0.1, -0.2], dtype=jnp.float64))

    assert float(loglike) == pytest.approx(123.0)
    assert signed_curvature_calls == [3]
    assert captured["n"] == 3
    assert captured["presence_weight"] == pytest.approx(image_presence_penalty_weight)
    np.testing.assert_allclose(captured["kappa_eff"], np.full(3, 2.0, dtype=float))


def test_refreshing_surrogate_enables_with_sampled_flat_wcdm_cosmology() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=_minimal_stage4_surrogate_state(fit_cosmology_flat_wcdm=True),
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        image_plane_newton_steps=0,
    )

    assert evaluator.fit_cosmology_flat_wcdm is True
    assert evaluator.surrogate_enabled is True
    assert evaluator.surrogate_param_indices.tolist() == [0, 1, 2]


def test_packed_lens_state_uses_lenstool_sigma0_and_ellipticity() -> None:
    state = _minimal_stage4_surrogate_state()
    state.packed_lens_spec.ellipticite_base[:] = np.asarray([0.28, 0.12], dtype=float)
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        image_plane_newton_steps=0,
    )

    packed_state, details = evaluator._build_packed_lens_state_details_from_physical(
        jnp.asarray([1.0], dtype=jnp.float64),
        2.0,
        kpc_per_arcsec=jnp.asarray(10.0, dtype=jnp.float64),
        dpie_sigma0_factor=jnp.asarray(1.0, dtype=jnp.float64),
    )

    np.testing.assert_allclose(np.asarray(details["sigma0"]), [900.0**2 / 0.5, 120.0**2 / 0.1])
    np.testing.assert_allclose(np.asarray(packed_state["sigma0"]), [900.0**2 / 0.5, 120.0**2 / 0.1])
    expected_q = np.sqrt((1.0 - np.asarray([0.28, 0.12])) / (1.0 + np.asarray([0.28, 0.12])))
    expected_e1 = (1.0 - expected_q) / (1.0 + expected_q)
    np.testing.assert_allclose(np.asarray(details["e1"]), expected_e1)
    np.testing.assert_allclose(np.asarray(details["e2"]), [0.0, 0.0], atol=1.0e-15)


def test_sampled_cosmology_geometry_vectorizes_effective_redshift_factors() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=_minimal_stage4_surrogate_state(fit_cosmology_flat_wcdm=True),
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        image_plane_newton_steps=0,
    )
    physical_params = jnp.asarray([1.1, 0.31, -0.95], dtype=jnp.float64)

    kpc_per_arcsec, sigma0_factors = evaluator._sampled_cosmology_geometry_for_physical(physical_params)

    assert evaluator.traced_bin_data[0].effective_z_index == 0
    np.testing.assert_allclose(
        float(kpc_per_arcsec),
        float(flat_wcdm_kpc_per_arcsec(evaluator.state.z_lens, evaluator.cosmology_h0, 0.31, -0.95)),
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(sigma0_factors),
        np.asarray(dpie_sigma0_factor(evaluator.state.z_lens, [2.0], evaluator.cosmology_h0, 0.31, -0.95)),
        rtol=1.0e-10,
    )


def test_stage4_refreshing_surrogate_rejects_positive_newton_steps() -> None:
    with pytest.raises(ValueError, match="image_plane_newton_steps=0"):
        cluster_solver.ClusterJAXEvaluator(
            state=_minimal_stage4_surrogate_state(),
            match_tolerance_arcsec=0.1,
            sampling_engine="refreshing_surrogate",
            sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
            image_plane_newton_steps=1,
        )


def test_scaling_scatter_specs_use_plain_lognormal_prior() -> None:
    specs, indices = cluster_solver._build_scaling_scatter_parameter_specs(
        [{"id": "potfile", "type": 81}],
        {"sigma", "cut"},
        start_index=7,
        scatter_max=0.123,
    )

    assert indices == [{"sigma": 7, "cut": 8}]
    assert [spec.field for spec in specs] == ["sigma_log_scatter", "cut_log_scatter"]
    for spec in specs:
        assert spec.prior_kind == "normal"
        assert spec.lower == float("-inf")
        assert spec.upper == float("inf")
        assert spec.mean == pytest.approx(np.log(0.02))
        assert spec.std == pytest.approx(0.5)
        assert spec.transform_kind == "log_positive"
        assert spec.physical_lower == 0.0
        assert spec.physical_upper is None
        assert spec.physical_mean == pytest.approx(0.02)


def test_solver_active_approximation_warning_reports_active_features(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_logs: list[str] = []
    state = SimpleNamespace(
        family_data=[object(), object()],
        bin_data=[object()],
        parameter_specs=[SimpleNamespace(component_family="source_position")],
        potfiles=[{"id": "members"}],
    )
    evaluator = SimpleNamespace(
        state=state,
        surrogate_enabled=True,
        inactive_scaling_component_indices=np.asarray([2, 3, 4], dtype=int),
        active_scaling_component_indices=np.asarray([0, 1], dtype=int),
        scaling_component_indices=np.asarray([0, 1, 2, 3, 4], dtype=int),
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        source_position_parameterization="prior-whitened",
        scaling_scatter_param_indices=np.asarray([5], dtype=int),
        source_metric_cache_by_z={2.0: {}},
    )

    monkeypatch.setattr(cluster_solver, "_log", lambda _args, message: captured_logs.append(str(message)))

    cluster_solver._log_solver_active_approximation_warning(argparse.Namespace(), evaluator)

    assert len(captured_logs) == 1
    warning = captured_logs[0]
    assert "refreshing_surrogate=active" in warning
    assert "z_bins=active grouped_families=2 bins=1" in warning
    assert "sample_likelihood=linearized-forward-beta-image-plane" in warning
    assert "source_position_parameterization=prior-whitened" in warning
    assert "active_scaling_subset=active 2/5" in warning
    assert "scaling_scatter_cache=linearized" in warning
    assert "source_metric_cache=refreshed" in warning


def test_solver_active_approximation_items_warn_when_image_presence_curvature_dominates() -> None:
    state = SimpleNamespace(
        family_data=[object()],
        bin_data=[
            SimpleNamespace(
                sigma_per_image=np.asarray([0.2, 0.2], dtype=float),
            )
        ],
        parameter_specs=[SimpleNamespace(component_family="source_position")],
        potfiles=[],
    )
    evaluator = SimpleNamespace(
        state=state,
        surrogate_enabled=False,
        inactive_scaling_component_indices=np.asarray([], dtype=int),
        active_scaling_component_indices=np.asarray([], dtype=int),
        scaling_component_indices=np.asarray([], dtype=int),
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        source_position_parameterization="conditional-whitened",
        image_presence_penalty_weight=3.0,
        image_presence_temperature_arcsec=0.05,
        image_plane_scatter_floor_arcsec=0.1,
        source_plane_covariance_floor=0.0,
        scaling_scatter_param_indices=np.asarray([], dtype=int),
        source_metric_cache_by_z={},
    )

    items = cluster_solver._solver_active_approximation_items(evaluator)

    assert any("image_presence_penalty=non-Gaussian conditional transport is approximate" in item for item in items)
    assert any("image_presence_penalty=curvature may dominate Gaussian beta transport" in item for item in items)


def test_solver_active_approximation_items_empty_for_exact_full_no_subset() -> None:
    state = SimpleNamespace(
        family_data=[object(), object()],
        bin_data=[object(), object()],
        parameter_specs=[],
        potfiles=[],
    )
    evaluator = SimpleNamespace(
        state=state,
        surrogate_enabled=False,
        inactive_scaling_component_indices=np.asarray([], dtype=int),
        active_scaling_component_indices=np.asarray([], dtype=int),
        scaling_component_indices=np.asarray([], dtype=int),
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        source_position_parameterization="direct",
        scaling_scatter_param_indices=np.asarray([], dtype=int),
        source_metric_cache_by_z={},
    )

    assert cluster_solver._solver_active_approximation_items(evaluator) == []


def test_solver_active_approximation_items_warn_for_active_subset() -> None:
    state = SimpleNamespace(
        family_data=[object()],
        bin_data=[object()],
        parameter_specs=[],
        potfiles=[{"id": "pot"}],
    )
    evaluator = SimpleNamespace(
        state=state,
        sampling_engine="active_subset",
        surrogate_enabled=False,
        inactive_scaling_component_indices=np.asarray([3, 4], dtype=int),
        active_scaling_component_indices=np.asarray([1, 2], dtype=int),
        scaling_component_indices=np.asarray([1, 2, 3, 4], dtype=int),
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        source_position_parameterization="direct",
        scaling_scatter_param_indices=np.asarray([], dtype=int),
        source_metric_cache_by_z={},
    )

    items = cluster_solver._solver_active_approximation_items(evaluator)

    assert any("active_subset=fit target omits inactive scaling potentials" in item for item in items)
    assert "active_scaling_subset=active 2/4" in items


def test_active_subset_fit_component_indices_include_large_and_active_only() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.sampling_engine = "active_subset"
    evaluator.scaling_component_indices = np.asarray([1, 2, 3], dtype=np.int32)
    evaluator.active_scaling_component_indices = np.asarray([1, 3], dtype=np.int32)
    evaluator.inactive_scaling_component_indices = np.asarray([2], dtype=np.int32)
    evaluator.active_component_indices = np.asarray([0, 1, 3], dtype=np.int32)

    np.testing.assert_array_equal(
        cluster_solver.ClusterJAXEvaluator._fit_component_indices(evaluator),
        np.asarray([0, 1, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        cluster_solver.ClusterJAXEvaluator._fit_scaling_component_indices(evaluator),
        np.asarray([1, 3], dtype=np.int32),
    )


def test_active_subset_with_no_inactive_components_behaves_like_full() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.sampling_engine = "active_subset"
    evaluator.scaling_component_indices = np.asarray([1, 2], dtype=np.int32)
    evaluator.active_scaling_component_indices = np.asarray([1, 2], dtype=np.int32)
    evaluator.inactive_scaling_component_indices = np.asarray([], dtype=np.int32)
    evaluator.active_component_indices = np.asarray([0, 1, 2], dtype=np.int32)

    assert cluster_solver.ClusterJAXEvaluator._fit_component_indices(evaluator) is None
    np.testing.assert_array_equal(
        cluster_solver.ClusterJAXEvaluator._fit_scaling_component_indices(evaluator),
        np.asarray([1, 2], dtype=np.int32),
    )


def test_active_subset_source_likelihood_passes_subset_to_ray_shooting() -> None:
    traced_bin = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0, 0], dtype=jnp.int32),
        x_obs=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        y_obs=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.2, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True], dtype=bool),
    )

    class FakeEvaluator:
        surrogate_enabled = False
        surrogate_cache_by_z: dict[float, object] = {}
        sample_likelihood_mode = SAMPLE_LIKELIHOOD_SOURCE
        traced_bin_data = [traced_bin]
        fit_cosmology_flat_wcdm = False
        source_plane_covariance_floor = 1.0e-6
        source_plane_outlier_sigma_arcsec = 10.0
        likelihood_stabilizer_max_gain = 0.0
        likelihood_stabilizer_max_residual_arcsec = 0.0
        likelihood_stabilizer_residual_loss = "gaussian"
        likelihood_stabilizer_student_t_nu = 4.0

        def __init__(self) -> None:
            self.seen_component_indices: list[np.ndarray | None] = []

        def _fit_component_indices(self):
            return np.asarray([0, 2], dtype=np.int32)

        def _physical_parameter_vector(self, params):
            return params

        def _source_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.0, dtype=jnp.float64)

        def _image_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.0, dtype=jnp.float64)

        def _build_packed_lens_state_with_validity_from_physical(self, *_args, **_kwargs):
            return {}, {"is_valid": jnp.asarray(True), "reason_flags": jnp.asarray([False])}

        def _maybe_record_invalid_state(self, _validity):
            return None

        def _ray_shooting_for_components(self, _z_source, x, y, _packed_state, component_indices=None):
            self.seen_component_indices.append(None if component_indices is None else np.asarray(component_indices))
            return x, y

        def _scaling_scatter_extra_variance_from_physical(self, _physical_params, _bin_data, beta_x, beta_y):
            return jnp.zeros_like(beta_x), jnp.zeros_like(beta_y)

        def _magnification_inv_abs_mu(self, bin_data):
            return jnp.ones_like(bin_data.sigma_per_image)

    fake = FakeEvaluator()

    loglike = cluster_solver.ClusterJAXEvaluator._source_loglike_impl(fake, jnp.asarray([0.0], dtype=jnp.float64))

    assert np.isfinite(float(loglike))
    assert len(fake.seen_component_indices) == 1
    np.testing.assert_array_equal(fake.seen_component_indices[0], np.asarray([0, 2], dtype=np.int32))


def test_surrogate_beta_and_jacobian_composes_active_exact_and_inactive_cache() -> None:
    traced_bin = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0, 0], dtype=jnp.int32),
        x_obs=jnp.asarray([10.0, 20.0], dtype=jnp.float64),
        y_obs=jnp.asarray([30.0, 40.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True], dtype=bool),
    )
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.surrogate_reference_param_values = np.asarray([1.0], dtype=float)
    evaluator.surrogate_param_indices_jax = jnp.asarray([0], dtype=jnp.int32)
    evaluator.active_component_indices = np.asarray([0], dtype=np.int32)
    evaluator.surrogate_cache_by_z = {
        2.0: SurrogateBinCache(
            effective_z_source=2.0,
            inactive_alpha_x=np.asarray([0.1, 0.2], dtype=float),
            inactive_alpha_y=np.asarray([0.3, 0.4], dtype=float),
            inactive_alpha_dx_dparams=np.asarray([[1.0, 2.0]], dtype=float),
            inactive_alpha_dy_dparams=np.asarray([[3.0, 4.0]], dtype=float),
            inactive_jacobian_delta_a00=np.asarray([0.01, 0.02], dtype=float),
            inactive_jacobian_delta_a01=np.asarray([-0.03, -0.01], dtype=float),
            inactive_jacobian_delta_a10=np.asarray([0.04, 0.05], dtype=float),
            inactive_jacobian_delta_a11=np.asarray([-0.02, -0.04], dtype=float),
            inactive_jacobian_delta_da00_dparams=np.asarray([[0.1, 0.2]], dtype=float),
            inactive_jacobian_delta_da01_dparams=np.asarray([[0.2, 0.1]], dtype=float),
            inactive_jacobian_delta_da10_dparams=np.asarray([[-0.1, 0.1]], dtype=float),
            inactive_jacobian_delta_da11_dparams=np.asarray([[0.3, -0.2]], dtype=float),
        )
    }
    evaluator._maybe_record_invalid_state = lambda _validity: None
    evaluator._build_packed_lens_state_with_validity_from_physical = lambda *_args, **_kwargs: (
        {"tag": jnp.asarray(1, dtype=jnp.int32)},
        {"is_valid": jnp.asarray(True), "reason_flags": jnp.asarray([False])},
    )

    def fake_ray_shooting(_z_source, x, y, _packed_state, component_indices):
        np.testing.assert_array_equal(component_indices, np.asarray([0], dtype=np.int32))
        return (
            x - jnp.asarray([0.5, 0.6], dtype=jnp.float64),
            y - jnp.asarray([0.7, 0.8], dtype=jnp.float64),
        )

    def fake_lensing_jacobian(_z_source, x, y, _packed_state, component_indices):
        np.testing.assert_array_equal(component_indices, np.asarray([0], dtype=np.int32))
        return (
            jnp.asarray([0.9, 0.8], dtype=jnp.float64) + 0.0 * x,
            jnp.asarray([0.01, 0.02], dtype=jnp.float64) + 0.0 * x,
            jnp.asarray([-0.02, -0.01], dtype=jnp.float64) + 0.0 * y,
            jnp.asarray([1.1, 1.2], dtype=jnp.float64) + 0.0 * y,
        )

    evaluator._ray_shooting_for_components = fake_ray_shooting
    evaluator._lensing_jacobian_for_components = fake_lensing_jacobian

    params = jnp.asarray([1.2], dtype=jnp.float64)
    beta_x, beta_y, invalid, packed_state = cluster_solver.ClusterJAXEvaluator._surrogate_beta(
        evaluator,
        params,
        params,
        traced_bin,
    )
    jacobian = cluster_solver.ClusterJAXEvaluator._surrogate_jacobian_entries(
        evaluator,
        params,
        traced_bin,
        packed_state,
        invalid,
    )

    np.testing.assert_allclose(np.asarray(beta_x), np.asarray([9.2, 18.8]))
    np.testing.assert_allclose(np.asarray(beta_y), np.asarray([28.4, 38.0]))
    np.testing.assert_allclose(np.asarray(jacobian[0]), np.asarray([0.93, 0.86]))
    np.testing.assert_allclose(np.asarray(jacobian[1]), np.asarray([0.02, 0.03]))
    np.testing.assert_allclose(np.asarray(jacobian[2]), np.asarray([0.0, 0.06]), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(jacobian[3]), np.asarray([1.14, 1.12]))


def test_surrogate_uses_cosmology_parameter_derivatives() -> None:
    traced_bin = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0, 0], dtype=jnp.int32),
        x_obs=jnp.asarray([10.0, 20.0], dtype=jnp.float64),
        y_obs=jnp.asarray([30.0, 40.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True], dtype=bool),
    )
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.surrogate_reference_param_values = np.asarray([1.0, 0.3, -1.0], dtype=float)
    evaluator.surrogate_param_indices_jax = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    evaluator.active_component_indices = np.asarray([0], dtype=np.int32)
    evaluator.surrogate_cache_by_z = {
        2.0: SurrogateBinCache(
            effective_z_source=2.0,
            inactive_alpha_x=np.asarray([0.1, 0.2], dtype=float),
            inactive_alpha_y=np.asarray([0.3, 0.4], dtype=float),
            inactive_alpha_dx_dparams=np.asarray([[1.0, 2.0], [10.0, 20.0], [-1.0, -2.0]], dtype=float),
            inactive_alpha_dy_dparams=np.asarray([[3.0, 4.0], [-2.0, -4.0], [6.0, 8.0]], dtype=float),
            inactive_jacobian_delta_a00=np.asarray([0.01, 0.02], dtype=float),
            inactive_jacobian_delta_a01=np.asarray([-0.03, -0.01], dtype=float),
            inactive_jacobian_delta_a10=np.asarray([0.04, 0.05], dtype=float),
            inactive_jacobian_delta_a11=np.asarray([-0.02, -0.04], dtype=float),
            inactive_jacobian_delta_da00_dparams=np.asarray(
                [[0.1, 0.2], [1.0, 2.0], [-0.5, -1.0]], dtype=float
            ),
            inactive_jacobian_delta_da01_dparams=np.asarray(
                [[0.2, 0.1], [0.0, 0.0], [1.0, 2.0]], dtype=float
            ),
            inactive_jacobian_delta_da10_dparams=np.asarray(
                [[-0.1, 0.1], [0.0, 0.0], [-0.2, 0.2]], dtype=float
            ),
            inactive_jacobian_delta_da11_dparams=np.asarray(
                [[0.3, -0.2], [-1.0, 1.0], [0.4, -0.4]], dtype=float
            ),
        )
    }
    evaluator._maybe_record_invalid_state = lambda _validity: None
    evaluator._build_packed_lens_state_with_validity_from_physical = lambda *_args, **_kwargs: (
        {"tag": jnp.asarray(1, dtype=jnp.int32)},
        {"is_valid": jnp.asarray(True), "reason_flags": jnp.asarray([False])},
    )
    evaluator._ray_shooting_for_components = lambda _z_source, x, y, _packed_state, _component_indices: (
        x - jnp.asarray([0.5, 0.6], dtype=jnp.float64),
        y - jnp.asarray([0.7, 0.8], dtype=jnp.float64),
    )
    evaluator._lensing_jacobian_for_components = lambda _z_source, x, y, _packed_state, _component_indices: (
        jnp.asarray([0.9, 0.8], dtype=jnp.float64) + 0.0 * x,
        jnp.asarray([0.01, 0.02], dtype=jnp.float64) + 0.0 * x,
        jnp.asarray([-0.02, -0.01], dtype=jnp.float64) + 0.0 * y,
        jnp.asarray([1.1, 1.2], dtype=jnp.float64) + 0.0 * y,
    )

    params = jnp.asarray([1.2, 0.31, -0.95], dtype=jnp.float64)
    beta_x, beta_y, invalid, packed_state = cluster_solver.ClusterJAXEvaluator._surrogate_beta(
        evaluator,
        params,
        params,
        traced_bin,
    )
    jacobian = cluster_solver.ClusterJAXEvaluator._surrogate_jacobian_entries(
        evaluator,
        params,
        traced_bin,
        packed_state,
        invalid,
    )

    np.testing.assert_allclose(np.asarray(beta_x), np.asarray([9.15, 18.7]))
    np.testing.assert_allclose(np.asarray(beta_y), np.asarray([28.12, 37.64]))
    np.testing.assert_allclose(np.asarray(jacobian[0]), np.asarray([0.915, 0.83]))
    np.testing.assert_allclose(np.asarray(jacobian[1]), np.asarray([0.07, 0.13]))
    np.testing.assert_allclose(np.asarray(jacobian[2]), np.asarray([-0.01, 0.07]), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(jacobian[3]), np.asarray([1.15, 1.11]))


def test_source_plane_surrogate_beta_allows_cache_without_jacobian_fields() -> None:
    traced_bin = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0, 0], dtype=jnp.int32),
        x_obs=jnp.asarray([10.0, 20.0], dtype=jnp.float64),
        y_obs=jnp.asarray([30.0, 40.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True], dtype=bool),
    )
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.surrogate_reference_param_values = np.asarray([1.0], dtype=float)
    evaluator.surrogate_param_indices_jax = jnp.asarray([0], dtype=jnp.int32)
    evaluator.active_component_indices = np.asarray([0], dtype=np.int32)
    evaluator.surrogate_cache_by_z = {
        2.0: SurrogateBinCache(
            effective_z_source=2.0,
            inactive_alpha_x=np.asarray([1.0, 2.0], dtype=float),
            inactive_alpha_y=np.asarray([3.0, 4.0], dtype=float),
            inactive_alpha_dx_dparams=np.asarray([[0.5, 0.25]], dtype=float),
            inactive_alpha_dy_dparams=np.asarray([[-0.5, -0.25]], dtype=float),
        )
    }
    evaluator._maybe_record_invalid_state = lambda _validity: None
    evaluator._build_packed_lens_state_with_validity_from_physical = lambda *_args, **_kwargs: (
        {},
        {"is_valid": jnp.asarray(True), "reason_flags": jnp.asarray([False])},
    )
    evaluator._ray_shooting_for_components = lambda _z_source, x, y, _packed_state, _component_indices: (
        x - jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        y - jnp.asarray([0.3, 0.4], dtype=jnp.float64),
    )

    beta_x, beta_y, invalid, _packed_state = cluster_solver.ClusterJAXEvaluator._surrogate_beta(
        evaluator,
        jnp.asarray([1.5], dtype=jnp.float64),
        jnp.asarray([1.5], dtype=jnp.float64),
        traced_bin,
    )

    assert bool(np.asarray(invalid)) is False
    np.testing.assert_allclose(np.asarray(beta_x), np.asarray([8.65, 17.675]))
    np.testing.assert_allclose(np.asarray(beta_y), np.asarray([26.95, 35.725]))


def test_explicit_beta_surrogate_branch_uses_returned_packed_state() -> None:
    traced_bin = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0], dtype=jnp.int32),
        x_obs=jnp.asarray([0.0], dtype=jnp.float64),
        y_obs=jnp.asarray([0.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True], dtype=bool),
    )

    class FakeEvaluator:
        source_position_conditional = False
        source_position_param_indices_by_family = {"1": (0, 1)}
        surrogate_enabled = True
        surrogate_cache_by_z = {2.0: object()}
        sample_likelihood_mode = SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
        image_plane_newton_steps = 0
        image_plane_scatter_floor_arcsec = 1.0e-3
        source_plane_covariance_floor = 1.0e-6
        source_plane_outlier_sigma_arcsec = 10.0
        image_presence_penalty_weight = 0.0
        image_presence_match_radius_arcsec = 0.30
        image_presence_temperature_arcsec = 0.10
        image_presence_count_softness = 0.05
        image_presence_count_margin = 0.05
        likelihood_stabilizer_max_gain = 0.0
        likelihood_stabilizer_max_residual_arcsec = 0.0
        likelihood_stabilizer_residual_loss = "gaussian"
        likelihood_stabilizer_student_t_nu = 4.0
        traced_bin_data = [traced_bin]

        def __init__(self) -> None:
            self.surrogate_calls = 0
            self.jacobian_calls = 0

        def _physical_parameter_vector(self, params):
            return params

        def _source_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.0, dtype=jnp.float64)

        def _image_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.05, dtype=jnp.float64)

        def _surrogate_beta(self, _params, _physical_params, bin_data):
            self.surrogate_calls += 1
            packed_state = {"tag": jnp.asarray(1, dtype=jnp.int32)}
            return bin_data.x_obs, bin_data.y_obs, jnp.asarray(False), packed_state

        def _surrogate_jacobian_entries(self, _params, bin_data, packed_state, _invalid):
            self.jacobian_calls += 1
            ones = jnp.ones_like(bin_data.x_obs) + 0.0 * packed_state["tag"]
            zeros = jnp.zeros_like(bin_data.y_obs)
            return ones, zeros, zeros, ones

        def _ray_shooting_for_components(self, _z_source, x, y, packed_state, *_args):
            return x + 0.0 * packed_state["tag"], y + 0.0 * packed_state["tag"]

        def _lensing_jacobian_for_components(self, _z_source, x, y, packed_state):
            self.jacobian_calls += 1
            ones = jnp.ones_like(x) + 0.0 * packed_state["tag"]
            zeros = jnp.zeros_like(y)
            return ones, zeros, zeros, ones

        def _scaling_scatter_extra_variance_from_physical(self, _physical_params, bin_data, _beta_x, _beta_y):
            return jnp.zeros_like(bin_data.x_obs), jnp.zeros_like(bin_data.y_obs)

        def _maybe_record_invalid_state(self, _validity):
            return None

        def _cab_morphology_loglike_for_bin(self, *_args, **_kwargs):
            return jnp.asarray(0.0, dtype=jnp.float64)

    fake = FakeEvaluator()
    fake._source_position_vectors_for_bin = cluster_solver.ClusterJAXEvaluator._source_position_vectors_for_bin.__get__(
        fake,
        FakeEvaluator,
    )
    fake._explicit_source_position_vectors_for_bin = (
        cluster_solver.ClusterJAXEvaluator._explicit_source_position_vectors_for_bin.__get__(fake, FakeEvaluator)
    )
    fake._linearized_image_plane_residuals_for_components = (
        cluster_solver.ClusterJAXEvaluator._linearized_image_plane_residuals_for_components.__get__(fake, FakeEvaluator)
    )
    fake._linearized_image_plane_residuals_from_observed_beta = (
        cluster_solver.ClusterJAXEvaluator._linearized_image_plane_residuals_from_observed_beta.__get__(
            fake,
            FakeEvaluator,
        )
    )

    loglike = cluster_solver.ClusterJAXEvaluator._source_loglike_impl(fake, jnp.asarray([0.0, 0.0], dtype=jnp.float64))

    assert np.isfinite(float(loglike))
    assert fake.surrogate_calls == 1
    assert fake.jacobian_calls >= 1


def test_anchored_solved_source_loglike_branch_scores_local_image_residuals() -> None:
    traced_bin = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0], dtype=jnp.int32),
        x_obs=jnp.asarray([0.0], dtype=jnp.float64),
        y_obs=jnp.asarray([0.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True], dtype=bool),
    )

    class FakeEvaluator:
        source_position_conditional = False
        source_position_param_indices_by_family = {"1": (0, 1)}
        surrogate_enabled = False
        surrogate_cache_by_z = {}
        sample_likelihood_mode = SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE
        anchored_image_plane_solve_steps = 3
        anchored_image_plane_trust_radius_arcsec = 0.3
        anchored_image_plane_lm_damping_relative = 1.0e-3
        anchored_image_plane_lm_damping_absolute = 1.0e-6
        image_plane_scatter_floor_arcsec = 1.0e-3
        source_plane_covariance_floor = 1.0e-6
        source_plane_outlier_sigma_arcsec = 10.0
        image_presence_penalty_weight = 2.0
        image_presence_match_radius_arcsec = 0.30
        image_presence_temperature_arcsec = 0.10
        image_presence_count_softness = 0.05
        image_presence_count_margin = 0.05
        likelihood_stabilizer_residual_loss = "student-t"
        likelihood_stabilizer_student_t_nu = 4.0
        traced_bin_data = [traced_bin]

        def _physical_parameter_vector(self, params):
            return params

        def _source_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.0, dtype=jnp.float64)

        def _image_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.05, dtype=jnp.float64)

        def _build_packed_lens_state_with_validity_from_physical(self, _physical_params, _z_source, stop_gradient=True):
            return {}, {"is_valid": jnp.asarray(True)}

        def _ray_shooting_for_components(self, _z_source, x, y, _packed_state):
            return x, y

        def _lensing_jacobian_for_components(self, _z_source, x, y, _packed_state):
            return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(y), jnp.ones_like(y)

        def _scaling_scatter_extra_variance_from_physical(self, _physical_params, bin_data, _beta_x, _beta_y):
            return jnp.zeros_like(bin_data.x_obs), jnp.zeros_like(bin_data.y_obs)

        def _maybe_record_invalid_state(self, _validity):
            return None

        def _cab_morphology_loglike_for_bin(self, *_args, **_kwargs):
            return jnp.asarray(0.0, dtype=jnp.float64)

    fake = FakeEvaluator()
    fake._source_position_vectors_for_bin = cluster_solver.ClusterJAXEvaluator._source_position_vectors_for_bin.__get__(
        fake,
        FakeEvaluator,
    )
    fake._explicit_source_position_vectors_for_bin = (
        cluster_solver.ClusterJAXEvaluator._explicit_source_position_vectors_for_bin.__get__(fake, FakeEvaluator)
    )
    fake._anchored_solved_image_plane_residuals_for_components = (
        cluster_solver.ClusterJAXEvaluator._anchored_solved_image_plane_residuals_for_components.__get__(
            fake,
            FakeEvaluator,
        )
    )

    loglike = cluster_solver.ClusterJAXEvaluator._source_loglike_impl(fake, jnp.asarray([0.0, 0.0], dtype=jnp.float64))

    assert np.isfinite(float(loglike))
    assert float(loglike) > 0.0


def test_zero_step_anchored_source_loglike_uses_observed_anchor_lm_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traced_bin = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0], dtype=jnp.int32),
        x_obs=jnp.asarray([0.0], dtype=jnp.float64),
        y_obs=jnp.asarray([0.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True], dtype=bool),
    )
    captured: dict[str, np.ndarray] = {}

    def fake_bin_loglike(**kwargs):
        captured["residual_x"] = np.asarray(kwargs["residual_x"], dtype=float)
        captured["residual_y"] = np.asarray(kwargs["residual_y"], dtype=float)
        return jnp.asarray(123.0, dtype=jnp.float64)

    monkeypatch.setattr(cluster_solver, "_linearized_image_plane_bin_loglike", fake_bin_loglike)

    class FakeEvaluator:
        source_position_conditional = False
        source_position_param_indices_by_family = {"1": (0, 1)}
        surrogate_enabled = False
        surrogate_cache_by_z = {}
        sample_likelihood_mode = SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE
        anchored_image_plane_solve_steps = 0
        anchored_image_plane_trust_radius_arcsec = 10.0
        anchored_image_plane_lm_damping_relative = 1.0e-3
        anchored_image_plane_lm_damping_absolute = 1.0e-6
        image_plane_scatter_floor_arcsec = 1.0e-3
        source_plane_covariance_floor = 1.0e-6
        source_plane_outlier_sigma_arcsec = 10.0
        image_presence_penalty_weight = 0.0
        image_presence_match_radius_arcsec = 0.30
        image_presence_temperature_arcsec = 0.10
        image_presence_count_softness = 0.05
        image_presence_count_margin = 0.05
        likelihood_stabilizer_residual_loss = "gaussian"
        likelihood_stabilizer_student_t_nu = 4.0
        traced_bin_data = [traced_bin]

        def _physical_parameter_vector(self, params):
            return params

        def _source_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.0, dtype=jnp.float64)

        def _image_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.05, dtype=jnp.float64)

        def _build_packed_lens_state_with_validity_from_physical(self, _physical_params, _z_source, stop_gradient=True):
            return {}, {"is_valid": jnp.asarray(True)}

        def _ray_shooting_for_components(self, _z_source, x, y, _packed_state):
            return x + 0.4, y - 0.1

        def _lensing_jacobian_for_components(self, _z_source, x, y, _packed_state):
            return (
                jnp.full_like(x, 2.0),
                jnp.zeros_like(x),
                jnp.zeros_like(y),
                jnp.ones_like(y),
            )

        def _anchored_solved_image_plane_residuals_for_components(self, *_args, **_kwargs):
            raise AssertionError("zero-step anchored likelihood must not call the iterative local solver")

        def _scaling_scatter_extra_variance_from_physical(self, _physical_params, bin_data, _beta_x, _beta_y):
            return jnp.zeros_like(bin_data.x_obs), jnp.zeros_like(bin_data.y_obs)

        def _maybe_record_invalid_state(self, _validity):
            return None

        def _cab_morphology_loglike_for_bin(self, *_args, **_kwargs):
            return jnp.asarray(0.0, dtype=jnp.float64)

    fake = FakeEvaluator()
    fake._source_position_vectors_for_bin = cluster_solver.ClusterJAXEvaluator._source_position_vectors_for_bin.__get__(
        fake,
        FakeEvaluator,
    )
    fake._explicit_source_position_vectors_for_bin = (
        cluster_solver.ClusterJAXEvaluator._explicit_source_position_vectors_for_bin.__get__(fake, FakeEvaluator)
    )

    params = jnp.asarray([0.1, -0.2], dtype=jnp.float64)
    loglike = cluster_solver.ClusterJAXEvaluator._source_loglike_impl(fake, params)
    expected_dx, expected_dy, expected_finite = _anchored_solved_image_plane_step_from_jacobian(
        f_x=jnp.asarray([0.3], dtype=jnp.float64),
        f_y=jnp.asarray([0.1], dtype=jnp.float64),
        jac_a00=jnp.asarray([2.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
        trust_radius_arcsec=10.0,
        lm_damping_relative=1.0e-3,
        lm_damping_absolute=1.0e-6,
    )

    assert float(loglike) == pytest.approx(123.0)
    assert bool(np.asarray(expected_finite)[0])
    np.testing.assert_allclose(captured["residual_x"], np.asarray(expected_dx), rtol=1.0e-12)
    np.testing.assert_allclose(captured["residual_y"], np.asarray(expected_dy), rtol=1.0e-12)


def _critical_arc_loglike_for_test(
    *,
    residual_x: float,
    residual_y: float,
    jac_a00: float = 0.0,
    jac_a01: float = 0.0,
    jac_a10: float = 0.0,
    jac_a11: float = 1.0,
    reliability: float = 0.999,
    outlier_sigma_arcsec: float = 50.0,
) -> float:
    return float(
        _critical_arc_mixture_image_plane_bin_loglike(
            residual_x=jnp.asarray([residual_x], dtype=jnp.float64),
            residual_y=jnp.asarray([residual_y], dtype=jnp.float64),
            jac_a00=jnp.asarray([jac_a00], dtype=jnp.float64),
            jac_a01=jnp.asarray([jac_a01], dtype=jnp.float64),
            jac_a10=jnp.asarray([jac_a10], dtype=jnp.float64),
            jac_a11=jnp.asarray([jac_a11], dtype=jnp.float64),
            family_idx=None,
            n_families=None,
            sigma_per_image=jnp.asarray([0.05], dtype=jnp.float64),
            reliability_per_image=jnp.asarray([reliability], dtype=jnp.float64),
            image_has_constraint=jnp.asarray([True], dtype=bool),
            image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
            covariance_floor=1.0e-12,
            outlier_sigma_arcsec=outlier_sigma_arcsec,
            image_scatter_floor_arcsec=1.0e-6,
            image_presence_penalty_weight=0.0,
            residual_loss="student-t",
            student_t_nu=4.0,
            critical_direction_sigma_arcsec=5.0,
            base_prob=0.10,
            max_prob=0.80,
            singular_threshold=0.20,
            singular_softness=0.05,
        )
    )


def test_critical_arc_branch_probability_increases_near_singular() -> None:
    probabilities = np.asarray(
        _critical_arc_branch_probability(
            jnp.asarray([1.0, 0.01], dtype=jnp.float64),
            base_prob=0.10,
            max_prob=0.80,
            singular_threshold=0.20,
            singular_softness=0.05,
        )
    )

    assert probabilities[1] > probabilities[0]
    assert probabilities[0] >= 0.10
    assert probabilities[1] <= 0.80


def test_critical_arc_lm_step_rank_deficient_jacobian_is_finite_and_bounded() -> None:
    dx, dy, finite = _critical_arc_lm_step_from_jacobian(
        f_x=jnp.asarray([100.0], dtype=jnp.float64),
        f_y=jnp.asarray([-50.0], dtype=jnp.float64),
        jac_a00=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([0.0], dtype=jnp.float64),
        trust_radius_arcsec=20.0,
        lm_damping_relative=1.0e-3,
        lm_damping_absolute=1.0e-6,
    )

    radius = np.hypot(np.asarray(dx), np.asarray(dy))
    assert bool(np.asarray(finite)[0])
    assert np.isfinite(radius).all()
    assert float(radius[0]) <= 20.0


def test_critical_arc_fused_lm_geometry_matches_anchored_step() -> None:
    f_x = jnp.asarray([0.3, -0.2], dtype=jnp.float64)
    f_y = jnp.asarray([0.1, 0.4], dtype=jnp.float64)
    jac_a00 = jnp.asarray([2.0, 0.4], dtype=jnp.float64)
    jac_a01 = jnp.asarray([0.1, -0.3], dtype=jnp.float64)
    jac_a10 = jnp.asarray([0.0, 0.2], dtype=jnp.float64)
    jac_a11 = jnp.asarray([1.0, 0.8], dtype=jnp.float64)

    fused_dx, fused_dy, *_geometry, fused_finite = _critical_arc_lm_geometry_from_jacobian(
        f_x,
        f_y,
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
        trust_radius_arcsec=20.0,
        lm_damping_relative=1.0e-3,
        lm_damping_absolute=1.0e-6,
    )
    anchored_dx, anchored_dy, anchored_finite = _anchored_solved_image_plane_step_from_jacobian(
        f_x,
        f_y,
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
        trust_radius_arcsec=20.0,
        lm_damping_relative=1.0e-3,
        lm_damping_absolute=1.0e-6,
    )

    np.testing.assert_allclose(np.asarray(fused_dx), np.asarray(anchored_dx), rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(fused_dy), np.asarray(anchored_dy), rtol=1.0e-12)
    np.testing.assert_array_equal(np.asarray(fused_finite), np.asarray(anchored_finite))


def test_critical_arc_projector_quadratics_match_frame_projection() -> None:
    residual_x = jnp.asarray([0.7, -0.4], dtype=jnp.float64)
    residual_y = jnp.asarray([0.2, 0.5], dtype=jnp.float64)
    jac_a00 = jnp.asarray([2.0, 0.4], dtype=jnp.float64)
    jac_a01 = jnp.asarray([0.3, -0.2], dtype=jnp.float64)
    jac_a10 = jnp.asarray([0.1, 0.1], dtype=jnp.float64)
    jac_a11 = jnp.asarray([1.0, 0.8], dtype=jnp.float64)

    *_singulars, critical_p00, critical_p01, critical_p11, finite = _critical_arc_geometry_from_jacobian(
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
    )
    critical_direction_quad, noncritical_direction_quad = _critical_arc_projected_quadratics(
        residual_x,
        residual_y,
        critical_p00,
        critical_p01,
        critical_p11,
    )
    critical_direction_x, critical_direction_y, noncritical_direction_x, noncritical_direction_y, *_frame = cluster_solver._critical_arc_jacobian_frame(
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
    )
    expected_critical_direction_quad = jnp.square(residual_x * critical_direction_x + residual_y * critical_direction_y)
    expected_noncritical_direction_quad = jnp.square(residual_x * noncritical_direction_x + residual_y * noncritical_direction_y)

    assert bool(np.asarray(jnp.all(finite)))
    np.testing.assert_allclose(np.asarray(critical_direction_quad), np.asarray(expected_critical_direction_quad), rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(noncritical_direction_quad), np.asarray(expected_noncritical_direction_quad), rtol=1.0e-12)


def test_critical_arc_mixture_is_finite_for_singular_jacobian() -> None:
    loglike = _critical_arc_loglike_for_test(
        residual_x=2.0,
        residual_y=-1.0,
        jac_a00=0.0,
        jac_a01=0.0,
        jac_a10=0.0,
        jac_a11=0.0,
    )

    assert np.isfinite(loglike)


def test_critical_arc_large_critical_direction_displacement_is_cheaper_than_noncritical() -> None:
    critical_direction_loglike = _critical_arc_loglike_for_test(residual_x=2.0, residual_y=0.0)
    noncritical_direction_loglike = _critical_arc_loglike_for_test(residual_x=0.0, residual_y=2.0)

    assert critical_direction_loglike > noncritical_direction_loglike


def test_critical_arc_image_presence_counts_along_arc_displacement_as_recovered() -> None:
    kwargs = dict(
        jac_a00=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.05], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True], dtype=bool),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=50.0,
        image_scatter_floor_arcsec=1.0e-6,
        image_presence_match_radius_arcsec=0.5,
        image_presence_temperature_arcsec=0.1,
        image_presence_count_softness=0.05,
        image_presence_count_margin=0.05,
        residual_loss="student-t",
        student_t_nu=4.0,
        critical_direction_sigma_arcsec=5.0,
        base_prob=0.10,
        max_prob=0.85,
        singular_threshold=0.20,
        singular_softness=0.05,
    )

    def penalty_delta(residual_x: float, residual_y: float) -> float:
        base = _critical_arc_mixture_image_plane_bin_loglike(
            residual_x=jnp.asarray([residual_x], dtype=jnp.float64),
            residual_y=jnp.asarray([residual_y], dtype=jnp.float64),
            image_presence_penalty_weight=0.0,
            **kwargs,
        )
        with_presence = _critical_arc_mixture_image_plane_bin_loglike(
            residual_x=jnp.asarray([residual_x], dtype=jnp.float64),
            residual_y=jnp.asarray([residual_y], dtype=jnp.float64),
            image_presence_penalty_weight=2.0,
            **kwargs,
        )
        return float(with_presence - base)

    along_arc_penalty = penalty_delta(2.0, 0.0)
    noncritical_penalty = penalty_delta(0.0, 2.0)

    assert along_arc_penalty > -0.05
    assert noncritical_penalty < -0.5
    assert along_arc_penalty > noncritical_penalty + 0.5


def test_critical_arc_student_t_keeps_critical_direction_cheaper_than_noncritical() -> None:
    kwargs = dict(
        jac_a00=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
        family_idx=None,
        n_families=None,
        sigma_per_image=jnp.asarray([0.05], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True], dtype=bool),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=50.0,
        image_scatter_floor_arcsec=1.0e-6,
        image_presence_penalty_weight=0.0,
        residual_loss="student-t",
        student_t_nu=4.0,
        critical_direction_sigma_arcsec=5.0,
        base_prob=0.10,
        max_prob=0.80,
        singular_threshold=0.20,
        singular_softness=0.05,
    )

    critical_direction_loglike = _critical_arc_mixture_image_plane_bin_loglike(
        residual_x=jnp.asarray([2.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0], dtype=jnp.float64),
        **kwargs,
    )
    noncritical_direction_loglike = _critical_arc_mixture_image_plane_bin_loglike(
        residual_x=jnp.asarray([0.0], dtype=jnp.float64),
        residual_y=jnp.asarray([2.0], dtype=jnp.float64),
        **kwargs,
    )

    assert np.isfinite(float(critical_direction_loglike))
    assert np.isfinite(float(noncritical_direction_loglike))
    assert float(critical_direction_loglike) > float(noncritical_direction_loglike)


def test_critical_arc_det_sign_crossing_stays_finite_and_continuous() -> None:
    positive = _critical_arc_loglike_for_test(
        residual_x=0.4,
        residual_y=0.2,
        jac_a00=1.0,
        jac_a11=0.02,
    )
    negative = _critical_arc_loglike_for_test(
        residual_x=0.4,
        residual_y=0.2,
        jac_a00=1.0,
        jac_a11=-0.02,
    )

    assert np.isfinite(positive)
    assert np.isfinite(negative)
    assert positive == pytest.approx(negative, abs=1.0e-12)


def test_critical_arc_reliability_outlier_mixture_remains_active() -> None:
    high_reliability = _critical_arc_loglike_for_test(residual_x=0.0, residual_y=8.0, reliability=0.999)
    low_reliability = _critical_arc_loglike_for_test(residual_x=0.0, residual_y=8.0, reliability=0.10)

    assert low_reliability > high_reliability


def _constant_arc_jacobian(eps: float = 0.05):
    def jacobian_at(x_values, _y_values):
        x_array = np.asarray(x_values, dtype=float).reshape(-1)
        return (
            np.full_like(x_array, eps, dtype=float),
            np.zeros_like(x_array, dtype=float),
            np.zeros_like(x_array, dtype=float),
            np.ones_like(x_array, dtype=float),
        )

    return jacobian_at


def _circle_tangent_arc_jacobian(eps: float = 0.05):
    def jacobian_at(x_values, y_values):
        x_array = np.asarray(x_values, dtype=float).reshape(-1)
        y_array = np.asarray(y_values, dtype=float).reshape(-1)
        radius = np.maximum(np.hypot(x_array, y_array), 1.0e-12)
        tangent_x = -y_array / radius
        tangent_y = x_array / radius
        normal_x = x_array / radius
        normal_y = y_array / radius
        a00 = eps * tangent_x * tangent_x + normal_x * normal_x
        a01 = eps * tangent_x * tangent_y + normal_x * normal_y
        a10 = eps * tangent_y * tangent_x + normal_y * normal_x
        a11 = eps * tangent_y * tangent_y + normal_y * normal_y
        return a00, a01, a10, a11

    return jacobian_at


def test_critical_arc_aware_support_prioritizes_point_recovery() -> None:
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([10.0]),
        beta_residual_y=np.asarray([10.0]),
        jac_a00=np.asarray([0.05]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        point_recovered_mask=np.asarray([True]),
        point_residual_arcsec=np.asarray([0.07]),
    )

    assert details["arc_recovery_status"].tolist() == ["point_recovered"]
    assert details["arc_supported_mask"].tolist() == [False]
    assert details["arc_aware_image_residual_arcsec"][0] == pytest.approx(0.07)
    assert details["arc_aware_image_rms_arcsec"] == pytest.approx(0.07)


def test_critical_arc_aware_support_tolerates_large_critical_direction_residual() -> None:
    eps = 0.05
    critical_direction_delta = 4.0
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-eps * critical_direction_delta]),
        beta_residual_y=np.asarray([0.0]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=_constant_arc_jacobian(eps),
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_supported_mask"].tolist() == [True]
    assert details["arc_noncritical_direction_residual_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_critical_direction_residual_arcsec"][0] > 1.0
    assert details["arc_aware_image_residual_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_curve_distance_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_curve_arclength_arcsec"][0] == pytest.approx(critical_direction_delta, abs=0.12)
    assert details["arc_curve_finite"].tolist() == [True]
    assert json.loads(details["arc_support_curve_x_arcsec"][0])
    assert json.loads(details["arc_support_curve_y_arcsec"][0])
    assert math.hypot(details["arc_critical_direction_x"][0], details["arc_critical_direction_y"][0]) == pytest.approx(1.0)
    assert math.hypot(details["arc_noncritical_direction_x"][0], details["arc_noncritical_direction_y"][0]) == pytest.approx(1.0)
    assert details["arc_s_min"][0] == pytest.approx(eps)
    assert details["arc_s_max"][0] == pytest.approx(1.0)
    assert details["arc_detA"][0] == pytest.approx(eps)
    assert details["arc_support_finite_mask"].tolist() == [True]


def test_critical_arc_aware_support_rejects_large_noncritical_direction_residual() -> None:
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([0.0]),
        beta_residual_y=np.asarray([-1.0]),
        jac_a00=np.asarray([0.05]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=_constant_arc_jacobian(0.05),
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_recovery_status"].tolist() == ["not_recovered"]
    assert details["arc_supported_mask"].tolist() == [False]
    assert details["arc_curve_distance_arcsec"][0] > 0.5
    assert np.isnan(details["arc_aware_image_residual_arcsec"][0])


def test_critical_arc_aware_support_rejects_beyond_arclength_cap() -> None:
    eps = 0.05
    critical_direction_delta = 6.0
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-eps * critical_direction_delta]),
        beta_residual_y=np.asarray([0.0]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=_constant_arc_jacobian(eps),
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_recovery_status"].tolist() == ["not_recovered"]
    assert details["arc_supported_mask"].tolist() == [False]
    assert details["arc_curve_distance_arcsec"][0] > 0.5
    assert np.isnan(details["arc_aware_image_residual_arcsec"][0])


def test_critical_arc_aware_support_accepts_curved_support_curve() -> None:
    eps = 0.05
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-eps]),
        beta_residual_y=np.asarray([1.0]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([1.0]),
        jacobian_at=_circle_tangent_arc_jacobian(eps),
        curve_step_arcsec=0.05,
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_supported_mask"].tolist() == [True]
    assert details["arc_curve_distance_arcsec"][0] < 0.08
    assert 0.0 < details["arc_curve_arclength_arcsec"][0] <= cluster_solver.DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC
    assert details["arc_aware_image_residual_arcsec"][0] == pytest.approx(details["arc_curve_distance_arcsec"][0])


def test_critical_arc_aware_support_rejects_noncritical_large_critical_direction_residual() -> None:
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-4.0]),
        beta_residual_y=np.asarray([0.0]),
        jac_a00=np.asarray([1.0]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=_constant_arc_jacobian(1.0),
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_recovery_status"].tolist() == ["not_recovered"]
    assert details["arc_supported_mask"].tolist() == [False]
    assert np.isnan(details["arc_aware_image_residual_arcsec"][0])


def test_critical_arc_aware_support_requires_finite_curve_trace() -> None:
    def broken_jacobian_at(_x_values, _y_values):
        raise RuntimeError("no curve")

    eps = 0.05
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-eps * 4.0]),
        beta_residual_y=np.asarray([0.0]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=broken_jacobian_at,
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_recovery_status"].tolist() == ["not_recovered"]
    assert details["arc_supported_mask"].tolist() == [False]
    assert details["arc_curve_finite"].tolist() == [False]
    assert np.isnan(details["arc_aware_image_residual_arcsec"][0])


def test_critical_arc_aware_evaluator_uses_configured_validation_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        n_images=1,
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
    )
    fake = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    fake.models_by_effective_z = {2.0: object()}
    fake.arc_aware_noncritical_support_radius_arcsec = 1.25
    fake.arc_aware_max_arclength_arcsec = 12.0
    fake.arc_aware_curve_step_arcsec = 0.2
    fake.critical_arc_lm_trust_radius_arcsec = cluster_solver.DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC
    fake.critical_arc_lm_damping_relative = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE
    fake.critical_arc_lm_damping_absolute = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE
    fake.critical_arc_base_prob = cluster_solver.DEFAULT_CRITICAL_ARC_BASE_PROB
    fake.critical_arc_max_prob = cluster_solver.DEFAULT_CRITICAL_ARC_MAX_PROB
    fake.critical_arc_singular_threshold = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD
    fake.critical_arc_singular_softness = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS
    fake._build_packed_lens_state = lambda _params, _z_source: {}

    def fake_ray_shooting(_z_source, _x, _y, _packed_state):
        return jnp.asarray([0.0], dtype=jnp.float64), jnp.asarray([0.0], dtype=jnp.float64)

    fake._ray_shooting_for_components = fake_ray_shooting
    fake._lensing_jacobian_for_components = lambda _z_source, _x, _y, _packed_state: (
        jnp.asarray([0.05], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
    )
    captured: dict[str, float] = {}

    def fake_arc_support(*_args, **kwargs):
        captured["noncritical_support_radius_arcsec"] = float(kwargs["noncritical_support_radius_arcsec"])
        captured["max_arclength_arcsec"] = float(kwargs["max_arclength_arcsec"])
        captured["curve_step_arcsec"] = float(kwargs["curve_step_arcsec"])
        return {"sentinel": True}

    monkeypatch.setattr(cluster_solver, "_arc_aware_image_support_from_local_linearization", fake_arc_support)

    details = cluster_solver.ClusterJAXEvaluator._arc_aware_image_support_details(
        fake,
        np.asarray([], dtype=float),
        family,
        0.0,
        0.0,
    )

    assert details == {"sentinel": True}
    assert captured["noncritical_support_radius_arcsec"] == pytest.approx(1.25)
    assert captured["max_arclength_arcsec"] == pytest.approx(12.0)
    assert captured["curve_step_arcsec"] == pytest.approx(0.2)


def test_critical_arc_aware_recovery_rows_and_counts_propagate() -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        n_images=3,
        sigma_arcsec=0.2,
        image_labels=["1.1", "1.2", "1.3"],
        x_obs=np.asarray([0.0, 1.0, 2.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0, 0.0], dtype=float),
    )
    exact_details = {
        "produced_image_count": 1,
        "recovered_image_count": 1,
        "missing_image_count": 2,
        "extra_image_count": 0,
        "multiplicity_failed": True,
        "multiplicity_failure_reason": "missing_model_images",
        "matched_model_x_arcsec": np.asarray([0.05, np.nan, np.nan], dtype=float),
        "matched_model_y_arcsec": np.asarray([0.0, np.nan, np.nan], dtype=float),
        "recovered_image_mask": np.asarray([True, False, False], dtype=bool),
        "arc_recovery_status": np.asarray(["point_recovered", "arc_supported", "not_recovered"], dtype=object),
        "arc_aware_image_residual_arcsec": np.asarray([0.05, 0.08, np.nan], dtype=float),
        "arc_noncritical_direction_residual_arcsec": np.asarray([0.02, 0.08, 0.6], dtype=float),
        "arc_critical_direction_residual_arcsec": np.asarray([0.04, 5.0, 1.0], dtype=float),
        "arc_critical_direction_x": np.asarray([1.0, 0.0, 0.0], dtype=float),
        "arc_critical_direction_y": np.asarray([0.0, 1.0, 1.0], dtype=float),
        "arc_noncritical_direction_x": np.asarray([0.0, 1.0, 1.0], dtype=float),
        "arc_noncritical_direction_y": np.asarray([1.0, 0.0, 0.0], dtype=float),
        "arc_s_min": np.asarray([0.1, 0.04, 0.04], dtype=float),
        "arc_s_max": np.asarray([1.0, 1.2, 1.3], dtype=float),
        "arc_detA": np.asarray([0.1, 0.048, 0.052], dtype=float),
        "arc_prior_probability": np.asarray([0.6, 0.75, 0.75], dtype=float),
        "arc_curve_distance_arcsec": np.asarray([np.nan, 0.08, np.nan], dtype=float),
        "arc_curve_arclength_arcsec": np.asarray([np.nan, 3.2, np.nan], dtype=float),
        "arc_curve_finite": np.asarray([False, True, False], dtype=bool),
        "arc_support_anchor_x_arcsec": np.asarray([np.nan, 1.0, np.nan], dtype=float),
        "arc_support_anchor_y_arcsec": np.asarray([np.nan, 3.2, np.nan], dtype=float),
        "arc_support_curve_x_arcsec": np.asarray(["[]", "[1,1,1]", "[]"], dtype=object),
        "arc_support_curve_y_arcsec": np.asarray(["[]", "[-1,0,1]", "[]"], dtype=object),
        "arc_supported_mask": np.asarray([False, True, False], dtype=bool),
        "arc_support_finite_mask": np.asarray([True, True, True], dtype=bool),
        "cab_has_constraint": np.asarray([False, True, False], dtype=bool),
        "cab_anchor_x_arcsec": np.asarray([np.nan, 1.1, np.nan], dtype=float),
        "cab_anchor_y_arcsec": np.asarray([np.nan, 0.2, np.nan], dtype=float),
        "cab_tangent_angle_obs_rad": np.asarray([np.nan, 0.8, np.nan], dtype=float),
        "cab_tangent_angle_model_rad": np.asarray([np.nan, 0.85, np.nan], dtype=float),
        "cab_tangent_residual_rad": np.asarray([np.nan, 0.05, np.nan], dtype=float),
        "cab_curvature_obs_arcsec_inv": np.asarray([np.nan, 0.09, np.nan], dtype=float),
        "cab_curvature_model_arcsec_inv": np.asarray([np.nan, 0.11, np.nan], dtype=float),
        "cab_curvature_residual_arcsec_inv": np.asarray([np.nan, 0.02, np.nan], dtype=float),
        "cab_loglike": np.asarray([0.0, -1.25, 0.0], dtype=float),
        "cab_finite": np.asarray([False, True, False], dtype=bool),
        "failed": True,
    }

    image_rows, _extra_rows, _count_info = image_diagnostics.family_image_recovery_rows(family, exact_details)
    row_by_label = {row["image_label"]: row for row in image_rows}

    assert row_by_label["1.1"]["arc_recovery_status"] == "point_recovered"
    assert row_by_label["1.1"]["arc_aware_image_residual_arcsec"] == pytest.approx(0.05)
    assert row_by_label["1.2"]["image_recovery_status"] == "not_recovered"
    assert row_by_label["1.2"]["arc_recovery_status"] == "arc_supported"
    assert row_by_label["1.2"]["arc_supported"] is True
    assert row_by_label["1.2"]["arc_support_finite"] is True
    assert row_by_label["1.2"]["arc_critical_direction_y"] == pytest.approx(1.0)
    assert row_by_label["1.2"]["arc_s_max"] == pytest.approx(1.2)
    assert row_by_label["1.2"]["arc_detA"] == pytest.approx(0.048)
    assert row_by_label["1.2"]["arc_aware_image_residual_arcsec"] == pytest.approx(0.08)
    assert row_by_label["1.2"]["arc_curve_distance_arcsec"] == pytest.approx(0.08)
    assert row_by_label["1.2"]["arc_curve_arclength_arcsec"] == pytest.approx(3.2)
    assert row_by_label["1.2"]["arc_curve_finite"] is True
    assert row_by_label["1.2"]["arc_support_curve_x_arcsec"] == "[1,1,1]"
    assert row_by_label["1.2"]["arc_support_curve_y_arcsec"] == "[-1,0,1]"
    assert row_by_label["1.2"]["cab_has_constraint"] is True
    assert row_by_label["1.2"]["cab_anchor_x_arcsec"] == pytest.approx(1.1)
    assert row_by_label["1.2"]["cab_tangent_residual_rad"] == pytest.approx(0.05)
    assert row_by_label["1.2"]["cab_curvature_residual_arcsec_inv"] == pytest.approx(0.02)
    assert row_by_label["1.2"]["cab_loglike"] == pytest.approx(-1.25)
    assert row_by_label["1.2"]["cab_finite"] is True
    assert row_by_label["1.3"]["arc_recovery_status"] == "not_recovered"
    assert np.isnan(row_by_label["1.3"]["arc_aware_image_residual_arcsec"])

    table = image_diagnostics.image_count_recovery_table(SimpleNamespace(family_data=[family]), pd.DataFrame(image_rows))
    assert int(table.loc[0, "arc_aware_recovered_image_count"]) == 2
    assert int(table.loc[0, "arc_aware_missing_image_count"]) == 1
    assert int(table.loc[0, "arc_supported_image_count"]) == 1
    assert table.loc[0, "arc_aware_image_rms_arcsec"] == pytest.approx(math.sqrt((0.05**2 + 0.08**2) / 2.0))
    assert not image_diagnostics.exact_details_hard_failed(
        {
            "multiplicity_failure_reason": "exact_image_prediction_failed",
            "arc_aware_image_residual_arcsec": np.asarray([np.nan, 0.08], dtype=float),
            "failed": True,
        }
    )
    assert image_diagnostics.exact_details_hard_failed(
        {
            "multiplicity_failure_reason": "exact_image_prediction_failed",
            "arc_aware_image_residual_arcsec": np.asarray([np.nan, np.nan], dtype=float),
            "failed": True,
        }
    )


def test_critical_arc_aware_support_survives_exact_solver_failure() -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.2,
        n_images=1,
        image_labels=["1.1"],
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
    )
    fake = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    fake.validation_cache = {"1": FamilyValidationCache()}
    fake.source_plane_covariance_floor = 0.0
    fake.sample_likelihood_mode = SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE
    fake.models_by_effective_z = {2.0: object()}
    fake.critical_arc_lm_trust_radius_arcsec = cluster_solver.DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC
    fake.critical_arc_lm_damping_relative = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE
    fake.critical_arc_lm_damping_absolute = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE
    fake.critical_arc_base_prob = cluster_solver.DEFAULT_CRITICAL_ARC_BASE_PROB
    fake.critical_arc_max_prob = cluster_solver.DEFAULT_CRITICAL_ARC_MAX_PROB
    fake.critical_arc_singular_threshold = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD
    fake.critical_arc_singular_softness = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS
    fake._build_packed_lens_state = lambda _params, _z_source: {}
    fake._source_sigma_int_numpy = lambda _params: 0.0
    fake._source_position_for_family_numpy = lambda _params, _family_id: (0.0, 0.0)
    fake._exact_source_ray_shooting = (
        lambda _family, _packed_state: (np.asarray([-0.2], dtype=float), np.asarray([0.0], dtype=float))
    )
    fake._ray_shooting_for_components = (
        lambda _z_source, _x, _y, _packed_state: (jnp.asarray([-0.2], dtype=jnp.float64), jnp.asarray([0.0], dtype=jnp.float64))
    )
    fake._lensing_jacobian_for_components = lambda _z_source, _x, _y, _packed_state: (
        jnp.asarray([0.05], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
    )

    def fail_exact_solver(_family, _packed_state, _source_x, _source_y):
        raise RuntimeError("solver failed")

    fake._solve_exact_images_lenstronomy = fail_exact_solver

    details = cluster_solver.ClusterJAXEvaluator._exact_family_prediction_details(
        fake,
        np.asarray([], dtype=float),
        family,
    )

    assert details["failed"] is True
    assert details["multiplicity_failure_reason"] == "exact_image_prediction_failed"
    assert details["arc_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_supported_mask"].tolist() == [True]
    assert np.isfinite(details["arc_aware_image_residual_arcsec"][0])
    assert not image_diagnostics.exact_details_hard_failed(details)


def test_source_mode_exact_recovery_computes_arc_support_from_centroid_after_solver_failure() -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.2,
        n_images=1,
        image_labels=["1.1"],
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
    )
    fake = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    fake.validation_cache = {"1": FamilyValidationCache()}
    fake.source_plane_covariance_floor = 0.0
    fake.sample_likelihood_mode = SAMPLE_LIKELIHOOD_SOURCE
    fake.models_by_effective_z = {2.0: object()}
    fake.critical_arc_lm_trust_radius_arcsec = cluster_solver.DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC
    fake.critical_arc_lm_damping_relative = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE
    fake.critical_arc_lm_damping_absolute = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE
    fake.critical_arc_base_prob = cluster_solver.DEFAULT_CRITICAL_ARC_BASE_PROB
    fake.critical_arc_max_prob = cluster_solver.DEFAULT_CRITICAL_ARC_MAX_PROB
    fake.critical_arc_singular_threshold = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD
    fake.critical_arc_singular_softness = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS
    fake._build_packed_lens_state = lambda _params, _z_source: {}
    fake._source_sigma_int_numpy = lambda _params: 0.0
    fake._source_position_for_family_numpy = lambda _params, _family_id: (50.0, 50.0)
    fake._exact_source_ray_shooting = (
        lambda _family, _packed_state: (np.asarray([-0.2], dtype=float), np.asarray([0.0], dtype=float))
    )
    fake._ray_shooting_for_components = (
        lambda _z_source, _x, _y, _packed_state: (jnp.asarray([-0.2], dtype=jnp.float64), jnp.asarray([0.0], dtype=jnp.float64))
    )
    fake._lensing_jacobian_for_components = lambda _z_source, _x, _y, _packed_state: (
        jnp.asarray([0.05], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
    )

    def fail_exact_solver(_family, _packed_state, _source_x, _source_y):
        raise RuntimeError("solver failed")

    fake._solve_exact_images_lenstronomy = fail_exact_solver

    details = cluster_solver.ClusterJAXEvaluator._exact_family_prediction_details(
        fake,
        np.asarray([], dtype=float),
        family,
    )

    assert details["failed"] is True
    assert details["multiplicity_failure_reason"] == "exact_image_prediction_failed"
    assert details["arc_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_supported_mask"].tolist() == [True]
    assert details["arc_s_min"][0] == pytest.approx(0.05)
    assert details["arc_noncritical_direction_residual_arcsec"][0] == pytest.approx(0.0, abs=1.0e-8)
    assert np.isfinite(details["arc_critical_direction_residual_arcsec"][0])
    assert np.isfinite(details["arc_prior_probability"][0])
    assert not image_diagnostics.exact_details_hard_failed(details)


def test_exact_recovery_arc_support_falls_back_when_jacobian_unavailable() -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.2,
        n_images=1,
        image_labels=["1.1"],
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
    )
    fake = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    fake.validation_cache = {"1": FamilyValidationCache()}
    fake.source_plane_covariance_floor = 0.0
    fake.sample_likelihood_mode = SAMPLE_LIKELIHOOD_SOURCE
    fake.models_by_effective_z = {2.0: object()}
    fake.critical_arc_lm_trust_radius_arcsec = cluster_solver.DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC
    fake.critical_arc_lm_damping_relative = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE
    fake.critical_arc_lm_damping_absolute = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE
    fake.critical_arc_base_prob = cluster_solver.DEFAULT_CRITICAL_ARC_BASE_PROB
    fake.critical_arc_max_prob = cluster_solver.DEFAULT_CRITICAL_ARC_MAX_PROB
    fake.critical_arc_singular_threshold = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD
    fake.critical_arc_singular_softness = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS
    fake._build_packed_lens_state = lambda _params, _z_source: {}
    fake._source_sigma_int_numpy = lambda _params: 0.0
    fake._source_position_for_family_numpy = lambda _params, _family_id: None
    fake._exact_source_ray_shooting = (
        lambda _family, _packed_state: (np.asarray([-0.2], dtype=float), np.asarray([0.0], dtype=float))
    )
    fake._ray_shooting_for_components = (
        lambda _z_source, _x, _y, _packed_state: (jnp.asarray([-0.2], dtype=jnp.float64), jnp.asarray([0.0], dtype=jnp.float64))
    )

    def fail_jacobian(_z_source, _x, _y, _packed_state):
        raise RuntimeError("jacobian unavailable")

    def fail_exact_solver(_family, _packed_state, _source_x, _source_y):
        raise RuntimeError("solver failed")

    fake._lensing_jacobian_for_components = fail_jacobian
    fake._solve_exact_images_lenstronomy = fail_exact_solver

    details = cluster_solver.ClusterJAXEvaluator._exact_family_prediction_details(
        fake,
        np.asarray([], dtype=float),
        family,
    )

    assert details["failed"] is True
    assert details["arc_recovery_status"].tolist() == ["not_recovered"]
    assert details["arc_supported_mask"].tolist() == [False]
    assert np.isnan(details["arc_s_min"][0])
    assert np.isnan(details["arc_noncritical_direction_residual_arcsec"][0])
    assert image_diagnostics.exact_details_hard_failed(details)


def test_critical_arc_source_loglike_branch_uses_anchor_lm_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traced_bin = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0], dtype=jnp.int32),
        x_obs=jnp.asarray([0.0], dtype=jnp.float64),
        y_obs=jnp.asarray([0.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True], dtype=bool),
    )
    captured: dict[str, np.ndarray] = {}

    def fake_bin_loglike(**kwargs):
        captured["residual_x"] = np.asarray(kwargs["residual_x"], dtype=float)
        captured["residual_y"] = np.asarray(kwargs["residual_y"], dtype=float)
        captured["jac_a00"] = np.asarray(kwargs["jac_a00"], dtype=float)
        captured["jac_a11"] = np.asarray(kwargs["jac_a11"], dtype=float)
        return jnp.asarray(321.0, dtype=jnp.float64)

    monkeypatch.setattr(cluster_solver, "_critical_arc_mixture_image_plane_bin_loglike", fake_bin_loglike)

    class FakeEvaluator:
        source_position_conditional = False
        source_position_param_indices_by_family = {"1": (0, 1)}
        surrogate_enabled = False
        surrogate_cache_by_z = {}
        sample_likelihood_mode = SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE
        critical_arc_critical_direction_sigma_arcsec = 5.0
        critical_arc_base_prob = 0.10
        critical_arc_max_prob = 0.80
        critical_arc_singular_threshold = 0.20
        critical_arc_singular_softness = 0.05
        critical_arc_lm_damping_relative = 1.0e-3
        critical_arc_lm_damping_absolute = 1.0e-6
        critical_arc_lm_trust_radius_arcsec = 20.0
        image_plane_scatter_floor_arcsec = 1.0e-3
        source_plane_covariance_floor = 1.0e-6
        source_plane_outlier_sigma_arcsec = 10.0
        image_presence_penalty_weight = 0.0
        image_presence_match_radius_arcsec = 0.30
        image_presence_temperature_arcsec = 0.10
        image_presence_count_softness = 0.05
        image_presence_count_margin = 0.05
        likelihood_stabilizer_residual_loss = "gaussian"
        likelihood_stabilizer_student_t_nu = 4.0
        traced_bin_data = [traced_bin]

        def _physical_parameter_vector(self, params):
            return params

        def _source_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.0, dtype=jnp.float64)

        def _image_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.05, dtype=jnp.float64)

        def _build_packed_lens_state_with_validity_from_physical(self, _physical_params, _z_source, stop_gradient=True):
            return {}, {"is_valid": jnp.asarray(True)}

        def _ray_shooting_for_components(self, _z_source, x, y, _packed_state):
            return x + 0.4, y - 0.1

        def _lensing_jacobian_for_components(self, _z_source, x, y, _packed_state):
            return (
                jnp.full_like(x, 2.0),
                jnp.zeros_like(x),
                jnp.zeros_like(y),
                jnp.ones_like(y),
            )

        def _scaling_scatter_extra_variance_from_physical(self, _physical_params, bin_data, _beta_x, _beta_y):
            return jnp.zeros_like(bin_data.x_obs), jnp.zeros_like(bin_data.y_obs)

        def _maybe_record_invalid_state(self, _validity):
            return None

        def _cab_morphology_loglike_for_bin(self, *_args, **_kwargs):
            return jnp.asarray(0.0, dtype=jnp.float64)

    fake = FakeEvaluator()
    fake._source_position_vectors_for_bin = cluster_solver.ClusterJAXEvaluator._source_position_vectors_for_bin.__get__(
        fake,
        FakeEvaluator,
    )
    fake._explicit_source_position_vectors_for_bin = (
        cluster_solver.ClusterJAXEvaluator._explicit_source_position_vectors_for_bin.__get__(fake, FakeEvaluator)
    )

    params = jnp.asarray([0.1, -0.2], dtype=jnp.float64)
    loglike = cluster_solver.ClusterJAXEvaluator._source_loglike_impl(fake, params)
    expected_dx, expected_dy, expected_finite = _critical_arc_lm_step_from_jacobian(
        f_x=jnp.asarray([0.3], dtype=jnp.float64),
        f_y=jnp.asarray([0.1], dtype=jnp.float64),
        jac_a00=jnp.asarray([2.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
        trust_radius_arcsec=20.0,
        lm_damping_relative=1.0e-3,
        lm_damping_absolute=1.0e-6,
    )

    assert float(loglike) == pytest.approx(321.0)
    assert bool(np.asarray(expected_finite)[0])
    np.testing.assert_allclose(captured["residual_x"], np.asarray(expected_dx), rtol=1.0e-12)
    np.testing.assert_allclose(captured["residual_y"], np.asarray(expected_dy), rtol=1.0e-12)
    np.testing.assert_allclose(captured["jac_a00"], np.asarray([2.0]))
    np.testing.assert_allclose(captured["jac_a11"], np.asarray([1.0]))


def test_critical_arc_source_loglike_uses_full_exact_mixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traced_bin = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        x_obs=jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float64),
        y_obs=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.2, 0.2, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999, 0.999, 0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True, True], dtype=bool),
    )
    captured: dict[str, Any] = {}

    def fake_critical_bin_loglike(**kwargs):
        captured["critical_n"] = int(np.asarray(kwargs["residual_x"]).size)
        captured["has_soft_anisotropic"] = "soft_anisotropic" in kwargs
        captured["has_cheap_singular_proxy"] = "use_cheap_singular_proxy" in kwargs
        captured["critical_presence_weight"] = float(kwargs["image_presence_penalty_weight"])
        return jnp.asarray(100.0, dtype=jnp.float64)

    def fake_linearized_bin_loglike(**_kwargs):
        raise AssertionError("exact critical-arc likelihood must not split far rows into linearized likelihood")

    monkeypatch.setattr(cluster_solver, "_critical_arc_mixture_image_plane_bin_loglike", fake_critical_bin_loglike)
    monkeypatch.setattr(cluster_solver, "_linearized_image_plane_bin_loglike", fake_linearized_bin_loglike)

    class FakeEvaluator:
        source_position_conditional = False
        source_position_param_indices_by_family = {"1": (0, 1)}
        surrogate_enabled = True
        surrogate_cache_by_z = {2.0: SimpleNamespace()}
        sample_likelihood_mode = SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE
        critical_arc_critical_direction_sigma_arcsec = 5.0
        critical_arc_base_prob = 0.10
        critical_arc_max_prob = 0.80
        critical_arc_singular_threshold = 0.20
        critical_arc_singular_softness = 0.05
        critical_arc_lm_damping_relative = 1.0e-3
        critical_arc_lm_damping_absolute = 1.0e-6
        critical_arc_lm_trust_radius_arcsec = 20.0
        image_plane_scatter_floor_arcsec = 1.0e-3
        source_plane_covariance_floor = 1.0e-6
        source_plane_outlier_sigma_arcsec = 10.0
        image_presence_penalty_weight = 0.0
        image_presence_match_radius_arcsec = 0.30
        image_presence_temperature_arcsec = 0.10
        image_presence_count_softness = 0.05
        image_presence_count_margin = 0.05
        likelihood_stabilizer_residual_loss = "gaussian"
        likelihood_stabilizer_student_t_nu = 4.0
        traced_bin_data = [traced_bin]

        def _physical_parameter_vector(self, params):
            return params

        def _source_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.0, dtype=jnp.float64)

        def _image_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.05, dtype=jnp.float64)

        def _build_packed_lens_state_with_validity_from_physical(self, _physical_params, _z_source, stop_gradient=True):
            return {}, {"is_valid": jnp.asarray(True)}

        def _ray_shooting_for_components(self, _z_source, x, y, _packed_state):
            return x + 0.4, y - 0.1

        def _surrogate_beta(self, _params, _physical_params, bin_data):
            return bin_data.x_obs + 0.4, bin_data.y_obs - 0.1, jnp.asarray(False), {}

        def _surrogate_jacobian_entries(self, _params, bin_data, _packed_state, _invalid):
            return (
                jnp.asarray([0.05, 1.0, 0.05], dtype=jnp.float64),
                jnp.zeros_like(bin_data.x_obs),
                jnp.zeros_like(bin_data.y_obs),
                jnp.ones_like(bin_data.y_obs),
            )

        def _scaling_scatter_extra_variance_from_physical(self, _physical_params, bin_data, _beta_x, _beta_y):
            return jnp.zeros_like(bin_data.x_obs), jnp.zeros_like(bin_data.y_obs)

        def _maybe_record_invalid_state(self, _validity):
            return None

        def _cab_morphology_loglike_for_bin(self, *_args, **_kwargs):
            return jnp.asarray(0.0, dtype=jnp.float64)

    fake = FakeEvaluator()
    fake._source_position_vectors_for_bin = cluster_solver.ClusterJAXEvaluator._source_position_vectors_for_bin.__get__(
        fake,
        FakeEvaluator,
    )
    fake._explicit_source_position_vectors_for_bin = (
        cluster_solver.ClusterJAXEvaluator._explicit_source_position_vectors_for_bin.__get__(fake, FakeEvaluator)
    )

    loglike = cluster_solver.ClusterJAXEvaluator._source_loglike_impl(fake, jnp.asarray([0.1, -0.2], dtype=jnp.float64))

    assert float(loglike) == pytest.approx(100.0)
    assert captured["critical_n"] == 3
    assert captured["has_soft_anisotropic"] is False
    assert captured["has_cheap_singular_proxy"] is False
    assert captured["critical_presence_weight"] == pytest.approx(0.0)


def test_cluster_solver_accepts_fit_cosmology_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-cosmology-flat-wcdm",
        ],
    )

    args = _parse_args()

    assert args.fit_cosmology_flat_wcdm is True


def test_cluster_solver_rejects_removed_fit_cosmology_all_stages_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-cosmology-all-stages",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


def test_cluster_solver_accepts_cosmology_init_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-cosmology-flat-wcdm",
            "--cosmology-init-om0",
            "0.25",
            "--cosmology-init-w0",
            "-0.8",
        ],
    )

    args = _parse_args()
    _normalize_stage_fit_controls(args)

    assert args.cosmology_init_om0 == 0.25
    assert args.cosmology_init_w0 == -0.8


def test_cosmology_init_flags_seed_state_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-mode",
            "joint",
            "--fit-cosmology-flat-wcdm",
            "--cosmology-init-om0",
            "0.25",
            "--cosmology-init-w0",
            "-0.8",
        ],
    )

    args = _parse_args()
    state = cluster_solver._build_state_from_inputs(args)

    assert state.svi_init_values is not None
    assert state.svi_init_values[cluster_solver.COSMOLOGY_OM0_SAMPLE_NAME] == 0.25
    assert state.svi_init_values[cluster_solver.COSMOLOGY_W0_SAMPLE_NAME] == -0.8


def test_cosmology_init_flags_override_warm_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-mode",
            "joint",
            "--fit-cosmology-flat-wcdm",
            "--cosmology-init-om0",
            "0.25",
            "--cosmology-init-w0",
            "-0.8",
        ],
    )
    warm_values = {
        cluster_solver.COSMOLOGY_OM0_SAMPLE_NAME: 0.5,
        cluster_solver.COSMOLOGY_W0_SAMPLE_NAME: -1.8,
    }

    args = _parse_args()
    state = cluster_solver._build_state_from_inputs(args, svi_init_physical_values=warm_values)

    assert state.svi_init_values is not None
    assert state.svi_init_values[cluster_solver.COSMOLOGY_OM0_SAMPLE_NAME] == 0.25
    assert state.svi_init_values[cluster_solver.COSMOLOGY_W0_SAMPLE_NAME] == -0.8


@pytest.mark.parametrize(
    ("flag", "value", "match"),
    [
        ("--cosmology-init-om0", "nan", "must be finite"),
        ("--cosmology-init-om0", "0.01", "within"),
        ("--cosmology-init-om0", "0.9", "within"),
        ("--cosmology-init-w0", "-2.5", "within"),
        ("--cosmology-init-w0", "-0.1", "within"),
    ],
)
def test_cosmology_init_invalid_values_fail(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
    match: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-cosmology-flat-wcdm",
            flag,
            value,
        ],
    )

    args = _parse_args()

    with pytest.raises(SystemExit, match=match):
        _normalize_stage_fit_controls(args)


def test_cosmology_init_without_fit_cosmology_is_accepted_and_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--cosmology-init-om0",
            "0.25",
            "--cosmology-init-w0",
            "-0.8",
        ],
    )

    args = _parse_args()
    _normalize_stage_fit_controls(args)
    state = cluster_solver._build_state_from_inputs(args)

    assert args.cosmology_init_om0 == 0.25
    assert args.cosmology_init_w0 == -0.8
    assert all(spec.component_family != "cosmology" for spec in state.parameter_specs)
    assert state.svi_init_values is None or cluster_solver.COSMOLOGY_OM0_SAMPLE_NAME not in state.svi_init_values
    assert state.svi_init_values is None or cluster_solver.COSMOLOGY_W0_SAMPLE_NAME not in state.svi_init_values


def test_cluster_solver_defaults_to_prior_whitened_source_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "data/clustersim/input.par"])

    args = _parse_args()

    assert args.source_position_parameterization == "prior-whitened"


def test_fixed_image_sigma_int_removes_sampled_image_scatter_and_evaluator_returns_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["cluster_solver", "--par-path", "data/clustersim/input.par", "--fit-mode", "joint"],
    )
    sampled_args = _parse_args()
    sampled_args.sample_likelihood_mode = SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
    source_position_priors = {str(index): (0.0, 0.0) for index in range(1, 128)}
    sampled_state = cluster_solver._build_state_from_inputs(
        sampled_args,
        source_position_prior_values=source_position_priors,
    )

    assert any(spec.sample_name == "image_sigma_int" for spec in sampled_state.parameter_specs)

    fixed_args = argparse.Namespace(**vars(sampled_args))
    fixed_args.fix_image_sigma_int_arcsec = 0.35
    fixed_state = cluster_solver._build_state_from_inputs(
        fixed_args,
        source_position_prior_values=source_position_priors,
    )
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=fixed_state,
        match_tolerance_arcsec=0.1,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        fixed_image_sigma_int_arcsec=0.35,
    )

    assert all(spec.sample_name != "image_sigma_int" for spec in fixed_state.parameter_specs)
    assert evaluator.image_sigma_int_sampled is False
    assert evaluator._image_sigma_int_numpy(np.zeros(len(fixed_state.parameter_specs), dtype=float)) == pytest.approx(0.35)


def test_cluster_solver_quick_diagnostics_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "data/clustersim/input.par"])
    args = _parse_args()
    assert args.quick_diagnostics is False
    assert args.exact_image_diagnostics_stage3 is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--quick-diagnostics",
        ],
    )
    args = _parse_args()
    assert args.quick_diagnostics is True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--exact-image-diagnostics-stage3",
        ],
    )
    args = _parse_args()
    assert args.exact_image_diagnostics_stage3 is True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--quick-diagnostics",
            "--exact-image-diagnostics-stage3",
        ],
    )
    args = _parse_args()
    with pytest.raises(SystemExit, match="exact-image-diagnostics-stage3"):
        cluster_solver._normalize_stage_fit_controls(args)


def test_critical_det_diagnostic_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "data/clustersim/input.par"])
    args = _parse_args()
    assert args.critical_det_diagnostic_threshold == pytest.approx(
        cluster_solver.DEFAULT_CRITICAL_DET_DIAGNOSTIC_THRESHOLD
    )
    assert args.skip_critical_det_diagnostic is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--critical-det-diagnostic-threshold",
            "0.0025",
            "--skip-critical-det-diagnostic",
        ],
    )
    args = _parse_args()
    assert args.critical_det_diagnostic_threshold == pytest.approx(2.5e-3)
    assert args.skip_critical_det_diagnostic is True


def test_cluster_solver_fit_quality_draws_default_and_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "data/clustersim/input.par"])
    args = _parse_args()
    assert args.fit_quality_draws == 0

    monkeypatch.setattr(
        sys,
        "argv",
        ["cluster_solver", "--par-path", "data/clustersim/input.par", "--fit-quality-draws", "0"],
    )
    args = _parse_args()
    assert args.fit_quality_draws == 0
    cluster_solver._normalize_stage_fit_controls(args)

    args.fit_quality_draws = -1
    with pytest.raises(SystemExit):
        cluster_solver._normalize_stage_fit_controls(args)


def test_validation_parser_accepts_fit_cosmology_flag() -> None:
    args = validation._build_parser().parse_args(["--fit-cosmology-flat-wcdm"])

    assert args.fit_cosmology_flat_wcdm is True


def test_validation_parser_rejects_removed_fit_cosmology_all_stages_flag() -> None:
    parser = validation._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--fit-cosmology-all-stages"])


def test_validation_parser_defaults_to_prior_whitened_source_positions() -> None:
    args = validation._build_parser().parse_args([])

    assert args.source_position_parameterization == "prior-whitened"
    assert args.linearized_beta_prior_sigma_arcsec == pytest.approx(2.0)


def test_flat_wcdm_jax_factor_is_differentiable() -> None:
    z_lens = 0.3734
    z_source = 3.0
    h0 = 70.0

    def factor(om0: jnp.ndarray) -> jnp.ndarray:
        return dpie_sigma0_factor(z_lens, z_source, h0, om0, -1.0)

    grad_value = jax.grad(factor)(jnp.asarray(0.3, dtype=jnp.float64))

    assert np.isfinite(float(grad_value))


def test_cluster_solver_original_profile_maps_dpie_to_dpie_nie() -> None:
    _specs, _assignments, lens_model_list = _build_parameter_specs(
        [
            {"id": "halo", "profil": 81, "priors": {}},
            {"id": "shear", "profil": 14, "priors": {}},
        ],
    )

    assert lens_model_list == ["DPIE_NIE", "SHEAR"]


def test_decode_mode9_prior_accepts_finite_and_lower_only_truncation() -> None:
    finite = cluster_solver._decode_parameter_prior([9, 900.0, 100.0, 300.0, 1500.0], "halo.v_disp")
    lower_only = cluster_solver._decode_parameter_prior([9, 900.0, 100.0, 300.0], "halo.v_disp")

    assert finite == {
        "prior_kind": "truncated_normal",
        "lower": 300.0,
        "upper": 1500.0,
        "step": 100.0,
        "mean": 900.0,
        "std": 100.0,
    }
    assert lower_only["prior_kind"] == "truncated_normal"
    assert lower_only["lower"] == pytest.approx(300.0)
    assert np.isinf(lower_only["upper"])
    assert lower_only["mean"] == pytest.approx(900.0)
    assert lower_only["std"] == pytest.approx(100.0)

    with pytest.raises(ValueError, match="std must be positive"):
        cluster_solver._decode_parameter_prior([9, 900.0, 0.0, 300.0], "halo.v_disp")
    with pytest.raises(ValueError, match="lower bound must be less"):
        cluster_solver._decode_parameter_prior([9, 900.0, 100.0, 300.0, 300.0], "halo.v_disp")


def test_dpie_v_disp_normal_prior_uses_truncated_identity_transform() -> None:
    specs, _assignments, _lens_model_list = _build_parameter_specs(
        [
            {
                "id": "halo",
                "profil": cluster_solver.DP_IE_PROFILE,
                "priors": {"v_disp": [3, 900.0, 600.0, 0.1]},
            },
        ],
    )

    spec = specs[0]
    latent = np.asarray([[cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S], [1.0], [float(spec.mean)]], dtype=float)
    physical = cluster_solver._convert_sample_matrix_to_physical(
        latent,
        specs,
    )
    prior = cluster_solver._distribution_for_spec(spec)

    assert spec.name == "halo.v_disp"
    assert spec.prior_kind == "truncated_normal"
    assert spec.transform_kind == "identity"
    assert spec.lower == pytest.approx(cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S)
    assert np.isinf(spec.upper)
    assert spec.mean == pytest.approx(900.0)
    assert spec.std == pytest.approx(600.0)
    np.testing.assert_allclose(physical, latent)
    assert np.isfinite(float(prior.log_prob(jnp.asarray(1.0))))
    with pytest.warns(UserWarning, match="Out-of-support values"):
        below_floor_log_prob = prior.log_prob(jnp.asarray(0.0))
    assert float(below_floor_log_prob) == -np.inf


def test_dpie_v_disp_mode9_prior_uses_decoded_truncation_bounds() -> None:
    specs, _assignments, _lens_model_list = _build_parameter_specs(
        [
            {
                "id": "halo",
                "profil": cluster_solver.DP_IE_PROFILE,
                "priors": {"v_disp": [9, 900.0, 100.0, 300.0, 1500.0]},
            },
        ],
    )

    spec = specs[0]

    assert spec.name == "halo.v_disp"
    assert spec.prior_kind == "truncated_normal"
    assert spec.transform_kind == "identity"
    assert spec.lower == pytest.approx(300.0)
    assert spec.upper == pytest.approx(1500.0)
    assert spec.mean == pytest.approx(900.0)
    assert spec.std == pytest.approx(100.0)
    assert spec.physical_lower == pytest.approx(300.0)
    assert spec.physical_upper == pytest.approx(1500.0)


def test_potfile_mode9_scaling_priors_transform_bounds_to_latent_log_space() -> None:
    specs, param_indices, _lens_model_list = cluster_solver._build_scaling_parameter_specs(
        [
            {
                "id": "members",
                "type": cluster_solver.DP_IE_PROFILE,
                "catalog_df": pd.DataFrame({"id": ["member"]}),
                "core_arcsec": 0.1,
                "sigma": [9, 245.0, 55.0, 190.0, 300.0],
                "cut": [9, 5.25, 4.75, 0.5, 10.0],
            }
        ],
        kpc_per_arcsec=10.0,
    )
    sigma_spec = specs[param_indices[0]["sigma"]]
    cut_spec = specs[param_indices[0]["cutkpc"]]
    expected_sigma_mean, expected_sigma_std = cluster_solver._positive_lognormal_parameters(245.0, 55.0)
    expected_cut_mean, expected_cut_std = cluster_solver._positive_lognormal_parameters(51.5, 47.5)

    assert sigma_spec.prior_kind == "truncated_normal"
    assert sigma_spec.transform_kind == "log_positive"
    assert sigma_spec.lower == pytest.approx(np.log(190.0))
    assert sigma_spec.upper == pytest.approx(np.log(300.0))
    assert sigma_spec.mean == pytest.approx(expected_sigma_mean)
    assert sigma_spec.std == pytest.approx(expected_sigma_std)
    assert sigma_spec.physical_lower == pytest.approx(190.0)
    assert sigma_spec.physical_upper == pytest.approx(300.0)
    assert sigma_spec.physical_mean == pytest.approx(245.0)
    assert sigma_spec.physical_std == pytest.approx(55.0)

    assert cut_spec.prior_kind == "truncated_normal"
    assert cut_spec.transform_kind == "log_offset_positive"
    assert cut_spec.transform_offset == pytest.approx(1.0)
    assert cut_spec.lower == pytest.approx(np.log(4.0))
    assert cut_spec.upper == pytest.approx(np.log(99.0))
    assert cut_spec.mean == pytest.approx(expected_cut_mean)
    assert cut_spec.std == pytest.approx(expected_cut_std)
    assert cut_spec.physical_lower == pytest.approx(5.0)
    assert cut_spec.physical_upper == pytest.approx(100.0)
    assert cut_spec.physical_mean == pytest.approx(52.5)
    assert cut_spec.physical_std == pytest.approx(47.5)


def test_potfile_mass_size_reparam_is_opt_in_and_preserves_public_fields() -> None:
    potfiles = [
        {
            "id": "members",
            "type": cluster_solver.DP_IE_PROFILE,
            "catalog_df": pd.DataFrame({"id": ["member"]}),
            "core_arcsec": 0.1,
            "sigma": [9, 245.0, 55.0, 190.0, 300.0],
            "cut": [9, 5.25, 4.75, 0.5, 10.0],
        }
    ]

    specs_off, param_indices_off, _ = cluster_solver._build_scaling_parameter_specs(
        potfiles,
        kpc_per_arcsec=10.0,
    )
    specs_on, param_indices_on, _ = cluster_solver._build_scaling_parameter_specs(
        potfiles,
        kpc_per_arcsec=10.0,
        potfile_mass_size_reparam=True,
    )

    sigma_off = specs_off[param_indices_off[0]["sigma"]]
    cut_off = specs_off[param_indices_off[0]["cutkpc"]]
    sigma_on = specs_on[param_indices_on[0]["sigma"]]
    cut_on = specs_on[param_indices_on[0]["cutkpc"]]

    assert sigma_off.sample_site_name is None
    assert cut_off.sample_site_name is None
    assert sigma_on.name == "members.sigma"
    assert cut_on.name == "members.cutkpc"
    assert sigma_on.sample_name == "members_sigma"
    assert cut_on.sample_name == "members_cutkpc"
    assert sigma_on.sample_site_name == cut_on.sample_site_name == "members_mass_size"
    assert sigma_on.sample_site_index == 0
    assert cut_on.sample_site_index == 1
    assert sigma_on.coupled_role == cluster_solver.COUPLED_ROLE_MASS_NORM
    assert cut_on.coupled_role == cluster_solver.COUPLED_ROLE_SIZE


def test_potfile_mass_size_reparam_roundtrip_and_prior_jacobian() -> None:
    specs, param_indices, _ = cluster_solver._build_scaling_parameter_specs(
        [
            {
                "id": "members",
                "type": cluster_solver.DP_IE_PROFILE,
                "catalog_df": pd.DataFrame({"id": ["member"]}),
                "core_arcsec": 0.1,
                "sigma": [9, 245.0, 55.0, 190.0, 300.0],
                "cut": [9, 5.25, 4.75, 0.5, 10.0],
            }
        ],
        kpc_per_arcsec=10.0,
        potfile_mass_size_reparam=True,
    )
    sigma_spec = specs[param_indices[0]["sigma"]]
    cut_spec = specs[param_indices[0]["cutkpc"]]
    theta = np.asarray([0.0, 0.0], dtype=float)

    physical = cluster_solver._convert_theta_to_physical(theta, specs)
    roundtrip = cluster_solver._convert_theta_to_latent(physical, specs)

    mass_raw = sigma_spec.coupled_mass_center + sigma_spec.coupled_mass_scale * theta[0]
    size_raw = sigma_spec.coupled_size_center + sigma_spec.coupled_size_scale * theta[1]
    expected_sigma = math.exp(0.5 * (mass_raw - size_raw))
    expected_cut = cut_spec.transform_offset + math.exp(size_raw)
    old_prior_log_prob = (
        cluster_solver._distribution_for_spec(sigma_spec).log_prob(jnp.asarray(math.log(expected_sigma)))
        + cluster_solver._distribution_for_spec(cut_spec).log_prob(jnp.asarray(math.log(expected_cut - cut_spec.transform_offset)))
    )
    expected_jacobian = math.log(0.5 * sigma_spec.coupled_mass_scale * sigma_spec.coupled_size_scale)

    np.testing.assert_allclose(physical, [expected_sigma, expected_cut])
    np.testing.assert_allclose(roundtrip, theta)
    assert float(cluster_solver._prior_log_prob(specs, jnp.asarray(theta, dtype=jnp.float64))) == pytest.approx(
        float(old_prior_log_prob) + expected_jacobian
    )


def test_potfile_mass_size_reparam_vector_site_init_and_extraction() -> None:
    specs, _param_indices, _ = cluster_solver._build_scaling_parameter_specs(
        [
            {
                "id": "members",
                "type": cluster_solver.DP_IE_PROFILE,
                "catalog_df": pd.DataFrame({"id": ["member"]}),
                "core_arcsec": 0.1,
                "sigma": [1, 190.0, 300.0],
                "cut": [1, 0.5, 10.0],
            }
        ],
        kpc_per_arcsec=10.0,
        potfile_mass_size_reparam=True,
    )
    site_name = specs[0].sample_site_name
    samples_dict = {site_name: np.asarray([[[0.1, -0.2], [0.3, -0.4]]], dtype=float)}
    seed = ChainSeed(values=np.asarray([0.1, -0.2], dtype=float), source_label="seed")

    trace = cluster_solver.numpyro.handlers.trace(
        cluster_solver.numpyro.handlers.seed(cluster_solver._sample_site_model(specs), jax.random.PRNGKey(11))
    ).get_trace()
    init_params = cluster_solver._seed_values_to_init_params(specs, [seed])
    extracted = cluster_solver._extract_grouped_samples(samples_dict, specs, thin=1)

    assert np.asarray(trace[site_name]["value"]).shape == (2,)
    assert list(init_params) == [site_name]
    assert np.asarray(init_params[site_name]).shape == (2,)
    np.testing.assert_allclose(extracted, np.asarray([[[0.1, -0.2], [0.3, -0.4]]], dtype=float))


def test_stage2_large_priors_keep_truncated_vdisp_identity_summary_values() -> None:
    large_specs, _assignments, _lens_model_list = _build_parameter_specs(
        [
            {
                "id": "halo",
                "profil": cluster_solver.DP_IE_PROFILE,
                "priors": {"v_disp": [3, 900.0, 600.0, 0.1]},
            },
        ],
    )
    summary = Stage1PriorSummary(
        map_values={"halo_v_disp": 920.0},
        means={"halo_v_disp": 900.0},
        stds={"halo_v_disp": 90.0},
    )

    stage2_specs = cluster_solver._build_stage2_large_parameter_specs(large_specs, summary)
    stage2_spec = stage2_specs[0]
    physical_center = cluster_solver._convert_theta_to_physical(np.asarray([stage2_spec.mean]), stage2_specs)[0]

    assert stage2_spec.prior_kind == "truncated_normal"
    assert stage2_spec.transform_kind == "identity"
    assert stage2_spec.lower == pytest.approx(cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S)
    assert stage2_spec.mean == pytest.approx(900.0)
    assert stage2_spec.std == pytest.approx(90.0)
    assert physical_center == pytest.approx(900.0)


def test_truncated_vdisp_default_theta_and_init_are_clipped_to_support() -> None:
    spec = ParameterSpec(
        "halo.v_disp",
        "halo_v_disp",
        "halo",
        cluster_solver.DP_IE_PROFILE,
        "v_disp",
        "truncated_normal",
        cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S,
        np.inf,
        0.1,
        mean=-10.0,
        std=1.0,
        transform_kind="identity",
        physical_lower=cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S,
        physical_mean=-10.0,
        physical_std=1.0,
    )

    default_theta = cluster_solver._default_theta([spec])
    clipped_init = cluster_solver._initial_latent_value_from_physical(-5.0, spec)

    assert default_theta[0] > cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S
    assert clipped_init > cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S


def test_packed_lens_validity_allows_negative_finite_vdisp_and_rejects_nonfinite() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    details = {
        "is_dpie": jnp.asarray([True, True]),
        "is_shear": jnp.asarray([False, False]),
        "is_scaling": jnp.asarray([False, False]),
        "sigma0": jnp.asarray([1.0, 1.0]),
        "ra_raw": jnp.asarray([1.0, 1.0]),
        "rs_raw": jnp.asarray([10.0, 10.0]),
        "v_disp": jnp.asarray([500.0, -10.0]),
        "vdslope": jnp.asarray([1.0, 1.0]),
        "slope": jnp.asarray([1.0, 1.0]),
        "x_center": jnp.asarray([0.0, 0.0]),
        "y_center": jnp.asarray([0.0, 0.0]),
        "gamma1": jnp.asarray([0.0, 0.0]),
        "gamma2": jnp.asarray([0.0, 0.0]),
        "e1": jnp.asarray([0.0, 0.0]),
        "e2": jnp.asarray([0.0, 0.0]),
        "factor_array": jnp.asarray(1.0),
    }

    validity = cluster_solver.ClusterJAXEvaluator._packed_lens_validity(evaluator, details)
    reason_index = cluster_solver.INVALID_STATE_REASON_NAMES.index("nonpositive_vdisp")

    assert bool(validity["is_valid"]) is True
    assert bool(np.asarray(validity["reason_flags"])[reason_index]) is False

    details["v_disp"] = jnp.asarray([500.0, jnp.nan])
    validity = cluster_solver.ClusterJAXEvaluator._packed_lens_validity(evaluator, details)

    assert bool(validity["is_valid"]) is False
    assert bool(np.asarray(validity["reason_flags"])[reason_index]) is True


def test_nuts_quality_diagnostics_flag_stuck_tree_depth_and_rhat() -> None:
    grouped = np.stack(
        [np.full((10, 1), float(chain_index), dtype=float) for chain_index in range(4)],
        axis=0,
    )
    posterior = PosteriorResults(
        samples=grouped.reshape(-1, 1),
        log_prob=np.zeros(40, dtype=float),
        accept_prob=np.ones(40, dtype=float),
        diverging=np.zeros(40, dtype=bool),
        num_steps=np.full(40, 255.0, dtype=float),
        warmup_steps=10,
        sample_steps=10,
        num_chains=4,
        grouped_samples=grouped,
        sampler="numpyro_nuts",
    )
    spec = ParameterSpec("halo.v_disp", "halo_v_disp", "halo", 81, "v_disp", "normal", -np.inf, np.inf, 0.1)

    metrics, warnings = cluster_solver._nuts_quality_diagnostics(
        argparse.Namespace(max_tree_depth=8),
        posterior,
        [spec],
    )

    assert metrics["max_tree_depth_saturation_fraction"] == pytest.approx(1.0)
    assert any("max-tree-depth saturation" in warning for warning in warnings)
    assert any("extreme Rhat" in warning for warning in warnings)


def test_sampler_debug_extra_fields_are_flag_gated() -> None:
    default_fields = cluster_solver._sampler_debug_extra_field_names(
        argparse.Namespace(debug_sampler_diagnostics=False)
    )
    debug_fields = cluster_solver._sampler_debug_extra_field_names(
        argparse.Namespace(debug_sampler_diagnostics=True)
    )

    assert default_fields == ("accept_prob", "diverging", "num_steps", "potential_energy")
    assert "energy" in debug_fields
    assert "adapt_state.step_size" in debug_fields
    assert "adapt_state.inverse_mass_matrix" in debug_fields


def test_sanitize_grouped_posterior_filters_nested_debug_extra_fields() -> None:
    spec = ParameterSpec(
        "theta",
        "theta",
        "theta",
        0,
        "theta",
        "normal",
        -np.inf,
        np.inf,
        0.1,
        mean=0.0,
        std=1.0,
    )
    samples = {"theta": np.asarray([[0.0, 0.1], [np.nan, 0.2]], dtype=float)}
    extra = {
        "potential_energy": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        "adapt_state.inverse_mass_matrix": {("theta",): np.ones((2, 2, 1, 1), dtype=float)},
    }

    filtered_samples, filtered_extra, diagnostics = cluster_solver._sanitize_grouped_posterior(
        samples,
        extra,
        [spec],
    )

    assert diagnostics["retained_chain_indices"] == [0]
    assert filtered_samples["theta"].shape == (1, 2)
    assert filtered_extra["potential_energy"].shape == (1, 2)
    assert filtered_extra["adapt_state.inverse_mass_matrix"][("theta",)].shape == (1, 2, 1, 1)


def test_nuts_integrator_debug_table_flags_constant_stuck_chain(tmp_path: Path) -> None:
    grouped = np.stack(
        [
            np.linspace(0.0, 1.0, 5, dtype=float).reshape(5, 1),
            np.full((5, 1), 2.0, dtype=float),
        ],
        axis=0,
    )
    posterior = PosteriorResults(
        samples=grouped.reshape(-1, 1),
        log_prob=np.zeros(10, dtype=float),
        accept_prob=np.ones(10, dtype=float),
        diverging=np.zeros(10, dtype=bool),
        num_steps=np.full(10, 255.0, dtype=float),
        warmup_steps=5,
        sample_steps=5,
        num_chains=2,
        init_diagnostics={"chain_seed_labels": ["a", "b"]},
        grouped_samples=grouped,
        grouped_log_prob=np.zeros((2, 5), dtype=float),
        sampler="numpyro_nuts",
    )
    extra = {
        "accept_prob": np.ones((2, 5), dtype=float),
        "num_steps": np.full((2, 5), 255.0, dtype=float),
        "potential_energy": np.zeros((2, 5), dtype=float),
        "energy": np.asarray([[1.0, 1.1, 0.9, 1.0, 1.2], [2.0, 2.0, 2.0, 2.0, 2.0]], dtype=float),
        "adapt_state.step_size": np.asarray([[0.01] * 5, [1.0e-8] * 5], dtype=float),
        "adapt_state.inverse_mass_matrix": {("theta",): np.ones((2, 5, 1, 1), dtype=float)},
    }

    df = cluster_solver._write_nuts_integrator_debug_table(
        tmp_path / "nuts_integrator_diagnostics.csv",
        argparse.Namespace(max_tree_depth=8),
        posterior,
        extra,
        [0, 1],
    )

    assert (tmp_path / "nuts_integrator_diagnostics.csv").exists()
    assert df.loc[1, "stuck_parameter_count_range_lt_1e_9"] == 1
    assert df.loc[1, "parameter_range_min"] == pytest.approx(0.0)
    assert df.loc[1, "max_tree_depth_saturation_fraction"] == pytest.approx(1.0)
    assert df.loc[1, "step_size_last"] == pytest.approx(1.0e-8)


def test_critical_arc_debug_terms_sum_to_bin_loglike_without_presence_penalty() -> None:
    bin_data = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0, 0], dtype=jnp.int32),
        x_obs=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        y_obs=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.25, 0.25], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True], dtype=bool),
    )

    class FakeCriticalArcEvaluator:
        sample_likelihood_mode = SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE
        fit_cosmology_flat_wcdm = False
        surrogate_enabled = False
        surrogate_cache_by_z: dict[float, object] = {}
        traced_bin_data = [bin_data]
        critical_arc_lm_trust_radius_arcsec = 20.0
        critical_arc_lm_damping_relative = 0.001
        critical_arc_lm_damping_absolute = 1.0e-6
        source_plane_covariance_floor = 1.0e-6
        image_plane_scatter_floor_arcsec = 1.0e-3
        source_plane_outlier_sigma_arcsec = 10.0
        image_presence_penalty_weight = 0.0
        image_presence_match_radius_arcsec = 1.0
        image_presence_temperature_arcsec = 0.5
        image_presence_count_softness = 0.05
        image_presence_count_margin = 0.05
        likelihood_stabilizer_residual_loss = cluster_solver.LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_GAUSSIAN
        likelihood_stabilizer_student_t_nu = 4.0
        critical_arc_critical_direction_sigma_arcsec = 10.0
        critical_arc_base_prob = 0.1
        critical_arc_max_prob = 0.85
        critical_arc_singular_threshold = 0.4
        critical_arc_singular_softness = 0.1

        def _physical_parameter_vector(self, params: jnp.ndarray) -> jnp.ndarray:
            return params

        def _image_sigma_int_from_physical(self, physical_params: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(0.1, dtype=jnp.float64)

        def _build_packed_lens_state_with_validity_from_physical(self, *_args, **_kwargs):
            return {}, {"is_valid": jnp.asarray(True)}

        def _ray_shooting_for_components(self, *_args, **_kwargs):
            return jnp.asarray([0.2, 1.2], dtype=jnp.float64), jnp.asarray([0.1, -0.1], dtype=jnp.float64)

        def _lensing_jacobian_for_components(self, *_args, **_kwargs):
            ones = jnp.ones(2, dtype=jnp.float64)
            zeros = jnp.zeros(2, dtype=jnp.float64)
            return ones, zeros, zeros, 0.2 * ones

        def _explicit_source_position_vectors_for_bin(self, *_args, **_kwargs):
            return (
                jnp.asarray([0.0, 1.0], dtype=jnp.float64),
                jnp.asarray([0.0, 0.0], dtype=jnp.float64),
                jnp.asarray(True),
                jnp.asarray(0.0, dtype=jnp.float64),
            )

    evaluator = FakeCriticalArcEvaluator()
    theta = np.asarray([0.0], dtype=float)
    image_rows, bin_rows = cluster_solver._critical_arc_debug_terms_for_state(
        SimpleNamespace(),
        evaluator,
        theta,
        state_index=0,
        state_label="probe",
        chain=1,
        draw=0,
    )

    params = jnp.asarray(theta, dtype=jnp.float64)
    physical_params = evaluator._physical_parameter_vector(params)
    image_sigma_int = evaluator._image_sigma_int_from_physical(physical_params)
    beta_x, beta_y = evaluator._ray_shooting_for_components(None, bin_data.x_obs, bin_data.y_obs, {})
    jacobian_entries = evaluator._lensing_jacobian_for_components(None, bin_data.x_obs, bin_data.y_obs, {})
    beta_family_x, beta_family_y, _has_sources, transport = evaluator._explicit_source_position_vectors_for_bin(
        params,
        physical_params,
        bin_data,
        beta_x,
        beta_y,
        image_sigma_int,
        jacobian_entries,
    )
    (
        residual_x,
        residual_y,
        singular_min,
        singular_max,
        critical_p00,
        critical_p01,
        critical_p11,
        _residual_finite,
    ) = cluster_solver._critical_arc_lm_geometry_from_jacobian(
        beta_x - beta_family_x,
        beta_y - beta_family_y,
        *jacobian_entries,
        trust_radius_arcsec=evaluator.critical_arc_lm_trust_radius_arcsec,
        lm_damping_relative=evaluator.critical_arc_lm_damping_relative,
        lm_damping_absolute=evaluator.critical_arc_lm_damping_absolute,
    )
    direct_bin_loglike = cluster_solver._critical_arc_mixture_image_plane_bin_loglike(
        residual_x=residual_x,
        residual_y=residual_y,
        jac_a00=jacobian_entries[0],
        jac_a01=jacobian_entries[1],
        jac_a10=jacobian_entries[2],
        jac_a11=jacobian_entries[3],
        family_idx=bin_data.family_index_per_image,
        n_families=bin_data.n_families,
        sigma_per_image=bin_data.sigma_per_image,
        reliability_per_image=bin_data.reliability_per_image,
        image_has_constraint=bin_data.image_has_constraint,
        image_sigma_int=image_sigma_int,
        covariance_floor=evaluator.source_plane_covariance_floor,
        outlier_sigma_arcsec=evaluator.source_plane_outlier_sigma_arcsec,
        image_scatter_floor_arcsec=evaluator.image_plane_scatter_floor_arcsec,
        image_presence_penalty_weight=evaluator.image_presence_penalty_weight,
        image_presence_match_radius_arcsec=evaluator.image_presence_match_radius_arcsec,
        image_presence_temperature_arcsec=evaluator.image_presence_temperature_arcsec,
        image_presence_count_softness=evaluator.image_presence_count_softness,
        image_presence_count_margin=evaluator.image_presence_count_margin,
        residual_loss=evaluator.likelihood_stabilizer_residual_loss,
        student_t_nu=evaluator.likelihood_stabilizer_student_t_nu,
        critical_direction_sigma_arcsec=evaluator.critical_arc_critical_direction_sigma_arcsec,
        base_prob=evaluator.critical_arc_base_prob,
        max_prob=evaluator.critical_arc_max_prob,
        singular_threshold=evaluator.critical_arc_singular_threshold,
        singular_softness=evaluator.critical_arc_singular_softness,
        singular_min_precomputed=singular_min,
        singular_max_precomputed=singular_max,
        critical_direction_projector_entries=(critical_p00, critical_p01, critical_p11),
    )

    assert len(image_rows) == 2
    assert len(bin_rows) == 1
    assert sum(row["final_image_mixture_contribution"] for row in image_rows) == pytest.approx(
        bin_rows[0]["bin_loglike_without_transport"]
    )
    assert bin_rows[0]["bin_loglike_with_transport"] == pytest.approx(
        float(direct_bin_loglike + transport)
    )
    for row in image_rows:
        assert "outlier_responsibility" in row
        assert "point_mixture_responsibility" in row
        assert "arc_mixture_responsibility" in row
        assert row["inlier_responsibility"] + row["outlier_responsibility"] == pytest.approx(1.0)
        assert (
            row["point_mixture_responsibility"]
            + row["arc_mixture_responsibility"]
            + row["outlier_responsibility"]
        ) == pytest.approx(1.0)
        assert row["point_inlier_responsibility"] + row["arc_inlier_responsibility"] == pytest.approx(1.0)
    outlier_values = np.asarray([row["outlier_responsibility"] for row in image_rows], dtype=float)
    assert bin_rows[0]["outlier_responsibility_mean"] == pytest.approx(float(np.mean(outlier_values)))
    assert bin_rows[0]["outlier_responsibility_max"] == pytest.approx(float(np.max(outlier_values)))
    assert bin_rows[0]["outlier_responsibility_sum"] == pytest.approx(float(np.sum(outlier_values)))
    assert bin_rows[0]["outlier_responsibility_count_gt_0p1"] == int(np.sum(outlier_values > 0.1))
    assert bin_rows[0]["outlier_responsibility_count_gt_0p5"] == int(np.sum(outlier_values > 0.5))


def test_sampler_debug_writer_creates_core_csvs(tmp_path: Path) -> None:
    spec = ParameterSpec(
        "theta",
        "theta",
        "theta",
        0,
        "theta",
        "normal",
        -np.inf,
        np.inf,
        0.1,
        mean=0.0,
        std=1.0,
    )
    grouped = np.asarray([[[0.0], [0.1], [0.2]]], dtype=float)
    posterior = PosteriorResults(
        samples=grouped.reshape(-1, 1),
        log_prob=np.asarray([0.0, -0.1, -0.2], dtype=float),
        accept_prob=np.ones(3, dtype=float),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.ones(3, dtype=float),
        warmup_steps=2,
        sample_steps=3,
        num_chains=1,
        init_diagnostics={"retained_chain_indices": [0], "chain_seed_labels": ["seed"]},
        grouped_samples=grouped,
        grouped_log_prob=np.asarray([[0.0, -0.1, -0.2]], dtype=float),
        sampler="numpyro_nuts",
    )
    nuts_init = NUTSInitialization(
        init_params={},
        chain_seeds=[ChainSeed(values=np.asarray([0.0], dtype=float), source_label="seed")],
        diagnostics={"retained_chain_indices": [0], "chain_seed_labels": ["seed"]},
        reference_theta=np.asarray([0.0], dtype=float),
    )

    class FakeEvaluator:
        sample_likelihood_mode = SAMPLE_LIKELIHOOD_SOURCE

        def _source_loglike_fn(self, theta: jnp.ndarray) -> jnp.ndarray:
            return -0.5 * jnp.sum(jnp.square(theta))

    extra = {
        "accept_prob": np.ones((1, 3), dtype=float),
        "num_steps": np.ones((1, 3), dtype=float),
        "potential_energy": -posterior.grouped_log_prob,
        "energy": np.asarray([[1.0, 1.1, 1.2]], dtype=float),
        "adapt_state.step_size": np.asarray([[0.01, 0.01, 0.01]], dtype=float),
        "adapt_state.inverse_mass_matrix": {("theta",): np.ones((1, 3, 1, 1), dtype=float)},
    }

    diagnostics = cluster_solver._write_sampler_debug_diagnostics(
        argparse.Namespace(output_dir=str(tmp_path), debug_sampler_diagnostics=True, max_tree_depth=8),
        SimpleNamespace(run_name="run", parameter_specs=[spec]),
        FakeEvaluator(),
        nuts_init,
        posterior,
        extra,
    )

    tables = tmp_path / "run" / "tables"
    assert diagnostics["nuts_integrator_diagnostics_rows"] == 1
    integrator_df = pd.read_csv(tables / "nuts_integrator_diagnostics.csv")
    state_df = pd.read_csv(tables / "sampler_state_diagnostics.csv")
    scan_df = pd.read_csv(tables / "sampler_direction_scan.csv")
    assert {"chain", "step_size_last", "energy_bfmi_like", "inverse_mass_diag_median"} <= set(
        integrator_df.columns
    )
    assert {"state_label", "prior_log_prob", "likelihood_log_prob", "gradient_norm"} <= set(
        state_df.columns
    )
    assert {"state_label", "parameter", "delta", "posterior_log_prob"} <= set(scan_df.columns)


def _svi_health_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            "halo.x_center",
            "halo_x_center",
            "halo",
            81,
            "x_center",
            "normal",
            -np.inf,
            np.inf,
            0.1,
            mean=0.0,
            std=1.0,
        ),
        ParameterSpec(
            "halo.y_center",
            "halo_y_center",
            "halo",
            81,
            "y_center",
            "uniform",
            -2.0,
            2.0,
            0.1,
        ),
    ]


def _svi_health_samples() -> np.ndarray:
    grid = np.linspace(-0.2, 0.2, 100)
    return np.column_stack([grid, grid[::-1]])


def _svi_health_chain_seeds() -> list[ChainSeed]:
    return [
        ChainSeed(values=np.asarray([0.01, -0.01], dtype=float), source_label="chain_1"),
        ChainSeed(values=np.asarray([-0.01, 0.01], dtype=float), source_label="chain_2"),
    ]


def test_svi_health_diagnostics_healthy_guide_has_no_warnings() -> None:
    samples = _svi_health_samples()
    log_prob = np.linspace(-12.0, -8.0, samples.shape[0])

    metrics, warnings = cluster_solver._svi_health_diagnostics(
        _svi_health_specs(),
        samples,
        log_prob,
        np.asarray([0.0, 0.0], dtype=float),
        -10.0,
        _svi_health_chain_seeds(),
        np.asarray([-10.2, -9.9], dtype=float),
    )

    assert warnings == []
    assert metrics["guide_finite_draw_fraction"] == pytest.approx(1.0)
    assert metrics["center_log_prob_percentile"] == pytest.approx(50.0)
    assert metrics["chain_start_distinct_count"] == 2


def test_svi_health_diagnostics_warns_on_nonfinite_guide_draws() -> None:
    samples = _svi_health_samples()
    log_prob = np.linspace(-12.0, -8.0, samples.shape[0])
    log_prob[5] = np.nan

    metrics, warnings = cluster_solver._svi_health_diagnostics(
        _svi_health_specs(),
        samples,
        log_prob,
        np.asarray([0.0, 0.0], dtype=float),
        -10.0,
        _svi_health_chain_seeds(),
        np.asarray([-10.2, -9.9], dtype=float),
    )

    assert metrics["guide_finite_draw_fraction"] < 1.0
    assert any("non-finite SVI guide draws" in warning for warning in warnings)


def test_svi_health_diagnostics_warns_on_large_logprob_spread() -> None:
    samples = _svi_health_samples()
    log_prob = np.linspace(-5000.0, 0.0, samples.shape[0])

    metrics, warnings = cluster_solver._svi_health_diagnostics(
        _svi_health_specs(),
        samples,
        log_prob,
        np.asarray([0.0, 0.0], dtype=float),
        -10.0,
        _svi_health_chain_seeds(),
        np.asarray([-10.2, -9.9], dtype=float),
    )

    assert metrics["guide_log_prob_q95_q05_width"] > cluster_solver.SVI_HEALTH_LOGPROB_SPREAD_WARNING
    assert any("wide SVI guide log-prob spread" in warning for warning in warnings)


def test_svi_health_diagnostics_warns_on_poor_chain_start_logprob() -> None:
    samples = _svi_health_samples()
    log_prob = np.linspace(-12.0, -8.0, samples.shape[0])

    metrics, warnings = cluster_solver._svi_health_diagnostics(
        _svi_health_specs(),
        samples,
        log_prob,
        np.asarray([0.0, 0.0], dtype=float),
        -10.0,
        _svi_health_chain_seeds(),
        np.asarray([-100.0, -9.9], dtype=float),
    )

    assert metrics["chain_start_log_prob_delta_min_from_guide_q05"] < -50.0
    assert any("poor SVI chain-start log-prob" in warning for warning in warnings)


def test_svi_health_diagnostics_warns_on_near_zero_parameter_spread() -> None:
    samples = _svi_health_samples()
    samples[:, 0] = 0.0
    log_prob = np.linspace(-12.0, -8.0, samples.shape[0])

    metrics, warnings = cluster_solver._svi_health_diagnostics(
        _svi_health_specs(),
        samples,
        log_prob,
        np.asarray([0.0, 0.0], dtype=float),
        -10.0,
        _svi_health_chain_seeds(),
        np.asarray([-10.2, -9.9], dtype=float),
    )

    worst = metrics["guide_worst_std_over_prior_scale"][0]
    assert worst["sample_name"] == "halo_x_center"
    assert worst["std_over_prior_scale"] == pytest.approx(0.0)
    assert any("near-zero SVI guide spread" in warning for warning in warnings)


def test_svi_health_diagnostics_are_recorded_in_init_diagnostics() -> None:
    posterior = PosteriorResults(
        samples=_svi_health_samples(),
        log_prob=np.linspace(-12.0, -8.0, 100),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=100,
        num_chains=0,
        init_diagnostics={},
        sampler="svi",
    )
    svi_diagnostics: dict[str, Any] = {}
    metrics = {"guide_finite_draw_fraction": 1.0}
    warnings = ["synthetic SVI health warning"]

    cluster_solver._record_svi_health_diagnostics(posterior, svi_diagnostics, metrics, warnings)

    assert posterior.init_diagnostics["svi_health_metrics"] == metrics
    assert posterior.init_diagnostics["svi_health_warnings"] == warnings
    assert svi_diagnostics["svi_health_metrics"] == metrics
    assert svi_diagnostics["svi_health_warnings"] == warnings


def test_svi_refresh_cache_probe_delta_is_bounded_and_tracks_metadata() -> None:
    before_cache = {
        1.0: {
            "a": np.arange(1000, dtype=float),
            "b": np.ones((2, 2), dtype=float),
            "remove": np.ones(5, dtype=float),
        }
    }
    after_cache = {
        1.0: {
            "a": np.arange(1000, dtype=float) + 5.0,
            "added": np.ones(5, dtype=float),
            "b": np.ones((3, 2), dtype=float),
        }
    }

    before = cluster_solver._svi_refresh_cache_snapshot(
        before_cache,
        np.asarray([0.0, 0.0], dtype=float),
        max_arrays=2,
        max_values_per_array=4,
    )
    after = cluster_solver._svi_refresh_cache_snapshot(
        after_cache,
        np.asarray([1.0, 0.0], dtype=float),
        max_arrays=2,
        max_values_per_array=4,
    )

    assert before.shapes == {
        "z=1.a": (1000,),
        "z=1.b": (2, 2),
        "z=1.remove": (5,),
    }
    assert set(before.probes) == {"z=1.a", "z=1.b"}
    assert before.probes["z=1.a"].values.tolist() == pytest.approx([0.0, 333.0, 666.0, 999.0])

    delta = cluster_solver._svi_refresh_cache_delta(before, after)

    assert delta.compared_arrays == 1
    assert delta.compared_values == 4
    assert delta.max_abs == pytest.approx(5.0)
    assert delta.rms == pytest.approx(5.0)
    assert delta.added_arrays == 1
    assert delta.removed_arrays == 1
    assert delta.shape_changed_arrays == 1


def test_svi_refresh_cache_probe_delta_reports_unchanged_disabled_and_empty() -> None:
    cache = {1.0: {"a": np.asarray([1.0, 2.0, 3.0], dtype=float)}}
    before = cluster_solver._svi_refresh_cache_snapshot(cache, np.asarray([0.0], dtype=float))
    after = cluster_solver._svi_refresh_cache_snapshot(cache, np.asarray([0.0], dtype=float))

    unchanged = cluster_solver._svi_refresh_cache_delta(before, after)
    assert unchanged.compared_values == 3
    assert unchanged.max_abs == pytest.approx(0.0)
    assert unchanged.rms == pytest.approx(0.0)

    disabled = cluster_solver._svi_refresh_cache_snapshot({}, None, enabled=False)
    assert cluster_solver._format_svi_refresh_cache_delta(
        "surrogate",
        cluster_solver._svi_refresh_cache_delta(disabled, disabled),
    ) == "surrogate=disabled"

    empty = cluster_solver._svi_refresh_cache_snapshot({}, None)
    assert cluster_solver._format_svi_refresh_cache_delta(
        "scaling",
        cluster_solver._svi_refresh_cache_delta(empty, empty),
    ) == "scaling=empty arrays=0"


def test_run_svi_fit_logs_bounded_refresh_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = _svi_health_specs()
    centers = [
        np.asarray([1.0, -1.0], dtype=float),
        np.asarray([1.5, -1.5], dtype=float),
        np.asarray([1.75, -1.25], dtype=float),
    ]
    logs: list[str] = []

    class FakeGuide:
        def __init__(self, center: np.ndarray) -> None:
            self.center = np.asarray(center, dtype=float)

        def median(self, _params: dict[str, Any]) -> dict[str, float]:
            return {spec.sample_name: float(self.center[index]) for index, spec in enumerate(specs)}

        def sample_posterior(self, _key: Any, _params: dict[str, Any], sample_shape: tuple[int, ...]):
            return {
                spec.sample_name: jnp.asarray(np.full(sample_shape, self.center[index], dtype=float))
                for index, spec in enumerate(specs)
            }

    class FakeSVI:
        run_count = 0

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def run(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
            FakeSVI.run_count += 1
            return SimpleNamespace(state=object(), losses=np.asarray([10.0 - FakeSVI.run_count], dtype=float))

        def get_params(self, _state: object) -> dict[str, Any]:
            return {}

    class FakeEvaluator:
        surrogate_enabled = True

        def __init__(self) -> None:
            self.timing_totals: dict[str, float] = {}
            self.surrogate_reference_params = np.asarray([0.0, 0.0], dtype=float)
            self.scaling_scatter_reference_params = np.asarray([0.0, 0.0], dtype=float)
            self.source_metric_reference_params = np.asarray([0.0, 0.0], dtype=float)
            self._set_caches(self.surrogate_reference_params)

        def _set_caches(self, params: np.ndarray) -> None:
            base = float(np.sum(params))
            grid = np.arange(1024, dtype=float)
            self.surrogate_cache_by_z = {
                1.0: SurrogateBinCache(
                    effective_z_source=1.0,
                    inactive_alpha_x=grid + base,
                    inactive_alpha_y=grid - base,
                    inactive_alpha_dx_dparams=np.full((2, 1024), base, dtype=float),
                    inactive_alpha_dy_dparams=np.full((2, 1024), -base, dtype=float),
                )
            }
            self.scaling_scatter_cache_by_z = {
                1.0: {
                    "sigma_x": grid + 2.0 * base,
                    "sigma_y": grid - 2.0 * base,
                }
            }
            self.source_metric_cache_by_z = {
                1.0: {
                    "inv_abs_mu": np.ones(1024, dtype=float) + base,
                }
            }

        def refresh_surrogate(self, params: np.ndarray, reason: str = "manual") -> None:
            del reason
            self.surrogate_reference_params = np.asarray(params, dtype=float).copy()
            self._set_caches(self.surrogate_reference_params)

        def refresh_scaling_scatter_cache(self, params: np.ndarray, reason: str = "manual") -> None:
            del reason
            self.scaling_scatter_reference_params = np.asarray(params, dtype=float).copy()
            self._set_caches(self.scaling_scatter_reference_params)

        def refresh_source_metric_cache(self, params: np.ndarray, reason: str = "manual") -> None:
            del reason
            self.source_metric_reference_params = np.asarray(params, dtype=float).copy()
            self._set_caches(self.source_metric_reference_params)

    def fake_make_guide(_sample_model: object, _parameter_specs: list[ParameterSpec], _init_values=None) -> FakeGuide:
        return FakeGuide(centers.pop(0))

    monkeypatch.setattr(cluster_solver, "_make_auto_normal_guide", fake_make_guide)
    monkeypatch.setattr(cluster_solver, "SVI", FakeSVI)
    monkeypatch.setattr(cluster_solver, "_run_logged_phase", lambda _args, _phase_name, fn, **_kwargs: fn())
    monkeypatch.setattr(cluster_solver, "_posterior_logprob_matrix", lambda _specs, _evaluator, samples: np.zeros(samples.shape[0]))
    monkeypatch.setattr(cluster_solver, "_log_posterior_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log", lambda _args, message: logs.append(str(message)))

    args = argparse.Namespace(
        svi_steps=3,
        refresh_every=1,
        svi_learning_rate=0.1,
        seed=11,
        nuts_init_boundary_frac=0.0,
        samples=1,
        chains=1,
    )
    state = SimpleNamespace(
        parameter_specs=specs,
        svi_init_values=None,
    )

    best_fit, posterior, diagnostics = cluster_solver._run_svi_fit(args, state, FakeEvaluator(), object())

    assert best_fit.tolist() == pytest.approx([1.75, -1.25])
    assert posterior.samples.shape == (1, 2)
    assert diagnostics["svi_cache_refresh_count"] == 2
    refresh_logs = [message for message in logs if message.startswith("[svi] refresh ")]
    assert [message.split()[2] for message in refresh_logs] == [
        "reason=svi_block_1",
        "reason=svi_block_2",
        "reason=svi_final",
    ]
    assert all("theta_linf=" in message and "top=" in message for message in refresh_logs)
    assert all("surrogate=max=" in message and "probes=" in message for message in refresh_logs)
    assert all("scaling=max=" in message and "source_metric=max=" in message for message in refresh_logs)


def test_cluster_solver_rejects_removed_profile_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for flag, value in (("--profile-variant", "original"), ("--compact-skip-factor", "1.0")):
        monkeypatch.setattr(
            sys,
            "argv",
            ["cluster_solver", "--par-path", "data/clustersim/input.par", flag, value],
        )

        with pytest.raises(SystemExit):
            _parse_args()


def test_rejects_legacy_pjaffe_artifacts() -> None:
    with pytest.raises(ValueError, match="unsupported compact/PJAFFE"):
        cluster_solver._validate_supported_lens_model_list(["DPIE_NIE", "PJAFFE_ELLIPSE_POTENTIAL"], "legacy")


def test_bulk_lensing_jacobian_matches_manual_dpie_finite_difference() -> None:
    fake = SimpleNamespace(
        use_bulk_ray_shooting=True,
        bulk_index_list=np.asarray([0], dtype=np.int32),
        models_by_effective_z={2.0: cluster_solver.LensModelBulk(unique_lens_model_list=["DPIE_NIE"], multi_plane=False)},
    )
    fake._bulk_ray_shooting_kwargs_from_indices = cluster_solver.ClusterJAXEvaluator._bulk_ray_shooting_kwargs_from_indices.__get__(
        fake,
        type(fake),
    )
    packed_state = {
        "sigma0": jnp.asarray([1.2], dtype=jnp.float64),
        "Ra": jnp.asarray([0.15], dtype=jnp.float64),
        "Rs": jnp.asarray([3.0], dtype=jnp.float64),
        "e1": jnp.asarray([0.05], dtype=jnp.float64),
        "e2": jnp.asarray([-0.02], dtype=jnp.float64),
        "center_x": jnp.asarray([0.1], dtype=jnp.float64),
        "center_y": jnp.asarray([-0.1], dtype=jnp.float64),
        "gamma1": jnp.asarray([0.0], dtype=jnp.float64),
        "gamma2": jnp.asarray([0.0], dtype=jnp.float64),
    }
    x = jnp.asarray([0.2, 1.0, 3.0], dtype=jnp.float64)
    y = jnp.asarray([0.4, 2.0, -1.0], dtype=jnp.float64)
    eps = jnp.asarray(1.0e-5, dtype=jnp.float64)

    jacobian = cluster_solver.ClusterJAXEvaluator._lensing_jacobian_for_components(fake, 2.0, x, y, packed_state)
    kwargs = fake._bulk_ray_shooting_kwargs_from_indices(packed_state)
    model = fake.models_by_effective_z[2.0]
    beta_x_plus, beta_y_plus = model.ray_shooting(x + eps, y, kwargs)
    beta_x_minus, beta_y_minus = model.ray_shooting(x - eps, y, kwargs)
    beta_x_y_plus, beta_y_y_plus = model.ray_shooting(x, y + eps, kwargs)
    beta_x_y_minus, beta_y_y_minus = model.ray_shooting(x, y - eps, kwargs)
    expected = (
        (beta_x_plus - beta_x_minus) / (2.0 * eps),
        (beta_x_y_plus - beta_x_y_minus) / (2.0 * eps),
        (beta_y_plus - beta_y_minus) / (2.0 * eps),
        (beta_y_y_plus - beta_y_y_minus) / (2.0 * eps),
    )

    for value, reference in zip(jacobian, expected):
        np.testing.assert_allclose(np.asarray(value), np.asarray(reference), atol=1.0e-5, rtol=1.0e-5)


def test_cluster_solver_rejects_removed_likelihood_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--likelihood-mode",
            "source",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


def test_cluster_solver_image_plane_mode_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "data/clustersim/input.par"])

    args = _parse_args()

    assert args.image_plane_mode == IMAGE_PLANE_MODE_NONE
    assert args.sample_likelihood_mode == SAMPLE_LIKELIHOOD_SOURCE
    controls = _normalize_stage_fit_controls(args)
    assert controls["stage2"].fit_method == "svi+nuts"
    assert controls["stage3"].fit_method == "svi+nuts"
    assert controls["stage4"].fit_method == "svi+nuts"


def test_cluster_solver_accepts_local_jacobian_image_plane_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        ],
    )

    args = _parse_args()

    assert args.image_plane_mode == IMAGE_PLANE_MODE_LOCAL_JACOBIAN


def test_cluster_solver_accepts_start_at_stage3_for_local_jacobian(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
            "--start-at-stage3",
            "--resume-fast",
            "--fit-method",
            "svi+nuts",
            "svi",
            "--svi-steps",
            "1000",
            "400",
            "--warmup",
            "200",
            "0",
            "--samples",
            "50",
            "20",
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert args.start_at_stage3 is True
    assert args.resume_fast is True
    assert controls["stage3"].fit_method == "svi"
    assert controls["stage3"].svi_steps == 400
    assert controls["stage3"].warmup == 0
    assert controls["stage3"].samples == 20


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"fit_mode": "joint"}, "fit-mode sequential"),
        ({"image_plane_mode": IMAGE_PLANE_MODE_NONE}, "stage-3-capable"),
        (
            {
                "image_plane_mode": IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
                "skip_stage3_image_plane_local_jacobian": True,
            },
            "skip-stage3",
        ),
    ],
)
def test_cluster_solver_rejects_invalid_start_at_stage3_controls(
    updates: dict[str, Any],
    message: str,
) -> None:
    args = argparse.Namespace(
        fit_mode=cluster_solver.FIT_MODE_SEQUENTIAL,
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        start_at_stage3=True,
        skip_stage3_image_plane_local_jacobian=False,
        resume_fast=False,
        fit_method=["svi+nuts", "svi"],
        svi_steps=[1000, 400],
        warmup=[200, 0],
        samples=[50, 20],
    )
    for key, value in updates.items():
        setattr(args, key, value)

    with pytest.raises(SystemExit, match=message):
        _normalize_stage_fit_controls(args)


def test_cluster_solver_accepts_linearized_forward_beta_image_plane_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        ],
    )

    args = _parse_args()

    assert args.image_plane_mode == IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA


def test_cluster_solver_accepts_blocked_linearized_forward_beta_image_plane_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED,
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert args.image_plane_mode == IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED
    assert cluster_solver._stage4_sample_likelihood_mode(args) == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
    assert cluster_solver._stage4_run_directory_name(args) == "stage4_blocked_linearized_image_plane"
    assert controls["stage4"].fit_method == "svi+nuts"


def test_cluster_solver_accepts_forward_metric_image_plane_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_FORWARD_METRIC,
            "--image-presence-penalty-weight",
            "3.5",
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert args.image_plane_mode == IMAGE_PLANE_MODE_FORWARD_METRIC
    assert args.image_presence_penalty_weight == pytest.approx(3.5)
    assert cluster_solver._stage4_sample_likelihood_mode(args) == SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE
    assert cluster_solver._stage4_run_directory_name(args) == "stage4_forward_metric_image_plane"
    assert controls["stage4"].fit_method == "svi+nuts"


def test_cluster_solver_accepts_anchored_solved_forward_beta_image_plane_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
            "--sampling-engine",
            "full",
            "--anchored-image-plane-solve-steps",
            "0",
            "--anchored-image-plane-trust-radius-arcsec",
            "0.25",
            "--anchored-image-plane-lm-damping-relative",
            "0.002",
            "--anchored-image-plane-lm-damping-absolute",
            "1e-5",
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert args.image_plane_mode == IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA
    assert args.anchored_image_plane_solve_steps == 0
    assert args.anchored_image_plane_trust_radius_arcsec == pytest.approx(0.25)
    assert args.anchored_image_plane_lm_damping_relative == pytest.approx(0.002)
    assert args.anchored_image_plane_lm_damping_absolute == pytest.approx(1.0e-5)
    assert cluster_solver._stage4_sample_likelihood_mode(args) == SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE
    assert cluster_solver._stage4_run_directory_name(args) == "stage4_anchored_solved_image_plane"
    assert controls["stage4"].fit_method == "svi+nuts"


def test_cluster_solver_accepts_critical_arc_mixture_image_plane_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
            "--sampling-engine",
            "refreshing_surrogate",
            "--critical-arc-critical-direction-sigma-arcsec",
            "4.5",
            "--critical-arc-base-prob",
            "0.2",
            "--critical-arc-max-prob",
            "0.7",
            "--critical-arc-singular-threshold",
            "0.15",
            "--critical-arc-singular-softness",
            "0.04",
            "--critical-arc-lm-damping-relative",
            "0.002",
            "--critical-arc-lm-damping-absolute",
            "1e-5",
            "--critical-arc-lm-trust-radius-arcsec",
            "18.0",
            "--arc-aware-noncritical-support-radius-arcsec",
            "1.25",
            "--arc-aware-max-arclength-arcsec",
            "12.0",
            "--arc-aware-curve-step-arcsec",
            "0.2",
            "--image-catalog-family-cutout-image-dir",
            "data/BUFFALO_Images",
            "--image-catalog-family-cutout-image-scale",
            "30mas",
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert args.image_plane_mode == IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE
    assert args.critical_arc_critical_direction_sigma_arcsec == pytest.approx(4.5)
    assert args.critical_arc_base_prob == pytest.approx(0.2)
    assert args.critical_arc_max_prob == pytest.approx(0.7)
    assert args.critical_arc_singular_threshold == pytest.approx(0.15)
    assert args.critical_arc_singular_softness == pytest.approx(0.04)
    assert args.critical_arc_lm_damping_relative == pytest.approx(0.002)
    assert args.critical_arc_lm_damping_absolute == pytest.approx(1.0e-5)
    assert args.critical_arc_lm_trust_radius_arcsec == pytest.approx(18.0)
    assert args.arc_aware_noncritical_support_radius_arcsec == pytest.approx(1.25)
    assert args.arc_aware_max_arclength_arcsec == pytest.approx(12.0)
    assert args.arc_aware_curve_step_arcsec == pytest.approx(0.2)
    assert args.image_catalog_family_cutout_image_dir == "data/BUFFALO_Images"
    assert args.image_catalog_family_cutout_image_scale == "30mas"
    assert cluster_solver._stage4_sample_likelihood_mode(args) == SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE
    assert cluster_solver._stage4_run_directory_name(args) == "stage4_critical_arc_mixture_image_plane"
    assert controls["stage4"].fit_method == "svi+nuts"


def test_cluster_solver_accepts_fold_regularized_image_plane_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
            "--sampling-engine",
            "refreshing_surrogate",
            "--fold-curvature-arcsec-inv",
            "1.25",
            "--image-plane-newton-steps",
            "0",
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert args.image_plane_mode == IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA
    assert args.fold_curvature_arcsec_inv == pytest.approx(1.25)
    assert (
        cluster_solver._stage4_sample_likelihood_mode(args)
        == SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE
    )
    assert cluster_solver._stage4_run_directory_name(args) == "stage4_fold_regularized_image_plane"
    assert controls["stage4"].fit_method == "svi+nuts"


def test_cluster_solver_accepts_ff_sims_cutout_image_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-catalog-family-cutout-image-dir",
            "/data/ff_sims/fits",
            "--image-catalog-family-cutout-image-scale",
            "auto",
            "--image-catalog-family-cutout-bands",
            "F435W",
            "F606W",
            "F814W",
            "--kappa-true-fits",
            "data/ff_sims/hera/kappa_z9_0.fits",
        ],
    )

    args = _parse_args()

    assert args.image_catalog_family_cutout_image_dir == "/data/ff_sims/fits"
    assert args.image_catalog_family_cutout_image_scale == "auto"
    assert args.image_catalog_family_cutout_bands == ["F435W", "F606W", "F814W"]
    assert args.kappa_true_fits == "data/ff_sims/hera/kappa_z9_0.fits"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--critical-arc-speed-mode", "all"),
        ("--critical-arc-mask-margin-softness", "4.0"),
    ],
)
def test_cluster_solver_rejects_removed_critical_arc_speed_controls(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
            flag,
            value,
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


def test_cluster_solver_rejects_invalid_image_catalog_cutout_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-catalog-family-cutout-image-dir",
            "data/BUFFALO_Images",
            "--image-catalog-family-cutout-image-scale",
            "15mas",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


def test_cluster_solver_rejects_conditional_whitened_for_critical_arc_mixture_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
            "--source-position-parameterization",
            "conditional-whitened",
        ],
    )

    args = _parse_args()
    with pytest.raises(SystemExit, match="conditional-whitened"):
        _normalize_stage_fit_controls(args)


def test_cluster_solver_rejects_conditional_whitened_for_fold_regularized_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
            "--source-position-parameterization",
            "conditional-whitened",
        ],
    )

    args = _parse_args()
    with pytest.raises(SystemExit, match="conditional-whitened"):
        _normalize_stage_fit_controls(args)


def test_cluster_solver_rejects_positive_newton_steps_for_fold_regularized_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
            "--image-plane-newton-steps",
            "1",
        ],
    )

    args = _parse_args()
    with pytest.raises(SystemExit, match="image-plane-newton-steps must be 0"):
        _normalize_stage_fit_controls(args)


def test_cluster_solver_accepts_refreshing_surrogate_for_zero_step_anchored_solved_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
            "--sampling-engine",
            "refreshing_surrogate",
            "--anchored-image-plane-solve-steps",
            "0",
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert args.anchored_image_plane_solve_steps == 0
    assert controls["stage4"].fit_method == "svi+nuts"


def test_cluster_solver_rejects_refreshing_surrogate_for_iterative_anchored_solved_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
            "--sampling-engine",
            "refreshing_surrogate",
            "--anchored-image-plane-solve-steps",
            "1",
        ],
    )

    args = _parse_args()
    with pytest.raises(SystemExit, match="anchored-image-plane-solve-steps"):
        _normalize_stage_fit_controls(args)


def test_cluster_solver_rejects_marginal_image_plane_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            "marginal-" "image-plane",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


@pytest.mark.parametrize("stage4_fit_method", ["svi", "smc", "nuts"])
def test_blocked_linearized_forward_beta_requires_stage4_svi_nuts(
    monkeypatch: pytest.MonkeyPatch,
    stage4_fit_method: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED,
            "--fit-method",
            "svi+nuts",
            "svi+nuts",
            stage4_fit_method,
        ],
    )

    args = _parse_args()
    with pytest.raises(SystemExit, match="requires stage-4 --fit-method svi\\+nuts"):
        _normalize_stage_fit_controls(args)


def test_direct_nuts_initialization_uses_reference_and_clipped_chain_seeds() -> None:
    specs = [
        ParameterSpec("halo.x", "halo_x", "halo", 0, "x", "uniform", -1.0, 1.0, 0.1),
        ParameterSpec(
            "halo.y",
            "halo_y",
            "halo",
            0,
            "y",
            "truncated_normal",
            -2.0,
            2.0,
            0.1,
            mean=0.0,
            std=1.0,
        ),
    ]
    args = argparse.Namespace(
        chains=3,
        seed=12,
        nuts_init_jitter_frac=0.2,
        nuts_init_boundary_frac=0.1,
    )

    init = cluster_solver._nuts_initialization_from_reference(
        args,
        specs,
        np.asarray([0.5, 1.0], dtype=float),
    )

    assert init.diagnostics["strategy_requested"] == cluster_solver.FIT_METHOD_NUTS
    assert init.diagnostics["strategy_used"] == "previous_stage"
    assert init.diagnostics["svi_used"] is False
    assert init.diagnostics["chain_seed_labels"] == [
        "direct_nuts_chain_1",
        "direct_nuts_chain_2",
        "direct_nuts_chain_3",
    ]
    assert np.asarray(init.init_params["halo_x"]).shape == (3,)
    assert np.asarray(init.init_params["halo_y"]).shape == (3,)
    for seed in init.chain_seeds:
        assert -1.0 < seed.values[0] < 1.0
        assert -2.0 < seed.values[1] < 2.0
    np.testing.assert_allclose(init.reference_theta, np.asarray([0.5, 1.0], dtype=float))


def test_run_inference_direct_nuts_uses_state_init_without_svi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    specs = [
        ParameterSpec("halo.x", "halo_x", "halo", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0),
        ParameterSpec("halo.y", "halo_y", "halo", 0, "y", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0),
    ]
    state = SimpleNamespace(
        run_name="stage4_critical_arc_mixture_image_plane",
        par_path="data/clustersim/input.par",
        parameter_specs=specs,
        family_data=[],
        bin_data=[],
        svi_init_values={"halo_x": 0.25, "halo_y": -0.5},
    )

    class FakeEvaluator:
        def __init__(self) -> None:
            self.surrogate_enabled = False
            self.sampling_engine = cluster_solver.SAMPLING_ENGINE_FULL
            self.timing_totals = {"nuts_runtime": 0.0}
            self.refresh_reasons: list[str] = []

        def refresh_scaling_scatter_cache(self, _theta, reason: str) -> None:
            self.refresh_reasons.append(reason)

        def refresh_source_metric_cache(self, _theta, reason: str) -> None:
            self.refresh_reasons.append(reason)

        def release_runtime_caches(self) -> None:
            self.refresh_reasons.append("released")

    evaluator = FakeEvaluator()
    captured: dict[str, Any] = {}

    def fake_run_nuts(_args, _state, _evaluator, _sample_model, nuts_init):
        captured["nuts_init"] = nuts_init
        return PosteriorResults(
            samples=np.asarray([[0.2, -0.4], [0.35, -0.2]], dtype=float),
            log_prob=np.asarray([-2.0, -1.0], dtype=float),
            accept_prob=np.asarray([0.8, 0.9], dtype=float),
            diverging=np.asarray([False, False], dtype=bool),
            num_steps=np.asarray([3.0, 5.0], dtype=float),
            warmup_steps=2,
            sample_steps=2,
            num_chains=1,
            init_diagnostics={"retained_chain_indices": [0]},
            grouped_samples=np.asarray([[[0.2, -0.4], [0.35, -0.2]]], dtype=float),
            grouped_log_prob=np.asarray([[-2.0, -1.0]], dtype=float),
            sampler="numpyro_nuts",
        )

    monkeypatch.setattr(cluster_solver, "_configure_debug_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_prepare_direct_evaluator", lambda _args, _state: (evaluator, np.zeros(2)))
    monkeypatch.setattr(cluster_solver, "_posterior_model", lambda _specs, _evaluator: object())
    monkeypatch.setattr(cluster_solver, "_run_numpyro_nuts_sampler", fake_run_nuts)
    monkeypatch.setattr(
        cluster_solver,
        "_run_svi_fit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("SVI should not run")),
    )
    monkeypatch.setattr(
        cluster_solver,
        "_save_inference_checkpoint",
        lambda _args, _state, _evaluator, run_dir, best_fit, posterior: (Path(run_dir) / "artifacts", best_fit, posterior),
    )
    monkeypatch.setattr(cluster_solver, "_output_evaluator_for_validation", lambda _args, _state, evaluator, _best_fit: evaluator)
    monkeypatch.setattr(
        cluster_solver,
        "_approximate_evaluation",
        lambda _evaluator, _best_fit: EvaluationResult(loglike=0.0, family_predictions={}),
    )
    monkeypatch.setattr(cluster_solver, "_write_truth_validation_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log_posterior_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_max_likelihood_best_fit_from_posterior",
        lambda _args, _evaluator, _posterior, fallback: np.asarray(fallback, dtype=float),
    )
    args = argparse.Namespace(
        fit_method=cluster_solver.FIT_METHOD_NUTS,
        output_dir=str(tmp_path),
        skip_validation=True,
        quick_diagnostics=False,
        skip_plots=True,
        quiet=True,
        chains=2,
        seed=22,
        nuts_init_jitter_frac=0.0,
        nuts_init_boundary_frac=0.1,
    )

    cluster_solver._run_inference(args, state, tmp_path / state.run_name)

    nuts_init = captured["nuts_init"]
    assert nuts_init.diagnostics["svi_used"] is False
    assert nuts_init.diagnostics["strategy_requested"] == cluster_solver.FIT_METHOD_NUTS
    np.testing.assert_allclose(nuts_init.reference_theta, np.asarray([0.25, -0.5], dtype=float))
    assert [seed.source_label for seed in nuts_init.chain_seeds] == [
        "direct_nuts_chain_1",
        "direct_nuts_chain_2",
    ]
    assert "nuts_initial" in evaluator.refresh_reasons
    assert "post_nuts" in evaluator.refresh_reasons


@pytest.mark.parametrize("dense_mass", [True, False])
def test_numpyro_nuts_sampler_passes_dense_mass_to_numpyro(
    monkeypatch: pytest.MonkeyPatch,
    dense_mass: bool,
) -> None:
    captured: dict[str, Any] = {}

    def fake_nuts(_model, **kwargs):
        captured["nuts_kwargs"] = dict(kwargs)
        return object()

    class FakeMCMC:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, *_args, **_kwargs) -> None:
            return None

        def get_samples(self, group_by_chain: bool = False):
            assert group_by_chain is True
            return {"halo_x": jnp.asarray([[0.1, 0.2]], dtype=jnp.float64)}

        def get_extra_fields(self, group_by_chain: bool = False):
            assert group_by_chain is True
            return {
                "accept_prob": jnp.asarray([[0.8, 0.9]], dtype=jnp.float64),
                "diverging": jnp.asarray([[False, False]], dtype=bool),
                "num_steps": jnp.asarray([[3.0, 5.0]], dtype=jnp.float64),
                "potential_energy": jnp.asarray([[2.0, 1.0]], dtype=jnp.float64),
            }

    monkeypatch.setattr(cluster_solver, "NUTS", fake_nuts)
    monkeypatch.setattr(cluster_solver, "MCMC", FakeMCMC)
    monkeypatch.setattr(cluster_solver, "_apply_nuts_quality_gate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log_posterior_summary", lambda *_args, **_kwargs: None)

    specs = [
        ParameterSpec("halo.x", "halo_x", "halo", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0)
    ]
    state = SimpleNamespace(parameter_specs=specs)
    evaluator = SimpleNamespace(
        timing_totals={"nuts_runtime": 0.0},
        invalid_state_rejection_count=0,
        invalid_state_reason_counts={},
    )
    init = NUTSInitialization(
        init_params={"halo_x": jnp.asarray([0.0], dtype=jnp.float64)},
        chain_seeds=[ChainSeed(values=np.asarray([0.0], dtype=float), source_label="seed")],
        diagnostics={"strategy_used": "test", "distinct_chain_seeds": 1},
        reference_theta=np.asarray([0.0], dtype=float),
    )
    args = argparse.Namespace(
        chains=1,
        warmup=0,
        samples=2,
        thin=1,
        target_accept=0.8,
        max_tree_depth=4,
        initial_step_size=0.1,
        seed=123,
        dense_mass=dense_mass,
        debug_sampler_diagnostics=False,
        quiet=True,
    )

    posterior = cluster_solver._run_numpyro_nuts_sampler(args, state, evaluator, object(), init)

    assert captured["nuts_kwargs"]["dense_mass"] is dense_mass
    assert posterior.init_diagnostics["dense_mass"] is dense_mass
    assert posterior.init_diagnostics["nuts_dense_mass"] is dense_mass


@pytest.mark.parametrize("dense_mass", [True, False])
def test_block_nuts_once_passes_dense_mass_to_numpyro(
    monkeypatch: pytest.MonkeyPatch,
    dense_mass: bool,
) -> None:
    captured: dict[str, Any] = {}

    def fake_nuts(_model, **kwargs):
        captured["nuts_kwargs"] = dict(kwargs)
        return object()

    class FakeMCMC:
        def __init__(self, *_args, **_kwargs) -> None:
            self.last_state = SimpleNamespace(
                adapt_state=SimpleNamespace(
                    step_size=jnp.asarray(0.1),
                    inverse_mass_matrix=jnp.asarray([1.0]),
                )
            )

        def run(self, *_args, **_kwargs) -> None:
            return None

        def get_samples(self, group_by_chain: bool = False):
            assert group_by_chain is False
            return {"halo_x": jnp.asarray([0.25], dtype=jnp.float64)}

        def get_extra_fields(self, group_by_chain: bool = False):
            assert group_by_chain is False
            return {
                "accept_prob": jnp.asarray([0.9], dtype=jnp.float64),
                "diverging": jnp.asarray([False], dtype=bool),
                "num_steps": jnp.asarray([3.0], dtype=jnp.float64),
                "potential_energy": jnp.asarray([1.0], dtype=jnp.float64),
            }

    monkeypatch.setattr(cluster_solver, "NUTS", fake_nuts)
    monkeypatch.setattr(cluster_solver, "MCMC", FakeMCMC)
    monkeypatch.setattr(cluster_solver, "_conditioned_posterior_model_for_block", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cluster_solver, "_block_init_params", lambda *_args, **_kwargs: {})

    specs = [
        ParameterSpec("halo.x", "halo_x", "halo", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0)
    ]
    args = argparse.Namespace(
        target_accept=0.8,
        max_tree_depth=4,
        initial_step_size=0.1,
        dense_mass=dense_mass,
        quiet=True,
    )
    evaluator = SimpleNamespace()
    block = cluster_solver.BlockedNUTSParameterBlock("non_source", (0,))

    cluster_solver._run_block_nuts_once(
        args,
        specs,
        evaluator,
        np.asarray([0.0], dtype=float),
        block,
        jax.random.PRNGKey(0),
        num_warmup=0,
    )

    assert captured["nuts_kwargs"]["dense_mass"] is dense_mass


def test_blocked_nuts_parameter_blocks_partition_source_positions() -> None:
    specs = [
        ParameterSpec("halo.x", "halo_x", "halo", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0),
        ParameterSpec(
            "source.1.beta_x",
            "source_1_beta_x",
            "1",
            0,
            "beta_x",
            "normal",
            -np.inf,
            np.inf,
            0.1,
            mean=0.0,
            std=1.0,
            component_family="source_position",
        ),
        ParameterSpec("scale.sigma", "scale_sigma", "scale", 0, "sigma", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0),
    ]

    blocks = cluster_solver._blocked_nuts_parameter_blocks(specs)

    assert [(block.name, block.indices) for block in blocks] == [
        ("non_source", (0, 2)),
        ("source_position", (1,)),
    ]


def test_blocked_nuts_conditioned_toy_sampler_reconstructs_full_samples() -> None:
    specs = [
        ParameterSpec("lens.x", "lens_x", "lens", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0),
        ParameterSpec(
            "source.1.beta_x",
            "source_1_beta_x",
            "1",
            0,
            "beta_x",
            "normal",
            -np.inf,
            np.inf,
            0.1,
            mean=0.0,
            std=1.0,
            component_family="source_position",
        ),
    ]

    class ToyEvaluator:
        def __init__(self) -> None:
            self.timing_totals = {"nuts_runtime": 0.0}
            self.invalid_state_rejection_count = 0
            self.invalid_state_reason_counts = {}
            self._source_loglike_fn = jax.jit(
                lambda theta: -0.5 * jnp.square(theta[0] + theta[1] - 0.75)
            )

    args = argparse.Namespace(
        chains=1,
        warmup=2,
        samples=2,
        thin=1,
        target_accept=0.7,
        initial_step_size=0.2,
        max_tree_depth=3,
        seed=11,
        blocked_nuts_cycles=2,
        blocked_nuts_pilot_warmup=2,
        quiet=True,
    )
    evaluator = ToyEvaluator()
    blocks = cluster_solver._blocked_nuts_parameter_blocks(specs)
    theta = np.asarray([0.0, 0.25], dtype=float)
    updated, _metrics, _adapted = cluster_solver._run_block_nuts_once(
        args,
        specs,
        evaluator,
        theta,
        blocks[0],
        jax.random.PRNGKey(5),
        num_warmup=2,
        adapt=True,
    )
    assert updated[1] == pytest.approx(theta[1])
    assert abs(updated[0] - theta[0]) > 1.0e-12

    state = SimpleNamespace(parameter_specs=specs)
    init = NUTSInitialization(
        init_params={},
        chain_seeds=[ChainSeed(values=np.asarray([0.0, 0.0], dtype=float), source_label="toy")],
        diagnostics={"strategy_used": "toy", "distinct_chain_seeds": 1},
        reference_theta=np.asarray([0.0, 0.0], dtype=float),
    )
    posterior = cluster_solver._run_blocked_numpyro_nuts_sampler(args, state, evaluator, init)

    assert posterior.sampler == "numpyro_blocked_nuts"
    assert posterior.grouped_samples.shape == (1, 2, 2)
    assert posterior.samples.shape == (2, 2)
    assert posterior.grouped_log_prob.shape == (1, 2)
    assert posterior.init_diagnostics["blocked_nuts_block_sizes"] == {
        "non_source": 1,
        "source_position": 1,
    }


def test_cluster_solver_rejects_removed_stage35_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--run-stage35-" "marginal-" "image-plane",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


@pytest.mark.parametrize("stage4_fit_method", ["smc", "nuts"])
@pytest.mark.parametrize(
    "image_plane_mode",
    [
        IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        IMAGE_PLANE_MODE_FORWARD_METRIC,
        IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
        IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
        IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
    ],
)
def test_cluster_solver_accepts_direct_stage4_samplers_for_non_blocked_stage4_modes(
    monkeypatch: pytest.MonkeyPatch,
    image_plane_mode: str,
    stage4_fit_method: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            image_plane_mode,
            "--fit-method",
            "svi+nuts",
            "svi+nuts",
            stage4_fit_method,
            "--smc-particles",
            "32",
            "--smc-mcmc-kernel",
            "mala",
            "--smc-mcmc-steps",
            "2",
            "--smc-target-ess-frac",
            "0.7",
            "--smc-max-temperature-steps",
            "8",
            "--smc-rmh-scale",
            "0.6",
            "--smc-mala-step-size",
            "0.02",
            "--anchored-image-plane-solve-steps",
            "0",
            "--jax-default-device",
            "cpu",
            "--smc-device",
            "auto",
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert controls["stage2"].fit_method == "svi+nuts"
    assert controls["stage3"].fit_method == "svi+nuts"
    assert controls["stage4"].fit_method == stage4_fit_method
    assert args.smc_particles == 32
    assert args.smc_mcmc_kernel == "mala"
    assert args.smc_mcmc_steps == 2
    assert args.smc_target_ess_frac == pytest.approx(0.7)
    assert args.smc_max_temperature_steps == 8
    assert args.smc_rmh_scale == pytest.approx(0.6)
    assert args.smc_mala_step_size == pytest.approx(0.02)
    assert args.jax_default_device == "cpu"
    assert args.smc_device == "auto"


@pytest.mark.parametrize(
    "extra_args",
    [
        [
            "--image-plane-mode",
            IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED,
        ],
        [
            "--image-plane-mode",
            IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
            "--fit-method",
            "svi+nuts",
            "svi+nuts",
            "smc",
        ],
    ],
)
def test_cluster_solver_rejects_potfile_mass_size_reparam_unsupported_sampler_paths(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--potfile-mass-size-reparam",
            *extra_args,
        ],
    )

    args = _parse_args()
    with pytest.raises(SystemExit, match="potfile-mass-size-reparam"):
        _normalize_stage_fit_controls(args)


def test_cluster_solver_rejects_unavailable_explicit_gpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cluster_solver.jax, "devices", lambda kind=None: [] if kind == "gpu" else [object()])
    with pytest.raises(ValueError, match="--smc-device=gpu"):
        cluster_solver._resolve_jax_device("gpu", flag_name="--smc-device")


def test_cluster_solver_resolves_cpu_and_auto_devices() -> None:
    assert cluster_solver._resolve_jax_device("auto", flag_name="--jax-default-device") is None
    cpu_device = cluster_solver._resolve_jax_device("cpu", flag_name="--jax-default-device")
    assert cpu_device is not None
    assert cluster_solver._jax_device_backend(cpu_device) == "cpu"


@pytest.mark.parametrize(
    "fit_methods",
    [
        ["smc", "svi+nuts", "svi+nuts"],
        ["svi+nuts", "smc", "svi+nuts"],
        ["smc"],
        ["nuts", "svi+nuts", "svi+nuts"],
        ["svi+nuts", "nuts", "svi+nuts"],
        ["nuts"],
    ],
)
def test_cluster_solver_rejects_direct_stage4_samplers_outside_stage4(
    monkeypatch: pytest.MonkeyPatch,
    fit_methods: list[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
            "--fit-method",
            *fit_methods,
        ],
    )

    args = _parse_args()
    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--smc-particles", "0"),
        ("--smc-mcmc-steps", "0"),
        ("--smc-target-ess-frac", "0"),
        ("--smc-target-ess-frac", "1.1"),
        ("--smc-max-temperature-steps", "0"),
        ("--smc-rmh-scale", "0"),
        ("--smc-mala-step-size", "0"),
    ],
)
def test_cluster_solver_rejects_invalid_smc_controls(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            flag,
            value,
        ],
    )

    args = _parse_args()
    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_cluster_solver_accepts_image_presence_penalty_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
            "--image-plane-scatter-floor-arcsec",
            "0.05",
            "--image-plane-scatter-prior",
            "lognormal",
            "--image-plane-scatter-prior-median-arcsec",
            "0.25",
            "--image-plane-scatter-prior-log-sigma",
            "0.4",
            "--image-presence-penalty-weight",
            "3.5",
            "--image-presence-match-radius-arcsec",
            "0.4",
            "--image-presence-temperature-arcsec",
            "0.08",
            "--image-presence-count-softness",
            "0.03",
            "--image-presence-count-margin",
            "0.02",
            "--likelihood-stabilizer-max-gain",
            "50",
            "--likelihood-stabilizer-max-residual-arcsec",
            "3",
            "--likelihood-stabilizer-residual-loss",
            "student-t",
            "--likelihood-stabilizer-student-t-nu",
            "4",
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert args.image_plane_scatter_floor_arcsec == pytest.approx(0.05)
    assert args.image_plane_scatter_prior == "lognormal"
    assert args.image_plane_scatter_prior_median_arcsec == pytest.approx(0.25)
    assert args.image_plane_scatter_prior_log_sigma == pytest.approx(0.4)
    assert args.image_presence_penalty_weight == pytest.approx(3.5)
    assert args.image_presence_match_radius_arcsec == pytest.approx(0.4)
    assert args.image_presence_temperature_arcsec == pytest.approx(0.08)
    assert args.image_presence_count_softness == pytest.approx(0.03)
    assert args.image_presence_count_margin == pytest.approx(0.02)
    assert args.likelihood_stabilizer_max_gain == pytest.approx(50.0)
    assert args.likelihood_stabilizer_max_residual_arcsec == pytest.approx(3.0)
    assert args.likelihood_stabilizer_residual_loss == "student-t"
    assert args.likelihood_stabilizer_student_t_nu == pytest.approx(4.0)
    assert controls["stage4"].fit_method == "svi+nuts"


@pytest.mark.parametrize(
    "old_flag",
    [
        "--linearized-image-plane-max-gain",
        "--linearized-image-plane-max-residual-arcsec",
        "--linearized-image-plane-residual-loss",
        "--linearized-image-plane-student-t-nu",
    ],
)
def test_cluster_solver_rejects_old_likelihood_stabilizer_flags(
    monkeypatch: pytest.MonkeyPatch,
    old_flag: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            old_flag,
            "1",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


def test_saved_likelihood_stabilizer_arg_prefers_new_key_and_reads_legacy_key() -> None:
    assert (
        cluster_solver._saved_likelihood_stabilizer_arg(
            {"linearized_image_plane_max_gain": 12.0},
            "likelihood_stabilizer_max_gain",
            0.0,
        )
        == 12.0
    )
    assert (
        cluster_solver._saved_likelihood_stabilizer_arg(
            {
                "likelihood_stabilizer_max_gain": 8.0,
                "linearized_image_plane_max_gain": 12.0,
            },
            "likelihood_stabilizer_max_gain",
            0.0,
        )
        == 8.0
    )
    assert (
        cluster_solver._saved_likelihood_stabilizer_arg(
            {},
            "likelihood_stabilizer_max_gain",
            0.0,
        )
        == 0.0
    )
    normalized = cluster_solver._normalized_saved_likelihood_stabilizer_args(
        {
            "linearized_image_plane_max_gain": 12.0,
            "linearized_image_plane_max_residual_arcsec": 3.0,
            "linearized_image_plane_residual_loss": "student-t",
            "linearized_image_plane_student_t_nu": 6.0,
        }
    )
    assert normalized == {
        "likelihood_stabilizer_max_gain": 12.0,
        "likelihood_stabilizer_max_residual_arcsec": 3.0,
        "likelihood_stabilizer_residual_loss": "student-t",
        "likelihood_stabilizer_student_t_nu": 6.0,
    }
    plot_args = {
        **normalized,
        "linearized_image_plane_max_gain": 12.0,
    }
    cluster_solver._drop_legacy_likelihood_stabilizer_args(plot_args)
    assert "likelihood_stabilizer_max_gain" in plot_args
    assert "linearized_image_plane_max_gain" not in plot_args


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--image-presence-penalty-weight", "-0.1"),
        ("--image-presence-match-radius-arcsec", "0"),
        ("--image-presence-temperature-arcsec", "0"),
        ("--image-presence-count-softness", "0"),
        ("--image-presence-count-margin", "-0.1"),
        ("--image-plane-scatter-floor-arcsec", "0"),
        ("--image-plane-scatter-floor-arcsec", "-0.1"),
        ("--image-plane-scatter-floor-arcsec", "nan"),
        ("--image-plane-scatter-prior-median-arcsec", "0"),
        ("--image-plane-scatter-prior-median-arcsec", "nan"),
        ("--image-plane-scatter-prior-log-sigma", "0"),
        ("--image-plane-scatter-prior-log-sigma", "nan"),
        ("--likelihood-stabilizer-max-gain", "-0.1"),
        ("--likelihood-stabilizer-max-gain", "nan"),
        ("--likelihood-stabilizer-max-residual-arcsec", "-0.1"),
        ("--likelihood-stabilizer-max-residual-arcsec", "inf"),
        ("--likelihood-stabilizer-student-t-nu", "0"),
        ("--likelihood-stabilizer-student-t-nu", "nan"),
    ],
)
def test_cluster_solver_rejects_invalid_image_presence_controls(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            flag,
            value,
        ],
    )

    args = _parse_args()
    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--image-plane-scatter-floor-arcsec", "0.5", "--image-plane-scatter-upper-arcsec", "0.5"],
        [
            "--image-plane-scatter-prior",
            "lognormal",
            "--image-plane-scatter-floor-arcsec",
            "0.1",
            "--image-plane-scatter-upper-arcsec",
            "0.5",
            "--image-plane-scatter-prior-median-arcsec",
            "0.05",
        ],
        [
            "--image-plane-scatter-prior",
            "lognormal",
            "--image-plane-scatter-floor-arcsec",
            "0.1",
            "--image-plane-scatter-upper-arcsec",
            "0.5",
            "--image-plane-scatter-prior-median-arcsec",
            "0.6",
        ],
    ],
)
def test_cluster_solver_rejects_image_scatter_support_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            *extra_args,
        ],
    )

    args = _parse_args()
    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_image_presence_effective_default_weight_only_stage4() -> None:
    assert _effective_image_presence_penalty_weight(
        None,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    ) == pytest.approx(2.0)
    assert _effective_image_presence_penalty_weight(
        None,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_FORWARD_METRIC,
    ) == pytest.approx(2.0)
    assert _effective_image_presence_penalty_weight(
        None,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
    ) == pytest.approx(2.0)
    assert _effective_image_presence_penalty_weight(
        None,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    ) == pytest.approx(0.0)
    assert _effective_image_presence_penalty_weight(
        None,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        fit_mode="evidence-ns",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    ) == pytest.approx(0.0)
    assert _effective_image_presence_penalty_weight(
        0.0,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    ) == pytest.approx(0.0)


def test_cluster_solver_accepts_evidence_ns_fit_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-mode",
            "evidence-ns",
            "--evidence-source-prior-sigma-arcsec",
            "5.0",
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert args.fit_mode == "evidence-ns"
    assert args.evidence_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
    assert controls["stage2"].to_json() == {
        "fit_method": "ns",
        "svi_steps": 0,
        "warmup": 0,
        "samples": 0,
        "max_tree_depth": 10,
    }


def test_cluster_solver_rejects_marginal_evidence_likelihood_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-mode",
            "evidence-ns",
            "--evidence-source-prior-sigma-arcsec",
            "5.0",
            "--evidence-likelihood-mode",
            "linearized-" "marginal-" "beta-image-plane",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


def test_cluster_solver_evidence_ns_accepts_sampled_source_image_plane_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-mode",
            "evidence-ns",
            "--evidence-source-prior-sigma-arcsec",
            "5.0",
            "--evidence-likelihood-mode",
            SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
            "--source-position-parameterization",
            "direct",
            "--image-plane-newton-steps",
            "1",
            "--sampling-engine",
            "full",
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert args.evidence_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
    assert args.source_position_parameterization == "direct"
    assert args.image_plane_newton_steps == 1
    assert controls["stage2"].to_json() == {
        "fit_method": "ns",
        "svi_steps": 0,
        "warmup": 0,
        "samples": 0,
        "max_tree_depth": 10,
    }


def test_cluster_solver_accepts_active_subset_sampling_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--sampling-engine",
            "active_subset",
        ],
    )

    args = _parse_args()

    assert args.sampling_engine == "active_subset"


def test_cluster_solver_evidence_ns_rejects_active_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-mode",
            "evidence-ns",
            "--evidence-source-prior-sigma-arcsec",
            "5.0",
            "--sampling-engine",
            "active_subset",
        ],
    )

    args = _parse_args()
    with pytest.raises(SystemExit, match="active_subset"):
        _normalize_stage_fit_controls(args)


def test_cluster_solver_evidence_ns_rejects_missing_source_prior_sigma(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-mode",
            "evidence-ns",
        ],
    )

    args = _parse_args()
    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_cluster_solver_evidence_ns_ignores_fit_method_and_stage_controls() -> None:
    controls = _normalize_stage_fit_controls(
        argparse.Namespace(
            fit_mode="evidence-ns",
            fit_method=["svi+nuts", "svi"],
            warmup=[-1, 200],
            samples=[0, 25],
            image_plane_mode=IMAGE_PLANE_MODE_NONE,
            evidence_source_prior_sigma_arcsec=5.0,
        )
    )

    assert controls["stage2"].to_json() == {
        "fit_method": "ns",
        "svi_steps": 0,
        "warmup": 0,
        "samples": 0,
        "max_tree_depth": 10,
    }


def test_cluster_solver_rejects_removed_linearized_stage_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--linearized-image-plane-stage",
            "joint-beta",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


def test_cluster_solver_accepts_two_value_fit_controls_for_local_jacobian(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
            "--fit-method",
            "svi+nuts",
            "svi",
            "--svi-steps",
            "1200",
            "400",
            "--warmup",
            "2000",
            "0",
            "--samples",
            "250",
            "100",
            "--max-tree-depth",
            "9",
            "7",
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1200,
        "warmup": 2000,
        "samples": 250,
        "max_tree_depth": 9,
    }
    assert controls["stage3"].to_json() == {
        "fit_method": "svi",
        "svi_steps": 400,
        "warmup": 0,
        "samples": 100,
        "max_tree_depth": 7,
    }
    assert controls["stage4"].to_json() == {
        "fit_method": "svi",
        "svi_steps": 400,
        "warmup": 0,
        "samples": 100,
        "max_tree_depth": 7,
    }


def test_cluster_solver_rejects_unknown_image_plane_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            "full-solver",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


def test_cluster_solver_rejects_removed_ott_sinkhorn_image_plane_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--image-plane-mode",
            "ott-sinkhorn-forward-beta-image-plane",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


def test_stage_fit_controls_scalar_values_apply_to_stage2_and_stage3() -> None:
    args = argparse.Namespace(
        fit_method="svi+nuts",
        svi_steps=1500,
        warmup=2000,
        samples=250,
        max_tree_depth=9,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    controls = _normalize_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1500,
        "warmup": 2000,
        "samples": 250,
        "max_tree_depth": 9,
    }
    assert controls["stage3"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1500,
        "warmup": 2000,
        "samples": 250,
        "max_tree_depth": 9,
    }
    assert controls["stage4"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1500,
        "warmup": 2000,
        "samples": 250,
        "max_tree_depth": 9,
    }


def test_stage_fit_controls_two_values_map_to_stage2_and_stage3() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi"],
        svi_steps=[1500, 400],
        warmup=[2000, 0],
        samples=[250, 100],
        max_tree_depth=[9, 7],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    controls = _normalize_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1500,
        "warmup": 2000,
        "samples": 250,
        "max_tree_depth": 9,
    }
    assert controls["stage3"].to_json() == {
        "fit_method": "svi",
        "svi_steps": 400,
        "warmup": 0,
        "samples": 100,
        "max_tree_depth": 7,
    }
    assert controls["stage4"].to_json() == {
        "fit_method": "svi",
        "svi_steps": 400,
        "warmup": 0,
        "samples": 100,
        "max_tree_depth": 7,
    }


def test_stage_fit_controls_three_values_map_to_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        svi_steps=[1500, 1000, 200],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 20],
        max_tree_depth=[10, 8, 6],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    )

    controls = _normalize_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1500,
        "warmup": 2000,
        "samples": 250,
        "max_tree_depth": 10,
    }
    assert controls["stage3"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1000,
        "warmup": 1000,
        "samples": 100,
        "max_tree_depth": 8,
    }
    assert controls["stage4"].to_json() == {
        "fit_method": "svi",
        "svi_steps": 200,
        "warmup": 0,
        "samples": 20,
        "max_tree_depth": 6,
    }


def test_stage_fit_controls_reject_four_values() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi", "svi+nuts"],
        warmup=[2000, 1000, 0, 500],
        samples=[250, 100, 20, 80],
        max_tree_depth=[10, 8, 6, 5],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
    )

    with pytest.raises(SystemExit, match="at most three"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_linearized_stage4_skipped_stage3_uses_stage4_values() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi"],
        svi_steps=[1500, 200],
        warmup=[2000, 0],
        samples=[250, 20],
        max_tree_depth=[10, 6],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=True,
    )

    controls = _normalize_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1500,
        "warmup": 2000,
        "samples": 250,
        "max_tree_depth": 10,
    }
    assert controls["stage4"].to_json() == {
        "fit_method": "svi",
        "svi_steps": 200,
        "warmup": 0,
        "samples": 20,
        "max_tree_depth": 6,
    }


def test_stage_fit_controls_reject_refreshing_surrogate_stage4_newton_steps() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 20],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        sampling_engine="refreshing_surrogate",
        image_plane_newton_steps=1,
    )

    with pytest.raises(SystemExit, match="refreshing_surrogate"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_three_values_without_stage4_mode() -> None:
    args = argparse.Namespace(
        fit_method=["svi", "svi", "svi"],
        warmup=1,
        samples=2,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_two_values_without_image_plane_stage() -> None:
    args = argparse.Namespace(
        fit_method="svi+nuts",
        svi_steps=[1000, 500],
        warmup=1,
        samples=2,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_two_values_for_non_sequential_runs() -> None:
    args = argparse.Namespace(
        fit_method="svi+nuts",
        svi_steps=[1000, 500],
        warmup=1,
        samples=2,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_four_svi_step_values() -> None:
    args = argparse.Namespace(
        fit_method="svi+nuts",
        svi_steps=[1000, 500, 100, 50],
        warmup=1,
        samples=2,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    )

    with pytest.raises(SystemExit, match="at most three"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_two_values_map_to_stage4_when_stage3_skipped() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi"],
        svi_steps=[1500, 200],
        warmup=[2000, 0],
        samples=[250, 20],
        max_tree_depth=[9, 5],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=True,
    )

    controls = _normalize_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1500,
        "warmup": 2000,
        "samples": 250,
        "max_tree_depth": 9,
    }
    assert controls["stage4"].to_json() == {
        "fit_method": "svi",
        "svi_steps": 200,
        "warmup": 0,
        "samples": 20,
        "max_tree_depth": 5,
    }


def test_stage_fit_controls_reject_ns_for_stage2_without_image_plane() -> None:
    args = argparse.Namespace(
        fit_method="ns",
        warmup=0,
        samples=50,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    with pytest.raises(SystemExit, match="evidence-ns"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_ns_for_final_local_jacobian_stage3() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "ns"],
        warmup=[2000, 0],
        samples=[250, 100],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit, match="evidence-ns"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_ns_for_final_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "ns"],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 50],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    )

    with pytest.raises(SystemExit, match="evidence-ns"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_ns_for_stage4_when_stage3_skipped() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "ns"],
        warmup=[2000, 0],
        samples=[250, 50],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=True,
    )

    with pytest.raises(SystemExit, match="evidence-ns"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_ns_before_final_image_plane_stage() -> None:
    args = argparse.Namespace(
        fit_method=["ns", "svi"],
        warmup=[0, 100],
        samples=[50, 50],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_scalar_ns_when_image_plane_stage_enabled() -> None:
    args = argparse.Namespace(
        fit_method="ns",
        warmup=0,
        samples=50,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_skip_stage3_without_stage4_mode() -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=0,
        samples=2,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        skip_stage3_image_plane_local_jacobian=True,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_non_positive_samples() -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=0,
        samples=0,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_negative_max_tree_depth() -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=0,
        samples=2,
        max_tree_depth=-1,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    with pytest.raises(SystemExit, match="max-tree-depth"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_accept_unlimited_ns_max_samples() -> None:
    args = argparse.Namespace(
        fit_method="svi+nuts",
        warmup=10,
        samples=5,
        fit_mode="evidence-ns",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        ns_max_samples=None,
        evidence_source_prior_sigma_arcsec=5.0,
    )

    controls = _normalize_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {
        "fit_method": "ns",
        "svi_steps": 0,
        "warmup": 0,
        "samples": 0,
        "max_tree_depth": 10,
    }


def test_stage_fit_controls_reject_non_positive_ns_max_samples() -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=0,
        samples=1,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        ns_max_samples=0,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_invalid_ns_max_samples_string() -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=0,
        samples=1,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        ns_max_samples="forever",
    )

    with pytest.raises(SystemExit, match="positive integer"):
        _normalize_stage_fit_controls(args)


def test_evidence_state_uses_forward_beta_source_positions_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-mode",
            "evidence-ns",
            "--evidence-source-prior-sigma-arcsec",
            "5.0",
            "--pos-sigma-arcsec",
            "0",
            "--fit-cosmology-flat-wcdm",
        ],
    )
    args = _parse_args()
    evidence_args = cluster_solver._clone_args(
        args,
        sample_likelihood_mode=cluster_solver.DEFAULT_EVIDENCE_LIKELIHOOD_MODE,
    )

    state = cluster_solver._build_state_from_inputs(evidence_args, fit_mode_override="evidence-ns")

    assert state.sigma_arcsec == pytest.approx(0.0)
    assert sum(spec.component_family == "source_position" for spec in state.parameter_specs) == 2 * len(state.family_data)
    assert sum(spec.component_family == "image_scatter" for spec in state.parameter_specs) == 1
    assert {spec.sample_name for spec in state.parameter_specs if spec.component_family == "cosmology"} == {
        "cosmology_Om0",
        "cosmology_w0",
    }


def test_sampled_source_evidence_state_uses_shared_evidence_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-mode",
            "evidence-ns",
            "--evidence-source-prior-sigma-arcsec",
            "5.0",
            "--evidence-source-prior-mean-x-arcsec",
            "0.2",
            "--evidence-source-prior-mean-y-arcsec",
            "-0.1",
            "--evidence-likelihood-mode",
            SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
            "--pos-sigma-arcsec",
            "0",
        ],
    )
    args = _parse_args()
    evidence_args = cluster_solver._clone_args(
        args,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
    )

    state = cluster_solver._build_state_from_inputs(evidence_args, fit_mode_override="evidence-ns")

    source_specs = [spec for spec in state.parameter_specs if spec.component_family == "source_position"]
    assert len(source_specs) == 2 * len(state.family_data)
    assert sum(spec.component_family == "image_scatter" for spec in state.parameter_specs) == 1
    assert sum(spec.component_family == "source_scatter" for spec in state.parameter_specs) == 0
    assert {spec.transform_kind for spec in source_specs} == {"affine"}
    assert all(spec.physical_std == pytest.approx(5.0) for spec in source_specs)
    assert all(spec.physical_mean == pytest.approx(0.2) for spec in source_specs if spec.field == "beta_x")
    assert all(spec.physical_mean == pytest.approx(-0.1) for spec in source_specs if spec.field == "beta_y")


def test_validation_stage_fit_controls_scalar_values_apply_to_stage2_and_stage3() -> None:
    args = argparse.Namespace(
        fit_method="svi+nuts",
        svi_steps=1200,
        warmup=300,
        samples=500,
        max_tree_depth=8,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1200,
        "warmup": 300,
        "samples": 500,
        "max_tree_depth": 8,
    }
    assert controls["stage3"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1200,
        "warmup": 300,
        "samples": 500,
        "max_tree_depth": 8,
    }
    assert controls["stage4"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1200,
        "warmup": 300,
        "samples": 500,
        "max_tree_depth": 8,
    }


def test_validation_stage_fit_controls_two_values_map_to_stage2_and_stage3() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi"],
        svi_steps=[1200, 300],
        warmup=[1000, 0],
        samples=[250, 100],
        max_tree_depth=[8, 6],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1200,
        "warmup": 1000,
        "samples": 250,
        "max_tree_depth": 8,
    }
    assert controls["stage3"].to_json() == {
        "fit_method": "svi",
        "svi_steps": 300,
        "warmup": 0,
        "samples": 100,
        "max_tree_depth": 6,
    }
    assert controls["stage4"].to_json() == {
        "fit_method": "svi",
        "svi_steps": 300,
        "warmup": 0,
        "samples": 100,
        "max_tree_depth": 6,
    }


def test_validation_stage_fit_controls_accept_start_at_stage3() -> None:
    args = argparse.Namespace(
        solver_fit_mode=validation.SOLVER_FIT_MODE_SEQUENTIAL,
        fit_method=["svi+nuts", "svi"],
        svi_steps=[1200, 300],
        warmup=[1000, 0],
        samples=[250, 100],
        max_tree_depth=[8, 6],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        start_at_stage3=True,
        skip_stage3_image_plane_local_jacobian=False,
        resume_fast=True,
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage3"].to_json() == {
        "fit_method": "svi",
        "svi_steps": 300,
        "warmup": 0,
        "samples": 100,
        "max_tree_depth": 6,
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"solver_fit_mode": validation.SOLVER_FIT_MODE_EVIDENCE_NS},
            "solver-fit-mode sequential",
        ),
        ({"image_plane_mode": IMAGE_PLANE_MODE_NONE}, "stage-3-capable"),
        (
            {
                "image_plane_mode": IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
                "skip_stage3_image_plane_local_jacobian": True,
            },
            "skip-stage3",
        ),
    ],
)
def test_validation_stage_fit_controls_reject_invalid_start_at_stage3_controls(
    updates: dict[str, Any],
    message: str,
) -> None:
    payload = dict(
        solver_fit_mode=validation.SOLVER_FIT_MODE_SEQUENTIAL,
        fit_method=["svi+nuts", "svi"],
        svi_steps=[1200, 300],
        warmup=[1000, 0],
        samples=[250, 100],
        max_tree_depth=[8, 6],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        start_at_stage3=True,
        skip_stage3_image_plane_local_jacobian=False,
        resume_fast=False,
    )
    payload.update(updates)

    with pytest.raises(SystemExit, match=message):
        _normalize_validation_stage_fit_controls(argparse.Namespace(**payload))


def test_validation_stage_fit_controls_three_values_map_to_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        svi_steps=[1200, 600, 100],
        warmup=[1000, 500, 0],
        samples=[250, 100, 20],
        max_tree_depth=[10, 8, 6],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1200,
        "warmup": 1000,
        "samples": 250,
        "max_tree_depth": 10,
    }
    assert controls["stage3"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 600,
        "warmup": 500,
        "samples": 100,
        "max_tree_depth": 8,
    }
    assert controls["stage4"].to_json() == {
        "fit_method": "svi",
        "svi_steps": 100,
        "warmup": 0,
        "samples": 20,
        "max_tree_depth": 6,
    }


def test_validation_stage_fit_controls_reject_refreshing_surrogate_stage4_newton_steps() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[1000, 500, 0],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        sampling_engine="refreshing_surrogate",
        image_plane_newton_steps=1,
    )

    with pytest.raises(SystemExit, match="refreshing_surrogate"):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_accept_anchored_solved_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi+nuts"],
        svi_steps=[1200, 600, 100],
        warmup=[1000, 500, 20],
        samples=[250, 100, 20],
        max_tree_depth=[10, 8, 6],
        image_plane_mode=IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
        sampling_engine="full",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        anchored_image_plane_solve_steps=3,
        anchored_image_plane_trust_radius_arcsec=0.3,
        anchored_image_plane_lm_damping_relative=1.0e-3,
        anchored_image_plane_lm_damping_absolute=1.0e-6,
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage4"].fit_method == "svi+nuts"
    assert validation._validation_final_stage_name(args) == "stage4_anchored_solved_image_plane"


def test_validation_stage_fit_controls_accept_critical_arc_mixture_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi+nuts"],
        svi_steps=[1200, 600, 100],
        warmup=[1000, 500, 20],
        samples=[250, 100, 20],
        max_tree_depth=[10, 8, 6],
        image_plane_mode=IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
        sampling_engine="refreshing_surrogate",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        critical_arc_critical_direction_sigma_arcsec=5.0,
        critical_arc_base_prob=0.10,
        critical_arc_max_prob=0.80,
        critical_arc_singular_threshold=0.20,
        critical_arc_singular_softness=0.05,
        critical_arc_lm_damping_relative=1.0e-3,
        critical_arc_lm_damping_absolute=1.0e-6,
        critical_arc_lm_trust_radius_arcsec=20.0,
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage4"].fit_method == "svi+nuts"
    assert validation._validation_final_stage_name(args) == "stage4_critical_arc_mixture_image_plane"


def test_validation_stage_fit_controls_accept_fold_regularized_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi+nuts"],
        svi_steps=[1200, 600, 100],
        warmup=[1000, 500, 20],
        samples=[250, 100, 20],
        max_tree_depth=[10, 8, 6],
        image_plane_mode=IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
        sampling_engine="refreshing_surrogate",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        fold_curvature_arcsec_inv=1.25,
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage4"].fit_method == "svi+nuts"
    assert validation._validation_final_stage_name(args) == "stage4_fold_regularized_image_plane"


@pytest.mark.parametrize("stage4_fit_method", ["smc", "nuts"])
@pytest.mark.parametrize(
    ("image_plane_mode", "stage_name"),
    [
        (IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA, "stage4_linearized_image_plane"),
        (IMAGE_PLANE_MODE_FORWARD_METRIC, "stage4_forward_metric_image_plane"),
        (IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA, "stage4_anchored_solved_image_plane"),
        (IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE, "stage4_critical_arc_mixture_image_plane"),
        (IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA, "stage4_fold_regularized_image_plane"),
    ],
)
def test_validation_stage_fit_controls_accept_direct_stage4_samplers_for_non_blocked_stage4_modes(
    image_plane_mode: str,
    stage_name: str,
    stage4_fit_method: str,
) -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", stage4_fit_method],
        svi_steps=[1200, 600, 100],
        warmup=[1000, 500, 20],
        samples=[250, 100, 20],
        max_tree_depth=[10, 8, 6],
        image_plane_mode=image_plane_mode,
        sampling_engine="full",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        critical_arc_critical_direction_sigma_arcsec=5.0,
        critical_arc_base_prob=0.10,
        critical_arc_max_prob=0.80,
        critical_arc_singular_threshold=0.20,
        critical_arc_singular_softness=0.05,
        critical_arc_lm_damping_relative=1.0e-3,
        critical_arc_lm_damping_absolute=1.0e-6,
        critical_arc_lm_trust_radius_arcsec=20.0,
        fold_curvature_arcsec_inv=1.0,
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage4"].fit_method == stage4_fit_method
    assert validation._validation_final_stage_name(args) == stage_name


@pytest.mark.parametrize("stage4_fit_method", ["smc", "nuts"])
def test_validation_stage_fit_controls_reject_direct_sampler_for_blocked_linearized_stage4(
    stage4_fit_method: str,
) -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", stage4_fit_method],
        svi_steps=[1200, 600, 100],
        warmup=[1000, 500, 20],
        samples=[250, 100, 20],
        max_tree_depth=[10, 8, 6],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED,
        sampling_engine="full",
        image_plane_newton_steps=0,
    )

    with pytest.raises(SystemExit, match="requires stage-4 --fit-method svi\\+nuts"):
        _normalize_validation_stage_fit_controls(args)


@pytest.mark.parametrize(
    "fit_methods",
    [
        ["nuts", "svi+nuts", "svi+nuts"],
        ["svi+nuts", "nuts", "svi+nuts"],
        ["nuts"],
    ],
)
def test_validation_stage_fit_controls_reject_nuts_outside_stage4(fit_methods: list[str]) -> None:
    args = argparse.Namespace(
        fit_method=fit_methods,
        svi_steps=[1200, 600, 100],
        warmup=[1000, 500, 20],
        samples=[250, 100, 20],
        max_tree_depth=[10, 8, 6],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        sampling_engine="full",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
    )

    with pytest.raises(SystemExit, match="--fit-method nuts"):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_reject_conditional_whitened_for_critical_arc_mixture_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi+nuts"],
        warmup=[1000, 500, 20],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
        sampling_engine="refreshing_surrogate",
        image_plane_newton_steps=0,
        source_position_parameterization="conditional-whitened",
    )

    with pytest.raises(SystemExit, match="conditional-whitened"):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_reject_conditional_whitened_for_fold_regularized_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi+nuts"],
        warmup=[1000, 500, 20],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
        sampling_engine="refreshing_surrogate",
        image_plane_newton_steps=0,
        source_position_parameterization="conditional-whitened",
        fold_curvature_arcsec_inv=1.0,
    )

    with pytest.raises(SystemExit, match="conditional-whitened"):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_reject_positive_newton_steps_for_fold_regularized_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi+nuts"],
        warmup=[1000, 500, 20],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
        sampling_engine="refreshing_surrogate",
        image_plane_newton_steps=1,
        source_position_parameterization="prior-whitened",
        fold_curvature_arcsec_inv=1.0,
    )

    with pytest.raises(SystemExit, match="image-plane-newton-steps must be 0"):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_accept_refreshing_surrogate_for_zero_step_anchored_solved_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi+nuts"],
        warmup=[1000, 500, 20],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
        sampling_engine="refreshing_surrogate",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        anchored_image_plane_solve_steps=0,
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage4"].fit_method == "svi+nuts"


def test_validation_stage_fit_controls_reject_refreshing_surrogate_for_iterative_anchored_solved_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi+nuts"],
        warmup=[1000, 500, 20],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
        sampling_engine="refreshing_surrogate",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        anchored_image_plane_solve_steps=1,
    )

    with pytest.raises(SystemExit, match="anchored-image-plane-solve-steps"):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_reject_ns_for_final_local_jacobian_stage3() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "ns"],
        warmup=[1000, 0],
        samples=[250, 100],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit, match="evidence-ns"):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_reject_ns_for_final_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "ns"],
        warmup=[1000, 500, 0],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    )

    with pytest.raises(SystemExit, match="evidence-ns"):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_reject_scalar_ns_when_image_plane_stage_enabled() -> None:
    args = argparse.Namespace(
        fit_method="ns",
        warmup=0,
        samples=50,
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_reject_invalid_image_presence_control() -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=1,
        samples=2,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        image_presence_temperature_arcsec=0.0,
    )

    with pytest.raises(SystemExit, match="image-presence-temperature"):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_accept_forward_metric_image_presence_penalty() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        svi_steps=[1200, 600, 100],
        warmup=[1000, 500, 0],
        samples=[250, 100, 20],
        max_tree_depth=[10, 8, 6],
        image_plane_mode=IMAGE_PLANE_MODE_FORWARD_METRIC,
        image_plane_newton_steps=0,
        image_presence_penalty_weight=4.0,
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage4"].fit_method == "svi"
    assert args.image_presence_penalty_weight == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("image_plane_scatter_floor_arcsec", 0.0, "image-plane-scatter-floor"),
        ("image_plane_scatter_floor_arcsec", -0.1, "image-plane-scatter-floor"),
        ("image_plane_scatter_floor_arcsec", float("nan"), "image-plane-scatter-floor"),
        ("image_plane_scatter_prior", "gamma", "image-plane-scatter-prior"),
        ("image_plane_scatter_prior_median_arcsec", 0.0, "image-plane-scatter-prior-median"),
        ("image_plane_scatter_prior_median_arcsec", float("nan"), "image-plane-scatter-prior-median"),
        ("image_plane_scatter_prior_log_sigma", 0.0, "image-plane-scatter-prior-log-sigma"),
        ("image_plane_scatter_prior_log_sigma", float("nan"), "image-plane-scatter-prior-log-sigma"),
    ],
)
def test_validation_stage_fit_controls_reject_invalid_image_scatter_control(
    field: str,
    value: float | str,
    message: str,
) -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=1,
        samples=2,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )
    setattr(args, field, value)

    with pytest.raises(SystemExit, match=message):
        _normalize_validation_stage_fit_controls(args)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "image_plane_scatter_floor_arcsec": 0.5,
                "image_plane_scatter_upper_arcsec": 0.5,
            },
            "image-plane-scatter-upper",
        ),
        (
            {
                "image_plane_scatter_prior": "lognormal",
                "image_plane_scatter_floor_arcsec": 0.1,
                "image_plane_scatter_upper_arcsec": 0.5,
                "image_plane_scatter_prior_median_arcsec": 0.05,
            },
            "image-plane-scatter-prior-median",
        ),
        (
            {
                "image_plane_scatter_prior": "lognormal",
                "image_plane_scatter_floor_arcsec": 0.1,
                "image_plane_scatter_upper_arcsec": 0.5,
                "image_plane_scatter_prior_median_arcsec": 0.6,
            },
            "image-plane-scatter-prior-median",
        ),
    ],
)
def test_validation_stage_fit_controls_reject_image_scatter_support_mismatches(
    updates: dict[str, Any],
    message: str,
) -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=1,
        samples=2,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        **updates,
    )

    with pytest.raises(SystemExit, match=message):
        _normalize_validation_stage_fit_controls(args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("likelihood_stabilizer_max_gain", -1.0, "likelihood-stabilizer-max-gain"),
        ("likelihood_stabilizer_max_gain", float("nan"), "likelihood-stabilizer-max-gain"),
        ("likelihood_stabilizer_max_residual_arcsec", -1.0, "likelihood-stabilizer-max-residual"),
        ("likelihood_stabilizer_max_residual_arcsec", float("inf"), "likelihood-stabilizer-max-residual"),
        ("likelihood_stabilizer_student_t_nu", 0.0, "likelihood-stabilizer-student-t-nu"),
        ("likelihood_stabilizer_student_t_nu", float("nan"), "likelihood-stabilizer-student-t-nu"),
    ],
)
def test_validation_stage_fit_controls_reject_invalid_likelihood_stabilizer(
    field: str,
    value: float,
    message: str,
) -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=1,
        samples=2,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )
    setattr(args, field, value)

    with pytest.raises(SystemExit, match=message):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_reject_three_values_without_stage4_mode() -> None:
    args = argparse.Namespace(
        fit_method=["svi", "svi", "svi"],
        warmup=1,
        samples=2,
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_reject_two_values_without_image_plane_stage() -> None:
    args = argparse.Namespace(
        fit_method="svi+nuts",
        svi_steps=[1000, 500],
        warmup=1,
        samples=2,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    with pytest.raises(SystemExit):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_reject_four_svi_step_values() -> None:
    args = argparse.Namespace(
        fit_method="svi+nuts",
        svi_steps=[1000, 500, 100, 50],
        warmup=1,
        samples=2,
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    )

    with pytest.raises(SystemExit, match="at most three"):
        _normalize_validation_stage_fit_controls(args)


def test_validation_stage_fit_controls_reject_invalid_numeric_values() -> None:
    with pytest.raises(SystemExit):
        _normalize_validation_stage_fit_controls(
            argparse.Namespace(
                fit_method="svi",
                warmup=-1,
                samples=2,
                image_plane_mode=IMAGE_PLANE_MODE_NONE,
            )
        )
    with pytest.raises(SystemExit):
        _normalize_validation_stage_fit_controls(
            argparse.Namespace(
                fit_method="svi",
                warmup=0,
                samples=0,
                image_plane_mode=IMAGE_PLANE_MODE_NONE,
            )
        )
    with pytest.raises(SystemExit, match="max-tree-depth"):
        _normalize_validation_stage_fit_controls(
            argparse.Namespace(
                fit_method="svi",
                warmup=0,
                samples=1,
                max_tree_depth=-1,
                image_plane_mode=IMAGE_PLANE_MODE_NONE,
            )
        )
    with pytest.raises(SystemExit):
        _normalize_validation_stage_fit_controls(
            argparse.Namespace(
                fit_method="svi",
                warmup=0,
                samples=1,
                image_plane_mode=IMAGE_PLANE_MODE_NONE,
                ns_max_samples=0,
            )
        )
    with pytest.raises(SystemExit, match="positive integer"):
        _normalize_validation_stage_fit_controls(
            argparse.Namespace(
                fit_method="svi",
                warmup=0,
                samples=1,
                image_plane_mode=IMAGE_PLANE_MODE_NONE,
                ns_max_samples="forever",
            )
        )


def test_validation_stage_fit_controls_accept_unlimited_ns_max_samples() -> None:
    controls = _normalize_validation_stage_fit_controls(
        argparse.Namespace(
            solver_fit_mode="evidence-ns",
            fit_method="svi+nuts",
            warmup=300,
            samples=5,
            image_plane_mode=IMAGE_PLANE_MODE_NONE,
            ns_max_samples=None,
            evidence_source_prior_sigma_arcsec=5.0,
        )
    )

    assert controls["stage2"].to_json() == {
        "fit_method": "ns",
        "svi_steps": 0,
        "warmup": 0,
        "samples": 0,
        "max_tree_depth": 8,
    }


def test_validation_stage_fit_controls_accept_evidence_ns() -> None:
    controls = _normalize_validation_stage_fit_controls(
        argparse.Namespace(
            solver_fit_mode="evidence-ns",
            fit_method=["svi+nuts", "svi"],
            warmup=[-1, 25],
            samples=[0, 25],
            image_plane_mode=IMAGE_PLANE_MODE_NONE,
            evidence_source_prior_sigma_arcsec=5.0,
            ns_max_samples=None,
        )
    )

    assert controls["stage2"].to_json() == {
        "fit_method": "ns",
        "svi_steps": 0,
        "warmup": 0,
        "samples": 0,
        "max_tree_depth": 8,
    }


def test_validation_stage_fit_controls_reject_evidence_ns_without_sigma() -> None:
    with pytest.raises(SystemExit):
        _normalize_validation_stage_fit_controls(
            argparse.Namespace(
                solver_fit_mode="evidence-ns",
                fit_method="svi+nuts",
                warmup=0,
                samples=25,
                image_plane_mode=IMAGE_PLANE_MODE_NONE,
                ns_max_samples=None,
            )
        )


def test_validation_parser_accepts_two_value_controls() -> None:
    args = validation._build_parser().parse_args(
        [
            "--image-plane-mode",
            IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
            "--fit-method",
            "svi+nuts",
            "svi",
            "--svi-steps",
            "1200",
            "300",
            "--warmup",
            "1000",
            "0",
            "--samples",
            "250",
            "100",
            "--max-tree-depth",
            "8",
            "6",
        ]
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {
        "fit_method": "svi+nuts",
        "svi_steps": 1200,
        "warmup": 1000,
        "samples": 250,
        "max_tree_depth": 8,
    }
    assert controls["stage3"].to_json() == {
        "fit_method": "svi",
        "svi_steps": 300,
        "warmup": 0,
        "samples": 100,
        "max_tree_depth": 6,
    }


def test_validation_parser_accepts_start_at_stage3() -> None:
    args = validation._build_parser().parse_args(
        [
            "--start-at-stage3",
            "--image-plane-mode",
            IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        ]
    )

    assert args.start_at_stage3 is True


def test_validation_parser_rejects_marginal_image_plane_mode() -> None:
    with pytest.raises(SystemExit):
        validation._build_parser().parse_args(
            [
                "--image-plane-mode",
                "marginal-" "image-plane",
            ]
        )


def test_validation_parser_accepts_linearized_image_plane_three_value_controls() -> None:
    args = validation._build_parser().parse_args(
        [
            "--image-plane-mode",
            IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
            "--fit-method",
            "svi+nuts",
            "svi+nuts",
            "svi",
            "--svi-steps",
            "1200",
            "600",
            "100",
            "--warmup",
            "1000",
            "500",
            "0",
            "--samples",
            "250",
            "100",
            "20",
            "--max-tree-depth",
            "8",
            "7",
            "6",
        ]
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert args.image_plane_mode == IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA
    assert controls["stage4"].to_json() == {
        "fit_method": "svi",
        "svi_steps": 100,
        "warmup": 0,
        "samples": 20,
        "max_tree_depth": 6,
    }


def test_validation_parser_accepts_anchored_solved_image_plane_controls() -> None:
    args = validation._build_parser().parse_args(
        [
            "--image-plane-mode",
            IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
            "--sampling-engine",
            "full",
            "--anchored-image-plane-solve-steps",
            "4",
            "--anchored-image-plane-trust-radius-arcsec",
            "0.25",
        ]
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert args.image_plane_mode == IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA
    assert args.anchored_image_plane_solve_steps == 4
    assert args.anchored_image_plane_trust_radius_arcsec == pytest.approx(0.25)
    assert controls["stage4"].fit_method == "svi+nuts"


def test_validation_parser_accepts_critical_arc_mixture_image_plane_controls() -> None:
    args = validation._build_parser().parse_args(
        [
            "--image-plane-mode",
            IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
            "--sampling-engine",
            "refreshing_surrogate",
            "--critical-arc-critical-direction-sigma-arcsec",
            "4.5",
            "--critical-arc-base-prob",
            "0.2",
            "--critical-arc-max-prob",
            "0.7",
            "--critical-arc-singular-threshold",
            "0.15",
            "--critical-arc-singular-softness",
            "0.04",
            "--critical-arc-lm-damping-relative",
            "0.002",
            "--critical-arc-lm-damping-absolute",
            "1e-5",
            "--critical-arc-lm-trust-radius-arcsec",
            "18.0",
            "--arc-aware-noncritical-support-radius-arcsec",
            "1.25",
            "--arc-aware-max-arclength-arcsec",
            "12.0",
            "--arc-aware-curve-step-arcsec",
            "0.2",
        ]
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert args.image_plane_mode == IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE
    assert args.critical_arc_critical_direction_sigma_arcsec == pytest.approx(4.5)
    assert args.critical_arc_base_prob == pytest.approx(0.2)
    assert args.critical_arc_max_prob == pytest.approx(0.7)
    assert args.critical_arc_singular_threshold == pytest.approx(0.15)
    assert args.critical_arc_singular_softness == pytest.approx(0.04)
    assert args.critical_arc_lm_damping_relative == pytest.approx(0.002)
    assert args.critical_arc_lm_damping_absolute == pytest.approx(1.0e-5)
    assert args.critical_arc_lm_trust_radius_arcsec == pytest.approx(18.0)
    assert args.arc_aware_noncritical_support_radius_arcsec == pytest.approx(1.25)
    assert args.arc_aware_max_arclength_arcsec == pytest.approx(12.0)
    assert args.arc_aware_curve_step_arcsec == pytest.approx(0.2)
    assert controls["stage4"].fit_method == "svi+nuts"


def test_validation_parser_accepts_fold_regularized_image_plane_controls() -> None:
    args = validation._build_parser().parse_args(
        [
            "--image-plane-mode",
            IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
            "--sampling-engine",
            "refreshing_surrogate",
            "--fold-curvature-arcsec-inv",
            "1.25",
            "--image-plane-newton-steps",
            "0",
        ]
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert args.image_plane_mode == IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA
    assert args.fold_curvature_arcsec_inv == pytest.approx(1.25)
    assert validation._validation_final_stage_name(args) == "stage4_fold_regularized_image_plane"
    assert controls["stage4"].fit_method == "svi+nuts"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--critical-arc-speed-mode", "all"),
        ("--critical-arc-mask-margin-softness", "4.0"),
    ],
)
def test_validation_parser_rejects_removed_critical_arc_speed_controls(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        validation._build_parser().parse_args(
            [
                "--image-plane-mode",
                IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
                flag,
                value,
            ]
        )


def test_run_xsh_critical_arc_mode_selects_new_image_plane_mode() -> None:
    text = (Path(__file__).resolve().parents[1] / "run.xsh").read_text(encoding="utf-8")

    assert 'mode = "critical_arc"' in text
    assert '"fold_regularized": "fold-regularized-forward-beta-image-plane"' in text
    assert '"--fold-curvature-arcsec-inv", 1.0' in text
    assert '"critical_arc": "critical-arc-mixture-image-plane"' in text
    assert 'sampling_engine = "refreshing_surrogate"' in text
    assert '"--critical-arc-critical-direction-sigma-arcsec", 10.0' in text
    assert '"--critical-arc-base-prob", 0.10' in text
    assert '"--critical-arc-max-prob", 0.85' in text
    assert '"--critical-arc-singular-threshold", 0.40' in text
    assert '"--critical-arc-singular-softness", 0.10' in text
    assert '"--critical-arc-lm-trust-radius-arcsec", 20.0' in text
    assert "--critical-arc-speed-mode" not in text
    assert '"--arc-aware-noncritical-support-radius-arcsec", 1.0' in text
    assert '"--arc-aware-max-arclength-arcsec", 10.0' in text
    assert '"--arc-aware-curve-step-arcsec", 0.1' in text
    assert '"--match-tolerance-arcsec", 2.0' in text
    assert '"kappa_true_fits": "data/ff_sims/ares/kappa_z9_0.fits"' in text
    assert '"kappa_true_fits": "data/ff_sims/hera/kappa_z9_0.fits"' in text
    assert '"image_catalog_family_cutout_image_dir": "data/ff_sims"' in text
    assert 'kappa_true_args = ["--kappa-true-fits", kappa_true_fits] if kappa_true_fits else []' in text
    assert '"--caustic-source-redshift", 9.0' in text
    assert "*(kappa_true_args)" in text


def test_validation_parser_rejects_removed_stage35_flag() -> None:
    with pytest.raises(SystemExit):
        validation._build_parser().parse_args(["--run-stage35-" "marginal-" "image-plane"])


def test_validation_stage_fit_controls_reject_four_values() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi", "svi+nuts"],
        warmup=[1000, 500, 0, 250],
        samples=[250, 100, 20, 80],
        max_tree_depth=[8, 7, 6, 5],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    )

    with pytest.raises(SystemExit, match="at most three"):
        _normalize_validation_stage_fit_controls(args)


def test_validation_parser_accepts_evidence_ns_controls() -> None:
    args = validation._build_parser().parse_args(
        [
            "--solver-fit-mode",
            "evidence-ns",
            "--evidence-source-prior-sigma-arcsec",
            "5.0",
        ]
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert args.solver_fit_mode == "evidence-ns"
    assert args.evidence_likelihood_mode == cluster_solver.DEFAULT_EVIDENCE_LIKELIHOOD_MODE
    assert args.evidence_source_prior_sigma_arcsec == pytest.approx(5.0)
    assert controls["stage2"].fit_method == "ns"
    assert controls["stage2"].samples == 0


def test_validation_parser_accepts_posterior_diagnostic_mode() -> None:
    defaults = validation._build_parser().parse_args([])
    args = validation._build_parser().parse_args(["--posterior-diagnostic-mode", "approximate", "--quick-diagnostics"])
    exact_args = validation._build_parser().parse_args(["--exact-image-diagnostics-stage3"])

    assert defaults.posterior_diagnostic_mode == validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT
    assert defaults.quick_diagnostics is False
    assert defaults.exact_image_diagnostics_stage3 is False
    assert args.posterior_diagnostic_mode == validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE
    assert args.quick_diagnostics is True
    assert exact_args.exact_image_diagnostics_stage3 is True


def test_validation_rejects_quick_and_exact_stage3_diagnostics() -> None:
    args = validation._build_parser().parse_args(["--quick-diagnostics", "--exact-image-diagnostics-stage3"])

    with pytest.raises(SystemExit, match="exact-image-diagnostics-stage3"):
        validation._validate_validation_args(args)


def test_validation_parser_accepts_recovery_profile_draws() -> None:
    defaults = validation._build_parser().parse_args([])
    args = validation._build_parser().parse_args(["--recovery-profile-draws", "32"])
    zero_args = validation._build_parser().parse_args(["--recovery-profile-draws", "0"])
    negative_args = validation._build_parser().parse_args(["--recovery-profile-draws", "-4"])

    assert defaults.recovery_profile_draws == validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP
    assert args.recovery_profile_draws == 32
    assert zero_args.recovery_profile_draws == 0
    assert negative_args.recovery_profile_draws == -4


def test_validation_parser_accepts_write_stage3_recovery() -> None:
    defaults = validation._build_parser().parse_args([])
    args = validation._build_parser().parse_args(["--write-stage3-recovery"])

    assert defaults.write_stage3_recovery is False
    assert args.write_stage3_recovery is True


def test_validation_parser_accepts_sampled_source_evidence_controls() -> None:
    args = validation._build_parser().parse_args(
        [
            "--solver-fit-mode",
            "evidence-ns",
            "--evidence-source-prior-sigma-arcsec",
            "5.0",
            "--evidence-likelihood-mode",
            SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
            "--source-position-parameterization",
            "direct",
            "--image-plane-newton-steps",
            "1",
            "--sampling-engine",
            "full",
        ]
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert args.evidence_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
    assert args.source_position_parameterization == "direct"
    assert args.image_plane_newton_steps == 1
    assert controls["stage2"].fit_method == "ns"


def test_validation_parser_rejects_marginal_evidence_likelihood_mode() -> None:
    with pytest.raises(SystemExit):
        validation._build_parser().parse_args(
            [
                "--solver-fit-mode",
                "evidence-ns",
                "--evidence-source-prior-sigma-arcsec",
                "5.0",
                "--evidence-likelihood-mode",
                "linearized-" "marginal-" "beta-image-plane",
            ]
        )


def test_validation_parser_accepts_active_subset_sampling_engine() -> None:
    args = validation._build_parser().parse_args(["--sampling-engine", "active_subset"])

    assert args.sampling_engine == "active_subset"


def test_validation_parser_uses_primary_and_subhalo_family_counts() -> None:
    parser = validation._build_parser()

    defaults = parser.parse_args([])
    validation._validate_validation_args(defaults)

    assert defaults.n_primary_families == 20
    assert defaults.n_subhalo_families == 0
    assert defaults.max_images_per_family is None
    assert defaults.primary_source_redshifts == "1.5,2.0,3.0"
    assert defaults.subhalo_source_redshifts == "1.5,2.0,3.0"
    assert defaults.critical_caustic_plot_grid_scale_arcsec == pytest.approx(0.2)
    assert defaults.primary_image_min_distance_arcsec == pytest.approx(
        mock_cluster.DEFAULT_PRIMARY_IMAGE_MIN_DISTANCE_ARCSEC
    )
    assert defaults.subhalo_image_min_distance_arcsec == pytest.approx(
        mock_cluster.DEFAULT_SUBHALO_IMAGE_MIN_DISTANCE_ARCSEC
    )

    args = parser.parse_args(
        [
            "--n-primary-families",
            "4",
            "--n-subhalo-families",
            "2",
            "--primary-source-redshifts",
            "2.0,3.0",
            "--subhalo-source-redshifts",
            "5.0",
            "--primary-image-min-distance-arcsec",
            "2.5",
            "--subhalo-image-min-distance-arcsec",
            "0.75",
            "--critical-caustic-plot-grid-scale-arcsec",
            "0.15",
        ]
    )
    validation._validate_validation_args(args)

    assert args.n_primary_families == 4
    assert args.n_subhalo_families == 2
    assert args.min_images_per_family == 3
    assert args.primary_image_min_distance_arcsec == pytest.approx(2.5)
    assert args.subhalo_image_min_distance_arcsec == pytest.approx(0.75)
    assert args.critical_caustic_plot_grid_scale_arcsec == pytest.approx(0.15)
    assert validation._parse_source_redshifts(args.primary_source_redshifts, fallback=2.0) == (2.0, 3.0)
    assert validation._parse_source_redshifts(args.subhalo_source_redshifts, fallback=2.0) == (5.0,)


def test_validation_parser_accepts_max_images_per_family() -> None:
    parser = validation._build_parser()

    capped = parser.parse_args(["--max-images-per-family", "5"])
    validation._validate_validation_args(capped)

    assert capped.max_images_per_family == 5

    unlimited = parser.parse_args(["--max-images-per-family", "none"])
    validation._validate_validation_args(unlimited)

    assert unlimited.max_images_per_family is None


def test_validation_parser_rejects_removed_n_families_flag() -> None:
    with pytest.raises(SystemExit):
        validation._build_parser().parse_args(["--n-families", "3"])


def test_validation_parser_rejects_removed_source_redshifts_flag() -> None:
    with pytest.raises(SystemExit):
        validation._build_parser().parse_args(["--source-redshifts", "1.5,2.0"])


def test_validation_parser_rejects_removed_posterior_diagnostic_workers_flag() -> None:
    with pytest.raises(SystemExit):
        validation._build_parser().parse_args(["--posterior-diagnostic-workers", "2"])


def test_validation_parser_rejects_invalid_family_counts() -> None:
    parser = validation._build_parser()
    with pytest.raises(SystemExit):
        validation._validate_validation_args(parser.parse_args(["--n-primary-families", "-1"]))
    with pytest.raises(SystemExit):
        validation._validate_validation_args(parser.parse_args(["--n-primary-families", "0", "--n-subhalo-families", "0"]))
    with pytest.raises(SystemExit):
        validation._validate_validation_args(parser.parse_args(["--min-images-per-family", "1"]))
    with pytest.raises(SystemExit):
        validation._validate_validation_args(
            parser.parse_args(["--min-images-per-family", "4", "--max-images-per-family", "3"])
        )
    with pytest.raises(SystemExit):
        validation._validate_validation_args(parser.parse_args(["--critical-caustic-plot-grid-scale-arcsec", "0"]))
    with pytest.raises(SystemExit):
        validation._validate_validation_args(parser.parse_args(["--primary-image-min-distance-arcsec", "0"]))
    with pytest.raises(SystemExit):
        validation._validate_validation_args(parser.parse_args(["--subhalo-image-min-distance-arcsec", "-1"]))
    with pytest.raises(SystemExit):
        validation._validate_validation_args(parser.parse_args(["--primary-image-min-distance-arcsec", "nan"]))

    validation._validate_validation_args(parser.parse_args(["--recovery-profile-draws", "0"]))
    validation._validate_validation_args(parser.parse_args(["--recovery-profile-draws", "-4"]))


def test_validation_parser_rejects_removed_ott_sinkhorn_image_plane_mode() -> None:
    with pytest.raises(SystemExit):
        validation._build_parser().parse_args(
            [
                "--image-plane-mode",
                "ott-sinkhorn-forward-beta-image-plane",
            ]
        )


def test_validation_parser_accepts_resume_flag() -> None:
    args = validation._build_parser().parse_args(["--resume"])

    assert args.resume is True


def test_validation_parser_accepts_resume_fast_flag() -> None:
    args = validation._build_parser().parse_args(["--resume-fast"])

    assert args.resume_fast is True


def test_validation_parser_accepts_jax_device_flags() -> None:
    args = validation._build_parser().parse_args(["--jax-default-device", "cpu", "--smc-device", "gpu"])

    assert args.jax_default_device == "cpu"
    assert args.smc_device == "gpu"


def test_cluster_solver_parser_accepts_resume_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "data/clustersim/input.par", "--resume"])

    args = _parse_args()

    assert args.resume is True


def test_cluster_solver_parser_accepts_resume_fast_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "data/clustersim/input.par", "--resume-fast"])

    args = _parse_args()

    assert args.resume_fast is True


def test_cluster_solver_parser_accepts_jax_device_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--jax-default-device",
            "cpu",
            "--smc-device",
            "gpu",
        ],
    )

    args = _parse_args()

    assert args.jax_default_device == "cpu"
    assert args.smc_device == "gpu"


def test_cluster_solver_rejects_resume_fast_outside_sequential() -> None:
    with pytest.raises(SystemExit, match="--resume-fast"):
        cluster_solver._normalize_stage_fit_controls(
            argparse.Namespace(fit_mode=cluster_solver.FIT_MODE_EVIDENCE_NS, resume_fast=True)
        )


def test_validation_rejects_resume_fast_outside_sequential() -> None:
    with pytest.raises(SystemExit, match="--resume-fast"):
        validation._normalize_validation_stage_fit_controls(
            argparse.Namespace(solver_fit_mode=validation.SOLVER_FIT_MODE_EVIDENCE_NS, resume_fast=True)
        )


def test_cluster_solver_parser_accepts_nested_sampling_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-method",
            "ns",
            "--samples",
            "25",
            "--ns-num-live-points",
            "200",
            "--ns-max-samples",
            "3000",
            "--ns-dlogz",
            "0.01",
        ],
    )

    args = _parse_args()

    assert args.fit_method == ["ns"]
    assert args.ns_num_live_points == 200
    assert args.ns_max_samples == 3000
    assert args.ns_dlogz == pytest.approx(0.01)


def test_cluster_solver_parser_accepts_unlimited_ns_max_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-method",
            "ns",
            "--samples",
            "25",
            "--ns-max-samples",
            "none",
        ],
    )

    args = _parse_args()

    assert args.ns_max_samples is None
    assert args.linearized_beta_prior_sigma_arcsec == pytest.approx(2.0)


def test_cluster_solver_parser_defaults_to_unlimited_ns_max_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
        ],
    )

    args = _parse_args()

    assert args.ns_max_samples is None


def test_validation_parser_accepts_nested_sampling_controls() -> None:
    args = validation._build_parser().parse_args(
        [
            "--fit-method",
            "ns",
            "--samples",
            "25",
            "--ns-num-live-points",
            "200",
            "--ns-max-samples",
            "3000",
            "--ns-dlogz",
            "0.01",
        ]
    )

    assert args.fit_method == ["ns"]
    assert args.ns_num_live_points == 200
    assert args.ns_max_samples == 3000
    assert args.ns_dlogz == pytest.approx(0.01)


def test_validation_parser_accepts_unlimited_ns_max_samples() -> None:
    args = validation._build_parser().parse_args(["--ns-max-samples", "None"])

    assert args.ns_max_samples is None


def test_validation_parser_defaults_to_unlimited_ns_max_samples() -> None:
    args = validation._build_parser().parse_args([])

    assert args.ns_max_samples is None


def test_validation_parser_rejects_invalid_ns_max_samples() -> None:
    with pytest.raises(SystemExit):
        validation._build_parser().parse_args(["--ns-max-samples", "forever"])


def test_validation_parser_rejects_fit_quality_workers() -> None:
    parser = validation._build_parser()

    defaults = parser.parse_args([])
    validation._validate_validation_args(defaults)

    assert not hasattr(defaults, "fit_quality_workers")

    with pytest.raises(SystemExit):
        parser.parse_args(["--fit-quality-workers", "32"])


def test_cluster_solver_parser_rejects_fit_quality_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-quality-workers",
            "32",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


def test_validation_parser_accepts_schechter_subhalo_controls_and_rejects_removed_model_flags() -> None:
    parser = validation._build_parser()

    defaults = parser.parse_args([])
    validation._validate_validation_args(defaults)

    assert defaults.subhalo_schechter_alpha == pytest.approx(-0.7)
    assert defaults.subhalo_parent_factor == 1000
    assert defaults.subhalo_mag_faint_limit == pytest.approx(24.0)
    assert defaults.bcg_position_prior_half_width_arcsec == pytest.approx(
        mock_cluster.DEFAULT_BCG_POSITION_PRIOR_HALF_WIDTH_ARCSEC
    )

    args = parser.parse_args(
        [
            "--subhalo-schechter-alpha",
            "-0.5",
            "--subhalo-parent-factor",
            "9",
            "--subhalo-mag-faint-limit",
            "23.5",
            "--subhalo-mass-min",
            "1e8",
            "--subhalo-mass-max",
            "1e12",
            "--subhalo-mass-ref",
            "5e11",
        ]
    )
    validation._validate_validation_args(args)

    assert args.subhalo_schechter_alpha == pytest.approx(-0.5)
    assert args.subhalo_parent_factor == 9
    assert args.subhalo_mag_faint_limit == pytest.approx(23.5)
    assert args.subhalo_mass_min == pytest.approx(1.0e8)
    assert args.subhalo_mass_max == pytest.approx(1.0e12)
    assert args.subhalo_mass_ref == pytest.approx(5.0e11)

    with pytest.raises(SystemExit):
        parser.parse_args(["--subhalo-population-model", "shmf"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--subhalo-shmf-alpha-ln", "0.8"])

    invalid_bcg_args = parser.parse_args(["--bcg-position-prior-half-width-arcsec", "0"])
    with pytest.raises(SystemExit):
        validation._validate_validation_args(invalid_bcg_args)


def _validation_solver_args(**updates) -> argparse.Namespace:
    payload = dict(
        solver_fit_mode="sequential",
        fit_method="svi+nuts",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        start_at_stage3=False,
        skip_stage3_image_plane_local_jacobian=False,
        image_plane_newton_steps=0,
        anchored_image_plane_solve_steps=validation.DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS,
        anchored_image_plane_trust_radius_arcsec=validation.DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
        anchored_image_plane_lm_damping_relative=validation.DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
        anchored_image_plane_lm_damping_absolute=validation.DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
        critical_arc_critical_direction_sigma_arcsec=validation.DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC,
        critical_arc_base_prob=validation.DEFAULT_CRITICAL_ARC_BASE_PROB,
        critical_arc_max_prob=validation.DEFAULT_CRITICAL_ARC_MAX_PROB,
        critical_arc_singular_threshold=validation.DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD,
        critical_arc_singular_softness=validation.DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS,
        critical_arc_lm_damping_relative=validation.DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE,
        critical_arc_lm_damping_absolute=validation.DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE,
        critical_arc_lm_trust_radius_arcsec=validation.DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC,
        fold_curvature_arcsec_inv=validation.DEFAULT_FOLD_CURVATURE_ARCSEC_INV,
        linearized_beta_prior_sigma_arcsec=validation.DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC,
        source_position_parameterization="prior-whitened",
        image_plane_scatter_upper_arcsec=2.0,
        image_plane_scatter_floor_arcsec=validation.DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC,
        image_plane_scatter_prior="log-uniform",
        image_plane_scatter_prior_median_arcsec=0.3,
        image_plane_scatter_prior_log_sigma=0.5,
        fix_image_sigma_int_arcsec=None,
        image_presence_penalty_weight=None,
        image_presence_match_radius_arcsec=0.30,
        image_presence_temperature_arcsec=0.10,
        image_presence_count_softness=0.05,
        image_presence_count_margin=0.05,
        likelihood_stabilizer_max_gain=0.0,
        likelihood_stabilizer_max_residual_arcsec=0.0,
        likelihood_stabilizer_residual_loss="gaussian",
        likelihood_stabilizer_student_t_nu=4.0,
        evidence_likelihood_mode=cluster_solver.DEFAULT_EVIDENCE_LIKELIHOOD_MODE,
        evidence_source_prior_sigma_arcsec=None,
        evidence_source_prior_mean_x_arcsec=0.0,
        evidence_source_prior_mean_y_arcsec=0.0,
        svi_steps=10,
        warmup=300,
        samples=500,
        chains=1,
        ns_num_live_points=None,
        ns_max_samples=None,
        ns_dlogz=1.0e-4,
        sampling_engine="refreshing_surrogate",
        source_plane_covariance_floor=1.0e-6,
        z_bin_efficiency_tol=0.01,
        fit_cosmology_flat_wcdm=False,
        active_scaling_selection="adaptive",
        active_scaling_cumulative_fraction=0.995,
        active_scaling_min=4,
        active_scaling_galaxies=None,
        fit_scaling_scatter=False,
        n_primary_families=1,
        n_subhalo_families=0,
        min_images_per_family=3,
        max_images_per_family=None,
        primary_image_min_distance_arcsec=mock_cluster.DEFAULT_PRIMARY_IMAGE_MIN_DISTANCE_ARCSEC,
        subhalo_image_min_distance_arcsec=mock_cluster.DEFAULT_SUBHALO_IMAGE_MIN_DISTANCE_ARCSEC,
        bcg_position_prior_half_width_arcsec=mock_cluster.DEFAULT_BCG_POSITION_PRIOR_HALF_WIDTH_ARCSEC,
        caustic_compute_window_arcsec=160.0,
        caustic_grid_scale_arcsec=0.2,
        critical_caustic_plot_grid_scale_arcsec=0.2,
        caustic_min_area_arcsec2=1.0e-5,
        caustic_boundary_margin_arcsec=0.5,
        n_subhalos=0,
        subhalo_schechter_alpha=-0.7,
        subhalo_parent_factor=1000,
        subhalo_mag_faint_limit=24.0,
        subhalo_mass_min=1.0e9,
        subhalo_mass_max=1.0e13,
        subhalo_mass_ref=1.0e12,
        subhalo_sigma_scatter_dex=0.07,
        subhalo_cut_scatter_dex=0.20,
        scaling_scatter_max=0.5,
        pos_sigma_arcsec=0.15,
        seed=12345,
        target_accept=0.85,
        dense_mass=True,
        jax_default_device="auto",
        smc_device="auto",
        max_tree_depth=8,
        skip_plots=True,
        quick_diagnostics=False,
        exact_image_diagnostics_stage3=False,
        quiet=False,
        resume=False,
        resume_fast=False,
        recovery_profile_draws=validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
        write_stage3_recovery=False,
    )
    payload.update(updates)
    return argparse.Namespace(**payload)


def _validation_run_args(tmp_path: Path, **updates) -> argparse.Namespace:
    payload = vars(_validation_solver_args()).copy()
    payload.update(
        dict(
            mock="single-bcg",
            output_dir=str(tmp_path),
            run_name="validation_log",
            realizations=1,
            n_primary_families=1,
            n_subhalo_families=0,
            min_images_per_family=3,
            max_images_per_family=None,
            source_redshift=2.0,
            primary_source_redshifts="1.5",
            subhalo_source_redshifts="1.5",
            source_sigma_int_arcsec=0.05,
            posterior_diagnostic_draws=2,
            posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        )
    )
    payload.update(updates)
    return argparse.Namespace(**payload)


def _option_values(cmd: list[str], option: str) -> list[str]:
    start = cmd.index(option) + 1
    values: list[str] = []
    for value in cmd[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def test_validation_parser_parses_dense_mass_boolean_optional() -> None:
    parser = validation._build_parser()

    default_args = parser.parse_args([])
    assert default_args.dense_mass is True

    diagonal_args = parser.parse_args(["--no-dense-mass"])
    assert diagonal_args.dense_mass is False

    dense_args = parser.parse_args(["--dense-mass"])
    assert dense_args.dense_mass is True


def test_validation_parser_accepts_potfile_mass_size_reparam_flag() -> None:
    parser = validation._build_parser()

    default_args = parser.parse_args([])
    assert default_args.potfile_mass_size_reparam is False

    reparam_args = parser.parse_args(["--potfile-mass-size-reparam"])
    validation._validate_validation_args(reparam_args)
    assert reparam_args.potfile_mass_size_reparam is True


def test_validation_parser_accepts_and_validates_fixed_image_sigma_int() -> None:
    parser = validation._build_parser()

    default_args = parser.parse_args([])
    validation._validate_validation_args(default_args)
    assert default_args.fix_image_sigma_int_arcsec is None

    fixed_args = parser.parse_args(["--fix-image-sigma-int-arcsec", "0.35"])
    validation._validate_validation_args(fixed_args)
    assert fixed_args.fix_image_sigma_int_arcsec == pytest.approx(0.35)

    zero_args = parser.parse_args(["--fix-image-sigma-int-arcsec", "0.0"])
    validation._validate_validation_args(zero_args)
    assert zero_args.fix_image_sigma_int_arcsec == pytest.approx(0.0)

    for invalid_value in ("-0.1", "nan"):
        invalid_args = parser.parse_args(["--fix-image-sigma-int-arcsec", invalid_value])
        with pytest.raises(SystemExit, match="fix-image-sigma-int"):
            validation._validate_validation_args(invalid_args)


@pytest.mark.parametrize(
    "updates",
    [
        {"solver_fit_mode": "evidence-ns", "image_plane_mode": IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA},
        {"image_plane_mode": IMAGE_PLANE_MODE_NONE},
        {"image_plane_mode": IMAGE_PLANE_MODE_LOCAL_JACOBIAN},
        {
            "image_plane_mode": IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
            "skip_stage3_image_plane_local_jacobian": True,
        },
    ],
)
def test_validate_validation_args_rejects_invalid_stage3_recovery_modes(
    tmp_path: Path,
    updates: dict[str, Any],
) -> None:
    args = _validation_run_args(tmp_path, write_stage3_recovery=True, **updates)

    with pytest.raises(SystemExit):
        validation._validate_validation_args(args)


def test_validation_jsonable_sanitizes_structured_values(tmp_path: Path) -> None:
    converted = validation._validation_jsonable(
        {
            "path": tmp_path / "x",
            "array": np.asarray([1.0, np.nan, np.inf]),
            "scalar": np.float64(np.nan),
            "df": pd.DataFrame({"a": [1.0, np.nan], "b": ["x", "y"]}),
            "component": validation.DPIETruth(
                potential_id="halo",
                x_centre=0.0,
                y_centre=0.0,
                ellipticite=0.2,
                angle_pos=35.0,
                core_radius_arcsec=5.0,
                cut_radius_arcsec=220.0,
                v_disp=760.0,
            ),
        }
    )

    assert converted["path"] == str(tmp_path / "x")
    assert converted["array"] == [1.0, None, None]
    assert converted["scalar"] is None
    assert converted["df"]["records"][1]["a"] is None
    assert converted["component"]["potential_id"] == "halo"
    json.dumps(converted, allow_nan=False)


def test_write_validation_results_json_embeds_stage_table_artifacts(tmp_path: Path) -> None:
    args = _validation_run_args(tmp_path)
    root = tmp_path / "single_bcg" / "validation_log"
    realization_dir = root / "seed_12345"
    mock_dir = realization_dir / "mock"
    mock_dir.mkdir(parents=True)
    paths = validation.MockClusterPaths(
        root=mock_dir,
        par_path=mock_dir / "single_bcg_mock.par",
        image_catalog_path=mock_dir / "obs_arcs.cat",
        truth_path=mock_dir / "truth.json",
        mock_images_path=mock_dir / "mock_images.json",
    )
    truth = {
        "parameter_truth": {"halo.v_disp": 760.0},
        "sources": [{"family_id": "1", "beta_x": 0.1, "beta_y": -0.2, "z_source": 1.5}],
        "subhalos": [],
    }
    paths.par_path.write_text("mock par", encoding="utf-8")
    paths.image_catalog_path.write_text("mock catalog", encoding="utf-8")
    paths.truth_path.write_text(json.dumps(truth), encoding="utf-8")
    paths.mock_images_path.write_text(json.dumps([{"family_id": "1", "image_label": "1.1"}]), encoding="utf-8")
    (mock_dir / "members.cat").write_text("member catalog", encoding="utf-8")
    images = pd.DataFrame([{"family_id": "1", "image_label": "1.1"}])

    solver_run_dir = realization_dir / "solver" / "fit" / "stage2_joint"
    tables_dir = solver_run_dir / "tables"
    tables_dir.mkdir(parents=True)
    (tables_dir / "run_summary.json").write_text(json.dumps({"fit_method": "svi", "n_images": 1}), encoding="utf-8")
    (tables_dir / "family_diagnostics.csv").write_text(
        "family_id,exact_image_rms_arcsec\n1,0.03\n",
        encoding="utf-8",
    )
    (tables_dir / "notes.txt").write_text("table notes", encoding="utf-8")
    summary_path = realization_dir / "run_summary.txt"
    summary_path.write_text("summary text", encoding="utf-8")
    (root / "run_debug.log").write_text("debug text", encoding="utf-8")

    path = validation.write_validation_results_json(
        args=args,
        seed=12345,
        realization_dir=realization_dir,
        config=validation.SingleBCGMockConfig(seed=12345, n_primary_families=1),
        paths=paths,
        images=images,
        truth_payload=truth,
        solver_run_dir=solver_run_dir,
        summary_path=summary_path,
        output_paths={"summary_plot": realization_dir / "validation_summary.pdf"},
        recovery_payload={
            "summary": {"median_abs_parameter_bias": np.nan},
            "tables": {"parameters": {"records": [{"parameter": "halo.v_disp", "bias": np.inf}]}},
        },
        controls=validation._normalize_validation_stage_fit_controls(args),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    stage = payload["solver"]["stage_manifests"][0]
    assert path == root / "seed_12345_results.json"
    assert payload["mock_cluster"]["files"]["member_catalog_text"] == "member catalog"
    assert payload["mock_cluster"]["truth"]["sources"][0]["family_id"] == "1"
    assert payload["validation"]["run_summary"]["text"] == "summary text"
    assert payload["validation"]["output_paths"]["results_json"] == str(path)
    assert payload["validation"]["recovery"]["final"]["summary"]["median_abs_parameter_bias"] is None
    assert payload["validation"]["recovery"]["final"]["tables"]["parameters"]["records"][0]["bias"] is None
    assert payload["debug_log"]["text"] == "debug text"
    assert stage["stage"] == "stage2_joint"
    assert stage["table_artifacts"]["run_summary.json"]["data"]["fit_method"] == "svi"
    assert stage["table_artifacts"]["family_diagnostics.csv"]["data"]["records"][0]["family_id"] == 1
    assert stage["table_artifacts"]["notes.txt"]["text"] == "table notes"


def test_validation_run_cluster_solver_forwards_scalar_stage_controls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        assert check is True
        assert Path(cwd).exists()
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args()

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)
    cmd = captured["cmd"]

    assert _option_values(cmd, "--fit-method") == ["svi+nuts"]
    assert _option_values(cmd, "--svi-steps") == ["10"]
    assert _option_values(cmd, "--warmup") == ["300"]
    assert _option_values(cmd, "--samples") == ["500"]
    assert _option_values(cmd, "--max-tree-depth") == ["8"]
    assert "--dense-mass" in cmd
    assert "--no-dense-mass" not in cmd
    assert "--potfile-mass-size-reparam" not in cmd
    assert "--ns-num-live-points" not in cmd
    assert "--ns-max-samples" not in cmd
    assert "--ns-dlogz" not in cmd
    assert _option_values(cmd, "--image-plane-mode") == [IMAGE_PLANE_MODE_NONE]
    assert "--linearized-image-plane-stage" not in cmd
    assert "--start-at-stage3" not in cmd
    assert "--skip-stage3-image-plane-local-jacobian" not in cmd
    assert "--ott-sinkhorn-epsilon" not in cmd
    assert "--ott-sinkhorn-max-iterations" not in cmd
    assert "--ott-sinkhorn-threshold" not in cmd
    assert "--ott-sinkhorn-lse-mode" not in cmd
    assert _option_values(cmd, "--image-plane-newton-steps") == ["0"]
    assert _option_values(cmd, "--linearized-beta-prior-sigma-arcsec") == ["2.0"]
    assert _option_values(cmd, "--source-position-parameterization") == ["prior-whitened"]
    assert _option_values(cmd, "--image-plane-scatter-upper-arcsec") == ["2.0"]
    assert _option_values(cmd, "--image-plane-scatter-floor-arcsec") == ["0.001"]
    assert _option_values(cmd, "--image-plane-scatter-prior") == ["log-uniform"]
    assert _option_values(cmd, "--image-plane-scatter-prior-median-arcsec") == ["0.3"]
    assert _option_values(cmd, "--image-plane-scatter-prior-log-sigma") == ["0.5"]
    assert "--fix-image-sigma-int-arcsec" not in cmd
    assert "--fit-quality-workers" not in cmd
    assert run_dir == tmp_path / "solver" / "fit" / "stage2_joint"


def test_validation_run_cluster_solver_forwards_fixed_image_sigma_int(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        assert check is True
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(fix_image_sigma_int_arcsec=0.35)

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert _option_values(captured["cmd"], "--fix-image-sigma-int-arcsec") == ["0.35"]


def test_validation_run_cluster_solver_forwards_no_dense_mass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        assert check is True
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(dense_mass=False)

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert "--no-dense-mass" in captured["cmd"]
    assert "--dense-mass" not in captured["cmd"]


def test_validation_run_cluster_solver_forwards_potfile_mass_size_reparam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        assert check is True
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(potfile_mass_size_reparam=True)

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert "--potfile-mass-size-reparam" in captured["cmd"]


def test_validation_run_cluster_solver_does_not_forward_fit_quality_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args()

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert "--fit-quality-workers" not in captured["cmd"]
    assert "--posterior-diagnostic-workers" not in captured["cmd"]


def test_validation_run_cluster_solver_rejects_nested_sampling_in_sequential_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(
        fit_method="ns",
        warmup=0,
        samples=25,
        ns_num_live_points=200,
        ns_max_samples=3000,
        ns_dlogz=0.01,
    )

    with pytest.raises(SystemExit, match="evidence-ns"):
        validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)
    assert captured == {}


def test_validation_run_cluster_solver_forwards_evidence_ns_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(
        solver_fit_mode="evidence-ns",
        fit_method="svi+nuts",
        warmup=300,
        samples=500,
        ns_num_live_points=200,
        ns_max_samples=3000,
        ns_dlogz=0.01,
        evidence_source_prior_sigma_arcsec=5.0,
        evidence_source_prior_mean_x_arcsec=0.2,
        evidence_source_prior_mean_y_arcsec=-0.1,
    )

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)
    cmd = captured["cmd"]

    assert _option_values(cmd, "--fit-mode") == ["evidence-ns"]
    assert "--fit-method" not in cmd
    assert "--warmup" not in cmd
    assert "--samples" not in cmd
    assert _option_values(cmd, "--ns-num-live-points") == ["200"]
    assert _option_values(cmd, "--ns-max-samples") == ["3000"]
    assert _option_values(cmd, "--ns-dlogz") == ["0.01"]
    assert _option_values(cmd, "--evidence-likelihood-mode") == [cluster_solver.DEFAULT_EVIDENCE_LIKELIHOOD_MODE]
    assert _option_values(cmd, "--evidence-source-prior-sigma-arcsec") == ["5.0"]
    assert _option_values(cmd, "--evidence-source-prior-mean-x-arcsec") == ["0.2"]
    assert _option_values(cmd, "--evidence-source-prior-mean-y-arcsec") == ["-0.1"]
    assert "--linearized-beta-prior-sigma-arcsec" not in cmd
    assert _option_values(cmd, "--source-position-parameterization") == ["prior-whitened"]
    assert _option_values(cmd, "--image-plane-newton-steps") == ["0"]
    assert run_dir == tmp_path / "solver" / "fit"


def test_validation_run_cluster_solver_forwards_sampled_source_evidence_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(
        solver_fit_mode="evidence-ns",
        ns_num_live_points=200,
        ns_max_samples=3000,
        ns_dlogz=0.01,
        evidence_source_prior_sigma_arcsec=5.0,
        evidence_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        source_position_parameterization="direct",
        image_plane_newton_steps=2,
        sampling_engine="full",
    )

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)
    cmd = captured["cmd"]

    assert _option_values(cmd, "--fit-mode") == ["evidence-ns"]
    assert _option_values(cmd, "--evidence-likelihood-mode") == [
        SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
    ]
    assert _option_values(cmd, "--image-plane-newton-steps") == ["2"]
    assert _option_values(cmd, "--source-position-parameterization") == ["direct"]
    assert "--linearized-beta-prior-sigma-arcsec" not in cmd
    assert run_dir == tmp_path / "solver" / "fit"


def test_validation_run_cluster_solver_forwards_unlimited_ns_max_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(
        solver_fit_mode="evidence-ns",
        evidence_source_prior_sigma_arcsec=5.0,
        ns_max_samples=None,
    )

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert _option_values(captured["cmd"], "--ns-max-samples") == ["none"]


def test_validation_run_cluster_solver_writes_debug_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(cmd, cwd, check):
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args()
    validation._configure_debug_log(args, "solver_log", tmp_path)
    try:
        run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)
    finally:
        validation._close_debug_log()

    log_text = (tmp_path / "run_debug.log").read_text(encoding="utf-8")
    assert "VALIDATION SOLVER" in log_text
    assert "launching solver" in log_text
    assert "solver complete" in log_text
    assert str(run_dir) in log_text


def test_validation_run_cluster_solver_logs_grouped_configured_approximation_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_logs: list[str] = []

    def fake_run(cmd, cwd, check):
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    monkeypatch.setattr(validation, "_log", lambda _args, message: captured_logs.append(str(message)))
    args = _validation_solver_args(
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        source_position_parameterization="prior-whitened",
        active_scaling_galaxies=[5],
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
    )

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    warnings = [message for message in captured_logs if "warning approximations active" in message]
    assert len(warnings) == 1
    warning = warnings[0]
    assert "refreshing_surrogate=configured" in warning
    assert "z_bins=configured" in warning
    assert "image_plane_mode=linearized-forward-beta-image-plane" in warning
    assert "source_position_parameterization=prior-whitened" in warning
    assert "active_scaling_selection=adaptive" in warning
    assert "active_scaling_galaxies=finite counts [5]" in warning
    assert "posterior_diagnostic_mode=approximate" in warning


def test_validation_run_cluster_solver_exact_full_configuration_logs_no_approximation_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_logs: list[str] = []

    def fake_run(cmd, cwd, check):
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    monkeypatch.setattr(validation, "_log", lambda _args, message: captured_logs.append(str(message)))
    args = _validation_solver_args(
        sampling_engine="full",
        z_bin_efficiency_tol=0.0,
        active_scaling_selection="fixed",
        active_scaling_galaxies=None,
        source_position_parameterization="direct",
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
    )

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert not any("warning approximations active" in message for message in captured_logs)


def test_validation_run_cluster_solver_forwards_two_value_stage_controls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(
        fit_method=["svi+nuts", "svi"],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        svi_steps=[1000, 500],
        warmup=[1000, 0],
        samples=[250, 100],
        max_tree_depth=[10, 8],
    )

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)
    cmd = captured["cmd"]

    assert _option_values(cmd, "--fit-method") == ["svi+nuts", "svi"]
    assert _option_values(cmd, "--svi-steps") == ["1000", "500"]
    assert _option_values(cmd, "--warmup") == ["1000", "0"]
    assert _option_values(cmd, "--samples") == ["250", "100"]
    assert _option_values(cmd, "--max-tree-depth") == ["10", "8"]
    assert _option_values(cmd, "--image-plane-mode") == [IMAGE_PLANE_MODE_LOCAL_JACOBIAN]
    assert run_dir == tmp_path / "solver" / "fit" / "stage3_image_plane"


def test_validation_run_cluster_solver_forwards_three_value_stage_controls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        image_plane_newton_steps=1,
        svi_steps=[1000, 500, 100],
        warmup=[1000, 500, 0],
        samples=[250, 100, 20],
        max_tree_depth=[10, 8, 6],
        sampling_engine="full",
        image_plane_scatter_floor_arcsec=0.05,
        image_plane_scatter_prior="lognormal",
        image_plane_scatter_prior_median_arcsec=0.25,
        image_plane_scatter_prior_log_sigma=0.4,
        image_presence_penalty_weight=3.0,
        image_presence_match_radius_arcsec=0.4,
        image_presence_temperature_arcsec=0.08,
        image_presence_count_softness=0.03,
        image_presence_count_margin=0.02,
        likelihood_stabilizer_max_gain=50.0,
        likelihood_stabilizer_max_residual_arcsec=3.0,
        likelihood_stabilizer_residual_loss="student-t",
        likelihood_stabilizer_student_t_nu=4.0,
    )

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)
    cmd = captured["cmd"]

    assert _option_values(cmd, "--fit-method") == ["svi+nuts", "svi+nuts", "svi"]
    assert _option_values(cmd, "--svi-steps") == ["1000", "500", "100"]
    assert _option_values(cmd, "--warmup") == ["1000", "500", "0"]
    assert _option_values(cmd, "--samples") == ["250", "100", "20"]
    assert _option_values(cmd, "--max-tree-depth") == ["10", "8", "6"]
    assert _option_values(cmd, "--image-plane-mode") == [IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA]
    assert "--linearized-image-plane-stage" not in cmd
    assert "--ott-sinkhorn-epsilon" not in cmd
    assert "--ott-sinkhorn-max-iterations" not in cmd
    assert "--ott-sinkhorn-threshold" not in cmd
    assert "--ott-sinkhorn-lse-mode" not in cmd
    assert _option_values(cmd, "--image-plane-newton-steps") == ["1"]
    assert _option_values(cmd, "--source-position-parameterization") == ["prior-whitened"]
    assert _option_values(cmd, "--image-plane-scatter-floor-arcsec") == ["0.05"]
    assert _option_values(cmd, "--image-plane-scatter-prior") == ["lognormal"]
    assert _option_values(cmd, "--image-plane-scatter-prior-median-arcsec") == ["0.25"]
    assert _option_values(cmd, "--image-plane-scatter-prior-log-sigma") == ["0.4"]
    assert _option_values(cmd, "--image-presence-penalty-weight") == ["3.0"]
    assert _option_values(cmd, "--image-presence-match-radius-arcsec") == ["0.4"]
    assert _option_values(cmd, "--image-presence-temperature-arcsec") == ["0.08"]
    assert _option_values(cmd, "--image-presence-count-softness") == ["0.03"]
    assert _option_values(cmd, "--image-presence-count-margin") == ["0.02"]
    assert _option_values(cmd, "--likelihood-stabilizer-max-gain") == ["50.0"]
    assert _option_values(cmd, "--likelihood-stabilizer-max-residual-arcsec") == ["3.0"]
    assert _option_values(cmd, "--likelihood-stabilizer-residual-loss") == ["student-t"]
    assert _option_values(cmd, "--likelihood-stabilizer-student-t-nu") == ["4.0"]
    assert run_dir == tmp_path / "solver" / "fit" / "stage4_linearized_image_plane"


def test_validation_run_cluster_solver_forwards_stage3_skip_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(
        fit_method=["svi+nuts", "svi"],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=True,
        svi_steps=[1000, 100],
        warmup=[1000, 0],
        samples=[250, 20],
    )

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)
    cmd = captured["cmd"]

    assert "--skip-stage3-image-plane-local-jacobian" in cmd
    assert _option_values(cmd, "--fit-method") == ["svi+nuts", "svi"]
    assert _option_values(cmd, "--svi-steps") == ["1000", "100"]
    assert run_dir == tmp_path / "solver" / "fit" / "stage4_linearized_image_plane"


def test_validation_run_cluster_solver_forwards_start_at_stage3_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(
        fit_method=["svi+nuts", "svi"],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        start_at_stage3=True,
        svi_steps=[1000, 100],
        warmup=[1000, 0],
        samples=[250, 20],
    )

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)
    cmd = captured["cmd"]

    assert "--start-at-stage3" in cmd
    assert "--skip-stage3-image-plane-local-jacobian" not in cmd
    assert _option_values(cmd, "--fit-method") == ["svi+nuts", "svi"]
    assert _option_values(cmd, "--svi-steps") == ["1000", "100"]
    assert run_dir == tmp_path / "solver" / "fit" / "stage3_image_plane"


def test_validation_run_cluster_solver_forwards_resume_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(resume=True)

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert "--resume" in captured["cmd"]


def test_validation_run_cluster_solver_forwards_resume_fast_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(resume_fast=True)

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert "--resume-fast" in captured["cmd"]


def test_validation_run_cluster_solver_forwards_jax_device_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(jax_default_device="cpu", smc_device="gpu")

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert captured["cmd"][captured["cmd"].index("--jax-default-device") + 1] == "cpu"
    assert captured["cmd"][captured["cmd"].index("--smc-device") + 1] == "gpu"


def test_validation_run_cluster_solver_forwards_active_subset_sampling_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(sampling_engine="active_subset")

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert captured["cmd"][captured["cmd"].index("--sampling-engine") + 1] == "active_subset"


def test_validation_run_cluster_solver_forwards_fit_cosmology_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(fit_cosmology_flat_wcdm=True)

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert "--fit-cosmology-flat-wcdm" in captured["cmd"]
    assert "--fit-cosmology-all-stages" not in captured["cmd"]


def test_validation_run_cluster_solver_forwards_forward_metric_image_plane_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(
        image_plane_mode=IMAGE_PLANE_MODE_FORWARD_METRIC,
    )

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert _option_values(captured["cmd"], "--image-plane-mode") == [IMAGE_PLANE_MODE_FORWARD_METRIC]
    assert run_dir == tmp_path / "solver" / "fit" / "stage4_forward_metric_image_plane"


def test_validation_run_cluster_solver_forwards_anchored_solved_image_plane_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(
        image_plane_mode=IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
        sampling_engine="refreshing_surrogate",
        anchored_image_plane_solve_steps=0,
        anchored_image_plane_trust_radius_arcsec=0.25,
        anchored_image_plane_lm_damping_relative=0.002,
        anchored_image_plane_lm_damping_absolute=1.0e-5,
    )

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert _option_values(captured["cmd"], "--image-plane-mode") == [IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA]
    assert _option_values(captured["cmd"], "--anchored-image-plane-solve-steps") == ["0"]
    assert _option_values(captured["cmd"], "--anchored-image-plane-trust-radius-arcsec") == ["0.25"]
    assert _option_values(captured["cmd"], "--anchored-image-plane-lm-damping-relative") == ["0.002"]
    assert _option_values(captured["cmd"], "--anchored-image-plane-lm-damping-absolute") == ["1e-05"]
    assert run_dir == tmp_path / "solver" / "fit" / "stage4_anchored_solved_image_plane"


def test_validation_run_cluster_solver_forwards_critical_arc_mixture_image_plane_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(
        image_plane_mode=IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
        sampling_engine="refreshing_surrogate",
        critical_arc_critical_direction_sigma_arcsec=4.5,
        critical_arc_base_prob=0.2,
        critical_arc_max_prob=0.7,
        critical_arc_singular_threshold=0.15,
        critical_arc_singular_softness=0.04,
        critical_arc_lm_damping_relative=0.002,
        critical_arc_lm_damping_absolute=1.0e-5,
        critical_arc_lm_trust_radius_arcsec=18.0,
        arc_aware_noncritical_support_radius_arcsec=1.25,
        arc_aware_max_arclength_arcsec=12.0,
        arc_aware_curve_step_arcsec=0.2,
    )

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert _option_values(captured["cmd"], "--image-plane-mode") == [IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE]
    assert _option_values(captured["cmd"], "--critical-arc-critical-direction-sigma-arcsec") == ["4.5"]
    assert _option_values(captured["cmd"], "--critical-arc-base-prob") == ["0.2"]
    assert _option_values(captured["cmd"], "--critical-arc-max-prob") == ["0.7"]
    assert _option_values(captured["cmd"], "--critical-arc-singular-threshold") == ["0.15"]
    assert _option_values(captured["cmd"], "--critical-arc-singular-softness") == ["0.04"]
    assert _option_values(captured["cmd"], "--critical-arc-lm-damping-relative") == ["0.002"]
    assert _option_values(captured["cmd"], "--critical-arc-lm-damping-absolute") == ["1e-05"]
    assert _option_values(captured["cmd"], "--critical-arc-lm-trust-radius-arcsec") == ["18.0"]
    assert "--critical-arc-speed-mode" not in captured["cmd"]
    assert "--critical-arc-mask-margin-softness" not in captured["cmd"]
    assert _option_values(captured["cmd"], "--arc-aware-noncritical-support-radius-arcsec") == ["1.25"]
    assert _option_values(captured["cmd"], "--arc-aware-max-arclength-arcsec") == ["12.0"]
    assert _option_values(captured["cmd"], "--arc-aware-curve-step-arcsec") == ["0.2"]
    assert run_dir == tmp_path / "solver" / "fit" / "stage4_critical_arc_mixture_image_plane"


def test_validation_run_cluster_solver_forwards_fold_regularized_image_plane_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(
        image_plane_mode=IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
        sampling_engine="refreshing_surrogate",
        fold_curvature_arcsec_inv=1.25,
    )

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert _option_values(captured["cmd"], "--image-plane-mode") == [IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA]
    assert _option_values(captured["cmd"], "--fold-curvature-arcsec-inv") == ["1.25"]
    assert run_dir == tmp_path / "solver" / "fit" / "stage4_fold_regularized_image_plane"


def test_validation_run_cluster_solver_forwards_quick_diagnostics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(quick_diagnostics=True)

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert "--quick-diagnostics" in captured["cmd"]


def test_validation_run_cluster_solver_forwards_exact_image_diagnostics_stage3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(exact_image_diagnostics_stage3=True)

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert "--exact-image-diagnostics-stage3" in captured["cmd"]
    assert "--quick-diagnostics" not in captured["cmd"]


def _touch_complete_stage(stage_dir: Path) -> None:
    (stage_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (stage_dir / "artifacts" / "plot_bundle.h5").write_bytes(b"")
    (stage_dir / "tables").mkdir(parents=True, exist_ok=True)
    (stage_dir / "tables" / "run_summary.json").write_text("{}", encoding="utf-8")


def _touch_stage1_summary(stage_dir: Path, map_values: dict[str, float] | None = None) -> None:
    (stage_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    payload = {"map_values": map_values or {}, "means": {}, "stds": {}}
    (stage_dir / "artifacts" / "stage1_prior_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _touch_artifact_checkpoint(stage_dir: Path) -> None:
    (stage_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (stage_dir / "artifacts" / "plot_bundle.h5").write_bytes(b"")


def _write_fake_stage_tables(
    stage_dir: Path,
    *,
    fit_method: str,
    likelihood: str,
    sampler: str,
    runtime_sec: float,
    best_loglike: float,
    exact_values: list[float],
    source_values: list[float],
    approx_values: list[float],
) -> None:
    tables_dir = stage_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "fit_method": fit_method,
                "sample_likelihood_mode": likelihood,
                "sampler": sampler,
                "runtime_sec": runtime_sec,
                "best_loglike": best_loglike,
                "accept_prob_mean": 0.91,
                "divergence_count": 2,
                "mean_num_steps": 17.5,
                "n_families": len(exact_values),
                "n_images": 12,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "family_id": [str(index + 1) for index in range(len(exact_values))],
            "exact_image_rms_arcsec": exact_values,
            "source_plane_rms_arcsec": source_values,
            "approx_image_rms_arcsec": approx_values,
        }
    ).to_csv(tables_dir / "family_diagnostics.csv", index=False)


def test_run_inference_saves_artifacts_before_validation_crash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    phases: list[str] = []
    saved_best_fits: list[np.ndarray] = []

    class FakeEvaluator:
        surrogate_enabled = False

        def __init__(self) -> None:
            self.timing_totals = {"validation_runtime": 0.0}

        def refresh_scaling_scatter_cache(self, _params, reason: str) -> None:
            return None

        def refresh_source_metric_cache(self, _params, reason: str) -> None:
            return None

        def _source_loglike_fn(self, params):
            params_array = jnp.asarray(params, dtype=jnp.float64)
            return -jnp.square(params_array[0])

        def evaluate(self, _params, validate_all_families: bool = False):
            raise RuntimeError("validation boom")

    posterior = PosteriorResults(
        samples=np.asarray([[0.0], [1.0]], dtype=float),
        log_prob=np.asarray([0.0, 10.0], dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=2,
        num_chains=0,
        init_diagnostics={},
        sampler="numpyro_jaxns",
    )
    evaluator = FakeEvaluator()

    def fake_logged_phase(_args, phase_name, fn, **_kwargs):
        phases.append(phase_name)
        return fn()

    def fake_save_artifacts(_artifacts_dir, _state, _args, best_fit, _posterior_for_output):
        saved_best_fits.append(np.asarray(best_fit, dtype=float))

    monkeypatch.setattr(cluster_solver, "_configure_debug_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(cluster_solver, "_prepare_direct_evaluator", lambda _args, _state: (evaluator, np.asarray([0.0], dtype=float)))
    monkeypatch.setattr(cluster_solver, "_posterior_model", lambda _specs, _evaluator: object())
    monkeypatch.setattr(cluster_solver, "_reference_theta_from_init_values", lambda _specs, _init_values, midpoint: np.asarray(midpoint, dtype=float))
    monkeypatch.setattr(cluster_solver, "_run_numpyro_nested_sampler", lambda *_args, **_kwargs: posterior)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_reported_physical", lambda theta, _specs, _evaluator: np.asarray(theta, dtype=float))
    monkeypatch.setattr(cluster_solver, "_posterior_results_to_reported_physical", lambda result, _specs, _evaluator: result)
    monkeypatch.setattr(cluster_solver, "_save_artifacts", fake_save_artifacts)

    args = argparse.Namespace(
        fit_method=cluster_solver.FIT_METHOD_NS,
        skip_validation=False,
        skip_plots=True,
    )
    state = SimpleNamespace(
        run_name="fit",
        par_path="input.par",
        parameter_specs=[SimpleNamespace(sample_name="p")],
        family_data=[],
        bin_data=[],
        svi_init_values=None,
        fit_mode="evidence-ns",
    )

    with pytest.raises(RuntimeError, match="validation boom"):
        cluster_solver._run_inference(args, state, tmp_path / "fit")

    assert saved_best_fits and saved_best_fits[0].tolist() == pytest.approx([0.0])
    assert phases.index("output.save_artifacts") < phases.index("validation.source_summary")


def test_active_subset_output_evaluator_switches_to_full_for_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FitEvaluator:
        sampling_engine = "active_subset"
        active_scaling_component_indices = np.asarray([1, 2], dtype=np.int32)
        inactive_scaling_component_indices = np.asarray([3], dtype=np.int32)

        def _active_subset_effective(self) -> bool:
            return True

    class OutputEvaluator:
        sampling_engine = "full"

        def __init__(self) -> None:
            self.refresh_reasons: list[str] = []

        def refresh_scaling_scatter_cache(self, _params, reason: str) -> None:
            self.refresh_reasons.append(f"scatter:{reason}")

        def refresh_source_metric_cache(self, _params, reason: str) -> None:
            self.refresh_reasons.append(f"source:{reason}")

    output = OutputEvaluator()
    seen_sampling_engines: list[str] = []

    def fake_build(args, _state, *, sampling_engine=None, quick_diagnostics=None):
        del quick_diagnostics
        seen_sampling_engines.append(str(sampling_engine or args.sampling_engine))
        return output

    monkeypatch.setattr(cluster_solver, "_build_cluster_evaluator_from_args", fake_build)
    monkeypatch.setattr(cluster_solver, "_log_evaluator_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log", lambda *_args, **_kwargs: None)

    result = cluster_solver._output_evaluator_for_validation(
        argparse.Namespace(sampling_engine="active_subset"),
        SimpleNamespace(),
        FitEvaluator(),
        np.asarray([0.0], dtype=float),
    )

    assert result is output
    assert seen_sampling_engines == ["full"]
    assert output.fit_sampling_engine == "active_subset"
    assert output.final_validation_sampling_engine == "full"
    assert output.fit_active_scaling_components == 2
    assert output.fit_ignored_inactive_scaling_components == 1
    assert output.refresh_reasons == [
        "scatter:active_subset_full_output",
        "source:active_subset_full_output",
    ]


def test_save_artifacts_replaces_plot_bundle_atomically(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    final_path = artifacts_dir / "plot_bundle.h5"
    final_path.write_bytes(b"old")

    def fake_write(path, *_args):
        Path(path).write_bytes(b"new")

    monkeypatch.setattr(cluster_solver, "_save_plot_bundle_h5", fake_write)

    cluster_solver._save_artifacts(
        artifacts_dir,
        SimpleNamespace(),
        argparse.Namespace(),
        np.asarray([], dtype=float),
        SimpleNamespace(),
    )

    assert final_path.read_bytes() == b"new"
    assert not (artifacts_dir / ".plot_bundle.h5.tmp").exists()


def test_save_artifacts_keeps_existing_plot_bundle_on_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    final_path = artifacts_dir / "plot_bundle.h5"
    final_path.write_bytes(b"old")

    def fake_write(path, *_args):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("write failed")

    monkeypatch.setattr(cluster_solver, "_save_plot_bundle_h5", fake_write)

    with pytest.raises(RuntimeError, match="write failed"):
        cluster_solver._save_artifacts(
            artifacts_dir,
            SimpleNamespace(),
            argparse.Namespace(),
            np.asarray([], dtype=float),
            SimpleNamespace(),
        )

    assert final_path.read_bytes() == b"old"
    assert not (artifacts_dir / ".plot_bundle.h5.tmp").exists()


def test_validation_stage_recovery_metrics_collects_ordered_stage_summaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    solver_root = tmp_path / "solver" / "fit"
    stage2_dir = solver_root / "stage2_joint"
    stage1_dir = solver_root / "stage1_large_only"
    _write_fake_stage_tables(
        stage2_dir,
        fit_method="svi+nuts",
        likelihood="source",
        sampler="numpyro_nuts",
        runtime_sec=20.0,
        best_loglike=-2.0,
        exact_values=[0.4, np.nan],
        source_values=[0.2, 0.3],
        approx_values=[0.5, 0.7],
    )
    _write_fake_stage_tables(
        stage1_dir,
        fit_method="svi",
        likelihood="source",
        sampler="svi",
        runtime_sec=10.0,
        best_loglike=-3.0,
        exact_values=[0.8, 1.0],
        source_values=[0.6, 0.4],
        approx_values=[0.9, 1.1],
    )
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps({"parameter_truth": {"p1": 1.0, "p2": 2.0}}), encoding="utf-8")

    def fake_load_plot_bundle(stage_dir: Path):
        if Path(stage_dir).name == "stage1_large_only":
            samples = np.asarray([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]], dtype=float)
        else:
            samples = np.asarray([[1.5, 2.0], [1.5, 2.0], [1.5, 2.0]], dtype=float)
        state = SimpleNamespace(parameter_specs=[SimpleNamespace(name="p1"), SimpleNamespace(name="p2")])
        arrays = {"samples": samples, "best_fit": np.median(samples, axis=0)}
        return state, {}, arrays, {}

    monkeypatch.setattr(validation, "_load_plot_bundle", fake_load_plot_bundle)

    rows = validation._collect_validation_stage_recovery_metrics(stage2_dir, truth_path)

    assert [row["stage"] for row in rows] == ["stage1_large_only", "stage2_joint"]
    assert rows[0]["exact_image_rms_mean"] == pytest.approx(0.9)
    assert rows[1]["failed_or_missing_exact"] == 1
    assert rows[0]["truth_parameter_count"] == 2
    assert rows[0]["parameter_mean_abs_bias"] == pytest.approx(0.5)
    assert rows[0]["worst_parameter"] == "p2"


def test_validation_stage_recovery_metrics_accepts_single_direct_solver_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    solver_root = tmp_path / "solver" / "fit"
    _write_fake_stage_tables(
        solver_root,
        fit_method="ns",
        likelihood=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        sampler="numpyro_jaxns",
        runtime_sec=30.0,
        best_loglike=-5.0,
        exact_values=[0.2],
        source_values=[0.1],
        approx_values=[0.3],
    )
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps({"parameter_truth": {}}), encoding="utf-8")
    monkeypatch.setattr(
        validation,
        "_load_plot_bundle",
        lambda _stage_dir: (SimpleNamespace(parameter_specs=[]), {}, {"samples": np.empty((1, 0)), "best_fit": np.empty((0,))}, {}),
    )

    rows = validation._collect_validation_stage_recovery_metrics(solver_root, truth_path)

    assert [row["stage"] for row in rows] == ["fit"]
    assert rows[0]["fit_method"] == "ns"
    assert rows[0]["sample_likelihood_mode"] == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE


def test_validation_run_summary_text_formats_metrics_and_na_values(tmp_path: Path) -> None:
    rows = [
        {
            "stage": "stage1_large_only",
            "fit_method": "svi",
            "sample_likelihood_mode": "source",
            "sampler": "svi",
            "runtime_sec": 10.0,
            "best_loglike": -3.0,
            "accept_prob_mean": np.nan,
            "divergence_count": np.nan,
            "mean_num_steps": np.nan,
            "n_families": 2,
            "n_images": 12,
            "family_count": 2,
            "exact_family_count": 2,
            "failed_or_missing_exact": 0,
            "exact_image_rms_mean": 0.9,
            "exact_image_rms_median": 0.9,
            "source_rms_mean": 0.5,
            "approx_image_rms_mean": 1.0,
            "truth_parameter_count": 2,
            "parameter_median_abs_bias": 0.5,
            "parameter_mean_abs_bias": 0.5,
            "parameter_coverage_68_fraction": 0.5,
            "worst_parameter": "p2",
            "worst_parameter_abs_bias": 1.0,
        },
        {
            "stage": "stage2_joint",
            "fit_method": "svi+nuts",
            "sample_likelihood_mode": "source",
            "sampler": "numpyro_nuts",
            "runtime_sec": 20.0,
            "best_loglike": -2.0,
            "accept_prob_mean": 0.91,
            "divergence_count": 2,
            "mean_num_steps": 17.5,
            "n_families": 2,
            "n_images": 12,
            "family_count": 2,
            "exact_family_count": 1,
            "failed_or_missing_exact": 1,
            "exact_image_rms_mean": 0.4,
            "exact_image_rms_median": 0.4,
            "source_rms_mean": 0.25,
            "approx_image_rms_mean": 0.6,
            "truth_parameter_count": np.nan,
            "parameter_median_abs_bias": np.nan,
            "parameter_mean_abs_bias": np.nan,
            "parameter_coverage_68_fraction": np.nan,
            "worst_parameter": "na",
            "worst_parameter_abs_bias": np.nan,
        },
    ]

    text = validation._format_validation_run_summary(
        rows,
        run_name="validation_log",
        seed=12345,
        solver_run_dir=tmp_path / "solver" / "fit" / "stage2_joint",
    )

    assert "run_name=validation_log" in text
    assert "final_stage=stage2_joint" in text
    assert "exact_image_rms_mean" in text
    assert "source_rms_mean" in text
    assert "param_med_abs_bias" in text
    assert "stage1_large_only" in text
    assert "stage2_joint" in text
    assert " na " in text
    assert "worst_parameter=p2" in text


def test_solver_sequential_run_summary_txt_aggregates_existing_stages(tmp_path: Path) -> None:
    root = tmp_path / "mock_run"
    stage1 = root / "stage1_large_only"
    stage2 = root / "stage2_joint"
    missing_stage = root / "stage3_image_plane"
    for stage, headline_chi_square, arc_chi_square in [(stage1, 10.0, 8.0), (stage2, 4.0, 3.0)]:
        tables_dir = stage / "tables"
        tables_dir.mkdir(parents=True)
        (tables_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "fit_method": "svi+nuts",
                    "sample_likelihood_mode": "source",
                    "sampler": "numpyro_nuts",
                    "n_families": 2,
                    "n_images": 6,
                    "headline_chi_square": headline_chi_square,
                    "headline_dof": 3,
                    "headline_reduced_chi_square": headline_chi_square / 3.0,
                    "arc_aware_chi_square": arc_chi_square,
                    "arc_aware_dof": 4,
                    "arc_aware_reduced_chi_square": arc_chi_square / 4.0,
                    "arc_aware_arc_supported_image_count": 1,
                    "arc_aware_missing_image_count": 2,
                    "ess_min": 12.0,
                    "rhat_max": 1.01,
                    "runtime_sec": 20.0,
                }
            ),
            encoding="utf-8",
        )

    path, text = cluster_solver._write_sequential_run_summary_txt(
        root,
        "mock_run",
        [stage1, missing_stage, stage2],
    )

    assert path == root / "run_summary.txt"
    assert path.exists()
    assert "Sequential Cluster Solver Run Summary" in text
    assert "Stage Quality Comparison" in text
    assert "stage1_large_only" in text
    assert "stage2_joint" in text
    assert "stage3_image_plane" not in text
    assert "headline_chi2" in text
    assert "arc_chi2" in text
    assert "AIC" not in text
    assert "BIC" not in text


def test_plot_run_summary_includes_nested_sampling_evidence() -> None:
    args = argparse.Namespace(
        run_name="ns_run",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        skip_stage3_image_plane_local_jacobian=False,
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        sampling_engine="full",
        nuts_init_boundary_frac=0.02,
        nuts_init_jitter_frac=0.02,
        image_plane_scatter_floor_arcsec=0.05,
        image_plane_scatter_prior="lognormal",
        image_plane_scatter_prior_median_arcsec=0.25,
        image_plane_scatter_prior_log_sigma=0.4,
        svi_steps=10,
        svi_learning_rate=0.005,
        ns_num_live_points=200,
        ns_max_samples=None,
        ns_dlogz=0.01,
        warmup=0,
        samples=25,
        chains=4,
        thin=1,
        max_tree_depth=8,
        target_accept=0.85,
        seed=123,
    )
    state = SimpleNamespace(
        run_name="ns_run",
        par_path="mock.par",
        fit_mode="joint",
        parameter_specs=[],
        family_data=[],
        packed_lens_spec=SimpleNamespace(component_family=np.asarray([], dtype=int)),
        potfiles=[],
        geometry_cache=None,
        cosmo_config={"H0": 70.0},
        fit_cosmology_flat_wcdm=False,
    )
    evaluator = SimpleNamespace(
        active_scaling_galaxies_by_potfile=[],
        active_scaling_component_indices=[],
        inactive_scaling_component_indices=[],
        requested_active_scaling_by_potfile={},
        actual_active_scaling_by_potfile={},
        total_scaling_by_potfile={},
        invalid_state_rejection_count=0,
        invalid_state_reason_counts={},
        eval_wall_times=[],
        timing_totals={},
        surrogate_enabled=False,
        approximate_eval_count=0,
        full_refresh_count=0,
    )
    posterior = PosteriorResults(
        samples=np.empty((0, 0), dtype=float),
        log_prob=np.empty((0,), dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=25,
        num_chains=0,
        sampler="numpyro_jaxns",
        init_diagnostics={
            "strategy_requested": "ns",
            "strategy_used": "ns",
            "svi_used": False,
            "ns_num_live_points": 200,
            "ns_max_samples": None,
            "ns_dlogz": 0.01,
            "ns_log_z_mean": -12.5,
            "ns_log_z_uncert": 0.2,
            "ns_ess": 42.0,
            "ns_total_num_samples": 321,
            "ns_posterior_samples": 4096,
            "ns_posterior_resampling": "jaxns.get_samples",
            "ns_total_num_likelihood_evaluations": 1234,
            "ns_termination_reason": 1,
        },
    )

    summary = _run_summary(args, state, 1.5, posterior, -3.0, evaluator)

    assert summary["sampler"] == "numpyro_jaxns"
    assert summary["ns_settings"] == {"num_live_points": 200, "max_samples": None, "dlogz": 0.01}
    assert summary["ns_log_z_mean"] == pytest.approx(-12.5)
    assert summary["ns_log_z_uncert"] == pytest.approx(0.2)
    assert summary["ns_posterior_samples"] == 4096
    assert summary["ns_posterior_resampling"] == "jaxns.get_samples"
    assert summary["ns_total_num_likelihood_evaluations"] == 1234
    assert summary["likelihood_stabilizer_max_gain"] == pytest.approx(0.0)
    assert summary["likelihood_stabilizer_max_residual_arcsec"] == pytest.approx(0.0)
    assert summary["likelihood_stabilizer_residual_loss"] == "gaussian"
    assert summary["likelihood_stabilizer_student_t_nu"] == pytest.approx(4.0)
    assert summary["image_plane_scatter_floor_arcsec"] == pytest.approx(0.05)
    assert summary["image_plane_scatter_prior"] == "lognormal"
    assert summary["image_plane_scatter_prior_median_arcsec"] == pytest.approx(0.25)
    assert summary["image_plane_scatter_prior_log_sigma"] == pytest.approx(0.4)
    assert "linearized_image_plane_max_gain" not in summary
    assert "linearized_image_plane_max_residual_arcsec" not in summary
    assert "linearized_image_plane_residual_loss" not in summary
    assert "linearized_image_plane_student_t_nu" not in summary


@pytest.mark.parametrize(("quiet", "expected_verbose", "expected_progress_count"), [(False, True, 1), (True, False, 0)])
def test_numpyro_nested_sampler_runner_uses_fake_sampler_and_records_evidence(
    monkeypatch: pytest.MonkeyPatch,
    quiet: bool,
    expected_verbose: bool,
    expected_progress_count: int,
) -> None:
    spec = ParameterSpec(
        name="x",
        sample_name="x",
        potential_id="mock",
        profile_type=81,
        field="x",
        prior_kind="uniform",
        lower=-1.0,
        upper=1.0,
        step=0.1,
    )
    args = argparse.Namespace(
        samples=3,
        thin=1,
        seed=123,
        ns_num_live_points=12,
        ns_max_samples=None,
        ns_dlogz=0.05,
        max_tree_depth=8,
        quiet=quiet,
    )
    state = SimpleNamespace(parameter_specs=[spec])
    evaluator = SimpleNamespace(
        timing_totals={},
        invalid_state_rejection_count=0,
        invalid_state_reason_counts={},
        _source_loglike_fn=lambda theta: -0.5 * jnp.square(theta[0]),
    )
    calls: dict[str, Any] = {}
    progress_events: list[tuple[str, Any]] = []

    class FakeProgress:
        def __init__(self, *columns, **kwargs):
            progress_events.append(("init", {"columns": columns, "kwargs": kwargs}))

        def __enter__(self):
            progress_events.append(("enter", None))
            return self

        def __exit__(self, exc_type, exc, traceback):
            progress_events.append(("exit", {"exc_type": exc_type, "exc": exc, "traceback": traceback}))
            return False

        def add_task(self, description, total=None):
            progress_events.append(("add_task", {"description": description, "total": total}))
            return 1

    monkeypatch.setattr(cluster_solver, "Progress", FakeProgress)

    class FakeNestedSampler:
        def __init__(self, model, constructor_kwargs, termination_kwargs):
            calls["model"] = model
            calls["constructor_kwargs"] = dict(constructor_kwargs)
            calls["termination_kwargs"] = dict(termination_kwargs)
            self._results = SimpleNamespace(
                log_Z_mean=jnp.asarray(-4.0),
                log_Z_uncert=jnp.asarray(0.25),
                ESS=jnp.asarray(7.0),
                total_num_samples=jnp.asarray(11),
                H_mean=jnp.asarray(3.0),
                log_L_samples=jnp.linspace(-9.0, -1.0, 11),
                log_dp_mean=jnp.linspace(-8.0, -2.0, 11),
                log_X_mean=-jnp.linspace(0.1, 5.0, 11),
                num_live_points_per_sample=jnp.full((11,), 12),
                num_likelihood_evaluations_per_sample=jnp.arange(1, 12),
                log_efficiency=jnp.asarray(-4.5),
                samples={"x": jnp.linspace(-1.0, 1.0, 11)},
                total_num_likelihood_evaluations=jnp.asarray(99),
                termination_reason=jnp.asarray(2),
            )

        def run(self, rng_key):
            calls["run_key_shape"] = tuple(np.asarray(rng_key).shape)

        def get_samples(self, rng_key, num_samples):
            calls["get_samples_key_shape"] = tuple(np.asarray(rng_key).shape)
            calls["get_samples_num_samples"] = int(num_samples)
            return {"x": jnp.linspace(-0.5, 0.5, int(num_samples))}

    posterior = cluster_solver._run_numpyro_nested_sampler(
        args,
        state,
        evaluator,
        sample_model=lambda: None,
        nested_sampler_factory=FakeNestedSampler,
    )

    assert calls["constructor_kwargs"]["num_live_points"] == 12
    assert calls["constructor_kwargs"]["max_samples"] is None
    assert calls["constructor_kwargs"]["verbose"] is expected_verbose
    assert calls["termination_kwargs"] == {"dlogZ": 0.05}
    assert sum(1 for event, _payload in progress_events if event == "init") == expected_progress_count
    assert sum(1 for event, _payload in progress_events if event == "add_task") == expected_progress_count
    assert posterior.sampler == "numpyro_jaxns"
    assert calls["get_samples_key_shape"] == (2,)
    assert calls["get_samples_num_samples"] == cluster_solver.DEFAULT_NS_POSTERIOR_SAMPLES
    assert posterior.samples.shape == (cluster_solver.DEFAULT_NS_POSTERIOR_SAMPLES, 1)
    np.testing.assert_allclose(
        posterior.samples[:, 0],
        np.linspace(-0.5, 0.5, cluster_solver.DEFAULT_NS_POSTERIOR_SAMPLES),
        atol=1.0e-6,
    )
    assert posterior.sample_weights is not None
    np.testing.assert_allclose(
        posterior.sample_weights,
        np.full(cluster_solver.DEFAULT_NS_POSTERIOR_SAMPLES, 1.0 / cluster_solver.DEFAULT_NS_POSTERIOR_SAMPLES),
        atol=1.0e-12,
    )
    assert posterior.warmup_steps == 0
    assert posterior.sample_steps == cluster_solver.DEFAULT_NS_POSTERIOR_SAMPLES
    assert posterior.num_chains == 0
    assert posterior.init_diagnostics["ns_log_z_mean"] == pytest.approx(-4.0)
    assert posterior.init_diagnostics["ns_max_samples"] is None
    assert posterior.init_diagnostics["ns_posterior_samples"] == cluster_solver.DEFAULT_NS_POSTERIOR_SAMPLES
    assert posterior.init_diagnostics["ns_posterior_resampling"] == "jaxns.get_samples"
    assert posterior.init_diagnostics["ns_total_num_likelihood_evaluations"] == 99
    assert posterior.ns_diagnostics is not None
    assert "samples" not in posterior.ns_diagnostics
    np.testing.assert_allclose(posterior.ns_diagnostics["log_L_samples"], np.linspace(-9.0, -1.0, 11))


def test_numpyro_nested_sampler_resamples_fixed_posterior_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ParameterSpec(
        name="x",
        sample_name="x",
        potential_id="mock",
        profile_type=81,
        field="x",
        prior_kind="uniform",
        lower=-1.0,
        upper=1.0,
        step=0.1,
    )
    args = argparse.Namespace(
        samples=3,
        thin=1,
        seed=123,
        ns_num_live_points=12,
        ns_max_samples=None,
        ns_dlogz=0.05,
        max_tree_depth=8,
        quiet=True,
    )
    state = SimpleNamespace(parameter_specs=[spec])
    evaluator = SimpleNamespace(
        timing_totals={},
        invalid_state_rejection_count=0,
        invalid_state_reason_counts={},
        _source_loglike_fn=lambda theta: -0.5 * jnp.square(theta[0]),
    )

    def fake_logprob(_parameter_specs, _evaluator, samples):
        return -np.square(np.asarray(samples, dtype=float)[:, 0])

    monkeypatch.setattr(cluster_solver, "_posterior_logprob_matrix", fake_logprob)
    calls: dict[str, Any] = {}

    class FakeNestedSampler:
        def __init__(self, model, constructor_kwargs, termination_kwargs):
            self._results = SimpleNamespace(
                log_Z_mean=jnp.asarray(-4.0),
                log_Z_uncert=jnp.asarray(0.25),
                ESS=jnp.asarray(7.0),
                total_num_samples=jnp.asarray(11),
                H_mean=jnp.asarray(3.0),
                log_L_samples=jnp.linspace(-9.0, -1.0, 11),
                log_dp_mean=jnp.linspace(-8.0, -2.0, 11),
                log_X_mean=-jnp.linspace(0.1, 5.0, 11),
                num_live_points_per_sample=jnp.full((11,), 12),
                num_likelihood_evaluations_per_sample=jnp.arange(1, 12),
                log_efficiency=jnp.asarray(-4.5),
                samples={"x": jnp.linspace(-1.0, 1.0, 11)},
                total_num_likelihood_evaluations=jnp.asarray(99),
                termination_reason=jnp.asarray(2),
            )

        def run(self, rng_key):
            return None

        def get_samples(self, rng_key, num_samples):
            calls["get_samples_num_samples"] = int(num_samples)
            return {"x": jnp.linspace(-1.0, 1.0, int(num_samples))}

    posterior = cluster_solver._run_numpyro_nested_sampler(
        args,
        state,
        evaluator,
        sample_model=lambda: None,
        nested_sampler_factory=FakeNestedSampler,
    )

    assert calls["get_samples_num_samples"] == cluster_solver.DEFAULT_NS_POSTERIOR_SAMPLES
    assert posterior.samples.shape == (cluster_solver.DEFAULT_NS_POSTERIOR_SAMPLES, 1)
    assert posterior.sample_steps == cluster_solver.DEFAULT_NS_POSTERIOR_SAMPLES
    assert posterior.ns_diagnostics is not None
    assert "samples" not in posterior.ns_diagnostics
    assert posterior.ns_diagnostics["log_dp_mean"].shape == (11,)
    assert posterior.sample_weights is not None
    np.testing.assert_allclose(
        posterior.sample_weights,
        np.full(cluster_solver.DEFAULT_NS_POSTERIOR_SAMPLES, 1.0 / cluster_solver.DEFAULT_NS_POSTERIOR_SAMPLES),
    )
    assert posterior.init_diagnostics["ns_total_num_samples"] == 11
    assert posterior.init_diagnostics["ns_posterior_samples"] == cluster_solver.DEFAULT_NS_POSTERIOR_SAMPLES
    assert posterior.init_diagnostics["ns_posterior_resampling"] == "jaxns.get_samples"


def test_posterior_logprob_matrix_batches_without_changing_values() -> None:
    spec = ParameterSpec(
        name="x",
        sample_name="x",
        potential_id="mock",
        profile_type=81,
        field="x",
        prior_kind="uniform",
        lower=-2.0,
        upper=2.0,
        step=0.1,
    )
    evaluator = SimpleNamespace(_source_loglike_fn=lambda theta: -0.5 * jnp.square(theta[0]))
    samples = np.linspace(-1.0, 1.0, 5, dtype=float).reshape(-1, 1)

    batched = cluster_solver._posterior_logprob_matrix([spec], evaluator, samples, batch_size=2)
    unbatched = cluster_solver._posterior_logprob_matrix([spec], evaluator, samples, batch_size=10)

    np.testing.assert_allclose(batched, unbatched)


def test_smc_normalized_coordinate_round_trip_and_prior_particles() -> None:
    specs = [
        ParameterSpec(
            name="u",
            sample_name="u",
            potential_id="mock",
            profile_type=81,
            field="u",
            prior_kind="uniform",
            lower=-2.0,
            upper=3.0,
            step=0.1,
        ),
        ParameterSpec(
            name="n",
            sample_name="n",
            potential_id="mock",
            profile_type=81,
            field="n",
            prior_kind="normal",
            lower=float("-inf"),
            upper=float("inf"),
            step=0.1,
            mean=10.0,
            std=2.0,
        ),
        ParameterSpec(
            name="t",
            sample_name="t",
            potential_id="mock",
            profile_type=81,
            field="t",
            prior_kind="truncated_normal",
            lower=0.0,
            upper=2.0,
            step=0.1,
            mean=1.0,
            std=0.3,
        ),
    ]
    normalization = cluster_solver._smc_normalization_arrays(specs)
    theta = jnp.asarray([0.25, 11.0, 1.2], dtype=jnp.float64)

    normalized = cluster_solver._smc_theta_to_normalized(theta, normalization)
    round_tripped = cluster_solver._smc_normalized_to_theta(normalized, normalization)
    particles = cluster_solver._smc_prior_particles(jax.random.PRNGKey(11), specs, 7)

    np.testing.assert_allclose(np.asarray(round_tripped), np.asarray(theta), rtol=1.0e-9, atol=1.0e-9)
    assert particles.shape == (7, 3)
    assert np.isfinite(np.asarray(particles)).all()


def test_blackjax_smc_sampler_runner_with_fake_algorithm(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ParameterSpec(
        name="x",
        sample_name="x",
        potential_id="mock",
        profile_type=81,
        field="x",
        prior_kind="uniform",
        lower=-1.0,
        upper=1.0,
        step=0.1,
    )
    args = argparse.Namespace(
        seed=3,
        smc_particles=4,
        smc_mcmc_kernel="rmh",
        smc_mcmc_steps=2,
        smc_target_ess_frac=0.75,
        smc_max_temperature_steps=3,
        smc_rmh_scale=0.5,
        smc_mala_step_size=0.02,
        jax_default_device="cpu",
        smc_device="cpu",
        max_tree_depth=8,
        quiet=True,
    )
    state = SimpleNamespace(parameter_specs=[spec])

    def loglike(theta):
        return -0.5 * jnp.square(theta[0])

    evaluator = SimpleNamespace(
        timing_totals={},
        invalid_state_rejection_count=0,
        invalid_state_reason_counts={},
        surrogate_enabled=False,
        _source_loglike_fn=loglike,
        _source_loglike_impl=loglike,
    )
    compile_devices: list[Any | None] = []

    def fake_compile(evaluator, *, device=None):
        compile_devices.append(device)
        evaluator._source_loglike_fn = loglike
        return True

    def fake_logprob(_parameter_specs, _evaluator, samples):
        return -np.square(np.asarray(samples, dtype=float)[:, 0])

    monkeypatch.setattr(cluster_solver, "_posterior_logprob_matrix", fake_logprob)
    monkeypatch.setattr(cluster_solver, "_compile_evaluator_source_loglike", fake_compile)

    class FakeSMCState(NamedTuple):
        particles: jnp.ndarray
        weights: jnp.ndarray
        tempering_param: jnp.ndarray

    class FakeUpdateInfo(NamedTuple):
        acceptance_rate: jnp.ndarray
        is_accepted: jnp.ndarray

    class FakeInfo(NamedTuple):
        log_likelihood_increment: jnp.ndarray
        update_info: FakeUpdateInfo

    class FakeAlgorithm:
        def init(self, particles):
            return FakeSMCState(
                particles=particles,
                weights=jnp.full((particles.shape[0],), 1.0 / particles.shape[0]),
                tempering_param=jnp.asarray(0.0),
            )

        def step(self, _rng_key, smc_state):
            return (
                FakeSMCState(
                    particles=smc_state.particles,
                    weights=jnp.asarray([0.5, 0.25, 0.125, 0.125]),
                    tempering_param=jnp.asarray(1.0),
                ),
                FakeInfo(
                    log_likelihood_increment=jnp.asarray(2.5),
                    update_info=FakeUpdateInfo(
                        acceptance_rate=jnp.asarray([0.25, 0.5, 0.75, 1.0]),
                        is_accepted=jnp.asarray([False, True, True, True]),
                    ),
                ),
            )

    def fake_factory(**_kwargs):
        return FakeAlgorithm()

    posterior = cluster_solver._run_blackjax_smc_sampler(
        args,
        state,
        evaluator,
        smc_algorithm_factory=fake_factory,
    )

    assert posterior.sampler == "blackjax_smc"
    assert posterior.samples.shape == (4, 1)
    assert posterior.sample_weights is not None
    np.testing.assert_allclose(posterior.sample_weights, [0.5, 0.25, 0.125, 0.125])
    np.testing.assert_allclose(posterior.temperature_schedule, [0.0, 1.0])
    assert posterior.ess_history is not None
    assert posterior.move_acceptance_history is not None
    np.testing.assert_allclose(posterior.move_acceptance_history, [0.625])
    assert posterior.init_diagnostics["smc_particles"] == 4
    assert posterior.init_diagnostics["smc_mcmc_kernel"] == "rmh"
    assert posterior.init_diagnostics["smc_temperature_steps"] == 1
    assert posterior.init_diagnostics["smc_logz_estimate"] == pytest.approx(2.5)
    assert posterior.init_diagnostics["jax_default_device"].startswith("cpu")
    assert posterior.init_diagnostics["smc_device"].startswith("cpu")
    assert posterior.init_diagnostics["smc_device_backend"] == "cpu"
    assert np.isfinite(posterior.init_diagnostics["smc_first_step_compile_run_sec"])
    assert len(compile_devices) == 2
    assert all(getattr(device, "platform", None) == "cpu" for device in compile_devices)


def test_plot_bundle_round_trips_ns_diagnostics(tmp_path: Path) -> None:
    packed = PackedLensSpec(
        **{
            field_name: np.asarray([], dtype=float)
            for field_name in PackedLensSpec.__dataclass_fields__
        }
    )
    state = BuildState(
        run_name="ns_run",
        par_path="mock.par",
        cosmo_config={"H0": 70.0},
        z_lens=0.4,
        sigma_arcsec=0.1,
        parsed={},
        parameter_specs=[],
        base_components=[],
        packed_lens_spec=packed,
        family_data=[],
        bin_data=[],
        lens_model_list=[],
        reference=(0, 0.0, 0.0),
        fit_mode="joint",
        potfiles=[],
        scaling_component_records=[],
        previous_stage_best_values={"halo_v_disp": 1100.0},
        source_position_parameterization="direct",
    )
    results = PosteriorResults(
        samples=np.empty((2, 0), dtype=float),
        log_prob=np.asarray([-1.0, -2.0], dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=2,
        num_chains=0,
        sampler="numpyro_jaxns",
        sample_weights=np.asarray([0.75, 0.25], dtype=float),
        temperature_schedule=np.asarray([0.0, 0.4, 1.0], dtype=float),
        ess_history=np.asarray([2.0, 1.5, 1.2], dtype=float),
        move_acceptance_history=np.asarray([0.8, 0.7], dtype=float),
        ns_diagnostics={
            "log_L_samples": np.asarray([-5.0, -4.0, -3.0], dtype=float),
            "log_dp_mean": np.asarray([-7.0, -6.0, -5.0], dtype=float),
        },
    )
    path = tmp_path / "plot_bundle.h5"

    cluster_solver._save_plot_bundle_h5(path, state, argparse.Namespace(foo="bar"), np.empty((0,), dtype=float), results)
    loaded_state, _args, arrays, _init_diag = cluster_solver._rebuild_state_from_h5(path)

    assert loaded_state.previous_stage_best_values == {"halo_v_disp": 1100.0}
    assert "ns_diagnostics" in arrays
    np.testing.assert_allclose(arrays["sample_weights"], [0.75, 0.25])
    np.testing.assert_allclose(arrays["temperature_schedule"], [0.0, 0.4, 1.0])
    np.testing.assert_allclose(arrays["ess_history"], [2.0, 1.5, 1.2])
    np.testing.assert_allclose(arrays["move_acceptance_history"], [0.8, 0.7])
    np.testing.assert_allclose(arrays["ns_diagnostics"]["log_L_samples"], [-5.0, -4.0, -3.0])
    assert "samples" not in arrays["ns_diagnostics"]


def test_plot_bundle_round_trips_cab_arc_arrays(tmp_path: Path) -> None:
    packed = PackedLensSpec(
        **{
            field_name: np.asarray([], dtype=float)
            for field_name in PackedLensSpec.__dataclass_fields__
        }
    )
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.2,
        image_labels=["1.a", "1.b"],
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0], dtype=float),
        reliability=np.asarray([1.0, 0.9], dtype=float),
        arc_anchor_x=np.asarray([0.1, 1.0], dtype=float),
        arc_anchor_y=np.asarray([0.2, 0.0], dtype=float),
        arc_tangent_angle_rad=np.asarray([0.7, 0.0], dtype=float),
        arc_curvature_arcsec_inv=np.asarray([0.08, 0.0], dtype=float),
        arc_sigma_tangent_angle_rad=np.asarray([0.05, 0.001], dtype=float),
        arc_sigma_curvature_arcsec_inv=np.asarray([0.02, 0.0001], dtype=float),
        arc_reliability=np.asarray([0.8, 1.0], dtype=float),
        arc_has_constraint=np.asarray([True, False], dtype=bool),
    )
    bin_item = cluster_solver.BinData(
        effective_z_source=2.0,
        family_ids=["1"],
        family_index_per_image=np.asarray([0, 0], dtype=int),
        x_obs=family.x_obs,
        y_obs=family.y_obs,
        sigma_per_image=np.asarray([0.2, 0.2], dtype=float),
        reliability_per_image=family.reliability,
        arc_anchor_x=family.arc_anchor_x,
        arc_anchor_y=family.arc_anchor_y,
        arc_tangent_angle_rad=family.arc_tangent_angle_rad,
        arc_curvature_arcsec_inv=family.arc_curvature_arcsec_inv,
        arc_sigma_tangent_angle_rad=family.arc_sigma_tangent_angle_rad,
        arc_sigma_curvature_arcsec_inv=family.arc_sigma_curvature_arcsec_inv,
        arc_reliability=family.arc_reliability,
        arc_has_constraint=family.arc_has_constraint,
    )
    state = BuildState(
        run_name="cab_run",
        par_path="mock.par",
        cosmo_config={"H0": 70.0},
        z_lens=0.4,
        sigma_arcsec=0.2,
        parsed={},
        parameter_specs=[],
        base_components=[],
        packed_lens_spec=packed,
        family_data=[family],
        bin_data=[bin_item],
        lens_model_list=[],
        reference=(0, 0.0, 0.0),
        fit_mode="joint",
        potfiles=[],
        scaling_component_records=[],
        previous_stage_best_values=None,
        source_position_parameterization="direct",
    )
    results = PosteriorResults(
        samples=np.empty((1, 0), dtype=float),
        log_prob=np.asarray([-1.0], dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=1,
        num_chains=0,
    )
    path = tmp_path / "plot_bundle.h5"

    cluster_solver._save_plot_bundle_h5(path, state, argparse.Namespace(foo="bar"), np.empty((0,), dtype=float), results)
    loaded_state, _args, _arrays, _init_diag = cluster_solver._rebuild_state_from_h5(path)

    loaded_family = loaded_state.family_data[0]
    loaded_bin = loaded_state.bin_data[0]
    np.testing.assert_allclose(loaded_family.arc_anchor_x, [0.1, 1.0])
    np.testing.assert_allclose(loaded_family.arc_tangent_angle_rad, [0.7, 0.0])
    assert loaded_family.arc_has_constraint.tolist() == [True, False]
    np.testing.assert_allclose(loaded_bin.arc_curvature_arcsec_inv, [0.08, 0.0])
    np.testing.assert_allclose(loaded_bin.arc_reliability, [0.8, 1.0])
    assert loaded_bin.arc_has_constraint.tolist() == [True, False]


def _expected_prefit_output_paths(realization_dir: Path) -> dict[str, Path]:
    return {
        "subhalo_shmf_plot": realization_dir / "subhalo_shmf.pdf",
        "prefit_subhalo_spatial_distribution_plot": realization_dir / "prefit_subhalo_spatial_distribution.pdf",
        "prefit_critical_lines_plot": realization_dir / "prefit_critical_lines.pdf",
    }


def test_validation_run_single_bcg_validation_logs_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call_order: list[str] = []
    generated_configs: list[validation.SingleBCGMockConfig] = []

    def fake_generate(root, config, progress_callback=None):
        del progress_callback
        generated_configs.append(config)
        paths = validation.MockClusterPaths(
            root=Path(root),
            par_path=Path(root) / "mock.par",
            image_catalog_path=Path(root) / "obs_arcs.cat",
            truth_path=Path(root) / "truth.json",
            mock_images_path=Path(root) / "mock_images.csv",
        )
        images = pd.DataFrame({"family_id": ["1", "1"], "image_label": ["1.1", "1.2"]})
        return paths, images, {"parameter_truth": {}}

    def fake_solver(par_path, output_dir, run_name, args):
        call_order.append("solver")
        return Path(output_dir) / run_name / "stage3_image_plane"

    def fake_prefit(truth, images, output_dir):
        del truth, images
        call_order.append("prefit")
        return _expected_prefit_output_paths(Path(output_dir))

    def fake_recovery(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir,
        posterior_diagnostic_draws,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        critical_caustic_plot_grid_scale_arcsec=validation.DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC,
        recovery_profile_draws=validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
        quick_diagnostics=False,
        progress_args=None,
        recovery_payload=None,
    ):
        del recovery_profile_draws
        call_order.append("recovery")
        return {"summary_plot": Path(output_dir) / "validation_summary.pdf"}

    monkeypatch.setattr(validation, "generate_single_bcg_mock", fake_generate)
    monkeypatch.setattr(validation, "write_prefit_validation_diagnostics", fake_prefit)
    monkeypatch.setattr(validation, "_run_cluster_solver", fake_solver)
    monkeypatch.setattr(validation, "write_recovery_outputs", fake_recovery)
    monkeypatch.setattr(
        validation,
        "write_validation_run_summary",
        lambda _solver_run_dir, _truth_path, output_dir, run_name, seed: Path(output_dir) / "run_summary.txt",
    )
    args = _validation_run_args(
        tmp_path,
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        bcg_position_prior_half_width_arcsec=2.0,
    )

    outputs = validation.run_single_bcg_validation(args)
    validation._close_debug_log()

    root = tmp_path / "single_bcg" / "validation_log"
    realization_dir = root / "seed_12345"
    assert outputs == [
        {
            **_expected_prefit_output_paths(realization_dir),
            "summary_plot": realization_dir / "validation_summary.pdf",
            "results_json": root / "seed_12345_results.json",
        }
    ]
    assert generated_configs[0].bcg_position_prior_half_width_arcsec == pytest.approx(2.0)
    assert call_order == ["prefit", "solver", "recovery"]
    log_text = (tmp_path / "single_bcg" / "validation_log" / "run_debug.log").read_text(encoding="utf-8")
    assert "VALIDATION REALIZATION 1/1" in log_text
    assert "realization start" in log_text
    assert "mock complete images=2" in log_text
    assert "recovery complete files=5" in log_text
    assert "validation complete realizations=1" in log_text


def test_validation_run_single_bcg_validation_forwards_posterior_diagnostic_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_modes: list[str] = []
    captured_profile_draws: list[int] = []

    def fake_generate(root, config, progress_callback=None):
        del progress_callback
        paths = validation.MockClusterPaths(
            root=Path(root),
            par_path=Path(root) / "mock.par",
            image_catalog_path=Path(root) / "obs_arcs.cat",
            truth_path=Path(root) / "truth.json",
            mock_images_path=Path(root) / "mock_images.csv",
        )
        images = pd.DataFrame({"family_id": ["1"], "image_label": ["1.1"]})
        return paths, images, {"parameter_truth": {}}

    def fake_recovery(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir,
        posterior_diagnostic_draws,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        critical_caustic_plot_grid_scale_arcsec=validation.DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC,
        recovery_profile_draws=validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
        quick_diagnostics=False,
        progress_args=None,
        recovery_payload=None,
    ):
        captured_modes.append(str(posterior_diagnostic_mode))
        captured_profile_draws.append(int(recovery_profile_draws))
        return {"summary_plot": Path(output_dir) / "validation_summary.pdf"}

    monkeypatch.setattr(validation, "generate_single_bcg_mock", fake_generate)
    monkeypatch.setattr(
        validation,
        "_run_cluster_solver",
        lambda _par_path, output_dir, run_name, _args: Path(output_dir) / run_name / "stage3_image_plane",
    )
    monkeypatch.setattr(validation, "write_recovery_outputs", fake_recovery)
    monkeypatch.setattr(
        validation,
        "write_validation_run_summary",
        lambda _solver_run_dir, _truth_path, output_dir, run_name, seed: Path(output_dir) / "run_summary.txt",
    )
    args = _validation_run_args(
        tmp_path,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
        recovery_profile_draws=32,
    )

    validation.run_single_bcg_validation(args)
    validation._close_debug_log()

    assert captured_modes == [validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE]
    assert captured_profile_draws == [32]


def test_validation_run_single_bcg_validation_keeps_mock_and_plot_caustic_scales_separate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_mock_grid_scales: list[float] = []
    captured_plot_grid_scales: list[float] = []
    captured_mock_image_min_distances: list[tuple[float, float]] = []

    def fake_generate(root, config, progress_callback=None):
        del progress_callback
        captured_mock_grid_scales.append(float(config.caustic_grid_scale_arcsec))
        captured_mock_image_min_distances.append(
            (
                float(config.primary_image_min_distance_arcsec),
                float(config.subhalo_image_min_distance_arcsec),
            )
        )
        paths = validation.MockClusterPaths(
            root=Path(root),
            par_path=Path(root) / "mock.par",
            image_catalog_path=Path(root) / "obs_arcs.cat",
            truth_path=Path(root) / "truth.json",
            mock_images_path=Path(root) / "mock_images.csv",
        )
        images = pd.DataFrame({"family_id": ["1"], "image_label": ["1.1"]})
        return paths, images, {"parameter_truth": {}}

    def fake_recovery(
        _run_dir,
        _truth_path,
        _mock_images_path,
        output_dir,
        posterior_diagnostic_draws,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        critical_caustic_plot_grid_scale_arcsec=validation.DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC,
        recovery_profile_draws=validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
        quick_diagnostics=False,
        progress_args=None,
        recovery_payload=None,
    ):
        del posterior_diagnostic_draws, posterior_diagnostic_mode
        del recovery_profile_draws, quick_diagnostics, progress_args, recovery_payload
        captured_plot_grid_scales.append(float(critical_caustic_plot_grid_scale_arcsec))
        return {"summary_plot": Path(output_dir) / "validation_summary.pdf"}

    monkeypatch.setattr(validation, "generate_single_bcg_mock", fake_generate)
    monkeypatch.setattr(
        validation,
        "_run_cluster_solver",
        lambda _par_path, output_dir, run_name, _args: Path(output_dir) / run_name / "stage3_image_plane",
    )
    monkeypatch.setattr(validation, "write_recovery_outputs", fake_recovery)
    monkeypatch.setattr(
        validation,
        "write_validation_run_summary",
        lambda _solver_run_dir, _truth_path, output_dir, run_name, seed: Path(output_dir) / "run_summary.txt",
    )
    args = _validation_run_args(
        tmp_path,
        caustic_grid_scale_arcsec=0.5,
        critical_caustic_plot_grid_scale_arcsec=0.2,
        primary_image_min_distance_arcsec=2.75,
        subhalo_image_min_distance_arcsec=0.9,
    )

    validation.run_single_bcg_validation(args)
    validation._close_debug_log()

    assert captured_mock_grid_scales == [pytest.approx(0.5)]
    assert captured_mock_image_min_distances[0][0] == pytest.approx(2.75)
    assert captured_mock_image_min_distances[0][1] == pytest.approx(0.9)
    assert captured_plot_grid_scales == [pytest.approx(0.2)]


def test_validation_run_single_bcg_validation_forwards_quick_diagnostics_to_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_quick: list[bool] = []

    def fake_generate(root, config, progress_callback=None):
        del progress_callback
        paths = validation.MockClusterPaths(
            root=Path(root),
            par_path=Path(root) / "mock.par",
            image_catalog_path=Path(root) / "obs_arcs.cat",
            truth_path=Path(root) / "truth.json",
            mock_images_path=Path(root) / "mock_images.csv",
        )
        images = pd.DataFrame({"family_id": ["1"], "image_label": ["1.1"]})
        return paths, images, {"parameter_truth": {}}

    def fake_recovery(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir,
        posterior_diagnostic_draws,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        critical_caustic_plot_grid_scale_arcsec=validation.DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC,
        recovery_profile_draws=validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
        quick_diagnostics=False,
        progress_args=None,
        recovery_payload=None,
    ):
        del recovery_profile_draws
        captured_quick.append(bool(quick_diagnostics))
        return {"summary_plot": Path(output_dir) / "validation_summary.pdf"}

    monkeypatch.setattr(validation, "generate_single_bcg_mock", fake_generate)
    monkeypatch.setattr(
        validation,
        "_run_cluster_solver",
        lambda _par_path, output_dir, run_name, _args: Path(output_dir) / run_name / "stage3_image_plane",
    )
    monkeypatch.setattr(validation, "write_recovery_outputs", fake_recovery)
    monkeypatch.setattr(
        validation,
        "write_validation_run_summary",
        lambda _solver_run_dir, _truth_path, output_dir, run_name, seed: Path(output_dir) / "run_summary.txt",
    )
    args = _validation_run_args(tmp_path, quick_diagnostics=True)

    validation.run_single_bcg_validation(args)
    validation._close_debug_log()

    assert captured_quick == [True]


def test_validation_run_single_bcg_validation_writes_stage3_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_generate(root, config, progress_callback=None):
        del progress_callback
        paths = validation.MockClusterPaths(
            root=Path(root),
            par_path=Path(root) / "mock.par",
            image_catalog_path=Path(root) / "obs_arcs.cat",
            truth_path=Path(root) / "truth.json",
            mock_images_path=Path(root) / "mock_images.csv",
        )
        images = pd.DataFrame({"family_id": ["1"], "image_label": ["1.1"]})
        return paths, images, {"parameter_truth": {}}

    recovery_calls: list[tuple[Path, Path]] = []
    captured_profile_draws: list[int] = []
    artifact_checks: list[Path] = []

    def fake_recovery(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir,
        posterior_diagnostic_draws,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        critical_caustic_plot_grid_scale_arcsec=validation.DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC,
        recovery_profile_draws=validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
        quick_diagnostics=False,
        progress_args=None,
        recovery_payload=None,
    ):
        recovery_calls.append((Path(run_dir), Path(output_dir)))
        captured_profile_draws.append(int(recovery_profile_draws))
        if recovery_payload is not None:
            recovery_payload.update(
                {
                    "summary": {
                        "run_dir": str(Path(run_dir)),
                        "output_dir": str(Path(output_dir)),
                    },
                    "tables": {
                        "parameters": {
                            "records": [{"parameter": "halo.v_disp", "median": 760.0}],
                        }
                    },
                }
            )
        return {
            "summary_plot": Path(output_dir) / "validation_summary.pdf",
            "image_recovery_plot": Path(output_dir) / "image_recovery.pdf",
        }

    def fake_has_artifacts(run_dir):
        artifact_checks.append(Path(run_dir))
        return True

    monkeypatch.setattr(validation, "generate_single_bcg_mock", fake_generate)
    monkeypatch.setattr(
        validation,
        "_run_cluster_solver",
        lambda _par_path, output_dir, run_name, _args: Path(output_dir) / run_name / "stage4_linearized_image_plane",
    )
    monkeypatch.setattr(validation, "_validation_stage_has_recovery_artifacts", fake_has_artifacts)
    monkeypatch.setattr(validation, "write_recovery_outputs", fake_recovery)
    monkeypatch.setattr(
        validation,
        "write_validation_run_summary",
        lambda _solver_run_dir, _truth_path, output_dir, run_name, seed: Path(output_dir) / "run_summary.txt",
    )
    args = _validation_run_args(
        tmp_path,
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        write_stage3_recovery=True,
        recovery_profile_draws=48,
    )

    outputs = validation.run_single_bcg_validation(args)
    validation._close_debug_log()

    realization_dir = tmp_path / "single_bcg" / "validation_log" / "seed_12345"
    final_run_dir = realization_dir / "solver" / "fit" / "stage4_linearized_image_plane"
    stage3_run_dir = realization_dir / "solver" / "fit" / "stage3_image_plane"
    assert artifact_checks == [stage3_run_dir]
    assert recovery_calls == [
        (final_run_dir, realization_dir),
        (stage3_run_dir, realization_dir / "stage3_recovery"),
    ]
    assert captured_profile_draws == [48, 48]
    assert outputs == [
        {
            **_expected_prefit_output_paths(realization_dir),
            "summary_plot": realization_dir / "validation_summary.pdf",
            "image_recovery_plot": realization_dir / "image_recovery.pdf",
            "stage3_summary_plot": realization_dir / "stage3_recovery" / "validation_summary.pdf",
            "stage3_image_recovery_plot": realization_dir / "stage3_recovery" / "image_recovery.pdf",
            "results_json": tmp_path / "single_bcg" / "validation_log" / "seed_12345_results.json",
        }
    ]
    results_payload = json.loads((tmp_path / "single_bcg" / "validation_log" / "seed_12345_results.json").read_text())
    assert results_payload["validation"]["recovery"]["final"]["summary"]["output_dir"] == str(realization_dir)
    assert results_payload["validation"]["recovery"]["stage3"]["summary"]["output_dir"] == str(
        realization_dir / "stage3_recovery"
    )
    assert results_payload["validation"]["output_paths"]["stage3_summary_plot"] == str(
        realization_dir / "stage3_recovery" / "validation_summary.pdf"
    )


def test_validation_resume_reuses_existing_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mock_dir = tmp_path / "single_bcg" / "validation_log" / "seed_12345" / "mock"
    mock_dir.mkdir(parents=True)
    (mock_dir / "single_bcg_mock.par").write_text("mock par", encoding="utf-8")
    (mock_dir / "obs_arcs.cat").write_text("", encoding="utf-8")
    (mock_dir / "truth.json").write_text(json.dumps({"parameter_truth": {}}), encoding="utf-8")
    (mock_dir / "mock_images.json").write_text(
        json.dumps([{"family_id": "1", "image_label": "1.1"}]),
        encoding="utf-8",
    )
    solver_calls: list[Path] = []
    call_order: list[str] = []

    def fail_generate(*_args, **_kwargs):
        raise AssertionError("generate_single_bcg_mock should not run in resume mode with complete mock inputs")

    def fake_prefit(truth, images, output_dir):
        del truth, images
        call_order.append("prefit")
        return _expected_prefit_output_paths(Path(output_dir))

    def fake_solver(par_path, output_dir, run_name, args):
        call_order.append("solver")
        solver_calls.append(Path(par_path))
        return Path(output_dir) / run_name / "stage2_joint"

    monkeypatch.setattr(validation, "generate_single_bcg_mock", fail_generate)
    monkeypatch.setattr(validation, "write_prefit_validation_diagnostics", fake_prefit)
    monkeypatch.setattr(validation, "_run_cluster_solver", fake_solver)
    monkeypatch.setattr(
        validation,
        "write_recovery_outputs",
        lambda _run_dir, _truth_path, _mock_images_path, output_dir, posterior_diagnostic_draws, posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT, critical_caustic_plot_grid_scale_arcsec=validation.DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC, recovery_profile_draws=validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP, quick_diagnostics=False, progress_args=None, recovery_payload=None: {
            "summary_plot": Path(output_dir) / "validation_summary.pdf"
        },
    )
    monkeypatch.setattr(
        validation,
        "write_validation_run_summary",
        lambda _solver_run_dir, _truth_path, output_dir, run_name, seed: Path(output_dir) / "run_summary.txt",
    )
    args = _validation_run_args(tmp_path, resume=True)

    outputs = validation.run_single_bcg_validation(args)
    validation._close_debug_log()

    assert solver_calls == [mock_dir / "single_bcg_mock.par"]
    assert call_order == ["prefit", "solver"]
    root = tmp_path / "single_bcg" / "validation_log"
    assert outputs == [
        {
            **_expected_prefit_output_paths(root / "seed_12345"),
            "summary_plot": root / "seed_12345" / "validation_summary.pdf",
            "results_json": root / "seed_12345_results.json",
        }
    ]
    log_text = (tmp_path / "single_bcg" / "validation_log" / "run_debug.log").read_text(encoding="utf-8")
    assert "[resume] reusing mock" in log_text


def test_validation_resume_refreshes_complete_realization_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    realization_dir = tmp_path / "single_bcg" / "validation_log" / "seed_12345"
    realization_dir.mkdir(parents=True)
    (realization_dir / "run_summary.txt").write_text("done", encoding="utf-8")
    for path in validation._validation_recovery_output_paths(realization_dir).values():
        path.write_text("pdf", encoding="utf-8")
    mock_dir = realization_dir / "mock"
    mock_dir.mkdir(parents=True)
    (mock_dir / "single_bcg_mock.par").write_text("mock par", encoding="utf-8")
    (mock_dir / "obs_arcs.cat").write_text("", encoding="utf-8")
    (mock_dir / "truth.json").write_text(json.dumps({"parameter_truth": {}}), encoding="utf-8")
    (mock_dir / "mock_images.json").write_text(
        json.dumps([{"family_id": "1", "image_label": "1.1"}]),
        encoding="utf-8",
    )
    solver_calls: list[tuple[Path, Path, str]] = []
    recovery_calls: list[Path] = []
    summary_calls: list[Path] = []

    monkeypatch.setattr(
        validation,
        "generate_single_bcg_mock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("mock generation should be skipped")),
    )

    def fake_solver(par_path, output_dir, run_name, args):
        solver_calls.append((Path(par_path), Path(output_dir), run_name))
        return Path(output_dir) / run_name / "stage2_joint"

    def fake_recovery(
        _run_dir,
        _truth_path,
        _mock_images_path,
        output_dir,
        posterior_diagnostic_draws,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        critical_caustic_plot_grid_scale_arcsec=validation.DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC,
        recovery_profile_draws=validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
        quick_diagnostics=False,
        progress_args=None,
        recovery_payload=None,
    ):
        del recovery_profile_draws
        recovery_calls.append(Path(output_dir))
        return {"summary_plot": Path(output_dir) / "validation_summary.pdf"}

    def fake_summary(_solver_run_dir, _truth_path, output_dir, run_name, seed):
        summary_calls.append(Path(output_dir))
        return Path(output_dir) / "run_summary.txt"

    monkeypatch.setattr(validation, "_run_cluster_solver", fake_solver)
    monkeypatch.setattr(validation, "write_recovery_outputs", fake_recovery)
    monkeypatch.setattr(validation, "write_validation_run_summary", fake_summary)
    args = _validation_run_args(tmp_path, resume=True)

    outputs = validation.run_single_bcg_validation(args)
    validation._close_debug_log()

    assert solver_calls == [(mock_dir / "single_bcg_mock.par", realization_dir / "solver", "fit")]
    assert recovery_calls == [realization_dir]
    assert summary_calls == [realization_dir]
    assert outputs == [
        {
            **_expected_prefit_output_paths(realization_dir),
            "summary_plot": realization_dir / "validation_summary.pdf",
            "results_json": tmp_path / "single_bcg" / "validation_log" / "seed_12345_results.json",
        }
    ]
    log_text = (tmp_path / "single_bcg" / "validation_log" / "run_debug.log").read_text(encoding="utf-8")
    assert "[resume] reusing mock" in log_text
    assert "[resume] refreshing validation outputs" in log_text


def test_validation_quiet_suppresses_console_but_writes_debug_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_generate(root, config, progress_callback=None):
        del progress_callback
        paths = validation.MockClusterPaths(
            root=Path(root),
            par_path=Path(root) / "mock.par",
            image_catalog_path=Path(root) / "obs_arcs.cat",
            truth_path=Path(root) / "truth.json",
            mock_images_path=Path(root) / "mock_images.csv",
        )
        return paths, pd.DataFrame({"family_id": ["1"], "image_label": ["1.1"]}), {}

    monkeypatch.setattr(validation, "generate_single_bcg_mock", fake_generate)
    monkeypatch.setattr(validation, "_run_cluster_solver", lambda _par_path, output_dir, run_name, _args: Path(output_dir) / run_name / "stage2_joint")
    monkeypatch.setattr(
        validation,
        "write_recovery_outputs",
        lambda _run_dir, _truth_path, _mock_images_path, output_dir, posterior_diagnostic_draws, posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT, critical_caustic_plot_grid_scale_arcsec=validation.DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC, recovery_profile_draws=validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP, quick_diagnostics=False, progress_args=None, recovery_payload=None: {
            "summary_plot": Path(output_dir) / "validation_summary.pdf"
        },
    )
    monkeypatch.setattr(
        validation,
        "write_validation_run_summary",
        lambda _solver_run_dir, _truth_path, output_dir, run_name, seed: Path(output_dir) / "run_summary.txt",
    )
    args = _validation_run_args(tmp_path, quiet=True)

    validation.run_single_bcg_validation(args)
    validation._close_debug_log()
    captured = capsys.readouterr()

    assert "validation complete" not in captured.out
    assert "validation complete" not in captured.err
    log_text = (tmp_path / "single_bcg" / "validation_log" / "run_debug.log").read_text(encoding="utf-8")
    assert "validation complete realizations=1" in log_text


def _source_plane_loglike_common(**updates) -> dict[str, Any]:
    payload = dict(
        beta_x=jnp.asarray([0.0, 2.0], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.ones(2, dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999999, 0.999999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        inv_abs_mu=jnp.ones(2, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=100.0,
    )
    payload.update(updates)
    return payload


def test_source_plane_loglike_matches_default_gaussian_weighted_centroid() -> None:
    value = _source_plane_bin_loglike(**_source_plane_loglike_common())

    expected = 2.0 * (-0.5 * (1.0 + np.log((2.0 * np.pi) ** 2)))
    np.testing.assert_allclose(float(value), expected, rtol=0.0, atol=5.0e-6)


def test_source_plane_max_gain_floors_near_critical_magnification_weight() -> None:
    common = _source_plane_loglike_common(
        beta_x=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        sigma_per_image=jnp.ones(2, dtype=jnp.float64),
        inv_abs_mu=jnp.asarray([1.0e-12, 1.0e-12], dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=1.0e-3,
    )

    raw = _source_plane_bin_loglike(**common)
    stabilized = _source_plane_bin_loglike(max_gain=50.0, **common)

    assert np.isfinite(float(stabilized))
    assert float(stabilized) > float(raw)


def test_source_plane_smooth_residual_cap_bounds_large_residual() -> None:
    capped_x, capped_y = _smooth_residual_cap(
        jnp.asarray([100.0], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        max_residual_arcsec=3.0,
    )
    radius = float(np.hypot(float(np.asarray(capped_x)[0]), float(np.asarray(capped_y)[0])))
    common = _source_plane_loglike_common(
        beta_x=jnp.asarray([0.0, 200.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        outlier_sigma_arcsec=1.0e-3,
    )

    raw = _source_plane_bin_loglike(**common)
    capped = _source_plane_bin_loglike(max_residual_arcsec=3.0, **common)

    assert radius <= 3.0 + 1.0e-10
    assert radius > 2.9
    assert np.isfinite(float(capped))
    assert float(capped) > float(raw)


def test_source_plane_student_t_loss_is_less_punitive_for_large_residual() -> None:
    common = _source_plane_loglike_common(
        beta_x=jnp.asarray([0.0, 10.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        outlier_sigma_arcsec=1.0e-3,
    )

    gaussian = _source_plane_bin_loglike(residual_loss="gaussian", **common)
    student_t = _source_plane_bin_loglike(residual_loss="student-t", student_t_nu=4.0, **common)

    assert np.isfinite(float(student_t))
    assert float(student_t) > float(gaussian)


def test_local_jacobian_loglike_matches_diagonal_weighted_centroid() -> None:
    value = _local_jacobian_bin_loglike(
        beta_x=jnp.asarray([0.0, 2.0], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.ones(2, dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999999, 0.999999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=100.0,
    )

    expected = 2.0 * (-0.5 * (1.0 + np.log((2.0 * np.pi) ** 2)))
    np.testing.assert_allclose(float(value), expected, rtol=0.0, atol=5.0e-6)


def test_local_jacobian_loglike_changes_with_off_diagonal_covariance() -> None:
    common = dict(
        beta_x=jnp.asarray([0.0, 2.0], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.ones(2, dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999999, 0.999999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=100.0,
    )
    diagonal = _local_jacobian_bin_loglike(jac_a01=jnp.zeros(2, dtype=jnp.float64), **common)
    sheared = _local_jacobian_bin_loglike(jac_a01=jnp.asarray([0.5, 0.5], dtype=jnp.float64), **common)

    assert np.isfinite(float(sheared))
    assert not np.isclose(float(diagonal), float(sheared))


def test_jittered_2x2_covariance_det_stabilizes_rank_one_covariance() -> None:
    c00, c11, det = _jittered_2x2_covariance_det(
        jnp.asarray([1.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
    )
    inv00 = c11 / det
    inv11 = c00 / det
    inv01 = -jnp.asarray([1.0], dtype=jnp.float64) / det

    assert float(c00[0]) > 1.0
    assert float(c11[0]) > 1.0
    assert float(det[0]) > cluster_solver.MIN_COVARIANCE_DETERMINANT_GUARD
    assert np.isfinite(float(jnp.log(det[0])))
    assert np.all(np.isfinite(np.asarray([inv00[0], inv01[0], inv11[0]])))


class _TinyCriticalDetEvaluator:
    def __init__(self, det_values: list[float]) -> None:
        self.det_values = np.asarray(det_values, dtype=float)
        self.fit_cosmology_flat_wcdm = False
        self.traced_bin_data = (
            SimpleNamespace(
                effective_z_source=2.0,
                family_ids=("1", "2"),
                family_index_per_image=jnp.asarray([0, 0, 1], dtype=jnp.int32),
                x_obs=jnp.asarray([10.0, 11.0, -5.0], dtype=jnp.float64),
                y_obs=jnp.asarray([20.0, 21.0, 7.0], dtype=jnp.float64),
                effective_z_index=0,
            ),
        )

    def reported_physical_to_latent_parameter_vector(self, best_fit_physical):
        return np.asarray(best_fit_physical, dtype=float)

    def _physical_parameter_vector(self, params):
        return jnp.asarray(params, dtype=jnp.float64)

    def _build_packed_lens_state_with_validity_from_physical(self, _physical_params, _z_source, **_kwargs):
        return {}, {"is_valid": jnp.asarray(True)}

    def _lensing_jacobian_for_components(self, _z_source, _x_obs, _y_obs, _packed_state):
        return (
            jnp.asarray(self.det_values, dtype=jnp.float64),
            jnp.zeros_like(jnp.asarray(self.det_values, dtype=jnp.float64)),
            jnp.zeros_like(jnp.asarray(self.det_values, dtype=jnp.float64)),
            jnp.ones_like(jnp.asarray(self.det_values, dtype=jnp.float64)),
        )


def _tiny_critical_det_state():
    return SimpleNamespace(
        parameter_specs=[],
        family_data=[
            FamilyData(
                family_id="1",
                z_source=2.0,
                effective_z_source=2.0,
                sigma_arcsec=0.1,
                image_labels=["1.1", "1.2"],
                x_obs=np.asarray([10.0, 11.0], dtype=float),
                y_obs=np.asarray([20.0, 21.0], dtype=float),
                reliability=np.ones(2, dtype=float),
            ),
            FamilyData(
                family_id="2",
                z_source=2.3,
                effective_z_source=2.0,
                sigma_arcsec=0.1,
                image_labels=["2.1"],
                x_obs=np.asarray([-5.0], dtype=float),
                y_obs=np.asarray([7.0], dtype=float),
                reliability=np.ones(1, dtype=float),
            ),
        ],
    )


def test_critical_det_diagnostic_flags_only_images_below_threshold() -> None:
    result = cluster_solver._stage3_critical_det_diagnostic_from_evaluator(
        _tiny_critical_det_state(),
        _TinyCriticalDetEvaluator([5.0e-3, 2.0e-2, -9.0e-3]),
        np.asarray([], dtype=float),
        threshold=1.0e-2,
    )

    assert result.total_images == 3
    assert result.min_abs_detA == pytest.approx(5.0e-3)
    assert result.flagged["family_id"].tolist() == ["1", "2"]
    assert result.flagged["image_label"].tolist() == ["1.1", "2.1"]
    np.testing.assert_allclose(result.flagged["detA"].to_numpy(dtype=float), [5.0e-3, -9.0e-3])
    np.testing.assert_allclose(result.flagged["abs_detA"].to_numpy(dtype=float), [5.0e-3, 9.0e-3])


def test_critical_det_diagnostic_empty_table_has_headers(tmp_path: Path) -> None:
    result = cluster_solver._stage3_critical_det_diagnostic_from_evaluator(
        _tiny_critical_det_state(),
        _TinyCriticalDetEvaluator([5.0e-3, 2.0e-2, -9.0e-3]),
        np.asarray([], dtype=float),
        threshold=1.0e-4,
    )
    table_path = cluster_solver._write_critical_det_diagnostic_table(
        result,
        tmp_path / "critical_det_images.csv",
    )

    assert result.flagged.empty
    assert list(result.flagged.columns) == list(cluster_solver.CRITICAL_DET_DIAGNOSTIC_COLUMNS)
    loaded = pd.read_csv(table_path)
    assert loaded.empty
    assert list(loaded.columns) == list(cluster_solver.CRITICAL_DET_DIAGNOSTIC_COLUMNS)


def test_local_jacobian_loglike_stabilizes_near_singular_covariance() -> None:
    value = _local_jacobian_bin_loglike(
        beta_x=jnp.asarray([0.0, 0.01], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.ones(2, dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.9, 0.9], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.ones(2, dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0e-12, 1.0e-12], dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=100.0,
    )

    assert np.isfinite(float(value))


def test_local_jacobian_max_gain_makes_near_critical_covariance_less_punitive() -> None:
    common = dict(
        beta_x=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.ones(2, dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999999, 0.999999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        jac_a00=jnp.zeros(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.zeros(2, dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=1.0e-3,
    )

    raw = _local_jacobian_bin_loglike(**common)
    stabilized = _local_jacobian_bin_loglike(max_gain=50.0, **common)

    assert np.isfinite(float(stabilized))
    assert float(stabilized) > float(raw)


def test_local_jacobian_smooth_residual_cap_is_less_punitive_for_large_residual() -> None:
    common = dict(
        beta_x=jnp.asarray([0.0, 200.0], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999999, 0.999999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=1.0e-3,
    )

    raw = _local_jacobian_bin_loglike(**common)
    capped = _local_jacobian_bin_loglike(max_residual_arcsec=3.0, **common)

    assert np.isfinite(float(capped))
    assert float(capped) > float(raw)


def test_local_jacobian_student_t_loss_is_less_punitive_for_large_residual() -> None:
    common = dict(
        beta_x=jnp.asarray([0.0, 10.0], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999999, 0.999999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=1.0e-3,
    )

    gaussian = _local_jacobian_bin_loglike(residual_loss="gaussian", **common)
    student_t = _local_jacobian_bin_loglike(residual_loss="student-t", student_t_nu=4.0, **common)

    assert np.isfinite(float(student_t))
    assert float(student_t) > float(gaussian)


def test_forward_metric_image_plane_matches_linearized_identity_metric() -> None:
    common = dict(
        sigma_per_image=jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999, 0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=10.0,
        residual_loss="gaussian",
    )
    forward = _forward_metric_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([0.03, -0.02], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.01, 0.04], dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        **common,
    )
    linearized = _linearized_image_plane_bin_loglike(
        residual_x=jnp.asarray([0.03, -0.02], dtype=jnp.float64),
        residual_y=jnp.asarray([0.01, 0.04], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        image_presence_penalty_weight=0.0,
        **common,
    )

    np.testing.assert_allclose(float(forward), float(linearized), rtol=1.0e-7, atol=1.0e-10)


def test_fold_regularized_image_plane_matches_linearized_identity_with_zero_curvature() -> None:
    common = dict(
        sigma_per_image=jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.95, 0.80], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=10.0,
        residual_loss="gaussian",
    )
    residual_x = jnp.asarray([0.03, -0.02], dtype=jnp.float64)
    residual_y = jnp.asarray([0.01, 0.04], dtype=jnp.float64)
    fold_regularized = _fold_regularized_image_plane_bin_loglike(
        residual_beta_x=residual_x,
        residual_beta_y=residual_y,
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        fold_curvature_arcsec_inv=0.0,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        **common,
    )
    linearized = _linearized_image_plane_bin_loglike(
        residual_x=residual_x,
        residual_y=residual_y,
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        image_presence_penalty_weight=0.0,
        **common,
    )

    np.testing.assert_allclose(float(fold_regularized), float(linearized), rtol=1.0e-7, atol=1.0e-10)


def _synthetic_fold_jacobian_entries(x: Any, y: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_array = jnp.asarray(x, dtype=jnp.float64)
    y_array = jnp.asarray(y, dtype=jnp.float64)
    return (
        0.08 + 0.30 * x_array + 0.05 * y_array,
        0.02 + 0.07 * x_array - 0.03 * y_array,
        -0.04 + 0.11 * y_array,
        1.20 + 0.02 * x_array + 0.09 * y_array,
    )


def _reference_fold_signed_curvature(
    x_obs: jnp.ndarray,
    y_obs: jnp.ndarray,
    observed_jacobian_entries: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    fold_frame: tuple[jnp.ndarray, ...],
) -> jnp.ndarray:
    (
        source_critical_x,
        source_critical_y,
        _source_noncritical_x,
        _source_noncritical_y,
        image_critical_x,
        image_critical_y,
        _image_noncritical_x,
        _image_noncritical_y,
        _singular_min,
        _singular_max,
        frame_finite,
    ) = fold_frame
    del observed_jacobian_entries
    step = jnp.asarray(cluster_solver.DEFAULT_FOLD_CURVATURE_FINITE_DIFFERENCE_STEP_ARCSEC, dtype=jnp.float64)
    plus_entries = _synthetic_fold_jacobian_entries(
        x_obs + step * image_critical_x,
        y_obs + step * image_critical_y,
    )
    minus_entries = _synthetic_fold_jacobian_entries(
        x_obs - step * image_critical_x,
        y_obs - step * image_critical_y,
    )
    da00_dcrit = (plus_entries[0] - minus_entries[0]) / (2.0 * step)
    da01_dcrit = (plus_entries[1] - minus_entries[1]) / (2.0 * step)
    da10_dcrit = (plus_entries[2] - minus_entries[2]) / (2.0 * step)
    da11_dcrit = (plus_entries[3] - minus_entries[3]) / (2.0 * step)
    d_a_v_x = da00_dcrit * image_critical_x + da01_dcrit * image_critical_y
    d_a_v_y = da10_dcrit * image_critical_x + da11_dcrit * image_critical_y
    kappa_eff = source_critical_x * d_a_v_x + source_critical_y * d_a_v_y
    return jnp.where(frame_finite & jnp.isfinite(kappa_eff), kappa_eff, jnp.full_like(kappa_eff, jnp.nan))


class _FoldCurvatureProbeEvaluator:
    use_bulk_ray_shooting = True

    def __init__(self) -> None:
        self.calls: list[int] = []

    def _lensing_jacobian_for_components(
        self,
        z_source: float,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: dict[str, Any],
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        del z_source, packed_state, component_indices
        self.calls.append(int(jnp.asarray(x).shape[0]))
        return _synthetic_fold_jacobian_entries(x, y)


def test_fold_signed_curvature_packs_centered_difference_probes() -> None:
    x_obs = jnp.asarray([-0.4, 0.1, 0.9], dtype=jnp.float64)
    y_obs = jnp.asarray([0.2, -0.3, 0.7], dtype=jnp.float64)
    observed_jacobian_entries = _synthetic_fold_jacobian_entries(x_obs, y_obs)
    fold_frame = cluster_solver._fold_regularized_singular_frame_from_jacobian(*observed_jacobian_entries)
    expected = _reference_fold_signed_curvature(x_obs, y_obs, observed_jacobian_entries, fold_frame)
    fake = _FoldCurvatureProbeEvaluator()

    actual = cluster_solver.ClusterJAXEvaluator._fold_signed_curvature_from_observed_jacobian(
        fake,
        2.0,
        x_obs,
        y_obs,
        {},
        observed_jacobian_entries,
        fold_frame=fold_frame,
        fill_value=1.0,
    )

    assert fake.calls == [2 * x_obs.shape[0]]
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1.0e-10, atol=1.0e-10)


def test_fold_signed_curvature_skips_unconstrained_rows_without_changing_loglike() -> None:
    x_obs = jnp.asarray([-0.5, 0.0, 0.6], dtype=jnp.float64)
    y_obs = jnp.asarray([0.4, -0.2, 0.3], dtype=jnp.float64)
    observed_jacobian_entries = _synthetic_fold_jacobian_entries(x_obs, y_obs)
    fold_frame = cluster_solver._fold_regularized_singular_frame_from_jacobian(*observed_jacobian_entries)
    expected_all = _reference_fold_signed_curvature(x_obs, y_obs, observed_jacobian_entries, fold_frame)
    constrained_indices = jnp.asarray([0, 2], dtype=jnp.int32)
    fake = _FoldCurvatureProbeEvaluator()

    skipped = cluster_solver.ClusterJAXEvaluator._fold_signed_curvature_from_observed_jacobian(
        fake,
        2.0,
        x_obs,
        y_obs,
        {},
        observed_jacobian_entries,
        fold_frame=fold_frame,
        image_indices=constrained_indices,
        fill_value=7.0,
    )

    assert fake.calls == [2 * constrained_indices.shape[0]]
    np.testing.assert_allclose(np.asarray(skipped)[[0, 2]], np.asarray(expected_all)[[0, 2]], rtol=1.0e-10, atol=1.0e-10)
    assert float(np.asarray(skipped)[1]) == pytest.approx(7.0)

    common = dict(
        residual_beta_x=jnp.asarray([0.05, 10.0, -0.04], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([-0.02, -10.0, 0.03], dtype=jnp.float64),
        jac_a00=observed_jacobian_entries[0],
        jac_a01=observed_jacobian_entries[1],
        jac_a10=observed_jacobian_entries[2],
        jac_a11=observed_jacobian_entries[3],
        sigma_per_image=jnp.asarray([0.2, 0.2, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.9, 0.9, 0.9], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, False, True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(3, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(3, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=10.0,
        fold_curvature_arcsec_inv=7.0,
        fold_frame=fold_frame,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )
    reference_loglike = _fold_regularized_image_plane_bin_loglike(
        fold_kappa_eff=expected_all,
        **common,
    )
    skipped_loglike = _fold_regularized_image_plane_bin_loglike(
        fold_kappa_eff=skipped,
        **common,
    )

    np.testing.assert_allclose(float(skipped_loglike), float(reference_loglike), rtol=1.0e-10, atol=1.0e-10)


def test_fold_regularized_image_plane_exact_fold_uses_root_distance_exponent() -> None:
    residual = 0.25
    sigma = 1.0
    curvature = 2.0
    value = _fold_regularized_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([residual], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.zeros(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.ones(1, dtype=jnp.float64),
        sigma_per_image=jnp.asarray([sigma], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=1.0e-3,
        fold_curvature_arcsec_inv=curvature,
        fold_kappa_eff=jnp.asarray([curvature], dtype=jnp.float64),
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )

    expected_quad = 2.0 * abs(residual) / (curvature * sigma**2)
    expected = math.log(1.0 - 1.0e-6) - 0.5 * (expected_quad + 2.0 * math.log(2.0 * math.pi * sigma**2))
    np.testing.assert_allclose(float(value), expected, rtol=1.0e-8, atol=1.0e-8)


def test_fold_regularized_image_plane_does_not_include_source_plane_logdet() -> None:
    residual = 0.25
    sigma = 1.0
    curvature = 2.0
    value = _fold_regularized_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([residual], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.zeros(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.ones(1, dtype=jnp.float64),
        sigma_per_image=jnp.asarray([sigma], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=1.0e-3,
        fold_curvature_arcsec_inv=curvature,
        fold_kappa_eff=jnp.asarray([curvature], dtype=jnp.float64),
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )

    expected_quad = 2.0 * abs(residual) / (curvature * sigma**2)
    expected = math.log(1.0 - 1.0e-6) - 0.5 * (expected_quad + 2.0 * math.log(2.0 * math.pi * sigma**2))
    pseudo_covariance_logdet_extra = -0.5 * math.log(0.5 * curvature * sigma**2 * abs(residual))
    np.testing.assert_allclose(float(value), expected, rtol=1.0e-8, atol=1.0e-8)
    assert not math.isclose(float(value), expected + pseudo_covariance_logdet_extra, rel_tol=1.0e-8, abs_tol=1.0e-8)


def test_fold_regularized_image_plane_noncritical_residual_stays_quadratic() -> None:
    residual = 0.20
    sigma = 1.0
    value = _fold_regularized_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([0.0], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([residual], dtype=jnp.float64),
        jac_a00=jnp.zeros(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.ones(1, dtype=jnp.float64),
        sigma_per_image=jnp.asarray([sigma], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=1.0e-3,
        fold_curvature_arcsec_inv=100.0,
        fold_kappa_eff=jnp.asarray([100.0], dtype=jnp.float64),
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )

    expected_quad = residual**2 / sigma**2
    expected = math.log(1.0 - 1.0e-6) - 0.5 * (expected_quad + 2.0 * math.log(2.0 * math.pi * sigma**2))
    np.testing.assert_allclose(float(value), expected, rtol=1.0e-10, atol=1.0e-10)


def test_fold_regularized_image_plane_away_from_criticality_matches_forward_metric() -> None:
    residual_x = jnp.asarray([0.03, -0.02], dtype=jnp.float64)
    residual_y = jnp.asarray([0.01, 0.04], dtype=jnp.float64)
    common = dict(
        sigma_per_image=jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.95, 0.80], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=10.0,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )
    fold_regularized = _fold_regularized_image_plane_bin_loglike(
        residual_beta_x=residual_x,
        residual_beta_y=residual_y,
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        fold_curvature_arcsec_inv=2.0,
        fold_kappa_eff=jnp.asarray([2.0, -3.0], dtype=jnp.float64),
        **common,
    )
    forward = _forward_metric_image_plane_bin_loglike(
        residual_beta_x=residual_x,
        residual_beta_y=residual_y,
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        **common,
    )

    np.testing.assert_allclose(float(fold_regularized), float(forward), rtol=1.0e-6, atol=1.0e-7)


def test_fold_regularized_image_plane_wrong_side_root_falls_back_to_forward_metric() -> None:
    residual = -0.25
    common = dict(
        residual_beta_x=jnp.asarray([residual], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.zeros(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.ones(1, dtype=jnp.float64),
        sigma_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=1.0e-3,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )
    fold_regularized = _fold_regularized_image_plane_bin_loglike(
        fold_curvature_arcsec_inv=2.0,
        fold_kappa_eff=jnp.asarray([2.0], dtype=jnp.float64),
        **common,
    )
    forward = _forward_metric_image_plane_bin_loglike(**common)

    np.testing.assert_allclose(float(fold_regularized), float(forward), rtol=1.0e-10, atol=1.0e-10)


def test_fold_regularized_image_plane_zero_jacobian_has_finite_value_and_gradient() -> None:
    def loglike(rx):
        return _fold_regularized_image_plane_bin_loglike(
            residual_beta_x=jnp.asarray([rx], dtype=jnp.float64),
            residual_beta_y=jnp.asarray([0.2], dtype=jnp.float64),
            jac_a00=jnp.zeros(1, dtype=jnp.float64),
            jac_a01=jnp.zeros(1, dtype=jnp.float64),
            jac_a10=jnp.zeros(1, dtype=jnp.float64),
            jac_a11=jnp.zeros(1, dtype=jnp.float64),
            sigma_per_image=jnp.asarray([1.0], dtype=jnp.float64),
            reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
            image_has_constraint=jnp.asarray([True]),
            image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
            scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
            scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
            covariance_floor=1.0e-6,
            outlier_sigma_arcsec=1.0e-3,
            fold_curvature_arcsec_inv=1.0,
            max_gain=0.0,
            max_residual_arcsec=0.0,
            residual_loss="gaussian",
        )

    value = loglike(jnp.asarray(0.1, dtype=jnp.float64))
    grad = jax.grad(loglike)(jnp.asarray(0.1, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))


def test_fold_regularized_image_plane_near_caustic_has_finite_value_and_gradient() -> None:
    def loglike(rx):
        return _fold_regularized_image_plane_bin_loglike(
            residual_beta_x=jnp.asarray([rx], dtype=jnp.float64),
            residual_beta_y=jnp.asarray([0.05], dtype=jnp.float64),
            jac_a00=jnp.asarray([1.0e-3], dtype=jnp.float64),
            jac_a01=jnp.zeros(1, dtype=jnp.float64),
            jac_a10=jnp.zeros(1, dtype=jnp.float64),
            jac_a11=jnp.ones(1, dtype=jnp.float64),
            sigma_per_image=jnp.asarray([1.0], dtype=jnp.float64),
            reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
            image_has_constraint=jnp.asarray([True]),
            image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
            scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
            scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
            covariance_floor=1.0e-6,
            outlier_sigma_arcsec=1.0e-3,
            fold_curvature_arcsec_inv=2.0,
            fold_kappa_eff=jnp.asarray([2.0], dtype=jnp.float64),
            max_gain=0.0,
            max_residual_arcsec=0.0,
            residual_loss="gaussian",
        )

    value = loglike(jnp.asarray(0.1, dtype=jnp.float64))
    grad = jax.grad(loglike)(jnp.asarray(0.1, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))


def test_forward_metric_image_plane_adds_image_presence_penalty_from_forward_metric() -> None:
    residual_beta_x = jnp.asarray([0.0, 0.03, 0.8], dtype=jnp.float64)
    residual_beta_y = jnp.asarray([0.0, -0.02, 0.0], dtype=jnp.float64)
    jac_a00 = jnp.asarray([2.0, 2.0, 2.0], dtype=jnp.float64)
    jac_a01 = jnp.zeros(3, dtype=jnp.float64)
    jac_a10 = jnp.zeros(3, dtype=jnp.float64)
    jac_a11 = jnp.ones(3, dtype=jnp.float64)
    family_idx = jnp.asarray([0, 0, 0], dtype=jnp.int32)
    reliability = jnp.asarray([0.9, 0.9, 0.9], dtype=jnp.float64)
    common = dict(
        residual_beta_x=residual_beta_x,
        residual_beta_y=residual_beta_y,
        jac_a00=jac_a00,
        jac_a01=jac_a01,
        jac_a10=jac_a10,
        jac_a11=jac_a11,
        family_idx=family_idx,
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=reliability,
        image_has_constraint=jnp.asarray([True, True, True]),
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(3, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(3, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=10.0,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )
    zero_weight = _forward_metric_image_plane_bin_loglike(
        image_presence_penalty_weight=0.0,
        **common,
    )
    with_penalty = _forward_metric_image_plane_bin_loglike(
        image_presence_penalty_weight=3.0,
        image_presence_match_radius_arcsec=0.1,
        image_presence_temperature_arcsec=0.05,
        image_presence_count_softness=0.05,
        image_presence_count_margin=0.0,
        **common,
    )
    presence_residual2, presence_finite = _forward_metric_image_presence_residual2(
        residual_beta_x,
        residual_beta_y,
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
        common["sigma_per_image"],
        common["image_sigma_int"],
        common["covariance_floor"],
        max_gain=0.0,
    )
    expected_penalty = _soft_observed_image_presence_loglike_from_residual2(
        residual2=presence_residual2,
        family_idx=family_idx,
        n_families=1,
        reliability_per_image=reliability,
        image_has_constraint=jnp.asarray([True, True, True]),
        penalty_weight=3.0,
        match_radius_arcsec=0.1,
        temperature_arcsec=0.05,
        count_softness=0.05,
        count_margin=0.0,
    )

    assert bool(np.all(np.asarray(presence_finite)))
    assert float(expected_penalty) < 0.0
    np.testing.assert_allclose(
        float(with_penalty),
        float(zero_weight + expected_penalty),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_forward_metric_image_presence_residual_matches_identity_image_plane() -> None:
    residual_x = jnp.asarray([0.02, -0.04, 0.10], dtype=jnp.float64)
    residual_y = jnp.asarray([0.03, 0.05, -0.20], dtype=jnp.float64)
    residual2, finite = _forward_metric_image_presence_residual2(
        residual_x,
        residual_y,
        jac_a00=jnp.ones(3, dtype=jnp.float64),
        jac_a01=jnp.zeros(3, dtype=jnp.float64),
        jac_a10=jnp.zeros(3, dtype=jnp.float64),
        jac_a11=jnp.ones(3, dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.10, 0.20, 0.30], dtype=jnp.float64),
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        max_gain=0.0,
    )

    assert bool(np.all(np.asarray(finite)))
    np.testing.assert_allclose(
        np.asarray(residual2),
        np.asarray(jnp.square(residual_x) + jnp.square(residual_y)),
        rtol=1.0e-7,
        atol=1.0e-12,
    )


def test_forward_metric_image_presence_is_finite_for_singular_jacobian() -> None:
    value = _forward_metric_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([0.1, -0.2], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.05, 0.3], dtype=jnp.float64),
        jac_a00=jnp.zeros(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.zeros(2, dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.9, 0.9], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=10.0,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
        image_presence_penalty_weight=3.0,
        image_presence_match_radius_arcsec=0.1,
        image_presence_temperature_arcsec=0.05,
        image_presence_count_softness=0.05,
        image_presence_count_margin=0.0,
    )

    assert np.isfinite(float(value))
    assert float(value) > -1.0e20


def test_forward_metric_image_presence_is_finite_for_near_rank_one_covariance() -> None:
    value = _forward_metric_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([0.1, -0.2], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.05, 0.3], dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.ones(2, dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0e-12, 1.0e-12], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.9, 0.9], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=10.0,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
        image_presence_penalty_weight=3.0,
        image_presence_match_radius_arcsec=0.1,
        image_presence_temperature_arcsec=0.05,
        image_presence_count_softness=0.05,
        image_presence_count_margin=0.0,
    )

    assert np.isfinite(float(value))
    assert float(value) > -1.0e20


def test_forward_metric_image_plane_uses_current_jacobian_covariance() -> None:
    common = dict(
        residual_beta_x=jnp.asarray([0.2], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.1], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=10.0,
    )
    identity = _forward_metric_image_plane_bin_loglike(
        jac_a00=jnp.ones(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.ones(1, dtype=jnp.float64),
        **common,
    )
    magnified = _forward_metric_image_plane_bin_loglike(
        jac_a00=jnp.asarray([2.0], dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.asarray([2.0], dtype=jnp.float64),
        **common,
    )

    assert np.isfinite(float(magnified))
    assert not np.isclose(float(identity), float(magnified))


def test_forward_metric_image_plane_stabilizes_near_singular_covariance() -> None:
    value = _forward_metric_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([0.01], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.ones(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.ones(1, dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0e-12], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.9], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=10.0,
    )

    assert np.isfinite(float(value))


def test_linearized_image_plane_residual_identity_jacobian() -> None:
    dx, dy, finite = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([0.2], dtype=jnp.float64),
        f_y=jnp.asarray([-0.1], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
    )

    assert bool(np.asarray(finite)[0])
    assert np.allclose(np.asarray(dx), [-0.2])
    assert np.allclose(np.asarray(dy), [0.1])


def test_linearized_image_plane_default_damped_inverse_is_finite_for_singular_jacobian() -> None:
    dx, dy, finite = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([1.0], dtype=jnp.float64),
        f_y=jnp.asarray([-2.0], dtype=jnp.float64),
        jac_a00=jnp.zeros(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.zeros(1, dtype=jnp.float64),
    )

    assert bool(np.asarray(finite)[0])
    assert np.isfinite(float(np.asarray(dx)[0]))
    assert np.isfinite(float(np.asarray(dy)[0]))
    assert float(np.asarray(dx)[0]) == pytest.approx(0.0)
    assert float(np.asarray(dy)[0]) == pytest.approx(0.0)


def test_linearized_image_plane_damped_inverse_limits_near_critical_gain() -> None:
    default_dx, _, default_finite = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([1.0], dtype=jnp.float64),
        f_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0e-6], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
    )
    dx, dy, finite = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([1.0], dtype=jnp.float64),
        f_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0e-6], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
        max_gain=50.0,
    )

    assert bool(np.asarray(default_finite)[0])
    assert np.isfinite(float(np.asarray(default_dx)[0]))
    assert bool(np.asarray(finite)[0])
    assert np.isfinite(float(np.asarray(dx)[0]))
    assert abs(float(np.asarray(dx)[0])) < 1.0
    assert abs(float(np.asarray(dx)[0])) < 1.0e-3 * abs(float(np.asarray(default_dx)[0]))
    assert float(np.asarray(dy)[0]) == pytest.approx(0.0)


def test_linearized_image_plane_smooth_residual_cap_bounds_large_step() -> None:
    dx, dy, finite = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([100.0], dtype=jnp.float64),
        f_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
        max_residual_arcsec=3.0,
    )

    radius = float(np.hypot(float(np.asarray(dx)[0]), float(np.asarray(dy)[0])))
    assert bool(np.asarray(finite)[0])
    assert radius <= 3.0 + 1.0e-10
    assert radius > 2.9


def test_anchored_solved_image_plane_zero_residual_for_matching_source() -> None:
    class FakeEvaluator:
        anchored_image_plane_solve_steps = 3
        anchored_image_plane_trust_radius_arcsec = 0.3
        anchored_image_plane_lm_damping_relative = 1.0e-3
        anchored_image_plane_lm_damping_absolute = 1.0e-6

        def _ray_shooting_for_components(self, _z_source, x, y, _packed_state):
            return x, y

        def _lensing_jacobian_for_components(self, _z_source, x, y, _packed_state):
            return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(y), jnp.ones_like(y)

    fake = FakeEvaluator()
    residual_method = cluster_solver.ClusterJAXEvaluator._anchored_solved_image_plane_residuals_for_components.__get__(
        fake,
        FakeEvaluator,
    )
    x_obs = jnp.asarray([0.2, -0.4], dtype=jnp.float64)
    y_obs = jnp.asarray([0.1, 0.3], dtype=jnp.float64)
    dx, dy, finite = residual_method(2.0, x_obs, y_obs, x_obs, y_obs, {})

    assert bool(np.all(np.asarray(finite)))
    np.testing.assert_allclose(np.asarray(dx), np.zeros(2), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(dy), np.zeros(2), atol=1.0e-12)


def test_anchored_solved_image_plane_improves_nonlinear_map_over_one_linear_step() -> None:
    class FakeEvaluator:
        anchored_image_plane_solve_steps = 3
        anchored_image_plane_trust_radius_arcsec = 10.0
        anchored_image_plane_lm_damping_relative = 1.0e-8
        anchored_image_plane_lm_damping_absolute = 1.0e-12

        def _ray_shooting_for_components(self, _z_source, x, y, _packed_state):
            return x + x**2, y

        def _lensing_jacobian_for_components(self, _z_source, x, y, _packed_state):
            return 1.0 + 2.0 * x, jnp.zeros_like(x), jnp.zeros_like(y), jnp.ones_like(y)

    fake = FakeEvaluator()
    residual_method = cluster_solver.ClusterJAXEvaluator._anchored_solved_image_plane_residuals_for_components.__get__(
        fake,
        FakeEvaluator,
    )
    x_obs = jnp.asarray([0.5], dtype=jnp.float64)
    y_obs = jnp.asarray([0.0], dtype=jnp.float64)
    target_x = jnp.asarray([0.0], dtype=jnp.float64)
    target_y = jnp.asarray([0.0], dtype=jnp.float64)
    anchored_dx, anchored_dy, anchored_finite = residual_method(2.0, x_obs, y_obs, target_x, target_y, {})

    beta_obs_x, beta_obs_y = fake._ray_shooting_for_components(2.0, x_obs, y_obs, {})
    jacobian = fake._lensing_jacobian_for_components(2.0, x_obs, y_obs, {})
    linear_dx, linear_dy, linear_finite = _linearized_image_plane_residual_from_jacobian(
        beta_obs_x - target_x,
        beta_obs_y - target_y,
        *jacobian,
        max_residual_arcsec=0.0,
    )
    linear_x = x_obs + linear_dx
    linear_y = y_obs + linear_dy
    anchored_x = x_obs + anchored_dx
    anchored_y = y_obs + anchored_dy
    linear_beta_x, linear_beta_y = fake._ray_shooting_for_components(2.0, linear_x, linear_y, {})
    anchored_beta_x, anchored_beta_y = fake._ray_shooting_for_components(2.0, anchored_x, anchored_y, {})
    linear_residual = float(np.hypot(np.asarray(linear_beta_x - target_x)[0], np.asarray(linear_beta_y - target_y)[0]))
    anchored_residual = float(
        np.hypot(np.asarray(anchored_beta_x - target_x)[0], np.asarray(anchored_beta_y - target_y)[0])
    )

    assert bool(np.asarray(linear_finite)[0])
    assert bool(np.asarray(anchored_finite)[0])
    assert anchored_residual < 0.2 * linear_residual


def test_anchored_solved_image_plane_rank_deficient_jacobian_is_finite_and_bounded() -> None:
    dx, dy, finite = _anchored_solved_image_plane_step_from_jacobian(
        f_x=jnp.asarray([100.0], dtype=jnp.float64),
        f_y=jnp.asarray([100.0], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([0.0], dtype=jnp.float64),
        trust_radius_arcsec=0.3,
        lm_damping_relative=1.0e-3,
        lm_damping_absolute=1.0e-6,
    )

    radius = float(np.hypot(float(np.asarray(dx)[0]), float(np.asarray(dy)[0])))
    assert bool(np.asarray(finite)[0])
    assert np.isfinite(radius)
    assert radius <= 0.3 + 1.0e-10


def test_anchored_solved_image_plane_trust_radius_smoothly_bounds_step() -> None:
    dx, dy, finite = _anchored_solved_image_plane_step_from_jacobian(
        f_x=jnp.asarray([100.0], dtype=jnp.float64),
        f_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
        trust_radius_arcsec=0.3,
        lm_damping_relative=1.0e-8,
        lm_damping_absolute=1.0e-12,
    )

    radius = float(np.hypot(float(np.asarray(dx)[0]), float(np.asarray(dy)[0])))
    assert bool(np.asarray(finite)[0])
    assert radius <= 0.3 + 1.0e-10
    assert radius > 0.29


def test_anchored_solved_image_plane_student_t_and_presence_scoring_is_finite() -> None:
    common = dict(
        residual_x=jnp.asarray([2.0, 0.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999, 0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=1.0e-3,
        image_presence_penalty_weight=2.0,
        image_presence_match_radius_arcsec=0.3,
        image_presence_temperature_arcsec=0.1,
    )
    gaussian = _linearized_image_plane_bin_loglike(residual_loss="gaussian", **common)
    student_t = _linearized_image_plane_bin_loglike(residual_loss="student-t", student_t_nu=4.0, **common)

    assert np.isfinite(float(student_t))
    assert float(student_t) > float(gaussian)


def test_linearized_image_plane_loglike_scores_zero_residual() -> None:
    value = _linearized_image_plane_bin_loglike(
        residual_x=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999, 0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=10.0,
        image_presence_penalty_weight=2.0,
    )

    assert np.isfinite(float(value))
    assert float(value) > 0.0


def test_image_plane_scatter_effective_sigma2_ignores_floor_argument() -> None:
    sigma2 = cluster_solver._image_plane_effective_sigma2(
        sigma_per_image=jnp.asarray([0.2], dtype=jnp.float64),
        image_sigma_int=jnp.asarray(0.1, dtype=jnp.float64),
        covariance_floor=0.01,
        image_scatter_floor_arcsec=0.3,
    )

    assert np.asarray(sigma2).tolist() == pytest.approx([0.2**2 + 0.1**2 + 0.01])


def test_linearized_image_plane_loglike_image_plane_scatter_floor_is_support_only() -> None:
    common = dict(
        residual_x=jnp.asarray([0.12, -0.05], dtype=jnp.float64),
        residual_y=jnp.asarray([0.02, 0.09], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999, 0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        covariance_floor=0.01,
        outlier_sigma_arcsec=10.0,
        image_presence_penalty_weight=0.0,
    )

    floor_value = _linearized_image_plane_bin_loglike(
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        image_scatter_floor_arcsec=0.3,
        **common,
    )
    zero_floor_value = _linearized_image_plane_bin_loglike(
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        image_scatter_floor_arcsec=0.0,
        **common,
    )

    assert float(floor_value) == pytest.approx(float(zero_floor_value), abs=1.0e-12)


def test_linearized_image_plane_student_t_loss_is_less_punitive_for_large_residual() -> None:
    common = dict(
        residual_x=jnp.asarray([5.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=1.0e-3,
        image_presence_penalty_weight=0.0,
    )
    gaussian = _linearized_image_plane_bin_loglike(
        residual_loss="gaussian",
        **common,
    )
    student_t = _linearized_image_plane_bin_loglike(
        residual_loss="student-t",
        student_t_nu=4.0,
        **common,
    )

    assert np.isfinite(float(student_t))
    assert float(student_t) > float(gaussian)


def test_soft_observed_image_presence_penalty_is_near_zero_for_present_images() -> None:
    value = _soft_observed_image_presence_loglike(
        residual_x=jnp.asarray([0.0, 0.01, -0.01], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0, -0.01, 0.01], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        n_families=1,
        reliability_per_image=jnp.ones(3, dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True, True]),
        penalty_weight=2.0,
        match_radius_arcsec=0.30,
        temperature_arcsec=0.10,
        count_softness=0.05,
        count_margin=0.05,
    )

    assert np.isfinite(float(value))
    assert -1.0e-3 < float(value) <= 0.0


def test_soft_observed_image_presence_penalty_detects_missing_observed_anchor() -> None:
    value = _soft_observed_image_presence_loglike(
        residual_x=jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        n_families=1,
        reliability_per_image=jnp.ones(3, dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True, True]),
        penalty_weight=2.0,
        match_radius_arcsec=0.30,
        temperature_arcsec=0.10,
        count_softness=0.05,
        count_margin=0.05,
    )

    assert float(value) < -0.5


def test_soft_observed_image_presence_penalty_respects_reliability() -> None:
    common = dict(
        residual_x=jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        n_families=1,
        image_has_constraint=jnp.asarray([True, True, True]),
        penalty_weight=2.0,
        match_radius_arcsec=0.30,
        temperature_arcsec=0.10,
        count_softness=0.05,
        count_margin=0.05,
    )
    high_reliability = _soft_observed_image_presence_loglike(
        reliability_per_image=jnp.ones(3, dtype=jnp.float64),
        **common,
    )
    low_reliability = _soft_observed_image_presence_loglike(
        reliability_per_image=jnp.asarray([1.0, 1.0, 0.1], dtype=jnp.float64),
        **common,
    )

    assert float(low_reliability) > float(high_reliability)


def test_soft_observed_image_presence_penalty_zero_weight_is_neutral() -> None:
    value = _soft_observed_image_presence_loglike(
        residual_x=jnp.asarray([0.0, 0.0, 10.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        n_families=1,
        reliability_per_image=jnp.ones(3, dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True, True]),
        penalty_weight=0.0,
        match_radius_arcsec=0.30,
        temperature_arcsec=0.10,
        count_softness=0.05,
        count_margin=0.05,
    )

    assert float(value) == 0.0


def test_linearized_image_plane_off_diagonal_jacobian_changes_residual() -> None:
    diagonal_dx, diagonal_dy, _ = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([0.2], dtype=jnp.float64),
        f_y=jnp.asarray([0.1], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
    )
    sheared_dx, sheared_dy, finite = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([0.2], dtype=jnp.float64),
        f_y=jnp.asarray([0.1], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.5], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
    )

    assert bool(np.asarray(finite)[0])
    assert not np.allclose(
        np.asarray([diagonal_dx, diagonal_dy]),
        np.asarray([sheared_dx, sheared_dy]),
    )


def test_validation_metrics_summary_reports_finite_aggregates() -> None:
    result = EvaluationResult(
        loglike=-1.0,
        family_predictions={
            "exact_good_a": {
                "exact_image_rms": 0.2,
                "approx_image_rms_arcsec": 0.3,
                "source_plane_rms": 0.02,
            },
            "exact_good_b": {
                "exact_image_rms": 0.4,
                "approx_image_rms_arcsec": 0.5,
                "source_plane_rms": 0.04,
            },
            "failed_exact": {
                "failed": True,
                "exact_image_rms": 9.0,
                "approx_image_rms_arcsec": 0.7,
            },
            "approx_only": {
                "approx_image_rms_arcsec": 0.9,
                "source_plane_rms": 0.08,
            },
            "invalid_metrics": {
                "exact_image_rms": float("nan"),
                "approx_image_rms_arcsec": None,
                "source_plane_rms": "bad",
            },
        },
    )

    summary = _validation_metrics_summary(result)

    assert "validated_families=5" in summary
    assert "exact_families=2" in summary
    assert "exact_image_rms_mean=0.3" in summary
    assert "exact_image_rms_median=0.3" in summary
    assert "approx_image_rms_mean=0.6" in summary
    assert "source_rms_mean=0.04667" in summary
    assert "nan" not in summary.lower()


def test_evaluate_uses_source_summary_without_exact_image_prediction() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    family = SimpleNamespace(family_id="1", z_source=2.0, n_images=3)
    cache = FamilyValidationCache()
    evaluator.state = SimpleNamespace(family_data=[family])
    evaluator.surrogate_enabled = False
    evaluator.quick_diagnostics = False
    evaluator.validation_cache = {"1": cache}
    evaluator.source_loglike = lambda _params: -123.0
    evaluator._family_source_summary = lambda _params: {"1": {"source_plane_rms": 0.2, "failed": False}}

    def fail_exact_prediction(_params, _exact_family):
        raise AssertionError("source-summary evaluation should not solve exact image positions")

    evaluator._exact_family_prediction = fail_exact_prediction

    result = evaluator.evaluate(np.asarray([0.0], dtype=float), validate_all_families=False)

    assert result.loglike == pytest.approx(-123.0)
    assert result.family_predictions["1"]["failed"] is False
    assert result.family_predictions["1"]["approx_image_rms_arcsec"] == pytest.approx(0.2)
    assert result.family_predictions["1"]["used_exact_refresh"] is False
    assert result.family_predictions["1"]["refresh_reason"] == "source_summary"
    assert cache.exact_validation_count == 0
    assert cache.multiplicity_mismatch_count == 0
    assert cache.match_failure_count == 0


def test_image_match_diagnostics_counts_extra_model_images() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.match_tolerance_arcsec = 0.2
    family = SimpleNamespace(
        family_id="1",
        n_images=2,
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0], dtype=float),
    )

    diagnostics = evaluator._image_match_diagnostics(
        np.asarray([0.01, 1.02, 5.0], dtype=float),
        np.asarray([0.01, 0.02, 5.0], dtype=float),
        family,
    )

    assert diagnostics["produced_image_count"] == 3
    assert diagnostics["recovered_image_count"] == 2
    assert diagnostics["missing_image_count"] == 0
    assert diagnostics["extra_image_count"] == 1
    assert diagnostics["multiplicity_failed"] is True
    assert diagnostics["multiplicity_failure_reason"] == "extra_model_images"
    np.testing.assert_array_equal(diagnostics["recovered_image_mask"], np.asarray([True, True], dtype=bool))
    np.testing.assert_allclose(diagnostics["matched_model_x_arcsec"], np.asarray([0.01, 1.02], dtype=float))
    np.testing.assert_allclose(diagnostics["matched_model_y_arcsec"], np.asarray([0.01, 0.02], dtype=float))
    np.testing.assert_allclose(diagnostics["extra_model_x_arcsec"], np.asarray([5.0], dtype=float))
    np.testing.assert_allclose(diagnostics["extra_model_y_arcsec"], np.asarray([5.0], dtype=float))


def test_image_match_diagnostics_recovers_all_observed_with_extra_model_images() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.match_tolerance_arcsec = 0.2
    family = SimpleNamespace(
        family_id="1",
        n_images=3,
        x_obs=np.asarray([0.0, 1.0, 2.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0, 0.0], dtype=float),
    )

    diagnostics = evaluator._image_match_diagnostics(
        np.asarray([0.01, 1.02, 2.03, 8.0, -9.0], dtype=float),
        np.asarray([0.01, 0.02, 0.03, 1.0, -1.0], dtype=float),
        family,
    )

    assert diagnostics["produced_image_count"] == 5
    assert diagnostics["recovered_image_count"] == 3
    assert diagnostics["missing_image_count"] == 0
    assert diagnostics["extra_image_count"] == 2
    assert diagnostics["multiplicity_failed"] is True
    assert diagnostics["multiplicity_failure_reason"] == "extra_model_images"
    np.testing.assert_array_equal(diagnostics["recovered_image_mask"], np.asarray([True, True, True], dtype=bool))
    np.testing.assert_allclose(diagnostics["matched_model_x_arcsec"], np.asarray([0.01, 1.02, 2.03], dtype=float))
    np.testing.assert_allclose(diagnostics["matched_model_y_arcsec"], np.asarray([0.01, 0.02, 0.03], dtype=float))
    np.testing.assert_allclose(diagnostics["extra_model_x_arcsec"], np.asarray([8.0, -9.0], dtype=float))
    np.testing.assert_allclose(diagnostics["extra_model_y_arcsec"], np.asarray([1.0, -1.0], dtype=float))


def test_image_match_diagnostics_counts_missing_model_images() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.match_tolerance_arcsec = 0.2
    family = SimpleNamespace(
        family_id="1",
        n_images=3,
        x_obs=np.asarray([0.0, 1.0, 2.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0, 0.0], dtype=float),
    )

    diagnostics = evaluator._image_match_diagnostics(
        np.asarray([0.01, 2.02], dtype=float),
        np.asarray([0.01, 0.02], dtype=float),
        family,
    )

    assert diagnostics["produced_image_count"] == 2
    assert diagnostics["recovered_image_count"] == 2
    assert diagnostics["missing_image_count"] == 1
    assert diagnostics["extra_image_count"] == 0
    assert diagnostics["multiplicity_failed"] is True
    assert diagnostics["multiplicity_failure_reason"] == "missing_model_images"
    np.testing.assert_array_equal(diagnostics["recovered_image_mask"], np.asarray([True, False, True], dtype=bool))
    np.testing.assert_allclose(diagnostics["matched_model_x_arcsec"], np.asarray([0.01, np.nan, 2.02], dtype=float))
    np.testing.assert_allclose(diagnostics["matched_model_y_arcsec"], np.asarray([0.01, np.nan, 0.02], dtype=float))
    assert diagnostics["extra_model_x_arcsec"].size == 0
    assert diagnostics["extra_model_y_arcsec"].size == 0


def test_image_match_diagnostics_counts_partial_same_multiplicity_match_failure() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.match_tolerance_arcsec = 0.2
    family = SimpleNamespace(
        family_id="1",
        n_images=2,
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0], dtype=float),
    )

    diagnostics = evaluator._image_match_diagnostics(
        np.asarray([0.01, 4.0], dtype=float),
        np.asarray([0.01, 0.0], dtype=float),
        family,
    )

    assert diagnostics["produced_image_count"] == 2
    assert diagnostics["recovered_image_count"] == 1
    assert diagnostics["missing_image_count"] == 1
    assert diagnostics["extra_image_count"] == 1
    assert diagnostics["multiplicity_failed"] is True
    assert diagnostics["multiplicity_failure_reason"] == "match_tolerance_exceeded"
    np.testing.assert_array_equal(diagnostics["recovered_image_mask"], np.asarray([True, False], dtype=bool))
    np.testing.assert_allclose(diagnostics["matched_model_x_arcsec"], np.asarray([0.01, np.nan], dtype=float))
    np.testing.assert_allclose(diagnostics["matched_model_y_arcsec"], np.asarray([0.01, np.nan], dtype=float))
    np.testing.assert_allclose(diagnostics["extra_model_x_arcsec"], np.asarray([4.0], dtype=float))
    np.testing.assert_allclose(diagnostics["extra_model_y_arcsec"], np.asarray([0.0], dtype=float))


def test_exact_image_solver_uses_family_bounding_box_search_center() -> None:
    captured_kwargs: list[dict[str, Any]] = []

    class FakeSolver:
        def image_position_from_source(self, _source_x, _source_y, _kwargs_lens, **kwargs):
            captured_kwargs.append(dict(kwargs))
            return np.asarray([0.0], dtype=float), np.asarray([0.0], dtype=float)

    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.timing_totals = {}
    evaluator._get_exact_model_solver = lambda _z_source: (object(), FakeSolver())
    evaluator._packed_to_kwargs_lens = lambda _packed_state: []
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.1,
        image_labels=["1.a", "1.b", "1.c", "1.d", "1.e"],
        x_obs=np.asarray([7.9, -19.9, -22.2, 9.5, 0.6], dtype=float),
        y_obs=np.asarray([-26.4, 13.1, -0.3, 12.6, 0.9], dtype=float),
        reliability=np.ones(5, dtype=float),
    )

    x_pred, y_pred = evaluator._solve_exact_images_lenstronomy(
        family,
        packed_state={},
        source_x=1.2,
        source_y=-3.4,
    )

    assert x_pred.tolist() == [0.0]
    assert y_pred.tolist() == [0.0]
    assert len(captured_kwargs) == 1
    kwargs = captured_kwargs[0]
    assert kwargs["solver"] == "lenstronomy"
    assert kwargs["min_distance"] == pytest.approx(0.2)
    assert kwargs["search_window"] == pytest.approx(family.search_window)
    assert kwargs["x_center"] == pytest.approx(0.5 * (np.min(family.x_obs) + np.max(family.x_obs)))
    assert kwargs["y_center"] == pytest.approx(0.5 * (np.min(family.y_obs) + np.max(family.y_obs)))


def test_exact_family_prediction_details_reports_solver_failure_counts() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    family = SimpleNamespace(family_id="1", z_source=2.0, n_images=2)
    cache = FamilyValidationCache()
    evaluator.validation_cache = {"1": cache}
    evaluator._build_packed_lens_state = lambda _params, _z_source: {}

    def fail_ray_shooting(_family, _packed_state):
        raise RuntimeError("ray shooting failed")

    evaluator._exact_source_ray_shooting = fail_ray_shooting

    diagnostics = evaluator._exact_family_prediction_details(np.asarray([], dtype=float), family)

    assert diagnostics["failed"] is True
    assert np.isnan(diagnostics["produced_image_count"])
    assert np.isnan(diagnostics["recovered_image_count"])
    assert diagnostics["multiplicity_failed"] is True
    assert diagnostics["multiplicity_failure_reason"] == "source_ray_shooting_failed"
    assert cache.multiplicity_mismatch_count == 1


def test_quick_diagnostics_evaluate_skips_exact_image_prediction() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    family = SimpleNamespace(family_id="1", z_source=2.0, n_images=2)
    evaluator.state = SimpleNamespace(family_data=[family])
    evaluator.surrogate_enabled = False
    evaluator.quick_diagnostics = True
    evaluator.source_loglike = lambda _params: -12.0
    evaluator._family_source_summary = lambda _params: {"1": {"source_plane_rms": 0.25}}

    def fail_exact(_params, _family):
        raise AssertionError("quick diagnostics should not call exact image prediction")

    evaluator._exact_family_prediction = fail_exact

    result = evaluator.evaluate(np.asarray([0.0], dtype=float), validate_all_families=True)

    assert result.loglike == pytest.approx(-12.0)
    assert result.family_predictions["1"]["approx_image_rms_arcsec"] == pytest.approx(0.25)
    assert result.family_predictions["1"]["used_exact_refresh"] is False
    assert result.family_predictions["1"]["refresh_reason"] == "quick_diagnostics"
    assert result.family_predictions["1"]["x_pred"].shape == (2,)
    assert np.isnan(result.family_predictions["1"]["x_pred"]).all()
    assert result.family_predictions["1"]["y_pred"].shape == (2,)
    assert np.isnan(result.family_predictions["1"]["y_pred"]).all()


def test_family_source_summary_handles_zero_measurement_and_source_scatter() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.0,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 0.0]),
        reliability=np.ones(2),
    )
    bin_data = cluster_solver.BinData(
        effective_z_source=2.0,
        family_ids=["1"],
        family_index_per_image=np.asarray([0, 0], dtype=int),
        x_obs=family.x_obs,
        y_obs=family.y_obs,
        sigma_per_image=np.zeros(2, dtype=float),
        reliability_per_image=np.ones(2, dtype=float),
    )
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.state = SimpleNamespace(bin_data=[bin_data], family_data=[family])
    evaluator.traced_bin_data_by_z = {
        2.0: cluster_solver.TracedBinData(
            effective_z_source=2.0,
            family_ids=("1",),
            n_families=1,
            family_index_per_image=jnp.asarray([0, 0], dtype=jnp.int32),
            x_obs=jnp.asarray(family.x_obs, dtype=jnp.float64),
            y_obs=jnp.asarray(family.y_obs, dtype=jnp.float64),
            sigma_per_image=jnp.zeros(2, dtype=jnp.float64),
            reliability_per_image=jnp.ones(2, dtype=jnp.float64),
            image_has_constraint=jnp.asarray([True, True]),
        )
    }
    evaluator.surrogate_enabled = False
    evaluator.source_plane_covariance_floor = 1.0e-6
    evaluator.sample_likelihood_mode = SAMPLE_LIKELIHOOD_SOURCE
    evaluator.likelihood_stabilizer_max_gain = 0.0
    evaluator.image_plane_scatter_floor_arcsec = 1.0e-3
    evaluator._fit_component_indices = lambda: None
    evaluator._physical_parameter_vector = lambda params: params
    evaluator._source_sigma_int_numpy = lambda _params: 0.0
    evaluator._image_sigma_int_numpy = lambda _params: 0.001
    evaluator._build_packed_lens_state = lambda _params, _z_source: {}
    evaluator._packed_lens_validity_from_params = lambda _params, _z_source, stop_gradient=False: {
        "is_valid": jnp.asarray(True),
        "reason_flags": np.zeros(len(cluster_solver.INVALID_STATE_REASON_NAMES), dtype=bool),
    }
    evaluator._record_invalid_state_callback = lambda _flags: None
    evaluator._ray_shooting_for_components = lambda _z_source, _x, _y, _packed_state: (
        jnp.asarray([0.0, 2.0], dtype=jnp.float64),
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
    )
    evaluator._lensing_jacobian_for_components = lambda _z_source, x, y, _packed_state: (
        jnp.ones_like(x),
        jnp.zeros_like(x),
        jnp.zeros_like(y),
        jnp.ones_like(y),
    )
    evaluator._source_position_for_family_numpy = lambda _params, _family_id: None

    summaries = cluster_solver.ClusterJAXEvaluator._family_source_summary(evaluator, np.asarray([], dtype=float))
    summary = summaries["1"]

    assert summary["failed"] is False
    assert summary["source_x"] == pytest.approx(1.0)
    assert summary["source_y"] == pytest.approx(0.0)
    assert summary["source_sigma_eff_arcsec"] == pytest.approx(1.0e-3)
    assert np.isfinite(summary["source_plane_rms"])
    assert summary["source_plane_rms"] == pytest.approx(1.0)


def test_exact_family_prediction_handles_zero_measurement_sigma() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.0,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 0.0]),
        reliability=np.ones(2),
    )

    class FakeModel:
        def ray_shooting(self, _x, _y, _kwargs_lens):
            return (
                jnp.asarray([0.0, 2.0], dtype=jnp.float64),
                jnp.asarray([0.0, 0.0], dtype=jnp.float64),
            )

    class FakeSolver:
        def __init__(self) -> None:
            self.calls: list[tuple[float, float]] = []

        def image_position_from_source(self, source_x, source_y, _kwargs_lens, **_kwargs):
            self.calls.append((float(source_x), float(source_y)))
            return np.asarray([0.0, 1.0], dtype=float), np.asarray([0.0, 0.0], dtype=float)

    solver = FakeSolver()
    cache = FamilyValidationCache()
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.validation_cache = {"1": cache}
    evaluator.source_plane_covariance_floor = 1.0e-6
    evaluator.sample_likelihood_mode = SAMPLE_LIKELIHOOD_SOURCE
    evaluator.timing_totals = {"exact_solver": 0.0}
    evaluator._get_exact_model_solver = lambda _z_source: (FakeModel(), solver)
    evaluator._build_packed_lens_state = lambda _params, _z_source: {}
    evaluator._packed_to_kwargs_lens = lambda _packed_state: []
    evaluator._source_sigma_int_numpy = lambda _params: 0.0
    evaluator._source_position_for_family_numpy = lambda _params, _family_id: None
    evaluator._match_images = lambda x_pred, y_pred, _family: (np.asarray(x_pred, dtype=float), np.asarray(y_pred, dtype=float))

    prediction = cluster_solver.ClusterJAXEvaluator._exact_family_prediction(
        evaluator,
        np.asarray([], dtype=float),
        family,
    )

    assert prediction is not None
    assert solver.calls == [(pytest.approx(1.0), pytest.approx(0.0))]
    assert cache.last_source_x == pytest.approx(1.0)
    assert cache.last_source_y == pytest.approx(0.0)
    assert np.isfinite(cache.source_plane_rms)


def test_sequential_skips_stage3_when_image_plane_mode_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str, str, int, int, bool]] = []
    banners: list[tuple[str, str | None]] = []

    def fake_run_single_stage(args, fit_mode, run_name, **_kwargs):
        calls.append((fit_mode, run_name, args.fit_method, args.warmup, args.samples, bool(args.quick_diagnostics)))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda _args, title, details=None: banners.append((title, details)))
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method="svi+nuts",
        warmup=2000,
        samples=250,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        fit_mode="sequential",
        quick_diagnostics=False,
    )

    cluster_solver._run_sequential(args)

    assert calls == [
        ("large-only", "fit/stage1_large_only", "svi", 2000, 250, True),
        ("joint", "fit/stage2_joint", "svi+nuts", 2000, 250, False),
    ]
    assert [item[0] for item in banners] == ["SEQUENTIAL WORKFLOW", "SEQUENTIAL WORKFLOW COMPLETE"]
    assert "stage3=disabled" in str(banners[0][1])


def test_sequential_fit_cosmology_applies_only_to_image_plane_local_jacobian_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_run_single_stage(args, _fit_mode, run_name, **_kwargs):
        calls.append((run_name, bool(getattr(args, "fit_cosmology_flat_wcdm", False))))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts"],
        warmup=[10, 10],
        samples=[5, 5],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        fit_mode="sequential",
        fit_cosmology_flat_wcdm=True,
    )

    cluster_solver._run_sequential(args)

    assert calls == [
        ("fit/stage1_large_only", False),
        ("fit/stage2_joint", False),
        ("fit/stage3_image_plane", True),
    ]
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["fit_cosmology_flat_wcdm"] is True
    assert summary["stage_cosmology_fit"] == {"stage1": False, "stage2": False, "stage3": True, "stage4": False}
    assert summary["sequential_fiducial_cosmology_config"]["H0"] == pytest.approx(70.0)
    assert summary["sequential_fiducial_cosmology_config"]["Om0"] == pytest.approx(0.3)
    assert "fit_cosmology_all_stages" not in summary


def test_sequential_fit_cosmology_applies_only_to_stage3_and_linearized_stage4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_run_single_stage(args, _fit_mode, run_name, **_kwargs):
        calls.append((run_name, bool(getattr(args, "fit_cosmology_flat_wcdm", False))))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {})
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[10, 10, 0],
        samples=[5, 5, 3],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        fit_mode="sequential",
        fit_cosmology_flat_wcdm=True,
    )

    cluster_solver._run_sequential(args)

    assert calls == [
        ("fit/stage1_large_only", False),
        ("fit/stage2_joint", False),
        ("fit/stage3_image_plane", True),
        ("fit/stage4_linearized_image_plane", True),
    ]


def test_sequential_applies_stage_max_tree_depth_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_run_single_stage(args, _fit_mode, run_name, **_kwargs):
        calls.append((run_name, int(args.max_tree_depth)))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {})
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[10, 10, 0],
        samples=[5, 5, 3],
        max_tree_depth=[10, 8, 6],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        fit_mode="sequential",
    )

    cluster_solver._run_sequential(args)

    assert calls == [
        ("fit/stage1_large_only", 10),
        ("fit/stage2_joint", 10),
        ("fit/stage3_image_plane", 8),
        ("fit/stage4_linearized_image_plane", 6),
    ]
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["stage_fit_controls"]["stage2"]["max_tree_depth"] == 10
    assert summary["stage_fit_controls"]["stage3"]["max_tree_depth"] == 8
    assert summary["stage_fit_controls"]["stage4"]["max_tree_depth"] == 6


def test_sequential_without_fit_cosmology_leaves_local_jacobian_stages_fixed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_run_single_stage(args, _fit_mode, run_name, **_kwargs):
        calls.append((run_name, bool(getattr(args, "fit_cosmology_flat_wcdm", False))))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts"],
        warmup=[10, 10],
        samples=[5, 5],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        fit_mode="sequential",
        fit_cosmology_flat_wcdm=False,
    )

    cluster_solver._run_sequential(args)

    assert calls == [
        ("fit/stage1_large_only", False),
        ("fit/stage2_joint", False),
        ("fit/stage3_image_plane", False),
    ]
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["fit_cosmology_flat_wcdm"] is False
    assert "fit_cosmology_all_stages" not in summary


def test_sequential_without_fit_cosmology_leaves_linearized_stages_fixed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_run_single_stage(args, _fit_mode, run_name, **_kwargs):
        calls.append((run_name, bool(getattr(args, "fit_cosmology_flat_wcdm", False))))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {})
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[10, 10, 0],
        samples=[5, 5, 3],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        fit_mode="sequential",
        fit_cosmology_flat_wcdm=False,
    )

    cluster_solver._run_sequential(args)

    assert calls == [
        ("fit/stage1_large_only", False),
        ("fit/stage2_joint", False),
        ("fit/stage3_image_plane", False),
        ("fit/stage4_linearized_image_plane", False),
    ]


def test_sequential_passes_direct_nuts_to_stage4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_run_single_stage(args, _fit_mode, run_name, **_kwargs):
        calls.append((run_name, str(args.fit_method)))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {})
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "nuts"],
        warmup=[10, 10, 5],
        samples=[5, 5, 3],
        image_plane_mode=IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
        sampling_engine="full",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        skip_stage3_image_plane_local_jacobian=False,
        fit_mode="sequential",
        fit_cosmology_flat_wcdm=False,
    )

    cluster_solver._run_sequential(args)

    assert calls == [
        ("fit/stage1_large_only", "svi"),
        ("fit/stage2_joint", "svi+nuts"),
        ("fit/stage3_image_plane", "svi+nuts"),
        ("fit/stage4_critical_arc_mixture_image_plane", "nuts"),
    ]


def test_stage2_cosmology_initializes_from_stage1_summary_when_fit_cosmology_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-cosmology-flat-wcdm",
        ],
    )
    args = _parse_args()
    stage1_summary = Stage1PriorSummary(
        map_values={
            cluster_solver.COSMOLOGY_OM0_SAMPLE_NAME: 0.42,
            cluster_solver.COSMOLOGY_W0_SAMPLE_NAME: -0.75,
        },
        means={},
        stds={},
    )

    state = cluster_solver._build_state_from_inputs(
        args,
        fit_mode_override="joint",
        stage1_prior_summary=stage1_summary,
    )

    assert state.svi_init_values is not None
    assert state.svi_init_values[cluster_solver.COSMOLOGY_OM0_SAMPLE_NAME] == pytest.approx(0.42)
    assert state.svi_init_values[cluster_solver.COSMOLOGY_W0_SAMPLE_NAME] == pytest.approx(-0.75)


def test_sequential_stage12_cosmology_override_uses_fiducial_fixed_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--fit-cosmology-flat-wcdm",
        ],
    )
    args = _parse_args()
    stage_args = cluster_solver._clone_args(
        args,
        fit_cosmology_flat_wcdm=False,
        cosmology_init_om0=None,
        cosmology_init_w0=None,
        cosmology_config_override=cluster_solver._sequential_fiducial_cosmology_config(),
    )

    state = cluster_solver._build_state_from_inputs(stage_args, fit_mode_override="large-only")

    assert state.fit_cosmology_flat_wcdm is False
    assert state.cosmo_config["class"] == "FlatLambdaCDM"
    assert state.cosmo_config["H0"] == pytest.approx(70.0)
    assert state.cosmo_config["Om0"] == pytest.approx(0.3)
    assert all(spec.component_family != "cosmology" for spec in state.parameter_specs)


def test_sequential_resume_skips_completed_stage1_and_runs_stage2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    rerenders: list[Path] = []
    _touch_complete_stage(tmp_path / "fit" / "stage1_large_only")

    def fake_run_single_stage(args, fit_mode, run_name, **_kwargs):
        calls.append(run_name)
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_rerender_plots", lambda _args, run_dir: rerenders.append(Path(run_dir)))
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method="svi+nuts",
        warmup=2000,
        samples=250,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        fit_mode="sequential",
        resume=True,
        skip_plots=False,
    )

    cluster_solver._run_sequential(args)

    assert rerenders == [tmp_path / "fit" / "stage1_large_only"]
    assert calls == ["fit/stage2_joint"]


def test_sequential_resume_reruns_stage1_with_old_cosmology_sampled_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool, dict[str, Any] | None]] = []
    rerenders: list[Path] = []
    _touch_complete_stage(tmp_path / "fit" / "stage1_large_only")
    (tmp_path / "fit" / "stage1_large_only" / "tables" / "run_summary.json").write_text(
        json.dumps({"fit_cosmology_flat_wcdm": True}),
        encoding="utf-8",
    )

    def fake_run_single_stage(args, fit_mode, run_name, **_kwargs):
        calls.append(
            (
                run_name,
                bool(getattr(args, "fit_cosmology_flat_wcdm", False)),
                getattr(args, "cosmology_config_override", None),
            )
        )
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_rerender_plots", lambda _args, run_dir: rerenders.append(Path(run_dir)))
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method="svi+nuts",
        warmup=2000,
        samples=250,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        fit_mode="sequential",
        resume=True,
        skip_plots=False,
        fit_cosmology_flat_wcdm=True,
    )

    cluster_solver._run_sequential(args)

    assert rerenders == []
    assert [(item[0], item[1]) for item in calls] == [
        ("fit/stage1_large_only", False),
        ("fit/stage2_joint", False),
    ]
    assert calls[0][2]["H0"] == pytest.approx(70.0)
    assert calls[0][2]["Om0"] == pytest.approx(0.3)


def test_sequential_resume_finalizes_checkpointed_stage1_and_runs_stage2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    rerenders: list[Path] = []
    _touch_artifact_checkpoint(tmp_path / "fit" / "stage1_large_only")

    def fake_run_single_stage(args, fit_mode, run_name, **_kwargs):
        calls.append(run_name)
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_rerender_plots", lambda _args, run_dir: rerenders.append(Path(run_dir)))
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method="svi+nuts",
        warmup=2000,
        samples=250,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        fit_mode="sequential",
        resume=True,
        skip_plots=True,
    )

    cluster_solver._run_sequential(args)

    assert rerenders == [tmp_path / "fit" / "stage1_large_only"]
    assert calls == ["fit/stage2_joint"]


def test_nonsequential_resume_refreshes_completed_run_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _touch_complete_stage(tmp_path / "fit")
    rerenders: list[Path] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "fit",
            "--fit-mode",
            "joint",
            "--resume",
        ],
    )
    monkeypatch.setattr(
        cluster_solver,
        "_run_single_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("completed run should be reused")),
    )
    monkeypatch.setattr(cluster_solver, "_rerender_plots", lambda _args, run_dir: rerenders.append(Path(run_dir)))

    cluster_solver.main()

    assert rerenders == [tmp_path / "fit"]


def test_nonsequential_resume_finalizes_checkpointed_run_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _touch_artifact_checkpoint(tmp_path / "fit")
    rerenders: list[Path] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "fit",
            "--fit-mode",
            "joint",
            "--resume",
            "--skip-plots",
        ],
    )
    monkeypatch.setattr(
        cluster_solver,
        "_run_single_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("checkpointed run should be finalized")),
    )
    monkeypatch.setattr(cluster_solver, "_rerender_plots", lambda _args, run_dir: rerenders.append(Path(run_dir)))

    cluster_solver.main()

    assert rerenders == [tmp_path / "fit"]


def test_plots_only_root_passes_final_available_stage_to_rerenders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for stage_name in ["stage2_joint", "stage3_image_plane"]:
        artifact = tmp_path / "fit" / stage_name / "artifacts" / "plot_bundle.h5"
        artifact.parent.mkdir(parents=True)
        artifact.touch()
    rerenders: list[tuple[Path, str | None]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "fit",
            "--plots-only",
        ],
    )
    monkeypatch.setattr(cluster_solver, "_configure_debug_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_rerender_plots",
        lambda _args, stage_dir, exact_diagnostics_stage=None: rerenders.append(
            (Path(stage_dir), None if exact_diagnostics_stage is None else str(exact_diagnostics_stage))
        ),
    )

    cluster_solver.main()

    assert rerenders == [
        (tmp_path / "fit" / "stage2_joint", "stage3_image_plane"),
        (tmp_path / "fit" / "stage3_image_plane", "stage3_image_plane"),
    ]


def test_evidence_ns_main_runs_single_marginal_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str, str, int, str]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "fit",
            "--fit-mode",
            "evidence-ns",
            "--evidence-source-prior-sigma-arcsec",
            "5.0",
            "--skip-plots",
        ],
    )

    def fake_run_single_stage(args, fit_mode, run_name, sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE, **_kwargs):
        calls.append((fit_mode, run_name, args.fit_method, args.samples, sample_likelihood_mode))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(
        cluster_solver,
        "_run_sequential",
        lambda _args: (_ for _ in ()).throw(AssertionError("evidence-ns must not run sequential stages")),
    )

    cluster_solver.main()

    assert calls == [
        ("evidence-ns", "fit", "ns", 0, SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE),
    ]


def test_evidence_ns_main_runs_single_sampled_source_image_plane_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str, int, str]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "fit",
            "--fit-mode",
            "evidence-ns",
            "--evidence-source-prior-sigma-arcsec",
            "5.0",
            "--evidence-likelihood-mode",
            SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
            "--skip-plots",
        ],
    )

    def fake_run_single_stage(args, fit_mode, run_name, sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE, **_kwargs):
        calls.append((fit_mode, run_name, args.fit_method, args.samples, sample_likelihood_mode))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(
        cluster_solver,
        "_run_sequential",
        lambda _args: (_ for _ in ()).throw(AssertionError("evidence-ns must not run sequential stages")),
    )

    cluster_solver.main()

    assert calls == [
        ("evidence-ns", "fit", "ns", 0, SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE),
    ]


def test_evidence_ns_resume_finalizes_checkpointed_run_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _touch_artifact_checkpoint(tmp_path / "fit")
    rerenders: list[Path] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "fit",
            "--fit-mode",
            "evidence-ns",
            "--evidence-source-prior-sigma-arcsec",
            "5.0",
            "--resume",
            "--skip-plots",
        ],
    )
    monkeypatch.setattr(
        cluster_solver,
        "_run_single_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("checkpointed evidence run should be finalized")),
    )
    monkeypatch.setattr(cluster_solver, "_rerender_plots", lambda _args, run_dir: rerenders.append(Path(run_dir)))
    monkeypatch.setattr(
        cluster_solver,
        "_run_sequential",
        lambda _args: (_ for _ in ()).throw(AssertionError("evidence-ns must not run sequential stages")),
    )

    cluster_solver.main()

    assert rerenders == [tmp_path / "fit"]


def test_rerender_plots_banner_includes_stage_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    banners: list[tuple[str, str | None]] = []

    monkeypatch.setattr(cluster_solver, "_configure_debug_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda _args, title, details=None: banners.append((title, details)))
    monkeypatch.setattr(cluster_solver, "_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_run_logged_phase", lambda _args, _phase_name, fn, **_kwargs: fn())
    monkeypatch.setattr(cluster_solver, "_load_artifacts", lambda _artifacts_dir: (_ for _ in ()).throw(RuntimeError("stop after banner")))
    args = argparse.Namespace()
    run_dir = tmp_path / "fit" / "stage3_image_plane"

    with pytest.raises(RuntimeError, match="stop after banner"):
        cluster_solver._rerender_plots(args, run_dir)

    assert banners == [("PLOTS ONLY: STAGE 3: stage3_image_plane", f"run_dir={run_dir}")]


def test_rerender_plots_forces_quick_diagnostics_before_stage4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    plot_quick: list[bool] = []

    class FakeEvaluator:
        def __init__(self, *args: Any, quick_diagnostics: bool = False, **_kwargs: Any) -> None:
            self.quick_diagnostics = bool(quick_diagnostics)
            self.surrogate_enabled = False
            self.timing_totals = {"plot_runtime": 0.0}

        def reported_physical_to_latent_parameter_vector(self, values: np.ndarray) -> np.ndarray:
            return np.asarray(values, dtype=float)

        def refresh_surrogate(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def refresh_scaling_scatter_cache(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def refresh_source_metric_cache(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def evaluate(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
            calls.append("exact")
            return SimpleNamespace(loglike=0.0)

    state = SimpleNamespace(parameter_specs=[], fit_mode="joint")
    arrays = {
        "best_fit": np.empty((0,), dtype=float),
        "samples": np.empty((0, 0), dtype=float),
        "log_prob": np.empty((0,), dtype=float),
        "accept_prob": np.empty((0,), dtype=float),
        "diverging": np.empty((0,), dtype=bool),
        "num_steps": np.empty((0,), dtype=float),
    }
    saved_args = {"quick_diagnostics": False, "warmup": 0, "samples": 0, "chains": 1}

    monkeypatch.setattr(cluster_solver, "_configure_debug_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_run_logged_phase", lambda _args, _phase_name, fn, **_kwargs: fn())
    monkeypatch.setattr(cluster_solver, "_load_artifacts", lambda _artifacts_dir: (state, saved_args, arrays, {}))
    monkeypatch.setattr(cluster_solver, "_log_state_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_maybe_convert_loaded_posterior_arrays_to_physical", lambda arrays, *_args: (arrays, False))
    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_log_solver_active_approximation_warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_approximate_evaluation",
        lambda *_args, **_kwargs: calls.append("approx") or SimpleNamespace(loglike=0.0),
    )
    monkeypatch.setattr(
        cluster_solver,
        "_generate_plots_and_tables",
        lambda **kwargs: plot_quick.append(bool(getattr(kwargs["args"], "quick_diagnostics", False))),
    )
    later_artifact = tmp_path / "fit" / "stage3_image_plane" / "artifacts" / "plot_bundle.h5"
    later_artifact.parent.mkdir(parents=True)
    later_artifact.touch()

    cluster_solver._rerender_plots(argparse.Namespace(quick_diagnostics=False), tmp_path / "fit" / "stage2_joint")

    assert calls == ["approx"]
    assert plot_quick == [True]


def test_rerender_plots_exact_image_diagnostics_stage3_overrides_saved_quick(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    plot_quick: list[bool] = []

    class FakeEvaluator:
        def __init__(self, *args: Any, quick_diagnostics: bool = False, **_kwargs: Any) -> None:
            self.quick_diagnostics = bool(quick_diagnostics)
            self.surrogate_enabled = False
            self.timing_totals = {"plot_runtime": 0.0}

        def reported_physical_to_latent_parameter_vector(self, values: np.ndarray) -> np.ndarray:
            return np.asarray(values, dtype=float)

        def refresh_surrogate(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def refresh_scaling_scatter_cache(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def refresh_source_metric_cache(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def evaluate(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
            calls.append("exact")
            return SimpleNamespace(loglike=0.0)

    state = SimpleNamespace(parameter_specs=[], fit_mode="joint")
    arrays = {
        "best_fit": np.empty((0,), dtype=float),
        "samples": np.empty((0, 0), dtype=float),
        "log_prob": np.empty((0,), dtype=float),
        "accept_prob": np.empty((0,), dtype=float),
        "diverging": np.empty((0,), dtype=bool),
        "num_steps": np.empty((0,), dtype=float),
    }
    saved_args = {"quick_diagnostics": True, "warmup": 0, "samples": 0, "chains": 1}

    monkeypatch.setattr(cluster_solver, "_configure_debug_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_run_logged_phase", lambda _args, _phase_name, fn, **_kwargs: fn())
    monkeypatch.setattr(cluster_solver, "_load_artifacts", lambda _artifacts_dir: (state, saved_args, arrays, {}))
    monkeypatch.setattr(cluster_solver, "_log_state_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_maybe_convert_loaded_posterior_arrays_to_physical", lambda arrays, *_args: (arrays, False))
    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_log_solver_active_approximation_warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_approximate_evaluation",
        lambda *_args, **_kwargs: calls.append("approx") or SimpleNamespace(loglike=0.0),
    )
    monkeypatch.setattr(
        cluster_solver,
        "_generate_plots_and_tables",
        lambda **kwargs: plot_quick.append(bool(getattr(kwargs["args"], "quick_diagnostics", False))),
    )

    cluster_solver._rerender_plots(
        argparse.Namespace(quick_diagnostics=False, exact_image_diagnostics_stage3=True),
        tmp_path / "fit" / "stage3_image_plane",
        exact_diagnostics_stage="stage4_linearized_image_plane",
    )

    assert calls == ["exact"]
    assert plot_quick == [False]


def test_rerender_plots_treats_direct_stage2_as_final_without_later_siblings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    plot_quick: list[bool] = []

    class FakeEvaluator:
        def __init__(self, *args: Any, quick_diagnostics: bool = False, **_kwargs: Any) -> None:
            self.quick_diagnostics = bool(quick_diagnostics)
            self.surrogate_enabled = False
            self.timing_totals = {"plot_runtime": 0.0}

        def reported_physical_to_latent_parameter_vector(self, values: np.ndarray) -> np.ndarray:
            return np.asarray(values, dtype=float)

        def refresh_surrogate(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def refresh_scaling_scatter_cache(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def refresh_source_metric_cache(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def evaluate(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
            calls.append("exact")
            return SimpleNamespace(loglike=0.0)

    state = SimpleNamespace(parameter_specs=[], fit_mode="joint")
    arrays = {
        "best_fit": np.empty((0,), dtype=float),
        "samples": np.empty((0, 0), dtype=float),
        "log_prob": np.empty((0,), dtype=float),
        "accept_prob": np.empty((0,), dtype=float),
        "diverging": np.empty((0,), dtype=bool),
        "num_steps": np.empty((0,), dtype=float),
    }
    saved_args = {"quick_diagnostics": False, "warmup": 0, "samples": 0, "chains": 1}

    monkeypatch.setattr(cluster_solver, "_configure_debug_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_run_logged_phase", lambda _args, _phase_name, fn, **_kwargs: fn())
    monkeypatch.setattr(cluster_solver, "_load_artifacts", lambda _artifacts_dir: (state, saved_args, arrays, {}))
    monkeypatch.setattr(cluster_solver, "_log_state_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_maybe_convert_loaded_posterior_arrays_to_physical", lambda arrays, *_args: (arrays, False))
    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_log_solver_active_approximation_warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_approximate_evaluation",
        lambda *_args, **_kwargs: calls.append("approx") or SimpleNamespace(loglike=0.0),
    )
    monkeypatch.setattr(
        cluster_solver,
        "_generate_plots_and_tables",
        lambda **kwargs: plot_quick.append(bool(getattr(kwargs["args"], "quick_diagnostics", False))),
    )

    cluster_solver._rerender_plots(argparse.Namespace(quick_diagnostics=False), tmp_path / "fit" / "stage2_joint")

    assert calls == ["exact"]
    assert plot_quick == [False]


def test_rerender_plots_keeps_stage4_exact_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    plot_quick: list[bool] = []

    class FakeEvaluator:
        def __init__(self, *args: Any, quick_diagnostics: bool = False, **_kwargs: Any) -> None:
            self.quick_diagnostics = bool(quick_diagnostics)
            self.surrogate_enabled = False
            self.timing_totals = {"plot_runtime": 0.0}

        def reported_physical_to_latent_parameter_vector(self, values: np.ndarray) -> np.ndarray:
            return np.asarray(values, dtype=float)

        def refresh_surrogate(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def refresh_scaling_scatter_cache(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def refresh_source_metric_cache(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def evaluate(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
            calls.append("exact")
            return SimpleNamespace(loglike=0.0)

    state = SimpleNamespace(parameter_specs=[], fit_mode="joint")
    arrays = {
        "best_fit": np.empty((0,), dtype=float),
        "samples": np.empty((0, 0), dtype=float),
        "log_prob": np.empty((0,), dtype=float),
        "accept_prob": np.empty((0,), dtype=float),
        "diverging": np.empty((0,), dtype=bool),
        "num_steps": np.empty((0,), dtype=float),
    }
    saved_args = {"quick_diagnostics": False, "warmup": 0, "samples": 0, "chains": 1}

    monkeypatch.setattr(cluster_solver, "_configure_debug_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_run_logged_phase", lambda _args, _phase_name, fn, **_kwargs: fn())
    monkeypatch.setattr(cluster_solver, "_load_artifacts", lambda _artifacts_dir: (state, saved_args, arrays, {}))
    monkeypatch.setattr(cluster_solver, "_log_state_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_maybe_convert_loaded_posterior_arrays_to_physical", lambda arrays, *_args: (arrays, False))
    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_log_solver_active_approximation_warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_approximate_evaluation",
        lambda *_args, **_kwargs: calls.append("approx") or SimpleNamespace(loglike=0.0),
    )
    monkeypatch.setattr(
        cluster_solver,
        "_generate_plots_and_tables",
        lambda **kwargs: plot_quick.append(bool(getattr(kwargs["args"], "quick_diagnostics", False))),
    )

    cluster_solver._rerender_plots(
        argparse.Namespace(quick_diagnostics=False),
        tmp_path / "fit" / "stage4_linearized_image_plane",
    )

    assert calls == ["exact"]
    assert plot_quick == [False]


def test_sequential_forces_quick_diagnostics_until_stage4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_run_single_stage(args, _fit_mode, run_name, **_kwargs):
        calls.append((run_name, bool(getattr(args, "quick_diagnostics", False))))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {})
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        fit_mode="sequential",
        quick_diagnostics=False,
    )

    cluster_solver._run_sequential(args)

    assert calls == [
        ("fit/stage1_large_only", True),
        ("fit/stage2_joint", True),
        ("fit/stage3_image_plane", True),
        ("fit/stage4_linearized_image_plane", False),
    ]


def test_sequential_exact_image_diagnostics_stage3_keeps_stage3_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_run_single_stage(args, _fit_mode, run_name, **_kwargs):
        calls.append((run_name, bool(getattr(args, "quick_diagnostics", False))))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {})
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        fit_mode="sequential",
        quick_diagnostics=False,
        exact_image_diagnostics_stage3=True,
    )

    cluster_solver._run_sequential(args)

    assert calls == [
        ("fit/stage1_large_only", True),
        ("fit/stage2_joint", True),
        ("fit/stage3_image_plane", False),
        ("fit/stage4_linearized_image_plane", False),
    ]


def test_sequential_skip_stage3_keeps_stage4_exact_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_run_single_stage(args, _fit_mode, run_name, **_kwargs):
        calls.append((run_name, bool(getattr(args, "quick_diagnostics", False))))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {})
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi"],
        warmup=[2000, 0],
        samples=[250, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=True,
        fit_mode="sequential",
        quick_diagnostics=False,
    )

    cluster_solver._run_sequential(args)

    assert calls == [
        ("fit/stage1_large_only", True),
        ("fit/stage2_joint", True),
        ("fit/stage4_linearized_image_plane", False),
    ]


def test_sequential_adds_local_jacobian_stage3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str, str, int, int, str, dict[str, float] | None, dict[str, float] | None, bool]] = []
    banners: list[tuple[str, str | None]] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        previous_stage_best_values=None,
        **_kwargs,
    ):
        calls.append(
            (
                fit_mode,
                run_name,
                args.fit_method,
                args.warmup,
                args.samples,
                sample_likelihood_mode,
                svi_init_physical_values,
                previous_stage_best_values,
                bool(args.quick_diagnostics),
            )
        )
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda _args, title, details=None: banners.append((title, details)))
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {"halo_v_disp": 1100.0})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi"],
        warmup=[2000, 0],
        samples=[250, 100],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        fit_mode="sequential",
        quick_diagnostics=False,
    )

    cluster_solver._run_sequential(args)

    assert [item[1] for item in calls] == ["fit/stage1_large_only", "fit/stage2_joint", "fit/stage3_image_plane"]
    assert calls[0][2:6] == ("svi", 2000, 250, SAMPLE_LIKELIHOOD_SOURCE)
    assert calls[1][2:6] == ("svi+nuts", 2000, 250, SAMPLE_LIKELIHOOD_SOURCE)
    assert calls[2][2:6] == ("svi", 0, 100, SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN)
    assert calls[2][6] == {"halo_v_disp": 1100.0}
    assert calls[1][7] == {"halo_v_disp": 1000.0}
    assert calls[2][7] == {"halo_v_disp": 1100.0}
    assert [item[8] for item in calls] == [True, True, False]
    assert "stage3=enabled" in str(banners[0][1])
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["stage_fit_controls"] == {
        "stage2": {"fit_method": "svi+nuts", "svi_steps": 2000, "warmup": 2000, "samples": 250, "max_tree_depth": 10},
        "stage3": {"fit_method": "svi", "svi_steps": 2000, "warmup": 0, "samples": 100, "max_tree_depth": 10},
        "stage4": {"fit_method": "svi", "svi_steps": 2000, "warmup": 0, "samples": 100, "max_tree_depth": 10},
    }


def test_sequential_start_at_stage3_runs_stage3_without_previous_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[
        tuple[
            str,
            str,
            str,
            int,
            int,
            str,
            Stage1PriorSummary | None,
            dict[str, float] | None,
            dict[str, float] | None,
            bool,
        ]
    ] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        stage1_prior_summary=None,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        previous_stage_best_values=None,
        **_kwargs,
    ):
        calls.append(
            (
                fit_mode,
                run_name,
                args.fit_method,
                args.warmup,
                args.samples,
                sample_likelihood_mode,
                stage1_prior_summary,
                svi_init_physical_values,
                previous_stage_best_values,
                bool(getattr(args, "quick_diagnostics", False)),
            )
        )
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: (_ for _ in ()).throw(AssertionError("stage1 summary should not be loaded")),
    )
    monkeypatch.setattr(
        cluster_solver,
        "_physical_best_fit_values_from_artifacts",
        lambda _artifacts_dir: (_ for _ in ()).throw(AssertionError("stage2 artifacts should not be loaded")),
    )
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi"],
        svi_steps=[1000, 400],
        warmup=[2000, 0],
        samples=[250, 100],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        fit_mode="sequential",
        quick_diagnostics=False,
        start_at_stage3=True,
    )

    cluster_solver._run_sequential(args)

    assert [item[1] for item in calls] == ["fit/stage3_image_plane"]
    assert calls[0][2:6] == ("svi", 0, 100, SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN)
    assert calls[0][6] is None
    assert calls[0][7] is None
    assert calls[0][8] is None
    assert calls[0][9] is False
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["start_at_stage3"] is True
    assert "stage1_run_dir" not in summary
    assert "stage2_run_dir" not in summary
    assert summary["stage3_run_dir"].endswith("stage3_image_plane")


def test_sequential_start_at_stage3_then_stage4_uses_stage3_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[
        tuple[
            str,
            str,
            str,
            int,
            int,
            str,
            Stage1PriorSummary | None,
            dict[str, float] | None,
            dict[str, tuple[float, float]] | None,
            dict[str, float] | None,
            bool,
        ]
    ] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        stage1_prior_summary=None,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        previous_stage_best_values=None,
        **_kwargs,
    ):
        calls.append(
            (
                fit_mode,
                run_name,
                args.fit_method,
                args.warmup,
                args.samples,
                sample_likelihood_mode,
                stage1_prior_summary,
                svi_init_physical_values,
                source_position_prior_values,
                previous_stage_best_values,
                bool(getattr(args, "quick_diagnostics", False)),
            )
        )
        return tmp_path / run_name

    def fake_best_fit(artifacts_dir: Path) -> dict[str, float]:
        assert str(artifacts_dir).endswith("stage3_image_plane/artifacts")
        return {"halo_v_disp": 1200.0}

    def fake_source_priors(artifacts_dir: Path) -> dict[str, tuple[float, float]]:
        assert str(artifacts_dir).endswith("stage3_image_plane/artifacts")
        return {"1": (0.1, -0.2)}

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: (_ for _ in ()).throw(AssertionError("stage1 summary should not be loaded")),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", fake_source_priors)
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        svi_steps=[1000, 400, 200],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_FORWARD_METRIC,
        skip_stage3_image_plane_local_jacobian=False,
        fit_mode="sequential",
        quick_diagnostics=False,
        start_at_stage3=True,
    )

    cluster_solver._run_sequential(args)

    assert [item[1] for item in calls] == ["fit/stage3_image_plane", "fit/stage4_forward_metric_image_plane"]
    assert calls[0][2:6] == ("svi+nuts", 1000, 100, SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN)
    assert calls[0][6] is None
    assert calls[0][7] is None
    assert calls[0][9] is None
    assert calls[0][10] is True
    assert calls[1][2:6] == ("svi", 0, 20, SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE)
    assert calls[1][6] is None
    assert calls[1][7] == {"halo_v_disp": 1200.0}
    assert calls[1][8] == {"1": (0.1, -0.2)}
    assert calls[1][9] == {"halo_v_disp": 1200.0}
    assert calls[1][10] is False
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["start_at_stage3"] is True
    assert "stage1_run_dir" not in summary
    assert "stage2_run_dir" not in summary
    assert summary["stage3_run_dir"].endswith("stage3_image_plane")
    assert summary["stage4_run_dir"].endswith("stage4_forward_metric_image_plane")


def test_sequential_resume_skips_completed_stage1_and_stage2_then_runs_stage3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []
    rerenders: list[Path] = []
    _touch_complete_stage(tmp_path / "fit" / "stage1_large_only")
    _touch_complete_stage(tmp_path / "fit" / "stage2_joint")

    def fake_run_single_stage(args, fit_mode, run_name, sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE, **_kwargs):
        calls.append((run_name, sample_likelihood_mode))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_rerender_plots", lambda _args, run_dir: rerenders.append(Path(run_dir)))
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {"halo_v_disp": 1100.0})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi"],
        warmup=[2000, 0],
        samples=[250, 100],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        fit_mode="sequential",
        resume=True,
        skip_plots=False,
    )

    cluster_solver._run_sequential(args)

    assert rerenders == [tmp_path / "fit" / "stage1_large_only", tmp_path / "fit" / "stage2_joint"]
    assert calls == [("fit/stage3_image_plane", SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN)]


def test_sequential_resume_skip_plots_reuses_without_rerendering(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    _touch_complete_stage(tmp_path / "fit" / "stage1_large_only")

    def fake_run_single_stage(args, fit_mode, run_name, **_kwargs):
        calls.append(run_name)
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(
        cluster_solver,
        "_rerender_plots",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("skip_plots should not rerender solver plots")),
    )
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={}, means={}, stds={}),
    )
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method="svi+nuts",
        warmup=2000,
        samples=250,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        fit_mode="sequential",
        resume=True,
        skip_plots=True,
    )

    cluster_solver._run_sequential(args)

    assert calls == ["fit/stage2_joint"]


def test_sequential_adds_linearized_stage4(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[
        tuple[
            str,
            str,
            str,
            int,
            int,
            str,
            dict[str, float] | None,
            dict[str, tuple[float, float]] | None,
            dict[str, float] | None,
        ]
    ] = []
    banners: list[tuple[str, str | None]] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        previous_stage_best_values=None,
        **_kwargs,
    ):
        calls.append(
            (
                fit_mode,
                run_name,
                args.fit_method,
                args.warmup,
                args.samples,
                sample_likelihood_mode,
                svi_init_physical_values,
                source_position_prior_values,
                previous_stage_best_values,
            )
        )
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda _args, title, details=None: banners.append((title, details)))
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )
    def fake_best_fit(artifacts_dir: Path) -> dict[str, float]:
        text = str(artifacts_dir)
        if text.endswith("stage2_joint/artifacts"):
            return {"halo_v_disp": 1100.0}
        if text.endswith("stage3_image_plane/artifacts"):
            return {"halo_v_disp": 1200.0}
        raise AssertionError(f"unexpected artifacts_dir={artifacts_dir}")

    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {"1": (0.1, -0.2)})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        fit_mode="sequential",
    )

    cluster_solver._run_sequential(args)

    assert [item[1] for item in calls] == [
        "fit/stage1_large_only",
        "fit/stage2_joint",
        "fit/stage3_image_plane",
        "fit/stage4_linearized_image_plane",
    ]
    assert calls[3][2:6] == ("svi", 0, 20, SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE)
    assert calls[2][6] == {"halo_v_disp": 1100.0}
    assert calls[2][8] == {"halo_v_disp": 1100.0}
    assert calls[3][6] == {"halo_v_disp": 1200.0}
    assert calls[3][7] == {"1": (0.1, -0.2)}
    assert calls[3][8] == {"halo_v_disp": 1200.0}
    assert "stage4=enabled" in str(banners[0][1])
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["stage4_run_dir"].endswith("stage4_linearized_image_plane")
    assert summary["image_plane_mode"] == IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA
    assert summary["skip_stage3_image_plane_local_jacobian"] is False


def test_sequential_stage3_triggers_critical_det_before_stage4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str]] = []

    def fake_run_single_stage(
        args,
        _fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        **_kwargs,
    ):
        events.append(("stage", run_name))
        return tmp_path / run_name

    def fake_critical_det(_args, stage3_run_dir):
        events.append(("critical_det", Path(stage3_run_dir).name))
        return cluster_solver.CriticalDetDiagnosticResult(
            flagged=cluster_solver._empty_critical_det_diagnostic_table(),
            min_abs_detA=float("nan"),
            total_images=0,
        )

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_run_stage3_critical_det_diagnostic", fake_critical_det)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )

    def fake_best_fit(artifacts_dir: Path) -> dict[str, float]:
        text = str(artifacts_dir)
        if text.endswith("stage2_joint/artifacts"):
            return {"halo_v_disp": 1100.0}
        if text.endswith("stage3_image_plane/artifacts"):
            return {"halo_v_disp": 1200.0}
        raise AssertionError(f"unexpected artifacts_dir={artifacts_dir}")

    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {"1": (0.1, -0.2)})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        skip_critical_det_diagnostic=False,
        critical_det_diagnostic_threshold=1.0e-2,
        fit_mode="sequential",
    )

    cluster_solver._run_sequential(args)

    assert events == [
        ("stage", "fit/stage1_large_only"),
        ("stage", "fit/stage2_joint"),
        ("stage", "fit/stage3_image_plane"),
        ("critical_det", "stage3_image_plane"),
        ("stage", "fit/stage4_linearized_image_plane"),
    ]


def test_sequential_skip_critical_det_suppresses_stage3_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    diagnostic_calls: list[Path] = []

    def fake_run_single_stage(args, _fit_mode, run_name, **_kwargs):
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {"halo_v_disp": 1200.0})
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {"1": (0.1, -0.2)})
    monkeypatch.setattr(
        cluster_solver,
        "_run_stage3_critical_det_diagnostic",
        lambda _args, stage3_run_dir: diagnostic_calls.append(Path(stage3_run_dir)),
    )
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        skip_critical_det_diagnostic=True,
        critical_det_diagnostic_threshold=1.0e-2,
        fit_mode="sequential",
    )

    cluster_solver._run_sequential(args)

    assert diagnostic_calls == []


def test_sequential_skip_stage3_does_not_attempt_critical_det(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    diagnostic_calls: list[Path] = []

    def fake_run_single_stage(args, _fit_mode, run_name, **_kwargs):
        return tmp_path / run_name

    def fake_best_fit(artifacts_dir):
        assert str(artifacts_dir).endswith("stage2_joint/artifacts")
        return {"halo_v_disp": 1200.0}

    def fake_source_priors(artifacts_dir):
        assert str(artifacts_dir).endswith("stage2_joint/artifacts")
        return {"1": (0.3, 0.4)}

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", fake_source_priors)
    monkeypatch.setattr(
        cluster_solver,
        "_run_stage3_critical_det_diagnostic",
        lambda _args, stage3_run_dir: diagnostic_calls.append(Path(stage3_run_dir)),
    )
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi"],
        warmup=[2000, 0],
        samples=[250, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=True,
        skip_critical_det_diagnostic=False,
        critical_det_diagnostic_threshold=1.0e-2,
        fit_mode="sequential",
    )

    cluster_solver._run_sequential(args)

    assert diagnostic_calls == []


def test_sequential_blocked_linearized_image_plane_runs_as_separate_stage4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str, dict[str, float] | None, dict[str, tuple[float, float]] | None]] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        **_kwargs,
    ):
        calls.append((fit_mode, run_name, sample_likelihood_mode, svi_init_physical_values, source_position_prior_values))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {"halo_v_disp": 1200.0})
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {"1": (0.1, -0.2)})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi+nuts"],
        warmup=[2000, 1000, 10],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED,
        skip_stage3_image_plane_local_jacobian=False,
        fit_mode="sequential",
    )

    cluster_solver._run_sequential(args)

    assert [item[1] for item in calls] == [
        "fit/stage1_large_only",
        "fit/stage2_joint",
        "fit/stage3_image_plane",
        "fit/stage4_blocked_linearized_image_plane",
    ]
    assert calls[3][2] == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
    assert calls[3][3] == {"halo_v_disp": 1200.0}
    assert calls[3][4] == {"1": (0.1, -0.2)}
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["stage4_run_dir"].endswith("stage4_blocked_linearized_image_plane")
    assert summary["image_plane_mode"] == IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED


def test_sequential_forward_metric_image_plane_runs_as_stage4_with_cosmology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[
        tuple[
            str,
            str,
            str,
            bool,
            float | None,
            dict[str, float] | None,
            dict[str, tuple[float, float]] | None,
        ]
    ] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        **_kwargs,
    ):
        calls.append(
            (
                fit_mode,
                run_name,
                sample_likelihood_mode,
                bool(getattr(args, "fit_cosmology_flat_wcdm", False)),
                getattr(args, "evidence_source_prior_sigma_arcsec", None),
                svi_init_physical_values,
                source_position_prior_values,
            )
        )
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )

    def fake_best_fit(artifacts_dir: Path) -> dict[str, float]:
        text = str(artifacts_dir)
        if text.endswith("stage2_joint/artifacts"):
            return {"halo_v_disp": 1100.0}
        if text.endswith("stage3_image_plane/artifacts"):
            return {"halo_v_disp": 1200.0}
        raise AssertionError(f"unexpected artifacts_dir={artifacts_dir}")

    def fake_source_priors(artifacts_dir: Path) -> dict[str, tuple[float, float]]:
        text = str(artifacts_dir)
        if text.endswith("stage3_image_plane/artifacts"):
            return {"1": (0.1, -0.2)}
        raise AssertionError(f"unexpected artifacts_dir={artifacts_dir}")

    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", fake_source_priors)
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_FORWARD_METRIC,
        skip_stage3_image_plane_local_jacobian=False,
        linearized_beta_prior_sigma_arcsec=0.42,
        fit_mode="sequential",
        fit_cosmology_flat_wcdm=True,
    )

    cluster_solver._run_sequential(args)

    assert [item[1] for item in calls] == [
        "fit/stage1_large_only",
        "fit/stage2_joint",
        "fit/stage3_image_plane",
        "fit/stage4_forward_metric_image_plane",
    ]
    assert [item[3] for item in calls] == [False, False, True, True]
    assert calls[3][2] == SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE
    assert calls[3][4] is None
    assert calls[3][5] == {"halo_v_disp": 1200.0}
    assert calls[3][6] == {"1": (0.1, -0.2)}
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["stage4_run_dir"].endswith("stage4_forward_metric_image_plane")
    assert summary["image_plane_mode"] == IMAGE_PLANE_MODE_FORWARD_METRIC
    assert summary["stage_fit_controls"]["stage4"] == {
        "fit_method": "svi",
        "svi_steps": 2000,
        "warmup": 0,
        "samples": 20,
        "max_tree_depth": 10,
    }


def test_sequential_fold_regularized_image_plane_runs_as_stage4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str, dict[str, float] | None, dict[str, tuple[float, float]] | None]] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        **_kwargs,
    ):
        calls.append((fit_mode, run_name, sample_likelihood_mode, svi_init_physical_values, source_position_prior_values))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )

    def fake_best_fit(artifacts_dir: Path) -> dict[str, float]:
        text = str(artifacts_dir)
        if text.endswith("stage2_joint/artifacts"):
            return {"halo_v_disp": 1100.0}
        if text.endswith("stage3_image_plane/artifacts"):
            return {"halo_v_disp": 1200.0}
        raise AssertionError(f"unexpected artifacts_dir={artifacts_dir}")

    def fake_source_priors(artifacts_dir: Path) -> dict[str, tuple[float, float]]:
        text = str(artifacts_dir)
        if text.endswith("stage3_image_plane/artifacts"):
            return {"1": (0.1, -0.2)}
        raise AssertionError(f"unexpected artifacts_dir={artifacts_dir}")

    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", fake_source_priors)
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        linearized_beta_prior_sigma_arcsec=0.42,
        fit_mode="sequential",
        sampling_engine="refreshing_surrogate",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        fold_curvature_arcsec_inv=1.25,
    )

    cluster_solver._run_sequential(args)

    assert [item[1] for item in calls] == [
        "fit/stage1_large_only",
        "fit/stage2_joint",
        "fit/stage3_image_plane",
        "fit/stage4_fold_regularized_image_plane",
    ]
    assert calls[3][2] == SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE
    assert calls[3][3] == {"halo_v_disp": 1200.0}
    assert calls[3][4] == {"1": (0.1, -0.2)}
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["stage4_run_dir"].endswith("stage4_fold_regularized_image_plane")
    assert summary["image_plane_mode"] == IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA
    assert summary["fold_curvature_arcsec_inv"] == pytest.approx(1.25)


def test_sequential_anchored_solved_image_plane_runs_as_stage4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str, dict[str, float] | None, dict[str, tuple[float, float]] | None]] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        **_kwargs,
    ):
        calls.append((fit_mode, run_name, sample_likelihood_mode, svi_init_physical_values, source_position_prior_values))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )

    def fake_best_fit(artifacts_dir: Path) -> dict[str, float]:
        text = str(artifacts_dir)
        if text.endswith("stage2_joint/artifacts"):
            return {"halo_v_disp": 1100.0}
        if text.endswith("stage3_image_plane/artifacts"):
            return {"halo_v_disp": 1200.0}
        raise AssertionError(f"unexpected artifacts_dir={artifacts_dir}")

    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {"1": (0.1, -0.2)})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi+nuts"],
        warmup=[2000, 1000, 10],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        skip_critical_det_diagnostic=True,
        fit_mode="sequential",
        sampling_engine="full",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        anchored_image_plane_solve_steps=3,
        anchored_image_plane_trust_radius_arcsec=0.3,
        anchored_image_plane_lm_damping_relative=1.0e-3,
        anchored_image_plane_lm_damping_absolute=1.0e-6,
    )

    cluster_solver._run_sequential(args)

    assert [item[1] for item in calls] == [
        "fit/stage1_large_only",
        "fit/stage2_joint",
        "fit/stage3_image_plane",
        "fit/stage4_anchored_solved_image_plane",
    ]
    assert calls[3][2] == SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE
    assert calls[3][3] == {"halo_v_disp": 1200.0}
    assert calls[3][4] == {"1": (0.1, -0.2)}
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["stage4_run_dir"].endswith("stage4_anchored_solved_image_plane")
    assert summary["image_plane_mode"] == IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA
    assert summary["anchored_image_plane_solve_steps"] == 3


def test_sequential_critical_arc_mixture_image_plane_runs_as_stage4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str, str, dict[str, float] | None, dict[str, tuple[float, float]] | None]] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        **_kwargs,
    ):
        calls.append(
            (
                fit_mode,
                run_name,
                sample_likelihood_mode,
                args.fit_method,
                svi_init_physical_values,
                source_position_prior_values,
            )
        )
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )

    def fake_best_fit(artifacts_dir: Path) -> dict[str, float]:
        text = str(artifacts_dir)
        if text.endswith("stage2_joint/artifacts"):
            return {"halo_v_disp": 1100.0}
        if text.endswith("stage3_image_plane/artifacts"):
            return {"halo_v_disp": 1200.0}
        raise AssertionError(f"unexpected artifacts_dir={artifacts_dir}")

    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {"1": (0.1, -0.2)})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "smc"],
        warmup=[2000, 1000, 10],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
        skip_stage3_image_plane_local_jacobian=False,
        skip_critical_det_diagnostic=True,
        fit_mode="sequential",
        sampling_engine="refreshing_surrogate",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        critical_arc_critical_direction_sigma_arcsec=4.5,
        critical_arc_base_prob=0.2,
        critical_arc_max_prob=0.7,
        critical_arc_singular_threshold=0.15,
        critical_arc_singular_softness=0.04,
        critical_arc_lm_damping_relative=0.002,
        critical_arc_lm_damping_absolute=1.0e-5,
        critical_arc_lm_trust_radius_arcsec=18.0,
    )

    cluster_solver._run_sequential(args)

    assert [item[1] for item in calls] == [
        "fit/stage1_large_only",
        "fit/stage2_joint",
        "fit/stage3_image_plane",
        "fit/stage4_critical_arc_mixture_image_plane",
    ]
    assert calls[3][2] == SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE
    assert calls[3][3] == "smc"
    assert calls[3][4] == {"halo_v_disp": 1200.0}
    assert calls[3][5] == {"1": (0.1, -0.2)}
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["stage4_run_dir"].endswith("stage4_critical_arc_mixture_image_plane")
    assert summary["image_plane_mode"] == IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE
    assert summary["critical_arc_critical_direction_sigma_arcsec"] == pytest.approx(4.5)
    assert summary["critical_arc_lm_trust_radius_arcsec"] == pytest.approx(18.0)


def test_sequential_linearized_stage4_can_skip_stage3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[
        tuple[str, str, str, dict[str, float] | None, dict[str, tuple[float, float]] | None, dict[str, float] | None]
    ] = []
    banners: list[tuple[str, str | None]] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        previous_stage_best_values=None,
        **_kwargs,
    ):
        calls.append(
            (
                run_name,
                sample_likelihood_mode,
                args.fit_method,
                svi_init_physical_values,
                source_position_prior_values,
                previous_stage_best_values,
            )
        )
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda _args, title, details=None: banners.append((title, details)))
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )

    def fake_best_fit(artifacts_dir):
        assert str(artifacts_dir).endswith("stage2_joint/artifacts")
        return {"halo_v_disp": 1200.0}

    def fake_source_priors(artifacts_dir):
        assert str(artifacts_dir).endswith("stage2_joint/artifacts")
        return {"1": (0.3, 0.4)}

    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", fake_source_priors)
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi"],
        warmup=[2000, 0],
        samples=[250, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=True,
        fit_mode="sequential",
    )

    cluster_solver._run_sequential(args)

    assert [item[0] for item in calls] == [
        "fit/stage1_large_only",
        "fit/stage2_joint",
        "fit/stage4_linearized_image_plane",
    ]
    assert calls[2][1] == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
    assert calls[2][2] == "svi"
    assert calls[2][3] == {"halo_v_disp": 1200.0}
    assert calls[2][4] == {"1": (0.3, 0.4)}
    assert calls[2][5] == {"halo_v_disp": 1200.0}
    assert "stage3=disabled" in str(banners[0][1])
    assert "stage4=enabled" in str(banners[0][1])


def test_sequential_anchored_solved_stage4_can_skip_stage3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str, dict[str, float] | None, dict[str, tuple[float, float]] | None]] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        **_kwargs,
    ):
        calls.append((run_name, sample_likelihood_mode, args.fit_method, svi_init_physical_values, source_position_prior_values))
        return tmp_path / run_name

    def fake_best_fit(artifacts_dir):
        assert str(artifacts_dir).endswith("stage2_joint/artifacts")
        return {"halo_v_disp": 1200.0}

    def fake_source_priors(artifacts_dir):
        assert str(artifacts_dir).endswith("stage2_joint/artifacts")
        return {"1": (0.3, 0.4)}

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", fake_source_priors)
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts"],
        warmup=[2000, 20],
        samples=[250, 20],
        image_plane_mode=IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=True,
        skip_critical_det_diagnostic=False,
        fit_mode="sequential",
        sampling_engine="full",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        anchored_image_plane_solve_steps=3,
        anchored_image_plane_trust_radius_arcsec=0.3,
        anchored_image_plane_lm_damping_relative=1.0e-3,
        anchored_image_plane_lm_damping_absolute=1.0e-6,
    )

    cluster_solver._run_sequential(args)

    assert [item[0] for item in calls] == [
        "fit/stage1_large_only",
        "fit/stage2_joint",
        "fit/stage4_anchored_solved_image_plane",
    ]
    assert calls[2][1] == SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE
    assert calls[2][3] == {"halo_v_disp": 1200.0}
    assert calls[2][4] == {"1": (0.3, 0.4)}
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["stage4_run_dir"].endswith("stage4_anchored_solved_image_plane")


def test_sequential_critical_arc_mixture_stage4_can_skip_stage3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str, dict[str, float] | None, dict[str, tuple[float, float]] | None]] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        **_kwargs,
    ):
        calls.append((run_name, sample_likelihood_mode, args.fit_method, svi_init_physical_values, source_position_prior_values))
        return tmp_path / run_name

    def fake_best_fit(artifacts_dir):
        assert str(artifacts_dir).endswith("stage2_joint/artifacts")
        return {"halo_v_disp": 1200.0}

    def fake_source_priors(artifacts_dir):
        assert str(artifacts_dir).endswith("stage2_joint/artifacts")
        return {"1": (0.3, 0.4)}

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", fake_source_priors)
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts"],
        warmup=[2000, 20],
        samples=[250, 20],
        image_plane_mode=IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
        skip_stage3_image_plane_local_jacobian=True,
        skip_critical_det_diagnostic=False,
        fit_mode="sequential",
        sampling_engine="refreshing_surrogate",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
    )

    cluster_solver._run_sequential(args)

    assert [item[0] for item in calls] == [
        "fit/stage1_large_only",
        "fit/stage2_joint",
        "fit/stage4_critical_arc_mixture_image_plane",
    ]
    assert calls[2][1] == SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE
    assert calls[2][3] == {"halo_v_disp": 1200.0}
    assert calls[2][4] == {"1": (0.3, 0.4)}
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["stage4_run_dir"].endswith("stage4_critical_arc_mixture_image_plane")


def test_sequential_marginal_stage4_can_skip_stage3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[
        tuple[
            str,
            str,
            float | None,
            dict[str, float] | None,
            dict[str, tuple[float, float]] | None,
            dict[str, float] | None,
        ]
    ] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        previous_stage_best_values=None,
        **_kwargs,
    ):
        calls.append(
            (
                run_name,
                sample_likelihood_mode,
                getattr(args, "evidence_source_prior_sigma_arcsec", None),
                svi_init_physical_values,
                source_position_prior_values,
                previous_stage_best_values,
            )
        )
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: Stage1PriorSummary(map_values={"halo_v_disp": 1000.0}, means={}, stds={}),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {"halo_v_disp": 1200.0})
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {"1": (0.3, 0.4)})
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi"],
        warmup=[2000, 0],
        samples=[250, 20],
        image_plane_mode=IMAGE_PLANE_MODE_FORWARD_METRIC,
        skip_stage3_image_plane_local_jacobian=True,
        linearized_beta_prior_sigma_arcsec=0.31,
        fit_mode="sequential",
    )

    cluster_solver._run_sequential(args)

    assert [item[0] for item in calls] == [
        "fit/stage1_large_only",
        "fit/stage2_joint",
        "fit/stage4_forward_metric_image_plane",
    ]
    assert calls[2][1] == SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE
    assert calls[2][2] is None
    assert calls[2][3] == {"halo_v_disp": 1200.0}
    assert calls[2][4] == {"1": (0.3, 0.4)}
    assert calls[2][5] == {"halo_v_disp": 1200.0}


def test_sequential_resume_fast_runs_only_stage4_from_stage3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _touch_stage1_summary(tmp_path / "fit" / "stage1_large_only", {"halo_v_disp": 1000.0})
    _touch_artifact_checkpoint(tmp_path / "fit" / "stage2_joint")
    _touch_artifact_checkpoint(tmp_path / "fit" / "stage3_image_plane")
    calls: list[
        tuple[str, str, str, dict[str, float] | None, dict[str, tuple[float, float]] | None]
    ] = []
    diagnostic_calls: list[Path] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        **_kwargs,
    ):
        calls.append((fit_mode, run_name, sample_likelihood_mode, svi_init_physical_values, source_position_prior_values))
        return tmp_path / run_name

    def fake_best_fit(artifacts_dir):
        assert str(artifacts_dir).endswith("stage3_image_plane/artifacts")
        return {"halo_v_disp": 1300.0}

    def fake_source_priors(artifacts_dir):
        assert str(artifacts_dir).endswith("stage3_image_plane/artifacts")
        return {"1": (0.1, -0.2)}

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", fake_source_priors)
    monkeypatch.setattr(
        cluster_solver,
        "_run_stage3_critical_det_diagnostic",
        lambda _args, stage3_run_dir: diagnostic_calls.append(Path(stage3_run_dir)),
    )

    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "smc"],
        warmup=[2000, 1000, 1],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        skip_critical_det_diagnostic=False,
        fit_mode="sequential",
        resume_fast=True,
        resume=False,
    )

    cluster_solver._run_sequential(args)

    assert calls == [
        (
            "joint",
            "fit/stage4_linearized_image_plane",
            SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
            {"halo_v_disp": 1300.0},
            {"1": (0.1, -0.2)},
        )
    ]
    assert diagnostic_calls == [tmp_path / "fit" / "stage3_image_plane"]
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["resume_fast"] is True
    assert summary["stage2_run_dir"].endswith("stage2_joint")
    assert summary["stage3_run_dir"].endswith("stage3_image_plane")
    assert summary["stage4_run_dir"].endswith("stage4_linearized_image_plane")


def test_sequential_start_at_stage3_resume_fast_runs_only_stage4_from_stage3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _touch_artifact_checkpoint(tmp_path / "fit" / "stage3_image_plane")
    calls: list[
        tuple[str, str, str, dict[str, float] | None, dict[str, tuple[float, float]] | None]
    ] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        **_kwargs,
    ):
        calls.append((fit_mode, run_name, sample_likelihood_mode, svi_init_physical_values, source_position_prior_values))
        return tmp_path / run_name

    def fake_best_fit(artifacts_dir):
        assert str(artifacts_dir).endswith("stage3_image_plane/artifacts")
        return {"halo_v_disp": 1300.0}

    def fake_source_priors(artifacts_dir):
        assert str(artifacts_dir).endswith("stage3_image_plane/artifacts")
        return {"1": (0.1, -0.2)}

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: (_ for _ in ()).throw(AssertionError("stage1 summary should not be loaded")),
    )
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", fake_source_priors)

    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "smc"],
        warmup=[2000, 1000, 1],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        skip_critical_det_diagnostic=True,
        fit_mode="sequential",
        resume_fast=True,
        resume=False,
        start_at_stage3=True,
    )

    cluster_solver._run_sequential(args)

    assert calls == [
        (
            "joint",
            "fit/stage4_linearized_image_plane",
            SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
            {"halo_v_disp": 1300.0},
            {"1": (0.1, -0.2)},
        )
    ]
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["resume_fast"] is True
    assert summary["start_at_stage3"] is True
    assert "stage1_run_dir" not in summary
    assert "stage2_run_dir" not in summary
    assert summary["stage3_run_dir"].endswith("stage3_image_plane")
    assert summary["stage4_run_dir"].endswith("stage4_linearized_image_plane")


def test_sequential_resume_fast_stage4_skip_stage3_uses_stage2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _touch_stage1_summary(tmp_path / "fit" / "stage1_large_only", {"halo_v_disp": 1000.0})
    _touch_artifact_checkpoint(tmp_path / "fit" / "stage2_joint")
    calls: list[tuple[str, dict[str, float] | None, dict[str, tuple[float, float]] | None]] = []

    def fake_run_single_stage(
        args,
        _fit_mode,
        run_name,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        **_kwargs,
    ):
        calls.append((run_name, svi_init_physical_values, source_position_prior_values))
        return tmp_path / run_name

    def fake_best_fit(artifacts_dir):
        assert str(artifacts_dir).endswith("stage2_joint/artifacts")
        return {"halo_v_disp": 1200.0}

    def fake_source_priors(artifacts_dir):
        assert str(artifacts_dir).endswith("stage2_joint/artifacts")
        return {"1": (0.3, 0.4)}

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", fake_source_priors)

    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "smc"],
        warmup=[2000, 1],
        samples=[250, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=True,
        fit_mode="sequential",
        resume_fast=True,
        resume=False,
    )

    cluster_solver._run_sequential(args)

    assert calls == [("fit/stage4_linearized_image_plane", {"halo_v_disp": 1200.0}, {"1": (0.3, 0.4)})]


def test_sequential_resume_fast_runs_only_stage3_from_stage2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _touch_stage1_summary(tmp_path / "fit" / "stage1_large_only", {"halo_v_disp": 1000.0})
    _touch_artifact_checkpoint(tmp_path / "fit" / "stage2_joint")
    calls: list[tuple[str, str, dict[str, float] | None]] = []

    def fake_run_single_stage(args, fit_mode, run_name, svi_init_physical_values=None, **_kwargs):
        calls.append((fit_mode, run_name, svi_init_physical_values))
        return tmp_path / run_name

    def fake_best_fit(artifacts_dir):
        assert str(artifacts_dir).endswith("stage2_joint/artifacts")
        return {"halo_v_disp": 1100.0}

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", fake_best_fit)

    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi"],
        warmup=[2000, 100],
        samples=[250, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        fit_mode="sequential",
        resume_fast=True,
        resume=False,
    )

    cluster_solver._run_sequential(args)

    assert calls == [("joint", "fit/stage3_image_plane", {"halo_v_disp": 1100.0})]


def test_sequential_start_at_stage3_resume_fast_runs_stage3_without_previous_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, dict[str, float] | None, dict[str, float] | None]] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        svi_init_physical_values=None,
        previous_stage_best_values=None,
        **_kwargs,
    ):
        calls.append((fit_mode, run_name, svi_init_physical_values, previous_stage_best_values))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_load_stage1_summary",
        lambda _artifacts_dir: (_ for _ in ()).throw(AssertionError("stage1 summary should not be loaded")),
    )
    monkeypatch.setattr(
        cluster_solver,
        "_physical_best_fit_values_from_artifacts",
        lambda _artifacts_dir: (_ for _ in ()).throw(AssertionError("stage2 artifacts should not be loaded")),
    )

    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi"],
        warmup=[2000, 100],
        samples=[250, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        skip_critical_det_diagnostic=True,
        fit_mode="sequential",
        resume_fast=True,
        resume=False,
        start_at_stage3=True,
    )

    cluster_solver._run_sequential(args)

    assert calls == [("joint", "fit/stage3_image_plane", None, None)]
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["resume_fast"] is True
    assert summary["start_at_stage3"] is True
    assert "stage1_run_dir" not in summary
    assert "stage2_run_dir" not in summary
    assert summary["stage3_run_dir"].endswith("stage3_image_plane")
    assert "stage4_run_dir" not in summary


def test_sequential_resume_fast_runs_only_stage2_from_stage1_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _touch_stage1_summary(tmp_path / "fit" / "stage1_large_only", {"halo_v_disp": 1000.0})
    calls: list[tuple[str, str, dict[str, float] | None]] = []

    def fake_run_single_stage(args, fit_mode, run_name, previous_stage_best_values=None, **_kwargs):
        calls.append((fit_mode, run_name, previous_stage_best_values))
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)

    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method="svi+nuts",
        warmup=2000,
        samples=250,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        fit_mode="sequential",
        resume_fast=True,
        resume=False,
    )

    cluster_solver._run_sequential(args)

    assert calls == [("joint", "fit/stage2_joint", {"halo_v_disp": 1000.0})]


def test_sequential_resume_fast_requires_stage1_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cluster_solver,
        "_run_single_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume-fast should fail before running")),
    )
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method="svi+nuts",
        warmup=2000,
        samples=250,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        fit_mode="sequential",
        resume_fast=True,
        resume=False,
    )

    with pytest.raises(SystemExit, match="stage1 artifacts"):
        cluster_solver._run_sequential(args)


def test_sequential_resume_fast_requires_previous_stage_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _touch_stage1_summary(tmp_path / "fit" / "stage1_large_only", {"halo_v_disp": 1000.0})
    monkeypatch.setattr(
        cluster_solver,
        "_run_single_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume-fast should fail before final stage")),
    )
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi"],
        warmup=[2000, 100],
        samples=[250, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        fit_mode="sequential",
        resume_fast=True,
        resume=False,
    )

    with pytest.raises(SystemExit, match="stage2_joint plot artifacts"):
        cluster_solver._run_sequential(args)


def test_sequential_start_at_stage3_resume_fast_stage4_requires_stage3_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cluster_solver,
        "_run_single_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume-fast should fail before running")),
    )
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "smc"],
        warmup=[2000, 1000, 1],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        fit_mode="sequential",
        resume_fast=True,
        resume=False,
        start_at_stage3=True,
    )

    with pytest.raises(SystemExit, match="stage3_image_plane plot artifacts"):
        cluster_solver._run_sequential(args)


def test_sequential_resume_fast_with_resume_only_rerenders_final_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _touch_stage1_summary(tmp_path / "fit" / "stage1_large_only", {"halo_v_disp": 1000.0})
    _touch_artifact_checkpoint(tmp_path / "fit" / "stage2_joint")
    _touch_artifact_checkpoint(tmp_path / "fit" / "stage3_image_plane")
    _touch_complete_stage(tmp_path / "fit" / "stage4_linearized_image_plane")
    rerenders: list[Path] = []
    diagnostic_calls: list[Path] = []

    monkeypatch.setattr(
        cluster_solver,
        "_run_single_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("final complete stage should resume")),
    )
    monkeypatch.setattr(cluster_solver, "_rerender_plots", lambda _args, run_dir: rerenders.append(Path(run_dir)))
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {"halo_v_disp": 1300.0})
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {"1": (0.1, -0.2)})
    monkeypatch.setattr(
        cluster_solver,
        "_run_stage3_critical_det_diagnostic",
        lambda _args, stage3_run_dir: diagnostic_calls.append(Path(stage3_run_dir)),
    )

    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "smc"],
        warmup=[2000, 1000, 1],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
        skip_critical_det_diagnostic=False,
        fit_mode="sequential",
        resume_fast=True,
        resume=True,
    )

    cluster_solver._run_sequential(args)

    assert rerenders == [tmp_path / "fit" / "stage4_linearized_image_plane"]
    assert diagnostic_calls == [tmp_path / "fit" / "stage3_image_plane"]


def test_adaptive_active_scaling_count_uses_importance_curve() -> None:
    importance = np.asarray([10.0, 4.0, 1.0, 0.2, 0.05, 0.01])

    selected, cumulative_count, knee_count = _adaptive_active_scaling_count(
        importance,
        cumulative_fraction=0.95,
        min_count=2,
        max_count=5,
    )

    assert cumulative_count == 3
    assert 1 <= knee_count <= len(importance)
    assert selected == 3


def test_adaptive_active_scaling_count_respects_cap() -> None:
    importance = np.ones(10)

    selected, _cumulative_count, _knee_count = _adaptive_active_scaling_count(
        importance,
        cumulative_fraction=0.99,
        min_count=2,
        max_count=4,
    )

    assert selected == 4


def test_parameter_recovery_table_exact_truth_has_zero_bias() -> None:
    samples = np.asarray([[1.0, 2.0, 0.1, -0.2], [1.0, 2.0, 0.1, -0.2], [1.0, 2.0, 0.1, -0.2]])
    table = parameter_recovery_table(
        samples,
        ["halo.v_disp", "bcg.v_disp", "source.1.beta_x", "source.1.beta_y"],
        {"halo.v_disp": 1.0, "bcg.v_disp": 2.0, "source.1.beta_x": 0.1, "source.1.beta_y": -0.2},
        best_fit=np.asarray([1.0, 2.0, 0.1, -0.2]),
    )

    np.testing.assert_allclose(table["bias"], 0.0)
    assert table["covered_68"].tolist() == [True, True, True, True]
    np.testing.assert_allclose(table["truth_percentile"], 1.0)


def test_parameter_truth_with_source_positions_derives_legacy_source_truth() -> None:
    truth = {
        "parameter_truth": {"halo.v_disp": 1000.0},
        "sources": [
            {"family_id": "1", "beta_x": 0.25, "beta_y": -0.15},
            {"family_id": "2", "beta_x": "-0.1", "beta_y": "0.3"},
        ],
    }

    values = _parameter_truth_with_source_positions(truth)

    assert values["halo.v_disp"] == 1000.0
    assert values["source.1.beta_x"] == pytest.approx(0.25)
    assert values["source.1.beta_y"] == pytest.approx(-0.15)
    assert values["source.2.beta_x"] == pytest.approx(-0.1)
    assert values["source.2.beta_y"] == pytest.approx(0.3)


def test_log10_abs_parameter_values_handles_signs_and_zero() -> None:
    values = _log10_abs_parameter_values(np.asarray([10.0, 0.1, -0.1, 0.0, 1.0e-6, -1.0e-6], dtype=float))

    np.testing.assert_allclose(
        values,
        np.asarray(
            [
                1.0,
                -1.0,
                -1.0,
                np.log10(PARAMETER_RECOVERY_LOG_ABS_FLOOR),
                np.log10(PARAMETER_RECOVERY_LOG_ABS_FLOOR),
                np.log10(PARAMETER_RECOVERY_LOG_ABS_FLOOR),
            ],
            dtype=float,
        ),
    )


def test_plot_parameter_recovery_writes_log_and_linear_pdfs(tmp_path: Path) -> None:
    parameter_df = pd.DataFrame(
        {
            "parameter": ["halo.v_disp", "source.1.beta_x", "source.1.beta_y", "source.2.beta_x"],
            "truth": [760.0, 0.25, -0.08, 0.0],
            "q16": [700.0, 0.1, -0.2, -0.05],
            "median": [780.0, 0.2, -0.1, 1.0e-6],
            "q84": [820.0, 0.4, -0.02, 0.05],
        }
    )
    log_path = tmp_path / "parameter_recovery_log.pdf"
    linear_path = tmp_path / "parameter_recovery_linear.pdf"

    validation._plot_parameter_recovery(parameter_df, log_path, scale="log_abs")
    validation._plot_parameter_recovery(parameter_df, linear_path, scale="linear")

    assert log_path.exists()
    assert log_path.stat().st_size > 0
    assert linear_path.exists()
    assert linear_path.stat().st_size > 0


def test_plot_parameter_recovery_linear_handles_negative_zero_positive(tmp_path: Path) -> None:
    parameter_df = pd.DataFrame(
        {
            "parameter": ["negative", "zero", "positive"],
            "truth": [-2.0, 0.0, 3.0],
            "q16": [-2.5, -0.1, 2.5],
            "median": [-2.1, 0.0, 3.1],
            "q84": [-1.8, 0.1, 3.6],
        }
    )
    path = tmp_path / "parameter_recovery_linear.pdf"

    validation._plot_parameter_recovery(parameter_df, path, scale="linear")

    assert path.exists()
    assert path.stat().st_size > 0


def test_magnification_recovery_table_handles_sign_and_small_truth() -> None:
    truth = pd.DataFrame(
        {
            "image_label": ["1.1", "1.2", "1.3"],
            "magnification_true": [2.0, -4.0, 0.0],
        }
    )
    recovered = pd.DataFrame(
        {
            "image_label": ["1.1", "1.2", "1.3"],
            "magnification_recovered": [2.5, 4.0, 0.1],
        }
    )

    table = magnification_recovery_table(truth, recovered, epsilon=0.5)

    assert table["parity_match"].tolist() == [True, False, False]
    np.testing.assert_allclose(table["abs_magnification_fractional_error"], [0.25, 0.0, 0.2])


def test_annular_surface_density_helper_returns_finite_positive_values() -> None:
    model = validation.LensModel(lens_model_list=["SIS"], z_lens=0.4, z_source=2.0)
    values = validation._annular_surface_density_msun_per_arcsec2(
        model,
        [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}],
        [0],
        np.asarray([2.0, 5.0], dtype=float),
        sigma_crit_angle=10.0,
        n_radial=8,
        n_azimuth=12,
    )

    assert values.shape == (2,)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)
    assert values[0] > values[1]


def test_capped_evenly_spaced_posterior_draws_preserves_small_inputs_and_spans_large_inputs() -> None:
    fewer = np.arange(20, dtype=float).reshape(10, 2)
    exact = np.arange(256, dtype=float).reshape(128, 2)
    larger = np.arange(400, dtype=float).reshape(200, 2)

    np.testing.assert_array_equal(
        validation._capped_evenly_spaced_posterior_draws(fewer),
        fewer,
    )
    np.testing.assert_array_equal(
        validation._capped_evenly_spaced_posterior_draws(exact),
        exact,
    )

    capped = validation._capped_evenly_spaced_posterior_draws(larger)
    expected_indices = np.linspace(
        0,
        larger.shape[0] - 1,
        validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
        dtype=int,
    )
    np.testing.assert_array_equal(capped, larger[expected_indices])
    assert capped.shape == (validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP, 2)
    np.testing.assert_array_equal(capped[0], larger[0])
    np.testing.assert_array_equal(capped[-1], larger[-1])


def test_validation_recovery_progress_tracks_parent_and_subtasks(monkeypatch: pytest.MonkeyPatch) -> None:
    progress_instances = _install_recording_progress(monkeypatch)

    with validation._ValidationRecoveryProgress(argparse.Namespace(quiet=False)) as progress:
        progress.begin_phase("posterior uncertainty")
        subtask = progress.add_subtask("posterior uncertainty: draws x families", total=6)
        progress.update_subtask(subtask, "posterior uncertainty: draw=1/3 family=1 z=2.0000 failed_exact=0")
        progress.advance_subtask(subtask)
        progress.advance_phase()

    assert len(progress_instances) == 1
    recorder = progress_instances[0]
    assert recorder.kwargs["transient"] is True
    assert ("add_task", 1, "recovery: starting", 0) in recorder.events
    assert (
        "update",
        1,
        {"total": 1, "description": "recovery: posterior uncertainty"},
    ) in recorder.events
    assert ("add_task", 2, "posterior uncertainty: draws x families", 6) in recorder.events
    assert ("advance", 2, 1) in recorder.events
    assert ("advance", 1, 1) in recorder.events
    assert ("update", 1, {"description": "recovery: complete"}) in recorder.events

    quiet_instances = _install_recording_progress(monkeypatch)
    with validation._ValidationRecoveryProgress(argparse.Namespace(quiet=True)) as quiet_progress:
        quiet_progress.begin_phase("load inputs")
        quiet_subtask = quiet_progress.add_subtask("unused", total=1)
        quiet_progress.advance_subtask(quiet_subtask)
        quiet_progress.advance_phase()

    assert quiet_instances == []


def test_validation_mock_progress_tracks_redshifts_families_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    progress_instances = _install_recording_progress(monkeypatch)
    messages: list[str] = []
    monkeypatch.setattr(validation, "_log", lambda _args, message: messages.append(message))

    with validation._ValidationMockProgress(argparse.Namespace(quiet=False)) as progress:
        progress.callback(
            "subhalos_complete",
            {
                "requested_subhalos": 100,
                "selected_subhalos": 100,
                "parent_count": 500,
                "observable_count": 420,
                "retry_count": 0,
                "mag_faint_limit": 24.0,
            },
        )
        progress.callback(
            "redshift_start",
            {
                "z_source": 2.0,
                "redshift_index": 1,
                "redshift_count": 2,
                "lens_component_count": 102,
                "caustic_grid_pixels": 801,
            },
        )
        progress.callback(
            "redshift_complete",
            {
                "z_source": 2.0,
                "redshift_index": 1,
                "redshift_count": 2,
                "lens_component_count": 102,
                "caustic_grid_pixels": 801,
                "caustic_compute_window_arcsec": 160.0,
                "caustic_grid_scale_arcsec": 0.2,
                "caustic_count": 3,
            },
        )
        progress.callback(
            "family_start",
            {
                "family_index": 1,
                "family_count": 3,
                "caustic_class": "primary",
                "z_source": 2.0,
                "max_attempts": 100,
            },
        )
        progress.callback(
            "family_attempt",
            {
                "family_index": 1,
                "family_count": 3,
                "caustic_class": "primary",
                "z_source": 2.0,
                "attempt": 25,
                "max_attempts": 100,
                "image_count": 2,
            },
        )
        progress.callback(
            "family_accept",
            {
                "family_index": 1,
                "family_count": 3,
                "caustic_class": "primary",
                "z_source": 2.0,
                "attempt": 26,
                "max_attempts": 100,
                "image_count": 4,
                "caustic_index": 0,
            },
        )
        progress.callback(
            "outputs_complete",
            {
                "image_count": 12,
                "par_path": "mock/single_bcg_mock.par",
                "image_catalog_path": "mock/obs_arcs.cat",
                "truth_path": "mock/truth.json",
            },
        )

    assert len(progress_instances) == 1
    recorder = progress_instances[0]
    assert recorder.kwargs["transient"] is True
    assert ("add_task", 1, "mock caustics: starting", 2) in recorder.events
    assert ("add_task", 2, "mock families: starting", 3) in recorder.events
    assert ("advance", 1, 1) in recorder.events
    assert ("advance", 2, 1) in recorder.events
    assert any(
        event[0] == "update" and "attempt=25/100" in event[2].get("description", "")
        for event in recorder.events
    )
    assert any("mock subhalos selected=100/100" in message for message in messages)
    assert not any("mock caustics" in message for message in messages)
    assert not any("mock family" in message for message in messages)
    assert any("mock outputs complete images=12" in message for message in messages)


def test_validation_mock_progress_quiet_suppresses_rich_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    progress_instances = _install_recording_progress(monkeypatch)
    messages: list[str] = []
    monkeypatch.setattr(validation, "_log", lambda _args, message: messages.append(message))

    with validation._ValidationMockProgress(argparse.Namespace(quiet=True)) as progress:
        progress.callback(
            "family_attempt",
            {
                "family_index": 1,
                "family_count": 1,
                "caustic_class": "primary",
                "z_source": 2.0,
                "attempt": 25,
                "max_attempts": 100,
                "image_count": 2,
            },
        )

    assert progress_instances == []
    assert messages == []


def test_recovered_model_tables_reports_partial_image_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    families = [
        SimpleNamespace(
            family_id="1",
            z_source=2.0,
            effective_z_source=2.0,
            sigma_arcsec=0.1,
            n_images=2,
            image_labels=["1.1", "1.2"],
            x_obs=np.asarray([0.0, 1.0], dtype=float),
            y_obs=np.asarray([0.0, 0.0], dtype=float),
        ),
        SimpleNamespace(
            family_id="2",
            z_source=3.0,
            effective_z_source=3.0,
            sigma_arcsec=0.1,
            n_images=3,
            image_labels=["2.1", "2.2", "2.3"],
            x_obs=np.asarray([2.0, 3.0, 4.0], dtype=float),
            y_obs=np.asarray([0.0, 0.0, 0.0], dtype=float),
        ),
    ]
    state = SimpleNamespace(parameter_specs=[], family_data=families)
    images = pd.DataFrame(
        {
            "family_id": ["1", "1", "2", "2", "2"],
            "image_label": ["1.1", "1.2", "2.1", "2.2", "2.3"],
            "x_obs_arcsec": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y_obs_arcsec": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    class FakeModel:
        def magnification(self, x, y, kwargs_lens):
            del y, kwargs_lens
            return np.asarray(x, dtype=float) + 10.0

    class FakeEvaluator:
        source_plane_covariance_floor = 0.0

        def __init__(self, *args, **kwargs) -> None:
            self.state = kwargs["state"]

        def reported_physical_to_latent_parameter_vector(self, sample):
            return np.asarray(sample, dtype=float)

        def _image_sigma_int_numpy(self, _sample_latent):
            return 0.2

        def evaluate(self, _sample_latent):
            return SimpleNamespace(
                family_predictions={
                    "1": {"source_x": 0.1, "source_y": 0.2, "source_plane_rms": 0.3, "failed": False},
                    "2": {"source_x": 1.1, "source_y": 1.2, "source_plane_rms": 1.3, "failed": False},
                }
            )

        def _get_exact_model_solver(self, _z_source):
            return FakeModel(), None

        def _build_packed_lens_state(self, _sample_latent, _z_source):
            return {}

        def _packed_to_kwargs_lens(self, _packed_state):
            return []

        def _exact_family_prediction_details(self, _sample_latent, family):
            if str(family.family_id) == "1":
                return {
                    "produced_image_count": 2,
                    "recovered_image_count": 2,
                    "missing_image_count": 0,
                    "extra_image_count": 0,
                    "multiplicity_failed": False,
                    "multiplicity_failure_reason": "",
                    "matched_model_x_arcsec": np.asarray([0.05, 1.05], dtype=float),
                    "matched_model_y_arcsec": np.asarray([0.0, 0.0], dtype=float),
                    "recovered_image_mask": np.asarray([True, True], dtype=bool),
                    "failed": False,
                    "exact_image_rms": 0.05,
                }
            return {
                "produced_image_count": 3,
                "recovered_image_count": 2,
                "missing_image_count": 1,
                "extra_image_count": 1,
                "multiplicity_failed": True,
                "multiplicity_failure_reason": "match_tolerance_exceeded",
                "matched_model_x_arcsec": np.asarray([2.1, np.nan, 4.2], dtype=float),
                "matched_model_y_arcsec": np.asarray([0.0, np.nan, 0.0], dtype=float),
                "recovered_image_mask": np.asarray([True, False, True], dtype=bool),
                "extra_model_x_arcsec": np.asarray([8.0], dtype=float),
                "extra_model_y_arcsec": np.asarray([-2.0], dtype=float),
                "failed": True,
            }

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)

    magnification_df, image_df, source_df = validation._recovered_model_tables(
        state,
        np.asarray([0.0], dtype=float),
        images,
    )

    assert len(image_df) == 5
    indexed = image_df.set_index("image_label")
    assert indexed["image_recovery_status"].tolist() == ["recovered", "recovered", "recovered", "not_recovered", "recovered"]
    assert indexed["exact_image_prediction_failed"].tolist() == [False, False, False, True, False]
    assert indexed.loc["2.1", "x_model_arcsec"] == pytest.approx(2.1)
    assert np.isnan(indexed.loc["2.2", "x_model_arcsec"])
    assert indexed.loc["2.3", "image_residual_arcsec"] == pytest.approx(0.2)
    assert int(indexed.loc["2.1", "model_recovered_image_count"]) == 2
    assert int(indexed.loc["2.1", "model_missing_image_count"]) == 1
    assert int(indexed.loc["2.1", "model_extra_image_count"]) == 1
    assert magnification_df.set_index("image_label").loc["2.3", "magnification_recovered"] == pytest.approx(14.0)
    assert source_df.set_index("family_id").loc["1", "exact_image_rms_arcsec"] == pytest.approx(0.05)


def test_posterior_prediction_uncertainty_tables_advances_progress_per_draw_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeExactModel:
        def magnification(self, x, y, kwargs_lens):
            return np.ones_like(np.asarray(x, dtype=float))

    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            self.state = kwargs["state"]

        def reported_physical_to_latent_parameter_vector(self, sample):
            return np.asarray(sample, dtype=float)

        def _family_source_summary(self, sample_latent):
            return {
                str(family.family_id): {
                    "x_pred": family.x_obs + 0.1,
                    "y_pred": family.y_obs + 0.2,
                    "source_x": 0.01 * float(family.family_id),
                    "source_y": 0.02 * float(family.family_id),
                    "source_plane_rms": 0.03,
                }
                for family in self.state.family_data
            }

        def _get_exact_model_solver(self, z_source):
            return FakeExactModel(), None

        def _build_packed_lens_state(self, sample_latent, z_source):
            return {}

        def _packed_to_kwargs_lens(self, packed_state):
            return []

        def _exact_family_prediction(self, sample_latent, family):
            return family.x_obs + 0.01, family.y_obs + 0.02, 0.03

    class FakeProgress:
        def __init__(self) -> None:
            self.added: list[tuple[str, int | None]] = []
            self.updated: list[str] = []
            self.advanced: list[int | None] = []

        def add_subtask(self, description: str, total: int | None) -> int:
            self.added.append((description, total))
            return 17

        def update_subtask(self, task_id: int | None, description: str) -> None:
            self.updated.append(description)

        def advance_subtask(self, task_id: int | None) -> None:
            self.advanced.append(task_id)

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_latent", lambda sample, _specs: np.asarray(sample, dtype=float))
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 1)
    state = SimpleNamespace(
        parameter_specs=[],
        family_data=[
            SimpleNamespace(
                family_id="1",
                z_source=2.0,
                n_images=1,
                image_labels=["1.1"],
                x_obs=np.asarray([0.0], dtype=float),
                y_obs=np.asarray([0.0], dtype=float),
            ),
            SimpleNamespace(
                family_id="2",
                z_source=3.0,
                n_images=1,
                image_labels=["2.1"],
                x_obs=np.asarray([1.0], dtype=float),
                y_obs=np.asarray([1.0], dtype=float),
            ),
        ],
    )
    images = pd.DataFrame(
        {
            "family_id": ["1", "2"],
            "image_label": ["1.1", "2.1"],
            "x_obs_arcsec": [0.0, 1.0],
            "y_obs_arcsec": [0.0, 1.0],
        }
    )
    progress = FakeProgress()

    validation._posterior_prediction_uncertainty_tables(
        state,
        np.zeros((5, 0), dtype=float),
        images,
        max_draws=3,
        progress=progress,
    )

    assert progress.added == [("posterior uncertainty: draws x families", 6)]
    assert progress.advanced == [17] * 6
    assert len(progress.updated) == 6
    assert progress.updated[0] == "posterior uncertainty: draw=1/3 family=1 z=2.0000 failed_exact=0"
    assert progress.updated[-1] == "posterior uncertainty: draw=3/3 family=2 z=3.0000 failed_exact=0"


def test_posterior_prediction_uncertainty_tables_threaded_matches_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeExactModel:
        def magnification(self, x, y, kwargs_lens):
            sample_offset = float(kwargs_lens[0]["sample_offset"])
            return np.asarray(x, dtype=float) + np.asarray(y, dtype=float) + sample_offset

    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            self.state = kwargs["state"]

        def reported_physical_to_latent_parameter_vector(self, sample):
            return np.asarray(sample, dtype=float)

        def _family_source_summary(self, sample_latent):
            sample_offset = float(np.sum(sample_latent))
            return {
                str(family.family_id): {
                    "x_pred": family.x_obs + sample_offset,
                    "y_pred": family.y_obs - sample_offset,
                    "source_x": sample_offset + 0.1 * float(family.family_id),
                    "source_y": -sample_offset + 0.2 * float(family.family_id),
                    "source_plane_rms": 0.01 + sample_offset,
                }
                for family in self.state.family_data
            }

        def _get_exact_model_solver(self, z_source):
            return FakeExactModel(), None

        def _build_packed_lens_state(self, sample_latent, z_source):
            return {"sample_offset": float(np.sum(np.asarray(sample_latent, dtype=float)))}

        def _packed_to_kwargs_lens(self, packed_state):
            return [{"sample_offset": float(packed_state["sample_offset"])}]

        def _exact_family_prediction(self, sample_latent, family):
            sample_offset = float(np.sum(sample_latent))
            return (
                family.x_obs + sample_offset + 0.01 * float(family.family_id),
                family.y_obs - sample_offset - 0.02 * float(family.family_id),
                0.03 + sample_offset + 0.001 * float(family.family_id),
            )

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_latent", lambda sample, _specs: np.asarray(sample, dtype=float))
    state = SimpleNamespace(
        parameter_specs=[],
        family_data=[
            SimpleNamespace(
                family_id="1",
                z_source=2.0,
                n_images=2,
                image_labels=["1.1", "1.2"],
                x_obs=np.asarray([0.0, 1.0], dtype=float),
                y_obs=np.asarray([0.5, 1.5], dtype=float),
            ),
            SimpleNamespace(
                family_id="2",
                z_source=3.0,
                n_images=1,
                image_labels=["2.1"],
                x_obs=np.asarray([2.0], dtype=float),
                y_obs=np.asarray([2.5], dtype=float),
            ),
        ],
    )
    images = pd.DataFrame(
        {
            "family_id": ["1", "1", "2"],
            "image_label": ["1.1", "1.2", "2.1"],
            "x_obs_arcsec": [0.0, 1.0, 2.0],
            "y_obs_arcsec": [0.5, 1.5, 2.5],
        }
    )
    samples = np.asarray([[0.0], [0.5], [1.0], [1.5]], dtype=float)

    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 1)
    serial = validation._posterior_prediction_uncertainty_tables(
        state,
        samples,
        images,
        max_draws=3,
    )
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 2)
    threaded = validation._posterior_prediction_uncertainty_tables(
        state,
        samples,
        images,
        max_draws=3,
    )

    for serial_df, threaded_df, key in zip(serial, threaded, ["image_label", "image_label", "family_id"]):
        pd.testing.assert_frame_equal(
            serial_df.sort_values(key).reset_index(drop=True),
            threaded_df.sort_values(key).reset_index(drop=True),
        )


def test_posterior_prediction_uncertainty_tables_threaded_skips_failed_exact_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_calls: list[str] = []
    exact_call_lock = threading.Lock()

    class FakeExactModel:
        def magnification(self, x, y, kwargs_lens):
            return np.ones_like(np.asarray(x, dtype=float))

    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            self.state = kwargs["state"]

        def reported_physical_to_latent_parameter_vector(self, sample):
            return np.asarray(sample, dtype=float)

        def _family_source_summary(self, sample_latent):
            sample_offset = float(np.sum(sample_latent))
            return {
                str(family.family_id): {
                    "x_pred": family.x_obs + sample_offset,
                    "y_pred": family.y_obs + sample_offset,
                    "source_x": sample_offset,
                    "source_y": sample_offset,
                    "source_plane_rms": sample_offset,
                }
                for family in self.state.family_data
            }

        def _get_exact_model_solver(self, z_source):
            return FakeExactModel(), None

        def _build_packed_lens_state(self, sample_latent, z_source):
            return {}

        def _packed_to_kwargs_lens(self, packed_state):
            return []

        def _exact_family_prediction(self, sample_latent, family):
            with exact_call_lock:
                exact_calls.append(str(family.family_id))
            if str(family.family_id) == "2":
                return None
            return family.x_obs + 0.01, family.y_obs + 0.02, 0.03

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_latent", lambda sample, _specs: np.asarray(sample, dtype=float))
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 2)
    state = SimpleNamespace(
        parameter_specs=[],
        family_data=[
            SimpleNamespace(
                family_id="1",
                z_source=2.0,
                n_images=1,
                image_labels=["1.1"],
                x_obs=np.asarray([0.0], dtype=float),
                y_obs=np.asarray([0.0], dtype=float),
            ),
            SimpleNamespace(
                family_id="2",
                z_source=3.0,
                n_images=1,
                image_labels=["2.1"],
                x_obs=np.asarray([1.0], dtype=float),
                y_obs=np.asarray([1.0], dtype=float),
            ),
        ],
    )
    images = pd.DataFrame(
        {
            "family_id": ["1", "2"],
            "image_label": ["1.1", "2.1"],
            "x_obs_arcsec": [0.0, 1.0],
            "y_obs_arcsec": [0.0, 1.0],
        }
    )

    _mag_df, _image_df, source_df = validation._posterior_prediction_uncertainty_tables(
        state,
        np.asarray([[0.0], [1.0], [2.0]], dtype=float),
        images,
        max_draws=3,
    )

    assert exact_calls.count("1") == 3
    assert exact_calls.count("2") == 1
    family2 = source_df[source_df["family_id"] == "2"].iloc[0]
    assert np.isnan(family2["exact_image_rms_q50"])


def test_posterior_prediction_uncertainty_tables_keeps_partial_exact_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_offsets: list[float] = []

    class FakeExactModel:
        def magnification(self, x, y, kwargs_lens):
            del y
            offset = float(kwargs_lens[0]["offset"])
            return np.asarray(x, dtype=float) + offset

    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            self.state = kwargs["state"]

        def reported_physical_to_latent_parameter_vector(self, sample):
            return np.asarray(sample, dtype=float)

        def _family_source_summary(self, sample_latent):
            offset = float(np.sum(sample_latent))
            return {
                "1": {
                    "source_x": offset,
                    "source_y": -offset,
                    "source_plane_rms": offset + 0.5,
                }
            }

        def _get_exact_model_solver(self, _z_source):
            return FakeExactModel(), None

        def _build_packed_lens_state(self, sample_latent, _z_source):
            return {"offset": float(np.sum(np.asarray(sample_latent, dtype=float)))}

        def _packed_to_kwargs_lens(self, packed_state):
            return [{"offset": float(packed_state["offset"])}]

        def _exact_family_prediction_details(self, sample_latent, _family):
            offset = float(np.sum(sample_latent))
            exact_offsets.append(offset)
            return {
                "produced_image_count": 3,
                "recovered_image_count": 2,
                "missing_image_count": 1,
                "extra_image_count": 1,
                "multiplicity_failed": True,
                "multiplicity_failure_reason": "match_tolerance_exceeded",
                "matched_model_x_arcsec": np.asarray([0.1 + offset, np.nan, 2.1 + offset], dtype=float),
                "matched_model_y_arcsec": np.asarray([0.0, np.nan, 0.0], dtype=float),
                "recovered_image_mask": np.asarray([True, False, True], dtype=bool),
                "extra_model_x_arcsec": np.asarray([8.0 + offset], dtype=float),
                "extra_model_y_arcsec": np.asarray([-2.0], dtype=float),
                "arc_recovery_status": np.asarray(["point_recovered", "arc_supported", "point_recovered"], dtype=object),
                "arc_aware_image_residual_arcsec": np.asarray([0.1 + offset, 0.05 + offset, 0.1 + offset], dtype=float),
                "arc_noncritical_direction_residual_arcsec": np.asarray([0.02, 0.05 + offset, 0.03], dtype=float),
                "arc_critical_direction_residual_arcsec": np.asarray([0.0, 5.0 + offset, 0.0], dtype=float),
                "arc_s_min": np.asarray([0.2, 0.04, 0.2], dtype=float),
                "arc_prior_probability": np.asarray([0.4, 0.75, 0.4], dtype=float),
                "arc_supported_mask": np.asarray([False, True, False], dtype=bool),
                "arc_aware_image_rms_arcsec": 0.25 + offset,
                "failed": True,
            }

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_latent", lambda sample, _specs: np.asarray(sample, dtype=float))
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 1)
    state = SimpleNamespace(
        parameter_specs=[],
        family_data=[
            SimpleNamespace(
                family_id="1",
                z_source=2.0,
                n_images=3,
                image_labels=["1.1", "1.2", "1.3"],
                x_obs=np.asarray([0.0, 1.0, 2.0], dtype=float),
                y_obs=np.asarray([0.0, 0.0, 0.0], dtype=float),
            )
        ],
    )
    images = pd.DataFrame(
        {
            "family_id": ["1", "1", "1"],
            "image_label": ["1.1", "1.2", "1.3"],
            "x_obs_arcsec": [0.0, 1.0, 2.0],
            "y_obs_arcsec": [0.0, 0.0, 0.0],
        }
    )

    _mag_df, image_df, source_df = validation._posterior_prediction_uncertainty_tables(
        state,
        np.asarray([[0.0], [1.0], [2.0]], dtype=float),
        images,
        max_draws=3,
    )

    assert exact_offsets == [0.0, 1.0, 2.0]
    indexed = image_df.set_index("image_label")
    assert indexed.loc["1.1", "x_model_q50"] == pytest.approx(1.1)
    assert np.isnan(indexed.loc["1.2", "x_model_q50"])
    assert indexed.loc["1.3", "x_model_q50"] == pytest.approx(3.1)
    assert indexed.loc["1.3", "image_residual_q50"] == pytest.approx(1.1)
    assert indexed.loc["1.2", "arc_aware_image_residual_q50"] == pytest.approx(1.05)
    assert source_df.set_index("family_id").loc["1", "arc_aware_image_rms_q50"] == pytest.approx(1.25)


def test_posterior_prediction_uncertainty_tables_approximate_uses_median_std_without_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_calls: list[str] = []

    class FakeExactModel:
        def magnification(self, x, y, kwargs_lens):
            sample_offset = float(kwargs_lens[0]["sample_offset"])
            return np.asarray(x, dtype=float) + np.asarray(y, dtype=float) + sample_offset

    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            self.state = kwargs["state"]

        def reported_physical_to_latent_parameter_vector(self, sample):
            return np.asarray(sample, dtype=float)

        def _family_source_summary(self, sample_latent):
            sample_offset = float(np.sum(sample_latent))
            return {
                str(family.family_id): {
                    "x_pred": np.full(family.n_images, np.nan),
                    "y_pred": np.full(family.n_images, np.nan),
                    "source_x": sample_offset,
                    "source_y": 2.0 * sample_offset,
                    "source_plane_rms": 10.0 + sample_offset,
                }
                for family in self.state.family_data
            }

        def _get_exact_model_solver(self, z_source):
            return FakeExactModel(), object()

        def _build_packed_lens_state(self, sample_latent, z_source):
            return {"sample_offset": float(np.sum(np.asarray(sample_latent, dtype=float)))}

        def _packed_to_kwargs_lens(self, packed_state):
            return [{"sample_offset": float(packed_state["sample_offset"])}]

        def _exact_family_prediction(self, sample_latent, family):
            exact_calls.append(str(family.family_id))
            raise AssertionError("approximate mode should not call exact image validation")

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_latent", lambda sample, _specs: np.asarray(sample, dtype=float))
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 2)
    state = SimpleNamespace(
        parameter_specs=[],
        family_data=[
            SimpleNamespace(
                family_id="1",
                z_source=2.0,
                n_images=1,
                image_labels=["1.1"],
                x_obs=np.asarray([1.0], dtype=float),
                y_obs=np.asarray([2.0], dtype=float),
            )
        ],
    )
    images = pd.DataFrame(
        {
            "family_id": ["1"],
            "image_label": ["1.1"],
            "x_obs_arcsec": [1.0],
            "y_obs_arcsec": [2.0],
        }
    )

    mag_df, image_df, source_df = validation._posterior_prediction_uncertainty_tables(
        state,
        np.asarray([[0.0], [1.0], [3.0]], dtype=float),
        images,
        max_draws=3,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
    )

    assert exact_calls == []
    assert image_df.empty
    mag_row = mag_df.set_index("image_label").loc["1.1"]
    source_row = source_df.set_index("family_id").loc["1"]

    mag_values = np.asarray([3.0, 4.0, 6.0], dtype=float)
    assert mag_row["magnification_q16"] == pytest.approx(float(np.median(mag_values) - np.std(mag_values)))
    assert mag_row["magnification_q50"] == pytest.approx(float(np.median(mag_values)))
    assert mag_row["magnification_q84"] == pytest.approx(float(np.median(mag_values) + np.std(mag_values)))

    source_x_values = np.asarray([0.0, 1.0, 3.0], dtype=float)
    assert source_row["source_x_q16"] == pytest.approx(float(np.median(source_x_values) - np.std(source_x_values)))
    assert source_row["source_x_q50"] == pytest.approx(float(np.median(source_x_values)))
    assert source_row["source_x_q84"] == pytest.approx(float(np.median(source_x_values) + np.std(source_x_values)))

    source_y_values = np.asarray([0.0, 2.0, 6.0], dtype=float)
    assert source_row["source_y_q16"] == pytest.approx(float(np.median(source_y_values) - np.std(source_y_values)))
    assert source_row["source_y_q50"] == pytest.approx(float(np.median(source_y_values)))
    assert source_row["source_y_q84"] == pytest.approx(float(np.median(source_y_values) + np.std(source_y_values)))

    source_rms_values = np.asarray([10.0, 11.0, 13.0], dtype=float)
    assert source_row["source_plane_rms_q16"] == pytest.approx(
        float(np.median(source_rms_values) - np.std(source_rms_values))
    )
    assert source_row["source_plane_rms_q50"] == pytest.approx(float(np.median(source_rms_values)))
    assert source_row["source_plane_rms_q84"] == pytest.approx(
        float(np.median(source_rms_values) + np.std(source_rms_values))
    )


def test_write_recovery_outputs_caps_mass_profile_posterior_draws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "config": {"z_lens": 0.4, "source_redshift": 2.0},
                "kwargs_lens": [],
                "images": [{"family_id": "1", "image_label": "1.1", "magnification_true": 2.0}],
                "subhalos": [],
            }
        ),
        encoding="utf-8",
    )
    mock_images_path = tmp_path / "mock_images.json"
    mock_images_path.write_text(
        json.dumps([{"family_id": "1", "image_label": "1.1", "magnification_true": 2.0}]),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    all_samples = np.arange(200, dtype=float).reshape(200, 1)
    best_fit = np.asarray([123.0], dtype=float)
    recovery_profile_draws = 32
    captured_profile_samples: list[np.ndarray] = []
    captured_posterior_modes: list[str | None] = []
    captured_recovered_quick: list[bool] = []
    captured_logs: list[str] = []
    approximate_recovery_payload: dict[str, Any] = {}

    def fake_logged_phase(args, phase_name, fn, **kwargs):
        return fn()

    def touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pdf", encoding="utf-8")

    def fake_mass_profiles(_state, profile_samples, _truth, radii_arcsec, **_kwargs):
        captured_profile_samples.append(np.asarray(profile_samples, dtype=float))
        return pd.DataFrame(), pd.DataFrame()

    def fake_posterior_uncertainty(*_args, **kwargs):
        captured_posterior_modes.append(kwargs.get("posterior_diagnostic_mode"))
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def fake_recovered_model_tables(*_args, **kwargs):
        captured_recovered_quick.append(bool(kwargs.get("quick_diagnostics", False)))
        return (
            pd.DataFrame({"image_label": ["1.1"], "magnification_recovered": [2.1]}),
            pd.DataFrame({"image_label": ["1.1"], "image_residual_arcsec": [0.05]}),
            pd.DataFrame({"family_id": ["1"]}),
        )

    monkeypatch.setattr(validation, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(validation, "_log", lambda _args, message: captured_logs.append(str(message)))
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 5)
    monkeypatch.setattr(
        validation,
        "_load_plot_bundle",
        lambda _run_dir: (
            SimpleNamespace(
                parameter_specs=[],
                lens_model_list=[],
                packed_lens_spec=SimpleNamespace(component_family=np.asarray([], dtype=int)),
            ),
            {},
            {"samples": all_samples, "best_fit": best_fit},
            {},
        ),
    )
    monkeypatch.setattr(validation, "_artifact_parameter_names", lambda _state: ["p"])
    monkeypatch.setattr(validation, "_parameter_truth_with_source_positions", lambda _truth: {"p": 0.0})
    monkeypatch.setattr(
        validation,
        "parameter_recovery_table",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "parameter": ["p"],
                "truth": [0.0],
                "q16": [-1.0],
                "median": [0.0],
                "q84": [1.0],
                "bias": [0.0],
                "covered_68": [True],
            }
        ),
    )
    monkeypatch.setattr(validation, "_recovered_model_tables", fake_recovered_model_tables)
    monkeypatch.setattr(
        validation,
        "_posterior_prediction_uncertainty_tables",
        fake_posterior_uncertainty,
    )
    monkeypatch.setattr(validation, "_mass_and_surface_density_profiles_for_samples", fake_mass_profiles)
    monkeypatch.setattr(
        validation,
        "_scaling_parameter_subset",
        lambda *_args, **_kwargs: ([], np.zeros((200, 0), dtype=float), np.zeros((0,), dtype=float)),
    )
    monkeypatch.setattr(
        validation,
        "_plot_corner_pdf",
        lambda output_dir, _samples, _specs, filename="corner.pdf", truth_values=None, best_fit_values=None, **_kwargs: touch(
            Path(output_dir) / filename
        ),
    )
    monkeypatch.setattr(validation, "_plot_parameter_recovery", lambda _df, path, scale="log_abs": touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_magnification_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_absolute_magnification_recovery_grid", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(validation, "_plot_absolute_magnification_recovery", lambda _grid, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_residual_histogram", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_source_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_shmf", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_radial", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_validation_summary", lambda _summary, _uncertainty, path: touch(Path(path)))

    validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=output_dir,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
        recovery_profile_draws=recovery_profile_draws,
        recovery_payload=approximate_recovery_payload,
    )

    approximate_warnings = [
        message
        for message in captured_logs
        if "warning approximations active" in message
        and "posterior_diagnostic_mode=approximate" in message
    ]
    assert len(approximate_warnings) == 1
    assert "median+/-std" in approximate_warnings[0]
    assert "exact per-draw image validation skipped" in approximate_warnings[0]
    assert approximate_recovery_payload["posterior_diagnostics"]["workers"] == 5
    assert approximate_recovery_payload["posterior_diagnostics"]["recovery_profile_draws"] == recovery_profile_draws
    assert approximate_recovery_payload["posterior_diagnostics"]["recovery_profile_draws_effective"] == recovery_profile_draws
    assert approximate_recovery_payload["posterior_diagnostics"]["recovery_profile_mode"] == "posterior"

    captured_logs.clear()
    validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=tmp_path / "out_exact",
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        recovery_profile_draws=recovery_profile_draws,
    )

    assert not any("posterior_diagnostic_mode=approximate" in message for message in captured_logs)

    captured_logs.clear()
    validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=tmp_path / "out_quick",
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        recovery_profile_draws=recovery_profile_draws,
        quick_diagnostics=True,
    )

    assert any("quick_diagnostics=active" in message for message in captured_logs)
    assert captured_posterior_modes == [
        validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
        validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
    ]
    assert captured_recovered_quick == [False, False, True]
    assert len(captured_profile_samples) == 3
    expected_indices = np.linspace(
        0,
        all_samples.shape[0] - 1,
        recovery_profile_draws,
        dtype=int,
    )
    for profile_samples in captured_profile_samples:
        assert profile_samples.shape == (recovery_profile_draws, 1)
        np.testing.assert_array_equal(profile_samples, all_samples[expected_indices])

    best_fit_only_recovery_payload: dict[str, Any] = {}
    validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=tmp_path / "out_best_fit_only",
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        recovery_profile_draws=0,
        recovery_payload=best_fit_only_recovery_payload,
    )

    assert captured_profile_samples[-1].shape == (1, 1)
    np.testing.assert_array_equal(captured_profile_samples[-1], best_fit.reshape(1, -1))
    assert best_fit_only_recovery_payload["posterior_diagnostics"]["recovery_profile_draws"] == 0
    assert best_fit_only_recovery_payload["posterior_diagnostics"]["recovery_profile_draws_effective"] == 1
    assert best_fit_only_recovery_payload["posterior_diagnostics"]["recovery_profile_mode"] == "best_fit"

    negative_draws_recovery_payload: dict[str, Any] = {}
    validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=tmp_path / "out_negative_draws",
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        recovery_profile_draws=-4,
        recovery_payload=negative_draws_recovery_payload,
    )

    assert captured_profile_samples[-1].shape == (1, 1)
    np.testing.assert_array_equal(captured_profile_samples[-1], best_fit.reshape(1, -1))
    assert negative_draws_recovery_payload["posterior_diagnostics"]["recovery_profile_draws"] == -4
    assert negative_draws_recovery_payload["posterior_diagnostics"]["recovery_profile_draws_effective"] == 1
    assert negative_draws_recovery_payload["posterior_diagnostics"]["recovery_profile_mode"] == "best_fit"


def test_write_recovery_outputs_always_includes_critical_arc_support_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "config": {"z_lens": 0.4, "source_redshift": 2.0},
                "kwargs_lens": [],
                "images": [{"family_id": "1", "image_label": "1.1", "magnification_true": 2.0}],
                "subhalos": [],
            }
        ),
        encoding="utf-8",
    )
    mock_images_path = tmp_path / "mock_images.json"
    mock_images_path.write_text(
        json.dumps([{"family_id": "1", "image_label": "1.1", "magnification_true": 2.0}]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    touched: list[Path] = []

    def fake_logged_phase(args, phase_name, fn, **kwargs):
        return fn()

    def touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pdf", encoding="utf-8")
        touched.append(path)

    state = SimpleNamespace(
        parameter_specs=[],
        lens_model_list=[],
        packed_lens_spec=SimpleNamespace(component_family=np.asarray([], dtype=int)),
        family_data=[
            SimpleNamespace(
                family_id="1",
                z_source=2.0,
                effective_z_source=2.0,
                n_images=1,
            )
        ],
    )
    image_df = pd.DataFrame(
        {
            "family_id": ["1"],
            "image_label": ["1.1"],
            "z_source": [2.0],
            "effective_z_source": [2.0],
            "image_residual_arcsec": [0.4],
            "arc_aware_image_residual_arcsec": [0.03],
            "arc_noncritical_direction_residual_arcsec": [0.03],
            "arc_critical_direction_residual_arcsec": [4.0],
            "arc_s_min": [0.04],
            "arc_prior_probability": [0.75],
            "arc_recovery_status": ["arc_supported"],
            "arc_supported": [True],
            "model_produced_image_count": [1],
            "model_recovered_image_count": [0],
            "model_missing_image_count": [1],
            "model_extra_image_count": [0],
            "model_multiplicity_failed": [True],
            "model_multiplicity_failure_reason": ["missing_model_images"],
        }
    )

    monkeypatch.setattr(validation, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(
        validation,
        "_load_plot_bundle",
        lambda _run_dir: (
            state,
            {"sample_likelihood_mode": SAMPLE_LIKELIHOOD_SOURCE},
            {"samples": np.zeros((3, 0), dtype=float), "best_fit": np.zeros((0,), dtype=float)},
            {},
        ),
    )
    monkeypatch.setattr(validation, "_artifact_parameter_names", lambda _state: [])
    monkeypatch.setattr(
        validation,
        "parameter_recovery_table",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "parameter": ["p"],
                "truth": [0.0],
                "q16": [-1.0],
                "median": [0.0],
                "q84": [1.0],
                "bias": [0.0],
                "covered_68": [True],
            }
        ),
    )
    monkeypatch.setattr(
        validation,
        "_recovered_model_tables",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"image_label": ["1.1"], "magnification_recovered": [2.1]}),
            image_df.copy(),
            pd.DataFrame({"family_id": ["1"]}),
        ),
    )
    monkeypatch.setattr(validation, "_posterior_prediction_uncertainty_tables", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(validation, "_mass_and_surface_density_profiles_for_samples", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(validation, "_scaling_parameter_subset", lambda *_args, **_kwargs: ([], np.zeros((3, 0), dtype=float), np.zeros((0,), dtype=float)))
    monkeypatch.setattr(validation, "_plot_corner_pdf", lambda output_dir, _samples, _specs, filename="corner.pdf", **_kwargs: touch(Path(output_dir) / filename))
    monkeypatch.setattr(validation, "_plot_parameter_recovery", lambda _df, path, scale="log_abs": touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_magnification_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_absolute_magnification_recovery_grid", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(validation, "_plot_absolute_magnification_recovery", lambda _grid, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_residual_histogram", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_critical_arc_support_histogram", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_critical_arc_support_phase_space", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_critical_arc_recovery_by_family", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_source_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_shmf", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_radial", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_validation_summary", lambda _summary, _uncertainty, path: touch(Path(path)))

    outputs = validation.write_recovery_outputs(
        tmp_path / "run",
        truth_path,
        mock_images_path,
        output_dir=output_dir,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
    )

    assert outputs["critical_arc_support_histogram_plot"] == output_dir / "critical_arc_support_histogram.pdf"
    assert outputs["critical_arc_support_phase_space_plot"] == output_dir / "critical_arc_support_phase_space.pdf"
    assert outputs["critical_arc_recovery_by_family_plot"] == output_dir / "critical_arc_recovery_by_family.pdf"
    assert outputs["critical_arc_support_histogram_plot"] in touched
    assert outputs["critical_arc_support_phase_space_plot"] in touched
    assert outputs["critical_arc_recovery_by_family_plot"] in touched


def test_write_recovery_outputs_includes_cosmology_corner_for_cosmology_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "parameter_truth": {"cosmology_Om0": 0.3, "cosmology_w0": -1.0},
                "subhalos": [],
            }
        ),
        encoding="utf-8",
    )
    mock_images_path = tmp_path / "mock_images.json"
    mock_images_path.write_text(
        json.dumps([{"family_id": "1", "image_label": "1.1", "magnification_true": 2.0}]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    samples = np.asarray([[0.28, -1.1], [0.30, -1.0], [0.32, -0.9]], dtype=float)
    best_fit = np.asarray([0.31, -0.95], dtype=float)
    specs = [
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
    plot_calls: list[tuple[str, list[str], np.ndarray]] = []

    def fake_logged_phase(args, phase_name, fn, **kwargs):
        return fn()

    def touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pdf", encoding="utf-8")

    def fake_plot_corner_pdf(
        output_dir_arg,
        plot_samples,
        plot_specs,
        filename="corner.pdf",
        truth_values=None,
        best_fit_values=None,
        previous_stage_best_values=None,
    ):
        del truth_values, best_fit_values, previous_stage_best_values
        plot_calls.append((str(filename), [spec.name for spec in plot_specs], np.asarray(plot_samples, dtype=float)))
        touch(Path(output_dir_arg) / str(filename))

    monkeypatch.setattr(validation, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(
        validation,
        "_load_plot_bundle",
        lambda _run_dir: (
            SimpleNamespace(
                parameter_specs=specs,
                lens_model_list=[],
                packed_lens_spec=SimpleNamespace(component_family=np.asarray([], dtype=int)),
            ),
            {},
            {"samples": samples, "best_fit": best_fit},
            {},
        ),
    )
    monkeypatch.setattr(validation, "_artifact_parameter_names", lambda _state: ["cosmology.Om0", "cosmology.w0"])
    monkeypatch.setattr(
        validation,
        "parameter_recovery_table",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "parameter": ["cosmology.Om0", "cosmology.w0"],
                "truth": [0.3, -1.0],
                "q16": [0.28, -1.1],
                "median": [0.30, -1.0],
                "q84": [0.32, -0.9],
                "bias": [0.0, 0.0],
                "covered_68": [True, True],
            }
        ),
    )
    monkeypatch.setattr(
        validation,
        "_recovered_model_tables",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"image_label": ["1.1"], "magnification_recovered": [2.1]}),
            pd.DataFrame({"image_label": ["1.1"], "image_residual_arcsec": [0.05]}),
            pd.DataFrame({"family_id": ["1"]}),
        ),
    )
    monkeypatch.setattr(validation, "_posterior_prediction_uncertainty_tables", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(validation, "_plot_corner_pdf", fake_plot_corner_pdf)
    monkeypatch.setattr(validation, "_plot_parameter_recovery", lambda _df, path, scale="log_abs": touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_magnification_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_absolute_magnification_recovery_grid", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(validation, "_plot_absolute_magnification_recovery", lambda _grid, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_residual_histogram", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_source_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_shmf", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_radial", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_validation_summary", lambda _summary, _uncertainty, path: touch(Path(path)))

    outputs = validation.write_recovery_outputs(
        tmp_path / "run",
        truth_path,
        mock_images_path,
        output_dir=output_dir,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
    )

    assert outputs["cosmology_corner_plot"] == output_dir / "cosmology_corner.pdf"
    cosmology_calls = [call for call in plot_calls if call[0] == "cosmology_corner.pdf"]
    assert len(cosmology_calls) == 1
    assert cosmology_calls[0][1] == ["cosmology.Om0", "cosmology.w0"]
    np.testing.assert_allclose(cosmology_calls[0][2], samples)


def test_combined_mass_surface_density_profiles_match_separate_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def alpha(self, x, y, kwargs_lens, k=None):
            indices = list(k or [])
            scale = sum(float(kwargs_lens[index]["scale"]) for index in indices)
            return np.asarray(x, dtype=float) * scale, np.zeros_like(np.asarray(y, dtype=float))

    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def _build_packed_lens_state(self, sample_latent, z_source):
            return {"scale": float(np.asarray(sample_latent, dtype=float).reshape(-1)[0])}

        def _packed_to_kwargs_lens(self, packed_state):
            scale = float(packed_state["scale"])
            return [{"scale": scale}, {"scale": 2.0 * scale}]

        def release_runtime_caches(self) -> None:
            return None

    def fake_annular_surface_density(model, kwargs_lens, indices, radii_arcsec, sigma_crit_angle):
        scale = sum(float(kwargs_lens[index]["scale"]) for index in indices)
        return 0.5 * scale * np.asarray(radii_arcsec, dtype=float)

    monkeypatch.setattr(validation, "LensModel", FakeModel)
    monkeypatch.setattr(validation, "_annular_surface_density_msun_per_arcsec2", fake_annular_surface_density)
    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_latent", lambda sample, _specs: np.asarray(sample, dtype=float))
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 1)

    state = SimpleNamespace(
        lens_model_list=["fake_halo", "fake_subhalo"],
        parameter_specs=[],
        packed_lens_spec=SimpleNamespace(component_family=np.asarray([0, 1], dtype=int)),
    )
    samples = np.asarray([[1.0], [3.0]], dtype=float)
    truth = {
        "config": {"z_lens": 0.4, "source_redshift": 2.0},
        "kwargs_lens": [{"scale": 10.0}, {"scale": 20.0}],
    }
    radii = np.asarray([2.0, 5.0], dtype=float)

    mass_df, surface_df = validation._mass_and_surface_density_profiles_for_samples(state, samples, truth, radii)
    expected_mass_df = validation._deflection_profile_for_samples(state, samples, truth, radii)
    expected_surface_df = validation._surface_density_profile_for_samples(state, samples, truth, radii)

    pd.testing.assert_frame_equal(mass_df.reset_index(drop=True), expected_mass_df.reset_index(drop=True))
    pd.testing.assert_frame_equal(surface_df.reset_index(drop=True), expected_surface_df.reset_index(drop=True))

    executor_worker_counts: list[int] = []
    submitted_samples: list[int] = []

    class InlineExecutor:
        def __init__(self, *, max_workers: int) -> None:
            executor_worker_counts.append(int(max_workers))

        def __enter__(self) -> "InlineExecutor":
            return self

        def __exit__(self, *_exc: Any) -> bool:
            return False

        def submit(self, fn: Any, sample_index: int, sample: np.ndarray) -> Future[Any]:
            submitted_samples.append(int(sample_index))
            future: Future[Any] = Future()
            try:
                future.set_result(fn(sample_index, sample))
            except BaseException as exc:
                future.set_exception(exc)
            return future

    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 4)
    monkeypatch.setattr(validation, "ThreadPoolExecutor", InlineExecutor)
    threaded_mass_df, threaded_surface_df = validation._mass_and_surface_density_profiles_for_samples(
        state,
        samples,
        truth,
        radii,
    )

    assert executor_worker_counts == [2]
    assert submitted_samples == [1, 2]
    pd.testing.assert_frame_equal(threaded_mass_df.reset_index(drop=True), mass_df.reset_index(drop=True))
    pd.testing.assert_frame_equal(threaded_surface_df.reset_index(drop=True), surface_df.reset_index(drop=True))


def test_plot_surface_density_recovery_writes_pdf(tmp_path: Path) -> None:
    rows = []
    for component, label, scale in [("total", "total", 1.0), ("halo", "halo", 0.7)]:
        for radius in [2.0, 5.0, 10.0]:
            truth = 1.0e12 * scale / radius
            rows.append(
                {
                    "radius_arcsec": radius,
                    "component": component,
                    "component_label": label,
                    "truth": truth,
                    "q16": 0.9 * truth,
                    "median": 1.05 * truth,
                    "q84": 1.2 * truth,
                }
            )
    path = tmp_path / "surface_density_recovery.pdf"

    validation._plot_surface_density_recovery(pd.DataFrame(rows), path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_critical_caustic_recovery_writes_pdf(tmp_path: Path) -> None:
    truth_contour_z2 = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.2, 0.0, 0.2, 0.0, -0.2]),
        beta_y=np.asarray([0.0, 0.2, 0.0, -0.2, 0.0]),
        critical_x=np.asarray([-2.0, 0.0, 2.0, 0.0, -2.0]),
        critical_y=np.asarray([0.0, 2.0, 0.0, -2.0, 0.0]),
        caustic_area_arcsec2=0.08,
        critical_area_arcsec2=8.0,
    )
    truth_contour_z9 = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.3, 0.0, 0.3, 0.0, -0.3]),
        beta_y=np.asarray([0.0, 0.3, 0.0, -0.3, 0.0]),
        critical_x=np.asarray([-3.0, 0.0, 3.0, 0.0, -3.0]),
        critical_y=np.asarray([0.0, 3.0, 0.0, -3.0, 0.0]),
        caustic_area_arcsec2=0.18,
        critical_area_arcsec2=18.0,
    )
    recovered_contour_z2 = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.18, 0.02, 0.22, 0.02, -0.18]),
        beta_y=np.asarray([0.01, 0.21, 0.01, -0.19, 0.01]),
        critical_x=np.asarray([-1.9, 0.1, 2.1, 0.1, -1.9]),
        critical_y=np.asarray([0.1, 2.1, 0.1, -1.9, 0.1]),
        caustic_area_arcsec2=0.08,
        critical_area_arcsec2=8.0,
    )
    recovered_contour_z9 = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.28, 0.02, 0.32, 0.02, -0.28]),
        beta_y=np.asarray([0.01, 0.31, 0.01, -0.29, 0.01]),
        critical_x=np.asarray([-2.9, 0.1, 3.1, 0.1, -2.9]),
        critical_y=np.asarray([0.1, 3.1, 0.1, -2.9, 0.1]),
        caustic_area_arcsec2=0.18,
        critical_area_arcsec2=18.0,
    )
    images = pd.DataFrame(
        {
            "x_obs_arcsec": [-1.0, 1.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0, 1.0],
        }
    )
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [-1.0, 1.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0, 1.0],
            "x_model_arcsec": [-0.9, 1.1, 0.1],
            "y_model_arcsec": [0.0, 0.1, 1.1],
        }
    )
    source_df = pd.DataFrame(
        {
            "beta_x": [0.0],
            "beta_y": [0.0],
            "source_x_recovered": [0.02],
            "source_y_recovered": [0.01],
        }
    )
    subhalo_df = pd.DataFrame(
        {
            "x_arcsec": [2.5],
            "y_arcsec": [-1.0],
            "luminosity_ratio": [1.0],
        }
    )
    path = tmp_path / "critical_caustic_recovery.pdf"

    validation._plot_critical_caustic_recovery(
        {"2.00000000": [truth_contour_z2], "9.00000000": [truth_contour_z9]},
        {"2.00000000": [recovered_contour_z2], "9.00000000": [recovered_contour_z9]},
        images,
        image_df,
        source_df,
        subhalo_df,
        path,
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_select_critical_caustic_plot_contours_keeps_z9_only() -> None:
    contour = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([0.0, 0.1, 0.0]),
        beta_y=np.asarray([0.0, 0.1, 0.0]),
        critical_x=np.asarray([0.0, 1.0, 0.0]),
        critical_y=np.asarray([0.0, 1.0, 0.0]),
        caustic_area_arcsec2=0.01,
        critical_area_arcsec2=1.0,
    )

    selected = validation._select_critical_caustic_plot_contours(
        {
            "2.00000000": [contour],
            "9.00000000": [contour],
            "bad-key": [contour],
            "9.00000200": [contour],
            "9.00000050": [],
        }
    )

    assert selected == {"9.00000000": [contour]}


def test_plot_critical_caustic_recovery_ignores_marker_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.axes import Axes

    contour = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.3, 0.0, 0.3, 0.0, -0.3]),
        beta_y=np.asarray([0.0, 0.3, 0.0, -0.3, 0.0]),
        critical_x=np.asarray([-3.0, 0.0, 3.0, 0.0, -3.0]),
        critical_y=np.asarray([0.0, 3.0, 0.0, -3.0, 0.0]),
        caustic_area_arcsec2=0.18,
        critical_area_arcsec2=18.0,
    )
    scatter_labels: list[str | None] = []
    original_scatter = Axes.scatter

    def record_scatter(self: Axes, *args: object, **kwargs: object) -> object:
        scatter_labels.append(kwargs.get("label"))  # type: ignore[arg-type]
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", record_scatter)

    validation._plot_critical_caustic_recovery(
        {"9.00000000": [contour]},
        {"9.00000000": [contour]},
        pd.DataFrame({"x_obs_arcsec": [-1.0], "y_obs_arcsec": [0.0]}),
        pd.DataFrame(
            {
                "x_obs_arcsec": [-1.0],
                "y_obs_arcsec": [0.0],
                "x_model_arcsec": [-0.9],
                "y_model_arcsec": [0.1],
            }
        ),
        pd.DataFrame(
            {
                "beta_x": [0.0],
                "beta_y": [0.0],
                "source_x_recovered": [0.02],
                "source_y_recovered": [0.01],
            }
        ),
        pd.DataFrame({"x_arcsec": [2.5], "y_arcsec": [-1.0], "luminosity_ratio": [1.0]}),
        tmp_path / "critical_caustic_recovery.pdf",
    )

    assert scatter_labels == ["truth caustic", "recovered caustic"]


def test_write_recovery_outputs_filters_recovered_caustics_to_z9_and_logs_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contour_payload = {
        "caustic_index": 0,
        "caustic_class": "primary",
        "critical_x": [-1.0, 0.0, 1.0, -1.0],
        "critical_y": [0.0, 1.0, 0.0, 0.0],
        "caustic_beta_x": [-0.1, 0.0, 0.1, -0.1],
        "caustic_beta_y": [0.0, 0.1, 0.0, 0.0],
        "caustic_area_arcsec2": 0.01,
        "critical_area_arcsec2": 1.0,
    }
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "config": {"z_lens": 0.4, "source_redshift": 2.0, "caustic_grid_scale_arcsec": 0.5},
                "kwargs_lens": [],
                "caustics_by_source_redshift": {
                    "2.00000000": [contour_payload],
                    "9.00000000": [contour_payload],
                },
                "subhalos": [],
            }
        ),
        encoding="utf-8",
    )
    mock_images_path = tmp_path / "mock_images.json"
    mock_images_path.write_text(
        json.dumps([{"family_id": "1", "image_label": "1.1", "magnification_true": 2.0}]),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    phases: list[str] = []
    requested_truth_caustic_keys: list[list[str]] = []
    requested_caustic_keys: list[list[str]] = []
    truth_plot_grid_scales: list[float] = []
    recovered_plot_grid_scales: list[float] = []
    progress_instances = _install_recording_progress(monkeypatch)

    def fake_logged_phase(args, phase_name, fn, **kwargs):
        phases.append(phase_name)
        return fn()

    def touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pdf", encoding="utf-8")

    contour = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.1, 0.0, 0.1, -0.1]),
        beta_y=np.asarray([0.0, 0.1, 0.0, 0.0]),
        critical_x=np.asarray([-1.0, 0.0, 1.0, -1.0]),
        critical_y=np.asarray([0.0, 1.0, 0.0, 0.0]),
        caustic_area_arcsec2=0.01,
        critical_area_arcsec2=1.0,
    )

    monkeypatch.setattr(validation, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(
        validation,
        "_load_plot_bundle",
        lambda _run_dir: (
            SimpleNamespace(parameter_specs=[], lens_model_list=[], packed_lens_spec=SimpleNamespace(component_family=np.asarray([], dtype=int))),
            {},
            {"samples": np.zeros((2, 0), dtype=float), "best_fit": np.zeros((0,), dtype=float)},
            {},
        ),
    )
    monkeypatch.setattr(validation, "_artifact_parameter_names", lambda _state: [])
    monkeypatch.setattr(validation, "_parameter_truth_with_source_positions", lambda _truth: {})
    monkeypatch.setattr(
        validation,
        "parameter_recovery_table",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "parameter": ["p"],
                "truth": [1.0],
                "q16": [0.9],
                "median": [1.0],
                "q84": [1.1],
                "bias": [0.0],
                "covered_68": [True],
            }
        ),
    )
    monkeypatch.setattr(
        validation,
        "_recovered_model_tables",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"image_label": ["1.1"], "magnification_recovered": [2.1]}),
            pd.DataFrame({"image_label": ["1.1"], "image_residual_arcsec": [0.05]}),
            pd.DataFrame({"family_id": ["1"]}),
        ),
    )
    monkeypatch.setattr(validation, "_posterior_prediction_uncertainty_tables", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(validation, "_mass_and_surface_density_profiles_for_samples", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(validation, "_scaling_parameter_subset", lambda *_args, **_kwargs: ([], np.zeros((2, 0), dtype=float), np.zeros((0,), dtype=float)))

    def fake_truth_plot_caustics(_state, _truth, z_keys, **kwargs):
        requested_truth_caustic_keys.append(list(z_keys))
        truth_plot_grid_scales.append(float(kwargs["caustic_grid_scale_arcsec"]))
        return {"9.00000000": [contour]}

    def fake_recovered_caustics(_state, _best_fit, _truth, z_keys, **kwargs):
        requested_caustic_keys.append(list(z_keys))
        recovered_plot_grid_scales.append(float(kwargs["caustic_grid_scale_arcsec"]))
        return {"9.00000000": [contour]}

    monkeypatch.setattr(validation, "_truth_caustic_contours_by_z_for_plot", fake_truth_plot_caustics)
    monkeypatch.setattr(validation, "_recovered_caustic_contours_by_z", fake_recovered_caustics)
    monkeypatch.setattr(
        validation,
        "_plot_corner_pdf",
        lambda output_dir, _samples, _specs, filename="corner.pdf", truth_values=None, best_fit_values=None, **_kwargs: touch(
            Path(output_dir) / filename
        ),
    )
    monkeypatch.setattr(validation, "_plot_parameter_recovery", lambda _df, path, scale="log_abs": touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_critical_caustic_recovery", lambda *_args: touch(Path(_args[-1])))
    monkeypatch.setattr(validation, "_plot_magnification_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_absolute_magnification_recovery_grid", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(validation, "_plot_absolute_magnification_recovery", lambda _grid, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_residual_histogram", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_source_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_shmf", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_radial", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_validation_summary", lambda _summary, _uncertainty, path: touch(Path(path)))

    outputs = validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=output_dir,
        posterior_diagnostic_draws=1,
        critical_caustic_plot_grid_scale_arcsec=0.2,
        progress_args=argparse.Namespace(quiet=False),
    )

    assert requested_truth_caustic_keys == [["9.00000000"]]
    assert requested_caustic_keys == [["9.00000000"]]
    assert truth_plot_grid_scales == [pytest.approx(0.2)]
    assert recovered_plot_grid_scales == [pytest.approx(0.2)]
    assert "critical_caustic_plot" in outputs
    assert "cosmology_corner_plot" not in outputs
    assert "validation.recovery.load_plot_bundle" in phases
    assert "validation.recovery.truth_plot_caustics" in phases
    assert "validation.recovery.recovered_caustics" in phases
    assert "validation.recovery.plot_corner" in phases
    assert len(progress_instances) == 1
    progress_events = progress_instances[0].events
    parent_task = next(event[1] for event in progress_events if event[:3] == ("add_task", 1, "recovery: starting"))
    parent_advances = [event for event in progress_events if event == ("advance", parent_task, 1)]
    assert len(parent_advances) == len(phases)
    assert any(
        event[0] == "update"
        and event[1] == parent_task
        and event[2].get("description") == "recovery: posterior uncertainty"
        for event in progress_events
    )


@pytest.mark.slow
def test_single_bcg_recovery_smoke(tmp_path: Path) -> None:
    if os.environ.get("LENSCLUSTER_RUN_SLOW") != "1":
        pytest.skip("Set LENSCLUSTER_RUN_SLOW=1 to run the inference-backed validation smoke test.")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "lenscluster.validation",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "smoke",
            "--realizations",
            "1",
            "--n-primary-families",
            "1",
            "--n-subhalo-families",
            "0",
            "--fit-method",
            "svi",
            "--svi-steps",
            "2",
            "--samples",
            "4",
            "--warmup",
            "2",
            "--chains",
            "1",
            "--skip-plots",
        ],
        check=True,
    )
    run_dir = tmp_path / "single_bcg" / "smoke" / "seed_12345"
    expected_figures = [
        "parameter_recovery_log.pdf",
        "parameter_recovery_linear.pdf",
        "mass_profile_recovery.pdf",
        "surface_density_recovery.pdf",
        "critical_caustic_recovery.pdf",
        "magnification_recovery.pdf",
        "absolute_magnification_recovery.pdf",
        "image_recovery.pdf",
        "image_residual_histogram.pdf",
        "source_recovery.pdf",
        "subhalo_recovery_shmf.pdf",
        "subhalo_recovery_radial.pdf",
        "validation_summary.pdf",
    ]
    for figure_name in expected_figures:
        figure_path = run_dir / figure_name
        assert figure_path.exists()
        assert figure_path.stat().st_size > 0
    assert not list(run_dir.glob("*.csv"))
