import argparse
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import jax
import jax.numpy as jnp
from astropy.cosmology import FlatwCDM

import lenscluster.cluster_solver as cluster_solver
import lenscluster.validation as validation
from lenscluster.jax_cosmology import (
    dpie_sigma0_factor,
    dpie_sigma0_factor_from_lensing_efficiency,
    flat_wcdm_comoving_distance_mpc,
    flat_wcdm_kpc_per_arcsec,
    flat_wcdm_lens_geometry_factors,
    flat_wcdm_lensing_efficiency,
)
from lenscluster.cluster_solver import (
    IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    IMAGE_PLANE_MODE_NONE,
    SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    SAMPLE_LIKELIHOOD_SOURCE,
    _adaptive_active_scaling_count,
    _build_cosmology,
    _build_parameter_specs,
    _effective_image_presence_penalty_weight,
    _linearized_image_plane_bin_loglike,
    _linearized_marginal_beta_image_plane_bin_loglike,
    _linearized_image_plane_residual_from_jacobian,
    _soft_observed_image_presence_loglike,
    _local_jacobian_bin_loglike,
    _normalize_stage_fit_controls,
    _parse_args,
    _validation_metrics_summary,
)
from lenscluster.lenstool_parser import _split_image_label, load_best_par
from lenscluster.model import (
    BuildState,
    GeometryCache,
    EvaluationResult,
    FamilyData,
    FamilyValidationCache,
    PackedLensSpec,
    ParameterSpec,
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
from lenscluster.utils import _rich_log_text, format_stage_banner


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
    assert all(label.split(".", 1)[0].isdigit() and label.split(".", 1)[1].isdigit() for label in arc_labels)
    assert len(potentials_with_priors) == 2
    assert images_df["family_id"].nunique() == 2
    assert sorted(images_df.groupby("family_id")["catalog_z"].first().round(3).tolist()) == [1.5, 2.0]
    assert (images.groupby("family_id").size() >= config.min_images_per_family).all()
    assert {source["caustic_class"] for source in truth["sources"]} == {"primary"}
    assert all(int(source["n_images"]) >= 3 for source in truth["sources"])
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
        ("1.1a", ("1", "1a")),
        ("B200.2a", ("B200", "2a")),
        ("1a", ("1", "a")),
        ("10e", ("10", "e")),
        ("A200a", ("A200", "a")),
        ("candidate", ("candidate", "")),
        ("A200", ("A200", "")),
    ],
)
def test_split_image_label_supports_letter_suffixes(label: str, expected: tuple[str, str]) -> None:
    assert _split_image_label(label) == expected


def test_load_best_par_groups_letter_suffixed_image_labels(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "1a 10.0000 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "1b 10.0002 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "1c 10.0004 20.0000 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "letter_suffix.par"
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

    assert images_df["image_label"].astype(str).tolist() == ["1a", "1b", "1c"]
    assert images_df["family_id"].astype(str).tolist() == ["1", "1", "1"]
    assert images_df["image_id"].astype(str).tolist() == ["a", "b", "c"]


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


def test_build_state_ignores_non_positive_redshift_families(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_catalog_path = tmp_path / "obs_arcs.cat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "1.1 10.0000 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "1.2 10.0002 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "2.1 10.0100 20.0000 1.0 1.0 0.0 0.0 25.0\n"
        "2.2 10.0102 20.0000 1.0 1.0 0.0 0.0 25.0\n"
        "3.1 10.0200 20.0000 1.0 1.0 0.0 3.0 25.0\n",
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
    assert len(truth["subhalos"]) == config.n_subhalos
    assert len(truth["subhalo_components"]) == config.n_subhalos
    assert {source["caustic_class"] for source in truth["sources"]} == {"primary"}
    assert np.isfinite(images["magnification_true"].to_numpy(dtype=float)).all()


def test_generate_single_bcg_mock_can_request_subhalo_caustic_family(tmp_path: Path) -> None:
    config = SingleBCGMockConfig(
        seed=11,
        n_primary_families=1,
        n_subhalo_families=1,
        n_subhalos=8,
        pos_sigma_arcsec=0.0,
        source_redshifts=(2.0,),
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
        validate_top_k_families=0,
        sampling_engine="refreshing_surrogate",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        image_plane_newton_steps=0,
    )

    assert evaluator.surrogate_enabled is True


def test_refreshing_surrogate_enables_with_sampled_flat_wcdm_cosmology() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=_minimal_stage4_surrogate_state(fit_cosmology_flat_wcdm=True),
        match_tolerance_arcsec=0.1,
        validate_top_k_families=0,
        sampling_engine="refreshing_surrogate",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        image_plane_newton_steps=0,
    )

    assert evaluator.fit_cosmology_flat_wcdm is True
    assert evaluator.surrogate_enabled is True
    assert evaluator.surrogate_param_indices.tolist() == [0, 1, 2]


def test_sampled_cosmology_geometry_vectorizes_effective_redshift_factors() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=_minimal_stage4_surrogate_state(fit_cosmology_flat_wcdm=True),
        match_tolerance_arcsec=0.1,
        validate_top_k_families=0,
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
            validate_top_k_families=0,
            sampling_engine="refreshing_surrogate",
            sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
            image_plane_newton_steps=1,
        )


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
        validation_approx="adaptive",
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
    assert "validation_approx=adaptive" in warning
    assert "sample_likelihood=linearized-forward-beta-image-plane" in warning
    assert "source_position_parameterization=prior-whitened" in warning
    assert "active_scaling_subset=active 2/5" in warning
    assert "scaling_scatter_cache=linearized" in warning
    assert "source_metric_cache=refreshed" in warning


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
        validation_approx="exact",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        source_position_parameterization="direct",
        scaling_scatter_param_indices=np.asarray([], dtype=int),
        source_metric_cache_by_z={},
    )

    assert cluster_solver._solver_active_approximation_items(evaluator) == []


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
        source_plane_covariance_floor = 1.0e-6
        source_plane_outlier_sigma_arcsec = 10.0
        image_presence_penalty_weight = 0.0
        image_presence_match_radius_arcsec = 0.30
        image_presence_temperature_arcsec = 0.10
        image_presence_count_softness = 0.05
        image_presence_count_margin = 0.05
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


def test_cluster_solver_accepts_fit_cosmology_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["cluster_solver", "--par-path", "data/clustersim/input.par", "--fit-cosmology-flat-wcdm"],
    )

    args = _parse_args()

    assert args.fit_cosmology_flat_wcdm is True


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


def test_cosmology_init_without_fit_cosmology_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--cosmology-init-om0",
            "0.25",
        ],
    )

    args = _parse_args()

    with pytest.raises(SystemExit, match="require --fit-cosmology-flat-wcdm"):
        _normalize_stage_fit_controls(args)


def test_cluster_solver_defaults_to_prior_whitened_source_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "data/clustersim/input.par"])

    args = _parse_args()

    assert args.source_position_parameterization == "prior-whitened"


def test_cluster_solver_exact_image_solver_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "data/clustersim/input.par"])
    args = _parse_args()
    assert args.exact_image_solver == "auto"
    assert args.quick_diagnostics is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster_solver",
            "--par-path",
            "data/clustersim/input.par",
            "--exact-image-solver",
            "lenstronomy",
            "--quick-diagnostics",
        ],
    )
    args = _parse_args()
    assert args.exact_image_solver == "lenstronomy"
    assert args.quick_diagnostics is True


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


def test_validation_parser_defaults_to_prior_whitened_source_positions() -> None:
    args = validation._build_parser().parse_args([])

    assert args.source_position_parameterization == "prior-whitened"


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


def test_dpie_v_disp_normal_prior_uses_positive_latent_transform() -> None:
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
    physical = cluster_solver._convert_sample_matrix_to_physical(
        np.asarray([[-5.0], [0.0], [float(spec.mean)]], dtype=float),
        specs,
    )

    assert spec.name == "halo.v_disp"
    assert spec.transform_kind == "log_positive"
    assert spec.physical_mean == pytest.approx(900.0)
    assert spec.physical_std == pytest.approx(600.0)
    assert np.all(physical[:, 0] > 0.0)


def test_stage2_large_priors_convert_positive_physical_summary_to_latent() -> None:
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

    assert stage2_spec.transform_kind == "log_positive"
    assert stage2_spec.mean < 10.0
    assert physical_center == pytest.approx(895.533, rel=1.0e-3)


def test_packed_lens_validity_rejects_nonpositive_vdisp() -> None:
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
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert args.image_presence_penalty_weight == pytest.approx(3.5)
    assert args.image_presence_match_radius_arcsec == pytest.approx(0.4)
    assert args.image_presence_temperature_arcsec == pytest.approx(0.08)
    assert args.image_presence_count_softness == pytest.approx(0.03)
    assert args.image_presence_count_margin == pytest.approx(0.02)
    assert controls["stage4"].fit_method == "svi+nuts"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--image-presence-penalty-weight", "-0.1"),
        ("--image-presence-match-radius-arcsec", "0"),
        ("--image-presence-temperature-arcsec", "0"),
        ("--image-presence-count-softness", "0"),
        ("--image-presence-count-margin", "-0.1"),
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


def test_image_presence_effective_default_weight_only_stage4() -> None:
    assert _effective_image_presence_penalty_weight(
        None,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
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
    assert args.evidence_likelihood_mode == SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE
    assert controls["stage2"].to_json() == {"fit_method": "ns", "warmup": 0, "samples": 0}


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
    assert controls["stage2"].to_json() == {"fit_method": "ns", "warmup": 0, "samples": 0}


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

    assert controls["stage2"].to_json() == {"fit_method": "ns", "warmup": 0, "samples": 0}


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
            "--warmup",
            "2000",
            "0",
            "--samples",
            "250",
            "100",
        ],
    )

    args = _parse_args()
    controls = _normalize_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {"fit_method": "svi+nuts", "warmup": 2000, "samples": 250}
    assert controls["stage3"].to_json() == {"fit_method": "svi", "warmup": 0, "samples": 100}
    assert controls["stage4"].to_json() == {"fit_method": "svi", "warmup": 0, "samples": 100}


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
        warmup=2000,
        samples=250,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    controls = _normalize_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {"fit_method": "svi+nuts", "warmup": 2000, "samples": 250}
    assert controls["stage3"].to_json() == {"fit_method": "svi+nuts", "warmup": 2000, "samples": 250}
    assert controls["stage4"].to_json() == {"fit_method": "svi+nuts", "warmup": 2000, "samples": 250}


def test_stage_fit_controls_two_values_map_to_stage2_and_stage3() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi"],
        warmup=[2000, 0],
        samples=[250, 100],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    controls = _normalize_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {"fit_method": "svi+nuts", "warmup": 2000, "samples": 250}
    assert controls["stage3"].to_json() == {"fit_method": "svi", "warmup": 0, "samples": 100}
    assert controls["stage4"].to_json() == {"fit_method": "svi", "warmup": 0, "samples": 100}


def test_stage_fit_controls_three_values_map_to_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 20],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    )

    controls = _normalize_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {"fit_method": "svi+nuts", "warmup": 2000, "samples": 250}
    assert controls["stage3"].to_json() == {"fit_method": "svi+nuts", "warmup": 1000, "samples": 100}
    assert controls["stage4"].to_json() == {"fit_method": "svi", "warmup": 0, "samples": 20}


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
        fit_method=["svi+nuts", "svi"],
        warmup=1,
        samples=2,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_two_values_for_non_sequential_runs() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi"],
        warmup=1,
        samples=2,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_two_values_map_to_stage4_when_stage3_skipped() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi"],
        warmup=[2000, 0],
        samples=[250, 20],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=True,
    )

    controls = _normalize_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {"fit_method": "svi+nuts", "warmup": 2000, "samples": 250}
    assert controls["stage4"].to_json() == {"fit_method": "svi", "warmup": 0, "samples": 20}


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

    assert controls["stage2"].to_json() == {"fit_method": "ns", "warmup": 0, "samples": 0}


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


def test_evidence_state_uses_image_scatter_without_source_positions(monkeypatch: pytest.MonkeyPatch) -> None:
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
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE,
    )

    state = cluster_solver._build_state_from_inputs(evidence_args, fit_mode_override="evidence-ns")

    assert state.sigma_arcsec == pytest.approx(0.0)
    assert sum(spec.component_family == "source_position" for spec in state.parameter_specs) == 0
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
        warmup=300,
        samples=500,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {"fit_method": "svi+nuts", "warmup": 300, "samples": 500}
    assert controls["stage3"].to_json() == {"fit_method": "svi+nuts", "warmup": 300, "samples": 500}
    assert controls["stage4"].to_json() == {"fit_method": "svi+nuts", "warmup": 300, "samples": 500}


def test_validation_stage_fit_controls_two_values_map_to_stage2_and_stage3() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi"],
        warmup=[1000, 0],
        samples=[250, 100],
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {"fit_method": "svi+nuts", "warmup": 1000, "samples": 250}
    assert controls["stage3"].to_json() == {"fit_method": "svi", "warmup": 0, "samples": 100}
    assert controls["stage4"].to_json() == {"fit_method": "svi", "warmup": 0, "samples": 100}
    assert controls["stage4"].to_json() == {"fit_method": "svi", "warmup": 0, "samples": 100}


def test_validation_stage_fit_controls_three_values_map_to_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[1000, 500, 0],
        samples=[250, 100, 20],
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {"fit_method": "svi+nuts", "warmup": 1000, "samples": 250}
    assert controls["stage3"].to_json() == {"fit_method": "svi+nuts", "warmup": 500, "samples": 100}
    assert controls["stage4"].to_json() == {"fit_method": "svi", "warmup": 0, "samples": 20}


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
        fit_method=["svi+nuts", "svi"],
        warmup=1,
        samples=2,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    with pytest.raises(SystemExit):
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

    assert controls["stage2"].to_json() == {"fit_method": "ns", "warmup": 0, "samples": 0}


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

    assert controls["stage2"].to_json() == {"fit_method": "ns", "warmup": 0, "samples": 0}


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
            "--warmup",
            "1000",
            "0",
            "--samples",
            "250",
            "100",
        ]
    )

    controls = _normalize_validation_stage_fit_controls(args)

    assert controls["stage2"].to_json() == {"fit_method": "svi+nuts", "warmup": 1000, "samples": 250}
    assert controls["stage3"].to_json() == {"fit_method": "svi", "warmup": 0, "samples": 100}


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

    assert defaults.posterior_diagnostic_mode == validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT
    assert defaults.quick_diagnostics is False
    assert args.posterior_diagnostic_mode == validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE
    assert args.quick_diagnostics is True


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


def test_validation_parser_uses_primary_and_subhalo_family_counts() -> None:
    parser = validation._build_parser()

    defaults = parser.parse_args([])
    validation._validate_validation_args(defaults)

    assert defaults.n_primary_families == 20
    assert defaults.n_subhalo_families == 0

    args = parser.parse_args(["--n-primary-families", "4", "--n-subhalo-families", "2"])
    validation._validate_validation_args(args)

    assert args.n_primary_families == 4
    assert args.n_subhalo_families == 2
    assert args.min_images_per_family == 3


def test_validation_parser_rejects_removed_n_families_flag() -> None:
    with pytest.raises(SystemExit):
        validation._build_parser().parse_args(["--n-families", "3"])


def test_validation_parser_rejects_invalid_family_counts() -> None:
    parser = validation._build_parser()
    with pytest.raises(SystemExit):
        validation._validate_validation_args(parser.parse_args(["--n-primary-families", "-1"]))
    with pytest.raises(SystemExit):
        validation._validate_validation_args(parser.parse_args(["--n-primary-families", "0", "--n-subhalo-families", "0"]))
    with pytest.raises(SystemExit):
        validation._validate_validation_args(parser.parse_args(["--min-images-per-family", "1"]))


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


def test_cluster_solver_parser_accepts_resume_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cluster_solver", "--par-path", "data/clustersim/input.par", "--resume"])

    args = _parse_args()

    assert args.resume is True


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


def _validation_solver_args(**updates) -> argparse.Namespace:
    payload = dict(
        solver_fit_mode="sequential",
        fit_method="svi+nuts",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        skip_stage3_image_plane_local_jacobian=False,
        image_plane_newton_steps=0,
        linearized_beta_prior_sigma_arcsec=0.3,
        source_position_parameterization="prior-whitened",
        image_plane_scatter_upper_arcsec=2.0,
        image_presence_penalty_weight=None,
        image_presence_match_radius_arcsec=0.30,
        image_presence_temperature_arcsec=0.10,
        image_presence_count_softness=0.05,
        image_presence_count_margin=0.05,
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
        caustic_compute_window_arcsec=160.0,
        caustic_grid_scale_arcsec=0.2,
        caustic_min_area_arcsec2=1.0e-5,
        caustic_boundary_margin_arcsec=0.5,
        n_subhalos=0,
        subhalo_sigma_scatter_dex=0.07,
        subhalo_cut_scatter_dex=0.20,
        scaling_scatter_max=0.5,
        pos_sigma_arcsec=0.15,
        seed=12345,
        target_accept=0.85,
        max_tree_depth=8,
        skip_plots=True,
        quiet=False,
        resume=False,
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
            source_redshift=2.0,
            source_redshifts="1.5",
            source_sigma_int_arcsec=0.05,
            posterior_diagnostic_draws=2,
            posterior_diagnostic_workers=1,
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
    assert _option_values(cmd, "--warmup") == ["300"]
    assert _option_values(cmd, "--samples") == ["500"]
    assert "--ns-num-live-points" not in cmd
    assert "--ns-max-samples" not in cmd
    assert "--ns-dlogz" not in cmd
    assert _option_values(cmd, "--image-plane-mode") == [IMAGE_PLANE_MODE_NONE]
    assert "--linearized-image-plane-stage" not in cmd
    assert "--skip-stage3-image-plane-local-jacobian" not in cmd
    assert "--ott-sinkhorn-epsilon" not in cmd
    assert "--ott-sinkhorn-max-iterations" not in cmd
    assert "--ott-sinkhorn-threshold" not in cmd
    assert "--ott-sinkhorn-lse-mode" not in cmd
    assert _option_values(cmd, "--image-plane-newton-steps") == ["0"]
    assert _option_values(cmd, "--source-position-parameterization") == ["prior-whitened"]
    assert run_dir == tmp_path / "solver" / "fit" / "stage2_joint"


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
    assert "--source-position-parameterization" not in cmd
    assert "--image-plane-newton-steps" not in cmd
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
        warmup=[1000, 0],
        samples=[250, 100],
    )

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)
    cmd = captured["cmd"]

    assert _option_values(cmd, "--fit-method") == ["svi+nuts", "svi"]
    assert _option_values(cmd, "--warmup") == ["1000", "0"]
    assert _option_values(cmd, "--samples") == ["250", "100"]
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
        warmup=[1000, 500, 0],
        samples=[250, 100, 20],
        sampling_engine="full",
        image_presence_penalty_weight=3.0,
        image_presence_match_radius_arcsec=0.4,
        image_presence_temperature_arcsec=0.08,
        image_presence_count_softness=0.03,
        image_presence_count_margin=0.02,
    )

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)
    cmd = captured["cmd"]

    assert _option_values(cmd, "--fit-method") == ["svi+nuts", "svi+nuts", "svi"]
    assert _option_values(cmd, "--warmup") == ["1000", "500", "0"]
    assert _option_values(cmd, "--samples") == ["250", "100", "20"]
    assert _option_values(cmd, "--image-plane-mode") == [IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA]
    assert "--linearized-image-plane-stage" not in cmd
    assert "--ott-sinkhorn-epsilon" not in cmd
    assert "--ott-sinkhorn-max-iterations" not in cmd
    assert "--ott-sinkhorn-threshold" not in cmd
    assert "--ott-sinkhorn-lse-mode" not in cmd
    assert _option_values(cmd, "--image-plane-newton-steps") == ["1"]
    assert _option_values(cmd, "--source-position-parameterization") == ["prior-whitened"]
    assert _option_values(cmd, "--image-presence-penalty-weight") == ["3.0"]
    assert _option_values(cmd, "--image-presence-match-radius-arcsec") == ["0.4"]
    assert _option_values(cmd, "--image-presence-temperature-arcsec") == ["0.08"]
    assert _option_values(cmd, "--image-presence-count-softness") == ["0.03"]
    assert _option_values(cmd, "--image-presence-count-margin") == ["0.02"]
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
        warmup=[1000, 0],
        samples=[250, 20],
    )

    run_dir = validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)
    cmd = captured["cmd"]

    assert "--skip-stage3-image-plane-local-jacobian" in cmd
    assert _option_values(cmd, "--fit-method") == ["svi+nuts", "svi"]
    assert run_dir == tmp_path / "solver" / "fit" / "stage4_linearized_image_plane"


def test_validation_run_cluster_solver_forwards_resume_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(resume=True)

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert "--resume" in captured["cmd"]


def test_validation_run_cluster_solver_forwards_fit_cosmology_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(fit_cosmology_flat_wcdm=True)

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert "--fit-cosmology-flat-wcdm" in captured["cmd"]


def test_validation_run_cluster_solver_forwards_quick_diagnostics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    args = _validation_solver_args(quick_diagnostics=True)

    validation._run_cluster_solver(tmp_path / "input.par", tmp_path / "solver", "fit", args)

    assert "--quick-diagnostics" in captured["cmd"]


def _touch_complete_stage(stage_dir: Path) -> None:
    (stage_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (stage_dir / "artifacts" / "plot_bundle.h5").write_bytes(b"")
    (stage_dir / "tables").mkdir(parents=True, exist_ok=True)
    (stage_dir / "tables" / "run_summary.json").write_text("{}", encoding="utf-8")


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
        validation_family_ids = {"1"}

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
    assert phases.index("output.save_artifacts") < phases.index("validation.evaluate")


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
        likelihood=SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE,
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
    assert rows[0]["sample_likelihood_mode"] == SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE


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
    for stage, chi_square in [(stage1, 10.0), (stage2, 4.0)]:
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
                    "chi_square": chi_square,
                    "dof": 3,
                    "reduced_chi_square": chi_square / 3.0,
                    "aic": chi_square + 6.0,
                    "bic": chi_square + 7.0,
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


def test_plot_run_summary_includes_nested_sampling_evidence() -> None:
    args = argparse.Namespace(
        run_name="ns_run",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        skip_stage3_image_plane_local_jacobian=False,
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        sampling_engine="full",
        validation_approx="adaptive",
        nuts_init_boundary_frac=0.02,
        nuts_init_jitter_frac=0.02,
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
        validation_fallback_count=0,
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
    np.testing.assert_allclose(arrays["ns_diagnostics"]["log_L_samples"], [-5.0, -4.0, -3.0])
    assert "samples" not in arrays["ns_diagnostics"]


def test_validation_run_single_bcg_validation_logs_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_generate(root, config):
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
        return Path(output_dir) / run_name / "stage3_image_plane"

    def fake_recovery(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir,
        posterior_diagnostic_draws,
        posterior_diagnostic_workers=1,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        quick_diagnostics=False,
        progress_args=None,
    ):
        return {"summary_plot": Path(output_dir) / "validation_summary.pdf"}

    monkeypatch.setattr(validation, "generate_single_bcg_mock", fake_generate)
    monkeypatch.setattr(validation, "_run_cluster_solver", fake_solver)
    monkeypatch.setattr(validation, "write_recovery_outputs", fake_recovery)
    monkeypatch.setattr(
        validation,
        "write_validation_run_summary",
        lambda _solver_run_dir, _truth_path, output_dir, run_name, seed: Path(output_dir) / "run_summary.txt",
    )
    args = _validation_run_args(tmp_path, image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN)

    outputs = validation.run_single_bcg_validation(args)
    validation._close_debug_log()

    assert outputs == [{"summary_plot": tmp_path / "single_bcg" / "validation_log" / "seed_12345" / "validation_summary.pdf"}]
    log_text = (tmp_path / "single_bcg" / "validation_log" / "run_debug.log").read_text(encoding="utf-8")
    assert "VALIDATION REALIZATION 1/1" in log_text
    assert "realization start" in log_text
    assert "mock complete images=2" in log_text
    assert "recovery complete files=1" in log_text
    assert "validation complete realizations=1" in log_text


def test_validation_run_single_bcg_validation_forwards_posterior_diagnostic_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_modes: list[str] = []

    def fake_generate(root, config):
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
        posterior_diagnostic_workers=1,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        quick_diagnostics=False,
        progress_args=None,
    ):
        captured_modes.append(str(posterior_diagnostic_mode))
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
    )

    validation.run_single_bcg_validation(args)
    validation._close_debug_log()

    assert captured_modes == [validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE]


def test_validation_run_single_bcg_validation_forwards_quick_diagnostics_to_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_quick: list[bool] = []

    def fake_generate(root, config):
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
        posterior_diagnostic_workers=1,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        quick_diagnostics=False,
        progress_args=None,
    ):
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

    def fail_generate(*_args, **_kwargs):
        raise AssertionError("generate_single_bcg_mock should not run in resume mode with complete mock inputs")

    def fake_solver(par_path, output_dir, run_name, args):
        solver_calls.append(Path(par_path))
        return Path(output_dir) / run_name / "stage2_joint"

    monkeypatch.setattr(validation, "generate_single_bcg_mock", fail_generate)
    monkeypatch.setattr(validation, "_run_cluster_solver", fake_solver)
    monkeypatch.setattr(
        validation,
        "write_recovery_outputs",
        lambda _run_dir, _truth_path, _mock_images_path, output_dir, posterior_diagnostic_draws, posterior_diagnostic_workers=1, posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT, quick_diagnostics=False, progress_args=None: {
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
    assert outputs == [{"summary_plot": tmp_path / "single_bcg" / "validation_log" / "seed_12345" / "validation_summary.pdf"}]
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
        posterior_diagnostic_workers=1,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        quick_diagnostics=False,
        progress_args=None,
    ):
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
    assert outputs == [{"summary_plot": realization_dir / "validation_summary.pdf"}]
    log_text = (tmp_path / "single_bcg" / "validation_log" / "run_debug.log").read_text(encoding="utf-8")
    assert "[resume] reusing mock" in log_text
    assert "[resume] refreshing validation outputs" in log_text


def test_validation_quiet_suppresses_console_but_writes_debug_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_generate(root, config):
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
        lambda _run_dir, _truth_path, _mock_images_path, output_dir, posterior_diagnostic_draws, posterior_diagnostic_workers=1, posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT, quick_diagnostics=False, progress_args=None: {
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
        jac_a00=jnp.zeros(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.zeros(2, dtype=jnp.float64),
        covariance_floor=1.0e-3,
        outlier_sigma_arcsec=100.0,
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


def test_linearized_marginal_beta_loglike_matches_numerical_gaussian_integration() -> None:
    beta_x = np.asarray([0.0, 0.4], dtype=float)
    beta_y = np.asarray([0.0, -0.2], dtype=float)
    image_sigma = 0.3
    prior_sigma = 1.2
    prior_mean_x = 0.1
    prior_mean_y = -0.05
    analytic = _linearized_marginal_beta_image_plane_bin_loglike(
        beta_x=jnp.asarray(beta_x, dtype=jnp.float64),
        beta_y=jnp.asarray(beta_y, dtype=jnp.float64),
        jacobian_entries=(
            jnp.ones(2, dtype=jnp.float64),
            jnp.zeros(2, dtype=jnp.float64),
            jnp.zeros(2, dtype=jnp.float64),
            jnp.ones(2, dtype=jnp.float64),
        ),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.full((2,), image_sigma, dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=0.0,
        source_prior_mean_x=prior_mean_x,
        source_prior_mean_y=prior_mean_y,
        source_prior_sigma_arcsec=prior_sigma,
    )

    grid = np.linspace(-6.0, 6.0, 501)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    prior = np.exp(
        -0.5 * (((xx - prior_mean_x) ** 2 + (yy - prior_mean_y) ** 2) / prior_sigma**2)
    ) / (2.0 * np.pi * prior_sigma**2)
    likelihood = np.ones_like(prior)
    for obs_x, obs_y in zip(beta_x, beta_y, strict=True):
        likelihood *= np.exp(-0.5 * (((obs_x - xx) ** 2 + (obs_y - yy) ** 2) / image_sigma**2)) / (
            2.0 * np.pi * image_sigma**2
        )
    numerical = np.log(np.trapezoid(np.trapezoid(prior * likelihood, grid, axis=0), grid))

    assert float(analytic) == pytest.approx(float(numerical), abs=5.0e-4)


def test_linearized_marginal_beta_broad_prior_matches_profiled_shape_with_prior_volume_penalty() -> None:
    beta_x = np.asarray([0.0, 0.4], dtype=float)
    beta_y = np.asarray([0.0, -0.2], dtype=float)
    image_sigma = 0.3
    prior_sigma = 1.0e6
    value = _linearized_marginal_beta_image_plane_bin_loglike(
        beta_x=jnp.asarray(beta_x, dtype=jnp.float64),
        beta_y=jnp.asarray(beta_y, dtype=jnp.float64),
        jacobian_entries=(
            jnp.ones(2, dtype=jnp.float64),
            jnp.zeros(2, dtype=jnp.float64),
            jnp.zeros(2, dtype=jnp.float64),
            jnp.ones(2, dtype=jnp.float64),
        ),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.full((2,), image_sigma, dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=0.0,
        source_prior_mean_x=0.0,
        source_prior_mean_y=0.0,
        source_prior_sigma_arcsec=prior_sigma,
    )

    profiled_x = float(np.mean(beta_x))
    profiled_y = float(np.mean(beta_y))
    image_sigma2 = image_sigma**2
    profiled_loglike = float(
        np.sum(
            -0.5
            * (
                ((beta_x - profiled_x) ** 2 + (beta_y - profiled_y) ** 2) / image_sigma2
                + 2.0 * np.log(2.0 * np.pi * image_sigma2)
            )
        )
    )
    prior_volume_penalty = -0.5 * (2.0 * np.log(prior_sigma**2) + 2.0 * np.log(len(beta_x) / image_sigma2))

    assert float(value) == pytest.approx(profiled_loglike + prior_volume_penalty, abs=1.0e-5)


def test_linearized_marginal_beta_near_singular_jacobian_stays_finite() -> None:
    value = _linearized_marginal_beta_image_plane_bin_loglike(
        beta_x=jnp.asarray([0.0, 1.0e-8], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, -1.0e-8], dtype=jnp.float64),
        jacobian_entries=(
            jnp.asarray([1.0e-12, 1.0e-12], dtype=jnp.float64),
            jnp.zeros(2, dtype=jnp.float64),
            jnp.zeros(2, dtype=jnp.float64),
            jnp.ones(2, dtype=jnp.float64),
        ),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.full((2,), 0.15, dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        source_prior_mean_x=0.0,
        source_prior_mean_y=0.0,
        source_prior_sigma_arcsec=20.0,
    )

    assert np.isfinite(float(value))
    assert float(value) < 1.0e6


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
        used_exact_validation=True,
    )

    summary = _validation_metrics_summary(result)

    assert "validated_families=5" in summary
    assert "exact_families=2" in summary
    assert "exact_image_rms_mean=0.3" in summary
    assert "exact_image_rms_median=0.3" in summary
    assert "approx_image_rms_mean=0.6" in summary
    assert "source_rms_mean=0.04667" in summary
    assert "nan" not in summary.lower()


def test_exact_validation_failure_is_diagnostic_not_loglike_penalty() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    family = SimpleNamespace(family_id="1", z_source=2.0, n_images=3)
    cache = FamilyValidationCache()
    evaluator.state = SimpleNamespace(family_data=[family])
    evaluator.surrogate_enabled = False
    evaluator.validation_family_ids = {"1"}
    evaluator.validation_approx = "exact"
    evaluator.validation_fallback_count = 0
    evaluator.validation_cache = {"1": cache}
    evaluator.source_loglike = lambda _params: -123.0
    evaluator._family_source_summary = lambda _params: {"1": {"source_plane_rms": 0.2}}
    evaluator._should_run_exact_validation = lambda _family, _prediction: (True, "exact_mode")

    def fake_exact_prediction(_params, exact_family):
        evaluator.validation_cache[exact_family.family_id].exact_validation_count += 1
        evaluator.validation_cache[exact_family.family_id].multiplicity_mismatch_count += 1
        return None

    evaluator._exact_family_prediction = fake_exact_prediction

    result = evaluator.evaluate(np.asarray([0.0], dtype=float), validate_all_families=False)

    assert result.loglike == pytest.approx(-123.0)
    assert result.used_exact_validation is True
    assert result.family_predictions["1"]["failed"] is True
    assert result.family_predictions["1"]["used_exact_refresh"] is True
    assert result.family_predictions["1"]["refresh_reason"] == "exact_mode"
    assert cache.exact_validation_count == 1
    assert cache.multiplicity_mismatch_count == 1
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
    evaluator.validation_family_ids = {"1"}
    evaluator.validation_approx = "exact"
    evaluator.quick_diagnostics = True
    evaluator.source_loglike = lambda _params: -12.0
    evaluator._family_source_summary = lambda _params: {"1": {"source_plane_rms": 0.25}}

    def fail_exact(_params, _family):
        raise AssertionError("quick diagnostics should not call exact image prediction")

    evaluator._exact_family_prediction = fail_exact

    result = evaluator.evaluate(np.asarray([0.0], dtype=float), validate_all_families=True)

    assert result.loglike == pytest.approx(-12.0)
    assert result.used_exact_validation is False
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
    evaluator.surrogate_enabled = False
    evaluator.source_plane_covariance_floor = 1.0e-6
    evaluator.sample_likelihood_mode = SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE
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
    evaluator.sample_likelihood_mode = SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE
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


def test_exact_family_prediction_auto_falls_back_from_jax_mismatch() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.1,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 0.0]),
        reliability=np.ones(2),
    )
    cache = FamilyValidationCache()
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.validation_cache = {"1": cache}
    evaluator.source_plane_covariance_floor = 1.0e-6
    evaluator.sample_likelihood_mode = SAMPLE_LIKELIHOOD_SOURCE
    evaluator.exact_image_solver = "auto"
    evaluator.exact_jax_fallback_count = 0
    evaluator._build_packed_lens_state = lambda _params, _z_source: {}
    evaluator._exact_source_ray_shooting = lambda _family, _packed_state: (
        np.asarray([0.0, 2.0], dtype=float),
        np.asarray([0.0, 0.0], dtype=float),
    )
    evaluator._source_sigma_int_numpy = lambda _params: 0.0
    evaluator._source_position_for_family_numpy = lambda _params, _family_id: None
    calls: list[str] = []

    def fake_jax(_family, _packed_state, _source_x, _source_y):
        calls.append("jax")
        return np.asarray([0.0], dtype=float), np.asarray([0.0], dtype=float)

    def fake_lenstronomy(_family, _packed_state, _source_x, _source_y):
        calls.append("lenstronomy")
        return np.asarray([0.0, 1.0], dtype=float), np.asarray([0.0, 0.0], dtype=float)

    def fake_match(x_pred, y_pred, _family):
        if len(x_pred) != 2:
            return None
        return np.asarray(x_pred, dtype=float), np.asarray(y_pred, dtype=float)

    evaluator._solve_exact_images_jax = fake_jax
    evaluator._solve_exact_images_lenstronomy = fake_lenstronomy
    evaluator._match_images = fake_match

    prediction = cluster_solver.ClusterJAXEvaluator._exact_family_prediction(
        evaluator,
        np.asarray([], dtype=float),
        family,
    )

    assert prediction is not None
    assert calls == ["jax", "lenstronomy"]
    assert evaluator.exact_jax_fallback_count == 1
    assert cache.multiplicity_mismatch_count == 0
    np.testing.assert_allclose(prediction[0], [0.0, 1.0])


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


def test_sequential_fit_cosmology_applies_only_to_final_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        ("evidence-ns", "fit", "ns", 0, SAMPLE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE),
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
        "stage2": {"fit_method": "svi+nuts", "warmup": 2000, "samples": 250},
        "stage3": {"fit_method": "svi", "warmup": 0, "samples": 100},
        "stage4": {"fit_method": "svi", "warmup": 0, "samples": 100},
    }


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

    serial = validation._posterior_prediction_uncertainty_tables(
        state,
        samples,
        images,
        max_draws=3,
        max_workers=1,
    )
    threaded = validation._posterior_prediction_uncertainty_tables(
        state,
        samples,
        images,
        max_draws=3,
        max_workers=2,
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
        max_workers=2,
    )

    assert exact_calls.count("1") == 3
    assert exact_calls.count("2") == 1
    family2 = source_df[source_df["family_id"] == "2"].iloc[0]
    assert np.isnan(family2["exact_image_rms_q50"])


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
        max_workers=2,
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
    captured_profile_samples: list[np.ndarray] = []
    captured_posterior_modes: list[str | None] = []
    captured_recovered_quick: list[bool] = []
    captured_logs: list[str] = []

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
            {"samples": all_samples, "best_fit": np.zeros((1,), dtype=float)},
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
    monkeypatch.setattr(validation, "_plot_image_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_source_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(
        validation,
        "_plot_subhalo_population",
        lambda _subhalos, _images, _parameter_df, path: touch(Path(path)),
    )
    monkeypatch.setattr(validation, "_plot_validation_summary", lambda _summary, _uncertainty, path: touch(Path(path)))

    validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=output_dir,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
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

    captured_logs.clear()
    validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=tmp_path / "out_exact",
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
    )

    assert not any("posterior_diagnostic_mode=approximate" in message for message in captured_logs)

    captured_logs.clear()
    validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=tmp_path / "out_quick",
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
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
        validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
        dtype=int,
    )
    for profile_samples in captured_profile_samples:
        assert profile_samples.shape == (validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP, 1)
        np.testing.assert_array_equal(profile_samples, all_samples[expected_indices])


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
    monkeypatch.setattr(validation, "_plot_image_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_source_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_population", lambda _subhalos, _images, _parameter_df, path: touch(Path(path)))
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
    truth_contour_z7 = validation.CausticContour(
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
    recovered_contour_z7 = validation.CausticContour(
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
        {"2.00000000": [truth_contour_z2], "7.00000000": [truth_contour_z7]},
        {"2.00000000": [recovered_contour_z2], "7.00000000": [recovered_contour_z7]},
        images,
        image_df,
        source_df,
        subhalo_df,
        path,
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_select_critical_caustic_plot_contours_keeps_z7_only() -> None:
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
            "7.00000000": [contour],
            "bad-key": [contour],
            "7.00000200": [contour],
            "7.00000050": [],
        }
    )

    assert selected == {"7.00000000": [contour]}


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
        {"7.00000000": [contour]},
        {"7.00000000": [contour]},
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


def test_write_recovery_outputs_filters_recovered_caustics_to_z7_and_logs_phases(
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
                "config": {"z_lens": 0.4, "source_redshift": 2.0},
                "kwargs_lens": [],
                "caustics_by_source_redshift": {
                    "2.00000000": [contour_payload],
                    "7.00000000": [contour_payload],
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
    requested_caustic_keys: list[list[str]] = []
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

    def fake_recovered_caustics(_state, _best_fit, _truth, z_keys, **_kwargs):
        requested_caustic_keys.append(list(z_keys))
        return {"7.00000000": [contour]}

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
    monkeypatch.setattr(validation, "_plot_image_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_source_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_population", lambda _subhalos, _images, _parameter_df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_validation_summary", lambda _summary, _uncertainty, path: touch(Path(path)))

    outputs = validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=output_dir,
        posterior_diagnostic_draws=1,
        progress_args=argparse.Namespace(quiet=False),
    )

    assert requested_caustic_keys == [["7.00000000"]]
    assert "critical_caustic_plot" in outputs
    assert "cosmology_corner_plot" not in outputs
    assert "validation.recovery.load_plot_bundle" in phases
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
        "image_recovery.pdf",
        "source_recovery.pdf",
        "subhalo_population.pdf",
        "validation_summary.pdf",
    ]
    for figure_name in expected_figures:
        figure_path = run_dir / figure_name
        assert figure_path.exists()
        assert figure_path.stat().st_size > 0
    assert not list(run_dir.glob("*.csv"))
