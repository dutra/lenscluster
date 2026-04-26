import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lenscluster.cluster_solver import _adaptive_active_scaling_count
from lenscluster.lenstool_parser import load_best_par
from lenscluster.validation import (
    SingleBCGMockConfig,
    generate_single_bcg_mock,
    load_chires_family_summary,
    load_chires_table,
    magnification_recovery_table,
    parameter_recovery_table,
)


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


def test_generate_single_bcg_mock_parses_and_has_finite_magnifications(tmp_path: Path) -> None:
    config = SingleBCGMockConfig(seed=7, n_families=2, pos_sigma_arcsec=0.0)

    paths, images, truth = generate_single_bcg_mock(tmp_path, config)
    _parsed, _potentials_df, images_df, potentials_with_priors = load_best_par(paths.par_path)

    assert paths.par_path.exists()
    assert paths.image_catalog_path.exists()
    assert len(potentials_with_priors) == 2
    assert images_df["family_id"].nunique() == 2
    assert sorted(images_df.groupby("family_id")["catalog_z"].first().round(3).tolist()) == [1.5, 2.0]
    assert (images.groupby("family_id").size() >= config.min_images_per_family).all()
    assert np.isfinite(images["magnification_true"].to_numpy(dtype=float)).all()
    assert set(truth["parameter_truth"]) >= {"halo.v_disp", "bcg.v_disp", "source.sigma_int"}
    assert truth["parameter_truth"]["source.sigma_int"] == config.source_sigma_int_arcsec


def test_generate_single_bcg_mock_with_subhalos_uses_potfile(tmp_path: Path) -> None:
    config = SingleBCGMockConfig(seed=11, n_families=1, n_subhalos=8, pos_sigma_arcsec=0.0)

    paths, images, truth = generate_single_bcg_mock(tmp_path, config)
    parsed, _potentials_df, images_df, potentials_with_priors = load_best_par(paths.par_path)

    assert (tmp_path / "members.cat").exists()
    assert len(parsed["potfiles"]) == 1
    assert len(parsed["potfiles"][0]["catalog_df"]) == config.n_subhalos
    assert len(potentials_with_priors) == 2
    assert images_df["family_id"].nunique() == 1
    assert len(truth["subhalos"]) == config.n_subhalos
    assert len(truth["subhalo_components"]) == config.n_subhalos
    assert np.isfinite(images["magnification_true"].to_numpy(dtype=float)).all()


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
    samples = np.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    table = parameter_recovery_table(
        samples,
        ["halo.v_disp", "bcg.v_disp"],
        {"halo.v_disp": 1.0, "bcg.v_disp": 2.0},
        best_fit=np.asarray([1.0, 2.0]),
    )

    np.testing.assert_allclose(table["bias"], 0.0)
    assert table["covered_68"].tolist() == [True, True]
    np.testing.assert_allclose(table["truth_percentile"], 1.0)


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
            "--n-families",
            "1",
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
        "parameter_recovery.pdf",
        "mass_profile_recovery.pdf",
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
