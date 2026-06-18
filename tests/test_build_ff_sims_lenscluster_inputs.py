from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from lenscluster.cluster_solver import _build_parameter_specs
from lenscluster.lenstool_parser import load_best_par


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_ff_sims_lenscluster_inputs.py"
spec = importlib.util.spec_from_file_location("build_ff_sims_lenscluster_inputs", SCRIPT_PATH)
assert spec is not None
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


def _write_ff_sims_fixture(root: Path) -> Path:
    ares = root / "ares"
    hera = root / "hera"
    ares.mkdir(parents=True)
    hera.mkdir(parents=True)
    ares.joinpath("multimages.txt").write_text(
        "\n".join(
            [
                "10.0 -2.0 1 1 1.50",
                "12.0 -3.0 1 2 1.50",
                "-4.0 8.0 2 1 2.20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    hera.joinpath("multimages.txt").write_text(
        "\n".join(
            [
                "5.0 1.0 3 1 2.00",
                "7.0 2.0 3 2 2.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ares.joinpath("clgal_cat.txt").write_text(
        "\n".join(
            [
                "# id x y AB(f435w) AB(f606w) AB(f814w) AB(f105w) AB(f125w) AB(f140w) AB(f160w)",
                "1 5.0 6.0 21.0 20.0 19.0 18.0 18.0 18.0 18.0",
                "2 -3.0 4.0 22.0 21.0 20.0 19.0 19.0 19.0 19.0",
                "3 -8.0 -8.0 22.2 21.2 20.2 19.2 19.2 19.2 19.2",
                "4 9.0 9.0 22.4 21.4 20.4 19.4 19.4 19.4 19.4",
                "5 10.0 -10.0 22.6 21.6 20.6 19.6 19.6 19.6 19.6",
                "6 -11.0 11.0 23.0 22.0 21.0 21.0 21.0 21.0 21.0",
                "7 -12.0 12.0 23.5 22.5 21.5 22.5 22.5 22.5 22.5",
                "11 56.5925 22.6934 22.1 21.1 18.4532 18.0 18.0 18.0 18.4532",
                "17 18.3245 -38.8525 22.3 21.3 18.8095 18.2 18.2 18.2 18.8095",
                "31 26.5704 -39.9680 22.5 21.5 19.3712 18.8 18.8 18.8 19.3712",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    hera.joinpath("clgal_cat.txt").write_text(
        "\n".join(
            [
                "1 -9.0 1.0 21.0 20.0 18.0 18.0 18.0 18.0 18.0",
                "2 2.0 3.0 22.0 21.0 19.0 19.0 19.0 19.0 19.0",
                "3 -4.0 -4.0 23.0 22.0 20.0 20.0 20.0 20.0 20.0",
                "4 -5.0 5.0 24.0 23.0 20.5 20.5 20.5 20.5 20.5",
                "5 -6.0 6.0 25.0 24.0 21.0 21.0 21.0 21.0 21.0",
                "6 -7.0 7.0 25.5 24.5 21.5 21.5 21.5 21.5 21.5",
                "7 -7.5 7.5 26.0 25.0 22.0 22.0 22.0 22.0 22.0",
                "8 -8.0 8.0 26.5 25.5 23.0 22.5 22.5 22.5 22.5",
                "9 -9.0 9.0 27.0 26.0 23.5 23.0 23.0 23.0 23.0",
                "60 1.5 1.1 25.0 24.0 22.0 21.0 21.0 21.0 20.6",
                "10 ************ -4.0 23.0 22.0 20.0 20.0 20.0 20.0 20.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    hera.joinpath("galcat2.txt").write_text(
        "\n".join(
            [
                "#filter: F814W; z=0.5; mstar=19.82",
                "1 9.2 1.1 0.000250 0.000125 35.0 18.0 0.0",
                "2 -22.0 33.0 0.000300 0.000200 -15.0 19.0 0.0",
                "3 44.0 -4.5 0.000100 0.000090 5.0 20.0 0.0",
                "4 5.1 5.1 0.000400 0.000300 60.0 20.5 0.0",
                "999 6.1 6.1 0.000500 0.000400 70.0 21.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for cluster_dir in (ares, hera):
        for band in ("f435w", "f606w", "f814w"):
            cluster_dir.joinpath(f"simulation_hst_{band}.fits").write_bytes(b"fixture")
    return root


def _limit_block(par_text: str, component_id: str) -> str:
    return par_text.split(f"limit {component_id}", maxsplit=1)[1].split("\n    end", maxsplit=1)[0]


def _limit_line(par_text: str, component_id: str, field_name: str) -> str:
    for line in _limit_block(par_text, component_id).splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field_name} "):
            return stripped
    raise AssertionError(f"missing {field_name} limit for {component_id}")


def test_resolve_source_root_uses_first_existing_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _write_ff_sims_fixture(tmp_path / "source")
    monkeypatch.setattr(builder, "DEFAULT_SOURCE_ROOT_CANDIDATES", (tmp_path / "missing", source))

    assert builder._resolve_source_root() == source


def test_apply_matched_shapes_requires_same_id_within_tolerance() -> None:
    members = [
        builder.MemberRow("1", 0.0, 0.0, 1.0, 1.0, 0.0, 18.0, 17.0, 17.0, 1.0),
        builder.MemberRow("2", 10.0, 0.0, 1.0, 1.0, 0.0, 19.0, 18.0, 18.0, 1.0),
    ]
    shapes = [
        builder.MemberShape("1", 0.2, 0.0, 1.8, 1.2, 35.0),
        builder.MemberShape("2", 10.31, 0.0, 2.0, 1.4, -15.0),
    ]

    matched = builder._apply_matched_shapes(members, shapes, tolerance_arcsec=0.3)

    assert matched[0].a_axis == pytest.approx(1.8)
    assert matched[0].b_axis == pytest.approx(1.2)
    assert matched[0].theta_deg == pytest.approx(35.0)
    assert matched[1].a_axis == pytest.approx(1.0)
    assert matched[1].b_axis == pytest.approx(1.0)


def test_render_converts_catalogs_and_writes_loadable_pars(tmp_path: Path) -> None:
    source = _write_ff_sims_fixture(tmp_path / "source")
    output = tmp_path / "data" / "ff_sims"

    rows = builder.render(source_root=source, output_dir=output)

    assert {row["cluster_key"] for row in rows} == {"ares", "hera"}
    assert {row["cluster_key"]: row["n_skipped_members"] for row in rows} == {"ares": 0, "hera": 1}
    assert {row["cluster_key"]: row["n_members"] for row in rows} == {"ares": 5, "hera": 4}
    assert {row["cluster_key"]: row["n_explicit_galaxies"] for row in rows} == {"ares": 5, "hera": 6}
    assert {row["cluster_key"]: row["explicit_galaxy_ids"] for row in rows} == {
        "ares": "1;2;3;4;5",
        "hera": "1;2;3;4;5;6",
    }
    assert {row["cluster_key"]: row["member_selection"] for row in rows} == {
        "ares": "F814W<24.00",
        "hera": "F814W<24.00",
    }
    assert {row["cluster_key"]: row["scaling_band"] for row in rows} == {"ares": "F160W", "hera": "F160W"}
    assert {row["cluster_key"]: row["n_staged_fits"] for row in rows} == {"ares": 3, "hera": 3}
    manifest = (output / "ff_sims_manifest.csv").read_text(encoding="utf-8").splitlines()
    assert manifest[0].split(",")[:13] == [
        "cluster_key",
        "display_name",
        "z_lens",
        "n_images",
        "n_image_families",
        "n_members",
        "n_skipped_members",
        "n_explicit_galaxies",
        "explicit_galaxy_ids",
        "member_selection",
        "scaling_band",
        "mag0",
        "n_staged_fits",
    ]
    ares_image_lines = (output / "ares" / "ares_obs_arcs.cat").read_text(encoding="utf-8").splitlines()
    assert ares_image_lines[0] == "#REFERENCE 3"
    assert ares_image_lines[1].split()[:4] == ["1.a", "-10.00000000", "-2.00000000", "0.3734"]
    assert ares_image_lines[2].split()[0] == "1.b"
    assert (output / "ares" / "simulation_hst_f435w.fits").is_file()
    assert (output / "hera" / "simulation_hst_f814w.fits").is_file()

    ares_members = (output / "ares" / "ares_cluster_members_potfile.cat").read_text(encoding="utf-8").splitlines()
    assert ares_members[1] == "# FF-SIMS scaling-law member potfile generated from clgal_cat.txt."
    assert any("# Member 1 is excluded from the scaling potfile; modeled explicitly as G1. It also anchors O1." == line for line in ares_members)
    assert any("# Member 2 is excluded from the scaling potfile; modeled explicitly as G2." == line for line in ares_members)
    assert any("# Member 3 is excluded from the scaling potfile; modeled explicitly as G3. It also anchors O2." == line for line in ares_members)
    assert any("# Member 4 is excluded from the scaling potfile; modeled explicitly as G4." == line for line in ares_members)
    assert any("# Member 5 is excluded from the scaling potfile; modeled explicitly as G5." == line for line in ares_members)
    assert not any("modeled explicitly as G31" in line for line in ares_members)
    assert not any("modeled explicitly as G11" in line for line in ares_members)
    assert not any("modeled explicitly as G17" in line for line in ares_members)
    assert any(line.startswith("#          1 -5.00000000  6.00000000") for line in ares_members)
    active_ares_member_lines = [line for line in ares_members if not line.startswith("#")]
    assert len(active_ares_member_lines) == 5
    assert [line.split()[0] for line in active_ares_member_lines] == ["11", "17", "31", "6", "7"]
    active_ares_ids = {line.split()[0] for line in active_ares_member_lines}
    assert not ({"1", "2", "3", "4", "5"} & active_ares_ids)
    assert {"11", "17", "31"} <= active_ares_ids
    assert active_ares_member_lines[0].split()[:4] == ["11", "-56.59250000", "22.69340000", "1.00000000"]
    assert active_ares_member_lines[0].split()[6] == "18.453200"

    hera_members = (output / "hera" / "hera_cluster_members_potfile.cat").read_text(encoding="utf-8").splitlines()
    assert any("# Member 1 is excluded from the scaling potfile; modeled explicitly as G1. It also anchors O1." == line for line in hera_members)
    assert any("# Member 2 is excluded from the scaling potfile; modeled explicitly as G2. It also anchors O2." == line for line in hera_members)
    assert any("# Member 3 is excluded from the scaling potfile; modeled explicitly as G3." == line for line in hera_members)
    assert any("# Member 4 is excluded from the scaling potfile; modeled explicitly as G4." == line for line in hera_members)
    assert any("# Member 5 is excluded from the scaling potfile; modeled explicitly as G5." == line for line in hera_members)
    assert any("# Member 6 is excluded from the scaling potfile; modeled explicitly as G6." == line for line in hera_members)
    assert not any("modeled explicitly as G9" in line for line in hera_members)
    assert not any("modeled explicitly as G60" in line for line in hera_members)
    commented_hera_id2 = next(line for line in hera_members if line.startswith("#          2 "))
    hera_id2_parts = commented_hera_id2.split()
    assert hera_id2_parts[:3] == ["#", "2", "-2.00000000"]
    assert hera_id2_parts[3] == "3.00000000"
    assert float(hera_id2_parts[4]) == pytest.approx(1.0)
    assert float(hera_id2_parts[5]) == pytest.approx(1.0)
    active_hera_member_lines = [line for line in hera_members if not line.startswith("#")]
    assert len(active_hera_member_lines) == 4
    assert [line.split()[0] for line in active_hera_member_lines] == ["60", "7", "8", "9"]
    assert active_hera_member_lines[0].split()[:6] == [
        "60",
        "-1.50000000",
        "1.10000000",
        "1.00000000",
        "1.00000000",
        "0.0000",
    ]

    ares_par = output / "ares" / "ares_lenscluster.par"
    parsed, potentials_df, images_df, arcs_df, potentials_with_priors = load_best_par(ares_par)
    assert len(potentials_df) == 7
    assert len(images_df) == 3
    assert len(arcs_df) == 0
    assert images_df["family_id"].nunique() == 2
    assert len(parsed["potfiles"][0]["catalog_df"]) == 5
    assert {item["id"] for item in potentials_with_priors} == {"O1", "O2", "G1", "G2", "G3", "G4", "G5"}
    assert parsed["potfiles"][0]["mag0"] == pytest.approx(18.5)
    assert parsed["potfiles"][0]["sigma_nominal"] == pytest.approx(100.0)
    assert parsed["potfiles"][0]["cutkpc_nominal"] == pytest.approx(270.0)
    ares_par_text = ares_par.read_text(encoding="utf-8")
    assert "x_centre -5.00000000" in ares_par_text
    assert "x_centre 8.00000000" in ares_par_text
    assert "# O2 is the second smooth dPIE clump, anchored on clgal_cat.txt member 3." in ares_par_text
    assert _limit_line(ares_par_text, "O1", "x_centre") == "x_centre 1 15.00000000 45.00000000 0.10000000"
    assert _limit_line(ares_par_text, "O1", "y_centre") == "y_centre 1 -85.00000000 -30.00000000 0.10000000"
    assert _limit_line(ares_par_text, "O2", "x_centre") == "x_centre 1 -45.00000000 -35.00000000 0.10000000"
    assert _limit_line(ares_par_text, "O2", "y_centre") == "y_centre 1 35.00000000 45.00000000 0.10000000"
    assert _limit_line(ares_par_text, "O1", "ellipticite") == "ellipticite 1 0.00000000 0.50000000 0.02000000"
    assert _limit_line(ares_par_text, "O2", "ellipticite") == "ellipticite 1 0.00000000 0.50000000 0.02000000"
    assert _limit_line(ares_par_text, "O1", "core_radius") == "core_radius 1 5.00000000 60.00000000 0.10000000"
    assert _limit_line(ares_par_text, "O2", "core_radius") == "core_radius 1 5.00000000 45.00000000 0.10000000"
    assert _limit_line(ares_par_text, "O1", "v_disp") == "v_disp 9 950.00000000 250.00000000 500.00000000 1800.00000000"
    assert _limit_line(ares_par_text, "O2", "v_disp") == "v_disp 9 950.00000000 175.00000000 600.00000000 1400.00000000"
    assert _limit_line(ares_par_text, "G1", "x_centre") == "x_centre 0 -5.00000000 0"
    assert _limit_line(ares_par_text, "G1", "y_centre") == "y_centre 0 6.00000000 0"
    expected_ares_explicit_core_prior = "core_radius_kpc 9 0.70000000 0.50000000 0.15000000 3.00000000"
    expected_ares_fixed_core = "core_radius_kpc 0 0.70000000 0"
    for component_id in ("G1", "G2"):
        assert _limit_line(ares_par_text, component_id, "core_radius_kpc") == expected_ares_explicit_core_prior
        component_block = ares_par_text.split(f"potentiel {component_id}", maxsplit=1)[1].split("limit", maxsplit=1)[0]
        assert "\n    core_radius_kpc 0.70000000\n" in component_block
    for component_id in ("G3", "G4", "G5"):
        assert _limit_line(ares_par_text, component_id, "core_radius_kpc") == expected_ares_fixed_core
        component_block = ares_par_text.split(f"potentiel {component_id}", maxsplit=1)[1].split("limit", maxsplit=1)[0]
        assert "\n    core_radius_kpc 0.70000000\n" in component_block
    assert _limit_line(ares_par_text, "G2", "x_centre") == "x_centre 0 3.00000000 0"
    assert _limit_line(ares_par_text, "G2", "y_centre") == "y_centre 0 4.00000000 0"
    assert _limit_line(ares_par_text, "G2", "v_disp") == "v_disp 1 43.67129597 218.35647984 1.00000000"
    assert _limit_line(ares_par_text, "G3", "x_centre") == "x_centre 0 8.00000000 0"
    assert _limit_line(ares_par_text, "G3", "y_centre") == "y_centre 0 -8.00000000 0"
    assert _limit_line(ares_par_text, "G4", "x_centre") == "x_centre 0 -9.00000000 0"
    assert _limit_line(ares_par_text, "G4", "y_centre") == "y_centre 0 9.00000000 0"
    assert _limit_line(ares_par_text, "G5", "x_centre") == "x_centre 0 -10.00000000 0"
    assert _limit_line(ares_par_text, "G5", "y_centre") == "y_centre 0 -10.00000000 0"
    assert "sigma 9 100.00000000 15.00000000 70.00000000 135.00000000" in ares_par_text
    assert "cutkpc 9 270.00000000 35.00000000 160.00000000 340.00000000" in ares_par_text
    assert "potentiel S1" not in ares_par_text
    assert "profil 14" not in ares_par_text
    assert "potentiel G4" in ares_par_text
    assert "potentiel G5" in ares_par_text
    assert "potentiel G31" not in ares_par_text
    assert "potentiel G11" not in ares_par_text
    assert "potentiel G17" not in ares_par_text
    assert all(" -5.00000000 " not in line for line in active_ares_member_lines)
    ares_specs, _ares_assignments, _ares_lens_models = _build_parameter_specs(potentials_with_priors)
    ares_sample_names = {spec.sample_name for spec in ares_specs}
    assert {"O1_x_centre", "O1_y_centre", "O2_x_centre", "O2_y_centre"} <= ares_sample_names
    assert {
        "G1_core_radius_kpc",
        "G2_core_radius_kpc",
        "G3_cut_radius_kpc",
        "G3_v_disp",
        "G4_cut_radius_kpc",
        "G4_v_disp",
        "G5_cut_radius_kpc",
        "G5_v_disp",
    } <= ares_sample_names
    assert not ({"G3_core_radius_kpc", "G4_core_radius_kpc", "G5_core_radius_kpc"} & ares_sample_names)
    assert not (
        {
            "G1_x_centre",
            "G1_y_centre",
            "G2_x_centre",
            "G2_y_centre",
            "G3_x_centre",
            "G3_y_centre",
            "G4_x_centre",
            "G4_y_centre",
            "G5_x_centre",
            "G5_y_centre",
        }
        & ares_sample_names
    )

    hera_par = output / "hera" / "hera_lenscluster.par"
    _hera_parsed, _hera_potentials_df, _hera_images_df, _hera_arcs_df, hera_potentials_with_priors = load_best_par(hera_par)
    hera_par_text = hera_par.read_text(encoding="utf-8")
    assert "# O1 is the first smooth dPIE clump, anchored on clgal_cat.txt member 1." in hera_par_text
    assert "# O2 is the second smooth dPIE clump, anchored on clgal_cat.txt member 2." in hera_par_text
    hera_o1_block = hera_par_text.split("potentiel O1", maxsplit=1)[1].split("potentiel O2", maxsplit=1)[0]
    hera_o2_block = hera_par_text.split("potentiel O2", maxsplit=1)[1].split("potentiel G1", maxsplit=1)[0]
    assert "\n    x_centre 9.00000000\n    y_centre 1.00000000" in hera_o1_block
    assert "\n    core_radius 8.00000000" in hera_o1_block
    assert "\n    angle_pos 30.00000000" in hera_o1_block
    assert "\n    v_disp 800.00000000" in hera_o1_block
    assert _limit_line(hera_par_text, "O1", "x_centre") == "x_centre 1 4.00000000 14.00000000 0.10000000"
    assert _limit_line(hera_par_text, "O1", "y_centre") == "y_centre 1 -4.00000000 6.00000000 0.10000000"
    assert "\n    x_centre -2.00000000\n    y_centre 3.00000000" in hera_o2_block
    assert "\n    core_radius 5.00000000" in hera_o2_block
    assert "\n    angle_pos 24.00000000" in hera_o2_block
    assert "\n    v_disp 700.00000000" in hera_o2_block
    assert _limit_line(hera_par_text, "O2", "x_centre") == "x_centre 1 -7.00000000 3.00000000 0.10000000"
    assert _limit_line(hera_par_text, "O2", "y_centre") == "y_centre 1 -2.00000000 8.00000000 0.10000000"
    assert _limit_line(hera_par_text, "G1", "x_centre") == "x_centre 0 9.00000000 0"
    assert _limit_line(hera_par_text, "G1", "y_centre") == "y_centre 0 1.00000000 0"
    assert _limit_line(hera_par_text, "G1", "core_radius_kpc") == "core_radius_kpc 0 0.15000000 0"
    assert _limit_line(hera_par_text, "G2", "x_centre") == "x_centre 0 -2.00000000 0"
    assert _limit_line(hera_par_text, "G2", "y_centre") == "y_centre 0 3.00000000 0"
    assert _limit_line(hera_par_text, "G3", "x_centre") == "x_centre 0 4.00000000 0"
    assert _limit_line(hera_par_text, "G3", "y_centre") == "y_centre 0 -4.00000000 0"
    assert _limit_line(hera_par_text, "G4", "x_centre") == "x_centre 0 5.00000000 0"
    assert _limit_line(hera_par_text, "G4", "y_centre") == "y_centre 0 5.00000000 0"
    assert _limit_line(hera_par_text, "G5", "x_centre") == "x_centre 0 6.00000000 0"
    assert _limit_line(hera_par_text, "G5", "y_centre") == "y_centre 0 6.00000000 0"
    assert _limit_line(hera_par_text, "G6", "x_centre") == "x_centre 0 7.00000000 0"
    assert _limit_line(hera_par_text, "G6", "y_centre") == "y_centre 0 7.00000000 0"
    assert "potentiel G3" in hera_par_text
    assert "potentiel G4" in hera_par_text
    assert "potentiel G5" in hera_par_text
    assert "potentiel G6" in hera_par_text
    assert "potentiel G9" not in hera_par_text
    assert "potentiel G60" not in hera_par_text
    assert "potentiel S1" not in hera_par_text
    assert "profil 14" not in hera_par_text
    assert "\n    gamma 0.04000000\n    angle_pos 40.00000000" not in hera_par_text
    assert hera_par_text.count("core_radius 1 2.00000000 15.00000000 0.10000000") == 2
    assert "sigma 9 96.70000000 40.00000000 30.00000000 250.00000000" in hera_par_text
    assert "cutkpc 9 33.00000000 25.00000000 3.00000000 250.00000000" in hera_par_text
    assert "vdslope 0 4.00000000 0" in hera_par_text
    assert "slope 0 4.00000000 0" in hera_par_text
    assert "vdslope 1 2.0 6.0 0.1" not in hera_par_text
    assert "slope   1 1.0 6.0 0.1" not in hera_par_text
    hera_specs, _hera_assignments, _hera_lens_models = _build_parameter_specs(hera_potentials_with_priors)
    hera_sample_names = {spec.sample_name for spec in hera_specs}
    assert {"O1_x_centre", "O1_y_centre", "O2_x_centre", "O2_y_centre"} <= hera_sample_names
    assert not (
        {
            "G1_x_centre",
            "G1_y_centre",
            "G2_x_centre",
            "G2_y_centre",
            "G3_x_centre",
            "G3_y_centre",
            "G4_x_centre",
            "G4_y_centre",
            "G5_x_centre",
            "G5_y_centre",
            "G6_x_centre",
            "G6_y_centre",
        }
        & hera_sample_names
    )


def test_validate_outputs_reports_generated_counts(tmp_path: Path) -> None:
    source = _write_ff_sims_fixture(tmp_path / "source")
    output = tmp_path / "data" / "ff_sims"
    builder.render(source_root=source, output_dir=output, clusters=["hera"])

    rows = builder.validate_outputs(output, clusters=["hera"])

    assert rows == [
        {
            "cluster_key": "hera",
            "n_potentials": 8,
            "n_images": 2,
            "n_image_families": 1,
            "n_potfiles": 1,
            "n_members": 4,
            "n_prior_components": 8,
        }
    ]
