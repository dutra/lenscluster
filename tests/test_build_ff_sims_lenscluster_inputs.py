from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

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
    assert {row["cluster_key"]: row["n_members"] for row in rows} == {"ares": 3, "hera": 6}
    assert {row["cluster_key"]: row["n_explicit_galaxies"] for row in rows} == {"ares": 3, "hera": 4}
    assert {row["cluster_key"]: row["explicit_galaxy_ids"] for row in rows} == {
        "ares": "1;2;3",
        "hera": "1;2;3;5",
    }
    assert {row["cluster_key"]: row["member_selection"] for row in rows} == {
        "ares": "F160W<22.00",
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
    assert not any("modeled explicitly as G4" in line for line in ares_members)
    assert not any("modeled explicitly as G5" in line for line in ares_members)
    assert any(line.startswith("#          1 -5.00000000  6.00000000") for line in ares_members)
    active_ares_member_lines = [line for line in ares_members if not line.startswith("#")]
    assert len(active_ares_member_lines) == 3
    assert [line.split()[0] for line in active_ares_member_lines] == ["4", "5", "6"]
    assert active_ares_member_lines[0].split()[:4] == ["4", "-9.00000000", "9.00000000", "1.00000000"]
    assert active_ares_member_lines[0].split()[6] == "19.400000"

    hera_members = (output / "hera" / "hera_cluster_members_potfile.cat").read_text(encoding="utf-8").splitlines()
    assert any("# Member 1 is excluded from the scaling potfile; modeled explicitly as G1. It also anchors O1." == line for line in hera_members)
    assert any("# Member 2 is excluded from the scaling potfile; modeled explicitly as G2. It also anchors O2." == line for line in hera_members)
    assert any("# Member 3 is excluded from the scaling potfile; modeled explicitly as G3." == line for line in hera_members)
    assert any("# Member 5 is excluded from the scaling potfile; modeled explicitly as G5." == line for line in hera_members)
    assert not any("modeled explicitly as G4" in line for line in hera_members)
    assert not any("modeled explicitly as G9" in line for line in hera_members)
    assert not any("modeled explicitly as G60" in line for line in hera_members)
    assert not any(line.endswith("modeled explicitly as G6.") for line in hera_members)
    commented_hera_id2 = next(line for line in hera_members if line.startswith("#          2 "))
    hera_id2_parts = commented_hera_id2.split()
    assert hera_id2_parts[:3] == ["#", "2", "-2.00000000"]
    assert hera_id2_parts[3] == "3.00000000"
    assert float(hera_id2_parts[4]) == pytest.approx(1.0)
    assert float(hera_id2_parts[5]) == pytest.approx(1.0)
    active_hera_member_lines = [line for line in hera_members if not line.startswith("#")]
    assert len(active_hera_member_lines) == 6
    assert [line.split()[0] for line in active_hera_member_lines] == ["4", "60", "6", "7", "8", "9"]
    assert active_hera_member_lines[0].split()[:6] == [
        "4",
        "5.00000000",
        "5.00000000",
        "1.00000000",
        "1.00000000",
        "0.0000",
    ]

    ares_par = output / "ares" / "ares_lenscluster.par"
    parsed, potentials_df, images_df, arcs_df, potentials_with_priors = load_best_par(ares_par)
    assert len(potentials_df) == 5
    assert len(images_df) == 3
    assert len(arcs_df) == 0
    assert images_df["family_id"].nunique() == 2
    assert len(parsed["potfiles"][0]["catalog_df"]) == 3
    assert {item["id"] for item in potentials_with_priors} == {"O1", "O2", "G1", "G2", "G3"}
    assert parsed["potfiles"][0]["mag0"] == pytest.approx(18.5)
    assert parsed["potfiles"][0]["sigma_nominal"] == pytest.approx(98.0)
    assert parsed["potfiles"][0]["cutkpc_nominal"] == pytest.approx(262.0)
    ares_par_text = ares_par.read_text(encoding="utf-8")
    assert "x_centre -5.00000000" in ares_par_text
    assert "x_centre 8.00000000" in ares_par_text
    assert "# O2 is the second smooth dPIE clump, anchored on clgal_cat.txt member 3." in ares_par_text
    assert "sigma 9 98.00000000 40.00000000 30.00000000 250.00000000" in ares_par_text
    assert "cutkpc 9 262.00000000 100.00000000 20.00000000 700.00000000" in ares_par_text
    assert "potentiel S1" not in ares_par_text
    assert "profil 14" not in ares_par_text
    assert "potentiel G4" not in ares_par_text
    assert "potentiel G5" not in ares_par_text
    assert all(" -5.00000000 " not in line for line in active_ares_member_lines)

    hera_par_text = (output / "hera" / "hera_lenscluster.par").read_text(encoding="utf-8")
    assert "# O1 is the first smooth dPIE clump, anchored on clgal_cat.txt member 1." in hera_par_text
    assert "# O2 is the second smooth dPIE clump, anchored on clgal_cat.txt member 2." in hera_par_text
    hera_o1_block = hera_par_text.split("potentiel O1", maxsplit=1)[1].split("potentiel O2", maxsplit=1)[0]
    hera_o2_block = hera_par_text.split("potentiel O2", maxsplit=1)[1].split("potentiel S1", maxsplit=1)[0]
    assert "\n    x_centre 9.00000000\n    y_centre 1.00000000" in hera_o1_block
    assert "\n    core_radius 8.00000000" in hera_o1_block
    assert "\n    angle_pos 30.00000000" in hera_o1_block
    assert "\n    v_disp 800.00000000" in hera_o1_block
    assert "limit O1\n    x_centre 1 14.00000000 24.00000000 0.10000000\n    y_centre 1 -2.00000000 7.00000000 0.10000000" in hera_o1_block
    assert "\n    x_centre -2.00000000\n    y_centre 3.00000000" in hera_o2_block
    assert "\n    core_radius 5.00000000" in hera_o2_block
    assert "\n    angle_pos 24.00000000" in hera_o2_block
    assert "\n    v_disp 700.00000000" in hera_o2_block
    assert "limit O2\n    x_centre 1 -5.00000000 5.00000000 0.10000000\n    y_centre 1 -4.00000000 4.00000000 0.10000000" in hera_o2_block
    assert "potentiel G3" in hera_par_text
    assert "potentiel G4" not in hera_par_text
    assert "potentiel G9" not in hera_par_text
    assert "potentiel G60" not in hera_par_text
    assert "potentiel G5" in hera_par_text
    assert not any(line == "potentiel G6" for line in hera_par_text.splitlines())
    assert "potentiel S1" in hera_par_text
    assert "profil 14" in hera_par_text
    assert "\n    gamma 0.04000000\n    angle_pos 40.00000000" in hera_par_text
    assert hera_par_text.count("core_radius 1 2.00000000 15.00000000 0.10000000") == 2
    assert "sigma 9 96.70000000 40.00000000 30.00000000 250.00000000" in hera_par_text
    assert "cutkpc 9 33.00000000 25.00000000 3.00000000 250.00000000" in hera_par_text
    assert "vdslope 0 4.00000000 0" in hera_par_text
    assert "slope 0 4.00000000 0" in hera_par_text
    assert "vdslope 1 2.0 6.0 0.1" not in hera_par_text
    assert "slope   1 1.0 6.0 0.1" not in hera_par_text


def test_validate_outputs_reports_generated_counts(tmp_path: Path) -> None:
    source = _write_ff_sims_fixture(tmp_path / "source")
    output = tmp_path / "data" / "ff_sims"
    builder.render(source_root=source, output_dir=output, clusters=["hera"])

    rows = builder.validate_outputs(output, clusters=["hera"])

    assert rows == [
        {
            "cluster_key": "hera",
            "n_potentials": 7,
            "n_images": 2,
            "n_image_families": 1,
            "n_potfiles": 1,
            "n_members": 6,
            "n_prior_components": 7,
        }
    ]
