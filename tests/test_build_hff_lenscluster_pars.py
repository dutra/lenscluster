from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from lenscluster.lenstool_parser import load_best_par


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_hff_lenscluster_pars.py"
spec = importlib.util.spec_from_file_location("build_hff_lenscluster_pars", SCRIPT_PATH)
assert spec is not None
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


def _minimal_config() -> dict:
    return {
        "version": 1,
        "defaults": {
            "image_sigma_arcsec": 0.5,
            "grid": {"nombre": 64, "polaire": 0},
            "cosmology": {"H0": 70.0, "omega": 0.3, "lambda": 0.7},
            "potfile": {
                "type": 81,
                "corekpc": 0.15,
                "mag0": 20.0,
                "sigma": 200.0,
                "cutkpc": 30.0,
            },
        },
        "clusters": {
            "toy": {
                "display_name": "Toy Cluster",
                "source_file": "source.par",
                "z_lens": 0.4,
                "runmode_reference": {"system": 3, "ra_deg": 10.0, "dec_deg": -1.0},
                "components": [
                    {
                        "id": "O1",
                        "label": "main halo",
                        "enabled": True,
                        "profile": 81,
                        "params": {
                            "x_centre": {"value": 1.0, "free": True, "sigma": 2.0},
                            "y_centre": {"value": -2.0, "free": True, "sigma": 3.0},
                            "ellipticite": {"value": 0.3, "free": True, "sigma": 0.1},
                            "angle_pos": {"value": 45.0, "free": True, "sigma": 10.0},
                            "core_radius_kpc": {"value": 20.0, "free": True, "sigma": 5.0},
                            "cut_radius_kpc": {"value": 800.0, "free": False},
                            "v_disp": {"value": 600.0, "free": True, "sigma": 80.0},
                            "z_lens": {"value": 0.4, "free": False},
                        },
                    },
                    {
                        "id": "S1",
                        "label": "external shear",
                        "enabled": True,
                        "profile": 14,
                        "params": {
                            "gamma": {"value": 0.05, "free": True, "sigma": 0.01},
                            "angle_pos": {"value": 12.0, "free": True, "sigma": 5.0},
                            "kappa": {"value": 0.0, "free": False},
                            "z_lens": {"value": 0.4, "free": False},
                        },
                    },
                ],
            }
        },
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "large_halos_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_default_paths_write_generated_products_under_results() -> None:
    parser = builder.build_arg_parser()
    args = parser.parse_args(["render"])

    assert builder.DEFAULT_CONFIG_PATH == Path("results") / "hff_lenscluster_pars" / "large_halos_config.yaml"
    assert builder.DEFAULT_OUTPUT_DIR == Path("results") / "hff_lenscluster_pars"
    assert args.config == builder.DEFAULT_CONFIG_PATH
    assert args.output_dir is None


def test_validate_config_requires_requested_cluster() -> None:
    with pytest.raises(ValueError, match="missing requested cluster"):
        builder.validate_config(_minimal_config(), ["missing"])


def test_validate_config_requires_sigma_for_free_fields() -> None:
    config = _minimal_config()
    del config["clusters"]["toy"]["components"][0]["params"]["x_centre"]["sigma"]

    with pytest.raises(ValueError, match="free but missing sigma"):
        builder.validate_config(config)


def test_render_writes_normal_priors_centered_on_yaml_values(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _minimal_config())

    rows = builder.render(config_path)

    assert len(rows) == 1
    par_path = tmp_path / "toy" / "toy_lenscluster.par"
    text = par_path.read_text(encoding="utf-8")
    limit_lines = [line.strip() for line in text.splitlines() if line.strip().startswith(("x_centre", "gamma"))]
    assert "x_centre 3 1.000000 2.000000 0.01000000" in limit_lines
    assert "gamma 3 0.05000000 0.01000000 0.00100000" in limit_lines
    assert "cut_radius_kpc 3" not in text
    assert "v_disp 3 600.000000 80.000000 0.10000000" in text


def test_render_outputs_blank_catalogs_and_loadable_par(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _minimal_config())

    builder.render(config_path)
    par_path = tmp_path / "toy" / "toy_lenscluster.par"
    obs_path = tmp_path / "toy" / "toy_obs_arcs.cat"
    potfile_path = tmp_path / "toy" / "toy_cluster_members_potfile.cat"

    assert obs_path.read_text(encoding="utf-8") == "#REFERENCE 3\n"
    assert potfile_path.read_text(encoding="utf-8") == "#REFERENCE 3\n"
    parsed, potentials_df, images_df, arcs_df, potentials_with_priors = load_best_par(par_path)
    assert len(potentials_df) == 2
    assert len(images_df) == 0
    assert len(arcs_df) == 0
    assert len(parsed["potfiles"]) == 1
    assert len(parsed["potfiles"][0]["catalog_df"]) == 0
    assert {item["id"] for item in potentials_with_priors} == {"O1", "S1"}


def test_validate_outputs_fails_when_rendered_par_is_missing(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _minimal_config())

    with pytest.raises(FileNotFoundError, match="Generated par file is missing"):
        builder.validate_outputs(config_path)
