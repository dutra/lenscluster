#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("results/hff_lenscluster_pars/large_halos_config.yaml")
DEFAULT_OUTPUT_DIR = Path("results/hff_lenscluster_pars")
DP_IE_PROFILE = 81
SHEAR_PROFILE = 14
DP_IE_FIELDS = (
    "x_centre",
    "y_centre",
    "ellipticite",
    "angle_pos",
    "core_radius_kpc",
    "cut_radius_kpc",
    "v_disp",
    "z_lens",
)
SHEAR_FIELDS = ("gamma", "angle_pos", "kappa", "z_lens")
DEFAULT_STEPS = {
    "x_centre": 0.01,
    "y_centre": 0.01,
    "ellipticite": 0.01,
    "angle_pos": 0.1,
    "core_radius_kpc": 0.1,
    "cut_radius_kpc": 0.1,
    "v_disp": 0.1,
    "gamma": 0.001,
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config '{config_path}' must contain a YAML mapping.")
    return loaded


def _as_float(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric.") from exc
    return result


def _cluster_items(config: dict[str, Any], requested: list[str] | None = None) -> list[tuple[str, dict[str, Any]]]:
    clusters = config.get("clusters")
    if not isinstance(clusters, dict) or not clusters:
        raise ValueError("Config must define a non-empty 'clusters' mapping.")
    keys = requested if requested else list(clusters.keys())
    result: list[tuple[str, dict[str, Any]]] = []
    for key in keys:
        if key not in clusters:
            raise ValueError(f"Config is missing requested cluster '{key}'.")
        cluster = clusters[key]
        if not isinstance(cluster, dict):
            raise ValueError(f"Cluster '{key}' must be a mapping.")
        result.append((str(key), cluster))
    return result


def _params_for_component(cluster_key: str, component: dict[str, Any]) -> dict[str, dict[str, Any]]:
    params = component.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"Cluster '{cluster_key}' component '{component.get('id')}' is missing params mapping.")
    normalized: dict[str, dict[str, Any]] = {}
    for field_name, field_config in params.items():
        if not isinstance(field_config, dict):
            raise ValueError(
                f"Cluster '{cluster_key}' component '{component.get('id')}' field '{field_name}' must be a mapping."
            )
        if "value" not in field_config:
            raise ValueError(
                f"Cluster '{cluster_key}' component '{component.get('id')}' field '{field_name}' is missing value."
            )
        normalized[str(field_name)] = field_config
    return normalized


def validate_config(config: dict[str, Any], clusters: list[str] | None = None) -> None:
    defaults = config.get("defaults", {})
    if defaults is not None and not isinstance(defaults, dict):
        raise ValueError("Config 'defaults' must be a mapping when present.")
    for cluster_key, cluster in _cluster_items(config, clusters):
        reference = cluster.get("runmode_reference")
        if not isinstance(reference, dict):
            raise ValueError(f"Cluster '{cluster_key}' is missing runmode_reference mapping.")
        for field_name in ("system", "ra_deg", "dec_deg"):
            if field_name not in reference:
                raise ValueError(f"Cluster '{cluster_key}' runmode_reference is missing '{field_name}'.")
        _as_float(reference["ra_deg"], f"{cluster_key}.runmode_reference.ra_deg")
        _as_float(reference["dec_deg"], f"{cluster_key}.runmode_reference.dec_deg")
        if "z_lens" not in cluster:
            raise ValueError(f"Cluster '{cluster_key}' is missing z_lens.")
        _as_float(cluster["z_lens"], f"{cluster_key}.z_lens")
        components = cluster.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError(f"Cluster '{cluster_key}' must define at least one component.")
        enabled_count = 0
        for idx, component in enumerate(components):
            if not isinstance(component, dict):
                raise ValueError(f"Cluster '{cluster_key}' component #{idx} must be a mapping.")
            component_id = component.get("id")
            if component_id is None or str(component_id).strip() == "":
                raise ValueError(f"Cluster '{cluster_key}' component #{idx} is missing id.")
            profile = int(component.get("profile", -1))
            if profile not in {DP_IE_PROFILE, SHEAR_PROFILE}:
                raise ValueError(f"Cluster '{cluster_key}' component '{component_id}' has unsupported profile {profile}.")
            if bool(component.get("enabled", True)):
                enabled_count += 1
            params = _params_for_component(cluster_key, component)
            required_fields = DP_IE_FIELDS if profile == DP_IE_PROFILE else ("gamma", "angle_pos")
            for field_name in required_fields:
                if field_name not in params:
                    raise ValueError(f"Cluster '{cluster_key}' component '{component_id}' is missing '{field_name}'.")
            for field_name, field_config in params.items():
                _as_float(field_config["value"], f"{cluster_key}.{component_id}.{field_name}.value")
                if bool(field_config.get("free", False)):
                    if "sigma" not in field_config:
                        raise ValueError(
                            f"Cluster '{cluster_key}' component '{component_id}' field '{field_name}' is free but missing sigma."
                        )
                    sigma = _as_float(field_config["sigma"], f"{cluster_key}.{component_id}.{field_name}.sigma")
                    if sigma <= 0.0:
                        raise ValueError(
                            f"Cluster '{cluster_key}' component '{component_id}' field '{field_name}' sigma must be positive."
                        )
        if enabled_count == 0:
            raise ValueError(f"Cluster '{cluster_key}' has no enabled components.")


def _format_float(value: Any) -> str:
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:.6f}"
    if abs(number) >= 100:
        return f"{number:.6f}"
    if abs(number) >= 1:
        return f"{number:.6f}"
    return f"{number:.8f}"


def _field_value(params: dict[str, dict[str, Any]], field_name: str, fallback: Any | None = None) -> float:
    if field_name in params:
        return _as_float(params[field_name]["value"], field_name)
    if fallback is None:
        raise ValueError(f"Missing required field '{field_name}'.")
    return _as_float(fallback, field_name)


def _free_fields(params: dict[str, dict[str, Any]], field_order: tuple[str, ...]) -> list[tuple[str, float, float, float]]:
    fields: list[tuple[str, float, float, float]] = []
    for field_name in field_order:
        field_config = params.get(field_name)
        if not field_config or not bool(field_config.get("free", False)):
            continue
        mean = _as_float(field_config["value"], field_name)
        sigma = _as_float(field_config["sigma"], field_name)
        step = _as_float(field_config.get("step", DEFAULT_STEPS.get(field_name, 0.1)), field_name)
        fields.append((field_name, mean, sigma, step))
    return fields


def _render_par(cluster_key: str, cluster: dict[str, Any], output_dir: Path, defaults: dict[str, Any]) -> str:
    reference = cluster["runmode_reference"]
    grid = {**defaults.get("grid", {}), **cluster.get("grid", {})}
    cosmology = {**defaults.get("cosmology", {}), **cluster.get("cosmology", {})}
    potfile = {**defaults.get("potfile", {}), **cluster.get("potfile", {})}
    z_lens = _as_float(cluster["z_lens"], f"{cluster_key}.z_lens")
    obs_file = f"{cluster_key}_obs_arcs.cat"
    potfile_file = f"{cluster_key}_cluster_members_potfile.cat"
    lines = [
        "# Generated by scripts/build_hff_lenscluster_pars.py from large_halos_config.yaml",
        f"# cluster_key {cluster_key}",
        f"# source_file {cluster.get('source_file', '')}",
        "runmode",
        f"    reference {int(reference['system'])} {_format_float(reference['ra_deg'])} {_format_float(reference['dec_deg'])}",
        "    end",
        "grille",
        f"    nombre {int(grid.get('nombre', 128))}",
        f"    polaire {int(grid.get('polaire', 0))}",
        "    end",
        "image",
        f"    multfile 1 {obs_file}",
        "    forme -2",
        "    mult_wcs 1",
        f"    sigposArcsec {_format_float(defaults.get('image_sigma_arcsec', 0.5))}",
        "    end",
    ]

    limit_blocks: list[tuple[str, list[tuple[str, float, float, float]]]] = []
    for component in cluster["components"]:
        if not bool(component.get("enabled", True)):
            continue
        profile = int(component["profile"])
        component_id = str(component["id"])
        params = _params_for_component(cluster_key, component)
        lines.append(f"potentiel {component_id}")
        if component.get("label"):
            lines.append(f"    # {component['label']}")
        lines.append(f"    profil {profile}")
        if profile == DP_IE_PROFILE:
            for field_name in DP_IE_FIELDS:
                value = _field_value(params, field_name, z_lens if field_name == "z_lens" else None)
                lines.append(f"    {field_name} {_format_float(value)}")
            free_fields = _free_fields(params, DP_IE_FIELDS)
        else:
            for field_name in SHEAR_FIELDS:
                if field_name == "z_lens":
                    value = _field_value(params, field_name, z_lens)
                elif field_name == "kappa" and field_name not in params:
                    value = 0.0
                else:
                    value = _field_value(params, field_name)
                lines.append(f"    {field_name} {_format_float(value)}")
            free_fields = _free_fields(params, SHEAR_FIELDS)
        lines.append("    end")
        if free_fields:
            limit_blocks.append((component_id, free_fields))

    for component_id, free_fields in limit_blocks:
        lines.append(f"limit {component_id}")
        for field_name, mean, sigma, step in free_fields:
            lines.append(f"    {field_name} 3 {_format_float(mean)} {_format_float(sigma)} {_format_float(step)}")
        lines.append("    end")

    lines.extend(
        [
            "potfile",
            f"    filein 3 {potfile_file}",
            f"    zlens {_format_float(potfile.get('zlens', z_lens))}",
            f"    type {int(potfile.get('type', DP_IE_PROFILE))}",
            f"    corekpc {_format_float(potfile.get('corekpc', 0.15))}",
            f"    mag0 {_format_float(potfile.get('mag0', 20.0))}",
            f"    sigma {_format_float(potfile.get('sigma', 200.0))}",
            f"    cutkpc {_format_float(potfile.get('cutkpc', 30.0))}",
            f"    vdslope {_format_float(potfile.get('vdslope', 4.0))}",
            f"    slope {_format_float(potfile.get('slope', 4.0))}",
            "    end",
            "cosmologie",
            f"    H0 {_format_float(cosmology.get('H0', 70.0))}",
            f"    omega {_format_float(cosmology.get('omega', 0.3))}",
            f"    lambda {_format_float(cosmology.get('lambda', 0.7))}",
            "    end",
            "champ",
            f"    dmax {_format_float(cluster.get('field_dmax_arcsec', defaults.get('field_dmax_arcsec', 220.0)))}",
            "    end",
            "fini",
            "",
        ]
    )
    return "\n".join(lines)


def _write_blank_catalogs(cluster_key: str, cluster_dir: Path) -> tuple[Path, Path]:
    obs_path = cluster_dir / f"{cluster_key}_obs_arcs.cat"
    potfile_path = cluster_dir / f"{cluster_key}_cluster_members_potfile.cat"
    obs_path.write_text("#REFERENCE 3\n", encoding="utf-8")
    potfile_path.write_text("#REFERENCE 3\n", encoding="utf-8")
    return obs_path, potfile_path


def _write_components_csv(cluster_key: str, cluster: dict[str, Any], cluster_dir: Path) -> Path:
    path = cluster_dir / f"{cluster_key}_source_components.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_key",
                "component_id",
                "label",
                "enabled",
                "profile",
                "field",
                "value",
                "free",
                "sigma",
                "source_file",
            ],
        )
        writer.writeheader()
        for component in cluster["components"]:
            params = _params_for_component(cluster_key, component)
            for field_name, field_config in params.items():
                writer.writerow(
                    {
                        "cluster_key": cluster_key,
                        "component_id": component["id"],
                        "label": component.get("label", ""),
                        "enabled": bool(component.get("enabled", True)),
                        "profile": int(component["profile"]),
                        "field": field_name,
                        "value": field_config.get("value"),
                        "free": bool(field_config.get("free", False)),
                        "sigma": field_config.get("sigma", ""),
                        "source_file": cluster.get("source_file", ""),
                    }
                )
    return path


def render(config_path: str | Path, output_dir: str | Path | None = None, clusters: list[str] | None = None) -> list[dict[str, Any]]:
    config = load_config(config_path)
    validate_config(config, clusters)
    defaults = config.get("defaults", {}) or {}
    root = Path(output_dir) if output_dir is not None else Path(config_path).parent
    root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for cluster_key, cluster in _cluster_items(config, clusters):
        cluster_dir = root / cluster_key
        cluster_dir.mkdir(parents=True, exist_ok=True)
        par_path = cluster_dir / f"{cluster_key}_lenscluster.par"
        par_path.write_text(_render_par(cluster_key, cluster, cluster_dir, defaults), encoding="utf-8")
        obs_path, potfile_path = _write_blank_catalogs(cluster_key, cluster_dir)
        components_path = _write_components_csv(cluster_key, cluster, cluster_dir)
        n_enabled = sum(1 for item in cluster["components"] if bool(item.get("enabled", True)))
        manifest_rows.append(
            {
                "cluster_key": cluster_key,
                "display_name": cluster.get("display_name", ""),
                "z_lens": cluster.get("z_lens", ""),
                "n_enabled_components": n_enabled,
                "par_path": str(par_path),
                "obs_arcs_path": str(obs_path),
                "potfile_path": str(potfile_path),
                "source_components_path": str(components_path),
                "source_file": cluster.get("source_file", ""),
            }
        )
    _write_manifest(root / "hff_lenscluster_par_manifest.csv", manifest_rows)
    return manifest_rows


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "cluster_key",
        "display_name",
        "z_lens",
        "n_enabled_components",
        "par_path",
        "obs_arcs_path",
        "potfile_path",
        "source_components_path",
        "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_outputs(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    clusters: list[str] | None = None,
) -> list[dict[str, Any]]:
    from lenscluster.lenstool_parser import load_best_par

    config = load_config(config_path)
    validate_config(config, clusters)
    root = Path(output_dir) if output_dir is not None else Path(config_path).parent
    rows: list[dict[str, Any]] = []
    for cluster_key, _cluster in _cluster_items(config, clusters):
        par_path = root / cluster_key / f"{cluster_key}_lenscluster.par"
        if not par_path.exists():
            raise FileNotFoundError(f"Generated par file is missing for cluster '{cluster_key}': {par_path}")
        parsed, potentials_df, images_df, potentials_with_priors = load_best_par(par_path)
        rows.append(
            {
                "cluster_key": cluster_key,
                "n_potentials": len(potentials_df),
                "n_images": len(images_df),
                "n_potfiles": len(parsed.get("potfiles", [])),
                "n_prior_components": sum(1 for item in potentials_with_priors if item.get("priors")),
            }
        )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    for command in ("render", "validate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
        sub.add_argument("--output-dir", type=Path, default=None)
        sub.add_argument("--clusters", nargs="+", default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--clusters", nargs="+", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    command = args.command or "render"
    if command == "render":
        rows = render(args.config, args.output_dir, args.clusters)
        print(f"Rendered {len(rows)} HFF lenscluster par files.")
        return 0
    if command == "validate":
        rows = validate_outputs(args.config, args.output_dir, args.clusters)
        print(f"Validated {len(rows)} HFF lenscluster par files.")
        return 0
    parser.error(f"Unsupported command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
