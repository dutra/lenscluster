from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy import constants as astro_const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_lenscluster_validation")
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .plotting import _plot_corner, _plot_potfile_corner, _scaling_parameter_subset

ORIGINAL_DPIE_PROFILE_NAME = "PJAFFE_ELLIPSE_POTENTIAL"


CHIRES_COLUMNS = (
    "index",
    "family_id",
    "z",
    "n_arcs",
    "chi_total",
    "chi_x",
    "chi_y",
    "chi_a",
    "source_rms_arcsec",
    "image_rms_arcsec",
    "dx_arcsec",
    "dy_arcsec",
    "n_warn",
)


@dataclass(frozen=True)
class DPIETruth:
    potential_id: str
    x_centre: float
    y_centre: float
    ellipticite: float
    angle_pos: float
    core_radius_arcsec: float
    cut_radius_arcsec: float
    v_disp: float


@dataclass(frozen=True)
class SourceTruth:
    family_id: str
    beta_x: float
    beta_y: float
    z_source: float


@dataclass(frozen=True)
class SingleBCGMockConfig:
    z_lens: float = 0.396
    reference_ra_deg: float = 64.0381417
    reference_dec_deg: float = -24.0674722
    pos_sigma_arcsec: float = 0.15
    seed: int = 12345
    source_redshift: float = 2.0
    source_redshifts: tuple[float, ...] = (1.5, 2.0, 3.0)
    source_sigma_int_arcsec: float = 0.05
    n_families: int = 3
    min_images_per_family: int = 2
    max_sources_to_try: int = 400
    n_subhalos: int = 0
    subhalo_field_radius_arcsec: float = 65.0
    subhalo_mag0: float = 17.0
    subhalo_mag_mean: float = 20.5
    subhalo_mag_sigma: float = 1.1
    subhalo_sigma_ref: float = 245.0
    subhalo_sigma_ref_std: float = 35.0
    subhalo_sigma_scatter_dex: float = 0.07
    subhalo_core_radius_arcsec: float = 0.0001
    subhalo_cut_radius_arcsec: float = 35.0
    subhalo_cut_lower_arcsec: float = 8.0
    subhalo_cut_upper_arcsec: float = 80.0
    subhalo_cut_scatter_dex: float = 0.20
    subhalo_vdslope: float = 3.33333
    subhalo_slope: float = 3.33333
    halo: DPIETruth = DPIETruth(
        potential_id="halo",
        x_centre=0.0,
        y_centre=0.0,
        ellipticite=0.28,
        angle_pos=35.0,
        core_radius_arcsec=5.0,
        cut_radius_arcsec=220.0,
        v_disp=760.0,
    )
    bcg: DPIETruth = DPIETruth(
        potential_id="bcg",
        x_centre=0.35,
        y_centre=-0.22,
        ellipticite=0.12,
        angle_pos=10.0,
        core_radius_arcsec=0.15,
        cut_radius_arcsec=35.0,
        v_disp=285.0,
    )


@dataclass(frozen=True)
class MockClusterPaths:
    root: Path
    par_path: Path
    image_catalog_path: Path
    truth_path: Path
    mock_images_path: Path


def _parse_chires_float(value: str) -> float | None:
    if value.upper() == "N/A":
        return None
    return float(value)


def load_chires_table(path: str | Path) -> pd.DataFrame:
    """Load a Lenstool ``chires.dat`` table.

    The file includes one row per image plus one summary row per family. Numeric
    ``N/A`` cells are returned as missing values by pandas.
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("chi ") or line.startswith("N "):
                continue
            parts = line.split()
            if len(parts) != len(CHIRES_COLUMNS):
                continue
            row: dict[str, Any] = {
                "index": int(parts[0]),
                "family_id": parts[1],
                "z": float(parts[2]),
                "n_arcs": int(parts[3]),
                "n_warn": int(parts[12]),
            }
            for column, raw_value in zip(CHIRES_COLUMNS[4:12], parts[4:12]):
                row[column] = _parse_chires_float(raw_value)
            rows.append(row)
    return pd.DataFrame(rows, columns=CHIRES_COLUMNS)


def load_chires_family_summary(path: str | Path) -> pd.DataFrame:
    """Return only family-summary rows from a Lenstool ``chires.dat`` table."""
    table = load_chires_table(path)
    if table.empty:
        return table
    summary = table[table["n_arcs"] > 1].copy()
    return summary.sort_values(["index", "family_id"]).reset_index(drop=True)


def _config_to_jsonable(config: SingleBCGMockConfig, kpc_per_arcsec: float) -> dict[str, Any]:
    payload = asdict(config)
    payload["source_redshifts"] = [float(value) for value in config.source_redshifts]
    payload["subhalo_sigma_log_scatter"] = _dex_scatter_to_ln(config.subhalo_sigma_scatter_dex)
    payload["subhalo_cut_log_scatter"] = _dex_scatter_to_ln(config.subhalo_cut_scatter_dex)
    for component_name in ("halo", "bcg"):
        component = payload[component_name]
        component["core_radius_kpc"] = float(component["core_radius_arcsec"]) * float(kpc_per_arcsec)
        component["cut_radius_kpc"] = float(component["cut_radius_arcsec"]) * float(kpc_per_arcsec)
    return payload


def _dex_scatter_to_ln(scatter_dex: float) -> float:
    return float(np.log(10.0) * max(float(scatter_dex), 0.0))


def _scaled_subhalo_params(row: dict[str, Any], config: SingleBCGMockConfig) -> DPIETruth:
    luminosity_ratio = float(row["luminosity_ratio"])
    sigma_log_offset = float(row.get("sigma_log_offset", 0.0))
    cut_log_offset = float(row.get("cut_log_offset", 0.0))
    return DPIETruth(
        potential_id=str(row["id"]),
        x_centre=float(row["x_arcsec"]),
        y_centre=float(row["y_arcsec"]),
        ellipticite=float(row["ellipticite"]),
        angle_pos=float(row["angle_pos"]),
        core_radius_arcsec=float(config.subhalo_core_radius_arcsec * luminosity_ratio**0.5),
        cut_radius_arcsec=float(
            config.subhalo_cut_radius_arcsec
            * luminosity_ratio ** (2.0 / config.subhalo_slope)
            * np.exp(cut_log_offset)
        ),
        v_disp=float(config.subhalo_sigma_ref * luminosity_ratio ** (1.0 / config.subhalo_vdslope) * np.exp(sigma_log_offset)),
    )


def _generate_subhalo_catalog(config: SingleBCGMockConfig, rng: np.random.Generator) -> list[dict[str, Any]]:
    if config.n_subhalos <= 0:
        return []
    rows: list[dict[str, Any]] = []
    # A compact projected member distribution: many faint galaxies at large radius,
    # with a few bright perturbers deliberately allowed near the strong-lensing zone.
    for idx in range(config.n_subhalos):
        if idx < min(4, config.n_subhalos):
            radius = rng.uniform(4.0, 18.0)
            mag = rng.normal(config.subhalo_mag0 + 1.4, 0.45)
        else:
            radius = min(config.subhalo_field_radius_arcsec, rng.gamma(shape=2.0, scale=14.0) + 3.0)
            mag = rng.normal(config.subhalo_mag_mean, config.subhalo_mag_sigma)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        x_arcsec = float(radius * np.cos(theta))
        y_arcsec = float(radius * np.sin(theta))
        q = float(rng.uniform(0.65, 1.0))
        ellipticite = 1.0 - q
        angle_pos = float(rng.uniform(-90.0, 90.0))
        mag = float(np.clip(mag, config.subhalo_mag0 - 0.2, config.subhalo_mag0 + 6.0))
        luminosity_ratio = float(10.0 ** (-0.4 * (mag - config.subhalo_mag0)))
        sigma_log_offset = float(rng.normal(0.0, _dex_scatter_to_ln(config.subhalo_sigma_scatter_dex)))
        cut_log_offset = float(rng.normal(0.0, _dex_scatter_to_ln(config.subhalo_cut_scatter_dex)))
        rows.append(
            {
                "id": f"member{idx + 1:03d}",
                "x_arcsec": x_arcsec,
                "y_arcsec": y_arcsec,
                "catalog_a": 0.3734,
                "catalog_b": 0.3734 * q,
                "catalog_theta": angle_pos,
                "catalog_mag": mag,
                "catalog_lum": luminosity_ratio,
                "ellipticite": ellipticite,
                "angle_pos": angle_pos,
                "luminosity_ratio": luminosity_ratio,
                "sigma_log_offset": sigma_log_offset,
                "cut_log_offset": cut_log_offset,
            }
        )
    return rows


def _component_kwargs(component: DPIETruth, config: SingleBCGMockConfig, z_source: float, cosmo: Any) -> dict[str, float]:
    sigma0 = _dpie_sigma0_from_vel_disp_local(
        component.v_disp,
        component.core_radius_arcsec,
        component.cut_radius_arcsec,
        config.z_lens,
        z_source,
        cosmo,
    )
    q = max(1.0e-3, 1.0 - component.ellipticite)
    phi = math.radians(component.angle_pos)
    fac = (1.0 - q) / (1.0 + q)
    return {
        "sigma0": float(sigma0),
        "Ra": float(component.core_radius_arcsec),
        "Rs": float(component.cut_radius_arcsec),
        "e1": float(fac * math.cos(2.0 * phi)),
        "e2": float(fac * math.sin(2.0 * phi)),
        "center_x": float(component.x_centre),
        "center_y": float(component.y_centre),
    }


def _dpie_sigma0_from_vel_disp_local(
    vel_disp: float,
    ra_arcsec: float,
    rs_arcsec: float,
    z_lens: float,
    z_source: float,
    cosmo: Any,
) -> float:
    ds = cosmo.angular_diameter_distance(z_source).to(u.m).value
    dds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to(u.m).value
    if ds <= 0 or dds <= 0 or rs_arcsec <= ra_arcsec:
        return float("nan")
    arcsec_rad = np.deg2rad(1.0 / 3600.0)
    c_si = astro_const.c.to_value(u.m / u.s)
    sigma0 = (
        (float(vel_disp) * 1000.0 / c_si) ** 2
        * 2.0
        * np.pi
        * dds
        / ds
        * ((float(rs_arcsec) - float(ra_arcsec)) / (float(rs_arcsec) * float(ra_arcsec)))
        / arcsec_rad
    )
    return float(sigma0)


def _truth_parameter_values(config: SingleBCGMockConfig, kpc_per_arcsec: float) -> dict[str, float]:
    truth: dict[str, float] = {}
    for component in (config.halo, config.bcg):
        prefix = component.potential_id
        truth[f"{prefix}.x_centre"] = float(component.x_centre)
        truth[f"{prefix}.y_centre"] = float(component.y_centre)
        truth[f"{prefix}.ellipticite"] = float(component.ellipticite)
        truth[f"{prefix}.angle_pos"] = float(component.angle_pos)
        truth[f"{prefix}.core_radius_kpc"] = float(component.core_radius_arcsec) * float(kpc_per_arcsec)
        truth[f"{prefix}.cut_radius_kpc"] = float(component.cut_radius_arcsec) * float(kpc_per_arcsec)
        truth[f"{prefix}.v_disp"] = float(component.v_disp)
    truth["source.sigma_int"] = float(config.source_sigma_int_arcsec)
    if config.n_subhalos > 0:
        truth["potfile.sigma"] = float(config.subhalo_sigma_ref)
        truth["potfile.cutkpc"] = float(config.subhalo_cut_radius_arcsec) * float(kpc_per_arcsec)
        if config.subhalo_sigma_scatter_dex > 0.0:
            truth["potfile.sigma_log_scatter"] = _dex_scatter_to_ln(config.subhalo_sigma_scatter_dex)
        if config.subhalo_cut_scatter_dex > 0.0:
            truth["potfile.cut_log_scatter"] = _dex_scatter_to_ln(config.subhalo_cut_scatter_dex)
    return truth


def _write_member_catalog(path: Path, subhalos: list[dict[str, Any]]) -> None:
    path.write_text(
        "#REFERENCE 3\n"
        + "\n".join(
            (
                f"{row['id']:>10s} {row['x_arcsec']: .8f} {row['y_arcsec']: .8f} "
                f"{row['catalog_a']:.6f} {row['catalog_b']:.6f} {row['catalog_theta']:.4f} "
                f"{row['catalog_mag']:.6f} {row['catalog_lum']:.8e}"
            )
            for row in subhalos
        )
        + ("\n" if subhalos else ""),
        encoding="utf-8",
    )


def _write_single_bcg_par(path: Path, config: SingleBCGMockConfig, subhalos: list[dict[str, Any]] | None = None) -> None:
    def potential_block(component: DPIETruth) -> str:
        return f"""
potentiel {component.potential_id}
    profil      81
    x_centre    {component.x_centre:.8f}
    y_centre    {component.y_centre:.8f}
    ellipticite {component.ellipticite:.8f}
    angle_pos   {component.angle_pos:.8f}
    core_radius {component.core_radius_arcsec:.8f}
    cut_radius  {component.cut_radius_arcsec:.8f}
    v_disp      {component.v_disp:.8f}
    z_lens      {config.z_lens:.8f}
    end

limit {component.potential_id}
    x_centre    1 {component.x_centre - 8.0:.8f} {component.x_centre + 8.0:.8f} 0.05
    y_centre    1 {component.y_centre - 8.0:.8f} {component.y_centre + 8.0:.8f} 0.05
    ellipticite 1 0.0 0.75 0.02
    angle_pos   1 -90.0 90.0 0.5
    core_radius 1 {max(0.001, component.core_radius_arcsec * 0.2):.8f} {component.core_radius_arcsec * 3.0:.8f} 0.02
    cut_radius  1 {component.cut_radius_arcsec * 0.4:.8f} {component.cut_radius_arcsec * 2.0:.8f} 0.5
    v_disp      1 {component.v_disp * 0.55:.8f} {component.v_disp * 1.45:.8f} 1.0
    end
"""

    subhalos = subhalos or []
    potfile_block = ""
    if subhalos:
        potfile_block = f"""
potfile
    filein 3 members.cat
    type 81
    mag0 {config.subhalo_mag0:.8f}
    core {config.subhalo_core_radius_arcsec:.8f}
    z_lens {config.z_lens:.8f}
    sigma 3 {config.subhalo_sigma_ref:.8f} {config.subhalo_sigma_ref_std:.8f}
    cut 1 {config.subhalo_cut_lower_arcsec:.8f} {config.subhalo_cut_upper_arcsec:.8f}
    vdslope 0 {config.subhalo_vdslope:.8f} 0
    slope 0 {config.subhalo_slope:.8f} 0
    end
"""

    content = f"""runmode
    reference 3 {config.reference_ra_deg:.8f} {config.reference_dec_deg:.8f}
    inverse   3 0.5 100
    end

image
    multfile    1 {path.with_name("obs_arcs.cat").name}
    sigposArcsec {config.pos_sigma_arcsec:.8f}
    forme       0
    end

cosmology
    H0 70.0
    omega 0.3
    lambda 0.7
    end

grille
    nlens 1000
    nlens_opt 6
    nombre 256
    end
{potential_block(config.halo)}
{potential_block(config.bcg)}
{potfile_block}
finish
"""
    path.write_text(content, encoding="utf-8")


def _deduplicate_images(x: np.ndarray, y: np.ndarray, tolerance: float = 1.0e-3) -> tuple[np.ndarray, np.ndarray]:
    keep: list[int] = []
    for idx, (x_i, y_i) in enumerate(zip(x, y)):
        if not np.isfinite(x_i) or not np.isfinite(y_i):
            continue
        if all(math.hypot(float(x_i - x[j]), float(y_i - y[j])) > tolerance for j in keep):
            keep.append(idx)
    return np.asarray(x[keep], dtype=float), np.asarray(y[keep], dtype=float)


def _candidate_sources(config: SingleBCGMockConfig, rng: np.random.Generator) -> list[tuple[float, float]]:
    anchors = [
        (0.25, 0.08),
        (-0.22, 0.12),
        (0.14, -0.24),
        (-0.30, -0.04),
        (0.08, 0.30),
        (0.38, -0.18),
    ]
    random_items = rng.normal(loc=0.0, scale=0.32, size=(max(config.max_sources_to_try, 0), 2))
    return anchors + [(float(x), float(y)) for x, y in random_items]


def _family_source_redshifts(config: SingleBCGMockConfig) -> tuple[float, ...]:
    values = tuple(float(value) for value in config.source_redshifts if np.isfinite(float(value)) and float(value) > config.z_lens)
    return values if values else (float(config.source_redshift),)


def _mock_model_and_kwargs(
    config: SingleBCGMockConfig,
    subhalo_components: list[DPIETruth],
    z_source: float,
    cosmo: Any,
) -> tuple[LensModel, list[dict[str, float]]]:
    lens_model_list = [ORIGINAL_DPIE_PROFILE_NAME, ORIGINAL_DPIE_PROFILE_NAME] + [
        ORIGINAL_DPIE_PROFILE_NAME for _ in subhalo_components
    ]
    model = LensModel(
        lens_model_list=lens_model_list,
        z_lens=config.z_lens,
        z_source=float(z_source),
        cosmo=cosmo,
    )
    kwargs_lens = [
        _component_kwargs(config.halo, config, float(z_source), cosmo),
        _component_kwargs(config.bcg, config, float(z_source), cosmo),
    ] + [_component_kwargs(component, config, float(z_source), cosmo) for component in subhalo_components]
    return model, kwargs_lens


def generate_single_bcg_mock(
    output_dir: str | Path,
    config: SingleBCGMockConfig | None = None,
    *,
    overwrite: bool = True,
) -> tuple[MockClusterPaths, pd.DataFrame, dict[str, Any]]:
    """Generate a deterministic single-BCG mock cluster on disk."""
    config = config or SingleBCGMockConfig()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    par_path = root / "single_bcg_mock.par"
    image_catalog_path = root / "obs_arcs.cat"
    member_catalog_path = root / "members.cat"
    truth_path = root / "truth.json"
    mock_images_path = root / "mock_images.json"
    if not overwrite and any(path.exists() for path in (par_path, image_catalog_path, member_catalog_path, truth_path, mock_images_path)):
        raise FileExistsError(f"Mock output already exists under {root}.")

    rng = np.random.default_rng(config.seed)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    kpc_per_arcsec = float(cosmo.kpc_proper_per_arcmin(config.z_lens).to("kpc/arcsec").value)
    subhalos = _generate_subhalo_catalog(config, rng)
    subhalo_components = [_scaled_subhalo_params(row, config) for row in subhalos]
    source_redshifts = _family_source_redshifts(config)
    model_cache: dict[float, tuple[LensModel, LensEquationSolver, list[dict[str, float]]]] = {}

    def get_model(z_source: float) -> tuple[LensModel, LensEquationSolver, list[dict[str, float]]]:
        z_key = float(z_source)
        if z_key not in model_cache:
            model, kwargs_lens = _mock_model_and_kwargs(config, subhalo_components, z_key, cosmo)
            model_cache[z_key] = (model, LensEquationSolver(model), kwargs_lens)
        return model_cache[z_key]

    image_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for beta_x, beta_y in _candidate_sources(config, rng):
        if len(source_rows) >= config.n_families:
            break
        z_source = source_redshifts[len(source_rows) % len(source_redshifts)]
        model, solver, kwargs_lens = get_model(z_source)
        x_img, y_img = solver.image_position_from_source(
            beta_x,
            beta_y,
            kwargs_lens,
            solver="lenstronomy",
            search_window=80.0,
            min_distance=0.05,
            num_iter_max=300,
            precision_limit=1.0e-8,
        )
        x_arr, y_arr = _deduplicate_images(np.asarray(x_img, dtype=float), np.asarray(y_img, dtype=float))
        if len(x_arr) < config.min_images_per_family:
            continue
        family_id = str(len(source_rows) + 1)
        source_rows.append(
            {
                "family_id": family_id,
                "beta_x": float(beta_x),
                "beta_y": float(beta_y),
                "z_source": float(z_source),
                "n_images": int(len(x_arr)),
            }
        )
        magnification = np.asarray(model.magnification(x_arr, y_arr, kwargs_lens), dtype=float)
        for image_index, (x_true, y_true, mu_true) in enumerate(zip(x_arr, y_arr, magnification), start=1):
            x_obs = float(x_true + rng.normal(0.0, config.pos_sigma_arcsec))
            y_obs = float(y_true + rng.normal(0.0, config.pos_sigma_arcsec))
            image_rows.append(
                {
                    "image_label": f"{family_id}.{image_index}",
                    "family_id": family_id,
                    "image_id": str(image_index),
                    "z_source": float(z_source),
                    "x_true_arcsec": float(x_true),
                    "y_true_arcsec": float(y_true),
                    "x_obs_arcsec": x_obs,
                    "y_obs_arcsec": y_obs,
                    "magnification_true": float(mu_true),
                }
            )

    if len(source_rows) < config.n_families:
        raise RuntimeError(
            f"Generated {len(source_rows)} multiply imaged families, fewer than requested {config.n_families}."
        )

    images = pd.DataFrame(image_rows)
    image_catalog_path.write_text(
        "#REFERENCE 3\n"
        + "\n".join(
            (
                f"{row.image_label:>8s} {row.x_obs_arcsec: .8f} {row.y_obs_arcsec: .8f} "
                f"0.3734 0.3734 90.0 {row.z_source:.8f} 25.0"
            )
            for row in images.itertuples(index=False)
        )
        + "\n",
        encoding="utf-8",
    )
    if subhalos:
        _write_member_catalog(member_catalog_path, subhalos)
    _write_single_bcg_par(par_path, config, subhalos)

    truth_payload = {
        "mock": "single-bcg-subhalos" if subhalos else "single-bcg",
        "config": _config_to_jsonable(config, kpc_per_arcsec),
        "kpc_per_arcsec": kpc_per_arcsec,
        "parameter_truth": _truth_parameter_values(config, kpc_per_arcsec),
        "sources": source_rows,
        "images": image_rows,
        "subhalos": subhalos,
        "subhalo_components": [asdict(component) for component in subhalo_components],
        "lens_model_list": [ORIGINAL_DPIE_PROFILE_NAME, ORIGINAL_DPIE_PROFILE_NAME]
        + [ORIGINAL_DPIE_PROFILE_NAME for _ in subhalo_components],
        "kwargs_lens": get_model(float(config.source_redshift))[2],
        "kwargs_lens_by_source_redshift": {f"{z:.8f}": get_model(float(z))[2] for z in source_redshifts},
    }
    truth_path.write_text(json.dumps(truth_payload, indent=2), encoding="utf-8")
    mock_images_path.write_text(json.dumps(image_rows, indent=2), encoding="utf-8")
    return MockClusterPaths(root, par_path, image_catalog_path, truth_path, mock_images_path), images, truth_payload


def parameter_recovery_table(
    samples: np.ndarray,
    parameter_names: list[str],
    truth: dict[str, float],
    *,
    best_fit: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarize posterior recovery against known truth values."""
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2:
        raise ValueError("samples must be a 2D array.")
    if sample_array.shape[1] != len(parameter_names):
        raise ValueError("parameter_names length must match sample columns.")
    best_array = None if best_fit is None else np.asarray(best_fit, dtype=float)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(parameter_names):
        values = sample_array[:, index]
        finite = values[np.isfinite(values)]
        truth_value = truth.get(name)
        if finite.size:
            q16, median, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
        else:
            q16 = median = q84 = np.nan
        if truth_value is None or not np.isfinite(float(truth_value)) or finite.size == 0:
            truth_percentile = np.nan
            bias = np.nan
            covered_68 = False
        else:
            truth_f = float(truth_value)
            truth_percentile = float(np.mean(finite <= truth_f))
            bias = float(median - truth_f)
            covered_68 = bool(q16 <= truth_f <= q84)
        rows.append(
            {
                "parameter": name,
                "truth": np.nan if truth_value is None else float(truth_value),
                "best_fit": np.nan if best_array is None else float(best_array[index]),
                "q16": float(q16),
                "median": float(median),
                "q84": float(q84),
                "bias": float(bias),
                "truth_percentile": float(truth_percentile),
                "covered_68": covered_68,
            }
        )
    return pd.DataFrame(rows)


def magnification_recovery_table(
    truth_images: pd.DataFrame,
    recovered: pd.DataFrame,
    *,
    epsilon: float = 1.0e-8,
) -> pd.DataFrame:
    merged = truth_images.merge(recovered, on="image_label", how="left", suffixes=("_truth", "_recovered"))
    mu_true = pd.to_numeric(merged["magnification_true"], errors="coerce")
    mu_rec = pd.to_numeric(merged["magnification_recovered"], errors="coerce")
    merged["magnification_bias"] = mu_rec - mu_true
    denom = np.maximum(np.abs(mu_true.to_numpy(dtype=float)), float(epsilon))
    merged["abs_magnification_fractional_error"] = (
        np.abs(np.abs(mu_rec.to_numpy(dtype=float)) - np.abs(mu_true.to_numpy(dtype=float))) / denom
    )
    merged["parity_match"] = np.sign(mu_true.to_numpy(dtype=float)) == np.sign(mu_rec.to_numpy(dtype=float))
    for suffix in ("q16", "q50", "q84"):
        column = f"magnification_{suffix}"
        if column in merged:
            values = pd.to_numeric(merged[column], errors="coerce").to_numpy(dtype=float)
            merged[f"abs_magnification_fractional_error_{suffix}"] = (
                np.abs(np.abs(values) - np.abs(mu_true.to_numpy(dtype=float))) / denom
            )
    return merged


def _load_truth(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_plot_bundle(path: str | Path) -> tuple[Any, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    from .cluster_solver import _load_artifacts

    artifacts_dir = Path(path)
    if artifacts_dir.name != "artifacts":
        artifacts_dir = artifacts_dir / "artifacts"
    return _load_artifacts(artifacts_dir)


def _artifact_parameter_names(state: Any) -> list[str]:
    return [str(spec.name) for spec in state.parameter_specs]


def _recovered_model_tables(
    state: Any,
    best_fit_physical: np.ndarray,
    images: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import jax.numpy as jnp

    from .cluster_solver import (
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
    )

    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=DEFAULT_MATCH_TOLERANCE,
        validate_top_k_families=len(state.family_data),
        sampling_engine="full",
        active_scaling_galaxies=DEFAULT_ACTIVE_SCALING_GALAXIES,
        refresh_every=DEFAULT_REFRESH_EVERY,
        refresh_param_drift_frac=DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        validation_approx="exact",
    )
    best_fit_latent = _convert_theta_to_latent(np.asarray(best_fit_physical, dtype=float), state.parameter_specs)
    magnification_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    best_eval = evaluator.evaluate(best_fit_latent, likelihood_mode="image")
    for family in state.family_data:
        model, _solver = evaluator._get_exact_model_solver(family.z_source)
        packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit_latent, dtype=jnp.float64), family.z_source)
        kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
        family_images = images[images["family_id"].astype(str) == str(family.family_id)].copy()
        mu = np.asarray(
            model.magnification(
                family_images["x_obs_arcsec"].to_numpy(dtype=float),
                family_images["y_obs_arcsec"].to_numpy(dtype=float),
                kwargs_lens,
            ),
            dtype=float,
        )
        for label, value in zip(family_images["image_label"].astype(str), mu):
            magnification_rows.append({"image_label": label, "magnification_recovered": float(value)})
        prediction = best_eval.family_predictions.get(str(family.family_id), {})
        x_pred = np.asarray(prediction.get("x_pred", np.full(family.n_images, np.nan)), dtype=float)
        y_pred = np.asarray(prediction.get("y_pred", np.full(family.n_images, np.nan)), dtype=float)
        for label, x_obs, y_obs, x_model, y_model in zip(
            family.image_labels,
            family.x_obs,
            family.y_obs,
            x_pred,
            y_pred,
        ):
            residual = math.hypot(float(x_model - x_obs), float(y_model - y_obs)) if np.isfinite(x_model + y_model) else np.nan
            image_rows.append(
                {
                    "image_label": str(label),
                    "family_id": str(family.family_id),
                    "x_obs_arcsec": float(x_obs),
                    "y_obs_arcsec": float(y_obs),
                    "x_model_arcsec": float(x_model),
                    "y_model_arcsec": float(y_model),
                    "image_residual_arcsec": float(residual),
                }
            )
        source_rows.append(
            {
                "family_id": str(family.family_id),
                "source_x_recovered": float(prediction.get("source_x", np.nan)),
                "source_y_recovered": float(prediction.get("source_y", np.nan)),
                "source_plane_rms_arcsec": float(prediction.get("source_plane_rms", np.nan)),
                "exact_image_rms_arcsec": float(prediction.get("exact_image_rms", np.nan)),
                "failed": bool(prediction.get("failed", False)),
            }
        )
    return pd.DataFrame(magnification_rows), pd.DataFrame(image_rows), pd.DataFrame(source_rows)


def _quantile_summary(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan, np.nan
    q16, q50, q84 = np.quantile(array, [0.16, 0.5, 0.84])
    return float(q16), float(q50), float(q84)


def _nanmedian_no_warning(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.median(array))


def _posterior_prediction_uncertainty_tables(
    state: Any,
    samples_physical: np.ndarray,
    images: pd.DataFrame,
    *,
    max_draws: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import jax.numpy as jnp

    from .cluster_solver import (
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
    )

    sample_array = np.asarray(samples_physical, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if sample_array.shape[0] > max_draws:
        indices = np.linspace(0, sample_array.shape[0] - 1, max_draws, dtype=int)
        sample_array = sample_array[indices]

    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=DEFAULT_MATCH_TOLERANCE,
        validate_top_k_families=len(state.family_data),
        sampling_engine="full",
        active_scaling_galaxies=DEFAULT_ACTIVE_SCALING_GALAXIES,
        refresh_every=DEFAULT_REFRESH_EVERY,
        refresh_param_drift_frac=DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        validation_approx="exact",
    )
    mag_by_label: dict[str, list[float]] = {}
    x_by_label: dict[str, list[float]] = {}
    y_by_label: dict[str, list[float]] = {}
    residual_by_label: dict[str, list[float]] = {}
    source_x_by_family: dict[str, list[float]] = {}
    source_y_by_family: dict[str, list[float]] = {}
    source_rms_by_family: dict[str, list[float]] = {}
    exact_rms_by_family: dict[str, list[float]] = {}
    exact_failed_families: set[str] = set()

    for sample in sample_array:
        sample_latent = _convert_theta_to_latent(sample, state.parameter_specs)
        family_predictions = evaluator._family_source_summary(sample_latent)
        for family in state.family_data:
            family_id = str(family.family_id)
            model, _solver = evaluator._get_exact_model_solver(family.z_source)
            packed_state = evaluator._build_packed_lens_state(jnp.asarray(sample_latent, dtype=jnp.float64), family.z_source)
            kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
            family_images = images[images["family_id"].astype(str) == family_id].copy()
            mu = np.asarray(
                model.magnification(
                    family_images["x_obs_arcsec"].to_numpy(dtype=float),
                    family_images["y_obs_arcsec"].to_numpy(dtype=float),
                    kwargs_lens,
                ),
                dtype=float,
            )
            for label, value in zip(family_images["image_label"].astype(str), mu):
                mag_by_label.setdefault(label, []).append(float(value))

            prediction = family_predictions.get(family_id, {})
            if family_id not in exact_failed_families:
                exact_prediction = evaluator._exact_family_prediction(sample_latent, family)
                if exact_prediction is None:
                    exact_failed_families.add(family_id)
                else:
                    x_pred_exact, y_pred_exact, exact_rms = exact_prediction
                    prediction["x_pred"] = x_pred_exact
                    prediction["y_pred"] = y_pred_exact
                    prediction["exact_image_rms"] = exact_rms
            x_pred = np.asarray(prediction.get("x_pred", np.full(family.n_images, np.nan)), dtype=float)
            y_pred = np.asarray(prediction.get("y_pred", np.full(family.n_images, np.nan)), dtype=float)
            for label, x_obs, y_obs, x_model, y_model in zip(
                family.image_labels,
                family.x_obs,
                family.y_obs,
                x_pred,
                y_pred,
            ):
                label = str(label)
                x_by_label.setdefault(label, []).append(float(x_model))
                y_by_label.setdefault(label, []).append(float(y_model))
                residual = math.hypot(float(x_model - x_obs), float(y_model - y_obs)) if np.isfinite(x_model + y_model) else np.nan
                residual_by_label.setdefault(label, []).append(float(residual))

            source_x_by_family.setdefault(family_id, []).append(float(prediction.get("source_x", np.nan)))
            source_y_by_family.setdefault(family_id, []).append(float(prediction.get("source_y", np.nan)))
            source_rms_by_family.setdefault(family_id, []).append(float(prediction.get("source_plane_rms", np.nan)))
            exact_rms_by_family.setdefault(family_id, []).append(float(prediction.get("exact_image_rms", np.nan)))

    mag_rows: list[dict[str, Any]] = []
    for label, values in mag_by_label.items():
        q16, q50, q84 = _quantile_summary(values)
        mag_rows.append(
            {
                "image_label": label,
                "magnification_q16": q16,
                "magnification_q50": q50,
                "magnification_q84": q84,
            }
        )

    image_rows: list[dict[str, Any]] = []
    for label in sorted(set(x_by_label) | set(y_by_label) | set(residual_by_label)):
        x16, x50, x84 = _quantile_summary(x_by_label.get(label, []))
        y16, y50, y84 = _quantile_summary(y_by_label.get(label, []))
        r16, r50, r84 = _quantile_summary(residual_by_label.get(label, []))
        image_rows.append(
            {
                "image_label": label,
                "x_model_q16": x16,
                "x_model_q50": x50,
                "x_model_q84": x84,
                "y_model_q16": y16,
                "y_model_q50": y50,
                "y_model_q84": y84,
                "image_residual_q16": r16,
                "image_residual_q50": r50,
                "image_residual_q84": r84,
            }
        )

    source_rows: list[dict[str, Any]] = []
    for family_id in sorted(set(source_x_by_family) | set(source_y_by_family)):
        sx16, sx50, sx84 = _quantile_summary(source_x_by_family.get(family_id, []))
        sy16, sy50, sy84 = _quantile_summary(source_y_by_family.get(family_id, []))
        sr16, sr50, sr84 = _quantile_summary(source_rms_by_family.get(family_id, []))
        er16, er50, er84 = _quantile_summary(exact_rms_by_family.get(family_id, []))
        source_rows.append(
            {
                "family_id": family_id,
                "source_x_q16": sx16,
                "source_x_q50": sx50,
                "source_x_q84": sx84,
                "source_y_q16": sy16,
                "source_y_q50": sy50,
                "source_y_q84": sy84,
                "source_plane_rms_q16": sr16,
                "source_plane_rms_q50": sr50,
                "source_plane_rms_q84": sr84,
                "exact_image_rms_q16": er16,
                "exact_image_rms_q50": er50,
                "exact_image_rms_q84": er84,
            }
        )
    return pd.DataFrame(mag_rows), pd.DataFrame(image_rows), pd.DataFrame(source_rows)


def _magnifications_for_images(state: Any, best_fit_physical: np.ndarray, images: pd.DataFrame) -> pd.DataFrame:
    magnification, _image, _source = _recovered_model_tables(state, best_fit_physical, images)
    return magnification


def _deflection_profile_for_samples(
    state: Any,
    samples: np.ndarray,
    truth: dict[str, Any],
    radii_arcsec: np.ndarray,
) -> pd.DataFrame:
    import jax.numpy as jnp

    from .cluster_solver import (
        DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_ACTIVE_SCALING_MIN,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
    )

    config = truth["config"]
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    z_lens = float(config["z_lens"])
    z_source = float(config["source_redshift"])
    model = LensModel(
        lens_model_list=list(state.lens_model_list),
        z_lens=z_lens,
        z_source=z_source,
        cosmo=cosmo,
    )
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=DEFAULT_MATCH_TOLERANCE,
        validate_top_k_families=0,
        sampling_engine="full",
        active_scaling_galaxies=DEFAULT_ACTIVE_SCALING_GALAXIES,
        active_scaling_selection="adaptive",
        active_scaling_cumulative_fraction=DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        active_scaling_min=DEFAULT_ACTIVE_SCALING_MIN,
        refresh_every=DEFAULT_REFRESH_EVERY,
        refresh_param_drift_frac=DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        validation_approx="exact",
    )
    component_family = np.asarray(state.packed_lens_spec.component_family, dtype=int)
    n_components = len(state.lens_model_list)
    group_indices: dict[str, list[int]] = {
        "total": list(range(n_components)),
        "halo": [0] if n_components > 0 else [],
        "bcg": [1] if n_components > 1 else [],
        "subhalos": np.where(component_family == 1)[0].astype(int).tolist(),
    }
    group_indices["bcg_plus_subhalos"] = group_indices["bcg"] + group_indices["subhalos"]
    display_names = {
        "total": "total",
        "halo": "halo",
        "bcg": "BCG",
        "subhalos": "subhalos",
        "bcg_plus_subhalos": "BCG + subhalos",
    }
    truth_kwargs_by_z = truth.get("kwargs_lens_by_source_redshift", {})
    truth_kwargs = truth_kwargs_by_z.get(f"{z_source:.8f}", truth.get("kwargs_lens", []))

    def alpha_magnitude(kwargs_lens: list[dict[str, float]], radius: float, indices: list[int]) -> float:
        if not indices:
            return 0.0
        alpha_x, alpha_y = model.alpha(np.asarray([radius]), np.asarray([0.0]), kwargs_lens, k=indices)
        return float(np.hypot(float(alpha_x[0]), float(alpha_y[0])))

    rows: list[dict[str, Any]] = []
    sample_values_by_group_radius: dict[tuple[str, float], list[float]] = {
        (group, float(radius)): [] for group in group_indices for radius in radii_arcsec
    }
    for sample in np.asarray(samples, dtype=float):
        sample_latent = _convert_theta_to_latent(sample, state.parameter_specs)
        packed_state = evaluator._build_packed_lens_state(jnp.asarray(sample_latent, dtype=jnp.float64), z_source)
        kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
        for radius in radii_arcsec:
            radius_f = float(radius)
            for group, indices in group_indices.items():
                sample_values_by_group_radius[(group, radius_f)].append(alpha_magnitude(kwargs_lens, radius_f, indices))

    for group, indices in group_indices.items():
        if group in {"bcg", "subhalos"} and not indices:
            continue
        for radius in radii_arcsec:
            radius_f = float(radius)
            finite = np.asarray(sample_values_by_group_radius[(group, radius_f)], dtype=float)
            finite = finite[np.isfinite(finite)]
            q16, median, q84 = np.quantile(finite, [0.16, 0.5, 0.84]) if finite.size else (np.nan, np.nan, np.nan)
            truth_value = alpha_magnitude(truth_kwargs, radius_f, indices)
            rows.append(
                {
                    "radius_arcsec": radius_f,
                    "component": group,
                    "component_label": display_names[group],
                    "quantity": f"{group}_deflection_magnitude_arcsec",
                    "truth": truth_value,
                    "q16": float(q16),
                    "median": float(median),
                    "q84": float(q84),
                    "bias": float(median - truth_value),
                }
            )
    return pd.DataFrame(rows)


def write_recovery_outputs(
    run_dir: str | Path,
    truth_path: str | Path,
    mock_images_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    posterior_diagnostic_draws: int = 8,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = _load_truth(truth_path)
    images = pd.DataFrame(json.loads(Path(mock_images_path).read_text(encoding="utf-8")))
    state, _saved_args, arrays, _init_diagnostics = _load_plot_bundle(run_dir)
    samples = np.asarray(arrays["samples"], dtype=float)
    best_fit = np.asarray(arrays["best_fit"], dtype=float)
    parameter_names = _artifact_parameter_names(state)
    parameter_df = parameter_recovery_table(
        samples,
        parameter_names,
        {str(key): float(value) for key, value in truth["parameter_truth"].items()},
        best_fit=best_fit,
    )
    recovered_mu, image_df, source_df = _recovered_model_tables(state, best_fit, images)
    mag_uncertainty_df, image_uncertainty_df, source_uncertainty_df = _posterior_prediction_uncertainty_tables(
        state,
        samples,
        images,
        max_draws=int(posterior_diagnostic_draws),
    )
    if not mag_uncertainty_df.empty:
        recovered_mu = recovered_mu.merge(mag_uncertainty_df, on="image_label", how="left")
    if not image_uncertainty_df.empty:
        image_df = image_df.merge(image_uncertainty_df, on="image_label", how="left")
    if not source_uncertainty_df.empty:
        source_df = source_df.merge(source_uncertainty_df, on="family_id", how="left")
    magnification_df = magnification_recovery_table(images, recovered_mu)
    source_truth_df = pd.DataFrame(truth.get("sources", []))
    if not source_truth_df.empty:
        source_df = source_truth_df.merge(source_df, on="family_id", how="left")
        source_df["source_position_error_arcsec"] = np.hypot(
            source_df["source_x_recovered"].to_numpy(dtype=float) - source_df["beta_x"].to_numpy(dtype=float),
            source_df["source_y_recovered"].to_numpy(dtype=float) - source_df["beta_y"].to_numpy(dtype=float),
        )
        if {"source_x_q16", "source_x_q50", "source_x_q84", "source_y_q16", "source_y_q50", "source_y_q84"}.issubset(source_df.columns):
            for suffix in ("q16", "q50", "q84"):
                source_df[f"source_position_error_{suffix}"] = np.hypot(
                    source_df[f"source_x_{suffix}"].to_numpy(dtype=float) - source_df["beta_x"].to_numpy(dtype=float),
                    source_df[f"source_y_{suffix}"].to_numpy(dtype=float) - source_df["beta_y"].to_numpy(dtype=float),
                )
    mass_profile_df = _deflection_profile_for_samples(
        state,
        samples,
        truth,
        radii_arcsec=np.asarray([2.0, 5.0, 10.0, 20.0, 40.0], dtype=float),
    )
    summary = {
        "n_parameters": len(parameter_df),
        "median_abs_parameter_bias": float(np.nanmedian(np.abs(parameter_df["bias"]))),
        "parameter_coverage_68_fraction": float(np.mean(parameter_df["covered_68"])),
        "n_images": len(magnification_df),
        "median_image_residual_arcsec": _nanmedian_no_warning(image_df["image_residual_arcsec"]),
        "median_source_position_error_arcsec": _nanmedian_no_warning(source_df["source_position_error_arcsec"])
        if "source_position_error_arcsec" in source_df
        else np.nan,
        "median_abs_magnification_frac_error": _nanmedian_no_warning(
            magnification_df["abs_magnification_fractional_error"]
        ),
        "parity_match_fraction": float(np.nanmean(magnification_df["parity_match"].astype(float))),
    }
    summary_uncertainty = _summary_uncertainty(parameter_df, image_df, source_df, magnification_df)
    paths = {
        "corner_plot": output_dir / "corner.pdf",
        "potfile_corner_plot": output_dir / "potfile_corner.pdf",
        "parameter_pull_plot": output_dir / "parameter_recovery.pdf",
        "mass_profile_plot": output_dir / "mass_profile_recovery.pdf",
        "magnification_plot": output_dir / "magnification_recovery.pdf",
        "image_recovery_plot": output_dir / "image_recovery.pdf",
        "source_recovery_plot": output_dir / "source_recovery.pdf",
        "subhalo_population_plot": output_dir / "subhalo_population.pdf",
        "summary_plot": output_dir / "validation_summary.pdf",
    }
    truth_values = {str(key): float(value) for key, value in truth["parameter_truth"].items()}
    _plot_corner_pdf(output_dir, samples, state.parameter_specs, "corner.pdf", truth_values=truth_values)
    scaling_specs, scaling_samples, _scaling_best_fit = _scaling_parameter_subset(
        state.parameter_specs,
        samples,
        best_fit,
    )
    _plot_corner_pdf(output_dir, scaling_samples, scaling_specs, "potfile_corner.pdf", truth_values=truth_values)
    _plot_parameter_recovery(parameter_df, paths["parameter_pull_plot"])
    _plot_mass_profile_recovery(mass_profile_df, paths["mass_profile_plot"])
    _plot_magnification_recovery(magnification_df, paths["magnification_plot"])
    _plot_image_recovery(image_df, paths["image_recovery_plot"])
    _plot_source_recovery(source_df, paths["source_recovery_plot"])
    _plot_subhalo_population(pd.DataFrame(truth.get("subhalos", [])), images, parameter_df, paths["subhalo_population_plot"])
    _plot_validation_summary(summary, summary_uncertainty, paths["summary_plot"])
    return paths


def _plot_parameter_recovery(parameter_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(parameter_df))))
    y = np.arange(len(parameter_df))
    median = parameter_df["median"].to_numpy(dtype=float)
    q16 = parameter_df["q16"].to_numpy(dtype=float)
    q84 = parameter_df["q84"].to_numpy(dtype=float)
    truth = parameter_df["truth"].to_numpy(dtype=float)
    ax.errorbar(median, y, xerr=[median - q16, q84 - median], fmt="o", color="tab:blue", label="posterior 1 sigma")
    ax.scatter(truth, y, marker="x", color="black", label="truth")
    ax.set_yticks(y, parameter_df["parameter"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("parameter value")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_corner_placeholder(samples: np.ndarray, parameter_names: list[str], path: Path, plot_name: str) -> None:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.size == 0:
        n_samples = 0
        n_params = len(parameter_names)
        n_dynamic = 0
    else:
        finite_rows = sample_array[np.all(np.isfinite(sample_array), axis=1)]
        n_samples = int(finite_rows.shape[0])
        n_params = int(finite_rows.shape[1]) if finite_rows.ndim == 2 else len(parameter_names)
        if finite_rows.ndim == 2 and finite_rows.shape[0] > 0:
            spans = np.nanmax(finite_rows, axis=0) - np.nanmin(finite_rows, axis=0)
            n_dynamic = int(np.sum(np.isfinite(spans) & (spans > 0.0)))
        else:
            n_dynamic = 0
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(
        0.5,
        0.62,
        f"{plot_name} was not generated",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.42,
        (
            "The saved posterior has fewer than two parameters with dynamic range.\n"
            f"finite samples: {n_samples}, parameters: {n_params}, dynamic parameters: {n_dynamic}.\n"
            "This usually means the sampler/guide posterior collapsed or all retained samples are identical."
        ),
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_corner_pdf(
    output_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[Any],
    filename: str = "corner.pdf",
    truth_values: dict[str, float] | None = None,
) -> None:
    path = output_dir / filename
    if path.exists():
        path.unlink()
    try:
        if filename == "corner.pdf":
            _plot_corner(output_dir, samples, parameter_specs, truth_values=truth_values)
        else:
            _plot_potfile_corner(output_dir, samples, parameter_specs, truth_values=truth_values)
    except Exception as exc:  # pragma: no cover - defensive plotting fallback
        _write_corner_placeholder(samples, [getattr(spec, "name", str(spec)) for spec in parameter_specs], path, filename)
        _log_message = f"[validation:corner] wrote placeholder {path}: {exc}"
        print(_log_message)
        return
    if not path.exists():
        _write_corner_placeholder(samples, [getattr(spec, "name", str(spec)) for spec in parameter_specs], path, filename)


def _summary_uncertainty(
    parameter_df: pd.DataFrame,
    image_df: pd.DataFrame,
    source_df: pd.DataFrame,
    magnification_df: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    def interval_from_columns(df: pd.DataFrame, q16_col: str, q84_col: str) -> tuple[float, float]:
        if q16_col not in df or q84_col not in df:
            return np.nan, np.nan
        low = _nanmedian_no_warning(df[q16_col])
        high = _nanmedian_no_warning(df[q84_col])
        return (min(low, high), max(low, high)) if np.isfinite(low + high) else (low, high)

    image_interval = interval_from_columns(image_df, "image_residual_q16", "image_residual_q84")
    source_interval = interval_from_columns(source_df, "source_position_error_q16", "source_position_error_q84")
    mag_interval = interval_from_columns(
        magnification_df,
        "abs_magnification_fractional_error_q16",
        "abs_magnification_fractional_error_q84",
    )
    coverage_values = parameter_df["covered_68"].astype(float).to_numpy(dtype=float)
    coverage_se = (
        float(np.sqrt(np.nanmean(coverage_values) * (1.0 - np.nanmean(coverage_values)) / max(np.sum(np.isfinite(coverage_values)), 1)))
        if coverage_values.size
        else np.nan
    )
    coverage_mean = float(np.nanmean(coverage_values)) if coverage_values.size else np.nan
    parity_values = magnification_df["parity_match"].astype(float).to_numpy(dtype=float)
    parity_se = (
        float(np.sqrt(np.nanmean(parity_values) * (1.0 - np.nanmean(parity_values)) / max(np.sum(np.isfinite(parity_values)), 1)))
        if parity_values.size
        else np.nan
    )
    parity_mean = float(np.nanmean(parity_values)) if parity_values.size else np.nan
    return {
        "median_image_residual_arcsec": image_interval,
        "median_source_position_error_arcsec": source_interval,
        "median_abs_magnification_frac_error": mag_interval,
        "parameter_coverage_68_fraction": (np.nan, np.nan) if not np.isfinite(coverage_se) else (coverage_mean - coverage_se, coverage_mean + coverage_se),
        "parity_match_fraction": (np.nan, np.nan) if not np.isfinite(parity_se) else (parity_mean - parity_se, parity_mean + parity_se),
    }


def _plot_mass_profile_recovery(profile_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25], "hspace": 0.08},
    )
    ax = axes[0]
    ratio_ax = axes[1]
    component_order = ["total", "halo", "bcg_plus_subhalos", "bcg", "subhalos"]
    colors = {
        "total": "tab:blue",
        "halo": "tab:orange",
        "bcg_plus_subhalos": "tab:green",
        "bcg": "tab:red",
        "subhalos": "tab:purple",
    }
    narrow_messages: list[str] = []
    for component in component_order:
        comp_df = profile_df[profile_df["component"] == component].sort_values("radius_arcsec")
        if comp_df.empty:
            continue
        label = str(comp_df["component_label"].iloc[0])
        color = colors.get(component, "0.4")
        radius = comp_df["radius_arcsec"].to_numpy(dtype=float)
        median = comp_df["median"].to_numpy(dtype=float)
        q16 = comp_df["q16"].to_numpy(dtype=float)
        q84 = comp_df["q84"].to_numpy(dtype=float)
        low = np.minimum(q16, q84)
        high = np.maximum(q16, q84)
        truth = comp_df["truth"].to_numpy(dtype=float)
        yerr = [np.maximum(0.0, median - low), np.maximum(0.0, high - median)]
        band_width = high - low
        finite_scale = np.nanmax(np.abs(median[np.isfinite(median)])) if np.isfinite(median).any() else np.nan
        narrow_band = bool(
            np.isfinite(finite_scale)
            and finite_scale > 0.0
            and np.isfinite(band_width).any()
            and np.nanmax(band_width[np.isfinite(band_width)]) < 0.003 * finite_scale
        )
        if narrow_band:
            narrow_messages.append(label)
        line_width = 2.0 if component == "total" else 1.4
        alpha = 0.22 if component == "total" else 0.14
        ax.fill_between(radius, low, high, color=color, alpha=alpha)
        ax.errorbar(
            radius,
            median,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=4 if component == "total" else 3,
            linewidth=line_width,
            label=f"{label} posterior",
        )
        ax.plot(radius, median, color=color, linewidth=line_width)
        ax.plot(radius, truth, color=color, linestyle="--", linewidth=line_width, label=f"{label} truth")

        denom = np.maximum(np.abs(truth), 1.0e-12)
        ratio_median = (median - truth) / denom
        ratio_low = (low - truth) / denom
        ratio_high = (high - truth) / denom
        ratio_ax.fill_between(radius, ratio_low, ratio_high, color=color, alpha=alpha)
        ratio_ax.errorbar(
            radius,
            ratio_median,
            yerr=[
                np.maximum(0.0, ratio_median - np.minimum(ratio_low, ratio_high)),
                np.maximum(0.0, np.maximum(ratio_low, ratio_high) - ratio_median),
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=4 if component == "total" else 3,
        )
        ratio_ax.plot(radius, ratio_median, color=color, linewidth=line_width)

    ax.set_ylabel("deflection magnitude [arcsec]")
    ax.legend(loc="best", fontsize=8)
    if narrow_messages:
        ax.text(
            0.98,
            0.04,
            "narrow 1 sigma bands: " + ", ".join(narrow_messages),
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=8,
            color="0.35",
        )
    ratio_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ratio_ax.set_xlabel("radius [arcsec]")
    ratio_ax.set_ylabel("(post. - truth) / truth")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_magnification_recovery(magnification_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    truth = magnification_df["magnification_true"].to_numpy(dtype=float)
    recovered = magnification_df["magnification_recovered"].to_numpy(dtype=float)
    plotted = recovered
    if {"magnification_q16", "magnification_q50", "magnification_q84"}.issubset(magnification_df.columns):
        q16 = magnification_df["magnification_q16"].to_numpy(dtype=float)
        q50 = magnification_df["magnification_q50"].to_numpy(dtype=float)
        q84 = magnification_df["magnification_q84"].to_numpy(dtype=float)
        plotted = q50
        ax.errorbar(
            truth,
            q50,
            yerr=[np.maximum(0.0, q50 - q16), np.maximum(0.0, q84 - q50)],
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            alpha=0.8,
            label="posterior 1 sigma",
        )
    else:
        ax.scatter(truth, recovered, color="tab:blue", label="best fit")
    finite = np.concatenate([truth[np.isfinite(truth)], plotted[np.isfinite(plotted)]])
    if finite.size:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_xlabel("true signed magnification")
    ax.set_ylabel("recovered signed magnification")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_image_recovery(image_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.scatter(image_df["x_obs_arcsec"], image_df["y_obs_arcsec"], color="black", s=22, label="observed")
    if {"x_model_q16", "x_model_q50", "x_model_q84", "y_model_q16", "y_model_q50", "y_model_q84"}.issubset(image_df.columns):
        x_model = image_df["x_model_q50"].to_numpy(dtype=float)
        y_model = image_df["y_model_q50"].to_numpy(dtype=float)
        ax.errorbar(
            x_model,
            y_model,
            xerr=[
                np.maximum(0.0, x_model - image_df["x_model_q16"].to_numpy(dtype=float)),
                np.maximum(0.0, image_df["x_model_q84"].to_numpy(dtype=float) - x_model),
            ],
            yerr=[
                np.maximum(0.0, y_model - image_df["y_model_q16"].to_numpy(dtype=float)),
                np.maximum(0.0, image_df["y_model_q84"].to_numpy(dtype=float) - y_model),
            ],
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            markersize=4,
            label="model 1 sigma",
        )
    else:
        x_model = image_df["x_model_arcsec"].to_numpy(dtype=float)
        y_model = image_df["y_model_arcsec"].to_numpy(dtype=float)
        ax.scatter(image_df["x_model_arcsec"], image_df["y_model_arcsec"], color="tab:blue", s=18, label="model")
    for row, x_fit, y_fit in zip(image_df.itertuples(index=False), x_model, y_model):
        if np.isfinite(x_fit) and np.isfinite(y_fit):
            ax.plot([row.x_obs_arcsec, x_fit], [row.y_obs_arcsec, y_fit], color="0.6", lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_title("Image positions")
    ax.legend(loc="best", fontsize=8)

    residual = (
        image_df["image_residual_q50"].to_numpy(dtype=float)
        if "image_residual_q50" in image_df
        else image_df["image_residual_arcsec"].to_numpy(dtype=float)
    )
    x_index = np.arange(len(image_df))
    if {"image_residual_q16", "image_residual_q84"}.issubset(image_df.columns):
        axes[1].errorbar(
            x_index,
            residual,
            yerr=[
                np.maximum(0.0, residual - image_df["image_residual_q16"].to_numpy(dtype=float)),
                np.maximum(0.0, image_df["image_residual_q84"].to_numpy(dtype=float) - residual),
            ],
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
        )
    else:
        axes[1].scatter(x_index, residual, color="tab:blue")
    axes[1].set_xlabel("image index")
    axes[1].set_ylabel("image residual [arcsec]")
    axes[1].set_title("Image residuals with 1 sigma intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_source_recovery(source_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    if {"beta_x", "beta_y"}.issubset(source_df.columns):
        ax.scatter(source_df["beta_x"], source_df["beta_y"], color="black", s=28, label="truth")
    if {"source_x_q16", "source_x_q50", "source_x_q84", "source_y_q16", "source_y_q50", "source_y_q84"}.issubset(source_df.columns):
        sx = source_df["source_x_q50"].to_numpy(dtype=float)
        sy = source_df["source_y_q50"].to_numpy(dtype=float)
        ax.errorbar(
            sx,
            sy,
            xerr=[
                np.maximum(0.0, sx - source_df["source_x_q16"].to_numpy(dtype=float)),
                np.maximum(0.0, source_df["source_x_q84"].to_numpy(dtype=float) - sx),
            ],
            yerr=[
                np.maximum(0.0, sy - source_df["source_y_q16"].to_numpy(dtype=float)),
                np.maximum(0.0, source_df["source_y_q84"].to_numpy(dtype=float) - sy),
            ],
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            markersize=5,
            label="recovered 1 sigma",
        )
    else:
        sx = source_df["source_x_recovered"].to_numpy(dtype=float)
        sy = source_df["source_y_recovered"].to_numpy(dtype=float)
        ax.scatter(source_df["source_x_recovered"], source_df["source_y_recovered"], color="tab:blue", s=24, label="recovered")
    if {"beta_x", "beta_y"}.issubset(source_df.columns):
        for row, sx_fit, sy_fit in zip(source_df.itertuples(index=False), sx, sy):
            if np.isfinite(sx_fit) and np.isfinite(sy_fit):
                ax.plot([row.beta_x, sx_fit], [row.beta_y, sy_fit], color="0.6", lw=0.8)
    ax.set_xlabel(r"$\beta_x$ [arcsec]")
    ax.set_ylabel(r"$\beta_y$ [arcsec]")
    ax.set_title("Source positions")
    ax.legend(loc="best", fontsize=8)

    if "source_position_error_arcsec" in source_df:
        values = (
            source_df["source_position_error_q50"].to_numpy(dtype=float)
            if "source_position_error_q50" in source_df
            else source_df["source_position_error_arcsec"].to_numpy(dtype=float)
        )
        x_index = np.arange(len(source_df))
        if {"source_position_error_q16", "source_position_error_q84"}.issubset(source_df.columns):
            axes[1].errorbar(
                x_index,
                values,
                yerr=[
                    np.maximum(0.0, values - source_df["source_position_error_q16"].to_numpy(dtype=float)),
                    np.maximum(0.0, source_df["source_position_error_q84"].to_numpy(dtype=float) - values),
                ],
                fmt="o",
                color="tab:blue",
                ecolor="tab:blue",
            )
        else:
            axes[1].scatter(x_index, values, color="tab:blue")
        axes[1].set_xlabel("family index")
        axes[1].set_ylabel("source position error [arcsec]")
    else:
        values = (
            source_df["source_plane_rms_q50"].to_numpy(dtype=float)
            if "source_plane_rms_q50" in source_df
            else source_df["source_plane_rms_arcsec"].to_numpy(dtype=float)
        )
        x_index = np.arange(len(source_df))
        if {"source_plane_rms_q16", "source_plane_rms_q84"}.issubset(source_df.columns):
            axes[1].errorbar(
                x_index,
                values,
                yerr=[
                    np.maximum(0.0, values - source_df["source_plane_rms_q16"].to_numpy(dtype=float)),
                    np.maximum(0.0, source_df["source_plane_rms_q84"].to_numpy(dtype=float) - values),
                ],
                fmt="o",
                color="tab:blue",
                ecolor="tab:blue",
            )
        else:
            axes[1].scatter(x_index, values, color="tab:blue")
        axes[1].set_xlabel("family index")
        axes[1].set_ylabel("source-plane RMS [arcsec]")
    axes[1].set_title("Source recovery with 1 sigma intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_subhalo_population(
    subhalo_df: pd.DataFrame,
    images: pd.DataFrame,
    parameter_df: pd.DataFrame,
    path: Path,
) -> None:
    del parameter_df
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(images["x_obs_arcsec"], images["y_obs_arcsec"], color="black", marker="x", s=26, label="images")
    if not subhalo_df.empty:
        sizes = 12.0 + 80.0 * np.sqrt(subhalo_df["luminosity_ratio"].to_numpy(dtype=float))
        scatter = ax.scatter(
            subhalo_df["x_arcsec"],
            subhalo_df["y_arcsec"],
            s=sizes,
            c=subhalo_df["catalog_mag"],
            cmap="viridis_r",
            alpha=0.75,
            label="subhalos",
        )
        fig.colorbar(scatter, ax=ax, label="member magnitude")
    ax.scatter([0.0], [0.0], color="tab:red", marker="+", s=80, label="BCG")
    ax.invert_xaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_title("Subhalo field")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_validation_summary(summary: dict[str, float], uncertainty: dict[str, tuple[float, float]], path: Path) -> None:
    labels = [
        "median image residual",
        "median source error",
        "median |mu| frac. error",
        "parameter 1 sigma coverage",
        "parity match fraction",
    ]
    values = [
        summary["median_image_residual_arcsec"],
        summary["median_source_position_error_arcsec"],
        summary["median_abs_magnification_frac_error"],
        summary["parameter_coverage_68_fraction"],
        summary["parity_match_fraction"],
    ]
    keys = [
        "median_image_residual_arcsec",
        "median_source_position_error_arcsec",
        "median_abs_magnification_frac_error",
        "parameter_coverage_68_fraction",
        "parity_match_fraction",
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(labels))
    ax.barh(y, values, color=["tab:blue", "tab:cyan", "tab:purple", "tab:green", "tab:orange"], alpha=0.85)
    for idx, (key, value) in enumerate(zip(keys, values)):
        low, high = uncertainty.get(key, (np.nan, np.nan))
        if np.isfinite(value) and np.isfinite(low) and np.isfinite(high):
            ax.errorbar(
                value,
                idx,
                xerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
            )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("metric value")
    ax.set_title("Mock recovery summary with 1 sigma intervals")
    for idx, value in enumerate(values):
        if np.isfinite(value):
            ax.text(value, idx, f" {value:.3g}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_cluster_solver(par_path: Path, output_dir: Path, run_name: str, args: argparse.Namespace) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "lenscluster.cluster_solver",
        "--par-path",
        str(par_path),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--fit-method",
        str(args.fit_method),
        "--svi-steps",
        str(args.svi_steps),
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--chains",
        str(args.chains),
        "--sampling-engine",
        str(args.sampling_engine),
        "--source-plane-covariance-floor",
        str(args.source_plane_covariance_floor),
        "--active-scaling-selection",
        str(args.active_scaling_selection),
        "--active-scaling-cumulative-fraction",
        str(args.active_scaling_cumulative_fraction),
        "--active-scaling-min",
        str(args.active_scaling_min),
        "--likelihood-mode",
        "source",
        "--validation-approx",
        "exact",
        "--validate-top-k-families",
        "999",
        "--pos-sigma-arcsec",
        str(args.pos_sigma_arcsec),
        "--seed",
        str(args.seed),
        "--target-accept",
        str(args.target_accept),
        "--max-tree-depth",
        str(args.max_tree_depth),
    ]
    if args.active_scaling_galaxies is not None:
        cmd.append("--active-scaling-galaxies")
        cmd.extend(str(value) for value in args.active_scaling_galaxies)
    if args.fit_scaling_scatter and int(args.n_subhalos) > 0:
        scatter_fields: list[str] = []
        if float(args.subhalo_sigma_scatter_dex) > 0.0:
            scatter_fields.append("sigma")
        if float(args.subhalo_cut_scatter_dex) > 0.0:
            scatter_fields.append("cut")
        if scatter_fields:
            scatter_max = max(
                float(args.scaling_scatter_max),
                1.25 * _dex_scatter_to_ln(float(args.subhalo_sigma_scatter_dex)),
                1.25 * _dex_scatter_to_ln(float(args.subhalo_cut_scatter_dex)),
            )
            cmd.extend(
                [
                    "--scaling-scatter",
                    "--scaling-scatter-fields",
                    ",".join(scatter_fields),
                    "--scaling-scatter-max",
                    f"{scatter_max:.8g}",
                ]
            )
    if args.skip_plots:
        cmd.append("--skip-plots")
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], check=True)
    return output_dir / run_name / "stage2_joint"


def run_single_bcg_validation(args: argparse.Namespace) -> list[dict[str, Path]]:
    root = Path(args.output_dir) / "single_bcg" / str(args.run_name)
    outputs: list[dict[str, Path]] = []
    source_redshifts = _parse_source_redshifts(args.source_redshifts, fallback=float(args.source_redshift))
    for realization in range(int(args.realizations)):
        seed = int(args.seed) + realization
        realization_dir = root / f"seed_{seed}"
        config = SingleBCGMockConfig(
            seed=seed,
            pos_sigma_arcsec=float(args.pos_sigma_arcsec),
            n_families=int(args.n_families),
            source_redshift=float(args.source_redshift),
            source_redshifts=source_redshifts,
            source_sigma_int_arcsec=float(args.source_sigma_int_arcsec),
            n_subhalos=int(args.n_subhalos),
            subhalo_sigma_scatter_dex=float(args.subhalo_sigma_scatter_dex),
            subhalo_cut_scatter_dex=float(args.subhalo_cut_scatter_dex),
        )
        paths, _images, _truth = generate_single_bcg_mock(realization_dir / "mock", config)
        solver_run_name = "fit"
        solver_run_dir = _run_cluster_solver(paths.par_path, realization_dir / "solver", solver_run_name, args)
        output_paths = write_recovery_outputs(
            solver_run_dir,
            paths.truth_path,
            paths.mock_images_path,
            output_dir=realization_dir,
            posterior_diagnostic_draws=int(args.posterior_diagnostic_draws),
        )
        outputs.append(output_paths)
    return outputs


def _parse_source_redshifts(raw: str | None, *, fallback: float) -> tuple[float, ...]:
    if raw is None or not str(raw).strip():
        return (float(fallback),)
    values = tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values:
        return (float(fallback),)
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mock-recovery validation suite for lenscluster.")
    parser.add_argument("--mock", choices=("single-bcg",), default="single-bcg")
    parser.add_argument("--output-dir", default="validation_runs")
    parser.add_argument("--run-name", default="single_bcg_recovery")
    parser.add_argument("--realizations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-families", type=int, default=3)
    parser.add_argument("--n-subhalos", type=int, default=0)
    parser.add_argument(
        "--subhalo-sigma-scatter-dex",
        type=float,
        default=0.07,
        help="Injected log10 scatter in the subhalo velocity-dispersion scaling relation.",
    )
    parser.add_argument(
        "--subhalo-cut-scatter-dex",
        type=float,
        default=0.20,
        help="Injected log10 scatter in the subhalo cut-radius scaling relation.",
    )
    parser.add_argument("--source-redshift", type=float, default=2.0)
    parser.add_argument(
        "--source-redshifts",
        default="1.5,2.0,3.0",
        help="Comma-separated source redshifts cycled across mock families. Empty string falls back to --source-redshift.",
    )
    parser.add_argument("--source-sigma-int-arcsec", type=float, default=0.05)
    parser.add_argument("--pos-sigma-arcsec", type=float, default=0.15)
    parser.add_argument("--fit-method", choices=("svi", "svi+nuts"), default="svi+nuts")
    parser.add_argument("--svi-steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--sampling-engine", choices=("full", "refreshing_surrogate"), default="refreshing_surrogate")
    parser.add_argument("--source-plane-covariance-floor", type=float, default=1.0e-6)
    parser.add_argument(
        "--active-scaling-galaxies",
        type=int,
        nargs="+",
        default=None,
        help="Fixed active counts in fixed mode, or adaptive per-potfile caps in adaptive mode. Negative uses all.",
    )
    parser.add_argument("--active-scaling-selection", choices=("fixed", "adaptive"), default="adaptive")
    parser.add_argument("--active-scaling-cumulative-fraction", type=float, default=0.995)
    parser.add_argument("--active-scaling-min", type=int, default=4)
    parser.add_argument(
        "--fit-scaling-scatter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit scaling-relation scatter hyperparameters when subhalos with injected scatter are present.",
    )
    parser.add_argument(
        "--scaling-scatter-max",
        type=float,
        default=0.5,
        help="Upper bound, in natural-log units, for fitted scaling-scatter hyperparameters.",
    )
    parser.add_argument(
        "--posterior-diagnostic-draws",
        type=int,
        default=8,
        help="Maximum posterior draws used for validation uncertainty bars; parameter and mass-profile summaries still use all saved samples.",
    )
    parser.add_argument("--target-accept", type=float, default=0.85)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip the standard solver plot suite. Validation recovery figures are still written as PDFs.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_single_bcg_validation(args)


if __name__ == "__main__":
    main()
