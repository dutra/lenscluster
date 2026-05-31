from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_lenscluster_validation")
os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/numba_cache_{os.getuid()}")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from matplotlib.path import Path as MplPath
from skimage.measure import find_contours

from .jax_cosmology import (
    dpie_sigma0_from_vel_disp as _jax_dpie_sigma0_from_vel_disp,
    flat_wcdm_config,
    kpc_per_arcsec_from_config,
)

ORIGINAL_DPIE_PROFILE_NAME = "DPIE_NIE"
DEFAULT_CAUSTIC_COMPUTE_WINDOW_ARCSEC = 160.0
DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC = 0.2
DEFAULT_CAUSTIC_MIN_AREA_ARCSEC2 = 1.0e-5
DEFAULT_CAUSTIC_BOUNDARY_MARGIN_ARCSEC = 0.5


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
class CausticContour:
    caustic_index: int
    caustic_class: str
    beta_x: np.ndarray
    beta_y: np.ndarray
    critical_x: np.ndarray
    critical_y: np.ndarray
    caustic_area_arcsec2: float
    critical_area_arcsec2: float


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
    n_primary_families: int = 20
    n_subhalo_families: int = 0
    min_images_per_family: int = 3
    max_images_per_family: int | None = None
    max_sources_to_try: int = 400
    caustic_compute_window_arcsec: float = DEFAULT_CAUSTIC_COMPUTE_WINDOW_ARCSEC
    caustic_grid_scale_arcsec: float = DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC
    caustic_min_area_arcsec2: float = DEFAULT_CAUSTIC_MIN_AREA_ARCSEC2
    caustic_boundary_margin_arcsec: float = DEFAULT_CAUSTIC_BOUNDARY_MARGIN_ARCSEC
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

    @property
    def n_families(self) -> int:
        return int(self.n_primary_families) + int(self.n_subhalo_families)


@dataclass(frozen=True)
class MockClusterPaths:
    root: Path
    par_path: Path
    image_catalog_path: Path
    truth_path: Path
    mock_images_path: Path


def _lenstool_ellipticite_to_axis_ratio(ellipticite: float) -> float:
    safe_e = min(max(float(ellipticite), 0.0), 1.0 - 1.0e-9)
    q = math.sqrt((1.0 - safe_e) / (1.0 + safe_e))
    return min(max(q, 1.0e-3), 1.0)


def _axis_ratio_to_lenstool_ellipticite(q: float) -> float:
    safe_q = min(max(float(q), 1.0e-3), 1.0)
    return float((1.0 - safe_q * safe_q) / (1.0 + safe_q * safe_q))


def _config_to_jsonable(config: SingleBCGMockConfig, kpc_per_arcsec: float) -> dict[str, Any]:
    payload = asdict(config)
    payload["n_families"] = int(config.n_families)
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
        ellipticite = _axis_ratio_to_lenstool_ellipticite(q)
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


def _component_kwargs(
    component: DPIETruth,
    config: SingleBCGMockConfig,
    z_source: float,
    cosmo_config: dict[str, Any],
) -> dict[str, float]:
    sigma0 = _dpie_sigma0_from_vel_disp_local(
        component.v_disp,
        component.core_radius_arcsec,
        component.cut_radius_arcsec,
        config.z_lens,
        z_source,
        cosmo_config,
    )
    q = _lenstool_ellipticite_to_axis_ratio(component.ellipticite)
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
    cosmo_config: dict[str, Any],
) -> float:
    return _jax_dpie_sigma0_from_vel_disp(
        vel_disp,
        ra_arcsec,
        rs_arcsec,
        z_lens,
        z_source,
        cosmo_config,
    )


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


def _single_bcg_champ_dmax(images: pd.DataFrame, subhalos: list[dict[str, Any]], padding_arcsec: float = 10.0) -> int:
    radii: list[float] = []
    if not images.empty and {"x_obs_arcsec", "y_obs_arcsec"}.issubset(images.columns):
        image_positions = images.loc[:, ["x_obs_arcsec", "y_obs_arcsec"]].to_numpy(dtype=float)
        finite_images = image_positions[np.isfinite(image_positions).all(axis=1)]
        radii.extend(float(radius) for radius in np.hypot(finite_images[:, 0], finite_images[:, 1]))

    for subhalo in subhalos:
        x_arcsec = float(subhalo["x_arcsec"])
        y_arcsec = float(subhalo["y_arcsec"])
        if np.isfinite(x_arcsec) and np.isfinite(y_arcsec):
            radii.append(float(math.hypot(x_arcsec, y_arcsec)))

    max_radius_arcsec = max(radii, default=0.0)
    return int(math.ceil(max_radius_arcsec + float(padding_arcsec)))


def _write_single_bcg_par(
    path: Path,
    config: SingleBCGMockConfig,
    images: pd.DataFrame,
    subhalos: list[dict[str, Any]] | None = None,
) -> None:
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
    large_scale_lens_count = 2
    nlens = large_scale_lens_count + len(subhalos)
    nlens_opt = large_scale_lens_count
    champ_dmax = _single_bcg_champ_dmax(images, subhalos)
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
    nlens {nlens}
    nlens_opt {nlens_opt}
    nombre 256
    end
{potential_block(config.halo)}
{potential_block(config.bcg)}
{potfile_block}
champ
    dmax {champ_dmax}
    end

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


def _lowercase_image_suffix(index: int) -> str:
    value = int(index)
    if value < 1:
        raise ValueError("Image suffix index must be positive.")
    value -= 1
    letters: list[str] = []
    while True:
        letters.append(chr(ord("a") + (value % 26)))
        value = value // 26 - 1
        if value < 0:
            break
    return "".join(reversed(letters))


def _closed_polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if x_arr.size < 3:
        return 0.0
    if math.hypot(float(x_arr[0] - x_arr[-1]), float(y_arr[0] - y_arr[-1])) > 0.0:
        x_arr = np.concatenate([x_arr, x_arr[:1]])
        y_arr = np.concatenate([y_arr, y_arr[:1]])
    return float(0.5 * abs(np.sum(x_arr[:-1] * y_arr[1:] - x_arr[1:] * y_arr[:-1])))


def _curve_hits_boundary(
    x: np.ndarray,
    y: np.ndarray,
    *,
    compute_window_arcsec: float,
    boundary_margin_arcsec: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> bool:
    half_window = 0.5 * float(compute_window_arcsec)
    margin = max(float(boundary_margin_arcsec), 0.0)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    return bool(
        np.any(x_arr <= center_x - half_window + margin)
        or np.any(x_arr >= center_x + half_window - margin)
        or np.any(y_arr <= center_y - half_window + margin)
        or np.any(y_arr >= center_y + half_window - margin)
    )


def _compute_tangential_caustic_contours(
    lens_model: LensModel,
    kwargs_lens: list[dict[str, float]],
    config: SingleBCGMockConfig,
    *,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> list[CausticContour]:
    compute_window = float(config.caustic_compute_window_arcsec)
    grid_scale = float(config.caustic_grid_scale_arcsec)
    if compute_window <= 0.0 or grid_scale <= 0.0:
        raise ValueError("Caustic compute window and grid scale must be positive.")
    num_pix = max(16, int(math.ceil(compute_window / grid_scale)) + 1)
    if num_pix % 2 == 0:
        num_pix += 1
    x_axis = np.linspace(center_x - 0.5 * compute_window, center_x + 0.5 * compute_window, num_pix)
    y_axis = np.linspace(center_y - 0.5 * compute_window, center_y + 0.5 * compute_window, num_pix)
    xx, yy = np.meshgrid(x_axis, y_axis)
    f_xx, f_xy, f_yx, f_yy = lens_model.hessian(xx.ravel(), yy.ravel(), kwargs_lens)
    f_xx = np.asarray(f_xx, dtype=float).reshape(xx.shape)
    f_yy = np.asarray(f_yy, dtype=float).reshape(xx.shape)
    f_xy = np.asarray(f_xy, dtype=float).reshape(xx.shape)
    f_yx = np.asarray(f_yx, dtype=float).reshape(xx.shape)
    kappa = 0.5 * (f_xx + f_yy)
    gamma1 = 0.5 * (f_xx - f_yy)
    gamma2 = 0.5 * (f_xy + f_yx)
    lambda_tan = 1.0 - kappa - np.hypot(gamma1, gamma2)
    contours: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]] = []
    for vertices in find_contours(lambda_tan, 0.0):
        if vertices.shape[0] < 4:
            continue
        row = vertices[:, 0]
        col = vertices[:, 1]
        crit_x = center_x - 0.5 * compute_window + col * (compute_window / (num_pix - 1))
        crit_y = center_y - 0.5 * compute_window + row * (compute_window / (num_pix - 1))
        if not np.all(np.isfinite(crit_x)) or not np.all(np.isfinite(crit_y)):
            continue
        if _curve_hits_boundary(
            crit_x,
            crit_y,
            compute_window_arcsec=compute_window,
            boundary_margin_arcsec=float(config.caustic_boundary_margin_arcsec),
            center_x=center_x,
            center_y=center_y,
        ):
            continue
        close_gap = math.hypot(float(crit_x[0] - crit_x[-1]), float(crit_y[0] - crit_y[-1]))
        if close_gap > 1.5 * grid_scale:
            continue
        beta_x, beta_y = lens_model.ray_shooting(crit_x, crit_y, kwargs_lens)
        beta_x = np.asarray(beta_x, dtype=float)
        beta_y = np.asarray(beta_y, dtype=float)
        if not np.all(np.isfinite(beta_x)) or not np.all(np.isfinite(beta_y)):
            continue
        crit_area = _closed_polygon_area(crit_x, crit_y)
        caustic_area = _closed_polygon_area(beta_x, beta_y)
        if caustic_area < float(config.caustic_min_area_arcsec2):
            continue
        contours.append((crit_x, crit_y, beta_x, beta_y, crit_area, caustic_area))
    if not contours:
        return []
    primary_position = int(np.argmax([item[5] for item in contours]))
    results: list[CausticContour] = []
    for index, (crit_x, crit_y, beta_x, beta_y, crit_area, caustic_area) in enumerate(contours):
        results.append(
            CausticContour(
                caustic_index=index,
                caustic_class="primary" if index == primary_position else "subhalo",
                beta_x=beta_x,
                beta_y=beta_y,
                critical_x=crit_x,
                critical_y=crit_y,
                caustic_area_arcsec2=float(caustic_area),
                critical_area_arcsec2=float(crit_area),
            )
        )
    return results


def _sample_point_in_caustic(contour: CausticContour, rng: np.random.Generator, *, max_tries: int = 1000) -> tuple[float, float]:
    vertices = np.column_stack([np.asarray(contour.beta_x, dtype=float), np.asarray(contour.beta_y, dtype=float)])
    path = MplPath(vertices)
    x_min, y_min = np.nanmin(vertices, axis=0)
    x_max, y_max = np.nanmax(vertices, axis=0)
    if not np.isfinite([x_min, x_max, y_min, y_max]).all() or x_max <= x_min or y_max <= y_min:
        raise RuntimeError(f"Cannot sample from degenerate caustic contour {contour.caustic_index}.")
    for _ in range(max(1, int(max_tries))):
        x = float(rng.uniform(x_min, x_max))
        y = float(rng.uniform(y_min, y_max))
        if path.contains_point((x, y), radius=0.0):
            return x, y
    raise RuntimeError(f"Failed to sample a point inside caustic contour {contour.caustic_index}.")


def _sample_point_near_caustic(contour: CausticContour, rng: np.random.Generator) -> tuple[float, float]:
    vertices = np.column_stack([np.asarray(contour.beta_x, dtype=float), np.asarray(contour.beta_y, dtype=float)])
    x_min, y_min = np.nanmin(vertices, axis=0)
    x_max, y_max = np.nanmax(vertices, axis=0)
    if not np.isfinite([x_min, x_max, y_min, y_max]).all() or x_max <= x_min or y_max <= y_min:
        raise RuntimeError(f"Cannot sample near degenerate caustic contour {contour.caustic_index}.")
    span_x = max(float(x_max - x_min), 0.1)
    span_y = max(float(y_max - y_min), 0.1)
    return (
        float(rng.uniform(x_min - 0.5 * span_x, x_max + 0.5 * span_x)),
        float(rng.uniform(y_min - 0.5 * span_y, y_max + 0.5 * span_y)),
    )


def _sample_caustic_source_candidate(
    contours: list[CausticContour],
    caustic_class: str,
    rng: np.random.Generator,
    *,
    inside_caustic: bool = True,
) -> tuple[float, float, CausticContour]:
    class_contours = [contour for contour in contours if contour.caustic_class == caustic_class]
    if not class_contours:
        raise RuntimeError(f"No valid {caustic_class} caustics were found for source placement.")
    weights = np.asarray([max(contour.caustic_area_arcsec2, 0.0) for contour in class_contours], dtype=float)
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
        weights = np.ones(len(class_contours), dtype=float)
    probabilities = weights / np.sum(weights)
    contour = class_contours[int(rng.choice(len(class_contours), p=probabilities))]
    if inside_caustic:
        beta_x, beta_y = _sample_point_in_caustic(contour, rng)
    else:
        beta_x, beta_y = _sample_point_near_caustic(contour, rng)
    return beta_x, beta_y, contour


def _caustic_contour_to_json(contour: CausticContour) -> dict[str, Any]:
    return {
        "caustic_index": int(contour.caustic_index),
        "caustic_class": str(contour.caustic_class),
        "caustic_area_arcsec2": float(contour.caustic_area_arcsec2),
        "critical_area_arcsec2": float(contour.critical_area_arcsec2),
        "critical_x": np.asarray(contour.critical_x, dtype=float).tolist(),
        "critical_y": np.asarray(contour.critical_y, dtype=float).tolist(),
        "caustic_beta_x": np.asarray(contour.beta_x, dtype=float).tolist(),
        "caustic_beta_y": np.asarray(contour.beta_y, dtype=float).tolist(),
    }


def _caustic_contours_by_z_from_truth(truth: dict[str, Any]) -> dict[str, list[CausticContour]]:
    parsed: dict[str, list[CausticContour]] = {}
    raw_by_z = truth.get("caustics_by_source_redshift", {})
    if not isinstance(raw_by_z, dict):
        return parsed
    for z_key, raw_contours in raw_by_z.items():
        contours: list[CausticContour] = []
        if not isinstance(raw_contours, list):
            continue
        for raw in raw_contours:
            if not isinstance(raw, dict):
                continue
            required = {"critical_x", "critical_y", "caustic_beta_x", "caustic_beta_y"}
            if not required.issubset(raw):
                continue
            critical_x = np.asarray(raw["critical_x"], dtype=float)
            critical_y = np.asarray(raw["critical_y"], dtype=float)
            beta_x = np.asarray(raw["caustic_beta_x"], dtype=float)
            beta_y = np.asarray(raw["caustic_beta_y"], dtype=float)
            if (
                critical_x.ndim != 1
                or critical_y.ndim != 1
                or beta_x.ndim != 1
                or beta_y.ndim != 1
                or critical_x.size != critical_y.size
                or beta_x.size != beta_y.size
                or critical_x.size < 3
                or beta_x.size < 3
                or not np.all(np.isfinite(critical_x))
                or not np.all(np.isfinite(critical_y))
                or not np.all(np.isfinite(beta_x))
                or not np.all(np.isfinite(beta_y))
            ):
                continue
            contours.append(
                CausticContour(
                    caustic_index=int(raw.get("caustic_index", len(contours))),
                    caustic_class=str(raw.get("caustic_class", "unknown")),
                    beta_x=beta_x,
                    beta_y=beta_y,
                    critical_x=critical_x,
                    critical_y=critical_y,
                    caustic_area_arcsec2=float(raw.get("caustic_area_arcsec2", _closed_polygon_area(beta_x, beta_y))),
                    critical_area_arcsec2=float(raw.get("critical_area_arcsec2", _closed_polygon_area(critical_x, critical_y))),
                )
            )
        if contours:
            parsed[str(z_key)] = contours
    return parsed


def _family_source_redshifts(config: SingleBCGMockConfig) -> tuple[float, ...]:
    values = tuple(float(value) for value in config.source_redshifts if np.isfinite(float(value)) and float(value) > config.z_lens)
    return values if values else (float(config.source_redshift),)


def _image_count_requirement_text(min_images: int, max_images: int | None) -> str:
    if max_images is None:
        return f"at least {int(min_images)} images"
    if int(max_images) == int(min_images):
        return f"exactly {int(min_images)} images"
    return f"between {int(min_images)} and {int(max_images)} images"


def _image_count_within_requirement(n_images: int, min_images: int, max_images: int | None) -> bool:
    return int(n_images) >= int(min_images) and (max_images is None or int(n_images) <= int(max_images))


def _mock_model_and_kwargs(
    config: SingleBCGMockConfig,
    subhalo_components: list[DPIETruth],
    z_source: float,
    cosmo: Any,
    cosmo_config: dict[str, Any],
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
        _component_kwargs(config.halo, config, float(z_source), cosmo_config),
        _component_kwargs(config.bcg, config, float(z_source), cosmo_config),
    ] + [_component_kwargs(component, config, float(z_source), cosmo_config) for component in subhalo_components]
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
    cosmo_config = flat_wcdm_config(h0=70.0, om0=0.3)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    kpc_per_arcsec = kpc_per_arcsec_from_config(config.z_lens, cosmo_config)
    subhalos = _generate_subhalo_catalog(config, rng)
    subhalo_components = [_scaled_subhalo_params(row, config) for row in subhalos]
    source_redshifts = _family_source_redshifts(config)
    model_cache: dict[float, tuple[LensModel, LensEquationSolver, list[dict[str, float]]]] = {}
    caustic_cache: dict[float, list[CausticContour]] = {}

    def get_model(z_source: float) -> tuple[LensModel, LensEquationSolver, list[dict[str, float]]]:
        z_key = float(z_source)
        if z_key not in model_cache:
            model, kwargs_lens = _mock_model_and_kwargs(config, subhalo_components, z_key, cosmo, cosmo_config)
            model_cache[z_key] = (model, LensEquationSolver(model), kwargs_lens)
        return model_cache[z_key]

    def get_caustics(z_source: float) -> list[CausticContour]:
        z_key = float(z_source)
        if z_key not in caustic_cache:
            model, _solver, kwargs_lens = get_model(z_key)
            caustic_cache[z_key] = _compute_tangential_caustic_contours(model, kwargs_lens, config)
        return caustic_cache[z_key]

    image_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    target_classes = ["primary"] * int(config.n_primary_families) + ["subhalo"] * int(config.n_subhalo_families)
    for caustic_class in target_classes:
        z_source = source_redshifts[len(source_rows) % len(source_redshifts)]
        model, solver, kwargs_lens = get_model(z_source)
        contours = get_caustics(z_source)
        accepted: tuple[float, float, CausticContour, np.ndarray, np.ndarray] | None = None
        inside_caustic = config.max_images_per_family is None or int(config.max_images_per_family) > 3
        for _attempt in range(max(1, int(config.max_sources_to_try))):
            beta_x, beta_y, contour = _sample_caustic_source_candidate(
                contours,
                caustic_class,
                rng,
                inside_caustic=inside_caustic,
            )
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
            if _image_count_within_requirement(
                len(x_arr),
                int(config.min_images_per_family),
                config.max_images_per_family,
            ):
                accepted = (float(beta_x), float(beta_y), contour, x_arr, y_arr)
                break
        if accepted is None:
            raise RuntimeError(
                f"Failed to generate a {caustic_class} source with "
                f"{_image_count_requirement_text(config.min_images_per_family, config.max_images_per_family)} "
                f"at z_source={z_source:.6g} after "
                f"{config.max_sources_to_try} attempts."
            )
        beta_x, beta_y, contour, x_arr, y_arr = accepted
        family_id = str(len(source_rows) + 1)
        source_rows.append(
            {
                "family_id": family_id,
                "beta_x": float(beta_x),
                "beta_y": float(beta_y),
                "z_source": float(z_source),
                "n_images": int(len(x_arr)),
                "caustic_class": str(contour.caustic_class),
                "caustic_index": int(contour.caustic_index),
                "caustic_area_arcsec2": float(contour.caustic_area_arcsec2),
                "critical_area_arcsec2": float(contour.critical_area_arcsec2),
            }
        )
        magnification = np.asarray(model.magnification(x_arr, y_arr, kwargs_lens), dtype=float)
        for image_index, (x_true, y_true, mu_true) in enumerate(zip(x_arr, y_arr, magnification), start=1):
            x_obs = float(x_true + rng.normal(0.0, config.pos_sigma_arcsec))
            y_obs = float(y_true + rng.normal(0.0, config.pos_sigma_arcsec))
            image_id = _lowercase_image_suffix(image_index)
            image_rows.append(
                {
                    "image_label": f"{family_id}.{image_id}",
                    "family_id": family_id,
                    "image_id": image_id,
                    "z_source": float(z_source),
                    "x_true_arcsec": float(x_true),
                    "y_true_arcsec": float(y_true),
                    "x_obs_arcsec": x_obs,
                    "y_obs_arcsec": y_obs,
                    "magnification_true": float(mu_true),
                    "caustic_class": str(contour.caustic_class),
                    "caustic_index": int(contour.caustic_index),
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
                f"{row.image_label} {row.x_obs_arcsec:.8f} {row.y_obs_arcsec:.8f} "
                f"0.3734 0.3734 90.0 {row.z_source:.8f} 25.0"
            )
            for row in images.itertuples(index=False)
        )
        + "\n",
        encoding="utf-8",
    )
    if subhalos:
        _write_member_catalog(member_catalog_path, subhalos)
    _write_single_bcg_par(par_path, config, images, subhalos)

    parameter_truth = _truth_parameter_values(config, kpc_per_arcsec)
    parameter_truth["cosmology_Om0"] = 0.3
    parameter_truth["cosmology_w0"] = -1.0
    for source_row in source_rows:
        family_id = str(source_row["family_id"])
        parameter_truth[f"source.{family_id}.beta_x"] = float(source_row["beta_x"])
        parameter_truth[f"source.{family_id}.beta_y"] = float(source_row["beta_y"])

    truth_payload = {
        "mock": "single-bcg-subhalos" if subhalos else "single-bcg",
        "profile_convention": "lenstool_profile_81",
        "config": _config_to_jsonable(config, kpc_per_arcsec),
        "kpc_per_arcsec": kpc_per_arcsec,
        "parameter_truth": parameter_truth,
        "sources": source_rows,
        "images": image_rows,
        "subhalos": subhalos,
        "subhalo_components": [asdict(component) for component in subhalo_components],
        "caustics_by_source_redshift": {
            f"{z:.8f}": [_caustic_contour_to_json(contour) for contour in get_caustics(float(z))]
            for z in source_redshifts
        },
        "lens_model_list": [ORIGINAL_DPIE_PROFILE_NAME, ORIGINAL_DPIE_PROFILE_NAME]
        + [ORIGINAL_DPIE_PROFILE_NAME for _ in subhalo_components],
        "kwargs_lens": get_model(float(config.source_redshift))[2],
        "kwargs_lens_by_source_redshift": {f"{z:.8f}": get_model(float(z))[2] for z in source_redshifts},
    }
    truth_path.write_text(json.dumps(truth_payload, indent=2), encoding="utf-8")
    mock_images_path.write_text(json.dumps(image_rows, indent=2), encoding="utf-8")
    return MockClusterPaths(root, par_path, image_catalog_path, truth_path, mock_images_path), images, truth_payload


def _caustic_config_from_truth(truth: dict[str, Any]) -> SingleBCGMockConfig:
    raw = truth.get("config", {})
    config = raw if isinstance(raw, dict) else {}
    defaults = SingleBCGMockConfig()
    source_redshifts_raw = config.get("source_redshifts", defaults.source_redshifts)
    try:
        source_redshifts = tuple(float(value) for value in source_redshifts_raw)
    except TypeError:
        source_redshifts = defaults.source_redshifts
    max_images_raw = config.get("max_images_per_family", defaults.max_images_per_family)
    max_images_per_family = None if max_images_raw is None else int(max_images_raw)
    return SingleBCGMockConfig(
        z_lens=float(config.get("z_lens", defaults.z_lens)),
        source_redshift=float(config.get("source_redshift", defaults.source_redshift)),
        source_redshifts=source_redshifts,
        n_primary_families=int(config.get("n_primary_families", defaults.n_primary_families)),
        n_subhalo_families=int(config.get("n_subhalo_families", defaults.n_subhalo_families)),
        min_images_per_family=int(config.get("min_images_per_family", defaults.min_images_per_family)),
        max_images_per_family=max_images_per_family,
        caustic_compute_window_arcsec=float(
            config.get("caustic_compute_window_arcsec", defaults.caustic_compute_window_arcsec)
        ),
        caustic_grid_scale_arcsec=float(config.get("caustic_grid_scale_arcsec", defaults.caustic_grid_scale_arcsec)),
        caustic_min_area_arcsec2=float(config.get("caustic_min_area_arcsec2", defaults.caustic_min_area_arcsec2)),
        caustic_boundary_margin_arcsec=float(
            config.get("caustic_boundary_margin_arcsec", defaults.caustic_boundary_margin_arcsec)
        ),
    )
