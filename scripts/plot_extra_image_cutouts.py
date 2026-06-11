#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-extra-image-cutouts")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
import astropy.units as u

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_path in (SCRIPT_DIR, REPO_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from analyze_literature_family_diagnostics import load_master_catalog  # noqa: E402
from plot_literature_family_cutouts import (  # noqa: E402
    AXIS_FACE_COLOR,
    CUTOUT_LABEL_FONT_SIZE,
    CUTOUT_PANEL_SIZE_INCH,
    CUTOUT_TEXT_BBOX,
    DEFAULT_BANDS,
    DEFAULT_CUTOUT_SIZE_ARCSEC,
    DEFAULT_HFF_CATALOG_ROOT,
    DEFAULT_IMAGE_DIR,
    DEFAULT_IMAGE_SCALE,
    IMAGE_SCALE_CHOICES,
    PAGE_FACE_COLOR,
    SAVEFIG_KWARGS,
    BandImage,
    _draw_marker,
    _format_number,
    _style_cutout_axis,
    _style_cutout_figure,
    build_rgb_display,
    extract_band_cutout,
    find_rgb_band_paths,
    load_rgb_metadata,
    make_rgb_cutout,
)


DEFAULT_MATCH_RADIUS_ARCSEC = 0.5
DEFAULT_IMAGES_PER_PAGE = 12
TABLES_DIR_NAME = "tables"
ARTIFACTS_DIR_NAME = "artifacts"
IMAGE_FIT_QUALITY_CSV_NAME = "image_fit_quality.csv"
EXTRA_IMAGES_CSV_NAME = "image_recovery_extra_images.csv"
DEFAULT_OUTPUT_NAME = "extra_image_cutouts.pdf"
DEFAULT_EXTRAS_OUTPUT_NAME = "extra_image_redshift_marked_cutouts.pdf"
DEFAULT_MATCHES_NAME = "extra_image_cutout_matches.csv"
REQUIRED_IMAGE_FIT_COLUMNS = ("family_id", "image_label", "x_obs_arcsec", "y_obs_arcsec", "image_recovery_status")
REQUIRED_EXTRA_COLUMNS = ("family_id", "extra_image_index", "x_model_arcsec", "y_model_arcsec")
STATUS_OBSERVED = "OBSERVED"
STATUS_RECOVERED = "RECOVERED"
STATUS_MISSING = "MISSING"
STATUS_EXTRA = "EXTRA"
STATUS_LABEL_COLORS = {
    STATUS_OBSERVED: "#00e5ff",
    STATUS_RECOVERED: "#43d463",
    STATUS_MISSING: "#ff4d5e",
    STATUS_EXTRA: "#ffdf4d",
}
STATUS_LABEL_FONT_SIZE = 7.0
STATUS_TEXT_BBOX = {"facecolor": "black", "alpha": 0.62, "edgecolor": "none", "pad": 0.55}
DETAIL_TEXT_Y = 0.78
OBSERVED_MARKER_COLOR = STATUS_LABEL_COLORS[STATUS_OBSERVED]
RECOVERED_MARKER_COLOR = STATUS_LABEL_COLORS[STATUS_RECOVERED]
MISSING_MARKER_COLOR = STATUS_LABEL_COLORS[STATUS_MISSING]
EXTRA_MARKER_COLOR = "#ffdf4d"
MASTER_MARKER_COLOR = "#00d5ff"
LABEL_COLOR_DEFAULT = "white"
LABEL_COLOR_SPECZ_MATCH = "#43d463"
LABEL_COLOR_PHOTOZ_MATCH = "#4da3ff"
SPECZ_LABEL_DELTA_MAX = 0.1
PHOTOZ_LABEL_DELTA_MAX = 0.5
CLUSTER_ALIASES = {
    "a307": "a370",
    "a370": "a370",
    "abell370": "a370",
    "a2744": "a2744",
    "a2744_cats": "a2744",
    "abell2744": "a2744",
    "as1063": "as1063",
    "abells1063": "as1063",
    "rxcj2248": "as1063",
    "m0416": "m0416",
    "macs0416": "m0416",
    "m0717": "m0717",
    "macs0717": "m0717",
    "m1149": "m1149",
    "macs1149": "m1149",
}
CLUSTER_TOKEN_ORDER = tuple(sorted(CLUSTER_ALIASES, key=len, reverse=True))
MAG_BANDS = ("F435W", "F606W", "F814W", "F105W", "F125W", "F140W", "F160W")


@dataclass(frozen=True)
class ReferenceFrame:
    reference_kind: int
    ra0_deg: float
    dec0_deg: float
    par_path: Path | None = None


def resolve_stage_dir(stage_dir: Path) -> Path:
    path = Path(stage_dir)
    if not path.is_dir():
        raise ValueError(f"stage_dir must be an existing directory: {path}")
    return path


def extra_images_csv_path(stage_dir: Path) -> Path:
    return resolve_stage_dir(stage_dir) / TABLES_DIR_NAME / EXTRA_IMAGES_CSV_NAME


def image_fit_quality_csv_path(stage_dir: Path) -> Path:
    return resolve_stage_dir(stage_dir) / TABLES_DIR_NAME / IMAGE_FIT_QUALITY_CSV_NAME


def default_output_path(stage_dir: Path) -> Path:
    return resolve_stage_dir(stage_dir) / TABLES_DIR_NAME / DEFAULT_OUTPUT_NAME


def default_extras_output_path(stage_dir: Path) -> Path:
    return resolve_stage_dir(stage_dir) / TABLES_DIR_NAME / DEFAULT_EXTRAS_OUTPUT_NAME


def default_matches_output_path(stage_dir: Path) -> Path:
    return resolve_stage_dir(stage_dir) / TABLES_DIR_NAME / DEFAULT_MATCHES_NAME


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def offsets_to_radec(
    x_arcsec: Any,
    y_arcsec: Any,
    *,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[float, float]:
    x_value = float(x_arcsec)
    y_value = float(y_arcsec)
    cos_dec0 = math.cos(math.radians(float(dec0_deg)))
    if cos_dec0 == 0.0:
        raise ValueError("Reference declination is too close to a pole for offset conversion.")
    return float(ra0_deg) - x_value / (3600.0 * cos_dec0), float(dec0_deg) + y_value / 3600.0


def _read_h5_json(path: Path, key: str) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        if key not in handle:
            return {}
        raw = handle[key][()]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return json.loads(text)


def _resolve_par_path(path_value: Any, run_dir: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    for base in (REPO_ROOT, run_dir, Path.cwd()):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (REPO_ROOT / candidate).resolve()


def _parse_reference_from_par(par_path: Path) -> ReferenceFrame:
    in_runmode = False
    with Path(par_path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line == "runmode":
                in_runmode = True
                continue
            if in_runmode and line == "end":
                break
            if not in_runmode:
                continue
            parts = line.split()
            if parts and parts[0] == "reference" and len(parts) >= 4:
                return ReferenceFrame(int(parts[1]), float(parts[2]), float(parts[3]), Path(par_path))
    raise ValueError(f"Could not find runmode.reference in {par_path}")


def load_reference_frame(stage_dir: Path, par_path: Path | None = None) -> ReferenceFrame:
    run_dir = resolve_stage_dir(stage_dir)
    artifact = run_dir / ARTIFACTS_DIR_NAME / "plot_bundle.h5"
    artifact_par_path: Path | None = None
    if artifact.exists():
        try:
            meta = _read_h5_json(artifact, "state/build_state_meta_json")
            reference = meta.get("reference")
            artifact_par_path = _resolve_par_path(meta.get("par_path"), run_dir)
            if isinstance(reference, list) and len(reference) >= 3:
                return ReferenceFrame(int(reference[0]), float(reference[1]), float(reference[2]), artifact_par_path)
        except Exception:
            pass
        try:
            cli_args = _read_h5_json(artifact, "cli_args_json")
            artifact_par_path = _resolve_par_path(cli_args.get("par_path"), run_dir)
        except Exception:
            artifact_par_path = None

    fallback_par_path = Path(par_path) if par_path is not None else artifact_par_path
    if fallback_par_path is None:
        raise ValueError(
            "Could not infer runmode.reference from plot_bundle.h5. Pass --par-path to read it from the Lenstool .par file."
        )
    return _parse_reference_from_par(fallback_par_path)


def _canonical_cluster(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if not normalized:
        return None
    for token in CLUSTER_TOKEN_ORDER:
        if re.search(rf"(^|_){re.escape(token)}($|_)", normalized):
            return CLUSTER_ALIASES[token]
    return CLUSTER_ALIASES.get(normalized, normalized)


def infer_cluster(stage_dir: Path, reference: ReferenceFrame | None = None) -> str:
    candidates: list[str] = []
    if reference is not None and reference.par_path is not None:
        candidates.extend(str(part) for part in reference.par_path.parts)
    candidates.extend(str(part) for part in Path(stage_dir).parts)
    for candidate in candidates:
        cluster = _canonical_cluster(candidate)
        if cluster is not None:
            return cluster
    return "a370"


def _read_csv_allow_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _add_offset_radec_columns(
    data: pd.DataFrame,
    reference: ReferenceFrame,
    *,
    x_column: str,
    y_column: str,
    ra_column: str,
    dec_column: str,
) -> pd.DataFrame:
    data = data.copy()
    ra_values: list[float] = []
    dec_values: list[float] = []
    for row in data.itertuples(index=False):
        x_value = _safe_float(getattr(row, x_column))
        y_value = _safe_float(getattr(row, y_column))
        if np.isfinite(x_value) and np.isfinite(y_value):
            ra, dec = offsets_to_radec(
                x_value,
                y_value,
                ra0_deg=reference.ra0_deg,
                dec0_deg=reference.dec0_deg,
            )
        else:
            ra, dec = float("nan"), float("nan")
        ra_values.append(ra)
        dec_values.append(dec)
    data[ra_column] = ra_values
    data[dec_column] = dec_values
    return data


def _empty_extra_images() -> pd.DataFrame:
    return pd.DataFrame(columns=[*REQUIRED_EXTRA_COLUMNS, "ra", "dec", "extra_label"])


def load_extra_images(extra_images_csv: Path, reference: ReferenceFrame) -> pd.DataFrame:
    data = _read_csv_allow_empty(extra_images_csv)
    if data.empty and not set(REQUIRED_EXTRA_COLUMNS).issubset(data.columns):
        return _empty_extra_images()
    missing = [column for column in REQUIRED_EXTRA_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{extra_images_csv} is missing required columns: {missing}")
    data = _add_offset_radec_columns(
        data,
        reference,
        x_column="x_model_arcsec",
        y_column="y_model_arcsec",
        ra_column="ra",
        dec_column="dec",
    )
    data["extra_label"] = [
        f"{family}.extra{int(index)}"
        for family, index in zip(data["family_id"].astype(str), data["extra_image_index"], strict=True)
    ]
    return data


def _model_position_from_row(row: pd.Series) -> tuple[float, float]:
    x_model = _safe_float(row.get("x_model_q50"))
    y_model = _safe_float(row.get("y_model_q50"))
    if np.isfinite(x_model) and np.isfinite(y_model):
        return x_model, y_model
    return _safe_float(row.get("x_model_arcsec")), _safe_float(row.get("y_model_arcsec"))


def load_image_fit_quality(image_fit_csv: Path, reference: ReferenceFrame) -> pd.DataFrame:
    data = _read_csv_allow_empty(image_fit_csv)
    missing = [column for column in REQUIRED_IMAGE_FIT_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{image_fit_csv} is missing required columns: {missing}")
    data = _add_offset_radec_columns(
        data,
        reference,
        x_column="x_obs_arcsec",
        y_column="y_obs_arcsec",
        ra_column="observed_ra",
        dec_column="observed_dec",
    )
    model_x_values: list[float] = []
    model_y_values: list[float] = []
    for _, row in data.iterrows():
        x_model, y_model = _model_position_from_row(row)
        model_x_values.append(x_model)
        model_y_values.append(y_model)
    data["x_model_cutout_arcsec"] = model_x_values
    data["y_model_cutout_arcsec"] = model_y_values
    data = _add_offset_radec_columns(
        data,
        reference,
        x_column="x_model_cutout_arcsec",
        y_column="y_model_cutout_arcsec",
        ra_column="model_ra",
        dec_column="model_dec",
    )
    return data


def _master_field(master_row: pd.Series | None, column: str, default: Any = np.nan) -> Any:
    if master_row is None:
        return default
    return master_row.get(column, default)


def _match_columns(row: pd.Series, master_row: pd.Series | None, separation_arcsec: float | None) -> dict[str, Any]:
    result = row.to_dict()
    matched = master_row is not None and separation_arcsec is not None and np.isfinite(float(separation_arcsec))
    result.update(
        {
            "matched": bool(matched),
            "match_separation_arcsec": np.nan if separation_arcsec is None else float(separation_arcsec),
            "master_row_index": np.nan if master_row is None else int(master_row.name),
            "master_object_id": "" if master_row is None else str(_master_field(master_row, "object_id", "")),
            "master_object_source": "" if master_row is None else str(_master_field(master_row, "object_source", "")),
            "master_catalog_sources": "" if master_row is None else str(_master_field(master_row, "catalog_sources", "")),
            "master_ra": _safe_float(_master_field(master_row, "ra")),
            "master_dec": _safe_float(_master_field(master_row, "dec")),
            "master_zspec_best": _safe_float(_master_field(master_row, "zspec_best")),
            "master_zspec_best_source": "" if master_row is None else str(_master_field(master_row, "zspec_best_source", "")),
            "master_zspec_best_confidence": "" if master_row is None else str(_master_field(master_row, "zspec_best_confidence", "")),
            "master_zphot_best": _safe_float(_master_field(master_row, "zphot_best")),
        }
    )
    for band in MAG_BANDS:
        result[f"master_mag_{band}"] = _safe_float(_master_field(master_row, f"mag_{band}"))
    return result


def _empty_match_columns(extra_images: pd.DataFrame) -> list[str]:
    empty_row = pd.Series({column: np.nan for column in extra_images.columns})
    return list(_match_columns(empty_row, None, None))


def match_extra_images_to_master(
    extra_images: pd.DataFrame,
    master: pd.DataFrame,
    *,
    radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    extra_images = extra_images.reset_index(drop=True)
    master = master.reset_index(drop=True)
    matches: dict[int, tuple[int, float]] = {}
    if extra_images.empty:
        return pd.DataFrame(columns=_empty_match_columns(extra_images))

    if not extra_images.empty and not master.empty:
        extra_ra = pd.to_numeric(extra_images["ra"], errors="coerce").to_numpy(dtype=float)
        extra_dec = pd.to_numeric(extra_images["dec"], errors="coerce").to_numpy(dtype=float)
        master_ra = pd.to_numeric(master.get("ra"), errors="coerce").to_numpy(dtype=float)
        master_dec = pd.to_numeric(master.get("dec"), errors="coerce").to_numpy(dtype=float)
        valid_extra = np.isfinite(extra_ra) & np.isfinite(extra_dec)
        valid_master = np.isfinite(master_ra) & np.isfinite(master_dec)
        if valid_extra.any() and valid_master.any():
            extra_indices = np.flatnonzero(valid_extra)
            master_indices = np.flatnonzero(valid_master)
            extra_coords = SkyCoord(ra=extra_ra[valid_extra] * u.deg, dec=extra_dec[valid_extra] * u.deg)
            master_coords = SkyCoord(ra=master_ra[valid_master] * u.deg, dec=master_dec[valid_master] * u.deg)
            nearest, separation, _ = extra_coords.match_to_catalog_sky(master_coords)
            for local_extra_index, local_master_index, sep in zip(extra_indices, nearest, separation.arcsec, strict=True):
                if float(sep) <= float(radius_arcsec):
                    matches[int(local_extra_index)] = (int(master_indices[int(local_master_index)]), float(sep))

    for index, row in extra_images.iterrows():
        match = matches.get(int(index))
        master_row = master.iloc[match[0]] if match is not None else None
        rows.append(_match_columns(row, master_row, None if match is None else match[1]))
    return pd.DataFrame(rows)


def _coord_from_row(row: pd.Series, prefix: str = "") -> SkyCoord | None:
    ra = _safe_float(row.get(f"{prefix}ra"))
    dec = _safe_float(row.get(f"{prefix}dec"))
    if not np.isfinite(ra) or not np.isfinite(dec):
        return None
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")


def _format_z(value: Any, prefix: str) -> str:
    numeric = _safe_float(value)
    return f"{prefix}=na" if not np.isfinite(numeric) else f"{prefix}={numeric:.4g}"


def _format_match_label(row: pd.Series) -> str:
    if not bool(row.get("matched", False)):
        return "master: no match"
    object_id = str(row.get("master_object_id", "")).strip() or "unknown"
    separation = _safe_float(row.get("match_separation_arcsec"))
    zspec = _format_z(row.get("master_zspec_best"), "zs")
    zphot = _format_z(row.get("master_zphot_best"), "zp")
    source = str(row.get("master_object_source", "")).strip()
    source_suffix = f" {source}" if source else ""
    return f'master: {object_id}{source_suffix} ({separation:.3f}")\n{zspec}  {zphot}'


def _panel_label(row: pd.Series) -> str:
    x_model = _safe_float(row.get("x_model_arcsec"))
    y_model = _safe_float(row.get("y_model_arcsec"))
    redshift = _format_z(row.get("z_source"), "z")
    effective_redshift = _format_z(row.get("effective_z_source"), "zeff")
    return (
        f"fam {row.get('family_id')} extra {row.get('extra_image_index')}\n"
        f"x={x_model:.2f} y={y_model:.2f}\n"
        f"{redshift}  {effective_redshift}\n"
        f"{_format_match_label(row)}"
    )


def _within_redshift_delta(value: Any, reference: Any, max_delta: float) -> bool:
    numeric = _safe_float(value)
    reference_numeric = _safe_float(reference)
    if not np.isfinite(numeric) or not np.isfinite(reference_numeric):
        return False
    return abs(numeric - reference_numeric) <= float(max_delta) + 1.0e-12


def _panel_label_color(row: pd.Series) -> str:
    if not bool(row.get("matched", False)):
        return LABEL_COLOR_DEFAULT
    z_source = row.get("z_source")
    if not np.isfinite(_safe_float(z_source)):
        return LABEL_COLOR_DEFAULT
    if _within_redshift_delta(z_source, row.get("master_zspec_best"), SPECZ_LABEL_DELTA_MAX):
        return LABEL_COLOR_SPECZ_MATCH
    if _within_redshift_delta(z_source, row.get("master_zphot_best"), PHOTOZ_LABEL_DELTA_MAX):
        return LABEL_COLOR_PHOTOZ_MATCH
    return LABEL_COLOR_DEFAULT


def _status_label_color(status: str) -> str:
    return STATUS_LABEL_COLORS.get(str(status).upper(), LABEL_COLOR_DEFAULT)


def _format_xy_label(prefix: str, x_value: Any, y_value: Any) -> str:
    return f"{prefix} x={_format_number(x_value, precision=3)} y={_format_number(y_value, precision=3)}"


def _image_panel_label(row: pd.Series) -> str:
    residual = _safe_float(row.get("image_residual_q50"))
    if not np.isfinite(residual):
        residual = _safe_float(row.get("image_residual_arcsec"))
    redshift = _format_z(row.get("z_source"), "z")
    effective_redshift = _format_z(row.get("effective_z_source"), "zeff")
    return (
        f"fam {row.get('family_id')} image {row.get('image_label')}\n"
        f"{_format_xy_label('obs', row.get('x_obs_arcsec'), row.get('y_obs_arcsec'))}\n"
        f"{_format_xy_label('model', row.get('x_model_cutout_arcsec'), row.get('y_model_cutout_arcsec'))}\n"
        f"resid={_format_number(residual, precision=3)}  {redshift}  {effective_redshift}"
    )


def _family_sort_key_value(family_id: Any) -> tuple[int, str]:
    family_text = str(family_id)
    family_number = int(family_text) if family_text.isdigit() else 10**9
    return family_number, family_text


def _family_ids_for_panels(image_fit: pd.DataFrame, extra_matches: pd.DataFrame) -> list[str]:
    family_ids = set(image_fit.get("family_id", pd.Series(dtype=object)).dropna().astype(str))
    family_ids.update(extra_matches.get("family_id", pd.Series(dtype=object)).dropna().astype(str))
    return sorted(family_ids, key=_family_sort_key_value)


def _image_panel(row: pd.Series, *, panel_status: str, panel_index: int, center_prefix: str) -> dict[str, Any]:
    result = row.to_dict()
    result.update(
        {
            "panel_status": panel_status,
            "panel_index": int(panel_index),
            "ra": _safe_float(row.get(f"{center_prefix}_ra")),
            "dec": _safe_float(row.get(f"{center_prefix}_dec")),
            "detail_label": _image_panel_label(row),
            "detail_label_color": LABEL_COLOR_DEFAULT,
            "status_label_color": _status_label_color(panel_status),
        }
    )
    return result


def _extra_panel(row: pd.Series, *, panel_index: int, mark_status_by_redshift: bool = False) -> dict[str, Any]:
    redshift_color = _panel_label_color(row)
    result = row.to_dict()
    result.update(
        {
            "panel_status": STATUS_EXTRA,
            "panel_index": int(panel_index),
            "detail_label": _panel_label(row),
            "detail_label_color": redshift_color,
            "status_label_color": redshift_color if mark_status_by_redshift else _status_label_color(STATUS_EXTRA),
        }
    )
    return result


def _sorted_family_extra_matches(extra_matches: pd.DataFrame, family_id: Any) -> pd.DataFrame:
    if "family_id" not in extra_matches.columns:
        return extra_matches.iloc[0:0]
    family_extra = extra_matches.loc[extra_matches["family_id"].astype(str) == str(family_id)]
    if family_extra.empty:
        return family_extra
    return family_extra.iloc[
        sorted(range(len(family_extra)), key=lambda index: _sort_key(family_extra.iloc[index]))
    ].reset_index(drop=True)


def build_recovery_cutout_panels(image_fit: pd.DataFrame, extra_matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_id in _family_ids_for_panels(image_fit, extra_matches):
        panel_index = 0
        family_images = image_fit.loc[image_fit["family_id"].astype(str) == str(family_id)].reset_index(drop=True)
        recovered = family_images.loc[
            family_images["image_recovery_status"].fillna("").astype(str).str.lower().eq("recovered")
        ].reset_index(drop=True)
        missing = family_images.loc[
            ~family_images["image_recovery_status"].fillna("").astype(str).str.lower().eq("recovered")
        ].reset_index(drop=True)
        for _, image_row in recovered.iterrows():
            rows.append(_image_panel(image_row, panel_status=STATUS_OBSERVED, panel_index=panel_index, center_prefix="observed"))
            panel_index += 1
            rows.append(_image_panel(image_row, panel_status=STATUS_RECOVERED, panel_index=panel_index, center_prefix="model"))
            panel_index += 1
        for _, image_row in missing.iterrows():
            rows.append(_image_panel(image_row, panel_status=STATUS_MISSING, panel_index=panel_index, center_prefix="observed"))
            panel_index += 1

        family_extra = _sorted_family_extra_matches(extra_matches, family_id)
        for _, extra_row in family_extra.iterrows():
            rows.append(_extra_panel(extra_row, panel_index=panel_index))
            panel_index += 1
    return pd.DataFrame(rows)


def build_extra_redshift_cutout_panels(extra_matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_id in _family_ids_for_panels(pd.DataFrame(), extra_matches):
        for panel_index, (_, extra_row) in enumerate(_sorted_family_extra_matches(extra_matches, family_id).iterrows()):
            rows.append(_extra_panel(extra_row, panel_index=panel_index, mark_status_by_redshift=True))
    return pd.DataFrame(rows)


def _draw_extra_marker(
    ax: plt.Axes,
    image: BandImage,
    center_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
) -> None:
    _draw_marker(
        ax,
        image,
        center_coord,
        center_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        edgecolor=EXTRA_MARKER_COLOR,
        alpha=0.9,
        linestyle="-",
        linewidth=0.95,
        zorder=7,
    )


def _draw_master_marker(
    ax: plt.Axes,
    image: BandImage,
    row: pd.Series,
    center_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
) -> None:
    master_coord = _coord_from_row(row, "master_")
    if master_coord is None:
        return
    _draw_marker(
        ax,
        image,
        center_coord,
        master_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        edgecolor=MASTER_MARKER_COLOR,
        alpha=0.85,
        linestyle="--",
        linewidth=0.85,
        zorder=8,
    )


def _coord_from_columns(row: pd.Series, ra_column: str, dec_column: str) -> SkyCoord | None:
    ra = _safe_float(row.get(ra_column))
    dec = _safe_float(row.get(dec_column))
    if not np.isfinite(ra) or not np.isfinite(dec):
        return None
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")


def _draw_row_marker(
    ax: plt.Axes,
    image: BandImage,
    row: pd.Series,
    center_coord: SkyCoord,
    *,
    ra_column: str,
    dec_column: str,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    edgecolor: str,
    linestyle: str = "-",
    linewidth: float = 0.9,
    alpha: float = 0.9,
    zorder: float = 8,
) -> None:
    target_coord = _coord_from_columns(row, ra_column, dec_column)
    if target_coord is None:
        return
    _draw_marker(
        ax,
        image,
        center_coord,
        target_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        edgecolor=edgecolor,
        alpha=alpha,
        linestyle=linestyle,
        linewidth=linewidth,
        zorder=zorder,
    )


def _draw_recovery_panel_markers(
    ax: plt.Axes,
    image: BandImage,
    row: pd.Series,
    center_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
) -> None:
    status = str(row.get("panel_status", "")).upper()
    if status == STATUS_EXTRA:
        _draw_extra_marker(
            ax,
            image,
            center_coord,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
        )
        _draw_master_marker(
            ax,
            image,
            row,
            center_coord,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
        )
        return
    if status == STATUS_MISSING:
        _draw_row_marker(
            ax,
            image,
            row,
            center_coord,
            ra_column="observed_ra",
            dec_column="observed_dec",
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
            edgecolor=MISSING_MARKER_COLOR,
            linewidth=1.0,
            zorder=9,
        )
        return
    if status == STATUS_OBSERVED:
        _draw_row_marker(
            ax,
            image,
            row,
            center_coord,
            ra_column="observed_ra",
            dec_column="observed_dec",
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
            edgecolor=OBSERVED_MARKER_COLOR,
            linewidth=1.0,
            zorder=9,
        )
        _draw_row_marker(
            ax,
            image,
            row,
            center_coord,
            ra_column="model_ra",
            dec_column="model_dec",
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
            edgecolor=RECOVERED_MARKER_COLOR,
            linestyle="--",
            linewidth=0.85,
            alpha=0.85,
            zorder=8,
        )
        return
    if status == STATUS_RECOVERED:
        _draw_row_marker(
            ax,
            image,
            row,
            center_coord,
            ra_column="model_ra",
            dec_column="model_dec",
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
            edgecolor=RECOVERED_MARKER_COLOR,
            linewidth=1.0,
            zorder=9,
        )
        _draw_row_marker(
            ax,
            image,
            row,
            center_coord,
            ra_column="observed_ra",
            dec_column="observed_dec",
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
            edgecolor=OBSERVED_MARKER_COLOR,
            linestyle="--",
            linewidth=0.85,
            alpha=0.85,
            zorder=8,
        )


def _draw_extra_legend(ax: plt.Axes) -> None:
    legend_bbox = {"facecolor": "black", "alpha": 0.38, "edgecolor": "none", "pad": 0.45}
    ax.text(
        0.73,
        0.982,
        " \n ",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=4.6,
        color="white",
        linespacing=0.85,
        clip_on=True,
        bbox=legend_bbox,
    )
    entries = (("extra", EXTRA_MARKER_COLOR, "-", 0.955), ("master", MASTER_MARKER_COLOR, "--", 0.905))
    for label, color, linestyle, y_pos in entries:
        ax.add_patch(
            Circle(
                (0.755, y_pos),
                radius=0.012,
                transform=ax.transAxes,
                edgecolor=color,
                facecolor="none",
                linewidth=0.7,
                linestyle=linestyle,
                alpha=0.9,
                zorder=9,
                clip_on=True,
            )
        )
        ax.text(
            0.782,
            y_pos,
            label,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=4.6,
            color="white",
            clip_on=True,
            zorder=9,
        )


def _sort_key(row: pd.Series) -> tuple[int, str, int, int]:
    family_number, family_text = _family_sort_key_value(row.get("family_id", ""))
    panel_index = _safe_float(row.get("panel_index"))
    extra_index = _safe_float(row.get("extra_image_index"))
    sequence_index = int(panel_index) if np.isfinite(panel_index) else int(extra_index) if np.isfinite(extra_index) else 10**9
    return family_number, family_text, sequence_index, int(row.name)


def _figure_size(rows: int, cols: int) -> tuple[float, float]:
    return CUTOUT_PANEL_SIZE_INCH * float(cols), CUTOUT_PANEL_SIZE_INCH * float(rows)


def _sorted_panel_rows(panels: pd.DataFrame) -> pd.DataFrame:
    sorted_indices = sorted(range(len(panels)), key=lambda idx: _sort_key(panels.iloc[idx]))
    return panels.iloc[sorted_indices].reset_index(drop=True)


def _family_page_groups(panels: pd.DataFrame, images_per_page: int) -> list[list[pd.DataFrame]]:
    if images_per_page <= 0:
        raise ValueError("--images-per-page must be positive.")

    sorted_matches = _sorted_panel_rows(panels)
    family_groups = [
        family.reset_index(drop=True)
        for _, family in sorted_matches.groupby(sorted_matches["family_id"].astype(str), sort=False)
    ]
    pages: list[list[pd.DataFrame]] = []
    current_page: list[pd.DataFrame] = []
    current_count = 0
    for family in family_groups:
        family_count = len(family)
        if current_page and current_count + family_count > int(images_per_page):
            pages.append(current_page)
            current_page = []
            current_count = 0
        current_page.append(family)
        current_count += family_count
    if current_page:
        pages.append(current_page)
    return pages


def write_extra_image_cutout_pdf(
    panels: pd.DataFrame,
    band_images: dict[str, BandImage],
    output: Path,
    *,
    bands: Sequence[str] = DEFAULT_BANDS,
    cutout_size_arcsec: float = DEFAULT_CUTOUT_SIZE_ARCSEC,
    images_per_page: int = DEFAULT_IMAGES_PER_PAGE,
) -> int:
    if panels.empty:
        raise ValueError("No recovery image panels are available for cutout plotting.")
    if images_per_page <= 0:
        raise ValueError("--images-per-page must be positive.")
    output = Path(output).with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    page_groups = _family_page_groups(panels, int(images_per_page))
    n_pages = len(page_groups)
    rgb_display = build_rgb_display(band_images, bands=bands)
    with PdfPages(output) as pdf:
        for page_families in page_groups:
            rows = len(page_families)
            cols = max(len(family) for family in page_families)
            fig, axes = plt.subplots(rows, cols, figsize=_figure_size(rows, cols), squeeze=False)
            _style_cutout_figure(fig)
            fig.patch.set_facecolor(PAGE_FACE_COLOR)
            for row_index, family in enumerate(page_families):
                for col_index in range(cols):
                    ax = axes[row_index, col_index]
                    _style_cutout_axis(ax)
                    ax.set_facecolor(AXIS_FACE_COLOR)
                    if col_index >= len(family):
                        ax.set_axis_off()
                        continue
                    image_row = family.iloc[col_index]
                    coord = _coord_from_row(image_row)
                    if coord is None:
                        ax.set_axis_off()
                        continue
                    cutouts = {
                        str(band): extract_band_cutout(band_images[str(band)], coord, cutout_size_arcsec=cutout_size_arcsec)
                        for band in bands
                    }
                    rgb = make_rgb_cutout(cutouts, bands=bands, rgb_display=rgb_display)
                    ax.imshow(rgb, origin="lower", interpolation="nearest")
                    reference_image = band_images[str(bands[-1])]
                    _draw_recovery_panel_markers(
                        ax,
                        reference_image,
                        image_row,
                        coord,
                        cutout_size_arcsec=cutout_size_arcsec,
                        rendered_shape=rgb.shape[:2],
                    )
                    if str(image_row.get("panel_status", "")).upper() == STATUS_EXTRA:
                        _draw_extra_legend(ax)
                    ax.text(
                        0.04,
                        0.96,
                        str(image_row.get("panel_status", "")).upper(),
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=STATUS_LABEL_FONT_SIZE,
                        fontweight="bold",
                        color=str(image_row.get("status_label_color", LABEL_COLOR_DEFAULT)),
                        linespacing=0.95,
                        clip_on=True,
                        bbox=STATUS_TEXT_BBOX,
                    )
                    ax.text(
                        0.04,
                        DETAIL_TEXT_Y,
                        str(image_row.get("detail_label", "")),
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=CUTOUT_LABEL_FONT_SIZE,
                        color=str(image_row.get("detail_label_color", LABEL_COLOR_DEFAULT)),
                        linespacing=0.95,
                        clip_on=True,
                        bbox=CUTOUT_TEXT_BBOX,
                    )
            pdf.savefig(fig, facecolor=fig.get_facecolor(), **SAVEFIG_KWARGS)
            plt.close(fig)
    return n_pages


def run(
    stage_dir: Path,
    *,
    output: Path | None = None,
    extras_output: Path | None = None,
    matches_output: Path | None = None,
    cluster: str | None = None,
    catalog_root: Path = DEFAULT_HFF_CATALOG_ROOT,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    image_scale: str = DEFAULT_IMAGE_SCALE,
    bands: Sequence[str] = DEFAULT_BANDS,
    match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
    cutout_size_arcsec: float = DEFAULT_CUTOUT_SIZE_ARCSEC,
    images_per_page: int = DEFAULT_IMAGES_PER_PAGE,
    par_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    stage_dir = resolve_stage_dir(stage_dir)
    image_fit_csv = image_fit_quality_csv_path(stage_dir)
    extra_images_csv = extra_images_csv_path(stage_dir)
    if not image_fit_csv.exists():
        raise FileNotFoundError(f"Missing image-fit quality CSV: {image_fit_csv}")
    if not extra_images_csv.exists():
        raise FileNotFoundError(f"Missing extra-images CSV: {extra_images_csv}")
    if len(bands) != 3:
        raise ValueError("--bands must provide exactly three bands in blue green red order.")
    reference = load_reference_frame(stage_dir, par_path=par_path)
    cluster_key = _canonical_cluster(cluster) if cluster is not None else infer_cluster(stage_dir, reference)
    if cluster_key is None:
        raise ValueError("Could not infer cluster key; pass --cluster.")
    image_fit = load_image_fit_quality(image_fit_csv, reference)
    extra_images = load_extra_images(extra_images_csv, reference)
    master = load_master_catalog(Path(catalog_root), cluster_key)
    matches = match_extra_images_to_master(extra_images, master, radius_arcsec=match_radius_arcsec)
    panels = build_recovery_cutout_panels(image_fit, matches)
    extra_panels = build_extra_redshift_cutout_panels(matches)

    matches_path = Path(matches_output) if matches_output is not None else default_matches_output_path(stage_dir)
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(matches_path, index=False)

    band_paths = find_rgb_band_paths(Path(image_dir), cluster=cluster_key, bands=bands, image_scale=image_scale)
    band_images = load_rgb_metadata(band_paths, bands=bands)
    output_path = Path(output) if output is not None else default_output_path(stage_dir)
    write_extra_image_cutout_pdf(
        panels,
        band_images,
        output_path,
        bands=bands,
        cutout_size_arcsec=cutout_size_arcsec,
        images_per_page=images_per_page,
    )
    extras_output_path = Path(extras_output) if extras_output is not None else default_extras_output_path(stage_dir)
    write_extra_image_cutout_pdf(
        extra_panels,
        band_images,
        extras_output_path,
        bands=bands,
        cutout_size_arcsec=cutout_size_arcsec,
        images_per_page=images_per_page,
    )
    return output_path.with_suffix(".pdf"), matches_path, extras_output_path.with_suffix(".pdf")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot WCS cutouts for observed, recovered, missing, and extra model images from a stage/run directory."
    )
    parser.add_argument(
        "stage_dir",
        type=Path,
        help=(
            "Stage/run directory containing tables/image_fit_quality.csv, "
            "tables/image_recovery_extra_images.csv, and artifacts/plot_bundle.h5."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--extras-output", type=Path, default=None)
    parser.add_argument("--matches-output", type=Path, default=None)
    parser.add_argument("--cluster", default=None)
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_HFF_CATALOG_ROOT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--image-scale", choices=IMAGE_SCALE_CHOICES, default=DEFAULT_IMAGE_SCALE)
    parser.add_argument("--bands", nargs=3, default=list(DEFAULT_BANDS), metavar=("BLUE", "GREEN", "RED"))
    parser.add_argument("--match-radius-arcsec", type=float, default=DEFAULT_MATCH_RADIUS_ARCSEC)
    parser.add_argument("--cutout-size-arcsec", type=float, default=DEFAULT_CUTOUT_SIZE_ARCSEC)
    parser.add_argument("--images-per-page", type=int, default=DEFAULT_IMAGES_PER_PAGE)
    parser.add_argument("--par-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output, matches_output, extras_output = run(
            args.stage_dir,
            output=args.output,
            extras_output=args.extras_output,
            matches_output=args.matches_output,
            cluster=args.cluster,
            catalog_root=args.catalog_root,
            image_dir=args.image_dir,
            image_scale=args.image_scale,
            bands=tuple(args.bands),
            match_radius_arcsec=args.match_radius_arcsec,
            cutout_size_arcsec=args.cutout_size_arcsec,
            images_per_page=args.images_per_page,
            par_path=args.par_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {output}")
    print(f"Wrote {extras_output}")
    print(f"Wrote {matches_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
