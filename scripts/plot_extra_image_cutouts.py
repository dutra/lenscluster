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
    extract_band_cutout,
    find_rgb_band_paths,
    load_rgb_metadata,
    make_rgb_cutout,
)


DEFAULT_MATCH_RADIUS_ARCSEC = 0.5
DEFAULT_IMAGES_PER_PAGE = 12
TABLES_DIR_NAME = "tables"
ARTIFACTS_DIR_NAME = "artifacts"
EXTRA_IMAGES_CSV_NAME = "image_recovery_extra_images.csv"
DEFAULT_OUTPUT_NAME = "extra_image_cutouts.pdf"
DEFAULT_MATCHES_NAME = "extra_image_cutout_matches.csv"
REQUIRED_EXTRA_COLUMNS = ("family_id", "extra_image_index", "x_model_arcsec", "y_model_arcsec")
EXTRA_MARKER_COLOR = "#ffdf4d"
MASTER_MARKER_COLOR = "#00d5ff"
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


def default_output_path(stage_dir: Path) -> Path:
    return resolve_stage_dir(stage_dir) / TABLES_DIR_NAME / DEFAULT_OUTPUT_NAME


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


def load_extra_images(extra_images_csv: Path, reference: ReferenceFrame) -> pd.DataFrame:
    data = pd.read_csv(extra_images_csv)
    missing = [column for column in REQUIRED_EXTRA_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{extra_images_csv} is missing required columns: {missing}")
    data = data.copy()
    ra_values: list[float] = []
    dec_values: list[float] = []
    for row in data.itertuples(index=False):
        ra, dec = offsets_to_radec(
            getattr(row, "x_model_arcsec"),
            getattr(row, "y_model_arcsec"),
            ra0_deg=reference.ra0_deg,
            dec0_deg=reference.dec0_deg,
        )
        ra_values.append(ra)
        dec_values.append(dec)
    data["ra"] = ra_values
    data["dec"] = dec_values
    data["extra_label"] = [
        f"{family}.extra{int(index)}"
        for family, index in zip(data["family_id"].astype(str), data["extra_image_index"], strict=True)
    ]
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
    family_text = str(row.get("family_id", ""))
    family_number = int(family_text) if family_text.isdigit() else 10**9
    extra_index = int(_safe_float(row.get("extra_image_index"))) if np.isfinite(_safe_float(row.get("extra_image_index"))) else 10**9
    return family_number, family_text, extra_index, int(row.name)


def _figure_size(rows: int, cols: int) -> tuple[float, float]:
    return CUTOUT_PANEL_SIZE_INCH * float(cols), CUTOUT_PANEL_SIZE_INCH * float(rows)


def write_extra_image_cutout_pdf(
    matches: pd.DataFrame,
    band_images: dict[str, BandImage],
    output: Path,
    *,
    bands: Sequence[str] = DEFAULT_BANDS,
    cutout_size_arcsec: float = DEFAULT_CUTOUT_SIZE_ARCSEC,
    images_per_page: int = DEFAULT_IMAGES_PER_PAGE,
) -> int:
    if matches.empty:
        raise ValueError("No extra images are available for cutout plotting.")
    if images_per_page <= 0:
        raise ValueError("--images-per-page must be positive.")
    output = Path(output).with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    sorted_matches = matches.iloc[sorted(range(len(matches)), key=lambda idx: _sort_key(matches.iloc[idx]))].reset_index(drop=True)
    cols = min(4, int(images_per_page))
    n_pages = int(math.ceil(len(sorted_matches) / int(images_per_page)))
    with PdfPages(output) as pdf:
        for page_index in range(n_pages):
            page = sorted_matches.iloc[page_index * images_per_page : (page_index + 1) * images_per_page].reset_index(drop=True)
            rows = int(math.ceil(len(page) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=_figure_size(rows, cols), squeeze=False)
            _style_cutout_figure(fig)
            fig.patch.set_facecolor(PAGE_FACE_COLOR)
            for panel_index in range(rows * cols):
                row_index = panel_index // cols
                col_index = panel_index % cols
                ax = axes[row_index, col_index]
                _style_cutout_axis(ax)
                ax.set_facecolor(AXIS_FACE_COLOR)
                if panel_index >= len(page):
                    ax.set_axis_off()
                    continue
                image_row = page.iloc[panel_index]
                coord = _coord_from_row(image_row)
                if coord is None:
                    ax.set_axis_off()
                    continue
                cutouts = {
                    str(band): extract_band_cutout(band_images[str(band)], coord, cutout_size_arcsec=cutout_size_arcsec)
                    for band in bands
                }
                rgb = make_rgb_cutout(cutouts, bands=bands)
                ax.imshow(rgb, origin="lower", interpolation="nearest")
                reference_image = band_images[str(bands[-1])]
                _draw_extra_marker(
                    ax,
                    reference_image,
                    coord,
                    cutout_size_arcsec=cutout_size_arcsec,
                    rendered_shape=rgb.shape[:2],
                )
                _draw_master_marker(
                    ax,
                    reference_image,
                    image_row,
                    coord,
                    cutout_size_arcsec=cutout_size_arcsec,
                    rendered_shape=rgb.shape[:2],
                )
                _draw_extra_legend(ax)
                ax.text(
                    0.04,
                    0.94,
                    _panel_label(image_row),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=CUTOUT_LABEL_FONT_SIZE,
                    color="white",
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
) -> tuple[Path, Path]:
    stage_dir = resolve_stage_dir(stage_dir)
    extra_images_csv = extra_images_csv_path(stage_dir)
    if not extra_images_csv.exists():
        raise FileNotFoundError(f"Missing extra-images CSV: {extra_images_csv}")
    if len(bands) != 3:
        raise ValueError("--bands must provide exactly three bands in blue green red order.")
    reference = load_reference_frame(stage_dir, par_path=par_path)
    cluster_key = _canonical_cluster(cluster) if cluster is not None else infer_cluster(stage_dir, reference)
    if cluster_key is None:
        raise ValueError("Could not infer cluster key; pass --cluster.")
    extra_images = load_extra_images(extra_images_csv, reference)
    master = load_master_catalog(Path(catalog_root), cluster_key)
    matches = match_extra_images_to_master(extra_images, master, radius_arcsec=match_radius_arcsec)

    matches_path = Path(matches_output) if matches_output is not None else default_matches_output_path(stage_dir)
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(matches_path, index=False)

    band_paths = find_rgb_band_paths(Path(image_dir), cluster=cluster_key, bands=bands, image_scale=image_scale)
    band_images = load_rgb_metadata(band_paths, bands=bands)
    output_path = Path(output) if output is not None else default_output_path(stage_dir)
    write_extra_image_cutout_pdf(
        matches,
        band_images,
        output_path,
        bands=bands,
        cutout_size_arcsec=cutout_size_arcsec,
        images_per_page=images_per_page,
    )
    return output_path.with_suffix(".pdf"), matches_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot WCS cutouts for extra model images from a stage/run directory."
    )
    parser.add_argument(
        "stage_dir",
        type=Path,
        help="Stage/run directory containing tables/image_recovery_extra_images.csv and artifacts/plot_bundle.h5.",
    )
    parser.add_argument("--output", type=Path, default=None)
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
        output, matches_output = run(
            args.stage_dir,
            output=args.output,
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
    print(f"Wrote {matches_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
