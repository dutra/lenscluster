#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-compare-literature")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


DEFAULT_CATALOG_ROOT = Path("results") / "hff_master_catalogs"
DEFAULT_LITERATURE_ROOT = Path("data") / "literature_lenstool_models"
DEFAULT_OUT_DIR = Path("plots") / "compare_literature"
DEFAULT_MATCH_RADIUS_ARCSEC = 1.0
DEFAULT_CLUSTERS = ("a2744", "a370", "as1063", "m0416", "m0717", "m1149")
MANIFEST_NAME = "literature_copy_manifest.csv"


@dataclass(frozen=True)
class LiteratureCatalog:
    cluster: str
    source_slug: str
    source_id: str
    catalog_kind: str
    path: Path
    data: pd.DataFrame
    status: str = "parsed"
    note: str = ""


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _parse_reference_header_tokens(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split()
    if parts[0] == "#REFERENCE":
        return parts
    if len(parts) >= 2 and parts[0] == "#" and parts[1] == "REFERENCE":
        return ["#REFERENCE", *parts[2:]]
    return None


def extract_reference_from_par(path: Path) -> tuple[int, float, float] | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        parts = stripped.split()
        if len(parts) >= 4 and parts[0].lower() == "reference":
            try:
                return int(float(parts[1])), float(parts[2]), float(parts[3])
            except ValueError:
                return None
    return None


def _offsets_to_radec(x_arcsec: float, y_arcsec: float, ra0_deg: float, dec0_deg: float) -> tuple[float, float]:
    cos_dec0 = math.cos(math.radians(dec0_deg))
    if cos_dec0 == 0.0:
        raise ValueError("Reference declination is too close to a pole.")
    return ra0_deg - x_arcsec / (3600.0 * cos_dec0), dec0_deg + y_arcsec / 3600.0


def _split_image_label(label: Any) -> tuple[str, str]:
    text = str(label).strip()
    family_id, separator, image_id = text.partition(".")
    if separator:
        return family_id, image_id
    match = re.match(r"^([A-Za-z]*\d+)([A-Za-z]+)$", text)
    if match:
        return match.group(1), match.group(2)
    return text, ""


def parse_lenstool_catalog(
    path: Path,
    *,
    catalog_kind: str,
    par_reference: tuple[int, float, float] | None = None,
) -> pd.DataFrame:
    header_reference: int | None = None
    header_ra0: float | None = None
    header_dec0: float | None = None
    rows: list[list[str]] = []

    with Path(path).open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            reference_parts = _parse_reference_header_tokens(line)
            if reference_parts is not None:
                if len(reference_parts) < 2:
                    raise ValueError(f"Invalid #REFERENCE header in {path}")
                header_reference = int(float(reference_parts[1]))
                if len(reference_parts) >= 4:
                    header_ra0 = float(reference_parts[2])
                    header_dec0 = float(reference_parts[3])
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                rows.append(parts)

    if not rows:
        return _empty_catalog(catalog_kind)
    if header_reference is None:
        header_reference = 0

    records: list[dict[str, Any]] = []
    for parts in rows:
        min_columns = 4 if catalog_kind == "image" else 8
        if len(parts) < min_columns:
            continue
        coord_1 = _safe_float(parts[1])
        coord_2 = _safe_float(parts[2])
        if not np.isfinite(coord_1) or not np.isfinite(coord_2):
            continue
        if header_reference == 0:
            ra, dec = coord_1, coord_2
        elif header_reference == 3:
            reference = (header_reference, header_ra0, header_dec0) if header_ra0 is not None else par_reference
            if reference is None or reference[1] is None or reference[2] is None:
                raise ValueError(f"Cannot convert #REFERENCE 3 coordinates in {path} without a reference center.")
            _, ra0, dec0 = reference
            ra, dec = _offsets_to_radec(coord_1, coord_2, float(ra0), float(dec0))
        else:
            raise ValueError(f"Unsupported #REFERENCE {header_reference} in {path}")

        if catalog_kind == "image":
            z_value = _safe_float(parts[6]) if len(parts) >= 8 else _safe_float(parts[3])
            mag_value = _safe_float(parts[7]) if len(parts) >= 8 else float("nan")
            quality_value = str(parts[4]).strip() if len(parts) == 5 else ""
            family_id, image_id = _split_image_label(parts[0])
            records.append(
                {
                    "literature_id": str(parts[0]),
                    "family_id": family_id,
                    "image_id": image_id,
                    "ra": ra,
                    "dec": dec,
                    "catalog_a": _safe_float(parts[3]) if len(parts) >= 8 else float("nan"),
                    "catalog_b": _safe_float(parts[4]) if len(parts) >= 8 else float("nan"),
                    "catalog_theta": _safe_float(parts[5]) if len(parts) >= 8 else float("nan"),
                    "catalog_z": z_value,
                    "catalog_mag": mag_value,
                    "catalog_quality": quality_value,
                }
            )
        else:
            records.append(
                {
                    "literature_id": str(parts[0]),
                    "ra": ra,
                    "dec": dec,
                    "catalog_a": _safe_float(parts[3]),
                    "catalog_b": _safe_float(parts[4]),
                    "catalog_theta": _safe_float(parts[5]),
                    "catalog_mag": _safe_float(parts[6]),
                    "catalog_lum": _safe_float(parts[7]),
                }
            )
    return pd.DataFrame(records)


def parse_plain_member_table(path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or set(line) <= {"-"}:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            ra = _safe_float(parts[1])
            dec = _safe_float(parts[2])
            mag = _safe_float(parts[3])
            if not np.isfinite(ra) or not np.isfinite(dec):
                continue
            records.append(
                {
                    "literature_id": str(parts[0]),
                    "ra": ra,
                    "dec": dec,
                    "catalog_mag": mag,
                    "selection": parts[5] if len(parts) > 5 else "",
                }
            )
    return pd.DataFrame(records)


def _empty_catalog(catalog_kind: str) -> pd.DataFrame:
    if catalog_kind == "image":
        return pd.DataFrame(
            columns=["literature_id", "family_id", "image_id", "ra", "dec", "catalog_z", "catalog_mag", "catalog_quality"]
        )
    return pd.DataFrame(columns=["literature_id", "ra", "dec", "catalog_mag"])


def _role_from_filename(path: Path) -> str:
    name = path.name.lower()
    if (
        "obs_arcs" in name
        or name.startswith("mul")
        or name.startswith("img")
        or "_sl-" in name
        or name.endswith("_sl-final.dat")
        or name.endswith("_sl-gold.dat")
    ):
        return "image_catalog"
    if name == "cluster_members_final.txt" or "member" in name or "potfile" in name or name.startswith("cm_"):
        return "member_catalog"
    if "galcat" in name or "galsortcut" in name:
        return "member_catalog"
    if path.suffix.lower() == ".par":
        return "model_par"
    return "catalog"


def _load_literature_manifest(literature_root: Path) -> pd.DataFrame:
    manifest_path = literature_root / MANIFEST_NAME
    if manifest_path.exists():
        return pd.read_csv(manifest_path)
    rows: list[dict[str, Any]] = []
    for path in sorted(literature_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            cluster = path.relative_to(literature_root).parts[0]
            source_slug = path.relative_to(literature_root).parts[1]
        except IndexError:
            continue
        rows.append(
            {
                "cluster": cluster,
                "source_slug": source_slug,
                "file_role": _role_from_filename(path),
                "copied_path": str(path),
                "source_path": "",
                "status": "copied",
            }
        )
    return pd.DataFrame(rows)


def discover_literature_catalogs(literature_root: Path) -> tuple[pd.DataFrame, list[LiteratureCatalog]]:
    literature_root = Path(literature_root)
    manifest = _load_literature_manifest(literature_root)
    if manifest.empty:
        return pd.DataFrame(), []

    copied = manifest.loc[manifest["status"].fillna("") == "copied"].copy()
    par_references: dict[tuple[str, str], tuple[int, float, float]] = {}
    for row in copied.itertuples(index=False):
        path = Path(str(getattr(row, "copied_path", "")))
        if path.suffix.lower() == ".par" and path.exists():
            reference = extract_reference_from_par(path)
            if reference is not None:
                par_references[(str(row.cluster), str(row.source_slug))] = reference

    source_rows: list[dict[str, Any]] = []
    catalogs: list[LiteratureCatalog] = []
    for row in copied.itertuples(index=False):
        role = str(getattr(row, "file_role", ""))
        if role not in {"image_catalog", "member_catalog", "catalog"}:
            continue
        cluster = str(row.cluster)
        source_slug = str(row.source_slug)
        path = Path(str(row.copied_path))
        if not path.exists():
            continue
        source_id = f"{cluster}/{source_slug}/{path.name}"
        catalog_kind = "image" if role == "image_catalog" else "member"
        try:
            if path.name.lower() == "cluster_members_final.txt":
                data = parse_plain_member_table(path)
            else:
                data = parse_lenstool_catalog(
                    path,
                    catalog_kind=catalog_kind,
                    par_reference=par_references.get((cluster, source_slug)),
                )
            status = "parsed" if not data.empty else "empty"
            note = ""
        except Exception as exc:
            data = _empty_catalog(catalog_kind)
            status = "parse_error"
            note = str(exc)
        for column, value in {
            "cluster": cluster,
            "source_slug": source_slug,
            "source_id": source_id,
            "catalog_kind": catalog_kind,
            "source_path": str(getattr(row, "source_path", "")),
            "copied_path": str(path),
            "n_rows": len(data),
            "status": status,
            "note": note,
        }.items():
            if column not in data.columns:
                data[column] = value
        source_rows.append(
            {
                "cluster": cluster,
                "source_slug": source_slug,
                "source_id": source_id,
                "catalog_kind": catalog_kind,
                "copied_path": str(path),
                "n_rows": len(data),
                "status": status,
                "note": note,
            }
        )
        if status == "parsed":
            catalogs.append(LiteratureCatalog(cluster, source_slug, source_id, catalog_kind, path, data, status, note))

    missing = manifest.loc[manifest["status"].fillna("") == "missing_source"]
    for row in missing.itertuples(index=False):
        source_rows.append(
            {
                "cluster": str(row.cluster),
                "source_slug": str(row.source_slug),
                "source_id": f"{row.cluster}/{row.source_slug}",
                "catalog_kind": "missing",
                "copied_path": "",
                "n_rows": 0,
                "status": "missing_source",
                "note": "",
            }
        )
    return pd.DataFrame(source_rows), catalogs


def sky_match_nearest(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
    left_ra: str = "ra",
    left_dec: str = "dec",
    right_ra: str = "ra",
    right_dec: str = "dec",
) -> pd.DataFrame:
    if left.empty or right.empty:
        return pd.DataFrame(columns=["left_index", "right_index", "separation_arcsec"])
    left_coords = SkyCoord(
        ra=pd.to_numeric(left[left_ra], errors="coerce").to_numpy(dtype=float) * u.deg,
        dec=pd.to_numeric(left[left_dec], errors="coerce").to_numpy(dtype=float) * u.deg,
    )
    right_coords = SkyCoord(
        ra=pd.to_numeric(right[right_ra], errors="coerce").to_numpy(dtype=float) * u.deg,
        dec=pd.to_numeric(right[right_dec], errors="coerce").to_numpy(dtype=float) * u.deg,
    )
    left_indices, right_indices, separations, _ = right_coords.search_around_sky(left_coords, radius_arcsec * u.arcsec)
    if len(left_indices) == 0:
        return pd.DataFrame(columns=["left_index", "right_index", "separation_arcsec"])
    candidates = pd.DataFrame(
        {
            "left_index": left_indices.astype(int),
            "right_index": right_indices.astype(int),
            "separation_arcsec": separations.arcsec,
        }
    ).sort_values(["separation_arcsec", "left_index", "right_index"], kind="mergesort")
    used_left: set[int] = set()
    used_right: set[int] = set()
    rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        left_index = int(candidate.left_index)
        right_index = int(candidate.right_index)
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        rows.append(
            {
                "left_index": left_index,
                "right_index": right_index,
                "separation_arcsec": float(candidate.separation_arcsec),
            }
        )
    return pd.DataFrame(rows)


def load_our_members(catalog_root: Path, cluster: str) -> pd.DataFrame:
    path = Path(catalog_root) / cluster / f"{cluster}_cluster_members.csv"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path, low_memory=False)
    if "object_id" not in data.columns:
        data["object_id"] = data.index.map(lambda idx: f"row:{idx}")
    return data


def load_our_family_members(catalog_root: Path, cluster: str) -> pd.DataFrame:
    path = Path(catalog_root) / cluster / f"{cluster}_candidate_family_members.csv"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path, low_memory=False)
    if "object_id" not in data.columns:
        data["object_id"] = data.index.map(lambda idx: f"row:{idx}")
    return data


def _member_match_rows(
    cluster: str,
    source: LiteratureCatalog,
    our: pd.DataFrame,
    *,
    subset_name: str,
    match_radius_arcsec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matches = sky_match_nearest(our, source.data, radius_arcsec=match_radius_arcsec)
    match_by_left = {int(row.left_index): row for row in matches.itertuples(index=False)}
    matched_right = {int(row.right_index) for row in matches.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for idx, row in our.reset_index(drop=True).iterrows():
        match = match_by_left.get(int(idx))
        lit_row = source.data.iloc[int(match.right_index)] if match is not None else None
        rows.append(
            {
                "cluster": cluster,
                "source_id": source.source_id,
                "source_slug": source.source_slug,
                "subset": subset_name,
                "match_type": "our",
                "our_object_id": str(row.get("object_id", "")),
                "literature_id": "" if lit_row is None else str(lit_row.get("literature_id", "")),
                "separation_arcsec": np.nan if match is None else float(match.separation_arcsec),
                "matched": match is not None,
                "our_ra": _safe_float(row.get("ra")),
                "our_dec": _safe_float(row.get("dec")),
                "literature_ra": np.nan if lit_row is None else _safe_float(lit_row.get("ra")),
                "literature_dec": np.nan if lit_row is None else _safe_float(lit_row.get("dec")),
            }
        )
    for idx, lit_row in source.data.reset_index(drop=True).iterrows():
        if int(idx) in matched_right:
            continue
        rows.append(
            {
                "cluster": cluster,
                "source_id": source.source_id,
                "source_slug": source.source_slug,
                "subset": subset_name,
                "match_type": "literature_only",
                "our_object_id": "",
                "literature_id": str(lit_row.get("literature_id", "")),
                "separation_arcsec": np.nan,
                "matched": False,
                "our_ra": np.nan,
                "our_dec": np.nan,
                "literature_ra": _safe_float(lit_row.get("ra")),
                "literature_dec": _safe_float(lit_row.get("dec")),
            }
        )
    summary = {
        f"n_our_{subset_name}": int(len(our)),
        f"n_literature_members_{subset_name}": int(len(source.data)),
        f"n_our_{subset_name}_matched": int(len(matches)),
        f"n_literature_members_{subset_name}_matched": int(len(matched_right)),
        f"our_{subset_name}_matched_fraction": float(len(matches) / len(our)) if len(our) else np.nan,
        f"literature_member_recovery_fraction_{subset_name}": float(len(matched_right) / len(source.data)) if len(source.data) else np.nan,
    }
    return rows, summary


def compare_member_source(
    cluster: str,
    source: LiteratureCatalog,
    our_members: pd.DataFrame,
    *,
    match_radius_arcsec: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_rows, all_summary = _member_match_rows(
        cluster,
        source,
        our_members,
        subset_name="members",
        match_radius_arcsec=match_radius_arcsec,
    )
    lensing = our_members.loc[our_members.get("member_for_lensing", pd.Series(False, index=our_members.index)).map(_bool_value)].copy()
    lensing_rows, lensing_summary = _member_match_rows(
        cluster,
        source,
        lensing,
        subset_name="lensing",
        match_radius_arcsec=match_radius_arcsec,
    )
    summary = {**all_summary, **lensing_summary}
    return pd.DataFrame([*all_rows, *lensing_rows]), summary


def compare_image_source(
    cluster: str,
    source: LiteratureCatalog,
    our_images: pd.DataFrame,
    *,
    match_radius_arcsec: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    matches = sky_match_nearest(our_images, source.data, radius_arcsec=match_radius_arcsec)
    match_by_left = {int(row.left_index): row for row in matches.itertuples(index=False)}
    matched_right = {int(row.right_index) for row in matches.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for idx, row in our_images.reset_index(drop=True).iterrows():
        match = match_by_left.get(int(idx))
        lit_row = source.data.iloc[int(match.right_index)] if match is not None else None
        rows.append(
            {
                "cluster": cluster,
                "source_id": source.source_id,
                "source_slug": source.source_slug,
                "match_type": "our",
                "our_object_id": str(row.get("object_id", "")),
                "our_family_id": str(row.get("candidate_family_id", "")),
                "literature_id": "" if lit_row is None else str(lit_row.get("literature_id", "")),
                "literature_family_id": "" if lit_row is None else str(lit_row.get("family_id", "")),
                "separation_arcsec": np.nan if match is None else float(match.separation_arcsec),
                "matched": match is not None,
                "our_ra": _safe_float(row.get("ra")),
                "our_dec": _safe_float(row.get("dec")),
                "literature_ra": np.nan if lit_row is None else _safe_float(lit_row.get("ra")),
                "literature_dec": np.nan if lit_row is None else _safe_float(lit_row.get("dec")),
            }
        )
    for idx, lit_row in source.data.reset_index(drop=True).iterrows():
        if int(idx) in matched_right:
            continue
        rows.append(
            {
                "cluster": cluster,
                "source_id": source.source_id,
                "source_slug": source.source_slug,
                "match_type": "literature_only",
                "our_object_id": "",
                "our_family_id": "",
                "literature_id": str(lit_row.get("literature_id", "")),
                "literature_family_id": str(lit_row.get("family_id", "")),
                "separation_arcsec": np.nan,
                "matched": False,
                "our_ra": np.nan,
                "our_dec": np.nan,
                "literature_ra": _safe_float(lit_row.get("ra")),
                "literature_dec": _safe_float(lit_row.get("dec")),
            }
        )
    image_matches = pd.DataFrame(rows)
    family_overlap = family_overlap_from_image_matches(image_matches)
    summary = {
        "n_our_images": int(len(our_images)),
        "n_literature_images": int(len(source.data)),
        "n_our_images_matched": int(len(matches)),
        "n_literature_images_matched": int(len(matched_right)),
        "our_image_matched_fraction": float(len(matches) / len(our_images)) if len(our_images) else np.nan,
        "literature_image_recovery_fraction": float(len(matched_right) / len(source.data)) if len(source.data) else np.nan,
        "n_our_families": int(our_images["candidate_family_id"].nunique()) if "candidate_family_id" in our_images.columns else 0,
        "n_literature_families": int(source.data["family_id"].nunique()) if "family_id" in source.data.columns else 0,
        "n_agreed_families": int((family_overlap.get("agreement", pd.Series(dtype=str)) == "agreed").sum()) if not family_overlap.empty else 0,
    }
    summary["family_agreement_fraction"] = (
        float(summary["n_agreed_families"] / summary["n_our_families"]) if summary["n_our_families"] else np.nan
    )
    return image_matches, family_overlap, summary


def family_overlap_from_image_matches(image_matches: pd.DataFrame) -> pd.DataFrame:
    if image_matches.empty:
        return pd.DataFrame(
            columns=[
                "cluster",
                "source_id",
                "source_slug",
                "our_family_id",
                "n_our_images",
                "n_matched_images",
                "dominant_literature_family_id",
                "dominant_match_count",
                "dominant_fraction",
                "agreement",
            ]
        )
    our_rows = image_matches.loc[image_matches["match_type"] == "our"].copy()
    rows: list[dict[str, Any]] = []
    for (cluster, source_id, source_slug, our_family_id), group in our_rows.groupby(
        ["cluster", "source_id", "source_slug", "our_family_id"], dropna=False
    ):
        matched = group.loc[group["matched"].map(_bool_value) & group["literature_family_id"].astype(str).ne("")]
        counts = matched["literature_family_id"].astype(str).value_counts()
        if counts.empty:
            dominant_id = ""
            dominant_count = 0
            dominant_fraction = 0.0
            agreement = "unmatched"
        else:
            dominant_id = str(counts.index[0])
            dominant_count = int(counts.iloc[0])
            dominant_fraction = float(dominant_count / len(matched))
            agreement = "agreed" if dominant_count >= 2 and dominant_fraction >= 0.67 else "partial"
        rows.append(
            {
                "cluster": cluster,
                "source_id": source_id,
                "source_slug": source_slug,
                "our_family_id": our_family_id,
                "n_our_images": int(len(group)),
                "n_matched_images": int(len(matched)),
                "dominant_literature_family_id": dominant_id,
                "dominant_match_count": dominant_count,
                "dominant_fraction": dominant_fraction,
                "agreement": agreement,
            }
        )
    return pd.DataFrame(rows)


def compare_all(
    catalog_root: Path = DEFAULT_CATALOG_ROOT,
    literature_root: Path = DEFAULT_LITERATURE_ROOT,
    *,
    match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
    clusters: Iterable[str] = DEFAULT_CLUSTERS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources_df, catalogs = discover_literature_catalogs(literature_root)
    requested_clusters = tuple(clusters)
    member_rows: list[pd.DataFrame] = []
    image_rows: list[pd.DataFrame] = []
    family_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for cluster in requested_clusters:
        cluster_catalogs = [catalog for catalog in catalogs if catalog.cluster == cluster]
        our_members = load_our_members(catalog_root, cluster)
        our_images = load_our_family_members(catalog_root, cluster)
        if not cluster_catalogs:
            summary_rows.append(
                {
                    "cluster": cluster,
                    "source_id": "",
                    "source_slug": "",
                    "catalog_kind": "missing",
                    "status": "missing_literature_source",
                    "match_radius_arcsec": match_radius_arcsec,
                }
            )
            continue

        for source in cluster_catalogs:
            row = {
                "cluster": cluster,
                "source_id": source.source_id,
                "source_slug": source.source_slug,
                "catalog_kind": source.catalog_kind,
                "status": source.status,
                "match_radius_arcsec": match_radius_arcsec,
            }
            if source.catalog_kind == "member":
                matches, summary = compare_member_source(cluster, source, our_members, match_radius_arcsec=match_radius_arcsec)
                member_rows.append(matches)
                row.update(summary)
            elif source.catalog_kind == "image":
                matches, family_overlap, summary = compare_image_source(
                    cluster,
                    source,
                    our_images,
                    match_radius_arcsec=match_radius_arcsec,
                )
                image_rows.append(matches)
                family_rows.append(family_overlap)
                row.update(summary)
            summary_rows.append(row)

    member_matches = pd.concat(member_rows, ignore_index=True) if member_rows else pd.DataFrame()
    image_matches = pd.concat(image_rows, ignore_index=True) if image_rows else pd.DataFrame()
    family_overlap = pd.concat(family_rows, ignore_index=True) if family_rows else pd.DataFrame()
    cluster_summary = pd.DataFrame(summary_rows)
    return sources_df, member_matches, image_matches, family_overlap, cluster_summary


def _write_scatter_overlay(path: Path, title: str, layers: list[tuple[pd.DataFrame, str, str, str]]) -> None:
    if not any(not frame.empty for frame, *_ in layers):
        return
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for frame, label, color, marker in layers:
        if frame.empty:
            continue
        ax.scatter(
            pd.to_numeric(frame["ra"], errors="coerce"),
            pd.to_numeric(frame["dec"], errors="coerce"),
            s=16,
            alpha=0.65,
            label=label,
            color=color,
            marker=marker,
            linewidths=0.5,
        )
    ax.invert_xaxis()
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_plots(
    out_dir: Path,
    catalog_root: Path,
    catalogs: list[LiteratureCatalog],
    member_matches: pd.DataFrame,
    image_matches: pd.DataFrame,
    family_overlap: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> None:
    out_dir = Path(out_dir)
    for source in catalogs:
        cluster_dir = out_dir / source.cluster
        safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.source_id)
        if source.catalog_kind == "member":
            our_members = load_our_members(catalog_root, source.cluster)
            lensing = our_members.loc[our_members.get("member_for_lensing", pd.Series(False, index=our_members.index)).map(_bool_value)].copy()
            _write_scatter_overlay(
                cluster_dir / f"{safe_source}_members_overlay.png",
                f"{source.cluster} members vs {source.source_slug}",
                [
                    (source.data, "literature", "#777777", "."),
                    (our_members, "our members", "#1f77b4", "o"),
                    (lensing, "our lensing", "#d62728", "x"),
                ],
            )
        elif source.catalog_kind == "image":
            our_images = load_our_family_members(catalog_root, source.cluster)
            _write_scatter_overlay(
                cluster_dir / f"{safe_source}_images_overlay.png",
                f"{source.cluster} images vs {source.source_slug}",
                [
                    (source.data, "literature images", "#777777", "."),
                    (our_images, "our images", "#2ca02c", "o"),
                ],
            )
            overlap = family_overlap.loc[family_overlap.get("source_id", pd.Series(dtype=str)) == source.source_id]
            _write_family_heatmap(cluster_dir / f"{safe_source}_family_overlap.png", overlap, image_matches, source.source_id)
    _write_match_histograms(out_dir, member_matches, image_matches)
    _write_global_summary(out_dir / "global_literature_agreement_summary.png", cluster_summary)


def _write_family_heatmap(path: Path, overlap: pd.DataFrame, image_matches: pd.DataFrame, source_id: str) -> None:
    matched = image_matches.loc[
        (image_matches.get("source_id", pd.Series(dtype=str)) == source_id)
        & image_matches.get("matched", pd.Series(dtype=bool)).map(_bool_value)
        & image_matches.get("our_family_id", pd.Series(dtype=str)).astype(str).ne("")
        & image_matches.get("literature_family_id", pd.Series(dtype=str)).astype(str).ne("")
    ]
    if matched.empty:
        return
    table = pd.crosstab(matched["our_family_id"].astype(str), matched["literature_family_id"].astype(str))
    if table.empty:
        return
    table = table.iloc[:40, :40]
    fig, ax = plt.subplots(figsize=(max(6, 0.25 * len(table.columns)), max(5, 0.25 * len(table.index))), constrained_layout=True)
    image = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_xticklabels(table.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(table.index, fontsize=6)
    ax.set_xlabel("Literature family")
    ax.set_ylabel("Our family")
    ax.set_title("Matched image family overlap")
    fig.colorbar(image, ax=ax, label="matched images")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_match_histograms(out_dir: Path, member_matches: pd.DataFrame, image_matches: pd.DataFrame) -> None:
    frames = []
    if not member_matches.empty:
        temp = member_matches.loc[member_matches["matched"].map(_bool_value) & member_matches["match_type"].eq("our")].copy()
        temp["kind"] = "members"
        frames.append(temp)
    if not image_matches.empty:
        temp = image_matches.loc[image_matches["matched"].map(_bool_value) & image_matches["match_type"].eq("our")].copy()
        temp["kind"] = "images"
        frames.append(temp)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for kind, group in combined.groupby("kind"):
        ax.hist(pd.to_numeric(group["separation_arcsec"], errors="coerce").dropna(), bins=np.linspace(0, DEFAULT_MATCH_RADIUS_ARCSEC, 20), alpha=0.6, label=kind)
    ax.set_xlabel("Match separation [arcsec]")
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "match_separation_histograms.png", dpi=180)
    plt.close(fig)


def _write_global_summary(path: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    metric_columns = [
        "our_members_matched_fraction",
        "our_lensing_matched_fraction",
        "our_image_matched_fraction",
        "family_agreement_fraction",
    ]
    available = [column for column in metric_columns if column in summary.columns]
    if not available:
        return
    plot_data = summary[["cluster", "source_slug", *available]].copy()
    plot_data["label"] = plot_data["cluster"].astype(str) + "/" + plot_data["source_slug"].fillna("").astype(str)
    plot_data = plot_data.dropna(subset=available, how="all")
    if plot_data.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(plot_data)), 5), constrained_layout=True)
    x = np.arange(len(plot_data))
    width = 0.8 / max(1, len(available))
    for idx, column in enumerate(available):
        ax.bar(x + (idx - (len(available) - 1) / 2) * width, pd.to_numeric(plot_data[column], errors="coerce"), width=width, label=column)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data["label"], rotation=90, fontsize=7)
    ax.legend(loc="best", fontsize="small")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_outputs(
    out_dir: Path,
    sources: pd.DataFrame,
    member_matches: pd.DataFrame,
    image_matches: pd.DataFrame,
    family_overlap: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    catalog_root: Path,
    catalogs: list[LiteratureCatalog],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources.to_csv(out_dir / "literature_sources.csv", index=False)
    member_matches.to_csv(out_dir / "member_matches.csv", index=False)
    image_matches.to_csv(out_dir / "image_matches.csv", index=False)
    family_overlap.to_csv(out_dir / "family_overlap.csv", index=False)
    cluster_summary.to_csv(out_dir / "cluster_summary.csv", index=False)
    write_plots(out_dir, catalog_root, catalogs, member_matches, image_matches, family_overlap, cluster_summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--literature-root", type=Path, default=DEFAULT_LITERATURE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--match-radius-arcsec", type=float, default=DEFAULT_MATCH_RADIUS_ARCSEC)
    parser.add_argument("--clusters", default=",".join(DEFAULT_CLUSTERS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    clusters = tuple(item.strip() for item in str(args.clusters).split(",") if item.strip())
    sources, member_matches, image_matches, family_overlap, cluster_summary = compare_all(
        args.catalog_root,
        args.literature_root,
        match_radius_arcsec=args.match_radius_arcsec,
        clusters=clusters,
    )
    _, catalogs = discover_literature_catalogs(args.literature_root)
    write_outputs(args.out_dir, sources, member_matches, image_matches, family_overlap, cluster_summary, args.catalog_root, catalogs)
    print(f"Wrote comparison outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
