#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SOURCE_ROOT = Path("/data/lenstool_models")
DEFAULT_OUTPUT_ROOT = Path("data") / "literature_lenstool_models"
MANIFEST_NAME = "literature_copy_manifest.csv"

EXCLUDED_SUFFIXES = {
    ".fits",
    ".fit",
    ".fz",
    ".jpg",
    ".jpeg",
    ".png",
    ".pdf",
    ".tgz",
    ".gz",
    ".zip",
    ".docx",
    ".npy",
    ".npz",
    ".pkl",
}
EXCLUDED_FILENAMES = {
    ".ds_store",
    "bayes.dat",
    "burnin.dat",
    "chires.dat",
    "robots.txt",
}
PROVENANCE_SUFFIXES = {".txt", ".html", ""}


@dataclass(frozen=True)
class LiteratureSource:
    cluster: str
    source_slug: str
    relative_root: str
    include_globs: tuple[str, ...]


DEFAULT_SOURCES: tuple[LiteratureSource, ...] = (
    LiteratureSource(
        "a2744",
        "bergamini23",
        "Bergamini/A2744_Bergamini23",
        ("*.par", "*README*", "README", "CM_A2744_mag21.cat", "obs_arcs.cat"),
    ),
    LiteratureSource(
        "a2744",
        "johan_richard_muse",
        "johan_richard_MUSE/models/A2744",
        ("*.par", "mul*.cat", "img*.cat", "potfile*.txt", "new_potfile*.txt", "*README*", "README"),
    ),
    LiteratureSource(
        "a370",
        "niemiec_buffalo",
        "A370_BUFFALO_Niemiec_Catalog/abell370/catalogs/niemiec-lensing-dr1",
        ("*readme*.txt", "*sl-final.dat", "*sl-gold.dat", "*galcat-full.dat", "*galcat-redseq.cat"),
    ),
    LiteratureSource(
        "a370",
        "johan_richard_muse",
        "johan_richard_MUSE/models/A370",
        ("*.par", "mul*.cat", "galsortcut*.cat", "*README*", "README"),
    ),
    LiteratureSource(
        "as1063",
        "beauchesne23_no_bspline",
        "AS1063_Beauchesne23/Model_no_bspline",
        ("*.par", "mul.cat", "potfile*.txt", "*README*", "README"),
    ),
    LiteratureSource(
        "as1063",
        "beauchesne23_bspline_4x4",
        "AS1063_Beauchesne23/Model_bspline_4x4",
        ("*.par", "mul.cat", "potfile*.txt", "*README*", "README"),
    ),
    LiteratureSource(
        "as1063",
        "beauchesne23",
        "AS1063_Beauchesne23",
        ("*README*", "README"),
    ),
    LiteratureSource(
        "as1063",
        "johan_richard_muse",
        "johan_richard_MUSE/models/AS1063",
        ("*.par", "mul*.cat", "*README*", "README"),
    ),
    LiteratureSource(
        "m0416",
        "bergamini22",
        "Bergamini/M0416_Bergamini22",
        ("*.par", "*README*", "README", "CM_cat_MACSJ0416.cat", "obs_arcs.cat"),
    ),
    LiteratureSource(
        "m0416",
        "johan_richard_muse",
        "johan_richard_MUSE/models/MACS0416",
        ("*.par", "mul*.cat", "*README*", "README"),
    ),
    LiteratureSource(
        "m1149",
        "schuldt24",
        "Bergamini/M1149_Schuldt2024",
        ("Cluster_members_final.txt", "*README*", "README"),
    ),
    LiteratureSource(
        "m1149",
        "barry_frontier_model",
        "Lenstool_files Barry/MACS1149",
        ("*.par", "*params*.txt", "*readme*.txt", "*README*", "README"),
    ),
)


def _normalize_name(path: Path) -> str:
    return path.name.lower()


def classify_file_role(path: Path) -> str:
    name = _normalize_name(path)
    suffix = path.suffix.lower()
    if suffix == ".par":
        return "model_par"
    if "readme" in name or name == "readme" or suffix == ".html":
        return "provenance"
    if name in {"cluster_members_final.txt"} or "member" in name or "potfile" in name or name.startswith("cm_"):
        return "member_catalog"
    if "galcat" in name or "galsortcut" in name:
        return "member_catalog"
    if (
        "obs_arcs" in name
        or name.startswith("mul")
        or name.startswith("img")
        or "_sl-" in name
        or name.endswith("_sl-final.dat")
        or name.endswith("_sl-gold.dat")
    ):
        return "image_catalog"
    if suffix in {".cat", ".dat"}:
        return "catalog"
    return "provenance" if suffix in PROVENANCE_SUFFIXES else "other"


def should_copy_file(path: Path) -> bool:
    name = _normalize_name(path)
    if name in EXCLUDED_FILENAMES:
        return False
    if any(part == "www.fe.infn.it" for part in path.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    role = classify_file_role(path)
    return role in {"image_catalog", "member_catalog", "model_par", "provenance", "catalog"}


def _glob_source_files(root: Path, include_globs: Iterable[str]) -> list[Path]:
    files: dict[Path, None] = {}
    for pattern in include_globs:
        for path in root.glob(pattern):
            if path.is_file() and should_copy_file(path):
                files[path.resolve()] = None
    paths = sorted(files)
    provenance = [path for path in paths if classify_file_role(path) == "provenance"]
    has_non_html_provenance = any(path.suffix.lower() != ".html" for path in provenance)
    if has_non_html_provenance:
        paths = [path for path in paths if path.suffix.lower() != ".html"]
    return paths


def stage_literature_catalogs(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    sources: Iterable[LiteratureSource] = DEFAULT_SOURCES,
) -> pd.DataFrame:
    source_root = Path(source_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    copied_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    seen_destinations: set[Path] = set()

    for source in sources:
        source_dir = source_root / source.relative_root
        destination_dir = output_root / source.cluster / source.source_slug
        if not source_dir.exists():
            rows.append(
                {
                    "cluster": source.cluster,
                    "source_slug": source.source_slug,
                    "file_role": "missing_source",
                    "source_path": str(source_dir),
                    "copied_path": "",
                    "size_bytes": 0,
                    "copied_at": copied_at,
                    "status": "missing_source",
                }
            )
            continue

        files = _glob_source_files(source_dir, source.include_globs)
        if not files:
            rows.append(
                {
                    "cluster": source.cluster,
                    "source_slug": source.source_slug,
                    "file_role": "no_matching_files",
                    "source_path": str(source_dir),
                    "copied_path": "",
                    "size_bytes": 0,
                    "copied_at": copied_at,
                    "status": "no_matching_files",
                }
            )
            continue

        destination_dir.mkdir(parents=True, exist_ok=True)
        for path in files:
            relative_path = path.relative_to(source_dir.resolve())
            destination = destination_dir / relative_path
            if destination in seen_destinations:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            seen_destinations.add(destination)
            rows.append(
                {
                    "cluster": source.cluster,
                    "source_slug": source.source_slug,
                    "file_role": classify_file_role(path),
                    "source_path": str(path),
                    "copied_path": str(destination),
                    "size_bytes": int(destination.stat().st_size),
                    "copied_at": copied_at,
                    "status": "copied",
                }
            )

    if not any(row["cluster"] == "m0717" for row in rows):
        rows.append(
            {
                "cluster": "m0717",
                "source_slug": "no_local_literature_catalog",
                "file_role": "missing_source",
                "source_path": "",
                "copied_path": "",
                "size_bytes": 0,
                "copied_at": copied_at,
                "status": "missing_source",
            }
        )

    manifest = pd.DataFrame(
        rows,
        columns=["cluster", "source_slug", "file_role", "source_path", "copied_path", "size_bytes", "copied_at", "status"],
    )
    manifest.to_csv(output_root / MANIFEST_NAME, index=False)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = stage_literature_catalogs(args.source_root, args.output_root)
    copied = int((manifest["status"] == "copied").sum()) if not manifest.empty else 0
    total_bytes = int(pd.to_numeric(manifest.get("size_bytes", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
    print(f"Copied {copied} files ({total_bytes / 1024.0 / 1024.0:.2f} MiB) to {args.output_root}")
    print(f"Wrote {args.output_root / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
