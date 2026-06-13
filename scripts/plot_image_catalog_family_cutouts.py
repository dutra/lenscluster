#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-image-catalog-family-cutouts")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_hff_to_literature import (  # noqa: E402
    LiteratureCatalog,
    extract_reference_from_par,
    parse_lenstool_catalog,
)
from plot_literature_family_cutouts import (  # noqa: E402
    DEFAULT_BANDS,
    DEFAULT_CUTOUT_SIZE_ARCSEC,
    DEFAULT_FAMILIES_PER_PAGE,
    DEFAULT_IMAGE_DIR,
    DEFAULT_IMAGE_SCALE,
    DEFAULT_MAX_IMAGES_PER_FAMILY,
    IMAGE_SCALE_CHOICES,
    find_rgb_band_paths,
    load_rgb_metadata,
    write_family_cutout_pdf,
)


DEFAULT_OUTPUT_DIR = Path("results") / "image_catalog_family_cutouts"
DEFAULT_SOURCE_SLUG = "direct_image_catalog"
CLUSTER_ALIASES = {
    "a2744": "a2744",
    "a2744_cats": "a2744",
    "abell2744": "a2744",
    "abell_2744": "a2744",
    "a307": "a370",
    "a370": "a370",
    "abell370": "a370",
    "abell_370": "a370",
    "ares": "ares",
    "as1063": "as1063",
    "abells1063": "as1063",
    "abell_s1063": "as1063",
    "rxcj2248": "as1063",
    "rxcj_2248": "as1063",
    "hera": "hera",
    "m0416": "m0416",
    "macs0416": "m0416",
    "macs_0416": "m0416",
    "m0717": "m0717",
    "macs0717": "m0717",
    "macs_0717": "m0717",
    "m1149": "m1149",
    "macs1149": "m1149",
    "macs_1149": "m1149",
    "m1206": "m1206",
    "macsj1206": "m1206",
    "macs_j1206": "m1206",
}


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "plot"


def canonical_cluster(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    compact = normalized.replace("_", "")
    return CLUSTER_ALIASES.get(normalized, CLUSTER_ALIASES.get(compact, normalized))


def default_output_path(cluster: str, image_catalog_path: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{_safe_filename(cluster)}_{_safe_filename(Path(image_catalog_path).stem)}_family_cutouts.pdf"


def load_image_catalog(
    cluster: str,
    image_catalog_path: Path,
    *,
    par_path: Path | None = None,
) -> LiteratureCatalog:
    path = Path(image_catalog_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing image catalog: {path}")

    par_reference = None
    if par_path is not None:
        par_path = Path(par_path)
        if not par_path.is_file():
            raise FileNotFoundError(f"Missing Lenstool .par file: {par_path}")
        par_reference = extract_reference_from_par(par_path)
        if par_reference is None:
            raise ValueError(f"Could not find a reference line in Lenstool .par file: {par_path}")

    data = parse_lenstool_catalog(path, catalog_kind="image", par_reference=par_reference)
    if data.empty:
        raise ValueError(f"Image catalog has no parsed images: {path}")

    cluster_key = canonical_cluster(cluster)
    source_id = f"{cluster_key}/{DEFAULT_SOURCE_SLUG}/{path.name}"
    data = data.copy()
    for column, value in {
        "cluster": cluster_key,
        "source_slug": DEFAULT_SOURCE_SLUG,
        "source_id": source_id,
        "catalog_kind": "image",
        "source_path": str(path),
        "copied_path": str(path),
        "n_rows": len(data),
        "status": "parsed",
        "note": "",
    }.items():
        if column not in data.columns:
            data[column] = value

    return LiteratureCatalog(
        cluster=cluster_key,
        source_slug=DEFAULT_SOURCE_SLUG,
        source_id=source_id,
        catalog_kind="image",
        path=path,
        data=data,
    )


def run(
    cluster: str,
    image_catalog_path: Path,
    *,
    output: Path | None = None,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    image_scale: str = DEFAULT_IMAGE_SCALE,
    bands: Sequence[str] = DEFAULT_BANDS,
    cutout_size_arcsec: float = DEFAULT_CUTOUT_SIZE_ARCSEC,
    families_per_page: int = DEFAULT_FAMILIES_PER_PAGE,
    max_images_per_family: int = DEFAULT_MAX_IMAGES_PER_FAMILY,
    par_path: Path | None = None,
) -> Path:
    if len(bands) < 3:
        raise ValueError("--bands must provide at least three bands.")
    cluster_key = canonical_cluster(cluster)
    catalog = load_image_catalog(cluster_key, Path(image_catalog_path), par_path=par_path)
    band_paths = find_rgb_band_paths(Path(image_dir), cluster=cluster_key, bands=bands, image_scale=image_scale)
    band_images = load_rgb_metadata(band_paths, bands=bands)
    output_path = Path(output) if output is not None else default_output_path(cluster_key, Path(image_catalog_path))
    write_family_cutout_pdf(
        catalog,
        band_images,
        output_path,
        bands=bands,
        cutout_size_arcsec=cutout_size_arcsec,
        families_per_page=families_per_page,
        max_images_per_family=max_images_per_family,
    )
    return output_path.with_suffix(".pdf")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot WCS RGB cutouts for a direct multiple-image catalog.")
    parser.add_argument("cluster", help="Cluster name or alias used to select BUFFALO RGB FITS images.")
    parser.add_argument("image_catalog_path", type=Path, help="Lenstool-style multiple-image catalog path.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--image-scale", choices=IMAGE_SCALE_CHOICES, default=DEFAULT_IMAGE_SCALE)
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS), metavar="BAND")
    parser.add_argument("--cutout-size-arcsec", type=float, default=DEFAULT_CUTOUT_SIZE_ARCSEC)
    parser.add_argument("--families-per-page", type=int, default=DEFAULT_FAMILIES_PER_PAGE)
    parser.add_argument("--max-images-per-family", type=int, default=DEFAULT_MAX_IMAGES_PER_FAMILY)
    parser.add_argument("--par-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = run(
            args.cluster,
            args.image_catalog_path,
            output=args.output,
            image_dir=args.image_dir,
            image_scale=args.image_scale,
            bands=tuple(args.bands),
            cutout_size_arcsec=args.cutout_size_arcsec,
            families_per_page=args.families_per_page,
            max_images_per_family=args.max_images_per_family,
            par_path=args.par_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
