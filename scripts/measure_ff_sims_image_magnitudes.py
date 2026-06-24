#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lenscluster.image_tools import (
    ImagePhotometryConfig,
    band_photometry_wide_table,
    calibrate_band_zeropoints,
    discover_simulation_bands,
    measure_image_catalog_photometry,
    write_photometric_image_catalog,
)

SUPPORTED_CLUSTERS = ("ares", "hera")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure FF-SIMS multiple-image magnitudes from all available simulation FITS bands."
    )
    parser.add_argument(
        "--cluster",
        choices=(*SUPPORTED_CLUSTERS, "all"),
        default="all",
        help="FF-SIMS cluster to process.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/ff_sims"),
        help="Root containing {cluster}/{cluster}_obs_arcs.cat and published/{cluster}/simulation_*.fits.",
    )
    parser.add_argument("--aperture-radius-arcsec", type=float, default=0.35)
    parser.add_argument("--annulus-inner-radius-arcsec", type=float, default=0.70)
    parser.add_argument("--annulus-outer-radius-arcsec", type=float, default=1.20)
    return parser


def _process_cluster(cluster: str, data_root: Path, config: ImagePhotometryConfig) -> None:
    cluster = cluster.lower()
    image_catalog = data_root / cluster / f"{cluster}_obs_arcs.cat"
    image_dir = data_root / "published" / cluster
    calibration_catalog = image_dir / "clgal_cat.txt"
    output_catalog = data_root / cluster / f"{cluster}_obs_arcs_photutils.cat"
    output_band_table = data_root / cluster / f"{cluster}_obs_arcs_photutils_bands.csv"
    output_band_wide_table = data_root / cluster / f"{cluster}_obs_arcs_photutils_band_magnitudes.csv"
    output_zeropoints = data_root / cluster / f"{cluster}_obs_arcs_photutils_zeropoints.csv"

    bands = discover_simulation_bands(image_dir)
    if not bands:
        raise SystemExit(f"No simulation_*.fits files found in {image_dir}.")
    bands, zeropoints = calibrate_band_zeropoints(bands, calibration_catalog, config)
    catalog, band_table, reference = measure_image_catalog_photometry(image_catalog, bands, config)
    write_photometric_image_catalog(catalog, output_catalog, reference=reference)
    output_band_table.parent.mkdir(parents=True, exist_ok=True)
    band_table.to_csv(output_band_table, index=False)
    band_wide_table = band_photometry_wide_table(catalog, band_table)
    output_band_wide_table.parent.mkdir(parents=True, exist_ok=True)
    band_wide_table.to_csv(output_band_wide_table, index=False)
    output_zeropoints.parent.mkdir(parents=True, exist_ok=True)
    zeropoints.to_csv(output_zeropoints, index=False)

    n_measured = int(catalog["catalog_mag_err"].notna().sum())
    n_catalog_bands = int(zeropoints["use_for_catalog"].sum()) if "use_for_catalog" in zeropoints else 0
    print(
        f"{cluster}: bands={len(bands)} calibrated_catalog_bands={n_catalog_bands} "
        f"images={len(catalog)} measured={n_measured}"
    )
    print(f"{cluster}: wrote {output_catalog}")
    print(f"{cluster}: wrote {output_band_table}")
    print(f"{cluster}: wrote {output_band_wide_table}")
    print(f"{cluster}: wrote {output_zeropoints}")


def main() -> None:
    args = _build_parser().parse_args()
    config = ImagePhotometryConfig(
        aperture_radius_arcsec=float(args.aperture_radius_arcsec),
        annulus_inner_radius_arcsec=float(args.annulus_inner_radius_arcsec),
        annulus_outer_radius_arcsec=float(args.annulus_outer_radius_arcsec),
    )
    clusters = SUPPORTED_CLUSTERS if args.cluster == "all" else (str(args.cluster),)
    for cluster in clusters:
        _process_cluster(cluster, args.data_root, config)


if __name__ == "__main__":
    main()
