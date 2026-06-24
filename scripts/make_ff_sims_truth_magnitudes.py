#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lenscluster.image_tools import TruthMagnitudeConfig, build_truth_magnitude_catalog


SUPPORTED_CLUSTERS = ("ares", "hera")
CLUSTER_COSMOLOGY = {
    "ares": {"h0": 70.4, "om0": 0.272, "z_lens": 0.5},
    "hera": {"h0": 70.4, "om0": 0.272, "z_lens": 0.5},
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate FF-SIMS ideal multi-band image magnitudes from true magnification maps."
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
        help="Root containing {cluster}/{cluster}_obs_arcs.cat and published/{cluster}/kappa/shear FITS maps.",
    )
    parser.add_argument("--random-seed", type=int, default=12_345)
    parser.add_argument("--source-mag-f160w-mean", type=float, default=27.0)
    parser.add_argument("--source-mag-f160w-sigma", type=float, default=1.0)
    parser.add_argument("--color-scatter-sigma", type=float, default=0.15)
    parser.add_argument("--magnitude-error", type=float, default=0.01)
    parser.add_argument("--mu-floor", type=float, default=1.0e-3)
    return parser


def _process_cluster(cluster: str, data_root: Path, args: argparse.Namespace) -> None:
    cluster = cluster.lower()
    cosmo = CLUSTER_COSMOLOGY[cluster]
    image_catalog = data_root / cluster / f"{cluster}_obs_arcs.cat"
    map_dir = data_root / "published" / cluster
    output_catalog = data_root / cluster / f"{cluster}_obs_arcs_truthmag.cat"
    output_band_table = data_root / cluster / f"{cluster}_obs_arcs_truthmag_band_magnitudes.csv"
    config = TruthMagnitudeConfig(
        z_lens=float(cosmo["z_lens"]),
        h0=float(cosmo["h0"]),
        om0=float(cosmo["om0"]),
        random_seed=int(args.random_seed),
        source_mag_f160w_mean=float(args.source_mag_f160w_mean),
        source_mag_f160w_sigma=float(args.source_mag_f160w_sigma),
        color_scatter_sigma=float(args.color_scatter_sigma),
        magnitude_error=float(args.magnitude_error),
        mu_floor=float(args.mu_floor),
    )
    catalog, band_table, _reference = build_truth_magnitude_catalog(
        image_catalog,
        map_dir,
        output_catalog,
        output_band_table,
        config=config,
    )
    print(
        f"{cluster}: images={len(catalog)} bands=7 finite_magnitudes="
        f"{int(band_table.filter(regex='^mag_').notna().all(axis=1).sum())}"
    )
    print(f"{cluster}: wrote {output_catalog}")
    print(f"{cluster}: wrote {output_band_table}")


def main() -> None:
    args = _build_parser().parse_args()
    clusters = SUPPORTED_CLUSTERS if args.cluster == "all" else (str(args.cluster),)
    for cluster in clusters:
        _process_cluster(cluster, args.data_root, args)


if __name__ == "__main__":
    main()
