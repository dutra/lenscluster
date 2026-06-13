"""Command-line interface: batch (DS9 regions) and interactive measurement."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .arcfile import write_arcfile, write_sidetable
from .config import ArcMeasureConfig, SegmentationConfig
from .errors import ArcfileWriteError, ArctraceError, RegionParseError
from .measure import ArcMeasurement, measure_arc
from .mosaics import load_band_mosaics
from .regions import normalize_arc_id, parse_ds9_regions

LOGGER = logging.getLogger("arctrace")


def _parse_band_path(value: str) -> tuple[str, Path]:
    band, sep, path = value.partition("=")
    if not sep or not band.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"--image expects BAND=PATH, got {value!r}")
    return band.strip().upper(), Path(path.strip())


def _parse_psf_override(value: str) -> tuple[str, float]:
    band, sep, fwhm = value.partition("=")
    if not sep:
        try:
            return "*", float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"--psf-fwhm expects BAND=FWHM or a bare float, got {value!r}") from exc
    try:
        return band.strip().upper(), float(fwhm)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--psf-fwhm expects BAND=FWHM, got {value!r}") from exc


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--image",
        action="append",
        type=_parse_band_path,
        required=True,
        metavar="BAND=PATH",
        help="Band mosaic to load (repeatable), e.g. F814W=/data/.../..._f814w_v1.0_drz.fits",
    )
    parser.add_argument("--reference-band", default="F814W", help="Band whose measurement feeds the arcfile.")
    parser.add_argument(
        "--measure-bands",
        default="",
        help="Comma-separated bands to measure for the achromaticity cross-check (default: reference only).",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arcfile-name", default="arcfile.cat")
    parser.add_argument("--cutout-arcsec", type=float, default=20.0)
    parser.add_argument("--max-cutout-arcsec", type=float, default=40.0)
    parser.add_argument("--threshold-sigma", type=float, default=1.2)
    parser.add_argument("--max-gap-arcsec", type=float, default=0.5)
    parser.add_argument("--max-area-arcsec2", type=float, default=80.0)
    parser.add_argument(
        "--psf-fwhm",
        action="append",
        type=_parse_psf_override,
        default=None,
        metavar="BAND=FWHM",
        help="PSF FWHM override in arcsec (repeatable; a bare float applies to all bands).",
    )
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--rng-seed", type=int, default=1234)
    parser.add_argument("--curvature-sigma-floor", type=float, default=0.005)
    parser.add_argument("--sigma-e", type=float, default=0.25, help="Intrinsic ellipticity dispersion for the tangent floor.")
    parser.add_argument(
        "--fit-halfspan-arcsec",
        type=float,
        default=None,
        help="Restrict the circle fit to ridge points within this arclength of the anchor (long arcs).",
    )
    parser.add_argument("--refine", choices=("none", "forward"), default="none")
    parser.add_argument("--reliability", type=float, default=None, help="Override the reliability for all arcs.")
    parser.add_argument("--no-qa", action="store_true", help="Skip QA PNG rendering.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arctrace", description=__doc__)
    parser.add_argument("--version", action="version", version=f"arctrace {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure_parser = subparsers.add_parser("measure", help="Batch-measure arcs from a DS9 region file.")
    _add_common_arguments(measure_parser)
    measure_parser.add_argument("--regions", required=True, type=Path, help="DS9 .reg file with seed points.")
    labeling = measure_parser.add_mutually_exclusive_group()
    labeling.add_argument("--skip-unlabeled", action="store_true", help="Skip seeds without a text={...} label.")
    labeling.add_argument(
        "--fail-unlabeled",
        action="store_true",
        help="Abort when a seed lacks a label (default behavior).",
    )

    interactive_parser = subparsers.add_parser("interactive", help="Interactive matplotlib clicking session.")
    _add_common_arguments(interactive_parser)
    interactive_parser.add_argument("--center", default=None, help='Field center "RA DEC" in degrees (default: mosaic center).')
    interactive_parser.add_argument("--view-arcsec", type=float, default=120.0)
    interactive_parser.add_argument(
        "--backend",
        default=None,
        help="Explicit matplotlib GUI backend, e.g. TkAgg or QtAgg (default: auto-select).",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ArcMeasureConfig:
    psf_overrides = dict(args.psf_fwhm) if args.psf_fwhm else {}
    measure_bands = tuple(b.strip().upper() for b in str(args.measure_bands).split(",") if b.strip())
    segmentation = SegmentationConfig(
        threshold_sigma=float(args.threshold_sigma),
        max_bridge_gap_arcsec=float(args.max_gap_arcsec),
        max_area_arcsec2=float(args.max_area_arcsec2),
    )
    return ArcMeasureConfig(
        cutout_size_arcsec=float(args.cutout_arcsec),
        max_cutout_size_arcsec=float(args.max_cutout_arcsec),
        psf_fwhm_arcsec=psf_overrides,
        segmentation=segmentation,
        reference_band=str(args.reference_band).upper(),
        measure_bands=measure_bands,
        n_bootstrap=int(args.n_bootstrap),
        sigma_e_floor=float(args.sigma_e),
        curvature_sigma_floor_arcsec_inv=float(args.curvature_sigma_floor),
        fit_halfspan_arcsec=(float(args.fit_halfspan_arcsec) if args.fit_halfspan_arcsec else None),
        refine=str(args.refine),
        rng_seed=int(args.rng_seed),
    )


def _resolve_labels(seeds, *, skip_unlabeled: bool) -> list[tuple]:
    resolved = []
    used: dict[str, int] = {}
    for seed in seeds:
        raw = seed.label_raw
        if raw is None:
            if skip_unlabeled:
                LOGGER.warning("Skipping unlabeled seed at line %d.", seed.line_number)
                continue
            raise RegionParseError(
                f"Seed at region line {seed.line_number} has no text={{...}} label; "
                "label it or pass --skip-unlabeled."
            )
        try:
            label = normalize_arc_id(raw)
        except ValueError:
            raise RegionParseError(f"Seed label {raw!r} (line {seed.line_number}) is not a valid arc ID.") from None
        used[label] = used.get(label, 0) + 1
        resolved.append((seed, label))
    duplicates = sorted(label for label, count in used.items() if count > 1)
    if duplicates:
        raise RegionParseError(f"Duplicate labels in the region file: {duplicates}.")
    return resolved


def _write_run_json(args: argparse.Namespace, cfg: ArcMeasureConfig, output_dir: Path) -> None:
    payload = {
        "arctrace_version": __version__,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "images": {band: str(path) for band, path in args.image},
        "config": json.loads(json.dumps(dataclasses.asdict(cfg), default=str)),
    }
    (output_dir / "arctrace_run.json").write_text(json.dumps(payload, indent=2))


def run_measure(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arcfile_path = output_dir / str(args.arcfile_name)
    if arcfile_path.exists() and not args.overwrite:
        LOGGER.error("%s exists; pass --overwrite to replace it.", arcfile_path)
        return 2

    seeds = parse_ds9_regions(args.regions)
    labeled = _resolve_labels(seeds, skip_unlabeled=bool(args.skip_unlabeled))
    LOGGER.info("Measuring %d seeds from %s.", len(labeled), args.regions)

    mosaics = load_band_mosaics({band: path for band, path in args.image})
    if cfg.reference_band not in mosaics:
        LOGGER.error("Reference band %s not among loaded images %s.", cfg.reference_band, sorted(mosaics))
        return 2

    measurements: list[ArcMeasurement] = []
    rng = np.random.default_rng(cfg.rng_seed)
    for seed, label in labeled:
        LOGGER.info("Measuring %s at %.6f %.6f ...", label, seed.ra_deg, seed.dec_deg)
        measurement, artifacts = measure_arc(
            mosaics,
            seed.coord,
            cfg,
            label=label,
            rng=rng,
            reliability_override=args.reliability,
            return_artifacts=True,
        )
        measurements.append(measurement)
        if not measurement.success:
            LOGGER.warning("  %s FAILED: %s", label, measurement.failure_reason)
            continue
        LOGGER.info(
            "  %s: phi=%.4f+-%.4f rad  kappa=%.4f+-%.4f /arcsec  rel=%.2f",
            label,
            measurement.tangent_angle_offset_rad,
            measurement.sigma_tangent_rad,
            measurement.curvature_arcsec_inv,
            measurement.sigma_curvature_arcsec_inv,
            measurement.reliability,
        )
        for warning in measurement.warnings:
            LOGGER.warning("  %s: %s", label, warning)
        if not args.no_qa:
            reference_artifacts = artifacts.get(measurement.reference_band)
            if reference_artifacts is not None:
                from .qa import extract_display_cutouts, save_qa_png

                display = extract_display_cutouts(
                    mosaics, seed.coord, reference_artifacts.cutout.data.shape[0] * reference_artifacts.cutout.pixel_scale_arcsec
                )
                safe_label = label.replace(".", "_")
                save_qa_png(measurement, reference_artifacts, output_dir / "qa" / f"{safe_label}.png", display)

    write_sidetable(measurements, output_dir / "arctrace_sidetable.csv", output_dir / "arctrace_sidetable.json")
    _write_run_json(args, cfg, output_dir)
    successes = [m for m in measurements if m.success]
    if successes:
        try:
            write_arcfile(measurements, arcfile_path, overwrite=bool(args.overwrite))
            LOGGER.info("Wrote %d arc rows to %s.", len(successes), arcfile_path)
        except ArcfileWriteError as exc:
            LOGGER.error("Arcfile not written: %s", exc)
            return 1
    else:
        LOGGER.warning("No successful measurements; arcfile not written.")
    LOGGER.info("Side table and run metadata written to %s.", output_dir)
    return 0


def run_interactive(args: argparse.Namespace) -> int:
    from .interactive import run_interactive_session

    cfg = config_from_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mosaics = load_band_mosaics({band: path for band, path in args.image})
    if cfg.reference_band not in mosaics:
        LOGGER.error("Reference band %s not among loaded images %s.", cfg.reference_band, sorted(mosaics))
        return 2
    center = None
    if args.center:
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        tokens = str(args.center).replace(",", " ").split()
        if len(tokens) != 2:
            LOGGER.error('--center expects "RA DEC".')
            return 2
        if ":" in tokens[0]:
            center = SkyCoord(ra=tokens[0], dec=tokens[1], unit=(u.hourangle, u.deg))
        else:
            center = SkyCoord(ra=float(tokens[0]) * u.deg, dec=float(tokens[1]) * u.deg)
    measurements = run_interactive_session(
        mosaics,
        cfg,
        center=center,
        view_size_arcsec=float(args.view_arcsec),
        output_dir=output_dir,
        reliability_override=args.reliability,
        arcfile_name=str(args.arcfile_name),
        overwrite=bool(args.overwrite),
        no_qa=bool(args.no_qa),
        backend=args.backend,
    )
    return 0 if measurements is not None else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        if args.command == "measure":
            import matplotlib

            matplotlib.use("Agg")
            return run_measure(args)
        if args.command == "interactive":
            return run_interactive(args)
    except (ArctraceError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 2
