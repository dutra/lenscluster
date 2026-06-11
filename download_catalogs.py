from __future__ import annotations

import argparse
import io
import json
import shlex
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ipac.ned import Ned
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table


BUFFALO_BASE_URL = "https://archive.stsci.edu/hlsps/buffalo"
DEFAULT_OUTPUT_DIR = Path("data") / "Pagul2024"
DEFAULT_REDSHIFT_OUTPUT_DIR = Path("data") / "HFF_Redshifts"
DEFAULT_BUFFALO_IMAGE_OUTPUT_DIR = Path("data") / "BUFFALO_Images"
DEFAULT_TIMEOUT_SEC = 60.0
DEFAULT_REDSHIFT_TIMEOUT_SEC = 120.0
DEFAULT_HFF_REDSHIFT_RADIUS_ARCMIN = 5.0
DEFAULT_BUFFALO_IMAGE_SCALE = "60mas"
CHUNK_SIZE = 1024 * 1024
SIMBAD_TAP_URL = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"

PAGUL2024_CLUSTERS = (
    "abell2744",
    "abell370",
    "abells1063",
    "macs0416",
    "macs0717",
    "macs1149",
)
PAGUL2024_SUFFIXES = ("catalog.fits", "readme.txt")
BUFFALO_IMAGE_SCALES = ("60mas", "30mas")


@dataclass(frozen=True)
class DownloadTarget:
    catalog: str
    cluster: str
    filename: str
    url: str
    destination: Path


@dataclass(frozen=True)
class HFFFieldSpec:
    key: str
    target: str
    display_name: str
    z_lens: float
    core_ra: str
    core_dec: str
    parallel_ra: str
    parallel_dec: str


@dataclass(frozen=True)
class RedshiftQueryTarget:
    cluster: HFFFieldSpec
    field: str
    service: str
    center: SkyCoord
    radius_arcmin: float
    destination: Path

    @property
    def label(self) -> str:
        return f"{self.cluster.key}:{self.field}:{self.service}"


HFF_FIELD_SPECS = (
    HFFFieldSpec("a2744", "abell2744", "Abell 2744", 0.308, "00:14:21.2", "-30:23:50.1", "00:13:53.6", "-30:22:54.3"),
    HFFFieldSpec("m0416", "macs0416", "MACS 0416", 0.396, "04:16:08.9", "-24:04:28.7", "04:16:33.1", "-24:06:48.7"),
    HFFFieldSpec("m0717", "macs0717", "MACS 0717", 0.545, "07:17:34.0", "+37:44:49.0", "07:17:17.0", "+37:49:47.3"),
    HFFFieldSpec("m1149", "macs1149", "MACS 1149", 0.543, "11:49:36.3", "+22:23:58.1", "11:49:40.5", "+22:18:02.3"),
    HFFFieldSpec("as1063", "abells1063", "Abell S1063", 0.348, "22:48:44.4", "-44:31:48.5", "22:49:17.7", "-44:32:43.8"),
    HFFFieldSpec("a370", "abell370", "Abell 370", 0.375, "02:39:52.9", "-01:34:36.5", "02:40:13.4", "-01:37:32.8"),
)
REDSHIFT_SERVICES = ("ned", "simbad")
REDSHIFT_SERVICE_CHOICES = ("all", *REDSHIFT_SERVICES)
REDSHIFT_FIELDS = ("core", "parallel")
SIMBAD_REDSHIFT_COLUMNS = ["main_id", "ra", "dec", "otype", "rvz_redshift", "rvz_qual", "rvz_bibcode"]
HFF_CLUSTER_CHOICES = tuple(spec.key for spec in HFF_FIELD_SPECS) + tuple(spec.target for spec in HFF_FIELD_SPECS)
HFF_FIELD_BY_CLUSTER_CHOICE = {
    cluster_choice: spec
    for spec in HFF_FIELD_SPECS
    for cluster_choice in (spec.key, spec.target)
}


def resolve_hff_cluster(cluster: str | None) -> HFFFieldSpec | None:
    if cluster is None:
        return None
    try:
        return HFF_FIELD_BY_CLUSTER_CHOICE[cluster]
    except KeyError as exc:
        raise ValueError(f"Unsupported HFF cluster {cluster!r}. Valid choices: {HFF_CLUSTER_CHOICES}.") from exc


def _selected_hff_field_specs(cluster: str | None) -> tuple[HFFFieldSpec, ...]:
    spec = resolve_hff_cluster(cluster)
    if spec is None:
        return HFF_FIELD_SPECS
    return (spec,)


def _selected_mast_clusters(cluster: str | None) -> tuple[str, ...]:
    spec = resolve_hff_cluster(cluster)
    if spec is None:
        return PAGUL2024_CLUSTERS
    return (spec.target,)


def build_pagul2024_targets(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    cluster: str | None = None,
) -> list[DownloadTarget]:
    """Build direct MAST download targets for the BUFFALO Pagul v2.0 catalogs."""

    root = Path(output_dir)
    targets: list[DownloadTarget] = []
    for mast_cluster in _selected_mast_clusters(cluster):
        product_url_root = f"{BUFFALO_BASE_URL}/{mast_cluster}/catalogs/pagul-v2.0"
        for suffix in PAGUL2024_SUFFIXES:
            filename = f"hlsp_buffalo_hst_ir-weighted_{mast_cluster}_multi_v2.0_{suffix}"
            targets.append(
                DownloadTarget(
                    catalog="pagul2024",
                    cluster=mast_cluster,
                    filename=filename,
                    url=f"{product_url_root}/{filename}",
                    destination=root / filename,
                )
            )
    return targets


def buffalo_image_script_url(cluster: str, image_scale: str) -> str:
    if image_scale not in BUFFALO_IMAGE_SCALES:
        raise ValueError(f"Unsupported BUFFALO image scale {image_scale!r}. Valid choices: {BUFFALO_IMAGE_SCALES}.")
    return (
        f"{BUFFALO_BASE_URL}/download_scripts/"
        f"hlsp_buffalo_hst_multi_{cluster}_multi_v1.0_images-{image_scale}-download.sh"
    )


def fetch_text(url: str, *, timeout: float = DEFAULT_TIMEOUT_SEC) -> str:
    request = Request(url, headers={"User-Agent": "lenscluster-catalog-downloader"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def parse_buffalo_image_script(
    script_text: str,
    *,
    output_dir: str | Path = DEFAULT_BUFFALO_IMAGE_OUTPUT_DIR,
    cluster: str,
) -> list[DownloadTarget]:
    root = Path(output_dir)
    targets: list[DownloadTarget] = []
    for raw_line in script_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("curl "):
            continue
        tokens = shlex.split(line)
        try:
            output_index = tokens.index("--output") + 1
        except ValueError:
            continue
        if output_index >= len(tokens):
            continue
        relative_destination = Path(tokens[output_index])
        if relative_destination.is_absolute() or ".." in relative_destination.parts:
            raise ValueError(f"Unsafe BUFFALO image output path in MAST script: {relative_destination}")
        if not relative_destination.name.endswith("_drz.fits"):
            continue
        urls = [token for token in tokens if token.startswith(("http://", "https://"))]
        if not urls:
            continue
        targets.append(
            DownloadTarget(
                catalog="buffalo-images",
                cluster=cluster,
                filename=relative_destination.name,
                url=urls[-1],
                destination=root / relative_destination,
            )
        )
    return targets


def build_buffalo_image_targets(
    output_dir: str | Path = DEFAULT_BUFFALO_IMAGE_OUTPUT_DIR,
    *,
    image_scale: str = DEFAULT_BUFFALO_IMAGE_SCALE,
    cluster: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> list[DownloadTarget]:
    targets: list[DownloadTarget] = []
    for mast_cluster in _selected_mast_clusters(cluster):
        script_url = buffalo_image_script_url(mast_cluster, image_scale)
        script_text = fetch_text(script_url, timeout=timeout)
        targets.extend(parse_buffalo_image_script(script_text, output_dir=output_dir, cluster=mast_cluster))
    return targets


def _skycoord_from_hmsdms(ra: str, dec: str) -> SkyCoord:
    return SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")


def _field_center(spec: HFFFieldSpec, field: str) -> SkyCoord:
    if field == "core":
        return _skycoord_from_hmsdms(spec.core_ra, spec.core_dec)
    if field == "parallel":
        return _skycoord_from_hmsdms(spec.parallel_ra, spec.parallel_dec)
    raise ValueError(f"Unsupported HFF field {field!r}.")


def resolve_redshift_services(service: str | tuple[str, ...] = "all") -> tuple[str, ...]:
    if isinstance(service, str):
        services = REDSHIFT_SERVICES if service == "all" else (service,)
    else:
        services = service
    invalid = sorted(set(services) - set(REDSHIFT_SERVICES))
    if invalid:
        raise ValueError(f"Unsupported redshift service(s): {invalid}. Valid choices: {REDSHIFT_SERVICE_CHOICES}")
    return tuple(services)


def build_hff_redshift_targets(
    output_dir: str | Path = DEFAULT_REDSHIFT_OUTPUT_DIR,
    *,
    radius_arcmin: float = DEFAULT_HFF_REDSHIFT_RADIUS_ARCMIN,
    services: str | tuple[str, ...] = "all",
    cluster: str | None = None,
) -> list[RedshiftQueryTarget]:
    root = Path(output_dir)
    selected_services = resolve_redshift_services(services)
    targets: list[RedshiftQueryTarget] = []
    for spec in _selected_hff_field_specs(cluster):
        for field in REDSHIFT_FIELDS:
            center = _field_center(spec, field)
            for service in selected_services:
                targets.append(
                    RedshiftQueryTarget(
                        cluster=spec,
                        field=field,
                        service=service,
                        center=center,
                        radius_arcmin=radius_arcmin,
                        destination=root / spec.key / f"{service}_{field}_redshifts.csv",
                    )
                )
    return targets


def remote_content_length(url: str, timeout: float = DEFAULT_TIMEOUT_SEC) -> int | None:
    """Return the remote Content-Length when MAST reports it."""

    request = Request(url, method="HEAD", headers={"User-Agent": "lenscluster-catalog-downloader"})
    with urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
    if length is None:
        return None
    try:
        return int(length)
    except ValueError:
        return None


def existing_file_matches(path: Path, expected_size: int | None) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if expected_size is None:
        return False
    return path.stat().st_size == expected_size


def make_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(style="bright_cyan"),
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None, complete_style="green", finished_style="bright_green", pulse_style="cyan"),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def stream_download(target: DownloadTarget, *, progress: Progress, timeout: float) -> None:
    target.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target.destination.with_suffix(f"{target.destination.suffix}.part")
    request = Request(target.url, headers={"User-Agent": "lenscluster-catalog-downloader"})

    with urlopen(request, timeout=timeout) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        bytes_written = 0
        task_id: TaskID = progress.add_task(
            "download",
            filename=target.filename,
            total=total,
        )
        with temporary_path.open("wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
                progress.update(task_id, advance=len(chunk))
        if total is not None:
            progress.update(task_id, completed=total)

    if total is not None and bytes_written != total:
        raise OSError(f"Incomplete download for {target.filename}: got {bytes_written} of {total} bytes.")

    shutil.move(str(temporary_path), target.destination)


def render_dry_run(targets: list[DownloadTarget], console: Console, *, title: str = "Planned Downloads") -> None:
    table = Table(title=title)
    table.add_column("Cluster", style="cyan", no_wrap=True)
    table.add_column("File", style="green", overflow="fold")
    table.add_column("Destination", style="magenta", overflow="fold")
    table.add_column("URL", style="blue", overflow="fold")
    for target in targets:
        table.add_row(target.cluster, target.filename, str(target.destination), target.url)
    console.print(table)


def make_redshift_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(style="bright_cyan"),
        TextColumn("[bold blue]{task.fields[label]}", justify="right"),
        BarColumn(bar_width=None, complete_style="green", finished_style="bright_green", pulse_style="cyan"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )


def render_redshift_dry_run(targets: list[RedshiftQueryTarget], console: Console) -> None:
    table = Table(title="Planned HFF Redshift Queries")
    table.add_column("Cluster", style="cyan", no_wrap=True)
    table.add_column("Field", style="green", no_wrap=True)
    table.add_column("Service", style="blue", no_wrap=True)
    table.add_column("RA", style="magenta", no_wrap=True)
    table.add_column("Dec", style="magenta", no_wrap=True)
    table.add_column("Radius", style="yellow", no_wrap=True)
    table.add_column("Destination", style="white", overflow="fold")
    for target in targets:
        table.add_row(
            target.cluster.key,
            target.field,
            target.service,
            f"{target.center.ra.deg:.8f}",
            f"{target.center.dec.deg:.8f}",
            f"{target.radius_arcmin:.3f} arcmin",
            str(target.destination),
        )
    console.print(table)


def query_simbad_redshifts(target: RedshiftQueryTarget, *, timeout: float) -> pd.DataFrame:
    query = f"""
SELECT main_id, ra, dec, otype, rvz_redshift, rvz_qual, rvz_bibcode
FROM basic
WHERE rvz_redshift IS NOT NULL
  AND CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {target.center.ra.deg:.10f}, {target.center.dec.deg:.10f}, {target.radius_arcmin / 60.0:.10f})
  ) = 1
"""
    body = urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}).encode("utf-8")
    request = Request(
        SIMBAD_TAP_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "lenscluster-catalog-downloader"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    if "ERROR" in text[:500].upper():
        raise RuntimeError(f"SIMBAD TAP query failed for {target.label}:\n{text[:1000]}")
    if not text.strip():
        return pd.DataFrame(columns=SIMBAD_REDSHIFT_COLUMNS)
    return pd.read_csv(io.StringIO(text), comment="#")


def _decode_dataframe_bytes(df: pd.DataFrame) -> pd.DataFrame:
    decoded = df.copy()
    for column in decoded.columns:
        if decoded[column].dtype == object:
            decoded[column] = decoded[column].map(
                lambda value: value.decode("utf-8", errors="replace")
                if isinstance(value, bytes | bytearray)
                else value
            )
    return decoded


def query_ned_redshifts(target: RedshiftQueryTarget, *, timeout: float) -> pd.DataFrame:
    original_timeout = Ned.TIMEOUT
    Ned.TIMEOUT = timeout
    try:
        table = Ned.query_region(target.center, radius=target.radius_arcmin * u.arcmin)
    finally:
        Ned.TIMEOUT = original_timeout
    df = _decode_dataframe_bytes(table.to_pandas())
    if "Redshift" not in df.columns:
        return df
    redshift = pd.to_numeric(df["Redshift"], errors="coerce")
    return df.loc[redshift.notna()].reset_index(drop=True)


def query_redshift_target(target: RedshiftQueryTarget, *, timeout: float) -> pd.DataFrame:
    if target.service == "simbad":
        return query_simbad_redshifts(target, timeout=timeout)
    if target.service == "ned":
        return query_ned_redshifts(target, timeout=timeout)
    raise ValueError(f"Unsupported redshift service {target.service!r}.")


def _write_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _existing_csv_row_count(path: Path) -> int | None:
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


def _redshift_manifest_record(
    target: RedshiftQueryTarget,
    *,
    status: str,
    row_count: int | None = None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "cluster": target.cluster.key,
        "target": target.cluster.target,
        "display_name": target.cluster.display_name,
        "z_lens": target.cluster.z_lens,
        "field": target.field,
        "service": target.service,
        "ra_deg": float(target.center.ra.deg),
        "dec_deg": float(target.center.dec.deg),
        "radius_arcmin": float(target.radius_arcmin),
        "destination": str(target.destination),
        "status": status,
        "row_count": row_count,
        "error": error,
    }


def write_redshift_manifest(
    output_dir: str | Path,
    *,
    radius_arcmin: float,
    services: tuple[str, ...],
    records: list[dict[str, object]],
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "queried": sum(1 for record in records if record["status"] == "queried"),
        "skipped": sum(1 for record in records if record["status"] == "skipped"),
        "failed": sum(1 for record in records if record["status"] == "failed"),
        "rows": sum(int(record["row_count"] or 0) for record in records),
    }
    manifest = {
        "catalog": "hff-redshifts",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "radius_arcmin": float(radius_arcmin),
        "services": list(services),
        "fields": list(REDSHIFT_FIELDS),
        "summary": summary,
        "queries": records,
    }
    (root / "hff_redshift_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def download_targets(
    targets: list[DownloadTarget],
    *,
    label: str = "Downloads",
    force: bool = False,
    dry_run: bool = False,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    console: Console | None = None,
) -> int:
    console = console or Console()
    if dry_run:
        render_dry_run(targets, console, title=f"Planned {label}")
        return 0

    downloaded = 0
    skipped = 0
    failed = 0

    with make_progress(console) as progress:
        for target in targets:
            try:
                expected_size = remote_content_length(target.url, timeout=timeout)
                if not force and existing_file_matches(target.destination, expected_size):
                    skipped += 1
                    console.print(
                        f"[dim]skip[/dim] [green]{target.filename}[/green] "
                        f"[dim]({target.destination}, {expected_size} bytes)[/dim]"
                    )
                    continue
                stream_download(target, progress=progress, timeout=timeout)
                downloaded += 1
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                failed += 1
                console.print(f"[red]failed[/red] {target.filename}: {exc}")

    console.print(
        f"[bold]{label} complete:[/bold] "
        f"[green]{downloaded} downloaded[/green], "
        f"[cyan]{skipped} skipped[/cyan], "
        f"[red]{failed} failed[/red]"
    )
    return 1 if failed else 0


def download_hff_redshifts(
    *,
    output_dir: str | Path = DEFAULT_REDSHIFT_OUTPUT_DIR,
    radius_arcmin: float = DEFAULT_HFF_REDSHIFT_RADIUS_ARCMIN,
    service: str = "all",
    cluster: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    timeout: float = DEFAULT_REDSHIFT_TIMEOUT_SEC,
    console: Console | None = None,
) -> int:
    console = console or Console()
    selected_services = resolve_redshift_services(service)
    targets = build_hff_redshift_targets(
        output_dir,
        radius_arcmin=radius_arcmin,
        services=selected_services,
        cluster=cluster,
    )
    if dry_run:
        render_redshift_dry_run(targets, console)
        return 0

    records: list[dict[str, object]] = []
    queried = 0
    skipped = 0
    failed = 0

    with make_redshift_progress(console) as progress:
        task_id = progress.add_task("queries", label="hff-redshifts", total=len(targets))
        for target in targets:
            progress.update(task_id, label=target.label)
            if target.destination.exists() and not force:
                skipped += 1
                row_count = _existing_csv_row_count(target.destination)
                records.append(_redshift_manifest_record(target, status="skipped", row_count=row_count))
                console.print(f"[dim]skip[/dim] [green]{target.label}[/green] [dim]({target.destination})[/dim]")
                progress.advance(task_id)
                continue
            try:
                df = query_redshift_target(target, timeout=timeout)
                _write_dataframe_csv(df, target.destination)
                queried += 1
                records.append(_redshift_manifest_record(target, status="queried", row_count=int(len(df))))
            except Exception as exc:
                failed += 1
                records.append(_redshift_manifest_record(target, status="failed", error=str(exc)))
                console.print(f"[red]failed[/red] {target.label}: {exc}")
            finally:
                progress.advance(task_id)

    write_redshift_manifest(output_dir, radius_arcmin=radius_arcmin, services=selected_services, records=records)
    console.print(
        f"[bold]HFF redshift queries complete:[/bold] "
        f"[green]{queried} queried[/green], "
        f"[cyan]{skipped} skipped[/cyan], "
        f"[red]{failed} failed[/red]"
    )
    return 1 if failed else 0


def download_pagul2024(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cluster: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    console: Console | None = None,
) -> int:
    targets = build_pagul2024_targets(output_dir, cluster=cluster)
    return download_targets(
        targets,
        label="Pagul 2024",
        force=force,
        dry_run=dry_run,
        timeout=timeout,
        console=console,
    )


def download_buffalo_images(
    *,
    output_dir: str | Path = DEFAULT_BUFFALO_IMAGE_OUTPUT_DIR,
    image_scale: str = DEFAULT_BUFFALO_IMAGE_SCALE,
    cluster: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    console: Console | None = None,
) -> int:
    targets = build_buffalo_image_targets(output_dir, image_scale=image_scale, cluster=cluster, timeout=timeout)
    return download_targets(
        targets,
        label=f"BUFFALO {image_scale} science images",
        force=force,
        dry_run=dry_run,
        timeout=timeout,
        console=console,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download external catalogs used by lenscluster.")
    parser.add_argument(
        "--catalog",
        choices=("pagul2024", "hff-redshifts", "buffalo-images"),
        default="pagul2024",
        help="Catalog set to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for downloaded catalog files. Defaults depend on --catalog.",
    )
    parser.add_argument(
        "--cluster",
        choices=HFF_CLUSTER_CHOICES,
        default=None,
        help="Limit downloads to one HFF cluster, accepting short keys or MAST target names.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite files even when the size matches MAST.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned downloads without writing files.")
    parser.add_argument(
        "--service",
        choices=REDSHIFT_SERVICE_CHOICES,
        default="all",
        help="Redshift service to query for --catalog hff-redshifts.",
    )
    parser.add_argument(
        "--radius-arcmin",
        type=float,
        default=DEFAULT_HFF_REDSHIFT_RADIUS_ARCMIN,
        help="Cone-search radius for --catalog hff-redshifts.",
    )
    parser.add_argument(
        "--image-scale",
        choices=BUFFALO_IMAGE_SCALES,
        default=DEFAULT_BUFFALO_IMAGE_SCALE,
        help="BUFFALO image mosaic pixel scale for --catalog buffalo-images.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="HTTP timeout in seconds for each request.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.catalog == "pagul2024":
        return download_pagul2024(
            output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
            cluster=args.cluster,
            force=args.force,
            dry_run=args.dry_run,
            timeout=args.timeout if args.timeout is not None else DEFAULT_TIMEOUT_SEC,
        )
    if args.catalog == "hff-redshifts":
        return download_hff_redshifts(
            output_dir=args.output_dir or DEFAULT_REDSHIFT_OUTPUT_DIR,
            radius_arcmin=args.radius_arcmin,
            service=args.service,
            cluster=args.cluster,
            force=args.force,
            dry_run=args.dry_run,
            timeout=args.timeout if args.timeout is not None else DEFAULT_REDSHIFT_TIMEOUT_SEC,
        )
    if args.catalog == "buffalo-images":
        return download_buffalo_images(
            output_dir=args.output_dir or DEFAULT_BUFFALO_IMAGE_OUTPUT_DIR,
            image_scale=args.image_scale,
            cluster=args.cluster,
            force=args.force,
            dry_run=args.dry_run,
            timeout=args.timeout if args.timeout is not None else DEFAULT_TIMEOUT_SEC,
        )
    raise ValueError(f"Unsupported catalog {args.catalog!r}.")


if __name__ == "__main__":
    sys.exit(main())
