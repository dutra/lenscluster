from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CatalogHeader:
    reference: int
    ra0: float | None = None
    dec0: float | None = None

    def format(self) -> str:
        if self.ra0 is not None and self.dec0 is not None:
            return f"#REFERENCE {self.reference} {self.ra0:.6f} {self.dec0:.6f}"
        return f"#REFERENCE {self.reference}"


@dataclass(frozen=True)
class CatalogData:
    path: Path
    kind: str
    header: CatalogHeader
    data: pd.DataFrame


def _offsets_to_radec(x_arcsec: float, y_arcsec: float, ra0_deg: float, dec0_deg: float) -> tuple[float, float]:
    cos_dec0 = math.cos(math.radians(dec0_deg))
    if abs(cos_dec0) < 1.0e-12:
        raise ValueError("Reference declination is too close to a pole for offset conversion.")
    ra_deg = ra0_deg - float(x_arcsec) / (3600.0 * cos_dec0)
    dec_deg = dec0_deg + float(y_arcsec) / 3600.0
    return ra_deg, dec_deg


def _radec_to_offsets_arcsec(
    ra: np.ndarray,
    dec: np.ndarray,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_arcsec = (ra0_deg - np.asarray(ra, dtype=float)) * math.cos(math.radians(dec0_deg)) * 3600.0
    y_arcsec = (np.asarray(dec, dtype=float) - dec0_deg) * 3600.0
    return x_arcsec, y_arcsec


def _parse_reference_header_tokens(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None
    parts = stripped.split()
    if not parts:
        return None
    if parts[0] == "#REFERENCE":
        return parts
    if len(parts) >= 2 and parts[0] == "#" and parts[1] == "REFERENCE":
        return ["#REFERENCE", *parts[2:]]
    return None


def _parse_reference_header(path: Path) -> tuple[CatalogHeader, list[list[str]]]:
    header: CatalogHeader | None = None
    rows: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = _parse_reference_header_tokens(line)
            if parts is not None:
                if len(parts) not in {2, 4}:
                    raise ValueError(f"Invalid #REFERENCE header in '{path}'.")
                reference = int(parts[1])
                if reference not in {0, 3}:
                    raise ValueError(f"Unsupported #REFERENCE value {reference} in '{path}'.")
                if len(parts) == 4:
                    header = CatalogHeader(reference=reference, ra0=float(parts[2]), dec0=float(parts[3]))
                else:
                    header = CatalogHeader(reference=reference)
                continue
            if line.startswith("#"):
                continue
            rows.append(line.split())
    if header is None:
        raise ValueError(f"Missing #REFERENCE header in '{path}'.")
    if header.reference == 3 and (header.ra0 is None or header.dec0 is None):
        raise ValueError(f"#REFERENCE 3 requires ra0/dec0 in '{path}'.")
    return header, rows


def _load_catalog(path_like: str | Path, kind: str) -> CatalogData:
    path = Path(path_like)
    header, rows = _parse_reference_header(path)
    if kind == "galaxy":
        columns = ["id", "coord_1", "coord_2", "a", "b", "theta", "mag", "lum"]
    elif kind == "image":
        columns = ["id", "coord_1", "coord_2", "a", "b", "theta", "z", "mag"]
    else:
        raise ValueError(f"Unsupported catalog kind '{kind}'.")
    if not rows:
        data = pd.DataFrame(columns=columns + ["ra", "dec"])
        return CatalogData(path=path, kind=kind, header=header, data=data)

    data = pd.DataFrame(rows, columns=columns)
    data["id"] = data["id"].astype(str)
    for column in columns[1:]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if header.reference == 0:
        data["ra"] = data["coord_1"].astype(float)
        data["dec"] = data["coord_2"].astype(float)
    else:
        converted = data.apply(
            lambda row: _offsets_to_radec(row["coord_1"], row["coord_2"], float(header.ra0), float(header.dec0)),
            axis=1,
            result_type="expand",
        )
        data["ra"] = converted[0].astype(float)
        data["dec"] = converted[1].astype(float)

    return CatalogData(path=path, kind=kind, header=header, data=data.reset_index(drop=True))


def _select_reference_center(galaxy_catalog: CatalogData, image_catalog: CatalogData) -> tuple[float, float]:
    for header in (image_catalog.header, galaxy_catalog.header):
        if header.ra0 is not None and header.dec0 is not None:
            return float(header.ra0), float(header.dec0)
    if not image_catalog.data.empty:
        row = image_catalog.data.iloc[0]
        return float(row["ra"]), float(row["dec"])
    if not galaxy_catalog.data.empty:
        row = galaxy_catalog.data.iloc[0]
        return float(row["ra"]), float(row["dec"])
    raise ValueError("Cannot determine a reference center from two empty catalogs.")


def _rank_galaxies(
    galaxy_catalog: CatalogData,
    image_catalog: CatalogData,
    mag0: float,
) -> pd.DataFrame:
    galaxies = galaxy_catalog.data.copy()
    if galaxies.empty:
        return galaxies.assign(
            min_distance_arcsec=pd.Series(dtype=float),
            brightness=pd.Series(dtype=float),
            importance=pd.Series(dtype=float),
            rank=pd.Series(dtype=int),
        )
    if image_catalog.data.empty:
        raise ValueError(f"Image catalog '{image_catalog.path}' contains no rows.")

    ra0_deg, dec0_deg = _select_reference_center(galaxy_catalog, image_catalog)
    x_gal, y_gal = _radec_to_offsets_arcsec(galaxies["ra"].to_numpy(dtype=float), galaxies["dec"].to_numpy(dtype=float), ra0_deg, dec0_deg)
    x_img, y_img = _radec_to_offsets_arcsec(
        image_catalog.data["ra"].to_numpy(dtype=float),
        image_catalog.data["dec"].to_numpy(dtype=float),
        ra0_deg,
        dec0_deg,
    )

    dx = x_gal[:, None] - x_img[None, :]
    dy = y_gal[:, None] - y_img[None, :]
    min_distance = np.min(np.sqrt(dx**2 + dy**2), axis=1)
    magnitudes = galaxies["mag"].to_numpy(dtype=float)
    brightness = np.power(10.0, -0.4 * (magnitudes - float(mag0)))
    importance = brightness / np.square(min_distance + 0.5)
    order = np.argsort(-importance)

    ranked = galaxies.iloc[order].copy().reset_index(drop=True)
    ranked["min_distance_arcsec"] = min_distance[order]
    ranked["brightness"] = brightness[order]
    ranked["importance"] = importance[order]
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked


def _write_galaxy_catalog(path_like: str | Path, header: CatalogHeader, galaxies: pd.DataFrame) -> None:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{header.format()}\n")
        for row in galaxies.itertuples(index=False):
            handle.write(
                f"{str(row.id):>8s} "
                f"{float(row.coord_1):12.6f} "
                f"{float(row.coord_2):11.6f} "
                f"{float(row.a):10.6f} "
                f"{float(row.b):10.6f} "
                f"{float(row.theta):8.2f} "
                f"{float(row.mag):8.4f} "
                f"{float(row.lum):8.4f}\n"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank a galaxy catalog using magnitude brightness and image distance, then write the top N rows."
    )
    parser.add_argument("--galaxy-catalog", required=True, help="Path to the input galaxy .dat catalog.")
    parser.add_argument("--image-catalog", required=True, help="Path to the input image .dat catalog.")
    parser.add_argument("--output-catalog", required=True, help="Path for the filtered output galaxy .dat catalog.")
    parser.add_argument("--top-n", type=int, required=True, help="Number of ranked galaxies to keep.")
    parser.add_argument("--mag0", type=float, required=True, help="Reference magnitude used to convert catalog mag into brightness.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive.")

    galaxy_catalog = _load_catalog(args.galaxy_catalog, kind="galaxy")
    image_catalog = _load_catalog(args.image_catalog, kind="image")
    ranked = _rank_galaxies(
        galaxy_catalog=galaxy_catalog,
        image_catalog=image_catalog,
        mag0=float(args.mag0),
    )

    top_n = min(int(args.top_n), len(ranked))
    selected = ranked.head(top_n).copy().reset_index(drop=True)
    _write_galaxy_catalog(args.output_catalog, galaxy_catalog.header, selected)

    print(f"Loaded {len(galaxy_catalog.data)} galaxies from {galaxy_catalog.path}")
    print(f"Loaded {len(image_catalog.data)} images from {image_catalog.path}")
    print(f"Wrote {len(selected)} ranked galaxies to {Path(args.output_catalog)}")
    if not selected.empty:
        preview = selected.loc[:, ["rank", "id", "importance", "min_distance_arcsec"]].head(10)
        print(preview.to_string(index=False, justify='right'))


if __name__ == "__main__":
    main()
