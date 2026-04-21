from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


CHIRES_COLUMNS = (
    "index",
    "family_id",
    "z",
    "n_arcs",
    "chi_total",
    "chi_x",
    "chi_y",
    "chi_a",
    "source_rms_arcsec",
    "image_rms_arcsec",
    "dx_arcsec",
    "dy_arcsec",
    "n_warn",
)


def _parse_chires_float(value: str) -> float | None:
    if value.upper() == "N/A":
        return None
    return float(value)


def load_chires_table(path: str | Path) -> pd.DataFrame:
    """Load a Lenstool ``chires.dat`` table.

    The file includes one row per image plus one summary row per family. Numeric
    ``N/A`` cells are returned as missing values by pandas.
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("chi ") or line.startswith("N "):
                continue
            parts = line.split()
            if len(parts) != len(CHIRES_COLUMNS):
                continue
            row: dict[str, Any] = {
                "index": int(parts[0]),
                "family_id": parts[1],
                "z": float(parts[2]),
                "n_arcs": int(parts[3]),
                "n_warn": int(parts[12]),
            }
            for column, raw_value in zip(CHIRES_COLUMNS[4:12], parts[4:12]):
                row[column] = _parse_chires_float(raw_value)
            rows.append(row)
    return pd.DataFrame(rows, columns=CHIRES_COLUMNS)


def load_chires_family_summary(path: str | Path) -> pd.DataFrame:
    """Return only family-summary rows from a Lenstool ``chires.dat`` table."""
    table = load_chires_table(path)
    if table.empty:
        return table
    summary = table[table["n_arcs"] > 1].copy()
    return summary.sort_values(["index", "family_id"]).reset_index(drop=True)
