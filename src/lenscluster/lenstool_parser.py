from __future__ import annotations

from pathlib import Path
from typing import Any

import math

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, LambdaCDM, wCDM


DP_IE_PROFILE = 81
SHEAR_PROFILE = 14


def _coerce_token(token: str) -> Any:
    text = token.strip()
    if text == "":
        return ""

    try:
        value = float(text)
    except ValueError:
        return text

    if math.isfinite(value) and value.is_integer() and all(ch not in text.lower() for ch in ".e"):
        return int(value)
    return value


def _coerce_values(tokens: list[str]) -> Any:
    values = [_coerce_token(token) for token in tokens]
    if len(values) == 1:
        return values[0]
    return values


def _pluralize(name: str) -> str:
    if name.endswith("s"):
        return name
    if name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
        return f"{name[:-1]}ies"
    return f"{name}s"


def _is_missing_or_zero(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) == 0.0
    return False


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _normalize_block_id(block_id: Any) -> str | None:
    if not isinstance(block_id, str):
        return None

    normalized = block_id.split("#", 1)[0].strip()
    if not normalized:
        return None
    return normalized


def _parse_named_block(lines: list[str], start_index: int) -> tuple[dict[str, Any], int]:
    header_tokens = lines[start_index].split()
    block_name = header_tokens[0]
    block: dict[str, Any] = {}

    if len(header_tokens) > 1:
        block["id"] = " ".join(header_tokens[1:])

    index = start_index + 1
    while index < len(lines):
        raw_line = lines[index].strip()
        if raw_line == "end":
            return {"name": block_name, "data": block}, index + 1

        parts = raw_line.split()
        key = parts[0]
        block[key] = _coerce_values(parts[1:])
        index += 1

    raise ValueError(f"Block '{block_name}' starting on line {start_index + 1} is missing 'end'.")


def _extract_reference(parsed: dict[str, Any]) -> tuple[int, float, float] | None:
    runmode = parsed.get("runmode")
    if not isinstance(runmode, dict):
        return None

    reference = runmode.get("reference")
    if not isinstance(reference, list) or len(reference) < 3:
        return None

    try:
        return int(reference[0]), float(reference[1]), float(reference[2])
    except (TypeError, ValueError):
        return None


def _offsets_to_radec(x_arcsec: Any, y_arcsec: Any, ra0_deg: float, dec0_deg: float) -> tuple[float, float]:
    x_value = float(x_arcsec)
    y_value = float(y_arcsec)
    cos_dec0 = math.cos(math.radians(dec0_deg))
    if cos_dec0 == 0.0:
        raise ValueError("Reference declination is too close to a pole for offset conversion.")

    ra_deg = ra0_deg - x_value / (3600.0 * cos_dec0)
    dec_deg = dec0_deg + y_value / 3600.0
    return ra_deg, dec_deg


def _fallback_radec_to_offsets(
    ra: Any,
    dec: Any,
    ra0: float,
    dec0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ra_values = np.asarray(ra, dtype=float)
    dec_values = np.asarray(dec, dtype=float)
    x_arcsec = (ra0 - ra_values) * np.cos(np.radians(dec0)) * 3600.0
    y_arcsec = (dec_values - dec0) * 3600.0
    nan_values = np.full_like(x_arcsec, np.nan, dtype=float)
    return nan_values, nan_values.copy(), x_arcsec, y_arcsec


def _compute_offsets(
    ra: Any,
    dec: Any,
    ra0: float,
    dec0: float,
    z: Any,
    cosmo: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from .utils import radec_to_offsets as utils_radec_to_offsets
    except Exception:
        utils_radec_to_offsets = None

    if utils_radec_to_offsets is not None and cosmo is not None:
        z_value = 0.0 if z is None else z
        return utils_radec_to_offsets(ra, dec, ra0, dec0, z_value, cosmo)

    return _fallback_radec_to_offsets(ra, dec, ra0, dec0)


def _build_cosmology(parsed: dict[str, Any]):
    cosmo_block = parsed.get("cosmology")
    if not isinstance(cosmo_block, dict):
        cosmo_block = parsed.get("cosmologie")
    if not isinstance(cosmo_block, dict):
        return FlatLambdaCDM(H0=70.0, Om0=0.3)
    h0 = float(cosmo_block.get("H0", 70.0))
    om0 = float(cosmo_block.get("omegaM", cosmo_block.get("omega", 0.3)))
    ode0 = float(cosmo_block.get("omegaX", cosmo_block.get("lambda", 0.7)))
    ok0 = float(cosmo_block.get("omegaK", 0.0))
    w0 = float(cosmo_block.get("wX", cosmo_block.get("w", -1.0)))
    wa = float(cosmo_block.get("wa", 0.0))
    if abs(ok0) < 1.0e-10:
        if abs(w0 + 1.0) < 1.0e-10 and abs(wa) < 1.0e-10:
            return FlatLambdaCDM(H0=h0, Om0=om0)
        return FlatwCDM(H0=h0, Om0=om0, w0=w0)
    if abs(w0 + 1.0) < 1.0e-10 and abs(wa) < 1.0e-10:
        return LambdaCDM(H0=h0, Om0=om0, Ode0=ode0)
    return wCDM(H0=h0, Om0=om0, Ode0=ode0, w0=w0)


def _kpc_per_arcsec(cosmo: Any, z_lens: float) -> float:
    return float(cosmo.kpc_proper_per_arcmin(z_lens).to("kpc/arcsec").value)


def _scale_prior_values(values: Any, scale: float) -> Any:
    if isinstance(values, list):
        if not values:
            return []
        return [values[0], *[float(item) * scale for item in values[1:]]]
    return float(values) * scale


def _require_numeric_field(potential: dict[str, Any], field_name: str, potential_id: str) -> float:
    value = _safe_float(potential.get(field_name))
    if value is None:
        raise ValueError(f"Potential {potential_id} is missing required field '{field_name}'.")
    return float(value)


def _normalize_potential_definition(
    potential: dict[str, Any],
    priors: dict[str, Any],
    cosmo: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = dict(potential)
    normalized_priors = dict(priors)
    potential_id = str(normalized.get("id", "potential"))
    if "profil" not in normalized and "profile" in normalized:
        normalized["profil"] = normalized.pop("profile")
    if "profil" not in normalized_priors and "profile" in normalized_priors:
        normalized_priors["profil"] = normalized_priors.pop("profile")
    profile_type = int(normalized.get("profil"))

    if profile_type == DP_IE_PROFILE:
        if "ellipticite" not in normalized and "ellipticity" in normalized:
            normalized["ellipticite"] = normalized.pop("ellipticity")
        if "ellipticite" not in normalized_priors and "ellipticity" in normalized_priors:
            normalized_priors["ellipticite"] = normalized_priors.pop("ellipticity")
        for field_name in ("x_centre", "y_centre", "ellipticite", "angle_pos", "v_disp", "z_lens"):
            _require_numeric_field(normalized, field_name, potential_id)
        z_lens = _require_numeric_field(normalized, "z_lens", potential_id)
        scale = _kpc_per_arcsec(cosmo, z_lens)

        if _safe_float(normalized.get("core_radius_kpc")) is None:
            core_arcsec = _safe_float(normalized.get("core_radius"))
            if core_arcsec is None:
                raise ValueError(
                    f"Potential {potential_id} is missing required field 'core_radius' or 'core_radius_kpc'."
                )
            normalized["core_radius_kpc"] = float(core_arcsec) * scale
        if _safe_float(normalized.get("cut_radius_kpc")) is None:
            cut_arcsec = _safe_float(normalized.get("cut_radius"))
            if cut_arcsec is None:
                raise ValueError(
                    f"Potential {potential_id} is missing required field 'cut_radius' or 'cut_radius_kpc'."
                )
            normalized["cut_radius_kpc"] = float(cut_arcsec) * scale

        if "core_radius_kpc" not in normalized_priors and "core_radius" in normalized_priors:
            normalized_priors["core_radius_kpc"] = _scale_prior_values(normalized_priors.pop("core_radius"), scale)
        if "cut_radius_kpc" not in normalized_priors and "cut_radius" in normalized_priors:
            normalized_priors["cut_radius_kpc"] = _scale_prior_values(normalized_priors.pop("cut_radius"), scale)
    elif profile_type == SHEAR_PROFILE:
        for field_name in ("gamma", "angle_pos"):
            _require_numeric_field(normalized, field_name, potential_id)

    return normalized, normalized_priors


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


def _load_dat_catalog(
    filepath: str | Path,
    par_reference: tuple[int, float, float] | None,
    catalog_kind: str = "potfile_galaxies",
) -> pd.DataFrame:
    path = Path(filepath)
    header_reference: int | None = None
    header_ra0: float | None = None
    header_dec0: float | None = None
    rows: list[list[str]] = []

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = _parse_reference_header_tokens(line)
            if parts is not None:
                if len(parts) < 2:
                    raise ValueError(f"Invalid #REFERENCE header in '{path}'.")
                header_reference = int(parts[1])
                if header_reference == 0 and len(parts) >= 4:
                    header_ra0 = float(parts[2])
                    header_dec0 = float(parts[3])
                continue
            if line.startswith("#"):
                continue
            rows.append(line.split())

    if header_reference is None:
        raise ValueError(f"Missing #REFERENCE header in '{path}'.")

    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "catalog_reference",
                "catalog_source",
                "ra",
                "dec",
                "catalog_a",
                "catalog_b",
                "catalog_theta",
                "catalog_mag",
                "catalog_z",
                "catalog_lum",
            ]
        )

    if catalog_kind == "multiple_images":
        df = pd.DataFrame(rows, columns=["id", "coord_1", "coord_2", "a", "b", "theta", "z", "mag"])
        df["lum"] = np.nan
    elif catalog_kind == "potfile_galaxies":
        df = pd.DataFrame(rows, columns=["id", "coord_1", "coord_2", "a", "b", "theta", "mag", "lum"])
        df["z"] = np.nan
    else:
        raise ValueError(f"Unsupported catalog kind '{catalog_kind}' for '{path}'.")
    df["id"] = df["id"].astype(str)
    for column in ["coord_1", "coord_2", "a", "b", "theta", "mag", "z", "lum"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if header_reference == 0:
        df["ra"] = df["coord_1"]
        df["dec"] = df["coord_2"]
    elif header_reference == 3:
        if par_reference is None:
            raise ValueError(
                f"Cannot interpret #REFERENCE 3 coordinates in '{path}' without a runmode.reference in the .par file."
            )
        _, ra0_deg, dec0_deg = par_reference
        converted = df.apply(
            lambda row: _offsets_to_radec(row["coord_1"], row["coord_2"], ra0_deg, dec0_deg),
            axis=1,
            result_type="expand",
        )
        df["ra"] = converted[0]
        df["dec"] = converted[1]
    else:
        raise ValueError(f"Unsupported #REFERENCE value {header_reference} in '{path}'.")

    df["catalog_reference"] = header_reference
    df["catalog_source"] = str(path)
    df = df.rename(
        columns={
            "a": "catalog_a",
            "b": "catalog_b",
            "theta": "catalog_theta",
            "mag": "catalog_mag",
            "z": "catalog_z",
            "lum": "catalog_lum",
        }
    )

    keep_columns = [
        "id",
        "catalog_reference",
        "catalog_source",
        "ra",
        "dec",
        "catalog_a",
        "catalog_b",
        "catalog_theta",
        "catalog_mag",
        "catalog_z",
        "catalog_lum",
    ]
    return df.loc[:, keep_columns].drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)


def _empty_images_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "image_label",
            "family_id",
            "image_id",
            "catalog_reference",
            "catalog_source",
            "ra",
            "dec",
            "catalog_a",
            "catalog_b",
            "catalog_theta",
            "catalog_mag",
            "catalog_z",
            "catalog_lum",
        ]
    )


def _split_image_label(image_label: str) -> tuple[str, str]:
    label = image_label.strip()
    family_id, separator, image_id = label.partition(".")
    if not separator:
        return label, ""
    return family_id, image_id


def _load_multiple_images_catalog(
    filepath: str | Path,
    par_reference: tuple[int, float, float] | None,
) -> pd.DataFrame:
    catalog_df = _load_dat_catalog(filepath, par_reference, catalog_kind="multiple_images")
    if catalog_df.empty:
        return _empty_images_df()

    image_labels = catalog_df["id"].astype(str)
    family_and_image = image_labels.apply(_split_image_label)

    result = pd.DataFrame(
        {
            "image_label": image_labels,
            "family_id": family_and_image.str[0],
            "image_id": family_and_image.str[1],
            "catalog_reference": catalog_df["catalog_reference"].to_numpy(),
            "catalog_source": catalog_df["catalog_source"].to_numpy(),
            "ra": catalog_df["ra"].to_numpy(),
            "dec": catalog_df["dec"].to_numpy(),
            "catalog_a": catalog_df["catalog_a"].to_numpy(),
            "catalog_b": catalog_df["catalog_b"].to_numpy(),
            "catalog_theta": catalog_df["catalog_theta"].to_numpy(),
            "catalog_mag": catalog_df["catalog_mag"].to_numpy(),
            "catalog_z": catalog_df["catalog_z"].to_numpy(),
            "catalog_lum": catalog_df["catalog_lum"].to_numpy(),
        }
    )
    return result.reset_index(drop=True)


def _resolve_catalog_path(path_value: Any, base_dir: Path) -> Path | None:
    if not isinstance(path_value, str):
        return None

    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _extract_image_catalog_paths(parsed: dict[str, Any], base_dir: Path) -> list[Path]:
    discovered_paths: list[Path] = []
    seen_paths: set[Path] = set()

    def add_candidate(candidate: Any) -> None:
        resolved = _resolve_catalog_path(candidate, base_dir)
        if resolved is None or resolved in seen_paths:
            return
        seen_paths.add(resolved)
        discovered_paths.append(resolved)

    runmode = parsed.get("runmode")
    if isinstance(runmode, dict):
        image_value = runmode.get("image")
        if isinstance(image_value, list) and len(image_value) >= 2:
            add_candidate(image_value[1])
        elif isinstance(image_value, str):
            add_candidate(image_value)

    image_block = parsed.get("image")
    if isinstance(image_block, dict):
        multfile_value = image_block.get("multfile")
        if isinstance(multfile_value, list) and len(multfile_value) >= 2:
            add_candidate(multfile_value[1])
        elif isinstance(multfile_value, str):
            add_candidate(multfile_value)

    return discovered_paths


def _resolve_potfile_catalog_path(filein_value: Any, base_dir: Path) -> Path | None:
    if isinstance(filein_value, list) and len(filein_value) >= 2:
        candidate = filein_value[1]
    else:
        candidate = filein_value
    return _resolve_catalog_path(candidate, base_dir)


def _potfile_nominal_value(value: Any, context: str) -> float:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Empty potfile value for {context}.")
        flag = int(value[0])
        if flag == 0 and len(value) >= 2:
            return float(value[1])
        if flag == 1 and len(value) >= 3:
            return 0.5 * (float(value[1]) + float(value[2]))
        if flag == 3 and len(value) >= 3:
            return float(value[1])
    return float(value)


def _normalize_potfile_blocks(
    parsed: dict[str, Any],
    base_dir: Path,
    par_reference: tuple[int, float, float] | None,
) -> list[dict[str, Any]]:
    raw_potfiles = parsed.get("potfiles", [])
    if not isinstance(raw_potfiles, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw_potfile in raw_potfiles:
        if not isinstance(raw_potfile, dict):
            continue
        potfile = dict(raw_potfile)
        potfile_id = str(potfile.get("id", "potfile"))
        catalog_path = _resolve_potfile_catalog_path(potfile.get("filein"), base_dir)
        if catalog_path is None:
            raise ValueError(f"Potfile {potfile_id} is missing a valid filein catalog path.")
        catalog_df = _load_dat_catalog(catalog_path, par_reference, catalog_kind="potfile_galaxies")
        potfile["id"] = potfile_id
        potfile["catalog_path"] = str(catalog_path)
        potfile["catalog_df"] = catalog_df
        zlens_value = potfile.get("zlens", potfile.get("z_lens"))
        corekpc_value = potfile.get("corekpc")
        core_arcsec_value = potfile.get("core")
        cutkpc_value = potfile.get("cutkpc")
        cut_arcsec_value = potfile.get("cut")
        potfile["type"] = int(_potfile_nominal_value(potfile.get("type"), f"{potfile_id}.type"))
        potfile["zlens"] = float(_potfile_nominal_value(zlens_value, f"{potfile_id}.zlens"))
        potfile["corekpc"] = (
            _potfile_nominal_value(corekpc_value, f"{potfile_id}.corekpc")
            if corekpc_value is not None
            else None
        )
        potfile["core_arcsec"] = (
            _potfile_nominal_value(core_arcsec_value, f"{potfile_id}.core")
            if core_arcsec_value is not None
            else None
        )
        potfile["mag0"] = _potfile_nominal_value(potfile.get("mag0"), f"{potfile_id}.mag0")
        potfile["sigma_nominal"] = _potfile_nominal_value(potfile.get("sigma"), f"{potfile_id}.sigma")
        potfile["cutkpc_nominal"] = (
            _potfile_nominal_value(cutkpc_value, f"{potfile_id}.cutkpc")
            if cutkpc_value is not None
            else None
        )
        potfile["cut_arcsec_nominal"] = (
            _potfile_nominal_value(cut_arcsec_value, f"{potfile_id}.cut")
            if cut_arcsec_value is not None
            else None
        )
        potfile["vdslope_nominal"] = _potfile_nominal_value(potfile.get("vdslope"), f"{potfile_id}.vdslope")
        potfile["slope_nominal"] = _potfile_nominal_value(potfile.get("slope"), f"{potfile_id}.slope")
        normalized.append(potfile)
    return normalized


def _enrich_potentials_with_images(potentials_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    if potentials_df.empty or images_df.empty:
        return potentials_df.copy()

    result = potentials_df.copy()
    image_catalog_sources = images_df["catalog_source"].dropna().astype(str).unique()
    image_catalog_references = images_df["catalog_reference"].dropna().unique()
    image_redshifts = pd.to_numeric(images_df["catalog_z"], errors="coerce").dropna()

    result["image_catalog_source"] = (
        image_catalog_sources[0] if len(image_catalog_sources) == 1 else list(image_catalog_sources)
    )
    result["image_catalog_reference"] = (
        image_catalog_references[0] if len(image_catalog_references) == 1 else list(image_catalog_references)
    )
    result["image_count"] = int(len(images_df))
    result["image_family_count"] = int(images_df["family_id"].nunique(dropna=True))

    if not image_redshifts.empty:
        result["image_redshift_min"] = float(image_redshifts.min())
        result["image_redshift_max"] = float(image_redshifts.max())

    return result


def _build_potentials_with_priors(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    potentials = parsed.get("potentials", [])
    if not isinstance(potentials, list):
        return []

    limits = parsed.get("limits", [])
    if isinstance(limits, dict):
        limits = [limits]
    elif not isinstance(limits, list):
        limits = []

    limit_lookup: dict[str, dict[str, Any]] = {}
    for limit_block in limits:
        if not isinstance(limit_block, dict):
            continue

        normalized_id = _normalize_block_id(limit_block.get("id"))
        if normalized_id is None:
            continue
        if normalized_id in limit_lookup:
            raise ValueError(f"Duplicate limit block normalized id '{normalized_id}'.")

        limit_lookup[normalized_id] = limit_block

    cosmo = _build_cosmology(parsed)
    potentials_with_priors: list[dict[str, Any]] = []
    for potential in potentials:
        if not isinstance(potential, dict):
            continue

        potential_with_priors = dict(potential)
        normalized_potential_id = _normalize_block_id(potential.get("id"))
        matched_limit = limit_lookup.get(normalized_potential_id) if normalized_potential_id is not None else None
        if matched_limit is None:
            priors = {}
        else:
            priors = {
                key: value for key, value in matched_limit.items() if key != "id"
            }
        normalized_potential, normalized_priors = _normalize_potential_definition(
            potential_with_priors,
            priors,
            cosmo,
        )
        normalized_potential["priors"] = normalized_priors

        potentials_with_priors.append(normalized_potential)

    return potentials_with_priors


def _enrich_potentials(
    potentials_df: pd.DataFrame,
    dat_files: list[str | Path],
    parsed: dict[str, Any],
    cosmo: Any,
) -> pd.DataFrame:
    if potentials_df.empty:
        return potentials_df.copy()

    result = potentials_df.copy()
    result["id"] = result["id"].astype(str)

    for column in [
        "ra",
        "dec",
        "catalog_reference",
        "catalog_source",
        "catalog_a",
        "catalog_b",
        "catalog_theta",
        "catalog_mag",
        "catalog_z",
        "catalog_lum",
    ]:
        if column not in result.columns:
            result[column] = np.nan if column != "catalog_source" else None

    par_reference = _extract_reference(parsed)

    for dat_file in dat_files:
        catalog_df = _load_dat_catalog(dat_file, par_reference, catalog_kind="potfile_galaxies")
        if catalog_df.empty:
            continue

        catalog_df = catalog_df.set_index("id")
        for row_index, potential_id in result["id"].items():
            if potential_id not in catalog_df.index:
                continue

            current_source = result.at[row_index, "catalog_source"]
            current_ra = result.at[row_index, "ra"]
            current_dec = result.at[row_index, "dec"]
            has_complete_match = current_source is not None and not pd.isna(current_ra) and not pd.isna(current_dec)
            if has_complete_match:
                continue

            catalog_row = catalog_df.loc[potential_id]
            if isinstance(catalog_row, pd.DataFrame):
                catalog_row = catalog_row.iloc[0]

            for column in catalog_df.columns:
                value = catalog_row[column]
                if column == "catalog_source":
                    if result.at[row_index, column] is None:
                        result.at[row_index, column] = value
                    continue
                if pd.isna(result.at[row_index, column]):
                    result.at[row_index, column] = value

    if par_reference is None:
        return result

    _, ra0_deg, dec0_deg = par_reference
    rows_with_coords = result["ra"].notna() & result["dec"].notna()
    if not rows_with_coords.any():
        return result

    z_values = result.loc[rows_with_coords, "catalog_z"] if "catalog_z" in result.columns else None
    _, _, x_arcsec, y_arcsec = _compute_offsets(
        result.loc[rows_with_coords, "ra"].to_numpy(dtype=float),
        result.loc[rows_with_coords, "dec"].to_numpy(dtype=float),
        ra0_deg,
        dec0_deg,
        None if z_values is None else z_values.to_numpy(dtype=float),
        cosmo,
    )

    if "x_centre" not in result.columns:
        result["x_centre"] = np.nan
    if "y_centre" not in result.columns:
        result["y_centre"] = np.nan

    coord_index = result.index[rows_with_coords]
    for offset, row_index in enumerate(coord_index):
        if _is_missing_or_zero(result.at[row_index, "x_centre"]):
            result.at[row_index, "x_centre"] = float(x_arcsec[offset])
        if _is_missing_or_zero(result.at[row_index, "y_centre"]):
            result.at[row_index, "y_centre"] = float(y_arcsec[offset])

    return result


def load_best_par(
    filepath: str | Path,
    dat_files: list[str | Path] | None = None,
    cosmo: Any = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    path = Path(filepath)
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()

    parsed: dict[str, Any] = {"header_comments": []}
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("#"):
            parsed["header_comments"].append(stripped)
            index += 1
            continue

        if stripped in {"finish", "fini"}:
            parsed["finish"] = True
            index += 1
            continue

        block_info, index = _parse_named_block(lines, index)
        block_name = block_info["name"]
        block_data = block_info["data"]

        if block_name in {"potential", "potentiel"}:
            parsed.setdefault("potentials", []).append(block_data)
            continue

        if block_name.startswith("potfile"):
            potfile_data = dict(block_data)
            potfile_data["id"] = block_name
            parsed.setdefault("potfiles", []).append(potfile_data)
            continue

        if block_name in parsed:
            plural_name = _pluralize(block_name)
            if plural_name not in parsed:
                existing = parsed.pop(block_name)
                parsed[plural_name] = [existing]
            parsed[plural_name].append(block_data)
        else:
            parsed[block_name] = block_data

    cosmo = _build_cosmology(parsed)
    raw_potentials = parsed.get("potentials", [])
    normalized_potentials: list[dict[str, Any]] = []
    for potential in raw_potentials if isinstance(raw_potentials, list) else []:
        if not isinstance(potential, dict):
            continue
        normalized_potential, _normalized_priors = _normalize_potential_definition(potential, {}, cosmo)
        normalized_potentials.append(normalized_potential)
    parsed["potentials"] = normalized_potentials

    potentials = parsed.get("potentials", [])
    potentials_df = pd.DataFrame(potentials)
    if not potentials_df.empty and "id" in potentials_df.columns:
        ordered_columns = ["id"] + [column for column in potentials_df.columns if column != "id"]
        potentials_df = potentials_df.loc[:, ordered_columns]

    if dat_files:
        potentials_df = _enrich_potentials(potentials_df, dat_files, parsed, cosmo)

    par_reference = _extract_reference(parsed)
    parsed["potfiles"] = _normalize_potfile_blocks(parsed, path.parent, par_reference)
    image_catalog_paths = _extract_image_catalog_paths(parsed, path.parent)
    images_frames = [_load_multiple_images_catalog(image_catalog_path, par_reference) for image_catalog_path in image_catalog_paths]
    if images_frames:
        images_df = pd.concat(images_frames, axis=0, ignore_index=True)
        images_df = images_df.drop_duplicates(subset=["image_label"], keep="first").reset_index(drop=True)
    else:
        images_df = _empty_images_df()

    potentials_df = _enrich_potentials_with_images(potentials_df, images_df)

    potentials_with_priors = _build_potentials_with_priors(parsed)

    return parsed, potentials_df, images_df, potentials_with_priors
