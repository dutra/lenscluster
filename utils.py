import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


def load_lensmodel_dat(filepath):
    """Load a Lenstool-style .dat catalog into a pandas DataFrame."""
    columns = ["id", "ra", "dec", "a", "b", "theta", "mag", "z"]
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        comment="#",
        header=None,
        names=columns,
    )

    if df.shape[1] != len(columns):
        raise ValueError(
            f"Expected {len(columns)} columns ({columns}), found {df.shape[1]} in '{filepath}'."
        )

    df["id"] = df["id"].astype(int)
    return df


def save_lensmodel_dat(df, filepath, reference=0):
    """Save a DataFrame to Lenstool-style .dat catalog format."""
    columns = ["id", "ra", "dec", "a", "b", "theta", "mag", "z"]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    data = df.loc[:, columns].copy()

    nan_or_empty_cols = []
    for col in data.columns:
        has_nan = data[col].isna().any()
        has_empty = pd.api.types.is_string_dtype(data[col]) and data[col].fillna("").astype(str).str.strip().eq("").any()
        if has_nan or has_empty:
            nan_or_empty_cols.append(col)

    if nan_or_empty_cols:
        print(f"Columns with NaN or empty values: {nan_or_empty_cols}")
    
    data["id"] = data["id"].astype(int)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"#REFERENCE {int(reference)}\n")
        for row in data.itertuples(index=False):
            f.write(
                f"{int(row.id):5d} "
                f"{float(row.ra):12.6f} "
                f"{float(row.dec):11.6f} "
                f"{float(row.a):10.6f} "
                f"{float(row.b):10.6f} "
                f"{float(row.theta):8.2f} "
                f"{float(row.mag):8.4f} "
                f"{float(row.z):5.1f}\n"
            )

def merge_radec(df1, df2, radec1=("ra", "dec"), radec2=("ra", "dec"),
                                         tol_arcsec=0.2, suffixes=("_1", "_2"),
                                         add_sep_col=True, ensure_unique=True,
                                         verbose=True, fields=None, join="inner"):
    """
    Nearest-neighbor match df1 -> df2 on-sky (astropy), keep matches within tol_arcsec.
    Supports inner-style output (matched rows only), left-style output (all df1 rows),
    or outer-style output (all df1 rows plus unmatched df2 rows).

    Returns:
      matched_df, unmatched_df1, unmatched_df2

    Notes:
    - Matching is done from df1 to df2 using match_to_catalog_sky (nearest neighbor).
    - If ensure_unique=True, df2 rows are used at most once (greedy by smallest separation).
    """
    ra1, dec1 = radec1
    ra2, dec2 = radec2

    if ra1 not in df1.columns or dec1 not in df1.columns:
        raise ValueError(f"df1 must contain columns {radec1}")
    if ra2 not in df2.columns or dec2 not in df2.columns:
        raise ValueError(f"df2 must contain columns {radec2}")

    if join not in {"inner", "left", "outer"}:
        raise ValueError("join must be 'inner', 'left', or 'outer'")
    if not isinstance(suffixes, (tuple, list)) or len(suffixes) != 2:
        raise ValueError("suffixes must be a 2-item tuple/list")

    suffixes_norm = tuple("" if s is None else str(s) for s in suffixes)
    coalesce_mode = suffixes_norm[0] == "" and suffixes_norm[1] == ""

    if fields is None:
        fields = list(df2.columns)
    else:
        fields = list(fields)
        missing_fields = [f for f in fields if f not in df2.columns]
        if missing_fields:
            raise ValueError(f"df2 is missing requested field(s): {missing_fields}")

    c1 = SkyCoord(df1[ra1].to_numpy() * u.deg, df1[dec1].to_numpy() * u.deg, frame="icrs")
    c2 = SkyCoord(df2[ra2].to_numpy() * u.deg, df2[dec2].to_numpy() * u.deg, frame="icrs")

    idx2, sep2d, _ = c1.match_to_catalog_sky(c2)
    tol = tol_arcsec * u.arcsec
    within = sep2d <= tol

    # Candidate matches: rows in df1 that have a close neighbor in df2
    cand_i1 = pd.Index(df1.index[within])
    cand_i2 = pd.Index(df2.index[idx2[within]])
    cand_sep = sep2d[within].arcsec

    if ensure_unique and len(cand_i1) > 0:
        # Greedy unique assignment by ascending separation
        order = pd.Series(cand_sep).sort_values().index.to_numpy()
        used2 = set()
        keep_pos = []
        for p in order:
            j2 = int(cand_i2[p])
            if j2 in used2:
                continue
            used2.add(j2)
            keep_pos.append(p)
        keep_pos = sorted(keep_pos)
        i1 = cand_i1[keep_pos]
        i2 = cand_i2[keep_pos]
        sep_arcsec = pd.Series(cand_sep).iloc[keep_pos].to_numpy()
    else:
        i1 = cand_i1
        i2 = cand_i2
        sep_arcsec = cand_sep

    # Unmatched (kept stable across join modes)
    unmatched_df1 = df1.drop(index=i1).copy()
    unmatched_df2 = df2.drop(index=i2).copy()

    # Determine overlapping non-coordinate columns to suffix (pandas-merge style)
    common = set(df1.columns) & set(fields)
    coord_cols = {ra1, dec1, ra2, dec2}
    common_noncoord = common - coord_cols

    left_rename = {c: f"{c}{suffixes_norm[0]}" for c in common_noncoord}
    right_rename = {c: f"{c}{suffixes_norm[1]}" for c in common_noncoord}

    # matched subsets
    left_match = df1.loc[i1].copy().rename(columns=left_rename)
    right_match = df2.loc[i2, fields].copy().rename(columns=right_rename)

    # If coordinate names collide, suffix df2 coordinate columns
    coord_rename = {}
    if ra2 in right_match.columns and ra2 in left_match.columns:
        coord_rename[ra2] = f"{ra2}{suffixes_norm[1]}"
    if dec2 in right_match.columns and dec2 in left_match.columns:
        coord_rename[dec2] = f"{dec2}{suffixes_norm[1]}"
    if coord_rename:
        right_match = right_match.rename(columns=coord_rename)

    def _empty_mask(s):
        empty = s.isna()
        if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
            stripped_empty = s.fillna("").astype(str).str.strip() == ""
            empty = empty | stripped_empty
        return empty

    def _merge_columns(left_df, right_df):
        out = left_df.copy()
        for col in right_df.columns:
            if col in out.columns:
                left_empty = _empty_mask(out[col])
                right_empty = _empty_mask(right_df[col])
                use_right = left_empty & (~right_empty)
                if use_right.any():
                    # `Series.where` promotes dtypes when needed, which avoids
                    # pandas warnings from assigning strings/bools into float
                    # placeholder columns during outer joins.
                    out[col] = out[col].where(~use_right, right_df[col])
            else:
                out[col] = right_df[col]
        return out

    if join == "inner":
        left_out = left_match.reset_index(drop=True)
        right_out = right_match.reset_index(drop=True)
        if coalesce_mode:
            matched = _merge_columns(left_out, right_out)
        else:
            matched = pd.concat([left_out, right_out], axis=1)
        if add_sep_col:
            matched["match_sep_arcsec"] = sep_arcsec
    else:
        left_out = df1.copy().rename(columns=left_rename)
        # Build an empty typed frame rather than a float NaN block so later
        # assignment of object/bool columns does not hit incompatible-dtype warnings.
        right_out = right_match.iloc[:0].reindex(left_out.index)
        if len(i1) > 0:
            right_values = right_match.reset_index(drop=True)
            right_values.index = i1
            right_out.loc[i1, right_values.columns] = right_values

        left_out_reset = left_out.reset_index(drop=True)
        right_out_reset = right_out.reset_index(drop=True)
        if coalesce_mode:
            matched = _merge_columns(left_out_reset, right_out_reset)
        else:
            matched = pd.concat([left_out_reset, right_out_reset], axis=1)
        if add_sep_col:
            sep_series = pd.Series(np.nan, index=left_out.index, dtype=float)
            if len(i1) > 0:
                sep_series.loc[i1] = sep_arcsec
            matched["match_sep_arcsec"] = sep_series.reset_index(drop=True).to_numpy()

        if join == "outer":
            left_nan_block = left_out.iloc[:0].reindex(unmatched_df2.index)
            right_unmatched = unmatched_df2.loc[:, fields].copy().rename(columns=right_rename)
            if coord_rename:
                right_unmatched = right_unmatched.rename(columns=coord_rename)
            left_nan_block_reset = left_nan_block.reset_index(drop=True)
            right_unmatched_reset = right_unmatched.reset_index(drop=True)
            if coalesce_mode:
                outer_tail = _merge_columns(left_nan_block_reset, right_unmatched_reset)
            else:
                outer_tail = pd.concat([left_nan_block_reset, right_unmatched_reset], axis=1)
            if add_sep_col:
                outer_tail["match_sep_arcsec"] = np.nan
            matched = pd.concat([matched, outer_tail], axis=0, ignore_index=True)

    if verbose:
        print(f"Total df1: {len(df1)}, Total df2: {len(df2)}")
        print(f"Matched: {len(matched)}, Unmatched df1: {len(unmatched_df1)}, Unmatched df2: {len(unmatched_df2)}")

    return matched, unmatched_df1, unmatched_df2

def radec_to_offsets(ra, dec, ra0, dec0, z, cosmo):
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)

    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    c0 = SkyCoord(ra=ra0*u.deg, dec=dec0*u.deg, frame="icrs")

    # small-angle tangent-plane offsets, with correct sign conventions:
    # +x = East (increasing RA), +y = North (increasing Dec)
    dlon, dlat = c.spherical_offsets_to(c0)  # returns (lon_offset, lat_offset)
    x_arcsec = (dlon.to(u.arcsec)).value    
    y_arcsec = (-dlat.to(u.arcsec)).value    # flip sign so +y is North (Dec increasing)

    arcsec2kpc = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec).value
    x_kpc = x_arcsec * arcsec2kpc
    y_kpc = y_arcsec * arcsec2kpc

    return x_kpc, y_kpc, x_arcsec, y_arcsec
