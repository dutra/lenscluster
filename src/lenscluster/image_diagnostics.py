from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def finite_or(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def unavailable_image_count_info(family: Any, reason: str) -> dict[str, Any]:
    return {
        "produced_image_count": np.nan,
        "recovered_image_count": np.nan,
        "missing_image_count": np.nan,
        "extra_image_count": np.nan,
        "multiplicity_failed": True,
        "multiplicity_failure_reason": reason,
    }


def successful_image_count_info(family: Any) -> dict[str, Any]:
    n_images = int(getattr(family, "n_images", 0))
    return {
        "produced_image_count": n_images,
        "recovered_image_count": n_images,
        "missing_image_count": 0,
        "extra_image_count": 0,
        "multiplicity_failed": False,
        "multiplicity_failure_reason": "",
    }


def image_count_info_from_exact_details(family: Any, details: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(details, dict):
        return unavailable_image_count_info(family, "exact_prediction_failed")
    return {
        "produced_image_count": details.get("produced_image_count", np.nan),
        "recovered_image_count": details.get("recovered_image_count", np.nan),
        "missing_image_count": details.get("missing_image_count", np.nan),
        "extra_image_count": details.get("extra_image_count", np.nan),
        "multiplicity_failed": bool(details.get("multiplicity_failed", details.get("failed", True))),
        "multiplicity_failure_reason": str(details.get("multiplicity_failure_reason", "")),
    }


def model_count_fields_from_count_info(count_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_produced_image_count": count_info.get("produced_image_count", np.nan),
        "model_recovered_image_count": count_info.get("recovered_image_count", np.nan),
        "model_missing_image_count": count_info.get("missing_image_count", np.nan),
        "model_extra_image_count": count_info.get("extra_image_count", np.nan),
        "model_multiplicity_failed": bool(count_info.get("multiplicity_failed", True)),
        "model_multiplicity_failure_reason": str(count_info.get("multiplicity_failure_reason", "")),
    }


def image_count_recovery_row(family: Any, count_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "family_id": str(getattr(family, "family_id", "")),
        "z_source": finite_or(getattr(family, "z_source", np.nan)),
        "effective_z_source": finite_or(getattr(family, "effective_z_source", np.nan)),
        "observed_image_count": int(getattr(family, "n_images", 0)),
        **count_info,
    }


def image_count_recovery_table(state: Any, image_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "family_id",
        "z_source",
        "effective_z_source",
        "observed_image_count",
        "recovered_image_count",
        "produced_image_count",
        "missing_image_count",
        "extra_image_count",
        "multiplicity_failed",
        "multiplicity_failure_reason",
        "arc_aware_recovered_image_count",
        "arc_aware_missing_image_count",
        "arc_supported_image_count",
        "arc_aware_image_rms_arcsec",
    ]
    rows: list[dict[str, Any]] = []
    family_by_id = {str(family.family_id): family for family in getattr(state, "family_data", [])}
    required = {
        "family_id",
        "model_produced_image_count",
        "model_recovered_image_count",
        "model_missing_image_count",
        "model_extra_image_count",
        "model_multiplicity_failed",
        "model_multiplicity_failure_reason",
    }
    if image_df is not None and not image_df.empty and required.issubset(image_df.columns):
        for family_id, group_df in image_df.groupby("family_id", sort=False):
            family = family_by_id.get(str(family_id))
            first = group_df.iloc[0]
            observed_count = int(getattr(family, "n_images", len(group_df)))
            if "arc_aware_image_residual_arcsec" in group_df:
                arc_residuals = pd.to_numeric(group_df["arc_aware_image_residual_arcsec"], errors="coerce").to_numpy(dtype=float)
                arc_finite = np.isfinite(arc_residuals)
                arc_aware_recovered = int(np.sum(arc_finite))
                arc_aware_missing = int(max(0, observed_count - arc_aware_recovered))
                arc_aware_rms = float(np.sqrt(np.mean(np.square(arc_residuals[arc_finite])))) if np.any(arc_finite) else np.nan
            else:
                arc_aware_recovered = np.nan
                arc_aware_missing = np.nan
                arc_aware_rms = np.nan
            if "arc_supported" in group_df:
                arc_supported_count = int(np.sum(group_df["arc_supported"].astype(bool).to_numpy()))
            elif "arc_recovery_status" in group_df:
                arc_supported_count = int(np.sum(group_df["arc_recovery_status"].astype(str).to_numpy() == "arc_supported"))
            else:
                arc_supported_count = np.nan
            rows.append(
                {
                    "family_id": str(family_id),
                    "z_source": finite_or(first.get("z_source", getattr(family, "z_source", np.nan))),
                    "effective_z_source": finite_or(getattr(family, "effective_z_source", np.nan)),
                    "observed_image_count": observed_count,
                    "recovered_image_count": first.get("model_recovered_image_count", np.nan),
                    "produced_image_count": first.get("model_produced_image_count", np.nan),
                    "missing_image_count": first.get("model_missing_image_count", np.nan),
                    "extra_image_count": first.get("model_extra_image_count", np.nan),
                    "multiplicity_failed": bool(first.get("model_multiplicity_failed", True)),
                    "multiplicity_failure_reason": str(first.get("model_multiplicity_failure_reason", "")),
                    "arc_aware_recovered_image_count": arc_aware_recovered,
                    "arc_aware_missing_image_count": arc_aware_missing,
                    "arc_supported_image_count": arc_supported_count,
                    "arc_aware_image_rms_arcsec": arc_aware_rms,
                }
            )
    else:
        rows = [
            image_count_recovery_row(family, unavailable_image_count_info(family, "not_available"))
            for family in getattr(state, "family_data", [])
        ]
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(rows, columns=columns)
    missing = pd.to_numeric(table["missing_image_count"], errors="coerce")
    extra = pd.to_numeric(table["extra_image_count"], errors="coerce")
    observed = pd.to_numeric(table["observed_image_count"], errors="coerce")
    table["_mismatch_sort"] = np.nan_to_num(missing + extra, nan=-1.0)
    table["_observed_sort"] = np.nan_to_num(observed, nan=0.0)
    table = table.sort_values(
        ["_mismatch_sort", "_observed_sort", "family_id"],
        ascending=[False, False, True],
    ).drop(columns=["_mismatch_sort", "_observed_sort"])
    return table.reset_index(drop=True)


def image_count_recovery_summary(image_count_df: pd.DataFrame | None) -> dict[str, Any]:
    empty = {
        "model_recovered_image_count": None,
        "model_produced_image_count": None,
        "model_missing_image_count": None,
        "model_extra_image_count": None,
    }
    if image_count_df is None or image_count_df.empty:
        return empty
    summary: dict[str, Any] = {}
    for summary_key, column in [
        ("model_recovered_image_count", "recovered_image_count"),
        ("model_produced_image_count", "produced_image_count"),
        ("model_missing_image_count", "missing_image_count"),
        ("model_extra_image_count", "extra_image_count"),
    ]:
        values = pd.to_numeric(image_count_df[column], errors="coerce") if column in image_count_df.columns else pd.Series(dtype=float)
        finite = values[np.isfinite(values)]
        summary[summary_key] = int(np.sum(finite)) if len(finite) else None
    return {**empty, **summary}


def diagnostic_detail_array(details: dict[str, Any], key: str, size: int, dtype: Any = float) -> np.ndarray | None:
    if key not in details:
        return None
    try:
        values = np.asarray(details[key], dtype=dtype).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.shape != (size,):
        return None
    return values


def extra_image_rows(
    family: Any,
    details: dict[str, Any] | None,
    model_count_fields: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(details, dict):
        return []
    try:
        x_extra = np.asarray(details.get("extra_model_x_arcsec", []), dtype=float).reshape(-1)
        y_extra = np.asarray(details.get("extra_model_y_arcsec", []), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return []
    if x_extra.shape != y_extra.shape:
        return []
    rows: list[dict[str, Any]] = []
    for index, (x_model, y_model) in enumerate(zip(x_extra, y_extra), start=1):
        rows.append(
            {
                "family_id": str(family.family_id),
                "extra_image_index": int(index),
                "image_recovery_status": "extra",
                "x_model_arcsec": float(x_model),
                "y_model_arcsec": float(y_model),
                "z_source": finite_or(getattr(family, "z_source", np.nan)),
                "effective_z_source": finite_or(getattr(family, "effective_z_source", np.nan)),
                **model_count_fields,
            }
        )
    return rows


def family_image_recovery_rows(
    family: Any,
    exact_details: dict[str, Any] | None,
    *,
    sigma_arcsec: float = np.nan,
    image_sigma_int_arcsec: float = np.nan,
    image_sigma_eff_arcsec: float = np.nan,
    unavailable_reason: str = "exact_prediction_failed",
    unavailable_status: str = "not_recovered",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    n_images = int(getattr(family, "n_images", 0))
    x_pred = np.full(n_images, np.nan, dtype=float)
    y_pred = np.full(n_images, np.nan, dtype=float)
    image_failed = np.ones(n_images, dtype=bool)
    image_recovery_status = np.full(n_images, str(unavailable_status), dtype=object)
    arc_recovery_status: np.ndarray | None = None
    arc_aware_residual = np.full(n_images, np.nan, dtype=float)
    arc_noncritical_direction_residual = np.full(n_images, np.nan, dtype=float)
    arc_critical_direction_residual = np.full(n_images, np.nan, dtype=float)
    arc_critical_direction_x = np.full(n_images, np.nan, dtype=float)
    arc_critical_direction_y = np.full(n_images, np.nan, dtype=float)
    arc_noncritical_direction_x = np.full(n_images, np.nan, dtype=float)
    arc_noncritical_direction_y = np.full(n_images, np.nan, dtype=float)
    arc_s_min = np.full(n_images, np.nan, dtype=float)
    arc_s_max = np.full(n_images, np.nan, dtype=float)
    arc_detA = np.full(n_images, np.nan, dtype=float)
    arc_prior_probability = np.full(n_images, np.nan, dtype=float)
    arc_curve_distance = np.full(n_images, np.nan, dtype=float)
    arc_curve_arclength = np.full(n_images, np.nan, dtype=float)
    arc_curve_finite = np.zeros(n_images, dtype=bool)
    arc_support_anchor_x = np.full(n_images, np.nan, dtype=float)
    arc_support_anchor_y = np.full(n_images, np.nan, dtype=float)
    arc_support_curve_x = np.full(n_images, "[]", dtype=object)
    arc_support_curve_y = np.full(n_images, "[]", dtype=object)
    arc_supported = np.zeros(n_images, dtype=bool)
    arc_support_finite = np.zeros(n_images, dtype=bool)
    cab_has_constraint = np.zeros(n_images, dtype=bool)
    cab_anchor_x = np.full(n_images, np.nan, dtype=float)
    cab_anchor_y = np.full(n_images, np.nan, dtype=float)
    cab_tangent_obs = np.full(n_images, np.nan, dtype=float)
    cab_tangent_model = np.full(n_images, np.nan, dtype=float)
    cab_tangent_residual = np.full(n_images, np.nan, dtype=float)
    cab_curvature_obs = np.full(n_images, np.nan, dtype=float)
    cab_curvature_model = np.full(n_images, np.nan, dtype=float)
    cab_curvature_residual = np.full(n_images, np.nan, dtype=float)
    cab_loglike = np.zeros(n_images, dtype=float)
    cab_finite = np.zeros(n_images, dtype=bool)

    if isinstance(exact_details, dict):
        count_info = image_count_info_from_exact_details(family, exact_details)
        recovered_mask = diagnostic_detail_array(exact_details, "recovered_image_mask", n_images, bool)
        matched_x = diagnostic_detail_array(exact_details, "matched_model_x_arcsec", n_images, float)
        matched_y = diagnostic_detail_array(exact_details, "matched_model_y_arcsec", n_images, float)
        if recovered_mask is not None and matched_x is not None and matched_y is not None:
            x_pred = matched_x
            y_pred = matched_y
            image_recovery_status = np.where(recovered_mask, "recovered", "not_recovered")
            image_failed = ~recovered_mask
        elif not bool(exact_details.get("failed", True)):
            x_exact = diagnostic_detail_array(exact_details, "x_pred", n_images, float)
            y_exact = diagnostic_detail_array(exact_details, "y_pred", n_images, float)
            if x_exact is not None and y_exact is not None:
                x_pred = x_exact
                y_pred = y_exact
                image_recovery_status = np.full(n_images, "recovered", dtype=object)
                image_failed = np.zeros(n_images, dtype=bool)
            else:
                image_recovery_status = np.full(n_images, "not_recovered", dtype=object)
        else:
            image_recovery_status = np.full(n_images, "not_recovered", dtype=object)
        arc_status_values = diagnostic_detail_array(exact_details, "arc_recovery_status", n_images, object)
        if arc_status_values is not None:
            arc_recovery_status = arc_status_values.astype(object)
        arc_array = diagnostic_detail_array(exact_details, "arc_aware_image_residual_arcsec", n_images, float)
        if arc_array is not None:
            arc_aware_residual = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_noncritical_direction_residual_arcsec", n_images, float)
        if arc_array is not None:
            arc_noncritical_direction_residual = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_critical_direction_residual_arcsec", n_images, float)
        if arc_array is not None:
            arc_critical_direction_residual = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_critical_direction_x", n_images, float)
        if arc_array is not None:
            arc_critical_direction_x = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_critical_direction_y", n_images, float)
        if arc_array is not None:
            arc_critical_direction_y = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_noncritical_direction_x", n_images, float)
        if arc_array is not None:
            arc_noncritical_direction_x = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_noncritical_direction_y", n_images, float)
        if arc_array is not None:
            arc_noncritical_direction_y = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_s_min", n_images, float)
        if arc_array is not None:
            arc_s_min = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_s_max", n_images, float)
        if arc_array is not None:
            arc_s_max = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_detA", n_images, float)
        if arc_array is not None:
            arc_detA = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_prior_probability", n_images, float)
        if arc_array is not None:
            arc_prior_probability = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_curve_distance_arcsec", n_images, float)
        if arc_array is not None:
            arc_curve_distance = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_curve_arclength_arcsec", n_images, float)
        if arc_array is not None:
            arc_curve_arclength = arc_array
        arc_bool = diagnostic_detail_array(exact_details, "arc_curve_finite", n_images, bool)
        if arc_bool is not None:
            arc_curve_finite = arc_bool
        arc_array = diagnostic_detail_array(exact_details, "arc_support_anchor_x_arcsec", n_images, float)
        if arc_array is not None:
            arc_support_anchor_x = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_support_anchor_y_arcsec", n_images, float)
        if arc_array is not None:
            arc_support_anchor_y = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_support_curve_x_arcsec", n_images, object)
        if arc_array is not None:
            arc_support_curve_x = arc_array
        arc_array = diagnostic_detail_array(exact_details, "arc_support_curve_y_arcsec", n_images, object)
        if arc_array is not None:
            arc_support_curve_y = arc_array
        arc_bool = diagnostic_detail_array(exact_details, "arc_supported_mask", n_images, bool)
        if arc_bool is None:
            arc_bool = diagnostic_detail_array(exact_details, "arc_supported", n_images, bool)
        if arc_bool is not None:
            arc_supported = arc_bool
        arc_bool = diagnostic_detail_array(exact_details, "arc_support_finite_mask", n_images, bool)
        if arc_bool is None:
            arc_bool = diagnostic_detail_array(exact_details, "arc_support_finite", n_images, bool)
        if arc_bool is not None:
            arc_support_finite = arc_bool
        cab_bool = diagnostic_detail_array(exact_details, "cab_has_constraint", n_images, bool)
        if cab_bool is not None:
            cab_has_constraint = cab_bool
        cab_array = diagnostic_detail_array(exact_details, "cab_anchor_x_arcsec", n_images, float)
        if cab_array is not None:
            cab_anchor_x = cab_array
        cab_array = diagnostic_detail_array(exact_details, "cab_anchor_y_arcsec", n_images, float)
        if cab_array is not None:
            cab_anchor_y = cab_array
        cab_array = diagnostic_detail_array(exact_details, "cab_tangent_angle_obs_rad", n_images, float)
        if cab_array is not None:
            cab_tangent_obs = cab_array
        cab_array = diagnostic_detail_array(exact_details, "cab_tangent_angle_model_rad", n_images, float)
        if cab_array is not None:
            cab_tangent_model = cab_array
        cab_array = diagnostic_detail_array(exact_details, "cab_tangent_residual_rad", n_images, float)
        if cab_array is not None:
            cab_tangent_residual = cab_array
        cab_array = diagnostic_detail_array(exact_details, "cab_curvature_obs_arcsec_inv", n_images, float)
        if cab_array is not None:
            cab_curvature_obs = cab_array
        cab_array = diagnostic_detail_array(exact_details, "cab_curvature_model_arcsec_inv", n_images, float)
        if cab_array is not None:
            cab_curvature_model = cab_array
        cab_array = diagnostic_detail_array(exact_details, "cab_curvature_residual_arcsec_inv", n_images, float)
        if cab_array is not None:
            cab_curvature_residual = cab_array
        cab_array = diagnostic_detail_array(exact_details, "cab_loglike", n_images, float)
        if cab_array is not None:
            cab_loglike = cab_array
        cab_bool = diagnostic_detail_array(exact_details, "cab_finite", n_images, bool)
        if cab_bool is not None:
            cab_finite = cab_bool
    else:
        count_info = unavailable_image_count_info(family, unavailable_reason)
    if arc_recovery_status is None:
        arc_recovery_status = np.where(image_recovery_status == "recovered", "point_recovered", "not_recovered").astype(object)

    model_count_fields = model_count_fields_from_count_info(count_info)
    rows: list[dict[str, Any]] = []
    for (
        label,
        x_obs,
        y_obs,
        x_model,
        y_model,
        status,
        failed,
        arc_status,
        arc_residual,
        noncritical_direction_residual,
        critical_direction_residual,
        critical_direction_x,
        critical_direction_y,
        noncritical_direction_x,
        noncritical_direction_y,
        s_min,
        s_max,
        det_a,
        arc_prior,
        curve_distance,
        curve_arclength,
        curve_finite,
        support_anchor_x,
        support_anchor_y,
        support_curve_x,
        support_curve_y,
        is_arc_supported,
        is_arc_support_finite,
        has_cab_constraint,
        cab_x,
        cab_y,
        cab_tan_obs,
        cab_tan_model,
        cab_tan_residual,
        cab_curv_obs,
        cab_curv_model,
        cab_curv_residual,
        cab_ll,
        is_cab_finite,
    ) in zip(
        getattr(family, "image_labels", []),
        getattr(family, "x_obs", []),
        getattr(family, "y_obs", []),
        x_pred,
        y_pred,
        image_recovery_status,
        image_failed,
        arc_recovery_status,
        arc_aware_residual,
        arc_noncritical_direction_residual,
        arc_critical_direction_residual,
        arc_critical_direction_x,
        arc_critical_direction_y,
        arc_noncritical_direction_x,
        arc_noncritical_direction_y,
        arc_s_min,
        arc_s_max,
        arc_detA,
        arc_prior_probability,
        arc_curve_distance,
        arc_curve_arclength,
        arc_curve_finite,
        arc_support_anchor_x,
        arc_support_anchor_y,
        arc_support_curve_x,
        arc_support_curve_y,
        arc_supported,
        arc_support_finite,
        cab_has_constraint,
        cab_anchor_x,
        cab_anchor_y,
        cab_tangent_obs,
        cab_tangent_model,
        cab_tangent_residual,
        cab_curvature_obs,
        cab_curvature_model,
        cab_curvature_residual,
        cab_loglike,
        cab_finite,
    ):
        residual = (
            math.hypot(float(x_model) - float(x_obs), float(y_model) - float(y_obs))
            if np.isfinite(float(x_model) + float(y_model))
            else np.nan
        )
        arc_residual_value = float(arc_residual)
        if not np.isfinite(arc_residual_value) and str(arc_status) == "point_recovered":
            arc_residual_value = float(residual)
        rows.append(
            {
                "family_id": str(family.family_id),
                "image_label": str(label),
                "x_obs_arcsec": float(x_obs),
                "y_obs_arcsec": float(y_obs),
                "z_source": finite_or(getattr(family, "z_source", np.nan)),
                "effective_z_source": finite_or(getattr(family, "effective_z_source", np.nan)),
                "sigma_arcsec": float(sigma_arcsec),
                "image_sigma_int_arcsec": float(image_sigma_int_arcsec),
                "image_sigma_eff_arcsec": float(image_sigma_eff_arcsec),
                "radius_arcsec": float(math.hypot(float(x_obs), float(y_obs))),
                "angle_deg": float(np.degrees(np.arctan2(float(y_obs), float(x_obs)))),
                "image_recovery_status": str(status),
                **model_count_fields,
                "x_model_arcsec": float(x_model),
                "y_model_arcsec": float(y_model),
                "image_residual_arcsec": float(residual),
                "exact_image_prediction_failed": bool(failed),
                "arc_recovery_status": str(arc_status),
                "arc_aware_image_residual_arcsec": arc_residual_value,
                "arc_noncritical_direction_residual_arcsec": float(noncritical_direction_residual),
                "arc_critical_direction_residual_arcsec": float(critical_direction_residual),
                "arc_critical_direction_x": float(critical_direction_x),
                "arc_critical_direction_y": float(critical_direction_y),
                "arc_noncritical_direction_x": float(noncritical_direction_x),
                "arc_noncritical_direction_y": float(noncritical_direction_y),
                "arc_s_min": float(s_min),
                "arc_s_max": float(s_max),
                "arc_detA": float(det_a),
                "arc_prior_probability": float(arc_prior),
                "arc_curve_distance_arcsec": float(curve_distance),
                "arc_curve_arclength_arcsec": float(curve_arclength),
                "arc_curve_finite": bool(curve_finite),
                "arc_support_anchor_x_arcsec": float(support_anchor_x),
                "arc_support_anchor_y_arcsec": float(support_anchor_y),
                "arc_support_curve_x_arcsec": str(support_curve_x),
                "arc_support_curve_y_arcsec": str(support_curve_y),
                "arc_supported": bool(is_arc_supported),
                "arc_support_finite": bool(is_arc_support_finite),
                "cab_has_constraint": bool(has_cab_constraint),
                "cab_anchor_x_arcsec": float(cab_x),
                "cab_anchor_y_arcsec": float(cab_y),
                "cab_tangent_angle_obs_rad": float(cab_tan_obs),
                "cab_tangent_angle_model_rad": float(cab_tan_model),
                "cab_tangent_residual_rad": float(cab_tan_residual),
                "cab_curvature_obs_arcsec_inv": float(cab_curv_obs),
                "cab_curvature_model_arcsec_inv": float(cab_curv_model),
                "cab_curvature_residual_arcsec_inv": float(cab_curv_residual),
                "cab_loglike": float(cab_ll),
                "cab_finite": bool(is_cab_finite),
            }
        )
    return rows, extra_image_rows(family, exact_details, model_count_fields), count_info


def exact_details_hard_failed(details: dict[str, Any] | None) -> bool:
    if not isinstance(details, dict):
        return True
    reason = str(details.get("multiplicity_failure_reason", ""))
    if reason == "exact_image_prediction_failed":
        try:
            arc_residuals = np.asarray(details.get("arc_aware_image_residual_arcsec", []), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            arc_residuals = np.asarray([], dtype=float)
        if np.isfinite(arc_residuals).any():
            return False
        return True
    if reason in {"source_ray_shooting_failed", "prediction_shape_mismatch"}:
        return True
    if (
        "recovered_image_mask" in details
        or "matched_model_x_arcsec" in details
        or "matched_model_y_arcsec" in details
    ):
        return False
    if not bool(details.get("failed", False)):
        return False
    return bool(details.get("failed", False)) and reason in {"", "nonfinite_prediction", "no_model_images"}
