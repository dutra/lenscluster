from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_lenscluster_validation")
os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/numba_cache_{os.getuid()}")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
try:
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal test environments
    class Progress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "Progress":
            return self

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
            return False

        def add_task(self, description: str, total: int | None = None) -> int:
            return 0

        def update(self, task_id: int, **kwargs: Any) -> None:
            return None

        def advance(self, task_id: int, advance: int = 1) -> None:
            return None

    class BarColumn:
        pass

    class MofNCompleteColumn:
        pass

    class TextColumn:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class TimeElapsedColumn:
        pass

from .jax_cosmology import (
    critical_surface_density_angle_from_config,
    flat_wcdm_config,
)
import matplotlib

matplotlib.use("Agg")
from matplotlib.path import Path as MplPath
from matplotlib import pyplot as plt

from .mock_cluster import (
    DEFAULT_CAUSTIC_BOUNDARY_MARGIN_ARCSEC,
    DEFAULT_CAUSTIC_COMPUTE_WINDOW_ARCSEC,
    DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC,
    DEFAULT_CAUSTIC_MIN_AREA_ARCSEC2,
    ORIGINAL_DPIE_PROFILE_NAME,
    CausticContour,
    DPIETruth,
    MockClusterPaths,
    SingleBCGMockConfig,
    SourceTruth,
    _axis_ratio_to_lenstool_ellipticite,
    _caustic_config_from_truth,
    _caustic_contours_by_z_from_truth,
    _compute_tangential_caustic_contours,
    _dex_scatter_to_ln,
    _image_count_requirement_text,
    _lenstool_ellipticite_to_axis_ratio,
    _sample_point_in_caustic,
    generate_single_bcg_mock,
)
from .plotting import (
    _best_fit_values_for_specs,
    _cosmology_parameter_subset,
    _corner_without_source_positions,
    _plot_corner,
    _plot_cosmology_corner,
    _plot_potfile_corner,
    _scaling_parameter_subset,
)
from .utils import (
    close_debug_log as _close_debug_log,
    configure_debug_log as _configure_debug_log,
    fmt_seconds as _fmt_seconds,
    log_exception as _log_exception,
    log_message as _log,
    log_stage_banner as _log_stage_banner,
    run_logged_phase as _run_logged_phase,
)

FIT_METHOD_SVI = "svi"
FIT_METHOD_SVI_NUTS = "svi+nuts"
FIT_METHOD_NS = "ns"
FIT_METHOD_SMC = "smc"
SOLVER_FIT_MODE_SEQUENTIAL = "sequential"
SOLVER_FIT_MODE_EVIDENCE_NS = "evidence-ns"
JAX_DEVICE_AUTO = "auto"
JAX_DEVICE_CPU = "cpu"
JAX_DEVICE_GPU = "gpu"
JAX_DEVICE_CHOICES = (JAX_DEVICE_AUTO, JAX_DEVICE_CPU, JAX_DEVICE_GPU)
IMAGE_PLANE_MODE_NONE = "none"
IMAGE_PLANE_MODE_LOCAL_JACOBIAN = "local-jacobian"
IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA = "linearized-forward-beta-image-plane"
IMAGE_PLANE_MODE_MARGINAL = "marginal-image-plane"
EVIDENCE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE = "linearized-marginal-beta-image-plane"
EVIDENCE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE = "linearized-forward-beta-image-plane"
EVIDENCE_LIKELIHOOD_MODES = (
    EVIDENCE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE,
    EVIDENCE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
)
DEFAULT_EVIDENCE_LIKELIHOOD_MODE = EVIDENCE_LIKELIHOOD_LINEARIZED_MARGINAL_BETA_IMAGE_PLANE
DEFAULT_SMC_PARTICLES = 4096
DEFAULT_SMC_MCMC_KERNEL = "rmh"
SMC_MCMC_KERNELS = ("rmh", "mala")
DEFAULT_SMC_MCMC_STEPS = 4
DEFAULT_SMC_TARGET_ESS_FRAC = 0.8
DEFAULT_SMC_MAX_TEMPERATURE_STEPS = 256
DEFAULT_SMC_RMH_SCALE = 1.0
DEFAULT_SMC_MALA_STEP_SIZE = 0.05
DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC = 0.3
DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC = 2.0
DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC = 0.0
IMAGE_PLANE_SCATTER_PRIOR_LOG_UNIFORM = "log-uniform"
IMAGE_PLANE_SCATTER_PRIOR_LOGNORMAL = "lognormal"
IMAGE_PLANE_SCATTER_PRIORS = (
    IMAGE_PLANE_SCATTER_PRIOR_LOG_UNIFORM,
    IMAGE_PLANE_SCATTER_PRIOR_LOGNORMAL,
)
DEFAULT_IMAGE_PLANE_SCATTER_PRIOR = IMAGE_PLANE_SCATTER_PRIOR_LOG_UNIFORM
DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC = 0.3
DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA = 0.5
DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC = 0.30
DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC = 0.10
DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS = 0.05
DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN = 0.05
DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN = 0.0
DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC = 0.0
LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_GAUSSIAN = "gaussian"
LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_STUDENT_T = "student-t"
LIKELIHOOD_STABILIZER_RESIDUAL_LOSSES = (
    LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_GAUSSIAN,
    LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_STUDENT_T,
)
DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS = LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_GAUSSIAN
DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU = 4.0
POSTERIOR_DIAGNOSTIC_MODE_EXACT = "exact"
POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE = "approximate"
POSTERIOR_DIAGNOSTIC_MODES = (
    POSTERIOR_DIAGNOSTIC_MODE_EXACT,
    POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
)


class _ValidationRecoveryProgress:
    def __init__(self, args: argparse.Namespace | None = None) -> None:
        self.enabled = not bool(getattr(args, "quiet", False))
        self._progress_cm: Progress | None = None
        self._progress: Progress | None = None
        self._parent_task: int | None = None
        self._parent_total = 0

    def __enter__(self) -> "_ValidationRecoveryProgress":
        if not self.enabled:
            return self
        self._progress_cm = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        )
        self._progress = self._progress_cm.__enter__()
        self._parent_task = self._progress.add_task("recovery: starting", total=0)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._progress is not None and self._parent_task is not None and exc_type is None:
            self._progress.update(self._parent_task, description="recovery: complete")
        if self._progress_cm is not None:
            return bool(self._progress_cm.__exit__(exc_type, exc, traceback))
        return False

    def begin_phase(self, description: str) -> None:
        if self._progress is None or self._parent_task is None:
            return
        self._parent_total += 1
        self._progress.update(
            self._parent_task,
            total=self._parent_total,
            description=f"recovery: {description}",
        )

    def advance_phase(self) -> None:
        if self._progress is None or self._parent_task is None:
            return
        self._progress.advance(self._parent_task)

    def add_subtask(self, description: str, total: int | None) -> int | None:
        if self._progress is None:
            return None
        return self._progress.add_task(description, total=total)

    def update_subtask(self, task_id: int | None, description: str) -> None:
        if self._progress is None or task_id is None:
            return
        self._progress.update(task_id, description=description)

    def advance_subtask(self, task_id: int | None) -> None:
        if self._progress is None or task_id is None:
            return
        self._progress.advance(task_id)


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


@dataclass(frozen=True)
class ValidationStageFitControls:
    fit_method: str
    warmup: int
    samples: int
    max_tree_depth: int

    def to_json(self) -> dict[str, str | int]:
        return {
            "fit_method": self.fit_method,
            "warmup": self.warmup,
            "samples": self.samples,
            "max_tree_depth": self.max_tree_depth,
        }


def _parse_optional_positive_int(value: str) -> int | None:
    text = str(value).strip()
    if text.lower() in {"none", "null"}:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return parsed


def _format_optional_positive_int(value: int | None) -> str:
    return "none" if value is None else str(int(value))


def _validation_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _validation_jsonable(asdict(value))
    if isinstance(value, argparse.Namespace):
        return _validation_jsonable(vars(value))
    if isinstance(value, pd.DataFrame):
        return _validation_dataframe_payload(value)
    if isinstance(value, pd.Series):
        return _validation_jsonable(value.tolist())
    if isinstance(value, np.ndarray):
        return _validation_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _validation_jsonable(value.item())
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(key): _validation_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_validation_jsonable(item) for item in value]
    return str(value)


def _validation_dataframe_payload(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [str(column) for column in df.columns],
        "records": _validation_jsonable(df.to_dict(orient="records")),
    }


def _write_strict_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_validation_jsonable(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


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


def parameter_recovery_table(
    samples: np.ndarray,
    parameter_names: list[str],
    truth: dict[str, float],
    *,
    best_fit: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarize posterior recovery against known truth values."""
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2:
        raise ValueError("samples must be a 2D array.")
    if sample_array.shape[1] != len(parameter_names):
        raise ValueError("parameter_names length must match sample columns.")
    best_array = None if best_fit is None else np.asarray(best_fit, dtype=float)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(parameter_names):
        values = sample_array[:, index]
        finite = values[np.isfinite(values)]
        truth_value = truth.get(name)
        if finite.size:
            q16, median, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
        else:
            q16 = median = q84 = np.nan
        if truth_value is None or not np.isfinite(float(truth_value)) or finite.size == 0:
            truth_percentile = np.nan
            bias = np.nan
            covered_68 = False
        else:
            truth_f = float(truth_value)
            truth_percentile = float(np.mean(finite <= truth_f))
            bias = float(median - truth_f)
            covered_68 = bool(q16 <= truth_f <= q84)
        rows.append(
            {
                "parameter": name,
                "truth": np.nan if truth_value is None else float(truth_value),
                "best_fit": np.nan if best_array is None else float(best_array[index]),
                "q16": float(q16),
                "median": float(median),
                "q84": float(q84),
                "bias": float(bias),
                "truth_percentile": float(truth_percentile),
                "covered_68": covered_68,
            }
        )
    return pd.DataFrame(rows)


def magnification_recovery_table(
    truth_images: pd.DataFrame,
    recovered: pd.DataFrame,
    *,
    epsilon: float = 1.0e-8,
) -> pd.DataFrame:
    merged = truth_images.merge(recovered, on="image_label", how="left", suffixes=("_truth", "_recovered"))
    mu_true = pd.to_numeric(merged["magnification_true"], errors="coerce")
    mu_rec = pd.to_numeric(merged["magnification_recovered"], errors="coerce")
    merged["magnification_bias"] = mu_rec - mu_true
    denom = np.maximum(np.abs(mu_true.to_numpy(dtype=float)), float(epsilon))
    merged["abs_magnification_fractional_error"] = (
        np.abs(np.abs(mu_rec.to_numpy(dtype=float)) - np.abs(mu_true.to_numpy(dtype=float))) / denom
    )
    merged["parity_match"] = np.sign(mu_true.to_numpy(dtype=float)) == np.sign(mu_rec.to_numpy(dtype=float))
    for suffix in ("q16", "q50", "q84"):
        column = f"magnification_{suffix}"
        if column in merged:
            values = pd.to_numeric(merged[column], errors="coerce").to_numpy(dtype=float)
            merged[f"abs_magnification_fractional_error_{suffix}"] = (
                np.abs(np.abs(values) - np.abs(mu_true.to_numpy(dtype=float))) / denom
            )
    return merged


def _load_truth(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parameter_truth_with_source_positions(truth: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in dict(truth.get("parameter_truth", {})).items():
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_f):
            values[str(key)] = value_f
    for source in truth.get("sources", []):
        if not isinstance(source, dict):
            continue
        family_id = source.get("family_id")
        if family_id is None:
            continue
        for source_key, suffix in (("beta_x", "beta_x"), ("beta_y", "beta_y")):
            if source_key not in source:
                continue
            try:
                value_f = float(source[source_key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value_f):
                values[f"source.{family_id}.{suffix}"] = value_f
    return values


def _load_plot_bundle(path: str | Path) -> tuple[Any, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    from .cluster_solver import _load_artifacts

    artifacts_dir = Path(path)
    if artifacts_dir.name != "artifacts":
        artifacts_dir = artifacts_dir / "artifacts"
    return _load_artifacts(artifacts_dir)


def _artifact_parameter_names(state: Any) -> list[str]:
    return [str(spec.name) for spec in state.parameter_specs]


def _recovered_model_tables(
    state: Any,
    best_fit_physical: np.ndarray,
    images: pd.DataFrame,
    *,
    quick_diagnostics: bool = False,
    progress: _ValidationRecoveryProgress | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import jax.numpy as jnp

    from .cluster_solver import (
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
    )

    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=DEFAULT_MATCH_TOLERANCE,
        sampling_engine="full",
        active_scaling_galaxies=DEFAULT_ACTIVE_SCALING_GALAXIES,
        refresh_every=DEFAULT_REFRESH_EVERY,
        refresh_param_drift_frac=DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        quick_diagnostics=bool(quick_diagnostics),
    )
    if hasattr(evaluator, "reported_physical_to_latent_parameter_vector"):
        best_fit_latent = evaluator.reported_physical_to_latent_parameter_vector(np.asarray(best_fit_physical, dtype=float))
    else:
        best_fit_latent = _convert_theta_to_latent(np.asarray(best_fit_physical, dtype=float), state.parameter_specs)
    magnification_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    best_predictions = evaluator.evaluate(best_fit_latent).family_predictions
    progress_task = progress.add_subtask("recovered models: families", total=len(state.family_data)) if progress else None
    for family in state.family_data:
        if progress:
            progress.update_subtask(
                progress_task,
                f"recovered models: family={family.family_id} z={float(family.z_source):.4f}",
            )
        model, _solver = evaluator._get_exact_model_solver(family.z_source)
        packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit_latent, dtype=jnp.float64), family.z_source)
        kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
        family_images = images[images["family_id"].astype(str) == str(family.family_id)].copy()
        mu = np.asarray(
            model.magnification(
                family_images["x_obs_arcsec"].to_numpy(dtype=float),
                family_images["y_obs_arcsec"].to_numpy(dtype=float),
                kwargs_lens,
            ),
            dtype=float,
        )
        for label, value in zip(family_images["image_label"].astype(str), mu):
            magnification_rows.append({"image_label": label, "magnification_recovered": float(value)})
        prediction = best_predictions.get(str(family.family_id), {})
        exact_details: dict[str, Any] = {}
        if not quick_diagnostics:
            try:
                exact_details = evaluator._exact_family_prediction_details(best_fit_latent, family)
            except Exception:
                exact_details = {"failed": True}
        x_pred = np.asarray(exact_details.get("x_pred", np.full(family.n_images, np.nan)), dtype=float)
        y_pred = np.asarray(exact_details.get("y_pred", np.full(family.n_images, np.nan)), dtype=float)
        for label, x_obs, y_obs, x_model, y_model in zip(
            family.image_labels,
            family.x_obs,
            family.y_obs,
            x_pred,
            y_pred,
        ):
            residual = math.hypot(float(x_model - x_obs), float(y_model - y_obs)) if np.isfinite(x_model + y_model) else np.nan
            image_rows.append(
                {
                    "image_label": str(label),
                    "family_id": str(family.family_id),
                    "x_obs_arcsec": float(x_obs),
                    "y_obs_arcsec": float(y_obs),
                    "x_model_arcsec": float(x_model),
                    "y_model_arcsec": float(y_model),
                    "image_residual_arcsec": float(residual),
                }
            )
        source_rows.append(
            {
                "family_id": str(family.family_id),
                "source_x_recovered": float(prediction.get("source_x", np.nan)),
                "source_y_recovered": float(prediction.get("source_y", np.nan)),
                "source_plane_rms_arcsec": float(prediction.get("source_plane_rms", np.nan)),
                "exact_image_rms_arcsec": float(exact_details.get("exact_image_rms", np.nan)),
                "failed": bool(prediction.get("failed", False) or exact_details.get("failed", False)),
            }
        )
        if progress:
            progress.advance_subtask(progress_task)
    return pd.DataFrame(magnification_rows), pd.DataFrame(image_rows), pd.DataFrame(source_rows)


def _quantile_summary(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan, np.nan
    q16, q50, q84 = np.quantile(array, [0.16, 0.5, 0.84])
    return float(q16), float(q50), float(q84)


def _median_std_summary(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan, np.nan
    median = float(np.median(array))
    std = float(np.std(array))
    return median - std, median, median + std


RECOVERY_PROFILE_POSTERIOR_DRAW_CAP = 128


def _capped_evenly_spaced_posterior_draws(
    samples: np.ndarray,
    *,
    max_draws: int = RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=float)
    if int(max_draws) <= 0:
        raise ValueError("max_draws must be positive.")
    if sample_array.shape[0] <= int(max_draws):
        return sample_array
    indices = np.linspace(0, sample_array.shape[0] - 1, int(max_draws), dtype=int)
    return sample_array[indices]


def _nanmedian_no_warning(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.median(array))


_VALIDATION_STAGE_ORDER = (
    "stage1_large_only",
    "stage2_joint",
    "stage3_image_plane",
    "stage4_linearized_image_plane",
    "stage4_marginal_image_plane",
)


def _finite_mean(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.mean(array))


def _finite_median(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.median(array))


def _metric_text(value: Any, *, precision: int = 4) -> str:
    if value is None:
        return "na"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        text = str(value)
        return text if text else "na"
    if not np.isfinite(value_f):
        return "na"
    return f"{value_f:.{precision}g}"


def _stage_root_from_run_dir(solver_run_dir: str | Path) -> Path:
    run_dir = Path(solver_run_dir)
    if run_dir.name in _VALIDATION_STAGE_ORDER:
        return run_dir.parent
    return run_dir


def _validation_stage_dirs(solver_run_dir: str | Path) -> list[tuple[str, Path]]:
    root = _stage_root_from_run_dir(solver_run_dir)
    if (root / "tables" / "run_summary.json").exists():
        return [(root.name, root)]
    stages: list[tuple[str, Path]] = []
    for stage_name in _VALIDATION_STAGE_ORDER:
        stage_dir = root / stage_name
        if (stage_dir / "tables" / "run_summary.json").exists():
            stages.append((stage_name, stage_dir))
    return stages


def _load_stage_run_summary(stage_dir: Path) -> dict[str, Any]:
    path = stage_dir / "tables" / "run_summary.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _stage_family_recovery_metrics(stage_dir: Path) -> dict[str, Any]:
    path = stage_dir / "tables" / "family_diagnostics.csv"
    if not path.exists():
        return {
            "family_count": np.nan,
            "exact_family_count": np.nan,
            "failed_or_missing_exact": np.nan,
            "exact_image_rms_mean": np.nan,
            "exact_image_rms_median": np.nan,
            "source_rms_mean": np.nan,
            "approx_image_rms_mean": np.nan,
        }
    try:
        family_df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return {
            "family_count": np.nan,
            "exact_family_count": np.nan,
            "failed_or_missing_exact": np.nan,
            "exact_image_rms_mean": np.nan,
            "exact_image_rms_median": np.nan,
            "source_rms_mean": np.nan,
            "approx_image_rms_mean": np.nan,
        }
    family_count = int(len(family_df))
    exact_values = (
        pd.to_numeric(family_df.get("exact_image_rms_arcsec", pd.Series(dtype=float)), errors="coerce")
        .to_numpy(dtype=float)
    )
    exact_finite = exact_values[np.isfinite(exact_values)]
    source_values = (
        pd.to_numeric(family_df.get("source_plane_rms_arcsec", pd.Series(dtype=float)), errors="coerce")
        .to_numpy(dtype=float)
    )
    approx_values = (
        pd.to_numeric(family_df.get("approx_image_rms_arcsec", pd.Series(dtype=float)), errors="coerce")
        .to_numpy(dtype=float)
    )
    return {
        "family_count": family_count,
        "exact_family_count": int(exact_finite.size),
        "failed_or_missing_exact": int(family_count - exact_finite.size),
        "exact_image_rms_mean": _finite_mean(exact_values),
        "exact_image_rms_median": _finite_median(exact_values),
        "source_rms_mean": _finite_mean(source_values),
        "approx_image_rms_mean": _finite_mean(approx_values),
    }


def _stage_parameter_recovery_metrics(stage_dir: Path, truth: dict[str, Any]) -> dict[str, Any]:
    default = {
        "truth_parameter_count": np.nan,
        "parameter_median_abs_bias": np.nan,
        "parameter_mean_abs_bias": np.nan,
        "parameter_coverage_68_fraction": np.nan,
        "worst_parameter": "na",
        "worst_parameter_abs_bias": np.nan,
    }
    try:
        state, _saved_args, arrays, _init_diagnostics = _load_plot_bundle(stage_dir)
    except Exception:
        return default
    if "samples" not in arrays or "best_fit" not in arrays:
        return default
    try:
        table = parameter_recovery_table(
            np.asarray(arrays["samples"], dtype=float),
            _artifact_parameter_names(state),
            _parameter_truth_with_source_positions(truth),
            best_fit=np.asarray(arrays["best_fit"], dtype=float),
        )
    except Exception:
        return default
    if table.empty or "bias" not in table or "truth" not in table:
        return default
    truth_values = pd.to_numeric(table["truth"], errors="coerce").to_numpy(dtype=float)
    bias_values = pd.to_numeric(table["bias"], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(truth_values) & np.isfinite(bias_values)
    if not np.any(finite_mask):
        return default
    finite_bias = bias_values[finite_mask]
    abs_bias = np.abs(finite_bias)
    finite_table = table.loc[finite_mask].reset_index(drop=True)
    worst_index = int(np.nanargmax(abs_bias))
    coverage_values = finite_table["covered_68"].astype(float).to_numpy(dtype=float) if "covered_68" in finite_table else np.asarray([], dtype=float)
    return {
        "truth_parameter_count": int(abs_bias.size),
        "parameter_median_abs_bias": float(np.median(abs_bias)),
        "parameter_mean_abs_bias": float(np.mean(abs_bias)),
        "parameter_coverage_68_fraction": _finite_mean(coverage_values),
        "worst_parameter": str(finite_table.loc[worst_index, "parameter"]),
        "worst_parameter_abs_bias": float(abs_bias[worst_index]),
    }


def _collect_validation_stage_recovery_metrics(
    solver_run_dir: str | Path,
    truth_path: str | Path,
) -> list[dict[str, Any]]:
    truth = _load_truth(truth_path)
    rows: list[dict[str, Any]] = []
    for stage_name, stage_dir in _validation_stage_dirs(solver_run_dir):
        run_summary = _load_stage_run_summary(stage_dir)
        row: dict[str, Any] = {
            "stage": stage_name,
            "stage_dir": str(stage_dir),
            "fit_method": run_summary.get("fit_method", "na"),
            "sample_likelihood_mode": run_summary.get("sample_likelihood_mode", "na"),
            "sampler": run_summary.get("sampler", "na"),
            "runtime_sec": run_summary.get("runtime_sec", np.nan),
            "best_loglike": run_summary.get("best_loglike", np.nan),
            "accept_prob_mean": run_summary.get("accept_prob_mean", np.nan),
            "divergence_count": run_summary.get("divergence_count", np.nan),
            "mean_num_steps": run_summary.get("mean_num_steps", np.nan),
            "n_families": run_summary.get("n_families", np.nan),
            "n_images": run_summary.get("n_images", np.nan),
            "fit_cosmology_flat_wcdm": run_summary.get("fit_cosmology_flat_wcdm", False),
            "cosmology_Om0_median": run_summary.get("cosmology_Om0_median", np.nan),
            "cosmology_w0_median": run_summary.get("cosmology_w0_median", np.nan),
        }
        row.update(_stage_family_recovery_metrics(stage_dir))
        row.update(_stage_parameter_recovery_metrics(stage_dir, truth))
        rows.append(row)
    return rows


def _format_validation_run_summary(
    rows: list[dict[str, Any]],
    *,
    run_name: str,
    seed: int,
    solver_run_dir: str | Path,
) -> str:
    solver_run_path = Path(solver_run_dir)
    solver_root = _stage_root_from_run_dir(solver_run_path)
    final_stage = solver_run_path.name if solver_run_path.name in _VALIDATION_STAGE_ORDER else (rows[-1]["stage"] if rows else "na")

    def first_finite_from_end(key: str) -> Any:
        for row in reversed(rows):
            value = row.get(key, np.nan)
            try:
                if np.isfinite(float(value)):
                    return value
            except (TypeError, ValueError):
                continue
        return np.nan

    family_count = first_finite_from_end("n_families")
    image_count = first_finite_from_end("n_images")
    lines = [
        "Validation recovery run summary",
        f"run_name={run_name}",
        f"seed={seed}",
        f"solver_root={solver_root}",
        f"final_stage={final_stage}",
        f"families={_metric_text(family_count)} images={_metric_text(image_count)}",
        "",
    ]
    if not rows:
        lines.append("No stage summaries were found.")
        return "\n".join(lines) + "\n"
    columns = [
        ("stage", "stage"),
        ("fit", "fit_method"),
        ("likelihood", "sample_likelihood_mode"),
        ("sampler", "sampler"),
        ("runtime_s", "runtime_sec"),
        ("best_loglike", "best_loglike"),
        ("accept", "accept_prob_mean"),
        ("div", "divergence_count"),
        ("steps", "mean_num_steps"),
        ("fit_cosmo", "fit_cosmology_flat_wcdm"),
        ("Om0_med", "cosmology_Om0_median"),
        ("w0_med", "cosmology_w0_median"),
        ("families", "family_count"),
        ("exact_fams", "exact_family_count"),
        ("failed_exact", "failed_or_missing_exact"),
        ("exact_image_rms_mean", "exact_image_rms_mean"),
        ("exact_image_rms_median", "exact_image_rms_median"),
        ("source_rms_mean", "source_rms_mean"),
        ("approx_image_rms_mean", "approx_image_rms_mean"),
        ("truth_params", "truth_parameter_count"),
        ("param_med_abs_bias", "parameter_median_abs_bias"),
        ("param_mean_abs_bias", "parameter_mean_abs_bias"),
        ("coverage68", "parameter_coverage_68_fraction"),
    ]
    rendered_rows = [
        {header: _metric_text(row.get(key)) for header, key in columns}
        for row in rows
    ]
    widths = {
        header: max(len(header), *(len(rendered[header]) for rendered in rendered_rows))
        for header, _key in columns
    }
    lines.append("Per-stage metrics:")
    lines.append(" ".join(header.ljust(widths[header]) for header, _key in columns))
    lines.append(" ".join("-" * widths[header] for header, _key in columns))
    for rendered in rendered_rows:
        lines.append(" ".join(rendered[header].ljust(widths[header]) for header, _key in columns))
    lines.extend(["", "Largest parameter bias by stage:"])
    for row in rows:
        lines.append(
            (
                f"{row.get('stage', 'na')}: "
                f"worst_parameter={_metric_text(row.get('worst_parameter'))} "
                f"worst_parameter_abs_bias={_metric_text(row.get('worst_parameter_abs_bias'))}"
            )
        )
    return "\n".join(lines) + "\n"


def write_validation_run_summary(
    solver_run_dir: str | Path,
    truth_path: str | Path,
    output_dir: str | Path,
    *,
    run_name: str,
    seed: int,
) -> Path:
    rows = _collect_validation_stage_recovery_metrics(solver_run_dir, truth_path)
    text = _format_validation_run_summary(rows, run_name=run_name, seed=int(seed), solver_run_dir=solver_run_dir)
    path = Path(output_dir) / "run_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _posterior_prediction_uncertainty_tables(
    state: Any,
    samples_physical: np.ndarray,
    images: pd.DataFrame,
    *,
    max_draws: int = 8,
    max_workers: int = 1,
    posterior_diagnostic_mode: str = POSTERIOR_DIAGNOSTIC_MODE_EXACT,
    progress: _ValidationRecoveryProgress | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import jax.numpy as jnp

    from .cluster_solver import (
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
    )

    diagnostic_mode = str(posterior_diagnostic_mode)
    if diagnostic_mode not in POSTERIOR_DIAGNOSTIC_MODES:
        raise ValueError(
            f"posterior_diagnostic_mode must be one of {POSTERIOR_DIAGNOSTIC_MODES}; got {diagnostic_mode!r}."
        )
    use_exact_predictions = diagnostic_mode == POSTERIOR_DIAGNOSTIC_MODE_EXACT
    summary_fn = _quantile_summary if use_exact_predictions else _median_std_summary

    sample_array = np.asarray(samples_physical, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if sample_array.shape[0] > max_draws:
        indices = np.linspace(0, sample_array.shape[0] - 1, max_draws, dtype=int)
        sample_array = sample_array[indices]

    def make_evaluator() -> Any:
        return ClusterJAXEvaluator(
            state=state,
            match_tolerance_arcsec=DEFAULT_MATCH_TOLERANCE,
            sampling_engine="full",
            active_scaling_galaxies=DEFAULT_ACTIVE_SCALING_GALAXIES,
            refresh_every=DEFAULT_REFRESH_EVERY,
            refresh_param_drift_frac=DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        )

    evaluator = make_evaluator()
    worker_count = min(max(1, int(max_workers)), max(1, len(state.family_data))) if use_exact_predictions else 1
    thread_local = threading.local()

    def family_task_evaluator() -> Any:
        if worker_count <= 1:
            return evaluator
        local_evaluator = getattr(thread_local, "evaluator", None)
        if local_evaluator is None:
            local_evaluator = make_evaluator()
            thread_local.evaluator = local_evaluator
        return local_evaluator

    family_ids = [str(family.family_id) for family in state.family_data]
    empty_family_images = images.iloc[0:0].copy()
    if "family_id" in images:
        image_family_ids = images["family_id"].astype(str)
        images_by_family = {
            family_id: images.loc[image_family_ids == family_id].copy()
            for family_id in family_ids
        }
    else:
        images_by_family = {family_id: empty_family_images for family_id in family_ids}

    mag_by_label: dict[str, list[float]] = {}
    x_by_label: dict[str, list[float]] = {}
    y_by_label: dict[str, list[float]] = {}
    residual_by_label: dict[str, list[float]] = {}
    source_x_by_family: dict[str, list[float]] = {}
    source_y_by_family: dict[str, list[float]] = {}
    source_rms_by_family: dict[str, list[float]] = {}
    exact_rms_by_family: dict[str, list[float]] = {}
    exact_failed_families: set[str] = set()

    n_draws = int(sample_array.shape[0])
    n_families = len(state.family_data)
    progress_task = (
        progress.add_subtask("posterior uncertainty: draws x families", total=n_draws * n_families)
        if progress
        else None
    )

    def process_family_prediction(
        sample_latent: np.ndarray,
        prediction: dict[str, Any],
        family: Any,
        *,
        skip_exact: bool,
    ) -> dict[str, Any]:
        task_evaluator = family_task_evaluator()
        family_id = str(family.family_id)
        model, _solver = task_evaluator._get_exact_model_solver(family.z_source)
        packed_state = task_evaluator._build_packed_lens_state(jnp.asarray(sample_latent, dtype=jnp.float64), family.z_source)
        kwargs_lens = task_evaluator._packed_to_kwargs_lens(packed_state)
        family_images = images_by_family.get(family_id, empty_family_images)
        mu = np.asarray(
            model.magnification(
                family_images["x_obs_arcsec"].to_numpy(dtype=float),
                family_images["y_obs_arcsec"].to_numpy(dtype=float),
                kwargs_lens,
            ),
            dtype=float,
        )
        task_prediction = dict(prediction)
        exact_failed = False
        if use_exact_predictions and not skip_exact:
            exact_prediction = task_evaluator._exact_family_prediction(sample_latent, family)
            if exact_prediction is None:
                exact_failed = True
            else:
                x_pred_exact, y_pred_exact, exact_rms = exact_prediction
                task_prediction["x_pred"] = x_pred_exact
                task_prediction["y_pred"] = y_pred_exact
                task_prediction["exact_image_rms"] = exact_rms
        x_pred = np.asarray(task_prediction.get("x_pred", np.full(family.n_images, np.nan)), dtype=float)
        y_pred = np.asarray(task_prediction.get("y_pred", np.full(family.n_images, np.nan)), dtype=float)
        residuals = np.asarray(
            [
                math.hypot(float(x_model - x_obs), float(y_model - y_obs))
                if np.isfinite(float(x_model) + float(y_model))
                else np.nan
                for x_obs, y_obs, x_model, y_model in zip(family.x_obs, family.y_obs, x_pred, y_pred)
            ],
            dtype=float,
        )
        return {
            "family_id": family_id,
            "image_labels": [str(label) for label in family.image_labels],
            "magnification_labels": [str(label) for label in family_images["image_label"].astype(str)],
            "magnification": mu,
            "x_pred": x_pred,
            "y_pred": y_pred,
            "residuals": residuals,
            "source_x": float(task_prediction.get("source_x", np.nan)),
            "source_y": float(task_prediction.get("source_y", np.nan)),
            "source_plane_rms": float(task_prediction.get("source_plane_rms", np.nan)),
            "exact_image_rms": float(task_prediction.get("exact_image_rms", np.nan)),
            "exact_failed": exact_failed,
        }

    def merge_family_result(result: dict[str, Any]) -> None:
        family_id = str(result["family_id"])
        for label, value in zip(result["magnification_labels"], result["magnification"]):
            mag_by_label.setdefault(str(label), []).append(float(value))
        for label, x_model, y_model, residual in zip(
            result["image_labels"],
            result["x_pred"],
            result["y_pred"],
            result["residuals"],
        ):
            label = str(label)
            if not use_exact_predictions and not np.isfinite(float(x_model) + float(y_model) + float(residual)):
                continue
            x_by_label.setdefault(label, []).append(float(x_model))
            y_by_label.setdefault(label, []).append(float(y_model))
            residual_by_label.setdefault(label, []).append(float(residual))
        source_x_by_family.setdefault(family_id, []).append(float(result["source_x"]))
        source_y_by_family.setdefault(family_id, []).append(float(result["source_y"]))
        source_rms_by_family.setdefault(family_id, []).append(float(result["source_plane_rms"]))
        exact_rms_by_family.setdefault(family_id, []).append(float(result["exact_image_rms"]))

    executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
    try:
        for draw_index, sample in enumerate(sample_array, start=1):
            if hasattr(evaluator, "reported_physical_to_latent_parameter_vector"):
                sample_latent = evaluator.reported_physical_to_latent_parameter_vector(sample)
            else:
                sample_latent = _convert_theta_to_latent(sample, state.parameter_specs)
            family_predictions = evaluator._family_source_summary(sample_latent)
            if worker_count <= 1:
                for family in state.family_data:
                    family_id = str(family.family_id)
                    if progress:
                        progress.update_subtask(
                            progress_task,
                            (
                                f"posterior uncertainty: draw={draw_index}/{n_draws} "
                                f"family={family_id} z={float(family.z_source):.4f} "
                                f"failed_exact={len(exact_failed_families)}"
                            ),
                        )
                    result = process_family_prediction(
                        sample_latent,
                        family_predictions.get(family_id, {}),
                        family,
                        skip_exact=family_id in exact_failed_families,
                    )
                    if result["exact_failed"]:
                        exact_failed_families.add(family_id)
                    merge_family_result(result)
                    if progress:
                        progress.advance_subtask(progress_task)
                continue

            if executor is None:  # pragma: no cover - defensive guard
                raise RuntimeError("posterior uncertainty worker executor was not initialized.")
            results_by_index: dict[int, dict[str, Any]] = {}
            future_by_index = {}
            failed_exact_count_by_index = {}
            for family_index, family in enumerate(state.family_data):
                family_id = str(family.family_id)
                failed_exact_count_by_index[family_index] = len(exact_failed_families)
                future = executor.submit(
                    process_family_prediction,
                    sample_latent,
                    family_predictions.get(family_id, {}),
                    family,
                    skip_exact=family_id in exact_failed_families,
                )
                future_by_index[future] = family_index
            for future in as_completed(future_by_index):
                family_index = future_by_index[future]
                family = state.family_data[family_index]
                family_id = str(family.family_id)
                result = future.result()
                results_by_index[family_index] = result
                if result["exact_failed"]:
                    exact_failed_families.add(family_id)
                if progress:
                    progress.update_subtask(
                        progress_task,
                        (
                            f"posterior uncertainty: draw={draw_index}/{n_draws} "
                            f"family={family_id} z={float(family.z_source):.4f} "
                            f"failed_exact={failed_exact_count_by_index[family_index]}"
                        ),
                    )
                    progress.advance_subtask(progress_task)
            for family_index in range(len(state.family_data)):
                merge_family_result(results_by_index[family_index])
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    mag_rows: list[dict[str, Any]] = []
    for label, values in mag_by_label.items():
        q16, q50, q84 = summary_fn(values)
        mag_rows.append(
            {
                "image_label": label,
                "magnification_q16": q16,
                "magnification_q50": q50,
                "magnification_q84": q84,
            }
        )

    image_rows: list[dict[str, Any]] = []
    for label in sorted(set(x_by_label) | set(y_by_label) | set(residual_by_label)):
        x16, x50, x84 = summary_fn(x_by_label.get(label, []))
        y16, y50, y84 = summary_fn(y_by_label.get(label, []))
        r16, r50, r84 = summary_fn(residual_by_label.get(label, []))
        image_rows.append(
            {
                "image_label": label,
                "x_model_q16": x16,
                "x_model_q50": x50,
                "x_model_q84": x84,
                "y_model_q16": y16,
                "y_model_q50": y50,
                "y_model_q84": y84,
                "image_residual_q16": r16,
                "image_residual_q50": r50,
                "image_residual_q84": r84,
            }
        )

    source_rows: list[dict[str, Any]] = []
    for family_id in sorted(set(source_x_by_family) | set(source_y_by_family)):
        sx16, sx50, sx84 = summary_fn(source_x_by_family.get(family_id, []))
        sy16, sy50, sy84 = summary_fn(source_y_by_family.get(family_id, []))
        sr16, sr50, sr84 = summary_fn(source_rms_by_family.get(family_id, []))
        er16, er50, er84 = summary_fn(exact_rms_by_family.get(family_id, []))
        source_rows.append(
            {
                "family_id": family_id,
                "source_x_q16": sx16,
                "source_x_q50": sx50,
                "source_x_q84": sx84,
                "source_y_q16": sy16,
                "source_y_q50": sy50,
                "source_y_q84": sy84,
                "source_plane_rms_q16": sr16,
                "source_plane_rms_q50": sr50,
                "source_plane_rms_q84": sr84,
                "exact_image_rms_q16": er16,
                "exact_image_rms_q50": er50,
                "exact_image_rms_q84": er84,
            }
        )
    return pd.DataFrame(mag_rows), pd.DataFrame(image_rows), pd.DataFrame(source_rows)


def _magnifications_for_images(state: Any, best_fit_physical: np.ndarray, images: pd.DataFrame) -> pd.DataFrame:
    magnification, _image, _source = _recovered_model_tables(state, best_fit_physical, images)
    return magnification


def _mass_profile_component_groups(state: Any) -> tuple[dict[str, list[int]], dict[str, str]]:
    component_family = np.asarray(state.packed_lens_spec.component_family, dtype=int)
    n_components = len(state.lens_model_list)
    group_indices: dict[str, list[int]] = {
        "total": list(range(n_components)),
        "halo": [0] if n_components > 0 else [],
        "bcg": [1] if n_components > 1 else [],
        "subhalos": np.where(component_family == 1)[0].astype(int).tolist(),
    }
    group_indices["bcg_plus_subhalos"] = group_indices["bcg"] + group_indices["subhalos"]
    display_names = {
        "total": "total",
        "halo": "halo",
        "bcg": "BCG",
        "subhalos": "subhalos",
        "bcg_plus_subhalos": "BCG + subhalos",
    }
    return group_indices, display_names


def _surface_density_annulus_edges(radii_arcsec: np.ndarray) -> np.ndarray:
    radii = np.asarray(radii_arcsec, dtype=float)
    if radii.ndim != 1 or radii.size == 0 or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii_arcsec must be a one-dimensional array of positive finite radii.")
    radii = np.sort(radii)
    if radii.size == 1:
        return np.asarray([0.5 * radii[0], 1.5 * radii[0]], dtype=float)
    midpoints = 0.5 * (radii[:-1] + radii[1:])
    first_width = midpoints[0] - radii[0]
    last_width = radii[-1] - midpoints[-1]
    first_edge = max(0.0, radii[0] - first_width)
    last_edge = radii[-1] + last_width
    return np.concatenate(([first_edge], midpoints, [last_edge])).astype(float)


def _annular_surface_density_msun_per_arcsec2(
    model: LensModel,
    kwargs_lens: list[dict[str, float]],
    indices: list[int],
    radii_arcsec: np.ndarray,
    sigma_crit_angle: float,
    *,
    n_radial: int = 80,
    n_azimuth: int = 96,
) -> np.ndarray:
    if not indices:
        return np.zeros_like(np.asarray(radii_arcsec, dtype=float), dtype=float)
    radii = np.asarray(radii_arcsec, dtype=float)
    edges = _surface_density_annulus_edges(radii)
    theta = (np.arange(int(n_azimuth), dtype=float) + 0.5) * (2.0 * np.pi / float(n_azimuth))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    values: list[float] = []
    for inner, outer in zip(edges[:-1], edges[1:]):
        if outer <= inner:
            values.append(np.nan)
            continue
        area_fraction = (np.arange(int(n_radial), dtype=float) + 0.5) / float(n_radial)
        radial = np.sqrt(inner * inner + area_fraction * (outer * outer - inner * inner))
        x = (radial[:, None] * cos_theta[None, :]).reshape(-1)
        y = (radial[:, None] * sin_theta[None, :]).reshape(-1)
        kappa = np.asarray(model.kappa(x, y, kwargs_lens, k=indices), dtype=float)
        mean_kappa = float(np.nanmean(kappa)) if kappa.size else np.nan
        values.append(mean_kappa * float(sigma_crit_angle))
    return np.asarray(values, dtype=float)


def _deflection_profile_for_samples(
    state: Any,
    samples: np.ndarray,
    truth: dict[str, Any],
    radii_arcsec: np.ndarray,
) -> pd.DataFrame:
    import jax.numpy as jnp

    from .cluster_solver import (
        DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_ACTIVE_SCALING_MIN,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
    )

    config = truth["config"]
    cosmo_config = flat_wcdm_config(h0=70.0, om0=0.3)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    z_lens = float(config["z_lens"])
    z_source = float(config["source_redshift"])
    model = LensModel(
        lens_model_list=list(state.lens_model_list),
        z_lens=z_lens,
        z_source=z_source,
        cosmo=cosmo,
    )
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=DEFAULT_MATCH_TOLERANCE,
        sampling_engine="full",
        active_scaling_galaxies=DEFAULT_ACTIVE_SCALING_GALAXIES,
        active_scaling_selection="adaptive",
        active_scaling_cumulative_fraction=DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        active_scaling_min=DEFAULT_ACTIVE_SCALING_MIN,
        refresh_every=DEFAULT_REFRESH_EVERY,
        refresh_param_drift_frac=DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
    )
    group_indices, display_names = _mass_profile_component_groups(state)
    truth_kwargs_by_z = truth.get("kwargs_lens_by_source_redshift", {})
    truth_kwargs = truth_kwargs_by_z.get(f"{z_source:.8f}", truth.get("kwargs_lens", []))

    def alpha_magnitude(kwargs_lens: list[dict[str, float]], radius: float, indices: list[int]) -> float:
        if not indices:
            return 0.0
        alpha_x, alpha_y = model.alpha(np.asarray([radius]), np.asarray([0.0]), kwargs_lens, k=indices)
        return float(np.hypot(float(alpha_x[0]), float(alpha_y[0])))

    rows: list[dict[str, Any]] = []
    sample_values_by_group_radius: dict[tuple[str, float], list[float]] = {
        (group, float(radius)): [] for group in group_indices for radius in radii_arcsec
    }
    for sample in np.asarray(samples, dtype=float):
        sample_latent = _convert_theta_to_latent(sample, state.parameter_specs)
        packed_state = evaluator._build_packed_lens_state(jnp.asarray(sample_latent, dtype=jnp.float64), z_source)
        kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
        for radius in radii_arcsec:
            radius_f = float(radius)
            for group, indices in group_indices.items():
                sample_values_by_group_radius[(group, radius_f)].append(alpha_magnitude(kwargs_lens, radius_f, indices))

    for group, indices in group_indices.items():
        if group in {"bcg", "subhalos"} and not indices:
            continue
        for radius in radii_arcsec:
            radius_f = float(radius)
            finite = np.asarray(sample_values_by_group_radius[(group, radius_f)], dtype=float)
            finite = finite[np.isfinite(finite)]
            q16, median, q84 = np.quantile(finite, [0.16, 0.5, 0.84]) if finite.size else (np.nan, np.nan, np.nan)
            truth_value = alpha_magnitude(truth_kwargs, radius_f, indices)
            rows.append(
                {
                    "radius_arcsec": radius_f,
                    "component": group,
                    "component_label": display_names[group],
                    "quantity": f"{group}_deflection_magnitude_arcsec",
                    "truth": truth_value,
                    "q16": float(q16),
                    "median": float(median),
                    "q84": float(q84),
                    "bias": float(median - truth_value),
                }
            )
    return pd.DataFrame(rows)


def _surface_density_profile_for_samples(
    state: Any,
    samples: np.ndarray,
    truth: dict[str, Any],
    radii_arcsec: np.ndarray,
) -> pd.DataFrame:
    import jax.numpy as jnp

    from .cluster_solver import (
        DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_ACTIVE_SCALING_MIN,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
    )

    config = truth["config"]
    cosmo_config = flat_wcdm_config(h0=70.0, om0=0.3)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    z_lens = float(config["z_lens"])
    z_source = float(config["source_redshift"])
    model = LensModel(
        lens_model_list=list(state.lens_model_list),
        z_lens=z_lens,
        z_source=z_source,
        cosmo=cosmo,
    )
    sigma_crit_angle = critical_surface_density_angle_from_config(z_lens, z_source, cosmo_config)
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=DEFAULT_MATCH_TOLERANCE,
        sampling_engine="full",
        active_scaling_galaxies=DEFAULT_ACTIVE_SCALING_GALAXIES,
        active_scaling_selection="adaptive",
        active_scaling_cumulative_fraction=DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        active_scaling_min=DEFAULT_ACTIVE_SCALING_MIN,
        refresh_every=DEFAULT_REFRESH_EVERY,
        refresh_param_drift_frac=DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
    )
    group_indices, display_names = _mass_profile_component_groups(state)
    truth_kwargs_by_z = truth.get("kwargs_lens_by_source_redshift", {})
    truth_kwargs = truth_kwargs_by_z.get(f"{z_source:.8f}", truth.get("kwargs_lens", []))

    rows: list[dict[str, Any]] = []
    sample_values_by_group_radius: dict[tuple[str, float], list[float]] = {
        (group, float(radius)): [] for group in group_indices for radius in radii_arcsec
    }
    for sample in np.asarray(samples, dtype=float):
        sample_latent = _convert_theta_to_latent(sample, state.parameter_specs)
        packed_state = evaluator._build_packed_lens_state(jnp.asarray(sample_latent, dtype=jnp.float64), z_source)
        kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
        for group, indices in group_indices.items():
            values = _annular_surface_density_msun_per_arcsec2(
                model,
                kwargs_lens,
                indices,
                radii_arcsec,
                sigma_crit_angle,
            )
            for radius, value in zip(radii_arcsec, values):
                sample_values_by_group_radius[(group, float(radius))].append(float(value))

    truth_values_by_group = {
        group: _annular_surface_density_msun_per_arcsec2(
            model,
            truth_kwargs,
            indices,
            radii_arcsec,
            sigma_crit_angle,
        )
        for group, indices in group_indices.items()
    }
    for group, indices in group_indices.items():
        if group in {"bcg", "subhalos"} and not indices:
            continue
        for radius_index, radius in enumerate(radii_arcsec):
            radius_f = float(radius)
            finite = np.asarray(sample_values_by_group_radius[(group, radius_f)], dtype=float)
            finite = finite[np.isfinite(finite)]
            q16, median, q84 = np.quantile(finite, [0.16, 0.5, 0.84]) if finite.size else (np.nan, np.nan, np.nan)
            truth_value = float(truth_values_by_group[group][radius_index])
            rows.append(
                {
                    "radius_arcsec": radius_f,
                    "component": group,
                    "component_label": display_names[group],
                    "quantity": f"{group}_surface_density_msun_per_arcsec2",
                    "truth": truth_value,
                    "q16": float(q16),
                    "median": float(median),
                    "q84": float(q84),
                    "bias": float(median - truth_value),
                }
            )
    return pd.DataFrame(rows)


def _mass_and_surface_density_profiles_for_samples(
    state: Any,
    samples: np.ndarray,
    truth: dict[str, Any],
    radii_arcsec: np.ndarray,
    *,
    progress: _ValidationRecoveryProgress | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import jax.numpy as jnp

    from .cluster_solver import (
        DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_ACTIVE_SCALING_MIN,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
    )

    config = truth["config"]
    cosmo_config = flat_wcdm_config(h0=70.0, om0=0.3)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    z_lens = float(config["z_lens"])
    z_source = float(config["source_redshift"])
    model = LensModel(
        lens_model_list=list(state.lens_model_list),
        z_lens=z_lens,
        z_source=z_source,
        cosmo=cosmo,
    )
    sigma_crit_angle = critical_surface_density_angle_from_config(z_lens, z_source, cosmo_config)
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=DEFAULT_MATCH_TOLERANCE,
        sampling_engine="full",
        active_scaling_galaxies=DEFAULT_ACTIVE_SCALING_GALAXIES,
        active_scaling_selection="adaptive",
        active_scaling_cumulative_fraction=DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        active_scaling_min=DEFAULT_ACTIVE_SCALING_MIN,
        refresh_every=DEFAULT_REFRESH_EVERY,
        refresh_param_drift_frac=DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
    )
    group_indices, display_names = _mass_profile_component_groups(state)
    truth_kwargs_by_z = truth.get("kwargs_lens_by_source_redshift", {})
    truth_kwargs = truth_kwargs_by_z.get(f"{z_source:.8f}", truth.get("kwargs_lens", []))

    def alpha_magnitude(kwargs_lens: list[dict[str, float]], radius: float, indices: list[int]) -> float:
        if not indices:
            return 0.0
        alpha_x, alpha_y = model.alpha(np.asarray([radius]), np.asarray([0.0]), kwargs_lens, k=indices)
        return float(np.hypot(float(alpha_x[0]), float(alpha_y[0])))

    def empty_group_radius_values() -> dict[tuple[str, float], list[float]]:
        return {(group, float(radius)): [] for group in group_indices for radius in radii_arcsec}

    mass_values_by_group_radius = empty_group_radius_values()
    surface_values_by_group_radius = empty_group_radius_values()
    sample_array = np.asarray(samples, dtype=float)
    progress_task = (
        progress.add_subtask("profile bands: posterior draws", total=int(sample_array.shape[0]))
        if progress
        else None
    )
    for sample_index, sample in enumerate(sample_array, start=1):
        if progress:
            progress.update_subtask(
                progress_task,
                f"profile bands: draw={sample_index}/{int(sample_array.shape[0])}",
            )
        sample_latent = _convert_theta_to_latent(sample, state.parameter_specs)
        packed_state = evaluator._build_packed_lens_state(jnp.asarray(sample_latent, dtype=jnp.float64), z_source)
        kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
        for radius in radii_arcsec:
            radius_f = float(radius)
            for group, indices in group_indices.items():
                mass_values_by_group_radius[(group, radius_f)].append(alpha_magnitude(kwargs_lens, radius_f, indices))
        for group, indices in group_indices.items():
            values = _annular_surface_density_msun_per_arcsec2(
                model,
                kwargs_lens,
                indices,
                radii_arcsec,
                sigma_crit_angle,
            )
            for radius, value in zip(radii_arcsec, values):
                surface_values_by_group_radius[(group, float(radius))].append(float(value))
        if progress:
            progress.advance_subtask(progress_task)
    evaluator.release_runtime_caches()

    truth_surface_values_by_group = {
        group: _annular_surface_density_msun_per_arcsec2(
            model,
            truth_kwargs,
            indices,
            radii_arcsec,
            sigma_crit_angle,
        )
        for group, indices in group_indices.items()
    }
    mass_rows: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    for group, indices in group_indices.items():
        if group in {"bcg", "subhalos"} and not indices:
            continue
        for radius_index, radius in enumerate(radii_arcsec):
            radius_f = float(radius)
            mass_finite = np.asarray(mass_values_by_group_radius[(group, radius_f)], dtype=float)
            mass_finite = mass_finite[np.isfinite(mass_finite)]
            mass_q16, mass_median, mass_q84 = (
                np.quantile(mass_finite, [0.16, 0.5, 0.84]) if mass_finite.size else (np.nan, np.nan, np.nan)
            )
            mass_truth = alpha_magnitude(truth_kwargs, radius_f, indices)
            mass_rows.append(
                {
                    "radius_arcsec": radius_f,
                    "component": group,
                    "component_label": display_names[group],
                    "quantity": f"{group}_deflection_magnitude_arcsec",
                    "truth": mass_truth,
                    "q16": float(mass_q16),
                    "median": float(mass_median),
                    "q84": float(mass_q84),
                    "bias": float(mass_median - mass_truth),
                }
            )

            surface_finite = np.asarray(surface_values_by_group_radius[(group, radius_f)], dtype=float)
            surface_finite = surface_finite[np.isfinite(surface_finite)]
            surface_q16, surface_median, surface_q84 = (
                np.quantile(surface_finite, [0.16, 0.5, 0.84]) if surface_finite.size else (np.nan, np.nan, np.nan)
            )
            surface_truth = float(truth_surface_values_by_group[group][radius_index])
            surface_rows.append(
                {
                    "radius_arcsec": radius_f,
                    "component": group,
                    "component_label": display_names[group],
                    "quantity": f"{group}_surface_density_msun_per_arcsec2",
                    "truth": surface_truth,
                    "q16": float(surface_q16),
                    "median": float(surface_median),
                    "q84": float(surface_q84),
                    "bias": float(surface_median - surface_truth),
                }
            )
    return pd.DataFrame(mass_rows), pd.DataFrame(surface_rows)


def _recovered_caustic_contours_by_z(
    state: Any,
    best_fit_physical: np.ndarray,
    truth: dict[str, Any],
    z_keys: list[str],
    *,
    progress: _ValidationRecoveryProgress | None = None,
) -> dict[str, list[CausticContour]]:
    import jax.numpy as jnp

    from .cluster_solver import (
        DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_ACTIVE_SCALING_MIN,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
    )

    config = _caustic_config_from_truth(truth)
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=DEFAULT_MATCH_TOLERANCE,
        sampling_engine="full",
        active_scaling_galaxies=DEFAULT_ACTIVE_SCALING_GALAXIES,
        active_scaling_selection="adaptive",
        active_scaling_cumulative_fraction=DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        active_scaling_min=DEFAULT_ACTIVE_SCALING_MIN,
        refresh_every=DEFAULT_REFRESH_EVERY,
        refresh_param_drift_frac=DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
    )
    if hasattr(evaluator, "reported_physical_to_latent_parameter_vector"):
        best_fit_latent = evaluator.reported_physical_to_latent_parameter_vector(np.asarray(best_fit_physical, dtype=float))
    else:
        best_fit_latent = _convert_theta_to_latent(np.asarray(best_fit_physical, dtype=float), state.parameter_specs)
    contours_by_z: dict[str, list[CausticContour]] = {}
    progress_task = progress.add_subtask("recovered caustics: redshifts", total=len(z_keys)) if progress else None
    for z_key in z_keys:
        if progress:
            progress.update_subtask(progress_task, f"recovered caustics: z={z_key}")
        try:
            z_source = float(z_key)
        except (TypeError, ValueError):
            if progress:
                progress.advance_subtask(progress_task)
            continue
        model, _solver = evaluator._get_exact_model_solver(z_source)
        packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit_latent, dtype=jnp.float64), z_source)
        kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
        contours = _compute_tangential_caustic_contours(model, kwargs_lens, config)
        if contours:
            contours_by_z[str(z_key)] = contours
        if progress:
            progress.advance_subtask(progress_task)
    return contours_by_z


def _caustic_contour_payload(contours_by_z: dict[str, list[CausticContour]]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(z_source): [_validation_jsonable(contour) for contour in contours]
        for z_source, contours in sorted(contours_by_z.items(), key=lambda item: str(item[0]))
    }


def _recovery_payload_from_tables(
    *,
    run_dir: Path,
    output_dir: Path,
    posterior_diagnostic_draws: int,
    posterior_diagnostic_workers: int,
    posterior_diagnostic_mode: str,
    quick_diagnostics: bool,
    samples: np.ndarray,
    best_fit_values: dict[str, float],
    previous_stage_best_values: Any,
    parameter_names: list[str],
    parameter_df: pd.DataFrame,
    image_df: pd.DataFrame,
    source_df: pd.DataFrame,
    magnification_df: pd.DataFrame,
    mass_profile_df: pd.DataFrame,
    surface_density_df: pd.DataFrame,
    summary: dict[str, float],
    summary_uncertainty: dict[str, tuple[float, float]],
    truth_caustics_by_z: dict[str, list[CausticContour]],
    recovered_caustics_by_z: dict[str, list[CausticContour]],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "posterior_diagnostics": {
            "draws": int(posterior_diagnostic_draws),
            "workers": int(posterior_diagnostic_workers),
            "mode": str(posterior_diagnostic_mode),
            "quick_diagnostics": bool(quick_diagnostics),
        },
        "posterior_sample_count": int(np.asarray(samples).shape[0]),
        "parameter_names": list(parameter_names),
        "best_fit_values": best_fit_values,
        "previous_stage_best_values": previous_stage_best_values,
        "summary": summary,
        "summary_uncertainty": summary_uncertainty,
        "tables": {
            "parameters": _validation_dataframe_payload(parameter_df),
            "images": _validation_dataframe_payload(image_df),
            "sources": _validation_dataframe_payload(source_df),
            "magnification": _validation_dataframe_payload(magnification_df),
            "mass_profile": _validation_dataframe_payload(mass_profile_df),
            "surface_density": _validation_dataframe_payload(surface_density_df),
        },
        "caustics": {
            "truth_by_z": _caustic_contour_payload(truth_caustics_by_z),
            "recovered_by_z": _caustic_contour_payload(recovered_caustics_by_z),
        },
        "output_paths": output_paths,
    }


def write_recovery_outputs(
    run_dir: str | Path,
    truth_path: str | Path,
    mock_images_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    posterior_diagnostic_draws: int = 8,
    posterior_diagnostic_workers: int = 1,
    posterior_diagnostic_mode: str = POSTERIOR_DIAGNOSTIC_MODE_EXACT,
    quick_diagnostics: bool = False,
    progress_args: argparse.Namespace | None = None,
    recovery_payload: dict[str, Any] | None = None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_args: argparse.Namespace | None = None
    posterior_diagnostic_mode = str(posterior_diagnostic_mode)
    if quick_diagnostics:
        posterior_diagnostic_mode = POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE
        _log_validation_approximation_items(
            progress_args,
            [
                "quick_diagnostics=active source-plane and median+/-std post-fit diagnostics; "
                "exact image-position validation skipped"
            ],
        )
    if posterior_diagnostic_mode == POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE:
        _log_validation_approximation_items(
            progress_args,
            [
                "posterior_diagnostic_mode=approximate median+/-std bars; "
                "exact per-draw image validation skipped; image-position posterior bars may be absent"
            ],
        )

    with _ValidationRecoveryProgress(progress_args) as recovery_progress:

        def run_recovery_phase(description: str, phase_name: str, fn):
            recovery_progress.begin_phase(description)
            result = _run_logged_phase(phase_args, phase_name, fn)
            recovery_progress.advance_phase()
            return result

        def load_inputs() -> tuple[dict[str, Any], pd.DataFrame]:
            truth_payload = _load_truth(truth_path)
            if mock_images_path is None:
                if "images" not in truth_payload:
                    raise ValueError(
                        "Truth file must contain an 'images' list when mock_images_path is not provided."
                    )
                image_table = pd.DataFrame(truth_payload["images"])
            else:
                image_table = pd.DataFrame(json.loads(Path(mock_images_path).read_text(encoding="utf-8")))
            return truth_payload, image_table

        truth, images = run_recovery_phase("load inputs", "validation.recovery.load_inputs", load_inputs)
        state, _saved_args, arrays, _init_diagnostics = run_recovery_phase(
            "load plot bundle",
            "validation.recovery.load_plot_bundle",
            lambda: _load_plot_bundle(run_dir),
        )
        samples = np.asarray(arrays["samples"], dtype=float)
        best_fit = np.asarray(arrays["best_fit"], dtype=float)
        parameter_names = _artifact_parameter_names(state)
        truth_values = _parameter_truth_with_source_positions(truth)
        best_fit_values = _best_fit_values_for_specs(state.parameter_specs, best_fit)
        previous_stage_best_values = getattr(state, "previous_stage_best_values", None)
        parameter_df = run_recovery_phase(
            "parameter table",
            "validation.recovery.parameter_table",
            lambda: parameter_recovery_table(
                samples,
                parameter_names,
                truth_values,
                best_fit=best_fit,
            ),
        )
        recovered_mu, image_df, source_df = run_recovery_phase(
            "recovered model tables",
            "validation.recovery.recovered_model_tables",
            lambda: _recovered_model_tables(
                state,
                best_fit,
                images,
                quick_diagnostics=bool(quick_diagnostics),
                progress=recovery_progress,
            ),
        )
        mag_uncertainty_df, image_uncertainty_df, source_uncertainty_df = run_recovery_phase(
            "posterior uncertainty",
            "validation.recovery.posterior_uncertainty_tables",
            lambda: _posterior_prediction_uncertainty_tables(
                state,
                samples,
                images,
                max_draws=int(posterior_diagnostic_draws),
                max_workers=int(posterior_diagnostic_workers),
                posterior_diagnostic_mode=posterior_diagnostic_mode,
                progress=recovery_progress,
            ),
        )

        def finalize_recovery_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            recovered_mu_local = recovered_mu
            image_df_local = image_df
            source_df_local = source_df
            if not mag_uncertainty_df.empty:
                recovered_mu_local = recovered_mu_local.merge(mag_uncertainty_df, on="image_label", how="left")
            if not image_uncertainty_df.empty:
                image_df_local = image_df_local.merge(image_uncertainty_df, on="image_label", how="left")
            if not source_uncertainty_df.empty:
                source_df_local = source_df_local.merge(source_uncertainty_df, on="family_id", how="left")
            magnification_df_local = magnification_recovery_table(images, recovered_mu_local)
            source_truth_df = pd.DataFrame(truth.get("sources", []))
            if not source_truth_df.empty:
                source_df_local = source_truth_df.merge(source_df_local, on="family_id", how="left")
                source_df_local["source_position_error_arcsec"] = np.hypot(
                    source_df_local["source_x_recovered"].to_numpy(dtype=float) - source_df_local["beta_x"].to_numpy(dtype=float),
                    source_df_local["source_y_recovered"].to_numpy(dtype=float) - source_df_local["beta_y"].to_numpy(dtype=float),
                )
                if {
                    "source_x_q16",
                    "source_x_q50",
                    "source_x_q84",
                    "source_y_q16",
                    "source_y_q50",
                    "source_y_q84",
                }.issubset(source_df_local.columns):
                    for suffix in ("q16", "q50", "q84"):
                        source_df_local[f"source_position_error_{suffix}"] = np.hypot(
                            source_df_local[f"source_x_{suffix}"].to_numpy(dtype=float) - source_df_local["beta_x"].to_numpy(dtype=float),
                            source_df_local[f"source_y_{suffix}"].to_numpy(dtype=float) - source_df_local["beta_y"].to_numpy(dtype=float),
                        )
            return recovered_mu_local, image_df_local, source_df_local, magnification_df_local

        recovered_mu, image_df, source_df, magnification_df = run_recovery_phase(
            "finalize tables",
            "validation.recovery.finalize_tables",
            finalize_recovery_tables,
        )
        mass_profile_df = pd.DataFrame()
        surface_density_df = pd.DataFrame()
        truth_caustics_by_z = run_recovery_phase(
            "truth caustics",
            "validation.recovery.truth_caustics",
            lambda: _caustic_contours_by_z_from_truth(truth),
        )
        recovered_caustics_by_z: dict[str, list[CausticContour]] = {}
        has_mass_profile_truth = "config" in truth and (
            "kwargs_lens" in truth or "kwargs_lens_by_source_redshift" in truth
        )
        if has_mass_profile_truth:
            profile_radii_arcsec = np.asarray([2.0, 5.0, 10.0, 20.0, 40.0], dtype=float)
            profile_samples = _capped_evenly_spaced_posterior_draws(samples)
            mass_profile_df, surface_density_df = run_recovery_phase(
                "mass/surface profile bands",
                "validation.recovery.mass_surface_density_profiles",
                lambda: _mass_and_surface_density_profiles_for_samples(
                    state,
                    profile_samples,
                    truth,
                    radii_arcsec=profile_radii_arcsec,
                    progress=recovery_progress,
                ),
            )
            truth_caustics_z7 = _select_critical_caustic_plot_contours(truth_caustics_by_z)
            if truth_caustics_z7:
                try:
                    recovered_caustics_by_z = run_recovery_phase(
                        "recovered caustics",
                        "validation.recovery.recovered_caustics",
                        lambda: _recovered_caustic_contours_by_z(
                            state,
                            best_fit,
                            truth,
                            sorted(truth_caustics_z7),
                            progress=recovery_progress,
                        ),
                    )
                except Exception as exc:  # pragma: no cover - defensive plotting fallback
                    print(f"[validation:critical-caustic] skipped recovered caustic computation: {exc}")
                    recovered_caustics_by_z = {}

        def build_summary() -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
            summary_payload = {
                "n_parameters": len(parameter_df),
                "median_abs_parameter_bias": float(np.nanmedian(np.abs(parameter_df["bias"]))),
                "parameter_coverage_68_fraction": float(np.mean(parameter_df["covered_68"])),
                "n_images": len(magnification_df),
                "median_image_residual_arcsec": _nanmedian_no_warning(image_df["image_residual_arcsec"]),
                "median_source_position_error_arcsec": _nanmedian_no_warning(source_df["source_position_error_arcsec"])
                if "source_position_error_arcsec" in source_df
                else np.nan,
                "median_abs_magnification_frac_error": _nanmedian_no_warning(
                    magnification_df["abs_magnification_fractional_error"]
                ),
                "parity_match_fraction": float(np.nanmean(magnification_df["parity_match"].astype(float))),
            }
            return summary_payload, _summary_uncertainty(parameter_df, image_df, source_df, magnification_df)

        summary, summary_uncertainty = run_recovery_phase(
            "summary",
            "validation.recovery.summary",
            build_summary,
        )
        paths = {
            "corner_plot": output_dir / "corner.pdf",
            "potfile_corner_plot": output_dir / "potfile_corner.pdf",
            "parameter_recovery_log_plot": output_dir / "parameter_recovery_log.pdf",
            "parameter_recovery_linear_plot": output_dir / "parameter_recovery_linear.pdf",
            "mass_profile_plot": output_dir / "mass_profile_recovery.pdf",
            "surface_density_plot": output_dir / "surface_density_recovery.pdf",
            "critical_caustic_plot": output_dir / "critical_caustic_recovery.pdf",
            "magnification_plot": output_dir / "magnification_recovery.pdf",
            "image_recovery_plot": output_dir / "image_recovery.pdf",
            "source_recovery_plot": output_dir / "source_recovery.pdf",
            "subhalo_population_plot": output_dir / "subhalo_population.pdf",
            "summary_plot": output_dir / "validation_summary.pdf",
        }
        run_recovery_phase(
            "corner plot",
            "validation.recovery.plot_corner",
            lambda: _plot_corner_pdf(
                output_dir,
                samples,
                state.parameter_specs,
                "corner.pdf",
                truth_values=truth_values,
                best_fit_values=best_fit_values,
                previous_stage_best_values=previous_stage_best_values,
            ),
        )
        scaling_specs, scaling_samples, scaling_best_fit = run_recovery_phase(
            "scaling subset",
            "validation.recovery.scaling_subset",
            lambda: _scaling_parameter_subset(
                state.parameter_specs,
                samples,
                best_fit,
            ),
        )
        scaling_best_fit_values = _best_fit_values_for_specs(scaling_specs, scaling_best_fit)
        run_recovery_phase(
            "potfile corner plot",
            "validation.recovery.plot_potfile_corner",
            lambda: _plot_corner_pdf(
                output_dir,
                scaling_samples,
                scaling_specs,
                "potfile_corner.pdf",
                truth_values=truth_values,
                best_fit_values=scaling_best_fit_values,
                previous_stage_best_values=previous_stage_best_values,
            ),
        )
        if any(getattr(spec, "component_family", None) == "cosmology" for spec in state.parameter_specs):
            cosmology_specs, cosmology_samples, cosmology_best_fit = run_recovery_phase(
                "cosmology subset",
                "validation.recovery.cosmology_subset",
                lambda: _cosmology_parameter_subset(
                    state.parameter_specs,
                    samples,
                    best_fit,
                ),
            )
            if cosmology_specs:
                paths["cosmology_corner_plot"] = output_dir / "cosmology_corner.pdf"
                cosmology_best_fit_values = _best_fit_values_for_specs(cosmology_specs, cosmology_best_fit)
                run_recovery_phase(
                    "cosmology corner plot",
                    "validation.recovery.plot_cosmology_corner",
                    lambda: _plot_corner_pdf(
                        output_dir,
                        cosmology_samples,
                        cosmology_specs,
                        "cosmology_corner.pdf",
                        truth_values=truth_values,
                        best_fit_values=cosmology_best_fit_values,
                        previous_stage_best_values=previous_stage_best_values,
                    ),
                )
        run_recovery_phase(
            "parameter recovery log plot",
            "validation.recovery.plot_parameter_recovery_log",
            lambda: _plot_parameter_recovery(parameter_df, paths["parameter_recovery_log_plot"], scale="log_abs"),
        )
        run_recovery_phase(
            "parameter recovery linear plot",
            "validation.recovery.plot_parameter_recovery_linear",
            lambda: _plot_parameter_recovery(parameter_df, paths["parameter_recovery_linear_plot"], scale="linear"),
        )
        if not mass_profile_df.empty:
            run_recovery_phase(
                "mass profile plot",
                "validation.recovery.plot_mass_profile",
                lambda: _plot_mass_profile_recovery(mass_profile_df, paths["mass_profile_plot"]),
            )
        else:
            paths.pop("mass_profile_plot", None)
        if not surface_density_df.empty:
            run_recovery_phase(
                "surface density plot",
                "validation.recovery.plot_surface_density",
                lambda: _plot_surface_density_recovery(surface_density_df, paths["surface_density_plot"]),
            )
        else:
            paths.pop("surface_density_plot", None)
        truth_caustics_z7 = _select_critical_caustic_plot_contours(truth_caustics_by_z)
        recovered_caustics_z7 = _select_critical_caustic_plot_contours(recovered_caustics_by_z)
        if truth_caustics_z7 and recovered_caustics_z7:
            run_recovery_phase(
                "critical caustic plot",
                "validation.recovery.plot_critical_caustic",
                lambda: _plot_critical_caustic_recovery(
                    truth_caustics_z7,
                    recovered_caustics_z7,
                    images,
                    image_df,
                    source_df,
                    pd.DataFrame(truth.get("subhalos", [])),
                    paths["critical_caustic_plot"],
                ),
            )
        else:
            paths.pop("critical_caustic_plot", None)
        run_recovery_phase(
            "magnification plot",
            "validation.recovery.plot_magnification",
            lambda: _plot_magnification_recovery(magnification_df, paths["magnification_plot"]),
        )
        run_recovery_phase(
            "image recovery plot",
            "validation.recovery.plot_image",
            lambda: _plot_image_recovery(image_df, paths["image_recovery_plot"]),
        )
        run_recovery_phase(
            "source recovery plot",
            "validation.recovery.plot_source",
            lambda: _plot_source_recovery(source_df, paths["source_recovery_plot"]),
        )
        run_recovery_phase(
            "subhalo population plot",
            "validation.recovery.plot_subhalo_population",
            lambda: _plot_subhalo_population(
                pd.DataFrame(truth.get("subhalos", [])),
                images,
                parameter_df,
                paths["subhalo_population_plot"],
            ),
        )
        run_recovery_phase(
            "summary plot",
            "validation.recovery.plot_summary",
            lambda: _plot_validation_summary(summary, summary_uncertainty, paths["summary_plot"]),
        )
        if recovery_payload is not None:
            recovery_payload.clear()
            recovery_payload.update(
                _validation_jsonable(
                    _recovery_payload_from_tables(
                        run_dir=run_dir,
                        output_dir=output_dir,
                        posterior_diagnostic_draws=int(posterior_diagnostic_draws),
                        posterior_diagnostic_workers=int(posterior_diagnostic_workers),
                        posterior_diagnostic_mode=posterior_diagnostic_mode,
                        quick_diagnostics=bool(quick_diagnostics),
                        samples=samples,
                        best_fit_values=best_fit_values,
                        previous_stage_best_values=previous_stage_best_values,
                        parameter_names=parameter_names,
                        parameter_df=parameter_df,
                        image_df=image_df,
                        source_df=source_df,
                        magnification_df=magnification_df,
                        mass_profile_df=mass_profile_df,
                        surface_density_df=surface_density_df,
                        summary=summary,
                        summary_uncertainty=summary_uncertainty,
                        truth_caustics_by_z=truth_caustics_by_z,
                        recovered_caustics_by_z=recovered_caustics_by_z,
                        output_paths=paths,
                    )
                )
            )
        return paths


PARAMETER_RECOVERY_LOG_ABS_FLOOR = 1.0e-4
CRITICAL_CAUSTIC_RECOVERY_SOURCE_REDSHIFT = 7.0
CRITICAL_CAUSTIC_RECOVERY_REDSHIFT_TOL = 1.0e-6


def _log10_abs_parameter_values(values: np.ndarray, floor: float = PARAMETER_RECOVERY_LOG_ABS_FLOOR) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    transformed = np.full_like(array, np.nan, dtype=float)
    finite = np.isfinite(array)
    if np.any(finite):
        transformed[finite] = np.log10(np.maximum(np.abs(array[finite]), float(floor)))
    return transformed


def _plot_parameter_recovery(parameter_df: pd.DataFrame, path: Path, *, scale: str = "log_abs") -> None:
    if scale not in {"log_abs", "linear"}:
        raise ValueError("scale must be 'log_abs' or 'linear'.")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(parameter_df))))
    y = np.arange(len(parameter_df))
    median_raw = parameter_df["median"].to_numpy(dtype=float)
    q16_raw = parameter_df["q16"].to_numpy(dtype=float)
    q84_raw = parameter_df["q84"].to_numpy(dtype=float)
    truth_raw = parameter_df["truth"].to_numpy(dtype=float)
    if scale == "log_abs":
        median = _log10_abs_parameter_values(median_raw)
        q16 = _log10_abs_parameter_values(q16_raw)
        q84 = _log10_abs_parameter_values(q84_raw)
        truth = _log10_abs_parameter_values(truth_raw)
        xlabel = "log10(abs(parameter value))"
    else:
        median = median_raw
        q16 = q16_raw
        q84 = q84_raw
        truth = truth_raw
        xlabel = "parameter value"
    low = np.minimum(q16, q84)
    high = np.maximum(q16, q84)
    ax.errorbar(
        median,
        y,
        xerr=[np.maximum(0.0, median - low), np.maximum(0.0, high - median)],
        fmt="o",
        color="tab:blue",
        label="posterior 1 sigma",
    )
    ax.scatter(truth, y, marker="x", color="black", linewidths=1.6, label="truth", zorder=5)
    ax.set_yticks(y, parameter_df["parameter"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    finite_values = np.concatenate([median_raw, q16_raw, q84_raw, truth_raw])
    finite_values = finite_values[np.isfinite(finite_values)]
    if scale == "log_abs" and finite_values.size and np.any(np.abs(finite_values) < PARAMETER_RECOVERY_LOG_ABS_FLOOR):
        ax.text(
            0.98,
            0.02,
            f"abs(values) < {PARAMETER_RECOVERY_LOG_ABS_FLOOR:g} clipped",
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
            transform=ax.transAxes,
        )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_corner_placeholder(samples: np.ndarray, parameter_names: list[str], path: Path, plot_name: str) -> None:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.size == 0:
        n_samples = 0
        n_params = len(parameter_names)
        n_dynamic = 0
    else:
        finite_rows = sample_array[np.all(np.isfinite(sample_array), axis=1)]
        n_samples = int(finite_rows.shape[0])
        n_params = int(finite_rows.shape[1]) if finite_rows.ndim == 2 else len(parameter_names)
        if finite_rows.ndim == 2 and finite_rows.shape[0] > 0:
            spans = np.nanmax(finite_rows, axis=0) - np.nanmin(finite_rows, axis=0)
            n_dynamic = int(np.sum(np.isfinite(spans) & (spans > 0.0)))
        else:
            n_dynamic = 0
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(
        0.5,
        0.62,
        f"{plot_name} was not generated",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.42,
        (
            "The saved posterior has fewer than two parameters with dynamic range.\n"
            f"finite samples: {n_samples}, parameters: {n_params}, dynamic parameters: {n_dynamic}.\n"
            "This usually means the sampler/guide posterior collapsed or all retained samples are identical."
        ),
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_corner_pdf(
    output_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[Any],
    filename: str = "corner.pdf",
    truth_values: dict[str, float] | None = None,
    best_fit_values: dict[str, float] | None = None,
    previous_stage_best_values: dict[str, float] | None = None,
) -> None:
    path = output_dir / filename
    if path.exists():
        path.unlink()
    try:
        if filename == "corner.pdf":
            _plot_corner(
                output_dir,
                samples,
                parameter_specs,
                truth_values=truth_values,
                best_fit_values=best_fit_values,
                previous_stage_best_values=previous_stage_best_values,
            )
        elif filename == "cosmology_corner.pdf":
            _plot_cosmology_corner(
                output_dir,
                samples,
                parameter_specs,
                truth_values=truth_values,
                best_fit_values=best_fit_values,
                previous_stage_best_values=previous_stage_best_values,
            )
        else:
            _plot_potfile_corner(
                output_dir,
                samples,
                parameter_specs,
                truth_values=truth_values,
                best_fit_values=best_fit_values,
                previous_stage_best_values=previous_stage_best_values,
            )
    except Exception as exc:  # pragma: no cover - defensive plotting fallback
        placeholder_samples, placeholder_specs = (
            _corner_without_source_positions(samples, parameter_specs, filename)
            if filename == "corner.pdf"
            else (samples, parameter_specs)
        )
        _write_corner_placeholder(
            placeholder_samples,
            [getattr(spec, "name", str(spec)) for spec in placeholder_specs],
            path,
            filename,
        )
        _log_message = f"[validation:corner] wrote placeholder {path}: {exc}"
        print(_log_message)
        return
    if not path.exists():
        placeholder_samples, placeholder_specs = (
            _corner_without_source_positions(samples, parameter_specs, filename)
            if filename == "corner.pdf"
            else (samples, parameter_specs)
        )
        _write_corner_placeholder(
            placeholder_samples,
            [getattr(spec, "name", str(spec)) for spec in placeholder_specs],
            path,
            filename,
        )


def _summary_uncertainty(
    parameter_df: pd.DataFrame,
    image_df: pd.DataFrame,
    source_df: pd.DataFrame,
    magnification_df: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    def interval_from_columns(df: pd.DataFrame, q16_col: str, q84_col: str) -> tuple[float, float]:
        if q16_col not in df or q84_col not in df:
            return np.nan, np.nan
        low = _nanmedian_no_warning(df[q16_col])
        high = _nanmedian_no_warning(df[q84_col])
        return (min(low, high), max(low, high)) if np.isfinite(low + high) else (low, high)

    image_interval = interval_from_columns(image_df, "image_residual_q16", "image_residual_q84")
    source_interval = interval_from_columns(source_df, "source_position_error_q16", "source_position_error_q84")
    mag_interval = interval_from_columns(
        magnification_df,
        "abs_magnification_fractional_error_q16",
        "abs_magnification_fractional_error_q84",
    )
    coverage_values = parameter_df["covered_68"].astype(float).to_numpy(dtype=float)
    coverage_se = (
        float(np.sqrt(np.nanmean(coverage_values) * (1.0 - np.nanmean(coverage_values)) / max(np.sum(np.isfinite(coverage_values)), 1)))
        if coverage_values.size
        else np.nan
    )
    coverage_mean = float(np.nanmean(coverage_values)) if coverage_values.size else np.nan
    parity_values = magnification_df["parity_match"].astype(float).to_numpy(dtype=float)
    parity_se = (
        float(np.sqrt(np.nanmean(parity_values) * (1.0 - np.nanmean(parity_values)) / max(np.sum(np.isfinite(parity_values)), 1)))
        if parity_values.size
        else np.nan
    )
    parity_mean = float(np.nanmean(parity_values)) if parity_values.size else np.nan
    return {
        "median_image_residual_arcsec": image_interval,
        "median_source_position_error_arcsec": source_interval,
        "median_abs_magnification_frac_error": mag_interval,
        "parameter_coverage_68_fraction": (np.nan, np.nan) if not np.isfinite(coverage_se) else (coverage_mean - coverage_se, coverage_mean + coverage_se),
        "parity_match_fraction": (np.nan, np.nan) if not np.isfinite(parity_se) else (parity_mean - parity_se, parity_mean + parity_se),
    }


def _plot_mass_profile_recovery(profile_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25], "hspace": 0.08},
    )
    ax = axes[0]
    ratio_ax = axes[1]
    component_order = ["total", "halo", "bcg_plus_subhalos", "bcg", "subhalos"]
    colors = {
        "total": "tab:blue",
        "halo": "tab:orange",
        "bcg_plus_subhalos": "tab:green",
        "bcg": "tab:red",
        "subhalos": "tab:purple",
    }
    narrow_messages: list[str] = []
    for component in component_order:
        comp_df = profile_df[profile_df["component"] == component].sort_values("radius_arcsec")
        if comp_df.empty:
            continue
        label = str(comp_df["component_label"].iloc[0])
        color = colors.get(component, "0.4")
        radius = comp_df["radius_arcsec"].to_numpy(dtype=float)
        median = comp_df["median"].to_numpy(dtype=float)
        q16 = comp_df["q16"].to_numpy(dtype=float)
        q84 = comp_df["q84"].to_numpy(dtype=float)
        low = np.minimum(q16, q84)
        high = np.maximum(q16, q84)
        truth = comp_df["truth"].to_numpy(dtype=float)
        yerr = [np.maximum(0.0, median - low), np.maximum(0.0, high - median)]
        band_width = high - low
        finite_scale = np.nanmax(np.abs(median[np.isfinite(median)])) if np.isfinite(median).any() else np.nan
        narrow_band = bool(
            np.isfinite(finite_scale)
            and finite_scale > 0.0
            and np.isfinite(band_width).any()
            and np.nanmax(band_width[np.isfinite(band_width)]) < 0.003 * finite_scale
        )
        if narrow_band:
            narrow_messages.append(label)
        line_width = 2.0 if component == "total" else 1.4
        alpha = 0.22 if component == "total" else 0.14
        ax.fill_between(radius, low, high, color=color, alpha=alpha)
        ax.errorbar(
            radius,
            median,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=4 if component == "total" else 3,
            linewidth=line_width,
            label=f"{label} posterior",
        )
        ax.plot(radius, median, color=color, linewidth=line_width)
        ax.plot(radius, truth, color=color, linestyle="--", linewidth=line_width, label=f"{label} truth")

        denom = np.maximum(np.abs(truth), 1.0e-12)
        ratio_median = (median - truth) / denom
        ratio_low = (low - truth) / denom
        ratio_high = (high - truth) / denom
        ratio_ax.fill_between(radius, ratio_low, ratio_high, color=color, alpha=alpha)
        ratio_ax.errorbar(
            radius,
            ratio_median,
            yerr=[
                np.maximum(0.0, ratio_median - np.minimum(ratio_low, ratio_high)),
                np.maximum(0.0, np.maximum(ratio_low, ratio_high) - ratio_median),
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=4 if component == "total" else 3,
        )
        ratio_ax.plot(radius, ratio_median, color=color, linewidth=line_width)

    ax.set_ylabel("deflection magnitude [arcsec]")
    ax.legend(loc="best", fontsize=8)
    if narrow_messages:
        ax.text(
            0.98,
            0.04,
            "narrow 1 sigma bands: " + ", ".join(narrow_messages),
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=8,
            color="0.35",
        )
    ratio_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ratio_ax.set_xlabel("radius [arcsec]")
    ratio_ax.set_ylabel("(post. - truth) / truth")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_surface_density_recovery(profile_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25], "hspace": 0.08},
    )
    ax = axes[0]
    ratio_ax = axes[1]
    component_order = ["total", "halo", "bcg_plus_subhalos", "bcg", "subhalos"]
    colors = {
        "total": "tab:blue",
        "halo": "tab:orange",
        "bcg_plus_subhalos": "tab:green",
        "bcg": "tab:red",
        "subhalos": "tab:purple",
    }
    positive_values: list[float] = []
    for component in component_order:
        comp_df = profile_df[profile_df["component"] == component].sort_values("radius_arcsec")
        if comp_df.empty:
            continue
        label = str(comp_df["component_label"].iloc[0])
        color = colors.get(component, "0.4")
        radius = comp_df["radius_arcsec"].to_numpy(dtype=float)
        median = comp_df["median"].to_numpy(dtype=float)
        q16 = comp_df["q16"].to_numpy(dtype=float)
        q84 = comp_df["q84"].to_numpy(dtype=float)
        low = np.minimum(q16, q84)
        high = np.maximum(q16, q84)
        truth = comp_df["truth"].to_numpy(dtype=float)
        finite_positive = np.concatenate([median, low, high, truth])
        finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0.0)]
        positive_values.extend(float(value) for value in finite_positive)
        yerr = [np.maximum(0.0, median - low), np.maximum(0.0, high - median)]
        line_width = 2.0 if component == "total" else 1.4
        alpha = 0.22 if component == "total" else 0.14
        ax.fill_between(radius, low, high, color=color, alpha=alpha)
        ax.errorbar(
            radius,
            median,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=4 if component == "total" else 3,
            linewidth=line_width,
            label=f"{label} posterior",
        )
        ax.plot(radius, median, color=color, linewidth=line_width)
        ax.plot(radius, truth, color=color, linestyle="--", linewidth=line_width, label=f"{label} truth")

        denom = np.maximum(np.abs(truth), 1.0e-12)
        ratio_median = (median - truth) / denom
        ratio_low = (low - truth) / denom
        ratio_high = (high - truth) / denom
        ratio_ax.fill_between(radius, ratio_low, ratio_high, color=color, alpha=alpha)
        ratio_ax.errorbar(
            radius,
            ratio_median,
            yerr=[
                np.maximum(0.0, ratio_median - np.minimum(ratio_low, ratio_high)),
                np.maximum(0.0, np.maximum(ratio_low, ratio_high) - ratio_median),
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=4 if component == "total" else 3,
        )
        ratio_ax.plot(radius, ratio_median, color=color, linewidth=line_width)

    ax.set_ylabel(r"$\Sigma$ [M$_\odot$ arcsec$^{-2}$]")
    if positive_values:
        all_values = profile_df[["truth", "q16", "median", "q84"]].to_numpy(dtype=float).reshape(-1)
        finite_values = all_values[np.isfinite(all_values)]
        if finite_values.size and np.all(finite_values > 0.0):
            ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    ratio_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ratio_ax.set_xlabel("radius [arcsec]")
    ratio_ax.set_ylabel("(post. - truth) / truth")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_magnification_recovery(magnification_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    truth = magnification_df["magnification_true"].to_numpy(dtype=float)
    recovered = magnification_df["magnification_recovered"].to_numpy(dtype=float)
    plotted = recovered
    if {"magnification_q16", "magnification_q50", "magnification_q84"}.issubset(magnification_df.columns):
        q16 = magnification_df["magnification_q16"].to_numpy(dtype=float)
        q50 = magnification_df["magnification_q50"].to_numpy(dtype=float)
        q84 = magnification_df["magnification_q84"].to_numpy(dtype=float)
        plotted = q50
        ax.errorbar(
            truth,
            q50,
            yerr=[np.maximum(0.0, q50 - q16), np.maximum(0.0, q84 - q50)],
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            alpha=0.8,
            label="posterior 1 sigma",
        )
    else:
        ax.scatter(truth, recovered, color="tab:blue", label="best fit")
    finite = np.concatenate([truth[np.isfinite(truth)], plotted[np.isfinite(plotted)]])
    if finite.size:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_xlabel("true signed magnification")
    ax.set_ylabel("recovered signed magnification")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_image_recovery(image_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.scatter(image_df["x_obs_arcsec"], image_df["y_obs_arcsec"], color="black", s=22, label="observed")
    if {"x_model_q16", "x_model_q50", "x_model_q84", "y_model_q16", "y_model_q50", "y_model_q84"}.issubset(image_df.columns):
        x_model = image_df["x_model_q50"].to_numpy(dtype=float)
        y_model = image_df["y_model_q50"].to_numpy(dtype=float)
        ax.errorbar(
            x_model,
            y_model,
            xerr=[
                np.maximum(0.0, x_model - image_df["x_model_q16"].to_numpy(dtype=float)),
                np.maximum(0.0, image_df["x_model_q84"].to_numpy(dtype=float) - x_model),
            ],
            yerr=[
                np.maximum(0.0, y_model - image_df["y_model_q16"].to_numpy(dtype=float)),
                np.maximum(0.0, image_df["y_model_q84"].to_numpy(dtype=float) - y_model),
            ],
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            markersize=4,
            label="model 1 sigma",
        )
    else:
        x_model = image_df["x_model_arcsec"].to_numpy(dtype=float)
        y_model = image_df["y_model_arcsec"].to_numpy(dtype=float)
        ax.scatter(image_df["x_model_arcsec"], image_df["y_model_arcsec"], color="tab:blue", s=18, label="model")
    for row, x_fit, y_fit in zip(image_df.itertuples(index=False), x_model, y_model):
        if np.isfinite(x_fit) and np.isfinite(y_fit):
            ax.plot([row.x_obs_arcsec, x_fit], [row.y_obs_arcsec, y_fit], color="0.6", lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_title("Image positions")
    ax.legend(loc="best", fontsize=8)

    residual = (
        image_df["image_residual_q50"].to_numpy(dtype=float)
        if "image_residual_q50" in image_df
        else image_df["image_residual_arcsec"].to_numpy(dtype=float)
    )
    x_index = np.arange(len(image_df))
    if {"image_residual_q16", "image_residual_q84"}.issubset(image_df.columns):
        axes[1].errorbar(
            x_index,
            residual,
            yerr=[
                np.maximum(0.0, residual - image_df["image_residual_q16"].to_numpy(dtype=float)),
                np.maximum(0.0, image_df["image_residual_q84"].to_numpy(dtype=float) - residual),
            ],
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
        )
    else:
        axes[1].scatter(x_index, residual, color="tab:blue")
    axes[1].set_xlabel("image index")
    axes[1].set_ylabel("image residual [arcsec]")
    axes[1].set_title("Image residuals with 1 sigma intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_source_recovery(source_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    if {"beta_x", "beta_y"}.issubset(source_df.columns):
        ax.scatter(source_df["beta_x"], source_df["beta_y"], color="black", s=28, label="truth")
    if {"source_x_q16", "source_x_q50", "source_x_q84", "source_y_q16", "source_y_q50", "source_y_q84"}.issubset(source_df.columns):
        sx = source_df["source_x_q50"].to_numpy(dtype=float)
        sy = source_df["source_y_q50"].to_numpy(dtype=float)
        ax.errorbar(
            sx,
            sy,
            xerr=[
                np.maximum(0.0, sx - source_df["source_x_q16"].to_numpy(dtype=float)),
                np.maximum(0.0, source_df["source_x_q84"].to_numpy(dtype=float) - sx),
            ],
            yerr=[
                np.maximum(0.0, sy - source_df["source_y_q16"].to_numpy(dtype=float)),
                np.maximum(0.0, source_df["source_y_q84"].to_numpy(dtype=float) - sy),
            ],
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            markersize=5,
            label="recovered 1 sigma",
        )
    else:
        sx = source_df["source_x_recovered"].to_numpy(dtype=float)
        sy = source_df["source_y_recovered"].to_numpy(dtype=float)
        ax.scatter(source_df["source_x_recovered"], source_df["source_y_recovered"], color="tab:blue", s=24, label="recovered")
    if {"beta_x", "beta_y"}.issubset(source_df.columns):
        for row, sx_fit, sy_fit in zip(source_df.itertuples(index=False), sx, sy):
            if np.isfinite(sx_fit) and np.isfinite(sy_fit):
                ax.plot([row.beta_x, sx_fit], [row.beta_y, sy_fit], color="0.6", lw=0.8)
    ax.set_xlabel(r"$\beta_x$ [arcsec]")
    ax.set_ylabel(r"$\beta_y$ [arcsec]")
    ax.set_title("Source positions")
    ax.legend(loc="best", fontsize=8)

    if "source_position_error_arcsec" in source_df:
        values = (
            source_df["source_position_error_q50"].to_numpy(dtype=float)
            if "source_position_error_q50" in source_df
            else source_df["source_position_error_arcsec"].to_numpy(dtype=float)
        )
        x_index = np.arange(len(source_df))
        if {"source_position_error_q16", "source_position_error_q84"}.issubset(source_df.columns):
            axes[1].errorbar(
                x_index,
                values,
                yerr=[
                    np.maximum(0.0, values - source_df["source_position_error_q16"].to_numpy(dtype=float)),
                    np.maximum(0.0, source_df["source_position_error_q84"].to_numpy(dtype=float) - values),
                ],
                fmt="o",
                color="tab:blue",
                ecolor="tab:blue",
            )
        else:
            axes[1].scatter(x_index, values, color="tab:blue")
        axes[1].set_xlabel("family index")
        axes[1].set_ylabel("source position error [arcsec]")
    else:
        values = (
            source_df["source_plane_rms_q50"].to_numpy(dtype=float)
            if "source_plane_rms_q50" in source_df
            else source_df["source_plane_rms_arcsec"].to_numpy(dtype=float)
        )
        x_index = np.arange(len(source_df))
        if {"source_plane_rms_q16", "source_plane_rms_q84"}.issubset(source_df.columns):
            axes[1].errorbar(
                x_index,
                values,
                yerr=[
                    np.maximum(0.0, values - source_df["source_plane_rms_q16"].to_numpy(dtype=float)),
                    np.maximum(0.0, source_df["source_plane_rms_q84"].to_numpy(dtype=float) - values),
                ],
                fmt="o",
                color="tab:blue",
                ecolor="tab:blue",
            )
        else:
            axes[1].scatter(x_index, values, color="tab:blue")
        axes[1].set_xlabel("family index")
        axes[1].set_ylabel("source-plane RMS [arcsec]")
    axes[1].set_title("Source recovery with 1 sigma intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_subhalo_population(
    subhalo_df: pd.DataFrame,
    images: pd.DataFrame,
    parameter_df: pd.DataFrame,
    path: Path,
) -> None:
    del parameter_df
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(images["x_obs_arcsec"], images["y_obs_arcsec"], color="black", marker="x", s=26, label="images")
    if not subhalo_df.empty:
        sizes = 12.0 + 80.0 * np.sqrt(subhalo_df["luminosity_ratio"].to_numpy(dtype=float))
        scatter = ax.scatter(
            subhalo_df["x_arcsec"],
            subhalo_df["y_arcsec"],
            s=sizes,
            c=subhalo_df["catalog_mag"],
            cmap="viridis_r",
            alpha=0.75,
            label="subhalos",
        )
        fig.colorbar(scatter, ax=ax, label="member magnitude")
    ax.scatter([0.0], [0.0], color="tab:red", marker="+", s=80, label="BCG")
    ax.invert_xaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_title("Subhalo field")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _select_critical_caustic_plot_contours(
    contours_by_z: dict[str, list[CausticContour]],
) -> dict[str, list[CausticContour]]:
    selected: dict[str, list[CausticContour]] = {}
    for z_key, contours in contours_by_z.items():
        try:
            z_source = float(z_key)
        except (TypeError, ValueError):
            continue
        if abs(z_source - CRITICAL_CAUSTIC_RECOVERY_SOURCE_REDSHIFT) <= CRITICAL_CAUSTIC_RECOVERY_REDSHIFT_TOL and contours:
            selected[str(z_key)] = contours
    return selected


def _plot_critical_caustic_recovery(
    truth_contours_by_z: dict[str, list[CausticContour]],
    recovered_contours_by_z: dict[str, list[CausticContour]],
    images: pd.DataFrame,
    image_df: pd.DataFrame,
    source_df: pd.DataFrame,
    subhalo_df: pd.DataFrame,
    path: Path,
) -> None:
    truth_contours_by_z = _select_critical_caustic_plot_contours(truth_contours_by_z)
    recovered_contours_by_z = _select_critical_caustic_plot_contours(recovered_contours_by_z)
    if not truth_contours_by_z or not recovered_contours_by_z:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    image_ax, source_ax = axes

    truth_line_labeled = False
    recovered_line_labeled = False
    for contours in truth_contours_by_z.values():
        for contour in contours:
            image_ax.plot(
                contour.critical_x,
                contour.critical_y,
                color="black",
                lw=0.8,
                alpha=0.75,
                label="truth critical line" if not truth_line_labeled else None,
            )
            truth_line_labeled = True
    for contours in recovered_contours_by_z.values():
        for contour in contours:
            image_ax.plot(
                contour.critical_x,
                contour.critical_y,
                color="tab:blue",
                lw=0.8,
                linestyle="--",
                alpha=0.85,
                label="recovered critical line" if not recovered_line_labeled else None,
            )
            recovered_line_labeled = True

    image_ax.invert_xaxis()
    image_ax.set_aspect("equal", adjustable="box")
    image_ax.set_xlabel("x [arcsec]")
    image_ax.set_ylabel("y [arcsec]")
    image_ax.set_title(r"Image plane, $z_s=7$")
    image_ax.legend(loc="best", fontsize=7)

    truth_caustic_labeled = False
    recovered_caustic_labeled = False
    for contours in truth_contours_by_z.values():
        for contour in contours:
            source_ax.scatter(
                contour.beta_x,
                contour.beta_y,
                color="black",
                s=2.0,
                alpha=0.55,
                linewidths=0.0,
                label="truth caustic" if not truth_caustic_labeled else None,
            )
            truth_caustic_labeled = True
    for contours in recovered_contours_by_z.values():
        for contour in contours:
            source_ax.scatter(
                contour.beta_x,
                contour.beta_y,
                color="tab:blue",
                s=2.0,
                alpha=0.65,
                linewidths=0.0,
                label="recovered caustic" if not recovered_caustic_labeled else None,
            )
            recovered_caustic_labeled = True

    source_ax.set_aspect("equal", adjustable="box")
    source_ax.set_xlabel(r"$\beta_x$ [arcsec]")
    source_ax.set_ylabel(r"$\beta_y$ [arcsec]")
    source_ax.set_title(r"Source plane, $z_s=7$")
    source_ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_validation_summary(summary: dict[str, float], uncertainty: dict[str, tuple[float, float]], path: Path) -> None:
    labels = [
        "median image residual",
        "median source error",
        "median |mu| frac. error",
        "parameter 1 sigma coverage",
        "parity match fraction",
    ]
    values = [
        summary["median_image_residual_arcsec"],
        summary["median_source_position_error_arcsec"],
        summary["median_abs_magnification_frac_error"],
        summary["parameter_coverage_68_fraction"],
        summary["parity_match_fraction"],
    ]
    keys = [
        "median_image_residual_arcsec",
        "median_source_position_error_arcsec",
        "median_abs_magnification_frac_error",
        "parameter_coverage_68_fraction",
        "parity_match_fraction",
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(labels))
    ax.barh(y, values, color=["tab:blue", "tab:cyan", "tab:purple", "tab:green", "tab:orange"], alpha=0.85)
    for idx, (key, value) in enumerate(zip(keys, values)):
        low, high = uncertainty.get(key, (np.nan, np.nan))
        if np.isfinite(value) and np.isfinite(low) and np.isfinite(high):
            ax.errorbar(
                value,
                idx,
                xerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
            )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("metric value")
    ax.set_title("Mock recovery summary with 1 sigma intervals")
    for idx, value in enumerate(values):
        if np.isfinite(value):
            ax.text(value, idx, f" {value:.3g}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _validation_stage_arg_values(value: Any, *, flag_name: str) -> list[Any]:
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if not values:
        raise SystemExit(f"{flag_name} requires one to three values.")
    if len(values) > 3:
        raise SystemExit(f"{flag_name} accepts at most three values: stage 2, stage 3, and stage 4.")
    return values


def _validation_linearized_stage_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE)) == IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA


def _validation_marginal_image_plane_stage_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE)) == IMAGE_PLANE_MODE_MARGINAL


def _validation_stage4_enabled(args: argparse.Namespace) -> bool:
    return _validation_linearized_stage_enabled(args) or _validation_marginal_image_plane_stage_enabled(args)


def _normalize_validation_stage_fit_controls(args: argparse.Namespace) -> dict[str, ValidationStageFitControls]:
    solver_fit_mode = str(getattr(args, "solver_fit_mode", SOLVER_FIT_MODE_SEQUENTIAL))
    mode = str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE))
    for attr_name, flag_name in (
        ("jax_default_device", "--jax-default-device"),
        ("smc_device", "--smc-device"),
    ):
        if str(getattr(args, attr_name, JAX_DEVICE_AUTO)) not in JAX_DEVICE_CHOICES:
            raise SystemExit(f"{flag_name} must be one of {', '.join(JAX_DEVICE_CHOICES)}.")
    if bool(getattr(args, "resume_fast", False)) and solver_fit_mode != SOLVER_FIT_MODE_SEQUENTIAL:
        raise SystemExit("--resume-fast is only valid with --solver-fit-mode sequential.")
    ns_num_live_points = getattr(args, "ns_num_live_points", None)
    if ns_num_live_points is not None and int(ns_num_live_points) <= 0:
        raise SystemExit("--ns-num-live-points must be positive when provided.")
    ns_max_samples = getattr(args, "ns_max_samples", None)
    if ns_max_samples is not None:
        try:
            ns_max_samples_int = int(ns_max_samples)
        except (TypeError, ValueError) as exc:
            raise SystemExit("--ns-max-samples must be a positive integer or 'none'.") from exc
        if ns_max_samples_int <= 0:
            raise SystemExit("--ns-max-samples must be positive.")
    if float(getattr(args, "ns_dlogz", 1.0e-4)) <= 0.0:
        raise SystemExit("--ns-dlogz must be positive.")
    if int(getattr(args, "smc_particles", DEFAULT_SMC_PARTICLES)) <= 0:
        raise SystemExit("--smc-particles must be positive.")
    if str(getattr(args, "smc_mcmc_kernel", DEFAULT_SMC_MCMC_KERNEL)) not in SMC_MCMC_KERNELS:
        raise SystemExit(f"--smc-mcmc-kernel must be one of {', '.join(SMC_MCMC_KERNELS)}.")
    if int(getattr(args, "smc_mcmc_steps", DEFAULT_SMC_MCMC_STEPS)) <= 0:
        raise SystemExit("--smc-mcmc-steps must be positive.")
    smc_target_ess_frac = float(getattr(args, "smc_target_ess_frac", DEFAULT_SMC_TARGET_ESS_FRAC))
    if not np.isfinite(smc_target_ess_frac) or smc_target_ess_frac <= 0.0 or smc_target_ess_frac > 1.0:
        raise SystemExit("--smc-target-ess-frac must be in (0, 1].")
    if int(getattr(args, "smc_max_temperature_steps", DEFAULT_SMC_MAX_TEMPERATURE_STEPS)) <= 0:
        raise SystemExit("--smc-max-temperature-steps must be positive.")
    if (
        not np.isfinite(float(getattr(args, "smc_rmh_scale", DEFAULT_SMC_RMH_SCALE)))
        or float(getattr(args, "smc_rmh_scale", DEFAULT_SMC_RMH_SCALE)) <= 0.0
    ):
        raise SystemExit("--smc-rmh-scale must be positive.")
    if (
        not np.isfinite(float(getattr(args, "smc_mala_step_size", DEFAULT_SMC_MALA_STEP_SIZE)))
        or float(getattr(args, "smc_mala_step_size", DEFAULT_SMC_MALA_STEP_SIZE)) <= 0.0
    ):
        raise SystemExit("--smc-mala-step-size must be positive.")

    evidence_prior_sigma = getattr(args, "evidence_source_prior_sigma_arcsec", None)
    if evidence_prior_sigma is not None and float(evidence_prior_sigma) <= 0.0:
        raise SystemExit("--evidence-source-prior-sigma-arcsec must be positive.")
    evidence_likelihood_mode = str(
        getattr(args, "evidence_likelihood_mode", DEFAULT_EVIDENCE_LIKELIHOOD_MODE)
    )
    if evidence_likelihood_mode not in EVIDENCE_LIKELIHOOD_MODES:
        raise SystemExit(
            "--evidence-likelihood-mode must be one of "
            f"{', '.join(EVIDENCE_LIKELIHOOD_MODES)}."
        )
    max_tree_depths = [
        int(value)
        for value in _validation_stage_arg_values(
            getattr(args, "max_tree_depth", 8),
            flag_name="--max-tree-depth",
        )
    ]
    if any(value < 0 for value in max_tree_depths):
        raise SystemExit("--max-tree-depth values must be non-negative.")
    if float(getattr(args, "image_plane_scatter_upper_arcsec", DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC)) <= 0.0:
        raise SystemExit("--image-plane-scatter-upper-arcsec must be positive.")
    if (
        not np.isfinite(float(getattr(args, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)))
        or float(getattr(args, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)) < 0.0
    ):
        raise SystemExit("--image-plane-scatter-floor-arcsec must be non-negative.")
    if str(getattr(args, "image_plane_scatter_prior", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR)) not in IMAGE_PLANE_SCATTER_PRIORS:
        raise SystemExit(
            "--image-plane-scatter-prior must be one of "
            f"{', '.join(IMAGE_PLANE_SCATTER_PRIORS)}."
        )
    if (
        not np.isfinite(float(getattr(args, "image_plane_scatter_prior_median_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC)))
        or float(getattr(args, "image_plane_scatter_prior_median_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC)) <= 0.0
    ):
        raise SystemExit("--image-plane-scatter-prior-median-arcsec must be positive.")
    if (
        not np.isfinite(float(getattr(args, "image_plane_scatter_prior_log_sigma", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA)))
        or float(getattr(args, "image_plane_scatter_prior_log_sigma", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA)) <= 0.0
    ):
        raise SystemExit("--image-plane-scatter-prior-log-sigma must be positive.")
    image_presence_penalty_weight = getattr(args, "image_presence_penalty_weight", None)
    if image_presence_penalty_weight is not None and (
        not np.isfinite(float(image_presence_penalty_weight)) or float(image_presence_penalty_weight) < 0.0
    ):
        raise SystemExit("--image-presence-penalty-weight must be non-negative when provided.")
    if (
        not np.isfinite(float(getattr(args, "image_presence_match_radius_arcsec", DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC)))
        or float(getattr(args, "image_presence_match_radius_arcsec", DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC)) <= 0.0
    ):
        raise SystemExit("--image-presence-match-radius-arcsec must be positive.")
    if (
        not np.isfinite(float(getattr(args, "image_presence_temperature_arcsec", DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)))
        or float(getattr(args, "image_presence_temperature_arcsec", DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)) <= 0.0
    ):
        raise SystemExit("--image-presence-temperature-arcsec must be positive.")
    if (
        not np.isfinite(float(getattr(args, "image_presence_count_softness", DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS)))
        or float(getattr(args, "image_presence_count_softness", DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS)) <= 0.0
    ):
        raise SystemExit("--image-presence-count-softness must be positive.")
    if (
        not np.isfinite(float(getattr(args, "image_presence_count_margin", DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN)))
        or float(getattr(args, "image_presence_count_margin", DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN)) < 0.0
    ):
        raise SystemExit("--image-presence-count-margin must be non-negative.")
    if (
        not np.isfinite(float(getattr(args, "likelihood_stabilizer_max_gain", DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN)))
        or float(getattr(args, "likelihood_stabilizer_max_gain", DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN)) < 0.0
    ):
        raise SystemExit("--likelihood-stabilizer-max-gain must be non-negative.")
    if (
        not np.isfinite(float(getattr(args, "likelihood_stabilizer_max_residual_arcsec", DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC)))
        or float(getattr(args, "likelihood_stabilizer_max_residual_arcsec", DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC)) < 0.0
    ):
        raise SystemExit("--likelihood-stabilizer-max-residual-arcsec must be non-negative.")
    if str(getattr(args, "likelihood_stabilizer_residual_loss", DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS)) not in LIKELIHOOD_STABILIZER_RESIDUAL_LOSSES:
        raise SystemExit(
            "--likelihood-stabilizer-residual-loss must be one of "
            f"{', '.join(LIKELIHOOD_STABILIZER_RESIDUAL_LOSSES)}."
        )
    if (
        not np.isfinite(float(getattr(args, "likelihood_stabilizer_student_t_nu", DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU)))
        or float(getattr(args, "likelihood_stabilizer_student_t_nu", DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU)) <= 0.0
    ):
        raise SystemExit("--likelihood-stabilizer-student-t-nu must be positive.")
    if bool(getattr(args, "fit_cosmology_all_stages", False)) and not bool(getattr(args, "fit_cosmology_flat_wcdm", False)):
        raise SystemExit("--fit-cosmology-all-stages requires --fit-cosmology-flat-wcdm.")
    if solver_fit_mode == SOLVER_FIT_MODE_EVIDENCE_NS:
        sampled_source_evidence = (
            evidence_likelihood_mode == EVIDENCE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
        )
        if evidence_prior_sigma is None:
            raise SystemExit("--solver-fit-mode evidence-ns requires --evidence-source-prior-sigma-arcsec.")
        if mode != IMAGE_PLANE_MODE_NONE:
            raise SystemExit("--solver-fit-mode evidence-ns requires --image-plane-mode none.")
        if str(getattr(args, "sampling_engine", "full")) == "active_subset":
            raise SystemExit("--sampling-engine active_subset is not valid with --solver-fit-mode evidence-ns.")
        if bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)):
            raise SystemExit("--skip-stage3-image-plane-local-jacobian is not valid with --solver-fit-mode evidence-ns.")
        if int(getattr(args, "image_plane_newton_steps", 0)) != 0 and not sampled_source_evidence:
            raise SystemExit(
                "--image-plane-newton-steps is only valid with --solver-fit-mode evidence-ns "
                "--evidence-likelihood-mode linearized-forward-beta-image-plane."
            )
        if (
            sampled_source_evidence
            and str(getattr(args, "sampling_engine", "full")) == "refreshing_surrogate"
            and int(getattr(args, "image_plane_newton_steps", 0)) > 0
        ):
            raise SystemExit(
                "--sampling-engine refreshing_surrogate with linearized-forward-beta-image-plane "
                "requires --image-plane-newton-steps 0."
            )
        if (
            str(getattr(args, "source_position_parameterization", "prior-whitened")) != "prior-whitened"
            and not sampled_source_evidence
        ):
            raise SystemExit(
                "--source-position-parameterization is only valid with --solver-fit-mode evidence-ns "
                "--evidence-likelihood-mode linearized-forward-beta-image-plane."
            )
        controls = {
            "stage2": ValidationStageFitControls(
                fit_method=FIT_METHOD_NS,
                warmup=0,
                samples=0,
                max_tree_depth=int(max_tree_depths[0]),
            ),
            "stage3": ValidationStageFitControls(
                fit_method=FIT_METHOD_NS,
                warmup=0,
                samples=0,
                max_tree_depth=int(max_tree_depths[0]),
            ),
            "stage4": ValidationStageFitControls(
                fit_method=FIT_METHOD_NS,
                warmup=0,
                samples=0,
                max_tree_depth=int(max_tree_depths[0]),
            ),
        }
        return controls
    if evidence_likelihood_mode != DEFAULT_EVIDENCE_LIKELIHOOD_MODE:
        raise SystemExit("--evidence-likelihood-mode is only valid with --solver-fit-mode evidence-ns.")

    fit_methods = [
        str(value)
        for value in _validation_stage_arg_values(
            getattr(args, "fit_method", FIT_METHOD_SVI_NUTS),
            flag_name="--fit-method",
        )
    ]
    warmups = [
        int(value)
        for value in _validation_stage_arg_values(
            getattr(args, "warmup", 300),
            flag_name="--warmup",
        )
    ]
    samples = [
        int(value)
        for value in _validation_stage_arg_values(
            getattr(args, "samples", 500),
            flag_name="--samples",
        )
    ]

    invalid_fit_methods = sorted(
        set(fit_methods).difference({FIT_METHOD_SVI, FIT_METHOD_SVI_NUTS, FIT_METHOD_NS, FIT_METHOD_SMC})
    )
    if invalid_fit_methods:
        raise SystemExit(f"--fit-method has unsupported value(s): {', '.join(invalid_fit_methods)}")
    if any(value == FIT_METHOD_NS for value in fit_methods):
        raise SystemExit("--fit-method ns is only valid with --solver-fit-mode evidence-ns.")
    if any(value < 0 for value in warmups):
        raise SystemExit("--warmup values must be non-negative.")
    if any(value <= 0 for value in samples):
        raise SystemExit("--samples values must be positive.")

    max_value_count = max(len(fit_methods), len(warmups), len(samples), len(max_tree_depths))
    has_stage_specific_values = max_value_count >= 2
    has_three_stage_values = max_value_count == 3
    has_stage3_or_stage4 = mode in {
        IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        IMAGE_PLANE_MODE_MARGINAL,
    }
    has_stage4 = _validation_stage4_enabled(args)
    stage3_active = mode == IMAGE_PLANE_MODE_LOCAL_JACOBIAN or (
        mode in {IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA, IMAGE_PLANE_MODE_MARGINAL}
        and not bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False))
    )
    if bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)) and not has_stage4:
        raise SystemExit(
            "--skip-stage3-image-plane-local-jacobian is only valid with a final stage-4 image-plane mode."
        )
    if (
        _validation_linearized_stage_enabled(args)
        and str(getattr(args, "sampling_engine", "full")) == "refreshing_surrogate"
        and int(getattr(args, "image_plane_newton_steps", 0)) > 0
    ):
        raise SystemExit(
            "--sampling-engine refreshing_surrogate with linearized-forward-beta-image-plane "
            "requires --image-plane-newton-steps 0."
        )
    if has_stage_specific_values and not has_stage3_or_stage4:
        raise SystemExit(
            "Two-value --fit-method, --warmup, --samples, or --max-tree-depth is only valid with "
            "an image-plane mode."
        )
    if has_three_stage_values and not has_stage4:
        raise SystemExit(
            "Three-value --fit-method, --warmup, --samples, or --max-tree-depth is only valid with "
            "a final stage-4 image-plane mode."
        )
    if float(getattr(args, "linearized_beta_prior_sigma_arcsec", DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC)) <= 0.0:
        raise SystemExit("--linearized-beta-prior-sigma-arcsec must be positive.")
    def stage_value(values: list[Any], index: int) -> Any:
        return values[index] if len(values) > index else values[0]

    def stage4_value(values: list[Any]) -> Any:
        if len(values) > 2:
            return values[2]
        if bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)) and len(values) > 1:
            return values[1]
        if len(values) > 1:
            return values[1]
        return values[0]

    controls = {
        "stage2": ValidationStageFitControls(
            fit_method=str(stage_value(fit_methods, 0)),
            warmup=int(stage_value(warmups, 0)),
            samples=int(stage_value(samples, 0)),
            max_tree_depth=int(stage_value(max_tree_depths, 0)),
        ),
        "stage3": ValidationStageFitControls(
            fit_method=str(stage_value(fit_methods, 1)),
            warmup=int(stage_value(warmups, 1)),
            samples=int(stage_value(samples, 1)),
            max_tree_depth=int(stage_value(max_tree_depths, 1)),
        ),
        "stage4": ValidationStageFitControls(
            fit_method=str(stage4_value(fit_methods)),
            warmup=int(stage4_value(warmups)),
            samples=int(stage4_value(samples)),
            max_tree_depth=int(stage4_value(max_tree_depths)),
        ),
    }
    smc_stages: list[str] = []
    if controls["stage2"].fit_method == FIT_METHOD_SMC:
        smc_stages.append("stage2")
    if stage3_active and controls["stage3"].fit_method == FIT_METHOD_SMC:
        smc_stages.append("stage3")
    if has_stage4 and controls["stage4"].fit_method == FIT_METHOD_SMC:
        smc_stages.append("stage4")
    if smc_stages:
        if smc_stages != ["stage4"] or not _validation_linearized_stage_enabled(args):
            raise SystemExit("--fit-method smc is only valid for explicit-beta sequential stage 4.")
    return controls


def _append_stage_option(cmd: list[str], option: str, values: Any) -> None:
    cmd.append(option)
    cmd.extend(str(value) for value in _validation_stage_arg_values(values, flag_name=option))


def _validation_root(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / "single_bcg" / str(args.run_name)


def _validation_final_stage_name(args: argparse.Namespace) -> str:
    if str(getattr(args, "solver_fit_mode", SOLVER_FIT_MODE_SEQUENTIAL)) == SOLVER_FIT_MODE_EVIDENCE_NS:
        return "fit"
    if _validation_linearized_stage_enabled(args):
        return "stage4_linearized_image_plane"
    if _validation_marginal_image_plane_stage_enabled(args):
        return "stage4_marginal_image_plane"
    if str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE)) == IMAGE_PLANE_MODE_LOCAL_JACOBIAN:
        return "stage3_image_plane"
    return "stage2_joint"


def _validation_mock_paths(mock_dir: str | Path) -> MockClusterPaths:
    root = Path(mock_dir)
    return MockClusterPaths(
        root=root,
        par_path=root / "single_bcg_mock.par",
        image_catalog_path=root / "obs_arcs.cat",
        truth_path=root / "truth.json",
        mock_images_path=root / "mock_images.json",
    )


def _validation_mock_complete(paths: MockClusterPaths) -> bool:
    return paths.par_path.exists() and paths.truth_path.exists() and paths.mock_images_path.exists()


def _load_existing_single_bcg_mock(mock_dir: str | Path) -> tuple[MockClusterPaths, pd.DataFrame, dict[str, Any]]:
    paths = _validation_mock_paths(mock_dir)
    if not _validation_mock_complete(paths):
        raise FileNotFoundError(f"Cannot resume; mock inputs are incomplete under {paths.root}")
    images = pd.DataFrame(json.loads(paths.mock_images_path.read_text(encoding="utf-8")))
    truth = _load_truth(paths.truth_path)
    return paths, images, truth


def _validation_recovery_output_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "corner_plot": root / "corner.pdf",
        "potfile_corner_plot": root / "potfile_corner.pdf",
        "parameter_recovery_log_plot": root / "parameter_recovery_log.pdf",
        "parameter_recovery_linear_plot": root / "parameter_recovery_linear.pdf",
        "magnification_plot": root / "magnification_recovery.pdf",
        "image_recovery_plot": root / "image_recovery.pdf",
        "source_recovery_plot": root / "source_recovery.pdf",
        "subhalo_population_plot": root / "subhalo_population.pdf",
        "summary_plot": root / "validation_summary.pdf",
    }


def _validation_realization_complete(realization_dir: str | Path) -> bool:
    root = Path(realization_dir)
    if not (root / "run_summary.txt").exists():
        return False
    return all(path.exists() for path in _validation_recovery_output_paths(root).values())


def _validation_stage_has_recovery_artifacts(run_dir: str | Path) -> bool:
    artifacts_dir = Path(run_dir) / "artifacts"
    return (artifacts_dir / "plot_bundle.h5").exists() or (artifacts_dir / "posterior_arrays.npz").exists()


VALIDATION_RESULTS_SCHEMA_VERSION = 1


def _read_text_file_optional(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _read_json_file_optional(path: str | Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_table_artifact(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": path,
        "name": path.name,
        "suffix": path.suffix.lower(),
    }
    try:
        if path.suffix.lower() == ".json":
            payload["data"] = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".csv":
            payload["data"] = _validation_dataframe_payload(pd.read_csv(path))
        elif path.suffix.lower() == ".txt":
            payload["text"] = path.read_text(encoding="utf-8")
        else:
            payload["text"] = path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive bundle fallback
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _collect_stage_table_artifacts(stage_dir: str | Path) -> dict[str, dict[str, Any]]:
    tables_dir = Path(stage_dir) / "tables"
    if not tables_dir.exists():
        return {}
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(tables_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".txt"}:
            artifacts[path.name] = _read_table_artifact(path)
    return artifacts


def _validation_stage_manifest(solver_run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(solver_run_dir)
    root = _stage_root_from_run_dir(run_dir)
    candidates: list[tuple[str, Path]] = []
    if run_dir.name in _VALIDATION_STAGE_ORDER:
        for stage_name in _VALIDATION_STAGE_ORDER:
            stage_dir = root / stage_name
            if stage_dir.exists() or stage_dir == run_dir:
                candidates.append((stage_name, stage_dir))
    else:
        candidates.append((root.name, root))

    manifests: list[dict[str, Any]] = []
    for stage_name, stage_dir in candidates:
        artifacts_dir = stage_dir / "artifacts"
        tables_dir = stage_dir / "tables"
        manifests.append(
            {
                "stage": stage_name,
                "stage_dir": stage_dir,
                "exists": stage_dir.exists(),
                "artifacts_dir": artifacts_dir,
                "tables_dir": tables_dir,
                "has_artifacts_dir": artifacts_dir.exists(),
                "has_tables_dir": tables_dir.exists(),
                "has_plot_bundle": (artifacts_dir / "plot_bundle.h5").exists(),
                "has_posterior_arrays": (artifacts_dir / "posterior_arrays.npz").exists(),
                "run_summary_path": tables_dir / "run_summary.json",
                "run_summary": _load_stage_run_summary(stage_dir) if (tables_dir / "run_summary.json").exists() else None,
                "table_artifacts": _collect_stage_table_artifacts(stage_dir),
            }
        )
    return manifests


def _mock_input_payload(paths: MockClusterPaths, images: pd.DataFrame, truth_payload: dict[str, Any]) -> dict[str, Any]:
    member_catalog_path = paths.root / "members.cat"
    truth = _read_json_file_optional(paths.truth_path)
    if not isinstance(truth, dict):
        truth = truth_payload if isinstance(truth_payload, dict) else {}
    mock_images = _read_json_file_optional(paths.mock_images_path)
    if mock_images is None:
        mock_images = images.to_dict(orient="records")
    return {
        "paths": {
            "root": paths.root,
            "par_path": paths.par_path,
            "image_catalog_path": paths.image_catalog_path,
            "truth_path": paths.truth_path,
            "mock_images_path": paths.mock_images_path,
            "member_catalog_path": member_catalog_path if member_catalog_path.exists() else None,
        },
        "files": {
            "par_text": _read_text_file_optional(paths.par_path),
            "image_catalog_text": _read_text_file_optional(paths.image_catalog_path),
            "member_catalog_text": _read_text_file_optional(member_catalog_path) if member_catalog_path.exists() else None,
        },
        "truth": truth,
        "mock_images": mock_images,
        "images": images.to_dict(orient="records"),
        "sources": truth.get("sources", []),
        "subhalos": truth.get("subhalos", []),
        "parameter_truth": truth.get("parameter_truth", {}),
    }


def write_validation_results_json(
    *,
    args: argparse.Namespace,
    seed: int,
    realization_dir: str | Path,
    config: SingleBCGMockConfig,
    paths: MockClusterPaths,
    images: pd.DataFrame,
    truth_payload: dict[str, Any],
    solver_run_dir: str | Path,
    summary_path: str | Path,
    output_paths: dict[str, Path],
    recovery_payload: dict[str, Any],
    stage3_recovery_payload: dict[str, Any] | None = None,
    controls: dict[str, ValidationStageFitControls] | None = None,
) -> Path:
    root = _validation_root(args)
    output_path = root / f"seed_{int(seed)}_results.json"
    solver_run_path = Path(solver_run_dir)
    solver_root = _stage_root_from_run_dir(solver_run_path)
    debug_log_path = root / "run_debug.log"
    sequential_summary_path = solver_root / "sequential_summary.json"
    all_output_paths = dict(output_paths)
    all_output_paths["results_json"] = output_path
    try:
        stage_recovery_metrics: Any = _collect_validation_stage_recovery_metrics(solver_run_path, paths.truth_path)
    except Exception as exc:  # pragma: no cover - defensive bundle fallback
        stage_recovery_metrics = {"error": f"{type(exc).__name__}: {exc}"}
    payload = {
        "schema_version": VALIDATION_RESULTS_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "run": {
            "run_name": str(args.run_name),
            "mock": str(getattr(args, "mock", "single-bcg")),
            "realization_dir": Path(realization_dir),
            "validation_root": root,
            "cwd": Path.cwd(),
            "executable": sys.executable,
            "argv": list(sys.argv),
            "args": vars(args),
            "stage_controls": controls or {},
        },
        "mock_cluster": {
            "config": config,
            **_mock_input_payload(paths, images, truth_payload),
        },
        "solver": {
            "run_name": "fit",
            "root_dir": solver_root,
            "final_run_dir": solver_run_path,
            "final_stage": solver_run_path.name,
            "sequential_summary_path": sequential_summary_path,
            "sequential_summary": _read_json_file_optional(sequential_summary_path),
            "stage_manifests": _validation_stage_manifest(solver_run_path),
        },
        "validation": {
            "output_paths": all_output_paths,
            "run_summary": {
                "path": Path(summary_path),
                "text": _read_text_file_optional(summary_path),
            },
            "stage_recovery_metrics": stage_recovery_metrics,
            "recovery": {
                "final": recovery_payload,
                "stage3": stage3_recovery_payload,
            },
        },
        "debug_log": {
            "path": debug_log_path,
            "text": _read_text_file_optional(debug_log_path),
        },
    }
    return _write_strict_json(output_path, payload)


def _format_stage_controls_for_log(controls: dict[str, ValidationStageFitControls]) -> str:
    return (
        f"stage2={controls['stage2'].fit_method}/warmup={controls['stage2'].warmup}/"
        f"samples={controls['stage2'].samples}/max_tree_depth={controls['stage2'].max_tree_depth} "
        f"stage3={controls['stage3'].fit_method}/warmup={controls['stage3'].warmup}/"
        f"samples={controls['stage3'].samples}/max_tree_depth={controls['stage3'].max_tree_depth} "
        f"stage4={controls['stage4'].fit_method}/warmup={controls['stage4'].warmup}/"
        f"samples={controls['stage4'].samples}/max_tree_depth={controls['stage4'].max_tree_depth}"
    )


def _finite_active_scaling_values(values: Any) -> list[int]:
    if values is None:
        return []
    raw_values = values if isinstance(values, (list, tuple)) else [values]
    finite_values: list[int] = []
    for value in raw_values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            finite_values.append(parsed)
    return finite_values


def _validation_configured_approximation_items(args: argparse.Namespace) -> list[str]:
    items: list[str] = []
    sampling_engine = str(getattr(args, "sampling_engine", "full"))
    if sampling_engine == "refreshing_surrogate":
        items.append("refreshing_surrogate=configured first-order inactive-deflection surrogate")
    elif sampling_engine == "active_subset":
        items.append("active_subset=configured inactive scaling potentials omitted during solver fitting")
    try:
        z_bin_tol = float(getattr(args, "z_bin_efficiency_tol", 0.0))
    except (TypeError, ValueError):
        z_bin_tol = 0.0
    if z_bin_tol > 0.0:
        items.append(f"z_bins=configured lensing-efficiency grouping tol={z_bin_tol:.4g}")

    solver_fit_mode = str(getattr(args, "solver_fit_mode", SOLVER_FIT_MODE_SEQUENTIAL))
    image_plane_mode = str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE))
    if image_plane_mode == IMAGE_PLANE_MODE_LOCAL_JACOBIAN:
        items.append("image_plane_mode=local-jacobian local Jacobian likelihood")
    elif image_plane_mode == IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA:
        items.append("image_plane_mode=linearized-forward-beta-image-plane linearized image-plane likelihood")
    elif image_plane_mode == IMAGE_PLANE_MODE_MARGINAL:
        items.append("image_plane_mode=marginal-image-plane marginalized linearized image-plane likelihood")

    evidence_likelihood_mode = str(
        getattr(args, "evidence_likelihood_mode", DEFAULT_EVIDENCE_LIKELIHOOD_MODE)
    )
    if solver_fit_mode == SOLVER_FIT_MODE_EVIDENCE_NS and evidence_likelihood_mode in EVIDENCE_LIKELIHOOD_MODES:
        items.append(f"evidence_likelihood_mode={evidence_likelihood_mode} linearized evidence target")

    uses_explicit_source_positions = (
        image_plane_mode == IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA
        or (
            solver_fit_mode == SOLVER_FIT_MODE_EVIDENCE_NS
            and evidence_likelihood_mode == EVIDENCE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
        )
    )
    source_position_parameterization = str(getattr(args, "source_position_parameterization", "direct"))
    if uses_explicit_source_positions and source_position_parameterization != "direct":
        items.append(f"source_position_parameterization={source_position_parameterization}")

    active_scaling_selection = str(getattr(args, "active_scaling_selection", "fixed"))
    if active_scaling_selection == "adaptive":
        items.append("active_scaling_selection=adaptive ranked active subset")
    finite_active_values = _finite_active_scaling_values(getattr(args, "active_scaling_galaxies", None))
    if finite_active_values:
        items.append(f"active_scaling_galaxies=finite counts {finite_active_values}")

    if str(getattr(args, "posterior_diagnostic_mode", POSTERIOR_DIAGNOSTIC_MODE_EXACT)) == POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE:
        items.append("posterior_diagnostic_mode=approximate median+/-std bars; exact per-draw image validation skipped")
    if bool(getattr(args, "quick_diagnostics", False)):
        items.append("quick_diagnostics=active exact post-fit image-position diagnostics skipped")
    return items


def _log_validation_approximation_items(args: argparse.Namespace | None, items: list[str]) -> None:
    if items:
        _log(args, "[validation] warning approximations active: " + "; ".join(items))


def _log_validation_configured_approximation_warning(args: argparse.Namespace) -> None:
    _log_validation_approximation_items(args, _validation_configured_approximation_items(args))


def _log_validation_runtime_summary(args: argparse.Namespace, controls: dict[str, ValidationStageFitControls]) -> None:
    _log(
        args,
        (
            f"[runtime] python={sys.executable} output_dir={args.output_dir} run_name={args.run_name} "
            f"mock={args.mock} realizations={args.realizations} seed={args.seed}"
        ),
    )
    _log(
        args,
        (
            f"[validation] n_primary_families={args.n_primary_families} "
            f"n_subhalo_families={args.n_subhalo_families} n_subhalos={args.n_subhalos} "
            f"source_redshifts={args.source_redshifts} pos_sigma={args.pos_sigma_arcsec} "
            f"min_images_per_family={getattr(args, 'min_images_per_family', 3)} "
            f"max_images_per_family={getattr(args, 'max_images_per_family', None)} "
            f"solver_fit_mode={getattr(args, 'solver_fit_mode', SOLVER_FIT_MODE_SEQUENTIAL)} "
            f"image_plane_mode={getattr(args, 'image_plane_mode', IMAGE_PLANE_MODE_NONE)} "
            f"skip_stage3_image_plane_local_jacobian={getattr(args, 'skip_stage3_image_plane_local_jacobian', False)} "
            f"image_plane_newton_steps={getattr(args, 'image_plane_newton_steps', 0)} "
            f"source_position_parameterization={getattr(args, 'source_position_parameterization', 'prior-whitened')} "
            f"evidence_likelihood_mode={getattr(args, 'evidence_likelihood_mode', DEFAULT_EVIDENCE_LIKELIHOOD_MODE)} "
            f"evidence_source_prior_sigma_arcsec={getattr(args, 'evidence_source_prior_sigma_arcsec', None)} "
            f"evidence_source_prior_mean=({getattr(args, 'evidence_source_prior_mean_x_arcsec', 0.0)},"
            f"{getattr(args, 'evidence_source_prior_mean_y_arcsec', 0.0)}) "
            f"fit_cosmology_flat_wcdm={bool(getattr(args, 'fit_cosmology_flat_wcdm', False))} "
            f"fit_cosmology_all_stages={bool(getattr(args, 'fit_cosmology_all_stages', False))} "
            f"{_format_stage_controls_for_log(controls)} chains={args.chains} "
            f"fit_quality_workers={int(getattr(args, 'fit_quality_workers', 1))} "
            f"sampling_engine={args.sampling_engine} skip_plots={args.skip_plots} "
            f"quick_diagnostics={bool(getattr(args, 'quick_diagnostics', False))} "
            f"write_stage3_recovery={bool(getattr(args, 'write_stage3_recovery', False))}"
        ),
    )


def _validate_validation_args(args: argparse.Namespace) -> None:
    if int(getattr(args, "n_primary_families", 0)) < 0:
        raise SystemExit("--n-primary-families must be non-negative.")
    if int(getattr(args, "n_subhalo_families", 0)) < 0:
        raise SystemExit("--n-subhalo-families must be non-negative.")
    if int(getattr(args, "n_primary_families", 0)) + int(getattr(args, "n_subhalo_families", 0)) <= 0:
        raise SystemExit("At least one source family is required.")
    if int(getattr(args, "min_images_per_family", 3)) < 2:
        raise SystemExit("--min-images-per-family must be at least 2.")
    max_images_per_family = getattr(args, "max_images_per_family", None)
    if max_images_per_family is not None and int(max_images_per_family) < int(getattr(args, "min_images_per_family", 3)):
        raise SystemExit("--max-images-per-family must be at least --min-images-per-family.")
    if int(getattr(args, "fit_quality_workers", 1)) <= 0:
        raise SystemExit("--fit-quality-workers must be positive.")
    if float(getattr(args, "caustic_compute_window_arcsec", DEFAULT_CAUSTIC_COMPUTE_WINDOW_ARCSEC)) <= 0.0:
        raise SystemExit("--caustic-compute-window-arcsec must be positive.")
    if float(getattr(args, "caustic_grid_scale_arcsec", DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC)) <= 0.0:
        raise SystemExit("--caustic-grid-scale-arcsec must be positive.")
    if float(getattr(args, "caustic_min_area_arcsec2", DEFAULT_CAUSTIC_MIN_AREA_ARCSEC2)) <= 0.0:
        raise SystemExit("--caustic-min-area-arcsec2 must be positive.")
    if float(getattr(args, "caustic_boundary_margin_arcsec", DEFAULT_CAUSTIC_BOUNDARY_MARGIN_ARCSEC)) < 0.0:
        raise SystemExit("--caustic-boundary-margin-arcsec must be non-negative.")
    if bool(getattr(args, "write_stage3_recovery", False)):
        if str(getattr(args, "solver_fit_mode", SOLVER_FIT_MODE_SEQUENTIAL)) != SOLVER_FIT_MODE_SEQUENTIAL:
            raise SystemExit("--write-stage3-recovery requires --solver-fit-mode sequential.")
        if not _validation_stage4_enabled(args):
            raise SystemExit("--write-stage3-recovery requires a stage 4 --image-plane-mode.")
        if bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)):
            raise SystemExit("--write-stage3-recovery requires stage 3; remove --skip-stage3-image-plane-local-jacobian.")


def _run_cluster_solver(par_path: Path, output_dir: Path, run_name: str, args: argparse.Namespace) -> Path:
    controls = _normalize_validation_stage_fit_controls(args)
    solver_fit_mode = str(getattr(args, "solver_fit_mode", SOLVER_FIT_MODE_SEQUENTIAL))
    cmd = [
        sys.executable,
        "-m",
        "lenscluster.cluster_solver",
        "--par-path",
        str(par_path),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--fit-mode",
        solver_fit_mode,
        "--svi-steps",
        str(args.svi_steps),
        "--chains",
        str(args.chains),
        "--image-plane-scatter-upper-arcsec",
        str(getattr(args, "image_plane_scatter_upper_arcsec", DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC)),
        "--image-plane-scatter-floor-arcsec",
        str(getattr(args, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)),
        "--image-plane-scatter-prior",
        str(getattr(args, "image_plane_scatter_prior", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR)),
        "--image-plane-scatter-prior-median-arcsec",
        str(getattr(args, "image_plane_scatter_prior_median_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC)),
        "--image-plane-scatter-prior-log-sigma",
        str(getattr(args, "image_plane_scatter_prior_log_sigma", DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA)),
        "--image-presence-match-radius-arcsec",
        str(getattr(args, "image_presence_match_radius_arcsec", DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC)),
        "--image-presence-temperature-arcsec",
        str(getattr(args, "image_presence_temperature_arcsec", DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC)),
        "--image-presence-count-softness",
        str(getattr(args, "image_presence_count_softness", DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS)),
        "--image-presence-count-margin",
        str(getattr(args, "image_presence_count_margin", DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN)),
        "--likelihood-stabilizer-max-gain",
        str(getattr(args, "likelihood_stabilizer_max_gain", DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN)),
        "--likelihood-stabilizer-max-residual-arcsec",
        str(
            getattr(
                args,
                "likelihood_stabilizer_max_residual_arcsec",
                DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC,
            )
        ),
        "--likelihood-stabilizer-residual-loss",
        str(getattr(args, "likelihood_stabilizer_residual_loss", DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS)),
        "--likelihood-stabilizer-student-t-nu",
        str(getattr(args, "likelihood_stabilizer_student_t_nu", DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU)),
        "--sampling-engine",
        str(args.sampling_engine),
        "--source-plane-covariance-floor",
        str(args.source_plane_covariance_floor),
        "--z-bin-efficiency-tol",
        str(args.z_bin_efficiency_tol),
        "--active-scaling-selection",
        str(args.active_scaling_selection),
        "--active-scaling-cumulative-fraction",
        str(args.active_scaling_cumulative_fraction),
        "--active-scaling-min",
        str(args.active_scaling_min),
        "--pos-sigma-arcsec",
        str(args.pos_sigma_arcsec),
        "--seed",
        str(args.seed),
        "--target-accept",
        str(args.target_accept),
        "--fit-quality-workers",
        str(getattr(args, "fit_quality_workers", 1)),
        "--jax-default-device",
        str(getattr(args, "jax_default_device", JAX_DEVICE_AUTO)),
        "--smc-device",
        str(getattr(args, "smc_device", JAX_DEVICE_AUTO)),
        "--smc-particles",
        str(getattr(args, "smc_particles", DEFAULT_SMC_PARTICLES)),
        "--smc-mcmc-kernel",
        str(getattr(args, "smc_mcmc_kernel", DEFAULT_SMC_MCMC_KERNEL)),
        "--smc-mcmc-steps",
        str(getattr(args, "smc_mcmc_steps", DEFAULT_SMC_MCMC_STEPS)),
        "--smc-target-ess-frac",
        str(getattr(args, "smc_target_ess_frac", DEFAULT_SMC_TARGET_ESS_FRAC)),
        "--smc-max-temperature-steps",
        str(getattr(args, "smc_max_temperature_steps", DEFAULT_SMC_MAX_TEMPERATURE_STEPS)),
        "--smc-rmh-scale",
        str(getattr(args, "smc_rmh_scale", DEFAULT_SMC_RMH_SCALE)),
        "--smc-mala-step-size",
        str(getattr(args, "smc_mala_step_size", DEFAULT_SMC_MALA_STEP_SIZE)),
    ]
    _append_stage_option(cmd, "--max-tree-depth", args.max_tree_depth)
    if getattr(args, "image_presence_penalty_weight", None) is not None:
        cmd.extend(["--image-presence-penalty-weight", str(args.image_presence_penalty_weight)])
    if solver_fit_mode == SOLVER_FIT_MODE_SEQUENTIAL:
        _append_stage_option(cmd, "--fit-method", args.fit_method)
        _append_stage_option(cmd, "--warmup", args.warmup)
        _append_stage_option(cmd, "--samples", args.samples)
        cmd.extend(
            [
                "--image-plane-mode",
                str(getattr(args, "image_plane_mode", IMAGE_PLANE_MODE_NONE)),
                "--image-plane-newton-steps",
                str(getattr(args, "image_plane_newton_steps", 0)),
                "--linearized-beta-prior-sigma-arcsec",
                str(getattr(args, "linearized_beta_prior_sigma_arcsec", DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC)),
                "--source-position-parameterization",
                str(getattr(args, "source_position_parameterization", "prior-whitened")),
            ]
        )
    if bool(getattr(args, "fit_cosmology_flat_wcdm", False)):
        cmd.append("--fit-cosmology-flat-wcdm")
    if bool(getattr(args, "fit_cosmology_all_stages", False)):
        cmd.append("--fit-cosmology-all-stages")
    if solver_fit_mode == SOLVER_FIT_MODE_SEQUENTIAL and bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)):
        cmd.append("--skip-stage3-image-plane-local-jacobian")
    if bool(getattr(args, "quick_diagnostics", False)):
        cmd.append("--quick-diagnostics")
    if solver_fit_mode == SOLVER_FIT_MODE_EVIDENCE_NS:
        evidence_likelihood_mode = str(
            getattr(args, "evidence_likelihood_mode", DEFAULT_EVIDENCE_LIKELIHOOD_MODE)
        )
        cmd.extend(
            [
                "--ns-max-samples",
                _format_optional_positive_int(getattr(args, "ns_max_samples", None)),
                "--ns-dlogz",
                str(getattr(args, "ns_dlogz", 1.0e-4)),
            ]
        )
        if getattr(args, "ns_num_live_points", None) is not None:
            cmd.extend(["--ns-num-live-points", str(int(args.ns_num_live_points))])
        cmd.extend(
            [
                "--evidence-likelihood-mode",
                evidence_likelihood_mode,
                "--evidence-source-prior-sigma-arcsec",
                str(getattr(args, "evidence_source_prior_sigma_arcsec")),
                "--evidence-source-prior-mean-x-arcsec",
                str(getattr(args, "evidence_source_prior_mean_x_arcsec", 0.0)),
                "--evidence-source-prior-mean-y-arcsec",
                str(getattr(args, "evidence_source_prior_mean_y_arcsec", 0.0)),
            ]
        )
        if evidence_likelihood_mode == EVIDENCE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE:
            cmd.extend(
                [
                    "--image-plane-newton-steps",
                    str(getattr(args, "image_plane_newton_steps", 0)),
                    "--source-position-parameterization",
                    str(getattr(args, "source_position_parameterization", "prior-whitened")),
                ]
            )
    if args.active_scaling_galaxies is not None:
        cmd.append("--active-scaling-galaxies")
        cmd.extend(str(value) for value in args.active_scaling_galaxies)
    if args.fit_scaling_scatter and int(args.n_subhalos) > 0:
        scatter_fields: list[str] = []
        if float(args.subhalo_sigma_scatter_dex) > 0.0:
            scatter_fields.append("sigma")
        if float(args.subhalo_cut_scatter_dex) > 0.0:
            scatter_fields.append("cut")
        if scatter_fields:
            scatter_max = max(
                float(args.scaling_scatter_max),
                1.25 * _dex_scatter_to_ln(float(args.subhalo_sigma_scatter_dex)),
                1.25 * _dex_scatter_to_ln(float(args.subhalo_cut_scatter_dex)),
            )
            cmd.extend(
                [
                    "--scaling-scatter",
                    "--scaling-scatter-fields",
                    ",".join(scatter_fields),
                    "--scaling-scatter-max",
                    f"{scatter_max:.8g}",
                ]
            )
    if args.skip_plots:
        cmd.append("--skip-plots")
    if bool(getattr(args, "resume", False)):
        cmd.append("--resume")
    if bool(getattr(args, "resume_fast", False)):
        cmd.append("--resume-fast")
    final_stage = _validation_final_stage_name(args)
    final_run_dir = (
        output_dir / run_name
        if solver_fit_mode == SOLVER_FIT_MODE_EVIDENCE_NS
        else output_dir / run_name / final_stage
    )
    start = time.time()
    _log_stage_banner(
        args,
        "VALIDATION SOLVER",
        f"run_name={run_name} final_stage={final_stage} output_dir={output_dir}",
    )
    _log(
        args,
        (
            f"[validation] launching solver run_name={run_name} final_stage={final_stage} "
            f"{_format_stage_controls_for_log(controls)} output_dir={output_dir}"
        ),
    )
    _log_validation_configured_approximation_warning(args)
    _log(args, f"[validation:solver-cmd] {' '.join(cmd)}")
    _run_logged_phase(
        args,
        "validation.cluster_solver",
        lambda: subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], check=True),
        detail=f"run_name={run_name}",
    )
    _log(args, f"[validation] solver complete elapsed={_fmt_seconds(time.time() - start)} final_run_dir={final_run_dir}")
    return final_run_dir


def run_single_bcg_validation(args: argparse.Namespace) -> list[dict[str, Path]]:
    _validate_validation_args(args)
    controls = _normalize_validation_stage_fit_controls(args)
    root = _validation_root(args)
    _configure_debug_log(args, str(args.run_name), root)
    _log_validation_runtime_summary(args, controls)
    outputs: list[dict[str, Path]] = []
    source_redshifts = _run_logged_phase(
        args,
        "validation.parse_source_redshifts",
        lambda: _parse_source_redshifts(args.source_redshifts, fallback=float(args.source_redshift)),
    )
    total_start = time.time()
    for realization in range(int(args.realizations)):
        seed = int(args.seed) + realization
        realization_dir = root / f"seed_{seed}"
        realization_start = time.time()
        _log_stage_banner(
            args,
            f"VALIDATION REALIZATION {realization + 1}/{int(args.realizations)}",
            f"seed={seed} dir={realization_dir}",
        )
        _log(
            args,
            (
                f"[stage] realization start index={realization + 1}/{int(args.realizations)} "
                f"seed={seed} dir={realization_dir}"
            ),
        )
        config = SingleBCGMockConfig(
            seed=seed,
            pos_sigma_arcsec=float(args.pos_sigma_arcsec),
            n_primary_families=int(args.n_primary_families),
            n_subhalo_families=int(args.n_subhalo_families),
            min_images_per_family=int(args.min_images_per_family),
            max_images_per_family=getattr(args, "max_images_per_family", None),
            source_redshift=float(args.source_redshift),
            source_redshifts=source_redshifts,
            source_sigma_int_arcsec=float(args.source_sigma_int_arcsec),
            n_subhalos=int(args.n_subhalos),
            subhalo_sigma_scatter_dex=float(args.subhalo_sigma_scatter_dex),
            subhalo_cut_scatter_dex=float(args.subhalo_cut_scatter_dex),
            caustic_compute_window_arcsec=float(args.caustic_compute_window_arcsec),
            caustic_grid_scale_arcsec=float(args.caustic_grid_scale_arcsec),
            caustic_min_area_arcsec2=float(args.caustic_min_area_arcsec2),
            caustic_boundary_margin_arcsec=float(args.caustic_boundary_margin_arcsec),
        )
        _log(
            args,
            (
                f"[load] generating mock primary_families={config.n_primary_families} "
                f"subhalo_families={config.n_subhalo_families} subhalos={config.n_subhalos} "
                f"image_count={_image_count_requirement_text(config.min_images_per_family, config.max_images_per_family)} "
                f"source_redshifts={','.join(f'{value:.4g}' for value in source_redshifts)}"
            ),
        )
        mock_dir = realization_dir / "mock"
        resume_mock_paths = _validation_mock_paths(mock_dir)
        if bool(getattr(args, "resume", False)) and _validation_mock_complete(resume_mock_paths):
            paths, images, _truth = _run_logged_phase(
                args,
                "validation.load_existing_single_bcg_mock",
                lambda: _load_existing_single_bcg_mock(mock_dir),
                detail=f"seed={seed}",
            )
            _log(args, f"[resume] reusing mock seed={seed} dir={mock_dir}")
        else:
            paths, images, _truth = _run_logged_phase(
                args,
                "validation.generate_single_bcg_mock",
                lambda: generate_single_bcg_mock(mock_dir, config),
                detail=f"seed={seed}",
            )
        _log(
            args,
            (
                f"[load] mock complete images={len(images)} par={paths.par_path} "
                f"catalog={paths.image_catalog_path} truth={paths.truth_path}"
            ),
        )
        if bool(getattr(args, "resume", False)):
            _log(args, f"[resume] refreshing validation outputs seed={seed} dir={realization_dir}")
        solver_run_name = "fit"
        solver_run_dir = _run_cluster_solver(paths.par_path, realization_dir / "solver", solver_run_name, args)
        _log(args, f"[output] writing recovery outputs from {solver_run_dir} to {realization_dir}")
        recovery_payload: dict[str, Any] = {}
        output_paths = _run_logged_phase(
            args,
            "validation.write_recovery_outputs",
            lambda: write_recovery_outputs(
                solver_run_dir,
                paths.truth_path,
                paths.mock_images_path,
                output_dir=realization_dir,
                posterior_diagnostic_draws=int(args.posterior_diagnostic_draws),
                posterior_diagnostic_workers=int(getattr(args, "posterior_diagnostic_workers", 1)),
                posterior_diagnostic_mode=str(
                    getattr(args, "posterior_diagnostic_mode", POSTERIOR_DIAGNOSTIC_MODE_EXACT)
                ),
                quick_diagnostics=bool(getattr(args, "quick_diagnostics", False)),
                progress_args=args,
                recovery_payload=recovery_payload,
            ),
            detail=f"seed={seed}",
        )
        stage3_recovery_payload: dict[str, Any] | None = None
        if bool(getattr(args, "write_stage3_recovery", False)):
            stage3_run_dir = solver_run_dir.parent / "stage3_image_plane"
            if not _validation_stage_has_recovery_artifacts(stage3_run_dir):
                raise FileNotFoundError(f"Cannot write stage 3 recovery; missing artifacts under {stage3_run_dir}")
            stage3_recovery_dir = realization_dir / "stage3_recovery"
            _log(args, f"[output] writing stage3 recovery outputs from {stage3_run_dir} to {stage3_recovery_dir}")
            stage3_recovery_payload = {}
            stage3_output_paths = _run_logged_phase(
                args,
                "validation.write_stage3_recovery_outputs",
                lambda: write_recovery_outputs(
                    stage3_run_dir,
                    paths.truth_path,
                    paths.mock_images_path,
                    output_dir=stage3_recovery_dir,
                    posterior_diagnostic_draws=int(args.posterior_diagnostic_draws),
                    posterior_diagnostic_workers=int(getattr(args, "posterior_diagnostic_workers", 1)),
                    posterior_diagnostic_mode=str(
                        getattr(args, "posterior_diagnostic_mode", POSTERIOR_DIAGNOSTIC_MODE_EXACT)
                    ),
                    quick_diagnostics=bool(getattr(args, "quick_diagnostics", False)),
                    progress_args=args,
                    recovery_payload=stage3_recovery_payload,
                ),
                detail=f"seed={seed}",
            )
            output_paths.update({f"stage3_{key}": value for key, value in stage3_output_paths.items()})
        summary_path = _run_logged_phase(
            args,
            "validation.write_run_summary_txt",
            lambda: write_validation_run_summary(
                solver_run_dir,
                paths.truth_path,
                realization_dir,
                run_name=str(args.run_name),
                seed=seed,
            ),
            detail=f"seed={seed}",
        )
        _log(args, f"[output] validation run summary written to {summary_path}")
        results_json_path = _run_logged_phase(
            args,
            "validation.write_results_json",
            lambda: write_validation_results_json(
                args=args,
                seed=seed,
                realization_dir=realization_dir,
                config=config,
                paths=paths,
                images=images,
                truth_payload=_truth,
                solver_run_dir=solver_run_dir,
                summary_path=summary_path,
                output_paths=output_paths,
                recovery_payload=recovery_payload,
                stage3_recovery_payload=stage3_recovery_payload,
                controls=controls,
            ),
            detail=f"seed={seed}",
        )
        output_paths["results_json"] = results_json_path
        _log(args, f"[output] validation results json written to {results_json_path}")
        _log(args, f"[output] recovery complete files={len(output_paths)} names={','.join(sorted(output_paths))}")
        outputs.append(output_paths)
        _log(
            args,
            (
                f"[stage] realization end index={realization + 1}/{int(args.realizations)} "
                f"elapsed={_fmt_seconds(time.time() - realization_start)}"
            ),
        )
    _log(args, f"[done] validation complete realizations={len(outputs)} elapsed={_fmt_seconds(time.time() - total_start)} root={root}")
    return outputs


def _parse_source_redshifts(raw: str | None, *, fallback: float) -> tuple[float, ...]:
    if raw is None or not str(raw).strip():
        return (float(fallback),)
    values = tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values:
        return (float(fallback),)
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mock-recovery validation suite for lenscluster.")
    parser.add_argument("--mock", choices=("single-bcg",), default="single-bcg")
    parser.add_argument("--output-dir", default="validation_runs")
    parser.add_argument("--run-name", default="single_bcg_recovery")
    parser.add_argument("--realizations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing mock inputs, completed solver stages, and completed validation realization outputs.",
    )
    parser.add_argument(
        "--resume-fast",
        action="store_true",
        help="Pass --resume-fast to the sequential cluster solver so each realization runs only the final enabled stage.",
    )
    parser.add_argument("--n-primary-families", type=int, default=20)
    parser.add_argument("--n-subhalo-families", type=int, default=0)
    parser.add_argument("--min-images-per-family", type=int, default=3)
    parser.add_argument(
        "--max-images-per-family",
        type=_parse_optional_positive_int,
        default=None,
        help="Optional maximum accepted image multiplicity per generated mock family. Use 'none' for unlimited.",
    )
    parser.add_argument("--caustic-compute-window-arcsec", type=float, default=DEFAULT_CAUSTIC_COMPUTE_WINDOW_ARCSEC)
    parser.add_argument("--caustic-grid-scale-arcsec", type=float, default=DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC)
    parser.add_argument("--caustic-min-area-arcsec2", type=float, default=DEFAULT_CAUSTIC_MIN_AREA_ARCSEC2)
    parser.add_argument("--caustic-boundary-margin-arcsec", type=float, default=DEFAULT_CAUSTIC_BOUNDARY_MARGIN_ARCSEC)
    parser.add_argument("--n-subhalos", type=int, default=0)
    parser.add_argument(
        "--subhalo-sigma-scatter-dex",
        type=float,
        default=0.07,
        help="Injected log10 scatter in the subhalo velocity-dispersion scaling relation.",
    )
    parser.add_argument(
        "--subhalo-cut-scatter-dex",
        type=float,
        default=0.20,
        help="Injected log10 scatter in the subhalo cut-radius scaling relation.",
    )
    parser.add_argument("--source-redshift", type=float, default=2.0)
    parser.add_argument(
        "--source-redshifts",
        default="1.5,2.0,3.0",
        help="Comma-separated source redshifts cycled across mock families. Empty string falls back to --source-redshift.",
    )
    parser.add_argument("--source-sigma-int-arcsec", type=float, default=0.05)
    parser.add_argument("--pos-sigma-arcsec", type=float, default=0.15)
    parser.add_argument(
        "--solver-fit-mode",
        choices=(SOLVER_FIT_MODE_SEQUENTIAL, SOLVER_FIT_MODE_EVIDENCE_NS),
        default=SOLVER_FIT_MODE_SEQUENTIAL,
        help="Solver workflow: staged sequential fit or one-shot nested-sampling evidence.",
    )
    parser.add_argument(
        "--fit-method",
        nargs="+",
        choices=(FIT_METHOD_SVI, FIT_METHOD_SVI_NUTS, FIT_METHOD_NS, FIT_METHOD_SMC),
        default=[FIT_METHOD_SVI_NUTS],
        metavar="{svi,svi+nuts,ns,smc}",
        help=(
            "Sequential solver fit method. Pass one value for all sampled stages, two values for "
            "stage2_joint and stage3_image_plane, or three values when stage 4 is enabled. "
            "Ignored for --solver-fit-mode evidence-ns, which always uses nested sampling internally."
        ),
    )
    parser.add_argument(
        "--image-plane-mode",
        choices=(
            IMAGE_PLANE_MODE_NONE,
            IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
            IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
            IMAGE_PLANE_MODE_MARGINAL,
        ),
        default=IMAGE_PLANE_MODE_NONE,
        help="Optional solver image-plane refinement mode.",
    )
    parser.add_argument(
        "--skip-stage3-image-plane-local-jacobian",
        action="store_true",
        help="Skip solver stage 3 before a final stage 4 image-plane mode.",
    )
    parser.add_argument(
        "--write-stage3-recovery",
        action="store_true",
        help="Also write truth recovery plots for stage3_image_plane when running a stage 4 validation workflow.",
    )
    parser.add_argument(
        "--image-plane-newton-steps",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
        help="Additional stage-4 Newton updates after the initial local linear solve.",
    )
    parser.add_argument(
        "--linearized-beta-prior-sigma-arcsec",
        type=float,
        default=DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC,
    )
    parser.add_argument(
        "--source-position-parameterization",
        choices=("direct", "prior-whitened", "conditional-whitened"),
        default="prior-whitened",
        help="Stage-4 explicit source-position sampling coordinate passed through to cluster_solver.",
    )
    parser.add_argument(
        "--image-plane-scatter-upper-arcsec",
        type=float,
        default=DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC,
    )
    parser.add_argument(
        "--image-plane-scatter-floor-arcsec",
        type=float,
        default=DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC,
    )
    parser.add_argument(
        "--image-plane-scatter-prior",
        choices=IMAGE_PLANE_SCATTER_PRIORS,
        default=DEFAULT_IMAGE_PLANE_SCATTER_PRIOR,
    )
    parser.add_argument(
        "--image-plane-scatter-prior-median-arcsec",
        type=float,
        default=DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC,
    )
    parser.add_argument(
        "--image-plane-scatter-prior-log-sigma",
        type=float,
        default=DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA,
    )
    parser.add_argument("--image-presence-penalty-weight", type=float, default=None)
    parser.add_argument(
        "--image-presence-match-radius-arcsec",
        type=float,
        default=DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC,
    )
    parser.add_argument(
        "--image-presence-temperature-arcsec",
        type=float,
        default=DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC,
    )
    parser.add_argument(
        "--image-presence-count-softness",
        type=float,
        default=DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS,
    )
    parser.add_argument(
        "--image-presence-count-margin",
        type=float,
        default=DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN,
    )
    parser.add_argument(
        "--likelihood-stabilizer-max-gain",
        type=float,
        default=DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN,
    )
    parser.add_argument(
        "--likelihood-stabilizer-max-residual-arcsec",
        type=float,
        default=DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC,
    )
    parser.add_argument(
        "--likelihood-stabilizer-residual-loss",
        choices=LIKELIHOOD_STABILIZER_RESIDUAL_LOSSES,
        default=DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS,
    )
    parser.add_argument(
        "--likelihood-stabilizer-student-t-nu",
        type=float,
        default=DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU,
    )
    parser.add_argument(
        "--evidence-source-prior-sigma-arcsec",
        type=float,
        default=None,
        help="Required for --solver-fit-mode evidence-ns; fixed Gaussian source prior sigma shared by all families.",
    )
    parser.add_argument("--evidence-source-prior-mean-x-arcsec", type=float, default=0.0)
    parser.add_argument("--evidence-source-prior-mean-y-arcsec", type=float, default=0.0)
    parser.add_argument(
        "--evidence-likelihood-mode",
        choices=EVIDENCE_LIKELIHOOD_MODES,
        default=DEFAULT_EVIDENCE_LIKELIHOOD_MODE,
        help="One-shot evidence likelihood target passed through to cluster_solver.",
    )
    parser.add_argument("--svi-steps", type=int, default=1000)
    parser.add_argument(
        "--warmup",
        type=int,
        nargs="+",
        default=[300],
        help="Solver NUTS warmup steps. Accepts one value or staged values through optional stage 3/stage 4.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[500],
        help="Solver posterior draws per chain. Accepts one value or staged values through optional stage 3/stage 4.",
    )
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--ns-num-live-points", type=int, default=None)
    parser.add_argument(
        "--ns-max-samples",
        type=_parse_optional_positive_int,
        default=None,
        help="JAXNS maximum nested-sampling samples for --solver-fit-mode evidence-ns. Defaults to unlimited; pass a positive integer to cap.",
    )
    parser.add_argument("--ns-dlogz", type=float, default=1.0e-4)
    parser.add_argument("--jax-default-device", choices=JAX_DEVICE_CHOICES, default=JAX_DEVICE_AUTO)
    parser.add_argument("--smc-device", choices=JAX_DEVICE_CHOICES, default=JAX_DEVICE_AUTO)
    parser.add_argument("--smc-particles", type=int, default=DEFAULT_SMC_PARTICLES)
    parser.add_argument("--smc-mcmc-kernel", choices=SMC_MCMC_KERNELS, default=DEFAULT_SMC_MCMC_KERNEL)
    parser.add_argument("--smc-mcmc-steps", type=int, default=DEFAULT_SMC_MCMC_STEPS)
    parser.add_argument("--smc-target-ess-frac", type=float, default=DEFAULT_SMC_TARGET_ESS_FRAC)
    parser.add_argument("--smc-max-temperature-steps", type=int, default=DEFAULT_SMC_MAX_TEMPERATURE_STEPS)
    parser.add_argument("--smc-rmh-scale", type=float, default=DEFAULT_SMC_RMH_SCALE)
    parser.add_argument("--smc-mala-step-size", type=float, default=DEFAULT_SMC_MALA_STEP_SIZE)
    parser.add_argument(
        "--sampling-engine",
        choices=("full", "refreshing_surrogate", "active_subset"),
        default="refreshing_surrogate",
    )
    parser.add_argument("--source-plane-covariance-floor", type=float, default=1.0e-6)
    parser.add_argument("--z-bin-efficiency-tol", type=float, default=0.01)
    parser.add_argument(
        "--fit-cosmology-flat-wcdm",
        action="store_true",
        help="Forward solver sampling of flat wCDM Omega_m,w0 in the final fitting stage by default.",
    )
    parser.add_argument(
        "--fit-cosmology-all-stages",
        action="store_true",
        help="With --fit-cosmology-flat-wcdm, forward all-stage sequential cosmology sampling to the solver.",
    )
    parser.add_argument(
        "--active-scaling-galaxies",
        type=int,
        nargs="+",
        default=None,
        help="Fixed active counts in fixed mode, or adaptive per-potfile caps in adaptive mode. Negative uses all.",
    )
    parser.add_argument("--active-scaling-selection", choices=("fixed", "adaptive"), default="adaptive")
    parser.add_argument("--active-scaling-cumulative-fraction", type=float, default=0.995)
    parser.add_argument("--active-scaling-min", type=int, default=4)
    parser.add_argument(
        "--fit-scaling-scatter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit scaling-relation scatter hyperparameters when subhalos with injected scatter are present.",
    )
    parser.add_argument(
        "--scaling-scatter-max",
        type=float,
        default=0.5,
        help="Upper bound, in natural-log units, for fitted scaling-scatter hyperparameters.",
    )
    parser.add_argument(
        "--posterior-diagnostic-draws",
        type=int,
        default=8,
        help=(
            "Maximum posterior draws used for image/source validation uncertainty bars; "
            "mass-profile and surface-density bands use a fixed capped posterior subsample."
        ),
    )
    parser.add_argument(
        "--posterior-diagnostic-workers",
        type=int,
        default=1,
        help=(
            "Thread workers for posterior image/source validation uncertainty families. "
            "Use 1 for serial behavior."
        ),
    )
    parser.add_argument(
        "--fit-quality-workers",
        type=int,
        default=1,
        help="Worker threads for the underlying solver's fit-quality diagnostics.",
    )
    parser.add_argument(
        "--posterior-diagnostic-mode",
        choices=POSTERIOR_DIAGNOSTIC_MODES,
        default=POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        help=(
            "Posterior image/source validation uncertainty mode. exact solves image positions per draw; "
            "approximate uses posterior median +/- standard deviation summaries and skips exact image validation."
        ),
    )
    parser.add_argument(
        "--quick-diagnostics",
        action="store_true",
        help=(
            "Fast post-fit diagnostics for the solver and validation recovery: skip exact image-position "
            "validation and use approximate median +/- std posterior diagnostics."
        ),
    )
    parser.add_argument("--target-accept", type=float, default=0.85)
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        nargs="+",
        default=[8],
        help="Solver NUTS max tree depth. Accepts one value or staged values through optional stage 3/stage 4.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip the standard solver plot suite. Validation recovery figures are still written as PDFs.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress validation wrapper logs while keeping solver output.")
    return parser


def main() -> None:
    try:
        args = _build_parser().parse_args()
        _validate_validation_args(args)
        _normalize_validation_stage_fit_controls(args)
        _configure_debug_log(args, str(args.run_name), _validation_root(args))
        _log(args, "[main] startup")
        run_single_bcg_validation(args)
    except BaseException as exc:
        _log_exception("validation.main", exc)
        raise
    finally:
        _close_debug_log()


if __name__ == "__main__":
    main()
