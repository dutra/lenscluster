from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .planning import RunPlan


@dataclass(frozen=True)
class ParameterArtifact:
    path: Path
    samples: np.ndarray
    best_fit: np.ndarray
    log_prob: np.ndarray
    parameter_specs: list[dict[str, Any]]
    parameter_names: list[str]
    model_config: dict[str, Any] | None
    runtime_args: dict[str, Any] | None
    state_meta: dict[str, Any]
    init_diagnostics: dict[str, Any]
    sample_weights: np.ndarray | None = None
    grouped_samples: np.ndarray | None = None
    grouped_log_prob: np.ndarray | None = None
    map_fit: np.ndarray | None = None
    maximum_likelihood_fit: np.ndarray | None = None
    median_fit: np.ndarray | None = None
    accept_prob: np.ndarray | None = None
    diverging: np.ndarray | None = None
    num_steps: np.ndarray | None = None


def root_run_dir(plan: RunPlan) -> Path:
    return plan.output.output_dir / plan.output.run_name


def artifacts_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    return path if path.name == "artifacts" else path / "artifacts"


def _decode_json_dataset(value: Any) -> Any:
    if isinstance(value, bytes):
        text = value.decode("utf-8")
    else:
        text = np.bytes_(value).decode("utf-8") if isinstance(value, np.bytes_) else str(value)
    return json.loads(text)


def _read_json(handle: h5py.File | h5py.Group, name: str, default: Any = None) -> Any:
    if name not in handle:
        return default
    return _decode_json_dataset(handle[name][()])


def _resolve_parameter_artifact_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.name == "artifacts":
        h5_path = candidate / "plot_bundle.h5"
        if h5_path.exists():
            return h5_path
    direct = candidate / "artifacts" / "plot_bundle.h5"
    if direct.exists():
        return direct
    matches = sorted(candidate.glob("**/artifacts/plot_bundle.h5"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"No parameter artifact found under {candidate}")


def _optional_array(group: h5py.Group, name: str) -> np.ndarray | None:
    return np.asarray(group[name]) if name in group else None


def load_parameter_artifact(path: str | Path) -> ParameterArtifact:
    """Load saved fit parameters from ``plot_bundle.h5`` without rebuilding model code.

    The artifact contains numeric posterior arrays plus JSON metadata. It is safe
    for notebook inspection and does not deserialize pickled or compiled objects.
    """
    h5_path = _resolve_parameter_artifact_path(path)
    with h5py.File(h5_path, "r") as handle:
        posterior = handle["posterior"]
        state_group = handle["state"]
        state_meta = _read_json(state_group, "build_state_meta_json", default={}) or {}
        parameter_specs = list(state_meta.get("parameter_specs") or [])
        parameter_names = [
            str(spec.get("sample_name") or spec.get("name") or f"theta_{idx}")
            for idx, spec in enumerate(parameter_specs)
        ]
        return ParameterArtifact(
            path=h5_path,
            samples=np.asarray(posterior["samples"], dtype=float),
            best_fit=np.asarray(posterior["best_fit"], dtype=float),
            log_prob=np.asarray(posterior["log_prob"], dtype=float),
            parameter_specs=parameter_specs,
            parameter_names=parameter_names,
            model_config=_read_json(handle, "model_config_json", default=None),
            runtime_args=_read_json(handle, "runtime_args_json", default=None),
            state_meta=state_meta,
            init_diagnostics=_read_json(handle, "init_diagnostics_json", default={}) or {},
            sample_weights=_optional_array(posterior, "sample_weights"),
            grouped_samples=_optional_array(posterior, "grouped_samples"),
            grouped_log_prob=_optional_array(posterior, "grouped_log_prob"),
            map_fit=_optional_array(posterior, "map_fit"),
            maximum_likelihood_fit=_optional_array(posterior, "maximum_likelihood_fit"),
            median_fit=_optional_array(posterior, "median_fit"),
            accept_prob=_optional_array(posterior, "accept_prob"),
            diverging=_optional_array(posterior, "diverging"),
            num_steps=_optional_array(posterior, "num_steps"),
        )
