from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any


class MlflowRun(AbstractContextManager["MlflowRun"]):
    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        run_name: str | None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow logging requires the optional `mlops` extra. "
                "Install it with `uv sync --extra mlops` or `python -m pip install -e .[mlops]`."
            ) from exc

        self._mlflow = mlflow
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tags = dict(tags or {})

    def __enter__(self) -> MlflowRun:
        self._mlflow.set_tracking_uri(self._tracking_uri)
        self._mlflow.set_experiment(self._experiment_name)
        self._mlflow.start_run(run_name=self._run_name)
        if self._tags:
            self._mlflow.set_tags(self._tags)
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        status = "FAILED" if exc_type is not None else "FINISHED"
        self._mlflow.end_run(status=status)

    def log_params(self, params: Mapping[str, Any]) -> None:
        clean_params = {key: _stringify_param(value) for key, value in params.items()}
        self._mlflow.log_params(clean_params)

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        clean_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float) and value is not None
        }
        if clean_metrics:
            self._mlflow.log_metrics(clean_metrics, step=step)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        if path.exists():
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _stringify_param(value: Any) -> str | int | float | bool:
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return "None"
    return str(value)


def maybe_mlflow_run(
    *,
    enabled: bool,
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None,
    tags: Mapping[str, str] | None = None,
) -> AbstractContextManager[MlflowRun | None]:
    if not enabled:
        return nullcontext(None)
    return MlflowRun(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags,
    )
