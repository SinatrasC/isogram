from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from isogram.config import CommonPaths, write_json
from isogram.data.splits import Split
from isogram.inference import Predictor
from isogram.metrics import compute_classification_report
from isogram.tracking import maybe_mlflow_run


def evaluate_checkpoint(
    *,
    checkpoint: str | Path,
    data: str | Path = CommonPaths().processed_dir / Split.VAL.filename,
    output: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 32,
    threshold: float | None = None,
    mlflow: bool = False,
    mlflow_tracking_uri: str = "http://127.0.0.1:8080",
    mlflow_experiment: str = "isogram",
    mlflow_run_name: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    data_path = Path(data)
    output_path = (
        Path(output)
        if output is not None
        else CommonPaths().report_dir / f"{checkpoint_path.stem}_evaluation.json"
    )

    frame = pd.read_csv(data_path)
    texts = frame["text"].astype(str).tolist()
    labels = frame["label"].astype(int).tolist()
    predictor = Predictor(checkpoint_path, device=device)

    started = time.perf_counter()
    scores = predictor.predict_batch(texts, batch_size=batch_size)
    elapsed = time.perf_counter() - started
    selected_threshold = threshold if threshold is not None else predictor.threshold
    report = compute_classification_report(labels, scores, threshold=selected_threshold)
    report.update(
        {
            "checkpoint": str(checkpoint_path),
            "data": str(data_path),
            "model_version": predictor.model_version,
            "rows": len(texts),
            "batch_size": batch_size,
            "elapsed_seconds": elapsed,
            "examples_per_second": len(texts) / elapsed if elapsed > 0 else None,
        }
    )
    write_json(output_path, report)
    with maybe_mlflow_run(
        enabled=mlflow,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
        run_name=mlflow_run_name or f"eval-{checkpoint_path.stem}",
        tags={"stage": "evaluation", "checkpoint": str(checkpoint_path)},
    ) as mlflow_run:
        if mlflow_run is not None:
            mlflow_run.log_params(
                {
                    "checkpoint": checkpoint_path,
                    "data": data_path,
                    "rows": len(texts),
                    "batch_size": batch_size,
                    "threshold": selected_threshold,
                }
            )
            mlflow_run.log_metrics(report)
            mlflow_run.log_artifact(output_path, artifact_path="reports")
    print(f"Saved evaluation report to {output_path}")
    print(report)
    return report


def main() -> None:
    import fire

    fire.Fire(evaluate_checkpoint)


if __name__ == "__main__":
    main()
