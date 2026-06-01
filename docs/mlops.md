# MLOps Workflow

## Dataset Versioning

`params.yaml` stores dataset construction parameters and `dvc.yaml` defines a `build_dataset` stage. DVC is configured with separate local remotes for data and model artifacts:

- `data`: `.dvc_storage/data`
- `models`: `.dvc_storage/models`

The training command calls the dataset guard before fitting. It tries DVC first, then falls back to downloading the public Hugging Face splits.

## Experiment Configs

Hydra configs live under the repository-level `configs/` directory:

- `data/main.yaml` defines the public Hugging Face source, optional sampling, and split paths
- `model/char_cnn.yaml` defines the baseline
- `model/deberta.yaml` defines the transformer classifier
- `trainer/default.yaml` defines Lightning runtime settings
- `logging/mlflow.yaml` defines the tracking endpoint

Each run saves the resolved config under `reports/run_configs/`.

## MLflow Tracking

Normal training logs to `http://127.0.0.1:8080`. Each run records hyperparameters, the Git commit ID, per-epoch losses and metrics, best metrics, plots, reports, configs, and checkpoints.

For local smoke tests without a server, pass:

```bash
logging.enabled=false
```
