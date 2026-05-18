# MLOps Workflow

## Dataset Versioning

`params.yaml` stores the dataset construction parameters. `dvc.yaml` defines a
`build_dataset` stage that rebuilds the merged sample from the local CSV source
and the Hugging Face source. The generated data files stay under `data/`, which
is ignored by Git.

The generated `metadata.json` records:

- source dataset names and declared licenses
- source row counts and label counts
- duplicate/conflicting-label removal counts
- train/validation/test split sizes
- random seed and split fractions

## Experiment Configs

Hydra configs live under `src/isogram/conf/`:

- `dataset/merged.yaml` defines the merged train/validation/test paths
- `model/char_cnn.yaml` defines the baseline run
- `model/deberta.yaml` defines the transformer run

Each Hydra run writes a resolved config artifact under `reports/run_configs/`
and passes that file to training so it is logged to MLflow.

## MLflow Tracking

Training logs to the local `mlruns/` directory by default. Each run records:

- model and dataset parameters
- row counts and hardware device
- per-epoch validation metrics
- best validation metrics
- resolved Hydra config
- dataset metadata
- training report JSON
- checkpoint artifact

Launch the local UI with:

```bash
mlflow ui --backend-store-uri mlruns
```
