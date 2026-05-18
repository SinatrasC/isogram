# Isogram

Isogram is a PyTorch service for classifying English text as human-written or
AI-generated. The repository includes dataset preparation, model training,
evaluation, checkpointing, FastAPI serving, CI, pre-commit checks, and MLOps
metadata for reproducible experiments.

The main implementation path uses a sampled merged dataset under
`data/processed/merged/` and keeps raw data, processed CSV files, checkpoints,
and MLflow runs out of Git.

## Main Commands

Build the merged dataset:

```bash
python -m isogram.data.build_dataset \
  --local-path data/raw/daigt-v2 \
  --local-source-name drcat-v2 \
  --local-license apache-2.0 \
  --hf-sample-rows 60000 \
  --output-dir data/processed/merged
```

Train with Hydra and MLflow:

```bash
isogram-train-hydra model=char_cnn
isogram-train-hydra model=deberta
```

Evaluate a checkpoint:

```bash
python -m isogram.evaluate \
  --checkpoint artifacts/checkpoints/deberta_merged.pt \
  --data data/processed/merged/test.csv
```
