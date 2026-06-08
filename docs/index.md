# Isogram

Isogram is an AI-generated essay detection package with data building, PyTorch Lightning training, evaluation, FastAPI serving, DVC data handling, Hydra configs, MLflow logging, CI, and pre-commit checks.

## Main Commands

Build or fetch the default dataset:

```bash
uv run isogram data
```

Train the baseline and main model:

```bash
uv run --extra mlops isogram train model=char_cnn
uv run --extra mlops isogram train model=deberta
```

Run a quick local smoke training job without MLflow:

```bash
uv run isogram train model=char_cnn trainer.limit_rows=512 model.max_epochs=3 logging.enabled=false
```

Evaluate a configured checkpoint:

```bash
uv run isogram evaluate model=char_cnn logging.enabled=false
```
