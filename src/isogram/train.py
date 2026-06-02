from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from isogram.checkpoint import save_checkpoint
from isogram.config import DEFAULT_SEED, ensure_parent, get_git_commit, set_seed, write_json
from isogram.data.schema import sample_balanced_by_label
from isogram.metrics import compute_classification_report
from isogram.models.char_cnn import CharCnnClassifier, CharTokenizer
from isogram.models.deberta import DebertaBatcher, DebertaTextClassifier


class TextCsvDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {"text": self.texts[index], "label": self.labels[index]}


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def _path_from_config(config: Any, key: str, fallback: Path) -> Path:
    value = cfg_get(config, f"{key}_path", cfg_get(config, key, fallback))
    return Path(str(value))


def read_split(
    path: Path, *, limit_rows: int | None = None, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    normalized = frame.loc[:, ["text", "label"]].copy()
    normalized["text"] = normalized["text"].astype(str).str.strip()
    normalized["label"] = normalized["label"].astype(int)
    normalized = normalized[normalized["text"].str.len() > 0].reset_index(drop=True)
    if normalized["label"].nunique() < 2:
        raise ValueError(f"{path} must contain both labels")
    if limit_rows is not None:
        normalized = sample_balanced_by_label(normalized, max_rows=limit_rows, seed=seed)
    return normalized


def make_collate_fn(
    *,
    model_type: str,
    char_tokenizer: CharTokenizer | None,
    deberta_batcher: DebertaBatcher | None,
) -> Callable[[list[dict[str, object]]], dict[str, torch.Tensor]]:
    def collate(rows: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        texts = [str(row["text"]) for row in rows]
        labels = torch.tensor(
            [int(cast(int, row["label"])) for row in rows],
            dtype=torch.float32,
        )
        if model_type == "char_cnn":
            if char_tokenizer is None:
                raise RuntimeError("Char tokenizer is required for char_cnn")
            return {"input_ids": char_tokenizer.batch_encode(texts), "labels": labels}
        if deberta_batcher is None:
            raise RuntimeError("DeBERTa batcher is required for deberta")
        batch = deberta_batcher(texts)
        batch["labels"] = labels
        return batch

    return collate


def build_model_stack(model_cfg: Any) -> tuple[nn.Module, dict[str, Any], Callable]:
    model_type = str(cfg_get(model_cfg, "name", "char_cnn"))
    dropout = float(cfg_get(model_cfg, "dropout", 0.2))
    if model_type == "char_cnn":
        tokenizer = CharTokenizer(max_length=int(cfg_get(model_cfg, "char_max_length", 2048)))
        kernel_sizes = tuple(
            int(value) for value in cfg_get(model_cfg, "char_kernel_sizes", (3, 5, 7))
        )
        embedding_dim = int(cfg_get(model_cfg, "char_embedding_dim", 64))
        channels = int(cfg_get(model_cfg, "char_channels", 96))
        model_config = {
            "chars": tokenizer.chars,
            "max_length": tokenizer.max_length,
            "embedding_dim": embedding_dim,
            "channels": channels,
            "kernel_sizes": kernel_sizes,
            "dropout": dropout,
        }
        model: nn.Module = CharCnnClassifier(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            channels=channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )
        collate = make_collate_fn(
            model_type=model_type,
            char_tokenizer=tokenizer,
            deberta_batcher=None,
        )
        return model, model_config, collate

    if model_type != "deberta":
        raise ValueError(f"Unknown model type: {model_type}")

    model_name = str(cfg_get(model_cfg, "deberta_model_name", "microsoft/deberta-v3-base"))
    max_length = int(cfg_get(model_cfg, "deberta_max_length", 512))
    batcher = DebertaBatcher(model_name, max_length=max_length)
    model_config = {"model_name": model_name, "max_length": max_length, "dropout": dropout}
    model = DebertaTextClassifier.from_pretrained(model_name, dropout=dropout)
    collate = make_collate_fn(
        model_type=model_type,
        char_tokenizer=None,
        deberta_batcher=batcher,
    )
    return model, model_config, collate


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        train_path: Path,
        val_path: Path,
        collate_fn: Callable[[list[dict[str, object]]], dict[str, torch.Tensor]],
        batch_size: int,
        num_workers: int,
        limit_rows: int | None,
        seed: int,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit_rows = limit_rows
        self.seed = seed
        self.train_frame: pd.DataFrame | None = None
        self.val_frame: pd.DataFrame | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"}:
            self.train_frame = read_split(
                self.train_path,
                limit_rows=self.limit_rows,
                seed=self.seed,
            )
            self.val_frame = read_split(
                self.val_path,
                limit_rows=self.limit_rows,
                seed=self.seed + 1,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_frame is None:
            raise RuntimeError("Data module has not been set up")
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            TextCsvDataset(self.train_frame),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            generator=generator,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_frame is None:
            raise RuntimeError("Data module has not been set up")
        return DataLoader(
            TextCsvDataset(self.val_frame),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )


class LightningTextClassifier(pl.LightningModule):
    def __init__(
        self,
        *,
        model: nn.Module,
        model_type: str,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.BCEWithLogitsLoss()
        self.history: list[dict[str, Any]] = []
        self.best_metrics: dict[str, Any] | None = None
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.best_score = -1.0
        self._train_losses: list[tuple[float, int]] = []
        self._val_losses: list[tuple[float, int]] = []
        self._val_labels: list[int] = []
        self._val_scores: list[float] = []

    def forward_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.model_type == "char_cnn":
            return self.model(batch["input_ids"])
        return self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        labels = batch["labels"]
        logits = self.forward_logits(batch)
        loss = self.criterion(logits, labels)
        batch_size = int(labels.shape[0])
        self._train_losses.append((float(loss.detach().cpu()), batch_size))
        self.log("train_loss", loss, on_epoch=True, on_step=False, batch_size=batch_size)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        labels = batch["labels"]
        logits = self.forward_logits(batch)
        loss = self.criterion(logits, labels)
        scores = torch.sigmoid(logits).detach().cpu().tolist()
        self._val_losses.append((float(loss.detach().cpu()), int(labels.shape[0])))
        self._val_labels.extend(int(label) for label in labels.detach().cpu().tolist())
        self._val_scores.extend(float(score) for score in scores)
        self.log("val_loss", loss, on_epoch=True, on_step=False, batch_size=int(labels.shape[0]))
        return loss

    def on_train_epoch_start(self) -> None:
        self._train_losses = []

    def on_validation_epoch_start(self) -> None:
        self._val_losses = []
        self._val_labels = []
        self._val_scores = []

    def on_validation_epoch_end(self) -> None:
        if not self._val_labels:
            return

        train_loss = _weighted_average(self._train_losses)
        val_loss = _weighted_average(self._val_losses)
        report = compute_classification_report(self._val_labels, self._val_scores)
        record = {
            "epoch": int(self.current_epoch) + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **report,
        }
        self.history.append(record)

        loggable = {
            f"val_{key}": value
            for key, value in report.items()
            if isinstance(value, int | float) and value is not None
        }
        if val_loss is not None:
            loggable["val_loss_epoch"] = val_loss
        if loggable:
            self.log_dict(loggable, on_epoch=True, on_step=False)

        raw_score = report.get("roc_auc", report.get("f1", -1.0))
        score = float(raw_score) if raw_score is not None else -1.0
        if score > self.best_score:
            self.best_score = score
            self.best_metrics = record
            self.best_state_dict = {
                key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()
            }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def _weighted_average(parts: list[tuple[float, int]]) -> float | None:
    total = sum(value * count for value, count in parts)
    count = sum(count for _, count in parts)
    return total / count if count else None


def write_training_plots(
    *,
    history: list[dict[str, Any]],
    output_dir: Path,
    run_name: str,
) -> list[Path]:
    if not history:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("loss", ("train_loss", "val_loss"), "Loss"),
        ("ranking", ("roc_auc", "pr_auc"), "Area Under Curve"),
        ("threshold", ("precision", "recall", "f1"), "Threshold Metrics"),
    ]
    paths: list[Path] = []

    for slug, metrics, title in plot_specs:
        fig, axis = plt.subplots(figsize=(7, 4))
        for metric_name in metrics:
            values = [
                (int(record["epoch"]), float(record[metric_name]))
                for record in history
                if record.get(metric_name) is not None
            ]
            if values:
                epochs, metric_values = zip(*values, strict=True)
                axis.plot(epochs, metric_values, marker="o", label=metric_name)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        path = output_dir / f"{run_name}_{slug}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def _build_loggers(cfg: Any, *, run_name: str, git_commit: str) -> tuple[list[Any], Any | None]:
    report_dir = Path(str(cfg_get(cfg.paths, "report_dir", "reports")))
    csv_logger = CSVLogger(save_dir=str(report_dir / "lightning"), name=run_name)
    loggers: list[Any] = [csv_logger]

    logging_cfg = cfg_get(cfg, "logging", {})
    if not bool(cfg_get(logging_cfg, "enabled", False)):
        return loggers, None

    try:
        from lightning.pytorch.loggers import MLFlowLogger
    except ImportError as exc:
        raise RuntimeError(
            "MLflow logging requires the optional `mlops` extra. "
            "Install it with `uv sync --extra mlops`."
        ) from exc

    mlflow_logger = MLFlowLogger(
        experiment_name=str(cfg_get(logging_cfg, "experiment", "isogram")),
        tracking_uri=str(cfg_get(logging_cfg, "tracking_uri", "http://127.0.0.1:8080")),
        run_name=str(cfg_get(logging_cfg, "run_name", run_name)),
    )
    mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "git_commit", git_commit)
    loggers.append(mlflow_logger)
    return loggers, mlflow_logger


def _save_resolved_config(cfg: Any, path: Path) -> None:
    ensure_parent(path)
    try:
        from omegaconf import OmegaConf

        OmegaConf.save(config=cfg, f=path)
    except ImportError:
        write_json(path.with_suffix(".json"), {"config": str(cfg)})


def _log_artifacts(mlflow_logger: Any | None, paths: list[Path], *, artifact_path: str) -> None:
    if mlflow_logger is None:
        return
    for path in paths:
        if path.exists():
            mlflow_logger.experiment.log_artifact(
                mlflow_logger.run_id,
                str(path),
                artifact_path=artifact_path,
            )


def train_from_config(cfg: Any) -> dict[str, Any]:
    data_cfg = cfg.data
    model_cfg = cfg.model
    trainer_cfg = cfg.trainer
    paths_cfg = cfg.paths

    seed = int(cfg_get(trainer_cfg, "seed", cfg_get(data_cfg, "seed", DEFAULT_SEED)))
    set_seed(seed)

    output_dir = Path(str(cfg_get(data_cfg, "output_dir", "data/processed/main")))
    train_path = _path_from_config(data_cfg, "train", output_dir / "train.csv")
    val_path = _path_from_config(data_cfg, "val", output_dir / "val.csv")
    model_type = str(cfg_get(model_cfg, "name", "char_cnn"))
    data_name = str(cfg_get(data_cfg, "name", "main"))
    logging_cfg = cfg_get(cfg, "logging", {})
    run_name = str(cfg_get(logging_cfg, "run_name", f"{model_type}-{data_name}"))

    checkpoint_path = Path(
        str(cfg_get(paths_cfg, "checkpoint", f"artifacts/checkpoints/{model_type}_{data_name}.pt"))
    )
    report_path = Path(str(cfg_get(paths_cfg, "training_report", f"reports/{run_name}.json")))
    run_config_path = Path(
        str(cfg_get(paths_cfg, "run_config", f"reports/run_configs/{run_name}.yaml"))
    )
    plot_dir = Path(str(cfg_get(paths_cfg, "plot_dir", "plots")))

    model, model_config, collate = build_model_stack(model_cfg)
    data_module = TextDataModule(
        train_path=train_path,
        val_path=val_path,
        collate_fn=collate,
        batch_size=int(cfg_get(model_cfg, "batch_size", 32)),
        num_workers=int(cfg_get(trainer_cfg, "num_workers", 0)),
        limit_rows=cfg_get(trainer_cfg, "limit_rows", None),
        seed=seed,
    )
    lightning_model = LightningTextClassifier(
        model=model,
        model_type=model_type,
        learning_rate=float(cfg_get(model_cfg, "learning_rate", 1e-3)),
        weight_decay=float(cfg_get(model_cfg, "weight_decay", 0.0)),
    )

    git_commit = get_git_commit()
    loggers, mlflow_logger = _build_loggers(cfg, run_name=run_name, git_commit=git_commit)
    if mlflow_logger is not None:
        mlflow_logger.log_hyperparams(
            {
                "model": model_type,
                "model_version": cfg_get(model_cfg, "model_version", f"{model_type}-v1"),
                "train_path": train_path,
                "val_path": val_path,
                "batch_size": int(cfg_get(model_cfg, "batch_size", 32)),
                "learning_rate": float(cfg_get(model_cfg, "learning_rate", 1e-3)),
                "max_epochs": int(
                    cfg_get(model_cfg, "max_epochs", cfg_get(model_cfg, "epochs", 1))
                ),
                "limit_rows": cfg_get(trainer_cfg, "limit_rows", None),
                "seed": seed,
                "git_commit": git_commit,
            }
        )

    precision = (
        "16-mixed"
        if bool(cfg_get(model_cfg, "amp", False))
        else str(cfg_get(trainer_cfg, "precision", "32-true"))
    )
    trainer = pl.Trainer(
        accelerator=str(cfg_get(trainer_cfg, "accelerator", "auto")),
        devices=cfg_get(trainer_cfg, "devices", "auto"),
        max_epochs=int(cfg_get(model_cfg, "max_epochs", cfg_get(model_cfg, "epochs", 1))),
        accumulate_grad_batches=int(cfg_get(trainer_cfg, "grad_accum_steps", 1)),
        precision=cast(Any, precision),
        logger=loggers,
        enable_checkpointing=False,
        enable_progress_bar=bool(cfg_get(trainer_cfg, "enable_progress_bar", True)),
        log_every_n_steps=int(cfg_get(trainer_cfg, "log_every_n_steps", 10)),
        deterministic=bool(cfg_get(trainer_cfg, "deterministic", True)),
        num_sanity_val_steps=0,
    )
    trainer.fit(lightning_model, datamodule=data_module)

    if lightning_model.best_state_dict is not None:
        lightning_model.model.load_state_dict(lightning_model.best_state_dict)

    best_metrics = lightning_model.best_metrics or (
        lightning_model.history[-1] if lightning_model.history else {}
    )
    threshold = float(best_metrics.get("threshold", 0.5))
    save_checkpoint(
        checkpoint_path,
        model=lightning_model.model,
        model_type=model_type,
        model_config=model_config,
        metrics=best_metrics,
        threshold=threshold,
        model_version=str(cfg_get(model_cfg, "model_version", f"{model_type}-v1")),
    )

    _save_resolved_config(cfg, run_config_path)
    plot_paths = write_training_plots(
        history=lightning_model.history, output_dir=plot_dir, run_name=run_name
    )
    report = {
        "model": model_type,
        "model_version": str(cfg_get(model_cfg, "model_version", f"{model_type}-v1")),
        "checkpoint": str(checkpoint_path),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "git_commit": git_commit,
        "best_metrics": best_metrics,
        "history": lightning_model.history,
        "plots": [str(path) for path in plot_paths],
    }
    write_json(report_path, report)

    if mlflow_logger is not None:
        if best_metrics:
            for key, value in best_metrics.items():
                if isinstance(value, int | float) and value is not None:
                    mlflow_logger.experiment.log_metric(
                        mlflow_logger.run_id,
                        f"best_{key}",
                        float(value),
                    )
        _log_artifacts(mlflow_logger, [report_path], artifact_path="reports")
        _log_artifacts(mlflow_logger, [checkpoint_path], artifact_path="checkpoints")
        _log_artifacts(mlflow_logger, [run_config_path], artifact_path="config")
        _log_artifacts(mlflow_logger, plot_paths, artifact_path="plots")

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved training report to {report_path}")
    return report


def main() -> None:
    from isogram.commands import main as commands_main

    commands_main()


if __name__ == "__main__":
    main()
