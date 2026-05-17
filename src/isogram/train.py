from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from isogram.checkpoint import save_checkpoint
from isogram.config import CommonPaths, TrainingDefaults, set_seed, write_json
from isogram.data.splits import Split
from isogram.metrics import compute_classification_report
from isogram.models.char_cnn import CharCnnClassifier, CharTokenizer
from isogram.models.deberta import DebertaBatcher, DebertaTextClassifier
from isogram.tracking import maybe_mlflow_run


class TextCsvDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {"text": self.texts[index], "label": self.labels[index]}


def read_split(path: Path, *, limit_rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if limit_rows is not None:
        frame = frame.head(limit_rows)
    return frame


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
            assert char_tokenizer is not None
            return {"input_ids": char_tokenizer.batch_encode(texts), "labels": labels}
        assert deberta_batcher is not None
        batch = deberta_batcher(texts)
        batch["labels"] = labels
        return batch

    return collate


def forward_model(
    model: nn.Module,
    model_type: str,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    if model_type == "char_cnn":
        return model(batch["input_ids"])
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    *,
    model: nn.Module,
    model_type: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    grad_accum_steps: int,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        labels = batch.pop("labels")
        context = torch.autocast(device_type="cuda", enabled=True) if amp else nullcontext()
        with context:
            logits = forward_model(model, model_type, batch)
            loss = criterion(logits, labels)
            scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        should_step = (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        if log_every > 0 and (step + 1) % log_every == 0:
            print(
                f"  batch={step + 1}/{len(loader)} "
                f"avg_train_loss={total_loss / max(total_examples, 1):.4f}",
                flush=True,
            )

    return total_loss / max(total_examples, 1)


def predict_scores(
    *,
    model: nn.Module,
    model_type: str,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    model.eval()
    labels_out: list[int] = []
    scores_out: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            labels = batch.pop("labels")
            logits = forward_model(model, model_type, batch)
            scores = torch.sigmoid(logits).detach().cpu().tolist()
            labels_out.extend(int(label) for label in labels.detach().cpu().tolist())
            scores_out.extend(float(score) for score in scores)
    return labels_out, scores_out


def build_model_stack(args: argparse.Namespace) -> tuple[nn.Module, dict[str, Any], Callable]:
    if args.model == "char_cnn":
        tokenizer = CharTokenizer(max_length=args.char_max_length)
        model_config = {
            "chars": tokenizer.chars,
            "max_length": tokenizer.max_length,
            "embedding_dim": args.char_embedding_dim,
            "channels": args.char_channels,
            "kernel_sizes": tuple(args.char_kernel_sizes),
            "dropout": args.dropout,
        }
        model: nn.Module = CharCnnClassifier(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=args.char_embedding_dim,
            channels=args.char_channels,
            kernel_sizes=tuple(args.char_kernel_sizes),
            dropout=args.dropout,
        )
        collate = make_collate_fn(
            model_type=args.model,
            char_tokenizer=tokenizer,
            deberta_batcher=None,
        )
        return model, model_config, collate

    batcher = DebertaBatcher(args.deberta_model_name, max_length=args.deberta_max_length)
    model_config = {
        "model_name": args.deberta_model_name,
        "max_length": args.deberta_max_length,
        "dropout": args.dropout,
    }
    model = DebertaTextClassifier.from_pretrained(args.deberta_model_name, dropout=args.dropout)
    collate = make_collate_fn(
        model_type=args.model,
        char_tokenizer=None,
        deberta_batcher=batcher,
    )
    return model, model_config, collate


def default_output_path(model_type: str) -> Path:
    return CommonPaths().checkpoint_dir / f"{model_type}_best.pt"


def build_parser() -> argparse.ArgumentParser:
    paths = CommonPaths()
    defaults = TrainingDefaults()
    parser = argparse.ArgumentParser(description="Train Isogram PyTorch models.")
    parser.add_argument("--model", choices=["char_cnn", "deberta"], required=True)
    parser.add_argument("--train", type=Path, default=paths.processed_dir / Split.TRAIN.filename)
    parser.add_argument("--val", type=Path, default=paths.processed_dir / Split.VAL.filename)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--model-version")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--char-max-length", type=int, default=defaults.char_max_length)
    parser.add_argument("--char-embedding-dim", type=int, default=64)
    parser.add_argument("--char-channels", type=int, default=96)
    parser.add_argument("--char-kernel-sizes", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--deberta-model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--deberta-max-length", type=int, default=defaults.deberta_max_length)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default="file:mlruns")
    parser.add_argument("--mlflow-experiment", default="isogram")
    parser.add_argument("--mlflow-run-name")
    parser.add_argument("--run-config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    defaults = TrainingDefaults()
    set_seed(args.seed)

    if args.output is None:
        args.output = default_output_path(args.model)
    if args.report_output is None:
        args.report_output = CommonPaths().report_dir / f"{args.model}_training.json"
    if args.model_version is None:
        args.model_version = f"{args.model}-v1"
    if args.epochs is None:
        args.epochs = defaults.char_epochs if args.model == "char_cnn" else defaults.deberta_epochs
    if args.batch_size is None:
        args.batch_size = (
            defaults.char_batch_size if args.model == "char_cnn" else defaults.deberta_batch_size
        )
    if args.learning_rate is None:
        args.learning_rate = (
            defaults.char_learning_rate
            if args.model == "char_cnn"
            else defaults.deberta_learning_rate
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    train_frame = read_split(args.train, limit_rows=args.limit_rows)
    val_frame = read_split(args.val, limit_rows=args.limit_rows)
    model, model_config, collate = build_model_stack(args)
    model.to(device)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TextCsvDataset(train_frame),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    val_loader = DataLoader(
        TextCsvDataset(val_frame),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_score = -1.0
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    with maybe_mlflow_run(
        enabled=args.mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name or args.model_version,
        tags={"model": args.model, "model_version": args.model_version},
    ) as mlflow_run:
        if mlflow_run is not None:
            mlflow_run.log_params(
                {
                    "model": args.model,
                    "model_version": args.model_version,
                    "train_path": args.train,
                    "val_path": args.val,
                    "rows_train": len(train_frame),
                    "rows_val": len(val_frame),
                    "device": str(device),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "grad_accum_steps": args.grad_accum_steps,
                    "amp": use_amp,
                    "limit_rows": args.limit_rows,
                    "dropout": args.dropout,
                    "char_max_length": args.char_max_length,
                    "char_embedding_dim": args.char_embedding_dim,
                    "char_channels": args.char_channels,
                    "char_kernel_sizes": args.char_kernel_sizes,
                    "deberta_model_name": args.deberta_model_name,
                    "deberta_max_length": args.deberta_max_length,
                    "seed": args.seed,
                }
            )
            metadata_path = args.train.parent / "metadata.json"
            mlflow_run.log_artifact(metadata_path, artifact_path="data")
            if args.run_config is not None:
                mlflow_run.log_artifact(args.run_config, artifact_path="config")

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                model_type=args.model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                amp=use_amp,
                grad_accum_steps=args.grad_accum_steps,
                log_every=args.log_every,
            )
            labels, scores = predict_scores(
                model=model,
                model_type=args.model,
                loader=val_loader,
                device=device,
            )
            metrics = compute_classification_report(labels, scores)
            metrics["train_loss"] = train_loss
            metrics["epoch"] = epoch
            history.append(metrics)

            if mlflow_run is not None:
                mlflow_run.log_metrics(metrics, step=epoch)

            score = metrics["roc_auc"] if metrics["roc_auc"] is not None else metrics["f1"]
            if float(score) > best_score:
                best_score = float(score)
                best_metrics = metrics
                save_checkpoint(
                    args.output,
                    model=model,
                    model_type=args.model,
                    model_config=model_config,
                    metrics=metrics,
                    threshold=float(metrics["threshold"]),
                    model_version=args.model_version,
                )
            print(f"epoch={epoch} train_loss={train_loss:.4f} metrics={metrics}")

        report = {
            "model": args.model,
            "model_version": args.model_version,
            "checkpoint": str(args.output),
            "device": str(device),
            "train_path": str(args.train),
            "val_path": str(args.val),
            "rows_train": len(train_frame),
            "rows_val": len(val_frame),
            "best_metrics": best_metrics,
            "history": history,
        }
        write_json(args.report_output, report)
        if mlflow_run is not None:
            if best_metrics is not None:
                mlflow_run.log_metrics(
                    {f"best_{key}": value for key, value in best_metrics.items()}
                )
            mlflow_run.log_artifact(args.report_output, artifact_path="reports")
            mlflow_run.log_artifact(args.output, artifact_path="checkpoints")

    print(f"Saved checkpoint to {args.output}")
    print(f"Saved report to {args.report_output}")


if __name__ == "__main__":
    main()
