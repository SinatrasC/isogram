from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from isogram.config import ensure_parent
from isogram.models.char_cnn import CharCnnClassifier, CharTokenizer
from isogram.models.deberta import DebertaTextClassifier


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    model_type: str,
    model_config: dict[str, Any],
    metrics: dict[str, Any],
    threshold: float,
    model_version: str,
) -> None:
    ensure_parent(path)
    torch.save(
        {
            "model_type": model_type,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "threshold": threshold,
            "model_version": model_version,
        },
        path,
    )


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def build_model_from_checkpoint(payload: dict[str, Any]) -> nn.Module:
    model_type = payload["model_type"]
    config = payload.get("model_config", {})
    if model_type == "char_cnn":
        tokenizer = CharTokenizer(
            chars=config.get("chars", CharTokenizer().chars),
            max_length=int(config.get("max_length", 2048)),
        )
        return CharCnnClassifier(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=int(config.get("embedding_dim", 64)),
            channels=int(config.get("channels", 96)),
            kernel_sizes=tuple(config.get("kernel_sizes", (3, 5, 7))),
            dropout=float(config.get("dropout", 0.2)),
        )
    if model_type == "deberta":
        return DebertaTextClassifier.from_pretrained(
            config.get("model_name", "microsoft/deberta-v3-base"),
            dropout=float(config.get("dropout", 0.1)),
        )
    raise ValueError(f"Unknown model_type: {model_type}")
