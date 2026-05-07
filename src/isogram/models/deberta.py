from __future__ import annotations

import torch
from torch import nn


def _load_transformers() -> tuple[object, object, object]:
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for the DeBERTa model. "
            "Install the project with `python -m pip install -e .`."
        ) from exc
    return AutoConfig, AutoModel, AutoTokenizer


class DebertaTextClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, *, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    @classmethod
    def from_pretrained(cls, model_name: str, *, dropout: float = 0.1) -> "DebertaTextClassifier":
        AutoConfig, AutoModel, _ = _load_transformers()
        config = AutoConfig.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name, config=config)
        return cls(encoder, hidden_size=config.hidden_size, dropout=dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls_token)).squeeze(-1)


class DebertaBatcher:
    def __init__(self, model_name: str, *, max_length: int = 512) -> None:
        _, _, AutoTokenizer = _load_transformers()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
