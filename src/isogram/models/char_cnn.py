from __future__ import annotations

import string
from dataclasses import dataclass, field

import torch
from torch import nn


DEFAULT_CHARS = string.ascii_lowercase + string.digits + string.punctuation + " \n\t"


@dataclass(frozen=True)
class CharTokenizer:
    chars: str = DEFAULT_CHARS
    max_length: int = 2048
    _char_to_id: dict[str, int] = field(init=False, repr=False)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_char_to_id", {char: idx + 2 for idx, char in enumerate(self.chars)}
        )

    def encode(self, text: str) -> torch.Tensor:
        normalized = text.lower()
        ids = [self._char_to_id.get(char, self.unk_id) for char in normalized[: self.max_length]]
        if len(ids) < self.max_length:
            ids.extend([self.pad_id] * (self.max_length - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([self.encode(text) for text in texts], dim=0)


class CharCnnClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int = 64,
        channels: int = 96,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, channels, kernel_size=kernel_size)
            for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))
            pooled.append(torch.amax(features, dim=-1))
        joined = torch.cat(pooled, dim=1)
        return self.classifier(self.dropout(joined)).squeeze(-1)
