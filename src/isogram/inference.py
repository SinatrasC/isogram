from __future__ import annotations

from pathlib import Path

import torch

from isogram import DEFAULT_MODEL_VERSION
from isogram.checkpoint import build_model_from_checkpoint, load_checkpoint
from isogram.models.char_cnn import CharTokenizer
from isogram.models.deberta import DebertaBatcher


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Predictor:
    def __init__(self, checkpoint_path: Path, *, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device)
        payload = load_checkpoint(self.checkpoint_path, map_location=self.device)
        self.payload = payload
        self.model_type = payload["model_type"]
        self.model_config = payload.get("model_config", {})
        self.model_version = payload.get("model_version", DEFAULT_MODEL_VERSION)
        self.threshold = float(payload.get("threshold", 0.5))

        self.model = build_model_from_checkpoint(payload)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.char_tokenizer: CharTokenizer | None = None
        self.deberta_batcher: DebertaBatcher | None = None
        if self.model_type == "char_cnn":
            self.char_tokenizer = CharTokenizer(
                chars=self.model_config.get("chars", CharTokenizer().chars),
                max_length=int(self.model_config.get("max_length", 2048)),
            )
        elif self.model_type == "deberta":
            self.deberta_batcher = DebertaBatcher(
                self.model_config.get("model_name", "microsoft/deberta-v3-base"),
                max_length=int(self.model_config.get("max_length", 512)),
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _make_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
        if self.model_type == "char_cnn":
            assert self.char_tokenizer is not None
            return {"input_ids": self.char_tokenizer.batch_encode(texts).to(self.device)}
        assert self.deberta_batcher is not None
        batch = self.deberta_batcher(texts)
        return {key: value.to(self.device) for key, value in batch.items()}

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.model_type == "char_cnn":
            return self.model(batch["input_ids"])
        return self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    def predict_batch(self, texts: list[str], *, batch_size: int = 32) -> list[float]:
        if not texts:
            return []
        probabilities: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                batch = self._make_batch(batch_texts)
                logits = self._forward(batch)
                probs = torch.sigmoid(logits).detach().cpu().tolist()
                probabilities.extend(float(prob) for prob in probs)
        return probabilities

    def predict(self, text: str) -> float:
        return self.predict_batch([text], batch_size=1)[0]
