from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from isogram.checkpoint import save_checkpoint
from isogram.models.char_cnn import CharCnnClassifier, CharTokenizer
from isogram.serve import create_app


def test_api_predicts_with_char_checkpoint(tmp_path: Path) -> None:
    tokenizer = CharTokenizer(max_length=32)
    model = CharCnnClassifier(vocab_size=tokenizer.vocab_size, channels=8)
    checkpoint = tmp_path / "char.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        model_type="char_cnn",
        model_config={
            "chars": tokenizer.chars,
            "max_length": tokenizer.max_length,
            "embedding_dim": 64,
            "channels": 8,
            "kernel_sizes": (3, 5, 7),
            "dropout": 0.2,
        },
        metrics={"f1": 0.0},
        threshold=0.5,
        model_version="char-test",
    )

    client = TestClient(create_app(checkpoint=checkpoint, device="cpu"))

    health = client.get("/health")
    prediction = client.post("/predict", json={"text": "A short essay."})
    empty = client.post("/predict", json={"text": "   "})

    assert health.status_code == 200
    assert health.json()["model_version"] == "char-test"
    assert prediction.status_code == 200
    assert set(prediction.json()) == {"prob_ai", "model_version"}
    assert empty.status_code == 422
