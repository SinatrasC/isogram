from __future__ import annotations

import torch

from isogram.models.char_cnn import CharCnnClassifier, CharTokenizer


def test_char_cnn_forward_shape() -> None:
    tokenizer = CharTokenizer(max_length=32)
    model = CharCnnClassifier(vocab_size=tokenizer.vocab_size, channels=8)
    input_ids = tokenizer.batch_encode(["hello", "generated text"])

    logits = model(input_ids)

    assert logits.shape == torch.Size([2])
