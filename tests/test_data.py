from __future__ import annotations

import pandas as pd

from isogram.data.schema import normalize_frame, stratified_train_val_split


def test_normalize_frame_maps_text_and_label_columns() -> None:
    frame = pd.DataFrame(
        {
            "essay": ["Human sentence.", "Generated sentence."],
            "generated": [0, 1],
            "source": ["human", "llm"],
        }
    )

    normalized = normalize_frame(frame)

    assert list(normalized.columns) == ["text", "label", "source"]
    assert normalized["text"].tolist() == ["Human sentence.", "Generated sentence."]
    assert normalized["label"].tolist() == [0, 1]


def test_stratified_split_keeps_both_classes() -> None:
    frame = pd.DataFrame(
        {
            "text": [f"human {idx}" for idx in range(4)] + [f"ai {idx}" for idx in range(4)],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    train, val = stratified_train_val_split(frame, val_fraction=0.25, seed=42)

    assert len(train) == 6
    assert len(val) == 2
    assert set(train["label"]) == {0, 1}
    assert set(val["label"]) == {0, 1}
