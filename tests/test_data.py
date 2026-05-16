from __future__ import annotations

import pandas as pd

from isogram.data.build_dataset import build_merged_dataset
from isogram.data.schema import (
    normalize_frame,
    sample_balanced_by_label,
    stratified_train_val_split,
    stratified_train_val_test_split,
)
from isogram.data.splits import Split


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


def test_split_enum_maps_to_csv_filenames() -> None:
    assert Split.TRAIN.filename == "train.csv"
    assert Split.VAL.filename == "val.csv"
    assert Split.TEST.filename == "test.csv"
    assert Split.ALL.filename == "all.csv"


def test_balanced_sampling_caps_rows_per_label() -> None:
    frame = pd.DataFrame(
        {
            "text": [f"human {idx}" for idx in range(10)] + [f"ai {idx}" for idx in range(10)],
            "label": [0] * 10 + [1] * 10,
        }
    )

    sampled = sample_balanced_by_label(frame, max_rows=6, seed=42)

    assert len(sampled) == 6
    assert sampled["label"].value_counts().to_dict() == {0: 3, 1: 3}


def test_stratified_train_val_test_split_keeps_both_classes() -> None:
    frame = pd.DataFrame(
        {
            "text": [f"human {idx}" for idx in range(10)] + [f"ai {idx}" for idx in range(10)],
            "label": [0] * 10 + [1] * 10,
        }
    )

    train, val, test = stratified_train_val_test_split(
        frame,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )

    assert len(train) == 12
    assert len(val) == 4
    assert len(test) == 4
    assert set(train["label"]) == {0, 1}
    assert set(val["label"]) == {0, 1}
    assert set(test["label"]) == {0, 1}


def test_build_merged_dataset_from_local_source(tmp_path) -> None:
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "text": [f"human {idx}" for idx in range(4)] + [f"ai {idx}" for idx in range(4)],
            "label": [0] * 4 + [1] * 4,
        }
    ).to_csv(raw_path, index=False)
    output_dir = tmp_path / "processed"

    metadata = build_merged_dataset(
        output_dir=output_dir,
        local_paths=[raw_path],
        local_source_name="local-test",
        local_license="mit",
        local_sample_rows=None,
        hf_dataset="unused",
        hf_split="train",
        hf_sample_rows=2,
        hf_license="mit",
        hf_shuffle_buffer=1,
        skip_hf=True,
        val_fraction=0.25,
        test_fraction=0.25,
        seed=42,
    )

    assert metadata["rows_total"] == 8
    assert metadata["rows_train"] == 4
    assert metadata["rows_val"] == 2
    assert metadata["rows_test"] == 2
    assert (output_dir / "train.csv").exists()
    assert (output_dir / "val.csv").exists()
    assert (output_dir / "test.csv").exists()
