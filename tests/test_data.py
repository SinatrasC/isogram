from __future__ import annotations

import pandas as pd

from isogram.data import build_dataset as build_dataset_module
from isogram.data.build_dataset import (
    STANDARDIZED_OUTPUT_COLUMNS,
    build_training_dataset,
    load_local_source,
)
from isogram.data.licenses import filter_permissive_source_rows
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


def test_build_training_dataset_from_local_source(tmp_path) -> None:
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "text": [f"human {idx}" for idx in range(4)] + [f"ai {idx}" for idx in range(4)],
            "label": [0] * 4 + [1] * 4,
        }
    ).to_csv(raw_path, index=False)
    output_dir = tmp_path / "processed"

    metadata = build_training_dataset(
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
    for split in ("all", "train", "val", "test"):
        split_frame = pd.read_csv(output_dir / f"{split}.csv")
        assert tuple(split_frame.columns) == STANDARDIZED_OUTPUT_COLUMNS
        assert set(split_frame["label"]) == {0, 1}


def test_license_filter_drops_non_permissive_daigt_sources() -> None:
    frame = pd.DataFrame(
        {
            "text": ["allowed chatgpt", "blocked persuade", "blocked llama", "allowed radek"],
            "label": [1, 0, 1, 1],
            "source": ["chat_gpt_moth", "persuade_corpus", "llama_70b_v1", "radek_500"],
        }
    )

    filtered, metadata = filter_permissive_source_rows(frame)

    assert filtered["text"].tolist() == ["allowed chatgpt", "allowed radek"]
    assert filtered["source_detail"].tolist() == ["chat_gpt_moth", "radek_500"]
    assert set(filtered["source_license"]) == {"mit", "cc0-1.0"}
    assert metadata["enabled"] is True
    assert metadata["rows_removed"] == 2
    assert metadata["removed_sources"] == {"persuade_corpus": 1, "llama_70b_v1": 1}


def test_local_source_loader_filters_non_permissive_daigt_sources(tmp_path) -> None:
    raw_path = tmp_path / "daigt.csv"
    pd.DataFrame(
        {
            "text": ["allowed chatgpt", "blocked persuade", "allowed falcon"],
            "label": [1, 0, 1],
            "source": ["chat_gpt_moth", "persuade_corpus", "falcon_180b_v1"],
        }
    ).to_csv(raw_path, index=False)

    frame, metadata = load_local_source(
        raw_path=raw_path,
        source_name="local-daigt",
        declared_license="mixed",
        sample_rows=None,
        seed=42,
    )

    assert set(frame["text"]) == {"allowed chatgpt", "allowed falcon"}
    assert set(frame["source_detail"]) == {"chat_gpt_moth", "falcon_180b_v1"}
    assert set(frame["source_license"]) == {"mit", "apache-2.0"}
    assert metadata["license_filter"]["rows_removed"] == 1


def test_build_hf_split_dataset_preserves_public_splits(tmp_path, monkeypatch) -> None:
    def fake_load_hf_split(*, dataset_name: str, split: str) -> pd.DataFrame:
        assert dataset_name == "example/permissive-splits"
        rows_by_split = {
            "train": 4,
            "validation": 2,
            "test": 2,
        }
        return pd.DataFrame(
            {
                "text": [f"{split} human", f"{split} ai"] * (rows_by_split[split] // 2),
                "label": [0, 1] * (rows_by_split[split] // 2),
                "source_dataset": ["example/permissive-splits"] * rows_by_split[split],
                "source_detail": [split] * rows_by_split[split],
                "source_license": ["mit"] * rows_by_split[split],
                "upstream_url": ["https://example.test/dataset"] * rows_by_split[split],
            }
        )

    monkeypatch.setattr(build_dataset_module, "load_hf_split", fake_load_hf_split)

    metadata = build_dataset_module.build_hf_split_dataset(
        output_dir=tmp_path,
        dataset_name="example/permissive-splits",
        declared_license="mit",
        train_split="train",
        val_split="validation",
        test_split="test",
        sample_rows=8,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42,
    )

    assert metadata["rows_total"] == 8
    assert metadata["rows_train"] == 4
    assert metadata["rows_val"] == 2
    assert metadata["rows_test"] == 2
    assert metadata["sources"][0]["kind"] == "huggingface_presplit"
    for split in ("all", "train", "val", "test"):
        split_frame = pd.read_csv(tmp_path / f"{split}.csv")
        assert tuple(split_frame.columns) == STANDARDIZED_OUTPUT_COLUMNS
        assert set(split_frame["source_license"]) == {"mit"}


def test_build_hf_split_dataset_samples_inside_public_splits(tmp_path, monkeypatch) -> None:
    def fake_load_hf_split(*, dataset_name: str, split: str) -> pd.DataFrame:
        assert dataset_name == "example/main-splits"
        rows_by_split = {
            "train": 8,
            "validation": 4,
            "test": 4,
        }
        row_count = rows_by_split[split]
        half = row_count // 2
        labels = [0] * half + [1] * half
        return pd.DataFrame(
            {
                "text": [f"{split} row {index}" for index in range(row_count)],
                "label": labels,
                "source_dataset": ["example/main-splits"] * row_count,
                "source_detail": [split] * row_count,
                "source_license": ["mit"] * row_count,
                "upstream_url": ["https://example.test/dataset"] * row_count,
            }
        )

    monkeypatch.setattr(build_dataset_module, "load_hf_split", fake_load_hf_split)

    metadata = build_dataset_module.build_hf_split_dataset(
        output_dir=tmp_path,
        dataset_name="example/main-splits",
        declared_license="mit",
        train_split="train",
        val_split="validation",
        test_split="test",
        sample_rows=8,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42,
    )

    assert metadata["rows_total"] == 8
    assert metadata["rows_train"] == 4
    assert metadata["rows_val"] == 2
    assert metadata["rows_test"] == 2
    assert metadata["sources"][0]["sampled_rows_by_split"] == {
        "train": 4,
        "val": 2,
        "test": 2,
    }

    expected_source_details = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }
    for split, expected_source_detail in expected_source_details.items():
        split_frame = pd.read_csv(tmp_path / f"{split}.csv")
        assert tuple(split_frame.columns) == STANDARDIZED_OUTPUT_COLUMNS
        assert set(split_frame["source_detail"]) == {expected_source_detail}
        assert set(split_frame["label"]) == {0, 1}
