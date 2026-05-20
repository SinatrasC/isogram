from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


TEXT_COLUMNS = ("text", "essay", "full_text", "content", "abstract")
LABEL_COLUMNS = ("generated", "label", "target", "is_generated")
OPTIONAL_METADATA_COLUMNS = (
    "prompt_name",
    "prompt",
    "source",
    "model",
    "source_dataset",
    "source_detail",
    "source_license",
    "upstream_url",
    "source_split",
)


def find_text_column(frame: pd.DataFrame) -> str:
    for column in TEXT_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError(f"No text column found. Expected one of: {', '.join(TEXT_COLUMNS)}")


def find_label_column(frame: pd.DataFrame) -> str:
    for column in LABEL_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError(f"No label column found. Expected one of: {', '.join(LABEL_COLUMNS)}")


def normalize_label(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Label is missing")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "generated", "ai", "llm", "machine"}:
            return 1
        if normalized in {"0", "false", "human", "student", "real"}:
            return 0
        try:
            label = int(float(normalized))
        except ValueError:
            raise ValueError(f"Label must be binary, got {value!r}") from None
        if label not in {0, 1}:
            raise ValueError(f"Label must be binary, got {value!r}")
        return label
    label = int(cast(Any, value))
    if label not in {0, 1}:
        raise ValueError(f"Label must be binary, got {value!r}")
    return label


def normalize_frame(frame: pd.DataFrame, *, require_binary: bool = True) -> pd.DataFrame:
    text_column = find_text_column(frame)
    label_column = find_label_column(frame)

    normalized = pd.DataFrame(
        {
            "text": frame[text_column].astype("string").fillna("").str.strip(),
            "label": frame[label_column].map(normalize_label).astype("int64"),
        }
    )

    for column in OPTIONAL_METADATA_COLUMNS:
        if column in frame.columns:
            normalized[column] = frame[column].astype("string").fillna("")

    normalized = normalized[normalized["text"].str.len() > 0]
    normalized = normalized.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    if normalized.empty:
        raise ValueError("No usable rows after normalization")
    if require_binary and normalized["label"].nunique() < 2:
        raise ValueError("Dataset must contain both human and generated examples")
    return normalized


def stratified_train_val_split(
    frame: pd.DataFrame,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for _, group in frame.groupby("label", sort=True):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_fraction)))
        if val_count >= len(indices):
            val_count = max(1, len(indices) - 1)
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train = frame.loc[train_indices].reset_index(drop=True)
    val = frame.loc[val_indices].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError("Split produced an empty train or validation set")
    return train, val


def sample_balanced_by_label(
    frame: pd.DataFrame,
    *,
    max_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_rows is None or max_rows >= len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_rows < 2:
        raise ValueError("max_rows must be at least 2 for a binary dataset")

    labels = sorted(int(label) for label in frame["label"].unique())
    if set(labels) != {0, 1}:
        raise ValueError("Balanced sampling expects binary labels 0 and 1")

    rng = np.random.default_rng(seed)
    target_by_label = {0: max_rows // 2, 1: max_rows - (max_rows // 2)}
    sampled_indices: list[int] = []
    remaining_indices: list[int] = []

    for label in labels:
        group_indices = frame.index[frame["label"] == label].to_numpy()
        rng.shuffle(group_indices)
        target = min(target_by_label[label], len(group_indices))
        sampled_indices.extend(group_indices[:target].tolist())
        remaining_indices.extend(group_indices[target:].tolist())

    if len(sampled_indices) < max_rows and remaining_indices:
        remaining = np.array(remaining_indices)
        rng.shuffle(remaining)
        sampled_indices.extend(remaining[: max_rows - len(sampled_indices)].tolist())

    rng.shuffle(sampled_indices)
    return frame.loc[sampled_indices].reset_index(drop=True)


def stratified_train_val_test_split(
    frame: pd.DataFrame,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be below 1")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for _, group in frame.groupby("label", sort=True):
        indices = group.index.to_numpy()
        rng.shuffle(indices)

        test_count = max(1, int(round(len(indices) * test_fraction)))
        val_count = max(1, int(round(len(indices) * val_fraction)))
        if test_count + val_count >= len(indices):
            test_count = max(1, min(test_count, len(indices) - 2))
            val_count = max(1, min(val_count, len(indices) - test_count - 1))

        test_indices.extend(indices[:test_count].tolist())
        val_indices.extend(indices[test_count : test_count + val_count].tolist())
        train_indices.extend(indices[test_count + val_count :].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    train = frame.loc[train_indices].reset_index(drop=True)
    val = frame.loc[val_indices].reset_index(drop=True)
    test = frame.loc[test_indices].reset_index(drop=True)
    if train.empty or val.empty or test.empty:
        raise ValueError("Split produced an empty train, validation, or test set")
    return train, val, test


def find_csv(raw_path: Path) -> Path:
    if raw_path.is_file():
        return raw_path
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw path does not exist: {raw_path}")

    preferred_names = (
        "train_v2_drcat_02.csv",
        "train_v2_drcat_01.csv",
        "daigt-v2-train-dataset.csv",
        "train.csv",
    )
    for name in preferred_names:
        candidate = raw_path / name
        if candidate.exists():
            return candidate

    csv_files = sorted(raw_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found below {raw_path}")
    return csv_files[0]
