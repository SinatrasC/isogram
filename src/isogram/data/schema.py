from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TEXT_COLUMNS = ("text", "essay", "full_text", "content")
LABEL_COLUMNS = ("generated", "label", "target", "is_generated")
OPTIONAL_METADATA_COLUMNS = ("prompt_name", "prompt", "source", "model")


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
    label = int(value)
    if label not in {0, 1}:
        raise ValueError(f"Label must be binary, got {value!r}")
    return label


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
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
    if normalized["label"].nunique() < 2:
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
