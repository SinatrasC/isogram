from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

from isogram.data.licenses import filter_permissive_source_rows
from isogram.data.schema import (
    LABEL_COLUMNS,
    TEXT_COLUMNS,
    normalize_label,
    stratified_train_val_test_split,
)


DEFAULT_REPO_ID = "sinatras/isogram-ai-text-detection-splits"
DEFAULT_OUTPUT_DIR = Path("data/processed/hf_upload_splits")
DEFAULT_DAIGT_PATH = Path("data/raw/daigt-v2/train_v2_drcat_02.csv")
DEFAULT_HF_DATASET = "srikanthgali/ai-text-detection-pile-cleaned"


def find_key(row: Mapping[str, Any], candidates: Iterable[str], *, kind: str) -> str:
    for candidate in candidates:
        if candidate in row:
            return candidate
    raise ValueError(f"No {kind} key found in row. Expected one of {tuple(candidates)}")


def collect_hf_rows(
    *,
    dataset_name: str,
    human_target: int,
    ai_target: int,
    seed: int,
    shuffle_buffer: int,
) -> pd.DataFrame:
    streamed = load_dataset(dataset_name, split="train", streaming=True)
    if shuffle_buffer > 0:
        streamed = streamed.shuffle(seed=seed, buffer_size=shuffle_buffer)

    targets = {0: human_target, 1: ai_target}
    counts = {0: 0, 1: 0}
    rows: list[dict[str, object]] = []

    for raw_row in streamed:
        text_key = find_key(raw_row, TEXT_COLUMNS, kind="text")
        label_key = find_key(raw_row, LABEL_COLUMNS, kind="label")
        label = normalize_label(raw_row[label_key])
        if counts[label] >= targets[label]:
            continue

        text = str(raw_row[text_key]).strip()
        if not text:
            continue

        rows.append(
            {
                "text": text,
                "label": label,
                "source_dataset": dataset_name,
                "source_detail": dataset_name,
                "source_license": "mit",
                "upstream_url": "https://huggingface.co/datasets/srikanthgali/ai-text-detection-pile-cleaned",
            }
        )
        counts[label] += 1
        if counts == targets:
            break

    if counts != targets:
        raise RuntimeError(f"Collected {counts}, but target was {targets}")
    return pd.DataFrame(rows)


def load_permissive_daigt_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["text", "label", "source"])
    frame, _ = filter_permissive_source_rows(frame)
    frame["text"] = frame["text"].astype("string").fillna("").str.strip()
    frame = frame[frame["text"].str.len() > 0]
    frame["label"] = frame["label"].map(normalize_label).astype("int64")
    output_columns = [
        "text",
        "label",
        "source_dataset",
        "source_detail",
        "source_license",
        "upstream_url",
    ]
    return frame[output_columns].reset_index(drop=True)


def remove_duplicate_texts(frame: pd.DataFrame) -> pd.DataFrame:
    conflict_mask = frame.groupby("text")["label"].transform("nunique") > 1
    without_conflicts = frame.loc[~conflict_mask].copy()
    return without_conflicts.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)


def source_counts(frame: pd.DataFrame) -> dict[str, dict[str, int | str]]:
    counts = (
        frame.groupby(["source_detail", "source_license"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    return {
        str(row.source_detail): {
            "rows": int(row.rows),
            "source_license": str(row.source_license),
        }
        for row in counts.itertuples(index=False)
    }


def parse_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def build_dataset_card(metadata: dict[str, Any]) -> str:
    return f"""---
license: other
language:
  - en
pretty_name: Isogram AI Text Detection Permissive Splits
task_categories:
  - text-classification
tags:
  - ai-detection
  - human-vs-ai
  - text-classification
  - permissive-license
size_categories:
  - 10K<n<100K
---

# Isogram AI Text Detection Permissive Splits

This dataset contains train/validation/test splits for binary AI-generated text detection.
It is built from sources whose dataset-level licenses were checked as permissive or
public-domain-compatible.

## Schema

- `text`: essay text.
- `label`: `0` for human-written text, `1` for AI-generated text.
- `source_dataset`: upstream dataset identifier.
- `source_detail`: source label retained from the upstream data.
- `source_license`: row-level upstream license marker.
- `upstream_url`: source URL used for provenance.

## Splits

- Train rows: {metadata["rows_train"]}
- Validation rows: {metadata["rows_val"]}
- Test rows: {metadata["rows_test"]}
- Label counts: `{json.dumps(metadata["label_counts"], sort_keys=True)}`

## Source Licenses

The dataset is marked `other` because it combines several permissive licenses:

- MIT: `srikanthgali/ai-text-detection-pile-cleaned`
- MIT: `alejopaullier/daigt-external-dataset`
- Apache 2.0: Falcon rows from `nbroad/daigt-data-llama-70b-and-falcon180b`
- CC0 1.0: `radek1/llm-generated-essays`
- CC0 1.0: `kingki19/llm-generated-essay-using-palm-from-google-gen-ai`

Excluded examples include PERSUADE corpus rows because that source is CC BY-NC-SA 4.0,
Llama rows because they are subject to the Llama license, and Claude rows because the
source dataset metadata reports an unknown license.

## Build Parameters

```json
{json.dumps(metadata, indent=2, sort_keys=True)}
```
"""


def write_split_files(dataset: DatasetDict, output_dir: Path, metadata: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_dataset in dataset.items():
        split_frame = split_dataset.to_pandas()
        split_frame.to_parquet(output_dir / f"{split_name}.parquet", index=False)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(build_dataset_card(metadata), encoding="utf-8")
    return readme_path


def push_dataset(dataset: DatasetDict, *, repo_id: str, readme_path: Path) -> str:
    dataset.push_to_hub(repo_id, private=False)
    HfApi().upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        commit_message="Add permissive split dataset card",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def main(
    repo_id: str = DEFAULT_REPO_ID,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    daigt_path: str | Path = DEFAULT_DAIGT_PATH,
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_human_rows: int = 30000,
    seed: int = 42,
    shuffle_buffer: int = 10000,
    push: bool | str = True,
) -> dict[str, Any]:
    daigt_frame = load_permissive_daigt_rows(Path(daigt_path))
    hf_ai_rows = max(0, hf_human_rows - int((daigt_frame["label"] == 1).sum()))
    hf_frame = collect_hf_rows(
        dataset_name=hf_dataset,
        human_target=hf_human_rows,
        ai_target=hf_ai_rows,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
    )
    combined = remove_duplicate_texts(pd.concat([hf_frame, daigt_frame], ignore_index=True))
    train, validation, test = stratified_train_val_test_split(
        combined,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=seed,
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train, preserve_index=False),
            "validation": Dataset.from_pandas(validation, preserve_index=False),
            "test": Dataset.from_pandas(test, preserve_index=False),
        }
    )
    metadata = {
        "repo_id": repo_id,
        "seed": seed,
        "hf_dataset": hf_dataset,
        "hf_human_rows": hf_human_rows,
        "hf_ai_rows": hf_ai_rows,
        "local_daigt_rows": int(len(daigt_frame)),
        "rows_total": int(len(combined)),
        "rows_train": int(len(train)),
        "rows_val": int(len(validation)),
        "rows_test": int(len(test)),
        "label_counts": {
            str(label): int(count) for label, count in combined["label"].value_counts().items()
        },
        "source_counts": source_counts(combined),
    }
    readme_path = write_split_files(dataset, Path(output_dir), metadata)
    if parse_bool(push):
        metadata["hub_url"] = push_dataset(dataset, repo_id=repo_id, readme_path=readme_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


if __name__ == "__main__":
    import fire

    fire.Fire(main)
