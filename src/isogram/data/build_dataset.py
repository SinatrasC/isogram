from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

from isogram.config import CommonPaths, TrainingDefaults, dataclass_to_jsonable, write_json
from isogram.data.licenses import filter_permissive_source_rows
from isogram.data.schema import (
    LABEL_COLUMNS,
    TEXT_COLUMNS,
    find_csv,
    read_tabular,
    normalize_frame,
    normalize_label,
    sample_balanced_by_label,
    stratified_train_val_test_split,
)
from isogram.data.splits import Split


DEFAULT_HF_DATASET = "sinatras/isogram-ai-text-detection-splits"
DEFAULT_HF_LICENSE = "other"
DEFAULT_HF_SPLIT = "train"
DEFAULT_HF_SAMPLE_ROWS = 60_000
DEFAULT_HF_SHUFFLE_BUFFER = 10_000
DEFAULT_HF_PRE_SPLIT = True
DEFAULT_HF_TRAIN_SPLIT = "train"
DEFAULT_HF_VAL_SPLIT = "validation"
DEFAULT_HF_TEST_SPLIT = "test"

STANDARDIZED_OUTPUT_COLUMNS = (
    "text",
    "label",
    "source_dataset",
    "source_detail",
    "source_license",
    "upstream_url",
)


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(label): int(count) for label, count in frame["label"].value_counts().items()}


def _find_mapping_key(row: Mapping[str, Any], candidates: Iterable[str], kind: str) -> str:
    for candidate in candidates:
        if candidate in row:
            return candidate
    raise ValueError(f"No {kind} field found. Expected one of: {', '.join(candidates)}")


def _normalize_mapping_row(
    row: Mapping[str, Any],
    *,
    source_dataset: str,
    source_split: str,
) -> dict[str, object] | None:
    text_key = _find_mapping_key(row, TEXT_COLUMNS, "text")
    label_key = _find_mapping_key(row, LABEL_COLUMNS, "label")
    text = str(row[text_key]).strip()
    if not text:
        return None

    # We deliberately do not copy the HF `source` column: in
    # `srikanthgali/ai-text-detection-pile-cleaned` it holds "human"/"ai" — a
    # string echo of `generated`, not a provenance field. Carrying it would mix
    # semantics with local CSVs whose `source` column holds the generator name.
    return {
        "text": text,
        "label": normalize_label(row[label_key]),
        "source_dataset": source_dataset,
        "source_split": source_split,
    }


def collect_hf_balanced_sample(
    *,
    dataset_name: str,
    split: str,
    sample_rows: int,
    seed: int,
    shuffle_buffer: int,
) -> pd.DataFrame:
    if sample_rows < 2:
        raise ValueError("sample_rows must be at least 2 for a binary dataset")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face dataset loading requires the optional `hf` extra. "
            "Install it with `uv sync --extra hf` or `python -m pip install -e .[hf]`."
        ) from exc

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)

    targets = {0: sample_rows // 2, 1: sample_rows - (sample_rows // 2)}
    counts = {0: 0, 1: 0}
    rows: list[dict[str, object]] = []

    for raw_row in dataset:
        normalized = _normalize_mapping_row(
            raw_row,
            source_dataset=dataset_name,
            source_split=split,
        )
        if normalized is None:
            continue
        label = int(cast(int, normalized["label"]))
        if label not in counts or counts[label] >= targets[label]:
            continue
        rows.append(normalized)
        counts[label] += 1
        if counts == targets:
            break

    if counts != targets:
        raise RuntimeError(
            f"Could only collect {counts} rows from {dataset_name}:{split}; target was {targets}."
        )

    return normalize_frame(pd.DataFrame(rows))


def load_hf_split(*, dataset_name: str, split: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face dataset loading requires the optional `hf` extra. "
            "Install it with `uv sync --extra hf` or `python -m pip install -e .[hf]`."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    frame = dataset.to_pandas()
    normalized = normalize_frame(frame)
    normalized["source_split"] = split
    return normalized


def _allocate_presplit_sample_rows(
    frames: Mapping[str, pd.DataFrame], *, sample_rows: int
) -> dict[str, int]:
    split_sizes = {split: int(len(frame)) for split, frame in frames.items()}
    min_rows_per_split = 2
    min_required = min_rows_per_split * len(split_sizes)
    if sample_rows < min_required:
        raise ValueError(
            f"sample_rows must be at least {min_required} to keep train, validation, "
            "and test splits non-empty with both labels"
        )

    too_small = [split for split, size in split_sizes.items() if size < min_rows_per_split]
    if too_small:
        raise ValueError(f"Cannot sample from splits with fewer than 2 rows: {too_small}")

    total_rows = sum(split_sizes.values())
    raw_targets = {
        split: (sample_rows * split_size) / total_rows for split, split_size in split_sizes.items()
    }
    allocations = {
        split: min(split_sizes[split], max(min_rows_per_split, math.floor(target)))
        for split, target in raw_targets.items()
    }
    split_order = {split: index for index, split in enumerate(split_sizes)}

    while sum(allocations.values()) > sample_rows:
        candidates = [
            split
            for split, allocated_rows in allocations.items()
            if allocated_rows > min_rows_per_split
        ]
        if not candidates:
            raise ValueError("Could not allocate sample rows while preserving all public splits")
        split_to_reduce = max(
            candidates,
            key=lambda split: (
                allocations[split] - raw_targets[split],
                allocations[split],
                -split_order[split],
            ),
        )
        allocations[split_to_reduce] -= 1

    while sum(allocations.values()) < sample_rows:
        candidates = [
            split
            for split, allocated_rows in allocations.items()
            if allocated_rows < split_sizes[split]
        ]
        if not candidates:
            raise ValueError("Could not allocate requested sample rows from available splits")
        split_to_increase = max(
            candidates,
            key=lambda split: (
                raw_targets[split] - allocations[split],
                split_sizes[split],
                -split_order[split],
            ),
        )
        allocations[split_to_increase] += 1

    return allocations


def _sample_presplit_frames(
    *,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    sample_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    frames = {"train": train, "val": val, "test": test}
    allocations = _allocate_presplit_sample_rows(frames, sample_rows=sample_rows)
    sampled = {
        split: sample_balanced_by_label(frame, max_rows=allocations[split], seed=seed + offset)
        for offset, (split, frame) in enumerate(frames.items())
    }
    return sampled["train"], sampled["val"], sampled["test"], allocations


def _write_presplit_dataset(
    *,
    output_dir: Path,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, object]:
    combined = pd.concat([train, val, test], ignore_index=True, sort=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / Split.ALL.filename
    train_path = output_dir / Split.TRAIN.filename
    val_path = output_dir / Split.VAL.filename
    test_path = output_dir / Split.TEST.filename
    _select_output_columns(combined).to_csv(all_path, index=False)
    _select_output_columns(train).to_csv(train_path, index=False)
    _select_output_columns(val).to_csv(val_path, index=False)
    _select_output_columns(test).to_csv(test_path, index=False)
    return {
        "rows_total": int(len(combined)),
        "rows_train": int(len(train)),
        "rows_val": int(len(val)),
        "rows_test": int(len(test)),
        "label_counts": _label_counts(combined),
        "train_label_counts": _label_counts(train),
        "val_label_counts": _label_counts(val),
        "test_label_counts": _label_counts(test),
        "all_path": str(all_path),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
    }


def build_hf_split_dataset(
    *,
    output_dir: Path,
    dataset_name: str,
    declared_license: str,
    train_split: str,
    val_split: str,
    test_split: str,
    sample_rows: int | None,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, object]:
    train = load_hf_split(dataset_name=dataset_name, split=train_split)
    val = load_hf_split(dataset_name=dataset_name, split=val_split)
    test = load_hf_split(dataset_name=dataset_name, split=test_split)
    combined = pd.concat([train, val, test], ignore_index=True, sort=False)
    sampled_rows_by_split: dict[str, int] | None = None

    if sample_rows is not None and sample_rows < len(combined):
        train, val, test, sampled_rows_by_split = _sample_presplit_frames(
            train=train,
            val=val,
            test=test,
            sample_rows=sample_rows,
            seed=seed,
        )

    split_metadata = _write_presplit_dataset(
        output_dir=output_dir,
        train=train,
        val=val,
        test=test,
    )
    rows_total = split_metadata["rows_total"]
    rows_used = rows_total if isinstance(rows_total, int) else int(str(rows_total))

    metadata = {
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "sources": [
            {
                "kind": "huggingface_presplit",
                "dataset": dataset_name,
                "declared_license": declared_license,
                "train_split": train_split,
                "val_split": val_split,
                "test_split": test_split,
                "sample_rows": sample_rows,
                "rows_used": rows_used,
                "sampled_rows_by_split": sampled_rows_by_split,
            }
        ],
        "rows_before_cross_source_deduplication": int(len(combined)),
        "rows_with_conflicting_labels_removed": 0,
        "duplicate_text_rows_removed": 0,
        **split_metadata,
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def load_local_source(
    *,
    raw_path: Path,
    source_name: str,
    declared_license: str,
    sample_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    csv_path = find_csv(raw_path)
    frame = pd.read_csv(csv_path)
    filtered, license_filter = filter_permissive_source_rows(frame)
    normalized = normalize_frame(filtered, require_binary=False)
    if "source_dataset" not in normalized.columns or normalized["source_dataset"].eq("").all():
        normalized["source_dataset"] = source_name
    normalized["source_split"] = "local"
    sampled = sample_balanced_by_label(normalized, max_rows=sample_rows, seed=seed)
    metadata = {
        "kind": "local_csv",
        "source_name": source_name,
        "declared_license": declared_license,
        "source_csv": str(csv_path),
        "license_filter": license_filter,
        "rows_normalized": int(len(normalized)),
        "rows_used": int(len(sampled)),
        "label_counts": _label_counts(sampled),
    }
    return sampled, metadata


def load_local_split(
    *,
    raw_path: Path,
    source_name: str,
    declared_license: str,
    split: str,
) -> pd.DataFrame:
    frame = read_tabular(raw_path)
    normalized = normalize_frame(frame)
    if "source_dataset" not in normalized.columns or normalized["source_dataset"].eq("").all():
        normalized["source_dataset"] = source_name
    if "source_detail" not in normalized.columns or normalized["source_detail"].eq("").all():
        normalized["source_detail"] = source_name
    if "source_license" not in normalized.columns or normalized["source_license"].eq("").all():
        normalized["source_license"] = declared_license
    if "upstream_url" not in normalized.columns:
        normalized["upstream_url"] = ""
    normalized["source_split"] = split
    return normalized


def build_local_split_dataset(
    *,
    output_dir: Path,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    source_name: str,
    declared_license: str,
    sample_rows: int | None,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, object]:
    train = load_local_split(
        raw_path=train_path,
        source_name=source_name,
        declared_license=declared_license,
        split="train",
    )
    val = load_local_split(
        raw_path=val_path,
        source_name=source_name,
        declared_license=declared_license,
        split="validation",
    )
    test = load_local_split(
        raw_path=test_path,
        source_name=source_name,
        declared_license=declared_license,
        split="test",
    )
    combined = pd.concat([train, val, test], ignore_index=True, sort=False)
    sampled_rows_by_split: dict[str, int] | None = None

    if sample_rows is not None and sample_rows < len(combined):
        train, val, test, sampled_rows_by_split = _sample_presplit_frames(
            train=train,
            val=val,
            test=test,
            sample_rows=sample_rows,
            seed=seed,
        )

    split_metadata = _write_presplit_dataset(
        output_dir=output_dir,
        train=train,
        val=val,
        test=test,
    )
    rows_total = split_metadata["rows_total"]
    rows_used = rows_total if isinstance(rows_total, int) else int(str(rows_total))
    metadata = {
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "sources": [
            {
                "kind": "dvc_local_presplit",
                "source_name": source_name,
                "declared_license": declared_license,
                "train_path": str(train_path),
                "val_path": str(val_path),
                "test_path": str(test_path),
                "sample_rows": sample_rows,
                "rows_used": rows_used,
                "sampled_rows_by_split": sampled_rows_by_split,
            }
        ],
        "rows_before_cross_source_deduplication": int(len(combined)),
        "rows_with_conflicting_labels_removed": 0,
        "duplicate_text_rows_removed": 0,
        **split_metadata,
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def remove_cross_source_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    conflict_mask = frame.groupby("text")["label"].transform("nunique") > 1
    rows_with_conflicting_labels = int(conflict_mask.sum())
    without_conflicts = frame.loc[~conflict_mask].copy()

    before_dedup = len(without_conflicts)
    deduped = without_conflicts.drop_duplicates(subset=["text"], keep="first").reset_index(
        drop=True
    )
    return deduped, {
        "rows_with_conflicting_labels_removed": rows_with_conflicting_labels,
        "duplicate_text_rows_removed": int(before_dedup - len(deduped)),
    }


def _select_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Restrict a frame to the standardized output schema.

    Drops auxiliary columns (HF's polluted `source`, local CSV's `prompt_name`,
    `source_split`, etc.) so every saved split has the same columns regardless
    of which sources fed it. Missing columns are filled with empty strings.
    """
    columns = list(STANDARDIZED_OUTPUT_COLUMNS)
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = ""
    return out[columns]


def write_dataset_splits(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, object]:
    if frame["label"].nunique() < 2:
        raise ValueError("Dataset must contain both human and generated examples")

    train, val, test = stratified_train_val_test_split(
        frame,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / Split.ALL.filename
    train_path = output_dir / Split.TRAIN.filename
    val_path = output_dir / Split.VAL.filename
    test_path = output_dir / Split.TEST.filename
    _select_output_columns(frame).to_csv(all_path, index=False)
    _select_output_columns(train).to_csv(train_path, index=False)
    _select_output_columns(val).to_csv(val_path, index=False)
    _select_output_columns(test).to_csv(test_path, index=False)

    return {
        "rows_total": int(len(frame)),
        "rows_train": int(len(train)),
        "rows_val": int(len(val)),
        "rows_test": int(len(test)),
        "label_counts": _label_counts(frame),
        "train_label_counts": _label_counts(train),
        "val_label_counts": _label_counts(val),
        "test_label_counts": _label_counts(test),
        "all_path": str(all_path),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
    }


def build_training_dataset(
    *,
    output_dir: Path,
    local_paths: list[Path],
    local_source_name: str | None,
    local_license: str,
    local_sample_rows: int | None,
    local_pre_split: bool = False,
    local_train_path: str | Path | None = None,
    local_val_path: str | Path | None = None,
    local_test_path: str | Path | None = None,
    hf_dataset: str,
    hf_split: str,
    hf_sample_rows: int,
    hf_license: str,
    hf_shuffle_buffer: int,
    skip_hf: bool,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    hf_pre_split: bool = DEFAULT_HF_PRE_SPLIT,
    hf_train_split: str = DEFAULT_HF_TRAIN_SPLIT,
    hf_val_split: str = DEFAULT_HF_VAL_SPLIT,
    hf_test_split: str = DEFAULT_HF_TEST_SPLIT,
) -> dict[str, object]:
    if local_pre_split:
        if not local_train_path or not local_val_path or not local_test_path:
            raise ValueError(
                "local_pre_split requires local_train_path, local_val_path, and local_test_path"
            )
        sample_rows = local_sample_rows if local_sample_rows is not None else hf_sample_rows
        return build_local_split_dataset(
            output_dir=output_dir,
            train_path=Path(local_train_path),
            val_path=Path(local_val_path),
            test_path=Path(local_test_path),
            source_name=local_source_name or hf_dataset,
            declared_license=local_license,
            sample_rows=sample_rows,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )

    if hf_pre_split and not skip_hf and not local_paths:
        return build_hf_split_dataset(
            output_dir=output_dir,
            dataset_name=hf_dataset,
            declared_license=hf_license,
            train_split=hf_train_split,
            val_split=hf_val_split,
            test_split=hf_test_split,
            sample_rows=hf_sample_rows,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )

    frames: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []

    if not skip_hf:
        hf_frame = collect_hf_balanced_sample(
            dataset_name=hf_dataset,
            split=hf_split,
            sample_rows=hf_sample_rows,
            seed=seed,
            shuffle_buffer=hf_shuffle_buffer,
        )
        frames.append(hf_frame)
        sources.append(
            {
                "kind": "huggingface_streaming",
                "dataset": hf_dataset,
                "split": hf_split,
                "declared_license": hf_license,
                "rows_used": int(len(hf_frame)),
                "label_counts": _label_counts(hf_frame),
            }
        )

    for raw_path in local_paths:
        source_name = local_source_name or raw_path.name or str(raw_path)
        local_frame, local_metadata = load_local_source(
            raw_path=raw_path,
            source_name=source_name,
            declared_license=local_license,
            sample_rows=local_sample_rows,
            seed=seed,
        )
        frames.append(local_frame)
        sources.append(local_metadata)

    if not frames:
        raise ValueError("At least one Hugging Face or local source must be enabled")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    deduped, dedupe_metadata = remove_cross_source_duplicates(combined)
    split_metadata = write_dataset_splits(
        frame=deduped,
        output_dir=output_dir,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    metadata = {
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "sources": sources,
        "rows_before_cross_source_deduplication": int(len(combined)),
        **dedupe_metadata,
        **split_metadata,
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def main(
    output_dir: str | Path = CommonPaths().processed_dir,
    local_path: str | Path | list[str | Path] | None = None,
    local_sample_rows: int | None = None,
    local_pre_split: bool = False,
    local_train_path: str | Path | None = None,
    local_val_path: str | Path | None = None,
    local_test_path: str | Path | None = None,
    local_source_name: str | None = None,
    local_license: str = "unverified",
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_split: str = DEFAULT_HF_SPLIT,
    hf_sample_rows: int = DEFAULT_HF_SAMPLE_ROWS,
    hf_license: str = DEFAULT_HF_LICENSE,
    hf_shuffle_buffer: int = DEFAULT_HF_SHUFFLE_BUFFER,
    hf_pre_split: bool = DEFAULT_HF_PRE_SPLIT,
    hf_train_split: str = DEFAULT_HF_TRAIN_SPLIT,
    hf_val_split: str = DEFAULT_HF_VAL_SPLIT,
    hf_test_split: str = DEFAULT_HF_TEST_SPLIT,
    skip_hf: bool = False,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = TrainingDefaults().seed,
) -> None:
    if local_path is None:
        local_paths: list[Path] = []
    elif isinstance(local_path, str | Path):
        local_paths = [Path(local_path)]
    else:
        local_paths = [Path(path) for path in local_path]
    metadata = build_training_dataset(
        output_dir=Path(output_dir),
        local_paths=local_paths,
        local_source_name=local_source_name,
        local_license=local_license,
        local_sample_rows=local_sample_rows,
        local_pre_split=local_pre_split,
        local_train_path=local_train_path,
        local_val_path=local_val_path,
        local_test_path=local_test_path,
        hf_dataset=hf_dataset,
        hf_split=hf_split,
        hf_sample_rows=hf_sample_rows,
        hf_license=hf_license,
        hf_shuffle_buffer=hf_shuffle_buffer,
        hf_pre_split=hf_pre_split,
        hf_train_split=hf_train_split,
        hf_val_split=hf_val_split,
        hf_test_split=hf_test_split,
        skip_hf=skip_hf,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    print("Built training dataset:")
    for key, value in {**dataclass_to_jsonable(CommonPaths()), **metadata}.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
