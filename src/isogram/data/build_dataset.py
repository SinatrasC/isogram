from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

from isogram.config import CommonPaths, TrainingDefaults, dataclass_to_jsonable, write_json
from isogram.data.schema import (
    LABEL_COLUMNS,
    TEXT_COLUMNS,
    find_csv,
    normalize_frame,
    normalize_label,
    sample_balanced_by_label,
    stratified_train_val_test_split,
)
from isogram.data.splits import Split


DEFAULT_HF_DATASET = "srikanthgali/ai-text-detection-pile-cleaned"
DEFAULT_HF_LICENSE = "mit"
DEFAULT_HF_SPLIT = "train"
DEFAULT_HF_SAMPLE_ROWS = 60_000
DEFAULT_HF_SHUFFLE_BUFFER = 10_000


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

    normalized = {
        "text": text,
        "label": normalize_label(row[label_key]),
        "source": str(row.get("source", "")),
        "source_dataset": source_dataset,
        "source_split": source_split,
    }
    if "model" in row:
        normalized["model"] = str(row.get("model", ""))
    return normalized


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
    normalized = normalize_frame(frame)
    normalized["source_dataset"] = source_name
    normalized["source_split"] = "local"
    sampled = sample_balanced_by_label(normalized, max_rows=sample_rows, seed=seed)
    metadata = {
        "kind": "local_csv",
        "source_name": source_name,
        "declared_license": declared_license,
        "source_csv": str(csv_path),
        "rows_normalized": int(len(normalized)),
        "rows_used": int(len(sampled)),
        "label_counts": _label_counts(sampled),
    }
    return sampled, metadata


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


def write_dataset_splits(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, object]:
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
    frame.to_csv(all_path, index=False)
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

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


def build_merged_dataset(
    *,
    output_dir: Path,
    local_paths: list[Path],
    local_source_name: str | None,
    local_license: str,
    local_sample_rows: int | None,
    hf_dataset: str,
    hf_split: str,
    hf_sample_rows: int,
    hf_license: str,
    hf_shuffle_buffer: int,
    skip_hf: bool,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, object]:
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


def build_parser() -> argparse.ArgumentParser:
    paths = CommonPaths()
    defaults = TrainingDefaults()
    parser = argparse.ArgumentParser(
        description="Build merged, sampled AI-text detection train/val/test splits."
    )
    parser.add_argument("--output-dir", type=Path, default=paths.processed_dir)
    parser.add_argument(
        "--local-path",
        type=Path,
        action="append",
        default=[],
        help="Optional local CSV file or directory to merge, such as data/raw/daigt-v2.",
    )
    parser.add_argument(
        "--local-sample-rows",
        type=int,
        help="Optional balanced row cap applied to each local source.",
    )
    parser.add_argument(
        "--local-source-name",
        help="Source name to write into metadata for local CSV rows.",
    )
    parser.add_argument(
        "--local-license",
        default="unverified",
        help="Declared license to write into metadata for local CSV rows.",
    )
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-split", default=DEFAULT_HF_SPLIT)
    parser.add_argument("--hf-sample-rows", type=int, default=DEFAULT_HF_SAMPLE_ROWS)
    parser.add_argument("--hf-license", default=DEFAULT_HF_LICENSE)
    parser.add_argument("--hf-shuffle-buffer", type=int, default=DEFAULT_HF_SHUFFLE_BUFFER)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metadata = build_merged_dataset(
        output_dir=args.output_dir,
        local_paths=args.local_path,
        local_source_name=args.local_source_name,
        local_license=args.local_license,
        local_sample_rows=args.local_sample_rows,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_sample_rows=args.hf_sample_rows,
        hf_license=args.hf_license,
        hf_shuffle_buffer=args.hf_shuffle_buffer,
        skip_hf=args.skip_hf,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print("Built merged dataset:")
    for key, value in {**dataclass_to_jsonable(CommonPaths()), **metadata}.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
