from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from isogram.config import CommonPaths, TrainingDefaults, dataclass_to_jsonable, write_json
from isogram.data.schema import find_csv, normalize_frame, stratified_train_val_split
from isogram.data.splits import Split


def download_with_kaggle(dataset: str, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    kaggle_executable = Path(sys.executable).with_name("kaggle")
    if not kaggle_executable.exists():
        discovered = shutil.which("kaggle")
        if discovered is None:
            raise RuntimeError(
                "Kaggle CLI is not available. Install with `python -m pip install -e .[kaggle]`."
            )
        kaggle_executable = Path(discovered)
    command = [
        os.fspath(kaggle_executable),
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(raw_dir),
        "--unzip",
    ]
    subprocess.run(command, check=True)


def prepare_dataset(
    *,
    raw_path: Path,
    output_dir: Path,
    val_fraction: float,
    seed: int,
) -> dict[str, object]:
    csv_path = find_csv(raw_path)
    frame = pd.read_csv(csv_path)
    normalized = normalize_frame(frame)
    train, val = stratified_train_val_split(normalized, val_fraction=val_fraction, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / Split.TRAIN.filename
    val_path = output_dir / Split.VAL.filename
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)

    metadata = {
        "source_csv": str(csv_path),
        "rows_total": int(len(normalized)),
        "rows_train": int(len(train)),
        "rows_val": int(len(val)),
        "label_counts": {
            str(label): int(count) for label, count in normalized["label"].value_counts().items()
        },
        "seed": seed,
        "val_fraction": val_fraction,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    paths = CommonPaths()
    defaults = TrainingDefaults()
    parser = argparse.ArgumentParser(description="Prepare DAIGT v2 train/validation splits.")
    parser.add_argument("--raw-path", type=Path, default=paths.raw_dir)
    parser.add_argument("--output-dir", type=Path, default=paths.processed_dir)
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Use the Kaggle CLI to download DAIGT v2 before preparing splits.",
    )
    parser.add_argument(
        "--kaggle-dataset",
        default="thedrcat/daigt-v2-train-dataset",
        help="Kaggle dataset slug used with --download.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.download:
        download_with_kaggle(args.kaggle_dataset, args.raw_path)
    metadata = prepare_dataset(
        raw_path=args.raw_path,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print("Prepared dataset:")
    for key, value in {**dataclass_to_jsonable(CommonPaths()), **metadata}.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
