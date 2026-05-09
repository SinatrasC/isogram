from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from isogram.config import CommonPaths, write_json
from isogram.inference import Predictor
from isogram.metrics import compute_classification_report


def build_parser() -> argparse.ArgumentParser:
    paths = CommonPaths()
    parser = argparse.ArgumentParser(description="Evaluate an Isogram checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=paths.processed_dir / "val.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.output is None:
        args.output = CommonPaths().report_dir / f"{args.checkpoint.stem}_evaluation.json"

    frame = pd.read_csv(args.data)
    texts = frame["text"].astype(str).tolist()
    labels = frame["label"].astype(int).tolist()
    predictor = Predictor(args.checkpoint, device=args.device)

    started = time.perf_counter()
    scores = predictor.predict_batch(texts, batch_size=args.batch_size)
    elapsed = time.perf_counter() - started
    threshold = args.threshold if args.threshold is not None else predictor.threshold
    report = compute_classification_report(labels, scores, threshold=threshold)
    report.update(
        {
            "checkpoint": str(args.checkpoint),
            "data": str(args.data),
            "model_version": predictor.model_version,
            "rows": len(texts),
            "batch_size": args.batch_size,
            "elapsed_seconds": elapsed,
            "examples_per_second": len(texts) / elapsed if elapsed > 0 else None,
        }
    )
    write_json(args.output, report)
    print(f"Saved evaluation report to {args.output}")
    print(report)


if __name__ == "__main__":
    main()
