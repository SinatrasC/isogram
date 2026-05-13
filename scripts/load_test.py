from __future__ import annotations

import argparse
import statistics
import time

import httpx


DEFAULT_TEXT = (
    "Careful source evaluation matters because reliable arguments depend on "
    "evidence, context, and transparent reasoning."
)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure Isogram API response latency.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    latencies: list[float] = []
    with httpx.Client(timeout=args.timeout) as client:
        for _ in range(args.requests):
            started = time.perf_counter()
            response = client.post(args.url, json={"text": args.text})
            response.raise_for_status()
            latencies.append(time.perf_counter() - started)

    print(f"requests: {len(latencies)}")
    print(f"median_seconds: {statistics.median(latencies):.4f}")
    print(f"p95_seconds: {percentile(latencies, 0.95):.4f}")


if __name__ == "__main__":
    main()
