from __future__ import annotations

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


def run(
    url: str = "http://127.0.0.1:8000/predict",
    requests: int = 50,
    timeout: float = 30.0,
    text: str = DEFAULT_TEXT,
) -> None:
    latencies: list[float] = []
    with httpx.Client(timeout=timeout) as client:
        for _ in range(requests):
            started = time.perf_counter()
            response = client.post(url, json={"text": text})
            response.raise_for_status()
            latencies.append(time.perf_counter() - started)

    print(f"requests: {len(latencies)}")
    print(f"median_seconds: {statistics.median(latencies):.4f}")
    print(f"p95_seconds: {percentile(latencies, 0.95):.4f}")


def main() -> None:
    import fire

    fire.Fire(run)


if __name__ == "__main__":
    main()
