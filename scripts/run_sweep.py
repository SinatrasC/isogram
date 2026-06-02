from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("reports/sweeps/full_sweep_summary.json")


def _run(command: list[str]) -> dict[str, Any]:
    started = time.time()
    result = subprocess.run(command, check=False, text=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "duration_seconds": round(time.time() - started, 3),
    }


def _trial_overrides(*, trial_name: str, model_name: str, overrides: Iterable[str]) -> list[str]:
    return [
        f"model={model_name}",
        f"logging.run_name={trial_name}",
        f"paths.checkpoint=artifacts/checkpoints/{trial_name}.pt",
        f"paths.training_report=reports/{trial_name}_training.json",
        f"paths.evaluation_report=reports/{trial_name}_evaluation.json",
        f"paths.run_config=reports/run_configs/{trial_name}.yaml",
        *overrides,
    ]


def _trials(mode: str) -> list[tuple[str, str, list[str]]]:
    smoke_suffix = ["trainer.limit_rows=2048", "model.max_epochs=1"] if mode == "smoke" else []
    full_trials = [
        (
            "char_cnn_lr1e-3_do0.2_ch96",
            "char_cnn",
            ["model.learning_rate=0.001", "model.dropout=0.2", "model.char_channels=96"],
        ),
        (
            "char_cnn_lr5e-4_do0.2_ch128",
            "char_cnn",
            ["model.learning_rate=0.0005", "model.dropout=0.2", "model.char_channels=128"],
        ),
        (
            "char_cnn_lr1e-3_do0.1_ch128",
            "char_cnn",
            ["model.learning_rate=0.001", "model.dropout=0.1", "model.char_channels=128"],
        ),
        (
            "deberta_lr1e-5_do0.1_bs64_ep2",
            "deberta",
            [
                "model.learning_rate=0.00001",
                "model.dropout=0.1",
                "model.batch_size=64",
                "trainer.eval_batch_size=64",
                "model.max_epochs=2",
            ],
        ),
        (
            "deberta_lr2e-5_do0.2_bs64_ep2",
            "deberta",
            [
                "model.learning_rate=0.00002",
                "model.dropout=0.2",
                "model.batch_size=64",
                "trainer.eval_batch_size=64",
                "model.max_epochs=2",
            ],
        ),
        (
            "deberta_lr3e-5_do0.2_bs64_ep2",
            "deberta",
            [
                "model.learning_rate=0.00003",
                "model.dropout=0.2",
                "model.batch_size=64",
                "trainer.eval_batch_size=64",
                "model.max_epochs=2",
            ],
        ),
    ]
    if mode == "full":
        return full_trials
    smoke_trial_names = {"char_cnn_lr1e-3_do0.2_ch96", "deberta_lr2e-5_do0.2_bs64_ep2"}
    return [
        (trial_name, model_name, [*overrides, *smoke_suffix])
        for trial_name, model_name, overrides in full_trials
        if trial_name in smoke_trial_names
    ]


def run(mode: str = "full", output: str | Path = DEFAULT_OUTPUT) -> None:
    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be 'full' or 'smoke'")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    data_command = ["uv", "run", "isogram", "data"]
    data_record = _run(data_command)
    records.append({"kind": "data", **data_record})
    if data_record["returncode"] != 0:
        output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(data_record["returncode"])

    for trial_name, model_name, overrides in _trials(mode):
        trial_overrides = _trial_overrides(
            trial_name=trial_name,
            model_name=model_name,
            overrides=overrides,
        )
        train_command = ["uv", "run", "isogram", "train", *trial_overrides]
        train_record = _run(train_command)
        records.append(
            {
                "kind": "train",
                "trial_name": trial_name,
                "model": model_name,
                **train_record,
            }
        )
        output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        if train_record["returncode"] != 0:
            raise SystemExit(train_record["returncode"])

        eval_command = ["uv", "run", "isogram", "evaluate", *trial_overrides]
        eval_record = _run(eval_command)
        records.append(
            {
                "kind": "evaluate",
                "trial_name": trial_name,
                "model": model_name,
                **eval_record,
            }
        )
        output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        if eval_record["returncode"] != 0:
            raise SystemExit(eval_record["returncode"])

    print(f"Wrote sweep summary to {output_path}")


def main() -> None:
    import fire

    fire.Fire(run)


if __name__ == "__main__":
    main()
