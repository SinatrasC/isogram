from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 42


@dataclass(frozen=True)
class CommonPaths:
    raw_dir: Path = Path("data/raw/daigt-v2")
    processed_dir: Path = Path("data/processed")
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    report_dir: Path = Path("reports")


@dataclass(frozen=True)
class TrainingDefaults:
    seed: int = DEFAULT_SEED
    val_fraction: float = 0.2
    char_max_length: int = 2048
    deberta_max_length: int = 512
    char_batch_size: int = 64
    deberta_batch_size: int = 8
    char_learning_rate: float = 1e-3
    deberta_learning_rate: float = 2e-5
    char_epochs: int = 5
    deberta_epochs: int = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def dataclass_to_jsonable(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    return {key: str(item) if isinstance(item, Path) else item for key, item in payload.items()}


def get_git_commit(root: Path = PROJECT_ROOT) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return output.strip()
