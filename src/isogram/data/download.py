from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from isogram.data.build_dataset import (
    DEFAULT_HF_DATASET,
    DEFAULT_HF_LICENSE,
    DEFAULT_HF_PRE_SPLIT,
    DEFAULT_HF_SAMPLE_ROWS,
    DEFAULT_HF_SHUFFLE_BUFFER,
    DEFAULT_HF_SPLIT,
    DEFAULT_HF_TEST_SPLIT,
    DEFAULT_HF_TRAIN_SPLIT,
    DEFAULT_HF_VAL_SPLIT,
    build_training_dataset,
)


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def _path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, str | Path):
        return [Path(value)]
    return [Path(str(item)) for item in value]


def _configured_local_paths(data_cfg: Any) -> list[Path]:
    paths = _path_list(cfg_get(data_cfg, "local_paths", None))
    singular = cfg_get(data_cfg, "local_path", None)
    if singular is not None:
        paths.extend(_path_list(singular))
    return paths


def split_paths(data_cfg: Any) -> list[Path]:
    output_dir = Path(str(cfg_get(data_cfg, "output_dir", "data/processed/main")))
    return [
        Path(str(cfg_get(data_cfg, "all_path", output_dir / "all.csv"))),
        Path(str(cfg_get(data_cfg, "train_path", output_dir / "train.csv"))),
        Path(str(cfg_get(data_cfg, "val_path", output_dir / "val.csv"))),
        Path(str(cfg_get(data_cfg, "test_path", output_dir / "test.csv"))),
    ]


def dataset_ready(data_cfg: Any) -> bool:
    return all(path.exists() for path in split_paths(data_cfg))


def _metadata_path(data_cfg: Any) -> Path:
    output_dir = Path(str(cfg_get(data_cfg, "output_dir", "data/processed/main")))
    return Path(str(cfg_get(data_cfg, "metadata_path", output_dir / "metadata.json")))


def _read_metadata(data_cfg: Any) -> dict[str, Any]:
    path = _metadata_path(data_cfg)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return dict(json.load(handle))


def _dvc_executable() -> str | None:
    candidate = Path(sys.executable).with_name("dvc")
    if candidate.exists():
        return str(candidate)
    return shutil.which("dvc")


def pull_data_with_dvc(data_cfg: Any) -> bool:
    executable = _dvc_executable()
    if executable is None:
        return False
    command = [executable, "pull", *[str(path) for path in split_paths(data_cfg)]]
    result = subprocess.run(command, check=False)
    return result.returncode == 0


def download_data(data_cfg: Any) -> dict[str, Any]:
    local_paths = [path for path in _configured_local_paths(data_cfg) if path.exists()]
    if not local_paths:
        print("No configured local source exists; building from the public Hugging Face source.")

    metadata = build_training_dataset(
        output_dir=Path(str(cfg_get(data_cfg, "output_dir", "data/processed/main"))),
        local_paths=local_paths,
        local_source_name=cfg_get(data_cfg, "local_source_name", None),
        local_license=str(cfg_get(data_cfg, "local_license", "unverified")),
        local_sample_rows=cfg_get(data_cfg, "local_sample_rows", None),
        hf_dataset=str(cfg_get(data_cfg, "hf_dataset", DEFAULT_HF_DATASET)),
        hf_split=str(cfg_get(data_cfg, "hf_split", DEFAULT_HF_SPLIT)),
        hf_sample_rows=int(cfg_get(data_cfg, "hf_sample_rows", DEFAULT_HF_SAMPLE_ROWS)),
        hf_license=str(cfg_get(data_cfg, "hf_license", DEFAULT_HF_LICENSE)),
        hf_shuffle_buffer=int(cfg_get(data_cfg, "hf_shuffle_buffer", DEFAULT_HF_SHUFFLE_BUFFER)),
        hf_pre_split=bool(cfg_get(data_cfg, "hf_pre_split", DEFAULT_HF_PRE_SPLIT)),
        hf_train_split=str(cfg_get(data_cfg, "hf_train_split", DEFAULT_HF_TRAIN_SPLIT)),
        hf_val_split=str(cfg_get(data_cfg, "hf_val_split", DEFAULT_HF_VAL_SPLIT)),
        hf_test_split=str(cfg_get(data_cfg, "hf_test_split", DEFAULT_HF_TEST_SPLIT)),
        skip_hf=bool(cfg_get(data_cfg, "skip_hf", False)),
        val_fraction=float(cfg_get(data_cfg, "val_fraction", 0.1)),
        test_fraction=float(cfg_get(data_cfg, "test_fraction", 0.1)),
        seed=int(cfg_get(data_cfg, "seed", 42)),
    )
    return metadata


def ensure_dataset(cfg: Any, *, rebuild: bool = False) -> dict[str, Any]:
    data_cfg = cfg_get(cfg, "data", cfg)
    if not rebuild and dataset_ready(data_cfg):
        return _read_metadata(data_cfg)

    dvc_cfg = cfg_get(cfg, "dvc", {})
    use_dvc = bool(cfg_get(dvc_cfg, "enabled", cfg_get(data_cfg, "use_dvc", True)))
    if not rebuild and use_dvc:
        pull_data_with_dvc(data_cfg)
        if dataset_ready(data_cfg):
            return _read_metadata(data_cfg)

    return download_data(data_cfg)
