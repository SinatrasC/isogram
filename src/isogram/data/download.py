from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from isogram.config import DataDefaults
from isogram.dvc import pull_data_artifacts, restore_dvc_imports
from isogram.data.build_dataset import build_training_dataset


DATA_DEFAULTS = DataDefaults()


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
    paths.extend(_configured_presplit_paths(data_cfg))
    return paths


def _configured_presplit_paths(data_cfg: Any) -> list[Path]:
    if bool(cfg_get(data_cfg, "local_pre_split", False)):
        paths: list[Path] = []
        for key in ("local_train_path", "local_val_path", "local_test_path"):
            value = cfg_get(data_cfg, key, None)
            if value is not None:
                paths.append(Path(str(value)))
        return paths
    return []


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


def pull_data_with_dvc(data_cfg: Any, dvc_cfg: Any) -> bool:
    return pull_data_artifacts(dvc_cfg, targets=[*split_paths(data_cfg), _metadata_path(data_cfg)])


def download_data(data_cfg: Any, dvc_cfg: Any | None = None) -> dict[str, Any]:
    configured_local_paths = _configured_local_paths(data_cfg)
    if dvc_cfg is not None and configured_local_paths:
        restore_dvc_imports(configured_local_paths, dvc_cfg)

    if bool(cfg_get(data_cfg, "local_pre_split", False)):
        missing = [path for path in _configured_presplit_paths(data_cfg) if not path.exists()]
        if missing:
            missing_paths = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                "Configured local pre-split data files are missing after DVC restore: "
                f"{missing_paths}"
            )

    local_paths = [path for path in configured_local_paths if path.exists()]
    if not local_paths:
        print("No configured local source exists; building from the public Hugging Face source.")

    metadata = build_training_dataset(
        output_dir=Path(str(cfg_get(data_cfg, "output_dir", "data/processed/main"))),
        local_paths=local_paths,
        local_source_name=cfg_get(data_cfg, "local_source_name", None),
        local_license=str(cfg_get(data_cfg, "local_license", "unverified")),
        local_sample_rows=cfg_get(data_cfg, "local_sample_rows", None),
        local_pre_split=bool(cfg_get(data_cfg, "local_pre_split", False)),
        local_train_path=cfg_get(data_cfg, "local_train_path", None),
        local_val_path=cfg_get(data_cfg, "local_val_path", None),
        local_test_path=cfg_get(data_cfg, "local_test_path", None),
        hf_dataset=str(cfg_get(data_cfg, "hf_dataset", DATA_DEFAULTS.hf_dataset)),
        hf_split=str(cfg_get(data_cfg, "hf_split", DATA_DEFAULTS.hf_split)),
        hf_sample_rows=int(cfg_get(data_cfg, "hf_sample_rows", DATA_DEFAULTS.hf_sample_rows)),
        hf_license=str(cfg_get(data_cfg, "hf_license", DATA_DEFAULTS.hf_license)),
        hf_shuffle_buffer=int(
            cfg_get(data_cfg, "hf_shuffle_buffer", DATA_DEFAULTS.hf_shuffle_buffer)
        ),
        hf_pre_split=bool(cfg_get(data_cfg, "hf_pre_split", DATA_DEFAULTS.hf_pre_split)),
        hf_train_split=str(cfg_get(data_cfg, "hf_train_split", DATA_DEFAULTS.hf_train_split)),
        hf_val_split=str(cfg_get(data_cfg, "hf_val_split", DATA_DEFAULTS.hf_val_split)),
        hf_test_split=str(cfg_get(data_cfg, "hf_test_split", DATA_DEFAULTS.hf_test_split)),
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
        pull_data_with_dvc(data_cfg, dvc_cfg)
        if dataset_ready(data_cfg):
            return _read_metadata(data_cfg)

    return download_data(data_cfg, dvc_cfg=dvc_cfg if use_dvc else None)
