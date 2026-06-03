from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from isogram.config import PROJECT_ROOT


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def dvc_executable() -> str | None:
    candidate = Path(sys.executable).with_name("dvc")
    if candidate.exists():
        return str(candidate)
    return shutil.which("dvc")


def dvc_is_enabled(dvc_cfg: Any) -> bool:
    return bool(cfg_get(dvc_cfg, "enabled", True))


def _dvc_remote(dvc_cfg: Any, key: str, default: str) -> str:
    return str(cfg_get(dvc_cfg, key, default))


def _run_dvc(args: list[str]) -> bool:
    executable = dvc_executable()
    if executable is None:
        return False
    result = subprocess.run([executable, *args], cwd=PROJECT_ROOT, check=False)
    return result.returncode == 0


def pull_dvc_targets(
    targets: list[Path],
    *,
    remote: str,
    allow_missing: bool = False,
) -> bool:
    target_args = [str(target) for target in targets]
    if not target_args:
        return True
    command = ["pull", "-r", remote, *target_args]
    if allow_missing:
        command.append("--allow-missing")
    return _run_dvc(command)


def dvc_pointer_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.dvc")


def pull_data_artifacts(dvc_cfg: Any, *, targets: list[Path]) -> bool:
    if not dvc_is_enabled(dvc_cfg) or not bool(cfg_get(dvc_cfg, "pull_data", True)):
        return False
    remote = _dvc_remote(dvc_cfg, "data_remote", "data")
    return pull_dvc_targets(targets, remote=remote, allow_missing=True)


def restore_dvc_imports(paths: list[Path], dvc_cfg: Any) -> bool:
    if not dvc_is_enabled(dvc_cfg) or not bool(cfg_get(dvc_cfg, "pull_imports", True)):
        return False

    remote = _dvc_remote(dvc_cfg, "data_remote", "data")
    restored = True
    for path in paths:
        if path.exists():
            continue

        pointer_path = dvc_pointer_path(path)
        if not pointer_path.exists():
            restored = False
            continue

        pull_dvc_targets([pointer_path, path], remote=remote, allow_missing=True)
        if not path.exists():
            _run_dvc(["update", str(pointer_path)])
        restored = restored and path.exists()

    return restored


def pull_checkpoint_artifact(checkpoint_path: Path, dvc_cfg: Any) -> bool:
    if checkpoint_path.exists():
        return True
    if not dvc_is_enabled(dvc_cfg) or not bool(cfg_get(dvc_cfg, "pull_models", True)):
        return False
    remote = _dvc_remote(dvc_cfg, "model_remote", "models")
    pointer_path = dvc_pointer_path(checkpoint_path)
    targets = [checkpoint_path]
    if pointer_path.exists():
        targets.insert(0, pointer_path)
    pull_dvc_targets(targets, remote=remote, allow_missing=True)
    return checkpoint_path.exists()


def add_and_push_checkpoint(checkpoint_path: Path, dvc_cfg: Any) -> bool:
    if not checkpoint_path.exists():
        return False
    if not dvc_is_enabled(dvc_cfg) or not bool(cfg_get(dvc_cfg, "push_models", True)):
        return False
    remote = _dvc_remote(dvc_cfg, "model_remote", "models")
    added = _run_dvc(["add", str(checkpoint_path)])
    if not added:
        return False
    return _run_dvc(["push", "-r", remote, str(dvc_pointer_path(checkpoint_path))])
