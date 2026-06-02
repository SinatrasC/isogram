from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from isogram import commands
from isogram.config import PROJECT_ROOT


def _yaml_texts(config_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(config_dir)): path.read_text(encoding="utf-8")
        for path in sorted(config_dir.rglob("*.yaml"))
    }


def test_packaged_configs_match_root_configs() -> None:
    root_config_dir = PROJECT_ROOT / "configs"
    package_config_dir = Path(str(files("isogram").joinpath("configs")))

    assert _yaml_texts(package_config_dir) == _yaml_texts(root_config_dir)


def test_compose_config_falls_back_to_packaged_configs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(commands, "ROOT_CONFIG_DIR", tmp_path / "missing-configs")

    cfg = commands._compose_config(["logging.enabled=false"])

    assert cfg.data.name == "main"
    assert cfg.model.name == "char_cnn"
