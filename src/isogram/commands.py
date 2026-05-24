from __future__ import annotations

from pathlib import Path
from typing import Any

from isogram.config import PROJECT_ROOT
from isogram.data.download import ensure_dataset
from isogram.evaluate import evaluate_checkpoint
from isogram.serve import serve as serve_app
from isogram.train import cfg_get, train_from_config


CONFIG_DIR = PROJECT_ROOT / "configs"


def _compose_config(overrides: list[str] | None = None, *, config_name: str = "config") -> Any:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    OmegaConf.resolve(cfg)
    return cfg


def _clean_overrides(overrides: tuple[str, ...]) -> list[str]:
    return [str(override) for override in overrides if str(override).strip()]


def _without_model_override(overrides: list[str]) -> list[str]:
    return [override for override in overrides if not override.startswith("model=")]


class Commands:
    def data(self, *overrides: str, rebuild: bool = False) -> dict[str, Any]:
        cfg = _compose_config(_clean_overrides(overrides))
        metadata = ensure_dataset(cfg, rebuild=rebuild)
        print(f"Dataset ready under {cfg.data.output_dir}")
        return metadata

    def train(self, *overrides: str) -> dict[str, Any]:
        cfg = _compose_config(_clean_overrides(overrides))
        if bool(cfg_get(cfg.data, "ensure", True)):
            ensure_dataset(cfg)
        return train_from_config(cfg)

    def train_all(self, *overrides: str) -> list[dict[str, Any]]:
        base_overrides = _without_model_override(_clean_overrides(overrides))
        reports: list[dict[str, Any]] = []
        for model_name in ("char_cnn", "deberta"):
            cfg = _compose_config([f"model={model_name}", *base_overrides])
            if bool(cfg_get(cfg.data, "ensure", True)):
                ensure_dataset(cfg)
            reports.append(train_from_config(cfg))
        return reports

    def evaluate(self, *overrides: str) -> dict[str, Any]:
        cfg = _compose_config(_clean_overrides(overrides))
        if bool(cfg_get(cfg.data, "ensure", True)):
            ensure_dataset(cfg)
        output_dir = Path(str(cfg.data.output_dir))
        data_path = Path(str(cfg_get(cfg.data, "test_path", output_dir / "test.csv")))
        checkpoint_path = Path(str(cfg.paths.checkpoint))
        return evaluate_checkpoint(
            checkpoint=checkpoint_path,
            data=data_path,
            output=Path(str(cfg.paths.evaluation_report)),
            device=str(cfg_get(cfg.trainer, "inference_device", "auto")),
            batch_size=int(
                cfg_get(cfg.trainer, "eval_batch_size", cfg_get(cfg.model, "batch_size", 32))
            ),
            mlflow=bool(cfg.logging.enabled),
            mlflow_tracking_uri=str(cfg.logging.tracking_uri),
            mlflow_experiment=str(cfg.logging.experiment),
            mlflow_run_name=f"eval-{cfg.model.name}-{cfg.data.name}",
        )

    def serve(self, *overrides: str) -> None:
        cfg = _compose_config(_clean_overrides(overrides))
        serve_app(
            checkpoint=Path(str(cfg_get(cfg.serve, "checkpoint", cfg.paths.checkpoint))),
            host=str(cfg_get(cfg.serve, "host", "127.0.0.1")),
            port=int(cfg_get(cfg.serve, "port", 8000)),
            device=str(cfg_get(cfg.serve, "device", "auto")),
        )


def main() -> None:
    import fire

    fire.Fire(Commands)


if __name__ == "__main__":
    main()
