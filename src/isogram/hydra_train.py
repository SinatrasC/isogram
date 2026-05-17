from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from isogram.config import ensure_parent
from isogram.train import main as train_main


def _append_optional(args: list[str], flag: str, value: Any) -> None:
    if value is not None:
        args.extend([flag, str(value)])


def _append_sequence(args: list[str], flag: str, values: list[int] | None) -> None:
    if values:
        args.append(flag)
        args.extend(str(value) for value in values)


def _resolved_config(cfg: DictConfig) -> DictConfig:
    return cast(DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))


def _write_run_config(cfg: DictConfig) -> Path:
    output_dir = Path(str(cfg.paths.run_config_dir))
    ensure_parent(output_dir / "placeholder")
    path = output_dir / f"{cfg.model.name}_{cfg.dataset.name}_hydra.yaml"
    OmegaConf.save(config=_resolved_config(cfg), f=path)
    return path


def _train_args_from_config(cfg: DictConfig, run_config: Path) -> list[str]:
    args = [
        "--model",
        str(cfg.model.name),
        "--train",
        str(cfg.dataset.train),
        "--val",
        str(cfg.dataset.val),
        "--output",
        str(cfg.paths.output),
        "--report-output",
        str(cfg.paths.report_output),
        "--model-version",
        str(cfg.model.model_version),
        "--device",
        str(cfg.training.device),
        "--seed",
        str(cfg.training.seed),
        "--epochs",
        str(cfg.model.epochs),
        "--batch-size",
        str(cfg.model.batch_size),
        "--learning-rate",
        str(cfg.model.learning_rate),
        "--grad-accum-steps",
        str(cfg.training.grad_accum_steps),
        "--log-every",
        str(cfg.training.log_every),
        "--dropout",
        str(cfg.model.dropout),
        "--run-config",
        str(run_config),
    ]
    if bool(cfg.model.amp):
        args.append("--amp")
    _append_optional(args, "--limit-rows", cfg.training.limit_rows)
    _append_optional(args, "--char-max-length", OmegaConf.select(cfg.model, "char_max_length"))
    _append_optional(
        args,
        "--char-embedding-dim",
        OmegaConf.select(cfg.model, "char_embedding_dim"),
    )
    _append_optional(args, "--char-channels", OmegaConf.select(cfg.model, "char_channels"))
    _append_sequence(
        args,
        "--char-kernel-sizes",
        OmegaConf.select(cfg.model, "char_kernel_sizes"),
    )
    _append_optional(
        args, "--deberta-model-name", OmegaConf.select(cfg.model, "deberta_model_name")
    )
    _append_optional(
        args, "--deberta-max-length", OmegaConf.select(cfg.model, "deberta_max_length")
    )
    if bool(cfg.mlflow.enabled):
        args.append("--mlflow")
        args.extend(
            [
                "--mlflow-tracking-uri",
                str(cfg.mlflow.tracking_uri),
                "--mlflow-experiment",
                str(cfg.mlflow.experiment),
                "--mlflow-run-name",
                str(cfg.mlflow.run_name),
            ]
        )
    return args


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_config = _write_run_config(cfg)
    train_main(_train_args_from_config(cfg, run_config))


if __name__ == "__main__":
    main()
