from __future__ import annotations

from pathlib import Path

from isogram import dvc as dvc_utils
from isogram.data.download import pull_data_with_dvc


def test_pull_data_uses_configured_data_remote(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run_dvc(args: list[str]) -> bool:
        calls.append(args)
        return True

    monkeypatch.setattr(dvc_utils, "_run_dvc", fake_run_dvc)

    ok = pull_data_with_dvc(
        {
            "output_dir": "data/processed/main",
            "metadata_path": "data/processed/main/metadata.json",
        },
        {"enabled": True, "data_remote": "data-store", "pull_data": True},
    )

    assert ok is True
    assert calls == [
        [
            "pull",
            "-r",
            "data-store",
            "data/processed/main/all.csv",
            "data/processed/main/train.csv",
            "data/processed/main/val.csv",
            "data/processed/main/test.csv",
            "data/processed/main/metadata.json",
            "--allow-missing",
        ]
    ]


def test_add_and_push_checkpoint_uses_model_remote(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls: list[list[str]] = []

    def fake_run_dvc(args: list[str]) -> bool:
        calls.append(args)
        return True

    monkeypatch.setattr(dvc_utils, "_run_dvc", fake_run_dvc)

    ok = dvc_utils.add_and_push_checkpoint(
        checkpoint,
        {"enabled": True, "model_remote": "model-store", "push_models": True},
    )

    assert ok is True
    assert calls == [
        ["add", str(checkpoint)],
        ["push", "-r", "model-store", str(checkpoint.with_name("model.pt.dvc"))],
    ]


def test_pull_checkpoint_uses_pointer_when_checkpoint_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.with_name("model.pt.dvc").write_text("outs: []\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run_dvc(args: list[str]) -> bool:
        calls.append(args)
        checkpoint.write_bytes(b"restored")
        return True

    monkeypatch.setattr(dvc_utils, "_run_dvc", fake_run_dvc)

    ok = dvc_utils.pull_checkpoint_artifact(
        checkpoint,
        {"enabled": True, "model_remote": "model-store", "pull_models": True},
    )

    assert ok is True
    assert calls == [
        [
            "pull",
            "-r",
            "model-store",
            str(checkpoint.with_name("model.pt.dvc")),
            str(checkpoint),
            "--allow-missing",
        ]
    ]


def test_restore_dvc_import_updates_from_url_when_remote_cache_misses(
    tmp_path: Path, monkeypatch
) -> None:
    imported = tmp_path / "train.parquet"
    imported.with_name("train.parquet.dvc").write_text("outs: []\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run_dvc(args: list[str]) -> bool:
        calls.append(args)
        if args[0] == "update":
            imported.write_bytes(b"from huggingface")
        return True

    monkeypatch.setattr(dvc_utils, "_run_dvc", fake_run_dvc)

    ok = dvc_utils.restore_dvc_imports(
        [imported],
        {"enabled": True, "data_remote": "data-store", "pull_imports": True},
    )

    assert ok is True
    assert calls == [
        [
            "pull",
            "-r",
            "data-store",
            str(imported.with_name("train.parquet.dvc")),
            str(imported),
            "--allow-missing",
        ],
        ["update", str(imported.with_name("train.parquet.dvc"))],
    ]
