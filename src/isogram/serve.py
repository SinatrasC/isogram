from __future__ import annotations

import argparse
import os
from importlib.resources import files
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from isogram.inference import Predictor


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class PredictResponse(BaseModel):
    prob_ai: float
    model_version: str


def load_index_html() -> str:
    return files("isogram").joinpath("static/index.html").read_text(encoding="utf-8")


def create_app(*, checkpoint: Path | None = None, device: str = "auto") -> FastAPI:
    checkpoint_path = checkpoint or os.getenv("ISOGRAM_CHECKPOINT")
    if checkpoint_path is None:
        raise RuntimeError("A checkpoint path is required via --checkpoint or ISOGRAM_CHECKPOINT")
    predictor = Predictor(Path(checkpoint_path), device=os.getenv("ISOGRAM_DEVICE", device))

    app = FastAPI(title="Isogram AI Text Detection", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return load_index_html()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": predictor.model_version}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        try:
            probability = predictor.predict(request.text)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="prediction failed") from exc
        return PredictResponse(prob_ai=probability, model_version=predictor.model_version)

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Isogram predictions with FastAPI.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    app = create_app(checkpoint=args.checkpoint, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
