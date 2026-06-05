FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip uv \
    && uv sync --frozen --no-dev --no-editable

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

EXPOSE 8000

CMD ["python", "-m", "isogram.serve", "--host", "0.0.0.0", "--port", "8000"]
