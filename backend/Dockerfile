# syntax=docker/dockerfile:1

# --- Build stage: install dependencies with uv into /app/.venv -----------------------------
FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first: this layer is reused until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# --- Runtime stage: just Python, the virtualenv and the code -------------------------------
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYDANTIC_AI_NO_BANNER=1 \
    # Cloud Run's filesystem is read-only except /tmp (the ingest job downloads the dataset).
    HF_HOME=/tmp/huggingface

USER app
EXPOSE 8080

# Cloud Run sets $PORT; 8080 otherwise. The same image also runs the jobs:
#   alembic upgrade head · python -m scripts.ingest · python -m scripts.enrich
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
