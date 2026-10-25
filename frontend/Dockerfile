# syntax=docker/dockerfile:1

# --- Build stage: install dependencies with uv into /app/.venv -----------------------------
FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY app.py ./


# --- Runtime stage ---------------------------------------------------------------------------
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER app
EXPOSE 8501

# Cloud Run sets $PORT; 8501 otherwise. API_BASE_URL and API_KEY come from the environment.
CMD ["sh", "-c", "exec streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true"]
