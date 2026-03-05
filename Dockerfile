# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
  && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV PATH="/root/.local/bin:$PATH"
ENV UV_LINK_MODE=copy

# Install dependencies first (layer-cached unless pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-editable

# Copy full source and install the project
COPY src/ ./src/
RUN uv sync --locked --no-editable

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

WORKDIR /app
RUN chown app:app /app

USER app

# App configuration (production overrides dev defaults)
COPY --chown=app:app config/production.settings.toml /app/config/settings.toml
COPY --chown=app:app config/uvicorn-logging-config.json /app/config/uvicorn-logging-config.json

# FAISS RAG index (built separately via scripts/build_index.py)
COPY --chown=app:app data/rag_index /app/data/rag_index

# Python environment from builder
COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV CONFIG_DIR="/app/config"
EXPOSE 8080

CMD ["/app/.venv/bin/uvicorn", \
     "brightai.ai_copilot.entrypoint:app", \
     "--log-config", "/app/config/uvicorn-logging-config.json", \
     "--host", "0.0.0.0", \
     "--port", "8080"]
