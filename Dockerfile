ARG DOCKER_REGISTRY_CACHE

# Base Image - Shared with `builder` and `runtime`
FROM ${DOCKER_REGISTRY_CACHE}python:3.13-slim AS base

WORKDIR /app

RUN groupadd --gid 1000 app && \
  useradd --uid 1000 --gid app --shell /bin/bash --create-home app && \
  chown app:app ./

SHELL ["/bin/bash", "-c"]


FROM base AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV PATH="/root/.local/bin:$PATH"
ENV UV_LINK_MODE=copy
# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git && \
  rm -rf /var/lib/apt/lists/* && \
  uv tool install keyring --with git+https://github.com/elventear/keyrings.codeartifact.git@assume_role

COPY docker-build-aws-credentials.sh /bin/

# hadolint ignore=SC1091
RUN --mount=type=secret,id=AWS_ROLE_ARN \
  --mount=type=secret,id=AWS_WEB_IDENTITY_TOKEN_FILE \
  --mount=type=secret,id=AWS_SHARED_CREDENTIALS_FILE \
  --mount=type=secret,id=KEYRING_CFG \
  --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  --mount=type=bind,source=README.md,target=README.md \
  source docker-build-aws-credentials.sh && \
  uv sync --locked --no-install-project --no-editable

# Copy the project into the intermediate image
ADD . /app

# hadolint ignore=SC1091
RUN --mount=type=secret,id=AWS_ROLE_ARN \
  --mount=type=secret,id=AWS_WEB_IDENTITY_TOKEN_FILE \
  --mount=type=secret,id=AWS_SHARED_CREDENTIALS_FILE \
  --mount=type=secret,id=KEYRING_CFG \
  --mount=type=cache,target=/root/.cache/uv \
  source docker-build-aws-credentials.sh && \
  uv sync --locked --no-editable

FROM base AS runtime

USER app

# Copy config
COPY --from=builder --chown=app:app /app/config/production.settings.toml /app/config/settings.toml
COPY --from=builder --chown=app:app /app/config/uvicorn-logging-config.json /app/config/uvicorn-logging-config.json

# Copy RAG index data (FAISS index, chunks, metadata)
COPY --from=builder --chown=app:app /app/data/rag_index /app/data/rag_index

# Copy the environment, but not the source code
COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV CONFIG_DIR="/app/config"
EXPOSE 8080

# Run the application
CMD ["/app/.venv/bin/uvicorn", \
  "brightai.ai_copilot.entrypoint:app", \
  "--log-config", "/app/config/uvicorn-logging-config.json", \
  "--host", "0.0.0.0", \
  "--port", "8080"]
