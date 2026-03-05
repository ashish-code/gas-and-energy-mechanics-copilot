# run `just dev`
default: dev

# update the uv project's environment
deps:
  uv sync

# format code with `ruff`
fmt:
  uv run --extra dev ruff format

# start uvicorn server with hot-reload
dev:
  just deps
  CONFIG_DIR="./config/" uv run --dev \
    uvicorn brightai.ai_copilot.entrypoint:app \
    --port 8080 \
    --reload \
    --reload-include './config/*.toml' \
    --reload-exclude './scripts/*.py' \
    --log-config ./config/uvicorn-logging-config.json

# starts streamlit chat application for interacting with Strands agent via A2AProtocol
chat:
  uv run --extra dev streamlit run scripts/streamlit.py

# start application on local KinD cluster using Tilt
dev-k8s:
  kind create cluster --config=kind.yaml
  tilt up

test TARGET:
    CONFIG_DIR="./config/" uv run --extra test pytest -v -k {{ TARGET }}

test-all:
    CONFIG_DIR="./config/" uv run --extra test pytest -v --cov

tag := `uvx uv-dynamic-versioning | sed 's/+/_/'`

# build container image
build-docker TAG=tag:
  just deps
  docker build \
    --secret "id=AWS_SHARED_CREDENTIALS_FILE,src=$HOME/.aws/credentials" \
    --secret "id=KEYRING_CFG,src=$HOME/.config/python_keyring/keyringrc.cfg"  \
    -t 312851193143.dkr.ecr.us-west-2.amazonaws.com/ai-copilot:{{ TAG }} .

# build and run container image
run-docker TAG=tag:
  just deps
  just build-docker {{ TAG }}
  docker run -i -t -p 8080:8080 --rm 312851193143.dkr.ecr.us-west-2.amazonaws.com/ai-copilot:{{ TAG }}

# build and attach shell to container image
shell-docker TAG=tag:
  just deps
  just build-docker {{ TAG }}
  docker run \
    -i -t \
    --entrypoint "bash" \
    --rm \
    312851193143.dkr.ecr.us-west-2.amazonaws.com/ai-copilot:{{ TAG }}
