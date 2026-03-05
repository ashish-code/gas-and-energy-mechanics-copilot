# Gas & Energy Mechanics Copilot — task runner
# Install just: brew install just

aws_profile := env_var_or_default("AWS_PROFILE", "vscode-user")
aws_region  := env_var_or_default("AWS_REGION",  "us-east-1")

default: dev

# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------

# Sync Python dependencies
deps:
  uv sync

# Format code with ruff
fmt:
  uv run --extra dev ruff format

# Start uvicorn with hot-reload (requires data/rag_index — run build-index first)
dev:
  just deps
  CONFIG_DIR="./config/" uv run \
    uvicorn brightai.ai_copilot.entrypoint:app \
    --port 8080 \
    --reload \
    --reload-include './config/*.toml' \
    --reload-exclude './scripts/*.py' \
    --log-config ./config/uvicorn-logging-config.json

# Start the Streamlit chat UI (connects to local uvicorn server)
chat:
  uv run --extra dev streamlit run scripts/streamlit.py

# Run tests
test TARGET:
    CONFIG_DIR="./config/" uv run --extra test pytest -v -k {{ TARGET }}

test-all:
    CONFIG_DIR="./config/" uv run --extra test pytest -v --cov

# Build the FAISS RAG index from Wikipedia (output: data/rag_index/)
build-index:
  AWS_PROFILE={{ aws_profile }} uv run python scripts/build_index.py

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

# Build linux/amd64 image and push to ECR (required before first `just deploy`)
push TAG="latest":
  AWS_PROFILE={{ aws_profile }} AWS_REGION={{ aws_region }} \
    bash scripts/build-and-push.sh {{ TAG }}

# Build and run locally for smoke-testing (native arch, not for App Runner)
run-local:
  docker build -t gas-copilot-local .
  docker run --rm -p 8080:8080 \
    -e AWS_REGION={{ aws_region }} \
    -e CONFIG_DIR=/app/config \
    -v ~/.aws:/root/.aws:ro \
    gas-copilot-local

# ---------------------------------------------------------------------------
# Terraform — infrastructure lifecycle
# ---------------------------------------------------------------------------

# Initialise Terraform (run once after cloning)
tf-init:
  terraform -chdir=terraform init

# Show what Terraform would change without applying
tf-plan:
  terraform -chdir=terraform plan

# Stand up the full deployment (ECR + IAM + App Runner)
# Run `just push` first if the ECR image doesn't exist yet.
deploy:
  terraform -chdir=terraform apply -auto-approve
  @echo ""
  @echo "==> Live at:"
  @terraform -chdir=terraform output -raw service_url

# Tear down ONLY the App Runner service to stop compute costs.
# ECR image and IAM roles are retained — run `just deploy` to bring it back.
teardown:
  terraform -chdir=terraform destroy \
    -target=aws_apprunner_service.app \
    -auto-approve
  @echo ""
  @echo "==> App Runner destroyed. Run 'just deploy' to bring it back up."

# Tear down EVERYTHING (ECR + IAM + App Runner). Requires `just push` before next deploy.
destroy-all:
  terraform -chdir=terraform destroy -auto-approve

# Import existing manually-created AWS resources into Terraform state.
# Run ONCE to hand over management of the already-running service to Terraform.
tf-import:
  @echo "==> Importing ECR repository..."
  terraform -chdir=terraform import aws_ecr_repository.app gas-and-energy-mechanics-copilot
  @echo "==> Importing IAM task role..."
  terraform -chdir=terraform import aws_iam_role.task_role gas-energy-copilot-apprunner-task-role
  @echo "==> Importing IAM inline policy..."
  terraform -chdir=terraform import \
    aws_iam_role_policy.task_policy \
    "gas-energy-copilot-apprunner-task-role:gas-energy-copilot-bedrock-policy"
  @echo "==> Importing App Runner service..."
  terraform -chdir=terraform import \
    aws_apprunner_service.app \
    arn:aws:apprunner:us-east-1:414994224379:service/gas-and-energy-mechanics-copilot/929272659b70473f99944f98b9bd0835
  @echo ""
  @echo "==> Import complete. Run 'just tf-plan' to verify no drift."
