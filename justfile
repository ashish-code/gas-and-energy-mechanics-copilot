# Gas & Energy Mechanics Copilot — task runner
# Install just: brew install just
#
# ── Typical workflows ──────────────────────────────────────────────────────────
#
#   First-time deployment (fresh repo clone):
#     just bootstrap          # init terraform → create ECR → push image → deploy
#
#   Daily dev:
#     just dev                # local uvicorn server with hot-reload
#     just chat               # Streamlit UI (connect to local server)
#
#   Teardown to pause costs (keeps ECR image, redeploy in ~3 min):
#     just teardown           # destroy only App Runner
#     just deploy             # bring it back from the existing ECR image
#
#   Ship a new version:
#     just push && just redeploy
#
#   Nuclear option (wipes everything — requires full bootstrap before next deploy):
#     just destroy-all
#
# ──────────────────────────────────────────────────────────────────────────────

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

# Start uvicorn with hot-reload (run `just build-index` first if index is missing)
dev:
    just deps
    CONFIG_DIR="./config/" uv run \
        uvicorn gas_energy_copilot.ai_copilot.entrypoint:app \
        --port 8080 \
        --reload \
        --reload-include './config/*.toml' \
        --reload-exclude './scripts/*.py' \
        --log-config ./config/uvicorn-logging-config.json

# Start the Streamlit chat UI (connects to local uvicorn server on :8080)
chat:
    uv run --extra dev streamlit run scripts/streamlit.py

# Run a specific test file or test name (e.g. just test test_api)
test TARGET:
    CONFIG_DIR="./config/" uv run --extra test pytest -v -k {{ TARGET }}

# Run the full test suite with coverage
test-all:
    CONFIG_DIR="./config/" uv run --extra test pytest -v --cov

# Build the FAISS RAG index from Wikipedia articles (output: data/rag_index/)
build-index:
    AWS_PROFILE={{ aws_profile }} uv run python scripts/build_index.py

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

# Build linux/amd64 image and push to ECR (run after `just deploy` creates ECR)
push TAG="latest":
    AWS_PROFILE={{ aws_profile }} AWS_REGION={{ aws_region }} \
        bash scripts/build-and-push.sh {{ TAG }}

# Build and run locally for smoke-testing (native arch — not for App Runner)
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

# Initialise Terraform (run once after cloning, or after updating providers)
tf-init:
    terraform -chdir=terraform init

# Show what Terraform would change without applying
tf-plan:
    AWS_PROFILE={{ aws_profile }} terraform -chdir=terraform plan \
        -var="aws_profile={{ aws_profile }}" \
        -var="aws_region={{ aws_region }}"

# ---------------------------------------------------------------------------
# Deployment lifecycle
# ---------------------------------------------------------------------------

# FIRST-TIME SETUP: initialise → create ECR → push image → deploy App Runner
# Safe to re-run: terraform is idempotent, push overwrites the :latest tag.
bootstrap:
    @echo "==> Step 1/4  Initialising Terraform..."
    just tf-init
    @echo ""
    @echo "==> Step 2/4  Creating ECR repository and IAM roles..."
    AWS_PROFILE={{ aws_profile }} terraform -chdir=terraform apply -auto-approve \
        -target=aws_ecr_repository.app \
        -target=aws_ecr_lifecycle_policy.app \
        -target=aws_iam_role.task_role \
        -target=aws_iam_role_policy.task_policy \
        -var="aws_profile={{ aws_profile }}" \
        -var="aws_region={{ aws_region }}"
    @echo ""
    @echo "==> Step 3/4  Building and pushing Docker image to ECR..."
    just push
    @echo ""
    @echo "==> Step 4/4  Deploying App Runner service..."
    AWS_PROFILE={{ aws_profile }} terraform -chdir=terraform apply -auto-approve \
        -var="aws_profile={{ aws_profile }}" \
        -var="aws_region={{ aws_region }}"
    @echo ""
    @echo "==> Bootstrap complete. Service URL:"
    @AWS_PROFILE={{ aws_profile }} terraform -chdir=terraform output -raw service_url

# Deploy (or re-deploy) the full stack — ECR + IAM + App Runner.
# Run `just push` first if no ECR image exists yet.
deploy:
    AWS_PROFILE={{ aws_profile }} terraform -chdir=terraform apply -auto-approve \
        -var="aws_profile={{ aws_profile }}" \
        -var="aws_region={{ aws_region }}"
    @echo ""
    @echo "==> Live at:"
    @AWS_PROFILE={{ aws_profile }} terraform -chdir=terraform output -raw service_url

# Tear down ONLY the App Runner service to pause compute costs.
# ECR image and IAM roles are retained — `just deploy` restores service in ~3 min.
teardown:
    AWS_PROFILE={{ aws_profile }} terraform -chdir=terraform destroy \
        -target=aws_apprunner_service.app \
        -auto-approve \
        -var="aws_profile={{ aws_profile }}" \
        -var="aws_region={{ aws_region }}"
    @echo ""
    @echo "==> App Runner destroyed. Run 'just deploy' to bring it back."

# Push a new image and restart App Runner to pick it up.
redeploy TAG="latest":
    @echo "==> Pushing new image..."
    just push {{ TAG }}
    @echo ""
    @echo "==> Restarting App Runner with new image..."
    AWS_PROFILE={{ aws_profile }} AWS_DEFAULT_OUTPUT=json \
        aws apprunner start-deployment \
        --service-arn "$( \
            AWS_PROFILE={{ aws_profile }} \
            terraform -chdir=terraform output -raw service_arn \
        )" \
        --region {{ aws_region }}
    @echo ""
    @echo "==> Deployment triggered. Check status with: just status"

# Show current status and URL of the live App Runner service.
status:
    @AWS_PROFILE={{ aws_profile }} AWS_DEFAULT_OUTPUT=json \
        aws apprunner list-services \
        --region {{ aws_region }} \
        --query 'ServiceSummaryList[?ServiceName==`gas-and-energy-mechanics-copilot`].{Status:Status,URL:ServiceUrl}' \
        2>/dev/null || echo "No running service found."

# Destroy EVERYTHING (ECR images, IAM roles, App Runner).
# You will need to run `just bootstrap` before the next deployment.
destroy-all:
    AWS_PROFILE={{ aws_profile }} terraform -chdir=terraform destroy -auto-approve \
        -var="aws_profile={{ aws_profile }}" \
        -var="aws_region={{ aws_region }}"
    @echo ""
    @echo "==> All resources destroyed. Run 'just bootstrap' to start fresh."
