#!/usr/bin/env bash
# Build a linux/amd64 Docker image and push it to ECR.
#
# Usage:
#   ./scripts/build-and-push.sh [image-tag]
#   AWS_PROFILE=vscode-user ./scripts/build-and-push.sh latest
#
# Prerequisites: docker, aws CLI, AWS credentials configured.
# IMPORTANT: must use linux/amd64 — App Runner does not support arm64.

set -euo pipefail

IMAGE_TAG="${1:-latest}"
AWS_PROFILE="${AWS_PROFILE:-vscode-user}"
AWS_REGION="${AWS_REGION:-us-east-1}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve ECR URL: prefer terraform output, fall back to AWS STS account lookup
ECR_URL=""
if command -v terraform &>/dev/null && [ -f "${REPO_DIR}/terraform/terraform.tfstate" ]; then
    ECR_URL=$(terraform -chdir="${REPO_DIR}/terraform" output -raw ecr_repository_url 2>/dev/null || true)
fi
if [ -z "${ECR_URL}" ]; then
    ACCOUNT_ID=$(AWS_PROFILE="${AWS_PROFILE}" AWS_DEFAULT_OUTPUT=json \
        aws sts get-caller-identity \
        --region "${AWS_REGION}" \
        --query Account \
        --output text)
    ECR_URL="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/gas-and-energy-mechanics-copilot"
fi

FULL_IMAGE="${ECR_URL}:${IMAGE_TAG}"

echo "==> Building linux/amd64 image: ${FULL_IMAGE}"
docker buildx build \
    --platform linux/amd64 \
    -t "${FULL_IMAGE}" \
    "${REPO_DIR}"

echo "==> Logging in to ECR (${AWS_REGION})"
AWS_PROFILE="${AWS_PROFILE}" \
    aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login \
        --username AWS \
        --password-stdin \
        "${ECR_URL%%/*}"   # registry hostname only (strips /repo-name)

echo "==> Pushing ${FULL_IMAGE}"
docker push "${FULL_IMAGE}"

echo ""
echo "==> Done: ${FULL_IMAGE}"
