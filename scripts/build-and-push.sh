#!/usr/bin/env bash
# Build a linux/amd64 Docker image and push it to ECR.
# Usage: ./scripts/build-and-push.sh [image-tag]
#
# Requires: docker, aws CLI (any version with ecr support), AWS profile vscode-user

set -euo pipefail

IMAGE_TAG="${1:-latest}"
AWS_PROFILE="${AWS_PROFILE:-vscode-user}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Resolve ECR URL from Terraform output (falls back to hard-coded account/region)
if command -v terraform &>/dev/null && [ -f "$(dirname "$0")/../terraform/terraform.tfstate" ]; then
  ECR_URL=$(terraform -chdir="$(dirname "$0")/../terraform" output -raw ecr_repository_url 2>/dev/null || true)
fi
if [ -z "${ECR_URL:-}" ]; then
  ACCOUNT_ID=$(python3 -c "
import boto3, os
s = boto3.Session(profile_name=os.environ.get('AWS_PROFILE','vscode-user'))
print(s.client('sts').get_caller_identity()['Account'])
")
  ECR_URL="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/gas-and-energy-mechanics-copilot"
fi

FULL_IMAGE="${ECR_URL}:${IMAGE_TAG}"

echo "==> Building linux/amd64 image: ${FULL_IMAGE}"
docker buildx build \
  --platform linux/amd64 \
  -t "${FULL_IMAGE}" \
  "$(dirname "$0")/.."

echo "==> Logging in to ECR"
python3 -c "
import boto3, base64, subprocess, os
session = boto3.Session(profile_name=os.environ.get('AWS_PROFILE','vscode-user'), region_name=os.environ.get('AWS_REGION','us-east-1'))
ecr = session.client('ecr')
token = ecr.get_authorization_token()['authorizationData'][0]
user, pwd = base64.b64decode(token['authorizationToken']).decode().split(':',1)
endpoint = token['proxyEndpoint']
subprocess.run(['docker','login','--username',user,'--password-stdin',endpoint], input=pwd.encode(), check=True)
"

echo "==> Pushing ${FULL_IMAGE}"
docker push "${FULL_IMAGE}"
echo "==> Done: ${FULL_IMAGE}"
