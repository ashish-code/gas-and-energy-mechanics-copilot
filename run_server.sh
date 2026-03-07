#!/bin/bash
# Gas & Energy Mechanics Copilot — A2A Server Launcher

set -e

echo "==> Gas & Energy Mechanics Copilot — Starting A2A Server"
echo ""

# Load .env if present
if [ -f ".env" ]; then
    echo "Loading environment from .env ..."
    source .env
fi

export AWS_PROFILE="${AWS_PROFILE:-vscode-user}"
export CONFIG_DIR="${CONFIG_DIR:-./config}"

echo "AWS_PROFILE : $AWS_PROFILE"
echo "CONFIG_DIR  : $CONFIG_DIR"
echo ""

# Verify AWS credentials
echo "Verifying AWS credentials ..."
if aws sts get-caller-identity --profile "$AWS_PROFILE" >/dev/null 2>&1; then
    ACCOUNT=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text)
    echo "Authenticated  (account: $ACCOUNT)"
else
    echo "ERROR: AWS authentication failed. Check AWS_PROFILE=$AWS_PROFILE"
    exit 1
fi

echo ""

# Free port 8080 if occupied
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Port 8080 in use — killing existing process ..."
    lsof -ti:8080 | xargs kill -9
    sleep 1
fi

echo "Starting server on http://localhost:8080 (Ctrl+C to stop)"
echo ""

python3 -m uvicorn gas_energy_copilot.ai_copilot.entrypoint:app \
    --reload \
    --port 8080 \
    --app-dir src
