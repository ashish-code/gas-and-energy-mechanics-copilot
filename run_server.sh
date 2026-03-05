#!/bin/bash

# AI Copilot - A2A Server Launcher
# This script sets up the environment and launches the A2A server

set -e

echo "🚀 AI Copilot - Starting A2A Server..."
echo "================================================"
echo ""

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "📋 Loading environment from .env file..."
    source .env
    echo "✅ Environment variables loaded"
    echo ""
fi

# Check if required environment variables are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY not set."
    echo ""
    echo "Please either:"
    echo "   1. Create a .env file with your API key (recommended)"
    echo "   2. Export it manually: export OPENAI_API_KEY='your-key'"
    echo ""
    exit 1
fi

# Set required environment variables if not already set
export AWS_PROFILE="${AWS_PROFILE:?Please set AWS_PROFILE to your personal AWS profile name}"
export CONFIG_DIR="${CONFIG_DIR:-./config}"

echo "✅ Environment configured:"
echo "   AWS_PROFILE: $AWS_PROFILE"
echo "   CONFIG_DIR: $CONFIG_DIR"
echo "   OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..." # Show only first 10 chars
echo ""

# Verify AWS credentials
echo "🔐 Verifying AWS credentials..."
if aws sts get-caller-identity >/dev/null 2>&1; then
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    echo "✅ AWS authenticated (Account: $ACCOUNT)"
else
    echo "❌ AWS authentication failed!"
    echo "   Please run: gimme-aws-creds"
    echo ""
    exit 1
fi

echo ""

# Check if port 8080 is already in use
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 8080 is already in use. Killing existing process..."
    lsof -ti:8080 | xargs kill -9
    sleep 2
fi

echo "🚀 Starting A2A server on http://localhost:8080..."
echo "================================================"
echo ""
echo "Available endpoints:"
echo "  - Application: http://localhost:8080"
echo "  - API Docs: http://localhost:8080/docs"
echo "  - Health Check: http://localhost:8080/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Determine which Python to use
PYTHON_CMD="python3"
echo "📦 Using system Python: $(which python3)"
echo "   Version: $(python3 --version)"

# Verify required packages are available
echo ""
echo "🔍 Checking required packages..."
MISSING_PACKAGES=()

for package in openai faiss boto3 pandas pyarrow structlog uvicorn; do
    if ! $PYTHON_CMD -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=($package)
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "❌ ERROR: Missing packages: ${MISSING_PACKAGES[@]}"
    echo ""
    echo "Please install required packages:"
    if [ "$package" = "faiss" ]; then
        echo "   pip3 install --user openai faiss-cpu boto3 pandas pyarrow structlog uvicorn"
    else
        echo "   pip3 install --user ${MISSING_PACKAGES[@]}"
    fi
    echo ""
    exit 1
fi

echo "✅ All required packages verified"

echo ""

# Launch server
$PYTHON_CMD -m uvicorn brightai.ai_copilot.entrypoint:app \
    --reload \
    --port 8080 \
    --app-dir src
