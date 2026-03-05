#!/bin/bash

# AI Copilot - Streamlit Chatbot Launcher
# This script sets up the environment and launches the Streamlit chatbot UI

set -e

echo "🤖 AI Copilot - Starting Chatbot..."
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
    echo "⚠️  Warning: OPENAI_API_KEY not set. RAG embeddings may not work."
    echo ""
    echo "Please either:"
    echo "   1. Create a .env file with your API key (recommended)"
    echo "   2. Export it manually: export OPENAI_API_KEY='your-key'"
    echo ""
fi

# Set required environment variables if not already set
export AWS_PROFILE="${AWS_PROFILE:?Please set AWS_PROFILE to your personal AWS profile name}"
export CONFIG_DIR="${CONFIG_DIR:-./config}"

echo "✅ Environment configured:"
echo "   AWS_PROFILE: $AWS_PROFILE"
echo "   CONFIG_DIR: $CONFIG_DIR"
echo "   OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..." # Show only first 10 chars
echo ""

# Check if A2A server is running
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "✅ A2A server is running on port 8080"
else
    echo "❌ A2A server is NOT running on port 8080"
    echo ""
    echo "Please start the server first with:"
    echo "   ./run_server.sh"
    echo ""
    echo "Or manually:"
    echo "   source .env"
    echo "   python3 -m uvicorn brightai.ai_copilot.entrypoint:app --reload --port 8080 --app-dir src"
    echo ""
    exit 1
fi

echo ""
echo "🚀 Launching Streamlit chatbot..."
echo "================================================"
echo ""

# Launch Streamlit
streamlit run scripts/streamlit.py
