#!/bin/bash
# Gas & Energy Mechanics Copilot — Streamlit Chatbot Launcher

set -e

echo "==> Gas & Energy Mechanics Copilot — Starting Chatbot UI"
echo ""

# Load .env if present
if [ -f ".env" ]; then
    echo "Loading environment from .env ..."
    source .env
fi

export AWS_PROFILE="${AWS_PROFILE:-vscode-user}"
export CONFIG_DIR="${CONFIG_DIR:-./config}"

echo "AWS_PROFILE : $AWS_PROFILE"
echo ""

# Verify backend server is running
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Backend server is running on :8080"
else
    echo "ERROR: Backend server is not running."
    echo "Start it first with: ./run_server.sh"
    exit 1
fi

echo ""
echo "Launching Streamlit UI at http://localhost:8501 (Ctrl+C to stop)"
echo ""

streamlit run scripts/streamlit.py
