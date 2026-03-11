"""
End-to-end test for the Gas & Energy Mechanics Copilot REST API.
Tests the full flow: POST /v1/chat → CrewAI crew → RAG retrieval → Bedrock LLM.

Usage:
    CONFIG_DIR="./config" uv run pytest tests/test_chatbot_e2e.py -v -s
"""

import httpx
import pytest

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT = 300


@pytest.mark.integration
def test_chat_endpoint_returns_answer():
    """POST /v1/chat should return a non-empty answer string."""
    query = "What are the pressure testing requirements for steel pipelines under 49 CFR §192.505?"

    resp = httpx.post(
        f"{DEFAULT_BASE_URL}/v1/chat",
        json={"question": query},
        timeout=DEFAULT_TIMEOUT,
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    body = resp.json()
    assert "answer" in body, f"Missing 'answer' key in response: {body}"
    assert len(body["answer"]) > 50, "Answer seems too short"

    # Verify the response mentions regulatory language
    has_sources = any(
        kw in body["answer"].lower()
        for kw in ["§", "cfr", "phmsa", "pressure", "pipeline"]
    )
    assert has_sources, "Answer does not appear to reference pipeline regulations"


@pytest.mark.integration
def test_chat_endpoint_rejects_empty_question():
    """POST /v1/chat with an empty question should return 400."""
    resp = httpx.post(
        f"{DEFAULT_BASE_URL}/v1/chat",
        json={"question": "   "},
        timeout=10,
    )
    assert resp.status_code == 400
