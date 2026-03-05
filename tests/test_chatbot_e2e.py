"""
End-to-end test for AI Copilot chatbot.
This script tests the full flow: A2A server -> Strands Agent -> RAG retrieval -> Bedrock LLM.
"""

import asyncio
import json
import sys
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart
import httpx
from uuid import uuid4

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT = 300  # 5 minutes

def create_message(*, role: Role = Role.user, text: str) -> Message:
    """Create a properly formatted A2A message."""
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
    )

async def test_chatbot(query: str, base_url: str = DEFAULT_BASE_URL):
    """
    Test the chatbot with a specific query.

    Args:
        query: The user's question
        base_url: The A2A server URL
    """
    print("=" * 80)
    print("🤖 AI Copilot - End-to-End Test")
    print("=" * 80)
    print()

    print(f"📝 Query: {query}")
    print()

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as httpx_client:
            # Step 1: Get agent card
            print("🔗 Step 1: Connecting to A2A server...")
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            print(f"   ✅ Connected to: {agent_card.name}")
            print()

            # Step 2: Create streaming client
            print("🔧 Step 2: Creating streaming client...")
            config = ClientConfig(
                httpx_client=httpx_client,
                streaming=True,
            )
            factory = ClientFactory(config)
            client = factory.create(agent_card)
            print("   ✅ Client created")
            print()

            # Step 3: Send message and collect response
            print("📤 Step 3: Sending message to agent...")
            msg = create_message(text=query)
            print("   ✅ Message sent, waiting for response...")
            print()

            print("💬 Step 4: Streaming response:")
            print("-" * 80)

            full_response = ""
            chunk_count = 0

            async for event in client.send_message(msg):
                if isinstance(event, tuple) and len(event) == 2:
                    task, update_event = event

                    # Extract text from task status
                    if (
                        task
                        and hasattr(task, "status")
                        and task.status
                        and hasattr(task.status, "message")
                        and task.status.message
                    ):
                        status_message = task.status.message
                        if hasattr(status_message, "parts") and status_message.parts:
                            for part in status_message.parts:
                                if hasattr(part, "root") and hasattr(part.root, "text"):
                                    text_chunk = part.root.text
                                    if text_chunk.strip():
                                        full_response += text_chunk
                                        chunk_count += 1
                                        # Print chunk in real-time
                                        print(text_chunk, end="", flush=True)

                elif isinstance(event, Message):
                    for part in event.parts or []:
                        if hasattr(part, "root") and hasattr(part.root, "text"):
                            text_chunk = part.root.text
                            if text_chunk.strip():
                                full_response += text_chunk
                                chunk_count += 1
                                print(text_chunk, end="", flush=True)

            print()
            print("-" * 80)
            print()

            # Results
            print("📊 Step 5: Test Results")
            print("-" * 80)
            print(f"✅ Response received: {len(full_response)} characters")
            print(f"✅ Chunks streamed: {chunk_count}")
            print()

            # Check if response mentions documentation sources
            has_sources = any(keyword in full_response.lower() for keyword in [
                'manual', 'documentation', 'page', 'section', 'source', 'according to'
            ])

            if has_sources:
                print("✅ Response includes documentation sources (RAG working!)")
            else:
                print("⚠️  Response may not include documentation sources")

            print()
            print("=" * 80)
            print("✅ End-to-end test PASSED!")
            print("=" * 80)

            return True

    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ Test FAILED with error: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test query about engine troubleshooting
    test_query = "What are the troubleshooting steps for engine issues?"

    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])

    success = asyncio.run(test_chatbot(test_query))
    sys.exit(0 if success else 1)
