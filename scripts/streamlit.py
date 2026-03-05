"""
AI Copilot - Interactive RAG Chatbot

A professional Streamlit interface for the AI Copilot RAG chatbot.
This application connects to the A2A server running locally and provides
an enhanced chat experience with:

- Real-time streaming responses from AWS Bedrock (Nova Lite)
- RAG-powered answers using FAISS vector search (8,524+ docs)
- Persistent conversation history within session
- Clean, professional UI with source citations
- Connection testing and error handling

Architecture:
- Backend: FastAPI A2A server with Strands Agent SDK
- LLM: AWS Bedrock (us.amazon.nova-lite-v1:0)
- Retrieval: FAISS index with OpenAI embeddings
- Frontend: Streamlit chat interface

To run:
1. Start A2A server: `export AWS_PROFILE=your-aws-profile && python3 -m uvicorn brightai.ai_copilot.entrypoint:app --reload --port 8080 --app-dir src`
2. Run chatbot: `export AWS_PROFILE=your-aws-profile && streamlit run scripts/streamlit.py`
"""

import asyncio
import logging
from typing import AsyncGenerator
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart
import httpx
import streamlit as st

# Configure logging to be less verbose for the UI
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300  # 5 minutes timeout
DEFAULT_BASE_URL = "http://127.0.0.1:8080"


def create_message(*, role: Role = Role.user, text: str) -> Message:
    """Create a properly formatted A2A message."""
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
    )


async def send_streaming_message(message: str, base_url: str = DEFAULT_BASE_URL) -> AsyncGenerator[str, None]:
    """
    Send a message to the A2A server and yield streaming response chunks.

    Args:
        message: The user's message to send
        base_url: The A2A server URL

    Yields:
        String chunks of the response as they arrive
    """
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as httpx_client:
            # Get agent card from the server
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()

            # Create streaming client
            config = ClientConfig(
                httpx_client=httpx_client,
                streaming=True,
            )
            factory = ClientFactory(config)
            client = factory.create(agent_card)

            # Create and send message
            msg = create_message(text=message)

            # Process streaming response
            async for event in client.send_message(msg):
                if isinstance(event, tuple) and len(event) == 2:
                    # Handle (Task, UpdateEvent) tuples - this is the main response format
                    task, update_event = event

                    # Extract text from the task's status message if it exists
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
                                # Extract text from Part.root.text (based on debug output)
                                if hasattr(part, "root") and hasattr(part.root, "text"):
                                    yield part.root.text
                                # Fallback for other part structures
                                elif hasattr(part, "content") and hasattr(part.content, "text"):
                                    yield part.content.text
                                elif hasattr(part, "text"):
                                    yield part.text

                elif isinstance(event, Message):
                    # Handle direct Message events (less common but possible)
                    for part in event.parts or []:
                        if hasattr(part, "root") and hasattr(part.root, "text"):
                            yield part.root.text
                        elif hasattr(part, "content") and hasattr(part.content, "text"):
                            yield part.content.text
                        elif hasattr(part, "text"):
                            yield part.text
                else:
                    # Fallback for other response types
                    yield str(event)

    except Exception as e:
        yield f"Error: {str(e)}"


async def test_connection(base_url: str = DEFAULT_BASE_URL) -> tuple[bool, str]:
    """Test connection to the A2A server."""
    try:
        async with httpx.AsyncClient(timeout=10) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            return True, f"Connected successfully. Agent: {agent_card.name}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="AI Copilot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            padding-top: 2rem;
        }

        /* Chat message styling - ensure proper contrast */
        .stChatMessage {
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        /* User messages - blue background with dark text */
        [data-testid="stChatMessageContent"]:has(.stChatMessage[data-testid*="user"]) {
            background-color: #e3f2fd !important;
            color: #1a1a1a !important;
        }

        /* Assistant messages - light gray background with dark text */
        [data-testid="stChatMessageContent"]:has(.stChatMessage[data-testid*="assistant"]) {
            background-color: #f5f5f5 !important;
            color: #1a1a1a !important;
        }

        /* Header styling */
        .chat-header {
            background: linear-gradient(90deg, #1e3a8a 0%, #3730a3 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
        }

        /* Info boxes - high contrast white background with border */
        .stAlert {
            background-color: #ffffff !important;
            border: 2px solid #3730a3 !important;
            color: #1a1a1a !important;
        }

        /* Info box text content */
        .stAlert > div {
            color: #1a1a1a !important;
        }

        /* Ensure all text in alerts is dark */
        .stAlert * {
            color: #1a1a1a !important;
        }

        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .status-success {
            background-color: #10b981;
            color: white;
        }
        .status-error {
            background-color: #ef4444;
            color: white;
        }

        /* Ensure markdown text is visible */
        .stMarkdown {
            color: inherit;
        }

        /* Chat input styling */
        .stChatInput {
            border-color: #3730a3;
        }

        /* Sidebar styling - white background with dark text */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
        }

        [data-testid="stSidebar"] * {
            color: #1a1a1a !important;
        }

        /* Sidebar headers */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #1e3a8a !important;
        }

        /* Sidebar markdown and text */
        [data-testid="stSidebar"] .stMarkdown {
            color: #1a1a1a !important;
        }

        /* Sidebar text input */
        [data-testid="stSidebar"] input {
            color: #1a1a1a !important;
            background-color: #ffffff !important;
        }

        /* Success/error messages in sidebar */
        [data-testid="stSidebar"] .stSuccess,
        [data-testid="stSidebar"] .stError,
        [data-testid="stSidebar"] .stInfo {
            color: #1a1a1a !important;
        }

        /* Success/error messages */
        .stSuccess {
            background-color: #d1f2eb !important;
            color: #0f5132 !important;
            border: 1px solid #0f5132 !important;
        }

        .stError {
            background-color: #f8d7da !important;
            color: #721c24 !important;
            border: 1px solid #721c24 !important;
        }

        /* Ensure button text is visible */
        .stButton button {
            color: #ffffff;
            background-color: #3730a3;
            border: none;
        }

        .stButton button:hover {
            background-color: #1e3a8a;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="chat-header">
            <h1 style="margin: 0; font-size: 2.5rem;">🤖 AI Copilot</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                AI-powered engineering documentation assistant with RAG retrieval
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Server configuration
        st.subheader("Server Settings")
        server_url = st.text_input(
            "A2A Server URL",
            value=DEFAULT_BASE_URL,
            help="URL where the A2A server is running"
        )

        # Connection status
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🔌 Test", use_container_width=True):
                with st.spinner("Testing..."):
                    success, message = asyncio.run(test_connection(server_url))
                    st.session_state.connection_status = (success, message)

        if "connection_status" in st.session_state:
            success, message = st.session_state.connection_status
            if success:
                st.success("✅ Connected", icon="✅")
            else:
                st.error("❌ Disconnected", icon="❌")

        st.markdown("---")

        # System information
        st.subheader("📊 System Info")
        st.markdown("""
        **Model**: AWS Bedrock Nova Lite
        **Vector DB**: FAISS (8,524 docs)
        **Embeddings**: OpenAI (1536D)
        **Region**: us-west-2
        """)

        st.markdown("---")

        # Usage instructions
        st.subheader("📖 Quick Start")
        st.markdown("""
        1. Ensure A2A server is running
        2. Test connection above
        3. Ask questions about:
           - Engine troubleshooting
           - Configuration parameters
           - Error codes
           - Maintenance procedures
        """)

        st.markdown("---")

        # Chat controls
        st.subheader("🗑️ Chat Controls")
        if st.button("Clear History", use_container_width=True):
            st.session_state.messages = []
            if "connection_status" in st.session_state:
                del st.session_state.connection_status
            st.rerun()

        # Show message count
        if "messages" in st.session_state and st.session_state.messages:
            msg_count = len(st.session_state.messages)
            st.info(f"💬 {msg_count} messages in history")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Main chat area
    st.markdown("### 💬 Conversation")

    # Show welcome message if no messages
    if not st.session_state.messages:
        st.info("""
        👋 **Welcome to AI Copilot!**

        I'm here to help you with engineering documentation queries. I can assist with:
        - 🔧 Engine troubleshooting and diagnostics
        - ⚙️ Configuration parameters and settings
        - 🚨 Error codes and warnings
        - 🔍 Maintenance and repair procedures

        **Try asking:**
        - "What are the troubleshooting steps for engine issues?"
        - "How do I check voltage readings?"
        - "What does error code XYZ mean?"
        """)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input with better prompt
    if prompt := st.chat_input("Ask me anything about engineering documentation..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Use asyncio to handle the streaming response
                async def get_response():
                    response_text = ""
                    async for chunk in send_streaming_message(prompt, server_url):
                        if chunk.strip():  # Only add non-empty chunks
                            response_text += chunk
                            # Update the UI with current response + typing indicator
                            message_placeholder.markdown(response_text + "▌")
                    return response_text

                with st.spinner("Thinking..."):
                    full_response = asyncio.run(get_response())

                message_placeholder.markdown(full_response)

            except Exception as e:
                error_message = f"Error communicating with A2A server: {str(e)}"
                message_placeholder.error(error_message)
                full_response = error_message

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
