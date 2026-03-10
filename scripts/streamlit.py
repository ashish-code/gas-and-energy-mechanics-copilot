"""
Gas & Energy Mechanics Copilot — Streamlit Frontend

Chat interface that calls the CrewAI-backed REST API.

To run:
    # Terminal 1 — start the backend
    just dev

    # Terminal 2 — start the UI
    just chat
"""

import logging
import re

import httpx
import streamlit as st

logging.basicConfig(level=logging.WARNING)

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT = 300

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Gas & Energy Mechanics Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Header banner ─────────────────────────────────────────────────── */
.copilot-header {
    background: linear-gradient(135deg, #1E40AF 0%, #1E3A8A 60%, #0F172A 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid #2563EB;
}
.copilot-header h1 { margin: 0; font-size: 1.9rem; color: #F1F5F9; letter-spacing: -0.5px; }
.copilot-header p  { margin: 0.3rem 0 0 0; color: #94A3B8; font-size: 0.95rem; }

/* ── Sidebar ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #3B82F6 !important; }

/* ── Source citation expanders ─────────────────────────────────────── */
.source-chip {
    display: inline-block;
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #93C5FD;
    margin: 2px 3px;
    font-family: monospace;
}

/* ── Status pills ──────────────────────────────────────────────────── */
.pill-ok  { color: #34D399; font-weight: 600; }
.pill-err { color: #F87171; font-weight: 600; }

/* ── Welcome card ──────────────────────────────────────────────────── */
.welcome-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-left: 4px solid #3B82F6;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.welcome-card h4 { margin: 0 0 0.5rem 0; color: #93C5FD; }
.welcome-card ul { margin: 0; padding-left: 1.2rem; color: #CBD5E1; }
.welcome-card li { margin-bottom: 0.25rem; }

/* ── Chat messages ─────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 10px;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "connected" not in st.session_state:
    st.session_state.connected = None
if "conn_detail" not in st.session_state:
    st.session_state.conn_detail = ""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚡ Gas & Energy\nMechanics Copilot")
    st.markdown("---")

    st.markdown("### 🔌 Server")
    server_url = st.text_input("Backend URL", value=DEFAULT_BASE_URL, label_visibility="collapsed")

    if st.button("Test Connection", use_container_width=True):
        try:
            resp = httpx.get(f"{server_url}/health", timeout=10)
            if resp.status_code == 200:
                st.session_state.connected = True
                st.session_state.conn_detail = "API server healthy"
            else:
                st.session_state.connected = False
                st.session_state.conn_detail = f"HTTP {resp.status_code}"
        except Exception as exc:
            st.session_state.connected = False
            st.session_state.conn_detail = str(exc)

    if st.session_state.connected is True:
        st.markdown(f'<p class="pill-ok">● Connected — {st.session_state.conn_detail}</p>',
                    unsafe_allow_html=True)
    elif st.session_state.connected is False:
        st.markdown('<p class="pill-err">● Disconnected</p>', unsafe_allow_html=True)
        st.caption(st.session_state.conn_detail)

    st.markdown("---")
    st.markdown("### 📚 Model Info")
    st.markdown("""
| Parameter | Value |
|---|---|
| LLM | GPT-OSS 120B via Bedrock |
| Embeddings | Titan V2 (1024D) |
| Index | FAISS IndexFlatIP |
| Sources | 49 CFR 192, 193, 195 |
| Region | us-east-1 |
""")

    st.markdown("---")
    st.markdown("### 🗑️ Session")
    if st.button("Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.session_state.messages:
        n = len(st.session_state.messages)
        st.caption(f"{n} message{'s' if n != 1 else ''} in session")

# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="copilot-header">
  <h1>⚡ Gas &amp; Energy Mechanics Copilot</h1>
  <p>AI assistant grounded in PHMSA regulations · 49 CFR Parts 192 · 193 · 195</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Welcome card (only when no messages)
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    st.markdown("""
<div class="welcome-card">
  <h4>How can I help you today?</h4>
  <ul>
    <li>Pipeline safety regulations and PHMSA compliance (49 CFR Part 192)</li>
    <li>LNG facility design, operations, and safety requirements (Part 193)</li>
    <li>Hazardous liquid pipeline integrity management (Part 195)</li>
    <li>Compressor station operations and troubleshooting</li>
    <li>Corrosion control, cathodic protection, and IMP requirements</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 {len(msg['sources'])} source(s) retrieved", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<span class="source-chip">{src}</span>',
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask about pipeline safety, LNG operations, compressor stations…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation and synthesising answer…"):
            try:
                resp = httpx.post(
                    f"{server_url}/v1/chat",
                    json={"question": prompt},
                    timeout=DEFAULT_TIMEOUT,
                )
                resp.raise_for_status()
                full_response = resp.json()["answer"]
            except Exception as exc:
                full_response = f"❌ Error: {exc}"

        st.markdown(full_response)

    sources_found = list(dict.fromkeys(
        re.findall(r"(?:§\s*\d+\.\d+|49 CFR Part \d+)", full_response)
    ))

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources_found,
    })
    st.rerun()
