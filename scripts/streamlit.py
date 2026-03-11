"""
Gas & Energy Mechanics Copilot — Streamlit Frontend

Chat interface that calls the CrewAI-backed REST API.

WHAT'S NEW (multi-agent branch):
    - session_id  — UUID persisted in session state so the DynamoDB memory manager
                    can load prior turns. Each browser tab gets its own session.
    - Judge verdict badge — shows the judge agent's verdict (approved/needs_revision/
                    rejected) and confidence score next to each assistant response.
    - Streaming mode — consumes SSE events from /v1/chat/stream, showing agent steps
                    in real-time before the final answer arrives.
    - Eval metrics sidebar panel — fetches /v1/eval/metrics and displays the RAG
                    Triad scores (context relevance, groundedness, answer relevance)
                    from TruLens so you can monitor quality across sessions.
    - Pipeline selector — toggle between simple 2-agent pipeline (fast) and full
                    4-agent pipeline (router → retrieval → synthesis → judge).

HOW session_id WORKS:
    Streamlit re-executes the entire script on each interaction. The UUID is
    stored in st.session_state so it survives re-runs within the same browser tab.
    When the user clicks "Clear History", we generate a new UUID to start a fresh
    memory session in DynamoDB — the old turns are not deleted (TTL handles expiry).

HOW STREAMING WORKS:
    We use httpx's streaming context manager to consume Server-Sent Events from
    /v1/chat/stream. Each SSE event is a JSON object with:
        {"type": "step", "agent": "retrieval_specialist", "data": "..."}
        {"type": "complete", "answer": "...", "verdict": "approved", ...}
    Streamlit does not have native SSE support, so we parse the `data: ` lines
    manually and accumulate text into a placeholder widget.

To run:
    # Terminal 1 — start the backend
    just dev

    # Terminal 2 — start the UI
    just chat
"""

import json
import logging
import re
import uuid

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
.pill-ok       { color: #34D399; font-weight: 600; }
.pill-warn     { color: #FBBF24; font-weight: 600; }
.pill-err      { color: #F87171; font-weight: 600; }

/* ── Verdict badges ─────────────────────────────────────────────────── */
.verdict-approved     { background: #064E3B; color: #6EE7B7; border: 1px solid #059669;
                        border-radius: 12px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
.verdict-needs_revision { background: #78350F; color: #FDE68A; border: 1px solid #D97706;
                        border-radius: 12px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
.verdict-rejected     { background: #7F1D1D; color: #FCA5A5; border: 1px solid #DC2626;
                        border-radius: 12px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }

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

/* ── Metric cards ─────────────────────────────────────────────────── */
.metric-row { display: flex; gap: 0.5rem; margin-top: 0.4rem; }
.metric-card {
    flex: 1;
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    text-align: center;
}
.metric-card .metric-val { font-size: 1.3rem; font-weight: 700; color: #60A5FA; }
.metric-card .metric-lbl { font-size: 0.65rem; color: #94A3B8; text-transform: uppercase; }
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

# session_id persists for the lifetime of the browser tab.
# A new UUID is generated when the session starts or when the user clears chat.
# This UUID is sent to the API so the DynamoDB memory manager can store/retrieve
# the conversation history for multi-turn context awareness.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "eval_metrics" not in st.session_state:
    st.session_state.eval_metrics = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _verdict_badge(verdict: str | None) -> str:
    """Return an HTML badge for the judge verdict."""
    if not verdict:
        return ""
    css = f"verdict-{verdict}"
    labels = {
        "approved": "✓ Approved",
        "needs_revision": "⚠ Needs Revision",
        "rejected": "✗ Rejected",
    }
    label = labels.get(verdict, verdict)
    return f'<span class="{css}">{label}</span>'


def _call_api_blocking(server_url: str, question: str, session_id: str, use_full_pipeline: bool) -> dict:
    """
    Call the standard (non-streaming) POST /v1/chat endpoint.

    Returns the full JSON response dict, or a dict with 'error' on failure.
    """
    try:
        resp = httpx.post(
            f"{server_url}/v1/chat",
            json={
                "question": question,
                "session_id": session_id,
                "use_full_pipeline": use_full_pipeline,
            },
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"answer": f"❌ Error: {exc}", "error": str(exc)}


def _call_api_streaming(
    server_url: str,
    question: str,
    session_id: str,
    use_full_pipeline: bool,
    step_placeholder,
    answer_placeholder,
) -> dict:
    """
    Consume Server-Sent Events from POST /v1/chat/stream.

    WHY asyncio.Queue / SSE parsing:
        The backend sends SSE events as newline-delimited `data: {...}` lines.
        httpx.stream() iterates over lines synchronously, which is fine here
        because Streamlit runs on the main thread in blocking mode.

    Event types:
        {"type": "step", "agent": "<name>", "data": "<step text>"}   — agent step
        {"type": "complete", "answer": "...", "verdict": "...", ...}  — final answer

    Args:
        step_placeholder:   st.empty() for displaying live agent steps
        answer_placeholder: st.empty() for displaying the final answer as it arrives

    Returns:
        The parsed final answer dict (same shape as non-streaming response).
    """
    result = {"answer": "", "error": None}
    steps_seen = []

    try:
        with httpx.stream(
            "POST",
            f"{server_url}/v1/chat/stream",
            json={
                "question": question,
                "session_id": session_id,
                "use_full_pipeline": use_full_pipeline,
            },
            timeout=DEFAULT_TIMEOUT,
        ) as response:
            for line in response.iter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "step":
                    agent = event.get("agent", "agent")
                    data = event.get("data", "")
                    steps_seen.append(f"**{agent}**: {data[:200]}")
                    # Show running log of agent steps
                    step_placeholder.markdown(
                        "**Pipeline Progress:**\n\n" + "\n\n".join(steps_seen[-5:])
                    )

                elif event.get("type") == "complete":
                    step_placeholder.empty()  # hide step log once answer is ready
                    result = event
                    answer_placeholder.markdown(event.get("answer", ""))

                elif event.get("type") == "error":
                    result = {"answer": f"❌ {event.get('message', 'Unknown error')}", "error": event.get("message")}

    except Exception as exc:
        result = {"answer": f"❌ Streaming error: {exc}", "error": str(exc)}

    return result


def _fetch_eval_metrics(server_url: str) -> dict | None:
    """Fetch aggregate eval metrics from /v1/eval/metrics. Returns None on failure."""
    try:
        resp = httpx.get(f"{server_url}/v1/eval/metrics", timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


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

    # ── Pipeline settings ─────────────────────────────────────────────
    st.markdown("### ⚙️ Pipeline")

    use_full_pipeline = st.toggle(
        "Full 4-agent pipeline",
        value=True,
        help=(
            "ON: Router → Retrieval → Synthesis → Judge (slower, higher accuracy).\n"
            "OFF: Retrieval → Synthesis only (faster, no quality gating)."
        ),
    )

    use_streaming = st.toggle(
        "Streaming mode (SSE)",
        value=False,
        help="Stream agent steps as they happen. Requires /v1/chat/stream endpoint.",
    )

    pipeline_label = "Full (4-agent)" if use_full_pipeline else "Simple (2-agent)"
    st.caption(f"Pipeline: **{pipeline_label}**")
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    st.markdown("---")

    # ── Model info ───────────────────────────────────────────────────
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

    # ── Eval metrics ─────────────────────────────────────────────────
    st.markdown("### 📊 RAG Quality Metrics")
    st.caption("Powered by TruLens (live) and DeepEval (CI gate)")

    col_refresh, _ = st.columns([2, 1])
    with col_refresh:
        if st.button("Refresh Metrics", use_container_width=True):
            st.session_state.eval_metrics = _fetch_eval_metrics(server_url)

    m = st.session_state.eval_metrics
    if m and m.get("status") == "ok":
        scores = m.get("scores", {})

        def _metric_html(label: str, value: float | None) -> str:
            if value is None:
                return f'<div class="metric-card"><div class="metric-val">—</div><div class="metric-lbl">{label}</div></div>'
            colour = "#34D399" if value >= 0.7 else "#FBBF24" if value >= 0.5 else "#F87171"
            return (
                f'<div class="metric-card">'
                f'<div class="metric-val" style="color:{colour}">{value:.2f}</div>'
                f'<div class="metric-lbl">{label}</div>'
                f"</div>"
            )

        st.markdown(
            '<div class="metric-row">'
            + _metric_html("Context<br>Relevance", scores.get("context_relevance"))
            + _metric_html("Groundedness", scores.get("groundedness"))
            + _metric_html("Answer<br>Relevance", scores.get("answer_relevance"))
            + "</div>",
            unsafe_allow_html=True,
        )
        if m.get("num_records"):
            st.caption(f"Based on {m['num_records']} recorded interactions")
    elif m and m.get("status") == "disabled":
        st.caption("Evaluation disabled (set [app.eval] enabled=true)")
    elif m is not None:
        st.caption("Could not load metrics")
    else:
        st.caption("Click 'Refresh Metrics' to load")

    st.markdown("---")

    # ── Session management ───────────────────────────────────────────
    st.markdown("### 🗑️ Session")
    if st.button("Clear History", use_container_width=True):
        st.session_state.messages = []
        # Generate a new session_id so DynamoDB memory starts fresh for new chat
        st.session_state.session_id = str(uuid.uuid4())
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

        # Show judge verdict badge for assistant messages (full pipeline only)
        if msg["role"] == "assistant" and msg.get("verdict"):
            badge = _verdict_badge(msg["verdict"])
            conf = msg.get("confidence")
            conf_str = f" · confidence {conf:.0%}" if conf is not None else ""
            st.markdown(
                f'{badge}{conf_str}',
                unsafe_allow_html=True,
            )
            # Show issues if the judge flagged any
            if msg.get("issues"):
                with st.expander("⚠️ Judge flagged issues", expanded=False):
                    for issue in msg["issues"]:
                        st.markdown(f"- {issue}")

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
        if use_streaming:
            # ── Streaming mode ────────────────────────────────────────────
            # We show live agent step updates, then replace with the final answer.
            step_area = st.empty()
            answer_area = st.empty()
            response_data = _call_api_streaming(
                server_url=server_url,
                question=prompt,
                session_id=st.session_state.session_id,
                use_full_pipeline=use_full_pipeline,
                step_placeholder=step_area,
                answer_placeholder=answer_area,
            )
            full_response = response_data.get("answer", "")
        else:
            # ── Blocking mode ─────────────────────────────────────────────
            with st.spinner("Searching documentation and synthesising answer…"):
                response_data = _call_api_blocking(
                    server_url=server_url,
                    question=prompt,
                    session_id=st.session_state.session_id,
                    use_full_pipeline=use_full_pipeline,
                )
            full_response = response_data.get("answer", "")
            st.markdown(full_response)

        # Extract judge verdict metadata from response
        verdict = response_data.get("verdict")
        confidence = response_data.get("confidence")
        issues = response_data.get("issues", [])

        # Show verdict badge immediately (before rerun saves to history)
        if verdict:
            badge = _verdict_badge(verdict)
            conf_str = f" · confidence {confidence:.0%}" if confidence is not None else ""
            st.markdown(f'{badge}{conf_str}', unsafe_allow_html=True)
            if issues:
                with st.expander("⚠️ Judge flagged issues", expanded=False):
                    for issue in issues:
                        st.markdown(f"- {issue}")

    # Extract inline CFR citations from the answer text for the sources chip bar
    sources_found = list(dict.fromkeys(
        re.findall(r"(?:§\s*\d+\.\d+|49 CFR Part \d+)", full_response)
    ))

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources_found,
        "verdict": verdict,
        "confidence": confidence,
        "issues": issues,
    })
    st.rerun()
