"""Gas & Energy Mechanics Copilot — Streamlit demo.

Shows the plan-execute-verify pipeline tier by tier: the planner's decomposition (or
refusal), the retrieved evidence sections, and the verified answer with per-claim
verification state. Default Streamlit components only.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure the repo root is importable (`streamlit run app/streamlit_app.py` puts app/ on the
# path, not the root) so `src.*` and `app.*` imports resolve both locally and on Streamlit Cloud.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

# --- Bridge Streamlit Cloud secrets into the environment BEFORE importing src.config -------
# (src.config reads os.environ at import; on Streamlit Cloud creds live in st.secrets.)
for _key in (
    "AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
    "SUPABASE_DB_URL", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_TRACING",
    "LOG_JSON", "LOG_LEVEL",
):
    if _key in st.secrets and _key not in os.environ:
        os.environ[_key] = str(st.secrets[_key])

from app.example_queries import EXAMPLE_QUERIES  # noqa: E402

from src.agents.graph import Copilot  # noqa: E402
from src.logging_setup import configure_logging  # noqa: E402
from src.observability.langsmith_setup import setup_langsmith  # noqa: E402

st.set_page_config(page_title="Gas & Energy Mechanics Copilot", page_icon="🛢️", layout="wide")


@st.cache_resource(show_spinner="Warming up: loading BM25 index + connecting to Bedrock/Supabase…")
def get_copilot() -> Copilot:
    configure_logging()
    setup_langsmith()
    return Copilot()


def render_plan(plan) -> None:  # type: ignore[no-untyped-def]
    if not plan.in_scope:
        return
    with st.expander(f"🧭 Plan — {len(plan.sub_questions)} sub-question(s)", expanded=True):
        for sq in plan.sub_questions:
            hint = f"  ·  _source: {sq.source_hint}_" if sq.source_hint else ""
            st.markdown(f"**{sq.id}.** {sq.question}{hint}")


def render_evidence(evidence) -> None:  # type: ignore[no-untyped-def]
    with st.expander(f"📚 Evidence — {len(evidence)} unique section(s)", expanded=False):
        for ev in evidence:
            cite = ev.metadata.get("citation") or " > ".join(ev.section_path) or ev.parent_id
            url = ev.metadata.get("url")
            head = f"**{cite}**  ·  _{ev.source}_  ·  score {ev.score:.3f}"
            head += f"  ·  [source]({url})" if url else ""
            st.markdown(head)
            st.caption(ev.text[:600] + ("…" if len(ev.text) > 600 else ""))
            st.divider()


def render_answer(answer) -> None:  # type: ignore[no-untyped-def]
    if answer.refused:
        st.warning(f"🚫 **Out of scope.** {answer.refusal_reason}")
        st.caption("The planner found no in-scope sub-questions — the system refuses rather than guess.")
        return

    st.markdown("### Answer")
    st.markdown(answer.summary or "_No answer synthesized._")

    if answer.claims:
        st.markdown("#### ✅ Verified claims")
        for c in answer.claims:
            st.markdown(f"- {c.text}  \n  ↳ `{c.citation}` — _“{c.quote[:140]}”_")
    if answer.unsupported_claims:
        st.markdown("#### ⚠️ Unsupported claims (surfaced, not dropped)")
        for c in answer.unsupported_claims:
            flags = []
            if not c.citation_exists:
                flags.append("✗ citation missing")
            elif not c.quote_matches:
                flags.append("✗ quote not found in source")
            elif not c.entailed:
                flags.append("⚠ not entailed by evidence")
            st.markdown(f"- {c.text}  \n  ↳ `{c.citation}` — {', '.join(flags)}  ·  _{c.reason}_")


def run_query(query: str) -> None:
    copilot = get_copilot()
    plan_box, evidence_box, answer_box = st.container(), st.container(), st.container()
    with st.status("Running plan → execute → verify…", expanded=True) as status:
        evidence = []
        for node, update in copilot.stream(query):
            if node == "plan":
                status.update(label="Planned. Retrieving evidence…")
                with plan_box:
                    render_plan(update["plan"])
            elif node == "execute":
                evidence = update["evidence"]
                status.update(label=f"Retrieved {len(evidence)} sections. Synthesizing + verifying…")
                with evidence_box:
                    render_evidence(evidence)
            elif node in ("verify", "refuse"):
                with answer_box:
                    render_answer(update["answer"])
                status.update(label="Done.", state="complete")


def main() -> None:
    st.title("🛢️ Gas & Energy Mechanics Copilot")
    st.caption(
        "Multi-agent **plan → execute → verify** RAG over 49 CFR Parts 192/193/195, "
        "PHMSA enforcement actions, and NTSB pipeline accident reports."
    )

    with st.sidebar:
        st.subheader("Example queries")
        st.caption("One-click — the last one demonstrates an out-of-corpus refusal.")
        for ex in EXAMPLE_QUERIES:
            if st.button(ex["label"], use_container_width=True):
                st.session_state["query"] = ex["query"]
        st.divider()
        st.caption("First query is slow (~30–60s cold start: warm-up + ~1 req/s Bedrock).")

    query = st.text_area("Ask a question", value=st.session_state.get("query", ""), height=90)
    if st.button("Submit", type="primary") and query.strip():
        run_query(query.strip())


if __name__ == "__main__":
    main()
else:
    main()
