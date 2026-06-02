"""LangGraph state machine: PLAN -> (EXECUTE -> VERIFY | REFUSE).

A 3-node plan-execute-verify graph with a conditional refusal branch off the planner. Every
node is auto-traced by LangSmith when enabled, so the plan, each retrieval, and each claim
verdict are visible as a span tree.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents import executor, planner, verifier
from src.agents.schemas import Plan, VerifiedAnswer
from src.logging_setup import bind_correlation_id, clear_context, get_logger
from src.retrieval.parent_doc import Evidence
from src.retrieval.pipeline import Retriever

log = get_logger(__name__)


class GraphState(TypedDict, total=False):
    query: str
    plan: Plan
    evidence_by_sq: dict[str, list[Evidence]]
    evidence: list[Evidence]
    answer: VerifiedAnswer


def build_graph(retriever: Retriever):  # type: ignore[no-untyped-def]
    """Compile the plan-execute-verify graph bound to a retriever."""

    def plan_node(state: GraphState) -> dict:
        return {"plan": planner.plan(state["query"])}

    def execute_node(state: GraphState) -> dict:
        per_sq, flat = executor.execute(state["plan"], retriever)
        return {"evidence_by_sq": per_sq, "evidence": flat}

    def verify_node(state: GraphState) -> dict:
        return {"answer": verifier.verify(state["query"], state["evidence"])}

    def refuse_node(state: GraphState) -> dict:
        p = state["plan"]
        return {
            "answer": VerifiedAnswer(
                summary="",
                refused=True,
                refusal_reason=p.refusal_reason or "This question is outside the corpus.",
            )
        }

    def route(state: GraphState) -> str:
        p = state["plan"]
        return "execute" if (p.in_scope and p.sub_questions) else "refuse"

    sg = StateGraph(GraphState)
    sg.add_node("plan", plan_node)
    sg.add_node("execute", execute_node)
    sg.add_node("verify", verify_node)
    sg.add_node("refuse", refuse_node)
    sg.add_edge(START, "plan")
    sg.add_conditional_edges("plan", route, {"execute": "execute", "refuse": "refuse"})
    sg.add_edge("execute", "verify")
    sg.add_edge("verify", END)
    sg.add_edge("refuse", END)
    return sg.compile()


class Copilot:
    """Holds the retriever + compiled graph; one per process (cache in the app)."""

    def __init__(self, retriever: Optional[Retriever] = None) -> None:
        self.retriever = retriever or Retriever.open()
        self.graph = build_graph(self.retriever)

    def ask(self, query: str) -> GraphState:
        """Run the full graph for a query and return the final state."""
        cid = bind_correlation_id()
        log.info("copilot.ask", query=query[:120], correlation_id=cid)
        try:
            return self.graph.invoke({"query": query})  # type: ignore[return-value]
        finally:
            clear_context()
