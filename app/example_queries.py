"""Example queries surfaced as one-click suggestions in the Streamlit UI (verbatim from spec).

The out-of-corpus refusal query is the most important one in the demo: the refusal must be
visible (planner decides out-of-scope), which is a trust signal.
"""

EXAMPLE_QUERIES = [
    {
        "label": "Single-section lookup",
        "query": "What does 49 CFR §192.505 require for strength testing of steel pipelines?",
        "demonstrates": "Simple retrieval; planner produces 1 sub-question.",
    },
    {
        "label": "Multi-part comparison",
        "query": (
            "Compare pressure-testing requirements between gas pipelines under "
            "49 CFR Part 192 Subpart J and hazardous-liquid pipelines under Part 195 Subpart E."
        ),
        "demonstrates": "Planner decomposes into per-Part sub-questions; cross-Part synthesis.",
    },
    {
        "label": "Adjudicative reasoning",
        "query": "What corrective actions did PHMSA require in recent enforcement actions for integrity-management violations?",
        "demonstrates": "Retrieval over enforcement actions; aggregation across multiple actions.",
    },
    {
        "label": "Narrative synthesis",
        "query": "What were the probable causes in recent NTSB pipeline accident investigations, and what 49 CFR sections did they cite?",
        "demonstrates": "Cross-source synthesis (NTSB ↔ regulation).",
    },
    {
        "label": "Three-source synthesis (hardest)",
        "query": (
            "For pipeline accidents where corrosion was identified as a contributing cause, "
            "what corrective actions did the operator commit to in the resulting PHMSA "
            "consent agreement, and how do those map to corrosion-control requirements in 49 CFR Part 192 Subpart I?"
        ),
        "demonstrates": "Three-source decomposition; the hardest case the planner handles.",
    },
    {
        "label": "Out-of-corpus refusal",
        "query": "What is the speed of light in vacuum?",
        "demonstrates": "Planner identifies no in-scope sub-questions; system refuses with explanation. Visible refusal is a trust signal.",
    },
]
