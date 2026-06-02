"""Hand-curated multi-hop evaluation questions (provided verbatim in the v2 contract).

These are designed to be answerable from the corpus, require planner decomposition
(multi-hop), and exercise the trace tree visibly. The agent is not expected to answer all 10
perfectly — visible decomposition + visible per-claim verification is the demo signal.
"""

HAND_CURATED_MULTI_HOP_QUESTIONS = [
    {
        "id": "MH-01",
        "question": (
            "Compare pressure-testing requirements between gas pipelines under "
            "49 CFR Part 192 Subpart J and hazardous-liquid pipelines under "
            "Part 195 Subpart E. Where do they differ in test duration, "
            "test medium, and minimum test pressure?"
        ),
        "tags": ["cross-part-comparison", "regulations-only"],
    },
    {
        "id": "MH-02",
        "question": (
            "What corrective actions did PHMSA require in recent enforcement "
            "actions related to pipeline integrity management, and which "
            "sections of 49 CFR Part 192 Subpart O were cited?"
        ),
        "tags": ["enforcement-to-regulation-join", "two-source"],
    },
    {
        "id": "MH-03",
        "question": (
            "For the NTSB pipeline accident reports in this corpus, what were "
            "the probable causes, and did any cite operator violations of "
            "49 CFR pressure-testing or pipeline-integrity requirements?"
        ),
        "tags": ["accident-to-regulation-join", "two-source"],
    },
    {
        "id": "MH-04",
        "question": (
            "How does 49 CFR Part 195 define 'high consequence area' (HCA), "
            "and what additional requirements apply to pipeline segments "
            "classified as HCA?"
        ),
        "tags": ["definition-plus-traversal", "single-source"],
    },
    {
        "id": "MH-05",
        "question": (
            "What civil-penalty amounts have been assessed in PHMSA enforcement "
            "actions for failures to comply with corrosion-control requirements "
            "(49 CFR §192.451 et seq. or §195.551 et seq.), and what were the "
            "most common findings?"
        ),
        "tags": ["enforcement-aggregation", "two-source"],
    },
    {
        "id": "MH-06",
        "question": (
            "What are the maximum allowable operating pressure (MAOP) "
            "requirements for gas pipelines under 49 CFR Part 192, and how "
            "does §192.619 (MAOP determination) interact with §192.620 "
            "(alternative MAOP)?"
        ),
        "tags": ["intra-part-interaction", "single-source"],
    },
    {
        "id": "MH-07",
        "question": (
            "For NTSB pipeline accident reports in this corpus that issued "
            "Safety Recommendations to PHMSA, what specific recommendations "
            "were made, and which 49 CFR sections did they address?"
        ),
        "tags": ["ntsb-to-regulation-traceback", "two-source"],
    },
    {
        "id": "MH-08",
        "question": (
            "Compare the operator-qualification requirements in 49 CFR Part "
            "192 Subpart N with those in Part 195 Subpart G. Where are they "
            "aligned and where do they diverge?"
        ),
        "tags": ["cross-part-comparison", "regulations-only"],
    },
    {
        "id": "MH-09",
        "question": (
            "For pipeline accidents where corrosion was identified as a "
            "contributing cause, what corrective actions did the operator "
            "commit to in the resulting PHMSA consent agreement, and how do "
            "those map to the corrosion-control requirements in 49 CFR Part "
            "192 Subpart I?"
        ),
        "tags": ["three-source-synthesis", "hardest"],
    },
    {
        "id": "MH-10",
        "question": (
            "What is the relationship between PHMSA's Integrity Management "
            "Program (IMP) requirements under §192.911 and the broader "
            "regulatory framework — which other sections cross-reference IMP, "
            "and what consequences follow from IMP-related findings in "
            "enforcement actions?"
        ),
        "tags": ["framework-plus-enforcement-consequences", "two-source"],
    },
]
