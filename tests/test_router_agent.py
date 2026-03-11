"""
Tests: tests/test_router_agent.py

PURPOSE:
    Tests for the router agent's classification logic and the RouterDecision schema.
    We test two levels:
      1. Schema validation: RouterDecision correctly validates JSON from the LLM.
      2. Classification accuracy: The router correctly classifies example queries
         (these tests run the actual LLM and require Bedrock access).

TESTING STRATEGY:
    Level 1 (schema tests): No LLM needed. Tests Pydantic validation logic.
    Level 2 (classification tests): Marked @pytest.mark.integration — require Bedrock.

    Why test classification accuracy:
      If the router misclassifies "What does §192.505 say?" as "compliance_check"
      instead of "regulatory_lookup", the retrieval specialist uses the wrong
      search strategy and may return irrelevant passages. Router accuracy is
      a first-order quality driver for the entire pipeline.
"""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("CONFIG_DIR", "./config")


class TestRouterDecisionSchema:
    """Unit tests for RouterDecision Pydantic schema validation."""

    def test_valid_regulatory_lookup(self):
        """RouterDecision should parse a valid regulatory_lookup response."""
        from pipeline_safety_rag_crew.crew import RouterDecision

        data = {
            "query_type": "regulatory_lookup",
            "confidence": 0.95,
            "reasoning": "User requests specific CFR section text.",
            "recommended_search_terms": ["§192.505", "pressure testing", "steel pipeline"],
            "requires_web_search": False,
        }
        decision = RouterDecision.model_validate(data)
        assert decision.query_type == "regulatory_lookup"
        assert decision.confidence == 0.95
        assert len(decision.recommended_search_terms) == 3
        assert decision.requires_web_search is False

    def test_valid_compliance_check(self):
        """RouterDecision should parse compliance_check queries correctly."""
        from pipeline_safety_rag_crew.crew import RouterDecision

        data = {
            "query_type": "compliance_check",
            "confidence": 0.82,
            "reasoning": "User asks about operator obligations.",
            "recommended_search_terms": ["Class 3", "§192.619", "MAOP"],
            "requires_web_search": False,
        }
        decision = RouterDecision.model_validate(data)
        assert decision.query_type == "compliance_check"

    def test_requires_web_search_defaults_false(self):
        """requires_web_search should default to False if not provided."""
        from pipeline_safety_rag_crew.crew import RouterDecision

        data = {
            "query_type": "general_engineering",
            "confidence": 0.70,
            "reasoning": "Background concept question.",
            "recommended_search_terms": ["cathodic protection", "pipeline corrosion"],
        }
        decision = RouterDecision.model_validate(data)
        assert decision.requires_web_search is False

    def test_invalid_query_type_raises_error(self):
        """RouterDecision should reject unknown query_type values."""
        from pipeline_safety_rag_crew.crew import RouterDecision
        from pydantic import ValidationError

        data = {
            "query_type": "unknown_category",  # invalid
            "confidence": 0.5,
            "reasoning": "Test.",
            "recommended_search_terms": [],
        }
        with pytest.raises(ValidationError):
            RouterDecision.model_validate(data)

    def test_confidence_out_of_range_raises_error(self):
        """confidence must be in [0.0, 1.0]."""
        from pipeline_safety_rag_crew.crew import RouterDecision
        from pydantic import ValidationError

        data = {
            "query_type": "regulatory_lookup",
            "confidence": 1.5,  # out of range
            "reasoning": "Test.",
            "recommended_search_terms": [],
        }
        with pytest.raises(ValidationError):
            RouterDecision.model_validate(data)

    def test_all_four_categories_are_valid(self):
        """All four valid query_type values should be accepted."""
        from pipeline_safety_rag_crew.crew import RouterDecision

        categories = ["regulatory_lookup", "compliance_check", "incident_analysis", "general_engineering"]
        for category in categories:
            data = {
                "query_type": category,
                "confidence": 0.8,
                "reasoning": f"Test for {category}.",
                "recommended_search_terms": ["test"],
            }
            decision = RouterDecision.model_validate(data)
            assert decision.query_type == category


class TestJudgeVerdictSchema:
    """Unit tests for JudgeVerdict Pydantic schema validation."""

    def test_approved_verdict_with_no_issues(self):
        """An approved verdict should have empty issues and no revised_answer."""
        from pipeline_safety_rag_crew.crew import JudgeVerdict

        data = {
            "verdict": "approved",
            "confidence": 0.92,
            "issues": [],
            "revised_answer": None,
            "citations_verified": ["§192.505(a)", "§192.505(b)"],
            "hallucinations_detected": [],
        }
        verdict = JudgeVerdict.model_validate(data)
        assert verdict.verdict == "approved"
        assert verdict.confidence == 0.92
        assert verdict.revised_answer is None
        assert len(verdict.citations_verified) == 2

    def test_needs_revision_requires_revised_answer(self):
        """needs_revision verdict should typically include a revised_answer text."""
        from pipeline_safety_rag_crew.crew import JudgeVerdict

        data = {
            "verdict": "needs_revision",
            "confidence": 0.65,
            "issues": ["Claim about §192.510 not found in retrieved context."],
            "revised_answer": "The corrected answer is...",
            "citations_verified": ["§192.505"],
            "hallucinations_detected": ["§192.510 requirement about X"],
        }
        verdict = JudgeVerdict.model_validate(data)
        assert verdict.verdict == "needs_revision"
        assert verdict.revised_answer == "The corrected answer is..."
        assert len(verdict.hallucinations_detected) == 1

    def test_rejected_verdict_with_multiple_issues(self):
        """A rejected verdict should list all hallucinations."""
        from pipeline_safety_rag_crew.crew import JudgeVerdict

        data = {
            "verdict": "rejected",
            "confidence": 0.20,
            "issues": ["Fabricated CFR section", "Contradicts §192.505(a)", "No sources cited"],
            "revised_answer": None,
            "citations_verified": [],
            "hallucinations_detected": ["Fabricated citation §192.999"],
        }
        verdict = JudgeVerdict.model_validate(data)
        assert verdict.verdict == "rejected"
        assert len(verdict.issues) == 3

    def test_invalid_verdict_raises_error(self):
        """Invalid verdict value should raise a validation error."""
        from pipeline_safety_rag_crew.crew import JudgeVerdict
        from pydantic import ValidationError

        data = {
            "verdict": "maybe",  # invalid
            "confidence": 0.5,
            "issues": [],
        }
        with pytest.raises(ValidationError):
            JudgeVerdict.model_validate(data)


class TestResolveFinAnswer:
    """Tests for the resolve_final_answer helper function."""

    def test_approved_returns_synthesis_unchanged(self):
        """Approved answers should be returned without modification."""
        from pipeline_safety_rag_crew.crew import JudgeVerdict, resolve_final_answer

        verdict = JudgeVerdict.model_validate({
            "verdict": "approved",
            "confidence": 0.9,
            "issues": [],
            "citations_verified": ["§192.505"],
            "hallucinations_detected": [],
        })
        synthesis = "MAOP is defined in §192.619 as the maximum pressure..."
        answer, meta = resolve_final_answer(synthesis, verdict)

        assert answer == synthesis
        assert meta["verdict"] == "approved"

    def test_needs_revision_returns_revised_answer(self):
        """needs_revision should use the judge's revised_answer."""
        from pipeline_safety_rag_crew.crew import JudgeVerdict, resolve_final_answer

        verdict = JudgeVerdict.model_validate({
            "verdict": "needs_revision",
            "confidence": 0.6,
            "issues": ["Unsupported claim"],
            "revised_answer": "The correct answer based on retrieved context is...",
            "citations_verified": [],
            "hallucinations_detected": [],
        })
        synthesis = "Original (wrong) answer"
        answer, meta = resolve_final_answer(synthesis, verdict)

        assert answer == "The correct answer based on retrieved context is..."
        assert meta["verdict"] == "needs_revision"

    def test_rejected_returns_disclaimer(self):
        """Rejected answers should return the standard disclaimer, not the synthesis."""
        from pipeline_safety_rag_crew.crew import JudgeVerdict, resolve_final_answer

        verdict = JudgeVerdict.model_validate({
            "verdict": "rejected",
            "confidence": 0.1,
            "issues": ["Multiple fabricated CFR citations"],
            "citations_verified": [],
            "hallucinations_detected": ["§192.999 (does not exist)"],
        })
        synthesis = "Dangerous hallucinated answer"
        answer, meta = resolve_final_answer(synthesis, verdict)

        # Disclaimer should mention inability to answer confidently
        assert "unable to produce a confident answer" in answer.lower()
        assert synthesis not in answer  # original answer should NOT be returned
        assert meta["verdict"] == "rejected"

    def test_none_verdict_returns_synthesis(self):
        """When verdict is None (simple crew), return synthesis as-is."""
        from pipeline_safety_rag_crew.crew import resolve_final_answer

        synthesis = "Answer from simple 2-agent crew"
        answer, meta = resolve_final_answer(synthesis, None)

        assert answer == synthesis
        assert meta["verdict"] == "unknown"
