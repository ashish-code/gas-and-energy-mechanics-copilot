"""Run the agent over a question set and score it with RAGAS.

For each question we run the full plan-execute-verify graph, then assemble RAGAS samples
(user_input, response, retrieved_contexts, [reference]). The 40-set (with references) gets the
full quartet (faithfulness, answer relevancy, context precision, context recall); the
hand-curated 10-set (no references) gets faithfulness + answer relevancy. Defaults only — no tuning.
"""

from __future__ import annotations

from src.agents.graph import Copilot
from src.logging_setup import get_logger

# NOTE: ragas (and src.eval._bedrock, which imports it) are imported lazily inside
# evaluate_samples — NOT at module top. Importing ragas in the same process that then runs
# the agent pipeline causes the retrieval phase to hang (ragas' import side effects perturb
# the run). Keeping the import after all agent runs (build_samples) avoids the interaction.

log = get_logger(__name__)


def build_samples(copilot: Copilot, questions: list[dict]) -> list[dict]:
    """Run the agent over questions -> RAGAS sample dicts. `questions` items: {question, reference?}."""
    samples: list[dict] = []
    for i, q in enumerate(questions, 1):
        state = copilot.ask(q["question"])
        answer = state.get("answer")
        evidence = state.get("evidence", []) or []
        if answer and not answer.refused:
            response = answer.summary or ""
        else:
            response = (answer.refusal_reason if answer else "") or ""
        sample = {
            "user_input": q["question"],
            "response": response,
            "retrieved_contexts": [ev.text for ev in evidence] or ["(no evidence retrieved)"],
        }
        if q.get("reference"):
            sample["reference"] = q["reference"]
        samples.append(sample)
        log.info("eval.sample", done=i, total=len(questions), contexts=len(sample["retrieved_contexts"]))
    return samples


def evaluate_samples(samples: list[dict], *, with_reference: bool) -> dict[str, float]:
    """Compute RAGAS metrics over the samples."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, ResponseRelevancy

    from src.eval._bedrock import ragas_embeddings, ragas_llm

    metrics = [Faithfulness(), ResponseRelevancy()]
    if with_reference:
        from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

        metrics += [LLMContextPrecisionWithReference(), LLMContextRecall()]

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(dataset, metrics=metrics, llm=ragas_llm(), embeddings=ragas_embeddings())
    df = result.to_pandas()
    scores: dict[str, float] = {}
    for col in df.columns:
        if col in ("user_input", "response", "retrieved_contexts", "reference", "reference_contexts"):
            continue
        try:
            scores[col] = float(df[col].mean(skipna=True))
        except (TypeError, ValueError):
            continue
    log.info("eval.scores", with_reference=with_reference, **{k: round(v, 3) for k, v in scores.items()})
    return scores
