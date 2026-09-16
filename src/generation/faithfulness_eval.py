from deepeval.metrics import FaithfulnessMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.llms.openai_model import OpenAIModel
from src.retrieval.two_stage_retriever import TwoStageRetriever
from src.generation.answer_generator import AnswerGenerator
from src.retrieval.eval_questions_rerank import EVAL_QUESTIONS

# Caps the judge's completion length well below either model's max output.
# Without this, a runaway generation (model never emits valid structured JSON)
# burns the full budget before failing — slow and expensive. A normal verdict
# response for a handful of claims needs a few hundred tokens at most, so 1000
# is a safety net, not a limit that should ever bind on legitimate output.
JUDGE_MAX_COMPLETION_TOKENS = 1000

# Faithfulness's gpt-4o-mini judge both failed to complete 47% of the time
# (runaway generation) and, when it did complete, flagged several factually
# correct answers as unfaithful over date-phrasing pedantry ("2026" vs "half
# year ended 31 December 2025", "June 2025" vs "Jun 25") — see SPEC.md §17.
# Testing whether a more capable judge fixes both. Context Precision had zero
# failures and sensible reasoning on gpt-4o-mini, so it stays as-is — only
# the metric that was actually struggling gets the stronger (costlier) judge.
FAITHFULNESS_JUDGE_MODEL = "gpt-4o"
CONTEXT_PRECISION_JUDGE_MODEL = "gpt-4o-mini"


def _make_judge_model(model: str) -> OpenAIModel:
    return OpenAIModel(
        model=model,
        generation_kwargs={"max_completion_tokens": JUDGE_MAX_COMPLETION_TOKENS},
    )


def evaluate_faithfulness_and_precision(top_k: int = 5) -> dict:
    retriever = TwoStageRetriever()
    generator = AnswerGenerator()
    faithfulness_metric = FaithfulnessMetric(model=_make_judge_model(FAITHFULNESS_JUDGE_MODEL), include_reason=True)
    precision_metric = ContextualPrecisionMetric(model=_make_judge_model(CONTEXT_PRECISION_JUDGE_MODEL), include_reason=True)

    per_question = []

    for item in EVAL_QUESTIONS:
        question = item["question"]
        expected_answer = item["expected_answer"]

        result = retriever.retrieve(question, candidate_k=20, final_k=top_k)
        chunks = result["reranked"]
        answer = generator.generate_answer(question, chunks)
        retrieval_context = [c["chunk_content"] for c in chunks]

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_answer,
            retrieval_context=retrieval_context,
        )

        # Individual DeepEval judge calls can occasionally hit a runaway
        # generation failure (gpt-4o-mini exhausts its completion-token budget
        # without producing valid structured output — see SPEC.md §17). Catch
        # per-question so one flaky judge call doesn't drop the whole batch.
        record = {"question": question, "answer": answer, "expected_answer": expected_answer}

        try:
            faithfulness_metric.measure(test_case)
            record["faithfulness_score"] = faithfulness_metric.score
            record["faithfulness_reason"] = faithfulness_metric.reason
        except Exception as e:
            record["faithfulness_score"] = None
            record["faithfulness_reason"] = f"FAILED: {e}"

        try:
            precision_metric.measure(test_case)
            record["context_precision_score"] = precision_metric.score
            record["context_precision_reason"] = precision_metric.reason
        except Exception as e:
            record["context_precision_score"] = None
            record["context_precision_reason"] = f"FAILED: {e}"

        per_question.append(record)

    faithfulness_scores = [q["faithfulness_score"] for q in per_question if q["faithfulness_score"] is not None]
    precision_scores = [q["context_precision_score"] for q in per_question if q["context_precision_score"] is not None]

    return {
        "per_question": per_question,
        "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None,
        "avg_context_precision": sum(precision_scores) / len(precision_scores) if precision_scores else None,
        "faithfulness_failures": len(per_question) - len(faithfulness_scores),
        "context_precision_failures": len(per_question) - len(precision_scores),
        "total_cost_usd": retriever.finops.usage.total_cost_usd + generator.finops.usage.total_cost_usd,
    }
