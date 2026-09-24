from unittest.mock import MagicMock

from src.generation import faithfulness_eval as faithfulness_eval_module
from src.generation.faithfulness_eval import evaluate_faithfulness_and_precision


class FakeMetric:
    """A DeepEval metric double: .measure(test_case) mutates .score/.reason
    (like the real metric does), or raises, per a pre-scripted sequence --
    one entry consumed per question, in question order."""
    def __init__(self, script):
        self._script = list(script)
        self.score = None
        self.reason = None

    def __call__(self, model=None, include_reason=True):
        # Metrics are constructed once (FaithfulnessMetric(...)) then reused
        # across the whole per-question loop -- so the "constructor" call
        # just returns this same pre-scripted double.
        return self

    def measure(self, test_case):
        outcome = self._script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        self.score, self.reason = outcome


def make_fake_retriever(chunks_by_question):
    retriever = MagicMock()
    retriever.finops.usage.total_cost_usd = 0.01

    def retrieve(question, candidate_k, final_k):
        return {"reranked": chunks_by_question[question]}

    retriever.retrieve.side_effect = retrieve
    return retriever


def make_fake_generator(answer_by_question):
    generator = MagicMock()
    generator.finops.usage.total_cost_usd = 0.02
    generator.generate_answer.side_effect = lambda question, chunks: answer_by_question[question]
    return generator


def setup(monkeypatch, questions, faithfulness_script, precision_script):
    monkeypatch.setattr(faithfulness_eval_module, "EVAL_QUESTIONS", questions)
    monkeypatch.setattr(faithfulness_eval_module, "_make_judge_model", lambda model: MagicMock())

    chunks_by_question = {
        q["question"]: [{"chunk_content": "context chunk", "source_file": "1H26.pdf"}] for q in questions
    }
    answer_by_question = {q["question"]: f"answer to: {q['question']}" for q in questions}

    monkeypatch.setattr(
        faithfulness_eval_module, "TwoStageRetriever",
        lambda: make_fake_retriever(chunks_by_question)
    )
    monkeypatch.setattr(
        faithfulness_eval_module, "AnswerGenerator",
        lambda: make_fake_generator(answer_by_question)
    )
    monkeypatch.setattr(faithfulness_eval_module, "FaithfulnessMetric", FakeMetric(faithfulness_script))
    monkeypatch.setattr(faithfulness_eval_module, "ContextualPrecisionMetric", FakeMetric(precision_script))


TWO_QUESTIONS = [
    {"question": "What was NPAT?", "expected_answer": "$5,412 million"},
    {"question": "What was ROE?", "expected_answer": "13.8%"},
]


def test_all_questions_succeed_computes_correct_average(monkeypatch):
    setup(
        monkeypatch, TWO_QUESTIONS,
        faithfulness_script=[(0.9, "faithful"), (0.8, "faithful")],
        precision_script=[(1.0, "precise"), (0.7, "precise")],
    )

    result = evaluate_faithfulness_and_precision(top_k=5)

    assert result["avg_faithfulness"] == (0.9 + 0.8) / 2
    assert result["avg_context_precision"] == (1.0 + 0.7) / 2
    assert result["faithfulness_failures"] == 0
    assert result["context_precision_failures"] == 0
    assert len(result["per_question"]) == 2


def test_one_failing_metric_does_not_drop_the_question(monkeypatch):
    # First question's faithfulness judge call raises (a runaway-generation
    # style failure); the second question must still be processed normally,
    # and the failure must be excluded from the average, not counted as 0.
    setup(
        monkeypatch, TWO_QUESTIONS,
        faithfulness_script=[RuntimeError("judge exhausted completion budget"), (0.8, "faithful")],
        precision_script=[(1.0, "precise"), (0.9, "precise")],
    )

    result = evaluate_faithfulness_and_precision(top_k=5)

    assert len(result["per_question"]) == 2
    assert result["per_question"][0]["faithfulness_score"] is None
    assert "FAILED:" in result["per_question"][0]["faithfulness_reason"]
    assert result["per_question"][1]["faithfulness_score"] == 0.8

    assert result["faithfulness_failures"] == 1
    assert result["avg_faithfulness"] == 0.8  # only the surviving score, not (0 + 0.8) / 2

    assert result["context_precision_failures"] == 0
    assert result["avg_context_precision"] == (1.0 + 0.9) / 2


def test_all_metrics_failing_reports_none_average(monkeypatch):
    setup(
        monkeypatch, TWO_QUESTIONS,
        faithfulness_script=[RuntimeError("boom"), RuntimeError("boom again")],
        precision_script=[(1.0, "precise"), (0.9, "precise")],
    )

    result = evaluate_faithfulness_and_precision(top_k=5)

    assert result["avg_faithfulness"] is None
    assert result["faithfulness_failures"] == 2


def test_records_include_question_answer_and_expected_answer(monkeypatch):
    setup(
        monkeypatch, TWO_QUESTIONS,
        faithfulness_script=[(0.9, "r"), (0.8, "r")],
        precision_script=[(1.0, "r"), (0.9, "r")],
    )

    result = evaluate_faithfulness_and_precision(top_k=5)

    record = result["per_question"][0]
    assert record["question"] == "What was NPAT?"
    assert record["expected_answer"] == "$5,412 million"
    assert record["answer"] == "answer to: What was NPAT?"


def test_total_cost_sums_retriever_and_generator_finops(monkeypatch):
    setup(
        monkeypatch, TWO_QUESTIONS,
        faithfulness_script=[(0.9, "r"), (0.8, "r")],
        precision_script=[(1.0, "r"), (0.9, "r")],
    )

    result = evaluate_faithfulness_and_precision(top_k=5)

    assert result["total_cost_usd"] == 0.03
