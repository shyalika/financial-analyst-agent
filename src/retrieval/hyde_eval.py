import os
from openai import OpenAI
from src.retrieval.hyde import HyDEGenerator
from src.retrieval.similarity_search import SimilaritySearch
from src.retrieval.eval_questions import EVAL_QUESTIONS
from src.retrieval.embedding_utils import embed_text
from src.utils.billing import TokenBudgetController


def evaluate_hit_rate(top_k: int = 5) -> dict:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    baseline_finops = TokenBudgetController()
    hyde_generator = HyDEGenerator()
    search = SimilaritySearch()

    per_question = []
    baseline_hits = 0
    hyde_hits = 0

    for item in EVAL_QUESTIONS:
        question = item["question"]
        expected = item["expected_parent_ids"]

        baseline_embedding = embed_text(client, baseline_finops, question)
        baseline_results = search.search(baseline_embedding, top_k=top_k)
        baseline_hit = any(r["parent_id"] in expected for r in baseline_results)

        _, hyde_embedding = hyde_generator.generate_hypothetical_embedding(question)
        hyde_results = search.search(hyde_embedding, top_k=top_k)
        hyde_hit = any(r["parent_id"] in expected for r in hyde_results)

        baseline_hits += int(baseline_hit)
        hyde_hits += int(hyde_hit)
        per_question.append({
            "question": question,
            "baseline_hit": baseline_hit,
            "hyde_hit": hyde_hit,
        })

    total = len(EVAL_QUESTIONS)
    return {
        "per_question": per_question,
        "baseline_hit_rate": baseline_hits / total,
        "hyde_hit_rate": hyde_hits / total,
        "total_cost_usd": baseline_finops.usage.total_cost_usd + hyde_generator.finops.usage.total_cost_usd,
    }
