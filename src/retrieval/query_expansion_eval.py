import os
from openai import OpenAI
from src.retrieval.query_expansion import QueryExpansionGenerator
from src.retrieval.similarity_search import SimilaritySearch
from src.retrieval.eval_questions import EVAL_QUESTIONS
from src.retrieval.embedding_utils import embed_text
from src.utils.billing import TokenBudgetController


def evaluate_hit_rate(top_k: int = 5) -> dict:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    baseline_finops = TokenBudgetController()
    expansion_generator = QueryExpansionGenerator()
    search = SimilaritySearch()

    per_question = []
    baseline_hits = 0
    multi_query_hits = 0

    for item in EVAL_QUESTIONS:
        question = item["question"]
        expected = item["expected_parent_ids"]

        baseline_embedding = embed_text(client, baseline_finops, question)
        baseline_results = search.search(baseline_embedding, top_k=top_k)
        baseline_hit = any(r["parent_id"] in expected for r in baseline_results)

        union_parent_ids = {r["parent_id"] for r in baseline_results}
        phrasings = expansion_generator.generate_alternative_phrasings(question)
        for phrasing in phrasings:
            variant_embedding = embed_text(client, expansion_generator.finops, phrasing)
            variant_results = search.search(variant_embedding, top_k=top_k)
            union_parent_ids.update(r["parent_id"] for r in variant_results)

        multi_query_hit = bool(union_parent_ids & expected)

        baseline_hits += int(baseline_hit)
        multi_query_hits += int(multi_query_hit)
        per_question.append({
            "question": question,
            "baseline_hit": baseline_hit,
            "multi_query_hit": multi_query_hit,
        })

    total = len(EVAL_QUESTIONS)
    return {
        "per_question": per_question,
        "baseline_hit_rate": baseline_hits / total,
        "multi_query_hit_rate": multi_query_hits / total,
        "total_cost_usd": baseline_finops.usage.total_cost_usd + expansion_generator.finops.usage.total_cost_usd,
    }
