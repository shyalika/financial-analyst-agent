import os
import sys
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Fatal Security Exception: OPENAI_API_KEY configuration variable missing.")
    sys.exit(1)

from src.retrieval.query_expansion import QueryExpansionGenerator
from src.retrieval.hyde import HyDEGenerator
from src.retrieval.similarity_search import SimilaritySearch
from src.utils.billing import TokenBudgetController

SAMPLE_QUERY = "What was CBA's net profit for the 2026 half year?"


def embed_text(client, finops, text: str) -> list:
    response = client.embeddings.create(
        input=[text.replace("\n", " ")],
        model="text-embedding-3-small"
    )
    finops.update_usage_and_verify(
        model="gpt-4o-mini",
        prompt=response.usage.prompt_tokens,
        completion=0
    )
    return response.data[0].embedding


def print_results(label: str, results: list):
    print(f"\n--- {label} ---")
    for rank, result in enumerate(results, start=1):
        snippet = result["chunk_content"][:80].replace("\n", " ")
        print(f" #{rank} [{result['similarity']:.4f}] ({result['source_file']} / {result['section_title']}) {snippet}...")


if __name__ == "__main__":
    try:
        search = SimilaritySearch()

        # --- Baseline: embed the raw query directly ---
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        baseline_finops = TokenBudgetController()

        baseline_embedding = embed_text(client, baseline_finops, SAMPLE_QUERY)
        print_results(f"Baseline: '{SAMPLE_QUERY}'", search.search(baseline_embedding, top_k=5))

        # --- Query expansion: embed each generated phrasing ---
        expansion_generator = QueryExpansionGenerator()
        phrasings = expansion_generator.generate_alternative_phrasings(SAMPLE_QUERY)
        for phrasing in phrasings:
            phrasing_embedding = embed_text(client, expansion_generator.finops, phrasing)
            print_results(f"Query Expansion variant: '{phrasing}'", search.search(phrasing_embedding, top_k=5))

        # --- HyDE: already returns an embedding directly ---
        hyde_generator = HyDEGenerator()
        hypothetical_answer, hyde_embedding = hyde_generator.generate_hypothetical_embedding(SAMPLE_QUERY)
        print_results(f"HyDE: '{hypothetical_answer[:60]}...'", search.search(hyde_embedding, top_k=5))

        total_cost = (
            baseline_finops.usage.total_cost_usd
            + expansion_generator.finops.usage.total_cost_usd
            + hyde_generator.finops.usage.total_cost_usd
        )
        print(f"\nTotal Run Costs: ${total_cost:.6f} USD")
    except Exception as e:
        print(f"\n💥 Lifecycle Exception Triggered: {str(e)}")
