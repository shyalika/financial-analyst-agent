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

if __name__ == "__main__":
    try:
        generator = QueryExpansionGenerator()
        sample_query = "What was CBA's net profit for the 2026 half year?"

        print(f"Original query: {sample_query}\n")
        phrasings = generator.generate_alternative_phrasings(sample_query)

        print("Alternative phrasings:")
        for phrasing in phrasings:
            print(f" -> {phrasing}")

        print(f"\nTotal Run Costs: ${generator.finops.usage.total_cost_usd:.6f} USD")
    except Exception as e:
        print(f"\n💥 Lifecycle Exception Triggered: {str(e)}")
