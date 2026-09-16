import os
import sys
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Fatal Security Exception: OPENAI_API_KEY configuration variable missing.")
    sys.exit(1)

from src.retrieval.hyde import HyDEGenerator

if __name__ == "__main__":
    try:
        generator = HyDEGenerator()
        sample_query = "What was CBA's net profit for the 2026 half year?"

        print(f"Original query: {sample_query}\n")
        hypothetical_answer, embedding_vector = generator.generate_hypothetical_embedding(sample_query)

        print(f"Hypothetical answer:\n{hypothetical_answer}\n")
        print(f"Embedding dimensionality: {len(embedding_vector)}")
        print(f"\nTotal Run Costs: ${generator.finops.usage.total_cost_usd:.6f} USD")
    except Exception as e:
        print(f"\n💥 Lifecycle Exception Triggered: {str(e)}")
