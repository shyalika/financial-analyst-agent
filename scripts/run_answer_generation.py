import os
import sys
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Fatal Security Exception: OPENAI_API_KEY configuration variable missing.")
    sys.exit(1)

from src.retrieval.two_stage_retriever import TwoStageRetriever
from src.generation.answer_generator import AnswerGenerator

# Case 1: a real question, answerable from the document.
ANSWERABLE_QUERY = "What was CBA's statutory net profit after tax for the 2026 half year?"

# Case 2: a plausible-sounding question the document does not actually cover —
# tests whether the generator says so honestly instead of fabricating an answer.
UNANSWERABLE_QUERY = "What was CBA's marketing budget for 1H26?"

if __name__ == "__main__":
    try:
        retriever = TwoStageRetriever()
        generator = AnswerGenerator()

        for label, query in [("ANSWERABLE", ANSWERABLE_QUERY), ("UNANSWERABLE", UNANSWERABLE_QUERY)]:
            print(f"\n=== {label} ===")
            print(f"Question: {query}\n")

            result = retriever.retrieve(query, candidate_k=20, final_k=5)
            chunks = result["reranked"]

            print("Retrieved chunks (post-rerank):")
            for r in chunks:
                snippet = r["chunk_content"][:70].replace("\n", " ")
                print(f" - [{r['relevance_score']:.4f}] parent_id={r['parent_id']:>3}  {snippet}")

            answer = generator.generate_answer(query, chunks)
            print(f"\nGenerated answer:\n{answer}")

        total_cost = retriever.finops.usage.total_cost_usd + generator.finops.usage.total_cost_usd
        print(f"\nTotal Run Costs: ${total_cost:.6f} USD")
    except Exception as e:
        print(f"\n💥 Lifecycle Exception Triggered: {str(e)}")
