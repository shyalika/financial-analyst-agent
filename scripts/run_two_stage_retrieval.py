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

SAMPLE_QUERY = "What was CBA's statutory net profit after tax for the 2026 half year?"

if __name__ == "__main__":
    try:
        retriever = TwoStageRetriever()
        result = retriever.retrieve(SAMPLE_QUERY, candidate_k=20, final_k=5)

        print(f"Query: {SAMPLE_QUERY}\n")
        print(f"Stage 1 — top {len(result['stage1_candidates'])} candidates by vector similarity:\n")
        for rank, r in enumerate(result["stage1_candidates"], start=1):
            snippet = r["chunk_content"][:70].replace("\n", " ")
            print(f"{rank:>2}. [{r['similarity']:.4f}] parent_id={r['parent_id']:>3}  {snippet}")

        print(f"\nStage 2 — top {len(result['reranked'])} after Cohere rerank:\n")
        for new_rank, r in enumerate(result["reranked"], start=1):
            snippet = r["chunk_content"][:70].replace("\n", " ")
            moved = "" if new_rank == r["stage1_rank"] else f"  (was rank {r['stage1_rank']})"
            print(f"{new_rank:>2}. [{r['relevance_score']:.4f}] parent_id={r['parent_id']:>3}  {snippet}{moved}")

        print(f"\nStage 1 latency: {result['stage1_latency_ms']:.1f} ms")
        print(f"Stage 2 latency: {result['stage2_latency_ms']:.1f} ms")
        print(f"Total run cost (OpenAI portion only): ${retriever.finops.usage.total_cost_usd:.6f} USD")
    except Exception as e:
        print(f"\n💥 Lifecycle Exception Triggered: {str(e)}")
