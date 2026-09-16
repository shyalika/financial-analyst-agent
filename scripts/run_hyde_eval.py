import os
import sys
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Fatal Security Exception: OPENAI_API_KEY configuration variable missing.")
    sys.exit(1)

from src.retrieval.hyde_eval import evaluate_hit_rate

if __name__ == "__main__":
    try:
        report = evaluate_hit_rate(top_k=5)

        print(f"{'#':<3} {'Baseline':<10} {'HyDE':<10} Question")
        for idx, item in enumerate(report["per_question"], start=1):
            baseline_mark = "HIT" if item["baseline_hit"] else "miss"
            hyde_mark = "HIT" if item["hyde_hit"] else "miss"
            print(f"{idx:<3} {baseline_mark:<10} {hyde_mark:<10} {item['question']}")

        print("\n--- Summary (hit@5) ---")
        print(f"Baseline hit rate: {report['baseline_hit_rate']:.0%}")
        print(f"HyDE hit rate:     {report['hyde_hit_rate']:.0%}")
        print(f"\nTotal Run Costs: ${report['total_cost_usd']:.6f} USD")
    except Exception as e:
        print(f"\n💥 Lifecycle Exception Triggered: {str(e)}")
