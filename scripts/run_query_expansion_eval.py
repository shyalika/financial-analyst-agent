import os
import sys
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Fatal Security Exception: OPENAI_API_KEY configuration variable missing.")
    sys.exit(1)

from src.retrieval.query_expansion_eval import evaluate_hit_rate


def print_report(top_k: int, report: dict):
    print(f"\n=== hit@{top_k} ===")
    print(f"{'#':<3} {'Baseline':<10} {'Multi-Query':<12} Question")
    for idx, item in enumerate(report["per_question"], start=1):
        baseline_mark = "HIT" if item["baseline_hit"] else "miss"
        multi_mark = "HIT" if item["multi_query_hit"] else "miss"
        print(f"{idx:<3} {baseline_mark:<10} {multi_mark:<12} {item['question']}")

    print(f"\nBaseline hit rate:    {report['baseline_hit_rate']:.0%}")
    print(f"Multi-query hit rate: {report['multi_query_hit_rate']:.0%}")
    print(f"Run cost: ${report['total_cost_usd']:.6f} USD")


if __name__ == "__main__":
    try:
        total_cost = 0.0
        for top_k in (5, 1):
            report = evaluate_hit_rate(top_k=top_k)
            print_report(top_k, report)
            total_cost += report["total_cost_usd"]

        print(f"\nTotal Run Costs: ${total_cost:.6f} USD")
    except Exception as e:
        print(f"\n💥 Lifecycle Exception Triggered: {str(e)}")
