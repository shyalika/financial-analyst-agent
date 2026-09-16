import os
import sys
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Fatal Security Exception: OPENAI_API_KEY configuration variable missing.")
    sys.exit(1)

from src.generation.faithfulness_eval import evaluate_faithfulness_and_precision

if __name__ == "__main__":
    try:
        report = evaluate_faithfulness_and_precision(top_k=5)

        def fmt(score):
            return f"{score:.2f}" if score is not None else "FAILED"

        for idx, item in enumerate(report["per_question"], start=1):
            print(f"\n--- Q{idx}: {item['question']} ---")
            print(f"Expected: {item['expected_answer']}")
            print(f"Actual:   {item['answer']}")
            print(f"Faithfulness: {fmt(item['faithfulness_score'])}  ({item['faithfulness_reason']})")
            print(f"Context Precision: {fmt(item['context_precision_score'])}  ({item['context_precision_reason']})")

        print("\n=== Summary ===")
        print(f"Avg Faithfulness:       {fmt(report['avg_faithfulness'])}  ({report['faithfulness_failures']} judge failures out of {len(report['per_question'])})")
        print(f"Avg Context Precision:  {fmt(report['avg_context_precision'])}  ({report['context_precision_failures']} judge failures out of {len(report['per_question'])})")
        print(f"\nTotal Run Costs (OpenAI generation/retrieval portion only): ${report['total_cost_usd']:.6f} USD")
    except Exception as e:
        print(f"\n💥 Lifecycle Exception Triggered: {str(e)}")
