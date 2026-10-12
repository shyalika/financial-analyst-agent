import os
import sys
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Fatal Security Exception: OPENAI_API_KEY configuration variable missing.")
    sys.exit(1)

from src.agents.agent_loop import AgentLoop

# Scenario 1: specific enough to search for -- the model should call
# retrieve_disclosures and relay a grounded answer.
CLEAR_QUERY = "What was CBA's statutory net profit after tax for the 2026 half year?"

# Scenario 2: missing the detail needed to search (no period, no entity,
# not even which profit figure) -- the model should ask a clarifying
# question instead of calling the tool. The follow-up then answers that
# clarification, resuming the same conversation via its reference_id.
AMBIGUOUS_QUERY = "What was the profit?"
CLARIFYING_REPLY = "I mean CBA's statutory NPAT for the 2026 half year."


def print_total_cost(loop: AgentLoop):
    total_cost_usd = (
        loop.finops.usage.total_cost_usd
        + loop.tool.retriever.finops.usage.total_cost_usd
        + loop.tool.generator.finops.usage.total_cost_usd
    )
    print(f"Total run cost so far (OpenAI portion only, excludes Cohere rerank): ${total_cost_usd:.6f} USD")


if __name__ == "__main__":
    try:
        loop = AgentLoop()

        print("=== Scenario 1: clear question -- should retrieve and answer ===\n")
        answer, reference_id = loop.run_loop(CLEAR_QUERY)
        print(f"Query: {CLEAR_QUERY}")
        print(f"Answer: {answer}")
        print(f"Reference ID: {reference_id}\n")

        print("=== Scenario 2: ambiguous question -- should ask a clarifying question ===\n")
        answer, reference_id = loop.run_loop(AMBIGUOUS_QUERY)
        print(f"Query: {AMBIGUOUS_QUERY}")
        print(f"Answer: {answer}")
        print(f"Reference ID: {reference_id}\n")

        print("=== Resuming Scenario 2 with the clarifying reply ===\n")
        answer, reference_id = loop.run_loop(CLARIFYING_REPLY, reference_id=reference_id)
        print(f"Follow-up: {CLARIFYING_REPLY}")
        print(f"Answer: {answer}")
        print(f"Reference ID: {reference_id}\n")

        print("========================\n")
        print_total_cost(loop)
    except Exception as e:
        print(f"\n💥 Lifecycle Exception Triggered: {str(e)}")
