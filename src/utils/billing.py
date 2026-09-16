import os
import logging
from typing import Dict, Any
import tiktoken
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger("FinOpsController")

class TokenUsageSummary(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0

class TokenBudgetController:
    """
    Principal-grade FinOps harness for multi-agent loops.
    Tracks token burn-rate across states and implements runtime emergency stops.
    """
    PRICING_MATRIX: Dict[str, Dict[str, float]] = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "o3-mini": {"input": 1.10, "output": 4.40}
    }

    def __init__(self, default_model: str = "gpt-4o-mini"):
        self.default_model = default_model
        self.max_budget_usd = float(os.getenv("MAX_TOKEN_BUDGET_PER_RUN", 0.50))
        self.usage = TokenUsageSummary()
        self._encoder = tiktoken.get_encoding("cl100k_base")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def estimate_input_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def update_usage_and_verify(self, model: str, prompt: int, completion: int) -> float:
        rates = self.PRICING_MATRIX.get(model, self.PRICING_MATRIX[self.default_model])
        cost_input = (prompt / 1_000_000) * rates["input"]
        cost_output = (completion / 1_000_000) * rates["output"]
        iteration_cost = cost_input + cost_output

        self.usage.prompt_tokens += prompt
        self.usage.completion_tokens += completion
        self.usage.total_cost_usd += iteration_cost

        logger.info(
            f"Step Cost: ${iteration_cost:.5f} | "
            f"Cumulative Cost: ${self.usage.total_cost_usd:.5f} / ${self.max_budget_usd:.2f}"
        )

        if self.usage.total_cost_usd >= self.max_budget_usd:
            logger.critical(f"FinOps Boundary Breached! Runaway thread stopped at ${self.usage.total_cost_usd:.4f}")
            raise PermissionError(
                f"Execution terminated by FinOps Controller. Budget limit of ${self.max_budget_usd:.2f} reached."
            )
        return self.usage.total_cost_usd

    def execute_safely(self, messages: list, model: str = None, **kwargs) -> Dict[str, Any]:
        target_model = model or self.default_model
        if self.usage.total_cost_usd >= self.max_budget_usd:
            raise PermissionError("Aborting turn execution: Running budget allocation is fully depleted.")

        try:
            response = self.client.chat.completions.create(
                model=target_model, messages=messages, **kwargs
            )
            usage_meta = response.usage
            self.update_usage_and_verify(
                model=target_model,
                prompt=usage_meta.prompt_tokens,
                completion=usage_meta.completion_tokens
            )
            return {"content": response.choices.message.content, "raw_response": response}
        except Exception as e:
            logger.error(f"API Execution Failure or Boundary Intervention: {str(e)}")
            raise e
