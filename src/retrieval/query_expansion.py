import os
from openai import OpenAI
from src.utils.billing import TokenBudgetController

SYSTEM_PROMPT = (
    "You are a query rewriting assistant for a financial research retrieval system. "
    "Given a user's research question, produce alternative phrasings of it that preserve "
    "the original intent, entities, and numbers, while varying the wording and sentence "
    "structure. Respond with exactly one phrasing per line. Do not number the lines, and "
    "do not add any commentary, explanation, or the original question."
)

class QueryExpansionGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.finops = TokenBudgetController()

    def generate_alternative_phrasings(self, query: str, num_variants: int = 3) -> list:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate {num_variants} alternative phrasings of this question:\n{query}"}
            ]
        )

        self.finops.update_usage_and_verify(
            model="gpt-4o-mini",
            prompt=response.usage.prompt_tokens,
            completion=response.usage.completion_tokens
        )

        content = response.choices[0].message.content or ""
        phrasings = []
        for line in content.splitlines():
            cleaned = line.strip().lstrip("-*0123456789.) ").strip()
            if cleaned:
                phrasings.append(cleaned)

        return phrasings[:num_variants]
