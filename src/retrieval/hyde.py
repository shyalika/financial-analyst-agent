import os
from openai import OpenAI
from src.utils.billing import TokenBudgetController

SYSTEM_PROMPT = (
    "You are drafting a hypothetical passage for a financial research retrieval system. "
    "Given a user's research question, write a short (2-4 sentence) passage that plausibly "
    "answers it, in the confident, factual voice of a financial report or ASX announcement. "
    "Do not hedge, do not say you don't know, and do not add any meta-commentary about the "
    "question itself — just write the passage."
)

class HyDEGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.finops = TokenBudgetController()

    def generate_hypothetical_embedding(self, query: str, embedding_model: str = "text-embedding-3-small") -> tuple:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
        )

        self.finops.update_usage_and_verify(
            model="gpt-4o-mini",
            prompt=response.usage.prompt_tokens,
            completion=response.usage.completion_tokens
        )

        hypothetical_answer = (response.choices[0].message.content or "").strip()

        embed_response = self.client.embeddings.create(
            input=[hypothetical_answer.replace("\n", " ")],
            model=embedding_model
        )

        self.finops.update_usage_and_verify(
            model="gpt-4o-mini",
            prompt=embed_response.usage.prompt_tokens,
            completion=0
        )

        return hypothetical_answer, embed_response.data[0].embedding
