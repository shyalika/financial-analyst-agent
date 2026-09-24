import os
from openai import OpenAI
from src.utils.billing import TokenBudgetController
from pydantic import BaseModel

SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer the user's question using "
    "ONLY the provided context chunks below — do not use outside knowledge and "
    "do not guess. Each chunk is labeled with its source document. Chunks that "
    "share the same source are fragments of one document — read those together "
    "as combined context (e.g. a reporting period established in one chunk "
    "applies to a figure in another chunk from the same source), and do not "
    "refuse to answer just because a single chunk lacks context that another "
    "chunk from the same source provides. Do NOT merge or conflate facts across "
    "chunks from different sources — treat those as independent unless the "
    "question is explicitly asking you to compare them. Cite specific figures "
    "from the context where relevant. If the answer is genuinely not present, "
    "say so honestly (e.g. 'The provided context does not contain this "
    "information') rather than fabricating one. "
    "Respond with a single JSON object with exactly two fields: \"answer\", a "
    "string with your answer, and \"parent_ids\", a JSON array of the integer "
    "parent_id values shown above that you actually relied on. Return valid "
    "JSON only, nothing else before or after it."
)

class Answer(BaseModel):
    answer: str
    parent_ids: list[int]

class AnswerGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.finops = TokenBudgetController()

    def generate_answer(self, question: str, chunks: list, model: str = "gpt-4o-mini") -> Answer:
        context = "\n\n".join(
            f"[parent_id: {c['parent_id']}  | Source: {c['source_file']}] {c['chunk_content']}"
            for i, c in enumerate(chunks)
        )
        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        response = self.client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )

        self.finops.update_usage_and_verify(
            model=model,
            prompt=response.usage.prompt_tokens,
            completion=response.usage.completion_tokens
        )

        return Answer.model_validate_json(response.choices[0].message.content)
