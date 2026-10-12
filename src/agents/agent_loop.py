import os
import uuid
from typing import NamedTuple

from openai import OpenAI, pydantic_function_tool
from src.utils.billing import TokenBudgetController
from src.agents.tools import SearchDisclosuresInput
from src.agents.tools import RetrievalTool
from src.agents.conversation_store import ConversationStore

SYSTEM_PROMPT = (
    "You are a financial research assistant with access to one tool, "
    "retrieve_disclosures, which searches Commonwealth Bank of Australia's "
    "financial disclosures and returns a grounded answer.\n\n"
    "For each user question, decide one of three things:\n"
    "1. If the question is clear and specific enough to search for (it names "
    "or implies a specific metric, period, or fact), call retrieve_disclosures.\n"
    "2. If the question is ambiguous or missing something you'd need to search "
    "correctly — e.g. it doesn't say which reporting period, which entity, or "
    "which metric — do not call the tool. Instead, respond with a short "
    "clarifying question asking for the missing detail.\n"
    "3. If the question is clearly unrelated to CBA's financial disclosures, "
    "do not call the tool. Respond honestly that you can't help with that.\n\n"
    "After retrieve_disclosures returns a result, that result is already a "
    "grounded answer to the query you searched for. If it contains real "
    "information, relay it to the user as your final answer — you may "
    "lightly rephrase, but do not alter any figures.\n\n"
    "If the tool's result indicates the information wasn't found, you may "
    "rephrase your search once or twice and try again. But if you have "
    "already tried retrieve_disclosures 3 times for this question without "
    "finding the answer, stop retrying — tell the user honestly that you "
    "can't answer this from the available disclosures, rather than guessing "
    "or trying indefinitely."
)

class ModelTurn(NamedTuple):
    content: str
    tool_calls: list
    conversation: list


class AgentLoop:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.finops = TokenBudgetController()
        self.tool = RetrievalTool()
        self.conversation_store = ConversationStore()

    def run_loop(self, query: str, reference_id: str = None, max_turns: int = 5) -> tuple:
        """Answers one question, or advances one in-progress conversation by
        one turn. Pass no reference_id to start fresh; pass back the
        reference_id a prior call returned (e.g. after it asked a
        clarifying question) to continue that same conversation.

        Returns (answer, reference_id) -- always hand the reference_id
        back to the caller, even on a final answer, so they have it
        available if they want to ask a follow-up in the same conversation.

        Each turn: call the model, see whether it answered in text or
        asked to call a tool. A tool call gets executed and its result fed
        back in as the next turn's context; a text response ends the loop.
        """
        if reference_id:
            conversation = self.conversation_store.get_conversation(reference_id)
        else:
            reference_id = str(uuid.uuid4())
            conversation = []

        query_to_send = query

        for turn in range(max_turns):
            turn_result = self._call_model(conversation=conversation, query=query_to_send)

            if turn_result.content:
                self.conversation_store.save_conversation(reference_id, turn_result.conversation)
                return turn_result.content, reference_id
            elif turn_result.tool_calls:
                tool_calls = turn_result.tool_calls
                tool_input = SearchDisclosuresInput.model_validate_json(tool_calls[0].function.arguments)
                disclosures = self.tool.retrieve_disclosures(SearchDisclosuresInput(query=tool_input.query))
                query_to_send = ""

                conversation.append({"role": "tool", "tool_call_id": tool_calls[0].id, "content": disclosures.answer})
            else:
                self.conversation_store.save_conversation(reference_id, conversation)
                return "No response from the model.", reference_id

        self.conversation_store.save_conversation(reference_id, conversation)
        return "I couldn't resolve this in a reasonable number of steps.", reference_id

    def _call_model(self, conversation: list, query: str) -> ModelTurn:
        complete_convo = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation:
            complete_convo.extend(conversation)

        if query:
            complete_convo.append({"role": "user", "content": query})
            conversation.append({"role": "user", "content": query})

        openai_tools = [
            pydantic_function_tool(
                name="retrieve_disclosures",
                description="Searches internal database for relevant disclosures.",
                model=SearchDisclosuresInput
            )
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=complete_convo,
            tools=openai_tools
        )

        self.finops.update_usage_and_verify(
            model="gpt-4o-mini",
            prompt=response.usage.prompt_tokens,
            completion=response.usage.completion_tokens
        )

        content = (response.choices[0].message.content or "").strip()
        tool_calls = response.choices[0].message.tool_calls

        if tool_calls:
            conversation.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        else:
            conversation.append({"role": "assistant", "content": content})

        return ModelTurn(content=content, tool_calls=tool_calls, conversation=conversation)
