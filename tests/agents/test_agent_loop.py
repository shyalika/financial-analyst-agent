import json
from unittest.mock import MagicMock

import pytest

from src.agents import agent_loop as agent_loop_module
from src.agents.agent_loop import AgentLoop


@pytest.fixture
def loop(monkeypatch):
    # RetrievalTool() and ConversationStore() both default to real files
    # under data/ when unconfigured -- patch the classes so constructing
    # AgentLoop never touches them.
    monkeypatch.setattr(agent_loop_module, "RetrievalTool", MagicMock())
    monkeypatch.setattr(agent_loop_module, "ConversationStore", MagicMock())
    instance = AgentLoop()
    instance.client = MagicMock()
    instance.conversation_store.get_conversation.return_value = []
    return instance


def chat_response(content=None, tool_calls=None, prompt_tokens=20, completion_tokens=10):
    message = MagicMock(content=content, tool_calls=tool_calls)
    choice = MagicMock(message=message)
    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return MagicMock(choices=[choice], usage=usage)


def fake_tool_call(call_id="call_1", query="What was NPAT?", final_k=5):
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.arguments = json.dumps({"query": query, "final_k": final_k})
    return tool_call


# ---------------------------------------------------------------------------
# run_loop -- starting fresh vs. resuming
# ---------------------------------------------------------------------------

def test_run_loop_starts_fresh_when_no_reference_id_given(loop):
    loop.client.chat.completions.create.return_value = chat_response(content="Final answer", tool_calls=None)

    answer, reference_id = loop.run_loop("What was NPAT?")

    assert answer == "Final answer"
    assert isinstance(reference_id, str) and reference_id
    loop.conversation_store.get_conversation.assert_not_called()
    loop.conversation_store.save_conversation.assert_called_once()
    assert loop.conversation_store.save_conversation.call_args.args[0] == reference_id


def test_run_loop_resumes_existing_conversation_by_reference_id(loop):
    prior_conversation = [
        {"role": "user", "content": "What was the profit?"},
        {"role": "assistant", "content": "Which period do you mean?"},
    ]
    loop.conversation_store.get_conversation.return_value = prior_conversation
    loop.client.chat.completions.create.return_value = chat_response(content="Final answer", tool_calls=None)

    answer, returned_id = loop.run_loop("The 2026 half year", reference_id="existing-ref")

    assert returned_id == "existing-ref"  # not regenerated
    loop.conversation_store.get_conversation.assert_called_once_with("existing-ref")

    sent_messages = loop.client.chat.completions.create.call_args.kwargs["messages"]
    assert sent_messages[1] == prior_conversation[0]
    assert sent_messages[2] == prior_conversation[1]
    assert sent_messages[-1] == {"role": "user", "content": "The 2026 half year"}


# ---------------------------------------------------------------------------
# run_loop -- the tool-call round trip
# ---------------------------------------------------------------------------

def test_run_loop_executes_tool_then_relays_final_answer(loop):
    tool_call = fake_tool_call(query="What was NPAT?")
    loop.client.chat.completions.create.side_effect = [
        chat_response(content="", tool_calls=[tool_call]),
        chat_response(content="Relayed: $5,412 million", tool_calls=None),
    ]
    loop.tool.retrieve_disclosures.return_value = MagicMock(answer="$5,412 million")

    answer, reference_id = loop.run_loop("What was NPAT?")

    assert answer == "Relayed: $5,412 million"
    loop.tool.retrieve_disclosures.assert_called_once()
    called_input = loop.tool.retrieve_disclosures.call_args.args[0]
    assert called_input.query == "What was NPAT?"

    # The tool's result must have been fed back into the conversation
    # before the second model call.
    second_call_messages = loop.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "$5,412 million"
    assert tool_messages[0]["tool_call_id"] == "call_1"

    loop.conversation_store.save_conversation.assert_called_once()


# ---------------------------------------------------------------------------
# run_loop -- stopping conditions
# ---------------------------------------------------------------------------

def test_run_loop_returns_fallback_when_model_gives_no_content_and_no_tool_calls(loop):
    loop.client.chat.completions.create.return_value = chat_response(content=None, tool_calls=None)

    answer, reference_id = loop.run_loop("anything")

    assert answer == "No response from the model."
    loop.conversation_store.save_conversation.assert_called_once()


def test_run_loop_gives_up_after_max_turns(loop):
    # The model keeps asking to call the tool and never gives a final
    # answer -- max_turns is the hard guarantee that this still terminates.
    tool_call = fake_tool_call()
    loop.client.chat.completions.create.return_value = chat_response(content="", tool_calls=[tool_call])
    loop.tool.retrieve_disclosures.return_value = MagicMock(answer="still not enough")

    answer, reference_id = loop.run_loop("What was NPAT?", max_turns=3)

    assert answer == "I couldn't resolve this in a reasonable number of steps."
    assert loop.client.chat.completions.create.call_count == 3
    assert loop.tool.retrieve_disclosures.call_count == 3
    loop.conversation_store.save_conversation.assert_called_once()


# ---------------------------------------------------------------------------
# _call_model -- the single-turn building block
# ---------------------------------------------------------------------------

def test_call_model_sends_system_prompt_then_user_message(loop):
    loop.client.chat.completions.create.return_value = chat_response(content="hello back", tool_calls=None)

    turn = loop._call_model(conversation=[], query="hello")

    sent_messages = loop.client.chat.completions.create.call_args.kwargs["messages"]
    assert sent_messages[0] == {"role": "system", "content": agent_loop_module.SYSTEM_PROMPT}
    assert sent_messages[1] == {"role": "user", "content": "hello"}
    assert turn.content == "hello back"
    assert turn.tool_calls is None


def test_call_model_mutates_conversation_in_place(loop):
    loop.client.chat.completions.create.return_value = chat_response(content="hello back", tool_calls=None)

    conversation = []
    turn = loop._call_model(conversation=conversation, query="hello")

    assert conversation == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hello back"},
    ]
    assert turn.conversation is conversation


def test_call_model_appends_tool_calls_to_conversation_when_present(loop):
    tool_call = fake_tool_call()
    loop.client.chat.completions.create.return_value = chat_response(content="", tool_calls=[tool_call])

    conversation = []
    turn = loop._call_model(conversation=conversation, query="What was NPAT?")

    assert conversation[-1]["role"] == "assistant"
    assert conversation[-1]["tool_calls"] == [tool_call]
    assert turn.tool_calls == [tool_call]


def test_call_model_tracks_cost(loop):
    loop.client.chat.completions.create.return_value = chat_response(
        content="hello back", tool_calls=None, prompt_tokens=123, completion_tokens=45
    )

    loop._call_model(conversation=[], query="hello")

    assert loop.finops.usage.prompt_tokens == 123
    assert loop.finops.usage.completion_tokens == 45
