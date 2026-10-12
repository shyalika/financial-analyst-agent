import json
import sqlite3
import time

import pytest

from src.agents.conversation_store import ConversationStore


@pytest.fixture
def store(tmp_path):
    return ConversationStore(db_path=str(tmp_path / "test_conversations.db"))


class FakeToolCall:
    """Stands in for an OpenAI SDK ChatCompletionMessageToolCall -- the
    real thing has a .model_dump() method and isn't JSON-serializable
    directly."""
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


def _insert_raw(store: ConversationStore, reference_id: str, conversation: list, last_touched: float):
    """Writes a row directly, bypassing save_conversation, so a specific
    last_touched timestamp (e.g. one already past the TTL) can be forced --
    save_conversation always stamps time.time() itself."""
    conn = sqlite3.connect(store.db_path)
    conn.execute(
        "INSERT INTO agent_conversations (reference_id, conversation_json, last_touched) VALUES (?, ?, ?);",
        (reference_id, json.dumps(conversation), last_touched),
    )
    conn.commit()
    conn.close()


def test_init_creates_table(store):
    conn = sqlite3.connect(store.db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")}
    conn.close()
    assert "agent_conversations" in tables


def test_get_conversation_missing_returns_empty_list(store):
    assert store.get_conversation("unknown-id") == []


def test_save_then_get_round_trips_plain_messages(store):
    conversation = [
        {"role": "user", "content": "What was NPAT?"},
        {"role": "assistant", "content": "It was $5,412 million."},
    ]
    store.save_conversation("ref-1", conversation)

    assert store.get_conversation("ref-1") == conversation


def test_save_serializes_fresh_tool_calls_via_model_dump(store):
    tool_call_data = {"id": "call_1", "type": "function", "function": {"name": "retrieve_disclosures", "arguments": "{}"}}
    conversation = [
        {"role": "assistant", "content": "", "tool_calls": [FakeToolCall(tool_call_data)]},
    ]

    store.save_conversation("ref-2", conversation)
    loaded = store.get_conversation("ref-2")

    assert loaded[0]["tool_calls"] == [tool_call_data]


def test_save_handles_already_plain_dict_tool_calls(store):
    # Regression test: a conversation that already round-tripped through
    # get_conversation() once has tool_calls as plain dicts (from
    # json.loads), not SDK objects with .model_dump(). Re-saving that
    # conversation (e.g. after a later turn) must not crash trying to call
    # .model_dump() on something that's already a dict.
    already_plain = {"id": "call_1", "type": "function", "function": {"name": "retrieve_disclosures", "arguments": "{}"}}
    conversation = [
        {"role": "assistant", "content": "", "tool_calls": [already_plain]},
    ]

    store.save_conversation("ref-3", conversation)  # must not raise
    loaded = store.get_conversation("ref-3")

    assert loaded[0]["tool_calls"] == [already_plain]


def test_save_conversation_upserts_existing_reference_id(store):
    store.save_conversation("ref-4", [{"role": "user", "content": "first"}])
    store.save_conversation("ref-4", [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}])

    loaded = store.get_conversation("ref-4")
    assert len(loaded) == 2
    assert loaded[1]["content"] == "second"

    conn = sqlite3.connect(store.db_path)
    row_count = conn.execute("SELECT COUNT(*) FROM agent_conversations WHERE reference_id = 'ref-4';").fetchone()[0]
    conn.close()
    assert row_count == 1  # upsert, not a second row


def test_get_conversation_deletes_expired_row_before_fetching(store):
    expired_timestamp = time.time() - store.TTL_SECONDS - 100
    _insert_raw(store, "expired-ref", [{"role": "user", "content": "old"}], expired_timestamp)

    result = store.get_conversation("expired-ref")

    assert result == []
    conn = sqlite3.connect(store.db_path)
    row_count = conn.execute("SELECT COUNT(*) FROM agent_conversations WHERE reference_id = 'expired-ref';").fetchone()[0]
    conn.close()
    assert row_count == 0  # actually deleted, not just treated as absent


def test_get_conversation_leaves_other_non_expired_rows_alone(store):
    expired_timestamp = time.time() - store.TTL_SECONDS - 100
    _insert_raw(store, "expired-ref", [{"role": "user", "content": "old"}], expired_timestamp)
    store.save_conversation("fresh-ref", [{"role": "user", "content": "new"}])

    store.get_conversation("expired-ref")  # triggers the expired sweep

    assert store.get_conversation("fresh-ref") == [{"role": "user", "content": "new"}]
