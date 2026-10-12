import os
import sqlite3
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConversationStore")


class ConversationStore:
    """
    SQLite-backed cache for in-progress agent conversations -- e.g. one
    paused on a clarifying question, waiting for the caller to answer.
    Rows are a cache, not a durable record: anything older than
    TTL_SECONDS is treated as gone, whether or not it was ever resumed.
    """

    TTL_SECONDS = 30 * 60

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/agent_conversations.db")
        )
        self._initialize_database()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_database(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_conversations (
                    reference_id TEXT PRIMARY KEY,
                    conversation_json TEXT NOT NULL,
                    last_touched REAL NOT NULL
                );
            """)
            conn.commit()
        logger.info(f"⚡ Conversation store initialized at: {self.db_path}")

    def get_conversation(self, reference_id: str) -> list:
        """Returns the cached conversation for reference_id, or an empty
        list if it's missing or expired."""
        with self._get_connection() as conn:
            # Deleting everything expired before every fetch keeps a stale
            # reference_id correctly behaving as "not found" without a
            # separate sweep step. Piggybacking cleanup on every fetch is a
            # pragmatic stand-in for a single hand-built script -- cleaning
            # up the expired conversations should be done as a separate
            # background thread as a future change, instead of happening
            # inline with request handling like this.
            conn.execute(
                "DELETE FROM agent_conversations WHERE last_touched < ?;",
                (time.time() - self.TTL_SECONDS,)
            )
            conn.commit()

            row = conn.execute(
                "SELECT conversation_json FROM agent_conversations WHERE reference_id = ?;",
                (reference_id,)
            ).fetchone()

        if row is None:
            return []
        return json.loads(row[0])

    def save_conversation(self, reference_id: str, conversation: list):
        serializable_conversation = [self._serialize_message(message) for message in conversation]
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_conversations (reference_id, conversation_json, last_touched)
                VALUES (?, ?, ?)
                ON CONFLICT(reference_id) DO UPDATE SET
                    conversation_json = excluded.conversation_json,
                    last_touched = excluded.last_touched;
                """,
                (reference_id, json.dumps(serializable_conversation), time.time())
            )
            conn.commit()

    @staticmethod
    def _serialize_message(message: dict) -> dict:
        # A freshly-generated assistant message with tool_calls carries raw
        # OpenAI SDK objects (ChatCompletionMessageToolCall), not plain
        # dicts -- json.dumps can't serialize those directly. But once a
        # conversation has round-tripped through get_conversation() once,
        # its tool_calls are already plain dicts (from json.loads), so
        # re-saving it (e.g. after a later turn) must not assume every
        # tool_calls entry still has .model_dump(). Plain dicts round-trip
        # back into the API fine either way -- it only needs JSON-shaped
        # data, not the actual SDK class.
        if message.get("tool_calls"):
            message = {
                **message,
                "tool_calls": [
                    tc.model_dump() if hasattr(tc, "model_dump") else tc
                    for tc in message["tool_calls"]
                ],
            }
        return message
