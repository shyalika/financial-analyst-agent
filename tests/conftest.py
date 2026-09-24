import os
from unittest.mock import MagicMock

import pytest

# Dummy keys so client constructors (OpenAI(api_key=...), cohere.Client(...))
# don't raise at __init__ time. No test in this suite performs a real network
# call — every client method that would hit the network is mocked/faked.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("COHERE_API_KEY", "test-key-not-real")


@pytest.fixture
def sample_embedding():
    return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def make_embedding_response():
    """Builds a fake object shaped like an OpenAI embeddings.create() response."""
    def _make(vectors, prompt_tokens=10):
        data = [MagicMock(embedding=v) for v in vectors]
        usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=0)
        return MagicMock(data=data, usage=usage)
    return _make


@pytest.fixture
def make_chat_response():
    """Builds a fake object shaped like an OpenAI chat.completions.create() response."""
    def _make(content, prompt_tokens=20, completion_tokens=10):
        message = MagicMock(content=content)
        choice = MagicMock(message=message)
        usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        return MagicMock(choices=[choice], usage=usage)
    return _make


@pytest.fixture
def mock_openai_client():
    from openai import OpenAI
    client = MagicMock(spec=OpenAI)
    client.embeddings = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    return client


@pytest.fixture
def mock_cohere_client():
    import cohere
    return MagicMock(spec=cohere.Client)


@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test_financial_intelligence.db")
