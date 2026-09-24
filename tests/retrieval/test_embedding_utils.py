from unittest.mock import MagicMock

from src.retrieval.embedding_utils import embed_text


def test_embed_text_strips_newlines_and_returns_embedding(mock_openai_client, make_embedding_response):
    mock_openai_client.embeddings.create.return_value = make_embedding_response([[0.5, 0.6]], prompt_tokens=42)
    finops = MagicMock()

    result = embed_text(mock_openai_client, finops, "line one\nline two")

    mock_openai_client.embeddings.create.assert_called_once_with(
        input=["line one line two"], model="text-embedding-3-small"
    )
    assert result == [0.5, 0.6]


def test_embed_text_tracks_cost_via_finops(mock_openai_client, make_embedding_response):
    mock_openai_client.embeddings.create.return_value = make_embedding_response([[0.1]], prompt_tokens=17)
    finops = MagicMock()

    embed_text(mock_openai_client, finops, "some text")

    finops.update_usage_and_verify.assert_called_once_with(model="gpt-4o-mini", prompt=17, completion=0)


def test_embed_text_respects_custom_model(mock_openai_client, make_embedding_response):
    mock_openai_client.embeddings.create.return_value = make_embedding_response([[0.1]])
    finops = MagicMock()

    embed_text(mock_openai_client, finops, "text", model="text-embedding-3-large")

    assert mock_openai_client.embeddings.create.call_args.kwargs["model"] == "text-embedding-3-large"
