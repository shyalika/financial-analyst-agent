from src.retrieval.hyde import HyDEGenerator


def make_generator(mock_openai_client):
    gen = HyDEGenerator()
    gen.client = mock_openai_client
    return gen


def test_generate_hypothetical_embedding_calls_chat_then_embed_in_order(
    mock_openai_client, make_chat_response, make_embedding_response
):
    gen = make_generator(mock_openai_client)
    mock_openai_client.chat.completions.create.return_value = make_chat_response(
        "CBA reported statutory NPAT of $5,412 million for 1H26.", prompt_tokens=30, completion_tokens=15
    )
    mock_openai_client.embeddings.create.return_value = make_embedding_response([[0.2, 0.3]], prompt_tokens=12)

    text, embedding = gen.generate_hypothetical_embedding("What was NPAT?")

    assert text == "CBA reported statutory NPAT of $5,412 million for 1H26."
    assert embedding == [0.2, 0.3]
    mock_openai_client.chat.completions.create.assert_called_once()
    mock_openai_client.embeddings.create.assert_called_once_with(
        input=["CBA reported statutory NPAT of $5,412 million for 1H26."],
        model="text-embedding-3-small",
    )


def test_generate_hypothetical_embedding_tracks_cost_for_both_calls(
    mock_openai_client, make_chat_response, make_embedding_response
):
    gen = make_generator(mock_openai_client)
    mock_openai_client.chat.completions.create.return_value = make_chat_response(
        "passage", prompt_tokens=30, completion_tokens=15
    )
    mock_openai_client.embeddings.create.return_value = make_embedding_response([[0.1]], prompt_tokens=12)

    gen.generate_hypothetical_embedding("question")

    assert gen.finops.usage.prompt_tokens == 30 + 12
    assert gen.finops.usage.completion_tokens == 15


def test_generate_hypothetical_embedding_strips_newlines_before_embedding(
    mock_openai_client, make_chat_response, make_embedding_response
):
    gen = make_generator(mock_openai_client)
    mock_openai_client.chat.completions.create.return_value = make_chat_response("line one\nline two")
    mock_openai_client.embeddings.create.return_value = make_embedding_response([[0.1]])

    gen.generate_hypothetical_embedding("question")

    assert mock_openai_client.embeddings.create.call_args.kwargs["input"] == ["line one line two"]
