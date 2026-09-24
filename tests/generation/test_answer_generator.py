import json

from src.generation.answer_generator import Answer, AnswerGenerator, SYSTEM_PROMPT


def make_generator(mock_openai_client):
    gen = AnswerGenerator()
    gen.client = mock_openai_client
    return gen


def json_answer(answer="answer", parent_ids=()):
    return json.dumps({"answer": answer, "parent_ids": list(parent_ids)})


def test_generate_answer_builds_parent_id_labeled_context_block(mock_openai_client, make_chat_response):
    gen = make_generator(mock_openai_client)
    mock_openai_client.chat.completions.create.return_value = make_chat_response(json_answer())
    chunks = [
        {"parent_id": 11, "source_file": "1H26.pdf", "chunk_content": "NPAT was $5,412m."},
        {"parent_id": 22, "source_file": "FY26.pdf", "chunk_content": "ROE was 13.8%."},
    ]

    gen.generate_answer("What was NPAT?", chunks)

    user_message = mock_openai_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "[parent_id: 11  | Source: 1H26.pdf] NPAT was $5,412m." in user_message
    assert "[parent_id: 22  | Source: FY26.pdf] ROE was 13.8%." in user_message
    assert "Question: What was NPAT?" in user_message


def test_generate_answer_sends_system_prompt_and_requests_json(mock_openai_client, make_chat_response):
    gen = make_generator(mock_openai_client)
    mock_openai_client.chat.completions.create.return_value = make_chat_response(json_answer())

    gen.generate_answer("question", [])

    kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert kwargs["response_format"] == {"type": "json_object"}


def test_generate_answer_parses_model_json_into_answer(mock_openai_client, make_chat_response):
    gen = make_generator(mock_openai_client)
    mock_openai_client.chat.completions.create.return_value = make_chat_response(
        json_answer("Statutory NPAT was $5,412 million.", [11, 22])
    )

    result = gen.generate_answer("What was NPAT?", [])

    assert result == Answer(answer="Statutory NPAT was $5,412 million.", parent_ids=[11, 22])


def test_generate_answer_tracks_cost_with_requested_model(mock_openai_client, make_chat_response):
    gen = make_generator(mock_openai_client)
    mock_openai_client.chat.completions.create.return_value = make_chat_response(
        json_answer(), prompt_tokens=200, completion_tokens=80
    )

    gen.generate_answer("question", [], model="gpt-4o")

    assert mock_openai_client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o"
    assert gen.finops.usage.prompt_tokens == 200
    assert gen.finops.usage.completion_tokens == 80
