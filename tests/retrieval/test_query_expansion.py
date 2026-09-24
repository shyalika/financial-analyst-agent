from src.retrieval.query_expansion import QueryExpansionGenerator, SYSTEM_PROMPT


def make_generator(mock_openai_client):
    gen = QueryExpansionGenerator()
    gen.client = mock_openai_client
    return gen


def test_generate_alternative_phrasings_parses_lines_and_strips_numbering(mock_openai_client, make_chat_response):
    gen = make_generator(mock_openai_client)
    mock_openai_client.chat.completions.create.return_value = make_chat_response(
        "1. What was CBA's NPAT for 1H26?\n"
        "- How much net profit did CBA report for the half?\n"
        "* CBA's statutory profit after tax for 1H26 was how much?\n"
    )

    phrasings = gen.generate_alternative_phrasings("What was NPAT?", num_variants=3)

    assert phrasings == [
        "What was CBA's NPAT for 1H26?",
        "How much net profit did CBA report for the half?",
        "CBA's statutory profit after tax for 1H26 was how much?",
    ]


def test_generate_alternative_phrasings_truncates_to_num_variants(mock_openai_client, make_chat_response):
    gen = make_generator(mock_openai_client)
    mock_openai_client.chat.completions.create.return_value = make_chat_response(
        "phrasing one\nphrasing two\nphrasing three\nphrasing four"
    )

    phrasings = gen.generate_alternative_phrasings("question", num_variants=2)

    assert phrasings == ["phrasing one", "phrasing two"]


def test_generate_alternative_phrasings_drops_blank_lines(mock_openai_client, make_chat_response):
    gen = make_generator(mock_openai_client)
    mock_openai_client.chat.completions.create.return_value = make_chat_response(
        "phrasing one\n\n   \nphrasing two"
    )

    phrasings = gen.generate_alternative_phrasings("question", num_variants=5)

    assert phrasings == ["phrasing one", "phrasing two"]


def test_generate_alternative_phrasings_sends_system_prompt_and_variant_count(mock_openai_client, make_chat_response):
    gen = make_generator(mock_openai_client)
    mock_openai_client.chat.completions.create.return_value = make_chat_response("a\nb")

    gen.generate_alternative_phrasings("What was NPAT?", num_variants=2)

    messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "Generate 2 alternative phrasings" in messages[1]["content"]
    assert "What was NPAT?" in messages[1]["content"]


def test_generate_alternative_phrasings_tracks_cost(mock_openai_client, make_chat_response):
    gen = make_generator(mock_openai_client)
    mock_openai_client.chat.completions.create.return_value = make_chat_response(
        "a\nb", prompt_tokens=50, completion_tokens=25
    )

    gen.generate_alternative_phrasings("question")

    assert gen.finops.usage.prompt_tokens == 50
    assert gen.finops.usage.completion_tokens == 25
