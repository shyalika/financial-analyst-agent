def embed_text(client, finops, text: str, model: str = "text-embedding-3-small") -> list:
    response = client.embeddings.create(
        input=[text.replace("\n", " ")],
        model=model
    )
    finops.update_usage_and_verify(
        model="gpt-4o-mini",
        prompt=response.usage.prompt_tokens,
        completion=0
    )
    return response.data[0].embedding
