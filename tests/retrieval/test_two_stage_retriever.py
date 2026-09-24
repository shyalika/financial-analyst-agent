from unittest.mock import MagicMock

from src.retrieval import two_stage_retriever as two_stage_retriever_module
from src.retrieval.two_stage_retriever import TwoStageRetriever


def make_retriever(monkeypatch, fake_search, mock_cohere_client):
    monkeypatch.setattr(two_stage_retriever_module, "cohere", MagicMock(Client=lambda api_key: mock_cohere_client))
    retriever = TwoStageRetriever(search=fake_search)
    retriever.client = MagicMock()
    retriever.cohere_client = mock_cohere_client
    return retriever


def test_fetch_candidates_embeds_query_then_searches(monkeypatch, mock_cohere_client):
    fake_search = MagicMock()
    fake_search.search.return_value = [{"chunk_content": "c1"}]
    retriever = make_retriever(monkeypatch, fake_search, mock_cohere_client)
    monkeypatch.setattr(
        two_stage_retriever_module, "embed_text",
        lambda client, finops, query: [0.1, 0.2]
    )

    result = retriever.fetch_candidates("What was NPAT?", candidate_k=20)

    fake_search.search.assert_called_once_with([0.1, 0.2], top_k=20)
    assert result == [{"chunk_content": "c1"}]


def test_rerank_maps_results_back_to_candidates_with_score_and_rank(monkeypatch, mock_cohere_client):
    fake_search = MagicMock()
    retriever = make_retriever(monkeypatch, fake_search, mock_cohere_client)

    candidates = [
        {"chunk_content": "first candidate", "parent_id": 1},
        {"chunk_content": "second candidate", "parent_id": 2},
        {"chunk_content": "third candidate", "parent_id": 3},
    ]
    rerank_result_a = MagicMock(index=2, relevance_score=0.95)
    rerank_result_b = MagicMock(index=0, relevance_score=0.42)
    mock_cohere_client.rerank.return_value = MagicMock(results=[rerank_result_a, rerank_result_b])

    reranked = retriever.rerank("query", candidates, final_k=2)

    assert len(reranked) == 2
    assert reranked[0]["parent_id"] == 3
    assert reranked[0]["relevance_score"] == 0.95
    assert reranked[0]["stage1_rank"] == 3
    assert reranked[1]["parent_id"] == 1
    assert reranked[1]["stage1_rank"] == 1
    mock_cohere_client.rerank.assert_called_once_with(
        model=two_stage_retriever_module.RERANK_MODEL,
        query="query",
        documents=["first candidate", "second candidate", "third candidate"],
        top_n=2,
    )


def test_retrieve_wires_stage1_into_stage2_and_reports_latency(monkeypatch, mock_cohere_client):
    fake_search = MagicMock()
    fake_search.search.return_value = [{"chunk_content": "c1", "parent_id": 1}]
    retriever = make_retriever(monkeypatch, fake_search, mock_cohere_client)
    monkeypatch.setattr(two_stage_retriever_module, "embed_text", lambda client, finops, query: [0.1])
    mock_cohere_client.rerank.return_value = MagicMock(
        results=[MagicMock(index=0, relevance_score=0.9)]
    )

    result = retriever.retrieve("question", candidate_k=20, final_k=5)

    assert result["stage1_candidates"] == [{"chunk_content": "c1", "parent_id": 1}]
    assert result["reranked"][0]["relevance_score"] == 0.9
    assert isinstance(result["stage1_latency_ms"], float)
    assert isinstance(result["stage2_latency_ms"], float)
    assert result["stage1_latency_ms"] >= 0
    assert result["stage2_latency_ms"] >= 0
