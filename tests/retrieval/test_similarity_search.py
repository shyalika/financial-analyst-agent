import json
from unittest.mock import MagicMock

from src.retrieval import similarity_search as similarity_search_module
from src.retrieval.similarity_search import SimilaritySearch


def make_search(monkeypatch, canned_chunks):
    monkeypatch.setattr(similarity_search_module, "SQLiteVectorStore", MagicMock())
    search = SimilaritySearch()
    search.db = MagicMock()
    search.db.fetch_all_child_chunks.return_value = canned_chunks
    return search


def chunk(child_id, embedding):
    return {
        "child_id": child_id,
        "parent_id": child_id,
        "chunk_content": f"chunk {child_id}",
        "embedding_json": json.dumps(embedding),
        "source_file": "report.pdf",
        "section_title": "S",
        "parent_content": "parent text",
    }


def test_search_sorts_descending_by_similarity(monkeypatch):
    canned = [
        chunk(1, [0.0, 1.0]),   # orthogonal to query -> similarity 0
        chunk(2, [1.0, 0.0]),   # identical to query -> similarity 1
        chunk(3, [0.7, 0.7]),   # partial match
    ]
    search = make_search(monkeypatch, canned)

    results = search.search([1.0, 0.0], top_k=5)

    assert [r["child_id"] for r in results] == [2, 3, 1]
    assert results[0]["similarity"] == 1.0


def test_search_truncates_to_top_k(monkeypatch):
    canned = [chunk(i, [1.0, 0.0]) for i in range(10)]
    search = make_search(monkeypatch, canned)

    results = search.search([1.0, 0.0], top_k=3)

    assert len(results) == 3


def test_search_zero_norm_candidate_scores_zero_not_error(monkeypatch):
    canned = [chunk(1, [0.0, 0.0])]
    search = make_search(monkeypatch, canned)

    results = search.search([1.0, 0.0], top_k=5)

    assert results[0]["similarity"] == 0.0


def test_search_strips_embedding_json_from_results(monkeypatch):
    canned = [chunk(1, [1.0, 0.0])]
    search = make_search(monkeypatch, canned)

    [result] = search.search([1.0, 0.0], top_k=5)

    assert "embedding_json" not in result
    assert result["chunk_content"] == "chunk 1"
