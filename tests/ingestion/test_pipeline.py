from unittest.mock import MagicMock

import pytest

from src.ingestion import pipeline as pipeline_module
from src.ingestion.pipeline import StructuralIngestionPipeline


@pytest.fixture
def pipeline(monkeypatch):
    # SQLiteVectorStore() with no args resolves to the production db path --
    # patch the class so construction never touches it.
    monkeypatch.setattr(pipeline_module, "SQLiteVectorStore", MagicMock())
    p = StructuralIngestionPipeline()
    p.client = MagicMock()
    return p


# ---------------------------------------------------------------------------
# get_embedding_vector / get_embedding_vectors
# ---------------------------------------------------------------------------

def test_get_embedding_vector_returns_embedding_and_tracks_cost(pipeline, make_embedding_response):
    pipeline.client.embeddings.create.return_value = make_embedding_response([[0.1, 0.2]], prompt_tokens=100)

    result = pipeline.get_embedding_vector("hello\nworld")

    assert result == [0.1, 0.2]
    pipeline.client.embeddings.create.assert_called_once_with(
        input=["hello world"], model="text-embedding-3-small"
    )
    assert pipeline.finops.usage.prompt_tokens == 100


def test_get_embedding_vectors_empty_list_makes_no_call(pipeline):
    assert pipeline.get_embedding_vectors([]) == []
    pipeline.client.embeddings.create.assert_not_called()


def test_get_embedding_vectors_batches_calls_and_preserves_order(pipeline, make_embedding_response):
    texts = ["a", "b", "c", "d", "e"]
    # batch_size=2 over 5 texts -> batches of [a,b] [c,d] [e]
    responses = [
        make_embedding_response([[1, 0], [2, 0]]),
        make_embedding_response([[3, 0], [4, 0]]),
        make_embedding_response([[5, 0]]),
    ]
    pipeline.client.embeddings.create.side_effect = responses

    result = pipeline.get_embedding_vectors(texts, batch_size=2)

    assert result == [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
    assert pipeline.client.embeddings.create.call_count == 3
    call_inputs = [c.kwargs["input"] for c in pipeline.client.embeddings.create.call_args_list]
    assert call_inputs == [["a", "b"], ["c", "d"], ["e"]]


# ---------------------------------------------------------------------------
# _cosine_distance
# ---------------------------------------------------------------------------

def test_cosine_distance_identical_vectors_is_zero(pipeline):
    assert pipeline._cosine_distance([1, 0, 0], [1, 0, 0]) == pytest.approx(0.0)


def test_cosine_distance_orthogonal_vectors_is_one(pipeline):
    assert pipeline._cosine_distance([1, 0], [0, 1]) == pytest.approx(1.0)


def test_cosine_distance_zero_norm_returns_one(pipeline):
    assert pipeline._cosine_distance([0, 0], [1, 1]) == 1.0


# ---------------------------------------------------------------------------
# segment_text_semantically
# ---------------------------------------------------------------------------

def test_segment_text_semantically_single_sentence_skips_embedding_call(pipeline):
    result = pipeline.segment_text_semantically("Just one sentence.")
    assert result == ["Just one sentence."]
    pipeline.client.embeddings.create.assert_not_called()


def test_segment_text_semantically_splits_on_threshold_breakpoint(pipeline, monkeypatch):
    text = "First topic sentence. Second topic sentence. Totally different topic."
    # 3 sentences -> 3 embeddings. Distance(0,1) below threshold (same chunk),
    # distance(1,2) above threshold (new chunk).
    embeddings = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    monkeypatch.setattr(pipeline, "get_embedding_vectors", lambda sentences: embeddings)

    chunks = pipeline.segment_text_semantically(text, threshold=0.35)

    assert len(chunks) == 2
    assert "First topic sentence." in chunks[0]
    assert "Second topic sentence." in chunks[0]
    assert chunks[1] == "Totally different topic."


# ---------------------------------------------------------------------------
# _extract_structured_text dispatch
# ---------------------------------------------------------------------------

def test_extract_structured_text_dispatches_pdf(pipeline, monkeypatch):
    sentinel = {"narrative_text": "n", "table_blocks": ["t"]}
    mock_extract = MagicMock(return_value=sentinel)
    monkeypatch.setattr(pipeline_module, "extract_structured_text", mock_extract)

    result = pipeline._extract_structured_text("/tmp/report.pdf")

    assert result == sentinel
    mock_extract.assert_called_once_with("/tmp/report.pdf")


def test_extract_structured_text_dispatches_txt(pipeline, tmp_path):
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("plain narrative content")

    result = pipeline._extract_structured_text(str(txt_path))

    assert result == {"narrative_text": "plain narrative content", "table_blocks": []}


def test_extract_structured_text_unsupported_extension_raises(pipeline):
    with pytest.raises(ValueError):
        pipeline._extract_structured_text("/tmp/file.docx")


# ---------------------------------------------------------------------------
# _ingest_narrative / _ingest_tables / run_file_ingestion orchestration
# ---------------------------------------------------------------------------

def test_ingest_narrative_embeds_in_one_batch_and_inserts_per_parent_block(pipeline, monkeypatch):
    monkeypatch.setattr(
        pipeline, "segment_text_semantically",
        lambda text, threshold=0.35: ["Sentence one. Sentence two.", "Sentence three."]
    )
    monkeypatch.setattr(
        pipeline, "get_embedding_vectors",
        lambda texts: [[float(i)] for i in range(len(texts))]
    )

    pipeline._ingest_narrative("report.pdf", "irrelevant, segmenter is mocked")

    assert pipeline.db.insert_document_pipeline.call_count == 2
    first_call = pipeline.db.insert_document_pipeline.call_args_list[0]
    assert first_call.kwargs["source_file"] == "report.pdf"
    assert first_call.kwargs["section_title"] == "Narrative Block #1"
    assert first_call.kwargs["parent_text"] == "Sentence one. Sentence two."


def test_ingest_tables_skips_when_empty(pipeline):
    pipeline._ingest_tables("report.pdf", [])
    pipeline.db.insert_document_pipeline.assert_not_called()


def test_ingest_tables_embeds_and_inserts_each_block(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline, "get_embedding_vectors", lambda texts: [[1.0], [2.0]])

    pipeline._ingest_tables("report.pdf", ["NPAT: $5,412m", "ROE: 13.8%"])

    assert pipeline.db.insert_document_pipeline.call_count == 2
    calls = pipeline.db.insert_document_pipeline.call_args_list
    assert calls[0].kwargs["parent_text"] == "NPAT: $5,412m"
    assert calls[0].kwargs["child_tuples"] == [("NPAT: $5,412m", [1.0])]
    assert calls[1].kwargs["parent_text"] == "ROE: 13.8%"


def test_run_file_ingestion_raises_when_file_missing(pipeline):
    with pytest.raises(FileNotFoundError):
        pipeline.run_file_ingestion("/nonexistent/path.pdf")


def test_run_file_ingestion_orchestrates_extract_then_ingest(pipeline, monkeypatch, tmp_path):
    existing_file = tmp_path / "report.pdf"
    existing_file.write_bytes(b"fake pdf bytes")

    monkeypatch.setattr(
        pipeline, "_extract_structured_text",
        lambda path: {"narrative_text": "some narrative", "table_blocks": ["NPAT: $1m"]}
    )
    narrative_calls = []
    table_calls = []
    monkeypatch.setattr(pipeline, "_ingest_narrative", lambda sf, nt: narrative_calls.append((sf, nt)))
    monkeypatch.setattr(pipeline, "_ingest_tables", lambda sf, tb: table_calls.append((sf, tb)))

    pipeline.run_file_ingestion(str(existing_file))

    assert narrative_calls == [("report.pdf", "some narrative")]
    assert table_calls == [("report.pdf", ["NPAT: $1m"])]
