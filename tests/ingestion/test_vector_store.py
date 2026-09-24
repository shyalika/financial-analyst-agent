import json
import sqlite3

import pytest

from src.ingestion.vector_store import SQLiteVectorStore


@pytest.fixture
def store(temp_db_path):
    return SQLiteVectorStore(db_path=temp_db_path)


def test_init_uses_given_db_path_not_production(store, temp_db_path):
    assert store.db_path == temp_db_path


def test_schema_creates_expected_tables(store, temp_db_path):
    conn = sqlite3.connect(temp_db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")}
    conn.close()
    assert {"parent_documents", "child_chunks"} <= tables


def test_insert_document_pipeline_writes_parent_and_children(store):
    store.insert_document_pipeline(
        source_file="report.pdf",
        section_title="Narrative Block #1",
        parent_text="Full parent text.",
        child_tuples=[("chunk one", [0.1, 0.2]), ("chunk two", [0.3, 0.4])],
    )

    chunks = store.fetch_all_child_chunks()
    assert len(chunks) == 2
    contents = {c["chunk_content"] for c in chunks}
    assert contents == {"chunk one", "chunk two"}
    assert all(c["source_file"] == "report.pdf" for c in chunks)
    assert all(c["parent_content"] == "Full parent text." for c in chunks)
    assert json.loads(chunks[0]["embedding_json"]) in ([0.1, 0.2], [0.3, 0.4])


def test_insert_document_pipeline_rolls_back_parent_on_child_failure(store):
    # A non-JSON-serializable "vector" (a set) makes json.dumps blow up
    # mid-transaction; the parent row must not survive the rollback.
    bad_child_tuples = [("chunk one", {1, 2, 3})]

    with pytest.raises(TypeError):
        store.insert_document_pipeline(
            source_file="report.pdf",
            section_title="Section",
            parent_text="Text that should not persist.",
            child_tuples=bad_child_tuples,
        )

    assert store.get_ingested_source_files() == set()
    assert store.fetch_all_child_chunks() == []


def test_get_ingested_source_files_returns_distinct_sources(store):
    store.insert_document_pipeline("a.pdf", "S1", "text a", [("c1", [0.1])])
    store.insert_document_pipeline("a.pdf", "S2", "text a2", [("c2", [0.2])])
    store.insert_document_pipeline("b.pdf", "S1", "text b", [("c3", [0.3])])

    assert store.get_ingested_source_files() == {"a.pdf", "b.pdf"}


def test_fetch_all_child_chunks_joins_parent_fields(store):
    store.insert_document_pipeline(
        source_file="report.pdf",
        section_title="Table Block #1",
        parent_text="$5,412 million",
        child_tuples=[("$5,412 million", [0.9])],
    )

    [chunk] = store.fetch_all_child_chunks()
    assert chunk["section_title"] == "Table Block #1"
    assert chunk["parent_content"] == "$5,412 million"
    assert chunk["chunk_content"] == "$5,412 million"
    assert isinstance(chunk["parent_id"], int)
    assert isinstance(chunk["child_id"], int)
