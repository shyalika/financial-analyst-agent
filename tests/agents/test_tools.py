from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.agents import tools
from src.agents.tools import RetrievalTool, RetrievalToolError, SearchDisclosuresInput, Answer
#from src.generation import Answer


@pytest.fixture
def mock_retriever(monkeypatch):
    retriever = MagicMock()
    retriever.retrieve.return_value = {
        "reranked": [
            {"parent_id": 4046, "chunk_content": "c1", "source_file": "a.pdf"},
            {"parent_id": 8693, "chunk_content": "c2", "source_file": "b.pdf"},
        ]
    }
    monkeypatch.setattr(tools, "TwoStageRetriever", MagicMock(return_value=retriever))
    return retriever


@pytest.fixture
def mock_generator(monkeypatch):
    generator = MagicMock()
    generator.generate_answer.return_value = Answer(answer="NPAT was $10,982m", parent_ids=[4046, 8693])
    monkeypatch.setattr(tools, "AnswerGenerator", MagicMock(return_value=generator))
    return generator


def test_search_disclosures_input_rejects_non_string_query():
    with pytest.raises(ValidationError):
        SearchDisclosuresInput(query=123)


def test_retrieve_returns_answer_and_parent_ids(mock_retriever, mock_generator):
    result = RetrievalTool().retrieve(SearchDisclosuresInput(query="What was NPAT?"))

    assert result.answer == "NPAT was $10,982m"
    assert result.parent_ids == [4046, 8693]


def test_retrieve_passes_query_and_final_k_to_retriever(mock_retriever, mock_generator):
    RetrievalTool().retrieve(SearchDisclosuresInput(query="What was NPAT?", final_k=3))

    mock_retriever.retrieve.assert_called_once_with("What was NPAT?", candidate_k=20, final_k=3)


def test_retrieve_wraps_errors_in_retrieval_tool_error(mock_retriever, mock_generator):
    mock_retriever.retrieve.side_effect = RuntimeError("db down")

    with pytest.raises(RetrievalToolError):
        RetrievalTool().retrieve(SearchDisclosuresInput(query="What was NPAT?"))
