import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from generation.generator import ResponseGenerator


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="RAG systems ground answers in retrieved evidence.",
            metadata={"source": "rag_guide.txt", "page": 0, "filename": "rag_guide.txt"}
        ),
    ]


def test_generator_returns_answer_and_sources(sample_docs):
    gen = ResponseGenerator()
    with patch.object(gen, "chain") as mock_chain:
        mock_chain.invoke.return_value = "RAG grounds answers in retrieved context."
        answer, sources = gen.generate("What does RAG do?", sample_docs)
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(sources, list)
        assert sources[0]["source"] == "rag_guide.txt"


def test_generator_handles_empty_docs():
    gen = ResponseGenerator()
    with patch.object(gen, "chain") as mock_chain:
        mock_chain.invoke.return_value = "I do not have enough information."
        answer, sources = gen.generate("Any question?", [])
        assert isinstance(answer, str)
        assert sources == []


def test_generator_extracts_sources_deduped():
    docs = [
        Document(page_content="content A", metadata={"source": "doc.txt", "filename": "doc.txt", "page": 0}),
        Document(page_content="content B", metadata={"source": "doc.txt", "filename": "doc.txt", "page": 0}),
    ]
    gen = ResponseGenerator()
    sources = gen._extract_sources(docs)
    assert len(sources) == 1
