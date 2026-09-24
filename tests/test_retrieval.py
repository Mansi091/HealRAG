import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from retrieval.retriever import DocumentRetriever
from retrieval.relevance_grader import RelevanceGrader
from retrieval.query_rewriter import QueryRewriter


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="RAG systems use vector search to retrieve context.", metadata={"source": "test.txt", "page": 0}),
        Document(page_content="Bounded retries prevent infinite loops in healing.", metadata={"source": "guide.txt", "page": 1}),
    ]


def test_retriever_returns_list(sample_docs):
    with patch("retrieval.retriever.chroma_store") as mock_store:
        mock_store.similarity_search.return_value = sample_docs
        retriever = DocumentRetriever(top_k=2)
        results = retriever.retrieve("What is RAG?")
        assert isinstance(results, list)
        assert len(results) == 2


def test_retriever_handles_error():
    with patch("retrieval.retriever.chroma_store") as mock_store:
        mock_store.similarity_search.side_effect = Exception("DB error")
        retriever = DocumentRetriever(top_k=2)
        results = retriever.retrieve("test query")
        assert results == []


def test_relevance_grader_no_documents():
    grader = RelevanceGrader(threshold=0.5)
    result = grader.grade("test question", [])
    assert result.relevant is False
    assert result.relevance_score == 0.0


def test_relevance_grader_with_docs(sample_docs):
    grader = RelevanceGrader(threshold=0.1)
    with patch.object(grader, "chain") as mock_chain:
        mock_resp = MagicMock()
        mock_resp.content = '{"relevant": true, "score": 0.85, "reason": "Related topic."}'
        mock_chain.invoke.return_value = mock_resp
        result = grader.grade("RAG retrieval", sample_docs)
        assert isinstance(result.relevance_score, float)


def test_query_rewriter_returns_result():
    rewriter = QueryRewriter()
    with patch.object(rewriter, "chain") as mock_chain:
        mock_resp = MagicMock()
        mock_resp.content = '{"rewritten_query": "key concepts of RAG retrieval", "reason": "Added specificity."}'
        mock_chain.invoke.return_value = mock_resp
        result = rewriter.rewrite("RAG?")
        assert result.rewritten_query != ""
        assert result.original_query == "RAG?"
