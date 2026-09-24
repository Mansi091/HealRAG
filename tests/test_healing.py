import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from healing.hallucination_checker import HallucinationChecker
from healing.retry_manager import RetryManager
from evidence.evidence_logger import EvidenceLogger


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="Bounded retries prevent infinite loops.", metadata={"source": "test.txt", "page": 0}),
    ]


# --- HallucinationChecker Tests ---
def test_checker_no_documents():
    checker = HallucinationChecker(threshold=0.7)
    result = checker.check("question", "some answer", [])
    assert result.grounded is False
    assert result.score == 0.0


def test_checker_grounded_answer(sample_docs):
    checker = HallucinationChecker(threshold=0.7)
    with patch.object(checker, "chain") as mock_chain:
        mock_resp = MagicMock()
        mock_resp.content = '{"grounded": true, "score": 0.92, "reason": "Fully supported."}'
        mock_chain.invoke.return_value = mock_resp
        result = checker.check("What prevents infinite loops?", "Bounded retries prevent infinite loops.", sample_docs)
        assert result.grounded is True
        assert result.score >= 0.7


def test_checker_ungrounded_answer(sample_docs):
    checker = HallucinationChecker(threshold=0.7)
    with patch.object(checker, "chain") as mock_chain:
        mock_resp = MagicMock()
        mock_resp.content = '{"grounded": false, "score": 0.25, "reason": "Not in context."}'
        mock_chain.invoke.return_value = mock_resp
        result = checker.check("question", "fabricated answer", sample_docs)
        assert result.grounded is False


# --- RetryManager Tests ---
def test_retry_manager_can_retry():
    mgr = RetryManager(max_retries=2)
    assert mgr.can_retry(0) is True
    assert mgr.can_retry(1) is True
    assert mgr.can_retry(2) is False


def test_retry_manager_reaches_limit():
    mgr = RetryManager(max_retries=2)
    with patch("healing.retry_manager.evidence_logger") as mock_logger:
        mock_logger.log_event.return_value = {}
        record = mgr.record_retry("test query", 1, "ungrounded_answer", "regenerate_answer")
        assert record.attempt_number == 2
        assert record.can_retry is False


def test_evidence_logger_writes_and_reads(tmp_path):
    journal_path = str(tmp_path / "test_journal.jsonl")
    logger = EvidenceLogger(journal_path=journal_path)
    event = logger.log_event(
        query="What is RAG?",
        failure_type="poor_retrieval",
        action="query_rewrite",
        attempt=1,
        result="recovered",
        status="success"
    )
    assert event["query"] == "What is RAG?"
    events = logger.get_recent_events(limit=10)
    assert len(events) == 1
    assert events[0]["action"] == "query_rewrite"
