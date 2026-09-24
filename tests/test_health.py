import pytest
from unittest.mock import patch, MagicMock
from core.health.health_checker import HealthChecker


def test_health_returns_healthy():
    checker = HealthChecker()
    with patch("core.health.health_checker.chroma_store") as mock_store, \
         patch("core.health.health_checker.baseline_manager") as mock_bm, \
         patch("core.health.health_checker.evidence_logger") as mock_logger:

        mock_store.get_vectorstore.return_value = MagicMock()
        mock_store.as_retriever.return_value = MagicMock()
        mock_bm.get_baseline.return_value = None
        mock_logger.get_recent_events.return_value = []

        report = checker.check_health()
        assert report.status in ["HEALTHY", "DEGRADED"]
        assert report.summary != ""


def test_health_degraded_on_vectorstore_failure():
    checker = HealthChecker()
    with patch("core.health.health_checker.chroma_store") as mock_store, \
         patch("core.health.health_checker.baseline_manager") as mock_bm, \
         patch("core.health.health_checker.evidence_logger") as mock_logger:

        mock_store.get_vectorstore.side_effect = Exception("DB unavailable")
        mock_store.as_retriever.return_value = MagicMock()
        mock_bm.get_baseline.return_value = None
        mock_logger.get_recent_events.return_value = []

        report = checker.check_health()
        assert report.status == "DEGRADED"
        assert report.diagnostics.get("vectorstore_ok") is False


def test_health_degraded_on_excess_failures():
    checker = HealthChecker()
    with patch("core.health.health_checker.chroma_store") as mock_store, \
         patch("core.health.health_checker.baseline_manager") as mock_bm, \
         patch("core.health.health_checker.evidence_logger") as mock_logger:

        mock_store.get_vectorstore.return_value = MagicMock()
        mock_store.as_retriever.return_value = MagicMock()
        mock_bm.get_baseline.return_value = None
        mock_logger.get_recent_events.return_value = [
            {"status": "failed"} for _ in range(5)
        ]

        report = checker.check_health()
        assert report.status == "DEGRADED"
