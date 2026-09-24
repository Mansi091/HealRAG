import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from core.vectorstore.chroma_store import chroma_store
from core.evaluation.baseline import baseline_manager
from core.evaluation.regression import regression_detector
from core.evidence.evidence_logger import evidence_logger

logger = logging.getLogger(__name__)


class HealthReport(BaseModel):
    """Structured overall system health report."""
    status: str = Field(description="'HEALTHY' or 'DEGRADED'")
    diagnostics: Dict[str, Any]
    summary: str


class HealthChecker:
    """Evaluates multi-component operational health and regression state of HealRAG."""

    def check_health(self) -> HealthReport:
        """Runs diagnostics across vector store, retriever, evaluation metrics, and evidence logs."""
        diagnostics = {}
        issues = []

        # 1. Vectorstore check
        try:
            vs = chroma_store.get_vectorstore()
            diagnostics["vectorstore_ok"] = vs is not None
        except Exception as e:
            diagnostics["vectorstore_ok"] = False
            issues.append(f"Vectorstore failure: {e}")

        # 2. Retriever check
        try:
            retriever = chroma_store.as_retriever()
            diagnostics["retriever_ok"] = retriever is not None
        except Exception as e:
            diagnostics["retriever_ok"] = False
            issues.append(f"Retriever failure: {e}")

        # 3. Baseline & Regression status check
        baseline = baseline_manager.get_baseline()
        if baseline:
            report = regression_detector.detect_regression(baseline)
            diagnostics["regression_status"] = report.status
            diagnostics["regression_detected"] = report.regression_detected
            if report.regression_detected:
                issues.append("Regression detected in evaluation metrics.")
        else:
            diagnostics["regression_status"] = "HEALTHY"
            diagnostics["regression_detected"] = False

        # 4. Recent healing failure check
        recent_events = evidence_logger.get_recent_events(limit=20)
        failed_count = sum(1 for e in recent_events if e.get("status") == "failed")
        diagnostics["recent_failed_healings"] = failed_count
        if failed_count > 3:
            issues.append(f"Excessive recent healing failures ({failed_count}).")

        # Determine overall health status
        is_healthy = len(issues) == 0 and diagnostics.get("vectorstore_ok", True) and diagnostics.get("retriever_ok", True)
        status = "HEALTHY" if is_healthy else "DEGRADED"
        summary = "All components operating normally." if is_healthy else "; ".join(issues)

        logger.info(f"HealthChecker: status={status}, summary='{summary}'")
        return HealthReport(
            status=status,
            diagnostics=diagnostics,
            summary=summary
        )


# Global health checker instance
health_checker = HealthChecker()
