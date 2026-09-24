import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from core.ingestion.indexer import index_documents
from core.vectorstore.chroma_store import chroma_store
from core.evidence.evidence_logger import evidence_logger

logger = logging.getLogger(__name__)


class RecoveryResult(BaseModel):
    """Result of recovery workflow execution."""
    action_executed: str
    status: str
    diagnosis: str
    recovered: bool
    details: Dict[str, Any] = Field(default_factory=dict)


class SystemRecovery:
    """Orchestrates system-level self-healing recovery actions when health is DEGRADED."""

    def diagnose_issue(self, health_status: Dict[str, Any]) -> str:
        """Determines recovery action based on health diagnostic details."""
        diagnostics = health_status.get("diagnostics", {})

        if not diagnostics.get("vectorstore_ok", True):
            return "reset_vectorstore"
        elif not diagnostics.get("retriever_ok", True):
            return "rebuild_index"
        elif diagnostics.get("regression_detected", False):
            return "reindex_documents"
        else:
            return "reindex_documents"

    def execute_recovery(self, action: str) -> RecoveryResult:
        """Executes targeted recovery procedure."""
        logger.warning(f"SystemRecovery: Initiating recovery action '{action}'")
        try:
            if action == "reindex_documents" or action == "rebuild_index":
                logger.info("Executing document re-indexing recovery...")
                res = index_documents()
                recovered = res.get("status") == "success"
                details = res
                diagnosis = "Re-indexed document store to restore chunk precision."

            elif action == "reset_vectorstore":
                logger.info("Resetting vectorstore collection and re-indexing...")
                chroma_store.reset_collection()
                res = index_documents()
                recovered = res.get("status") == "success"
                details = res
                diagnosis = "Reset ChromaDB collection and rebuilt vector embeddings."

            else:
                logger.info(f"Fallback recovery executed for action '{action}'")
                res = index_documents()
                recovered = res.get("status") == "success"
                details = res
                diagnosis = f"Executed fallback re-indexing procedure for '{action}'."

            status_str = "success" if recovered else "failed"

            evidence_logger.log_event(
                query="SYSTEM_RECOVERY",
                failure_type="health_degradation",
                action=action,
                attempt=1,
                result=diagnosis,
                status=status_str
            )

            return RecoveryResult(
                action_executed=action,
                status=status_str,
                diagnosis=diagnosis,
                recovered=recovered,
                details=details
            )

        except Exception as e:
            logger.error(f"Error during system recovery execution: {e}")
            evidence_logger.log_event(
                query="SYSTEM_RECOVERY",
                failure_type="health_degradation",
                action=action,
                attempt=1,
                result=f"Recovery exception: {e}",
                status="failed"
            )
            return RecoveryResult(
                action_executed=action,
                status="failed",
                diagnosis=f"Recovery failed with error: {e}",
                recovered=False,
                details={"error": str(e)}
            )


# Global system recovery instance
system_recovery = SystemRecovery()
