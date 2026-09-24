import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from evidence.evidence_logger import evidence_logger
from config import settings

logger = logging.getLogger(__name__)


class RetryRecord(BaseModel):
    """Record of a single retry execution."""
    attempt_number: int
    failure_type: str
    action_taken: str
    can_retry: bool


class RetryManager:
    """Manages bounded retry attempts to guarantee infinite loop protection."""

    def __init__(self, max_retries: int = settings.MAX_RETRIES):
        self.max_retries = max_retries

    def can_retry(self, current_retry_count: int) -> bool:
        """Determines if additional retry attempts are allowed within MAX_RETRIES bound."""
        return current_retry_count < self.max_retries

    def record_retry(
        self,
        query: str,
        current_retry_count: int,
        failure_type: str,
        action_taken: str
    ) -> RetryRecord:
        """Records retry attempt, logs evidence, and returns retry state."""
        new_count = current_retry_count + 1
        can_retry_further = new_count < self.max_retries

        logger.warning(
            f"RetryManager: Attempt {new_count}/{self.max_retries} for failure '{failure_type}'. "
            f"Action: '{action_taken}'"
        )

        evidence_logger.log_event(
            query=query,
            failure_type=failure_type,
            action=action_taken,
            attempt=new_count,
            result="retry_initiated" if can_retry_further else "max_retries_reached",
            status="active" if can_retry_further else "bounded_stop"
        )

        return RetryRecord(
            attempt_number=new_count,
            failure_type=failure_type,
            action_taken=action_taken,
            can_retry=can_retry_further
        )
