import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from config import settings

logger = logging.getLogger(__name__)


class EvidenceLogger:
    """Logs healing actions and system events into a persistent JSONL journal."""

    def __init__(self, journal_path: str = settings.JOURNAL_PATH):
        self.journal_path = journal_path
        os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)

    def log_event(
        self,
        query: str,
        failure_type: str,
        action: str,
        attempt: int,
        result: str,
        status: str = "success"
    ) -> Dict[str, Any]:
        """Appends a structured event to the journal.jsonl file."""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "failure_type": failure_type,
            "action": action,
            "attempt": attempt,
            "result": result,
            "status": status
        }

        try:
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            logger.info(f"Evidence logged: action='{action}', failure_type='{failure_type}', attempt={attempt}")
        except Exception as e:
            logger.error(f"Failed to write evidence log: {e}")

        return event

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Reads recent events from the JSONL log file."""
        if not os.path.exists(self.journal_path):
            return []

        events = []
        try:
            with open(self.journal_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error reading journal logs: {e}")

        return events


# Global evidence logger instance
evidence_logger = EvidenceLogger()
