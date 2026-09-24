import os
import json
import logging
from typing import Dict, Any, Optional
from config import settings

logger = logging.getLogger(__name__)


class BaselineManager:
    """Manages creation, loading, and explicit updating of baseline quality metrics."""

    def __init__(self, baseline_path: str = settings.BASELINE_PATH):
        self.baseline_path = baseline_path
        os.makedirs(os.path.dirname(self.baseline_path), exist_ok=True)

    def get_baseline(self) -> Optional[Dict[str, float]]:
        """Loads baseline metrics from disk if present."""
        if not os.path.exists(self.baseline_path):
            return None
        try:
            with open(self.baseline_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading baseline metrics: {e}")
            return None

    def create_or_update_baseline(
        self,
        metrics: Dict[str, float],
        force_overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Saves given evaluation metrics as baseline.
        Requires force_overwrite=True if baseline already exists to prevent accidental overwrites.
        """
        existing = self.get_baseline()
        if existing and not force_overwrite:
            logger.warning("Baseline already exists. Set force_overwrite=True to update.")
            return {
                "status": "warning",
                "message": "Baseline already exists. Use force_overwrite to update.",
                "baseline": existing
            }

        try:
            with open(self.baseline_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Baseline created/updated successfully at {self.baseline_path}: {metrics}")
            return {
                "status": "success",
                "message": "Baseline created/updated successfully",
                "baseline": metrics
            }
        except Exception as e:
            logger.error(f"Failed to write baseline: {e}")
            return {
                "status": "error",
                "message": str(e)
            }


# Global baseline manager instance
baseline_manager = BaselineManager()
