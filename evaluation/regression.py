import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from evaluation.baseline import baseline_manager
from config import settings

logger = logging.getLogger(__name__)


class RegressionReport(BaseModel):
    """Structured regression detection report."""
    status: str = Field(description="'HEALTHY' or 'DEGRADED'")
    regression_detected: bool
    tolerated_drop: float
    current_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    degraded_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    summary: str


class RegressionDetector:
    """Detects quality regressions by comparing current evaluation against baseline metrics."""

    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance

    def detect_regression(
        self,
        current_metrics: Dict[str, float],
        custom_baseline: Dict[str, float] = None
    ) -> RegressionReport:
        """
        Compares current metrics against baseline metrics.
        Returns RegressionReport with HEALTHY or DEGRADED status.
        """
        baseline = custom_baseline or baseline_manager.get_baseline()

        if not baseline:
            logger.info("No baseline found. Initializing current metrics as baseline.")
            baseline_manager.create_or_update_baseline(current_metrics, force_overwrite=True)
            baseline = current_metrics

        degraded = {}
        is_degraded = False

        for metric_name, current_val in current_metrics.items():
            base_val = baseline.get(metric_name, current_val)
            allowed_threshold = base_val - self.tolerance

            if current_val < allowed_threshold:
                is_degraded = True
                degraded[metric_name] = {
                    "current": current_val,
                    "baseline": base_val,
                    "drop": round(base_val - current_val, 3)
                }

        status = "DEGRADED" if is_degraded else "HEALTHY"
        summary = (
            f"Regression detected in metrics: {list(degraded.keys())}"
            if is_degraded
            else "All evaluation metrics are within baseline tolerance."
        )

        logger.info(f"Regression Check result: status={status}, summary='{summary}'")
        return RegressionReport(
            status=status,
            regression_detected=is_degraded,
            tolerated_drop=self.tolerance,
            current_metrics=current_metrics,
            baseline_metrics=baseline,
            degraded_metrics=degraded,
            summary=summary
        )


# Global regression detector instance
regression_detector = RegressionDetector()
