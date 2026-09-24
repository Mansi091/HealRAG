import pytest
from evaluation.regression import RegressionDetector
from evaluation.baseline import BaselineManager


@pytest.fixture
def sample_metrics():
    return {
        "context_precision": 0.82,
        "context_recall": 0.79,
        "faithfulness": 0.91,
        "answer_relevancy": 0.88
    }


def test_no_regression_when_metrics_match(sample_metrics):
    detector = RegressionDetector(tolerance=0.05)
    report = detector.detect_regression(
        current_metrics=sample_metrics,
        custom_baseline=sample_metrics
    )
    assert report.status == "HEALTHY"
    assert report.regression_detected is False


def test_regression_detected_on_drop(sample_metrics):
    degraded = {
        "context_precision": 0.60,
        "context_recall": 0.58,
        "faithfulness": 0.65,
        "answer_relevancy": 0.60
    }
    detector = RegressionDetector(tolerance=0.05)
    report = detector.detect_regression(
        current_metrics=degraded,
        custom_baseline=sample_metrics
    )
    assert report.status == "DEGRADED"
    assert report.regression_detected is True
    assert len(report.degraded_metrics) > 0


def test_regression_within_tolerance(sample_metrics):
    slightly_lower = {k: v - 0.03 for k, v in sample_metrics.items()}
    detector = RegressionDetector(tolerance=0.05)
    report = detector.detect_regression(
        current_metrics=slightly_lower,
        custom_baseline=sample_metrics
    )
    assert report.status == "HEALTHY"


def test_baseline_save_and_load(tmp_path, sample_metrics):
    path = str(tmp_path / "baseline.json")
    manager = BaselineManager(baseline_path=path)

    res = manager.create_or_update_baseline(sample_metrics)
    assert res["status"] == "success"

    loaded = manager.get_baseline()
    assert loaded is not None
    assert loaded["faithfulness"] == 0.91


def test_baseline_prevents_accidental_overwrite(tmp_path, sample_metrics):
    path = str(tmp_path / "baseline.json")
    manager = BaselineManager(baseline_path=path)
    manager.create_or_update_baseline(sample_metrics)

    # Attempt overwrite without force
    res = manager.create_or_update_baseline({"faithfulness": 0.5})
    assert res["status"] == "warning"
