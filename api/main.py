import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import structlog

from graph.workflow import run_healrag_pipeline
from caching.redis_cache import redis_cache
from ingestion.indexer import index_documents
from evaluation.ragas_evaluator import ragas_evaluator
from evaluation.baseline import baseline_manager
from evaluation.regression import regression_detector
from health.health_checker import health_checker
from healing.recovery import system_recovery
from evidence.evidence_logger import evidence_logger

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="HealRAG API",
    description="Self-Healing Retrieval-Augmented Generation API with LangGraph and RAGAS",
    version="1.0.0"
)

# Enable CORS for React frontend & dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request & Response Schemas
class QueryRequest(BaseModel):
    question: str = Field(..., example="What key failure modes affect standard RAG systems?")


class BaselineRequest(BaseModel):
    force_overwrite: bool = Field(default=False)


@app.get("/")
def read_root():
    return {
        "system": "HealRAG",
        "description": "Self-Healing RAG Architecture",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health")
def get_health():
    """GET /health - Returns current multi-component system health status."""
    report = health_checker.check_health()
    return report.dict()


@app.post("/query")
def process_query(request: QueryRequest):
    """POST /query - Executes HealRAG pipeline with self-healing retries."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Check semantic cache first
        cached = redis_cache.get_cached_response(request.question)
        if cached:
            return cached

        # Run pipeline if not cached
        response = run_healrag_pipeline(request.question)
        
        # Save to cache
        redis_cache.set_cached_response(request.question, response)
        
        return response
    except Exception as e:
        logger.error("error_processing_query", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
def trigger_ingestion():
    """POST /ingest - Triggers document loading, splitting, embedding, and indexing."""
    try:
        result = index_documents()
        return result
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
def trigger_evaluation():
    """POST /evaluate - Runs RAGAS evaluation against golden dataset."""
    try:
        metrics = ragas_evaluator.evaluate(run_healrag_pipeline)
        report = regression_detector.detect_regression(metrics)
        return {
            "metrics": metrics,
            "regression_report": report.dict()
        }
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation")
def get_evaluation():
    """GET /evaluation - Returns latest baseline metrics and regression state."""
    baseline = baseline_manager.get_baseline()
    if not baseline:
        return {
            "baseline": None,
            "status": "No baseline established. Run POST /evaluate first."
        }
    report = regression_detector.detect_regression(baseline)
    return {
        "baseline": baseline,
        "regression_report": report.dict()
    }


@app.post("/baseline")
def set_baseline(request: BaselineRequest):
    """POST /baseline - Explicitly saves or updates baseline evaluation metrics."""
    metrics = ragas_evaluator.evaluate(run_healrag_pipeline)
    res = baseline_manager.create_or_update_baseline(metrics, force_overwrite=request.force_overwrite)
    return res


@app.get("/evidence")
def get_evidence(limit: int = 50):
    """GET /evidence - Returns recent healing journal events."""
    events = evidence_logger.get_recent_events(limit=limit)
    return {
        "total_events": len(events),
        "events": events
    }


@app.post("/recover")
def trigger_recovery():
    """POST /recover - Triggers automated recovery workflow when health is DEGRADED."""
    health = health_checker.check_health()
    action = system_recovery.diagnose_issue(health.dict())
    result = system_recovery.execute_recovery(action)

    # Re-evaluate to verify recovery
    new_metrics = ragas_evaluator.evaluate(run_healrag_pipeline)
    new_health = health_checker.check_health()

    return {
        "recovery_result": result.dict(),
        "post_recovery_metrics": new_metrics,
        "post_recovery_health": new_health.dict(),
        "verified": new_health.status == "HEALTHY"
    }
