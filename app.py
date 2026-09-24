import sys
import argparse
import uvicorn
from api.main import app
from ingestion.indexer import index_documents
from graph.workflow import run_healrag_pipeline
from evaluation.ragas_evaluator import ragas_evaluator
from health.health_checker import health_checker
from healing.recovery import system_recovery


def main():
    parser = argparse.ArgumentParser(description="HealRAG: Self-Healing RAG Architecture")
    parser.add_argument("--server", action="store_true", help="Start FastAPI Uvicorn Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API Host")
    parser.add_argument("--port", type=int, default=8000, help="API Port")
    parser.add_argument("--ingest", action="store_true", help="Run document indexing")
    parser.add_argument("--query", type=str, help="Execute a query against HealRAG pipeline")
    parser.add_argument("--evaluate", action="store_true", help="Run golden evaluation")
    parser.add_argument("--health", action="store_true", help="Check system health")

    args = parser.parse_args()

    if args.ingest:
        print("Running Document Ingestion & Indexing...")
        res = index_documents()
        print(f"Result: {res}")
        return

    if args.query:
        print(f"Querying HealRAG: '{args.query}'")
        res = run_healrag_pipeline(args.query)
        print("\n--- RESPONSE ---")
        print(f"Answer:\n{res['answer']}\n")
        print(f"Sources: {res['sources']}")
        print(f"Retrieval Score: {res['retrieval']['relevance_score']}")
        print(f"Grounding Score: {res['grounding']['score']}")
        print(f"Healing Triggered: {res['healing']['triggered']}")
        return

    if args.evaluate:
        print("Running Evaluation against Golden Dataset...")
        metrics = ragas_evaluator.evaluate(run_healrag_pipeline)
        print(f"Metrics: {metrics}")
        return

    if args.health:
        print("Checking System Health...")
        report = health_checker.check_health()
        print(f"Status: {report.status}")
        print(f"Summary: {report.summary}")
        print(f"Diagnostics: {report.diagnostics}")
        return

    # Default: Start Uvicorn Server
    print(f"Starting HealRAG FastAPI Server on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
