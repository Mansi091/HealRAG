from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END

from core.graph.state import GraphState
from core.retrieval.retriever import DocumentRetriever
from core.retrieval.relevance_grader import RelevanceGrader
from core.retrieval.query_rewriter import QueryRewriter
from core.generation.generator import ResponseGenerator
from core.healing.hallucination_checker import HallucinationChecker
from core.healing.retry_manager import RetryManager
import structlog
from config import settings

logger = structlog.get_logger(__name__)


# Instantiate pipeline components
retriever = DocumentRetriever(top_k=settings.TOP_K)
relevance_grader = RelevanceGrader(threshold=settings.RELEVANCE_THRESHOLD)
query_rewriter = QueryRewriter()
generator = ResponseGenerator()
hallucination_checker = HallucinationChecker(threshold=settings.GROUNDING_THRESHOLD)
retry_manager = RetryManager(max_retries=settings.MAX_RETRIES)


# Define Nodes
def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """Retrieves document chunks using current query or rewritten query."""
    query = state.get("rewritten_question") or state["question"]
    logger.info(f"[Node: Retrieve] Fetching context for: '{query}'")
    documents = retriever.retrieve(query)
    return {"documents": documents}


def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """Grades relevance of retrieved documents."""
    query = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])
    logger.info(f"[Node: Grade Documents] Evaluating {len(documents)} documents...")

    result = relevance_grader.grade(query, documents)
    return {
        "relevant": result.relevant,
        "relevance_score": result.relevance_score,
        "documents": result.relevant_documents if result.relevant_documents else documents
    }


def rewrite_query_node(state: GraphState) -> Dict[str, Any]:
    """Rewrites query when retrieval quality is poor and records retry."""
    current_retry = state.get("retry_count", 0)
    query = state["question"]
    logger.info(f"[Node: Rewrite Query] Attempting query rewrite (retry_count={current_retry})...")

    record = retry_manager.record_retry(
        query=query,
        current_retry_count=current_retry,
        failure_type="poor_retrieval",
        action_taken="query_rewrite"
    )

    rewrite_res = query_rewriter.rewrite(query)
    return {
        "rewritten_question": rewrite_res.rewritten_query,
        "retry_count": record.attempt_number,
        "healing_action": "query_rewrite",
        "failure_type": "poor_retrieval"
    }


def generate_node(state: GraphState) -> Dict[str, Any]:
    """Generates grounded answer and extracts sources."""
    query = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])
    logger.info(f"[Node: Generate] Producing answer from {len(documents)} context docs...")

    answer, sources = generator.generate(query, documents)
    return {
        "answer": answer,
        "sources": sources
    }


def check_grounding_node(state: GraphState) -> Dict[str, Any]:
    """Checks whether generated answer is grounded in context."""
    query = state.get("rewritten_question") or state["question"]
    answer = state.get("answer", "")
    documents = state.get("documents", [])
    logger.info("[Node: Check Grounding] Verifying factual grounding...")

    res = hallucination_checker.check(query, answer, documents)
    status_str = "success" if res.grounded else "degraded"

    return {
        "grounded": res.grounded,
        "grounding_score": res.score,
        "status": status_str
    }


def retry_generation_node(state: GraphState) -> Dict[str, Any]:
    """Records retry for ungrounded generation."""
    current_retry = state.get("retry_count", 0)
    query = state["question"]
    logger.info(f"[Node: Retry Generation] Handling ungrounded answer (retry_count={current_retry})...")

    record = retry_manager.record_retry(
        query=query,
        current_retry_count=current_retry,
        failure_type="ungrounded_answer",
        action_taken="regenerate_answer"
    )

    return {
        "retry_count": record.attempt_number,
        "healing_action": "regenerate_answer",
        "failure_type": "ungrounded_answer"
    }


# Conditional Edge Router Functions
def decide_after_grading(state: GraphState) -> str:
    """Decides whether to proceed to generation or rewrite query."""
    if state.get("relevant", False) or state.get("retry_count", 0) >= settings.MAX_RETRIES:
        logger.info("Decide After Grading -> Proceeding to Generate")
        return "generate"
    else:
        logger.info("Decide After Grading -> Proceeding to Rewrite Query")
        return "rewrite_query"


def decide_after_grounding(state: GraphState) -> str:
    """Decides whether to conclude workflow or retry generation."""
    if state.get("grounded", False) or state.get("retry_count", 0) >= settings.MAX_RETRIES:
        logger.info("Decide After Grounding -> Ending Workflow")
        return END
    else:
        logger.info("Decide After Grounding -> Retrying Generation")
        return "retry_generation"


# Build LangGraph Workflow Graph
def create_healrag_workflow() -> StateGraph:
    """Constructs and compiles the complete HealRAG StateGraph."""
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("check_grounding", check_grounding_node)
    workflow.add_node("retry_generation", retry_generation_node)

    # Add Edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query"
        }
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", "check_grounding")

    workflow.add_conditional_edges(
        "check_grounding",
        decide_after_grounding,
        {
            END: END,
            "retry_generation": "retry_generation"
        }
    )

    workflow.add_edge("retry_generation", "generate")

    return workflow.compile()


# Compiled application workflow graph
healrag_graph = create_healrag_workflow()


def run_healrag_pipeline(question: str) -> Dict[str, Any]:
    """
    Executes the HealRAG LangGraph workflow for a user question.
    Returns structured API response.
    """
    initial_state: GraphState = {
        "question": question,
        "rewritten_question": None,
        "documents": [],
        "answer": None,
        "sources": [],
        "relevant": False,
        "grounded": False,
        "relevance_score": 0.0,
        "grounding_score": 0.0,
        "retry_count": 0,
        "healing_action": None,
        "failure_type": None,
        "status": "processing"
    }

    final_state = healrag_graph.invoke(initial_state)

    healing_triggered = final_state.get("retry_count", 0) > 0 or final_state.get("healing_action") is not None
    actions = [final_state["healing_action"]] if final_state.get("healing_action") else []

    return {
        "question": final_state["question"],
        "answer": final_state.get("answer", "No answer generated."),
        "sources": final_state.get("sources", []),
        "retrieval": {
            "relevance_score": final_state.get("relevance_score", 0.0),
            "relevant": final_state.get("relevant", False),
            "rewritten_query": final_state.get("rewritten_question")
        },
        "grounding": {
            "score": final_state.get("grounding_score", 0.0),
            "grounded": final_state.get("grounded", False)
        },
        "healing": {
            "triggered": healing_triggered,
            "actions": actions,
            "retry_count": final_state.get("retry_count", 0)
        },
        "status": "success" if final_state.get("grounded", False) else "completed_with_warnings"
    }
