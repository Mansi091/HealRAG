from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Explicit LangGraph state representing the entire lifecycle of a query through HealRAG.
    """
    question: str
    rewritten_question: Optional[str]
    documents: List[Document]
    answer: Optional[str]
    sources: List[Dict[str, Any]]
    relevant: bool
    grounded: bool
    relevance_score: float
    grounding_score: float
    retry_count: int
    healing_action: Optional[str]
    failure_type: Optional[str]
    status: str
