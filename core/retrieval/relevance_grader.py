import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from core.generation.prompts import RELEVANCE_GRADER_PROMPT
from config import settings

logger = logging.getLogger(__name__)


class RelevanceResult(BaseModel):
    """Structured relevance evaluation output."""
    relevant: bool = Field(description="Whether the document set is relevant to the question")
    relevance_score: float = Field(description="Confidence score between 0.0 and 1.0")
    reason: str = Field(description="Explanation of relevance judgment")
    relevant_documents: List[Document] = Field(default_factory=list)


class RelevanceGrader:
    """Grades retrieved document relevance against user query."""

    def __init__(self, threshold: float = settings.RELEVANCE_THRESHOLD):
        self.threshold = threshold
        llm_kwargs = {
            "model": settings.llm_model,
            "temperature": 0.0,
            "api_key": settings.api_key
        }
        if settings.base_url:
            llm_kwargs["openai_api_base"] = settings.base_url

        self.llm = ChatOpenAI(**llm_kwargs)
        self.chain = RELEVANCE_GRADER_PROMPT | self.llm

    def grade(self, question: str, documents: List[Document]) -> RelevanceResult:
        """Evaluates relevance of documents for question."""
        if not documents:
            logger.info("RelevanceGrader: No documents provided.")
            return RelevanceResult(
                relevant=False,
                relevance_score=0.0,
                reason="No documents were retrieved.",
                relevant_documents=[]
            )

        relevant_docs = []
        scores = []
        reasons = []

        for doc in documents:
            try:
                response = self.chain.invoke({
                    "question": question,
                    "document": doc.page_content
                })
                content = response.content.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                parsed = json.loads(content)
                is_rel = parsed.get("relevant", False)
                score = float(parsed.get("score", 0.5 if is_rel else 0.0))
                reason = parsed.get("reason", "")

                if is_rel or score >= self.threshold:
                    relevant_docs.append(doc)
                    scores.append(score)
                reasons.append(reason)
            except Exception as e:
                logger.warning(f"Error grading document relevance: {e}. Falling back to keyword check.")
                # Fallback heuristic
                q_words = set(question.lower().split())
                doc_words = set(doc.page_content.lower().split())
                overlap = len(q_words.intersection(doc_words)) / max(len(q_words), 1)
                if overlap > 0.1:
                    relevant_docs.append(doc)
                    scores.append(0.6)
                else:
                    scores.append(0.2)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        is_overall_relevant = len(relevant_docs) > 0 and avg_score >= self.threshold

        result = RelevanceResult(
            relevant=is_overall_relevant,
            relevance_score=round(avg_score, 2),
            reason="; ".join(reasons[:2]) if reasons else "Completed relevance evaluation.",
            relevant_documents=relevant_docs if relevant_docs else documents
        )
        logger.info(f"Relevance Grading complete: relevant={result.relevant}, score={result.relevance_score}")
        return result
