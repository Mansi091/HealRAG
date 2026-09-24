import json
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from generation.prompts import QUERY_REWRITER_PROMPT
from config import settings

logger = logging.getLogger(__name__)


class RewriteResult(BaseModel):
    """Structured query rewrite output."""
    original_query: str
    rewritten_query: str
    reason: str


class QueryRewriter:
    """Optimizes user query for improved vector store retrieval when initial retrieval fails."""

    def __init__(self):
        llm_kwargs = {
            "model": settings.llm_model,
            "temperature": 0.2,
            "api_key": settings.api_key
        }
        if settings.base_url:
            llm_kwargs["openai_api_base"] = settings.base_url

        self.llm = ChatOpenAI(**llm_kwargs)
        self.chain = QUERY_REWRITER_PROMPT | self.llm

    def rewrite(self, question: str) -> RewriteResult:
        """Rewrites question for better semantic keyword matching."""
        logger.info(f"QueryRewriter: Optimizing question '{question}'")
        try:
            response = self.chain.invoke({"question": question})
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(content)
            rewritten = parsed.get("rewritten_query", question)
            reason = parsed.get("reason", "Query optimized for vector retrieval.")
        except Exception as e:
            logger.warning(f"Error during query rewriting: {e}. Applying fallback query expansion.")
            rewritten = f"{question} key concepts detailed summary"
            reason = "Applied fallback keyword expansion."

        logger.info(f"Rewritten query: '{rewritten}' (Reason: {reason})")
        return RewriteResult(
            original_query=question,
            rewritten_query=rewritten,
            reason=reason
        )
