import json
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from generation.prompts import GROUNDING_CHECKER_PROMPT
from config import settings

logger = logging.getLogger(__name__)


class GroundingResult(BaseModel):
    """Structured grounding / hallucination evaluation result."""
    grounded: bool = Field(description="Whether the answer is grounded in context")
    score: float = Field(description="Grounding confidence score between 0.0 and 1.0")
    reason: str = Field(description="Explanation of grounding decision")


class HallucinationChecker:
    """Verifies whether generated answers are strictly grounded in retrieved context."""

    def __init__(self, threshold: float = settings.GROUNDING_THRESHOLD):
        self.threshold = threshold
        llm_kwargs = {
            "model": settings.llm_model,
            "temperature": 0.0,
            "api_key": settings.api_key
        }
        if settings.base_url:
            llm_kwargs["openai_api_base"] = settings.base_url

        self.llm = ChatOpenAI(**llm_kwargs)
        self.chain = GROUNDING_CHECKER_PROMPT | self.llm

    def check(self, question: str, answer: str, documents: List[Document]) -> GroundingResult:
        """Determines if answer is supported by context documents."""
        if not documents:
            logger.info("HallucinationChecker: No documents available to verify grounding.")
            return GroundingResult(
                grounded=False,
                score=0.0,
                reason="No context documents were available for grounding check."
            )

        context_str = "\n\n".join([f"Snippet: {d.page_content}" for d in documents])

        try:
            response = self.chain.invoke({
                "context": context_str,
                "question": question,
                "answer": answer
            })
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(content)
            is_grounded = parsed.get("grounded", False)
            score = float(parsed.get("score", 0.8 if is_grounded else 0.2))
            reason = parsed.get("reason", "Grounding check complete.")

            is_valid = is_grounded and score >= self.threshold

            result = GroundingResult(
                grounded=is_valid,
                score=round(score, 2),
                reason=reason
            )
        except Exception as e:
            logger.warning(f"Grounding check error: {e}. Falling back to keyword overlap.")
            # Fallback check
            ans_words = set(answer.lower().split())
            ctx_words = set(context_str.lower().split())
            overlap = len(ans_words.intersection(ctx_words)) / max(len(ans_words), 1)
            is_valid = overlap > 0.3
            result = GroundingResult(
                grounded=is_valid,
                score=round(overlap, 2),
                reason="Fallback heuristic overlap evaluation."
            )

        logger.info(f"Grounding Check complete: grounded={result.grounded}, score={result.score}")
        return result
