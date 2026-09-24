import logging
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from core.generation.prompts import ANSWER_GENERATION_PROMPT
from config import settings

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates answers strictly from supplied context using configurable LLM."""

    def __init__(self):
        llm_kwargs = {
            "model": settings.llm_model,
            "temperature": 0.0,
            "api_key": settings.api_key
        }
        if settings.base_url:
            llm_kwargs["openai_api_base"] = settings.base_url

        self.llm = ChatOpenAI(**llm_kwargs)
        self.output_parser = StrOutputParser()
        self.chain = ANSWER_GENERATION_PROMPT | self.llm | self.output_parser

    def _format_context(self, documents: List[Document]) -> str:
        """Formats document list into a clear text context block with metadata headers."""
        if not documents:
            return "No relevant context available."

        formatted_chunks = []
        for d in documents:
            src = d.metadata.get("filename") or d.metadata.get("source", "Unknown")
            pg = d.metadata.get("page", 0) + 1
            formatted_chunks.append(f"[Source: {src} (Page {pg})]\n{d.page_content}")

        return "\n\n".join(formatted_chunks)

    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extracts unique source metadata from retrieved documents."""
        sources = []
        seen = set()

        for d in documents:
            src = d.metadata.get("filename") or d.metadata.get("source", "Unknown")
            pg = d.metadata.get("page", 0) + 1
            key = f"{src}_p{pg}"
            if key not in seen:
                seen.add(key)
                sources.append({"source": src, "page": pg})

        return sources

    def generate(self, question: str, documents: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
        """Generates grounded answer and returns answer string along with source citations."""
        logger.info(f"Generating answer for question: '{question}' using {len(documents)} context docs.")
        context_str = self._format_context(documents)

        try:
            answer = self.chain.invoke({
                "context": context_str,
                "question": question
            })
        except Exception as e:
            logger.error(f"Error invoking LLM generator chain: {e}")
            answer = "I encountered an error while attempting to generate an answer."

        sources = self._extract_sources(documents)
        return answer, sources
