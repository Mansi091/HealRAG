import logging
from typing import List
from langchain_core.documents import Document
from vectorstore.chroma_store import chroma_store
from config import settings

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Retrieves top-K relevant document chunks from ChromaDB for a given query."""

    def __init__(self, top_k: int = settings.TOP_K):
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Document]:
        """Performs vector similarity search against ChromaDB."""
        logger.info(f"Retrieving top {self.top_k} documents for query: '{query}'")
        try:
            docs = chroma_store.similarity_search(query, top_k=self.top_k)
            logger.info(f"Retrieved {len(docs)} document chunks.")
            return docs
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
