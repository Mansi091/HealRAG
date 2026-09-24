import os
import logging
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from core.embeddings.embedder import get_embedder
from config import settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Manages persistent ChromaDB vector store initialization, indexing, and retrieval."""

    def __init__(
        self,
        persist_dir: str = settings.CHROMA_PERSIST_DIR,
        collection_name: str = settings.COLLECTION_NAME
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = get_embedder()
        self._vectorstore: Optional[Chroma] = None

    def get_vectorstore(self) -> Chroma:
        """Returns lazy-initialized Chroma vector store instance."""
        if self._vectorstore is None:
            os.makedirs(self.persist_dir, exist_ok=True)
            logger.info(f"Loading ChromaDB store from {self.persist_dir}, collection: {self.collection_name}")
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedder,
                persist_directory=self.persist_dir
            )
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> Chroma:
        """Adds document chunks to ChromaDB and persists them."""
        os.makedirs(self.persist_dir, exist_ok=True)
        logger.info(f"Adding {len(documents)} document chunks to ChromaDB...")
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedder,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir
        )
        logger.info("ChromaDB indexing completed.")
        return self._vectorstore

    def similarity_search(self, query: str, top_k: int = settings.TOP_K) -> List[Document]:
        """Performs similarity search against ChromaDB vector index."""
        vs = self.get_vectorstore()
        return vs.similarity_search(query, k=top_k)

    def as_retriever(self, top_k: int = settings.TOP_K):
        """Exposes standard LangChain retriever interface."""
        vs = self.get_vectorstore()
        return vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

    def reset_collection(self):
        """Clears existing vectorstore collection for recovery operations."""
        logger.warning(f"Resetting ChromaDB collection: {self.collection_name}")
        vs = self.get_vectorstore()
        try:
            vs.delete_collection()
        except Exception as e:
            logger.warning(f"Collection reset warning: {e}")
        self._vectorstore = None


# Global singleton instance
chroma_store = ChromaVectorStore()
