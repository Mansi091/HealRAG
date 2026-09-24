from typing import List, Optional
from langchain_core.documents import Document
from config import settings
import structlog

logger = structlog.get_logger(__name__)

class PineconeStore:
    """Singleton wrapper for Pinecone Vector Store."""
    _instance = None
    _vectorstore = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PineconeStore, cls).__new__(cls)
        return cls._instance

    def _initialize(self):
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec
        from core.embeddings.embedder import get_embedder

        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set.")
        
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index_name = settings.PINECONE_INDEX_NAME

        if index_name not in pc.list_indexes().names():
            logger.info("creating_pinecone_index", index_name=index_name)
            pc.create_index(
                name=index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        
        self._vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=get_embedder(),
            pinecone_api_key=settings.PINECONE_API_KEY
        )

    def get_vectorstore(self):
        if self._vectorstore is None:
            self._initialize()
        return self._vectorstore

    def as_retriever(self, **kwargs):
        return self.get_vectorstore().as_retriever(**kwargs)

    def add_documents(self, documents: List[Document]):
        logger.info("adding_documents_to_pinecone", count=len(documents))
        self.get_vectorstore().add_documents(documents)

    def similarity_search(self, query: str, k: int = 4):
        return self.get_vectorstore().similarity_search(query, k=k)

pinecone_store = PineconeStore()

