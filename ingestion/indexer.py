import logging
from typing import Dict, Any
from .loader import DocumentLoader
from .splitter import DocumentSplitter
from vectorstore.chroma_store import chroma_store

logger = logging.getLogger(__name__)


def index_documents(docs_dir: str = None) -> Dict[str, Any]:
    """
    Independent workflow function: loads, splits, embeds, and indexes documents into ChromaDB.
    Returns indexing statistics.
    """
    logger.info("Starting document indexing workflow...")
    loader = DocumentLoader(docs_dir=docs_dir) if docs_dir else DocumentLoader()
    splitter = DocumentSplitter()

    documents = loader.load_documents()
    if not documents:
        logger.warning("No documents found to index.")
        return {
            "status": "warning",
            "message": "No documents found",
            "documents_loaded": 0,
            "chunks_indexed": 0
        }

    chunks = splitter.split_documents(documents)
    chroma_store.add_documents(chunks)

    result = {
        "status": "success",
        "documents_loaded": len(documents),
        "chunks_indexed": len(chunks)
    }
    logger.info(f"Indexing completed successfully: {result}")
    return result
