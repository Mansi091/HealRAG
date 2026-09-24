from config import settings

def _get_store():
    if settings.VECTOR_STORE_TYPE == "pinecone":
        from .pinecone_store import pinecone_store
        return pinecone_store
    else:
        from .chroma_store import ChromaVectorStore, chroma_store
        return chroma_store

# Lazy-load: only imports the selected backend at first access
chroma_store = _get_store()
active_store = chroma_store
