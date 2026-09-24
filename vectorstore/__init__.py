from config import settings

if settings.VECTOR_STORE_TYPE == "pinecone":
    from .pinecone_store import pinecone_store as active_store
else:
    from .chroma_store import chroma_store as active_store

# Create alias for compatibility with existing code
chroma_store = active_store
