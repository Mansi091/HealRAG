from langchain_chroma import Chroma

def create_vectorstore(chunks, embeddings, collection_name: str = "rag_production", persist_dir: str = "./chroma_db"):
    """Indexes document chunks into a ChromaDB vector store."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
