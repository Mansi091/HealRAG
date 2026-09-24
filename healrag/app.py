import sys
from healrag.config import API_KEY, BASE_URL, DOCS_DIR, CHROMA_DB_DIR, COLLECTION_NAME, MODEL_NAME, TOP_K
from healrag.ingestion import load_documents, split_documents, get_embedding_model
from healrag.vectorstore import create_vectorstore
from healrag.retrieval import get_retriever
from healrag.generation import build_rag_chain

def main():
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    if not API_KEY:
        raise ValueError("API Key missing! Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env")

    print("Running Simple RAG Pipeline...")

    # 1. Ingestion
    documents = load_documents(DOCS_DIR)
    chunks = split_documents(documents)
    embeddings = get_embedding_model()

    # 2. Vectorstore & Retriever
    vectorstore = create_vectorstore(chunks, embeddings, COLLECTION_NAME, CHROMA_DB_DIR)
    retriever = get_retriever(vectorstore, top_k=TOP_K)

    # 3. Generation Chain
    chain = build_rag_chain(retriever, model_name=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL)

    # 4. Query & Output
    question = "What is artificial intelligence?"
    print(f"\n[QUESTION] {question}")

    retrieved_docs = retriever.invoke(question)
    answer = chain.invoke(question)

    print(f"\n[ANSWER]\n{answer}")
    print("\n[SOURCES]")
    for d in retrieved_docs:
        print(f"  * Source: {d.metadata.get('source', 'Unknown')} (Page {d.metadata.get('page', 0) + 1})")

if __name__ == "__main__":
    main()
