"""
Complete RAG Pipeline Orchestrator.
Manages Document Ingestion, Retrieval, Context Augmentation, and LLM Generation.
"""
from typing import List, Dict, Any, Optional, Tuple
from healrag.chunker import Document, Chunk, TextChunker
from healrag.embedder import BaseEmbedder, SimpleTFIDFEmbedder, OpenAIEmbedder
from healrag.vector_store import VectorStore


class RAGPipeline:
    """
    RAG Pipeline unifying Ingestion, Vector Indexing, Similarity Search, and Grounded Generation.
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = embedder or SimpleTFIDFEmbedder()
        self.vector_store = VectorStore()
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def ingest_documents(self, documents: List[Document]) -> int:
        """Chunk documents and add them to the vector store."""
        chunks = self.chunker.split_documents(documents)
        self.vector_store.add_chunks(chunks, self.embedder)
        return len(chunks)

    def ingest_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> int:
        """Helper to quickly ingest raw text strings."""
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {"source": f"doc_{i+1}"}
            docs.append(Document(text=text, metadata=meta))
        return self.ingest_documents(docs)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Retrieve top-K context chunks matching the query."""
        return self.vector_store.similarity_search(query, self.embedder, top_k=top_k)

    def build_prompt(self, query: str, retrieved_chunks: List[Tuple[Chunk, float]]) -> str:
        """Assemble the context-augmented prompt for the LLM."""
        context_blocks = []
        for idx, (chunk, score) in enumerate(retrieved_chunks, 1):
            src = chunk.metadata.get("source", "Unknown Source")
            context_blocks.append(
                f"[Source {idx}: {src} | Similarity: {score:.3f}]\n{chunk.text}"
            )
        
        context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context below.
If the information required to answer the question is not present in the context, explicitly state that you do not have enough information.

--- CONTEXT START ---
{context_str}
--- CONTEXT END ---

User Question: {query}
Answer:"""
        return prompt

    def query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Full RAG Pipeline execution:
        1. Retrieve top-K relevant chunks
        2. Construct augmented prompt
        3. Call LLM (OpenRouter/OpenAI API if key provided, otherwise return prompt and context)
        """
        retrieved = self.retrieve(query, top_k=top_k)
        prompt = self.build_prompt(query, retrieved)

        answer = ""
        if self.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                answer = response.choices[0].message.content or ""
            except Exception as e:
                answer = f"[LLM Call Error: {str(e)}]\n\nGenerated Prompt:\n{prompt}"
        else:
            answer = (
                "[Offline Mode - No API Key provided]\n"
                "The pipeline successfully retrieved relevant contexts and constructed the grounded prompt below:\n\n"
                f"{prompt}"
            )

        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "prompt": prompt
        }
