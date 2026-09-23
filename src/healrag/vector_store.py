"""
In-Memory Vector Database / Store module.
Provides indexing and cosine similarity vector search over text chunks.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from healrag.chunker import Chunk
from healrag.embedder import BaseEmbedder


class VectorStore:
    """
    In-memory vector store for chunk embeddings with Cosine Similarity search.
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None  # Shape (N, vector_dim)

    def add_chunks(self, chunks: List[Chunk], embedder: BaseEmbedder) -> None:
        """Embed and index a list of text chunks."""
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        new_vectors = embedder.embed_texts(texts)

        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
        
        self.chunks.extend(chunks)

    def similarity_search(
        self, query: str, embedder: BaseEmbedder, top_k: int = 3
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for the top_k chunks most similar to the user query.
        Returns list of tuples: (Chunk, similarity_score).
        """
        if not self.chunks or self.vectors is None:
            return []

        query_vector = embedder.embed_query(query)  # Shape (vector_dim,)

        # Cosine similarity (since vectors are L2-normalized, dot product equals cosine similarity)
        scores = np.dot(self.vectors, query_vector)

        # Get top-K indices sorted by score descending
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[Chunk, float]] = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(scores[idx])
            results.append((chunk, score))

        return results

    def clear(self) -> None:
        """Clear indexed chunks and vectors."""
        self.chunks = []
        self.vectors = None
