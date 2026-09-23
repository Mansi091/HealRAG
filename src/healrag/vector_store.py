from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from healrag.chunker import Chunk
from healrag.embedder import BaseEmbedder


class VectorStore:

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[Chunk], embedder: BaseEmbedder) -> None:
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
        if not self.chunks or self.vectors is None:
            return []

        query_vector = embedder.embed_query(query)
        scores = np.dot(self.vectors, query_vector)

        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[Chunk, float]] = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(scores[idx])
            results.append((chunk, score))

        return results

    def clear(self) -> None:
        self.chunks = []
        self.vectors = None
