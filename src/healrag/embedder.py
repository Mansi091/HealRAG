from abc import ABC, abstractmethod
from typing import List
import math
import re
import numpy as np


class BaseEmbedder(ABC):

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        pass


class SimpleTFIDFEmbedder(BaseEmbedder):

    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size

    def _text_to_vector(self, text: str) -> np.ndarray:
        text_clean = text.lower()
        words = re.findall(r'\w+', text_clean)
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        
        if not words:
            return vec

        for word in words:
            h = hash(word) % self.vocab_size
            vec[h] += 1.0
            
            for i in range(len(word) - 2):
                trigram = word[i:i+3]
                h_tri = hash(trigram) % self.vocab_size
                vec[h_tri] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = [self._text_to_vector(t) for t in texts]
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._text_to_vector(query)


class OpenAIEmbedder(BaseEmbedder):

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required for OpenAIEmbedder.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        
        response = self.client.embeddings.create(input=texts, model=self.model)
        embeddings = [item.embedding for item in response.data]
        vecs = np.array(embeddings, dtype=np.float32)
        
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_query(self, query: str) -> np.ndarray:
        res = self.embed_texts([query])
        return res[0]
