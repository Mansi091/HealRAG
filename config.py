import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class Settings(BaseModel):
    """Centralized configuration for HealRAG system."""

    # API Keys & Endpoints
    OPENROUTER_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    OPENROUTER_MODEL: str = Field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
    OPENAI_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    # Embedding Settings
    EMBEDDING_MODEL: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    # Vectorstore Settings
    VECTOR_STORE_TYPE: str = Field(default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "chroma")) # 'chroma' or 'pinecone'
    CHROMA_PERSIST_DIR: str = Field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
    COLLECTION_NAME: str = Field(default_factory=lambda: os.getenv("COLLECTION_NAME", "healrag_production"))
    PINECONE_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("PINECONE_API_KEY"))
    PINECONE_INDEX_NAME: str = Field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "healrag-index"))

    # Caching Settings
    REDIS_URL: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    # Chunking & Retrieval Parameters
    TOP_K: int = Field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    CHUNK_SIZE: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    CHUNK_OVERLAP: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))

    # Thresholds & Limits
    MAX_RETRIES: int = Field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "2")))
    RELEVANCE_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("RELEVANCE_THRESHOLD", "0.5")))
    GROUNDING_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("GROUNDING_THRESHOLD", "0.7")))
    RAGAS_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("RAGAS_THRESHOLD", "0.7")))

    # Paths
    DOCS_DIR: str = Field(default="data/documents")
    GOLDEN_DATASET_PATH: str = Field(default="data/golden/golden_dataset.json")
    BASELINE_PATH: str = Field(default="data/golden/baseline.json")
    JOURNAL_PATH: str = Field(default="evidence/journal.jsonl")

    @property
    def api_key(self) -> str:
        key = self.OPENROUTER_API_KEY or self.OPENAI_API_KEY or ""
        return key

    @property
    def base_url(self) -> Optional[str]:
        if self.OPENROUTER_API_KEY:
            return "https://openrouter.ai/api/v1"
        return None

    @property
    def llm_model(self) -> str:
        if self.base_url:
            return self.OPENROUTER_MODEL
        return "gpt-4o-mini"


# Global settings instance
settings = Settings()
