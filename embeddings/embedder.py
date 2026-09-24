import logging
from langchain_core.embeddings import Embeddings
from config import settings

logger = logging.getLogger(__name__)


def get_embedder() -> Embeddings:
    """
    Factory function returning the configured pre-trained embedding model.
    Supports OpenAIEmbeddings or local HuggingFaceEmbeddings based on settings.
    """
    model_name = settings.EMBEDDING_MODEL

    if model_name.startswith("text-embedding") or "openai" in model_name.lower():
        from langchain_openai import OpenAIEmbeddings
        embed_kwargs = {
            "model": model_name,
            "api_key": settings.api_key
        }
        if settings.base_url:
            embed_kwargs["openai_api_base"] = settings.base_url
        logger.info(f"Initializing OpenAIEmbeddings with model: {model_name}")
        return OpenAIEmbeddings(**embed_kwargs)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info(f"Initializing HuggingFaceEmbeddings with model: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)
