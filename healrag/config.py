import os
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = "data/documents"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "rag_production"
MODEL_NAME = "openai/gpt-4o-mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4

API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
