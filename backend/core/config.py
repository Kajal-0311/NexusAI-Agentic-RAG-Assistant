"""
Configuration settings for NexusAI.
Supports loading from environment variables and .env file.
"""
from pathlib import Path
from typing import Literal
import os

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore

# Base repository directory (parent of backend)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings schema with sensible production defaults."""
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General
    APP_NAME: str = "NexusAI - Agentic RAG Research Assistant"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 8501

    # Storage Paths
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOADS_DIR: Path = BASE_DIR / "data" / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    EVALUATION_DIR: Path = BASE_DIR / "data" / "evaluation"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "data" / "vectorstore"
    SQLITE_DB_PATH: Path = BASE_DIR / "data" / "nexus_ai.db"

    # Ollama LLM & Embeddings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "llama3"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_TIMEOUT_SECONDS: float = 60.0
    FALLBACK_EMBED_DIM: int = 384  # nomic-embed-text/minilm fallback vector dimension

    # Document Ingestion & Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: list[str] = [".pdf"]

    # ChromaDB Vector Store
    COLLECTION_NAME: str = "nexus_knowledge_base"

    # Retrieval Engine Settings
    DEFAULT_RETRIEVAL_MODE: Literal["semantic", "keyword", "hybrid"] = "hybrid"
    DEFAULT_TOP_K: int = 5
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    RRF_K: int = 60
    USE_RERANKER: bool = False
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Agent Safeguards & Control
    MAX_AGENT_RETRIES: int = 2
    CONFIDENCE_THRESHOLD: float = 0.60
    EVIDENCE_SUFFICIENT_MIN_SCORE: float = 0.015  # For RRF scores

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for path in [
            self.DATA_DIR,
            self.UPLOADS_DIR,
            self.PROCESSED_DIR,
            self.EVALUATION_DIR,
            self.CHROMA_PERSIST_DIR,
            self.SQLITE_DB_PATH.parent
        ]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
