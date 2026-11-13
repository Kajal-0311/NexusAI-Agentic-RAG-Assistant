"""
Settings inspection and configuration endpoints.
"""
from typing import Any, Dict
from fastapi import APIRouter
from backend.api.schemas import SettingsUpdateRequest
from backend.core.config import settings

router = APIRouter(prefix="/settings", tags=["Settings"])


@router.get("", response_model=Dict[str, Any])
def get_current_settings():
    """Returns active runtime configuration."""
    return {
        "app_name": settings.APP_NAME,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "ollama_llm_model": settings.OLLAMA_LLM_MODEL,
        "ollama_embed_model": settings.OLLAMA_EMBED_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "default_retrieval_mode": settings.DEFAULT_RETRIEVAL_MODE,
        "default_top_k": settings.DEFAULT_TOP_K,
        "bm25_k1": settings.BM25_K1,
        "bm25_b": settings.BM25_B,
        "rrf_k": settings.RRF_K,
        "use_reranker": settings.USE_RERANKER,
        "reranker_model": settings.RERANKER_MODEL,
        "max_agent_retries": settings.MAX_AGENT_RETRIES,
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
    }


@router.post("", response_model=Dict[str, Any])
def update_settings(req: SettingsUpdateRequest):
    """Updates runtime configuration settings."""
    if req.chunk_size is not None:
        settings.CHUNK_SIZE = req.chunk_size
    if req.chunk_overlap is not None:
        settings.CHUNK_OVERLAP = req.chunk_overlap
    if req.default_retrieval_mode is not None:
        settings.DEFAULT_RETRIEVAL_MODE = req.default_retrieval_mode  # type: ignore
    if req.default_top_k is not None:
        settings.DEFAULT_TOP_K = req.default_top_k
    if req.use_reranker is not None:
        settings.USE_RERANKER = req.use_reranker
    if req.ollama_model is not None:
        settings.OLLAMA_LLM_MODEL = req.ollama_model

    return get_current_settings()
