"""
Health check and telemetry endpoints.
"""
from fastapi import APIRouter
from backend.api.schemas import HealthResponse
from backend.database.database import get_db
from backend.generation.llm_service import get_llm_service
from backend.vectorstore.chroma_store import get_vector_store

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_system_health():
    """Returns comprehensive system health status across LLM, vector store, and SQLite."""
    llm = get_llm_service()
    vector_store = get_vector_store()
    db = get_db()

    ollama_status = llm.check_health()
    vector_status = vector_store.get_stats()

    # Check SQLite connection
    db_status = "connected"
    try:
        db.get_all_documents()
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    overall_status = "healthy"
    if not ollama_status.get("is_running"):
        overall_status = "degraded (ollama offline - fallback engine active)"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        ollama_status=ollama_status,
        vector_store_status=vector_status,
        database_status=db_status,
    )
