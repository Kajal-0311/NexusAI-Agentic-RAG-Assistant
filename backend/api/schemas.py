"""
Pydantic API request and response schemas for NexusAI.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.agents.workflow import WorkflowStepTrace
from backend.generation.answer_generator import Citation


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    ollama_status: Dict[str, Any]
    vector_store_status: Dict[str, Any]
    database_status: str


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    file_size_bytes: int
    total_pages: int
    total_chunks: int
    uploaded_at: str
    is_processed: bool
    is_scanned: bool
    error_message: Optional[str] = None
    collection_name: str


class DocumentUploadResponse(BaseModel):
    message: str
    documents: List[DocumentResponse]


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    retrieval_mode: str = "hybrid"  # "semantic" | "keyword" | "hybrid" | "hybrid_rerank"
    top_k: int = 5
    document_ids: Optional[List[str]] = None
    use_reranker: Optional[bool] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[Citation]
    has_sufficient_context: bool
    is_grounded: bool
    retrieval_mode_used: str
    original_query: str
    rewritten_query: str
    retrieved_passages: List[Dict[str, Any]]
    step_traces: List[WorkflowStepTrace]
    total_latency_ms: float


class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    document_ids: List[str]


class MessageResponse(BaseModel):
    message_id: str
    session_id: str
    role: str
    content: str
    citations: List[Dict[str, Any]]
    retrieved_passages: List[Dict[str, Any]]
    step_traces: List[Dict[str, Any]]
    latency_ms: float
    timestamp: str


class SessionDetailResponse(BaseModel):
    session: SessionResponse
    messages: List[MessageResponse]


class BenchmarkRunRequest(BaseModel):
    k: int = 5
    strategies: Optional[List[str]] = None


class SettingsUpdateRequest(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    default_retrieval_mode: Optional[str] = None
    default_top_k: Optional[int] = None
    use_reranker: Optional[bool] = None
    ollama_model: Optional[str] = None
