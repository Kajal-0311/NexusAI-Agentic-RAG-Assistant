"""
Pydantic schemas and database record models for SQLite tables in NexusAI.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """Schema for document records stored in SQLite."""
    document_id: str
    filename: str
    file_path: str
    file_hash: str
    file_size_bytes: int
    total_pages: int
    total_chunks: int = 0
    uploaded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_processed: bool = False
    is_scanned: bool = False
    error_message: Optional[str] = None
    collection_name: str = "nexus_knowledge_base"


class SessionRecord(BaseModel):
    """Schema for user chat sessions."""
    session_id: str
    title: str = "New Research Session"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    document_ids: List[str] = Field(default_factory=list)


class MessageRecord(BaseModel):
    """Schema for conversation messages within a session."""
    message_id: str
    session_id: str
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_passages: List[Dict[str, Any]] = Field(default_factory=list)
    step_traces: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
