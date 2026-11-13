"""
Data models for document and chunk metadata in NexusAI.
"""
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata associated with an individual text chunk."""
    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(ge=1, description="1-indexed page number")
    chunk_index: int = Field(ge=0, description="Sequential index within document")
    char_count: int
    token_estimate: int
    has_equations: bool = False
    has_tables: bool = False

    def to_chroma_dict(self) -> Dict[str, Any]:
        """Convert metadata to flat key-value pairs suitable for ChromaDB."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": int(self.page_number),
            "chunk_index": int(self.chunk_index),
            "char_count": int(self.char_count),
            "token_estimate": int(self.token_estimate),
            "has_equations": bool(self.has_equations),
            "has_tables": bool(self.has_tables),
        }


class DocumentChunk(BaseModel):
    """A chunk of text along with its associated metadata."""
    content: str
    metadata: ChunkMetadata


class DocumentMetadata(BaseModel):
    """Document-level metadata stored in SQLite database."""
    document_id: str
    filename: str
    file_path: str
    file_hash: str
    file_size_bytes: int
    total_pages: int
    total_chunks: int = 0
    uploaded_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_processed: bool = False
    is_scanned: bool = False
    error_message: Optional[str] = None
    collection_name: str = "nexus_knowledge_base"
