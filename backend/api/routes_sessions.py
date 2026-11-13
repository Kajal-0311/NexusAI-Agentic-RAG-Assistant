"""
Session and conversation history management endpoints.
"""
from typing import List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.api.schemas import (
    MessageResponse,
    SessionDetailResponse,
    SessionResponse,
)
from backend.database.database import get_db
from backend.database.models import SessionRecord
from backend.memory.conversation_memory import get_memory

router = APIRouter(prefix="/sessions", tags=["Sessions"])


class CreateSessionRequest(BaseModel):
    title: str = "New Research Session"
    document_ids: List[str] = []


@router.get("", response_model=List[SessionResponse])
def list_sessions():
    """Lists all stored chat sessions."""
    memory = get_memory()
    sessions = memory.list_sessions()
    return [SessionResponse(**s.model_dump()) for s in sessions]


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(request: CreateSessionRequest):
    """Creates a new explicit chat session."""
    memory = get_memory()
    session = memory.get_or_create_session(title=request.title)
    if request.document_ids:
        db = get_db()
        db.update_session(session.session_id, document_ids=request.document_ids)
        session.document_ids = request.document_ids
    return SessionResponse(**session.model_dump())


@router.get("/{session_id}", response_model=SessionDetailResponse)
def get_session(session_id: str):
    """Retrieves session metadata and full message history."""
    db = get_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = db.get_session_messages(session_id)
    return SessionDetailResponse(
        session=SessionResponse(**session.model_dump()),
        messages=[MessageResponse(**m.model_dump()) for m in messages],
    )


@router.delete("/{session_id}", status_code=status.HTTP_200_OK)
def delete_session(session_id: str):
    """Deletes a chat session and all associated messages."""
    memory = get_memory()
    deleted = memory.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": f"Session {session_id} deleted."}


@router.delete("/{session_id}/messages", status_code=status.HTTP_200_OK)
def clear_session_messages(session_id: str):
    """Clears all conversation messages in the session while keeping the session record."""
    memory = get_memory()
    memory.clear_session(session_id)
    return {"message": f"Messages in session {session_id} cleared."}
