"""
Conversation Memory Manager for NexusAI.
Maintains persistent multi-turn chat sessions and provides sliding context windows
without overriding grounded document evidence.
"""
from typing import Any, Dict, List, Optional
import uuid

from backend.core.logging_config import setup_logger
from backend.database.database import Database, get_db
from backend.database.models import MessageRecord, SessionRecord

logger = setup_logger("conversation_memory")


class ConversationMemory:
    """Provides high-level conversational memory operations over SQLite storage."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_db()

    def get_or_create_session(
        self, session_id: Optional[str] = None, title: Optional[str] = None
    ) -> SessionRecord:
        """Retrieves existing session or creates a new one."""
        if session_id:
            existing = self.db.get_session(session_id)
            if existing:
                return existing

        new_id = session_id or str(uuid.uuid4())
        session = SessionRecord(
            session_id=new_id,
            title=title or "New Research Session",
        )
        self.db.create_session(session)
        logger.info(f"Created new session: {new_id} ('{session.title}')")
        return session

    def add_user_message(self, session_id: str, content: str) -> MessageRecord:
        """Appends a user message to the session history."""
        msg = MessageRecord(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=content,
        )
        self.db.add_message(msg)

        # Update session title if it's the first message
        messages = self.db.get_session_messages(session_id)
        if len(messages) == 1:
            snippet = content[:35] + ("..." if len(content) > 35 else "")
            self.db.update_session(session_id, title=snippet)

        return msg

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        retrieved_passages: Optional[List[Dict[str, Any]]] = None,
        step_traces: Optional[List[Dict[str, Any]]] = None,
        latency_ms: float = 0.0,
    ) -> MessageRecord:
        """Appends an assistant response with citations and execution telemetry."""
        msg = MessageRecord(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=content,
            citations=citations or [],
            retrieved_passages=retrieved_passages or [],
            step_traces=step_traces or [],
            latency_ms=latency_ms,
        )
        self.db.add_message(msg)
        return msg

    def get_context_window(
        self, session_id: str, max_turns: int = 4
    ) -> List[Dict[str, str]]:
        """
        Retrieves the most recent messages formatted as role-content dicts
        for passing to query rewriting.
        """
        all_msgs = self.db.get_session_messages(session_id)
        recent = all_msgs[-(max_turns * 2) :] if all_msgs else []
        return [{"role": m.role, "content": m.content} for m in recent]

    def get_full_history(self, session_id: str) -> List[MessageRecord]:
        """Returns all messages stored in a session."""
        return self.db.get_session_messages(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clears all conversation messages in a session."""
        self.db.clear_session_messages(session_id)
        logger.info(f"Cleared messages in session {session_id}")

    def list_sessions(self) -> List[SessionRecord]:
        """Lists all stored chat sessions."""
        return self.db.get_all_sessions()

    def delete_session(self, session_id: str) -> bool:
        """Deletes session and all its messages."""
        return self.db.delete_session(session_id)


_default_memory: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    """Singleton getter for ConversationMemory."""
    global _default_memory
    if _default_memory is None:
        _default_memory = ConversationMemory()
    return _default_memory
