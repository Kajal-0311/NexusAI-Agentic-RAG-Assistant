"""
SQLite database engine for NexusAI.
Manages persistent records for documents, chat sessions, and multi-turn message history.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from backend.core.config import settings
from backend.core.logging_config import setup_logger
from backend.database.models import DocumentRecord, MessageRecord, SessionRecord

logger = setup_logger("database")


class Database:
    """Encapsulates SQLite operations with WAL journal mode and auto-migration."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = str(db_path or settings.SQLITE_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a thread-safe connection with dictionary-like row factory."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        """Creates tables if they don't already exist."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    file_size_bytes INTEGER NOT NULL,
                    total_pages INTEGER NOT NULL,
                    total_chunks INTEGER DEFAULT 0,
                    uploaded_at TEXT NOT NULL,
                    is_processed INTEGER DEFAULT 0,
                    is_scanned INTEGER DEFAULT 0,
                    error_message TEXT,
                    collection_name TEXT DEFAULT 'nexus_knowledge_base'
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    document_ids TEXT DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations TEXT DEFAULT '[]',
                    retrieved_passages TEXT DEFAULT '[]',
                    step_traces TEXT DEFAULT '[]',
                    latency_ms REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(file_hash);
                CREATE INDEX IF NOT EXISTS idx_msgs_session ON messages(session_id);
                """
            )
        logger.info(f"SQLite database initialized at '{self.db_path}'")

    # =========================================================================
    # DOCUMENT CRUD
    # =========================================================================
    def insert_document(self, doc: DocumentRecord) -> None:
        """Inserts a new document record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, filename, file_path, file_hash, file_size_bytes,
                    total_pages, total_chunks, uploaded_at, is_processed, is_scanned,
                    error_message, collection_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.document_id,
                    doc.filename,
                    doc.file_path,
                    doc.file_hash,
                    doc.file_size_bytes,
                    doc.total_pages,
                    doc.total_chunks,
                    doc.uploaded_at,
                    1 if doc.is_processed else 0,
                    1 if doc.is_scanned else 0,
                    doc.error_message,
                    doc.collection_name,
                ),
            )

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """Fetches a document by document_id."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_doc(row)

    def get_document_by_hash(self, file_hash: str) -> Optional[DocumentRecord]:
        """Fetches a document by SHA256 file_hash."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_doc(row)

    def get_all_documents(self) -> List[DocumentRecord]:
        """Returns all documents ordered by uploaded_at descending."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
            return [self._row_to_doc(r) for r in cur.fetchall()]

    def update_document_status(
        self,
        document_id: str,
        is_processed: bool,
        total_chunks: int,
        error_message: Optional[str] = None,
    ) -> None:
        """Updates ingestion processing state."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET is_processed = ?, total_chunks = ?, error_message = ?
                WHERE document_id = ?
                """,
                (1 if is_processed else 0, total_chunks, error_message, document_id),
            )

    def delete_document(self, document_id: str) -> bool:
        """Deletes a document record from SQLite."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            return cur.rowcount > 0

    @staticmethod
    def _row_to_doc(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            file_size_bytes=row["file_size_bytes"],
            total_pages=row["total_pages"],
            total_chunks=row["total_chunks"],
            uploaded_at=row["uploaded_at"],
            is_processed=bool(row["is_processed"]),
            is_scanned=bool(row["is_scanned"]),
            error_message=row["error_message"],
            collection_name=row["collection_name"],
        )

    # =========================================================================
    # SESSIONS CRUD
    # =========================================================================
    def create_session(self, session: SessionRecord) -> None:
        """Creates a new chat session."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, title, created_at, updated_at, document_ids)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.title,
                    session.created_at,
                    session.updated_at,
                    json.dumps(session.document_ids),
                ),
            )

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Retrieves session record by id."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            return SessionRecord(
                session_id=row["session_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                document_ids=json.loads(row["document_ids"] or "[]"),
            )

    def get_all_sessions(self) -> List[SessionRecord]:
        """Lists all chat sessions ordered by updated_at descending."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
            records = []
            for row in cur.fetchall():
                records.append(
                    SessionRecord(
                        session_id=row["session_id"],
                        title=row["title"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        document_ids=json.loads(row["document_ids"] or "[]"),
                    )
                )
            return records

    def update_session(self, session_id: str, title: Optional[str] = None, document_ids: Optional[List[str]] = None) -> None:
        """Updates session metadata."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            if title is not None and document_ids is not None:
                conn.execute(
                    "UPDATE sessions SET title = ?, document_ids = ?, updated_at = ? WHERE session_id = ?",
                    (title, json.dumps(document_ids), now, session_id),
                )
            elif title is not None:
                conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (title, now, session_id),
                )
            elif document_ids is not None:
                conn.execute(
                    "UPDATE sessions SET document_ids = ?, updated_at = ? WHERE session_id = ?",
                    (json.dumps(document_ids), now, session_id),
                )

    def delete_session(self, session_id: str) -> bool:
        """Deletes session and all its messages."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return cur.rowcount > 0

    # =========================================================================
    # MESSAGES CRUD
    # =========================================================================
    def add_message(self, msg: MessageRecord) -> None:
        """Appends a new message to a session."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    message_id, session_id, role, content, citations,
                    retrieved_passages, step_traces, latency_ms, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.message_id,
                    msg.session_id,
                    msg.role,
                    msg.content,
                    json.dumps(msg.citations),
                    json.dumps(msg.retrieved_passages),
                    json.dumps(msg.step_traces),
                    msg.latency_ms,
                    msg.timestamp,
                ),
            )
            # Update session's updated_at
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, msg.session_id),
            )

    def get_session_messages(self, session_id: str) -> List[MessageRecord]:
        """Returns all messages in a session in chronological order."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            )
            messages = []
            for row in cur.fetchall():
                messages.append(
                    MessageRecord(
                        message_id=row["message_id"],
                        session_id=row["session_id"],
                        role=row["role"],
                        content=row["content"],
                        citations=json.loads(row["citations"] or "[]"),
                        retrieved_passages=json.loads(row["retrieved_passages"] or "[]"),
                        step_traces=json.loads(row["step_traces"] or "[]"),
                        latency_ms=float(row["latency_ms"] or 0.0),
                        timestamp=row["timestamp"],
                    )
                )
            return messages

    def clear_session_messages(self, session_id: str) -> None:
        """Removes all messages from a session."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


_default_db: Optional[Database] = None


def get_db() -> Database:
    """Singleton getter for global SQLite database."""
    global _default_db
    if _default_db is None:
        _default_db = Database()
    return _default_db
