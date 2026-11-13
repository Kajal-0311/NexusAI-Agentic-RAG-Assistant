"""
Centralized API client for communication between Streamlit UI and FastAPI backend.
"""
from typing import Any, Dict, List, Optional
import httpx

BACKEND_BASE_URL = "http://127.0.0.1:8000"


class APIClient:
    """Handles all HTTP interactions with the NexusAI backend."""

    def __init__(self, base_url: str = BACKEND_BASE_URL, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -------------------------------------------------------------------------
    # Health & System
    # -------------------------------------------------------------------------
    def get_health(self) -> Optional[Dict[str, Any]]:
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/health")
                return res.json() if res.status_code == 200 else None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Documents
    # -------------------------------------------------------------------------
    def get_documents(self) -> List[Dict[str, Any]]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/documents")
                return res.json() if res.status_code == 200 else []
        except Exception:
            return []

    def upload_documents(self, file_tuples: List[tuple]) -> Dict[str, Any]:
        """file_tuples is a list of ('files', (filename, bytes, 'application/pdf'))."""
        with httpx.Client(timeout=120.0) as client:
            res = client.post(f"{self.base_url}/documents/upload", files=file_tuples)
            if res.status_code in (200, 201):
                return res.json()
            error_data = res.json() if res.headers.get("content-type") == "application/json" else {}
            detail = error_data.get("detail") or res.text
            raise RuntimeError(f"Upload failed: {detail}")

    def delete_document(self, document_id: str) -> bool:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.delete(f"{self.base_url}/documents/{document_id}")
                return res.status_code == 200
        except Exception:
            return False

    def reindex_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/documents/{document_id}/reindex")
                return res.json() if res.status_code == 200 else None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Chat & Sessions
    # -------------------------------------------------------------------------
    def send_chat_message(
        self,
        query: str,
        session_id: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        use_reranker: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload = {
            "query": query,
            "session_id": session_id,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "document_ids": document_ids,
            "use_reranker": use_reranker,
        }
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(f"{self.base_url}/chat", json=payload)
            if res.status_code == 200:
                return res.json()
            raise RuntimeError(f"Chat request failed: {res.text}")

    def get_sessions(self) -> List[Dict[str, Any]]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/sessions")
                return res.json() if res.status_code == 200 else []
        except Exception:
            return []

    def create_session(self, title: str = "New Research Session", document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        payload = {"title": title, "document_ids": document_ids or []}
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(f"{self.base_url}/sessions", json=payload)
            return res.json() if res.status_code == 201 else {}

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/sessions/{session_id}")
                return res.json() if res.status_code == 200 else None
        except Exception:
            return None

    def delete_session(self, session_id: str) -> bool:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.delete(f"{self.base_url}/sessions/{session_id}")
                return res.status_code == 200
        except Exception:
            return False

    def clear_session_messages(self, session_id: str) -> bool:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.delete(f"{self.base_url}/sessions/{session_id}/messages")
                return res.status_code == 200
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    def run_benchmark(self, k: int = 5, strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        payload = {"k": k, "strategies": strategies}
        with httpx.Client(timeout=180.0) as client:
            res = client.post(f"{self.base_url}/evaluation/run", json=payload)
            if res.status_code == 200:
                return res.json()
            raise RuntimeError(f"Evaluation benchmark failed: {res.text}")

    def get_benchmark_results(self) -> List[Dict[str, Any]]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/evaluation/results")
                return res.json() if res.status_code == 200 else []
        except Exception:
            return []

    def get_benchmark_dataset(self) -> List[Dict[str, Any]]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/evaluation/dataset")
                return res.json() if res.status_code == 200 else []
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------
    def get_settings(self) -> Dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/settings")
                return res.json() if res.status_code == 200 else {}
        except Exception:
            return {}

    def update_settings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(f"{self.base_url}/settings", json=payload)
            return res.json() if res.status_code == 200 else {}


api_client = APIClient()
