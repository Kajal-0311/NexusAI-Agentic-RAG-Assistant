"""
Embedding service for NexusAI.
Connects to Ollama for 'nomic-embed-text' embeddings, with automatic offline
deterministic fallback for continuous testing and graceful degradation.
"""
import hashlib
import math
from typing import List, Optional
import httpx

from backend.core.config import settings
from backend.core.logging_config import setup_logger

logger = setup_logger("embedding_service")


class EmbeddingService:
    """Handles generating dense vector embeddings via Ollama or local fallback."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.OLLAMA_EMBED_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT_SECONDS
        self.dim = settings.FALLBACK_EMBED_DIM
        self._ollama_available: Optional[bool] = None

    def check_availability(self) -> bool:
        """Checks whether the Ollama server is reachable and responsive."""
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                self._ollama_available = res.status_code == 200
                return self._ollama_available
        except Exception:
            self._ollama_available = False
            return False

    def _generate_fallback_vector(self, text: str) -> List[float]:
        """
        Generates a deterministic unit-normalized pseudo-semantic embedding vector
        using hashed n-grams. Used when Ollama is offline or during unit tests.
        """
        vec = [0.0] * self.dim
        if not text:
            return vec

        # Tokenize and hash n-grams
        words = text.lower().split()
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            vec[idx] += sign

            # Bigram feature
            if i > 0:
                bigram = f"{words[i-1]}_{word}"
                bh = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
                b_idx = bh % self.dim
                b_sign = 1.0 if ((bh >> 8) % 2 == 0) else -1.0
                vec[b_idx] += b_sign * 1.5

        # L2 Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_query(self, query: str) -> List[float]:
        """Generates embedding for a single user query string."""
        results = self.embed_documents([query])
        return results[0] if results else [0.0] * self.dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of document strings.
        Tries Ollama /api/embed, falls back to /api/embeddings, and finally fallback vector.
        """
        if not texts:
            return []

        # If Ollama is available, try remote embedding
        if self._ollama_available is None:
            self.check_availability()

        if self._ollama_available:
            try:
                embeddings = self._embed_via_ollama(texts)
                if embeddings:
                    return embeddings
            except Exception as e:
                logger.warning(
                    f"Ollama embedding call failed: {e}. Falling back to deterministic embeddings."
                )

        # Fallback path
        return [self._generate_fallback_vector(t) for t in texts]

    def _embed_via_ollama(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Attempts batch or sequential embedding using Ollama."""
        # 1. Try batch /api/embed (Ollama v0.1.34+)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model_name, "input": texts},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "embeddings" in data and len(data["embeddings"]) == len(texts):
                        return data["embeddings"]
        except Exception:
            pass

        # 2. Try classic /api/embeddings sequentially
        results: List[List[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for text in texts:
                resp = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                )
                if resp.status_code == 200:
                    results.append(resp.json()["embedding"])
                else:
                    return None
        return results if len(results) == len(texts) else None


_default_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Singleton getter for the global embedding service instance."""
    global _default_embedding_service
    if _default_embedding_service is None:
        _default_embedding_service = EmbeddingService()
    return _default_embedding_service
