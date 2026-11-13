"""
Semantic vector retrieval for NexusAI.
Retrieves relevant document passages based on dense vector cosine similarity.
"""
import time
from typing import Any, Dict, List, Optional

from backend.core.logging_config import setup_logger
from backend.vectorstore.chroma_store import ChromaStore, get_vector_store

logger = setup_logger("semantic_retriever")


class SemanticRetriever:
    """Retrieves top-k chunks using ChromaDB dense vector cosine similarity."""

    def __init__(self, vector_store: Optional[ChromaStore] = None):
        self.vector_store = vector_store or get_vector_store()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search against vector database.
        Returns top-k passages with scores >= min_score.
        """
        start_time = time.perf_counter()
        results = self.vector_store.search(
            query=query, top_k=top_k, document_ids=document_ids
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        filtered = [r for r in results if r["score"] >= min_score]
        for r in filtered:
            r["retrieval_strategy"] = "semantic"

        logger.info(
            f"Semantic retrieval for '{query[:30]}...' returned {len(filtered)} chunks in {elapsed_ms:.2f}ms."
        )
        return filtered
