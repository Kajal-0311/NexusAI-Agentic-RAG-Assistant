"""
Step 3: Retrieval Agent & Execution Coordinator.
"""
from typing import Any, Dict, List, Optional
from backend.core.config import settings
from backend.core.logging_config import setup_logger
from backend.retrieval.hybrid_search import HybridRetriever, RetrievalMode, get_hybrid_retriever

logger = setup_logger("retrieval_agent")


class RetrievalAgent:
    """Decides and executes the optimal retrieval strategy for the query."""

    def __init__(self, hybrid_retriever: Optional[HybridRetriever] = None):
        self.retriever = hybrid_retriever or get_hybrid_retriever()

    def retrieve_evidence(
        self,
        query: str,
        retrieval_mode: Optional[RetrievalMode] = None,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        use_reranker: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Coordinates retrieval with chosen strategy, top_k, and optional reranker.
        """
        mode = retrieval_mode or settings.DEFAULT_RETRIEVAL_MODE
        k = top_k or settings.DEFAULT_TOP_K

        logger.info(
            f"RetrievalAgent executing mode='{mode}', top_k={k}, doc_filter={document_ids} for '{query[:30]}...'"
        )

        return self.retriever.retrieve(
            query=query,
            mode=mode,
            top_k=k,
            document_ids=document_ids,
            use_reranker=use_reranker,
        )
