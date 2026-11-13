"""
Hybrid Retrieval Engine for NexusAI.
Combines Dense Vector Semantic Search and Sparse BM25 Keyword Search
using Reciprocal Rank Fusion (RRF), with optional Cross-Encoder reranking.
"""
from typing import Any, Dict, List, Literal, Optional

from backend.core.config import settings
from backend.core.logging_config import setup_logger
from backend.retrieval.keyword_search import BM25Retriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.semantic_search import SemanticRetriever

logger = setup_logger("hybrid_retriever")

RetrievalMode = Literal["semantic", "keyword", "hybrid", "hybrid_rerank"]


class HybridRetriever:
    """
    Unified retrieval orchestrator supporting Semantic, BM25 Keyword,
    Reciprocal Rank Fusion (RRF) Hybrid Search, and Cross-Encoder Reranking.
    """

    def __init__(
        self,
        semantic_retriever: Optional[SemanticRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        rrf_k: Optional[int] = None,
    ):
        self.semantic_retriever = semantic_retriever or SemanticRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.rrf_k = rrf_k or settings.RRF_K

    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Combines two ranked lists using Reciprocal Rank Fusion (RRF):
        RRF_score(d) = sum( weight / (k + rank(d)) )
        Deduplicates chunks while summing rank contributions.
        """
        rrf_scores: Dict[str, float] = {}
        chunks_map: Dict[str, Dict[str, Any]] = {}
        found_in: Dict[str, List[str]] = {}

        # Process Semantic results
        for rank, item in enumerate(semantic_results, start=1):
            cid = item["chunk_id"]
            chunks_map[cid] = item
            score_contrib = semantic_weight / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contrib
            found_in.setdefault(cid, []).append("semantic")

        # Process Keyword (BM25) results
        for rank, item in enumerate(keyword_results, start=1):
            cid = item["chunk_id"]
            if cid not in chunks_map:
                chunks_map[cid] = item
            score_contrib = keyword_weight / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contrib
            found_in.setdefault(cid, []).append("keyword")

        # Compile and sort combined candidates
        fused_results: List[Dict[str, Any]] = []
        for cid, score in rrf_scores.items():
            entry = chunks_map[cid].copy()
            entry["score"] = float(round(score, 6))
            entry["rrf_score"] = float(round(score, 6))
            entry["retrieval_strategy"] = "+".join(found_in.get(cid, ["hybrid"]))
            fused_results.append(entry)

        fused_results.sort(key=lambda x: x["score"], reverse=True)
        return fused_results

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        use_reranker: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves passages based on the selected mode:
        - 'semantic': Dense vector cosine retrieval
        - 'keyword': BM25 Okapi lexical retrieval
        - 'hybrid': Reciprocal Rank Fusion of semantic and BM25
        - 'hybrid_rerank': Hybrid RRF followed by Cross-Encoder reranking
        """
        if not query.strip():
            return []

        # Determine whether reranking should be performed
        apply_rerank = (
            use_reranker
            if use_reranker is not None
            else (mode == "hybrid_rerank" or settings.USE_RERANKER)
        )

        # Retrieve candidates
        if mode == "semantic":
            results = self.semantic_retriever.retrieve(
                query=query, top_k=top_k * 2 if apply_rerank else top_k, document_ids=document_ids
            )
        elif mode == "keyword":
            results = self.bm25_retriever.retrieve(
                query=query, top_k=top_k * 2 if apply_rerank else top_k, document_ids=document_ids
            )
        else:  # 'hybrid' or 'hybrid_rerank'
            # Fetch candidate pool from both systems
            candidate_pool = max(top_k * 2, 10)
            sem_results = self.semantic_retriever.retrieve(
                query=query, top_k=candidate_pool, document_ids=document_ids
            )
            bm25_results = self.bm25_retriever.retrieve(
                query=query, top_k=candidate_pool, document_ids=document_ids
            )
            results = self.reciprocal_rank_fusion(sem_results, bm25_results)

        # Apply reranking if specified
        if apply_rerank and results:
            results = self.reranker.rerank(query=query, passages=results, top_k=top_k)
        else:
            results = results[:top_k]

        logger.info(
            f"Retrieved {len(results)} chunks for mode='{mode}' (rerank={apply_rerank}) on query='{query[:30]}...'"
        )
        return results


_default_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Singleton getter for global HybridRetriever."""
    global _default_hybrid_retriever
    if _default_hybrid_retriever is None:
        _default_hybrid_retriever = HybridRetriever()
    return _default_hybrid_retriever
