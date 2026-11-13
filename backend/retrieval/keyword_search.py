"""
BM25 Keyword search retrieval for NexusAI.
Indexes document passages into an inverted BM25 model for exact keyword
and technical term matching.
"""
import re
import time
from typing import Any, Dict, List, Optional

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # Handled with fallback below

from backend.core.config import settings
from backend.core.logging_config import setup_logger
from backend.vectorstore.chroma_store import ChromaStore, get_vector_store

logger = setup_logger("bm25_retriever")


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes text for BM25 while preserving technical phrases, hyphenated terms,
    and numbers (e.g. 'gpt-4', 'llama3.2', 'o(n^2)').
    """
    # Lowercase and split on spaces and standard delimiters, preserving intra-word hyphens/dots
    tokens = re.findall(r"\b[\w\.\-]+[a-zA-Z0-9]\b|\b\w+\b", text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25Retriever:
    """Keyword-based retriever using Okapi BM25 algorithm."""

    def __init__(
        self,
        vector_store: Optional[ChromaStore] = None,
        k1: Optional[float] = None,
        b: Optional[float] = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.k1 = k1 or settings.BM25_K1
        self.b = b or settings.BM25_B
        self._corpus_chunks: List[Dict[str, Any]] = []
        self._bm25: Optional[Any] = None
        self._is_indexed = False

    def build_index(self, document_ids: Optional[List[str]] = None) -> None:
        """Loads chunks from vector store and initializes the BM25 index."""
        try:
            where_clause = None
            if document_ids:
                if len(document_ids) == 1:
                    where_clause = {"document_id": document_ids[0]}
                else:
                    where_clause = {"document_id": {"$in": document_ids}}

            raw = self.vector_store.collection.get(
                where=where_clause,
                include=["documents", "metadatas"],
            )

            self._corpus_chunks = []
            if raw and raw["ids"]:
                for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
                    self._corpus_chunks.append(
                        {
                            "chunk_id": cid,
                            "content": doc,
                            "document_id": meta.get("document_id", ""),
                            "filename": meta.get("filename", "unknown"),
                            "page_number": int(meta.get("page_number", 1)),
                            "chunk_index": int(meta.get("chunk_index", 0)),
                        }
                    )

            if not self._corpus_chunks:
                self._bm25 = None
                self._is_indexed = True
                return

            tokenized_corpus = [
                tokenize_text(c["content"]) for c in self._corpus_chunks
            ]

            if BM25Okapi is not None:
                self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
            else:
                self._bm25 = None

            self._is_indexed = True
            logger.info(
                f"Built BM25 index with {len(self._corpus_chunks)} chunks."
            )
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self._bm25 = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes BM25 keyword query against corpus.
        Returns top-k matching passages sorted descending by score.
        """
        start_time = time.perf_counter()

        # Always re-sync index if not indexed or if filter is applied
        if not self._is_indexed or document_ids is not None:
            self.build_index(document_ids=document_ids)

        if not self._corpus_chunks:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        results: List[Dict[str, Any]] = []

        if self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
            max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

            for idx, score in enumerate(scores):
                if score > 0:
                    chunk = self._corpus_chunks[idx].copy()
                    # Normalized BM25 score
                    chunk["score"] = float(round(score / max_score, 4))
                    chunk["raw_score"] = float(round(score, 4))
                    chunk["retrieval_strategy"] = "keyword"
                    results.append(chunk)
        else:
            # Fallback simple token overlap scoring if rank_bm25 is not loaded
            for chunk in self._corpus_chunks:
                chunk_tokens = set(tokenize_text(chunk["content"]))
                overlap = sum(1 for t in query_tokens if t in chunk_tokens)
                if overlap > 0:
                    c = chunk.copy()
                    score = overlap / len(query_tokens)
                    c["score"] = float(round(score, 4))
                    c["raw_score"] = float(overlap)
                    c["retrieval_strategy"] = "keyword"
                    results.append(c)

        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"BM25 retrieval for '{query[:30]}...' returned {len(top_results)} chunks in {elapsed_ms:.2f}ms."
        )
        return top_results
