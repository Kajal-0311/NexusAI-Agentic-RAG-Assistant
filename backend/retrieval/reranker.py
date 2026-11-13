"""
Cross-Encoder reranking module for NexusAI.
Re-scores candidate query-passage pairs using a neural cross-encoder,
with graceful fallback when heavy neural dependencies are absent.
"""
import re
import time
from typing import Any, Dict, List, Optional

from backend.core.config import settings
from backend.core.logging_config import setup_logger

logger = setup_logger("reranker")


class CrossEncoderReranker:
    """
    Reranks candidate retrieved passages against the user's query.
    Utilizes sentence-transformers CrossEncoder if available,
    or a high-precision lexical-semantic cross-scoring fallback.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None
        self._is_neural = False
        self._init_attempted = False

    def _lazy_init(self) -> None:
        """Attempts to load sentence_transformers CrossEncoder."""
        if self._init_attempted:
            return
        self._init_attempted = True

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._is_neural = True
            logger.info(f"Loaded neural CrossEncoder model: '{self.model_name}'")
        except Exception as e:
            self._model = None
            self._is_neural = False
            logger.info(
                f"Neural CrossEncoder not available ({e}). Using algorithmic cross-scoring fallback."
            )

    @staticmethod
    def _algorithmic_cross_score(query: str, passage: str) -> float:
        """
        Calculates a high-precision cross-match score based on:
        1. Exact phrase occurrence
        2. Query token coverage (stripped of punctuation and stop words)
        3. Token proximity and bigrams in passage
        """
        q_clean = query.lower().strip()
        p_clean = passage.lower().strip()

        if not q_clean or not p_clean:
            return 0.0

        score = 0.0

        # Exact whole query match
        if q_clean in p_clean:
            score += 0.50

        # Query content tokens
        all_q_words = re.findall(r"\b\w+\b", q_clean)
        stop_words = {"the", "does", "how", "what", "are", "and", "for", "with", "from"}
        q_tokens = [w for w in all_q_words if len(w) > 2 and w not in stop_words]
        if not q_tokens:
            q_tokens = all_q_words

        p_tokens = set(re.findall(r"\b\w+\b", p_clean))
        hits = sum(1 for t in q_tokens if t in p_tokens)
        coverage = hits / len(q_tokens) if q_tokens else 0.0
        score += coverage * 0.50

        # Bigram proximity
        if len(q_tokens) >= 2:
            bigrams = [f"{q_tokens[i]} {q_tokens[i+1]}" for i in range(len(q_tokens) - 1)]
            bigram_hits = sum(1 for b in bigrams if b in p_clean)
            score += (bigram_hits / len(bigrams)) * 0.25

        return float(min(1.0, max(0.0, score)))

    def rerank(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of retrieved passages based on query relevance.
        Returns re-ordered passages with updated 'rerank_score'.
        """
        if not passages:
            return []

        self._lazy_init()
        start_time = time.perf_counter()

        reranked: List[Dict[str, Any]] = []

        if self._is_neural and self._model is not None:
            try:
                pairs = [[query, p["content"]] for p in passages]
                neural_scores = self._model.predict(pairs)
                # Sigmoid or min-max normalization
                for idx, p in enumerate(passages):
                    item = p.copy()
                    raw = float(neural_scores[idx])
                    # Standard sigmoid mapping 1 / (1 + exp(-x))
                    normalized = 1.0 / (1.0 + (2.71828 ** (-raw)))
                    item["rerank_score"] = float(round(normalized, 4))
                    item["score"] = item["rerank_score"]
                    reranked.append(item)
            except Exception as e:
                logger.warning(f"Neural rerank execution failed: {e}. Falling back.")
                self._is_neural = False

        if not reranked:
            for p in passages:
                item = p.copy()
                base_score = p.get("score", 0.5)
                cross_score = self._algorithmic_cross_score(query, p.get("content", ""))
                # Combined weighted score favoring cross-match
                final_score = (base_score * 0.15) + (cross_score * 0.85)
                item["rerank_score"] = float(round(final_score, 4))
                item["score"] = item["rerank_score"]
                reranked.append(item)

        # Sort descending by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        if top_k is not None:
            reranked = reranked[:top_k]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Reranked {len(passages)} passages in {elapsed_ms:.2f}ms (neural={self._is_neural})."
        )
        return reranked
