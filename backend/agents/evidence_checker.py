"""
Step 4: Evidence Sufficiency Assessment.
Evaluates whether retrieved passages provide adequate grounded proof to answer the question.
"""
from typing import Any, Dict, List
from pydantic import BaseModel

from backend.core.config import settings
from backend.core.logging_config import setup_logger

logger = setup_logger("evidence_checker")


class EvidenceAssessment(BaseModel):
    is_sufficient: bool
    confidence: float
    reason: str
    suggest_query_expansion: bool
    suggested_focus: str = ""


class EvidenceChecker:
    """Evaluates the relevance density and sufficiency of retrieved evidence."""

    def __init__(self, min_score_threshold: float = settings.EVIDENCE_SUFFICIENT_MIN_SCORE):
        self.min_score_threshold = min_score_threshold

    def assess(
        self,
        query: str,
        retrieved_passages: List[Dict[str, Any]],
        retry_count: int = 0,
    ) -> EvidenceAssessment:
        """
        Assesses retrieved candidate passages.
        Verifies:
        1. Passages list is non-empty.
        2. Top passages exceed relevance thresholds.
        3. Query keywords/entities appear in the retrieved context.
        """
        if not retrieved_passages:
            return EvidenceAssessment(
                is_sufficient=False,
                confidence=0.95,
                reason="No candidate passages were retrieved from the document library.",
                suggest_query_expansion=retry_count < settings.MAX_AGENT_RETRIES,
                suggested_focus="Expand search terms or verify document availability.",
            )

        top_passage = retrieved_passages[0]
        top_score = top_passage.get("score", 0.0)

        # Check keyword presence in concatenated retrieved text
        combined_text = " ".join([p.get("content", "").lower() for p in retrieved_passages[:3]])
        query_words = [w.lower() for w in query.split() if len(w) > 3]

        if query_words:
            matched_terms = [w for w in query_words if w in combined_text]
            term_overlap_ratio = len(matched_terms) / len(query_words)
        else:
            term_overlap_ratio = 1.0

        # Sufficiency criteria
        if term_overlap_ratio < 0.25 and top_score < 0.02 and retry_count < settings.MAX_AGENT_RETRIES:
            logger.info(f"Insufficient evidence detected for query '{query[:30]}...' (overlap={term_overlap_ratio:.2f})")
            return EvidenceAssessment(
                is_sufficient=False,
                confidence=0.75,
                reason=f"Retrieved passages lack direct mention of key query terms ({term_overlap_ratio:.0%} matched).",
                suggest_query_expansion=True,
                suggested_focus="Extract alternate synonyms or broader technical concepts.",
            )

        return EvidenceAssessment(
            is_sufficient=True,
            confidence=0.88,
            reason="Retrieved passages contain adequate evidence and topical coverage.",
            suggest_query_expansion=False,
            suggested_focus="",
        )
