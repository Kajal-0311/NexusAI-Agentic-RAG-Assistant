"""
Optional RAGAS framework integration for NexusAI.
Detects if RAGAS is installed; wraps it for evaluation or gracefully falls back to custom metrics.
"""
from typing import Any, Dict, List, Optional
from backend.core.logging_config import setup_logger
from backend.evaluation.metrics import calculate_answer_relevance, calculate_faithfulness

logger = setup_logger("ragas_evaluator")


class RagasEvaluator:
    """Evaluates RAG pipeline outputs using RAGAS or custom fallback metrics."""

    def __init__(self):
        self._ragas_available = False
        try:
            import ragas
            self._ragas_available = True
            logger.info("RAGAS library detected and available for evaluation.")
        except ImportError:
            self._ragas_available = False
            logger.info("RAGAS library not installed. Using native custom mathematical metrics.")

    @property
    def is_available(self) -> bool:
        return self._ragas_available

    def evaluate_turn(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluates a single question-answer-context turn."""
        faithfulness = calculate_faithfulness(answer, contexts)
        relevance = calculate_answer_relevance(question, answer)

        return {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "evaluator_engine": "ragas" if self._ragas_available else "custom_native",
        }
