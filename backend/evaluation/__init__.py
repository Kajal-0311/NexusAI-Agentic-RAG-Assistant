"""Evaluation module for NexusAI."""
from .metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k,
    calculate_retrieval_metrics,
    calculate_faithfulness,
    calculate_answer_relevance,
)
from .benchmark import BenchmarkRunner, EvaluationItem, BenchmarkReport

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "hit_rate_at_k",
    "calculate_retrieval_metrics",
    "calculate_faithfulness",
    "calculate_answer_relevance",
    "BenchmarkRunner",
    "EvaluationItem",
    "BenchmarkReport",
]
