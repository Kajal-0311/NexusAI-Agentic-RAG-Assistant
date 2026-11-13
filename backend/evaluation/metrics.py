"""
Rigorous Information Retrieval (IR) and Generation Evaluation Metrics for NexusAI.
Includes Precision@K, Recall@K, MRR, Hit Rate@K, Faithfulness, and Answer Relevance.
"""
import re
from typing import Any, Dict, List, Set


def precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """
    Precision@K = (# of relevant retrieved items in top k) / k
    Measures the purity of the top-k retrieved list.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_retrieved = sum(1 for item in top_k if item in ground_truth_ids)
    return float(round(relevant_retrieved / k, 4))


def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """
    Recall@K = (# of relevant retrieved items in top k) / (total # of relevant items)
    Measures coverage of known ground-truth items.
    """
    if not ground_truth_ids:
        return 1.0  # Vacuously true if no target exists
    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for item in top_k if item in ground_truth_ids)
    return float(round(relevant_retrieved / len(ground_truth_ids), 4))


def reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """
    Reciprocal Rank (RR) = 1 / rank_of_first_relevant_item
    If no relevant item is found, RR = 0.0.
    MRR is the mean of RR across all queries.
    """
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in ground_truth_ids:
            return float(round(1.0 / rank, 4))
    return 0.0


def hit_rate_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """
    HitRate@K = 1.0 if at least one relevant document appears in top k, else 0.0.
    """
    top_k = retrieved_ids[:k]
    return 1.0 if any(item in ground_truth_ids for item in top_k) else 0.0


def calculate_retrieval_metrics(
    retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5
) -> Dict[str, float]:
    """Computes comprehensive retrieval evaluation metrics."""
    return {
        f"precision@{k}": precision_at_k(retrieved_ids, ground_truth_ids, k),
        f"recall@{k}": recall_at_k(retrieved_ids, ground_truth_ids, k),
        f"hit_rate@{k}": hit_rate_at_k(retrieved_ids, ground_truth_ids, k),
        "mrr": reciprocal_rank(retrieved_ids, ground_truth_ids),
    }


def calculate_faithfulness(
    answer: str, context_passages: List[str]
) -> float:
    """
    Evaluates what fraction of sentence claims in the answer can be corroborated
    by the retrieved passages.
    Score ranges from 0.0 (total hallucination) to 1.0 (fully grounded).
    """
    if not answer or not context_passages:
        return 0.0

    combined_context = " ".join(context_passages).lower()

    # Split answer into individual claim sentences
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", answer)
        if len(s.strip()) > 10 and not s.strip().startswith("#")
    ]

    if not sentences:
        return 1.0

    supported_count = 0
    for sentence in sentences:
        # Extract meaningful content words (ignore stop words and formatting)
        words = [
            w.lower()
            for w in re.findall(r"\b[a-zA-Z]{3,}\b", sentence)
            if w.lower() not in {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were"}
        ]
        if not words:
            supported_count += 1
            continue

        # Check word overlap against context
        matches = sum(1 for w in words if w in combined_context)
        overlap = matches / len(words)
        if overlap >= 0.50:
            supported_count += 1

    return float(round(supported_count / len(sentences), 4))


def calculate_answer_relevance(question: str, answer: str) -> float:
    """
    Computes lexical topical overlap between the user question and the answer.
    """
    q_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", question.lower()))
    a_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", answer.lower()))

    if not q_tokens:
        return 0.5

    overlap = q_tokens.intersection(a_tokens)
    return float(round(min(1.0, len(overlap) / len(q_tokens)), 4))
