"""
Unit tests for IR evaluation metrics (Precision@K, Recall@K, MRR, HitRate@K, Faithfulness).
"""
from backend.evaluation.benchmark import BenchmarkRunner
from backend.evaluation.metrics import (
    calculate_answer_relevance,
    calculate_faithfulness,
    calculate_retrieval_metrics,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_ir_metrics_precision_recall_mrr():
    ground_truth = {"doc_A", "doc_B"}
    retrieved = ["doc_A", "doc_X", "doc_B", "doc_Y", "doc_Z"]

    # Precision@1: doc_A is relevant -> 1/1 = 1.0
    assert precision_at_k(retrieved, ground_truth, k=1) == 1.0

    # Precision@2: doc_A, doc_X -> 1/2 = 0.5
    assert precision_at_k(retrieved, ground_truth, k=2) == 0.5

    # Recall@2: 1 relevant out of 2 target -> 1/2 = 0.5
    assert recall_at_k(retrieved, ground_truth, k=2) == 0.5

    # Recall@3: doc_A, doc_X, doc_B -> 2 out of 2 target -> 1.0
    assert recall_at_k(retrieved, ground_truth, k=3) == 1.0

    # Reciprocal Rank: first relevant is at rank 1 -> 1/1 = 1.0
    assert reciprocal_rank(retrieved, ground_truth) == 1.0

    # Hit Rate@1: 1.0
    assert hit_rate_at_k(retrieved, ground_truth, k=1) == 1.0

    # Test when first relevant is at rank 3
    retrieved_lagged = ["doc_X", "doc_Y", "doc_A"]
    assert reciprocal_rank(retrieved_lagged, ground_truth) == round(1.0 / 3, 4)


def test_calculate_faithfulness():
    context = [
        "The Transformer uses multi-head attention to attend to information at different positions."
    ]
    # Fully grounded answer
    grounded_answer = "The Transformer uses multi-head attention to attend to different positions."
    assert calculate_faithfulness(grounded_answer, context) >= 0.80

    # Hallucinated answer
    hallucinated_answer = "Quantum entanglement produces faster-than-light warp speeds."
    assert calculate_faithfulness(hallucinated_answer, context) <= 0.20


def test_benchmark_runner_execution(temp_dir):
    runner = BenchmarkRunner()
    # Run small evaluation on k=3 for semantic and keyword
    report = runner.run_benchmark(k=3, strategies=["semantic", "keyword"])

    assert report.total_samples > 0
    assert "semantic" in report.strategies
    assert "keyword" in report.strategies
    assert report.strategies["semantic"].precision_at_k >= 0.0
    assert report.strategies["keyword"].precision_at_k >= 0.0
