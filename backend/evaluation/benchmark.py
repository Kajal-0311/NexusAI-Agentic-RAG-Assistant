"""
Benchmark runner and evaluation dataset harness for NexusAI.
Executes systematic comparisons across retrieval configurations
(Semantic, BM25 Keyword, Hybrid RRF, and Hybrid + Reranker).
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.core.config import settings
from backend.core.logging_config import setup_logger
from backend.evaluation.metrics import (
    calculate_answer_relevance,
    calculate_faithfulness,
    calculate_retrieval_metrics,
)
from backend.generation.answer_generator import AnswerGenerator
from backend.retrieval.hybrid_search import HybridRetriever, RetrievalMode, get_hybrid_retriever

logger = setup_logger("benchmark")


class EvaluationItem(BaseModel):
    """A ground-truth question item for evaluation."""
    id: str
    question: str
    expected_answer: str
    target_filename: str
    target_page: int
    expected_keywords: List[str]


class StrategyMetric(BaseModel):
    """Aggregate performance metrics for a specific retrieval strategy."""
    strategy_name: str
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    faithfulness: float
    answer_relevance: float
    mean_latency_ms: float
    total_queries: int


class BenchmarkReport(BaseModel):
    """Complete benchmark execution report."""
    run_id: str
    timestamp: str
    total_samples: int
    k: int
    strategies: Dict[str, StrategyMetric]
    sample_evaluations: List[Dict[str, Any]] = Field(default_factory=list)


# Verified ground-truth benchmark suite for AI/ML and Computer Science research
DEFAULT_BENCHMARK_DATASET: List[EvaluationItem] = [
    EvaluationItem(
        id="eval_001",
        question="What is the attention mechanism in Transformer architecture?",
        expected_answer="The attention mechanism computes a weighted sum of values based on the query-key dot product.",
        target_filename="attention_is_all_you_need.pdf",
        target_page=3,
        expected_keywords=["attention", "query", "key", "value", "scaled dot-product"],
    ),
    EvaluationItem(
        id="eval_002",
        question="How does Reciprocal Rank Fusion combine rankings from different retrieval systems?",
        expected_answer="RRF sums reciprocal ranks 1/(k + rank) across multiple rankers without requiring score calibration.",
        target_filename="hybrid_search_rrf.pdf",
        target_page=2,
        expected_keywords=["reciprocal", "rank", "fusion", "constant k", "ranker"],
    ),
    EvaluationItem(
        id="eval_003",
        question="What is the role of the Cross-Encoder reranker in RAG pipelines?",
        expected_answer="A Cross-Encoder performs joint self-attention over the query and candidate passage together.",
        target_filename="cross_encoder_reranking.pdf",
        target_page=4,
        expected_keywords=["cross-encoder", "reranker", "joint attention", "candidate", "relevance"],
    ),
    EvaluationItem(
        id="eval_004",
        question="What are the key failure modes of vector-only retrieval for technical terms?",
        expected_answer="Vector embeddings often suffer from out-of-vocabulary mismatches and loss of exact identifier matching.",
        target_filename="bm25_vs_dense.pdf",
        target_page=1,
        expected_keywords=["sparse", "dense", "out-of-vocabulary", "exact match", "keyword"],
    ),
    EvaluationItem(
        id="eval_005",
        question="How does controlled agentic query rewriting improve follow-up question accuracy?",
        expected_answer="Query rewriting expands conversational pronouns using prior context into self-contained search targets.",
        target_filename="agentic_rag_survey.pdf",
        target_page=5,
        expected_keywords=["rewriting", "pronouns", "conversation history", "self-contained", "context"],
    ),
]


class BenchmarkRunner:
    """Executes systematic retrieval and generation benchmarks."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[AnswerGenerator] = None,
        dataset: Optional[List[EvaluationItem]] = None,
    ):
        self.retriever = retriever or get_hybrid_retriever()
        self.generator = generator or AnswerGenerator()
        self.dataset = dataset or DEFAULT_BENCHMARK_DATASET

    def run_benchmark(
        self,
        k: int = 5,
        strategies: Optional[List[RetrievalMode]] = None,
    ) -> BenchmarkReport:
        """
        Executes benchmark evaluation across specified retrieval strategies.
        Saves real run output to data/evaluation/runs/.
        """
        eval_strategies: List[RetrievalMode] = strategies or [
            "semantic",
            "keyword",
            "hybrid",
            "hybrid_rerank",
        ]

        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting Benchmark run {run_id} with {len(self.dataset)} questions across {len(eval_strategies)} strategies.")

        strategy_results: Dict[str, StrategyMetric] = {}
        sample_evals: List[Dict[str, Any]] = []

        for strat in eval_strategies:
            p_scores = []
            r_scores = []
            hr_scores = []
            mrr_scores = []
            faith_scores = []
            rel_scores = []
            latencies = []

            for item in self.dataset:
                start_t = time.perf_counter()

                # Execute retrieval with current strategy
                is_rerank = strat == "hybrid_rerank"
                base_mode = "hybrid" if is_rerank else strat

                retrieved = self.retriever.retrieve(
                    query=item.question,
                    mode=base_mode,  # type: ignore
                    top_k=k,
                    use_reranker=is_rerank,
                )
                lat_ms = (time.perf_counter() - start_t) * 1000
                latencies.append(lat_ms)

                # Determine target matches based on keywords or filename/page
                target_ids = {item.id}
                # Check retrieved passages for keyword hits
                retrieved_pseudo_ids = []
                for p_idx, p in enumerate(retrieved):
                    content = p.get("content", "").lower()
                    has_kw = any(kw.lower() in content for kw in item.expected_keywords)
                    # If passage content matches keywords or filename, consider it relevant
                    if has_kw or p.get("filename", "") == item.target_filename:
                        retrieved_pseudo_ids.append(item.id)
                    else:
                        retrieved_pseudo_ids.append(f"irrelevant_{p_idx}")

                # Calculate IR metrics
                metrics = calculate_retrieval_metrics(retrieved_pseudo_ids, target_ids, k=k)
                p_scores.append(metrics[f"precision@{k}"])
                r_scores.append(metrics[f"recall@{k}"])
                hr_scores.append(metrics[f"hit_rate@{k}"])
                mrr_scores.append(metrics["mrr"])

                # Generation check
                passages_content = [p.get("content", "") for p in retrieved]
                gen_out = self.generator.generate_answer(item.question, retrieved)
                faith = calculate_faithfulness(gen_out.answer, passages_content)
                rel = calculate_answer_relevance(item.question, gen_out.answer)

                faith_scores.append(faith)
                rel_scores.append(rel)

                if strat == "hybrid_rerank":
                    sample_evals.append({
                        "question": item.question,
                        "expected_answer": item.expected_answer,
                        "generated_answer": gen_out.answer,
                        "strategy": strat,
                        "precision": metrics[f"precision@{k}"],
                        "recall": metrics[f"recall@{k}"],
                        "faithfulness": faith,
                        "citations": [c.model_dump() for c in gen_out.citations],
                    })

            n = len(self.dataset)
            strategy_results[strat] = StrategyMetric(
                strategy_name=strat,
                precision_at_k=round(sum(p_scores) / n, 4) if n else 0.0,
                recall_at_k=round(sum(r_scores) / n, 4) if n else 0.0,
                hit_rate_at_k=round(sum(hr_scores) / n, 4) if n else 0.0,
                mrr=round(sum(mrr_scores) / n, 4) if n else 0.0,
                faithfulness=round(sum(faith_scores) / n, 4) if n else 0.0,
                answer_relevance=round(sum(rel_scores) / n, 4) if n else 0.0,
                mean_latency_ms=round(sum(latencies) / n, 2) if n else 0.0,
                total_queries=n,
            )

        report = BenchmarkReport(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_samples=len(self.dataset),
            k=k,
            strategies=strategy_results,
            sample_evaluations=sample_evals,
        )

        # Save to disk
        save_dir = settings.EVALUATION_DIR / "runs"
        save_dir.mkdir(parents=True, exist_ok=True)
        report_file = save_dir / f"{run_id}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

        logger.info(f"Saved evaluation benchmark report to {report_file}")
        return report

    @staticmethod
    def list_previous_runs() -> List[Dict[str, Any]]:
        """Lists previous benchmark runs saved in data/evaluation/runs."""
        save_dir = settings.EVALUATION_DIR / "runs"
        if not save_dir.exists():
            return []

        runs = []
        for file in sorted(save_dir.glob("run_*.json"), reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    runs.append(data)
            except Exception as e:
                logger.warning(f"Failed to read run file {file}: {e}")
        return runs
