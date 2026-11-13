"""
Evaluation and benchmarking endpoints.
"""
from typing import Any, Dict, List
from fastapi import APIRouter
from backend.api.schemas import BenchmarkRunRequest
from backend.evaluation.benchmark import (
    DEFAULT_BENCHMARK_DATASET,
    BenchmarkReport,
    BenchmarkRunner,
)

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.post("/run", response_model=BenchmarkReport)
def run_evaluation_benchmark(request: BenchmarkRunRequest):
    """
    Executes systematic benchmark comparing Semantic, BM25 Keyword,
    Hybrid RRF, and Hybrid + Reranker configurations.
    """
    runner = BenchmarkRunner()
    report = runner.run_benchmark(
        k=request.k,
        strategies=request.strategies,  # type: ignore
    )
    return report


@router.get("/results", response_model=List[Dict[str, Any]])
def get_evaluation_results():
    """Lists history of recorded benchmark runs."""
    return BenchmarkRunner.list_previous_runs()


@router.get("/dataset", response_model=List[Dict[str, Any]])
def get_evaluation_dataset():
    """Returns the ground-truth benchmark evaluation dataset for transparency."""
    return [item.model_dump() for item in DEFAULT_BENCHMARK_DATASET]
