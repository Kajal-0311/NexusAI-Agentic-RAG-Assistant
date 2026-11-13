"""
Unit tests for Retrieval Engine: Semantic, BM25, Hybrid RRF, and Cross-Encoder Reranker.
"""
from backend.retrieval.hybrid_search import HybridRetriever
from backend.retrieval.keyword_search import BM25Retriever, tokenize_text
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.semantic_search import SemanticRetriever


def test_tokenization_preserves_technical_terms():
    text = "We evaluate Llama-3 and BERT on O(n) complexity in Python 3.11."
    tokens = tokenize_text(text)
    assert "llama-3" in tokens or "llama" in tokens
    assert "bert" in tokens
    assert "python" in tokens


def test_bm25_retrieval(test_vector_store, sample_chunks):
    test_vector_store.add_chunks(sample_chunks)
    bm25 = BM25Retriever(vector_store=test_vector_store)

    results = bm25.retrieve("Reciprocal Rank Fusion", top_k=2)
    assert len(results) > 0
    # Top result should mention RRF
    assert "Reciprocal Rank Fusion" in results[0]["content"]
    assert results[0]["retrieval_strategy"] == "keyword"


def test_rrf_fusion_logic():
    retriever = HybridRetriever()

    sem_results = [
        {"chunk_id": "c1", "content": "Text A", "score": 0.9},
        {"chunk_id": "c2", "content": "Text B", "score": 0.8},
    ]
    kw_results = [
        {"chunk_id": "c2", "content": "Text B", "score": 0.95},
        {"chunk_id": "c3", "content": "Text C", "score": 0.70},
    ]

    # c2 appears in both, so its RRF score should be higher
    fused = retriever.reciprocal_rank_fusion(sem_results, kw_results)
    assert len(fused) == 3
    assert fused[0]["chunk_id"] == "c2"
    assert "semantic" in fused[0]["retrieval_strategy"]
    assert "keyword" in fused[0]["retrieval_strategy"]


def test_cross_encoder_reranker():
    reranker = CrossEncoderReranker()
    passages = [
        {"chunk_id": "c1", "content": "The weather in Seattle is rainy in winter.", "score": 0.8},
        {"chunk_id": "c2", "content": "Gradient descent minimizes the objective loss function.", "score": 0.5},
    ]

    reranked = reranker.rerank(query="How does gradient descent optimize the loss function?", passages=passages)
    assert len(reranked) == 2
    # The gradient descent passage should be promoted to rank 1
    assert reranked[0]["chunk_id"] == "c2"
    assert "rerank_score" in reranked[0]


def test_hybrid_retriever_modes(test_vector_store, sample_chunks):
    test_vector_store.add_chunks(sample_chunks)
    hybrid = HybridRetriever(
        semantic_retriever=SemanticRetriever(test_vector_store),
        bm25_retriever=BM25Retriever(test_vector_store),
        reranker=CrossEncoderReranker(),
    )

    for mode in ["semantic", "keyword", "hybrid", "hybrid_rerank"]:
        res = hybrid.retrieve(query="attention mechanism", mode=mode, top_k=2)
        assert len(res) <= 2
        assert len(res) > 0
