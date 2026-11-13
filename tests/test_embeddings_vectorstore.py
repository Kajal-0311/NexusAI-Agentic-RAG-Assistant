"""
Tests for Embedding generation and ChromaDB vector store indexing and query.
"""
import math
from backend.embeddings.embedding_service import EmbeddingService
from backend.ingestion.metadata import ChunkMetadata, DocumentChunk
from backend.vectorstore.chroma_store import ChromaStore


def test_embedding_service_fallback_properties(test_embedding_service):
    text = "Agentic RAG using ChromaDB and BM25"
    vec = test_embedding_service.embed_query(text)

    assert len(vec) == 384
    # Check L2 unit norm: sum(v^2) ~ 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-4

    # Batch embedding
    batch = test_embedding_service.embed_documents(["First document", "Second document"])
    assert len(batch) == 2
    assert len(batch[0]) == 384
    assert len(batch[1]) == 384


def test_chroma_store_crud(test_vector_store, sample_chunks):
    # 1. Add Chunks
    added = test_vector_store.add_chunks(sample_chunks)
    assert added == len(sample_chunks)

    # 2. Get Stats
    stats = test_vector_store.get_stats()
    assert stats["total_chunks"] == len(sample_chunks)
    assert stats["unique_documents"] == len(sample_chunks)

    # 3. Search
    results = test_vector_store.search("attention mechanism Transformer", top_k=2)
    assert len(results) <= 2
    assert len(results) > 0
    top = results[0]
    assert "content" in top
    assert "score" in top
    assert top["score"] >= 0.0 and top["score"] <= 1.0

    # 4. Filter by document_id
    filtered = test_vector_store.search("retrieval", top_k=2, document_ids=["doc_1"])
    for r in filtered:
        assert r["document_id"] == "doc_1"

    # 5. Delete Document
    test_vector_store.delete_document("doc_0")
    new_stats = test_vector_store.get_stats()
    assert new_stats["total_chunks"] == len(sample_chunks) - 1
