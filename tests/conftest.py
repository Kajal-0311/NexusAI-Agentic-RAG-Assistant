"""
Shared test fixtures and in-memory test databases for NexusAI pytest suite.
"""
import io
import shutil
import tempfile
import pytest
from pypdf import PdfWriter

from backend.core.config import settings
from backend.database.database import Database
from backend.embeddings.embedding_service import EmbeddingService
from backend.ingestion.metadata import ChunkMetadata, DocumentChunk
from backend.vectorstore.chroma_store import ChromaStore


def create_sample_pdf_bytes(num_pages: int = 2) -> bytes:
    """Generates valid minimal PDF bytes for testing."""
    # A standard valid minimal 2-page PDF
    writer = PdfWriter()
    for i in range(num_pages):
        writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


@pytest.fixture
def temp_dir():
    """Provides an isolated temporary directory for test storage."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def test_db(temp_dir):
    """Provides an isolated SQLite database in a temp directory."""
    db_path = f"{temp_dir}/test_nexus.db"
    db = Database(db_path=db_path)
    return db


@pytest.fixture
def test_embedding_service():
    """Provides a deterministic offline embedding service for tests."""
    return EmbeddingService()


@pytest.fixture
def test_vector_store(temp_dir, test_embedding_service):
    """Provides an isolated ChromaDB vector store."""
    store_dir = f"{temp_dir}/chroma_test"
    store = ChromaStore(
        persist_dir=store_dir,
        collection_name="test_collection",
        embedding_service=test_embedding_service,
    )
    yield store


@pytest.fixture
def sample_chunks():
    """Provides sample DocumentChunks for retrieval tests."""
    chunks = []
    sample_texts = [
        (
            "The Transformer architecture relies entirely on an attention mechanism to draw global dependencies between input and output.",
            1,
            "attention_paper.pdf",
        ),
        (
            "Reciprocal Rank Fusion (RRF) is an effective algorithm to combine search rankings from multiple retrieval systems without needing normalized scores.",
            2,
            "rrf_search.pdf",
        ),
        (
            "BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query based on term frequency.",
            1,
            "information_retrieval.pdf",
        ),
        (
            "Cross-Encoder rerankers score candidate passages by performing joint full self-attention over the query and document simultaneously.",
            3,
            "reranking_guide.pdf",
        ),
    ]

    for idx, (text, page, filename) in enumerate(sample_texts):
        meta = ChunkMetadata(
            chunk_id=f"doc_{idx}_p{page}_c0",
            document_id=f"doc_{idx}",
            filename=filename,
            page_number=page,
            chunk_index=0,
            char_count=len(text),
            token_estimate=len(text) // 4,
            has_tables=False,
            has_equations=False,
        )
        chunks.append(DocumentChunk(content=text, metadata=meta))

    return chunks
