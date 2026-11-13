"""
Unit tests for PDF Ingestion, Text Cleaning, Chunking, and Metadata preservation.
"""
import pytest
from backend.core.exceptions import DocumentProcessingError, DuplicateDocumentError
from backend.ingestion.chunking import RecursiveTextChunker
from backend.ingestion.metadata import ChunkMetadata, DocumentChunk
from backend.ingestion.pdf_loader import PDFLoader
from backend.ingestion.text_cleaner import clean_text, detect_scanned_page
from tests.conftest import create_sample_pdf_bytes


def test_text_cleaner_whitespace_and_hyphenation():
    raw = "The trans-\nformer model is su-\nperior.   Too   many    spaces.\n\n\n\nNew paragraph."
    cleaned = clean_text(raw)
    assert "transformer model is superior." in cleaned
    assert "Too many spaces." in cleaned
    assert "\n\n\n" not in cleaned


def test_detect_scanned_page():
    empty_page = "   \n\n  "
    assert detect_scanned_page(empty_page, min_chars=30) is True

    valid_page = "This is a comprehensive research paper with ample textual content describing agentic RAG."
    assert detect_scanned_page(valid_page, min_chars=30) is False


def test_pdf_loader_validation():
    loader = PDFLoader(max_size_mb=1)

    # Invalid extension
    with pytest.raises(DocumentProcessingError, match="Unsupported file format"):
        loader.validate_file("document.txt", b"some text")

    # Empty file
    with pytest.raises(DocumentProcessingError, match="is empty"):
        loader.validate_file("document.pdf", b"")

    # Oversized file
    huge_bytes = b"x" * (2 * 1024 * 1024)
    with pytest.raises(DocumentProcessingError, match="exceeds maximum allowed size"):
        loader.validate_file("document.pdf", huge_bytes)


def test_pdf_loader_duplicate_detection(temp_dir):
    loader = PDFLoader()
    pdf_bytes = create_sample_pdf_bytes(num_pages=2)
    file_hash = loader.calculate_file_hash(pdf_bytes)

    existing_hashes = {file_hash}
    with pytest.raises(DuplicateDocumentError):
        loader.process_and_save("paper.pdf", pdf_bytes, existing_hashes=existing_hashes)


def test_recursive_chunker_metadata_preservation():
    chunker = RecursiveTextChunker(chunk_size=150, chunk_overlap=30)

    pages_data = [
        {
            "page_number": 1,
            "raw_text": "First page text. " * 15,
            "cleaned_text": "First page text. " * 15,
            "char_count": len("First page text. " * 15),
        },
        {
            "page_number": 2,
            "raw_text": "Second page text with distinct information. " * 10,
            "cleaned_text": "Second page text with distinct information. " * 10,
            "char_count": len("Second page text with distinct information. " * 10),
        },
    ]

    chunks = chunker.chunk_pages(pages_data, document_id="doc_test_123", filename="paper.pdf")

    assert len(chunks) > 1
    # Check that metadata is retained on all chunks
    for c in chunks:
        assert c.metadata.document_id == "doc_test_123"
        assert c.metadata.filename == "paper.pdf"
        assert c.metadata.page_number in (1, 2)
        assert c.metadata.char_count > 0
        assert c.metadata.token_estimate >= 1
        assert len(c.content) <= 180  # Respects chunk size limit
