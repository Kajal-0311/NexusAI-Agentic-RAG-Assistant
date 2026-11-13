"""
PDF Loader and validation engine for NexusAI.
Extracts text with per-page tracking and handles file validation and duplicate detection.
"""
import hashlib
import io
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pypdf

from backend.core.config import settings
from backend.core.exceptions import (
    DocumentProcessingError,
    DuplicateDocumentError,
    ScannedPDFError,
)
from backend.core.logging_config import setup_logger
from backend.ingestion.metadata import DocumentMetadata
from backend.ingestion.text_cleaner import clean_text, detect_scanned_page

logger = setup_logger("pdf_loader")


class PDFLoader:
    """Handles PDF validation, duplicate hashing, and page-by-page text extraction."""

    def __init__(self, max_size_mb: Optional[int] = None):
        self.max_size_bytes = (max_size_mb or settings.MAX_UPLOAD_SIZE_MB) * 1024 * 1024

    @staticmethod
    def calculate_file_hash(file_bytes: bytes) -> str:
        """Computes SHA-256 hash of file contents for deduplication."""
        return hashlib.sha256(file_bytes).hexdigest()

    def validate_file(self, filename: str, file_bytes: bytes) -> None:
        """Validates file extension and size constraints."""
        if not filename.lower().endswith(".pdf"):
            raise DocumentProcessingError(
                f"Unsupported file format '{filename}'. Only PDF files are supported."
            )
        if len(file_bytes) == 0:
            raise DocumentProcessingError(f"File '{filename}' is empty (0 bytes).")
        if len(file_bytes) > self.max_size_bytes:
            raise DocumentProcessingError(
                f"File '{filename}' exceeds maximum allowed size of {self.max_size_bytes // (1024 * 1024)}MB."
            )

    def extract_pages(
        self, file_bytes: bytes, filename: str
    ) -> Tuple[List[Dict[str, any]], bool]:
        """
        Extracts raw text page by page from PDF bytes.
        Returns a tuple of (list of page dicts, is_scanned_flag).
        Each page dict contains:
            - page_number: 1-indexed int
            - raw_text: string
            - cleaned_text: string
            - char_count: int
        """
        pages_data: List[Dict[str, any]] = []
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        except Exception as e:
            logger.error(f"Failed to parse PDF {filename}: {e}")
            raise DocumentProcessingError(
                f"Corrupted or invalid PDF file '{filename}': {str(e)}"
            )

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise DocumentProcessingError(f"PDF document '{filename}' contains 0 pages.")

        scanned_page_count = 0

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(
                    f"Warning: page {page_num} of {filename} failed extraction: {e}"
                )
                raw_text = ""

            cleaned = clean_text(raw_text)
            char_count = len(cleaned)

            if detect_scanned_page(cleaned, min_chars=40):
                scanned_page_count += 1

            pages_data.append(
                {
                    "page_number": page_num,
                    "raw_text": raw_text,
                    "cleaned_text": cleaned,
                    "char_count": char_count,
                }
            )

        # If more than 75% of pages have virtually no text, flag document as scanned
        is_scanned = (scanned_page_count / total_pages) > 0.75

        logger.info(
            f"Extracted {total_pages} pages from '{filename}'. Scanned detected: {is_scanned}"
        )
        return pages_data, is_scanned

    def process_and_save(
        self,
        filename: str,
        file_bytes: bytes,
        existing_hashes: Optional[set[str]] = None,
    ) -> Tuple[DocumentMetadata, List[Dict[str, any]]]:
        """
        Validates, checks duplicates, extracts page data, saves to disk,
        and returns document metadata and extracted pages.
        """
        self.validate_file(filename, file_bytes)

        file_hash = self.calculate_file_hash(file_bytes)
        if existing_hashes and file_hash in existing_hashes:
            raise DuplicateDocumentError(
                f"Document '{filename}' has already been uploaded (matching SHA256: {file_hash[:10]}...)."
            )

        doc_id = str(uuid.uuid4())
        safe_filename = f"{doc_id[:8]}_{Path(filename).name}"
        save_path = settings.UPLOADS_DIR / safe_filename

        with open(save_path, "wb") as f:
            f.write(file_bytes)

        pages, is_scanned = self.extract_pages(file_bytes, filename)

        error_message = None
        if is_scanned:
            error_message = (
                "Document appears to be a scanned PDF or contains rasterized pages. "
                "Text extraction was minimal; an optical character recognition (OCR) engine "
                "is recommended for full indexing."
            )

        doc_meta = DocumentMetadata(
            document_id=doc_id,
            filename=filename,
            file_path=str(save_path),
            file_hash=file_hash,
            file_size_bytes=len(file_bytes),
            total_pages=len(pages),
            is_processed=False,
            is_scanned=is_scanned,
            error_message=error_message,
            collection_name=settings.COLLECTION_NAME,
        )

        return doc_meta, pages
