"""
Recursive Character Chunking for NexusAI.
Splits documents while preserving page-level and document-level metadata.
"""
from typing import Dict, List, Optional
import uuid

from backend.core.config import settings
from backend.core.logging_config import setup_logger
from backend.ingestion.metadata import ChunkMetadata, DocumentChunk
from backend.ingestion.text_cleaner import detect_tables_and_equations

logger = setup_logger("chunker")


class RecursiveTextChunker:
    """
    Splits text recursively using prioritized separators:
    Paragraphs ("\n\n") -> Lines ("\n") -> Sentences (". ", "? ", "! ") -> Words (" ") -> Characters ("").
    Guarantees page-level and document-level metadata tracking on every generated chunk.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text into chunks under chunk_size."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        current_doc: List[str] = []
        total_len = 0

        for s in splits:
            s_len = len(s) + (len(separator) if current_doc else 0)
            if total_len + s_len > self.chunk_size:
                if current_doc:
                    doc_str = separator.join(current_doc)
                    if doc_str.strip():
                        final_chunks.append(doc_str.strip())
                    # Calculate overlap window
                    while total_len > self.chunk_overlap and current_doc:
                        removed = current_doc.pop(0)
                        total_len -= len(removed) + len(separator)

                if s_len > self.chunk_size and new_separators:
                    sub_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(sub_chunks)
                    current_doc = []
                    total_len = 0
                else:
                    current_doc = [s]
                    total_len = len(s)
            else:
                current_doc.append(s)
                total_len += s_len

        if current_doc:
            doc_str = separator.join(current_doc)
            if doc_str.strip():
                final_chunks.append(doc_str.strip())

        return final_chunks

    def chunk_pages(
        self,
        pages_data: List[Dict[str, any]],
        document_id: str,
        filename: str,
    ) -> List[DocumentChunk]:
        """
        Chunks extracted pages while strictly preserving 1-indexed page numbers.
        Returns a list of DocumentChunk objects.
        """
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in pages_data:
            page_num = page["page_number"]
            page_text = page["cleaned_text"]

            if not page_text or len(page_text.strip()) == 0:
                continue

            raw_chunks = self._split_text(page_text, self.separators)

            for c_idx, raw_chunk in enumerate(raw_chunks):
                if not raw_chunk.strip():
                    continue

                chunk_id = f"{document_id}_p{page_num}_c{c_idx}"
                has_tables, has_eqs = detect_tables_and_equations(raw_chunk)

                char_count = len(raw_chunk)
                token_estimate = max(1, char_count // 4)

                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    page_number=page_num,
                    chunk_index=global_chunk_idx,
                    char_count=char_count,
                    token_estimate=token_estimate,
                    has_tables=has_tables,
                    has_equations=has_eqs,
                )

                all_chunks.append(DocumentChunk(content=raw_chunk, metadata=meta))
                global_chunk_idx += 1

        logger.info(
            f"Document '{filename}' ({document_id}) chunked into {len(all_chunks)} chunks across {len(pages_data)} pages."
        )
        return all_chunks
