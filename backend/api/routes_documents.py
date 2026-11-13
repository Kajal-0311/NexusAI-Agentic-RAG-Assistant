"""
Document management endpoints: upload, listing, deletion, and re-indexing.
"""
from pathlib import Path
from typing import List
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.api.schemas import DocumentResponse, DocumentUploadResponse
from backend.core.config import settings
from backend.core.exceptions import (
    DocumentProcessingError,
    DuplicateDocumentError,
    ScannedPDFError,
)
from backend.core.logging_config import setup_logger
from backend.database.database import get_db
from backend.database.models import DocumentRecord
from backend.ingestion.chunking import RecursiveTextChunker
from backend.ingestion.pdf_loader import PDFLoader
from backend.vectorstore.chroma_store import get_vector_store

logger = setup_logger("routes_documents")
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Accepts one or multiple PDF documents, validates them, detects duplicate hashes,
    extracts per-page text, performs recursive character chunking, generates embeddings,
    and indexes chunks into ChromaDB and SQLite.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    db = get_db()
    vector_store = get_vector_store()
    loader = PDFLoader()
    chunker = RecursiveTextChunker()

    existing_docs = db.get_all_documents()
    existing_hashes = {d.file_hash for d in existing_docs}

    processed_responses: List[DocumentResponse] = []
    errors: List[str] = []

    for file in files:
        filename = file.filename or "uploaded.pdf"
        try:
            file_bytes = await file.read()
            doc_meta, pages = loader.process_and_save(
                filename=filename,
                file_bytes=file_bytes,
                existing_hashes=existing_hashes,
            )

            # Insert document into SQLite
            db_doc = DocumentRecord(
                document_id=doc_meta.document_id,
                filename=doc_meta.filename,
                file_path=doc_meta.file_path,
                file_hash=doc_meta.file_hash,
                file_size_bytes=doc_meta.file_size_bytes,
                total_pages=doc_meta.total_pages,
                total_chunks=0,
                uploaded_at=doc_meta.uploaded_at,
                is_processed=False,
                is_scanned=doc_meta.is_scanned,
                error_message=doc_meta.error_message,
                collection_name=doc_meta.collection_name,
            )
            db.insert_document(db_doc)
            existing_hashes.add(doc_meta.file_hash)

            # Chunk document
            chunks = chunker.chunk_pages(
                pages_data=pages,
                document_id=doc_meta.document_id,
                filename=doc_meta.filename,
            )

            # Index chunks in vector store
            if chunks:
                indexed_count = vector_store.add_chunks(chunks)
                db.update_document_status(
                    document_id=doc_meta.document_id,
                    is_processed=True,
                    total_chunks=indexed_count,
                    error_message=doc_meta.error_message,
                )
            else:
                db.update_document_status(
                    document_id=doc_meta.document_id,
                    is_processed=True,
                    total_chunks=0,
                    error_message="Document generated 0 chunks.",
                )

            updated = db.get_document(doc_meta.document_id)
            if updated:
                processed_responses.append(DocumentResponse(**updated.model_dump()))

        except DuplicateDocumentError as e:
            errors.append(f"'{filename}': {str(e)}")
        except DocumentProcessingError as e:
            errors.append(f"'{filename}': {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error processing '{filename}': {e}")
            errors.append(f"'{filename}': Unexpected processing error: {str(e)}")

    if not processed_responses and errors:
        raise HTTPException(status_code=400, detail=" | ".join(errors))

    msg = f"Successfully uploaded and indexed {len(processed_responses)} document(s)."
    if errors:
        msg += f" Warnings/Skipped: {'; '.join(errors)}"

    return DocumentUploadResponse(message=msg, documents=processed_responses)


@router.get("", response_model=List[DocumentResponse])
def list_documents():
    """Lists all uploaded documents stored in the database."""
    db = get_db()
    docs = db.get_all_documents()
    return [DocumentResponse(**d.model_dump()) for d in docs]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str):
    """Retrieves document metadata by ID."""
    db = get_db()
    doc = db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentResponse(**doc.model_dump())


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(document_id: str):
    """Deletes a document from disk, removes its SQLite record, and purges all its ChromaDB chunks."""
    db = get_db()
    vector_store = get_vector_store()

    doc = db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # 1. Delete vector store chunks
    try:
        vector_store.delete_document(document_id)
    except Exception as e:
        logger.warning(f"Error purging vector store chunks for {document_id}: {e}")

    # 2. Delete file from disk
    try:
        p = Path(doc.file_path)
        if p.exists():
            p.unlink()
    except Exception as e:
        logger.warning(f"Error deleting file {doc.file_path}: {e}")

    # 3. Delete database record
    db.delete_document(document_id)

    return {"message": f"Document '{doc.filename}' ({document_id}) deleted successfully."}


@router.post("/{document_id}/reindex", response_model=DocumentResponse)
def reindex_document(document_id: str):
    """Re-chunks and re-embeds an existing document with current configuration."""
    db = get_db()
    vector_store = get_vector_store()
    doc = db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = Path(doc.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Underlying document file not found on disk.")

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    loader = PDFLoader()
    chunker = RecursiveTextChunker()

    # Re-extract
    pages, is_scanned = loader.extract_pages(file_bytes, doc.filename)

    # Purge old chunks from vector store
    vector_store.delete_document(document_id)

    # Re-chunk and re-index
    chunks = chunker.chunk_pages(pages, document_id, doc.filename)
    indexed_count = vector_store.add_chunks(chunks)

    db.update_document_status(
        document_id=document_id,
        is_processed=True,
        total_chunks=indexed_count,
        error_message=None if not is_scanned else doc.error_message,
    )

    updated = db.get_document(document_id)
    return DocumentResponse(**updated.model_dump())
