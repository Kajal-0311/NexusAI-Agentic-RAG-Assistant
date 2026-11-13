"""
ChromaDB vector store integration for NexusAI.
Handles persistent collection indexing, embedding storage, vector queries,
and cascading document chunk deletions.
"""
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.core.config import settings
from backend.core.exceptions import VectorStoreError
from backend.core.logging_config import setup_logger
from backend.embeddings.embedding_service import EmbeddingService, get_embedding_service
from backend.ingestion.metadata import DocumentChunk

logger = setup_logger("chroma_store")


class ChromaStore:
    """Manages persistent ChromaDB vector storage and semantic retrieval."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.persist_dir = str(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.embedding_service = embedding_service or get_embedding_service()

        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "NexusAI research documents", "hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB initialized at '{self.persist_dir}' with collection '{self.collection_name}'."
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise VectorStoreError(f"ChromaDB initialization failure: {str(e)}")

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64) -> int:
        """
        Computes embeddings and indexes a list of DocumentChunks into ChromaDB.
        Processes in batches to avoid memory bottlenecks.
        """
        if not chunks:
            return 0

        total_added = 0
        try:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                ids = [c.metadata.chunk_id for c in batch]
                documents = [c.content for c in batch]
                metadatas = [c.metadata.to_chroma_dict() for c in batch]

                embeddings = self.embedding_service.embed_documents(documents)

                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                total_added += len(batch)

            logger.info(f"Successfully indexed {total_added} chunks into ChromaDB.")
            return total_added
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}")
            raise VectorStoreError(f"Failed to store chunks: {str(e)}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic vector search against the collection.
        Returns list of result dicts with normalized similarity scores [0, 1].
        """
        if not query.strip() or self.collection.count() == 0:
            return []

        try:
            query_embedding = self.embedding_service.embed_query(query)

            where_clause = None
            if document_ids:
                if len(document_ids) == 1:
                    where_clause = {"document_id": document_ids[0]}
                else:
                    where_clause = {"document_id": {"$in": document_ids}}

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, max(1, self.collection.count())),
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            formatted_results: List[Dict[str, Any]] = []
            if results and results["ids"] and len(results["ids"][0]) > 0:
                ids = results["ids"][0]
                docs = results["documents"][0] if results["documents"] else [""] * len(ids)
                metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
                distances = results["distances"][0] if results["distances"] else [1.0] * len(ids)

                for chunk_id, content, meta, dist in zip(ids, docs, metas, distances):
                    # Chroma cosine distance is in [0, 2]; convert to similarity score in [0, 1]
                    similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                    formatted_results.append(
                        {
                            "chunk_id": chunk_id,
                            "content": content,
                            "document_id": meta.get("document_id", ""),
                            "filename": meta.get("filename", "unknown"),
                            "page_number": int(meta.get("page_number", 1)),
                            "chunk_index": int(meta.get("chunk_index", 0)),
                            "score": float(round(similarity, 4)),
                            "retrieval_strategy": "semantic",
                        }
                    )

            # Sort descending by similarity score
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            return formatted_results[:top_k]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise VectorStoreError(f"Vector search failed: {str(e)}")

    def delete_document(self, document_id: str) -> None:
        """Deletes all chunks associated with a specific document ID."""
        try:
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted all vector records for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise VectorStoreError(f"Delete failed: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the current vector collection."""
        try:
            total_chunks = self.collection.count()
            # Fetch metadatas to find unique documents
            results = self.collection.get(include=["metadatas"])
            doc_ids = set()
            filenames = set()
            if results and results["metadatas"]:
                for m in results["metadatas"]:
                    if m.get("document_id"):
                        doc_ids.add(m["document_id"])
                    if m.get("filename"):
                        filenames.add(m["filename"])

            return {
                "collection_name": self.collection_name,
                "total_chunks": total_chunks,
                "unique_documents": len(doc_ids),
                "indexed_files": sorted(list(filenames)),
            }
        except Exception as e:
            logger.warning(f"Failed to get vector store stats: {e}")
            return {
                "collection_name": self.collection_name,
                "total_chunks": 0,
                "unique_documents": 0,
                "indexed_files": [],
            }

    def reset(self) -> None:
        """Deletes and recreates the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "NexusAI research documents", "hnsw:space": "cosine"},
            )
            logger.info(f"Reset collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise VectorStoreError(f"Reset failed: {str(e)}")


_default_vector_store: Optional[ChromaStore] = None


def get_vector_store() -> ChromaStore:
    """Singleton getter for global ChromaStore instance."""
    global _default_vector_store
    if _default_vector_store is None:
        _default_vector_store = ChromaStore()
    return _default_vector_store
