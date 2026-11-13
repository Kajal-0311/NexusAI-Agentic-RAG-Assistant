"""Custom exceptions for NexusAI."""


class NexusAIException(Exception):
    """Base exception for all NexusAI errors."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentProcessingError(NexusAIException):
    """Raised when PDF extraction or parsing fails."""
    pass


class ScannedPDFError(DocumentProcessingError):
    """Raised when a PDF contains insufficient text and requires OCR."""
    pass


class DuplicateDocumentError(NexusAIException):
    """Raised when a duplicate document hash is detected."""
    pass


class VectorStoreError(NexusAIException):
    """Raised on ChromaDB or vector store operational errors."""
    pass


class ModelUnavailableError(NexusAIException):
    """Raised when Ollama or a local model is unreachable."""
    pass


class RetrievalError(NexusAIException):
    """Raised when retrieval operations fail."""
    pass


class InsufficientEvidenceError(NexusAIException):
    """Raised when retrieved evidence cannot ground an answer."""
    pass


class AgentWorkflowError(NexusAIException):
    """Raised when agent workflow execution fails."""
    pass
