"""Generation module for NexusAI."""
from .prompt_templates import (
    SYSTEM_RESEARCH_PROMPT,
    format_rag_prompt,
    QUERY_CLASSIFIER_PROMPT,
    QUERY_REWRITER_PROMPT,
    VERIFICATION_PROMPT,
)
from .llm_service import OllamaLLMService, get_llm_service
from .answer_generator import AnswerGenerator, GenerationResult

__all__ = [
    "SYSTEM_RESEARCH_PROMPT",
    "format_rag_prompt",
    "QUERY_CLASSIFIER_PROMPT",
    "QUERY_REWRITER_PROMPT",
    "VERIFICATION_PROMPT",
    "OllamaLLMService",
    "get_llm_service",
    "AnswerGenerator",
    "GenerationResult",
]
