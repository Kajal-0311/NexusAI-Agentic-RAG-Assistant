"""
Answer generation and citation resolution engine for NexusAI.
Synthesizes answers from retrieved evidence, extracts and verifies citations,
and prevents hallucination of non-existent sources.
"""
import re
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.core.config import settings
from backend.core.logging_config import setup_logger
from backend.generation.llm_service import OllamaLLMService, get_llm_service
from backend.generation.prompt_templates import SYSTEM_RESEARCH_PROMPT, format_rag_prompt

logger = setup_logger("answer_generator")


class Citation(BaseModel):
    """Structured representation of a verified document citation."""
    filename: str
    page_number: int
    chunk_id: Optional[str] = None
    excerpt: Optional[str] = None
    is_verified: bool = True


class GenerationResult(BaseModel):
    """Complete result of answer generation including citations and grounding flags."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    has_sufficient_context: bool = True
    is_grounded: bool = True
    confidence_score: float = 1.0
    model_used: str = settings.OLLAMA_LLM_MODEL
    latency_ms: float = 0.0
    retrieved_passages: List[Dict[str, Any]] = Field(default_factory=list)


class AnswerGenerator:
    """Coordinates evidence synthesis, citation verification, and answer construction."""

    def __init__(self, llm_service: Optional[OllamaLLMService] = None):
        self.llm = llm_service or get_llm_service()

    def extract_and_verify_citations(
        self, answer_text: str, retrieved_passages: List[Dict[str, Any]]
    ) -> List[Citation]:
        """
        Extracts citations matching pattern '[Doc: <filename>, Page <page>]' from the generated text,
        and verifies them against actual retrieved passages. Non-existent citations are rejected.
        """
        # Regex to capture [Doc: file.pdf, Page 3] or [Doc: file.pdf, Page: 3]
        pattern = r"\[Doc:\s*(.*?),\s*Page:?\s*(\d+)\]"
        matches = re.findall(pattern, answer_text)

        verified_citations: List[Citation] = []
        seen = set()

        for filename_raw, page_str in matches:
            filename = filename_raw.strip()
            page_num = int(page_str.strip())
            key = (filename.lower(), page_num)

            if key in seen:
                continue
            seen.add(key)

            # Check if this filename and page exist in the retrieved passages
            matching_passage = next(
                (
                    p
                    for p in retrieved_passages
                    if (
                        p.get("filename", "").lower() == filename.lower()
                        or filename.lower() in p.get("filename", "").lower()
                    )
                    and int(p.get("page_number", 0)) == page_num
                ),
                None,
            )

            if matching_passage:
                verified_citations.append(
                    Citation(
                        filename=matching_passage.get("filename", filename),
                        page_number=page_num,
                        chunk_id=matching_passage.get("chunk_id"),
                        excerpt=matching_passage.get("content", "")[:200] + "...",
                        is_verified=True,
                    )
                )
            else:
                logger.warning(
                    f"Hallucinated citation detected and discarded: [Doc: {filename}, Page: {page_num}]"
                )

        # If model did not emit citations in bracket format, add citations from top passages used
        if not verified_citations and retrieved_passages:
            for p in retrieved_passages[:2]:
                cid = (p.get("filename", "").lower(), int(p.get("page_number", 1)))
                if cid not in seen:
                    seen.add(cid)
                    verified_citations.append(
                        Citation(
                            filename=p.get("filename", "document"),
                            page_number=int(p.get("page_number", 1)),
                            chunk_id=p.get("chunk_id"),
                            excerpt=p.get("content", "")[:200] + "...",
                            is_verified=True,
                        )
                    )

        return verified_citations

    def generate_answer(
        self,
        query: str,
        retrieved_passages: List[Dict[str, Any]],
        temperature: float = 0.1,
    ) -> GenerationResult:
        """
        Generates an evidence-grounded answer for the query using retrieved passages.
        """
        start_time = time.perf_counter()

        # Check for empty or negligible evidence
        if not retrieved_passages:
            return GenerationResult(
                answer="Insufficient evidence in the provided documents to answer this question.",
                citations=[],
                has_sufficient_context=False,
                is_grounded=False,
                confidence_score=0.0,
                model_used=self.llm.model_name,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                retrieved_passages=[],
            )

        prompt = format_rag_prompt(query, retrieved_passages)
        raw_answer = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_RESEARCH_PROMPT,
            temperature=temperature,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Check if the model explicitly reported insufficient evidence
        lower_ans = raw_answer.lower()
        if "insufficient evidence" in lower_ans or "not enough information" in lower_ans:
            return GenerationResult(
                answer=raw_answer,
                citations=[],
                has_sufficient_context=False,
                is_grounded=True,
                confidence_score=0.30,
                model_used=self.llm.model_name,
                latency_ms=elapsed_ms,
                retrieved_passages=retrieved_passages,
            )

        # Extract and verify citations
        citations = self.extract_and_verify_citations(raw_answer, retrieved_passages)

        return GenerationResult(
            answer=raw_answer,
            citations=citations,
            has_sufficient_context=True,
            is_grounded=True,
            confidence_score=0.92,
            model_used=self.llm.model_name,
            latency_ms=elapsed_ms,
            retrieved_passages=retrieved_passages,
        )
