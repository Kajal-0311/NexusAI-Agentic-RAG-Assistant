"""
Unit tests for Answer Generation, Grounding Verification, and Citation extraction.
"""
from backend.generation.answer_generator import AnswerGenerator
from backend.generation.llm_service import OllamaLLMService
from backend.generation.prompt_templates import format_rag_prompt


def test_format_rag_prompt():
    passages = [
        {"filename": "ai_paper.pdf", "page_number": 2, "chunk_id": "c1", "content": "Sample content"}
    ]
    formatted = format_rag_prompt("What is AI?", passages)
    assert "PASSAGE 1 [Doc: ai_paper.pdf | Page 2 | ID: c1]" in formatted
    assert "### USER QUESTION:\nWhat is AI?" in formatted


def test_citation_verification_rejects_hallucination():
    gen = AnswerGenerator()

    # Passages contain attention_paper.pdf on page 3
    retrieved_passages = [
        {"filename": "attention_paper.pdf", "page_number": 3, "chunk_id": "c_attn_3", "content": "Attention is all you need."}
    ]

    # Generated text cites both real doc and fake doc
    answer = (
        "The model uses self-attention [Doc: attention_paper.pdf, Page 3]. "
        "Also quantum computing is fast [Doc: quantum_fake.pdf, Page 99]."
    )

    verified = gen.extract_and_verify_citations(answer, retrieved_passages)

    # Only attention_paper.pdf should be verified; quantum_fake.pdf must be rejected
    assert len(verified) == 1
    assert verified[0].filename == "attention_paper.pdf"
    assert verified[0].page_number == 3


def test_insufficient_context_handling():
    gen = AnswerGenerator()
    # Empty retrieved passages
    result = gen.generate_answer("How does nuclear fusion work?", retrieved_passages=[])

    assert result.has_sufficient_context is False
    assert "insufficient evidence" in result.answer.lower()
    assert len(result.citations) == 0
