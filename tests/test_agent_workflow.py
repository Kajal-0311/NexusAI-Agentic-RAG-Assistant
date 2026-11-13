"""
Unit tests for the 7-step Agentic Workflow and Sub-Agents.
"""
from backend.agents.evidence_checker import EvidenceChecker
from backend.agents.query_classifier import QueryCategory, QueryClassifier
from backend.agents.query_rewriter import QueryRewriter
from backend.agents.workflow import AgenticWorkflow
from backend.generation.answer_generator import AnswerGenerator
from backend.retrieval.hybrid_search import HybridRetriever
from backend.retrieval.keyword_search import BM25Retriever
from backend.retrieval.semantic_search import SemanticRetriever


def test_query_classifier():
    classifier = QueryClassifier()

    c_conv = classifier.classify("Hello there, how are you today?")
    assert c_conv.category == QueryCategory.CONVERSATIONAL
    assert c_conv.requires_retrieval is False

    c_meta = classifier.classify("What can you do and what documents are loaded?")
    assert c_meta.category == QueryCategory.META_QUESTION

    c_rag = classifier.classify("Explain the self-attention mechanism in Transformers.")
    assert c_rag.category == QueryCategory.RETRIEVAL_REQUIRED
    assert c_rag.requires_retrieval is True


def test_query_rewriter():
    rewriter = QueryRewriter()

    history = [
        {"role": "user", "content": "What is the Transformer architecture?"},
        {"role": "assistant", "content": "It is a deep learning model based on self-attention mechanisms."},
    ]

    # Follow-up with pronoun 'it'
    rewritten = rewriter.rewrite("What are its main advantages?", chat_history=history)
    assert len(rewritten) > 0


def test_evidence_checker_sufficiency():
    checker = EvidenceChecker()

    # Empty context -> insufficient
    assess_empty = checker.assess("Transformer", retrieved_passages=[], retry_count=0)
    assert assess_empty.is_sufficient is False
    assert assess_empty.suggest_query_expansion is True

    # Relevant context with matching keyword
    passages = [{"content": "The Transformer uses multi-head attention.", "score": 0.85}]
    assess_good = checker.assess("Transformer attention", retrieved_passages=passages, retry_count=0)
    assert assess_good.is_sufficient is True


def test_agentic_workflow_end_to_end(test_vector_store, sample_chunks):
    test_vector_store.add_chunks(sample_chunks)

    workflow = AgenticWorkflow(
        classifier=QueryClassifier(),
        rewriter=QueryRewriter(),
        retrieval_agent=None,  # Uses default
        evidence_checker=EvidenceChecker(),
        answer_generator=AnswerGenerator(),
    )

    response = workflow.run(
        query="How does Reciprocal Rank Fusion combine rankings?",
        retrieval_mode="hybrid",
        top_k=3,
    )

    assert response.answer is not None
    assert len(response.answer) > 0
    assert len(response.step_traces) >= 5
    # Verify Step 1 through Step 7 traces exist
    step_numbers = [t.step_number for t in response.step_traces]
    assert 1 in step_numbers  # Understanding
    assert 2 in step_numbers  # Rewriting
    assert 3 in step_numbers  # Retrieval
    assert 4 in step_numbers  # Assessment
    assert 5 in step_numbers  # Generation
    assert 6 in step_numbers  # Verification
    assert 7 in step_numbers  # Synthesis
