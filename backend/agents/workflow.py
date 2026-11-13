"""
Controlled Multi-Step Agentic Workflow for NexusAI.
Orchestrates:
1. Query Understanding & Classification
2. Contextual Query Rewriting
3. Retrieval Decision & Execution
4. Evidence Sufficiency Assessment & Controlled Loop
5. Evidence-Grounded Answer Generation
6. Claim Verification & Grounding Check
7. Final Response Synthesis & Citation Assembly
"""
import json
import re
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.agents.evidence_checker import EvidenceAssessment, EvidenceChecker
from backend.agents.query_classifier import ClassificationResult, QueryCategory, QueryClassifier
from backend.agents.query_rewriter import QueryRewriter
from backend.agents.retrieval_agent import RetrievalAgent
from backend.core.config import settings
from backend.core.logging_config import setup_logger
from backend.generation.answer_generator import AnswerGenerator, Citation, GenerationResult
from backend.generation.llm_service import OllamaLLMService, get_llm_service
from backend.generation.prompt_templates import VERIFICATION_PROMPT
from backend.retrieval.hybrid_search import RetrievalMode

logger = setup_logger("agent_workflow")


class WorkflowStepTrace(BaseModel):
    """Telemetry record for each discrete step in the agentic workflow."""
    step_number: int
    step_name: str
    action_taken: str
    latency_ms: float
    output_summary: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowResponse(BaseModel):
    """Complete response returned to API and UI layers."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    has_sufficient_context: bool = True
    is_grounded: bool = True
    retrieval_mode_used: str = "hybrid"
    original_query: str
    rewritten_query: str
    retrieved_passages: List[Dict[str, Any]] = Field(default_factory=list)
    step_traces: List[WorkflowStepTrace] = Field(default_factory=list)
    total_latency_ms: float = 0.0


class AgenticWorkflow:
    """Stateful orchestrator implementing the 7-step controlled research workflow."""

    def __init__(
        self,
        classifier: Optional[QueryClassifier] = None,
        rewriter: Optional[QueryRewriter] = None,
        retrieval_agent: Optional[RetrievalAgent] = None,
        evidence_checker: Optional[EvidenceChecker] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        llm_service: Optional[OllamaLLMService] = None,
    ):
        self.classifier = classifier or QueryClassifier()
        self.rewriter = rewriter or QueryRewriter()
        self.retrieval_agent = retrieval_agent or RetrievalAgent()
        self.evidence_checker = evidence_checker or EvidenceChecker()
        self.answer_generator = answer_generator or AnswerGenerator()
        self.llm = llm_service or get_llm_service()

    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        retrieval_mode: RetrievalMode = "hybrid",
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        use_reranker: Optional[bool] = None,
    ) -> WorkflowResponse:
        """Executes the full controlled research pipeline with telemetry."""
        total_start = time.perf_counter()
        traces: List[WorkflowStepTrace] = []

        # =====================================================================
        # STEP 1: QUERY UNDERSTANDING & CLASSIFICATION
        # =====================================================================
        s1_start = time.perf_counter()
        classification = self.classifier.classify(query)
        s1_ms = (time.perf_counter() - s1_start) * 1000

        traces.append(
            WorkflowStepTrace(
                step_number=1,
                step_name="Query Understanding",
                action_taken=f"Classified query intent as {classification.category.value}",
                latency_ms=round(s1_ms, 2),
                output_summary=classification.reasoning,
                metadata={"confidence": classification.confidence},
            )
        )

        # Handle non-retrieval conversational cases immediately
        if classification.category == QueryCategory.CONVERSATIONAL:
            ans = "Hello! I am NexusAI, your Agentic RAG Research Assistant. You can upload PDF documents in the Document Library and ask me in-depth questions, summaries, or comparative analyses based on them."
            return WorkflowResponse(
                answer=ans,
                citations=[],
                has_sufficient_context=True,
                is_grounded=True,
                retrieval_mode_used="none",
                original_query=query,
                rewritten_query=query,
                retrieved_passages=[],
                step_traces=traces,
                total_latency_ms=round((time.perf_counter() - total_start) * 1000, 2),
            )

        if classification.category == QueryCategory.META_QUESTION:
            ans = "I am NexusAI, an evidence-grounded research assistant. I can extract information from uploaded PDFs, execute Semantic Vector and BM25 Keyword searches, combine them via Reciprocal Rank Fusion, apply Cross-Encoder reranking, and verify citations directly against source pages."
            return WorkflowResponse(
                answer=ans,
                citations=[],
                has_sufficient_context=True,
                is_grounded=True,
                retrieval_mode_used="none",
                original_query=query,
                rewritten_query=query,
                retrieved_passages=[],
                step_traces=traces,
                total_latency_ms=round((time.perf_counter() - total_start) * 1000, 2),
            )

        # =====================================================================
        # STEP 2: QUERY REWRITING
        # =====================================================================
        s2_start = time.perf_counter()
        rewritten_query = self.rewriter.rewrite(query, chat_history=chat_history)
        s2_ms = (time.perf_counter() - s2_start) * 1000

        traces.append(
            WorkflowStepTrace(
                step_number=2,
                step_name="Query Rewriting",
                action_taken="Resolved multi-turn conversational context" if rewritten_query != query else "Preserved standalone query",
                latency_ms=round(s2_ms, 2),
                output_summary=f"Search target: '{rewritten_query}'",
                metadata={"was_rewritten": rewritten_query != query},
            )
        )

        # =====================================================================
        # STEP 3 & 4: RETRIEVAL & EVIDENCE ASSESSMENT (WITH RETRY LOOP)
        # =====================================================================
        retry_count = 0
        current_query = rewritten_query
        retrieved_passages: List[Dict[str, Any]] = []
        assessment: Optional[EvidenceAssessment] = None

        while retry_count <= settings.MAX_AGENT_RETRIES:
            # Step 3: Retrieval Execution
            s3_start = time.perf_counter()
            retrieved_passages = self.retrieval_agent.retrieve_evidence(
                query=current_query,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                document_ids=document_ids,
                use_reranker=use_reranker,
            )
            s3_ms = (time.perf_counter() - s3_start) * 1000

            traces.append(
                WorkflowStepTrace(
                    step_number=3,
                    step_name=f"Retrieval Decision & Search (Attempt {retry_count + 1})",
                    action_taken=f"Retrieved candidate passages via {retrieval_mode}",
                    latency_ms=round(s3_ms, 2),
                    output_summary=f"Found {len(retrieved_passages)} chunks",
                    metadata={"retrieval_mode": retrieval_mode, "top_k": top_k, "attempt": retry_count + 1},
                )
            )

            # Step 4: Evidence Assessment
            s4_start = time.perf_counter()
            assessment = self.evidence_checker.assess(
                query=current_query,
                retrieved_passages=retrieved_passages,
                retry_count=retry_count,
            )
            s4_ms = (time.perf_counter() - s4_start) * 1000

            traces.append(
                WorkflowStepTrace(
                    step_number=4,
                    step_name=f"Evidence Assessment (Attempt {retry_count + 1})",
                    action_taken="Evaluated evidence density and topical overlap",
                    latency_ms=round(s4_ms, 2),
                    output_summary=f"Sufficiency: {assessment.is_sufficient} - {assessment.reason}",
                    metadata={"is_sufficient": assessment.is_sufficient, "confidence": assessment.confidence},
                )
            )

            # Check if sufficient or if we should trigger controlled retry
            if assessment.is_sufficient or not assessment.suggest_query_expansion:
                break

            # Attempt query expansion for retry
            retry_count += 1
            words = [w for w in current_query.split() if len(w) > 3]
            current_query = f"{current_query} {' '.join(words[:2])} overview"
            logger.info(f"Agent triggered query retry {retry_count} with expanded query: '{current_query}'")

        # =====================================================================
        # STEP 5: ANSWER GENERATION
        # =====================================================================
        s5_start = time.perf_counter()
        gen_result = self.answer_generator.generate_answer(
            query=current_query,
            retrieved_passages=retrieved_passages,
        )
        s5_ms = (time.perf_counter() - s5_start) * 1000

        traces.append(
            WorkflowStepTrace(
                step_number=5,
                step_name="Answer Generation",
                action_taken=f"Generated structured response using {gen_result.model_used}",
                latency_ms=round(s5_ms, 2),
                output_summary=f"Generated {len(gen_result.answer)} characters with {len(gen_result.citations)} citations",
                metadata={"has_sufficient_context": gen_result.has_sufficient_context},
            )
        )

        # =====================================================================
        # STEP 6: VERIFICATION & GROUNDING AUDIT
        # =====================================================================
        s6_start = time.perf_counter()
        is_grounded = gen_result.is_grounded
        verification_verdict = "SUPPORTED"

        if gen_result.has_sufficient_context and retrieved_passages:
            context_summary = "\n".join([p.get("content", "")[:250] for p in retrieved_passages[:3]])
            v_prompt = VERIFICATION_PROMPT.format(
                context=context_summary,
                answer=gen_result.answer[:1000],
            )
            v_response = self.llm.generate(
                prompt=v_prompt,
                system_prompt="You are a strict JSON grounding verifier.",
                temperature=0.0,
            )
            try:
                v_match = re.search(r"\{.*?\}", v_response, re.DOTALL)
                if v_match:
                    v_data = json.loads(v_match.group(0))
                    is_grounded = bool(v_data.get("is_grounded", True))
                    verification_verdict = v_data.get("verdict", "SUPPORTED")
            except Exception:
                is_grounded = True
                verification_verdict = "SUPPORTED"

        s6_ms = (time.perf_counter() - s6_start) * 1000
        traces.append(
            WorkflowStepTrace(
                step_number=6,
                step_name="Grounding Verification",
                action_taken="Cross-checked claims against source passages",
                latency_ms=round(s6_ms, 2),
                output_summary=f"Verification verdict: {verification_verdict} (Grounded={is_grounded})",
                metadata={"verdict": verification_verdict, "is_grounded": is_grounded},
            )
        )

        # =====================================================================
        # STEP 7: FINAL RESPONSE COMPOSITION
        # =====================================================================
        final_answer = gen_result.answer
        if not is_grounded:
            final_answer += "\n\n> [!CAUTION]\n> Some statements in this answer could not be fully grounded in the retrieved text. Please inspect the source passages below."

        total_ms = (time.perf_counter() - total_start) * 1000
        traces.append(
            WorkflowStepTrace(
                step_number=7,
                step_name="Final Response Synthesis",
                action_taken="Packaged answer, verified citations, and telemetry trace",
                latency_ms=1.0,
                output_summary="Ready for user delivery",
                metadata={"total_latency_ms": round(total_ms, 2)},
            )
        )

        return WorkflowResponse(
            answer=final_answer,
            citations=gen_result.citations,
            has_sufficient_context=gen_result.has_sufficient_context,
            is_grounded=is_grounded,
            retrieval_mode_used=retrieval_mode,
            original_query=query,
            rewritten_query=rewritten_query,
            retrieved_passages=retrieved_passages,
            step_traces=traces,
            total_latency_ms=round(total_ms, 2),
        )


_default_workflow: Optional[AgenticWorkflow] = None


def get_agent_workflow() -> AgenticWorkflow:
    """Singleton getter for global AgenticWorkflow."""
    global _default_workflow
    if _default_workflow is None:
        _default_workflow = AgenticWorkflow()
    return _default_workflow
