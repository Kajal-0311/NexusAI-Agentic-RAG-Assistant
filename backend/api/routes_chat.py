"""
Chat and agentic research interaction endpoint.
"""
from fastapi import APIRouter, HTTPException
from backend.agents.workflow import get_agent_workflow
from backend.api.schemas import ChatRequest, ChatResponse
from backend.core.logging_config import setup_logger
from backend.memory.conversation_memory import get_memory

logger = setup_logger("routes_chat")
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
def execute_research_chat(request: ChatRequest):
    """
    Executes the 7-step controlled agentic research workflow for a user query.
    Handles conversational memory, multi-turn follow-ups, and citation extraction.
    """
    memory = get_memory()
    workflow = get_agent_workflow()

    # Get or create active session
    session = memory.get_or_create_session(session_id=request.session_id)
    session_id = session.session_id

    # Retrieve sliding context window for follow-up resolution
    chat_history = memory.get_context_window(session_id=session_id, max_turns=4)

    # Persist the user message
    memory.add_user_message(session_id=session_id, content=request.query)

    try:
        # Run agentic workflow
        result = workflow.run(
            query=request.query,
            chat_history=chat_history,
            retrieval_mode=request.retrieval_mode,  # type: ignore
            top_k=request.top_k,
            document_ids=request.document_ids,
            use_reranker=request.use_reranker,
        )

        # Persist the assistant message
        memory.add_assistant_message(
            session_id=session_id,
            content=result.answer,
            citations=[c.model_dump() for c in result.citations],
            retrieved_passages=result.retrieved_passages,
            step_traces=[t.model_dump() for t in result.step_traces],
            latency_ms=result.total_latency_ms,
        )

        return ChatResponse(
            session_id=session_id,
            answer=result.answer,
            citations=result.citations,
            has_sufficient_context=result.has_sufficient_context,
            is_grounded=result.is_grounded,
            retrieval_mode_used=result.retrieval_mode_used,
            original_query=result.original_query,
            rewritten_query=result.rewritten_query,
            retrieved_passages=result.retrieved_passages,
            step_traces=result.step_traces,
            total_latency_ms=result.total_latency_ms,
        )

    except Exception as e:
        logger.error(f"Error during agentic chat execution: {e}")
        error_msg = f"An error occurred during workflow processing: {str(e)}"
        memory.add_assistant_message(session_id=session_id, content=error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
