"""
Step 2: Contextual Query Rewriter for multi-turn conversations.
Resolves pronouns and context references into a self-contained retrieval query.
"""
from typing import List, Optional
from backend.core.logging_config import setup_logger
from backend.generation.llm_service import OllamaLLMService, get_llm_service
from backend.generation.prompt_templates import QUERY_REWRITER_PROMPT

logger = setup_logger("query_rewriter")


class QueryRewriter:
    """Rewrites conversational follow-up questions into standalone retrieval queries."""

    def __init__(self, llm_service: Optional[OllamaLLMService] = None):
        self.llm = llm_service or get_llm_service()

    def rewrite(
        self,
        query: str,
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """
        Takes the user query and recent chat messages.
        If history exists and the query contains pronouns or conversational ellipses,
        produces an expanded standalone query.
        """
        if not chat_history:
            return query.strip()

        # Format last 4 messages into dialogue history
        recent_turns = chat_history[-4:]
        history_lines = []
        for msg in recent_turns:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "").strip()
            if content:
                # Truncate very long assistant replies to keep context focused
                truncated_content = content[:300] + "..." if len(content) > 300 else content
                history_lines.append(f"{role}: {truncated_content}")

        history_str = "\n".join(history_lines)

        # Quick heuristic: if question does not contain referential words, keep original
        referential_terms = ["it", "its", "they", "them", "their", "this", "that", "these", "those", "above", "former", "latter", "again", "also"]
        query_words = set(query.lower().split())
        has_referential = bool(query_words.intersection(referential_terms))

        if not has_referential and len(query.split()) > 5:
            # Already a specific standalone query
            return query.strip()

        prompt = QUERY_REWRITER_PROMPT.format(history=history_str, query=query.strip())
        rewritten = self.llm.generate(
            prompt=prompt,
            system_prompt="You are an expert query reformulation engine. Output only the rewritten query.",
            temperature=0.0,
        )

        cleaned_rewritten = rewritten.strip().strip('"').strip("'")
        if cleaned_rewritten and len(cleaned_rewritten) > 3:
            logger.info(f"Rewrote query '{query}' -> '{cleaned_rewritten}'")
            return cleaned_rewritten

        return query.strip()
