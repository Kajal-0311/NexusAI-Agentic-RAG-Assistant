"""
Step 1: Query Understanding & Intent Classification.
"""
from enum import Enum
import json
import re
from typing import Optional
from pydantic import BaseModel, Field

from backend.core.logging_config import setup_logger
from backend.generation.llm_service import OllamaLLMService, get_llm_service
from backend.generation.prompt_templates import QUERY_CLASSIFIER_PROMPT

logger = setup_logger("query_classifier")


class QueryCategory(str, Enum):
    RETRIEVAL_REQUIRED = "RETRIEVAL_REQUIRED"
    CONVERSATIONAL = "CONVERSATIONAL"
    META_QUESTION = "META_QUESTION"
    AMBIGUOUS = "AMBIGUOUS"


class ClassificationResult(BaseModel):
    category: QueryCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    requires_retrieval: bool


class QueryClassifier:
    """Classifies incoming user questions to direct workflow routing."""

    def __init__(self, llm_service: Optional[OllamaLLMService] = None):
        self.llm = llm_service or get_llm_service()

    def classify(self, query: str) -> ClassificationResult:
        """Determines intent and whether retrieval is needed."""
        clean_q = query.strip()
        if not clean_q:
            return ClassificationResult(
                category=QueryCategory.AMBIGUOUS,
                confidence=1.0,
                reasoning="Empty query provided.",
                requires_retrieval=False,
            )

        # Quick heuristic check for obvious conversational greetings
        lower_q = clean_q.lower()
        if lower_q in ["hi", "hello", "hey", "good morning", "good evening", "how are you", "thanks", "thank you"]:
            return ClassificationResult(
                category=QueryCategory.CONVERSATIONAL,
                confidence=0.98,
                reasoning="Common conversational greeting detected.",
                requires_retrieval=False,
            )

        prompt = QUERY_CLASSIFIER_PROMPT.format(query=clean_q)
        raw_response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a strict JSON classifier. Always return valid JSON only.",
            temperature=0.0,
        )

        try:
            # Extract JSON block
            json_match = re.search(r"\{.*?\}", raw_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                cat_str = data.get("classification", "RETRIEVAL_REQUIRED").upper()
                category = (
                    QueryCategory(cat_str)
                    if cat_str in QueryCategory.__members__
                    else QueryCategory.RETRIEVAL_REQUIRED
                )
                confidence = float(data.get("confidence", 0.85))
                reasoning = data.get("reasoning", "Classified via LLM analysis.")
            else:
                category = QueryCategory.RETRIEVAL_REQUIRED
                confidence = 0.70
                reasoning = "Defaulted to retrieval after non-JSON response."
        except Exception as e:
            logger.warning(f"Classification parsing failed: {e}. Defaulting to RETRIEVAL_REQUIRED.")
            category = QueryCategory.RETRIEVAL_REQUIRED
            confidence = 0.70
            reasoning = "Defaulted to retrieval due to parse error."

        requires_retrieval = category == QueryCategory.RETRIEVAL_REQUIRED
        return ClassificationResult(
            category=category,
            confidence=confidence,
            reasoning=reasoning,
            requires_retrieval=requires_retrieval,
        )
