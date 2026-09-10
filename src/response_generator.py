"""Response generation module.

Generates customer support responses strictly grounded in historical brand resolutions
and retrieved context. Prevents hallucinations and maintains brand voice.
"""

from dataclasses import dataclass
from typing import List, Optional
from src.retrieval import RetrievedCase


@dataclass
class GenerationResult:
    """Generated response output contract."""

    response_text: str
    grounded_evidence: List[RetrievedCase]
    generation_notes: Optional[str] = None


class ResponseGenerator:
    """Interface for generating grounded support responses."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name

    def generate(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_cases: List[RetrievedCase],
    ) -> GenerationResult:
        """Generate a response grounded in retrieved historical evidence."""
        raise NotImplementedError("Response generator will be implemented in subsequent phases.")
