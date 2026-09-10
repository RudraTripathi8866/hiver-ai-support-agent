"""Intent classification module.

Classifies incoming customer messages into brand-specific intents defined in the taxonomy.
Produces predicted intent, confidence score, and classification rationale.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IntentClassificationResult:
    """Output contract for intent classification."""

    predicted_intent: str
    confidence: float
    rationale: Optional[str] = None


class IntentClassifier:
    """Base classifier interface for intent identification."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name

    def classify(self, message: str) -> IntentClassificationResult:
        """Classify a customer message into an intent category."""
        raise NotImplementedError("Intent classification will be implemented in subsequent phases.")
