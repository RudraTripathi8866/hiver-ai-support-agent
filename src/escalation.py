"""Escalation decision logic module.

Evaluates whether an incoming message should be automatically handled (AUTO_HANDLE)
or escalated to a human agent (ESCALATE), based on:
1. Intent policy rules (e.g. account compromise, billing dispute, high distress)
2. Intent classifier confidence
3. Retrieval relevance score & grounding evidence quality
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from src.retrieval import RetrievedCase


class EscalationDecision(str, Enum):
    """Possible escalation outcomes."""

    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE = "ESCALATE"


@dataclass
class EscalationResult:
    """Escalation assessment output contract."""

    decision: EscalationDecision
    reason: str
    confidence: float
    supporting_evidence: List[RetrievedCase]


class EscalationEngine:
    """Determines auto-handling eligibility vs. human handoff."""

    def __init__(self, confidence_threshold: float = 0.75) -> None:
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieved_cases: List[RetrievedCase],
    ) -> EscalationResult:
        """Evaluate message and evidence to render escalation decision."""
        raise NotImplementedError("Escalation engine will be implemented in subsequent phases.")
