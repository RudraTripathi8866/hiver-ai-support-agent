"""End-to-end support agent pipeline.

Orchestrates:
1. Intent classification
2. Historical resolution retrieval
3. Grounded response generation
4. Escalation evaluation (AUTO_HANDLE vs ESCALATE) with explicit reason and evidence
"""

from dataclasses import dataclass
from typing import List, Optional

from src.retrieval import RetrievedCase
from src.escalation import EscalationDecision


@dataclass
class AgentOutput:
    """Final unified response contract from the support agent."""

    customer_message: str
    intent: str
    intent_confidence: float
    decision: EscalationDecision
    escalation_reason: str
    response: Optional[str]
    evidence: List[RetrievedCase]


class SupportAgentPipeline:
    """Orchestrates end-to-end customer message processing."""

    def __init__(self) -> None:
        pass

    def process_message(self, message: str) -> AgentOutput:
        """Process an incoming customer support message through the full pipeline."""
        raise NotImplementedError("End-to-end pipeline execution will be implemented in subsequent phases.")
