"""Sanity and contract tests for the AI Support Agent project."""

from src.config import config
from src.escalation import EscalationDecision


def test_config_defaults():
    """Verify default configuration values load correctly."""
    assert config.target_brand is not None
    assert config.top_k_retrieval > 0
    assert 0.0 <= config.confidence_threshold <= 1.0


def test_escalation_decision_enum():
    """Verify escalation decision contract has AUTO_HANDLE and ESCALATE."""
    assert EscalationDecision.AUTO_HANDLE.value == "AUTO_HANDLE"
    assert EscalationDecision.ESCALATE.value == "ESCALATE"
