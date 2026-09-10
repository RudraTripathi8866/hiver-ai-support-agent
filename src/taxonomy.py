"""Intent taxonomy definition and validation for customer support.

Defines the brand-specific intent classes, descriptions, and escalation policy tags.
The concrete taxonomy classes will be derived from exploratory data analysis of the dataset.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class IntentDefinition:
    """Representation of an intent category in the taxonomy."""

    name: str
    description: str
    examples: List[str]
    default_action: str  # AUTO_HANDLE or ESCALATE
    escalation_criteria: Optional[str] = None


# Placeholder list of intents. Finalized taxonomy will be derived empirically from data.
BRAND_INTENTS: List[IntentDefinition] = []


def get_available_intents() -> List[str]:
    """Return list of valid intent labels."""
    return [intent.name for intent in BRAND_INTENTS]
