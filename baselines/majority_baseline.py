"""Trivial Baseline: Majority Class Predictor.

Always predicts the most frequent intent label and fixed default action/canned response.
Serves as the bottom-line benchmark to verify whether learned or retrieval models
provide genuine statistical and practical lift.
"""

from typing import Any, List, Optional
import pandas as pd


class MajorityClassBaseline:
    """Predicts majority class intent and fixed canned response."""

    def __init__(self) -> None:
        self.majority_intent: Optional[str] = None
        self.canned_response: Optional[str] = None

    def fit(self, training_data: pd.DataFrame) -> None:
        """Fit baseline on training dataset."""
        raise NotImplementedError("Majority baseline fitting will be implemented in subsequent phases.")

    def predict(self, messages: List[str]) -> List[dict[str, Any]]:
        """Predict majority intent and response for input messages."""
        raise NotImplementedError("Majority baseline prediction will be implemented in subsequent phases.")
