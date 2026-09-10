"""Configuration management for Hiver AI Support Agent.

Loads environment variables, default paths, and pipeline parameters.
No provider, model, dataset filename, brand, or secrets are hardcoded.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GOLDEN_DATA_DIR = DATA_DIR / "golden"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Load .env if present
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class AppConfig:
    """Application and pipeline configuration container."""

    # Provider & Model Settings (to be selected in subsequent phases)
    llm_provider: str | None = os.getenv("LLM_PROVIDER")
    llm_model: str | None = os.getenv("LLM_MODEL")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")

    embedding_model: str | None = os.getenv("EMBEDDING_MODEL")

    # Data Settings (paths to be finalized after raw dataset inspection)
    raw_data_path: Path | None = (
        Path(os.getenv("RAW_DATA_PATH")) if os.getenv("RAW_DATA_PATH") else None
    )
    processed_data_path: Path | None = (
        Path(os.getenv("PROCESSED_DATA_PATH")) if os.getenv("PROCESSED_DATA_PATH") else None
    )
    golden_data_path: Path = Path(
        os.getenv("GOLDEN_DATA_PATH", str(GOLDEN_DATA_DIR / "golden_evaluation_set.jsonl"))
    )

    # Agent / Pipeline Parameters (target brand to be chosen based on dataset exploration)
    target_brand: str | None = os.getenv("TARGET_BRAND")
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "3"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))


# Default global configuration instance
config = AppConfig()
