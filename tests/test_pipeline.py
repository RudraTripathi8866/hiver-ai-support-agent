"""Sanity, contract, and EDA pipeline tests compatible with unittest and pytest."""

from pathlib import Path
import tempfile
import unittest
import pandas as pd

from src.config import config
from src.escalation import EscalationDecision
from src.preprocessing import (
    discover_raw_files,
    inspect_raw_file_header,
    analyze_dataset_statistics,
)


class TestPipelineAndEDA(unittest.TestCase):
    """Test suite for pipeline configuration and dataset inspection."""

    def test_config_defaults(self):
        """Verify default configuration values load correctly."""
        self.assertGreater(config.top_k_retrieval, 0)
        self.assertTrue(0.0 <= config.confidence_threshold <= 1.0)

    def test_escalation_decision_enum(self):
        """Verify escalation decision contract has AUTO_HANDLE and ESCALATE."""
        self.assertEqual(EscalationDecision.AUTO_HANDLE.value, "AUTO_HANDLE")
        self.assertEqual(EscalationDecision.ESCALATE.value, "ESCALATE")

    def test_discover_raw_files_empty(self):
        """Verify raw file discovery handles directories without data files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / ".gitkeep").write_text("# gitkeep")
            files = discover_raw_files(tmp_path)
            self.assertEqual(len(files), 0)

    def test_eda_on_sample_csv(self):
        """Verify header inspection and streaming EDA work on sample tabular data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_csv = Path(tmp_dir) / "test_support.csv"
            df = pd.DataFrame(
                {
                    "tweet_id": [1, 2, 3, 4],
                    "author_id": ["CustA", "BrandX", "CustB", "BrandX"],
                    "inbound": [True, False, True, False],
                    "created_at": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
                    "text": ["Help please", "Sure, what is wrong?", "My item is late", "Let me check"],
                    "response_tweet_id": ["2", None, "4", None],
                    "in_response_to_tweet_id": [None, 1.0, None, 3.0],
                }
            )
            df.to_csv(sample_csv, index=False)

            cols, dtypes, sample_df = inspect_raw_file_header(sample_csv, nrows=2)
            self.assertIn("tweet_id", cols)
            self.assertIn("author_id", cols)
            self.assertEqual(len(sample_df), 2)

            eda_res = analyze_dataset_statistics(sample_csv, chunksize=2)
            self.assertEqual(eda_res["total_rows"], 4)
            self.assertGreater(len(eda_res["top_candidate_brands"]), 0)
            self.assertEqual(eda_res["top_candidate_brands"][0]["brand_name"], "BrandX")
            self.assertEqual(eda_res["key_fields_identified"]["inbound_col"], "inbound")


if __name__ == "__main__":
    unittest.main()
