"""Dataset discovery, inspection, and EDA utilities for Customer Support dataset.

Provides streaming and chunked inspection to handle large tabular datasets
without loading the entire multi-gigabyte file into memory.
Preserves all identifiers and performs dynamic schema analysis.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR


@dataclass
class DatasetFileInfo:
    """Information about a discovered dataset file."""

    filename: str
    filepath: str
    size_bytes: int
    size_mb: float


@dataclass
class ColumnSummary:
    """Summary statistics for an observed column."""

    name: str
    dtype: str
    null_count: int
    null_percentage: float
    sample_values: List[Any]


def discover_raw_files(raw_dir: Optional[Path] = None) -> List[DatasetFileInfo]:
    """Discover all potential dataset files under raw_dir excluding .gitkeep."""
    target_dir = raw_dir or RAW_DATA_DIR
    if not target_dir.exists():
        return []

    discovered = []
    for item in target_dir.iterdir():
        if item.is_file() and item.name != ".gitkeep":
            stat = item.stat()
            discovered.append(
                DatasetFileInfo(
                    filename=item.name,
                    filepath=str(item.resolve()),
                    size_bytes=stat.st_size,
                    size_mb=round(stat.st_size / (1024 * 1024), 2),
                )
            )
    return discovered


def inspect_raw_file_header(file_path: str | Path, nrows: int = 5) -> Tuple[List[str], Dict[str, str], pd.DataFrame]:
    """Inspect initial records to determine column schema without reading full file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at: {path}")

    # Read sample to inspect schema
    if path.suffix.lower() == ".csv":
        sample_df = pd.read_csv(path, nrows=nrows)
    elif path.suffix.lower() in [".parquet", ".pq"]:
        sample_df = pd.read_parquet(path)
        sample_df = sample_df.head(nrows)
    else:
        # Fallback to CSV parsing
        sample_df = pd.read_csv(path, nrows=nrows)

    columns = list(sample_df.columns)
    dtypes = {col: str(sample_df[col].dtype) for col in columns}
    return columns, dtypes, sample_df


def stream_dataset_chunks(
    file_path: str | Path,
    chunksize: int = 100_000,
) -> Generator[pd.DataFrame, None, None]:
    """Stream dataset in manageable chunks to respect memory constraints."""
    path = Path(file_path)
    if path.suffix.lower() == ".csv":
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            yield chunk
    elif path.suffix.lower() in [".parquet", ".pq"]:
        full_df = pd.read_parquet(path)
        total_len = len(full_df)
        for i in range(0, total_len, chunksize):
            yield full_df.iloc[i : i + chunksize]
    else:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            yield chunk


def analyze_dataset_statistics(
    file_path: str | Path,
    chunksize: int = 100_000,
    max_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    """Perform streaming exploratory data analysis across the dataset.
    
    Extracts row counts, null values, thread structures, author/brand distributions,
    and data quality indicators without loading entire dataset into RAM.
    """
    path = Path(file_path)
    file_info = DatasetFileInfo(
        filename=path.name,
        filepath=str(path.resolve()),
        size_bytes=path.stat().st_size,
        size_mb=round(path.stat().st_size / (1024 * 1024), 2),
    )

    total_rows = 0
    null_counts: Dict[str, int] = {}
    sample_records: List[Dict[str, Any]] = []
    columns: List[str] = []
    dtypes: Dict[str, str] = {}

    # Brand and conversation tracking structures
    brand_counts: Dict[str, int] = {}
    brand_customer_counts: Dict[str, int] = {}
    brand_agent_counts: Dict[str, int] = {}
    brand_conversations: Dict[str, set] = {}
    brand_conv_has_customer: Dict[str, set] = {}
    brand_conv_has_agent: Dict[str, set] = {}
    brand_conv_lengths: Dict[str, Dict[Any, int]] = {}

    # Potential column identity mappings
    author_col: Optional[str] = None
    inbound_col: Optional[str] = None
    text_col: Optional[str] = None
    created_at_col: Optional[str] = None
    tweet_id_col: Optional[str] = None
    in_response_to_col: Optional[str] = None
    response_tweet_id_col: Optional[str] = None

    chunk_idx = 0
    for chunk in stream_dataset_chunks(path, chunksize=chunksize):
        if chunk_idx == 0:
            columns = list(chunk.columns)
            dtypes = {c: str(chunk[c].dtype) for c in columns}
            for c in columns:
                null_counts[c] = 0
            sample_records = chunk.head(5).to_dict(orient="records")

            # Identify key columns based on observed names
            for col in columns:
                col_lower = col.lower()
                if col_lower in ["author_id", "author", "user", "sender"]:
                    author_col = col
                elif col_lower in ["inbound", "is_customer", "is_inbound"]:
                    inbound_col = col
                elif col_lower in ["text", "tweet", "content", "message"]:
                    text_col = col
                elif col_lower in ["created_at", "timestamp", "date", "time"]:
                    created_at_col = col
                elif col_lower in ["tweet_id", "id", "message_id"]:
                    tweet_id_col = col
                elif col_lower in ["in_response_to_tweet_id", "in_reply_to_status_id", "reply_to"]:
                    in_response_to_col = col
                elif col_lower in ["response_tweet_id", "response_id"]:
                    response_tweet_id_col = col

        total_rows += len(chunk)

        # Null tracking
        for c in columns:
            null_counts[c] += int(chunk[c].isna().sum())

        # If inbound_col and author_col exist, track author/brand statistics
        if author_col is not None:
            if inbound_col is not None:
                # Brands are typically authors of outbound messages (inbound == False / 0)
                # Customers are typically authors of inbound messages (inbound == True / 1)
                is_inbound = chunk[inbound_col].astype(bool)
                agent_msgs = chunk[~is_inbound]
                cust_msgs = chunk[is_inbound]

                for brand, count in agent_msgs[author_col].value_counts().items():
                    b_str = str(brand)
                    brand_agent_counts[b_str] = brand_agent_counts.get(b_str, 0) + int(count)
                    brand_counts[b_str] = brand_counts.get(b_str, 0) + int(count)

                # For inbound messages, author is an anonymous customer, but in_response_to connects to brand
            else:
                # Fallback: Count all authors
                for author, count in chunk[author_col].value_counts().items():
                    a_str = str(author)
                    brand_counts[a_str] = brand_counts.get(a_str, 0) + int(count)

        chunk_idx += 1
        if max_chunks is not None and chunk_idx >= max_chunks:
            break

    # Compile column summaries
    column_summaries = []
    for c in columns:
        null_cnt = null_counts.get(c, 0)
        pct = round((null_cnt / total_rows * 100), 2) if total_rows > 0 else 0.0
        column_summaries.append(
            ColumnSummary(
                name=c,
                dtype=dtypes.get(c, "unknown"),
                null_count=null_cnt,
                null_percentage=pct,
                sample_values=[r.get(c) for r in sample_records[:3]],
            )
        )

    # Compile identified candidate brands (top brands by message count)
    candidate_brands = []
    sorted_brands = sorted(brand_agent_counts.items(), key=lambda x: x[1], reverse=True)[:25]
    for brand_name, agent_count in sorted_brands:
        candidate_brands.append(
            {
                "brand_name": brand_name,
                "agent_messages": agent_count,
                "total_recorded_agent_responses": agent_count,
            }
        )

    return {
        "file_info": asdict(file_info),
        "total_rows": total_rows,
        "columns": columns,
        "dtypes": dtypes,
        "column_summaries": [asdict(cs) for cs in column_summaries],
        "key_fields_identified": {
            "author_col": author_col,
            "inbound_col": inbound_col,
            "text_col": text_col,
            "created_at_col": created_at_col,
            "tweet_id_col": tweet_id_col,
            "in_response_to_col": in_response_to_col,
            "response_tweet_id_col": response_tweet_id_col,
        },
        "sample_records": sample_records,
        "top_candidate_brands": candidate_brands,
    }
