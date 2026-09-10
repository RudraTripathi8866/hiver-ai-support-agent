"""Dataset discovery, inspection, and EDA runner script.

Usage:
    py scripts/prepare_data.py [--input path/to/raw/file.csv] [--max-chunks 10]

Inspects actual dataset files, schema, missing values, candidate brands,
and conversation structures without assumptions or hardcoded schemas.
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path when running as script
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from src.config import PROJECT_ROOT, RAW_DATA_DIR, OUTPUTS_DIR
from src.preprocessing import discover_raw_files, analyze_dataset_statistics


def print_banner(title: str) -> None:
    """Print a clean section banner."""
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def run_eda(input_path: Path | None = None, max_chunks: int | None = None) -> None:
    """Execute dataset discovery and exploratory analysis."""
    print_banner("1. Dataset File Discovery")

    if input_path is not None:
        target_file = input_path
        if not target_file.exists():
            print(f"[ERROR] Specified input file does not exist: {target_file}")
            sys.exit(1)
        raw_files = [target_file]
    else:
        discovered = discover_raw_files(RAW_DATA_DIR)
        if not discovered:
            print(f"[WARNING] No raw dataset files found in: {RAW_DATA_DIR}")
            print("Please place the Customer Support dataset (e.g. twcs.csv) inside 'data/raw/'.")
            print("Directory contents:")
            for item in RAW_DATA_DIR.iterdir():
                print(f"  - {item.name}")
            return
        target_file = Path(discovered[0].filepath)
        raw_files = [Path(d.filepath) for d in discovered]

    print(f"Discovered {len(raw_files)} raw dataset file(s):")
    for f in raw_files:
        size_mb = round(f.stat().st_size / (1024 * 1024), 2)
        print(f"  - File: {f.name} | Path: {f} | Size: {size_mb} MB ({f.stat().st_size:,} bytes)")

    print_banner(f"2. Inspecting Schema & Streaming Statistics: {target_file.name}")
    print("Analyzing dataset in chunks to respect memory limits...")

    results = analyze_dataset_statistics(target_file, chunksize=100_000, max_chunks=max_chunks)

    # 1. File info
    info = results["file_info"]
    print(f"\nFile: {info['filename']}")
    print(f"Total Size: {info['size_mb']} MB ({info['size_bytes']:,} bytes)")
    print(f"Total Rows Analyzed: {results['total_rows']:,}")

    # 2. Columns and Data Types
    print_banner("3. Columns, Types & Null Value Distribution")
    print(f"{'Column Name':<28} {'Data Type':<12} {'Null Count':<14} {'Null %':<10}")
    print("-" * 68)
    for col in results["column_summaries"]:
        print(f"{col['name']:<28} {col['dtype']:<12} {col['null_count']:<14,} {col['null_percentage']:<10.2f}%")

    # 3. Key Identified Fields
    print_banner("4. Key Semantic Fields Identified")
    for k, v in results["key_fields_identified"].items():
        print(f"  - {k:<25}: {v if v else '[Not Detected / Needs Manual Inspection]'}")

    # 4. Top Candidate Brands
    print_banner("5. Top Candidate Brands (By Outbound Agent Message Volume)")
    brands = results["top_candidate_brands"]
    if brands:
        print(f"{'Rank':<6} {'Brand Name':<24} {'Agent Responses':<18}")
        print("-" * 50)
        for i, b in enumerate(brands[:15], 1):
            print(f"{i:<6} {b['brand_name']:<24} {b['agent_messages']:<18,}")
    else:
        print("No brand/agent records identified based on automated column detection.")

    # 5. Raw Record Samples
    print_banner("6. Raw Record Samples (First 2 Records)")
    for i, sample in enumerate(results["sample_records"][:2], 1):
        print(f"\n--- Record #{i} ---")
        for k, v in sample.items():
            print(f"  {k}: {v}")

    # Save machine-readable outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUTS_DIR / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Complete machine-readable statistics saved to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and inspect Customer Support dataset.")
    parser.add_argument("--input", type=str, default=None, help="Explicit path to raw dataset file")
    parser.add_argument("--max-chunks", type=int, default=None, help="Max chunks to process (for quick testing)")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else None
    run_eda(input_path=input_path, max_chunks=args.max_chunks)


if __name__ == "__main__":
    main()
