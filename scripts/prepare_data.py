"""Data preparation script.

Usage:
    python scripts/prepare_data.py --input <path_to_raw_dataset> --brand <target_brand>

Processes raw Twitter support dataset into cleaned, structured brand dialogue pairs.
Dataset schema and target brand will be finalized following dataset inspection.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and clean customer support data.")
    parser.add_argument("--input", type=str, default=None, help="Path to raw dataset")
    parser.add_argument("--output", type=str, default="data/processed/cleaned_conversations.parquet", help="Output path")
    parser.add_argument("--brand", type=str, default=None, help="Target brand name to filter and process")
    args = parser.parse_args()

    print("Data preparation script placeholder.")
    print(f"Target Brand: {args.brand}")
    print(f"Raw Input: {args.input}")


if __name__ == "__main__":
    main()
