"""Data preparation script.

Usage:
    python scripts/prepare_data.py --input data/raw/customer_support.csv --brand AmazonHelp

Processes raw Twitter support dataset into cleaned, structured brand dialogue pairs.
Dataset schema will be inspected prior to finalizing transformation logic.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and clean customer support data.")
    parser.add_argument("--input", type=str, default="data/raw/customer_support.csv", help="Path to raw dataset")
    parser.add_argument("--output", type=str, default="data/processed/cleaned_conversations.parquet", help="Output path")
    parser.add_argument("--brand", type=str, default="AmazonHelp", help="Target brand name")
    args = parser.parse_args()

    print("Data preparation script placeholder.")
    print(f"Target Brand: {args.brand}")
    print(f"Raw Input: {args.input}")


if __name__ == "__main__":
    main()
