"""Golden evaluation set builder script.

Usage:
    python scripts/build_golden_set.py --size 200

Samples representative customer queries across identified intents and prepares
the annotation schema for human labelling (150-250 examples).
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample and build golden evaluation set template.")
    parser.add_argument("--size", type=int, default=200, help="Target sample size (between 150 and 250)")
    parser.add_argument("--output", type=str, default="data/golden/golden_evaluation_set.jsonl", help="Output path")
    args = parser.parse_args()

    print("Golden evaluation set builder script placeholder.")
    print(f"Target size: {args.size} samples")
    print(f"Output path: {args.output}")


if __name__ == "__main__":
    main()
