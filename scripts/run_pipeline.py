"""CLI runner for the AI Support Agent pipeline.

Usage:
    python scripts/run_pipeline.py --message "Where is my order #12345? It was supposed to arrive yesterday."

Processes customer message through intent classification, retrieval,
grounded generation, and escalation evaluation.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI Support Agent on a customer query.")
    parser.add_argument("--message", type=str, required=True, help="Customer support inquiry text")
    args = parser.parse_args()

    print("Support Agent Pipeline runner placeholder.")
    print(f"Received query: '{args.message}'")


if __name__ == "__main__":
    main()
