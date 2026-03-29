#!/usr/bin/env python3
"""
Load the merged ED dataframe, run the cleaning pipeline, and save the
cleaned result to data/processed/.

Usage (from repo root):
    python scripts/run_cleaning.py
    python scripts/run_cleaning.py --input data/processed/merged_ed_dataframe.pkl \
                                   --output data/processed/cleaned_ed_dataframe.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd  # noqa: E402
from data.cleaning import clean  # noqa: E402


DEFAULT_INPUT = "data/processed/merged_ed_dataframe.pkl"
DEFAULT_OUTPUT = "data/processed/cleaned_ed_dataframe.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the merged NHAMCS ED dataframe.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to merged pickle (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output path for cleaned pickle (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    in_path = (REPO_ROOT / args.input).resolve()
    out_path = (REPO_ROOT / args.output).resolve()

    if not in_path.exists():
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {in_path} ...")
    df = pd.read_pickle(in_path)

    cleaned = clean(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_pickle(out_path)
    print(f"Saved cleaned dataframe to {out_path}")


if __name__ == "__main__":
    main()
