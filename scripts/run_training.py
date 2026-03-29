#!/usr/bin/env python3
"""
Train the stacked ensemble model on the merged NHAMCS ED dataframe.

Prerequisites:
    The merged pickle must already exist.  If it doesn't, run:
        python scripts/build_merged_ed_dataframe.py

Usage (from repo root):
    python scripts/run_training.py
    python scripts/run_training.py --input data/processed/merged_ed_dataframe.pkl \
                                   --artifact artifacts/models/readmission_model.json \
                                   --predictions data/processed/predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modeling.training import run  # noqa: E402

DEFAULT_INPUT = "data/processed/merged_ed_dataframe.pkl"
DEFAULT_ARTIFACT = "artifacts/models/readmission_model.json"
DEFAULT_PREDICTIONS = "artifacts/predictions/predictions.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the readmission ensemble model on ED data.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to merged pickle (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--artifact",
        default=DEFAULT_ARTIFACT,
        help=f"Output path for model artifact JSON (default: {DEFAULT_ARTIFACT}).",
    )
    parser.add_argument(
        "--predictions",
        default=DEFAULT_PREDICTIONS,
        help=f"Output path for test-set predictions CSV (default: {DEFAULT_PREDICTIONS}).",
    )
    args = parser.parse_args()

    merged_pkl = (REPO_ROOT / args.input).resolve()
    artifact_path = (REPO_ROOT / args.artifact).resolve()
    predictions_path = (REPO_ROOT / args.predictions).resolve()

    if not merged_pkl.exists():
        print(
            f"Error: merged data not found at {merged_pkl}\n"
            "Run  python scripts/build_merged_ed_dataframe.py  first.",
            file=sys.stderr,
        )
        sys.exit(1)

    run(
        merged_pkl=merged_pkl,
        artifact_path=artifact_path,
        predictions_path=predictions_path,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
