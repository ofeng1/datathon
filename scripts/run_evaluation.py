#!/usr/bin/env python3
"""
Evaluate the model using the predictions CSV produced by run_training.py.

Prerequisites:
    The predictions CSV must already exist.  If it doesn't, run:
        python scripts/run_training.py

Usage (from repo root):
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --predictions data/processed/predictions.csv \
                                     --save-cm-plot artifacts/plots/confusion_matrix.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modeling.evaluation import evaluate, save_confusion_matrix_plot  # noqa: E402

import pandas as pd  # noqa: E402

DEFAULT_PREDICTIONS = "artifacts/predictions/predictions.csv"
DEFAULT_CM_PLOT = "artifacts/plots/confusion_matrix.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions and print metrics.",
    )
    parser.add_argument(
        "--predictions",
        default=DEFAULT_PREDICTIONS,
        help=f"Path to predictions CSV (default: {DEFAULT_PREDICTIONS}).",
    )
    parser.add_argument(
        "--y-true-col",
        default="y_true",
        help="Column name for ground-truth labels (default: y_true).",
    )
    parser.add_argument(
        "--y-pred-col",
        default="y_pred",
        help="Column name for predicted labels (default: y_pred).",
    )
    parser.add_argument(
        "--save-cm-plot",
        default=DEFAULT_CM_PLOT,
        help=f"Path to save confusion matrix PNG (default: {DEFAULT_CM_PLOT}).",
    )
    args = parser.parse_args()

    csv_path = (REPO_ROOT / args.predictions).resolve()
    if not csv_path.exists():
        print(
            f"Error: predictions CSV not found at {csv_path}\n"
            "Run  python scripts/run_training.py  first.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)
    y_true = df[args.y_true_col].astype(int)
    y_pred = df[args.y_pred_col].astype(int)

    results = evaluate(y_true=y_true, y_pred=y_pred)
    cm = results["confusion_matrix"]

    print("\n=== Evaluation Results ===")
    print(f"\nConfusion Matrix (rows=true, cols=pred):\n{cm}")
    print(f"\nMSE:      {results['mse']:.6f}")
    print(f"Accuracy: {results['accuracy']:.6f}")

    if args.save_cm_plot:
        plot_path = (REPO_ROOT / args.save_cm_plot).resolve()
        save_confusion_matrix_plot(cm, plot_path)
        print(f"\nConfusion matrix plot saved to: {plot_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
