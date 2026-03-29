"""
Evaluate binary classification predictions from a CSV file.

Usage (from repo root, with PYTHONPATH=src):
    python -m modeling.evaluation --predictions-csv data/processed/predictions.csv

Expected columns:
    - y_true (default)
    - y_pred (default) OR y_proba (if --y-proba-col is provided)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _load_labels(
    csv_path: Path,
    y_true_col: str,
    y_pred_col: str,
    y_proba_col: str | None,
    threshold: float,
) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)

    if y_true_col not in df.columns:
        raise ValueError(f"Missing y_true column: {y_true_col}")

    y_true = df[y_true_col].astype(int)

    if y_proba_col:
        if y_proba_col not in df.columns:
            raise ValueError(f"Missing probability column: {y_proba_col}")
        y_pred = (df[y_proba_col].astype(float) >= threshold).astype(int)
    else:
        if y_pred_col not in df.columns:
            raise ValueError(f"Missing y_pred column: {y_pred_col}")
        y_pred = df[y_pred_col].astype(int)

    return y_true, y_pred


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    mse = mean_squared_error(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return {"confusion_matrix": cm, "mse": mse, "accuracy": acc}


def save_confusion_matrix_plot(cm, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate binary predictions from CSV.")
    parser.add_argument(
        "--predictions-csv",
        required=True,
        help="Path to CSV containing y_true and y_pred (or y_proba).",
    )
    parser.add_argument("--y-true-col", default="y_true", help="Ground-truth column name.")
    parser.add_argument("--y-pred-col", default="y_pred", help="Predicted-label column name.")
    parser.add_argument(
        "--y-proba-col",
        default=None,
        help="Optional predicted-probability column; if set, y_pred is computed using --threshold.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for converting y_proba to y_pred.",
    )
    parser.add_argument(
        "--save-cm-plot",
        default=None,
        help="Optional path to save confusion matrix PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = _resolve(args.predictions_csv)
    y_true, y_pred = _load_labels(
        csv_path=csv_path,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        y_proba_col=args.y_proba_col,
        threshold=args.threshold,
    )

    results = evaluate(y_true=y_true, y_pred=y_pred)
    cm = results["confusion_matrix"]

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"\nMSE: {results['mse']:.6f}")
    print(f"Accuracy: {results['accuracy']:.6f}")

    if args.save_cm_plot:
        out_path = _resolve(args.save_cm_plot)
        save_confusion_matrix_plot(cm, out_path)
        print(f"Confusion matrix plot saved to: {out_path}")


if __name__ == "__main__":
    main()
