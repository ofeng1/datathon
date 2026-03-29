"""
Train an ensemble (RandomForest + LightGBM + LogisticRegression → stacked meta-model)
on the cleaned NHAMCS ED dataframe and write a model artifact JSON.

Usage (from repo root, with PYTHONPATH=src):
    python -m modeling.train_ensemble

Or import and call `run()` directly.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cleaning import SELECTED_FEATURES, filter_nonnegative  # noqa: E402

TARGET_COL = "RETRNED"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

ARTIFACT_RELATIVE = Path("artifacts") / "models" / "readmission_model.json"
ENSEMBLE_RELATIVE = Path("artifacts") / "models" / "ensemble.joblib"
PREDICTIONS_RELATIVE = Path("artifacts") / "predictions" / "predictions.csv"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_serializable(obj):
    """json.dump default handler for numpy scalars."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _resolve_path(relative: Path) -> Path:
    return (REPO_ROOT / relative).resolve()


# ---------------------------------------------------------------------------
# data prep
# ---------------------------------------------------------------------------

def load_and_prepare(merged_pkl: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the merged pickle, apply cleaning filters, and split into X, y.

    Uses ``filter_nonnegative`` (the same row filter that produces
    ``cleaned_ed_dataframe.pkl``) so training sees identical rows.
    """
    raw = pd.read_pickle(merged_pkl)
    df = filter_nonnegative(raw)

    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET_COL]

    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].astype(int).reset_index(drop=True)

    X = X.dropna(axis=0)
    y = y.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)

    return X, y


# ---------------------------------------------------------------------------
# model creation
# ---------------------------------------------------------------------------

def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def build_lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.6,
        num_leaves=31,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[RandomForestClassifier, object, LogisticRegression, StandardScaler]:
    """Fit the three base classifiers on the training set."""
    rf = build_random_forest()
    lgbm = build_lgbm()
    logreg = build_logistic_regression()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    steps = [
        ("RandomForest", rf, X_train),
        ("LightGBM", lgbm, X_train),
        ("LogisticRegression", logreg, X_train_scaled),
    ]
    pbar = tqdm(steps, desc="Training base models", file=sys.stderr, dynamic_ncols=True)
    for name, model, X_data in pbar:
        pbar.set_postfix_str(name)
        model.fit(X_data, y_train)

    return rf, lgbm, logreg, scaler


def train_meta_model(
    X: pd.DataFrame,
    y: pd.Series,
    rf: RandomForestClassifier,
    lgbm,
    logreg: LogisticRegression,
) -> LogisticRegression:
    """5-fold stacking: generate OOF predictions → fit meta LogisticRegression."""
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_rf = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    oof_logreg = np.zeros(len(X))

    folds = list(kf.split(X, y))
    pbar = tqdm(folds, desc="Stacking folds", file=sys.stderr, dynamic_ncols=True)
    for fold, (train_idx, val_idx) in enumerate(pbar):
        pbar.set_postfix_str(f"fold {fold + 1}/{N_SPLITS}")
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]

        rf_clone = clone(rf)
        lgb_clone = clone(lgbm)
        logreg_clone = clone(logreg)

        rf_clone.fit(X_fold_train, y_fold_train)
        lgb_clone.fit(X_fold_train, y_fold_train)
        logreg_clone.fit(X_fold_train, y_fold_train)

        oof_rf[val_idx] = rf_clone.predict_proba(X_fold_val)[:, 1]
        oof_lgb[val_idx] = lgb_clone.predict_proba(X_fold_val)[:, 1]
        oof_logreg[val_idx] = logreg_clone.predict_proba(X_fold_val)[:, 1]

    meta_X = np.column_stack([oof_logreg, oof_rf, oof_lgb])
    meta_model = LogisticRegression()
    meta_model.fit(meta_X, y)

    # Refit base models on full X for production use
    rf.fit(X, y)
    lgbm.fit(X, y)
    logreg.fit(X, y)

    return meta_model


# ---------------------------------------------------------------------------
# artifact generation
# ---------------------------------------------------------------------------

def generate_artifact(
    rf: RandomForestClassifier,
    lgbm,
    logreg: LogisticRegression,
    meta_model: LogisticRegression,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Write the model metadata JSON to *output_path*."""
    artifact = {
        "model_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "task": "classification",
        "target": TARGET_COL,
        "feature_names": feature_names,
        "preprocessing": {
            "scaler": "StandardScaler",
            "used_for": "logistic_regression_only",
        },
        "base_models": [
            {"name": "RandomForestClassifier", "params": rf.get_params()},
            {"name": "LGBMClassifier", "params": lgbm.get_params()},
            {"name": "LogisticRegression", "params": logreg.get_params()},
        ],
        "meta_model": {
            "name": "LogisticRegression",
            "params": meta_model.get_params(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2, default=_to_serializable)

    print(f"Artifact saved → {output_path}")


def save_ensemble(
    rf: RandomForestClassifier,
    lgbm,
    logreg: LogisticRegression,
    scaler: StandardScaler,
    meta_model: LogisticRegression,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Serialize the full fitted ensemble to a single joblib file."""
    bundle = {
        "rf": rf,
        "lgbm": lgbm,
        "logreg": logreg,
        "scaler": scaler,
        "meta_model": meta_model,
        "feature_names": feature_names,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    print(f"Ensemble saved → {output_path}")


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def save_predictions(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    rf,
    lgbm,
    logreg,
    meta_model,
    output_path: Path,
) -> None:
    """Generate stacked ensemble predictions on the test set and save to CSV."""
    p_rf = rf.predict_proba(X_test)[:, 1]
    p_lgb = lgbm.predict_proba(X_test)[:, 1]
    p_lr = logreg.predict_proba(X_test)[:, 1]

    meta_X = np.column_stack([p_lr, p_rf, p_lgb])
    y_proba = meta_model.predict_proba(meta_X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    out = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "y_proba": y_proba,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Predictions saved → {output_path}")


def run(
    merged_pkl: Path | None = None,
    artifact_path: Path | None = None,
    ensemble_path: Path | None = None,
    predictions_path: Path | None = None,
) -> dict:
    """
    Full pipeline: load merged data → clean → train/test split → fit base
    models → ensemble via stacking → write artifact JSON + ensemble joblib
    → save predictions CSV.

    Returns a dict with the fitted models + split data for optional evaluation.
    """
    if merged_pkl is None:
        merged_pkl = _resolve_path(Path("data") / "processed" / "merged_ed_dataframe.pkl")
    if artifact_path is None:
        artifact_path = _resolve_path(ARTIFACT_RELATIVE)
    if ensemble_path is None:
        ensemble_path = _resolve_path(ENSEMBLE_RELATIVE)
    if predictions_path is None:
        predictions_path = _resolve_path(PREDICTIONS_RELATIVE)

    X, y = load_and_prepare(merged_pkl)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    rf, lgbm, logreg, scaler = train_base_models(X_train, y_train)
    meta_model = train_meta_model(X, y, rf, lgbm, logreg)

    generate_artifact(rf, lgbm, logreg, meta_model, list(X.columns), artifact_path)
    save_ensemble(rf, lgbm, logreg, scaler, meta_model, list(X.columns), ensemble_path)
    save_predictions(X_test, y_test, rf, lgbm, logreg, meta_model, predictions_path)

    return {
        "rf": rf,
        "lgbm": lgbm,
        "logreg": logreg,
        "scaler": scaler,
        "meta_model": meta_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()
