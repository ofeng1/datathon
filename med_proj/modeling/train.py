import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

from med_proj.features.schema import CATEGORICAL, NUMERIC
from med_proj.modeling.calibrate import calibrate
from med_proj.common.logging import get_logger

log = get_logger("train")

def train_one(df: pd.DataFrame, label: str, seed: int, test_size: float, cal_method: str):
    X = df[CATEGORICAL + NUMERIC].copy()
    y = df[label].astype(int).values

    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )

    base = HistGradientBoostingClassifier(
        random_state=seed,
        max_depth=6,
        learning_rate=0.06,
        max_iter=300
    )

    pipe = Pipeline([("pre", pre), ("model", base)])
    pipe.fit(X_train, y_train)

    cal = calibrate(pipe, X_train, y_train, method=cal_method)
    return cal, X_test, y_test

def save(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    log.info("Saved model -> %s", path)