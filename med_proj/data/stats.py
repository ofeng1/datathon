"""Build aggregate stats from NHAMCS raw data for the Stats tab and chatbot."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

REGION_NAMES = {
    1: "Northeast",
    2: "Midwest",
    3: "South",
    4: "West",
}

CONDITION_COLUMNS = [
    "COPD", "CHF", "CAD", "ASTHMA", "CKD", "ESRD", "HTN",
    "DIABTYP1", "DIABTYP2", "DIABTYP0", "CANCER", "CEBVD",
    "DEPRN", "ALZHD", "HYPLIPID", "OBESITY", "OSA", "SUBSTAB",
    "EDHIV", "ETOHAB", "OSTPRSIS", "INJURY",
]

CONDITION_DISPLAY_NAMES = {
    "COPD": "COPD",
    "CHF": "Heart Failure (CHF)",
    "CAD": "Coronary Artery Disease",
    "ASTHMA": "Asthma",
    "CKD": "Chronic Kidney Disease",
    "ESRD": "End-Stage Renal Disease",
    "HTN": "Hypertension",
    "DIABTYP1": "Diabetes Type 1",
    "DIABTYP2": "Diabetes Type 2",
    "DIABTYP0": "Diabetes",
    "CANCER": "Cancer",
    "CEBVD": "Stroke / Cerebrovascular",
    "DEPRN": "Depression",
    "ALZHD": "Alzheimer's / Dementia",
    "HYPLIPID": "Hyperlipidemia",
    "OBESITY": "Obesity",
    "OSA": "Sleep Apnea",
    "SUBSTAB": "Substance Use Disorder",
    "EDHIV": "HIV",
    "ETOHAB": "Alcohol Use Disorder",
    "OSTPRSIS": "Osteoporosis",
    "INJURY": "Injury / Trauma",
}


def _rate(pos: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(100.0 * pos / total, 2)


def _safe_float(val: Any) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def build_stats_from_raw(raw: pd.DataFrame) -> Dict[str, Any]:
    """Compute region and condition stats from NHAMCS raw dataframe."""
    out: Dict[str, Any] = {"regions": [], "conditions": [], "national": {}}

    seen72_pos = {1, 1.0}
    admit_pos = {1, 1.0}

    def is_seen72(x) -> bool:
        v = _safe_float(x)
        return v in seen72_pos if not np.isnan(v) else False

    def is_admit(x) -> bool:
        v = _safe_float(x)
        return v in admit_pos if not np.isnan(v) else False

    # National totals
    n = len(raw)
    seen72_col = raw["SEEN72"] if "SEEN72" in raw.columns else pd.Series([0] * n)
    admit_col = raw["ADMITHOS"] if "ADMITHOS" in raw.columns else pd.Series([0] * n)
    out["national"] = {
        "n_visits": int(n),
        "pct_72h_revisit": _rate(seen72_col.apply(is_seen72).sum(), n),
        "pct_admitted": _rate(admit_col.apply(is_admit).sum(), n),
    }

    # By region (REGION 1-4)
    if "REGION" in raw.columns:
        reg = raw["REGION"].apply(_safe_float)
        for rid, rname in REGION_NAMES.items():
            mask = (reg >= rid - 0.5) & (reg < rid + 0.5)
            if not mask.any():
                continue
            sub = raw.loc[mask]
            n_r = int(mask.sum())
            s72 = seen72_col.loc[mask].apply(is_seen72).sum()
            adm = admit_col.loc[mask].apply(is_admit).sum()
            out["regions"].append({
                "id": rid,
                "name": rname,
                "n_visits": n_r,
                "pct_72h_revisit": _rate(s72, n_r),
                "pct_admitted": _rate(adm, n_r),
            })
    else:
        # No REGION column: single "All U.S." row
        out["regions"].append({
            "id": 0,
            "name": "All U.S.",
            "n_visits": out["national"]["n_visits"],
            "pct_72h_revisit": out["national"]["pct_72h_revisit"],
            "pct_admitted": out["national"]["pct_admitted"],
        })

    # By condition (among rows where condition indicator = 1)
    for col in CONDITION_COLUMNS:
        if col not in raw.columns:
            continue
        cond_val = raw[col].apply(_safe_float)
        mask = (cond_val >= 0.99) & (cond_val <= 1.01)
        if not mask.any():
            continue
        sub = raw.loc[mask]
        n_c = int(mask.sum())
        s72 = seen72_col.loc[mask].apply(is_seen72).sum()
        adm = admit_col.loc[mask].apply(is_admit).sum()
        out["conditions"].append({
            "id": col,
            "name": CONDITION_DISPLAY_NAMES.get(col, col),
            "n_visits": n_c,
            "pct_72h_revisit": _rate(s72, n_c),
            "pct_admitted": _rate(adm, n_c),
        })

    # Sort conditions by 72h revisit rate descending
    out["conditions"].sort(key=lambda x: -x["pct_72h_revisit"])

    return out
